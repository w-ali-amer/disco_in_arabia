# -*- coding: utf-8 -*-
"""
exp7.py
-------
Builds on exp6 with two additional fixes in the pipeline:
  1. arabic_dep_reader.py now appends CAMeL morphological tags (ASP/PER/NUM/GEN)
     to every Word box label — Past `قرات_ASP-p_PER-3_NUM-s_GEN-f` vs
     Present `تقرا_ASP-i_PER-2_NUM-d_GEN-m` → different circuit parameters,
     giving the optimizer a stronger morphological signal.
  2. Verb-rescue heuristic in arabic_dep_reader: when Stanza fails to assign
     a root verb, the first VERB-tagged word in the analyses is promoted.
  3. camel_test2 token-mismatch is now DEBUG (no more warning spam).
  4. warm-start _vec_for_word strips morph-tag suffix before AraVec lookup
     so enriched labels like `قرات_ASP-p_...` still hit AraVec via `قرات`.
  5. Same experiment sets and SPSA settings as exp6.

Experiment sets:
       - LexicalAmbiguity_6class  (3 polysemous pairs × 2 senses × 10 samples)
       - Morphology_Tense         (binary Past/Present — best result from exp5)
       - Morphology_Number        (Sg/Du/Pl — 3-class, needs warm-start to lift)
       - Morphology_Possession    (1st/2nd/3rd person — 3-class)
  4. Ansatzes: IQP (n_layers=2) and Sim14 (n_layers=2) only.
     StronglyEntangling dropped — performed worst in exp5 on every task.

Usage:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 exp6.py
"""

import os, sys, json, math, logging, time, hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import Counter

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("qnlp_experiment_exp7.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("exp7")

# ── lambeq ────────────────────────────────────────────────────────────────────
from lambeq.backend.grammar import Ty, Word, Diagram
from lambeq import (IQPAnsatz, Sim14Ansatz, RemoveCupsRewriter,
                    NumpyModel, QuantumTrainer, SPSAOptimizer, Dataset)
from lambeq.training import CrossEntropyLoss, BinaryCrossEntropyLoss

N = Ty('n')
S = Ty('s')

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# ── AraVec (gensim KeyedVectors) ──────────────────────────────────────────────
try:
    from gensim.models import KeyedVectors, Word2Vec
    ARAVEC_PATH = "aravec/full_uni_cbow_300_twitter.mdl"
    _kv = None
    if os.path.exists(ARAVEC_PATH):
        try:
            m = Word2Vec.load(ARAVEC_PATH)
            _kv = m.wv
            logger.info(f"AraVec loaded: {len(_kv)} vectors × {_kv.vector_size}d")
        except Exception as e:
            logger.warning(f"AraVec load failed: {e}")
    else:
        logger.warning(f"AraVec not found at {ARAVEC_PATH}. Using hash fallback.")
    ARAVEC_KV = _kv
except ImportError:
    ARAVEC_KV = None
    logger.warning("gensim not available. Using hash fallback for warm-start.")

# ── arabic_dep_reader ─────────────────────────────────────────────────────────
from arabic_dep_reader import sentences_to_diagrams

# ── output dir ───────────────────────────────────────────────────────────────
OUTPUT_BASE = Path("qnlp_experiment_outputs_per_set_v2") / "exp7_morph_enriched"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# ── settings ──────────────────────────────────────────────────────────────────
N_FOLDS    = 5
EPOCHS     = 300
BATCH_SIZE = 8
SEED       = 42

SPSA_HYPERPARAMS = {
    "a":     0.05,
    "c":     0.06,
    "A":     30,        # 10% of 300 epochs
    "alpha": 0.602,
    "gamma": 0.101,
}

ANSATZ_CONFIGS = [
    ("IQP",   2),
    ("Sim14", 2),
]

# The 6 LexicalAmbiguity classes — 3 complete polysemous pairs with 10 samples each
LEXICO_6 = {
    "Ambiguity_Man", "Ambiguity_Leg",       # رجل : man / leg
    "Ambiguity_Hit", "Ambiguity_Multiply",  # ضرب : hit / multiply
    "Ambiguity_King", "Ambiguity_Possess",  # ملك : king / possess
}


# ═══════════════════════════════════════════════════════════════════════════════
#  WARM-START
# ═══════════════════════════════════════════════════════════════════════════════

def _vec_for_word(word: str) -> Optional[np.ndarray]:
    """
    Return AraVec embedding for `word`.

    Handles morphology-enriched labels like 'قرات_ASP-p_PER-3_NUM-s_GEN-f'
    by extracting the base Arabic word (everything before the first morph-tag
    separator '_ASP', '_PER', '_NUM', '_GEN') before lookup.
    Also tries definite article stripping.
    """
    if ARAVEC_KV is None:
        return None

    # Strip morph tag suffix: 'قرات_ASP-p_...' → 'قرات'
    # Arabic chars are U+0600–U+06FF; morph tags start with '_' + Latin
    base = word
    for sep in ('_ASP', '_PER', '_NUM', '_GEN'):
        if sep in word:
            base = word.split(sep)[0]
            break

    candidates = [word, base]
    if base.startswith("ال"):
        candidates.append(base[2:])
    if base.startswith("وال"):
        candidates.append(base[3:])
        candidates.append(base[1:])
    # deduplicate preserving order
    seen = set()
    unique = [c for c in candidates if c not in seen and not seen.add(c)]
    for c in unique:
        if ARAVEC_KV.has_index_for(c):
            return ARAVEC_KV.get_vector(c)
    return None


def warmstart_weights(model: NumpyModel) -> np.ndarray:
    """
    Build warm-start weights from AraVec.

    For each lambeq symbol '{word}__{type}_{idx}':
      - Look up the word in AraVec to get a 300-d vector.
      - Use component [idx % 300], mapped from [-1,1] → [0, 2π].
      - Hash fallback when word is OOV.
    """
    weights = np.empty(len(model.symbols))
    hits = 0
    for i, sym in enumerate(model.symbols):
        name = str(sym)
        word = name.split("__")[0]
        idx  = int(name.rsplit("_", 1)[-1])

        vec = _vec_for_word(word)
        if vec is not None:
            # map component from [-1,1] to [0, 2π]
            weights[i] = (float(vec[idx % len(vec)]) + 1.0) * math.pi
            hits += 1
        else:
            # deterministic hash fallback
            h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
            weights[i] = (h / 0xFFFFFFFF) * 2 * math.pi

    logger.info(f"  Warm-start: {hits}/{len(model.symbols)} symbols "
                f"hit AraVec ({100*hits/max(len(model.symbols),1):.0f}%)")
    return weights


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_experiments() -> List[Dict]:
    data = json.load(open("sentences.json", encoding="utf-8"))
    exps = []

    # ── LexicalAmbiguity reduced to 6 classes (3 polysemous pairs) ────────────
    la_filtered = [d for d in data["LexicalAmbiguity"] if d["label"] in LEXICO_6]
    exps.append({
        "name":      "LexicalAmbiguity_6class",
        "sentences": [d["sentence"] for d in la_filtered],
        "labels":    [d["label"]    for d in la_filtered],
    })
    logger.info(f"LexicalAmbiguity_6class: {len(la_filtered)} sentences, "
                f"{len(set(d['label'] for d in la_filtered))} classes")

    # ── Morphology sub-experiments ────────────────────────────────────────────
    morph = data["Morphology"]

    number_map = {
        "Morph_SgMasc":"Sg", "Morph_SgFem":"Sg",
        "Morph_DualMasc":"Du", "Morph_DualFem":"Du",
        "Morph_PlMasc":"Pl", "Morph_PlFem":"Pl",
        "Morph_PlBroken":"Pl", "Morph_AdjPlMasc":"Pl", "Morph_AdjPlFem":"Pl",
    }
    tense_map  = {"Morph_Past":"Past", "Morph_Pres":"Pres"}
    poss_map   = {
        "Morph_Poss1Sg":"Poss_1st",  "Morph_Poss1Pl":"Poss_1st",
        "Morph_Poss2MascSg":"Poss_2nd","Morph_Poss2Pl":"Poss_2nd",
        "Morph_Poss3MascSg":"Poss_3rd","Morph_Poss3FemSg":"Poss_3rd",
    }

    for subname, mapping in [("Morphology_Tense", tense_map),
                              ("Morphology_Number", number_map),
                              ("Morphology_Possession", poss_map)]:
        sub = [{"sentence": d["sentence"], "label": mapping[d["label"]]}
               for d in morph if d["label"] in mapping]
        exps.append({
            "name":      subname,
            "sentences": [d["sentence"] for d in sub],
            "labels":    [d["label"]    for d in sub],
        })
        logger.info(f"{subname}: {len(sub)} sentences, "
                    f"{len(set(d['label'] for d in sub))} classes")

    return exps


# ═══════════════════════════════════════════════════════════════════════════════
#  CIRCUIT BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def _make_ansatz(name: str, n_s_qubits: int, n_layers: int):
    ob = {N: 1, S: n_s_qubits}
    if name == "IQP":
        return IQPAnsatz(ob, n_layers=n_layers, discard=False)
    elif name == "Sim14":
        return Sim14Ansatz(ob, n_layers=n_layers, discard=False)
    raise ValueError(name)


def build_circuits(diagrams: List[Diagram], ansatz) -> Tuple[List, List[int]]:
    rw = RemoveCupsRewriter()
    circuits, valid = [], []
    for i, d in enumerate(diagrams):
        try:
            circuits.append(ansatz(rw(d)))
            valid.append(i)
        except Exception as e:
            logger.warning(f"  [diagram {i}] circuit failed: {e}")
    return circuits, valid


# ═══════════════════════════════════════════════════════════════════════════════
#  LABEL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def encode_labels(labels: List[str], class_list: List[str],
                  n_s_qubits: int) -> np.ndarray:
    """One-hot into shape (n, 2, 2, ...) with n_s_qubits twos."""
    n_flat = 2 ** n_s_qubits
    flat   = np.zeros((len(labels), n_flat))
    for i, lbl in enumerate(labels):
        flat[i, class_list.index(lbl)] = 1.0
    out_shape = tuple(2 for _ in range(n_s_qubits))
    return flat.reshape((len(labels),) + out_shape)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAIN / EVAL ONE FOLD
# ═══════════════════════════════════════════════════════════════════════════════

def train_fold(
    all_circuits, all_labels_enc,
    tr_idx, te_idx,
    is_binary: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:

    train_c = [all_circuits[i] for i in tr_idx]
    test_c  = [all_circuits[i] for i in te_idx]
    train_y = all_labels_enc[tr_idx]
    test_y  = all_labels_enc[te_idx]

    try:
        model = NumpyModel.from_diagrams(all_circuits, use_jit=False)
    except Exception as e:
        logger.error(f"    Model creation failed: {e}")
        return 0.0, np.array([]), np.array([])

    # ── warm-start ────────────────────────────────────────────────────────
    model.weights = warmstart_weights(model)

    loss_fn = BinaryCrossEntropyLoss() if is_binary else CrossEntropyLoss()

    trainer = QuantumTrainer(
        model             = model,
        loss_function     = loss_fn,
        epochs            = EPOCHS,
        optimizer         = SPSAOptimizer,
        optim_hyperparams = SPSA_HYPERPARAMS,
        seed              = SEED,
        verbose           = "suppress",
    )

    train_ds = Dataset(train_c, train_y, batch_size=BATCH_SIZE, shuffle=True)
    val_ds   = Dataset(test_c,  test_y,  batch_size=BATCH_SIZE, shuffle=False)

    try:
        trainer.fit(train_ds, val_ds, log_interval=100, eval_interval=100)
    except Exception as e:
        logger.error(f"    Training failed: {e}")
        return 0.0, np.array([]), np.array([])

    try:
        preds      = model(test_c)
        preds_flat = preds.reshape(len(test_c), -1)
        targs_flat = test_y.reshape(len(test_c), -1)
        y_pred = np.argmax(preds_flat, axis=1)
        y_true = np.argmax(targs_flat, axis=1)
        return float(accuracy_score(y_true, y_pred)), y_true, y_pred
    except Exception as e:
        logger.error(f"    Eval failed: {e}")
        return 0.0, np.array([]), np.array([])


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL CV EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(exp_name, sentences, labels, ansatz_name, n_layers) -> Dict:
    logger.info(f"\n{'='*62}")
    logger.info(f"  {exp_name}  |  {ansatz_name}")
    logger.info(f"  n={len(sentences)}  classes={sorted(set(labels))}")

    class_list = sorted(set(labels))
    n_classes  = len(class_list)
    n_s_qubits = max(1, math.ceil(math.log2(max(n_classes, 2))))
    is_binary  = (n_classes == 2)

    # 1. diagrams
    t0 = time.time()
    logger.info("  Building diagrams …")
    diagrams = sentences_to_diagrams(sentences, debug=False)
    logger.info(f"  Done in {time.time()-t0:.1f}s")

    # 2. circuits
    ansatz = _make_ansatz(ansatz_name, n_s_qubits, n_layers)
    circuits, valid_idx = build_circuits(diagrams, ansatz)

    if len(circuits) < max(N_FOLDS, n_classes):
        logger.error(f"  Only {len(circuits)} circuits — skipping.")
        return {"exp_name": exp_name, "ansatz": ansatz_name,
                "error": "insufficient circuits", "accuracy_mean": 0.0,
                "accuracy_std": 0.0}

    valid_labels = [labels[i] for i in valid_idx]
    label_ints   = np.array([class_list.index(l) for l in valid_labels])
    labels_enc   = encode_labels(valid_labels, class_list, n_s_qubits)

    n_skip = len(sentences) - len(circuits)
    if n_skip:
        logger.warning(f"  {n_skip} circuits skipped.")

    # 3. CV
    min_count    = min(Counter(valid_labels).values())
    actual_folds = min(N_FOLDS, min_count)
    logger.info(f"  {actual_folds}-fold StratifiedKFold | "
                f"n_s_qubits={n_s_qubits} | binary={is_binary}")

    skf        = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=SEED)
    fold_accs  = []
    all_ytrue, all_ypred = [], []

    for fold, (tr, te) in enumerate(skf.split(circuits, label_ints)):
        logger.info(f"    Fold {fold+1}/{actual_folds}  "
                    f"(train={len(tr)}, test={len(te)})")
        t_fold = time.time()
        acc, yt, yp = train_fold(circuits, labels_enc, tr, te, is_binary)
        fold_accs.append(acc)
        if len(yt):
            all_ytrue.extend(yt.tolist())
            all_ypred.extend(yp.tolist())
        logger.info(f"    Fold {fold+1} acc={acc:.4f}  ({time.time()-t_fold:.1f}s)")

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))
    logger.info(f"  CV accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    report = "N/A"
    if all_ytrue:
        report = classification_report(
            all_ytrue, all_ypred,
            labels=list(range(n_classes)),
            target_names=class_list,
            zero_division=0,
        )
        logger.info(f"\n{report}")

    return {
        "exp_name":        exp_name,
        "ansatz":          ansatz_name,
        "n_samples":       len(circuits),
        "n_classes":       n_classes,
        "n_s_qubits":      n_s_qubits,
        "fold_accuracies": fold_accs,
        "accuracy_mean":   mean_acc,
        "accuracy_std":    std_acc,
        "class_list":      class_list,
        "classification_report": report,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("  Arabic QNLP Experiment 7 — Morph-Enriched Labels + AraVec Warm-Start")
    logger.info("=" * 70)

    experiments = load_experiments()
    all_results = []

    for ex in experiments:
        for ansatz_name, n_layers in ANSATZ_CONFIGS:
            result = run_experiment(
                ex["name"], ex["sentences"], ex["labels"],
                ansatz_name, n_layers,
            )
            all_results.append(result)

            out_dir  = OUTPUT_BASE / ansatz_name / ex["name"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"results_{ex['name']}.json"
            safe = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in result.items()}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(safe, f, ensure_ascii=False, indent=2)
            logger.info(f"  Saved → {out_path}")

    # master summary
    def _j(o):
        if isinstance(o, (np.ndarray,)): return o.tolist()
        if isinstance(o, np.floating):   return float(o)
        if isinstance(o, np.integer):    return int(o)
        raise TypeError(type(o))

    summary_path = OUTPUT_BASE / "run_summary_exp7.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": all_results,
                   "settings": {"epochs": EPOCHS, "n_folds": N_FOLDS,
                                 "batch_size": BATCH_SIZE, "warmstart": True,
                                 "morph_enriched_labels": True,
                                 "verb_rescue": True,
                                 "optimizer": "SPSA", "spsa": SPSA_HYPERPARAMS}},
                  f, ensure_ascii=False, indent=2, default=_j)
    logger.info(f"\nSummary → {summary_path}")

    # ── final table ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  FINAL RESULTS (exp7)")
    logger.info("=" * 70)
    logger.info(f"{'Experiment':<35} {'Ansatz':<10} {'Mean Acc':>10} {'Std':>8}")
    logger.info("-" * 65)
    for r in all_results:
        m = r.get("accuracy_mean", 0)
        s = r.get("accuracy_std", 0)
        logger.info(f"{r['exp_name']:<35} {r['ansatz']:<10} {m:>10.4f} {s:>8.4f}")


if __name__ == "__main__":
    main()
