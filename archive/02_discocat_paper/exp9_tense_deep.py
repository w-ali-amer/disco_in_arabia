# -*- coding: utf-8 -*-
"""
exp9_tense_deep.py
------------------
Deep ablation on Morphology_Tense — the task where quantum beats classical.

Ablation dimensions:
  - n_layers: 1, 2, 3
  - seeds:    42, 123, 777  (3 independent runs per config)
  - ansatzes: IQP, Sim14
  - epochs:   500

Also runs Morphology_Number and Morphology_Possession with n_layers=2
(best from Tense) to check whether more layers help those tasks too.

Uses the augmented sentences.json (Tense: 50 sentences, up from 28).

Usage:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 exp9_tense_deep.py
"""

import os, sys, json, math, hashlib, logging, warnings, time
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("exp9_tense_deep.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("exp9")

# ── AraVec ────────────────────────────────────────────────────────────────────
try:
    from gensim.models import Word2Vec
    _kv = Word2Vec.load("aravec/full_uni_cbow_300_twitter.mdl").wv
    logger.info(f"AraVec: {len(_kv)} vectors × {_kv.vector_size}d")
    ARAVEC_DIM = _kv.vector_size
    ARAVEC_KV  = _kv
except Exception as e:
    logger.critical(f"AraVec load failed: {e}")
    sys.exit(1)

# ── lambeq ────────────────────────────────────────────────────────────────────
from lambeq import (
    IQPAnsatz, Sim14Ansatz, RemoveCupsRewriter,
    NumpyModel, QuantumTrainer, SPSAOptimizer, Dataset,
)
from lambeq.backend.grammar import Ty
from lambeq.training import BinaryCrossEntropyLoss, CrossEntropyLoss

_remove_cups = RemoveCupsRewriter()

# ── arabic parser ──────────────────────────────────────────────────────────────
from arabic_dep_reader import sentences_to_diagrams

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# ── settings ──────────────────────────────────────────────────────────────────
N_FOLDS    = 5
EPOCHS     = 500
BATCH_SIZE = 8
SEEDS      = [42, 123, 777]

N = Ty('n')
S = Ty('s')

SPSA_HYPERPARAMS = {"a": 0.05, "c": 0.06, "A": 50, "alpha": 0.602, "gamma": 0.101}

OUTPUT_DIR = Path("qnlp_experiment_outputs_per_set_v2") / "exp9_tense_deep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Morphology label mappings
TENSE_MAP = {"Morph_Past": "Past", "Morph_Pres": "Pres"}
NUMBER_MAP = {
    "Morph_SgMasc": "Sg", "Morph_SgFem": "Sg",
    "Morph_DualMasc": "Du", "Morph_DualFem": "Du",
    "Morph_PlMasc": "Pl", "Morph_PlFem": "Pl",
    "Morph_PlBroken": "Pl", "Morph_AdjPlMasc": "Pl", "Morph_AdjPlFem": "Pl",
}
POSS_MAP = {
    "Morph_Poss1Sg": "Poss_1st",  "Morph_Poss1Pl": "Poss_1st",
    "Morph_Poss2MascSg": "Poss_2nd", "Morph_Poss2Pl": "Poss_2nd",
    "Morph_Poss3MascSg": "Poss_3rd", "Morph_Poss3FemSg": "Poss_3rd",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  AraVec warm-start
# ═══════════════════════════════════════════════════════════════════════════════

def _vec_for_word(word: str) -> Optional[np.ndarray]:
    base = word
    for sep in ('_ASP', '_PER', '_NUM', '_GEN'):
        if sep in word:
            base = word.split(sep)[0]
            break
    candidates = list(dict.fromkeys([word, base]))
    if base.startswith("ال"):
        candidates.append(base[2:])
    for c in candidates:
        if ARAVEC_KV.has_index_for(c):
            return ARAVEC_KV.get_vector(c)
    return None


def warmstart_weights(model: NumpyModel) -> np.ndarray:
    weights = np.empty(len(model.symbols))
    hits = 0
    for i, sym in enumerate(model.symbols):
        name = str(sym)
        word = name.split("__")[0]
        try:
            idx = int(name.rsplit("_", 1)[-1])
        except ValueError:
            idx = 0
        vec = _vec_for_word(word)
        if vec is not None:
            weights[i] = (float(vec[idx % len(vec)]) + 1.0) * math.pi
            hits += 1
        else:
            h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
            weights[i] = (h / 0xFFFFFFFF) * 2 * math.pi
    logger.debug(f"    Warm-start: {hits}/{len(weights)} AraVec hits")
    return weights


# ═══════════════════════════════════════════════════════════════════════════════
#  CIRCUIT BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def make_ansatz(ans_name: str, n_s_qubits: int, n_layers: int):
    ob = {S: n_s_qubits, N: 1}
    if ans_name == "IQP":
        return IQPAnsatz(ob, n_layers=n_layers, discard=False)
    elif ans_name == "Sim14":
        return Sim14Ansatz(ob, n_layers=n_layers, discard=False)
    raise ValueError(ans_name)


def build_circuits(diagrams, ansatz) -> Tuple[list, List[int]]:
    circuits, valid = [], []
    for i, d in enumerate(diagrams):
        try:
            circuits.append(ansatz(_remove_cups(d)))
            valid.append(i)
        except Exception as e:
            logger.warning(f"  Circuit [{i}] failed: {e}")
    return circuits, valid


# ═══════════════════════════════════════════════════════════════════════════════
#  LABEL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def encode_labels(labels: List[str], class_list: List[str], n_s_qubits: int) -> np.ndarray:
    n_flat = 2 ** n_s_qubits
    flat   = np.zeros((len(labels), n_flat))
    for i, lbl in enumerate(labels):
        flat[i, class_list.index(lbl)] = 1.0
    return flat.reshape((len(labels),) + tuple(2 for _ in range(n_s_qubits)))


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAIN ONE FOLD
# ═══════════════════════════════════════════════════════════════════════════════

def train_fold(all_circuits, tr_idx, te_idx, labels_enc, n_classes: int, seed: int) -> Tuple[float, list, list]:
    train_c = [all_circuits[i] for i in tr_idx]
    test_c  = [all_circuits[i] for i in te_idx]
    train_y = labels_enc[tr_idx]
    test_y  = labels_enc[te_idx]

    try:
        model = NumpyModel.from_diagrams(all_circuits, use_jit=False)
    except Exception as e:
        logger.error(f"    NumpyModel failed: {e}")
        return 0.0, [], []

    model.weights = warmstart_weights(model)
    loss_fn = BinaryCrossEntropyLoss() if n_classes == 2 else CrossEntropyLoss()

    trainer = QuantumTrainer(
        model             = model,
        loss_function     = loss_fn,
        epochs            = EPOCHS,
        optimizer         = SPSAOptimizer,
        optim_hyperparams = SPSA_HYPERPARAMS,
        seed              = seed,
        verbose           = "suppress",
    )
    train_ds = Dataset(train_c, train_y, batch_size=BATCH_SIZE, shuffle=True)
    val_ds   = Dataset(test_c,  test_y,  batch_size=BATCH_SIZE, shuffle=False)

    try:
        trainer.fit(train_ds, val_ds, log_interval=200, eval_interval=200)
    except Exception as e:
        logger.error(f"    Training failed: {e}")
        return 0.0, [], []

    try:
        preds      = model(test_c)
        preds_flat = preds.reshape(len(test_c), -1)
        targs_flat = test_y.reshape(len(test_c), -1)
        y_pred = np.argmax(preds_flat, axis=1).tolist()
        y_true = np.argmax(targs_flat, axis=1).tolist()
        acc = float(accuracy_score(y_true, y_pred))
        return acc, y_true, y_pred
    except Exception as e:
        logger.error(f"    Prediction failed: {e}")
        return 0.0, [], []


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL CV RUN FOR ONE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

def run_config(
    exp_name: str,
    sentences: List[str],
    labels: List[str],
    ans_name: str,
    n_layers: int,
    seed: int,
) -> Dict:
    t_start = time.time()
    class_list = sorted(set(labels))
    n_classes  = len(class_list)
    n_s_qubits = max(1, math.ceil(math.log2(max(n_classes, 2))))

    tag = f"{exp_name}/{ans_name}/L{n_layers}/seed{seed}"
    logger.info(f"  {tag}  n={len(sentences)}  n_s_qubits={n_s_qubits}")

    # Build diagrams once per experiment (independent of seed/ansatz)
    ansatz = make_ansatz(ans_name, n_s_qubits, n_layers)

    # We cache diagrams per exp_name to avoid re-parsing every config
    diagrams = sentences_to_diagrams(sentences, log_interval=0)
    circuits, valid_idx = build_circuits(diagrams, ansatz)

    if len(circuits) < n_classes * 2:
        logger.error(f"  Insufficient circuits: {len(circuits)}")
        return {"tag": tag, "error": "insufficient circuits", "mean": 0.0, "std": 0.0}

    valid_labels = [labels[i] for i in valid_idx]
    labels_enc   = encode_labels(valid_labels, class_list, n_s_qubits)

    min_count = min(Counter(valid_labels).values())
    folds     = min(N_FOLDS, min_count)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    fold_accs = []
    all_true, all_pred = [], []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(circuits, valid_labels)):
        t0 = time.time()
        acc, yt, yp = train_fold(circuits, tr_idx, te_idx, labels_enc, n_classes, seed + fold_i)
        fold_accs.append(acc)
        all_true.extend(yt)
        all_pred.extend(yp)
        logger.info(f"    fold {fold_i+1}/{folds}  acc={acc:.4f}  ({time.time()-t0:.0f}s)")

    mean = float(np.mean(fold_accs))
    std  = float(np.std(fold_accs))

    report = ""
    if all_true and all_pred:
        try:
            report = classification_report(all_true, all_pred,
                                           labels=list(range(n_classes)),
                                           target_names=class_list,
                                           zero_division=0)
        except Exception:
            pass

    elapsed = time.time() - t_start
    logger.info(f"  → {tag}: {mean:.4f} ± {std:.4f}  ({elapsed:.0f}s total)")

    return {
        "tag": tag, "exp_name": exp_name, "ansatz": ans_name,
        "n_layers": n_layers, "seed": seed,
        "n_samples": len(circuits), "n_classes": n_classes,
        "chance": 1.0 / n_classes,
        "mean": mean, "std": std, "folds": fold_accs, "report": report,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_morph_experiments() -> List[Dict]:
    data  = json.load(open("sentences.json", encoding="utf-8"))
    morph = data.get("Morphology", [])
    exps  = []

    for name, mapping in [
        ("Morphology_Tense",      TENSE_MAP),
        ("Morphology_Number",     NUMBER_MAP),
        ("Morphology_Possession", POSS_MAP),
    ]:
        sub = [{"sentence": d["sentence"], "label": mapping[d["label"]]}
               for d in morph if d["label"] in mapping]
        exps.append({
            "name": name,
            "sentences": [d["sentence"] for d in sub],
            "labels":    [d["label"]    for d in sub],
        })
        logger.info(f"{name}: {len(sub)} sentences  classes={sorted(set(d['label'] for d in sub))}")
    return exps


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("  Arabic QNLP — exp9: Tense Deep Ablation + Morphology (augmented)")
    logger.info("=" * 70)
    logger.info(f"  Epochs={EPOCHS}  Seeds={SEEDS}  Folds={N_FOLDS}")

    exps = load_morph_experiments()

    # Ablation plan:
    # Tense: n_layers=[1,2,3], ansatzes=[IQP,Sim14], seeds=3  → 18 configs
    # Number, Possession: n_layers=2 only (best guess from Tense), ansatzes=[IQP,Sim14], seeds=3 → 12 configs each
    configs = []

    for ex in exps:
        if "Tense" in ex["name"]:
            for ans in ["IQP", "Sim14"]:
                for nl in [1, 2, 3]:
                    for seed in SEEDS:
                        configs.append((ex, ans, nl, seed))
        else:
            # Number and Possession: n_layers=2 only (saves time)
            for ans in ["IQP", "Sim14"]:
                for seed in SEEDS:
                    configs.append((ex, ans, 2, seed))

    total = len(configs)
    logger.info(f"  Total configs: {total}")

    all_results = []

    for ci, (ex, ans_name, n_layers, seed) in enumerate(configs):
        logger.info(f"\n[{ci+1}/{total}]")
        result = run_config(ex["name"], ex["sentences"], ex["labels"],
                            ans_name, n_layers, seed)
        all_results.append(result)

        fname = OUTPUT_DIR / f"results_{ex['name']}_{ans_name}_L{n_layers}_s{seed}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # ── Summary: Tense ablation ───────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  TENSE ABLATION SUMMARY (mean across seeds)")
    logger.info("=" * 70)
    logger.info(f"{'Ansatz':<8} {'Layers':<8} {'Mean±Std (avg 3 seeds)':>28}  folds")
    logger.info("-" * 60)

    from collections import defaultdict
    tense_res = [r for r in all_results if "Tense" in r.get("exp_name","") and "error" not in r]

    # Group by (ansatz, n_layers)
    grouped = defaultdict(list)
    for r in tense_res:
        grouped[(r["ansatz"], r["n_layers"])].append(r["mean"])

    for (ans, nl) in sorted(grouped.keys()):
        vals  = grouped[(ans, nl)]
        mu    = float(np.mean(vals))
        sigma = float(np.std(vals))
        logger.info(f"{ans:<8} L={nl:<6} {mu:.4f} ± {sigma:.4f}   seeds={[round(v,3) for v in vals]}")

    # Best overall
    best_tense = max((r for r in tense_res), key=lambda r: r["mean"])
    logger.info(f"\n  Best single run: {best_tense['tag']}  acc={best_tense['mean']:.4f}")
    logger.info(f"  Classical best (SVM_rbf): 54.7%  →  Quantum gap: {best_tense['mean']-0.547:+.1%}")

    # ── Morphology summary ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  MORPHOLOGY SUMMARY (n_layers=2, avg 3 seeds)")
    logger.info("=" * 70)
    logger.info(f"{'Experiment':<28} {'Ansatz':<8} {'Mean':>8} {'Std':>8}  Classical best")
    logger.info("-" * 65)

    cl_best = {"Morphology_Tense": 0.547, "Morphology_Number": 0.888, "Morphology_Possession": 0.849}

    morph_res = [r for r in all_results if "error" not in r]
    by_exp_ans = defaultdict(list)
    for r in morph_res:
        by_exp_ans[(r["exp_name"], r["ansatz"], r.get("n_layers", 2))].append(r["mean"])

    for exp_name in ["Morphology_Tense", "Morphology_Number", "Morphology_Possession"]:
        for ans in ["IQP", "Sim14"]:
            nl = 2  # summarise n_layers=2 for comparability
            vals = by_exp_ans.get((exp_name, ans, nl), [])
            if not vals:
                continue
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            cl = cl_best.get(exp_name, 0)
            logger.info(f"{exp_name:<28} {ans:<8} {mu:>8.4f} {sd:>8.4f}  {cl:.1%}")

    # Save full summary
    summary_path = OUTPUT_DIR / "exp9_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()
