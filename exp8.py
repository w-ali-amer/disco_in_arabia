# -*- coding: utf-8 -*-
"""
exp8.py
-------
Binary QNLP classifiers for each polysemous Arabic word pair, using the
augmented dataset (sentences_augmented.json, 15 samples per class).

Strategy:
  - 7 binary experiments, one per polysemous pair (رجل/عين/ملك/ضرب/جمل/فتح/علم)
  - n_s_qubits = 1  →  NumpyModel output shape = (batch, 2)
  - SPSA 300 epochs, batch 8, IQP + Sim14 ansatzes
  - AraVec warm-start
  - 5-fold StratifiedKFold CV (or 3-fold if class count < 5)
  - Also runs augmented 14-class and 6-class experiments for comparison

Usage:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 exp8.py
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
        logging.FileHandler("exp8_binary_lexico.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("exp8")

# ── AraVec ────────────────────────────────────────────────────────────────────
try:
    from gensim.models import Word2Vec
    _kv = Word2Vec.load("aravec/full_uni_cbow_300_twitter.mdl").wv
    logger.info(f"AraVec loaded: {len(_kv)} vectors × {_kv.vector_size}d")
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
from lambeq.training import CrossEntropyLoss, BinaryCrossEntropyLoss

_remove_cups = RemoveCupsRewriter()

# ── arabic parser ──────────────────────────────────────────────────────────────
from arabic_dep_reader import sentences_to_diagrams

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── settings ──────────────────────────────────────────────────────────────────
N_FOLDS    = 5
SEED       = 42
EPOCHS     = 300
BATCH_SIZE = 8
OUTPUT_DIR = Path("qnlp_experiment_outputs_per_set_v2") / "exp8_binary_lexico"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = "sentences_augmented.json"

ANSATZ_NAMES = ["IQP", "Sim14"]

N = Ty('n')
S = Ty('s')

def _make_ansatz(name: str, n_s_qubits: int):
    ob = {S: n_s_qubits, N: 1}
    if name == "IQP":
        return IQPAnsatz(ob, n_layers=1, discard=False)
    elif name == "Sim14":
        return Sim14Ansatz(ob, n_layers=1, discard=False)
    raise ValueError(name)

SPSA_PARAMS = {"a": 0.05, "c": 0.06, "A": 30, "alpha": 0.602, "gamma": 0.101}

# Polysemous pairs: (word_arabic, class_A, class_B)
PAIRS = [
    ("رجل",  "Ambiguity_Man",       "Ambiguity_Leg"),
    ("عين",  "Ambiguity_Eye",       "Ambiguity_Spring"),
    ("ملك",  "Ambiguity_King",      "Ambiguity_Possess"),
    ("ضرب",  "Ambiguity_Hit",       "Ambiguity_Multiply"),
    ("جمل",  "Ambiguity_Camel",     "Ambiguity_Sentences"),
    ("فتح",  "Ambiguity_Open",      "Ambiguity_Conquer"),
    ("علم",  "Ambiguity_Knowledge", "Ambiguity_Flag"),
]

LEXICO_6 = {"Ambiguity_Man", "Ambiguity_Leg",
            "Ambiguity_Hit", "Ambiguity_Multiply",
            "Ambiguity_King", "Ambiguity_Possess"}


# ═══════════════════════════════════════════════════════════════════════════════
#  AraVec warm-start
# ═══════════════════════════════════════════════════════════════════════════════

def _vec_for_word(word: str) -> Optional[np.ndarray]:
    """Look up a word (or its morphological base) in AraVec."""
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
        else:
            h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
            weights[i] = (h / 0xFFFFFFFF) * 2 * math.pi
    return weights


# ═══════════════════════════════════════════════════════════════════════════════
#  CIRCUIT BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def build_circuits(sentences: List[str], ansatz) -> Tuple[list, List[int]]:
    """
    Build quantum circuits from sentences. Returns (circuits, valid_indices).
    valid_indices: indices of sentences that produced a usable circuit.
    """
    raw_diagrams = sentences_to_diagrams(sentences, log_interval=20)

    circuits = []
    valid_idx = []
    for i, diag in enumerate(raw_diagrams):
        try:
            rewritten = _remove_cups(diag)
            circuit   = ansatz(rewritten)
            circuits.append(circuit)
            valid_idx.append(i)
        except Exception as exc:
            logger.warning(f"  Circuit build failed [{i}] '{sentences[i][:30]}': {exc}")

    logger.info(f"  Circuits: {len(circuits)}/{len(sentences)} succeeded.")
    return circuits, valid_idx


# ═══════════════════════════════════════════════════════════════════════════════
#  LABEL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def encode_labels(labels: List[str], class_list: List[str], n_s_qubits: int) -> np.ndarray:
    n_flat     = 2 ** n_s_qubits
    out_shape  = tuple(2 for _ in range(n_s_qubits))
    flat       = np.zeros((len(labels), n_flat))
    for i, lbl in enumerate(labels):
        flat[i, class_list.index(lbl)] = 1.0
    return flat.reshape((len(labels),) + out_shape)


# ═══════════════════════════════════════════════════════════════════════════════
#  SINGLE FOLD TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_eval_fold(
    train_circuits, train_labels,
    test_circuits,  test_labels,
    class_list:    List[str],
    n_s_qubits:    int,
    ansatz_name:   str,
    fold_num:      int,
) -> Dict:
    n_classes = len(class_list)
    all_circuits = train_circuits + test_circuits
    try:
        model = NumpyModel.from_diagrams(all_circuits, use_jit=False)
    except Exception as e:
        logger.error(f"    NumpyModel build failed: {e}")
        return {"acc": 0.0, "fold": fold_num}

    model.weights = warmstart_weights(model)

    train_targets = encode_labels(train_labels, class_list, n_s_qubits)
    test_targets  = encode_labels(test_labels,  class_list, n_s_qubits)

    is_binary = (n_classes == 2)
    loss_fn   = BinaryCrossEntropyLoss() if is_binary else CrossEntropyLoss()

    trainer = QuantumTrainer(
        model             = model,
        loss_function     = loss_fn,
        epochs            = EPOCHS,
        optimizer         = SPSAOptimizer,
        optim_hyperparams = SPSA_PARAMS,
        seed              = SEED + fold_num,
        verbose           = "suppress",
    )

    train_ds = Dataset(train_circuits, train_targets, batch_size=BATCH_SIZE, shuffle=True)
    val_ds   = Dataset(test_circuits,  test_targets,  batch_size=BATCH_SIZE, shuffle=False)

    try:
        trainer.fit(train_ds, val_ds, log_interval=100, eval_interval=100)
    except Exception as e:
        logger.error(f"    Training failed: {e}")
        return {"acc": 0.0, "fold": fold_num}

    # Predict on test set
    try:
        preds = model(test_circuits)
        preds_flat  = preds.reshape(len(test_circuits), -1)
        targs_flat  = test_targets.reshape(len(test_circuits), -1)
        y_pred = np.argmax(preds_flat, axis=1)
        y_true = np.argmax(targs_flat, axis=1)
        acc = float(accuracy_score(y_true, y_pred))
    except Exception as e:
        logger.error(f"    Prediction failed: {e}")
        acc = 0.0
        y_pred = np.zeros(len(test_circuits), dtype=int)
        y_true = np.zeros(len(test_circuits), dtype=int)

    return {"acc": acc, "y_true": y_true.tolist(), "y_pred": y_pred.tolist(), "fold": fold_num}


# ═══════════════════════════════════════════════════════════════════════════════
#  CV RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_cv_experiment(
    exp_name:  str,
    sentences: List[str],
    labels:    List[str],
    ansatz_name: str,
) -> Dict:
    logger.info(f"\n{'─'*60}")
    logger.info(f"  {exp_name} | {ansatz_name}  n={len(sentences)}")

    class_list  = sorted(set(labels))
    n_classes   = len(class_list)
    n_s_qubits  = max(1, math.ceil(math.log2(max(n_classes, 2))))
    chance      = 1.0 / n_classes

    logger.info(f"  classes={class_list}  n_s_qubits={n_s_qubits}  chance={chance:.1%}")

    ansatz = _make_ansatz(ansatz_name, n_s_qubits)

    # Build all circuits once
    circuits, valid_idx = build_circuits(sentences, ansatz)
    if len(circuits) < n_classes * 2:
        logger.error(f"  Too few circuits ({len(circuits)}), skipping.")
        return {"exp_name": exp_name, "ansatz": ansatz_name, "error": "too few circuits"}

    valid_sents  = [sentences[i] for i in valid_idx]
    valid_labels = [labels[i]    for i in valid_idx]

    min_count = min(Counter(valid_labels).values())
    folds     = min(N_FOLDS, min_count)
    if folds < 2:
        logger.error(f"  Not enough samples for CV (min_count={min_count}), skipping.")
        return {"exp_name": exp_name, "ansatz": ansatz_name, "error": "insufficient samples"}

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    fold_accs   = []
    all_true    = []
    all_pred    = []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(circuits, valid_labels)):
        t0 = time.time()
        logger.info(f"  Fold {fold_i+1}/{folds}  train={len(tr_idx)} test={len(te_idx)}")

        train_circuits = [circuits[i] for i in tr_idx]
        test_circuits  = [circuits[i] for i in te_idx]
        train_labels_f = [valid_labels[i] for i in tr_idx]
        test_labels_f  = [valid_labels[i] for i in te_idx]

        result = train_and_eval_fold(
            train_circuits, train_labels_f,
            test_circuits,  test_labels_f,
            class_list, n_s_qubits,
            ansatz_name, fold_i,
        )
        fold_accs.append(result["acc"])
        if "y_true" in result:
            all_true.extend(result["y_true"])
            all_pred.extend(result["y_pred"])
        elapsed = time.time() - t0
        logger.info(f"    fold acc={result['acc']:.4f}  ({elapsed:.0f}s)")

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))
    logger.info(f"  → {exp_name}/{ansatz_name}: {mean_acc:.4f} ± {std_acc:.4f}  (chance {chance:.1%})")

    report = ""
    if all_true and all_pred:
        try:
            le = LabelEncoder()
            le.fit(class_list)
            report = classification_report(
                all_true, all_pred,
                labels=list(range(n_classes)),
                target_names=class_list,
                zero_division=0,
            )
        except Exception:
            pass

    return {
        "exp_name": exp_name,
        "ansatz":   ansatz_name,
        "n_samples": len(valid_sents),
        "n_classes": n_classes,
        "chance":    chance,
        "n_folds":   folds,
        "mean":      mean_acc,
        "std":       std_acc,
        "folds":     fold_accs,
        "report":    report,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> Dict:
    return json.load(open(DATA_FILE, encoding="utf-8"))


def build_experiments(data: Dict) -> List[Dict]:
    exps = []
    la = data.get("LexicalAmbiguity", [])

    # 7 binary pair experiments
    la_by_label: Dict[str, List[str]] = {}
    for d in la:
        la_by_label.setdefault(d["label"], []).append(d["sentence"])

    for word_ar, cls_a, cls_b in PAIRS:
        sents_a = la_by_label.get(cls_a, [])
        sents_b = la_by_label.get(cls_b, [])
        sents  = sents_a + sents_b
        labels = [cls_a] * len(sents_a) + [cls_b] * len(sents_b)
        pair_name = f"Binary_{cls_a.replace('Ambiguity_','')}_{cls_b.replace('Ambiguity_','')}"
        exps.append({"name": pair_name, "sentences": sents, "labels": labels})
        logger.info(f"Pair {pair_name}: {len(sents_a)}+{len(sents_b)} sentences")

    # 6-class experiment (augmented)
    la6 = [d for d in la if d["label"] in LEXICO_6]
    exps.append({
        "name": "LexicalAmbiguity_6class_aug",
        "sentences": [d["sentence"] for d in la6],
        "labels":    [d["label"]    for d in la6],
    })

    # 14-class experiment (augmented)
    exps.append({
        "name": "LexicalAmbiguity_14class_aug",
        "sentences": [d["sentence"] for d in la],
        "labels":    [d["label"]    for d in la],
    })

    return exps


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("  Arabic QNLP — exp8: Binary LexicalAmbiguity (augmented dataset)")
    logger.info("=" * 70)
    logger.info(f"  Data: {DATA_FILE}  |  Epochs: {EPOCHS}  |  Folds: {N_FOLDS}")

    data = load_data()
    experiments = build_experiments(data)

    all_results = []

    for ex in experiments:
        exp_results = []
        for ans_name in ANSATZ_NAMES:
            result = run_cv_experiment(
                ex["name"], ex["sentences"], ex["labels"],
                ans_name,
            )
            exp_results.append(result)
            all_results.append(result)

            # Save per-experiment-per-ansatz
            fname = OUTPUT_DIR / f"results_{ex['name']}_{ans_name}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    # ── Summary table ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("  EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Experiment':<42} {'Chance':>7} {'IQP':>8} {'Sim14':>8}")
    logger.info("-" * 70)

    # Group by experiment name
    by_exp: Dict[str, Dict] = {}
    for r in all_results:
        by_exp.setdefault(r["exp_name"], {})[r["ansatz"]] = r

    for exp_name, ans_dict in by_exp.items():
        chance = ans_dict.get("IQP", ans_dict.get("Sim14", {})).get("chance", 0.0)
        iqp    = ans_dict.get("IQP",   {}).get("mean", float("nan"))
        sim14  = ans_dict.get("Sim14", {}).get("mean", float("nan"))
        logger.info(
            f"{exp_name:<42} {chance:>7.1%} {iqp:>8.1%} {sim14:>8.1%}"
        )

    # Save full summary
    summary_path = OUTPUT_DIR / "exp8_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()
