# -*- coding: utf-8 -*-
"""
exp5_training.py
----------------
Arabic QNLP experiment with proper lambeq training loop.

Improvements over exp4.py:
- Uses arabic_dep_reader.py for mathematically correct pregroup diagrams
  (all diagrams guaranteed cod == Ty('s'))
- Uses NumpyModel + QuantumTrainer with SPSAOptimizer (trained weights, not
  fixed AraVec embeddings)
- 5-fold StratifiedKFold cross-validation for robust accuracy estimates
- Morphology restructured into 3 sub-experiments:
    * Number  (Sg / Dual / Pl)         — 3-class
    * Tense   (Past / Present)          — binary
    * Possession (1st / 2nd / 3rd)      — 3-class
- Results saved in same directory structure as exp4.py outputs

Usage:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 exp5_training.py
"""

import os
import sys
import json
import math
import logging
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("qnlp_experiment_exp5.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("exp5")

# ── lambeq grammar ────────────────────────────────────────────────────────────
try:
    from lambeq.backend.grammar import Ty, Word, Diagram
    N = Ty('n')
    S = Ty('s')
    logger.info("lambeq grammar types loaded.")
except ImportError as e:
    logger.critical(f"Cannot import lambeq grammar: {e}")
    sys.exit(1)

# ── lambeq ansatzes ───────────────────────────────────────────────────────────
try:
    from lambeq import IQPAnsatz, StronglyEntanglingAnsatz, Sim14Ansatz
    from lambeq import RemoveCupsRewriter
    logger.info("lambeq ansatzes loaded.")
except ImportError as e:
    logger.critical(f"Cannot import lambeq ansatzes: {e}")
    sys.exit(1)

# ── lambeq training ───────────────────────────────────────────────────────────
try:
    from lambeq import NumpyModel, QuantumTrainer, SPSAOptimizer, Dataset
    from lambeq.training import CrossEntropyLoss, BinaryCrossEntropyLoss
    logger.info("lambeq training modules loaded.")
except ImportError as e:
    logger.critical(f"Cannot import lambeq training: {e}")
    sys.exit(1)

# ── sklearn ───────────────────────────────────────────────────────────────────
try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, classification_report
    logger.info("sklearn loaded.")
except ImportError as e:
    logger.critical(f"Cannot import sklearn: {e}")
    sys.exit(1)

# ── arabic_dep_reader (our new diagram builder) ───────────────────────────────
try:
    from arabic_dep_reader import sentences_to_diagrams
    logger.info("arabic_dep_reader loaded.")
except ImportError as e:
    logger.critical(f"Cannot import arabic_dep_reader: {e}")
    sys.exit(1)

# ── output directory ──────────────────────────────────────────────────────────
OUTPUT_BASE = Path("qnlp_experiment_outputs_per_set_v2") / "exp5_trained"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# ── experiment settings ───────────────────────────────────────────────────────
N_FOLDS   = 5
EPOCHS    = 150
BATCH_SIZE = 8
SEED      = 42

SPSA_HYPERPARAMS = {
    "a": 0.05,    # step size scale
    "c": 0.06,    # perturbation scale
    "A": 15,      # stability constant (≈10% of EPOCHS)
    "alpha": 0.602,
    "gamma": 0.101,
}

ANSATZ_CONFIGS = [
    ("IQP",               2, 1),   # (name, n_layers, n_single_q_params)
    ("StronglyEntangling", 1, 1),
    ("Sim14",             2, 1),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_sentences_json(path: str = "sentences.json") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _group_morphology(raw: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Split Morphology (17 classes) into 3 tractable sub-experiments.

    Number sub-exp:
        Sg   ← SgMasc, SgFem
        Du   ← DualMasc, DualFem
        Pl   ← PlMasc, PlFem, PlBroken, AdjPlMasc, AdjPlFem

    Tense sub-exp:
        Past ← Past
        Pres ← Pres

    Possession sub-exp:
        Poss_1st ← Poss1Sg, Poss1Pl
        Poss_2nd ← Poss2MascSg, Poss2Pl
        Poss_3rd ← Poss3MascSg, Poss3FemSg
    """
    number_map = {
        "Morph_SgMasc": "Sg",   "Morph_SgFem": "Sg",
        "Morph_DualMasc": "Du", "Morph_DualFem": "Du",
        "Morph_PlMasc": "Pl",   "Morph_PlFem": "Pl",
        "Morph_PlBroken": "Pl", "Morph_AdjPlMasc": "Pl",
        "Morph_AdjPlFem": "Pl",
    }
    tense_map = {
        "Morph_Past": "Past",
        "Morph_Pres": "Pres",
    }
    poss_map = {
        "Morph_Poss1Sg":    "Poss_1st", "Morph_Poss1Pl":    "Poss_1st",
        "Morph_Poss2MascSg":"Poss_2nd", "Morph_Poss2Pl":    "Poss_2nd",
        "Morph_Poss3MascSg":"Poss_3rd", "Morph_Poss3FemSg": "Poss_3rd",
    }

    number_data, tense_data, poss_data = [], [], []
    for item in raw:
        lbl = item["label"]
        if lbl in number_map:
            number_data.append({"sentence": item["sentence"],
                                 "label": number_map[lbl]})
        if lbl in tense_map:
            tense_data.append({"sentence": item["sentence"],
                                "label": tense_map[lbl]})
        if lbl in poss_map:
            poss_data.append({"sentence": item["sentence"],
                               "label": poss_map[lbl]})

    return {
        "Morphology_Number":     number_data,
        "Morphology_Tense":      tense_data,
        "Morphology_Possession": poss_data,
    }


def build_experiments(data: Dict) -> List[Dict]:
    """
    Returns a list of experiment configs:
      { 'name': str, 'sentences': [...], 'labels': [...] }
    """
    experiments = []

    # WordOrder — keep as-is (3 classes)
    wo = data.get("WordOrder", [])
    if wo:
        experiments.append({
            "name": "WordOrder",
            "sentences": [d["sentence"] for d in wo],
            "labels":    [d["label"]    for d in wo],
        })

    # LexicalAmbiguity — keep all 14 classes
    la = data.get("LexicalAmbiguity", [])
    if la:
        experiments.append({
            "name": "LexicalAmbiguity",
            "sentences": [d["sentence"] for d in la],
            "labels":    [d["label"]    for d in la],
        })

    # Morphology — split into 3 sub-experiments
    mo = data.get("Morphology", [])
    if mo:
        for name, sub_data in _group_morphology(mo).items():
            if len(sub_data) >= N_FOLDS * 2:   # enough samples for CV
                experiments.append({
                    "name": name,
                    "sentences": [d["sentence"] for d in sub_data],
                    "labels":    [d["label"]    for d in sub_data],
                })
            else:
                logger.warning(f"Skipping {name}: only {len(sub_data)} samples.")

    return experiments


# ═══════════════════════════════════════════════════════════════════════════════
#  DIAGRAM + CIRCUIT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_ansatz(name: str, n_s_qubits: int, n_layers: int):
    """Return a configured ansatz instance."""
    ob = {N: 1, S: n_s_qubits}
    if name == "IQP":
        return IQPAnsatz(ob, n_layers=n_layers, discard=False)
    elif name == "StronglyEntangling":
        return StronglyEntanglingAnsatz(ob, n_layers=n_layers, discard=False)
    elif name == "Sim14":
        return Sim14Ansatz(ob, n_layers=n_layers, discard=False)
    else:
        raise ValueError(f"Unknown ansatz: {name}")


def build_circuits(diagrams: List[Diagram], ansatz) -> Tuple[List, List[int]]:
    """
    Apply ansatz to grammar diagrams.  Returns (circuits, valid_indices).
    Diagrams that fail are dropped with a warning.
    """
    rewriter = RemoveCupsRewriter()
    circuits, valid = [], []
    for i, d in enumerate(diagrams):
        try:
            rewritten = rewriter(d)
            circuit   = ansatz(rewritten)
            circuits.append(circuit)
            valid.append(i)
        except Exception as exc:
            logger.warning(f"  Diagram[{i}] circuit build failed: {exc}")
    return circuits, valid


# ═══════════════════════════════════════════════════════════════════════════════
#  LABEL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def encode_labels(labels: List[str], class_list: List[str],
                  n_s_qubits: int) -> np.ndarray:
    """
    One-hot encode labels into the tensor shape NumpyModel expects.

    NumpyModel with n_s_qubits outputs shape (batch, 2, 2, ...) with n_s_qubits
    twos.  We encode as flat length-2^n_s_qubits one-hot then reshape.

    e.g. n_s_qubits=2 → shape (n, 2, 2)
         n_s_qubits=1 → shape (n, 2)
         n_s_qubits=4 → shape (n, 2, 2, 2, 2)
    """
    n_flat = 2 ** n_s_qubits
    out_shape = tuple(2 for _ in range(n_s_qubits))
    flat = np.zeros((len(labels), n_flat), dtype=float)
    for i, lbl in enumerate(labels):
        idx = class_list.index(lbl)
        flat[i, idx] = 1.0
    return flat.reshape((len(labels),) + out_shape)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING + EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_eval_fold(
    all_circuits: list,
    all_labels_enc: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    is_binary: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Train NumpyModel on train split, return (accuracy, y_true, y_pred_int).
    """
    train_circuits = [all_circuits[i] for i in train_idx]
    test_circuits  = [all_circuits[i] for i in test_idx]
    train_targets  = all_labels_enc[train_idx]
    test_targets   = all_labels_enc[test_idx]

    # Build model from ALL circuits so every word's parameter is included
    try:
        model = NumpyModel.from_diagrams(all_circuits, use_jit=False)
    except Exception as e:
        logger.error(f"    NumpyModel creation failed: {e}")
        return 0.0, np.array([]), np.array([])

    model.initialise_weights()

    loss_fn = (BinaryCrossEntropyLoss()
               if is_binary else CrossEntropyLoss())

    trainer = QuantumTrainer(
        model            = model,
        loss_function    = loss_fn,
        epochs           = EPOCHS,
        optimizer        = SPSAOptimizer,
        optim_hyperparams= SPSA_HYPERPARAMS,
        seed             = SEED,
        verbose          = "suppress",
    )

    train_ds = Dataset(train_circuits, train_targets,
                       batch_size=BATCH_SIZE, shuffle=True)
    val_ds   = Dataset(test_circuits,  test_targets,
                       batch_size=BATCH_SIZE, shuffle=False)

    try:
        trainer.fit(train_ds, val_ds, log_interval=50, eval_interval=50)
    except Exception as e:
        logger.error(f"    Training failed: {e}")
        return 0.0, np.array([]), np.array([])

    # Evaluate
    try:
        preds = model(test_circuits)                    # (n_test, 2, ...) or (n_test, 2)
        preds_flat = preds.reshape(len(test_circuits), -1)   # (n_test, 2^n_s)
        targs_flat = test_targets.reshape(len(test_circuits), -1)
        y_pred = np.argmax(preds_flat, axis=1)
        y_true = np.argmax(targs_flat, axis=1)
        acc = float(accuracy_score(y_true, y_pred))
        return acc, y_true, y_pred
    except Exception as e:
        logger.error(f"    Evaluation failed: {e}")
        return 0.0, np.array([]), np.array([])


def run_cv_experiment(
    exp_name: str,
    sentences: List[str],
    labels: List[str],
    ansatz_name: str,
    n_layers: int,
) -> Dict:
    """
    Full cross-validated experiment for one (dataset, ansatz) pair.
    Returns a result dict.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  EXP: {exp_name}  |  ANSATZ: {ansatz_name}")
    logger.info(f"  Samples: {len(sentences)}  |  Classes: {len(set(labels))}")

    class_list = sorted(set(labels))
    n_classes  = len(class_list)
    n_s_qubits = max(1, math.ceil(math.log2(max(n_classes, 2))))
    n_out      = 2 ** n_s_qubits
    is_binary  = (n_classes == 2)

    logger.info(f"  n_classes={n_classes}  n_s_qubits={n_s_qubits}  "
                f"n_out={n_out}  binary={is_binary}")

    # ── 1. Build grammar diagrams ──────────────────────────────────────────
    t0 = time.time()
    logger.info("  Building grammar diagrams …")
    all_diagrams = sentences_to_diagrams(sentences, debug=False)
    logger.info(f"  Diagrams built in {time.time()-t0:.1f}s")

    # ── 2. Build circuits ──────────────────────────────────────────────────
    logger.info("  Building quantum circuits …")
    ansatz   = _make_ansatz(ansatz_name, n_s_qubits, n_layers)
    circuits, valid_idx = build_circuits(all_diagrams, ansatz)

    if len(circuits) < N_FOLDS:
        logger.error(f"  Not enough valid circuits ({len(circuits)}). Skipping.")
        return {"exp_name": exp_name, "ansatz": ansatz_name,
                "error": "insufficient circuits", "accuracy_mean": 0.0}

    # Filter sentences/labels to match valid circuits
    valid_labels = [labels[i]    for i in valid_idx]
    valid_sents  = [sentences[i] for i in valid_idx]

    n_skipped = len(sentences) - len(circuits)
    if n_skipped:
        logger.warning(f"  Skipped {n_skipped} sentences (circuit build failure).")

    # ── 3. Encode labels ──────────────────────────────────────────────────
    labels_enc = encode_labels(valid_labels, class_list, n_s_qubits)
    label_ints = np.array([class_list.index(l) for l in valid_labels])

    # ── 4. Cross-validation ───────────────────────────────────────────────
    # Check minimum class count vs folds
    from collections import Counter
    min_count = min(Counter(valid_labels).values())
    actual_folds = min(N_FOLDS, min_count)
    if actual_folds < N_FOLDS:
        logger.warning(f"  Reducing to {actual_folds} folds "
                       f"(smallest class has {min_count} samples).")

    logger.info(f"  Running {actual_folds}-fold StratifiedKFold …")
    skf  = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=SEED)
    fold_accs   = []
    all_y_true  = []
    all_y_pred  = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(circuits, label_ints)):
        logger.info(f"    Fold {fold+1}/{actual_folds}  "
                    f"(train={len(tr_idx)}, test={len(te_idx)})")
        t_fold = time.time()

        acc, y_true, y_pred = train_and_eval_fold(
            circuits, labels_enc, tr_idx, te_idx, is_binary
        )

        fold_accs.append(acc)
        if len(y_true) > 0:
            all_y_true.extend(y_true.tolist())
            all_y_pred.extend(y_pred.tolist())

        logger.info(f"    Fold {fold+1} acc={acc:.4f}  "
                    f"({time.time()-t_fold:.1f}s)")

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))
    logger.info(f"  CV accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    # Classification report (combined folds)
    if all_y_true:
        report = classification_report(
            all_y_true, all_y_pred,
            labels=list(range(n_classes)),
            target_names=class_list,
            zero_division=0,
        )
        logger.info(f"\n{report}")
    else:
        report = "N/A"

    return {
        "exp_name":        exp_name,
        "ansatz":          ansatz_name,
        "n_samples":       len(valid_sents),
        "n_classes":       n_classes,
        "n_s_qubits":      n_s_qubits,
        "fold_accuracies": fold_accs,
        "accuracy_mean":   mean_acc,
        "accuracy_std":    std_acc,
        "classification_report": report,
        "class_list":      class_list,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("  Arabic QNLP Experiment 5 — lambeq Training Loop")
    logger.info("=" * 70)

    # Load data
    data = load_sentences_json()
    experiments = build_experiments(data)
    logger.info(f"Loaded {len(experiments)} experiment sets.")
    for ex in experiments:
        logger.info(f"  {ex['name']}: {len(ex['sentences'])} sentences, "
                    f"{len(set(ex['labels']))} classes")

    all_results = []

    for ex in experiments:
        for ansatz_name, n_layers, _ in ANSATZ_CONFIGS:
            result = run_cv_experiment(
                exp_name    = ex["name"],
                sentences   = ex["sentences"],
                labels      = ex["labels"],
                ansatz_name = ansatz_name,
                n_layers    = n_layers,
            )
            all_results.append(result)

            # Save per-experiment result immediately
            out_dir = OUTPUT_BASE / ansatz_name / ex["name"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"results_{ex['name']}.json"

            # Make result JSON-serializable
            safe_result = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in result.items()}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(safe_result, f, ensure_ascii=False, indent=2)
            logger.info(f"  Saved → {out_path}")

    # Save master summary
    summary_path = OUTPUT_BASE / "run_summary_exp5.json"
    summary = {
        "total_experiments": len(all_results),
        "results": all_results,
        "settings": {
            "n_folds":   N_FOLDS,
            "epochs":    EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "SPSA",
            "spsa_hyperparams": SPSA_HYPERPARAMS,
            "ansatzes": ANSATZ_CONFIGS,
        }
    }
    # JSON-serialize
    def _safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_safe)
    logger.info(f"\nMaster summary saved → {summary_path}")

    # Print final table
    logger.info("\n" + "=" * 70)
    logger.info("  FINAL RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Experiment':<35} {'Ansatz':<20} {'Acc Mean':>10} {'Acc Std':>10}")
    logger.info("-" * 75)
    for r in all_results:
        logger.info(
            f"{r['exp_name']:<35} {r['ansatz']:<20} "
            f"{r.get('accuracy_mean', 0):.4f}     "
            f"{r.get('accuracy_std', 0):.4f}"
        )


if __name__ == "__main__":
    main()
