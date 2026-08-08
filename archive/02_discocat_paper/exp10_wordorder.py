# -*- coding: utf-8 -*-
"""
exp10_wordorder.py
------------------
Word Order classification — the theoretically ideal task for quantum
compositional models.

Key insight:
  - SVO:     n ⊗ (n.r⊗s⊗n.l) ⊗ n  →  distinct circuit topology
  - VSO:     (s⊗n.l⊗n.l) ⊗ n ⊗ n  →  different cup wiring + Swap
  - Nominal: n ⊗ (n.r⊗s)           →  two-box circuit

Classical AraVec averaged embeddings are ORDER-BLIND (mean is commutative),
so SVO and VSO produce IDENTICAL feature vectors → classical SVM cannot
distinguish them. Quantum circuits encode word order in their topology.

Augmented dataset: 40 per class × 3 classes = 120 sentences (up from 45).
5-fold CV: 96 train / 24 test.

Also runs the classical AraVec baseline alongside for direct comparison.

Usage:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 exp10_wordorder.py
"""

import os, sys, json, math, hashlib, logging, warnings, time
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("exp10_wordorder.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("exp10")

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
from lambeq.training import CrossEntropyLoss

_remove_cups = RemoveCupsRewriter()
from arabic_dep_reader import sentences_to_diagrams

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

# ── settings ──────────────────────────────────────────────────────────────────
N_FOLDS    = 5
EPOCHS     = 500
BATCH_SIZE = 8
SEEDS      = [42, 123, 777]

N = Ty('n')
S = Ty('s')

SPSA_HYPERPARAMS = {"a": 0.05, "c": 0.06, "A": 50, "alpha": 0.602, "gamma": 0.101}

OUTPUT_DIR = Path("qnlp_experiment_outputs_per_set_v2") / "exp10_wordorder"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANSATZES  = ["IQP", "Sim14"]
N_LAYERS_LIST = [1, 2]


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
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


def make_ansatz(ans_name: str, n_s_qubits: int, n_layers: int):
    ob = {S: n_s_qubits, N: 1}
    if ans_name == "IQP":
        return IQPAnsatz(ob, n_layers=n_layers, discard=False)
    return Sim14Ansatz(ob, n_layers=n_layers, discard=False)


def encode_labels(labels, class_list, n_s_qubits):
    n_flat = 2 ** n_s_qubits
    flat   = np.zeros((len(labels), n_flat))
    for i, lbl in enumerate(labels):
        flat[i, class_list.index(lbl)] = 1.0
    return flat.reshape((len(labels),) + tuple(2 for _ in range(n_s_qubits)))


def sentence_to_vec(sentence: str) -> np.ndarray:
    try:
        import stanza
        if not hasattr(sentence_to_vec, '_nlp'):
            sentence_to_vec._nlp = stanza.Pipeline("ar", processors="tokenize",
                                                     verbose=False, download_method=None)
        tokens = [w.text for sent in sentence_to_vec._nlp(sentence).sentences
                  for w in sent.words]
    except Exception:
        tokens = sentence.strip().split()
    vecs = []
    for tok in tokens:
        cands = [tok, tok[2:]] if tok.startswith("ال") else [tok]
        for c in cands:
            if ARAVEC_KV.has_index_for(c):
                vecs.append(ARAVEC_KV.get_vector(c)); break
    if not vecs:
        return np.zeros(ARAVEC_DIM)
    v = np.mean(vecs, axis=0)
    n = np.linalg.norm(v); return v / n if n > 0 else v


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSICAL BASELINE (run once, fast)
# ═══════════════════════════════════════════════════════════════════════════════

def run_classical(sentences, labels):
    logger.info("\n--- Classical AraVec Baseline ---")
    X  = np.stack([sentence_to_vec(s) for s in sentences])
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    classifiers = {
        "SVM_linear": SVC(kernel="linear", C=1.0, random_state=42),
        "SVM_rbf":    SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42),
        "RF":         RandomForestClassifier(n_estimators=200, random_state=42),
    }
    results = {}
    for cname, clf_proto in classifiers.items():
        fold_accs = []
        for tr, te in skf.split(X, y):
            clf = clone(clf_proto)
            clf.fit(X[tr], y[tr])
            fold_accs.append(accuracy_score(y[te], clf.predict(X[te])))
        mean = float(np.mean(fold_accs))
        std  = float(np.std(fold_accs))
        logger.info(f"  {cname:<14} {mean:.4f} ± {std:.4f}")
        results[cname] = {"mean": mean, "std": std, "folds": fold_accs}
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTUM CV
# ═══════════════════════════════════════════════════════════════════════════════

def run_quantum_config(sentences, labels, ans_name, n_layers, seed):
    class_list = sorted(set(labels))
    n_classes  = len(class_list)
    n_s_qubits = max(1, math.ceil(math.log2(max(n_classes, 2))))
    tag = f"WordOrder/{ans_name}/L{n_layers}/seed{seed}"

    logger.info(f"\n  {tag}  n={len(sentences)}  n_s_qubits={n_s_qubits}")

    ansatz   = make_ansatz(ans_name, n_s_qubits, n_layers)
    diagrams = sentences_to_diagrams(sentences, log_interval=20)

    circuits, valid_idx = [], []
    for i, d in enumerate(diagrams):
        try:
            circuits.append(ansatz(_remove_cups(d)))
            valid_idx.append(i)
        except Exception as e:
            logger.warning(f"  Circuit [{i}] failed: {e}")

    logger.info(f"  Circuits: {len(circuits)}/{len(sentences)}")
    if len(circuits) < n_classes * N_FOLDS:
        logger.error("  Too few circuits, skipping.")
        return {"tag": tag, "error": "insufficient circuits", "mean": 0.0, "std": 0.0}

    valid_labels = [labels[i] for i in valid_idx]
    labels_enc   = encode_labels(valid_labels, class_list, n_s_qubits)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_accs = []
    all_true, all_pred = [], []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(circuits, valid_labels)):
        t0 = time.time()
        train_c = [circuits[i] for i in tr_idx]
        test_c  = [circuits[i] for i in te_idx]
        train_y = labels_enc[tr_idx]
        test_y  = labels_enc[te_idx]

        try:
            model = NumpyModel.from_diagrams(circuits, use_jit=False)
        except Exception as e:
            logger.error(f"    NumpyModel failed: {e}")
            fold_accs.append(0.0); continue

        model.weights = warmstart_weights(model)
        loss_fn = CrossEntropyLoss()

        trainer = QuantumTrainer(
            model=model, loss_function=loss_fn, epochs=EPOCHS,
            optimizer=SPSAOptimizer, optim_hyperparams=SPSA_HYPERPARAMS,
            seed=seed + fold_i, verbose="suppress",
        )
        train_ds = Dataset(train_c, train_y, batch_size=BATCH_SIZE, shuffle=True)
        val_ds   = Dataset(test_c,  test_y,  batch_size=BATCH_SIZE, shuffle=False)

        try:
            trainer.fit(train_ds, val_ds, log_interval=200, eval_interval=200)
        except Exception as e:
            logger.error(f"    Training failed: {e}")
            fold_accs.append(0.0); continue

        try:
            preds   = model(test_c)
            y_pred  = np.argmax(preds.reshape(len(test_c), -1), axis=1).tolist()
            y_true  = np.argmax(test_y.reshape(len(test_c), -1), axis=1).tolist()
            acc = float(accuracy_score(y_true, y_pred))
        except Exception as e:
            logger.error(f"    Prediction failed: {e}")
            acc = 0.0; y_true = []; y_pred = []

        fold_accs.append(acc)
        all_true.extend(y_true)
        all_pred.extend(y_pred)
        logger.info(f"    fold {fold_i+1}/{N_FOLDS}  acc={acc:.4f}  ({time.time()-t0:.0f}s)")

    mean = float(np.mean(fold_accs))
    std  = float(np.std(fold_accs))
    logger.info(f"  → {tag}: {mean:.4f} ± {std:.4f}")

    report = ""
    if all_true and all_pred:
        try:
            report = classification_report(all_true, all_pred,
                                           labels=list(range(n_classes)),
                                           target_names=class_list, zero_division=0)
        except Exception:
            pass

    return {
        "tag": tag, "ansatz": ans_name, "n_layers": n_layers, "seed": seed,
        "n_samples": len(circuits), "n_classes": n_classes, "chance": 1/n_classes,
        "mean": mean, "std": std, "folds": fold_accs, "report": report,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("  Arabic QNLP — exp10: Word Order Classification (120 sentences)")
    logger.info("=" * 70)
    logger.info(f"  Epochs={EPOCHS}  Seeds={SEEDS}  n_layers={N_LAYERS_LIST}")
    logger.info("  Hypothesis: quantum circuit topology encodes word order;")
    logger.info("  classical averaged embeddings are order-blind (mean is commutative).")

    data      = json.load(open("sentences.json", encoding="utf-8"))
    wo        = data.get("WordOrder", [])
    sentences = [d["sentence"] for d in wo]
    labels    = [d["label"] for d in wo]

    from collections import Counter
    c = Counter(labels)
    logger.info(f"  {dict(c)}  total={len(sentences)}")

    # ── Classical baseline (quick) ────────────────────────────────────────────
    cl_results = run_classical(sentences, labels)
    with open(OUTPUT_DIR / "classical_baseline.json", "w", encoding="utf-8") as f:
        json.dump(cl_results, f, ensure_ascii=False, indent=2)

    # ── Quantum configs ───────────────────────────────────────────────────────
    configs = [(ans, nl, seed)
               for ans in ANSATZES
               for nl in N_LAYERS_LIST
               for seed in SEEDS]
    total = len(configs)
    logger.info(f"\n  Quantum configs: {total}")

    all_q_results = []
    for ci, (ans, nl, seed) in enumerate(configs):
        logger.info(f"\n[{ci+1}/{total}]")
        r = run_quantum_config(sentences, labels, ans, nl, seed)
        all_q_results.append(r)
        fname = OUTPUT_DIR / f"results_{ans}_L{nl}_s{seed}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  WORD ORDER RESULTS SUMMARY")
    logger.info("=" * 70)

    logger.info("\n-- Classical (order-blind) --")
    for cname, cr in cl_results.items():
        logger.info(f"  {cname:<14} {cr['mean']:.4f} ± {cr['std']:.4f}")

    logger.info("\n-- Quantum (order-aware topology) -- avg across seeds --")
    grouped = defaultdict(list)
    for r in all_q_results:
        if "error" not in r:
            grouped[(r["ansatz"], r["n_layers"])].append(r["mean"])

    best_q = 0.0
    for (ans, nl) in sorted(grouped.keys()):
        vals = grouped[(ans, nl)]
        mu   = float(np.mean(vals))
        sd   = float(np.std(vals))
        best_q = max(best_q, mu)
        logger.info(f"  {ans} L={nl}  {mu:.4f} ± {sd:.4f}  seeds={[round(v,3) for v in vals]}")

    cl_best = max(cr["mean"] for cr in cl_results.values())
    logger.info(f"\n  Chance: 33.3%")
    logger.info(f"  Classical best: {cl_best:.1%}")
    logger.info(f"  Quantum best:   {best_q:.1%}")
    delta = best_q - cl_best
    logger.info(f"  Quantum vs classical: {delta:+.1%}  {'← quantum wins!' if delta > 0.02 else ''}")

    summary = {"classical": cl_results, "quantum": all_q_results,
               "chance": 1/3, "cl_best": cl_best, "q_best": best_q}
    with open(OUTPUT_DIR / "exp10_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved → {OUTPUT_DIR}/exp10_summary.json")


if __name__ == "__main__":
    main()
