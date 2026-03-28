# -*- coding: utf-8 -*-
"""
exp12_quantum_advantage.py
--------------------------
Three targeted demonstrations of quantum advantage in Arabic DisCoCat.

The key insight shared by all three scenarios:
  Quantum circuit TOPOLOGY encodes grammatical structure.
  Averaged word embeddings are STRUCTURE-BLIND (the mean is commutative).
  Quantum circuits are STRUCTURE-SENSITIVE (cup wiring, Swap gates matter).

SCENARIO 1 — Morphology / Tense  (reference result, no new training)
    Past-tense كتب → Stanza parses as NOUN → nominal circuit (n @ n.r⊗s → s)
    Present-tense يكتب → Stanza parses as VERB → verbal circuit (n @ s⊗n.l⊗n.l → s)
    Result (exp5): quantum 75.3%  vs classical 54.7%  → quantum wins by +20.6pp

SCENARIO 2 — Word Order / SVO vs VSO  (controlled matched pairs)
    Dataset: 40 SVO + 40 VSO sentences — SAME WORDS, DIFFERENT ORDER.
    Classical averaged AraVec:  SVO_vec == VSO_vec  → SVM accuracy ≈ 50% (chance).
    Quantum circuit:  SVO has Cup(N,N.r)@Id(S)@Cup(N.l,N);
                      VSO has Swap(N.l,N) then different cups — topologically distinct.
    Method A  QFM  : warm-start circuit outputs as features → SVM (no SPSA training).
    Method B  SPSA : full QuantumTrainer end-to-end.

SCENARIO 3 — Lexical Ambiguity / King vs Possess  (structural disambiguation)
    "ملك" as King → nominal/copular syntax: كان الملك عادلاً
    "ملك" as Possess → transitive verbal syntax: ملك الفلاح الأرض
    Classical SVM: has same root word in both classes, limited context separation.
    Quantum QFM:  nominal circuit topology ≠ VSO verbal topology
                  → different probability distributions → SVM separates.

Quantum Feature Map (QFM):
    1. Build circuits using AraVec warm-start parameters (FIXED — no SPSA).
    2. NumpyModel.forward() gives probability distributions per circuit.
    3. Use these as feature vectors for a classical SVM.
    Advantage: isolates topological contribution; runs in seconds, not hours.

Usage:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 exp12_quantum_advantage.py
"""

import os, sys, json, math, hashlib, logging, warnings, time
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("exp12_quantum_advantage.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("exp12")

# ── AraVec ────────────────────────────────────────────────────────────────────
try:
    from gensim.models import Word2Vec
    _kv = Word2Vec.load("aravec/full_uni_cbow_300_twitter.mdl").wv
    logger.info(f"AraVec: {len(_kv)} vectors × {_kv.vector_size}d")
    ARAVEC_KV = _kv
    ARAVEC_DIM = _kv.vector_size
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
from arabic_dep_reader import sentences_to_diagrams

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# ── settings ──────────────────────────────────────────────────────────────────
N_FOLDS    = 5
SEED       = 42
EPOCHS     = 500       # for SPSA scenarios
BATCH_SIZE = 8
DATA_FILE  = "sentences.json"

OUTPUT_DIR = Path("qnlp_experiment_outputs_per_set_v2") / "exp12_quantum_advantage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N = Ty('n')
S = Ty('s')

SPSA_PARAMS = {"a": 0.05, "c": 0.06, "A": 50, "alpha": 0.602, "gamma": 0.101}


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _vec_for_word(word: str) -> Optional[np.ndarray]:
    base = word
    for sep in ('__SA', '__SB', '_ASP', '_PER', '_NUM', '_GEN'):
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


def sentence_to_vec(sentence: str) -> np.ndarray:
    """Classical AraVec averaged embedding (bag-of-words, order-blind)."""
    try:
        import stanza
        if not hasattr(sentence_to_vec, '_nlp'):
            sentence_to_vec._nlp = stanza.Pipeline(
                "ar", processors="tokenize", verbose=False, download_method=None)
        tokens = [w.text for sent in sentence_to_vec._nlp(sentence).sentences
                  for w in sent.words]
    except Exception:
        tokens = sentence.strip().split()
    vecs = []
    for tok in tokens:
        cands = [tok, tok[2:]] if tok.startswith("ال") else [tok]
        for c in cands:
            if ARAVEC_KV.has_index_for(c):
                vecs.append(ARAVEC_KV.get_vector(c))
                break
    if not vecs:
        return np.zeros(ARAVEC_DIM)
    v = np.mean(vecs, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_circuits_and_idx(sentences: List[str], ansatz) -> Tuple[list, List[int]]:
    """Build circuits from sentences, return (circuits, valid_indices)."""
    diagrams = sentences_to_diagrams(sentences, log_interval=0)
    circuits, valid_idx = [], []
    for i, d in enumerate(diagrams):
        try:
            circuits.append(ansatz(_remove_cups(d)))
            valid_idx.append(i)
        except Exception as e:
            logger.warning(f"  Circuit [{i}] failed: {e}")
    return circuits, valid_idx


def encode_labels(labels: List[str], class_list: List[str],
                   n_s_qubits: int) -> np.ndarray:
    n_flat = 2 ** n_s_qubits
    flat   = np.zeros((len(labels), n_flat))
    for i, lbl in enumerate(labels):
        flat[i, class_list.index(lbl)] = 1.0
    return flat.reshape((len(labels),) + tuple(2 for _ in range(n_s_qubits)))


# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTUM FEATURE MAP (QFM)
# ═══════════════════════════════════════════════════════════════════════════════

def quantum_feature_map(
    circuits: list,
    model:    NumpyModel,
) -> np.ndarray:
    """
    Run circuits through the (warm-started, untrained) model and return
    probability distributions as feature vectors.

    Shape: (n_circuits, 2^n_s_qubits)
    """
    preds = model(circuits)
    return preds.reshape(len(circuits), -1)


def run_qfm_cv(
    circuits:    list,
    labels:      List[str],
    class_list:  List[str],
    n_s_qubits:  int,
    tag:         str,
) -> Dict:
    """
    Cross-validate a classical SVM on quantum feature-map outputs.
    No SPSA training — warm-start parameters only.
    """
    logger.info(f"\n  QFM cross-validation: {tag}  n={len(circuits)}")
    n = len(circuits)

    model = NumpyModel.from_diagrams(circuits, use_jit=False)
    model.weights = warmstart_weights(model)

    Q = quantum_feature_map(circuits, model)  # (n, 2^n_s_qubits)

    y = np.array([class_list.index(lbl) for lbl in labels])

    min_count = min(Counter(labels).values())
    folds     = min(N_FOLDS, min_count)
    skf       = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    fold_accs = []
    all_true, all_pred = [], []
    for fold_i, (tr, te) in enumerate(skf.split(Q, y)):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale", random_state=SEED)),
        ])
        pipe.fit(Q[tr], y[tr])
        yp = pipe.predict(Q[te])
        fold_accs.append(float(accuracy_score(y[te], yp)))
        all_true.extend(y[te].tolist())
        all_pred.extend(yp.tolist())

    mean = float(np.mean(fold_accs))
    std  = float(np.std(fold_accs))
    logger.info(f"  QFM {tag}: {mean:.4f} ± {std:.4f}  (chance {1/len(class_list):.1%})")

    report = ""
    try:
        report = classification_report(
            all_true, all_pred, labels=list(range(len(class_list))),
            target_names=class_list, zero_division=0)
    except Exception:
        pass

    return {"tag": tag, "mean": mean, "std": std, "folds": fold_accs, "report": report}


def run_classical_cv(
    sentences:   List[str],
    labels:      List[str],
    class_list:  List[str],
    tag:         str,
) -> Dict:
    """Classical AraVec bag-of-words + SVM baseline (order-blind)."""
    logger.info(f"\n  Classical AraVec baseline: {tag}")
    X = np.stack([sentence_to_vec(s) for s in sentences])
    y = np.array([class_list.index(lbl) for lbl in labels])

    min_count = min(Counter(labels).values())
    folds     = min(N_FOLDS, min_count)
    skf       = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    fold_accs = []
    for _, (tr, te) in enumerate(skf.split(X, y)):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale", random_state=SEED)),
        ])
        pipe.fit(X[tr], y[tr])
        fold_accs.append(float(accuracy_score(y[te], pipe.predict(X[te]))))

    mean = float(np.mean(fold_accs))
    std  = float(np.std(fold_accs))
    logger.info(f"  Classical {tag}: {mean:.4f} ± {std:.4f}  (chance {1/len(class_list):.1%})")
    return {"tag": tag, "mean": mean, "std": std, "folds": fold_accs}


def run_spsa_cv(
    circuits:    list,
    labels:      List[str],
    class_list:  List[str],
    n_s_qubits:  int,
    tag:         str,
) -> Dict:
    """Full SPSA-trained QuantumTrainer CV."""
    logger.info(f"\n  SPSA training: {tag}  n={len(circuits)}")
    n_classes = len(class_list)
    is_binary = (n_classes == 2)
    loss_fn   = BinaryCrossEntropyLoss() if is_binary else CrossEntropyLoss()

    min_count = min(Counter(labels).values())
    folds     = min(N_FOLDS, min_count)
    skf       = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    fold_accs = []
    all_true, all_pred = [], []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(circuits, labels)):
        t0 = time.time()
        train_c = [circuits[i] for i in tr_idx]
        test_c  = [circuits[i] for i in te_idx]
        train_l = [labels[i]   for i in tr_idx]
        test_l  = [labels[i]   for i in te_idx]

        try:
            model = NumpyModel.from_diagrams(circuits, use_jit=False)
        except Exception as e:
            logger.error(f"    NumpyModel failed: {e}")
            fold_accs.append(0.0); continue

        model.weights = warmstart_weights(model)
        train_y = encode_labels(train_l, class_list, n_s_qubits)
        test_y  = encode_labels(test_l,  class_list, n_s_qubits)

        trainer = QuantumTrainer(
            model=model, loss_function=loss_fn, epochs=EPOCHS,
            optimizer=SPSAOptimizer, optim_hyperparams=SPSA_PARAMS,
            seed=SEED + fold_i, verbose="suppress",
        )
        train_ds = Dataset(train_c, train_y, batch_size=BATCH_SIZE, shuffle=True)
        val_ds   = Dataset(test_c,  test_y,  batch_size=BATCH_SIZE, shuffle=False)

        try:
            trainer.fit(train_ds, val_ds, log_interval=200, eval_interval=200)
        except Exception as e:
            logger.error(f"    Training failed: {e}")
            fold_accs.append(0.0); continue

        try:
            preds = model(test_c).reshape(len(test_c), -1)
            targs = test_y.reshape(len(test_c), -1)
            yp = np.argmax(preds, axis=1).tolist()
            yt = np.argmax(targs, axis=1).tolist()
            acc = float(accuracy_score(yt, yp))
        except Exception as e:
            logger.error(f"    Predict failed: {e}")
            acc = 0.0; yt = []; yp = []

        fold_accs.append(acc)
        all_true.extend(yt)
        all_pred.extend(yp)
        logger.info(f"    fold {fold_i+1}/{folds} acc={acc:.4f} ({time.time()-t0:.0f}s)")

    mean = float(np.mean(fold_accs))
    std  = float(np.std(fold_accs))
    logger.info(f"  SPSA {tag}: {mean:.4f} ± {std:.4f}")
    return {"tag": tag, "mean": mean, "std": std, "folds": fold_accs}


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 2: WORD ORDER (SVO vs VSO binary — matched pairs)
# ═══════════════════════════════════════════════════════════════════════════════

def run_scenario2_wordorder(data: Dict) -> Dict:
    logger.info("\n" + "=" * 70)
    logger.info("  SCENARIO 2: Word Order — SVO vs VSO (Controlled Matched Pairs)")
    logger.info("=" * 70)
    logger.info("  Hypothesis: classical AraVec ≈ 50% (identical word bags);")
    logger.info("              quantum circuit topology (cup wiring) > 50%.")

    wo = data.get("WordOrder", [])
    svo = [x for x in wo if x["label"] == "WordOrder_SVO"]
    vso = [x for x in wo if x["label"] == "WordOrder_VSO"]

    # Use only SVO vs VSO (matched pairs, same words)
    sentences = [x["sentence"] for x in svo] + [x["sentence"] for x in vso]
    labels    = ["SVO"] * len(svo) + ["VSO"] * len(vso)
    class_list = ["SVO", "VSO"]
    n_s_qubits = 1   # binary

    logger.info(f"  n={len(sentences)}  ({len(svo)} SVO + {len(vso)} VSO)")
    logger.info("  Note: All pairs use same Arabic words in different order.")
    logger.info("  → AraVec mean(SVO sentence) == AraVec mean(VSO sentence)")

    results = {}

    # Classical baseline (expected ≈ 50%)
    results["classical"] = run_classical_cv(sentences, labels, class_list,
                                             "WordOrder_SVO-VSO_Classical")

    # Build quantum circuits
    ansatz = IQPAnsatz({S: n_s_qubits, N: 1}, n_layers=1, discard=False)
    circuits, valid_idx = build_circuits_and_idx(sentences, ansatz)
    valid_sents  = [sentences[i] for i in valid_idx]
    valid_labels = [labels[i]    for i in valid_idx]
    logger.info(f"  Circuits built: {len(circuits)}/{len(sentences)}")

    # QFM: warm-start only, no SPSA — isolates TOPOLOGICAL contribution
    results["qfm_iqp_l1"] = run_qfm_cv(
        circuits, valid_labels, class_list, n_s_qubits,
        "WordOrder_SVO-VSO_QFM_IQP_L1")

    # QFM with L2 (richer topology)
    ansatz_l2 = IQPAnsatz({S: n_s_qubits, N: 1}, n_layers=2, discard=False)
    circ_l2, vi2 = build_circuits_and_idx(sentences, ansatz_l2)
    valid_l2 = [labels[i] for i in vi2]
    results["qfm_iqp_l2"] = run_qfm_cv(
        circ_l2, valid_l2, class_list, n_s_qubits,
        "WordOrder_SVO-VSO_QFM_IQP_L2")

    # SPSA trained (for comparison with QFM)
    results["spsa_iqp_l1"] = run_spsa_cv(
        circuits, valid_labels, class_list, n_s_qubits,
        "WordOrder_SVO-VSO_SPSA_IQP_L1")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 3: LEXICAL AMBIGUITY — Structural Pairs (King/Possess)
# ═══════════════════════════════════════════════════════════════════════════════

def run_scenario3_lexico_structural(data: Dict) -> Dict:
    logger.info("\n" + "=" * 70)
    logger.info("  SCENARIO 3: Lexical Ambiguity — Structural Disambiguation")
    logger.info("=" * 70)
    logger.info("  King (الملك): nominal/copular syntax → nominal circuit topology")
    logger.info("  Possess (ملك): transitive verbal syntax → VSO circuit topology")
    logger.info("  Hypothesis: circuit topology encodes sense through syntax.")

    la = data.get("LexicalAmbiguity", [])
    la_by_label = defaultdict(list)
    for d in la:
        la_by_label[d["label"]].append(d["sentence"])

    # Structural pairs: King/Possess (syntactically most distinct)
    structural_pairs = [
        ("Ambiguity_King",   "Ambiguity_Possess",  "King_Possess"),
        ("Ambiguity_Hit",    "Ambiguity_Multiply",  "Hit_Multiply"),  # verb: physical vs math
        ("Ambiguity_Open",   "Ambiguity_Conquer",   "Open_Conquer"),  # same verb, diff obj type
    ]

    all_results = {}
    for cls_a, cls_b, pair_tag in structural_pairs:
        sents_a   = la_by_label.get(cls_a, [])
        sents_b   = la_by_label.get(cls_b, [])
        sentences = sents_a + sents_b
        labels    = [cls_a] * len(sents_a) + [cls_b] * len(sents_b)
        class_list = sorted(set(labels))

        logger.info(f"\n  --- {pair_tag}: {len(sents_a)} {cls_a} + {len(sents_b)} {cls_b} ---")

        pair_results = {}

        # Classical baseline
        pair_results["classical"] = run_classical_cv(
            sentences, labels, class_list, f"{pair_tag}_Classical")

        # QFM
        ansatz = IQPAnsatz({S: 1, N: 1}, n_layers=1, discard=False)
        circuits, valid_idx = build_circuits_and_idx(sentences, ansatz)
        valid_labels = [labels[i] for i in valid_idx]
        logger.info(f"  Circuits: {len(circuits)}/{len(sentences)}")

        pair_results["qfm_iqp_l1"] = run_qfm_cv(
            circuits, valid_labels, class_list, 1, f"{pair_tag}_QFM_IQP_L1")

        # SPSA (for comparison)
        pair_results["spsa_iqp_l1"] = run_spsa_cv(
            circuits, valid_labels, class_list, 1, f"{pair_tag}_SPSA_IQP_L1")

        all_results[pair_tag] = pair_results

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("  Arabic QNLP — exp12: Three Quantum Advantage Scenarios")
    logger.info("=" * 70)
    logger.info("  Mechanism: circuit TOPOLOGY encodes structure that AraVec CANNOT.")
    logger.info(f"  Data: {DATA_FILE}  |  SPSA Epochs: {EPOCHS}")

    data = json.load(open(DATA_FILE, encoding="utf-8"))

    # ── Scenario 1: Tense (reference) ─────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  SCENARIO 1: Morphology / Tense  (reference — from exp5/exp9)")
    logger.info("=" * 70)
    logger.info("  Past-tense  كتب  → Stanza: NOUN → nominal circuit (n@n.r⊗s → s)")
    logger.info("  Pres-tense  يكتب → Stanza: VERB → verbal circuit  (n@s⊗n.l → s)")
    logger.info("  Result (exp5):  Quantum 75.3%  vs  Classical 54.7%  → +20.6pp")
    logger.info("  (Sim14 L2, exp9):  64.0% averaged over 3 seeds (converging)")
    logger.info("  Advantage source: structural parse difference → circuit topology.")
    s1_result = {
        "quantum_exp5":  0.753,
        "classical_best": 0.547,
        "delta": 0.206,
        "source": "exp5_trained / exp9_tense_deep",
    }

    # ── Scenario 2: Word Order ─────────────────────────────────────────────────
    s2_results = run_scenario2_wordorder(data)

    # ── Scenario 3: Lexical Structural Disambiguation ─────────────────────────
    s3_results = run_scenario3_lexico_structural(data)

    # ── Final summary table ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("  QUANTUM ADVANTAGE — FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\n{'Scenario':<38} {'Classical':>10} {'QFM':>10} {'SPSA':>10} {'Advantage':>12}")
    logger.info("-" * 80)

    # S1
    logger.info(
        f"{'S1: Tense (nominal vs verbal parse)':<38} "
        f"{s1_result['classical_best']:>10.1%} "
        f"{'N/A':>10} "
        f"{s1_result['quantum_exp5']:>10.1%} "
        f"{s1_result['delta']:>+12.1%}")

    # S2
    s2_cl = s2_results.get("classical", {}).get("mean", 0)
    s2_qfm = s2_results.get("qfm_iqp_l1", {}).get("mean", 0)
    s2_sp  = s2_results.get("spsa_iqp_l1", {}).get("mean", 0)
    s2_best_q = max(s2_qfm, s2_sp)
    logger.info(
        f"{'S2: Word Order (SVO/VSO same words)':<38} "
        f"{s2_cl:>10.1%} "
        f"{s2_qfm:>10.1%} "
        f"{s2_sp:>10.1%} "
        f"{s2_best_q - s2_cl:>+12.1%}")

    # S3
    for pair_tag, pr in s3_results.items():
        s3_cl  = pr.get("classical",   {}).get("mean", 0)
        s3_qfm = pr.get("qfm_iqp_l1", {}).get("mean", 0)
        s3_sp  = pr.get("spsa_iqp_l1",{}).get("mean", 0)
        s3_best = max(s3_qfm, s3_sp)
        logger.info(
            f"{'S3: ' + pair_tag + ' (structural sense)':<38} "
            f"{s3_cl:>10.1%} "
            f"{s3_qfm:>10.1%} "
            f"{s3_sp:>10.1%} "
            f"{s3_best - s3_cl:>+12.1%}")

    logger.info("\n  Key: QFM = Quantum Feature Map (no SPSA training)")
    logger.info("  Quantum advantage = circuit topology encodes structure")
    logger.info("  Classical baseline = AraVec bag-of-words (structure-blind)")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "scenario1_tense": s1_result,
        "scenario2_wordorder": s2_results,
        "scenario3_lexico_structural": s3_results,
    }
    spath = OUTPUT_DIR / "exp12_summary.json"
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved → {spath}")


if __name__ == "__main__":
    main()
