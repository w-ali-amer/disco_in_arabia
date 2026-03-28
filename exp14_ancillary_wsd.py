# -*- coding: utf-8 -*-
"""
exp14_ancillary_wsd.py  (v2 — vocabulary-controlled dataset)
-------------------------------------------------------------
Ancillary-qubit Word-Sense Disambiguation experiment.

Dataset:  WordSenseDisambiguation_v2 from sentences.json
          4 verbs × 2 senses × 25 sentences = 200 total
          VOCABULARY-CONTROLLED structural disambiguation:
            رفع  — 8 exact matched pairs (same string, different label)
                    + polysemous objects (الملف/الورقة/التقرير in both classes)
            حمل  — shared objects (الرسالة/الفكرة/الخبر in both classes)
                    subject type (animate vs inanimate semiotic) is the signal
            قطع  — shared 13-subject pool across cut/sever
            ضرب  — shared 14-subject pool across strike/exemplify

Methods compared (per-verb binary, then pooled):
  1. AraVec_SVM      — averaged embeddings + SVM
  2. AraVec_RF       — averaged embeddings + RF
  3. AraBERT_frozen  — frozen CLS vector + SVM
  4. QFM_base        — IQP L1, discard=False, n_ancillas=0 (warm-start, no SPSA)
  5. QFM_ancilla     — IQP L1, discard=True,  n_ancillas=1
  6. SPSA_base       — IQP L1, discard=False, n_ancillas=0 (trained)
  7. SPSA_ancilla    — IQP L1, discard=True,  n_ancillas=1 (trained)

SPSA_ancilla fix (v1 bug: shape mismatch):
  With discard=True + n_ancillas=1, NumpyModel output is (batch, 2, 2) —
  a batch of 2×2 density matrices for the 1-qubit sentence system.
  Labels must be encoded as 2×2 pure-state density matrices:
    class 0: [[1,0],[0,0]]   class 1: [[0,0],[0,1]]
  Prediction: argmax of density-matrix diagonal.

Theory (Coecke et al. 2020, arXiv:2012.03755):
  The ancillary qubit entangles with the meaning wire during the verb box.
  When traced out (Discard), it leaves a mixed state whose density matrix
  encodes a superposition of sense readings. The object noun's circuit
  (shaped by selectional fit) collapses the mixture toward the correct sense.

CV: 5-fold × 3-repeat = 15 splits  (RepeatedStratifiedKFold)
QFM: 10 seeds averaged.  SPSA: 5 seeds × 15 folds.

Usage:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 exp14_ancillary_wsd.py
"""

import sys, json, math, hashlib, logging, warnings, time
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
        logging.FileHandler("exp14_ancillary_wsd_v2.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("exp14")

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

# ── AraBERT ───────────────────────────────────────────────────────────────────
ARABERT_NAME = "aubmindlab/bert-base-arabertv02"
_arabert_tokenizer = None
_arabert_model     = None
_device            = None

def _load_arabert():
    global _arabert_tokenizer, _arabert_model, _device
    if _arabert_tokenizer is not None:
        return
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"AraBERT device: {_device}")
        _arabert_tokenizer = AutoTokenizer.from_pretrained(ARABERT_NAME)
        _arabert_model = AutoModel.from_pretrained(ARABERT_NAME).to(_device)
        _arabert_model.eval()
        logger.info("AraBERT loaded (frozen CLS mode)")
    except Exception as e:
        logger.warning(f"AraBERT unavailable: {e}")


def get_arabert_cls(texts):
    _load_arabert()
    if _arabert_tokenizer is None:
        return np.zeros((len(texts), 768))
    import torch
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = texts[i:i+16]
            enc = _arabert_tokenizer(batch, return_tensors="pt",
                                     padding=True, truncation=True,
                                     max_length=64).to(_device)
            out = _arabert_model(**enc)
            vecs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(vecs)


# ── lambeq ────────────────────────────────────────────────────────────────────
from lambeq import (
    IQPAnsatz, RemoveCupsRewriter,
    NumpyModel, QuantumTrainer, SPSAOptimizer, Dataset,
)
from lambeq.backend.grammar import Ty
from lambeq.training import CrossEntropyLoss

_remove_cups = RemoveCupsRewriter()
from arabic_dep_reader import sentences_to_diagrams

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

# ── settings ──────────────────────────────────────────────────────────────────
N_SPLITS   = 5
N_REPEATS  = 3
N_FOLDS    = N_SPLITS * N_REPEATS   # 15
EPOCHS     = 500
BATCH_SIZE = 8
QFM_SEEDS  = list(range(10))
SPSA_SEEDS = [42, 123, 777, 7, 99]
N_LAYERS   = 1

N = Ty('n')
S = Ty('s')

SPSA_HYPERPARAMS = {"a": 0.05, "c": 0.06, "A": 50, "alpha": 0.602, "gamma": 0.101}

OUTPUT_DIR = Path("qnlp_experiment_outputs_per_set_v2") / "exp14_ancillary_wsd_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_KEY = "WordSenseDisambiguation_v2"


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def sentence_to_aravec(sentence):
    try:
        import stanza
        if not hasattr(sentence_to_aravec, '_nlp'):
            sentence_to_aravec._nlp = stanza.Pipeline(
                "ar", processors="tokenize", verbose=False, download_method=None)
        tokens = [w.text for sent in sentence_to_aravec._nlp(sentence).sentences
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
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _vec_for_word(word):
    base = word
    for sep in ('_ASP', '_PER', '_NUM', '_GEN'):
        if sep in word:
            base = word.split(sep)[0]; break
    candidates = list(dict.fromkeys([word, base]))
    if base.startswith("ال"):
        candidates.append(base[2:])
    for c in candidates:
        if ARAVEC_KV.has_index_for(c):
            return ARAVEC_KV.get_vector(c)
    return None


def warmstart_weights(model):
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


def build_circuits(sentences, n_ancillas=0, discard=False, n_s_qubits=1):
    ansatz = IQPAnsatz({S: n_s_qubits, N: 1}, n_layers=N_LAYERS,
                       discard=discard, n_ancillas=n_ancillas)
    diagrams = sentences_to_diagrams(sentences, log_interval=30)
    circuits, valid_idx = [], []
    for i, d in enumerate(diagrams):
        try:
            circuits.append(ansatz(_remove_cups(d)))
            valid_idx.append(i)
        except Exception as exc:
            logger.warning(f"  Circuit [{i}] skipped: {exc}")
    return circuits, valid_idx


def encode_labels_base(labels, class_list, n_s_qubits=1):
    """Standard one-hot encoding for discard=False models."""
    n_flat = 2 ** n_s_qubits
    flat   = np.zeros((len(labels), n_flat))
    for i, lbl in enumerate(labels):
        flat[i, class_list.index(lbl)] = 1.0
    return flat.reshape((len(labels),) + tuple(2 for _ in range(n_s_qubits)))


def encode_labels_ancilla(labels, class_list):
    """
    Density-matrix encoding for discard=True + n_ancillas=1 models.
    NumpyModel output shape: (batch, 2, 2) — 2×2 density matrices.
    Pure state |0⟩: dm = [[1,0],[0,0]]
    Pure state |1⟩: dm = [[0,0],[0,1]]
    """
    dm = np.zeros((len(labels), 2, 2))
    for i, lbl in enumerate(labels):
        dm[i, class_list.index(lbl), class_list.index(lbl)] = 1.0
    return dm


def predict_from_output(preds, n_circuits, discard):
    """Extract class predictions from model output regardless of shape."""
    if discard:
        # preds shape: (batch, 2, 2) density matrices — take diagonal
        flat = np.diagonal(preds.reshape(n_circuits, 2, 2), axis1=-2, axis2=-1)
    else:
        flat = preds.reshape(n_circuits, -1)
    return np.argmax(flat, axis=1).tolist()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSICAL BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

def run_classical(sentences, labels, tag=""):
    X  = np.stack([sentence_to_aravec(s) for s in sentences])
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=42)
    classifiers = {
        "AraVec_SVM_linear": SVC(kernel="linear", C=1.0, random_state=42),
        "AraVec_SVM_rbf":    SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42),
        "AraVec_RF":         RandomForestClassifier(n_estimators=200, random_state=42),
    }
    results = {}
    for cname, clf_proto in classifiers.items():
        fold_accs = []
        for tr, te in rskf.split(X, y):
            clf = clone(clf_proto)
            clf.fit(X[tr], y[tr])
            fold_accs.append(accuracy_score(y[te], clf.predict(X[te])))
        mean = float(np.mean(fold_accs))
        std  = float(np.std(fold_accs))
        logger.info(f"  [{tag}] {cname:<22} {mean:.4f} ± {std:.4f}")
        results[cname] = {"mean": mean, "std": std, "folds": fold_accs}
    return results


def run_arabert(sentences, labels, tag=""):
    logger.info(f"  [{tag}] Building AraBERT CLS vectors...")
    X  = get_arabert_cls(sentences)
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=42)
    fold_accs = []
    for tr, te in rskf.split(X, y):
        clf = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
        clf.fit(X[tr], y[tr])
        fold_accs.append(accuracy_score(y[te], clf.predict(X[te])))
    mean = float(np.mean(fold_accs))
    std  = float(np.std(fold_accs))
    logger.info(f"  [{tag}] AraBERT_frozen             {mean:.4f} ± {std:.4f}")
    return {"mean": mean, "std": std, "folds": fold_accs}


# ═══════════════════════════════════════════════════════════════════════════════
#  QFM (no SPSA — warm-start circuits as feature extractors → SVM)
# ═══════════════════════════════════════════════════════════════════════════════

def run_qfm(sentences, labels, n_ancillas, discard,
            circuits_cache=None, valid_labels_cache=None, tag="QFM"):
    class_list = sorted(set(labels))
    if circuits_cache is not None:
        circuits, valid_labels = circuits_cache, valid_labels_cache
    else:
        circuits, valid_idx = build_circuits(sentences, n_ancillas=n_ancillas,
                                             discard=discard, n_s_qubits=1)
        valid_labels = [labels[i] for i in valid_idx]

    if len(circuits) < 4:
        logger.warning(f"  [{tag}] Too few circuits, skipping.")
        return {"mean": 0.0, "std": 0.0}

    le = LabelEncoder()
    y  = le.fit_transform(valid_labels)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=42)

    all_seed_means = []
    for seed in QFM_SEEDS:
        try:
            model = NumpyModel.from_diagrams(circuits, use_jit=False)
        except Exception as exc:
            logger.error(f"  [{tag}] NumpyModel failed: {exc}"); continue
        model.weights = warmstart_weights(model)
        rng = np.random.default_rng(seed)
        model.weights += rng.normal(0, 0.05, size=len(model.weights))

        try:
            raw = model(circuits)
            # Flatten appropriately — density matrix or ket
            if discard:
                X = np.diagonal(raw.reshape(len(circuits), 2, 2),
                                axis1=-2, axis2=-1)
            else:
                X = raw.reshape(len(circuits), -1)
        except Exception as exc:
            logger.error(f"  [{tag}] Forward pass failed: {exc}"); continue

        fold_accs = []
        for tr, te in rskf.split(X, y):
            clf = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed)
            clf.fit(X[tr], y[tr])
            fold_accs.append(accuracy_score(y[te], clf.predict(X[te])))
        all_seed_means.append(float(np.mean(fold_accs)))

    if not all_seed_means:
        return {"mean": 0.0, "std": 0.0}
    mean = float(np.mean(all_seed_means))
    std  = float(np.std(all_seed_means))
    logger.info(f"  [{tag}] QFM n_anc={n_ancillas} discard={discard}  {mean:.4f} ± {std:.4f}")
    return {"mean": mean, "std": std, "seed_means": all_seed_means,
            "n_circuits": len(circuits)}


# ═══════════════════════════════════════════════════════════════════════════════
#  SPSA (trained)
# ═══════════════════════════════════════════════════════════════════════════════

def run_spsa(sentences, labels, n_ancillas, discard,
             circuits_cache=None, valid_labels_cache=None, tag="SPSA"):
    class_list = sorted(set(labels))
    if circuits_cache is not None:
        circuits, valid_labels = circuits_cache, valid_labels_cache
    else:
        circuits, valid_idx = build_circuits(sentences, n_ancillas=n_ancillas,
                                             discard=discard, n_s_qubits=1)
        valid_labels = [labels[i] for i in valid_idx]

    if len(circuits) < 4:
        logger.warning(f"  [{tag}] Too few circuits, skipping.")
        return {"mean": 0.0, "std": 0.0}

    # Label encoding — density matrix for ancilla, one-hot for base
    if discard:
        labels_enc = encode_labels_ancilla(valid_labels, class_list)
    else:
        labels_enc = encode_labels_base(valid_labels, class_list, n_s_qubits=1)

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=42)
    all_accs = []
    for seed in SPSA_SEEDS:
        fold_accs = []
        for fold_i, (tr_idx, te_idx) in enumerate(rskf.split(circuits, valid_labels)):
            t0 = time.time()
            train_c = [circuits[i] for i in tr_idx]
            test_c  = [circuits[i] for i in te_idx]
            train_y = labels_enc[tr_idx]
            test_y  = labels_enc[te_idx]

            try:
                model = NumpyModel.from_diagrams(circuits, use_jit=False)
            except Exception as exc:
                logger.error(f"  [{tag}] NumpyModel failed: {exc}")
                fold_accs.append(0.0); continue

            model.weights = warmstart_weights(model)
            trainer = QuantumTrainer(
                model=model, loss_function=CrossEntropyLoss(), epochs=EPOCHS,
                optimizer=SPSAOptimizer, optim_hyperparams=SPSA_HYPERPARAMS,
                seed=seed + fold_i, verbose="suppress",
            )
            train_ds = Dataset(train_c, train_y, batch_size=BATCH_SIZE, shuffle=True)
            val_ds   = Dataset(test_c,  test_y,  batch_size=BATCH_SIZE, shuffle=False)

            try:
                trainer.fit(train_ds, val_ds, log_interval=200, eval_interval=200)
            except Exception as exc:
                logger.error(f"  [{tag}] Training failed: {exc}")
                fold_accs.append(0.0); continue

            try:
                preds  = model(test_c)
                y_pred = predict_from_output(preds, len(test_c), discard)
                if discard:
                    y_true = np.argmax(
                        np.diagonal(test_y, axis1=-2, axis2=-1), axis=-1
                    ).tolist()
                else:
                    y_true = np.argmax(
                        test_y.reshape(len(test_c), -1), axis=1
                    ).tolist()
                acc = float(accuracy_score(y_true, y_pred))
            except Exception as exc:
                logger.error(f"  [{tag}] Prediction failed: {exc}")
                acc = 0.0

            fold_accs.append(acc)
            logger.info(f"    [{tag}] s={seed} f={fold_i+1}/{N_FOLDS} "
                        f"acc={acc:.4f}  ({time.time()-t0:.0f}s)")

        all_accs.extend(fold_accs)
        logger.info(f"  [{tag}] seed={seed} mean={np.mean(fold_accs):.4f}")

    mean = float(np.mean(all_accs)) if all_accs else 0.0
    std  = float(np.std(all_accs))  if all_accs else 0.0
    logger.info(f"  [{tag}] SPSA n_anc={n_ancillas} discard={discard}  "
                f"{mean:.4f} ± {std:.4f}  ({len(all_accs)} fold-seed combos)")
    return {"mean": mean, "std": std, "all_fold_accs": all_accs,
            "n_circuits": len(circuits)}


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-VERB RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_verb(verb, entries):
    sents  = [e["sentence"] for e in entries]
    labels = [e["sense"]    for e in entries]
    c = Counter(labels)
    logger.info(f"\n{'='*60}")
    logger.info(f"  VERB: {verb}  senses={dict(c)}  n={len(sents)}")
    logger.info(f"{'='*60}")

    results = {"verb": verb, "n": len(sents), "distribution": dict(c)}

    logger.info(f"\n  [Classical baselines]")
    results["classical"] = run_classical(sents, labels, tag=verb)
    logger.info(f"\n  [AraBERT frozen]")
    results["arabert_frozen"] = run_arabert(sents, labels, tag=verb)

    # Build circuits once per config
    logger.info(f"\n  [Building base circuits (n_ancillas=0, discard=False)]")
    c_base, vi_base = build_circuits(sents, n_ancillas=0, discard=False, n_s_qubits=1)
    vl_base = [labels[i] for i in vi_base]
    logger.info(f"  base circuits: {len(c_base)}/{len(sents)}")

    logger.info(f"\n  [Building ancilla circuits (n_ancillas=1, discard=True)]")
    c_anc, vi_anc = build_circuits(sents, n_ancillas=1, discard=True, n_s_qubits=1)
    vl_anc = [labels[i] for i in vi_anc]
    logger.info(f"  ancilla circuits: {len(c_anc)}/{len(sents)}")

    logger.info(f"\n  [QFM base]")
    results["qfm_base"] = run_qfm(
        sents, labels, 0, False, c_base, vl_base, tag=f"{verb}/QFM_base")

    logger.info(f"\n  [QFM ancilla]")
    results["qfm_ancilla"] = run_qfm(
        sents, labels, 1, True, c_anc, vl_anc, tag=f"{verb}/QFM_ancilla")

    logger.info(f"\n  [SPSA base]")
    results["spsa_base"] = run_spsa(
        sents, labels, 0, False, c_base, vl_base, tag=f"{verb}/SPSA_base")

    logger.info(f"\n  [SPSA ancilla]")
    results["spsa_ancilla"] = run_spsa(
        sents, labels, 1, True, c_anc, vl_anc, tag=f"{verb}/SPSA_ancilla")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  POOLED (all verbs — 100 per class)
# ═══════════════════════════════════════════════════════════════════════════════

_SENSE_A = {"lift", "carry", "cut", "strike"}
_SENSE_B = {"file", "convey", "sever", "exemplify"}

def run_pooled(all_entries):
    sents  = [e["sentence"] for e in all_entries]
    labels = ["structural_A" if e["sense"] in _SENSE_A else "structural_B"
              for e in all_entries]
    c = Counter(labels)
    logger.info(f"\n{'='*60}")
    logger.info(f"  POOLED (all verbs)  labels={dict(c)}  n={len(sents)}")
    logger.info(f"{'='*60}")

    results = {"verb": "POOLED", "n": len(sents), "distribution": dict(c)}

    logger.info(f"\n  [Classical baselines]")
    results["classical"] = run_classical(sents, labels, tag="POOLED")
    logger.info(f"\n  [AraBERT frozen]")
    results["arabert_frozen"] = run_arabert(sents, labels, tag="POOLED")

    logger.info(f"\n  [Building pooled base circuits]")
    c_base, vi_base = build_circuits(sents, n_ancillas=0, discard=False, n_s_qubits=1)
    vl_base = [labels[i] for i in vi_base]
    logger.info(f"  base circuits: {len(c_base)}/{len(sents)}")

    logger.info(f"\n  [Building pooled ancilla circuits]")
    c_anc, vi_anc = build_circuits(sents, n_ancillas=1, discard=True, n_s_qubits=1)
    vl_anc = [labels[i] for i in vi_anc]
    logger.info(f"  ancilla circuits: {len(c_anc)}/{len(sents)}")

    logger.info(f"\n  [QFM base]")
    results["qfm_base"] = run_qfm(
        sents, labels, 0, False, c_base, vl_base, tag="POOLED/QFM_base")
    logger.info(f"\n  [QFM ancilla]")
    results["qfm_ancilla"] = run_qfm(
        sents, labels, 1, True, c_anc, vl_anc, tag="POOLED/QFM_ancilla")
    logger.info(f"\n  [SPSA base]")
    results["spsa_base"] = run_spsa(
        sents, labels, 0, False, c_base, vl_base, tag="POOLED/SPSA_base")
    logger.info(f"\n  [SPSA ancilla]")
    results["spsa_ancilla"] = run_spsa(
        sents, labels, 1, True, c_anc, vl_anc, tag="POOLED/SPSA_ancilla")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("  Arabic QNLP — exp14 v2: Ancillary-Qubit WSD (vocab-controlled)")
    logger.info("=" * 70)
    logger.info(f"  Dataset: {DATASET_KEY}  |  N_LAYERS={N_LAYERS}  EPOCHS={EPOCHS}")
    logger.info(f"  CV: {N_SPLITS}-fold × {N_REPEATS}-repeat = {N_FOLDS} splits")
    logger.info(f"  QFM seeds={len(QFM_SEEDS)}  SPSA seeds={SPSA_SEEDS}")
    logger.info(f"  Design: 8 exact matched pairs (رفع), shared objects (حمل),")
    logger.info(f"          shared subjects (قطع, ضرب) → expected AraVec ~79%")

    data    = json.load(open("sentences.json", encoding="utf-8"))
    wsd_all = data.get(DATASET_KEY, [])
    if not wsd_all:
        logger.critical(f"{DATASET_KEY} not found in sentences.json. "
                        "Run generate_exp14_data_v2.py first.")
        sys.exit(1)

    logger.info(f"\n  Loaded {len(wsd_all)} entries")
    c = Counter(e["label"] for e in wsd_all)
    for k, v in sorted(c.items()):
        logger.info(f"    {k}: {v}")

    by_verb = defaultdict(list)
    for e in wsd_all:
        by_verb[e["verb"]].append(e)

    all_results = {}

    for verb in ["رفع", "حمل", "قطع", "ضرب"]:
        entries = by_verb.get(verb, [])
        if not entries:
            logger.warning(f"No entries for verb {verb}"); continue
        r = run_verb(verb, entries)
        all_results[verb] = r
        fname = OUTPUT_DIR / f"results_{verb}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    pooled = run_pooled(wsd_all)
    all_results["POOLED"] = pooled
    with open(OUTPUT_DIR / "results_POOLED.json", "w", encoding="utf-8") as f:
        json.dump(pooled, f, ensure_ascii=False, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  EXP14 v2 SUMMARY — Vocabulary-Controlled Structural WSD")
    logger.info("=" * 70)
    logger.info(f"\n  {'Task':<8} {'AraVec_SVM':>12} {'AraBERT':>10} "
                f"{'QFM_base':>10} {'QFM_anc':>9} {'SPSA_base':>11} {'SPSA_anc':>10}")
    logger.info(f"  {'-'*72}")

    for key in ["رفع", "حمل", "قطع", "ضرب", "POOLED"]:
        r = all_results.get(key)
        if r is None: continue
        av  = r["classical"].get("AraVec_SVM_rbf", {}).get("mean", 0.0)
        ab  = r["arabert_frozen"].get("mean", 0.0)
        qb  = r["qfm_base"].get("mean", 0.0)
        qa  = r["qfm_ancilla"].get("mean", 0.0)
        sb  = r["spsa_base"].get("mean", 0.0)
        sa  = r["spsa_ancilla"].get("mean", 0.0)
        logger.info(f"  {key:<8} {av:>12.4f} {ab:>10.4f} "
                    f"{qb:>10.4f} {qa:>9.4f} {sb:>11.4f} {sa:>10.4f}")
        logger.info(f"  {'':8} Ancilla Δ → QFM: {qa-qb:+.4f}   SPSA: {sa-sb:+.4f}")

    logger.info("\n  Chance level (binary): 50.0%")
    logger.info(f"  Saved → {OUTPUT_DIR}/")

    with open(OUTPUT_DIR / "exp14_v2_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
