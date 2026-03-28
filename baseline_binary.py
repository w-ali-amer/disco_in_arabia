# -*- coding: utf-8 -*-
"""
baseline_binary.py
------------------
Classical AraVec + ML baseline for the 7 binary polysemous pairs,
matching the exp8 setup exactly (same augmented dataset, same 5-fold CV).

Usage:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 baseline_binary.py
"""

import os, sys, json, logging, warnings
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Dict

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("qnlp_baseline_binary.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("baseline_binary")

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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

try:
    import stanza
    _nlp = stanza.Pipeline("ar", processors="tokenize", verbose=False,
                            download_method=None)
    def tokenize(s): return [w.text for sent in _nlp(s).sentences for w in sent.words]
    logger.info("Stanza tokenizer loaded.")
except Exception as e:
    logger.warning(f"Stanza unavailable ({e}), using whitespace split.")
    def tokenize(s): return s.strip().split()

N_FOLDS = 5
SEED    = 42
OUTPUT_DIR = Path("qnlp_experiment_outputs_per_set_v2") / "classical_baseline_binary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("Ambiguity_Man",       "Ambiguity_Leg"),
    ("Ambiguity_Eye",       "Ambiguity_Spring"),
    ("Ambiguity_King",      "Ambiguity_Possess"),
    ("Ambiguity_Hit",       "Ambiguity_Multiply"),
    ("Ambiguity_Camel",     "Ambiguity_Sentences"),
    ("Ambiguity_Open",      "Ambiguity_Conquer"),
    ("Ambiguity_Knowledge", "Ambiguity_Flag"),
]

CLASSIFIERS = {
    "SVM_linear": SVC(kernel="linear", C=1.0, random_state=SEED),
    "SVM_rbf":    SVC(kernel="rbf", C=10.0, gamma="scale", random_state=SEED),
    "RF":         RandomForestClassifier(n_estimators=200, random_state=SEED),
    "MLP":        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                random_state=SEED, early_stopping=True),
}

# Exp8 quantum IQP results for direct comparison
EXP8_IQP = {
    "Binary_Man_Leg":          0.700,
    "Binary_Eye_Spring":       0.633,
    "Binary_King_Possess":     0.533,
    "Binary_Hit_Multiply":     0.533,
    "Binary_Camel_Sentences":  0.400,
    "Binary_Open_Conquer":     0.400,
    "Binary_Knowledge_Flag":   0.700,
}


def sentence_to_vec(sentence: str) -> np.ndarray:
    tokens = tokenize(sentence)
    vecs = []
    for tok in tokens:
        candidates = [tok, tok[2:]] if tok.startswith("ال") else [tok]
        for c in candidates:
            if ARAVEC_KV.has_index_for(c):
                vecs.append(ARAVEC_KV.get_vector(c))
                break
    if not vecs:
        return np.zeros(ARAVEC_DIM)
    v = np.mean(vecs, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def embed_sentences(sentences: List[str]) -> np.ndarray:
    return np.stack([sentence_to_vec(s) for s in sentences])


def run_pair(cls_a: str, cls_b: str, la_data: List[Dict]) -> Dict:
    sents_a = [d["sentence"] for d in la_data if d["label"] == cls_a]
    sents_b = [d["sentence"] for d in la_data if d["label"] == cls_b]
    sentences = sents_a + sents_b
    labels    = [cls_a] * len(sents_a) + [cls_b] * len(sents_b)

    name_a = cls_a.replace("Ambiguity_", "")
    name_b = cls_b.replace("Ambiguity_", "")
    pair_name = f"Binary_{name_a}_{name_b}"

    logger.info(f"\n{'='*55}")
    logger.info(f"  {pair_name}  ({len(sents_a)}+{len(sents_b)} sentences)")

    X  = embed_sentences(sentences)
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    min_count = min(Counter(labels).values())
    folds = min(N_FOLDS, min_count)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    results = {}
    for clf_name, clf_proto in CLASSIFIERS.items():
        fold_accs = []
        all_yt, all_yp = [], []
        for tr, te in skf.split(X, y):
            clf = clone(clf_proto)
            clf.fit(X[tr], y[tr])
            yp = clf.predict(X[te])
            fold_accs.append(accuracy_score(y[te], yp))
            all_yt.extend(y[te].tolist())
            all_yp.extend(yp.tolist())
        mean = float(np.mean(fold_accs))
        std  = float(np.std(fold_accs))
        report = classification_report(all_yt, all_yp,
                                       target_names=le.classes_.tolist(),
                                       zero_division=0)
        results[clf_name] = {"mean": mean, "std": std, "folds": fold_accs, "report": report}
        logger.info(f"  {clf_name:<15} {mean:.4f} ± {std:.4f}")

    return {"pair_name": pair_name, "n": len(sentences),
            "cls_a": cls_a, "cls_b": cls_b, "classifiers": results}


def main():
    logger.info("=" * 65)
    logger.info("  Binary LexicalAmbiguity — Classical AraVec Baseline")
    logger.info("=" * 65)

    data = json.load(open("sentences.json", encoding="utf-8"))
    la   = data.get("LexicalAmbiguity", [])

    all_results = []
    for cls_a, cls_b in PAIRS:
        r = run_pair(cls_a, cls_b, la)
        all_results.append(r)
        out = OUTPUT_DIR / f"results_{r['pair_name']}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    # ── Comparison table ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 75)
    logger.info("  BINARY PAIRS: QUANTUM (IQP) vs CLASSICAL")
    logger.info("=" * 75)
    logger.info(f"\n{'Pair':<28} {'Q-IQP':>7} {'SVM_lin':>9} {'SVM_rbf':>9}"
                f" {'RF':>7} {'MLP':>7}  {'Q wins?':>8}")
    logger.info("-" * 75)

    for r in all_results:
        pn     = r["pair_name"]
        clfs   = r["classifiers"]
        q_iqp  = EXP8_IQP.get(pn, 0.0)
        svm_l  = clfs["SVM_linear"]["mean"]
        svm_r  = clfs["SVM_rbf"]["mean"]
        rf     = clfs["RF"]["mean"]
        mlp    = clfs["MLP"]["mean"]
        cl_best = max(svm_l, svm_r, rf, mlp)
        q_wins = "YES ↑" if q_iqp > cl_best + 0.02 else ("TIE ~" if abs(q_iqp - cl_best) <= 0.02 else "NO ↓")
        logger.info(
            f"{pn:<28} {q_iqp:>7.1%} {svm_l:>9.1%} {svm_r:>9.1%}"
            f" {rf:>7.1%} {mlp:>7.1%}  {q_wins:>8}"
        )

    summary_path = OUTPUT_DIR / "binary_baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()
