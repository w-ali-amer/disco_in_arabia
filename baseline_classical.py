# -*- coding: utf-8 -*-
"""
baseline_classical.py
---------------------
Classical AraVec + ML baseline for direct comparison with quantum results.

Runs the SAME experiment sets and SAME StratifiedKFold splits as exp5/6/7
but replaces the quantum circuit with simple averaged AraVec embeddings
fed into SVM, Random Forest, and MLP classifiers.

Usage:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 baseline_classical.py
"""

import os, sys, json, math, logging, warnings
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("qnlp_baseline_classical.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("baseline")

# ── AraVec ───────────────────────────────────────────────────────────────────
try:
    from gensim.models import Word2Vec
    _kv = Word2Vec.load("aravec/full_uni_cbow_300_twitter.mdl").wv
    logger.info(f"AraVec: {len(_kv)} vectors × {_kv.vector_size}d")
    ARAVEC_DIM = _kv.vector_size
    ARAVEC_KV  = _kv
except Exception as e:
    logger.critical(f"AraVec load failed: {e}")
    sys.exit(1)

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# ── CAMeL tokenizer (reuse Stanza via camel_test2 token lists) ───────────────
try:
    import stanza
    _nlp = stanza.Pipeline("ar", processors="tokenize", verbose=False,
                            download_method=None)
    def tokenize(sentence: str) -> List[str]:
        doc = _nlp(sentence)
        return [w.text for sent in doc.sentences for w in sent.words]
    logger.info("Stanza tokenizer loaded.")
except Exception as e:
    logger.warning(f"Stanza unavailable ({e}), using whitespace split.")
    def tokenize(sentence: str) -> List[str]:
        return sentence.strip().split()

# ── settings ──────────────────────────────────────────────────────────────────
N_FOLDS    = 5
SEED       = 42
OUTPUT_DIR = Path("qnlp_experiment_outputs_per_set_v2") / "classical_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LexicalAmbiguity subset — same 6 classes as exp6/7
LEXICO_6 = {"Ambiguity_Man","Ambiguity_Leg",
            "Ambiguity_Hit","Ambiguity_Multiply",
            "Ambiguity_King","Ambiguity_Possess"}

CLASSIFIERS = {
    "SVM_linear": SVC(kernel="linear", C=1.0, random_state=SEED),
    "SVM_rbf":    SVC(kernel="rbf",    C=10.0, gamma="scale", random_state=SEED),
    "RF":         RandomForestClassifier(n_estimators=200, random_state=SEED),
    "MLP":        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                random_state=SEED, early_stopping=True),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SENTENCE EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

def sentence_to_vec(sentence: str) -> np.ndarray:
    """
    Average AraVec vectors for all tokens in the sentence.
    OOV tokens contribute a zero vector (and are counted for normalisation).
    Falls back to stripping definite article if base word is OOV.
    """
    tokens = tokenize(sentence)
    vecs = []
    for tok in tokens:
        candidates = [tok]
        if tok.startswith("ال"):
            candidates.append(tok[2:])
        for c in candidates:
            if ARAVEC_KV.has_index_for(c):
                vecs.append(ARAVEC_KV.get_vector(c))
                break
    if not vecs:
        return np.zeros(ARAVEC_DIM)
    return np.mean(vecs, axis=0)


def embed_sentences(sentences: List[str]) -> np.ndarray:
    X = np.stack([sentence_to_vec(s) for s in sentences])
    # L2 normalise each row
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (same groupings as exp5/6/7)
# ═══════════════════════════════════════════════════════════════════════════════

def load_experiments() -> List[Dict]:
    data = json.load(open("sentences.json", encoding="utf-8"))
    exps = []

    # WordOrder
    wo = data.get("WordOrder", [])
    exps.append({"name":"WordOrder",
                 "sentences":[d["sentence"] for d in wo],
                 "labels":[d["label"] for d in wo]})

    # LexicalAmbiguity full (14 classes)
    la = data.get("LexicalAmbiguity", [])
    exps.append({"name":"LexicalAmbiguity_14class",
                 "sentences":[d["sentence"] for d in la],
                 "labels":[d["label"] for d in la]})

    # LexicalAmbiguity 6-class (same as exp6/7)
    la6 = [d for d in la if d["label"] in LEXICO_6]
    exps.append({"name":"LexicalAmbiguity_6class",
                 "sentences":[d["sentence"] for d in la6],
                 "labels":[d["label"] for d in la6]})

    # Morphology sub-experiments (same as exp5/6/7)
    morph = data.get("Morphology", [])

    number_map = {"Morph_SgMasc":"Sg","Morph_SgFem":"Sg",
                  "Morph_DualMasc":"Du","Morph_DualFem":"Du",
                  "Morph_PlMasc":"Pl","Morph_PlFem":"Pl",
                  "Morph_PlBroken":"Pl","Morph_AdjPlMasc":"Pl","Morph_AdjPlFem":"Pl"}
    tense_map  = {"Morph_Past":"Past","Morph_Pres":"Pres"}
    poss_map   = {"Morph_Poss1Sg":"Poss_1st","Morph_Poss1Pl":"Poss_1st",
                  "Morph_Poss2MascSg":"Poss_2nd","Morph_Poss2Pl":"Poss_2nd",
                  "Morph_Poss3MascSg":"Poss_3rd","Morph_Poss3FemSg":"Poss_3rd"}

    for subname, mapping in [("Morphology_Number", number_map),
                              ("Morphology_Tense",  tense_map),
                              ("Morphology_Possession", poss_map)]:
        sub = [{"sentence":d["sentence"],"label":mapping[d["label"]]}
               for d in morph if d["label"] in mapping]
        exps.append({"name":subname,
                     "sentences":[d["sentence"] for d in sub],
                     "labels":[d["label"] for d in sub]})

    return exps


# ═══════════════════════════════════════════════════════════════════════════════
#  CV EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_cv(exp_name: str, sentences: List[str], labels: List[str]) -> Dict:
    logger.info(f"\n{'='*60}")
    logger.info(f"  {exp_name}  —  n={len(sentences)}  classes={sorted(set(labels))}")

    X = embed_sentences(sentences)
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    n_classes = len(le.classes_)

    min_count = min(Counter(labels).values())
    folds     = min(N_FOLDS, min_count)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    results = {}
    for clf_name, clf_proto in CLASSIFIERS.items():
        fold_accs = []
        all_yt, all_yp = [], []

        for tr, te in skf.split(X, y):
            from sklearn.base import clone
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
        results[clf_name] = {"mean": mean, "std": std,
                              "folds": fold_accs, "report": report}
        logger.info(f"  {clf_name:<15} {mean:.4f} ± {std:.4f}")

    return {"exp_name": exp_name, "n_samples": len(sentences),
            "n_classes": n_classes, "chance": 1.0/n_classes,
            "classifiers": results}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("  Arabic QNLP — Classical AraVec Baseline")
    logger.info("=" * 70)

    experiments = load_experiments()
    all_results = []

    for ex in experiments:
        result = run_cv(ex["name"], ex["sentences"], ex["labels"])
        all_results.append(result)

        out = OUTPUT_DIR / f"results_{ex['name']}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # ── Final comparison table ────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  BASELINE RESULTS vs QUANTUM BEST")
    logger.info("=" * 70)

    # quantum best results from exp5/6/7 for direct comparison
    quantum_best = {
        "WordOrder":              ("Sim14/exp5", 0.1556),
        "LexicalAmbiguity_14class": ("IQP/exp5",  0.0640),
        "LexicalAmbiguity_6class":  ("IQP/exp6",  0.1667),
        "Morphology_Number":      ("IQP/exp6",   0.3263),
        "Morphology_Tense":       ("Sim14/exp5", 0.7533),
        "Morphology_Possession":  ("IQP/exp5",   0.2275),
    }

    logger.info(f"\n{'Experiment':<30} {'Chance':>7} {'Q-best':>8} {'SVM_lin':>9} "
                f"{'SVM_rbf':>9} {'RF':>8} {'MLP':>8}")
    logger.info("-" * 82)

    for r in all_results:
        name   = r["exp_name"]
        chance = r["chance"]
        qb_lbl, qb_acc = quantum_best.get(name, ("—", 0.0))
        clfs   = r["classifiers"]
        logger.info(
            f"{name:<30} {chance:>7.1%} {qb_acc:>8.1%}"
            f" {clfs['SVM_linear']['mean']:>9.1%}"
            f" {clfs['SVM_rbf']['mean']:>9.1%}"
            f" {clfs['RF']['mean']:>8.1%}"
            f" {clfs['MLP']['mean']:>8.1%}"
        )

    summary_path = OUTPUT_DIR / "baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()
