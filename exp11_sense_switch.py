# -*- coding: utf-8 -*-
"""
exp11_sense_switch.py
---------------------
Sense-Switch experiment for Arabic polysemous words using lambeq DisCoCat.

Core idea:
  In standard DisCoCat (exp8), the polysemous word has ONE parameter set shared
  across both senses — the circuit must disambiguate from context alone.
  In exp11, we give the polysemous word TWO parameter sets:
    - {word}__SA  : parameters specialised for sense A
    - {word}__SB  : parameters specialised for sense B
  For each sentence we build BOTH variants (circuit_SA and circuit_SB).
  During training we feed the CORRECT-sense circuit to the loss.
  At test time we run both circuits; the one with higher confidence wins
  ("auto" inference). We also report the oracle upper-bound where the
  correct-sense circuit is always chosen.

Seven binary pairs, matching exp8 exactly (same data, epochs, ansatz).
Comparison table at the end: exp8 IQP | exp11 oracle | exp11 auto | classical SVM.

Usage:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 exp11_sense_switch.py
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
        logging.FileHandler("exp11_sense_switch.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("exp11")

# ── AraVec ────────────────────────────────────────────────────────────────────
try:
    from gensim.models import Word2Vec
    _kv = Word2Vec.load("aravec/full_uni_cbow_300_twitter.mdl").wv
    logger.info(f"AraVec loaded: {len(_kv)} vectors × {_kv.vector_size}d")
    ARAVEC_KV = _kv
except Exception as e:
    logger.critical(f"AraVec load failed: {e}")
    sys.exit(1)

# ── lambeq ────────────────────────────────────────────────────────────────────
from lambeq import (
    IQPAnsatz, Sim14Ansatz, RemoveCupsRewriter,
    NumpyModel, QuantumTrainer, SPSAOptimizer, Dataset,
)
from lambeq.backend.grammar import Ty
from lambeq.training import BinaryCrossEntropyLoss

_remove_cups = RemoveCupsRewriter()

# ── arabic parser ─────────────────────────────────────────────────────────────
try:
    from camel_test2 import analyze_arabic_sentence_with_morph
    _ANALYSIS_OK = True
    logger.info("camel_test2 loaded.")
except ImportError as e:
    logger.critical(f"camel_test2 import failed: {e}")
    sys.exit(1)

from arabic_dep_reader import sentence_to_diagram_from_parse

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# ── settings ──────────────────────────────────────────────────────────────────
N_FOLDS    = 5
SEED       = 42
EPOCHS     = 300
BATCH_SIZE = 8
DATA_FILE  = "sentences.json"   # augmented dataset (15/class)

OUTPUT_DIR = Path("qnlp_experiment_outputs_per_set_v2") / "exp11_sense_switch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N = Ty('n')
S = Ty('s')

ANSATZ_NAMES = ["IQP", "Sim14"]
SPSA_PARAMS  = {"a": 0.05, "c": 0.06, "A": 30, "alpha": 0.602, "gamma": 0.101}

# polysemous pairs: (pivot_arabic_root, cls_a, cls_b)
PAIRS = [
    ("رجل",  "Ambiguity_Man",       "Ambiguity_Leg"),
    ("عين",  "Ambiguity_Eye",       "Ambiguity_Spring"),
    ("ملك",  "Ambiguity_King",      "Ambiguity_Possess"),
    ("ضرب",  "Ambiguity_Hit",       "Ambiguity_Multiply"),
    ("جمل",  "Ambiguity_Camel",     "Ambiguity_Sentences"),
    ("فتح",  "Ambiguity_Open",      "Ambiguity_Conquer"),
    ("علم",  "Ambiguity_Knowledge", "Ambiguity_Flag"),
]

# exp8 IQP results for comparison (loaded from results files at runtime)
EXP8_IQP = {
    "Binary_Man_Leg":          None,
    "Binary_Eye_Spring":       None,
    "Binary_King_Possess":     None,
    "Binary_Hit_Multiply":     None,
    "Binary_Camel_Sentences":  None,
    "Binary_Open_Conquer":     None,
    "Binary_Knowledge_Flag":   None,
}

# classical SVM baseline (from baseline_binary.py results)
CLASSICAL_SVM = {
    "Binary_Man_Leg":          None,
    "Binary_Eye_Spring":       None,
    "Binary_King_Possess":     None,
    "Binary_Hit_Multiply":     None,
    "Binary_Camel_Sentences":  None,
    "Binary_Open_Conquer":     None,
    "Binary_Knowledge_Flag":   None,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_exp8_results():
    """Load exp8 IQP results from saved JSON files."""
    exp8_dir = Path("qnlp_experiment_outputs_per_set_v2") / "exp8_binary_lexico"
    for pair_name in EXP8_IQP:
        fpath = exp8_dir / f"results_{pair_name}_IQP.json"
        if fpath.exists():
            try:
                data = json.load(open(fpath, encoding="utf-8"))
                EXP8_IQP[pair_name] = data.get("mean")
            except Exception:
                pass


def _load_classical_results():
    """Load classical SVM_linear baseline results."""
    cl_dir = Path("qnlp_experiment_outputs_per_set_v2") / "classical_baseline_binary"
    for pair_name in CLASSICAL_SVM:
        fpath = cl_dir / f"results_{pair_name}.json"
        if fpath.exists():
            try:
                data = json.load(open(fpath, encoding="utf-8"))
                clfs = data.get("classifiers", {})
                svm_l = clfs.get("SVM_linear", {}).get("mean")
                svm_r = clfs.get("SVM_rbf",    {}).get("mean")
                rf    = clfs.get("RF",          {}).get("mean")
                mlp   = clfs.get("MLP",         {}).get("mean")
                vals = [v for v in [svm_l, svm_r, rf, mlp] if v is not None]
                if vals:
                    CLASSICAL_SVM[pair_name] = max(vals)
            except Exception:
                pass


def _make_ansatz(name: str, n_layers: int = 1):
    ob = {S: 1, N: 1}   # n_s_qubits=1 (binary)
    if name == "IQP":
        return IQPAnsatz(ob, n_layers=n_layers, discard=False)
    return Sim14Ansatz(ob, n_layers=n_layers, discard=False)


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


def encode_labels(labels: List[str], class_list: List[str]) -> np.ndarray:
    """Encode labels as one-hot arrays of shape (n, 2) for n_s_qubits=1."""
    flat = np.zeros((len(labels), 2))
    for i, lbl in enumerate(labels):
        flat[i, class_list.index(lbl)] = 1.0
    return flat.reshape(len(labels), 2)   # shape (n, 2) for 1-qubit output


# ═══════════════════════════════════════════════════════════════════════════════
#  SENSE-TAGGED DIAGRAM BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def _token_matches_pivot(token: str, pivot_surface: str) -> bool:
    """
    Check whether a token text is a surface form of the pivot polysemous word.
    Uses substring matching to handle Arabic morphological variation.
    """
    stripped = token[2:] if token.startswith("ال") else token
    return pivot_surface in token or pivot_surface in stripped


def _build_sense_tagged_diagram(
    tokens:       List[str],
    analyses:     List[dict],
    structure:    str,
    roles:        dict,
    pivot_surface: str,
    sense_tag:    str,   # "SA" or "SB"
):
    """
    Build a grammar diagram where all pivot-word tokens are renamed to
    '{original_text}__{sense_tag}'.  Returns Diagram with cod == S.
    """
    modified = []
    for ana in analyses:
        word_text = ana.get("text", "")
        if _token_matches_pivot(word_text, pivot_surface):
            ana = dict(ana)
            ana["text"] = f"{word_text}__{sense_tag}"
        modified.append(ana)
    return sentence_to_diagram_from_parse(tokens, modified, structure, roles)


def build_dual_circuits(
    sentences:     List[str],
    pivot_surface: str,
    ansatz,
) -> Tuple[list, list, List[int]]:
    """
    For every sentence build two circuits: one with pivot tagged as SA,
    one with pivot tagged as SB.

    Returns:
        circuits_SA  : list of valid SA circuits
        circuits_SB  : list of valid SB circuits (parallel to circuits_SA)
        valid_idx    : indices into `sentences` that produced usable circuits
    """
    circuits_SA, circuits_SB, valid_idx = [], [], []

    for i, sent in enumerate(sentences):
        if not sent or not sent.strip():
            continue
        try:
            tokens, analyses, structure, roles = analyze_arabic_sentence_with_morph(sent)

            diag_sa = _build_sense_tagged_diagram(
                tokens, analyses, structure, roles, pivot_surface, "SA")
            diag_sb = _build_sense_tagged_diagram(
                tokens, analyses, structure, roles, pivot_surface, "SB")

            circ_sa = ansatz(_remove_cups(diag_sa))
            circ_sb = ansatz(_remove_cups(diag_sb))

            circuits_SA.append(circ_sa)
            circuits_SB.append(circ_sb)
            valid_idx.append(i)

        except Exception as exc:
            logger.warning(f"  [{i}] Build failed for '{sent[:30]}': {exc}")

    logger.info(
        f"  Circuits: {len(circuits_SA)}/{len(sentences)} succeeded "
        f"({'no pivot found' if len(circuits_SA)==0 else 'ok'})"
    )
    return circuits_SA, circuits_SB, valid_idx


# ═══════════════════════════════════════════════════════════════════════════════
#  CV RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_sense_switch_pair(
    pair_name:     str,
    sentences:     List[str],
    labels:        List[str],
    cls_a:         str,
    pivot_surface: str,
    ansatz_name:   str,
    n_layers:      int = 1,
) -> Dict:
    """
    Run 5-fold CV with sense-switch circuits for one binary pair.

    Returns a dict with oracle and auto accuracy metrics.
    """
    tag = f"{pair_name}/{ansatz_name}/L{n_layers}"
    logger.info(f"\n{'─'*65}")
    logger.info(f"  {tag}  pivot='{pivot_surface}'  n={len(sentences)}")

    class_list = sorted(set(labels))   # ["Ambiguity_X", "Ambiguity_Y"]
    n_classes  = len(class_list)
    assert n_classes == 2, f"Expected binary pair, got {class_list}"

    ansatz = _make_ansatz(ansatz_name, n_layers)

    # Build dual circuits for all sentences
    circuits_SA, circuits_SB, valid_idx = build_dual_circuits(
        sentences, pivot_surface, ansatz)

    if len(circuits_SA) < 4:
        logger.error(f"  Too few circuits ({len(circuits_SA)}), skipping.")
        return {
            "tag": tag, "error": "insufficient circuits",
            "oracle_mean": 0.0, "oracle_std": 0.0,
            "auto_mean":   0.0, "auto_std":   0.0,
        }

    valid_labels = [labels[i] for i in valid_idx]
    min_count    = min(Counter(valid_labels).values())
    n_folds      = min(N_FOLDS, min_count)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    oracle_accs, auto_accs = [], []
    all_true_oracle, all_pred_oracle = [], []
    all_true_auto,   all_pred_auto   = [], []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(circuits_SA, valid_labels)):
        t0 = time.time()

        # ── Correct-sense circuits per sentence ────────────────────────────
        def pick_correct(idx_list):
            return [
                circuits_SA[i] if valid_labels[i] == cls_a else circuits_SB[i]
                for i in idx_list
            ]

        train_circuits = pick_correct(tr_idx)
        test_correct   = pick_correct(te_idx)
        test_sa        = [circuits_SA[i] for i in te_idx]
        test_sb        = [circuits_SB[i] for i in te_idx]
        train_labels_f = [valid_labels[i] for i in tr_idx]
        test_labels_f  = [valid_labels[i] for i in te_idx]

        # ── Build NumpyModel with all SA+SB circuits so it has both symbol sets
        all_for_model = (
            [circuits_SA[i] for i in tr_idx] +
            [circuits_SB[i] for i in tr_idx] +
            test_sa + test_sb
        )
        try:
            model = NumpyModel.from_diagrams(all_for_model, use_jit=False)
        except Exception as e:
            logger.error(f"    NumpyModel failed: {e}")
            oracle_accs.append(0.0)
            auto_accs.append(0.0)
            continue

        model.weights = warmstart_weights(model)

        train_targets = encode_labels(train_labels_f, class_list)
        test_targets  = encode_labels(test_labels_f,  class_list)

        trainer = QuantumTrainer(
            model             = model,
            loss_function     = BinaryCrossEntropyLoss(),
            epochs            = EPOCHS,
            optimizer         = SPSAOptimizer,
            optim_hyperparams = SPSA_PARAMS,
            seed              = SEED + fold_i,
            verbose           = "suppress",
        )
        train_ds = Dataset(train_circuits, train_targets,
                           batch_size=BATCH_SIZE, shuffle=True)
        val_ds   = Dataset(test_correct,   test_targets,
                           batch_size=BATCH_SIZE, shuffle=False)

        try:
            trainer.fit(train_ds, val_ds, log_interval=100, eval_interval=100)
        except Exception as e:
            logger.error(f"    Training failed: {e}")
            oracle_accs.append(0.0)
            auto_accs.append(0.0)
            continue

        y_true = np.argmax(test_targets, axis=1).tolist()

        # ── Oracle evaluation (correct-sense circuit) ──────────────────────
        try:
            preds_oracle   = model(test_correct).reshape(len(te_idx), 2)
            y_pred_oracle  = np.argmax(preds_oracle, axis=1).tolist()
            oracle_acc     = float(accuracy_score(y_true, y_pred_oracle))
        except Exception as e:
            logger.error(f"    Oracle eval failed: {e}")
            oracle_acc    = 0.0
            y_pred_oracle = [0] * len(te_idx)

        oracle_accs.append(oracle_acc)
        all_true_oracle.extend(y_true)
        all_pred_oracle.extend(y_pred_oracle)

        # ── Auto evaluation (dual-circuit confidence selection) ────────────
        auto_preds = []
        try:
            preds_sa = model(test_sa).reshape(len(te_idx), 2)
            preds_sb = model(test_sb).reshape(len(te_idx), 2)
            for j in range(len(te_idx)):
                conf_sa = float(np.max(preds_sa[j]))
                conf_sb = float(np.max(preds_sb[j]))
                if conf_sa >= conf_sb:
                    auto_preds.append(int(np.argmax(preds_sa[j])))
                else:
                    auto_preds.append(int(np.argmax(preds_sb[j])))
        except Exception as e:
            logger.error(f"    Auto eval failed: {e}")
            auto_preds = [0] * len(te_idx)

        auto_acc = float(accuracy_score(y_true, auto_preds))
        auto_accs.append(auto_acc)
        all_true_auto.extend(y_true)
        all_pred_auto.extend(auto_preds)

        elapsed = time.time() - t0
        logger.info(
            f"    fold {fold_i+1}/{n_folds}  oracle={oracle_acc:.4f}  "
            f"auto={auto_acc:.4f}  ({elapsed:.0f}s)"
        )

    oracle_mean = float(np.mean(oracle_accs))
    oracle_std  = float(np.std(oracle_accs))
    auto_mean   = float(np.mean(auto_accs))
    auto_std    = float(np.std(auto_accs))

    logger.info(
        f"  → {tag}: oracle={oracle_mean:.4f}±{oracle_std:.4f}  "
        f"auto={auto_mean:.4f}±{auto_std:.4f}"
    )

    # Classification reports
    oracle_report = auto_report = ""
    if all_true_oracle and all_pred_oracle:
        try:
            oracle_report = classification_report(
                all_true_oracle, all_pred_oracle,
                labels=[0, 1], target_names=class_list, zero_division=0)
        except Exception:
            pass
    if all_true_auto and all_pred_auto:
        try:
            auto_report = classification_report(
                all_true_auto, all_pred_auto,
                labels=[0, 1], target_names=class_list, zero_division=0)
        except Exception:
            pass

    return {
        "tag":          tag,
        "pair_name":    pair_name,
        "ansatz":       ansatz_name,
        "n_layers":     n_layers,
        "pivot":        pivot_surface,
        "n_samples":    len(circuits_SA),
        "n_folds":      n_folds,
        "oracle_mean":  oracle_mean,
        "oracle_std":   oracle_std,
        "oracle_folds": oracle_accs,
        "oracle_report": oracle_report,
        "auto_mean":    auto_mean,
        "auto_std":     auto_std,
        "auto_folds":   auto_accs,
        "auto_report":  auto_report,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("  Arabic QNLP — exp11: Sense-Switch (Dual Parameter Sets)")
    logger.info("=" * 70)
    logger.info(f"  Data: {DATA_FILE}  |  Epochs: {EPOCHS}  |  Folds: {N_FOLDS}")
    logger.info(
        "  Strategy: polysemous word gets two parameter sets (SA / SB).")
    logger.info(
        "  Oracle: correct sense tag at test time (upper bound).")
    logger.info(
        "  Auto:   dual-circuit confidence selection (real deployment).\n")

    # ── Load comparison baselines ──────────────────────────────────────────
    _load_exp8_results()
    _load_classical_results()

    # ── Load dataset ──────────────────────────────────────────────────────
    data = json.load(open(DATA_FILE, encoding="utf-8"))
    la   = data.get("LexicalAmbiguity", [])
    la_by_label: Dict[str, List[str]] = {}
    for d in la:
        la_by_label.setdefault(d["label"], []).append(d["sentence"])

    all_results = []

    for pivot_ar, cls_a, cls_b in PAIRS:
        sents_a   = la_by_label.get(cls_a, [])
        sents_b   = la_by_label.get(cls_b, [])
        sentences = sents_a + sents_b
        labels    = [cls_a] * len(sents_a) + [cls_b] * len(sents_b)

        name_a    = cls_a.replace("Ambiguity_", "")
        name_b    = cls_b.replace("Ambiguity_", "")
        pair_name = f"Binary_{name_a}_{name_b}"

        logger.info(f"\n{'='*70}")
        logger.info(f"  Pair: {pair_name}  pivot='{pivot_ar}'")
        logger.info(f"  {len(sents_a)} {cls_a} + {len(sents_b)} {cls_b}")

        pair_results = []
        for ans_name in ANSATZ_NAMES:
            r = run_sense_switch_pair(
                pair_name, sentences, labels, cls_a, pivot_ar, ans_name,
                n_layers=1,
            )
            pair_results.append(r)
            all_results.append(r)

            fpath = OUTPUT_DIR / f"results_{pair_name}_{ans_name}.json"
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=2)

    # ── Summary table ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 90)
    logger.info("  SENSE-SWITCH RESULTS SUMMARY (chance = 50%)")
    logger.info("=" * 90)
    header = (f"{'Pair':<28} {'exp8_IQP':>10} {'e11_Oracle':>12} "
              f"{'e11_Auto':>10} {'Classical':>10}")
    logger.info(header)
    logger.info("-" * 90)

    # collect by pair_name, ansatz IQP
    by_pair: Dict[str, Dict] = {}
    for r in all_results:
        if r.get("ansatz") == "IQP" and "error" not in r:
            by_pair[r["pair_name"]] = r

    for pivot_ar, cls_a, cls_b in PAIRS:
        name_a    = cls_a.replace("Ambiguity_", "")
        name_b    = cls_b.replace("Ambiguity_", "")
        pair_name = f"Binary_{name_a}_{name_b}"

        r      = by_pair.get(pair_name, {})
        e8     = EXP8_IQP.get(pair_name)
        cl     = CLASSICAL_SVM.get(pair_name)
        oracle = r.get("oracle_mean")
        auto   = r.get("auto_mean")

        def _fmt(v): return f"{v:.1%}" if v is not None else "  N/A"

        row = (f"{pair_name:<28} {_fmt(e8):>10} {_fmt(oracle):>12} "
               f"{_fmt(auto):>10} {_fmt(cl):>10}")
        logger.info(row)

        # Flag quantum wins
        if oracle is not None and e8 is not None and oracle > e8 + 0.02:
            logger.info(f"  ↑ Sense-switch oracle beats exp8 by {oracle - e8:+.1%}")
        if auto is not None and e8 is not None and auto > e8 + 0.02:
            logger.info(f"  ↑ Sense-switch auto beats exp8 by {auto - e8:+.1%}")

    # ── Aggregate ─────────────────────────────────────────────────────────
    valid_oracles = [r["oracle_mean"] for r in all_results
                     if r.get("ansatz") == "IQP" and "error" not in r]
    valid_autos   = [r["auto_mean"]   for r in all_results
                     if r.get("ansatz") == "IQP" and "error" not in r]

    if valid_oracles:
        logger.info(f"\n  Avg oracle (IQP L1): {np.mean(valid_oracles):.4f}")
        logger.info(f"  Avg auto   (IQP L1): {np.mean(valid_autos):.4f}")
        e8_vals = [v for v in EXP8_IQP.values() if v is not None]
        if e8_vals:
            logger.info(f"  Avg exp8 IQP:        {np.mean(e8_vals):.4f}")

    # ── Save summary ──────────────────────────────────────────────────────
    summary = {
        "exp8_iqp":  EXP8_IQP,
        "classical": CLASSICAL_SVM,
        "exp11":     all_results,
    }
    spath = OUTPUT_DIR / "exp11_summary.json"
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved → {spath}")


if __name__ == "__main__":
    main()
