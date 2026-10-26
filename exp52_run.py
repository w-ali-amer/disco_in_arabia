# -*- coding: utf-8 -*-
"""exp52_run.py -- doc 36 (6986d13, §1.3 amendment 162994d, §8 amendment 5efc91f,
§8.6 0747cb0). Does the PUBLISHED Arabic word-order result survive (C) lexical tying
and (D) the nested-cup VSO convention (doc 82 erratum P5)?

Runs wholly on WSL ~/discocat_arabic_v2/qiskit_lambeq_env (doc 36 §8.1).
The published pipeline is NOT copied: the exact function sources are extracted
from archive/02_discocat_paper/exp13_arabert_comparison.py with `ast` and executed,
and their sha256 is recorded (code-identity gate). Archive files are never edited.

Phases (doc 36): A reproduction; B baseline integrity (+ §8.6 twin diagnostic);
C tied ablation; D nested-cup convention (+ D'' crossed-inner, D3 descriptive).
"""
import os, sys, ast, json, math, hashlib, time, platform, logging, warnings
from collections import Counter
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(ROOT, "archive", "02_discocat_paper")
PIPE = os.path.join(ROOT, "archive", "01_parser_pipeline")
EXP13 = os.path.join(PAPER, "exp13_arabert_comparison.py")
READER = os.path.join(PIPE, "arabic_dep_reader.py")
sys.path.insert(0, PIPE)
os.environ["QFM_WARMSTART"] = "legacy"          # doc 36 §1.3 (1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(os.path.join(ROOT, "exp52_run.log"), mode="w", encoding="utf-8")])
logger = logging.getLogger("exp52")

import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec
from lambeq import IQPAnsatz, RemoveCupsRewriter, NumpyModel
from lambeq.backend.grammar import Ty, Word, Cup, Id, Swap, Diagram
import arabic_dep_reader as adr
from arabic_dep_reader import sentences_to_diagrams

def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

# ---------------------------------------------------------------- code identity
SRC = open(EXP13, encoding="utf-8").read()
TREE = ast.parse(SRC)
WANT = ["_aravec_vec", "sentence_to_aravec", "warmstart_weights", "make_ansatz",
        "build_circuits", "run_qfm_cv", "_count_controlled", "_total_boxes",
        "run_topology_only"]
func_src = {}
for node in TREE.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT:
        func_src[node.name] = ast.get_source_segment(SRC, node)
assert set(func_src) == set(WANT), set(WANT) - set(func_src)

_kv = Word2Vec.load(os.path.join(ROOT, "aravec", "full_uni_cbow_300_twitter.mdl")).wv
NS = dict(np=np, math=math, hashlib=hashlib, os=os, logger=logger,
          StratifiedKFold=StratifiedKFold, accuracy_score=accuracy_score, SVC=SVC,
          StandardScaler=StandardScaler, Pipeline=Pipeline, NumpyModel=NumpyModel,
          IQPAnsatz=IQPAnsatz, _remove_cups=RemoveCupsRewriter(),
          sentences_to_diagrams=sentences_to_diagrams, S_ty=Ty("s"), N_ty=Ty("n"),
          N_FOLDS=5, ARAVEC_KV=_kv, ARAVEC_DIM=_kv.vector_size, Optional=None,
          List=list, Dict=dict, Tuple=tuple)
from typing import Optional, List, Dict, Tuple
NS.update(Optional=Optional, List=List, Dict=Dict, Tuple=Tuple)
for name in WANT:
    exec(func_src[name], NS)
CODE_SHA = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in func_src.items()}

SMOKE = os.environ.get("EXP52_SMOKE") == "1"
SEEDS = [0] if SMOKE else list(range(10))
N, S = Ty("n"), Ty("s")

# ---------------------------------------------------------------- VSO conventions
_ORIG_VSO = adr._vso
VSO_LOG = []                      # which sentences routed through _vso

def vso_A(verb, subj, obj):       # published (crossed, outer -> subject)
    d = _ORIG_VSO(verb, subj, obj)
    VSO_LOG.append(("A", verb, subj, obj))
    return d

def vso_D(verb, subj, obj):       # nested planar, inner -> subject (Lambek)
    words = adr._verb_trans_vso(verb) @ adr._noun(subj) @ adr._noun(obj)
    d = words >> (Id(S) @ Id(N.l) @ Cup(N.l, N) @ Id(N)) >> (Id(S) @ Cup(N.l, N))
    VSO_LOG.append(("D", verb, subj, obj))
    return d

def vso_D2(verb, subj, obj):      # crossed, inner -> subject
    words = adr._verb_trans_vso(verb) @ adr._noun(subj) @ adr._noun(obj)
    d = (words >> (Id(S) @ Id(N.l) @ Id(N.l) @ Swap(N, N))
               >> (Id(S) @ Id(N.l) @ Swap(N.l, N) @ Id(N))
               >> (Id(S) @ Cup(N.l, N) @ Cup(N.l, N)))
    VSO_LOG.append(("D2", verb, subj, obj))
    return d

CONV = {"A": vso_A, "D": vso_D, "D2": vso_D2}

def set_conv(c):
    adr._vso = CONV[c]

def has_swap(d):
    return any(type(b).__name__ == "Swap" or getattr(b, "name", "").startswith("SWAP")
               for b in d.boxes)

def word_boxes(d):
    return [(b.name, str(b.cod)) for b in d.boxes if isinstance(b, Word)]

# ---------------------------------------------------------------- data
os.chdir(PAPER)
data = json.load(open("sentences.json", encoding="utf-8"))
wo = data["WordOrderMatched"]
svo = [d for d in wo if d["label"] == "WordOrder_SVO"]
vso = [d for d in wo if d["label"] == "WordOrder_VSO"]
NW = min(len(svo), len(vso))
SENTS = [d["sentence"] for d in svo[:NW]] + [d["sentence"] for d in vso[:NW]]
LABELS = ["WordOrder_SVO"] * NW + ["WordOrder_VSO"] * NW
logger.info("data: %d sentences (%d/class)", len(SENTS), NW)

R = {"experiment": "exp52", "design_doc": "qnlp_private_docs/36_exp52_wordorder_tied_ablation.md "
     "(6986d13 + 162994d + 5efc91f + 0747cb0)",
     "platform": platform.platform(), "python": sys.version.split()[0],
     "env": "qiskit_lambeq_env", "QFM_WARMSTART": os.environ["QFM_WARMSTART"],
     "sha256": {"sentences.json": sha("sentences.json"), "exp13": sha(EXP13),
                "arabic_dep_reader": sha(READER), "self": sha(os.path.join(ROOT, "exp52_run.py"))},
     "code_identity_sha256": CODE_SHA, "n": len(SENTS), "gates": {}, "phases": {}}
import lambeq, sklearn, stanza
R["versions"] = {"lambeq": lambeq.__version__, "numpy": np.__version__,
                 "sklearn": sklearn.__version__, "stanza": stanza.__version__}

def qfm(n_layers, conv, tie=None):
    """Published run_task QFM loop (exp13 L492-515) with optional tying."""
    set_conv(conv)
    circs, vidx = NS["build_circuits"](SENTS, n_layers, 1)
    vlab = [LABELS[i] for i in vidx]
    seed_means = []
    if tie is None:
        for seed in SEEDS:
            r = NS["run_qfm_cv"](SENTS, LABELS, n_layers, seed, 1, "QFM",
                                 circuits_cache=circs, valid_labels_cache=vlab)
            seed_means.append(r["mean"])
    else:
        seed_means = qfm_tied(circs, vlab, tie)
    return {"mean": float(np.mean(seed_means)), "std": float(np.std(seed_means)),
            "per_seed": seed_means, "n_valid": len(vidx), "valid_idx": vidx}

def tie_weights(model, tie):
    """doc 36 §4, operationalised BEFORE any full run (reviewer review 2026-09-23):
    T-verb: ALL content-verb boxes share one parameter tensor -- across SVO
      (n.r@s@n.l) and VSO (s@n.l@n.l) types alike -- keyed by (qubit count, param
      index); a content verb is a box whose type contains 's' AND '@' (this
      excludes the bare-'s' single-box parse fallbacks).
    T-all: additionally all noun boxes (type 'n') share one tensor.
    Tied = every member takes the weight of the group's first member (sorted)."""
    w = NS["warmstart_weights"](model).copy()
    groups = {}
    for i, sym in enumerate(model.symbols):
        name = str(sym)
        word, _, rest = name.partition("__")
        ty, _, idx = rest.rpartition("_")
        is_verb = ("s" in ty.split("@")) and ("@" in ty)
        is_noun = (ty == "n")
        if is_verb:
            key = ("V", ty.count("@") + 1, idx)
        elif is_noun and tie == "all":
            key = ("N", 1, idx)
        else:
            continue
        groups.setdefault(key, []).append((name, i))
    for key, mem in groups.items():
        mem.sort()
        for _, i in mem:
            w[i] = w[mem[0][1]]
    TIE_LOG[tie] = {str(k): len(v) for k, v in groups.items()}
    return w

TIE_LOG = {}

def qfm_tied(circs, vlab, tie):
    means = []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        accs = []
        for tr, te in skf.split(circs, vlab):
            model = NumpyModel.from_diagrams(circs, use_jit=False)
            model.weights = tie_weights(model, tie)
            trf = model([circs[i] for i in tr]).reshape(len(tr), -1)
            tef = model([circs[i] for i in te]).reshape(len(te), -1)
            pipe = Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed))])
            pipe.fit(trf, [vlab[i] for i in tr])
            accs.append(accuracy_score([vlab[i] for i in te], pipe.predict(tef)))
        means.append(float(np.mean(accs)))
    return means

def topo(conv):
    set_conv(conv)
    circs, vidx = NS["build_circuits"](SENTS, 1, 1)
    vlab = [LABELS[i] for i in vidx]
    counts = [NS["_count_controlled"](c) for c in circs]
    per_class = {c: dict(Counter(k for k, l in zip(counts, vlab) if l == c)) for c in ("WordOrder_SVO", "WordOrder_VSO")}
    r = NS["run_topology_only"](SENTS, LABELS, 1)
    acc = r["mean"]
    return {"acc": acc, "symmetrised": max(acc, 1 - acc), "controlled_per_class": per_class,
            "n_valid": len(vidx)}

# ================================================================ GATES (diagrams)
t0 = time.time()
diag = {}
for c in ("A", "D", "D2"):
    set_conv(c)
    VSO_LOG.clear()
    diag[c] = sentences_to_diagrams(SENTS, log_interval=0)
    R["gates"].setdefault("n_vso_routed", {})[c] = len(VSO_LOG)
vso_idx = [i for i in range(len(SENTS)) if diag["A"][i] != diag["D"][i]]
same_idx = [i for i in range(len(SENTS)) if i not in vso_idx]
g1 = all(diag["D"][i].cod == S and not has_swap(diag["D"][i]) for i in vso_idx)
g2 = all(diag["A"][i] == diag["D"][i] == diag["D2"][i] for i in same_idx)
g5 = all(word_boxes(diag["A"][i]) == word_boxes(diag["D"][i]) == word_boxes(diag["D2"][i]) for i in vso_idx)
# g4 designed-to-fail (reviewer review): the g1 predicate must FAIL on the published
# crossed diagrams -- otherwise g1 is vacuous.
g1_pred = lambda d: d.cod == S and not has_swap(d)
g4 = (len(vso_idx) > 0) and all(not g1_pred(diag["A"][i]) for i in vso_idx)
fallback = {c: sum(1 for d in diag["A"] if len(d.boxes) == 1 and d.cod == S) for c in ["A"]}
R["gates"].update({"g1_nested_cod_s_no_swap": g1, "g2_nonvso_identical": g2,
                   "g4_designed_to_fail_raises": g4, "g5_vso_same_words": g5,
                   "n_vso_diagrams": len(vso_idx),
                   "vso_idx_by_label": dict(Counter(LABELS[i] for i in vso_idx)),
                   "single_box_fallback_diagrams": fallback["A"],
                   "fallback_by_label": dict(Counter(LABELS[i] for i, d in enumerate(diag["A"]) if len(d.boxes) == 1))})
logger.info("GATES %s", {k: v for k, v in R["gates"].items()})
R["gates"]["published_A_vso_has_swap"] = all(has_swap(diag["A"][i]) for i in vso_idx)
R["gates"]["vso_routing_complete"] = {"n_vso_label": LABELS.count("WordOrder_VSO"), "n_vso_diagrams": len(vso_idx),
    "n_vso_routed": R["gates"]["n_vso_routed"], "vso_label_fallbacks": sum(1 for i, d in enumerate(diag["A"]) if len(d.boxes) == 1 and LABELS[i] == "WordOrder_VSO")}
R["gates"]["duplicate_sentences_descriptive"] = {"n_unique": len(set(SENTS)), "n": len(SENTS)}
# DESCRIPTIVE, added after the smoke run (not a criterion): per-sentence structure signature
# under the published reader, to locate where the topology signal comes from (smoke showed
# only 17/60 VSO sentences routed through _vso).
def _sig(d):
    return "|".join(str(b.cod) for b in d.boxes if isinstance(b, Word))
R["structure_by_label"] = {lab: dict(Counter(_sig(diag["A"][i]) for i in range(len(SENTS)) if LABELS[i] == lab))
                           for lab in ("WordOrder_SVO", "WordOrder_VSO")}
R["per_sentence_structure"] = [{"i": i, "sentence": SENTS[i], "label": LABELS[i], "sig": _sig(diag["A"][i]),
                                "routed_vso": i in vso_idx} for i in range(len(SENTS))]
R["gates"]["word_boxes_nonempty"] = all(len(word_boxes(diag["A"][i])) >= 3 for i in vso_idx)
if SMOKE:
    set_conv("A"); _c, _v = NS["build_circuits"](SENTS[:3] + SENTS[-3:], 1, 1)
    _m = NumpyModel.from_diagrams(_c, use_jit=False)
    logger.info("SMOKE symbol sample: %s", [str(s) for s in _m.symbols[:12]])
assert g1 and g2 and g4 and g5 and R["gates"]["published_A_vso_has_swap"] and R["gates"]["word_boxes_nonempty"], "diagram gates failed -- nothing runs"

# ================================================================ PHASE A
A1 = qfm(1, "A")
A0 = qfm(0, "A")
repro = abs(A1["mean"] - 0.6492) <= 0.0315
R["phases"]["A"] = {"QFM_IQP_L1": {k: v for k, v in A1.items() if k != "valid_idx"},
                    "QFM_IQP_L0": {k: v for k, v in A0.items() if k != "valid_idx"},
                    "wordorder_reproduced": repro, "published": {"mean": 0.6492, "std": 0.0315}}
logger.info("PHASE A L1 %.4f ± %.4f reproduced=%s", A1["mean"], A1["std"], repro)
muA, sdA = (A1["mean"], A1["std"]) if repro else (0.6492, 0.0315)

# ================================================================ PHASE B (+ §8.6)
TA = topo("A")
bag = [tuple(sorted(s.split())) for s in SENTS]
le = LabelEncoder().fit(LABELS); y = le.transform(LABELS)
X = np.stack([NS["sentence_to_aravec"](s) for s in SENTS])
splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(X, y))
accs, strat = [], {"twin_in_train": [0, 0], "twin_not_in_train": [0, 0]}
for tr, te in splits:
    pipe = Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42))])
    pipe.fit(X[tr], y[tr]); pr = pipe.predict(X[te])
    accs.append(accuracy_score(y[te], pr))
    trbags = Counter((bag[i], y[i]) for i in tr)
    for j, p in zip(te, pr):
        twin = any(trbags.get((bag[j], lab), 0) for lab in set(y) if lab != y[j])
        k = "twin_in_train" if twin else "twin_not_in_train"
        strat[k][0] += int(p == y[j]); strat[k][1] += 1
cond = {k: (v[0] / v[1] if v[1] else None, v[1]) for k, v in strat.items()}
anti = (cond["twin_in_train"][0] is not None and cond["twin_in_train"][0] <= 0.25 and
        (cond["twin_not_in_train"][0] is None or 0.35 <= cond["twin_not_in_train"][0] <= 0.65))
pred_dist = Counter()
for tr, te in splits:
    pipe = Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42))])
    pipe.fit(X[tr], y[tr]); pred_dist.update(pipe.predict(X[te]).tolist())
maj_frac = max(pred_dist.values()) / sum(pred_dist.values())
# §3 operationalised (pinned in code before any run): "near-constant" = majority predicted class >= 0.90
sub_expl = (max(float(np.mean(accs)), 1 - float(np.mean(accs))) >= 0.45) and maj_frac >= 0.90
R["phases"]["B"] = {"aravec_mean": float(np.mean(accs)), "aravec_std": float(np.std(accs)),
                    "aravec_twin_stratified": cond,
                    "aravec_subchance_is_twin_antilearning": anti,
                    "topology_only_A": TA,
                    "topology_only_symmetrised_matches_erratum": abs(TA["symmetrised"] - 0.6417) <= 0.005,
                    "aravec_pred_class_distribution": {str(k): v for k, v in pred_dist.items()}, "aravec_majority_pred_frac": maj_frac,
                    "baselines_sub_chance_explained": sub_expl,
                    "baselines_sub_chance_explained_operationalisation": "symmetrised >= 0.45 (≈0.5 or above) AND majority predicted class >= 0.90 (near-constant); pinned in code before any run"}
logger.info("PHASE B aravec %.4f twin-strat %s anti=%s | topo A %s", np.mean(accs), cond, anti, TA)

# ================================================================ PHASE C
C = {}
if repro:
    for tie in ("verb", "all"):
        C[tie] = qfm(1, "A", tie=tie)
    R["phases"]["C"] = {
        "T_verb": {k: v for k, v in C["verb"].items() if k != "valid_idx"},
        "T_all": {k: v for k, v in C["all"].items() if k != "valid_idx"},
        "tie_groups": TIE_LOG,
        "wordorder_survives_verb_tying": C["verb"]["mean"] >= 0.6492 - 0.0315,
        "wordorder_survives_full_tying": C["all"]["mean"] >= 0.6492 - 0.0315}
else:
    R["phases"]["C"] = {"not_run": "Phase A did not reproduce (doc 36 §2 gate)"}

# ================================================================ PHASE D
D1 = qfm(1, "D"); DD = qfm(1, "D2")
assert D1["valid_idx"] == A1["valid_idx"] == DD["valid_idx"], "g3 valid_idx differs"
R["gates"]["g3_valid_idx_identical"] = True
TD = topo("D"); TD2 = topo("D2")
D0 = qfm(0, "D"); DTall = qfm(1, "D", tie="all")
surv = D1["mean"] >= muA - sdA
floor = D1["mean"] <= 0.5 + sdA
outcome = "SURVIVES" if surv else ("COLLAPSES" if floor else "PARTIAL")
R["phases"]["D"] = {
    "reference": {"mu_A": muA, "sd_A": sdA, "source": "phase A" if repro else "archived (A not reproduced)"},
    "D_nested_QFM_L1": {k: v for k, v in D1.items() if k != "valid_idx"},
    "D2_crossed_inner_QFM_L1": {k: v for k, v in DD.items() if k != "valid_idx"},
    "D1_topology_nested": TD, "D1_topology_crossed_inner": TD2,
    "D3_nested_QFM_L0": {k: v for k, v in D0.items() if k != "valid_idx"},
    "D3_nested_T_all": {k: v for k, v in DTall.items() if k != "valid_idx"},
    "wordorder_survives_nested": surv, "nested_at_theorem_floor": floor,
    "wordorder_leg_assignment_matters": abs(DD["mean"] - A1["mean"]) >= sdA,
    "named_outcome": outcome}
R["wall_time_sec"] = time.time() - t0
out = "/tmp/exp52_smoke.json" if SMOKE else os.path.join(ROOT, "results_exp52.json")
if SMOKE and os.path.exists(out): os.remove(out)
assert not os.path.exists(out), "Rule 3: results_exp52.json exists"
json.dump(R, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
logger.info("PHASE D nested %.4f±%.4f crossed-inner %.4f±%.4f -> %s", D1["mean"], D1["std"], DD["mean"], DD["std"], outcome)
logger.info("wrote %s", out)
