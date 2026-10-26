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
                              logging.FileHandler(os.path.join(ROOT, "exp52_dedup.log"), mode="w", encoding="utf-8")])
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
SEEDS = list(range(10))
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
# doc 36 §10: keep first occurrence of each duplicate sentence
_seen, _S, _L = set(), [], []
for _s, _l in zip(SENTS, LABELS):
    if _s in _seen: continue
    _seen.add(_s); _S.append(_s); _L.append(_l)
N_BEFORE = len(SENTS); SENTS, LABELS = _S, _L
logger.info("data: %d sentences (%d/class)", len(SENTS), NW)

R = {"experiment": "exp52", "design_doc": "qnlp_private_docs/36_exp52_wordorder_tied_ablation.md "
     "(6986d13 + 162994d + 5efc91f + 0747cb0)",
     "platform": platform.platform(), "python": sys.version.split()[0],
     "env": "qiskit_lambeq_env", "QFM_WARMSTART": os.environ["QFM_WARMSTART"],
     "sha256": {"sentences.json": sha("sentences.json"), "exp13": sha(EXP13),
                "arabic_dep_reader": sha(READER), "self": sha(os.path.join(ROOT, "exp52_dedup.py"))},
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


t0 = time.time()
A1 = qfm(1, "A"); D1 = qfm(1, "D")
R["dedup"] = {"n_before": N_BEFORE, "n_after": len(SENTS),
    "per_label": {l: LABELS.count(l) for l in set(LABELS)},
    "A_published_QFM_L1": {k: v for k, v in A1.items() if k != "valid_idx"},
    "D_nested_QFM_L1": {k: v for k, v in D1.items() if k != "valid_idx"},
    "published": 0.6492, "sd_A": 0.0315,
    "within_one_sd_of_published": abs(A1["mean"] - 0.6492) <= 0.0315}
R["wall_time_sec"] = time.time() - t0
out = os.path.join(ROOT, "results_exp52_dedup.json")
assert not os.path.exists(out), "Rule 3"
json.dump(R, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
logger.info("DEDUP n=%d A %.4f±%.4f D %.4f±%.4f", len(SENTS), A1["mean"], A1["std"], D1["mean"], D1["std"])
