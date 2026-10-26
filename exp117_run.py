# -*- coding: utf-8 -*-
"""exp117_run.py -- doc 117 (REGISTERED ffd6ec1, A0 1d780a8): corrected numbers for erratum items V7 (WSD) and V9 (tense).

Runs in ~/paper_mac_env (lambeq 0.5.0, Stanza, camel-tools).  Stage 0: G-REPRO = the archived reader reproduces
results_a2_published_structure_check.json's collision statistics exactly (else STOP).  T-TENSE: verb-typing check,
the A0 fix (no Stanza VERB in the sentence -> first non-al token with any CAMeL verb analysis becomes VERB), QFM
(published hash parameters, IQP 1 qubit per type) + SVM, 10 seeds x 5-fold StratifiedGroupKFold (groups = diagram
identity), vs bag-of-symbols logistic regression on identical splits.  T-WSD (WSD_v2): drop identical-string
opposite-label items, then keep items whose diagram is label-consistent; same models per verb.
Writes results_exp117_stage0.json and results_exp117.json (Rule 3).
"""
import os, sys, json, hashlib, logging, math, platform, time
from collections import Counter, defaultdict
import numpy as np
logging.disable(logging.WARNING)
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(ROOT, "archive", "02_discocat_paper"); PIPE = os.path.join(ROOT, "archive", "01_parser_pipeline")
sys.path.insert(0, PIPE)
import torch
torch.set_num_threads(1)
import lambeq
from lambeq import IQPAnsatz, RemoveCupsRewriter, NumpyModel
from lambeq.backend.grammar import Ty, Word
import arabic_dep_reader as ADR
import camel_test2
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

SEEDS = list(range(10)); NF = 5
TAG = os.environ.get("EXP117_TAG", "")              # Stage 0 iteration suffix (Rule 3)
NOMINAL_SIG = "n|n.r @ s"
_rc = RemoveCupsRewriter()


def write_new(name, obj):
    p = os.path.join(ROOT, name)
    assert not os.path.exists(p), "Rule 3: %s exists" % name
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", name, flush=True)


def sig(d):
    return "|".join(str(b.cod) for b in d.boxes if isinstance(b, Word))


def diagrams(sents):
    return ADR.sentences_to_diagrams(sents, log_interval=0)


# ---- the A0 tense fix: a wrapper around the reader's analysis call (reader unchanged) ----
_orig = ADR.analyze_arabic_sentence_with_morph
FIX_LOG = []


def _camel_verb(tok):
    return any(a.get("pos") == "verb" for a in (camel_test2.CAMEL_ANALYZER.analyze(tok) or []))


def _fixed(sentence, *a, **k):
    tokens, analyses, structure, roles = _orig(sentence, *a, **k)
    if analyses and not any(x.get("upos") == "VERB" for x in analyses):
        for i, x in enumerate(analyses):
            if not x["text"].startswith("ال") and _camel_verb(x["text"]):
                FIX_LOG.append({"sentence": sentence, "token": x["text"], "old_upos": x.get("upos")})
                x["upos"] = "VERB"
                break
    return tokens, analyses, structure, roles


def a2_stats(data):
    out = {}
    for task, items in data.items():
        sents = [x["sentence"] for x in items]; labs = [x["label"] for x in items]
        k = [repr(d) for d in diagrams(sents)]
        byk = defaultdict(set)
        for kk, l in zip(k, labs):
            byk[kk].add(l)
        coll = [len(byk[kk]) > 1 for kk in k]
        ss = defaultdict(set)
        for s, l in zip(sents, labs):
            ss[s].add(l)
        strc = [len(ss[s]) > 1 for s in sents]
        t = {"n_diagram_collision_opposite_label": sum(coll), "n_identical_string_opposite_label": sum(strc)}
        if "verb" in items[0]:
            t["per_verb"] = {v: sum(coll[i] for i, x in enumerate(items) if x["verb"] == v)
                             for v in sorted({x["verb"] for x in items})}
        out[task] = t
    return out


def qfm_features(diags, n_layers):
    ans = IQPAnsatz({Ty("n"): 1, Ty("s"): 1}, n_layers=n_layers, discard=False)
    circs = [ans(_rc(d)) for d in diags]
    model = NumpyModel.from_diagrams(circs, use_jit=False)
    w = np.empty(len(model.symbols))
    for i, sym in enumerate(model.symbols):                      # published hash parameters (QFM_WARMSTART=legacy)
        w[i] = (int(hashlib.md5(str(sym).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF) * 2 * math.pi
    model.weights = w
    return np.asarray(model(circs)).reshape(len(circs), -1)


def cv(X, y, groups, kind):
    y = np.asarray(y); groups = np.asarray(groups)
    nmin = min(Counter(y).values()); ng = len(set(groups))
    nf = min(NF, nmin, ng)
    if nf < 2:
        return {"n": len(y), "insufficient": True}
    accs = []
    for seed in SEEDS:
        skf = StratifiedGroupKFold(n_splits=nf, shuffle=True, random_state=seed)
        f = []
        for tr, te in skf.split(X, y, groups):
            if kind == "svm":
                m = Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed))])
            else:
                m = LogisticRegression(max_iter=2000)
            m.fit(X[tr], y[tr]); f.append(float((m.predict(X[te]) == y[te]).mean()))
        accs.append(float(np.mean(f)))
    return {"n": int(len(y)), "folds": nf, "mean": float(np.mean(accs)), "seed_sd": float(np.std(accs)),
            "per_seed": accs, "majority": float(max(Counter(y).values()) / len(y))}


def verbless_key(s):
    toks = s.split()
    vi = next((i for i, t in enumerate(toks) if not t.startswith("ال") and _camel_verb(t)), None)
    return " ".join(t for i, t in enumerate(toks) if i != vi)


def evaluate(sents, labs, diags, groups=None):
    groups = groups or [repr(d) for d in diags]
    bow = CountVectorizer(token_pattern=r"\S+").fit_transform(sents).toarray().astype(float)
    res = {"bow_logreg": cv(bow, labs, groups, "lr")}
    for L in (1, 0, 2):
        res["QFM_L%d" % L] = cv(qfm_features(diags, L), labs, groups, "svm")
    res["n_distinct_diagrams"] = len(set(repr(d) for d in diags)); res["n_groups"] = len(set(groups))
    return res


def main():
    t0 = time.time()
    data = json.load(open(os.path.join(PAPER, "sentences.json"), encoding="utf-8"))
    a2 = json.load(open(os.path.join(ROOT, "results_a2_published_structure_check.json"), encoding="utf-8"))
    import stanza, camel_tools, qiskit, sklearn
    versions = {"python": platform.python_version(), "lambeq": lambeq.__version__, "stanza": stanza.__version__,
                "camel_tools": camel_tools.__version__, "qiskit": qiskit.__version__, "sklearn": sklearn.__version__,
                "numpy": np.__version__, "torch": torch.__version__, "ARABIC_POS_FUSION": os.environ.get("ARABIC_POS_FUSION", "0")}
    mine = a2_stats(data)
    diff = {}
    for task, t in mine.items():
        ref = a2["tasks"][task]
        for k in ("n_diagram_collision_opposite_label", "n_identical_string_opposite_label"):
            if t[k] != ref[k]:
                diff["%s.%s" % (task, k)] = (t[k], ref[k])
        for v, c in t.get("per_verb", {}).items():
            if c != ref["per_verb"][v]["diagram_collision"]:
                diff["%s.%s" % (task, v)] = (c, ref["per_verb"][v]["diagram_collision"])
    st0 = {"doc": "117 (ffd6ec1, A0 1d780a8)", "versions": versions, "G_REPRO": {"pass": not diff, "diffs": diff, "stats": mine},
           "aravec": "dropped (no model on this machine; doc 117 A0.1)", "wall_s": time.time() - t0}
    write_new("results_exp117_stage0%s.json" % TAG, st0)
    print("G_REPRO", not diff, diff, flush=True)
    if diff:
        print("STOP: G-REPRO failed", flush=True); return
    if os.environ.get("EXP117_MODE") == "tense_twin":                 # doc 117 A1
        T = data["TenseBinary"]; ts = [x["sentence"] for x in T]; tl = [x["label"] for x in T]
        d_before = diagrams(ts)
        ADR.analyze_arabic_sentence_with_morph = _fixed; d_after = diagrams(ts); ADR.analyze_arabic_sentence_with_morph = _orig
        g = [verbless_key(s) for s in ts]
        out = {"doc": "117 A1 (twin-grouped tense)", "versions": versions, "n_groups": len(set(g)),
               "group_sizes": dict(Counter(Counter(g).values())),
               "T_TENSE_twin_grouped": {"after_fix": evaluate(ts, tl, d_after, g), "before_fix": evaluate(ts, tl, d_before, g)}}
        print("T_TENSE_twin", json.dumps({k: {m: r.get("mean") for m, r in v.items() if isinstance(r, dict)} for k, v in out["T_TENSE_twin_grouped"].items()}), out["n_groups"], out["group_sizes"], flush=True)
        write_new("results_exp117%s.json" % TAG, out); return
    out = {"doc": "117 (ffd6ec1, A0 1d780a8)", "versions": versions}
    # ---- T-TENSE ----
    T = data["TenseBinary"]; ts = [x["sentence"] for x in T]; tl = [x["label"] for x in T]
    d_before = diagrams(ts)
    nlp = camel_test2.nlp
    raw = {"Tense_Past": [0, 0], "Tense_Pres": [0, 0]}
    for s, l in zip(ts, tl):
        words = [w for sent in nlp(s).sentences for w in sent.words]
        cand = next((w for w in words if not w.text.startswith("ال") and _camel_verb(w.text)), None)
        raw[l][1] += 1
        raw[l][0] += int(cand is not None and cand.upos in ("VERB", "AUX"))
    ADR.analyze_arabic_sentence_with_morph = _fixed
    d_after = diagrams(ts)
    ADR.analyze_arabic_sentence_with_morph = _orig
    def vt(ds):
        return {l: sum(1 for d, ll in zip(ds, tl) if ll == l and sig(d) != NOMINAL_SIG) for l in ("Tense_Past", "Tense_Pres")}
    out["tense_check"] = {"verb_typed_before": vt(d_before), "verb_typed_after": vt(d_after), "n_per_label": dict(Counter(tl)),
                          "raw_stanza_verb_on_first_camel_verb_token": {l: {"verb": a, "n": n} for l, (a, n) in raw.items()},
                          "fix_fired": len(FIX_LOG), "fix_log": FIX_LOG,
                          "structure_after": {l: dict(Counter(sig(d) for d, ll in zip(d_after, tl) if ll == l)) for l in sorted(set(tl))}}
    print("tense check", {k: v for k, v in out["tense_check"].items() if k != "fix_log"}, flush=True)
    out["T_TENSE"] = {"after_fix": evaluate(ts, tl, d_after), "before_fix_grouped": evaluate(ts, tl, d_before)}
    print("T_TENSE", json.dumps({k: {m: r.get("mean") for m, r in v.items() if isinstance(r, dict)} for k, v in out["T_TENSE"].items()}), flush=True)
    # ---- T-WSD ----
    W = data["WordSenseDisambiguation_v2"]
    wsd = {}
    for v in sorted({x["verb"] for x in W}):
        items = [x for x in W if x["verb"] == v]
        ss = defaultdict(set)
        for x in items:
            ss[x["sentence"]].add(x["label"])
        kept = [x for x in items if len(ss[x["sentence"]]) == 1]
        n_strerr = len(items) - len(kept)
        ds = diagrams([x["sentence"] for x in kept])
        byk = defaultdict(set)
        for d, x in zip(ds, kept):
            byk[repr(d)].add(x["label"])
        surv = [(x, d) for x, d in zip(kept, ds) if len(byk[repr(d)]) == 1]
        rec = {"n_items": len(items), "dropped_identical_string_opposite_label": n_strerr,
               "dropped_diagram_collision": len(kept) - len(surv), "n_survivors": len(surv),
               "collapse_rate": (len(kept) - len(surv)) / len(kept) if kept else None,
               "survivor_labels": dict(Counter(x["label"] for x, _ in surv))}
        if len(set(x["label"] for x, _ in surv)) == 2:
            rec["results"] = evaluate([x["sentence"] for x, _ in surv], [x["label"] for x, _ in surv], [d for _, d in surv])
        else:
            rec["results"] = {"insufficient": True}
        wsd[v] = rec
        print("WSD", v, {k: r for k, r in rec.items() if k != "results"},
              {m: r.get("mean") for m, r in rec["results"].items() if isinstance(r, dict)}, flush=True)
    out["T_WSD"] = wsd
    out["wall_s"] = time.time() - t0
    write_new("results_exp117%s.json" % TAG, out)


if __name__ == "__main__":
    main()
