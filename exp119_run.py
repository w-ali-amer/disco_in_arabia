# -*- coding: utf-8 -*-
"""exp119_run.py -- doc 119 (REGISTERED 984ae87, A0 c049911): the paper's tasks under an improved pipeline.

Runs in ~/paper_mac_env (Stanza 1.10.1).  Improved parser = ARABIC_POS_FUSION=1 + the exp117 A0 verb-rescue wrapper;
agreement disagreements (verb vs subject GEN/NUM tags) are counted, not rejected.  Stage 0: structure + collision
stats under original vs improved parser; E1 box->token match; evaluator pass over all diagrams.
Encodings: E0 surface Word boxes (as published); E1 = every content Word box replaced by R_<root> (state) >>
P_<pattern> (map) from the CAMeL MLE analysis of its token (function words / rootless tokens stay surface).
Models (exp119_tn): Q, R, C (C uses the exp120 conditioning choice).  Baselines: bag-of-words LR, AraBERT frozen
[CLS] + LR, topology counter (LR on the diagram signature; word order).  CV: 5-fold StratifiedGroupKFold x 5 seeds;
word order grouped by pair_id; tense by the verbless key (root-seen / root-unseen tags); WSD per verb, identical
opposite-label strings dropped, grouped by sentence.  Writes results_exp119_stage0.json / results_exp119.json.
"""
import os, sys, json, re, time
from collections import Counter, defaultdict
import numpy as np
os.environ["ARABIC_POS_FUSION"] = "0"
ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(ROOT, "archive", "02_discocat_paper"); PIPE = os.path.join(ROOT, "archive", "01_parser_pipeline")
sys.path.insert(0, PIPE); sys.path.insert(0, ROOT)
import torch
torch.set_num_threads(1)
from lambeq.backend.grammar import Ty, Box, Word, Diagram, Id
import arabic_dep_reader as ADR
import camel_test2
from camel_tools.disambig.mle import MLEDisambiguator
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import exp119_tn as T

TAG = os.environ.get("EXP119_TAG", "")
DIMS = {"n": 4, "s": 2}
KINDS = ("Q", "R", "C"); ENCS = ("E0", "E1")
SEEDS = range(5)
CASE = re.compile(r"[ً-ْ]+$")
MLE = MLEDisambiguator.pretrained()
CONTENT = {"noun", "verb", "adj", "noun_quant", "noun_num", "adj_comp", "adv"}


def write_new(name, obj):
    p = os.path.join(ROOT, name); assert not os.path.exists(p), "Rule 3: " + name
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float); print("wrote", name, flush=True)


_orig = ADR.analyze_arabic_sentence_with_morph


def _camel_verb(tok):
    return any(a.get("pos") == "verb" for a in (camel_test2.CAMEL_ANALYZER.analyze(tok) or []))


def _fixed(sentence, *a, **k):
    tokens, analyses, structure, roles = _orig(sentence, *a, **k)
    if analyses and not any(x.get("upos") == "VERB" for x in analyses):
        for x in analyses:
            if not x["text"].startswith("ال") and _camel_verb(x["text"]):
                x["upos"] = "VERB"; break
    return tokens, analyses, structure, roles


def parse(sents, improved):
    os.environ["ARABIC_POS_FUSION"] = "1" if improved else "0"
    ADR.analyze_arabic_sentence_with_morph = _fixed if improved else _orig
    try:
        return ADR.sentences_to_diagrams(sents, log_interval=0)
    finally:
        ADR.analyze_arabic_sentence_with_morph = _orig; os.environ["ARABIC_POS_FUSION"] = "0"


def sig(d):
    return "|".join(str(b.cod) for b in d.boxes if isinstance(b, Word))


def surface(name):
    return name.split("_")[0]


def tags(name):
    return dict(t.split("-", 1) for t in name.split("_")[1:] if "-" in t)


def mle_map(sentence):
    toks = sentence.split()
    res = {}
    for tok, d in zip(toks, MLE.disambiguate(toks)):
        a = d.analyses[0].analysis
        res[tok] = {"root": a.get("root"), "pattern": CASE.sub("", a.get("pattern") or ""), "pos": a.get("pos")}
    return res


def factorise(d, sentence, stats):
    """E1: replace each content Word box (a state) by R_root >> P_pattern on the same codomain."""
    mm = mle_map(sentence)
    def f(box):
        if isinstance(box, Word) and len(box.dom) == 0:
            a = mm.get(surface(box.name))
            if a is None:
                stats["unmatched"] += 1; return box
            if a["pos"] not in CONTENT or not a["root"] or a["root"].count("#") >= 2:   # function word / no usable root
                stats["surface_kept"] += 1; return box
            stats["factorised"] += 1
            return Box("R_" + a["root"] + "|" + str(box.cod), Ty(), box.cod) >> Box("P_" + a["pattern"] + "|" + str(box.cod), box.cod, box.cod)
        return box
    out = None
    for layer in d.layers:
        b = f(layer.box)
        step = Id(layer.left) @ b @ Id(layer.right)
        out = step if out is None else out >> step
    return out


def lr_acc(docs, y, tr, te):
    X = CountVectorizer(token_pattern=r"\S+").fit_transform(docs)
    return (LogisticRegression(max_iter=5000).fit(X[tr], y[tr]).predict(X[te]) == y[te]).astype(int)


def verbless_key(s):
    toks = s.split(); vi = next((i for i, t in enumerate(toks) if not t.startswith("ال") and _camel_verb(t)), None)
    return " ".join(t for i, t in enumerate(toks) if i != vi), (toks[vi] if vi is not None else None)


def agreement_disagreements(diags):
    bad = 0
    for d in diags:
        ws = [b for b in d.boxes if isinstance(b, Word)]
        v = next((b for b in ws if "ASP" in tags(b.name)), None)
        if v is None:
            continue
        nouns = [b for b in ws if b is not v and str(b.cod) == "n"]
        if nouns:
            tv, tn = tags(v.name), tags(nouns[0].name)
            if tv.get("GEN") and tn.get("GEN") and tv["GEN"] != tn["GEN"]:
                bad += 1
    return bad


def main():
    t0 = time.time()
    data = json.load(open(os.path.join(PAPER, "sentences.json"), encoding="utf-8"))
    tasks = {"word_order": data["WordOrderMatched"], "tense": data["TenseBinary"]}
    W = data["WordSenseDisambiguation_v2"]
    for v in sorted({x["verb"] for x in W}):
        items = [x for x in W if x["verb"] == v]
        ss = defaultdict(set)
        for x in items:
            ss[x["sentence"]].add(x["label"])
        tasks["wsd_" + v] = [x for x in items if len(ss[x["sentence"]]) == 1]
    c120 = json.load(open(os.path.join(ROOT, "results_exp120.json"), encoding="utf-8"))["c_conditioning"]["chosen"]
    c_lr = float(c120.split(",")[0][3:]); c_init = None if c120.endswith("None") else 1.0
    st0 = {"doc": "119 (984ae87, A0 c049911)", "c_conditioning_from_exp120": c120, "tasks": {}}
    prepared = {}
    for name, items in tasks.items():
        sents = [x["sentence"] for x in items]; labs = [x["label"] for x in items]
        classes = sorted(set(labs)); y = np.array([classes.index(l) for l in labs])
        d_orig = parse(sents, False); d_imp = parse(sents, True)
        def coll(ds):
            byk = defaultdict(set)
            for d, l in zip(ds, labs):
                byk[repr(d)].add(l)
            return sum(len(byk[repr(d)]) > 1 for d in ds)
        stats = Counter()
        e1 = [factorise(d, s_, stats) for d, s_ in zip(d_imp, sents)]
        fails = 0
        for enc, ds in (("E0", d_imp), ("E1", e1)):
            for d in ds:
                try:
                    T.Structure(d, DIMS)
                except Exception:
                    fails += 1
            try:                                                   # v2: full model build + forward catches shape clashes
                m = T.TNModel(ds, DIMS, "C", 0, shape_keyed=True); m(list(range(len(ds))))
            except Exception as e:
                fails += 1; print("model build failed", name, enc, e, flush=True)
        st0["tasks"][name] = {"n": len(items), "classes": classes,
                              "structure_original": dict(Counter(sig(d) for d in d_orig)),
                              "structure_improved": dict(Counter(sig(d) for d in d_imp)),
                              "collisions_original": coll(d_orig), "collisions_improved": coll(d_imp),
                              "agreement_disagreements_improved": agreement_disagreements(d_imp),
                              "e1_box_stats": dict(stats), "evaluator_failures": fails}
        print("stage0", name, st0["tasks"][name], int(time.time() - t0), flush=True)
        prepared[name] = (items, sents, y, d_imp, e1)
    write_new("results_exp119_stage0%s.json" % TAG, st0)
    if any(v["evaluator_failures"] for v in st0["tasks"].values()):
        print("STOP: evaluator failures", flush=True); return
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02"); bm = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv02").eval()
    out = {"doc": st0["doc"], "c_conditioning_from_exp120": c120, "tasks": {}}
    for name, (items, sents, y, d0, d1) in prepared.items():
        if name == "word_order":
            groups = [x["pair_id"] for x in items]
        elif name == "tense":
            groups = [verbless_key(s_)[0] for s_ in sents]
        else:
            groups = sents
        with torch.no_grad():
            E = bm(**tok(sents, return_tensors="pt", padding=True)).last_hidden_state[:, 0].numpy()
        sig_docs = [sig(d).replace(" ", "") for d in d0]
        res = defaultdict(list); rootseen = defaultdict(lambda: defaultdict(list))
        vroot = {}
        if name == "tense":
            for s_ in sents:
                v = verbless_key(s_)[1]; vroot[s_] = mle_map(s_).get(v, {}).get("root") if v else None
        nmin = min(Counter(y).values()); nf = min(5, nmin, len(set(groups)))
        for seed in SEEDS:
            for tr, te in StratifiedGroupKFold(nf, shuffle=True, random_state=seed).split(sents, y, groups):
                tr, te = tr.tolist(), te.tolist()
                res["bow_logreg"].append(lr_acc(sents, y, np.array(tr), np.array(te)).mean())
                res["arabert_frozen"].append(float((LogisticRegression(max_iter=5000).fit(E[tr], y[tr]).predict(E[te]) == y[te]).mean()))
                if name == "word_order":
                    res["topology_counter"].append(lr_acc(sig_docs, y, np.array(tr), np.array(te)).mean())
                seen = {vroot.get(sents[i]) for i in tr} if name == "tense" else set()
                for kind in KINDS:
                    for enc, ds in (("E0", d0), ("E1", d1)):
                        kw = {"lr": c_lr, "init": c_init} if kind == "C" else {}
                        r = T.train_eval(ds, y.tolist(), tr, te, DIMS, kind, seed, shape_keyed=True, **kw)
                        arm = kind + "-" + enc
                        res[arm].append(r["test_acc"]); res[arm + "_train"].append(r["train_acc"])
                        if name == "tense":
                            for i, pr in zip(te, r["test_pred"]):
                                rootseen[arm]["seen" if vroot.get(sents[i]) in seen else "unseen"].append(int(pr == y[i]))
        rec = {k: float(np.mean(v)) for k, v in res.items()}
        rec["n"] = len(items); rec["folds"] = nf
        if name == "tense":
            rec["root_seen_split"] = {a: {k: (float(np.mean(v)), len(v)) for k, v in d.items()} for a, d in rootseen.items()}
            rec["E1_minus_E0_root_unseen"] = {k: (rec["root_seen_split"].get(k + "-E1", {}).get("unseen", (np.nan,))[0]
                                                  - rec["root_seen_split"].get(k + "-E0", {}).get("unseen", (np.nan,))[0]) for k in KINDS}
        out["tasks"][name] = rec
        print("task", name, {k: round(v, 3) for k, v in rec.items() if isinstance(v, float)}, int(time.time() - t0), flush=True)
        if name == "tense":
            print("   tense root split", rec["root_seen_split"], rec["E1_minus_E0_root_unseen"], flush=True)
    out["wall_s"] = time.time() - t0
    write_new("results_exp119%s.json" % TAG, out)


if __name__ == "__main__":
    main()
