# -*- coding: utf-8 -*-
"""exp121_run.py -- doc 121 (REGISTERED 67bfa7b): fair classical comparators + contextual disambiguator.

A1 (doc 120 data, splits unchanged): arms Q, R, C (n = 4; within-run references), CL (real tensors, logit readout;
n = 4 and n = 8), BOE-MLP (mean token embedding -> MLP -> logits); encodings E0 / E1m / E1g.  New classical arms pick
(lr, init) by final train loss on split (i) fold 1.  Gate complex_specific on split (ii).
A2 (doc 119 tense + WSD, twin-/sentence-grouped CV unchanged): E1 analyses from CAMeL BERTUnfactoredDisambiguator.
Validity first: BERT vs MLE agreement with gold on doc 120 data; aspect consistency on the 100 tense items.
Writes results_exp121.json (Rule 3).
"""
import os, sys, json, time, itertools, re
from collections import Counter, defaultdict
import multiprocessing as mp
import numpy as np
import torch
torch.set_num_threads(1)
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import exp119_tn as T
import exp120_run as R120
import exp119_run as R119

CASE = re.compile(r"[ً-ْ]+$")
G = {}                      # globals shared with forked workers


# ---------------- BOE-MLP ----------------
class BOE(torch.nn.Module):
    def __init__(self, vocab, seed, init=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.E = torch.nn.Parameter(torch.randn(vocab, 16, generator=g, dtype=torch.float64) * (init or 0.1))
        self.l1 = torch.nn.Linear(16, 32).double(); self.l2 = torch.nn.Linear(32, 2).double()
        torch.manual_seed(seed)
        for l in (self.l1, self.l2):
            torch.nn.init.normal_(l.weight, std=1 / np.sqrt(l.in_features)); torch.nn.init.zeros_(l.bias)

    def forward(self, bags):                                   # bags: list of index tensors
        h = torch.stack([self.E[b].mean(0) for b in bags])
        return torch.softmax(self.l2(torch.relu(self.l1(h))), dim=1)[:, 1]


def boe_train_eval(tokens, labels, tr, te, seed, lr=0.05, init=None, epochs=300):
    vocab = {t: i for i, t in enumerate(sorted({t for ts in tokens for t in ts}))}
    bags = [torch.tensor([vocab[t] for t in ts]) for ts in tokens]
    m = BOE(len(vocab), seed, init); y = torch.tensor(labels, dtype=torch.float64)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        p = m([bags[i] for i in tr]).clamp(1e-9, 1 - 1e-9)
        loss = torch.nn.functional.binary_cross_entropy(p, y[tr]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pt = m([bags[i] for i in te]) if len(te) else torch.zeros(0, dtype=torch.float64); ptr = m([bags[i] for i in tr])
    return {"test_pred": (pt > 0.5).long().tolist(), "test_acc": float(((pt > 0.5).double() == y[te]).double().mean()) if len(te) else float("nan"),
            "train_acc": float(((ptr > 0.5).double() == y[tr]).double().mean()), "final_loss": float(loss.detach())}


def job(spec):
    """spec = (ds_key, arm, enc, tr, te, seed, cond) -> result dict (run in a forked worker)."""
    key, arm, enc, tr, te, seed, cond = spec
    data = G[key]
    if arm == "MLP":
        r = boe_train_eval(data["tokens"][enc], data["y"], tr, te, seed, **cond)
    else:
        kind, n = arm.split("@")
        r = T.train_eval(data["diag"][enc], data["y"], tr, te, {"n": int(n), "s": 2}, kind, seed,
                         shape_keyed=data.get("shape_keyed", False), **cond)
    r["spec"] = (key, arm, enc, seed); r["te"] = te
    return r


def run_jobs(specs, pool):
    return pool.map(job, specs, chunksize=1)


# ---------------- A1 data (doc 120) ----------------
def a1_data():
    items = json.load(open(os.path.join(ROOT, "data_exp120_v2.json"), encoding="utf-8"))["items"]
    y = [it["label"] for it in items]
    diag = {e: [R120.diagram(it, e) for it in items] for e in ("E0", "E1m", "E1g")}
    tok = {"E0": [it["sentence"].split() for it in items],
           "E1g": [["R_" + it["gold"]["verb"]["root"], "P_" + it["gold"]["verb"]["pattern"], it["subject"],
                    "R_" + it["gold"]["obj"]["root"], "P_" + it["gold"]["obj"]["pattern"]] for it in items],
           "E1m": [["R_" + it["mle"][0]["root"], "P_" + it["mle"][0]["pattern"], it["subject"],
                    "R_" + it["mle"][2]["root"], "P_" + it["mle"][2]["pattern"]] for it in items]}
    return items, {"diag": diag, "tokens": tok, "y": y}


# ---------------- A2 helpers (BERT disambiguation) ----------------
BERT = None


def bert_map(sentence):
    toks = sentence.split(); res = {}
    for tok, d in zip(toks, BERT.disambiguate(toks)):
        a = d.analyses[0].analysis
        res[tok] = {"root": a.get("root"), "pattern": CASE.sub("", a.get("pattern") or ""), "pos": a.get("pos"), "asp": a.get("asp")}
    return res


def main():
    global BERT
    t0 = time.time()
    out_p = os.path.join(ROOT, "results_exp121.json"); assert not os.path.exists(out_p), "Rule 3"
    from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
    BERT = BERTUnfactoredDisambiguator.pretrained("msa")
    out = {"doc": "121 (67bfa7b)"}
    # ---- validity (a): BERT vs MLE on doc 120 constructed data ----
    items, A1 = a1_data()
    agree = Counter(); n = 0
    for it in items:
        bm = bert_map(it["sentence"]); toks = it["sentence"].split()
        v, o = bm[toks[0]], bm[toks[2]]
        agree["bert_verb_pattern"] += v["pattern"] == it["gold"]["verb"]["pattern"]
        agree["bert_obj_pattern"] += o["pattern"] == it["gold"]["obj"]["pattern"]
        agree["bert_obj_root"] += o["root"] == it["gold"]["obj"]["root"]
        agree["mle_verb_pattern"] += it["mle"][0]["pattern"] == it["gold"]["verb"]["pattern"]
        agree["mle_obj_pattern"] += it["mle"][2]["pattern"] == it["gold"]["obj"]["pattern"]
        n += 1
    out["validity_doc120"] = {k: c / n for k, c in agree.items()}
    print("validity doc120", out["validity_doc120"], flush=True)
    # ---- A2 data: doc 119 tense + WSD with BERT factorisation ----
    paper = json.load(open(os.path.join(R119.PAPER, "sentences.json"), encoding="utf-8"))
    tasks = {"tense": paper["TenseBinary"]}
    W = paper["WordSenseDisambiguation_v2"]
    for v in sorted({x["verb"] for x in W}):
        its = [x for x in W if x["verb"] == v]; ss = defaultdict(set)
        for x in its:
            ss[x["sentence"]].add(x["label"])
        tasks["wsd_" + v] = [x for x in its if len(ss[x["sentence"]]) == 1]
    a2 = {}
    R119.mle_map = bert_map                                     # factorise() now reads BERT analyses
    for name, its in tasks.items():
        sents = [x["sentence"] for x in its]; labs = [x["label"] for x in its]
        classes = sorted(set(labs)); y = [classes.index(l) for l in labs]
        d0 = R119.parse(sents, True); stats = Counter()
        d1 = [R119.factorise(d, s_, stats) for d, s_ in zip(d0, sents)]
        bms = [bert_map(s_) for s_ in sents]
        def e1tok(s_, bm):
            ts = []
            for t in s_.split():
                a = bm[t]
                if a["pos"] in R119.CONTENT and a["root"] and a["root"].count("#") < 2:
                    ts += ["R_" + a["root"], "P_" + a["pattern"]]
                else:
                    ts.append(t)
            return ts
        groups = [R119.verbless_key(s_)[0] for s_ in sents] if name == "tense" else sents
        info = {"e1_box_stats": dict(stats)}
        vroot = {}
        if name == "tense":
            cons = 0
            for s_, l, bm in zip(sents, labs, bms):
                vt = R119.verbless_key(s_)[1]; a = bm.get(vt, {}) if vt else {}
                vroot[s_] = a.get("root")
                cons += (l == "Tense_Past" and a.get("asp") == "p") or (l == "Tense_Pres" and a.get("asp") == "i")
            info["aspect_consistency"] = cons / len(sents)
            print("tense aspect consistency", info["aspect_consistency"], flush=True)
        a2[name] = {"sents": sents, "y": y, "groups": groups, "vroot": vroot, "info": info,
                    "data": {"diag": {"E0": d0, "E1": d1}, "tokens": {"E0": [s_.split() for s_ in sents], "E1": [e1tok(s_, b) for s_, b in zip(sents, bms)]},
                             "y": y, "shape_keyed": True}}
    BERT = None                                                  # not needed in workers
    G["a1"] = A1
    for name in a2:
        G[name] = a2[name]["data"]
    ctx = mp.get_context("fork")
    with ctx.Pool(6) as pool:
        # ---- conditioning for new classical arms (doc 120 split (i) fold 1, train loss) ----
        from sklearn.model_selection import StratifiedGroupKFold
        y1 = np.array(A1["y"]); groups1 = [it["sentence"] for it in items]
        f_tr, _ = next(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(items, y1, groups1))
        f_tr = f_tr.tolist(); cond = {}
        for arm in ("CL@4", "CL@8", "MLP"):
            specs, keys = [], []
            for lr, init in itertools.product((0.05, 0.1), (None, 1.0)):
                for enc in ("E0", "E1m"):
                    specs.append(("a1", arm, enc, f_tr, [], 0, {"lr": lr, "init": init})); keys.append((lr, init))
            res = run_jobs(specs, pool)
            loss = defaultdict(list)
            for k, r in zip(keys, res):
                loss[k].append(r["final_loss"])
            best = min(loss, key=lambda k: np.mean(loss[k]))
            cond[arm] = {"lr": best[0], "init": best[1]}
            out.setdefault("conditioning", {})[arm] = {"grid": {"lr=%s,init=%s" % k: float(np.mean(v)) for k, v in loss.items()}, "chosen": cond[arm]}
        cond["C@4"] = {"lr": 0.1, "init": 1.0}                   # exp120's registered choice
        cond["Q@4"] = {}; cond["R@4"] = {}
        print("conditioning", out["conditioning"], int(time.time() - t0), flush=True)
        # ---- A1 split (ii) ----
        arms = ("Q@4", "R@4", "C@4", "CL@4", "CL@8", "MLP")
        specs = []
        for off in range(3):
            tr, te = R120.split_ii(items, off)
            for seed in range(3):
                for arm in arms:
                    for enc in ("E0", "E1m", "E1g"):
                        specs.append(("a1", arm, enc, tr, te, 100 * off + seed, cond[arm]))
        res = run_jobs(specs, pool)
        per_item = defaultdict(lambda: defaultdict(list)); trains = defaultdict(list)
        for sp, r in zip(specs, res):
            arm, enc, seed = sp[1], sp[2], sp[5]; off = seed // 100
            trains[arm + "-" + enc].append(r["train_acc"])
            for i, pr in zip(r["te"], r["test_pred"]):
                per_item[arm + "-" + enc][(off, i)].append(int(pr == A1["y"][i]))
        acc = {k: {"test": float(np.mean([np.mean(v) for v in d.values()])), "train": float(np.mean(trains[k]))} for k, d in per_item.items()}
        gate = {}
        for enc in ("E1m", "E1g"):
            best_cls = max(("CL@4", "CL@8", "MLP"), key=lambda a: acc[a + "-" + enc]["test"])
            by_root = defaultdict(list)
            for (off, i), v in per_item["Q@4-" + enc].items():
                by_root[items[i]["gold"]["obj"]["root"]].append(np.mean(v) - np.mean(per_item[best_cls + "-" + enc][(off, i)]))
            by_root = {k: np.array(v) for k, v in by_root.items()}
            diff = float(np.concatenate(list(by_root.values())).mean()); ci = R120.cluster_boot(by_root)
            gate[enc] = {"best_classical": best_cls, "Q_minus_best_classical": diff, "CI": ci,
                         "complex_specific": bool(diff >= 0.05 and ci[0] > 0)}
        out["A1_split_ii"] = {"acc": acc, "gate": gate}
        print("A1", {k: (round(v["test"], 3), round(v["train"], 3)) for k, v in acc.items()}, gate, int(time.time() - t0), flush=True)
        # ---- A2 ----
        a2arms = ("Q@4", "R@4", "C@4", "CL@4", "MLP")
        out["A2"] = {}
        for name, T2 in a2.items():
            y = np.array(T2["y"]); nmin = min(Counter(T2["y"]).values()); nf = min(5, nmin, len(set(T2["groups"])))
            specs = []
            for seed in range(5):
                for tr, te in StratifiedGroupKFold(nf, shuffle=True, random_state=seed).split(T2["sents"], y, T2["groups"]):
                    for arm in a2arms:
                        for enc in ("E0", "E1"):
                            specs.append((name, arm, enc, tr.tolist(), te.tolist(), seed, cond[arm]))
            res = run_jobs(specs, pool)
            accs = defaultdict(list); split = defaultdict(lambda: defaultdict(list))
            for sp, r in zip(specs, res):
                k = sp[1] + "-" + sp[2]; accs[k].append(r["test_acc"]); accs[k + "_train"].append(r["train_acc"])
                if name == "tense":
                    seen = {T2["vroot"].get(T2["sents"][i]) for i in sp[3]}
                    for i, pr in zip(r["te"], r["test_pred"]):
                        split[k]["seen" if T2["vroot"].get(T2["sents"][i]) in seen else "unseen"].append(int(pr == y[i]))
            rec = {k: float(np.mean(v)) for k, v in accs.items()}; rec.update(T2["info"]); rec["n"] = len(y)
            if name == "tense":
                rec["root_split"] = {k: {s: (float(np.mean(v)), len(v)) for s, v in d.items()} for k, d in split.items()}
                rec["E1_minus_E0_unseen"] = {a: rec["root_split"][a + "-E1"].get("unseen", (np.nan,))[0] - rec["root_split"][a + "-E0"].get("unseen", (np.nan,))[0] for a in a2arms}
            out["A2"][name] = rec
            print("A2", name, {k: round(v, 3) for k, v in rec.items() if isinstance(v, float)}, int(time.time() - t0), flush=True)
            if name == "tense":
                print("   unseen deltas", rec["E1_minus_E0_unseen"], flush=True)
    out["wall_s"] = time.time() - t0
    json.dump(out, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print("done", int(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
