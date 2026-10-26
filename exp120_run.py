# -*- coding: utf-8 -*-
"""exp120_run.py -- doc 120 (REGISTERED 984ae87, A0 c049911): SRE factorial on data_exp120_v2.json.

Encodings: E0 surface boxes; E1m root+pattern boxes from CAMeL MLE (primary, the SRE operating point); E1g from gold
analyses (MLE object-pattern agreement 0.948 < 0.95, so both run).  Models (exp119_tn): Q complex/unitary, R real
unit-norm/orthogonal, C unconstrained real (the classical comparator).  Diagram: VSO, planar cups (verb s.n^l.n^l;
inner cup to the subject, outer to the object).
Step 0: C-free conditioning grid (lr x init) by FINAL TRAIN LOSS on split (i) fold 1 only.
Split (i): 5-fold StratifiedGroupKFold (group = sentence) x 3 seeds.  Split (ii): per root (across ALL verbs) one
form held out, round-robin over s/d/p with offset o in {0,1,2}; kept in test only if its (pattern, num) occurs >= 2
times in training with other roots (else returned to training); 3 offsets x 3 init seeds.
Gates on split (ii) (cluster bootstrap over roots, 2000): Delta_C, Delta_Q (E1m - E0), conversion gap; confound
branch (bag-of-patterns > 0.55); pattern-balanced stratum.  Baselines: LR on bag-of-forms / roots / patterns;
AraBERT frozen [CLS] + LR.  Writes results_exp120.json (Rule 3).
"""
import json, os, sys, time, itertools
from collections import Counter, defaultdict
import numpy as np
import torch
torch.set_num_threads(1)
from lambeq.backend.grammar import Ty, Box, Cup, Id
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import exp119_tn as T

ROOT = os.path.dirname(os.path.abspath(__file__))
n, s = Ty("n"), Ty("s"); VT = s @ n.l @ n.l
DIMS = {"n": 4, "s": 2}
KINDS = ("Q", "R", "C")
ENCS = ("E0", "E1m", "E1g")


def word(surface, ty, enc, root, pat):
    if enc == "E0":
        return Box(surface, Ty(), ty)
    return Box("R_" + root, Ty(), ty) >> Box("P_" + pat, ty, ty)


def diagram(it, enc):
    v = it["sentence"].split()[0]
    if enc == "E1g":
        vr, vp, orr, op = it["gold"]["verb"]["root"], it["gold"]["verb"]["pattern"], it["gold"]["obj"]["root"], it["gold"]["obj"]["pattern"]
    else:
        m = it["mle"]; vr, vp, orr, op = m[0]["root"], m[0]["pattern"], m[2]["root"], m[2]["pattern"]
    d = word(v, VT, enc, vr, vp) @ Box(it["subject"], Ty(), n) @ word(it["form"], n, enc, orr, op)
    return d >> (Id(s) @ Id(n.l) @ Cup(n.l, n) @ Id(n)) >> (Id(s) @ Cup(n.l, n))


def lr_acc(docs, y, tr, te):
    X = CountVectorizer(token_pattern=r"\S+").fit_transform(docs)
    m = LogisticRegression(max_iter=5000).fit(X[tr], y[tr])
    return (m.predict(X[te]) == y[te]).astype(int)


def split_ii(items, offset):
    roots = sorted({it["gold"]["obj"]["root"] for it in items})
    nums = ("s", "d", "p")
    have = defaultdict(set)
    for it in items:
        have[it["gold"]["obj"]["root"]].add(it["num"])
    hold = {}
    for k, r in enumerate(roots):
        for j in range(3):
            nm = nums[(k + offset + j) % 3]
            if nm in have[r]:
                hold[r] = nm; break
    test = [i for i, it in enumerate(items) if hold.get(it["gold"]["obj"]["root"]) == it["num"]]
    train = [i for i in range(len(items)) if i not in set(test)]
    pat_roots = defaultdict(set)
    for i in train:
        it = items[i]; pat_roots[(it["gold"]["obj"]["pattern"], it["num"])].add(it["gold"]["obj"]["root"])
    keep, back = [], []
    for i in test:
        it = items[i]; others = pat_roots[(it["gold"]["obj"]["pattern"], it["num"])] - {it["gold"]["obj"]["root"]}
        cnt = sum(1 for j in train if (items[j]["gold"]["obj"]["pattern"], items[j]["num"]) == (it["gold"]["obj"]["pattern"], it["num"])
                  and items[j]["gold"]["obj"]["root"] != it["gold"]["obj"]["root"])
        (keep if cnt >= 2 else back).append(i)
    return sorted(train + back), sorted(keep)


def cluster_boot(diff_by_root, B=2000, seed=0):
    rng = np.random.default_rng(seed); roots = list(diff_by_root)
    means = []
    for _ in range(B):
        pick = rng.choice(len(roots), len(roots), replace=True)
        vals = np.concatenate([diff_by_root[roots[p]] for p in pick])
        means.append(vals.mean())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    t0 = time.time()
    out_p = os.path.join(ROOT, "results_exp120.json"); assert not os.path.exists(out_p), "Rule 3"
    data = json.load(open(os.path.join(ROOT, "data_exp120_v2.json"), encoding="utf-8"))
    items = data["items"]; y = np.array([it["label"] for it in items])
    D = {e: [diagram(it, e) for it in items] for e in ENCS}
    out = {"doc": "120 (984ae87, A0 c049911)", "n_items": len(items)}
    # ---- step 0: C-free conditioning on split (i) fold 1, train loss only ----
    groups = [it["sentence"] for it in items]
    f1_tr, f1_te = next(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(items, y, groups))
    grid = {}
    for lr, init in itertools.product((0.05, 0.1), (None, 1.0)):
        losses = [T.train_eval(D[e], y.tolist(), f1_tr.tolist(), [], DIMS, "C", 0, lr=lr, init=init)["final_loss"] for e in ("E0", "E1m")]
        grid["lr=%s,init=%s" % (lr, init)] = float(np.mean(losses))
    best = min(grid, key=grid.get); c_lr = float(best.split(",")[0][3:]); c_init = None if best.endswith("None") else 1.0
    out["c_conditioning"] = {"grid_final_train_loss": grid, "chosen": best}
    print("C conditioning", grid, "->", best, int(time.time() - t0), flush=True)

    def run(kind, enc, tr, te, seed):
        kw = {"lr": c_lr, "init": c_init} if kind == "C" else {}
        return T.train_eval(D[enc], y.tolist(), tr, te, DIMS, kind, seed, **kw)
    # ---- split (i) ----
    si = defaultdict(list)
    for seed in range(3):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(items, y, groups):
            for kind in KINDS:
                for enc in ENCS:
                    r = run(kind, enc, tr.tolist(), te.tolist(), seed)
                    si[kind + "-" + enc].append((r["test_acc"], r["train_acc"]))
    out["split_i"] = {k: {"test": float(np.mean([a for a, _ in v])), "train": float(np.mean([b for _, b in v])), "runs": len(v)} for k, v in si.items()}
    print("split (i)", {k: (round(v["test"], 3), round(v["train"], 3)) for k, v in out["split_i"].items()}, int(time.time() - t0), flush=True)
    # ---- split (ii) ----
    per_item = defaultdict(lambda: defaultdict(list))     # arm -> item index -> list of correctness
    trains = defaultdict(list); sizes = []; base = defaultdict(list); strat = defaultdict(list)
    docs_form = [it["sentence"] for it in items]
    docs_root = [" ".join([it["gold"]["verb"]["root"], it["subject"], it["gold"]["obj"]["root"]]) for it in items]
    docs_pat = [" ".join([it["gold"]["verb"]["pattern"], it["subject"], it["gold"]["obj"]["pattern"] + "|" + it["num"]]) for it in items]
    for off in range(3):
        tr, te = split_ii(items, off); sizes.append({"offset": off, "train": len(tr), "test": len(te),
                                                       "test_roots": len({items[i]["gold"]["obj"]["root"] for i in te})})
        lab_by_pat = defaultdict(set)
        for i in tr:
            lab_by_pat[(items[i]["gold"]["obj"]["pattern"], items[i]["num"])].add(items[i]["label"])
        strat[off] = [i for i in te if len(lab_by_pat[(items[i]["gold"]["obj"]["pattern"], items[i]["num"])]) == 2]
        for nm, docs in (("bag_of_forms", docs_form), ("bag_of_roots", docs_root), ("bag_of_patterns", docs_pat)):
            base[nm].append(float(lr_acc(docs, y, np.array(tr), np.array(te)).mean()))
        for seed in range(3):
            for kind in KINDS:
                for enc in ENCS:
                    r = run(kind, enc, tr, te, 100 * off + seed)
                    trains[kind + "-" + enc].append(r["train_acc"])
                    for i, pr in zip(te, r["test_pred"]):
                        per_item[kind + "-" + enc][(off, i)].append(int(pr == y[i]))
        print("split (ii) offset", off, sizes[-1], "stratum", len(strat[off]), int(time.time() - t0), flush=True)
    # AraBERT frozen [CLS] + LR
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02"); bm = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv02").eval()
    with torch.no_grad():
        E = np.concatenate([bm(**tok(docs_form[k:k + 64], return_tensors="pt", padding=True)).last_hidden_state[:, 0].numpy()
                            for k in range(0, len(docs_form), 64)])
    for off in range(3):
        tr, te = split_ii(items, off)
        m = LogisticRegression(max_iter=5000).fit(E[tr], y[tr]); base["arabert_frozen"].append(float((m.predict(E[te]) == y[te]).mean()))
    arms = {}
    for arm, d in per_item.items():
        allc = [np.mean(v) for v in d.values()]
        arms[arm] = {"test": float(np.mean(allc)), "train": float(np.mean(trains[arm]))}
    out["split_ii"] = {"sizes": sizes, "arms": arms, "baselines": {k: float(np.mean(v)) for k, v in base.items()},
                       "baselines_per_offset": dict(base), "stratum_sizes": {o: len(v) for o, v in strat.items()}}

    def delta(kind, enc1="E1m", subset=None):
        by_root = defaultdict(list)
        for (off, i), v in per_item[kind + "-" + enc1].items():
            if subset is not None and i not in subset[off]:
                continue
            v0 = per_item[kind + "-E0"][(off, i)]
            by_root[items[i]["gold"]["obj"]["root"]].append(np.mean(v) - np.mean(v0))
        by_root = {r: np.array(v) for r, v in by_root.items()}
        if not by_root:
            return None, None, by_root
        allv = np.concatenate(list(by_root.values()))
        return float(allv.mean()), cluster_boot(by_root), by_root

    def gap(enc1="E1m", subset=None):
        _, _, bc = delta("C", enc1, subset); _, _, bq = delta("Q", enc1, subset)
        g = {r: bc[r] - bq[r] for r in bc if r in bq}
        allv = np.concatenate(list(g.values())) if g else np.array([])
        return (float(allv.mean()) if len(allv) else None), (cluster_boot(g) if g else None)
    gates = {}
    for enc1 in ("E1m", "E1g"):
        dc, cic, _ = delta("C", enc1); dq, ciq, _ = delta("Q", enc1); dr, cir, _ = delta("R", enc1); gp, cig = gap(enc1)
        sc, scc, _ = delta("C", enc1, strat); sq, sqc, _ = delta("Q", enc1, strat); sg, sgc = gap(enc1, strat)
        gates[enc1] = {"Delta_C": dc, "Delta_C_CI": cic, "Delta_Q": dq, "Delta_Q_CI": ciq, "Delta_R": dr, "Delta_R_CI": cir,
                       "gap": gp, "gap_CI": cig,
                       "sre_converts_classically": bool(dc is not None and dc >= 0.05),
                       "sre_converts_quantumly": bool(dq is not None and dq >= 0.05),
                       "conversion_gap": bool(gp is not None and gp >= 0.05 and cig[0] > 0),
                       "stratum": {"Delta_C": sc, "Delta_C_CI": scc, "Delta_Q": sq, "Delta_Q_CI": sqc, "gap": sg, "gap_CI": sgc}}
    c_train_i = out["split_i"]["C-E1m"]["train"]
    out["gates"] = gates
    out["confounded"] = bool(out["split_ii"]["baselines"]["bag_of_patterns"] > 0.55)
    out["C_free_interpretable"] = bool(c_train_i >= 0.95)
    out["wall_s"] = time.time() - t0
    json.dump(out, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print("arms", {k: (round(v["test"], 3), round(v["train"], 3)) for k, v in arms.items()})
    print("baselines", out["split_ii"]["baselines"]); print("gates", json.dumps(gates, default=float)[:1500])
    print("confounded", out["confounded"], "C interpretable", out["C_free_interpretable"], flush=True)


if __name__ == "__main__":
    main()
