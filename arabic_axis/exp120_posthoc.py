# -*- coding: utf-8 -*-
"""exp120_posthoc.py -- UNREGISTERED diagnostics for doc 120 (reviewer-requested, after the registered run).
(a) dof-matched real models: C and R at n = 8, s = 2 on split (ii) (complex phases vs twice the real dof);
(b) conditioning sensitivity: C at the a-priori settings (lr 0.05, default init) on split (ii);
(c) below-chance E0: per-offset test majority-class rate.  Writes results_exp120_posthoc.json (Rule 3)."""
import json, os
import numpy as np
import exp120_run as R
import exp119_tn as T
out_p = os.path.join(R.ROOT, "results_exp120_posthoc.json"); assert not os.path.exists(out_p)
items = json.load(open(os.path.join(R.ROOT, "data_exp120_v2.json"), encoding="utf-8"))["items"]
y = np.array([it["label"] for it in items])
D = {e: [R.diagram(it, e) for it in items] for e in ("E0", "E1g", "E1m")}
res = {}
def run(tag, kind, dims, **kw):
    acc = {}
    for enc in ("E0", "E1m", "E1g"):
        a, trn = [], []
        for off in range(3):
            tr, te = R.split_ii(items, off)
            for seed in range(3):
                r = T.train_eval(D[enc], y.tolist(), tr, te, dims, kind, 100 * off + seed, **kw)
                a.append(r["test_acc"]); trn.append(r["train_acc"])
        acc[enc] = (float(np.mean(a)), float(np.mean(trn)))
    res[tag] = acc; print(tag, acc, flush=True)
run("C_n8_chosen", "C", {"n": 8, "s": 2}, lr=0.1, init=1.0)
run("R_n8", "R", {"n": 8, "s": 2})
run("C_n4_apriori", "C", {"n": 4, "s": 2})
run("C_n8_apriori", "C", {"n": 8, "s": 2})
maj = []
for off in range(3):
    tr, te = R.split_ii(items, off); f = y[te].mean(); maj.append({"offset": off, "frac_class1": float(f), "n": len(te)})
res["test_class_balance"] = maj; print(maj)
json.dump(res, open(out_p, "w", encoding="utf-8"), indent=1)
