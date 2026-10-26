# -*- coding: utf-8 -*-
"""exp113_run.py -- doc 113 (REGISTERED 5722343): close the which-referent K = 3/4, C = 6, m = 2 cell.

Classical: exp100 MILP formulation, with class R[0]'s decoder FIXED to each non-decreasing map {4 messages} -> {K
positions} (message-relabelling representatives: 15 for K = 3, 35 for K = 4); HiGHS gap 0, 2 h each, 8 in parallel.
Classical exact = max over reps iff all proved; else certified upper bound = max over (proved value or dual).
Quantum: exp100 optimiser, D = 4, 256 restarts (lower bound).  Gate multi_referent_2q: LB - classical >= 0.05.
Writes results_exp113.json (Rule 3).
"""
import itertools, json, time
from multiprocessing import Pool
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
import exp94_common as C94
import exp100_accounting as A100
import exp110_run as R110

M = 4


def milp_fixed(args):
    K, rep, tl = args
    pb = A100.problem(R110.uniform_rows(K)); X, Rr, A, P = pb["X"], pb["R"], pb["A"], pb["P"]
    idx = {}
    def var(k):
        if k not in idx:
            idx[k] = len(idx)
        return idx[k]
    for x in X:
        for s in range(M):
            var(("e", x, s))
    for r in Rr:
        for s in range(M):
            for a in A[r]:
                var(("d", r, s, a))
    for (x, r, a) in P:
        for s in range(M):
            var(("z", x, r, a, s))
    n = len(idx); c = np.zeros(n); integ = np.zeros(n); lb = np.zeros(n); ub = np.ones(n)
    for k, i in idx.items():
        if k[0] in ("e", "d"):
            integ[i] = 1
        else:
            c[i] = -P[(k[1], k[2], k[3])]
    r0 = Rr[0]
    for s in range(M):                                       # fix class r0's decoder to the representative map
        for a in A[r0]:
            v = 1.0 if a == A[r0][rep[s]] else 0.0
            lb[idx[("d", r0, s, a)]] = v; ub[idx[("d", r0, s, a)]] = v
    rows, lo, hi = [], [], []
    for x in X:
        rows.append({idx[("e", x, s)]: 1 for s in range(M)}); lo.append(1); hi.append(1)
    for r in Rr:
        for s in range(M):
            rows.append({idx[("d", r, s, a)]: 1 for a in A[r]}); lo.append(1); hi.append(1)
    for (x, r, a) in P:
        for s in range(M):
            z = idx[("z", x, r, a, s)]
            rows.append({z: 1, idx[("e", x, s)]: -1}); lo.append(-np.inf); hi.append(0)
            rows.append({z: 1, idx[("d", r, s, a)]: -1}); lo.append(-np.inf); hi.append(0)
    Am = lil_matrix((len(rows), n))
    for i, row in enumerate(rows):
        for j, v in row.items():
            Am[i, j] = v
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(Am.tocsr(), lo, hi), integrality=integ, bounds=Bounds(lb, ub),
               options={"time_limit": tl, "mip_rel_gap": 0.0, "disp": False})
    val = -res.fun if res.x is not None else None
    dual = -getattr(res, "mip_dual_bound", res.fun) if res.x is not None else None
    return {"K": K, "rep": list(rep), "value": val, "dual": dual, "proved": bool(res.status == 0), "status": int(res.status),
            "wall": time.time() - t0}


def main():
    reps = {K: [r for r in itertools.combinations_with_replacement(range(K), M)] for K in (3, 4)}
    jobs = [(K, r, 7200) for K in (3, 4) for r in reps[K]]
    print("sub-MILPs", {K: len(v) for K, v in reps.items()}, flush=True)
    with Pool(8) as pool:
        subs = []
        for res in pool.imap_unordered(milp_fixed, jobs):
            subs.append(res); print(res["K"], res["rep"], round(res["value"] or -1, 6), round(res["dual"] or -1, 6), res["proved"], int(res["wall"]), flush=True)
    out = {"doc": "113 (5722343)", "subs": subs, "classical": {}, "quantum": {}, "gate": {}}
    A100.RESTARTS = 256
    for K in (3, 4):
        S = [s for s in subs if s["K"] == K]
        allp = all(s["proved"] for s in S)
        best = max(s["value"] for s in S)
        ubound = max((s["value"] if s["proved"] else s["dual"]) for s in S)
        out["classical"][K] = {"all_proved": allp, "exact": best if allp else None, "best_incumbent": best, "certified_upper": ubound}
        q = A100.quantum(A100.problem(R110.uniform_rows(K)), 2)
        out["quantum"][K] = {"lower_bound": q["value"], "G_POVM": q["G_POVM"], "restarts": 256}
        cref = best if allp else ubound
        margin = q["value"] - cref
        out["gate"][K] = {"reference": "exact" if allp else "certified_upper", "margin": margin,
                          "multi_referent_2q": bool(margin >= 0.05),
                          "undetermined": bool(not allp and q["value"] - best >= 0.05 and margin < 0.05)}
        print("K", K, out["classical"][K], out["quantum"][K], out["gate"][K], flush=True)
    out["provenance"] = C94.provenance()
    C94.write_new_json("results_exp113.json", out)


if __name__ == "__main__":
    main()
