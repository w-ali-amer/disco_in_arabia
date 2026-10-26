# -*- coding: utf-8 -*-
"""exp116_run.py -- doc 116 (REGISTERED 1243999): tighten the K = 3/4, C = 6, m = 2 two-bit classical bound.

A1: exp113 sub-MILP (class R[0]'s decoder fixed to rep) PLUS fibre-lex rows: for consecutive messages s, s+1 in the
same fibre of rep, L(s) <= L(s+1), L(s) = sum_{j=1}^{C-1} K^(C-1-j) pos_j(s), pos_j(s) = sum_a idx(a) d[R[j], s, a].
G-LEX-VALID: A1 on the nine K = 3 reps exp113 proved (30 min) must reach the known optimum within 1e-9.
A2: the six open K = 3 reps, plain exp113 formulation, 8 h.  One Pool(10), priority order (doc 116 s3).
Combination: per-rep upper = min over valid bounds; certified upper = max over reps; gate = doc 113's.
Writes results_exp116.json (Rule 3).
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


def solve(args):
    arm, K, rep, tl, lex = args
    pb = A100.problem(R110.uniform_rows(K)); X, Rr, A, P = pb["X"], pb["R"], pb["A"], pb["P"]
    C = len(Rr)
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
    for s in range(M):
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
    nlex = 0
    if lex:
        for s in range(M - 1):
            if rep[s] != rep[s + 1]:
                continue                                         # not in the same fibre
            row = {}
            for j in range(1, C):
                w = float(K ** (C - 1 - j)); r = Rr[j]
                for ai, a in enumerate(A[r]):
                    row[idx[("d", r, s, a)]] = row.get(idx[("d", r, s, a)], 0) + w * ai
                    row[idx[("d", r, s + 1, a)]] = row.get(idx[("d", r, s + 1, a)], 0) - w * ai
            rows.append(row); lo.append(-np.inf); hi.append(0); nlex += 1
    Am = lil_matrix((len(rows), n))
    for i, row in enumerate(rows):
        for j, v in row.items():
            Am[i, j] = v
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(Am.tocsr(), lo, hi), integrality=integ, bounds=Bounds(lb, ub),
               options={"time_limit": tl, "mip_rel_gap": 0.0, "disp": False})
    val = -res.fun if res.x is not None else None
    dual = -getattr(res, "mip_dual_bound", res.fun) if res.x is not None else None
    return {"arm": arm, "K": K, "rep": list(rep), "lex_rows": nlex, "value": val, "dual": dual,
            "proved": bool(res.status == 0), "status": int(res.status), "wall": time.time() - t0, "time_limit": tl}


def main():
    t0 = time.time()
    r113 = json.load(open("results_exp113.json"))
    subs113 = {(s["K"], tuple(s["rep"])): s for s in r113["subs"]}
    proved3 = [r for (K, r), s in subs113.items() if K == 3 and s["proved"]]
    open3 = [r for (K, r), s in subs113.items() if K == 3 and not s["proved"]]
    reps4 = list(itertools.combinations_with_replacement(range(4), M))
    jobs = ([("GLEX", 3, r, 1800, True) for r in proved3] + [("A1", 3, r, 28800, True) for r in open3] +
            [("A2", 3, r, 28800, False) for r in open3] + [("A1", 4, r, 7200, True) for r in reps4])
    print("jobs", {a: sum(j[0] == a and j[1] == k for j in jobs) for a, k in (("GLEX", 3), ("A1", 3), ("A2", 3), ("A1", 4))}, flush=True)
    subs = []
    with Pool(10) as pool:
        for res in pool.imap_unordered(solve, jobs):
            subs.append(res)
            print(res["arm"], res["K"], res["rep"], round(res["value"] or -1, 6), round(res["dual"] or -1, 6), res["proved"],
                  int(res["wall"]), int(time.time() - t0), flush=True)
    out = {"doc": "116 (1243999)", "subs": subs}
    gl = [s for s in subs if s["arm"] == "GLEX"]
    gchk = []
    for s in gl:
        known = subs113[(3, tuple(s["rep"]))]["value"]
        gchk.append({"rep": s["rep"], "known": known, "lex_value": s["value"], "lex_proved": s["proved"], "wall": s["wall"],
                     "reaches": s["value"] is not None and abs(s["value"] - known) < 1e-9,
                     "not_above": not s["proved"] or s["value"] <= known + 1e-9})
    glex_pass = all(g["reaches"] and g["not_above"] for g in gchk)
    out["G_LEX_VALID"] = {"pass": glex_pass, "checks": gchk}
    out["classical"], out["gate"] = {}, {}
    for K in (3, 4):
        reps = [r for (k, r) in subs113 if k == K]
        per = {}
        for r in reps:
            cands = []; vals = []; proved = False
            s = subs113[(K, r)]
            cands.append(s["value"] if s["proved"] else s["dual"]); vals.append(s["value"]); proved |= s["proved"]
            for t in subs:
                if t["K"] == K and tuple(t["rep"]) == r and t["arm"] != "GLEX" and (t["arm"] != "A1" or glex_pass):
                    cands.append(t["value"] if t["proved"] else t["dual"]); vals.append(t["value"]); proved |= t["proved"]
            per[str(list(r))] = {"upper": min(cands), "best_value": max(v for v in vals if v is not None), "proved": proved}
        allp = all(p["proved"] for p in per.values())
        best = max(p["best_value"] for p in per.values()); up = max(p["upper"] for p in per.values())
        q = r113["quantum"][str(K)]["lower_bound"]
        cref = best if allp else up
        out["classical"][K] = {"per_rep": per, "all_proved": allp, "exact": best if allp else None,
                               "best_incumbent": best, "certified_upper": up}
        out["gate"][K] = {"quantum_lower_bound": q, "reference": "exact" if allp else "certified_upper",
                          "margin": q - cref, "multi_referent_2q": bool(q - cref >= 0.05),
                          "certified_strict": bool(q - up > 0), "undetermined": bool(not allp and q - cref < 0.05 and q - best >= 0.05)}
        print("K", K, {k: v for k, v in out["classical"][K].items() if k != "per_rep"}, out["gate"][K], flush=True)
    out["wall_s"] = time.time() - t0
    out["provenance"] = C94.provenance()
    C94.write_new_json("results_exp116.json", out)


if __name__ == "__main__":
    main()
