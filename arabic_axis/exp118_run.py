# -*- coding: utf-8 -*-
"""exp118_run.py -- doc 118 (REGISTERED ffd6ec1): K = 4 two-bit cell with class-relabelling symmetry.

solve() = exp116's sub-MILP (rep-fixed R[0] decoder + fibre-lex rows) plus, if cmin, one row per class r != R[0]:
sum_a 5^(K-1-a) count_a(r) >= L_rep (R[0] has the lexicographically smallest decoder count vector).
G-CLASSMIN-VALID: 15 K = 3 reps, 30 min: max incumbent = 13/18 within 1e-9 and no proved per-rep value above the
exp113/116 optimum.  Then 35 K = 4 reps at 8 h (class-min + lex if the gate passes, else lex only).  Pool(9).
Per-rep upper = min over exp113 / exp116 / exp118 bounds; certified upper = max over reps.  Writes results_exp118.json.
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
POOL = 9


def solve(args):
    arm, K, rep, tl, lex, cmin = args
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
    ncm = 0
    if cmin:                                                     # class-min rows (doc 118 s1)
        Lrep = sum(5 ** (K - 1 - ai) * sum(1 for s in range(M) if rep[s] == ai) for ai in range(K))
        for j in range(1, C):
            r = Rr[j]; row = {}
            for ai, a in enumerate(A[r]):
                for s in range(M):
                    row[idx[("d", r, s, a)]] = float(5 ** (K - 1 - ai))
            rows.append(row); lo.append(float(Lrep)); hi.append(np.inf); ncm += 1
    Am = lil_matrix((len(rows), n))
    for i, row in enumerate(rows):
        for j, v in row.items():
            Am[i, j] = v
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(Am.tocsr(), lo, hi), integrality=integ, bounds=Bounds(lb, ub),
               options={"time_limit": tl, "mip_rel_gap": 0.0, "disp": False})
    val = -res.fun if res.x is not None else None
    dual = -getattr(res, "mip_dual_bound", res.fun) if res.x is not None else None
    return {"arm": arm, "K": K, "rep": list(rep), "lex_rows": nlex, "classmin_rows": ncm, "value": val, "dual": dual,
            "proved": bool(res.status == 0), "status": int(res.status), "wall": time.time() - t0, "time_limit": tl}


def bound(s):
    return s["value"] if s["proved"] else s["dual"]


def main():
    t0 = time.time()
    r113 = json.load(open("results_exp113.json")); r116 = json.load(open("results_exp116.json"))
    prior = {}                                                   # (K, rep) -> list of prior sub results
    for s in r113["subs"]:
        prior.setdefault((s["K"], tuple(s["rep"])), []).append(s)
    for s in r116["subs"]:
        if s["arm"] != "GLEX":
            prior.setdefault((s["K"], tuple(s["rep"])), []).append(s)
    reps3 = list(itertools.combinations_with_replacement(range(3), M))
    reps4 = list(itertools.combinations_with_replacement(range(4), M))
    known3 = {r: max(p["value"] for p in prior[(3, r)] if p["proved"]) for r in reps3}
    out = {"doc": "118 (ffd6ec1)"}
    with Pool(POOL) as pool:
        gsubs = []
        for res in pool.imap_unordered(solve, [("GCM", 3, r, 1800, True, True) for r in reps3]):
            gsubs.append(res); print("GCM", res["rep"], res["classmin_rows"], round(res["value"] or -1, 6),
                                     round(res["dual"] or -1, 6), res["proved"], int(res["wall"]), flush=True)
        gmax = max(s["value"] for s in gsubs if s["value"] is not None)
        above = [s["rep"] for s in gsubs if s["proved"] and s["value"] > known3[tuple(s["rep"])] + 1e-9]
        gpass = abs(gmax - 13 / 18) < 1e-9 and not above
        out["G_CLASSMIN_VALID"] = {"pass": gpass, "max_incumbent": gmax, "proved_above_known": above, "subs": gsubs}
        print("G_CLASSMIN_VALID", gpass, gmax, above, int(time.time() - t0), flush=True)
        subs = []
        for res in pool.imap_unordered(solve, [("K4", 4, r, 28800, True, gpass) for r in reps4]):
            subs.append(res); print("K4", res["rep"], round(res["value"] or -1, 6), round(res["dual"] or -1, 6),
                                    res["proved"], int(res["wall"]), int(time.time() - t0), flush=True)
    out["subs"] = subs
    per = {}
    for r in reps4:
        cands = [bound(p) for p in prior[(4, r)]] + [bound(s) for s in subs if tuple(s["rep"]) == r]
        vals = [p["value"] for p in prior[(4, r)]] + [s["value"] for s in subs if tuple(s["rep"]) == r]
        prov = any(p["proved"] for p in prior[(4, r)]) or any(s["proved"] for s in subs if tuple(s["rep"]) == r)
        per[str(list(r))] = {"upper": min(cands), "best_value": max(v for v in vals if v is not None), "proved": prov}
    allp = all(p["proved"] for p in per.values())
    best = max(p["best_value"] for p in per.values()); up = max(p["upper"] for p in per.values())
    q = r116["gate"]["4"]["quantum_lower_bound"]; cref = best if allp else up
    out["classical"] = {"per_rep": per, "all_proved": allp, "exact": best if allp else None, "best_incumbent": best,
                        "certified_upper": up, "rows": "classmin+lex" if gpass else "lex"}
    out["gate"] = {"quantum_lower_bound": q, "reference": "exact" if allp else "certified_upper", "margin": q - cref,
                   "multi_referent_2q": bool(q - cref >= 0.05), "certified_strict": bool(q - up > 0),
                   "needed_upper_for_true": q - 0.05}
    print("K4", {k: v for k, v in out["classical"].items() if k != "per_rep"}, out["gate"], flush=True)
    out["wall_s"] = time.time() - t0
    out["provenance"] = C94.provenance()
    C94.write_new_json("results_exp118.json", out)


if __name__ == "__main__":
    main()
