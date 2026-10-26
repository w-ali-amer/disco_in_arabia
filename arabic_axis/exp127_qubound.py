# -*- coding: utf-8 -*-
"""exp127_qubound.py -- doc 126 (registered 7660eb0): certified level-1 Hilbert-Schmidt Gram upper bound for
prepare-and-measure games in dimension d with general POVMs.  Calibrations: (2,1)/(3,1) QRAC d=2, which-referent K=2 C=6
d=2; target K=3 C=6 d=4; K=4 C=6 d=4 descriptive.  Run with ~/qrao_env.  Writes results_exp127.json (Rule 3)."""
import itertools, json, math, os, sys
import numpy as np, cvxpy as cp

ROOT = os.path.dirname(os.path.abspath(__file__))


def game_qrac(n):
    X = list(itertools.product((0, 1), repeat=n)); w = 1.0 / (len(X) * n)
    return {"X": X, "dec": [2] * n, "pay": [(xi, i, x[i], w) for xi, x in enumerate(X) for i in range(n)]}


def game_wr(K, C=6):
    X = list(itertools.permutations(range(C), K)); w = 1.0 / (len(X) * K)
    return {"X": X, "dec": [K] * C, "pay": [(xi, v, pos, w) for xi, x in enumerate(X) for pos, v in enumerate(x)]}


def solve(g, d, solver="CLARABEL"):
    N = len(g["X"]); midx = {}; n = 1 + N
    for v, na in enumerate(g["dec"]):
        for a in range(na):
            midx[(v, a)] = n; n += 1
    I = 0; R = list(range(1, 1 + N)); M = list(midx.values())
    C = np.zeros((n, n))
    for xi, v, a, w in g["pay"]:
        i, j = 1 + xi, midx[(v, a)]; C[i, j] += w / 2; C[j, i] += w / 2
    # every constraint: (terms [(i, j, coef)], const)  meaning  sum coef*G[i,j] + const  (== 0 | <= 0)
    eq, le = [], []
    eq.append(([(I, I, 1.0)], -d))
    for r in R:
        eq.append(([(I, r, 1.0)], -1.0)); le.append(([(r, r, 1.0)], -1.0))
    for v, na in enumerate(g["dec"]):
        idx = [midx[(v, a)] for a in range(na)]
        for j in range(n):
            eq.append(([(k, j, 1.0) for k in idx] + [(I, j, -1.0)], 0.0))
    for m in M:
        le.append(([(m, m, 1.0), (I, m, -1.0)], 0.0)); le.append(([(I, m, -1.0)], 0.0)); le.append(([(I, m, 1.0)], -d))
    P = R + M
    for a_, b_ in itertools.combinations(P, 2):
        le.append(([(a_, b_, -1.0)], 0.0))
    for r in R:
        for m in M:
            le.append(([(r, m, 1.0)], -1.0)); le.append(([(r, m, 1.0), (I, m, -1.0)], 0.0))
    G = cp.Variable((n, n), symmetric=True)
    def expr(t):
        terms, c0 = t; return sum(c * G[i, j] for i, j, c in terms) + c0
    ce = [expr(t) == 0 for t in eq]; cl = [expr(t) <= 0 for t in le]
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(C, G))), [G >> 0] + ce + cl)
    prob.solve(solver=solver)
    def mat(terms):
        A = np.zeros((n, n))
        for i, j, c in terms:
            if i == j:
                A[i, i] += c
            else:
                A[i, j] += c / 2; A[j, i] += c / 2
        return A
    # Lagrangian bound: for feasible G (eq = 0, le <= 0, z >= 0, G PSD):
    #   <C,G> <= <C,G> - sum y*eq(G) - sum z*le(G) = <S,G> - sum y c - sum z c,  S = C - sum y A - sum z A
    #   <S,G> <= max(0, lam_max(S)) * tr G,  tr G <= T
    S = C.copy(); const = 0.0
    for t, c in zip(eq, ce):
        y = float(c.dual_value); S -= y * mat(t[0]); const -= y * t[1]
    for t, c in zip(le, cl):
        z = max(0.0, float(c.dual_value)); S -= z * mat(t[0]); const -= z * t[1]
    lam = float(np.linalg.eigvalsh((S + S.T) / 2).max())
    T = d + N + len(g["dec"]) * d
    corr = max(0.0, lam) * T
    return {"primal": float(prob.value), "status": prob.status, "certified_ub": const + corr + 1e-9, "dual_const": const,
            "lam_max_S": lam, "T": T, "correction": corr, "n": n, "n_eq": len(eq), "n_le": len(le), "solver": solver}


def main():
    out_j = os.path.join(ROOT, "results_exp127.json"); assert not os.path.exists(out_j), "Rule 3"
    cases = [("qrac21_d2", game_qrac(2), 2, 0.5 + 1 / (2 * math.sqrt(2))),
             ("qrac31_d2", game_qrac(3), 2, 0.5 + 1 / (2 * math.sqrt(3))),
             ("wr_K2_C6_d2", game_wr(2), 2, 0.5 + (2 * math.sqrt(2) + 1) / 10),
             ("wr_K3_C6_d4", game_wr(3), 4, 0.7781249235008523),
             ("wr_K4_C6_d4", game_wr(4), 4, 0.6314295355281454)]
    only = sys.argv[1:] or [c[0] for c in cases]
    # G-WEIGHTS: K=3 payoff equals exp100 problem weights
    sys.path.insert(0, ROOT)
    ref = json.load(open(os.path.join(ROOT, "exp127_weights_ref.json")))      # exp100 problem(uniform_rows(3)).P, dumped in qnlp_mac_env (torch)
    g3 = game_wr(3); mine = {(tuple(g3["X"][xi]), v, pos): w for xi, v, pos, w in g3["pay"]}
    L = ["ms", "fs", "md", "fd", "mp", "fp"]                                     # exp110_run.CLASSES (index = class id)
    gw = max(abs(mine[(tuple(L.index(c) for c in x), L.index(r), a)] - p) for x, r, a, p in ref) if len(mine) == len(ref) else 1.0
    res = {"doc": "126 (7660eb0)", "G_WEIGHTS_max_abs": gw, "G_WEIGHTS_pass": bool(gw < 1e-12), "cases": {}}
    print("G_WEIGHTS", gw, flush=True)
    for name, g, d, known in cases:
        if name not in only:
            continue
        r = solve(g, d, solver=os.environ.get("EXP127_SOLVER", "CLARABEL")); r["known"] = known; r["ub_minus_known"] = r["certified_ub"] - known
        r["G_SANITY"] = bool(r["certified_ub"] >= known - 1e-9); r["G_DUAL"] = bool(r["certified_ub"] >= r["primal"] - 1e-7 and r["correction"] < 1e-4)
        res["cases"][name] = r
        print(name, json.dumps({k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    if "wr_K3_C6_d4" in res["cases"]:
        res["qubound_k3_tight"] = bool(res["cases"]["wr_K3_C6_d4"]["certified_ub"] - 0.7781 <= 0.01)
    json.dump(res, open(out_j if only == [c[0] for c in cases] else out_j.replace(".json", "_partial_%s%s.json" % ("_".join(only), "" if os.environ.get("EXP127_SOLVER", "CLARABEL") == "CLARABEL" else "_" + os.environ["EXP127_SOLVER"])), "w"), indent=1)


if __name__ == "__main__":
    main()
