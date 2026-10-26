# -*- coding: utf-8 -*-
"""exp127b_qubound_vec.py -- doc 126 §9: the exp127 level-1 HS-Gram relaxation with VECTORISED constraints (same set) and
the same Lagrangian certificate (S assembled entrywise from the block duals).  G-REPRO on the calibrations and K = 3, then
K = 4 (descriptive).  Run with ~/qrao_env.  Writes results_exp127b.json (Rule 3)."""
import json, math, os, sys, time
import numpy as np, cvxpy as cp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp127_qubound import game_qrac, game_wr

ROOT = os.path.dirname(os.path.abspath(__file__))


def solve(g, d, solver=os.environ.get("EXP127_SOLVER", "CLARABEL")):
    N = len(g["X"]); dec = g["dec"]; n = 1 + N + sum(dec)
    R = slice(1, 1 + N); M = slice(1 + N, n); P = slice(1, n); nM = n - 1 - N
    groups, k = [], 1 + N
    for na in dec:
        groups.append(list(range(k, k + na))); k += na
    C = np.zeros((n, n)); midx = {}
    k = 1 + N
    for v, na in enumerate(dec):
        for a in range(na):
            midx[(v, a)] = k; k += 1
    for xi, v, a, w in g["pay"]:
        i, j = 1 + xi, midx[(v, a)]; C[i, j] += w / 2; C[j, i] += w / 2
    G = cp.Variable((n, n), symmetric=True); dg = cp.diag(G)
    cons = {
        "e_II": G[0, 0] - d == 0,
        "e_trR": G[0, R] - 1 == 0,
        "l_purR": dg[R] - 1 <= 0,
        "e_povm": [cp.sum(G[grp, :], axis=0) - G[0, :] == 0 for grp in groups],
        "l_MM": dg[M] - G[0, M] <= 0,
        "l_trMpos": -G[0, M] <= 0,
        "l_trMmax": G[0, M] - d <= 0,
        "l_pos": -G[P, P] <= 0,
        "l_RM1": G[R, M] - 1 <= 0,
        "l_RMtr": G[R, M] - cp.reshape(G[0, M], (1, nM), order="C") <= 0,
    }
    flat = [cons[k] for k in cons if k != "e_povm"] + cons["e_povm"]
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(C, G))), [G >> 0] + flat)
    t0 = time.time()
    kw = {"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iters": 2000000} if (solver == "SCS" and os.environ.get("EXP127_TIGHT")) else {}
    prob.solve(solver=solver, **kw); wall = time.time() - t0
    # ---- certificate: <C,G> <= <S,G> + const, S = C - sym(Grad);  G PSD, tr G <= T
    Gr = np.zeros((n, n)); const = 0.0
    dv = lambda c: np.asarray(c.dual_value, dtype=float)
    y = float(dv(cons["e_II"])); Gr[0, 0] += y; const -= y * (-d)
    y = dv(cons["e_trR"]).ravel(); Gr[0, R] += y; const -= (y * -1).sum()
    z = np.clip(dv(cons["l_purR"]).ravel(), 0, None); idx = np.arange(1, 1 + N); Gr[idx, idx] += z; const -= (z * -1).sum()
    for grp, c in zip(groups, cons["e_povm"]):
        y = dv(c).ravel()
        for kk in grp:
            Gr[kk, :] += y
        Gr[0, :] -= y
    z = np.clip(dv(cons["l_MM"]).ravel(), 0, None); im = np.arange(1 + N, n); Gr[im, im] += z; Gr[0, M] -= z
    z = np.clip(dv(cons["l_trMpos"]).ravel(), 0, None); Gr[0, M] -= z
    z = np.clip(dv(cons["l_trMmax"]).ravel(), 0, None); Gr[0, M] += z; const -= (z * -d).sum()
    Z = np.clip(dv(cons["l_pos"]), 0, None); Gr[P, P] -= Z
    Z = np.clip(dv(cons["l_RM1"]), 0, None); Gr[R, M] += Z; const -= (Z * -1).sum()
    Z = np.clip(dv(cons["l_RMtr"]), 0, None); Gr[R, M] += Z; Gr[0, M] -= Z.sum(axis=0)
    S = C - (Gr + Gr.T) / 2
    lam = float(np.linalg.eigvalsh(S).max()); T = d + N + len(dec) * d; corr = max(0.0, lam) * T
    return {"primal": float(prob.value), "status": prob.status, "certified_ub": const + corr + 1e-9, "dual_const": const,
            "lam_max_S": lam, "T": T, "correction": corr, "n": n, "wall_s": wall, "solver": solver}


def main():
    out_j = os.path.join(ROOT, "results_exp127e_scs_tight.json") if os.environ.get("EXP127_TIGHT") else os.path.join(ROOT, ("results_exp127c_%s.json" % os.environ.get("EXP127_SOLVER", "CLARABEL")) if sys.argv[1:] == ["k4only"] else "results_exp127b.json"); assert not os.path.exists(out_j), "Rule 3"
    prev = json.load(open(os.path.join(ROOT, "results_exp127_partial_wr_K3_C6_d4.json")))["cases"]["wr_K3_C6_d4"]["certified_ub"]
    cases = [("qrac21_d2", game_qrac(2), 2, 0.5 + 1 / (2 * math.sqrt(2)), 0.8535537), ("qrac31_d2", game_qrac(3), 2, 0.5 + 1 / (2 * math.sqrt(3)), 0.7886752),
             ("wr_K2_C6_d2", game_wr(2), 2, 0.5 + (2 * math.sqrt(2) + 1) / 10, 0.8872984), ("wr_K3_C6_d4", game_wr(3), 4, 0.7781249235008523, prev),
             ("wr_K4_C6_d4", game_wr(4), 4, 0.6314295355281454, None)]
    res = {"doc": "126 §9 (vectorised)", "cases": {}}
    if sys.argv[1:] == ["k4only"]:                                            # doc 126 §9 amendment (G-REPRO' met in exp127b)
        cases = cases[-1:]
    for name, g, d, known, ref in cases:
        r = solve(g, d); r["known_lower"] = known; r["G_SANITY"] = bool(r["certified_ub"] >= known - 1e-9)
        r["G_DUAL"] = bool(r["certified_ub"] >= r["primal"] - 1e-7 and r["correction"] < 1e-4)
        if ref is not None:
            r["exp127_ub"] = ref
            if os.environ.get("EXP127_TIGHT"):                                    # doc 126 §9 amendment G-REPRO': primal agreement
                ref_primal = {"wr_K3_C6_d4": 0.8201975}.get(name)
                r["G_REPRO"] = bool(abs(r["primal"] - (ref_primal if ref_primal else r["primal"])) <= 1e-6 and r["certified_ub"] >= ref_primal - 1e-6 if ref_primal else abs(r["certified_ub"] - ref) <= 2e-6)
            else:
                r["G_REPRO"] = bool(abs(r["certified_ub"] - ref) <= (1e-5 if name.startswith("wr_K3") else 1e-6))
        res["cases"][name] = r
        print(name, json.dumps({k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        if ref is not None and not r["G_REPRO"]:
            res["stopped"] = "G-REPRO failed at " + name; break
        json.dump(res, open(out_j + ".partial", "w"), indent=1)
    json.dump(res, open(out_j, "w"), indent=1); os.path.exists(out_j + ".partial") and os.remove(out_j + ".partial")


if __name__ == "__main__":
    main()
