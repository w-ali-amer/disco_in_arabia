# -*- coding: utf-8 -*-
"""exp134_run.py -- doc 134: certified upper bound on Eve's guessing probability (classical side information, SDI randomness)
for the which-referent two-qubit strategies, via the homogenised exp127 level-1 HS-Gram relaxation split into guess blocks.
All constraints live in ONE sparse affine system (A x + c <= 0 / == 0) used both by the solver and by the certificate.
usage: qnlp_mac_env/bin/python -I exp134_run.py stats  -> exp134_logs/stats.npz (statistics of the exp133 strategies)
       qrao_env/bin/python     -I exp134_run.py run    -> results_exp134.json (Rule 3)"""
import hashlib, json, math, os, subprocess, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import itertools


def game_qrac(n):                                                                # verbatim copy of exp127_qubound.game_qrac (no cvxpy here)
    X = list(itertools.product((0, 1), repeat=n)); w = 1.0 / (len(X) * n)
    return {"X": X, "dec": [2] * n, "pay": [(xi, i, x[i], w) for xi, x in enumerate(X) for i in range(n)]}


def game_wr(K, C=6):                                                             # verbatim copy of exp127_qubound.game_wr
    X = list(itertools.permutations(range(C), K)); w = 1.0 / (len(X) * K)
    return {"X": X, "dec": [K] * C, "pay": [(xi, v, pos, w) for xi, x in enumerate(X) for pos, v in enumerate(x)]}

import exp94_common as C94

LOGS = os.path.join(ROOT, "exp134_logs"); STATS = os.path.join(LOGS, "stats.npz"); OUT = os.path.join(ROOT, "results_exp134.json")
L = ["ms", "fs", "md", "fd", "mp", "fp"]                                          # exp110_run.CLASSES (index = class id), as exp127
P_Q = {3: 0.7781249235008523, 4: 0.6314295355281454}
REPRO = {"qrac21_d2": 0.8535534, "qrac31_d2": 0.7886751, "wr_K2_C6_d2": 0.8872983, "wr_K3_C6_d4": 0.8201978, "wr_K4_C6_d4": 0.6607919}
EPS = 1e-7


# ---------------------------------------------------------------- stats (qnlp_mac_env)
def stats():
    import exp100_accounting as A100, exp110_run as R110
    assert not os.path.exists(STATS), "Rule 3"
    z = np.load(os.path.join(ROOT, "exp133_strategies.npz")); out = {}
    for K in (3, 4):
        pb = A100.problem(R110.uniform_rows(K)); g = game_wr(K); xi = {x: i for i, x in enumerate(g["X"])}
        psi = z["K%d_psi" % K]; Pm = np.zeros((len(g["X"]), 6, K)); Etr = np.zeros((6, K))
        for i, x in enumerate(pb["X"]):
            xx = tuple(L.index(c) for c in x)
            for r in x:
                E = z["K%d_E_%s" % (K, r)]
                for ai, a in enumerate(pb["A"][r]):
                    Pm[xi[xx], L.index(r), a] = float(np.real(psi[i].conj() @ E[ai] @ psi[i]))
        for r in pb["R"]:
            E = z["K%d_E_%s" % (K, r)]
            for ai, a in enumerate(pb["A"][r]):
                Etr[L.index(r), a] = float(np.real(np.trace(E[ai])))
        val = sum(w * Pm[x, v, pos] for x, v, pos, w in g["pay"])
        print("K", K, "value", repr(val), "stored", P_Q[K], "diff", abs(val - P_Q[K]), flush=True)
        out["K%d_P" % K] = Pm; out["K%d_Etr" % K] = Etr
    os.makedirs(LOGS, exist_ok=True); np.savez(STATS, **out)


# ---------------------------------------------------------------- sparse affine system
class Sys:
    def __init__(s, n, nb):
        s.n, s.nb = n, nb; s.nv = nb * n * n
        s.rows = {"ineq": [[], [], [], []], "eq": [[], [], [], []]}               # r, col, val ; const list
        s.m = {"ineq": 0, "eq": 0}

    def e(s, b, i, j):
        return b * s.n * s.n + i + j * s.n                                       # F-order vec, block offset

    def add(s, kind, terms, const):
        R = s.rows[kind]; r = s.m[kind]
        for col, val in terms:
            R[0].append(r); R[1].append(col); R[2].append(val)
        R[3].append(const); s.m[kind] += 1

    def mats(s):
        import scipy.sparse as sp
        out = {}
        for k in ("ineq", "eq"):
            R = s.rows[k]
            out[k] = (sp.csr_matrix((R[2], (R[0], R[1])), shape=(s.m[k], s.nv)), np.array(R[3], dtype=float))
        return out


def layout(g):
    N = len(g["X"]); n = 1 + N + sum(g["dec"]); midx = {}; groups = []; k = 1 + N
    for v, na in enumerate(g["dec"]):
        groups.append(list(range(k, k + na)))
        for a in range(na):
            midx[(v, a)] = k; k += 1
    return N, n, midx, groups


def build(g, d, nb, coupling, probe=None):
    """coupling: ('none',) single-block game value; ('stats', [(xi, v, b, p)]); ('avg', W).  probe = (xi, v)."""
    N, n, midx, groups = layout(g); S = Sys(n, nb); Ms = range(1 + N, n); Rs = range(1, 1 + N)
    for b in range(nb):
        e = lambda i, j: S.e(b, i, j); q = [(e(0, 0), -1.0 / d)]
        for r in Rs:
            S.add("eq", [(e(0, r), 1.0)] + q, 0.0)                                # tr rho = q
            S.add("ineq", [(e(r, r), 1.0)] + q, 0.0)                              # tr rho^2 <= q
        for grp in groups:
            for j in range(n):
                S.add("eq", [(e(k, j), 1.0) for k in grp] + [(e(0, j), -1.0)], 0.0)
        for m in Ms:
            S.add("ineq", [(e(m, m), 1.0), (e(0, m), -1.0)], 0.0)                 # tr M^2 <= tr M
            S.add("ineq", [(e(0, m), -1.0)], 0.0)                                 # tr M >= 0
            S.add("ineq", [(e(0, m), 1.0), (e(0, 0), -1.0)], 0.0)                 # tr M <= d q
        for i in range(1, n):
            for j in range(i, n):
                S.add("ineq", [(e(i, j), -1.0)], 0.0)                             # tr(O_i O_j) >= 0
        for r in Rs:
            for m in Ms:
                S.add("ineq", [(e(r, m), 1.0)] + q, 0.0)                          # tr rho M <= q
                S.add("ineq", [(e(r, m), 1.0), (e(0, m), -1.0)], 0.0)             # tr rho M <= tr M
    S.add("eq", [(S.e(b, 0, 0), 1.0) for b in range(nb)], -float(d))              # sum_b q_b = 1
    C = np.zeros((n, n))
    for xi, v, a, w in g["pay"]:
        i, j = 1 + xi, midx[(v, a)]; C[i, j] += w / 2; C[j, i] += w / 2
    cvec = C.ravel(order="F"); obj = np.zeros(S.nv)
    if coupling[0] == "none":
        obj[:] = np.tile(cvec, nb)
    else:
        xs, vs = probe
        for b in range(nb):
            obj[S.e(b, 1 + xs, midx[(vs, b)])] += 0.5; obj[S.e(b, midx[(vs, b)], 1 + xs)] += 0.5
        if coupling[0] == "stats":
            for xi, v, bb, p in coupling[1]:
                t = [(S.e(b, 1 + xi, midx[(v, bb)]), 1.0) for b in range(nb)]
                S.add("ineq", t, -(p + EPS)); S.add("ineq", [(c, -1.0) for c, _ in t], p - EPS)
        elif coupling[0] == "avg":
            nz = np.nonzero(cvec)[0]
            S.add("ineq", [(b * n * n + k, -cvec[k]) for b in range(nb) for k in nz], float(coupling[1]))
    return S, obj, n, N, groups


def solve(g, d, nb, coupling, probe=None, tag="", solver="SCS", save=None, extra=None):
    import cvxpy as cp
    S, obj, n, N, groups = build(g, d, nb, coupling, probe); A = S.mats()
    Gs = [cp.Variable((n, n), symmetric=True) for _ in range(nb)]
    x = cp.hstack([cp.vec(G, order="F") for G in Gs])
    ci = A["ineq"][0] @ x + A["ineq"][1] <= 0; ce = A["eq"][0] @ x + A["eq"][1] == 0
    prob = cp.Problem(cp.Maximize(obj @ x), [G >> 0 for G in Gs] + [ci, ce])
    t0 = time.time()
    if solver == "SCS":
        prob.solve(solver="SCS", eps_abs=1e-9, eps_rel=1e-9, max_iters=2000000, **(extra or {}))
    elif solver == "CLARABEL_TIGHT":                                              # doc 134 §12 amendment: tightened Clarabel
        prob.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, tol_ktratio=1e-8, max_iter=1000)
    else:                                                                         # doc 134 §10 amendment: Clarabel, default settings
        prob.solve(solver=solver)
    wall = time.time() - t0
    yi = np.clip(np.asarray(ci.dual_value, dtype=float).ravel(), 0, None); ye = np.asarray(ce.dual_value, dtype=float).ravel()
    s = obj - A["ineq"][0].T @ yi - A["eq"][0].T @ ye; const = -(yi @ A["ineq"][1]) - (ye @ A["eq"][1])
    Sb = [s[b * n * n:(b + 1) * n * n].reshape((n, n), order="F") for b in range(nb)]; Sb = [(M + M.T) / 2 for M in Sb]
    lam = max(float(np.linalg.eigvalsh(M).max()) for M in Sb); T = d + N + len(groups) * d; corr = max(0.0, lam) * T
    # G-LAGR: L(x) computed directly == const + sum_b <S_b, G_b> on random symmetric blocks
    rng = np.random.default_rng(134); worst = 0.0
    for _ in range(5):
        Gr = [rng.standard_normal((n, n)) for _ in range(nb)]; Gr = [(M + M.T) / 2 for M in Gr]
        xv = np.concatenate([M.ravel(order="F") for M in Gr])
        Ld = obj @ xv - yi @ (A["ineq"][0] @ xv + A["ineq"][1]) - ye @ (A["eq"][0] @ xv + A["eq"][1])
        Lc = const + sum(float((Sm * M).sum()) for Sm, M in zip(Sb, Gr))
        worst = max(worst, abs(Ld - Lc) / max(1.0, abs(Ld)))
    r = {"primal": float(prob.value), "status": prob.status, "certified_ub": float(const + corr + 1e-9), "dual_const": float(const),
         "lam_max_S": lam, "T": T, "correction": corr, "n": n, "blocks": nb, "m_ineq": int(S.m["ineq"]), "m_eq": int(S.m["eq"]),
         "wall_s": wall, "G_LAGR_rel": worst, "G_LAGR": bool(worst < 1e-9), "solver": solver}
    r["G_DUAL"] = bool(r["certified_ub"] >= r["primal"] - 1e-7 and corr < 1e-4)
    if save:                                                                      # doc 134 §16: keep primal blocks and duals for offline checks
        np.savez(save, **{"G%d" % b: Gs[b].value for b in range(nb)}, y_ineq=yi, y_eq=ye, obj=obj)
    print(tag, json.dumps({k: (round(v, 9) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    return r


# ---------------------------------------------------------------- strategies -> statistics
def qrac21_stats(kind):
    sz = np.diag([1.0, -1.0]); sx = np.array([[0, 1.0], [1.0, 0]]); I = np.eye(2)
    g = game_qrac(2); rows = []
    for xi, x in enumerate(g["X"]):
        if kind == "opt":
            rho = (I + ((-1) ** x[1] * sx + (-1) ** x[0] * sz) / math.sqrt(2)) / 2; ops = [sz, sx]
        else:                                                                     # deterministic: store x0, answer x0 to both queries
            rho = (I + (-1) ** x[0] * sz) / 2; ops = [sz, sz]
        for v in (0, 1):
            for b in (0, 1):
                rows.append((xi, v, b, float(np.trace(rho @ (I + (-1) ** b * ops[v]) / 2))))
    return g, rows


def wr_rows(K, Pm, Etr, vis):
    g = game_wr(K); rows = []
    for xi, x in enumerate(g["X"]):
        for v in x:
            for b in range(K):
                rows.append((xi, v, b, vis * Pm[xi, v, b] + (1 - vis) * Etr[v, b] / 4))
    return g, rows


def run():
    assert not os.path.exists(OUT), "Rule 3"
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = {"doc": "134 (exp134)", "git_head": head, "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
           "eps": EPS, "gates": {}, "repro": {}, "cal": {}, "cases": {}, "secondary": {}, "bisect": {}}
    t0 = time.time(); certs = []
    def stop(why):
        res["stopped"] = why; res["wall_s"] = time.time() - t0; C94.write_new_json(OUT, res); print("STOP", why, flush=True); sys.exit(0)
    import exp127_qubound as E127                                                 # the copies must equal exp127's games
    res["gates"]["G_GAMES_EQ_EXP127"] = bool(all(game_wr(K) == E127.game_wr(K) for K in (2, 3, 4)) and all(game_qrac(k) == E127.game_qrac(k) for k in (2, 3)))
    # G-SYM
    rng = np.random.default_rng(1340); ok = True
    for K in (3, 4):
        g = game_wr(K); pay = {(g["X"][xi], v, a): w for xi, v, a, w in g["pay"]}
        for _ in range(3):
            pc = rng.permutation(6); pp = rng.permutation(K)
            for (x, v, a), w in pay.items():
                y = [None] * K
                for i in range(K):
                    y[pp[i]] = int(pc[x[i]])
                ok &= abs(pay.get((tuple(y), int(pc[v]), int(pp[a])), -1) - w) == 0
    res["gates"]["G_SYM"] = bool(ok)
    # G-REPRO (single block)
    for name, g, d in [("qrac21_d2", game_qrac(2), 2), ("qrac31_d2", game_qrac(3), 2), ("wr_K2_C6_d2", game_wr(2), 2),
                       ("wr_K3_C6_d4", game_wr(3), 4), ("wr_K4_C6_d4", game_wr(4), 4)]:
        r = solve(g, d, 1, ("none",), tag="REPRO " + name); r["ref_primal"] = REPRO[name]
        r["G_REPRO"] = bool(abs(r["primal"] - REPRO[name]) <= 1e-6); res["repro"][name] = r; certs.append(r)
    res["gates"]["G_REPRO"] = all(r["G_REPRO"] for r in res["repro"].values())
    # G-STATS
    z = np.load(STATS); dev = {}
    for K in (3, 4):
        g = game_wr(K); Pm = z["K%d_P" % K]; dev[K] = abs(sum(w * Pm[x, v, a] for x, v, a, w in g["pay"]) - P_Q[K])
    res["gates"]["G_STATS"] = bool(max(dev.values()) <= 1e-9); res["stats_dev"] = {str(k): v for k, v in dev.items()}
    # G-CAL (2->1 QRAC, probe x=(0,0), v=0)
    for kind in ("det", "opt"):
        g, rows = qrac21_stats(kind); r = solve(g, 2, 2, ("stats", rows), probe=(0, 0), tag="CAL " + kind)
        r["trivial_lb"] = max(p for xi, v, b, p in rows if xi == 0 and v == 0); res["cal"][kind] = r; certs.append(r)
    res["gates"]["G_DET"] = bool(res["cal"]["det"]["certified_ub"] >= 1 - 1e-6)
    res["gates"]["G_CAL_opt_not_below"] = bool(res["cal"]["opt"]["certified_ub"] >= 0.5 * (1 + 1 / math.sqrt(2)) - 1e-6)
    res["cal"]["opt"]["prediction_0p8536_within_1e-4"] = bool(abs(res["cal"]["opt"]["certified_ub"] - 0.5 * (1 + 1 / math.sqrt(2))) <= 1e-4)
    pre = ["G_GAMES_EQ_EXP127", "G_SYM", "G_REPRO", "G_STATS", "G_DET", "G_CAL_opt_not_below"]
    if not all(res["gates"][k] for k in pre) or not all(c["G_DUAL"] and c["G_LAGR"] for c in certs):
        stop("pre-gates failed: %s" % {k: res["gates"][k] for k in pre})
    print("PRE-GATES PASS", round(time.time() - t0), "s", flush=True)
    vstar = {K: json.load(open(os.path.join(ROOT, "results_exp133.json")))["cases"]["K%d_d4" % K]["v_star"] for K in (3, 4)}
    for K in (3, 4):
        Pm, Etr = z["K%d_P" % K], z["K%d_Etr" % K]; g = game_wr(K)
        pairs = sorted(((float(Pm[xi, v].max()), xi, v) for xi, x in enumerate(g["X"]) for v in x), key=lambda t: (t[0], t[1], t[2]))
        pick = {"most": pairs[-1], "median": pairs[len(pairs) // 2], "least": pairs[0]}
        for lab, (mx, xs, vs) in pick.items():
            for vis in (1.0, 0.99, 0.95):
                g, rows = wr_rows(K, Pm, Etr, vis)
                r = solve(g, 4, K, ("stats", rows), probe=(xs, vs), tag="K%d %s v=%.2f" % (K, lab, vis))
                r["pair"] = [list(g["X"][xs]), vs]; r["trivial_lb"] = max(p for xi, v, b, p in rows if xi == xs and v == vs)
                r["G_SOUND"] = bool(r["certified_ub"] >= r["trivial_lb"] - 1e-6); r["H_min"] = -math.log2(min(1.0, r["certified_ub"]))
                res["cases"]["K%d_%s_v%.2f" % (K, lab, vis)] = r; certs.append(r)
        # secondary: average-score constraint at W = P_Q, headline pair
        mx, xs, vs = pick["most"]
        r = solve(g, 4, K, ("avg", P_Q[K]), probe=(xs, vs), tag="K%d secondary avg" % K); res["secondary"]["K%d" % K] = r; certs.append(r)
        # randomness-critical visibility (headline pair)
        top = res["cases"]["K%d_most_v1.00" % K]["certified_ub"]; bis = {"v_star_exp133": vstar[K], "grid": []}
        if top >= 1 - 1e-6:
            bis["v_r"] = None; bis["note"] = "P_g = 1 at v = 1: no randomness certified by this relaxation"
        else:
            lo, hi = vstar[K], 1.0                                                # invariant: P_g(hi) < 1 - 1e-6
            for _ in range(12):
                mid = (lo + hi) / 2; g, rows = wr_rows(K, Pm, Etr, mid)
                rr = solve(g, 4, K, ("stats", rows), probe=(xs, vs), tag="K%d bisect v=%.6f" % (K, mid)); certs.append(rr)
                bis["grid"].append([mid, rr["certified_ub"], rr["G_DUAL"], rr["G_LAGR"]])
                if rr["certified_ub"] < 1 - 1e-6:
                    hi = mid
                else:
                    lo = mid
            bis["v_r"] = hi; bis["v_r_bracket"] = [lo, hi]
        res["bisect"]["K%d" % K] = bis
    res["gates"]["G_DUAL_all"] = all(c["G_DUAL"] for c in certs); res["gates"]["G_LAGR_all"] = all(c["G_LAGR"] for c in certs)
    res["gates"]["G_SOUND_all"] = all(c["G_SOUND"] for c in res["cases"].values())
    res["all_gates_pass"] = all(res["gates"].values()); res["wall_s"] = time.time() - t0; res["provenance"] = C94.provenance()
    C94.write_new_json(OUT, res)
    print("DONE", res["gates"], {k: round(c["certified_ub"], 6) for k, c in res["cases"].items()}, res["bisect"], flush=True)


SOLV = os.environ.get("EXP134_SOLVER", "CLARABEL"); SUF = {"CLARABEL": "", "CLARABEL_TIGHT": "_tight", "SCS": "_scs"}[SOLV]


def amend_a():                                                                    # doc 134 §10 (a) / §12 (a2): headline case
    out = os.path.join(ROOT, "results_exp134_amend_a%s.json" % SUF); assert not os.path.exists(out), "Rule 3"
    z = np.load(STATS); Pm, Etr = z["K3_P"], z["K3_Etr"]; g = game_wr(3)
    pairs = sorted(((float(Pm[xi, v].max()), xi, v) for xi, x in enumerate(g["X"]) for v in x), key=lambda t: (t[0], t[1], t[2]))
    mx, xs, vs = pairs[-1]; g, rows = wr_rows(3, Pm, Etr, 1.0)
    try:
        r = solve(g, 4, 3, ("stats", rows), probe=(xs, vs), tag="AMEND-A K3 most v1 " + SOLV, solver=SOLV)
    except Exception as ex:
        r = {"error": repr(ex)}
    r["pair"] = [list(g["X"][xs]), vs]; r["scs_smoke_ub"] = 1.000000003
    r["works"] = bool("error" not in r and r["G_DUAL"] and r["G_LAGR"] and abs(r["certified_ub"] - 1.000000003) <= 1e-6)
    C94.write_new_json(out, {"doc": "134 §10 (a) / §12", "solver": SOLV, "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "case": r})
    print("AMEND-A works", r["works"], flush=True)


def amend_b():                                                                    # doc 134 §10 (b): pre-gates + least predictable K=3 pair
    out = os.path.join(ROOT, "results_exp134_amend_b%s.json" % SUF); assert not os.path.exists(out), "Rule 3"
    import exp127_qubound as E127
    res = {"doc": "134 §10 (b)", "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "gates": {}, "repro": {}, "cal": {}}
    res["gates"]["G_GAMES_EQ_EXP127"] = bool(all(game_wr(K) == E127.game_wr(K) for K in (2, 3, 4)) and all(game_qrac(k) == E127.game_qrac(k) for k in (2, 3)))
    rng = np.random.default_rng(1340); ok = True
    for K in (3, 4):
        g = game_wr(K); pay = {(g["X"][xi], v, a): w for xi, v, a, w in g["pay"]}
        for _ in range(3):
            pc = rng.permutation(6); pp = rng.permutation(K)
            for (x, v, a), w in pay.items():
                y = [None] * K
                for i in range(K):
                    y[pp[i]] = int(pc[x[i]])
                ok &= abs(pay.get((tuple(y), int(pc[v]), int(pp[a])), -1) - w) == 0
    res["gates"]["G_SYM"] = bool(ok); certs = []
    for name, g, d in [("qrac21_d2", game_qrac(2), 2), ("qrac31_d2", game_qrac(3), 2), ("wr_K2_C6_d2", game_wr(2), 2),
                       ("wr_K3_C6_d4", game_wr(3), 4), ("wr_K4_C6_d4", game_wr(4), 4)]:
        r = solve(g, d, 1, ("none",), tag="REPRO " + name); r["G_REPRO"] = bool(abs(r["primal"] - REPRO[name]) <= 1e-6)
        res["repro"][name] = r; certs.append(r)
    res["gates"]["G_REPRO"] = all(r["G_REPRO"] for r in res["repro"].values())
    z = np.load(STATS)
    res["gates"]["G_STATS"] = bool(max(abs(sum(w * z["K%d_P" % K][x, v, a] for x, v, a, w in game_wr(K)["pay"]) - P_Q[K]) for K in (3, 4)) <= 1e-9)
    for kind in ("det", "opt"):
        g, rows = qrac21_stats(kind); r = solve(g, 2, 2, ("stats", rows), probe=(0, 0), tag="CAL " + kind); res["cal"][kind] = r; certs.append(r)
    res["gates"]["G_DET"] = bool(res["cal"]["det"]["certified_ub"] >= 1 - 1e-6)
    res["gates"]["G_CAL_opt_not_below"] = bool(res["cal"]["opt"]["certified_ub"] >= 0.5 * (1 + 1 / math.sqrt(2)) - 1e-6)
    res["gates"]["G_DUAL_LAGR_pre"] = all(c["G_DUAL"] and c["G_LAGR"] for c in certs)
    if all(res["gates"].values()):
        Pm, Etr = z["K3_P"], z["K3_Etr"]; g = game_wr(3)
        pairs = sorted(((float(Pm[xi, v].max()), xi, v) for xi, x in enumerate(g["X"]) for v in x), key=lambda t: (t[0], t[1], t[2]))
        mx, xs, vs = pairs[0]; g, rows = wr_rows(3, Pm, Etr, 1.0)
        r = solve(g, 4, 3, ("stats", rows), probe=(xs, vs), tag="AMEND-B K3 least v1 " + SOLV, solver=SOLV)
        r["pair"] = [list(g["X"][xs]), vs]; r["trivial_lb"] = max(p for xi, v, b, p in rows if xi == xs and v == vs)
        r["G_SOUND"] = bool(r["certified_ub"] >= r["trivial_lb"] - 1e-6); r["H_min"] = -math.log2(min(1.0, r["certified_ub"]))
        res["least_K3_v1"] = r
    else:
        res["stopped"] = "pre-gates failed"
    C94.write_new_json(out, res); print("AMEND-B", res["gates"], res.get("least_K3_v1", {}).get("certified_ub"), flush=True)


def seesaw_k2(restarts=200, seed=1342):                                          # doc 134 §16: best complex-qubit code for K = 2, C = 6
    g = game_wr(2); rng = np.random.default_rng(seed); best = None
    pos = {(xi, v): p for xi, v, p, w in g["pay"]}; w = g["pay"][0][3]
    def value(psi, E):
        return sum(w * float(np.real(psi[xi].conj() @ E[v][p] @ psi[xi])) for (xi, v), p in pos.items())
    for _ in range(restarts):
        E = []
        for v in range(6):
            u = rng.standard_normal(2) + 1j * rng.standard_normal(2); u /= np.linalg.norm(u); P = np.outer(u, u.conj())
            E.append([P, np.eye(2) - P])
        last = -1.0
        for _ in range(10000):
            psi = []
            for xi, x in enumerate(g["X"]):
                O = sum(w * E[v][p] for p, v in enumerate(x)); ev, vec = np.linalg.eigh(O); psi.append(vec[:, -1])
            for v in range(6):
                D = np.zeros((2, 2), complex)
                for xi, x in enumerate(g["X"]):
                    if v in x:
                        rho = np.outer(psi[xi], psi[xi].conj()); D += w * rho if x.index(v) == 0 else -w * rho
                ev, vec = np.linalg.eigh(D); P = sum(np.outer(vec[:, i], vec[:, i].conj()) for i in range(2) if ev[i] > 0) if (ev > 0).any() else np.zeros((2, 2), complex)
                E[v] = [P, np.eye(2) - P]
            val = value(psi, E)
            if abs(val - last) < 1e-12:
                break
            last = val
        if best is None or val > best[0]:
            best = (val, [p.copy() for p in psi], [[e.copy() for e in pr] for pr in E])
    return g, best


def amend_c():                                                                    # doc 134 §16: K = 2 calibration
    out = os.path.join(ROOT, "results_exp134_amend_c.json"); assert not os.path.exists(out), "Rule 3"
    t0 = time.time(); certs = []; QOPT = 0.8828427124746194; REL = REPRO["wr_K2_C6_d2"]
    res = {"doc": "134 §16", "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "gates": {}, "cases": {}}
    def finish(why=None):
        if why:
            res["stopped"] = why
        res["wall_s"] = time.time() - t0; C94.write_new_json(out, res); print("AMEND-C", why or "done", res["gates"], flush=True)
    r = solve(game_wr(2), 2, 1, ("none",), tag="REPRO wr_K2_C6_d2"); certs.append(r)
    res["gates"]["G_REPRO"] = bool(abs(r["primal"] - REL) <= 1e-6); res["repro"] = r
    for kind in ("det", "opt"):
        gq, rows = qrac21_stats(kind); r = solve(gq, 2, 2, ("stats", rows), probe=(0, 0), tag="CAL " + kind); res[kind] = r; certs.append(r)
    res["gates"]["G_DET"] = bool(res["det"]["certified_ub"] >= 1 - 1e-6)
    res["gates"]["G_CAL_opt_not_below"] = bool(res["opt"]["certified_ub"] >= 0.5 * (1 + 1 / math.sqrt(2)) - 1e-6)
    g, (val, psi, E) = seesaw_k2(); res["seesaw_value"] = val; res["gates"]["G_SEESAW"] = bool(abs(val - QOPT) <= 1e-6)
    Pm = np.zeros((30, 6, 2)); Etr = np.zeros((6, 2))
    for xi, x in enumerate(g["X"]):
        for v in x:
            for b in range(2):
                Pm[xi, v, b] = float(np.real(psi[xi].conj() @ E[v][b] @ psi[xi]))
    for v in range(6):
        for b in range(2):
            Etr[v, b] = float(np.real(np.trace(E[v][b])))
    sc = sum(w * Pm[xi, v, p] for xi, v, p, w in g["pay"]); noise = sum(w * Etr[v, p] / 2 for xi, v, p, w in g["pay"])
    res["gates"]["G_STATS2"] = bool(abs(sc - val) <= 1e-9); res["noise_score"] = noise
    res["gates"]["G_DUAL_LAGR_pre"] = all(c["G_DUAL"] and c["G_LAGR"] for c in certs)
    np.savez(os.path.join(LOGS, "amend_c_code.npz"), psi=np.array(psi), E=np.array(E), Pm=Pm, Etr=Etr)
    table = sorted(((float(Pm[xi, v].max()), xi, v) for xi, x in enumerate(g["X"]) for v in x), key=lambda t: (t[0], t[1], t[2]))
    res["trivial_lb_table"] = [[m, list(g["X"][xi]), v] for m, xi, v in table]
    print("PRE", res["gates"], "seesaw", val, "noise_score", noise, "lb range", table[0][0], table[-1][0], round(time.time() - t0), "s", flush=True)
    if not all(res["gates"].values()):
        return finish("pre-gates failed")
    vis = {}
    for gap in (0.0045, 0.042, 0.02):
        vis[gap] = 1.0 if gap == 0.0045 else (REL - gap - noise) / (val - noise)
    pairs = [("least", table[0])] + ([("most", table[-1])] if table[-1][0] - table[0][0] > 1e-6 else [])
    for lab, (mx, xs, vs) in pairs:
        for gap in (0.0045, 0.042, 0.02):
            v = vis[gap]; rows = [(xi, cv, b, v * Pm[xi, cv, b] + (1 - v) * Etr[cv, b] / 2) for xi, x in enumerate(g["X"]) for cv in x for b in range(2)]
            key = "%s_gap%s" % (lab, gap)
            try:
                r = solve(g, 2, 2, ("stats", rows), probe=(xs, vs), tag="AMEND-C " + key, save=os.path.join(LOGS, "amend_c_%s.npz" % key), extra={"time_limit_secs": 1800})
            except Exception as ex:
                res["cases"][key] = {"error": repr(ex)}; return finish("solve error (%s)" % key)
            r["pair"] = [list(g["X"][xs]), vs]; r["visibility"] = v; r["score"] = v * val + (1 - v) * noise
            r["trivial_lb"] = max(p for xi, cv, b, p in rows if xi == xs and cv == vs)
            r["G_SOUND"] = bool(r["certified_ub"] >= r["trivial_lb"] - 1e-6); r["H_min_certified"] = -math.log2(min(1.0, r["certified_ub"])) if r["G_DUAL"] else None
            res["cases"][key] = r
            if r["wall_s"] > 1800:
                return finish("solve passed 30 minutes (%s)" % key)
    finish()


def amend_d():                                                                    # doc 134 §18: 2->1 QRAC under white noise
    out = os.path.join(ROOT, "results_exp134_amend_d.json"); assert not os.path.exists(out), "Rule 3"
    t0 = time.time(); QM = 0.5 * (1 + 1 / math.sqrt(2))
    g, rows = qrac21_stats("opt")
    res = {"doc": "134 §18", "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "gates": {}, "grid": {}, "bisect": []}
    def case(v, tag):
        rv = [(xi, q, b, v * p + (1 - v) * 0.5) for xi, q, b, p in rows]
        r = solve(g, 2, 2, ("stats", rv), probe=(0, 0), tag="AMEND-D %s v=%.6f" % (tag, v), save=os.path.join(LOGS, "amend_d_%s_%.6f.npz" % (tag, v)), extra={"time_limit_secs": 1800})
        r["visibility"] = v; r["success"] = v * QM + (1 - v) * 0.5; r["trivial_lb"] = max(p for xi, q, b, p in rv if xi == 0 and q == 0)
        r["G_SOUND"] = bool(r["certified_ub"] >= r["trivial_lb"] - 1e-6)
        r["H_min_certified"] = -math.log2(min(1.0, r["certified_ub"])) if r["G_DUAL"] else 0.0
        r["randomness"] = bool(r["G_DUAL"] and r["certified_ub"] < 1 - 1e-6)
        return r
    def finish(why=None):
        if why:
            res["stopped"] = why
        res["wall_s"] = time.time() - t0; C94.write_new_json(out, res); print("AMEND-D", why or "done", res["gates"], flush=True)
    try:
        r = case(1.0, "cal1"); res["cal1"] = r; res["gates"]["G_CAL1"] = bool(abs(r["certified_ub"] - 0.853555) <= 1e-5 and r["G_DUAL"] and r["G_LAGR"])
        if not res["gates"]["G_CAL1"]:
            return finish("G-CAL1 failed")
        for v in (0.984, 0.974, 0.99, 0.962, 0.933, 0.90):
            r = case(v, "grid"); res["grid"]["%.3f" % v] = r
            if r["wall_s"] > 1800:
                return finish("solve passed 30 minutes")
        lo, hi = (0.90, 1.0) if not res["grid"]["0.900"]["randomness"] else (1 / math.sqrt(2), 0.90)
        for _ in range(12):
            mid = (lo + hi) / 2; r = case(mid, "bisect"); res["bisect"].append(r)
            if r["wall_s"] > 1800:
                return finish("solve passed 30 minutes")
            if r["randomness"]:
                hi = mid
            else:
                lo = mid
        res["v_r"] = hi; res["v_r_bracket"] = [lo, hi]; res["success_at_v_r"] = hi * QM + (1 - hi) * 0.5
    except Exception as ex:
        res["error"] = repr(ex); return finish("error")
    res["gates"]["G_DUAL_LAGR_all"] = all(c["G_DUAL"] and c["G_LAGR"] for c in list(res["grid"].values()) + res["bisect"])
    res["gates"]["G_SOUND_all"] = all(c["G_SOUND"] for c in list(res["grid"].values()) + res["bisect"])
    finish()


if __name__ == "__main__":
    {"stats": stats, "run": run, "amend_a": amend_a, "amend_b": amend_b, "amend_c": amend_c, "amend_d": amend_d}[sys.argv[1]]()
