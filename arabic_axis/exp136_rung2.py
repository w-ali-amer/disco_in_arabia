# -*- coding: utf-8 -*-
"""exp136_rung2.py -- doc 136 stage C, amendment C-b: Navascues-Vertesi (1412.0924) dimension-respecting SDP pilot on
the K = 2, C = 6 which-referent game, d = 2, uniform prior.
Words (C-b): 1, rho_x (30), M_v (6, outcome "first"), rho_x M_v for v in x (60) -> 97.
Reductions (valid for the uniform game value): real part (complex conjugation) and S_6 class relabelling (Reynolds
average over orbits of word pairs).  Gamma ranges over the affine hull of orbit-averaged real moment matrices of
random qubit realisations (Haar pure states, general two-outcome POVMs), plus Gamma >= 0 and the exp127 level-1
inequalities (pairwise traces of {rho, M, 1-M} >= 0, tr M^2 <= tr M, 0 <= tr M <= 2) and tr(M rho M) <= tr(rho M).
Certificate: R = C + Z+ - sum z_i G_i (Z+ = PSD part of the PSD dual, z clipped >= 0);
  bound = <R, Gamma0> + z.h + ||P_L R||_F (T + ||Gamma0||_F),  T = 2 + 30 + 12 + 60 = 104 >= tr Gamma.
modes: check (G-SEE: span at d = 2 vs 8, full and reduced), run (L1 repro + C-b pilot).  Writes results_exp136_C.json.
"""
import hashlib, itertools, json, os, sys, time
import numpy as np
import exp94_common as C94

ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "results_exp136_C.json")
TOL = 1e-9; PATIENCE = 50; SEED = 1363
QOPT, L1REL = 0.8828427124746194, 0.8872983346207417
X = list(itertools.permutations(range(6), 2))
MEAS = {"kind": "rank1"}


def words(level):
    W = [("I",)] + [("r", x) for x in X] + [("m", v) for v in range(6)]
    if level == "Cb":
        W += [("rm", x, v) for x in X for v in x]
    return W


def act(p, w):
    if w[0] == "I":
        return w
    if w[0] == "r":
        return ("r", (p[w[1][0]], p[w[1][1]]))
    if w[0] == "m":
        return ("m", p[w[1]])
    return ("rm", (p[w[1][0]], p[w[1][1]]), p[w[2]])


def orbits(W):
    """orbit id for each unordered pair (i <= j) under S_6."""
    idx = {w: i for i, w in enumerate(W)}; n = len(W); lab = -np.ones((n, n), int); k = 0
    perms = list(itertools.permutations(range(6)))
    for i in range(n):
        for j in range(i, n):
            if lab[i, j] >= 0:
                continue
            for p in perms:
                a, b = idx[act(p, W[i])], idx[act(p, W[j])]; a, b = min(a, b), max(a, b); lab[a, b] = k; lab[b, a] = k
            k += 1
    return lab, k


def realisation(rng, d):
    rho = {}
    for x in X:
        v = rng.standard_normal(d) + 1j * rng.standard_normal(d); v /= np.linalg.norm(v); rho[x] = np.outer(v, v.conj())
    M = []                                                                       # C-c: rank-1 projectors (extremal qubit effects, all-rank-1 case)
    if MEAS["kind"] == "povm":                                                   # general two-outcome POVMs (L1 reproduction of exp127)
        for _ in range(6):
            Z = rng.standard_normal((2 * d, d)) + 1j * rng.standard_normal((2 * d, d)); V, _ = np.linalg.qr(Z); M.append(V[:d].conj().T @ V[:d])
        return rho, M
    for _ in range(6):
        u = rng.standard_normal(d) + 1j * rng.standard_normal(d); u /= np.linalg.norm(u); M.append(np.outer(u, u.conj()))
    return rho, M


def moment(W, rho, M, d):
    ops = []
    for w in W:
        ops.append(np.eye(d) if w[0] == "I" else rho[w[1]] if w[0] == "r" else M[w[1]] if w[0] == "m" else rho[w[1]] @ M[w[2]])
    A = np.array([o.reshape(-1) for o in ops]); return A.conj() @ A.T


def grow(sample_vec, dim, maxk=None):
    """Gram-Schmidt span of sample vectors in batches of PATIENCE (rel tol 1e-9); stop when a whole batch adds nothing."""
    maxk = maxk or dim; Q = np.zeros((dim, maxk)); k = 0; ns = 0
    while k < maxk:
        B = np.array([sample_vec() for _ in range(PATIENCE)]).T; ns += PATIENCE; B /= np.linalg.norm(B, axis=0); added = 0
        for _ in range(2):
            B = B - Q[:, :k] @ (Q[:, :k].T @ B)
        for j in range(B.shape[1]):
            r = B[:, j].copy()
            for _ in range(2):
                r = r - Q[:, k - added:k] @ (Q[:, k - added:k].T @ r)
            nr = np.linalg.norm(r)
            if nr > TOL and k < maxk:
                Q[:, k] = r / nr; k += 1; added += 1
        if not added:
            break
    return Q[:, :k], ns, None


def reduced_vec(W, lab, norb, rng, d):
    """orbit-averaged real moment matrix as a vector of orbit values."""
    G = moment(W, *realisation(rng, d), d).real; s = np.zeros(norb); c = np.zeros(norb)
    np.add.at(s, lab.ravel(), G.ravel()); np.add.at(c, lab.ravel(), 1.0); return s / c


def full_vec(W, rng, d):                                                         # C-c: real parts only (score is real; Re Gamma feasible)
    G = moment(W, *realisation(rng, d), d); iu = np.triu_indices(len(W)); return G[iu].real


def check():
    rng = np.random.default_rng(SEED); out = {}
    for level in ("L1", "Cb"):
        W = words(level); lab, norb = orbits(W); n = len(W)
        for d in (2, 8):
            Qf, nsf, _ = grow(lambda: full_vec(W, rng, d), n * (n + 1) // 2)
            Qr, nsr, _ = grow(lambda: reduced_vec(W, lab, norb, rng, d), norb)
            out["%s|d%d" % (level, d)] = {"n_words": n, "orbits": norb, "span_full": Qf.shape[1], "span_reduced": Qr.shape[1]}
            print(level, d, out["%s|d%d" % (level, d)], flush=True)
    out["G_SEE_full"] = out["Cb|d2"]["span_full"] < out["Cb|d8"]["span_full"]
    out["G_SEE_reduced"] = out["Cb|d2"]["span_reduced"] < out["Cb|d8"]["span_reduced"]
    return out


def ineqs(W):
    """rows (dict {(i,j): coef}, h) meaning sum coef * Gamma_ij <= h, valid for every qubit strategy."""
    idx = {w: i for i, w in enumerate(W)}; I = 0; R = [idx[("r", x)] for x in X]; Ms = [idx[("m", v)] for v in range(6)]; rows = []
    # pairwise traces of PSD operators >= 0 over {rho_x, M_v, 1 - M_v}; tr 1 = 2, tr rho = 1
    def pair(a, b):                                                       # tr(a b) as (coef dict, const) for a, b in ("r",i) / ("m",i) / ("c",i)
        if a[0] == "c" and b[0] == "c":
            cf = {(I, a[1]): -1.0}; cf[(I, b[1])] = cf.get((I, b[1]), 0.0) - 1.0; cf[(min(a[1], b[1]), max(a[1], b[1]))] = 1.0
            return cf, 2.0
        if a[0] == "c":
            a, b = b, a
        if b[0] == "c":
            return ({(I, a[1]): 1.0, (min(a[1], b[1]), max(a[1], b[1])): -1.0}, 0.0) if a[0] == "m" else ({(min(a[1], b[1]), max(a[1], b[1])): -1.0}, 1.0)
        return {(min(a[1], b[1]), max(a[1], b[1])): 1.0}, 0.0
    ops = [("r", r) for r in R] + [("m", m) for m in Ms] + [("c", m) for m in Ms]
    for i in range(len(ops)):
        for j in range(i, len(ops)):
            if ops[i] == ops[j] and ops[i][0] == "r":
                continue                                                  # tr rho^2 = 1 is in the span
            cf, c0 = pair(ops[i], ops[j]); rows.append(({k: -v for k, v in cf.items()}, c0))   # -(cf.G) <= c0
    for m in Ms:
        rows.append(({(I, m): -1.0}, 0.0)); rows.append(({(I, m): 1.0}, 2.0))
    if ("rm", X[0], X[0][0]) in idx:
        for x in X:
            for v in x:
                k = idx[("rm", x, v)]; rows.append(({(k, k): 1.0, (min(idx[("r", x)], idx[("m", v)]), max(idx[("r", x)], idx[("m", v)])): -1.0}, 0.0))
    return rows


def objective(W):
    idx = {w: i for i, w in enumerate(W)}; C = np.zeros((len(W), len(W))); const = 0.0; w = 1.0 / 60
    for x in X:
        for p, v in enumerate(x):
            a, b = idx[("r", x)], idx[("m", v)]; s = 1.0 if p == 0 else -1.0; const += 0.0 if p == 0 else w
            C[a, b] += s * w / 2; C[b, a] += s * w / 2
    return C, const


def sym(i, j, n):
    E = np.zeros((n, n)); E[i, j] += 0.5; E[j, i] += 0.5; return E


def pilot(level, rng, meas="rank1"):
    import cvxpy as cp
    MEAS["kind"] = meas
    W = words(level); n = len(W); lab, norb = orbits(W); t0 = time.time()
    O = [(lab == k).astype(float) for k in range(norb)]
    def mat(u):
        return sum(u[k] * O[k] for k in range(norb))
    Gs = [mat(reduced_vec(W, lab, norb, rng, 2))]
    G0 = Gs[0]; assert abs(G0[0, 0] - 2) < 1e-12
    def diff():
        Gm = mat(reduced_vec(W, lab, norb, rng, 2)); return (Gm - G0).ravel()
    Q, ns, _ = grow(diff, n * n, norb); Lb = [Q[:, k].reshape(n, n) for k in range(Q.shape[1])]   # Frobenius-orthonormal basis of L
    C, c0 = objective(W); rows = ineqs(W)
    Gm = np.zeros((len(rows), n, n)); h = np.zeros(len(rows))
    for r, (cf, hh) in enumerate(rows):
        for (i, j), v in cf.items():
            Gm[r] += v * sym(i, j, n)
        h[r] = hh
    t = cp.Variable(len(Lb)); Gam = G0 + sum(t[k] * Lb[k] for k in range(len(Lb))) if Lb else cp.Constant(G0)
    Gf = Gm.reshape(len(rows), -1); psd = (Gam + Gam.T) / 2 >> 0; lin = Gf @ cp.vec(Gam, order="C") <= h
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(C, Gam)) + c0), [psd, lin])
    prob.solve(solver="SCS", eps_abs=1e-9, eps_rel=1e-9, max_iters=2000000, time_limit_secs=7200)
    Z = np.asarray(psd.dual_value, float); Z = (Z + Z.T) / 2; ev, U = np.linalg.eigh(Z); Zp = (U * np.clip(ev, 0, None)) @ U.T
    z = np.clip(np.asarray(lin.dual_value, float).ravel(), 0, None)
    R = C + Zp - np.tensordot(z, Gm, 1)
    pl = np.array([float((R * L).sum()) for L in Lb]); corr = float(np.linalg.norm(pl)) * (104.0 + float(np.linalg.norm(G0)))
    ub = float((R * G0).sum()) + float(z @ h) + c0 + corr + 1e-9
    Gv = G0 + sum(float(t.value[k]) * Lb[k] for k in range(len(Lb))) if Lb else G0
    r = {"level": level, "n_words": n, "orbits": norb, "L_dim": len(Lb), "span_samples": ns, "primal": float(prob.value), "status": prob.status,
         "certified_ub": ub, "correction": corr, "lam_min_primal": float(np.linalg.eigvalsh((Gv + Gv.T) / 2).min()),
         "max_ineq_viol": float((Gf @ Gv.ravel() - h).max()), "trace_primal": float(np.trace(Gv)), "wall_s": time.time() - t0}
    r["G_DUAL"] = bool(ub >= r["primal"] - 1e-7 and corr < 1e-4)
    r["meas"] = meas; MEAS["kind"] = "rank1"; print(level, json.dumps(r), flush=True); return r


def trivial_bounds():
    """configurations with t >= 1 trivial measurements (M_v = 0 or 1): each trivial class scores exactly 5/60;
    the rest is bounded by the exp134 level-1 relaxation (general POVMs) of the game without those classes' terms."""
    import exp134_run as E
    g = E.game_wr(2); out = {}
    for t in range(1, 6):
        gt = {"X": g["X"], "dec": g["dec"], "pay": [p for p in g["pay"] if p[1] >= t]}
        r = E.solve(gt, 2, 1, ("none",), tag="trivial t=%d" % t)
        out[t] = {"l1_rest": r["certified_ub"], "G_DUAL": r["G_DUAL"], "bound": r["certified_ub"] + 5.0 * t / 60}
    out[6] = {"bound": 0.5, "G_DUAL": True}
    return out


def main(mode):
    if mode == "check":
        print(json.dumps(check(), indent=1)); return
    assert not os.path.exists(OUT), "Rule 3"
    rng = np.random.default_rng(SEED); t0 = time.time()
    res = {"doc": "136 stage C (amendment C-c)", "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "gates": {}}
    res["see"] = check(); res["gates"]["G_SEE"] = bool(res["see"]["G_SEE_full"] and res["see"]["G_SEE_reduced"])
    if not res["gates"]["G_SEE"]:
        res["stopped"] = "C-c word set / reduction does not see d = 2 in real parts"
    else:
        r1 = pilot("L1", rng, "povm"); res["L1"] = r1; res["gates"]["G_REPRO_L1"] = bool(abs(r1["certified_ub"] - L1REL) <= 1e-6)
        if not res["gates"]["G_REPRO_L1"]:
            res["stopped"] = "L1 mode does not reproduce 0.8872983"
        else:
            rc = pilot("Cb", rng); res["Cb"] = rc
            tb = trivial_bounds(); res["trivial"] = tb; ub = max([rc["certified_ub"]] + [v["bound"] for v in tb.values()]); res["certified_ub"] = ub
            res["gates"]["G_DUAL"] = bool(rc["G_DUAL"] and all(v["G_DUAL"] for v in tb.values()))
            res["gates"]["G_SANDWICH"] = bool(ub >= QOPT - 1e-7)
            res["improvement_over_L1"] = L1REL - ub; res["gap_to_qubit"] = ub - QOPT
            res["kill"] = bool(res["improvement_over_L1"] <= 1e-4 or rc["wall_s"] > 7200)
    res["wall_s"] = time.time() - t0; res["provenance"] = C94.provenance(); C94.write_new_json(OUT, res)
    print("DONE", json.dumps({k: res.get(k) for k in ("gates", "improvement_over_L1", "gap_to_qubit", "kill", "stopped")}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run")
