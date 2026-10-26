# -*- coding: utf-8 -*-
"""exp114_run.py -- doc 114 (REGISTERED 5722343): does STRICT P1 composition collapse to the classical bit at C >= 4?

Reduced objective (doc 114 s1): r_ab = R_b R_a z, g_v = sum_{b != v} (r_vb - r_bv), success = 1/2 + sum_v |g_v| / (4N),
N = C(C-1); box R(theta, phi) = rotation by theta about (cos phi, sin phi, 0); phi_0 = 0 fixed -> 2C-1 parameters.
Phase A: (i) grid (theta in [0, pi], phi in [0, 2pi), common spacing pi/(n-1)); calibration = top-200 grid points at
C = 3 refined by L-BFGS must reach >= 0.848; the same n at C = 4, top-200 refined.  (ii) 4096 random multistarts +
L-BFGS at C = 4, 5, 6.  Conjecture FALSE iff any C >= 4 point > classical + 1e-9.
Phase B (run regardless, as diagnostics; only reported as B if A survives): g_v . z check (random + algebraic identity),
state geometry at the best points.  Writes results_exp114.json (Rule 3).
"""
import heapq, itertools, math, time
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
import exp94_common as C94

CLASSICAL = {3: 5 / 6, 4: 5 / 6, 5: 0.8, 6: 0.8}
Z = np.array([0.0, 0.0, 1.0])
TOP = 200
NSTART = 4096


def rmat(th, ph):
    """Batched rotation by th about (cos ph, sin ph, 0); returns (..., 3, 3)."""
    th, ph = np.broadcast_arrays(np.asarray(th, float), np.asarray(ph, float))
    a = np.stack([np.cos(ph), np.sin(ph), np.zeros_like(ph)], -1)
    K = np.zeros(th.shape + (3, 3))
    K[..., 0, 2], K[..., 1, 2] = a[..., 1], -a[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -a[..., 1], a[..., 0]
    s, c = np.sin(th)[..., None, None], np.cos(th)[..., None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def unpack(x, C):
    th = np.concatenate([[x[0]], x[1::2][:C - 1]]); ph = np.concatenate([[0.0], x[2::2][:C - 1]])
    return th, ph


def gvecs(x, C):
    th, ph = unpack(x, C)
    R = rmat(th, ph); u = R @ Z                                   # u_a = R_a z
    r = np.einsum("bij,aj->abi", R, u)                            # r[a, b] = R_b R_a z
    g = r.sum(1) - r.sum(0)                                       # diagonal r_vv cancels
    return g, r


def success(x, C):
    g, _ = gvecs(x, C)
    return 0.5 + np.linalg.norm(g, axis=1).sum() / (4 * C * (C - 1))


def refine(x0, C):
    res = minimize(lambda x: -success(x, C), x0, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-12})
    return float(-res.fun), res.x


def grid(C, n):
    """Exhaustive grid; returns top-TOP (value, x) list."""
    d = math.pi / (n - 1)
    ths = np.arange(n) * d
    phs = np.arange(int(round(2 * math.pi / d))) * d
    T, P = np.meshgrid(ths, phs, indexing="ij"); T, P = T.ravel(), P.ravel()
    R = rmat(T, P); u = R @ Z                                     # combos for classes >= 1
    R0 = rmat(ths, 0 * ths); u0 = R0 @ Z                          # class 0: phi = 0
    Kc = len(T); heap = []
    outer = [range(n)] + [range(Kc)] * (C - 3)
    for idx in itertools.product(*outer):
        Rs = [R0[idx[0]]] + [R[i] for i in idx[1:]]; us = [u0[idx[0]]] + [u[i] for i in idx[1:]]
        m = len(Rs)                                               # fixed classes 0..m-1; last two vectorised (Kc, Kc)
        def rr(a, b):                                             # r_ab with a, b in {0..m-1, 'p', 'q'}
            Ra = {"p": R[:, None], "q": R[None, :]}; ua = {"p": u[:, None], "q": u[None, :]}
            Rb = Ra[b] if b in Ra else Rs[b]; uv = ua[a] if a in ua else us[a]
            return np.einsum("...ij,...j->...i", Rb, uv)
        labels = list(range(m)) + ["p", "q"]
        tot = 0.0
        for v in labels:
            gv = 0.0
            for b in labels:
                if b != v:
                    gv = gv + rr(v, b) - rr(b, v)
            gv = np.broadcast_to(gv, (Kc, Kc, 3))
            tot = tot + np.linalg.norm(gv, axis=-1)
        val = 0.5 + tot / (4 * C * (C - 1))
        flat = val.ravel(); k = min(TOP, flat.size)
        top = np.argpartition(-flat, k - 1)[:k]
        for t in top:
            p, q = divmod(int(t), Kc)
            item = (float(flat[t]), idx, p, q)
            if len(heap) < TOP:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
    out = []
    for v, idx, p, q in sorted(heap, reverse=True):
        x = [ths[idx[0]]]
        for i in list(idx[1:]) + [p, q]:
            x += [T[i], P[i]]
        out.append((v, np.array(x)))
    return out, len(ths) * Kc ** (C - 1)


def _refine_job(a):
    x0, C = a
    return refine(np.asarray(x0), C)


def _ms_job(a):
    C, seed = a
    rng = np.random.default_rng(114000 + 1000 * C + seed)
    x0 = rng.uniform(0, 2 * math.pi, 2 * C - 1)
    v, x = refine(x0, C)
    return v, x.tolist()


def geometry(x, C):
    g, r = gvecs(x, C)
    gn = np.linalg.norm(g, axis=1)
    gu = g[gn > 1e-9] / gn[gn > 1e-9, None]
    rs = np.array([r[a, b] for a in range(C) for b in range(C) if a != b])
    sv_g = np.linalg.svd(gu, compute_uv=False).tolist() if len(gu) else []
    sv_r = np.linalg.svd(rs - rs.mean(0), compute_uv=False).tolist()
    th, ph = unpack(x, C)
    return {"g_norms": gn.tolist(), "g_z_max": float(np.abs(g[:, 2]).max()), "g_unit_singular_values": sv_g,
            "r_centered_singular_values": sv_r, "r_z_values": sorted(set(np.round(rs[:, 2], 9).tolist())),
            "theta": (np.mod(th, 2 * math.pi)).tolist(), "phi": (np.mod(ph, 2 * math.pi)).tolist()}


def gz_check(trials=100000):
    """(R_b R_a - R_a R_b) z . z == 0 for P1 boxes. Identity: Rz(pi) R(th,ph) Rz(pi)^T = R(th,ph)^{-1}, so
    z.R_a R_b z = z.Rz(pi) R_b^{-1} R_a^{-1} Rz(pi)^T z ... = z.R_b R_a z (transpose + Rz(pi) z = z)."""
    rng = np.random.default_rng(114)
    th = rng.uniform(0, 2 * math.pi, (trials, 2)); ph = rng.uniform(0, 2 * math.pi, (trials, 2))
    Ra, Rb = rmat(th[:, 0], ph[:, 0]), rmat(th[:, 1], ph[:, 1])
    d = (Rb @ Ra @ Z - Ra @ Rb @ Z)[:, 2]
    Rz = np.diag([-1.0, -1.0, 1.0])
    conj = np.abs(Rz @ Ra @ Rz.T - np.transpose(Ra, (0, 2, 1))).max()
    return {"trials": trials, "max_abs_z_component": float(np.abs(d).max()), "max_conjugation_identity_error": float(conj)}


def main():
    t0 = time.time()
    out = {"doc": "114 (5722343)", "classical": CLASSICAL}
    # sanity: objective reproduces exp108 STRICT_P1 C = 3 (0.84912) and the free-axis form equals the exp108 form
    # calibration
    cal = {}
    with Pool(2) as pool:
        for n in (4, 6, 8, 10, 12, 16, 20):
            top, npts = grid(3, n)
            ref = pool.map(_refine_job, [(x.tolist(), 3) for _, x in top])
            best = max(v for v, _ in ref)
            cal[n] = {"grid_points": npts, "grid_best": top[0][0], "refined_best": best, "pass": best >= 0.848}
            print("cal n", n, cal[n], int(time.time() - t0), flush=True)
    passing = [n for n in cal if cal[n]["pass"]]
    n_use = max(passing) if passing else None                     # finest affordable passing grid (>= registered)
    out["calibration"] = {"per_n": cal, "smallest_passing_n": min(passing) if passing else None, "n_used": n_use}
    if n_use is None:
        out["phase_A"] = {"status": "CALIBRATION FAILED"}
        C94.write_new_json("results_exp114.json", out); return
    top4, npts4 = grid(4, n_use)
    print("grid C4 done", npts4, top4[0][0], int(time.time() - t0), flush=True)
    with Pool(2) as pool:                                 # exp113 holds 8 of 10 cores
        ref4 = pool.map(_refine_job, [(x.tolist(), 4) for _, x in top4])
        best_grid4 = max(ref4, key=lambda t: t[0])
        out["grid_C4"] = {"n": n_use, "points": npts4, "grid_best": top4[0][0], "refined_best": best_grid4[0],
                          "refined_x": best_grid4[1].tolist(),
                          "refined_values_top10": sorted([v for v, _ in ref4], reverse=True)[:10]}
        print("grid C4 refined", best_grid4[0], int(time.time() - t0), flush=True)
        ms = {}
        for C in (3, 4, 5, 6):
            res = pool.map(_ms_job, [(C, s) for s in range(NSTART)], chunksize=32)
            vals = np.array([v for v, _ in res]); b = int(vals.argmax())
            ms[C] = {"best": float(vals[b]), "best_x": res[b][1], "excess_over_classical": float(vals[b] - CLASSICAL[C]),
                     "n_within_1e-6_of_best": int((vals > vals[b] - 1e-6).sum()),
                     "quantiles": np.quantile(vals, [0.5, 0.9, 0.99]).tolist()}
            print("ms C", C, {k: v for k, v in ms[C].items() if k != "best_x"}, int(time.time() - t0), flush=True)
    out["multistart"] = ms
    viol = []
    if best_grid4[0] > CLASSICAL[4] + 1e-9:
        viol.append({"C": 4, "source": "grid", "value": best_grid4[0]})
    for C in (4, 5, 6):
        if ms[C]["best"] > CLASSICAL[C] + 1e-9:
            viol.append({"C": C, "source": "multistart", "value": ms[C]["best"]})
    out["phase_A"] = {"status": "FALSE" if viol else "SURVIVES", "violations": viol}
    out["phase_B_diagnostics"] = {"gz_check": gz_check(),
                                  "geometry": {C: geometry(np.array(ms[C]["best_x"]), C) for C in (3, 4, 5, 6)}}
    out["wall_s"] = time.time() - t0
    out["provenance"] = C94.provenance()
    C94.write_new_json("results_exp114.json", out)
    print("PHASE A", out["phase_A"], flush=True)


if __name__ == "__main__":
    main()
