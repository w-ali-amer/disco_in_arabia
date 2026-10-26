# -*- coding: utf-8 -*-
"""exp128_spike.py -- doc 127 SPIKE (registered 205a8d0; THROWAWAY): the K=3, C=6 which-referent game on a two-atom pair
under a GLOBAL drive = one qutrit (symmetric subspace), readout = number of excitations (3 outcomes).
S1: qutrit LB with orthonormal-basis decoders (global unitary + count) and with general POVMs; classical with n = 3 messages
(exp100 MILP generalised from 2^m to n).  S2: Lie-algebra dimension of the symmetric-subspace controls.
Run with ~/qnlp_mac_env.  usage: python exp128_spike.py [s1|s2|cl] ...  Writes results_exp128_spike[_part].json (Rule 3)."""
import itertools, json, math, os, sys, time
import numpy as np, torch
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
import exp100_accounting as A100, exp110_run as R110

ROOT = os.path.dirname(os.path.abspath(__file__))
RESTARTS, STEPS = 256, 1500


def classical_milp_n(pb, M, time_limit, rep0=None):
    """exp100.classical_milp with M messages (not 2^m).  rep0: optional non-decreasing decoder for class R[0] (symmetry rep)."""
    X, R, A, P = pb["X"], pb["R"], pb["A"], pb["P"]; idx = {}
    def var(k):
        if k not in idx:
            idx[k] = len(idx)
        return idx[k]
    for x in X:
        for s in range(M):
            var(("e", x, s))
    for r in R:
        for s in range(M):
            for a in A[r]:
                var(("d", r, s, a))
    for (x, r, a) in P:
        for s in range(M):
            var(("z", x, r, a, s))
    n = len(idx); c = np.zeros(n); integ = np.zeros(n)
    for k, i in idx.items():
        if k[0] in ("e", "d"):
            integ[i] = 1
        else:
            c[i] = -P[(k[1], k[2], k[3])]
    rows, lo, hi = [], [], []
    for x in X:
        rows.append({idx[("e", x, s)]: 1 for s in range(M)}); lo.append(1); hi.append(1)
    for r in R:
        for s in range(M):
            rows.append({idx[("d", r, s, a)]: 1 for a in A[r]}); lo.append(1); hi.append(1)
    if rep0 is not None:
        for s, a in enumerate(rep0):
            rows.append({idx[("d", R[0], s, a)]: 1}); lo.append(1); hi.append(1)
    for (x, r, a) in P:
        for s in range(M):
            z = idx[("z", x, r, a, s)]
            rows.append({z: 1, idx[("e", x, s)]: -1}); lo.append(-np.inf); hi.append(0)
            rows.append({z: 1, idx[("d", r, s, a)]: -1}); lo.append(-np.inf); hi.append(0)
    Amat = lil_matrix((len(rows), n))
    for i, row in enumerate(rows):
        for j, v in row.items():
            Amat[i, j] = v
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(Amat.tocsr(), lo, hi), integrality=integ, bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False})
    val = -res.fun if res.x is not None else None
    dual = -getattr(res, "mip_dual_bound", res.fun) if res.x is not None else None
    return {"value": val, "dual_bound": dual, "proved_optimal": res.status == 0, "wall_sec": time.time() - t0}


def classical_n(pb, M, time_limit):
    """Exact max over message-relabelling representatives of class R[0]'s decoder (multisets of positions), as exp113 §1."""
    R, A = pb["R"], pb["A"]; reps = list(itertools.combinations_with_replacement(A[R[0]], M)); subs = []
    for rep in reps:
        subs.append({"rep": list(rep), **classical_milp_n(pb, M, time_limit, rep0=rep)}); print("cl rep", rep, subs[-1], flush=True)
    proved = all(s["proved_optimal"] for s in subs)
    return {"M": M, "value": max(s["value"] for s in subs), "certified_upper": max(s["value"] if s["proved_optimal"] else s["dual_bound"] for s in subs),
            "all_proved": proved, "subs": subs}


def quantum_d(pb, D, projective, seed0=0):
    X, R, A, P = pb["X"], pb["R"], pb["A"], pb["P"]; xi = {x: i for i, x in enumerate(X)}
    W = {r: torch.zeros(len(X), len(A[r]), dtype=torch.float64) for r in R}
    for (x, r, a), p in P.items():
        W[r][xi[x], A[r].index(a)] = p
    def meas(B):
        Bc = torch.complex(B[0], B[1])                                            # (n_out, D, D)
        if projective:                                                            # orthonormal basis: n_out == D, rank-1
            Q, _ = torch.linalg.qr(Bc[0]); return torch.einsum("ia,ja->aij", Q, Q.conj())
        G = Bc @ Bc.conj().transpose(-1, -2); S = G.sum(0); ev, U = torch.linalg.eigh(S)
        Sm = U @ torch.diag(ev.clamp(min=1e-12) ** -0.5).to(U.dtype) @ U.conj().T
        return Sm @ G @ Sm
    def value(Bs):                                                                # optimal states = top eigenvectors
        Ax = torch.zeros(len(X), D, D, dtype=torch.complex128)
        for r in R:
            E = meas(Bs[r]); Ax = Ax + torch.einsum("xa,aij->xij", W[r].to(E.dtype), E)
        return torch.linalg.eigvalsh(Ax)[:, -1].sum()
    best = -1.0; t0 = time.time()
    for rs in range(RESTARTS):
        g = torch.Generator().manual_seed(128000 + 97 * D + 7 * projective + rs + seed0)
        Bs = {r: torch.randn(2, len(A[r]), D, D, generator=g, dtype=torch.float64).requires_grad_() for r in R}
        opt = torch.optim.Adam(list(Bs.values()), lr=0.05); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-4)
        for _ in range(STEPS):
            loss = -value(Bs); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        with torch.no_grad():
            v = float(value(Bs)); ok = all(float(torch.linalg.eigvalsh(meas(Bs[r])).min()) >= -1e-10 and
                                           float((meas(Bs[r]).sum(0) - torch.eye(D, dtype=torch.complex128)).abs().max()) <= 1e-9 for r in R)
        if v > best and ok:
            best = v
    return {"D": D, "projective": projective, "lower_bound": best, "restarts": RESTARTS, "steps": STEPS, "wall_sec": time.time() - t0}


def lie_dim(V_over_Om):
    """Symmetric subspace {gg, (gr+rg)/sqrt2, rr}: drive Om/2 * sum_i (e^{-i phi}|r><g|_i + h.c.) -> sqrt2 couplings; detuning -delta * n_r; V |rr><rr|."""
    s2 = math.sqrt(2); Hx = np.zeros((3, 3), complex); Hx[0, 1] = Hx[1, 0] = Hx[1, 2] = Hx[2, 1] = s2 / 2
    Hy = np.zeros((3, 3), complex); Hy[0, 1] = Hy[1, 2] = -1j * s2 / 2; Hy[1, 0] = Hy[2, 1] = 1j * s2 / 2
    Hn = np.diag([0, 1, 2]).astype(complex); HV = np.diag([0, 0, 1]).astype(complex) * V_over_Om
    gens = [1j * h for h in (Hx, Hy, Hn) + ((HV,) if V_over_Om else ())]
    basis = []
    def add(M):
        v = np.concatenate([M.real.ravel(), M.imag.ravel()])
        for b in basis:
            v = v - (v @ b) * b
        nv = np.linalg.norm(v)
        if nv > 1e-9:
            basis.append(v / nv); return True
        return False
    frontier = [g for g in gens if add(g)]
    while frontier:
        new = []
        for a in frontier:
            for b in list(gens) + frontier:
                c = a @ b - b @ a
                if add(c):
                    new.append(c)
        frontier = new
    # u(3) has dim 9 (incl. identity phase); su(3) = 8.  Trace part: remove identity direction count
    tr = np.concatenate([np.zeros(9), np.eye(3).ravel()]); tr /= np.linalg.norm(tr)                      # direction of i*Identity
    has_id = abs(sum((b @ tr) ** 2 for b in basis) - 1) < 1e-6
    return {"V_over_Omega": V_over_Om, "lie_dim": len(basis), "contains_identity_direction": bool(has_id), "su3_full": len(basis) - int(has_id) >= 8}


def main():
    parts = sys.argv[1:] or ["s2", "s1", "cl"]
    out_j = os.path.join(ROOT, "results_exp128_spike%s.json" % ("" if parts == ["s2", "s1", "cl"] else "_" + "_".join(parts)))
    assert not os.path.exists(out_j), "Rule 3"
    pb = A100.problem(R110.uniform_rows(3)); res = {"doc": "127 SPIKE (205a8d0), THROWAWAY", "parts": parts}
    if "check" in parts:      # generalised MILP == exp100 MILP at M = 2^m (replaces the registered n=4 check, infeasible in spike time)
        pb2 = A100.problem(R110.uniform_rows(2))
        a = classical_milp_n(pb2, 2, 600)["value"]; b = A100.classical_milp(pb2, 1, 600)["value"]
        c = classical_n(pb2, 2, 600)["value"]
        res["G_MILP_EQUIV"] = {"K2_M2_mine": a, "K2_m1_exp100": b, "K2_M2_reps": c, "pass": bool(abs(a - b) < 1e-9 and abs(c - b) < 1e-9)}
        print("G_MILP_EQUIV", res["G_MILP_EQUIV"], flush=True)
    if "s2" in parts:
        res["S2"] = [lie_dim(v) for v in (0.0, 0.3, 1.0, 3.0)]; print("S2", res["S2"], flush=True)
    if "s1" in parts:
        res["S1_qutrit_projective"] = quantum_d(pb, 3, True); print("S1 proj", res["S1_qutrit_projective"], flush=True)
        res["S1_qutrit_povm"] = quantum_d(pb, 3, False); print("S1 povm", res["S1_qutrit_povm"], flush=True)
    if "cl" in parts:
        res["classical_3msg"] = classical_n(pb, 3, 1800); print("CL3", {k: v for k, v in res["classical_3msg"].items() if k != "subs"}, flush=True)
    res["two_bits_exact"] = 13 / 18
    json.dump(res, open(out_j, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
