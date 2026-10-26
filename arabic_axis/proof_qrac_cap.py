# -*- coding: utf-8 -*-
"""proof_qrac_cap.py -- doc 94 §3 amendment: (a) explicit ATTAINMENT of 1/2 + (1+sqrt5)/12 by a compositional code
(analytic construction, checked numerically); (b) CERTIFIED upper bound for FREE REBIT 3->1 codes by Lipschitz
branch-and-bound on the two relative angles. Writes proof_qrac_cap.json (Rule 3).

(b) For planar (rebit) codes with optimal codewords, T = max_{e1,e2,e3 in S^1} sum_{s in {+-1}^3} |s1 e1 + s2 e2 + s3 e3|
    = F(beta, gamma) = 2 * sum_{s2,s3} |1 + s2 e^{i beta} + s3 e^{i gamma}|  (e1 fixed at angle 0 by rotation invariance).
    P = 1/2 + T/48.  Each |.| term is 1-Lipschitz in beta and in gamma, so |dF/dbeta|, |dF/dgamma| <= 8, and on a box of
    half-widths (h,h): F <= F(centre) + 16 h.  Boxes with bound <= M + eps are discarded; others are split. Terminates
    when no box remains: then sup F <= M + eps (float safety margin added).
"""
import json, math, hashlib, os
import numpy as np

OUT = "proof_qrac_cap.json"; assert not os.path.exists(OUT)
M = 4 * (1 + math.sqrt(5)); PHI = (1 + math.sqrt(5)) / 2
res = {}

# ---------------- (a) attainment
b = 0.5 * math.atan(2.0)
def Rz(t): return np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
def Rn(n, t):
    n = n / np.linalg.norm(n); K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K
cb, sb = math.cos(b), math.sin(b)
sig, kap = cb / math.sqrt(cb**2 + (sb + cb)**2), (sb + cb) / math.sqrt(cb**2 + (sb + cb)**2)
w1 = 2 * sig * np.array([1.0, 0, 0]); w2 = 2 * kap * np.array([0, 1.0, 0])
v0, v1 = (w1 + w2) / 2, (w2 - w1) / 2
D = Rz(2 * b); A = Rn(Rz(b) @ np.array([1.0, 0, 0]), math.pi)
codes = {(b1, b2, b3): np.linalg.matrix_power(A, b3) @ np.linalg.matrix_power(D, b2) @ (v0 if b1 == 0 else v1)
         for b1 in (0, 1) for b2 in (0, 1) for b3 in (0, 1)}
S = [sum(((-1) ** w[i]) * c for w, c in codes.items()) for i in range(3)]
T = sum(np.linalg.norm(s) for s in S)
res["a_attainment"] = {"b": b, "tan2b": math.tan(2 * b), "sigma": sig, "kappa": kap,
                       "unit_codewords": all(abs(np.linalg.norm(c) - 1) < 1e-12 for c in codes.values()),
                       "planar": float(max(abs(c[2]) for c in codes.values())),
                       "S_lengths": [float(np.linalg.norm(s)) for s in S],
                       "predicted": [8 * sig * cb, 8 * kap * sb, 8 * kap * cb],
                       "T": float(T), "T_target": M, "P": 0.5 + T / 48, "P_target": 0.5 + (1 + math.sqrt(5)) / 12}

# ---------------- (b) certified free-rebit bound
def F(bet, gam):
    z2, z3 = np.exp(1j * bet), np.exp(1j * gam)
    return 2 * sum(np.abs(1 + s2 * z2 + s3 * z3) for s2 in (1, -1) for s3 in (1, -1))
EPS, SAFETY = 1e-4, 1e-9
n0 = 512
h = math.pi / n0                                             # half-width
g = (np.arange(n0) * 2 + 1) * h
B, G = np.meshgrid(g, g, indexing="ij"); B, G = B.ravel(), G.ravel()
boxes_total, best_lower, it = 0, -1.0, 0
while B.size:
    it += 1
    val = F(B, G); boxes_total += B.size
    best_lower = max(best_lower, float(val.max()))
    keep = val + 16 * h > M + EPS - SAFETY
    B, G = B[keep], G[keep]
    h /= 2
    B = np.concatenate([B - h, B - h, B + h, B + h]); G = np.concatenate([G - h, G + h, G - h, G + h])
    if it > 40:
        raise RuntimeError("did not terminate")
res["b_free_rebit_certificate"] = {"certified_sup_T_leq": M + EPS, "certified_sup_P_leq": 0.5 + (M + EPS) / 48,
                                   "max_found_T": best_lower, "target_T": M, "boxes_evaluated": boxes_total,
                                   "iterations": it, "lipschitz": "dF/dbeta, dF/dgamma <= 8 (each modulus term 1-Lipschitz)",
                                   "note": "floating point: safety margin 1e-9 on the prune test; values computed in float64"}
res["self_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
