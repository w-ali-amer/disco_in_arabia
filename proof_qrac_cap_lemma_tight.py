# -*- coding: utf-8 -*-
"""proof_qrac_cap_lemma.py -- doc 94 §9(c): CERTIFY the compositional-qubit 3->1 cap via the 3-parameter reduction.

Lemma (L3): F = |C_D C_A e1|^2 + |S_D C_A e2 + C_D S_A e3|^2 <= phi^2 for rotations A, D and unit e's
(C_R=(I+R)/2, S_R=(I-R)/2). Reduction (each step an upper bound):
  F <= G(a,b,g) = ||C_D C_A||^2 + max_{q in [|cos g|,1]} [ r^2 + s_b^2 eta^2 + 2 s_b c_b eta r q - s_b^2 r^2 q^2 ],
  r = sin a, eta = ||P_{perp m} C_A||, A = rotation 2a about n = z, D = rotation 2b about m = (sin g, 0, cos g),
  domain a in [0, pi/2], b in [0, pi/2], g in [0, pi] (covers SO(3) x SO(3) up to global rotation).
Lipschitz bounds (derived in doc 94 §10): |dG/da| <= 10, |dG/db| <= 6, |dG/dg| <= 8.
Branch-and-bound: a box with centre value G_c and half-widths (ha,hb,hg) is discarded when
G_c + 10 ha + 6 hb + 8 hg <= phi^2 + EPS. Writes proof_qrac_cap_lemma.json (Rule 3).
Then sup F <= phi^2 + EPS  =>  T <= 8 sqrt(phi^2 + EPS)  =>  P <= 1/2 + sqrt(phi^2 + EPS)/6.
"""
import json, math, hashlib, os, time
import numpy as np

OUT = "proof_qrac_cap_lemma_eps2e-4.json"; assert not os.path.exists(OUT)
PHI2 = (3 + math.sqrt(5)) / 2
EPS = 2e-4
LA, LB, LG = 10.0, 6.0, 8.0


def G(a, b, g):
    a, b, g = map(np.asarray, (a, b, g))
    N = a.size
    ca, sa = np.cos(2 * a), np.sin(2 * a)
    A = np.zeros((N, 3, 3)); A[:, 0, 0] = ca; A[:, 0, 1] = -sa; A[:, 1, 0] = sa; A[:, 1, 1] = ca; A[:, 2, 2] = 1
    m = np.stack([np.sin(g), np.zeros(N), np.cos(g)], 1)
    K = np.zeros((N, 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -m[:, 2], m[:, 1], m[:, 2], -m[:, 0], -m[:, 1], m[:, 0]
    th = 2 * b
    D = np.eye(3)[None] + np.sin(th)[:, None, None] * K + (1 - np.cos(th))[:, None, None] * (K @ K)
    I = np.eye(3)[None]
    CA, CD = (I + A) / 2, (I + D) / 2
    Pm = I - m[:, :, None] * m[:, None, :]
    mu = np.linalg.norm(CD @ CA, ord=2, axis=(1, 2)) ** 2
    eta = np.linalg.norm(Pm @ CA, ord=2, axis=(1, 2))
    r = np.sin(a); sb, cb = np.sin(b), np.cos(b)
    lo = np.abs(np.cos(g))
    def f(q):
        return r * r + sb * sb * eta * eta + 2 * sb * cb * eta * r * q - sb * sb * r * r * q * q
    best = np.maximum(f(lo), f(np.ones(N)))
    with np.errstate(divide="ignore", invalid="ignore"):
        qs = np.where(sb * r > 1e-15, cb * eta / (sb * r), np.inf)
    inside = (qs >= lo) & (qs <= 1)
    best = np.where(inside, np.maximum(best, f(np.where(inside, qs, 1.0))), best)
    return mu + best


t0 = time.time()
n0 = (64, 64, 128)
ha, hb, hg = (math.pi / 2) / (2 * n0[0]), (math.pi / 2) / (2 * n0[1]), math.pi / (2 * n0[2])
ga = (np.arange(n0[0]) * 2 + 1) * ha; gb = (np.arange(n0[1]) * 2 + 1) * hb; gg = (np.arange(n0[2]) * 2 + 1) * hg
Aa, Bb, Gg = np.meshgrid(ga, gb, gg, indexing="ij"); Aa, Bb, Gg = Aa.ravel(), Bb.ravel(), Gg.ravel()
total, best_lower, it = 0, -1.0, 0
while Aa.size:
    it += 1
    vals = np.concatenate([G(Aa[i:i + 200000], Bb[i:i + 200000], Gg[i:i + 200000]) for i in range(0, Aa.size, 200000)])
    total += Aa.size; best_lower = max(best_lower, float(vals.max()))
    keep = vals + LA * ha + LB * hb + LG * hg > PHI2 + EPS - 1e-9
    Aa, Bb, Gg = Aa[keep], Bb[keep], Gg[keep]
    print("iter %d kept %d total %d best %.10f t=%.0fs" % (it, Aa.size, total, best_lower, time.time() - t0), flush=True)
    ha, hb, hg = ha / 2, hb / 2, hg / 2
    sa_, sb_, sg_ = np.array([-1, 1]) * ha, np.array([-1, 1]) * hb, np.array([-1, 1]) * hg
    Aa = np.concatenate([Aa + da for da in sa_ for db in sb_ for dg in sg_])
    Bb = np.concatenate([Bb + db for da in sa_ for db in sb_ for dg in sg_])
    Gg = np.concatenate([Gg + dg for da in sa_ for db in sb_ for dg in sg_])
    if it > 30:
        raise RuntimeError("no termination")
res = {"certified_sup_F_leq": PHI2 + EPS, "phi2": PHI2, "max_found_F": best_lower,
       "certified_P_leq": 0.5 + math.sqrt(PHI2 + EPS) / 6, "target_P": 0.5 + (1 + math.sqrt(5)) / 12,
       "boxes": total, "iterations": it, "lipschitz": [LA, LB, LG], "eps_F": EPS, "wall_sec": time.time() - t0,
       "self_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res, indent=1))
