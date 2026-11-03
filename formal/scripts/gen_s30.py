# -*- coding: utf-8 -*-
"""gen_s30.py -- doc 140: generate S30 (real-qubit strategy, integer vectors |coord| <= 30) and the classical argmax; emit Lean data + expected rationals.
Lean recomputes every number; nothing here is trusted by the proof."""
import itertools, json, sys
import numpy as np
from fractions import Fraction as F
c = [9, 3, 1265, 20, 275, 1537]; X = list(itertools.permutations(range(6), 2)); D = 30
Z = sum(c[a] * c[b] for a, b in X); W = {x: F(c[x[0]] * c[x[1]], 2 * Z) for x in X}
def matches(l, a, b): return int(((l >> a) & 1) == 0) + int(((l >> b) & 1) == 1)
cl = max(((sum(W[(a, b)] * max(matches(l0, a, b), matches(l1, a, b)) for a, b in X), l0, l1) for l0 in range(64) for l1 in range(64)), key=lambda t: (t[0], -t[1], -t[2]))
cval, l0, l1 = cl
rng = np.random.default_rng(0); best = (-1,)
def val(th, ph): return sum(float(W[(a, b)]) * (np.cos(th[i] - ph[a]) ** 2 + np.sin(th[i] - ph[b]) ** 2) for i, (a, b) in enumerate(X))
for r in range(200):
    ph = rng.uniform(0, np.pi, 6); th = np.zeros(30)
    for it in range(150):
        for i, (a, b) in enumerate(X): th[i] = np.angle(np.exp(2j * ph[a]) - np.exp(2j * ph[b])) / 2
        for v in range(6):
            cand = ph[v] + np.linspace(-0.05, 0.05, 21)
            ph[v] = cand[int(np.argmax([val(th, np.concatenate([ph[:v], [q], ph[v + 1:]])) for q in cand]))]
    v = val(th, ph)
    if v > best[0]: best = (v, th.copy(), ph.copy())
_, th, ph = best
iv = lambda t: (int(round(np.cos(t) * D)), int(round(np.sin(t) * D)))
psi = {x: iv(th[i]) for i, x in enumerate(X)}; m = [iv(p) for p in ph]
q = F(0)
for (a, b) in X:
    s, ma, mb = psi[(a, b)], m[a], m[b]; n = s[0] ** 2 + s[1] ** 2
    q += W[(a, b)] * (F((s[0] * ma[0] + s[1] * ma[1]) ** 2, n * (ma[0] ** 2 + ma[1] ** 2)) + 1 - F((s[0] * mb[0] + s[1] * mb[1]) ** 2, n * (mb[0] ** 2 + mb[1] ** 2)))
enc = {x: max((0, 1), key=lambda s: (matches((l0, l1)[s], *x), -s)) for x in X}
out = {"Z": Z, "classical": [cval.numerator, cval.denominator], "l0": l0, "l1": l1, "q30": [q.numerator, q.denominator], "q30_float": float(q),
       "margin": float(q - cval), "psi": {"%d,%d" % x: psi[x] for x in X}, "m": m, "enc": {"%d,%d" % x: enc[x] for x in X}}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: out[k] for k in ("Z", "classical", "l0", "l1", "q30", "q30_float", "margin")}))
