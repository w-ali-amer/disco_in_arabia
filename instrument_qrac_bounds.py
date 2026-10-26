# -*- coding: utf-8 -*-
"""instrument_qrac_bounds.py -- BOUND CALCULATOR for doc 94 (exp99), run BEFORE the criteria are written
(doc 55 §19.2 ordering rule). Trains nothing on any task.

(1) Classical 1-bit RAC maximum by EXHAUSTIVE enumeration (n = 2, 3 binary features, uniform, random access):
    encoders {0,1}^n -> {0,1}, decoders (bit, query i) -> {0,1}. Shared randomness = convex combination -> same max.
(2) Qubit n->1 optimum: analytic (1 + 1/sqrt(n))/2 for n = 2, 3; verified numerically by optimisation over Bloch vectors.
(3) Rebit (real amplitudes: Bloch vectors in the XZ plane, measurement axes in the XZ plane) optimum for n = 2, 3, by
    numerical optimisation with many restarts (average single-shot success).
(4) Compositional construction check: codeword = product of commuting morpheme boxes (RY gender, RY/RZ others) reaches the
    free optimum (explicit angles).
Writes diagnostic_qrac_bounds.json (Rule 3).
"""
import itertools, json, math, os, hashlib
import torch

OUT = "diagnostic_qrac_bounds.json"
assert not os.path.exists(OUT)
res = {}

# (1) classical exhaustive
for n in (2, 3):
    words = list(itertools.product((0, 1), repeat=n))
    best = 0.0
    for enc in itertools.product((0, 1), repeat=len(words)):
        for dec in itertools.product((0, 1), repeat=2 * n):          # dec[b*n + i]
            s = sum(dec[enc[w] * n + i] == words[w][i] for w in range(len(words)) for i in range(n))
            best = max(best, s / (len(words) * n))
    res["classical_1bit_max_n%d" % n] = best


def optimise(n, rebit, restarts=200, steps=1500, seed=0):
    g = torch.Generator().manual_seed(seed)
    best = 0.0
    for r in range(restarts):
        dim = 2 if rebit else 3
        C = torch.randn(2 ** n, dim, generator=g, requires_grad=True)      # codeword Bloch vectors
        A = torch.randn(n, dim, generator=g, requires_grad=True)           # measurement axes
        opt = torch.optim.Adam([C, A], lr=0.05)
        words = torch.tensor(list(itertools.product((0, 1), repeat=n)), dtype=torch.float32)
        sign = 1 - 2 * words                                               # bit 0 -> +1
        for _ in range(steps):
            c = C / C.norm(dim=1, keepdim=True).clamp(min=1e-9)
            a = A / A.norm(dim=1, keepdim=True).clamp(min=1e-9)
            p = 0.5 * (1 + sign * (c @ a.T))                                # P(correct) per word, query
            loss = -p.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        best = max(best, float(-loss))
    return best


for n in (2, 3):
    res["qubit_analytic_n%d" % n] = (1 + 1 / math.sqrt(n)) / 2
    res["qubit_numeric_n%d" % n] = optimise(n, rebit=False, restarts=60)
    res["rebit_numeric_n%d" % n] = optimise(n, rebit=True, restarts=200)

# (4) compositional construction: 2->1 with two RY boxes (angles add); 3->1 with RY (gender) + two RZ boxes (azimuth adds)
def bloch_ry_rz(theta, phi):
    return torch.tensor([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])
# 2->1: polar angle = a_g + a_n with a_g in {0, pi/2}, a_n in {pi/4, pi/4+pi}; axes Z and X
tot = 0.0
for g_, n_ in itertools.product((0, 1), repeat=2):
    th = (math.pi / 2) * g_ + math.pi / 4 + math.pi * n_ * 0     # construct below
    # codeword angles for (g,n): pi/4 + {0, pi/2} for g, then reflect for n via + pi*?  -> use explicit optimal set
res["compositional_note"] = ("2->1: RY(a_g)·RY(a_n) = RY(a_g+a_n); choosing a_g in {pi/4, 3pi/4}, a_n in {0, pi} gives "
                             "polar angles {pi/4,3pi/4,5pi/4,7pi/4}: the four optimal 2->1 codewords (XZ plane). "
                             "3->1: RY(theta_g) with theta_g in {arccos(1/sqrt3), pi-arccos(1/sqrt3)} then RZ(b1)·RZ(b2) = "
                             "RZ(b1+b2) with b1 in {pi/4, 5pi/4}, b2 in {0, pi/2}: azimuths {pi/4,3pi/4,5pi/4,7pi/4} -> the 8 cube-corner codewords.")
# numeric verification of the compositional constructions
def succ(codes, axes, n):
    words = list(itertools.product((0, 1), repeat=n)); tot = 0.0
    for w, c in zip(words, codes):
        for i in range(n):
            s = 1 - 2 * w[i]; tot += 0.5 * (1 + s * float(c @ axes[i]))
    return tot / (len(words) * n)
codes2 = [bloch_ry_rz(ag + an, 0.0) for ag, an in itertools.product((math.pi / 4, 3 * math.pi / 4), (0.0, math.pi))]
axes2 = [torch.tensor([0., 0., 1.]), torch.tensor([1., 0., 0.])]
# map words (g,n) -> codes; measure axis choices by sign fit (best of +-axis per query)
def best_axes(codes, axes, n):
    best = 0.0
    for sg in itertools.product((1, -1), repeat=n):
        best = max(best, succ(codes, [a * s for a, s in zip(axes, sg)], n))
    return best
res["compositional_2to1"] = best_axes(codes2, axes2, 2)
tg = math.acos(1 / math.sqrt(3))
codes3 = [bloch_ry_rz(t, b1 + b2) for t, b1, b2 in itertools.product((tg, math.pi - tg), (math.pi / 4, 5 * math.pi / 4), (0.0, math.pi / 2))]
axes3 = [torch.tensor([0., 0., 1.]), torch.tensor([1., 1., 0.]) / math.sqrt(2), torch.tensor([1., -1., 0.]) / math.sqrt(2)]
best3 = 0.0
for perm in itertools.permutations(range(3)):
    best3 = max(best3, best_axes(codes3, [axes3[p] for p in perm], 3))
res["compositional_3to1"] = best3
json.dump({"doc": "94 pre-registration bound calculator", "results": res,
           "self_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
