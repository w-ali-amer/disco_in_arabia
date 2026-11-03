# -*- coding: utf-8 -*-
"""gen_ub.py -- doc 141: Gram/AM-GM upper-bound certificate for every qubit strategy on the K = 2 real-count game.
U = 1/2 + sum_{a<b} w_ab / lam_ab + (1/8) sum_v mu_v, valid when diag(mu) - L >= 0, L = Laplacian with edge weights 2 w_ab lam_ab.
Solves for (lam, mu) in floats, rationalises, proves diag(mu) - L > 0 by an exact LDL^T in Fractions. Lean re-checks everything."""
import itertools, json, sys
import numpy as np, cvxpy as cp
from fractions import Fraction as F
c = [9, 3, 1265, 20, 275, 1537]; P = list(itertools.combinations(range(6), 2))
Z = sum(c[a] * c[b] for a in range(6) for b in range(6) if a != b); w = {p: F(c[p[0]] * c[p[1]], 2 * Z) for p in P}
lam = cp.Variable(len(P), pos=True); mu = cp.Variable(6)
L = 0
for i, (a, b) in enumerate(P):
    e = np.zeros((6, 6)); e[a, a] = e[b, b] = 1; e[a, b] = e[b, a] = -1
    L = L + 2 * float(w[(a, b)]) * lam[i] * e
obj = 0.5 + sum(float(w[p]) * cp.inv_pos(lam[i]) for i, p in enumerate(P)) + cp.sum(mu) / 8
prob = cp.Problem(cp.Minimize(obj), [cp.diag(mu) - L >> 0]); prob.solve(solver="CLARABEL")
print("float U", prob.value, prob.status)

def ldl(A):
    n = len(A); Lf = [[F(0)] * n for _ in range(n)]; d = [F(0)] * n
    for j in range(n):
        d[j] = A[j][j] - sum(Lf[j][k] ** 2 * d[k] for k in range(j))
        if d[j] <= 0: return None
        Lf[j][j] = F(1)
        for i in range(j + 1, n):
            Lf[i][j] = (A[i][j] - sum(Lf[i][k] * Lf[j][k] * d[k] for k in range(j))) / d[j]
    return Lf, d

DEN = 10 ** 6
lq = {p: F(float(lam.value[i])).limit_denominator(DEN) for i, p in enumerate(P)}
Lq = [[F(0)] * 6 for _ in range(6)]
for (a, b), l in lq.items():
    k = 2 * w[(a, b)] * l; Lq[a][a] += k; Lq[b][b] += k; Lq[a][b] -= k; Lq[b][a] -= k
for eps_exp in range(9, 2, -1):
    eps = F(1, 10 ** eps_exp)
    mq = [F(float(mu.value[v])).limit_denominator(DEN) + eps for v in range(6)]
    A = [[(mq[i] if i == j else 0) - Lq[i][j] for j in range(6)] for i in range(6)]
    r = ldl(A)
    if r: break
Lf, d = r
U = F(1, 2) + sum(w[p] / lq[p] for p in P) + sum(mq) / 8
print("eps", eps, "U exact", float(U), "gap to 0.958385", float(U) - 0.958385, "to L1 0.958392", float(U) - 0.958392)
out = {"Z": Z, "pairs": P, "lam": {"%d,%d" % p: [lq[p].numerator, lq[p].denominator] for p in P},
       "mu": [[m.numerator, m.denominator] for m in mq], "eps": [eps.numerator, eps.denominator],
       "ldl_L": [[[x.numerator, x.denominator] for x in row] for row in Lf], "ldl_d": [[x.numerator, x.denominator] for x in d],
       "U": [U.numerator, U.denominator], "U_float": float(U), "U_solver": prob.value}
json.dump(out, open(sys.argv[1], "w"), indent=1)
