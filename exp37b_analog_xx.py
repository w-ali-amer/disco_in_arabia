# -*- coding: utf-8 -*-
"""Exp 37b: Rydberg global-drive compilation of the exp38 CRz.XX solved
blocks.  Both components are exchange-symmetric, so the exp37 obstruction
is absent by construction — prediction: F -> 1 for ALL verbs."""
import json
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

X = np.array([[0,1],[1,0]], complex); Y = np.array([[0,-1j],[1j,0]], complex)
N = np.array([[0,0],[0,1]], complex); I2 = np.eye(2, dtype=complex)
X1, X2 = np.kron(X,I2), np.kron(I2,X)
Y1, Y2 = np.kron(Y,I2), np.kron(I2,Y)
N1, N2 = np.kron(N,I2), np.kron(I2,N); NN = np.kron(N,N)
XXM = np.kron(X,X)

def crz(t):
    d = np.ones(4, complex); d[3] = np.exp(1j*np.pi*t); return np.diag(d)
def xxg(t):
    return (np.cos(np.pi*t/2)*np.eye(4,dtype=complex)
            - 1j*np.sin(np.pi*t/2)*XXM)
def rzl(a): return np.diag([1, np.exp(1j*a)]).astype(complex)

B = 3
def analog_unitary(p):
    V = p[0]**2
    U = np.eye(4, dtype=complex)
    for b in range(B):
        Om, de, ph, tau = p[1+4*b : 5+4*b]
        H = (Om/2)*(np.cos(ph)*(X1+X2) + np.sin(ph)*(Y1+Y2)) \
            - de*(N1+N2) + V*NN
        U = expm(-1j*H*(tau**2)) @ U
    return U

def fidelity(p, Ut):
    za, zb, zc, zd = p[-4:]
    Ua = np.kron(rzl(zc), rzl(zd)) @ analog_unitary(p[:-4]) @ np.kron(rzl(za), rzl(zb))
    M = Ut.conj().T @ Ua
    return (abs(np.trace(M))**2 + np.trace(M @ M.conj().T).real) / 20.0

r = json.load(open("results_exp38_s0.json"))
npar = 1 + 4*B + 4
for v, d in r["variants"]["zxx4"]["verbs"].items():
    th = d["T3b_theta"]
    Ut = crz(th[0]) @ xxg(th[2])
    best = None
    for s in range(12):
        x0 = np.random.default_rng(s).uniform(-2, 2, npar)
        res = minimize(lambda p: 1 - fidelity(p, Ut), x0,
                       method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-10,
                                "fatol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    print(f"[37b] {v}: F={1-best.fun:.6f} (theta={th[0]:.3f}, xx={th[2]:.3f})",
          flush=True)
print("[37b] DONE", flush=True)
