# -*- coding: utf-8 -*-
"""Exp 37: gate-to-analog compilation PoC — can a global-drive Rydberg
sequence reproduce the solved verb blocks CRz(theta).CRx(phi)?

The "transformer": target 2-qubit block -> (interaction strength V i.e.
atom distance, plus B global pulse blocks (Omega, delta, phase, tau)).
Rydberg H per block (hbar=1, |1>=Rydberg):
  H = (Om/2)(cos p (X1+X2) + sin p (Y1+Y2)) - d (n1+n2) + V n1 n2
Local Z-frame freedom absorbed by free pre/post Rz on each qubit.
Metric: average gate fidelity F = (|Tr M|^2 + Tr M M+)/(d(d+1)), M=Ut+ Ua.
Sanity target: CZ (Levine-Pichler regime, should compile ~perfectly).
"""
import json
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

X = np.array([[0,1],[1,0]], complex)
Y = np.array([[0,-1j],[1j,0]], complex)
N = np.array([[0,0],[0,1]], complex)
I2 = np.eye(2, dtype=complex)
def kron(a,b): return np.kron(a,b)
X1, X2 = kron(X,I2), kron(I2,X)
Y1, Y2 = kron(Y,I2), kron(I2,Y)
N1, N2 = kron(N,I2), kron(I2,N)
NN = kron(N,N)

def crz(t):
    d = np.ones(4, complex); d[3] = np.exp(1j*np.pi*t); return np.diag(d)
def crx(t):
    m = np.eye(4, dtype=complex)
    c, s = np.cos(np.pi*t/2), -1j*np.sin(np.pi*t/2)
    m[2,2]=m[3,3]=c; m[2,3]=m[3,2]=s; return m
def rzl(a): return np.diag([1, np.exp(1j*a)]).astype(complex)

B = 3
def analog_unitary(p):
    V = p[0]**2                      # keep positive
    U = np.eye(4, dtype=complex)
    for b in range(B):
        Om, de, ph, tau = p[1+4*b : 5+4*b]
        tau = tau**2                 # positive time
        H = (Om/2)*(np.cos(ph)*(X1+X2) + np.sin(ph)*(Y1+Y2)) \
            - de*(N1+N2) + V*NN
        U = expm(-1j*H*tau) @ U
    return U

def fidelity(p, Ut):
    za, zb, zc, zd = p[-4:]
    Ua = kron(rzl(zc), rzl(zd)) @ analog_unitary(p[:-4]) @ kron(rzl(za), rzl(zb))
    M = Ut.conj().T @ Ua
    return (abs(np.trace(M))**2 + np.trace(M @ M.conj().T).real) / 20.0

r = json.load(open("results_exp33_s0.json"))
targets = {"CZ_sanity": np.diag([1,1,1,-1]).astype(complex)}
for v, d in r["variants"]["zx4"]["verbs"].items():
    th = d["T3b_theta"]
    targets[v] = crz(th[0]) @ crx(th[2])

OUT = {}
npar = 1 + 4*B + 4
for name, Ut in targets.items():
    best = None
    for s in range(12):
        x0 = np.random.default_rng(s).uniform(-2, 2, npar)
        res = minimize(lambda p: 1 - fidelity(p, Ut), x0,
                       method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-10,
                                "fatol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    F = 1 - best.fun
    p = best.x
    V = p[0]**2
    taus = [p[4+4*b]**2 for b in range(B)]
    oms = [abs(p[1+4*b]) for b in range(B)]
    OUT[name] = {"fidelity": float(F), "V": float(V),
                 "taus": [float(t) for t in taus],
                 "omegas": [float(o) for o in oms]}
    print(f"[37] {name}: F={F:.6f} | V={V:.2f} rad/us | "
          f"Om={[round(o,2) for o in oms]} | tau={[round(t,3) for t in taus]} us",
          flush=True)
json.dump(OUT, open("results_exp37.json", "w"), indent=2, ensure_ascii=False)
print("[37] DONE", flush=True)
