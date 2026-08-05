# -*- coding: utf-8 -*-
"""Exp 36c: entanglement scaling with the ACTUAL solved verb parameters
(exp33 zx4 T3b thetas), arbitrary-pair coupling.  The decisive check:
do semantic (solved) instances, not random ones, still reach volume law?"""
import numpy as np, json

r = json.load(open("results_exp33_s0.json"))
TUPLES = [v["T3b_theta"] for v in r["variants"]["zx4"]["verbs"].values()]
print("[36c] solved tuples:", [[round(x,2) for x in t] for t in TUPLES],
      flush=True)

def crz(t):
    d = np.ones(4, complex); d[3] = np.exp(1j*np.pi*t); return np.diag(d)
def crx(t):
    m = np.eye(4, dtype=complex)
    c, s = np.cos(np.pi*t/2), -1j*np.sin(np.pi*t/2)
    m[2,2]=m[3,3]=c; m[2,3]=m[3,2]=s; return m
def rz(t): return np.diag([1, np.exp(1j*np.pi*t)]).astype(complex)
def rx(t):
    c, s = np.cos(np.pi*t/2), -1j*np.sin(np.pi*t/2)
    return np.array([[c,s],[s,c]], complex)
def ap1(psi,U,i,k):
    psi=psi.reshape([2]*k); psi=np.moveaxis(psi,i,0).reshape(2,-1)
    psi=U@psi; return np.moveaxis(psi.reshape([2]+[2]*(k-1)),0,i).reshape(-1)
def ap2f(psi,U,i,j,k):
    psi=psi.reshape([2]*k)
    psi=np.moveaxis(psi,(i,j),(0,1)).reshape(4,-1); psi=U@psi
    return np.moveaxis(psi.reshape([2,2]+[2]*(k-2)),(0,1),(i,j)).reshape(-1)
def half_entropy(psi,k):
    m = psi.reshape(2**(k//2), -1)
    s = np.linalg.svd(m, compute_uv=False)
    p = s**2; p = p[p>1e-15]
    return float(-(p*np.log2(p)).sum())

T = 200
for k in (12, 16):
    rng = np.random.default_rng(11)
    psi = np.zeros(2**k, complex); psi[0]=1.0
    H = np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
    for q in range(k): psi = ap1(psi,H,q,k)
    S=[]
    for t in range(T):
        i, j = rng.choice(k, 2, replace=False)
        loc = rng.uniform(0,2,4)
        th1, th2, ph1, ph2 = TUPLES[int(rng.integers(0,len(TUPLES)))]
        psi = ap1(psi, rz(loc[0])@rx(loc[1]), int(i), k)
        psi = ap1(psi, rz(loc[2])@rx(loc[3]), int(j), k)
        psi = ap2f(psi, crz(th1), int(i), int(j), k)
        psi = ap2f(psi, crx(ph1), int(i), int(j), k)
        S.append(half_entropy(psi,k))
    print(f"[36c] k={k:2d} solved-zx4: S@30={S[29]:.3f} S@60={S[59]:.3f} "
          f"S@200={S[-1]:.3f} maxS={k//2} MPS-bond@60~{2**S[59]:.0f} "
          f"@200~{2**S[-1]:.0f}", flush=True)
print("[36c] DONE", flush=True)
