# -*- coding: utf-8 -*-
"""Exp 36b: entanglement scaling with ARBITRARY referent-pair coupling
(discourse co-reference is not a path graph).  Otherwise as exp36."""
import numpy as np
import json

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
out = {}
for k in (8, 12, 16):
    rng = np.random.default_rng(11)
    sched = []
    for _ in range(T):
        i, j = rng.choice(k, 2, replace=False)
        sched.append((int(i), int(j), rng.uniform(0,2,4),
                      rng.uniform(0,2), rng.uniform(0,2)))
    for reg in ("zz","zx4"):
        psi = np.zeros(2**k, complex); psi[0]=1.0
        H = np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
        for q in range(k): psi = ap1(psi,H,q,k)
        S = []
        for (i, j, loc, th, ph) in sched:
            psi = ap1(psi, rz(loc[0])@rx(loc[1]), i, k)
            psi = ap1(psi, rz(loc[2])@rx(loc[3]), j, k)
            psi = ap2f(psi, crz(th), i, j, k)
            if reg=="zx4": psi = ap2f(psi, crx(ph), i, j, k)
            S.append(half_entropy(psi,k))
        out[(k,reg)] = (S[29], S[59], S[-1])
        print(f"[36b] k={k:2d} {reg:4s}: S@30={S[29]:.3f} S@60={S[59]:.3f} "
              f"S@120={S[119]:.3f} S@200={S[-1]:.3f} maxS={k//2} "
              f"MPS-bond@60~{2**S[59]:.0f} @200~{2**S[-1]:.0f}", flush=True)
json.dump({f"k{k}_{reg}": {"S30": v[0], "S60": v[1], "S200": v[2]}
           for (k, reg), v in out.items()},
          open("results_exp36b.json", "w"), indent=2)
print("[36b] DONE", flush=True)
