# -*- coding: utf-8 -*-
"""Exp 36: entanglement scaling of discourse chains — zz vs zx4 entanglers.

Question (decision-relevant for the hardware-advantage program): does the
mixed CRz.CRx entangler push discourse-chain circuits out of the
tensor-network-approximable regime, relative to pure diagonal CRz?

Model of the DisCoCirc port: k referent wires, T sentences; each sentence
applies word-embedding-style local rotations Rz.Rx on a wire pair (same
angles for both regimes, drawn once) then the verb entangler:
    zz  : CRz(theta)
    zx4 : CRz(theta) . CRx(phi)
Metric: half-chain von Neumann entropy S(t) (base 2).  Tensor-network cost
per cut ~ 2^S.  Report growth rate, S at T=30 (a realistic text), final S.
"""
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
def ap2(psi,U,i,k):
    psi=psi.reshape([2]*k)
    psi=np.moveaxis(psi,(i,i+1),(0,1)).reshape(4,-1); psi=U@psi
    return np.moveaxis(psi.reshape([2,2]+[2]*(k-2)),(0,1),(i,i+1)).reshape(-1)

def half_entropy(psi,k):
    m = psi.reshape(2**(k//2), -1)
    s = np.linalg.svd(m, compute_uv=False)
    p = s**2; p = p[p>1e-15]
    return float(-(p*np.log2(p)).sum())

T = 60
out = {}
for k in (8, 12, 16):
    rng = np.random.default_rng(11)
    # one shared schedule: pair index + 4 local angles + theta + phi per step
    sched = [ (int(rng.integers(0,k-1)), rng.uniform(0,2,4),
               rng.uniform(0,2), rng.uniform(0,2)) for _ in range(T) ]
    for reg in ("zz","zx4"):
        psi = np.zeros(2**k, complex); psi[0]=1.0
        # initial |+>^k as in the IQP word layer
        H = np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
        for q in range(k): psi = ap1(psi,H,q,k)
        S = []
        for (i, loc, th, ph) in sched:
            psi = ap1(psi, rz(loc[0])@rx(loc[1]), i, k)
            psi = ap1(psi, rz(loc[2])@rx(loc[3]), i+1, k)
            psi = ap2(psi, crz(th), i, k)
            if reg=="zx4": psi = ap2(psi, crx(ph), i, k)
            S.append(half_entropy(psi,k))
        rate = np.polyfit(range(10,min(30,T)), S[10:min(30,T)], 1)[0]
        out[(k,reg)] = (S[29], S[-1], rate)
        print(f"[36] k={k:2d} {reg:4s}: S@30={S[29]:.3f}  S@60={S[-1]:.3f}  "
              f"rate(10-30)={rate:.4f}/sentence  maxS={k//2}  "
              f"MPS-bond@30~{2**S[29]:.0f}", flush=True)
json.dump({f"k{k}_{reg}": {"S30": v[0], "S60": v[1], "rate": v[2]}
           for (k, reg), v in out.items()},
          open("results_exp36.json", "w"), indent=2)
print("[36] DONE", flush=True)
