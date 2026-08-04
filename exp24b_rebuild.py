# -*- coding: utf-8 -*-
"""Exp 24b (rebuilt): disambiguation as purification — fresh engine.

Custom 4-qubit family [C=sense control, N=noun, S=sentence, X=context],
exact statevector simulation, no lambeq dependency (the 2025 H3 machinery
is bit-rotted at the tket boundary).

Construction:
  C: Ry(theta) — sense prior (theta=0 pure sense A, theta=0.25 equal mix (half-turn Ry))
  N: controlled preparation — block-diag(E_A, E_B), E_s = Rx a Rz b Rx c,
     sense parameters taken from exp23 Phase-2 SENSE ANCHORS for the real
     ambiguous words (رجل man/leg, جمل camel/sentence) or synthetic with
     distance dial Delta
  S: verb coupling — H, CRz(N->S), H  (IQP-style, fixed)
  X: context word whose preparation depends on C — block-diag(Rx(x), Rx(x+kappa));
     kappa = context informativeness (0 = useless context)
Readout: rho_S by partial trace.
  UNREAD context: trace X  |  READ context: measure X, condition, average.

Pre-registered:
  P1 ambiguous (theta=0.5) -> S(rho_S) > 0; fixed-sense (theta=0) -> ~0
  P2 S(rho_S) monotone in theta on [0, 0.5]
  P3 S(rho_S) increases with sense distance Delta (identical senses -> pure)
  P4 PURIFICATION: E_x[S(rho_S|x)] < S(rho_S unread), gap grows with kappa
Real-word table: رجل and جمل at equal prior, kappa sweep.
"""
import json, hashlib
import numpy as np

RNG = np.random.default_rng(42)

def h01(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

def rx(t):
    c, s = np.cos(np.pi * t), np.sin(np.pi * t)
    return np.array([[c, -1j * s], [-1j * s, c]])

def rz(t):
    return np.diag([np.exp(-1j * np.pi * t), np.exp(1j * np.pi * t)])

def ry(t):
    c, s = np.cos(np.pi * t), np.sin(np.pi * t)
    return np.array([[c, -s], [s, c]], dtype=complex)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def euler(p):
    return rx(p[0]) @ rz(p[1]) @ rx(p[2])

def apply1(psi, U, q):
    psi = np.tensordot(U, psi, ([1], [q]))
    return np.moveaxis(psi, 0, q)

def apply2(psi, U4, qa, qb):
    T = U4.reshape(2, 2, 2, 2)
    psi = np.tensordot(T, psi, ([2, 3], [qa, qb]))
    return np.moveaxis(psi, [0, 1], [qa, qb])

def ctrl(U):
    M = np.eye(4, dtype=complex)
    M[2:, 2:] = U
    return M

def crz(t):
    return np.diag([1, 1, np.exp(-1j * np.pi * t),
                    np.exp(1j * np.pi * t)]).astype(complex)

def blockdiag(UA, UB):
    M = np.zeros((4, 4), dtype=complex)
    M[:2, :2] = UA
    M[2:, 2:] = UB
    return M

def entropy(rho):
    ev = np.clip(np.real(np.linalg.eigvalsh((rho + rho.conj().T) / 2)),
                 1e-15, None)
    ev = ev / ev.sum()
    return float(-np.sum(ev * np.log2(ev)))

def rho_of(psi, keep):
    n = psi.ndim
    psi = np.moveaxis(psi, keep, n - 1)
    A = psi.reshape(-1, 2)
    return A.T @ A.conj()

G_V = 0.37  # fixed verb coupling (half-turns), never retuned

def run(theta, pA, pB, kappa, x0=0.21, with_context=True):
    """-> (S_unread, S_read_avg). Qubits [C,N,S,(X)]."""
    nq = 4 if with_context else 3
    psi = np.zeros((2,) * nq, dtype=complex)
    psi[(0,) * nq] = 1.0
    psi = apply1(psi, ry(theta), 0)
    psi = apply2(psi, blockdiag(euler(pA), euler(pB)), 0, 1)
    psi = apply1(psi, H, 2)
    psi = apply2(psi, crz(G_V), 1, 2)
    psi = apply1(psi, H, 2)
    if with_context:
        psi = apply2(psi, blockdiag(rx(x0), rx(x0 + kappa)), 0, 3)
    # post-select the noun wire in the |+> basis (H then <0|): with a
    # diagonal N-S coupling, <0| post-selection keeps only the branch that
    # never touched S (verified: everything exactly 0); the IQP H-sandwich
    # means the pipeline's effective projection is <+|, which lets both
    # noun components imprint on S
    psi = apply1(psi, H, 1)
    # post-select the noun wire on |0> (the pipeline's RemoveCups convention)
    # so that WITHOUT sense ambiguity the sentence state is pure and any
    # mixedness is attributable to the sense wire — first run showed that
    # tracing N instead measures N-S entanglement, swamping the signal
    psi = np.take(psi, 0, axis=1)
    psi = psi / np.linalg.norm(psi)
    # axes now [C, S] or [C, S, X]
    if with_context:
        S_un = entropy(rho_of(psi, 1))
        S_cond = 0.0
        for x in (0, 1):
            br = np.take(psi, x, axis=2)
            px = float(np.sum(np.abs(br) ** 2))
            if px > 1e-12:
                S_cond += px * entropy(rho_of(br / np.sqrt(px), 1))
        return S_un, S_cond
    return entropy(rho_of(psi, 1)), None

OUT = {"G_V": G_V}

# P1/P2: prior sweep (synthetic senses at fixed distance)
pA = [2 * h01(f"sA|{i}") for i in range(3)]
pB = [2 * h01(f"sB|{i}") for i in range(3)]
thetas = np.linspace(0, 0.25, 21)
sweep_theta = [run(t, pA, pB, 0.0, with_context=False)[0] for t in thetas]
OUT["P2_theta_sweep"] = {"theta": list(map(float, thetas)),
                         "S": sweep_theta}
OUT["P1"] = {"S_fixed_sense": sweep_theta[0],
             "S_equal_prior": sweep_theta[-1]}
print(f"[24b] P1: fixed-sense S={sweep_theta[0]:.4f}, "
      f"equal-prior S={sweep_theta[-1]:.4f}", flush=True)
mono = all(sweep_theta[i + 1] >= sweep_theta[i] - 1e-9
           for i in range(len(sweep_theta) - 1))
print(f"[24b] P2: monotone in theta: {mono}", flush=True)

# P3: sense-distance sweep at equal prior
deltas = np.linspace(0, 1.0, 21)
sweep_d = []
for d in deltas:
    pB_d = [pA[0] + d, pA[1] + d, pA[2] + d]
    sweep_d.append(run(0.25, pA, pB_d, 0.0, with_context=False)[0])
OUT["P3_delta_sweep"] = {"delta": list(map(float, deltas)), "S": sweep_d}
print(f"[24b] P3: S(delta=0)={sweep_d[0]:.4f} -> "
      f"S(delta=1)={sweep_d[-1]:.4f} (max {max(sweep_d):.4f})", flush=True)

# P4: purification vs context informativeness
kappas = np.linspace(0, 1.0, 21)
unread, read = [], []
for k in kappas:
    su, sr = run(0.25, pA, pB, k, with_context=True)
    unread.append(su)
    read.append(sr)
OUT["P4_kappa_sweep"] = {"kappa": list(map(float, kappas)),
                         "S_unread": unread, "S_read": read}
gaps = [u - r for u, r in zip(unread, read)]
print(f"[24b] P4: purification gap at kappa=0: {gaps[0]:.4f}; "
      f"at kappa=1: {gaps[-1]:.4f}; max gap {max(gaps):.4f} "
      f"at kappa={float(kappas[int(np.argmax(gaps))]):.2f}", flush=True)

# Real words: sense anchors from exp23 Phase-2 scheme
def anchor_params(root, lex):
    return [2 * h01(f"{root}::{lex}|{i}") for i in range(3)]

REAL = {"رجل (man/leg)": ("ر.ج.ل", "man", "leg"),
        "جمل (camel/sentence)": ("ج.م.ل", "camel", "sentence"),
        "جمل (camel/beauty)": ("ج.م.ل", "camel", "beauty")}
OUT["real_words"] = {}
for label, (root, lxA, lxB) in REAL.items():
    pa, pb = anchor_params(root, lxA), anchor_params(root, lxB)
    su0, _ = run(0.25, pa, pb, 0.0, with_context=False)
    su, sr = run(0.25, pa, pb, 0.8, with_context=True)
    OUT["real_words"][label] = {"S_ambiguous": su0,
                                "S_unread_ctx": su, "S_read_ctx": sr,
                                "purification_gap": su - sr}
    print(f"[24b] {label}: ambiguous S={su0:.4f}; with informative context "
          f"unread={su:.4f} read={sr:.4f} (gap {su - sr:+.4f})", flush=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.2))
a1.plot(thetas, sweep_theta, "o-")
a1.set_xlabel(r"sense prior $\theta$ (half-turns)")
a1.set_ylabel(r"$S(\rho_S)$ bits")
a1.set_title("P1/P2: ambiguity prior → sentence mixedness")
a2.plot(deltas, sweep_d, "s-", color="tab:orange")
a2.set_xlabel(r"sense distance $\Delta$")
a2.set_ylabel(r"$S(\rho_S)$ bits")
a2.set_title("P3: distinct senses required for mixedness")
a3.plot(kappas, unread, "o-", label="context unread")
a3.plot(kappas, read, "s-", label="context read (avg over outcomes)")
a3.set_xlabel(r"context informativeness $\kappa$")
a3.set_ylabel(r"$S(\rho_S)$ bits")
a3.set_title("P4: disambiguation as purification")
a3.legend()
fig.suptitle("Exp24b — ambiguity as mixedness, context as purifier "
             "(sense anchors from exp23)", fontsize=12)
fig.tight_layout()
fig.savefig("fig_exp24b.png", dpi=200, bbox_inches="tight")
json.dump(OUT, open("results_exp24b.json", "w"), indent=2, ensure_ascii=False)
print("[24b] DONE — results_exp24b.json + fig_exp24b.png", flush=True)
