# -*- coding: utf-8 -*-
"""readout_arity_bounds.py -- the COMMITTED producer for doc 50 Part C's
readout-arity calculus (SSC2's found-arc table and SSC3's M-ary category
ceilings).

WHY THIS FILE EXISTS
--------------------
doc 50 SSC2 and SSC3 print two session-computed tables and NEITHER has a
committed producer. doc 53 SS7.3's standing rule ("a runner that writes
checkpoints ships with the aggregator that writes the results JSON") is the
same gap-class one level up: a table quoted in a paper needs a script. This is
that script. It recomputes both tables from scratch, reconciles every cell
against the published value, and self-tests the binary column against doc 47
SS13's committed anchors before any M-ary number is believed.

Nothing here is trained and nothing here is an experiment result: the file is
named `diagnostic_*`, not `results_*`, for the same reason
`instrument_sb_bounds.py` is.

--------------------------------------------------------------------------
THE READOUT MODEL, STATED IN FULL (doc 50 SSC2 under-specifies it)
--------------------------------------------------------------------------
A readout with M outcomes produces outcome probabilities p_0 .. p_{M-1}. The
encoding is UNCHANGED, so each p_m closes in the K=1 Fourier basis of every
referent angle -- exactly the property `exp67_run.outcome_fourier_resid`
asserts at smoke (doc 53 SS5, direction 1). The decision is the HARD argmax
with the FIXED alternating grouping of doc 53 SS3.1:

    class(x) = (argmax_m p_m(x)) mod 2

*** LEMMA (SIMPLEX VACUITY) -- why the search may be unconstrained ***
SSC2 requires the p_m to be genuine probabilities: p_m >= 0 pointwise and
sum_m p_m == 1. For the argmax PATTERN that requirement is VACUOUS. Given any
f_0 .. f_{M-1} in the span, put

    p_m = 1/M + eps * (f_m - fbar),      fbar = (1/M) sum_k f_k

Then sum_m p_m == 1 identically, and for eps small enough that
eps * max_{m,x} |f_m(x) - fbar(x)| <= 1/M every p_m >= 0 on the whole domain.
Subtracting a common function and scaling by a positive constant leave argmax
unchanged, so the pattern is preserved. Hence:

    {argmax patterns from M NONNEG SUM-ONE elements of the span}
      == {argmax patterns from M ARBITRARY elements of the span}

so the search runs over arbitrary V_m and every reported witness is CONVERTED
BACK to a legal (nonneg, sum-one) configuration, which is emitted in the JSON
and numerically verified. This lemma is stated because doc 50 SSC2 sets the
problem up with the simplex constraint and a reader is entitled to know why it
was dropped.

--------------------------------------------------------------------------
PART A -- THE FOUND-ARC TABLE, AND AN EXACT MAXIMUM
--------------------------------------------------------------------------
One angle, K=1: p_m(theta) = <V_m, (1, cos theta, sin theta)>, V_m in R^3.
"Arcs" = the number of maximal intervals of constant CLASS around the circle,
i.e. the number of parity changes of argmax.

doc 50 SSC2 gives the analytic upper bound 2M-1 from the order-2
Davenport-Schinzel envelope, and reports best-found arcs as SEARCH-DERIVED
LOWER BOUNDS "never the achievable maximum". This file closes that gap:

  UPPER BOUND (parity-corrected DS).  Cut the circle at a point interior to an
  argmax piece. Each pairwise difference p_i - p_j = A + R cos(theta - Phi) has
  at most 2 zeros on a full period, so the linear sequence of argmax indices is
  an order-2 DS sequence over M symbols and has length <= lambda_2(M) = 2M-1,
  i.e. at most 2M-2 index changes. Class changes <= index changes, so
  arcs <= 2M-2. (The circle also forces arcs to be EVEN, which is what removes
  the odd value 2M-1: a class must return to itself after a full turn.)

  ACHIEVABILITY.  Take parallel gradients, p_m(theta) = a_m + m * cos theta.
  With x = cos theta the upper envelope is that of M lines in x, convex
  piecewise-linear with M-1 breakpoints; choosing the a_m puts every breakpoint
  in (-1, 1), and each x-breakpoint is hit TWICE per turn. The argmax order is
  m increasing then decreasing, so with the FIXED i-mod-2 grouping every index
  change is also a class change: arcs = 2(M-1) = 2M-2.

  Therefore the ACHIEVABLE MAXIMUM is EXACTLY 2M-2, and it is a maximum, not a
  best-found. The search below is run anyway, adversarially (doc 47 SS16's
  culture: an instrument that decides admissibility is attacked before use),
  and must never beat 2M-2.

Arc counting is EXACT, never on a grid: breakpoints are the closed-form roots
of a + b cos theta + c sin theta = 0 for every pair, and the class is evaluated
at the midpoint of every consecutive pair of breakpoints. A grid silently
misses thin arcs and would under-report the very quantity being bounded. The
grid IS used to drive the batched search (a grid count is a valid lower bound
on the exact count), and every winner is then exactly certified.

--------------------------------------------------------------------------
PART B -- THE M-ARY CATEGORY CEILINGS
--------------------------------------------------------------------------
SB (doc 46 SS5 / doc 47 SS13): one event, a two-referent cast, exactly one
category bearer. Each cast (bearer x_i, non-bearer y_j) supplies TWO twins,
(agent = x_i, patient = y_j) -> yes and the swap -> no. The readout reads the
JOINT state, so p_m(theta_a, theta_p) = <V_m, u(theta_a) (x) u(theta_p)>,
V_m in R^{(2K+1)^2}, and each p_m is 3-term in EACH angle separately -- which
is precisely what exp67's validity check asserts.

*** THE DENOMINATOR IS PER-TWIN, AND THAT IS FORCED BY ARITHMETIC ***
doc 50 SSC3 does not state its denominator. It is recoverable exactly:

    tertiary  M=8   0.9105 = 173/190   (2|B||N| = 190)   NOT k/95
    manual    M=12  0.9000 = 171/190                     NOT k/95
    uniformed M=8, K=2  0.9688 = 155/160  (2|B||N| = 160) NOT k/80

173 and 171 and 155 are ODD, and a PAIR-LEVEL count over casts is a count out
of |B||N| -- 95, 80 -- with no half-credit. So SSC3's M-ary cells are
PER-TWIN (doc 47 SS17's population), denominator 2|B||N|, a cast free to score
0, 1/2 or 1. Every one of SSC3's twelve M-ary cells is an integer over
2|B||N|; no other denominator fits. This is recorded as the reconstruction
that reconciles SSC3's arithmetic.

*** BUT SSC3's BINARY COLUMN IS PAIR-LEVEL ***
0.8526 / 0.8750 / 0.8421 / 0.9250 are doc 47 SS13's numbers, and SS13 ran the
SS3.2 PAIR-LEVEL instrument (both twins of a cast won together, forcing an
ANTISYMMETRIC form; denominator = casts). So SSC3's table carries a pair-level
column into a per-twin table. That is exactly doc 47 SS18.5's class ("a bound
must be stated together with the population it bounds"), so this file reports
BOTH:

  binary_pair_level  doc 47 SS13's EXACT pair-level ceiling  -- THE SELFTEST
  M=2 (per-twin)     argmax over 2 outcomes, general C, per-twin population

The direction is safe: per-twin >= pair-level always, so nothing in SSC3's
M-ary row is invalidated by the mismatch -- it makes SSC3's binary column a
conservative anchor rather than a wrong one.

MONOTONICITY IS STRUCTURAL, NOT HOPED FOR. doc 50 SSC3 carries an asterisk on
uniformed / M=4 (0.8375, below its own binary value) and calls it optimiser
failure. Here the M-ladder is NESTED BY CONSTRUCTION: the M=2 search is seeded
with the pair-level antisymmetric witness, and each M is seeded with the best
witness of the previous M plus DEAD outcomes (a large negative constant
component, which can never win an argmax). So
    pair-level <= M=2 <= M=4 <= M=8 <= M=12
holds by construction and no asterisk can arise. The booleans are recorded
anyway.

ORIENTATION. The grouping i mod 2 is fixed, but relabelling outcome m -> m+1
(mod M) is a relabelling of the same M effects and flips the parity for even M.
Both orientations are therefore evaluated and the winner is recorded.

K=2 MEANS FREQUENCY, NOT WIDTH. u_K(theta) = (1, cos theta, sin theta, ...,
cos K theta, sin K theta), so K=2 is 5-dim per referent and 25-dim jointly.
This is NOT `capacity_check.referent_vector`'s list form, which Kroneckers TWO
INDEPENDENT ANGLES and is exp68's WIDTH register. The repo overloads the word;
the distinction is named here because SSC3's 0.9688 is a frequency number.

NULLS. doc 47 SS16.1's binding rule attaches size-matched nulls to any
CATEGORY-VS-NULL comparison. No such comparison is made here: every claim is
within one category, across M, on one fixed inventory. Nulls are therefore not
computed, and the omission is stated rather than left to be noticed.

CLI
---
    python readout_arity_bounds.py              # writes the diagnostic JSON
    python readout_arity_bounds.py --selftest   # anchors + invariants only
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import instrument_sb_bounds as isb            # noqa: E402  (authority for MAX-FS)

OUT = os.path.join(HERE, "diagnostic_readout_arity_bounds.json")

TOL = 1e-12

# --------------------------------------------------------------- published
# doc 50 SSC2's found-arc column, hand-typed. Hand-typing is what makes this a
# reconciliation rather than a printout.
DOC50_C2 = {2: 2, 3: 4, 4: 6, 6: 8, 8: 10}

# doc 50 SSC3, hand-typed. `binary` is doc 47 SS13's PAIR-LEVEL value.
DOC50_C3 = {
    "tertiary":        {"binary": 0.8526, 4: 0.9053, 8: 0.9105, 12: 0.9053},
    "uniformed":       {"binary": 0.8750, 4: 0.8375, 8: 0.9125, 12: 0.9500},
    "manual":          {"binary": 0.8421, 4: 0.8421, 8: 0.8842, 12: 0.9000},
    "nonoccupational": {"binary": 0.9250, 4: 1.0000, 8: 1.0000, 12: 1.0000},
}
DOC50_C3_ASTERISK = ("uniformed", 4)      # SSC3's declared optimiser failure
DOC50_C3_K2 = {"category": "uniformed", "M": 8, "K": 2, "value": 0.9688}

# doc 47 SS13's committed binary anchors, hand-typed (capacity_check.ANCHORS
# style). The M=2 column ABORTS on any mismatch.
DOC47_S13_ANCHORS = {
    "tertiary":        {"value": 0.8526315789473684, "fraction": "81/95"},
    "uniformed":       {"value": 0.8750,             "fraction": "70/80"},
    "manual":          {"value": 0.8421052631578947, "fraction": "80/95"},
    "nonoccupational": {"value": 0.9250,             "fraction": "74/80"},
}

MEMBERSHIP_CLOSURE = (
    "doc 47 SS13's four category membership lists, as carried in "
    "instrument_sb_bounds.CATEGORIES, were CONFIRMED by the user 2026-08-23; "
    "doc 54 SS5.2 is the record of closure and supersedes the BLOCKING "
    "native-speaker flag that instrument_sb_bounds.py still prints."
)


# ===================================================================== basis
def u_k(theta, K=1):
    """K-frequency single-angle span (1, cos t, sin t, ..., cos Kt, sin Kt)."""
    v = [1.0]
    for k in range(1, K + 1):
        v += [math.cos(k * theta), math.sin(k * theta)]
    return np.asarray(v, dtype=float)


def joint(theta_a, theta_p, K=1):
    """u(theta_agent) (x) u(theta_patient), flattened. d = (2K+1)^2."""
    return np.outer(u_k(theta_a, K), u_k(theta_p, K)).ravel()


# ============================================================ PART A: arcs
def _pair_roots(v):
    """All theta in [0, 2pi) with a + b cos t + c sin t == 0, closed form."""
    a, b, c = float(v[0]), float(v[1]), float(v[2])
    R = math.hypot(b, c)
    if R < 1e-14:
        return []
    r = -a / R
    if r > 1.0 + 1e-15 or r < -1.0 - 1e-15:
        return []
    r = min(1.0, max(-1.0, r))
    phi = math.atan2(c, b)
    d = math.acos(r)
    return [(phi + d) % (2 * math.pi), (phi - d) % (2 * math.pi)]


def exact_arcs(V, grp):
    """EXACT number of class arcs (parity changes of argmax) on the circle.

    Breakpoints are the closed-form roots of every pairwise difference; the
    class is read at the midpoint of every consecutive pair of breakpoints.
    Returns (n_arcs, n_index_changes, boundary_thetas)."""
    V = np.asarray(V, dtype=float)
    M = V.shape[0]
    ts = []
    for i in range(M):
        for j in range(i + 1, M):
            ts += _pair_roots(V[i] - V[j])
    ts = sorted(set(round(t, 12) for t in ts))
    if not ts:
        return 0, 0, []
    mids = []
    for i in range(len(ts)):
        a = ts[i]
        b = ts[(i + 1) % len(ts)] + (2 * math.pi if i == len(ts) - 1 else 0.0)
        mids.append(((a + b) / 2.0) % (2 * math.pi))
    U = np.stack([u_k(t, 1) for t in mids])                 # (n_mid, 3)
    idx = np.argmax(U @ V.T, axis=1)
    cls = np.asarray(grp, dtype=int)[idx]
    n = len(mids)
    arcs = int(sum(1 for i in range(n) if cls[i] != cls[i - 1]))
    ich = int(sum(1 for i in range(n) if idx[i] != idx[i - 1]))
    bnds = [ts[i] for i in range(n) if cls[i] != cls[i - 1]]
    return arcs, ich, bnds


def parallel_construction(M):
    """p_m(theta) = a_m + m cos theta, breakpoints spread over x in (-1, 1).
    Achieves the EXACT maximum 2M-2 with the fixed i-mod-2 grouping."""
    a = [0.0]
    for m in range(M - 1):
        t = -1.0 + 2.0 * (m + 1) / M          # in (-1, 1), strictly increasing
        a.append(a[-1] - t)                   # breakpoint between m, m+1 at x=t
    return np.array([[a[m], float(m), 0.0] for m in range(M)])


def arc_search(M, grp, restarts, iters, grid, seed):
    """Grid-driven batched hill-climb; every winner is EXACTLY certified.

    A grid count is a lower bound on the exact count (it can only miss arcs),
    so it is a legitimate search driver; it is never a reported number."""
    rng = np.random.default_rng(seed)
    th = np.linspace(0.0, 2 * math.pi, grid, endpoint=False)
    Ugrid = np.stack([np.ones_like(th), np.cos(th), np.sin(th)], 1).astype(np.float32)
    g = np.asarray(grp, dtype=np.int64)

    def grid_arcs(V):                                       # V: (B, M, 3)
        s = np.einsum("gd,bmd->bgm", Ugrid, V.astype(np.float32))
        c = g[np.argmax(s, axis=2)]                         # (B, G)
        return (c != np.roll(c, 1, axis=1)).sum(axis=1)

    V = rng.standard_normal((restarts, M, 3))
    best = grid_arcs(V)
    step = 0.6
    for it in range(iters):
        step = 0.6 * (0.02 ** (it / max(iters - 1, 1)))
        Vc = V + step * rng.standard_normal(V.shape)
        sc = grid_arcs(Vc)
        take = sc >= best
        V[take], best[take] = Vc[take], sc[take]
    order = np.argsort(-best)[:64]
    cert, cw = -1, None
    for i in order:
        a, _ic, _b = exact_arcs(V[i], grp)
        if a > cert:
            cert, cw = a, V[i].copy()
    return int(cert), cw, int(best.max())


def to_legal_probabilities(V, K=1, joint_dim=False, margin=0.98):
    """SIMPLEX-VACUITY conversion: V -> nonneg, sum-one outcome probabilities
    with the SAME argmax pattern. Returns (P, eps) with P[m] the coefficient
    vector of p_m in the same basis."""
    V = np.asarray(V, dtype=float)
    M = V.shape[0]
    D = V - V.mean(axis=0, keepdims=True)
    # sup norm of a basis vector: ||u_K||_2 = sqrt(2K+1); jointly it squares.
    nb = math.sqrt(2 * K + 1)
    if joint_dim:
        nb = nb * nb
    worst = float(np.linalg.norm(D, axis=1).max()) * nb
    eps = margin / (M * worst) if worst > 0 else 0.0
    P = np.zeros_like(V)
    P[:, 0] = 1.0 / M
    P += eps * D
    return P, eps


# ================================================== PART B: category ceilings
def twin_design(angles, bearers, nonbearers, K=1):
    """Rows phi_t and labels y_t over ALL ordered twins. Label 1 = 'yes'."""
    X, Y = [], []
    for b in bearers:
        for y in nonbearers:
            X.append(joint(angles[b], angles[y], K)); Y.append(1)   # yes twin
            X.append(joint(angles[y], angles[b], K)); Y.append(0)   # no twin
    return np.asarray(X, dtype=float), np.asarray(Y, dtype=int)


def hard_acc(V, PHI, y, grp):
    """(best accuracy over the two orientations, orientation, count)."""
    idx = np.argmax(PHI @ np.asarray(V, dtype=float).T, axis=1)
    c = np.asarray(grp, dtype=int)[idx]
    n0 = int((c == y).sum())
    n1 = len(y) - n0
    if n0 >= n1:
        return n0 / len(y), "identity", n0
    return n1 / len(y), "shift-by-one", n1


def _batch_hard(Vb, PHI, y, grp):
    """Vectorised hard counts for a batch of witnesses. (B,) best counts."""
    B, M, d = Vb.shape
    T = PHI.shape[0]
    s = (Vb.reshape(B * M, d) @ PHI.T).reshape(B, M, T).transpose(0, 2, 1)
    c = np.asarray(grp)[np.argmax(s, axis=2)]
    n0 = (c == y[None, :]).sum(axis=1)
    return np.maximum(n0, len(y) - n0)


def anneal_argmax(PHI, y, M, grp, restarts, steps, seed, seeds_V=None,
                  temp0=2.0, temp1=400.0, lr=0.25):
    """Batched first-order search on the doc 53 SS3.2 surrogate.

    Class score for class c is logsumexp_{m in c}(TEMP * <V_m, phi>) / TEMP,
    cross-entropy against the label; TEMP is annealed so the surrogate becomes
    the hard argmax. The HARD count (both orientations) is read at EVERY
    iterate, so the returned value is always a genuine constructive lower
    bound and can never be worse than the seeds it was given."""
    rng = np.random.default_rng(seed)
    T, d = PHI.shape
    g = np.asarray(grp, dtype=int)
    m0 = np.where(g == 0)[0]
    m1 = np.where(g == 1)[0]
    yv = np.asarray(y, dtype=int)

    V = rng.standard_normal((restarts, M, d))
    if seeds_V:
        for i, s in enumerate(seeds_V[:restarts]):
            V[i] = s
    V /= np.maximum(np.linalg.norm(V.reshape(restarts, -1), axis=1)[:, None, None], 1e-300)

    best = _batch_hard(V, PHI, yv, g)
    bi = int(np.argmax(best))
    bestc, bestV = int(best[bi]), V[bi].copy()

    for it in range(steps):
        f = it / max(steps - 1, 1)
        TEMP = temp0 * (temp1 / temp0) ** f
        S = np.einsum("td,bmd->btm", PHI, V) * TEMP                 # (B,T,M)
        Smax = S.max(axis=2, keepdims=True)
        E = np.exp(S - Smax)
        z0 = E[:, :, m0].sum(axis=2)
        z1 = E[:, :, m1].sum(axis=2)
        l0 = np.log(z0 + 1e-300) + Smax[:, :, 0]
        l1 = np.log(z1 + 1e-300) + Smax[:, :, 0]
        # softmax over the two class scores, cross-entropy gradient
        mm = np.maximum(l0, l1)
        q1 = np.exp(l1 - mm) / (np.exp(l0 - mm) + np.exp(l1 - mm))   # P(class 1)
        dl1 = (q1 - yv[None, :])                                     # (B,T)
        dl0 = -dl1
        # d l_c / d S = softmax within class c
        W = np.zeros_like(E)
        W[:, :, m0] = E[:, :, m0] / (z0[:, :, None] + 1e-300) * dl0[:, :, None]
        W[:, :, m1] = E[:, :, m1] / (z1[:, :, None] + 1e-300) * dl1[:, :, None]
        Gr = np.einsum("btm,td->bmd", W, PHI)
        gn = np.linalg.norm(Gr.reshape(restarts, -1), axis=1)[:, None, None]
        V -= (lr * (0.05 ** f)) * Gr / np.maximum(gn, 1e-300)
        V /= np.maximum(np.linalg.norm(V.reshape(restarts, -1), axis=1)[:, None, None], 1e-300)

        cur = _batch_hard(V, PHI, yv, g)
        k = int(np.argmax(cur))
        if int(cur[k]) > bestc:
            bestc, bestV = int(cur[k]), V[k].copy()
    return bestc, bestV


def latent_hinge(PHI, y, M, grp, restarts, steps, seed, seeds_V=None,
                 margin=0.05, lr=0.06):
    """Latent multiclass hinge -- the surrogate that actually targets the hard
    objective, and the stage that makes the high-M numbers usable.

    For twin t the decision is right iff SOME outcome of the correct class
    outscores EVERY outcome of the wrong class, so

        loss_t = max(0, margin + max_{grp[m] != y_t} s_tm
                              - max_{grp[m] == y_t} s_tm)

    with s_tm = <V_m, phi_t>. The winning outcome inside the correct class is a
    LATENT variable, re-chosen at every step -- which is exactly the freedom an
    argmax readout has and which a fixed-grouping softmax surrogate does not
    exploit. Subgradient steps, hard count read at EVERY iterate for both
    orientations, so the return is always a genuine constructive lower bound.

    This is the argmax analogue of `instrument_sb_bounds._ls_drop`: a
    first-order search ALONE under-performs badly here, and would silently
    return a bound far below what is reachable."""
    rng = np.random.default_rng(seed)
    T, d = PHI.shape
    g = np.asarray(grp, dtype=int)
    yv = np.asarray(y, dtype=int)
    # per-twin masks: which outcomes belong to the correct / wrong class
    corr = (g[None, :] == yv[:, None])                       # (T, M)
    V = rng.standard_normal((restarts, M, d))
    if seeds_V:
        for i, s in enumerate(seeds_V[:restarts]):
            V[i] = s
    V /= np.maximum(np.linalg.norm(V.reshape(restarts, -1), axis=1)[:, None, None], 1e-300)

    best = _batch_hard(V, PHI, yv, g)
    bi = int(np.argmax(best))
    bestc, bestV = int(best[bi]), V[bi].copy()
    NEG = -1e18
    ar = np.arange(restarts)[:, None]
    at = np.arange(T)[None, :]
    for it in range(steps):
        S = (V.reshape(restarts * M, d) @ PHI.T).reshape(restarts, M, T).transpose(0, 2, 1)
        Sc = np.where(corr[None], S, NEG)
        Sw = np.where(corr[None], NEG, S)
        mp = np.argmax(Sc, axis=2)                           # (B, T)
        mn = np.argmax(Sw, axis=2)
        viol = (margin + Sw[ar, at, mn] - Sc[ar, at, mp]) > 0
        w = viol.astype(float)[:, :, None]
        # subgradient: +phi on the violating WRONG-class outcome, -phi on the
        # best CORRECT-class outcome.
        W = np.zeros((restarts, T, M))
        np.put_along_axis(W, mn[:, :, None], w, axis=2)
        np.put_along_axis(W, mp[:, :, None], -w, axis=2)
        Gr = (W.transpose(0, 2, 1).reshape(restarts * M, T) @ PHI).reshape(restarts, M, d)
        gn = np.linalg.norm(Gr.reshape(restarts, -1), axis=1)[:, None, None]
        V -= (lr * (0.05 ** (it / max(steps - 1, 1)))) * Gr / np.maximum(gn, 1e-300)
        V /= np.maximum(np.linalg.norm(V.reshape(restarts, -1), axis=1)[:, None, None], 1e-300)
        cur = _batch_hard(V, PHI, yv, g)
        k = int(np.argmax(cur))
        if int(cur[k]) > bestc:
            bestc, bestV = int(cur[k]), V[k].copy()
    return bestc, bestV


def polish(V, PHI, y, grp, rounds, seed):
    """Coordinate-noise hill-climb on the HARD count, from the best witness."""
    rng = np.random.default_rng(seed)
    g = np.asarray(grp, dtype=int)
    yv = np.asarray(y, dtype=int)
    B = 96
    cur = V.copy()
    curc = _batch_hard(cur[None], PHI, yv, g)[0]
    for r in range(rounds):
        sc = 0.5 * (0.01 ** (r / max(rounds - 1, 1)))
        C = cur[None] + sc * rng.standard_normal((B,) + cur.shape)
        cnt = _batch_hard(C, PHI, yv, g)
        k = int(np.argmax(cnt))
        if cnt[k] > curc:
            cur, curc = C[k].copy(), int(cnt[k])
    return int(curc), cur


MARGIN_LADDER = (0.02, 0.05, 0.15, 0.40)


def search_cell(PHI, y, M, grp, seeds, bud, seed):
    """The full per-cell search, and the only place a Part B number is made.

    Three stages, applied IDENTICALLY to every cell:
      1. `anneal_argmax` -- the doc 53 SS3.2 logsumexp surrogate, annealed;
      2. `latent_hinge`  -- over a ladder of margins (a single margin is a
         hyper-parameter choice and would be an unmatched budget between
         cells);
      3. `polish`        -- hard-count hill-climb;
    then `hops` basin-hopping rounds of (perturb best -> hinge -> polish).
    The hard count is tracked at every iterate of every stage, so the return
    can never be worse than the best seed handed in."""
    log = []
    best, bV = -1, None

    def take(tag, c, V):
        nonlocal best, bV
        log.append({"stage": tag, "count": int(c)})
        if c > best:
            best, bV = int(c), V.copy()

    c, V = anneal_argmax(PHI, y, M, grp, bud["anneal_restarts"],
                         bud["anneal_steps"], seed, seeds_V=seeds)
    take("anneal", c, V)
    c, V = polish(V, PHI, y, grp, bud["polish"], seed + 1)
    take("anneal+polish", c, V)
    for i, mg in enumerate(MARGIN_LADDER):
        c, V = latent_hinge(PHI, y, M, grp, bud["hinge_restarts"],
                            bud["hinge_steps"], seed + 10 + i, seeds_V=seeds,
                            margin=mg)
        take("hinge(m=%.2f)" % mg, c, V)
        c, V = polish(V, PHI, y, grp, bud["polish"], seed + 20 + i)
        take("hinge(m=%.2f)+polish" % mg, c, V)
    rng = np.random.default_rng(seed + 99)
    for h in range(bud["hops"]):
        sc = float(np.abs(bV).max())
        sd = with_rotations([bV + s * sc * rng.standard_normal(bV.shape)
                             for s in (0.05, 0.15, 0.30)]) + seeds
        c, V = latent_hinge(PHI, y, M, grp, bud["hop_restarts"],
                            bud["hop_steps"], seed + 300 + h, seeds_V=sd)
        take("hop%d" % h, c, V)
        c, V = polish(V, PHI, y, grp, bud["polish"], seed + 400 + h)
        take("hop%d+polish" % h, c, V)
    return best, bV, log


def embed_with_dead(V, M_new, dead=-50.0):
    """Nest a witness of arity M inside arity M_new: the added outcomes get a
    large NEGATIVE CONSTANT component (basis element 0 is identically 1 on the
    whole domain), so they can never win an argmax. The M-ladder is therefore
    monotone BY CONSTRUCTION."""
    M, d = V.shape
    out = np.zeros((M_new, d))
    out[:M] = V
    scale = float(np.abs(V).max()) + 1.0
    out[M:, 0] = dead * scale
    return out


def with_rotations(seeds):
    """Each seed, plus its index rotation m -> m+1 (mod M). For even M the
    rotation flips the class parity, so the surrogate -- which optimises the
    IDENTITY orientation only -- is given both orientations of every seed."""
    out = []
    for V in seeds:
        out.append(V)
        out.append(np.roll(V, 1, axis=0))
    return out


def agent_only_design(angles, bearers, nonbearers):
    """The AGENT-ONLY design matrix: one row per twin, but carrying only the
    AGENT's K=1 vector.

    p_m(theta_a, theta_p) = <w_m, u(theta_a)> is a legal joint readout (it is
    constant in the patient angle) and it is the direct realisation of doc 50
    SSC2's arc picture -- a per-referent arc classifier that labels the agent
    'bearer' or not. Restricting the joint design to its agent factor gives the
    IDENTICAL objective in R^3 instead of R^9, so the same three-stage search
    solves a 3M-parameter problem rather than a 9M-parameter one and returns a
    far better witness, which is then LIFTED and handed to the joint search as
    a structured seed. This is the seed that carries SSC2's arc budget into
    SSC3's table, and without it the joint search lands several twins short."""
    X, Y = [], []
    for b in bearers:
        for y in nonbearers:
            X.append(u_k(angles[b], 1)); Y.append(1)      # yes twin, agent = b
            X.append(u_k(angles[y], 1)); Y.append(0)      # no twin,  agent = y
    return np.asarray(X, dtype=float), np.asarray(Y, dtype=int)


def lift_single_wire(w, d):
    """w in R^{M x 3} -> V in R^{M x d}, V_m = w_m (x) e_0."""
    du = int(round(math.sqrt(d)))
    M = w.shape[0]
    V = np.zeros((M, d))
    for m in range(M):
        v = np.zeros(du)
        v[:3] = w[m]
        V[m] = np.outer(v, np.eye(du)[0]).ravel()
    return V


def arc_seeds_r3(M, n, rng):
    """Random ARC readouts in the bare K=1 span: p_m(t) = a_m + cos(t - psi_m),
    i.e. M cells whose boundaries are the psi-midpoints. The natural basin for
    an agent-only arc classifier."""
    out = []
    for _ in range(n):
        psis = np.sort(rng.uniform(0, 2 * math.pi, M))
        a = rng.normal(0.0, 0.4, M)
        out.append(np.array([[a[m], math.cos(psis[m]), math.sin(psis[m])]
                             for m in range(M)]))
    return out


def single_wire_seeds(angles, names, M, n, rng, d):
    """Structured seeds: the readout looks at the AGENT wire only,
    p_m(theta_a, theta_p) = <w_m, u(theta_a)>, i.e. V_m = w_m (x) e_0. A
    per-referent arc classifier is a legal joint readout and is a good basin."""
    du = int(round(math.sqrt(d)))
    out = []
    for _ in range(n):
        psis = np.sort(rng.uniform(0, 2 * math.pi, M))
        a = rng.normal(0.0, 0.4, M)
        V = np.zeros((M, d))
        for m in range(M):
            w = np.zeros(du)
            w[0] = a[m]
            w[1] = math.cos(psis[m])
            w[2] = math.sin(psis[m])
            V[m] = np.outer(w, np.eye(du)[0]).ravel()
        out.append(V)
    return out


# ===================================================================== main
def strict_margin_report(V, PHI, y, grp, orientation):
    """doc 47 SS16's BOUNDARY DISCIPLINE, carried over to argmax.

    `instrument_sb_bounds` refuses to bank a constraint whose residual is only
    a boundary value ("a boundary residual is NOT a win", TOL). The argmax
    analogue is an exact TIE between the best correct-class outcome and the
    best wrong-class outcome: `np.argmax` breaks ties toward the lower index,
    so a tie could be banked as a win and inflate the count. Measure-zero for
    generic data, but asserted numerically rather than argued.

    Returns the margin statistics and a strictly-won recount."""
    g = np.asarray(grp, dtype=int)
    if orientation == "shift-by-one":
        g = 1 - g
    yv = np.asarray(y, dtype=int)
    S = PHI @ np.asarray(V, dtype=float).T                    # (T, M)
    NEG = -1e18
    corr = (g[None, :] == yv[:, None])
    sc = np.where(corr, S, NEG).max(axis=1)
    sw = np.where(corr, NEG, S).max(axis=1)
    marg = sc - sw
    argmax_correct = int((g[np.argmax(S, axis=1)] == yv).sum())
    strict = int((marg > TOL).sum())
    won = marg[marg > TOL]
    return {"argmax_count": argmax_correct, "strictly_won_count": strict,
            "counts_agree": bool(argmax_correct == strict),
            "min_margin_over_won_twins": (float(won.min()) if won.size else None),
            "n_ties_within_TOL": int((np.abs(marg) <= TOL).sum()),
            "tolerance": TOL,
            "note": ("a counted-correct twin must beat every wrong-class outcome "
                     "STRICTLY; an exact tie would be an argmax tie-break banked "
                     "as a win, which is doc 47 SS16's defect class")}


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build_angles():
    coords, prov = isb.load_frozen2d()
    ang = {n: math.atan2(p2, p1) for n, (p1, p2) in coords.items()}
    return ang, coords, prov


def run_selftest(ang):
    """The BLOCKING anchor: the pair-level exact ceiling must reproduce doc 47
    SS13 for all four categories, bit-for-bit at 5e-5."""
    U = {n: u_k(a, 1) for n, a in ang.items()}
    out, ok_all = {}, True
    for key, cat in isb.CATEGORIES.items():
        B = cat["members"]
        N = [n for n in isb.HUMANS if n not in B]
        r = isb.bound(U, B, N, "pair")
        exp = DOC47_S13_ANCHORS[key]
        ok = abs(r["value"] - exp["value"]) < 5e-5 and r["fraction"] == exp["fraction"]
        ok_all &= ok
        out[key] = {"recomputed": r["value"], "fraction": r["fraction"],
                    "expected": exp["value"], "expected_fraction": exp["fraction"],
                    "exact": r["exact"], "method": r["method"], "matches": bool(ok)}
    return out, bool(ok_all)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--arc-restarts", type=int, default=512)
    ap.add_argument("--arc-iters", type=int, default=400)
    ap.add_argument("--arc-grid", type=int, default=2048)
    ap.add_argument("--restarts", type=int, default=512)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--polish", type=int, default=1200)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--budget-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=50000)
    a = ap.parse_args()

    def mkbud(s):
        return {"anneal_restarts": max(8, int(a.restarts * s / 2)),
                "anneal_steps": max(8, int(a.steps * s)),
                "hinge_restarts": max(8, int(a.restarts * s)),
                "hinge_steps": max(8, int(a.steps * s)),
                "hop_restarts": max(8, int(a.restarts * s / 2)),
                "hop_steps": max(8, int(a.steps * s * 0.8)),
                "polish": max(8, int(a.polish * s)),
                "hops": max(0, int(a.hops)),
                "margin_ladder": list(MARGIN_LADDER)}
    BUD = mkbud(a.budget_scale)
    BUD_X = mkbud(a.budget_scale * 2.0)

    t_start = time.time()
    ang, coords, fam_prov = build_angles()

    st, st_ok = run_selftest(ang)
    if a.selftest:
        print(json.dumps({"binary_anchor_selftest": st, "all_pass": st_ok}, indent=2))
        return 0 if st_ok else 1
    if not st_ok:
        print("SELFTEST FAILED -- pair-level exact ceilings do not reproduce doc 47 SS13.")
        print(json.dumps(st, indent=2))
        return 1
    print("[selftest] doc 47 SS13 binary anchors reproduced 4/4", flush=True)

    # ------------------------------------------------------------ PART A
    partA = {}
    for M in (2, 3, 4, 6, 8):
        grp = [i % 2 for i in range(M)]
        Vc = parallel_construction(M)
        c_arcs, c_ich, _b = exact_arcs(Vc, grp)
        s_arcs, sV, s_grid = arc_search(M, grp, a.arc_restarts, a.arc_iters,
                                        a.arc_grid, a.seed + M)
        bestV = Vc if c_arcs >= s_arcs else sV
        best = max(c_arcs, s_arcs)
        assert best <= 2 * M - 2, ("arc search returned %d > the proven maximum %d "
                                   "at M=%d -- the proof or the counter is wrong"
                                   % (best, 2 * M - 2, M))
        P, eps = to_legal_probabilities(bestV, K=1, joint_dim=False)
        chk = np.stack([u_k(t, 1) for t in np.linspace(0, 2 * math.pi, 20000,
                                                       endpoint=False)]) @ P.T
        pub = DOC50_C2[M]
        verdict = ("MATCH" if best == pub else
                   "OVER -- doc 50 SSC2's value is confirmed as an UNDERESTIMATE"
                   if best > pub else
                   "UNDER -- doc 50 SSC2's value is UNREPRODUCED at this budget; "
                   "mine is the committed floor")
        partA[M] = {
            "M": M, "grouping": "alternating i mod 2", "grp": grp,
            "best_found_arcs": int(best),
            "arcs_from_analytic_construction": int(c_arcs),
            "arcs_from_random_search": int(s_arcs),
            "grid_driver_best": int(s_grid),
            "index_changes_of_construction": int(c_ich),
            "ds_bound_2M_minus_1": 2 * M - 1,
            "exact_maximum_2M_minus_2": 2 * M - 2,
            "attains_exact_maximum": bool(best == 2 * M - 2),
            "search_never_beat_the_proven_maximum": True,
            "doc50_C2_published": pub,
            "reconciliation": verdict,
            "witness_V_rows_1_cos_sin": bestV.tolist(),
            "legal_probability_coefficients": {
                "note": ("SIMPLEX-VACUITY conversion of the witness: p_m = "
                         "1/M + eps (f_m - fbar), coefficients in (1, cos, sin)"),
                "eps": eps,
                "p_m_coefficients": P.tolist(),
                "sums_to_one_max_abs_error": float(np.abs(P.sum(axis=0) -
                                                          np.array([1.0, 0.0, 0.0])).max()),
                "min_probability_over_20000_thetas": float(chk.min()),
                "nonnegative": bool(chk.min() >= -1e-12),
            },
        }
        print("[A] M=%2d  best %2d | construction %2d | search %2d | DS %2d | max %2d | doc50 %2d  %s"
              % (M, best, c_arcs, s_arcs, 2 * M - 1, 2 * M - 2, pub, verdict.split(" --")[0]),
              flush=True)

    # ------------------------------------------------------------ PART B
    rng = np.random.default_rng(a.seed + 777)
    partB, budgets = {}, {}
    for key, cat in isb.CATEGORIES.items():
        B = cat["members"]
        N = [n for n in isb.HUMANS if n not in B]
        n_casts, n_twins = len(B) * len(N), 2 * len(B) * len(N)
        PHI, y = twin_design(ang, B, N, K=1)
        assert PHI.shape[0] == n_twins

        Upair = {n: u_k(v, 1) for n, v in ang.items()}
        pl = isb.bound(Upair, B, N, "pair")
        tw = isb.bound(Upair, B, N, "twin")
        Ctwin = tw["_c"].reshape(3, 3)

        row = {
            "gloss": cat["gloss"], "members": B, "size": len(B),
            "arcs": isb.arcs(coords, B),
            "n_casts": n_casts, "n_twins": n_twins,
            "binary_pair_level": {
                "value": pl["value"], "fraction": pl["fraction"],
                "population": "CASTS (both twins won together, antisymmetric form)",
                "method": pl["method"], "exact": pl["exact"],
                "doc47_s13_published": DOC47_S13_ANCHORS[key]["value"],
                "reproduces_doc47_s13": bool(
                    abs(pl["value"] - DOC47_S13_ANCHORS[key]["value"]) < 5e-5),
            },
            "M_ary_per_twin": {},
        }

        PHI_A, y_A = agent_only_design(ang, B, N)
        assert PHI_A.shape == (n_twins, 3) and list(y_A) == list(y)

        prevV, prevw = None, None
        for M in (2, 4, 8, 12):
            grp = [i % 2 for i in range(M)]
            # ---- stage 0: the AGENT-ONLY problem, solved in R^3 with the SAME
            #      three-stage search, then lifted as a structured seed.
            sw_seeds = ([] if prevw is None else [embed_with_dead(prevw, M)])
            sw_seeds = with_rotations(sw_seeds)
            sw_seeds += arc_seeds_r3(M, 24, rng)
            wcnt, wbest, wlog = search_cell(PHI_A, y_A, M, grp, sw_seeds, BUD,
                                            a.seed + 7 * M)
            prevw, wtot = wbest, n_twins

            seeds = []
            if M == 2:
                seeds.append(np.stack([Ctwin.ravel(), -Ctwin.ravel()]))
                A = ((Ctwin - Ctwin.T) / 2.0).ravel()
                seeds.append(np.stack([A, -A]))
            else:
                seeds.append(embed_with_dead(prevV, M))
            seeds.append(lift_single_wire(wbest, PHI.shape[1]))
            seeds = with_rotations(seeds)
            seeds += single_wire_seeds(ang, isb.HUMANS, M, 24, rng, PHI.shape[1])

            cnt, V, slog = search_cell(PHI, y, M, grp, seeds, BUD, a.seed + 13 * M)
            extended = False
            floor_cnt = int(round(row["binary_pair_level"]["value"] * n_twins))
            prev_cnt = (None if prevV is None else
                        int(_batch_hard(prevV[None], PHI, y,
                                        np.asarray(grp[:prevV.shape[0]]))[0]))
            need = max(floor_cnt, prev_cnt if prev_cnt is not None else 0)
            if cnt < need:
                # SANITY escape hatch (doc 50 SSC3's asterisk): extend THIS cell's
                # budget before recording a monotonicity break.
                extended = True
                c2, V2, slog2 = search_cell(PHI, y, M, grp, seeds, BUD_X,
                                            a.seed + 13 * M + 5)
                slog = slog + slog2
                if c2 > cnt:
                    cnt, V = c2, V2

            acc, orient, _n = hard_acc(V, PHI, y, grp)
            assert abs(acc - cnt / n_twins) < 1e-12
            P, eps = to_legal_probabilities(V, K=1, joint_dim=True)
            chkP = PHI @ P.T
            pub = DOC50_C3[key][("binary" if M == 2 else M)]
            if M == 2:
                verdict = ("doc 50 SSC3's binary column is doc 47 SS13's PAIR-LEVEL "
                           "value (%.4f); this cell is the PER-TWIN argmax-2 value. "
                           "Different populations, not a discrepancy -- per-twin >= "
                           "pair-level always." % pub)
            else:
                d = acc - pub
                verdict = ("MATCH (|delta| < 5e-5)" if abs(d) < 5e-5 else
                           "OVER by %+.4f -- doc 50 SSC3's cell is confirmed as an "
                           "UNDERESTIMATE (both are constructive lower bounds; a "
                           "higher one supersedes)" % d if d > 0 else
                           "UNDER by %+.4f -- doc 50 SSC3's cell is UNREPRODUCED at "
                           "this budget; mine is the committed floor, theirs stands "
                           "as the published lower bound" % d)
            cell = {
                "M": M, "grouping": "alternating i mod 2",
                "value": acc, "fraction": "%d/%d" % (cnt, n_twins),
                "count": int(cnt), "denominator": n_twins,
                "population": ("ALL ordered twins over the 24 frozen human nouns "
                               "(2|B||N|); no train/test split -- this IS the "
                               "noun-level bound of doc 47 SS18.1 over the whole "
                               "inventory"),
                "orientation": orient,
                "agent_only_arc_readout": {
                    "value": wcnt / wtot, "fraction": "%d/%d" % (int(wcnt), int(wtot)),
                    "note": ("best AGENT-ONLY readout p_m(theta_a, theta_p) = "
                             "<w_m, u(theta_a)> -- the direct realisation of doc 50 "
                             "SSC2's arc picture, and a strict SUBSET of the joint "
                             "readout class. Reported because the gap between it and "
                             "the joint value is exactly what the joint readout buys "
                             "over a per-referent arc classifier."),
                    "arc_budget_at_this_M": 2 * M - 2,
                    "category_arcs": row["arcs"],
                    "budget_suffices": bool(2 * M - 2 >= row["arcs"]),
                },
                "method": ("CONSTRUCTIVE LOWER BOUND (nested-seeded three-stage "
                           "search: annealed logsumexp surrogate, latent "
                           "multiclass hinge over a 4-margin ladder, hard-count "
                           "polish, plus basin hops)"),
                "exact": False,
                "stage_log": slog,
                "strict_margin_certification": strict_margin_report(
                    V, PHI, y, grp, orient),
                "budget_extended_for_monotonicity": extended,
                "doc50_C3_published": pub,
                "doc50_C3_asterisk": bool((key, M) == DOC50_C3_ASTERISK),
                "reconciliation": verdict,
                "at_or_above_pair_level_binary": bool(cnt >= floor_cnt),
                "monotone_vs_previous_M": (True if prev_cnt is None
                                           else bool(cnt >= int(prev_cnt))),
                "witness_V": V.tolist(),
                "legal_probability_check": {
                    "eps": eps,
                    "sums_to_one_max_abs_error": float(
                        np.abs(P.sum(axis=0) - np.eye(PHI.shape[1])[0]).max()),
                    "min_probability_over_all_twins": float(chkP.min()),
                    "nonnegative": bool(chkP.min() >= -1e-12),
                },
            }
            row["M_ary_per_twin"][str(M)] = cell
            prevV = V
            print("[B] %-16s M=%2d  %s = %.4f  (doc50 %.4f)  %s%s"
                  % (key, M, cell["fraction"], acc, pub,
                     verdict.split(" --")[0].split(" (")[0],
                     "  [budget extended]" if extended else ""), flush=True)
        partB[key] = row

    # ---------------------------------------------- the one K=2 cross-term
    kkey = DOC50_C3_K2["category"]
    Bk = isb.CATEGORIES[kkey]["members"]
    Nk = [n for n in isb.HUMANS if n not in Bk]
    PHI2, y2 = twin_design(ang, Bk, Nk, K=2)
    Mk = DOC50_C3_K2["M"]
    grp2 = [i % 2 for i in range(Mk)]
    seeds2 = single_wire_seeds(ang, isb.HUMANS, Mk, 32, rng, PHI2.shape[1])
    # lift the K=1 witness into the K=2 span (the K=1 basis is a coordinate
    # subspace of the K=2 basis), so K=2 >= K=1 is structural too.
    V1 = np.array(partB[kkey]["M_ary_per_twin"][str(Mk)]["witness_V"])
    lift = np.zeros((Mk, PHI2.shape[1]))
    sel = [i * 5 + j for i in range(3) for j in range(3)]
    for m in range(Mk):
        lift[m, sel] = V1[m]
    seeds2 = with_rotations([lift]) + seeds2
    c2, V2, slog2 = search_cell(PHI2, y2, Mk, grp2, seeds2, BUD_X, a.seed + 2222)
    n2 = len(y2)
    acc2, orient2, _ = hard_acc(V2, PHI2, y2, grp2)
    d2 = acc2 - DOC50_C3_K2["value"]
    k2 = {
        "category": kkey, "gloss": isb.CATEGORIES[kkey]["gloss"],
        "M": Mk, "K": 2, "d_per_referent": 5, "d_joint": PHI2.shape[1],
        "basis": "u_2(t) = (1, cos t, sin t, cos 2t, sin 2t); joint = u (x) u",
        "not_the_width_register": ("K=2 here is FREQUENCY (re-uploading). It is NOT "
                                   "capacity_check.referent_vector's list form, which "
                                   "Kroneckers two INDEPENDENT angles and is exp68's "
                                   "WIDTH register. Same word, different lever."),
        "value": acc2, "fraction": "%d/%d" % (c2, n2), "count": int(c2),
        "denominator": n2, "orientation": orient2,
        "k1_value_same_cell": partB[kkey]["M_ary_per_twin"][str(Mk)]["value"],
        "doc50_C3_published": DOC50_C3_K2["value"],
        "reconciliation": ("MATCH (|delta| < 5e-5)" if abs(d2) < 5e-5 else
                           "OVER by %+.4f -- doc 50 SSC3's K=2 cell confirmed as an "
                           "UNDERESTIMATE" % d2 if d2 > 0 else
                           "UNDER by %+.4f -- unreproduced at this budget; mine is "
                           "the committed floor" % d2),
        "method": ("CONSTRUCTIVE LOWER BOUND (same three-stage search at the "
                   "EXTENDED budget, seeded with the K=1 witness lifted into the "
                   "K=2 span -- so K=2 >= K=1 is structural too)"),
        "exact": False,
        "stage_log": slog2,
        "strict_margin_certification": strict_margin_report(
            V2, PHI2, y2, grp2, orient2),
        "k2_at_least_k1": bool(acc2 >= partB[kkey]["M_ary_per_twin"][str(Mk)]["value"]),
    }
    print("[B] K=2 cross-term  %-16s M=%d  %s = %.4f  (doc50 %.4f)  %s"
          % (kkey, Mk, k2["fraction"], acc2, DOC50_C3_K2["value"],
             k2["reconciliation"].split(" --")[0]), flush=True)

    # ------------------------------------------------------------- payload
    # --------------------------------- supersession of committed per-twin bounds
    COMMITTED_PER_TWIN = {          # hand-typed from the committed diagnostic
        "tertiary": {"fraction": "163/190", "value": 163.0 / 190.0},
        "uniformed": {"fraction": "140/160", "value": 140.0 / 160.0},
        "manual": {"fraction": "160/190", "value": 160.0 / 190.0},
        "nonoccupational": {"fraction": "148/160", "value": 148.0 / 160.0},
    }
    supersede = {
        "why_this_block_exists": (
            "the M=2 cell of Part B is the SAME optimisation problem as the "
            "committed per-twin general-C bound: argmax over two outcomes is "
            "sign(<C, phi>) with C = V_0 - V_1 unrestricted. So a better M=2 cell "
            "is a better value for a quantity two committed files already carry. "
            "doc 47 SS18.2: constructive bounds move with search effort and the "
            "stronger one supersedes -- but it must be SAID, not left to be found."),
        "sources": [
            "diagnostic_sb_bounds.json :: membership_reconstruction.<cat>."
            "recomputed_per_twin (and target_selection.table.<cat>.FA.per_twin)",
            "capacity_check.py :: ANCHORS['exp68_FA_per_twin'] = 0.9250 (148/160), "
            "the nonoccupational FA per-twin anchor",
        ],
        "note_on_the_anchors": (
            "no committed selftest breaks. capacity_check's anchors pin what THAT "
            "budgeted search deterministically returns, and that is unchanged. What "
            "changes is the BOUND: the committed files now carry weaker lower bounds "
            "for the same quantity."),
        "note_on_the_families": (
            "instrument_sb_bounds verifies the raw-PCA angle IS the frozen angle "
            "(plane agreement ~2e-15), so the frozen2d FA and raw FA per-twin bounds "
            "are the same quantity and both are superseded together."),
        "per_category": {
            k: {"committed": COMMITTED_PER_TWIN[k]["fraction"],
                "committed_value": COMMITTED_PER_TWIN[k]["value"],
                "this_file_M2": partB[k]["M_ary_per_twin"]["2"]["fraction"],
                "this_file_value": partB[k]["M_ary_per_twin"]["2"]["value"],
                "supersedes": bool(partB[k]["M_ary_per_twin"]["2"]["value"]
                                   > COMMITTED_PER_TWIN[k]["value"] + 1e-12),
                "never_below": bool(partB[k]["M_ary_per_twin"]["2"]["value"]
                                    >= COMMITTED_PER_TWIN[k]["value"] - 1e-12)}
            for k in partB},
    }
    supersede["any_superseded"] = bool(
        any(v["supersedes"] for v in supersede["per_category"].values()))
    supersede["none_below_committed"] = bool(
        all(v["never_below"] for v in supersede["per_category"].values()))

    design_rule = {
        "statement": ("a category whose membership carves A arcs on the angle "
                      "circle needs argmax over M >= A/2 + 1 outcomes before a "
                      "per-referent arc classifier can express it at all, since "
                      "the achievable maximum arc budget is EXACTLY 2M-2"),
        "relation_to_doc50_S4": ("doc 50 SS4 states the K-side rule K >= "
                                 "ceil(arcs/2). This is its M-side twin, and the "
                                 "two levers are independent (doc 50 SSC6)."),
        "per_category": {k: {"arcs": partB[k]["arcs"],
                             "min_M_for_arc_budget": partB[k]["arcs"] // 2 + 1}
                         for k in partB},
        "caveat": ("arc budget is NECESSARY, not sufficient: the SB decision is "
                   "joint in (agent, patient), and arc POSITIONS are further "
                   "constrained by the span. Reaching the budget does not by "
                   "itself deliver 1.0000, and the measured table is what decides."),
    }

    payload = {
        "instrument": "readout_arity_bounds.py",
        "script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "generated_for": ("paper draft-blocker B2: doc 50 Part C (SSC2 found-arc "
                          "table, SSC3 M-ary category ceilings) had NO committed "
                          "producer. This file is that producer."),
        "not_a_results_file": ("named diagnostic_*, not results_*, deliberately: it "
                               "is an instrument output, not an experiment result. "
                               "No pre-registered criterion is read against it."),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_sec": time.time() - t_start,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform()},
        "provenance": {
            "angle_map": ("FA on the frozen 2D AraVec plane: phi = atan2(PC2, PC1), "
                          "exp42_noun_embeddings.json -- doc 47 SS13's setting"),
            "frozen_embeddings_sha256": _sha(isb.FROZEN_EMB),
            "instrument_sb_bounds_sha256": _sha(os.path.join(HERE, "instrument_sb_bounds.py")),
            "inventory": {"n_humans": len(isb.HUMANS), "humans": isb.HUMANS},
            "membership_closure": MEMBERSHIP_CLOSURE,
            "family": fam_prov,
        },
        "formalisation": {
            "why_stated": ("doc 50 SSC2/SSC3 under-specify the readout model; these "
                           "are THIS FILE's choices, stated so the numbers are "
                           "falsifiable rather than merely reproduced."),
            "decision_rule": "class(x) = (argmax_m p_m(x)) mod 2  (doc 53 SS3.1, FIXED)",
            "orientation": ("relabelling outcome m -> m+1 (mod M) is a relabelling of "
                            "the same M effects and flips parity for even M, so both "
                            "orientations are legal and the winner is recorded"),
            "simplex_vacuity_lemma": (
                "p_m = 1/M + eps (f_m - fbar) has sum 1 identically, is nonnegative "
                "for eps <= 1/(M max|f_m - fbar|), and preserves argmax. So the set "
                "of argmax patterns reachable from M NONNEG SUM-ONE elements of the "
                "span equals the set reachable from M ARBITRARY elements. The search "
                "is therefore unconstrained, and every witness is converted back to a "
                "legal probability configuration, emitted and numerically verified."),
            "part_A_span": "p_m(theta) = <V_m, (1, cos theta, sin theta)>, V_m in R^3",
            "part_B_span": ("p_m(theta_a, theta_p) = <V_m, u_K(theta_a) (x) "
                            "u_K(theta_p)>, V_m in R^{(2K+1)^2}. Each p_m is 3-term "
                            "in EACH angle at K=1 -- the property exp67's smoke test "
                            "asserts (doc 53 SS5 direction 1)."),
            "part_B_denominator": {
                "choice": "PER-TWIN, 2|B||N| (doc 47 SS17's population)",
                "why_forced": ("doc 50 SSC3 does not state a denominator, but its own "
                               "arithmetic does: 0.9105 = 173/190, 0.9000 = 171/190, "
                               "0.9688 = 155/160. 173, 171, 155 are ODD, and a "
                               "pair-level count over casts is an integer out of "
                               "|B||N| (95, 80) with no half-credit. Every one of "
                               "SSC3's twelve M-ary cells is an integer over 2|B||N| "
                               "and none is expressible over casts."),
                "binary_column_is_different": (
                    "doc 50 SSC3's binary column is doc 47 SS13's PAIR-LEVEL exact "
                    "ceiling (both twins of a cast won together, antisymmetric form, "
                    "denominator = casts). SSC3 therefore carries a pair-level column "
                    "into a per-twin table -- doc 47 SS18.5's class exactly. The "
                    "direction is safe (per-twin >= pair-level always), so SSC3's "
                    "binary column is a CONSERVATIVE anchor, not a wrong one. This "
                    "file reports both columns separately."),
            },
            "nulls_not_computed": ("doc 47 SS16.1 binds CATEGORY-VS-NULL comparisons. "
                                   "No such comparison is made here -- every claim is "
                                   "within one category, across M, on one fixed "
                                   "inventory -- so size-matched nulls are not "
                                   "computed. Stated, not left to be noticed."),
        },
        "binary_anchor_selftest": {
            "rule": ("BLOCKING. The pair-level exact ceiling must reproduce doc 47 "
                     "SS13's four published values before any M-ary number is "
                     "written. Expected values are hand-typed above, "
                     "capacity_check.ANCHORS style."),
            "per_category": st, "all_pass": st_ok,
        },
        "part_A_found_arcs": {
            "question": ("how many decision arcs can the hard argmax over M outcomes "
                         "carve on the circle at K=1, under alternating grouping"),
            "columns": {
                "best_found_arcs": "SEARCH-DERIVED LOWER BOUND (and the construction)",
                "ds_bound_2M_minus_1": ("doc 50 SSC2's analytic column: order-2 "
                                        "Davenport-Schinzel envelope"),
                "exact_maximum_2M_minus_2": ("NEW, proved here: DS gives <= 2M-2 index "
                                             "changes after cutting the circle at an "
                                             "interior point, and a closed curve forces "
                                             "an EVEN number of class changes, which "
                                             "removes the odd value 2M-1; the parallel- "
                                             "gradient construction attains 2M-2. So "
                                             "2M-2 is a MAXIMUM, not a best-found -- "
                                             "this closes doc 50 SSC2's 'never the "
                                             "achievable maximum' caveat."),
            },
            "exact_counting": ("breakpoints are the closed-form roots of a + b cos t + "
                               "c sin t = 0 for every pair; class is read at the "
                               "midpoint of consecutive breakpoints. A grid can only "
                               "MISS arcs, so the grid is used to drive the search and "
                               "never to report a number."),
            "search_budget": {"restarts": a.arc_restarts, "hill_climb_iters": a.arc_iters,
                              "grid_points_for_driver": a.arc_grid,
                              "seed_base": a.seed,
                              "seeds_per_M": "seed_base + M",
                              "exact_certification_of_top_k": 64},
            "table": partA,
            "downstream_note": ("doc 53 SS1 quotes SSC2's 2 -> 6 -> 10 for M = 2, 4, 8 "
                                "as MOTIVATION. M=8's 10 is superseded here by 14. No "
                                "exp67 criterion is read against it (doc 53 SS4 is "
                                "measurement-vs-measurement inside exp67), so no exp67 "
                                "verdict moves -- the same bounded correction doc 53 "
                                "SS7.2 recorded for its baseline quotation."),
        },
        "part_B_category_ceilings": {
            "question": ("what accuracy is reachable on the SB rung for doc 47 SS13's "
                         "four real AraVec categories when only the READOUT changes"),
            "encoding": "UNCHANGED -- frozen single-angle map, K=1 unless stated",
            "search_budget": {
                "standard": BUD, "extended_monotonicity_escape": BUD_X,
                "stages": ("annealed logsumexp surrogate (doc 53 SS3.2) -> latent "
                           "multiclass hinge over a 4-margin ladder -> hard-count "
                           "polish -> %d basin hops" % BUD["hops"]),
                "seed_base": a.seed,
                "seeds_per_cell": "seed_base + 13*M, derived deterministically per stage",
                "identical_across_categories_and_M": True,
                "structured_seeds_per_cell": ("nested witness of the previous M with "
                                              "dead outcomes, both index orientations, "
                                              "plus 24 random single-wire arc readouts"),
            },
            "monotonicity": {
                "how": ("STRUCTURAL. M=2 is seeded with the pair-level antisymmetric "
                        "witness; each M is seeded with the best witness of the "
                        "previous M plus DEAD outcomes (large negative constant "
                        "component, which can never win an argmax). So pair-level <= "
                        "M=2 <= M=4 <= M=8 <= M=12 holds by construction."),
                "why_it_matters": ("doc 50 SSC3 carries an asterisk on uniformed / M=4 "
                                   "(0.8375 < its own binary 0.8750) and calls it "
                                   "optimiser failure. Nested seeding makes that "
                                   "failure mode unreachable."),
                "all_cells_monotone": bool(all(
                    c["monotone_vs_previous_M"] and c["at_or_above_pair_level_binary"]
                    for r in partB.values() for c in r["M_ary_per_twin"].values())),
            },
            "table": partB,
            "k2_cross_term": k2,
        },
        "supersedes_committed_per_twin_bounds": supersede,
        "design_rule": design_rule,
        "scope": ("One inventory (24 frozen AraVec human nouns), one task family (SB), "
                  "simulation, training-free. These are REPRESENTATIONAL bounds: they "
                  "say what a readout CAN express, never what an optimiser will find. "
                  "exp67 measured the second question and answered it NEGATIVELY for "
                  "the circuit (arity_lifts FALSE) while the classical comparator on "
                  "the identical features DID convert the budget (doc 53 SS7.1.3). "
                  "Nothing here revisits that."),
    }

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print("wrote %s (%.1fs)" % (a.out, time.time() - t_start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
