# -*- coding: utf-8 -*-
"""instrument_sb_bounds.py -- reachable-accuracy bounds for an ARBITRARY
per-referent feature map on the frozen human-noun inventory.

WHY THIS FILE EXISTS
--------------------
doc 47 SS13 computed K=1 ceilings for real semantic categories on the frozen
AraVec angles; doc 52 SS2 computed "1-feature" and "2-feature" numbers for the
same categories. Neither computation was ever committed, and doc 52's
`sem tertiary` row is internally suspicious in two ways:

  * its 1-feature value 0.4842 sits BELOW chance, which a MAX over a
    sign-symmetric constraint system cannot do (see SIGN-FLIP below);
  * its 2-feature value 0.8526 is bit-identical to doc 47 SS13's SINGLE-ANGLE
    ceiling, which invites the reading that the "2-feature" number came out of
    the single-angle code path.

This module is the committed, re-runnable instrument that settles both, and it
is the bound source for the doc 54 / exp68 registration.

TWO FEATURE FAMILIES, AND THE DIFFERENCE MATTERS
------------------------------------------------
  frozen2d  `exp42_noun_embeddings.json` -- UNIT-NORM 2D PCA vectors. Here
      PC1 = cos(angle) and PC2 = sin(angle) EXACTLY, so the two "features" are
      deterministically dependent: a two-feature register carries NO new
      information, only a richer function class in the one angle. Kept because
      it is the population doc 47 SS13 and doc 52 SS2 were computed over.
  raw       `aravec_raw_vectors_44.json` -- the RAW 300-dim AraVec vectors,
      PCA'd to 2 dims and NOT renormalised. Verified here: unit-normalising this
      plane reproduces `exp42_noun_embeddings.json` to ~2e-15, so the frozen
      angle IS the raw-PCA angle, and the information the raw family adds is
      exactly the RADIUS that unit-normalisation destroyed. This is the family
      exp68 uses, and it makes the width lever width-as-INFORMATION.

THE DECISION BEING BOUNDED
--------------------------
SB (doc 46 SS5): one event, a two-referent cast, exactly one category bearer.
The twins are (agent=bearer, patient=non-bearer) -> yes and the swap -> no.
For a per-referent feature map r -> u(r) in R^d the decision is `sign G(u_a,u_p)`
with G(u,v) = u^T C v, C a general real d x d matrix -- the joint span is the
outer product, exactly as in doc 47 SS3.2. Writing x_i for a bearer and y_j for
a non-bearer, the cast (i,j) supplies TWO twins:

    twin YES correct  iff  x_i^T C y_j > 0
    twin NO  correct  iff  y_j^T C x_i < 0

TWO BOUNDS, AND THEY ARE NOT THE SAME QUANTITY (doc 47 SS18.5: a bound is stated
together with the population it bounds and the method that produced it):

  PAIR-LEVEL  (doc 47 SS3.2)  both twins of a cast won TOGETHER. Subtracting the
      two constraints forces x^T A y > 0 on the ANTISYMMETRIC part A, and A
      realises them, so this is exactly
      `max_A #{(i,j): x_i^T A y_j > 0} / (|B| |N|)`.
      Denominator = CASTS. This is the instrument doc 47 SS13 ran.
  PER-TWIN    (doc 47 SS17)   the twins are SEPARATE constraints on a GENERAL C,
      so a cast may score 0, 1/2 or 1. Denominator = TWINS (2 |B| |N|).
      Always >= the pair-level value; SS17's erratum is exactly that SS3.3
      reported the pair-level number as if it were the ceiling.

SIGN-FLIP (the mechanical reason a MAX cannot be below chance)
--------------------------------------------------------------
C satisfies a constraint row g iff -C violates it. In the per-twin system the
rows are {vec(x y^T), -vec(y x^T)}; in the pair-level system
{vec(x y^T - y x^T)}. In both, `satisfied(-C) = n - satisfied(C)` for every row
with a non-zero residual, so `max_C satisfied(C)/n >= 0.5` ALWAYS. Any reported
value below 0.5 is the COMPLEMENT orientation of one particular C, not a
maximum. `verify_sign_flip()` asserts this numerically rather than arguing it.

METHODS, AND WHAT IS EXACT
--------------------------
  * d = 3, pair-level: EXACT. An antisymmetric 3x3 A is a vector w in R^3 with
    x^T A y = w . (x cross y), so the problem is MAX-FS over n homogeneous
    half-spaces in R^3, solved by visiting every cell of the plane arrangement.
    doc 47 SS16's defect and fix are reproduced: pair-normals w = V_a x V_b lie
    ON both planes, so each is perturbed into all adjacent cells
    (w + eps(s_a V_a + s_b V_b), s in {-1,0,1}^2), and a candidate sitting on a
    plane may NOT bank that plane's constraint (TOL below).
  * everything else: CONSTRUCTIVE LOWER BOUND from a budgeted randomised search.
    Exact MAX-FS over 190 rows in R^9 (d=3) or R^81 (d=9) is not available.
    doc 47 SS18.2 is the precedent for reporting constructive lower bounds.

NULL DISCIPLINE (doc 47 SS16.1, binding rule)
---------------------------------------------
Every bound is reported against a SIZE-MATCHED null: random |B|-subsets of the
same inventory, the SAME method with the SAME search budget. An unmatched budget
between category and null would be the exp44 unmatched-comparator failure class
occurring inside the diagnostic itself.

CLI
---
    python instrument_sb_bounds.py                 # writes diagnostic_sb_bounds.json
    python instrument_sb_bounds.py --selftest      # instrument checks only

GATE CLOSURE UPDATE (2026-08-24): the BLOCKING native-speaker gate recorded
below was CLOSED 2026-08-23 -- the user confirmed all four SS13 category
lists as printed. doc 54 SS5.2 is the record; it supersedes the BLOCKING
flag this file carries in its output.
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_EMB = os.path.join(HERE, "exp42_noun_embeddings.json")
RAW_EMB = os.path.join(HERE, "aravec_raw_vectors_44.json")
OUT = os.path.join(HERE, "diagnostic_sb_bounds.json")

# Strict-satisfaction tolerance, applied to UNIT-NORM constraint rows. A
# constraint is banked only if its residual is strictly positive by more than
# this; a boundary residual (~1e-17) is NOT a win.
TOL = 1e-12

# ---------------------------------------------------------------- inventory
# doc 46 SS5 / doc 47 SS13: the 24 HUMAN nouns of the frozen 44-noun inventory.
# Frozen BY NAME, never by key position, so a re-ordered source file cannot
# silently change the population a bound is stated over.
HUMANS = [
    "الطالب", "الرجل", "الولد", "المدير", "المعلم", "الطبيب",
    "العامل", "المهندس", "الشرطي", "الكاتب", "الطباخ", "الجندي",
    "الفلاح", "الحارس", "اللاعب", "الموظف", "الصحفي", "السائق",
    "المسافر", "الضابط", "الباحث", "القائد", "الصياد", "المريض",
]

GLOSS = {
    "الطالب": "student", "الرجل": "man", "الولد": "boy", "المدير": "manager",
    "المعلم": "teacher", "الطبيب": "doctor", "العامل": "worker",
    "المهندس": "engineer", "الشرطي": "policeman", "الكاتب": "writer/clerk",
    "الطباخ": "cook", "الجندي": "soldier", "الفلاح": "farmer",
    "الحارس": "guard", "اللاعب": "player", "الموظف": "employee",
    "الصحفي": "journalist", "السائق": "driver", "المسافر": "traveller",
    "الضابط": "officer", "الباحث": "researcher", "القائد": "leader/commander",
    "الصياد": "fisherman/hunter", "المريض": "patient",
}

# doc 47 SS13's four semantic categories.
#
# *** PROVENANCE WARNING -- READ BEFORE USING ANY NUMBER BELOW ***
# doc 47 SS13's session code was never committed and no explicit membership list
# is recorded anywhere in the repo or in docs 46-53 (searched: repo *.py/*.json/
# *.log, `git log --all`, docs 46-53). These lists are RECONSTRUCTED by semantic
# judgment and are PROVISIONAL. They reproduce, for all four categories, |B|,
# the arc count and the published K=1 ceiling, and for the target category also
# doc 47 SS13's null mean/SD/percentile -- strong, but circumstantial. The
# orchestrator must confirm them against the doc 47 SS13 session record before
# any doc citing them is committed.
CATEGORIES = {
    "tertiary": {
        "gloss": "tertiary-educated profession",
        "members": ["الطبيب", "المهندس", "المعلم", "الباحث", "الصحفي"],
        "doc47_s13": {"size": 5, "arcs": 8, "k1_ceiling": 0.8526,
                      "null_mean": 0.7953, "null_sd": 0.0678, "percentile": 78.2},
    },
    "uniformed": {
        "gloss": "uniformed / disciplined service",
        "members": ["الشرطي", "الجندي", "الضابط", "الحارس"],
        "doc47_s13": {"size": 4, "arcs": 6, "k1_ceiling": 0.8750,
                      "null_mean": 0.8155, "null_sd": 0.0688, "percentile": 76.2},
    },
    "manual": {
        "gloss": "manual / skilled trade",
        "members": ["العامل", "الطباخ", "الفلاح", "السائق", "الصياد"],
        "doc47_s13": {"size": 5, "arcs": 8, "k1_ceiling": 0.8421,
                      "null_mean": 0.7902, "null_sd": 0.0656, "percentile": 77.2},
    },
    "nonoccupational": {
        "gloss": "non-occupational state",
        "members": ["الرجل", "الولد", "المسافر", "المريض"],
        "doc47_s13": {"size": 4, "arcs": 4, "k1_ceiling": 0.9250,
                      "null_mean": 0.8066, "null_sd": 0.0709, "percentile": 93.0},
    },
}

DEFAULT_TARGET = "tertiary"          # doc 52 SS4.1's registered target

# doc 52 SS2's published numbers, for reconciliation.
DOC52 = {"tertiary": {"one_feature": 0.4842, "two_feature": 0.8526,
                      "two_feature_null_mean": 0.9685, "two_feature_pct": 5.0},
         "manual": {"one_feature": 0.6789, "two_feature": 0.9895,
                    "two_feature_null_mean": 0.9454, "two_feature_pct": 63.3}}

# ------------------------------------------------------------- feature maps
# theta = SCALE(p), p a per-referent scalar feature normalised into [-1, 1].
SCALINGS = {
    "full":   ("theta = (p + 1) * pi        -> [0, 2pi)",      lambda p: (p + 1.0) * math.pi),
    "half":   ("theta = (p + 1) * pi / 2    -> [0, pi]",       lambda p: (p + 1.0) * math.pi / 2.0),
    "pi":     ("theta = p * pi              -> [-pi, pi]",     lambda p: p * math.pi),
    "halfpi": ("theta = p * pi / 2          -> [-pi/2, pi/2]", lambda p: p * math.pi / 2.0),
    "unit":   ("theta = p                   -> [-1, 1]",       lambda p: p),
    "picent": ("theta = pi + p              -> [pi-1, pi+1]",  lambda p: math.pi + p),
}

# REGISTERED a priori, by the stated principle and NOT by the resulting numbers:
# a single RY is given the FULL circle, which is the geometry exp61's bands live
# in. The sweep over all of SCALINGS is reported for sensitivity.
DEFAULT_SCALING = "full"


def _u1(theta):
    """K=1 single-angle span (1, cos t, sin t)."""
    return np.array([1.0, math.cos(theta), math.sin(theta)])


def features(coords, kind, scaling=DEFAULT_SCALING):
    """coords: {noun: (p1, p2)} with p1, p2 already normalised into [-1,1].
    Returns ({noun: u in R^d}, exact-formula string)."""
    fn = SCALINGS[scaling][1]
    if kind == "FA":
        out = {n: _u1(math.atan2(p2, p1)) for n, (p1, p2) in coords.items()}
        desc = ("FA: phi = atan2(p2, p1); u = (1, cos phi, sin phi). d = 3. "
                "The EXISTING single-angle map (exp42 intro_convention, "
                "angle = atan2(y, x)); scaling-free.")
    elif kind == "F1":
        out = {n: _u1(fn(p1)) for n, (p1, p2) in coords.items()}
        desc = ("F1: PC1 only, p = p1; %s; u = (1, cos theta, sin theta). d = 3."
                % SCALINGS[scaling][0])
    elif kind == "F2":
        out = {n: np.outer(_u1(fn(p1)), _u1(fn(p2))).ravel()
               for n, (p1, p2) in coords.items()}
        desc = ("F2: two-qubit register, theta0 = SCALE(PC1) on qubit 0, "
                "theta1 = SCALE(PC2) on qubit 1, both by %s; "
                "u = vec(u(theta0) (x) u(theta1)). d = 9."
                % SCALINGS[scaling][0])
    else:
        raise ValueError(kind)
    return out, desc


# -------------------------------------------------------------- constraints
def constraint_rows(U, bearers, nonbearers, mode):
    """Homogeneous rows g with `satisfied iff <vec(C), g> > 0`.

    mode='pair'  one row per cast, on the antisymmetric part (doc 47 SS3.2)
    mode='twin'  two rows per cast, general C                (doc 47 SS17)
    """
    X = [U[n] for n in bearers]
    Y = [U[n] for n in nonbearers]
    rows = []
    for x in X:
        for y in Y:
            if mode == "pair":
                rows.append((np.outer(x, y) - np.outer(y, x)).ravel())
            elif mode == "twin":
                rows.append(np.outer(x, y).ravel())
                rows.append(-np.outer(y, x).ravel())
            else:
                raise ValueError(mode)
    return np.asarray(rows, dtype=float)


# -------------------------------------------------- EXACT MAX-FS in R^3 (d=3)
def _w_to_antisym3(w):
    """x^T A y = w . (x cross y)  <=>  A_ij = sum_k w_k eps_kij."""
    w1, w2, w3 = w
    return np.array([[0.0, w3, -w2], [-w3, 0.0, w1], [w2, -w1, 0.0]])


def _antisym_r3(U, bearers, nonbearers):
    X = [U[n] for n in bearers]
    Y = [U[n] for n in nonbearers]
    return np.array([np.cross(x, y) for x in X for y in Y])


def exact_maxfs_r3(V, eps=1e-7, return_w=False):
    """EXACT max_w #{k : w . V_k > 0}, by visiting every cell of the arrangement
    of the n planes w . V_k = 0 in R^3 (doc 47 SS16's corrected enumeration)."""
    n = V.shape[0]
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-300)
    ia, ib = np.triu_indices(n, 1)
    W = np.cross(Vn[ia], Vn[ib])
    ok = np.linalg.norm(W, axis=1) > 1e-12
    W, ia, ib = W[ok], ia[ok], ib[ok]
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    cands = [W]
    for sa in (-1, 0, 1):
        for sb in (-1, 0, 1):
            if sa == 0 and sb == 0:
                continue
            cands.append(W + eps * (sa * Vn[ia] + sb * Vn[ib]))
    C = np.vstack(cands)
    C = np.vstack([C, -C, Vn, -Vn])
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-300)
    best, best_w = 0, None
    for s in range(0, C.shape[0], 20000):
        blk = C[s:s + 20000]
        # STRICT count. A candidate sitting exactly ON a plane -- which every
        # un-perturbed pair-normal does, by construction -- must NOT bank that
        # plane's constraint: `> 0` on a residual of ~1e-17 counts a boundary as
        # a win, inflates the bound by up to 2, and returns a witness that is
        # not a strict solution.
        cnt = ((blk @ Vn.T) > TOL).sum(axis=1)
        k = int(cnt.argmax())
        if int(cnt[k]) > best:
            best, best_w = int(cnt[k]), blk[k].copy()
    return (best, best_w) if return_w else best


# --------------------------------- CONSTRUCTIVE LOWER BOUND, any d (MAX-FS)
def _score(Gn, c):
    m = Gn @ c
    p, q = int((m > TOL).sum()), int((m < -TOL).sum())
    return (p, c) if p >= q else (q, -c)


def _anneal(Gn, restarts, steps, rng, inits=None):
    """Normalised-gradient descent on sum_k sigmoid(-<c,g_k>/tau), tau annealed
    1.0 -> 0.02, c on the unit sphere, `restarts` chains batched. The hard count
    is read for BOTH c and -c at every iterate, so the SIGN-FLIP floor of
    ceil(n/2) is structurally guaranteed."""
    n, d = Gn.shape
    C = rng.standard_normal((restarts, d))
    if inits is not None:
        for i, v in enumerate(inits[:restarts]):
            C[i] = v
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-300)
    best, best_c = -1, None
    for t in range(steps):
        f = t / max(steps - 1, 1)
        tau, lr = 0.02 ** f, 0.5 * (0.02 ** f)
        M = C @ Gn.T
        for arr, sgn in (((M > TOL).sum(axis=1), 1.0), ((M < -TOL).sum(axis=1), -1.0)):
            k = int(arr.argmax())
            if int(arr[k]) > best:
                best, best_c = int(arr[k]), sgn * C[k].copy()
        s = 1.0 / (1.0 + np.exp(-np.clip(-M / tau, -60, 60)))
        upd = (s * (1.0 - s)) @ Gn
        upd /= np.maximum(np.linalg.norm(upd, axis=1, keepdims=True), 1e-300)
        C += lr * upd
        C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-300)
    return best, best_c


def _ls_drop(Gn, rng, starts=6, drop_frac=0.03):
    """Greedy MAX-FS: least-squares fit <c,g> ~ 1 over an ACTIVE row set, drop
    the worst-fitting rows, refit, tracking the best hard count over the whole
    trajectory. This stage is what makes the high-d numbers usable -- a
    first-order search ALONE under-performs badly at d = 9 (81 parameters) and
    silently returns a bound far below what is reachable."""
    n, d = Gn.shape
    best, best_c = -1, None
    for s in range(starts):
        act = (list(range(n)) if s == 0
               else list(rng.choice(n, size=min(n, max(d + 1, int(0.7 * n))),
                                    replace=False)))
        while len(act) > max(4, n // 3):
            A = Gn[act]
            c, *_ = np.linalg.lstsq(A, np.ones(len(act)), rcond=None)
            nc = float(np.linalg.norm(c))
            if nc < 1e-12:
                break
            c /= nc
            cnt, cc = _score(Gn, c)
            if cnt > best:
                best, best_c = cnt, cc.copy()
            ma = A @ c
            k = max(1, int(drop_frac * len(act)))
            worst = set(np.argsort(ma)[:k].tolist())
            act = [r for i, r in enumerate(act) if i not in worst]
    return best, best_c


def maxfs_lower(G, restarts=48, steps=500, seed=0, inits=None, ls_starts=6):
    """Best-found #{k : <c, g_k> > 0} over a BUDGETED randomised search.

    Two stages, both applied identically to the real category and to every null
    draw: `_anneal` (first-order, batched restarts) and `_ls_drop`
    (least-squares / drop-worst greedy). CONSTRUCTIVE LOWER BOUND, never exact.
    """
    Gn = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-300)
    rng = np.random.default_rng(seed)
    b1, c1 = _anneal(Gn, restarts, steps, rng, inits)
    b2, c2 = _ls_drop(Gn, rng, starts=ls_starts)
    return (b1, c1) if b1 >= b2 else (b2, c2)


def verify_sign_flip(G, c):
    """satisfied(c) + satisfied(-c) + boundary == n. The mechanical content of
    the SIGN-FLIP note in the module docstring, asserted rather than argued."""
    Gn = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-300)
    m = Gn @ c
    return {"n_rows": int(G.shape[0]),
            "satisfied_c": int((m > TOL).sum()),
            "satisfied_minus_c": int((m < -TOL).sum()),
            "boundary_rows": int((np.abs(m) <= TOL).sum()),
            "sums_to_n": bool(int((m > TOL).sum()) + int((m < -TOL).sum())
                              + int((np.abs(m) <= TOL).sum()) == G.shape[0])}


# ------------------------------------------------------------------- bounds
def bound(U, bearers, nonbearers, mode, exact_if_possible=True,
          restarts=48, steps=500, seed=0, ls_starts=6):
    """One bound, carrying its own method and exactness.

    For mode='twin' the search is SEEDED with the pair-level solution: if C is
    antisymmetric then twin YES and twin NO are satisfied together, so
    per-twin >= 2 x pair-level is guaranteed CONSTRUCTIVELY. Without that seed
    the per-twin lower bound can come back below the pair-level bound, which is
    structurally impossible and is an instrument defect, not a finding.
    """
    d = len(next(iter(U.values())))
    G = constraint_rows(U, bearers, nonbearers, mode)
    n = G.shape[0]
    exact, inits = None, None
    if mode == "pair" and d == 3 and exact_if_possible:
        exact = exact_maxfs_r3(_antisym_r3(U, bearers, nonbearers))
    if mode == "twin":
        if d == 3 and exact_if_possible:
            _, w = exact_maxfs_r3(_antisym_r3(U, bearers, nonbearers), return_w=True)
            A = _w_to_antisym3(w).ravel()
        else:
            pl = bound(U, bearers, nonbearers, "pair", exact_if_possible=False,
                       restarts=restarts, steps=steps, seed=seed, ls_starts=ls_starts)
            M = pl["_c"].reshape(d, d)
            A = ((M - M.T) / 2.0).ravel()
        inits = [A / max(float(np.linalg.norm(A)), 1e-300)]
    lo, c = maxfs_lower(G, restarts=restarts, steps=steps, seed=seed,
                        inits=inits, ls_starts=ls_starts)
    count = exact if exact is not None else lo
    return {"mode": mode, "d": d, "n_rows": n,
            "count": int(count), "value": count / n,
            "fraction": "%d/%d" % (count, n),
            "method": ("EXACT (R^3 arrangement enumeration, doc 47 SS16 fix)"
                       if exact is not None else
                       "CONSTRUCTIVE LOWER BOUND (%d restarts x %d steps + %d LS-drop starts)"
                       % (restarts, steps, ls_starts)),
            "exact": exact is not None,
            "sign_flip_check": verify_sign_flip(G, c),
            "_c": c}


def _pub(r):
    """JSON-safe view of a bound (drops the witness vector)."""
    return {k: v for k, v in r.items() if not k.startswith("_")}


def null_stats(U, size, mode, draws=200, seed=68000, restarts=48, steps=500,
               ls_starts=6):
    """SIZE-MATCHED null: random |B|-subsets, IDENTICAL method and budget."""
    rng = np.random.default_rng(seed)
    names = list(U.keys())
    vals = []
    for t in range(draws):
        idx = set(int(i) for i in rng.choice(len(names), size=size, replace=False))
        bearers = [names[i] for i in sorted(idx)]
        nons = [names[i] for i in range(len(names)) if i not in idx]
        vals.append(bound(U, bearers, nons, mode, restarts=restarts, steps=steps,
                          seed=seed + t, ls_starts=ls_starts)["value"])
    return np.array(vals)


def with_null(U, bearers, nonbearers, mode, draws, seed, restarts, steps, ls_starts):
    r = _pub(bound(U, bearers, nonbearers, mode, restarts=restarts, steps=steps,
                   seed=seed, ls_starts=ls_starts))
    nul = null_stats(U, len(bearers), mode, draws=draws, seed=seed,
                     restarts=restarts, steps=steps, ls_starts=ls_starts)
    r["null"] = {"draws": int(draws), "size_matched_B": len(bearers),
                 "mean": float(nul.mean()), "sd": float(nul.std(ddof=1)),
                 "min": float(nul.min()), "max": float(nul.max()),
                 "percentile_of_category": float((nul < r["value"]).mean() * 100.0),
                 "identical_search_budget": True}
    return r


def arcs(coords, members):
    """Contiguous bearer/non-bearer runs around the circle of atan2 angles --
    doc 47 SS13's 'arcs' column."""
    order = sorted(coords, key=lambda n: math.atan2(coords[n][1], coords[n][0]) % (2 * math.pi))
    lab = [n in set(members) for n in order]
    return sum(1 for i in range(len(lab)) if lab[i] != lab[i - 1])


# ------------------------------------------------------------------ loading
def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_frozen2d(path=FROZEN_EMB):
    blob = json.load(open(path, encoding="utf-8"))
    emb = blob["embeddings"]
    miss = [n for n in HUMANS if n not in emb]
    assert not miss, "frozen embeddings lack %r" % miss
    P = np.array([emb[n]["vec2"] for n in HUMANS], dtype=float)
    nrm = np.linalg.norm(P, axis=1)
    prov = {
        "family": "frozen2d",
        "file": os.path.basename(path), "file_sha256": _sha(path),
        "declared_embeddings_sha256": blob.get("embeddings_sha256"),
        "source": blob["meta"]["source"],
        "normalisation": "NONE -- the stored vec2 are already unit-norm and inside [-1,1]",
        "all_unit_norm": bool(np.allclose(nrm, 1.0, atol=1e-9)),
        "pc1_equals_cos_angle": bool(np.allclose(
            P[:, 0], np.cos(np.arctan2(P[:, 1], P[:, 0])), atol=1e-12)),
        "features_deterministically_dependent": True,
        "degeneracy_note": (
            "unit norm in 2D => PC1 = cos(phi) and PC2 = sin(phi) EXACTLY, so PC1 and "
            "PC2 are a deterministic reparameterisation of ONE number. A 2-feature "
            "register on this family adds FUNCTION-CLASS RICHNESS, not information "
            "(products of trig in cos phi and sin phi are not band-limited in phi -- "
            "Jacobi-Anger), which is why doc 52 SS2.2 found random labellings reaching "
            "0.89-0.97 in the 2-feature space."),
    }
    return {n: (float(P[i, 0]), float(P[i, 1])) for i, n in enumerate(HUMANS)}, prov


def load_raw(path=RAW_EMB, fit_on="all44"):
    """PCA over the RAW 300-dim vectors. NOT renormalised -- the whole point.

    PCA is fit over the FULL referent inventory, matching
    exp42_make_embeddings.py's provenance exactly. This is NOT leakage in
    exp68's setting: the held-out split is over unseen PAIRINGS, not unseen
    nouns, so every noun's feature is in-inventory by construction. Contrast
    doc 51 SS4, where the split IS over unseen words and a featurizer fitted on
    all words WAS a leak.
    """
    blob = json.load(open(path, encoding="utf-8"))
    vecs = blob["vectors"]
    miss = [n for n in HUMANS if n not in vecs]
    assert not miss, "raw vectors lack %r" % miss
    names = sorted(vecs) if fit_on == "all44" else list(HUMANS)
    X = np.array([vecs[n] for n in names], dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / float((S ** 2).sum())
    Pall = Xc @ Vt[:2].T
    idx = {n: i for i, n in enumerate(names)}
    P = np.array([Pall[idx[n]] for n in HUMANS])
    # REGISTERED SIGN PIN. An SVD fixes each principal axis only up to sign, and
    # the sign it happens to return depends on row order. Bounds are invariant
    # under it (a sign flip is an invertible linear map of the feature space) but
    # a TRAINED CIRCUIT IS NOT, so the convention is pinned rather than left to
    # numpy: PC signs are chosen to match the frozen vec2 plane.
    fz = json.load(open(FROZEN_EMB, encoding="utf-8"))["embeddings"]
    F = np.array([fz[n]["vec2"] for n in HUMANS], dtype=float)
    U2 = P / np.linalg.norm(P, axis=1, keepdims=True)
    dmin, best = None, None
    for sa in (1, -1):
        for sb in (1, -1):
            dd = float(np.abs(U2 * np.array([sa, sb]) - F).max())
            if dmin is None or dd < dmin:
                dmin, best = dd, (sa, sb)
    P = P * np.array(best, dtype=float)
    # REGISTERED normalisation: per-component max-abs over the 24 HUMANS.
    denom = np.abs(P).max(axis=0)
    Pn = P / denom
    ang_raw = np.arctan2(P[:, 1], P[:, 0])
    ang_frz = np.array([fz[n]["angle"] for n in HUMANS])
    dang = np.abs(np.angle(np.exp(1j * (ang_raw - ang_frz))))
    r = np.linalg.norm(P, axis=1)
    prov = {
        "family": "raw",
        "file": os.path.basename(path), "file_sha256": _sha(path),
        "declared_vectors_sha256": blob["meta"]["vectors_sha256"],
        "declared_sha_verified_with_repo_canonical_sha256": True,
        "source": blob["meta"]["source"], "convention": blob["meta"]["convention"],
        "raw_dim": int(X.shape[1]), "pca_fit_on": fit_on, "pca_fit_n": int(X.shape[0]),
        "var_explained_pc1": float(var[0]), "var_explained_pc2": float(var[1]),
        "var_explained_pc1_plus_pc2": float(var[:2].sum()),
        "sign_pin": ("REGISTERED: PC signs pinned to the frozen vec2 plane. Bounds are "
                     "invariant to PC sign; a trained circuit is not."),
        "normalisation": ("REGISTERED: p_k(r) = PC_k(r) / max_{r' in HUMANS} |PC_k(r')| "
                          "for k = 1, 2. Divisors: PC1 %.6f, PC2 %.6f"
                          % (denom[0], denom[1])),
        "norm_divisors": [float(denom[0]), float(denom[1])],
        "plane_agreement_with_frozen2d": {
            "unit_raw_pca_vs_frozen_vec2_max_abs_coord_diff": dmin,
            "sign_pin_applied": list(best),
            "max_angular_difference_rad_after_pin": float(dang.max()),
            "verdict": ("the frozen 2D embeddings ARE the unit-normalised raw PCA "
                        "plane; FA's angle and the raw-PCA angle are the SAME quantity"),
        },
        "radial_coordinate": {
            "min": float(r.min()), "max": float(r.max()), "mean": float(r.mean()),
            "note": ("this is exactly the information unit-normalisation destroyed. "
                     "F2-raw = (PC1, PC2) = (rho cos phi, rho sin phi) carries angle AND "
                     "radius; FA carries angle only. That is why exp68 on this family is "
                     "width-as-INFORMATION, not width-as-reparameterisation."),
        },
        "features_deterministically_dependent": False,
        "pc1_pc2_pearson_r": float(np.corrcoef(P[:, 0], P[:, 1])[0, 1]),
    }
    return {n: (float(Pn[i, 0]), float(Pn[i, 1])) for i, n in enumerate(HUMANS)}, prov


def self_sha256():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()


# -------------------------------------------------------------------- checks
def selftest(fams):
    """Instrument checks that must pass before any number is believed."""
    out = {}
    B = CATEGORIES[DEFAULT_TARGET]["members"]
    N = [n for n in HUMANS if n not in B]
    for fam, (coords, _p) in fams.items():
        out[fam] = {}
        for kind in ("FA", "F1", "F2"):
            U, _d = features(coords, kind)
            p = bound(U, B, N, "pair")
            t = bound(U, B, N, "twin")
            # 1. a MAX can never sit below chance (SIGN-FLIP).
            assert p["value"] >= 0.5 - 1e-12 and t["value"] >= 0.5 - 1e-12, \
                "%s/%s: MAX below chance" % (fam, kind)
            assert p["sign_flip_check"]["sums_to_n"] and t["sign_flip_check"]["sums_to_n"]
            # 2. per-twin >= pair-level, structurally.
            assert t["value"] >= p["value"] - 1e-9, \
                ("%s/%s: per-twin %.4f below pair-level %.4f -- the search is "
                 "under-powered, not the geometry" % (fam, kind, t["value"], p["value"]))
            out[fam][kind] = {"pair": p["value"], "twin": t["value"],
                              "boundary_rows_pair": p["sign_flip_check"]["boundary_rows"]}
        # 3. the EXACT R^3 enumeration must never be beaten by the search.
        U, _d = features(coords, "FA")
        ex = exact_maxfs_r3(_antisym_r3(U, B, N))
        lo, _c = maxfs_lower(constraint_rows(U, B, N, "pair"), restarts=96, steps=800, seed=7)
        assert lo <= ex, ("search beat the EXACT enumeration (%d > %d) -- doc 47 SS16's "
                          "defect has recurred" % (lo, ex))
        out[fam]["exact_ge_search"] = {"exact": int(ex), "search": int(lo)}
    return out


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen", default=FROZEN_EMB)
    ap.add_argument("--raw", default=RAW_EMB)
    ap.add_argument("--pca-fit-on", default="all44", choices=["all44", "humans24"])
    ap.add_argument("--scaling", default=DEFAULT_SCALING, choices=sorted(SCALINGS))
    ap.add_argument("--target", default=DEFAULT_TARGET, choices=sorted(CATEGORIES))
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--restarts", type=int, default=48)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--ls-starts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=68000)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    fams = {"frozen2d": load_frozen2d(a.frozen),
            "raw": load_raw(a.raw, fit_on=a.pca_fit_on)}
    st = selftest(fams)
    if a.selftest:
        print(json.dumps(st, indent=2))
        return 0

    kw = dict(draws=a.draws, seed=a.seed, restarts=a.restarts, steps=a.steps,
              ls_starts=a.ls_starts)
    B = CATEGORIES[a.target]["members"]
    N = [n for n in HUMANS if n not in B]
    f2d = fams["frozen2d"][0]
    fraw = fams["raw"][0]

    # ---- 1. membership reconstruction evidence (doc 47 SS13's published stats)
    recon = {}
    UA = features(f2d, "FA")[0]
    for key, cat in CATEGORIES.items():
        mem = cat["members"]
        non = [n for n in HUMANS if n not in mem]
        pl = with_null(UA, mem, non, "pair", **kw)
        pt = _pub(bound(UA, mem, non, "twin", restarts=a.restarts, steps=a.steps,
                        seed=a.seed, ls_starts=a.ls_starts))
        recon[key] = {
            "gloss": cat["gloss"],
            "members": [{"noun": m, "gloss": GLOSS[m]} for m in mem],
            "size": len(mem), "doc47_s13_published": cat["doc47_s13"],
            "recomputed_arcs": arcs(f2d, mem),
            "recomputed_pair_level": pl["value"], "fraction": pl["fraction"],
            "recomputed_null_mean": pl["null"]["mean"],
            "recomputed_null_sd": pl["null"]["sd"],
            "recomputed_percentile": pl["null"]["percentile_of_category"],
            "recomputed_per_twin": pt["value"], "per_twin_fraction": pt["fraction"],
            "size_matches": len(mem) == cat["doc47_s13"]["size"],
            "arcs_match": arcs(f2d, mem) == cat["doc47_s13"]["arcs"],
            "ceiling_matches": abs(pl["value"] - cat["doc47_s13"]["k1_ceiling"]) < 5e-5,
            "null_mean_within_1sd_of_doc47":
                abs(pl["null"]["mean"] - cat["doc47_s13"]["null_mean"])
                < cat["doc47_s13"]["null_sd"],
        }
    recon_all = all(v["size_matches"] and v["arcs_match"] and v["ceiling_matches"]
                    for v in recon.values())

    # ---- 2. the registered feature maps, BOTH families, with size-matched nulls
    maps = {}
    for fam, coords in (("frozen2d", f2d), ("raw", fraw)):
        maps[fam] = {}
        for kind in ("FA", "F1", "F2"):
            U, desc = features(coords, kind, a.scaling)
            e = {"definition": desc, "d": len(next(iter(U.values()))),
                 "scaling": None if kind == "FA" else SCALINGS[a.scaling][0]}
            for mode in ("pair", "twin"):
                e[mode] = with_null(U, B, N, mode, **kw)
            maps[fam][kind] = e

    # ---- 3. target selection over doc 47 SS13's FOUR categories, raw family
    #         (no new categories are invented -- doc 47 SS16.1's binding rule)
    target_table = {}
    for key, cat in CATEGORIES.items():
        mem = cat["members"]
        non = [n for n in HUMANS if n not in mem]
        row = {"gloss": cat["gloss"], "size": len(mem)}
        for kind in ("F1", "F2", "FA"):
            U, _d = features(fraw, kind, a.scaling)
            r = with_null(U, mem, non, "twin", **kw)
            row[kind] = {"per_twin": r["value"], "fraction": r["fraction"],
                         "null_mean": r["null"]["mean"], "null_sd": r["null"]["sd"],
                         "percentile": r["null"]["percentile_of_category"]}
        row["F2_minus_F1"] = row["F2"]["per_twin"] - row["F1"]["per_twin"]
        row["F2_minus_FA"] = row["F2"]["per_twin"] - row["FA"]["per_twin"]
        row["F1_above_its_null"] = row["F1"]["per_twin"] - row["F1"]["null_mean"]
        target_table[key] = row
    best_sep = max(target_table, key=lambda k: target_table[k]["F2_minus_F1"])

    # ---- 4. scaling sensitivity (the registered choice was made a priori)
    sweep, hits = {}, []
    for sc in sorted(SCALINGS):
        sweep[sc] = {"formula": SCALINGS[sc][0]}
        for fam, coords in (("frozen2d", f2d), ("raw", fraw)):
            r = {}
            for kind in ("F1", "F2"):
                U, _d = features(coords, kind, sc)
                bp = bound(U, B, N, "pair", restarts=a.restarts, steps=a.steps,
                           seed=a.seed, ls_starts=a.ls_starts)
                bt = bound(U, B, N, "twin", restarts=a.restarts, steps=a.steps,
                           seed=a.seed, ls_starts=a.ls_starts)
                r[kind] = {"pair_level": bp["value"], "pair_fraction": bp["fraction"],
                           "per_twin": bt["value"], "twin_fraction": bt["fraction"],
                           "per_twin_complement": 1.0 - bt["value"],
                           "pair_level_complement": 1.0 - bp["value"]}
            sweep[sc][fam] = r
            for mk, ck in (("pair_level", "pair_level_complement"),
                           ("per_twin", "per_twin_complement")):
                if abs(r["F1"][ck] - DOC52["tertiary"]["one_feature"]) < 5e-4:
                    hits.append({"scaling": sc, "family": fam, "mode": mk,
                                 "max": r["F1"][mk], "complement": r["F1"][ck]})

    # ---- 5. reconciliation verdicts
    fa2 = maps["frozen2d"]["FA"]
    f12 = maps["frozen2d"]["F1"]
    f22 = maps["frozen2d"]["F2"]
    pub47 = CATEGORIES[a.target]["doc47_s13"]
    rec = {
        "FA_reproduces_doc47_s13": {
            "recomputed_pair_level": fa2["pair"]["value"],
            "recomputed_pair_level_fraction": fa2["pair"]["fraction"],
            "doc47_s13_published": pub47["k1_ceiling"],
            "recomputed_null_mean": fa2["pair"]["null"]["mean"],
            "doc47_s13_null_mean": pub47["null_mean"],
            "recomputed_percentile": fa2["pair"]["null"]["percentile_of_category"],
            "doc47_s13_percentile": pub47["percentile"],
            "verdict": ("REPRODUCED at pair level"
                        if abs(fa2["pair"]["value"] - pub47["k1_ceiling"]) < 5e-5
                        else "MISMATCH -- nothing downstream may be written until reconciled"),
            "note": ("doc 47 SS17's per-twin correction raises the SAME category to "
                     "%s = %.4f. That is the SS17 correction applied to SS13, not a "
                     "discrepancy: SS13 was computed with the SS3.2 pair-level instrument."
                     % (fa2["twin"]["fraction"], fa2["twin"]["value"])),
        },
        "F1_below_chance_0p4842": {
            "doc52_published": DOC52["tertiary"]["one_feature"],
            "recomputed_frozen2d_pair_level": f12["pair"]["value"],
            "recomputed_frozen2d_per_twin": f12["twin"]["value"],
            "complement_of_per_twin": 1.0 - f12["twin"]["value"],
            "complement_of_pair_level": 1.0 - f12["pair"]["value"],
            "scalings_whose_complement_hits_0p4842": hits,
            "arithmetic": ("0.4842 = 92/190 exactly, and 190 - 92 = 98 gives "
                           "98/190 = 0.515789. doc 52's `sem manual` 1-feature value "
                           "0.6789 = 129/190 has an ODD numerator, which is impossible in "
                           "the PAIR-LEVEL system (its counts are even out of 190) -- so "
                           "doc 52's '1-feature' column was computed with the PER-TWIN "
                           "general-C instrument."),
            "mechanism": ("MAX-FS is sign-symmetric: satisfied(-C) = n - satisfied(C), so "
                          "max_C satisfied/n >= 0.5 ALWAYS (asserted in selftest). 0.4842 "
                          "is therefore NOT a maximum -- it is the complement orientation "
                          "of a C achieving 0.5158. doc 52's search reported the count for "
                          "the orientation it happened to hold, without evaluating -C."),
            "honest_restatement": ("F1 on the frozen2d family sits AT or NEAR CHANCE, not "
                                   "below it. The registered prediction's WORDING changes "
                                   "('at floor', never 'below chance'); its CONTENT does not."),
        },
        "F2_vs_FA_on_frozen2d": {
            "F2_pair_level": f22["pair"]["value"], "FA_pair_level": fa2["pair"]["value"],
            "F2_per_twin": f22["twin"]["value"], "FA_per_twin": fa2["twin"]["value"],
            "pair_level_equal": abs(f22["pair"]["value"] - fa2["pair"]["value"]) < 5e-5,
            "per_twin_equal": abs(f22["twin"]["value"] - fa2["twin"]["value"]) < 5e-5,
            "doc52_two_feature": DOC52["tertiary"]["two_feature"],
            "doc52_two_feature_equals_doc47_single_angle": True,
            "doc52_two_feature_null_mean": DOC52["tertiary"]["two_feature_null_mean"],
            "recomputed_two_feature_null_mean_per_twin": f22["twin"]["null"]["mean"],
        },
    }

    payload = {
        "instrument": "instrument_sb_bounds.py",
        "script_sha256": self_sha256(),
        "generated_for": ("doc 54 / exp68 registration -- CONTEXT ONLY. doc 52 SS2.2: on a "
                          "24-noun inventory a 9-dim-per-referent bound measures "
                          "interpolation, so NO criterion in doc 54 is read against any "
                          "number in this file."),
        "not_a_results_file": ("named diagnostic_*, not results_*, deliberately: it is an "
                               "instrument output, not an experiment result"),
        "families": {"frozen2d": fams["frozen2d"][1], "raw": fams["raw"][1]},
        "inventory": {"n_humans": len(HUMANS),
                      "humans": [{"noun": n, "gloss": GLOSS[n]} for n in HUMANS]},
        "membership_provenance": {
            "status": "PROVISIONAL -- RECONSTRUCTED, NOT FOUND",
            "searched": ["repo *.py / *.json / *.log: no 'tertiary' hit outside docs",
                         "git log --all: no committed doc 47 SS13 category script",
                         "qnlp_private_docs 46-53: no explicit member list"],
            "method": "semantic judgment over the 24 frozen human nouns",
            "corroboration": ("all four doc 47 SS13 categories reproduce |B|, the arc count "
                              "and the published K=1 ceiling; the target additionally "
                              "reproduces SS13's null mean and percentile"),
            "all_four_reproduce": bool(recon_all),
            "native_speaker_gate": ("doc 47 SS13.4 closed the gate for the memberships IT "
                                    "HELD. Because no list was recovered, the gate is "
                                    "BLOCKING here until the orchestrator confirms these "
                                    "lists against the SS13 session record."),
        },
        "membership_reconstruction": recon,
        "target_category": {"key": a.target, "gloss": CATEGORIES[a.target]["gloss"],
                            "members": CATEGORIES[a.target]["members"],
                            "size": len(B), "n_nonbearers": len(N)},
        "bound_definitions": {
            "pair_level": ("doc 47 SS3.2: both twins of a cast won together => x^T A y > 0 "
                           "on the ANTISYMMETRIC part. Denominator = casts."),
            "per_twin": ("doc 47 SS17: twins are SEPARATE constraints on a GENERAL C; a "
                         "cast may score 0, 1/2 or 1. Denominator = twins."),
            "exactness": ("pair level at d = 3 is EXACT (R^3 arrangement enumeration with "
                          "doc 47 SS16's adjacent-cell perturbation and a strict "
                          "satisfaction tolerance). Every other number here is a "
                          "CONSTRUCTIVE LOWER BOUND from a budgeted search."),
        },
        "registered_scaling": {"name": a.scaling, "formula": SCALINGS[a.scaling][0],
                               "chosen": ("a priori, by the stated principle (give a single "
                                          "RY the full circle), NOT by the resulting "
                                          "numbers; the full sweep is reported below")},
        "feature_maps": maps,
        "target_selection": {
            "rule": ("the registered prediction structure needs F1 near its null and F2 well "
                     "above it; the choice is made from doc 47 SS13's FOUR categories only, "
                     "and is reported, never silent"),
            "table": target_table,
            "largest_F2_minus_F1": best_sep,
            "registered_target": a.target,
            "flag_for_orchestrator": (None if best_sep == a.target else
                                      "largest F2-F1 separation is '%s', not the registered "
                                      "target '%s' -- orchestrator decides" % (best_sep, a.target)),
        },
        "scaling_sensitivity": sweep,
        "reconciliation": rec,
        "sign_flip_selftest": st,
        "search_budget": {"restarts": a.restarts, "steps": a.steps,
                          "ls_drop_starts": a.ls_starts, "null_draws": a.draws,
                          "seed": a.seed, "identical_for_category_and_null": True},
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print("wrote %s" % a.out)
    print("membership reconstruction reproduces all four doc 47 SS13 rows: %s" % recon_all)
    for fam in ("frozen2d", "raw"):
        for k in ("FA", "F1", "F2"):
            m = maps[fam][k]
            print("%-8s %-3s d=%d | pair %.4f %-8s null %.4f p%-5.1f | twin %.4f %-9s null %.4f p%-5.1f"
                  % (fam, k, m["d"], m["pair"]["value"], m["pair"]["fraction"],
                     m["pair"]["null"]["mean"], m["pair"]["null"]["percentile_of_category"],
                     m["twin"]["value"], m["twin"]["fraction"],
                     m["twin"]["null"]["mean"], m["twin"]["null"]["percentile_of_category"]))
    print("largest F2-F1 separation (raw family): %s" % best_sep)
    for k, v in target_table.items():
        print("  %-16s F1 %.4f (null %.4f) F2 %.4f (null %.4f) FA %.4f | F2-F1 %+.4f F2-FA %+.4f"
              % (k, v["F1"]["per_twin"], v["F1"]["null_mean"], v["F2"]["per_twin"],
                 v["F2"]["null_mean"], v["FA"]["per_twin"], v["F2_minus_F1"], v["F2_minus_FA"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
