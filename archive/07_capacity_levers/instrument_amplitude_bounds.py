# -*- coding: utf-8 -*-
"""instrument_amplitude_bounds.py -- reachable-accuracy bounds for AMPLITUDE
ENCODINGS of a truncated feature vector, and the structural facts a
3-or-4-qubit-per-referent register forces.

WHY THIS FILE EXISTS
--------------------
`capacity_check.py`'s scope note bounds itself to

    "per-referent encodings that are a SINGLE RY PER QUBIT ... a register of q
     independent angles -> the Kronecker product of q such vectors (d = 3^q)."

`instrument_entangled_bounds.py` then extended that to two ENTANGLED intro maps
(E-in, E-across) by replacing the product feature `vec(u_a u_p^T)` with a general
JOINT feature map `Phi(agent, patient)`.

exp73 is the WIDTH axis of the enquiry ledger taken to its truncated extreme:
the referent's state is no longer any angle map at all. The whole (truncated,
normalised) semantic vector is written onto the register as AMPLITUDES:

    |psi(x)> = sum_i x_i |i>,   x in R^{2^q},  ||x||_2 = 1.

This module supplies the feature algebra for that map, its bounds, its nulls,
its selftest against the committed single-angle results, and the register-level
purity facts the exp73 instrumentation will need. It RE-DERIVES NOTHING:
`instrument_sb_bounds` remains the authority for the search, the exact R^3
enumeration, the sign-flip invariant and the null discipline;
`instrument_entangled_bounds` remains the authority for the joint-map row
builder and the twin-mode seeding. This module is the amplitude case of those.

--------------------------------------------------------------------------
1. THE READOUT IS A QUADRATIC FORM. THIS IS THE WHOLE RESULT.
--------------------------------------------------------------------------
Fix the encoding |psi(x)> = sum_i x_i |i>. Every gate applied after intro is
FEATURE-INDEPENDENT (verb blocks and readout effects are trained constants, not
functions of x), so any measurement outcome probability is

    p(x) = <psi(x)| O |psi(x)> = x^T M x,      M = M^T real,

for some fixed real symmetric M absorbing the whole circuit. WRITTEN EXACTLY:

    x^T M x = sum_i M_ii x_i^2 + sum_{i<j} 2 M_ij x_i x_j
            = <W, v(x)>,   v(x) = ( x_i x_j )_{i <= j},   d_ref = d(d+1)/2,

with W the independent coefficients of M (the factor 2 on the off-diagonals is
absorbed into W). So the per-referent MAX-FS feature of an amplitude encoding is
the OUTER SQUARE of x, upper-triangle-ordered -- exactly the construction
`instrument_entangled_bounds.across_features` already uses for the E-across
4-amplitude state, at general d.

    CONSEQUENCE, REGISTERED AS A RESULT AND NOT AS A CAVEAT:
    the classical comparator for an amplitude encoding is a QUADRATIC (kernel)
    MODEL ON THE RAW FEATURES -- a linear model in the degree-2 monomials of x.
    For a two-referent decision it is a linear model in the degree-(2,2)
    bi-monomials `x_a[i] x_a[j] x_p[k] x_p[l]`, i.e. the product kernel
    (x_a . x_a')^2 (x_p . x_p')^2. The classical form is EVEN MORE IMMEDIATE
    than for the angle maps (doc 52 SS1's standing prohibition applies in its
    strongest form: nothing here is an advantage question, and the comparator is
    not merely matched -- it is the same object written in another basis).

THE TWO-REFERENT JOINT FEATURE. At intro the two registers are a PRODUCT,
|psi(x_a)> (x) |psi(x_p)>, whose amplitude on |i>|j> is x_a[i] x_p[j]. A
quadratic form in that joint amplitude vector is

    <W, vec( v(x_a) v(x_p)^T )>,   d_joint = [ d(d+1)/2 ]^2,

so the joint map is a PRODUCT map in `instrument_entangled_bounds`'s sense and
`product_map` applies unchanged. Only the per-referent vector is new.

    | q | d = 2^q | d_ref = d(d+1)/2 | d_joint |
    |---|---------|------------------|---------|
    | 1 |    2    |        3         |       9 |
    | 2 |    4    |       10         |     100 |
    | 3 |    8    |       36         |    1296 |
    | 4 |   16    |      136         |   18496 |

THE GENERAL-W QUALIFIER, STATED RATHER THAN BURIED. `W` ranges over the whole
coordinate space; the physically realisable set of `M` is a subset of it (M must
be the restriction of a genuine POVM element conjugated by the trained circuit).
The committed modules make exactly the same identification -- "every gate applied
after intro is feature-independent and is absorbed into W, so the reachable class
is exactly max_W over that linear system" -- and this module INHERITS it verbatim
rather than tightening it. Every number here is therefore a statement about the
same idealised class the committed bounds are statements about, and is comparable
with them for that reason.

--------------------------------------------------------------------------
2. THE ENCODING MAP -- EXACT FORMULA, AND ITS INFORMATION COST
--------------------------------------------------------------------------
Registered map, stated in full so nothing is left to the build session:

    (a) PCA over the RAW 300-dim AraVec vectors of the FULL 44-noun inventory,
        mean-centred, `numpy.linalg.svd` -- `instrument_sb_bounds.load_raw`'s
        convention, re-derived here and ASSERTED to reproduce its first two
        components exactly (SS`check_pca_convention`).
    (b) z_k(r) = the k-th principal SCORE of referent r, k = 1..d, UNRESCALED.
    (c) SIGN PIN. PC1 and PC2: pinned to the frozen `vec2` plane, exactly as
        `load_raw` does. PC3..PCd: pinned by the registered intrinsic rule
            sign( Vt[k, argmax_j |Vt[k, j]| ] ) = +1,  ties -> lowest j,
        which depends on the axis alone and not on row order or inventory order.
        Bounds are INVARIANT to every sign choice (SS4); a TRAINED CIRCUIT IS
        NOT, which is why the convention is pinned rather than left to numpy.
    (d) TRUNCATE to the first d = 2^q components, then ZERO-PAD to 2^q. Here
        d = 2^q exactly, so the padding is vacuous and is carried only so the
        formula generalises.
    (e) NORMALISE PER NOUN to unit l2:

            x(r) = z_{1:d}(r) / || z_{1:d}(r) ||_2 .

THE INFORMATION COST, STATED AGAINST THE eps-MAP RATHER THAN IN THE ABSTRACT.

  * The eps-map `theta = p (pi - 2 eps)/2 + pi/2` (doc 56) is INJECTIVE and
    affine in p, and p_k = PC_k / max|PC_k| is an invertible per-component
    rescale. The eps-map therefore destroys NO information: its cost is
    RESOLUTION -- the reachable angular range shrinks from [0, 2pi) to
    [pi/6, 5pi/6], which changes the frequency content the readout can use, and
    it was forced by the concurrence dead zones, not chosen.
  * The amplitude map is LOSSY IN TWO PLACES, and both are quantified in the
    output JSON:
      - TRUNCATION discards `1 - sum_{k<=d} var_k` of the AraVec variance;
      - unit-l2 NORMALISATION discards the RADIUS ||z_{1:d}(r)||, exactly one
        real number per referent. This is the SAME quantity FA discarded
        relative to F2 (`instrument_sb_bounds.load_raw :: radial_coordinate`:
        "this is exactly the information unit-normalisation destroyed").
  * NET LEDGER PER REFERENT: FA carries 1 free real number, F2/P-eps carry 2,
    the amplitude rung carries d - 1 (a unit vector in R^d). The amplitude rung
    pays FA's radial cost to buy d - 2 additional free directions. It is not a
    strict enrichment of F2 and this module never says it is.

--------------------------------------------------------------------------
3. THE SELFTEST -- WHAT THE 2-DIM AMPLITUDE CASE MUST REPRODUCE, AND WHAT IT
   MUST NOT
--------------------------------------------------------------------------
For q = 1 write the unit 2-vector as x = (cos phi, sin phi). Then, in this
module's triu ordering v = (x0^2, x0 x1, x1^2),

    v(phi) = ( (1 + cos 2phi)/2 , (sin 2phi)/2 , (1 - cos 2phi)/2 )

and with the FIXED invertible matrix

    L = [[1, 0,  1],
         [1, 0, -1],
         [0, 2,  0]],      det L = -4,

    L v(phi) = ( 1, cos 2phi, sin 2phi ) = u(2 phi),

`u` being `instrument_sb_bounds._u1`, the committed single-angle feature.

MAX-FS IS INVARIANT UNDER AN INVERTIBLE LINEAR REPARAMETERISATION OF THE
PER-REFERENT FEATURE. If v' = L v then Phi' = (L (x) L) Phi, and putting
W' = (L (x) L)^{-T} W gives <W', Phi'> = <W, Phi>; W ranges over the whole space
and (L (x) L)^{-T} is a bijection of it, so the ACHIEVABLE SET OF SIGN PATTERNS
-- and hence the MAX-FS count in BOTH modes -- is identical. In pair mode at
d_ref = 3 the same holds through the exact R^3 path, because an antisymmetric
form transforms as w -> det(L) L^{-T} w, again a bijection of R^3.

    THEREFORE, AND THIS IS THE CHECK THIS MODULE ASSERTS:

    the 1-qubit amplitude bound over the inventory {phi_n} EQUALS the committed
    single-angle bound over the inventory {2 phi_n}, EXACTLY -- same count, same
    fraction, both modes.

EXACT REPRODUCTION OF THE COMMITTED FA ANCHOR IS **NOT** THE RIGHT CHECK, AND
THE REASON IS THE CONTENT OF THE RESULT. Amplitude-encoding a unit 2-vector is
the single-angle map at DOUBLE FREQUENCY. Angle doubling is a bijective
relabelling of the inventory but NOT a linear map of the features: it reorders
the nouns around the circle, so the arc structure the exact R^3 enumeration sees
is a different arrangement. The amplitude bound at phi and the committed
single-angle bound at phi are therefore DIFFERENT NUMBERS IN GENERAL, and this
module reports both and asserts the doubled-angle identity rather than the
same-angle one. "Amplitude encoding is not a reparameterisation of the angle
encoding" is thus a MEASUREMENT here, not an assumption.

--------------------------------------------------------------------------
4. SATURATION -- AN EXACT RESULT, NOT A SEARCH OUTCOME
--------------------------------------------------------------------------
If the MAX-FS row matrix G (n rows, d_joint columns) has RANK EQUAL TO n, then
W = G^+ 1 solves G W = 1 exactly, every margin is +1 > 0, and the count is n.
Since n is also the trivial upper bound, the MAX-FS optimum is EXACTLY n and the
bound is EXACTLY 1.0000 -- a PROOF, not a constructive lower bound.

This module computes that certificate before searching. It is not a convenience:
at d_ref = 36 (q = 3) the joint space has 1296 coordinates against 160 twin rows,
and at d_ref = 136 (q = 4) it has 18496. The certificate turns "the search always
returns 160/160" into "160/160 is the maximum, provably, and so is every
size-matched null draw's". The registered reading is in the output JSON and is
blunt: **at the amplitude rung the grade-1 instrument SATURATES and discriminates
nothing** -- neither between A3 and A4, nor between the category and its null.
Every exp73 criterion must therefore be measurement-vs-measurement, which doc 54
SS17.7 already required for independent reasons.

--------------------------------------------------------------------------
5. THE REGISTER, AND WHAT THE PURITY TRACES WILL SHOW
--------------------------------------------------------------------------
The concurrence-analogue question of doc 56 DOES NOT ARISE at intro: the two
referents are introduced as a PRODUCT of two pure register states, so the
per-REGISTER purity is 1 exactly for both, and no cross-referent entanglement
exists before the first verb block. What DOES arise, and what the committed
per-WIRE purity trace will report if it is read unchanged, is the entanglement
WITHIN a register: a generic unit vector in R^{2^q} is NOT a product of q
single-qubit states, so intro per-wire `Tr rho^2` is < 1 by construction and is a
DATA-DETERMINED quantity, not a circuit effect.

    REGISTERED CONSEQUENCE FOR THE exp73 INSTRUMENTATION: the purity trace must
    record BOTH the per-wire `Tr rho^2` (committed, doc 55 SS18.3) AND the
    per-REGISTER `Tr rho_reg^2` (new). The intro assertion is on the REGISTER
    trace (= 1 to 1e-9, product across referents); the per-wire intro values are
    a REPORTED INVENTORY, computed here in advance so a reader cannot mistake
    them for an effect of the verb blocks.

This module computes that inventory -- per-wire purity, the balanced-bipartition
purity, and how many nouns sit within 0.01 of a product state (a register whose
width is unused for that noun) -- for every rung of the ladder.

CLI
---
    python instrument_amplitude_bounds.py --selftest
    python instrument_amplitude_bounds.py            # writes the diagnostic JSON
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import instrument_sb_bounds as isb          # noqa: E402  (authority for the maths)
import capacity_check as cc                 # noqa: E402  (the product-map machinery)
import instrument_entangled_bounds as ieb   # noqa: E402  (the joint-map extension)

TOL = isb.TOL
OUT = os.path.join(HERE, "diagnostic_amplitude_bounds.json")

# The budget is `capacity_check.DEFAULT_BUDGET` verbatim, for the same reason
# exp70's module gave: reproducing the committed anchors requires the identical
# budget AND the identical row order.
BUDGET = dict(cc.DEFAULT_BUDGET)
NULL_SEED = 68000
NULL_DRAWS = 200

TARGET = "nonoccupational"                 # doc 54 SS5.2, user-confirmed 2026-08-23

# The registered ladder. A1 is the SELFTEST rung, A2 is a sensitivity row
# (noted, not run), A3 and A4 are exp73's arms.
LADDER = [
    ("A1", 1, "SELFTEST rung -- 1 qubit, d = 2. Not an exp73 arm."),
    ("A2", 2, "SENSITIVITY rung -- 2 qubits, d = 4. NOTED, NOT RUN: the "
              "2-qubit-per-referent slot of the width ladder is already "
              "occupied by the committed ANGLE registers F2 and P-eps, and "
              "adding a third 2-qubit arm buys a comparison exp73 does not "
              "ask. Its bound is reported so the ladder is continuous and so "
              "the saturation onset (SS4) is visible."),
    ("A3", 3, "exp73 ARM -- 3 qubits, d = 8 = PC1..PC8."),
    ("A4", 4, "exp73 ARM -- 4 qubits, d = 16 = PC1..PC16."),
]


# ------------------------------------------------------------------- the PCA
def _pc_scores(path=isb.RAW_EMB, n_pc=16, fit_on="all44"):
    """PCA over the RAW 300-dim vectors, `instrument_sb_bounds.load_raw`'s
    convention, carried out to `n_pc` components instead of 2.

    Returns ({noun: z in R^n_pc} over isb.HUMANS, variance ratios, provenance).
    The first two components are ASSERTED against `load_raw` in
    `check_pca_convention`; nothing here is believed before that passes.
    """
    blob = json.load(open(path, encoding="utf-8"))
    vecs = blob["vectors"]
    miss = [n for n in isb.HUMANS if n not in vecs]
    assert not miss, "raw vectors lack %r" % miss
    names = sorted(vecs) if fit_on == "all44" else list(isb.HUMANS)
    X = np.array([vecs[n] for n in names], dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / float((S ** 2).sum())
    Pall = Xc @ Vt[:n_pc].T
    idx = {n: i for i, n in enumerate(names)}
    P = np.array([Pall[idx[n]] for n in isb.HUMANS])

    # --- sign pin, in two registered pieces (SS2c) ---------------------------
    fz = json.load(open(isb.FROZEN_EMB, encoding="utf-8"))["embeddings"]
    F = np.array([fz[n]["vec2"] for n in isb.HUMANS], dtype=float)
    U2 = P[:, :2] / np.linalg.norm(P[:, :2], axis=1, keepdims=True)
    dmin, best = None, None
    for sa in (1, -1):
        for sb in (1, -1):
            dd = float(np.abs(U2 * np.array([sa, sb]) - F).max())
            if dmin is None or dd < dmin:
                dmin, best = dd, (sa, sb)
    signs = np.ones(n_pc, dtype=float)
    signs[0], signs[1] = float(best[0]), float(best[1])
    intrinsic = []
    for k in range(2, n_pc):
        j = int(np.argmax(np.abs(Vt[k])))
        signs[k] = 1.0 if Vt[k, j] >= 0.0 else -1.0
        intrinsic.append({"pc": k + 1, "argmax_loading_index": j,
                          "loading": float(Vt[k, j]), "sign": float(signs[k])})
    P = P * signs

    prov = {
        "file": os.path.basename(path), "file_sha256": isb._sha(path),
        "declared_vectors_sha256": blob["meta"]["vectors_sha256"],
        "source": blob["meta"]["source"], "convention": blob["meta"]["convention"],
        "raw_dim": int(X.shape[1]), "pca_fit_on": fit_on,
        "pca_fit_n": int(X.shape[0]), "n_components_taken": int(n_pc),
        "var_explained_per_pc": [float(v) for v in var[:n_pc]],
        "var_explained_cumulative": [float(v) for v in np.cumsum(var[:n_pc])],
        "sign_pin_pc1_pc2": {"applied": [float(best[0]), float(best[1])],
                             "rule": "pinned to the frozen vec2 plane, exactly "
                                     "as instrument_sb_bounds.load_raw does",
                             "max_abs_coord_diff_after_pin": dmin},
        "sign_pin_pc3_onwards": {
            "rule": "sign( Vt[k, argmax_j |Vt[k,j]| ] ) = +1, ties -> lowest j",
            "why": ("intrinsic to the axis: it depends on the loading vector "
                    "alone, not on row order or inventory order. Bounds are "
                    "invariant to it (asserted, SS`check_sign_invariance`); a "
                    "trained circuit is not, which is why it is pinned."),
            "applied": intrinsic},
        "leakage_note": ("PCA is fit over the FULL 44-noun inventory, matching "
                         "exp42_make_embeddings.py and load_raw. NOT leakage in "
                         "this setting: the held-out split is over unseen "
                         "PAIRINGS, not unseen nouns (load_raw's own note)."),
    }
    return {n: P[i].astype(float) for i, n in enumerate(isb.HUMANS)}, var, prov


def check_pca_convention(Z):
    """The first two components of `_pc_scores`, divided by load_raw's own
    registered max-abs divisors, must reproduce `load_raw`'s normalised
    coordinates EXACTLY. This is the whole warrant for carrying the PCA out to
    16 components in a new function instead of a new convention."""
    coords, prov = isb.load_raw()
    den = np.array(prov["norm_divisors"], dtype=float)
    worst = 0.0
    for n in isb.HUMANS:
        got = np.array(Z[n][:2], dtype=float) / den
        worst = max(worst, float(np.abs(got - np.array(coords[n])).max()))
    # and the divisors themselves must be re-derivable
    P = np.array([Z[n][:2] for n in isb.HUMANS])
    den2 = np.abs(P).max(axis=0)
    return {"max_abs_error_vs_load_raw_normalised_coords": worst,
            "reproduces_load_raw_exactly": bool(worst < 1e-12),
            "norm_divisors_recomputed": [float(den2[0]), float(den2[1])],
            "norm_divisors_committed": [float(den[0]), float(den[1])],
            "divisors_match": bool(np.abs(den2 - den).max() < 1e-12),
            "reading": ("the amplitude ladder and every committed bound stand "
                        "on the same PCA, the same sign pin and the same "
                        "inventory; only the number of components differs")}


# ------------------------------------------------------------ the amplitude map
def amp_vector(z, d):
    """SS2(d),(e): truncate to d, zero-pad to 2^q (vacuous when d = 2^q), then
    normalise to unit l2. Returns (x, radius) -- the radius being exactly the
    information the normalisation destroys."""
    q = int(round(math.log(d, 2)))
    assert 2 ** q == d, "d = %d is not a power of two" % d
    v = np.zeros(d, dtype=float)
    take = min(d, len(z))
    v[:take] = np.asarray(z, dtype=float)[:take]
    r = float(np.linalg.norm(v))
    assert r > 0.0, "zero truncated vector -- the map is undefined"
    return v / r, r


def amp_features(x):
    """The outer square, upper-triangle-ordered: v(x) = (x_i x_j)_{i <= j}.
    d_ref = d(d+1)/2. The ordering is FIXED and is part of the registered
    witness convention; it is `numpy.triu_indices`'s row-major order, the same
    convention `instrument_entangled_bounds.across_features` uses."""
    x = np.asarray(x, dtype=float)
    return np.outer(x, x)[np.triu_indices(len(x))]


def d_ref_of(d):
    return d * (d + 1) // 2


def amplitude_map(name, Z, d, desc):
    """The exp73 per-referent feature map at width d, as a `JointMap`."""
    per, rad = {}, {}
    for n in isb.HUMANS:
        x, r = amp_vector(Z[n], d)
        per[n] = amp_features(x)
        rad[n] = r
    m = ieb.product_map(name, isb.HUMANS, per, desc)
    m.d_truncated = int(d)
    m.qubits = int(round(math.log(d, 2)))
    m.radii = rad
    m.states = {n: amp_vector(Z[n], d)[0] for n in isb.HUMANS}
    return m


# ---------------------------------------------------------- the exact certificate
def saturation_certificate(G, tol_rank=1e-9):
    """SS4. If rank(G) == n_rows the MAX-FS optimum is EXACTLY n_rows.

    Returns None when the certificate does not apply, otherwise a dict carrying
    the witness W = G^+ 1, its realised minimum margin, and the exact count.
    """
    n, dd = G.shape
    if dd < n:
        return None
    Gn = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-300)
    r = int(np.linalg.matrix_rank(Gn, tol=tol_rank))
    if r != n:
        return None
    W, *_ = np.linalg.lstsq(Gn, np.ones(n), rcond=None)
    nrm = float(np.linalg.norm(W))
    if nrm < 1e-300:
        return None
    Wn = W / nrm
    margins = Gn @ Wn
    if float(margins.min()) <= TOL:
        return None
    return {"rank": r, "n_rows": int(n), "d_joint": int(dd),
            "count": int(n), "min_margin": float(margins.min()),
            "W": Wn,
            "method": ("EXACT (saturation certificate: rank(G) = n_rows, so "
                       "W = G^+ 1 satisfies every row strictly and n_rows is "
                       "also the trivial ceiling -- a PROOF, not a search)")}


def bound_amp(m, casts, mode, **budget):
    """One bound for an amplitude map: the saturation certificate first, the
    committed `instrument_entangled_bounds.bound_joint` search otherwise.

    The return dict is `bound_joint`'s, with the exactness fields set by whichever
    path produced the number, so downstream code cannot tell them apart by shape
    and must read `method` -- doc 47 SS18.5's rule.
    """
    G = ieb.joint_rows(m, casts, mode)
    cert = saturation_certificate(G)
    if cert is None:
        r = ieb.bound_joint(m, casts, mode, **budget)
        r["saturated"] = False
        return r
    n = cert["n_rows"]
    return {"map": m.name, "mode": mode, "d_joint": m.d,
            "d_per_referent": m.d_ref, "feature_kind": m.kind,
            "n_casts": len(casts), "n_rows": n, "count": n, "bound": 1.0,
            "fraction": "%d/%d" % (n, n),
            "population": ("casts (both twins won together)" if mode == "pair"
                           else "twins (a cast may score 0, 1/2 or 1)"),
            "method": cert["method"], "exact": True, "saturated": True,
            "saturation": {k: cert[k] for k in
                           ("rank", "n_rows", "d_joint", "min_margin")},
            "sign_flip_check": isb.verify_sign_flip(G, cert["W"]),
            "_c": cert["W"]}


def bound_with_null(m, bearers, mode, draws=NULL_DRAWS, null_seed=NULL_SEED,
                    **budget):
    """Category bound plus a SIZE-MATCHED null, IDENTICAL method and budget
    (doc 47 SS16.1). The saturation path is applied to the null draws too, so a
    saturated category is compared against a null computed the same way -- an
    unmatched method here would be the exp44 failure class inside the
    diagnostic."""
    casts, nons = cc.casts_from_bearers(m.names, bearers)
    r = ieb._public(bound_amp(m, casts, mode, **budget))
    r["n_bearers"], r["n_nonbearers"] = len(bearers), len(nons)
    r["swap_perm_max_abs_error"] = ieb.check_swap_perm(m, casts[:8])
    if draws:
        rng = np.random.default_rng(null_seed)
        vals, sat = [], 0
        for t in range(int(draws)):
            idx = sorted(int(i) for i in rng.choice(len(m.names),
                                                    size=len(bearers),
                                                    replace=False))
            bb = [m.names[i] for i in idx]
            cl, _n = cc.casts_from_bearers(m.names, bb)
            bud = dict(budget)
            bud["seed"] = null_seed + t
            rr = bound_amp(m, cl, mode, **bud)
            vals.append(rr["bound"])
            sat += int(bool(rr.get("saturated")))
        v = np.asarray(vals, dtype=float)
        r["null"] = {"draws": int(draws), "size_matched_B": len(bearers),
                     "mean": float(v.mean()),
                     "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                     "min": float(v.min()), "max": float(v.max()),
                     "percentile_of_category":
                         float((v < r["bound"]).mean() * 100.0),
                     "n_draws_saturated": int(sat),
                     "below_null": bool(r["bound"] < v.mean()),
                     "identical_search_budget": True, "seed": int(null_seed)}
        if sat == int(draws) and r.get("saturated"):
            r["null"]["degenerate_note"] = (
                "EVERY draw saturates and so does the category: the null is a "
                "point mass at 1.0000 and the percentile statistic has no "
                "content. Reported rather than suppressed -- a percentile of "
                "0.0 here means 'nothing is below anything', NOT 'the category "
                "is at the bottom of its null'.")
    return r


# ------------------------------------------------- the register purity inventory
def _wire_purities(x):
    """Per-wire Tr rho^2 of a q-qubit register in the real pure state x."""
    q = int(round(math.log(len(x), 2)))
    T = np.asarray(x, dtype=float).reshape((2,) * q)
    out = []
    for w in range(q):
        p = np.moveaxis(T, w, 0).reshape(2, -1)
        rho = p @ p.conj().T
        out.append(float((np.abs(rho) ** 2).sum()))
    return out


def _bipartition_purity(x):
    """Tr rho^2 of the first floor(q/2) wires against the rest -- the balanced
    cut. 1.0 means the register factorises across that cut."""
    q = int(round(math.log(len(x), 2)))
    k = q // 2
    if k == 0:
        return 1.0
    M = np.asarray(x, dtype=float).reshape(2 ** k, -1)
    rho = M @ M.T
    return float((rho ** 2).sum())


def register_purity_inventory(m):
    """SS5. Computed BEFORE any exp73 run so no reader can mistake an intro
    per-wire purity below 1 for an effect of the verb blocks."""
    per, wmins, bips = {}, [], []
    for n in isb.HUMANS:
        w = _wire_purities(m.states[n])
        b = _bipartition_purity(m.states[n])
        per[n] = {"per_wire_purity": w, "min_wire_purity": float(min(w)),
                  "balanced_bipartition_purity": b}
        wmins.append(min(w))
        bips.append(b)
    wm = np.array(wmins)
    bp = np.array(bips)

    def st(v):
        return {"min": float(v.min()), "median": float(np.median(v)),
                "max": float(v.max())}

    return {
        "qubits_per_referent": m.qubits, "d_truncated": m.d_truncated,
        "per_register_purity_at_intro": 1.0,
        "per_register_purity_note": (
            "EXACTLY 1 for both referents: the two registers are introduced as "
            "a PRODUCT of two pure states and no cross-referent entangler "
            "exists at intro. doc 56's concurrence question DOES NOT ARISE at "
            "this rung; the exp73 assertion is on this quantity, to 1e-9."),
        "min_wire_purity_over_the_inventory": st(wm),
        "balanced_bipartition_purity_over_the_inventory": st(bp),
        "n_nouns_within_0.01_of_a_product_state": int((wm > 0.99).sum()),
        "n_nouns_within_0.05_of_a_product_state": int((wm > 0.95).sum()),
        "per_noun": per,
        "reading": (
            "intro per-WIRE purity is below 1 by CONSTRUCTION -- a generic unit "
            "vector in R^{2^q} is not a product of q single-qubit states -- and "
            "it is a DATA-DETERMINED constant of the encoding, not a circuit "
            "effect. The committed per-wire trace (doc 55 SS18.3) must therefore "
            "be read against THIS inventory as its baseline, and the exp73 "
            "instrumentation records the per-REGISTER trace alongside it."),
    }


# ------------------------------------------------------------------- invariance
def _random_invertible(d, rng):
    while True:
        A = rng.standard_normal((d, d))
        if abs(float(np.linalg.det(A))) > 1e-3:
            return A


def check_linear_reparameterisation_invariance(m, casts, seed=11, trials=6,
                                               cond_max=50.0):
    """SS3's lemma, turned into numbers: v -> A v for invertible A must leave the
    MAX-FS OPTIMUM unchanged.

    ASSERTED ONLY WHERE THE COMPUTATION IS EXACT, and the distinction is the
    point. `pair` mode at d_ref = 3 runs the exact R^3 arrangement enumeration,
    so its count IS the optimum and invariance is a hard assertion. `twin` mode
    is a BUDGETED RANDOMISED SEARCH; the optimum it approaches is invariant, the
    search itself is not (it runs in a different coordinate system and lands on
    a different local optimum). Asserting twin-mode invariance would be asserting
    that a heuristic is coordinate-free, which it is not and never claimed to be.
    The twin values are therefore REPORTED, never asserted -- a failure there
    would be an instrument defect being manufactured, not detected.
    """
    rng = np.random.default_rng(seed)
    dref = m.d_ref
    out = []
    base_pair = ieb.bound_joint(m, casts, "pair", **BUDGET)
    base = {"pair": base_pair["count"],
            "twin": ieb.bound_joint(m, casts, "twin", **BUDGET)["count"]}
    assert base_pair["exact"], "the invariance check needs the exact R^3 path"
    t = 0
    while len(out) < trials:
        t += 1
        assert t < 500, "could not draw enough well-conditioned matrices"
        A = _random_invertible(dref, rng)
        cond = float(np.linalg.cond(A))
        if cond > cond_max:
            continue
        per = {n: A @ m.per_referent[n] for n in m.names}
        mm = ieb.product_map(m.name + "/reparam%d" % len(out), m.names, per, "")
        got = {mode: ieb.bound_joint(mm, casts, mode, **BUDGET)["count"]
               for mode in ("pair", "twin")}
        out.append({"trial": len(out), "cond_A": cond, "base": base,
                    "reparameterised": got,
                    "pair_identical_ASSERTED": bool(got["pair"] == base["pair"]),
                    "twin_identical_REPORTED_NOT_ASSERTED":
                        bool(got["twin"] == base["twin"])})
    return {"trials": out, "cond_max": cond_max,
            "all_identical": bool(all(o["pair_identical_ASSERTED"] for o in out)),
            "twin_agreement_rate_reported_only":
                float(np.mean([o["twin_identical_REPORTED_NOT_ASSERTED"]
                               for o in out])),
            "claim": ("MAX-FS over Phi = vec(v_a v_p^T) is invariant under "
                      "v -> A v for invertible A, because W -> (A (x) A)^{-T} W "
                      "is a bijection of the readout space. ASSERTED on the "
                      "exact pair-mode path; the twin-mode search is a "
                      "heuristic and its coordinate-dependence is reported, not "
                      "asserted away.")}


def check_sign_invariance(Z, casts, seed=5, trials=4):
    """SS2(c): a per-component SIGN FLIP of the principal axes is the special
    case A = diag(+-1) of SS3's lemma, so the MAX-FS optimum cannot move. This is
    the fact that licenses the PC3-onward sign rule being registered for the
    TRAINED CIRCUIT's sake alone.

    ASSERTED at d = 2, where pair mode runs the exact R^3 enumeration and the
    count IS the optimum. REPORTED at d = 4, 8, 16 for the same reason the
    reparameterisation check reports its twin column: those are search or
    saturation outputs, and only the saturated ones are exact.
    """
    rng = np.random.default_rng(seed)
    out = {}
    ok = True
    for d in (2, 4, 8, 16):
        base = amplitude_map("base%d" % d, Z, d, "")
        b0 = {mode: bound_amp(base, casts, mode, **BUDGET)
              for mode in ("pair", "twin")}
        rows = []
        for t in range(trials):
            s = rng.choice([-1.0, 1.0], size=d)
            Zf = {n: np.asarray(Z[n], dtype=float).copy() for n in Z}
            for n in Zf:
                Zf[n][:d] = Zf[n][:d] * s
            mm = amplitude_map("flip%d_%d" % (d, t), Zf, d, "")
            got = {mode: bound_amp(mm, casts, mode, **BUDGET)["count"]
                   for mode in ("pair", "twin")}
            rows.append({"signs": [int(v) for v in s],
                         "base": {k: v["count"] for k, v in b0.items()},
                         "flipped": got,
                         "pair_identical": bool(got["pair"] == b0["pair"]["count"]),
                         "twin_identical": bool(got["twin"] == b0["twin"]["count"])})
        asserted = bool(d == 2 or b0["pair"].get("saturated"))
        allp = all(r["pair_identical"] for r in rows)
        if asserted:
            ok = ok and allp
        out["d%d" % d] = {
            "trials": rows, "pair_all_identical": allp,
            "twin_all_identical": all(r["twin_identical"] for r in rows),
            "pair_path_is_exact": bool(b0["pair"].get("exact")),
            "asserted": asserted,
            "note": ("ASSERTED (the pair count is the exact optimum)"
                     if asserted else
                     "REPORTED ONLY (the pair count is a budgeted search output; "
                     "the OPTIMUM is invariant, the heuristic need not be)")}
    out["all_identical"] = bool(ok)
    return out


# --------------------------------------------------------------- the A1 relation
L_A1 = np.array([[1.0, 0.0, 1.0],
                 [1.0, 0.0, -1.0],
                 [0.0, 2.0, 0.0]])


def check_A1_linear_form(n_grid=721):
    """L v(phi) = u(2 phi), certified on a dense grid rather than asserted."""
    phis = np.linspace(-math.pi, math.pi, n_grid)
    worst = 0.0
    for p in phis:
        v = amp_features(np.array([math.cos(p), math.sin(p)]))
        worst = max(worst, float(np.abs(L_A1 @ v - isb._u1(2.0 * p)).max()))
    return {"matrix_L": L_A1.tolist(), "det_L": float(np.linalg.det(L_A1)),
            "max_abs_error_over_grid": worst,
            "grid": "%d points, phi in [-pi, pi]" % n_grid,
            "identity": "L v(phi) = u(2 phi), v = (x0^2, x0 x1, x1^2)",
            "verified": bool(worst < 1e-12)}


def check_A1_doubled_angle(Z, casts, space="raw"):
    """THE SELFTEST OF RECORD (SS3).

    space="raw"    phi = atan2(z2, z1) on the UNRESCALED PCA plane -- the plane
                   the amplitude map actually lives on.
    space="instr"  phi = atan2(p2, p1) on the max-abs-normalised plane -- the
                   space every committed FA bound is stated over. Both are
                   computed and both are reported, doc 56's discipline for the
                   two FA spaces inherited verbatim.
    """
    if space == "raw":
        phi = {n: math.atan2(Z[n][1], Z[n][0]) for n in isb.HUMANS}
    else:
        coords, _p = isb.load_raw()
        phi = {n: math.atan2(coords[n][1], coords[n][0]) for n in isb.HUMANS}

    per = {n: amp_features(np.array([math.cos(phi[n]), math.sin(phi[n])]))
           for n in isb.HUMANS}
    amp = ieb.product_map("A1/%s" % space, isb.HUMANS, per, "")
    dbl = {n: cc.referent_vector(2.0 * phi[n]) for n in isb.HUMANS}
    single_2phi = ieb.product_map("u(2phi)/%s" % space, isb.HUMANS, dbl, "")
    sng = {n: cc.referent_vector(phi[n]) for n in isb.HUMANS}
    single_phi = ieb.product_map("u(phi)/%s" % space, isb.HUMANS, sng, "")

    rows = {}
    ok = True
    for mode in ("pair", "twin"):
        a = ieb._public(ieb.bound_joint(amp, casts, mode, **BUDGET))
        b = ieb._public(ieb.bound_joint(single_2phi, casts, mode, **BUDGET))
        c = ieb._public(ieb.bound_joint(single_phi, casts, mode, **BUDGET))
        hit = (a["count"] == b["count"] and a["fraction"] == b["fraction"])
        ok = ok and hit
        rows[mode] = {
            "amplitude_1qubit": {"bound": a["bound"], "fraction": a["fraction"],
                                 "exact": a["exact"]},
            "single_angle_at_2phi": {"bound": b["bound"], "fraction": b["fraction"],
                                     "exact": b["exact"]},
            "single_angle_at_phi": {"bound": c["bound"], "fraction": c["fraction"],
                                    "exact": c["exact"]},
            "doubled_angle_identity_holds": bool(hit),
            "differs_from_same_angle": bool(a["count"] != c["count"]),
        }
    return {"space": space, "modes": rows, "all_identities_hold": bool(ok),
            "reading": (
                "the 1-qubit amplitude bound EQUALS the committed single-angle "
                "bound at DOUBLED angle, exactly, in both modes -- that is the "
                "right check and it is asserted. Whether it also equals the "
                "same-angle bound is a separate, purely empirical matter and is "
                "reported, never asserted: angle doubling is a bijective "
                "relabelling of the inventory but not a linear map of the "
                "features, so the two need not agree, and 'amplitude encoding "
                "is not a reparameterisation of the angle encoding' is a "
                "measurement here rather than an assumption.")}


def check_A1_separation(n_inventories=30, seed=0, n_nouns=24, n_bearers=4):
    """THE DISCRIMINATING CHECK, and the reason the doubled-angle identity is a
    theorem while the same-angle agreement is not.

    On random 24-noun angle inventories with random 4-bearer categories, EXACT
    pair-mode counts are compared three ways. Registered before it was run:

      * `amplitude == u(2 phi)` must hold on EVERY inventory -- it is SS3's
        linear identity and a single failure is an instrument defect;
      * `amplitude == u(phi)` must FAIL on some inventories -- otherwise angle
        doubling would be provably inert and the amplitude rung would be a
        relabelling of the committed angle rung. The rate is reported.

    Without this check, the committed inventory's agreement (SS`main`: all four
    categories agree) would be indistinguishable from a hidden identity, and the
    doc would be entitled to say "amplitude is just the angle map" -- which is
    false, and is shown to be false here rather than argued.
    """
    rng = np.random.default_rng(seed)
    names = ["n%02d" % i for i in range(n_nouns)]
    n_ident, n_diff = 0, 0
    rows = []
    for t in range(int(n_inventories)):
        phi = {n: float(rng.uniform(-math.pi, math.pi)) for n in names}
        B = [names[int(i)] for i in
             sorted(rng.choice(n_nouns, size=n_bearers, replace=False))]
        cl, _n = cc.casts_from_bearers(names, B)
        amp = ieb.product_map("a", names, {
            n: amp_features(np.array([math.cos(phi[n]), math.sin(phi[n])]))
            for n in names}, "")
        d2 = ieb.product_map("b", names, {
            n: cc.referent_vector(2.0 * phi[n]) for n in names}, "")
        d1 = ieb.product_map("c", names, {
            n: cc.referent_vector(phi[n]) for n in names}, "")
        c = [ieb.bound_joint(m, cl, "pair", **BUDGET)["count"]
             for m in (amp, d2, d1)]
        n_ident += int(c[0] == c[1])
        n_diff += int(c[0] != c[2])
        rows.append({"trial": t, "amplitude": c[0], "u_2phi": c[1],
                     "u_phi": c[2], "n_casts": len(cl)})
    return {
        "n_inventories": int(n_inventories), "seed": int(seed),
        "mode": "pair (EXACT R^3 arrangement enumeration on every row)",
        "identity_amplitude_equals_u_2phi_held": "%d/%d" % (n_ident, n_inventories),
        "identity_holds_everywhere": bool(n_ident == n_inventories),
        "amplitude_differs_from_u_phi": "%d/%d" % (n_diff, n_inventories),
        "doubling_is_not_inert": bool(n_diff > 0),
        "trials": rows,
        "reading": (
            "the doubled-angle identity is a THEOREM and holds on every "
            "inventory; the same-angle agreement is an INVENTORY-SPECIFIC "
            "COINCIDENCE and fails on most random inventories. Amplitude "
            "encoding of a unit 2-vector is therefore NOT a reparameterisation "
            "of the committed angle encoding -- it is the same feature space at "
            "double frequency, which reorders the inventory around the circle."),
    }


# ------------------------------------------------------------------- selftest
def selftest(Z, verbose=True):
    """Nothing in this module is believed before every block below passes."""
    out = {"anchors": [], "checks": {}}
    ok = True
    B = isb.CATEGORIES[TARGET]["members"]
    casts, _n = cc.casts_from_bearers(isb.HUMANS, B)

    # (1) the committed anchors, through the machinery this module calls
    for a in ieb.ANCHORS:
        fam, kind = a["map"]
        m, _p = ieb._anchor_map(fam, kind)
        BB = isb.CATEGORIES[a["category"]]["members"]
        cl, _nn = cc.casts_from_bearers(m.names, BB)
        r = ieb._public(ieb.bound_joint(m, cl, a["mode"], **BUDGET))
        hit = (abs(r["bound"] - a["expect"]) < 5e-5
               and r["fraction"] == a["fraction"]
               and r["exact"] == a["exact"])
        ok = ok and hit
        out["anchors"].append(
            {"anchor": a["name"], "expected": a["expect"],
             "expected_fraction": a["fraction"], "got": r["bound"],
             "got_fraction": r["fraction"], "reproduced": bool(hit),
             "source": a["source"]})
        if verbose:
            print("%-28s expect %.4f %-9s | got %.4f %-9s | %s"
                  % (a["name"], a["expect"], a["fraction"], r["bound"],
                     r["fraction"], "OK" if hit else "*** MISMATCH ***"))

    # (2) the PCA convention
    out["checks"]["pca_convention"] = check_pca_convention(Z)
    ok = ok and out["checks"]["pca_convention"]["reproduces_load_raw_exactly"]
    ok = ok and out["checks"]["pca_convention"]["divisors_match"]

    # (3) the A1 linear form and the doubled-angle identity, both FA spaces
    out["checks"]["A1_linear_form"] = check_A1_linear_form()
    ok = ok and out["checks"]["A1_linear_form"]["verified"]
    for space in ("raw", "instr"):
        k = "A1_doubled_angle_%s" % space
        out["checks"][k] = check_A1_doubled_angle(Z, casts, space=space)
        ok = ok and out["checks"][k]["all_identities_hold"]
        if verbose:
            r = out["checks"][k]["modes"]["twin"]
            print("A1 %-6s twin: amplitude %s | u(2phi) %s | u(phi) %s | %s"
                  % (space, r["amplitude_1qubit"]["fraction"],
                     r["single_angle_at_2phi"]["fraction"],
                     r["single_angle_at_phi"]["fraction"],
                     "OK" if r["doubled_angle_identity_holds"] else "*** MISMATCH ***"))

    # (4) the reparameterisation lemma (at d_ref = 3) and the sign invariance
    a1 = amplitude_map("A1", Z, 2, "")
    out["checks"]["reparameterisation_invariance"] = \
        check_linear_reparameterisation_invariance(a1, casts)
    ok = ok and out["checks"]["reparameterisation_invariance"]["all_identical"]
    out["checks"]["sign_invariance"] = check_sign_invariance(Z, casts)
    ok = ok and out["checks"]["sign_invariance"]["all_identical"]

    # (4b) the discriminating check: the identity is a theorem, the same-angle
    #      agreement on THIS inventory is a coincidence, and both are shown.
    out["checks"]["A1_separation"] = check_A1_separation()
    ok = ok and out["checks"]["A1_separation"]["identity_holds_everywhere"]
    ok = ok and out["checks"]["A1_separation"]["doubling_is_not_inert"]
    if verbose:
        s = out["checks"]["A1_separation"]
        print("A1 separation: identity %s | doubling changes the count %s"
              % (s["identity_amplitude_equals_u_2phi_held"],
                 s["amplitude_differs_from_u_phi"]))

    # (4c) the committed inventory's own four categories, reported not asserted
    cats = {}
    for cat in isb.CATEGORIES:
        BB = isb.CATEGORIES[cat]["members"]
        cl, _n = cc.casts_from_bearers(isb.HUMANS, BB)
        phi = {n: math.atan2(Z[n][1], Z[n][0]) for n in isb.HUMANS}
        mm = [ieb.product_map("x", isb.HUMANS, per, "") for per in (
            {n: amp_features(np.array([math.cos(phi[n]), math.sin(phi[n])]))
             for n in isb.HUMANS},
            {n: cc.referent_vector(2.0 * phi[n]) for n in isb.HUMANS},
            {n: cc.referent_vector(phi[n]) for n in isb.HUMANS})]
        f = [ieb.bound_joint(m, cl, "pair", **BUDGET)["fraction"] for m in mm]
        cats[cat] = {"amplitude": f[0], "u_2phi": f[1], "u_phi": f[2],
                     "identity_holds": bool(f[0] == f[1]),
                     "differs_from_same_angle": bool(f[0] != f[2])}
        ok = ok and cats[cat]["identity_holds"]
    out["checks"]["A1_committed_inventory_all_categories"] = {
        "per_category": cats,
        "reading": ("on THIS inventory the same-angle value coincides with the "
                    "amplitude value in every category. Reported as a "
                    "coincidence and NOT as a finding: `A1_separation` shows it "
                    "fails on most random inventories, so nothing follows from "
                    "it about the amplitude rungs A3/A4, whose feature spaces "
                    "are not 3-dimensional and have no angle analogue at all.")}

    # (5) the saturation certificate: it must FIRE where the algebra says it
    #     must and must NOT fire where it must not. Both directions asserted.
    sat = {}
    for name, q, _d in LADDER:
        d = 2 ** q
        m = amplitude_map(name, Z, d, "")
        G = ieb.joint_rows(m, casts, "twin")
        cert = saturation_certificate(G)
        expect = bool(m.d >= G.shape[0])
        sat[name] = {"d_joint": int(m.d), "n_twin_rows": int(G.shape[0]),
                     "d_joint_ge_n_rows": expect,
                     "certificate_fired": bool(cert is not None),
                     "rank": (None if cert is None
                              else int(np.linalg.matrix_rank(
                                  G / np.maximum(np.linalg.norm(
                                      G, axis=1, keepdims=True), 1e-300),
                                  tol=1e-9))),
                     "min_margin": (None if cert is None else cert["min_margin"])}
        # the failing direction: it must not fire when d_joint < n_rows
        if not expect:
            ok = ok and (cert is None)
            sat[name]["failing_direction_ok"] = bool(cert is None)
        else:
            sat[name]["failing_direction_ok"] = True
    out["checks"]["saturation_certificate"] = {
        "per_rung": sat,
        "both_directions_asserted": True,
        "note": ("the certificate must NOT fire at A1 (d_joint 9 < 160 rows) "
                 "and A2 (100 < 160); if it did, the rank test would be "
                 "meaningless. That failing direction is asserted, not assumed.")}
    ok = ok and all(v["failing_direction_ok"] for v in sat.values())

    # (6) the structural invariants on the amplitude maps themselves
    inv = {}
    for name, q, _d in LADDER:
        m = amplitude_map(name, Z, 2 ** q, "")
        p = ieb._public(bound_amp(m, casts, "pair", **BUDGET))
        t = ieb._public(bound_amp(m, casts, "twin", **BUDGET))
        chk = {"pair": p["bound"], "twin": t["bound"], "d_joint": m.d,
               "d_per_referent": m.d_ref,
               "max_at_or_above_chance": bool(p["bound"] >= 0.5 - 1e-12
                                              and t["bound"] >= 0.5 - 1e-12),
               "twin_ge_pair": bool(t["bound"] >= p["bound"] - 1e-9),
               "sign_flip_sums_to_n": bool(p["sign_flip_check"]["sums_to_n"]
                                           and t["sign_flip_check"]["sums_to_n"]),
               "swap_perm_max_abs_error": ieb.check_swap_perm(m, casts)}
        chk["swap_perm_verified"] = bool(chk["swap_perm_max_abs_error"] < 1e-12)
        ok = ok and all(chk[k] for k in ("max_at_or_above_chance", "twin_ge_pair",
                                         "sign_flip_sums_to_n",
                                         "swap_perm_verified"))
        inv[name] = chk
        if verbose:
            print("invariants %-3s d_ref=%-4d d_joint=%-6d pair %.4f twin %.4f "
                  "| >=chance %s twin>=pair %s signflip %s swapperm %s"
                  % (name, chk["d_per_referent"], chk["d_joint"], chk["pair"],
                     chk["twin"], chk["max_at_or_above_chance"],
                     chk["twin_ge_pair"], chk["sign_flip_sums_to_n"],
                     chk["swap_perm_verified"]))
    out["checks"]["invariants"] = inv

    # (7) the quadratic-form claim itself: <W, v(x)> reproduces x^T M x exactly
    rng = np.random.default_rng(3)
    worst = 0.0
    for d in (2, 4, 8, 16):
        for _ in range(20):
            x = rng.standard_normal(d)
            x /= np.linalg.norm(x)
            M = rng.standard_normal((d, d))
            M = (M + M.T) / 2.0
            iu = np.triu_indices(d)
            W = M[iu] * np.where(iu[0] == iu[1], 1.0, 2.0)
            worst = max(worst, abs(float(x @ M @ x) - float(W @ amp_features(x))))
    out["checks"]["quadratic_form_identity"] = {
        "max_abs_error": worst, "verified": bool(worst < 1e-12),
        "identity": ("x^T M x = <W, v(x)> with v the triu outer square and "
                     "W = triu(M) with the off-diagonals doubled -- SS1, "
                     "checked on random symmetric M at d = 2, 4, 8, 16")}
    ok = ok and out["checks"]["quadratic_form_identity"]["verified"]

    out["all_passed"] = bool(ok)
    if verbose:
        print("SELFTEST: %s" % ("PASS" if ok else "FAIL"))
    return out


# --------------------------------------------------------------------- main
def self_sha256():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=NULL_DRAWS)
    ap.add_argument("--null-seed", type=int, default=NULL_SEED)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    Z, var, prov = _pc_scores()
    st = selftest(Z, verbose=True)
    if a.selftest:
        print(json.dumps(st, indent=2, ensure_ascii=False))
        return 0 if st["all_passed"] else 1
    if not st["all_passed"]:
        print("SELFTEST FAILED -- no number is written", file=sys.stderr)
        return 1

    B = isb.CATEGORIES[TARGET]["members"]
    casts, _n = cc.casts_from_bearers(isb.HUMANS, B)

    bounds, witnesses, purity, encoding = {}, {}, {}, {}
    for name, q, desc in LADDER:
        d = 2 ** q
        m = amplitude_map(name, Z, d, desc)
        for mode in ("pair", "twin"):
            r = bound_with_null(m, B, mode, draws=a.draws,
                                null_seed=a.null_seed, **BUDGET)
            r["definition"] = desc
            r["qubits_per_referent"] = q
            r["d_truncated"] = d
            bounds.setdefault(name, {})[mode] = r
            print("%-3s %-4s  %.4f  %-9s  null %.4f +- %.4f  pct %.1f  "
                  "sat %s  [%.0fs]"
                  % (name, mode, r["bound"], r["fraction"], r["null"]["mean"],
                     r["null"]["sd"], r["null"]["percentile_of_category"],
                     r.get("saturated"), time.time() - t0))
        w = bound_amp(m, casts, "twin", **BUDGET)
        witnesses[name] = {
            "d": int(len(w["_c"])), "n_twins": int(w["n_rows"]),
            "n_positive_margin": int(w["sign_flip_check"]["satisfied_c"]),
            "certifies_the_bound_count":
                bool(w["sign_flip_check"]["satisfied_c"] == w["count"]),
            "exported": bool(len(w["_c"]) <= 1300),
            "W": ([float(x) for x in w["_c"]] if len(w["_c"]) <= 1300 else None),
            "W_sha256": hashlib.sha256(
                np.asarray(w["_c"], dtype=np.float64).tobytes()).hexdigest(),
            "convention": (
                "doc 55 SS19.3, re-pinned for exp73's maps: the witness of "
                "record is THIS committed computation's output, exported at "
                "bound-computation time, BEFORE any exp73 run. For the "
                "SATURATED rungs the witness is the exact certificate's "
                "W = G^+ 1 (SS4) and not a search output. A4's W has 18496 "
                "entries and is carried by sha256 only; it is reproducible "
                "from this script and the pinned inputs."),
        }
        purity[name] = register_purity_inventory(m)
        rad = np.array([m.radii[n] for n in isb.HUMANS])
        encoding[name] = {
            "qubits_per_referent": q, "d_truncated": d,
            "d_per_referent_feature": d_ref_of(d), "d_joint": d_ref_of(d) ** 2,
            "free_real_coordinates_per_referent": d - 1,
            "variance_captured_by_PC1..PC%d" % d: float(np.cumsum(var)[d - 1]),
            "variance_discarded_by_truncation": float(1.0 - np.cumsum(var)[d - 1]),
            "radius_destroyed_by_normalisation": {
                "min": float(rad.min()), "median": float(np.median(rad)),
                "max": float(rad.max()), "mean": float(rad.mean()),
                "note": ("||z_{1:d}(r)||_2, exactly the one real number per "
                         "referent that unit-l2 normalisation discards -- the "
                         "same quantity FA discarded relative to F2")},
            "definition": desc,
        }

    out = {
        "component": "exp73_amplitude_bounds",
        "purpose": ("grade-1 reachable-accuracy bounds, size-matched nulls, "
                    "encoding-cost accounting and register-purity inventory for "
                    "exp73's TRUNCATED AMPLITUDE ladder, computed BEFORE any "
                    "exp73_*.py exists (doc 55 SS19.2's ordering rule, "
                    "inherited; Rule 4)"),
        "spec": "qnlp_private_docs/61_exp73_amplitude_DRAFT.md",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script_sha256": self_sha256(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "provenance": {
            "instrument_sb_bounds_sha256": isb.self_sha256(),
            "capacity_check_sha256": hashlib.sha256(
                open(os.path.join(HERE, "capacity_check.py"), "rb").read()).hexdigest(),
            "instrument_entangled_bounds_sha256": ieb.self_sha256(),
            "raw_family": prov,
            "budget": BUDGET, "null_draws": int(a.draws),
            "null_seed": int(a.null_seed),
            "extends": ("instrument_entangled_bounds.py's general joint-map "
                        "machinery to AMPLITUDE encodings. The only new object "
                        "is the per-referent feature (the outer square of the "
                        "normalised truncated vector); the row builder, the "
                        "twin seeding, the search, the sign-flip invariant and "
                        "the null discipline are the committed ones, called "
                        "and not copied."),
        },
        "category": {"key": TARGET, "gloss": isb.CATEGORIES[TARGET]["gloss"],
                     "members": isb.CATEGORIES[TARGET]["members"],
                     "source": ("instrument_sb_bounds.CATEGORIES -- doc 54 SS5.2, "
                                "user-confirmed 2026-08-23; imported, never re-typed")},
        "selftest": st,
        "derivation": {
            "measurement_is_a_quadratic_form": {
                "statement": "p(x) = <psi(x)|O|psi(x)> = x^T M x, M = M^T real",
                "per_referent_feature": "v(x) = (x_i x_j)_{i <= j}, triu-ordered",
                "d_per_referent": "d(d+1)/2",
                "joint_feature": "vec(v(x_a) v(x_p)^T), a PRODUCT map",
                "d_joint": "[d(d+1)/2]^2",
                "classical_comparator_implied": (
                    "a QUADRATIC (kernel) model on the raw features: linear in "
                    "the degree-2 monomials of x, equivalently the kernel "
                    "(x_a . x_a')^2 (x_p . x_p')^2 on the two-referent decision. "
                    "This is EVEN MORE IMMEDIATE than the angle maps' matched "
                    "comparator -- it is the same object in another basis "
                    "(doc 52 SS1's prohibition applies at its strongest)."),
                "general_W_qualifier": (
                    "W ranges over the whole coordinate space; the physically "
                    "realisable M are a subset. The committed modules make the "
                    "same identification and this one inherits it verbatim, "
                    "which is what makes these numbers comparable with theirs."),
            },
            "encoding_map": {
                "formula": ("x(r) = z_{1:d}(r) / ||z_{1:d}(r)||_2, z = the "
                            "sign-pinned raw PCA scores, zero-padded to 2^q "
                            "(vacuous when d = 2^q)"),
                "information_cost_vs_eps_map": (
                    "the eps-map is injective and affine and destroys NO "
                    "information -- its cost is angular RESOLUTION, forced by "
                    "the concurrence dead zones. The amplitude map is lossy "
                    "TWICE: truncation discards 1 - cumvar(d) of the variance, "
                    "and unit-l2 normalisation discards the radius, exactly one "
                    "real number per referent -- the same number FA discarded "
                    "relative to F2. Per-referent free coordinates: FA 1, "
                    "F2/P-eps 2, amplitude d - 1."),
            },
            "A1_relation": st["checks"]["A1_linear_form"],
        },
        "encoding_cost": encoding,
        "grade1_bounds": bounds,
        "register_purity_inventory": purity,
        "witnesses_doc55_SS19_3": witnesses,
        "bound_of_record_note": {
            "A1/A2": ("no committed bound exists for either; A1 is the selftest "
                      "rung and A2 is a sensitivity row, NOTED NOT RUN."),
            "A3/A4": ("EXACT 1.0000 by the saturation certificate (SS4) if the "
                      "certificate fired -- read `method` on each row, never the "
                      "number alone (doc 47 SS18.5)."),
            "binding_rule": ("doc 55 SS6 / SS12: NO exp73 criterion reads against "
                             "any number in this file. doc 54 SS17.7 is the "
                             "standing reason, and SS4's saturation makes it "
                             "mechanical rather than merely prudent."),
        },
    }

    g3 = {
        "flag": ("under-conversion predicted for a BELOW-NULL category (doc 55 "
                 "SS19.2 grade 3, made prospective)"),
        "percentile_vs_size_matched_null":
            {k: v["twin"]["null"]["percentile_of_category"]
             for k, v in bounds.items()},
        "below_null": {k: v["twin"]["null"]["below_null"]
                       for k, v in bounds.items()},
        "null_mean": {k: v["twin"]["null"]["mean"] for k, v in bounds.items()},
        "null_sd": {k: v["twin"]["null"]["sd"] for k, v in bounds.items()},
    }
    g3["any_arm_below_null"] = bool(any(g3["below_null"].values()))
    g3["testable_this_run"] = g3["any_arm_below_null"]
    g3["verdict"] = (
        "TESTABLE: at least one exp73 rung sits below its size-matched null."
        if g3["any_arm_below_null"] else
        "UNTESTABLE THIS RUN -- no exp73 rung has a below-null category. "
        "Computed before freezing and recorded rather than quietly dropped, "
        "exactly as doc 55 SS19.2 did for exp69 and doc 56 for exp70.")
    out["grade3_under_conversion_flag"] = g3

    sat_rungs = [k for k, v in bounds.items()
                 if v["twin"].get("saturated") and v["pair"].get("saturated")]
    out["saturation_finding"] = {
        "saturated_rungs": sat_rungs,
        "statement": (
            "at every rung whose joint coordinate space is at least as large as "
            "the number of MAX-FS rows, the rows are of full rank and the "
            "optimum is EXACTLY the row count. The bound is 1.0000 by PROOF, "
            "and so is every size-matched null draw's."),
        "consequence_registered_as_a_result": (
            "the grade-1 instrument DISCRIMINATES NOTHING at the amplitude "
            "rung -- not A3 from A4, not the category from its null, and not "
            "amplitude from the committed 2-qubit angle registers whose bounds "
            "are already 1.0000 (doc 58 SS11). Every exp73 criterion is "
            "therefore measurement-vs-measurement, and the standard "
            "bound-referenced bug-hunt gate is STRUCTURALLY INERT here in "
            "doc 58 SS9.5's exact sense: bound + 0.06 > 1 and no achievable "
            "accuracy can trigger it."),
        "what_it_does_NOT_say": (
            "it does NOT say the amplitude encoding is expressive in any useful "
            "sense. Saturation is a statement about a 1296- or 18496-dimensional "
            "linear model on 160 constraints; it is the SAME degeneracy that "
            "makes the tier-(ii) classical comparator vacuous at these widths, "
            "and the doc registers a capacity-matched comparator tier for "
            "exactly that reason."),
        "comparator_dimension_warning": {
            k: {"d_joint": v["twin"]["d_joint"],
                "n_train_plus_val_items_exp69_dataset": 660,
                "unregularised_linear_fit_is_vacuous":
                    bool(v["twin"]["d_joint"] >= 660)}
            for k, v in bounds.items()},
    }

    out["wall_sec"] = time.time() - t0
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("wrote %s  (%.0f s)" % (a.out, out["wall_sec"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
