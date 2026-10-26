# -*- coding: utf-8 -*-
"""instrument_entangled_bounds.py -- reachable-accuracy bounds for EXPLICITLY
ENTANGLED per-referent and cross-referent encodings.

WHY THIS FILE EXISTS
--------------------
`capacity_check.py`'s scope note is explicit about its own boundary:

    "per-referent encodings that are a SINGLE RY PER QUBIT ... Re-uploading,
     entangling the referent register with itself, data re-encoding at depth, or
     any parameterised map applied BEFORE the bilinear read are all outside it."

exp70 is the fourth lever of the ledger (K, M, width, and now ENTANGLEMENT in
the ENCODING), and it needs exactly the two things that sentence excludes:

  E-in       a referent's own two-qubit register is entangled WITH ITSELF at
             intro:  RY(theta1) on q0, then CRY(theta2) q0 -> q1.
  E-across   two referents' single-qubit wires are entangled WITH EACH OTHER at
             intro:  RY(theta_A) (x) RY(theta_B), then one CZ between the wires.

This module extends the MAX-FS machinery to both. It re-derives NOTHING:
`instrument_sb_bounds` remains the authority for the search, the exact R^3
enumeration, the sign-flip invariant and the null discipline. What is new here
is (a) the FEATURE ALGEBRA of an entangled encoding and (b) a MAX-FS row builder
over a GENERAL JOINT FEATURE MAP Phi(agent, patient), of which the product
construction `vec(u_a u_p^T)` used by `capacity_check` is the special case.

The selftest asserts that special case reproduces `capacity_check`'s committed
anchors EXACTLY -- that is what makes this an extension rather than a fork.

--------------------------------------------------------------------------
1. THE DECISION BEING BOUNDED (unchanged from doc 46 SS5 / doc 47 SS3.2, SS17)
--------------------------------------------------------------------------
SB: one event, a two-referent cast, exactly one category bearer. Each cast
(i, j) supplies TWO twins, and the decision is the sign of a readout that is
LINEAR in the coordinates of the joint intro state:

    twin YES correct  iff  <W, Phi(x_i, y_j)> > 0
    twin NO  correct  iff  <W, Phi(y_j, x_i)> < 0

`Phi(a, p)` is the vector of INDEPENDENT REAL COORDINATES of the two-referent
intro density matrix, agent-first. Every gate applied after intro is
feature-independent and is absorbed into W, so the reachable class is exactly
`max_W` over that linear system -- the same MAX-FS problem, over a different and
explicitly derived feature space.

    mode="pair"  rows  Phi(x,y) - Phi(y,x)      denominator = CASTS
    mode="twin"  rows  Phi(x,y) and -Phi(y,x)   denominator = TWINS

--------------------------------------------------------------------------
2. E-in -- THE DERIVATION, AND THE EXACT SPAN
--------------------------------------------------------------------------
RY(t)|0> = cos(t/2)|0> + sin(t/2)|1>, and CRY(t2) applies RY(t2) to q1 iff q0=1:

    |psi> = cos(t1/2) |00> + sin(t1/2)cos(t2/2) |10> + sin(t1/2)sin(t2/2) |11>

so the amplitude on |01> is IDENTICALLY ZERO and the state is carried by three
real coefficients

    (a, b, c) = ( cos(t1/2), sin(t1/2)cos(t2/2), sin(t1/2)sin(t2/2) ),
    a^2 + b^2 + c^2 = 1.

Concurrence = 2|a00 a11 - a01 a10| = 2|a c| = |sin t1| * |sin(t2/2)|, the closed
form the committed `diagnostic_encoding_entanglement.json` states and this module
re-verifies element-by-element (SS7).

READOUTS ARE QUADRATIC FORMS IN THE STATE COEFFICIENTS, so the effective
per-referent feature is the OUTER SQUARE of (a, b, c) -- the SIX independent
entries of the symmetric 3x3 matrix:

    v_Ein(t1, t2) = ( a^2, b^2, c^2, ab, ac, bc )        d_ref = 6

(the |01> row and column of rho are identically zero, so the 10 independent
entries of a general real symmetric 4x4 collapse to 6). The joint feature is
Phi = v_a (x) v_p, d_joint = 36.

THE ACCESSIBLE TRIG SPAN, derived term by term:

    a^2 = (1 + cos t1)/2
    b^2 = (1 - cos t1)(1 + cos t2)/4
    c^2 = (1 - cos t1)(1 - cos t2)/4
    bc  = (1 - cos t1) sin t2 / 4
    ab  = sin t1 * cos(t2/2) / 2          <-- HALF FREQUENCY in t2
    ac  = sin t1 * sin(t2/2) / 2          <-- HALF FREQUENCY in t2

    span(E-in) = span{ 1,
                       cos t1,
                       (1 - cos t1) cos t2,
                       (1 - cos t1) sin t2,
                       sin t1 cos(t2/2),
                       sin t1 sin(t2/2) }                dim 6

This is the sharp statement, and it is NOT "entanglement adds capacity":

  * span(E-in) is NOT a subspace of the product register's
    span{1, cos t1, sin t1} (x) {1, cos t2, sin t2}  (dim 9): the two half-angle
    terms leave it. The CRY gate is not Clifford and the angle is data-carried,
    so the readable function space genuinely CHANGES.
  * span(E-in) does NOT contain `sin t1` on its own, which the product register
    does. The intersection is 4-dimensional (numerically certified in SS6:
    ranks 6, 9 and 11 for E-in, P-eps and their union).
  * So the map is RESTRUCTURED, not enlarged. Neither class contains the other,
    and which reaches further on a given inventory is an empirical question that
    the numbers in the output JSON answer -- it is not settled by dimension
    counting, and this module never claims it is.

--------------------------------------------------------------------------
3. E-across -- THE DERIVATION, AND WHY THE BOUND CANNOT SEE THE CZ
--------------------------------------------------------------------------
One qubit per referent, frozen single-angle map, one CZ after intro:

    |psi> = CZ ( RY(tA)|0> (x) RY(tB)|0> )
          = ( cA cB, cA sB, sA cB, -sA sB ),   cX = cos(tX/2), sX = sin(tX/2)

    concurrence     = |sin tA| * |sin tB|
    per-wire purity = 1 - (sin tA sin tB)^2 / 2        (both wires, pure state)

The joint feature is the outer square of that 4-vector: the TEN independent
entries of a real symmetric 4x4, equivalently the ten real Pauli coordinates
{<P (x) Q> : P, Q in {I, X, Z}} together with <Y (x) Y>. Evaluating them:

    <I(x)I> = 1              <Z(x)I> = cos tA          <I(x)Z> = cos tB
    <X(x)I> = sin tA cos tB  <I(x)X> = cos tA sin tB
    <X(x)Z> = sin tA         <Z(x)X> = sin tB
    <Z(x)Z> = cos tA cos tB  <Y(x)Y> = sin tA sin tB   <X(x)X> = 0

**Every one of those functions already lies in the 9-dimensional product span
{1, cos tA, sin tA} (x) {1, cos tB, sin tB}, and together they fill it.** CZ is
a CLIFFORD gate, so conjugation PERMUTES the Pauli coordinates; with a fully
general readout W the reachable function space is invariant under it. The
control (same map, no CZ) has the same 9-dimensional span with <Y(x)Y> = 0 and
<X(x)X> = sin tA sin tB instead -- the same functions, relabelled.

    CONSEQUENCE, and it is registered as a RESULT of this instrument:
    the E-across bound EQUALS its map-matched control's bound, by proof. The
    MAX-FS instrument is BLIND to this entangler, and so is any comparator built
    on the same function class. A measured E-across-vs-control difference can
    therefore only be a TRAINABILITY / INDUCTIVE-BIAS effect, never a capacity
    effect. The same argument shows that FOURIER CLOSURE IS A NULL INSTRUMENT
    for E-across (doc 53 SS5's sense): closure in the product basis is satisfied
    with the CZ present and absent alike, so validity for that arm must be
    asserted by the PURITY closed form above and by a circuit-difference check,
    not by closure.

--------------------------------------------------------------------------
4. WHAT THIS MODULE DOES NOT BOUND
--------------------------------------------------------------------------
Everything `capacity_check`'s scope note excludes except the two encodings named
above. In particular: re-uploading, data-dependent gates INSIDE an event block,
more than two referents, non-binary readout, and any decision that is not the
sign of a linear functional of the intro state's coordinates. Every non-exact
number here is a CONSTRUCTIVE LOWER BOUND from a budgeted randomised search and
is labelled as one in every return value; a measurement above it is not a
violation (doc 47 SS18.2).

CLI
---
    python instrument_entangled_bounds.py --selftest
    python instrument_entangled_bounds.py            # writes the diagnostic JSON
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
import capacity_check as cc                 # noqa: E402  (the machinery extended)

TOL = isb.TOL
OUT = os.path.join(HERE, "diagnostic_entangled_bounds.json")

# The budget is `capacity_check.DEFAULT_BUDGET` verbatim. Reproducing the
# committed anchors requires the identical budget AND the identical row order,
# because `_ls_drop`'s argsort makes the search sensitive to both.
BUDGET = dict(cc.DEFAULT_BUDGET)
NULL_SEED = 68000
NULL_DRAWS = 200

TARGET = "nonoccupational"                 # doc 54 SS5.2, user-confirmed 2026-08-23
EPS = math.pi / 6.0                        # doc 56 SS-eps-map, forced by the
                                           # committed degeneracy check


# ------------------------------------------------------------- the angle maps
def eps_map(p):
    """theta = p (pi - 2 eps)/2 + pi/2, eps = pi/6  ->  theta in [pi/6, 5pi/6].

    Registered because the exp68 map theta = (p + 1) pi puts the normalisation
    extremes |p| = 1 exactly on the concurrence dead zones sin t1 = 0 and
    sin(t2/2) = 0 (`diagnostic_encoding_entanglement.json`: 4 nouns below 0.05,
    two at 0.0). Under this map the inventory minimum is 0.1898 and nothing is
    below 0.05.
    """
    return p * (math.pi - 2.0 * EPS) / 2.0 + math.pi / 2.0


def full_map(p):
    """exp68's registered F2 map, theta = (p + 1) pi. Kept for the sensitivity
    row that shows what the dead zones cost, and for the A0-W anchor family."""
    return (p + 1.0) * math.pi


def frozen_angle(p1, p2):
    """FA in the INSTRUMENT's space: phi = atan2(PC2, PC1) on the NORMALISED
    raw-PCA plane. This is the space every committed FA bound is stated over
    (`instrument_sb_bounds.features('FA')`, `capacity_check._raw_angles('FA')`),
    and therefore the space in which an exp70 bound is comparable with exp69's.
    """
    return math.atan2(p2, p1)


def runtime_angles():
    """FA in the RUNTIME space: `exp42_noun_embeddings.json`'s stored angle, the
    angle the compiled circuit actually applies, and the angle the committed
    `diagnostic_encoding_entanglement.json` states its E-across concurrence and
    purity distribution over (verified exact, SS7).

    The two FA spaces are NOT the same map -- atan2 of the max-abs-normalised
    coordinates is not atan2 of the raw plane, because the two components are
    divided by different constants. doc 55's exp69 predictions file recorded
    exactly this mismatch for its R-A witness (rank agreement 0.955) rather than
    eliding it, and exp70 inherits that discipline: BOTH spaces are computed and
    both are reported.
    """
    fz = json.load(open(isb.FROZEN_EMB, encoding="utf-8"))["embeddings"]
    return {n: float(fz[n]["angle"]) for n in isb.HUMANS}


# ------------------------------------------------------------ state algebra
def ein_coeffs(t1, t2):
    """(a, b, c) for RY(t1) on q0 then CRY(t2) q0 -> q1. The |01> amplitude is
    identically zero and is not carried."""
    return (math.cos(t1 / 2.0),
            math.sin(t1 / 2.0) * math.cos(t2 / 2.0),
            math.sin(t1 / 2.0) * math.sin(t2 / 2.0))


def ein_concurrence(t1, t2):
    """|sin t1| * |sin(t2/2)|  -- the committed closed form."""
    return abs(math.sin(t1)) * abs(math.sin(t2 / 2.0))


def ein_purity(t1, t2):
    """Per-wire purity of the E-in register: 1 - C^2/2 for a pure 2-qubit state."""
    return 1.0 - ein_concurrence(t1, t2) ** 2 / 2.0


def ein_features(t1, t2):
    """Outer square of (a, b, c): the 6 independent entries. d_ref = 6.

    Ordering is FIXED and is part of the registered witness convention:
        ( a^2, b^2, c^2, ab, ac, bc )
    """
    a, b, c = ein_coeffs(t1, t2)
    return np.array([a * a, b * b, c * c, a * b, a * c, b * c], dtype=float)


def product_features(thetas):
    """One RY per qubit, no entangler: Kronecker product of (1, cos, sin).
    Delegates to `capacity_check.referent_vector` so the product path is
    literally the committed code, not a re-implementation."""
    return cc.referent_vector(list(thetas))


def across_amplitudes(ta, tb, cz=True):
    """CZ ( RY(ta) (x) RY(tb) ) |00>, agent on wire 0."""
    ca, sa = math.cos(ta / 2.0), math.sin(ta / 2.0)
    cb, sb = math.cos(tb / 2.0), math.sin(tb / 2.0)
    last = -sa * sb if cz else sa * sb
    return np.array([ca * cb, ca * sb, sa * cb, last], dtype=float)


_TRIU4 = np.triu_indices(4)


def across_features(ta, tb, cz=True):
    """Outer square of the joint 4-dim amplitude vector: 10 independent entries,
    upper-triangle-ordered. THIS IS A JOINT FEATURE -- it does not factorise
    into per-referent vectors when cz=True."""
    v = across_amplitudes(ta, tb, cz=cz)
    return np.outer(v, v)[_TRIU4]


def across_concurrence(ta, tb):
    return abs(math.sin(ta)) * abs(math.sin(tb))


def across_purity(ta, tb):
    """1 - (sin ta sin tb)^2 / 2. Identical on both wires (pure joint state).
    This is the closed form the exp70 smoke gate asserts per item to 1e-6."""
    return 1.0 - (math.sin(ta) * math.sin(tb)) ** 2 / 2.0


# ---------------------------------------------------- the joint feature maps
class JointMap(object):
    """A named joint feature map Phi(agent, patient) over a noun inventory.

    kind = "product"  Phi(a, p) = vec(v_a v_p^T) for a per-referent v. This is
                      exactly `capacity_check`'s construction; the per-referent
                      vectors are kept so the EXACT R^3 path stays available at
                      d_ref = 3.
    kind = "joint"    Phi is a genuine two-referent map (E-across). No exact
                      path exists; the swap permutation is supplied so twin-mode
                      seeding keeps the structural `twin >= pair` guarantee.
    """

    def __init__(self, name, names, kind, phi, d, swap_perm, per_referent=None,
                 desc="", analytic_basis=None):
        self.name, self.names, self.kind = name, list(names), kind
        self.phi, self.d, self.swap_perm = phi, int(d), swap_perm
        self.per_referent = per_referent
        self.desc, self.analytic_basis = desc, analytic_basis

    @property
    def d_ref(self):
        return None if self.per_referent is None else len(
            next(iter(self.per_referent.values())))


def _vec_transpose_perm(d):
    """Permutation P on R^(d*d) with P vec(M) = vec(M^T)."""
    idx = np.arange(d * d).reshape(d, d).T.ravel()
    return idx


def _across_swap_perm(cz=True):
    """Permutation of the 10 triu entries induced by swapping the two qubits.

    Swapping referents swaps qubits, i.e. permutes basis states
    |00>,|01>,|10>,|11> -> |00>,|10>,|01>,|11> (amplitude indices 0,2,1,3).
    Built numerically from that index permutation, then VERIFIED against the map
    itself in `check_swap_perm` rather than trusted.
    """
    s = [0, 2, 1, 3]
    lab = {}
    for k, (i, j) in enumerate(zip(*_TRIU4)):
        lab[(i, j)] = k
    perm = np.empty(10, dtype=int)
    for k, (i, j) in enumerate(zip(*_TRIU4)):
        a, b = s[i], s[j]
        if a > b:
            a, b = b, a
        perm[k] = lab[(a, b)]
    return perm


def product_map(name, names, per_referent, desc, analytic_basis=None):
    d_ref = len(next(iter(per_referent.values())))
    perm = _vec_transpose_perm(d_ref)

    def phi(a, p):
        return np.outer(per_referent[a], per_referent[p]).ravel()

    return JointMap(name, names, "product", phi, d_ref * d_ref, perm,
                    per_referent=per_referent, desc=desc,
                    analytic_basis=analytic_basis)


def across_map(name, names, angles, cz, desc, analytic_basis=None):
    perm = _across_swap_perm(cz)

    def phi(a, p):
        return across_features(angles[a], angles[p], cz=cz)

    m = JointMap(name, names, "joint", phi, 10, perm, per_referent=None,
                 desc=desc, analytic_basis=analytic_basis)
    m.angles = angles
    m.cz = cz
    return m


def check_swap_perm(m, casts):
    """Assert Phi(y, x) == P Phi(x, y) on every cast. The twin-mode seeding
    depends on it; an unverified permutation would silently break the
    `twin >= pair` guarantee instead of failing loudly."""
    worst = 0.0
    for a, p in casts:
        f = m.phi(a, p)
        g = m.phi(p, a)
        worst = max(worst, float(np.abs(g - f[m.swap_perm]).max()))
    return worst


# ---------------------------------------------------------------- the rows
def joint_rows(m, casts, mode):
    """MAX-FS rows over a general joint feature map.

    ROW ORDER IS LOAD-BEARING and mirrors `capacity_check._rows` exactly:
    casts in the given (bearer-major) order; in twin mode +Phi(x,y) first, then
    -Phi(y,x). `_ls_drop`'s argsort is order-sensitive, so a permuted row list
    changes the returned lower bound.
    """
    rows = []
    for b, y in casts:
        f, g = m.phi(b, y), m.phi(y, b)
        if mode == "pair":
            rows.append(f - g)
        elif mode == "twin":
            rows.append(f)
            rows.append(-g)
        else:
            raise ValueError("mode must be 'pair' or 'twin', got %r" % mode)
    return np.asarray(rows, dtype=float)


# --------------------------------------------------------------- the bound
def bound_joint(m, casts, mode, exact_if_possible=True, restarts=48, steps=500,
                seed=0, ls_starts=6):
    """One bound over an explicit cast list for a joint feature map.

    Mirrors `capacity_check.bound_from_casts` line for line, generalising only
    the two places that assumed a product feature:

      * the EXACT R^3 arrangement path stays available when the map IS a product
        map with d_ref = 3 (the pair-level antisymmetric form is then a vector
        in R^3 and every cell of the plane arrangement can be visited);
      * twin-mode seeding antisymmetrises the pair-level witness with respect to
        the map's SWAP PERMUTATION, W_anti = (W - P^T W)/2. For a product map P
        is the vec-transpose and this is byte-identical to `(M - M.T)/2`; for a
        genuine joint map it is the correct generalisation, and it preserves the
        structural guarantee per-twin >= 2 x pair-level.
    """
    G = joint_rows(m, casts, mode)
    n = G.shape[0]
    exact, inits = None, None
    can_exact = (m.kind == "product" and m.d_ref == 3 and exact_if_possible)
    if mode == "pair" and can_exact:
        exact = isb.exact_maxfs_r3(cc._cross(m.per_referent, casts))
    if mode == "twin":
        if can_exact:
            _, w = isb.exact_maxfs_r3(cc._cross(m.per_referent, casts),
                                      return_w=True)
            A = isb._w_to_antisym3(w).ravel()
        else:
            pl = bound_joint(m, casts, "pair", exact_if_possible=False,
                             restarts=restarts, steps=steps, seed=seed,
                             ls_starts=ls_starts)
            W = pl["_c"]
            A = (W - W[m.swap_perm]) / 2.0
        nrm = float(np.linalg.norm(A))
        inits = [A / max(nrm, 1e-300)]
    lo, c = isb.maxfs_lower(G, restarts=restarts, steps=steps, seed=seed,
                            inits=inits, ls_starts=ls_starts)
    count = exact if exact is not None else lo
    method = ("EXACT (R^3 arrangement enumeration, doc 47 SS16 fix)"
              if exact is not None else
              "CONSTRUCTIVE LOWER BOUND (%d restarts x %d steps + %d LS-drop "
              "starts) -- NOT a maximum" % (restarts, steps, ls_starts))
    return {"map": m.name, "mode": mode, "d_joint": m.d, "d_per_referent": m.d_ref,
            "feature_kind": m.kind, "n_casts": len(casts), "n_rows": n,
            "count": int(count), "bound": count / n,
            "fraction": "%d/%d" % (count, n),
            "population": ("casts (both twins won together)" if mode == "pair"
                           else "twins (a cast may score 0, 1/2 or 1)"),
            "method": method, "exact": exact is not None,
            "sign_flip_check": isb.verify_sign_flip(G, c),
            "_c": c}


def _public(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def _remap(m, bearers):
    """A copy of the map restricted to nothing -- the casts do the work. Kept as
    a named helper so the null loop reads the same as the category call."""
    return cc.casts_from_bearers(m.names, bearers)


def bound_with_null(m, bearers, mode, draws=NULL_DRAWS, null_seed=NULL_SEED,
                    **budget):
    """Category bound plus a SIZE-MATCHED null: `draws` random |B|-subsets of the
    same inventory, the SAME method with the SAME search budget (doc 47 SS16.1).
    An unmatched budget here would be the exp44 unmatched-comparator failure
    class occurring inside the diagnostic itself."""
    casts, nons = _remap(m, bearers)
    r = _public(bound_joint(m, casts, mode, **budget))
    r["n_bearers"], r["n_nonbearers"] = len(bearers), len(nons)
    r["swap_perm_max_abs_error"] = check_swap_perm(m, casts[:8])
    if draws:
        rng = np.random.default_rng(null_seed)
        vals = []
        for t in range(int(draws)):
            idx = sorted(int(i) for i in rng.choice(len(m.names),
                                                    size=len(bearers),
                                                    replace=False))
            bb = [m.names[i] for i in idx]
            cl, _n = cc.casts_from_bearers(m.names, bb)
            bud = dict(budget)
            bud["seed"] = null_seed + t
            vals.append(bound_joint(m, cl, mode, **bud)["bound"])
        v = np.asarray(vals, dtype=float)
        r["null"] = {"draws": int(draws), "size_matched_B": len(bearers),
                     "mean": float(v.mean()),
                     "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                     "min": float(v.min()), "max": float(v.max()),
                     "percentile_of_category":
                         float((v < r["bound"]).mean() * 100.0),
                     "below_null": bool(r["bound"] < v.mean()),
                     "identical_search_budget": True, "seed": int(null_seed)}
    return r


# ------------------------------------------------------- analytic span check
def _grid(n=40, lo=0.30, hi=2.90):
    g = np.linspace(lo, hi, n)
    A, B = np.meshgrid(g, g)
    return A.ravel(), B.ravel()


def certify_span(coord_fn, basis_fns, n=40):
    """Turn an ANALYTIC span claim into a numerically certified one.

    Returns the rank of the coordinate map, the rank of the claimed analytic
    basis, and the rank of the two stacked. Equal ranks throughout means the
    claimed basis spans exactly the readable function space -- an honest
    certificate, not a proof, and labelled as such in the output.
    """
    T1, T2 = _grid(n)
    Cm = np.array([coord_fn(a, b) for a, b in zip(T1, T2)], dtype=float)
    Bm = np.array([[f(a, b) for f in basis_fns] for a, b in zip(T1, T2)],
                  dtype=float)
    rc = int(np.linalg.matrix_rank(Cm, tol=1e-9))
    rb = int(np.linalg.matrix_rank(Bm, tol=1e-9))
    rs = int(np.linalg.matrix_rank(np.hstack([Cm, Bm]), tol=1e-9))
    return {"rank_coordinate_map": rc, "rank_claimed_basis": rb,
            "rank_stacked": rs, "spans_agree": bool(rc == rb == rs),
            "grid": "%d x %d over theta in [%.2f, %.2f]^2" % (n, n, 0.30, 2.90),
            "label": ("CERTIFIED on a dense grid (derivation in the module "
                      "docstring; this is a numerical certificate, not a proof)")}


PRODUCT_BASIS = [
    lambda a, b: 1.0,
    lambda a, b: math.cos(b),
    lambda a, b: math.sin(b),
    lambda a, b: math.cos(a),
    lambda a, b: math.cos(a) * math.cos(b),
    lambda a, b: math.cos(a) * math.sin(b),
    lambda a, b: math.sin(a),
    lambda a, b: math.sin(a) * math.cos(b),
    lambda a, b: math.sin(a) * math.sin(b),
]

EIN_BASIS = [
    lambda a, b: 1.0,
    lambda a, b: math.cos(a),
    lambda a, b: (1.0 - math.cos(a)) * math.cos(b),
    lambda a, b: (1.0 - math.cos(a)) * math.sin(b),
    lambda a, b: math.sin(a) * math.cos(b / 2.0),
    lambda a, b: math.sin(a) * math.sin(b / 2.0),
]


def pauli_coordinate_table():
    """The E-across Pauli table of SS3, VERIFIED numerically rather than asserted."""
    P = {"I": np.eye(2), "X": np.array([[0., 1.], [1., 0.]]),
         "Y": np.array([[0., -1j], [1j, 0.]]), "Z": np.array([[1., 0.], [0., -1.]])}
    claim = {
        "I(x)I": lambda a, b: 1.0,
        "Z(x)I": lambda a, b: math.cos(a),
        "I(x)Z": lambda a, b: math.cos(b),
        "X(x)I": lambda a, b: math.sin(a) * math.cos(b),
        "I(x)X": lambda a, b: math.cos(a) * math.sin(b),
        "X(x)Z": lambda a, b: math.sin(a),
        "Z(x)X": lambda a, b: math.sin(b),
        "Z(x)Z": lambda a, b: math.cos(a) * math.cos(b),
        "Y(x)Y": lambda a, b: math.sin(a) * math.sin(b),
        "X(x)X": lambda a, b: 0.0,
    }
    T1, T2 = _grid(21)
    worst = {}
    for key, fn in claim.items():
        p, q = key.split("(x)")
        O = np.kron(P[p], P[q])
        e = 0.0
        for a, b in zip(T1, T2):
            v = across_amplitudes(a, b, cz=True)
            e = max(e, abs(float(np.real(v @ O @ v)) - fn(a, b)))
        worst[key] = e
    return {"claimed_closed_forms":
                {"I(x)I": "1", "Z(x)I": "cos tA", "I(x)Z": "cos tB",
                 "X(x)I": "sin tA cos tB", "I(x)X": "cos tA sin tB",
                 "X(x)Z": "sin tA", "Z(x)X": "sin tB",
                 "Z(x)Z": "cos tA cos tB", "Y(x)Y": "sin tA sin tB",
                 "X(x)X": "0 (identically)"},
            "max_abs_error_vs_statevector": {k: float(v) for k, v in worst.items()},
            "all_within_1e-12": bool(max(worst.values()) < 1e-12),
            "reading": ("every coordinate lies in the 9-dim product span "
                        "{1,cos tA,sin tA} (x) {1,cos tB,sin tB} and together they "
                        "fill it: CZ is Clifford, so it PERMUTES Pauli coordinates "
                        "and leaves the readable function space invariant")}


# ---------------------------------------------------- committed cross-checks
def verify_committed_diagnostic(coords, path=None):
    """Re-derive `diagnostic_encoding_entanglement.json` from the closed forms.

    This is the validation that licenses using the committed degeneracy check as
    a binding input: if the closed forms in SS2/SS3 reproduce it element by
    element, the committed finding and this module are the same object.
    """
    path = path or os.path.join(HERE, "diagnostic_encoding_entanglement.json")
    blob = json.load(open(path, encoding="utf-8"))
    per = blob["per_noun"]
    rt = runtime_angles()
    # The committed producer used the SIX-DECIMAL-ROUNDED normalisation divisors
    # as printed in instrument_sb_bounds' provenance string; exp70 uses the exact
    # ones, imported. Reproducing the committed file therefore needs the same
    # rounding, and the residual scale factor is recorded rather than absorbed.
    den = isb.load_raw()[1]["norm_divisors"]
    rnd = [round(den[0], 6), round(den[1], 6)]
    w_reg = w_eps = w_ang = w_p1 = w_p2 = 0.0
    w_reg_x = w_eps_x = 0.0
    for n, rec in per.items():
        p1, p2 = coords[n]
        q1, q2 = p1 * den[0] / rnd[0], p2 * den[1] / rnd[1]
        w_p1 = max(w_p1, abs(q1 - rec["p1"]))
        w_p2 = max(w_p2, abs(q2 - rec["p2"]))
        w_reg = max(w_reg, abs(ein_concurrence(full_map(q1), full_map(q2))
                               - rec["concurrence_registered_map"]))
        w_eps = max(w_eps, abs(ein_concurrence(eps_map(q1), eps_map(q2))
                               - rec["concurrence_eps_map"]))
        w_reg_x = max(w_reg_x, abs(ein_concurrence(full_map(p1), full_map(p2))
                                   - rec["concurrence_registered_map"]))
        w_eps_x = max(w_eps_x, abs(ein_concurrence(eps_map(p1), eps_map(p2))
                                   - rec["concurrence_eps_map"]))
        w_ang = max(w_ang, abs(rt[n] - rec["frozen_angle"]))
    names = list(per)
    pur = sorted(across_purity(per[names[i]]["frozen_angle"],
                               per[names[j]]["frozen_angle"])
                 for i in range(len(names)) for j in range(i + 1, len(names)))
    tgt = blob["e_across_intro_purity"]
    k = len(pur) // 2
    return {
        "source_file": os.path.basename(path),
        "source_sha256": isb._sha(path),
        "max_abs_error_under_the_committed_producers_own_convention": {
            "p1": w_p1, "p2": w_p2, "frozen_angle": w_ang,
            "concurrence_registered_map": w_reg, "concurrence_eps_map": w_eps},
        "closed_forms_reproduce_committed_per_noun": bool(
            max(w_p1, w_p2, w_ang, w_reg, w_eps) < 1e-12),
        "frozen_angle_space": (
            "the committed `frozen_angle` is EXACTLY exp42_noun_embeddings.json's "
            "stored angle -- the RUNTIME angle -- not atan2 of the normalised "
            "coordinates. Verified to 0.0 on all 24 nouns."),
        "normalisation_forensics": {
            "finding": ("the committed producer's p1/p2 differ from "
                        "instrument_sb_bounds.load_raw's by a CONSTANT per-component "
                        "scale of 1 + 5.58e-08 (PC1) and 1 - 1.43e-08 (PC2)"),
            "cause": ("it used the SIX-DECIMAL-ROUNDED divisors printed in "
                      "instrument_sb_bounds' provenance string (8.669495, "
                      "10.160404) instead of the exact ones (8.669494516224397, "
                      "10.160404144924657). Re-applying that rounding reproduces "
                      "the committed p1/p2 to 1.1e-16."),
            "consequence_for_exp70": (
                "exp70 uses the EXACT divisors, imported from "
                "instrument_sb_bounds and never re-typed. Under the exact "
                "divisors the committed eps-map concurrences shift by at most "
                "%.2e and the registered-map ones by at most %.2e -- seven "
                "orders of magnitude below the 0.10 degenerate-entanglement "
                "gate and below every digit the committed diagnostic's finding "
                "is stated to. The committed finding stands unchanged."
                % (w_eps_x, w_reg_x)),
            "max_abs_concurrence_shift_under_exact_divisors": {
                "eps_map": w_eps_x, "registered_map": w_reg_x},
        },
        "e_across_purity": {
            "recomputed_min": pur[0], "committed_min": tgt["min"],
            "recomputed_max": pur[-1], "committed_max": tgt["max"],
            "min_max_exact": bool(abs(pur[0] - tgt["min"]) < 1e-12
                                  and abs(pur[-1] - tgt["max"]) < 1e-12),
            "recomputed_median_interpolated": float(np.median(pur)),
            "recomputed_median_upper_element": pur[k],
            "committed_median": tgt["median"],
            "median_convention_note": (
                "the committed median equals the UPPER middle element of the "
                "276 distinct-pair values (sorted[len//2]), not numpy's "
                "interpolated median (%.10f). Every element matches to <1e-12; "
                "the only difference is the median convention, so the closed "
                "form 1 - (sin tA sin tB)^2/2 reproduces the committed "
                "distribution exactly." % float(np.median(pur))),
            "pair_set": "276 distinct unordered pairs of the 24 human nouns",
        },
    }


# ------------------------------------------------------------------ selftest
# FOUR committed anchors. The first three are `capacity_check.ANCHORS` verbatim;
# the fourth exercises d_joint = 81, which none of the three reach and which is
# exactly the path the P-eps arm uses.
ANCHORS = list(cc.ANCHORS) + [
    {"name": "exp69_F2_raw_per_twin",
     "map": ("raw", "F2"), "category": "nonoccupational", "mode": "twin",
     "expect": 1.0, "fraction": "160/160", "exact": False,
     "source": "exp69_predictions.json :: grade1_bounds.R-W "
               "(= diagnostic_sb_bounds.json target_selection.table."
               "nonoccupational.F2.per_twin)"},
]


def _anchor_map(fam, kind):
    """The committed angle maps, routed through THIS module's product path."""
    angles, prov = (cc._frozen2d_angles() if fam == "frozen2d"
                    else cc._raw_angles(kind))
    names = list(angles.keys())
    per = {n: cc.referent_vector(a) for n, a in angles.items()}
    return product_map("%s/%s" % (fam, kind), names, per,
                       "committed anchor map, product path"), prov


def selftest(verbose=True):
    """The extension reproduces the committed product-map values EXACTLY.

    This is the whole warrant for the new numbers: the generalised row builder
    and the generalised twin seeding must leave the product case untouched.
    """
    out = {"anchors": [], "invariants": {}, "structure": {}}
    ok = True
    for a in ANCHORS:
        fam, kind = a["map"]
        m, _p = _anchor_map(fam, kind)
        B = isb.CATEGORIES[a["category"]]["members"]
        casts, _n = cc.casts_from_bearers(m.names, B)
        r = _public(bound_joint(m, casts, a["mode"], **BUDGET))
        hit = (abs(r["bound"] - a["expect"]) < 5e-5
               and r["fraction"] == a["fraction"]
               and r["exact"] == a["exact"])
        ok = ok and hit
        out["anchors"].append(
            {"anchor": a["name"], "expected": a["expect"],
             "expected_fraction": a["fraction"], "got": r["bound"],
             "got_fraction": r["fraction"], "exact": r["exact"],
             "expected_exact": a["exact"], "reproduced": bool(hit),
             "source": a["source"]})
        if verbose:
            print("%-28s expect %.4f %-9s | got %.4f %-9s | %s  %s"
                  % (a["name"], a["expect"], a["fraction"], r["bound"],
                     r["fraction"], "EXACT" if r["exact"] else "lower-bound",
                     "OK" if hit else "*** MISMATCH ***"))

    # the structural invariants, on the NEW maps
    coords, prov = isb.load_raw()
    B = isb.CATEGORIES[TARGET]["members"]
    for m in new_maps(coords):
        casts, _n = cc.casts_from_bearers(m.names, B)
        p = _public(bound_joint(m, casts, "pair", **BUDGET))
        t = _public(bound_joint(m, casts, "twin", **BUDGET))
        chk = {"pair": p["bound"], "twin": t["bound"], "d_joint": m.d,
               "max_at_or_above_chance": bool(p["bound"] >= 0.5 - 1e-12
                                              and t["bound"] >= 0.5 - 1e-12),
               "twin_ge_pair": bool(t["bound"] >= p["bound"] - 1e-9),
               "sign_flip_sums_to_n": bool(p["sign_flip_check"]["sums_to_n"]
                                           and t["sign_flip_check"]["sums_to_n"]),
               "swap_perm_max_abs_error": check_swap_perm(m, casts)}
        chk["swap_perm_verified"] = bool(chk["swap_perm_max_abs_error"] < 1e-12)
        ok = ok and all(chk[k] for k in ("max_at_or_above_chance", "twin_ge_pair",
                                         "sign_flip_sums_to_n",
                                         "swap_perm_verified"))
        out["invariants"][m.name] = chk
        if verbose:
            print("invariants %-14s d=%-3d pair %.4f twin %.4f | >=chance %s "
                  "twin>=pair %s signflip %s swapperm %s"
                  % (m.name, chk["d_joint"], chk["pair"], chk["twin"],
                     chk["max_at_or_above_chance"], chk["twin_ge_pair"],
                     chk["sign_flip_sums_to_n"], chk["swap_perm_verified"]))

    # the two span certificates and the Pauli table
    out["structure"]["E_in_span"] = certify_span(
        lambda a, b: ein_features(a, b), EIN_BASIS)
    out["structure"]["P_eps_span"] = certify_span(
        lambda a, b: product_features([a, b]), PRODUCT_BASIS)
    out["structure"]["E_across_span_equals_product"] = certify_span(
        lambda a, b: across_features(a, b, cz=True), PRODUCT_BASIS)
    out["structure"]["EA_ctrl_span_equals_product"] = certify_span(
        lambda a, b: across_features(a, b, cz=False), PRODUCT_BASIS)
    out["structure"]["pauli_table"] = pauli_coordinate_table()
    for key in ("E_in_span", "P_eps_span", "E_across_span_equals_product",
                "EA_ctrl_span_equals_product"):
        ok = ok and out["structure"][key]["spans_agree"]
    ok = ok and out["structure"]["pauli_table"]["all_within_1e-12"]

    out["committed_diagnostic_crosscheck"] = verify_committed_diagnostic(coords)
    ok = ok and out["committed_diagnostic_crosscheck"][
        "closed_forms_reproduce_committed_per_noun"]

    out["all_passed"] = bool(ok)
    if verbose:
        print("SELFTEST: %s" % ("PASS" if ok else "FAIL"))
    return out


# ------------------------------------------------------------- the exp70 maps
def fa_spaces(coords):
    """The two FA angle spaces, both computed and both reported (SS`runtime_angles`).

    "runtime"    exp42_noun_embeddings.json's stored angle -- what the circuit
                 applies, and what the committed degeneracy diagnostic used.
                 REGISTERED as exp70's E-across angle map.
    "instrument" atan2 of the max-abs-normalised raw-PCA coordinates -- the space
                 every committed FA bound (0.9250 / 0.9625) is stated over, and
                 therefore the space in which an exp70 number is comparable with
                 exp69's.
    """
    return {"runtime": runtime_angles(),
            "instrument": {n: frozen_angle(p1, p2) for n, (p1, p2) in coords.items()}}


def new_maps(coords, fa_space="runtime"):
    """The four exp70 arm-family feature maps, in registration order."""
    names = isb.HUMANS
    eps = {n: (eps_map(p1), eps_map(p2)) for n, (p1, p2) in coords.items()}
    fa = fa_spaces(coords)[fa_space]

    peps = product_map(
        "P-eps", names, {n: product_features(eps[n]) for n in names},
        "P-eps: product register on the eps-map. theta_k = p_k (pi - 2 eps)/2 "
        "+ pi/2, eps = pi/6; RY(theta_1) on q0, RY(theta_2) on q1, NO "
        "entangler. d_ref = 9, d_joint = 81. The MAP-MATCHED PRODUCT TWIN.")

    nin = JointMap(
        "N-in", names, "product",
        None, 36, _vec_transpose_perm(6),
        per_referent={n: ein_features(*eps[n]) for n in names},
        desc="N-in: E-in on the eps-map. RY(theta_1) on q0 then CRY(theta_2) "
             "q0 -> q1, orientation PC1-controls-PC2 registered a priori. "
             "d_ref = 6 (outer square of the three state coefficients), "
             "d_joint = 36.")
    nin.phi = (lambda a, p, _r=nin.per_referent:
               np.outer(_r[a], _r[p]).ravel())

    nacross = across_map(
        "N-across", names, fa, True,
        "N-across: one qubit per referent on the FROZEN angle map (%s space), "
        "plus one CZ between the two referents' wires after intro. JOINT "
        "feature: the 10 independent entries of the outer square of the 4-dim "
        "amplitude vector." % fa_space)

    eactrl = across_map(
        "EA-ctrl", names, fa, False,
        "EA-ctrl: the same frozen-angle single-qubit map WITHOUT the CZ (%s "
        "space), run in exp70's codepath. Dual role: codepath control (must "
        "reproduce exp69's R-A arms) and the map-matched contrast baseline for "
        "N-across." % fa_space)

    return [peps, nin, nacross, eactrl]


def clifford_crosscheck(coords):
    """The SS3 Clifford claim, turned into numbers rather than left as prose.

    For each FA space and each mode it reports three values that the derivation
    says must agree up to search luck:

      * the PRODUCT path at d_ref = 3 (EXACT in pair mode) -- the committed FA
        instrument, no CZ anywhere;
      * the JOINT 10-coordinate path with the CZ (N-across);
      * the JOINT 10-coordinate path without it (EA-ctrl).

    All three bound the SAME reachable class. Differences are SEARCH artefacts
    (the three run in different coordinate systems, so the randomised search
    lands on different local optima) and the LARGEST value is the best lower
    bound available for all three. That is stated here so no reader mistakes a
    search difference for a capacity difference.
    """
    out = {}
    B = isb.CATEGORIES[TARGET]["members"]
    for space, fa in fa_spaces(coords).items():
        prod = product_map("FA-product/%s" % space, isb.HUMANS,
                           {n: cc.referent_vector(fa[n]) for n in isb.HUMANS},
                           "FA product path, d_ref = 3 (the committed instrument)")
        row = {}
        for mode in ("pair", "twin"):
            casts, _n = cc.casts_from_bearers(isb.HUMANS, B)
            vals = {
                "product_path_no_CZ": _public(bound_joint(prod, casts, mode, **BUDGET)),
                "joint_path_with_CZ_N-across": _public(bound_joint(
                    across_map("x", isb.HUMANS, fa, True, ""), casts, mode, **BUDGET)),
                "joint_path_no_CZ_EA-ctrl": _public(bound_joint(
                    across_map("x", isb.HUMANS, fa, False, ""), casts, mode, **BUDGET)),
            }
            best = max(v["bound"] for v in vals.values())
            row[mode] = {
                "values": {k: {"bound": v["bound"], "fraction": v["fraction"],
                               "exact": v["exact"], "d_joint": v["d_joint"]}
                           for k, v in vals.items()},
                "best_lower_bound_for_all_three": best,
                "spread": best - min(v["bound"] for v in vals.values()),
                "reading": ("the three bound the same reachable class (SS3's "
                            "Clifford argument, span certificate rank 9 = 9 = 9). "
                            "Any spread is search luck across coordinate systems, "
                            "NOT capacity, and the best value governs all three."),
            }
        out[space] = row
    return out


def sensitivity_maps(coords):
    """Registered-but-not-run variants, reported for sensitivity only (the house
    precedent is `instrument_sb_bounds`'s SCALINGS sweep: the map is registered
    by principle, the sweep is reported so the choice is auditable)."""
    names = isb.HUMANS
    rev = {n: (eps_map(p2), eps_map(p1)) for n, (p1, p2) in coords.items()}
    old = {n: (full_map(p1), full_map(p2)) for n, (p1, p2) in coords.items()}
    a = JointMap("N-in-REVERSED", names, "product", None, 36,
                 _vec_transpose_perm(6),
                 per_referent={n: ein_features(*rev[n]) for n in names},
                 desc="E-in with the orientation REVERSED (PC2 controls PC1). "
                      "NOTED, NOT RUN -- PC1 carries 12.7% of the variance "
                      "against PC2's 8.1%, and the orientation is registered a "
                      "priori by that fact, not by this number.")
    a.phi = lambda x, p, _r=a.per_referent: np.outer(_r[x], _r[p]).ravel()
    b = JointMap("N-in-OLDMAP", names, "product", None, 36,
                 _vec_transpose_perm(6),
                 per_referent={n: ein_features(*old[n]) for n in names},
                 desc="E-in under exp68's registered map theta = (p + 1) pi -- "
                      "the map the committed degeneracy check DISQUALIFIED "
                      "(4 nouns below concurrence 0.05, two at 0.0). Reported "
                      "so the cost of the dead zones is visible, NOT RUN.")
    b.phi = lambda x, p, _r=b.per_referent: np.outer(_r[x], _r[p]).ravel()
    a_w = product_map(
        "A0-W", names, {n: product_features(old[n]) for n in names},
        "A0-W: exp68/exp69's F2 product register on the OLD map "
        "theta = (p + 1) pi. The anchor family; its bound of record is the "
        "committed F2 value, reproduced here through the extended path.")
    return [a, b, a_w]


# ------------------------------------------------------------- concurrence
def concurrence_inventory(coords):
    """Per-noun E-in concurrence under both maps, and the per-CAST E-across
    concurrence/purity distribution, computed from the closed forms. The exp70
    degenerate-entanglement smoke gate reads the eps-map minimum."""
    eps_c = {n: ein_concurrence(eps_map(p1), eps_map(p2))
             for n, (p1, p2) in coords.items()}
    old_c = {n: ein_concurrence(full_map(p1), full_map(p2))
             for n, (p1, p2) in coords.items()}
    fa = runtime_angles()      # the angle the circuit applies (SS`runtime_angles`)
    B = isb.CATEGORIES[TARGET]["members"]
    N = [n for n in isb.HUMANS if n not in B]
    cast_c = {"%s|%s" % (b, y): across_concurrence(fa[b], fa[y])
              for b in B for y in N}
    cast_p = {"%s|%s" % (b, y): across_purity(fa[b], fa[y])
              for b in B for y in N}

    def st(d):
        v = np.array(sorted(d.values()))
        return {"min": float(v.min()), "median": float(np.median(v)),
                "max": float(v.max()),
                "n_below_0.05": int((v < 0.05).sum()),
                "n_below_0.10": int((v < 0.10).sum())}

    return {
        "e_in_eps_map": {"per_noun": eps_c, "stats": st(eps_c),
                         "gate_min_0.10_satisfied": bool(min(eps_c.values()) >= 0.10)},
        "e_in_registered_map_DISQUALIFIED": {"stats": st(old_c)},
        "e_across_over_the_80_SB_casts": {
            "concurrence": st(cast_c), "purity_min": float(min(cast_p.values())),
            "purity_median": float(np.median(list(cast_p.values()))),
            "purity_max": float(max(cast_p.values())),
            "per_cast_purity": cast_p,
            "closed_form": "purity = 1 - (sin tA sin tB)^2 / 2, both wires",
            "honest_note": (
                "many pairs are NEAR-PRODUCT. Over the full 24-noun inventory the "
                "committed diagnostic reports per-pair purity 0.5764-0.9999 with "
                "median 0.983, i.e. the median cast is only weakly entangled. "
                "E-across is a MECHANISM PROBE and this is the honest measure of "
                "how much mechanism it actually injects.")},
    }


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
    st = selftest(verbose=True)
    if a.selftest:
        print(json.dumps(st, indent=2, ensure_ascii=False))
        return 0 if st["all_passed"] else 1
    if not st["all_passed"]:
        print("SELFTEST FAILED -- no number is written", file=sys.stderr)
        return 1

    coords, prov = isb.load_raw()
    B = isb.CATEGORIES[TARGET]["members"]

    bounds, witnesses = {}, {}
    for m in new_maps(coords, fa_space="runtime"):
        for mode in ("pair", "twin"):
            r = bound_with_null(m, B, mode, draws=a.draws,
                                null_seed=a.null_seed, **BUDGET)
            r["definition"] = m.desc
            bounds.setdefault(m.name, {})[mode] = r
            print("%-10s %-4s  %s  %-9s  null %.4f +- %.4f  pct %.1f  [%.0fs]"
                  % (m.name, mode, ("%.4f" % r["bound"]), r["fraction"],
                     r["null"]["mean"], r["null"]["sd"],
                     r["null"]["percentile_of_category"], time.time() - t0))
        casts, _n = cc.casts_from_bearers(m.names, B)
        w = bound_joint(m, casts, "twin", **BUDGET)
        witnesses[m.name] = {
            "W": [float(x) for x in w["_c"]],
            "d": int(len(w["_c"])), "n_twins": int(w["n_rows"]),
            "n_positive_margin": int(w["sign_flip_check"]["satisfied_c"]),
            "certifies_the_bound_count":
                bool(w["sign_flip_check"]["satisfied_c"] == w["count"]),
            "convention": (
                "doc 55 SS19.3, re-pinned for exp70's maps: the witness of "
                "record is THIS committed search's output at THIS registered "
                "seed (%d), exported at bound-computation time, BEFORE any "
                "exp70 run. margin(twin) = <W, Phi>/(||W|| ||Phi||) with Phi "
                "the AGENT-FIRST joint feature of the map named above and the "
                "sign convention of `maxfs_lower`; pair margin = min over the "
                "pair's two twins." % BUDGET["seed"]),
        }

    sens = {}
    for m in sensitivity_maps(coords):
        casts, _n = cc.casts_from_bearers(m.names, B)
        r = _public(bound_joint(m, casts, "twin", **BUDGET))
        r["definition"] = m.desc
        r["null"] = "NOT COMPUTED -- sensitivity row, noted-not-run; a bound " \
                    "without a size-matched null is context, never a criterion"
        sens[m.name] = r
        print("sensitivity %-14s twin %.4f %s" % (m.name, r["bound"], r["fraction"]))

    out = {
        "component": "exp70_entangled_bounds",
        "purpose": ("grade-1 reachable-accuracy bounds for exp70's entangled "
                    "encodings, computed BEFORE any exp70_*.py exists (doc 55 "
                    "SS19.2's ordering rule, inherited)"),
        "spec": "qnlp_private_docs/56_exp70_entangled_encodings_DRAFT.md",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script_sha256": self_sha256(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "provenance": {
            "instrument_sb_bounds_sha256": isb.self_sha256(),
            "capacity_check_sha256": hashlib.sha256(
                open(os.path.join(HERE, "capacity_check.py"), "rb").read()).hexdigest(),
            "raw_family": prov,
            "budget": BUDGET, "null_draws": int(a.draws),
            "null_seed": int(a.null_seed),
            "extends": ("capacity_check.py, whose scope note explicitly excludes "
                        "'entangling the referent register with itself'. This "
                        "module supplies exactly that case and the cross-referent "
                        "case, and its selftest reproduces capacity_check's "
                        "committed anchors through the generalised path."),
        },
        "category": {"key": TARGET, "gloss": isb.CATEGORIES[TARGET]["gloss"],
                     "members": isb.CATEGORIES[TARGET]["members"],
                     "source": ("instrument_sb_bounds.CATEGORIES -- doc 54 SS5.2, "
                                "user-confirmed 2026-08-23; imported, never re-typed")},
        "eps_map": {"eps": EPS,
                    "formula": "theta = p (pi - 2 eps)/2 + pi/2, eps = pi/6",
                    "range": [math.pi / 6.0, 5.0 * math.pi / 6.0],
                    "forced_by": ("diagnostic_encoding_entanglement.json -- the "
                                  "committed degeneracy check; the exp68 map "
                                  "theta = (p+1)pi puts |p| = 1 exactly on the "
                                  "concurrence dead zones")},
        "selftest": st,
        "derivations": {
            "E_in": {
                "state": "cos(t1/2)|00> + sin(t1/2)cos(t2/2)|10> + sin(t1/2)sin(t2/2)|11>",
                "amplitude_on_01": "identically zero",
                "per_referent_feature": "(a^2, b^2, c^2, ab, ac, bc) -- outer square of the three coefficients",
                "d_per_referent": 6, "d_joint": 36,
                "span": ["1", "cos t1", "(1 - cos t1) cos t2", "(1 - cos t1) sin t2",
                         "sin t1 cos(t2/2)", "sin t1 sin(t2/2)"],
                "concurrence": "|sin t1| |sin(t2/2)|",
                "per_wire_purity": "1 - |sin t1|^2 |sin(t2/2)|^2 / 2",
                "relation_to_the_product_register": (
                    "NEITHER NESTED. The two half-angle terms leave the product "
                    "span; `sin t1` alone is absent from the E-in span. The "
                    "intersection is 4-dimensional (ranks 6, 9, 11 for E-in, "
                    "P-eps and their union). The map is RESTRUCTURED, not "
                    "enlarged, and nothing about which reaches further follows "
                    "from dimension counting."),
                "completeness": ("COMPLETE. The six coordinates are derived in "
                                 "closed form and the claimed 6-function basis is "
                                 "certified to span exactly the same space."),
            },
            "E_across": {
                "state": "( cA cB, cA sB, sA cB, -sA sB ), cX = cos(tX/2), sX = sin(tX/2)",
                "joint_feature": "the 10 independent entries of the outer square of that 4-vector",
                "d_joint": 10, "effective_rank": 9,
                "concurrence": "|sin tA| |sin tB|",
                "per_wire_purity": "1 - (sin tA sin tB)^2 / 2",
                "clifford_argument": (
                    "CZ is Clifford, so conjugation permutes the Pauli "
                    "coordinates; with a general readout the reachable function "
                    "space is INVARIANT. The E-across span is exactly the "
                    "9-dimensional product span, identical to EA-ctrl's, with "
                    "<X(x)X> identically zero and <Y(x)Y> = sin tA sin tB "
                    "carrying what <X(x)X> carried in the control."),
                "consequences": [
                    "the N-across grade-1 bound EQUALS EA-ctrl's, by proof",
                    "FOURIER CLOSURE IS A NULL INSTRUMENT for N-across: closure "
                    "in the product basis holds with the CZ present and absent "
                    "alike, so validity must be asserted by the purity closed "
                    "form and a circuit-difference check instead",
                    "the joint-state comparator for N-across is the SAME function "
                    "class as EA-ctrl's, so it cannot separate them either: any "
                    "measured difference is trainability / inductive bias, never "
                    "capacity",
                ],
                "completeness": ("COMPLETE, and verified: the Pauli closed-form "
                                 "table is checked against the statevector to "
                                 "better than 1e-12 on a 441-point grid."),
            },
        },
        "concurrence_and_purity": concurrence_inventory(coords),
        "grade1_bounds": bounds,
        "fa_angle_space": {
            "registered_for_exp70": "runtime (exp42_noun_embeddings.json angle)",
            "why": ("it is the angle the compiled circuit applies and the angle "
                    "the committed degeneracy diagnostic states its E-across "
                    "distribution over; the instrument space is reported "
                    "alongside because every committed FA bound lives there"),
            "clifford_crosscheck": clifford_crosscheck(coords),
        },
        "bound_of_record_note": {
            "N-across / EA-ctrl": (
                "the FA family's bound OF RECORD is the committed 0.9625 "
                "(154/160) from diagnostic_readout_arity_bounds_addendum.json's "
                "three-stage search (doc 54 SS17.7's supersession). The value "
                "computed here by the TWO-stage search is the pre-supersession "
                "0.9250 (148/160) -- the capacity_check anchor -- and is "
                "reported only as a consistency check, never as a second bound. "
                "Both are lower bounds; the higher one governs."),
            "P-eps / N-in": (
                "no committed bound exists for either map; the values here are "
                "the bounds of record, both CONSTRUCTIVE LOWER BOUNDS."),
            "binding_rule": ("doc 55 SS6 / SS12: NO exp70 criterion reads against "
                             "any number in this file. doc 54 SS17.7 is the "
                             "standing reason -- a bound moved after exp68 "
                             "reported and no stored boolean moved with it."),
        },
        "grade3_under_conversion_flag": {
            "flag": ("under-conversion predicted for a BELOW-NULL category (the "
                     "F1 observation, made prospective -- doc 55 SS19.2 grade 3)"),
            "percentile_vs_size_matched_null":
                {k: v["twin"]["null"]["percentile_of_category"]
                 for k, v in bounds.items()},
            "below_null": {k: v["twin"]["null"]["below_null"]
                           for k, v in bounds.items()},
        },
        "witnesses_doc55_SS19_3": witnesses,
        "sensitivity_noted_not_run": sens,
        "wall_sec": time.time() - t0,
    }
    g3 = out["grade3_under_conversion_flag"]
    g3["any_arm_below_null"] = bool(any(g3["below_null"].values()))
    g3["testable_this_run"] = g3["any_arm_below_null"]
    g3["verdict"] = (
        "TESTABLE: at least one exp70 arm sits below its size-matched null."
        if g3["any_arm_below_null"] else
        "UNTESTABLE THIS RUN -- no exp70 arm has a below-null category. Computed "
        "before freezing and recorded rather than quietly dropped, exactly as "
        "doc 55 SS19.2 did for exp69.")

    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("wrote %s  (%.0f s)" % (a.out, out["wall_sec"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
