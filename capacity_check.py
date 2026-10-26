# -*- coding: utf-8 -*-
"""capacity_check.py -- the capacity instrument as a REUSABLE TOOL.

WHAT THIS IS
------------
A thin, dependency-light wrapper around the MAX-FS reachable-accuracy machinery
already committed in `instrument_sb_bounds.py` (search + exact R^3 enumeration)
and `exp61_bands.py` (the band geometry and its LP ceilings). It exists so that a
capacity bound can be asked for BEFORE an experiment is designed, from an
arbitrary per-referent angle map, without copying code out of either module.

Nothing here re-derives the mathematics. `instrument_sb_bounds` remains the
authority; this file is an API and a CLI over it.

THE DECISION BEING BOUNDED (doc 46 SS5 / doc 47 SS3.2, SS17)
------------------------------------------------------------
SB: one event, a two-referent cast, exactly one category bearer. Each cast (i, j)
with bearer x_i and non-bearer y_j supplies TWO twins:

    twin YES correct  iff  x_i^T C y_j > 0
    twin NO  correct  iff  y_j^T C x_i < 0

with C a general real d x d matrix on the per-referent feature u(r) in R^d.

    mode="pair"  both twins of a cast won TOGETHER. Subtracting the two rows
                 forces x^T A y > 0 on the ANTISYMMETRIC part A, and A realises
                 them, so the quantity is
                     max_A #{(i,j) : x_i^T A y_j > 0} / (#casts).
                 Denominator = CASTS. This is doc 47 SS3.2, the instrument doc 47
                 SS13 ran.
    mode="twin"  the two twins are SEPARATE constraints on a GENERAL C, so a cast
                 may score 0, 1/2 or 1. Denominator = TWINS (2 x #casts). This is
                 doc 47 SS17's erratum, and it is ALWAYS >= the pair-level value.

They are different quantities over different populations. doc 47 SS18.5 is
binding: a bound is reported together with the population it bounds and the
method that produced it. This tool never returns a number without both.

EXACTNESS DISCIPLINE (doc 47 SS16, SS17, SS18.2)
------------------------------------------------
  * d = 3, mode="pair": EXACT. An antisymmetric 3x3 A is a vector w in R^3 with
    x^T A y = w . (x cross y), so the problem is MAX-FS over n homogeneous
    half-spaces in R^3 and every cell of the plane arrangement can be visited.
    doc 47 SS16's defect (un-perturbed pair-normals sit ON both planes and never
    enter the adjacent open cells) and its fix are inherited from
    `instrument_sb_bounds.exact_maxfs_r3`, together with its strict satisfaction
    tolerance.
  * everything else -- d > 3 in either mode, and d = 3 in mode="twin" -- is a
    CONSTRUCTIVE LOWER BOUND from a budgeted randomised search, labelled as such
    in every return value and every printed line. It is NEVER a maximum, and a
    measurement above it is not a violation. (mode="twin" at d = 3 is seeded with
    the EXACT pair-level witness, so it is guaranteed >= 2 x the pair count; that
    makes it a good lower bound, not an exact one.)

NULL DISCIPLINE (doc 47 SS16.1, binding rule)
---------------------------------------------
The optional null is SIZE-MATCHED -- random |B|-subsets of the SAME referent
inventory, scored by the SAME method with the SAME search budget. An unmatched
budget between the category and its null would be the exp44 unmatched-comparator
failure class occurring inside the diagnostic itself.

SCOPE -- WHAT THIS TOOL DOES **NOT** BOUND
------------------------------------------
This instrument is valid for, and only for:

  * per-referent encodings that are a SINGLE RY PER QUBIT, i.e. a referent
    becomes one angle theta -> u(theta) = (1, cos theta, sin theta), or a
    register of q independent angles -> the Kronecker product of q such vectors
    (d = 3^q). Re-uploading, entangling the referent register with itself, data
    re-encoding at depth, or any parameterised map applied BEFORE the bilinear
    read are all outside it.
  * SB-family twin constructions: exactly one bearer per cast, the twin pair
    being the event and its agent/patient swap, and a decision read as the sign
    of a bilinear form in the two referent features.

It is NOT a bound on arbitrary circuits, on multi-class readouts (see doc 53 SS2
for the Davenport-Schinzel arity ceilings, which are a different instrument), on
tasks with more than two referents, or on any decision that is not
`sign(u_a^T C u_p)`. It also bounds only the POPULATION it is handed: a bound
computed over band centres is a statement about band geometry and NOT about a
finite sampled inventory (doc 47 SS18.1 -- exp61's measured M2w/S1 exceeded the
band ceiling for exactly this reason).

API
---
    capacity(angles, bearers=..., mode="pair"|"twin", null_draws=0, ...) -> dict
    capacity(angles, casts=[(bearer, nonbearer), ...], mode=...)        -> dict
    band_ceiling(cfg, K)     -> exp61_bands.CORRECTED_CEILINGS lookup, verified

CLI
---
    python capacity_check.py --selftest
    python capacity_check.py --angles map.json --bearers a,b,c --mode twin \
                             --null-draws 200
`map.json` is either {noun: angle} or {noun: [angle_qubit0, angle_qubit1, ...]}.

MEMBERSHIP GATE NOTE (2026-08-24): instrument_sb_bounds.py records the
doc 47 SS13 category lists as RECONSTRUCTED with a BLOCKING native-speaker
gate. That gate was CLOSED 2026-08-23 -- the user confirmed all four lists
as printed (doc 54 SS5.2, the record of note). The instrument file predates
the closure; doc 54 SS5.2 supersedes its flag.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import instrument_sb_bounds as isb          # noqa: E402  (authority for the maths)
import exp61_bands                          # noqa: E402  (band geometry + ceilings)

TOL = isb.TOL

DEFAULT_BUDGET = {"restarts": 48, "steps": 500, "ls_starts": 6, "seed": 0}


# ------------------------------------------------------------ feature vectors
def referent_vector(angle):
    """theta (scalar) -> (1, cos, sin);  [theta_0, ...] -> Kronecker product.

    A LIST is a REGISTER: one RY per qubit, independent angles, no entangling
    gate before the bilinear read. d = 3^q.
    """
    if isinstance(angle, (int, float, np.floating, np.integer)):
        return isb._u1(float(angle))
    thetas = list(angle)
    if not thetas:
        raise ValueError("empty angle register")
    v = isb._u1(float(thetas[0]))
    for t in thetas[1:]:
        v = np.outer(v, isb._u1(float(t))).ravel()
    return v


def feature_map(angles):
    """{noun: angle-or-angle-list} -> ({noun: u in R^d}, d, qubits)."""
    U = {n: referent_vector(a) for n, a in angles.items()}
    ds = {len(v) for v in U.values()}
    if len(ds) != 1:
        raise ValueError("ragged register widths: d in %s" % sorted(ds))
    d = ds.pop()
    q = int(round(math.log(d, 3)))
    if 3 ** q != d:
        raise ValueError("d = %d is not 3^q -- not a single-RY-per-qubit map" % d)
    return U, d, q


# ------------------------------------------------------------------ the casts
def casts_from_bearers(names, bearers):
    """Every (bearer, non-bearer) ordered cast on the given inventory."""
    bset = list(bearers)
    unknown = [b for b in bset if b not in names]
    if unknown:
        raise KeyError("bearers not in the angle map: %r" % unknown)
    nons = [n for n in names if n not in set(bset)]
    if not bset or not nons:
        raise ValueError("need >= 1 bearer and >= 1 non-bearer (got %d / %d)"
                         % (len(bset), len(nons)))
    return [(b, y) for b in bset for y in nons], nons


# ------------------------------------------------------------------ the rows
def _rows(U, casts, mode):
    rows = []
    for b, y in casts:
        x, z = U[b], U[y]
        if mode == "pair":
            rows.append((np.outer(x, z) - np.outer(z, x)).ravel())
        elif mode == "twin":
            rows.append(np.outer(x, z).ravel())
            rows.append(-np.outer(z, x).ravel())
        else:
            raise ValueError("mode must be 'pair' or 'twin', got %r" % mode)
    return np.asarray(rows, dtype=float)


def _cross(U, casts):
    return np.array([np.cross(U[b], U[y]) for b, y in casts])


# ------------------------------------------------------------------ the bound
def bound_from_casts(U, casts, mode, exact_if_possible=True, restarts=48,
                     steps=500, seed=0, ls_starts=6):
    """One bound over an EXPLICIT cast list, carrying its own method/exactness.

    Mirrors `instrument_sb_bounds.bound` -- including the twin-mode seeding with
    the pair-level witness, without which the per-twin lower bound can come back
    BELOW the pair-level bound, which is structurally impossible and is an
    instrument defect rather than a finding.
    """
    d = len(next(iter(U.values())))
    G = _rows(U, casts, mode)
    n = G.shape[0]
    exact, inits = None, None
    if mode == "pair" and d == 3 and exact_if_possible:
        exact = isb.exact_maxfs_r3(_cross(U, casts))
    if mode == "twin":
        if d == 3 and exact_if_possible:
            _, w = isb.exact_maxfs_r3(_cross(U, casts), return_w=True)
            A = isb._w_to_antisym3(w).ravel()
        else:
            pl = bound_from_casts(U, casts, "pair", exact_if_possible=False,
                                  restarts=restarts, steps=steps, seed=seed,
                                  ls_starts=ls_starts)
            M = pl["_c"].reshape(d, d)
            A = ((M - M.T) / 2.0).ravel()
        inits = [A / max(float(np.linalg.norm(A)), 1e-300)]
    lo, c = isb.maxfs_lower(G, restarts=restarts, steps=steps, seed=seed,
                            inits=inits, ls_starts=ls_starts)
    count = exact if exact is not None else lo
    method = ("EXACT (R^3 arrangement enumeration, doc 47 SS16 fix)"
              if exact is not None else
              "CONSTRUCTIVE LOWER BOUND (%d restarts x %d steps + %d LS-drop "
              "starts) -- NOT a maximum" % (restarts, steps, ls_starts))
    return {"mode": mode, "d": d, "n_casts": len(casts), "n_rows": n,
            "count": int(count), "bound": count / n,
            "fraction": "%d/%d" % (count, n),
            "population": ("casts (both twins won together)" if mode == "pair"
                           else "twins (a cast may score 0, 1/2 or 1)"),
            "method": method, "exact": exact is not None,
            "sign_flip_check": isb.verify_sign_flip(G, c),
            "_c": c}


def _public(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


# -------------------------------------------------------------------- the API
def capacity(angles, bearers=None, casts=None, mode="pair",
             exact_if_possible=True, restarts=48, steps=500, seed=0,
             ls_starts=6, null_draws=0, null_seed=68000):
    """Reachable-accuracy bound for a per-referent angle map.

    angles      {noun: theta} or {noun: [theta_0, theta_1, ...]} (a REGISTER:
                one RY per qubit, d = 3^q).
    bearers     the bearer set B; the casts are then every (b, non-b) pair.
    casts       alternatively, an explicit list of (bearer, nonbearer) twin-pair
                constraints. Exactly one of `bearers` / `casts` is required.
    mode        "pair" (doc 47 SS3.2, denominator = casts) or
                "twin" (doc 47 SS17, denominator = twins).
    null_draws  > 0 adds a SIZE-MATCHED null: that many random |B|-subsets of the
                same inventory, IDENTICAL method and search budget. Requires
                `bearers`; a null over an arbitrary cast list is not defined.

    Returns a dict: bound, fraction, count, n_rows, method, exact, d, qubits,
    population, sign_flip_check, and (optionally) null.
    """
    if (bearers is None) == (casts is None):
        raise ValueError("pass exactly one of `bearers` or `casts`")
    U, d, q = feature_map(angles)
    names = list(angles.keys())
    if bearers is not None:
        cast_list, nons = casts_from_bearers(names, bearers)
        bset = list(bearers)
    else:
        cast_list = [tuple(c) for c in casts]
        miss = [x for c in cast_list for x in c if x not in U]
        if miss:
            raise KeyError("cast members not in the angle map: %r" % sorted(set(miss)))
        bset = sorted({c[0] for c in cast_list})
        nons = sorted({c[1] for c in cast_list})

    r = _public(bound_from_casts(U, cast_list, mode,
                                 exact_if_possible=exact_if_possible,
                                 restarts=restarts, steps=steps, seed=seed,
                                 ls_starts=ls_starts))
    r.update({"qubits": q, "n_referents": len(names), "n_bearers": len(bset),
              "n_nonbearers": len(nons),
              "cast_source": "all bearer x non-bearer" if casts is None
                             else "explicit cast list",
              "budget": {"restarts": restarts, "steps": steps,
                         "ls_starts": ls_starts, "seed": seed},
              "scope": ("single-RY-per-qubit angle encoding, SB-family twin "
                        "construction; see module docstring")})

    if null_draws:
        if bearers is None:
            raise ValueError("a size-matched null needs `bearers`, not `casts`")
        rng = np.random.default_rng(null_seed)
        vals = []
        for t in range(int(null_draws)):
            idx = sorted(int(i) for i in rng.choice(len(names), size=len(bset),
                                                    replace=False))
            bb = [names[i] for i in idx]
            cl, _nn = casts_from_bearers(names, bb)
            vals.append(bound_from_casts(U, cl, mode,
                                         exact_if_possible=exact_if_possible,
                                         restarts=restarts, steps=steps,
                                         seed=null_seed + t,
                                         ls_starts=ls_starts)["bound"])
        v = np.asarray(vals, dtype=float)
        r["null"] = {"draws": int(null_draws), "size_matched_B": len(bset),
                     "mean": float(v.mean()),
                     "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                     "min": float(v.min()), "max": float(v.max()),
                     "percentile_of_category": float((v < r["bound"]).mean() * 100.0),
                     "identical_search_budget": True,
                     "seed": int(null_seed)}
    return r


def band_ceiling(cfg, K, verify=False):
    """doc 47 SS17's CORRECTED per-twin band ceiling for an exp61 cell.

    These are LP-exact over the BAND geometry -- and therefore a bound over band
    centres, NOT over a finite sampled noun inventory (doc 47 SS18.1).
    """
    key = (cfg, int(K))
    if key not in exp61_bands.CORRECTED_CEILINGS:
        raise KeyError("no corrected ceiling for %r" % (key,))
    out = {"cell": cfg, "K": int(K),
           "bound": exp61_bands.CORRECTED_CEILINGS[key],
           "superseded_lower_bound": exp61_bands.FROZEN_CEILINGS[key],
           "method": "EXACT LP over subsets of per-twin band constraints "
                     "(exp61_bands.true_ceiling, doc 47 SS17)",
           "population": "BAND CENTRES, not a finite noun inventory "
                         "(doc 47 SS18.1)",
           "exact": True}
    if verify:
        out["recomputed"] = exp61_bands.true_ceiling(cfg, int(K))
        out["verified"] = abs(out["recomputed"] - out["bound"]) < 1e-9
    return out


# ---------------------------------------------------------------- angle maps
def _frozen2d_angles():
    """FA on the frozen 2D AraVec plane: phi = atan2(PC2, PC1). doc 47 SS13."""
    coords, prov = isb.load_frozen2d()
    return {n: math.atan2(p2, p1) for n, (p1, p2) in coords.items()}, prov


def _raw_angles(kind):
    """exp68's registered maps on the RAW (un-renormalised) AraVec PCA plane."""
    coords, prov = isb.load_raw()
    fn = isb.SCALINGS[isb.DEFAULT_SCALING][1]
    if kind == "F1":
        out = {n: fn(p1) for n, (p1, p2) in coords.items()}
    elif kind == "F2":
        out = {n: [fn(p1), fn(p2)] for n, (p1, p2) in coords.items()}
    elif kind == "FA":
        out = {n: math.atan2(p2, p1) for n, (p1, p2) in coords.items()}
    else:
        raise ValueError(kind)
    return out, prov


# ------------------------------------------------------------------ selftest
# Three COMMITTED anchors. Hand-typed here on purpose: that is what makes this a
# check rather than a printout.
ANCHORS = [
    {"name": "doc47_SS13_tertiary_pair",
     "map": ("frozen2d", "FA"), "category": "tertiary", "mode": "pair",
     "expect": 0.8526315789473684, "fraction": "81/95", "exact": True,
     "source": "doc 47 SS13 (published 0.8526); diagnostic_sb_bounds.json "
               "membership_reconstruction.tertiary.recomputed_pair_level"},
    {"name": "exp68_F1_raw_per_twin",
     "map": ("raw", "F1"), "category": "nonoccupational", "mode": "twin",
     "expect": 0.7625, "fraction": "122/160", "exact": False,
     "source": "diagnostic_sb_bounds.json target_selection.table."
               "nonoccupational.F1.per_twin (= results_exp68.json "
               "bounds_context_only_doc54_SS12.per_twin_bounds.F1)"},
    {"name": "exp68_FA_per_twin",
     "map": ("raw", "FA"), "category": "nonoccupational", "mode": "twin",
     "expect": 0.9250, "fraction": "148/160", "exact": False,
     "source": "diagnostic_sb_bounds.json target_selection.table."
               "nonoccupational.FA.per_twin (= results_exp68.json "
               "bounds_context_only_doc54_SS12.per_twin_bounds.FA)"},
]


def selftest(verbose=True, verify_bands=False):
    """Reproduce the three committed anchors, plus the structural invariants."""
    out = {"anchors": [], "invariants": {}, "bands": {}}
    ok = True

    for a in ANCHORS:
        fam, kind = a["map"]
        angles, _prov = (_frozen2d_angles() if fam == "frozen2d"
                         else _raw_angles(kind))
        B = isb.CATEGORIES[a["category"]]["members"]
        r = capacity(angles, bearers=B, mode=a["mode"], **DEFAULT_BUDGET)
        hit = (abs(r["bound"] - a["expect"]) < 5e-5
               and r["fraction"] == a["fraction"]
               and r["exact"] == a["exact"])
        ok = ok and hit
        row = {"anchor": a["name"], "expected": a["expect"],
               "expected_fraction": a["fraction"], "got": r["bound"],
               "got_fraction": r["fraction"], "exact": r["exact"],
               "expected_exact": a["exact"], "method": r["method"],
               "reproduced": bool(hit), "source": a["source"]}
        out["anchors"].append(row)
        if verbose:
            print("%-28s expect %.4f %-8s | got %.4f %-8s | %s  %s"
                  % (a["name"], a["expect"], a["fraction"], r["bound"],
                     r["fraction"], "EXACT" if r["exact"] else "lower-bound",
                     "OK" if hit else "*** MISMATCH ***"))

    # structural invariants, on the exp68 raw family (doc 47 SS17 / SIGN-FLIP)
    B = isb.CATEGORIES["nonoccupational"]["members"]
    inv = {}
    for kind in ("F1", "F2", "FA"):
        angles, _p = _raw_angles(kind)
        p = capacity(angles, bearers=B, mode="pair", **DEFAULT_BUDGET)
        t = capacity(angles, bearers=B, mode="twin", **DEFAULT_BUDGET)
        chk = {"pair": p["bound"], "twin": t["bound"], "d": p["d"],
               "max_at_or_above_chance": bool(p["bound"] >= 0.5 - 1e-12
                                              and t["bound"] >= 0.5 - 1e-12),
               "twin_ge_pair": bool(t["bound"] >= p["bound"] - 1e-9),
               "sign_flip_sums_to_n": bool(p["sign_flip_check"]["sums_to_n"]
                                           and t["sign_flip_check"]["sums_to_n"])}
        ok = ok and all(chk[k] for k in ("max_at_or_above_chance", "twin_ge_pair",
                                         "sign_flip_sums_to_n"))
        inv[kind] = chk
        if verbose:
            print("invariants %-3s d=%-2d pair %.4f twin %.4f | >=chance %s "
                  "twin>=pair %s signflip %s"
                  % (kind, chk["d"], chk["pair"], chk["twin"],
                     chk["max_at_or_above_chance"], chk["twin_ge_pair"],
                     chk["sign_flip_sums_to_n"]))
    out["invariants"] = inv

    for cfg in sorted(exp61_bands.CONFIGS):
        for K in (1, 2):
            bc = band_ceiling(cfg, K, verify=verify_bands)
            out["bands"]["%s_K%d" % (cfg, K)] = bc
            if verify_bands:
                ok = ok and bc["verified"]
    if verbose and verify_bands:
        print("exp61_bands CORRECTED_CEILINGS re-verified by LP: %s"
              % all(v.get("verified") for v in out["bands"].values()))

    out["all_passed"] = bool(ok)
    if verbose:
        print("SELFTEST: %s" % ("PASS" if ok else "FAIL"))
    return out


# ----------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--angles", help="JSON: {noun: angle} or {noun: [angles]}")
    ap.add_argument("--bearers", help="comma-separated bearer names")
    ap.add_argument("--casts", help="JSON file: [[bearer, nonbearer], ...]")
    ap.add_argument("--mode", default="pair", choices=["pair", "twin"])
    ap.add_argument("--null-draws", type=int, default=0)
    ap.add_argument("--null-seed", type=int, default=68000)
    ap.add_argument("--restarts", type=int, default=48)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--ls-starts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-exact", action="store_true",
                    help="disable the exact R^3 path (diagnostic use only)")
    ap.add_argument("--band-ceiling", metavar="CELL:K",
                    help="look up an exp61 corrected band ceiling, e.g. M2:1")
    ap.add_argument("--verify-bands", action="store_true",
                    help="re-derive the corrected band table by LP (slow)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", help="write the result dict to this JSON file")
    a = ap.parse_args(argv)

    if a.selftest:
        r = selftest(verbose=True, verify_bands=a.verify_bands)
        if a.out:
            json.dump(r, open(a.out, "w", encoding="utf-8"), indent=2,
                      ensure_ascii=False)
        return 0 if r["all_passed"] else 1

    if a.band_ceiling:
        cell, _, k = a.band_ceiling.partition(":")
        r = band_ceiling(cell, int(k or 1), verify=a.verify_bands)
        print(json.dumps(r, indent=2))
        return 0

    if not a.angles:
        ap.error("--angles is required (or use --selftest / --band-ceiling)")
    angles = json.load(open(a.angles, encoding="utf-8"))
    bearers = a.bearers.split(",") if a.bearers else None
    casts = json.load(open(a.casts, encoding="utf-8")) if a.casts else None
    r = capacity(angles, bearers=bearers, casts=casts, mode=a.mode,
                 exact_if_possible=not a.no_exact, restarts=a.restarts,
                 steps=a.steps, seed=a.seed, ls_starts=a.ls_starts,
                 null_draws=a.null_draws, null_seed=a.null_seed)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    if a.out:
        json.dump(r, open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
