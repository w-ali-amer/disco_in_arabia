# -*- coding: utf-8 -*-
"""instrument_scaling_bounds.py -- PRE-WORK INSTRUMENT for doc 63 / exp76.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is a BOUND CALCULATOR, not an experiment. It is written under the
instrument exception doc 54 records in its status block (docs 55/56/57 cite it
as "doc 54 SS0"): the only code that may exist before a registration commits is
an instrument, and it is named as such. NO exp76 experiment code exists and none
may be written before doc 63 reaches its build line.

It answers one question doc 63 must answer BEFORE its criteria are written
(doc 55 SS19.2's ordering rule: grade-1 bounds are computed and frozen before
the criteria are read):

    On the D4 inventory of record -- 519 confirmed human-denoting nouns,
    d4_confirmed_inventory.json, sha-stamped -- does the committed grade-1
    reachable-accuracy instrument still DISCRIMINATE, and does the wazn
    contrast sit above or below its size-matched null?

WHY THE QUESTION IS LIVE AT ALL
-------------------------------
Two committed findings make it live, and they point in opposite directions.

  * doc 51 SS10.5: "A cap claim needs a test inventory far larger than 24 nouns
    can provide."  At 24 nouns the wazn split left 8 test nouns and 15.8% of
    random splits reached the same 1.0000 bound. 519 nouns is that larger
    inventory.
  * doc 61 SS2.4: the saturation certificate. When the MAX-FS row matrix G has
    rank equal to its row count, the bound is EXACTLY 1.0000 by proof and the
    grade-1 instrument discriminates nothing. That fired at exp73's A3/A4
    because d_joint (1296, 18496) exceeded n_rows (160).

At the single-angle FA map d_joint = 9. Against a 519-noun inventory n_rows is
in the tens of thousands, so `rank(G) <= 9 << n_rows` and the certificate CANNOT
fire. That is asserted here numerically, not argued.

THE HONEST METHOD STATEMENT (doc 47 SS18.2, SS18.5)
---------------------------------------------------
`instrument_sb_bounds.exact_maxfs_r3` visits every cell of the arrangement of n
planes in R^3. Its cost is O(n^2) candidates x n rows = O(n^3). At the committed
24-noun scale n_rows <= 190 and it is EXACT and instant. At 519 nouns n_rows is
2060 (|B| = 4) to 28728 (|B| = 63) and the enumeration is measured here as
INFEASIBLE -- it did not complete in 120 s at the smallest contrast tried.

    REGISTERED: every 519-scale number this file emits is a CONSTRUCTIVE LOWER
    BOUND from the committed budgeted search (`maxfs_lower`, restarts x steps x
    ls_starts identical for the contrast and for every null draw), never an
    exact maximum. It is labelled as such in every leaf of the output. A
    measured accuracy above it is NOT a violation.

The exactness that is available is used where it is available: `--exact-check`
runs the committed EXACT enumeration on size-matched 24-noun SUB-inventories
drawn from the 519 and compares it against the constructive search on the same
rows, so the search's under-performance is measured rather than assumed.

NULL DISCIPLINE (doc 47 SS16.1, binding)
----------------------------------------
Every bound is reported against a SIZE-MATCHED null: random |B|-subsets of the
SAME 519-noun inventory, scored by the SAME method with the SAME search budget.
Parallelism across draws changes wall time and nothing else; the per-draw seed
is `seed + t`, exactly as `instrument_sb_bounds.null_stats` sets it, so the
output is deterministic and independent of worker count.

THE TWO CONTRAST LAYERS, AND WHY THERE ARE TWO
----------------------------------------------
(1) WAZN MEMBERSHIP, with its provenance made explicit. doc 51 SS10.6's standing
    rule -- "when an encoding 'carries' a linguistic class, test whether it
    carries the CLASS or a correlate of it" -- is why the faaCil contrast is
    computed under TWO membership definitions that disagree:

      faail_surface  the `wazn_tag` field shipped in d4_confirmed_inventory.json.
                     Its producer (`extract_aravec_candidates.py:99-112`) calls
                     it "Surface wazn tags" in its own comment and computes it by
                     REGEX over the CAMeL lemma's bare consonant string. The
                     lemma is morphological; the tag over it is a surface
                     pattern match.
      faail_camel    the CAMeL Tools `pattern` field of the admitting analysis,
                     restricted to the enumerated active-participle realisations
                     in FAAIL_CAMEL_PATTERNS below. This is morphological
                     analysis output, not a string test.

    They are NOT the same set (measured: 81 vs 63, intersection 62), and doc 63
    reports both with the difference, rather than picking one and calling it
    "the wazn labels".

(2) CONSTRUCTED ANGLE BANDS, the capacity-validation layer. A bearer set built to
    occupy exactly A contiguous arcs of the FA angle circle, at |B| matched to
    the wazn contrast. A = 2 must return EXACTLY 1.0000 (doc 47 SS13.3's sanity
    check: a single RY carves 2 arcs). Higher A calibrates what a bound of x
    MEANS in arcs at this inventory size, which is the only way to read a wazn
    number that is neither 0.5 nor 1.0.

SCOPE -- WHAT THIS FILE DOES NOT BOUND
--------------------------------------
`capacity_check.py`'s scope statement binds verbatim: the instrument is NOT a
bound "on tasks with more than two referents". Every number here is the n = 2 SB
decision `sign(u_a^T C u_p)` over the 519-noun inventory. It is a statement about
the INVENTORY, and it is not a grade-1 bound for any exp76 arm at n > 2. doc 63
states that as a scoping statement before its criteria, in the shape doc 62 SS2.2
used for the trained-encoding case.

CLI
---
    python instrument_scaling_bounds.py                 # -> diagnostic_scaling_bounds.json
    python instrument_scaling_bounds.py --selftest      # instrument checks only
    python instrument_scaling_bounds.py --exact-check   # add the 24-noun exact bridge
"""
import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import math
import os
import platform
import sys

import numpy as np

import instrument_sb_bounds as ISB

HERE = os.path.dirname(os.path.abspath(__file__))
INVENTORY = os.path.join(HERE, "d4_confirmed_inventory.json")
CANDIDATES = os.path.join(HERE, "candidate_human_nouns.json")
RAW519 = os.path.join(HERE, "aravec_raw_vectors_candidates.json")
RAW44 = os.path.join(HERE, "aravec_raw_vectors_44.json")
FROZEN_EMB = os.path.join(HERE, "exp42_noun_embeddings.json")
CAPACITY = os.path.join(HERE, "capacity_check.py")
OUT = os.path.join(HERE, "diagnostic_scaling_bounds.json")

# The 24 committed human nouns, imported by NAME from the committed instrument
# and never re-typed (doc 54 SS5.2's discipline).
HUMANS24 = ISB.HUMANS

# CAMeL Tools `pattern` strings that ARE the active participle faaCil, enumerated
# rather than matched by a regex, so the membership is auditable line by line.
# The gloss names the root class each realisation comes from.
FAAIL_CAMEL_PATTERNS = {
    "ال1ا2ِ3": "sound root (salim) -- the canonical al-faaCil",
    "ال1ائِ3": "hollow root (ajwaf) -- al-faaCil realised faa'iC",
    "ال1ا2ِي": "defective root (naqis) -- al-faaCil realised faaCi",
    "الوا2ِ3": "assimilated waw root (mithal) -- waaCiC",
    "ال1ا2ِئ": "hamzated lam -- faaCi'",
    "ال1ا2ّ": "geminate root (mudaaCaf) -- faaCC",
}
# Deliberately EXCLUDED, and the exclusions are stated rather than silent:
#   ال1اوِي, ال1ا2ِيَة  -- not the bare participle (extended / feminine forms)
#   ال1ا2ِ3ِيّ          -- a nisba built ON a participle; its final -iyy is the
#                          near-tautological character bigram doc 51 SS3 refused
#                          to quote as evidence
FAAIL_CAMEL_EXCLUDED = ["ال1اوِي", "ال1ا2ِيَة", "ال1ا2ِ3ِيّ"]

# Surface wazn tags shipped in the inventory, reported for the provenance table.
SURFACE_TAGS = ["faail", "mu_participle", "nisba_yaa", "faiil", "faaal_or_fa33aal"]

# Committed search budget, unchanged from instrument_sb_bounds.main's defaults.
BUDGET = dict(restarts=48, steps=500, ls_starts=6, seed=0)
NULL_SEED = 68000


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def canonical_sha256(obj):
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ features
def load_inventory519():
    """The 519 confirmed nouns, their shipped surface wazn tags, and the CAMeL
    pattern of the admitting analysis. Every field is READ, never re-typed."""
    inv = json.load(open(INVENTORY, encoding="utf-8"))
    cand = {c["noun"]: c for c in
            json.load(open(CANDIDATES, encoding="utf-8"))["candidates"]}
    nouns, rows = [], {}
    for rec in inv["nouns"]:
        n = rec["noun"]
        c = cand[n]
        nouns.append(n)
        rows[n] = {
            "surface_wazn_tag": rec["wazn_tag"],
            "camel_pattern": c.get("pattern"),
            "camel_lemma": c.get("lemma"),
            "camel_root": c.get("root"),
            "camel_stemgloss": c.get("camel_stemgloss"),
            "provenance_tier": c.get("provenance"),
            "aravec_freq_rank": c.get("aravec_count"),
            "audit_note": rec.get("note"),
        }
    assert len(nouns) == inv["meta"]["n_confirmed"] == 519
    # The inventory's own convention: sha256 over the JSON list of noun STRINGS
    # in file order (verified against meta.nouns_sha256, not assumed).
    declared = hashlib.sha256(
        json.dumps(nouns, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert declared == inv["meta"]["nouns_sha256"], \
        "d4_confirmed_inventory.json nouns_sha256 does not verify"
    return nouns, rows, inv["meta"]


def build_fa_features(nouns):
    """FA angle features over the 519, by the committed convention chain.

    The chain is `instrument_sb_bounds.load_raw`'s, applied to a NEW population,
    and every departure is registered here rather than discovered later:

      (a) PCA is fit over the UNION of the 519 and the committed 44 (563
          vectors). The 44 are in the fit for ONE reason: the committed sign pin
          is defined against the frozen vec2 plane, which exists only for the 44,
          and the 519 and the 44 are DISJOINT by construction (the extraction
          excluded the 44 -- candidate_human_nouns.json :: meta.excludes). Fitting
          on the 519 alone would leave the PC signs to numpy's row order.
      (b) The sign pin is then the committed one: PC signs chosen to match the
          frozen vec2 plane over the 24 HUMANS.
      (c) Normalisation divisors are per-component max-abs over THE 519 -- the
          population the bounds are stated over -- not over the 24. Registered as
          a change of population, doc 47 SS18.5.

    NOT leakage in this setting, for exp69's reason (`load_raw`'s docstring): an
    exp76 split is over unseen PAIRINGS/CASTS, not unseen nouns, so every noun's
    feature is in-inventory by construction. If any exp76 arm ever splits over
    unseen NOUNS, doc 51 SS4's leakage finding binds and this featurizer is
    invalid for it -- stated here so the condition travels with the numbers.
    """
    v519 = json.load(open(RAW519, encoding="utf-8"))["vectors"]
    v44 = json.load(open(RAW44, encoding="utf-8"))["vectors"]
    miss = [n for n in nouns if n not in v519]
    assert not miss, "raw vectors lack %r" % miss[:5]
    assert not (set(nouns) & set(v44)), "the 519 and the 44 are not disjoint"

    fit_names = list(nouns) + sorted(v44)
    X = np.array([(v519[n] if n in v519 else v44[n]) for n in fit_names], float)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / float((S ** 2).sum())
    Pall = Xc @ Vt[:2].T
    idx = {n: i for i, n in enumerate(fit_names)}

    # (b) sign pin against the frozen plane over the 24 committed humans
    fz = json.load(open(FROZEN_EMB, encoding="utf-8"))["embeddings"]
    H = np.array([Pall[idx[n]] for n in HUMANS24])
    F = np.array([fz[n]["vec2"] for n in HUMANS24], float)
    Hu = H / np.linalg.norm(H, axis=1, keepdims=True)
    dmin, best = None, None
    for sa in (1, -1):
        for sb in (1, -1):
            dd = float(np.abs(Hu * np.array([sa, sb]) - F).max())
            if dmin is None or dd < dmin:
                dmin, best = dd, (sa, sb)
    Pall = Pall * np.array(best, float)

    P = np.array([Pall[idx[n]] for n in nouns])
    denom = np.abs(P).max(axis=0)                       # (c) over the 519
    Pn = P / denom
    coords = {n: (float(Pn[i, 0]), float(Pn[i, 1])) for i, n in enumerate(nouns)}
    U, desc = ISB.features(coords, "FA")
    r = np.linalg.norm(P, axis=1)
    prov = {
        "family": "raw519_union_fit",
        "feature_kind": "FA", "d": 3, "d_joint": 9, "formula": desc,
        "raw_file": os.path.basename(RAW519), "raw_file_sha256": _sha(RAW519),
        "raw44_file": os.path.basename(RAW44), "raw44_file_sha256": _sha(RAW44),
        "frozen_emb_sha256": _sha(FROZEN_EMB),
        "pca_fit_on": "union of the 519 confirmed and the committed 44",
        "pca_fit_n": int(X.shape[0]), "raw_dim": int(X.shape[1]),
        "var_explained_pc1": float(var[0]), "var_explained_pc2": float(var[1]),
        "var_explained_pc1_plus_pc2": float(var[:2].sum()),
        "sign_pin_applied": list(best),
        "sign_pin_max_abs_coord_diff_over_humans24": dmin,
        "norm_divisors_over_519": [float(denom[0]), float(denom[1])],
        "normalisation": ("p_k(r) = PC_k(r) / max_{r' in the 519} |PC_k(r')|; "
                          "REGISTERED as a population change from load_raw's "
                          "max over the 24 HUMANS (doc 47 SS18.5)"),
        "radius_min_max_mean": [float(r.min()), float(r.max()), float(r.mean())],
        "leakage_condition": ("valid for splits over unseen PAIRINGS only; if any "
                              "arm splits over unseen NOUNS, doc 51 SS4 binds and "
                              "this featurizer is invalid for that arm"),
    }
    return coords, U, prov


# ------------------------------------------------------------------ contrasts
def wazn_sets(nouns, rows):
    surf = [n for n in nouns if rows[n]["surface_wazn_tag"] == "faail"]
    camel = [n for n in nouns if rows[n]["camel_pattern"] in FAAIL_CAMEL_PATTERNS]
    both = [n for n in nouns if n in set(surf) & set(camel)]
    return surf, camel, both


def band_members(coords, n_arcs, size, nouns):
    """A bearer set occupying EXACTLY `n_arcs` contiguous arcs of the FA circle,
    with |B| = size. Constructed, not semantic: this is the capacity-validation
    layer. Nouns are ordered by angle; the circle is cut into `n_arcs` blocks of
    bearers spaced as evenly as the inventory allows."""
    assert n_arcs % 2 == 0 and n_arcs >= 2
    order = sorted(nouns, key=lambda n: math.atan2(coords[n][1], coords[n][0]) % (2 * math.pi))
    k = n_arcs // 2                       # number of bearer RUNS
    per = [size // k + (1 if i < size % k else 0) for i in range(k)]
    N = len(order)
    out = []
    for i, cnt in enumerate(per):
        start = int(round(i * N / k))
        out.extend(order[start:start + cnt])
    assert len(out) == size, (len(out), size)
    return out


def arcs_of(coords, members, nouns):
    order = sorted(nouns, key=lambda n: math.atan2(coords[n][1], coords[n][0]) % (2 * math.pi))
    lab = [n in set(members) for n in order]
    return sum(1 for i in range(len(lab)) if lab[i] != lab[i - 1])


# ------------------------------------------------------------------- bounding
def _one_bound(U, bearers, nonbearers, mode, seed):
    G = ISB.constraint_rows(U, bearers, nonbearers, mode)
    lo, c = ISB.maxfs_lower(G, restarts=BUDGET["restarts"], steps=BUDGET["steps"],
                            seed=seed, ls_starts=BUDGET["ls_starts"])
    return G, lo, c


def constructive_bound(U, bearers, nonbearers, mode, seed=None):
    seed = BUDGET["seed"] if seed is None else seed
    G, lo, c = _one_bound(U, bearers, nonbearers, mode, seed)
    n = G.shape[0]
    rank = int(np.linalg.matrix_rank(G))
    return {
        "mode": mode, "d": len(next(iter(U.values()))), "d_joint": G.shape[1],
        "n_bearers": len(bearers), "n_nonbearers": len(nonbearers),
        "n_casts": len(bearers) * len(nonbearers), "n_rows": n,
        "count": int(lo), "value": lo / n, "fraction": "%d/%d" % (lo, n),
        "exact": False,
        "method": ("CONSTRUCTIVE LOWER BOUND (%d restarts x %d steps + %d LS-drop "
                   "starts); the EXACT R^3 arrangement enumeration is O(n_rows^3) "
                   "and is INFEASIBLE at this inventory size"
                   % (BUDGET["restarts"], BUDGET["steps"], BUDGET["ls_starts"])),
        "saturation_certificate": {
            "rank_G": rank, "n_rows": n, "d_joint": int(G.shape[1]),
            "fires": bool(rank == n),
            "note": ("doc 61 SS2.4's certificate fires iff rank(G) == n_rows. At "
                     "d_joint = 9 the rank is at most 9, so it CANNOT fire at this "
                     "inventory size and the grade-1 instrument DISCRIMINATES."),
        },
        "sign_flip_check": ISB.verify_sign_flip(G, c),
        "_c": c,
    }


def _null_draw(args):
    names, Uvals, size, mode, seed_t = args
    U = {n: v for n, v in zip(names, Uvals)}
    rng = np.random.default_rng(seed_t)
    idx = set(int(i) for i in rng.choice(len(names), size=size, replace=False))
    B = [names[i] for i in sorted(idx)]
    N = [names[i] for i in range(len(names)) if i not in idx]
    G, lo, _c = _one_bound(U, B, N, mode, seed_t)
    return lo / G.shape[0]


_NULL_CACHE = {}


def size_matched_null(U, size, mode, draws, workers):
    """SIZE-MATCHED null: random |B|-subsets of the SAME inventory, IDENTICAL
    method and budget. Per-draw seed is NULL_SEED + t, matching
    instrument_sb_bounds.null_stats, so the result does not depend on `workers`.
    """
    # The null depends ONLY on (inventory, |B|, mode, draws, NULL_SEED) -- never on
    # which contrast asked for it. Contrasts of equal |B| therefore share one null,
    # and caching it changes wall time and nothing else.
    key = (size, mode, draws)
    if key in _NULL_CACHE:
        return _NULL_CACHE[key]
    names = list(U.keys())
    Uvals = [U[n] for n in names]
    jobs = [(names, Uvals, size, mode, NULL_SEED + t) for t in range(draws)]
    if workers <= 1:
        vals = [_null_draw(j) for j in jobs]
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            vals = list(ex.map(_null_draw, jobs, chunksize=1))
    _NULL_CACHE[key] = np.array(vals)
    return _NULL_CACHE[key]


def with_null(U, bearers, nonbearers, mode, draws, workers):
    r = {k: v for k, v in constructive_bound(U, bearers, nonbearers, mode).items()
         if not k.startswith("_")}
    nul = size_matched_null(U, len(bearers), mode, draws, workers)
    pct = float((nul < r["value"]).mean() * 100.0)
    r["null"] = {
        "draws": int(draws), "size_matched_B": len(bearers), "null_seed": NULL_SEED,
        "mean": float(nul.mean()), "sd": float(nul.std(ddof=1)),
        "min": float(nul.min()), "max": float(nul.max()),
        "percentile_of_contrast": pct,
        "identical_search_budget": True,
        "z_vs_null": float((r["value"] - nul.mean()) / max(nul.std(ddof=1), 1e-12)),
    }
    # doc 55 SS19.2's grade 3: an under-conversion flag needs a BELOW-NULL contrast.
    r["grade3"] = {
        "below_null": bool(r["value"] < r["null"]["mean"]),
        "below_null_5th_percentile": bool(pct < 5.0),
        "testable": True,
        "note": ("grade 3 is TESTABLE at this inventory size because the null is "
                 "not a point mass (sd > 0) -- the condition doc 61 SS2.5 recorded "
                 "as failing at exp73's saturated rungs"),
    }
    return r


# ------------------------------------------------------------------- selftest
def selftest(U, coords, nouns, workers):
    """Instrument checks that must pass before any number below is believed."""
    out = {}

    # 1. the committed instrument still reproduces its committed anchor. The
    #    24-noun tertiary pair-level bound is 0.8526 (81/95), EXACT.
    f2d, _p = ISB.load_frozen2d()
    UA = ISB.features(f2d, "FA")[0]
    B = ISB.CATEGORIES["tertiary"]["members"]
    N = [n for n in ISB.HUMANS if n not in B]
    anch = ISB.bound(UA, B, N, "pair")
    out["committed_anchor_tertiary_pair"] = {
        "value": anch["value"], "fraction": anch["fraction"],
        "exact": anch["exact"],
        "published_doc47_s13": 0.8526,
        "reproduces": bool(abs(anch["value"] - 0.8526) < 5e-5),
    }
    assert out["committed_anchor_tertiary_pair"]["reproduces"], \
        "the committed instrument no longer reproduces doc 47 SS13's anchor"

    # 2. SIGN-FLIP: a MAX can never sit below chance. Asserted on a live contrast.
    Bb = band_members(coords, 2, 40, nouns)
    Nb = [n for n in nouns if n not in set(Bb)]
    b2 = constructive_bound(U, Bb, Nb, "pair")
    assert b2["value"] >= 0.5 - 1e-12, "MAX below chance at 519"
    assert b2["sign_flip_check"]["sums_to_n"]

    # 3. doc 47 SS13.3's sanity check, at 519: a 2-arc constructed band is exactly
    #    separable by a single RY, so its bound must be 1.0000.
    out["two_arc_band_is_exactly_one"] = {
        "size": 40, "arcs": arcs_of(coords, Bb, nouns),
        "value": b2["value"], "fraction": b2["fraction"],
        "passes": bool(b2["value"] >= 1.0 - 1e-12),
    }
    assert out["two_arc_band_is_exactly_one"]["arcs"] == 2
    assert out["two_arc_band_is_exactly_one"]["passes"], \
        ("a 2-arc band did not reach 1.0000 -- the constructive search is "
         "under-powered at this scale and NO number in this file is usable")

    # 4. per-twin >= pair-level, structurally (instrument_sb_bounds.selftest #2).
    tw = constructive_bound(U, Bb, Nb, "twin")
    assert tw["value"] >= b2["value"] - 1e-9, \
        "per-twin below pair-level: the search is under-powered, not the geometry"
    out["per_twin_ge_pair"] = {"pair": b2["value"], "twin": tw["value"]}

    # 5. the saturation certificate must NOT fire (its failing direction).
    out["certificate_does_not_fire"] = {
        "rank_G": b2["saturation_certificate"]["rank_G"],
        "n_rows": b2["saturation_certificate"]["n_rows"],
        "fires": b2["saturation_certificate"]["fires"],
        "as_expected": bool(not b2["saturation_certificate"]["fires"]),
    }
    assert out["certificate_does_not_fire"]["as_expected"]

    # 6. the null is not degenerate -- the condition grade 3 needs (doc 61 SS2.5).
    nul = size_matched_null(U, 40, "pair", 25, workers)
    out["null_is_not_a_point_mass"] = {
        "draws": 25, "mean": float(nul.mean()), "sd": float(nul.std(ddof=1)),
        "min": float(nul.min()), "max": float(nul.max()),
        "passes": bool(nul.std(ddof=1) > 1e-6),
    }
    assert out["null_is_not_a_point_mass"]["passes"]

    out["all_passed"] = True
    return out


def exact_bridge(U, nouns, contrast_members, draws, seed=4242):
    """The EXACT enumeration is feasible at the COMMITTED 24-noun scale and not at
    519. This measures the constructive search's under-performance where both can
    be run: random 24-noun sub-inventories of the 519, bearer count matched to the
    contrast's density, EXACT vs CONSTRUCTIVE on the same rows."""
    rng = np.random.default_rng(seed)
    frac = len(contrast_members) / float(len(nouns))
    size = max(2, int(round(24 * frac)))
    rows = []
    for t in range(draws):
        sub = [nouns[i] for i in sorted(rng.choice(len(nouns), size=24, replace=False))]
        Us = {n: U[n] for n in sub}
        B = [sub[i] for i in sorted(rng.choice(24, size=size, replace=False))]
        N = [n for n in sub if n not in set(B)]
        G = ISB.constraint_rows(Us, B, N, "pair")
        ex = ISB.exact_maxfs_r3(ISB._antisym_r3(Us, B, N))
        lo, _c = ISB.maxfs_lower(G, restarts=BUDGET["restarts"], steps=BUDGET["steps"],
                                 seed=BUDGET["seed"], ls_starts=BUDGET["ls_starts"])
        rows.append({"n_rows": int(G.shape[0]), "exact": int(ex),
                     "constructive": int(lo),
                     "gap": float((ex - lo) / G.shape[0])})
    gaps = np.array([r["gap"] for r in rows])
    return {
        "sub_inventory_size": 24, "bearer_size": size, "draws": draws,
        "per_draw": rows,
        "gap_mean": float(gaps.mean()), "gap_max": float(gaps.max()),
        "search_never_beat_exact": bool(all(r["constructive"] <= r["exact"] for r in rows)),
        "note": ("the constructive search's shortfall against the EXACT maximum, "
                 "measured at the only scale where both run. It does NOT transfer "
                 "as a correction to the 519-scale numbers; it bounds how much "
                 "those numbers may understate, at a smaller n_rows."),
    }


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--exact-check", action="store_true", default=True)
    ap.add_argument("--exact-draws", type=int, default=12)
    a = ap.parse_args()

    nouns, rows, meta = load_inventory519()
    coords, U, featprov = build_fa_features(nouns)

    st = selftest(U, coords, nouns, a.workers)
    if a.selftest:
        print(json.dumps(st, ensure_ascii=False, indent=2))
        return 0

    surf, camel, both = wazn_sets(nouns, rows)

    # ---- the membership provenance table (doc 47 SS13.4's spirit, doc 51 SS10.6)
    prov_table = {
        "why_this_table_exists": (
            "doc 47 SS13.4 makes category membership a DESIGN INPUT with a stated "
            "provenance; doc 54 SS5.1 records what happens when it is not written "
            "down. No 'membership provenance table' existed in the private repo "
            "before this file; this is its first instance."),
        "classes": {
            "faail_surface": {
                "n": len(surf),
                "definition": "d4_confirmed_inventory.json :: nouns[].wazn_tag == 'faail'",
                "produced_by": "extract_aravec_candidates.py :: wazn_tag(lex_bare), lines 99-112",
                "provenance_class": "SURFACE REGEX over a CAMeL-derived lemma",
                "producer_own_words": ("its own comment reads 'Surface wazn tags. "
                                       "DESCRIPTIVE metadata attached to the admitting "
                                       "analysis's lemma -- they are not admission rules "
                                       "and carry no semantic claim.'"),
                "confirmation": ("PROGRAMMATIC. The user's 2026-08-28 approval "
                                 "(doc 59 SS8.4) confirmed HUMAN-DENOTING-NESS over the "
                                 "600 candidates; it did not confirm any wazn tag."),
                "known_defect": ("rule ORDER: the m-initial and yaa-final tests fire "
                                 "before the faaCil test, so m-initial and defective-root "
                                 "participles are mis-tagged. The inventory carries an "
                                 "audit note recording one such case in its own data "
                                 "(المصري: 'Wazn tag mu_participle is an error')."),
            },
            "faail_camel": {
                "n": len(camel),
                "definition": ("candidate_human_nouns.json :: candidates[].pattern in the "
                               "enumerated set FAAIL_CAMEL_PATTERNS"),
                "produced_by": "CAMeL Tools MorphologyDB.builtin_db(), the admitting analysis",
                "provenance_class": "MORPHOLOGICAL ANALYSIS OUTPUT (not a string test)",
                "patterns": FAAIL_CAMEL_PATTERNS,
                "patterns_excluded": FAAIL_CAMEL_EXCLUDED,
                "confirmation": ("PROGRAMMATIC WITH SPOT-CHECK. Distinct in kind from the "
                                 "user-confirmed 24-noun / 9-member faaCil contrast of "
                                 "doc 51 SS7, which the user confirmed noun by noun."),
            },
            "faail_intersection": {
                "n": len(both),
                "definition": "both definitions agree",
                "provenance_class": "the conservative core",
            },
        },
        "disagreement": {
            "camel_only": sorted(set(camel) - set(surf)),
            "surface_only": sorted(set(surf) - set(camel)),
            "n_camel_only": len(set(camel) - set(surf)),
            "n_surface_only": len(set(surf) - set(camel)),
            "reading": ("the shipped surface tag MISSES weak-root realisations of the "
                        "same morphological class and ADMITS at least one loanword that "
                        "is not a participle at all. Reported, not repaired -- doc 63 "
                        "registers BOTH memberships and reads the difference."),
        },
        "surface_tag_counts": {
            t: sum(1 for n in nouns if rows[n]["surface_wazn_tag"] == t)
            for t in SURFACE_TAGS + ["other"]},
        "spot_check_sample": [
            {"noun": n, **{k: rows[n][k] for k in
                           ("surface_wazn_tag", "camel_pattern", "camel_lemma",
                            "camel_root", "camel_stemgloss")}}
            for n in (sorted(set(camel) - set(surf))[:10]
                      + sorted(set(surf) - set(camel))[:5]
                      + sorted(both)[:10])],
    }

    # ---- the contrasts
    contrasts = {}

    def add(key, members, kind, note):
        B = [n for n in nouns if n in set(members)]
        N = [n for n in nouns if n not in set(members)]
        r = with_null(U, B, N, "pair", a.draws, a.workers)
        r["contrast_kind"] = kind
        r["note"] = note
        r["arcs"] = arcs_of(coords, B, nouns)
        r["members_sha256"] = canonical_sha256(sorted(B))
        contrasts[key] = r
        print("[%-22s] |B|=%3d arcs=%3d bound=%.4f null=%.4f+-%.4f pct=%.1f"
              % (key, len(B), r["arcs"], r["value"], r["null"]["mean"],
                 r["null"]["sd"], r["null"]["percentile_of_contrast"]),
              flush=True)

    add("faail_surface", surf, "wazn (surface regex over CAMeL lemma)",
        "the tag as shipped in d4_confirmed_inventory.json")
    add("faail_camel", camel, "wazn (CAMeL pattern field)",
        "morphological analysis output; the provenance class doc 63 registers")
    add("faail_intersection", both, "wazn (both definitions agree)",
        "the conservative core")
    add("mu_participle_surface",
        [n for n in nouns if rows[n]["surface_wazn_tag"] == "mu_participle"],
        "wazn (surface regex)", "reported for the provenance table, not an exp76 arm")
    add("nisba_yaa_surface",
        [n for n in nouns if rows[n]["surface_wazn_tag"] == "nisba_yaa"],
        "wazn (surface regex)",
        "doc 51 SS3 marks the nisba row NEAR-TAUTOLOGICAL under a character map; "
        "reported here under the AraVec angle map, where that objection does not "
        "apply, and still NOT proposed as an exp76 arm")

    # constructed angle bands, |B| matched to the CAMeL faaCil contrast
    size = len(camel)
    for A in (2, 4, 8, 16, 32):
        add("band_%02darc" % A, band_members(coords, A, size, nouns),
            "constructed angle band (capacity-validation layer)",
            "|B| matched to faail_camel; exactly %d arcs by construction" % A)

    grade3_any = any(c["grade3"]["below_null"] for c in contrasts.values())

    out = {
        "component": "exp76_scaling_bounds",
        "purpose": ("grade-1 reachable-accuracy bounds and size-matched nulls for the "
                    "D4 519-noun inventory of record, under the committed single-angle "
                    "FA map, as PRE-WORK for doc 63 / exp76"),
        "status": ("PRE-WORK INSTRUMENT, doc 54's status-block instrument exception "
                   "(cited downstream as doc 54 SS0). This is a bound calculator, not "
                   "an experiment. No exp76 experiment code exists."),
        "spec": "qnlp_private_docs/63_exp76_scaling_DRAFT.md",
        "generated_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__,
        "script_sha256": _sha(os.path.abspath(__file__)),
        "provenance": {
            "instrument_sb_bounds_sha256": _sha(os.path.join(HERE, "instrument_sb_bounds.py")),
            "capacity_check_sha256": _sha(CAPACITY),
            "inventory_file": os.path.basename(INVENTORY),
            "inventory_file_sha256": _sha(INVENTORY),
            "inventory_nouns_sha256": meta["nouns_sha256"],
            "inventory_gate": meta["gate"],
            "inventory_caveat": meta["caveat"],
            "candidates_file_sha256": _sha(CANDIDATES),
            "budget": BUDGET, "null_draws": a.draws, "null_seed": NULL_SEED,
            "calls_not_copies": ("the search, the sign-flip invariant, the constraint "
                                 "rows and the null discipline are "
                                 "instrument_sb_bounds's, imported and called"),
        },
        "features": featprov,
        "population": {
            "n_nouns": len(nouns),
            "statement": ("EVERY bound here is the n = 2 SB decision sign(u_a^T C u_p) "
                          "over the 519-noun inventory. capacity_check.py's scope "
                          "statement binds: this instrument is NOT a bound 'on tasks "
                          "with more than two referents'. It is a statement about the "
                          "INVENTORY, and it is NOT a grade-1 bound for any exp76 arm "
                          "at n > 2."),
        },
        "exactness": {
            "exact_available": False,
            "why": ("exact_maxfs_r3 is O(n_rows^3); n_rows here is 2060-28728 against "
                    "<= 190 at the committed 24-noun scale. Measured: it did not "
                    "complete in 120 s at the smallest 519-scale contrast tried."),
            "consequence": ("every 519-scale value is a CONSTRUCTIVE LOWER BOUND. A "
                            "measured accuracy above it is not a violation."),
        },
        "selftest": st,
        "membership_provenance_table": prov_table,
        "contrasts": contrasts,
        "grade3_under_conversion_flag": {
            "any_contrast_below_null": grade3_any,
            "testable_this_run": True,
            "why_testable": ("the size-matched null has non-zero sd at this inventory "
                             "size, so the percentile has content -- the condition doc "
                             "61 SS2.5 recorded as FAILING at exp73's saturated rungs, "
                             "and doc 55 SS19.2 recorded as untestable at exp69"),
            "below_null_contrasts": sorted(k for k, c in contrasts.items()
                                           if c["grade3"]["below_null"]),
        },
    }
    if a.exact_check:
        out["exact_bridge_at_24"] = exact_bridge(U, nouns, camel, a.exact_draws)

    assert not os.path.exists(a.out), \
        "Rule 3: %s exists and is never overwritten" % a.out
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
