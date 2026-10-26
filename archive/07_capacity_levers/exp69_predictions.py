# -*- coding: utf-8 -*-
"""exp69_predictions.py -- doc 55 SS19.2 / SS19.3: the PROSPECTIVE instrument test.

THIS FILE RUNS BEFORE `exp69_datagen.py`. That ordering is the whole point and it
is registered twice (doc 55 SS19.2 "computed and frozen BEFORE datagen", doc 55
SS19.7 "predictions file (SS19.2/SS19.3, BEFORE datagen), then datagen"). exp69 is
the first OUT-OF-SAMPLE prospective test of the reachable-accuracy calculus this
programme has built, and a prediction written after the dataset exists is not one.
`exp69_datagen.py` refuses to run unless `exp69_predictions.json` is already on
disk, so the ordering is mechanised rather than remembered.

THREE PREDICTION GRADES, KEPT APART (doc 55 SS19.2). Flattening them into one word
"predictions" is exactly the failure this section was written to prevent.

  GRADE 1 -- BOUND. Every arm's per-twin reachable-accuracy bound, computed by
      the COMMITTED instrument, with its method label (exact vs constructive).
      R-A / A0 inherit FA's bound, R-W inherits F2's, R-M is the positional-
      featurizer (wazn) bound. RECOMPUTED HERE, not copied: the R-A and R-W cells
      are re-run through `readout_arity_bounds_addendum.py`'s exact protocol at
      its registered seed (51000) and asserted equal to the committed fractions;
      R-M is computed from scratch through `instrument_sb_bounds` and asserted
      against doc 55 SS19.2's registered 270/270 with null 0.7491 +/- 0.0640 at the
      100th percentile.

  GRADE 2 -- CONVERSION POINT PREDICTION. bound x the trained-population single-RY
      conversion median, with a +/- 0.06 interval MINIMUM, and THE EXTRAPOLATION
      ITSELF DECLARED UNDER TEST: the median comes from n = 13 rows in ONE task
      family and exp69 is a new family. A miss refutes the EXTRAPOLATION, not the
      bound. The median is read out of `diagnostic_capacity_validation.json`
      (trained_population / K1_and_width), never hand-typed.

  GRADE 3 -- UNDER-CONVERSION FLAG. The F1 observation made prospective: a
      BELOW-NULL category predicts incomplete conversion. Computed here BEFORE
      freezing: no exp69 arm has a below-null category (R-A 89.0th percentile,
      R-W 85.5th, R-M 100th), so grade 3 is UNTESTABLE this run and is recorded as
      such rather than quietly dropped.

SS19.3 -- THE WITNESS CONVENTION, PINNED. The MAX-FS witness C is not unique, so
the registered convention is: the committed search's output AT ITS REGISTERED
SEED, exported at bound-computation time (i.e. here, before datagen), with the
per-twin margins computed ONCE from that witness and frozen into this file. The
aggregator's SS19.3 Spearman analysis reads those frozen margins and never
recomputes a witness. Registered caveat, carried: R-W's witness comes from a
lower-bound search at d = 9, so its margins are noisier BY CONSTRUCTION than the
d = 3 arms'.

MARGIN DEFINITION, stated so the aggregator cannot drift from it.
    phi(agent, patient) = vec( u(agent) (x) u(patient) )       -- agent-first
    C                   = the witness, as a d*d vector
    margin(twin)        = s * <C, phi> / (||C|| ||phi||),  s = +1 for the YES twin
                          (agent is a bearer) and -1 for the NO twin
A twin is "reachable under the witness" iff its margin is > 0, and the count of
positive margins is asserted equal to the bound's own count -- that assertion is
what makes the exported witness the same object the bound was computed from.

ONE FEATURE-SPACE NOTE, and it travels with R-A's margins. doc 54 SS6.2 defines the
runtime FA angle as atan2(PC2, PC1) on the UNNORMALISED components, while
`instrument_sb_bounds.features(..., "FA")` computes atan2(p2, p1) on its NORMALISED
coordinates. The two differ by at most ~7.9e-02 rad and preserve the cyclic order
of the referents, which is why the d = 3 BOUND is identical either way (exp68_run
SS6.5 note (2)). The witness, however, lives in the space it was searched in, so
R-A's margins are computed in the INSTRUMENT's FA space and the rank agreement
with the runtime space is measured and reported alongside.

    python exp69_predictions.py                 # writes exp69_predictions.json
    python exp69_predictions.py --quick         # skip the ~10 min re-search and
                                                # use the committed witnesses
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import exp41_story_generator as gen                      # noqa: E402
import exp65_featurizer as WZ                            # noqa: E402
import instrument_sb_bounds as isb                       # noqa: E402
import readout_arity_bounds as rab                       # noqa: E402
import readout_arity_bounds_addendum as add              # noqa: E402

SPEC = "qnlp_private_docs/55_exp69_reunion.md (da4b8fe, REGISTERED through SS19.7)"
OUT = os.path.join(HERE, "exp69_predictions.json")
ADDENDUM = os.path.join(HERE, "diagnostic_readout_arity_bounds_addendum.json")
SB_BOUNDS = os.path.join(HERE, "diagnostic_sb_bounds.json")
CAPVAL = os.path.join(HERE, "diagnostic_capacity_validation.json")

CATEGORY = "nonoccupational"          # doc 55 SS4.1, unchanged from exp68
ADDENDUM_SEED = 51000                 # doc 55 SS19.3's "registered seed"
INTERVAL_MIN = 0.06                   # doc 55 SS19.2 grade 2

# doc 55 SS19.2, registered: which arms inherit which bound.
ARM_FAMILY = {"A0": "R-A", "R1-A": "R-A", "R2-A": "R-A",
              "R1-W": "R-W", "R2-W": "R-W",
              "R1-M": "R-M", "R2-M": "R-M"}

# doc 55 SS12, hand-typed from the committed artefacts so this file is a
# RECONCILIATION and not a printout (the addendum's own convention).
COMMITTED_FRACTION = {"R-A": (154, 160), "R-W": (160, 160), "R-M": (270, 270)}
REGISTERED_RM = {"fraction": "270/270", "value": 1.0,
                 "null_mean": 0.7491, "null_sd": 0.0640, "percentile": 100.0}


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# --------------------------------------------------------------- the designs
def twin_design(U, bearers, nonbearers):
    """doc 47 SS17's per-twin population, in the addendum's row order.

    Row order is IDENTICAL to `readout_arity_bounds_addendum.per_twin_design`, so
    the witness this file exports indexes the rows it was searched over."""
    rows = []
    for b in bearers:
        for n in nonbearers:
            rows.append({"cast": [b, n], "agent": b, "patient": n,
                         "label": "yes", "y": 1,
                         "phi": np.outer(U[b], U[n]).ravel()})
            rows.append({"cast": [b, n], "agent": n, "patient": b,
                         "label": "no", "y": 0,
                         "phi": np.outer(U[n], U[b]).ravel()})
    return rows


def margins_from_witness(rows, C):
    """doc 55 SS19.3's frozen margins. Scale-free, so the Spearman analysis is
    invariant to how the witness happened to be normalised."""
    Cn = np.asarray(C, dtype=float).ravel()
    Cn = Cn / max(float(np.linalg.norm(Cn)), 1e-300)
    out = []
    for r in rows:
        phi = r["phi"]
        pn = phi / max(float(np.linalg.norm(phi)), 1e-300)
        s = 1.0 if r["y"] == 1 else -1.0
        out.append(float(s * float(Cn @ pn)))
    return out


def pack_margins(rows, marg):
    """Per-twin margins keyed by (agent, patient); per-PAIR margin = the min of
    the pair's two twins (a cast is only as reachable as its harder twin)."""
    per_twin, per_pair = {}, {}
    for r, m in zip(rows, marg):
        per_twin["%s|%s" % (r["agent"], r["patient"])] = m
    for r in rows:
        k = "%s|%s" % (r["cast"][0], r["cast"][1])
        a = per_twin["%s|%s" % (r["cast"][0], r["cast"][1])]
        b = per_twin["%s|%s" % (r["cast"][1], r["cast"][0])]
        per_pair[k] = min(a, b)
    return per_twin, per_pair


# ------------------------------------------------- grade 1: the three bounds
def recompute_addendum_cells(quick=False):
    """Re-run `readout_arity_bounds_addendum.py`'s protocol verbatim.

    The addendum consumes ONE `numpy` Generator across all four categories in
    `instrument_sb_bounds.CATEGORIES` order, so the loop must be run in full for
    the non-occupational cells to sit on the same random stream. That is why this
    takes ~10 minutes and why `--quick` exists as an explicitly-labelled fallback
    rather than a silent shortcut."""
    com = json.load(open(ADDENDUM, encoding="utf-8"))
    if quick:
        out = {}
        for kind in ("FA", "F2"):
            cell = com["table"][CATEGORY][kind]
            out[kind] = {"count": int(cell["count"]), "value": float(cell["value"]),
                         "fraction": cell["fraction"],
                         "orientation": cell["orientation"],
                         "V": np.asarray(cell["witness_V"], dtype=float),
                         "recomputed": False,
                         "witness_max_abs_diff_vs_committed": 0.0}
        return out, com

    bud = {"anneal_restarts": 256, "anneal_steps": 1200,
           "hinge_restarts": 512, "hinge_steps": 1200,
           "hop_restarts": 256, "hop_steps": 960,
           "polish": 1200, "hops": 2,
           "margin_ladder": list(rab.MARGIN_LADDER)}
    coords, _prov = isb.load_raw()
    scaling = isb.DEFAULT_SCALING
    rng = np.random.default_rng(ADDENDUM_SEED + 7)
    maps = {}
    for kind in ("F1", "F2", "FA"):
        U, desc = isb.features(coords, kind, scaling)
        maps[kind] = {"U": U, "d": len(next(iter(U.values()))), "desc": desc}

    out = {}
    for key, cat in isb.CATEGORIES.items():
        B = cat["members"]
        N = [n for n in isb.HUMANS if n not in B]
        n_tw = 2 * len(B) * len(N)
        for kind in ("F1", "F2", "FA"):
            U, d = maps[kind]["U"], maps[kind]["d"]
            PHI, y = add.per_twin_design(U, B, N)
            PA, yA = add.agent_only_design_U(U, B, N)
            grp = [0, 1]
            isb.bound(U, B, N, "pair")                    # same call order
            tw = isb.bound(U, B, N, "twin")
            Cw = tw["_c"].reshape(d, d)
            seeds = [np.stack([Cw.ravel(), -Cw.ravel()])]
            A = ((Cw - Cw.T) / 2.0).ravel()
            seeds.append(np.stack([A, -A]))
            seeds = rab.with_rotations(seeds)
            sw_seeds = rab.with_rotations([]) + [rng.standard_normal((2, d))
                                                 for _ in range(24)]
            _cw, Vw, _lg = rab.search_cell(PA, yA, 2, grp, sw_seeds, bud,
                                           ADDENDUM_SEED + 11 * d)
            lift = np.zeros((2, d * d))
            e0 = np.eye(d)[0]
            for m in range(2):
                lift[m] = np.outer(Vw[m], e0).ravel()
            seeds = seeds + rab.with_rotations([lift])
            cnt, V, _slog = rab.search_cell(PHI, y, 2, grp, seeds, bud,
                                            ADDENDUM_SEED + 13 * d)
            acc, orient, _n = rab.hard_acc(V, PHI, y, grp)
            print("[recompute] %-16s %-2s  %d/%d = %.4f (%s)"
                  % (key, kind, cnt, n_tw, acc, orient), flush=True)
            if key == CATEGORY and kind in ("FA", "F2"):
                ref = com["table"][key][kind]
                out[kind] = {
                    "count": int(cnt), "value": float(acc),
                    "fraction": "%d/%d" % (cnt, n_tw), "orientation": orient,
                    "V": np.asarray(V, dtype=float), "recomputed": True,
                    "witness_max_abs_diff_vs_committed": float(np.abs(
                        np.asarray(V, dtype=float)
                        - np.asarray(ref["witness_V"], dtype=float)).max()),
                    "committed_fraction": ref["fraction"],
                    "committed_value": float(ref["value"]),
                }
    return out, com


def witness_C(cell):
    """M = 2 argmax is sign(<V_1 - V_0, phi>) under the `identity` orientation and
    sign(<V_0 - V_1, phi>) under `shift-by-one` -- `readout_arity_bounds.hard_acc`'s
    two cases, made explicit so the exported C predicts the same labels the bound
    counted."""
    V = np.asarray(cell["V"], dtype=float)
    C = V[1] - V[0] if cell["orientation"] == "identity" else V[0] - V[1]
    return C


def wazn_bound():
    """R-M, computed from scratch. doc 55 SS18.4: the featurizer is fit
    INVENTORY-WIDE, which is not leakage here because exp69 holds out PAIRINGS and
    not nouns (exp65 held out nouns, which is why exp65's rule was train-only)."""
    H = sorted(gen.HUMANS)
    assert sorted(H) == sorted(isb.HUMANS), \
        "generator inventory != instrument inventory -- population drift"
    ang = WZ.angles_for(H, isb.HUMANS)                    # fit on all 24, applied
    U = {n: isb._u1(ang[n]) for n in isb.HUMANS}
    B = [n for n in isb.HUMANS if n in WZ.FAAIL]
    N = [n for n in isb.HUMANS if n not in WZ.FAAIL]
    assert (len(B), len(N)) == (9, 15), \
        "doc 51 SS7 faail membership drift: |B|=%d |N|=%d" % (len(B), len(N))
    r = isb.with_null(U, B, N, "twin", 200, 68000, 48, 500, 6)
    return U, B, N, ang, r


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--quick", action="store_true",
                    help="use the committed witnesses instead of re-searching "
                         "(labelled in the output as recomputed=false)")
    a = ap.parse_args()
    t0 = time.time()

    if os.path.exists(a.out):
        raise RuntimeError("%s already exists -- Rule 3: this file is written "
                           "ONCE, before datagen, and never overwritten" % a.out)

    # ---------------- grade 1
    cells, com = recompute_addendum_cells(quick=a.quick)
    coords, prov_raw = isb.load_raw()
    U_fa, desc_fa = isb.features(coords, "FA", isb.DEFAULT_SCALING)
    U_f2, desc_f2 = isb.features(coords, "F2", isb.DEFAULT_SCALING)
    B_a = list(isb.CATEGORIES[CATEGORY]["members"])
    N_a = [n for n in isb.HUMANS if n not in set(B_a)]

    U_wz, B_b, N_b, ang_wz, rm = wazn_bound()
    print("[wazn] %s = %.4f  null %.4f +/- %.4f  pct %.1f"
          % (rm["fraction"], rm["value"], rm["null"]["mean"], rm["null"]["sd"],
             rm["null"]["percentile_of_category"]), flush=True)

    sb = json.load(open(SB_BOUNDS, encoding="utf-8"))
    sb_cell = sb["target_selection"]["table"][CATEGORY]

    grade1 = {}

    # R-A -------------------------------------------------------------------
    c = cells["FA"]
    rows_a = twin_design(U_fa, B_a, N_a)
    C_a = witness_C(c)
    marg_a = margins_from_witness(rows_a, C_a)
    n_pos_a = sum(1 for m in marg_a if m > 0)
    assert n_pos_a == c["count"], \
        ("exported R-A witness certifies %d twins, the bound counted %d -- the "
         "witness is not the object the bound was computed from"
         % (n_pos_a, c["count"]))
    assert (c["count"], len(rows_a)) == COMMITTED_FRACTION["R-A"], \
        ("R-A bound %d/%d != doc 55 SS12's committed 154/160"
         % (c["count"], len(rows_a)))
    grade1["R-A"] = {
        "arms": [k for k, v in ARM_FAMILY.items() if v == "R-A"],
        "feature_map": "FA", "definition": desc_fa, "d_per_referent": 3,
        "d_joint": 9,
        "bound_value": c["value"], "fraction": c["fraction"],
        "count": c["count"], "denominator": len(rows_a),
        "method": ("CONSTRUCTIVE LOWER BOUND -- readout_arity_bounds's three-stage "
                   "search (annealed logsumexp surrogate, latent multiclass hinge "
                   "over a 4-margin ladder, hard-count polish, plus basin hops), "
                   "SEEDED with the committed per-twin witness so it can never "
                   "return less than the committed value"),
        "exact": False,
        "source": "diagnostic_readout_arity_bounds_addendum.json",
        "recomputed_here": bool(c["recomputed"]),
        "witness_max_abs_diff_vs_committed":
            c.get("witness_max_abs_diff_vs_committed"),
        "supersession_note": (
            "doc 54 SS17.7: this bound moved 0.9250 -> 0.9625 AFTER exp68 reported, "
            "and no stored boolean moved with it, because doc 54 SS7 / doc 55 SS6 read "
            "no criterion against any bound. The superseded 0.9625 is what doc 55 "
            "SS12 carries and what exp69 inherits."),
        "null_context_from_diagnostic_sb_bounds": sb_cell["FA"],
        "note_pre_supersession": ("the null percentile 89.0 in that block is stated "
                                  "against the PRE-supersession 0.9250; it is quoted "
                                  "only for doc 55 SS19.2's grade-3 below-null test, "
                                  "which a supersession UPWARD cannot flip"),
    }

    # R-W -------------------------------------------------------------------
    c = cells["F2"]
    rows_w = twin_design(U_f2, B_a, N_a)
    C_w = witness_C(c)
    marg_w = margins_from_witness(rows_w, C_w)
    n_pos_w = sum(1 for m in marg_w if m > 0)
    assert n_pos_w == c["count"], \
        "exported R-W witness certifies %d twins, the bound counted %d" \
        % (n_pos_w, c["count"])
    assert (c["count"], len(rows_w)) == COMMITTED_FRACTION["R-W"], \
        "R-W bound %d/%d != doc 55 SS12's committed 160/160" % (c["count"], len(rows_w))
    grade1["R-W"] = {
        "arms": [k for k, v in ARM_FAMILY.items() if v == "R-W"],
        "feature_map": "F2", "definition": desc_f2, "d_per_referent": 9,
        "d_joint": 81,
        "bound_value": c["value"], "fraction": c["fraction"],
        "count": c["count"], "denominator": len(rows_w),
        "method": ("CONSTRUCTIVE LOWER BOUND at d = 9 (same three-stage search, "
                   "same budget). doc 55 SS12: 'the register bound at d = 9 is a "
                   "constructive lower bound from a budgeted search -- adequate for "
                   "design feasibility, NEVER for a criterion' (doc 54 SS16 item 4)."),
        "exact": False,
        "source": ("diagnostic_sb_bounds.json, confirmed unchanged by "
                   "diagnostic_readout_arity_bounds_addendum.json"),
        "recomputed_here": bool(c["recomputed"]),
        "witness_max_abs_diff_vs_committed":
            c.get("witness_max_abs_diff_vs_committed"),
        "did_not_move_under_the_stronger_search": True,
        "null_context_from_diagnostic_sb_bounds": sb_cell["F2"],
    }

    # R-M -------------------------------------------------------------------
    rows_m = twin_design(U_wz, B_b, N_b)
    r_m = isb.bound(U_wz, B_b, N_b, "twin")
    C_m = np.asarray(r_m["_c"], dtype=float)
    marg_m = margins_from_witness(rows_m, C_m)
    n_pos_m = sum(1 for m in marg_m if m > 0)
    assert n_pos_m == r_m["count"], \
        "exported R-M witness certifies %d twins, the bound counted %d" \
        % (n_pos_m, r_m["count"])
    assert (r_m["count"], len(rows_m)) == COMMITTED_FRACTION["R-M"], \
        "R-M bound %d/%d != doc 55 SS19.2's registered 270/270" \
        % (r_m["count"], len(rows_m))
    assert abs(rm["null"]["mean"] - REGISTERED_RM["null_mean"]) < 5e-4 and \
        abs(rm["null"]["sd"] - REGISTERED_RM["null_sd"]) < 5e-4 and \
        rm["null"]["percentile_of_category"] == REGISTERED_RM["percentile"], \
        ("R-M null stats %r do not reproduce doc 55 SS19.2's registered "
         "0.7491 +/- 0.0640 at the 100th percentile -- STOP and check the "
         "featurizer fit population before freezing" % (rm["null"],))
    grade1["R-M"] = {
        "arms": [k for k, v in ARM_FAMILY.items() if v == "R-M"],
        "feature_map": "WZ-positional",
        "definition": ("theta = exp65_featurizer's positional n-gram + PCA angle "
                       "(atan2 of the 2-D projection, unit-normalised); "
                       "u = (1, cos theta, sin theta). d = 3."),
        "d_per_referent": 3, "d_joint": 9,
        "bound_value": rm["value"], "fraction": rm["fraction"],
        "count": r_m["count"], "denominator": len(rows_m),
        "method": r_m["method"], "exact": bool(r_m["exact"]),
        "source": ("computed HERE, 2026-08-25, from instrument_sb_bounds + "
                   "exp65_featurizer -- doc 55 SS19.2's registered value"),
        "recomputed_here": True,
        "registered_value_doc55_SS19_2": REGISTERED_RM,
        "reproduces_registered": True,
        "null": rm["null"],
        "featurizer_fit_protocol": (
            "doc 55 SS18.4: fit INVENTORY-WIDE over all 24 humans and applied to "
            "all 24. This is not leakage in exp69's setting because exp69 holds out "
            "PAIRINGS, not nouns -- doc 54's raw-PCA precedent. exp65's train-only "
            "rule existed because exp65 held out NOUNS. Stated to preempt the doc-42 "
            "failure class."),
        "membership_source": ("exp65_featurizer.FAAIL -- doc 51 SS7, user-confirmed; "
                              "imported, never re-typed"),
        "bearers": B_b, "nonbearers": N_b,
    }

    # ---------------- grade 2
    cv = json.load(open(CAPVAL, encoding="utf-8"))
    conv = cv["conversion_by_bound_population"]["trained_population / K1_and_width"]
    median = float(conv["median"])
    assert abs(median - 0.997) < 5e-3, \
        "conversion median %.6f is not doc 55 SS19.2's 0.997" % median
    grade2 = {
        "conversion_median": median,
        "conversion_median_source": ("diagnostic_capacity_validation.json :: "
                                     "conversion_by_bound_population :: "
                                     "'trained_population / K1_and_width'"),
        "conversion_median_n_rows": int(conv["n"]),
        "conversion_median_range": [float(conv["min"]), float(conv["max"])],
        "interval_minimum": INTERVAL_MIN,
        "interval_provenance": ("0.06 ~ 2 x the binomial SD at n_test = 300 "
                                "(sqrt(0.25/300) = 0.02887) -- the bar exp61 SS8 / "
                                "exp67 SS4 / doc 54 SS7 used, inherited, and the "
                                "reason exp69's n_test is deliberately 300"),
        "extrapolation_under_test": (
            "doc 55 SS19.2: the median comes from n = %d rows in ONE task family "
            "(the single-RY K = 1 / width population) and exp69 is a NEW task "
            "family (two composed events, verb-indexed query). THE EXTRAPOLATION "
            "IS ITSELF UNDER TEST. A miss refutes the EXTRAPOLATION, not the bound, "
            "and may not be reported as the calculus failing."
            % int(conv["n"])),
        "by_arm": {},
    }
    for arm, fam in ARM_FAMILY.items():
        b = grade1[fam]["bound_value"]
        p = b * median
        grade2["by_arm"][arm] = {
            "family": fam, "bound": b, "point_prediction": p,
            "interval": [p - INTERVAL_MIN, p + INTERVAL_MIN],
            "interval_clipped_to_unit": [max(0.0, p - INTERVAL_MIN),
                                         min(1.0, p + INTERVAL_MIN)],
            "note": ("prediction is for HELD-OUT accuracy on exp69's own draw; the "
                     "criterion in doc 55 SS6 is NOT read against it (SS6 is "
                     "measurement-vs-measurement throughout)"),
        }

    # ---------------- grade 3
    pcts = {"R-A": float(sb_cell["FA"]["percentile"]),
            "R-W": float(sb_cell["F2"]["percentile"]),
            "R-M": float(rm["null"]["percentile_of_category"])}
    below = {k: bool(v < 50.0) for k, v in pcts.items()}
    grade3 = {
        "flag": "under-conversion predicted for a BELOW-NULL category (the F1 "
                "observation, made prospective)",
        "percentile_vs_size_matched_null": pcts,
        "below_null": below,
        "any_arm_below_null": bool(any(below.values())),
        "testable_this_run": bool(any(below.values())),
        "verdict": ("UNTESTABLE THIS RUN. doc 55 SS19.2 computed this before "
                    "freezing and recorded it rather than dropping it: no exp69 arm "
                    "has a below-null category (R-A %.1fth percentile, R-W %.1fth, "
                    "R-M %.1fth), so the grade-3 flag has nothing to fire on. It is "
                    "recorded as untestable, NOT as confirmed."
                    % (pcts["R-A"], pcts["R-W"], pcts["R-M"])),
    }

    # ---------------- SS19.3 witness export
    tw_a, pr_a = pack_margins(rows_a, marg_a)
    tw_w, pr_w = pack_margins(rows_w, marg_w)
    tw_m, pr_m = pack_margins(rows_m, marg_m)

    # rank agreement between the instrument's FA space and the runtime FA space
    import exp42_compiler as comp                       # noqa: E402
    frozen_ang, _emb_meta, emb_sha = comp.load_embeddings()
    U_fa_rt = {n: isb._u1(float(frozen_ang[n])) for n in isb.HUMANS}
    rows_a_rt = twin_design(U_fa_rt, B_a, N_a)
    marg_a_rt = margins_from_witness(rows_a_rt, C_a)
    o1 = np.argsort(np.argsort(np.asarray(marg_a)))
    o2 = np.argsort(np.argsort(np.asarray(marg_a_rt)))
    sp = float(np.corrcoef(o1, o2)[0, 1])

    witnesses = {
        "convention": (
            "doc 55 SS19.3: the MAX-FS witness C is NOT unique, so the registered "
            "convention is the COMMITTED SEARCH'S OUTPUT AT ITS REGISTERED SEED "
            "(%d for R-A / R-W via readout_arity_bounds_addendum.py; 0 for R-M via "
            "instrument_sb_bounds.bound), exported at bound-computation time -- "
            "i.e. in this file, BEFORE the dataset exists -- with margins computed "
            "ONCE and frozen." % ADDENDUM_SEED),
        "margin_formula": ("margin(twin) = s * <C, phi> / (||C|| ||phi||) with "
                           "phi = vec(u(agent) (x) u(patient)) AGENT-FIRST and "
                           "s = +1 for the YES twin, -1 for the NO twin. Scale-free, "
                           "so the SS19.3 Spearman is invariant to the witness's "
                           "normalisation."),
        "pair_margin_rule": "min over the pair's two twins",
        "registered_analysis": (
            "doc 55 SS19.3: Spearman rank correlation between witness margin and "
            "PER-ITEM held-out accuracy, per arm. DESCRIPTIVE, with direction "
            "(positive expected). At 150 test pairs per arm this has real power, "
            "unlike the seed-level descriptives. NO criterion reads it."),
        "registered_caveat": (
            "R-W's witness comes from a LOWER-BOUND search at d = 9, so its margins "
            "are noisier BY CONSTRUCTION than the d = 3 arms' (doc 55 SS19.3)."),
        "R-A": {"C": [float(x) for x in np.asarray(C_a).ravel()],
                "orientation": cells["FA"]["orientation"],
                "n_twins": len(rows_a), "n_positive_margin": n_pos_a,
                "certifies_the_bound_count": True,
                "per_twin": tw_a, "per_pair": pr_a,
                "feature_space_note": (
                    "computed in the INSTRUMENT's FA space (atan2 of the NORMALISED "
                    "coordinates), which is the space the witness was searched in. "
                    "The RUNTIME FA angle (doc 54 SS6.2: atan2 of the UNNORMALISED "
                    "components, i.e. exp42_noun_embeddings) differs by <= ~7.9e-02 "
                    "rad and preserves the referents' cyclic order, so the d = 3 "
                    "BOUND is identical either way. Rank agreement between the two "
                    "margin orderings is measured below."),
                "runtime_space_rank_agreement_pearson_of_ranks": sp,
                "runtime_embeddings_sha256": emb_sha},
        "R-W": {"C": [float(x) for x in np.asarray(C_w).ravel()],
                "orientation": cells["F2"]["orientation"],
                "n_twins": len(rows_w), "n_positive_margin": n_pos_w,
                "certifies_the_bound_count": True,
                "per_twin": tw_w, "per_pair": pr_w,
                "feature_space_note": ("F2's instrument features ARE the runtime "
                                       "features (theta_k = (p_k + 1) pi, outer "
                                       "product), so no space mismatch exists here")},
        "R-M": {"C": [float(x) for x in np.asarray(C_m).ravel()],
                "orientation": "instrument sign convention (maxfs_lower returns c "
                               "oriented so that satisfied rows are the positives)",
                "n_twins": len(rows_m), "n_positive_margin": n_pos_m,
                "certifies_the_bound_count": True,
                "per_twin": tw_m, "per_pair": pr_m,
                "feature_space_note": ("the wazn angle is the same quantity at "
                                       "bound time and at runtime")},
    }

    payload = {
        "component": "exp69_predictions",
        "spec": SPEC,
        "ordering_rule": (
            "doc 55 SS19.2 / SS19.7: this file is written BEFORE exp69_datagen.py "
            "runs. exp69_datagen.py asserts its existence, so the ordering is "
            "mechanised. Rule 4's strong form: the predictions cannot be fitted to "
            "a dataset that does not yet exist."),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script_sha256": _sha(os.path.abspath(__file__)),
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__,
        "provenance": {
            "instrument_sb_bounds_sha256": _sha(
                os.path.join(HERE, "instrument_sb_bounds.py")),
            "readout_arity_bounds_sha256": _sha(
                os.path.join(HERE, "readout_arity_bounds.py")),
            "readout_arity_bounds_addendum_sha256": _sha(
                os.path.join(HERE, "readout_arity_bounds_addendum.py")),
            "exp65_featurizer_sha256": _sha(os.path.join(HERE, "exp65_featurizer.py")),
            "addendum_json_sha256": _sha(ADDENDUM),
            "sb_bounds_json_sha256": _sha(SB_BOUNDS),
            "capacity_validation_json_sha256": _sha(CAPVAL),
            "raw_family": prov_raw,
            "addendum_search_seed": ADDENDUM_SEED,
            "recompute_mode": ("RECOMPUTED (full four-category loop, so the "
                               "non-occupational cells sit on the addendum's own "
                               "random stream)" if not a.quick
                               else "COMMITTED WITNESSES REUSED (--quick)"),
        },
        "arm_families": ARM_FAMILY,
        "category": {"key": CATEGORY,
                     "gloss": isb.CATEGORIES[CATEGORY]["gloss"],
                     "members": B_a,
                     "source": ("instrument_sb_bounds.CATEGORIES -- doc 54 SS5.2, "
                                "user-confirmed 2026-08-23; imported, never re-typed")},
        "grade1_bounds": grade1,
        "grade2_point_predictions": grade2,
        "grade3_under_conversion_flag": grade3,
        "witnesses_doc55_SS19_3": witnesses,
        "bounds_are_context_only": (
            "doc 55 SS6 / SS12: NO criterion in doc 55 SS6 is read against any number "
            "in this file. doc 54 SS17.7 is the standing reason -- a bound moved by "
            "0.0375 AFTER exp68 reported and no stored boolean moved with it. These "
            "predictions are a PROSPECTIVE TEST OF THE INSTRUMENT, graded, and they "
            "arm doc 55 SS9.2's bug-hunt row (any arm exceeding its bound by > 0.06). "
            "They are not criteria and may not be promoted onto one."),
        "wall_sec": time.time() - t0,
    }
    json.dump(payload, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n[exp69-pred] wrote %s (%.0fs)" % (a.out, payload["wall_sec"]))
    for fam in ("R-A", "R-W", "R-M"):
        g = grade1[fam]
        print("  %-4s bound %.4f (%s, %s)  arms %s"
              % (fam, g["bound_value"], g["fraction"],
                 "exact" if g["exact"] else "constructive", g["arms"]))
    print("  grade-2 median conversion %.4f (n=%d)" % (median, int(conv["n"])))
    for arm in ("A0", "R1-A", "R1-W", "R2-A", "R2-W", "R1-M", "R2-M"):
        p = grade2["by_arm"][arm]
        print("    %-5s point %.4f  interval [%.4f, %.4f]"
              % (arm, p["point_prediction"], p["interval"][0], p["interval"][1]))
    print("  grade-3: %s" % ("TESTABLE" if grade3["testable_this_run"]
                             else "UNTESTABLE (no below-null arm)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
