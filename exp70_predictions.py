# -*- coding: utf-8 -*-
"""exp70_predictions.py -- doc 56 SS9: the three prediction grades, FROZEN, and the
SS15.3 witness/margin export, written BEFORE any exp70 run exists.

WHAT THIS FILE IS, AND WHAT IT IS NOT
=====================================
doc 56 SS9.1 states that every grade-1 number is ALREADY frozen, in the committed
`diagnostic_entangled_bounds.json` produced by the committed
`instrument_entangled_bounds.py` under doc 54 SS0's spent instrument exception. This
file therefore INVENTS NOTHING. It does three things:

  1. COPIES BY RECOMPUTATION. Every grade-1 bound, null and witness is recomputed
     here from the committed instrument's own machinery (`new_maps`, `bound_joint`,
     `bound_with_null`, `BUDGET`, `NULL_SEED`) and ASSERTED to reproduce the
     committed diagnostic and doc 56 SS10's stated table. A copy that is merely
     read across would not catch a drift between the doc and the artefact; a copy
     that is recomputed does. Nothing committed is modified or overwritten.

  2. EXPORTS THE SS15.3 MARGINS. The committed diagnostic exports the witness
     VECTORS (`witnesses_doc55_SS19_3`) but not the per-twin / per-pair MARGINS,
     and doc 56 SS15.3's registered analysis -- Spearman(witness margin, per-item
     held-out accuracy), per arm -- is item-level. The margins are computed here,
     ONCE, from the committed witnesses, over the instrument's own row order, and
     frozen. The re-pinning doc 56 SS15.3 insists on is honoured: Phi is THE MAP'S
     OWN joint feature, which for P-eps and N-in is the outer product of
     per-referent vectors (81- and 36-dim) and for N-across / EA-ctrl is the
     genuinely joint 10-coordinate vector with NO per-referent factorisation.

  3. STATES GRADES 2 AND 3 with their caveats attached, per doc 56 SS9.2 / SS9.3.

BOUND OF RECORD, and the one place it differs from what this file computes
-------------------------------------------------------------------------
doc 56 SS10 note 1: for N-across and EA-ctrl the bound OF RECORD is the committed FA
family value **0.9625 (154/160)** from `diagnostic_readout_arity_bounds_addendum
.json`'s three-stage search (doc 54 SS17.7's supersession). The instrument's own
TWO-stage search returns the pre-supersession **0.9250 (148/160)**, and that value is
carried here ONLY as a consistency check, never as a second bound. Both are lower
bounds; the higher governs. The NULL and PERCENTILE for those two rows are stated at
the METHOD-MATCHED two-stage value, because no three-stage null exists and pairing a
three-stage bound with a two-stage null would be doc 47 SS16.1's unmatched search
budget occurring inside the diagnostic itself.

A0-W's family reuses exp69's committed R-W witness, and that reuse is ASSERTED, not
assumed: A0-W's feature map is byte-identical to exp69's F2 map (theta_k = (p_k + 1)
pi, outer product), and this file checks that element by element before importing
the witness.

NO CRITERION READS ANY NUMBER IN THIS FILE. doc 56 SS6's opening sentence and doc 54
SS17.7 are the standing reason: a bound moved by 0.0375 after exp68 reported and no
stored boolean moved with it. These are a PROSPECTIVE TEST OF THE INSTRUMENT,
graded, plus the arming of doc 56 SS6.6's bug-hunt row.

    python exp70_predictions.py
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

import instrument_sb_bounds as isb                        # noqa: E402
import capacity_check as cc                               # noqa: E402
import instrument_entangled_bounds as IEB                 # noqa: E402

SPEC = ("qnlp_private_docs/56_exp70_entangled_encodings.md (566b9ce, REGISTERED "
        "2026-08-25 with orchestrator rulings 1-7 in the status header)")
SPEC_COMMIT = "566b9ce"
OUT = os.path.join(HERE, "exp70_predictions.json")

DIAG = os.path.join(HERE, "diagnostic_entangled_bounds.json")
DEGEN = os.path.join(HERE, "diagnostic_encoding_entanglement.json")
CAPVAL = os.path.join(HERE, "diagnostic_capacity_validation.json")
ADDENDUM = os.path.join(HERE, "diagnostic_readout_arity_bounds_addendum.json")
EXP69_PRED = os.path.join(HERE, "exp69_predictions.json")

CATEGORY = IEB.TARGET                  # "nonoccupational", doc 54 SS5.2
INTERVAL_MIN = 0.06                    # doc 56 SS9.2 / SS6.4: 2 x binomial SD at 300
CONV_MEDIAN = 0.996875                 # doc 56 SS9.2, asserted against the artefact

# doc 56 SS5.2's arms, in registration order. `family` selects the feature map, `T`
# the composition depth. Registered names use ASCII "eps" for "P-eps" throughout
# the code; the doc writes it P-epsilon.
FAMILIES = ["A0-W", "EA-ctrl", "P-eps", "N-in", "N-across"]
ARMS = ["%s-T%d" % (f, t) for f in FAMILIES for t in (1, 2)]
ARM_FAMILY = {a: a.rsplit("-T", 1)[0] for a in ARMS}

# doc 56 SS10, the bounds OF RECORD. Read note 1 in this module's docstring before
# touching the two 0.9625 rows.
BOUND_OF_RECORD = {
    "A0-W": (1.0000, "160/160", "committed F2 (diagnostic_sb_bounds.json, "
                                "reproduced through the extended path)"),
    "EA-ctrl": (0.9625, "154/160", "committed FA three-stage addendum "
                                   "(diagnostic_readout_arity_bounds_addendum.json)"),
    "N-across": (0.9625, "154/160", "committed FA three-stage addendum "
                                    "(diagnostic_readout_arity_bounds_addendum.json)"),
    "P-eps": (1.0000, "160/160", "computed by instrument_entangled_bounds.py "
                                 "(constructive lower bound, no prior exists)"),
    "N-in": (1.0000, "160/160", "computed by instrument_entangled_bounds.py "
                                "(constructive lower bound, no prior exists)"),
}

# doc 56 SS10's table, asserted rather than quoted (twin mode).
DOC56_S10 = {
    "P-eps": {"bound": 1.0000, "fraction": "160/160", "d_joint": 81,
              "null_mean": 0.9429, "null_sd": 0.0403, "percentile": 92.5},
    "N-in": {"bound": 1.0000, "fraction": "160/160", "d_joint": 36,
             "null_mean": 0.8991, "null_sd": 0.0572, "percentile": 98.0},
    "N-across": {"bound": 0.9250, "fraction": "148/160", "d_joint": 10,
                 "null_mean": 0.7940, "null_sd": 0.0795, "percentile": 92.0},
    "EA-ctrl": {"bound": 0.9250, "fraction": "148/160", "d_joint": 10,
                "null_mean": 0.7938, "null_sd": 0.0798, "percentile": 92.5},
}

# doc 56 SS15.3's witness table.
DOC56_S15_3 = {"P-eps": (81, 160, 160), "N-in": (36, 160, 160),
               "N-across": (10, 160, 148), "EA-ctrl": (10, 160, 148)}

# doc 56 SS3 / SS4.4 / SS7.3, the entanglement distribution. Asserted.
DOC56_ENTANGLEMENT = {
    "e_in_eps_map_min": 0.1898, "e_in_eps_map_below_0.10": 0,
    "e_in_oldmap_below_0.05": 4,
    "e_across_conc_min": 0.0884, "e_across_conc_median": 0.4019,
    "e_across_conc_max": 0.8244, "e_across_casts_below_0.10": 4,
    "e_across_purity_min": 0.6602, "e_across_purity_median": 0.9192,
    "e_across_purity_max": 0.9961,
}


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------------------ margins
def twin_rows(m, casts):
    """The instrument's own twin-design rows, in ITS order, keyed for the join.

    `instrument_entangled_bounds.joint_rows(m, casts, "twin")` emits, per cast
    (b, y): +Phi(b, y) then -Phi(y, b). That is exactly (yes-twin, no-twin) with
    the agent-first convention, so the key for the join with a held-out item is
    (agent of the queried event, patient of the queried event).
    """
    rows = []
    for b, y in casts:
        rows.append({"cast": [b, y], "agent": b, "patient": y, "y": 1,
                     "phi": np.asarray(m.phi(b, y), dtype=float)})
        rows.append({"cast": [b, y], "agent": y, "patient": b, "y": 0,
                     "phi": np.asarray(m.phi(y, b), dtype=float)})
    return rows


def margins_from_witness(rows, C):
    """doc 56 SS15.3: margin(twin) = s <W, Phi> / (||W|| ||Phi||), s = +1 on the
    YES twin and -1 on the NO twin. Scale-free, so the Spearman is invariant to
    however `maxfs_lower` happened to normalise the witness."""
    W = np.asarray(C, dtype=float).ravel()
    W = W / max(float(np.linalg.norm(W)), 1e-300)
    out = []
    for r in rows:
        p = r["phi"]
        p = p / max(float(np.linalg.norm(p)), 1e-300)
        out.append(float((1.0 if r["y"] == 1 else -1.0) * float(W @ p)))
    return out


def pack_margins(rows, marg):
    """Per-twin margins keyed 'agent|patient'; per-PAIR margin = min over the
    pair's two twins (a cast is only as reachable as its harder twin)."""
    per_twin, per_pair = {}, {}
    for r, m in zip(rows, marg):
        per_twin["%s|%s" % (r["agent"], r["patient"])] = m
    for r in rows:
        b, y = r["cast"]
        per_pair["%s|%s" % (b, y)] = min(per_twin["%s|%s" % (b, y)],
                                         per_twin["%s|%s" % (y, b)])
    return per_twin, per_pair


# ----------------------------------------------------------------- the maps
def a0w_map(coords):
    """A0-W's family map: exp68/exp69's F2 register on the OLD map (p + 1) pi,
    routed through the instrument's product path (`sensitivity_maps` item 3)."""
    for m in IEB.sensitivity_maps(coords):
        if m.name == "A0-W":
            return m
    raise RuntimeError("instrument_entangled_bounds.sensitivity_maps lost A0-W")


def assert_a0w_is_exp69_F2(m, coords):
    """A0-W may only inherit exp69's committed R-W witness if its feature map is
    the SAME map. Checked element by element, never assumed."""
    U_f2, desc = isb.features(coords, "F2", isb.DEFAULT_SCALING)
    worst = max(float(np.abs(np.asarray(U_f2[n], dtype=float)
                             - np.asarray(m.per_referent[n], dtype=float)).max())
                for n in isb.HUMANS)
    assert worst == 0.0, \
        ("A0-W's instrument map differs from exp68/exp69's committed F2 map by "
         "%.3e -- exp69's R-W witness may NOT be inherited" % worst)
    return {"max_abs_diff_vs_instrument_sb_bounds_F2": worst,
            "exp69_F2_definition": desc}


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--quick", action="store_true",
                    help="skip the size-matched null recomputation and carry the "
                         "committed nulls (labelled in the output)")
    a = ap.parse_args()
    t0 = time.time()

    if os.path.exists(a.out):
        raise RuntimeError("%s already exists -- Rule 3: this file is written ONCE, "
                           "BEFORE any exp70 run, and never overwritten" % a.out)

    diag = json.load(open(DIAG, encoding="utf-8"))
    assert diag["script_sha256"] == _sha(
        os.path.join(HERE, "instrument_entangled_bounds.py")), \
        ("diagnostic_entangled_bounds.json was produced by a DIFFERENT "
         "instrument_entangled_bounds.py than the one on disk -- doc 56 SS9.1's "
         "frozen numbers cannot be copied by recomputation")

    coords, prov_raw = isb.load_raw()
    B = list(isb.CATEGORIES[CATEGORY]["members"])
    N = [n for n in isb.HUMANS if n not in set(B)]
    casts, _nons = cc.casts_from_bearers(isb.HUMANS, B)
    assert len(casts) == 80, "cast budget drifted: %d" % len(casts)

    maps = {m.name: m for m in IEB.new_maps(coords, fa_space="runtime")}
    maps["A0-W"] = a0w_map(coords)
    a0w_identity = assert_a0w_is_exp69_F2(maps["A0-W"], coords)

    # ------------------------------------------------------ grade 1, recomputed
    grade1, witnesses = {}, {}
    for fam in FAMILIES:
        m = maps[fam]
        r = IEB.bound_joint(m, casts, "twin", **IEB.BUDGET)
        W = np.asarray(r["_c"], dtype=float)
        rows = twin_rows(m, casts)
        marg = margins_from_witness(rows, W)
        n_pos = sum(1 for x in marg if x > 0)
        assert n_pos == r["count"], \
            ("%s: the exported witness certifies %d twins but the bound counted %d "
             "-- the witness is not the object the bound was computed from"
             % (fam, n_pos, r["count"]))

        pub = IEB._public(r)
        # ---- copy-by-recomputation against the COMMITTED diagnostic
        committed = None
        if fam in diag["grade1_bounds"]:
            committed = diag["grade1_bounds"][fam]["twin"]
            assert committed["count"] == r["count"] \
                and committed["fraction"] == r["fraction"], \
                ("%s twin bound %s does not reproduce the committed %s"
                 % (fam, r["fraction"], committed["fraction"]))
            cw = diag["witnesses_doc55_SS19_3"][fam]
            dw = float(np.abs(np.asarray(cw["W"], dtype=float) - W).max())
            assert dw == 0.0, \
                ("%s witness differs from the committed export by %.3e" % (fam, dw))
            assert cw["n_positive_margin"] == n_pos, \
                ("%s positive-margin count %d != committed %d"
                 % (fam, n_pos, cw["n_positive_margin"]))
            d, nt, npos = DOC56_S15_3[fam]
            assert (len(W), len(rows), n_pos) == (d, nt, npos), \
                ("%s witness (%d, %d, %d) != doc 56 SS15.3's (%d, %d, %d)"
                 % (fam, len(W), len(rows), n_pos, d, nt, npos))
            s10 = DOC56_S10[fam]
            assert abs(r["bound"] - s10["bound"]) < 5e-5 \
                and r["fraction"] == s10["fraction"] \
                and r["d_joint"] == s10["d_joint"], \
                "%s does not reproduce doc 56 SS10's row" % fam

        null = None
        if committed is not None:
            if a.quick:
                null = dict(committed["null"])
                null["recomputed_here"] = False
            else:
                rn = IEB.bound_with_null(m, B, "twin", draws=IEB.NULL_DRAWS,
                                         null_seed=IEB.NULL_SEED, **IEB.BUDGET)
                null = dict(rn["null"])
                null["recomputed_here"] = True
                for k in ("mean", "sd", "percentile_of_category"):
                    assert abs(null[k] - committed["null"][k]) < 1e-9, \
                        ("%s null.%s %.6f does not reproduce the committed %.6f"
                         % (fam, k, null[k], committed["null"][k]))
                s10 = DOC56_S10[fam]
                assert abs(null["mean"] - s10["null_mean"]) < 5e-5 \
                    and abs(null["sd"] - s10["null_sd"]) < 5e-5 \
                    and abs(null["percentile_of_category"]
                            - s10["percentile"]) < 1e-9, \
                    "%s null does not reproduce doc 56 SS10's row" % fam

        bor, bor_frac, bor_src = BOUND_OF_RECORD[fam]
        grade1[fam] = {
            "arms": [x for x in ARMS if ARM_FAMILY[x] == fam],
            "definition": m.desc,
            "feature_kind": m.kind, "d_per_referent": m.d_ref, "d_joint": m.d,
            "bound_of_record": bor, "fraction_of_record": bor_frac,
            "bound_of_record_source": bor_src,
            "recomputed_here": {
                "bound": r["bound"], "fraction": r["fraction"],
                "count": r["count"], "denominator": r["n_rows"],
                "method": r["method"], "exact": bool(r["exact"]),
                "sign_flip_check": r["sign_flip_check"],
                "swap_perm_max_abs_error": IEB.check_swap_perm(m, casts[:8]),
            },
            "reproduces_committed_diagnostic": bool(committed is not None),
            "null_size_matched": null,
            "note": (
                "the recomputed value is the instrument's TWO-stage search. For "
                "N-across / EA-ctrl the bound OF RECORD is the committed 0.9625 "
                "three-stage value (doc 54 SS17.7's supersession) and the 0.9250 "
                "here is a CONSISTENCY CHECK, never a second bound; the null and "
                "percentile are stated at the METHOD-MATCHED 0.9250 because no "
                "three-stage null exists (doc 56 SS10 note 1, doc 47 SS16.1)."
                if fam in ("N-across", "EA-ctrl") else
                "no committed prior bound exists for this map; the recomputed "
                "value IS the bound of record, a constructive lower bound."
                if fam in ("P-eps", "N-in") else
                "A0-W is the anchor family: exp68/exp69's committed F2 map and "
                "its committed 1.0000 (160/160), reproduced through the extended "
                "path. Its map identity with exp69's F2 is asserted, not assumed."),
        }
        tw, pr = pack_margins(rows, marg)
        witnesses[fam] = {
            "W": [float(x) for x in W], "d": int(len(W)),
            "n_twins": len(rows), "n_positive_margin": n_pos,
            "certifies_the_bound_count": True,
            "per_twin": tw, "per_pair": pr,
            "phi_definition": (
                "the outer product of per-referent vectors, agent-first "
                "(d_ref = %d, d_joint = %d)" % (m.d_ref, m.d)
                if m.kind == "product" else
                "the genuinely JOINT 10-coordinate vector (outer square of the "
                "4-dim real amplitude vector), agent-first. It has NO "
                "per-referent factorisation, which is exactly why doc 56 SS15.3 "
                "REPLACES rather than reuses exp69's 'agent-first outer product' "
                "phrasing for this family."),
        }
        print("[grade1] %-9s %-9s d=%-3d  witness %d/%d positive  null %s"
              % (fam, grade1[fam]["recomputed_here"]["fraction"], m.d, n_pos,
                 len(rows),
                 "%.4f +/- %.4f (pct %.1f)" % (null["mean"], null["sd"],
                                               null["percentile_of_category"])
                 if null else "n/a (anchor family)"), flush=True)

    # A0-W inherits exp69's committed R-W witness as a CROSS-CHECK on the same map
    ep = json.load(open(EXP69_PRED, encoding="utf-8"))
    rw = ep["witnesses_doc55_SS19_3"]["R-W"]
    rows_a0 = twin_rows(maps["A0-W"], casts)
    marg_69 = margins_from_witness(rows_a0, np.asarray(rw["C"], dtype=float))
    n_pos_69 = sum(1 for x in marg_69 if x > 0)
    tw69, pr69 = pack_margins(rows_a0, marg_69)
    o1 = np.argsort(np.argsort(np.asarray(
        [witnesses["A0-W"]["per_twin"][k] for k in sorted(tw69)])))
    o2 = np.argsort(np.argsort(np.asarray([tw69[k] for k in sorted(tw69)])))
    witnesses["A0-W"]["exp69_R-W_crosscheck"] = {
        "n_positive_margin": n_pos_69,
        "exp69_committed_n_positive_margin": rw["n_positive_margin"],
        "map_identity": a0w_identity,
        "rank_agreement_pearson_of_ranks": float(np.corrcoef(o1, o2)[0, 1]),
        "note": ("A0-W's map IS exp69's F2 map (asserted element-by-element "
                 "above), so exp69's committed R-W witness is a legitimate second "
                 "witness for the SAME reachable class. MAX-FS witnesses are not "
                 "unique, so the two need not agree pointwise; the rank agreement "
                 "is reported rather than asserted. exp70's witness of record for "
                 "A0-W is the one recomputed HERE, at doc 56's registered budget."),
    }

    # --------------------------------------------------------------- grade 2
    cv = json.load(open(CAPVAL, encoding="utf-8"))
    conv = cv["conversion_by_bound_population"]["trained_population / K1_and_width"]
    median = float(conv["median"])
    assert abs(median - CONV_MEDIAN) < 1e-9, \
        "conversion median %.9f is not doc 56 SS9.2's %.6f" % (median, CONV_MEDIAN)
    grade2 = {
        "conversion_median": median,
        "conversion_median_source": ("diagnostic_capacity_validation.json :: "
                                     "conversion_by_bound_population :: "
                                     "'trained_population / K1_and_width'"),
        "conversion_median_n_rows": int(conv["n"]),
        "conversion_median_range": [float(conv["min"]), float(conv["max"])],
        "interval_minimum": INTERVAL_MIN,
        "interval_provenance": ("0.06 ~ 2 x the binomial SD at n_test = 300 "
                                "(sqrt(0.25/300) = 0.02887), inherited verbatim "
                                "from exp61 SS8 / exp67 SS4 / doc 54 SS7 and the "
                                "reason exp70 keeps n_test = 300"),
        "extrapolation_under_test_TWO_PRECEDENT": (
            "doc 56 SS9.2. The median comes from n = %d rows in ONE task family "
            "(the single-RY K = 1 / width population). doc 55 SS20.4 recorded "
            "grade 2 holding for five of six exp69 arms and being REFUTED for "
            "R2-W -- the T = 2 PRODUCT-REGISTER arm, the same family P-eps-T2 and "
            "N-in-T2 belong to. A grade-2 miss on the four-wire T = 2 arms is "
            "therefore EXPECTED, is registered here as expected, and refutes the "
            "ONE-FAMILY EXTRAPOLATION rather than the bound." % int(conv["n"])),
        "by_arm": {},
    }
    for arm in ARMS:
        fam = ARM_FAMILY[arm]
        b = grade1[fam]["bound_of_record"]
        p = b * median
        grade2["by_arm"][arm] = {
            "family": fam, "bound_of_record": b, "point_prediction": p,
            "interval": [p - INTERVAL_MIN, p + INTERVAL_MIN],
            "interval_clipped_to_unit": [max(0.0, p - INTERVAL_MIN),
                                         min(1.0, p + INTERVAL_MIN)],
            "grade2_miss_expected": bool(fam in ("P-eps", "N-in")
                                         and arm.endswith("T2")),
            "note": ("prediction is for HELD-OUT accuracy on exp69's frozen Part A "
                     "draw; NO criterion in doc 56 SS6 is read against it"),
        }

    # --------------------------------------------------------------- grade 3
    g3c = diag["grade3_under_conversion_flag"]
    pcts = dict(g3c["percentile_vs_size_matched_null"])
    for fam in ("P-eps", "N-in", "N-across", "EA-ctrl"):
        assert abs(pcts[fam] - DOC56_S10[fam]["percentile"]) < 1e-9, \
            "grade-3 percentile for %s drifted from doc 56 SS9.3/SS10" % fam
    below = {k: bool(v < 50.0) for k, v in pcts.items()}
    grade3 = {
        "flag": ("under-conversion predicted for a BELOW-NULL category (the F1 "
                 "observation, made prospective -- doc 55 SS19.2 grade 3)"),
        "percentile_vs_size_matched_null": pcts,
        "below_null": below,
        "any_arm_below_null": bool(any(below.values())),
        "testable_this_run": bool(any(below.values())),
        "method_note": (
            "doc 56 SS9.3 / SS10 note 1: the N-across and EA-ctrl percentiles are "
            "computed at the METHOD-MATCHED two-stage value 0.9250 against a "
            "two-stage null, because no three-stage null exists. The verdict is "
            "robust to the choice -- 0.9625 > 0.9250 > 0.7940 -- so the category "
            "is above its null under either reading."),
        "verdict": (
            "UNTESTABLE THIS RUN -- no exp70 arm has a below-null category "
            "(P-eps %.1fth percentile, N-in %.1fth, N-across %.1fth, EA-ctrl "
            "%.1fth). Recorded rather than quietly dropped, exactly as doc 55 "
            "SS19.2 recorded it for exp69."
            % (pcts["P-eps"], pcts["N-in"], pcts["N-across"], pcts["EA-ctrl"])),
        "structural_observation_DESCRIPTIVE": (
            "doc 56 SS9.3: N-in and P-eps both reach 1.0000 but their NULLS "
            "differ -- N-in's size-matched null mean is %.4f +/- %.4f against "
            "P-eps's %.4f +/- %.4f, putting the real category at the %.1fth "
            "percentile for N-in and the %.1fth for P-eps. The 36-dimensional "
            "E-in space interpolates random labellings LESS than the "
            "81-dimensional product space while still reaching the real "
            "category's ceiling. Reported because two identical 1.0000s are NOT "
            "identical instruments. It licenses nothing (doc 52 SS2.2)."
            % (grade1["N-in"]["null_size_matched"]["mean"],
               grade1["N-in"]["null_size_matched"]["sd"],
               grade1["P-eps"]["null_size_matched"]["mean"],
               grade1["P-eps"]["null_size_matched"]["sd"],
               pcts["N-in"], pcts["P-eps"])
            if grade1["N-in"]["null_size_matched"] else None),
    }

    # ------------------------------------------- the entanglement distribution
    cp = diag["concurrence_and_purity"]
    ein = cp["e_in_eps_map"]
    ea = cp["e_across_over_the_80_SB_casts"]
    assert abs(ein["stats"]["min"] - DOC56_ENTANGLEMENT["e_in_eps_map_min"]) < 5e-5
    assert ein["stats"]["n_below_0.10"] == 0 and ein["gate_min_0.10_satisfied"]
    assert cp["e_in_registered_map_DISQUALIFIED"]["stats"]["n_below_0.05"] == \
        DOC56_ENTANGLEMENT["e_in_oldmap_below_0.05"]
    assert ea["concurrence"]["n_below_0.10"] == \
        DOC56_ENTANGLEMENT["e_across_casts_below_0.10"]
    for key, ref in (("min", "e_across_conc_min"), ("median", "e_across_conc_median"),
                     ("max", "e_across_conc_max")):
        assert abs(ea["concurrence"][key] - DOC56_ENTANGLEMENT[ref]) < 5e-5
    entanglement = {
        "e_in_eps_map_per_noun": ein["per_noun"],
        "e_in_eps_map_stats": ein["stats"],
        "degenerate_entanglement_gate_doc56_SS7_3": {
            "rule": ("EVERY N-in item's referents must have concurrence >= 0.10 "
                     "under the eps-map, computed from |sin t1| |sin(t2/2)| on the "
                     "item's own two nouns, asserted PER ITEM at smoke"),
            "threshold": 0.10,
            "inventory_minimum": ein["stats"]["min"],
            "n_nouns_below_threshold": ein["stats"]["n_below_0.10"],
            "why_it_is_asserted_anyway": (
                "doc 56 SS3's whole lesson is that 'the map should be fine' is not "
                "a gate. Under exp68's registered map four of the 24 nouns -- two "
                "exactly at zero -- would carry a PRODUCT state through an arm "
                "named entangled."),
            "NOT_applied_to_N_across": (
                "doc 56 SS7.3, registered honestly: 4 of the 80 E-across casts "
                "fall below concurrence 0.10 and are NOT excluded. Dropping casts "
                "from a frozen dataset to make an arm look entangled is the "
                "opposite of a validity gate. The weak-pair fraction is REPORTED "
                "and doc 56 SS6.8's per-item stratification exists so it can be "
                "read post-hoc without a rerun."),
        },
        "e_across_over_the_80_SB_casts": {
            k: v for k, v in ea.items() if k != "per_cast_purity"},
        "e_across_per_cast_purity": ea["per_cast_purity"],
        "honest_note_doc56_SS4_4": (
            "MANY PAIRS ARE NEAR-PRODUCT. The median E-across cast sits at purity "
            "%.4f, i.e. weakly entangled. E-across injects real but MODEST "
            "mechanism, and the size of any measured effect must be read against "
            "that, not against the maximum the gate could produce."
            % ea["purity_median"]),
    }

    payload = {
        "component": "exp70_predictions",
        "spec": SPEC, "spec_commit": SPEC_COMMIT,
        "ordering_rule": (
            "doc 56 SS17.3 item 5 / doc 55 SS19.2: this file is written FIRST, "
            "before exp70_run.py exists and before any exp70 number exists. "
            "exp70_run.py asserts its existence, so the ordering is mechanised. "
            "exp70 writes NO datagen (doc 56 SS5.1 / SS16.1: exp70_datagen.py does "
            "not exist and may not be written), so unlike exp69 the ordering rule "
            "is 'before the run', not 'before the dataset' -- the dataset is "
            "exp69's frozen Part A and predates this file entirely."),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script_sha256": _sha(os.path.abspath(__file__)),
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__,
        "provenance": {
            "instrument_entangled_bounds_sha256": _sha(
                os.path.join(HERE, "instrument_entangled_bounds.py")),
            "instrument_sb_bounds_sha256": _sha(
                os.path.join(HERE, "instrument_sb_bounds.py")),
            "capacity_check_sha256": _sha(os.path.join(HERE, "capacity_check.py")),
            "diagnostic_entangled_bounds_sha256": _sha(DIAG),
            "diagnostic_encoding_entanglement_sha256": _sha(DEGEN),
            "diagnostic_capacity_validation_sha256": _sha(CAPVAL),
            "diagnostic_readout_arity_bounds_addendum_sha256": _sha(ADDENDUM),
            "exp69_predictions_sha256": _sha(EXP69_PRED),
            "search_budget": dict(IEB.BUDGET), "null_seed": IEB.NULL_SEED,
            "null_draws": IEB.NULL_DRAWS,
            "raw_family": prov_raw,
            "eps_map": diag["eps_map"],
            "fa_angle_space_registered": "runtime "
                                         "(exp42_noun_embeddings.json angle)",
            "fa_angle_space_note": diag["fa_angle_space"]["why"],
            "normalisation_forensics": diag["provenance"].get(
                "committed_degeneracy_check", diag["provenance"]),
            "recompute_mode": ("RECOMPUTED (bounds, witnesses and size-matched "
                               "nulls all re-derived here through the committed "
                               "instrument's own machinery and asserted against "
                               "the committed diagnostic)" if not a.quick else
                               "COMMITTED NULLS CARRIED (--quick); bounds and "
                               "witnesses still recomputed"),
        },
        "arms": ARMS, "arm_family": ARM_FAMILY, "families": FAMILIES,
        "category": {"key": CATEGORY, "gloss": isb.CATEGORIES[CATEGORY]["gloss"],
                     "members": B, "nonbearers": N,
                     "source": ("instrument_sb_bounds.CATEGORIES -- doc 54 SS5.2, "
                                "user-confirmed; imported, never re-typed")},
        "grade1_bounds": grade1,
        "grade2_point_predictions": grade2,
        "grade3_under_conversion_flag": grade3,
        "entanglement_distribution_doc56_SS3_SS4_4_SS7_3": entanglement,
        "witnesses_doc56_SS15_3": witnesses,
        "witness_convention_doc56_SS15_3": (
            "the MAX-FS witness is NOT unique, so the registered convention is the "
            "committed search's output at its registered seed (%d, "
            "capacity_check.DEFAULT_BUDGET verbatim), exported at "
            "bound-computation time. THE RE-PINNING THAT MATTERS: Phi is the MAP'S "
            "OWN joint feature, not a universal one. For P-eps and N-in it is the "
            "outer product of per-referent vectors (81- and 36-dim); for N-across "
            "and EA-ctrl it is the genuinely joint 10-coordinate vector, which has "
            "no per-referent factorisation, so exp69's 'agent-first outer product' "
            "phrasing does not transfer and is REPLACED rather than reused. "
            "margin(twin) = <W, Phi>/(||W|| ||Phi||) with the sign convention of "
            "`maxfs_lower`; pair margin = min over the pair's two twins."
            % IEB.BUDGET["seed"]),
        "witness_caveat_doc56_SS15_3": (
            "ALL FOUR exp70 witnesses come from LOWER-BOUND searches -- no exact "
            "d = 3 pair witness is used anywhere -- so every margin is noisier BY "
            "CONSTRUCTION than exp69's d = 3 arms' were. The attenuation doc 55 "
            "SS20.4 saw only on the W arms should be expected ACROSS THE BOARD."),
        "registered_analysis_doc56_SS15_3": (
            "Spearman rank correlation between witness margin and per-item "
            "held-out accuracy, per arm. DESCRIPTIVE, positive expected, NO "
            "criterion reads it."),
        "bounds_are_context_only": (
            "doc 56 SS6's opening sentence and SS10: NO criterion in doc 56 SS6 is "
            "read against any number in this file. doc 54 SS17.7 is the standing "
            "reason -- a bound moved by 0.0375 AFTER exp68 reported and no stored "
            "boolean moved with it. These predictions are a PROSPECTIVE TEST OF "
            "THE INSTRUMENT, graded, and they arm doc 56 SS6.6's bug-hunt row "
            "(any arm exceeding its bound by > 0.06). They may not be promoted "
            "onto a criterion."),
        "clifford_result_doc56_SS4_5": {
            "statement": (
                "CZ is Clifford, so conjugating a general readout by it PERMUTES "
                "the Pauli coordinates and leaves the reachable function space "
                "invariant. N-across's grade-1 bound EQUALS EA-ctrl's BY PROOF, "
                "Fourier closure is a NULL INSTRUMENT for N-across (doc 56 SS7.2 "
                "replaces it), and the joint-state comparator cannot separate the "
                "two either -- so any measured N-across vs EA-ctrl difference is "
                "TRAINABILITY or INDUCTIVE BIAS, never capacity."),
            "certified": diag["derivations"].get("pauli_table", diag["derivations"]),
            "crosscheck": diag["fa_angle_space"]["clifford_crosscheck"],
        },
        "no_datagen_doc56_SS5_1": (
            "exp70 introduces NO new Arabic surface, no new noun and no new verb. "
            "Every arm runs on exp69's frozen Part A dataset "
            "(stories_exp69_QA.json, sha asserted at smoke) with the T = 1 and "
            "T = 2 item forms exp69 already generated and V-suite-cleared. "
            "exp70_datagen.py DOES NOT EXIST AND MAY NOT BE WRITTEN (doc 56 SS5.1 "
            "/ SS16.1). The confirm draw at generator seed + 1000 (doc 56 SS5.1.1 "
            "/ SS17.3 item 8) re-runs the COMMITTED exp69_datagen.py unmodified "
            "and fires only on a positive."),
        "wall_sec": time.time() - t0,
    }
    json.dump(payload, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("\n[exp70-pred] wrote %s (%.1fs)" % (a.out, payload["wall_sec"]))
    for fam in FAMILIES:
        g = grade1[fam]
        print("  %-9s bound of record %.4f (%s)   recomputed %.4f (%s)  arms %s"
              % (fam, g["bound_of_record"], g["fraction_of_record"],
                 g["recomputed_here"]["bound"], g["recomputed_here"]["fraction"],
                 g["arms"]))
    print("  grade-2 median conversion %.6f (n = %d)" % (median, int(conv["n"])))
    for arm in ARMS:
        p = grade2["by_arm"][arm]
        print("    %-13s point %.4f  interval [%.4f, %.4f]%s"
              % (arm, p["point_prediction"], p["interval"][0], p["interval"][1],
                 "   (miss EXPECTED -- doc 56 SS9.2)" if p["grade2_miss_expected"]
                 else ""))
    print("  grade-3: %s" % grade3["verdict"])
    print("  SS7.3 concurrence gate: eps-map inventory min %.4f, %d nouns < 0.10"
          % (ein["stats"]["min"], ein["stats"]["n_below_0.10"]))
    print("  SS4.4 E-across: concurrence %.4f-%.4f (median %.4f), %d/80 casts "
          "< 0.10; purity median %.4f"
          % (ea["concurrence"]["min"], ea["concurrence"]["max"],
             ea["concurrence"]["median"], ea["concurrence"]["n_below_0.10"],
             ea["purity_median"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
