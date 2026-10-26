# -*- coding: utf-8 -*-
"""exp70_aggregate.py -- doc 56: the ONLY writer of `results_exp70.json`.

doc 53 SS7.3's rule, applied for the fourth time: a runner that writes checkpoints
SHIPS WITH the aggregator that writes the results JSON, in the same commit. Nothing
else in exp70 writes a `results_exp70*.json` except this file and
`exp70_run.py --anchor-gate` (which writes `results_exp70_anchors.json`, separate by
Rule 3 so no exp68 or exp69 artefact is touched) and `exp70_run.py --smoke`.

WHAT IT COMPUTES -- doc 56 SS6, and EVERY criterion is measurement-vs-measurement.
NO CRITERION READS AGAINST ANY BOUND. doc 54 SS17.7 is the standing reason: a bound
moved by 0.0375 AFTER exp68 reported and no stored boolean moved with it.

  SS6.1  anchors_replicate_exp69                BLOCKING, READ FIRST
  SS6.2  the floor gate, ALL THREE SD readings, per arm
  SS6.3  binds_<arm>_T<t>                       six booleans
  SS6.4  THE EIGHT ONE-SIDED ENTANGLEMENT BOOLEANS, registered as PAIRS, with the
         third outcome written in advance: **helps and hurts MAY BOTH BE FALSE, and
         that is a RESULT, not a failure** -- it reads as "INDISTINGUISHABLE AT THIS
         POWER", i.e. the entangler moved the arm by less than 2 x the binomial SD
         at n = 300. A single `entanglement_helps` boolean would have silently
         collapsed "hurts" and "indistinguishable" into one FALSE, and doc 55 SS20.2's
         knife-edge (+0.0593 against a 0.06 bar) is the standing reminder that the
         region near the bar is where readings are made or lost.
  SS6.5  comparator_matches_or_exceeds_all_arms, per arm AND per comparator
  SS6.6  bug_hunt_triggered
  SS6.7  COMPOSITION COSTS for all five families, DESCRIPTIVE, no boolean, with
         NO CONFIDENT DIRECTION registered for cc_N-in
  SS6.8  REPORTED, NOT REGISTERED -- none may be promoted onto a criterion
  SS8.4  the four-row disambiguation table, which for the first time separates a
         TRAINABILITY failure from an ENCODING failure
  SS9    prediction-vs-measured at all three grades, KEPT APART
  SS15.3 Spearman(witness margin, per-item held-out accuracy), per arm
  SS15.1 / SS15.2  the purity contrasts, with the entangled arms' restoration target
         re-pinned to the INTRO purity rather than 1
  SS11   which outcome rows fired, quoted from the doc, chosen by the booleans

THE ONE DIRECTIONAL CALL. doc 56 SS9.4 registers `entanglement_hurts_T2` = TRUE as
the falsifiable prediction of the run -- the only cell where doc 56 SS2's query-1
barren-plateau mechanism and doc 55 SS20.3's register-collapse-under-composition
finding AGREE. It is checked against its outcome here and reported either way.

    python exp70_aggregate.py
"""
import argparse
import glob
import json
import os
import platform
import statistics as st
import subprocess
import sys
import time

import numpy as np
import torch

import exp42_compiler as comp
import exp70_run as R
from exp42_run import fresh_results_path

SPEC = R.SPEC
SPEC_COMMIT = R.SPEC_COMMIT
RESULTS_BASE = "results_exp70"
COMPARATOR = os.path.join(R.CKPT_DIR, "comparator.json")
SMOKE = R.SMOKE_OUT
ANCHORS = R.ANCHOR_OUT
PREDICTIONS = R.PREDICTIONS
VSUITE = R.EXP69_VSUITE

DELTA_BAR = 0.06        # doc 56 SS6.4: 2 x the binomial SD at n_test = 300
FLOOR_SD = 2.0          # doc 56 SS6.2
BINDS_ABS = 0.65        # doc 56 SS6.3, inherited verbatim from doc 54 SS7
TIE_BAND = 0.02         # doc 56 SS6.5
BUG_HUNT_MARGIN = R.BUG_HUNT_MARGIN

# doc 56 SS9.4's registered expectations, quoted so the run is scored against what
# was written BEFORE it, not against what it produced.
REGISTERED_EXPECTATION = {
    "anchors_replicate_exp69": "TRUE, bit-exact",
    "binds_P-eps_T1": "TRUE",
    "binds_P-eps_T2": "TRUE, LOW CONFIDENCE -- the same family that collapsed to "
                      "0.7293 under composition (doc 55 SS20.3)",
    "binds_N-in_T1": "TRUE -- the bound is 1.0000 at the 98th percentile of its null",
    "binds_N-in_T2": "TRUE, THE LOWEST CONFIDENCE IN THE DOC -- no precedent for an "
                     "entangled register under composition, and doc 56 SS2's query-1 "
                     "literature says entanglement between visible and hidden "
                     "subsystems is where trainability goes to die",
    "binds_N-across_T1": "TRUE -- span-identical to EA-ctrl, which measured 0.9227",
    "binds_N-across_T2": "TRUE -- span-identical to EA-ctrl, which measured 0.9187",
    "entanglement_helps_T1": "FALSE -- doc 56 SS2's query-2 prior: entangling "
                             "capability correlates only WEAKLY with accuracy",
    "entanglement_hurts_T1": "FALSE -- so the registered joint expectation at T = 1 "
                             "is INDISTINGUISHABLE AT THIS POWER",
    "entanglement_helps_T2": "FALSE",
    "entanglement_hurts_T2": "TRUE -- THE ONE DIRECTIONAL CALL IN THE DOC. The only "
                             "cell where doc 56 SS2's query-1 barren-plateau "
                             "mechanism and doc 55 SS20.3's register-collapse-under-"
                             "composition finding AGREE. Registered as the "
                             "falsifiable prediction of the run.",
    "across_helps_T1": "FALSE", "across_hurts_T1": "FALSE",
    "across_helps_T2": "FALSE", "across_hurts_T2": "FALSE",
    "comparator_matches_or_exceeds_all_arms": "TRUE -- doc 47 SS14, doc 52 SS1.1; "
                                              "measured TRUE in exp67, exp68, exp69",
    "bug_hunt_triggered": "FALSE",
}

DIRECTIONAL_CALL = "entanglement_hurts_T2"


# ------------------------------------------------------------------- helpers
def block(rows):
    rows = sorted(rows, key=lambda r: r["seed"])
    a = [r["heldout_test_acc"] for r in rows]
    return {"mean": st.mean(a), "sd": st.stdev(a) if len(a) > 1 else 0.0,
            "per_seed": a, "seeds": [r["seed"] for r in rows], "n_seeds": len(a),
            "T": rows[0]["T"], "family": rows[0]["family"],
            "encoding": rows[0]["encoding"], "role": rows[0]["role"],
            "val_per_seed": [r["val_acc"] for r in rows],
            "train_per_seed": [r["train_acc"] for r in rows],
            "selected_epoch_per_seed": [r["selected_epoch"] for r in rows],
            "val_minus_heldout_per_seed": [r["val_minus_heldout"] for r in rows],
            "mean_val_minus_heldout": st.mean([r["val_minus_heldout"] for r in rows]),
            "gate_met_per_seed": [r["gate_met"] for r in rows],
            "wires": rows[0]["wires"], "n_params": rows[0]["n_params"],
            "n_test": rows[0]["n_test"],
            "wall_sec_total": sum(r["wall_sec"] for r in rows),
            "ckpt_sha256": [r["ckpt_sha256"] for r in rows]}


def load_runs(tag=""):
    g = {}
    for f in sorted(glob.glob(os.path.join(R.CKPT_DIR, "*.json"))):
        if os.path.basename(f) == "comparator.json":
            continue
        r = json.load(open(f, encoding="utf-8"))
        if r.get("tag", "") != tag:
            continue
        g.setdefault(r["arm"], []).append(r)
    return g


def clearance(arm_blk, floor_blk):
    """doc 56 SS6.2: EVERY reading of '>= 2 SD above its matched floor'. An arm whose
    readings DISAGREE is reported as disagreeing rather than resolved silently."""
    d = arm_blk["mean"] - floor_blk["mean"]
    pooled = ((arm_blk["sd"] ** 2 + floor_blk["sd"] ** 2) / 2.0) ** 0.5
    readings = {
        "floor_sd": {"sd": floor_blk["sd"],
                     "bar": floor_blk["mean"] + FLOOR_SD * floor_blk["sd"],
                     "clears": bool(arm_blk["mean"]
                                    >= floor_blk["mean"] + FLOOR_SD * floor_blk["sd"])},
        "arm_sd": {"sd": arm_blk["sd"],
                   "bar": floor_blk["mean"] + FLOOR_SD * arm_blk["sd"],
                   "clears": bool(arm_blk["mean"]
                                  >= floor_blk["mean"] + FLOOR_SD * arm_blk["sd"])},
        "pooled_sd": {"sd": pooled, "bar": floor_blk["mean"] + FLOOR_SD * pooled,
                      "clears": bool(arm_blk["mean"]
                                     >= floor_blk["mean"] + FLOOR_SD * pooled)},
    }
    return {"arm_mean": arm_blk["mean"], "floor_mean": floor_blk["mean"],
            "delta_over_floor": d,
            "sd_units_over_floor_using_floor_sd":
                (d / floor_blk["sd"]) if floor_blk["sd"] > 0 else float("inf"),
            "readings": readings, "primary_reading": "floor_sd",
            "clears_floor": readings["floor_sd"]["clears"],
            "clears_under_all_readings":
                bool(all(v["clears"] for v in readings.values())),
            "clears_under_any_reading":
                bool(any(v["clears"] for v in readings.values())),
            "readings_agree": bool(len({v["clears"]
                                        for v in readings.values()}) == 1)}


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None

    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[o] = np.arange(1, len(v) + 1, dtype=float)
        for val in np.unique(v):
            m = v == val
            r[m] = r[m].mean()
        return r
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    pe = 0.0
    for c in (0, 1):
        pe += float((a == c).mean()) * float((b == c).mean())
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else None


def strat(rows, keyfn):
    cell = {}
    for r in rows:
        k = keyfn(r)
        cell.setdefault(k, [0, 0])
        cell[k][0] += int(r["correct"])
        cell[k][1] += 1
    return {str(k): {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(cell.items())}


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default=R.DATA_A)
    ap.add_argument("--ckpt-dir", default=R.CKPT_DIR)
    ap.add_argument("--smoke", default=None)
    ap.add_argument("--comparator", default=None)
    ap.add_argument("--first-draw", default=RESULTS_BASE + ".json",
                    help="the FIRST draw's results, for doc 56 SS11's confirm-vs-"
                         "first comparison and the rider verdict")
    ap.add_argument("--results-base", default=None)
    a = ap.parse_args()
    t0 = time.time()

    # Route the shared machinery at the requested draw; defaults are the first
    # draw's, so omitting the flags reproduces it exactly.
    R.DATA_A, R.CKPT_DIR = a.data, a.ckpt_dir
    CONFIRM = (os.path.abspath(a.data) != os.path.abspath(R.FIRST_DRAW_DATA))
    R.CONFIRM_MODE = CONFIRM
    globals()["COMPARATOR"] = a.comparator or os.path.join(R.CKPT_DIR,
                                                           "comparator.json")
    globals()["SMOKE"] = a.smoke or R.SMOKE_OUT
    results_base = a.results_base or (RESULTS_BASE + ("_confirm" if CONFIRM else ""))

    g = load_runs(a.tag)
    missing = [arm for arm in R.ARMS if len(g.get(arm, [])) != R.N_SEEDS]
    if missing:
        raise RuntimeError("incomplete exp70 checkpoints for %s (have %s)"
                           % (missing, {k: len(v) for k, v in g.items()}))
    n_runs = sum(len(v) for v in g.values())
    blocks = {arm: block(g[arm]) for arm in R.ARMS}
    arms = {b: blocks[b] for b in R.BASE_ARMS}
    floors = {b: blocks[R.FLOOR_OF[b]] for b in R.FLOORED}

    for f in (COMPARATOR, PREDICTIONS, ANCHORS, SMOKE):
        if not os.path.exists(f):
            raise RuntimeError("%s missing -- doc 56 SS8.6 / SS9 / SS6.1 / SS7 "
                               "require it before any result is read" % f)
    C = json.load(open(COMPARATOR, encoding="utf-8"))
    P = json.load(open(PREDICTIONS, encoding="utf-8"))
    AN = json.load(open(ANCHORS, encoding="utf-8"))
    SM = json.load(open(SMOKE, encoding="utf-8"))
    # The comparator MUST have been fitted on the draw being aggregated: its
    # features are a function of the ITEMS, so a carried-over comparator would be
    # doc 47 SS16.1's unmatched comparison and the exp44 failure class. The
    # first draw's comparator.json predates the `draw` block, so its absence is
    # read as "the first draw" rather than treated as a failure.
    cdraw = C.get("draw", {"dataset": R.FIRST_DRAW_DATA, "is_confirm_draw": False,
                           "dataset_sha256": None})
    assert bool(cdraw["is_confirm_draw"]) == CONFIRM, \
        ("the comparator at %s was fitted on a %s draw but the runs being "
         "aggregated are a %s draw -- unmatched comparator (doc 47 SS16.1)"
         % (COMPARATOR, "confirm" if cdraw["is_confirm_draw"] else "first",
            "confirm" if CONFIRM else "first"))
    assert SM.get("assertions_passed") is True, \
        "doc 56 SS7's smoke assertions did not pass -- no exp70 result is admissible"

    # ---- the dataset, and the structural join the item-level analyses need
    blob = json.load(open(R.DATA_A, encoding="utf-8"))
    assert comp.canonical_sha256(blob["items"]) == blob["meta"]["dataset_sha256"]
    ds = {"A": {"path": R.DATA_A, "sha256": blob["meta"]["dataset_sha256"],
                "is_confirm_draw": CONFIRM,
                "meta": {k: v for k, v in blob["meta"].items() if k != "items"}}}
    assert SM["dataset"]["sha256"] == ds["A"]["sha256"], \
        ("the smoke file %s validated dataset %s but the runs are on %s"
         % (SMOKE, SM["dataset"]["sha256"][:12], ds["A"]["sha256"][:12]))
    if cdraw.get("dataset_sha256"):
        assert cdraw["dataset_sha256"] == ds["A"]["sha256"], \
            "the comparator was fitted on a different dataset sha than the runs"
    item_of = {it["id"]: it for it in blob["items"]}
    for arm, rows in g.items():
        for r in rows:
            assert r["dataset_sha256"] == ds["A"]["sha256"], \
                "run %s/%d ran on a different dataset" % (arm, r["seed"])

    # ============================================== doc 56 SS6.1, READ FIRST
    # On a FRESH draw the gate is INAPPLICABLE, not merely skipped: it compares
    # per-seed accuracies against exp69's measurements ON EXP69'S OWN ITEMS, and
    # these are different items. What the gate certified -- that exp70's codepath
    # has not drifted -- is a property of the CODE, established on the first draw
    # and shared byte-for-byte here. It is carried as provenance, and the boolean
    # is recorded as None rather than as a manufactured TRUE.
    anchors_replicate_exp69 = (None if CONFIRM
                               else bool(AN["anchors_replicate_exp69"]))
    anchor_note = (
        "INAPPLICABLE ON A FRESH DRAW. doc 56 SS6.1's gate compares per-seed "
        "accuracies against exp69's measurements on EXP69'S OWN FROZEN ITEMS; the "
        "confirm draw has different items, so there is nothing to compare against "
        "and a 'pass' would be manufactured. The codepath it certifies was gated "
        "on the FIRST draw and PASSED BIT-EXACTLY on all four arms (%s), and this "
        "run shares that code byte-for-byte. CONSEQUENCE CARRIED: doc 56 SS5.3 "
        "omitted the anchor families' floors on the argument that the |delta| <= "
        "0.002 gate was transitively stronger than re-measuring them. THAT "
        "ARGUMENT DOES NOT SURVIVE A FRESH DRAW. The anchor arms are therefore "
        "UN-FLOORED DESCRIPTIVE BASELINES on this draw and NO BOOLEAN READS THEM; "
        "they are not silently promoted to gated arms, and two new floor families "
        "were not invented to rescue them (that would add 20 runs the "
        "registration never authorised, on the draw whose entire job is to "
        "re-test ONE registered boolean)."
        % ("bit-exact" if AN.get("all_bit_exact") else "within tolerance")
        if CONFIRM else
        "BLOCKING, READ FIRST -- doc 56 SS6.1, and it PASSED.")

    # ============================================== doc 56 SS6.2, the floor gate
    clear = {b: clearance(arms[b], floors[b]) for b in R.FLOORED}
    floor_gate = {b: clear[b]["clears_floor"] for b in R.FLOORED}
    readings_disagree = [b for b in R.FLOORED if not clear[b]["readings_agree"]]
    all_floors = bool(all(floor_gate.values()))

    # ============================================== the comparator, SS8 / SS6.5
    m = {b: arms[b]["mean"] for b in R.BASE_ARMS}
    cmp_per_tier, cmp_floor_per_tier, cmp_clear = {}, {}, {}
    for b in R.BASE_ARMS:
        fam = R.ARM_CFG[b]["family"]
        cmp_per_tier[b] = {t: {k: v for k, v in C["arms"][b][t].items() if k != "runs"}
                           for t in C["arms"][b]}
        if b in R.FLOOR_OF and b in C["floors"]:
            cmp_floor_per_tier[b] = {
                t: {k: v for k, v in C["floors"][b][t].items() if k != "runs"}
                for t in C["floors"][b]}
            cmp_clear[b] = {t: clearance(C["arms"][b][t], C["floors"][b][t])
                            for t in C["floors"][b]}
    cmp_minus_circuit = {b: {t: cmp_per_tier[b][t]["mean"] - m[b]
                             for t in cmp_per_tier[b]} for b in R.BASE_ARMS}
    cmp_ge_per_tier = {b: {t: bool(v >= -TIE_BAND)
                           for t, v in cmp_minus_circuit[b].items()}
                       for b in R.BASE_ARMS}
    cmp_ge_any = {b: bool(any(cmp_ge_per_tier[b].values())) for b in R.BASE_ARMS}
    cmp_ge_all = {b: bool(all(cmp_ge_per_tier[b].values())) for b in R.BASE_ARMS}
    comparator_matches_or_exceeds_all_arms = bool(all(cmp_ge_any.values()))
    comparator_matches_or_exceeds_all_arms_STRICT = bool(all(cmp_ge_all.values()))

    # ============================================== doc 56 SS6.3, the six binds
    # doc 56 SS6.3's names use an UNDERSCORE before the T (`binds_N-in_T2`), while
    # the arm keys use a hyphen (`N-in-T2`). The two must be reconciled explicitly:
    # left implicit, the six binds would drop silently out of the
    # expectation-vs-outcome table below instead of failing loudly.
    binds, binds_key = {}, {}
    for b in R.FLOORED:
        k = "binds_" + b.replace("-T", "_T")
        binds_key[b] = k
        binds[k] = bool(floor_gate[b] and m[b] >= BINDS_ABS)

    # ====================== doc 56 SS6.4, THE EIGHT ONE-SIDED BOOLEANS
    def one_sided(hi, lo, tag):
        """Both arms of the pair must clear their gates and SS8.4's comparator
        clause must not block. `hi` is the treatment arm, `lo` the matched
        control."""
        d = m[hi] - m[lo]
        gated = (floor_gate.get(hi, True) if hi in R.FLOORED else True) and \
                (floor_gate.get(lo, True) if lo in R.FLOORED else True)
        # SS8.4's clause: the circuit gap may not EXCEED the information gap. The
        # comparator gap is read on the tier the two arms SHARE.
        shared = [t for t in cmp_per_tier[hi] if t in cmp_per_tier[lo]]
        cgap = (max(cmp_per_tier[hi][t]["mean"] - cmp_per_tier[lo][t]["mean"]
                    for t in shared) if shared else None)
        blocked = bool(cgap is not None and abs(d) >= DELTA_BAR
                       and abs(cgap) < DELTA_BAR and d > 0)
        return {
            "pair": "%s - %s" % (hi, lo), "delta": d, "bar": DELTA_BAR,
            "both_arms_gated": bool(gated),
            "comparator_gap_shared_tier": cgap,
            "shared_tiers": shared,
            "comparator_clause_blocks": blocked,
            "helps": bool(d >= DELTA_BAR and gated and not blocked),
            "hurts": bool(d <= -DELTA_BAR and gated),
            "tag": tag,
        }

    pairs = {
        "T1_intra": one_sided("N-in-T1", "P-eps-T1", "entanglement"),
        "T2_intra": one_sided("N-in-T2", "P-eps-T2", "entanglement"),
        "T1_across": one_sided("N-across-T1", "EA-ctrl-T1", "across"),
        "T2_across": one_sided("N-across-T2", "EA-ctrl-T2", "across"),
    }
    eight = {
        "entanglement_helps_T1": pairs["T1_intra"]["helps"],
        "entanglement_hurts_T1": pairs["T1_intra"]["hurts"],
        "entanglement_helps_T2": pairs["T2_intra"]["helps"],
        "entanglement_hurts_T2": pairs["T2_intra"]["hurts"],
        "across_helps_T1": pairs["T1_across"]["helps"],
        "across_hurts_T1": pairs["T1_across"]["hurts"],
        "across_helps_T2": pairs["T2_across"]["helps"],
        "across_hurts_T2": pairs["T2_across"]["hurts"],
    }
    for k, v in pairs.items():
        assert not (v["helps"] and v["hurts"]), \
            "%s returned helps AND hurts -- doc 56 SS6.4 forbids it" % k
        v["comparator_clause_note"] = (
            "doc 56 SS6.4 requires SS8.4's comparator clause not to block, and SS8.4 "
            "is doc 55 SS8.2/SS8.3 inherited and EXTENDED TO THE COMPARATOR PAIR. "
            "Applied here in its faithful form: a POSITIVE circuit gap that clears "
            "the 0.06 bar while the comparator gap on the SHARED tier does not is an "
            "artefact of the control arm's trainability and MAY NOT be claimed. It is "
            "deliberately NOT applied to the `hurts` direction, because a negative "
            "circuit gap with no comparator gap IS the trainability reading, and "
            "SS8.4 row 2 registers that as a legitimate, reportable outcome rather "
            "than a blocked one. Blocking it would suppress exactly the branch doc 56 "
            "SS2's query-1 literature predicts.")
    third_outcome = {
        k: (("INDISTINGUISHABLE AT THIS POWER -- the entangler moved the arm by "
             "%+.4f, less than the 0.06 bar (2 x the binomial SD at n = 300). "
             "doc 56 SS6.4: THIS IS A RESULT, NOT A FAILURE." % v["delta"])
            if not v["helps"] and not v["hurts"] else
            ("HELPS by %+.4f" % v["delta"]) if v["helps"] else
            ("HURTS by %+.4f" % v["delta"]))
        for k, v in pairs.items()}

    # ============================================== doc 56 SS6.6, the bug hunt
    bounds = {b: P["grade1_bounds"][R.ARM_CFG[b]["family"]]["bound_of_record"]
              for b in R.BASE_ARMS}
    over = {b: m[b] - bounds[b] for b in R.BASE_ARMS}
    bug_hunt = {b: bool(over[b] > BUG_HUNT_MARGIN) for b in R.BASE_ARMS}
    bug_hunt_triggered = bool(any(bug_hunt.values()))

    # =========================== doc 56 SS6.7, COMPOSITION COSTS -- DESCRIPTIVE
    def cc(fam):
        t1, t2 = "%s-T1" % fam, "%s-T2" % fam
        per = [x - y for x, y in zip(arms[t1]["per_seed"], arms[t2]["per_seed"])]
        return {"pair": "%s - %s" % (t1, t2),
                "mean": arms[t1]["mean"] - arms[t2]["mean"],
                "per_seed": per, "sd": st.stdev(per) if len(per) > 1 else 0.0}
    ccs = {"cc_%s" % f: cc(f) for f in R.FAMILIES}

    def dcc(a_fam, b_fam, name, note):
        x, y = ccs["cc_%s" % a_fam], ccs["cc_%s" % b_fam]
        return {"definition": "%s = cc_%s - cc_%s" % (name, a_fam, b_fam),
                "mean": x["mean"] - y["mean"],
                "per_seed": [p - q for p, q in zip(x["per_seed"], y["per_seed"])],
                "note": note}
    interactions = {
        "dcc_entanglement": dcc("N-in", "P-eps", "dcc_entanglement",
                                "the ENTANGLEMENT interaction: does the entangled "
                                "register compose differently from its MAP-MATCHED "
                                "product twin? This is the cc form of doc 56 SS4.3's "
                                "contrast."),
        "dcc_across": dcc("N-across", "EA-ctrl", "dcc_across",
                          "the CROSS-WIRE interaction against the map-matched "
                          "control. doc 56 SS4.5: any movement here is trainability "
                          "or inductive bias, NEVER capacity."),
        "dcc_map": dcc("P-eps", "A0-W", "dcc_map",
                       "the ANGLE-MAP interaction within the product-register "
                       "family. DESCRIPTIVE CONTEXT ONLY (doc 56 SS4.3): P-eps "
                       "against the old map differs in angle map, so differencing "
                       "them is a population mismatch in doc 47 SS18.5's sense."),
    }
    composition = {
        "per_family": ccs, "interactions": interactions,
        "status": "DESCRIPTIVE ONLY -- NONE OF THESE IS A CRITERION",
        "no_promotion_clause": (
            "doc 56 SS6.7, stated before results existed: the exp44 no-promotion "
            "clause applies -- NONE of these may be PROMOTED ONTO A CRITERION after "
            "results exist. At 5 seeds they are knife-edge magnets, which is why the "
            "per-seed values are printed alongside every mean."),
        "no_confident_direction_for_cc_N_in": (
            "doc 56 SS6.7, registered: the precedents GENUINELY CONFLICT. A single "
            "RY composes essentially for free (cc_A = +0.0040, cc_M = +0.0033, doc 55 "
            "SS20.3); a product register COLLAPSES under composition (cc_W = +0.2527, "
            "same source); and AN ENTANGLED REGISTER HAS NO PRECEDENT in this "
            "programme or, per doc 56 SS2, in the literature. Either branch is "
            "readable and neither is a surprise. The nearest precedent for cc_P-eps "
            "is R2-W's collapse, since P-eps is the same product-register family on a "
            "different map; that is stated as the EXPECTATION and not as a criterion."),
        "population_note": (
            "doc 56 SS6.9(b) refuses any T = 1 vs T = 2 accuracy DELTA as a "
            "CRITERION, because R1 and R2 are different tasks with different label "
            "distributions and differencing them is doc 47 SS18.5's population "
            "mismatch. SS6.7 registers the same difference as a DESCRIPTIVE "
            "quantity, and the distinction is exactly that: reported, never read."),
        "exp69_reference_values": {"cc_A": 0.0040, "cc_M": 0.0033, "cc_W": 0.2527,
                                   "source": "doc 55 SS20.3"},
    }

    # =================================== doc 56 SS9, prediction vs measured
    pred_table = {}
    for b in R.BASE_ARMS:
        fam = R.ARM_CFG[b]["family"]
        g2 = P["grade2_point_predictions"]["by_arm"][b]
        pred_table[b] = {
            "family": fam, "grade1_bound_of_record": bounds[b],
            "grade1_fraction": P["grade1_bounds"][fam]["fraction_of_record"],
            "grade1_method": P["grade1_bounds"][fam]["recomputed_here"]["method"],
            "grade2_point": g2["point_prediction"],
            "grade2_interval": g2["interval"],
            "grade2_miss_was_expected": g2["grade2_miss_expected"],
            "measured": m[b],
            "measured_minus_point": m[b] - g2["point_prediction"],
            "inside_grade2_interval": bool(g2["interval"][0] <= m[b]
                                           <= g2["interval"][1]),
            "conversion_ratio_measured_over_bound": m[b] / bounds[b],
            "over_bound_by": over[b], "bug_hunt_row_fires": bug_hunt[b],
        }
    n_in_iv = sum(1 for v in pred_table.values() if v["inside_grade2_interval"])
    expected_misses = [b for b, v in pred_table.items()
                       if v["grade2_miss_was_expected"]]
    predictions_report = {
        "grades_kept_apart": (
            "doc 56 SS9. GRADE 1 is the BOUND, computed by the committed instrument "
            "BEFORE this build existed (doc 54 SS0's spent instrument exception) and "
            "frozen in exp70_predictions.json. GRADE 2 is the CONVERSION POINT "
            "PREDICTION, bound x the trained-population single-RY conversion median "
            "(%.6f, n = %d rows in ONE task family), with a +/- %.2f interval. GRADE "
            "3 was computed before freezing and recorded as UNTESTABLE: no exp70 arm "
            "has a below-null category."
            % (P["grade2_point_predictions"]["conversion_median"],
               P["grade2_point_predictions"]["conversion_median_n_rows"], DELTA_BAR)),
        "grade3": P["grade3_under_conversion_flag"],
        "table": pred_table,
        "n_arms_inside_grade2_interval": n_in_iv, "n_arms": len(pred_table),
        "arms_where_a_miss_was_REGISTERED_AS_EXPECTED": expected_misses,
        "extrapolation_under_test_TWO_PRECEDENT":
            P["grade2_point_predictions"]["extrapolation_under_test_TWO_PRECEDENT"],
        "extrapolation_verdict": (
            "the grade-2 extrapolation HOLDS on %d of %d arms" % (n_in_iv, len(pred_table))
            if n_in_iv == len(pred_table) else
            "the grade-2 extrapolation MISSES on %d of %d arms (%s). doc 56 SS9.2: a "
            "miss refutes THE ONE-FAMILY EXTRAPOLATION, NOT the bound and NOT the "
            "calculus -- and for the four-wire T = 2 arms the miss was REGISTERED IN "
            "ADVANCE AS EXPECTED, because doc 55 SS20.4 already refuted grade 2 for "
            "R2-W, the same product-register family. No criterion in doc 56 SS6 reads "
            "any of these numbers."
            % (len(pred_table) - n_in_iv, len(pred_table),
               [b for b, v in pred_table.items() if not v["inside_grade2_interval"]])),
        "bounds_are_context_only": P["bounds_are_context_only"],
    }

    # ================= item-level machinery: SS15.3, SS15.1/SS15.2, SS6.8
    def per_item(arm):
        acc, rep = {}, {}
        for r in sorted(g[arm], key=lambda x: x["seed"]):
            for row in r["heldout_items"]:
                acc.setdefault(row["id"], []).append(int(row["correct"]))
                rep.setdefault(row["id"], row)
        return acc, rep

    W = P["witnesses_doc56_SS15_3"]
    ENT = P["entanglement_distribution_doc56_SS3_SS4_4_SS7_3"]
    fm_conc = ENT["e_in_eps_map_per_noun"]
    ea_purity = ENT["e_across_per_cast_purity"]

    s153, s151, s195, s68, s_strat = {}, {}, {}, {}, {}
    for b in R.BASE_ARMS:
        acc, rep = per_item(b)
        fam = R.ARM_CFG[b]["family"]
        # ---- SS15.3 Spearman(margin, per-item held-out accuracy)
        tw = W[fam]["per_twin"]
        xs, ys, miss = [], [], 0
        for iid, ks in acc.items():
            key = "%s|%s" % (rep[iid]["agent"], rep[iid]["patient"])
            if key not in tw:
                miss += 1
                continue
            xs.append(tw[key])
            ys.append(float(np.mean(ks)))
        assert miss == 0, \
            ("SS15.3 margin join broke for %s: %d held-out items have no frozen "
             "witness margin. A nonzero count means the (agent, patient) keys do not "
             "index the twin design the bound was computed over -- loud, not a field."
             % (b, miss))
        rho = spearman(xs, ys)
        bycast = {}
        for iid, ks in acc.items():
            key = "%s|%s" % (rep[iid]["agent"], rep[iid]["patient"])
            bycast.setdefault(key, []).append(float(np.mean(ks)))
        rho_cast = spearman([tw[k] for k in bycast],
                            [float(np.mean(v)) for v in bycast.values()])
        s153[b] = {
            "witness_family": fam, "witness_d": W[fam]["d"],
            "n_items": len(xs), "n_unmatched": miss,
            "n_distinct_margins": len(set(xs)),
            "spearman_item_level": rho, "spearman_cast_level": rho_cast,
            "n_casts": len(bycast),
            "direction_registered": "positive expected (doc 56 SS15.3)",
            "status": "DESCRIPTIVE -- no criterion reads it",
            "phi_definition": W[fam]["phi_definition"],
            "resolution_note": (
                "the witness margin is a function of the (agent, patient) ORDERED "
                "PAIR alone, so all items sharing a queried-event direction on the "
                "same cast share ONE margin: %d distinct margin values over %d "
                "held-out items (%d casts). Both readings are given rather than the "
                "coarser one being hidden; ties are handled by midranks."
                % (len(set(xs)), len(xs), len(bycast))),
            "caveat": P["witness_caveat_doc56_SS15_3"],
        }
        # ---- SS15.1 / SS15.2 purity contrast, stratified by q and correctness
        cells = {}
        for iid, row in rep.items():
            q = row["query_index"]
            ok = bool(np.mean(acc[iid]) >= 0.5)
            k = "q%d_%s" % (q, "correct" if ok else "incorrect")
            c = cells.setdefault(k, {"n": 0, "stages": {}})
            c["n"] += 1
            for stage, vals in row["purity"].items():
                c["stages"].setdefault(stage, []).append(float(np.mean(vals)))
        cell_out = {k: {"n": c["n"],
                        "mean_wire_purity": {s: float(np.mean(v))
                                             for s, v in sorted(c["stages"].items())}}
                    for k, c in sorted(cells.items())}
        # SS15.2's amendment: for the ENTANGLED arms the restoration target is the
        # INTRO purity, not 1.
        restore = {}
        for q in (1, 2):
            ids = [i for i in rep if rep[i]["query_index"] == q
                   and "after_event1" in rep[i]["purity"]]
            if not ids:
                continue
            gap_e1 = [abs(float(np.mean(rep[i]["purity"]["at_readout"]))
                          - float(np.mean(rep[i]["purity"]["after_event1"])))
                      for i in ids]
            gap_intro = [abs(float(np.mean(rep[i]["purity"]["at_readout"]))
                             - float(np.mean(rep[i]["purity"]["after_intro"])))
                         for i in ids]
            restore["q%d" % q] = {
                "n": len(ids),
                "readout_vs_after_event1_mean_abs_gap": float(np.mean(gap_e1)),
                "readout_vs_INTRO_mean_abs_gap": float(np.mean(gap_intro)),
                "mean_intro_purity": float(np.mean(
                    [float(np.mean(rep[i]["purity"]["after_intro"])) for i in ids])),
            }
        s151[b] = {
            "T": R.ARM_CFG[b]["T"], "encoding": R.ARM_CFG[b]["enc"],
            "cells": cell_out, "restoration": restore,
            "registered_signature_doc56_SS15_2": (
                "doc 55 SS19.4 confirmed the registered asymmetry on exp69: on "
                "correct q = 2 items readout RESTORES per-wire purity to its "
                "post-event-1 level (0.8755 vs 0.8751); on q = 1 it does not (0.8017 "
                "vs 0.8953). **THE exp70 AMENDMENT (doc 56 SS15.2): for the ENTANGLED "
                "arms the restoration target is the INTRO PURITY, NOT 1**, and the "
                "same q-asymmetry is expected against that shifted baseline. Both "
                "gaps are reported so the shift is visible. DESCRIPTIVE."),
            "status": "DESCRIPTIVE -- doc 56 SS15.1 / SS15.2, no criterion reads it",
        }
        # ---- error overlap against BOTH comparators (doc 56 SS15.4's addition)
        s195[b] = {}
        for tier in C["arms"][b]:
            krows, subs = [], []
            for cr, mr in zip(sorted(g[b], key=lambda x: x["seed"]),
                              sorted(C["arms"][b][tier]["runs"],
                                     key=lambda x: x["seed"])):
                ca = {r["id"]: r for r in cr["heldout_items"]}
                ma = {r["id"]: r for r in mr["heldout_items"]}
                ids = sorted(set(ca) & set(ma))
                kp = cohen_kappa([ca[i]["pred"] for i in ids],
                                 [ma[i]["pred"] for i in ids])
                ce = {i for i in ids if not ca[i]["correct"]}
                me = {i for i in ids if not ma[i]["correct"]}
                krows.append(kp)
                subs.append({"seed": cr["seed"], "n": len(ids),
                             "n_circuit_errors": len(ce),
                             "n_comparator_errors": len(me),
                             "circuit_errors_that_are_also_comparator_errors":
                                 (len(ce & me) / len(ce)) if ce else None,
                             "comparator_errors_that_are_also_circuit_errors":
                                 (len(ce & me) / len(me)) if me else None})
            s195[b][tier] = {
                "cohen_kappa_per_seed": krows,
                "cohen_kappa_mean": (float(np.mean([k for k in krows if k is not None]))
                                     if any(k is not None for k in krows) else None),
                "error_subsets_per_seed": subs}
        s195[b]["reading"] = (
            "doc 56 SS15.4: errors-as-SUBSET reads as 'noisy linear model'; DISJOINT "
            "errors read as a different inductive bias. Computed against BOTH "
            "comparators where both exist, since THE DIFFERENCE BETWEEN THE TWO "
            "OVERLAPS IS ITSELF THE SS8.1 READING IN PER-ITEM FORM. Descriptive, no "
            "direction registered.")
        # ---- SS6.8's reported-not-registered strata
        rows_flat = [r for run in g[b] for r in run["heldout_items"]]
        for r in rows_flat:
            it = item_of.get(r["id"], {})
            r["_vpair"] = ("%s|%s" % (it.get("verb_slots", {}).get("V_A"),
                                      it.get("verb_slots", {}).get("V_B"))
                           if it.get("verb_slots") else "n/a")
        s68[b] = {
            "by_query_index_PRIMACY_RECENCY": strat(
                rows_flat, lambda r: "q%d" % r["query_index"]),
            "by_surface_order_pair": strat(rows_flat,
                                           lambda r: "/".join(r["order_flags"])),
            "by_surface_order_event1": strat(rows_flat, lambda r: r["order_flags"][0]),
            "by_verb_pair": strat(rows_flat, lambda r: r["_vpair"]),
            "by_queried_verb": strat(rows_flat, lambda r: r["queried_verb"]),
            "val_minus_heldout_per_seed": arms[b]["val_minus_heldout_per_seed"],
            "mean_val_minus_heldout": arms[b]["mean_val_minus_heldout"],
        }
        # ---- SS6.8's PER-ITEM ENTANGLEMENT STRATIFICATION
        enc = R.ARM_CFG[b]["enc"]
        if enc in ("E-in", "E-across"):
            def item_conc(r):
                if enc == "E-in":
                    return min(fm_conc[n] for n in r["cast"])
                a, p = r["agent"], r["patient"]
                pk = "%s|%s" % (a, p) if "%s|%s" % (a, p) in ea_purity \
                    else "%s|%s" % (p, a)
                return float(np.sqrt(max(0.0, 2.0 * (1.0 - ea_purity[pk]))))
            qs = sorted({round(item_conc(r), 12) for r in rows_flat})
            cut = qs[len(qs) // 2] if len(qs) > 1 else qs[0]
            s_strat[b] = {
                "quantity": ("min over the item's two referents of the E-in "
                             "concurrence |sin t1||sin(t2/2)|" if enc == "E-in"
                             else "the cast's E-across concurrence |sin tA||sin tB|, "
                                  "recovered from the frozen per-cast purity"),
                "median_split_at": cut,
                "low_half": strat([r for r in rows_flat if item_conc(r) < cut],
                                  lambda r: "all"),
                "high_half": strat([r for r in rows_flat if item_conc(r) >= cut],
                                   lambda r: "all"),
                "n_distinct_values": len(qs),
                "below_0.10": strat([r for r in rows_flat if item_conc(r) < 0.10],
                                    lambda r: "all") or None,
                "status": ("doc 56 SS6.8: REPORTED, NOT REGISTERED. It exists so "
                           "doc 56 SS7.3's decision NOT to gate N-across on "
                           "concurrence can be read post-hoc WITHOUT A RERUN, and "
                           "so the 4/80 weak casts are visible rather than "
                           "asserted away. MAY NOT BE PROMOTED ONTO A CRITERION."),
            }

    # ============================== doc 56 SS8.4, the four-row disambiguation
    def disambiguate(arm):
        fam = R.ARM_CFG[arm]["family"]
        tiers = C["arms"][arm]
        # THE FUNCTION-CLASS-MATCHED tier is "class" for the register families and
        # "joint" for the E-across families -- doc 56 SS8.2 registers the joint
        # 10-coordinate form as that family's function-class-matched comparator, and
        # there is no per-referent factorisation to build a "class" tier from.
        # Reading only "class" here would score an across arm as having NO clearing
        # comparator when its registered one had cleared.
        tclass = "class" if "class" in tiers else "joint"
        raw_ok = (cmp_clear.get(arm, {}).get("raw", {}) or {}).get("clears_floor")
        cls_ok = (cmp_clear.get(arm, {}).get(tclass, {}) or {}).get("clears_floor")
        if floor_gate.get(arm) and m[arm] >= BINDS_ABS:
            row, reading = ("SS8.4 row 1", (
                "%s CLEARS ITS FLOOR AND 0.65 (%.4f vs floor %.4f). The encoding "
                "CONVERTS; read doc 56 SS11." % (arm, m[arm], floors[arm]["mean"])))
        elif cls_ok:
            row, reading = ("SS8.4 row 2", (
                "CIRCUIT-TRAINABILITY FAILURE. %s sits at its floor (%.4f vs %.4f) "
                "while the FUNCTION-CLASS-MATCHED comparator (ii) on the IDENTICAL "
                "items clears its own floor (%.4f vs %.4f). The information is there "
                "and is LINEARLY EXTRACTABLE IN THE ENCODING'S OWN FUNCTION CLASS; "
                "the circuit did not learn it. **This is the branch doc 56 SS2's "
                "query-1 literature PREDICTS for an entangled encoding, and it was "
                "registered in advance.** May NOT be reported as 'the information "
                "isn't there', and may not be reported as a property of entangled "
                "encodings generally. (Tier read: %s.)"
                % (arm, m[arm], floors[arm]["mean"],
                   C["arms"][arm][tclass]["mean"], C["floors"][arm][tclass]["mean"],
                   tclass)))
        elif raw_ok:
            row, reading = ("SS8.4 row 3", (
                "THE ENCODING DESTROYED EXTRACTABLE INFORMATION. %s sits at its "
                "floor and so does comparator (ii), but the RAW-coordinate "
                "comparator (i) CLEARS (%.4f vs %.4f). The raw coordinates carry the "
                "category; the map's function class does not make it linearly "
                "available. A real and reportable NEGATIVE ABOUT THE MAP, distinct "
                "from a negative about training -- and the first time this programme "
                "has been able to tell those apart. May not be reported as a "
                "trainability failure."
                % (arm, C["arms"][arm]["raw"]["mean"],
                   C["floors"][arm]["raw"]["mean"])))
        else:
            row, reading = ("SS8.4 row 4", (
                "INFORMATION / SAMPLE-SIDE. %s sits at its floor and so does every "
                "registered comparator on the identical items. NO CIRCUIT CLAIM IS "
                "AVAILABLE EITHER WAY." % arm))
        return {"row": row, "reading": reading,
                "function_class_matched_tier": tclass,
                "comparator_tiers": sorted(tiers),
                "comparator_clears_floor": {t: (cmp_clear.get(arm, {}).get(t, {}) or {})
                                            .get("clears_floor") for t in tiers},
                "family": fam}
    arbiter = {b: disambiguate(b) for b in R.FLOORED}

    # SS8.1's reading rule, applied per arm that carries BOTH tiers
    two_tier_reading = {}
    for b in R.BASE_ARMS:
        tiers = cmp_ge_per_tier[b]
        if "raw" in tiers and "class" in tiers:
            if tiers["raw"]:
                verdict = ("(i) TIES the circuit -> THE MAP ADDED NOTHING "
                           "EXTRACTABLE. The two raw coordinates already carry "
                           "everything the circuit got out of them, and the map is a "
                           "REPARAMETERISATION however exotic its span (doc 56 SS8.1).")
            elif tiers["class"]:
                verdict = ("(i) FAILS to tie but (ii) TIES -> THE MAP GENUINELY "
                           "RESHAPED THE FUNCTION CLASS. The information needed a "
                           "nonlinear reorganisation the raw coordinates do not offer "
                           "linearly, and THE ENCODING SUPPLIED IT (doc 56 SS8.1).")
            else:
                verdict = ("NEITHER comparator ties the circuit. doc 56 SS8.1's rule "
                           "does not cover this cell; it is reported as uncovered "
                           "rather than forced into one of the two readings, and the "
                           "SS6.5 guard is FALSE for this arm.")
            two_tier_reading[b] = {
                "raw_ties": tiers["raw"], "class_ties": tiers["class"],
                "raw_minus_circuit": cmp_minus_circuit[b]["raw"],
                "class_minus_circuit": cmp_minus_circuit[b]["class"],
                "registered_reading": verdict}

    # ========== doc 56 SS11 / SS5.1.1: CONFIRM vs FIRST DRAW, AND THE RIDER
    confirm_report = None
    if CONFIRM:
        if not os.path.exists(a.first_draw):
            raise RuntimeError(
                "%s missing -- the confirm draw is only meaningful against the "
                "first draw it is confirming (doc 56 SS11)" % a.first_draw)
        F = json.load(open(a.first_draw, encoding="utf-8"))
        assert F["datasets"]["A"]["sha256"] != ds["A"]["sha256"], \
            "the 'first draw' file was produced on THIS dataset -- nothing to confirm"
        fm_arms, fm_floors = F["arms"], F["floors"]
        fv = F["verdicts"]
        per_arm_delta = {}
        for b in R.BASE_ARMS:
            per_arm_delta[b] = {
                "first_draw_mean": fm_arms[b]["mean"],
                "first_draw_sd": fm_arms[b]["sd"],
                "confirm_mean": arms[b]["mean"], "confirm_sd": arms[b]["sd"],
                "delta_confirm_minus_first": arms[b]["mean"] - fm_arms[b]["mean"],
                "floor_first": fm_floors[b]["mean"] if b in fm_floors else None,
                "floor_confirm": floors[b]["mean"] if b in floors else None,
                "is_unfloored_baseline_on_confirm": bool(b not in floors),
            }
        bool_cmp = {}
        for k in sorted(eight):
            bool_cmp[k] = {"first_draw": fv.get(k), "confirm_draw": eight[k],
                           "agrees": bool(fv.get(k) == eight[k])}
        for k in sorted(binds):
            bool_cmp[k] = {"first_draw": fv.get(k), "confirm_draw": binds[k],
                           "agrees": bool(fv.get(k) == binds[k])}
        first_hurts = bool(fv.get(DIRECTIONAL_CALL))
        confirm_hurts = bool(eight[DIRECTIONAL_CALL])
        stands = bool(first_hurts and confirm_hurts)
        rider = {
            "boolean": DIRECTIONAL_CALL,
            "first_draw": first_hurts, "confirm_draw": confirm_hurts,
            "first_draw_delta": fv["measured"]["N-in_minus_P-eps_T2"],
            "confirm_draw_delta": pairs["T2_intra"]["delta"],
            "headline_stands": stands,
            "rule": (
                "doc 56 SS11's row for `entanglement_hurts_T2` states the headline "
                "MAY NOT BE CLAIMED ON THE FIRST DRAW ALONE and makes the confirm "
                "draw at generator seed + 1000 with all arms present MANDATORY. "
                "The headline stands IFF the boolean is TRUE on BOTH draws."),
            "verdict": (
                ("THE HEADLINE STANDS. `entanglement_hurts_T2` is TRUE on BOTH "
                 "draws (first %+.4f, confirm %+.4f, bar -0.06). COMPOSITION IS "
                 "WHERE THE ENTANGLED REGISTER BREAKS, and it breaks in the SAME "
                 "PLACE the product register did (doc 55 SS20.3) -- two encodings, "
                 "same failure locus, now on two independent draws. The mechanism "
                 "is COMPOSITION AGAINST A MULTI-QUBIT REGISTER, not the "
                 "register's contents. STILL NOT: a claim about entangled "
                 "encodings in general (one encoding, one gate, one inventory, "
                 "ONE TASK FAMILY), and NOT quantum advantage in any form "
                 "(doc 56 SS0.1)."
                 % (fv["measured"]["N-in_minus_P-eps_T2"],
                    pairs["T2_intra"]["delta"]))
                if stands else
                ("THE HEADLINE DIES. `entanglement_hurts_T2` was %s on the first "
                 "draw (%+.4f) and %s on the confirm draw (%+.4f). doc 47 SS18's "
                 "precedent applies IN ITS EXACT FORM: A HEADLINE THAT SURVIVED "
                 "ONE DRAW DIED ON THE SECOND. The honest outcome is recorded per "
                 "doc 56 SS11's table and IS NOT SOFTENED: the first draw's "
                 "reading may not be reported as a finding, and the registered "
                 "directional call of doc 56 SS9.4 is REFUTED. What survives is "
                 "the draw-level instability itself, which is a result about the "
                 "measurement's power at 5 seeds and n_test = 300, not about "
                 "entanglement."
                 % (first_hurts, fv["measured"]["N-in_minus_P-eps_T2"],
                    confirm_hurts, pairs["T2_intra"]["delta"]))),
        }
        agree = [k for k, v in bool_cmp.items() if v["agrees"]]
        confirm_report = {
            "first_draw_results": a.first_draw,
            "first_draw_sha256": F["datasets"]["A"]["sha256"],
            "confirm_draw_sha256": ds["A"]["sha256"],
            "generator": ("exp69_datagen.py, THE COMMITTED GENERATOR, UNMODIFIED, "
                          "at generator seed + 1000 = 70000 (doc 56 SS5.1.1). No "
                          "new surface template, no new noun, no new verb -- only "
                          "fresh casts and pairings from the same inventories the "
                          "user already confirmed, so doc 50 SS5.2's "
                          "native-speaker gate stays closed (SS17.1)."),
            "vsuite_rerun_on_this_draw": R.VSUITE,
            "floors_redrawn": ("the S-shuffle floors are RE-DRAWN matched to this "
                               "draw at the same budget. Carrying the first "
                               "draw's floors over would be an unmatched "
                               "comparison in doc 47 SS16.1's sense."),
            "per_arm": per_arm_delta,
            "boolean_agreement": bool_cmp,
            "n_booleans_agreeing": len(agree),
            "n_booleans_compared": len(bool_cmp),
            "booleans_that_flipped": [k for k, v in bool_cmp.items()
                                      if not v["agrees"]],
            "RIDER_VERDICT": rider,
            "witness_margin_note": (
                "doc 56 SS15.3's margins TRANSFER EXACTLY, and the reason is "
                "structural rather than lucky: the witness margin is a function "
                "of the ORDERED (agent, patient) PAIR, and Part A's cast universe "
                "is ALL 80 ordered (bearer, non-bearer) pairs over the SAME "
                "24-noun inventory on both draws -- the fresh draw re-splits and "
                "re-pairs those same casts, it does not create new ones. The "
                "margins here are therefore the FROZEN witness C applied to this "
                "draw's casts, with every held-out item matching a frozen margin "
                "(asserted: zero unmatched). What differs between draws is WHICH "
                "casts land in the test split, not which margins exist."),
        }

    # ---------------------------------------------------------------- git note
    try:
        head = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                               "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        head = "unavailable"
    try:
        doc = subprocess.run(["git", "-C", os.path.expanduser("~/qnlp_private_docs"),
                              "log", "--oneline", "-1", "--",
                              "56_exp70_entangled_encodings.md"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        doc = "unavailable"

    verdicts = {
        "anchors_replicate_exp69": anchors_replicate_exp69,
        "anchors_replicate_exp69_note": anchor_note,
        "floor_gate_all_new_arms": all_floors, "floor_gate_per_arm": floor_gate,
        "floor_readings_disagree_for": readings_disagree,
    }
    verdicts.update(binds)
    verdicts.update(eight)
    verdicts.update({
        "comparator_matches_or_exceeds_all_arms":
            comparator_matches_or_exceeds_all_arms,
        "comparator_matches_or_exceeds_all_arms_STRICT_every_tier":
            comparator_matches_or_exceeds_all_arms_STRICT,
        "comparator_matches_or_exceeds_per_arm_any_tier": cmp_ge_any,
        "comparator_matches_or_exceeds_per_arm_per_tier": cmp_ge_per_tier,
        "bug_hunt_triggered": bug_hunt_triggered, "bug_hunt_per_arm": bug_hunt,
        "run_is_interpretable": bool(
            all_floors if CONFIRM else (anchors_replicate_exp69 and all_floors)),
    })
    directional = {
        "boolean": DIRECTIONAL_CALL,
        "registered_expectation": "TRUE",
        "outcome": bool(eight[DIRECTIONAL_CALL]),
        "delta_measured": pairs["T2_intra"]["delta"], "bar": DELTA_BAR,
        "verdict": ("THE DIRECTIONAL CALL HELD" if eight[DIRECTIONAL_CALL]
                    else "THE DIRECTIONAL CALL DID NOT HOLD"),
        "why_it_was_registered": REGISTERED_EXPECTATION[DIRECTIONAL_CALL],
        "reading": (
            "doc 56 SS9.4 registered exactly ONE directional call, and doc 56 SS18 "
            "item 2 recorded IN ADVANCE that it rests on an ANALOGY: the "
            "barren-plateau literature concerns visible/hidden splits in deep QNNs, "
            "not a two-qubit encoding register, and doc 55 SS20.3's collapse concerns "
            "a PRODUCT register. Two lines of evidence pointing the same way is "
            "better than one, and it is still not a precedent. %s"
            % ("It came back TRUE and doc 56 SS11 makes it the run's headline -- "
               "which MAY NOT BE CLAIMED ON THE FIRST DRAW ALONE: the confirm draw at "
               "generator seed + 1000 with all arms present is MANDATORY (doc 47 "
               "SS18)." if eight[DIRECTIONAL_CALL] else
               "It came back FALSE. A registered directional prediction that fails is "
               "the point of registering it, and it is reported as a refutation of "
               "the prediction rather than quietly dropped.")),
    }
    verdicts["the_one_directional_call_doc56_SS9_4"] = directional
    verdicts["one_sided_pairs_doc56_SS6_4"] = pairs
    verdicts["third_outcome_reading_doc56_SS6_4"] = third_outcome
    verdicts["criterion_bars"] = {
        "delta_bar": DELTA_BAR, "floor_sd_multiple": FLOOR_SD,
        "binds_absolute": BINDS_ABS, "tie_band": TIE_BAND,
        "anchor_tolerance": R.ANCHOR_TOLERANCE,
        "threshold_provenance": (
            "0.06 ~ 2 x the binomial SD at n_test = 300 (sqrt(0.25/300) = 0.02887), "
            "the bar exp61 SS8 / exp67 SS4 / doc 54 SS7 used. INHERITED, never chosen "
            "for doc 56, and the reason exp70 keeps n_test = 300. The 0.65 absolute "
            "bar is doc 54 SS7's, verbatim."),
        "no_criterion_reads_a_bound": (
            "doc 56 SS6's opening sentence and SS10. doc 54 SS17.7 is the standing "
            "reason: the FA per-twin bound moved 0.9250 -> 0.9625 AFTER exp68 "
            "reported and NO STORED BOOLEAN MOVED, precisely because none of them "
            "read a bound."),
    }
    verdicts["registered_expectations_doc56_SS9_4"] = REGISTERED_EXPECTATION
    # EVERY registered expectation must have a measured counterpart. Without this
    # guard a naming drift would silently drop booleans out of the scoring table --
    # which is exactly the failure class doc 56 SS6.4's two-sided form exists to
    # prevent, occurring one level up in the aggregator instead of in the criterion.
    dropped = sorted(set(REGISTERED_EXPECTATION) - set(verdicts))
    assert not dropped, \
        ("doc 56 SS9.4 registers expectations with no measured counterpart: %s -- "
         "the run would be scored against a table missing them" % dropped)
    verdicts["expectation_vs_outcome"] = {
        k: {"registered": REGISTERED_EXPECTATION[k],
            "measured": verdicts.get(k),
            "as_registered": (None if verdicts.get(k) is None else
                              bool(verdicts[k] ==
                                   REGISTERED_EXPECTATION[k].startswith("TRUE")))}
        for k in REGISTERED_EXPECTATION if k in verdicts}
    verdicts["measured"] = {
        "means": m,
        "N-in_minus_P-eps_T1": pairs["T1_intra"]["delta"],
        "N-in_minus_P-eps_T2": pairs["T2_intra"]["delta"],
        "N-across_minus_EA-ctrl_T1": pairs["T1_across"]["delta"],
        "N-across_minus_EA-ctrl_T2": pairs["T2_across"]["delta"],
    }
    verdicts["deliberately_not_registered_doc56_SS6_9"] = (
        "(a) Any criterion reading a bound (SS10). (b) Any T = 1 versus T = 2 "
        "subtraction as a CRITERION -- SS6.7's cc quantities are descriptive "
        "precisely because R1 and R2 are different tasks with different label "
        "distributions (doc 55 SS6.10b). (c) Any advantage-shaped quantity of any "
        "kind (SS0.1).")

    # ========================================================= doc 56 SS11 rows
    fired = []
    if anchors_replicate_exp69 is False:
        fired.append({"row": "either anchor fails |delta| <= 0.002",
                      "reading": "THE RUN STOPS. The exp70 pipeline has drifted from "
                                 "exp69's; doc 56 SS5.3's floor-omission argument is "
                                 "VOID. Hunt the build; interpret nothing.",
                      "may_not_claim": "Nothing, from any arm."})
    failed_floor = [b for b in R.FLOORED if not floor_gate[b]]
    if failed_floor:
        fired.append({"row": "any new arm fails its floor",
                      "reading": "those arms are REPORTED AS FAILED, NOT "
                                 "INTERPRETED; the arms that cleared are still "
                                 "read. Arms: %s" % failed_floor,
                      "may_not_claim": "no contrast involving a failed arm"})
    t1_flat = not eight["entanglement_helps_T1"] and not eight["entanglement_hurts_T1"]
    t2_flat = not eight["entanglement_helps_T2"] and not eight["entanglement_hurts_T2"]
    if t1_flat and t2_flat:
        fired.append({
            "row": "both entanglement_*_T1 FALSE and both *_T2 FALSE",
            "reading": (
                "THE REGISTERED EXPECTATION: intra-register entanglement is "
                "INDISTINGUISHABLE from its map-matched product twin at this power. "
                "With floors cleared at both T this is a CLEAN FOUR-CELL NEGATIVE, "
                "consistent with doc 56 SS2's query-2 prior, and it is the fourth "
                "lever's honest verdict: SUPPLY ADDED AS ENTANGLEMENT WITHIN A WIRE "
                "PAIR DOES NOT CONVERT EITHER. doc 54 SS17.4's statement then "
                "extends: within-a-wire (K, M) no; entangled-within-a-register no; "
                "parallel wires yes at one event, no across two."),
            "may_not_claim": (
                "NOT 'entanglement does not work' -- one encoding, one gate, one "
                "inventory, one task family. NOT a statement about entangled "
                "encodings in general: doc 56 SS2 found the map-matched comparison "
                "ABSENT from the literature, so exp70 adds ONE DATA POINT TO AN "
                "EMPTY CELL.")})
    if eight["entanglement_hurts_T2"] and not eight["entanglement_hurts_T1"] \
            and not eight["entanglement_helps_T1"]:
        fired.append({
            "row": "entanglement_hurts_T2 TRUE, _T1 FALSE",
            "reading": (
                "COMPOSITION IS WHERE THE ENTANGLED REGISTER BREAKS, and it breaks "
                "in the SAME PLACE the product register did (doc 55 SS20.3). Two "
                "encodings, same failure locus => the mechanism is COMPOSITION "
                "AGAINST A MULTI-QUBIT REGISTER, not the register's contents. This "
                "is the registered directional prediction (doc 56 SS9.4) and would "
                "be the run's headline."),
            "may_not_claim": (
                "MAY NOT BE CLAIMED ON THE FIRST DRAW ALONE -- the confirm draw at "
                "generator seed + 1000 with ALL ARMS PRESENT is MANDATORY (doc 47 "
                "SS18, doc 56 SS11)")})
    if eight["entanglement_helps_T2"] and not eight["entanglement_helps_T1"]:
        fired.append({
            "row": "entanglement_helps_T2 TRUE while _T1 FALSE",
            "reading": ("THE UPSIDE SURPRISE: entanglement is what survives "
                        "composition. The most interesting outcome available, and it "
                        "DIRECTLY CONTRADICTS doc 56 SS9.4's directional call."),
            "may_not_claim": ("may not be reported before the confirm draw; may not "
                              "be reported as advantage (doc 56 SS0.1)")})
    for arm in ("N-in-T1", "N-in-T2"):
        if not floor_gate.get(arm) or m[arm] < BINDS_ABS:
            fired.append({"row": "N-in at floor -- doc 56 SS8.4 / SS11",
                          "reading": arbiter[arm]["reading"],
                          "may_not_claim": "see the row's own text"})
    if any(eight[k] for k in ("across_helps_T1", "across_hurts_T1",
                              "across_helps_T2", "across_hurts_T2")):
        fired.append({
            "row": "any across_* boolean TRUE",
            "reading": (
                "A TRAINABILITY / INDUCTIVE-BIAS EFFECT, AND PROVABLY NOTHING MORE. "
                "doc 56 SS4.5: identical function class, identical comparator class, "
                "certified rank 9 = 9 = 9. Interesting, and the follow-up is a design "
                "where an EVENT creates the correlation (doc 56 SS13 item 3)."),
            "may_not_claim": (
                "MAY NOT be reported as a DisCoCirc result, a composition result, or "
                "a claim about language (doc 56 SS4.4). MAY NOT be reported as "
                "capacity.")})
    else:
        fired.append({
            "row": "all across_* FALSE",
            "reading": (
                "the SB readout is INSENSITIVE to a single cross-wire CZ at this "
                "entanglement level -- read together with doc 56 SS4.4's honest note "
                "that the median cast sits at purity 0.9192 and 4 of the 80 casts "
                "fall below concurrence 0.10."),
            "may_not_claim": (
                "may not be reported as 'entanglement between referents does not "
                "matter' -- THE PROBE WAS WEAK BY CONSTRUCTION AND THAT IS MEASURED, "
                "NOT ASSUMED")})
    if not floor_gate.get("P-eps-T2") or m["P-eps-T2"] < BINDS_ABS:
        fired.append({
            "row": "P-eps at T = 2 collapses like R2-W",
            "reading": ("the register-collapse-under-composition finding REPLICATES "
                        "ON A SECOND ANGLE MAP, strengthening doc 55 SS20.3 from one "
                        "map to two (P-eps-T2 %.4f against exp69's R2-W 0.7293)."
                        % m["P-eps-T2"]),
            "may_not_claim": "still one task family; may not be generalised past it"})
    if bug_hunt_triggered:
        fired.append({"row": "any arm exceeds its SS10 bound by > 0.06",
                      "reading": ("BUG HUNT, NOT RESULT (doc 47 SS4). The bounds are "
                                  "LOWER bounds, so a small excess is expected; a "
                                  "large one means the implementation is not "
                                  "computing what doc 56 says. Arms: %s"
                                  % [b for b in R.BASE_ARMS if bug_hunt[b]]),
                      "may_not_claim": "--"})

    out = {
        "experiment": "exp70 -- entangled encodings, the ledger's fourth lever (doc 56)",
        "spec": SPEC, "spec_commit": SPEC_COMMIT,
        "spec_commit_line": doc,
        "question": (
            "Does entanglement IN THE ENCODING change what the circuit LEARNS, at a "
            "scale where we can compute exactly what it could learn? The first three "
            "levers -- K, M and width -- all moved supply INSIDE the product-state "
            "family; exp70 is the first that leaves it (doc 56 SS0)."),
        "aggregated_from": "%s/*.json (tag=%r)" % (R.CKPT_DIR, a.tag),
        "aggregator": os.path.basename(os.path.abspath(__file__)),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "datasets": ds,
        "no_datagen_doc56_SS5_1": P["no_datagen_doc56_SS5_1"],
        "predictions_file": PREDICTIONS, "smoke_results": SMOKE,
        "anchor_results": ANCHORS, "vsuite_results_INHERITED_FROM_EXP69": VSUITE,
        "arms_run": R.ARMS, "n_seeds": R.N_SEEDS, "n_runs": n_runs,
        "sd_convention": "sample sd, ddof=1",
        "training": {"lr": R.LR, "batch": R.BATCH, "max_epochs": R.MAX_EPOCHS,
                     "optimiser": "Adam", "rng_base": R.RNG_BASE,
                     "model_selection": "best in-distribution val accuracy",
                     "seeds_vary": ("OPTIMISER INITIALISATION ONLY -- the features "
                                    "are deterministic constants (doc 56 SS5.3)"),
                     "lineage": ("the exp60 / exp61 / exp65 / exp68 / exp69 loop, "
                                 "unchanged; the entanglers add DATA-VALUED constant "
                                 "gates and ZERO trainable parameters")},
        "arms": arms, "floors": floors, "floor_clearance": clear,
        "anchor_gate": ({"status": "INAPPLICABLE ON A FRESH DRAW",
                         "note": anchor_note,
                         "first_draw_gate": {
                             "anchors_replicate_exp69":
                                 AN["anchors_replicate_exp69"],
                             "all_bit_exact": AN["all_bit_exact"]}}
                        if CONFIRM else AN),
        "comparator": {
            "source": COMPARATOR,
            "tiers_per_family": C["tiers_per_family"],
            "arms": cmp_per_tier, "floors": cmp_floor_per_tier,
            "floor_clearance": cmp_clear,
            "comparator_minus_circuit": cmp_minus_circuit,
            "two_tier_reading_doc56_SS8_1": two_tier_reading,
            "reading_rule": C["reading_rule_doc56_SS8_1"],
            "clifford_consequence": C["clifford_consequence_doc56_SS4_5"],
            "why_the_outer_square": C["why_the_outer_square_not_the_amplitudes_doc56_SS8_2"],
            "arbiter_doc56_SS8_4": arbiter,
            "function_class_qualifier": C["function_class_qualifier"]},
        "bounds_context_only_doc56_SS10": {
            b: {"family": R.ARM_CFG[b]["family"], "bound_of_record": bounds[b],
                "measured": m[b], "over_bound_by": over[b]} for b in R.BASE_ARMS},
        "composition_cost_doc56_SS6_7": composition,
        "predictions_vs_measured_doc56_SS9": predictions_report,
        "margin_accuracy_doc56_SS15_3": s153,
        "purity_contrast_doc56_SS15_1_SS15_2": s151,
        "error_overlap_doc56_SS15_4": s195,
        "entanglement_stratification_doc56_SS6_8": s_strat,
        "reported_not_registered_doc56_SS6_8": {
            "rule": ("doc 56 SS6.8: REPORTED, NOT REGISTERED, and deliberately so. "
                     "Per-q held-out accuracy at T = 2, per-surface-order accuracy, "
                     "the val-heldout selection gap per seed, per-verb-pair accuracy, "
                     "P-eps against exp69's R-W (SS4.3 -- DIFFERENT MAPS, descriptive "
                     "only), and the per-item entanglement stratification. NONE OF "
                     "THESE MAY BE PROMOTED ONTO A CRITERION AFTER THE FACT."),
            "surface_order_caveat": (
                "exp69's compiled circuit is INVARIANT to the SVO/VSO alternation by "
                "construction (DERIVATION DIFFERS / DENOTATION EQUAL), and exp70 "
                "inherits that compilation path unchanged. The per-surface-order "
                "numbers are a DATA-COMPOSITION statistic and are NOT evidence about "
                "word-order processing."),
            "P_eps_vs_exp69_R_W_DESCRIPTIVE_ONLY": {
                "P-eps-T1": m["P-eps-T1"], "exp69_R1-W": 0.9820,
                "P-eps-T2": m["P-eps-T2"], "exp69_R2-W": 0.7293,
                "note": ("doc 56 SS4.3: they differ in ANGLE MAP, so differencing "
                         "them is a POPULATION MISMATCH in doc 47 SS18.5's sense. "
                         "Reported because the map change is real and a reader will "
                         "ask; NOT a criterion because it cannot be one. A0-W is the "
                         "map-matched comparison and it is an ANCHOR, not a "
                         "contrast.")},
            "per_arm": s68},
        "verdicts": verdicts,
        "confirm_draw_doc56_SS11": confirm_report,
        "outcome_rows_doc56_SS11": fired,
        "meta_notes": {
            "scope_doc56_SS12": [
                "NOT QUANTUM ADVANTAGE, and that is PROVABLE, not modest. doc 52 SS1 "
                "closed the advantage question for the product-state family with an "
                "ARGUMENT (the kernel factorises, verified to 3.3e-16), and doc 56 "
                "SS0.1 stays firmly on the simulable side of the entangled family doc "
                "52 SS1.3 named: four qubits, every state in closed form, every "
                "comparator computed from those closed forms. NO EXP70 NUMBER MAY "
                "APPEAR IN A SENTENCE CONTAINING 'ADVANTAGE', AND THIS HOLDS FOR THE "
                "ENTANGLED ARMS MOST OF ALL -- stepping into the family doc 52 SS1.3 "
                "named is not the same as reaching the regime it described.",
                "NOT 'meaning'. One category, one task family, 24 nouns, 2 referents, "
                "simulation. doc 50 SS7's ceiling stands.",
                "NOT A CLAIM ABOUT DISCOCIRC, FROM THE E-ACROSS ARM. doc 56 SS4.4: "
                "that arm has NO LICENSE in the semantics -- in DisCoCirc referents "
                "are introduced as INDEPENDENT states and correlation is created by "
                "EVENTS -- and it is registered as a MECHANISM PROBE, with the "
                "non-linguistic status stated in the doc and here (orchestrator "
                "ruling 3).",
                "NOT 'entanglement works' or 'entanglement fails'. One encoding form, "
                "one entangler per site, one inventory. doc 54 SS17.3's prohibition "
                "applies: AN ENTANGLED-ENCODING SUBFAMILY WITH TWO MEMBERS SUPPORTS A "
                "RECORD, NOT A DESIGN RULE, and any claim must carry that count.",
                "NOT novel-filler generalisation. exp69's held-out-PAIRING split, "
                "inherited unchanged; V2 is reported un-gated for the same reason.",
                "NOT a statement about AraVec. PC1/PC2 of a 300-dim vector over 44 "
                "words explain 20.7 % of its variance.",
                "The bounds bound THIS feature map and THIS inventory, and NO "
                "criterion reads against them.",
            ],
            "the_clifford_finding_doc56_SS4_5": (
                "THE SHARPEST SINGLE RESULT OF THE PRE-WORK, and it is a NEGATIVE "
                "ABOUT AN INSTRUMENT: CZ is Clifford, so conjugating a general "
                "readout by it PERMUTES the Pauli coordinates and leaves the readable "
                "function space INVARIANT. Three consequences, all registered BEFORE "
                "the run: (1) N-across's grade-1 bound EQUALS EA-ctrl's BY PROOF; (2) "
                "FOURIER CLOSURE IS A NULL INSTRUMENT for N-across, so doc 56 SS7.2 "
                "replaces it rather than passing a vacuous gate; (3) the joint-state "
                "comparator is the SAME FUNCTION CLASS for both, so any measured "
                "difference is TRAINABILITY OR INDUCTIVE BIAS, NEVER CAPACITY. doc 49 "
                "SS8 has ADOPTED the resulting checklist entry (orchestrator ruling "
                "6): before asserting a validity gate, check whether the intervention "
                "is a SYMMETRY of the quantity the gate measures -- A GATE THAT "
                "CANNOT FAIL IS NOT A GATE."),
            "e_across_weak_by_construction_doc56_SS18_4": (
                "FOUR OF THE 80 E-ACROSS CASTS SIT BELOW CONCURRENCE 0.10 and are NOT "
                "excluded (doc 56 SS7.3, orchestrator ruling 7). The arm is therefore "
                "weaker than its name suggests on 5 %% of its casts, and the honest "
                "reading of any across_* null must carry that. The per-item purity "
                "gate is the working validity instrument for that arm, and the "
                "entanglement stratification above exists so this can be read "
                "post-hoc without a rerun. **AND THE HELD-OUT SPLIT IS WEAKER STILL, "
                "which doc 56 could not have known and which is measured here rather "
                "than inherited: over the 25 HELD-OUT casts the E-across concurrence "
                "runs %.4f-%.4f with median %.4f and %d of 25 casts (%d of 300 items) "
                "fall below 0.10, against doc 56 SS4.4's inventory-wide median of "
                "0.4019 and 4 of 80. Any across_* reading must be taken against the "
                "held-out figure, not the inventory-wide one -- the probe that was "
                "already weak by construction is weaker on the split it is actually "
                "scored on.**"
                % (SM["across_validity"]["per_T"]["T1"]["heldout_concurrence"]["min"],
                   SM["across_validity"]["per_T"]["T1"]["heldout_concurrence"]["max"],
                   SM["across_validity"]["per_T"]["T1"]["heldout_concurrence"]["median"],
                   2,
                   SM["across_validity"]["per_T"]["T1"]["heldout_concurrence"][
                       "n_items_below_0.10"])),
            "heldout_entanglement_measured": {
                "e_across": SM["across_validity"]["per_T"]["T1"]["heldout_concurrence"],
                "e_across_heldout_casts_below_0.10": 2,
                "e_across_all_casts_below_0.10_doc56": 4,
                "e_in_inventory_min": SM["features"]["eps_map"]["concurrence_min"],
                "e_in_items_below_gate": 0},
            "confirm_draw_obligation_doc56_SS5_1_1": (
                "doc 56 SS11 / SS17.3 item 8 mandate a CONFIRM DRAW at generator seed "
                "+ 1000 FOR ANY POSITIVE, with ALL ARMS PRESENT, re-running the "
                "COMMITTED exp69_datagen.py UNMODIFIED -- no new template, no new "
                "noun, no new verb, so doc 50 SS5.2's native-speaker gate stays "
                "closed. The full V-suite and every SS7 assertion re-run on the fresh "
                "draw, and the S-shuffle FLOORS ARE RE-DRAWN MATCHED TO IT (carrying "
                "them over would be doc 47 SS16.1's unmatched comparison). IT IS NOT "
                "INCLUDED IN THIS FILE. Any positive reported here is a FIRST-DRAW "
                "result, and doc 47 SS18 is the standing reminder that a headline "
                "which survived one draw died on the second."),
            "anchor_merge_orchestrator_ruling_2": AN["merge_note"],
            "no_wazn_variant_doc56_SS5_2": (
                "exp69's wazn arms (R1-M / R2-M) have NO exp70 variant. There is no "
                "wazn register here and inventing one would change the FEATURE "
                "SOURCE, which is doc 55 SS13 item 5's refusal, inherited."),
        },
        "git_note": {
            "repo_head": head, "spec_commit": SPEC_COMMIT,
            "spec_commit_line": doc,
            "note": ("doc 56 is committed at %s in qnlp_private_docs and is the "
                     "binding registration, including its status-header orchestrator "
                     "rulings 1-7. THIS RUN WROTE NO GIT STATE; the orchestrator "
                     "reviews and commits." % SPEC_COMMIT),
        },
        "wall_time_sec": time.time() - t0,
    }
    path = a.out or fresh_results_path(results_base)
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ------------------------------------------------------------------ console
    print("[exp70] %d runs, %d arms x %d seeds" % (n_runs, len(R.ARMS), R.N_SEEDS))
    print("\n  SS6.1 anchors_replicate_exp69 : %s  (%s)"
          % (anchors_replicate_exp69,
             "INAPPLICABLE on a fresh draw; first draw was bit-exact"
             if CONFIRM else "all bit-exact = %s" % AN["all_bit_exact"]))
    for arm in ([] if CONFIRM else R.ANCHOR_ARMS):
        p = AN["per_arm"][arm]
        print("        %-11s vs exp69 %-5s  max |delta| %.2e  bit-exact=%s"
              % (arm, p["exp69_arm"], p["max_abs_per_seed_delta"], p["bit_exact"]))
    print("\n  arm            T  enc        held-out mean +/- sd    floor            clears")
    for b in R.BASE_ARMS:
        fl = ("%.4f+/-%.4f" % (floors[b]["mean"], floors[b]["sd"])
              if b in floors else "  (anchor)     ")
        cl = ("%-5s (%.1f sd)" % (floor_gate[b],
                                  clear[b]["sd_units_over_floor_using_floor_sd"])
              if b in clear else "n/a")
        print("  %-14s %d  %-9s  %.4f +/- %.4f   %s  %s"
              % (b, R.ARM_CFG[b]["T"], R.ARM_CFG[b]["enc"], arms[b]["mean"],
                 arms[b]["sd"], fl, cl))
    print("\n  SS6.3 the six binds:")
    for k in sorted(binds):
        print("    %-24s %s" % (k, binds[k]))
    print("\n  SS6.4 THE EIGHT ONE-SIDED BOOLEANS:")
    for k in ("entanglement_helps_T1", "entanglement_hurts_T1",
              "entanglement_helps_T2", "entanglement_hurts_T2",
              "across_helps_T1", "across_hurts_T1",
              "across_helps_T2", "across_hurts_T2"):
        print("    %-24s %-5s   (registered: %s)"
              % (k, eight[k], REGISTERED_EXPECTATION[k].split(" --")[0]))
    for k, v in third_outcome.items():
        print("    %-10s %s" % (k, v))
    print("\n  THE ONE DIRECTIONAL CALL (%s): registered TRUE, measured %s -> %s"
          % (DIRECTIONAL_CALL, directional["outcome"], directional["verdict"]))
    print("\n  SS6.7 composition costs (DESCRIPTIVE):")
    for k, v in ccs.items():
        print("    %-14s %+.4f  (sd %.4f)" % (k, v["mean"], v["sd"]))
    for k, v in interactions.items():
        print("    %-14s %+.4f" % (k, v["mean"]))
    print("\n  SS6.5 comparator (tie band %.2f):" % TIE_BAND)
    for b in R.BASE_ARMS:
        print("    %-14s %s" % (b, {t: "%+.4f%s" % (cmp_minus_circuit[b][t],
                                                    "" if cmp_ge_per_tier[b][t] else " X")
                                    for t in sorted(cmp_minus_circuit[b])}))
    print("    all arms (any tier) = %s ; STRICT every tier = %s"
          % (comparator_matches_or_exceeds_all_arms,
             comparator_matches_or_exceeds_all_arms_STRICT))
    print("\n  SS15.3 Spearman (item / cast):")
    for b in R.BASE_ARMS:
        print("    %-14s %s / %s  (%d distinct margins)"
              % (b, "%.4f" % s153[b]["spearman_item_level"]
                 if s153[b]["spearman_item_level"] is not None else "n/a",
                 "%.4f" % s153[b]["spearman_cast_level"]
                 if s153[b]["spearman_cast_level"] is not None else "n/a",
                 s153[b]["n_distinct_margins"]))
    print("\n  SS6.6 bug_hunt_triggered = %s" % bug_hunt_triggered)
    print("  SS11 rows fired: %s" % [f["row"] for f in fired])
    if confirm_report:
        r = confirm_report["RIDER_VERDICT"]
        print("\n  ===== doc 56 SS11 CONFIRM DRAW vs FIRST DRAW =====")
        print("  arm            first    confirm   delta")
        for b in R.BASE_ARMS:
            d = confirm_report["per_arm"][b]
            print("  %-14s %.4f   %.4f   %+.4f%s"
                  % (b, d["first_draw_mean"], d["confirm_mean"],
                     d["delta_confirm_minus_first"],
                     "   [un-floored baseline]"
                     if d["is_unfloored_baseline_on_confirm"] else ""))
        print("  booleans agreeing: %d/%d   flipped: %s"
              % (confirm_report["n_booleans_agreeing"],
                 confirm_report["n_booleans_compared"],
                 confirm_report["booleans_that_flipped"] or "none"))
        print("\n  RIDER: %s  first=%s (%+.4f)  confirm=%s (%+.4f)"
              % (r["boolean"], r["first_draw"], r["first_draw_delta"],
                 r["confirm_draw"], r["confirm_draw_delta"]))
        print("  HEADLINE STANDS = %s" % r["headline_stands"])
        print("  %s" % r["verdict"])
    print("\n[exp70] wrote %s (%.1fs)" % (path, out["wall_time_sec"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
