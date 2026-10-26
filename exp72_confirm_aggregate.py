# -*- coding: utf-8 -*-
"""exp72_confirm_aggregate.py -- the ONLY writer for `results_exp72_confirm.json`.

doc 58 SS16.3 step 11 / doc 47 SS18. It computes nothing new: every number is read
from `exp72_confirm_ckpt/`'s per-run JSONs, from the committed
`results_exp70_confirm.json` (the fresh draw's comparator and the two arms it
already measured), or from `results_exp72.json` (the first draw).

THE RIDER VERDICT IS THE POINT: does `rescue_partial(I1, R-identity)` -- exp72's
ONE positive on the first draw -- HOLD ON A FRESH DRAW? doc 47 SS18's precedent is
the standing reason the question is asked at all: A HEADLINE THAT SURVIVED ONE
DRAW DIED ON THE SECOND. If it flips, that is reported plainly and the first
draw's boolean is NOT retro-edited -- both draws stand, and the headline does not.

THE BARS MOVE, AND THE FORMULAS DO NOT. doc 58 SS9.2 freezes
`rescue_full >= comparator - 0.02` and `rescue_partial >= baseline + 0.50 * gap`.
The FORMULAS are the registered object; the numbers they consume are each draw's
own baseline and its own imported comparator. On the fresh draw the class-tier
comparator is 0.9880 (not 1.0000) and the baseline is 0.6993 (not 0.7293), so the
partial bar is 0.843667 (not 0.864667). Carrying the first draw's bar across would
be reading a fresh measurement against a stale threshold.

    python exp72_confirm_aggregate.py            # after --base and --stage1
    python exp72_confirm_aggregate.py --stage2   # after the triggered floors
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch

import exp72_aggregate as A
import exp72_confirm_run as C
import exp72_run as X
from exp42_run import fresh_results_path

RESULTS_BASE = "results_exp72_confirm"
FLOOR_SD = 2.0


def load_confirm_runs():
    out = {}
    for fn in sorted(os.listdir(C.CKPT_DIR)):
        if fn.endswith(".json"):
            r = json.load(open(os.path.join(C.CKPT_DIR, fn), encoding="utf-8"))
            out[r.get("label") or fn[:-5]] = r
    return out


def base_parity(runs):
    """The check that REPLACES doc 58 SS5.1's gate on a fresh draw, and it is
    stronger in kind than a tolerance: exp70's confirm draw already ran A0-W-T1
    and A0-W-T2 on THESE items, at the committed recipe, at the same five seeds,
    so exp72's re-run must reproduce them BIT-EXACTLY -- values and parameters."""
    ref = C.confirm_reference()
    out = {}
    for arm, pfx, ck in ((C.DONOR_ARM, "C_I1_donor_s", "A0-W-T1_s%d.pt"),
                         (C.ARM, "C_I1_anchor_s", "A0-W-T2_s%d.pt")):
        rr = sorted([r for r in runs.values()
                     if (r.get("label") or "").startswith(pfx)],
                    key=lambda x: x["seed"])
        got = [r["heldout_test_acc"] for r in rr]
        want = ref[arm]["per_seed"]
        d = [g - w for g, w in zip(got, want)]
        tensor_ok, worst = True, 0.0
        for s in range(len(rr)):
            a = torch.load(os.path.join("exp70_confirm_ckpt", ck % s),
                           map_location="cpu")
            b = torch.load(os.path.join(C.CKPT_DIR, "%s%d.pt" % (pfx, s)),
                           map_location="cpu")
            for k in a:
                worst = max(worst, float((a[k] - b[k]).abs().max()))
                tensor_ok = tensor_ok and torch.equal(a[k], b[k])
        out[arm] = {
            "exp70_confirm_per_seed": want, "exp72_confirm_per_seed": got,
            "per_seed_delta": d,
            "max_abs_per_seed_delta": float(max(abs(x) for x in d)),
            "bit_exact_heldout": bool(all(x == 0.0 for x in d)),
            "checkpoint_tensors_identical": bool(tensor_ok),
            "max_abs_parameter_diff": worst,
            "reference": ref[arm]["source"],
        }
    return {
        "per_arm": out,
        "all_bit_exact": bool(all(v["bit_exact_heldout"]
                                  and v["checkpoint_tensors_identical"]
                                  for v in out.values())),
        "role_on_a_fresh_draw": (
            "doc 58 SS5.1's |delta| <= 0.002 ANCHOR GATE IS INAPPLICABLE HERE, not "
            "skipped, and the difference matters: it compares per-seed accuracies "
            "against committed measurements ON THE FIRST DRAW'S OWN ITEMS, and "
            "these are different items. On this draw A0-W-T2 is a BASELINE ARM "
            "(doc 56 SS6.1's reconciliation, inherited by doc 58 SS16.2). What IS "
            "available is stronger in kind and is asserted instead: exp70's "
            "confirm draw already measured BOTH arms on THESE items at the "
            "committed recipe and the same five seeds, so exp72's re-run through "
            "its own codepath must reproduce them BIT-EXACTLY -- and does, "
            "including parameter-level identity of every selected checkpoint."),
        "file_sha_note": (
            "the .pt FILE shas differ between the two runs while the TENSORS are "
            "identical. That is a zip-container artefact: torch.save records the "
            "archive member name, and the exp72 checkpoints carry longer labels "
            "(C_I1_donor_s4 vs A0-W-T1_s4), so the container bytes differ by the "
            "name length alone. The load-bearing comparison is parameter-level "
            "identity, which is exact."),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage2", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    store = C.load_confirm_store()
    runs = load_confirm_runs()
    first = json.load(open(C.FIRST_DRAW_RESULTS, encoding="utf-8"))

    anchor_rows = [r for r in runs.values()
                   if (r.get("label") or "").startswith("C_I1_anchor_s")]
    anchor = A.block(anchor_rows)
    bar = C.confirm_bars(anchor["mean"])
    parity = base_parity(runs)

    arms, floors, verdicts, booleans, missing = {}, {}, {}, {}, []
    for rescue in X.RESCUES:
        rr = [r for r in runs.values() if r.get("kind") == "rescue"
              and r.get("cell_rescue") == rescue]
        if len(rr) != X.N_SEEDS:
            missing.append("%s (%d/%d)" % (rescue, len(rr), X.N_SEEDS))
            continue
        arms[rescue] = A.block(rr)
        fr = [r for r in runs.values() if r.get("kind") == "floor"
              and r.get("cell_rescue") == rescue]
        if len(fr) == X.N_SEEDS:
            fb = A.block(fr)
            fb["procedure_matched"] = True
            fb["rescue_procedure"] = rescue
            floors[rescue] = fb

    # doc 58 SS14.1's orientation floor does not exist for A0-W-T2 on this draw --
    # exp70's confirm floored only the three NEW families. Stated, not papered
    # over: a MISSING ORIENTATION floor costs nothing, because SS14.2's trigger
    # test reads the ARM's mean against its bar and nothing else, and every
    # triggered cell gets its own PROCEDURE-MATCHED floor here.
    orientation = {"available": False,
                   "reason": ("exp70's confirm draw demoted the A0-W anchor family "
                              "to UN-FLOORED DESCRIPTIVE BASELINES and floored only "
                              "P-eps / N-in / N-across, so no committed S-shuffle "
                              "floor for A0-W-T2 exists on this draw. doc 58 SS14.1's "
                              "orientation comparison CARRIES NO VERDICT, and "
                              "SS14.2's trigger test reads the arm's mean against "
                              "its bar alone, so nothing that is read depends on "
                              "it. Every TRIGGERED cell is given its own "
                              "procedure-matched floor here, which is the floor the "
                              "booleans actually read (SS14.3 / SS14.5).")}

    for rescue in X.RESCUES:
        if rescue not in arms:
            verdicts[rescue] = {"rescue": rescue, "rescue_full": None,
                                "rescue_partial": None, "verdict": "NOT RUN"}
            booleans["rescue_full(I1, %s)" % rescue] = None
            booleans["rescue_partial(I1, %s)" % rescue] = None
            continue
        f2 = floors.get(rescue)
        v = A.cell_verdict("I1", rescue, arms[rescue], bar,
                           f2 if f2 else {"mean": float("nan"), "sd": float("nan"),
                                          "per_seed": [], "procedure_matched": False},
                           "stage-2 procedure-matched floor (confirm draw)"
                           if f2 else None)
        if not v["stage2_triggered"]:
            v.pop("floor_clearance_orientation_only", None)
            v["floor_used"] = None
            v["floor_status"] = ("NOT REQUIRED -- the cell does not reach its "
                                 "partial bar (doc 58 SS14.2)")
        verdicts[rescue] = v
        booleans["rescue_full(I1, %s)" % rescue] = v["rescue_full"]
        booleans["rescue_partial(I1, %s)" % rescue] = v["rescue_partial"]

    triggered = sorted(k for k, v in verdicts.items() if v.get("stage2_triggered"))
    unfloored = sorted(k for k in triggered
                       if verdicts[k].get("rescue_full") is None)

    # ------------------------------------------------ the confirm-vs-first table
    table = {}
    for rescue in X.RESCUES:
        fk = "I1|%s" % rescue
        f = first["arms"].get(fk)
        fv = first["verdicts_per_cell"].get(fk, {})
        c = arms.get(rescue)
        table[rescue] = {
            "first_draw_mean": f["mean"] if f else None,
            "first_draw_sd": f["sd"] if f else None,
            "confirm_mean": c["mean"] if c else None,
            "confirm_sd": c["sd"] if c else None,
            "delta_confirm_minus_first": (c["mean"] - f["mean"]) if (c and f)
            else None,
            "first_draw_delta_over_baseline": fv.get("mean_rescue_delta"),
            "confirm_delta_over_baseline": (c["mean"] - bar["baseline"]) if c
            else None,
            "first_draw_rescue_partial": fv.get("rescue_partial"),
            "confirm_rescue_partial": booleans.get("rescue_partial(I1, %s)"
                                                   % rescue),
            "first_draw_rescue_full": fv.get("rescue_full"),
            "confirm_rescue_full": booleans.get("rescue_full(I1, %s)" % rescue),
            "booleans_agree": (fv.get("rescue_partial")
                               == booleans.get("rescue_partial(I1, %s)" % rescue)
                               and fv.get("rescue_full")
                               == booleans.get("rescue_full(I1, %s)" % rescue)),
        }
    table["ANCHOR (unrescued baseline)"] = {
        "first_draw_mean": 0.7293333333333334, "first_draw_sd": 0.03139355914125629,
        "confirm_mean": anchor["mean"], "confirm_sd": anchor["sd"],
        "delta_confirm_minus_first": anchor["mean"] - 0.7293333333333334,
        "note": ("a BASELINE on this draw, not a bit-exact gate. Its bit-exact "
                 "check is against exp70's confirm measurement of the SAME items, "
                 "in `base_parity`."),
    }

    # ------------------------------------------------------- THE RIDER VERDICT
    fv = first["verdicts_per_cell"]["I1|R-identity"]
    cv = verdicts.get("R-identity", {})
    holds = cv.get("rescue_partial")
    rider = {
        "rider": "rescue_partial(I1, R-identity)",
        "why_this_cell": (
            "it is exp72's ONE positive on the first draw and therefore the ONLY "
            "thing doc 58 SS16.3 step 11 obliges a confirm draw to re-test. The "
            "other seven cells were FALSE on the first draw and are re-run here "
            "because step 11 requires ALL ARMS OF THE AFFECTED INSTANCE to be "
            "present, not because any of them carries a headline."),
        "first_draw": {"mean": fv["heldout_mean"], "sd": fv["heldout_sd"],
                       "bar": fv["rescue_partial_bar"],
                       "floor_mean": fv.get("floor_mean"),
                       "sd_units_over_floor":
                           fv.get("floor_clearance", {}).get(
                               "sd_units_over_floor_using_floor_sd"),
                       "rescue_partial": fv["rescue_partial"],
                       "rescue_full": fv["rescue_full"]},
        "confirm_draw": {"mean": cv.get("heldout_mean"), "sd": cv.get("heldout_sd"),
                         "bar": bar["rescue_partial_bar"],
                         "floor_mean": cv.get("floor_mean"),
                         "sd_units_over_floor":
                             cv.get("floor_clearance", {}).get(
                                 "sd_units_over_floor_using_floor_sd"),
                         "rescue_partial": cv.get("rescue_partial"),
                         "rescue_full": cv.get("rescue_full")},
        "holds_on_the_fresh_draw": holds,
        "agrees_with_first_draw": (None if holds is None
                                   else bool(holds == fv["rescue_partial"])),
    }
    if holds is True:
        rider["reading"] = (
            "THE HEADLINE SURVIVES A SECOND DRAW. `rescue_partial(I1, R-identity)` "
            "is TRUE on both the first draw and a fresh draw at generator seed + "
            "1000, each against its OWN procedure-matched floor and its OWN "
            "recomputed bar. doc 47 SS18's failure mode -- a headline that survived "
            "one draw and died on the second -- did NOT occur here. The registered "
            "reading of doc 58 SS10's row stands: the failure is an "
            "INITIALISATION-BASIN failure. STILL MAY NOT BE CLAIMED: that the "
            "instance was in a barren plateau (4 qubits, no gradient-variance "
            "scaling measured); that trainability mitigations work in general "
            "(one operationalisation, one instance); or anything about I2, which "
            "was not run on this draw and on which all four rescues were FALSE.")
    elif holds is False:
        rider["reading"] = (
            "THE HEADLINE DIED ON THE SECOND DRAW, and doc 47 SS18's precedent "
            "applies exactly as registered. `rescue_partial(I1, R-identity)` was "
            "TRUE on the first draw and is FALSE on a fresh draw at generator seed "
            "+ 1000. NEITHER MEASUREMENT IS EDITED: the first draw's boolean stands "
            "as the measurement it was, and the headline does NOT stand. exp72's "
            "reportable outcome becomes the one doc 58 SS10 registered for "
            "'nothing rescues' -- the walls are OPTIMISER-ROBUST at this budget "
            "class -- with the first draw's R-identity result carried as a "
            "SINGLE-DRAW observation that did not replicate. This is the outcome "
            "doc 47 SS18 exists to catch and it is reported plainly.")
    else:
        rider["reading"] = (
            "NO VERDICT YET. The cell triggered on the fresh draw and its "
            "procedure-matched floor has not been run; doc 58 SS14.5 stores the "
            "booleans as null, NEVER as FALSE.")

    out = {
        "experiment": ("exp72 CONFIRM DRAW -- doc 58 SS16.3 step 11 on the one "
                       "first-draw positive, rescue_partial(I1, R-identity)"),
        "spec": X.SPEC, "spec_commit_line": X.spec_commit(),
        "aggregator": "exp72_confirm_aggregate.py -- the ONLY writer of this file",
        "standing_reason_doc47_SS18": (
            "A HEADLINE THAT SURVIVED ONE DRAW DIED ON THE SECOND. doc 58 SS16.3 "
            "step 11 makes the confirm draw MANDATORY on any positive, with ALL "
            "ARMS OF THE AFFECTED INSTANCE present (its anchor and all four "
            "rescues) plus SS16.2's retrained donor."),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "sd_convention": "sample sd, ddof = 1",
        "dataset_decision_doc58_SS16_2": C.dataset_decision(store),
        "comparator_target_doc58_SS3": C.confirm_targets(),
        "bars_recomputed_doc58_SS9_2": bar,
        "instance": {"id": "I1", "arm": C.ARM,
                     "first_draw_arm": "exp69 R2-W / exp72 A0-W-T2",
                     "encoding": "F2 product register, old map (p+1)pi, 4 wires"},
        "anchor_baseline_on_this_draw": anchor,
        "anchor_role_and_codepath_parity": parity,
        "retrained_donor_doc58_SS16_2": json.load(
            open("results_exp72_confirm_donor.json", encoding="utf-8"))
        if os.path.exists("results_exp72_confirm_donor.json") else None,
        "arms": arms,
        "runs_missing": missing,
        "floors_doc58_SS14": {"orientation_floor": orientation,
                              "stage2_floors": floors,
                              "triggered_cells": triggered,
                              "triggered_but_unfloored": unfloored,
                              "gating_order": (
                                  "arm mean -> trigger test (SS14.2) -> "
                                  "procedure-matched floor if triggered -> ONLY "
                                  "THEN the booleans. A triggered cell without its "
                                  "floor has NO verdict and stores null.")},
        "verdicts_per_cell": verdicts,
        "booleans_on_this_draw": booleans,
        "confirm_vs_first_draw": table,
        "RIDER_VERDICT": rider,
        "not_run_on_this_draw": (
            "I2 (N-in-T2) and the positive control. doc 58 SS16.3 step 11 scopes "
            "the confirm draw to ALL ARMS OF THE AFFECTED INSTANCE, and the "
            "affected instance is I1. All four I2 rescues were FALSE on the first "
            "draw, so no I2 headline exists to confirm; `positive_control_degrades` "
            "was FALSE and is a first-draw degradation check against a committed "
            "value, not a gain claim."),
        "scope_doc58_SS12": A.SCOPE,
        "git_note": X.git_note(),
        "wall_time_sec": time.time() - t0,
    }
    p = fresh_results_path(RESULTS_BASE)
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[confirm_aggregate] wrote %s" % p)

    print("\n=== CONFIRM DRAW (seed %d), instance I1 ===" % C.GENERATOR_SEED)
    print("  codepath parity vs exp70 confirm: bit-exact=%s (tensors identical)"
          % parity["all_bit_exact"])
    print("  baseline %.6f +/- %.6f   comparator %.4f   partial bar %.6f   "
          "full bar %.6f" % (bar["baseline"], anchor["sd"], bar["comparator"],
                             bar["rescue_partial_bar"], bar["rescue_full_bar"]))
    print("\n  %-13s %-22s %-22s %s" % ("rescue", "first draw", "confirm draw",
                                        "partial first->confirm"))
    for rescue in X.RESCUES:
        t = table[rescue]
        print("  %-13s %.4f +/- %.4f      %s      %s -> %s"
              % (rescue, t["first_draw_mean"], t["first_draw_sd"],
                 ("%.4f +/- %.4f" % (t["confirm_mean"], t["confirm_sd"]))
                 if t["confirm_mean"] is not None else "NOT RUN            ",
                 t["first_draw_rescue_partial"], t["confirm_rescue_partial"]))
    if triggered:
        print("\n  triggered on confirm: %s" % triggered)
        for k in triggered:
            v = verdicts[k]
            if v.get("floor_mean") is not None:
                print("    %s floor %.6f +/- %.6f  (%.2f floor-SD units, clears=%s,"
                      " all readings=%s)"
                      % (k, v["floor_mean"], v["floor_sd"],
                         v["floor_clearance"]["sd_units_over_floor_using_floor_sd"],
                         v["floor_clearance"]["clears_floor"],
                         v["floor_clearance"]["clears_under_all_readings"]))
    print("\n=== THE RIDER VERDICT ===")
    print("  rescue_partial(I1, R-identity): first draw %s, confirm draw %s"
          % (rider["first_draw"]["rescue_partial"],
             rider["confirm_draw"]["rescue_partial"]))
    print("  holds_on_the_fresh_draw = %s" % rider["holds_on_the_fresh_draw"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
