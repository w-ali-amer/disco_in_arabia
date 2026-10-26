"""exp73 CONFIRM aggregation -- doc 61 SS21.4 (+ SS21.4.1), pinned before the run.

Reads the 45 confirm checkpoints in `exp73_confirm_ckpt/` and decides the two
fired positives on the RULES PINNED IN doc 61 SS21.4 as corrected by SS21.4.1:

  1. init_helps_A3_T2 CONFIRMED iff mean(A3-T2-identity) - mean(A3-T2-uniform)
     >= 0.06 on the confirm draw (the registered stage-1 bar, unchanged).
  2. amplitude_hurts_T2 CONFIRMED iff best confirm A3-T2 amplitude mean <=
     fresh-draw anchor mean (A0-W-T2 on this draw) - 0.06 [PRIMARY];
     the same comparison against the frozen committed 0.8340 is SECONDARY,
     reported not decisive.
  3. No verdict is read for a cell whose re-drawn floor is missing; the binds
     rule is the stage rule unchanged (mean >= 0.65 AND mean >= floor_mean +
     2 x floor_sd), and doc 61 SS15.5's contamination branch carries over.
  4. The third outcome (direction replicates, bar missed) is stated as such.

This is a SEPARATE confirm-side script (doc 61 SS21.3's closing rule): the
stage aggregator's anchor-replication gate compares to results_exp69.json and
must not be patched. Rule 3: writes ONLY results_exp73_confirm.json.
"""
import json
import os
import statistics as st

CKPT = "exp73_confirm_ckpt"
OUT = "results_exp73_confirm.json"
N_SEEDS = 5
BIND_BAR = 0.65
FLOOR_MULTIPLE = 2.0
DELTA_BAR = 0.06
FROZEN_T2_REF = 0.8340
CONFIRM_SHA = ("59e6ada8a98b8917a1bd8e927d09249132a9765f7002e40fceb807a4bd51"
               "cd7c")
CELLS = ["A3-T1-uniform", "A3-T1-identity", "A3-T2-uniform", "A3-T2-identity"]
ANCHOR = "A0-W-T2"


def cell_stats(runs, arm):
    rows = [runs[(arm, s)] for s in range(N_SEEDS) if (arm, s) in runs]
    if len(rows) < N_SEEDS:
        return None
    acc = [r["heldout_test_acc"] for r in rows]
    return {"mean": st.mean(acc), "sd": st.stdev(acc), "per_seed": acc}


def main():
    runs = {}
    for fn in sorted(os.listdir(CKPT)):
        if not fn.endswith(".json"):
            continue
        r = json.load(open(os.path.join(CKPT, fn), encoding="utf-8"))
        assert r["dataset_sha256"] == CONFIRM_SHA, \
            "%s was not trained on the confirm draw" % fn
        runs[(r["arm"], int(r["seed"]))] = r

    out = {"experiment": "exp73-confirm", "doc": "61 SS21.3-SS21.4.1",
           "dataset_sha256": CONFIRM_SHA, "n_runs_read": len(runs),
           "cells": {}, "verdicts": {}, "reasons": {}}

    anchor = cell_stats(runs, ANCHOR)
    assert anchor, "anchor %s incomplete -- the SS21.4 bar-2 reference is missing" % ANCHOR
    out["fresh_draw_anchor"] = anchor

    complete = True
    for c in CELLS:
        ab = cell_stats(runs, c)
        fl = cell_stats(runs, "S-shuffle-%s" % c)
        if not ab or not fl:
            out["cells"][c] = {"complete": False,
                               "note": "INCOMPLETE -- no verdict (doc 61 SS15.4)"}
            complete = False
            continue
        bar = fl["mean"] + FLOOR_MULTIPLE * fl["sd"]
        contaminated = bool(fl["mean"] >= BIND_BAR
                            or (ab["mean"] - fl["mean"]) < DELTA_BAR)
        out["cells"][c] = {
            "complete": True, "mean": ab["mean"], "sd": ab["sd"],
            "per_seed": ab["per_seed"], "floor_mean": fl["mean"],
            "floor_sd": fl["sd"], "floor_bar": bar,
            "clears_floor": bool(ab["mean"] >= bar),
            "floor_contaminated_doc61_SS15_5": contaminated,
            "binds": bool(ab["mean"] >= BIND_BAR and ab["mean"] >= bar),
        }

    v = out["verdicts"]
    r = out["reasons"]
    if complete:
        cu, ci = out["cells"]["A3-T2-uniform"], out["cells"]["A3-T2-identity"]
        # -- bar 1: init_helps_A3_T2 ------------------------------------
        d_init = ci["mean"] - cu["mean"]
        v["init_helps_A3_T2_confirmed"] = bool(d_init >= DELTA_BAR)
        r["init_helps_A3_T2"] = {
            "identity_mean": ci["mean"], "uniform_mean": cu["mean"],
            "delta": d_init, "bar": DELTA_BAR,
            "stage1_delta": 0.0980,
            "direction_replicates": bool(d_init > 0),
            "third_outcome": bool(0 < d_init < DELTA_BAR),
        }
        # -- bar 2: amplitude_hurts_T2 ----------------------------------
        live = [c for c in ("A3-T2-uniform", "A3-T2-identity")
                if not out["cells"][c]["floor_contaminated_doc61_SS15_5"]]
        if not live:
            v["amplitude_hurts_T2_confirmed"] = None
            r["amplitude_hurts_T2"] = {
                "note": ("BLOCKED -- both T=2 cells floor-contaminated "
                         "(doc 61 SS15.5); no gain criterion is read")}
        else:
            best = max(live, key=lambda c: out["cells"][c]["mean"])
            bm = out["cells"][best]["mean"]
            d_primary = bm - anchor["mean"]
            d_secondary = bm - FROZEN_T2_REF
            v["amplitude_hurts_T2_confirmed"] = bool(d_primary <= -DELTA_BAR)
            r["amplitude_hurts_T2"] = {
                "best_cell": best, "best_mean": bm,
                "primary_reference_fresh_anchor_mean": anchor["mean"],
                "delta_primary": d_primary,
                "hurt_bar_primary": anchor["mean"] - DELTA_BAR,
                "secondary_reference_frozen": FROZEN_T2_REF,
                "delta_secondary": d_secondary,
                "secondary_would_say_hurts": bool(d_secondary <= -DELTA_BAR),
                "stage1_delta_vs_frozen": -0.1093,
                "direction_replicates": bool(d_primary < 0),
                "third_outcome": bool(-DELTA_BAR < d_primary < 0),
            }
    else:
        v["init_helps_A3_T2_confirmed"] = None
        v["amplitude_hurts_T2_confirmed"] = None
        r["incomplete"] = "one or more cells lack their full seed set or floor"

    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print("=== exp73 CONFIRM -- doc 61 SS21.4/.1 ===")
    print("fresh-draw anchor %s mean %.4f" % (ANCHOR, anchor["mean"]))
    for c in CELLS:
        cc = out["cells"][c]
        if not cc.get("complete"):
            print("%-18s INCOMPLETE" % c)
            continue
        print("%-18s mean %.4f  floor %.4f  binds=%s%s"
              % (c, cc["mean"], cc["floor_mean"], cc["binds"],
                 "  CONTAMINATED" if cc["floor_contaminated_doc61_SS15_5"]
                 else ""))
    for k in sorted(v):
        print("%s = %s" % (k, v[k]))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
