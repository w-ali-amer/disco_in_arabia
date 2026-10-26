# -*- coding: utf-8 -*-
"""exp78b_aggregate.py -- sole writer of results_exp78b.json.

Registration: qnlp_private_docs/69_exp78b_budget_matched_scaling.md §4.

Reads all exp78b checkpoint JSONs from exp78b_ckpt/, aggregates per-arm
heldout_test_acc over 5 seeds, applies doc 69 §4's bars, and writes
results_exp78b.json.  This script is NOT RUN during training; it is run
only after all 50 checkpoint JSONs are present.

Doc 69 §4 bars (pinned pre-data):
  1. collapse_is_budget_artifact = TRUE iff at least the n=2 cell binds
     under the scaled budget: mean >= 0.65 AND mean >= floor_mean + 2 * floor_sd.
  2. scaling_converts = TRUE iff cells bind at n=2 AND the bound-cell
     set extends to any n>=3 (each cell judged by the same rule).
  3. Third outcomes named:
     a. "n2_only_binds": n=2 binds but no n>=3 cell binds -- budget was
        the n=2 constraint; depth/width walls remain at n>=3.
     b. "nothing_binds": no cell binds -- the wall is not a budget artifact;
        doc 63 §24.2 sharpens to architectural.
  4. Any fired positive owes a confirm draw at generator's seed+1000
     convention; reference cells named at that point.

Usage (after all 50 checkpoints are written):
  python exp78b_aggregate.py [--ckpt-dir exp78b_ckpt] [--out results_exp78b.json]
"""

import argparse
import json
import os
import statistics
import sys

# ============================================================= import arm config
# Import at runtime to get the live ARM_CFG and SCALING_TABLE.
import exp78b_run as W
import exp76_run as R


# ============================================================ bar constants
BINDS_BAR       = R.BINDS_BAR       # 0.65 -- inherited from doc 54 SS7 via exp76
FLOOR_MULTIPLE  = R.FLOOR_MULTIPLE  # 2.0  -- SS12 primary rule


# ==================================================== read and aggregate helpers
def load_checkpoints(ckpt_dir, arms, n_seeds=5):
    """Read all checkpoint JSONs; return {arm: [heldout_test_acc, ...] x 5}."""
    results = {}
    missing = []
    for arm in arms:
        accs = []
        for s in range(n_seeds):
            path = os.path.join(ckpt_dir, "%s_s%d.json" % (arm, s))
            if not os.path.exists(path):
                missing.append(path)
                continue
            r = json.load(open(path, encoding="utf-8"))
            accs.append(r["heldout_test_acc"])
        results[arm] = accs
    if missing:
        raise RuntimeError(
            "Missing %d checkpoint(s):\n  %s\n"
            "Wait for all 50 runs to complete before aggregating."
            % (len(missing), "\n  ".join(missing)))
    return results


def arm_stats(accs):
    """Mean and SD (ddof=1) over seeds."""
    mean = statistics.mean(accs)
    sd   = statistics.pstdev(accs) if len(accs) < 2 else statistics.stdev(accs)
    return mean, sd


def cell_binds(mean, floor_mean, floor_sd):
    """Doc 69 §4 bar 1 conjunctive rule."""
    return (mean >= BINDS_BAR) and (mean >= floor_mean + FLOOR_MULTIPLE * floor_sd)


# ======================================================== main aggregation logic
def aggregate(ckpt_dir, out_path, n_seeds=5):
    arms       = W.SCALING_CELLS
    floor_arms = ["S-shuffle-" + c for c in arms]

    print("[agg78b] loading checkpoints from %s ..." % ckpt_dir, flush=True)
    trained_accs = load_checkpoints(ckpt_dir, arms,       n_seeds)
    floor_accs   = load_checkpoints(ckpt_dir, floor_arms, n_seeds)

    # Per-cell statistics
    cell_rows = {}
    for cell in arms:
        fa               = "S-shuffle-" + cell
        n                = W.CELL_N[cell]
        t_accs           = trained_accs[cell]
        f_accs           = floor_accs[fa]
        t_mean, t_sd     = arm_stats(t_accs)
        f_mean, f_sd     = arm_stats(f_accs)
        binds            = cell_binds(t_mean, f_mean, f_sd)
        cell_rows[cell]  = {
            "n": n, "T": W.COMPUTE_TABLE["rows"][cell]["exp78b_est_per_run_s"],
            "n_params": 60 * n,
            "heldout_per_seed":      t_accs,
            "mean":                  t_mean,
            "sd":                    t_sd,
            "floor_arm":             fa,
            "floor_per_seed":        f_accs,
            "floor_mean":            f_mean,
            "floor_sd":              f_sd,
            "floor_bar":             f_mean + FLOOR_MULTIPLE * f_sd,
            "binds_bar":             BINDS_BAR,
            "floor_multiple":        FLOOR_MULTIPLE,
            "binds": binds,
            "binds_detail": {
                "mean_ge_065":       bool(t_mean >= BINDS_BAR),
                "mean_ge_floor_2sd": bool(t_mean >= f_mean + FLOOR_MULTIPLE * f_sd),
            },
        }

    # Doc 69 §4 bars
    n2_binds       = cell_rows["n2_T2"]["binds"]
    any_n3p_binds  = any(cell_rows[c]["binds"] for c in arms if W.CELL_N[c] >= 3)
    bound_cells    = [c for c in arms if cell_rows[c]["binds"]]

    collapse_is_budget_artifact = n2_binds
    scaling_converts            = n2_binds and any_n3p_binds

    # Third outcomes
    if collapse_is_budget_artifact and scaling_converts:
        verdict = "scaling_converts"
        verdict_desc = (
            "n=2 binds AND at least one n>=3 cell binds. Budget was the "
            "constraint at multiple system sizes; the wall moves with the "
            "per-wire budget.")
    elif collapse_is_budget_artifact and not scaling_converts:
        verdict = "n2_only_binds"
        verdict_desc = (
            "n=2 binds but no n>=3 cell binds. Budget was the n=2 constraint; "
            "depth/width walls remain at n>=3. Scaling the budget is necessary "
            "but not sufficient for the n>=3 collapse.")
    else:
        verdict = "nothing_binds"
        verdict_desc = (
            "No cell binds under the scaled budget. The collapse is not a "
            "budget artifact; doc 63 §24.2 sharpens to architectural.")

    # Confirm draw obligation
    confirm_obligation = None
    if collapse_is_budget_artifact or scaling_converts:
        confirm_cells = [c for c in bound_cells]
        confirm_obligation = {
            "required": True,
            "rule": "doc 69 §4 item 4 + doc 63 SS17.2-style rule",
            "convention": "generator seed + 1000 for each binding cell",
            "reference_cells": confirm_cells,
            "note": (
                "Any positive that survives the twice-evaluation (doc 69 §5) "
                "owes a confirm draw. The confirm-draw cells are named here "
                "at the time of the primary verdict. Run exp78b_datagen_confirm.py "
                "(to be registered) to generate the confirm draws."),
        }
    else:
        confirm_obligation = {"required": False, "reason": "no positives to confirm"}

    # Assemble results
    rep = {
        "experiment": "exp78b -- budget-matched scaling (doc 69)",
        "spec": "doc 69, registered 2026-09-03",
        "bars": {
            "collapse_is_budget_artifact": collapse_is_budget_artifact,
            "scaling_converts":            scaling_converts,
            "verdict":                     verdict,
            "verdict_desc":                verdict_desc,
            "bound_cells":                 bound_cells,
        },
        "confirm_obligation": confirm_obligation,
        "twice_evaluation_doc69_S5": (
            "The verdict table is subject to the standing twice-evaluation rule "
            "(doc 69 §5) before its addendum commits. Any positive here must "
            "survive independent re-read before the confirm draw is registered."),
        "cell_stats": cell_rows,
        "n_seeds":   n_seeds,
        "n_arms":    len(arms),
        "n_runs":    len(arms) * n_seeds + len(floor_arms) * n_seeds,
        "scaling_table": W.SCALING_TABLE,
        "compute_table": W.COMPUTE_TABLE,
        "t_inclusion_decision": W.COMPUTE_TABLE["t_inclusion_decision"],
        "ckpt_dir": ckpt_dir,
        "bar_constants": {
            "BINDS_BAR": BINDS_BAR, "FLOOR_MULTIPLE": FLOOR_MULTIPLE,
            "source": "inherited from exp76 (doc 54 SS7, doc 63 SS12); frozen"},
    }

    json.dump(rep, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[agg78b] wrote %s" % out_path, flush=True)

    # Print summary
    print("\n=== exp78b results summary ===", flush=True)
    print("Bars (doc 69 §4):", flush=True)
    print("  collapse_is_budget_artifact = %s" % collapse_is_budget_artifact, flush=True)
    print("  scaling_converts            = %s" % scaling_converts, flush=True)
    print("  verdict: %s" % verdict, flush=True)
    print("\nPer-cell:", flush=True)
    for cell in arms:
        row = cell_rows[cell]
        print("  %-10s n=%d T=%d  mean=%.4f floor=%.4f+2*%.4f=%.4f  BINDS=%s"
              % (cell, row["n"], W.CELL_N[cell],
                 row["mean"], row["floor_mean"], row["floor_sd"],
                 row["floor_bar"], row["binds"]), flush=True)
    print("", flush=True)
    return rep


# ========================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", default="exp78b_ckpt",
                    help="directory containing exp78b checkpoint JSONs")
    ap.add_argument("--out", default="results_exp78b.json",
                    help="output results file (sole writer)")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    if os.path.exists(a.out):
        print("[agg78b] WARNING: %s already exists; will overwrite." % a.out,
              flush=True)

    rep = aggregate(a.ckpt_dir, a.out, n_seeds=a.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
