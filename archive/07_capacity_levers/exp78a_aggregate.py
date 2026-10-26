# -*- coding: utf-8 -*-
"""exp78a_aggregate.py -- SOLE WRITER of results_exp78a.json.

Registration: qnlp_private_docs/68_exp78a_band32_replication.md §3.

DO NOT run this file until all 10 training runs (band32_n2_T2 x5 seeds +
S-shuffle-band32_n2_T2 x5 seeds) and the comparator are complete.

Implements doc 68 §3's conjunctive bars exactly:

  band32_replicates = TRUE iff BOTH:
    (a) fresh-draw quantum mean >= floor_mean + 2 * floor_sd  (ddof=1)
    (b) every comparator tier mean lies in 0.5 +/- 2*sqrt(0.25/N_test)

  Third outcomes (named, never collapsed to FALSE):
    (a)-without-(b): "the cell learns but so do comparators --
                      the SS10.7 violation was draw noise"
    (b)-without-(a): "comparators stay blind but the cell's
                      clearance was draw noise"

N_test is asserted from the run records (must equal 300), not hardcoded.

Reads:
  exp78a_ckpt/band32_n2_T2_s{0..4}.json
  exp78a_ckpt/S-shuffle-band32_n2_T2_s{0..4}.json
  exp78a_ckpt/comparator.json

Writes:
  results_exp78a.json

THIS FILE IS NOT RUN BY THE BUILD AGENT.  It is written here for the user
to run after training completes.  Results file does NOT exist before then.
"""
import glob
import json
import math
import os
import platform
import sys
import time

CKPT_DIR     = "exp78a_ckpt"
COMP_OUT     = os.path.join(CKPT_DIR, "comparator.json")
RESULTS_OUT  = "results_exp78a.json"
QUANTUM_ARM  = "band32_n2_T2"
FLOOR_ARM    = "S-shuffle-band32_n2_T2"
N_SEEDS_EXP  = 5
TIERS        = ("i_raw", "ii_committed", "iii_distractor")
SPEC         = "qnlp_private_docs/68_exp78a_band32_replication.md"


def load_ckpts(arm):
    """Load all per-seed checkpoints for `arm`, sorted by seed."""
    pattern = os.path.join(CKPT_DIR, "%s_s*.json" % arm)
    paths = sorted(glob.glob(pattern))
    assert len(paths) == N_SEEDS_EXP, (
        "expected %d checkpoints for %s, found %d: %r"
        % (N_SEEDS_EXP, arm, len(paths), paths))
    return [json.load(open(p, encoding="utf-8")) for p in paths]


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs, ddof=1):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - ddof))


def main():
    t0 = time.time()

    # --------------------------------------------------------- load training runs
    q_runs = load_ckpts(QUANTUM_ARM)
    f_runs = load_ckpts(FLOOR_ARM)

    q_accs = [r["heldout_test_acc"] for r in q_runs]
    f_accs = [r["heldout_test_acc"] for r in f_runs]

    # Assert N_test from records (not hardcoded)
    N_test_vals = set(r["n_test"] for r in q_runs + f_runs)
    assert len(N_test_vals) == 1, "n_test is not uniform across runs: %r" % N_test_vals
    N_test = N_test_vals.pop()
    assert N_test == 300, (
        "doc 68 §3 registered bars assume N_test=300; records show %d" % N_test)

    q_mean = mean(q_accs)
    f_mean = mean(f_accs)
    f_sd   = sd(f_accs, ddof=1)

    bar_a = q_mean >= f_mean + 2.0 * f_sd

    # ---------------------------------------------------- load comparator report
    comp = json.load(open(COMP_OUT, encoding="utf-8"))
    chance_half = 2.0 * math.sqrt(0.25 / N_test)   # 2*sqrt(0.25/300)

    tier_results = {}
    bar_b_parts  = {}
    for tier in TIERS:
        tier_mean = comp["arms"][QUANTUM_ARM][tier]["mean"]
        inside    = abs(tier_mean - 0.5) <= chance_half
        tier_results[tier] = {"mean": tier_mean, "inside_chance_band": inside}
        bar_b_parts[tier]  = inside

    bar_b = all(bar_b_parts.values())

    # ----------------------------------------------------- verdict (doc 68 §3)
    if bar_a and bar_b:
        band32_replicates     = True
        verdict               = "TRUE"
        verdict_reason        = ("both bars cleared: quantum clears floor by "
                                 ">= 2*floor_sd AND all three comparator tiers "
                                 "sit at chance")
    elif bar_a and not bar_b:
        band32_replicates     = None  # named third outcome, never FALSE
        verdict               = "(a)-without-(b)"
        verdict_reason        = ("the cell learns but so do comparators -- "
                                 "the SS10.7 violation was draw noise")
    elif bar_b and not bar_a:
        band32_replicates     = None  # named third outcome, never FALSE
        verdict               = "(b)-without-(a)"
        verdict_reason        = ("comparators stay blind but the cell's "
                                 "clearance was draw noise")
    else:
        band32_replicates     = False
        verdict               = "FALSE"
        verdict_reason        = "neither bar cleared: no replication signal"

    rep = {
        "experiment":   "exp78a -- band32 replication (doc 68)",
        "spec":         SPEC,
        "platform":     platform.platform(),
        "python":       platform.python_version(),
        "N_test":       N_test,
        "chance_half":  chance_half,
        "quantum_arm":  {
            "arm":          QUANTUM_ARM,
            "per_seed":     q_accs,
            "mean":         q_mean,
            "sd_ddof1":     sd(q_accs, ddof=1),
        },
        "floor_arm":    {
            "arm":          FLOOR_ARM,
            "per_seed":     f_accs,
            "mean":         f_mean,
            "sd_ddof1":     f_sd,
        },
        "bar_a": {
            "criterion":    "quantum_mean >= floor_mean + 2*floor_sd",
            "lhs":          q_mean,
            "rhs":          f_mean + 2.0 * f_sd,
            "passed":       bar_a,
        },
        "bar_b": {
            "criterion":    "all tier means in 0.5 +/- 2*sqrt(0.25/N_test)",
            "chance_half":  chance_half,
            "tiers":        tier_results,
            "passed":       bar_b,
        },
        "band32_replicates": band32_replicates,
        "verdict":           verdict,
        "verdict_reason":    verdict_reason,
        "floor_results": {
            "floor_comparator": {
                tier: comp["floors"][FLOOR_ARM][tier]
                for tier in TIERS
            }
        },
        "wall_time_sec": time.time() - t0,
    }

    json.dump(rep, open(RESULTS_OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[exp78a aggregate] wrote %s" % RESULTS_OUT)
    print("  bar (a): quantum %.4f >= floor %.4f + 2*%.4f = %.4f  -> %s"
          % (q_mean, f_mean, f_sd, f_mean + 2.0 * f_sd, bar_a))
    print("  bar (b): chance_half=%.4f" % chance_half)
    for tier in TIERS:
        tr = tier_results[tier]
        print("    %-18s mean=%.4f  inside=%s" % (tier, tr["mean"], tr["inside_chance_band"]))
    print("  VERDICT: %s" % verdict)
    print("  %s" % verdict_reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
