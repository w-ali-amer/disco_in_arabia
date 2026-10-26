# -*- coding: utf-8 -*-
"""exp83 aggregate -- SOLE WRITER of results_exp83.json.

NOT run during training. Run after all 75 units complete.

Registration: qnlp_private_docs/75_exp83_cost_shaping.md (REGISTERED 2026-09-16)

Computes the registered booleans (doc 75 §3):

  cost_shaping_helps    TRUE iff SOME arm in {A1, A2, A3} beats A0 with:
                          (i)  Holm-corrected one-sided p < 0.05 (across 3 comparisons)
                          (ii) mean paired diff >= 0.06
  MOVED_NOT_CLEARED     leg (i) fires but (ii) does not -- recorded, no confirm owed
  floor_admissible      FLOOR mean within 2 binomial SD of 0.5
                          (t-based bounds also reported)
  If floor_admissible is FALSE, the run is inadmissible -- no boolean is read.

Booleans are null (never false) if fewer than 75 checkpoints are available.

Usage:
  python exp83_aggregate.py
"""
import json
import math
import os
import platform
import sys
import time

from scipy import stats as sps

SPEC = "qnlp_private_docs/75_exp83_cost_shaping.md (REGISTERED 2026-09-16)"
EXP83_CKPT = "exp83_ckpt"
RESULTS_OUT = "results_exp83.json"

ARMS = ["A0_CE", "A1_SMOOTH", "A2_MARGIN", "A3_FOCAL", "FLOOR"]
COMPARE_ARMS = ["A1_SMOOTH", "A2_MARGIN", "A3_FOCAL"]   # compared to A0_CE
ANCHOR_ARM = "A0_CE"
FLOOR_ARM = "FLOOR"

SEED_BASE = 83000
N_SEEDS = 15
SEEDS = list(range(83000, 83015))
N_UNITS = len(ARMS) * N_SEEDS  # 75

BAR = 0.06          # standing magnitude bar (doc 75 §3)
ALPHA = 0.05        # one-sided


# ── load ─────────────────────────────────────────────────────────────────────
def load_checkpoint(arm, seed):
    path = os.path.join(EXP83_CKPT, "%s_s%d.json" % (arm, seed))
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


# ── statistics helpers ────────────────────────────────────────────────────────
def safe_mean(xs):
    return sum(xs) / len(xs) if xs else None


def safe_sd(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired_t_one_sided(diffs):
    """Paired one-sided t-test: H1: mean(diff) > 0.

    Uses scipy.stats.ttest_rel, alternative='greater'.
    Returns dict with all test statistics."""
    n = len(diffs)
    if n < 2:
        return None
    zeros = [0.0] * n
    result = sps.ttest_rel(diffs, zeros, alternative="greater")
    mean_d = sum(diffs) / n
    sd_d = safe_sd(diffs)
    se = sd_d / math.sqrt(n) if sd_d is not None else None
    return {
        "n": n,
        "df": n - 1,
        "mean_diff": mean_d,
        "sd_diff": sd_d,
        "se_diff": se,
        "t_stat": float(result.statistic),
        "p_value": float(result.pvalue),
        "one_sided_direction": "%s > A0_CE" % "arm",   # patched per-arm below
        "alpha": ALPHA,
        "test": "paired t-test (scipy.stats.ttest_rel, alternative='greater')",
    }


def holm_correction(p_values, arm_names, alpha=ALPHA):
    """Holm step-down correction across k comparisons.

    Returns:
      adjusted_p: dict arm -> adjusted p-value
      rejected:   dict arm -> bool (Holm-corrected rejection at alpha)

    Adjusted p-values: p_adj_k = max(p_adj_{k-1}, (m-k+1) * p_k) after sorting.
    Rejection is sequential: if rank k not rejected, all higher ranks are not rejected.
    """
    m = len(p_values)
    # Sort by raw p ascending
    order = sorted(range(m), key=lambda i: p_values[i])
    sorted_arms = [arm_names[i] for i in order]
    sorted_ps = [p_values[i] for i in order]

    # Compute adjusted p-values (Holm step-down)
    adj = [0.0] * m
    for k in range(m):
        raw_adj = (m - k) * sorted_ps[k]
        adj[k] = max(adj[k - 1], raw_adj) if k > 0 else raw_adj

    # Map back to arms
    adjusted_p = {sorted_arms[k]: adj[k] for k in range(m)}
    rejected = {sorted_arms[k]: bool(adj[k] < alpha) for k in range(m)}

    return {
        "adjusted_p": adjusted_p,
        "rejected": rejected,
        "sorted_order": sorted_arms,
        "sorted_raw_p": {sorted_arms[k]: sorted_ps[k] for k in range(m)},
        "sorted_adj_p": {sorted_arms[k]: adj[k] for k in range(m)},
        "m_comparisons": m,
        "alpha": alpha,
        "method": "Holm step-down (stepwise Bonferroni)",
        "sequential_stop": True,
    }


def arm_summary(accs, arm_name):
    if not accs:
        return None
    return {
        "arm": arm_name,
        "n": len(accs),
        "mean": safe_mean(accs),
        "sd": safe_sd(accs),
        "min": min(accs),
        "max": max(accs),
        "accs": accs,
    }


def binomial_sd(n, p=0.5):
    return math.sqrt(p * (1.0 - p) / n)


def floor_admissibility(acc, n):
    """Doc 70 rule: floor within 2 binomial SD of 0.5 -> admissible."""
    sd = binomial_sd(n)
    dev = abs(acc - 0.5)
    return {
        "mean_accuracy": acc,
        "n_test_items": n,
        "sd_binomial": sd,
        "deviation_from_chance": dev,
        "within_2sd": bool(dev <= 2 * sd),
        "admissible": bool(dev <= 2 * sd),
    }


def t_prediction_bounds(mean_acc, sd_acc, n_seeds, n_test, alpha=ALPHA):
    """t-based prediction bounds for floor bars (doc 70 rule 7).

    Two-sided confidence interval on the mean accuracy across seeds.
    Separate from the 2-binomial-SD check."""
    if n_seeds < 2 or sd_acc is None:
        return None
    se_mean = sd_acc / math.sqrt(n_seeds)
    try:
        t_crit = float(sps.t.ppf(1 - alpha / 2, df=n_seeds - 1))
    except Exception:
        t_crit = 2.145   # df=14, two-sided 5%
    lo = mean_acc - t_crit * se_mean
    hi = mean_acc + t_crit * se_mean
    return {
        "mean": mean_acc,
        "sd_across_seeds": sd_acc,
        "se_mean": se_mean,
        "n_seeds": n_seeds,
        "n_test_items": n_test,
        "alpha_two_sided": alpha,
        "t_critical": t_crit,
        "ci_lo": lo,
        "ci_hi": hi,
        "contains_0p5": bool(lo <= 0.5 <= hi),
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # ── Load all checkpoints ─────────────────────────────────────────────────
    runs = {arm: {} for arm in ARMS}
    missing = []
    for arm in ARMS:
        for seed in SEEDS:
            r = load_checkpoint(arm, seed)
            if r is None:
                missing.append((arm, seed))
            else:
                runs[arm][seed] = r

    n_loaded = sum(len(v) for v in runs.values())
    n_complete_full = (len(missing) == 0)
    complete_seeds = {
        arm: [s for s in SEEDS if s in runs[arm]]
        for arm in ARMS
    }

    # Seeds complete for all arms (for pairing)
    seeds_all_complete = [s for s in SEEDS
                          if all(s in runs[arm] for arm in ARMS)]
    n_complete_pairs = len(seeds_all_complete)

    # ── Per-seed table ───────────────────────────────────────────────────────
    per_seed = []
    for i, seed in enumerate(SEEDS):
        row = {"seed_index": i, "seed": seed}
        for arm in ARMS:
            key = arm.lower()
            row[key + "_available"] = (seed in runs[arm])
            if seed in runs[arm]:
                r = runs[arm][seed]
                row[key + "_acc"] = r["heldout_test_acc"]
                row[key + "_train_acc"] = r.get("train_acc")
                row[key + "_val_acc"] = r.get("val_acc")
                row[key + "_selected_epoch"] = r.get("selected_epoch")
                if arm != FLOOR_ARM:
                    rd = r.get("init_scheme", {}).get("redraw_doc62_SS24_3", {})
                    row[key + "_redraw_attempts"] = rd.get("attempts_used")
                if arm == FLOOR_ARM:
                    row["floor_scramble_seed"] = (
                        r.get("floor_scramble_info", {}).get("scramble_seed"))
        # Diffs vs A0 for complete seeds
        if seed in runs[ANCHOR_ARM]:
            a0 = runs[ANCHOR_ARM][seed]["heldout_test_acc"]
            for arm in COMPARE_ARMS:
                if seed in runs[arm]:
                    row["diff_%s_minus_A0" % arm] = (
                        runs[arm][seed]["heldout_test_acc"] - a0)
        per_seed.append(row)

    # ── Arm-level summaries ──────────────────────────────────────────────────
    arm_accs = {}
    for arm in ARMS:
        arm_accs[arm] = [runs[arm][s]["heldout_test_acc"]
                         for s in SEEDS if s in runs[arm]]

    # ── Registered booleans (null if incomplete) ─────────────────────────────
    if not n_complete_full:
        cost_shaping_helps = None
        MOVED_NOT_CLEARED = None
        floor_admissible = None
        holm_result = None
        comparisons = None
        floor_admiss_detail = None
        floor_t_bounds = None
        reason_incomplete = (
            "INCOMPLETE: %d of %d units missing. "
            "Booleans are null, never false (exp73_aggregate discipline)."
            % (len(missing), N_UNITS))
        cost_shaping_helps_reason = reason_incomplete
        floor_admissible_reason = reason_incomplete
        MOVED_NOT_CLEARED_reason = reason_incomplete

    else:
        reason_incomplete = None

        # ── Paired diffs: A1/A2/A3 vs A0 ──────────────────────────────────
        comparisons = {}
        p_values = []
        arm_order = []
        for arm in COMPARE_ARMS:
            diffs = [runs[arm][s]["heldout_test_acc"] - runs[ANCHOR_ARM][s]["heldout_test_acc"]
                     for s in SEEDS]
            t_res = paired_t_one_sided(diffs)
            t_res["one_sided_direction"] = "%s > A0_CE" % arm
            comparisons[arm] = {
                "arm": arm,
                "diffs": diffs,
                "t_test": t_res,
                "mean_diff": t_res["mean_diff"],
                "sd_diff": t_res["sd_diff"],
                "se_diff": t_res["se_diff"],
                "raw_p": t_res["p_value"],
            }
            p_values.append(t_res["p_value"])
            arm_order.append(arm)

        # ── Holm correction ────────────────────────────────────────────────
        holm_result = holm_correction(p_values, arm_order)
        for arm in COMPARE_ARMS:
            comparisons[arm]["holm_adj_p"] = holm_result["adjusted_p"][arm]
            comparisons[arm]["holm_rejected"] = holm_result["rejected"][arm]

        # ── Evaluate booleans ──────────────────────────────────────────────
        # Leg (i): Holm-corrected p < 0.05 for some arm
        # Leg (ii): mean paired diff >= 0.06 for that arm
        fired_arms_i = [arm for arm in COMPARE_ARMS
                        if holm_result["rejected"][arm]]
        fired_arms_ii = [arm for arm in fired_arms_i
                         if comparisons[arm]["mean_diff"] >= BAR]

        cost_shaping_helps = bool(len(fired_arms_ii) > 0)
        cost_shaping_helps_reason = (
            "Arms with Holm-corrected p < 0.05: %r. "
            "Arms also clearing bar (mean_diff >= %.2f): %r. "
            "cost_shaping_helps = %s."
            % (fired_arms_i, BAR, fired_arms_ii, cost_shaping_helps))

        # MOVED_NOT_CLEARED: (i) fires but (ii) does not
        moved_not_cleared_arms = [arm for arm in fired_arms_i
                                  if arm not in fired_arms_ii]
        MOVED_NOT_CLEARED = bool(len(moved_not_cleared_arms) > 0)
        MOVED_NOT_CLEARED_reason = (
            "Stat leg (i) fired for: %r. "
            "Magnitude leg (ii) not cleared for: %r. "
            "MOVED_NOT_CLEARED = %s."
            % (fired_arms_i, moved_not_cleared_arms, MOVED_NOT_CLEARED))

        # ── FLOOR admissibility ────────────────────────────────────────────
        floor_accs = arm_accs[FLOOR_ARM]
        floor_mean = safe_mean(floor_accs)
        floor_sd = safe_sd(floor_accs)

        # n_test from first available FLOOR checkpoint
        n_test_floor = runs[FLOOR_ARM][SEEDS[0]]["n_test"]

        floor_admiss_detail = floor_admissibility(floor_mean, n_test_floor)
        floor_admissible = floor_admiss_detail["admissible"]

        floor_t_bounds = t_prediction_bounds(
            floor_mean, floor_sd, len(floor_accs), n_test_floor)

        floor_admissible_reason = (
            "FLOOR mean_acc = %.4f, n_test = %d, "
            "2 binomial SD of 0.5 = %.4f, deviation = %.4f, "
            "within 2 SD = %s. "
            "floor_admissible = %s."
            % (floor_mean, n_test_floor,
               2 * binomial_sd(n_test_floor),
               abs(floor_mean - 0.5),
               floor_admiss_detail["within_2sd"],
               floor_admissible))

        if not floor_admissible:
            floor_admissible_reason += (
                " WARNING: floor inadmissible -- no registered boolean may be read.")

    # ── Redraw provenance summary ─────────────────────────────────────────────
    # Only for arms that use identity init (all non-FLOOR arms, but all 5 use identity)
    redraw_summary = {}
    for arm in ARMS:
        attempts_list = []
        for s in SEEDS:
            if s not in runs[arm]:
                continue
            rd = runs[arm][s].get("init_scheme", {}).get("redraw_doc62_SS24_3", {})
            attempts_list.append(rd.get("attempts_used"))
        redraw_summary[arm] = {
            "n_runs": len(attempts_list),
            "with_zero_redraws": sum(1 for a in attempts_list if a == 1),
            "with_one_or_more_redraws": sum(1 for a in attempts_list
                                            if a is not None and a > 1),
            "unknown_provenance": sum(1 for a in attempts_list if a is None),
        }

    # ── Obligations ───────────────────────────────────────────────────────────
    obligations = {
        "cost_shaping_helps_fires": (
            "If TRUE: standard confirm draw at seed_base 84000 (doc 75 §3), "
            "reruns the FIRED arm AND A0_CE AND FLOOR. Verdict on confirm alone."),
        "cost_shaping_helps_false": (
            "cost_shaping_helps = False completes the E5 slate. The wall survives "
            "six axes: content, structure, budget, channels, width, loss geometry."),
        "MOVED_NOT_CLEARED_recorded": (
            "Recorded as outcome; no confirm owed, no rerun licensed (doc 75 §3)."),
        "near_miss_band": (
            "±0.01 around either leg triggers symmetric bounded redraw of "
            "doc 62 §24.3 (attempt k at seed+100000·k, attempt-0 bit-identical, "
            "20-STOP)."),
    }

    out = {
        "experiment": "exp83 -- E5 cost shaping",
        "spec": SPEC,
        "written_by": "exp83_aggregate.py (sole writer of this file)",
        "platform": platform.platform(),
        "python": platform.python_version(),

        "n_seeds_registered": N_SEEDS,
        "seed_base": SEED_BASE,
        "seeds": SEEDS,
        "n_units_registered": N_UNITS,
        "n_loaded": n_loaded,
        "n_missing": len(missing),
        "missing_runs": missing,
        "n_complete_seeds_all_arms": n_complete_pairs,

        # ── Floor gate first -- inadmissibility invalidates everything ────────
        "floor_admissible": floor_admissible,
        "floor_admissible_reason": floor_admissible_reason if n_complete_full else reason_incomplete,
        "floor_admissibility_detail": floor_admiss_detail,
        "floor_t_prediction_bounds": floor_t_bounds,

        # ── Primary booleans ──────────────────────────────────────────────────
        "cost_shaping_helps": cost_shaping_helps,
        "cost_shaping_helps_reason": cost_shaping_helps_reason,
        "MOVED_NOT_CLEARED": MOVED_NOT_CLEARED,
        "MOVED_NOT_CLEARED_reason": MOVED_NOT_CLEARED_reason,

        # ── Holm correction ───────────────────────────────────────────────────
        "holm_correction": holm_result,
        "comparisons": comparisons,

        # ── Arm summaries ─────────────────────────────────────────────────────
        "arms": {arm: arm_summary(arm_accs[arm], arm) for arm in ARMS},

        # ── Per-seed table ────────────────────────────────────────────────────
        "per_seed": per_seed,

        # ── Redraw provenance (doc 62 §24.3) ──────────────────────────────────
        "redraw_provenance_doc62_SS24_3": redraw_summary,

        # ── Bar provenance (doc 75 §3) ─────────────────────────────────────────
        "bar_provenance": {
            "bar": BAR,
            "reason": ("Standing +0.06 bar retained. Derived from exp82's paired "
                       "sd_diff=0.0921, n=15, one-sided alpha=0.05; 80%-power MDE "
                       "approx 0.063. Bar matches the threshold the identity lever "
                       "itself was held to."),
            "sd_diff_from_exp82": 0.0921,
            "n_seeds": N_SEEDS,
            "alpha": ALPHA,
        },

        # ── Obligations ──────────────────────────────────────────────────────
        "obligations": obligations,

        "wall_time_sec": time.time() - t0,
    }

    json.dump(out, open(RESULTS_OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n=== exp83 aggregate ===")
    print("Loaded: %d/%d units  missing: %d" % (n_loaded, N_UNITS, len(missing)))
    if n_complete_full:
        print("\nFLOOR admissible = %s" % floor_admissible)
        if not floor_admissible:
            print("  WARNING: floor inadmissible -- run is inadmissible.")
        print("\nHolm-corrected comparisons (A1/A2/A3 vs A0_CE):")
        for arm in COMPARE_ARMS:
            c = comparisons[arm]
            print("  %-12s  mean_diff=%.4f  raw_p=%.4f  adj_p=%.4f  rej=%s"
                  % (arm, c["mean_diff"], c["raw_p"], c["holm_adj_p"],
                     c["holm_rejected"]))
        print("\ncost_shaping_helps = %s" % cost_shaping_helps)
        print("MOVED_NOT_CLEARED   = %s" % MOVED_NOT_CLEARED)
    else:
        print("Incomplete: booleans null. Missing: %r" % missing[:5],
              ("..." if len(missing) > 5 else ""))
    print("\nWrote: %s" % RESULTS_OUT)


if __name__ == "__main__":
    main()
