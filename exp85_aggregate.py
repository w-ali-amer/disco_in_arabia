# -*- coding: utf-8 -*-
"""exp85 aggregate -- SOLE WRITER of results_exp85.json.

NOT run during training. Run after all 45 units complete.

Registration: qnlp_private_docs/77_exp85_identity_T3.md (FROZEN 2026-09-17)

Computes the registered booleans (doc 77 §2):

  identity_direction_real_T3     TRUE iff identity arm beats uniform arm:
                                   paired one-sided t p < 0.05
  identity_effect_clears_bar_T3  TRUE iff mean paired diff >= 0.06
  LEVER_WITHOUT_TRACTION         direction real (leg i fires) AND both arms
                                   within ±0.0577 of 0.5 (chance band);
                                   distinct from "both drop but stay off floor"
  floor_admissible               FLOOR mean within 2 binomial SD of 0.5;
                                   if FALSE, run is inadmissible

  "both drop but stay off floor" reading: identity and uniform both above
  ±0.0577 chance band but direction still real -- reported descriptively.

Booleans are null (never false) if fewer than 45 checkpoints are available.

Usage:
  python exp85_aggregate.py
"""
import json
import math
import os
import platform
import sys
import time

from scipy import stats as sps

SPEC = "qnlp_private_docs/77_exp85_identity_T3.md (FROZEN 2026-09-17)"
EXP85_CKPT = "exp85_ckpt"
RESULTS_OUT = "results_exp85.json"

ARMS = ["A3-T3-identity", "A3-T3-uniform", "FLOOR"]
ID_ARM    = "A3-T3-identity"
UNIF_ARM  = "A3-T3-uniform"
FLOOR_ARM = "FLOOR"

SEED_BASE = 85000
N_SEEDS   = 15
SEEDS     = list(range(85000, 85015))
N_UNITS   = len(ARMS) * N_SEEDS   # 45

BAR       = 0.06    # standing magnitude bar (doc 77 §0; provenance from exp82 sd_diff=0.0921)
ALPHA     = 0.05    # one-sided

# Chance band for LEVER_WITHOUT_TRACTION: 2 binomial SD at n_test=300
# sqrt(0.25/300) * 2 ≈ 0.0577 (doc 77 §0)
N_TEST    = 300
CHANCE_BAND = 2 * math.sqrt(0.25 / N_TEST)   # ≈ 0.0577


# ── load ──────────────────────────────────────────────────────────────────────
def load_checkpoint(arm, seed):
    path = os.path.join(EXP85_CKPT, "%s_s%d.json" % (arm, seed))
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
    """Paired one-sided t-test: H1: mean(diff) > 0 (identity > uniform).

    Uses scipy.stats.ttest_rel, alternative='greater'."""
    n = len(diffs)
    if n < 2:
        return None
    result = sps.ttest_rel(diffs, [0.0] * n, alternative="greater")
    mean_d = sum(diffs) / n
    sd_d = safe_sd(diffs)
    se = sd_d / math.sqrt(n) if sd_d is not None else None
    return {
        "n": n, "df": n - 1,
        "mean_diff": mean_d,
        "sd_diff": sd_d,
        "se_diff": se,
        "t_stat": float(result.statistic),
        "p_value": float(result.pvalue),
        "one_sided_direction": "identity > uniform",
        "alpha": ALPHA,
        "test": "paired t-test (scipy.stats.ttest_rel, alternative='greater')",
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
    """t-based CI on mean accuracy across seeds (doc 70 rule 7)."""
    if n_seeds < 2 or sd_acc is None:
        return None
    se_mean = sd_acc / math.sqrt(n_seeds)
    try:
        t_crit = float(sps.t.ppf(1 - alpha / 2, df=n_seeds - 1))
    except Exception:
        t_crit = 2.145   # df=14 two-sided 5%
    lo = mean_acc - t_crit * se_mean
    hi = mean_acc + t_crit * se_mean
    return {
        "mean": mean_acc, "sd_across_seeds": sd_acc, "se_mean": se_mean,
        "n_seeds": n_seeds, "n_test_items": n_test,
        "alpha_two_sided": alpha, "t_critical": t_crit,
        "ci_lo": lo, "ci_hi": hi,
        "contains_0p5": bool(lo <= 0.5 <= hi),
    }


def within_chance_band(acc):
    return abs(acc - 0.5) <= CHANCE_BAND


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # ── Load checkpoints ──────────────────────────────────────────────────────
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
    seeds_all_complete = [s for s in SEEDS
                          if all(s in runs[arm] for arm in ARMS)]

    # ── Per-seed table ────────────────────────────────────────────────────────
    per_seed = []
    for i, seed in enumerate(SEEDS):
        row = {"seed_index": i, "seed": seed}
        for arm in ARMS:
            key = arm.replace("-", "_").lower()
            row[key + "_available"] = (seed in runs[arm])
            if seed in runs[arm]:
                r = runs[arm][seed]
                row[key + "_acc"] = r["heldout_test_acc"]
                row[key + "_train_acc"] = r.get("train_acc")
                row[key + "_val_acc"] = r.get("val_acc")
                row[key + "_selected_epoch"] = r.get("selected_epoch")
                ud = r.get("unitarity_drift", {})
                row[key + "_startup_drift"] = ud.get("startup_max_drift")
                if arm != FLOOR_ARM:
                    rd = r.get("init_scheme", {}).get("redraw_doc62_SS24_3", {})
                    row[key + "_redraw_attempts"] = rd.get("attempts_used")
        if seed in runs[ID_ARM] and seed in runs[UNIF_ARM]:
            diff = runs[ID_ARM][seed]["heldout_test_acc"] - runs[UNIF_ARM][seed]["heldout_test_acc"]
            row["diff_identity_minus_uniform"] = diff
        per_seed.append(row)

    # ── Arm-level summaries ───────────────────────────────────────────────────
    arm_accs = {arm: [runs[arm][s]["heldout_test_acc"]
                      for s in SEEDS if s in runs[arm]]
                for arm in ARMS}

    # ── Registered booleans ───────────────────────────────────────────────────
    # doc 77 §5 (2026-09-18, committed pre-verdict): units invalidated by the
    # registered depth-3 drift rule drop their SEED entirely; the paired
    # analysis runs on the complete-pair seeds (n=14 for the seed-85007
    # invalidation) and floor admissibility on the same seed set. Booleans are
    # null only if a unit is missing for any OTHER reason than the drift rule.
    DRIFT_INVALIDATED_SEEDS = {85007}
    unexplained_missing = [(a, s) for (a, s) in missing
                           if s not in DRIFT_INVALIDATED_SEEDS]
    analysis_seeds = [s for s in seeds_all_complete
                      if s not in DRIFT_INVALIDATED_SEEDS]

    if unexplained_missing:
        reason_incomplete = (
            "INCOMPLETE: %d of %d units missing for reasons OTHER than the "
            "registered drift invalidation. Booleans are null, never false."
            % (len(unexplained_missing), N_UNITS))
        identity_direction_real_T3 = None
        identity_effect_clears_bar_T3 = None
        LEVER_WITHOUT_TRACTION = None
        floor_admissible = None
        t_test_result = None
        floor_admiss_detail = None
        floor_t_bounds = None
        id_mean = unif_mean = None
        sd_diff = mean_diff = None
        diffs = None
        both_in_chance_band = None
        both_above_floor = None

    else:
        reason_incomplete = None

        # Paired diffs: identity - uniform across analysis seeds (doc 77 §5)
        diffs = [runs[ID_ARM][s]["heldout_test_acc"] - runs[UNIF_ARM][s]["heldout_test_acc"]
                 for s in analysis_seeds]
        t_test_result = paired_t_one_sided(diffs)
        mean_diff = t_test_result["mean_diff"]
        sd_diff   = t_test_result["sd_diff"]

        id_mean   = safe_mean([runs[ID_ARM][s]["heldout_test_acc"]
                               for s in analysis_seeds])
        unif_mean = safe_mean([runs[UNIF_ARM][s]["heldout_test_acc"]
                               for s in analysis_seeds])

        # Registered booleans (doc 77 §2, §0)
        identity_direction_real_T3    = bool(t_test_result["p_value"] < ALPHA)
        identity_effect_clears_bar_T3 = bool(mean_diff >= BAR)

        # LEVER_WITHOUT_TRACTION: direction real AND BOTH arms within chance band
        both_in_chance_band = (within_chance_band(id_mean)
                               and within_chance_band(unif_mean))
        LEVER_WITHOUT_TRACTION = bool(identity_direction_real_T3 and both_in_chance_band)

        # "both drop but stay off floor": BOTH > chance band, direction real
        # (the legible third reading: lever survives, wall holds, not at floor)
        both_above_floor = (not within_chance_band(id_mean)
                            and not within_chance_band(unif_mean))

        # FLOOR admissibility (analysis seeds only, doc 77 §5)
        floor_accs = [runs[FLOOR_ARM][s]["heldout_test_acc"]
                      for s in analysis_seeds]
        floor_mean = safe_mean(floor_accs)
        floor_sd   = safe_sd(floor_accs)
        n_test_floor = runs[FLOOR_ARM][analysis_seeds[0]]["n_test"]

        floor_admiss_detail = floor_admissibility(floor_mean, n_test_floor)
        floor_admissible = floor_admiss_detail["admissible"]

        floor_t_bounds = t_prediction_bounds(
            floor_mean, floor_sd, len(floor_accs), n_test_floor)

    # ── Redraw provenance ─────────────────────────────────────────────────────
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

    # ── Descriptive readings ──────────────────────────────────────────────────
    if not unexplained_missing:
        lever_status = (
            "LEVER_WITHOUT_TRACTION" if LEVER_WITHOUT_TRACTION
            else ("both_drop_but_stay_off_floor"
                  if (identity_direction_real_T3 and both_above_floor
                      and not identity_effect_clears_bar_T3)
                  else ("lever_and_bar_cleared"
                        if (identity_direction_real_T3 and identity_effect_clears_bar_T3)
                        else "direction_not_real")))
        descriptive = {
            "identity_mean": id_mean,
            "uniform_mean": unif_mean,
            "mean_diff_identity_minus_uniform": mean_diff,
            "realized_sd_diff": sd_diff,
            "chance_band_radius": CHANCE_BAND,
            "identity_within_chance_band": both_in_chance_band and within_chance_band(id_mean) if not unexplained_missing else None,
            "uniform_within_chance_band": both_in_chance_band and within_chance_band(unif_mean) if not unexplained_missing else None,
            "both_arms_within_chance_band": bool(both_in_chance_band) if not unexplained_missing else None,
            "both_arms_above_chance_band": bool(both_above_floor) if not unexplained_missing else None,
            "lever_reading": lever_status,
            "note": (
                "LEVER_WITHOUT_TRACTION: direction real AND both arms within "
                "±0.0577 of 0.5 (2 binomial SD at n=300, doc 77 §0). "
                "'both drop but stay off floor': both arms above chance band, "
                "direction real but bar not cleared -- legible third reading, "
                "distinct from LEVER_WITHOUT_TRACTION."),
        }
    else:
        descriptive = {"note": reason_incomplete}

    out = {
        "experiment": "exp85 -- identity lever at T=3",
        "spec": SPEC,
        "written_by": "exp85_aggregate.py (sole writer of this file)",
        "platform": platform.platform(),
        "python": platform.python_version(),

        "n_seeds_registered": N_SEEDS,
        "seed_base": SEED_BASE,
        "seeds": SEEDS,
        "n_units_registered": N_UNITS,
        "n_loaded": n_loaded,
        "n_missing": len(missing),
        "missing_runs": missing,
        "n_complete_seeds_all_arms": len(seeds_all_complete),

        # ── Floor gate first ──────────────────────────────────────────────────
        "floor_admissible": floor_admissible if not unexplained_missing else None,
        "floor_admissible_reason": (
            ("FLOOR mean=%.4f n=%d 2-binom-sd=%.4f dev=%.4f admissible=%s"
             % (floor_admiss_detail["mean_accuracy"],
                floor_admiss_detail["n_test_items"],
                2 * floor_admiss_detail["sd_binomial"],
                floor_admiss_detail["deviation_from_chance"],
                floor_admissible))
            if not unexplained_missing else reason_incomplete),
        "floor_admissibility_detail": floor_admiss_detail if not unexplained_missing else None,
        "floor_t_prediction_bounds": floor_t_bounds if not unexplained_missing else None,

        # ── Primary booleans ──────────────────────────────────────────────────
        "identity_direction_real_T3": identity_direction_real_T3 if not unexplained_missing else None,
        "identity_effect_clears_bar_T3": identity_effect_clears_bar_T3 if not unexplained_missing else None,
        "LEVER_WITHOUT_TRACTION": LEVER_WITHOUT_TRACTION if not unexplained_missing else None,
        "incomplete_note": reason_incomplete,

        # ── Paired t-test ─────────────────────────────────────────────────────
        "paired_t_test_identity_vs_uniform": t_test_result if not unexplained_missing else None,
        "paired_diffs": diffs if not unexplained_missing else None,

        # ── Descriptive readings ──────────────────────────────────────────────
        "descriptive": descriptive,

        # ── Arm summaries ─────────────────────────────────────────────────────
        "arms": {arm: arm_summary(arm_accs[arm], arm) for arm in ARMS},

        # ── Per-seed table ────────────────────────────────────────────────────
        "per_seed": per_seed,

        # ── Redraw provenance ─────────────────────────────────────────────────
        "redraw_provenance_doc62_SS24_3": redraw_summary,

        # ── Bar provenance (doc 77 §0) ─────────────────────────────────────────
        "bar_provenance": {
            "bar": BAR,
            "reason": (
                "Bar = +0.06; provenance from exp82's direct identity-vs-uniform "
                "paired sd_diff = 0.0921 (n=15, 80%-power MDE ≈ 0.063) -- "
                "the closest analogue pairing (doc 77 §0). "
                "If T=3 pairing is noisier than T=2's, the run is honestly "
                "underpowered at 0.06 and realized sd_diff is reported alongside."),
            "sd_diff_exp82_analogue": 0.0921,
            "chance_band_radius": CHANCE_BAND,
            "chance_band_formula": "2 * sqrt(0.25 / n_test) at n_test=%d" % N_TEST,
        },

        # ── Obligations ────────────────────────────────────────────────────────
        "obligations": {
            "confirm_base": 86000,
            "confirm_fires_if": (
                "identity_direction_real_T3 AND identity_effect_clears_bar_T3; "
                "fires identity arm + uniform arm + FLOOR at seed_base 86000."),
            "lever_without_traction_fires_if": (
                "direction real AND both arms within ±0.0577 of 0.5; "
                "no confirm owed (the wall eats the lever)."),
            "both_drop_but_stay_off_floor": "Descriptive; no confirm owed.",
        },

        "wall_time_sec": time.time() - t0,
    }

    json.dump(out, open(RESULTS_OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # Console summary
    print("\n=== exp85 aggregate ===")
    print("Loaded: %d/%d units  missing: %d" % (n_loaded, N_UNITS, len(missing)))
    if not unexplained_missing:
        print("\nFLOOR admissible = %s" % floor_admissible)
        if not floor_admissible:
            print("  WARNING: floor inadmissible -- run is inadmissible.")
        print("\nPaired t-test (identity - uniform, one-sided p<0.05):")
        print("  mean_diff=%.4f  sd_diff=%.4f  t=%.3f  p=%.4f"
              % (mean_diff, sd_diff, t_test_result["t_stat"], t_test_result["p_value"]))
        print("  identity_mean=%.4f  uniform_mean=%.4f  chance_band=±%.4f"
              % (id_mean, unif_mean, CHANCE_BAND))
        print("\nidentity_direction_real_T3    = %s" % identity_direction_real_T3)
        print("identity_effect_clears_bar_T3 = %s" % identity_effect_clears_bar_T3)
        print("LEVER_WITHOUT_TRACTION        = %s" % LEVER_WITHOUT_TRACTION)
        if not unexplained_missing:
            print("lever_reading: %s" % descriptive.get("lever_reading", "n/a"))
    else:
        print("Incomplete: booleans null. Missing: %r" % missing[:5],
              ("..." if len(missing) > 5 else ""))
    print("\nWrote: %s" % RESULTS_OUT)


if __name__ == "__main__":
    main()
