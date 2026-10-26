# -*- coding: utf-8 -*-
"""exp82 aggregate -- SOLE WRITER of results_exp82.json.

NOT run during training. Run after all 30 runs complete.

Computes the two registered booleans (doc 74 SS2), decoupled with reasons:
  identity_direction_real     -- paired seed t (identity - uniform), one-sided p < 0.05
                                 H1: mean diff > 0  (14 df at 15 seeds)
  identity_effect_clears_bar  -- mean difference >= 0.06 (standing magnitude bar)

The two booleans decouple existence from size. Both are null (never false)
if <30 checkpoints are available.

Original exp73 seeds 0-4 are quoted in a descriptive-only block; they
contribute zero data to any computation.

Usage:
  python exp82_aggregate.py
"""
import json
import math
import os
import platform
import sys
import time

from scipy import stats as sps

SPEC = "qnlp_private_docs/74_exp82_init_direction.md (REGISTERED 2026-09-10)"
EXP82_CKPT = "exp82_ckpt"
RESULTS_OUT = "results_exp82.json"

ARMS = ["A3-T2-identity", "A3-T2-uniform"]
SEED_BASE = 82000
N_SEEDS_82 = 15
SEEDS = list(range(SEED_BASE, SEED_BASE + N_SEEDS_82))  # 82000..82014
BAR = 0.06  # standing magnitude bar (doc 74 SS2)

# Original exp73 seeds 0-4: motivating record, descriptive-only
EXP73_DESCRIPTIVE = {
    "seeds": list(range(5)),
    "source": "results_exp73.json :: A3-T2-identity vs A3-T2-uniform (5 seeds)",
    "note": (
        "MOTIVATING RECORD ONLY -- these are the original exp73 seeds 0-4. "
        "Per doc 74 SS1 they are NEVER pooled into this test. "
        "Quoted descriptive-only for context."
    ),
    "observed_effects": {
        "exp73_A3_stage1": "+0.098",
        "exp73_confirm": "+0.057",
        "exp75_L2": "+0.039",
        "exp75_L3": "+0.040",
        "doc70_pooled_estimate_forbidden": "+0.078 (pooling was retroactively forbidden)",
    },
}


def load_checkpoint(arm, seed):
    ckj = os.path.join(EXP82_CKPT, "%s_s%d.json" % (arm, seed))
    if not os.path.exists(ckj):
        return None
    return json.load(open(ckj, encoding="utf-8"))


def safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def safe_sd(xs):
    if len(xs) < 2:
        return 0.0
    m = safe_mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired_t_one_sided(diffs):
    """Paired one-sided t-test: H1: mean(diff) > 0.
    Uses scipy.stats.ttest_rel with alternative='greater'.
    Returns a dict with all test statistics."""
    n = len(diffs)
    if n < 2:
        return None
    # ttest_rel takes two arrays; diff[i] = a[i] - b[i]; H1: a > b
    zeros = [0.0] * n
    result = sps.ttest_rel(diffs, zeros, alternative="greater")
    mean_d = safe_mean(diffs)
    sd_d = safe_sd(diffs)
    se = sd_d / math.sqrt(n)
    return {
        "n": n,
        "df": n - 1,
        "mean_diff": mean_d,
        "sd_diff": sd_d,
        "se_diff": se,
        "t_stat": float(result.statistic),
        "p_value": float(result.pvalue),
        "one_sided_direction": "identity > uniform",
        "alpha": 0.05,
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


def main():
    t0 = time.time()

    # Load checkpoints
    id_runs = {}
    un_runs = {}
    missing = []
    for seed in SEEDS:
        r = load_checkpoint("A3-T2-identity", seed)
        if r is None:
            missing.append(("A3-T2-identity", seed))
        else:
            id_runs[seed] = r
        r = load_checkpoint("A3-T2-uniform", seed)
        if r is None:
            missing.append(("A3-T2-uniform", seed))
        else:
            un_runs[seed] = r

    complete_pairs = [s for s in SEEDS if s in id_runs and s in un_runs]
    n_complete = len(complete_pairs)

    # Per-seed table
    per_seed = []
    diffs = []
    for i, seed in enumerate(SEEDS):
        row = {
            "seed_index": i,
            "seed": seed,
            "identity_available": seed in id_runs,
            "uniform_available": seed in un_runs,
        }
        if seed in id_runs:
            r = id_runs[seed]
            row["identity_acc"] = r["heldout_test_acc"]
            row["identity_train_acc"] = r.get("train_acc")
            row["identity_val_acc"] = r.get("val_acc")
            row["identity_selected_epoch"] = r.get("selected_epoch")
            rd = r.get("init_scheme", {}).get("redraw_doc62_SS24_3", {})
            row["redraw_attempts_used"] = rd.get("attempts_used")
            row["redraw_effective_seed"] = rd.get("effective_seed_of_record")
            row["redraw_refused_count"] = len(rd.get("refused_attempts", []))
        if seed in un_runs:
            r = un_runs[seed]
            row["uniform_acc"] = r["heldout_test_acc"]
            row["uniform_train_acc"] = r.get("train_acc")
            row["uniform_val_acc"] = r.get("val_acc")
            row["uniform_selected_epoch"] = r.get("selected_epoch")
        if seed in id_runs and seed in un_runs:
            d = id_runs[seed]["heldout_test_acc"] - un_runs[seed]["heldout_test_acc"]
            row["diff_identity_minus_uniform"] = d
            diffs.append(d)
        per_seed.append(row)

    id_accs = [id_runs[s]["heldout_test_acc"] for s in complete_pairs]
    un_accs = [un_runs[s]["heldout_test_acc"] for s in complete_pairs]

    # Compute booleans and t-test
    if n_complete < N_SEEDS_82:
        # Incomplete: booleans are null, never false (exp73_aggregate discipline)
        identity_direction_real = None
        identity_effect_clears_bar = None
        t_result = None
        mean_diff = None
        direction_reason = (
            "INCOMPLETE: only %d/%d seed pairs available. "
            "Booleans are null, never false, per the standing discipline."
            % (n_complete, N_SEEDS_82)
        )
        bar_reason = direction_reason
    else:
        mean_diff = safe_mean(diffs)
        t_result = paired_t_one_sided(diffs)

        # Boolean 1: direction real
        identity_direction_real = bool(t_result["p_value"] < 0.05)
        direction_reason = (
            "Paired seed t-test (diff_i = identity_acc_i - uniform_acc_i for "
            "seed_index i=0..14), H1: mean diff > 0, one-sided, df=14. "
            "t = %.4f, p = %.4f. Threshold: p < 0.05. "
            "Result: %s."
            % (
                t_result["t_stat"],
                t_result["p_value"],
                "REAL (p < 0.05)" if identity_direction_real
                else "NOT REAL (p >= 0.05)",
            )
        )

        # Boolean 2: magnitude clears bar
        identity_effect_clears_bar = bool(mean_diff >= BAR)
        bar_reason = (
            "Mean difference (identity - uniform) = %.4f. "
            "Standing bar: %.2f. "
            "Result: %s. %s"
            % (
                mean_diff,
                BAR,
                "CLEARS BAR" if identity_effect_clears_bar else "BELOW BAR",
                (
                    "identity_effect_clears_bar firing owes the standard "
                    "seed+1000 confirm draw (doc 74 SS2)."
                    if identity_effect_clears_bar
                    else "Bar not cleared; no confirm owed on this boolean."
                ),
            )
        )

    # Redraw provenance summary
    id_seeds_done = [s for s in SEEDS if s in id_runs]
    all_attempts = [
        id_runs[s].get("init_scheme", {}).get("redraw_doc62_SS24_3", {}).get("attempts_used")
        for s in id_seeds_done
    ]
    redraw_summary = {
        "total_identity_runs": len(id_runs),
        "runs_with_zero_redraws": sum(1 for a in all_attempts if a == 1),
        "runs_with_one_or_more_redraws": sum(1 for a in all_attempts
                                             if a is not None and a > 1),
        "runs_with_unknown_provenance": sum(1 for a in all_attempts if a is None),
        "attempts_per_seed": {
            str(s): id_runs[s].get("init_scheme", {})
                              .get("redraw_doc62_SS24_3", {})
                              .get("attempts_used")
            for s in id_seeds_done
        },
    }

    out = {
        "experiment": "exp82 -- A3-T2 identity direction, decided properly",
        "spec": SPEC,
        "written_by": "exp82_aggregate.py (sole writer of this file)",
        "platform": platform.platform(),
        "python": platform.python_version(),

        "n_seeds_registered": N_SEEDS_82,
        "seed_base": SEED_BASE,
        "seeds": SEEDS,
        "n_complete_pairs": n_complete,
        "n_missing": len(missing),
        "missing_runs": missing,

        # SS2 booleans, decoupled
        "identity_direction_real": identity_direction_real,
        "identity_direction_real_reason": direction_reason,
        "identity_effect_clears_bar": identity_effect_clears_bar,
        "identity_effect_clears_bar_reason": bar_reason,

        # t-test details
        "t_test": t_result,
        "mean_diff_identity_minus_uniform": mean_diff,
        "diffs": diffs if n_complete == N_SEEDS_82 else diffs,

        # arm summaries
        "arms": {
            "A3-T2-identity": arm_summary(id_accs, "A3-T2-identity"),
            "A3-T2-uniform": arm_summary(un_accs, "A3-T2-uniform"),
        },

        # per-seed table
        "per_seed": per_seed,

        # redraw provenance (doc 62 SS24.3)
        "redraw_provenance_doc62_SS24_3": redraw_summary,

        # motivating record (descriptive only, never pooled)
        "motivating_record_descriptive_only": EXP73_DESCRIPTIVE,

        # obligations and outcomes
        "obligations": {
            "identity_direction_real_alone": (
                "direction_real alone does not owe a confirm -- "
                "this IS the prospective confirm (doc 74 SS2)."
            ),
            "identity_effect_clears_bar_fires": (
                "identity_effect_clears_bar firing owes the standard "
                "seed+1000 draw (doc 74 SS2)."
            ),
            "p_ge_0.05_outcome": (
                "p >= 0.05 at this power = the direction was selection noise "
                "all along, closed as such (doc 74 SS3)."
            ),
        },

        "wall_time_sec": time.time() - t0,
    }

    json.dump(out, open(RESULTS_OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # Console summary
    print("\n=== exp82 aggregate ===")
    print("Complete pairs: %d/%d" % (n_complete, N_SEEDS_82))
    if n_complete == N_SEEDS_82:
        print("identity mean=%.4f  uniform mean=%.4f  diff=%.4f"
              % (safe_mean(id_accs), safe_mean(un_accs), mean_diff))
        print("t=%.4f  df=%d  p=%.4f  (one-sided, H1: identity > uniform)"
              % (t_result["t_stat"], t_result["df"], t_result["p_value"]))
        print("identity_direction_real    = %-5s  (p < 0.05 = %s)"
              % (identity_direction_real, identity_direction_real))
        print("identity_effect_clears_bar = %-5s  (mean diff >= %.2f = %s)"
              % (identity_effect_clears_bar, BAR, identity_effect_clears_bar))
    else:
        print("Incomplete: booleans null. Missing: %r" % missing)
    rd_sum = redraw_summary
    print("Redraw incidents: %d/%d identity seeds needed >= 1 redraw"
          % (rd_sum["runs_with_one_or_more_redraws"], rd_sum["total_identity_runs"]))
    print("Wrote: %s" % RESULTS_OUT)


if __name__ == "__main__":
    main()
