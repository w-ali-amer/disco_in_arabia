# -*- coding: utf-8 -*-
"""exp82c aggregate -- SOLE WRITER of results_exp82_confirm.json.

NOT run during training. Run after all 30 runs complete.

This is the magnitude-confirm aggregator for doc 74 §4.1: the same
30-run design as exp82 (A3-T2-identity vs A3-T2-uniform), on the
committed seed-70000 confirm dataset (stories_exp70_confirm_QA.json,
sha 59e6ada8...), seed base 82500. Verdict is on THIS draw alone.

Computes the two registered booleans (doc 74 §2), decoupled with reasons:
  identity_direction_real     -- paired seed t (identity - uniform),
                                 one-sided p < 0.05, df=14 at 15 seeds
  identity_effect_clears_bar  -- mean difference >= 0.06

Both booleans are null (never false) if <30 checkpoints are available.

Every checkpoint's dataset sha is asserted against the confirm sha
(59e6ada8...) before any computation. Seeds from the exp73 motivating
record (0-4) and the exp82 prospective-confirm draw (82000-82014) are
quoted descriptive-only; they contribute zero data to any computation.

Usage:
  python exp82c_aggregate.py
"""
import json
import math
import os
import platform
import sys
import time

from scipy import stats as sps

SPEC = "qnlp_private_docs/74_exp82_init_direction.md §4.1 (magnitude-confirm, 2026-09-10)"
EXP82C_CKPT = "exp82c_ckpt"
RESULTS_OUT = "results_exp82_confirm.json"

CONFIRM_SHA = ("59e6ada8a98b8917a1bd8e927d09249132a9765f7002e40fceb807a4bd51"
               "cd7c")

ARMS = ["A3-T2-identity", "A3-T2-uniform"]
SEED_BASE = 82500
N_SEEDS_82C = 15
SEEDS = list(range(SEED_BASE, SEED_BASE + N_SEEDS_82C))  # 82500..82514
BAR = 0.06  # standing magnitude bar (doc 74 §2)

# Exp82 (prospective-confirm, doc 74 §4.1): motivating record, descriptive-only
EXP82_DESCRIPTIVE = {
    "seeds": list(range(82000, 82015)),
    "seed_base": 82000,
    "source": "results_exp82.json :: A3-T2-identity vs A3-T2-uniform (15 seeds)",
    "note": (
        "PROSPECTIVE-CONFIRM DRAW (doc 74 §4.1) -- these are the exp82 seeds "
        "82000-82014. Per doc 74 §4.1 they are NEVER pooled into this test. "
        "Quoted descriptive-only: identity 0.7233 vs uniform 0.6387, "
        "diff +0.0847, paired t=3.56, df=14, p=0.0016 one-sided. "
        "Both booleans fired on that draw; THIS draw confirms the magnitude boolean."
    ),
    "observed": {
        "identity_mean": 0.7233,
        "uniform_mean": 0.6387,
        "mean_diff": 0.0847,
        "t_stat": 3.56,
        "p_value": 0.0016,
        "identity_direction_real": True,
        "identity_effect_clears_bar": True,
    },
}

# Exp73 motivating record (original 5 seeds), also descriptive-only
EXP73_DESCRIPTIVE = {
    "seeds": list(range(5)),
    "source": "results_exp73.json :: A3-T2-identity vs A3-T2-uniform (5 seeds)",
    "note": (
        "ORIGINAL MOTIVATING RECORD (exp73, seeds 0-4). "
        "Per doc 74 §1 never pooled into any test. Quoted descriptive-only."
    ),
    "observed_effects": {
        "exp73_A3_stage1": "+0.098",
        "exp73_confirm": "+0.057",
        "exp75_L2": "+0.039",
        "exp75_L3": "+0.040",
    },
}


def load_checkpoint(arm, seed):
    ckj = os.path.join(EXP82C_CKPT, "%s_s%d.json" % (arm, seed))
    if not os.path.exists(ckj):
        return None
    r = json.load(open(ckj, encoding="utf-8"))
    # Assert confirm sha on every loaded checkpoint
    ck_sha = r.get("dataset_sha256")
    assert ck_sha == CONFIRM_SHA, (
        "STOP: checkpoint %s s%d has dataset sha %r -- expected confirm sha %s"
        % (arm, seed, ck_sha, CONFIRM_SHA))
    return r


def safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def safe_sd(xs):
    if len(xs) < 2:
        return 0.0
    m = safe_mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired_t_one_sided(diffs):
    """Paired one-sided t-test: H1: mean(diff) > 0.
    Mirrors exp82_aggregate.py exactly: scipy.stats.ttest_rel vs zeros."""
    n = len(diffs)
    if n < 2:
        return None
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
    if n_complete < N_SEEDS_82C:
        identity_direction_real = None
        identity_effect_clears_bar = None
        t_result = None
        mean_diff = None
        direction_reason = (
            "INCOMPLETE: only %d/%d seed pairs available. "
            "Booleans are null, never false, per the standing discipline."
            % (n_complete, N_SEEDS_82C)
        )
        bar_reason = direction_reason
    else:
        mean_diff = safe_mean(diffs)
        t_result = paired_t_one_sided(diffs)

        # Boolean 1: direction real
        identity_direction_real = bool(t_result["p_value"] < 0.05)
        direction_reason = (
            "Paired seed t-test (diff_i = identity_acc_i - uniform_acc_i for "
            "seed_index i=0..14, seeds 82500..82514), H1: mean diff > 0, "
            "one-sided, df=14. "
            "t = %.4f, p = %.4f. Threshold: p < 0.05. "
            "Result: %s. "
            "VERDICT ON THIS DRAW ALONE (exp82 numbers quoted descriptive-only)."
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
                    "seed+1000 draw (doc 74 §2)."
                    if identity_effect_clears_bar
                    else "Bar not cleared; no further confirm owed on this boolean."
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
        "experiment": "exp82c -- A3-T2 identity magnitude-confirm (doc 74 §4.1)",
        "spec": SPEC,
        "written_by": "exp82c_aggregate.py (sole writer of results_exp82_confirm.json)",
        "platform": platform.platform(),
        "python": platform.python_version(),

        "dataset": "stories_exp70_confirm_QA.json",
        "dataset_sha_confirmed": CONFIRM_SHA,
        "vsuite": "results_exp70_confirm_WN_validation.json",

        "n_seeds_registered": N_SEEDS_82C,
        "seed_base": SEED_BASE,
        "seeds": SEEDS,
        "n_complete_pairs": n_complete,
        "n_missing": len(missing),
        "missing_runs": missing,

        # §2 booleans, decoupled, this draw alone
        "identity_direction_real": identity_direction_real,
        "identity_direction_real_reason": direction_reason,
        "identity_effect_clears_bar": identity_effect_clears_bar,
        "identity_effect_clears_bar_reason": bar_reason,

        # t-test details
        "t_test": t_result,
        "mean_diff_identity_minus_uniform": mean_diff,
        "diffs": diffs,

        # arm summaries
        "arms": {
            "A3-T2-identity": arm_summary(id_accs, "A3-T2-identity"),
            "A3-T2-uniform": arm_summary(un_accs, "A3-T2-uniform"),
        },

        # per-seed table
        "per_seed": per_seed,

        # redraw provenance (doc 62 SS24.3)
        "redraw_provenance_doc62_SS24_3": redraw_summary,

        # prior record (descriptive only, never pooled)
        "exp82_draw_descriptive_only": EXP82_DESCRIPTIVE,
        "exp73_motivating_record_descriptive_only": EXP73_DESCRIPTIVE,

        # obligations
        "obligations": {
            "identity_direction_real_verdict": (
                "direction_real on this draw is a further data point; "
                "the prospective confirm (doc 74 §2) was exp82 (seed base 82000). "
                "This draw supplies the magnitude-confirm verdict."
            ),
            "identity_effect_clears_bar_fires": (
                "identity_effect_clears_bar firing would owe the standard "
                "seed+1000 draw (doc 74 §2)."
            ),
            "draw_scope": (
                "VERDICT ON THIS DRAW ALONE. exp82 seeds 82000-82014 and "
                "exp73 seeds 0-4 are quoted descriptive-only; "
                "zero contribution to any computation here."
            ),
        },

        "wall_time_sec": time.time() - t0,
    }

    json.dump(out, open(RESULTS_OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # Console summary
    print("\n=== exp82c aggregate (magnitude-confirm) ===")
    print("Dataset: %s (sha %s...)" % ("stories_exp70_confirm_QA.json", CONFIRM_SHA[:16]))
    print("Complete pairs: %d/%d" % (n_complete, N_SEEDS_82C))
    if n_complete == N_SEEDS_82C:
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
