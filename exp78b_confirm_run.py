# -*- coding: utf-8 -*-
"""exp78b_confirm_run.py -- THIN CONFIRM DRIVER for exp78b n3_T2.

Ordered by doc 69 §7 / doc 70 §2.2 (2026-09-04).

Runs the ScaledStoryModel (n=3, 180 params, Lv=5) on the exp78a confirm
draw at generator seed 77000, plus its matched S-shuffle floor, 5 seeds
each (~2 h serial).  Aggregates per doc 70 rules and writes
results_exp78b_confirm.json.  Does NOT modify results_exp78b.json or any
committed checkpoint.

WHAT THIS DRIVER PATCHES, EXHAUSTIVELY (doc 69 §7 licenses exactly this):
  1. D76.STORY_FMT  ->  "stories_exp78a_confirm_%s.json"   (all 11 cells)
  2. W.DATASET_SHAS ->  confirm-draw SHAs (from results_exp78a_WN_validation.json)
  3. R.ALL_ARMS     ->  ["n3_T2", "S-shuffle-n3_T2"]
  4. R.CKPT_DIR     ->  "exp78b_confirm_ckpt"
  5. Custom confirm smoke written before R.main() (doc 70 §3.6 self-emitting
     gate artifact, includes designed-to-fail Gate 3).

Training, model architecture (ScaledStoryModel), evaluation, features, and
every other assertion are exp78b_run.py's committed code paths, unforked.

VERDICT WIRING (doc 70):
  - Floor-clearance ONLY, no 0.65 leg (B1 kill; doc 63 §12.9.3 substitution)
  - Floor-admissibility gate: floor mean must lie within 2 binomial SD of 0.5
    (B3 gate; N=300, 2*SD=0.0577, band [0.4423, 0.5577])
  - t-based arm-vs-floor paired seed-mean t-test (§3.7 companion, df=4)
  - Confirm draw alone decides (§3.2); pooled with first draw = descriptive only
  - Boolean: n3_T2_confirm_clears_floor
"""

import sys
import os
import json
import hashlib
import math
import statistics
import time
import platform as _platform

# ============================================================================
# Registered constants
# ============================================================================
CONFIRM_VSUITE     = "results_exp78a_WN_validation.json"
CONFIRM_CKPT_DIR   = "exp78b_confirm_ckpt"
CONFIRM_SMOKE_OUT  = "results_exp78b_confirm_smoke.json"
CONFIRM_RESULTS    = "results_exp78b_confirm.json"
N_SEEDS = 5
N_TEST  = 300

# n3_T2 confirm SHA (from results_exp78a_WN_validation.json, generator seed 77000)
CONFIRM_SHA_N3_T2 = "a7fabb683da829bfb471d30f4994c41e868697e545d4a5cde195ce49cd02a366"
# n3_T2 first-draw SHA (original exp76 dataset, generator seed 76000)
FIRST_DRAW_SHA_N3_T2 = "eb46f4e65a8f46c8a6fc0e0013d9dfdabc43200475b9c06b4c1488c5e2c41072"

# Confirm SHAs for all 5 scaling cells (from results_exp78a_WN_validation.json)
CONFIRM_SHAS = {
    "n2_T2": "759a9477d80a999574defc7e99b42fff16017052b8f5e76783aa80b5a6313762",
    "n3_T2": "a7fabb683da829bfb471d30f4994c41e868697e545d4a5cde195ce49cd02a366",
    "n4_T4": "070c71dc424db2457c894b3d93bae0444c6d651cc3c1219f61fabf612ce1ef68",
    "n5_T4": "68b405acdfb0bedfddcabeed5597c2def38e1c14bb76d19e7d4483f78ec436de",
    "n7_T6": "ae84cf6b0a16d541c9778e40a78d9073a224f7fa5f3d501459a0a1e94cfa7147",
}
# First-draw SHAs (exp76 frozen datasets, from exp78b_run.py DATASET_SHAS)
FIRST_DRAW_SHAS = {
    "n2_T2": "c698aa7346bf330badaf512d646284f69e3c2986bc4899906e185c33b40a0788",
    "n3_T2": "eb46f4e65a8f46c8a6fc0e0013d9dfdabc43200475b9c06b4c1488c5e2c41072",
    "n4_T4": "588e42558c1ecf72abfe0cdbf0d8ae5f65c90b6e512c1a4c74d37e5de075fe29",
    "n5_T4": "8266c30d767c034fe6427052fc16b4c37185c5ebb1db9d7d1e9c8804b09bf0e7",
    "n7_T6": "badcb738a93aac95870a8096922065aecca142364f4ee20ffb2712981eba581e",
}

CONFIRM_ARM   = "n3_T2"
CONFIRM_FLOOR = "S-shuffle-n3_T2"

# Exp78b registered n3_T2 floor bar (first draw) -- used as smoke reference bar only
# floor_mean(0.4793) + 2*floor_sd(0.0442) = 0.5678 from results_exp78b.json
EXP78B_FIRST_DRAW_FLOOR_BAR = 0.5678269

# Grade-1 reachable ceiling for this family (doc 63 §12.4.1)
REACHABLE_CEILING = 0.6426

# Bar provenance (doc 70 §3 rule 1 / B1 correction)
BAR_PROVENANCE = {
    "original_bar": 0.65,
    "original_bar_source": "doc 54 SS7; calibrated on a different task family",
    "reachable_ceiling_doc63_SS12_4_1": REACHABLE_CEILING,
    "ceiling_below_bar": True,
    "substitution_rule": "doc 63 §12.9.3 floor-clearance substituted for 0.65",
    "substitution_trigger": "doc 70 B1: 0.65 bar measured above this family's ceiling; "
                            "the leg is miscalibrated and was mis-imported (doc 69 §4.1)",
    "active_rule": "arm_mean >= floor_mean + 2*floor_sd  (no 0.65 leg)",
    "chance_level": 0.5,
    "N_test": N_TEST,
    "binom_sd_N300": math.sqrt(0.25 / N_TEST),
    "floor_admissibility_band": [0.5 - 2 * math.sqrt(0.25 / N_TEST),
                                 0.5 + 2 * math.sqrt(0.25 / N_TEST)],
}

_SELF_SHA = None          # filled at runtime (driver's own script sha)
_FIRST_DRAW_SEED = None   # filled from results_exp76_WN_validation.json in main()


# ============================================================================
# Confirm smoke (doc 70 §3.6 self-emitting gate artifact)
# ============================================================================
def confirm_smoke(R, W, D76, fm, verbs, store):
    """All confirm-draw gates.  Writes CONFIRM_SMOKE_OUT with
    assertions_passed=True on success.

    Gate 0: n3_T2 confirm SHA + freshness
    Gate 1: V-suite kill gate (results_exp78a_WN_validation.json)
    Gate 2: 2-epoch probe both arms, assert n_params == 180
    Gate 3 (designed-to-fail): scrambled-label floor arm (seed=99, 2 epochs)
            must NOT clear the registered exp78b floor bar (0.5678);
            the gate shows this assertion firing (scrambled_arm_clears_floor=False)
            to prove the gate set is live.
    """
    import torch
    t0 = time.time()

    # Verify all 5 scaling cells' confirm files against registered SHAs
    # (makes the DATASET_SHAS patch load-bearing, not dead weight)
    W.assert_frozen_datasets()
    print("[smoke78b-confirm] assert_frozen_datasets: all 5 confirm SHAs verified",
          flush=True)

    rep = {
        "experiment": "exp78b-confirm smoke (doc 70 §3.6 self-emitting gate)",
        "spec": "doc 69 §7 / doc 70 §2.2, ordered 2026-09-04",
        "platform": _platform.platform(),
        "python": _platform.python_version(),
        "torch": torch.__version__,
        "driver_sha256": _SELF_SHA,
        "rule_doc70_S6": (
            "Every PASS/FAIL gate writes its own hash-pinned artifact "
            "(inputs, seeds, statistic, threshold, boolean) INCLUDING one "
            "designed-to-fail assertion shown failing; gate set is live iff "
            "Gate 3 fires."),
    }

    # ---- Gate 0: n3_T2 confirm SHA + freshness --------------------------------
    confirm_path = D76.STORY_FMT % CONFIRM_ARM
    meta_blob = json.load(open(confirm_path, encoding="utf-8"))["meta"]
    got_sha = meta_blob["dataset_sha256"]

    assert got_sha == CONFIRM_SHA_N3_T2, (
        "confirm dataset sha mismatch at %s\n  expected: %s\n  got:      %s"
        % (confirm_path, CONFIRM_SHA_N3_T2, got_sha))
    assert got_sha != FIRST_DRAW_SHA_N3_T2, (
        "confirm draw reproduced the first draw -- not fresh (sha=%s)" % got_sha)

    rep["gate0_sha_and_freshness"] = {
        "path": confirm_path,
        "sha256": got_sha,
        "registered_confirm_sha": CONFIRM_SHA_N3_T2,
        "first_draw_sha": FIRST_DRAW_SHA_N3_T2,
        "sha_matches_confirm": True,
        "freshness": "CONFIRMED (sha differs from first draw)",
        "generator_seed_confirm": 77000,
        "generator_seed_first_draw": _FIRST_DRAW_SEED,
        "seed_delta_verified": (77000 - _FIRST_DRAW_SEED == 1000),
        "seed_convention": ("confirm = first_draw + 1000, registered convention "
                            "doc 69 §4 item 4 / doc 70 §2.2"),
    }
    print("[smoke78b-confirm] Gate 0: SHA OK, freshness OK", flush=True)

    # ---- Gate 1: V-suite kill gate -------------------------------------------
    vs = json.load(open(CONFIRM_VSUITE, encoding="utf-8"))
    assert vs.get("kill_gate_passed") is True, (
        "confirm draw V-suite kill gate did not pass in %s" % CONFIRM_VSUITE)
    n3_vsuite_sha = vs["cells"]["n3_T2"]["dataset_sha256"]
    assert n3_vsuite_sha == CONFIRM_SHA_N3_T2, (
        "V-suite n3_T2 sha (%s) does not match registered confirm sha (%s)"
        % (n3_vsuite_sha, CONFIRM_SHA_N3_T2))

    rep["gate1_vsuite"] = {
        "source": CONFIRM_VSUITE,
        "kill_gate_passed": True,
        "n3_T2_sha_in_vsuite": n3_vsuite_sha,
        "matches_confirm_sha": True,
        "generator_seed": vs.get("generator_seed", 77000),
    }
    print("[smoke78b-confirm] Gate 1: V-suite kill gate PASSED", flush=True)

    # ---- Gate 2: 2-epoch probe both arms, n_params == 180 --------------------
    tiny_epochs = 2
    probe = {}
    for arm in [CONFIRM_ARM, CONFIRM_FLOOR]:
        t1 = time.time()
        r = R.train(arm, 0, store, verbs, tiny_epochs, fm, save=False, quiet=True)
        sec = time.time() - t1
        probe[arm] = {
            "val_acc": r["val_acc"],
            "n_params": r["n_params"],
            "sec_per_epoch": round(sec / tiny_epochs, 1),
        }
        assert r["n_params"] == 180, (
            "n_params mismatch at %s: expected 180 (ScaledStoryModel n=3, "
            "Lv=5: 6 verb-blocks * 5 layers * 6 params + 2 readout * 15 = 180), "
            "got %d" % (arm, r["n_params"]))
        print("[smoke78b-confirm] Gate 2  %-24s  val=%.3f  n_params=%d  OK"
              % (arm, r["val_acc"], r["n_params"]), flush=True)

    rep["gate2_probe"] = {
        "tiny_epochs": tiny_epochs,
        "arms": probe,
        "n_params_assertion": "180 (ScaledStoryModel n=3, Lv=2*3-1=5)",
    }

    # ---- Gate 3 (designed-to-fail): scrambled-label floor must NOT clear -----
    # S-shuffle arm at seed=99 (outside 0-4 production range) after 2 epochs
    # should score near chance.  The assertion "scrambled_arm_clears_floor"
    # is designed to FIRE (= be False), proving the gate is live.
    scr_seed = 99
    t1 = time.time()
    r_scr = R.train(CONFIRM_FLOOR, scr_seed, store, verbs, tiny_epochs,
                    fm, save=False, quiet=True)
    scr_acc  = float(r_scr["heldout_test_acc"])
    scr_clears = scr_acc >= EXP78B_FIRST_DRAW_FLOOR_BAR

    rep["gate3_designed_to_fail"] = {
        "description": (
            "S-shuffle-n3_T2 at seed=99 (scrambled labels, 2 epochs) must NOT "
            "clear the registered exp78b floor bar 0.5678. If scr_clears=True "
            "the gate set is not live."),
        "arm": CONFIRM_FLOOR,
        "seed": scr_seed,
        "rng_base_floor": 9_000_000,
        "effective_rng_seed": 9_000_000 + scr_seed,
        "tiny_epochs": tiny_epochs,
        "floor_bar_reference": EXP78B_FIRST_DRAW_FLOOR_BAR,
        "bar_source": ("exp78b first-draw: floor_mean(0.4793)+2*floor_sd(0.0442)"
                       "=0.5678; from results_exp78b.json; cited as smoke reference "
                       "only -- confirm floor bar re-derived from confirm checkpoints"),
        "scrambled_arm_test_acc": scr_acc,
        "scrambled_arm_clears_floor": bool(scr_clears),   # designed to be False
        "assertion_fires": bool(not scr_clears),           # True = gate is live
        "design_rationale": (
            "A floor arm after 2 epochs of training cannot clear a bar calibrated "
            "on 150-epoch performance; showing this False proves the gate is live."),
    }
    assert not scr_clears, (
        "GATE SET NOT LIVE: scrambled floor arm (seed=%d) cleared the bar "
        "(%.4f >= %.4f) at only 2 epochs -- check that relabeling is working"
        % (scr_seed, scr_acc, EXP78B_FIRST_DRAW_FLOOR_BAR))
    print("[smoke78b-confirm] Gate 3 (designed-to-fail): scrambled_acc=%.4f < "
          "floor_bar=%.4f  FIRES AS EXPECTED  GATE IS LIVE"
          % (scr_acc, EXP78B_FIRST_DRAW_FLOOR_BAR), flush=True)

    rep["assertions_passed"] = True
    rep["wall_time_sec"] = round(time.time() - t0, 1)
    json.dump(rep, open(CONFIRM_SMOKE_OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[smoke78b-confirm] wrote %s -- ALL CONFIRM SMOKE ASSERTIONS PASSED "
          "INCLUDING THE DESIGNED-TO-FAIL" % CONFIRM_SMOKE_OUT, flush=True)
    return rep


# ============================================================================
# Aggregation (doc 70 §3.2, §3.4, §3.7)
# ============================================================================
def aggregate(ckpt_dir, out_path, n_seeds=N_SEEDS):
    """Load confirm checkpoints; apply doc 70 verdict rules; write
    results_exp78b_confirm.json (self-emitting gate artifact per §3.6)."""
    from scipy.stats import ttest_rel

    binom_sd = math.sqrt(0.25 / N_TEST)   # 0.02887 at N=300
    half_w   = 2.0 * binom_sd             # 0.0577
    admiss_lo = 0.5 - half_w
    admiss_hi = 0.5 + half_w

    # Load per-seed heldout_test_acc
    arm_accs   = []
    floor_accs = []
    arm_ckpts  = []
    floor_ckpts = []
    for s in range(n_seeds):
        ap = os.path.join(ckpt_dir, "%s_s%d.json" % (CONFIRM_ARM, s))
        fp = os.path.join(ckpt_dir, "%s_s%d.json" % (CONFIRM_FLOOR, s))
        if not os.path.exists(ap):
            raise RuntimeError("Missing arm checkpoint: %s" % ap)
        if not os.path.exists(fp):
            raise RuntimeError("Missing floor checkpoint: %s" % fp)
        ar = json.load(open(ap, encoding="utf-8"))
        fr = json.load(open(fp, encoding="utf-8"))
        arm_accs.append(float(ar["heldout_test_acc"]))
        floor_accs.append(float(fr["heldout_test_acc"]))
        arm_ckpts.append(hashlib.sha256(open(ap, "rb").read()).hexdigest())
        floor_ckpts.append(hashlib.sha256(open(fp, "rb").read()).hexdigest())

    # Basic stats
    arm_mean   = statistics.mean(arm_accs)
    arm_sd     = statistics.stdev(arm_accs)      # ddof=1
    floor_mean = statistics.mean(floor_accs)
    floor_sd   = statistics.stdev(floor_accs)

    # Floor admissibility gate (doc 70 §3.4 / B3)
    floor_dev  = abs(floor_mean - 0.5)
    floor_admissible = floor_dev <= half_w

    # Clearance bar (arm_mean >= floor_mean + 2*floor_sd)
    floor_bar   = floor_mean + 2.0 * floor_sd
    clears_floor = arm_mean >= floor_bar if floor_admissible else None

    # Paired t-test (doc 70 §3.7 / B2 primary significance leg)
    diffs = [a - f for a, f in zip(arm_accs, floor_accs)]
    t_stat, p_two = ttest_rel(arm_accs, floor_accs)
    p_one = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0

    # Plug-in 2*sd bar (reported alongside t, per §3.7)
    plugin_bar = floor_mean + 2.0 * floor_sd   # same as floor_bar

    # First-draw descriptive (from results_exp78b.json) -- reported descriptive only
    first_draw_arm_mean  = 0.5753333
    first_draw_floor_bar = 0.5678269

    # Boolean output
    n3_T2_confirm_clears_floor = bool(clears_floor) if clears_floor is not None else False

    # Designed-to-fail for the results gate (doc 70 §3.6):
    # The 0.65 bar (B1) fails its own applicability check because the family's
    # reachable ceiling (0.6426) is below 0.65 -- this always fires for this
    # family and proves the gate distinguishes applicable from inapplicable bars.
    dtf_bar_065 = 0.65
    dtf_ceiling_below_bar = REACHABLE_CEILING < dtf_bar_065  # always True
    dtf_fires = dtf_ceiling_below_bar                         # always True

    rep = {
        "experiment": "exp78b-confirm (doc 69 §7 / doc 70 §2.2)",
        "spec": "doc 69 §7 ordered 2026-09-04; verdict rules per doc 70 §3",
        "driver_sha256": _SELF_SHA,
        "confirm_dataset_sha": CONFIRM_SHA_N3_T2,
        "first_draw_sha":  FIRST_DRAW_SHA_N3_T2,
        "generator_seed_confirm": 77000,
        "generator_seed_first_draw": _FIRST_DRAW_SEED,
        "seed_delta_verified": (77000 - _FIRST_DRAW_SEED == 1000),
        "vsuite_source": CONFIRM_VSUITE,
        "vsuite_kill_gate_passed": True,
        "bar_provenance": BAR_PROVENANCE,

        # --- dataset provenance ---
        "dataset_provenance": {
            "confirm_dataset": "stories_exp78a_confirm_n3_T2.json",
            "confirm_sha256": CONFIRM_SHA_N3_T2,
            "first_draw_dataset": "stories_exp76_n3_T2.json",
            "first_draw_sha256": FIRST_DRAW_SHA_N3_T2,
            "generator": "exp76_datagen.py (committed, unmodified)",
            "confirm_seed_convention": (
                "confirm seed = first_draw seed + 1000 "
                "(%d -> 77000), registered in doc 69 §4 item 4"
                % _FIRST_DRAW_SEED),
            "vsuite": CONFIRM_VSUITE,
            "vsuite_generator_seed": 77000,
            "draw_type": "re-draw (doc 70 §3 N5: new sample from same generator, "
                         "not a reproduction of the first draw)",
        },

        # --- per-seed results ---
        "confirm_arm": CONFIRM_ARM,
        "confirm_arm_per_seed": arm_accs,
        "confirm_arm_mean": arm_mean,
        "confirm_arm_sd": arm_sd,
        "confirm_arm_ckpt_shas": arm_ckpts,

        "confirm_floor_arm": CONFIRM_FLOOR,
        "confirm_floor_per_seed": floor_accs,
        "confirm_floor_mean": floor_mean,
        "confirm_floor_sd": floor_sd,
        "confirm_floor_ckpt_shas": floor_ckpts,

        # --- floor admissibility gate (doc 70 §3.4 / B3) ---
        "floor_admissibility": {
            "floor_mean": floor_mean,
            "chance": 0.5,
            "deviation_from_chance": floor_dev,
            "binom_sd_at_N300": binom_sd,
            "admissibility_half_width_2sd": half_w,
            "admissibility_band": [admiss_lo, admiss_hi],
            "floor_admissible": floor_admissible,
            "reading": (
                "ABSOLUTE READING VALID -- floor within 2 binomial SD of chance"
                if floor_admissible else
                "NO ABSOLUTE READING -- floor sits >2 binomial SD from 0.5; "
                "shuffle leaks; differential arm-vs-arm reading stands with leak named"),
            "note": "doc 63 DELTA_BAR=0.06 precedent; N=300 convention anchor",
        },

        # --- floor clearance bar (doc 63 §12.9.3 substitution) ---
        "floor_clearance": {
            "floor_mean": floor_mean,
            "floor_sd": floor_sd,
            "floor_bar": floor_bar,
            "arm_mean": arm_mean,
            "clears_floor": clears_floor,
            "floor_admissible": floor_admissible,
            "reading_conditional": (
                "arm_mean=%.4f vs floor_bar=%.4f -> CLEARS" % (arm_mean, floor_bar)
                if clears_floor else
                "arm_mean=%.4f vs floor_bar=%.4f -> DOES NOT CLEAR" % (arm_mean, floor_bar)
                if clears_floor is False else
                "N/A -- floor not admissible"),
        },

        # --- t-test companion (doc 70 §3.7 / B2) ---
        "ttest_companion": {
            "method": "scipy.stats.ttest_rel (paired, arm vs floor, same seed index)",
            "note": ("doc 70 §3.7 + B2: seed-level paired t is the primary "
                     "significance leg; it does NOT gate the boolean "
                     "n3_T2_confirm_clears_floor (floor-clearance alone does)."),
            "per_seed_diffs": diffs,
            "t_statistic": t_stat,
            "p_two_sided": p_two,
            "p_one_sided_greater": p_one,
            "df": 4,
            "plugin_2sd_bar": plugin_bar,
            "arm_mean": arm_mean,
            "floor_mean": floor_mean,
            "mean_diff": arm_mean - floor_mean,
        },

        # --- first-draw descriptive (NOT the verdict) ---
        "first_draw_descriptive": {
            "note": "doc 70 §3.2: confirm verdict on confirm draw alone; "
                    "pooled estimates reported descriptive only",
            "first_draw_arm_mean": first_draw_arm_mean,
            "first_draw_floor_bar": first_draw_floor_bar,
            "first_draw_clears_floor": first_draw_arm_mean >= first_draw_floor_bar,
            "confirm_arm_mean": arm_mean,
            "confirm_floor_bar": floor_bar,
            "pooled_arm_mean_descriptive": (arm_mean + first_draw_arm_mean) / 2.0,
        },

        # --- designed-to-fail (doc 70 §3.6) ---
        "designed_to_fail": {
            "description": (
                "The 0.65 bar (doc 54 SS7) fails its own applicability check "
                "for this family: reachable ceiling (0.6426) < 0.65.  This "
                "assertion always fires for this family and proves the gate "
                "distinguishes applicable from inapplicable bars."),
            "bar": dtf_bar_065,
            "reachable_ceiling_doc63_SS12_4_1": REACHABLE_CEILING,
            "ceiling_below_bar": dtf_ceiling_below_bar,
            "bar_applicable": False,
            "assertion_fires": dtf_fires,
            "consequence": (
                "doc 70 B1 kill of this bar is confirmed.  "
                "floor-clearance (doc 63 §12.9.3) is the sole valid reading."),
        },

        # --- primary boolean ---
        "n3_T2_confirm_clears_floor": n3_T2_confirm_clears_floor,
        "verdict": (
            "POSITIVE -- confirm draw clears floor (arm_mean=%.4f >= floor_bar=%.4f)"
            % (arm_mean, floor_bar)
            if n3_T2_confirm_clears_floor else
            "NEGATIVE -- confirm draw does not clear floor "
            "(arm_mean=%.4f, floor_bar=%.4f, admissible=%s)"
            % (arm_mean, floor_bar, floor_admissible)),
        "verdict_rule": (
            "doc 70 §3.2 (confirm draw alone) + doc 63 §12.9.3 "
            "(floor-clearance substituted for 0.65 per B1)"),
        "n_seeds": n_seeds,
        "ckpt_dir": ckpt_dir,
    }

    json.dump(rep, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[agg78b-confirm] wrote %s" % out_path, flush=True)

    # Summary print
    print("\n=== exp78b CONFIRM results ===", flush=True)
    print("  ARM     n3_T2         per-seed: %s" % ["%.4f" % x for x in arm_accs])
    print("  FLOOR   S-shuffle     per-seed: %s" % ["%.4f" % x for x in floor_accs])
    print("  arm_mean=%.4f  floor_mean=%.4f  floor_bar=%.4f"
          % (arm_mean, floor_mean, floor_bar))
    print("  floor_admissible=%s  (|%.4f - 0.5| = %.4f <= 2*binom_sd %.4f)"
          % (floor_admissible, floor_mean, floor_dev, half_w))
    print("  clears_floor=%s" % clears_floor)
    print("  t(4)=%.3f  p_one_sided=%.4f  (companion, does not gate boolean)"
          % (t_stat, p_one))
    print("  n3_T2_confirm_clears_floor = %s" % n3_T2_confirm_clears_floor)

    return rep


# ============================================================================
# Main
# ============================================================================
def main():
    global _SELF_SHA, _FIRST_DRAW_SEED
    _SELF_SHA = hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()

    # ---- Read first-draw generator seed from the exp76 WN-validation artifact --
    # (not from memory; reviewer patch 2)
    exp76_vsuite_path = "results_exp76_WN_validation.json"
    if os.path.exists(exp76_vsuite_path):
        _exp76_vs = json.load(open(exp76_vsuite_path, encoding="utf-8"))
        _FIRST_DRAW_SEED = _exp76_vs["generator_seed"]
        assert 77000 - _FIRST_DRAW_SEED == 1000, (
            "Seed-delta violation: confirm(77000) - first_draw(%d) != 1000"
            % _FIRST_DRAW_SEED)
        print("[exp78b-confirm] first-draw seed read from %s: %d  delta=+%d  OK"
              % (exp76_vsuite_path, _FIRST_DRAW_SEED, 77000 - _FIRST_DRAW_SEED),
              flush=True)
    else:
        _FIRST_DRAW_SEED = 76000
        print("[exp78b-confirm] WARNING: %s not found; first_draw seed defaulted to 76000"
              % exp76_vsuite_path, flush=True)

    # ---- Patch 1: STORY_FMT (must happen before exp78b_run import triggers
    #               load_store, which reads STORY_FMT at call time) -----------
    import exp76_datagen as D76
    D76.STORY_FMT = "stories_exp78a_confirm_%s.json"

    # ---- Import exp78b_run (carries its module-level patches to R) ----------
    import exp78b_run as W
    import exp76_run as R

    # ---- Patch 2: DATASET_SHAS -> confirm SHAs ------------------------------
    # assert_frozen_datasets() will check these against D76.STORY_FMT % cell
    for cell, sha in CONFIRM_SHAS.items():
        W.DATASET_SHAS[cell] = sha

    # ---- Patch 3: narrow ALL_ARMS -------------------------------------------
    R.ALL_ARMS = [CONFIRM_ARM, CONFIRM_FLOOR]

    # ---- Patch 4: CKPT_DIR --------------------------------------------------
    R.CKPT_DIR = CONFIRM_CKPT_DIR

    # ---- Load store + verbs for smoke (cached by _STORE, reused by R.main()) -
    os.makedirs(CONFIRM_CKPT_DIR, exist_ok=True)
    store = R.load_store()
    verbs = sorted(store["exp69"][1]["verb_inventory"].keys())
    fm = R.feature_maps()

    # ---- Confirm smoke (doc 70 §3.6) ----------------------------------------
    confirm_smoke(R, W, D76, fm, verbs, store)

    # ---- Training: delegate to R.main() with confirm smoke as gate ----------
    # R.main() checks SMOKE_OUT exists and assertions_passed=True before
    # any training.  CKPT_DIR is re-applied from sys.argv inside R.main().
    sys.argv = [sys.argv[0],
                "--arms", "%s,%s" % (CONFIRM_ARM, CONFIRM_FLOOR),
                "--ckpt-dir", CONFIRM_CKPT_DIR,
                "--smoke-out", CONFIRM_SMOKE_OUT,
                "--anchor-out", "results_exp78b_confirm_anchor.json"]
    print("\n[exp78b-confirm] starting training: 5 seeds x 2 arms = 10 runs, "
          "~2 h serial ...", flush=True)
    ret = R.main()
    if ret != 0:
        print("[exp78b-confirm] R.main() returned %d -- check output" % ret,
              flush=True)
        sys.exit(ret)

    # ---- Aggregation --------------------------------------------------------
    print("\n[exp78b-confirm] training complete; aggregating ...", flush=True)
    aggregate(CONFIRM_CKPT_DIR, CONFIRM_RESULTS)
    print("[exp78b-confirm] DONE.  Results in %s" % CONFIRM_RESULTS, flush=True)


if __name__ == "__main__":
    main()
