# -*- coding: utf-8 -*-
"""exp77_aggregate.py — aggregate exp77 checkpoint results; write results_exp77.json.

DO NOT RUN THIS DURING THE TRAINING LAUNCH. Run after all 25 units complete.

Reads: exp77_ckpt/<arm>_s<seed>_results.json (per-pair records, §12.5 inputs).
Writes: results_exp77.json (sole writer; do not write manually).

Booleans decided per §12.5 (two-leg structure from exp42r):
  transfers_across_roots_within_wazn:
    Leg 1: wazn mean accuracy >= b1 mean accuracy + 0.06 on primary_test
    Leg 2: one-sided paired permutation p < 0.05 (10k resamples, seed 77777)
    (sign-flip over test items on seed-averaged per-item correctness)

  transfers_across_wazn_within_root:
    Same two-leg form on secondary_test.

  structure_not_semantics:
    Primary effect (leg 1 + leg 2) survives leave-فاعِل-out filter:
    same two-leg criterion on pairs where neither col is 'faail'.

  structure_specificity_control (invalidation clause §12.5):
    If wazn_shuf beats b1 by >= 0.06 on primary_test AND p < 0.05 =>
    primary boolean is INVALIDATED regardless of its own legs (effect is
    capacity, not structure).

Floors (informational — §12.5's boolean legs never read them):
  floor_wazn mean primary_test acc (expected near 0.5)
  floor_root mean secondary_test acc (expected near 0.5)

Usage:
  python exp77_aggregate.py           # aggregate all available checkpoints
  python exp77_aggregate.py --check   # print status only, no write
"""

import argparse
import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR     = os.path.join(_HERE, "exp77_ckpt")
RESULTS_PATH = os.path.join(_HERE, "results_exp77.json")

ARMS    = ["b1", "wazn", "wazn_shuf", "floor_wazn", "floor_root"]
SEED_BASE = 77000
ARM_OFFSET = {"b1": 0, "wazn": 1, "wazn_shuf": 2, "floor_wazn": 3, "floor_root": 4}
N_SEEDS = 5

DELTA = 0.06          # §12.1 (frozen)
N_RESAMPLES = 10000   # §12.5
PERM_SEED   = 77777   # pinned permutation seed (record in output)


def run_seed(arm, seed_idx):
    return SEED_BASE + ARM_OFFSET[arm] * 100 + seed_idx


# ---------------------------------------------------------------------------
# LOAD CHECKPOINTS
# ---------------------------------------------------------------------------

def load_checkpoints():
    """Load all available checkpoint files. Returns {arm: {seed_idx: ckpt}}."""
    results = {arm: {} for arm in ARMS}
    missing = []
    for arm in ARMS:
        for i in range(N_SEEDS):
            seed = run_seed(arm, i)
            path = os.path.join(CKPT_DIR, f"{arm}_s{seed}_results.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    results[arm][i] = json.load(fh)
            else:
                missing.append(f"{arm}_s{seed}")
    return results, missing


# ---------------------------------------------------------------------------
# PAIRED PERMUTATION TEST (§12.5, sign-flip over test items)
# ---------------------------------------------------------------------------

def paired_permutation_test(diff_per_pair, n_resamples=N_RESAMPLES, seed=PERM_SEED):
    """One-sided paired permutation: fraction of sign-flip resamples >= observed.

    diff_per_pair: list of (avg_correct_wazn - avg_correct_b1) per test pair.
    Returns (observed_mean_diff, p_value).
    """
    if not diff_per_pair:
        return float("nan"), float("nan")
    observed = sum(diff_per_pair) / len(diff_per_pair)
    rng = random.Random(seed)
    count_geq = 0
    for _ in range(n_resamples):
        perm_stat = sum(
            (d if rng.random() < 0.5 else -d) for d in diff_per_pair
        ) / len(diff_per_pair)
        if perm_stat >= observed:
            count_geq += 1
    p_value = count_geq / n_resamples
    return observed, p_value


# ---------------------------------------------------------------------------
# ALIGN PER-PAIR RECORDS ACROSS ARMS
# ---------------------------------------------------------------------------

def make_pair_key(rec):
    """Canonical pair key from a per-pair record."""
    return tuple(sorted([
        rec["word_a"] + "|" + rec["col_a"],
        rec["word_b"] + "|" + rec["col_b"],
    ]))


def align_pairs(ckpts_a, ckpts_b, split):
    """For two arms, compute seed-averaged per-pair correctness and diff.

    ckpts_a: {seed_idx: ckpt} for arm A
    ckpts_b: {seed_idx: ckpt} for arm B
    split: 'primary_test' or 'secondary_test'
    Returns (diff_per_pair, avg_acc_a, avg_acc_b, n_pairs_aligned).
    """
    # Collect all pairs from arm A (use seed 0 as the pair reference)
    seed_indices_a = sorted(ckpts_a.keys())
    seed_indices_b = sorted(ckpts_b.keys())
    if not seed_indices_a or not seed_indices_b:
        return [], float("nan"), float("nan"), 0

    # Build per-pair correct maps for each seed
    def build_pair_correct(ckpt, split):
        records = ckpt[split]["per_pair"]
        return {make_pair_key(r): r["correct"] for r in records}

    a_maps = {i: build_pair_correct(ckpts_a[i], split) for i in seed_indices_a}
    b_maps = {i: build_pair_correct(ckpts_b[i], split) for i in seed_indices_b}

    # Common pair keys
    all_keys = set(a_maps[seed_indices_a[0]].keys())
    for i in seed_indices_a[1:]:
        all_keys &= set(a_maps[i].keys())
    for i in seed_indices_b:
        all_keys &= set(b_maps[i].keys())

    if not all_keys:
        return [], float("nan"), float("nan"), 0

    diff_per_pair = []
    total_a = 0.0
    total_b = 0.0
    for key in all_keys:
        avg_a = sum(a_maps[i][key] for i in seed_indices_a) / len(seed_indices_a)
        avg_b = sum(b_maps[i][key] for i in seed_indices_b) / len(seed_indices_b)
        diff_per_pair.append(avg_a - avg_b)
        total_a += avg_a
        total_b += avg_b

    n = len(all_keys)
    return diff_per_pair, total_a / n, total_b / n, n


# ---------------------------------------------------------------------------
# LEAVE-فاعِل-OUT FILTER
# ---------------------------------------------------------------------------

def filter_leave_faail_out(ckpts, split):
    """Return ckpts with per_pair filtered to exclude pairs involving 'faail'."""
    filtered = {}
    for i, ckpt in ckpts.items():
        new_records = [
            r for r in ckpt[split]["per_pair"]
            if r["col_a"] != "faail" and r["col_b"] != "faail"
        ]
        new_acc = (sum(r["correct"] for r in new_records) / len(new_records)
                   if new_records else float("nan"))
        new_ckpt = dict(ckpt)
        new_ckpt[split] = {
            "accuracy": round(new_acc, 6),
            "n_pairs": len(new_records),
            "per_pair": new_records,
        }
        filtered[i] = new_ckpt
    return filtered


# ---------------------------------------------------------------------------
# SUMMARIZE ONE ARM
# ---------------------------------------------------------------------------

def arm_summary(ckpts, split):
    """Mean accuracy ± sd over seeds for one arm and split."""
    accs = [ckpts[i][split]["accuracy"] for i in sorted(ckpts.keys())
            if ckpts[i][split]["accuracy"] == ckpts[i][split]["accuracy"]]  # nan filter
    if not accs:
        return {"mean": float("nan"), "sd": float("nan"), "per_seed": []}
    mean = sum(accs) / len(accs)
    sd = math.sqrt(sum((x - mean) ** 2 for x in accs) / max(1, len(accs) - 1))
    return {"mean": round(mean, 6), "sd": round(sd, 6), "per_seed": [round(a, 6) for a in accs]}


# ---------------------------------------------------------------------------
# TWO-LEG BOOLEAN
# ---------------------------------------------------------------------------

def two_leg_boolean(wazn_ckpts, b1_ckpts, split, delta=DELTA):
    """Compute both legs for the §12.5 two-leg boolean.

    Returns dict with: mean_wazn, mean_b1, delta, leg1_pass, leg1_margin,
    leg2_p_value, leg2_pass, boolean_value.
    """
    diff, avg_wazn, avg_b1, n_aligned = align_pairs(wazn_ckpts, b1_ckpts, split)
    observed_diff, p_val = paired_permutation_test(diff)

    leg1_margin = avg_wazn - avg_b1
    leg1_pass   = (leg1_margin >= delta) and not math.isnan(leg1_margin)
    leg2_pass   = (p_val < 0.05) and not math.isnan(p_val)
    boolean_val = leg1_pass and leg2_pass

    return {
        "mean_wazn":      round(avg_wazn, 6),
        "mean_b1":        round(avg_b1, 6),
        "delta":          delta,
        "n_pairs_aligned": n_aligned,
        "leg1_margin":    round(leg1_margin, 6) if not math.isnan(leg1_margin) else None,
        "leg1_pass":      bool(leg1_pass),
        "leg2_observed_diff": round(observed_diff, 6) if not math.isnan(observed_diff) else None,
        "leg2_p_value":   round(p_val, 6) if not math.isnan(p_val) else None,
        "leg2_pass":      bool(leg2_pass),
        "permutation_n_resamples": N_RESAMPLES,
        "permutation_seed": PERM_SEED,
        "boolean_value":  bool(boolean_val),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Print checkpoint status only; do not write")
    a = ap.parse_args()

    print("[aggregate] loading checkpoints ...", flush=True)
    ckpts, missing = load_checkpoints()

    total = len(ARMS) * N_SEEDS
    found = total - len(missing)
    print(f"  found {found}/{total} checkpoints", flush=True)
    if missing:
        print(f"  missing: {missing}", flush=True)

    if a.check:
        print("[aggregate] --check mode: status printed, no write.", flush=True)
        return 0

    if missing:
        print(f"[aggregate] WARNING: {len(missing)} checkpoints missing. "
              f"Proceeding with partial results (will compute over available seeds).",
              flush=True)

    # -- Summaries per arm --
    summaries = {}
    for arm in ARMS:
        if not ckpts[arm]:
            continue
        summaries[arm] = {
            "primary_test":   arm_summary(ckpts[arm], "primary_test"),
            "secondary_test": arm_summary(ckpts[arm], "secondary_test"),
            "n_seeds_available": len(ckpts[arm]),
        }

    # -- Primary boolean: transfers_across_roots_within_wazn --
    print("[aggregate] primary boolean (wazn vs b1, primary_test) ...", flush=True)
    primary_bool = two_leg_boolean(ckpts["wazn"], ckpts["b1"], "primary_test")
    print(f"  mean_wazn={primary_bool['mean_wazn']:.4f}  "
          f"mean_b1={primary_bool['mean_b1']:.4f}  "
          f"margin={primary_bool['leg1_margin']}  "
          f"p={primary_bool['leg2_p_value']}", flush=True)
    print(f"  Leg1(margin>=0.06): {primary_bool['leg1_pass']}  "
          f"Leg2(p<0.05): {primary_bool['leg2_pass']}  "
          f"=> transfers_across_roots_within_wazn = {primary_bool['boolean_value']}",
          flush=True)

    # -- Secondary boolean: transfers_across_wazn_within_root --
    print("[aggregate] secondary boolean (wazn vs b1, secondary_test) ...", flush=True)
    secondary_bool = two_leg_boolean(ckpts["wazn"], ckpts["b1"], "secondary_test")
    print(f"  mean_wazn={secondary_bool['mean_wazn']:.4f}  "
          f"mean_b1={secondary_bool['mean_b1']:.4f}  "
          f"margin={secondary_bool['leg1_margin']}  "
          f"p={secondary_bool['leg2_p_value']}", flush=True)
    print(f"  Leg1(margin>=0.06): {secondary_bool['leg1_pass']}  "
          f"Leg2(p<0.05): {secondary_bool['leg2_pass']}  "
          f"=> transfers_across_wazn_within_root = {secondary_bool['boolean_value']}",
          flush=True)

    # -- structure_not_semantics: leave-فاعِل-out re-read --
    print("[aggregate] structure_not_semantics (leave-faail-out re-read) ...", flush=True)
    wazn_no_faail = filter_leave_faail_out(ckpts["wazn"], "primary_test")
    b1_no_faail   = filter_leave_faail_out(ckpts["b1"],   "primary_test")
    n_after_filter = (
        wazn_no_faail[list(wazn_no_faail.keys())[0]]["primary_test"]["n_pairs"]
        if wazn_no_faail else 0
    )
    sns_bool = two_leg_boolean(wazn_no_faail, b1_no_faail, "primary_test")
    print(f"  n_pairs after removing faail: {n_after_filter}", flush=True)
    print(f"  mean_wazn={sns_bool['mean_wazn']:.4f}  "
          f"mean_b1={sns_bool['mean_b1']:.4f}  "
          f"margin={sns_bool['leg1_margin']}  "
          f"p={sns_bool['leg2_p_value']}", flush=True)
    print(f"  => structure_not_semantics = {sns_bool['boolean_value']}", flush=True)

    # -- Structure-specificity control: wazn_shuf must NOT beat B1 --
    print("[aggregate] structure-specificity control (wazn_shuf vs b1) ...", flush=True)
    shuf_bool = two_leg_boolean(ckpts["wazn_shuf"], ckpts["b1"], "primary_test")
    shuf_beats_b1 = bool(shuf_bool["boolean_value"])
    invalidated = shuf_beats_b1
    print(f"  wazn_shuf mean={shuf_bool['mean_wazn']:.4f}  b1 mean={shuf_bool['mean_b1']:.4f}  "
          f"margin={shuf_bool['leg1_margin']}  p={shuf_bool['leg2_p_value']}", flush=True)
    print(f"  wazn_shuf beats b1: {shuf_beats_b1}", flush=True)
    if invalidated:
        print("  FLAG INVALIDATION: wazn_shuf beats b1 by >=0.06 AND p<0.05. "
              "Effect is CAPACITY, not structure. Primary boolean is INVALIDATED "
              "regardless of its own legs (§12.5).", flush=True)
    else:
        print("  Control holds: wazn_shuf does not beat b1. "
              "Primary boolean is NOT invalidated by capacity.", flush=True)

    # -- Floor summaries (informational) --
    floor_wazn_primary   = summaries.get("floor_wazn", {}).get("primary_test", {})
    floor_root_secondary = summaries.get("floor_root", {}).get("secondary_test", {})

    # -- Assemble final verdict --
    primary_transfers = primary_bool["boolean_value"] and not invalidated
    secondary_transfers = secondary_bool["boolean_value"]
    # structure_not_semantics requires primary to be non-invalidated too
    sns_final = sns_bool["boolean_value"] and not invalidated

    print("\n" + "="*60, flush=True)
    print("[aggregate] VERDICTS", flush=True)
    print("="*60, flush=True)
    print(f"  transfers_across_roots_within_wazn : {primary_transfers}", flush=True)
    print(f"    (raw two-leg: {primary_bool['boolean_value']}, "
          f"invalidated_by_capacity: {invalidated})", flush=True)
    print(f"  transfers_across_wazn_within_root  : {secondary_transfers}", flush=True)
    print(f"  structure_not_semantics            : {sns_final}", flush=True)
    print(f"  structure_specificity_control      : "
          f"{'FAILED (invalidation)' if invalidated else 'HOLDS'}", flush=True)
    print("="*60 + "\n", flush=True)

    # -- Write results --
    output = {
        "experiment": "exp77",
        "spec":       "doc 66 §12.5, REGISTERED 2026-09-04",
        "delta":      DELTA,
        "n_resamples": N_RESAMPLES,
        "permutation_seed": PERM_SEED,
        "n_checkpoints_found": found,
        "n_checkpoints_total": total,
        "missing_checkpoints":  missing,
        "arm_summaries": summaries,
        "booleans": {
            "transfers_across_roots_within_wazn": {
                "value": bool(primary_transfers),
                "two_leg": primary_bool,
                "invalidated_by_capacity_control": bool(invalidated),
            },
            "transfers_across_wazn_within_root": {
                "value": bool(secondary_transfers),
                "two_leg": secondary_bool,
            },
            "structure_not_semantics": {
                "value": bool(sns_final),
                "two_leg_leave_faail_out": sns_bool,
                "n_pairs_after_filter": n_after_filter,
                "note": (
                    "Leave-فاعِل-out re-read: primary_test pairs with col_a or col_b == 'faail' "
                    "removed. Tests whether effect survives semantically-loaded template exclusion."
                ),
            },
            "structure_specificity_control": {
                "wazn_shuf_beats_b1": bool(shuf_beats_b1),
                "two_leg_wazn_shuf_vs_b1": shuf_bool,
                "invalidation_triggered": bool(invalidated),
                "note": (
                    "§12.5 invalidation clause: if WAZN-SHUFFLED (permuted col→topology) "
                    "beats B1 by >=Δ AND p<0.05, the effect is capacity not structure, "
                    "and primary boolean is invalidated regardless of its own legs."
                ),
            },
        },
        "floors": {
            "floor_wazn_primary": {
                "mean_acc": floor_wazn_primary.get("mean"),
                "sd":       floor_wazn_primary.get("sd"),
                "per_seed": floor_wazn_primary.get("per_seed", []),
                "note": (
                    "WAZN trained on within-wazn (col-pair stratum) label shuffle. "
                    "Evaluated on real labels. Expected near 0.5 if wazn structure carries signal. "
                    "Informational — §12.5 boolean legs never read floor values."
                ),
            },
            "floor_root_secondary": {
                "mean_acc": floor_root_secondary.get("mean"),
                "sd":       floor_root_secondary.get("sd"),
                "per_seed": floor_root_secondary.get("per_seed", []),
                "note": (
                    "WAZN trained on within-root label shuffle. "
                    "Evaluated on real secondary_test labels. Expected near 0.5. Informational."
                ),
            },
        },
        "interpretation": {
            "both_true":     "Wazn axis converts to architecture — both transfer directions confirmed.",
            "primary_only":  "Root tier carries structure; melody direction doesn't transfer (or vice versa).",
            "secondary_only":"Wazn direction transfers within root but not across roots.",
            "both_false":    "Wazn axis does NOT convert to architecture on this task — E6 inversion is answered negatively.",
            "note": (
                "A tie with B2 (classical baseline, not an arm here) refutes nothing — "
                "H-T is decided against B1; B2 is the 'how far does learning get you' line (§2)."
            ),
        },
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=1)
    print(f"[aggregate] wrote {RESULTS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
