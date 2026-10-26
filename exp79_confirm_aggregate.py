# -*- coding: utf-8 -*-
"""exp79_confirm_aggregate.py -- SOLE WRITER of results_exp79_confirm.json.

Reads per-unit checkpoints from exp79_confirm_ckpt/ and C1 shards from
exp79_confirm_ctl/; computes the confirm verdict; writes results_exp79_confirm.json.

Bars (verdict on CONFIRM DRAW ALONE, per doc 71 §10.2 + §3 + §6):
  tier_q_channel_confirmed:
      (C-DISCARD beats B1, seed-level paired t, df=4, one-sided p<0.05)
      AND (content_earned: C-DISCARD C1 dial-swap exceeds 95th percentile)
  A1_at_chance:
      A1 seed-mean union accuracy within 2 binomial SD of 0.5.
      A1 ABOVE chance INVALIDATES the draw (doc 22 §3.2 / doc 70 §3.5).
  confirm_verdict:
      tier_q_channel_confirmed AND A1_at_chance.

Doc-70 compliance:
  -- primary test is seed-level paired t (df=4, one-sided)
  -- item-level sign-flip is descriptive only
  -- no absolute 0.65 bar (differential bars only: vs B1, vs A2)
  -- A1-at-chance: 2 binomial SD criterion

THIS FILE WRITES NO CHECKPOINT. Does not retrain anything.
"""

import json
import math
import os
import sys
import time

CONF_CKPT = "exp79_confirm_ckpt"
CONF_CTL  = "exp79_confirm_ctl"
OUT       = "results_exp79_confirm.json"
N_SEEDS   = 5

# arm-to-filename prefix (must match exp79_run.py's unit_paths79)
ARM_PREFIX = {
    "C-DISCARD": "CDISCARD",
    "A2": "A2",
    "B1": "B1",
    "A1": "A1",
}

BAR_PROVENANCE = {
    "chance_level": 0.5,
    "bars_are_differential": "vs B1 and vs A2; no 0.65 bar (doc 70 B1 correction)",
    "significance_primary": "seed-level paired t, df=4, one-sided p<0.05",
    "significance_descriptive": "item-level sign-flip permutation (reported, not gating)",
    "a1_gate": "A1 seed-mean within 2 binomial SD of 0.5; above-chance A1 "
               "invalidates the draw (doc 22 §3.2 / doc 70 §3.5)",
}


# ================================================================
# Loaders
# ================================================================
def load_unit(arm, seed, ckpt_dir=CONF_CKPT):
    prefix = ARM_PREFIX.get(arm, arm.replace("-", ""))
    path = os.path.join(ckpt_dir, "%s_s%d.json" % (prefix, seed))
    if not os.path.exists(path):
        return None, path
    with open(path, encoding="utf-8") as f:
        return json.load(f), path


def load_arm(arm, seeds=N_SEEDS, ckpt_dir=CONF_CKPT):
    units, missing = [], []
    for s in range(seeds):
        det, path = load_unit(arm, s, ckpt_dir)
        if det is None:
            missing.append({"seed": s, "path": path})
        else:
            units.append((s, det))
    return units, missing


def union_accs(units):
    return [d["test"]["test_union"]["accuracy"] for _s, d in units]


def union_correct_vecs(units):
    return [d.get("union_per_item_correct", []) for _s, d in units]


def seed_mean(units):
    accs = union_accs(units)
    return sum(accs) / len(accs) if accs else 0.0


# ================================================================
# Statistics helpers (mirror exp79_aggregate.py)
# ================================================================
def t_test_one_sided(xs, ys):
    """Paired one-sided t-test: H_a: mean(xs) > mean(ys). df = n-1."""
    n = len(xs)
    if n < 2:
        return {"t": None, "p": None, "n": n, "error": "n<2"}
    diffs = [x - y for x, y in zip(xs, ys)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var_d / n)
    if se == 0:
        t = float("inf") if mean_d > 0 else float("-inf")
        return {"t": t, "p": 0.0 if mean_d > 0 else 1.0, "n": n,
                "mean_diff": mean_d, "se": se}
    t = mean_d / se
    try:
        from scipy import stats as _st
        p = float(_st.t.sf(t, df=n - 1))
    except ImportError:
        p = float(0.5 * math.erfc(t / math.sqrt(2)))
    return {"t": round(t, 4), "p": round(p, 5), "n": n,
            "mean_diff": round(mean_d, 5), "se": round(se, 5), "df": n - 1}


def item_sign_flip(xs_per_seed, ys_per_seed):
    """Descriptive item-level sign-flip (doc 70 B2: NOT the primary test)."""
    if not xs_per_seed or not ys_per_seed:
        return {"p": None, "n_items": 0, "note": "no data"}
    n_items = len(xs_per_seed[0])
    if n_items == 0:
        return {"p": None, "n_items": 0, "note": "empty vectors"}
    mean_x = [sum(xs_per_seed[s][i] for s in range(len(xs_per_seed)))
               / len(xs_per_seed) for i in range(n_items)]
    mean_y = [sum(ys_per_seed[s][i] for s in range(len(ys_per_seed)))
               / len(ys_per_seed) for i in range(n_items)]
    obs = sum(mean_x[i] - mean_y[i] for i in range(n_items)) / n_items
    rng = __import__("random").Random(9901)
    n_exc = 0
    N_PERM = 1000
    for _ in range(N_PERM):
        pd = sum(rng.choice([1, -1]) * (mean_x[i] - mean_y[i])
                 for i in range(n_items)) / n_items
        if pd >= obs:
            n_exc += 1
    p = (n_exc + 1) / (N_PERM + 1)
    return {"p": round(p, 4), "observed_delta": round(obs, 5),
            "n_items": n_items, "n_perm": N_PERM,
            "note": "DESCRIPTIVE ONLY (doc 70 B2): primary is seed-level t"}


def binomial_sd(n, p=0.5):
    return math.sqrt(p * (1 - p) / n)


def at_chance(accs, n_items):
    """Seed-mean union acc within 2 binomial SD of 0.5."""
    if not accs:
        return {"at_chance": None, "note": "no units"}
    mean_acc = sum(accs) / len(accs)
    sd = binomial_sd(n_items)
    within = abs(mean_acc - 0.5) <= 2 * sd
    return {
        "seed_mean_acc": round(mean_acc, 4),
        "n_items": n_items,
        "binomial_sd": round(sd, 5),
        "deviation_from_chance": round(abs(mean_acc - 0.5), 5),
        "within_2sd": bool(within),
        "at_chance": bool(within),
        "above_chance": not within and mean_acc > 0.5,
        "invalidates_draw": not within and mean_acc > 0.5,
    }


# ================================================================
# C1 content_earned (from confirm c1_shard files)
# ================================================================
def load_c1_shards_confirm(n_shards=5, ctl_dir=CONF_CTL):
    shards, missing = [], []
    for k in range(n_shards):
        path = os.path.join(ctl_dir, "c1_confirm_shard%d.json" % k)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                shards.append(json.load(f))
        else:
            missing.append(path)
    return shards, missing


def compute_c1_content_earned(cdiscard_accs, shards):
    """content_earned: C-DISCARD seed-mean union acc exceeds C1 95th percentile."""
    if not shards:
        return {"content_earned": None,
                "note": "C1 confirm shards not found; cannot compute"}
    all_perm_accs = []
    for sh in shards:
        all_perm_accs.extend(sh.get("permuted_seed_mean_accs", []))
    if not all_perm_accs:
        return {"content_earned": None, "note": "no permuted accs in shards"}
    all_perm_accs.sort()
    pct95 = all_perm_accs[int(0.95 * len(all_perm_accs))]
    cdiscard_mean = sum(cdiscard_accs) / len(cdiscard_accs) if cdiscard_accs else 0.0
    earned = cdiscard_mean > pct95
    return {
        "content_earned": bool(earned),
        "cdiscard_seed_mean_union_acc": round(cdiscard_mean, 5),
        "c1_95th_percentile": round(pct95, 5),
        "n_permutations": len(all_perm_accs),
        "note": "C1 dial-swap on C-DISCARD with confirm data; 1000 perms seed 4242",
    }


# ================================================================
# main
# ================================================================
def aggregate():
    # ---- load units ------------------------------------------------
    cd_units, cd_miss = load_arm("C-DISCARD")
    a2_units, a2_miss = load_arm("A2")
    b1_units, b1_miss = load_arm("B1")
    a1_units, a1_miss = load_arm("A1")

    missing = {}
    for arm, miss in [("C-DISCARD", cd_miss), ("A2", a2_miss),
                      ("B1", b1_miss), ("A1", a1_miss)]:
        if miss:
            missing[arm] = miss

    if missing:
        print("[conf_agg] WARNING: missing units: %s" % missing, flush=True)

    # ---- per-seed union accuracies ---------------------------------
    cd_accs = union_accs(cd_units)
    a2_accs = union_accs(a2_units)
    b1_accs = union_accs(b1_units)
    a1_accs = union_accs(a1_units)

    # ---- primary significance (seed-level paired t) ----------------
    t_cd_vs_b1 = t_test_one_sided(cd_accs, b1_accs)
    t_cd_vs_a2 = t_test_one_sided(cd_accs, a2_accs)

    # ---- descriptive sign-flip (NOT primary) -----------------------
    sf_cd_vs_b1 = item_sign_flip(union_correct_vecs(cd_units),
                                  union_correct_vecs(b1_units))
    sf_cd_vs_a2 = item_sign_flip(union_correct_vecs(cd_units),
                                  union_correct_vecs(a2_units))

    # ---- C1 content_earned -----------------------------------------
    c1_shards, c1_miss = load_c1_shards_confirm()
    if c1_miss:
        print("[conf_agg] C1 shards missing: %s" % c1_miss, flush=True)
    c1_result = compute_c1_content_earned(cd_accs, c1_shards)

    # ---- A1-at-chance gate (invalidating) --------------------------
    n_items = (len(a1_units[0][1].get("union_per_item_correct", []))
               if a1_units else
               len(cd_units[0][1].get("union_per_item_correct", []))
               if cd_units else 500)
    a1_gate = at_chance(a1_accs, n_items)

    # ---- bars ------------------------------------------------------
    P_THRESHOLD = 0.05
    t_b1_pass = (t_cd_vs_b1.get("p") is not None
                 and t_cd_vs_b1["p"] < P_THRESHOLD
                 and t_cd_vs_b1.get("mean_diff", 0) > 0)
    c1_earned = bool(c1_result.get("content_earned"))

    tier_q_channel_confirmed = bool(t_b1_pass and c1_earned)
    a1_at_chance = bool(a1_gate.get("at_chance"))
    confirm_verdict = bool(tier_q_channel_confirmed and a1_at_chance)

    # ---- nonfinite_skips per seed (from C-DISCARD checkpoints) -----
    cd_nonfinite = {
        s: d.get("nonfinite_skips_doc71_SS9_3", [])
        for s, d in cd_units
    }

    # ---- assemble --------------------------------------------------
    results = {
        "component": "results_exp79_confirm -- confirm draw verdict (doc 71 §10.2)",
        "written_by": "exp79_confirm_aggregate.py (sole writer)",
        "registration": (
            "qnlp_private_docs/71_exp79_channel_composition.md §10.2: "
            "generator seed 43000, confirm draw ordered 2026-09-09"),
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

        "bar_provenance": BAR_PROVENANCE,
        "missing_units": missing,
        "c1_shards_missing": c1_miss,

        "seed_union_accs": {
            "C-DISCARD": [round(a, 5) for a in cd_accs],
            "A2":        [round(a, 5) for a in a2_accs],
            "B1":        [round(a, 5) for a in b1_accs],
            "A1":        [round(a, 5) for a in a1_accs],
        },
        "seed_means": {
            "C-DISCARD": round(seed_mean(cd_units), 5),
            "A2":        round(seed_mean(a2_units), 5),
            "B1":        round(seed_mean(b1_units), 5),
            "A1":        round(seed_mean(a1_units), 5),
        },

        "significance": {
            "primary_test": "seed-level paired t (doc 70 B2, df=4, one-sided)",
            "CDISCARD_vs_B1": t_cd_vs_b1,
            "CDISCARD_vs_A2": t_cd_vs_a2,
            "descriptive_sign_flip_CDISCARD_vs_B1": sf_cd_vs_b1,
            "descriptive_sign_flip_CDISCARD_vs_A2": sf_cd_vs_a2,
        },

        "c1_content_earned": c1_result,

        "a1_gate": {
            **a1_gate,
            "rule": ("doc 22 §3.2 / doc 70 §3.5: A1 above chance (seed-mean "
                     "union acc outside 2 binomial SD of 0.5) INVALIDATES "
                     "this confirm draw."),
        },

        "bars": {
            "tier_q_channel_confirmed": {
                "value": tier_q_channel_confirmed,
                "legs": {
                    "t_CDISCARD_vs_B1_pass": t_b1_pass,
                    "content_earned_c1": c1_earned,
                },
                "definition": (
                    "C-DISCARD beats B1 (seed t p<0.05) AND C1 content_earned "
                    "(C-DISCARD seed-mean union acc > C1 95th pct permutation)"),
            },
            "a1_at_chance": {
                "value": a1_at_chance,
                "definition": ("A1 seed-mean union acc within 2 binomial SD of 0.5; "
                               "failing this invalidates the draw"),
            },
            "confirm_verdict": {
                "value": confirm_verdict,
                "definition": ("tier_q_channel_confirmed AND a1_at_chance; "
                               "TRUE = first-draw positive on CONFIRM DRAW"),
            },
        },

        "c_discard_nonfinite_skips": {
            "per_seed": {str(s): len(v) for s, v in cd_nonfinite.items()},
            "note": ("doc 71 §9.3: non-finite grad skip count per seed. "
                     "Empty list = clean run."),
        },

        "per_arm_seed_detail": {
            arm: {
                "n_units": len(units),
                "seed_union_accs": [round(d["test"]["test_union"]["accuracy"], 5)
                                    for _s, d in units],
                "seed_val_accs":   [round(d["val_acc"], 5)
                                    for _s, d in units],
                "wall_times_sec":  [d.get("wall_time_sec") for _s, d in units],
            }
            for arm, units in [("C-DISCARD", cd_units), ("A2", a2_units),
                                ("B1", b1_units), ("A1", a1_units)]
        },
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # ---- summary print -------------------------------------------
    print("\n=== exp79 CONFIRM AGGREGATE ===")
    for arm, accs in [("C-DISCARD", cd_accs), ("A2", a2_accs),
                       ("B1", b1_accs), ("A1", a1_accs)]:
        m = sum(accs) / len(accs) if accs else float("nan")
        print("  %s: union_mean=%.5f  seeds=%s" % (
            arm, m, [round(a, 4) for a in accs]))
    print()
    print("C-DISCARD vs B1: t=%.4f  p=%.5f  mean_diff=%.5f" % (
        t_cd_vs_b1.get("t", float("nan")),
        t_cd_vs_b1.get("p", float("nan")),
        t_cd_vs_b1.get("mean_diff", float("nan"))))
    print("C-DISCARD vs A2: t=%.4f  p=%.5f  mean_diff=%.5f" % (
        t_cd_vs_a2.get("t", float("nan")),
        t_cd_vs_a2.get("p", float("nan")),
        t_cd_vs_a2.get("mean_diff", float("nan"))))
    print("C1 content_earned: %s  (cdiscard_mean=%.5f  95th_pct=%.5f)" % (
        c1_earned,
        c1_result.get("cdiscard_seed_mean_union_acc", float("nan")),
        c1_result.get("c1_95th_percentile", float("nan"))))
    print("A1 at_chance: %s  (seed_mean=%.4f  within_2sd=%s)" % (
        a1_at_chance,
        a1_gate.get("seed_mean_acc", float("nan")),
        a1_gate.get("within_2sd")))
    print()
    print("tier_q_channel_confirmed: %s" % tier_q_channel_confirmed)
    print("a1_at_chance:             %s" % a1_at_chance)
    print("CONFIRM_VERDICT:          %s" % confirm_verdict)
    print()
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(aggregate())
