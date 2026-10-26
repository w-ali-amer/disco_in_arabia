# -*- coding: utf-8 -*-
"""exp81_aggregate.py -- SOLE WRITER of results_exp81.json.

Registration: qnlp_private_docs/73_exp81_width_per_referent.md

THIS FILE WRITES NO CHECKPOINT. It reads per-unit checkpoints from
exp81_ckpt/ (A2q2, A2-WIDE, A2-RETRAIN -- written by exp81_run.py workers)
and exp42r_ckpt/ (A2 committed drift anchor, B1 committed read-only),
computes all bars and gates, and writes results_exp81.json.

Bars (doc 73 §3):
  `width_per_referent_helps`
      A2q2 beats A2-RETRAIN (seed-level paired t, df=4, one-sided p<0.05)
      AND margin >= 0.06.  TWO-LEG: both required.
      Primary paired test = A2q2 vs A2-RETRAIN (same-harness pairing for
      unbiased comparison; committed A2 is the drift anchor only).

  `tier_q_q2`
      Same two-leg form: A2q2 vs B1.
      B1 is the committed structure-matched classical baseline (read-only;
      5 seeds from exp42r_ckpt/).

  A2-WIDE invalidation clause:
      If A2-WIDE also beats A2-RETRAIN (p<0.05, margin>=0.06): the
      improvement is CAPACITY not architecture (WAZN-SHUFFLED pattern).
      `width_per_referent_helps` CANNOT fire; the bar is invalidated even
      if both legs otherwise pass.

  `gate_met` (trainability):
      If `width_per_referent_helps` is False everywhere (A2q2 never beats A2
      even directionally): the wall extends to local width.  Third outcome.

Third outcomes named (doc 73 §3):
  sub-bar positive direction: A2q2 beats A2-RETRAIN (p<0.05) but margin < 0.06
    → recorded, feeds doc 74-style closure later.
  trainability failure: gate_met false everywhere → wall extends to local width.

Owes-confirm clause:
  Any fired positive (including sub-bar positive) owes a seed-44000 confirm
  (same-draw arms A2q2 + A2 + B1 + A1).  This file records the boolean
  `owes_confirm`; the confirm is NOT run here.
  Near-miss symmetry: a negative result missing the bar by < 1 test item
  also fires owes_confirm (doc 70 rule 2).

Doc-70 compliance:
  -- seed-level paired t PRIMARY (df=4, one-sided)
  -- item-level sign-flip DESCRIPTIVE only
  -- floor-admissibility gate (reads from SG-FS artifact)
  -- near-miss symmetry (< 1 item triggers owes_confirm)
  -- MDE / power at bar
  -- bar provenance block

DOES NOT RUN TRAINING. Sole writer of results_exp81.json.
"""

import json
import math
import os
import sys

# ================================================================
# paths
# ================================================================
CKPT_DIR = os.environ.get("EXP81_CKPT_DIR", "exp81_ckpt")
CTL_DIR = os.environ.get("EXP81_CTL_DIR", "exp81_ctl")
EXP42R_CKPT_DIR = "exp42r_ckpt"
OUT = "results_exp81.json"
N_SEEDS = 5

SMOKE_FILE = "exp81_smoke_PASSED.json"
TIMING_FILE = "exp81_epoch_timing.json"
SG_ZC_FILE = "exp81_gate_SG_ZC.json"
SG_FS_FILE = "exp81_gate_SG_FS.json"
SG_PC_FILE = "exp81_gate_SG_PC.json"

MARGIN_BAR = 0.06      # doc 73 §3: margin >= 0.06
P_BAR = 0.05           # one-sided t, df=4
N_TEST_UNION = 1500    # test_union size (from exp42r splits)

BAR_PROVENANCE = {
    "chance_level": 0.5,
    "reason": "balanced yes/no task (exp41r)",
    "ceiling_comparator": "B1 (structure-matched classical baseline, ~0.50 union acc)",
    "bars_are_differential": "vs A2-RETRAIN and vs B1; NO absolute threshold",
    "significance_primary": "seed-level paired t, df=4, one-sided p<0.05 "
                            "(doc 70 B2 correction)",
    "significance_descriptive": "item-level sign-flip permutation (reported, "
                                "not decision-making)",
    "margin_leg": "mean(A2q2 seeds) - mean(baseline seeds) >= 0.06",
    "doc_70_b1_b2_b3": True,
    "doc_73_two_leg_form": True,
}


# ================================================================
# load checkpoints
# ================================================================
def arm_path(arm, seed, ckpt_dir=CKPT_DIR):
    tag = arm.replace("-", "").replace(" ", "")
    return os.path.join(ckpt_dir, "%s_s%d.json" % (tag, seed))


def load_arm_seeds(arm, seeds=N_SEEDS, ckpt_dir=CKPT_DIR):
    """Load all seeds for an arm. Returns list of (seed, detail) or None."""
    units = []
    for s in range(seeds):
        path = arm_path(arm, s, ckpt_dir)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                units.append((s, json.load(f)))
        else:
            units.append((s, None))
    return units


def load_committed_arm(arm, seeds=N_SEEDS):
    """Load committed A2 or B1 checkpoints from exp42r_ckpt/ (read-only)."""
    units = []
    for s in range(seeds):
        path = os.path.join(EXP42R_CKPT_DIR, "%s_s%d.json" % (arm, s))
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                units.append((s, json.load(f)))
        else:
            units.append((s, None))
    return units


def extract_union_accs(units):
    """Extract test_union accuracies from loaded units. Returns list of floats."""
    accs = []
    for (_s, detail) in units:
        if detail is None:
            accs.append(None)
        else:
            tu = detail.get("test", {}).get("test_union", {})
            accs.append(tu.get("accuracy"))
    return accs


def extract_per_item_correct(units, split="test_union"):
    """Extract per-item correct vectors (list of lists, one per seed)."""
    vecs = []
    for (_s, detail) in units:
        if detail is None:
            vecs.append(None)
        else:
            tu = detail.get("test", {}).get(split, {})
            c = tu.get("per_item_correct")
            vecs.append(c)
    return vecs


# ================================================================
# statistics helpers (verbatim from exp79_aggregate to ensure consistency)
# ================================================================
def t_test_one_sided(xs, ys):
    """Paired one-sided t-test: H_a: mean(xs) > mean(ys). df=n-1."""
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
                "mean_diff": mean_d, "se": se, "df": n - 1}
    t = mean_d / se
    try:
        from scipy import stats as _st
        p = float(_st.t.sf(t, df=n - 1))
    except ImportError:
        p = float(0.5 * math.erfc(t / math.sqrt(2)))
    return {"t": round(t, 4), "p": round(p, 5), "n": n,
            "mean_diff": round(mean_d, 4), "se": round(se, 4), "df": n - 1}


def item_sign_flip(xs_per_seed, ys_per_seed):
    """Descriptive item-level sign-flip (doc 70 B2: NOT the primary test)."""
    valid_xs = [v for v in xs_per_seed if v is not None]
    valid_ys = [v for v in ys_per_seed if v is not None]
    if not valid_xs or not valid_ys:
        return {"p": None, "n_items": 0,
                "note": "no data; DESCRIPTIVE ONLY (doc 70 B2)"}
    n_items = min(len(valid_xs[0]), len(valid_ys[0]))
    mean_x = [sum(v[i] for v in valid_xs) / len(valid_xs) for i in range(n_items)]
    mean_y = [sum(v[i] for v in valid_ys) / len(valid_ys) for i in range(n_items)]
    observed_delta = sum(mean_x[i] - mean_y[i] for i in range(n_items)) / n_items
    rng = __import__("random").Random(9981)
    n_exceeds = 0
    N_PERM = 1000
    for _ in range(N_PERM):
        perm = sum(rng.choice([1, -1]) * (mean_x[i] - mean_y[i])
                   for i in range(n_items)) / n_items
        if perm >= observed_delta:
            n_exceeds += 1
    p = (n_exceeds + 1) / (N_PERM + 1)
    return {"p": round(p, 4), "observed_delta": round(observed_delta, 4),
            "n_items": n_items, "n_perm": N_PERM,
            "note": "DESCRIPTIVE ONLY (doc 70 B2): primary is seed-level t"}


def floor_admissibility(acc, n):
    """Doc 70 B3: floor within 2 binomial SD of 0.5 -> admissible."""
    sd = math.sqrt(0.5 * 0.5 / n)
    return {"accuracy": acc, "n": n, "sd": round(sd, 4),
            "deviation_from_chance": round(abs(acc - 0.5), 4),
            "within_2sd": bool(abs(acc - 0.5) <= 2 * sd),
            "admissible": bool(abs(acc - 0.5) <= 2 * sd)}


def mde_against_bar(seed_accs_x, seed_accs_y):
    """MDE: minimum detectable effect against paired null SD, with power note."""
    valid = [(x, y) for x, y in zip(seed_accs_x, seed_accs_y)
             if x is not None and y is not None]
    n = len(valid)
    if n < 2:
        return {"error": "n<2"}
    xs, ys = zip(*valid)
    diffs = [x - y for x, y in zip(xs, ys)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var_d / n)
    return {"n_seeds": n, "paired_null_se": round(se, 5),
            "observed_mean_diff": round(mean_d, 5),
            "margin_bar": MARGIN_BAR,
            "mde_note": (
                "Paired null SE = %.4f; mean diff = %.4f; bar margin = %.2f. "
                "To achieve p<0.05 (one-sided, df=%d) requires effect >= ~%.4f "
                "(1.645 × SE). Power at bar (%.2f) with SE %.4f: approx %.0f%%."
                % (se, mean_d, MARGIN_BAR, n - 1, 1.645 * se,
                   MARGIN_BAR, se,
                   100 * max(0, min(1, 1 - 0.5 * math.erfc(
                       (MARGIN_BAR / se - 1.645) / math.sqrt(2))))))}


def near_miss(val, bar, n_test_items):
    """Doc 70 rule 2: negative missing bar by < 1 test item -> owes confirm."""
    acc_diff = bar - val
    if acc_diff <= 0:
        return False  # positive
    return (acc_diff * n_test_items) < 1.0


# ================================================================
# bar computation
# ================================================================
def compute_bar(arm_accs, baseline_accs, arm_name, baseline_name,
                n_test_union, per_item_x=None, per_item_y=None):
    """Two-leg bar: paired t AND margin >= 0.06.

    Returns bar dict. Does NOT apply invalidation clause (caller does that).
    """
    valid_pairs = [(x, y) for x, y in zip(arm_accs, baseline_accs)
                   if x is not None and y is not None]
    if not valid_pairs:
        return {"met": False, "error": "no valid seed pairs",
                "leg_t": None, "leg_margin": None}

    xs, ys = zip(*valid_pairs)
    n_pairs = len(xs)

    # Primary: seed-level paired t
    t_result = t_test_one_sided(list(xs), list(ys))
    p_val = t_result.get("p")
    t_ok = (p_val is not None and p_val < P_BAR)

    # Second leg: margin
    mean_x = sum(xs) / n_pairs
    mean_y = sum(ys) / n_pairs
    margin = mean_x - mean_y
    margin_ok = margin >= MARGIN_BAR

    met = bool(t_ok and margin_ok)

    # Near-miss check on each leg
    owes_nm = False
    if not met:
        # Margin near-miss: actual margin within 1/n_test items of bar (doc 70 rule 2)
        margin_shortfall = MARGIN_BAR - margin
        if margin_shortfall > 0 and (margin_shortfall * n_test_union) < 1.0:
            owes_nm = True
        # NOTE: bare t>0 clause removed -- doc 70 rule 2 is miss-by-<1-item,
        # not "any directional signal".  margin near-miss + sub_bar_positive cover
        # the registered cases.

    # Descriptive sign-flip
    sf = None
    if per_item_x is not None and per_item_y is not None:
        sf = item_sign_flip(per_item_x, per_item_y)

    # MDE/power
    mde = mde_against_bar(list(arm_accs), list(baseline_accs))

    # Sub-bar positive: directional but below margin or p threshold
    sub_bar_positive = (not met and t_result.get("t", 0) > 0
                        and p_val is not None and p_val < P_BAR)

    return {
        "met": met,
        "definition": ("%s beats %s: paired t p<%.2f (one-sided, df=%d) "
                       "AND margin >= %.2f" % (arm_name, baseline_name,
                                               P_BAR, n_pairs - 1, MARGIN_BAR)),
        "leg_t": t_result,
        "leg_margin": {
            "mean_%s" % arm_name: round(mean_x, 4),
            "mean_%s" % baseline_name: round(mean_y, 4),
            "margin": round(margin, 4),
            "bar": MARGIN_BAR,
            "met": margin_ok,
        },
        "sub_bar_positive_direction": sub_bar_positive,
        "owes_confirm_near_miss": owes_nm,
        "descriptive_item_sign_flip": sf,
        "mde": mde,
        "n_valid_seed_pairs": n_pairs,
    }


# ================================================================
# main aggregation
# ================================================================
def aggregate():
    # ---- prerequisite files ----------------------------------------
    missing = []
    for f in [SMOKE_FILE, SG_ZC_FILE, SG_FS_FILE, SG_PC_FILE]:
        if not os.path.exists(f):
            missing.append(f)
    if missing:
        print("[agg] MISSING gate files: %s" % missing)
        print("[agg] Run exp81_run.py --smoke first.")
        # not fatal: we still aggregate what we can

    smoke_ok = os.path.exists(SMOKE_FILE)

    # ---- load gate artifacts ---------------------------------------
    def load_json_if(path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    sg_zc = load_json_if(SG_ZC_FILE)
    sg_fs = load_json_if(SG_FS_FILE)
    sg_pc = load_json_if(SG_PC_FILE)
    timing = load_json_if(TIMING_FILE)

    # ---- load all arm checkpoints ----------------------------------
    print("[agg] loading A2q2 seeds...", flush=True)
    a2q2_units = load_arm_seeds("A2q2")
    print("[agg] loading A2-WIDE seeds...", flush=True)
    wide_units = load_arm_seeds("A2-WIDE")
    print("[agg] loading A2-RETRAIN seeds...", flush=True)
    a2ret_units = load_arm_seeds("A2-RETRAIN")
    print("[agg] loading A2q2-FLOOR seeds...", flush=True)
    floor_units = load_arm_seeds("A2q2-FLOOR")
    print("[agg] loading committed A2 seeds...", flush=True)
    a2_units = load_committed_arm("A2")
    print("[agg] loading committed B1 seeds...", flush=True)
    b1_units = load_committed_arm("B1")

    # Item-id drift check: assert all trained arms that have data share the same
    # union_item_ids (guards against harness drift changing what gets evaluated)
    def get_union_ids(units):
        for (_s, d) in units:
            if d is not None and "union_item_ids" in d:
                return tuple(sorted(d["union_item_ids"]))
        return None

    ref_ids = get_union_ids(a2q2_units)
    for arm_label, arm_units in (("A2-WIDE", wide_units),
                                  ("A2-RETRAIN", a2ret_units),
                                  ("A2q2-FLOOR", floor_units)):
        ids_here = get_union_ids(arm_units)
        if ref_ids is not None and ids_here is not None and ids_here != ref_ids:
            raise RuntimeError(
                "[agg] ITEM-ID DRIFT: A2q2 and %s union_item_ids differ -- "
                "pairing by index would be wrong" % arm_label)

    # ---- check completeness ----------------------------------------
    def completeness(units, arm):
        missing_s = [s for s, d in units if d is None]
        n = sum(1 for s, d in units if d is not None)
        if missing_s:
            print("[agg] WARNING: %s missing seeds %s" % (arm, missing_s))
        return n, missing_s

    a2q2_n, _ = completeness(a2q2_units, "A2q2")
    wide_n, _ = completeness(wide_units, "A2-WIDE")
    a2ret_n, _ = completeness(a2ret_units, "A2-RETRAIN")
    floor_n, _ = completeness(floor_units, "A2q2-FLOOR")
    a2_n, _ = completeness(a2_units, "A2 (committed)")
    b1_n, _ = completeness(b1_units, "B1 (committed)")

    # ---- extract accuracies ----------------------------------------
    a2q2_accs = extract_union_accs(a2q2_units)
    wide_accs = extract_union_accs(wide_units)
    a2ret_accs = extract_union_accs(a2ret_units)
    floor_accs = extract_union_accs(floor_units)
    a2_accs = extract_union_accs(a2_units)
    b1_accs = extract_union_accs(b1_units)

    # valid pairs only
    def valid_pairs_only(xs, ys):
        return [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]

    # per-item correctness for descriptive sign-flip
    a2q2_pi = extract_per_item_correct(a2q2_units)
    a2ret_pi = extract_per_item_correct(a2ret_units)
    b1_pi = extract_per_item_correct(b1_units)
    wide_pi = extract_per_item_correct(wide_units)

    # ---- floor admissibility from SG-FS (smoke gate) ---------------
    # Note: trained A2q2-FLOOR units provide the floor-of-record once complete.
    # SG-FS is the pre-training smoke proxy (40-item, 5 epochs).
    floor_adm = None
    if sg_fs is not None:
        floor_adm = floor_admissibility(
            sg_fs["scrambled_accuracy"], sg_fs["n_eval_items"])
        floor_adm["source"] = SG_FS_FILE
        floor_adm["gate_passed"] = sg_fs["passed"]
        floor_adm["note"] = ("Smoke proxy (SG-FS). Trained floor-of-record is "
                             "A2q2-FLOOR arm (5 seeds, 30 epochs).")

    # Floor-of-record: read from trained A2q2-FLOOR union accs if available
    floor_of_record = None
    valid_floor = [a for a in floor_accs if a is not None]
    if valid_floor:
        floor_mean = sum(valid_floor) / len(valid_floor)
        # Use a representative n (from first available unit)
        floor_n_items = None
        for (_s, d) in floor_units:
            if d is not None:
                tu = d.get("test", {}).get("test_union", {})
                floor_n_items = tu.get("n")
                break
        if floor_n_items:
            floor_of_record = floor_admissibility(floor_mean, floor_n_items)
            floor_of_record["source"] = "A2q2-FLOOR trained arm (mean over seeds)"
            floor_of_record["n_seeds"] = len(valid_floor)
            floor_of_record["per_seed_accs"] = valid_floor

    # ---- PRIMARY BARS (doc 73 §3) ----------------------------------
    print("[agg] computing bars...", flush=True)

    # Bar 1: width_per_referent_helps = A2q2 beats A2-RETRAIN
    bar_width = compute_bar(
        a2q2_accs, a2ret_accs,
        arm_name="A2q2", baseline_name="A2-RETRAIN",
        n_test_union=N_TEST_UNION,
        per_item_x=a2q2_pi, per_item_y=a2ret_pi)

    # Bar 2: tier_q_q2 = A2q2 beats B1
    bar_tier = compute_bar(
        a2q2_accs, b1_accs,
        arm_name="A2q2", baseline_name="B1",
        n_test_union=N_TEST_UNION,
        per_item_x=a2q2_pi, per_item_y=b1_pi)

    # A2-WIDE vs A2-RETRAIN (invalidation check)
    bar_wide_vs_a2ret = compute_bar(
        wide_accs, a2ret_accs,
        arm_name="A2-WIDE", baseline_name="A2-RETRAIN",
        n_test_union=N_TEST_UNION,
        per_item_x=wide_pi, per_item_y=a2ret_pi)

    # A2-WIDE invalidation clause
    wide_invalidates = bool(bar_wide_vs_a2ret["met"])

    # Apply invalidation: width_per_referent_helps cannot fire if A2-WIDE also beats A2
    width_per_referent_helps_final = bar_width["met"] and not wide_invalidates

    # Per-unit gate_met: from stored coupling tracking (doc 73 §3 engagement criterion).
    # gate_met_unit = loss_decreased AND max|Δcoupling| >= 0.10 rad.
    # trainability_failure = gate_met false in ALL A2q2 units.
    a2q2_unit_gate_met = []
    for (_s, detail) in a2q2_units:
        if detail is None:
            continue
        ugm = detail.get("gate_met_unit")
        if ugm is not None:
            a2q2_unit_gate_met.append(bool(ugm))
    if a2q2_unit_gate_met:
        gate_met = any(a2q2_unit_gate_met)
        trainability_failure = not any(a2q2_unit_gate_met)
    else:
        # No training data yet; fall back to directional check
        valid = valid_pairs_only(a2q2_accs, a2ret_accs)
        gate_met = any(x > y for x, y in valid)
        trainability_failure = not gate_met

    # Third outcomes
    sub_bar_positive = bar_width.get("sub_bar_positive_direction", False)

    # owes_confirm: fires if any positive leg fires OR near-miss
    owes_confirm = bool(
        width_per_referent_helps_final
        or bar_tier["met"]
        or sub_bar_positive
        or bar_width.get("owes_confirm_near_miss", False)
        or bar_tier.get("owes_confirm_near_miss", False))

    # ---- A2 committed drift anchor reference (informational) -------
    # Compare A2q2 vs committed A2 (for informational drift reference)
    bar_a2q2_vs_a2committed = compute_bar(
        a2q2_accs, a2_accs,
        arm_name="A2q2", baseline_name="A2-committed",
        n_test_union=N_TEST_UNION)

    # ---- descriptive: A2-RETRAIN vs A2-committed (pairing quality) -
    a2ret_vs_a2_t = None
    valid_ret = valid_pairs_only(a2ret_accs, a2_accs)
    if len(valid_ret) >= 2:
        xs, ys = zip(*valid_ret)
        a2ret_vs_a2_t = t_test_one_sided(list(xs), list(ys))

    # ---- per-arm summary means ------------------------------------
    def safe_mean(accs):
        v = [a for a in accs if a is not None]
        return round(sum(v) / len(v), 4) if v else None

    arm_means = {
        "A2q2": safe_mean(a2q2_accs),
        "A2-WIDE": safe_mean(wide_accs),
        "A2-RETRAIN": safe_mean(a2ret_accs),
        "A2q2-FLOOR": safe_mean(floor_accs),
        "A2 (committed)": safe_mean(a2_accs),
        "B1 (committed)": safe_mean(b1_accs),
    }

    # ---- print verdicts -------------------------------------------
    print("\n[agg] === VERDICTS ===")
    print("[agg] width_per_referent_helps (t+margin): %s" %
          width_per_referent_helps_final)
    if wide_invalidates:
        print("[agg]   --> INVALIDATED: A2-WIDE also beats A2-RETRAIN "
              "(capacity confound)")
    print("[agg] tier_q_q2 (A2q2 vs B1):             %s" % bar_tier["met"])
    print("[agg] gate_met (any positive direction):   %s" % gate_met)
    print("[agg] trainability_failure:                %s" % trainability_failure)
    print("[agg] sub_bar_positive_direction:          %s" % sub_bar_positive)
    print("[agg] owes_confirm (seed-44000):           %s" % owes_confirm)
    print("[agg] A2-WIDE invalidates:                 %s" % wide_invalidates)
    print("[agg]")
    print("[agg] ARM MEANS (test_union accuracy):")
    for arm, mean in arm_means.items():
        print("[agg]   %-25s %.4f" % (arm, mean or float("nan")))

    # ---- assemble results ------------------------------------------
    results = {
        "experiment": "exp81 width per referent (doc 73)",
        "registration": "qnlp_private_docs/73_exp81_width_per_referent.md",
        "sole_writer": "exp81_aggregate.py",
        "when": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                            __import__("time").gmtime()),

        # -- prerequisite gates --
        "smoke_gate": {"passed": smoke_ok, "file": SMOKE_FILE},
        "gate_SG_ZC": sg_zc,
        "gate_SG_FS": sg_fs,
        "gate_SG_PC": sg_pc,
        "timing": timing,
        "floor_admissibility": floor_adm,
        "floor_of_record": floor_of_record,

        # -- bar provenance --
        "bar_provenance": BAR_PROVENANCE,

        # -- PRIMARY BOOLEANS (doc 73 §3) --
        "width_per_referent_helps": width_per_referent_helps_final,
        "tier_q_q2": bar_tier["met"],
        "gate_met": gate_met,
        "trainability_failure": trainability_failure,
        "sub_bar_positive_direction": sub_bar_positive,
        "owes_confirm": owes_confirm,
        "owes_confirm_scope": ("seed-44000 same-draw run: A2q2 + A2 + B1 + A1 "
                               "(doc 73 §3)") if owes_confirm else None,

        # -- A2-WIDE invalidation --
        "A2_WIDE_invalidation": {
            "invalidates_width_bar": wide_invalidates,
            "bar_A2WIDE_vs_A2RETRAIN": bar_wide_vs_a2ret,
            "interpretation": (
                "A2-WIDE (same params, no second qubit) also beats A2-RETRAIN "
                "at p<0.05 and margin>=0.06 -- the improvement is CAPACITY not "
                "architecture (WAZN-SHUFFLED pattern). width_per_referent_helps "
                "is invalidated even if both its legs otherwise pass."
                if wide_invalidates else
                "A2-WIDE does NOT match A2q2 vs A2-RETRAIN -- capacity confound "
                "does not apply; architecture reading is admitted."),
        },

        # -- DETAILED BAR RESULTS --
        "bar_width_per_referent_helps": {
            "final_verdict": width_per_referent_helps_final,
            "pre_invalidation_verdict": bar_width["met"],
            "invalidated_by_A2_WIDE": wide_invalidates,
            "detail": bar_width,
        },
        "bar_tier_q_q2": {
            "final_verdict": bar_tier["met"],
            "detail": bar_tier,
        },
        "bar_A2q2_vs_committed_A2_informational": {
            "note": "INFORMATIONAL ONLY: committed A2 is drift anchor, not "
                    "primary paired baseline (use A2-RETRAIN for pairing)",
            "detail": bar_a2q2_vs_a2committed,
        },

        # -- PER-ARM ACCURACY SUMMARY --
        "arm_means_test_union": arm_means,
        "per_seed_accs": {
            "A2q2": a2q2_accs,
            "A2-WIDE": wide_accs,
            "A2-RETRAIN": a2ret_accs,
            "A2q2-FLOOR": floor_accs,
            "A2 (committed)": a2_accs,
            "B1 (committed)": b1_accs,
        },

        # -- PAIRING QUALITY --
        "A2RETRAIN_vs_A2committed_t": {
            "note": "Descriptive: checks A2-RETRAIN reproduces committed A2 "
                    "under same harness (t test, no bar)",
            "result": a2ret_vs_a2_t,
        },

        # -- COMPLETENESS --
        "completeness": {
            "A2q2": {"n_loaded": a2q2_n, "n_expected": N_SEEDS},
            "A2-WIDE": {"n_loaded": wide_n, "n_expected": N_SEEDS},
            "A2-RETRAIN": {"n_loaded": a2ret_n, "n_expected": N_SEEDS},
            "A2q2-FLOOR": {"n_loaded": floor_n, "n_expected": N_SEEDS},
            "A2 (committed)": {"n_loaded": a2_n, "n_expected": N_SEEDS},
            "B1 (committed)": {"n_loaded": b1_n, "n_expected": N_SEEDS},
        },

        # -- PARAM ACCOUNTING (from SG-PC) --
        "param_accounting": sg_pc.get("param_accounting") if sg_pc else None,
        "param_ratios": sg_pc.get("ratios") if sg_pc else None,
        "K_MAX": 12,

        # -- NAMED THIRD OUTCOMES --
        "third_outcomes": {
            "sub_bar_positive_direction": {
                "fired": sub_bar_positive,
                "interpretation": "A2q2 beats A2-RETRAIN (p<0.05) but margin < 0.06 "
                                  "-- directional signal below threshold; "
                                  "feeds doc 74-style closure later.",
            },
            "trainability_failure": {
                "fired": trainability_failure,
                "interpretation": "A2q2 NEVER fired gate_met_unit (loss decreased AND "
                                  "max|Δcoupling| >= 0.10 rad) in any seed. "
                                  "The wall extends to local width.",
            },
        },

        # -- PER-UNIT GATE_MET (coupling engagement, doc 73 §3) --
        "gate_met_per_unit": {
            "values": a2q2_unit_gate_met,
            "any_true": gate_met,
            "rule": "loss_decreased AND max|Δcoupling| >= 0.10 rad per unit",
            "n_units_with_data": len(a2q2_unit_gate_met),
        },
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n[agg] wrote %s" % OUT)
    return results


# ================================================================
# main
# ================================================================
if __name__ == "__main__":
    agg = aggregate()
    print("\n[agg] READY-TO-LAUNCH status:")
    print("  smoke_gate:                  %s" % agg["smoke_gate"]["passed"])
    print("  gate_SG_ZC passed:           %s" %
          (agg["gate_SG_ZC"]["passed"] if agg["gate_SG_ZC"] else "missing"))
    print("  gate_SG_FS passed:           %s" %
          (agg["gate_SG_FS"]["passed"] if agg["gate_SG_FS"] else "missing"))
    print("  gate_SG_PC passed:           %s" %
          (agg["gate_SG_PC"]["passed"] if agg["gate_SG_PC"] else "missing"))
    print("  width_per_referent_helps:    %s" % agg["width_per_referent_helps"])
    print("  tier_q_q2:                   %s" % agg["tier_q_q2"])
    print("  gate_met:                    %s" % agg["gate_met"])
    print("  owes_confirm:                %s" % agg["owes_confirm"])
    print("  A2-WIDE invalidates:         %s" %
          agg["A2_WIDE_invalidation"]["invalidates_width_bar"])
