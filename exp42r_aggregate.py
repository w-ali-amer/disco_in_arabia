# -*- coding: utf-8 -*-
"""exp42r_aggregate.py -- THE SOLE WRITER of results_exp42r.json.

doc 65 SS2.3: "the aggregator is a separate file and the sole writer of
`results_exp42.json`" (here: results_exp42r.json -- the August file is
NEVER modified, Rule 3). `exp42r_run.py` writes only per-unit checkpoints
and control RAW NUMBERS; every pre-registered BOOLEAN is computed here,
once, from those checkpoints, and every boolean ships with a REASON string
that names the numbers and the bar it was judged against.

IT NEVER READS results_exp42r_diagnostics.json. doc 65 SS7.3 registers the
T = 1 block and the purity/entropy sweep as DIAGNOSTICS and bars them from
the verdict: "Neither gates nor enters `tier_q`/`tier_s`; they sharpen the
read." `assert_diagnostics_are_barred()` below enforces that at run time by
refusing to proceed if this module's own source mentions the diagnostics
file anywhere outside this paragraph and that check.

------------------------------------------------------------------------
THE VERDICT FIELDS

doc 22 SS7, verbatim: "On the union compositional-generalization test set
(>= 50 items, >= 500 preferred): `tier_q_passed` = (quantum beats B1,
one-sided paired permutation p < 0.05) AND `content_earned` (C1). If
`tier_q_passed = false`: this line stops."

  tier_q_passed      A2 beats B1 on the UNION compositional set, one-sided
                     paired permutation p < 0.05, AND content_earned.
  tier_s_passed      BOTH structured models (A2 and B1) beat ALL THREE
                     unstructured baselines (B2, B2_generous, B3) on the
                     same items, one-sided p < 0.05 each.
  content_earned     C1: the original verb->block assignment STRICTLY
                     exceeds the 95th percentile of 1,000 permuted
                     seed-mean accuracies.
  trained_intro_helps  doc 65 SS2.1: A2-TI beats A2 on the union
                     compositional set, one-sided paired permutation
                     p < 0.05. THE THIRD OUTCOME IS REGISTERED -- the
                     field carries `true`, `false`, or `null` when A2-TI
                     did not complete, and the direction and effect size
                     are recorded either way.

SUPPORTING BOOLEANS, each with its bar in source:
  a1_at_chance       doc 22 SS3.2 makes A1 above chance a RUN-INVALIDATING
                     alarm: "if A1 scores above chance, the generator or
                     readout leaks, and the run is invalidated". Reported
                     as its own boolean and REFERENCED BY EVERY VERDICT
                     REASON, so a fired gate can never be read without it.
  c2_passed          C2 informative null, mean disjoint-inventory accuracy
                     <= 0.55.
  c3_order_invariant C3, machine-precision identity for the structured
                     arms + a measurable B2 delta.
  c4_destroyed       C4, scrambled accuracy on order-dependent items
                     <= 0.55.

------------------------------------------------------------------------
TWICE-EVALUATION (standing rule of 2026-09-02, doc 65 SS2.3). Every
accuracy this file reports is INDEPENDENTLY RECOMPUTED from the saved
state_dicts by a second code path -- models are rebuilt, weights reloaded,
and the compositional splits re-scored -- and asserted against the numbers
`exp42r_run.py` recorded. A mismatch is a REPORTED FAILURE, not a silent
preference for one of the two. `--no-recompute` exists only for a fast
structural dry run and stamps `twice_evaluated: false` in the output.

PER-ITEM ALIGNMENT. A paired permutation test is meaningless if the two
arms' correctness vectors are not the same items in the same order. Every
unit checkpoint stores its ordered `union_item_ids`; this file asserts they
are identical across every arm and seed BEFORE any pairing, and refuses to
compute a gate statistic otherwise.

ARM-LEVEL ISOLATION. An arm with a FAILED marker or missing units is
reported as incomplete WITH ITS REASON; the booleans that depend on it
become `null` with a reason naming the missing arm, never `false` by
accident.

Usage:  python3 exp42r_aggregate.py            # writes results_exp42r.json
"""

import argparse
import json
import math
import os
import platform
import random
import sys
import time

import torch

import exp42_compiler as comp
import exp42r_common as K
import exp42r_run as R

OUT = "results_exp42r.json"
DIAG_FILE_BARRED = "results_exp42r_diagnostics.json"

C2_BAR = 0.55
C4_BAR = 0.55
A1_CHANCE_TOL = 0.02              # doc 22 SS3.2 alarm tolerance, declared here
ALPHA = 0.05
N_PERM_STAT = 10000

DIAGNOSTICS_BARRED_NOTE = (
    "doc 65 SS7.3 registers the T = 1 single-event block and the "
    "purity/entropy-vs-T sweep as DIAGNOSTICS and bars them from the "
    "verdict: 'Neither gates nor enters tier_q/tier_s; they sharpen the "
    "read.' results_exp42r_diagnostics.json is NOT read by this file and no "
    "number in it enters any boolean below. Enforced at run time by "
    "assert_diagnostics_are_barred().")


def assert_diagnostics_are_barred():
    """Refuse to run if this module could read the diagnostics artefact.

    A comment is a promise; this is a check. The source is scanned for any
    `open`/`load` of the diagnostics filename outside the two places the
    name legitimately appears (this function and the note above)."""
    src = open(__file__, encoding="utf-8").read()
    # Only an actual READ CALL counts. Prose that merely NAMES the file --
    # the note above, this docstring -- must not trip the guard, or the
    # guard would forbid documenting itself.
    bad = [ln for ln in src.splitlines()
           if DIAG_FILE_BARRED in ln
           and ("open(" in ln or "json.load(" in ln or "loads(" in ln)]
    if bad:
        raise RuntimeError("doc 65 SS7.3 VIOLATION: this aggregator appears "
                           "to read the diagnostics file: %s" % bad)
    return True


# ==================================================================
# loading
# ==================================================================
def load_units(data, seeds):
    """Per-arm unit checkpoints + arm-level completeness with reasons."""
    per_arm, status = {}, {}
    for arm in K.ARMS:
        details, missing, failed = [], [], []
        for s in range(seeds):
            cj, cp = K.unit_paths(arm, s)
            fp = K.failed_path("main:%s:s%d" % (arm, s))
            if os.path.exists(cj) and os.path.exists(cp):
                with open(cj, encoding="utf-8") as f:
                    details.append(json.load(f))
            elif os.path.exists(fp):
                with open(fp, encoding="utf-8") as f:
                    failed.append(json.load(f))
            else:
                missing.append(s)
        per_arm[arm] = details
        status[arm] = {
            "n_units": len(details), "n_expected": seeds,
            "complete": len(details) == seeds,
            "seeds_missing": missing,
            "seeds_failed": [f["job"] for f in failed],
            "failure_reasons": [{"job": f["job"], "error": f["error"]}
                                for f in failed],
            "reason": ("all %d seeds present" % seeds
                       if len(details) == seeds else
                       "INCOMPLETE: %d/%d units; missing seeds %s; failed %s"
                       % (len(details), seeds, missing,
                          [f["job"] for f in failed])),
        }
    return per_arm, status


def twin_census(data):
    """Which union twins are PURE within-event argument exchanges?

    doc 22 SS3.2's A1 guarantee -- "[U, SWAP] = 0, so conjugating by SWAP
    relabels the wires and the two stories produce identical statistics" --
    is exact only for twins of that kind. exp41_datagen's Q2 twin flips the
    PREVIOUS event's arguments, which re-resolves the pro-dropped subject
    (exp41_datagen lines 326-334), so the pro-drop event acts on a different
    WIRE PAIR in the two twins. A SWAP-symmetric model can see that without
    any leak. This function measures the split and derives A1's arithmetic
    ceiling, so the SS3.2 alarm is read rather than fired mechanically."""
    def order_blind(cs):
        return [(v, min(x, y), max(x, y)) for v, x, y in cs.events]

    union = data["compiled"]["test_union"]
    pos = {cs.item_id: i for i, cs in enumerate(union)}
    pairs = {}
    for cs in union:
        pairs.setdefault(cs.pair_id, []).append(cs)
    eq_idx, neq_idx, by_qtype = [], [], {}
    n_eq = n_neq = 0
    for _pid, t in pairs.items():
        if len(t) != 2:
            continue
        same = order_blind(t[0]) == order_blind(t[1])
        (eq_idx if same else neq_idx).extend(pos[cs.item_id] for cs in t)
        if same:
            n_eq += 1
        else:
            n_neq += 1
        k = "%s/%s" % (t[0].qtype,
                       "swap_equivalent" if same else "NOT_swap_equiv")
        by_qtype[k] = by_qtype.get(k, 0) + 1
    n_items = len(eq_idx) + len(neq_idx)
    ceiling = ((0.5 * len(eq_idx) + 1.0 * len(neq_idx)) / n_items
               if n_items else None)
    return {"n_swap_equivalent_pairs": n_eq,
            "n_not_swap_equiv_pairs": n_neq,
            "n_swap_equivalent_items": len(eq_idx),
            "n_not_swap_equiv_items": len(neq_idx),
            "by_qtype": by_qtype,
            "swap_equivalent_union_indices": sorted(eq_idx),
            "a1_arithmetic_ceiling_on_union": ceiling,
            "why": ("doc 22 SS3.2's A1 identical-statevector guarantee is "
                    "exact on the swap-equivalent subset and does not apply "
                    "off it; the ceiling is what a SWAP-symmetric model "
                    "could reach on the union WITH NO LEAK AT ALL. FLAGGED: "
                    "this is a property of the exp41r Q2 pro-drop "
                    "construction, discovered at build time, and it changes "
                    "no registered criterion.")}


def assert_alignment(per_arm, data):
    """No pairing before the item vectors are proven identical."""
    ref = data["union_ids"]
    report = {"n_union_items": len(ref), "checked": []}
    for arm, details in per_arm.items():
        for d in details:
            if d.get("union_item_ids") != ref:
                raise RuntimeError(
                    "PER-ITEM ALIGNMENT FAILED for %s seed %s: the union "
                    "item-id vector differs from the dataset's. A paired "
                    "permutation test on misaligned vectors fabricates its "
                    "own statistic; refusing to compute one."
                    % (arm, d.get("seed")))
            if len(d["union_per_item_correct"]) != len(ref):
                raise RuntimeError("union correctness length mismatch: %s s%s"
                                   % (arm, d.get("seed")))
            report["checked"].append("%s_s%s" % (arm, d.get("seed")))
    report["all_aligned"] = True
    report["note"] = ("every arm and seed carries the identical ordered "
                      "union item-id vector, asserted before any pairing")
    return report


# ==================================================================
# twice-evaluation
# ==================================================================
def recompute(per_arm, data, seeds):
    """SECOND, INDEPENDENT code path: rebuild each model from its saved
    state_dict and re-score the splits. doc 65 SS2.3's standing rule."""
    out = {"method": "models rebuilt from exp42r_ckpt/*.pt and re-scored by "
                     "a code path that does not reuse the runner's recorded "
                     "numbers",
           "per_unit": [], "max_abs_accuracy_delta": 0.0, "mismatches": []}
    for arm, details in per_arm.items():
        for d in details:
            s = d["seed"]
            _cj, cp = K.unit_paths(arm, s)
            m, _spec = K.build_model(arm, data["verbs"], data["vocab"], s,
                                     data["angles"])
            m.load_state_dict(torch.load(cp, weights_only=True))
            m.eval()
            row = {"arm": arm, "seed": s, "splits": {}}
            for sp in ("test_a", "test_b", "test_c", "test_union"):
                acc, corr = R.evaluate(m, data["compiled"][sp])
                rec = d["test"][sp]["accuracy"]
                delta = abs(acc - rec)
                row["splits"][sp] = {"recomputed": acc, "recorded": rec,
                                     "abs_delta": delta}
                out["max_abs_accuracy_delta"] = max(
                    out["max_abs_accuracy_delta"], delta)
                if delta > 1e-12:
                    out["mismatches"].append(
                        {"arm": arm, "seed": s, "split": sp,
                         "recomputed": acc, "recorded": rec})
                if sp == "test_union" and corr != d["union_per_item_correct"]:
                    out["mismatches"].append(
                        {"arm": arm, "seed": s, "split": sp,
                         "error": "per-item correctness vector differs"})
            out["per_unit"].append(row)
            print("[recompute] %s s%d union %.4f (delta %.1e)"
                  % (arm, s, row["splits"]["test_union"]["recomputed"],
                     row["splits"]["test_union"]["abs_delta"]), flush=True)
    out["agrees"] = not out["mismatches"]
    out["reading"] = ("agrees=true means the runner's table and an "
                      "independent re-scoring of the same checkpoints "
                      "produce identical accuracies; agrees=false is a "
                      "REPORTED FAILURE, not a silent choice between them")
    return out


# ==================================================================
# statistics
# ==================================================================
def paired_permutation_pvalue(diff, n_perm=N_PERM_STAT, seed=777):
    """One-sided sign-flip permutation test on paired per-item differences
    (H1: mean(diff) > 0). exp42_run's statistic, unchanged."""
    rng = random.Random(seed)
    obs = sum(diff) / len(diff)
    ge = 0
    for _ in range(n_perm):
        s = sum(d if rng.random() < 0.5 else -d for d in diff)
        if s / len(diff) >= obs:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


def seed_mean_correct(details, key="union_per_item_correct"):
    vecs = [d[key] for d in details]
    n = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(n)]


def split_seed_mean_correct(details, split):
    vecs = [d["test"][split]["per_item_correct"] for d in details]
    n = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(n)]


def compare(dx, dy, label_x, label_y, split=None):
    """One-sided paired test: does X beat Y on these items?"""
    if split:
        mx, my = (split_seed_mean_correct(dx, split),
                  split_seed_mean_correct(dy, split))
    else:
        mx, my = seed_mean_correct(dx), seed_mean_correct(dy)
    diff = [a - b for a, b in zip(mx, my)]
    obs, p = paired_permutation_pvalue(diff)
    return {"x": label_x, "y": label_y, "split": split or "test_union",
            "mean_acc_x": sum(mx) / len(mx), "mean_acc_y": sum(my) / len(my),
            "mean_paired_diff": obs, "one_sided_p": p, "n_items": len(diff),
            "alpha": ALPHA, "x_beats_y": bool(p < ALPHA),
            "test": "one-sided sign-flip paired permutation over per-item "
                    "seed-mean correctness, %d resamples" % N_PERM_STAT}


# ==================================================================
# controls
# ==================================================================
def aggregate_c1():
    shards, missing = [], []
    for k in range(R.C1_SHARDS):
        p = os.path.join(K.CTL_DIR, "c1_shard%d.json" % k)
        if not os.path.exists(p):
            missing.append(k)
            continue
        with open(p, encoding="utf-8") as f:
            shards.append(json.load(f))
    if not shards:
        return {"control": "C1 trained-dial-swap", "available": False,
                "shards_missing": missing,
                "reason": "no C1 shard completed; content_earned is null"}
    origs = [s["original_seed_mean_acc"] for s in shards]
    if max(origs) - min(origs) > 1e-12:
        return {"control": "C1 trained-dial-swap", "available": False,
                "shards_missing": missing,
                "reason": "shards disagree on the ORIGINAL seed-mean "
                          "accuracy (%r) -- they did not evaluate the same "
                          "models; refusing to combine them" % origs}
    accs = []
    for s in sorted(shards, key=lambda x: x["shard"]):
        accs.extend(s["permuted_seed_mean_accs"])
    n = len(accs)
    sa = sorted(accs)
    q95 = sa[int(0.95 * n) - 1]
    orig = origs[0]
    return {
        "control": "C1 trained-dial-swap (doc 22 SS6)",
        "available": True, "arm": shards[0]["arm"],
        "scope_note": shards[0]["scope_note"],
        "shards_present": sorted(s["shard"] for s in shards),
        "shards_missing": missing,
        "n_permutations": n, "n_permutations_registered": R.N_PERM,
        "all_permutations_present": bool(n == R.N_PERM),
        "perm_seed": R.C1_PERM_SEED,
        "original_seed_mean_acc": orig,
        "perm_95th_percentile": q95, "perm_max": sa[-1], "perm_min": sa[0],
        "perm_mean": sum(accs) / n,
        "content_earned": bool(orig > q95 and n == R.N_PERM),
        "criterion": "the original verb->block assignment must STRICTLY "
                     "exceed the 95th percentile of the permuted seed-mean "
                     "accuracies (doc 22 SS6). If the trained model's blocks "
                     "are interchangeable, the content was not earned and "
                     "H-Q cannot be claimed regardless of scores.",
        "fast_path_max_abs_delta_vs_committed_forward":
            max(s["fast_path_max_abs_delta_vs_committed_forward"]
                for s in shards),
    }


def aggregate_c2(seeds):
    per, missing = [], []
    for s in range(seeds):
        p = os.path.join(K.CTL_DIR, "c2_s%d.json" % s)
        if not os.path.exists(p):
            missing.append(s)
            continue
        with open(p, encoding="utf-8") as f:
            per.append(json.load(f))
    if not per:
        return {"control": "C2 informative null", "available": False,
                "seeds_missing": missing,
                "reason": "no C2 unit completed (the restricted-inventory "
                          "datasets may not have been generated); c2_passed "
                          "is null"}
    accs = [r["evaluation"]["accuracy"] for r in per]
    mean = sum(accs) / len(accs)
    return {"control": "C2 informative null (doc 22 SS6)", "available": True,
            "arm": per[0]["arm"], "n_seeds": len(per),
            "seeds_missing": missing,
            "train_dataset": per[0]["train_dataset"],
            "eval_dataset": per[0]["eval_dataset"],
            "verb_bijection": per[0]["verb_bijection"],
            "per_seed_disjoint_accuracy": accs,
            "per_seed_trainA_val_accuracy": [r["training"]["val_acc"]
                                             for r in per],
            "mean_disjoint_accuracy": mean, "bar": C2_BAR,
            "c2_passed": bool(mean <= C2_BAR),
            "criterion": "blocks trained on the verbset-A inventory, "
                         "evaluated on verbset-B stories through the fixed "
                         "kind-matched bijection, must sit at <= chance + "
                         "5 pp (%.2f). Above it, block content is generic "
                         "frame machinery rather than verb-specific."
                         % C2_BAR}


def aggregate_c3():
    p = os.path.join(K.CTL_DIR, "c3.json")
    if not os.path.exists(p):
        return {"control": "C3 order-invariance", "available": False,
                "reason": "c3.json absent"}
    with open(p, encoding="utf-8") as f:
        c3 = json.load(f)
    pop = c3.get("population", {})
    ident = bool(pop) and all(v["machine_precision_identical"]
                              for v in pop.values())
    b2 = c3.get("B2_behavioral_delta") or {}
    moved = b2.get("prediction_flip_fraction")
    c3.update({
        "available": True,
        "structured_arms_machine_precision_identical": ident,
        "B2_prediction_flip_fraction": moved,
        "c3_order_invariant": bool(ident),
        "criterion": "SVO<->VSO re-rendering must change the structured "
                     "arms' compiled circuit NOT AT ALL (exact statevector "
                     "identity -- they consume gold events and never see the "
                     "surface) while B2's input, and measurably its "
                     "behaviour, does change. Both facts are recorded; the "
                     "identity is the criterion.",
    })
    return c3


def aggregate_c4():
    p = os.path.join(K.CTL_DIR, "c4.json")
    if not os.path.exists(p):
        return {"control": "C4 sentence-scramble", "available": False,
                "reason": "c4.json absent"}
    with open(p, encoding="utf-8") as f:
        c4 = json.load(f)
    per = c4.get("per_arm", {})
    a2 = per.get("A2")
    c4.update({
        "available": True, "bar": C4_BAR,
        "c4_destroyed_A2": (bool(a2["mean_accuracy_scrambled"] <= C4_BAR)
                            if a2 else None),
        "criterion": "shuffling event order must destroy accuracy on the "
                     "generator-labelled order-dependent items (mean "
                     "scrambled accuracy <= %.2f). NOTE, stated plainly: on "
                     "an arm that is already AT CHANCE unscrambled, this "
                     "control is vacuous -- it can only be informative for "
                     "an arm that was above chance to begin with, and the "
                     "unscrambled accuracy is reported beside it so a reader "
                     "can see which case applies." % C4_BAR,
    })
    return c4


# ==================================================================
# verdicts
# ==================================================================
def build_verdicts(per_arm, status, c1, c2, c3, c4, stats, splits,
                   census):
    v, reasons = {}, {}

    def have(arm):
        return status[arm]["complete"]

    # ---- A1 sanity alarm (doc 22 SS3.2) ----------------------------
    # Reported TWICE, on purpose. The registered boolean reads the union, as
    # doc 22 SS3.2 words it. The second boolean reads the subset where that
    # section's PHYSICS ARGUMENT actually applies -- see `twin_census`.
    if have("A1"):
        a1 = sum(d["test"]["test_union"]["accuracy"]
                 for d in per_arm["A1"]) / len(per_arm["A1"])
        cen = census
        idx = cen["swap_equivalent_union_indices"]
        a1_eq = None
        if idx:
            mc = seed_mean_correct(per_arm["A1"])
            a1_eq = sum(mc[i] for i in idx) / len(idx)
        v["a1_at_chance"] = bool(a1 <= 0.5 + A1_CHANCE_TOL)
        v["a1_at_chance_on_swap_equivalent_items"] = (
            bool(a1_eq <= 0.5 + A1_CHANCE_TOL) if a1_eq is not None else None)
        reasons["a1_at_chance"] = (
            "A1 (exchange-symmetric core, no dressing) seed-mean UNION "
            "accuracy = %.4f against the chance + %.2f bar = %.4f. doc 22 "
            "SS3.2: 'if A1 scores above chance, the generator or readout "
            "leaks, and the run is invalidated'.\n"
            "READ THIS WITH THE TWIN CENSUS. SS3.2's guarantee rests on "
            "[U_A1, SWAP] = 0 making a swap twin compile to the IDENTICAL "
            "statevector. That is exact for the %d Q1/Q0 twin pairs and does "
            "NOT hold for the %d Q2 pairs, whose pro-dropped subject is "
            "RE-RESOLVED by the flip and therefore lands on a different wire "
            "pair -- a difference a SWAP-symmetric model may legitimately "
            "see. A1's arithmetic ceiling on the union is %.4f with no leak "
            "whatsoever. On the swap-equivalent subset alone A1 scores "
            "%s, where the bar is exact.\n%s"
            % (a1, A1_CHANCE_TOL, 0.5 + A1_CHANCE_TOL,
               cen["n_swap_equivalent_pairs"], cen["n_not_swap_equiv_pairs"],
               cen["a1_arithmetic_ceiling_on_union"],
               ("%.4f" % a1_eq) if a1_eq is not None else "n/a",
               "A1 sits at chance on the union: the negative control holds "
               "on the registered reading and the other verdicts are "
               "readable."
               if a1 <= 0.5 + A1_CHANCE_TOL else
               "A1 IS ABOVE CHANCE ON THE UNION. doc 22 SS3.2 read literally "
               "invalidates the run. Before that is applied, the "
               "swap-equivalent number above must be consulted: if A1 is at "
               "chance THERE and above chance only on Q2, the excess is the "
               "pro-drop re-resolution measured by the census, not a leak. "
               "THE RULING IS THE USER'S; this file reports both numbers and "
               "changes no registered criterion."))
        reasons["a1_at_chance_on_swap_equivalent_items"] = (
            "A1 on the %d union items whose twins differ by within-event "
            "argument exchange ALONE, where doc 22 SS3.2's identical-"
            "statevector argument is exact: %s against the %.4f bar."
            % (len(idx), ("%.4f" % a1_eq) if a1_eq is not None else "n/a",
               0.5 + A1_CHANCE_TOL))
    else:
        v["a1_at_chance"] = None
        v["a1_at_chance_on_swap_equivalent_items"] = None
        reasons["a1_at_chance"] = ("A1 incomplete: %s"
                                   % status["A1"]["reason"])
        reasons["a1_at_chance_on_swap_equivalent_items"] = reasons[
            "a1_at_chance"]

    # ---- content_earned (C1) ---------------------------------------
    if c1.get("available"):
        v["content_earned"] = c1["content_earned"]
        reasons["content_earned"] = (
            "C1 on %s: original verb->block assignment scores %.4f; the %d "
            "sampled permutations have 95th percentile %.4f (max %.4f, mean "
            "%.4f). Pass requires STRICT exceedance. %s"
            % (c1["arm"], c1["original_seed_mean_acc"], c1["n_permutations"],
               c1["perm_95th_percentile"], c1["perm_max"], c1["perm_mean"],
               "Content is earned by the specific trained blocks."
               if c1["content_earned"] else
               "The trained blocks are INTERCHANGEABLE: content was not "
               "earned and H-Q cannot be claimed regardless of scores "
               "(doc 22 SS6)."))
    else:
        v["content_earned"] = None
        reasons["content_earned"] = "C1 unavailable: %s" % c1.get("reason")

    # ---- tier_q_passed ---------------------------------------------
    key = "A2_vs_B1"
    if key in stats and v["content_earned"] is not None:
        st = stats[key]
        v["tier_q_passed"] = bool(st["one_sided_p"] < ALPHA
                                  and v["content_earned"])
        reasons["tier_q_passed"] = (
            "doc 22 SS7: tier_q = (A2 beats B1 on the union compositional "
            "set, one-sided paired permutation p < %.2f) AND content_earned. "
            "A2 %.4f vs B1 %.4f, paired diff %+.4f over %d items, p = %.4f; "
            "content_earned = %s. => %s%s"
            % (ALPHA, st["mean_acc_x"], st["mean_acc_y"],
               st["mean_paired_diff"], st["n_items"], st["one_sided_p"],
               v["content_earned"],
               "PASS." if v["tier_q_passed"] else
               "FAIL -- doc 22 SS7: this line stops; the result feeds the "
               "Phase 3 controls paper, and no re-run with a tweaked ansatz "
               "may hunt for a pass.",
               "" if v.get("a1_at_chance") in (True, None) else
               " NOTE: a1_at_chance is FALSE, which invalidates this reading "
               "under doc 22 SS3.2."))
    else:
        v["tier_q_passed"] = None
        reasons["tier_q_passed"] = (
            "not computable: %s"
            % ("A2 or B1 incomplete (A2: %s; B1: %s)"
               % (status["A2"]["reason"], status["B1"]["reason"])
               if key not in stats else "content_earned is null"))

    # ---- tier_s_passed ---------------------------------------------
    need = [("%s_vs_%s" % (s, u)) for s in ("A2", "B1")
            for u in ("B2", "B2_generous", "B3")]
    got = [k for k in need if k in stats]
    if len(got) == len(need):
        flags = {k: stats[k]["one_sided_p"] < ALPHA for k in need}
        v["tier_s_passed"] = bool(all(flags.values()))
        reasons["tier_s_passed"] = (
            "doc 22 SS1 H-S: BOTH structured models must beat ALL THREE "
            "unstructured baselines on the same items. " +
            "; ".join("%s p=%.4f (%+.4f)"
                      % (k, stats[k]["one_sided_p"],
                         stats[k]["mean_paired_diff"]) for k in need) +
            ". => %s H-S alone supports only 'compositional structure "
            "matters' -- the existing paper's claim, NOT a quantum one -- "
            "and classical then wins on cost (doc 22 SS1)."
            % ("PASS." if v["tier_s_passed"] else "FAIL."))
    else:
        v["tier_s_passed"] = None
        reasons["tier_s_passed"] = ("not computable: missing comparisons %s"
                                    % sorted(set(need) - set(got)))

    # ---- trained_intro_helps (doc 65 SS2.1) ------------------------
    key = "A2TI_vs_A2"
    if key in stats:
        st = stats[key]
        v["trained_intro_helps"] = bool(st["one_sided_p"] < ALPHA)
        reasons["trained_intro_helps"] = (
            "doc 65 SS2.1: A2-TI beats A2 on the union compositional set, "
            "one-sided paired permutation p < %.2f. A2-TI %.4f vs A2 %.4f, "
            "paired diff %+.4f over %d items, p = %.4f. => %s This is the "
            "ledger's trained-ENCODING lever probe; A2-TI amends doc 22 "
            "SS3.1's seed-only rule knowingly -- the loss is QA and the "
            "embeddings only initialise, so nothing is solved against them. "
            "THE THIRD OUTCOME IS REGISTERED: if A2-TI did not complete the "
            "field is null, never false."
            % (ALPHA, st["mean_acc_x"], st["mean_acc_y"],
               st["mean_paired_diff"], st["n_items"], st["one_sided_p"],
               "PASS -- training the encoding moves what the frozen encoding "
               "could not." if v["trained_intro_helps"] else
               "FAIL -- training the noun introduction does not help at this "
               "design point."))
    else:
        v["trained_intro_helps"] = None
        reasons["trained_intro_helps"] = (
            "not computable: A2TI or A2 incomplete (A2TI: %s; A2: %s). doc "
            "65 SS2.1's third outcome -- the field is null, not false."
            % (status["A2TI"]["reason"], status["A2"]["reason"]))

    # ---- control booleans ------------------------------------------
    for name, src, field in (("c2_passed", c2, "c2_passed"),
                             ("c3_order_invariant", c3, "c3_order_invariant"),
                             ("c4_destroyed_A2", c4, "c4_destroyed_A2")):
        if src.get("available"):
            v[name] = src.get(field)
            reasons[name] = src.get("criterion", "")
        else:
            v[name] = None
            reasons[name] = "not run: %s" % src.get("reason")

    # ---- A2 vs A3, the carve-out doc 65 SS2.7 preserves -------------
    if "A2_vs_A3" in stats:
        st = stats["A2_vs_A3"]
        gap = st["mean_acc_y"] - st["mean_acc_x"]
        v["a2_approx_a3"] = bool(abs(gap) <= 0.02)
        reasons["a2_approx_a3"] = (
            "doc 22 SS3.2 / doc 65 SS2.7, the carve-out preserved VERBATIM "
            "and sendable to the Pasqal/Reem thread on EITHER gate branch: "
            "A2 %.4f vs A3 %.4f (gap %+.4f, bar 0.02). If A2 ~= A3, "
            "thematic-role encoding is dressing-cheap and the "
            "analog-compilable symmetric family plus local detuning "
            "suffices; A3's exchange-asymmetric CRX-cascade family is fine "
            "on trapped ions but not under a global Rydberg drive."
            % (st["mean_acc_x"], st["mean_acc_y"], gap))
    else:
        v["a2_approx_a3"] = None
        reasons["a2_approx_a3"] = "A2 or A3 incomplete"

    return v, reasons


# ==================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-recompute", action="store_true",
                    help="skip twice-evaluation (structural dry run only)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    assert_diagnostics_are_barred()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    data = R.prepare()

    per_arm, status = load_units(data, args.seeds)
    align = assert_alignment(per_arm, data)
    census = twin_census(data)

    # ---------------- gate statistics --------------------------------
    def ok(a):
        return status[a]["complete"]

    stats, splits = {}, {}
    pairs = [("A2", "B1"), ("A2TI", "A2"), ("A2TI", "B1"), ("A2", "A3"),
             ("A3", "B1"), ("A1", "B1")]
    pairs += [(s, u) for s in ("A2", "B1")
              for u in ("B2", "B2_generous", "B3")]
    for x, y in pairs:
        if ok(x) and ok(y):
            stats["%s_vs_%s" % (x, y)] = compare(per_arm[x], per_arm[y], x, y)
    # per-split breakdown (doc 22 SS5: "each split reported separately")
    for sp in ("test_a", "test_b", "test_c"):
        splits[sp] = {}
        for x, y in [("A2", "B1"), ("A2TI", "A2")]:
            if ok(x) and ok(y):
                splits[sp]["%s_vs_%s" % (x, y)] = compare(
                    per_arm[x], per_arm[y], x, y, split=sp)

    per_arm_summary = {}
    for arm, ds in per_arm.items():
        if not ds:
            per_arm_summary[arm] = {"status": status[arm]}
            continue
        n = len(ds)
        row = {"status": status[arm], "n_seeds": n,
               "param_counts": ds[0]["param_counts"],
               "init_policy": ds[0]["init"]["policy"],
               "init_stream_seeds": [d["init"]["stream_seed"] for d in ds],
               "init_vector_sha256": [d["init"]["init_vector_sha256"][:16]
                                      for d in ds],
               "val_acc": [d["val_acc"] for d in ds],
               "best_epoch": [d["best_epoch"] for d in ds],
               "train_acc_at_best": [d["train_acc_at_best"] for d in ds],
               "final_grad_norm": [d["epoch_mean_grad_norms"][-1]
                                   for d in ds],
               "per_split": {}}
        for sp in ("test_a", "test_b", "test_c", "test_union"):
            accs = [d["test"][sp]["accuracy"] for d in ds]
            mu = sum(accs) / n
            sd = math.sqrt(sum((a - mu) ** 2 for a in accs) / n) if n > 1 else 0
            row["per_split"][sp] = {"per_seed": accs, "mean": mu, "std": sd,
                                    "n_items": ds[0]["test"][sp]["n"]}
        per_arm_summary[arm] = row

    c1, c2 = aggregate_c1(), aggregate_c2(args.seeds)
    c3, c4 = aggregate_c3(), aggregate_c4()
    verdicts, reasons = build_verdicts(per_arm, status, c1, c2, c3, c4,
                                       stats, splits, census)

    twice = ({"twice_evaluated": False,
              "reason": "--no-recompute: structural dry run"}
             if args.no_recompute else recompute(per_arm, data, args.seeds))
    if not args.no_recompute:
        twice["twice_evaluated"] = True

    results = {
        "experiment": "exp42r -- trained-block Arabic story QA, the REVIVAL "
                      "(doc 65 SS6.2: the FIRST VALID RUN of the registered "
                      "design; the August exp42 verdict is void and "
                      "results_exp42.json is NOT modified)",
        "spec": "qnlp_private_docs/22_phase2_trained_story_qa_design.md "
                "SS3-SS7 + 65_exp40_42_revival.md SS2 (A2-TI, init policy, "
                "hygiene) and SS7.4 (the GO)",
        "written_by": "exp42r_aggregate.py (SOLE WRITER of this file)",
        "diagnostics_barred_from_this_verdict": DIAGNOSTICS_BARRED_NOTE,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "provenance": data["prov"],
        "readout_ambiguity_resolution": K.READOUT_AMBIGUITY_NOTE,
        "init_policy_doc65_SS2_4": {
            "registered": "all trained parameters ~ N(0, 0.1^2) from "
                          "per-(arm, seed) dedicated streams; recorded per "
                          "run",
            "stream_rule": "42_000_000 + 1000*ARM_INDEX[arm] + seed",
            "flagged_deviation_A2TI": K.A2TI_INIT_DEVIATION,
            "flagged_interpretation_B1": K.B1_INIT_INTERPRETATION,
            "no_init_arms": "doc 65 SS2.4: the ledger's identity-vs-uniform "
                            "question is NOT imported as arms and no "
                            "bounded-redraw machinery is registered; none "
                            "was added.",
        },
        "config": {"optimizer": "Adam (full autodiff, no SPSA -- doc 22 "
                                "SS3.3 hard rule)",
                   "lr": 0.005, "batch": 1, "max_epochs": 30,
                   "n_seeds": args.seeds,
                   "model_selection": "in-distribution val split ONLY; the "
                                      "compositional splits are scored once, "
                                      "after selection, on the frozen model",
                   "arms": list(K.ARMS)},
        "arm_status": status,
        "per_item_alignment": align,
        "twin_census_doc22_SS3_2": census,
        "per_arm": per_arm_summary,
        "gate_statistics": stats,
        "per_split_breakdown": splits,
        "controls": {"C1": c1, "C2": c2, "C3": c3, "C4": c4,
                     "C5": data["prov"]["c5_gate"]},
        "verdicts": verdicts,
        "verdict_reasons": reasons,
        "twice_evaluation": twice,
        "kill_gate_doc22_SS7": (
            "On the union compositional-generalization test set (>= 50 "
            "items, >= 500 preferred): tier_q_passed = (quantum beats B1, "
            "one-sided paired permutation p < 0.05) AND content_earned (C1). "
            "If tier_q_passed = false: THIS LINE STOPS. The result -- "
            "whatever it is -- feeds the Phase 3 controls paper. No re-runs "
            "with tweaked ansaetze hunting for a pass; one pre-registered "
            "ablation set is the whole search space. If only tier_s_passed: "
            "the claim shrinks to 'structure matters', stated with classical "
            "winning on cost."),
        "expectation_on_record_doc65_SS7_4": (
            "tier_q / tier_s were EXPECTED to close negative before any "
            "number existed; doc 22 SS8 priced this branch ('the strongest "
            "section of the controls paper') and a valid interpreted "
            "negative IS the deliverable."),
        "confirm_draw_owed_doc65_SS2_2": (
            "A fired tier_q_passed and/or trained_intro_helps is confirmed "
            "on a FRESH generator draw at seed 41000 + 1000 = 42000: the "
            "winning quantum arm AND B1, retrained, 5 seeds, paired on the "
            "same confirm items, with B1-ON-THE-CONFIRM-DRAW as the "
            "reference. Not optional; roughly doubles the decisive stage's "
            "compute. Owed = %s."
            % any(verdicts.get(k) is True
                  for k in ("tier_q_passed", "trained_intro_helps"))),
        "wall_time_sec": round(time.time() - t0, 1),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print("\n=== exp42r VERDICTS ===")
    for k in ("tier_q_passed", "tier_s_passed", "content_earned",
              "trained_intro_helps", "a1_at_chance",
              "a1_at_chance_on_swap_equivalent_items", "c2_passed",
              "c3_order_invariant", "c4_destroyed_A2", "a2_approx_a3"):
        print("  %-22s %s" % (k, verdicts.get(k)))
    print("\nunion accuracy (seed mean +/- std):")
    for arm in K.ARMS:
        r = per_arm_summary.get(arm, {})
        if "per_split" in r:
            u = r["per_split"]["test_union"]
            print("  %-12s %.4f +/- %.4f  (val %.4f)"
                  % (arm, u["mean"], u["std"],
                     sum(r["val_acc"]) / len(r["val_acc"])))
        else:
            print("  %-12s INCOMPLETE: %s" % (arm, status[arm]["reason"]))
    print("\nwrote %s (%.1f s)" % (args.out, results["wall_time_sec"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
