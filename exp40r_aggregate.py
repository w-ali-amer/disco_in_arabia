# -*- coding: utf-8 -*-
"""exp40r_aggregate.py -- SOLE WRITER of results_exp40r.json (doc 65 SS2.3).

Reads the raw stage artifacts and assembles the single results file:
    exp40r_smoke_raw.json    smoke battery S1-S3
    exp40r_c5_raw.json       C5 calibration + smoke S4
    exp40r_arabic_raw.json   Arabic diagnostic block (H / S / P / probe)

No stage runner writes results_exp40r.json; no other file does either.
`results_exp40b.json` (the August artifact) is opened READ-ONLY, for
comparison, and is never modified.

TWICE-EVALUATION (standing rule, doc 65 SS2.3): the C5 verdict is
INDEPENDENTLY RECOMPUTED here from the per-seed numbers by a separate code
path, rather than trusting the boolean the runner wrote. Agreement between
the two is recorded as `verdict_recomputation_agrees`; a disagreement would
itself be the finding.
"""

import json
import os
import sys
import time

SMOKE_RAW = "exp40r_smoke_raw.json"
C5_RAW = "exp40r_c5_raw.json"
ARABIC_RAW = "exp40r_arabic_raw.json"
AUGUST_C5 = "results_exp40b.json"
OUT = "results_exp40r.json"


def load(path, required=True):
    if not os.path.exists(path):
        if required:
            sys.exit("[aggregate] missing required artifact: %s" % path)
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recompute_c5_verdict(per_seed):
    """Independent recomputation of doc 22 SS9's pass mark.

    Deliberately written as a separate code path from exp40r_c5.py's
    `any(...)` expression: count qualifying seeds explicitly and report which.
    """
    qualifying = []
    for r in per_seed:
        ok = (r["train_acc"] >= 1.0 and r["validA_acc"] >= 1.0
              and r["comp_acc"] >= 0.95)
        if ok:
            qualifying.append(r["seed"])
    return len(qualifying) >= 1, qualifying


def main():
    t0 = time.time()
    smoke = load(SMOKE_RAW)
    c5 = load(C5_RAW)
    arabic = load(ARABIC_RAW, required=False)
    august = load(AUGUST_C5, required=False)

    # ---- twice-evaluation of the C5 verdict --------------------------
    recomputed, qualifying = recompute_c5_verdict(c5["per_seed"])
    agrees = bool(recomputed == c5["harness_calibrated"])

    # ---- August vs Mac per-seed comparison ---------------------------
    comparison = None
    if august:
        aug = {r["seed"]: r for r in august["per_seed"]}
        rows, max_abs = [], 0.0
        for r in c5["per_seed"]:
            a = aug.get(r["seed"])
            if not a:
                continue
            d = {k: abs(r[k] - a[k])
                 for k in ("train_acc", "validA_acc", "comp_acc")}
            max_abs = max(max_abs, max(d.values()))
            rows.append({
                "seed": r["seed"],
                "mac": {k: r[k] for k in ("train_acc", "validA_acc",
                                          "comp_acc", "best_epoch")},
                "august_wsl": {k: a[k] for k in ("train_acc", "validA_acc",
                                                 "comp_acc", "best_epoch")},
                "abs_diff": d,
                "best_epoch_matches": r["best_epoch"] == a["best_epoch"],
            })
        comparison = {
            "august_file": AUGUST_C5,
            "august_torch": august.get("torch_version"),
            "mac_torch": c5.get("torch_version"),
            "per_seed": rows,
            "max_abs_accuracy_difference": max_abs,
            "all_best_epochs_match": all(x["best_epoch_matches"]
                                         for x in rows),
            "speedup_vs_august": (round(august["wall_time_sec"]
                                        / c5["wall_time_sec"], 2)
                                  if c5["wall_time_sec"] else None),
        }

    # ---- smoke battery roll-up (S1-S3 + S4 from the C5 stage) --------
    smoke_tests = [{k: v for k, v in t.items() if k != "trajectory"}
                   for t in smoke["tests"]]
    smoke_tests.append(c5["smoke_s4"])
    smoke_all = all(t["passed"] for t in smoke_tests)

    results = {
        "experiment": "exp40r -- mini-Duneau calibration harness, REVIVAL "
                      "re-run on the Mac (doc 22 SS9 / doc 65 SS6.7)",
        "spec": "doc 22 SS9 (frozen, unchanged); doc 65 SS2.3 hygiene; "
                "target Duneau et al. arXiv:2409.08777",
        "written_by": "exp40r_aggregate.py (sole writer)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": c5["torch_version"],
        "python_version": c5["python_version"],

        "provenance": dict(c5["provenance"], **{
            "note": "The August WSL implementation is COMMITTED and present "
                    "on this box; it was reused UNMODIFIED rather than "
                    "re-implemented. Only drivers, the trajectory trainer, "
                    "the smoke battery, the Arabic diagnostic and this "
                    "aggregator are new. results_exp40b.json was opened "
                    "read-only and is unchanged.",
        }),

        # ---------------------------------------------------- C5
        "c5_calibration": {
            "dataset": c5["dataset"],
            "config": c5["config"],
            "per_seed": c5["per_seed"],
            "harness_calibrated": c5["harness_calibrated"],
            "verdict_rule": c5["verdict_rule"],
            "verdict_recomputed_independently": recomputed,
            "verdict_recomputation_agrees": agrees,
            "qualifying_seeds": qualifying,
            "wall_time_sec": c5["wall_time_sec"],
        },
        "august_comparison": comparison,

        # ------------------------------------------------- smoke
        "smoke_battery": {
            "description": "designed-to-fail builds per doc 65 SS2.3, run "
                           "BEFORE the calibration; every bar declared in "
                           "source before any run",
            "tests": smoke_tests,
            "all_passed": smoke_all,
        },

        # ---------------------------------------------- diagnostic
        "arabic_diagnostic": (
            {k: v for k, v in arabic.items() if k != "conditions"}
            if arabic else None),
        "arabic_diagnostic_conditions": (
            {name: [{k: v for k, v in r.items()
                     if k not in ("eval_acc",)} for r in seeds]
             for name, seeds in arabic["conditions"].items()}
            if arabic else None),
        "arabic_diagnostic_trajectory_note": (
            "Per-epoch curves (clean_train_acc, mean_loss, mean_grad_norm, "
            "running_train_acc) are retained in full above -- they are the "
            "instrument the August run lacked. Only the sparsely-populated "
            "per-epoch `eval_acc` list (validation was scored every 5 epochs, "
            "so it is mostly nulls) is dropped here; `final_eval_acc` is "
            "kept. The complete unabridged record, including `eval_acc`, is "
            "in %s." % ARABIC_RAW),

        "flagged_ambiguities": [
            "doc 22 SS9 specifies '3 layers Sim Circuit 4'. The calibrated "
            "harness uses SIX layers for the `follows` block (3 for the "
            "question effects). This is an August deviation, documented in "
            "exp40_mini_duneau.py's source but NOT in results_exp40b.json, "
            "whose config.ansatz string still reads '3 layers' and is "
            "therefore STALE. The oracle check finds the authors' repo uses "
            "3 layers uniformly, so this is a genuine divergence from the "
            "replication target, carried forward because it is what passed "
            "C5 in August and re-passes it here.",
            "doc 22 SS9's parenthetical 'NeurIPS-2024 compositional-learning "
            "workshop oral' is not corroborated: arXiv:2409.08777 carries no "
            "journal-ref or venue and appears to be a preprint only.",
            "doc 22 SS9's 'No public code or dataset exists' is FALSE for "
            "code (doc 65 SS6.5 already corrected this) and misleading for "
            "data: no dataset artifact is released, but the generator is "
            "public, so the data is fully regenerable.",
            "The mini's splits are narrower than the paper's by "
            "pre-registered scope (doc 22 SS9 fixes widths 2-12): the "
            "authors use compositional validation 9-20 and a 21-30 test "
            "tier, and stratify by story density. Scoped deviation, not a "
            "defect, but the mini therefore calibrates a strictly easier "
            "regime than the published headline.",
            "The Arabic diagnostic uses uniform[0, 2*pi) init (August "
            "parity), while doc 65 SS2.4 registers N(0, 0.1^2) for exp42 "
            "proper. Init is a live variable the parent owns.",
        ],
        "hygiene": {
            "git_commits_made": 0,
            "committed_artifacts_modified": [],
            "qnlp_private_docs_modified": False,
            "results_exp40b_modified": False,
            "gate_sets_touched_by_diagnostic": [],
            "note": "The Arabic diagnostic trains on `train` and reads `val` "
                    "only; test_a/test_b/test_c (the doc 22 SS7 gate sets) "
                    "were never evaluated, so exp42's pre-registered "
                    "comparison is uncontaminated.",
        },
        "aggregate_wall_time_sec": round(time.time() - t0, 2),
    }

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)

    print("=== exp40r aggregate ===")
    print("C5 harness_calibrated = %s (recomputed %s, agrees=%s), "
          "qualifying seeds %s"
          % (c5["harness_calibrated"], recomputed, agrees, qualifying))
    if comparison:
        print("August comparison: max |acc diff| = %.2e, all best_epochs "
              "match = %s, speedup %sx"
              % (comparison["max_abs_accuracy_difference"],
                 comparison["all_best_epochs_match"],
                 comparison["speedup_vs_august"]))
    print("smoke battery: %s (%d tests)"
          % ("ALL PASSED" if smoke_all else "FAILURE PRESENT",
             len(smoke_tests)))
    if arabic:
        s = arabic["summary"]
        print("Arabic diagnostic: honest mean max train %.4f vs scrambled "
              "%.4f (delta %.4f); honest mean final val %.4f"
              % (s["H_mean_max_train"], s["S_mean_max_train"],
                 s["honest_minus_scrambled_train"], s["H_mean_final_val"]))
    else:
        print("Arabic diagnostic: NOT PRESENT (artifact missing)")
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
