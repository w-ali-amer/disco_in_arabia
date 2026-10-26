# -*- coding: utf-8 -*-
"""exp40r_c5.py -- re-establish control C5 (mini-Duneau calibration) on the Mac.

Doc 22 SS9 pre-registered pass mark, VERBATIM and unchanged:
    harness_calibrated = (>= 1 of 5 seeds reaches 100% train AND 100%
    validation (widths <= 8) AND >= 95% on the compositional split
    (widths 9-12)).

PROVENANCE / WHAT IS PORTED VS REBUILT (doc 65 SS6.1):
exp40 already PASSED C5 in August on the WSL box (results_exp40b.json,
harness_calibrated=true, torch 2.7.1). That implementation --
`torch_sv_sim.py`, `duneau_task_gen.py`, `exp40_mini_duneau.py` -- is
committed and present on this box. It is reused UNMODIFIED: this driver
imports `exp40_mini_duneau.train_seed` and `physics_selftest` verbatim, so
the calibrated August training path is REPRODUCED, not re-implemented, and
any difference in outcome is attributable to the box (torch 2.13.0 /
py3.12 vs torch 2.7.1 / py3.10) rather than to a rewrite. NOTHING committed
is edited; this file and the raw artifact it writes are the only new
objects. results_exp40b.json is never opened for writing.

RECORDED DEVIATION FROM doc 22 SS9 (inherited from August, restated because
the August results JSON's own `config.ansatz` string is STALE and says "3
layers" for everything): the `follows` block is Circuit-4 with SIX layers,
not three (`exp40_mini_duneau.Exp40Model.__init__`). The in-code rationale
is that a 3-layer Circuit-4 on 3 wires cannot fit the follows channel (the
2-actor sanity plateaus ~0.86-0.93), because the channel must route the
follower's prior state onto the ancilla AND copy the followee's direction
in, which the adjacent-coupling CRX cascade needs the extra depth to do.
Question effects remain at 3 layers, per spec. This is flagged upward as an
open item, not silently carried.

Also runs S4 of the smoke battery, which needs a trained model:
  S4  WRONG-ORDER COMPILE.  The best seed's trained word unitaries are
      re-composed with the post-introduction events SHUFFLED. Accuracy must
      collapse -- if a scrambled story scores as well as the true one, the
      model is not reading event order and the task is not what it claims.
      BAR: scrambled accuracy < 0.80 (model scores ~1.0 unscrambled).

Writes exp40r_c5_raw.json. Sole writer of results_exp40r.json is
exp40r_aggregate.py.
"""

import argparse
import json
import random
import sys
import time

import torch

import duneau_task_gen as taskgen
import exp40_mini_duneau as mini

S4_MAX_SCRAMBLED_ACC_BAR = 0.80


def scramble_events(story, rng):
    """Shuffle post-introduction events; keep the introduction block first.

    Mirrors control C4 (sentence-scramble). The stored answer is NOT changed,
    so a model that reads order must lose accuracy against it.
    """
    t = dict(story)
    w = story["width"]
    intro, rest = story["events"][:w], list(story["events"][w:])
    rng.shuffle(rest)
    t["events"] = intro + rest
    return t


def oracle_unchanged_fraction(stories, scrambled):
    """Fraction of stories whose GROUND TRUTH survives the scramble.

    This is the ceiling a perfect order-reading model could score on the
    scrambled set against the original labels -- it makes the S4 bar
    interpretable instead of arbitrary.
    """
    same = 0
    for s0, s1 in zip(stories, scrambled):
        try:
            dirs = taskgen.simulate([tuple(e) for e in s1["events"]])
        except AssertionError:
            continue
        x, y = s1["question"]
        if x not in dirs or y not in dirs:
            continue
        truth = "yes" if dirs[x] == dirs[y] else "no"
        if truth == s0["answer"]:
            same += 1
    return same / len(stories)


def train_model_for_s4(seed, data, epochs, lr, batch):
    """Independently retrain one seed to obtain a TRAINED CHECKPOINT for S4.

    `exp40_mini_duneau.train_seed` keeps its selected state_dict internal and
    returns only metrics, and that committed file is not modified (hygiene).
    S4's claim -- "a model that has learned the task loses accuracy when the
    event order is scrambled" -- needs only *a* model that fits the task, not
    C5's specific selected checkpoint, so we refit the same seed under the
    same config with the trajectory trainer and stop at 100% clean train
    accuracy. This is recorded in the artifact so the substitution is visible.
    """
    import exp40r_trainer as trainer
    torch.manual_seed(seed)
    model = mini.Exp40Model(seed)
    traj = trainer.train_with_trajectory(
        model, data["train"], {"validA": data["validA"]},
        lambda s: mini.LABELS[s["answer"]], seed=seed, epochs=epochs, lr=lr,
        batch_size=batch, log_prefix="[S4-refit] ", early_stop_train=1.0)
    return model, traj


def run_s4(best_seed, model, data):
    print("\n=== S4 wrong-order compile (needs a trained model) ===", flush=True)
    stories = data["valid_comp"]
    rng = random.Random(24680)
    scrambled = [scramble_events(s, rng) for s in stories]

    acc_true = mini.evaluate(model, stories)
    # Evaluate the scrambled circuits against the ORIGINAL answers.
    paired = [dict(sc, answer=s0["answer"])
              for sc, s0 in zip(scrambled, stories)]
    acc_scr = mini.evaluate(model, paired)
    ceiling = oracle_unchanged_fraction(stories, scrambled)
    passed = bool(acc_scr < S4_MAX_SCRAMBLED_ACC_BAR)
    print("[S4] seed %d  true-order acc=%.4f  scrambled acc=%.4f  "
          "(ground-truth-unchanged ceiling %.4f)  bar < %.2f -> %s"
          % (best_seed, acc_true, acc_scr, ceiling, S4_MAX_SCRAMBLED_ACC_BAR,
             "PASS" if passed else "FAIL"), flush=True)
    return {"name": "S4_wrong_order_compile", "kind": "designed_to_fail",
            "bar": "scrambled accuracy < %.2f" % S4_MAX_SCRAMBLED_ACC_BAR,
            "seed_used": best_seed, "n_items": len(stories),
            "observed_true_order_acc": acc_true,
            "observed_scrambled_acc": acc_scr,
            "ground_truth_unchanged_ceiling": ceiling,
            "accuracy_drop": acc_true - acc_scr, "passed": passed}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--data", type=str, default="duneau_mini_data_b.json")
    ap.add_argument("--out", type=str, default="exp40r_c5_raw.json")
    args = ap.parse_args()

    t0 = time.time()
    print("[c5] torch %s  python %s"
          % (torch.__version__, sys.version.split()[0]), flush=True)

    mini.physics_selftest()

    # Determinism check across boxes: the generator must reproduce the sha
    # recorded by the August WSL run, or the difference is itself the finding.
    regen = taskgen.generate_dataset(40)
    AUG_SHA = ("f22eeea3cc9f14a084bffd05043fda226c274fafcd36f66cac696773"
               "bdb0fde3")
    sha_matches_august = regen["meta"]["sha256"] == AUG_SHA
    print("[c5] generator sha on this box = %s  (August WSL = %s)  match=%s"
          % (regen["meta"]["sha256"][:16], AUG_SHA[:16], sha_matches_august),
          flush=True)

    data = mini.load_or_generate_data(args.data, 40)
    run_data = {k: data[k] for k in ("train", "validA", "valid_comp")}

    per_seed = []
    for seed in range(args.seeds):
        r = mini.train_seed(seed, run_data, args.epochs, args.lr, args.batch)
        per_seed.append(r)

    harness_calibrated = any(
        r["train_acc"] >= 1.0 and r["validA_acc"] >= 1.0 and r["comp_acc"] >= 0.95
        for r in per_seed)

    # S4 uses the best-performing seed (by comp_acc, tie-broken by validA).
    best = max(per_seed, key=lambda r: (r["comp_acc"], r["validA_acc"]))
    s4_model, s4_traj = train_model_for_s4(
        best["seed"], run_data, args.epochs, args.lr, args.batch)
    import exp40r_trainer as trainer
    s4 = run_s4(best["seed"], s4_model, run_data)
    s4["refit_note"] = ("model independently refit with exp40r_trainer at the "
                        "same seed/config because exp40_mini_duneau.train_seed "
                        "does not expose its checkpoint and is not modified")
    s4["refit_final_clean_train_acc"] = s4_traj["final_clean_train_acc"]
    s4["refit_epochs_run"] = s4_traj["epochs_run"]

    payload = {
        "stage": "exp40r C5 mini-Duneau calibration (doc 22 SS9, Mac re-run)",
        "spec": "doc 22 SS9 (frozen); target Duneau et al. arXiv:2409.08777",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "provenance": {
            "ported_unmodified": ["torch_sv_sim.py", "duneau_task_gen.py",
                                  "exp40_mini_duneau.py"],
            "rebuilt": ["exp40r_c5.py (driver only)",
                        "exp40r_trainer.py", "exp40r_smoke.py"],
            "august_reference": "results_exp40b.json (NOT modified)",
            "august_torch": "2.7.1+cu126", "this_torch": torch.__version__,
        },
        "dataset": {
            "path": args.data, "seed": data["meta"]["seed"],
            "sha256": data["meta"]["sha256"],
            "august_wsl_sha256": AUG_SHA,
            "sha_matches_august_wsl": sha_matches_august,
            "sizes": {k: len(v) for k, v in run_data.items()},
        },
        "config": {
            "optimizer": "Adam (full autodiff, no SPSA)",
            "lr": args.lr, "batch_size": args.batch,
            "max_epochs": args.epochs, "n_seeds": args.seeds,
            "ansatz_follows_layers": 6,
            "ansatz_effect_layers": 3,
            "ansatz_note": "Sim-et-al Circuit 4. follows acts on 3 wires "
                           "(follower, followee, fresh ancilla traced at "
                           "readout) with SIX layers; question effects on 2 "
                           "wires with THREE layers. doc 22 SS9 says '3 "
                           "layers'; the 6-layer follows block is an August "
                           "deviation carried forward (see module docstring). "
                           "results_exp40b.json's own config.ansatz string is "
                           "STALE and does not record it.",
            "model_selection": "best validA accuracy only (never valid_comp)",
            "dtype": str(__import__("torch_sv_sim").CDTYPE),
        },
        "per_seed": per_seed,
        "smoke_s4": s4,
        "harness_calibrated": harness_calibrated,
        "verdict_rule": ">=1 of 5 seeds: 100% train AND 100% validA AND "
                        ">=95% valid_comp",
        "wall_time_sec": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)

    print("\n=== exp40r C5 summary ===", flush=True)
    for r in per_seed:
        print("seed %d: train=%.4f validA=%.4f valid_comp=%.4f (best epoch %d, "
              "ran %d, gn=%.3e)"
              % (r["seed"], r["train_acc"], r["validA_acc"], r["comp_acc"],
                 r["best_epoch"], r["epochs_run"],
                 r["mean_grad_norm_last_epoch"]), flush=True)
    print("harness_calibrated = %s" % harness_calibrated, flush=True)
    print("wrote %s (%.1f s)" % (args.out, payload["wall_time_sec"]), flush=True)


if __name__ == "__main__":
    main()
