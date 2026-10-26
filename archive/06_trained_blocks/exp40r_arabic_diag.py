# -*- coding: utf-8 -*-
"""exp40r_arabic_diag.py -- ONE diagnostic block: does training MOVE on the
clean Arabic dataset? (doc 65 SS6.2's unexplained August event.)

THIS IS A DIAGNOSTIC, NOT exp42. It runs at deliberately reduced scale, on
one arm only, and it does NOT evaluate any gate, ablation or control. No
verdict boolean from doc 22 SS7 is computed here. Its single output is a
TRAINING TRAJECTORY, reported to the parent so the decision to commit to the
full exp42 is made on evidence.

WHAT IT IS DIAGNOSING
doc 65 SS6.2 rules the August exp42 VOID for dataset reasons but preserves
the honest caveat verbatim: "the dataset shortcut does not explain
all-arms-at-chance -- the structured arms consumed gold structure and still
learned nothing." Reading results_exp42.json closely sharpens that caveat and
supplies the hypothesis this block tests:

  * A2's reported `train_acc` values (0.4885, 0.5275, 0.5570, 0.6065, 0.5945)
    are recorded AT THE BEST-VALIDATION EPOCH, and those best epochs are
    1, 1, 4, 13, 7. Seed 0's headline "0.4885" is therefore an EPOCH-1
    number, not an end-of-training number.
  * Over the same 30 epochs A2's mean loss fell monotonically from ~0.72 to
    ~0.63 (chance cross-entropy is ln 2 = 0.6931) with grad norms rising to
    ~1.9-2.0. A3 behaved the same way and reached train_acc 0.663 by epoch 23.
  * No per-epoch TRAIN ACCURACY was ever recorded, so nothing in the August
    artifact can distinguish "the optimizer is dead" from "the optimizer was
    still climbing when the 30-epoch budget ran out".
  * By contrast A1 (the symmetric negative control) sat at 0.50-0.52 with the
    loss pinned at ~0.693 -- exactly as designed -- and B1/B2/B3 were flat.

  NOTE, recorded so the reasoning is auditable: a first framing of this
  block as "August was step-starved" is WRONG and was discarded before
  running. At batch 1 August ran 2000 steps/epoch x 30 epochs = 60k
  optimizer steps; the 200-item block below runs 12k. August had MORE steps,
  against a 10x larger fit target. The surviving question is not step count
  but SHAPE: August's train accuracy climbed slowly (A3 reached 0.663) while
  validation never followed. Was that shape caused by the invalid dataset
  (doc 65 SS6.2) or is it intrinsic to the family?

THE DISCRIMINATOR (the headline number of this block)
"Train accuracy moved" is NOT evidence of learning: the exp40r smoke battery
measured this harness reaching 0.695 train accuracy on 200 SCRAMBLED
mini-Duneau labels in 15 epochs, so a few hundred parameters memorize small
sets. This block therefore runs HONEST and SCRAMBLED-LABEL conditions on the
SAME 200 items with the SAME seeds and config, and compares trajectories:

  honest climbs faster/higher than scrambled  => task signal is extracted
  the two are indistinguishable                => nothing generalizable is
                                                  learned from clean data,
                                                  i.e. the August event
                                                  REPRODUCES on clean data

DESIGN (fixed before running)
- Arm: A2, the doc 22 SS3.1 dressed-symmetric family (`exp42_models`), the
  registered bet and the A2-analog the parent named.
- 200 train items, balanced, sampled from the exp41r train split; 200 held-out
  val items for a generalization read.
- 5 seeds, Adam, lr 0.005, batch 1 -- AUGUST PARITY, deliberately.
- Init: uniform[0, 2*pi) as `exp42_models` does. Doc 65 SS2.4 registers
  N(0, 0.1^2) for the revival's exp42, but this block is REPRODUCING the
  August condition, so it must use the August init. The difference is
  flagged to the parent as a live variable it owns for exp42 proper.
- Epoch budget 60 (August: 30), because the localizing question is whether
  the climb continues.
- CONDITIONS:
    H  honest   200 items, 5 seeds, 60 epochs
    S  scrambled 200 items (answers permuted), 5 seeds, 60 epochs -- the
       memorization baseline H must beat to mean anything
    P  August-parity: 2000 items, 30 epochs, lr/batch identical to August,
       2 seeds. Directly asks whether the August SHAPE (shallow loss slope,
       validation flat) reproduces on the CLEAN dataset. August's A2 loss
       went 0.72 -> 0.63 over these 30 epochs; this condition records the
       per-epoch train accuracy August never did.
- LOCALIZER: a tiny overfit probe (1 seed, 20 items, 200 epochs). Cannot
  overfit 20 => harness/optimization problem. Overfits 20 but flat on 200 =>
  capacity/task-scale problem. Different verdicts for the parent; cheap.

SCOPE GUARDRAIL: this block trains on `train` and reads `val` ONLY. It never
touches test_a / test_b / test_c -- those are the doc 22 SS7 gate sets, and a
diagnostic that peeked at them would contaminate exp42's pre-registered
comparison. One arm (A2), no ablations, no controls, no verdict booleans.

Reuses `exp42_compiler.compile_story` / `split_items` as functions. It does
NOT call `exp42_compiler.load_dataset`, whose EXPECTED_DATASET_SHA256 pins
the AUGUST dataset; the committed file is not edited. Writes
exp40r_arabic_raw.json only.
"""

import argparse
import json
import random
import sys
import time

import torch

import exp42_compiler as comp
import exp40r_trainer as trainer
from exp42_models import QuantumStoryModel

# doc 65 SS6.3 dataset of record for the revival.
EXP41R_PATH = "stories_exp41r.json"
EXP41R_SHA = ("2326942ffb6ccf6bca95a7ec0502c82563957fabb159b143608a9fb31f"
              "8f3113")


def load_exp41r(path=EXP41R_PATH):
    """Load + provenance-gate the revival dataset (doc 22 SS6 sanity gate)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items, meta = data["items"], data["meta"]
    sha = comp.canonical_sha256(items)
    if sha != meta["dataset_sha256"] or sha != EXP41R_SHA:
        raise RuntimeError(
            "exp41r provenance FAILED: computed=%s manifest=%s pin=%s"
            % (sha, meta["dataset_sha256"], EXP41R_SHA))
    print("[data] %s provenance OK sha=%s n=%d" % (path, sha[:16], len(items)),
          flush=True)
    return items, meta, sha


def balanced_sample(items, n, seed):
    rng = random.Random(seed)
    yes = [it for it in items if it["answer"] == "yes"]
    no = [it for it in items if it["answer"] == "no"]
    rng.shuffle(yes)
    rng.shuffle(no)
    out = yes[:n // 2] + no[:n // 2]
    rng.shuffle(out)
    return out


def scramble_answers(items, seed):
    """Permute the answer field across items: story->answer link destroyed.

    A permutation preserves the exact yes/no balance, so the scrambled
    condition remains a balanced binary problem and its train-accuracy
    trajectory is a pure memorization baseline.
    """
    rng = random.Random(seed)
    out = [dict(it) for it in items]
    answers = [it["answer"] for it in out]
    for _ in range(100):
        rng.shuffle(answers)
        moved = sum(1 for a, it in zip(answers, out) if a != it["answer"])
        if moved >= 0.3 * len(out):
            break
    for it, a in zip(out, answers):
        it["answer"] = a
    return out


def run_condition(name, arm, verbs, train_cs, eval_sets, seeds, epochs, lr,
                  batch):
    """Train `seeds` models on one condition; return per-seed trajectories."""
    answer_of = lambda cs: cs.answer                       # noqa: E731
    per_seed = []
    for seed in range(seeds):
        print("\n=== [%s] %s seed %d ===" % (name, arm, seed), flush=True)
        torch.manual_seed(seed)
        model = QuantumStoryModel(arm, verbs, seed)
        traj = trainer.train_with_trajectory(
            model, train_cs, eval_sets, answer_of, seed=seed, epochs=epochs,
            lr=lr, batch_size=batch,
            log_prefix="[%s %s s%d] " % (name, arm, seed), eval_every=5,
            early_stop_train=1.0)
        rec = trainer.strip_state(traj)
        rec["seed"] = seed
        rec["param_counts"] = model.param_counts()
        per_seed.append(rec)
        print("[%s %s s%d] final train=%.4f  max train=%.4f  final evals=%s"
              % (name, arm, seed, traj["final_clean_train_acc"],
                 traj["max_clean_train_acc"], traj["final_eval_acc"]),
              flush=True)
    return per_seed


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", type=str, default="A2")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--probe-items", type=int, default=20)
    ap.add_argument("--probe-epochs", type=int, default=200)
    # August-parity condition (2 seeds, bounded: this stays a diagnostic)
    ap.add_argument("--parity-n", type=int, default=2000)
    ap.add_argument("--parity-epochs", type=int, default=30)
    ap.add_argument("--parity-seeds", type=int, default=2)
    ap.add_argument("--out", type=str, default="exp40r_arabic_raw.json")
    args = ap.parse_args()

    t0 = time.time()
    print("[diag] torch %s python %s"
          % (torch.__version__, sys.version.split()[0]), flush=True)

    items, meta, sha = load_exp41r()
    angles, emb_meta, emb_sha = comp.load_embeddings()
    print("[diag] embeddings sha=%s n=%d" % (emb_sha[:16], len(angles)),
          flush=True)

    by = comp.split_items(items)
    train_items = balanced_sample(by["train"], args.n_train, 41000)
    val_items = balanced_sample(by["val"], args.n_val, 41001)

    verbs = sorted({ev["verb"] for it in items for ev in it["events"]})
    print("[diag] verbs=%d train=%d val=%d"
          % (len(verbs), len(train_items), len(val_items)), flush=True)

    train_cs = comp.compile_all(train_items, angles)
    val_cs = comp.compile_all(val_items, angles)

    # ---- condition H: honest labels ----------------------------------
    cond_h = run_condition("H-honest", args.arm, verbs, train_cs,
                           {"val": val_cs}, args.seeds, args.epochs, args.lr,
                           args.batch)

    # ---- condition S: scrambled labels (memorization baseline) -------
    scrambled_items = scramble_answers(train_items, 41003)
    scrambled_cs = comp.compile_all(scrambled_items, angles)
    n_moved = sum(1 for a, b in zip(train_items, scrambled_items)
                  if a["answer"] != b["answer"])
    print("\n[diag] scrambled condition: %d/%d labels moved"
          % (n_moved, len(train_items)), flush=True)
    cond_s = run_condition("S-scrambled", args.arm, verbs, scrambled_cs, {},
                           args.seeds, args.epochs, args.lr, args.batch)

    # ---- condition P: August-parity scale ----------------------------
    cond_p = []
    if args.parity_seeds > 0:
        parity_items = balanced_sample(by["train"], args.parity_n,
                                       41004)
        parity_cs = comp.compile_all(parity_items, angles)
        print("\n[diag] August-parity condition: %d train items, %d epochs, "
              "%d seeds" % (len(parity_cs), args.parity_epochs,
                            args.parity_seeds), flush=True)
        cond_p = run_condition("P-parity", args.arm, verbs, parity_cs,
                               {"val": val_cs}, args.parity_seeds,
                               args.parity_epochs, args.lr, args.batch)

    # ---- localizer: tiny overfit probe -------------------------------
    print("\n=== overfit probe: %d items, %d epochs, 1 seed ==="
          % (args.probe_items, args.probe_epochs), flush=True)
    probe_items = balanced_sample(by["train"], args.probe_items, 41002)
    probe_cs = comp.compile_all(probe_items, angles)
    torch.manual_seed(0)
    probe_model = QuantumStoryModel(args.arm, verbs, 0)
    probe_traj = trainer.train_with_trajectory(
        probe_model, probe_cs, {}, lambda cs: cs.answer, seed=0,
        epochs=args.probe_epochs, lr=args.lr, batch_size=args.batch,
        log_prefix="[probe] ", early_stop_train=1.0)
    probe = trainer.strip_state(probe_traj)
    probe["n_items"] = len(probe_cs)

    # ---- read-out (all DESCRIPTIVE; no doc 22 verdict is computed) ----
    h_max = [r["max_clean_train_acc"] for r in cond_h]
    s_max = [r["max_clean_train_acc"] for r in cond_s]
    h_val = [r["final_eval_acc"]["val"] for r in cond_h]
    delta = mean(h_max) - mean(s_max)
    probe_overfits = probe["max_clean_train_acc"] >= 0.95

    payload = {
        "stage": "exp40r Arabic diagnostic block (NOT exp42; no gate "
                 "evaluated, no doc 22 SS7 verdict computed)",
        "purpose": "does training move at all on the clean exp41r dataset, "
                   "and does the August all-arms-at-chance event reproduce",
        "hypothesis_under_test": "the August structured arms were "
                                 "UNDER-TRAINED (30 epochs, best-val-epoch "
                                 "reporting), not untrainable",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "dataset": {"path": EXP41R_PATH, "sha256": sha,
                    "seed": meta.get("seed"),
                    "n_items_total": len(items)},
        "embeddings": {"path": comp.EMB_PATH, "sha256": emb_sha,
                       "n": len(angles)},
        "config": {
            "arm": args.arm, "n_train": len(train_cs), "n_val": len(val_cs),
            "n_seeds": args.seeds, "max_epochs": args.epochs, "lr": args.lr,
            "batch": args.batch, "optimizer": "Adam (autodiff, no SPSA)",
            "init": "uniform[0, 2*pi) -- AUGUST PARITY, deliberately; doc 65 "
                    "SS2.4 registers N(0, 0.1^2) for exp42 proper. Flagged "
                    "to the parent as a live variable.",
            "august_comparison": {
                "n_train": 2000, "max_epochs": 30, "lr": 0.005, "batch": 1},
        },
        "conditions": {
            "H_honest": cond_h,
            "S_scrambled": cond_s,
            "P_august_parity": cond_p,
        },
        "scrambled_labels_moved": n_moved,
        "overfit_probe": probe,
        "summary": {
            "H_max_clean_train_acc": h_max,
            "S_max_clean_train_acc": s_max,
            "H_mean_max_train": mean(h_max),
            "S_mean_max_train": mean(s_max),
            "honest_minus_scrambled_train": delta,
            "H_final_val_acc": h_val,
            "H_mean_final_val": mean(h_val),
            "P_max_clean_train_acc": [r["max_clean_train_acc"]
                                      for r in cond_p],
            "P_final_val_acc": [r["final_eval_acc"]["val"] for r in cond_p],
            "probe_max_train_acc": probe["max_clean_train_acc"],
            "probe_overfits_descriptive": bool(probe_overfits),
            "reading_key": "honest >> scrambled on train AND val above chance "
                           "=> task signal extracted. honest ~= scrambled "
                           "=> memorization only, August event reproduces on "
                           "clean data.",
        },
        "wall_time_sec": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)

    print("\n=== Arabic diagnostic summary ===", flush=True)
    print("-- H honest (200 items) --")
    for r in cond_h:
        print("  seed %d: max train=%.4f  final train=%.4f  final val=%s  "
              "(%d epochs)"
              % (r["seed"], r["max_clean_train_acc"],
                 r["final_clean_train_acc"], r["final_eval_acc"]["val"],
                 r["epochs_run"]), flush=True)
    print("-- S scrambled (200 items, memorization baseline) --")
    for r in cond_s:
        print("  seed %d: max train=%.4f  final train=%.4f  (%d epochs)"
              % (r["seed"], r["max_clean_train_acc"],
                 r["final_clean_train_acc"], r["epochs_run"]), flush=True)
    print("HONEST - SCRAMBLED (mean max train acc) = %.4f" % delta)
    print("H mean final val = %s (chance 0.5)" % mean(h_val))
    if cond_p:
        print("-- P August-parity (%d items, %d epochs) --"
              % (args.parity_n, args.parity_epochs))
        for r in cond_p:
            print("  seed %d: max train=%.4f  final train=%.4f  final val=%s"
                  % (r["seed"], r["max_clean_train_acc"],
                     r["final_clean_train_acc"], r["final_eval_acc"]["val"]),
                  flush=True)
    print("overfit probe (%d items): max train=%.4f in %d epochs"
          % (probe["n_items"], probe["max_clean_train_acc"],
             probe["epochs_run"]), flush=True)
    print("wrote %s (%.1f s)" % (args.out, payload["wall_time_sec"]),
          flush=True)


if __name__ == "__main__":
    main()
