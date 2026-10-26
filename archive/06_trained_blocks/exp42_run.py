# -*- coding: utf-8 -*-
"""exp42_run.py -- orchestrator for the trained-block Arabic story QA runs
(doc 22 SS3-SS7). Config-driven; 5 seeds per arm; model selection on
in-distribution val ONLY; compositional splits test_a (longer T), test_b
(larger K), test_c (unseen pairings) evaluated separately + union; gate
statistics and all pre-registered booleans written to results_exp42.json.

HARD GATE (pre-registered control C5, doc 22 SS6): exp42 REFUSES to run
real training unless results_exp40b.json exists with
harness_calibrated == true. A harness that cannot reproduce the published
positive makes our negative uninterpretable. Override --smoke runs a
mechanical smoke ONLY (capped at 20 train items / 3 epochs) and writes NO
results file.

Usage:
  smoke (allowed pre-gate):   python3 exp42_run.py --smoke [--arm A2]
  real run (post-gate only):  python3 exp42_run.py
"""

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time

import torch

import exp42_compiler as comp
import exp42_controls as ctl
from exp42_baselines import (BagBaseline, ClassicalDisCoCirc, SeqBaseline,
                             build_vocab, sample_name_mapping)
from exp42_models import QuantumStoryModel

GATE_FILE = "results_exp40b.json"
RESULTS_BASE = "results_exp42"
ALL_ARMS = ("A1", "A2", "A3", "B1", "B2", "B2_generous", "B3")
SMOKE_TRAIN_CAP = 20
SMOKE_EPOCHS = 3

READOUT_AMBIGUITY_NOTE = (
    "doc 22 SS3.4 does not specify how the question's verb identity enters "
    "the two-effect readout (Duneau's task had one fixed question). "
    "Resolution (flagged, decided before any training): the question verb's "
    "OWN trained block is applied as inverse preparation on the two queried "
    "wires, then two SHARED trained Circuit-4 effects (q_pos/q_neg, 15 "
    "params each) -- zero per-verb readout parameters, SS3.3 budget "
    "preserved, C1 permutes story and question coherently, A1's exact-"
    "chance twin guarantee preserved. Q1 bridge questions (two verbs) "
    "apply both blocks as inverse preparations on the queried (x, y) "
    "wires in reverse order. See exp42_models.py docstring.")


# ----------------------------------------------------------------- gate
def check_gate(smoke):
    if smoke:
        print("[gate] --smoke: mechanical smoke only (<=%d items, %d "
              "epochs), no results file will be written."
              % (SMOKE_TRAIN_CAP, SMOKE_EPOCHS))
        return
    if not os.path.exists(GATE_FILE):
        sys.exit("[gate] REFUSING to train: %s does not exist. exp42 is "
                 "blocked until the exp40 harness-calibration verdict "
                 "(pre-registered control C5, doc 22 SS6). Use --smoke for "
                 "mechanical checks only." % GATE_FILE)
    with open(GATE_FILE) as f:
        verdict = json.load(f)
    if verdict.get("harness_calibrated") is not True:
        sys.exit("[gate] REFUSING to train: %s has harness_calibrated=%r "
                 "(must be true). C5 not satisfied."
                 % (GATE_FILE, verdict.get("harness_calibrated")))
    print("[gate] C5 open: %s harness_calibrated=true" % GATE_FILE)


# ------------------------------------------------------------- factory
def make_model(arm, verbs, vocab, seed):
    if arm in QuantumStoryModel.ARMS:
        return QuantumStoryModel(arm, verbs, seed)
    if arm == "B1":
        return ClassicalDisCoCirc(verbs, seed, d_ref=2)
    if arm == "B2":
        return SeqBaseline(vocab, seed, variant="matched")
    if arm == "B2_generous":
        return SeqBaseline(vocab, seed, variant="generous")
    if arm == "B3":
        return BagBaseline(vocab, seed)
    raise ValueError(arm)


def is_seq(arm):
    return arm in ("B2", "B2_generous", "B3")


# ------------------------------------------------------------ training
def evaluate(model, compiled):
    correct = [1 if model.predict(cs) == cs.answer else 0 for cs in compiled]
    return sum(correct) / len(correct), correct


def grad_global_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return math.sqrt(total)


def train_seed(arm, seed, data, meta, vocab, cfg):
    torch.manual_seed(seed)
    model = make_model(arm, cfg["verbs"], vocab, seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = random.Random(9000 + seed)
    train, val = data["train"], data["val"]
    # Name augmentation (doc 22 SS4) is a training-protocol feature; under
    # --smoke it is DISABLED so the mechanical loss-decrease check is not
    # confounded by per-presentation name resampling on 20 items.
    augment = arm in ("B2", "B2_generous") and not cfg.get("smoke", False)

    best = {"val_acc": -1.0, "train_acc": -1.0, "epoch": -1, "state": None}
    epoch_grad_norms, epoch_losses = [], []
    first_batch_checked = False
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses, gnorms, run_correct = [], [], 0
        for start in range(0, len(order), cfg["batch"]):
            batch = [train[i] for i in order[start:start + cfg["batch"]]]
            opt.zero_grad()
            batch_loss = 0.0
            for cs in batch:
                if augment:
                    mapping = sample_name_mapping(meta["humans"],
                                                  meta["objects"], rng)
                    loss, pred = model.story_loss(cs, mapping=mapping)
                else:
                    loss, pred = model.story_loss(cs)
                batch_loss = batch_loss + loss
                if pred == cs.answer:
                    run_correct += 1
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            gn = grad_global_norm(model)
            assert math.isfinite(gn), \
                "non-finite gradient norm (%s seed %d)" % (arm, seed)
            if not first_batch_checked:
                assert gn > 0.0, \
                    "zero gradient on first batch (%s seed %d)" % (arm, seed)
                first_batch_checked = True
            gnorms.append(gn)
            opt.step()
            losses.append(batch_loss.item())

        val_acc, _ = evaluate(model, val)
        mean_loss = sum(losses) / len(losses)
        mean_gn = sum(gnorms) / len(gnorms)
        epoch_grad_norms.append(mean_gn)
        epoch_losses.append(mean_loss)
        if val_acc > best["val_acc"]:                     # ID val ONLY
            clean_train, _ = evaluate(model, train)
            best = {"val_acc": val_acc, "train_acc": clean_train,
                    "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
        print("[%s seed %d] epoch %2d/%d  loss=%.4f  train(run)=%.3f  "
              "val=%.3f  grad_norm=%.3e  best@%d(val=%.3f)  %.1fs"
              % (arm, seed, epoch, cfg["epochs"], mean_loss,
                 run_correct / len(train), val_acc, mean_gn,
                 best["epoch"], best["val_acc"], time.time() - t0),
              flush=True)
        if best["val_acc"] >= 1.0 and best["train_acc"] >= 1.0:
            print("[%s seed %d] early stop: train and val both 100%%"
                  % (arm, seed), flush=True)
            break

    model.load_state_dict(best["state"])
    detail = {"seed": seed, "best_epoch": best["epoch"],
              "epochs_run": epoch, "train_acc": best["train_acc"],
              "val_acc": best["val_acc"],
              "epoch_mean_grad_norms": epoch_grad_norms,
              "epoch_mean_losses": epoch_losses, "test": {}}
    for split in ("test_a", "test_b", "test_c", "test_union"):
        acc, correct = evaluate(model, data[split])
        detail["test"][split] = {"accuracy": acc, "n": len(correct)}
        if split == "test_union":
            detail["test_union_correct"] = correct        # for pairing
    return model, detail


# ------------------------------------------------------------ statistics
def paired_permutation_pvalue(diff, n_perm=10000, seed=777):
    """One-sided sign-flip permutation test on paired per-item differences
    (H1: mean(diff) > 0). Returns (observed_mean, p)."""
    rng = random.Random(seed)
    obs = sum(diff) / len(diff)
    ge = 0
    for _ in range(n_perm):
        s = sum(d if rng.random() < 0.5 else -d for d in diff)
        if s / len(diff) >= obs:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


def seed_mean_correct(per_seed_details):
    """Per-item mean correctness across seeds on the union test set."""
    vecs = [d["test_union_correct"] for d in per_seed_details]
    n = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(n)]


def compare(arm_details_x, arm_details_y):
    """One-sided paired test: does X beat Y on the union set?"""
    mx = seed_mean_correct(arm_details_x)
    my = seed_mean_correct(arm_details_y)
    diff = [a - b for a, b in zip(mx, my)]
    obs, p = paired_permutation_pvalue(diff)
    return {"mean_acc_x": sum(mx) / len(mx), "mean_acc_y": sum(my) / len(my),
            "mean_paired_diff": obs, "one_sided_p": p,
            "n_items": len(diff)}


# ------------------------------------------------------------- plumbing
def fresh_results_path(base):
    path = base + ".json"
    if not os.path.exists(path):
        return path
    for suffix in "bcdefghij":
        p = base + suffix + ".json"
        if not os.path.exists(p):
            return p
    raise RuntimeError("too many results files for base %s" % base)


def git_note():
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             capture_output=True, text=True,
                             timeout=30).stdout
        ours = [l for l in out.splitlines() if "exp42" in l]
        return {"exp42_paths": ours,
                "n_untracked": sum(1 for l in out.splitlines()
                                   if l.startswith("??"))}
    except Exception as e:                                # noqa: BLE001
        return {"error": str(e)}


def balanced_head(compiled, k, seed=0):
    rng = random.Random(seed)
    pool = compiled[:]
    rng.shuffle(pool)
    yes = [c for c in pool if c.answer == 0][:k // 2]
    no = [c for c in pool if c.answer == 1][:k // 2]
    return yes + no


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="mechanical smoke only: <=20 items, 3 epochs, "
                         "1 seed, NO results file (pre-gate allowed)")
    ap.add_argument("--arm", default="all",
                    help="one of %s or 'all'" % (ALL_ARMS,))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--batch", type=int, default=1)      # exp40b parity
    ap.add_argument("--fresh", action="store_true",
                    help="ignore per-arm checkpoints and retrain")
    args = ap.parse_args()

    check_gate(args.smoke)
    t_start = time.time()

    items, meta, dataset_sha = comp.load_dataset()
    angles, emb_meta, emb_sha = comp.load_embeddings()
    print("[data] dataset sha256=%s..  embeddings sha256=%s.. (%s)"
          % (dataset_sha[:16], emb_sha[:16], emb_meta["source"][:60]))

    verbs = sorted(meta["verb_inventory"].keys())
    by_split = comp.split_items(items)
    compiled = {k: comp.compile_all(v, angles, with_tokens=True)
                for k, v in by_split.items()}
    vocab = build_vocab(cs.tokens for cs in
                        compiled["train"] + compiled["val"]
                        + compiled["test_union"])

    arms = list(ALL_ARMS) if args.arm == "all" else [args.arm]
    for a in arms:
        assert a in ALL_ARMS, a

    cfg = {"verbs": verbs, "lr": args.lr, "batch": args.batch,
           "epochs": SMOKE_EPOCHS if args.smoke else args.epochs,
           "smoke": bool(args.smoke)}
    n_seeds = 1 if args.smoke else args.seeds

    data = dict(compiled)
    if args.smoke:
        data["train"] = balanced_head(compiled["train"], SMOKE_TRAIN_CAP)
        data["val"] = balanced_head(compiled["val"], 12)
        for s in ("test_a", "test_b", "test_c"):
            data[s] = balanced_head(compiled[s], 4)
        data["test_union"] = data["test_a"] + data["test_b"] + data["test_c"]
        print("[smoke] sizes: train=%d val=%d union=%d"
              % (len(data["train"]), len(data["val"]),
                 len(data["test_union"])))

    results_per_arm, models_per_arm = {}, {}
    smoke_report = {}
    for arm in arms:
        # Per-arm checkpointing (mechanical robustness only, full runs):
        # a crash in a later arm must not lose finished arms, and "rerun
        # THAT arm" = delete its two checkpoint files and relaunch.
        ck_json, ck_pt = "exp42_ckpt_%s.json" % arm, "exp42_ckpt_%s.pt" % arm
        if (not args.smoke and not args.fresh
                and os.path.exists(ck_json) and os.path.exists(ck_pt)):
            with open(ck_json, encoding="utf-8") as f:
                details = json.load(f)
            states = torch.load(ck_pt, weights_only=True)
            models = []
            for d, st in zip(details, states):
                m = make_model(arm, verbs, vocab, d["seed"])
                m.load_state_dict(st)
                models.append(m)
            print("[ckpt] %s: loaded %d seeds from checkpoint"
                  % (arm, len(models)), flush=True)
        else:
            details, models = [], []
            for seed in range(n_seeds):
                m, d = train_seed(arm, seed, data, meta, vocab, cfg)
                models.append(m)
                details.append(d)
            if not args.smoke:
                with open(ck_json, "w", encoding="utf-8") as f:
                    json.dump(details, f, ensure_ascii=False)
                torch.save([m.state_dict() for m in models], ck_pt)
                print("[ckpt] %s: saved" % arm, flush=True)
        results_per_arm[arm] = details
        models_per_arm[arm] = models
        if args.smoke:
            # mechanical verdicts only: loss decreased? grads finite?
            gns = details[0]["epoch_mean_grad_norms"]
            ls = details[0]["epoch_mean_losses"]
            smoke_report[arm] = {
                "param_counts": models[0].param_counts(),
                "grad_norms_per_epoch": gns,
                "grads_finite": all(math.isfinite(g) for g in gns),
                "losses_per_epoch": ls,
                "loss_decreased": bool(ls[-1] < ls[0]),
                "train_acc_best": details[0]["train_acc"],
            }

    if args.smoke:
        print("\n[smoke] SUMMARY (mechanical only -- NOT results):")
        ok = True
        for arm, r in smoke_report.items():
            ok = ok and r["grads_finite"] and r["loss_decreased"]
            print("  %-12s params=%-6d grads_finite=%s loss %s -> %s "
                  "(decreased=%s) train_acc=%.2f"
                  % (arm, r["param_counts"]["total_trained"],
                     r["grads_finite"], "%.4f" % r["losses_per_epoch"][0],
                     "%.4f" % r["losses_per_epoch"][-1],
                     r["loss_decreased"], r["train_acc_best"]))
        print("[smoke] %s -- no results file written (pre-registered)."
              % ("ALL MECHANICAL CHECKS PASSED" if ok
                 else "SMOKE FAILURE (see above)"))
        if not ok:
            sys.exit(1)
        return

    # ---------------- full-run gate statistics + controls (post-C5 only)
    stats, verdicts = {}, {}
    if "A2" in results_per_arm and "B1" in results_per_arm:
        stats["A2_vs_B1"] = compare(results_per_arm["A2"],
                                    results_per_arm["B1"])
        c1 = ctl.c1_permutation_test(models_per_arm["A2"],
                                     data["test_union"], verbs)
        stats["C1"] = c1
        verdicts["content_earned"] = c1["content_earned"]
        verdicts["tier_q_passed"] = bool(
            stats["A2_vs_B1"]["one_sided_p"] < 0.05
            and c1["content_earned"])
    tier_s = []
    for s_arm in ("A2", "B1"):
        for u_arm in ("B2", "B2_generous", "B3"):
            if s_arm in results_per_arm and u_arm in results_per_arm:
                key = "%s_vs_%s" % (s_arm, u_arm)
                stats[key] = compare(results_per_arm[s_arm],
                                     results_per_arm[u_arm])
                tier_s.append(stats[key]["one_sided_p"] < 0.05)
    if tier_s:
        verdicts["tier_s_passed"] = bool(all(tier_s))

    controls = {"C2_scaffold": ctl.c2_scaffold_report(
        items, meta["verb_inventory"])}
    if "A2" in models_per_arm:
        probe_item = next(it for it in by_split["test_a"]
                          if "SVO" in it["order_flags"]
                          and "VSO" in it["order_flags"])
        struct = {"A2_seed0": models_per_arm["A2"][0]}
        if "B1" in models_per_arm:
            struct["B1_seed0"] = models_per_arm["B1"][0]
        seq = models_per_arm.get("B2", [None])[0]
        controls["C3"] = ctl.c3_order_invariance(struct, seq, probe_item,
                                                 angles)
        if seq is not None:
            controls["C3_B2_delta"] = ctl.c3_b2_behavioral_delta(
                seq, by_split["test_union"][:200], angles)
        controls["C4_A2"] = ctl.c4_sentence_scramble(
            models_per_arm["A2"][0], data["test_union"])

    for arm, details in results_per_arm.items():
        for d in details:
            d.pop("test_union_correct", None)             # bulky; summarized

    results = {
        "experiment": "exp42 trained-block Arabic story QA",
        "spec": "doc 22 SS3-SS7 (frozen, commit cacac31)",
        "gate_c5": {"file": GATE_FILE, "harness_calibrated": True},
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "dataset_sha256": dataset_sha,
        "embeddings": {"sha256": emb_sha, "meta": emb_meta},
        "readout_ambiguity_resolution": READOUT_AMBIGUITY_NOTE,
        "config": {"optimizer": "Adam (full autodiff, no SPSA)",
                   "lr": args.lr, "batch": args.batch,
                   "max_epochs": args.epochs, "n_seeds": n_seeds,
                   "model_selection": "in-distribution val ONLY",
                   "arms": arms},
        "param_counts": {arm: models_per_arm[arm][0].param_counts()
                         for arm in arms},
        "per_arm": results_per_arm,
        "gate_statistics": stats,
        "controls": controls,
        "verdicts": verdicts,
        "git_note": git_note(),
        "wall_time_sec": round(time.time() - t_start, 1),
    }
    path = fresh_results_path(RESULTS_BASE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("[exp42] wrote %s  verdicts=%s" % (path, verdicts))


if __name__ == "__main__":
    main()
