# -*- coding: utf-8 -*-
"""exp43b_phase1.py -- ladder learnability gate, Phase 1 (doc 23, FROZEN).

Per rung (L1 then L2): train A3 (most expressive quantum arm) and B1 (the
decisive classical) only, 2 seeds each, over the frozen config grid
{lr 0.005, 0.02} x {batch 1, 8}, <= 60 epochs.

  rung_learnable = true  iff  ANY arm/config run reaches
      clean train accuracy >= 0.8 AND val accuracy >= 0.7
  (checked each epoch; a run stops early once the gate is met).

The grid exists ONLY to find whether learning is possible (instrument
calibration); it is exhausted as listed and never extended. Model
selection inside a run: none needed for the gate (the gate is an
existence claim over epochs); best-val state is still tracked and its
clean train/test accuracies recorded for completeness.

Phase 2 (full 6-arm comparison at ONE recorded config) is a separate,
explicitly authorized step -- this script never starts it.

Checkpointing: one JSON per (rung, arm, lr, batch, seed) run under
exp43b_ckpt/; finished runs are skipped on relaunch. Results (never
overwritten) -> results_exp43b_phase1.json.
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time

import torch

import exp42_compiler as comp
from exp42_baselines import ClassicalDisCoCirc
from exp42_models import QuantumStoryModel
from exp42_run import fresh_results_path, git_note, grad_global_norm

CKPT_DIR = "exp43b_ckpt"
GRID_LRS = (0.005, 0.02)                                  # FROZEN
GRID_BATCHES = (1, 8)                                     # FROZEN
SEEDS = (0, 1)                                            # FROZEN
MAX_EPOCHS = 60                                           # FROZEN
GATE_TRAIN, GATE_VAL = 0.8, 0.7                           # FROZEN
RUNGS = ("L1", "L2")
ARMS = ("A3", "B1")


def load_rung(rung):
    path = "stories_exp43b_%s.json" % rung
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items, meta = data["items"], data["meta"]
    sha = comp.canonical_sha256(items)
    if sha != meta["dataset_sha256"]:
        raise RuntimeError("rung %s dataset hash mismatch" % rung)
    return items, meta, sha


def evaluate(model, compiled):
    correct = 0
    with torch.no_grad():
        for cs in compiled:
            if model.predict(cs) == cs.answer:
                correct += 1
    return correct / len(compiled)


def make_model(arm, verbs, seed):
    if arm == "A3":
        return QuantumStoryModel("A3", verbs, seed)
    return ClassicalDisCoCirc(verbs, seed, d_ref=2)


def train_run(rung, arm, lr, batch, seed, data, verbs):
    torch.manual_seed(seed)
    model = make_model(arm, verbs, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(43000 + seed)
    train, valset = data["train"], data["val"]

    curves = {"loss": [], "train_run_acc": [], "val_acc": [],
              "grad_norm": []}
    gate_met, gate_epoch = False, None
    gate_train_acc, gate_val_acc = None, None
    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses, gnorms, run_correct = [], [], 0
        for start in range(0, len(order), batch):
            bat = [train[i] for i in order[start:start + batch]]
            opt.zero_grad()
            bloss = 0.0
            for cs in bat:
                loss, pred = model.story_loss(cs)
                bloss = bloss + loss
                if pred == cs.answer:
                    run_correct += 1
            bloss = bloss / len(bat)
            bloss.backward()
            gn = grad_global_norm(model)
            assert math.isfinite(gn), \
                "non-finite grad (%s %s lr%g b%d s%d)" % (rung, arm, lr,
                                                          batch, seed)
            gnorms.append(gn)
            opt.step()
            losses.append(bloss.item())

        val_acc = evaluate(model, valset)
        curves["loss"].append(sum(losses) / len(losses))
        curves["train_run_acc"].append(run_correct / len(train))
        curves["val_acc"].append(val_acc)
        curves["grad_norm"].append(sum(gnorms) / len(gnorms))
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
        line = ("[%s %s lr%g b%d s%d] ep %2d/%d loss=%.4f train(run)=%.3f "
                "val=%.3f gn=%.2e %.1fs"
                % (rung, arm, lr, batch, seed, epoch, MAX_EPOCHS,
                   curves["loss"][-1], curves["train_run_acc"][-1],
                   val_acc, curves["grad_norm"][-1], time.time() - t0))
        print(line, flush=True)
        if val_acc >= GATE_VAL:
            clean_train = evaluate(model, train)
            if clean_train >= GATE_TRAIN:
                gate_met, gate_epoch = True, epoch
                gate_train_acc, gate_val_acc = clean_train, val_acc
                print("[%s %s lr%g b%d s%d] GATE MET at epoch %d "
                      "(train=%.3f val=%.3f)"
                      % (rung, arm, lr, batch, seed, epoch, clean_train,
                         val_acc), flush=True)
                break

    # For the record: best-val state's clean accuracies.
    model.load_state_dict(best["state"])
    rec_train = evaluate(model, train)
    rec_test = evaluate(model, data["test"])
    return {
        "rung": rung, "arm": arm, "lr": lr, "batch": batch, "seed": seed,
        "gate_met": gate_met, "gate_epoch": gate_epoch,
        "gate_train_acc": gate_train_acc, "gate_val_acc": gate_val_acc,
        "epochs_run": len(curves["loss"]),
        "best_val_acc": best["val_acc"], "best_val_epoch": best["epoch"],
        "bestval_clean_train_acc": rec_train,
        "bestval_test_acc": rec_test,
        "curves": curves,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", default="all", choices=("L1", "L2", "all"))
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)

    angles, emb_meta, emb_sha = comp.load_embeddings()
    rungs = RUNGS if args.rung == "all" else (args.rung,)

    all_runs, rung_meta = {}, {}
    for rung in rungs:
        items, meta, sha = load_rung(rung)
        rung_meta[rung] = {"dataset_sha256": sha,
                           "n_items": meta["n_items"]}
        verbs = sorted(meta["verb_inventory"].keys())
        data = {}
        for split in ("train", "val", "test"):
            data[split] = comp.compile_all(
                [i for i in items if i["split"] == split], angles)
        print("[%s] compiled: train=%d val=%d test=%d verbs=%d"
              % (rung, len(data["train"]), len(data["val"]),
                 len(data["test"]), len(verbs)), flush=True)

        runs = []
        for arm in ARMS:
            for lr in GRID_LRS:
                for batch in GRID_BATCHES:
                    for seed in SEEDS:
                        key = "%s_%s_lr%g_b%d_s%d" % (rung, arm, lr,
                                                      batch, seed)
                        ck = os.path.join(CKPT_DIR, key + ".json")
                        if os.path.exists(ck):
                            with open(ck, encoding="utf-8") as f:
                                r = json.load(f)
                            print("[ckpt] %s loaded" % key, flush=True)
                        else:
                            r = train_run(rung, arm, lr, batch, seed,
                                          data, verbs)
                            with open(ck, "w", encoding="utf-8") as f:
                                json.dump(r, f, ensure_ascii=False)
                            print("[ckpt] %s saved (gate_met=%s)"
                                  % (key, r["gate_met"]), flush=True)
                        runs.append(r)
        all_runs[rung] = runs

    verdicts = {"rung_learnable_%s" % rung:
                bool(any(r["gate_met"] for r in runs))
                for rung, runs in all_runs.items()}
    results = {
        "experiment": "exp43b Phase 1 learnability gate",
        "spec": "doc 23 (commit ef15d76, FROZEN): A3+B1, 2 seeds, grid "
                "{lr 0.005, 0.02} x {batch 1, 8}, <=60 epochs; "
                "rung_learnable iff any run reaches train>=0.8 AND "
                "val>=0.7; grid never extended",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "rung_datasets": rung_meta,
        "embeddings_sha256": emb_sha,
        "frozen_grid": {"lrs": GRID_LRS, "batches": GRID_BATCHES,
                        "seeds": SEEDS, "max_epochs": MAX_EPOCHS,
                        "arms": ARMS},
        "runs": all_runs,
        "verdicts": verdicts,
        "phase2_note": "Phase 2 (full comparison at one recorded config) "
                       "requires explicit authorization; never started by "
                       "this script.",
        "git_note": git_note(),
        "wall_time_sec": round(time.time() - t0, 1),
    }
    path = fresh_results_path("results_exp43b_phase1")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("[exp43b] wrote %s  verdicts=%s" % (path, verdicts))


if __name__ == "__main__":
    main()
