# -*- coding: utf-8 -*-
"""exp40r_trainer.py -- generic trajectory-recording trainer (exp40 REVIVAL).

WHY THIS EXISTS (the August gap this file closes):
`exp42_run.py`'s train_seed records per-epoch LOSSES and GRAD NORMS but never
per-epoch TRAIN ACCURACY, and the `train_acc` it reports is the clean train
accuracy at the best-VALIDATION epoch -- not at the end of training. In
results_exp42.json this matters: A2 seed 0 reports train_acc 0.4885 with
best_epoch 1, so the headline "at chance" number is an EPOCH-1 number, and
the 30-epoch loss descent (0.7225 -> 0.6389) is invisible in it. Any
diagnosis of the August all-arms-at-chance event needs the trajectory this
module records.

This module NEVER writes results files and NEVER modifies committed code.
It is used by the exp40r smoke battery and the Arabic diagnostic block.
The C5 calibration itself deliberately does NOT use this trainer -- it calls
`exp40_mini_duneau.train_seed` verbatim, so the calibrated August training
path is reproduced rather than re-implemented.

No SPSA anywhere (doc 22 SS3.3, hard rule): exact statevector autodiff only.
"""

import copy
import math
import random
import time

import torch


def grad_global_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return math.sqrt(total)


def evaluate(model, items, answer_of):
    """Clean (no-grad) accuracy over `items`."""
    correct = 0
    with torch.no_grad():
        for it in items:
            vals = model.story_values(it)
            if int(torch.argmax(vals).item()) == answer_of(it):
                correct += 1
    return correct / len(items)


def train_with_trajectory(model, train_items, eval_sets, answer_of, *,
                          seed, epochs, lr, batch_size=1, shuffle_seed=None,
                          log_prefix="", eval_every=1, early_stop_train=None):
    """Train `model`, recording per-epoch CLEAN train accuracy.

    Args:
      model: exposes story_loss(item) -> (loss, pred_int) and
             story_values(item) -> tensor([v_class0, v_class1]).
      train_items: list of items.
      eval_sets: {name: items} evaluated every `eval_every` epochs.
      answer_of: item -> int class index.
      early_stop_train: stop once clean train accuracy >= this (or None).

    Returns a dict of trajectories. Writes nothing to disk.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(shuffle_seed if shuffle_seed is not None
                        else 9000 + seed)

    traj = {
        "epoch": [], "mean_loss": [], "mean_grad_norm": [],
        "running_train_acc": [], "clean_train_acc": [],
        "eval_acc": {k: [] for k in eval_sets},
    }
    first_batch_checked = False
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        order = list(range(len(train_items)))
        rng.shuffle(order)
        losses, gnorms, run_correct = [], [], 0

        for start in range(0, len(order), batch_size):
            batch = [train_items[i] for i in order[start:start + batch_size]]
            opt.zero_grad()
            batch_loss = 0.0
            for it in batch:
                loss, pred = model.story_loss(it)
                batch_loss = batch_loss + loss
                if pred == answer_of(it):
                    run_correct += 1
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            gn = grad_global_norm(model)
            if not first_batch_checked:
                assert gn > 0.0, "zero gradient norm on first batch"
                first_batch_checked = True
            gnorms.append(gn)
            opt.step()
            losses.append(batch_loss.item())

        clean_train = evaluate(model, train_items, answer_of)
        traj["epoch"].append(epoch)
        traj["mean_loss"].append(sum(losses) / len(losses))
        traj["mean_grad_norm"].append(sum(gnorms) / len(gnorms))
        traj["running_train_acc"].append(run_correct / len(train_items))
        traj["clean_train_acc"].append(clean_train)

        evals = {}
        if epoch % eval_every == 0 or epoch == epochs:
            for name, items in eval_sets.items():
                a = evaluate(model, items, answer_of)
                traj["eval_acc"][name].append(a)
                evals[name] = a
        else:
            for name in eval_sets:
                traj["eval_acc"][name].append(None)

        print("%sepoch %3d/%d  loss=%.4f  train(clean)=%.4f  gn=%.3e  %s %.1fs"
              % (log_prefix, epoch, epochs, traj["mean_loss"][-1], clean_train,
                 traj["mean_grad_norm"][-1],
                 " ".join("%s=%.4f" % (k, v) for k, v in evals.items()),
                 time.time() - t0), flush=True)

        if early_stop_train is not None and clean_train >= early_stop_train:
            print("%searly stop: clean train acc %.4f >= %.4f"
                  % (log_prefix, clean_train, early_stop_train), flush=True)
            break

    traj["epochs_run"] = len(traj["epoch"])
    traj["final_clean_train_acc"] = traj["clean_train_acc"][-1]
    traj["max_clean_train_acc"] = max(traj["clean_train_acc"])
    traj["final_eval_acc"] = {
        k: next((v for v in reversed(traj["eval_acc"][k]) if v is not None),
                None)
        for k in eval_sets}
    traj["wall_time_sec"] = round(time.time() - t_start, 1)
    traj["final_state"] = copy.deepcopy(model.state_dict())
    return traj


def strip_state(traj):
    """Trajectory without the torch state_dict (JSON-serializable)."""
    return {k: v for k, v in traj.items() if k != "final_state"}
