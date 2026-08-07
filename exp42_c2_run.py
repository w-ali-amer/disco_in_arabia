# -*- coding: utf-8 -*-
"""exp42_c2_run.py -- control C2 informative null (doc 22 SS6): train the
A2-architecture blocks on the verbset-A restricted dataset, evaluate on
the verbset-B dataset's multi-hop test stories through the fixed
kind-matched bijection (exp42_controls.c2_bijection). Pre-registered pass
criterion: accuracy <= chance + 5pp (0.55). If the borrowed blocks solve
held-out verbs' stories, block content is generic frame machinery, not
verb-specific -- the informative null.

Runs AFTER the main exp42 arms (same C5 gate). Writes
results_exp42_c2.json (fresh-suffix rule). 5 seeds, same protocol as the
main run: Adam, lr 0.005, batch 1, selection on the A-dataset's ID val
split only.
"""

import argparse
import copy
import json
import sys
import time

import torch

import exp42_compiler as comp
import exp42_controls as ctl
from exp42_models import QuantumStoryModel
from exp42_run import (GATE_FILE, check_gate, evaluate, fresh_results_path,
                       git_note, train_seed)

TRAIN_DATA = "stories_exp42_c2_trainA.json"
EVAL_DATA = "stories_exp42_c2_evalB.json"


def load_c2_dataset(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items, meta = data["items"], data["meta"]
    sha = comp.canonical_sha256(items)
    if sha != meta["dataset_sha256"]:
        raise RuntimeError("C2 dataset %s hash mismatch" % path)
    return items, meta, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.005)
    args = ap.parse_args()

    check_gate(smoke=False)                               # same C5 gate
    t0 = time.time()

    items_a, meta_a, sha_a = load_c2_dataset(TRAIN_DATA)
    items_b, meta_b, sha_b = load_c2_dataset(EVAL_DATA)
    angles, emb_meta, emb_sha = comp.load_embeddings()

    # Full-inventory swap flags for the split/bijection rule.
    inv = dict(meta_a["verb_inventory"])
    inv.update(meta_b["verb_inventory"])
    va, vb = ctl.c2_verb_split(inv)
    assert sorted(meta_a["verb_inventory"]) == va, "trainA inventory drift"
    assert sorted(meta_b["verb_inventory"]) == vb, "evalB inventory drift"

    by_a = comp.split_items(items_a)
    data = {k: comp.compile_all(v, angles) for k, v in by_a.items()}
    test_b = [it for it in items_b if it["split"].startswith("test_")]
    compiled_b = comp.compile_all(test_b, angles)

    cfg = {"verbs": va, "lr": args.lr, "batch": 1, "epochs": args.epochs}
    per_seed, evals = [], []
    for seed in range(args.seeds):
        model, detail = train_seed("A2", seed, data, meta_a, {}, cfg)
        detail.pop("test_union_correct", None)
        per_seed.append(detail)
        ev = ctl.c2_eval_disjoint(model, compiled_b, inv)
        ev["seed"] = seed
        evals.append(ev)
        print("[c2] seed %d: trainA val=%.3f  disjoint-eval acc=%.3f "
              "(pass=%s)" % (seed, detail["val_acc"], ev["accuracy"],
                             ev["passed"]), flush=True)

    accs = [e["accuracy"] for e in evals]
    mean_acc = sum(accs) / len(accs)
    results = {
        "experiment": "exp42 control C2 informative null",
        "spec": "doc 22 SS6 C2 (pass = disjoint-eval acc <= 0.55)",
        "gate_c5": {"file": GATE_FILE, "harness_calibrated": True},
        "train_dataset": {"path": TRAIN_DATA, "sha256": sha_a,
                          "verbset_A": va},
        "eval_dataset": {"path": EVAL_DATA, "sha256": sha_b,
                         "verbset_B": vb, "n_test_items": len(compiled_b)},
        "embeddings_sha256": emb_sha,
        "verb_bijection": ctl.c2_bijection(inv),
        "config": {"arm": "A2", "optimizer": "Adam", "lr": args.lr,
                   "batch": 1, "max_epochs": args.epochs,
                   "n_seeds": args.seeds,
                   "model_selection": "trainA ID val only"},
        "per_seed_training": per_seed,
        "per_seed_eval": evals,
        "mean_disjoint_accuracy": mean_acc,
        "c2_passed": bool(mean_acc <= 0.55),
        "git_note": git_note(),
        "wall_time_sec": round(time.time() - t0, 1),
    }
    path = fresh_results_path("results_exp42_c2")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("[c2] wrote %s  mean_disjoint_acc=%.3f  c2_passed=%s"
          % (path, mean_acc, results["c2_passed"]))


if __name__ == "__main__":
    main()
