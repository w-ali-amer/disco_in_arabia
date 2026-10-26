# -*- coding: utf-8 -*-
"""exp43b_hop_split_rerun.py -- hop-split diagnostic re-run (doc 25, FROZEN).

Phase 1 closed L2 as not learnable (rung_learnable_L2=false, a threshold
miss at train 0.721 / val 0.703), but its checkpoints never persisted
weights, so WHERE the above-chance signal lives (one-hop carry-over vs
partial two-hop learning) cannot be recomputed from committed artifacts
(see results_exp43b_hop_split.json "limitation").

This re-runs A3 on L2 ONLY, at the Phase-1 grid point lr 0.005 / batch 1,
seeds 0 and 1 (seed 1 is a CONTRAST, not a second sample -- doc 25 SS2),
<=60 epochs, identical optimiser/gate logic -- the single change is that
the best-val state_dict is SAVED, then evaluated on train/val/test
partitioned by hop_count.

THIS IS A DIAGNOSTIC, NOT A COMPARISON. It carries NO pass/fail verdict
and changes no Phase-1 verdict; rung_learnable_L2=false stays closed.
The interpretation bands (doc 25 SS3, frozen BEFORE the numbers) are
recorded as the frozen READING of the test two-hop subset (n=150) at the
best-val epoch:  <=0.55 no two-hop learning | 0.55-0.60 inconclusive |
>=0.60 partial two-hop learning below the gate.

B1 is excluded (doc 25 SS2): its Phase-1 L2 runs are confounded by the
initialisation doc 24 SS2 documents; splitting a confounded null by hop
count would produce a number with no interpretation.

Cross-machine honesty (doc 25 SS4): this runs on the Mac; Phase 1 ran on
WSL. Trajectory parity is NOT claimed (doc 24 SS8 P6); this is a fresh
run of the same configuration, never a reproduction. If its aggregate
lands >0.05 from Phase 1's, that is reported prominently.

Output (never overwritten) -> results_exp43b_hop_split_rerun.json.
"""

import copy
import json
import math
import os
import platform
import random
import sys
import time

import torch

import exp42_compiler as comp
from exp42_models import QuantumStoryModel
from exp42_run import fresh_results_path, git_note, grad_global_norm

RUNG = "L2"
ARM = "A3"
LR = 0.005                                                # FROZEN (doc 25)
BATCH = 1                                                 # FROZEN
SEEDS = (0, 1)                                            # FROZEN
MAX_EPOCHS = 60                                           # FROZEN (Phase 1)
GATE_TRAIN, GATE_VAL = 0.8, 0.7                           # FROZEN (Phase 1)
STATE_FMT = "exp43b_hop_rerun_s%d.pt"

BANDS = [
    ("<=0.55", "No two-hop learning. L2's signal is one-hop carry-over. "
               "The multi-hop wall is total at this budget, and doc 23's "
               "curriculum arm has no two-hop foothold to build on."),
    ("0.55-0.60", "Inconclusive. Reported as inconclusive, not as weak "
                  "evidence for either side."),
    (">=0.60", "Partial two-hop learning exists below the gate. The wall "
               "is a budget/sample-efficiency wall at two hops rather "
               "than a representational one -- which would sit alongside "
               "exp43a's representability_open=true rather than against "
               "it."),
]

# Phase-1 aggregates for the SAME grid points (results_exp43b_phase1.json,
# run on WSL) -- used only for the doc 25 SS4 divergence report.
PHASE1_REF = {
    0: {"best_val_acc": 0.7033333333333334,
        "bestval_clean_train_acc": 0.7283333333333334,
        "bestval_test_acc": 0.63},
    1: {"best_val_acc": 0.5333333333333333,
        "bestval_clean_train_acc": 0.5733333333333334,
        "bestval_test_acc": 0.5},
}
L1_TEST_REF = 0.780        # A3's best-val test accuracy on L1 (Phase 1)


def band_reading(acc):
    if acc <= 0.55:
        return BANDS[0]
    if acc < 0.60:
        return BANDS[1]
    return BANDS[2]


def evaluate(model, compiled):
    correct = 0
    with torch.no_grad():
        for cs in compiled:
            if model.predict(cs) == cs.answer:
                correct += 1
    return correct / len(compiled)


def eval_by_hop(model, compiled):
    tally = {}
    with torch.no_grad():
        for cs in compiled:
            ok = int(model.predict(cs) == cs.answer)
            c, n = tally.get(cs.hop_count, (0, 0))
            tally[cs.hop_count] = (c + ok, n + 1)
    out = {}
    for hop, (c, n) in sorted(tally.items()):
        out["hop_%d" % hop] = {"accuracy": c / n, "n_correct": c, "n": n}
    out["aggregate"] = {"accuracy": sum(c for c, _ in tally.values())
                        / sum(n for _, n in tally.values()),
                        "n": sum(n for _, n in tally.values())}
    return out


def train_run(seed, data, verbs):
    """exp43b_phase1.train_run semantics verbatim; the ONLY change is
    that the best-val state_dict is kept and saved (doc 25 SS2)."""
    torch.manual_seed(seed)
    model = QuantumStoryModel(ARM, verbs, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = random.Random(43000 + seed)
    train, valset = data["train"], data["val"]

    curves = {"loss": [], "train_run_acc": [], "val_acc": [], "grad_norm": []}
    gate_met, gate_epoch = False, None
    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses, gnorms, run_correct = [], [], 0
        for start in range(0, len(order), BATCH):
            bat = [train[i] for i in order[start:start + BATCH]]
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
            assert math.isfinite(gn), "non-finite grad (s%d)" % seed
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
        print("[%s %s lr%g b%d s%d] ep %2d/%d loss=%.4f train(run)=%.3f "
              "val=%.3f gn=%.2e %.1fs"
              % (RUNG, ARM, LR, BATCH, seed, epoch, MAX_EPOCHS,
                 curves["loss"][-1], curves["train_run_acc"][-1], val_acc,
                 curves["grad_norm"][-1], time.time() - t0), flush=True)
        if val_acc >= GATE_VAL:
            clean_train = evaluate(model, train)
            if clean_train >= GATE_TRAIN:
                gate_met, gate_epoch = True, epoch
                print("[s%d] GATE MET at epoch %d (train=%.3f val=%.3f)"
                      % (seed, epoch, clean_train, val_acc), flush=True)
                break

    model.load_state_dict(best["state"])
    torch.save({"state_dict": best["state"], "seed": seed,
                "best_val_acc": best["val_acc"],
                "best_val_epoch": best["epoch"]}, STATE_FMT % seed)
    print("[s%d] best-val state saved -> %s" % (seed, STATE_FMT % seed),
          flush=True)

    hop_eval = {sp: eval_by_hop(model, data[sp])
                for sp in ("train", "val", "test")}
    return {
        "rung": RUNG, "arm": ARM, "lr": LR, "batch": BATCH, "seed": seed,
        "gate_met": gate_met, "gate_epoch": gate_epoch,
        "epochs_run": len(curves["loss"]),
        "best_val_acc": best["val_acc"], "best_val_epoch": best["epoch"],
        "bestval_clean_train_acc": hop_eval["train"]["aggregate"]["accuracy"],
        "bestval_test_acc": hop_eval["test"]["aggregate"]["accuracy"],
        "bestval_eval_by_hop": hop_eval,
        "state_file": STATE_FMT % seed,
        "curves": curves,
    }


def main():
    t0 = time.time()
    with open("stories_exp43b_%s.json" % RUNG, encoding="utf-8") as f:
        raw = json.load(f)
    items, meta = raw["items"], raw["meta"]
    sha = comp.canonical_sha256(items)
    assert sha == meta["dataset_sha256"], "L2 dataset hash mismatch"
    angles, _emb_meta, emb_sha = comp.load_embeddings()
    verbs = sorted(meta["verb_inventory"].keys())
    data = {sp: comp.compile_all([i for i in items if i["split"] == sp],
                                 angles) for sp in ("train", "val", "test")}
    print("[rerun] compiled: train=%d val=%d test=%d verbs=%d"
          % (len(data["train"]), len(data["val"]), len(data["test"]),
             len(verbs)), flush=True)

    runs = [train_run(seed, data, verbs) for seed in SEEDS]

    readings, divergence = {}, {}
    for r in runs:
        seed = r["seed"]
        two_hop = r["bestval_eval_by_hop"]["test"]["hop_2"]["accuracy"]
        one_hop = r["bestval_eval_by_hop"]["test"]["hop_1"]["accuracy"]
        band, text = band_reading(two_hop)
        readings["seed_%d" % seed] = {
            "role": "primary (best Phase-1 config seed)" if seed == 0
                    else "contrast, not a second sample (doc 25 SS2)",
            "test_two_hop_accuracy": two_hop,
            "test_two_hop_n": r["bestval_eval_by_hop"]["test"]["hop_2"]["n"],
            "band": band,
            "frozen_reading": text,
            "test_one_hop_accuracy": one_hop,
            "one_hop_L2_vs_L1_reference": {
                "L1_bestval_test_acc_phase1": L1_TEST_REF,
                "delta": round(one_hop - L1_TEST_REF, 6),
                "note": "doc 25 SS3: if one-hop on L2 materially exceeds "
                        "L1's, mixed-difficulty training would be helping "
                        "the easy half."},
        }
        div = {}
        for k, ref in PHASE1_REF[seed].items():
            now = r[k]
            div[k] = {"phase1_wsl": ref, "this_run_mac": now,
                      "abs_diff": round(abs(now - ref), 6),
                      "exceeds_0.05": bool(abs(now - ref) > 0.05)}
        divergence["seed_%d" % seed] = div

    results = {
        "experiment": "exp43b hop-split diagnostic re-run",
        "spec": "doc 25 (qnlp_private_docs/25, FROZEN): A3 on L2 only, "
                "lr 0.005 / batch 1, seeds 0 and 1, <=60 epochs identical "
                "to Phase 1; best-val state saved and evaluated by "
                "hop_count; NO verdict -- frozen interpretation bands "
                "recorded as the reading only",
        "diagnostic_not_comparison": "carries NO pass/fail verdict; "
            "rung_learnable_L2=false (results_exp43b_phase1.json) stays "
            "closed regardless of this split",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "dataset_sha256": sha,
        "embeddings_sha256": emb_sha,
        "hop_mix_note": "L2 splits are exactly 50/50 one-hop/two-hop "
                        "(results_exp43b_hop_split.json)",
        "frozen_bands_test_two_hop_n150": [
            {"band": b, "reading": t} for b, t in BANDS],
        "runs": runs,
        "frozen_readings": readings,
        "cross_machine_divergence_vs_phase1": {
            "note": "doc 25 SS4: Phase 1 ran on WSL, this ran on the Mac; "
                    "trajectory parity is NOT claimed (doc 24 SS8 P6); "
                    "this is a fresh run of the same configuration, not a "
                    "reproduction. Divergences >0.05 are reported "
                    "prominently.",
            "per_seed": divergence,
            "any_exceeds_0.05": bool(any(
                d["exceeds_0.05"] for s in divergence.values()
                for d in s.values())),
        },
        "b1_exclusion_note": "B1 excluded per doc 25 SS2: its Phase-1 L2 "
            "null is confounded by the initialisation documented in doc "
            "24 SS2; splitting a confounded null by hop count would "
            "produce a number with no interpretation.",
        "git_note": git_note(),
        "wall_time_sec": round(time.time() - t0, 1),
    }
    path = fresh_results_path("results_exp43b_hop_split_rerun")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("[rerun] wrote %s" % path)
    for s, rd in readings.items():
        print("[rerun] %s: test two-hop=%.3f band=%s | one-hop=%.3f "
              "(L1 ref %.3f)"
              % (s, rd["test_two_hop_accuracy"], rd["band"],
                 rd["test_one_hop_accuracy"], L1_TEST_REF))


if __name__ == "__main__":
    main()
