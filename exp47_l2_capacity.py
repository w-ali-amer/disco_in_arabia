# -*- coding: utf-8 -*-
"""exp47_l2_capacity.py -- doc 31: is the L2 crossing about the QUANTUM
register, or about capacity?

exp45b crossed the multi-hop gate: rung_learnable_L2_2q = true, A4 at
train 0.8000000000 (margin +0.000000), val 0.750, test 0.757, in 1 of 8
configs, with the sibling seed missing on val by 0.0033.

The tempting reading -- "a two-qubit register per referent is what multi-hop
composition needed" -- is exactly the reading exp44 demolished one rung
down. On L1, B1 sat at 0.537 against A2's 0.799 until the classical arm was
given an UNCONSTRAINED parameterisation, whereupon B5 reached 0.7787 and
tier_q_B5_passed came back FALSE at p=0.120.

A4 carries 264 parameters against A3's 120, on 2K wires rather than K. So
capacity is the most likely explanation for the crossing, and doc 28 SS7
named the comparator in advance: B5 -- unconstrained real 16x16 verb maps on
R^4 referents -- is DIMENSION-MATCHED to a two-qubit register (both span a
4-dimensional referent space) and carries 7.8x A4's per-verb parameters.
The comparison is deliberately unfavourable to A4.

Criteria (doc 31 SS3, frozen):
  b5_learns_L2                          identical gate, identical grid
  l2_crossing_is_register_dimension     iff b5_learns_L2
  l2_crossing_survives_capacity_control iff NOT b5_learns_L2 AND B5 valid

A4 is NOT retrained: its per-config results are read from the committed
exp45b_ckpt/L2_A4_*.json under a sha256 integrity precondition (Rule 6).
"""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time

import torch

import exp42_compiler as comp
from exp42_run import fresh_results_path, git_note, grad_global_norm
from exp44_richer_comparators import make_model

DESIGN_DOC = ("qnlp_private_docs/31_exp47_l2_capacity_matched_comparator.md "
              "(commit 32740f8, FROZEN)")
RUNG = "L2"
ARM = "B5"
CKPT_DIR = "exp47_ckpt"
A4_CKPT_DIR = "exp45b_ckpt"
A4_RESULTS = "results_exp45b_phase1.json"
GRID = [(0.005, 1), (0.005, 8), (0.02, 1), (0.02, 8)]   # frozen Phase-1 grid
N_SEEDS = 2                                             # as A4 received
MAX_EPOCHS = 60
TRAIN_GATE, VAL_GATE = 0.8, 0.7
MEMO_ITEMS, MEMO_EPOCHS, MEMO_BAR = 64, 300, 0.95
F3_BAR = 0.60
AUDIT_PAIRS = 200
RNG_BASE = 47000

SCOPE_NOTE = (
    "doc 31 SS5: NO outcome licenses a quantum-advantage claim. One rung, one "
    "task family, simulation only, 6 verbs, 2 seeds, and a gate A4 cleared by "
    "EXACTLY ZERO margin. A4's L2 pass is knife-edge and does not improve "
    "here: if B5 misses the gate while landing near 0.79 train, the two arms "
    "are practically indistinguishable and that must be said rather than "
    "reporting a binary. Nothing about hardware follows. This is a "
    "learnability contrast at matched referent dimension, NOT a Phase-2 "
    "tier-q protocol; promoting it into one needs a new doc.")

NARROWING_RULE = (
    "doc 31 SS4, frozen: if b5_learns_L2 is true, the L2 crossing is a "
    "CAPACITY result, not a quantum one. rung_learnable_L2_2q stands as "
    "computed -- it was registered as a learnability question about A4 -- but "
    "every statement of it thereafter carries 'a dimension-matched "
    "unconstrained classical arm learns this rung too', exactly as tier-q "
    "narrowed after exp44.")


def evaluate(model, compiled):
    correct = [1 if model.predict(cs) == cs.answer else 0 for cs in compiled]
    return sum(correct) / len(correct), correct


def load_rung():
    with open("stories_exp43b_%s.json" % RUNG, encoding="utf-8") as f:
        d = json.load(f)
    return d["items"], d["meta"]


def train_run(seed, lr, batch, data, verbs, max_epochs,
              train_items=None, select_on_val=True, tag=""):
    torch.manual_seed(seed)
    model = make_model(ARM, verbs, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(RNG_BASE + seed)
    train = train_items if train_items is not None else data["train"]
    val = data["val"] if select_on_val else train
    best = {"val_acc": -1.0, "train_acc": -1.0, "epoch": -1, "state": None}
    steps = 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses, run_correct = [], 0
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
            assert math.isfinite(gn), "non-finite grad s%d" % seed
            opt.step()
            steps += 1
            losses.append(bloss.item())
        val_acc, _ = evaluate(model, val)
        if val_acc > best["val_acc"]:
            ta, _ = evaluate(model, train)
            best = {"val_acc": val_acc, "train_acc": ta, "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
        print("[%s%s s%d lr%.3f b%d] ep %2d/%d loss=%.4f train(run)=%.3f "
              "val=%.3f best@%d(%.3f) %.1fs"
              % (tag, ARM, seed, lr, batch, epoch, max_epochs,
                 sum(losses) / len(losses), run_correct / len(train), val_acc,
                 best["epoch"], best["val_acc"], time.time() - t0), flush=True)
        if best["val_acc"] >= 1.0 and best["train_acc"] >= 1.0:
            break
    model.load_state_dict(best["state"])
    test_acc, _ = (evaluate(model, data["test"]) if select_on_val
                   else (best["train_acc"], []))
    return model, {
        "arm": ARM, "seed": seed, "lr": lr, "batch": batch,
        "epochs_run": epoch, "steps_taken": steps, "best_epoch": best["epoch"],
        "val_acc": best["val_acc"], "train_acc": best["train_acc"],
        "test_acc": test_acc,
        "gate_met": bool(best["train_acc"] >= TRAIN_GATE
                         and best["val_acc"] >= VAL_GATE),
        "train_margin": best["train_acc"] - TRAIN_GATE,
        "val_margin": best["val_acc"] - VAL_GATE}


def init_audit(items, angles, verbs):
    by = {}
    for it in items:
        if it["split"] == "train":
            by.setdefault(it["pair_id"], []).append(it)
    pairs = [p for p in sorted(by) if len(by[p]) == 2][:AUDIT_PAIRS]
    out = {}
    for seed in range(N_SEEDS):
        m = make_model(ARM, verbs, seed)
        same = 0
        with torch.no_grad():
            for p in pairs:
                a, b = sorted(by[p], key=lambda x: x["id"])
                if m.predict(comp.compile_story(a, angles)) == \
                        m.predict(comp.compile_story(b, angles)):
                    same += 1
        out["seed%d" % seed] = same / len(pairs)
    frac = sum(out.values()) / len(out)
    return {"n_pairs": len(pairs), "per_seed": out,
            "frac_twin_pairs_predicting_identically": frac,
            "f3_bar": F3_BAR, "f3_pass": bool(frac <= F3_BAR)}


def memo_subset(items, angles):
    by = {}
    for it in items:
        if it["split"] == "train":
            by.setdefault(it["pair_id"], []).append(it)
    ids = []
    for p in sorted(by):
        if len(by[p]) == 2:
            ids += [x["id"] for x in sorted(by[p], key=lambda x: x["id"])]
        if len(ids) >= MEMO_ITEMS:
            break
    ids = set(ids[:MEMO_ITEMS])
    return comp.compile_all(
        [i for i in items if i["split"] == "train" and i["id"] in ids], angles)


def load_frozen_a4():
    """doc 31 SS2: A4 is NOT retrained. Read its committed L2 per-config
    records under an integrity precondition (Rule 6)."""
    with open(A4_RESULTS, encoding="utf-8") as f:
        res = json.load(f)
    ref = {(r["lr"], r["batch"], r["seed"]): r for r in res["per_rung"]["L2"]}
    recs, checks = [], []
    for (lr, b, s), r in sorted(ref.items()):
        path = os.path.join(A4_CKPT_DIR, "L2_A4_lr%s_b%d_s%d.json" % (lr, b, s))
        ok, sha = False, None
        if os.path.exists(path):
            raw = open(path, "rb").read()
            sha = hashlib.sha256(raw).hexdigest()
            d = json.loads(raw.decode("utf-8"))
            ok = (d["train_acc"] == r["train_acc"]
                  and d["val_acc"] == r["val_acc"])
        checks.append({"config": "lr%s_b%d_s%d" % (lr, b, s), "path": path,
                       "sha256": sha, "matches_results_json": bool(ok)})
        recs.append(r)
    return recs, checks, res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if not args.smoke:
        os.makedirs(CKPT_DIR, exist_ok=True)
    n_seeds = 1 if args.smoke else N_SEEDS
    max_epochs = 2 if args.smoke else MAX_EPOCHS
    grid = GRID[-1:] if args.smoke else GRID
    memo_epochs = 2 if args.smoke else MEMO_EPOCHS

    items, meta = load_rung()
    sha = meta["dataset_sha256"]
    angles, _em, emb_sha = comp.load_embeddings()
    verbs = sorted(meta["verb_inventory"].keys())
    data = {sp: comp.compile_all([i for i in items if i["split"] == sp], angles)
            for sp in ("train", "val", "test")}
    print("[exp47] L2 sha=%s train/val/test=%d/%d/%d verbs=%d"
          % (sha[:12], len(data["train"]), len(data["val"]),
             len(data["test"]), len(verbs)), flush=True)

    a4_recs, a4_checks, a4_res = load_frozen_a4()
    a4_ok = all(c["matches_results_json"] for c in a4_checks)
    a4_dataset_ok = a4_res["dataset_sha256"]["L2"] == sha
    print("[exp47] A4 reuse integrity: configs=%s dataset=%s"
          % (a4_ok, a4_dataset_ok), flush=True)
    if not (a4_ok and a4_dataset_ok):
        print("[exp47] HALT: doc 31 SS2 integrity precondition failed.",
              flush=True)
        return 1

    audit = init_audit(items, angles, verbs)
    print("[exp47] B5 init audit: collapse=%.3f F3(<=%.2f)=%s"
          % (audit["frac_twin_pairs_predicting_identically"], F3_BAR,
             audit["f3_pass"]), flush=True)

    subset = memo_subset(items, angles)
    memo_runs, memo_best = [], 0.0
    for lr, batch in grid:
        for seed in range(2 if not args.smoke else 1):
            _m, d = train_run(seed, lr, batch, data, verbs, memo_epochs,
                              train_items=subset, select_on_val=False,
                              tag="memo:")
            memo_best = max(memo_best, d["train_acc"])
            memo_runs.append({"lr": lr, "batch": batch, "seed": seed,
                              "best_train_acc": d["train_acc"],
                              "meets": bool(d["train_acc"] >= MEMO_BAR)})
    memo = {"n_items": len(subset), "bar": MEMO_BAR, "runs": memo_runs,
            "best_train_acc": memo_best,
            "arm_trainable": bool(memo_best >= MEMO_BAR)}
    print("[exp47] B5 memorisation: best=%.3f trainable=%s"
          % (memo_best, memo["arm_trainable"]), flush=True)

    runs = []
    for lr, batch in grid:
        for seed in range(n_seeds):
            key = "L2_B5_lr%s_b%d_s%d" % (lr, batch, seed)
            ckj = os.path.join(CKPT_DIR, key + ".json")
            ckp = os.path.join(CKPT_DIR, key + ".pt")
            if not args.smoke and os.path.exists(ckj) and os.path.exists(ckp):
                with open(ckj, encoding="utf-8") as f:
                    d = json.load(f)
                print("[ckpt] %s loaded" % key, flush=True)
            else:
                m, d = train_run(seed, lr, batch, data, verbs, max_epochs,
                                 tag="L2:")
                if not args.smoke:
                    with open(ckj, "w", encoding="utf-8") as f:
                        json.dump(d, f, ensure_ascii=False)
                    torch.save(m.state_dict(), ckp)
            runs.append(d)
    b5_learns = bool(any(r["gate_met"] for r in runs))
    b5_valid = bool(memo["arm_trainable"] and audit["f3_pass"])
    print("[exp47] B5 on L2: gate met in %d/%d configs; best train=%.4f "
          "val=%.4f" % (sum(1 for r in runs if r["gate_met"]), len(runs),
                        max(r["train_acc"] for r in runs),
                        max(r["val_acc"] for r in runs)), flush=True)

    if not b5_valid:
        verdicts = {
            "b5_learns_L2": "UNDECIDABLE",
            "l2_crossing_is_register_dimension": "UNDECIDABLE",
            "l2_crossing_survives_capacity_control": "UNDECIDABLE",
            "arm_trainable_B5": memo["arm_trainable"],
            "b5_f3_pass": audit["f3_pass"],
            "note": ("doc 31 SS4 + doc 24 SS4: B5 is an UNTRAINED INSTRUMENT; "
                     "the comparison is UNDECIDABLE, never a win for A4.")}
    else:
        verdicts = {
            "b5_learns_L2": b5_learns,
            "l2_crossing_is_register_dimension": b5_learns,
            "l2_crossing_survives_capacity_control": bool(not b5_learns),
            "arm_trainable_B5": True, "b5_f3_pass": True}

    out = {
        "experiment": "exp47",
        "spec": DESIGN_DOC,
        "question": ("is the L2 crossing about the quantum register or about "
                     "referent-space capacity?"),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__,
        "rung": RUNG, "dataset_sha256": sha, "embeddings_sha256": emb_sha,
        "grid": [{"lr": lr, "batch": b} for lr, b in grid],
        "n_seeds": n_seeds, "max_epochs": max_epochs,
        "gate": {"train": TRAIN_GATE, "val": VAL_GATE},
        "param_counts": {"B5": make_model(ARM, verbs, 0).param_counts(),
                         "A4_reference": a4_res["param_counts"]},
        "a4_reuse": {"retrained": False, "source": A4_RESULTS,
                     "integrity_checks": a4_checks, "integrity_ok": a4_ok,
                     "dataset_matches": a4_dataset_ok},
        "a4_L2_frozen": a4_recs,
        "b5_init_audit_f3": audit,
        "b5_memorisation_control": memo,
        "b5_L2_runs": runs,
        "verdicts": verdicts,
        "meta_notes": {"scope": SCOPE_NOTE, "narrowing_rule": NARROWING_RULE},
        "git_note": git_note(), "wall_time_sec": time.time() - t0,
    }
    print("\n[exp47] VERDICTS")
    for k, v in verdicts.items():
        print("   %-42s %s" % (k, v))
    if args.smoke:
        print("[exp47] smoke -- no results file")
        return 0
    p = fresh_results_path("results_exp47")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[exp47] wrote %s (%.1f min)" % (p, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
