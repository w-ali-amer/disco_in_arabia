# -*- coding: utf-8 -*-
"""exp48_l2_phase2.py -- doc 32: Phase-2 comparison at L2 (A4 vs B5).

exp47 found an 8-point generalisation gap on the multi-hop rung: B5
(unconstrained real 16x16 on R^4 referents, 256 params/verb) drives L2
training accuracy to ~0.99 and then misses the val bar in all 8 configs,
while A4 (2-qubit register, 33 params/verb) reaches val 0.750 / test 0.757.
That is DESCRIPTIVE ONLY -- exp47 registered a learnability gate, not a
comparison, and 2 seeds cannot carry a significance test.

exp44 is the precedent: a Phase-1-scale gap that looked decisive narrowed to
p=0.120 once the comparator got a fair parameterisation and the full
protocol. The gap is not a result until it is tested as one.

Criteria (doc 32 SS3, frozen):
  tier_m_passed      A4 vs B5 one-sided paired permutation p<0.05 on
                     seed-averaged per-item correctness, AND
                     content_earned_A4
  content_earned_A4  exhaustive 719-permutation C1 trained-dial-swap;
                     original seed-mean STRICTLY above the 95th percentile
  a5_sentinel_ok     every A5 seed's test accuracy <= 0.55, else NO result

Config selection, resolved in doc 32 SS2 BEFORE any Phase-2 number existed:
  A4 -> lr 0.02, batch 8 (mean best-val 0.7233 vs next-best 0.600)
  B5 -> the FULL frozen grid with per-seed val-only selection, the exp44 F2
        relaxation, deliberately in the comparator's favour.

Seeds 0-1 are reused from committed checkpoints; 2-4 are new. Reused
checkpoints are RE-EVALUATED by forward pass rather than trusting stored
scalars, and the recomputed test accuracy must match what the source run
recorded (integrity precondition, Rule 6).
"""

import argparse
import copy
import itertools
import json
import math
import os
import platform
import random
import sys
import time

import torch

import exp42_compiler as comp
import exp42_sim_adapter as sim
from exp42_run import (fresh_results_path, git_note, grad_global_norm,
                       paired_permutation_pvalue)
from exp44_richer_comparators import make_model as make_classical
from exp45_model import TwoQubitReferentModel

DESIGN_DOC = ("qnlp_private_docs/32_exp48_l2_phase2_comparison.md "
              "(commit 415f6bc, FROZEN)")
RUNG = "L2"
CKPT_DIR = "exp48_ckpt"
A4_CONFIG = (0.02, 8)                      # doc 32 SS2, resolved pre-run
A5_CONFIG = (0.02, 8)
B5_GRID = [(0.005, 1), (0.005, 8), (0.02, 1), (0.02, 8)]
N_SEEDS = 5
MAX_EPOCHS = 60
MIN_STEPS = 4500                           # doc 32 SS2 F1
SENTINEL_BAR = 0.55
RNG_BASE = 48000
REUSE = {("A4", 0): "exp45b_ckpt/L2_A4_lr0.02_b8_s0",
         ("A4", 1): "exp45b_ckpt/L2_A4_lr0.02_b8_s1"}

SCOPE_NOTE = (
    "doc 32 SS5: NOT quantum advantage. One rung, one task family, one "
    "dataset, simulation only, 6 verbs, 5 seeds. B5 is dimension-matched and "
    "unconstrained -- a stronger comparator than B1 ever was -- so a positive "
    "licenses only 'the two-qubit register class generalises on this rung "
    "where the dimension-matched classical linear class does not, at matched "
    "referent dimension and with the comparator given more parameters and "
    "more search'. A4's L2 gate pass was by EXACTLY ZERO margin on train and "
    "this doc does not repair that: if tier-m passes, the underlying "
    "learnability remains knife-edge and both facts travel together. B5 is "
    "not 'the classical method' -- sequence models and anything with a "
    "different inductive bias are untested here. No hardware reading follows.")

ASYMMETRY_NOTE = (
    "doc 32 SS2, DELIBERATE and in the COMPARATOR's favour: A4 runs at ONE "
    "frozen config while B5 gets the full 4-config grid with per-seed val-only "
    "selection. B5's four configs sat within 0.005 of each other on Phase-1 "
    "mean best-val, so fixing one would be near-arbitrary; giving it all four "
    "can only raise its score. Must be stated in every report (Rule 5).")

CONTENT_RULE = (
    "doc 32 SS4, frozen: if content_earned_A4 is FALSE, A4 solves L2 without "
    "using verb identity. That is the MORE IMPORTANT finding of the two, it "
    "leads the write-up, and it bears retroactively on how every A4 result is "
    "described.")


def evaluate(model, compiled):
    correct = [1 if model.predict(cs) == cs.answer else 0 for cs in compiled]
    return sum(correct) / len(correct), correct


def make_arm(arm, verbs, seed):
    if arm in ("A4", "A5"):
        return TwoQubitReferentModel(arm, verbs, seed)
    return make_classical("B5", verbs, seed)


def train_run(arm, seed, lr, batch, data, verbs, max_epochs, tag=""):
    torch.manual_seed(seed)
    model = make_arm(arm, verbs, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(RNG_BASE + seed)
    train = data["train"]
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
            assert math.isfinite(gn), "non-finite grad %s s%d" % (arm, seed)
            opt.step()
            steps += 1
            losses.append(bloss.item())
        val_acc, _ = evaluate(model, data["val"])
        if val_acc > best["val_acc"]:
            ta, _ = evaluate(model, train)
            best = {"val_acc": val_acc, "train_acc": ta, "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
        print("[%s%s s%d lr%.3f b%d] ep %2d/%d loss=%.4f train(run)=%.3f "
              "val=%.3f best@%d(%.3f) %.1fs"
              % (tag, arm, seed, lr, batch, epoch, max_epochs,
                 sum(losses) / len(losses), run_correct / len(train), val_acc,
                 best["epoch"], best["val_acc"], time.time() - t0), flush=True)
    assert steps >= MIN_STEPS or batch == 1 or max_epochs < MAX_EPOCHS, \
        "F1: %s s%d took %d steps < %d" % (arm, seed, steps, MIN_STEPS)
    model.load_state_dict(best["state"])
    test_acc, test_correct = evaluate(model, data["test"])
    return model, {
        "arm": arm, "seed": seed, "lr": lr, "batch": batch,
        "epochs_run": epoch, "steps_taken": steps, "best_epoch": best["epoch"],
        "val_acc": best["val_acc"], "train_acc": best["train_acc"],
        "test_acc": test_acc, "test_correct": test_correct,
        "train_margin": best["train_acc"] - 0.8,
        "val_margin": best["val_acc"] - 0.7}


def obtain(arm, seed, lr, batch, data, verbs, max_epochs, smoke):
    """Load a committed checkpoint if one exists, else train. Reused states
    are ALWAYS re-evaluated by forward pass; the recomputed test accuracy
    must match what the source run recorded (doc 32 SS2, Rule 6)."""
    integrity = None
    src = REUSE.get((arm, seed))
    if src and os.path.exists(src + ".pt") and not smoke:
        with open(src + ".json", encoding="utf-8") as f:
            ref = json.load(f)
        m = make_arm(arm, verbs, seed)
        m.load_state_dict(torch.load(src + ".pt", weights_only=True))
        test_acc, test_correct = evaluate(m, data["test"])
        integrity = {"source": src, "recomputed_test_acc": test_acc,
                     "source_test_acc": ref["test_acc"],
                     "matches": bool(abs(test_acc - ref["test_acc"]) < 1e-12)}
        d = {"arm": arm, "seed": seed, "lr": ref["lr"], "batch": ref["batch"],
             "epochs_run": ref["epochs_run"], "steps_taken": ref["steps_taken"],
             "best_epoch": ref["best_epoch"], "val_acc": ref["val_acc"],
             "train_acc": ref["train_acc"], "test_acc": test_acc,
             "test_correct": test_correct,
             "train_margin": ref["train_acc"] - 0.8,
             "val_margin": ref["val_acc"] - 0.7, "reused_from": src}
        print("[reuse] %s s%d from %s (test %.4f, integrity %s)"
              % (arm, seed, src, test_acc, integrity["matches"]), flush=True)
        return m, d, integrity

    key = "%s_%s_lr%s_b%d_s%d" % (RUNG, arm, lr, batch, seed)
    ckj, ckp = (os.path.join(CKPT_DIR, key + ".json"),
                os.path.join(CKPT_DIR, key + ".pt"))
    if not smoke and os.path.exists(ckj) and os.path.exists(ckp):
        with open(ckj, encoding="utf-8") as f:
            d = json.load(f)
        m = make_arm(arm, verbs, seed)
        m.load_state_dict(torch.load(ckp, weights_only=True))
        print("[ckpt] %s loaded" % key, flush=True)
        return m, d, None
    m, d = train_run(arm, seed, lr, batch, data, verbs, max_epochs,
                     tag=RUNG + ":")
    if not smoke:
        with open(ckj, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        torch.save(m.state_dict(), ckp)
    return m, d, None


# --------------------------------------------------------------- C1 (A4)
def _fast_eval_2q(model, compiled, vmap=None):
    vm = vmap or {}
    with torch.no_grad():
        mats = {v: model.verb_matrix(vm.get(v, v)) for v in model.verbs}
        effs = [model.q_pos.matrix().conj().T, model.q_neg.matrix().conj().T]
        n_ok = 0
        for cs in compiled:
            psi = sim.init_state(2 * cs.K)
            for w, _n, ang in cs.intro:
                psi = sim.apply_gate(psi, sim.ry(ang), (2 * w,))
            for v, wa, wp in cs.events:
                psi = sim.apply_gate(psi, mats[v],
                                     (2 * wa, 2 * wa + 1, 2 * wp, 2 * wp + 1))
            vseq, qa, qp = cs.question
            qw = (2 * qa, 2 * qa + 1, 2 * qp, 2 * qp + 1)
            for qv in reversed(vseq):
                psi = sim.apply_gate(psi, mats[qv].conj().T, qw)
            vals = [sim.prob_all_zero_on(sim.apply_gate(psi, E, qw), qw)
                    for E in effs]
            n_ok += ((0 if vals[0].item() >= vals[1].item() else 1) == cs.answer)
    return n_ok / len(compiled)


def c1_exhaustive_2q(models, compiled, verbs, cap=None):
    """C1 trained-dial-swap on the two-qubit-register wiring: permute the
    verb->block ASSIGNMENT consistently across story events and the question
    verb. Exhaustive over the 719 non-identity permutations of 6 verbs."""
    perms = [p for p in itertools.permutations(verbs) if list(p) != list(verbs)]
    if cap:
        perms = perms[:cap]
    orig = sum(_fast_eval_2q(m, compiled) for m in models) / len(models)
    accs = []
    for i, p in enumerate(perms):
        vmap = dict(zip(verbs, p))
        accs.append(sum(_fast_eval_2q(m, compiled, vmap)
                        for m in models) / len(models))
        if (i + 1) % 100 == 0:
            print("[C1] %d/%d permutations" % (i + 1, len(perms)), flush=True)
    s = sorted(accs)
    q95 = s[int(0.95 * len(perms)) - 1]
    return {"control": "C1 trained-dial-swap (exhaustive, 6 verbs, 2q wiring)",
            "n_permutations": len(perms), "original_seed_mean_acc": orig,
            "perm_95th_percentile": q95, "perm_max": s[-1],
            "perm_mean": sum(accs) / len(accs),
            "content_earned": bool(orig > q95)}


def seed_mean_correct(details):
    vecs = [d["test_correct"] for d in details]
    n = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(n)]


def compare(x, y):
    mx, my = seed_mean_correct(x), seed_mean_correct(y)
    diff = [a - b for a, b in zip(mx, my)]
    obs, p = paired_permutation_pvalue(diff)
    return {"mean_acc_x": sum(mx) / len(mx), "mean_acc_y": sum(my) / len(my),
            "mean_paired_diff": obs, "one_sided_p": p, "n_items": len(diff)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if not args.smoke:
        os.makedirs(CKPT_DIR, exist_ok=True)
    n_seeds = 1 if args.smoke else N_SEEDS
    max_epochs = 2 if args.smoke else MAX_EPOCHS
    b5_grid = B5_GRID[-1:] if args.smoke else B5_GRID
    cap = 20 if args.smoke else None

    with open("stories_exp43b_%s.json" % RUNG, encoding="utf-8") as f:
        blob = json.load(f)
    items, meta = blob["items"], blob["meta"]
    sha = meta["dataset_sha256"]
    angles, _em, emb_sha = comp.load_embeddings()
    verbs = sorted(meta["verb_inventory"].keys())
    data = {sp: comp.compile_all([i for i in items if i["split"] == sp], angles)
            for sp in ("train", "val", "test")}
    print("[exp48] L2 sha=%s train/val/test=%d/%d/%d"
          % (sha[:12], len(data["train"]), len(data["val"]),
             len(data["test"])), flush=True)

    integrity = []
    # ---- A4 at its single frozen config
    a4, a4_models = [], []
    for s in range(n_seeds):
        m, d, ig = obtain("A4", s, A4_CONFIG[0], A4_CONFIG[1], data, verbs,
                          max_epochs, args.smoke)
        a4.append(d)
        a4_models.append(m)
        if ig:
            integrity.append(ig)
    # ---- A5 sentinel
    a5 = []
    for s in range(n_seeds):
        _m, d, _ = obtain("A5", s, A5_CONFIG[0], A5_CONFIG[1], data, verbs,
                          max_epochs, args.smoke)
        a5.append(d)
    # ---- B5 over the full grid, per-seed val-only selection
    b5, b5_all = [], []
    for s in range(n_seeds):
        cands = []
        for lr, b in b5_grid:
            _m, d, _ = obtain("B5", s, lr, b, data, verbs, max_epochs,
                              args.smoke)
            cands.append(d)
            b5_all.append({k: v for k, v in d.items() if k != "test_correct"})
        best = min(cands, key=lambda c: (-c["val_acc"], c["lr"], c["batch"]))
        b5.append(best)
        print("[exp48] B5 s%d selected lr=%.3f b=%d val=%.3f test=%.3f"
              % (s, best["lr"], best["batch"], best["val_acc"],
                 best["test_acc"]), flush=True)

    integrity_ok = all(i["matches"] for i in integrity) if integrity else True
    a5_accs = [d["test_acc"] for d in a5]
    a5_ok = all(a <= SENTINEL_BAR for a in a5_accs)
    print("[exp48] A5 sentinel test accs=%s -> ok=%s"
          % ([round(a, 4) for a in a5_accs], a5_ok), flush=True)

    stats = {"A4_vs_B5": compare(a4, b5)}
    print("[exp48] A4 %.4f vs B5 %.4f  diff=%.4f  p=%.5f"
          % (stats["A4_vs_B5"]["mean_acc_x"], stats["A4_vs_B5"]["mean_acc_y"],
             stats["A4_vs_B5"]["mean_paired_diff"],
             stats["A4_vs_B5"]["one_sided_p"]), flush=True)

    print("[exp48] C1 on A4 (exhaustive)...", flush=True)
    c1 = c1_exhaustive_2q(a4_models, data["test"], verbs, cap)
    print("[exp48] C1: orig=%.4f q95=%.4f content_earned=%s"
          % (c1["original_seed_mean_acc"], c1["perm_95th_percentile"],
             c1["content_earned"]), flush=True)

    if not a5_ok:
        verdicts = {"a5_sentinel_ok": False,
                    "tier_m_passed": "NOT REPORTED",
                    "note": ("doc 32 SS4: the sentinel failed, so the rung or "
                             "readout leaks at this register size. No exp48 "
                             "result is reported; the finding is the leak.")}
    else:
        p = stats["A4_vs_B5"]["one_sided_p"]
        verdicts = {
            "a5_sentinel_ok": True,
            "tier_m_p": p,
            "content_earned_A4": c1["content_earned"],
            "tier_m_passed": bool(p < 0.05 and c1["content_earned"]),
            "reuse_integrity_ok": integrity_ok}

    out = {
        "experiment": "exp48", "spec": DESIGN_DOC,
        "question": ("does the L2 generalisation gap between A4 and the "
                     "dimension-matched B5 survive the full Phase-2 protocol?"),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__,
        "rung": RUNG, "dataset_sha256": sha, "embeddings_sha256": emb_sha,
        "n_seeds": n_seeds, "max_epochs": max_epochs,
        "a4_config": {"lr": A4_CONFIG[0], "batch": A4_CONFIG[1]},
        "b5_grid": [{"lr": lr, "batch": b} for lr, b in b5_grid],
        "param_counts": {"A4": make_arm("A4", verbs, 0).param_counts(),
                         "B5": make_classical("B5", verbs, 0).param_counts(),
                         "A5": make_arm("A5", verbs, 0).param_counts()},
        "reuse_integrity": integrity,
        "per_arm": {"A4": [{k: v for k, v in d.items() if k != "test_correct"}
                           for d in a4],
                    "B5": [{k: v for k, v in d.items() if k != "test_correct"}
                           for d in b5],
                    "A5": [{k: v for k, v in d.items() if k != "test_correct"}
                           for d in a5]},
        "b5_all_configs": b5_all,
        "gate_statistics": stats,
        "controls": {"C1_A4": c1},
        "verdicts": verdicts,
        "meta_notes": {"scope": SCOPE_NOTE, "search_asymmetry": ASYMMETRY_NOTE,
                       "content_rule": CONTENT_RULE},
        "git_note": git_note(), "wall_time_sec": time.time() - t0,
    }
    print("\n[exp48] VERDICTS")
    for k, v in verdicts.items():
        print("   %-24s %s" % (k, v))
    if args.smoke:
        print("[exp48] smoke -- no results file")
        return 0
    p = fresh_results_path("results_exp48")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[exp48] wrote %s (%.1f min)" % (p, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
