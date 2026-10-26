# -*- coding: utf-8 -*-
"""exp43b_phase2.py -- Phase 2 comparison on L1 (docs 23+24, FROZEN).

Authorized by exp43c's doc 24 SS5.4 outcome (results_exp43c.json, row
b1_matched_init_learnable_L1=false / arm_trainable_B1=true): "The
asymmetry survives its strongest challenge. Phase 2 proceeds; the audit
and this control are reported alongside every comparative number as the
reason a reader should believe it."

Protocol (doc 23 Phase 2, as bound by doc 24 SS4-SS7):
- Rung L1 (fixed, doc 24 SS7). ONE config, recorded in doc 24 SS7 before
  Phase-2 results existed: lr 0.02, batch 8. <=60 epochs, 5 seeds,
  val-only selection.
- Arms {A1, A2, A3, B1, B2, B3}. B1 = AS SHIPPED: exp43c showed the
  matched init passes F3 yet still cannot learn L1 (0/8), and B1-as-
  shipped passed the SS5.2 positive control (3/8 memorisation), so per
  SS5.4 it remains the tier-q comparator. A1 = leak sentinel (must stay
  at chance). B2 trains WITH name augmentation (the exp43c disable was
  control-only). B3 runs for the record but is an UNTRAINED INSTRUMENT
  on this rung (failed SS5.2 structurally: bag-identical twins); per doc
  24 SS4 + F5 that label travels with its numbers and it enters NO
  comparison.
- F1 equal optimiser steps: all arms run the same config, so step counts
  are equal by construction (verified + recorded); the only early stop
  is at train=val=1.0 (inherited from the exp42 runner), where further
  steps cannot matter.
- tier-q (doc 23, unchanged): A2 vs B1, one-sided paired permutation
  p<0.05 on the L1 test split (300 items), seed-mean per-item pairing;
  tier_q_passed additionally requires C1 content_earned on A2 (the exp42
  convention -- strictly harder, never looser).
- tier-q' (registered in doc 24 SS7): A3 vs B1, same test and bar,
  separate named boolean; its frozen scope caveat is quoted in the
  results JSON and can never license a quantum-advantage reading.
- tier-s: vs B2 only (B3 excluded per F5).
- C1 trained-dial-swap: 6 verbs -> ALL 719 non-identity permutations
  (exhaustive; the exp42 sampling rule was for >6 verbs), same
  seed-mean rule, original strictly above the 95th percentile.
- C3 order-invariance: machine-precision statevector identity for the
  structured arms; behavioural delta for B2.
- Pre-registered interpretation (doc 24 SS7, quoted): Phase 1 ran only
  A3 and B1, so whether A2 learns L1 is unknown. If A2 fails L1 while A3
  succeeds, the finding is about EXPRESSIVITY (universal vs restricted
  block class), not quantum-vs-classical, and tier-q is reported as
  failed on its own terms.

Checkpointing: per (arm, seed) JSON + state .pt under
exp43b_phase2_ckpt/; finished runs are skipped on relaunch. Results
(never overwritten) -> results_exp43b_phase2.json.
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
import exp42_controls as ctl
import exp42_sim_adapter as sim
from exp42_baselines import (BagBaseline, ClassicalDisCoCirc, SeqBaseline,
                             build_vocab, sample_name_mapping)
from exp42_models import QuantumStoryModel
from exp42_run import (fresh_results_path, git_note, grad_global_norm,
                       paired_permutation_pvalue)

CKPT_DIR = "exp43b_phase2_ckpt"
RUNG = "L1"
LR, BATCH = 0.02, 8                       # FROZEN, recorded in doc 24 SS7
N_SEEDS = 5                               # FROZEN (doc 23)
MAX_EPOCHS = 60                           # FROZEN (doc 23 Phase-1 budget)
ARMS = ("A1", "A2", "A3", "B1", "B2", "B3")
QUANTUM = ("A1", "A2", "A3")
RNG_BASE = 43000                          # exp43b-family convention

TIER_Q_PRIME_SCOPE = (
    "doc 24 SS7 (frozen with the criterion): A3 is full-SU(4) and has NO "
    "structure-matched classical counterpart in the arm set, so a "
    "positive tier-q' licenses only 'the full-SU(4) block class learns "
    "this rung where the matched classical class does not, at matched "
    "budget' -- NOT a quantum-advantage reading. This caveat travels "
    "with the number (Rule 5).")

INTERPRETATION_NOTE = (
    "doc 24 SS7 (pre-registered): Phase 1 ran only A3 and B1, so whether "
    "A2 learns L1 is unknown. If A2 fails L1 while A3 succeeds, the "
    "finding is about EXPRESSIVITY (universal vs restricted block "
    "class), not about quantum-vs-classical, and tier-q is reported as "
    "failed on its own terms.")

B1_CHOICE_NOTE = (
    "B1 runs AS SHIPPED per exp43c's SS5.4 row (false/true): the matched "
    "initialisation satisfies F3 (twin collapse 0.555/0.580 <= 0.60) yet "
    "meets the L1 gate in 0/8 configs, while B1-as-shipped passed the "
    "SS5.2 memorisation control (3/8 configs >= 0.95 on 64 items). The "
    "asymmetry survives its strongest challenge; the init audit "
    "(results_exp43b_b1_init_audit.json) and exp43c "
    "(results_exp43c.json) are reported alongside every comparative "
    "number here as the reason a reader should believe it.")

B3_LABEL = (
    "UNTRAINED INSTRUMENT on this rung (doc 24 SS4 + F5): B3 failed the "
    "SS5.2 memorisation control 0/8 at exactly 0.500 -- L1 twins are "
    "bag-identical, so a bag model is structurally blind to the rung. "
    "Its numbers are recorded for the sentinel value only and enter NO "
    "comparison.")

A1_SENTINEL_RULE = (
    "leak sentinel: Lemma 1 (exp43a analysis W1) pins SWAP-commuting "
    "blocks to exactly 50% on twin pairs; execution resolution of 'must "
    "remain at chance': every seed's best-val test accuracy must be "
    "<= 0.55 (the chance+5pp convention used by C2/C4).")


def load_l1():
    with open("stories_exp43b_%s.json" % RUNG, encoding="utf-8") as f:
        data = json.load(f)
    items, meta = data["items"], data["meta"]
    sha = comp.canonical_sha256(items)
    if sha != meta["dataset_sha256"]:
        raise RuntimeError("L1 dataset hash mismatch")
    return items, meta, sha


def make_model(arm, verbs, vocab, seed):
    if arm in QUANTUM:
        return QuantumStoryModel(arm, verbs, seed)
    if arm == "B1":
        return ClassicalDisCoCirc(verbs, seed, d_ref=2)
    if arm == "B2":
        return SeqBaseline(vocab, seed, variant="matched")
    if arm == "B3":
        return BagBaseline(vocab, seed)
    raise ValueError(arm)


def evaluate(model, compiled):
    correct = [1 if model.predict(cs) == cs.answer else 0 for cs in compiled]
    return sum(correct) / len(correct), correct


def train_seed(arm, seed, data, meta, vocab, verbs, max_epochs):
    """exp42_run.train_seed semantics: val-only selection, name
    augmentation for B2, early stop only at train=val=1.0."""
    torch.manual_seed(seed)
    model = make_model(arm, verbs, vocab, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = random.Random(RNG_BASE + seed)
    train, val = data["train"], data["val"]
    augment = arm == "B2"                 # training protocol (doc 22 SS4)

    best = {"val_acc": -1.0, "train_acc": -1.0, "epoch": -1, "state": None}
    curves = {"loss": [], "train_run_acc": [], "val_acc": [],
              "grad_norm": []}
    steps_taken = 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses, gnorms, run_correct = [], [], 0
        for start in range(0, len(order), BATCH):
            bat = [train[i] for i in order[start:start + BATCH]]
            opt.zero_grad()
            bloss = 0.0
            for cs in bat:
                if augment:
                    mapping = sample_name_mapping(meta["humans"],
                                                  meta["objects"], rng)
                    loss, pred = model.story_loss(cs, mapping=mapping)
                else:
                    loss, pred = model.story_loss(cs)
                bloss = bloss + loss
                if pred == cs.answer:
                    run_correct += 1
            bloss = bloss / len(bat)
            bloss.backward()
            gn = grad_global_norm(model)
            assert math.isfinite(gn), \
                "non-finite grad (%s s%d)" % (arm, seed)
            gnorms.append(gn)
            opt.step()
            steps_taken += 1
            losses.append(bloss.item())

        val_acc, _ = evaluate(model, val)
        curves["loss"].append(sum(losses) / len(losses))
        curves["train_run_acc"].append(run_correct / len(train))
        curves["val_acc"].append(val_acc)
        curves["grad_norm"].append(sum(gnorms) / len(gnorms))
        if val_acc > best["val_acc"]:                     # ID val ONLY
            clean_train, _ = evaluate(model, train)
            best = {"val_acc": val_acc, "train_acc": clean_train,
                    "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
        print("[%s s%d] ep %2d/%d loss=%.4f train(run)=%.3f val=%.3f "
              "gn=%.2e best@%d(val=%.3f) %.1fs"
              % (arm, seed, epoch, max_epochs, curves["loss"][-1],
                 curves["train_run_acc"][-1], val_acc,
                 curves["grad_norm"][-1], best["epoch"], best["val_acc"],
                 time.time() - t0), flush=True)
        if best["val_acc"] >= 1.0 and best["train_acc"] >= 1.0:
            print("[%s s%d] early stop: train and val both 100%%"
                  % (arm, seed), flush=True)
            break

    model.load_state_dict(best["state"])
    test_acc, test_correct = evaluate(model, data["test"])
    detail = {
        "arm": arm, "seed": seed, "lr": LR, "batch": BATCH,
        "epochs_run": len(curves["loss"]), "steps_taken": steps_taken,
        "best_epoch": best["epoch"], "val_acc": best["val_acc"],
        "train_acc": best["train_acc"], "test_acc": test_acc,
        "test_correct": test_correct,
        "phase1_style_gate_met_at_bestval": bool(
            best["train_acc"] >= 0.8 and best["val_acc"] >= 0.7),
        "curves": curves,
    }
    return model, detail


# --------------------------------------------------- C1 (exhaustive, fast)
def quantum_mats(model):
    with torch.no_grad():
        mats = {v: model.verb_matrix(v).detach() for v in model.verbs}
        effs = [model.q_pos.matrix().conj().T.detach(),
                model.q_neg.matrix().conj().T.detach()]
    return mats, effs


def quantum_fast_acc(mats, effs, compiled, vm=None):
    vm = vm or {}
    with torch.no_grad():
        n_ok = 0
        for cs in compiled:
            psi = sim.init_state(cs.K)
            for w, _n, ang in cs.intro:
                psi = sim.apply_gate(psi, sim.ry(ang), (w,))
            for v, wa, wp in cs.events:
                psi = sim.apply_gate(psi, mats[vm.get(v, v)], (wa, wp))
            vseq, qa, qp = cs.question
            qw = (qa, qp)
            for qv in reversed(vseq):
                psi = sim.apply_gate(psi, mats[vm.get(qv, qv)].conj().T, qw)
            vals = [sim.prob_all_zero_on(sim.apply_gate(psi, E, qw), qw)
                    for E in effs]
            n_ok += ((0 if vals[0].item() >= vals[1].item() else 1)
                     == cs.answer)
    return n_ok / len(compiled)


def c1_exhaustive(models, compiled_test, verbs, n_perm_cap=None):
    """C1 dial-swap with ALL non-identity permutations (6 verbs -> 719),
    exp42's seed-mean rule: the same permutation applied to every seed,
    original seed-mean strictly above the permuted 95th percentile."""
    cached = [quantum_mats(m) for m in models]
    # parity vs the committed predict path (first 20 test items, seed 0)
    m0 = models[0]
    ref = sum(1 for cs in compiled_test[:20]
              if m0.predict(cs) == cs.answer) / 20.0
    fast = quantum_fast_acc(*cached[0], compiled_test[:20])
    assert abs(ref - fast) < 1e-12, (ref, fast)

    orig = sum(quantum_fast_acc(ma, ef, compiled_test)
               for ma, ef in cached) / len(cached)
    perms = [p for p in itertools.permutations(verbs)
             if list(p) != list(verbs)]
    if n_perm_cap:
        perms = perms[:n_perm_cap]
    perm_accs = []
    for i, p in enumerate(perms):
        vmap = dict(zip(verbs, p))
        acc = sum(quantum_fast_acc(ma, ef, compiled_test, vmap)
                  for ma, ef in cached) / len(cached)
        perm_accs.append(acc)
        if (i + 1) % 100 == 0:
            print("[C1] %d/%d permutations" % (i + 1, len(perms)),
                  flush=True)
    perm_sorted = sorted(perm_accs)
    q95 = perm_sorted[int(0.95 * len(perms)) - 1]
    return {
        "control": "C1 trained-dial-swap (exhaustive, 6 verbs)",
        "n_permutations": len(perms),
        "original_seed_mean_acc": orig,
        "perm_95th_percentile": q95,
        "perm_max": perm_sorted[-1],
        "perm_mean": sum(perm_accs) / len(perm_accs),
        "content_earned": bool(orig > q95),
    }


# ------------------------------------------------------------- statistics
def seed_mean_correct(details):
    vecs = [d["test_correct"] for d in details]
    n = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(n)]


def compare(details_x, details_y):
    mx, my = seed_mean_correct(details_x), seed_mean_correct(details_y)
    diff = [a - b for a, b in zip(mx, my)]
    obs, p = paired_permutation_pvalue(diff)
    return {"mean_acc_x": sum(mx) / len(mx), "mean_acc_y": sum(my) / len(my),
            "mean_paired_diff": obs, "one_sided_p": p, "n_items": len(diff)}


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="mechanical check: 1 seed, 2 epochs, 20 C1 "
                         "perms, NO results file, NO checkpoints")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)
    n_seeds = 1 if args.smoke else N_SEEDS
    max_epochs = 2 if args.smoke else MAX_EPOCHS

    items, meta, sha = load_l1()
    angles, _emb_meta, emb_sha = comp.load_embeddings()
    verbs = sorted(meta["verb_inventory"].keys())
    data = {sp: comp.compile_all([i for i in items if i["split"] == sp],
                                 angles, with_tokens=True)
            for sp in ("train", "val", "test")}
    vocab = build_vocab(cs.tokens for sp in ("train", "val", "test")
                        for cs in data[sp])
    print("[phase2] compiled train=%d val=%d test=%d verbs=%d vocab=%d"
          % (len(data["train"]), len(data["val"]), len(data["test"]),
             len(verbs), len(vocab)), flush=True)

    per_arm, models_per_arm = {}, {}
    for arm in ARMS:
        details, models = [], []
        for seed in range(n_seeds):
            key = "p2_%s_s%d" % (arm, seed)
            ckj = os.path.join(CKPT_DIR, key + ".json")
            ckp = os.path.join(CKPT_DIR, key + ".pt")
            if not args.smoke and os.path.exists(ckj) \
                    and os.path.exists(ckp):
                with open(ckj, encoding="utf-8") as f:
                    d = json.load(f)
                m = make_model(arm, verbs, vocab, seed)
                m.load_state_dict(torch.load(ckp, weights_only=True))
                print("[ckpt] %s loaded" % key, flush=True)
            else:
                m, d = train_seed(arm, seed, data, meta, vocab, verbs,
                                  max_epochs)
                if not args.smoke:
                    with open(ckj, "w", encoding="utf-8") as f:
                        json.dump(d, f, ensure_ascii=False)
                    torch.save(m.state_dict(), ckp)
                    print("[ckpt] %s saved (val=%.3f test=%.3f)"
                          % (key, d["val_acc"], d["test_acc"]), flush=True)
            details.append(d)
            models.append(m)
        per_arm[arm] = details
        models_per_arm[arm] = models
        print("[phase2] %s done: val=%s test=%s"
              % (arm, [round(d["val_acc"], 3) for d in details],
                 [round(d["test_acc"], 3) for d in details]), flush=True)

    # ------------- F1 verification
    step_counts = {arm: [d["steps_taken"] for d in per_arm[arm]]
                   for arm in ARMS}
    all_steps = {s for v in step_counts.values() for s in v}
    f1_equal_steps = len(all_steps) == 1

    # ------------- statistics + controls
    stats = {
        "A2_vs_B1": compare(per_arm["A2"], per_arm["B1"]),
        "A3_vs_B1": compare(per_arm["A3"], per_arm["B1"]),
        "A2_vs_B2": compare(per_arm["A2"], per_arm["B2"]),
        "B1_vs_B2": compare(per_arm["B1"], per_arm["B2"]),
    }
    cap = 20 if args.smoke else None
    print("[phase2] C1 on A2...", flush=True)
    c1_a2 = c1_exhaustive(models_per_arm["A2"], data["test"], verbs, cap)
    print("[phase2] C1 on A3...", flush=True)
    c1_a3 = c1_exhaustive(models_per_arm["A3"], data["test"], verbs, cap)

    probe_item = next(
        (it for it in items if it["split"] == "test"
         and "SVO" in it["order_flags"] and "VSO" in it["order_flags"]),
        None)
    controls = {"C1_A2": c1_a2, "C1_A3": c1_a3}
    if probe_item is not None:
        struct = {"A2_seed0": models_per_arm["A2"][0],
                  "A3_seed0": models_per_arm["A3"][0],
                  "B1_seed0": models_per_arm["B1"][0]}
        controls["C3"] = ctl.c3_order_invariance(
            struct, models_per_arm["B2"][0], probe_item, angles)
        controls["C3_B2_delta"] = ctl.c3_b2_behavioral_delta(
            models_per_arm["B2"][0],
            [it for it in items if it["split"] == "test"][:200], angles)
    else:
        controls["C3"] = {"error": "no test item with both SVO and VSO"}

    a1_test = [d["test_acc"] for d in per_arm["A1"]]
    a1_ok = all(a <= 0.55 for a in a1_test)

    verdicts = {
        "tier_q_p": stats["A2_vs_B1"]["one_sided_p"],
        "content_earned_A2": c1_a2["content_earned"],
        "tier_q_passed": bool(stats["A2_vs_B1"]["one_sided_p"] < 0.05
                              and c1_a2["content_earned"]),
        "tier_q_prime_p": stats["A3_vs_B1"]["one_sided_p"],
        "content_earned_A3": c1_a3["content_earned"],
        "tier_q_prime_passed": bool(
            stats["A3_vs_B1"]["one_sided_p"] < 0.05
            and c1_a3["content_earned"]),
        "tier_s_passed": bool(
            stats["A2_vs_B2"]["one_sided_p"] < 0.05
            and stats["B1_vs_B2"]["one_sided_p"] < 0.05),
        "a1_leak_sentinel_ok": bool(a1_ok),
        "f1_equal_steps": bool(f1_equal_steps),
        "b3_untrained_instrument": True,
    }
    a2_gate = any(d["phase1_style_gate_met_at_bestval"]
                  for d in per_arm["A2"])
    a3_gate = any(d["phase1_style_gate_met_at_bestval"]
                  for d in per_arm["A3"])
    expressivity_reading_applies = bool((not a2_gate) and a3_gate)

    if args.smoke:
        print("[smoke] verdicts (NOT results):",
              json.dumps(verdicts))
        print("[smoke] no results file written.")
        return

    for arm in ARMS:                       # bulky vectors: summarized above
        for d in per_arm[arm]:
            d.pop("test_correct", None)

    results = {
        "experiment": "exp43b Phase 2 comparison on L1",
        "spec": "doc 23 Phase 2 (commit ef15d76) as bound by doc 24 "
                "SS4-SS7 (commit c6099b0): 5 seeds x {A1,A2,A3,B1,B2,B3} "
                "at the recorded config lr 0.02 / batch 8, <=60 epochs, "
                "val-only selection; tier-q = A2 vs B1 one-sided paired "
                "permutation p<0.05 on the L1 test split; tier-q' = A3 "
                "vs B1, separate criterion; C1 exhaustive dial-swap; C3; "
                "A1 leak sentinel; F1-F5 fairness provisions",
        "authorization": "exp43c (results_exp43c.json, commit 7e9b4af) "
                         "landed on doc 24 SS5.4 row false/true: Phase 2 "
                         "proceeds with B1 as shipped",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "dataset_sha256": sha,
        "embeddings_sha256": emb_sha,
        "config": {"rung": RUNG, "lr": LR, "batch": BATCH,
                   "max_epochs": MAX_EPOCHS, "n_seeds": N_SEEDS,
                   "optimizer": "Adam", "selection": "val only",
                   "rng_stream": "random.Random(43000+seed), exp43b "
                                 "convention",
                   "vocab": "B2/B3 vocab over all L1 splits, size %d"
                            % len(vocab)},
        "meta_notes": {
            "b1_as_shipped": B1_CHOICE_NOTE,
            "b3_label": B3_LABEL,
            "a1_sentinel_rule": A1_SENTINEL_RULE,
            "b2_augmentation": "name augmentation ON for training (doc "
                               "22 SS4 protocol); the exp43c disable was "
                               "control-only",
            "tier_q_rule": "tier_q_passed = (A2_vs_B1 one-sided p < "
                           "0.05) AND content_earned_A2 -- the exp42 "
                           "convention; combining with C1 can only make "
                           "the claim harder",
            "tier_q_prime_scope": TIER_Q_PRIME_SCOPE,
            "tier_s_rule": "tier_s_passed = A2_vs_B2 AND B1_vs_B2 both "
                           "p<0.05; B3 excluded per F5 (untrained "
                           "instrument); B2_generous not in the doc 23 "
                           "arm set",
            "pre_registered_interpretation": INTERPRETATION_NOTE,
        },
        "param_counts": {arm: models_per_arm[arm][0].param_counts()
                         for arm in ARMS},
        "f1_step_counts": step_counts,
        "per_arm": per_arm,
        "gate_statistics": stats,
        "controls": controls,
        "a1_test_accs": a1_test,
        "a2_meets_phase1_gate_any_seed": bool(a2_gate),
        "a3_meets_phase1_gate_any_seed": bool(a3_gate),
        "expressivity_reading_applies": expressivity_reading_applies,
        "verdicts": verdicts,
        "git_note": git_note(),
        "wall_time_sec": round(time.time() - t0, 1),
    }
    path = fresh_results_path("results_exp43b_phase2")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("[phase2] wrote %s" % path)
    print("[phase2] verdicts: %s" % json.dumps(verdicts))
    print("[phase2] expressivity_reading_applies=%s (A2 gate %s, A3 "
          "gate %s)" % (expressivity_reading_applies, a2_gate, a3_gate))


if __name__ == "__main__":
    main()
