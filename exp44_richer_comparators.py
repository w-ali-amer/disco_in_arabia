# -*- coding: utf-8 -*-
"""exp44_richer_comparators.py -- doc 27, richer classical comparators.

Answers the standing objection to tier-q: "B1 is not THE classical
comparator, it is one heavily constrained one -- you mirrored A2's
factorisation, then reported the classical arm lost."

Two new arms, in the EXACT ClassicalDisCoCirc composition tree (same
referent vectors, same event application, same renormalisation, same
adjoint-effect readout). Only the verb-map parameterisation and d change:

  B4  referent R^2, verb = UNCONSTRAINED real 4x4     16 params/verb
      -> the CLOSURE of B1's verb class (every B1 matrix
         kron(d0,d1) @ E @ kron(d2,d3) is some real 4x4), at 128 total
         params vs A2's 114 -- the tightest parameter match in the arm set.
  B5  referent R^4, verb = UNCONSTRAINED real 16x16  256 params/verb
      -> capacity arm, 2048 total = 18x A2. Its referent dimension 4 is
         the dimension of a 2-qubit register, making B5 the
         dimension-matched classical counterpart of doc 28's A4.

Doc 27 provisions implemented here:
  SS4 F1 -- 60 FULL epochs at every config, floor of 9000 steps. A2's
     frozen budget was 9000 steps; at batch 8 B4/B5 get exactly that, at
     batch 1 they get 72000 (8x). Step-CAPPING was considered and
     rejected (it would have starved the batch-1 configs at 7.5 epochs).
  SS4 F2 -- the full frozen Phase-1 grid {lr .005,.02} x {batch 1,8} = 4
     configs per arm, per-seed selection on VALIDATION ONLY. A2 ran at
     exactly one config. Both relaxations favour the comparator and are
     admissible under doc 24 SS3 (they can only make our claim harder).
  SS4 F3 -- matched init: every verb map and readout map drawn as a
     random special-orthogonal matrix, matrix_exp(A - A^T) with the
     strictly-upper triangle of A ~ torch.rand * 2pi. Training-free init
     audit must show twin collapse <= 0.60 before an arm's numbers enter
     a comparison.
  SS4 F4 -- torch.rand only, never torch.randn (not bit-stable across
     builds; doc 24 SS8 P3).
  SS5   -- memorisation control (64 items, train >= 0.95) gates
     everything; doc 24 SS4's standing rule applies (an arm failing it is
     an UNTRAINED INSTRUMENT, not a beaten one).
  SS6   -- A2 is NOT retrained. Its per-item test_correct vectors are read
     from the committed exp43b_phase2_ckpt/p2_A2_s*.json sidecars under a
     blocking integrity precondition (sha256 recorded, mean == test_acc
     exactly, length 300, dataset/embedding digests match).

Criteria (frozen, doc 27 SS6): tier_q_B4_passed, tier_q_B5_passed.
These are NEW and separately named. results_exp43b_phase2.json's
tier_q_passed is frozen, not recomputed and not amended.
"""

import argparse
import copy
import hashlib
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
from exp42_baselines import ClassicalDisCoCirc
from exp42_run import (fresh_results_path, git_note, grad_global_norm,
                       paired_permutation_pvalue)
from exp43b_phase2 import seed_mean_correct

DESIGN_DOC = ("qnlp_private_docs/27_exp44_richer_classical_comparators.md "
              "(commit 9b5f85e, FROZEN)")
RUNG = "L1"
CKPT_DIR = "exp44_ckpt"
PHASE2_CKPT_DIR = "exp43b_phase2_ckpt"
PHASE2_RESULTS = "results_exp43b_phase2.json"
N_SEEDS = 5                                # FROZEN (doc 23)
MAX_EPOCHS = 60                            # FROZEN (doc 27 SS4 F1)
STEP_FLOOR = 9000                          # A2's frozen budget (doc 27 SS4)
GRID = [(0.005, 1), (0.005, 8), (0.02, 1), (0.02, 8)]   # FROZEN Phase-1 grid
ARMS = ("B4", "B5")
RNG_BASE = 44000
INIT_SEED_OFFSET = 52000                   # independent recorded draw stream
AUDIT_N_PAIRS = 200
F3_BAR = 0.60
MEMO_ITEMS = 64
MEMO_EPOCHS = 300
MEMO_BAR = 0.95
VAL_GATE, TRAIN_GATE = 0.7, 0.8

SCOPE_NOTE = (
    "doc 27 SS8, frozen BEFORE any number existed: no outcome of exp44 is "
    "a quantum-advantage result. One rung, one task family, one dataset, "
    "simulation only, 6 verbs, 5 seeds. A favourable result says a "
    "14-parameter model beats 16- and 256-parameter classical models at a "
    "swap-sensitivity task -- a statement about model classes on this "
    "task, nothing more. The C1 margin from Phase 2 stays narrow (0.7993 "
    "vs a 0.7933 permutation 95th percentile) and every quotation of "
    "tier-q carries it.")

BUDGET_ASYMMETRY_NOTE = (
    "doc 27 SS4 F1+F2, DELIBERATE and in the COMPARATORS' favour: B4/B5 "
    "receive 60 full epochs at each of 4 configs with per-seed val-only "
    "selection, i.e. up to 8x A2's optimiser steps (72000 at batch 1 vs "
    "A2's 9000) and 4x its hyperparameter search (A2 ran at one frozen "
    "config). Admissible under doc 24 SS3 because it can only make our own "
    "claim harder. This asymmetry MUST be stated in every report of the "
    "result (Rule 5), including a favourable one.")

NARROWING_RULE = (
    "doc 27 SS7, frozen: if tier_q_B4_passed is false, tier_q_passed in "
    "results_exp43b_phase2.json STANDS AS COMPUTED (it was registered as "
    "A2 vs B1), but its interpretation narrows to 'A2 beats the "
    "structure-matched classical arm B1; an unconstrained classical arm of "
    "matched parameter count does not lose.' That narrowing travels with "
    "the tier-q number into every README, PDF, slide and conversation, and "
    "is stated BEFORE the tier-q number, not after it.")


# ------------------------------------------------------------ init (F3/F4)
def random_so(D, g):
    """Random special-orthogonal DxD: matrix_exp(A - A^T), strictly-upper
    triangle of A ~ torch.rand * 2pi (doc 27 SS4 F3; torch.rand only, F4)."""
    A = torch.zeros(D, D)
    iu = torch.triu_indices(D, D, offset=1)
    A[iu[0], iu[1]] = torch.rand(iu.shape[1], generator=g) * 2 * math.pi
    return torch.linalg.matrix_exp(A - A.T)


class UnconstrainedDisCoCirc(ClassicalDisCoCirc):
    """B4 (d=2) / B5 (d=4): the ClassicalDisCoCirc composition tree with an
    UNCONSTRAINED real DxD verb map replacing B1's
    kron(d0,d1) @ E @ kron(d2,d3) factorisation. Strict superset of B1's
    verb class. Readout maps keep B1's shape and are re-initialised to
    random SO(D) (B1 ships them at identity + 0.1*randn)."""

    def __init__(self, verbs, seed, d_ref=2):
        super().__init__(verbs, seed, d_ref=d_ref)
        D = self.d * self.d
        # drop B1's parameterisation entirely; keep wiring, d, readout shape
        del self.dress
        del self.core
        g = torch.Generator().manual_seed(INIT_SEED_OFFSET + seed)
        self.vmat = torch.nn.ParameterDict(
            {v: torch.nn.Parameter(random_so(D, g)) for v in self.verbs})
        with torch.no_grad():
            self.q_pos.data = random_so(D, g)
            self.q_neg.data = random_so(D, g)

    def verb_matrix(self, v):
        return self.vmat[v]

    def param_counts(self):
        D = self.d * self.d
        per_verb, readout = D * D, 2 * D * D
        total = sum(p.numel() for p in self.parameters())
        assert total == per_verb * len(self.verbs) + readout, \
            "param accounting drifted: %d" % total
        return {"arm": "B4" if self.d == 2 else "B5", "d_ref": self.d,
                "per_verb": per_verb, "n_verbs": len(self.verbs),
                "readout_shared": readout, "total_trained": total}


def make_model(arm, verbs, seed):
    return UnconstrainedDisCoCirc(verbs, seed, d_ref=2 if arm == "B4" else 4)


# ------------------------------------------------------------------ data
def load_rung():
    with open("stories_exp43b_%s.json" % RUNG, encoding="utf-8") as f:
        data = json.load(f)
    return data["items"], data["meta"], data["meta"]["dataset_sha256"]


def evaluate(model, compiled):
    correct = [1 if model.predict(cs) == cs.answer else 0 for cs in compiled]
    return sum(correct) / len(correct), correct


# ------------------------------------------- F3 training-free init audit
def init_audit(arm, items, angles, verbs):
    """exp43b_b1_init_audit / exp43c matched_init_audit measurement: what
    fraction of L1 twin pairs does the arm predict IDENTICALLY at init?
    Lemma 1 (exp43a W1) pins SWAP-commuting blocks to exactly 50% on twins;
    an arm starting inside that neighbourhood is not being tested by the
    rung. F3 bar: <= 0.60."""
    by_pair = {}
    for it in items:
        if it["split"] == "train":
            by_pair.setdefault(it["pair_id"], []).append(it)
    pairs = [p for p in sorted(by_pair) if len(by_pair[p]) == 2][:AUDIT_N_PAIRS]
    out = {}
    for seed in range(N_SEEDS):
        model = make_model(arm, verbs, seed)
        same = 0
        with torch.no_grad():
            for p in pairs:
                a, b = sorted(by_pair[p], key=lambda x: x["id"])
                ca, cb = comp.compile_story(a, angles), comp.compile_story(b, angles)
                if model.predict(ca) == model.predict(cb):
                    same += 1
        out["seed%d" % seed] = same / len(pairs)
    frac = sum(out.values()) / len(out)
    return {"n_pairs": len(pairs), "per_seed": out,
            "frac_twin_pairs_predicting_identically": frac,
            "f3_bar": F3_BAR, "f3_pass": bool(frac <= F3_BAR)}


# ---------------------------------------------------------------- training
def train_run(arm, seed, lr, batch, data, verbs, max_epochs,
              train_items=None, select_on_val=True, step_floor=0):
    """Phase-2 semantics (exp43b_phase2.train_seed): val-only selection,
    early stop only at train=val=1.0. doc 27 SS4 F1: 60 FULL epochs, with a
    9000-step floor -- never step-capped below A2's budget."""
    torch.manual_seed(seed)
    model = make_model(arm, verbs, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(RNG_BASE + seed)
    train = train_items if train_items is not None else data["train"]
    val = data["val"] if select_on_val else train

    best = {"val_acc": -1.0, "train_acc": -1.0, "epoch": -1, "state": None}
    curves = {"loss": [], "train_run_acc": [], "val_acc": [], "grad_norm": []}
    steps = 0
    epoch = 0
    while epoch < max_epochs or steps < step_floor:
        epoch += 1
        if epoch > max_epochs and steps >= step_floor:
            break
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
            assert math.isfinite(gn), "non-finite grad (%s s%d)" % (arm, seed)
            gnorms.append(gn)
            opt.step()
            steps += 1
            losses.append(bloss.item())

        val_acc, _ = evaluate(model, val)
        curves["loss"].append(sum(losses) / len(losses))
        curves["train_run_acc"].append(run_correct / len(train))
        curves["val_acc"].append(val_acc)
        curves["grad_norm"].append(sum(gnorms) / len(gnorms))
        if val_acc > best["val_acc"]:
            clean_train, _ = evaluate(model, train)
            best = {"val_acc": val_acc, "train_acc": clean_train,
                    "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        print("[%s s%d lr%.3f b%d] ep %2d loss=%.4f train(run)=%.3f "
              "val=%.3f gn=%.2e best@%d(%.3f) %.1fs"
              % (arm, seed, lr, batch, epoch, curves["loss"][-1],
                 curves["train_run_acc"][-1], val_acc, curves["grad_norm"][-1],
                 best["epoch"], best["val_acc"], time.time() - t0), flush=True)
        if best["val_acc"] >= 1.0 and best["train_acc"] >= 1.0:
            print("[%s s%d] early stop: train and val both 100%%"
                  % (arm, seed), flush=True)
            break

    model.load_state_dict(best["state"])
    test_acc, test_correct = evaluate(model, data["test"]) \
        if select_on_val else (best["train_acc"], [])
    return model, {
        "arm": arm, "seed": seed, "lr": lr, "batch": batch,
        "epochs_run": epoch, "steps_taken": steps,
        "best_epoch": best["epoch"], "val_acc": best["val_acc"],
        "train_acc": best["train_acc"], "test_acc": test_acc,
        "test_correct": test_correct,
        "phase1_style_gate_met_at_bestval":
            bool(best["train_acc"] >= TRAIN_GATE and best["val_acc"] >= VAL_GATE),
        "curves": curves}


# ------------------------------------------------- SS5 memorisation control
def memorisation_control(arm, data, verbs, max_epochs):
    """doc 24 SS5.2 verbatim: fixed 64-item subset (first 32 twin pairs by
    sorted pair_id), no held-out set, frozen grid, 2 seeds, <=300 epochs.
    arm_trainable = true iff any config reaches train >= 0.95."""
    subset = data["memo"]
    runs, best = [], 0.0
    for lr, batch in GRID:
        for seed in range(2):
            _m, d = train_run(arm, seed, lr, batch, data, verbs, max_epochs,
                              train_items=subset, select_on_val=False)
            acc = d["train_acc"]
            best = max(best, acc)
            runs.append({"lr": lr, "batch": batch, "seed": seed,
                         "best_train_acc": acc,
                         "meets": bool(acc >= MEMO_BAR)})
            print("[%s memo] lr=%.3f b=%d s%d -> train=%.3f"
                  % (arm, lr, batch, seed, acc), flush=True)
    return {"n_items": len(subset), "bar": MEMO_BAR, "runs": runs,
            "best_train_acc": best, "arm_trainable": bool(best >= MEMO_BAR)}


# --------------------------------------------------- SS6 A2 reuse integrity
def load_frozen_a2():
    """doc 27 SS6: A2 is NOT retrained. Read its per-item test_correct from
    the COMMITTED p2_A2_s*.json sidecars, under a blocking integrity
    precondition. Rule 6: every number traces to committed JSON."""
    with open(PHASE2_RESULTS, encoding="utf-8") as f:
        p2 = json.load(f)
    frozen = {d["seed"]: d for d in p2["per_arm"]["A2"]}
    details, checks = [], []
    for seed in range(N_SEEDS):
        path = os.path.join(PHASE2_CKPT_DIR, "p2_A2_s%d.json" % seed)
        raw = open(path, "rb").read()
        sha = hashlib.sha256(raw).hexdigest()
        d = json.loads(raw.decode("utf-8"))
        tc = d["test_correct"]
        mean_tc = sum(tc) / len(tc)
        ref = frozen[seed]["test_acc"]
        ok = (len(tc) == 300) and (mean_tc == ref)
        checks.append({"seed": seed, "path": path, "sha256": sha,
                       "len_test_correct": len(tc), "mean_test_correct": mean_tc,
                       "frozen_test_acc": ref, "exact_match": bool(ok)})
        details.append(d)
    return details, checks, p2


# ------------------------------------------------- C1 (classical-compatible)
def c1_exhaustive_classical(models, compiled_test, verbs, n_perm_cap=None):
    """C1 trained-dial-swap, exhaustive over the 719 non-identity
    permutations of 6 verbs -- the same control Phase 2 ran on A2/A3.

    exp43b_phase2.c1_exhaustive cannot be reused: it calls
    model.q_pos.matrix(), which exists only on the quantum arms' Circuit4
    readouts. B4/B5 carry q_pos/q_neg as raw real Parameters, so the
    permutation is applied through the shared model.predict(cs, verb_map)
    path instead. The permutation is of the verb->matrix ASSIGNMENT and is
    applied consistently to story events AND the question verb, exactly as
    in exp42_controls.

    DESCRIPTIVE ONLY (doc 27 SS6): whether a classical arm that learns the
    rung also earns its content is informative, and carries no criterion.
    """
    perms = [p for p in itertools.permutations(verbs) if list(p) != list(verbs)]
    if n_perm_cap:
        perms = perms[:n_perm_cap]

    def acc(m, vmap=None):
        return sum(1 for cs in compiled_test
                   if m.predict(cs, vmap) == cs.answer) / len(compiled_test)

    with torch.no_grad():
        orig = sum(acc(m) for m in models) / len(models)
        perm_accs = []
        for i, p in enumerate(perms):
            vmap = dict(zip(verbs, p))
            perm_accs.append(sum(acc(m, vmap) for m in models) / len(models))
            if (i + 1) % 100 == 0:
                print("[C1] %d/%d permutations" % (i + 1, len(perms)),
                      flush=True)
    perm_sorted = sorted(perm_accs)
    q95 = perm_sorted[int(0.95 * len(perms)) - 1]
    return {
        "control": "C1 trained-dial-swap (exhaustive, 6 verbs, classical path)",
        "n_permutations": len(perms),
        "original_seed_mean_acc": orig,
        "perm_95th_percentile": q95,
        "perm_max": perm_sorted[-1],
        "perm_mean": sum(perm_accs) / len(perm_accs),
        "content_earned": bool(orig > q95),
        "status": "DESCRIPTIVE ONLY (doc 27 SS6): carries no criterion.",
    }


# -------------------------------------------------------------- statistics
def compare(details_x, details_y):
    mx, my = seed_mean_correct(details_x), seed_mean_correct(details_y)
    diff = [a - b for a, b in zip(mx, my)]
    obs, p = paired_permutation_pvalue(diff)
    return {"mean_acc_x": sum(mx) / len(mx), "mean_acc_y": sum(my) / len(my),
            "mean_paired_diff": obs, "one_sided_p": p, "n_items": len(diff)}


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="mechanical check: 1 seed, 2 epochs, 1 config, "
                         "20 C1 perms, NO results file, NO checkpoints")
    args = ap.parse_args()
    t0 = time.time()
    if not args.smoke:
        os.makedirs(CKPT_DIR, exist_ok=True)
    n_seeds = 1 if args.smoke else N_SEEDS
    max_epochs = 2 if args.smoke else MAX_EPOCHS
    grid = GRID[-1:] if args.smoke else GRID
    memo_epochs = 2 if args.smoke else MEMO_EPOCHS
    # doc 27 SS4 F1's 9000-step floor binds ONLY the comparison runs. It must
    # not apply to the memorisation control (doc 24 SS5.2 fixes that at <=300
    # epochs on 64 items) nor to smoke runs.
    step_floor = 0 if args.smoke else STEP_FLOOR

    items, meta, sha = load_rung()
    angles, _em, emb_sha = comp.load_embeddings()
    verbs = sorted(meta["verb_inventory"].keys())
    data = {sp: comp.compile_all([i for i in items if i["split"] == sp], angles)
            for sp in ("train", "val", "test")}

    # doc 24 SS5.2: first 32 twin pairs by sorted pair_id -> 64 items
    by_pair = {}
    for it in items:
        if it["split"] == "train":
            by_pair.setdefault(it["pair_id"], []).append(it)
    memo_ids = []
    for p in sorted(by_pair):
        if len(by_pair[p]) == 2:
            memo_ids += [x["id"] for x in sorted(by_pair[p], key=lambda x: x["id"])]
        if len(memo_ids) >= MEMO_ITEMS:
            break
    memo_ids = set(memo_ids[:MEMO_ITEMS])
    data["memo"] = comp.compile_all(
        [i for i in items if i["split"] == "train" and i["id"] in memo_ids],
        angles)
    print("[exp44] compiled train=%d val=%d test=%d memo=%d verbs=%d"
          % (len(data["train"]), len(data["val"]), len(data["test"]),
             len(data["memo"]), len(verbs)), flush=True)

    # ---------- SS6 A2 integrity precondition (BLOCKING)
    a2_details, a2_checks, p2 = load_frozen_a2()
    integrity_ok = all(c["exact_match"] for c in a2_checks)
    digest_ok = (sha == p2["dataset_sha256"] and emb_sha == p2["embeddings_sha256"])
    print("[exp44] A2 reuse integrity: vectors=%s digests=%s"
          % (integrity_ok, digest_ok), flush=True)
    if not (integrity_ok and digest_ok):
        print("[exp44] HALT: doc 27 SS6 integrity precondition failed. "
              "Not proceeding on repaired data.", flush=True)
        if not args.smoke:
            path = fresh_results_path("results_exp44_HALTED")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"experiment": "exp44", "halted": True,
                           "reason": "doc 27 SS6 integrity precondition failed",
                           "a2_integrity_checks": a2_checks,
                           "dataset_digests_match": digest_ok}, f, indent=1)
            print("[exp44] wrote %s" % path, flush=True)
        return 1

    # ---------- F3 init audits + SS5 memorisation controls (gate everything)
    audits, memos = {}, {}
    for arm in ARMS:
        audits[arm] = init_audit(arm, items, angles, verbs)
        print("[exp44] %s init audit: collapse=%.3f F3(<=%.2f)=%s"
              % (arm, audits[arm]["frac_twin_pairs_predicting_identically"],
                 F3_BAR, audits[arm]["f3_pass"]), flush=True)
        memos[arm] = memorisation_control(arm, data, verbs, memo_epochs)
        print("[exp44] %s memorisation: best=%.3f trainable=%s"
              % (arm, memos[arm]["best_train_acc"], memos[arm]["arm_trainable"]),
              flush=True)

    # ---------- SS4 F2: full grid, per-seed val-only selection
    per_arm, models_per_arm, per_arm_all = {}, {}, {}
    for arm in ARMS:
        chosen, models, allruns = [], [], []
        for seed in range(n_seeds):
            cands = []
            for lr, batch in grid:
                key = "exp44_%s_s%d_lr%s_b%d" % (arm, seed, lr, batch)
                ckj = os.path.join(CKPT_DIR, key + ".json")
                ckp = os.path.join(CKPT_DIR, key + ".pt")
                if not args.smoke and os.path.exists(ckj) and os.path.exists(ckp):
                    with open(ckj, encoding="utf-8") as f:
                        d = json.load(f)
                    m = make_model(arm, verbs, seed)
                    m.load_state_dict(torch.load(ckp, weights_only=True))
                    print("[ckpt] %s loaded" % key, flush=True)
                else:
                    m, d = train_run(arm, seed, lr, batch, data, verbs,
                                     max_epochs, step_floor=step_floor)
                    if not args.smoke:
                        with open(ckj, "w", encoding="utf-8") as f:
                            json.dump(d, f, ensure_ascii=False)
                        torch.save(m.state_dict(), ckp)
                cands.append((d, m))
                allruns.append({k: v for k, v in d.items() if k != "curves"})
            # VAL-ONLY selection (doc 27 SS4 F2); ties -> lower lr, smaller batch
            best_d, best_m = min(cands, key=lambda c: (-c[0]["val_acc"],
                                                       c[0]["lr"], c[0]["batch"]))
            chosen.append(best_d)
            models.append(best_m)
            print("[exp44] %s s%d selected lr=%.3f b=%d val=%.3f test=%.3f"
                  % (arm, seed, best_d["lr"], best_d["batch"],
                     best_d["val_acc"], best_d["test_acc"]), flush=True)
        per_arm[arm] = chosen
        models_per_arm[arm] = models
        per_arm_all[arm] = allruns

    # ---------- criteria
    stats = {"A2_vs_B4": compare(a2_details, per_arm["B4"]),
             "A2_vs_B5": compare(a2_details, per_arm["B5"])}

    controls = {}
    cap = 20 if args.smoke else None
    for arm in ARMS:
        if max(d["val_acc"] for d in per_arm[arm]) >= VAL_GATE:
            print("[exp44] C1 on %s (descriptive)..." % arm, flush=True)
            controls["C1_%s" % arm] = c1_exhaustive_classical(
                models_per_arm[arm], data["test"], verbs, cap)

    verdicts = {}
    for arm in ARMS:
        key = "tier_q_%s" % arm
        p = stats["A2_vs_%s" % arm]["one_sided_p"]
        gated = audits[arm]["f3_pass"] and memos[arm]["arm_trainable"]
        verdicts["%s_p" % key] = p
        verdicts["%s_f3_pass" % key] = audits[arm]["f3_pass"]
        verdicts["arm_trainable_%s" % arm] = memos[arm]["arm_trainable"]
        if not memos[arm]["arm_trainable"]:
            verdicts["%s_passed" % key] = "UNDECIDABLE"
            verdicts["%s_note" % key] = (
                "doc 24 SS4 + doc 27 SS7: %s failed the SS5 memorisation "
                "control and is an UNTRAINED INSTRUMENT on this rung. Its "
                "criterion is UNDECIDABLE, never 'passed'." % arm)
        else:
            verdicts["%s_passed" % key] = bool(p < 0.05 and gated)
        verdicts["%s_meets_phase1_gate_any_seed" % arm.lower()] = bool(
            any(d["phase1_style_gate_met_at_bestval"] for d in per_arm[arm]))

    out = {
        "experiment": "exp44",
        "spec": DESIGN_DOC,
        "question": ("does the L1 tier-q gap survive an UNCONSTRAINED "
                     "classical arm at matched parameter count (B4) and an "
                     "18x-capacity one (B5), each given more steps and more "
                     "search than A2?"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "rung": RUNG,
        "dataset_sha256": sha,
        "embeddings_sha256": emb_sha,
        "grid": [{"lr": lr, "batch": b} for lr, b in grid],
        "max_epochs": max_epochs,
        "step_floor": STEP_FLOOR,
        "n_seeds": n_seeds,
        "param_counts": {arm: make_model(arm, verbs, 0).param_counts()
                         for arm in ARMS},
        "a2_reuse": {
            "retrained": False,
            "source_commit": "e8e7e78",
            "source_files": [c["path"] for c in a2_checks],
            "integrity_checks": a2_checks,
            "integrity_ok": integrity_ok,
            "dataset_digests_match": digest_ok},
        "init_audits_f3": audits,
        "memorisation_controls": memos,
        "per_arm_selected": {a: [{k: v for k, v in d.items() if k != "curves"}
                                 for d in per_arm[a]] for a in ARMS},
        "per_arm_all_configs": per_arm_all,
        "gate_statistics": stats,
        "controls": controls,
        "verdicts": verdicts,
        "meta_notes": {
            "scope": SCOPE_NOTE,
            "budget_asymmetry": BUDGET_ASYMMETRY_NOTE,
            "narrowing_rule_if_B4_wins": NARROWING_RULE,
            "tier_q_frozen": ("results_exp43b_phase2.json tier_q_passed=true "
                              "is FROZEN: not recomputed, not amended, not "
                              "renamed by exp44."),
            "b4_is_closure_of_b1": ("every B1 verb matrix "
                                    "kron(d0,d1) @ E @ kron(d2,d3) is a real "
                                    "4x4; B4 ranges over all real 4x4, so B4's "
                                    "verb class is a STRICT SUPERSET of B1's."),
            "b5_pairing": ("B5's referent dimension 4 = the dimension of a "
                           "2-qubit register, making it the dimension-matched "
                           "classical counterpart of doc 28's A4 arm."),
        },
        "git_note": git_note(),
        "wall_time_sec": time.time() - t0,
    }

    print("\n[exp44] VERDICTS")
    for k, v in verdicts.items():
        print("   %-38s %s" % (k, v))

    if args.smoke:
        print("\n[exp44] smoke run -- NO results file written")
        return 0
    path = fresh_results_path("results_exp44")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[exp44] wrote %s (%.1f min)" % (path, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
