# -*- coding: utf-8 -*-
"""exp51_run.py -- doc 35: does any arm learn the verb-twin rung?

VT twins differ ONLY in which verb attaches to which argument pair, so
answering REQUIRES binding a verb to a pair. A tied-block model compiles
both twins to the identical statevector and is pinned at exactly 0.500 --
the verb-direction analogue of Lemma 1, verified on this dataset at
delta=0.0, 100% identical predictions, accuracy 0.5000.

Arms (doc 35 SS4) -- note the ROLE INVERSION from L1:
  A1  2 params/verb   INFORMATIVE, not a sentinel. Its blocks differ per
      verb so nothing pins it; if 2 params/verb clears VT then binding is
      cheap and ORTHOGONAL to the order sensitivity L1 measured.
  A2  14              the analog-compilable arm
  A3  15              most expressive quantum arm
  B1  18              classical structure-matched, run for INFORMATION only
  T   1 shared block  THE SENTINEL: trained, must sit at exactly 0.500 and
      must FAIL the memorisation control (control-of-the-control)

Controls (doc 35 SS5):
  * C1 is NOT run -- it is provably blind here. A consistent relabeling maps
    a VT dataset onto itself (the task is verb-RELATIONAL), so a perfect
    binder scores its base accuracy under it. Registering a zero-power
    control would be theatre.
  * QUESTION-ONLY permutation is the content control, with an expectation
    DERIVED before any data existed: 359/719 = 0.4993, i.e. a perfect binder
    collapses to CHANCE. That makes the chance+5pp bar exactly right here.
  * Verb-twin init audit (F3 analogue), <= 0.60 -- the argument-twin audit
    does not transfer, so it is measured rather than assumed.
  * doc 24 SS5.2 memorisation control per arm; no arm is reported as "failed
    to learn" without one (doc 24 SS4).

No quantum-vs-classical criterion is registered at this scale (doc 35 SS6).
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
from exp42_baselines import ClassicalDisCoCirc, apply_real, real_prob_all_zero_on
from exp42_models import QuantumStoryModel
from exp42_run import fresh_results_path, git_note, grad_global_norm
from exp49_verb_agnostic import tied_model

DESIGN_DOC = ("qnlp_private_docs/35_exp51_verb_twin_rung.md "
              "(commit 6101ffe, FROZEN)")
DATA = "stories_exp51_VT.json"
EXPECTED_SHA = \
    "8aa71d1e286392ad96fe7a5f77dc8afb7998e5594b8abde67bbaef024e9d1064"
CKPT_DIR = "exp51_ckpt"
ARMS = ("A1", "A2", "A3", "B1", "T")
QUANTUMISH = ("A1", "A2", "A3", "T")
GRID = [(0.005, 1), (0.005, 8), (0.02, 1), (0.02, 8)]
N_SEEDS = 2
MAX_EPOCHS = 60
TRAIN_GATE, VAL_GATE = 0.8, 0.7
MEMO_ITEMS, MEMO_EPOCHS, MEMO_BAR = 64, 300, 0.95
F3_BAR = 0.60
COLLAPSE_BAR = 0.55
N_PERMS = 50
AUDIT_PAIRS = 200
RNG_BASE = 51000
PERM_SEED = 510000
BINDER_EXPECTATION = 359.0 / 719.0

SCOPE_NOTE = (
    "doc 35 SS8: no quantum-advantage reading in any outcome -- no comparison "
    "criterion is registered, B1 runs for information only, simulation only, "
    "6 verbs, 2 seeds, one rung. VT tests BINDING, not composition: attaching "
    "a verb to an argument pair is necessary for compositional semantics and "
    "nowhere near sufficient, and the write-up must not blur them. It does "
    "NOT retract exp49 -- L1 and L2 remain solvable verb-blind as computed; "
    "VT is a NEW rung, not a reinterpretation.")

C1_NOT_RUN = (
    "doc 35 SS5.1: C1 is deliberately NOT run. A relabeling applied "
    "consistently to story AND question maps a VT dataset onto itself (the "
    "task is verb-relational, not verb-absolute), so a perfect binder scores "
    "its BASE accuracy under it -- zero power by construction. Verified by "
    "direct computation while drafting doc 35.")


# ------------------------------------------------------------ arms
def make_arm(arm, verbs, seed):
    if arm == "T":
        return tied_model("A3", verbs, seed)
    if arm == "B1":
        return ClassicalDisCoCirc(verbs, seed, d_ref=2)
    return QuantumStoryModel(arm, verbs, seed)


def is_quantum(arm):
    return arm in QUANTUMISH


# ------------------------------------------- split-map forwards (story/q)
def vals_quantum(model, cs, smap, qmap):
    cache = {}

    def blk(v, m):
        v2 = m.get(v, v)
        if v2 not in cache:
            cache[v2] = model.verb_matrix(v2)
        return cache[v2]
    psi = sim.init_state(cs.K)
    for w, _n, ang in cs.intro:
        psi = sim.apply_gate(psi, sim.ry(ang), (w,))
    for v, wa, wp in cs.events:
        psi = sim.apply_gate(psi, blk(v, smap), (wa, wp))
    vseq, qa, qp = cs.question
    qw = (qa, qp)
    for qv in reversed(vseq):
        psi = sim.apply_gate(psi, blk(qv, qmap).conj().T, qw)
    return [sim.prob_all_zero_on(
        sim.apply_gate(psi, e.matrix().conj().T, qw), qw)
        for e in (model.q_pos, model.q_neg)]


def vals_classical(model, cs, smap, qmap):
    cache = {}

    def blk(v, m):
        v2 = m.get(v, v)
        if v2 not in cache:
            cache[v2] = model.verb_matrix(v2)
        return cache[v2]
    psi = model._intro_vec(cs.intro[0][2])
    for _w, _n, ang in cs.intro[1:]:
        psi = torch.einsum("...,j->...j", psi, model._intro_vec(ang))
    for v, wa, wp in cs.events:
        psi = apply_real(psi, blk(v, smap), (wa, wp))
        psi = psi / (psi.norm() + 1e-12)
    vseq, qa, qp = cs.question
    qw = (qa, qp)
    for qv in reversed(vseq):
        psi = apply_real(psi, blk(qv, qmap).T, qw)
    return [real_prob_all_zero_on(apply_real(psi, Q, qw), qw)
            for Q in (model.q_pos, model.q_neg)]


def vals(model, arm, cs, smap=None, qmap=None):
    f = vals_quantum if is_quantum(arm) else vals_classical
    return f(model, cs, smap or {}, qmap or {})


def predict(model, arm, cs, smap=None, qmap=None):
    with torch.no_grad():
        v = vals(model, arm, cs, smap, qmap)
    return 0 if float(v[0]) >= float(v[1]) else 1


def accuracy(model, arm, compiled, smap=None, qmap=None):
    ok = sum(1 for cs in compiled
             if predict(model, arm, cs, smap, qmap) == cs.answer)
    return ok / len(compiled)


def loss_of(model, arm, cs):
    v = vals(model, arm, cs)
    logits = torch.log(torch.stack(v) + 1e-12)
    return torch.nn.functional.cross_entropy(
        logits.unsqueeze(0), torch.tensor([cs.answer]))


# ---------------------------------------------------------------- training
def train_run(arm, seed, lr, batch, data, verbs, max_epochs,
              train_items=None, select_on_val=True, tag=""):
    torch.manual_seed(seed)
    model = make_arm(arm, verbs, seed)
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
        losses = []
        for s in range(0, len(order), batch):
            bat = [train[i] for i in order[s:s + batch]]
            opt.zero_grad()
            bl = 0.0
            for cs in bat:
                bl = bl + loss_of(model, arm, cs)
            bl = bl / len(bat)
            bl.backward()
            assert math.isfinite(grad_global_norm(model)), \
                "non-finite grad %s s%d" % (arm, seed)
            opt.step()
            steps += 1
            losses.append(bl.item())
        va = accuracy(model, arm, val)
        if va > best["val_acc"]:
            best = {"val_acc": va, "train_acc": accuracy(model, arm, train),
                    "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
        print("[%s%s s%d lr%.3f b%d] ep %2d/%d loss=%.4f val=%.3f best@%d(%.3f)"
              " %.1fs" % (tag, arm, seed, lr, batch, epoch, max_epochs,
                          sum(losses) / len(losses), va, best["epoch"],
                          best["val_acc"], time.time() - t0), flush=True)
    model.load_state_dict(best["state"])
    te = accuracy(model, arm, data["test"]) if select_on_val else best["train_acc"]
    return model, {
        "arm": arm, "seed": seed, "lr": lr, "batch": batch,
        "steps_taken": steps, "best_epoch": best["epoch"],
        "val_acc": best["val_acc"], "train_acc": best["train_acc"],
        "test_acc": te,
        "gate_met": bool(best["train_acc"] >= TRAIN_GATE
                         and best["val_acc"] >= VAL_GATE)}


def twin_pairs(items, angles, n):
    by = {}
    for it in items:
        if it["split"] == "train":
            by.setdefault(it["pair_id"], []).append(it)
    out = []
    for p in [q for q in sorted(by) if len(by[q]) == 2][:n]:
        a, b = sorted(by[p], key=lambda x: x["id"])
        out.append((comp.compile_story(a, angles), comp.compile_story(b, angles)))
    return out


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if not args.smoke:
        os.makedirs(CKPT_DIR, exist_ok=True)
    n_seeds = 1 if args.smoke else N_SEEDS
    max_ep = 2 if args.smoke else MAX_EPOCHS
    grid = GRID[-1:] if args.smoke else GRID
    memo_ep = 2 if args.smoke else MEMO_EPOCHS
    n_perms = 5 if args.smoke else N_PERMS

    with open(DATA, encoding="utf-8") as f:
        blob = json.load(f)
    items, meta = blob["items"], blob["meta"]
    sha = comp.canonical_sha256(items)
    if sha != meta["dataset_sha256"] or sha != EXPECTED_SHA:
        raise RuntimeError("VT provenance FAILED: %s" % sha)
    angles, _e, emb_sha = comp.load_embeddings()
    verbs = sorted(meta["verb_inventory"].keys())
    data = {sp: comp.compile_all([i for i in items if i["split"] == sp], angles)
            for sp in ("train", "val", "test")}
    print("[exp51] VT sha=%s train/val/test=%d/%d/%d verbs=%d"
          % (sha[:12], len(data["train"]), len(data["val"]),
             len(data["test"]), len(verbs)), flush=True)

    # ---------- doc 35 SS3 tied-block theorem (BLOCKING)
    tw = twin_pairs(items, angles, 100)
    worst, ident, ntot, corr = 0.0, 0, 0, 0
    for seed in range(5):
        m = make_arm("T", verbs, seed)
        with torch.no_grad():
            for ca, cb in tw:
                def sv(cs):
                    psi = sim.init_state(cs.K)
                    for w, _n, ang in cs.intro:
                        psi = sim.apply_gate(psi, sim.ry(ang), (w,))
                    for v, wa, wp in cs.events:
                        psi = sim.apply_gate(psi, m.verb_matrix(v), (wa, wp))
                    return psi
                worst = max(worst, (sv(ca) - sv(cb)).abs().max().item())
                pa, pb = predict(m, "T", ca), predict(m, "T", cb)
                ident += (pa == pb)
                ntot += 1
                corr += (pa == ca.answer) + (pb == cb.answer)
    sentinel = {"n_pairs": len(tw), "n_models": 5,
                "max_statevector_delta": worst, "delta_bar": 1e-12,
                "frac_identical": ident / ntot,
                "tied_accuracy": corr / (2 * ntot),
                "tied_sentinel_ok": bool(worst <= 1e-12 and ident == ntot
                                         and corr / (2 * ntot) == 0.5)}
    print("[exp51] tied sentinel: delta=%.3e identical=%.4f acc=%.4f ok=%s"
          % (worst, ident / ntot, corr / (2 * ntot),
             sentinel["tied_sentinel_ok"]), flush=True)
    if not sentinel["tied_sentinel_ok"]:
        print("[exp51] HALT: doc 35 SS3 -- construction broken, no result "
              "reported.", flush=True)
        return 1

    # ---------- F3 analogue: verb-twin init audit
    audits = {}
    for arm in ARMS:
        if arm == "T":
            continue
        fr = []
        for seed in range(n_seeds):
            m = make_arm(arm, verbs, seed)
            same = sum(1 for ca, cb in tw[:AUDIT_PAIRS]
                       if predict(m, arm, ca) == predict(m, arm, cb))
            fr.append(same / len(tw[:AUDIT_PAIRS]))
        audits[arm] = {"per_seed": fr, "frac": sum(fr) / len(fr),
                       "f3_bar": F3_BAR,
                       "f3_pass": bool(sum(fr) / len(fr) <= F3_BAR)}
        print("[exp51] %s verb-twin init audit: %.3f f3_pass=%s"
              % (arm, audits[arm]["frac"], audits[arm]["f3_pass"]), flush=True)

    # ---------- memorisation controls (T must FAIL)
    subset = memo_subset(items, angles)
    memo = {}
    for arm in ARMS:
        best = 0.0
        runs = []
        for lr, b in grid:
            for s in range(1 if args.smoke else 2):
                _m, d = train_run(arm, s, lr, b, data, verbs, memo_ep,
                                  train_items=subset, select_on_val=False,
                                  tag="memo:")
                best = max(best, d["train_acc"])
                runs.append({"lr": lr, "batch": b, "seed": s,
                             "best_train_acc": d["train_acc"]})
        memo[arm] = {"runs": runs, "best_train_acc": best,
                     "arm_trainable": bool(best >= MEMO_BAR)}
        print("[exp51] %s memorisation: best=%.3f trainable=%s"
              % (arm, best, memo[arm]["arm_trainable"]), flush=True)

    # ---------- Phase-1 style grid
    per_arm, models = {}, {}
    for arm in ARMS:
        runs, keep = [], []
        for lr, b in grid:
            for s in range(n_seeds):
                key = "VT_%s_lr%s_b%d_s%d" % (arm, lr, b, s)
                ckj = os.path.join(CKPT_DIR, key + ".json")
                ckp = os.path.join(CKPT_DIR, key + ".pt")
                if not args.smoke and os.path.exists(ckj) and os.path.exists(ckp):
                    with open(ckj, encoding="utf-8") as f:
                        d = json.load(f)
                    m = make_arm(arm, verbs, s)
                    m.load_state_dict(torch.load(ckp, weights_only=True))
                    print("[ckpt] %s loaded" % key, flush=True)
                else:
                    m, d = train_run(arm, s, lr, b, data, verbs, max_ep,
                                     tag="VT:")
                    if not args.smoke:
                        with open(ckj, "w", encoding="utf-8") as f:
                            json.dump(d, f, ensure_ascii=False)
                        torch.save(m.state_dict(), ckp)
                runs.append(d)
                keep.append(m)
        per_arm[arm] = runs
        best_i = max(range(len(runs)), key=lambda i: runs[i]["val_acc"])
        models[arm] = keep[best_i]
        print("[exp51] %s: gate met %d/%d, best val=%.3f test=%.3f"
              % (arm, sum(1 for r in runs if r["gate_met"]), len(runs),
                 max(r["val_acc"] for r in runs),
                 max(r["test_acc"] for r in runs)), flush=True)

    # ---------- question-only permutation control
    rng = random.Random(PERM_SEED)
    perms = []
    for _ in range(n_perms):
        q = verbs[:]
        while True:
            rng.shuffle(q)
            if q != verbs:
                break
        perms.append(dict(zip(verbs, q)))
    qonly = {}
    for arm in ARMS:
        m = models[arm]
        base = accuracy(m, arm, data["test"])
        accs = [accuracy(m, arm, data["test"], {}, vm) for vm in perms]
        mean = sum(accs) / len(accs)
        qonly[arm] = {"base_acc": base, "mean_question_only_acc": mean,
                      "min": min(accs), "max": max(accs),
                      "binder_expectation": BINDER_EXPECTATION,
                      "collapses": bool(mean <= COLLAPSE_BAR)}
        print("[exp51] %s question-only: base=%.4f perm=%.4f collapses=%s "
              "(binder expectation %.4f)"
              % (arm, base, mean, qonly[arm]["collapses"], BINDER_EXPECTATION),
              flush=True)

    verdicts = {"tied_sentinel_ok": True,
                "t_fails_memorisation": bool(not memo["T"]["arm_trainable"])}
    for arm in ARMS:
        if arm == "T":
            continue
        valid = memo[arm]["arm_trainable"] and audits[arm]["f3_pass"]
        met = any(r["gate_met"] for r in per_arm[arm])
        verdicts["vt_learnable_%s" % arm] = met if valid else "UNDECIDABLE"
        verdicts["arm_trainable_%s" % arm] = memo[arm]["arm_trainable"]
        verdicts["f3_pass_%s" % arm] = audits[arm]["f3_pass"]
        verdicts["question_only_collapses_%s" % arm] = qonly[arm]["collapses"]
    verdicts["verb_binding_learnable"] = bool(
        any(verdicts.get("vt_learnable_%s" % a) is True for a in ("A1", "A2", "A3")))

    out = {"experiment": "exp51", "spec": DESIGN_DOC,
           "question": "does any arm learn a rung that REQUIRES verb identity?",
           "platform": platform.platform(),
           "python": platform.python_version(), "torch": torch.__version__,
           "rung": "VT", "dataset_sha256": sha, "embeddings_sha256": emb_sha,
           "grid": [{"lr": lr, "batch": b} for lr, b in grid],
           "n_seeds": n_seeds, "max_epochs": max_ep,
           "gate": {"train": TRAIN_GATE, "val": VAL_GATE},
           "tied_sentinel": sentinel, "init_audits_f3": audits,
           "memorisation_controls": memo, "per_arm": per_arm,
           "question_only_permutation": qonly,
           "verdicts": verdicts,
           "meta_notes": {"scope": SCOPE_NOTE, "c1_not_run": C1_NOT_RUN,
                          "binder_expectation_derivation": (
                              "doc 35 SS5.2, derived before data existed: over "
                              "719 non-identity permutations of 6 verbs, twin "
                              "YES is right only if pi(v1)=v1 (119/719) and "
                              "twin NO only if pi(v1)!=v2 (1-120/719), giving "
                              "359/719 = 0.4993 -- a perfect binder collapses "
                              "to CHANCE.")},
           "git_note": git_note(), "wall_time_sec": time.time() - t0}
    print("\n[exp51] VERDICTS")
    for k, v in verdicts.items():
        print("   %-32s %s" % (k, v))
    if args.smoke:
        print("[exp51] smoke -- no results file")
        return 0
    p = fresh_results_path("results_exp51")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[exp51] wrote %s (%.1f min)" % (p, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
