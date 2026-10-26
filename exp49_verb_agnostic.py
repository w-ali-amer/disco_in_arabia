# -*- coding: utf-8 -*-
"""exp49_verb_agnostic.py -- doc 33: what is the verb-agnostic route, and is
C1 blind to it?

Three arms that post real accuracy fail C1, and A4 beats B5 on L2 by 11.9pp
at p=1e-4 while sitting BELOW its own 719-permutation 95th percentile.

Structural hypothesis (doc 33 SS2, stated before any measurement): C1
permutes the verb->block assignment CONSISTENTLY across story events and the
question verb. On a Q0 item the story applies block pi(v) on the queried
pair and the question inverse-preps with the SAME pi(v), so the relabeling
CANCELS on the queried event and only distractor events survive. If that
term dominates, C1 is structurally near-blind here -- a finding about the
CONTROL, not the models.

  P1  prediction stability under consistent permutation
      c1_structurally_blind      iff mean unchanged-fraction >= 0.90
  P2  question-only permutation (breaks the cancellation; the sharp test)
      question_swap_collapses    iff mean accuracy <= 0.55
  P3  tied-block ablation (decisive regardless of P1/P2)
      verb_blind_matches_{L1,L2} iff tied best test within 2pp of untied

Retracts nothing: every content_earned boolean stands as computed against
the criterion it was registered on. exp49 changes how they are READ.
"""

import argparse
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
import exp42_sim_adapter as sim
from exp42_models import QuantumStoryModel
from exp42_run import fresh_results_path, git_note, grad_global_norm
from exp45_model import TwoQubitReferentModel

DESIGN_DOC = ("qnlp_private_docs/33_exp49_verb_agnostic_route.md "
              "(commit 7e82b41, FROZEN)")
CKPT_DIR = "exp49_ckpt"
N_PERMS = 100
BLIND_BAR = 0.90
COLLAPSE_BAR = 0.55
MATCH_BAR = 0.02
GRID = [(0.005, 1), (0.005, 8), (0.02, 1), (0.02, 8)]
N_SEEDS = 2
MAX_EPOCHS = 60
PERM_SEED = 49000
RNG_BASE = 49000

SCOPE_NOTE = (
    "doc 33 SS5: retracts nothing. content_earned booleans across "
    "exp42/43b/44/48 stand as computed against the criteria they were "
    "registered on (Rule 4); exp49 changes how they are READ, never their "
    "values. This is NOT a claim that the exp41 generator leaked -- V1-V5 "
    "tested whether shortcut models can solve these datasets and they "
    "cannot; a verb-agnostic route inside a COMPOSITIONAL model is a "
    "different thing from a bag-of-words shortcut and this doc does not "
    "conflate them. No quantum reading: all three probes are about task "
    "structure and control adequacy. Simulation only, 6 verbs, two rungs.")


# ----------------------------------------------------- split-map forwards
def _vals_1q(model, cs, story_map, q_map):
    mats = {}

    def blk(v, m):
        v2 = m.get(v, v)
        if v2 not in mats:
            mats[v2] = model.verb_matrix(v2)
        return mats[v2]

    psi = sim.init_state(cs.K)
    for w, _n, ang in cs.intro:
        psi = sim.apply_gate(psi, sim.ry(ang), (w,))
    for v, wa, wp in cs.events:
        psi = sim.apply_gate(psi, blk(v, story_map), (wa, wp))
    vseq, qa, qp = cs.question
    qw = (qa, qp)
    for qv in reversed(vseq):
        psi = sim.apply_gate(psi, blk(qv, q_map).conj().T, qw)
    return [sim.prob_all_zero_on(
        sim.apply_gate(psi, e.matrix().conj().T, qw), qw).item()
        for e in (model.q_pos, model.q_neg)]


def _vals_2q(model, cs, story_map, q_map):
    mats = {}

    def blk(v, m):
        v2 = m.get(v, v)
        if v2 not in mats:
            mats[v2] = model.verb_matrix(v2)
        return mats[v2]

    psi = sim.init_state(2 * cs.K)
    for w, _n, ang in cs.intro:
        psi = sim.apply_gate(psi, sim.ry(ang), (2 * w,))
    for v, wa, wp in cs.events:
        psi = sim.apply_gate(psi, blk(v, story_map),
                             (2 * wa, 2 * wa + 1, 2 * wp, 2 * wp + 1))
    vseq, qa, qp = cs.question
    qw = (2 * qa, 2 * qa + 1, 2 * qp, 2 * qp + 1)
    for qv in reversed(vseq):
        psi = sim.apply_gate(psi, blk(qv, q_map).conj().T, qw)
    return [sim.prob_all_zero_on(
        sim.apply_gate(psi, e.matrix().conj().T, qw), qw).item()
        for e in (model.q_pos, model.q_neg)]


def predict_split(model, cs, story_map, q_map, two_q):
    with torch.no_grad():
        v = (_vals_2q if two_q else _vals_1q)(model, cs, story_map, q_map)
    return 0 if v[0] >= v[1] else 1


def preds(model, compiled, story_map, q_map, two_q):
    return [predict_split(model, cs, story_map, q_map, two_q)
            for cs in compiled]


# ----------------------------------------------------------- tied models
def tied_model(arm, verbs, seed):
    """Doc 33 P3: all verbs share ONE block. Implemented by overriding
    verb_matrix to a canonical verb, so exactly one block enters the
    computation graph and the rest receive no gradient. The EFFECTIVE
    per-verb parameter count is therefore 1 block, not len(verbs)."""
    if arm == "A4":
        m = TwoQubitReferentModel("A4", verbs, seed)
    else:
        m = QuantumStoryModel(arm, verbs, seed)
    canon = sorted(verbs)[0]
    base = m.verb_matrix

    def tied(_v, _canon=canon):
        return base(_canon)
    m.verb_matrix = tied
    return m


def evaluate(model, compiled, two_q):
    ident = {}
    c = [1 if predict_split(model, cs, ident, ident, two_q) == cs.answer else 0
         for cs in compiled]
    return sum(c) / len(c), c


def train_tied(arm, seed, lr, batch, data, verbs, max_epochs, two_q, tag):
    torch.manual_seed(seed)
    model = tied_model(arm, verbs, seed)
    opt = torch.optim.Adam([p for p in model.parameters()], lr=lr)
    rng = random.Random(RNG_BASE + seed)
    train = data["train"]
    best = {"val_acc": -1.0, "train_acc": -1.0, "state": None}
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
                vals = (_vals_2q if two_q else _vals_1q)(model, cs, {}, {})
                # differentiable path: recompute with grad
                bl = bl + _loss(model, cs, two_q)
            bl = bl / len(bat)
            bl.backward()
            assert math.isfinite(grad_global_norm(model))
            opt.step()
            losses.append(bl.item())
        va, _ = evaluate(model, data["val"], two_q)
        if va > best["val_acc"]:
            ta, _ = evaluate(model, train, two_q)
            best = {"val_acc": va, "train_acc": ta,
                    "state": copy.deepcopy(model.state_dict())}
        print("[%s tied %s s%d lr%.3f b%d] ep %2d/%d loss=%.4f val=%.3f "
              "best=%.3f %.1fs" % (tag, arm, seed, lr, batch, epoch,
                                   max_epochs, sum(losses) / len(losses), va,
                                   best["val_acc"], time.time() - t0),
              flush=True)
    model.load_state_dict(best["state"])
    te, _ = evaluate(model, data["test"], two_q)
    return {"arm": "tied_" + arm, "seed": seed, "lr": lr, "batch": batch,
            "val_acc": best["val_acc"], "train_acc": best["train_acc"],
            "test_acc": te}


def _loss(model, cs, two_q):
    v = (_vals_2q_t if two_q else _vals_1q_t)(model, cs)
    logits = torch.log(torch.stack(v) + 1e-12)
    return torch.nn.functional.cross_entropy(
        logits.unsqueeze(0), torch.tensor([cs.answer]))


def _vals_1q_t(model, cs):
    m = model.verb_matrix
    psi = sim.init_state(cs.K)
    for w, _n, ang in cs.intro:
        psi = sim.apply_gate(psi, sim.ry(ang), (w,))
    for v, wa, wp in cs.events:
        psi = sim.apply_gate(psi, m(v), (wa, wp))
    vseq, qa, qp = cs.question
    qw = (qa, qp)
    for qv in reversed(vseq):
        psi = sim.apply_gate(psi, m(qv).conj().T, qw)
    return [sim.prob_all_zero_on(
        sim.apply_gate(psi, e.matrix().conj().T, qw), qw)
        for e in (model.q_pos, model.q_neg)]


def _vals_2q_t(model, cs):
    m = model.verb_matrix
    psi = sim.init_state(2 * cs.K)
    for w, _n, ang in cs.intro:
        psi = sim.apply_gate(psi, sim.ry(ang), (2 * w,))
    for v, wa, wp in cs.events:
        psi = sim.apply_gate(psi, m(v),
                             (2 * wa, 2 * wa + 1, 2 * wp, 2 * wp + 1))
    vseq, qa, qp = cs.question
    qw = (2 * qa, 2 * qa + 1, 2 * qp, 2 * qp + 1)
    for qv in reversed(vseq):
        psi = sim.apply_gate(psi, m(qv).conj().T, qw)
    return [sim.prob_all_zero_on(
        sim.apply_gate(psi, e.matrix().conj().T, qw), qw)
        for e in (model.q_pos, model.q_neg)]


def load_rung(r):
    with open("stories_exp43b_%s.json" % r, encoding="utf-8") as f:
        d = json.load(f)
    return d["items"], d["meta"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if not args.smoke:
        os.makedirs(CKPT_DIR, exist_ok=True)
    n_perms = 5 if args.smoke else N_PERMS
    grid = GRID[-1:] if args.smoke else GRID
    max_ep = 2 if args.smoke else MAX_EPOCHS
    n_seeds = 1 if args.smoke else N_SEEDS

    angles, _e, emb_sha = comp.load_embeddings()
    rungs = {}
    for r in ("L1", "L2"):
        items, meta = load_rung(r)
        rungs[r] = {
            "meta": meta, "verbs": sorted(meta["verb_inventory"].keys()),
            "data": {sp: comp.compile_all(
                [i for i in items if i["split"] == sp], angles)
                for sp in ("train", "val", "test")}}

    # arms with committed checkpoints: (label, rung, ctor, ckpt paths, two_q)
    targets = [
        ("A2_L1", "L1", lambda v, s: QuantumStoryModel("A2", v, s),
         ["exp43b_phase2_ckpt/p2_A2_s%d.pt" % s for s in range(5)], False),
        ("A4_L2", "L2", lambda v, s: TwoQubitReferentModel("A4", v, s),
         ["exp48_ckpt/L2_A4_lr0.02_b8_s%d.pt" % s for s in range(2, 5)]
         + ["exp45b_ckpt/L2_A4_lr0.02_b8_s%d.pt" % s for s in (0, 1)], True),
    ]

    rng = random.Random(PERM_SEED)
    p1, p2 = {}, {}
    for label, rung, ctor, paths, two_q in targets:
        verbs = rungs[rung]["verbs"]
        test = rungs[rung]["data"]["test"]
        models = []
        for i, p in enumerate(paths):
            if not os.path.exists(p):
                continue
            m = ctor(verbs, i)
            m.load_state_dict(torch.load(p, weights_only=True))
            models.append(m)
        if not models:
            print("[exp49] %s: no checkpoints found, skipping" % label)
            continue
        base = [preds(m, test, {}, {}, two_q) for m in models]
        base_acc = sum(sum(1 for a, cs in zip(b, test) if a == cs.answer)
                       for b in base) / (len(models) * len(test))
        perms = []
        for _ in range(n_perms):
            q = verbs[:]
            while True:
                rng.shuffle(q)
                if q != verbs:
                    break
            perms.append(dict(zip(verbs, q)))
        # P1 consistent permutation -> prediction stability
        unchanged, permacc = [], []
        for vmap in perms:
            for m, b in zip(models, base):
                pr = preds(m, test, vmap, vmap, two_q)
                unchanged.append(sum(1 for x, y in zip(pr, b) if x == y)
                                 / len(pr))
                permacc.append(sum(1 for x, cs in zip(pr, test)
                                   if x == cs.answer) / len(pr))
        # P2 question-only permutation
        qacc = []
        for vmap in perms:
            for m in models:
                pr = preds(m, test, {}, vmap, two_q)
                qacc.append(sum(1 for x, cs in zip(pr, test)
                                if x == cs.answer) / len(pr))
        p1[label] = {"n_models": len(models), "n_perms": len(perms),
                     "base_acc": base_acc,
                     "mean_unchanged_fraction": sum(unchanged) / len(unchanged),
                     "min_unchanged_fraction": min(unchanged),
                     "mean_consistent_perm_acc": sum(permacc) / len(permacc)}
        p2[label] = {"mean_question_only_perm_acc": sum(qacc) / len(qacc),
                     "min": min(qacc), "max": max(qacc)}
        print("[exp49] %s base=%.4f | P1 unchanged=%.4f consistent-perm "
              "acc=%.4f | P2 question-only acc=%.4f"
              % (label, base_acc, p1[label]["mean_unchanged_fraction"],
                 p1[label]["mean_consistent_perm_acc"],
                 p2[label]["mean_question_only_perm_acc"]), flush=True)

    # ---- P3 tied-block ablation
    p3 = {}
    for arm, rung, two_q, untied_ref in (("A2", "L1", False, 0.7993),
                                         ("A4", "L2", True, 0.7167)):
        verbs = rungs[rung]["verbs"]
        runs = []
        for lr, b in grid:
            for s in range(n_seeds):
                key = "tied_%s_%s_lr%s_b%d_s%d" % (arm, rung, lr, b, s)
                ckj = os.path.join(CKPT_DIR, key + ".json")
                if not args.smoke and os.path.exists(ckj):
                    with open(ckj, encoding="utf-8") as f:
                        d = json.load(f)
                else:
                    d = train_tied(arm, s, lr, b, rungs[rung]["data"], verbs,
                                   max_ep, two_q, rung)
                    if not args.smoke:
                        with open(ckj, "w", encoding="utf-8") as f:
                            json.dump(d, f)
                runs.append(d)
        best = max(r["test_acc"] for r in runs)
        p3["%s_%s" % (arm, rung)] = {
            "runs": runs, "best_tied_test_acc": best,
            "untied_reference_test_acc": untied_ref,
            "gap": untied_ref - best,
            "matches_within_2pp": bool(abs(untied_ref - best) <= MATCH_BAR)}
        print("[exp49] tied %s on %s: best test=%.4f vs untied %.4f (gap "
              "%+.4f) matches=%s" % (arm, rung, best, untied_ref,
                                     untied_ref - best,
                                     p3["%s_%s" % (arm, rung)]
                                     ["matches_within_2pp"]), flush=True)

    blind = [v["mean_unchanged_fraction"] for v in p1.values()]
    qcol = [v["mean_question_only_perm_acc"] for v in p2.values()]
    verdicts = {
        "c1_structurally_blind": bool(blind and
                                      sum(blind) / len(blind) >= BLIND_BAR),
        "mean_unchanged_fraction": (sum(blind) / len(blind)) if blind else None,
        "question_swap_collapses": bool(qcol and
                                        sum(qcol) / len(qcol) <= COLLAPSE_BAR),
        "mean_question_only_acc": (sum(qcol) / len(qcol)) if qcol else None,
        "verb_blind_matches_L1": p3.get("A2_L1", {}).get("matches_within_2pp"),
        "verb_blind_matches_L2": p3.get("A4_L2", {}).get("matches_within_2pp"),
    }
    out = {"experiment": "exp49", "spec": DESIGN_DOC,
           "platform": platform.platform(),
           "python": platform.python_version(), "torch": torch.__version__,
           "embeddings_sha256": emb_sha,
           "dataset_sha256": {r: rungs[r]["meta"]["dataset_sha256"]
                              for r in ("L1", "L2")},
           "P1_prediction_stability": p1, "P2_question_only_permutation": p2,
           "P3_tied_block": p3, "verdicts": verdicts,
           "meta_notes": {"scope": SCOPE_NOTE},
           "git_note": git_note(), "wall_time_sec": time.time() - t0}
    print("\n[exp49] VERDICTS")
    for k, v in verdicts.items():
        print("   %-30s %s" % (k, v))
    if args.smoke:
        print("[exp49] smoke -- no results file")
        return 0
    path = fresh_results_path("results_exp49")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[exp49] wrote %s (%.1f min)" % (path, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
