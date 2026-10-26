# -*- coding: utf-8 -*-
"""exp43c_arm_validity.py -- arm validity controls (doc 24 SS5, FROZEN).

Runs FIRST and gates Phase 2 (doc 24 SS5). Two pre-registered parts:

(ii) SS5.2 memorisation control: for each arm, train on a FIXED 64-item
  subset of the L1 train split (the first 32 twin pairs by sorted
  pair_id), no held-out set, over the frozen grid {lr 0.005, 0.02} x
  {batch 1, 8}, 2 seeds, <=300 epochs.
      arm_trainable_X = true iff any config reaches train >= 0.95.
  Control-of-the-control: A1 (leak sentinel, SWAP-commuting by
  construction, Lemma-1-pinned to <=0.5 on twins) MUST fail:
  a1_fails_memorisation = true is the expected outcome. If A1 passes,
  exp43c is reported INCONCLUSIVE and acted on in no way.

  ADDITIVE AMENDMENT (flagged): doc 24 SS5.2 lists {A1, A3, B1, B2, B3}.
  A2 is INCLUDED here because F5 (doc 24 SS6) requires every arm entering
  a comparison to have passed SS5.2, and A2 is the frozen tier-q
  comparator (doc 23 / doc 24 SS7). Adding a control our own comparator
  must pass can only make our own claim harder to establish -- the doc 24
  SS3 admissibility direction. Recorded in the results meta.

(iii) SS5.3 matched-initialisation B1: same class, same wiring, same
  symmetric core generators, same 18 params/verb -- ONLY the initial
  values change: each dressing factor becomes a rotation by an angle
  uniform on [0, 2pi); q_pos/q_neg drawn generically rather than
  near-identity; all draws via torch.rand, never torch.randn (F4).
  Re-run the EXACT frozen L1 Phase-1 grid (2 lrs x 2 batches x 2 seeds,
  <=60 epochs, gate train>=0.8 AND val>=0.7). Re-run the init audit on
  the new init -> b1_matched_init_twin_collapse_at_init (F3: <= 0.60).
      b1_matched_init_learnable_L1 = true iff any config meets the gate.

Consequences are frozen in doc 24 SS5.4 and quoted verbatim in the
results JSON. Results (never overwritten) -> results_exp43c.json.
Checkpointing: one JSON per run under exp43c_ckpt/; finished runs are
skipped on relaunch.
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
from exp42_baselines import (BagBaseline, ClassicalDisCoCirc, SeqBaseline,
                             _pair_swap, build_vocab)
from exp42_models import QuantumStoryModel
from exp42_run import fresh_results_path, git_note, grad_global_norm

CKPT_DIR = "exp43c_ckpt"
RUNG = "L1"

GRID_LRS = (0.005, 0.02)                                  # FROZEN (doc 24)
GRID_BATCHES = (1, 8)                                     # FROZEN
SEEDS = (0, 1)                                            # FROZEN
MEMO_MAX_EPOCHS = 300                                     # FROZEN
MEMO_GATE = 0.95                                          # FROZEN
MEMO_N_PAIRS = 32                                         # FROZEN (64 items)
MEMO_ARMS = ("A1", "A2", "A3", "B1", "B2", "B3")          # A2 = amendment
GRID_MAX_EPOCHS = 60                                      # FROZEN (Phase 1)
GATE_TRAIN, GATE_VAL = 0.8, 0.7                           # FROZEN (Phase 1)
AUDIT_N_PAIRS = 200                                       # = exp43b_b1_init_audit
AUDIT_N_GRAD = 64
MATCHED_INIT_SEED_OFFSET = 51000   # fresh torch.rand stream for SS5.3 draws

CONSEQUENCE_TABLE_5_4 = {
    ("true", "any"):
        "The Phase-1 L1 asymmetry is an **initialisation artifact**. "
        "Reported as such, leading with it (Rule 4). The original B1 "
        "numbers stay in the record. **No quantum-vs-classical claim is "
        "drawn from Phase 1.** Phase 2 runs with matched-init B1 as the "
        "tier-q comparator.",
    ("false", "true"):
        "The asymmetry survives its strongest challenge. Phase 2 "
        "proceeds; the audit and this control are reported alongside "
        "every comparative number as the reason a reader should believe "
        "it.",
    ("false", "false"):
        "B1 is an untrained instrument. **tier-q as defined in doc 23 "
        "cannot be evaluated: it is reported as UNDECIDABLE**, never as "
        "passed or failed. Substituting a different comparator requires "
        "a new doc.",
}

AMENDMENT_NOTE_A2 = (
    "ADDITIVE AMENDMENT: A2 is included in the SS5.2 memorisation control "
    "although doc 24 SS5.2 lists {A1, A3, B1, B2, B3}. Reason: F5 (doc 24 "
    "SS6) requires every arm entering a comparison to have passed SS5.2, "
    "and A2 is the frozen tier-q comparator (doc 23 / doc 24 SS7); without "
    "this control A2 could not enter Phase 2 at all. The amendment is "
    "additive in the doc 24 SS3 sense: it adds a control our own "
    "comparator must pass, which can only make our own claim strictly "
    "harder to establish, never easier.")

B2_AUGMENTATION_NOTE = (
    "Name augmentation (an exp42 training-protocol feature for B2's "
    "generalisation) is DISABLED in the memorisation control: the control "
    "measures pure fitting/optimisation capacity on a fixed 64-item set, "
    "and per-presentation name resampling changes the target being fit. "
    "Flagged as an execution resolution; doc 24 SS5.2 does not mention "
    "augmentation.")

GENERIC_READOUT_NOTE = (
    "Execution resolution of SS5.3 'q_pos, q_neg drawn generically': each "
    "readout matrix is initialised at a generic SO(4) rotation built as "
    "the ordered product of 6 Givens rotations over index pairs "
    "(0,1),(0,2),(0,3),(1,2),(1,3),(2,3), each angle uniform on [0, 2pi) "
    "via torch.rand -- the real analogue of the A arms' generic uniform-"
    "angle init, far from identity, symmetry-breaking, and F4-compliant "
    "(torch.rand only). The PARAMETERISATION is unchanged (full 4x4 real "
    "matrix, 16 trained entries); only initial values differ. Dressing "
    "factors are 2x2 rotations R(theta), theta uniform on [0, 2pi), per "
    "the doc's explicit wording. Draws use a fresh generator seeded "
    "51000+seed so the matched draws are a recorded, independent stream; "
    "the symmetric-core init is inherited unchanged from B1-as-shipped "
    "(torch.rand(2)*2pi, generator seeded with the run seed).")


# ----------------------------------------------------------- matched B1
def _rot2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]])


def _givens(D, i, j, theta):
    G = torch.eye(D)
    c, s = math.cos(theta), math.sin(theta)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = -s
    G[j, i] = s
    return G


def _so4(angles):
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    M = torch.eye(4)
    for (i, j), th in zip(pairs, angles):
        M = M @ _givens(4, i, j, float(th))
    return M


class MatchedInitB1(ClassicalDisCoCirc):
    """doc 24 SS5.3: B1 with matched initialisation. Same class, same
    wiring, same symmetric core generators, same 18 params/verb; ONLY the
    initial values of the dressing factors and q_pos/q_neg change. All
    draws torch.rand (F4)."""

    def __init__(self, verbs, seed, d_ref=2):
        super().__init__(verbs, seed, d_ref=d_ref)
        assert d_ref == 2
        g = torch.Generator().manual_seed(MATCHED_INIT_SEED_OFFSET + seed)
        with torch.no_grad():
            for v in self.verbs:
                ang = torch.rand(4, generator=g) * 2 * math.pi
                self.dress[v].data = torch.stack(
                    [_rot2(float(a)) for a in ang])
            self.q_pos.data = _so4(torch.rand(6, generator=g) * 2 * math.pi)
            self.q_neg.data = _so4(torch.rand(6, generator=g) * 2 * math.pi)
        pc = self.param_counts()
        assert pc["per_verb"] == 18 and pc["total_trained"] == \
            18 * len(self.verbs) + 32, pc


# ---------------------------------------------------------------- data
def load_l1():
    with open("stories_exp43b_%s.json" % RUNG, encoding="utf-8") as f:
        data = json.load(f)
    items, meta = data["items"], data["meta"]
    sha = comp.canonical_sha256(items)
    if sha != meta["dataset_sha256"]:
        raise RuntimeError("L1 dataset hash mismatch")
    return items, meta, sha


def memo_subset(items):
    """First MEMO_N_PAIRS twin pairs of the L1 train split by sorted
    pair_id; both twins, sorted by item id. FROZEN in doc 24 SS5.2."""
    train = [i for i in items if i["split"] == "train"]
    by_pair = {}
    for it in train:
        by_pair.setdefault(it["pair_id"], []).append(it)
    pairs = [p for p in sorted(by_pair) if len(by_pair[p]) == 2]
    sel = pairs[:MEMO_N_PAIRS]
    sub = [it for p in sel for it in sorted(by_pair[p], key=lambda x: x["id"])]
    assert len(sub) == 2 * MEMO_N_PAIRS
    return sub, sel


def evaluate(model, compiled):
    correct = 0
    with torch.no_grad():
        for cs in compiled:
            if model.predict(cs) == cs.answer:
                correct += 1
    return correct / len(compiled)


def make_model(arm, verbs, vocab, seed):
    if arm in ("A1", "A2", "A3"):
        return QuantumStoryModel(arm, verbs, seed)
    if arm == "B1":
        return ClassicalDisCoCirc(verbs, seed, d_ref=2)
    if arm == "B1M":
        return MatchedInitB1(verbs, seed, d_ref=2)
    if arm == "B2":
        return SeqBaseline(vocab, seed, variant="matched")
    if arm == "B3":
        return BagBaseline(vocab, seed)
    raise ValueError(arm)


# ------------------------------------------------- SS5.2 memorisation
def memo_run(arm, lr, batch, seed, memo64, verbs, vocab, max_epochs):
    torch.manual_seed(seed)
    model = make_model(arm, verbs, vocab, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(43000 + seed)

    curves = {"loss": [], "clean_train_acc": []}
    reached, reach_epoch, best_acc = False, None, -1.0
    for epoch in range(1, max_epochs + 1):
        order = list(range(len(memo64)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), batch):
            bat = [memo64[i] for i in order[start:start + batch]]
            opt.zero_grad()
            bloss = 0.0
            for cs in bat:
                loss, _pred = model.story_loss(cs)
                bloss = bloss + loss
            bloss = bloss / len(bat)
            bloss.backward()
            gn = grad_global_norm(model)
            assert math.isfinite(gn), \
                "non-finite grad (memo %s lr%g b%d s%d)" % (arm, lr, batch,
                                                            seed)
            opt.step()
            losses.append(bloss.item())
        acc = evaluate(model, memo64)
        curves["loss"].append(sum(losses) / len(losses))
        curves["clean_train_acc"].append(acc)
        best_acc = max(best_acc, acc)
        if acc >= MEMO_GATE:
            reached, reach_epoch = True, epoch
            break
    return {"arm": arm, "lr": lr, "batch": batch, "seed": seed,
            "n_items": len(memo64), "reached_0.95": reached,
            "reach_epoch": reach_epoch, "epochs_run": len(curves["loss"]),
            "best_clean_train_acc": best_acc, "curves": curves}


# --------------------------------------- SS5.3 grid rerun (Phase-1 exact)
def grid_run(arm, lr, batch, seed, data, verbs, vocab, max_epochs):
    """Verbatim re-implementation of exp43b_phase1.train_run semantics
    (same rng stream, same gate logic, same records), model swapped."""
    torch.manual_seed(seed)
    model = make_model(arm, verbs, vocab, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(43000 + seed)
    train, valset = data["train"], data["val"]

    curves = {"loss": [], "train_run_acc": [], "val_acc": [],
              "grad_norm": []}
    gate_met, gate_epoch = False, None
    gate_train_acc, gate_val_acc = None, None
    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    for epoch in range(1, max_epochs + 1):
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
                "non-finite grad (%s lr%g b%d s%d)" % (arm, lr, batch, seed)
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
        print("[%s lr%g b%d s%d] ep %2d/%d loss=%.4f train(run)=%.3f "
              "val=%.3f gn=%.2e %.1fs"
              % (arm, lr, batch, seed, epoch, max_epochs,
                 curves["loss"][-1], curves["train_run_acc"][-1],
                 val_acc, curves["grad_norm"][-1], time.time() - t0),
              flush=True)
        if val_acc >= GATE_VAL:
            clean_train = evaluate(model, train)
            if clean_train >= GATE_TRAIN:
                gate_met, gate_epoch = True, epoch
                gate_train_acc, gate_val_acc = clean_train, val_acc
                print("[%s lr%g b%d s%d] GATE MET at epoch %d "
                      "(train=%.3f val=%.3f)"
                      % (arm, lr, batch, seed, epoch, clean_train, val_acc),
                      flush=True)
                break

    model.load_state_dict(best["state"])
    rec_train = evaluate(model, train)
    rec_test = evaluate(model, data["test"])
    return {"rung": RUNG, "arm": arm, "lr": lr, "batch": batch, "seed": seed,
            "gate_met": gate_met, "gate_epoch": gate_epoch,
            "gate_train_acc": gate_train_acc, "gate_val_acc": gate_val_acc,
            "epochs_run": len(curves["loss"]),
            "best_val_acc": best["val_acc"], "best_val_epoch": best["epoch"],
            "bestval_clean_train_acc": rec_train,
            "bestval_test_acc": rec_test,
            "curves": curves}


# --------------------------------------------- SS5.3 init audit (rerun)
def q(xs):
    xs = sorted(xs)
    n = len(xs)
    return {"min": xs[0], "p25": xs[n // 4], "median": xs[n // 2],
            "p75": xs[(3 * n) // 4], "max": xs[-1], "n": n}


def rel_commutator(M, S):
    M = M.to(torch.complex128) if M.is_complex() else M.to(torch.float64)
    S = S.to(M.dtype)
    num = torch.linalg.matrix_norm(M @ S - S @ M).item()
    den = torch.linalg.matrix_norm(M).item()
    return num / max(den, 1e-12)


def matched_init_audit(items, angles, verbs):
    """exp43b_b1_init_audit measurements, on MatchedInitB1's init."""
    train = [i for i in items if i["split"] == "train"]
    by_pair = {}
    for it in train:
        by_pair.setdefault(it["pair_id"], []).append(it)
    pairs = [p for p in sorted(by_pair)
             if len(by_pair[p]) == 2][:AUDIT_N_PAIRS]
    compiled = {}
    for p in pairs:
        a, b = sorted(by_pair[p], key=lambda x: x["id"])
        compiled[p] = (comp.compile_story(a, angles),
                       comp.compile_story(b, angles))
    flat = [cs for p in pairs for cs in compiled[p]]

    S = _pair_swap(2).to(torch.float64)
    per_seed = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = MatchedInitB1(verbs, seed, d_ref=2)
        comm = {v: rel_commutator(model.verb_matrix(v).detach(), S)
                for v in verbs}
        deltas, collapsed, correct = [], 0, 0
        with torch.no_grad():
            for p in pairs:
                ca, cb = compiled[p]
                va = model.story_values(ca).detach()
                vb = model.story_values(cb).detach()
                va = va / va.sum()
                vb = vb / vb.sum()
                deltas.append(float((va - vb).abs().sum()))
                pa = int(torch.argmax(va).item())
                pb = int(torch.argmax(vb).item())
                if pa == pb:
                    collapsed += 1
                correct += int(pa == ca.answer) + int(pb == cb.answer)
        model.zero_grad()
        bloss = 0.0
        for cs in flat[:AUDIT_N_GRAD]:
            loss, _ = model.story_loss(cs)
            bloss = bloss + loss
        (bloss / AUDIT_N_GRAD).backward()
        gsq, gsq_dress = 0.0, 0.0
        for name, prm in model.named_parameters():
            if prm.grad is None:
                continue
            g2 = float((prm.grad ** 2).sum())
            gsq += g2
            if name.startswith("dress"):
                gsq_dress += g2
        per_seed[str(seed)] = {
            "rel_commutator_with_SWAP_per_verb":
                {v: round(c, 6) for v, c in comm.items()},
            "rel_commutator_max": round(max(comm.values()), 6),
            "rel_commutator_mean": round(sum(comm.values()) / len(comm), 6),
            "twin_value_L1_gap": q([round(d, 8) for d in deltas]),
            "frac_twin_pairs_predicting_identically":
                collapsed / len(pairs),
            "init_accuracy_on_audited_items": correct / (2 * len(pairs)),
            "grad_global_norm_at_init": gsq ** 0.5,
            "grad_norm_dress_only_at_init": gsq_dress ** 0.5,
            "grad_frac_in_dress": (gsq_dress / gsq) if gsq > 0 else None,
        }
    return per_seed, len(pairs)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="mechanical check only: 3 memo epochs, 2 grid "
                         "epochs, NO results file")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)
    memo_epochs = 3 if args.smoke else MEMO_MAX_EPOCHS
    grid_epochs = 2 if args.smoke else GRID_MAX_EPOCHS

    items, meta, sha = load_l1()
    angles, emb_meta, emb_sha = comp.load_embeddings()
    verbs = sorted(meta["verb_inventory"].keys())

    # tokens are needed by B2/B3; vocab over ALL L1 splits (deterministic,
    # mirrors exp42_run's train+val+test construction).
    compiled_all = {sp: comp.compile_all(
        [i for i in items if i["split"] == sp], angles, with_tokens=True)
        for sp in ("train", "val", "test")}
    vocab = build_vocab(cs.tokens for sp in ("train", "val", "test")
                        for cs in compiled_all[sp])

    memo_items, memo_pair_ids = memo_subset(items)
    memo64 = comp.compile_all(memo_items, angles, with_tokens=True)
    n_yes = sum(1 for cs in memo64 if cs.answer == 0)
    print("[exp43c] memo subset: %d items (%d yes / %d no), pairs %s..%s"
          % (len(memo64), n_yes, len(memo64) - n_yes,
             memo_pair_ids[0], memo_pair_ids[-1]), flush=True)

    # ---------------- SS5.2 memorisation control
    memo_runs = {arm: [] for arm in MEMO_ARMS}
    for arm in MEMO_ARMS:
        for lr in GRID_LRS:
            for batch in GRID_BATCHES:
                for seed in SEEDS:
                    key = "memo_%s_lr%g_b%d_s%d" % (arm, lr, batch, seed)
                    ck = os.path.join(CKPT_DIR, key + ".json")
                    if not args.smoke and os.path.exists(ck):
                        with open(ck, encoding="utf-8") as f:
                            r = json.load(f)
                        print("[ckpt] %s loaded" % key, flush=True)
                    else:
                        ts = time.time()
                        r = memo_run(arm, lr, batch, seed, memo64, verbs,
                                     vocab, memo_epochs)
                        print("[memo %s lr%g b%d s%d] best=%.3f "
                              "reached=%s ep=%s (%.1fs)"
                              % (arm, lr, batch, seed,
                                 r["best_clean_train_acc"],
                                 r["reached_0.95"], r["reach_epoch"],
                                 time.time() - ts), flush=True)
                        if not args.smoke:
                            with open(ck, "w", encoding="utf-8") as f:
                                json.dump(r, f, ensure_ascii=False)
                    memo_runs[arm].append(r)

    arm_trainable = {arm: bool(any(r["reached_0.95"] for r in runs))
                     for arm, runs in memo_runs.items()}
    a1_fails = not arm_trainable["A1"]
    inconclusive = not a1_fails

    # ---------------- SS5.3 matched-initialisation B1
    audit, n_audit_pairs = matched_init_audit(items, angles, verbs)
    collapse = {s: audit[s]["frac_twin_pairs_predicting_identically"]
                for s in audit}
    f3_pass = bool(max(collapse.values()) <= 0.60)
    print("[exp43c] matched-init audit: collapse=%s F3(<=0.60)=%s"
          % (collapse, f3_pass), flush=True)

    data = {sp: compiled_all[sp] for sp in ("train", "val", "test")}
    b1m_runs = []
    for lr in GRID_LRS:
        for batch in GRID_BATCHES:
            for seed in SEEDS:
                key = "%s_B1M_lr%g_b%d_s%d" % (RUNG, lr, batch, seed)
                ck = os.path.join(CKPT_DIR, key + ".json")
                if not args.smoke and os.path.exists(ck):
                    with open(ck, encoding="utf-8") as f:
                        r = json.load(f)
                    print("[ckpt] %s loaded" % key, flush=True)
                else:
                    r = grid_run("B1M", lr, batch, seed, data, verbs,
                                 vocab, grid_epochs)
                    if not args.smoke:
                        with open(ck, "w", encoding="utf-8") as f:
                            json.dump(r, f, ensure_ascii=False)
                        print("[ckpt] %s saved (gate_met=%s)"
                              % (key, r["gate_met"]), flush=True)
                b1m_runs.append(r)

    b1m_learnable = bool(any(r["gate_met"] for r in b1m_runs))

    # ---------------- consequences (doc 24 SS5.4, frozen)
    row_key = ("true" if b1m_learnable else "false",
               "any" if b1m_learnable else
               ("true" if arm_trainable["B1"] else "false"))
    consequence = CONSEQUENCE_TABLE_5_4[row_key]

    verdicts = {}
    for arm in MEMO_ARMS:
        verdicts["arm_trainable_%s" % arm] = arm_trainable[arm]
    verdicts["a1_fails_memorisation"] = bool(a1_fails)
    verdicts["exp43c_inconclusive"] = bool(inconclusive)
    verdicts["b1_matched_init_twin_collapse_at_init"] = collapse
    verdicts["b1_matched_init_f3_pass"] = f3_pass
    verdicts["b1_matched_init_learnable_L1"] = b1m_learnable

    if args.smoke:
        print("[smoke] mechanical run complete -- no results file written.")
        print("[smoke] arm best accs:",
              {a: max(r["best_clean_train_acc"] for r in rs)
               for a, rs in memo_runs.items()})
        return

    results = {
        "experiment": "exp43c arm validity (memorisation control + "
                      "matched-initialisation B1)",
        "spec": "doc 24 SS5 (qnlp_private_docs/24, commit c6099b0, FROZEN): "
                "SS5.2 memorisation control, fixed 64-item L1 subset "
                "(first 32 twin pairs by sorted pair_id), grid "
                "{lr 0.005, 0.02} x {batch 1, 8}, 2 seeds, <=300 epochs, "
                "arm_trainable iff any config reaches train>=0.95; SS5.3 "
                "matched-init B1 on the exact frozen L1 Phase-1 grid, "
                "<=60 epochs, gate train>=0.8 AND val>=0.7; consequences "
                "frozen in SS5.4",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "dataset_sha256": sha,
        "embeddings_sha256": emb_sha,
        "meta": {
            "additive_amendment_A2": AMENDMENT_NOTE_A2,
            "b2_name_augmentation": B2_AUGMENTATION_NOTE,
            "matched_init_scheme": GENERIC_READOUT_NOTE,
            "vocab_note": "B2/B3 vocab built over all L1 splits "
                          "(train+val+test), size %d" % len(vocab),
            "memo_subset": {"n_items": len(memo64),
                            "n_pairs": len(memo_pair_ids),
                            "pair_ids_first_last": [memo_pair_ids[0],
                                                    memo_pair_ids[-1]],
                            "n_yes": n_yes},
            "control_of_the_control": "A1 is the leak sentinel; Lemma 1 "
                "pins it to <=0.5 on twins, so it MUST fail SS5.2. If A1 "
                "passes, exp43c is INCONCLUSIVE and is acted on in no "
                "way (doc 24 SS5.2).",
        },
        "frozen_grid": {"lrs": GRID_LRS, "batches": GRID_BATCHES,
                        "seeds": SEEDS,
                        "memo_max_epochs": MEMO_MAX_EPOCHS,
                        "memo_gate_train": MEMO_GATE,
                        "grid_max_epochs": GRID_MAX_EPOCHS,
                        "gate_train": GATE_TRAIN, "gate_val": GATE_VAL},
        "memorisation_runs": {
            arm: [{k: v for k, v in r.items() if k != "curves"}
                  for r in runs] for arm, runs in memo_runs.items()},
        "matched_init_audit": {
            "n_twin_pairs_audited": n_audit_pairs,
            "per_seed": audit,
            "f3_bar": 0.60,
            "b1_as_shipped_reference":
                "results_exp43b_b1_init_audit.json (0.860 / 0.865)"},
        "b1_matched_grid_runs": [
            {k: v for k, v in r.items() if k != "curves"}
            for r in b1m_runs],
        "verdicts": verdicts,
        "consequence_5_4": {
            "row": {"b1_matched_init_learnable_L1": b1m_learnable,
                    "arm_trainable_B1": arm_trainable["B1"]},
            "text_verbatim": consequence,
            "applies": not inconclusive,
            "note": (None if not inconclusive else
                     "exp43c INCONCLUSIVE (A1 passed the memorisation "
                     "control): the consequence row is recorded but NOT "
                     "acted upon, per doc 24 SS5.2."),
        },
        "git_note": git_note(),
        "wall_time_sec": round(time.time() - t0, 1),
    }
    path = fresh_results_path("results_exp43c")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("[exp43c] wrote %s" % path)
    print("[exp43c] verdicts: %s" % json.dumps(
        {k: v for k, v in verdicts.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
