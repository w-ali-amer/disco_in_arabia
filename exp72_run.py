# -*- coding: utf-8 -*-
"""exp72_run.py -- doc 58: THE RESCUE RUNG. Do the field's proposed trainability
mitigations rescue this programme's committed training-failure instances?

REGISTRATION: qnlp_private_docs/58_exp72_rescue_rung.md, REGISTERED 2026-08-27,
orchestrator rulings 1-6 in the status header, SS18's post-registration note
(the exp70 confirm REPRODUCED I2's gap; I2 stands as registered on the first
draw). **The doc is the specification; this file implements it and adds nothing.**

WHAT exp72 IS, AND WHAT IT IS NOT (doc 58 SS0)
==============================================
exp72 is NOT a lever experiment. It adds ZERO coefficients, ZERO features and
ZERO new surface. It varies ONLY THE OPTIMISATION PROCEDURE, on two frozen
instances, against frozen targets, on exp69's frozen Part A dataset.

  I1  = exp69 R2-W    (F2 product register, old map, 4 wires)  0.7293, comparator 1.0000
  I2  = exp70 N-in-T2 (E-in entangled register, eps-map)       0.6467, comparator 1.0000

Both instances' class-matched classical comparators score EXACTLY 1.0000 on the
same held-out items, so "the information is present and linearly extractable" is
a MEASUREMENT, not an assumption (doc 58 SS2). That controlled instance is the
contribution; the four mitigations are borrowed verbatim (doc 58 SS4.6).

**NO EXP72 NUMBER MAY APPEAR IN A SENTENCE CONTAINING "ADVANTAGE"** (doc 58
SS12): the comparators are already at 1.0000 and there is nothing to win.

THE FOUR RESCUE ARMS (doc 58 SS6), and every one of them changes ONLY how the
same parameters are found:

  R-budget     450 epochs (3x), cosine decay 0.02 -> ~0, init unchanged.
               ARM NUMBER ONE, and its rescue is the DEFLATIONARY branch
               ("undertrained, not untrainable" -- doc 58 SS0.4 / SS6.1).
  R-identity   verb blocks initialised at theta ~ N(0, 0.1^2) -- identity plus
               small noise (arXiv:1903.05076). READOUT INIT UNCHANGED, and that
               is a NULL-INSTRUMENT argument, not a stylistic one: two readout
               blocks at identity would be IDENTICAL, v_yes == v_no, and the
               model would output 0.5 everywhere (doc 58 SS6.2.2).
  R-layerwise  75/75/75 staged freezing (arXiv:2006.14904, ADAPTED per SS4.2 --
               the paper GROWS depth, exp72 freezes blocks at FIXED depth, and a
               null here is NOT a null on their published procedure).
               {event-1 blocks + readout} / {event-2 blocks only, readout FROZEN}
               / {everything}. Parameter-update-weighted budget EXACTLY 150.000;
               raw epochs 225 = 1.50x. BOTH numbers travel with any verdict.
               FRESH Adam(0.02) per stage over that stage's subset only (SS6.3.3).
  R-warmstart  all 264 parameters loaded from the registered same-family T = 1
               donor, then the committed 150-epoch recipe. Seed still varies BATCH
               ORDER (SS6.0), so five runs are five runs.

THE ANCHORS RUN FIRST AND THEY BLOCK EVERYTHING (doc 58 SS5.1). I1 and I2 are
re-run UNRESCUED through THIS FILE'S `train()` -- the same entry point the rescue
arms use, with the rescue options off (SS7.3) -- and their per-seed held-out
accuracies must reproduce the committed values to |delta| <= 0.002. The honest
expectation is BIT-EXACT, 10/10 seeds. If either fails, the exp72 pipeline has
drifted, SS5.2's floor-omission argument is void, and NOTHING ELSE IS READ.

THE FLOORS ARE STAGED, AND NO VERDICT IS READ BEFORE ITS FLOOR (doc 58 SS14).
Stage 1 compares against the COMMITTED floors as ORIENTATION ONLY. Any cell
reaching its `rescue_partial` bar triggers a floor RE-RUN UNDER ITS OWN RESCUE
PROCEDURE -- including, for the warm-start cells, SEVERED DONORS (SS14.4): the
T = 1 donor is retrained on the shuffled bearer designation, selected on the
SHUFFLED val, and the shuffle stream is SEED-LINKED across the two item sets.

RULE 3 / RULE 4 COMPLIANCE. exp72 writes ONLY `exp72_ckpt/` and `results_exp72*`
files, through `fresh_results_path` so nothing is ever overwritten. NO exp68,
exp69 or exp70 artefact is modified; the two donor checkpoints are READ-ONLY and
their shas are asserted before use. There is NO datagen, NO V-suite re-run, NO
comparator and NO new instrument (doc 58 SS3, SS15.4, SS16.1).

    python exp72_run.py --smoke                    # doc 58 SS7, ALL gates, FIRST
    python exp72_run.py --anchor-gate              # doc 58 SS5, BLOCKING
    python exp72_run.py --stage1 --worker i --nworkers 5
    python exp72_run.py --stage2 --cells I1:R-budget,... --worker i --nworkers 5
    python exp72_aggregate.py                      # the ONLY results writer
"""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)                              # doc 45 SS6.1, verified inert

import exp42_sim_adapter as sim                       # noqa: E402
import exp68_run as E                                 # noqa: E402  F2Model, config
import exp69_run as R69                               # noqa: E402  compile, floors
import exp70_run as R70                               # noqa: E402  circuits, models
import instrument_entangled_bounds as IEB             # noqa: E402
from exp42_run import fresh_results_path, grad_global_norm   # noqa: E402
from exp60_run import evaluate                        # noqa: E402

SPEC = ("qnlp_private_docs/58_exp72_rescue_rung.md (REGISTERED 2026-08-27, "
        "orchestrator rulings 1-6 in the status header, SS18 post-registration "
        "note on I2)")
SPEC_DOC = "58_exp72_rescue_rung.md"
PRIVATE_DOCS = os.path.expanduser("~/qnlp_private_docs")

CKPT_DIR = "exp72_ckpt"
SMOKE_BASE = "results_exp72_smoke_validation"
ANCHOR_BASE = "results_exp72_anchors"

EXP69_RESULTS = "results_exp69.json"
EXP70_RESULTS = "results_exp70.json"

# Training config: the exp60 / exp61 / exp65 / exp68 / exp69 / exp70 loop,
# UNCHANGED. Anything else here would put the anchors' bit-exactness out of reach.
LR, BATCH, MAX_EPOCHS = E.LR, E.BATCH, E.MAX_EPOCHS       # 0.02, 8, 150
TRAIN_GATE, VAL_GATE = E.TRAIN_GATE, E.VAL_GATE
RNG_BASE = E.RNG_BASE
N_SEEDS = E.N_SEEDS

ANCHOR_TOLERANCE = 0.002              # doc 58 SS5.1; honest expectation is 0.0

# doc 58 SS6.1: R-budget. THE SCHEDULE IS PINNED HERE, ONCE, because the doc
# registers "cosine decay 0.02 -> 0.0 over the full 450 epochs, no restarts" and
# does not write the formula. Registered form (torch's CosineAnnealingLR
# semantics with T_max = 450, eta_min = 0, stepped once per epoch):
#
#     lr(e) = 0.02 * 0.5 * (1 + cos(pi * (e - 1) / 450)),  e = 1 .. 450
#
# lr(1) = 0.02 EXACTLY; lr(450) = 2.4e-07, i.e. zero to seven decimals. No
# restarts. The formula, the endpoint and the 5 sampled epochs are recorded in
# every R-budget run record and asserted at smoke in BOTH directions.
BUDGET_EPOCHS = 450
BUDGET_MULTIPLE = 3

# doc 58 SS6.2: R-identity. sigma on the VERB blocks only.
IDENTITY_SIGMA = 0.1
IDENTITY_RNG_BASE = 7_200_000         # a DEDICATED stream; the committed stream
                                      # is left untouched so the readout init is
                                      # bit-identical to the committed one

# doc 58 SS6.3: R-layerwise. (epochs, trainable groups) per stage, and the
# parameter-update weights the SS6.3.2 arithmetic is computed from.
LAYERWISE_STAGES = [(75, ("event1", "readout")),
                    (75, ("event2",)),
                    (75, ("event1", "event2", "readout"))]

# doc 58 SS1.1 / SS6.4.1 / SS15.1. Frozen, and every number below is ASSERTED
# against the committed results files rather than trusted from this table.
INSTANCES = {
    "I1": {"arm": "A0-W-T2", "committed_arm": "R2-W", "source": EXP69_RESULTS,
           "comparator_tier": None, "donor_arm": "A0-W-T1",
           "donor_committed_arm": "R1-W", "donor_source": EXP69_RESULTS,
           "donor_ckpt": "exp69_ckpt/R1-W_s0.pt",
           "donor_sha256": ("0e2af9aee9ec2601f813326ebe1af3deb72a0bcacd256957"
                            "cfdf0c0fe79d50d6"),
           "donor_seed": 0,
           "donor_tie_break": "unique max on in-distribution val (0.95455)",
           "committed_mean": 0.7293, "committed_floor_mean": 0.5020,
           "committed_floor_sd": 0.0283,
           "encoding": "F2 product register, old map (p+1)pi, 4 wires",
           "n_params": 264},
    "I2": {"arm": "N-in-T2", "committed_arm": "N-in-T2", "source": EXP70_RESULTS,
           "comparator_tier": "class", "donor_arm": "N-in-T1",
           "donor_committed_arm": "N-in-T1", "donor_source": EXP70_RESULTS,
           "donor_ckpt": "exp70_ckpt/N-in-T1_s0.pt",
           "donor_sha256": ("08c943289ff56a619ae9ce4be6195a48b1573d318d82715d"
                            "116437177a9c258f"),
           "donor_seed": 0,
           "donor_tie_break": ("four-way tie at 0.92424 on val across seeds "
                               "0, 1, 3, 4 -> LOWEST SEED INDEX (doc 58 SS6.4.1, "
                               "orchestrator ruling 3, RATIFIED)"),
           "committed_mean": 0.6467, "committed_floor_mean": 0.5000,
           "committed_floor_sd": 0.0564,
           "encoding": "E-in entangled register, eps-map, 4 wires",
           "n_params": 264},
}
RESCUES = ["R-budget", "R-identity", "R-layerwise", "R-warmstart"]

# doc 58 SS9.4 / SS15.1: the positive control. R-budget on the HEALTHY R2-A.
PC_INSTANCE = "PC"
PC_ARM = "EA-ctrl-T2"                 # == exp69 R2-A, bit-exactly (exp70 anchor)
PC_COMMITTED_ARM = "R2-A"
PC_COMMITTED_MEAN = 0.9187
PC_DEGRADE_BAR = 0.8587               # the doc's LITERAL registered number
PC_RESCUE = "R-budget"

DATASET_SHA_DOC58 = ("2b85822b544571969fd40cab12b03a4ac256b08aee914d2b9643b856"
                     "a6308c96")


# ============================================================ small plumbing
def spec_commit():
    """doc 58's commit, read from git rather than typed."""
    try:
        out = subprocess.run(
            ["git", "-C", PRIVATE_DOCS, "log", "--oneline", "-1", "--", SPEC_DOC],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return out
    except Exception as exc:                                      # noqa: BLE001
        return "unavailable: %s" % exc


def git_note():
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             capture_output=True, text=True, timeout=30).stdout
        ours = [l for l in out.splitlines() if "exp72" in l]
        head = subprocess.run(["git", "log", "--oneline", "-1"],
                              capture_output=True, text=True,
                              timeout=30).stdout.strip()
        return {"repo_head": head, "exp72_paths": ours,
                "n_untracked": sum(1 for l in out.splitlines()
                                   if l.startswith("??")),
                "rule3_note": ("exp72 writes ONLY exp72_ckpt/ and results_exp72* "
                               "(through fresh_results_path). No exp68 / exp69 / "
                               "exp70 artefact is modified; the donor checkpoints "
                               "are READ-ONLY and sha-asserted before use.")}
    except Exception as exc:                                      # noqa: BLE001
        return {"error": str(exc)}


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def cosine_lr(base, ep, total):
    """doc 58 SS6.1's schedule, pinned above. `ep` is 1-indexed."""
    return base * 0.5 * (1.0 + math.cos(math.pi * (ep - 1) / total))


def assert_lr_trace(trace, total, base, tol=1e-9):
    """doc 58 SS7.2, `R-budget`: the lr trace matches the registered cosine
    schedule to 1e-9 at 5 sampled epochs, AND IS NOT CONSTANT.

    THE FAILING DIRECTION IS THE POINT: a constant-lr trace must fail the arm.
    Both halves are asserted, and `smoke()` feeds this function a constant trace
    and requires it to raise."""
    assert len(trace) == total, \
        ("R-budget did not take exactly %d epochs (%d)" % (total, len(trace)))
    picks = [1, total // 4, total // 2, (3 * total) // 4, total]
    worst = 0.0
    for e in picks:
        worst = max(worst, abs(trace[e - 1] - cosine_lr(base, e, total)))
    assert worst <= tol, \
        ("the realised lr trace differs from doc 58 SS6.1's registered cosine "
         "schedule by %.3e at the sampled epochs %s" % (worst, picks))
    span = max(trace) - min(trace)
    assert span > 1e-6, \
        ("the lr trace is CONSTANT (span %.3e) -- doc 58 SS6.1 registers a cosine "
         "DECAY and a constant-lr trace FAILS THE ARM (SS7.2's failing direction)"
         % span)
    return {"n_epochs": total, "sampled_epochs": picks,
            "sampled_lr": [trace[e - 1] for e in picks],
            "sampled_expected": [cosine_lr(base, e, total) for e in picks],
            "max_abs_dev": worst, "tolerance": tol,
            "lr_first": trace[0], "lr_last": trace[-1], "span": span,
            "formula": ("lr(e) = 0.02 * 0.5 * (1 + cos(pi * (e - 1) / 450)), "
                        "e = 1..450 -- torch CosineAnnealingLR(T_max=450, "
                        "eta_min=0) semantics stepped once per epoch, no restarts. "
                        "PINNED IN exp72_run.py because doc 58 SS6.1 registers the "
                        "schedule and not the formula."),
            "constant_lr_would_fail": True}


# ================================================================== the model
def verb_groups(store):
    """doc 58 SS6.3.1: `V_A` and `V_B` are DISJOINT by construction and at T = 2
    event 1 draws only from `V_A`, event 2 only from `V_B`. Read from the frozen
    dataset's own `meta.verb_slots`, never re-typed."""
    _items, meta, _path = store
    slots = meta["verb_slots"]
    va, vb = list(slots["V_A"]), list(slots["V_B"])
    assert not (set(va) & set(vb)), "V_A and V_B are not disjoint"
    return va, vb


def param_groups(m, va, vb):
    """The three registered parameter groups of doc 58 SS6.3.1 (3/8, 3/8, 2/8)."""
    assert hasattr(m, "blocks"), \
        ("R-layerwise is registered for the register arms only; this model has no "
         "per-verb block dict")
    return {"event1": [m.blocks[v].theta for v in va],
            "event2": [m.blocks[v].theta for v in vb],
            "readout": [m.q_pos.theta, m.q_neg.theta]}


def apply_identity_init(m, seed, va, vb):
    """doc 58 SS6.2 -- IDENTITY-BLOCK INITIALISATION (arXiv:1903.05076).

    The verb blocks are re-drawn as theta ~ N(0, sigma^2), sigma = 0.1: identity
    plus small noise. `Circuit4(theta = 0)` is EXACTLY the identity here because
    every gate in the ansatz is a parameterised rotation with rx(0) = rz(0) =
    crx(0) = I and there are no fixed entanglers (SS6.2.1) -- Grant et al.'s
    paired-inverse-block construction is not required.

    THE READOUT IS DELIBERATELY UNTOUCHED (SS6.2.2). It is drawn from the
    COMMITTED stream by `build_model` and is left exactly as the committed recipe
    made it; the noise comes from a DEDICATED generator so not one committed draw
    moves. Two readout blocks at identity would be IDENTICAL, so v_yes == v_no
    for every item and the model would output 0.5 everywhere -- a null instrument
    in doc 53 SS5's exact sense."""
    g = torch.Generator().manual_seed(IDENTITY_RNG_BASE + seed)
    touched = []
    with torch.no_grad():
        for v in sorted(list(va) + list(vb)):             # deterministic order
            t = m.blocks[v].theta
            t.copy_(torch.randn(t.shape, generator=g) * IDENTITY_SIGMA)
            touched.append(v)
    return touched


def identity_assertions(m, va, vb, seed):
    """doc 58 SS7.2's `R-identity` row, BOTH DIRECTIONS.

      (a) with all verb-block parameters set to 0, every verb block's matrix
          equals I to 1e-9;
      (b) the realised init's per-parameter sample sd lies in [0.08, 0.12];
      (c) THE FAILING DIRECTION: `q_pos` and `q_neg` must NOT be near-identity,
          and ||q_pos - q_neg|| > 0.1 at init -- SS6.2.2's null-instrument check.
    """
    worst_id = 0.0
    with torch.no_grad():
        for v in sorted(list(va) + list(vb)):
            blk = m.blocks[v]
            keep = blk.theta.detach().clone()
            blk.theta.zero_()
            M = blk.matrix()
            eye = torch.eye(M.shape[0], dtype=M.dtype)
            worst_id = max(worst_id, float((M - eye).abs().max()))
            blk.theta.copy_(keep)
    assert worst_id <= 1e-9, \
        ("doc 58 SS6.2.1 FAILED: a verb block at theta = 0 is not the identity "
         "(max |block - I| = %.3e). The arm would be named for a construction it "
         "does not implement." % worst_id)

    vals = torch.cat([m.blocks[v].theta.detach().reshape(-1)
                      for v in sorted(list(va) + list(vb))])
    sd = float(vals.std(unbiased=True))
    assert 0.08 <= sd <= 0.12, \
        ("the realised R-identity init has per-parameter sd %.4f, outside the "
         "registered [0.08, 0.12] for sigma = 0.1" % sd)

    with torch.no_grad():
        for name, blk in (("q_pos", m.q_pos), ("q_neg", m.q_neg)):
            M = blk.matrix()
            eye = torch.eye(M.shape[0], dtype=M.dtype)
            d = float((M - eye).abs().max())
            assert d > 0.1, \
                ("doc 58 SS6.2.2's NULL-INSTRUMENT CHECK FAILED: %s is "
                 "near-identity at init (max |%s - I| = %.3e). Two readout blocks "
                 "at identity are IDENTICAL, v_yes == v_no everywhere, and the "
                 "readout is a null instrument." % (name, name, d))
        sep = float((m.q_pos.matrix() - m.q_neg.matrix()).abs().pow(2).sum().sqrt())
        theta_sep = float((m.q_pos.theta.detach()
                           - m.q_neg.theta.detach()).norm())
    assert sep > 0.1 and theta_sep > 0.1, \
        ("doc 58 SS6.2.2: ||q_pos - q_neg|| = %.4f (theta %.4f) at init -- the two "
         "readout blocks are degenerate and the model outputs 0.5 everywhere"
         % (sep, theta_sep))
    return {"verb_block_identity_at_zero_max_abs": worst_id,
            "identity_tolerance": 1e-9,
            "realised_init_sd": sd, "sigma": IDENTITY_SIGMA,
            "sd_window": [0.08, 0.12],
            "readout_untouched": True,
            "readout_q_pos_distance_from_identity": float(
                (m.q_pos.matrix() - torch.eye(m.q_pos.matrix().shape[0],
                                              dtype=m.q_pos.matrix().dtype)
                 ).abs().max()),
            "readout_q_neg_distance_from_identity": float(
                (m.q_neg.matrix() - torch.eye(m.q_neg.matrix().shape[0],
                                              dtype=m.q_neg.matrix().dtype)
                 ).abs().max()),
            "readout_separation_matrix_fro": sep,
            "readout_separation_theta_l2": theta_sep,
            "identity_rng_stream": "torch.Generator().manual_seed(%d + seed)"
                                   % IDENTITY_RNG_BASE,
            "note": ("doc 58 SS6.2.2: the readout init is UNCHANGED and the noise "
                     "is drawn from a DEDICATED stream, so every committed draw "
                     "the readout consumes is bit-identical to the committed "
                     "recipe's.")}


def load_donor(path, sha, n_params):
    """doc 58 SS6.4.1: the donor is READ-ONLY and its sha is asserted BEFORE use."""
    assert os.path.exists(path), "donor checkpoint missing: %s" % path
    got = sha256_file(path)
    assert got == sha, \
        ("donor sha mismatch for %s: %s != registered %s" % (path, got, sha))
    state = torch.load(path, map_location="cpu")
    n = sum(int(v.numel()) for v in state.values())
    assert n == n_params, \
        ("donor carries %d parameters, the recipient %d -- doc 58 SS6.4 asserts "
         "parameter-count identity at T = 1 and T = 2" % (n, n_params))
    return state, got


def warmstart_assertions(m, state):
    """doc 58 SS7.2's `R-warmstart` row: max |theta_init - theta_donor| = 0.0 over
    all parameters. (The FAILING DIRECTION -- the five seeds' final parameter
    vectors must NOT be bit-identical -- is a cross-run property and is asserted
    once the cell's five runs exist; see `check_warmstart_spread`.)"""
    cur = dict(m.state_dict())
    assert set(cur) == set(state), "donor / recipient state-dict keys differ"
    worst = 0.0
    for k in sorted(cur):
        worst = max(worst, float((cur[k] - state[k]).abs().max()))
    assert worst == 0.0, \
        ("the warm start did not land: max |theta_init - theta_donor| = %.3e "
         "(doc 58 SS7.2)" % worst)
    return {"max_abs_init_minus_donor": worst,
            "n_params": sum(int(v.numel()) for v in cur.values()),
            "keys": sorted(cur)}


def check_warmstart_spread(finals):
    """doc 58 SS7.2 / SS6.0 / SS18's null-instrument-by-symmetry row. The
    initialisation assertion above would PASS on a build that loaded the donor and
    then ignored it, so the failing direction is asserted too: `seed` reshuffles
    the training order every epoch (SS6.0), so five R-warmstart runs of a cell must
    NOT be bit-identical to each other."""
    keys = sorted(finals)
    pairs, worst = [], 0.0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = finals[keys[i]], finals[keys[j]]
            d = max(float((a[k] - b[k]).abs().max()) for k in sorted(a))
            worst = max(worst, d)
            pairs.append({"seeds": [keys[i], keys[j]], "max_abs_diff": d})
    ok = all(p["max_abs_diff"] > 0.0 for p in pairs)
    return {"pairwise": pairs, "max_abs_diff_over_pairs": worst,
            "all_pairs_differ": bool(ok),
            "verdict": ("PASS -- the five seeds' FINAL parameter vectors differ, so "
                        "the loaded donor is being trained rather than ignored "
                        "(doc 58 SS6.0 / SS7.2's failing direction)" if ok else
                        "FAIL -- five R-warmstart runs are BIT-IDENTICAL. The batch "
                        "order is not varying with seed and the arm is a null "
                        "instrument by symmetry (doc 58 SS18, doc 56 SS14's row)."),
            "assertion_passed": bool(ok)}


# ================================================== data loading / compilation
def load_store():
    return R70.load_part(R70.DATA_A)


def compile_arm(arm, items, seed, fm, shuffled):
    """The committed compile path. `shuffled` is EXPLICIT here rather than keyed
    off the arm name, because exp72's floors run the SAME arm under a rescue
    procedure and the arm name must not change (doc 58 SS14.3: same items, same
    splits, same geometry -- only the feature <-> label correlation is severed)."""
    ang, _t1 = R70.arm_angles(arm, fm)
    use = R69.relabel_shuffle_q(items, seed) if shuffled else items
    return {s: R69.compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}


# ==================================================================== training
def train(arm, seed, store, verbs, fm, rescue=None, shuffled=False,
          donor_state=None, max_epochs=None, save=True, tag="", label=None,
          probe_epochs=None):
    """THE SINGLE TRAINING ENTRY POINT (doc 58 SS7.3).

    With `rescue=None` this is the committed exp69 / exp70 loop VERBATIM --
    Adam(lr 0.02), batch 8, 150 epochs, checkpoint selected on in-distribution val
    accuracy, held-out test read ONCE from the selected checkpoint, and the call
    order (compile -> manual_seed -> build -> Adam -> Random(RNG_BASE + seed))
    exp69's verbatim. **The anchors run through THIS function with the options
    off, not through a separate legacy path** -- otherwise doc 58 SS5's
    bit-exactness would prove nothing about the rescue arms' pipeline.

    NOT ONE EXTRA DRAW is taken from the committed torch or `random` streams when
    `rescue` is None; R-identity's noise comes from a dedicated generator and
    R-warmstart overwrites the parameters after construction, so the committed
    consumption order is preserved in every arm."""
    va, vb = verb_groups(store)
    items, meta, path = R70.arm_items(arm, store)
    data = compile_arm(arm, items, seed, fm, shuffled)

    torch.manual_seed(seed)
    m = R70.build_model(arm, verbs, seed, fm)
    n_params = sum(p.numel() for p in m.parameters())

    proc = {"rescue": rescue, "shuffled": bool(shuffled)}
    if probe_epochs:
        proc["PROBE_ONLY"] = (
            "SMOKE PROBE at a REDUCED epoch budget (%d). The procedure assertions "
            "below are enforced at THIS budget so the code path is exercised "
            "cheaply; the registered budgets (450 for R-budget, 75/75/75 for "
            "R-layerwise) are enforced on every real run and their arithmetic is "
            "asserted separately in `block_partition`." % probe_epochs)
    if rescue == "R-identity":
        touched = apply_identity_init(m, seed, va, vb)
        proc["identity"] = identity_assertions(m, va, vb, seed)
        proc["identity"]["blocks_reinitialised"] = touched
    elif rescue == "R-warmstart":
        assert donor_state is not None, "R-warmstart requires a donor state dict"
        m.load_state_dict(donor_state)
        proc["warmstart"] = warmstart_assertions(m, donor_state)

    # ---- the epoch budget and the stage plan
    if rescue == "R-budget":
        total = probe_epochs or BUDGET_EPOCHS
        stages = [(total, ("event1", "event2", "readout"))]
    elif rescue == "R-layerwise":
        stages = ([(probe_epochs, t) for _n, t in LAYERWISE_STAGES]
                  if probe_epochs else list(LAYERWISE_STAGES))
        total = sum(s[0] for s in stages)
    else:
        total = max_epochs or MAX_EPOCHS
        stages = [(total, ("event1", "event2", "readout"))]

    groups = param_groups(m, va, vb) if hasattr(m, "blocks") else None
    if rescue == "R-layerwise":
        assert groups is not None, "R-layerwise needs the block partition"

    rng = random.Random(RNG_BASE + seed)
    best = {"val": -1.0, "train": -1.0, "epoch": -1, "state": None, "stage": -1}
    lr_trace, curve, stage_log = [], [], []
    t0 = time.time()
    ep_global = 0

    # The committed loop constructs ONE Adam over ALL parameters, before the RNG
    # is drawn, and never rebuilds it. R-layerwise is the only arm that departs,
    # and doc 58 SS6.3.3 registers the departure.
    opt = None
    if rescue != "R-layerwise":
        opt = torch.optim.Adam(m.parameters(), lr=LR)

    for si, (n_ep, trainable) in enumerate(stages):
        entry_snapshot = None
        if rescue == "R-layerwise":
            frozen = [g for g in ("event1", "event2", "readout")
                      if g not in trainable]
            for g in ("event1", "event2", "readout"):
                for p in groups[g]:
                    p.requires_grad_(g in trainable)
                    if g not in trainable:
                        p.grad = None       # no stale grad crosses a stage boundary
            entry_snapshot = {g: [p.detach().clone() for p in groups[g]]
                              for g in frozen}
            tp = [p for g in trainable for p in groups[g]]
            opt = torch.optim.Adam(tp, lr=LR)     # FRESH, doc 58 SS6.3.3
            n_train_p = sum(int(p.numel()) for p in tp)
            n_opt_p = sum(int(p.numel())
                          for pg in opt.param_groups for p in pg["params"])
            assert len(opt.state) == 0, \
                ("doc 58 SS6.3.3: the optimiser's state is NOT EMPTY at the entry "
                 "to stage %d -- stale Adam moments accumulated while a parameter "
                 "was frozen are not staged training" % (si + 1))
            assert n_opt_p == n_train_p, \
                ("the optimiser's param_groups hold %d parameters, the stage's "
                 "trainable subset %d" % (n_opt_p, n_train_p))
            stage_log.append({"stage": si + 1, "epochs": n_ep,
                              "trainable_groups": list(trainable),
                              "frozen_groups": frozen,
                              "n_trainable_params": n_train_p,
                              "n_optimiser_params": n_opt_p,
                              "optimiser_state_empty_at_entry": True,
                              "weight": n_train_p / float(n_params)})

        for _e in range(n_ep):
            ep_global += 1
            lr_ep = (cosine_lr(LR, ep_global, total) if rescue == "R-budget" else LR)
            for pg in opt.param_groups:
                pg["lr"] = lr_ep
            lr_trace.append(lr_ep)
            order = list(range(len(data["train"])))
            rng.shuffle(order)
            for s in range(0, len(order), BATCH):
                opt.zero_grad()
                bl = 0.0
                for i in order[s:s + BATCH]:
                    cs = data["train"][i]
                    bl = bl + torch.nn.functional.cross_entropy(
                        torch.log(m.story_values(cs) + 1e-12).unsqueeze(0),
                        torch.tensor([cs.answer]))
                (bl / min(BATCH, len(order) - s)).backward()
                assert math.isfinite(grad_global_norm(m))
                opt.step()
            with torch.no_grad():
                va_acc = evaluate(m, data["val"])
            curve.append({"epoch": ep_global, "stage": si + 1, "lr": lr_ep,
                          "val_acc": va_acc})
            if va_acc > best["val"]:
                with torch.no_grad():
                    best = {"val": va_acc, "train": evaluate(m, data["train"]),
                            "epoch": ep_global, "stage": si + 1,
                            "state": copy.deepcopy(m.state_dict())}
            if ep_global % 50 == 0 or ep_global == total:
                print("[%s s%d %s] ep %3d/%d val=%.3f best=%.3f %.0fs"
                      % (arm, seed, rescue or "anchor", ep_global, total, va_acc,
                         best["val"], time.time() - t0), flush=True)

        if rescue == "R-layerwise":
            # doc 58 SS7.2: A FREEZE THAT DOES NOT FREEZE IS A NULL INSTRUMENT.
            worst = 0.0
            for g, snaps in entry_snapshot.items():
                for p, q in zip(groups[g], snaps):
                    worst = max(worst, float((p.detach() - q).abs().max()))
            assert worst == 0.0, \
                ("doc 58 SS7.2 FAILED: a FROZEN parameter moved by %.3e during "
                 "R-layerwise stage %d -- the freeze is not freezing and the arm "
                 "is a null instrument" % (worst, si + 1))
            stage_log[-1]["frozen_max_abs_move"] = worst
            stage_log[-1]["frozen_bit_identical"] = True

    if rescue == "R-layerwise":
        for p in m.parameters():
            p.requires_grad_(True)
        weighted = sum(s["epochs"] * s["weight"] for s in stage_log)
        proc["layerwise"] = {
            "stages": stage_log,
            "raw_epochs": total, "raw_multiple_of_baseline": total / float(MAX_EPOCHS),
            "parameter_update_weighted_budget": weighted,
            "baseline_budget": float(MAX_EPOCHS),
            "budget_matched_exactly": bool(abs(weighted - MAX_EPOCHS) < 1e-9),
            "trainable_counts": [s["n_trainable_params"] for s in stage_log],
            "fresh_optimiser_per_stage": True,
            "note": ("doc 58 SS6.3.2, STATED BOTH WAYS AND NEITHER QUOTED ALONE: "
                     "raw epochs %d = %.2fx the committed 150; parameter-update-"
                     "weighted budget %.3f = EXACTLY the baseline 150. The guard "
                     "that makes the raw excess non-fatal is registered in advance: "
                     "R-budget runs at 3.00x in the same units, so a R-layerwise "
                     "rescue while R-budget does not rescue CANNOT be attributed to "
                     "the extra 1.5x of epochs."
                     % (total, total / float(MAX_EPOCHS), weighted)),
            "adaptation_clause_doc58_SS4_2": (
                "R-layerwise is an ADAPTATION of arXiv:2006.14904, not a "
                "replication. Skolik et al. GROW the circuit; exp72's depth is "
                "FIXED by the instance, so this arm implements the paper's other "
                "mechanism -- training parameter subsets with the complement "
                "frozen. A NULL ON THIS ARM IS NOT A NULL ON THEIR PUBLISHED "
                "PROCEDURE."),
        }
    if rescue == "R-budget":
        proc["budget"] = assert_lr_trace(lr_trace, total, LR)
        proc["budget"]["multiple_of_baseline"] = BUDGET_MULTIPLE
        proc["budget"]["bundled_intervention_doc58_SS6_1_1"] = (
            "R-budget changes TWO things -- the epoch count AND the lr schedule "
            "(the committed loop is constant-lr Adam). A rescue licenses 'the "
            "committed recipe undertrained this instance' and does NOT license a "
            "decomposition into budget-versus-schedule; that decomposition is "
            "NOTED, NOT RUN (doc 58 SS13 item 3).")

    m.load_state_dict(best["state"])
    te, rows = R70.evaluate_with_traces(m, data["test"])
    assert len(data["test"]) == 300, \
        "n_test is %d, not the committed 300" % len(data["test"])

    pt = sha = None
    if save:
        os.makedirs(CKPT_DIR, exist_ok=True)
        pt = os.path.join(CKPT_DIR, "%s.pt" % (label or ("%s%s_s%d"
                                                         % (tag, arm, seed))))
        torch.save(best["state"], pt)
        sha = sha256_file(pt)
    cfg = R70.ARM_CFG[R70.base_arm(arm)]
    return {"label": label, "arm": arm, "rescue": rescue, "is_floor": bool(shuffled),
            "seed": seed, "T": cfg["T"], "encoding": cfg["enc"],
            "train_acc": best["train"], "val_acc": best["val"],
            "selected_epoch": best["epoch"], "selected_stage": best["stage"],
            "heldout_test_acc": te, "val_minus_heldout": best["val"] - te,
            "gate_met": bool(best["train"] >= TRAIN_GATE and best["val"] >= VAL_GATE),
            "n_params": n_params, "wires": cfg["qpr"] * 2,
            "n_train": len(data["train"]), "n_val": len(data["val"]),
            "n_test": len(data["test"]),
            "max_epochs": total, "lr": LR, "batch": BATCH,
            "dataset": path, "dataset_sha256": meta["dataset_sha256"],
            "ckpt_path": pt, "ckpt_sha256": sha,
            "procedure_assertions": proc,
            "curve": curve,
            "final_state": {k: v.detach().clone() for k, v in m.state_dict().items()}
            if rescue == "R-warmstart" else None,
            "heldout_items": rows,
            "wall_sec": time.time() - t0}


def _strip(rec):
    """The record as it is written to disk (tensors are not JSON)."""
    out = dict(rec)
    out.pop("final_state", None)
    return out


# ============================================================== doc 58 SS7 smoke
def check_imports_and_targets(store, rep):
    """doc 58 SS7.4 (dataset sha) and SS7.5 (the comparator targets are IMPORTED,
    never re-typed) -- plus the committed per-seed anchor values, read
    programmatically from the two results files."""
    _items, meta, path = store
    assert meta["dataset_sha256"] == DATASET_SHA_DOC58, \
        ("doc 58 SS7.4 FAILED: %s carries sha %s, not the registered %s"
         % (path, meta["dataset_sha256"], DATASET_SHA_DOC58))
    tgt = comparator_targets()
    for inst in ("I1", "I2"):
        v = tgt[inst]
        assert v["comparator"] == 1.0, \
            ("doc 58 SS7.5 FAILED: %s's class-tier comparator imports as %r, not "
             "1.0000 exactly" % (inst, v["comparator"]))
    rep["dataset"] = {"path": path, "sha256": meta["dataset_sha256"],
                      "registered_sha256": DATASET_SHA_DOC58, "matches": True,
                      "n_items_T2": len([i for i in _items
                                         if int(i.get("T", 1)) == 2]),
                      "no_datagen_doc58_SS16_1": (
                          "exp72 generates NO Arabic. The dataset is exp69's "
                          "Part A, byte-identical and sha-pinned, already V-suite "
                          "cleared. doc 50 SS5.2's native-speaker gate does not "
                          "reopen. Stated explicitly because 'nothing changed' is "
                          "exactly the claim that should be checked.")}
    rep["comparator_targets"] = tgt
    return rep


def comparator_targets():
    """doc 58 SS3: exp72 runs NO classical comparator. The two 1.0000 targets are
    READ PROGRAMMATICALLY from the committed results files and asserted."""
    r69 = json.load(open(EXP69_RESULTS, encoding="utf-8"))
    r70 = json.load(open(EXP70_RESULTS, encoding="utf-8"))
    c1 = r69["comparator"]["arms"]["R2-W"]
    c2 = r70["comparator"]["arms"]["N-in-T2"]["class"]
    return {
        "I1": {"comparator": float(c1["mean"]), "sd": float(c1["sd"]),
               "per_seed": list(c1["per_seed"]), "feature_dim": c1["feature_dim"],
               "source": "%s :: comparator.arms['R2-W']" % EXP69_RESULTS,
               "tier": "class (81-dim, doc 55 SS8)"},
        "I2": {"comparator": float(c2["mean"]), "sd": float(c2["sd"]),
               "per_seed": list(c2["per_seed"]), "feature_dim": c2["feature_dim"],
               "source": "%s :: comparator.arms['N-in-T2']['class']" % EXP70_RESULTS,
               "tier": "class (36-dim, doc 56 SS8.1 tier (ii))"},
        "note": ("doc 58 SS3 / SS7.5: exp72 runs NO comparator. The dataset is "
                 "byte-identical, the features are the same deterministic "
                 "constants, the splits and fit seed are the same and the fit is "
                 "deterministic; doc 47 SS14 is the standing reason not to "
                 "re-run it. The values are imported and asserted == 1.0000, "
                 "never re-typed."),
    }


def committed_anchor_values():
    """doc 58 SS1.1's per-seed committed values, read from the results files."""
    r69 = json.load(open(EXP69_RESULTS, encoding="utf-8"))
    r70 = json.load(open(EXP70_RESULTS, encoding="utf-8"))
    out = {
        "I1": {"per_seed": list(r69["arms"]["R2-W"]["per_seed"]),
               "mean": float(r69["arms"]["R2-W"]["mean"]),
               "sd": float(r69["arms"]["R2-W"]["sd"]),
               "seeds": list(r69["arms"]["R2-W"]["seeds"]),
               "floor_mean": float(r69["floors"]["R2-W"]["mean"]),
               "floor_sd": float(r69["floors"]["R2-W"]["sd"]),
               "floor_per_seed": list(r69["floors"]["R2-W"]["per_seed"]),
               "source": "%s :: arms['R2-W'] / floors['R2-W']" % EXP69_RESULTS},
        "I2": {"per_seed": list(r70["arms"]["N-in-T2"]["per_seed"]),
               "mean": float(r70["arms"]["N-in-T2"]["mean"]),
               "sd": float(r70["arms"]["N-in-T2"]["sd"]),
               "seeds": list(r70["arms"]["N-in-T2"]["seeds"]),
               "floor_mean": float(r70["floors"]["N-in-T2"]["mean"]),
               "floor_sd": float(r70["floors"]["N-in-T2"]["sd"]),
               "floor_per_seed": list(r70["floors"]["N-in-T2"]["per_seed"]),
               "source": "%s :: arms['N-in-T2'] / floors['N-in-T2']" % EXP70_RESULTS},
        "PC": {"per_seed": list(r69["arms"]["R2-A"]["per_seed"]),
               "mean": float(r69["arms"]["R2-A"]["mean"]),
               "sd": float(r69["arms"]["R2-A"]["sd"]),
               "seeds": list(r69["arms"]["R2-A"]["seeds"]),
               "source": "%s :: arms['R2-A']" % EXP69_RESULTS},
    }
    for k, v in (("I1", 0.7293), ("I2", 0.6467), ("PC", 0.9187)):
        assert abs(out[k]["mean"] - v) < 5e-5, \
            ("the imported %s mean %.6f does not match doc 58's %.4f"
             % (k, out[k]["mean"], v))
    return out


def bars():
    """doc 58 SS9.2's frozen bars, COMPUTED AT FULL PRECISION from the imported
    committed values -- never re-typed from the 4-dp display table."""
    anc = committed_anchor_values()
    tgt = comparator_targets()
    out = {}
    for inst in ("I1", "I2"):
        base = anc[inst]["mean"]
        comp = tgt[inst]["comparator"]
        gap = comp - base
        out[inst] = {"baseline": base, "comparator": comp, "gap": gap,
                     "rescue_full_bar": comp - 0.02,
                     "rescue_partial_bar": base + 0.50 * gap,
                     "doc58_display": {"I1": {"full": 0.980000, "partial": 0.864667},
                                       "I2": {"full": 0.980000,
                                              "partial": 0.823333}}[inst]}
        for k, d in (("rescue_full_bar", "full"), ("rescue_partial_bar", "partial")):
            assert abs(out[inst][k] - out[inst]["doc58_display"][d]) < 1e-5, \
                ("the computed %s bar for %s (%.6f) disagrees with doc 58 SS9.2's "
                 "table (%.6f)" % (d, inst, out[inst][k],
                                   out[inst]["doc58_display"][d]))
    return out


def check_instance_validity(fm, verbs, store, rep):
    """doc 58 SS7.1's inherited assertions, re-run through exp72's codepath for the
    two instances -- because a rescue arm that silently unwired the CRY would
    'rescue' I2 by making it a DIFFERENT INSTANCE."""
    out = {}
    for inst, arm in (("I1", "A0-W-T2"), ("I2", "N-in-T2")):
        items, _m, _p = R70.arm_items(arm, store)
        ang, _t1 = R70.arm_angles(arm, fm)
        data = R69.compile_all([i for i in items if i["split"] == "test"], ang)
        torch.manual_seed(0)
        m = R70.build_model(arm, verbs, 0, fm)
        w, wd = R70.fourier_resid_2d(m, data)
        if inst == "I1":
            assert w["3"] > 1e-3, \
                ("A0-W-T2 CLOSES in the 3-term basis of theta1 alone (%.3e) -- the "
                 "second qubit is unwired" % w["3"])
            assert w["9"] < 1e-6, \
                ("A0-W-T2 does not close in the 9-dim product basis (%.3e)" % w["9"])
        else:
            assert w["6"] < 1e-6, \
                ("N-in-T2 does not close in the 6-dim E-in basis (%.3e) -- the arm "
                 "is not computing what doc 56 SS4.1 says" % w["6"])
            assert w["3"] > 1e-3, \
                ("N-in-T2 CLOSES in the 3-term basis of theta1 alone (%.3e) -- THE "
                 "CRY'S TARGET IS UNWIRED (doc 56 SS7.1's double negative, first "
                 "direction)" % w["3"])
            assert w["9"] > 1e-3, \
                ("N-in-T2 CLOSES in the 9-dim PRODUCT basis (%.3e) -- the CRY has "
                 "been replaced by an uncontrolled RY (second direction)" % w["9"])
        assert wd > 1e-6, "perturbing theta2 does not move %s" % arm
        out[inst] = {"arm": arm, "resid_3term_theta1_only": w["3"],
                     "resid_9dim_product_basis": w["9"],
                     "resid_6dim_Ein_basis": w["6"],
                     "output_change_under_theta2_perturbation": wd}
    rep["fourier_closure"] = {
        "per_instance": out,
        "N_in_double_negative": (
            "doc 56 SS7.1 / doc 58 SS7.1, re-run because a rescue arm that silently "
            "unwired the CRY would 'rescue' I2 by making it a different instance. "
            "Measured on I2: %.3e (6-dim, MUST close), %.3e (3-term, MUST fail), "
            "%.3e (9-dim product, MUST fail)."
            % (out["I2"]["resid_6dim_Ein_basis"],
               out["I2"]["resid_3term_theta1_only"],
               out["I2"]["resid_9dim_product_basis"])),
    }
    R70.check_concurrence(fm, store, rep)
    R70.check_copy(fm, verbs, store, rep)
    R70.check_trace(fm, verbs, store, rep)
    check_shuffle(store, rep)
    return rep


def check_shuffle(store, rep):
    """doc 46 SS10.2 / doc 55 SS4.4, re-asserted through exp72's codepath because
    doc 58 SS14's floors are the committed S-shuffle construction run UNDER a
    rescue procedure. The geometry is bit-identical to the matched arm; only the
    feature <-> label correlation is severed, per PAIR, so twins stay on opposite
    labels and V1 stays exactly 0.500."""
    items, meta, _p = store
    bearers = set(meta["bearers"])
    rows = []
    for T in (1, 2):
        sel = [i for i in items if int(i.get("T", 1)) == T]
        for seed in range(N_SEEDS):
            sh = R69.relabel_shuffle_q(sel, seed)
            by = {}
            for it in sh:
                by.setdefault(it["pair_id"], []).append(it)
            opp = all(len(g) == 2 and g[0]["answer"] != g[1]["answer"]
                      for g in by.values())
            same_cast = all([r["name"] for r in g[0]["referents"]]
                            == [r["name"] for r in g[1]["referents"]]
                            for g in by.values())
            same_q = all(g[0]["query_index"] == g[1]["query_index"]
                         for g in by.values())
            n_yes = sum(1 for i in sh if i["answer"] == "yes")
            true_rule = sum(1 for i in sh
                            if (i["events"][i["query_index"] - 1]["agent"]
                                in bearers) == (i["answer"] == "yes"))
            assert opp and same_cast and same_q, \
                "S-shuffle broke the twin structure -- V1 would no longer be 0.500"
            assert n_yes / len(sh) == 0.5, "S-shuffle unbalanced the classes"
            assert abs(true_rule / len(sh) - 0.5) < 0.16, \
                ("S-shuffle did not sever features <-> label: the true bearer rule "
                 "still scores %.4f" % (true_rule / len(sh)))
            rows.append({"T": T, "seed": seed, "n_items": len(sh),
                         "twins_still_opposite": bool(opp),
                         "casts_unchanged": bool(same_cast),
                         "query_index_unchanged": bool(same_q),
                         "yes_fraction": n_yes / len(sh),
                         "true_bearer_rule_accuracy": true_rule / len(sh)})
    rep["shuffle"] = {
        "per_seed": rows,
        "mean_true_bearer_rule_accuracy":
            float(np.mean([r["true_bearer_rule_accuracy"] for r in rows])),
        "note": ("doc 46 SS10.2's floor, per PAIR, keyed on the QUERIED event's "
                 "agent, at BOTH T -- because doc 58 SS14.4's severed-donor "
                 "protocol shuffles the T = 1 item set as well, under the SAME "
                 "seed (the registered SEED LINKAGE). The TRUE bearer rule "
                 "collapses to chance on the shuffled labels, which is the "
                 "severance, measured rather than asserted."),
    }
    return rep


def check_partition(fm, verbs, store, rep):
    """doc 58 SS6.3.1: the block partition is EXACT, and it is a property of the
    FROZEN dataset. Asserted, not assumed -- including the zero-crossover claim on
    all T = 2 items."""
    va, vb = verb_groups(store)
    items, _m, _p = R70.arm_items("N-in-T2", store)
    x1 = {e["verb"] for it in items for e in it["events"][:1]}
    x2 = {e["verb"] for it in items for e in it["events"][1:2]}
    assert x1 <= set(va) and x2 <= set(vb), \
        ("doc 58 SS6.3.1 FAILED: event 1 draws %s (V_A = %s), event 2 draws %s "
         "(V_B = %s)" % (sorted(x1), va, sorted(x2), vb))
    torch.manual_seed(0)
    m = R70.build_model("N-in-T2", verbs, 0, fm)
    g = param_groups(m, va, vb)
    counts = {k: sum(int(p.numel()) for p in v) for k, v in g.items()}
    total = sum(p.numel() for p in m.parameters())
    assert counts == {"event1": 99, "event2": 99, "readout": 66} and total == 264, \
        ("doc 58 SS6.3.1's partition arithmetic FAILED: %s of %d" % (counts, total))
    stage_counts = [sum(counts[k] for k in tr) for _n, tr in LAYERWISE_STAGES]
    assert stage_counts == [165, 99, 264], \
        ("doc 58 SS7.2: the per-stage trainable counts are %s, not 165 / 99 / 264"
         % stage_counts)
    weighted = sum(n * (sum(counts[k] for k in tr) / float(total))
                   for n, tr in LAYERWISE_STAGES)
    assert abs(weighted - 150.0) < 1e-9, \
        ("doc 58 SS6.3.2's parameter-update-weighted budget is %.6f, not exactly "
         "150.000" % weighted)
    # parameter-count identity at T = 1 and T = 2 (doc 58 SS6.4)
    ident = {}
    for a, b in (("A0-W-T1", "A0-W-T2"), ("N-in-T1", "N-in-T2"),
                 ("EA-ctrl-T1", "EA-ctrl-T2")):
        torch.manual_seed(0)
        na = sum(p.numel() for p in R70.build_model(a, verbs, 0, fm).parameters())
        torch.manual_seed(0)
        nb = sum(p.numel() for p in R70.build_model(b, verbs, 0, fm).parameters())
        assert na == nb, "parameter counts differ between %s and %s" % (a, b)
        ident["%s / %s" % (a, b)] = na
    rep["block_partition"] = {
        "V_A": va, "V_B": vb,
        "event1_verbs_observed": sorted(x1), "event2_verbs_observed": sorted(x2),
        "zero_crossover_on_all_T2_items": True,
        "n_T2_items_checked": len(items),
        "param_counts": counts, "total_params": total,
        "per_stage_trainable": stage_counts,
        "parameter_update_weighted_budget": weighted,
        "raw_epochs": sum(n for n, _t in LAYERWISE_STAGES),
        "T1_T2_parameter_identity": ident,
        "note": ("doc 58 SS6.3.1 / SS6.3.2. The partition needs NO architectural "
                 "change: V_A and V_B are disjoint by construction and at T = 2 "
                 "event 1 draws only from V_A, event 2 only from V_B. The weighted "
                 "budget is EXACTLY 150.000 only because stage 2 freezes the "
                 "readout as well (orchestrator ruling 1)."),
    }
    return rep


def check_negative_directions(fm, verbs, store, rep):
    """doc 58 SS7.2 / SS16.3 step 6: **every assertion designed to fail is fed the
    input that must make it fail.** An assertion that has never been seen to fail
    is not a gate (doc 49 SS8's checklist entry)."""
    va, vb = verb_groups(store)
    out = {}

    # (1) a CONSTANT lr trace must fail R-budget's schedule assertion
    try:
        assert_lr_trace([LR] * BUDGET_EPOCHS, BUDGET_EPOCHS, LR)
        out["constant_lr_rejected"] = False
    except AssertionError as exc:
        out["constant_lr_rejected"] = True
        out["constant_lr_message"] = str(exc)[:200]
    assert out["constant_lr_rejected"], \
        "the lr assertion ACCEPTED a constant trace -- it is not a gate"

    # (2) a trace of the WRONG LENGTH must fail
    try:
        assert_lr_trace([cosine_lr(LR, e, 150) for e in range(1, 151)],
                        BUDGET_EPOCHS, LR)
        out["wrong_length_rejected"] = False
    except AssertionError:
        out["wrong_length_rejected"] = True
    assert out["wrong_length_rejected"]

    # (3) R-identity's null-instrument check must fire when the READOUT is also
    #     initialised at identity (doc 58 SS6.2.2 -- the amendment this paragraph
    #     exists to prevent)
    torch.manual_seed(0)
    m = R70.build_model("A0-W-T2", verbs, 0, fm)
    apply_identity_init(m, 0, va, vb)
    with torch.no_grad():
        m.q_pos.theta.zero_()
        m.q_neg.theta.zero_()
    try:
        identity_assertions(m, va, vb, 0)
        out["identity_readout_rejected"] = False
    except AssertionError as exc:
        out["identity_readout_rejected"] = True
        out["identity_readout_message"] = str(exc)[:200]
    assert out["identity_readout_rejected"], \
        ("doc 58 SS6.2.2's null-instrument check ACCEPTED an identity-initialised "
         "readout -- v_yes == v_no everywhere and the check is not a gate")

    # (4) the identity init's sd window must reject a WRONG sigma
    torch.manual_seed(0)
    m2 = R70.build_model("A0-W-T2", verbs, 0, fm)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for v in sorted(list(va) + list(vb)):
            t = m2.blocks[v].theta
            t.copy_(torch.randn(t.shape, generator=g) * 0.5)     # wrong sigma
    try:
        identity_assertions(m2, va, vb, 0)
        out["wrong_sigma_rejected"] = False
    except AssertionError:
        out["wrong_sigma_rejected"] = True
    assert out["wrong_sigma_rejected"]

    # (5) a FREEZE THAT DOES NOT FREEZE must fail: move a "frozen" parameter and
    #     require the stage check's comparison to be non-zero
    torch.manual_seed(0)
    m3 = R70.build_model("A0-W-T2", verbs, 0, fm)
    grp = param_groups(m3, va, vb)
    snap = [p.detach().clone() for p in grp["event2"]]
    with torch.no_grad():
        # 1e-4 rather than 1e-7: the simulator is float32 END TO END (exp70's
        # precision note), and eps at theta ~ 2pi is ~4.8e-07, so a 1e-07 nudge is
        # not representable and the negative test would be testing float32, not
        # the gate. The gate itself compares BIT-IDENTITY and admits no tolerance.
        grp["event2"][0][0, 0] += 1e-4
    moved = max(float((p.detach() - q).abs().max())
                for p, q in zip(grp["event2"], snap))
    out["freeze_check_detects_move"] = bool(moved > 0.0)
    out["freeze_check_smallest_detected_move"] = moved
    assert out["freeze_check_detects_move"], \
        "the freeze check cannot detect a moved parameter -- it is not a gate"

    # (6) a NON-EMPTY optimiser state at a stage boundary must fail
    opt = torch.optim.Adam([m3.q_pos.theta], lr=LR)
    m3.q_pos.theta.grad = torch.zeros_like(m3.q_pos.theta)
    opt.step()
    out["nonempty_optimiser_state_detected"] = bool(len(opt.state) > 0)
    assert out["nonempty_optimiser_state_detected"], \
        "the optimiser-state check cannot see a populated Adam state"

    # (7) a warm start that did NOT land must fail
    torch.manual_seed(0)
    m4 = R70.build_model("A0-W-T2", verbs, 0, fm)
    state, _sha = load_donor(INSTANCES["I1"]["donor_ckpt"],
                             INSTANCES["I1"]["donor_sha256"], 264)
    try:
        warmstart_assertions(m4, state)                  # NOT loaded on purpose
        out["unloaded_warmstart_rejected"] = False
    except AssertionError:
        out["unloaded_warmstart_rejected"] = True
    assert out["unloaded_warmstart_rejected"]

    # (8) bit-identical warm-start finals must fail the spread check
    fake = {0: {k: v.clone() for k, v in state.items()},
            1: {k: v.clone() for k, v in state.items()}}
    sp = check_warmstart_spread(fake)
    out["identical_finals_rejected"] = bool(not sp["assertion_passed"])
    assert out["identical_finals_rejected"], \
        ("the spread check ACCEPTED five bit-identical runs -- doc 58 SS18's "
         "null-instrument-by-symmetry row is not being enforced")

    # (9) a WRONG donor sha must fail
    try:
        load_donor(INSTANCES["I1"]["donor_ckpt"], "0" * 64, 264)
        out["wrong_donor_sha_rejected"] = False
    except AssertionError:
        out["wrong_donor_sha_rejected"] = True
    assert out["wrong_donor_sha_rejected"]

    rep["failing_directions"] = dict(
        out, note=("doc 58 SS7.2 / doc 48 SS10.6. EVERY procedure assertion is fed "
                   "the input that must make it fail, and the failure is recorded. "
                   "An assertion that has never been seen to fail is not a gate "
                   "(doc 49 SS8's checklist entry, orchestrator ruling 6 of doc "
                   "56)."))
    return rep


def smoke(fm, verbs, store):
    t0 = time.time()
    rep = {"experiment": "exp72 smoke / validity assertions (doc 58 SS7)",
           "spec": SPEC, "spec_commit_line": spec_commit(),
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "numpy": np.__version__}
    check_imports_and_targets(store, rep)
    rep["bars_doc58_SS9_2"] = bars()
    rep["committed_values"] = committed_anchor_values()
    check_partition(fm, verbs, store, rep)
    check_instance_validity(fm, verbs, store, rep)
    check_negative_directions(fm, verbs, store, rep)

    # ---- a 2-epoch probe of EVERY cell, so the procedure paths are exercised
    tiny = {}
    donors = {i: load_donor(INSTANCES[i]["donor_ckpt"],
                            INSTANCES[i]["donor_sha256"],
                            INSTANCES[i]["n_params"])[0] for i in INSTANCES}
    for inst in ("I1", "I2"):
        arm = INSTANCES[inst]["arm"]
        for rescue in [None] + RESCUES:
            kw = {}
            if rescue == "R-warmstart":
                kw["donor_state"] = donors[inst]
            r = train(arm, 0, store, verbs, fm, rescue=rescue,
                      max_epochs=2, probe_epochs=(2 if rescue == "R-layerwise"
                                                  else (6 if rescue == "R-budget"
                                                        else None)),
                      save=False, label="smoke", **kw)
            key = "%s|%s" % (inst, rescue or "anchor")
            tiny[key] = {"val_acc": r["val_acc"],
                         "heldout_test_acc": r["heldout_test_acc"],
                         "epochs": r["max_epochs"], "wall_sec": r["wall_sec"],
                         "n_params": r["n_params"],
                         "procedure_assertions": {
                             k: v for k, v in r["procedure_assertions"].items()
                             if k != "identity"}}
            if rescue == "R-identity":
                tiny[key]["identity"] = {
                    k: v for k, v in r["procedure_assertions"]["identity"].items()
                    if k != "blocks_reinitialised"}
            print("[smoke] %-22s ep=%d val=%.3f test=%.3f  %.0fs"
                  % (key, r["max_epochs"], r["val_acc"], r["heldout_test_acc"],
                     r["wall_sec"]), flush=True)
    # the positive control's path
    r = train(PC_ARM, 0, store, verbs, fm, rescue=None, max_epochs=2, save=False,
              label="smoke")
    tiny["PC|anchor-probe"] = {"val_acc": r["val_acc"],
                               "heldout_test_acc": r["heldout_test_acc"],
                               "n_params": r["n_params"]}
    # a shuffled path, so the floor driver is exercised before stage 2 needs it
    r = train("A0-W-T2", 0, store, verbs, fm, rescue=None, shuffled=True,
              max_epochs=2, save=False, label="smoke")
    tiny["I1|floor-probe"] = {"val_acc": r["val_acc"],
                              "heldout_test_acc": r["heldout_test_acc"]}
    rep["two_epoch_probe"] = tiny

    # R-warmstart's failing direction, measured on two real (2-epoch) runs
    finals = {}
    for s in (0, 1):
        rr = train(INSTANCES["I1"]["arm"], s, store, verbs, fm,
                   rescue="R-warmstart", donor_state=donors["I1"], max_epochs=2,
                   save=False, label="smoke")
        finals[s] = rr["final_state"]
    rep["warmstart_spread_probe"] = check_warmstart_spread(finals)
    assert rep["warmstart_spread_probe"]["assertion_passed"], \
        ("two R-warmstart runs at different seeds produced BIT-IDENTICAL "
         "parameters -- doc 58 SS6.0's premise is false for this build")

    rep["assertions_passed"] = True
    rep["wall_time_sec"] = time.time() - t0
    rep["git_note"] = git_note()
    path = fresh_results_path(SMOKE_BASE)
    json.dump(rep, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[smoke] wrote %s -- ALL doc 58 SS7 ASSERTIONS PASSED (including the "
          "failing directions)" % path)
    return rep


# ========================================================= doc 58 SS5, ANCHORS
def anchor_gate(fm, verbs, store, seeds, insts=("I1", "I2")):
    rows = []
    for inst in insts:
        arm = INSTANCES[inst]["arm"]
        for s in seeds:
            label = "%s_anchor_s%d" % (inst, s)
            ckj = os.path.join(CKPT_DIR, label + ".json")
            if os.path.exists(ckj):
                r = json.load(open(ckj, encoding="utf-8"))
                print("[anchor] %s already done: %.6f" % (label,
                                                          r["heldout_test_acc"]))
            else:
                r = train(arm, s, store, verbs, fm, rescue=None, label=label)
                r["instance"] = inst
                json.dump(_strip(r), open(ckj, "w", encoding="utf-8"))
                print("[anchor] %-14s held-out=%.6f  (%.0fs)"
                      % (label, r["heldout_test_acc"], r["wall_sec"]), flush=True)
            r["instance"] = inst
            rows.append(r)
    return rows


def anchor_report(rows, store):
    """doc 58 SS5.1, BLOCKING. Per-seed |delta| <= 0.002 against the COMMITTED
    per-seed values, arm means to |delta| <= 0.002 of 0.7293 / 0.6467. The honest
    expectation is |delta| = 0 EXACTLY, bit-exact, 10/10 seeds."""
    anc = committed_anchor_values()
    _items, meta, path = store
    per, ok_all, exact_all = {}, True, True
    for inst in ("I1", "I2"):
        rr = sorted([r for r in rows if r.get("instance") == inst],
                    key=lambda x: x["seed"])
        got = [r["heldout_test_acc"] for r in rr]
        want = anc[inst]["per_seed"]
        assert [r["seed"] for r in rr] == anc[inst]["seeds"], \
            "seed sets differ for %s" % inst
        d = [g - w for g, w in zip(got, want)]
        mean = float(np.mean(got))
        ok_seed = all(abs(x) <= ANCHOR_TOLERANCE for x in d)
        ok_mean = abs(mean - INSTANCES[inst]["committed_mean"]) <= ANCHOR_TOLERANCE
        exact = all(x == 0.0 for x in d)
        ok_all = ok_all and ok_seed and ok_mean
        exact_all = exact_all and exact
        per[inst] = {
            "instance_arm": INSTANCES[inst]["arm"],
            "committed_arm": INSTANCES[inst]["committed_arm"],
            "committed_source": anc[inst]["source"],
            "committed_per_seed": want, "committed_mean": anc[inst]["mean"],
            "exp72_per_seed": got, "exp72_mean": mean,
            "exp72_sd": float(np.std(got, ddof=1)),
            "per_seed_delta": d,
            "max_abs_per_seed_delta": float(max(abs(x) for x in d)),
            "mean_delta_vs_committed": mean - anc[inst]["mean"],
            "mean_delta_vs_doc58": mean - INSTANCES[inst]["committed_mean"],
            "bit_exact": bool(exact),
            "per_seed_within_tolerance": bool(ok_seed),
            "mean_within_tolerance": bool(ok_mean),
            "replicates": bool(ok_seed and ok_mean),
            "val_per_seed": [r["val_acc"] for r in rr],
            "selected_epoch_per_seed": [r["selected_epoch"] for r in rr],
            "wall_sec_total": float(sum(r["wall_sec"] for r in rr)),
            "ckpt_sha256": [r["ckpt_sha256"] for r in rr],
        }
    return {
        "component": "exp72 anchors -- doc 58 SS5.1, BLOCKING, READ FIRST",
        "spec": SPEC, "spec_commit_line": spec_commit(),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "dataset": path, "dataset_sha256": meta["dataset_sha256"],
        "tolerance": ANCHOR_TOLERANCE,
        "per_instance": per,
        "all_bit_exact": bool(exact_all),
        "anchors_replicate_committed": bool(ok_all),
        "codepath_assertion_doc58_SS7_3": (
            "the unrescued anchors ran through exp72_run.train() -- THE SAME ENTRY "
            "POINT the rescue arms use, with the rescue options off -- not a "
            "separate legacy path. Otherwise SS5's bit-exactness would prove "
            "nothing about the rescue arms' pipeline."),
        "honest_expectation": (
            "BIT-EXACT, 10/10 seeds. Both runs are deterministic given their seeds; "
            "the 0.002 tolerance exists only to absorb a documented float/BLAS "
            "platform difference and ANY non-zero delta is reported with its cause "
            "identified (doc 58 SS5.1). This is the gate's THIRD enforced use; both "
            "previous uses returned bit-exact."),
        "consequence_if_failed": (
            "doc 58 SS10 row 1: THE RUN STOPS. The exp72 pipeline has drifted from "
            "the exp69/exp70 pipeline, every 'rescue' number would be a comparison "
            "against a moving baseline, SS5.2's floor-omission argument is VOID, "
            "and NOTHING ELSE IN THE RUN IS READ. Hunt the build; interpret "
            "nothing."),
        "floors_omission_argument_doc58_SS5_2": (
            "the anchors carry NO fresh floor. A gate that pins an arm to within "
            "0.002 of a measurement that cleared its own matched floor by 8.02 SD "
            "(I1) and by a cleared doc 56 SS6.2 gate (I2) is TRANSITIVELY STRONGER "
            "than re-measuring the floor. IF EITHER ANCHOR FAILS, THIS ARGUMENT IS "
            "VOID ALONG WITH THE RUN."),
        "git_note": git_note(),
    }


# ================================================================ stage 1 / 2
def stage1_jobs():
    jobs = []
    for inst in ("I1", "I2"):
        for rescue in RESCUES:
            for s in range(N_SEEDS):
                jobs.append({"kind": "rescue", "instance": inst, "rescue": rescue,
                             "seed": s, "arm": INSTANCES[inst]["arm"],
                             "label": "%s_%s_s%d" % (inst, rescue, s)})
    for s in range(N_SEEDS):
        jobs.append({"kind": "control", "instance": PC_INSTANCE, "rescue": PC_RESCUE,
                     "seed": s, "arm": PC_ARM,
                     "label": "PC_%s_s%d" % (PC_RESCUE, s)})
    return jobs


def stage2_jobs(cells):
    """`cells` is a list of "I1:R-budget"-style strings, one per TRIGGERED cell.

    doc 58 SS14.4's severed donor and the floor run it initialises are emitted as
    ONE job, so the two can never be split across workers and the floor can never
    start from a donor that does not exist. The RUN COUNT is unchanged (5 donor
    runs + 5 floor runs per warm-start cell = the registered 10)."""
    jobs = []
    for c in cells:
        inst, rescue = c.split(":")
        assert inst in INSTANCES and rescue in RESCUES, "bad cell %s" % c
        kind = "warmstart_floor" if rescue == "R-warmstart" else "floor"
        for s in range(N_SEEDS):
            jobs.append({"kind": kind, "instance": inst, "rescue": rescue,
                         "seed": s, "arm": INSTANCES[inst]["arm"],
                         "label": "%s_%s_floor_s%d" % (inst, rescue, s),
                         "donor_label": "%s_%s_donorfloor_s%d" % (inst, rescue, s),
                         "donor_arm": INSTANCES[inst]["donor_arm"]})
    return jobs


def _run_one(label, kind, arm, seed, rescue, store, verbs, fm, extra, **kw):
    ckj = os.path.join(CKPT_DIR, label + ".json")
    if os.path.exists(ckj):
        print("[ckpt] %s already done" % label, flush=True)
        return json.load(open(ckj, encoding="utf-8"))
    r = train(arm, seed, store, verbs, fm, label=label, rescue=rescue, **kw)
    r.update(extra)
    r.update({"kind": kind, "cell_rescue": rescue})
    json.dump(_strip(r), open(ckj, "w", encoding="utf-8"))
    print("[exp72] %-32s train=%.3f val=%.3f HELD-OUT=%.4f  (%.0fs)"
          % (label, r["train_acc"], r["val_acc"], r["heldout_test_acc"],
             r["wall_sec"]), flush=True)
    return r


def run_jobs(jobs, fm, verbs, store):
    donors = {}
    for j in jobs:
        inst = j["instance"]
        extra = {"instance": inst}
        if j["kind"] in ("rescue", "control"):
            kw = {}
            if j["rescue"] == "R-warmstart":
                if inst not in donors:
                    donors[inst] = load_donor(INSTANCES[inst]["donor_ckpt"],
                                              INSTANCES[inst]["donor_sha256"],
                                              INSTANCES[inst]["n_params"])[0]
                kw["donor_state"] = donors[inst]
            _run_one(j["label"], j["kind"], j["arm"], j["seed"], j["rescue"],
                     store, verbs, fm, extra, **kw)
            continue

        if j["kind"] == "warmstart_floor":
            # doc 58 SS14.4, IN ORDER AND IN ONE JOB.
            #   1. retrain the T = 1 donor on the SHUFFLED bearer designation at
            #      the committed T = 1 recipe (150 epochs), seed-linked;
            #   2. its checkpoint is selected on the SHUFFLED val accuracy -- the
            #      committed selection rule applied to severed labels, so no true
            #      label ever reaches the initialisation;
            #   3. warm-start the shuffled T = 2 run from THAT donor.
            _run_one(j["donor_label"], "severed_donor", j["donor_arm"], j["seed"],
                     None, store, verbs, fm,
                     dict(extra, severed_donor_for=j["label"]), shuffled=True)
            sev = os.path.join(CKPT_DIR, j["donor_label"] + ".pt")
            assert os.path.exists(sev), \
                ("doc 58 SS14.4: the SEVERED donor %s does not exist. A warm-start "
                 "floor initialised from the TRULY-LABELLED donor would import "
                 "label structure through the init and would be no floor at all."
                 % j["donor_label"])
            state = torch.load(sev, map_location="cpu")
            _run_one(j["label"], "floor", j["arm"], j["seed"], j["rescue"],
                     store, verbs, fm, dict(extra, severed_donor=j["donor_label"],
                                            severed_donor_sha256=sha256_file(sev)),
                     shuffled=True, donor_state=state)
            continue

        _run_one(j["label"], "floor", j["arm"], j["seed"], j["rescue"],
                 store, verbs, fm, extra, shuffled=True)


# ======================================================================== main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--anchor-gate", action="store_true")
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--cells", default="",
                    help='comma-separated triggered cells, e.g. "I1:R-warmstart"')
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()

    store = R70.load_part(R70.DATA_A)
    verbs = sorted(store[1]["verb_inventory"].keys())
    fm = R70.feature_maps()
    os.makedirs(CKPT_DIR, exist_ok=True)

    if a.smoke:
        smoke(fm, verbs, store)
        return 0

    if a.anchor_gate:
        seeds = ([s for s in range(a.seeds) if s % a.nworkers == a.worker]
                 if a.nworkers > 1 else list(range(a.seeds)))
        rows = anchor_gate(fm, verbs, store, seeds)
        if a.nworkers > 1 or len(seeds) != N_SEEDS:
            print("[anchor w%d] done (%d runs); re-run with --nworkers 1 to write "
                  "the report" % (a.worker, len(rows)))
            return 0
        rep = anchor_report(rows, store)
        out = fresh_results_path(ANCHOR_BASE)
        json.dump(rep, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n[ANCHOR GATE] doc 58 SS5.1, |delta| <= %.3f, honest expectation "
              "BIT-EXACT" % ANCHOR_TOLERANCE)
        for inst in ("I1", "I2"):
            p = rep["per_instance"][inst]
            print("  %s (%s == %s)" % (inst, p["instance_arm"],
                                       p["committed_arm"]))
            print("      committed %s  mean %.6f"
                  % ([round(x, 6) for x in p["committed_per_seed"]],
                     p["committed_mean"]))
            print("      exp72     %s  mean %.6f"
                  % ([round(x, 6) for x in p["exp72_per_seed"]], p["exp72_mean"]))
            print("      delta     %s  max |d| %.2e  bit-exact=%s -> %s"
                  % ([round(x, 6) for x in p["per_seed_delta"]],
                     p["max_abs_per_seed_delta"], p["bit_exact"],
                     "PASS" if p["replicates"] else "FAIL"))
        print("  anchors_replicate_committed = %s  ->  %s"
              % (rep["anchors_replicate_committed"],
                 "PROCEED" if rep["anchors_replicate_committed"]
                 else "THE RUN STOPS (doc 58 SS10 row 1)"))
        print("[ANCHOR GATE] wrote %s" % out)
        return 0 if rep["anchors_replicate_committed"] else 1

    # everything below is gated on the anchors
    anchors = sorted((p for p in os.listdir(".")
                      if p.startswith(ANCHOR_BASE) and p.endswith(".json")),
                     key=os.path.getmtime)
    if not anchors:
        raise RuntimeError(
            "no %s*.json -- doc 58 SS16.3 step 7: THE ANCHORS RUN FIRST and block "
            "everything. Run `python exp72_run.py --anchor-gate`." % ANCHOR_BASE)
    anc = json.load(open(anchors[-1], encoding="utf-8"))
    if not anc.get("anchors_replicate_committed"):
        raise RuntimeError(
            "the anchor gate FAILED -- doc 58 SS10 row 1: the run stops, the build "
            "is hunted, and nothing is interpreted from any arm")

    if a.stage1:
        jobs = stage1_jobs()
    elif a.stage2:
        cells = [c for c in a.cells.split(",") if c]
        if not cells:
            raise RuntimeError(
                "--stage2 needs --cells. doc 58 SS14.2: only cells whose held-out "
                "mean REACHES their rescue_partial bar trigger a matched floor.")
        jobs = stage2_jobs(cells)
    else:
        raise RuntimeError("pass --smoke, --anchor-gate, --stage1 or --stage2")

    mine = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    print("[exp72 w%d/%d] %d of %d runs" % (a.worker, a.nworkers, len(mine),
                                            len(jobs)), flush=True)
    run_jobs(mine, fm, verbs, store)
    print("[exp72 w%d] done" % a.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
