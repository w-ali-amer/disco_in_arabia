# -*- coding: utf-8 -*-
"""exp81_run.py -- WIDTH PER REFERENT experiment (doc 73).

Registration: qnlp_private_docs/73_exp81_width_per_referent.md

Arms (on exp42r's task, dataset, splits, readout, controls -- UNCHANGED):
  A2q2     q=2 qubits per referent.  Each referent i (primary wire i in
           A2's K-wire scheme) gets a second qubit at wire K+i.  At
           INTRODUCTION TIME: Ry(frozen_angle) on primary wire i, then
           XX(coupling[i]) on (i, K+i).  The verb block (A2's 4x4
           dressed-symmetric core) acts on (wa, wp) -- PRIMARY wires only.
           Readout: A2's two-effect readout on the queried primary pair
           (qa, qp); secondary wires are traced out automatically by
           prob_all_zero_on.  5 seeds, 30 epochs.

  A2-WIDE  Capacity control.  A2's architecture (K wires, no second qubit)
           with parameter count matched EXACTLY to A2q2 via K_MAX
           additional trainable intro-slot angles.  A2-WIDE adds one extra
           trained Ry per referent SLOT (positional, not per-noun), applied
           as a second rotation after the frozen embedding Ry at intro time.
           Same K_MAX = 12 extra params as A2q2's coupling params, giving
           A2-WIDE total = A2q2 total (EXACT MATCH by construction).
           If A2-WIDE also beats A2, the bar reads CAPACITY not architecture
           (the WAZN-SHUFFLED pattern; doc 73 §3).

  A2       Committed exp42r arm.  Read from exp42r_ckpt/ as DRIFT ANCHOR;
           RETRAINED same-draw (same seeds, same data) for the paired test.
  B1       Committed exp42r arm.  Read from exp42r_ckpt/ read-only.

Construction note (A2q2 coupling gate family):
  "Without inventing an entangler family" (doc 73 §2) points at the
  committed symmetric core's own family.  The core uses CP (controlled-
  phase) and XX (IsingXX); XX(0)=I satisfies the zero-coupling reduction
  needed for SG-ZC.  Coupling gate: sim.xx(coupling[i]) on (i, K+i).
  CRY was exp79's local gadget and is NOT reused here.  Flagged.

Construction note (application point):
  Doc 73 §2 says "one trained two-site coupling gate per referent" --
  PER REFERENT (not per event).  This is interpreted as ONCE AT INTRO,
  right after the primary Ry(angle).  Exp79 by contrast said "per event"
  (per-event CRY).  Flagged: secondaries are not touched again after intro;
  A2q2 = A2 dynamics on a per-referent trained-mixedness initial state.

Construction note (coupling positional, not per-noun):
  A per-noun coupling would add a lexical channel absent from A2, making
  A2q2 differ in two ways at once -- the A2TI confound (exp42r_common.py).
  Coupling is indexed by wire SLOT (0..K_MAX-1).  Flagged.

Param accounting (derived, not guessed):
  n_verbs = 15, readout = 30, K_MAX = 12 (max K across all compiled splits)
  A2 total     = 14 * 15 + 30       = 240
  A2q2 total   = 14 * 15 + 30 + 12  = 252  (12 coupling angles)
  A2-WIDE total = 14 * 15 + 30 + 12  = 252  (12 extra intro-slot angles)
  A2q2:A2 ratio = 252 / 240 = 1.05
  A2-WIDE:A2q2 ratio = 1.00 (exact match by construction)

Doc-70 compliance:
  -- seed-level paired t as the primary significance leg (in aggregate)
  -- item-level sign-flip is descriptive only
  -- floor-admissibility gate: scrambled acc within 2 binomial SD of 0.5
  -- SELF-EMITTING gate artifacts: SG-ZC, SG-FS, SG-PC (each with DTF leg)

Timing measurement (doc 73 §3, derived-not-guessed clause):
  A2 baseline: 0.00878 s/item/epoch (exp79_epoch_timing.json).
  A2q2 at K: state is 2^(2K) vs A2's 2^K -- ~16x larger at K=4,
  ~256x at K=8.  Timing measured on 50 train items, 3 epochs.

Seed base: 81000; 5 seeds per arm (0..4); streams disjoint via
  EXP81_STREAM_BASE = 81_000_000.

Workers: exp81_w*.log, exp81_ckpt/
SMOKE MARKER: exp81_smoke_PASSED.json (workers check before training).

Usage:
  python3 exp81_run.py --smoke               # gate battery + write marker
  python3 exp81_run.py --time-epoch          # measure first-epoch time
  python3 exp81_run.py --worker i --nworkers 5
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback

import torch
from torch import nn

import exp42_compiler as comp
import exp42_sim_adapter as sim
import exp42r_common as K42
from exp42_models import QuantumStoryModel
from exp42r_run import (evaluate, prepare)

# claim / settled / record_failure use exp81's own lock+ctl dirs (not exp42r's)
def claim(job_id):
    """Exclusive-create lock in exp81_locks/."""
    os.makedirs(LOCK_DIR, exist_ok=True)
    path = os.path.join(LOCK_DIR, job_id + ".lock")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, ("pid %d %s\n" % (os.getpid(), time.ctime())).encode())
    os.close(fd)
    return True


def settled(outputs, job_id):
    return (all(os.path.exists(p) for p in outputs)
            or os.path.exists(failed_path81(job_id)))


def record_failure(job_id, exc, tag):
    payload = {"job": job_id, "tag": tag, "error": repr(exc),
               "traceback": traceback.format_exc(),
               "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "isolation": "arm-level failure; worker continues (doc 62 SS24)"}
    os.makedirs(CTL_DIR, exist_ok=True)
    with open(failed_path81(job_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("[FAIL] %s: %r (isolated)" % (job_id, exc), flush=True)

# ================================================================
# exp81 constants
# ================================================================
EXP81_ARMS = ("A2q2", "A2-WIDE", "A2-RETRAIN", "A2q2-FLOOR")  # B1 is read-only from ckpt
EXP81_TRAIN_ARMS = ("A2q2", "A2-WIDE", "A2-RETRAIN", "A2q2-FLOOR")  # arms that require training
EXP81_ARM_INDEX = {a: i for i, a in enumerate(EXP81_ARMS)}
EXP81_STREAM_BASE = 81_000_000   # disjoint from exp42r's 42_000_000 and exp79's 79_000_000
FLOOR_SCRAMBLE_SEED = 81999      # ONE fixed draw, used throughout; disjoint from exp79's 79999

CKPT_DIR = os.environ.get("EXP81_CKPT_DIR", "exp81_ckpt")
CTL_DIR = os.environ.get("EXP81_CTL_DIR", "exp81_ctl")
LOCK_DIR = os.environ.get("EXP81_LOCK_DIR", "exp81_locks")

SMOKE_MARKER = "exp81_smoke_PASSED.json"
SMOKE_REPORT = "exp81_smoke.json"
TIME_OUT = "exp81_epoch_timing.json"

EXP42R_CKPT_DIR = "exp42r_ckpt"   # read-only committed checkpoints

N_SEEDS = 5                        # 5 seeds per arm (doc 73 §2)
INIT_STD = 0.1                     # N(0, 0.1^2) for all params
PASS_TOL = 1e-5                    # SG-ZC tolerance (same as SG-ANT in exp79)

# §24.3 bounded rejection loop (doc 62 §24.1-24.3): sd window committed at [0.08, 0.12]
SD_WINDOW = (0.08, 0.12)
MAX_INIT_DRAWS = 20

# Coupling gate constant: K_MAX = max K across all compiled splits
# Derived from stories_exp41r.json: unique K values = [4,5,6,7,8,10,11,12]
# K_MAX = 12 → 12 coupling params (= 12 extra intro-slot params in A2-WIDE)
K_MAX = 12

# Param accounting (DERIVED, not guessed):
#   A2 total  = 14 * 15 + 30 = 240
#   A2q2 total = 240 + K_MAX = 252
#   A2-WIDE total = 240 + K_MAX = 252  (exact match by construction)
A2_TOTAL_PARAMS = 240
A2Q2_EXTRA_PARAMS = K_MAX         # coupling angles
EXPECTED_A2Q2_TOTAL = A2_TOTAL_PARAMS + A2Q2_EXTRA_PARAMS    # 252
EXPECTED_WIDE_TOTAL = EXPECTED_A2Q2_TOTAL                      # 252 (exact match)
RATIO_A2Q2_OVER_A2 = EXPECTED_A2Q2_TOTAL / A2_TOTAL_PARAMS   # 1.05

BARRIER_TIMEOUT_SEC = 12 * 3600
BARRIER_POLL_SEC = 30

A2Q2_CONSTRUCTION_NOTE = (
    "FLAGGED (doc 73 §2): A2q2 coupling gate family is IsingXX (sim.xx), "
    "same family as the committed symmetric core. CRY was exp79's gadget "
    "and is NOT used here. Application point: ONCE AT INTRO per referent "
    "(not per event as in exp79 C-DISCARD). Coupling is POSITIONAL (slot "
    "index 0..K_MAX-1), not per-noun, to keep the width axis pure (the "
    "A2TI confound). Secondary wires are untouched after intro: A2q2 "
    "implements A2 dynamics on a per-referent trained-mixedness initial "
    "state. The all-four-wire ladder is OUT OF SCOPE (doc 73 §2 naming "
    "clause).")

A2WIDE_CONSTRUCTION_NOTE = (
    "FLAGGED (doc 73 §2): A2-WIDE adds K_MAX=12 extra trained Ry angles, "
    "one per wire SLOT (positional, same index set as A2q2's coupling). "
    "Applied as a SECOND Ry after the frozen embedding Ry at intro time, "
    "on the PRIMARY wire only (no second qubit). This gives EXACT param "
    "match with A2q2 (both 252 total). Extra dressing layers on verb blocks "
    "were considered but would quantize at 12*n_verbs=180 extra params, "
    "massively overshooting K_MAX=12. The slot-wise intro-Ry is the natural "
    "positional analogue of A2q2's coupling and the only way to achieve "
    "exact match at this delta. Flagged as a deviation from 'extra dressing "
    "layers' if that phrase is read as verb-block dressings.")


# ================================================================
# A2q2 model
# ================================================================
class A2q2Model(QuantumStoryModel):
    """A2 + q=2 qubits per referent (doc 73 §2, PINNED construction).

    Wire layout (per story with cs.K primary slots):
      Primary wires:   0 .. cs.K-1      (same as A2)
      Secondary wires: cs.K .. 2*cs.K-1
      Referent slot i: primary=i, secondary=cs.K+i

    Intro: for each (w, noun, angle) in cs.intro:
      1. Ry(angle) on primary wire w  (identical to A2)
      2. XX(coupling[w]) on (w, cs.K+w)  [the minimal lift]
         When coupling[w]=0: XX(0)=I → secondary stays |0>
         → story_values == A2's story_values (SG-ZC assertion)

    Verb block: A2's 4x4 dressed-symmetric verb matrix on (wa, wp)
      (PRIMARY wires only).

    Readout: A2's two-effect readout on (qa, qp) (primary pair);
      prob_all_zero_on marginalises over all other wires including secondaries.

    coupling: nn.ParameterDict, keys "0".."K_MAX-1", one angle per slot.
      Slots 0..K_MAX_TRAIN-1 (=7, train K≤8) are trained; slots 8..11
      (from test_b K=10-12) initialise but are not exercised during training.
    """

    def __init__(self, verbs, seed):
        super().__init__("A2", verbs, seed)
        # arm_label carries A2q2 identity; self.arm stays "A2" so verb_matrix
        # dispatches to A2's block family. (A2TI precedent: same pattern.)
        self.arm_label = "A2q2"
        # K_MAX coupling angles (positional, not per-noun)
        self.coupling = nn.ParameterDict(
            {str(i): nn.Parameter(torch.zeros(())) for i in range(K_MAX)})

    def story_values(self, cs, verb_map=None):
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                raw = self.verb_matrix(v)   # A2's 4x4 (float32 products)
                # SVD polar decomposition: project to exact float32 unitary.
                # verb_matrix builds kron2(A,P) @ core @ kron2(A2,P2) in
                # float32; each matrix multiply adds ~3e-5 unitarity error.
                # On 2K=24-wire states with 10+ sequential events the
                # accumulated norm drift reaches ~1e-3, tripping NORM_TOL.
                # unitarize() corrects this without changing the architecture
                # (the projection error is O(eps32) and gradients flow through
                # torch.linalg.svd's autograd).
                cache[v] = sim.unitarize(raw)
            return cache[v]

        K = cs.K
        # 2K qubits: primaries 0..K-1, secondaries K..2K-1
        psi = sim.init_state(2 * K)

        for w, _noun, angle in cs.intro:
            # Step 1: Ry on primary wire (same as A2)
            psi = sim.apply_gate(psi, sim.ry(angle), (w,))
            # Step 2: XX coupling on (primary, secondary) -- the minimal lift
            coup_key = str(w)
            if coup_key in self.coupling:
                psi = sim.apply_gate(
                    psi, sim.xx(self.coupling[coup_key]), (w, K + w))
            # (if w >= K_MAX, the slot has no coupling; treated as zero)

        for v, wa, wp in cs.events:
            # Verb block on PRIMARY wires (wa, wp) -- identical to A2
            psi = sim.apply_gate(psi, block(v), (wa, wp))

        vseq, qa, qp = cs.question
        qw = (qa, qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)

        vals = []
        for eff in (self.q_pos, self.q_neg):
            phi = sim.apply_gate(psi, eff.matrix().conj().T, qw)
            # prob_all_zero_on traces out ALL wires except qw -- marginalises
            # over secondaries automatically
            vals.append(sim.prob_all_zero_on(phi, qw))
        return torch.stack(vals)

    def story_loss(self, cs):
        vals = self.story_values(cs)
        # exp79 SS9 numerical amendment: scale before log to prevent underflow
        scale = vals.detach().sum().clamp_min(1e-30)
        vals = vals / scale
        vals = vals.clamp_min(1e-6)
        target = torch.tensor([cs.answer])
        logits = torch.log(vals + 1e-12)
        loss = nn.functional.cross_entropy(logits.unsqueeze(0), target)
        pred = int(torch.argmax(vals).item())
        return loss, pred

    def predict(self, cs, verb_map=None):
        with torch.no_grad():
            return int(torch.argmax(self.story_values(cs, verb_map)).item())

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        readout = sum(p.numel() for p in self.q_pos.parameters()) + \
                  sum(p.numel() for p in self.q_neg.parameters())
        n_coupling = len(self.coupling)
        expected = 14 * len(self.verbs) + readout + n_coupling
        assert total == expected, \
            ("A2q2 param count: %d != 14*%d + %d(readout) + %d(coupling)"
             % (total, len(self.verbs), readout, n_coupling))
        return {"arm": "A2q2", "arm_label": "A2q2",
                "per_verb_A2_core": 14, "n_verbs": len(self.verbs),
                "readout_shared": readout, "n_coupling_params": n_coupling,
                "K_MAX": K_MAX, "total_trained": total,
                "A2q2_over_A2_ratio": round(total / A2_TOTAL_PARAMS, 4),
                "note": A2Q2_CONSTRUCTION_NOTE}


# ================================================================
# A2-WIDE model (capacity control)
# ================================================================
class A2WideModel(QuantumStoryModel):
    """A2 + K_MAX extra trained Ry angles at intro time (capacity control).

    Extra angles are indexed by wire SLOT (positional, matching A2q2's
    coupling index). Applied as an additional Ry on the PRIMARY wire after
    the embedding-based Ry, before any events.

    Total params: 14*n_verbs + 30 (readout) + K_MAX = EXACT MATCH with A2q2.

    When A2-WIDE also beats A2: the improvement reads as CAPACITY not
    architecture (the A2q2 advantage is not architecture-specific).
    See doc 73 §3 invalidation clause.
    """

    def __init__(self, verbs, seed):
        super().__init__("A2", verbs, seed)
        self.arm_label = "A2-WIDE"
        # K_MAX extra intro-slot angles (positional, initialized to 0)
        self.intro_extra = nn.ParameterDict(
            {str(i): nn.Parameter(torch.zeros(())) for i in range(K_MAX)})

    def story_values(self, cs, verb_map=None):
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)
            return cache[v]

        K = cs.K
        psi = sim.init_state(K)

        for w, _noun, angle in cs.intro:
            # Frozen embedding Ry (same as A2)
            psi = sim.apply_gate(psi, sim.ry(angle), (w,))
            # EXTRA trained Ry (A2-WIDE's extra capacity)
            extra_key = str(w)
            if extra_key in self.intro_extra:
                psi = sim.apply_gate(psi, sim.ry(self.intro_extra[extra_key]), (w,))

        for v, wa, wp in cs.events:
            psi = sim.apply_gate(psi, block(v), (wa, wp))

        vseq, qa, qp = cs.question
        qw = (qa, qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)

        vals = []
        for eff in (self.q_pos, self.q_neg):
            phi = sim.apply_gate(psi, eff.matrix().conj().T, qw)
            vals.append(sim.prob_all_zero_on(phi, qw))
        return torch.stack(vals)

    def story_loss(self, cs):
        vals = self.story_values(cs)
        # exp79 SS9 numerical amendment
        scale = vals.detach().sum().clamp_min(1e-30)
        vals = vals / scale
        vals = vals.clamp_min(1e-6)
        target = torch.tensor([cs.answer])
        logits = torch.log(vals + 1e-12)
        loss = nn.functional.cross_entropy(logits.unsqueeze(0), target)
        pred = int(torch.argmax(vals).item())
        return loss, pred

    def predict(self, cs, verb_map=None):
        with torch.no_grad():
            return int(torch.argmax(self.story_values(cs, verb_map)).item())

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        readout = sum(p.numel() for p in self.q_pos.parameters()) + \
                  sum(p.numel() for p in self.q_neg.parameters())
        n_extra = len(self.intro_extra)
        expected = 14 * len(self.verbs) + readout + n_extra
        assert total == expected, \
            ("A2-WIDE param count: %d != 14*%d + %d(readout) + %d(extra_intro)"
             % (total, len(self.verbs), readout, n_extra))
        return {"arm": "A2-WIDE", "arm_label": "A2-WIDE",
                "per_verb": 14, "n_verbs": len(self.verbs),
                "readout_shared": readout, "n_extra_intro_angles": n_extra,
                "K_MAX": K_MAX, "total_trained": total,
                "A2WIDE_over_A2_ratio": round(total / A2_TOTAL_PARAMS, 4),
                "note": A2WIDE_CONSTRUCTION_NOTE}


# ================================================================
# A2-RETRAIN model (same as A2 but retrained under exp81's harness)
# ================================================================
class A2RetrainModel(QuantumStoryModel):
    """A2 retrained under the exp81 harness (same-draw arms for pairing).

    Uses exp81's own stream seed (81_000_000 family) for init, NOT exp42r's
    stream. Param count is identical to committed A2 (240).
    """

    def __init__(self, verbs, seed):
        super().__init__("A2", verbs, seed)
        self.arm_label = "A2-RETRAIN"

    def story_loss(self, cs):
        vals = self.story_values(cs)
        # exp79 SS9 numerical amendment (applied consistently to all arms)
        scale = vals.detach().sum().clamp_min(1e-30)
        vals = vals / scale
        vals = vals.clamp_min(1e-6)
        target = torch.tensor([cs.answer])
        logits = torch.log(vals + 1e-12)
        loss = nn.functional.cross_entropy(logits.unsqueeze(0), target)
        pred = int(torch.argmax(vals).item())
        return loss, pred

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        readout = sum(p.numel() for p in self.q_pos.parameters()) + \
                  sum(p.numel() for p in self.q_neg.parameters())
        assert total == 14 * len(self.verbs) + readout, \
            "A2-RETRAIN param count drifted: %d" % total
        return {"arm": "A2-RETRAIN", "per_verb": 14, "n_verbs": len(self.verbs),
                "readout_shared": readout, "total_trained": total,
                "note": "A2 retrained under exp81 harness (paired test baseline)"}


# ================================================================
# init for exp81 arms
# ================================================================
def init_stream_seed81(arm, seed):
    """Dedicated stream per (arm, seed); disjoint from exp42r and exp79."""
    return EXP81_STREAM_BASE + 1000 * EXP81_ARM_INDEX[arm] + seed


def apply_init81(model, arm, seed):
    """All params ~ N(0, 0.1^2) from the exp81 stream.

    Implements doc 62 §24.3 bounded rejection loop: accept only when the
    drawn vector's sd falls in SD_WINDOW=[0.08, 0.12].  Effective stream seed
    for attempt k = base_stream + 100_000*k.  Exhaustion (k=MAX_INIT_DRAWS)
    raises RuntimeError (arm-level STOP, not a window widening).

    Indexed by sorted(named_parameters()) for reproducibility.
    Returns the recordable init spec including sha of the drawn vector.
    """
    # Floor arm uses A2q2's slot in the index (both use A2q2Model)
    stream_arm = "A2q2" if arm == "A2q2-FLOOR" else arm
    if stream_arm not in EXP81_ARM_INDEX:
        raise ValueError("Unknown exp81 arm for stream: %r" % arm)
    base_stream = init_stream_seed81(stream_arm, seed)

    refused_sds = []
    for attempt in range(MAX_INIT_DRAWS):
        stream = base_stream + 100_000 * attempt
        g = torch.Generator().manual_seed(stream)
        with torch.no_grad():
            for _name, p in sorted(model.named_parameters()):
                p.copy_(torch.randn(p.shape, generator=g) * INIT_STD)
            flat = torch.cat([p.detach().reshape(-1).to(torch.float64)
                              for _n, p in sorted(model.named_parameters())])
            actual_sd = float(flat.std())
        if SD_WINDOW[0] <= actual_sd <= SD_WINDOW[1]:
            with torch.no_grad():
                assert torch.isfinite(flat).all(), \
                    "non-finite init %s s%d attempt %d" % (arm, seed, attempt)
                h = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
            return {
                "spec": ("exp81 init: N(0,%.2f^2), sd-window [%.2f,%.2f] "
                         "(doc 62 §24.3)" % (INIT_STD, SD_WINDOW[0], SD_WINDOW[1])),
                "arm": arm, "seed": seed,
                "accepted_stream": stream,
                "accepted_sd": round(actual_sd, 6),
                "attempts": attempt + 1,
                "refused_sds": refused_sds,
                "stream_rule": ("81_000_000 + 1000*EXP81_ARM_INDEX[%r] + seed "
                                "+ 100_000*attempt" % stream_arm),
                "n_params": int(flat.numel()), "init_vector_sha256": h,
            }
        refused_sds.append(round(actual_sd, 6))

    raise RuntimeError(
        "apply_init81: exhausted %d draws for %s s%d; refused sds: %s"
        % (MAX_INIT_DRAWS, arm, seed, refused_sds))


def make_model81(arm, verbs):
    """Construct model for an exp81 arm (no init applied yet)."""
    if arm in ("A2q2", "A2q2-FLOOR"):        # floor uses same architecture
        return A2q2Model(verbs, seed=0)       # seed arg unused; init via apply_init81
    if arm == "A2-WIDE":
        return A2WideModel(verbs, seed=0)
    if arm == "A2-RETRAIN":
        return A2RetrainModel(verbs, seed=0)
    raise ValueError("unknown exp81 arm: %r" % arm)


def build_model81(arm, verbs, seed):
    """Construct + apply registered init. The ONE way a model is built."""
    m = make_model81(arm, verbs)
    spec = apply_init81(m, arm, seed)
    return m, spec


# ================================================================
# unit paths for exp81
# ================================================================
def unit_paths81(arm, seed, ckpt_dir=CKPT_DIR):
    tag = arm.replace("-", "").replace(" ", "")
    return (os.path.join(ckpt_dir, "%s_s%d.json" % (tag, seed)),
            os.path.join(ckpt_dir, "%s_s%d.pt" % (tag, seed)))


def failed_path81(job_id, ctl_dir=CTL_DIR):
    return os.path.join(ctl_dir, "FAILED_%s.json" % job_id)


def settled81(outputs, job_id):
    return (all(os.path.exists(p) for p in outputs)
            or os.path.exists(failed_path81(job_id)))


def record_failure81(job_id, exc, tag):
    payload = {"job": job_id, "tag": tag, "error": repr(exc),
               "traceback": traceback.format_exc(),
               "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "isolation": "arm-level failure; worker continues (doc 62 SS24)"}
    with open(failed_path81(job_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("[FAIL] %s: %r (isolated)" % (job_id, exc), flush=True)


# ================================================================
# fixed scramble (floor, doc 66 SS12.2 / doc 70 B3)
# ================================================================
def draw_fixed_scramble(union_item_ids, seed=FLOOR_SCRAMBLE_SEED):
    """ONE scramble drawn before training. Same scramble for all seeds."""
    rng = random.Random(seed)
    ids = list(union_item_ids)
    labels = [0] * (len(ids) // 2) + [1] * (len(ids) - len(ids) // 2)
    rng.shuffle(labels)
    return {iid: lab for iid, lab in zip(ids, labels)}


def evaluate_scrambled(model, compiled, scramble_map):
    """Evaluate model on compiled set with SCRAMBLED labels."""
    correct = []
    with torch.no_grad():
        for cs in compiled:
            pred = model.predict(cs)
            ans = scramble_map.get(cs.item_id, cs.answer)
            correct.append(1 if pred == ans else 0)
    return sum(correct) / len(correct), correct


# ================================================================
# training
# ================================================================
def train_unit81(arm, seed, data, cfg,
                 scramble_map=None,
                 splits=("test_a", "test_b", "test_c")):
    """One (arm, seed) training unit. Returns (model, detail).

    If scramble_map is given (A2q2-FLOOR): trains AND evaluates on scrambled
    labels (doc 71 SS6 / doc 70 B3 -- eval-only scrambling is vacuous).
    """
    t0 = time.time()
    model, init_spec = build_model81(arm, cfg["verbs"], seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = random.Random(9000 + seed)

    # Record coupling init angles for gate_met tracking (A2q2 arms only)
    coupling_init = {}
    if arm in ("A2q2", "A2q2-FLOOR"):
        with torch.no_grad():
            for k, v in model.coupling.items():
                coupling_init[k] = float(v.item())
    train = data["compiled"]["train"]
    val = data["compiled"]["val"]

    if scramble_map is not None:
        import copy as _copy
        def _scrambled(lst):
            out = []
            for cs in lst:
                c2 = _copy.copy(cs)
                c2.answer = scramble_map.get(cs.item_id, cs.answer)
                out.append(c2)
            return out
        train, val = _scrambled(train), _scrambled(val)

    best = {"val_acc": -1.0, "train_acc": None, "epoch": -1, "state": None}
    nonfinite_skips = []
    losses_e, gnorms_e, runacc_e, val_e = [], [], [], []
    first_grad_ok = False

    for epoch in range(1, cfg["epochs"] + 1):
        te = time.time()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses, gnorms, run_correct = [], [], 0
        for start in range(0, len(order), cfg["batch"]):
            batch = [train[i] for i in order[start:start + cfg["batch"]]]
            opt.zero_grad()
            bl = torch.tensor(0.0)
            for cs in batch:
                loss, pred = model.story_loss(cs)
                bl = bl + loss
                if pred == cs.answer:
                    run_correct += 1
            bl = bl / len(batch)
            bl.backward()
            tot = sum(p.grad.norm().item() ** 2 for p in model.parameters()
                      if p.grad is not None)
            gn = math.sqrt(tot)
            if not math.isfinite(gn):
                nonfinite_skips.append({"epoch": epoch, "batch_start": start})
                opt.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not first_grad_ok:
                if gn <= 0.0:
                    raise RuntimeError("zero gradient on first batch (%s s%d)"
                                       % (arm, seed))
                first_grad_ok = True
            gnorms.append(gn)
            opt.step()
            losses.append(bl.item())

        val_acc, _ = evaluate(model, val)
        losses_e.append(sum(losses) / max(len(losses), 1))
        gnorms_e.append(sum(gnorms) / max(len(gnorms), 1))
        runacc_e.append(run_correct / len(train))
        val_e.append(val_acc)
        if val_acc > best["val_acc"]:
            ctr, _ = evaluate(model, train)
            best = {"val_acc": val_acc, "train_acc": ctr, "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
        print("[%s s%d] epoch %2d/%d loss=%.4f train=%.3f val=%.3f "
              "gn=%.3e best@%d %.1fs"
              % (arm, seed, epoch, cfg["epochs"],
                 losses_e[-1], runacc_e[-1], val_acc,
                 gnorms_e[-1], best["epoch"], time.time() - te), flush=True)
        if best["val_acc"] >= 1.0 and best["train_acc"] >= 1.0:
            print("[%s s%d] early stop" % (arm, seed), flush=True)
            break

    model.load_state_dict(best["state"])

    # Coupling final state + per-unit gate_met (doc 73 §3: engagement criterion)
    coupling_detail = {}
    if arm in ("A2q2", "A2q2-FLOOR") and coupling_init:
        with torch.no_grad():
            coupling_final = {k: float(v.item()) for k, v in model.coupling.items()}
        coupling_delta = {k: coupling_final[k] - coupling_init[k]
                         for k in coupling_init}
        max_delta_coupling = max(abs(d) for d in coupling_delta.values()) if coupling_delta else 0.0
        loss_decreased = (losses_e[-1] < losses_e[0]) if len(losses_e) >= 2 else False
        gate_met_unit = bool(loss_decreased and max_delta_coupling >= 0.10)
        coupling_detail = {
            "coupling_init": coupling_init,
            "coupling_final": coupling_final,
            "coupling_delta": coupling_delta,
            "max_delta_coupling_rad": round(max_delta_coupling, 6),
            "loss_decreased": loss_decreased,
            "gate_met_unit": gate_met_unit,
            "gate_met_rule": ("loss_decreased AND max|Δcoupling| >= 0.10 rad "
                              "(doc 73 §3 engagement criterion)"),
        }

    detail = {
        "arm": arm, "arm_label": getattr(model, "arm_label", arm),
        "seed": seed, "best_epoch": best["epoch"],
        "epochs_run": len(losses_e), "max_epochs": cfg["epochs"],
        "train_acc_at_best": best["train_acc"], "val_acc": best["val_acc"],
        "epoch_mean_losses": losses_e, "epoch_mean_grad_norms": gnorms_e,
        "epoch_running_train_acc": runacc_e, "epoch_val_acc": val_e,
        "init": init_spec, "param_counts": model.param_counts(),
        "model_selection": "in-distribution val split ONLY (doc 22 SS4)",
        "test": {}, "wall_time_sec": round(time.time() - t0, 1),
        "nonfinite_skips": nonfinite_skips,
        "construction_note": A2Q2_CONSTRUCTION_NOTE if arm in ("A2q2", "A2q2-FLOOR")
                             else A2WIDE_CONSTRUCTION_NOTE if arm == "A2-WIDE"
                             else "A2 retrained under exp81 harness",
    }
    detail.update(coupling_detail)
    for sp in splits:
        if scramble_map is not None:
            acc, corr = evaluate_scrambled(model, data["compiled"][sp],
                                           scramble_map)
        else:
            acc, corr = evaluate(model, data["compiled"][sp])
        detail["test"][sp] = {"accuracy": acc, "n": len(corr),
                              "per_item_correct": corr,
                              "item_ids": [cs.item_id
                                           for cs in data["compiled"][sp]],
                              "scrambled": scramble_map is not None}
    if scramble_map is not None:
        ua, uc = evaluate_scrambled(model, data["compiled"]["test_union"],
                                    scramble_map)
    else:
        ua, uc = evaluate(model, data["compiled"]["test_union"])
    detail["test"]["test_union"] = {"accuracy": ua, "n": len(uc),
                                    "scrambled": scramble_map is not None}
    detail["union_item_ids"] = data["union_ids"]
    detail["union_per_item_correct"] = uc
    detail["vocab_sha256"] = K42.vocab_sha(data["vocab"])
    detail["dataset_sha256"] = data["prov"]["dataset"]["sha256"]
    detail["embeddings_sha256"] = data["emb_sha"]
    return model, detail


def job_main81(arm, seed, data, cfg, scramble_map=None):
    ck_json, ck_pt = unit_paths81(arm, seed)
    model, detail = train_unit81(arm, seed, data, cfg, scramble_map=scramble_map)
    # Flush any renorm incidents accumulated during train/eval
    n_renorm = sim.flush_renorm_log()
    if n_renorm:
        detail["renorm_incidents_flushed"] = n_renorm
    torch.save(model.state_dict(), ck_pt)
    with open(ck_json, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False)
    print("[unit] %s s%d done: val=%.3f union=%.3f (%.0fs) renorm_incidents=%d"
          % (arm, seed, detail["val_acc"],
             detail["test"]["test_union"]["accuracy"],
             detail["wall_time_sec"], n_renorm), flush=True)


# ================================================================
# drift anchor (A2 committed checkpoint re-eval)
# ================================================================
def run_drift_anchor(data):
    """Bit-exact re-eval of committed A2 seed 0 checkpoint."""
    ck_json = os.path.join(EXP42R_CKPT_DIR, "A2_s0.json")
    ck_pt = os.path.join(EXP42R_CKPT_DIR, "A2_s0.pt")
    if not (os.path.exists(ck_json) and os.path.exists(ck_pt)):
        raise RuntimeError("drift anchor: exp42r_ckpt/A2_s0.{json,pt} absent")
    with open(ck_json, encoding="utf-8") as f:
        committed = json.load(f)
    model, _ = K42.build_model("A2", data["verbs"], data["vocab"],
                                0, data["angles"])
    model.load_state_dict(torch.load(ck_pt, weights_only=True))
    model.eval()
    TOL = 1e-4
    results = {}
    all_ok = True
    for sp in ("test_a", "test_b", "test_c", "test_union"):
        acc, _ = evaluate(model, data["compiled"][sp])
        committed_acc = committed["test"][sp]["accuracy"]
        delta = abs(acc - committed_acc)
        ok = delta < TOL
        results[sp] = {"accuracy": acc, "committed_accuracy": committed_acc,
                       "delta": delta, "within_tol": ok}
        if not ok:
            all_ok = False
    anchor = {"component": "exp81 drift anchor (A2 committed checkpoint)",
              "re_eval_results": results,
              "all_within_tol": all_ok, "tol": TOL,
              "verdict": "PASS" if all_ok else "DRIFT DETECTED",
              "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not all_ok:
        raise RuntimeError("Drift anchor FAILED -- A2 baseline is unreliable")
    return anchor


# ================================================================
# SELF-EMITTING GATE ARTIFACTS (doc 70 SS3.6)
# ================================================================
def _artifact_hash(d):
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---- SG-ZC: zero-coupling gate ------------------------------------
SG_ZC_OUT = "exp81_gate_SG_ZC.json"


def gate_zero_coupling(data):
    """SG-ZC: when coupling=0, XX(0)=I, secondaries stay |0>, and
    A2q2's story_values == A2's story_values (max |delta| < 1e-5).

    PASS assertion: zero-coupling A2q2 == A2 (max |delta| < 1e-5).
    DESIGNED-TO-FAIL: random-coupling A2q2 == A2 (shown failing).

    This is the exact analogue of exp79's SG-ANT gate for the A2q2 architecture.
    """
    verbs = data["verbs"]
    compiled = data["compiled"]["test_union"][:20]

    # A2 model (K wires, committed)
    m_a2, _ = K42.build_model("A2", verbs, data["vocab"], 0, data["angles"])
    m_a2.eval()

    # A2q2 with coupling ZEROED (designed to equal A2)
    m_zero, _ = build_model81("A2q2", verbs, 0)
    with torch.no_grad():
        for k in m_zero.coupling:
            m_zero.coupling[k].fill_(0.0)
        # Copy A2's weights (core, dress, q_pos, q_neg)
        a2_sd = m_a2.state_dict()
        mz_sd = m_zero.state_dict()
        for k in a2_sd:
            if k in mz_sd:
                mz_sd[k] = a2_sd[k].clone()
        # Keep coupling at zero (don't overwrite with A2 weights -- A2 has no coupling)
        m_zero.load_state_dict(mz_sd)
    m_zero.eval()

    # A2q2 with RANDOM coupling (designed to diverge from A2)
    m_rand, spec_rand = build_model81("A2q2", verbs, 42)
    with torch.no_grad():
        # Copy A2 weights but keep coupling from random init
        m_rand_sd = m_rand.state_dict()
        for k in a2_sd:
            if k in m_rand_sd and "coupling" not in k:
                m_rand_sd[k] = a2_sd[k].clone()
        m_rand.load_state_dict(m_rand_sd)
    m_rand.eval()

    max_delta_zero = 0.0
    max_delta_rand = 0.0
    with torch.no_grad():
        for cs in compiled:
            a2_vals = m_a2.story_values(cs)
            zero_vals = m_zero.story_values(cs)
            rand_vals = m_rand.story_values(cs)
            max_delta_zero = max(max_delta_zero,
                                 float((zero_vals - a2_vals).abs().max()))
            max_delta_rand = max(max_delta_rand,
                                 float((rand_vals - a2_vals).abs().max()))

    zero_ok = max_delta_zero < PASS_TOL
    rand_differs = max_delta_rand > PASS_TOL

    dtf_assertion = "random_coupling A2q2 == A2 (max |delta| < 1e-5)"
    dtf_passes = not rand_differs   # expected False

    artifact = {
        "gate": "SG-ZC zero-coupling (doc 73 §2, doc 70 SS3.6)",
        "pass_assertion": "zero-coupling A2q2 == A2 (max |delta| < 1e-5)",
        "pass_result": zero_ok,
        "designed_to_fail_assertion": dtf_assertion,
        "designed_to_fail_result": dtf_passes,
        "designed_to_fail_shown_failing": not dtf_passes,
        "max_delta_zero_coupling_vs_A2": max_delta_zero,
        "max_delta_random_coupling_vs_A2": max_delta_rand,
        "pass_tol": PASS_TOL,
        "n_items": len(compiled),
        "mechanism": ("XX(0)=I (IsingXX at phi=0 is identity); secondaries "
                      "stay in |0>; prob_all_zero_on on (qa,qp) marginalises "
                      "them out, recovering A2's K-wire story_values exactly"),
        "passed": bool(zero_ok and rand_differs),
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    artifact["artifact_sha256"] = _artifact_hash(
        {k: v for k, v in artifact.items() if k != "artifact_sha256"})
    with open(SG_ZC_OUT, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=1)
    print("[SG-ZC] zero-coup delta=%.2e rand-coup delta=%.2e "
          "passed=%s dtf_shown_failing=%s"
          % (max_delta_zero, max_delta_rand,
             artifact["passed"], not dtf_passes), flush=True)
    return artifact


# ---- SG-FS: fixed-scramble gate -----------------------------------
SG_FS_OUT = "exp81_gate_SG_FS.json"


def gate_fixed_scramble(data):
    """SG-FS: ONE scramble drawn before training (FLOOR_SCRAMBLE_SEED=81999).
    Train A2q2 for 5 epochs on tiny subset, evaluate on scrambled labels.
    PASS: scrambled acc within 2 binomial SD of 0.5.
    DESIGNED-TO-FAIL: scrambled acc >= 0.70 (shown failing).
    """
    verbs = data["verbs"]
    rng = random.Random(0)
    pool = data["compiled"]["train"][:]
    rng.shuffle(pool)
    tiny = pool[:40]

    all_ids = sorted({cs.item_id for split in data["compiled"].values()
                      for cs in split})
    scramble_map = draw_fixed_scramble(all_ids, FLOOR_SCRAMBLE_SEED)
    scramble_sha = hashlib.sha256(
        json.dumps(scramble_map, sort_keys=True).encode()).hexdigest()

    # Train tiny A2q2 for 5 epochs on scrambled labels (doc 71 SS6 / doc 70 B3:
    # training on scrambled is mandatory; eval-only scrambling is vacuous)
    import copy as _copy
    def _scrambled(lst):
        out = []
        for cs in lst:
            c2 = _copy.copy(cs)
            c2.answer = scramble_map.get(cs.item_id, cs.answer)
            out.append(c2)
        return out

    model, init_spec = build_model81("A2q2", verbs, 0)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    for _ in range(5):
        for cs in _scrambled(tiny):
            opt.zero_grad()
            loss, _ = model.story_loss(cs)
            loss.backward()
            opt.step()

    eval_union = data["compiled"]["test_union"][:min(100, len(all_ids))]
    scr_acc, scr_corr = evaluate_scrambled(model, eval_union, scramble_map)
    real_acc, _ = evaluate(model, eval_union)

    n = len(scr_corr)
    sd_binom = math.sqrt(0.5 * 0.5 / n)
    within_2sd = abs(scr_acc - 0.5) <= 2 * sd_binom
    floor_admissible = within_2sd

    dtf_assertion = "scrambled_acc >= 0.70 (should fail)"
    dtf_passes = bool(scr_acc >= 0.70)

    artifact = {
        "gate": "SG-FS fixed-scramble floor (doc 66 SS12.2, doc 70 B3)",
        "scramble_seed": FLOOR_SCRAMBLE_SEED,
        "scramble_sha256": scramble_sha,
        "n_eval_items": n,
        "scrambled_accuracy": scr_acc,
        "real_accuracy": real_acc,
        "binomial_sd": sd_binom,
        "within_2sd_of_chance": within_2sd,
        "floor_admissible": floor_admissible,
        "pass_criterion": "scrambled_acc within 2 binomial SD of 0.5",
        "designed_to_fail_assertion": dtf_assertion,
        "designed_to_fail_result": dtf_passes,
        "designed_to_fail_shown_failing": not dtf_passes,
        "passed": floor_admissible,
        "inputs": {"n_train_items": len(tiny), "train_epochs": 5,
                   "arm": "A2q2", "init_spec": init_spec},
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    artifact["artifact_sha256"] = _artifact_hash(
        {k: v for k, v in artifact.items() if k != "artifact_sha256"})
    with open(SG_FS_OUT, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=1)
    print("[SG-FS] scrambled_acc=%.3f within_2sd=%s floor_admissible=%s "
          "dtf_shown_failing=%s"
          % (scr_acc, within_2sd, floor_admissible, not dtf_passes), flush=True)
    return artifact


# ---- SG-PC: param-count gate --------------------------------------
SG_PC_OUT = "exp81_gate_SG_PC.json"


def gate_param_count(data):
    """SG-PC: A2q2 total == A2-WIDE total (EXACT MATCH by construction).

    PASS assertion: A2q2 total params == A2-WIDE total params (exact).
    DESIGNED-TO-FAIL: A2q2 total == A2 total (shown failing; A2q2 > A2 by K_MAX).

    Records full param accounting for the build addendum.
    """
    verbs = data["verbs"]

    m_a2q2, _ = build_model81("A2q2", verbs, 0)
    m_wide, _ = build_model81("A2-WIDE", verbs, 0)
    m_a2, _ = K42.build_model("A2", verbs, data["vocab"], 0, data["angles"])

    a2q2_total = sum(p.numel() for p in m_a2q2.parameters())
    wide_total = sum(p.numel() for p in m_wide.parameters())
    a2_total = sum(p.numel() for p in m_a2.parameters())

    pc_a2q2 = m_a2q2.param_counts()
    pc_wide = m_wide.param_counts()
    pc_a2 = m_a2.param_counts()

    exact_match = (a2q2_total == wide_total)
    a2q2_over_a2 = (a2q2_total == a2_total)  # DTF: designed to fail (a2q2 > a2)

    dtf_assertion = "A2q2 total == A2 total (should fail; A2q2 has K_MAX extra params)"
    dtf_passes = a2q2_over_a2   # expected False

    artifact = {
        "gate": "SG-PC param-count (doc 73 §2 build addendum)",
        "pass_assertion": "A2q2 total == A2-WIDE total (exact match by construction)",
        "pass_result": exact_match,
        "designed_to_fail_assertion": dtf_assertion,
        "designed_to_fail_result": dtf_passes,
        "designed_to_fail_shown_failing": not dtf_passes,
        "K_MAX": K_MAX,
        "n_verbs": len(verbs),
        "param_accounting": {
            "A2": {"total": a2_total, "detail": pc_a2},
            "A2q2": {"total": a2q2_total, "detail": pc_a2q2,
                     "extra_over_A2": a2q2_total - a2_total},
            "A2-WIDE": {"total": wide_total, "detail": pc_wide,
                        "extra_over_A2": wide_total - a2_total},
        },
        "ratios": {
            "A2q2_over_A2": round(a2q2_total / a2_total, 4),
            "A2WIDE_over_A2": round(wide_total / a2_total, 4),
            "A2WIDE_over_A2q2": round(wide_total / a2q2_total, 4),
        },
        "passed": bool(exact_match),
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    artifact["artifact_sha256"] = _artifact_hash(
        {k: v for k, v in artifact.items() if k != "artifact_sha256"})
    with open(SG_PC_OUT, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=1)
    print("[SG-PC] A2q2=%d A2-WIDE=%d A2=%d match=%s dtf_shown_failing=%s"
          % (a2q2_total, wide_total, a2_total, exact_match, not dtf_passes),
          flush=True)
    return artifact


# ---- M4: max-K forward pass (K=10-12, the test_b regime) ----------
def gate_max_k_forward(data):
    """M4: one A2q2 forward pass on the max-K test_b item.

    Validates the 2K-wire path (up to 2^24 amplitudes at K=12) does not
    OOM or produce non-finite values.  Catches state-vector blowup BEFORE
    30 epochs of training would discover it.
    Returns (max_k, vals, passed).
    """
    test_b = data["compiled"].get("test_b", [])
    if not test_b:
        raise RuntimeError("M4: test_b empty -- cannot check max-K path")
    max_item = max(test_b, key=lambda cs: cs.K)
    verbs = data["verbs"]
    m, _ = build_model81("A2q2", verbs, 0)
    with torch.no_grad():
        for k in m.coupling:
            m.coupling[k].fill_(0.0)
        m.eval()
        vals = m.story_values(max_item)
    finite = bool(torch.isfinite(vals).all())
    nonzero = bool(float(vals.sum()) > 0.0)
    return max_item.K, float(vals[0]), float(vals[1]), (finite and nonzero)


# ================================================================
# smoke battery (exp81)
# ================================================================
def smoke_battery(data):
    """Run all 3 self-emitting gate artifacts + mechanical checks.
    Writes exp81_smoke_PASSED.json on success.
    """
    t0 = time.time()
    tests = []

    def add(name, passed, **kw):
        rec = {"test": name, "passed": bool(passed)}
        rec.update(kw)
        tests.append(rec)
        print("[smoke] %-50s %s" % (name, "PASS" if passed else "FAIL"),
              flush=True)

    verbs = data["verbs"]

    # --- M1: mechanical build + gradient check ----------------------
    rng = random.Random(0)
    pool = data["compiled"]["train"][:]
    rng.shuffle(pool)
    tiny = pool[:20]
    mech = {}
    ok_mech = True
    for arm in ("A2q2", "A2-WIDE", "A2-RETRAIN"):
        m, _ = build_model81(arm, verbs, 0)
        opt = torch.optim.Adam(m.parameters(), lr=0.005)
        losses = []
        gnorms = []
        for _ in range(3):
            ep_loss, ep_gn = 0.0, 0.0
            for cs in tiny:
                opt.zero_grad()
                loss, _ = m.story_loss(cs)
                loss.backward()
                gn = math.sqrt(sum(p.grad.norm().item() ** 2
                                   for p in m.parameters()
                                   if p.grad is not None))
                ep_loss += loss.item()
                ep_gn += gn
                opt.step()
            losses.append(ep_loss / len(tiny))
            gnorms.append(ep_gn / len(tiny))
        arm_ok = (all(math.isfinite(g) and g > 0 for g in gnorms)
                  and losses[-1] < losses[0])
        mech[arm] = {"losses": losses, "grad_norms": gnorms, "passed": arm_ok}
        ok_mech = ok_mech and arm_ok
    add("M1 mechanical build + gradient (all 3 arms)", ok_mech, per_arm=mech)

    # --- M2: zero-coupling A2q2 == A2 (inline check) ---------------
    m_zero, _ = build_model81("A2q2", verbs, 0)
    m_a2, _ = K42.build_model("A2", verbs, data["vocab"], 0, data["angles"])
    with torch.no_grad():
        for k in m_zero.coupling:
            m_zero.coupling[k].fill_(0.0)
        a2_sd = m_a2.state_dict()
        mz_sd = m_zero.state_dict()
        for k in a2_sd:
            if k in mz_sd:
                mz_sd[k] = a2_sd[k].clone()
        m_zero.load_state_dict(mz_sd)
        m_zero.eval()
        m_a2.eval()
        max_delta = 0.0
        for cs in data["compiled"]["test_union"][:20]:
            a2v = m_a2.story_values(cs)
            q2v = m_zero.story_values(cs)
            max_delta = max(max_delta, float((a2v - q2v).abs().max()))
    ok_m2 = max_delta < PASS_TOL
    add("M2 zero-coupling A2q2 == A2 (reduction check)",
        ok_m2, max_abs_delta=max_delta, tol=PASS_TOL)

    # --- M3: param count A2q2 == A2-WIDE ----------------------------
    m_q2, _ = build_model81("A2q2", verbs, 0)
    m_wd, _ = build_model81("A2-WIDE", verbs, 0)
    q2_total = sum(p.numel() for p in m_q2.parameters())
    wd_total = sum(p.numel() for p in m_wd.parameters())
    ok_m3 = (q2_total == wd_total)
    add("M3 A2q2 param count == A2-WIDE param count",
        ok_m3, A2q2_total=q2_total, A2WIDE_total=wd_total)

    # --- M4: max-K forward pass (K=10-12, 2^(2K) amplitudes) --------
    try:
        k_max, v0, v1, ok_m4 = gate_max_k_forward(data)
        add("M4 A2q2 max-K forward (K=%d, 2^%d amplitudes)" % (k_max, 2 * k_max),
            ok_m4, K=k_max, n_amplitudes=2 ** (2 * k_max),
            val_pos=round(v0, 6), val_neg=round(v1, 6))
    except Exception as exc:
        add("M4 A2q2 max-K forward", False, error=str(exc))
        ok_m4 = False

    # --- SG-ZC: zero-coupling gate (self-emitting) ------------------
    zc = gate_zero_coupling(data)
    add("SG-ZC zero-coupling A2q2 == A2 (self-emitting)", zc["passed"],
        zero_coup_delta=zc["max_delta_zero_coupling_vs_A2"],
        rand_coup_delta=zc["max_delta_random_coupling_vs_A2"],
        dtf_shown_failing=zc["designed_to_fail_shown_failing"],
        artifact=SG_ZC_OUT)

    # --- SG-FS: fixed-scramble floor (self-emitting) ----------------
    fs = gate_fixed_scramble(data)
    add("SG-FS fixed-scramble floor (self-emitting)", fs["passed"],
        floor_admissible=fs["floor_admissible"],
        scrambled_acc=fs["scrambled_accuracy"],
        dtf_shown_failing=fs["designed_to_fail_shown_failing"],
        artifact=SG_FS_OUT)

    # --- SG-PC: param-count gate (self-emitting) --------------------
    pc = gate_param_count(data)
    add("SG-PC param-count A2q2 == A2-WIDE (self-emitting)", pc["passed"],
        A2q2_total=pc["param_accounting"]["A2q2"]["total"],
        A2WIDE_total=pc["param_accounting"]["A2-WIDE"]["total"],
        ratios=pc["ratios"],
        dtf_shown_failing=pc["designed_to_fail_shown_failing"],
        artifact=SG_PC_OUT)

    # --- summary ----------------------------------------------------
    all_ok = all(t["passed"] for t in tests)
    report = {
        "component": "exp81 smoke battery (doc 73 §2, doc 70 SS3.6)",
        "tests": tests, "all_passed": all_ok,
        "gate_artifacts": [SG_ZC_OUT, SG_FS_OUT, SG_PC_OUT],
        "designed_to_fail_builds": [
            "SG-ZC random-coupling A2q2 == A2 (shown failing)",
            "SG-FS scrambled_acc >= 0.70 (shown failing)",
            "SG-PC A2q2 total == A2 total (shown failing; A2q2 > A2)",
            "M4 zero-coupling used so no new entanglement (state finite, not DTF)"],
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "provenance": data["prov"],
        "wall_time_sec": round(time.time() - t0, 1),
    }
    with open(SMOKE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("\n[smoke] %d/%d passed -- %s (%.0fs)"
          % (sum(t["passed"] for t in tests), len(tests),
             "SMOKE GATE OPEN" if all_ok else "SMOKE GATE CLOSED",
             report["wall_time_sec"]))
    if all_ok:
        with open(SMOKE_MARKER, "w", encoding="utf-8") as f:
            json.dump({"all_passed": True, "report": SMOKE_REPORT,
                       "when": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime())}, f, indent=1)
        print("[smoke] wrote %s -- workers may train." % SMOKE_MARKER)
        return 0
    print("[smoke] NO marker written.")
    return 1


# ================================================================
# first-epoch timing (doc 73 §3, derived-not-guessed clause)
# ================================================================
def measure_epoch_time(data, cfg, n_train=50):
    """Measure first-epoch wall time for A2q2 and A2-WIDE on n_train items.

    Baseline: A2 at 0.00878 s/item/epoch (exp79_epoch_timing.json).
    A2q2 at K: state is 2^(2K) vs A2's 2^K -- expected 16x at K=4, 256x at K=8.
    """
    verbs = cfg["verbs"]
    pool = list(data["compiled"]["train"])
    random.Random(81).shuffle(pool)
    tiny = pool[:n_train]

    # K distribution of the timing slice
    k_counts = {}
    for cs in tiny:
        k_counts[cs.K] = k_counts.get(cs.K, 0) + 1
    k_dist = sorted(k_counts.items())

    results = {}
    for arm in ("A2q2", "A2-WIDE"):
        m, _ = build_model81(arm, verbs, 0)
        opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
        # Warm-up (1 item) to avoid cold-start
        cs_warm = tiny[0]
        opt.zero_grad()
        loss_w, _ = m.story_loss(cs_warm)
        loss_w.backward()
        opt.step()
        # Timed run: 3 epochs on n_train items
        N_TIMED_EPOCHS = 3
        t0 = time.time()
        for _ep in range(N_TIMED_EPOCHS):
            for cs in tiny:
                opt.zero_grad()
                loss, _ = m.story_loss(cs)
                loss.backward()
                opt.step()
        elapsed = time.time() - t0
        sec_per_item_per_epoch = elapsed / (n_train * N_TIMED_EPOCHS)
        results[arm] = {
            "n_items_measured": n_train,
            "n_timed_epochs": N_TIMED_EPOCHS,
            "wall_sec_total": round(elapsed, 2),
            "sec_per_item_per_epoch": round(sec_per_item_per_epoch, 5),
        }
        print("[timing] %s: %.2fs for %d items × %d epochs = %.5f s/item/epoch"
              % (arm, elapsed, n_train, N_TIMED_EPOCHS, sec_per_item_per_epoch),
              flush=True)

    # A2 baseline from committed timing file
    a2_baseline = {}
    timing_file = "exp79_epoch_timing.json"
    if os.path.exists(timing_file):
        with open(timing_file) as f:
            t79 = json.load(f)
        a2_baseline = t79.get("A2_committed_reference", {})
    a2_spi = a2_baseline.get("sec_per_item_per_epoch", 0.00878)

    # Derived total estimate
    n_train_full = len(data["compiled"]["train"])
    epochs = cfg["epochs"]
    n_seeds = N_SEEDS
    n_trainable_units = n_seeds * len(EXP81_TRAIN_ARMS)  # 20 units (4 arms × 5 seeds)

    estimates = {}
    for arm, r in results.items():
        spi = r["sec_per_item_per_epoch"]
        train_sec = n_train_full * epochs * spi
        estimates[arm] = {
            "sec_per_item_per_epoch_measured": spi,
            "ratio_vs_A2_baseline": round(spi / a2_spi, 2),
            "estimated_sec_per_seed": round(train_sec, 0),
            "estimated_hours_per_seed": round(train_sec / 3600, 2),
            "n_seeds": n_seeds,
            "estimated_total_wall_hours_%dworkers" % n_seeds:
                round(train_sec * n_seeds / n_seeds / 3600, 2),
        }

    compute = {
        "derived_from": ("measured 3-epoch times (doc 73 §3: derived, not guessed). "
                         "Build agent task 1."),
        "n_items_measured": n_train,
        "n_timed_epochs": 3,
        "k_distribution_timing_slice": k_dist,
        "A2_committed_reference": a2_baseline,
        "arms": results,
        "estimates": estimates,
        "n_trainable_units_total": n_trainable_units,
        "note": ("A2q2 at 2K wires has 2^(2K) state vs A2's 2^K -- "
                 "ratio expected 16x at K=4, 256x at K=8. A2-WIDE uses "
                 "K wires (same as A2), overhead from extra intro Ry only."),
        "machine_load_note": ("exp82 workers may be running; ratio vs A2 "
                              "baseline is load-robust; absolute times may "
                              "be contention-affected."),
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(TIME_OUT, "w", encoding="utf-8") as f:
        json.dump(compute, f, ensure_ascii=False, indent=1)
    print("[timing] wrote %s" % TIME_OUT)
    return compute


# ================================================================
# worker main loop
# ================================================================
def worker_main(worker_id, n_workers, data, cfg):
    """exp81 worker: claims and runs training jobs (A2q2, A2-WIDE, A2-RETRAIN).

    A2 committed checkpoints are READ ONLY (drift anchor). B1 is also read-only.
    """
    # Verify smoke gate
    if not os.path.exists(SMOKE_MARKER):
        print("[w%d] ABORT: %s missing. Run --smoke first." % (worker_id, SMOKE_MARKER),
              flush=True)
        sys.exit(1)

    # Also draw the fixed scramble for A2q2-FLOOR (the scrambled arm)
    all_ids = sorted({cs.item_id for split in data["compiled"].values()
                      for cs in split})
    scramble_map = draw_fixed_scramble(all_ids, FLOOR_SCRAMBLE_SEED)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(CTL_DIR, exist_ok=True)
    os.makedirs(LOCK_DIR, exist_ok=True)

    print("[w%d] queue start (%d threads)" % (worker_id, 2), flush=True)

    for arm in ("A2q2", "A2-WIDE", "A2-RETRAIN", "A2q2-FLOOR"):
        for seed in range(N_SEEDS):
            job_id = "%s_s%d" % (arm.replace("-", ""), seed)
            ck_json, ck_pt = unit_paths81(arm, seed)
            if settled81([ck_json, ck_pt], job_id):
                continue
            if not claim(job_id):
                continue
            print("[w%d] claimed %s" % (worker_id, job_id), flush=True)
            try:
                if arm == "A2q2-FLOOR":  # floor arm: scrambled labels throughout
                    job_main81(arm, seed, data, cfg, scramble_map=scramble_map)
                else:
                    job_main81(arm, seed, data, cfg)
            except Exception as exc:
                record_failure81(job_id, exc, tag="train")

    print("[w%d] queue done." % worker_id, flush=True)


# ================================================================
# main entry point
# ================================================================
def main():
    ap = argparse.ArgumentParser(description="exp81 width per referent")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--time-epoch", action="store_true")
    ap.add_argument("--drift-anchor", action="store_true",
                    help="bit-exact re-eval of committed A2 checkpoint (drift check)")
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--nworkers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2,
                    help="intra-op parallelism per worker")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    cfg = {"lr": 0.005, "batch": 1, "epochs": 30}

    data = prepare()
    cfg["verbs"] = data["verbs"]

    if args.smoke:
        sys.exit(smoke_battery(data))

    if args.drift_anchor:
        anchor = run_drift_anchor(data)
        print(json.dumps(anchor, indent=1))
        return

    if args.time_epoch:
        compute = measure_epoch_time(data, cfg, n_train=50)
        print("\n=== TIMING TABLE ===")
        for arm, r in compute["arms"].items():
            est = compute["estimates"][arm]
            print("  %s: %.5f s/item/epoch  ratio_vs_A2=%.2fx  est_hours_per_seed=%.2f"
                  % (arm, r["sec_per_item_per_epoch"],
                     est["ratio_vs_A2_baseline"],
                     est["estimated_hours_per_seed"]))
        print("  A2 baseline: 0.00878 s/item/epoch (exp79_epoch_timing.json)")
        return

    if args.worker is not None:
        worker_main(args.worker, args.nworkers, data, cfg)
        return

    print("Usage: --smoke | --time-epoch | --drift-anchor | --worker I --nworkers N")


if __name__ == "__main__":
    main()
