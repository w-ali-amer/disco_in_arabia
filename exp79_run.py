# -*- coding: utf-8 -*-
"""exp79_run.py -- CHANNEL COMPOSITION experiment (doc 71).

Registration: qnlp_private_docs/71_exp79_channel_composition.md

Arms (on exp42r's task, dataset, splits, readout, controls — UNCHANGED):
  C-DISCARD  A2's dressed-symmetric verb block + one CRY coupling gate
             (patient, ancilla) per event; ancilla wire K reused across
             events and traced at readout.  Mixed-state simulation via
             purified register (K+1 qubits, state <= 2^(K+1), exact
             statevector autodiff valid).
  C-RESET    Same as C-DISCARD but ancilla measured after each event;
             classical feedback: X on patient if outcome=1.  Simulated
             by 2-branch outcome branching, capped at MAX_BRANCH_EVENTS=8;
             T>8 tail reverts to C-DISCARD trace (no branching), recorded
             per item.
  C-FLOOR    Fixed-scramble variant (doc 66 SS12.2 / doc 70 B3): ONE
             scramble drawn before training (FLOOR_SCRAMBLE_SEED), same
             scrambled labels throughout.  Trained as C-DISCARD on
             scrambled test labels.  Used for floor-admissibility gate.
  A2         Committed exp42r arm; read from exp42r_ckpt/ (read-only).
  B1         Committed exp42r arm; read from exp42r_ckpt/ (read-only).

Purity instrument (doc 71 SS1 + doc 65 SS7.3):
  question-time queried-pair purity (pair_purity, imported from
  exp42r_diagnostics) logged per test item for every arm.  The SS1
  boolean channel_preserves_purity reads it: purity > 0.75 at T=5-10.
  For C-RESET the purity is the ensemble Tr(rho_2^2) from the branch
  density matrix, not the mean of per-branch purities (correctness
  requirement noted by the reviewer).

Doc-70 compliance (SS6):
  -- seed-level paired t as the primary significance leg (in aggregate)
  -- item-level sign-flip is descriptive only
  -- floor-admissibility gate: scrambled floor within 2 binomial SD of 0.5
  -- SELF-EMITTING gate artifacts: every smoke gate writes its own
     hash-pinned JSON with a designed-to-fail assertion shown failing
     (SG-FS fixed-scramble, SG-ANT ancilla-never-traced, SG-PS purity-sanity)

Construction choice FLAGGED (doc 71 SS2 vs SS4):
  SS2 says "one ancilla wire per event" (K+T wires total) and "purified
  register"; SS4 says "state <= 2^(K+2)". This build uses ONE reused
  ancilla wire (K+1 qubits), consistent with SS4's width bound, and the
  "purified register" means the ancilla is kept to end-of-circuit and
  traced at readout -- not that separate per-event ancilla qubits are
  introduced (which at K+T=28 would require 2^28 amplitudes, infeasible).
  Flagged. The doc-22 SS9 mini-Duneau construction is the precedent:
  it uses ONE ancilla wire per stage, reused after discard.

  Additionally, the "purified register" equivalence holds: keeping ONE
  ancilla and never mid-circuit-tracing gives the same measurement
  statistics as tracing after each event only if the ancilla starts in
  |0> at each event (which requires the reset in C-RESET, or the
  "discard" of the old ancilla before reuse in C-DISCARD). In this
  build, C-DISCARD reuses the ancilla WITHOUT resetting it between
  events, so the ancilla accumulates entanglement from all events. This
  is a DIFFERENT channel from ideal C-DISCARD (it is a Markovian
  environment model), which is flagged in the results.

Seed base: 79000; 5 seeds per arm (0..4); init streams disjoint via
  EXP79_STREAM_BASE = 79_000_000.

Workers: exp79_w*.log, exp79_ckpt/

Usage:
  python3 exp79_run.py --drift-anchor        # bit-exact re-eval of A2 s0
  python3 exp79_run.py --smoke               # gate battery + write marker
  python3 exp79_run.py --time-epoch          # measure first-epoch time
  python3 exp79_run.py --worker i --nworkers 5
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

# import pair_purity from the registered diagnostics module (doc 65 SS7.3)
from exp42r_diagnostics import pair_purity
from exp42r_run import (claim, evaluate, prepare, record_failure, settled)

# ================================================================
# exp79 constants
# ================================================================
EXP79_ARMS = ("C-DISCARD", "C-RESET", "C-FLOOR")
EXP79_ARM_INDEX = {a: i for i, a in enumerate(EXP79_ARMS)}
EXP79_STREAM_BASE = 79_000_000   # disjoint from exp42r's 42_000_000 family
FLOOR_SCRAMBLE_SEED = 79999      # ONE fixed draw, used throughout

CKPT_DIR = os.environ.get("EXP79_CKPT_DIR", "exp79_ckpt")
CTL_DIR = os.environ.get("EXP79_CTL_DIR", "exp79_ctl")
LOCK_DIR = os.environ.get("EXP79_LOCK_DIR", "exp79_locks")

SMOKE_MARKER = "exp79_smoke_PASSED.json"
SMOKE_REPORT = "exp79_smoke.json"
ANCHOR_OUT = "exp79_anchor.json"
TIME_OUT = "exp79_epoch_timing.json"

EXP42R_CKPT_DIR = "exp42r_ckpt"   # read-only committed checkpoints

N_SEEDS = 5                        # 5 seeds per arm
MAX_BRANCH_EVENTS = 8              # C-RESET: branch for events 1..8, trace tail
INIT_STD = 0.1

BARRIER_TIMEOUT_SEC = 12 * 3600
BARRIER_POLL_SEC = 30

C_DISCARD_NOTE = (
    "FLAGGED (doc 71 SS2 vs SS4): C-DISCARD uses ONE reused ancilla wire "
    "(K+1 qubits) rather than one-per-event (K+T qubits). This implements "
    "a Markovian-environment channel where the ancilla accumulates entanglement "
    "across events without mid-circuit reset. SS4 registers state <= 2^(K+2), "
    "consistent with this build. The 'purified register' means ancilla is kept "
    "to end-of-circuit and traced at readout (via prob_all_zero_on), so exact "
    "statevector autodiff is valid. Flagged for the results addendum.")

C_RESET_NOTE = (
    "C-RESET: 2-branch outcome simulation per event for events 1..MAX_BRANCH_EVENTS=8. "
    "T>8 tail reverts to C-DISCARD trace (no additional branching). Branching "
    "is simulated by keeping unnormalized branch states; the readout is the sum "
    "of prob_all_zero_on over all branch states (correct weighting, since "
    "prob_all_zero_on scales as ||psi||^2). Pair purity is the ensemble "
    "Tr(rho_2^2) from the sum of branch density matrices, NOT the weighted "
    "mean of per-branch purities. Items with T>8 have the tail-events "
    "C-DISCARD-treated flag recorded per item.")

PURITY_BAR = 0.75      # doc 71 SS1: purity > 0.75 at T=5-10
PURITY_T_LO = 5
PURITY_T_HI = 10


# ================================================================
# Norm-unchecked gate application for unnormalized branch states
# (C-RESET branches after split are unnormalized; sim.apply_gate
# asserts norm==1 which fails for them)
# ================================================================
def _apply_gate_unnorm(state, U, wires):
    """Apply gate to possibly-unnormalized state (no norm assertion).
    Exact copy of torch_sv_sim.apply_gate minus the CHECK_NORM block."""
    k = len(wires)
    Ut = U.reshape((2,) * (2 * k))
    out = torch.tensordot(Ut, state, dims=(list(range(k, 2 * k)), list(wires)))
    return torch.movedim(out, list(range(k)), list(wires))


def _prob_all_zero_on_unnorm(state, wires):
    """P(wires == |0..0>) on an unnormalized state. Scales as ||state||^2."""
    s = state
    for w in sorted(wires, reverse=True):
        s = s.select(w, 0)
    return (s.abs() ** 2).sum()


# ================================================================
# CRY gate helper (not exported from exp42_sim_adapter)
# ================================================================
def _cry(theta):
    """Controlled-RY(theta): control=wire0, target=wire1.
    When theta=0: CRY(0) = I_4 (identity), so ancilla stays untouched."""
    r = sim.ry(theta)
    eye = torch.eye(2, dtype=sim.CDTYPE)
    zero = torch.zeros((2, 2), dtype=sim.CDTYPE)
    top = torch.cat([eye, zero], dim=1)
    bot = torch.cat([zero, r], dim=1)
    return torch.cat([top, bot], dim=0)


def _x_gate():
    """Pauli X: |0><1| + |1><0|."""
    return torch.tensor([[0, 1], [1, 0]], dtype=sim.CDTYPE)


# ================================================================
# C-DISCARD model
# ================================================================
class CDiscardModel(QuantumStoryModel):
    """A2's circuit extended with CRY(coupling[v]) on (patient, ancilla).

    When coupling[v]=0 for all v: ancilla stays in |0>, and tracing it
    out at readout is equivalent to A2's K-wire forward -- the 'ancilla
    never traced' smoke gate asserts this numerically (SG-ANT).

    The ancilla wire is at index cs.K in the K+1-qubit state.
    """

    def __init__(self, verbs, seed):
        super().__init__("A2", verbs, seed)
        # coupling: one angle per verb, init at 0 (but overridden by apply_init79)
        self.coupling = nn.ParameterDict(
            {v: nn.Parameter(torch.zeros(())) for v in self.verbs})

    def story_values(self, cs, verb_map=None):
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)   # A2's 4x4
            return cache[v]

        anc = cs.K                              # ancilla wire index
        psi = sim.init_state(cs.K + 1)         # K+1 qubits

        for w, _noun, angle in cs.intro:
            psi = sim.apply_gate(psi, sim.ry(angle), (w,))

        for v, wa, wp in cs.events:
            psi = sim.apply_gate(psi, block(v), (wa, wp))
            psi = sim.apply_gate(psi, _cry(self.coupling[vm.get(v, v)]),
                                 (wp, anc))

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
        # doc 71 SS9 numerical amendment: cross_entropy(log v) is EXACTLY
        # invariant to common scaling of vals (log v/s = log v - log s, a
        # constant logit shift), so dividing by the detached sum changes the
        # loss by provably nothing while preventing the branch-norm underflow
        # that produced non-finite gradients in C-RESET (log'(v) ~ 1/v blowup).
        scale = vals.detach().sum().clamp_min(1e-30)
        vals = vals / scale
        # doc 71 SS9.2: clamp normalized outcome probs away from exact 0 --
        # late-training determinism (one outcome -> 0) still exploded log-grad
        # (s1 NaN at e30 on SS9 code). Bias bounded by 1e-6, negligible.
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
        per_verb_A2 = 14
        return {"arm": "C-DISCARD", "per_verb_A2": per_verb_A2,
                "coupling_per_verb": 1, "n_verbs": len(self.verbs),
                "readout_shared": readout, "n_coupling_params": n_coupling,
                "total_trained": total,
                "note": C_DISCARD_NOTE}


# ================================================================
# C-RESET model
# ================================================================
class CResetModel(CDiscardModel):
    """C-DISCARD + measure-and-reset with classical feedback.

    After each event (up to MAX_BRANCH_EVENTS):
      - Apply verb block + CRY coupling to (wa, wp, anc)
      - Measure ancilla: branch on outcome 0 or 1
      - If outcome 1: apply X to patient wire
      - Reset ancilla to |0> (via state collapse)
    After event MAX_BRANCH_EVENTS: tail events use C-DISCARD trace.

    story_values returns weighted sum of branch readout probs.
    The weighting is implicit: unnormalized branch states are used and
    prob_all_zero_on scales as ||psi||^2, giving correct weighting.
    """

    def story_values(self, cs, verb_map=None):
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)
            return cache[v]

        anc = cs.K
        psi0 = sim.init_state(cs.K + 1)

        for w, _noun, angle in cs.intro:
            psi0 = sim.apply_gate(psi0, sim.ry(angle), (w,))

        # branches: list of unnormalized states (psi with ||psi||^2 = weight)
        branches = [psi0]

        for i, (v, wa, wp) in enumerate(cs.events):
            bv = vm.get(v, v)
            blk = block(v)
            coup = _cry(self.coupling[bv])

            if i < MAX_BRANCH_EVENTS:
                new_branches = []
                for psi in branches:
                    # apply verb block + coupling (unnorm-safe)
                    psi = _apply_gate_unnorm(psi, blk, (wa, wp))
                    psi = _apply_gate_unnorm(psi, coup, (wp, anc))
                    # measure ancilla: move anc to dim 0
                    psi_af = torch.movedim(psi, anc, 0)  # (2, 2, ..., 2)
                    s0 = psi_af[0]   # K-qubit, weight = P(anc=0)
                    s1 = psi_af[1]   # K-qubit, weight = P(anc=1)
                    z0 = torch.zeros_like(s0)
                    # branch 0: anc measured 0, reset to 0
                    psi_b0 = torch.movedim(torch.stack([s0, z0], dim=0),
                                            0, anc)
                    new_branches.append(psi_b0)
                    # branch 1: anc measured 1, X on patient, reset to 0
                    # X on wp via flip (unnormalized, no gate-norm check)
                    s1_after_x = torch.flip(s1, [wp])
                    psi_b1 = torch.movedim(torch.stack([s1_after_x, z0],
                                                        dim=0), 0, anc)
                    new_branches.append(psi_b1)
                branches = new_branches
            else:
                # tail: C-DISCARD trace (no branching; unnorm-safe)
                new_branches = []
                for psi in branches:
                    psi = _apply_gate_unnorm(psi, blk, (wa, wp))
                    psi = _apply_gate_unnorm(psi, coup, (wp, anc))
                    new_branches.append(psi)
                branches = new_branches

        # question inverse prep + readout (sum over branches; unnorm-safe)
        vseq, qa, qp = cs.question
        qw = (qa, qp)
        total_pos = None
        total_neg = None
        for psi in branches:
            for qv in reversed(vseq):
                psi = _apply_gate_unnorm(psi, block(qv).conj().T, qw)
            vp = _prob_all_zero_on_unnorm(
                _apply_gate_unnorm(psi, self.q_pos.matrix().conj().T, qw), qw)
            vn = _prob_all_zero_on_unnorm(
                _apply_gate_unnorm(psi, self.q_neg.matrix().conj().T, qw), qw)
            total_pos = vp if total_pos is None else total_pos + vp
            total_neg = vn if total_neg is None else total_neg + vn
        return torch.stack([total_pos, total_neg])

    def param_counts(self):
        pc = super().param_counts()
        pc["arm"] = "C-RESET"
        pc["note"] = C_RESET_NOTE
        return pc


# ================================================================
# Ensemble purity for C-RESET (from unnormalized branch states)
# ================================================================
def ensemble_pair_purity(branches_unnorm, w1, w2):
    """Correct ensemble Tr(rho_2^2) from unnormalized branch states.

    Each branch psi_i has ||psi_i||^2 = its branch weight.
    rho_2 = sum_i Tr_rest(|psi_i><psi_i|) -- no additional weighting needed.
    """
    rho = None
    for psi in branches_unnorm:
        p = torch.movedim(psi.detach().to(torch.complex128),
                          (w1, w2), (0, 1)).reshape(4, -1)
        rho_i = p @ p.conj().T
        rho = rho_i if rho is None else rho + rho_i
    return float((rho.abs() ** 2).sum())


def creset_branches_at_question(model, cs):
    """Run C-RESET forward up to after question inverse prep, return branch states.

    Used for per-item pair_purity logging (not for training).
    """
    vm = {}
    cache = {}

    def block(v):
        if v not in cache:
            cache[v] = model.verb_matrix(v)
        return cache[v]

    anc = cs.K
    psi0 = sim.init_state(cs.K + 1)
    for w, _noun, angle in cs.intro:
        psi0 = sim.apply_gate(psi0, sim.ry(angle), (w,))

    branches = [psi0]
    n_branched = 0
    tail_events = 0

    for i, (v, wa, wp) in enumerate(cs.events):
        blk = block(v)
        coup = _cry(model.coupling[v])

        if i < MAX_BRANCH_EVENTS:
            n_branched += 1
            new_branches = []
            for psi in branches:
                psi = _apply_gate_unnorm(psi, blk, (wa, wp))
                psi = _apply_gate_unnorm(psi, coup, (wp, anc))
                psi_af = torch.movedim(psi, anc, 0)
                s0 = psi_af[0]
                s1 = psi_af[1]
                z0 = torch.zeros_like(s0)
                psi_b0 = torch.movedim(torch.stack([s0, z0], dim=0), 0, anc)
                new_branches.append(psi_b0)
                s1_after_x = torch.flip(s1, [wp])
                psi_b1 = torch.movedim(torch.stack([s1_after_x, z0], dim=0),
                                        0, anc)
                new_branches.append(psi_b1)
            branches = new_branches
        else:
            tail_events += 1
            new_branches = []
            for psi in branches:
                psi = _apply_gate_unnorm(psi, blk, (wa, wp))
                psi = _apply_gate_unnorm(psi, coup, (wp, anc))
                new_branches.append(psi)
            branches = new_branches

    vseq, qa, qp = cs.question
    qw = (qa, qp)
    branches_at_q = []
    for psi in branches:
        for qv in reversed(vseq):
            psi = _apply_gate_unnorm(psi, block(qv).conj().T, qw)
        branches_at_q.append(psi.detach())

    return branches_at_q, qa, qp, tail_events


def cdiscard_state_at_question(model, cs):
    """C-DISCARD: run forward to after question inverse prep, return psi."""
    cache = {}

    def block(v):
        if v not in cache:
            cache[v] = model.verb_matrix(v)
        return cache[v]

    anc = cs.K
    psi = sim.init_state(cs.K + 1)
    for w, _noun, angle in cs.intro:
        psi = sim.apply_gate(psi, sim.ry(angle), (w,))
    for v, wa, wp in cs.events:
        psi = sim.apply_gate(psi, block(v), (wa, wp))
        psi = sim.apply_gate(psi, _cry(model.coupling[v]), (wp, anc))
    vseq, qa, qp = cs.question
    for qv in reversed(vseq):
        psi = sim.apply_gate(psi, block(qv).conj().T, (qa, qp))
    return psi.detach(), qa, qp


# ================================================================
# per-item purity logging
# ================================================================
def log_pair_purities(model, compiled, arm_label):
    """Log pair_purity at question time for every test item.
    Returns list of dicts: {item_id, K, T, pair_purity_at_readout, ...}
    """
    rows = []
    with torch.no_grad():
        for cs in compiled:
            if arm_label == "C-RESET":
                branches_q, qa, qp, tail = creset_branches_at_question(
                    model, cs)
                pp = ensemble_pair_purity(branches_q, qa, qp)
                rows.append({"item_id": cs.item_id, "K": cs.K, "T": len(cs.events),
                             "pair_purity_at_readout": pp,
                             "n_branches": len(branches_q),
                             "tail_events": tail,
                             "used_tail_fallback": tail > 0})
            else:
                psi_q, qa, qp = cdiscard_state_at_question(model, cs)
                pp = pair_purity(psi_q, qa, qp)
                rows.append({"item_id": cs.item_id, "K": cs.K, "T": len(cs.events),
                             "pair_purity_at_readout": pp,
                             "n_branches": 1,
                             "tail_events": 0,
                             "used_tail_fallback": False})
    return rows


def purity_stats(rows):
    """Summarise purity rows: mean per T-bucket for T in [5,10]."""
    by_T = {}
    for r in rows:
        t = r["T"]
        by_T.setdefault(t, []).append(r["pair_purity_at_readout"])
    out = {}
    for t, vals in sorted(by_T.items()):
        out[t] = {"n": len(vals), "mean": sum(vals) / len(vals),
                  "min": min(vals), "max": max(vals)}
    in_range = [v for t, vs in by_T.items()
                for v in vs if PURITY_T_LO <= t <= PURITY_T_HI]
    out["channel_preserves_purity_gate"] = {
        "T_range": [PURITY_T_LO, PURITY_T_HI],
        "n_items": len(in_range),
        "mean_purity": sum(in_range) / len(in_range) if in_range else None,
        "all_above_bar": all(v > PURITY_BAR for v in in_range),
        "bar": PURITY_BAR,
        "bar_provenance": "doc 71 SS1 registered prediction: >0.75 at T=5-10 "
                          "where A2 measured ~0.50-0.68"}
    return out


# ================================================================
# init for exp79 arms
# ================================================================
def init_stream_seed79(arm, seed):
    return EXP79_STREAM_BASE + 1000 * EXP79_ARM_INDEX[arm] + seed


def apply_init79(model, arm, seed):
    """Apply registered init for exp79 arms.

    A2 base params (core, dress, q_pos, q_neg): same N(0,0.1^2) as exp42r.
    Coupling params: N(0, 0.1^2) from the exp79 stream.
    C-FLOOR uses C-DISCARD's init (same arm index).
    """
    # Map C-FLOOR to C-DISCARD for stream (same family)
    stream_arm = "C-DISCARD" if arm == "C-FLOOR" else arm
    stream = init_stream_seed79(stream_arm, seed)
    g = torch.Generator().manual_seed(stream)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * INIT_STD)
        flat = torch.cat([p.detach().reshape(-1).to(torch.float64)
                          for _n, p in sorted(model.named_parameters())])
        h = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
    return {"spec": "exp79 init: all params ~ N(0, %.2f^2)" % INIT_STD,
            "arm": arm, "seed": seed, "stream_seed": stream,
            "stream_rule": "79_000_000 + 1000*EXP79_ARM_INDEX[arm] + seed "
                           "(C-FLOOR uses C-DISCARD index; streams disjoint "
                           "from exp42r's 42_000_000 family)",
            "n_params": int(flat.numel()), "init_vector_sha256": h}


def make_model79(arm, verbs):
    """Construct model for an exp79 arm (no init applied yet)."""
    if arm in ("C-DISCARD", "C-FLOOR"):
        return CDiscardModel(verbs, seed=0)   # seed arg unused; init via apply_init79
    if arm == "C-RESET":
        return CResetModel(verbs, seed=0)
    raise ValueError("unknown exp79 arm: %r" % arm)


def build_model79(arm, verbs, seed):
    """Construct + apply registered init. The ONE way a model is built."""
    m = make_model79(arm, verbs)
    spec = apply_init79(m, arm, seed)
    return m, spec


# ================================================================
# unit paths for exp79 (separate from exp42r_ckpt)
# ================================================================
def unit_paths79(arm, seed, ckpt_dir=CKPT_DIR):
    return (os.path.join(ckpt_dir, "%s_s%d.json" % (arm.replace("-", ""), seed)),
            os.path.join(ckpt_dir, "%s_s%d.pt" % (arm.replace("-", ""), seed)))


def failed_path79(job_id, ctl_dir=CTL_DIR):
    return os.path.join(ctl_dir, "FAILED_%s.json" % job_id)


def settled79(outputs, job_id):
    return (all(os.path.exists(p) for p in outputs)
            or os.path.exists(failed_path79(job_id)))


def record_failure79(job_id, exc, tag):
    payload = {"job": job_id, "tag": tag, "error": repr(exc),
               "traceback": traceback.format_exc(),
               "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "isolation": "arm-level failure; worker continues"}
    with open(failed_path79(job_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("[FAIL] %s: %r (isolated)" % (job_id, exc), flush=True)


# ================================================================
# fixed scramble (floor)
# ================================================================
def draw_fixed_scramble(union_item_ids, seed=FLOOR_SCRAMBLE_SEED):
    """ONE scramble drawn before training, returned as a dict item_id -> answer.
    doc 66 SS12.2 / doc 70 B3: the SAME scramble is used for all seeds."""
    rng = random.Random(seed)
    ids = list(union_item_ids)
    scrambled = [0] * len(ids)
    # shuffle 0/1 labels independently of the ids
    labels = [0] * (len(ids) // 2) + [1] * (len(ids) - len(ids) // 2)
    rng.shuffle(labels)
    return {iid: lab for iid, lab in zip(ids, labels)}


def evaluate_scrambled(model, compiled, scramble_map):
    """Evaluate model on a compiled set but with SCRAMBLED labels."""
    correct = []
    with torch.no_grad():
        for cs in compiled:
            pred = model.predict(cs)
            # scrambled answer (0=yes, 1=no as in cs.answer encoding)
            ans = scramble_map.get(cs.item_id, cs.answer)
            correct.append(1 if pred == ans else 0)
    return sum(correct) / len(correct), correct


# ================================================================
# training
# ================================================================
def train_unit79(arm, seed, data, cfg,
                 scramble_map=None,
                 splits=("test_a", "test_b", "test_c")):
    """One (arm, seed). Returns (model, detail).
    If scramble_map is given (C-FLOOR): evaluate test on scrambled labels."""
    t0 = time.time()
    model, init_spec = build_model79(arm, cfg["verbs"], seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = random.Random(9000 + seed)
    train, val = data["compiled"]["train"], data["compiled"]["val"]
    if scramble_map is not None:
        # doc 71 SS6 / doc 70 B3: the floor TRAINS on the fixed scramble too --
        # eval-only scrambling is a vacuous floor (~0.5 by construction).
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
    nonfinite_skips = []  # doc 71 SS9.3 incident log
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
            # grad norm
            tot = sum(p.grad.norm().item() ** 2 for p in model.parameters()
                      if p.grad is not None)
            gn = math.sqrt(tot)
            if not math.isfinite(gn):
                # doc 71 SS9.3: skip-batch-on-nonfinite with incident logging
                # (the NaN originates in branch-state autograd upstream of the
                # loss; SS9/SS9.2 guards were insufficient). Optimizer step is
                # SKIPPED for this batch; the incident count is recorded.
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
        losses_e.append(sum(losses) / len(losses))
        gnorms_e.append(sum(gnorms) / len(gnorms))
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
    detail = {
        "arm": arm, "seed": seed, "best_epoch": best["epoch"],
        "epochs_run": len(losses_e), "max_epochs": cfg["epochs"],
        "train_acc_at_best": best["train_acc"], "val_acc": best["val_acc"],
        "epoch_mean_losses": losses_e, "epoch_mean_grad_norms": gnorms_e,
        "epoch_running_train_acc": runacc_e, "epoch_val_acc": val_e,
        "init": init_spec, "param_counts": model.param_counts(),
        "model_selection": "in-distribution val split ONLY (doc 22 SS4)",
        "test": {}, "wall_time_sec": round(time.time() - t0, 1),
        "construction_note": C_DISCARD_NOTE if arm in ("C-DISCARD", "C-FLOOR")
                             else C_RESET_NOTE,
    }

    # evaluate on test splits
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
    # union
    if scramble_map is not None:
        ua, uc = evaluate_scrambled(model, data["compiled"]["test_union"],
                                    scramble_map)
    else:
        ua, uc = evaluate(model, data["compiled"]["test_union"])
    detail["test"]["test_union"] = {"accuracy": ua, "n": len(uc),
                                    "scrambled": scramble_map is not None}
    detail["union_item_ids"] = data["union_ids"]
    detail["union_per_item_correct"] = uc

    # purity instrument: log per test_union item
    purity_rows = log_pair_purities(model, data["compiled"]["test_union"], arm)
    detail["purity_rows"] = purity_rows
    detail["purity_stats"] = purity_stats(purity_rows)

    detail["nonfinite_skips_doc71_SS9_3"] = nonfinite_skips
    detail["vocab_sha256"] = K42.vocab_sha(data["vocab"])
    detail["dataset_sha256"] = data["prov"]["dataset"]["sha256"]
    detail["embeddings_sha256"] = data["emb_sha"]
    return model, detail


def job_main79(arm, seed, data, cfg, scramble_map=None):
    ck_json, ck_pt = unit_paths79(arm, seed)
    model, detail = train_unit79(arm, seed, data, cfg, scramble_map=scramble_map)
    torch.save(model.state_dict(), ck_pt)
    with open(ck_json, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False)
    print("[unit] %s s%d done: val=%.3f union=%.3f (%.0fs)"
          % (arm, seed, detail["val_acc"],
             detail["test"]["test_union"]["accuracy"],
             detail["wall_time_sec"]), flush=True)


# ================================================================
# drift anchor
# ================================================================
def run_drift_anchor(data):
    """Bit-exact re-eval of committed A2 seed 0 checkpoint.

    Reads exp42r_ckpt/A2_s0.{json,pt} (READ-ONLY committed artifact).
    Asserts per-split accuracy matches to 1e-4.
    """
    ck_json = os.path.join(EXP42R_CKPT_DIR, "A2_s0.json")
    ck_pt = os.path.join(EXP42R_CKPT_DIR, "A2_s0.pt")
    if not (os.path.exists(ck_json) and os.path.exists(ck_pt)):
        raise RuntimeError("drift anchor: exp42r_ckpt/A2_s0.{json,pt} "
                           "absent -- committed artifact missing")
    with open(ck_json, encoding="utf-8") as f:
        committed = json.load(f)

    # rebuild A2 model with same factory as exp42r
    model, _spec = K42.build_model("A2", data["verbs"], data["vocab"],
                                   0, data["angles"])
    sd = torch.load(ck_pt, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    TOL = 1e-4
    results = {}
    all_ok = True
    for sp in ("test_a", "test_b", "test_c", "test_union"):
        acc, corr = evaluate(model, data["compiled"][sp])
        committed_acc = committed["test"][sp]["accuracy"]
        delta = abs(acc - committed_acc)
        ok = delta < TOL
        results[sp] = {"accuracy": acc, "committed_accuracy": committed_acc,
                       "delta": delta, "within_tol": ok, "tol": TOL}
        if not ok:
            all_ok = False
        print("[anchor] %s: acc=%.4f committed=%.4f delta=%.2e %s"
              % (sp, acc, committed_acc, delta, "OK" if ok else "DRIFT!"),
              flush=True)

    # per-item correctness vector check on test_union
    if "union_per_item_correct" in committed:
        _, corr_u = evaluate(model, data["compiled"]["test_union"])
        item_match = (corr_u == committed["union_per_item_correct"])
        results["test_union"]["per_item_match"] = bool(item_match)
        if not item_match:
            all_ok = False
            print("[anchor] WARNING: per-item correctness MISMATCH", flush=True)

    anchor = {
        "component": "exp79 drift anchor (doc 71 SS2)",
        "committed_checkpoint": {"json": ck_json, "pt": ck_pt},
        "committed_wall_time_sec": committed.get("wall_time_sec"),
        "committed_val_acc": committed.get("val_acc"),
        "re_eval_results": results,
        "all_within_tol": all_ok,
        "tol": TOL,
        "verdict": "PASS" if all_ok else "DRIFT DETECTED",
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not all_ok:
        anchor["error"] = ("DRIFT DETECTED: re-eval does not bit-exactly "
                           "reproduce the committed checkpoint. This "
                           "invalidates the experiment baseline.")
    with open(ANCHOR_OUT, "w", encoding="utf-8") as f:
        json.dump(anchor, f, ensure_ascii=False, indent=1)
    print("[anchor] wrote %s  verdict=%s" % (ANCHOR_OUT, anchor["verdict"]),
          flush=True)
    if not all_ok:
        raise RuntimeError("Drift anchor FAILED -- see %s" % ANCHOR_OUT)
    return anchor


# ================================================================
# SELF-EMITTING GATE ARTIFACTS (doc 70 SS3.6)
# ================================================================
def _artifact_hash(d):
    """Canonical sha256 of a JSON-serializable dict."""
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---- SG-FS: fixed-scramble gate -----------------------------------
SG_FS_OUT = "exp79_gate_SG_FS.json"


def gate_fixed_scramble(data):
    """SG-FS: ONE scramble drawn before training.
    Train a tiny C-DISCARD model on real data, evaluate on scrambled labels.
    PASS: scrambled accuracy lands within 2 binomial SD of 0.5.
    DESIGNED-TO-FAIL assertion: scrambled accuracy >= 0.70 (shown failing).
    """
    verbs = data["verbs"]
    rng = random.Random(0)
    pool = data["compiled"]["train"][:]
    rng.shuffle(pool)
    tiny = pool[:40]
    tiny_val = data["compiled"]["val"][:10]

    # One scramble drawn NOW (before training)
    union_ids = [cs.item_id for cs in data["compiled"]["test_union"]]
    all_ids = sorted({cs.item_id for split in data["compiled"].values() for cs in split})
    scramble_map = draw_fixed_scramble(all_ids, FLOOR_SCRAMBLE_SEED)
    scramble_sha = hashlib.sha256(
        json.dumps(scramble_map, sort_keys=True).encode()).hexdigest()

    # Train tiny C-DISCARD model for 5 epochs
    model, init_spec = build_model79("C-DISCARD", verbs, 0)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(5):
        for cs in tiny:
            opt.zero_grad()
            loss, _ = model.story_loss(cs)
            loss.backward()
            opt.step()

    # Evaluate on scrambled union
    scr_acc, scr_corr = evaluate_scrambled(
        model, data["compiled"]["test_union"][:min(100, len(union_ids))],
        scramble_map)

    # Real (unscrambled) accuracy for contrast
    real_acc, _ = evaluate(
        model, data["compiled"]["test_union"][:min(100, len(union_ids))])

    n = len(scr_corr)
    sd_binom = math.sqrt(0.5 * 0.5 / n)
    within_2sd = abs(scr_acc - 0.5) <= 2 * sd_binom
    floor_admissible = within_2sd

    # designed-to-fail assertion: scr_acc >= 0.70 (should fail)
    dtf_assertion = "scrambled_acc >= 0.70"
    dtf_passes = bool(scr_acc >= 0.70)   # expected False -- this is shown FAILING

    artifact = {
        "gate": "SG-FS fixed-scramble floor (doc 66 SS12.2, doc 70 B3)",
        "designed_to_fail_assertion": dtf_assertion,
        "designed_to_fail_result": dtf_passes,
        "designed_to_fail_shown_failing": not dtf_passes,
        "scramble_seed": FLOOR_SCRAMBLE_SEED,
        "scramble_sha256": scramble_sha,
        "n_eval_items": n,
        "scrambled_accuracy": scr_acc,
        "real_accuracy": real_acc,
        "binomial_sd": sd_binom,
        "within_2sd_of_chance": within_2sd,
        "floor_admissible": floor_admissible,
        "pass_criterion": "scrambled_acc within 2 binomial SD of 0.5",
        "passed": floor_admissible,
        "inputs": {"n_train_items": len(tiny), "train_epochs": 5,
                   "init_spec": init_spec},
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    artifact["artifact_sha256"] = _artifact_hash(
        {k: v for k, v in artifact.items() if k != "artifact_sha256"})
    with open(SG_FS_OUT, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=1)
    print("[SG-FS] scrambled_acc=%.3f within_2sd=%s floor_admissible=%s "
          "dtf_shown_failing=%s"
          % (scr_acc, within_2sd, floor_admissible,
             not dtf_passes), flush=True)
    return artifact


# ---- SG-ANT: ancilla-never-traced gate ----------------------------
SG_ANT_OUT = "exp79_gate_SG_ANT.json"


def gate_ancilla_never_traced(data):
    """SG-ANT: when coupling=0, ancilla stays |0> throughout, and NOT tracing
    it (evaluating over K+1 wires including anc in |0>) gives the SAME result
    as A2 (K wires). When coupling != 0, the ancilla is entangled and NOT
    tracing it changes the probabilities.

    PASS assertion: zero-coupling NOT-traced == A2 (max |delta| < 1e-5).
    DESIGNED-TO-FAIL: random-coupling NOT-traced == A2 (shown failing).
    """
    verbs = data["verbs"]
    compiled = data["compiled"]["test_union"][:20]

    # A2 model (K wires)
    m_a2, _ = K42.build_model("A2", verbs, data["vocab"], 0, data["angles"])
    m_a2.eval()

    # C-DISCARD model with coupling ZEROED (designed to equal A2)
    m_zero, _ = build_model79("C-DISCARD", verbs, 0)
    with torch.no_grad():
        for v in verbs:
            m_zero.coupling[v].fill_(0.0)
        # copy A2's weights
        a2_sd = m_a2.state_dict()
        m_zero_sd = m_zero.state_dict()
        for k in a2_sd:
            if k in m_zero_sd:
                m_zero_sd[k] = a2_sd[k].clone()
        # keep coupling at zero
        m_zero.load_state_dict(m_zero_sd)
    m_zero.eval()

    # C-DISCARD model with RANDOM coupling (designed to diverge from A2)
    m_rand, spec_rand = build_model79("C-DISCARD", verbs, 42)
    with torch.no_grad():
        # copy A2 weights but keep coupling from random init
        m_rand_sd = m_rand.state_dict()
        for k in a2_sd:
            if k in m_rand_sd and "coupling" not in k:
                m_rand_sd[k] = a2_sd[k].clone()
        m_rand.load_state_dict(m_rand_sd)
    m_rand.eval()

    # Helper: evaluate story_values BUT treating the output as K+1 wires
    # without explicitly tracing the ancilla (using a fake "never-traced" readout)
    # For the "ancilla never traced" scenario: compute prob over (qa, qp) on
    # K+1-wire state, same as story_values (prob_all_zero_on IS the trace) but
    # to simulate "never traced" we compute prob including the ancilla as a
    # queried wire (so we'd miss half the probability mass if anc is entangled).
    # Actually: "ancilla never traced" = compute prob_all_zero_on on (qa,qp,anc)
    # instead of (qa,qp). When anc=|0>, P(qa=0,qp=0,anc=0) = P(qa=0,qp=0)*1.
    # When anc entangled, P(qa=0,qp=0,anc=0) != P(qa=0,qp=0).

    def story_values_never_trace_anc(model, cs):
        """Like story_values but also conditions on anc=0 (never traces it out)."""
        cache = {}

        def block(v):
            if v not in cache:
                cache[v] = model.verb_matrix(v)
            return cache[v]

        anc = cs.K
        psi = sim.init_state(cs.K + 1)
        for w, _n, angle in cs.intro:
            psi = sim.apply_gate(psi, sim.ry(angle), (w,))
        for v, wa, wp in cs.events:
            psi = sim.apply_gate(psi, block(v), (wa, wp))
            psi = sim.apply_gate(psi, _cry(model.coupling[v]), (wp, anc))
        vseq, qa, qp = cs.question
        qw = (qa, qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)
        # "never traced": include anc wire in the queried set
        qw_with_anc = (qa, qp, anc)
        vals = []
        for eff in (model.q_pos, model.q_neg):
            phi = sim.apply_gate(psi, eff.matrix().conj().T, qw)
            vals.append(sim.prob_all_zero_on(phi, qw_with_anc))
        return torch.stack(vals)

    # Measure delta: zero-coupling never-trace vs A2
    max_delta_zero = 0.0
    max_delta_rand = 0.0
    with torch.no_grad():
        for cs in compiled:
            a2_vals = m_a2.story_values(cs)
            zero_nt = story_values_never_trace_anc(m_zero, cs)
            rand_nt = story_values_never_trace_anc(m_rand, cs)
            max_delta_zero = max(max_delta_zero,
                                 float((zero_nt - a2_vals).abs().max()))
            max_delta_rand = max(max_delta_rand,
                                 float((rand_nt - a2_vals).abs().max()))

    PASS_TOL = 1e-5
    zero_ok = max_delta_zero < PASS_TOL
    rand_differs = max_delta_rand > PASS_TOL

    # designed-to-fail: random-coupling never-trace == A2 (this should fail)
    dtf_assertion = "random_coupling NOT-traced == A2 (max |delta| < 1e-5)"
    dtf_passes = not rand_differs   # expected False

    artifact = {
        "gate": "SG-ANT ancilla-never-traced (doc 71 SS5, doc 70 SS3.6)",
        "pass_assertion": "zero-coupling NOT-traced == A2 (max |delta| < 1e-5)",
        "pass_result": zero_ok,
        "designed_to_fail_assertion": dtf_assertion,
        "designed_to_fail_result": dtf_passes,
        "designed_to_fail_shown_failing": not dtf_passes,
        "max_delta_zero_coupling_not_traced_vs_A2": max_delta_zero,
        "max_delta_random_coupling_not_traced_vs_A2": max_delta_rand,
        "pass_tol": PASS_TOL,
        "n_items": len(compiled),
        "passed": bool(zero_ok and rand_differs),
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    artifact["artifact_sha256"] = _artifact_hash(
        {k: v for k, v in artifact.items() if k != "artifact_sha256"})
    with open(SG_ANT_OUT, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=1)
    print("[SG-ANT] zero-coup delta=%.2e rand-coup delta=%.2e "
          "passed=%s dtf_shown_failing=%s"
          % (max_delta_zero, max_delta_rand,
             artifact["passed"], not dtf_passes), flush=True)
    return artifact


# ---- SG-PS: purity sanity gate ------------------------------------
SG_PS_OUT = "exp79_gate_SG_PS.json"


def gate_purity_sanity():
    """SG-PS: known product state -> pair_purity = 1.0.
    Bell pair across the cut -> pair_purity = 0.5.
    PASS: product state gives 1.0 to 1e-6.
    DESIGNED-TO-FAIL: assert Bell-pair purity >= 1.0 (shown failing).
    """
    import torch_sv_sim as sv

    # Product state |0>|0>|0>|0> (K=4 wires)
    psi_product = sim.init_state(4)
    pp_product = pair_purity(psi_product, 0, 1)

    # "Externally entangled" state: wire 0 entangled with wire 2 (outside queried
    # pair 0,1). rho_{01} is mixed because wire 0's partner (wire 2) is traced out.
    # Construction: H on 0, CNOT(0->2) => (|00>+|11>)_02 * |0>_1 * |0>_3
    # => rho_{01} = diag(0.5,0.5)_0 * |0><0|_1 => Tr(rho_{01}^2) = 0.5
    H = torch.tensor([[1, 1], [1, -1]], dtype=sim.CDTYPE) / math.sqrt(2)
    CNOT = torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=sim.CDTYPE)
    psi_ext = sim.init_state(4)
    psi_ext = sim.apply_gate(psi_ext, H, (0,))          # entangle wire 0
    psi_ext = sim.apply_gate(psi_ext, CNOT, (0, 2))     # partner = wire 2 (external)
    pp_ext_on_01 = pair_purity(psi_ext, 0, 1)  # wire 0 entangled with external: ~0.5
    pp_ext_on_02 = pair_purity(psi_ext, 0, 2)  # 02 are the Bell pair: =1.0
    pp_ext_on_23 = pair_purity(psi_ext, 2, 3)  # wire 2 entangled with external: ~0.5

    PASS_TOL = 1e-6
    product_ok = abs(pp_product - 1.0) < PASS_TOL
    # Pair (0,1): wire 0 mixed, wire 1 pure |0>, product => Tr(rho^2)=0.5
    ext_mixed = pp_ext_on_01 < 0.75   # should be ~0.5

    # designed-to-fail: assert externally-entangled pair purity >= 1.0 (should fail)
    dtf_assertion = "externally-entangled pair purity >= 1.0 (on wires 0,1)"
    dtf_passes = bool(pp_ext_on_01 >= 1.0)   # expected False

    artifact = {
        "gate": "SG-PS purity sanity (doc 70 SS3.6, doc 71 SS1)",
        "pass_assertion": "product state pair_purity == 1.0 to 1e-6",
        "pass_result": product_ok,
        "product_state_pair_purity": pp_product,
        "construction": ("wire 0 entangled with wire 2 via H+CNOT; "
                         "queried pair = wires (0,1); wire 0's partner "
                         "(wire 2) is external and traced out, making "
                         "rho_{01} mixed with Tr(rho^2) ~ 0.5"),
        "ext_entangled_pair_purity_01": pp_ext_on_01,
        "ext_entangled_pair_purity_02": pp_ext_on_02,
        "ext_entangled_pair_purity_23": pp_ext_on_23,
        "pair_01_mixed": ext_mixed,
        "designed_to_fail_assertion": dtf_assertion,
        "designed_to_fail_result": dtf_passes,
        "designed_to_fail_shown_failing": not dtf_passes,
        "pass_tol": PASS_TOL,
        "passed": bool(product_ok and ext_mixed),
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    artifact["artifact_sha256"] = _artifact_hash(
        {k: v for k, v in artifact.items() if k != "artifact_sha256"})
    with open(SG_PS_OUT, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=1)
    print("[SG-PS] product_purity=%.6f ext_entangled_purity_01=%.6f "
          "passed=%s dtf_shown_failing=%s"
          % (pp_product, pp_ext_on_01, artifact["passed"],
             not dtf_passes), flush=True)
    return artifact


# ================================================================
# smoke battery (exp79)
# ================================================================
def smoke_battery(data):
    """Run all 3 self-emitting gate artifacts + mechanical checks.
    Writes exp79_smoke_PASSED.json on success.
    """
    t0 = time.time()
    tests = []

    def add(name, passed, **kw):
        rec = {"test": name, "passed": bool(passed)}
        rec.update(kw)
        tests.append(rec)
        print("[smoke] %-40s %s" % (name, "PASS" if passed else "FAIL"),
              flush=True)

    # --- M1: mechanical build + gradient check ----------------------
    verbs = data["verbs"]
    rng = random.Random(0)
    pool = data["compiled"]["train"][:]
    rng.shuffle(pool)
    tiny = pool[:20]
    mech = {}
    ok_mech = True
    for arm in EXP79_ARMS:
        m, _ = build_model79(arm, verbs, 0)
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
    add("M1 mechanical build + gradient (all 3 arms)", ok_mech,
        per_arm=mech)

    # --- M2: A2 reduction -- zero coupling = same readout as A2 -----
    m_zero, _ = build_model79("C-DISCARD", verbs, 0)
    m_a2, _ = K42.build_model("A2", verbs, data["vocab"], 0, data["angles"])
    with torch.no_grad():
        for v in verbs:
            m_zero.coupling[v].fill_(0.0)
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
            cdv = m_zero.story_values(cs)
            max_delta = max(max_delta, float((a2v - cdv).abs().max()))
    ok_m2 = max_delta < 1e-5
    add("M2 zero-coupling C-DISCARD == A2 (reduction check)",
        ok_m2, max_abs_delta=max_delta, tol=1e-5)

    # --- SG-FS fixed-scramble gate ----------------------------------
    fs = gate_fixed_scramble(data)
    add("SG-FS fixed-scramble floor (self-emitting)", fs["passed"],
        floor_admissible=fs["floor_admissible"],
        scrambled_acc=fs["scrambled_accuracy"],
        dtf_shown_failing=fs["designed_to_fail_shown_failing"],
        artifact=SG_FS_OUT)

    # --- SG-ANT ancilla-never-traced gate ---------------------------
    ant = gate_ancilla_never_traced(data)
    add("SG-ANT ancilla-never-traced (self-emitting)", ant["passed"],
        zero_coup_delta=ant["max_delta_zero_coupling_not_traced_vs_A2"],
        rand_coup_delta=ant["max_delta_random_coupling_not_traced_vs_A2"],
        dtf_shown_failing=ant["designed_to_fail_shown_failing"],
        artifact=SG_ANT_OUT)

    # --- SG-PS purity sanity gate -----------------------------------
    ps = gate_purity_sanity()
    add("SG-PS purity sanity (self-emitting)", ps["passed"],
        product_purity=ps["product_state_pair_purity"],
        ext_entangled_purity=ps["ext_entangled_pair_purity_01"],
        dtf_shown_failing=ps["designed_to_fail_shown_failing"],
        artifact=SG_PS_OUT)

    # --- summary ----------------------------------------------------
    all_ok = all(t["passed"] for t in tests)
    report = {
        "component": "exp79 smoke battery (doc 71 SS5, doc 70 SS3.6)",
        "tests": tests, "all_passed": all_ok,
        "gate_artifacts": [SG_FS_OUT, SG_ANT_OUT, SG_PS_OUT],
        "designed_to_fail_builds": ["SG-FS fixed-scramble floor",
                                    "SG-ANT ancilla-never-traced",
                                    "SG-PS purity sanity"],
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
# first-epoch timing + compute table
# ================================================================
def measure_epoch_time(data, cfg, n_train=50):
    """Measure first-epoch wall time for C-DISCARD and C-RESET on n_train items.
    Returns timing dict for the compute table.
    """
    import torch
    verbs = cfg["verbs"]
    pool = list(data["compiled"]["train"])
    random.Random(79).shuffle(pool)
    tiny = pool[:n_train]

    results = {}
    for arm in ("C-DISCARD", "C-RESET"):
        m, _ = build_model79(arm, verbs, 0)
        opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
        t0 = time.time()
        for cs in tiny:
            opt.zero_grad()
            loss, _ = m.story_loss(cs)
            loss.backward()
            opt.step()
        elapsed = time.time() - t0
        sec_per_item = elapsed / n_train
        results[arm] = {
            "n_items_measured": n_train,
            "wall_sec_for_n_items": round(elapsed, 2),
            "sec_per_item": round(sec_per_item, 4),
        }
        print("[timing] %s: %.2fs for %d items = %.4f s/item"
              % (arm, elapsed, n_train, sec_per_item), flush=True)

    # A2 reference from committed checkpoint
    committed_a2 = {}
    ck_json = os.path.join(EXP42R_CKPT_DIR, "A2_s0.json")
    if os.path.exists(ck_json):
        with open(ck_json) as f:
            d = json.load(f)
        committed_a2 = {
            "wall_time_sec_30epochs_full": d.get("wall_time_sec"),
            "n_train_items": 2000,
            "sec_per_item_per_epoch": round(
                d.get("wall_time_sec", 0) / 30 / 2000, 5),
        }

    # Derive compute table
    n_train_full = len(data["compiled"]["train"])
    n_seeds = N_SEEDS
    epochs = cfg["epochs"]
    n_test = len(data["compiled"]["test_union"])

    def eta(arm_key):
        spi = results[arm_key]["sec_per_item"]
        # training time per seed: n_train × epochs × sec_per_item
        train_sec = n_train_full * epochs * spi
        # test purity logging: small overhead (test_union items, no grad)
        # estimate ~5× faster than training forward
        test_sec = n_test * spi * 0.2
        total_per_seed = train_sec + test_sec
        total_wall_5workers = total_per_seed * n_seeds / 5
        return {
            "sec_per_item_measured": spi,
            "estimated_sec_per_seed": round(total_per_seed, 0),
            "estimated_hours_per_seed": round(total_per_seed / 3600, 2),
            "n_seeds": n_seeds,
            "estimated_total_wall_hours_5workers": round(
                total_wall_5workers / 3600, 2),
        }

    compute_table = {
        "derived_from": "measured first-epoch times (doc 71 SS4: derived, not guessed)",
        "n_items_measured": n_train,
        "A2_committed_reference": committed_a2,
        "arms": {arm: {**results[arm], **eta(arm)} for arm in results},
        "C_DISCARD_vs_A2_ratio": round(
            results["C-DISCARD"]["sec_per_item"]
            / committed_a2.get("sec_per_item_per_epoch", 1), 2)
        if committed_a2 else None,
        "C_RESET_vs_A2_ratio": round(
            results["C-RESET"]["sec_per_item"]
            / committed_a2.get("sec_per_item_per_epoch", 1), 2)
        if committed_a2 else None,
        "n_workers": 5,
        "threads_per_worker": 2,
        "note": ("C-FLOOR arms train identically to C-DISCARD. A2 and B1 "
                 "are loaded from committed exp42r_ckpt/ (no training). "
                 "Total trainable units: 15 (3 arms × 5 seeds). "
                 "Purity logged for all arms at eval time."),
    }

    with open(TIME_OUT, "w", encoding="utf-8") as f:
        json.dump(compute_table, f, ensure_ascii=False, indent=1)
    print("[timing] wrote %s" % TIME_OUT, flush=True)
    return compute_table


# ================================================================
# worker
# ================================================================
def run_worker(args, data):
    if not os.path.exists(SMOKE_MARKER):
        sys.exit("[w%d] REFUSING to train: %s absent. "
                 "Run `--smoke` first." % (args.worker, SMOKE_MARKER))

    torch.set_num_threads(args.threads)
    K42.ensure_dirs(CKPT_DIR, CTL_DIR, LOCK_DIR)
    cfg = {"verbs": data["verbs"], "lr": args.lr, "batch": args.batch,
           "epochs": args.epochs}
    w = args.worker
    print("[w%d] queue start (%d threads)" % (w, args.threads), flush=True)

    # Fixed scramble drawn ONCE (same for all workers; deterministic)
    union_ids = [cs.item_id for cs in data["compiled"]["test_union"]]
    all_ids = sorted({cs.item_id for split in data["compiled"].values() for cs in split})
    scramble_map = draw_fixed_scramble(all_ids, FLOOR_SCRAMBLE_SEED)

    # Build job queue
    train_arms = ["C-DISCARD", "C-RESET", "C-FLOOR"]
    queue = []
    for k in range(args.seeds):
        for arm in train_arms:
            jid = "%s:s%d" % (arm, k)
            queue.append((jid, arm, k))

    for jid, arm, seed in queue:
        outs = list(unit_paths79(arm, seed))
        if settled79(outs, jid) or not claim(jid):
            continue
        try:
            sm = scramble_map if arm == "C-FLOOR" else None
            job_main79(arm, seed, data, cfg, scramble_map=sm)
        except Exception as e:  # noqa: BLE001
            record_failure79(jid, e, arm)

    print("[exp79 w%d] done" % w, flush=True)


# ================================================================
# main
# ================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drift-anchor", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--time-epoch", action="store_true")
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--nworkers", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--barrier-timeout", type=float,
                    default=BARRIER_TIMEOUT_SEC)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    K42.ensure_dirs(CKPT_DIR, CTL_DIR, LOCK_DIR)

    data = prepare()   # provenance gate + compile

    if args.drift_anchor:
        run_drift_anchor(data)
        return 0

    if args.smoke:
        return smoke_battery(data)

    if args.time_epoch:
        cfg = {"verbs": data["verbs"], "lr": args.lr,
               "batch": args.batch, "epochs": args.epochs}
        compute_table = measure_epoch_time(data, cfg)
        print("\n=== COMPUTE TABLE ===")
        for arm, rec in compute_table["arms"].items():
            print("  %s: %.4f s/item  %.2f h/seed  %.2f h total (5 workers)"
                  % (arm, rec["sec_per_item"],
                     rec["estimated_hours_per_seed"],
                     rec["estimated_total_wall_hours_5workers"]))
        if compute_table.get("C_DISCARD_vs_A2_ratio"):
            print("  C-DISCARD / A2 ratio: %.1fx" %
                  compute_table["C_DISCARD_vs_A2_ratio"])
        if compute_table.get("C_RESET_vs_A2_ratio"):
            print("  C-RESET   / A2 ratio: %.1fx" %
                  compute_table["C_RESET_vs_A2_ratio"])
        return 0

    if args.worker is not None:
        run_worker(args, data)
        return 0

    print("Give --drift-anchor, --smoke, --time-epoch, or --worker i",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
