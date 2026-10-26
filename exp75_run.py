# -*- coding: utf-8 -*-
"""exp75_run.py -- THE RE-UPLOADING RUNG (doc 62, D2 of doc 59's six-direction plan).

REGISTRATION: qnlp_private_docs/62_exp75_reuploading.md (REGISTERED 2026-08-28,
orchestrator rulings 1-7 in the status header, BINDING over the body).

THE QUESTION (doc 62 SS0.1). If a referent's single scalar angle is uploaded L
times instead of once, with trainable processing between the uploads -- the
field's canonical construction for making a small circuit expressive -- does the
added frequency supply convert into trained held-out accuracy on this
programme's composition task?

WHAT THIS IS (doc 62 SS0.2): E4's REPAIR. `results_exp60.json` recorded
`fourier_arm_validity = false` and doc 46 SS10.5 had pre-registered that this
makes every S-reup result inadmissible. `reuploading_relieves = FALSE` IS NOT
EVIDENCE THAT RE-UPLOADING DOES NOT HELP. Three defects are repaired here:

  * E4's trainable scalar never moved (kappa: min -0.0201, max +0.0780, mean
    +0.0064 over 34 runs). Here KAPPA IS FROZEN AT 1 AND IS NOT A PARAMETER
    (SS4). The uploads are STRUCTURAL. What can still be inert is the interleave
    block V_j, and SS9 makes its movement a BLOCKING, TWO-LEVEL, MEASURED gate
    with a bar (0.10 rad) ABOVE E4's most-moved run.
  * E4's validity basis was mis-specified. Here the basis is DERIVED (SS2.1):
    under the kappa freeze the output lives in span{1, cos k.theta, sin k.theta :
    k = 1..L}, exactly 2L+1 real terms, and it is gated at smoke IN BOTH
    DIRECTIONS (SS8.2, gates C1-C5) -- including exp69's own `resid_3term < 1e-6`
    assertion RUN WITH ITS SIGN FLIPPED.
  * E4's arm was run where the baseline was at ceiling. Here the committed
    single-angle family sits at 0.9187-0.9227 held-out with TRAIN accuracy
    0.892-0.941 against a family bound of 0.9625 -- there is residual training
    loss for a richer model to consume (SS1.2), and this is the ONLY committed
    family in the programme whose grade-1 bound is not 1.0000 (SS1.3).

THE INTERVENTION, AND IT IS DELIBERATELY SMALL (doc 62 SS0.4, SS7). Per referent
wire the committed single `RY(theta)` becomes

    U_L(theta) = RY(theta) . V_{L-1} . RY(theta) . ... . V_1 . RY(theta)

with EVERY upload carrying the SAME angle theta (kappa = 1) and `V_j =
Circuit4(1, 3)` -- the COMMITTED ansatz at one qubit, 6 parameters, CRX cascade
empty by construction. The blocks are SHARED across referent wires (SS7.2). That
is +6 parameters at L = 2 and +12 at L = 3 on a committed 120-parameter model:
+5.0 % and +10.0 %, on a 2-wire family whose committed composition cost is
+0.0040. NO NEW ANSATZ (doc 55 SS4.7). NO NEW FEATURIZER. NO NEW GENERATOR.
NO NEW BOUNDS INSTRUMENT. Everything else is the committed recipe UNCHANGED.

NO DIRECTIONAL CALL IS PLACED (doc 62 SS13), and the declined candidate
(`reuploading_lifts_T2 = FALSE`) is recorded in SS21 item 5 as safe-by-
construction rather than falsifiable.

SCOPE, INHERITED AND NON-NEGOTIABLE (doc 62 SS0.3 / SS16): NO EXP75 NUMBER MAY
APPEAR IN A SENTENCE CONTAINING "ADVANTAGE" (doc 52 SS1; doc 56 SS0.1; doc 58
SS12; doc 61 SS0.3). The model's output is a degree-L trigonometric polynomial
in two scalars and the span-matched classical reference is a linear model on 25
or 49 features.

THE SPLIT PATTERN (doc 62 SS20.4): THIS FILE BUILDS, GATES AND TRAINS. IT WRITES
NO RESULTS FILE except the smoke report and the anchor gate. `exp75_aggregate.py`
is the ONLY writer of `results_exp75.json`.

    python exp75_run.py --smoke                    # doc 62 SS8 / SS9.2, ALL gates
    python exp75_run.py --worker i --nworkers 5    # anchor jobs are ordered FIRST
    python exp75_run.py --anchor-report            # writes results_exp75_anchors.json
    python exp75_comparator.py                     # doc 62 SS10, tiers (i)/(ii)/(iii)
    python exp75_aggregate.py                      # the ONLY results writer
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

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)                              # doc 45 SS6.1, verified inert

import exp42_compiler as comp
import exp42_sim_adapter as sim
import exp68_run as E                                 # committed feature maps, config
import exp69_run as R69                               # compile path, floors, gates
import exp72_run as R72                               # doc 58 SS6.2 identity init
import instrument_sb_bounds as INS
from exp42_models import QuantumStoryModel
from exp42_run import grad_global_norm
from exp60_run import evaluate

SPEC = ("qnlp_private_docs/62_exp75_reuploading.md (REGISTERED 2026-08-28, "
        "orchestrator rulings 1-7 in the status header, BINDING over the body)")
SPEC_DOC = "62_exp75_reuploading.md"

DATA_A = R69.DATA_A                                   # stories_exp69_QA.json, FROZEN
EXP69_RESULTS = "results_exp69.json"
EXP69_VSUITE = "results_exp69_WN_validation.json"

CKPT_DIR = "exp75_ckpt"
SMOKE_OUT = "results_exp75_smoke_validation.json"
ANCHOR_OUT = "results_exp75_anchors.json"
VSUITE = EXP69_VSUITE

# doc 62 SS8.5 -- the frozen dataset, asserted at smoke and never assumed.
DATASET_SHA_DOC62 = ("2b85822b544571969fd40cab12b03a4ac256b08aee914d2b9643b856"
                     "a6308c96")

# Training config: the exp60/exp61/exp65/exp68/exp69/exp70/exp72/exp73 loop,
# UNCHANGED (doc 62 SS6.2). Anything else here would put the drift anchor's
# bit-exactness out of reach.
LR, BATCH, MAX_EPOCHS = E.LR, E.BATCH, E.MAX_EPOCHS    # 0.02, 8, 150
TRAIN_GATE, VAL_GATE = E.TRAIN_GATE, E.VAL_GATE
RNG_BASE = E.RNG_BASE
N_SEEDS = E.N_SEEDS

ANCHOR_TOLERANCE = 0.002              # doc 62 SS6.3; honest expectation is 0.0
LAYERS = 3                            # `Circuit4(n, 3)`, the committed depth

# doc 58 SS6.2, IMPORTED from the committed module and never re-typed.
IDENTITY_SIGMA = R72.IDENTITY_SIGMA
IDENTITY_RNG_BASE = R72.IDENTITY_RNG_BASE

# THE INTERLEAVE BLOCKS' OWN DRAW STREAM. doc 62 SS7.4 registers the DISTRIBUTION
# (`V_j ~ U[0, 2pi)`, "the Circuit4 constructor's own draw") but not the stream.
# A DEDICATED generator is used so that NOT ONE COMMITTED DRAW MOVES -- the verb
# blocks and the readout are produced by `QuantumStoryModel.__init__`'s own
# generator exactly as the committed recipe produces them, which is doc 58
# SS6.2's discipline applied to a second block family. At L = 1 no V block is
# constructed at all, so the drift anchor is bit-exact BY STRUCTURE, not by
# tolerance. FLAGGED as a free choice in the launch report.
REUP_RNG_BASE = 7_500_000
REUP_IDENTITY_RNG_BASE = 7_600_000    # the identity-init stream for V_j (SS7.4)

# doc 62 SS9.3 -- THE ENGAGEMENT BAR. Chosen where the failure it exists to catch
# is caught: E4's kappa moved by max +0.0780 across all 34 S-reup runs (mean
# +0.0064, min -0.0201), so 0.10 rad would have failed EVERY ONE of them,
# including the most-moved; exp57's ROT arm moved 5.94e-05 rad, three orders
# below. ABSOLUTE, PER RUN, on the maximum over the block's 6 or 12 parameters,
# and NOT a mean over seeds (orchestrator ruling 3 ACCEPTS it as calibrated).
ENGAGEMENT_BAR = 0.10

# doc 62 SS9.2 -- the two level-1 bars.
M1_GRAD_BAR = 1e-8
M2_MOVE_BAR = 0.01
M2_PROBE_EPOCHS = 10

# doc 62 SS8.2 -- the closure thresholds, per gate.
CLOSE_TOL = 1e-6                      # C1 / C4 "closes"
OPEN_TOL = 1e-3                       # C2 / C3 / C4 "does not close"
HARMONIC_ENERGY_BAR = 1e-8            # C5, live harmonics at the uniform draw
# C5's failing direction says the k < L coefficients must be "EXACTLY 0" at
# V = 0. On complex64 forwards a least-squares fit returns absent-harmonic
# coefficients at the platform floor (~1e-8 in amplitude), so the assertion is
# operationalised on the harmonic ENERGY (a_k^2 + b_k^2) at 1e-12 and the raw
# value is REPORTED. FLAGGED in the launch report as an operationalisation.
HARMONIC_ZERO_BAR = 1e-12
F64_PURITY_TOL = 1e-9                 # doc 62 SS8.4 item 3, asserted in complex128
F32_FLOOR = 1e-6                      # the complex64 agreement floor (exp70/exp73)
# doc 62 SS8.4 item 3 registers "per-wire intro purity = 1.0 to 1e-9". THAT
# THRESHOLD IS ASSERTED, AS REGISTERED, ON A COMPLEX128 RECOMPUTATION of the same
# gate sequence from the model's own parameters (`intro_state_f64`). The RUNTIME
# forward is complex64 and its measured intro-purity deviation is ~1e-6 at L = 3
# -- accumulation over 2L-1 single-qubit gates, not a circuit effect -- so the
# runtime value is gated at a documented SANITY bound and REPORTED alongside. The
# discrimination is not weakened: the registered failing direction (an entangler
# in the intro) moves the purity by > 1e-3, three orders above this bound.
# FLAGGED in the launch report as an operationalisation (exp70/exp73 precedent).
F32_PURITY_SANITY = 1e-4

SWEEP_POINTS = 48                     # the committed `fourier_resid_1d` frame
SWEEP_STORIES = 4

# doc 62 SS7.3 -- the parameter arithmetic, asserted and never trusted from here.
N_PARAMS_BY_L = {1: 120, 2: 126, 3: 132}

# ------------------------------------------------------------------ the arms
# doc 62 SS6.1: 2 depths x 2 inits x 2 composition lengths = 8 CELLS, 5 seeds.
UPLOADS = (2, 3)
INITS = ("uniform", "identity")
TS = (1, 2)

# doc 62 SS6.3 -- THE DRIFT ANCHOR, BLOCKING, TWO ARMS, AND IT RUNS FIRST. R1-A
# and R2-A re-run through exp75's OWN codepath with the re-uploading intro OFF
# (L = 1). doc 62 SS6.3.1: BOTH baselines are anchored because SS12.4's lift
# booleans read the best cell at each T against THAT T's committed mean, so both
# committed means are load-bearing (doc 61 SS5.3 anchored one; this rung needs two).
ANCHOR_ARMS = ("A0-A-T1", "A0-A-T2")
ANCHOR_SOURCE = {"A0-A-T1": "R1-A", "A0-A-T2": "R2-A"}
ANCHOR_T = {"A0-A-T1": 1, "A0-A-T2": 2}
# doc 62 SS6.3's printed per-seed table. ASSERTED EQUAL to results_exp69.json at
# smoke (SS8.6) and never used in its place.
ANCHOR_PER_SEED_DOC62 = {
    "A0-A-T1": [0.903333, 0.936667, 0.926667, 0.920000, 0.926667],
    "A0-A-T2": [0.916667, 0.946667, 0.873333, 0.913333, 0.943333],
}
ANCHOR_MEAN_DOC62 = {"A0-A-T1": 0.9226667, "A0-A-T2": 0.9186667}

# doc 62 SS1.1 / SS8.6 -- the committed values, READ PROGRAMMATICALLY and asserted
# equal to the doc's printed table rather than re-typed into a criterion.
COMMITTED_DOC62_SS1_1 = {
    "R1-A": {"T": 1, "mean": 0.9226667, "sd": 0.0123, "floor_mean": 0.5120,
             "floor_sd": 0.0461, "comparator_9dim": 0.9200, "bound": 0.9625},
    "R2-A": {"T": 2, "mean": 0.9186667, "sd": 0.0295, "floor_mean": 0.5060,
             "floor_sd": 0.0218, "comparator_9dim": 0.9200, "bound": 0.9625},
}
COMMITTED_CC_A = 0.0040               # doc 55 SS18.1, the ladder's first rung

ARM_CFG = {}
for _L in UPLOADS:
    for _t in TS:
        for _i in INITS:
            ARM_CFG["RU-L%d-T%d-%s" % (_L, _t, _i)] = {
                "enc": "FA", "L": _L, "T": _t, "init": _i, "part": "A",
                "family": "RU-L%d-%s" % (_L, _i),
                "n_params": N_PARAMS_BY_L[_L],
                "role": ("re-uploading arm, L = %d uploads, %s init "
                         "(doc 62 SS6.1)" % (_L, _i))}
for _a in ANCHOR_ARMS:
    ARM_CFG[_a] = {
        "enc": "FA", "L": 1, "T": ANCHOR_T[_a], "init": "uniform", "part": "A",
        "family": "A0-A", "n_params": N_PARAMS_BY_L[1],
        "role": ("THE DRIFT ANCHOR (doc 62 SS6.3) -- the committed single-angle "
                 "arm %s at T = %d, through exp75's OWN train() entry point with "
                 "the re-uploading intro OFF (L = 1). BLOCKING, and it runs "
                 "first." % (ANCHOR_SOURCE[_a], ANCHOR_T[_a]))}

TRAINED_CELLS = ["RU-L%d-T%d-%s" % (L, t, i)
                 for L in UPLOADS for t in TS for i in INITS]
FLOOR_ARMS = ["S-shuffle-%s" % a for a in TRAINED_CELLS]
FLOOR_OF = {a: "S-shuffle-%s" % a for a in TRAINED_CELLS}
ALL_ARMS = list(ANCHOR_ARMS) + TRAINED_CELLS + FLOOR_ARMS   # 18 arms, 90 runs


def base_arm(arm):
    return arm[len("S-shuffle-"):] if arm.startswith("S-shuffle-") else arm


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def uploads_of(arm):
    return ARM_CFG[base_arm(arm)]["L"]


# ============================================================ the feature layer
def feature_maps():
    """doc 62 SS6.2: the committed FA feature map, IMPORTED. exp75 introduces NO
    featurizer and NO new angle -- the encoding ANGLE is the committed
    `phi_exp42`; what is new is how many times the SAME angle is applied
    (SS20.1)."""
    return R69.feature_maps()


def arm_angles(arm, fm):
    """Every exp75 arm is FA. `exp69_run.arm_angles` is the source of truth for
    the committed maps; exp75's arms are not in its ARM_CFG, so the FA branch is
    named here and ASSERTED bit-identical to it at smoke (SS8.4 item 5)."""
    assert ARM_CFG[base_arm(arm)]["enc"] == "FA", arm
    return fm["phi_exp42"], None


def referent_features(arm, fm):
    """The 3-term per-referent u-vector of the COMMITTED single encode -- tier
    (iii)'s frame, kept here so `exp75_comparator.py` can widen it to u_L."""
    a0, _a1 = arm_angles(arm, fm)
    return {n: np.array(R69._u(a0[n])) for n in INS.HUMANS}


# ================================================================== the models
def _intro_spec(L):
    """doc 62 SS7.1, DECLARATIVE. Per referent wire, in order:

        for j = 1 .. L:   RY(theta_w) ;  if j < L:  V_j

    Returned as a tuple of (kind, index) so SS8.4 item 1's gate-count assertion
    reads a SPEC rather than a comment, and so a broken build (L +- 1 uploads)
    is a one-line perturbation with a failing direction."""
    seq = []
    for j in range(1, L + 1):
        seq.append(("RY", 0))
        if j < L:
            seq.append(("V", j))
    return tuple(seq)


class ReuploadModel(QuantumStoryModel):
    """doc 62 SS7 -- THE COMMITTED A3 MODEL WITH A RE-UPLOADING INTRO.

    `QuantumStoryModel('A3', verbs, seed)` is the parent, constructed VERBATIM:
    six `Circuit4(2, 3)` verb blocks then `q_pos` then `q_neg`, from ITS OWN
    generator seeded by `seed`. Not one committed draw moves. The interleave
    blocks are `Circuit4(1, 3)` -- the committed ansatz at one qubit, where the
    CRX cascade `range(n-2, -1, -1)` is EMPTY by construction and the block is
    RX.RZ repeated three times, 6 parameters (SS3.1, SS7.1).

    THE BLOCKS ARE SHARED ACROSS REFERENT WIRES (SS7.2): one parameter tensor per
    layer index j, applied to EVERY referent wire. Per-wire blocks are refused
    because the wire index is an arbitrary compiler-assigned referent slot, not a
    semantic role, and the committed intro has ZERO per-wire parameters.

    AT L = 1 `self.reup` IS EMPTY. The parameter list, its order, the RNG
    consumption and the forward gate sequence are then bit-identical to the
    committed model -- which is why SS6.3's drift anchor is bit-exact BY
    STRUCTURE and not by tolerance (SS8.3: the anchor runs through THIS entry
    point, not a legacy path).

    THE INTRO HAS NO DATA-DEPENDENT PARAMETERS AND CONSUMES NO RNG ON THE DATA
    PATH (SS8.4 item 6). kappa = 1 is FIXED and is not a parameter anywhere in
    this class (SS4.1): there is no quantity that can be initialised at zero and
    stay there, which is E4's failure mode removed at the level of the build."""

    def __init__(self, verbs, seed, L, layers=LAYERS):
        super().__init__("A3", verbs, seed)
        self.L = int(L)
        self.layers = int(layers)
        gv = torch.Generator().manual_seed(REUP_RNG_BASE + int(seed))
        self.reup = nn.ModuleList(
            [sim.Circuit4(1, self.layers, generator=gv)
             for _ in range(self.L - 1)])
        self.intro_spec = _intro_spec(self.L)

    # -- the intro ---------------------------------------------------------
    def v_matrix(self, j, _wire):
        """`V_j` on ANY referent wire -- the SAME parameter tensor (SS7.2)."""
        return self.reup[j - 1].matrix()

    def upload_scale(self, _j):
        """kappa, FROZEN AT 1 (doc 62 SS4.1). Every upload injects the referent's
        own committed FA angle, unscaled and unshifted. This is a method rather
        than a literal so that SS8.2's failing direction (a non-unit prefactor
        must break C1) is a one-line override on a BROKEN subclass and never a
        switch on the production path."""
        return 1.0

    def intro_gate_trace(self, cs):
        """The gates the intro ACTUALLY applies, as (wire, kind, index). SS8.4
        item 1 counts THIS, not the spec, so the count assertion reads the
        execution path."""
        out = []
        for w, _noun, _angle in cs.intro:
            for kind, j in self.intro_spec:
                out.append((w, kind, j))
        return out

    def apply_intro(self, psi, cs):
        """THE SINGLE SOURCE OF TRUTH for the exp75 intro."""
        for w, _noun, angle in cs.intro:
            for kind, j in self.intro_spec:
                if kind == "RY":
                    psi = sim.apply_gate(
                        psi, sim.ry(float(angle) * self.upload_scale(j)), (w,))
                else:
                    psi = sim.apply_gate(psi, self.v_matrix(j, w), (w,))
        return psi

    # -- the forward pass --------------------------------------------------
    def story_values(self, cs, verb_map=None):
        """`exp42_models.QuantumStoryModel.story_values` with the intro replaced.
        The events + question path is the committed one GATE FOR GATE; at L = 1
        the intro loop applies exactly one `RY` per referent and the whole pass is
        bit-identical to the parent's (asserted at smoke, SS8.3)."""
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)
            return cache[v]

        psi = self.apply_intro(sim.init_state(cs.K), cs)
        for v, wa, wp in cs.events:
            psi = sim.apply_gate(psi, block(v), (wa, wp))
        vseq, qa, qp = cs.question
        qw = (qa, qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)
        return torch.stack([sim.prob_all_zero_on(
            sim.apply_gate(psi, eff.matrix().conj().T, qw), qw)
            for eff in (self.q_pos, self.q_neg)])

    # -- SS9's raw quantity -------------------------------------------------
    def v_theta(self):
        """The interleave blocks' parameter vector, flat, in layer order. Empty
        at L = 1. This is SS9.3's raw quantity and it is checkpointed at init, at
        the SELECTED epoch and at the FINAL epoch."""
        if not len(self.reup):
            return []
        return [float(x) for x in torch.cat(
            [b.theta.detach().reshape(-1) for b in self.reup])]

    def verb_theta(self):
        """The verb blocks' parameter vector -- SS11.4's contrast quantity."""
        return [float(x) for x in torch.cat(
            [self.blocks[v].theta.detach().reshape(-1) for v in sorted(self.verbs)])]


def build_model(arm, verbs, seed, _fm):
    """doc 62 SS8.3: ONE entry point. The anchor is `ReuploadModel(..., L = 1)`,
    which constructs the committed parent verbatim and adds nothing."""
    return ReuploadModel(verbs, seed, uploads_of(arm))


def apply_init_scheme(m, arm, seed):
    """doc 62 SS7.4 -- BOTH INITS ARE ARMS, and on THIS rung they are DIFFERENT
    CIRCUITS AT t = 0 (SS3.3), not two draws of one circuit.

    `uniform`  : V_j ~ U[0, 2pi), the committed recipe -- `build_model` has
                 already produced it, and the circuit begins with its full
                 frequency supply live.
    `identity` : the VERB blocks are re-drawn N(0, 0.1^2) by the COMMITTED code
                 object `exp72_run.apply_identity_init` (doc 58 SS6.2 verbatim,
                 same generator, same sigma, same sorted draw order, same readout
                 exclusion), and `V_j` is drawn N(0, 0.1^2) from its OWN
                 dedicated stream. At exact identity the circuit is RY(L.theta)
                 (SS3.2) -- an approximate SINGLE ENCODE at a rescaled angle,
                 whose intermediate harmonics k = 1..L-1 are ABSENT at O(1) and
                 present only at O(sigma).

    THE READOUT IS UNTOUCHED in both cases (doc 58 SS6.2.2 / doc 62 SS7.4): two
    readout blocks at identity would be IDENTICAL, v_yes == v_no for every item,
    and the model would output exactly 0.5 everywhere -- a NULL INSTRUMENT."""
    cfg = ARM_CFG[base_arm(arm)]
    if cfg["init"] == "uniform":
        return {"scheme": "uniform",
                "recipe": ("every trainable block ~ Uniform[0, 2pi) -- the "
                           "committed recipe; V_j from the Circuit4 "
                           "constructor's own draw on a DEDICATED stream so no "
                           "committed draw moves (doc 62 SS7.4)"),
                "committed_stream_untouched": True,
                "v_reinitialised": False}
    # doc 62 SS24.3 -- BOUNDED REDRAW on the committed window's own sampling
    # noise: attempt k draws at effective seed `seed + 100000*k`; attempt 0 is
    # byte-for-byte the pre-amendment behaviour, so passing seeds are UNCHANGED
    # BY CONSTRUCTION. Exhaustion is a STOP (exp69_datagen's MAX_DRAWS rule).
    REDRAW_MAX, REDRAW_STRIDE = 20, 100_000
    refusals = []
    rep = touched = None
    for k in range(REDRAW_MAX):
        eff = int(seed) + REDRAW_STRIDE * k
        try:
            touched = R72.apply_identity_init(m, eff, m.verbs, [])
            rep = R72.identity_assertions(m, m.verbs, [], eff)
            gv = torch.Generator().manual_seed(REUP_IDENTITY_RNG_BASE + eff)
            with torch.no_grad():
                for blk in m.reup:
                    blk.theta.copy_(torch.randn(blk.theta.shape, generator=gv)
                                    * IDENTITY_SIGMA)
            rep.update(identity_v_assertions(m))
            break
        except AssertionError as e:
            refusals.append({"attempt": k, "effective_seed": eff,
                             "refusal": str(e)})
            rep = None
    if rep is None:
        raise RuntimeError(
            "doc 62 SS24.3 STOP: %d consecutive identity draws refused by the "
            "committed window for arm %s seed %d -- this is no longer sampling "
            "noise. Refusals: %r" % (REDRAW_MAX, arm, seed, refusals))
    rep["redraw_doc62_SS24_3"] = {
        "attempts_used": len(refusals) + 1,
        "refused_attempts": refusals,
        "effective_seed_of_record": int(seed) + REDRAW_STRIDE * len(refusals)}
    rep.update({"scheme": "identity", "blocks_reinitialised": touched,
                "v_reinitialised": bool(len(m.reup)),
                "recipe": ("verb blocks ~ N(0, 0.1^2) via the COMMITTED "
                           "`exp72_run.apply_identity_init`; V_j ~ N(0, 0.1^2) "
                           "from a dedicated stream; READOUT (q_pos, q_neg) "
                           "UNCHANGED at Uniform[0, 2pi) -- doc 58 SS6.2 "
                           "verbatim, doc 62 SS7.4"),
                "readout_exclusion_reason": (
                    "doc 58 SS6.2.2 / doc 62 SS7.4: two readout blocks at "
                    "identity would be IDENTICAL, v_yes == v_no for every item, "
                    "the model would output exactly 0.5 everywhere, and the "
                    "readout would be a NULL INSTRUMENT at initialisation."),
                "spectral_reading_doc62_SS3_3": (
                    "ON THIS RUNG THE INIT CHANGES THE SPECTRAL CONTENT OF THE "
                    "INITIAL MODEL, not merely the effective depth at the first "
                    "update. At V = 0 exactly, RY(a).RY(b) = RY(a+b) and the "
                    "circuit IS RY(L.theta): the accessible spectrum collapses "
                    "from {0, +-1, ..., +-L} to {0, +-L} and the intermediate "
                    "harmonics are ABSENT, not small. doc 62 SS3.5 therefore "
                    "FORBIDS merging SS12.5's init booleans with doc 58 SS19.1's "
                    "or doc 61 SS10.4's."),
                "imported_from": "exp72_run.apply_identity_init (doc 58 SS6.2)"})
    return rep


def identity_v_assertions(m):
    """doc 62 SS8.4 item 4, the V_j half.

    (a) `Circuit4(1, 3)` at theta = 0 is EXACTLY the identity (SS3.1) -- there
        are no fixed entanglers and rx(0) = rz(0) = I with no global phase.
    (b) THE SD WINDOW. doc 62 SS8.4 item 4 asks for a realised sample sd in
        [0.08, 0.12] "over verb blocks AND V_j". The committed
        `exp72_run.identity_assertions` already gates the VERB-BLOCK pool; the
        window is applied here to the COMBINED pool (90 + 6 or 12 parameters).
        IT IS DELIBERATELY NOT APPLIED TO V_j ALONE: at n = 6 the sample sd of a
        N(0, 0.1^2) draw leaves [0.08, 0.12] a large fraction of the time, so a
        V-only window would be a threshold on the draw's own sampling noise and
        would kill admissible runs mid-flight. The V-only sd is MEASURED AND
        REPORTED, un-gated. FLAGGED as a reading in the launch report."""
    out = {}
    worst_id = 0.0
    with torch.no_grad():
        for blk in m.reup:
            keep = blk.theta.detach().clone()
            blk.theta.zero_()
            M = blk.matrix()
            eye = torch.eye(M.shape[0], dtype=M.dtype)
            worst_id = max(worst_id, float((M - eye).abs().max()))
            blk.theta.copy_(keep)
    assert worst_id <= 1e-9, \
        ("doc 62 SS3.1 / SS8.4 item 4 FAILED: an interleave block at theta = 0 is "
         "not the identity (max |V - I| = %.3e). SS3.2's RY(a)RY(b) = RY(a+b) "
         "reading of the identity arm would be void." % worst_id)
    v = ([] if not len(m.reup) else
         torch.cat([b.theta.detach().reshape(-1) for b in m.reup]))
    pool = torch.cat([m.blocks[x].theta.detach().reshape(-1)
                      for x in sorted(m.verbs)]
                     + ([] if not len(m.reup) else [v]))
    sd_pool = float(pool.std(unbiased=True))
    assert 0.08 <= sd_pool <= 0.12, \
        ("the realised identity init has combined per-parameter sd %.4f, outside "
         "the registered [0.08, 0.12] for sigma = 0.1 (doc 62 SS8.4 item 4)"
         % sd_pool)
    out["v_block_identity_at_zero_max_abs"] = worst_id
    out["realised_combined_sd"] = sd_pool
    out["realised_v_only_sd"] = (float(v.std(unbiased=True))
                                 if len(m.reup) > 1 or
                                 (len(m.reup) == 1 and v.numel() > 1) else None)
    out["sd_window_reading_doc62_SS8_4_4"] = (
        "the [0.08, 0.12] window is applied to the COMBINED verb + V pool, and "
        "the V-only sd is reported UN-GATED: at n = 6 the sample sd of a "
        "N(0, 0.1^2) draw leaves the window often enough that a V-only gate "
        "would encode sampling noise rather than a property of the build.")
    return out


# ================================================= doc 62 SS11.1 -- the traces
def wire_purities(psi, nwires):
    """Tr rho^2 per wire -- the COMMITTED trace (doc 55 SS18.3), imported."""
    return R69.wire_purities(psi, nwires)


def forward_traced(m, cs):
    """doc 62 SS11.1, inherited verbatim from doc 55 SS18.3: the per-wire purity
    trace at intro, after event 1, after event 2 and at readout, taken FROM THE
    SAME FORWARD PASS that produces the answer. `check_trace()` asserts it
    reproduces `story_values` BIT-FOR-BIT before any number is read.

    Intro purity is 1.0 BY CONSTRUCTION here (SS7.5): only single-qubit gates act
    on individual wires, so the referents are still introduced as a product state
    and doc 56's concurrence machinery has nothing to gate. The three later
    stages carry the story."""
    cache = {}

    def block(v):
        if v not in cache:
            cache[v] = m.verb_matrix(v)
        return cache[v]

    nw = cs.K
    psi = m.apply_intro(sim.init_state(nw), cs)
    trace = {"after_intro": wire_purities(psi, nw)}
    for k, (v, wa, wp) in enumerate(cs.events):
        psi = sim.apply_gate(psi, block(v), (wa, wp))
        trace["after_event%d" % (k + 1)] = wire_purities(psi, nw)
    vseq, qa, qp = cs.question
    qw = (qa, qp)
    for qv in reversed(vseq):
        psi = sim.apply_gate(psi, block(qv).conj().T, qw)
    trace["at_readout"] = wire_purities(psi, nw)
    vals = torch.stack([sim.prob_all_zero_on(
        sim.apply_gate(psi, eff.matrix().conj().T, qw), qw)
        for eff in (m.q_pos, m.q_neg)])
    return vals, trace


def evaluate_with_traces(m, compiled):
    rows, correct = [], 0
    with torch.no_grad():
        for cs in compiled:
            vals, tr = forward_traced(m, cs)
            pred = int(torch.argmax(vals).item())
            ok = int(pred == cs.answer)
            correct += ok
            r = dict(cs.meta)
            r.update({"pred": pred, "target": int(cs.answer), "correct": bool(ok),
                      "p_yes": float(vals[0]), "p_no": float(vals[1]),
                      "purity": tr})
            rows.append(r)
    return correct / len(compiled), rows


# ================================ the complex128 recomputation (doc 62 SS8.4.3)
def _ry64(t):
    c, s = math.cos(t / 2.0), math.sin(t / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rx64(t):
    c, s = math.cos(t / 2.0), math.sin(t / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _rz64(t):
    return np.array([[np.exp(-0.5j * t), 0.0], [0.0, np.exp(0.5j * t)]],
                    dtype=np.complex128)


def circuit4_1q_matrix64(theta):
    """`torch_sv_sim.Circuit4(1, n_layers).matrix()` in complex128. Per layer the
    committed loop applies RX then RZ (the CRX cascade is empty at n = 1), each
    LEFT-multiplied, so the layer matrix is RZ(p1) . RX(p0). Asserted equal to
    the committed code object's own matrix at smoke, so this is a verified
    re-derivation and not a second implementation of record."""
    th = np.asarray(theta, dtype=float)
    M = np.eye(2, dtype=np.complex128)
    for layer in range(th.shape[0]):
        p = th[layer]
        M = _rx64(float(p[0])) @ M
        M = _rz64(float(p[1])) @ M
    return M


def _apply64(vec, U, wires, nw):
    """`torch_sv_sim.apply_gate`'s contraction, in complex128 (exp70's helper)."""
    k = len(wires)
    T = vec.reshape((2,) * nw)
    Ut = U.reshape((2,) * (2 * k))
    T = np.tensordot(Ut, T, axes=(list(range(k, 2 * k)), list(wires)))
    T = np.moveaxis(T, list(range(k)), list(wires))
    return T.reshape(-1)


def intro_state_f64(m, cs):
    """`apply_intro` in complex128, from the model's OWN parameters, through the
    SAME gate sequence -- so doc 62 SS8.4 item 3's 1e-9 purity threshold can be
    asserted AS REGISTERED instead of being loosened to the complex64 floor."""
    nw = cs.K
    v = np.zeros(2 ** nw, dtype=np.complex128)
    v[0] = 1.0
    vmat = [circuit4_1q_matrix64(b.theta.detach().numpy()) for b in m.reup]
    for w, _noun, angle in cs.intro:
        for kind, j in m.intro_spec:
            if kind == "RY":
                v = _apply64(v, _ry64(float(angle) * m.upload_scale(j)), (w,), nw)
            else:
                v = _apply64(v, vmat[j - 1], (w,), nw)
    return v


def wire_purities64(v, nw):
    T = np.asarray(v).reshape((2,) * nw)
    out = []
    for w in range(nw):
        p = np.moveaxis(T, w, 0).reshape(2, -1)
        rho = p @ p.conj().T
        out.append(float((np.abs(rho) ** 2).sum()))
    return out


# ================================================= data loading / compilation
def load_part(path=None):
    path = path or DATA_A
    blob = json.load(open(path, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("Part A provenance FAILED for %s" % path)
    return items, meta, path


def arm_items(arm, store):
    cfg = ARM_CFG[base_arm(arm)]
    items, meta, path = store
    sel = [i for i in items if int(i.get("T", 1)) == cfg["T"]]
    assert sel, "no items at T=%d in %s" % (cfg["T"], path)
    return sel, meta, path


def compile_arm(arm, items, seed, fm):
    """doc 62 SS18.1's floor is the COMMITTED S-shuffle, IMPORTED
    (`exp69_run.relabel_shuffle_q`) and never re-implemented."""
    ang, _t1 = arm_angles(arm, fm)
    use = R69.relabel_shuffle_q(items, seed) if arm.startswith("S-shuffle") else items
    return {s: R69.compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}


# ============================== doc 62 SS8.2 -- THE WIDENED CLOSURE INSTRUMENT
def fourier_fit_1d(m, cs_list, ks, n_points=SWEEP_POINTS, n_stories=SWEEP_STORIES):
    """doc 62 SS8.2: THE COMMITTED `exp69_run.fourier_resid_1d` FRAME with the
    basis WIDENED to an arbitrary harmonic set.

    A `n_points`-point sweep of ONE referent's intro angle over [0, 2pi) on
    `n_stories` held-out stories, least-squares fit in {1} u {cos k.theta,
    sin k.theta : k in ks}, worst residual over stories. 48 points support
    harmonics to 23, so L <= 3 is comfortably resolved and the instrument is
    unchanged in every other respect.

    THE PARITY ASSERTION (SS8.2's discipline): at ks = [1] this function returns
    EXACTLY what `exp69_run.fourier_resid_1d(m, cs_list, 3)` returns -- asserted
    bit-for-bit at smoke, so the widened instrument is provably the committed one
    and not a second implementation."""
    ts = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    worst = 0.0
    energies = []
    for cs in cs_list[:n_stories]:
        ys = []
        for t in ts:
            c2 = copy.deepcopy(cs)
            c2.intro[0] = (c2.intro[0][0], c2.intro[0][1], float(t))
            with torch.no_grad():
                ys.append(float(m.story_values(c2)[0]))
        cols = [np.ones_like(ts)]
        for k in ks:
            cols += [np.cos(k * ts), np.sin(k * ts)]
        A = np.stack(cols, 1)
        ysa = np.array(ys)
        coef, *_ = np.linalg.lstsq(A, ysa, rcond=None)
        worst = max(worst, float(np.abs(A @ coef - ysa).max()))
        e = {}
        for idx, k in enumerate(ks):
            a_k, b_k = float(coef[1 + 2 * idx]), float(coef[2 + 2 * idx])
            e[int(k)] = a_k ** 2 + b_k ** 2
        energies.append(e)
    return worst, energies


def harmonic_budget(m, cs_list, L):
    """doc 62 SS11.3 -- THE DESCRIPTIVE E4 COULD NEVER PRODUCE. From the SS8.2
    sweep re-run on the TRAINED checkpoint, the normalised energy per harmonic
    e_k = (a_k^2 + b_k^2) / sum_j (a_j^2 + b_j^2), k = 1..L.

    DOES A TRAINED RE-UPLOADING MODEL USE THE HARMONICS IT WAS GIVEN? This is the
    measurement that makes a null interpretable: a cell that engaged (SS9.3) and
    lifted nothing with e_{k>=2} near zero is 'the optimiser declined the
    supply'; the same cell with substantial e_{k>=2} is 'the supply was used and
    did not help'.

    DESCRIPTIVE. NO BOOLEAN. NO CRITERION READS IT, AND IT MAY NOT BE PROMOTED
    AFTER RESULTS EXIST (the exp44 no-promotion clause, doc 62 SS11.3)."""
    ks = list(range(1, L + 1))
    resid, energies = fourier_fit_1d(m, cs_list, ks)
    per_story = []
    for e in energies:
        tot = sum(e.values())
        per_story.append({str(k): (e[k] / tot if tot > 0 else None) for k in ks})
    mean = {}
    for k in ks:
        vals = [p[str(k)] for p in per_story if p[str(k)] is not None]
        mean[str(k)] = float(np.mean(vals)) if vals else None
    return {"harmonics": ks, "resid_in_full_basis": resid,
            "normalised_energy_mean": mean,
            "normalised_energy_per_story": per_story,
            "raw_energy_per_story": [{str(k): v for k, v in e.items()}
                                     for e in energies],
            "status_doc62_SS11_3": ("DESCRIPTIVE. No boolean. No criterion reads "
                                    "it and it MAY NOT be promoted.")}


# ======================================== doc 62 SS8 -- the smoke battery gates
def check_dataset(store, rep):
    """doc 62 SS8.5 / SS20.1. The dataset is exp69's, BYTE-IDENTICAL, and its sha
    is asserted here so 'nothing changed' is CHECKED rather than assumed."""
    _items, meta, path = store
    ref = json.load(open(EXP69_RESULTS, encoding="utf-8"))
    exp69_sha = ref["datasets"]["A"]["sha256"]
    assert meta["dataset_sha256"] == exp69_sha, \
        ("exp75 is not running on exp69's Part A: %s vs %s"
         % (meta["dataset_sha256"], exp69_sha))
    assert meta["dataset_sha256"] == DATASET_SHA_DOC62, \
        ("the dataset sha does not match doc 62 SS8.5's registered value: %s vs %s"
         % (meta["dataset_sha256"], DATASET_SHA_DOC62))
    vs = json.load(open(VSUITE, encoding="utf-8"))
    assert vs.get("kill_gate_passed") is True, \
        "exp69's V-suite kill gate did not pass -- its results may not be re-used"
    rep["dataset"] = {
        "path": path, "sha256": meta["dataset_sha256"],
        "registered_sha_doc62_SS8_5": DATASET_SHA_DOC62,
        "matches_exp69_results": True,
        "vsuite_source": VSUITE, "vsuite_kill_gate_passed": True,
        "vsuite_status": ("INHERITED from exp69 on the strength of byte identity "
                          "(doc 62 SS8.1)"),
        "no_datagen_doc62_SS20_1": (
            "exp75 generates NO Arabic. The dataset is exp69's, byte-identical "
            "and sha-pinned, already V-suite-cleared; the noun inventory is the "
            "committed 44 and the category is the committed `nonoccupational` "
            "(doc 54 SS5.2, user-confirmed 2026-08-23). WHAT IS NEW IS HOW MANY "
            "TIMES THE SAME ANGLE IS APPLIED, so doc 50 SS5.2's native-speaker "
            "gate does not reopen. doc 59 SS8.4's 519-noun inventory is NOT used: "
            "switching inventories would forfeit the drift anchor."),
    }
    return rep


def check_committed_imports(rep):
    """doc 62 SS8.6 -- THE COMMITTED VALUES ARE IMPORTED, NOT RE-TYPED. R1-A's and
    R2-A's per-seed accuracies, means, sds, floors, comparator and bound are read
    PROGRAMMATICALLY from results_exp69.json and asserted equal to SS1.1's and
    SS6.3's printed tables."""
    src = json.load(open(EXP69_RESULTS, encoding="utf-8"))
    arms, floors = {}, {}
    for key, spec in COMMITTED_DOC62_SS1_1.items():
        a = src["arms"][key]
        f = src["floors"][key]
        c = src["comparator"]["arms"][key]
        b = src["bounds_context_only_doc55_SS12"][key]
        assert abs(float(a["mean"]) - spec["mean"]) < 5e-5, \
            ("doc 62 SS1.1's %s mean %.7f does not match %s :: %.7f"
             % (key, spec["mean"], EXP69_RESULTS, a["mean"]))
        assert abs(float(a["sd"]) - spec["sd"]) < 5e-4, \
            ("doc 62 SS1.1's %s sd %.4f does not match %.6f" % (key, spec["sd"],
                                                                a["sd"]))
        assert abs(float(f["mean"]) - spec["floor_mean"]) < 5e-5 \
            and abs(float(f["sd"]) - spec["floor_sd"]) < 5e-4, \
            ("doc 62 SS1.1's %s floor (%.4f +- %.4f) does not match (%.6f +- %.6f)"
             % (key, spec["floor_mean"], spec["floor_sd"], f["mean"], f["sd"]))
        assert abs(float(c["mean"]) - spec["comparator_9dim"]) < 5e-5, \
            ("doc 62 SS1.1's %s 9-dim comparator %.4f does not match %.6f"
             % (key, spec["comparator_9dim"], c["mean"]))
        assert abs(float(b["bound"]) - spec["bound"]) < 5e-5, \
            ("doc 62 SS1.3's reachable bound %.4f does not match %s :: %s = %.6f"
             % (spec["bound"], EXP69_RESULTS, key, b["bound"]))
        arms[key] = {"mean": float(a["mean"]), "sd": float(a["sd"]),
                     "per_seed": list(a["per_seed"]), "seeds": list(a["seeds"]),
                     "train_per_seed": list(a["train_per_seed"]),
                     "val_per_seed": list(a["val_per_seed"]),
                     "selected_epoch_per_seed": list(a["selected_epoch_per_seed"]),
                     "T": spec["T"], "doc62_SS1_1": spec["mean"],
                     "comparator_9dim": float(c["mean"]),
                     "bound_context_only": float(b["bound"]),
                     "wall_sec_total": float(a["wall_sec_total"])}
        floors[key] = {"mean": float(f["mean"]), "sd": float(f["sd"]),
                       "doc62_SS1_1": [spec["floor_mean"], spec["floor_sd"]]}
    # SS6.3's printed per-seed table, asserted against the FILE.
    for anchor, source in ANCHOR_SOURCE.items():
        want = arms[source]["per_seed"]
        printed = ANCHOR_PER_SEED_DOC62[anchor]
        assert all(abs(w - p) < 5e-5 for w, p in zip(want, printed)), \
            ("doc 62 SS6.3's printed per-seed values for %s do not match %s :: %s"
             % (anchor, EXP69_RESULTS, source))
    # SS1.2's precondition, DISCHARGED BY MEASUREMENT rather than assumed.
    tr = [x for k in COMMITTED_DOC62_SS1_1 for x in arms[k]["train_per_seed"]]
    assert max(tr) < 1.0, \
        ("doc 62 SS1.2's precondition FAILED: the committed family fits its own "
         "training split (max train acc %.4f) -- there is no residual training "
         "loss for a richer model to consume and doc 46 SS17.3(c) is not "
         "discharged" % max(tr))
    cc = arms["R1-A"]["mean"] - arms["R2-A"]["mean"]
    assert abs(cc - COMMITTED_CC_A) < 5e-4, \
        ("doc 62 SS1.1's cc_A = %.4f does not match the measured %.6f"
         % (COMMITTED_CC_A, cc))
    rep["committed_imports_doc62_SS8_6"] = {
        "arms": arms, "floors": floors, "cc_A_measured": cc,
        "lift_bar": {"T1": arms["R1-A"]["mean"] + 0.06,
                     "T2": arms["R2-A"]["mean"] + 0.06},
        "hurt_bar": {"T1": arms["R1-A"]["mean"] - 0.06,
                     "T2": arms["R2-A"]["mean"] - 0.06},
        "train_accuracy_range_doc62_SS1_2": [min(tr), max(tr)],
        "precondition_doc46_SS17_3c": (
            "DISCHARGED BY MEASUREMENT, not assumed: the committed model does NOT "
            "fit its own training split (train accuracy %.4f-%.4f over 528 items) "
            "and its family bound is 0.9625, below the task's attainable 1.0000. "
            "There is residual training loss for a richer model to consume, which "
            "is precisely the condition E4 lacked (doc 62 SS1.2)."
            % (min(tr), max(tr))),
        "honest_other_half_doc62_SS1_4": (
            "the committed arms sit 0.040-0.044 BELOW their own 0.9625 ceiling -- "
            "they have NOT converted the supply they already have. A lift "
            "therefore requires converting NEW supply while 0.04 of old supply "
            "sits unconverted, and doc 62 SS1.4 records this as the strongest "
            "in-programme reason to expect both `lifts` booleans FALSE."),
    }
    return rep


def check_features(fm, rep):
    """doc 62 SS8.4 item 5: the encoding angles consumed are BIT-IDENTICAL to the
    committed FA angles -- same featurizer, same divisors, same code object. A
    re-derived feature that differs anywhere fails the run."""
    R69.check_features(fm, rep)
    committed = E.feature_maps()["phi_exp42"]
    got, _ = arm_angles("A0-A-T1", fm)
    assert sorted(got) == sorted(committed), "the FA noun inventory drifted"
    worst = max(abs(float(got[n]) - float(committed[n])) for n in committed)
    assert worst == 0.0, \
        ("doc 62 SS8.4 item 5 FAILED: exp75's encoding angles are not bit-"
         "identical to the committed FA angles (max |d| = %.3e)" % worst)
    r1 = R69.arm_angles("R1-A", fm)[0]
    worst69 = max(abs(float(got[n]) - float(r1[n])) for n in r1)
    assert worst69 == 0.0, \
        "exp75's angles differ from exp69 R1-A's (max |d| = %.3e)" % worst69
    rep["encoding_angles_doc62_SS8_4_5"] = {
        "source": "exp68_run.feature_maps()['phi_exp42'] -- the COMMITTED map",
        "max_abs_deviation_vs_exp68": worst,
        "max_abs_deviation_vs_exp69_R1_A": worst69,
        "bit_identical": True, "n_nouns": len(got),
        "note": ("doc 62 SS7.5: THERE IS STILL NO LEARNED FEATURE. The encoding "
                 "angle is data-valued and consumes no RNG; what is new is that "
                 "the STATE the angle produces is processed by trained gates "
                 "before the verb blocks see it."),
    }
    return rep


def check_intro_structure(fm, verbs, store, rep):
    """doc 62 SS8.4, items 1, 2, 3 and 6 -- ALL BLOCKING."""
    out = {}
    items, _m, _p = arm_items("RU-L2-T1-uniform", store)
    ang, _a1 = arm_angles("RU-L2-T1-uniform", fm)
    data = R69.compile_all([i for i in items if i["split"] == "test"][:4], ang)
    cs = data[0]
    for arm in list(ANCHOR_ARMS) + TRAINED_CELLS:
        L = uploads_of(arm)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        apply_init_scheme(m, arm, 0)

        # item 1 -- EXACTLY L RY gates and L-1 V blocks per referent wire.
        tr = m.intro_gate_trace(cs)
        per_wire = {}
        for w, kind, _j in tr:
            per_wire.setdefault(w, {"RY": 0, "V": 0})[kind] += 1
        assert set(per_wire) == set(range(cs.K)), \
            "the intro does not touch every referent wire: %r" % sorted(per_wire)
        for w, c in per_wire.items():
            assert c["RY"] == L and c["V"] == L - 1, \
                ("doc 62 SS8.4 item 1 FAILED for %s on wire %d: %d RY and %d V "
                 "gates, registered %d and %d" % (arm, w, c["RY"], c["V"], L, L - 1))

        # item 2 -- the registered parameter count, and V_j SHARED across wires.
        npar = sum(p.numel() for p in m.parameters())
        assert npar == N_PARAMS_BY_L[L], \
            ("doc 62 SS8.4 item 2 FAILED for %s: %d trainable parameters, "
             "registered %d" % (arm, npar, N_PARAMS_BY_L[L]))
        assert len(m.reup) == L - 1, \
            "%s carries %d interleave blocks, registered %d" % (arm, len(m.reup),
                                                                L - 1)
        for j in range(1, L):
            same = max(float((m.v_matrix(j, 0) - m.v_matrix(j, w)).abs().max())
                       for w in range(cs.K))
            assert same == 0.0, \
                ("doc 62 SS7.2 / SS8.4 item 2 FAILED: V_%d is not SHARED across "
                 "referent wires for %s (max |d| = %.3e)" % (j, arm, same))
        vt = [id(b.theta) for b in m.reup]
        assert len(set(vt)) == len(vt), "an interleave block tensor is aliased"
        n_v_tensors = len(m.reup)
        n_tensors = len(list(m.parameters()))
        assert n_tensors == len(verbs) + 2 + n_v_tensors, \
            ("the trainable-tensor count for %s is %d, registered %d"
             % (arm, n_tensors, len(verbs) + 2 + n_v_tensors))

        # item 3 -- PER-WIRE INTRO PURITY = 1.0 to 1e-9, in complex128.
        wp64, wp32 = [], []
        for c in data:
            v = intro_state_f64(m, c)
            wp64 += wire_purities64(v, c.K)
            with torch.no_grad():
                psi = m.apply_intro(sim.init_state(c.K), c)
            wp32 += wire_purities(psi, c.K)
        dev64 = max(abs(x - 1.0) for x in wp64)
        dev32 = max(abs(x - 1.0) for x in wp32)
        assert dev64 <= F64_PURITY_TOL, \
            ("doc 62 SS8.4 item 3 FAILED for %s: per-wire intro purity deviates "
             "from 1 by %.3e in complex128 -- an ENTANGLER has entered the intro "
             "and this is a different experiment" % (arm, dev64))
        assert dev32 <= F32_PURITY_SANITY, \
            ("%s: the runtime complex64 intro purity deviates from 1 by %.3e, "
             "past the documented sanity bound %.0e -- the registered 1e-9 gate "
             "is asserted in complex128 and passed, so a deviation THIS large in "
             "the runtime pass is not accumulation" % (arm, dev32,
                                                       F32_PURITY_SANITY))

        # item 6 -- the intro consumes NO RNG and registers NO data-dependent
        # parameters.
        names = [n for n, _p in m.named_parameters()]
        assert not any(n.startswith("amp") or "noun" in n for n in names), names
        assert sum(1 for n in names if n.startswith("reup.")) == n_v_tensors
        st0 = torch.random.get_rng_state().clone()
        with torch.no_grad():
            a = m.story_values(cs)
            b = m.story_values(cs)
        st1 = torch.random.get_rng_state()
        assert bool(torch.equal(st0, st1)), \
            ("doc 62 SS8.4 item 6 FAILED for %s: the forward pass CONSUMED RNG"
             % arm)
        assert float((a - b).abs().max()) == 0.0, \
            "%s: two identical forward passes differ -- the intro is not a "\
            "function of the data alone" % arm

        # the complex128 re-derivation of Circuit4(1, 3) is VERIFIED against the
        # committed code object rather than trusted.
        vdev = 0.0
        for b_ in m.reup:
            M64 = circuit4_1q_matrix64(b_.theta.detach().numpy())
            M32 = b_.matrix().detach().numpy()
            vdev = max(vdev, float(np.abs(M64 - M32).max()))
        assert vdev <= F32_FLOOR, \
            ("the complex128 re-derivation of Circuit4(1, 3) disagrees with the "
             "committed code object by %.3e for %s" % (vdev, arm))

        out[arm] = {"L": L, "n_params": npar, "n_param_tensors": n_tensors,
                    "n_interleave_blocks": n_v_tensors,
                    "gates_per_wire": {str(k): v for k, v in sorted(per_wire.items())},
                    "intro_purity_max_dev_from_1_complex128": dev64,
                    "intro_purity_max_dev_from_1_complex64": dev32,
                    "circuit4_1q_f64_vs_committed_max_abs": vdev,
                    "v_shared_across_wires": True,
                    "intro_consumes_no_rng": True}
    rep["intro_structure_doc62_SS8_4"] = {
        "per_arm": out,
        "parameter_arithmetic_doc62_SS7_3": {
            "committed_FA": 120, "RU-L2": 126, "RU-L3": 132,
            "verb_blocks": "6 x 15 = 90", "readout": "2 x 15 = 30",
            "interleave": "(L - 1) x 6, SHARED across referent wires",
            "note": ("+5.0 % at L = 2 and +10.0 % at L = 3 against 528 training "
                     "items -- the SMALLEST capacity perturbation any rung in "
                     "this programme has registered (doc 62 SS0.4, SS7.3).")},
        "purity_reading_doc62_SS7_5": (
            "THE INTRO'S PER-WIRE PURITY REMAINS EXACTLY 1.0: only single-qubit "
            "gates act on individual wires, so the referents are still introduced "
            "as a PRODUCT STATE and doc 56's concurrence machinery has nothing to "
            "gate here. Asserted in complex128 at 1e-9 because the runtime "
            "complex64 forward has a ~1e-7 platform floor; both numbers are "
            "reported."),
    }
    return rep


def check_closures(fm, verbs, store, rep):
    """doc 62 SS8.2 -- THE FIVE FOURIER-CLOSURE GATES, PER LAYER, IN THE CORRECTED
    BASIS, WITH THEIR FAILING DIRECTIONS.

    C1  closure in the (2L+1)-term integer-harmonic basis, residual < 1e-6
    C2  NON-closure in the committed 3-term fundamental basis, > 1e-3, AND the
        anchor arms (L = 1) PASS exp69's own `< 1e-6` assertion in the same run
    C3  at the uniform draw, NON-closure in {1, cos L.theta, sin L.theta} > 1e-3
    C4  with all V_j EXACTLY 0: closure in {1, cos L.theta, sin L.theta} to 1e-6
        AND non-closure in {1, cos theta, sin theta} > 1e-3 for L >= 2
    C5  at the uniform draw every harmonic k = 1..L carries non-zero fitted
        energy; at V = 0 the k < L energies are zero

    SS8.2.1: FOR THE IDENTITY CELLS, C1, C2 AND C4 ARE BLOCKING; C3 AND C5 ARE
    MEASURED AND REPORTED with SS3.3's derived expectation stated alongside,
    because their k = 1..L-1 energy at t = 0 is O(sigma) and a fixed threshold
    would encode the noise scale rather than a property of the build."""
    out = {}
    # -- the instrument is the COMMITTED one, and that is asserted, not claimed.
    items1, _m, _p = arm_items("A0-A-T1", store)
    ang, _a1 = arm_angles("A0-A-T1", fm)
    d1 = R69.compile_all([i for i in items1 if i["split"] == "test"], ang)
    torch.manual_seed(0)
    m1 = build_model("A0-A-T1", verbs, 0, fm)
    parity_new, _e = fourier_fit_1d(m1, d1, [1])
    parity_old = R69.fourier_resid_1d(m1, d1, 3)
    assert parity_new == parity_old, \
        ("the widened closure instrument is NOT the committed frame: %.17e vs "
         "%.17e at ks = [1]" % (parity_new, parity_old))

    # -- C2's failing direction: the ANCHOR must PASS exp69's committed gate.
    anchors = {}
    for arm in ANCHOR_ARMS:
        items, _mm, _pp = arm_items(arm, store)
        a, _t = arm_angles(arm, fm)
        data = R69.compile_all([i for i in items if i["split"] == "test"], a)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        r3 = R69.fourier_resid_1d(m, data, 3)
        assert r3 < CLOSE_TOL, \
            ("%s does not close at 3 terms (%.3e) -- exp69's own assertion FAILS "
             "on the L = 1 anchor, so the instrument cannot discriminate and NO "
             "C2 result is admissible (doc 62 SS8.2)" % (arm, r3))
        anchors[arm] = {"resid_3term": r3, "closes_at_3": True,
                        "committed_gate": "exp69_run.fourier_resid_1d(m, data, 3) "
                                          "< 1e-6, called unchanged"}

    for arm in TRAINED_CELLS:
        L = uploads_of(arm)
        init = ARM_CFG[arm]["init"]
        items, _mm, _pp = arm_items(arm, store)
        a, _t = arm_angles(arm, fm)
        data = R69.compile_all([i for i in items if i["split"] == "test"], a)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        apply_init_scheme(m, arm, 0)
        ks_full = list(range(1, L + 1))

        c1, e_full = fourier_fit_1d(m, data, ks_full)
        assert c1 < CLOSE_TOL, \
            ("C1 FAILED for %s: the model does not close in the %d-term "
             "integer-harmonic basis (%.3e). A build needing a HIGHER basis has "
             "re-uploaded somewhere else, or the uploads are not unit-prefactor, "
             "and the arm FAILS (doc 62 SS8.2)" % (arm, 2 * L + 1, c1))
        c2, _e = fourier_fit_1d(m, data, [1])
        assert c2 > OPEN_TOL, \
            ("C2 FAILED for %s: the model CLOSES in the committed 3-term "
             "fundamental basis (%.3e) -- the re-uploads are inert and the arm is "
             "the single-encode circuit (doc 62 SS8.2)" % (arm, c2))
        c3, _e = fourier_fit_1d(m, data, [L])
        c5_live = {str(k): float(np.mean([e[k] for e in e_full])) for k in ks_full}
        c5_min = {str(k): float(np.min([e[k] for e in e_full])) for k in ks_full}
        if init == "uniform":
            assert c3 > OPEN_TOL, \
                ("C3 FAILED for %s: at the uniform draw the model CLOSES in "
                 "{1, cos %d.theta, sin %d.theta} (%.3e) -- it is a "
                 "frequency-rescaled single encode, i.e. V_j commutes with RY "
                 "and the interleave block is not doing its job (doc 62 SS8.2)"
                 % (arm, L, L, c3))
            for k in ks_full:
                assert c5_min[str(k)] > HARMONIC_ENERGY_BAR, \
                    ("C5 FAILED for %s: harmonic k = %d carries fitted energy "
                     "%.3e <= %.0e at the uniform draw -- the frequency is not "
                     "present (doc 62 SS8.2)" % (arm, k, c5_min[str(k)],
                                                 HARMONIC_ENERGY_BAR))

        # -- C4: SS3's derivation, MACHINE-CHECKED.
        keep = [b.theta.detach().clone() for b in m.reup]
        with torch.no_grad():
            for b in m.reup:
                b.theta.zero_()
        c4_close, e_zero = fourier_fit_1d(m, data, [L])
        c4_open, _e = fourier_fit_1d(m, data, [1])
        _r, e_zero_full = fourier_fit_1d(m, data, ks_full)
        with torch.no_grad():
            for b, k_ in zip(m.reup, keep):
                b.theta.copy_(k_)
        assert c4_close < CLOSE_TOL, \
            ("C4 FAILED for %s: with all V_j EXACTLY 0 the model does not close "
             "in {1, cos %d.theta, sin %d.theta} (%.3e). The build has not "
             "implemented RY(a)RY(b) = RY(a+b) composition and SS3's reading of "
             "the identity arm is VOID (doc 62 SS8.2 / SS3.4)"
             % (arm, L, L, c4_close))
        assert c4_open > OPEN_TOL, \
            ("C4 FAILED for %s: with all V_j EXACTLY 0 the model CLOSES in the "
             "fundamental basis (%.3e) -- at L >= 2 the circuit must be "
             "RY(%d.theta), whose spectrum is {0, +-%d} and NOT {0, +-1} "
             "(doc 62 SS3.2)" % (arm, c4_open, L, L))
        e_zero_low = {str(k): float(np.max([e[k] for e in e_zero_full]))
                      for k in ks_full if k < L}
        for k, v in e_zero_low.items():
            assert v < HARMONIC_ZERO_BAR, \
                ("C5's failing direction FAILED for %s: at V = 0 harmonic k = %s "
                 "carries energy %.3e, which is not zero -- the intermediate "
                 "harmonics must be ABSENT, not small (doc 62 SS3.3)"
                 % (arm, k, v))

        out[arm] = {
            "L": L, "init": init, "basis_terms": 2 * L + 1,
            "C1_resid_integer_harmonic_basis": c1, "C1_bar": CLOSE_TOL,
            "C1_passes": True,
            "C2_resid_fundamental_basis": c2, "C2_bar": OPEN_TOL,
            "C2_passes": bool(c2 > OPEN_TOL),
            "C3_resid_rescaled_single_encode_basis": c3, "C3_bar": OPEN_TOL,
            "C3_blocking": bool(init == "uniform"),
            "C3_passes": bool(c3 > OPEN_TOL),
            "C4_resid_at_V_zero_in_L_basis": c4_close,
            "C4_resid_at_V_zero_in_fundamental_basis": c4_open,
            "C4_passes": True,
            "C5_mean_fitted_energy_per_harmonic_uniform_draw": c5_live,
            "C5_min_fitted_energy_per_harmonic_uniform_draw": c5_min,
            "C5_blocking": bool(init == "uniform"),
            "C5_max_energy_below_L_at_V_zero": e_zero_low,
            "C5_zero_bar": HARMONIC_ZERO_BAR,
        }
    rep["fourier_closure_doc62_SS8_2"] = {
        "instrument": ("the COMMITTED `exp69_run.fourier_resid_1d` frame -- a "
                       "48-point sweep of one referent's intro angle over "
                       "[0, 2pi) on 4 held-out stories, least-squares fit, worst "
                       "residual -- WITH THE BASIS WIDENED. 48 points support "
                       "harmonics to 23, so L <= 3 is comfortably resolved."),
        "instrument_parity_assertion": {
            "exp75_fourier_fit_1d_ks_1": parity_new,
            "exp69_fourier_resid_1d_n_terms_3": parity_old,
            "bit_identical": True,
            "note": ("the widened instrument returns EXACTLY the committed "
                     "residual at ks = [1], so it is provably the committed "
                     "frame and not a second implementation.")},
        "anchors_pass_exp69_committed_gate": anchors,
        "per_cell": out,
        "derived_basis_doc62_SS2_1": (
            "each RY(theta) factor contributes frequencies +-1/2, so the entries "
            "of U_L(theta) carry frequencies (L - 2m)/2; every gate after the "
            "intro is data-independent and the readout is BILINEAR in amplitude "
            "and conjugate amplitude, so the OUTPUT frequencies are the "
            "differences m' - m. p(theta) lives in span{1, cos k.theta, "
            "sin k.theta : k = 1..L} -- exactly 2L+1 real terms, 5 at L = 2 and "
            "7 at L = 3. DERIVED, NOT ASSERTED -- this is E4's second defect "
            "(doc 46 SS17.2) repaired."),
        "sign_flip_note_doc62_SS2_1": (
            "exp69's `check_closures` failure message already names this "
            "experiment: 'the arm is not a single RY, or a referent angle has "
            "been RE-UPLOADED inside an event block'. EXP75 IS THE ARM THAT "
            "ASSERTION WAS WRITTEN TO DETECT, so exp75 runs it WITH ITS SIGN "
            "FLIPPED (C2) and requires the L = 1 anchors to pass it unchanged in "
            "the same test run."),
        "identity_asymmetry_doc62_SS8_2_1": (
            "FOR THE IDENTITY CELLS C1, C2 AND C4 ARE BLOCKING; C3 AND C5 ARE "
            "MEASURED AND REPORTED, NOT GATED. By SS3.3 their k = 1..L-1 energy "
            "at t = 0 is O(sigma), so a fixed threshold there would encode the "
            "NOISE SCALE rather than a property of the build. Stated in advance "
            "so the asymmetry is a DECISION and not an omission."),
    }
    return rep


def check_copy(fm, verbs, store, rep):
    """doc 62 SS8.1, MANDATORY AND RE-RUN UNCHANGED: with event 2 suppressed and
    q fixed to 1, the T = 2 forward pass must reproduce the T = 1 forward pass
    BIT-FOR-BIT.

    The re-uploads live in the INTRO, which happens exactly once before event 1,
    so doc 55 SS4.3's 'state prep happens EXACTLY ONCE' property is preserved in
    the form that matters -- one intro per story, whatever its internal depth.
    IF THIS FAILS, THE INTRO HAS LEAKED INTO THE EVENT LOOP AND THE RUN STOPS."""
    worst = {}
    for arm in ("A0-A-T2", "RU-L2-T2-uniform", "RU-L3-T2-identity"):
        items2, _m, _p = arm_items(arm, store)
        a, _t = arm_angles(arm, fm)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        apply_init_scheme(m, arm, 0)
        w = 0.0
        for it in items2[:6]:
            qi = it["query_index"]
            two = R69.compile_item(it, a)
            one_item = copy.deepcopy(it)
            one_item["events"] = [it["events"][qi - 1]]
            one_item["sentences"] = [it["sentences"][qi - 1]]
            one_item["order_flags"] = [it["order_flags"][qi - 1]]
            one_item["query_index"] = 1
            one = R69.compile_item(one_item, a)
            two_sup = copy.deepcopy(two)
            two_sup.events = [two.events[qi - 1]]
            with torch.no_grad():
                d = float((m.story_values(two_sup) - m.story_values(one)).abs().max())
            w = max(w, d)
        worst[arm] = w
        assert w == 0.0, \
            ("doc 62 SS8.1 COPY-VERIFICATION FAILED for %s: max diff %.3e -- the "
             "re-uploading intro has LEAKED INTO THE EVENT LOOP and the run "
             "stops" % (arm, w))
    rep["copy_verification_doc62_SS8_1"] = {
        "max_abs_diff": worst,
        "exact_zero": bool(all(v == 0.0 for v in worst.values())),
        "note": ("the re-uploads live in the INTRO, which happens exactly ONCE "
                 "before event 1, so doc 55 SS4.3's 'state prep happens EXACTLY "
                 "ONCE' property is preserved in the form that matters: ONE INTRO "
                 "PER STORY, WHATEVER ITS INTERNAL DEPTH. doc 62 SS17 item 4 "
                 "records re-uploading BETWEEN events as a DIFFERENT lever that "
                 "would break exactly this assertion and needs its own "
                 "registration."),
    }
    return rep


def check_anchor_codepath(fm, verbs, store, rep):
    """doc 62 SS8.3: THE ANCHOR RUNS THROUGH THE SAME `train()` ENTRY POINT as the
    re-uploading arms with L = 1 -- not a separate legacy path. Otherwise SS6.3's
    bit-exactness proves nothing about the arms' pipeline (doc 58 SS7.3, doc 61
    SS7.2 item 8).

    Asserted here at BUILD level: the L = 1 model's parameters, their ORDER and
    its forward pass are bit-identical to the committed
    `QuantumStoryModel('A3', verbs, seed)`."""
    out = {}
    for arm in ANCHOR_ARMS:
        items, _m, _p = arm_items(arm, store)
        a, _t = arm_angles(arm, fm)
        data = R69.compile_all([i for i in items if i["split"] == "test"][:8], a)
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            mine = build_model(arm, verbs, seed, fm)
            torch.manual_seed(seed)
            ref = QuantumStoryModel("A3", verbs, seed)
            mp = list(mine.parameters())
            rp = list(ref.parameters())
            assert len(mp) == len(rp), (len(mp), len(rp))
            pw = max(float((x - y).abs().max()) for x, y in zip(mp, rp))
            assert pw == 0.0, \
                ("doc 62 SS8.3 FAILED: the L = 1 model's parameters differ from "
                 "the committed QuantumStoryModel('A3') at seed %d (%.3e)"
                 % (seed, pw))
            assert len(mine.reup) == 0, "the L = 1 anchor carries interleave blocks"
            fw = 0.0
            with torch.no_grad():
                for cs in data:
                    fw = max(fw, float((mine.story_values(cs)
                                        - ref.story_values(cs)).abs().max()))
            assert fw == 0.0, \
                ("doc 62 SS8.3 FAILED: the L = 1 forward pass differs from the "
                 "committed model's at seed %d (%.3e)" % (seed, fw))
        out[arm] = {"seeds_checked": list(range(N_SEEDS)),
                    "max_abs_parameter_diff": 0.0,
                    "max_abs_forward_diff": 0.0, "n_interleave_blocks": 0}
    rep["anchor_codepath_doc62_SS8_3"] = {
        "per_arm": out,
        "note": ("the anchor is `ReuploadModel(verbs, seed, L = 1)`, which "
                 "constructs the committed parent VERBATIM from its own generator "
                 "and adds nothing: `self.reup` is EMPTY, so the parameter list, "
                 "its order, the RNG consumption and the forward gate sequence "
                 "are bit-identical. SS6.3's bit-exactness is therefore a "
                 "STRUCTURAL property of the build, not a tolerance."),
    }
    return rep


def check_trace(fm, verbs, store, rep):
    """doc 62 SS11.1: the traced pass must be THE SAME forward pass."""
    worst = {}
    for arm in ("A0-A-T2", "RU-L2-T2-uniform", "RU-L3-T1-identity"):
        items, _m, _p = arm_items(arm, store)
        a, _t = arm_angles(arm, fm)
        data = R69.compile_all([i for i in items if i["split"] == "test"][:6], a)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        apply_init_scheme(m, arm, 0)
        w = 0.0
        with torch.no_grad():
            for cs in data:
                v1 = m.story_values(cs)
                v2, tr = forward_traced(m, cs)
                w = max(w, float((v1 - v2).abs().max()))
                assert set(tr) == ({"after_intro", "at_readout"}
                                   | {"after_event%d" % (k + 1)
                                      for k in range(cs.T)}), tr.keys()
                for st, ps in tr.items():
                    assert len(ps) == cs.K, (st, len(ps), cs.K)
                    for p in ps:
                        assert 0.49 <= p <= 1.0001, (st, p)
                for p in tr["after_intro"]:
                    assert abs(p - 1.0) <= F32_PURITY_SANITY, \
                        ("doc 62 SS7.5: the intro purity is %.9f, not 1" % p)
        worst[arm] = w
        assert w == 0.0, \
            "doc 62 SS11.1: traced pass differs from story_values for %s: %.3e" \
            % (arm, w)
    rep["trace_parity_doc62_SS11_1"] = {
        "max_abs_diff_vs_story_values": worst,
        "exact_zero": bool(all(v == 0.0 for v in worst.values())),
        "stages": ["after_intro", "after_event1", "after_event2 (T=2)",
                   "at_readout"],
        "quantity": "per-wire purity Tr rho^2 of the single-qubit reduced state",
        "registered_question_doc62_SS11_2": (
            "does adding uploads change the q-asymmetry UNDO SIGNATURE, and does "
            "an arm that lifts (if any does) show a restored or a degraded one? "
            "doc 55 SS20.5's healthy profile was measured ON R2-A ITSELF -- the "
            "same family, the same wires, the same readout -- so exp75's arms are "
            "read against the committed profile AS MEASURED and doc 56 SS15.2's "
            "transposition and doc 61 SS9.4's re-basing are NOT needed: on "
            "correct q = 2 items the readout restores per-wire purity to its "
            "post-event-1 level (0.8755 vs 0.8751); on q = 1 items it does not "
            "(0.8017 vs 0.8953). DESCRIPTIVE, NO BOOLEAN, NO CRITERION READS IT, "
            "AND IT MAY NOT BE PROMOTED (doc 62 SS11.2, SS12.7)."),
    }
    return rep


def check_shuffle(store, rep):
    """doc 62 SS18.1: the floor severs features <-> label and changes NOTHING
    else. `exp69_run.relabel_shuffle_q` is IMPORTED, not re-implemented."""
    items, meta, _p = store
    bearers = set(meta["bearers"])
    rows = []
    for seed in range(N_SEEDS):
        sh = R69.relabel_shuffle_q(items, seed)
        by = {}
        for it in sh:
            by.setdefault(it["pair_id"], []).append(it)
        opp = all(len(gg) == 2 and gg[0]["answer"] != gg[1]["answer"]
                  for gg in by.values())
        same_cast = all([r["name"] for r in gg[0]["referents"]]
                        == [r["name"] for r in gg[1]["referents"]]
                        for gg in by.values())
        same_q = all(gg[0]["query_index"] == gg[1]["query_index"]
                     for gg in by.values())
        n_yes = sum(1 for i in sh if i["answer"] == "yes")
        true_rule = sum(1 for i in sh
                        if (i["events"][i["query_index"] - 1]["agent"] in bearers)
                        == (i["answer"] == "yes"))
        rows.append({"seed": seed, "n_items": len(sh),
                     "twins_still_opposite": bool(opp),
                     "casts_unchanged": bool(same_cast),
                     "query_index_unchanged": bool(same_q),
                     "yes_fraction": n_yes / len(sh),
                     "true_bearer_rule_accuracy": true_rule / len(sh)})
    for r in rows:
        assert r["twins_still_opposite"] and r["casts_unchanged"] \
            and r["query_index_unchanged"], "S-shuffle broke the twin structure"
        assert r["yes_fraction"] == 0.5, "S-shuffle unbalanced the classes"
        assert abs(r["true_bearer_rule_accuracy"] - 0.5) < 0.16, \
            ("S-shuffle did not sever features <-> label: %.4f"
             % r["true_bearer_rule_accuracy"])
    rep["shuffle_doc62_SS18"] = {
        "per_seed": rows,
        "mean_true_bearer_rule_accuracy":
            float(np.mean([r["true_bearer_rule_accuracy"] for r in rows])),
        "floored_cells": TRAINED_CELLS,
        "n_floors": len(FLOOR_ARMS), "n_floor_runs": len(FLOOR_ARMS) * N_SEEDS,
        "note": ("doc 62 SS18.1: THE FLOOR IS PROCEDURE-MATCHED AND BOTH L AND "
                 "THE INIT ARE PART OF THE PROCEDURE -- a floor measured under "
                 "one procedure does not bound what another can achieve on "
                 "severed labels. EIGHT CELLS, EIGHT FLOORS. Bearer designation "
                 "is reassigned PER PAIR and independently of the features, so "
                 "twins stay on opposite labels and V1 stays exactly 0.500."),
        "stage_1_decision_doc62_SS18_2": (
            "ALL EIGHT FLOORS RUN IN STAGE 1 (orchestrator ruling 1). doc 61 "
            "SS15 gated floors behind a trigger; HERE THE TRIGGER IS SATISFIED BY "
            "ARITHMETIC -- the committed family binds at 0.9187-0.9227, so every "
            "non-catastrophic cell clears 0.65 and all eight would trigger. "
            "Staging would buy nothing and would serialise 40 runs behind a "
            "foregone conclusion. SS18.3's trigger definition is RETAINED for "
            "auditability and every cell is treated as triggered."),
        "anchor_carries_no_fresh_floor_doc62_SS6_3_2": (
            "doc 56 SS5.3 / doc 58 SS5.2's transitive argument: the anchor is "
            "pinned to within 0.002 of measurements that cleared their own "
            "matched floors (R1-A by 8.91 floor-SD units, R2-A by 18.95). IF THE "
            "ANCHOR FAILS ITS GATE, THIS ARGUMENT IS VOID ALONG WITH THE RUN."),
    }
    return rep


def check_identity_init(fm, verbs, rep):
    """doc 62 SS7.4 / SS3 -- the identity-init cells, through the COMMITTED exp72
    code objects, with SS3's derivation recorded alongside."""
    out = {}
    for L in UPLOADS:
        arm = "RU-L%d-T1-identity" % L
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        info = apply_init_scheme(m, arm, 0)
        out[arm] = {k: v for k, v in info.items() if k != "blocks_reinitialised"}
        out[arm]["n_blocks_reinitialised"] = len(info["blocks_reinitialised"])
    rep["identity_init_doc62_SS7_4"] = {
        "per_cell": out,
        "two_inits_are_different_circuits_doc62_SS3_3": (
            "THE TWO INITS ARE DIFFERENT CIRCUITS, NOT TWO DRAWS OF ONE CIRCUIT. "
            "The uniform arm begins with its full frequency supply live: "
            "V_j ~ U[0, 2pi) is a generic SU(2) and all of {0, +-1, ..., +-L} "
            "carries non-zero weight. The identity arm begins a distance "
            "O(sigma) from a manifold on which THE SUPPLY DOES NOT EXIST -- at "
            "exact identity RY(a).RY(b) = RY(a+b) makes the circuit RY(L.theta), "
            "whose spectrum is {0, +-L} -- and must train its way off that "
            "manifold before any of the intervention's expressivity is "
            "available. This is why doc 59 SS2 refused to treat the init as an "
            "optimiser variant for D2, why both inits are arms, and why each "
            "gets its OWN procedure-matched floor."),
        "no_cross_experiment_init_narrative_doc62_SS3_5": (
            "because identity init REMOVES frequency supply at t = 0 while "
            "uniform init supplies it, `init_helps` and `init_hurts` on this rung "
            "DO NOT MEASURE THE SAME THING they measure on doc 58's and doc 61's "
            "rungs, where the init changed only the effective depth at the first "
            "update. SS16 FORBIDS the subtraction and SS12.8(e) refuses any "
            "cross-experiment init comparison as a criterion."),
    }
    return rep


# ======================== doc 62 SS9.2 -- E4's GHOST, LEVEL 1, BLOCKING, AT SMOKE
def v_grad_norms(m, cs_list):
    """||dL/dV_j|| per interleave block at the current parameters."""
    m.zero_grad(set_to_none=True)
    loss = 0.0
    for cs in cs_list:
        loss = loss + torch.nn.functional.cross_entropy(
            torch.log(m.story_values(cs) + 1e-12).unsqueeze(0),
            torch.tensor([cs.answer]))
    (loss / len(cs_list)).backward()
    out = []
    for b in m.reup:
        out.append(0.0 if b.theta.grad is None
                   else float(b.theta.grad.norm()))
    m.zero_grad(set_to_none=True)
    return out


def check_engagement_level1(fm, verbs, store, rep):
    """doc 62 SS9.2 -- M1, M2, M3. BLOCKING, BEFORE ANY ARM RUNS.

    SS9.1: what can still be inert under SS4's freeze is NOT the uploads -- they
    are structural -- but the INTERLEAVE BLOCK. If V_j never leaves its initial
    value, the L = 2/3 circuit is whatever its init made it and the TRAINABLE
    half of the re-uploading construction has not participated. That is E4 and
    exp57's `M-wide-ROT` again, and it is inadmissible."""
    out = {}
    for arm in TRAINED_CELLS:
        L = uploads_of(arm)
        items, _m, _p = arm_items(arm, store)
        a, _t = arm_angles(arm, fm)
        train_cs = R69.compile_all(
            [i for i in items if i["split"] == "train"][:BATCH], a)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        apply_init_scheme(m, arm, 0)

        # M1 -- GRADIENT LIVENESS at initialisation, per block.
        gn = v_grad_norms(m, train_cs)
        assert len(gn) == L - 1, (arm, len(gn), L - 1)
        for j, g in enumerate(gn, start=1):
            assert g > M1_GRAD_BAR, \
                ("M1 FAILED for %s: ||dL/dV_%d|| = %.3e <= %.0e at "
                 "initialisation -- the interleave block is DETACHED from the "
                 "graph or excluded from the optimiser's param_groups "
                 "(doc 62 SS9.2)" % (arm, j, g, M1_GRAD_BAR))

        # M3 -- exactly one optimiser parameter group, and the tensor count.
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        assert len(opt.param_groups) == 1, \
            "M3 FAILED for %s: %d optimiser parameter groups" % (
                arm, len(opt.param_groups))
        grp = opt.param_groups[0]["params"]
        for j, b in enumerate(m.reup, start=1):
            n_in = sum(1 for p in grp if p is b.theta)
            assert n_in == 1, \
                ("M3 FAILED for %s: V_%d appears in %d optimiser parameter "
                 "groups, registered exactly 1 (doc 62 SS9.2)" % (arm, j, n_in))
        assert len(grp) == len(verbs) + 2 + (L - 1), \
            ("M3 FAILED for %s: %d trainable tensors, registered %d "
             "(doc 62 SS8.4 item 2)" % (arm, len(grp), len(verbs) + 2 + (L - 1)))

        # M2 -- MOVEMENT under the registered probe: 10 epochs, seed 0, the
        # committed optimiser and lr, the cell's own init and data.
        pr = train(arm, 0, store, verbs, M2_PROBE_EPOCHS, fm, save=False,
                   quiet=True)
        mv = pr["max_abs_dtheta_V_final"]
        assert mv is not None and mv >= M2_MOVE_BAR, \
            ("M2 FAILED for %s: max_j max |dtheta_V| = %r after %d epochs, bar "
             "%.2f rad -- the optimiser does not touch the interleave blocks at "
             "all and the arm is E4's shape (doc 62 SS9.2)"
             % (arm, mv, M2_PROBE_EPOCHS, M2_MOVE_BAR))
        out[arm] = {"L": L, "M1_grad_norm_per_block": gn, "M1_bar": M1_GRAD_BAR,
                    "M2_max_abs_dtheta_V_final": mv,
                    "M2_max_abs_dtheta_V_selected": pr["max_abs_dtheta_V"],
                    "M2_epochs": M2_PROBE_EPOCHS, "M2_bar": M2_MOVE_BAR,
                    "M2_verb_block_movement_for_contrast":
                        pr["max_abs_dtheta_verb_final"],
                    "M3_optimiser_param_groups": 1,
                    "M3_trainable_tensors": len(grp),
                    "probe_wall_sec": pr["wall_sec"]}
        print("[smoke M] %-24s M1 min|g|=%.3e  M2 max|dV|=%.4f  (%.0fs)"
              % (arm, min(gn), mv, pr["wall_sec"]), flush=True)
    rep["engagement_level1_doc62_SS9_2"] = {
        "per_cell": out,
        "M2_bar_is_deliberately_loose": (
            "at %d epochs M2 certifies only that the optimiser TOUCHES the blocks "
            "at all. THE DEMANDING BAR IS LEVEL 2 (SS9.3): "
            "reupload_engaged(cell, seed) = max_j max |theta_V(selected) - "
            "theta_V(init)| >= %.2f rad, measured on EVERY trained checkpoint."
            % (M2_PROBE_EPOCHS, ENGAGEMENT_BAR)),
        "threshold_provenance_doc62_SS9_3": (
            "E4's kappa moved by MAX +0.0780 across all 34 S-reup runs (mean "
            "+0.0064, min -0.0201), so a 0.10 rad bar would have failed EVERY ONE "
            "of them, INCLUDING THE MOST-MOVED. exp57's `M-wide-ROT` moved by "
            "5.94e-05 rad, three orders below. The bar is ABSOLUTE, PER RUN, on "
            "the maximum over the block's 6 or 12 parameters, and IS NOT A MEAN "
            "OVER SEEDS -- E4's arm passed on its mean only because 34 runs were "
            "averaged into a summary. Orchestrator ruling 3 ACCEPTS it as "
            "calibrated."),
        "what_SS9_is_not_doc62_SS9_5": (
            "SS9 is NOT a criterion about whether the trained model USES its "
            "harmonics -- that is SS11.3's descriptive, it has no boolean, and it "
            "MAY NOT BE PROMOTED. A model whose blocks moved 1.2 rad and whose "
            "trained output is nonetheless effectively single-harmonic is "
            "ADMISSIBLE AND INFORMATIVE; a model whose blocks never moved is "
            "INADMISSIBLE AND UNINFORMATIVE. The distinction is the whole content "
            "of E4's erratum."),
    }
    return rep


# ==================== doc 48 SS10.6 / doc 62 SS22 -- THE FAILING DIRECTIONS
class _BrokenIntroCount(ReuploadModel):
    """L + delta uploads instead of L -- SS8.4 item 1's failing direction. The
    interleave blocks are left at their registered count (L - 1) and only the
    UPLOAD count moves, which is the slip the gate exists to catch."""

    def __init__(self, verbs, seed, L, delta=1):
        super().__init__(verbs, seed, L)
        seq = [("RY", 0)]
        for j in range(1, L):
            seq += [("V", j), ("RY", 0)]
        seq += [("RY", 0)] * delta
        self.intro_spec = tuple(seq)


class _BrokenPrefactor(ReuploadModel):
    """A non-unit prefactor on one upload -- C1's failing direction (doc 62 SS4.3
    / SS17 item 2: fixed non-unit prefactor grids are DECLINED, and this shows
    the C1 gate would catch one)."""

    def upload_scale(self, j):
        return 2.0 if j == 0 else 1.0


class _BrokenPerWire(ReuploadModel):
    """Per-wire interleave blocks -- SS7.2 / SS8.4 item 2's failing direction."""

    def __init__(self, verbs, seed, L, n_wires=2):
        super().__init__(verbs, seed, L)
        gv = torch.Generator().manual_seed(REUP_RNG_BASE + 999)
        self.reup_w = nn.ModuleList(
            [sim.Circuit4(1, LAYERS, generator=gv)
             for _ in range((L - 1) * n_wires)])
        self.n_wires = n_wires

    def v_matrix(self, j, wire):
        return self.reup_w[(j - 1) * self.n_wires + wire].matrix()


class _BrokenEntangledIntro(ReuploadModel):
    """An entangler inside the intro -- SS8.4 item 3's failing direction."""

    def apply_intro(self, psi, cs):
        psi = super().apply_intro(psi, cs)
        if cs.K >= 2:
            psi = sim.apply_gate(psi, sim.cphase(1.1), (0, 1))
        return psi


class _BrokenDetachedV(ReuploadModel):
    """V_j detached from the graph -- M1's failing direction."""

    def v_matrix(self, j, _wire):
        return self.reup[j - 1].matrix().detach()


class _BrokenFrozenV(ReuploadModel):
    """V_j registered but never optimised -- M2's failing direction."""

    def __init__(self, verbs, seed, L):
        super().__init__(verbs, seed, L)
        for p in self.reup.parameters():
            p.requires_grad_(False)


def check_failing_directions(fm, verbs, store, rep):
    """doc 48 SS10.6 / doc 49 SS8 / doc 62 SS22: EVERY ASSERTION HAS A FAILING
    DIRECTION AND THE FAILING DIRECTIONS ARE THEMSELVES ASSERTED. A GATE THAT
    CANNOT FAIL IS NOT A GATE, and doc 62 SS22's 'instrument defect' row is the
    entry E4 failed. Each broken build below is constructed and shown to FIRE the
    gate it is registered against, BEFORE any training is launched."""
    out = {}
    L = 2
    arm = "RU-L2-T1-uniform"
    items, _m, _p = arm_items(arm, store)
    a, _t = arm_angles(arm, fm)
    test = R69.compile_all([i for i in items if i["split"] == "test"], a)
    train_cs = R69.compile_all(
        [i for i in items if i["split"] == "train"][:BATCH], a)
    cs = test[0]

    # (1) A WRONG UPLOAD COUNT MUST FAIL SS8.4 item 1.
    torch.manual_seed(0)
    b1 = _BrokenIntroCount(verbs, 0, L, delta=1)
    per_wire = {}
    for w, kind, _j in b1.intro_gate_trace(cs):
        per_wire.setdefault(w, {"RY": 0, "V": 0})[kind] += 1
    n_ry = per_wire[0]["RY"]
    assert n_ry != L, "the upload-count failing direction did not change the count"
    fired1 = n_ry != L
    out["upload_count_slip_fires"] = {
        "registered_RY_per_wire": L, "broken_build_RY_per_wire": n_ry,
        "gate_fires": bool(fired1),
        "reading": ("SS8.4 item 1 counts the gates the intro ACTUALLY APPLIES "
                    "(`intro_gate_trace`), not a comment, so a build with L +- 1 "
                    "uploads is caught by the count itself.")}

    # (2) A NON-UNIT PREFACTOR MUST FAIL C1 (kappa != 1 widens the spectrum).
    torch.manual_seed(0)
    b2 = _BrokenPrefactor(verbs, 0, L)
    c1_bad, _e = fourier_fit_1d(b2, test, list(range(1, L + 1)))
    assert c1_bad > OPEN_TOL, \
        ("the non-unit-prefactor failing direction did NOT fire (C1 residual "
         "%.3e) -- C1 is a null instrument" % c1_bad)
    torch.manual_seed(0)
    b2ok = ReuploadModel(verbs, 0, L)
    c1_ok, _e = fourier_fit_1d(b2ok, test, list(range(1, L + 1)))
    out["non_unit_prefactor_fires"] = {
        "C1_residual_broken": c1_bad, "C1_residual_correct": c1_ok,
        "C1_bar": CLOSE_TOL,
        "reading": ("doc 62 SS4.1 freezes kappa = 1, so the accessible spectrum "
                    "is the INTEGER-HARMONIC set {0, +-1, ..., +-L} and C1 closes "
                    "at 2L+1 terms. A build whose uploads are NOT unit-prefactor "
                    "needs a HIGHER basis and C1 catches it -- which is E4's "
                    "second defect (doc 46 SS17.2) made into a build gate. This "
                    "is also why SS17 item 2's ternary-grid variant is NOTED, NOT "
                    "RUN: it would fail this arm's own validity gate.")}

    # (3) PER-WIRE BLOCKS MUST FAIL SS8.4 item 2.
    torch.manual_seed(0)
    b3 = _BrokenPerWire(verbs, 0, L)
    npar3 = sum(p.numel() for p in b3.parameters())
    shared_dev = max(float((b3.v_matrix(1, 0) - b3.v_matrix(1, w)).abs().max())
                     for w in range(cs.K))
    assert npar3 != N_PARAMS_BY_L[L] and shared_dev > 1e-3, \
        ("the per-wire failing direction did NOT fire: %d params, max |V_0 - V_w| "
         "= %.3e" % (npar3, shared_dev))
    out["per_wire_blocks_fire"] = {
        "registered_n_params": N_PARAMS_BY_L[L], "broken_n_params": npar3,
        "max_abs_V_difference_across_wires": shared_dev,
        "reading": ("doc 62 SS7.2: the wire index is an arbitrary "
                    "compiler-assigned referent SLOT, not a semantic role, so "
                    "per-wire parameters would make the model sensitive to an "
                    "arbitrary labelling -- and the committed intro has ZERO "
                    "per-wire parameters. Both halves of the gate (count, "
                    "sharing) are shown to fire.")}

    # (4) AN ENTANGLER IN THE INTRO MUST FAIL SS8.4 item 3.
    torch.manual_seed(0)
    b4 = _BrokenEntangledIntro(verbs, 0, L)
    with torch.no_grad():
        psi = b4.apply_intro(sim.init_state(cs.K), cs)
    dev4 = max(abs(x - 1.0) for x in wire_purities(psi, cs.K))
    assert dev4 > 1e-3, \
        ("the entangled-intro failing direction did NOT fire (max |purity - 1| = "
         "%.3e) -- SS8.4 item 3 is a null instrument" % dev4)
    out["entangled_intro_fires"] = {
        "max_abs_purity_deviation_from_1": dev4, "bar": F64_PURITY_TOL,
        "reading": ("doc 62 SS7.5: a per-wire purity below 1 means an ENTANGLER "
                    "HAS ENTERED THE INTRO, which is a DIFFERENT EXPERIMENT (doc "
                    "56's encoding-side entanglement rung). The gate is shown to "
                    "see one.")}

    # (5) A DETACHED V MUST RETURN EXACTLY 0.0 AND FAIL M1.
    torch.manual_seed(0)
    b5 = _BrokenDetachedV(verbs, 0, L)
    gn_bad = v_grad_norms(b5, train_cs)
    torch.manual_seed(0)
    b5ok = ReuploadModel(verbs, 0, L)
    gn_ok = v_grad_norms(b5ok, train_cs)
    assert all(g == 0.0 for g in gn_bad), \
        ("the detached-V failing direction did NOT return exactly 0.0: %r"
         % gn_bad)
    assert all(g > M1_GRAD_BAR for g in gn_ok), \
        "the correct build does not have live V gradients: %r" % gn_ok
    out["detached_V_fires_M1"] = {
        "grad_norms_detached_build": gn_bad, "grad_norms_correct_build": gn_ok,
        "M1_bar": M1_GRAD_BAR, "returns_exactly_zero": True,
        "reading": ("doc 62 SS9.2's M1 failing direction, VERBATIM: 'a build with "
                    "V_j detached from the graph (or excluded from the "
                    "optimiser's param_groups) must return EXACTLY 0.0 and FAIL "
                    "M1'.")}

    # (6) A FROZEN V MUST MOVE EXACTLY 0.0 AND FAIL M2.
    torch.manual_seed(0)
    b6 = _BrokenFrozenV(verbs, 0, L)
    pr6 = train(arm, 0, store, verbs, 2, fm, save=False, model=b6, quiet=True)
    assert pr6["max_abs_dtheta_V_final"] == 0.0, \
        ("the frozen-V failing direction moved by %r, not exactly 0.0"
         % pr6["max_abs_dtheta_V_final"])
    assert pr6["max_abs_dtheta_verb_final"] > 0.0, \
        "the frozen-V build did not train its VERB blocks either -- the probe is "\
        "not measuring what it claims"
    grp6 = torch.optim.Adam(
        [p for p in b6.parameters() if p.requires_grad], lr=LR).param_groups[0]
    n_v_opt = sum(1 for p in grp6["params"] for b in b6.reup if p is b.theta)
    assert n_v_opt == 0, "the frozen-V build still optimises V_j"
    out["frozen_V_fires_M2_and_M3"] = {
        "max_abs_dtheta_V_after_2_epochs": pr6["max_abs_dtheta_V_final"],
        "max_abs_dtheta_verb_after_2_epochs": pr6["max_abs_dtheta_verb_final"],
        "V_tensors_in_optimiser": n_v_opt, "M2_bar": M2_MOVE_BAR,
        "returns_exactly_zero": True,
        "reading": ("doc 62 SS9.2's M2 and M3 failing directions: 'a frozen-V "
                    "build must return EXACTLY 0.0 and FAIL M2' and 'a parameter "
                    "registered but not optimised fails M3'. The verb blocks move "
                    "in the same probe, so the run is measuring V's inertness and "
                    "not a dead optimiser. THE FROZEN BUILD IS ROUTED THROUGH THE "
                    "SAME `train()` ENTRY POINT as the arms, so the failing "
                    "direction tests the production loop.")}

    # (7) THE WRONG ANSATZ DEPTH MUST BREAK THE PARAMETER COUNT.
    torch.manual_seed(0)
    b7 = ReuploadModel(verbs, 0, L, layers=2)
    n7 = sum(p.numel() for p in b7.parameters())
    assert n7 != N_PARAMS_BY_L[L], \
        "a 2-layer interleave block reproduced the registered %d-parameter count" \
        % N_PARAMS_BY_L[L]
    out["wrong_ansatz_depth_fires"] = {
        "layers_2_param_count": n7,
        "registered_3_layer_count": N_PARAMS_BY_L[L]}

    # (8) doc 58 SS6.2.2's NULL-INSTRUMENT CHECK, RE-RUN: an identity-initialised
    #     READOUT must be REJECTED. The committed assertion is CALLED.
    torch.manual_seed(0)
    b8 = ReuploadModel(verbs, 0, L)
    R72.apply_identity_init(b8, 0, verbs, [])
    with torch.no_grad():
        b8.q_pos.theta.zero_()
        b8.q_neg.theta.zero_()
    rejected = False
    try:
        R72.identity_assertions(b8, verbs, [], 0)
    except AssertionError as exc:
        rejected = True
        out["identity_readout_message"] = str(exc)[:200]
    assert rejected, \
        ("doc 58 SS6.2.2's null-instrument check ACCEPTED an identity-initialised "
         "readout -- two identical readout blocks give v_yes == v_no everywhere")
    torch.manual_seed(0)
    b8b = ReuploadModel(verbs, 0, L)
    with torch.no_grad():
        b8b.q_pos.theta.zero_()
        b8b.q_neg.theta.zero_()
        vals = b8b.story_values(cs)
    out["identity_readout_rejected"] = True
    out["identity_readout_is_degenerate"] = {
        "v_yes": float(vals[0]), "v_no": float(vals[1]),
        "abs_v_yes_minus_v_no": float((vals[0] - vals[1]).abs()),
        "reading": ("doc 62 SS7.4, MEASURED rather than argued: with both readout "
                    "blocks at identity the two effects are THE SAME OPERATOR, so "
                    "v_yes == v_no EXACTLY for every item -- a NULL INSTRUMENT in "
                    "doc 53 SS5's exact sense. This is why sigma is applied to the "
                    "verb blocks and V_j only.")}

    # (9) THE WRONG SIGMA must be rejected by the combined sd window.
    torch.manual_seed(0)
    b9 = ReuploadModel(verbs, 0, L)
    g = torch.Generator().manual_seed(IDENTITY_RNG_BASE)
    with torch.no_grad():
        for v in sorted(verbs):
            t = b9.blocks[v].theta
            t.copy_(torch.randn(t.shape, generator=g) * 0.5)
        for b in b9.reup:
            b.theta.copy_(torch.randn(b.theta.shape, generator=g) * 0.5)
    rejected9 = False
    try:
        R72.identity_assertions(b9, verbs, [], 0)
    except AssertionError:
        rejected9 = True
    rejected9b = False
    try:
        identity_v_assertions(b9)
    except AssertionError:
        rejected9b = True
    assert rejected9 and rejected9b, \
        "the identity-init sd window ACCEPTED sigma = 0.5 (verb %s, combined %s)" \
        % (rejected9, rejected9b)
    out["wrong_sigma_rejected"] = {"committed_verb_window": True,
                                   "combined_verb_plus_V_window": True}

    # (10) THE COMBINED sd WINDOW MUST HAVE TEETH ON V ALONE -- and its measured
    #      LIMIT is recorded rather than asserted away.
    torch.manual_seed(0)
    b10 = ReuploadModel(verbs, 0, L)
    R72.apply_identity_init(b10, 0, verbs, [])
    gv = torch.Generator().manual_seed(REUP_IDENTITY_RNG_BASE)
    with torch.no_grad():
        for b in b10.reup:
            b.theta.copy_(torch.randn(b.theta.shape, generator=gv) * 1.0)
    fired10 = False
    try:
        identity_v_assertions(b10)
    except AssertionError as exc:
        fired10 = True
        out["v_sd_window_message"] = str(exc)[:200]
    assert fired10, \
        "the combined verb + V sd window ACCEPTED a sigma = 1.0 draw on V_j"
    # ... and the LIMIT, MEASURED: a CONSTANT V draw at the right magnitude is
    # NOT caught by a pooled sd window. Recorded as a measured limitation, which
    # is why the V-only sd is reported un-gated (SS8.4 item 4's reading).
    torch.manual_seed(0)
    b10b = ReuploadModel(verbs, 0, L)
    R72.apply_identity_init(b10b, 0, verbs, [])
    with torch.no_grad():
        for b in b10b.reup:
            b.theta.copy_(torch.zeros_like(b.theta) + IDENTITY_SIGMA)
    caught = False
    try:
        identity_v_assertions(b10b)
    except AssertionError:
        caught = True
    out["v_sigma_window"] = {
        "sigma_1p0_on_V_rejected": bool(fired10),
        "constant_V_draw_at_sigma_magnitude_caught": bool(caught),
        "reading": ("THE WINDOW HAS TEETH ON V (a sigma = 1.0 draw on the "
                    "interleave blocks alone moves the COMBINED pool sd outside "
                    "[0.08, 0.12] and is rejected), AND ITS LIMIT IS MEASURED "
                    "RATHER THAN ASSUMED AWAY: a CONSTANT V draw at the right "
                    "magnitude is not caught by a pooled sd statistic. That is "
                    "the price of pooling, which is paid because a V-only window "
                    "at n = 6 would gate on the draw's own sampling noise -- so "
                    "the V-only sd is REPORTED UN-GATED alongside every identity "
                    "run (doc 62 SS8.4 item 4's reading, FLAGGED). The registered "
                    "identity property itself -- Circuit4(1, 3) at theta = 0 IS "
                    "the identity to 1e-9 -- is asserted directly and separately, "
                    "and is not carried by the sd window.")}

    rep["failing_directions_doc62_SS22"] = {
        "checks": out,
        "rule": ("doc 48 SS10.6 / doc 49 SS8: an arm that does not implement what "
                 "it is named for is a NULL INSTRUMENT, and a gate that cannot "
                 "fail is not a gate. Every SS8 and SS9 assertion above is shown "
                 "to FIRE on a deliberately broken build BEFORE any training is "
                 "launched. This is the doc 62 SS22 'instrument defect' row -- "
                 "THE ENTRY E4 FAILED -- and it carries the heaviest gate set in "
                 "the registration."),
        "null_instrument_by_symmetry_doc56_SS14": (
            "applied per gate. THE LIVE CASE IS C1: a SINGLE-ENCODE build would "
            "also close in the widened basis at L = 1, so C2 asserts the "
            "COMPLEMENTARY non-closure and the anchor arms are required to pass "
            "exp69's ORIGINAL gate in the same test run. THE SECOND LIVE CASE IS "
            "M1, which a build that registers V_j but never steps it would pass "
            "-- so M2 (movement) and M3 (optimiser group membership) are asserted "
            "too, with detached and frozen builds as the failing directions."),
    }
    return rep


# ==================================================================== the smoke
def smoke(fm, verbs, store, tiny_epochs=2):
    t0 = time.time()
    rep = {"experiment": ("exp75 smoke / validity + engagement assertions "
                          "(doc 62 SS8, SS9.2)"),
           "spec": SPEC,
           "orchestrator_rulings": {
               "1_run_count": ("90 RUNS ACCEPTED -- all eight matched floors in "
                               "stage 1 per the exp73 precedent (doc 62 SS18.2, "
                               "SS19.1)"),
               "2_kappa": ("kappa = 1 FIXED, RATIFIED with the four-point "
                           "rationale (doc 62 SS4)"),
               "3_engagement_bar": ("0.10 rad ACCEPTED as calibrated -- it fails "
                                    "all 34 E4 runs (doc 62 SS9.3)"),
               "4_relaxation_bound": ("NOT computed as pre-work; registered "
                                      "computable-on-demand. Its nesting argument "
                                      "already proves it monotone >= 0.9625 and "
                                      "makes the bug-hunt gate INERT, which is the "
                                      "only use the doc has for it (SS2.4, SS15)"),
               "5_directional_call": ("the declined directional call is RATIFIED "
                                      "-- exp75 places NO call (doc 62 SS13)"),
               "6_record": ("directional-call record: ONE held twice-drawn "
                            "(exp70), ONE missed (exp72 warm-start), ONE pending "
                            "(exp73 amplitude_hurts_T1)"),
               "7_lit_ordinal": "doc 62 is the THIRTEENTH literature check"},
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "numpy": np.__version__}
    check_dataset(store, rep)
    check_committed_imports(rep)
    check_features(fm, rep)
    check_anchor_codepath(fm, verbs, store, rep)
    check_intro_structure(fm, verbs, store, rep)
    check_identity_init(fm, verbs, rep)
    check_closures(fm, verbs, store, rep)
    ang, _t1 = arm_angles("A0-A-T2", fm)
    items2, _m, _p = arm_items("A0-A-T2", store)
    data2 = {"test": R69.compile_all(
        [i for i in items2 if i["split"] == "test"], ang)}
    R69.check_word_order(fm, verbs, data2, rep)       # doc 62 SS8.1, inherited
    check_copy(fm, verbs, store, rep)
    check_trace(fm, verbs, store, rep)
    check_shuffle(store, rep)
    check_engagement_level1(fm, verbs, store, rep)
    check_failing_directions(fm, verbs, store, rep)

    # ---- the probe, and doc 62 SS20.3 step 8's REALISED COST MULTIPLIER -------
    tiny = {}
    for arm in ALL_ARMS:
        r = train(arm, 0, store, verbs, tiny_epochs, fm, save=False, quiet=True)
        tiny[arm] = {"val_acc": r["val_acc"],
                     "heldout_test_acc": r["heldout_test_acc"],
                     "wall_sec": r["wall_sec"], "n_params": r["n_params"],
                     "L": r["L"], "T": r["T"], "init": r["init"],
                     "sec_per_epoch": r["wall_sec"] / float(tiny_epochs)}
        print("[smoke] %-30s %dep val=%.3f test=%.3f  %.0fs  (%d params, L=%d, "
              "T=%d, %s)" % (arm, tiny_epochs, r["val_acc"],
                             r["heldout_test_acc"], r["wall_sec"],
                             r["n_params"], r["L"], r["T"], r["init"]),
              flush=True)
    rep["probe"] = {"epochs": tiny_epochs, "per_arm": tiny}

    npar = {a: tiny[a]["n_params"] for a in ALL_ARMS}
    for a in ALL_ARMS:
        assert npar[a] == N_PARAMS_BY_L[uploads_of(a)], \
            ("%s carries %d parameters, registered %d (doc 62 SS7.3)"
             % (a, npar[a], N_PARAMS_BY_L[uploads_of(a)]))
    rep["parameter_budget_doc62_SS7_3"] = {
        "per_arm": npar,
        "note": ("doc 62 SS7.3: the anchors carry the COMMITTED 120 because they "
                 "ARE the committed model; the L = 2 cells carry 126 (+5.0 %) and "
                 "the L = 3 cells 132 (+10.0 %). Against 528 training items this "
                 "is THE SMALLEST CAPACITY PERTURBATION ANY RUNG IN THIS "
                 "PROGRAMME HAS REGISTERED, and SS22's sample-complexity row rests "
                 "on it.")}

    # the realised intro cost multiplier, MEASURED against the registered 1.20x
    mult = {}
    for T in TS:
        base = tiny["A0-A-T%d" % T]["sec_per_epoch"]
        for L in UPLOADS:
            for init in INITS:
                k = "RU-L%d-T%d-%s" % (L, T, init)
                mult[k] = tiny[k]["sec_per_epoch"] / base if base > 0 else None
    est, serial = {}, 0.0
    for arm in ALL_ARMS:
        secs = tiny[arm]["sec_per_epoch"] * MAX_EPOCHS
        est[arm] = {"projected_sec_per_run_single_process": secs,
                    "runs": N_SEEDS,
                    "projected_sec_total_single_process": secs * N_SEEDS}
        serial += secs * N_SEEDS
    rep["projected_wall_time_doc62_SS19"] = {
        "per_arm": est,
        "total_runs": len(ALL_ARMS) * N_SEEDS,
        "serial_sec_single_process": serial,
        "serial_hours_single_process": serial / 3600.0,
        "contention_factor_of_record": 2.0,
        "serial_hours_with_contention": serial / 3600.0 * 2.0,
        "hours_on_5_workers": serial / 3600.0 * 2.0 / 5.0,
        "doc62_SS19_2_table_estimate_hours_on_5_workers": 3.1,
        "realised_intro_cost_multiplier": mult,
        "registered_planning_multiplier_doc62_SS19_2": 1.20,
        "basis": ("linear extrapolation of the %d-epoch probe to MAX_EPOCHS = "
                  "%d, measured SINGLE-PROCESS on this machine. doc 62 SS19.3's "
                  "MEASURED contention factor of record -- 2.0x, settled on "
                  "completed exp73 anchor runs rather than extrapolated -- is "
                  "applied to the single-process projection, as SS19.3 requires."
                  % (tiny_epochs, MAX_EPOCHS)),
    }
    rep["assertions_passed"] = True
    rep["wall_time_sec"] = time.time() - t0
    json.dump(rep, open(SMOKE_OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[smoke] wrote %s -- ALL doc 62 SS8 / SS9.2 ASSERTIONS PASSED"
          % SMOKE_OUT)
    return rep


# ==================================================================== training
def train(arm, seed, store, verbs, max_epochs, fm, save=True, tag="",
          model=None, quiet=False):
    """The exp60/exp61/exp65/exp68/exp69/exp70/exp72/exp73 loop, UNCHANGED --
    Adam(lr 0.02), batch 8, 150 epochs, checkpoint selected on in-distribution
    val accuracy, held-out test read ONCE from the selected checkpoint.

    THE CALL ORDER (compile -> manual_seed -> build -> init scheme -> Adam ->
    Random(RNG_BASE + seed)) IS exp69's / exp73's VERBATIM, because doc 62 SS6.3
    expects the DRIFT ANCHOR to reproduce exp69's R1-A and R2-A BIT-EXACTLY and
    any reordering of RNG consumption would break that. The identity init draws
    from DEDICATED streams (doc 58 SS6.2) and is applied AFTER `build_model`, so
    not one committed draw moves.

    doc 62 SS8.3: THE ANCHOR RUNS THROUGH THIS ENTRY POINT with L = 1 -- a
    separate legacy path would make SS6.3's bit-exactness prove nothing about the
    re-uploading arms' pipeline.

    doc 62 SS9.3 / SS20.4(c): the run record carries the interleave blocks'
    parameter vector AT INIT, AT THE SELECTED EPOCH and AT THE FINAL EPOCH, so
    `reupload_engaged` is a MEASURED per-run quantity and SS11.3's harmonic budget
    is computed on the SELECTED checkpoint without a rerun.

    `model` is used ONLY by the smoke battery's failing-direction builds, so that
    a deliberately broken build is exercised through THE PRODUCTION LOOP."""
    cfg = ARM_CFG[base_arm(arm)]
    L = cfg["L"]
    items, meta, path = arm_items(arm, store)
    data = compile_arm(arm, items, seed, fm)
    torch.manual_seed(seed)
    if model is None:
        m = build_model(arm, verbs, seed, fm)
        init_info = apply_init_scheme(m, arm, seed)
    else:
        m = model
        init_info = {"scheme": "SUPPLIED MODEL -- smoke failing-direction build "
                              "only; never used on the production path"}
    v_init = m.v_theta()
    verb_init = m.verb_theta()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    rng = random.Random(RNG_BASE + seed)
    best = {"val": -1.0, "train": -1.0, "epoch": -1, "state": None}
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
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
            va = evaluate(m, data["val"])
        if va > best["val"]:
            with torch.no_grad():
                best = {"val": va, "train": evaluate(m, data["train"]),
                        "epoch": ep, "state": copy.deepcopy(m.state_dict())}
        if not quiet and (ep % 25 == 0 or ep == max_epochs):
            print("[%s s%d] ep %3d/%d val=%.3f best=%.3f %.0fs"
                  % (arm, seed, ep, max_epochs, va, best["val"],
                     time.time() - t0), flush=True)
    v_final = m.v_theta()
    verb_final = m.verb_theta()
    m.load_state_dict(best["state"])
    v_sel = m.v_theta()
    verb_sel = m.verb_theta()

    def maxabs(a, b):
        return None if not a else float(max(abs(x - y) for x, y in zip(a, b)))

    d_sel = maxabs(v_init, v_sel)
    d_fin = maxabs(v_init, v_final)
    engaged = None if d_sel is None else bool(d_sel >= ENGAGEMENT_BAR)

    te, rows = evaluate_with_traces(m, data["test"])
    budget = (harmonic_budget(m, data["test"], L) if L > 1 else None)
    pt = sha = None
    if save:
        os.makedirs(CKPT_DIR, exist_ok=True)
        pt = os.path.join(CKPT_DIR, "%s%s_s%d.pt" % (tag, arm, seed))
        torch.save(best["state"], pt)                  # doc 45 SS5
        sha = sha256_file(pt)
    return {"arm": arm, "base_arm": base_arm(arm), "family": cfg["family"],
            "is_floor": arm.startswith("S-shuffle"),
            "seed": seed, "part": cfg["part"], "T": cfg["T"],
            "encoding": cfg["enc"], "L": L, "init": cfg["init"],
            "kappa": 1.0, "role": cfg["role"],
            "init_scheme": init_info,
            "train_acc": best["train"], "val_acc": best["val"],
            "selected_epoch": best["epoch"], "heldout_test_acc": te,
            "val_minus_heldout": best["val"] - te,
            "gate_met": bool(best["train"] >= TRAIN_GATE and best["val"] >= VAL_GATE),
            "n_params": sum(p.numel() for p in m.parameters()),
            "wires": 2, "qubits_per_referent": 1,
            "n_train": len(data["train"]), "n_val": len(data["val"]),
            "n_test": len(data["test"]),
            "max_epochs": max_epochs, "lr": LR, "batch": BATCH,
            "dataset": path, "dataset_sha256": meta["dataset_sha256"],
            "ckpt_path": pt, "ckpt_sha256": sha,
            # ---- doc 62 SS9.3, the ENGAGEMENT record, per run ----------------
            "v_theta_init": v_init, "v_theta_selected": v_sel,
            "v_theta_final": v_final,
            "max_abs_dtheta_V": d_sel,
            "max_abs_dtheta_V_final": d_fin,
            "max_abs_dtheta_verb": maxabs(verb_init, verb_sel),
            "max_abs_dtheta_verb_final": maxabs(verb_init, verb_final),
            "engagement_bar": ENGAGEMENT_BAR,
            "reupload_engaged": engaged,
            "engagement_definition_doc62_SS9_3": (
                "reupload_engaged(cell, seed) = max_j max |theta_V(selected) - "
                "theta_V(init)| >= 0.10 rad. ABSOLUTE, PER RUN, on the maximum "
                "over the block's 6 or 12 parameters, NOT a mean over seeds. A "
                "cell with < 5/5 seeds engaged is INADMISSIBLE: its verdicts are "
                "stored as null, NEVER as FALSE (doc 62 SS9.4)."),
            # ---- doc 62 SS11.3, DESCRIPTIVE ---------------------------------
            "harmonic_budget_doc62_SS11_3": budget,
            "heldout_items": rows,
            "spec": SPEC,
            "wall_sec": time.time() - t0}


# ============================================= doc 62 SS6.3 -- THE DRIFT ANCHOR
def anchor_report(store):
    """doc 62 SS6.3 / SS12.1, BLOCKING AND READ FIRST, BOTH ARMS, 10 SEED-RUNS.

    `A0-A-T1`'s and `A0-A-T2`'s PER-SEED held-out accuracies must reproduce
    results_exp69.json :: R1-A and R2-A to |delta| <= 0.002, and each arm mean to
    |delta| <= 0.002. Both runs are deterministic given their seeds, so THE
    HONEST EXPECTATION IS |delta| = 0 EXACTLY, BIT-EXACT, 10/10; the tolerance
    absorbs a documented float/BLAS platform difference and any non-zero delta is
    reported with its cause identified. FIFTH ENFORCED USE of this gate; the
    first four returned bit-exact (doc 55 SS20.1, doc 56 SS18, doc 58 SS19,
    results_exp73_anchors.json 2026-08-28).

    doc 62 SS6.3.1: BOTH baselines are anchored because SS12.4's lift booleans
    read the best cell at each T against THAT T's committed mean, so both
    committed means are load-bearing. The extra 5 runs cost ~0.58 h serial, which
    is the cheapest possible insurance on a criterion.

    IF THIS FAILS, THE EXP75 PIPELINE HAS DRIFTED, every re-uploading number is a
    comparison against a moving baseline, and NOTHING ELSE IN THE RUN IS READ.
    Rule 3: this writes its OWN results file and no exp68/69/70/72/73 artefact is
    touched."""
    _items, meta, path = store
    src = json.load(open(EXP69_RESULTS, encoding="utf-8"))
    per_arm, all_ok, all_exact = {}, True, True
    for arm in ANCHOR_ARMS:
        source = ANCHOR_SOURCE[arm]
        ref = src["arms"][source]
        rows = []
        for s in range(N_SEEDS):
            ckj = os.path.join(CKPT_DIR, "%s_s%d.json" % (arm, s))
            if not os.path.exists(ckj):
                raise RuntimeError(
                    "the drift anchor is incomplete: %s is missing. doc 62 "
                    "SS20.3 step 9 -- the anchor runs FIRST and blocks "
                    "everything." % ckj)
            rows.append(json.load(open(ckj, encoding="utf-8")))
        rows.sort(key=lambda r: r["seed"])
        got = [r["heldout_test_acc"] for r in rows]
        want = ref["per_seed"]
        assert [r["seed"] for r in rows] == ref["seeds"], \
            "seed sets differ between exp75 %s and exp69 %s" % (arm, source)
        assert all(abs(w - p) < 5e-5
                   for w, p in zip(want, ANCHOR_PER_SEED_DOC62[arm])), \
            ("doc 62 SS6.3's printed per-seed values for %s do not match %s :: %s"
             % (arm, EXP69_RESULTS, source))
        d = [g - w for g, w in zip(got, want)]
        mean = float(np.mean(got))
        ok_seed = all(abs(x) <= ANCHOR_TOLERANCE for x in d)
        ok_mean = abs(mean - ANCHOR_MEAN_DOC62[arm]) <= ANCHOR_TOLERANCE
        exact = all(x == 0.0 for x in d)
        all_ok = all_ok and ok_seed and ok_mean
        all_exact = all_exact and exact
        per_arm[arm] = {
            "exp69_source_arm": source,
            "exp69_per_seed": want, "exp69_mean": ref["mean"],
            "exp69_sd": ref["sd"],
            "exp75_per_seed": got, "exp75_mean": mean,
            "exp75_sd": float(np.std(got, ddof=1)),
            "per_seed_delta": d,
            "max_abs_per_seed_delta": float(max(abs(x) for x in d)),
            "mean_delta_vs_exp69": mean - ref["mean"],
            "registered_mean_doc62_SS6_3": ANCHOR_MEAN_DOC62[arm],
            "mean_delta_vs_registered": mean - ANCHOR_MEAN_DOC62[arm],
            "bit_exact": bool(exact),
            "per_seed_within_tolerance": bool(ok_seed),
            "mean_within_tolerance": bool(ok_mean),
            "replicates": bool(ok_seed and ok_mean),
            "val_per_seed": [r["val_acc"] for r in rows],
            "train_per_seed": [r["train_acc"] for r in rows],
            "selected_epoch_per_seed": [r["selected_epoch"] for r in rows],
            "n_params": rows[0]["n_params"], "L": rows[0]["L"],
            "wall_sec_total": float(sum(r["wall_sec"] for r in rows)),
        }
    return {
        "component": "exp75 drift anchor -- doc 62 SS6.3, the BLOCKING gate",
        "spec": SPEC,
        "rule3_note": ("the anchor writes its OWN results file (doc 62 SS19.5, "
                       "'its own file by Rule 3'). No exp68/69/70/72/73 artefact "
                       "is touched, read for writing, or overwritten."),
        "dataset": path, "dataset_sha256": meta["dataset_sha256"],
        "exp69_source": EXP69_RESULTS,
        "per_arm": per_arm,
        "n_runs": len(ANCHOR_ARMS) * N_SEEDS,
        "tolerance": ANCHOR_TOLERANCE,
        "bit_exact_all": bool(all_exact),
        "anchor_replicates_committed": bool(all_ok),
        "both_baselines_anchored_doc62_SS6_3_1": (
            "doc 61 SS5.3 anchored ONE arm; exp75 anchors TWO, because SS12.4's "
            "lift booleans read the best cell at each T against THAT T's "
            "committed mean and BOTH committed means are therefore load-bearing. "
            "The extra 5 runs cost ~0.58 h serial, the cheapest possible "
            "insurance on a criterion."),
        "same_entry_point_doc62_SS8_3": (
            "the anchors are trained by exp75's OWN `train()` with the "
            "re-uploading intro OFF (L = 1). `ReuploadModel(verbs, seed, 1)` "
            "constructs the committed `QuantumStoryModel('A3')` VERBATIM and "
            "carries NO interleave blocks, so the parameter list, its order, the "
            "RNG consumption and the forward gate sequence are bit-identical "
            "(asserted at smoke). A separate legacy path would make this "
            "bit-exactness prove nothing about the re-uploading arms' pipeline."),
        "honest_expectation": (
            "BIT-EXACT, 10/10. Both runs are deterministic given their seeds and "
            "exp75 constructs the COMMITTED model verbatim with exp69's exact "
            "call order; the 0.002 tolerance exists only to absorb a documented "
            "float/BLAS platform difference and any non-zero delta is reported "
            "with its cause identified. FIFTH ENFORCED USE; the first four "
            "returned bit-exact."),
        "consequence_if_failed": (
            "doc 62 SS6.3 / SS14 row 1: THE EXP75 PIPELINE HAS DRIFTED, every "
            "re-uploading number is a comparison against a moving baseline, and "
            "NOTHING ELSE IN THE RUN IS READ. SS6.3.2's floor-omission argument "
            "for the anchor is VOID along with the run. HUNT THE BUILD; INTERPRET "
            "NOTHING."),
    }


# ======================================================================== main
def job_list(arms, seeds):
    """doc 62 SS20.4(a) -- THE DRIFT ANCHOR RUNS FIRST. The two anchor arms' ten
    seeds are the first ten jobs, so with `--nworkers 5` every worker's first two
    jobs are anchor seeds and the blocking gate completes before any trained cell
    finishes. The read-time gate lives in `--anchor-report` and in
    `exp75_aggregate.py`; the launch order is what makes it cheap."""
    anchor = [(a, s) for a in ANCHOR_ARMS if a in arms for s in range(seeds)]
    rest = [(a, s) for a in arms if a not in ANCHOR_ARMS for s in range(seeds)]
    return anchor + rest


def main():
    global DATA_A, CKPT_DIR, SMOKE_OUT, VSUITE, ANCHOR_OUT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(ALL_ARMS),
                    help="default is doc 62 SS19.1's full 90-run set: 10 anchor "
                         "runs, 40 trained cells, 40 matched floors")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-epochs", type=int, default=2)
    ap.add_argument("--anchor-report", action="store_true",
                    help="read the ten anchor checkpoints and write %s" % ANCHOR_OUT)
    ap.add_argument("--data", default=DATA_A)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--smoke-out", default=SMOKE_OUT)
    ap.add_argument("--anchor-out", default=ANCHOR_OUT)
    ap.add_argument("--vsuite", default=VSUITE)
    a = ap.parse_args()

    DATA_A, CKPT_DIR, SMOKE_OUT = a.data, a.ckpt_dir, a.smoke_out
    ANCHOR_OUT, VSUITE = a.anchor_out, a.vsuite
    if os.path.abspath(a.data) != os.path.abspath(R69.DATA_A):
        raise RuntimeError(
            "doc 62 SS6.2 / SS20.1: exp75 runs on exp69's FROZEN Part A and there "
            "is no exp75 datagen. A different dataset is doc 62 SS20.2's CONFIRM "
            "DRAW, which is owed only on a positive headline and is not part of "
            "this run.")

    store = load_part()
    verbs = sorted(store[1]["verb_inventory"].keys())
    assert len(verbs) == 6, \
        ("doc 62 SS7.3's parameter arithmetic assumes 6 verbs (6 x 15 + 2 x 15 = "
         "120); the dataset carries %d" % len(verbs))
    fm = feature_maps()
    os.makedirs(CKPT_DIR, exist_ok=True)

    if a.smoke:
        smoke(fm, verbs, store, tiny_epochs=a.smoke_epochs)
        return 0

    if a.anchor_report:
        rep = anchor_report(store)
        json.dump(rep, open(ANCHOR_OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n[ANCHOR GATE] doc 62 SS6.3, |delta| <= %.3f, honest expectation "
              "BIT-EXACT 10/10" % ANCHOR_TOLERANCE)
        for arm in ANCHOR_ARMS:
            p = rep["per_arm"][arm]
            print("  exp69 %-5s %s  mean %.6f"
                  % (p["exp69_source_arm"],
                     [round(x, 6) for x in p["exp69_per_seed"]], p["exp69_mean"]))
            print("  exp75 %-5s %s  mean %.6f"
                  % (arm, [round(x, 6) for x in p["exp75_per_seed"]],
                     p["exp75_mean"]))
            print("  delta       %s  max |d| %.2e  bit-exact=%s"
                  % ([round(x, 6) for x in p["per_seed_delta"]],
                     p["max_abs_per_seed_delta"], p["bit_exact"]))
        print("  anchor_replicates_committed = %s  ->  %s"
              % (rep["anchor_replicates_committed"],
                 "PROCEED" if rep["anchor_replicates_committed"]
                 else "THE RUN STOPS (doc 62 SS6.3)"))
        print("[ANCHOR GATE] wrote %s" % ANCHOR_OUT)
        return 0 if rep["anchor_replicates_committed"] else 1

    if not os.path.exists(SMOKE_OUT):
        raise RuntimeError(
            "%s missing -- doc 62 SS20.3 step 7: every SS8 and SS9 assertion, "
            "INCLUDING THE ONES DESIGNED TO FAIL, runs BEFORE any training. Run "
            "`python exp75_run.py --smoke`." % SMOKE_OUT)
    sm = json.load(open(SMOKE_OUT, encoding="utf-8"))
    if not sm.get("assertions_passed"):
        raise RuntimeError("the exp75 smoke gate did not pass -- nothing may train")

    arms = [x for x in a.arms.split(",") if x]
    unknown = [x for x in arms if base_arm(x) not in ARM_CFG]
    assert not unknown, "unknown arms: %r" % unknown

    jobs = job_list(arms, a.seeds)
    mine = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    print("[exp75 w%d/%d] arms=%d -> %d of %d runs (anchor first)"
          % (a.worker, a.nworkers, len(arms), len(mine), len(jobs)), flush=True)
    for arm, s in mine:
        ckj = os.path.join(CKPT_DIR, "%s%s_s%d.json" % (a.tag, arm, s))
        if os.path.exists(ckj):
            print("[ckpt] %s%s_s%d already done" % (a.tag, arm, s), flush=True)
            continue
        r = train(arm, s, store, verbs, MAX_EPOCHS, fm, save=True, tag=a.tag)
        r["tag"] = a.tag
        json.dump(r, open(ckj, "w", encoding="utf-8"))
        print("[exp75] %-30s s%d  train=%.3f val=%.3f HELD-OUT=%.4f  "
              "max|dV|=%s engaged=%s  (%.0fs)"
              % (arm, s, r["train_acc"], r["val_acc"], r["heldout_test_acc"],
                 ("%.4f" % r["max_abs_dtheta_V"]
                  if r["max_abs_dtheta_V"] is not None else "n/a"),
                 r["reupload_engaged"], r["wall_sec"]), flush=True)
    print("[exp75 w%d] done" % a.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
