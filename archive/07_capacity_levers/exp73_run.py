# -*- coding: utf-8 -*-
"""exp73_run.py -- doc 61: THE AMPLITUDE RUNG (D1 of doc 59's six-direction plan).

THE QUESTION (doc 61 SS0.1). Can a referent's whole (truncated, normalised) AraVec
vector be written onto its register as AMPLITUDES rather than compressed into one or
two angles -- and if so, what does the readout actually recover from it?

    |psi(x)> = sum_i x_i |i>,   x in R^{2^q},  ||x||_2 = 1

A3 puts PC1..PC8 on 3 qubits per referent (6 wires). A4 puts PC1..PC16 on 4 qubits
(8 wires) and is a CONDITIONAL SECOND STAGE (orchestrator ruling 2) -- it is DEFINED
here and NOT RUN in stage 1.

THIS IS NOT AN ADVANTAGE EXPERIMENT, AND FOR THIS ENCODING THAT IS SHARPER THAN AT
ANY PREVIOUS RUNG (doc 61 SS0.3). Every gate after intro is feature-independent, so
any outcome probability is p(x) = x^T M x = <W, v(x)> with v(x) = (x_i x_j)_{i<=j}:
the readout IS a quadratic form in the raw features and the matched classical model
IS a quadratic-kernel model on the same features -- the same object in another basis,
not a construction. The equivalence is TEXTBOOK (doc 61 SS4.1) and is not claimed as
new. **NO EXP73 NUMBER MAY APPEAR IN A SENTENCE CONTAINING "ADVANTAGE"**
(doc 52 SS1; doc 56 SS0.1; doc 58 SS12; doc 61 SS0.3 / SS13).

THE PRE-WORK ALREADY RETURNED A DEFLATIONARY RESULT (doc 61 SS0.5 / SS2.4). At A3 and
A4 the grade-1 reachable-accuracy bound SATURATES BY PROOF -- exactly 1.0000, and so
is every one of 200 size-matched null draws. The capacity instrument that graded every
previous rung DISCRIMINATES NOTHING here. Every exp73 criterion is therefore
measurement-vs-measurement, and `bug_hunt_triggered` is STRUCTURALLY INERT (SS12.1).

ORCHESTRATOR RULINGS (doc 61 status header, BINDING over the body)
=================================================================
  1. THE VERB BLOCK IS THE MONOLITHIC `Circuit4(2q, 3)`. It reproduces the committed
     block at BOTH q = 1 (`Circuit4(2,3)`, the FA/EA-ctrl block) and q = 2
     (`Circuit4(4,3)`, the F2/P-eps block), uses 408 (A3) / 552 (A4) parameters
     against the brickwork's 1800 / 3360, and measures 2.5x cheaper. THE BRICKWORK
     MOVES TO NOTED-NOT-RUN. doc 61 SS7.2 item 6's registered counts are superseded:
     **408 at A3**, asserted at smoke.
  2. COMPUTE: A4 IS A CONDITIONAL SECOND STAGE. It runs only if A3 binds at T = 1 or
     shows an init effect >= 0.06. STAGE 1 IS A3-ONLY.
  3. THE BATCH HOIST IS DECLINED -- no registration touches shared committed code.
     `_readout` therefore rebuilds the block matrices per item, exactly as the
     committed loop does. doc 61 SS16.3 is NOT implemented here.
  4. The directional call `amplitude_hurts_T1 = TRUE` STANDS (bar: best amplitude
     T = 1 cell <= 0.9220).
  5. Tier (iii) comparator grid FROZEN at C in {1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2}
     (seven values -- this supersedes SS8.2's six-value body text).

NO DATAGEN. THIS IS A CLOSED GATE, NOT AN OMISSION (doc 61 SS17.1)
==================================================================
Every arm runs on exp69's FROZEN Part A dataset `stories_exp69_QA.json`, sha-pinned
and asserted at smoke. exp73 generates NO Arabic: the inventory, the surfaces, the
verbs and the category are the committed ones, and WHAT IS NEW IS ONLY WHICH
PRINCIPAL COMPONENTS OF THE SAME VECTORS ARE READ. `exp73_datagen.py` does not exist
and may not be written.

STAGE 1 ARMS (doc 61 SS5.1 under ruling 2), 45 runs
  A0-W-T2               the DRIFT ANCHOR (SS5.3), BLOCKING, ordered FIRST.
                        The committed F2 register at T = 2, re-run through exp73's
                        OWN `train()` entry point with the amplitude intro and the
                        amplitude block OFF. Its per-seed held-out accuracies must
                        reproduce results_exp69.json :: R2-W to |delta| <= 0.002.
                        Honest expectation: BIT-EXACT, 5/5.       ->  5 runs
  A3-T{1,2}-{uniform,identity}                                    -> 20 runs
  S-shuffle-A3-T{1,2}-{uniform,identity}                          -> 20 runs

THE FLOORS ARE RUN UP FRONT, AND THAT IS THE CONSERVATIVE CHOICE. doc 61 SS15 stages
the floors behind a trigger test; with A4 deferred, the WORST CASE is four A3 cells
triggering, i.e. exactly the 20 floor runs launched here. Running all four now means
no verdict can ever be read past a missing floor (SS15.4), and it costs nothing that
the worst case would not have cost anyway. **The init is part of the procedure the
floor must match (SS15.1), so each of the four cells gets its OWN floor.**

VALIDITY, BOTH DIRECTIONS (doc 61 SS7, doc 48 SS10.6's discipline)
  SS7.2/1  the intro state equals x amplitude by amplitude; ||x||_2 = 1
  SS7.2/2  the intro has ZERO trainable parameters and consumes NO RNG
  SS7.2/3  per-REGISTER purity at intro = 1.0 (the two referents are a PRODUCT)
  SS7.2/4  per-WIRE intro purity matches the pre-computed inventory
  SS7.2/5  the monolithic block at q = 1 is bit-identical to the committed
           `Circuit4(2,3)` verb block -- AND at q = 2 to `Circuit4(4,3)` (ruling 1)
  SS7.2/6  trainable parameter count is EXACTLY 408 at A3 (ruling 1)
  SS7.2/7  identity-init: blocks at theta = 0 are I; sd in [0.08, 0.12]; the
           READOUT is NOT near-identity and ||q_pos - q_neg|| > 0.1
  SS7.2/8  the drift anchor runs through the SAME `train()` entry point
  SS7.2/9  the realised z reproduce `instrument_amplitude_bounds._pc_scores`
           (sha-pinned) and PC1/PC2 reproduce `load_raw` under its divisors
  SS7.3    dataset sha; SS7.4 closure NOT APPLICABLE and why; SS7.5 the committed
           values are IMPORTED, never re-typed.
  Every one of these has a FAILING DIRECTION and the failing directions are
  themselves asserted (`check_failing_directions`).

THE PRECISION SPLIT, INHERITED FROM exp70 VERBATIM. `torch_sv_sim.CDTYPE` is
complex64 and is LEFT ALONE -- the anchor's bit-exactness depends on it. A 1e-12
assertion is BELOW the simulator's resolution (~1.2e-07). So the intro is ALSO
recomputed in complex128 from the same data through the same gate sequence, held to
the runtime state at float32 tolerance so it is provably THE SAME CIRCUIT, and doc
61's 1e-9 / 1e-12 thresholds are asserted against that AS WRITTEN. Both numbers are
reported. The checkpointed purity traces remain the RUNTIME ones, from the same
forward pass that produces the answer (doc 61 SS9.3).

    python exp73_run.py --smoke                    # doc 61 SS7, ALL gates
    python exp73_run.py --worker i --nworkers 5    # anchor jobs are ordered FIRST
    python exp73_run.py --anchor-report            # writes results_exp73_anchors.json
    python exp73_comparator.py                     # doc 61 SS8, tiers (i)/(ii)/(iii)
    python exp73_aggregate.py                      # the ONLY results writer
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
import exp68_run as E                                 # F2Model, committed feature maps
import exp69_run as R69                               # compile path, floors, gates
import exp72_run as R72                               # doc 58 SS6.2 identity init
import instrument_sb_bounds as INS
import instrument_amplitude_bounds as IAB             # the committed amplitude maths
from exp42_run import grad_global_norm
from exp60_run import evaluate

SPEC = ("qnlp_private_docs/61_exp73_amplitude.md (REGISTERED 2026-08-28, "
        "orchestrator rulings 1-5 in the status header, BINDING over the body)")
SPEC_DOC = "61_exp73_amplitude.md"

DATA_A = R69.DATA_A                                   # stories_exp69_QA.json, FROZEN
EXP69_RESULTS = "results_exp69.json"
EXP70_RESULTS = "results_exp70.json"
EXP69_VSUITE = "results_exp69_WN_validation.json"
AMP_DIAGNOSTIC = IAB.OUT                              # diagnostic_amplitude_bounds.json
AMP_INSTRUMENT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "instrument_amplitude_bounds.py")

CKPT_DIR = "exp73_ckpt"
SMOKE_OUT = "results_exp73_smoke_validation.json"
ANCHOR_OUT = "results_exp73_anchors.json"
VSUITE = EXP69_VSUITE

DATASET_SHA_DOC61 = ("2b85822b544571969fd40cab12b03a4ac256b08aee914d2b9643b856"
                     "a6308c96")                      # doc 61 SS7.3

# Training config: the exp60/exp61/exp65/exp68/exp69/exp70/exp72 loop, UNCHANGED.
# Anything else here would put the drift anchor's bit-exactness out of reach.
LR, BATCH, MAX_EPOCHS = E.LR, E.BATCH, E.MAX_EPOCHS    # 0.02, 8, 150
TRAIN_GATE, VAL_GATE = E.TRAIN_GATE, E.VAL_GATE
RNG_BASE = E.RNG_BASE
N_SEEDS = E.N_SEEDS

ANCHOR_TOLERANCE = 0.002              # doc 61 SS5.3; honest expectation is 0.0
F32_FLOOR = 1e-6                      # exp70's measured complex64 agreement floor
LAYERS = 3                            # `Circuit4(n, 3)`, the committed depth

# doc 58 SS6.2, IMPORTED from the committed module and never re-typed (SS5.1.1).
IDENTITY_SIGMA = R72.IDENTITY_SIGMA
IDENTITY_RNG_BASE = R72.IDENTITY_RNG_BASE

# doc 61 SS5.2 -- the committed 2-qubit rungs, CITED and NOT RE-RUN. Every number is
# asserted against the committed results files at smoke (SS7.5) and never trusted
# from this table.
COMMITTED_DOC61_SS5_2 = {
    "F2-T1":    {"source": EXP69_RESULTS, "arm": "R1-W",    "mean": 0.9820},
    "F2-T2":    {"source": EXP69_RESULTS, "arm": "R2-W",    "mean": 0.7293},
    "P-eps-T1": {"source": EXP70_RESULTS, "arm": "P-eps-T1", "mean": 0.9660},
    "P-eps-T2": {"source": EXP70_RESULTS, "arm": "P-eps-T2", "mean": 0.8340},
}
COMMITTED_FLOORS_DOC61_SS15_2 = {
    "F2-T1":    {"source": EXP69_RESULTS, "arm": "R1-W",     "mean": 0.5073,
                 "sd": 0.0293},
    "F2-T2":    {"source": EXP69_RESULTS, "arm": "R2-W",     "mean": 0.5020,
                 "sd": 0.0283},
    "P-eps-T1": {"source": EXP70_RESULTS, "arm": "P-eps-T1", "mean": 0.5180,
                 "sd": 0.0445},
    "P-eps-T2": {"source": EXP70_RESULTS, "arm": "P-eps-T2", "mean": 0.5093,
                 "sd": 0.0359},
}

# doc 61 SS5.3: the anchor's registered per-seed values. Read PROGRAMMATICALLY from
# results_exp69.json :: R2-W; this table is the doc's printed copy and is asserted
# EQUAL to the file rather than used in its place.
ANCHOR_ARM = "A0-W-T2"
ANCHOR_SOURCE_ARM = "R2-W"
ANCHOR_PER_SEED_DOC61 = [0.716667, 0.770000, 0.746667, 0.726667, 0.686667]
ANCHOR_MEAN_DOC61 = 0.7293

# ------------------------------------------------------------------ the arms
# `rung` -> (qubits per referent, d = 2^q, registered parameter count under ruling 1)
RUNGS = {
    "A3": {"q": 3, "d": 8,  "n_params": 408, "var_captured": 0.5023,
           "d_ref": 36,  "d_joint": 1296,
           "stage": 1, "desc": "exp73 ARM -- PC1..PC8 on 3 qubits, 6 wires"},
    "A4": {"q": 4, "d": 16, "n_params": 552, "var_captured": 0.7195,
           "d_ref": 136, "d_joint": 18496,
           "stage": 2, "desc": ("exp73 ARM -- PC1..PC16 on 4 qubits, 8 wires. "
                                "CONDITIONAL SECOND STAGE (orchestrator ruling 2): "
                                "runs ONLY if A3 binds at T = 1 or shows an init "
                                "effect >= 0.06. NOT BUILT INTO STAGE 1.")},
}
INITS = ("uniform", "identity")

ARM_CFG = {}
for _r, _rc in RUNGS.items():
    for _t in (1, 2):
        for _i in INITS:
            ARM_CFG["%s-T%d-%s" % (_r, _t, _i)] = {
                "enc": "amplitude", "rung": _r, "qpr": _rc["q"], "d": _rc["d"],
                "T": _t, "init": _i, "part": "A", "stage": _rc["stage"],
                "family": "%s-%s" % (_r, _i),
                "role": "amplitude arm, %s init (doc 61 SS5.1)" % _i}
ARM_CFG[ANCHOR_ARM] = {
    "enc": "F2", "rung": None, "qpr": 2, "d": None, "T": 2, "init": "uniform",
    "part": "A", "stage": 1, "family": "A0-W",
    "role": ("THE DRIFT ANCHOR (doc 61 SS5.3) -- the committed F2 register at T = 2, "
             "through exp73's OWN train() entry point with the amplitude options "
             "OFF. BLOCKING, and it runs first.")}

STAGE1_TRAINED = ["A3-T%d-%s" % (t, i) for t in (1, 2) for i in INITS]
STAGE1_FLOORS = ["S-shuffle-%s" % a for a in STAGE1_TRAINED]
STAGE1_ARMS = [ANCHOR_ARM] + STAGE1_TRAINED + STAGE1_FLOORS
STAGE2_TRAINED = ["A4-T%d-%s" % (t, i) for t in (1, 2) for i in INITS]
FLOOR_OF = {a: "S-shuffle-%s" % a for a in STAGE1_TRAINED + STAGE2_TRAINED}


def base_arm(arm):
    return arm[len("S-shuffle-"):] if arm.startswith("S-shuffle-") else arm


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ======================================================== the amplitude feature map
def amplitude_maps():
    """doc 61 SS3.1's map, produced by CALLING the sha-pinned instrument -- never
    re-derived here. The formulas are `instrument_amplitude_bounds`'s and the
    instrument is the authority for the maths (doc 61 SS2, SS6.6).

        (a) PCA over the RAW 300-dim AraVec vectors of the FULL 44-noun inventory,
            mean-centred, numpy.linalg.svd -- load_raw's convention
        (b) z_k(r) = the k-th principal SCORE, UNRESCALED
        (c) SIGN PIN: PC1/PC2 to the frozen vec2 plane (load_raw's rule);
            PC3..PCd  sign(Vt[k, argmax_j |Vt[k,j]|]) = +1
        (d) TRUNCATE to d = 2^q, zero-pad to 2^q (vacuous here)
        (e) NORMALISE per noun: x(r) = z_{1:d}(r) / ||z_{1:d}(r)||_2
    """
    Z, var, prov = IAB._pc_scores(n_pc=16, fit_on="all44")
    conv = IAB.check_pca_convention(Z)
    out = {"Z": Z, "var_explained": [float(v) for v in var[:16]],
           "var_cumulative": [float(v) for v in np.cumsum(var[:16])],
           "pca_provenance": prov, "pca_convention_check": conv,
           "states": {}, "radii": {}, "features": {}}
    for rung, rc in RUNGS.items():
        d = rc["d"]
        st, rad = {}, {}
        for n in INS.HUMANS:
            x, r = IAB.amp_vector(Z[n], d)
            st[n], rad[n] = x, float(r)
        out["states"][rung] = st
        out["radii"][rung] = rad
        out["features"][rung] = {n: IAB.amp_features(st[n]) for n in INS.HUMANS}
    return out


def feature_maps():
    """The committed ANGLE maps (for the drift anchor) plus exp73's amplitude maps.

    `exp68_run.feature_maps` is IMPORTED and called unchanged -- shared code objects
    are the strongest guarantee the anchor has (exp70's argument, verbatim)."""
    fm = E.feature_maps()
    fm["amp"] = amplitude_maps()
    fm["zero_angles"] = {n: 0.0 for n in INS.HUMANS}
    fm["zero_angles_note"] = (
        "the amplitude arms carry NO angle parameterisation at all (doc 61 SS7.4). "
        "`exp42_compiler.compile_story` requires an angle per referent to build the "
        "intro spec, so the amplitude arms are compiled with an identically-zero "
        "angle dict. THE VALUE IS NEVER READ: `AmplitudeModel.apply_intro` keys on "
        "the referent NAME carried in the same intro tuple. Asserted at smoke -- "
        "perturbing the angle dict must leave every amplitude forward pass "
        "BIT-IDENTICAL.")
    return fm


def arm_angles(arm, fm):
    """The angle dict `exp42_compiler` consumes for this arm."""
    cfg = ARM_CFG[base_arm(arm)]
    if cfg["enc"] == "F2":
        return fm["theta_pc1"], fm["theta_pc2"]
    return fm["zero_angles"], None


def referent_features(arm, fm):
    """The per-referent vector the arm's circuit is LINEAR in -- the object doc 61
    SS8's function-class-matched comparator must consume.

      amplitude   v(x) = (x_i x_j)_{i<=j}, the OUTER SQUARE.   d_ref = d(d+1)/2
      F2          u(theta1) (x) u(theta2)                      d_ref = 9

    **THE OUTER SQUARE AND NOT THE AMPLITUDES.** Readouts are expectation values,
    QUADRATIC in amplitudes; a linear model on the amplitudes is a strictly WEAKER
    class than the circuit's and would MANUFACTURE A SPURIOUS CIRCUIT WIN. doc 56
    SS8.2's warning, and doc 61 SS4.1 records that it would have been "an easy and
    invisible error"."""
    cfg = ARM_CFG[base_arm(arm)]
    if cfg["enc"] == "F2":
        return E.referent_features("F2", fm)
    return dict(fm["amp"]["features"][cfg["rung"]])


# ============================================ doc 61 SS6.6 -- the state preparation
#
# A FIXED, DATA-VALUED, ZERO-PARAMETER unitary whose ONLY requirement is that it
# produce |psi(x)> exactly. It is a CIRCUIT, not a gate, and doc 61 SS6.6 insists it
# must not become a silent second architecture -- so it is built as the registered
# construction: a Moettoenen-style cascade of 2^q - 1 UNIFORMLY CONTROLLED RY
# rotations, one level per wire.
#
#   level k (k = 0 .. q-1) carries 2^k rotations, one per k-bit prefix p, acting on
#   wire k and controlled on wires 0..k-1 being in |p>. In the simulator's wire
#   ordering (wire 0 = most significant axis) the whole level is the BLOCK-DIAGONAL
#   2^(k+1) x 2^(k+1) matrix diag(RY(theta_k[0]), ..., RY(theta_k[2^k - 1])) applied
#   to wires (0..k) of the register.
#
#   Angles, from the binary norm tree:
#       A[q] = x  (SIGNED)
#       A[k][p] = hypot(A[k+1][2p], A[k+1][2p+1])           k < q, NON-NEGATIVE
#       theta_k[p] = 2 * atan2(A[k+1][2p+1], A[k+1][2p])
#   so cos(theta/2) = A[k+1][2p]/A[k][p] and sin(theta/2) = A[k+1][2p+1]/A[k][p], the
#   telescoping product over the q levels is A[q][i]/A[0][0] = x_i / ||x|| = x_i, and
#   THE SIGNS ARE CARRIED BY THE LAST LEVEL'S ROTATIONS. No global phase, no
#   post-hoc sign fixup, nothing that could be invisible to the outer-square feature
#   while being fatal to the circuit (doc 61 SS7.2 item 1's failing direction).
def uc_ry_angles(x):
    """The q levels of uniformly-controlled RY angles for the real unit vector x."""
    x = np.asarray(x, dtype=float)
    d = len(x)
    q = int(round(math.log(d, 2)))
    assert 2 ** q == d, "amplitude vector length %d is not a power of two" % d
    levels = [None] * q
    cur = x.copy()                                   # A[q], SIGNED
    for k in range(q - 1, -1, -1):
        a0, a1 = cur[0::2], cur[1::2]
        levels[k] = 2.0 * np.arctan2(a1, a0)
        cur = np.hypot(a0, a1)                       # A[k], non-negative
    return levels, float(cur[0])                     # cur[0] == ||x|| == 1


def _uc_ry_block(theta_row, dtype, backend):
    """blockdiag(RY(t_0), ..., RY(t_{2^k-1})) of size 2^(k+1). Constant DATA."""
    n = len(theta_row)
    if backend == "torch":
        M = torch.zeros((2 * n, 2 * n), dtype=dtype)
    else:
        M = np.zeros((2 * n, 2 * n), dtype=dtype)
    for p, t in enumerate(theta_row):
        c, s = math.cos(float(t) / 2.0), math.sin(float(t) / 2.0)
        M[2 * p, 2 * p] = c
        M[2 * p, 2 * p + 1] = -s
        M[2 * p + 1, 2 * p] = s
        M[2 * p + 1, 2 * p + 1] = c
    return M


def prep_gates(x, backend="torch"):
    """The ordered gate spec for |psi(x)>: [(U_k, local_wires)], k = 0..q-1."""
    levels, nrm = uc_ry_angles(x)
    dtype = sim.CDTYPE if backend == "torch" else np.complex128
    return ([( _uc_ry_block(levels[k], dtype, backend), tuple(range(k + 1)))
             for k in range(len(levels))], levels, nrm)


def prep_spec(x):
    """The preparation as a value-comparable structural spec (doc 61 SS6.6): a
    cascade of 2^q - 1 uniformly controlled RYs, ZERO trainable parameters."""
    levels, _n = uc_ry_angles(x)
    return tuple(("UC-RY", k, tuple(range(k + 1)),
                  tuple(round(float(t), 12) for t in levels[k]))
                 for k in range(len(levels)))


def n_prep_rotations(q):
    return 2 ** q - 1


# ====================================================================== the models
class AmplitudeModel(nn.Module):
    """doc 61 SS6 under orchestrator ruling 1 -- THE MONOLITHIC ANSATZ.

    A verb (or readout) block on the two referents' registers is ONE
    `Circuit4(2q, 3)` acting on all 2q wires jointly: agent register = wires
    0..q-1 of the pair, patient register = wires q..2q-1 (doc 61 SS6.1's layout).
    At q = 1 this IS the committed `Circuit4(2,3)` FA/EA-ctrl verb block and at
    q = 2 it IS the committed `Circuit4(4,3)` F2/P-eps block -- both asserted
    bit-identical at smoke (SS7.2 item 5, ruling 1).

    THE CONSTRUCTION ORDER IS THE COMMITTED ONE (`exp45_model.TwoQubitReferentModel`
    verbatim): one `torch.Generator` seeded by `seed`, verb blocks in the given verb
    order, then `q_pos`, then `q_neg`. Reordering RNG consumption is exactly what
    doc 61 SS5.3's anchor exists to catch, and the amplitude arms are held to the
    same discipline so their seeds mean what the committed seeds mean.

    THE INTRO HAS ZERO TRAINABLE PARAMETERS AND CONSUMES NO RNG (SS7.2 item 2). The
    per-noun preparations are CONSTANT DATA built once at construction from the
    registered map; they are stored in a plain dict, NOT as Parameters and NOT as
    buffers that could be trained or checkpointed as state."""

    def __init__(self, verbs, seed, states, q, layers=LAYERS):
        super().__init__()
        self.verbs = list(verbs)
        self.q = int(q)
        self.layers = int(layers)
        g = torch.Generator().manual_seed(seed)
        self.blocks = nn.ModuleDict(
            {v: sim.Circuit4(2 * self.q, layers, generator=g) for v in self.verbs})
        self.q_pos = sim.Circuit4(2 * self.q, layers, generator=g)
        self.q_neg = sim.Circuit4(2 * self.q, layers, generator=g)
        # --- the intro: constant data, no Parameters, no RNG ---------------
        self.amp = {n: np.asarray(x, dtype=float) for n, x in states.items()}
        self._prep = {}
        self._prep_spec = {}
        for n, x in self.amp.items():
            gates, _lv, nrm = prep_gates(x, backend="torch")
            assert abs(nrm - 1.0) < 1e-12, \
                ("doc 61 SS7.2 item 1: ||x||_2 = %.15f != 1 for %r -- the amplitude "
                 "map is not normalised" % (nrm, n))
            self._prep[n] = gates
            self._prep_spec[n] = prep_spec(x)

    def _reg(self, w):
        """CONTIGUOUS registers: referent w occupies wires q*w .. q*w + q - 1.
        For a verb event this makes the block's wire order
        (agent register, patient register) = (0..q-1, q..2q-1), which is doc 61
        SS6.1's registered layout."""
        return tuple(range(self.q * w, self.q * (w + 1)))

    def verb_matrix(self, v):
        return self.blocks[v].matrix()

    def apply_intro(self, psi, cs):
        """THE SINGLE SOURCE OF TRUTH for the amplitude intro. Keys on the referent
        NAME; the compiled angle is never read (see `fm['zero_angles_note']`)."""
        for w, noun, _angle in cs.intro:
            base = self.q * w
            for U, lw in self._prep[noun]:
                psi = sim.apply_gate(psi, U, tuple(base + i for i in lw))
        return psi

    def story_values(self, cs, verb_map=None):
        return _readout(self, cs, verb_map)


def is_amplitude(m):
    return isinstance(m, AmplitudeModel)


def qubits_per_referent(m):
    return m.q if is_amplitude(m) else 2


def reg_of(m, w):
    return m._reg(w) if is_amplitude(m) else (2 * w, 2 * w + 1)


def apply_intro(m, psi, cs):
    """Dispatch. The ANCHOR never routes its TRAINING through this function -- it
    calls the committed `exp68_run.F2Model.story_values` -- so this is a
    re-implementation for it, and `check_trace()` asserts it reproduces the
    committed intro BIT-FOR-BIT before any anchor number is read (exp70's
    discipline, verbatim)."""
    if is_amplitude(m):
        return m.apply_intro(psi, cs)
    for w, _noun, angle in cs.intro:                  # the committed F2 intro
        psi = sim.apply_gate(psi, sim.ry(angle), (2 * w,))
        if m.wide:
            psi = sim.apply_gate(psi, sim.ry(m.theta1[_noun]), (2 * w + 1,))
    return psi


def _readout(m, cs, verb_map=None):
    """The events + question path. Gate order mirrors
    `exp45_model.TwoQubitReferentModel.story_values` and
    `exp68_run.F2Model.story_values` EXACTLY; only the intro and the register width
    differ. Orchestrator ruling 3: the block matrices are rebuilt PER ITEM, as the
    committed loop does -- doc 61 SS16.3's batch hoist is DECLINED and is not
    implemented anywhere in this file."""
    vm = verb_map or {}
    cache = {}

    def block(v):
        v = vm.get(v, v)
        if v not in cache:
            cache[v] = m.verb_matrix(v)
        return cache[v]

    nw = qubits_per_referent(m) * cs.K
    psi = apply_intro(m, sim.init_state(nw), cs)
    for v, wa, wp in cs.events:
        psi = sim.apply_gate(psi, block(v), reg_of(m, wa) + reg_of(m, wp))
    vseq, qa, qp = cs.question
    qw = reg_of(m, qa) + reg_of(m, qp)
    for qv in reversed(vseq):
        psi = sim.apply_gate(psi, block(qv).conj().T, qw)
    return torch.stack([sim.prob_all_zero_on(
        sim.apply_gate(psi, eff.matrix().conj().T, qw), qw)
        for eff in (m.q_pos, m.q_neg)])


def build_model(arm, verbs, seed, fm):
    """doc 61 SS5.3: the ANCHOR constructs the COMMITTED parent class verbatim, with
    the identical constructor arguments exp69 used, because the anchor is expected to
    reproduce exp69 BIT-EXACTLY."""
    cfg = ARM_CFG[base_arm(arm)]
    if cfg["enc"] == "F2":                            # == exp69 R2-W, verbatim
        return E.F2Model(verbs, seed, fm["theta_pc2"])
    return AmplitudeModel(verbs, seed, fm["amp"]["states"][cfg["rung"]], cfg["qpr"])


def apply_init_scheme(m, arm, seed):
    """doc 61 SS5.1.1 -- BOTH INITS ARE ARMS (doc 59 SS2's ruling), not an optimiser
    variant. `uniform` is the committed recipe and `build_model` has already
    produced it. `identity` re-draws the VERB blocks as theta ~ N(0, 0.1^2) from a
    DEDICATED stream, leaving the READOUT exactly as the committed recipe made it.

    `exp72_run.apply_identity_init` and `exp72_run.identity_assertions` are CALLED,
    not copied -- doc 58 SS6.2 verbatim means the same code object, including the
    generator, the sigma, the sorted-verb draw order and the readout exclusion."""
    cfg = ARM_CFG[base_arm(arm)]
    if cfg["init"] == "uniform":
        return {"scheme": "uniform",
                "recipe": "every trainable block ~ Uniform[0, 2pi) -- the committed "
                          "recipe (doc 61 SS5.1.1)",
                "committed_stream_untouched": True}
    touched = R72.apply_identity_init(m, seed, m.verbs, [])
    rep = R72.identity_assertions(m, m.verbs, [], seed)
    rep.update({"scheme": "identity", "blocks_reinitialised": touched,
                "recipe": ("verb blocks ~ N(0, 0.1^2); READOUT (q_pos, q_neg) "
                           "UNCHANGED at Uniform[0, 2pi) -- doc 58 SS6.2 verbatim, "
                           "doc 61 SS5.1.1 / SS5.1.2"),
                "readout_exclusion_reason": (
                    "doc 58 SS6.2.2 / doc 61 SS5.1.2: two readout blocks at identity "
                    "would be IDENTICAL, v_yes == v_no for every item, the model "
                    "would output exactly 0.5 everywhere, and the readout would be a "
                    "NULL INSTRUMENT at initialisation. The argument holds verbatim "
                    "at 6 and 8 wires."),
                "imported_from": "exp72_run.apply_identity_init (doc 58 SS6.2)"})
    return rep


# ============================================== doc 61 SS9.3 -- the two purity traces
def wire_purities(psi, nwires):
    """Per-WIRE Tr rho^2 -- the committed trace (doc 55 SS18.3), read against doc 61
    SS9.2's pre-computed inventory as its baseline, NOT against 1."""
    return R69.wire_purities(psi, nwires)


def register_purities(psi, regs, nwires):
    """Per-REGISTER Tr rho_reg^2 -- NEW at doc 61 SS9.3, and the one that carries the
    cross-referent story: 1.0 at intro BY CONSTRUCTION (the two referents are a
    product), and below 1 after the first verb block IFF the block has actually
    correlated the two referents."""
    out = []
    for reg in regs:
        rest = [w for w in range(nwires) if w not in reg]
        T = torch.permute(psi, list(reg) + rest).reshape(2 ** len(reg), -1)
        rho = T @ T.conj().transpose(0, 1)
        out.append(float((rho.abs() ** 2).sum()))
    return out


def forward_traced(m, cs):
    """doc 61 SS9.3: the forward pass with BOTH purity traces taken FROM THE SAME
    PASS that produces the answer, at FOUR stages."""
    cache = {}

    def block(v):
        if v not in cache:
            cache[v] = m.verb_matrix(v)
        return cache[v]

    nw = qubits_per_referent(m) * cs.K
    regs = [reg_of(m, w) for w in range(cs.K)]

    def snap(psi):
        return {"wire": wire_purities(psi, nw),
                "register": register_purities(psi, regs, nw)}

    psi = apply_intro(m, sim.init_state(nw), cs)
    trace = {"after_intro": snap(psi)}
    for k, (v, wa, wp) in enumerate(cs.events):
        psi = sim.apply_gate(psi, block(v), reg_of(m, wa) + reg_of(m, wp))
        trace["after_event%d" % (k + 1)] = snap(psi)
    vseq, qa, qp = cs.question
    qw = reg_of(m, qa) + reg_of(m, qp)
    for qv in reversed(vseq):
        psi = sim.apply_gate(psi, block(qv).conj().T, qw)
    trace["at_readout"] = snap(psi)
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


# ============================================ the complex128 recomputation (SS7.2)
def _apply64(vec, U, wires, nw):
    """`torch_sv_sim.apply_gate`'s contraction, in complex128 (exp70's helper)."""
    k = len(wires)
    T = vec.reshape((2,) * nw)
    Ut = U.reshape((2,) * (2 * k))
    T = np.tensordot(Ut, T, axes=(list(range(k, 2 * k)), list(wires)))
    T = np.moveaxis(T, list(range(k)), list(wires))
    return T.reshape(-1)


def intro_state_f64(m, cs):
    """`apply_intro` in complex128, from the model's OWN data, through the SAME gate
    sequence -- so doc 61's 1e-9 / 1e-12 thresholds can be asserted AS REGISTERED
    without loosening them to the complex64 platform floor."""
    nw = qubits_per_referent(m) * cs.K
    v = np.zeros(2 ** nw, dtype=np.complex128)
    v[0] = 1.0
    if is_amplitude(m):
        for w, noun, _a in cs.intro:
            base = m.q * w
            gates, _lv, _n = prep_gates(m.amp[noun], backend="numpy")
            for U, lw in gates:
                v = _apply64(v, U, tuple(base + i for i in lw), nw)
        return v
    for w, noun, angle in cs.intro:
        v = _apply64(v, _ry64(float(angle)), (2 * w,), nw)
        if m.wide:
            v = _apply64(v, _ry64(float(m.theta1[noun])), (2 * w + 1,), nw)
    return v


def _ry64(t):
    c, s = math.cos(t / 2.0), math.sin(t / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def wire_purities64(v, nw):
    T = np.asarray(v).reshape((2,) * nw)
    out = []
    for w in range(nw):
        p = np.moveaxis(T, w, 0).reshape(2, -1)
        rho = p @ p.conj().T
        out.append(float((np.abs(rho) ** 2).sum()))
    return out


def register_purities64(v, regs, nw):
    T = np.asarray(v).reshape((2,) * nw)
    out = []
    for reg in regs:
        rest = [w for w in range(nw) if w not in reg]
        M = np.transpose(T, list(reg) + rest).reshape(2 ** len(reg), -1)
        rho = M @ M.conj().T
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
    items, meta, path = store
    T = ARM_CFG[base_arm(arm)]["T"]
    sel = [i for i in items if int(i.get("T", 1)) == T]
    assert sel, "no items at T=%d in %s" % (T, path)
    return sel, meta, path


def compile_arm(arm, items, seed, fm):
    ang, _t1 = arm_angles(arm, fm)
    use = R69.relabel_shuffle_q(items, seed) if arm.startswith("S-shuffle") else items
    return {s: R69.compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}


# ================================================== doc 61 SS7 -- the smoke gates
def check_dataset(store, rep):
    """doc 61 SS7.3 / SS17.1. The dataset is exp69's, BYTE-IDENTICAL, and its sha is
    asserted here so 'nothing changed' is CHECKED rather than assumed."""
    _items, meta, path = store
    ref = json.load(open(EXP69_RESULTS, encoding="utf-8"))
    exp69_sha = ref["datasets"]["A"]["sha256"]
    assert meta["dataset_sha256"] == exp69_sha, \
        ("exp73 is not running on exp69's Part A: %s vs %s"
         % (meta["dataset_sha256"], exp69_sha))
    assert meta["dataset_sha256"] == DATASET_SHA_DOC61, \
        ("the dataset sha does not match doc 61 SS7.3's registered value: %s vs %s"
         % (meta["dataset_sha256"], DATASET_SHA_DOC61))
    vs = json.load(open(VSUITE, encoding="utf-8"))
    assert vs.get("kill_gate_passed") is True, \
        "exp69's V-suite kill gate did not pass -- its results may not be re-used"
    rep["dataset"] = {
        "path": path, "sha256": meta["dataset_sha256"],
        "registered_sha_doc61_SS7_3": DATASET_SHA_DOC61,
        "matches_exp69_results": True,
        "vsuite_source": VSUITE, "vsuite_kill_gate_passed": True,
        "vsuite_status": ("INHERITED from exp69 on the strength of byte identity "
                          "(doc 61 SS7.1)"),
        "no_datagen_doc61_SS17_1": (
            "exp73 generates NO Arabic. The dataset is exp69's, byte-identical and "
            "sha-pinned, already V-suite-cleared; the noun inventory is the "
            "committed 44 and the category is the committed `nonoccupational` "
            "(doc 54 SS5.2, user-confirmed 2026-08-23). WHAT IS NEW IS ONLY WHICH "
            "PRINCIPAL COMPONENTS OF THE SAME VECTORS ARE READ, so doc 50 SS5.2's "
            "native-speaker gate does not reopen. Stated explicitly because "
            "'nothing changed' is exactly the claim that should be checked."),
    }
    return rep


def check_instrument_provenance(rep):
    """doc 61 SS7.2 item 9 / SS2: the amplitude maths comes from a SHA-PINNED
    instrument, CALLED and not copied."""
    got = sha256_file(AMP_INSTRUMENT)
    diag = json.load(open(AMP_DIAGNOSTIC, encoding="utf-8"))
    assert got == diag["script_sha256"], \
        ("instrument_amplitude_bounds.py has changed since "
         "diagnostic_amplitude_bounds.json was produced (%s vs %s) -- exp73's "
         "features would not be the ones the bounds were computed over"
         % (got, diag["script_sha256"]))
    assert diag["selftest"]["all_passed"] is True, \
        "the amplitude instrument's selftest did not pass; its maps are not usable"
    rep["instrument_provenance"] = {
        "instrument": os.path.basename(AMP_INSTRUMENT),
        "instrument_sha256": got,
        "diagnostic": os.path.basename(AMP_DIAGNOSTIC),
        "diagnostic_declares_sha256": diag["script_sha256"],
        "selftest_all_passed": True,
        "anchors_reproduced_by_the_instrument": diag["selftest"]["anchors"],
        "note": ("doc 61 SS2: the search, the exact R^3 enumeration, the sign-flip "
                 "invariant, the twin-mode seeding and the null discipline are the "
                 "COMMITTED ones, CALLED and not copied. exp73 calls `_pc_scores`, "
                 "`amp_vector`, `amp_features` and `check_pca_convention` from this "
                 "module rather than re-deriving any formula."),
    }
    return rep


def check_features(fm, rep):
    """doc 61 SS7.2 item 9 + SS2.7 + SS3.1. The committed raw-PCA check is called
    unchanged; exp73 adds the truncation, the sign pin and the normalisation."""
    E.check_features(fm, rep)
    amp = fm["amp"]
    conv = amp["pca_convention_check"]
    assert conv["reproduces_load_raw_exactly"], \
        ("doc 61 SS2.7 FAILED: the 16-component PCA's first two components do not "
         "reproduce load_raw's normalised coordinates (%.3e)"
         % conv["max_abs_error_vs_load_raw_normalised_coords"])
    assert conv["divisors_match"], "load_raw's max-abs divisors are not re-derivable"

    diag = json.load(open(AMP_DIAGNOSTIC, encoding="utf-8"))
    per_rung = {}
    for rung, rc in RUNGS.items():
        st = amp["states"][rung]
        nrm = np.array([float(np.linalg.norm(st[n])) for n in INS.HUMANS])
        assert float(np.abs(nrm - 1.0).max()) < 1e-12, \
            ("doc 61 SS7.2 item 1: ||x||_2 != 1 at %s (worst %.3e)"
             % (rung, float(np.abs(nrm - 1.0).max())))
        feat = amp["features"][rung]
        dref = len(next(iter(feat.values())))
        assert dref == IAB.d_ref_of(rc["d"]) == rc["d_ref"], \
            ("per-referent feature dim %d != registered %d at %s"
             % (dref, rc["d_ref"], rung))
        rad = np.array([amp["radii"][rung][n] for n in INS.HUMANS])
        # the truncation's cost, recomputed rather than quoted (doc 61 SS3.3)
        cum = amp["var_cumulative"][rc["d"] - 1]
        assert abs(cum - rc["var_captured"]) < 5e-4, \
            ("cumulative variance at %s is %.4f, doc 61 SS2.3 registers %.4f"
             % (rung, cum, rc["var_captured"]))
        # and it must agree with the instrument's own diagnostic
        ec = diag["encoding_cost"][rung]
        assert ec["d_per_referent_feature"] == rc["d_ref"], (rung, ec)
        assert ec["d_joint"] == rc["d_joint"], (rung, ec)
        per_rung[rung] = {
            "qubits_per_referent": rc["q"], "d_truncated": rc["d"],
            "d_per_referent_feature": dref, "d_joint": rc["d_joint"],
            "free_real_coordinates_per_referent": rc["d"] - 1,
            "variance_captured_cumulative": cum,
            "variance_discarded_by_truncation": 1.0 - cum,
            "max_abs_norm_error": float(np.abs(nrm - 1.0).max()),
            "radius_min": float(rad.min()), "radius_median": float(np.median(rad)),
            "radius_max": float(rad.max()),
        }
    rep["amplitude_features"] = {
        "per_rung": per_rung,
        "map_doc61_SS3_1": (
            "(a) PCA over the RAW 300-dim AraVec vectors of the FULL 44-noun "
            "inventory, mean-centred, numpy.linalg.svd -- load_raw's convention; "
            "(b) z_k(r) = the k-th principal SCORE, UNRESCALED; (c) SIGN PIN: "
            "PC1/PC2 pinned to the frozen vec2 plane as load_raw does, PC3..PCd by "
            "sign(Vt[k, argmax_j |Vt[k,j]|]) = +1, ties -> lowest j; (d) TRUNCATE to "
            "d = 2^q then zero-pad to 2^q (vacuous here); (e) NORMALISE per noun."),
        "pca_convention_check": conv,
        "sign_pin_pc3_onwards": amp["pca_provenance"]["sign_pin_pc3_onwards"],
        "var_explained_per_pc": amp["var_explained"],
        "var_cumulative": amp["var_cumulative"],
        "information_cost_doc61_SS3_3": (
            "THE AMPLITUDE MAP IS LOSSY TWICE, and doc 61 SS1.3 refuses to call it a "
            "strict enrichment of F2. (a) TRUNCATION discards 1 - cumvar(d) of the "
            "AraVec variance: 0.4977 at A3, 0.2805 at A4. (b) UNIT-l2 NORMALISATION "
            "discards the RADIUS, exactly one real number per referent -- the same "
            "quantity FA discarded relative to F2, and it is a genuinely varying "
            "one (6.33-27.82 at A3, 8.20-29.33 at A4 over the 24 nouns). Per-referent "
            "FREE COORDINATES: FA 1, F2/P-eps 2, A3 7, A4 15."),
        "why_the_sign_pin_is_safe_doc61_SS3_2": (
            "bounds are invariant to every sign choice (a per-component flip is the "
            "special case A = diag(+-1) of the reparameterisation lemma, asserted at "
            "d = 2 on the exact path and reported at d = 4, 8, 16). A TRAINED CIRCUIT "
            "IS NOT invariant, which is the whole reason the convention is pinned "
            "rather than left to numpy."),
    }
    return rep


def check_intro_states(fm, verbs, store, rep):
    """doc 61 SS7.2 items 1-4, ALL BLOCKING, and the sharpest gates in this file.

    The compiled intro statevector must equal |psi(x)> AMPLITUDE BY AMPLITUDE. A
    global-sign slip on one amplitude, a wire-order slip or a half-angle error all
    fail here immediately -- and item 1's failing direction is explicit: THE
    OUTER-SQUARE FEATURE IS SIGN-BLIND BUT THE STATE IS NOT, so a sign error would
    be invisible to the bound and fatal to the circuit."""
    diag = json.load(open(AMP_DIAGNOSTIC, encoding="utf-8"))
    out = {}
    for rung in sorted(RUNGS):
        arm = "%s-T1-uniform" % rung
        items, _m, _p = arm_items(arm, store)
        ang, _a1 = arm_angles(arm, fm)
        data = R69.compile_all([i for i in items if i["split"] == "test"][:8], ang)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        st = fm["amp"]["states"][rung]
        inv = diag["register_purity_inventory"][rung]
        worst, worst64, worst_rt64 = 0.0, 0.0, 0.0
        wpur64, rpur64 = 0.0, 0.0
        with torch.no_grad():
            for cs in data:
                psi = apply_intro(m, sim.init_state(m.q * cs.K), cs)
                v = psi.reshape(-1).detach().numpy()
                v64 = intro_state_f64(m, cs)
                worst_rt64 = max(worst_rt64, float(np.abs(v - v64).max()))
                names = [n for _w, n, _a in cs.intro]
                ref = st[names[0]]
                for n in names[1:]:
                    ref = np.kron(ref, st[n])
                worst = max(worst, float(np.abs(np.real(v) - ref).max()),
                            float(np.abs(np.imag(v)).max()))
                worst64 = max(worst64, float(np.abs(np.real(v64) - ref).max()),
                              float(np.abs(np.imag(v64)).max()))
                # SS7.2 item 3 -- per-REGISTER purity = 1.0 (a PRODUCT of two states)
                nw = m.q * cs.K
                regs = [m._reg(w) for w in range(cs.K)]
                rp = register_purities64(v64, regs, nw)
                rpur64 = max(rpur64, float(max(abs(p - 1.0) for p in rp)))
                # SS7.2 item 4 -- per-WIRE purity vs the PRE-COMPUTED inventory
                wp = wire_purities64(v64, nw)
                for k, n in enumerate(names):
                    want = inv["per_noun"][n]["per_wire_purity"]
                    got = wp[m.q * k:m.q * (k + 1)]
                    wpur64 = max(wpur64, float(max(abs(a - b)
                                                   for a, b in zip(got, want))))
        assert worst64 < 1e-12, \
            ("doc 61 SS7.2 item 1 FAILED for %s: the intro state differs from "
             "|psi(x)> by %.3e in complex128" % (rung, worst64))
        assert worst < F32_FLOOR, \
            ("doc 61 SS7.2 item 1 FAILED for %s at runtime: %.3e" % (rung, worst))
        assert worst_rt64 < F32_FLOOR, \
            ("%s: the complex128 recomputation is not the same circuit as the "
             "runtime one (%.3e)" % (rung, worst_rt64))
        assert rpur64 < 1e-9, \
            ("doc 61 SS7.2 item 3 FAILED for %s: per-REGISTER intro purity departs "
             "from 1.0 by %.3e -- an entangler has been introduced at intro and "
             "this is a different experiment" % (rung, rpur64))
        assert wpur64 < 1e-9, \
            ("doc 61 SS7.2 item 4 FAILED for %s: per-WIRE intro purity differs from "
             "diagnostic_amplitude_bounds.json :: register_purity_inventory by %.3e "
             "-- the register was prepared in a different state than the bound was "
             "computed for" % (rung, wpur64))
        out[rung] = {
            "max_abs_diff_vs_psi_x_runtime": worst,
            "max_abs_diff_vs_psi_x_float64": worst64,
            "runtime_vs_float64_max_abs": worst_rt64,
            "per_register_intro_purity_max_abs_dev_from_1": rpur64,
            "per_wire_intro_purity_max_abs_dev_vs_inventory": wpur64,
            "n_uniformly_controlled_ry": n_prep_rotations(RUNGS[rung]["q"]),
            "inventory_min_wire_purity": inv["min_wire_purity_over_the_inventory"],
        }
    rep["intro_state_assertions"] = {
        "per_rung": out,
        "thresholds": {"amplitude_by_amplitude_float64": 1e-12,
                       "runtime_platform_floor": F32_FLOOR,
                       "purity_float64": 1e-9},
        "preparation_doc61_SS6_6": (
            "the intro is a FIXED, DATA-VALUED, ZERO-PARAMETER unitary and its only "
            "requirement is that it produce |psi(x)> exactly. It is built as a "
            "Moettoenen-style cascade of 2^q - 1 UNIFORMLY CONTROLLED RY rotations, "
            "one level per wire, with the SIGNS carried by the last level's "
            "rotations -- so it is visibly A CIRCUIT and not a single opaque gate, "
            "and it cannot become a silent second architecture. NO EXP73 VERDICT "
            "DEPENDS ON WHICH PREPARATION CIRCUIT IS USED, AND NONE MAY."),
        "precision_note": (
            "the simulator is complex64 END TO END (torch_sv_sim.CDTYPE, unchanged "
            "since exp40 and LEFT ALONE here because the drift anchor's "
            "bit-exactness depends on it), so the runtime agreement floor is "
            "~1.2e-07 and doc 61's 1e-9 / 1e-12 thresholds are BELOW the "
            "simulator's resolution. Loosening a registered threshold to make a gate "
            "pass would be changing a gate after seeing the platform, so the build "
            "does the other thing: the intro is recomputed in complex128 from the "
            "model's OWN data through the SAME gate sequence, held to the runtime "
            "state at float32 tolerance so it is provably the same circuit, and doc "
            "61's thresholds are asserted against it AS WRITTEN. Both numbers are "
            "reported. exp70's precision split, verbatim."),
        "sign_blindness_note_doc61_SS7_2_1": (
            "THE FAILING DIRECTION IS THE POINT: the outer-square feature v(x) is "
            "invariant under x -> -x, so a sign error on any amplitude would be "
            "INVISIBLE TO THE BOUND and FATAL TO THE CIRCUIT. The assertion is on "
            "the STATE, amplitude by amplitude, and `check_failing_directions` "
            "demonstrates that flipping one amplitude's sign fires it."),
    }
    return rep


def check_zero_parameter_intro(fm, verbs, store, rep):
    """doc 61 SS7.2 item 2, BOTH DIRECTIONS: the intro registers NO parameters and
    consumes NO RNG; and doc 61 SS5.1.3 -- 'intro init unchanged' is vacuous BY
    CONSTRUCTION, which needs saying rather than assuming."""
    out = {}
    for rung, rc in RUNGS.items():
        torch.manual_seed(0)
        states = fm["amp"]["states"][rung]
        before = torch.get_rng_state()
        m = AmplitudeModel(verbs, 0, states, rc["q"])
        # the model as a whole consumes only its OWN Generator, not the global one
        after_build = torch.get_rng_state()
        assert torch.equal(before, after_build), \
            ("doc 61 SS7.2 item 2: model construction consumed the GLOBAL torch RNG "
             "stream at %s" % rung)
        # and rebuilding only the preparations consumes nothing at all
        before2 = torch.get_rng_state()
        for n in INS.HUMANS:
            prep_gates(states[n], backend="torch")
        assert torch.equal(before2, torch.get_rng_state()), \
            "the state preparation consumed RNG draws"
        names = {nm for nm, _p in m.named_parameters()}
        assert not any(nm.startswith("_prep") or "amp" in nm for nm in names), \
            "the preparation registered trainable parameters"
        assert not list(m.buffers()), \
            ("the preparation registered buffers -- it must be plain constant data, "
             "not model state that could be checkpointed or trained")
        n_par = sum(p.numel() for p in m.parameters())
        want = rc["n_params"]
        assert n_par == want, \
            ("doc 61 SS7.2 item 6 UNDER ORCHESTRATOR RULING 1: %s carries %d "
             "trainable parameters, registered %d" % (rung, n_par, want))
        per_block = 3 * (3 * (2 * rc["q"]) - 1)
        assert n_par == (len(verbs) + 2) * per_block, \
            ("parameter accounting drifted at %s: %d != (%d verbs + 2 readout) x %d"
             % (rung, n_par, len(verbs), per_block))
        out[rung] = {"n_trainable_params": n_par,
                     "registered_doc61_ruling1": want,
                     "params_per_block_Circuit4_2q_3": per_block,
                     "n_blocks": len(verbs) + 2,
                     "n_verbs": len(verbs),
                     "wires": 2 * rc["q"],
                     "intro_trainable_params": 0,
                     "intro_rng_draws": 0,
                     "n_buffers": 0}
    rep["parameter_budget"] = {
        "per_rung": out,
        "ruling_1_note": (
            "ORCHESTRATOR RULING 1 SUPERSEDES doc 61 SS7.2 item 6's registered 1800 / "
            "3360: the verb block is the MONOLITHIC `Circuit4(2q, 3)`, so the counts "
            "are 408 (A3) and 552 (A4) -- 8 blocks (6 verbs + q_pos + q_neg) x "
            "3 layers x (3 * 2q - 1) parameters. The brickwork moves to "
            "NOTED-NOT-RUN and is not implemented in this file."),
        "coefficient_budget_doc61_SS6_5": (
            "408 trainable parameters against 528 training items at A3 -- milder "
            "than the brickwork's 1800 but still the larger ansatz against the "
            "committed register's 264. The held-out split is the instrument that "
            "detects the exposure and the val-minus-held-out gap is reported PER "
            "SEED (doc 61 SS10.6, SS19 row 2). Comparison points: exp69 F2 gap "
            "0.0353, FA 0.0813."),
        "no_learned_intro_doc61_SS5_1_3": (
            "THE AMPLITUDE STATE PREPARATION IS DATA-VALUED WITH ZERO TRAINABLE "
            "PARAMETERS -- the same standing as the committed RY / CRY intros. "
            "'Intro init unchanged' is VACUOUS BY CONSTRUCTION, and doc 61 SS5.1.3 "
            "says so twice rather than letting it pass as an assertion. A trained "
            "(variational) intro is NOTED, NOT RUN (SS14 item 6) and would void the "
            "bounds' applicability entirely."),
    }
    return rep


def check_block_reduction(verbs, rep):
    """doc 61 SS7.2 item 5 under ruling 1 -- THE DRIFT-ANCHOR LOGIC FOR THE ANSATZ.

    The monolithic block must be BIT-IDENTICAL to the committed verb block at BOTH
    q = 1 (`exp42_models.QuantumStoryModel('A3')`'s `Circuit4(2,3)`) and q = 2
    (`exp45_model.TwoQubitReferentModel('A4')`'s `Circuit4(4,3)`) ON THE SAME
    PARAMETER DRAW. A family that does not reduce to the committed block is a new
    ansatz wearing its name."""
    from exp42_models import QuantumStoryModel
    from exp45_model import TwoQubitReferentModel
    out = {}
    for q, mk in ((1, lambda: QuantumStoryModel("A3", verbs, 0)),
                  (2, lambda: TwoQubitReferentModel("A4", verbs, 0, layers=LAYERS))):
        torch.manual_seed(0)
        committed = mk()
        torch.manual_seed(0)
        mono = AmplitudeModel(verbs, 0, {n: np.eye(2 ** q)[0]
                                         for n in INS.HUMANS}, q)
        worst_theta, worst_mat = 0.0, 0.0
        with torch.no_grad():
            for v in verbs:
                a = (committed.blocks[v].theta if hasattr(committed, "blocks")
                     else None)
                assert a is not None, "the committed model has no per-verb block dict"
                b = mono.blocks[v].theta
                worst_theta = max(worst_theta, float((a - b).abs().max()))
                worst_mat = max(worst_mat, float(
                    (committed.verb_matrix(v) - mono.verb_matrix(v)).abs().max()))
            for name in ("q_pos", "q_neg"):
                a = getattr(committed, name)
                b = getattr(mono, name)
                worst_theta = max(worst_theta, float(
                    (a.theta - b.theta).abs().max()))
                worst_mat = max(worst_mat, float(
                    (a.matrix() - b.matrix()).abs().max()))
        assert worst_theta == 0.0 and worst_mat == 0.0, \
            ("doc 61 SS7.2 item 5 FAILED at q = %d: the monolithic Circuit4(%d, 3) "
             "is not bit-identical to the committed verb block (theta %.3e, matrix "
             "%.3e). A family that does not reduce to the committed block is a NEW "
             "ANSATZ WEARING ITS NAME." % (q, 2 * q, worst_theta, worst_mat))
        out["q=%d" % q] = {
            "committed_block": ("exp42_models.QuantumStoryModel('A3') -- "
                                "Circuit4(2,3), 15 params" if q == 1 else
                                "exp45_model.TwoQubitReferentModel('A4') -- "
                                "Circuit4(4,3), 33 params"),
            "max_abs_theta_diff": worst_theta,
            "max_abs_matrix_diff": worst_mat,
            "bit_identical": True,
            "params_per_block": 3 * (3 * (2 * q) - 1),
        }
    rep["block_reduction"] = {
        "per_q": out,
        "ruling_1_claim": (
            "the MONOLITHIC `Circuit4(2q, 3)` reproduces the committed block at BOTH "
            "q = 1 AND q = 2, where the brickwork reproduces it only at q = 1. Both "
            "are asserted here on the SAME PARAMETER DRAW, including the readout "
            "pair, because the claim is free to check and is the whole warrant for "
            "ruling 1 over doc 61 SS6.1's registered brickwork."),
        "brickwork_status": ("NOTED, NOT RUN (doc 61 SS14 item 4 as amended by "
                            "ruling 1). It is not implemented in this file, so no "
                            "exp73 number can silently come from it."),
    }
    return rep


def check_copy(fm, verbs, store, rep):
    """doc 61 SS7.1, inherited from doc 55 SS7.4 / doc 56 SS7.4 and run for the
    AMPLITUDE arm as well as the anchor: with event 2 suppressed and q fixed to 1,
    the T = 2 forward pass must reproduce the T = 1 forward pass BIT-FOR-BIT. Only
    possible because referent state preparation happens EXACTLY ONCE, before event
    1 -- which for the amplitude arms is what keeps the encoding a property of the
    referent rather than of the event."""
    items2, _m, _p = arm_items("A3-T2-uniform", store)
    worst = {}
    for arm in ("A3-T2-uniform", "A4-T2-uniform", ANCHOR_ARM):
        a, _t1 = arm_angles(arm, fm)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
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
        assert w == 0.0, "doc 61 SS7.1 COPY-VERIFICATION FAILED for %s: %.3e" % (arm, w)
    rep["copy_verification"] = {
        "max_abs_diff": worst, "exact_zero": True,
        "note": ("with event (3 - q) suppressed and the query fixed to the surviving "
                 "event, every T = 2 arm reproduces its T = 1 pipeline bit-for-bit. "
                 "For the AMPLITUDE arms this carries the extra weight that the "
                 "state preparation sits in the INTRO and nowhere else -- doc 61 "
                 "SS14 item 5 declines re-uploading the amplitude vector at depth, "
                 "and this is where that decision is machine-checked."),
    }
    return rep


def check_angle_independence(fm, verbs, store, rep):
    """The amplitude arms are compiled with an identically-zero angle dict
    (`fm['zero_angles']`), and THE VALUE MUST NEVER BE READ. Perturbing it must
    leave every amplitude forward pass BIT-IDENTICAL -- and the same perturbation
    must MOVE the anchor, which is the failing direction that makes this a gate."""
    arm = "A3-T1-uniform"
    items, _m, _p = arm_items(arm, store)
    torch.manual_seed(0)
    m = build_model(arm, verbs, 0, fm)
    zero = fm["zero_angles"]
    perturbed = {n: 0.37 + 0.11 * k for k, n in enumerate(INS.HUMANS)}
    sel = [i for i in items if i["split"] == "test"][:6]
    d_amp = 0.0
    with torch.no_grad():
        for it in sel:
            a = m.story_values(R69.compile_item(it, zero))
            b = m.story_values(R69.compile_item(it, perturbed))
            d_amp = max(d_amp, float((a - b).abs().max()))
    assert d_amp == 0.0, \
        ("the amplitude forward pass DEPENDS on the compiled angle (%.3e) -- the "
         "zero angle dict is being read and the encoding is not what it claims" % d_amp)
    # the failing direction: the same perturbation must move the ANCHOR
    items_a, _m2, _p2 = arm_items(ANCHOR_ARM, store)
    torch.manual_seed(0)
    ma = build_model(ANCHOR_ARM, verbs, 0, fm)
    ang, _t1 = arm_angles(ANCHOR_ARM, fm)
    d_anc = 0.0
    with torch.no_grad():
        for it in [i for i in items_a if i["split"] == "test"][:6]:
            a = ma.story_values(R69.compile_item(it, ang))
            b = ma.story_values(R69.compile_item(it, perturbed))
            d_anc = max(d_anc, float((a - b).abs().max()))
    assert d_anc > 1e-4, \
        ("the ANCHOR is also insensitive to its compiled angles (%.3e) -- the "
         "angle-independence check is a null instrument" % d_anc)
    rep["angle_independence"] = {
        "amplitude_max_abs_diff_under_angle_perturbation": d_amp,
        "anchor_max_abs_diff_under_the_same_perturbation": d_anc,
        "note": fm["zero_angles_note"],
        "failing_direction": ("the same perturbation MOVES the anchor by %.3e, so "
                              "the zero result above is a property of the amplitude "
                              "encoding and not of a dead test" % d_anc),
    }
    return rep


def check_closure_not_applicable(fm, rep):
    """doc 61 SS7.4 -- THE FOURIER-CLOSURE INSTRUMENT IS NOT APPLICABLE, AND THAT IS
    REGISTERED RATHER THAN LEFT AS AN OMISSION.

    doc 56 SS7.1's closure gates test whether a compiled intro spans a claimed
    TRIGONOMETRIC basis. The amplitude intro has NO ANGLE PARAMETERISATION AT ALL --
    x is a point on a sphere, not a function of angles -- so closure in any
    trigonometric basis is UNDEFINED, not merely uninformative. Validity is asserted
    by SS7.2 items 1-4 instead, which are STRICTLY STRONGER: they pin the state
    itself rather than the span it lies in.

    What IS asserted here is the one place where the amplitude map and the committed
    angle map DO meet -- doc 61 SS2.2's exact linear relation at q = 1:

        L v(phi) = u(2 phi),   L = [[1,0,1],[1,0,-1],[0,2,0]],  det L = -4

    i.e. the 1-qubit amplitude feature is the committed single-angle feature AT
    DOUBLE FREQUENCY. It is verified by the committed instrument on a 721-point grid
    and is called here, not re-derived. It is the closest thing to a closure
    statement that exists at this rung, and it is exact."""
    lin = IAB.check_A1_linear_form()
    assert lin["verified"] and lin["max_abs_error_over_grid"] < 1e-12, \
        ("doc 61 SS2.2's linear-form identity does not hold (%.3e)"
         % lin["max_abs_error_over_grid"])
    diag = json.load(open(AMP_DIAGNOSTIC, encoding="utf-8"))
    sep = diag["selftest"]["checks"]["A1_separation"]
    rep["closure_doc61_SS7_4"] = {
        "applicable": False,
        "why_not": (
            "doc 61 SS7.4: doc 56 SS7.1's closure gates test whether a compiled intro "
            "spans a claimed TRIGONOMETRIC basis. The amplitude intro has NO ANGLE "
            "PARAMETERISATION AT ALL -- x is a point on a sphere, not a function of "
            "angles -- so closure in any trigonometric basis is UNDEFINED, not merely "
            "uninformative. A gate that cannot fail is not a gate (doc 49 SS8, "
            "orchestrator ruling 6 of doc 56). Validity for the amplitude arms is "
            "asserted by SS7.2 items 1-4 instead, which are STRICTLY STRONGER: they "
            "pin the STATE itself rather than the span it lies in."),
        "the_exact_relation_that_IS_asserted_doc61_SS2_2": lin,
        "reading": (
            "at q = 1 the amplitude feature v(phi) = (x0^2, x0 x1, x1^2) is carried "
            "to the committed single-angle feature u(2 phi) by a FIXED INVERTIBLE "
            "L -- the amplitude map is the single-angle map AT DOUBLE FREQUENCY, not "
            "at the same frequency. doc 61 SS2.2 records that `amplitude == u(phi)` "
            "FAILS on 28 of 30 random inventories, so the committed inventory's "
            "agreement is A COINCIDENCE AND EXPLICITLY NOT A FINDING; and A3/A4's 36- "
            "and 136-dimensional feature spaces have no angle analogue at all."),
        "separation_check_from_the_instrument": {
            k: sep[k] for k in sorted(sep) if not isinstance(sep[k], (list, dict))},
    }
    return rep


def check_committed_imports(rep):
    """doc 61 SS7.5 -- THE COMMITTED VALUES ARE IMPORTED, NOT RE-TYPED. F2's and
    P-eps's held-out means and their matched floors are read PROGRAMMATICALLY from
    results_exp69.json and results_exp70.json and asserted equal to doc 61 SS5.2's
    and SS15.2's printed tables."""
    src = {EXP69_RESULTS: json.load(open(EXP69_RESULTS, encoding="utf-8")),
           EXP70_RESULTS: json.load(open(EXP70_RESULTS, encoding="utf-8"))}
    arms, floors = {}, {}
    for key, spec in COMMITTED_DOC61_SS5_2.items():
        got = float(src[spec["source"]]["arms"][spec["arm"]]["mean"])
        assert abs(got - spec["mean"]) < 5e-5, \
            ("doc 61 SS5.2's %s = %.4f does not match %s :: %s = %.6f"
             % (key, spec["mean"], spec["source"], spec["arm"], got))
        arms[key] = {"source": spec["source"], "source_arm": spec["arm"],
                     "measured": got, "doc61_SS5_2": spec["mean"],
                     "sd": float(src[spec["source"]]["arms"][spec["arm"]]["sd"])}
    for key, spec in COMMITTED_FLOORS_DOC61_SS15_2.items():
        f = src[spec["source"]]["floors"][spec["arm"]]
        gm, gs = float(f["mean"]), float(f["sd"])
        assert abs(gm - spec["mean"]) < 5e-5 and abs(gs - spec["sd"]) < 5e-5, \
            ("doc 61 SS15.2's %s floor (%.4f +- %.4f) does not match %s :: %s "
             "(%.6f +- %.6f)" % (key, spec["mean"], spec["sd"], spec["source"],
                                 spec["arm"], gm, gs))
        floors[key] = {"source": spec["source"], "source_arm": spec["arm"],
                       "measured_mean": gm, "measured_sd": gs,
                       "doc61_SS15_2": [spec["mean"], spec["sd"]]}
    best = {1: max(arms["F2-T1"]["measured"], arms["P-eps-T1"]["measured"]),
            2: max(arms["F2-T2"]["measured"], arms["P-eps-T2"]["measured"])}
    rep["committed_imports_doc61_SS7_5"] = {
        "arms": arms, "floors": floors,
        "best_committed_2qubit_register": {"T1": best[1], "T2": best[2]},
        "helps_bar": {"T1": best[1] + 0.06, "T2": best[2] + 0.06},
        "hurts_bar": {"T1": best[1] - 0.06, "T2": best[2] - 0.06},
        "not_re_run_doc61_SS5_2": (
            "exp73 runs NO F2 arm, NO P-eps arm and NO classical comparator for "
            "them. The dataset is byte-identical, the features are the same "
            "deterministic constants, the splits and fit seed are the same, and the "
            "fits are deterministic. doc 47 SS14 is the standing reason: 'registered "
            "as a computation, not an arm -- training a classical model to "
            "rediscover a bound we derived would be theatre.'"),
        "amplitude_helps_T1_is_inert_doc61_SS10_3": (
            "its bar is %.4f and held-out accuracy is bounded by 1, so the T = 1 "
            "cell is ONE-SIDED BY CONSTRUCTION: only `amplitude_hurts_T1` can fire. "
            "The boolean is computed and stored for completeness and NO READING "
            "DEPENDS ON IT." % (best[1] + 0.06)),
    }
    return rep


def check_shuffle(store, rep):
    """doc 46 SS10.2 / doc 61 SS15.1: the floor severs features <-> label and changes
    NOTHING else. `exp69_run.relabel_shuffle_q` is IMPORTED, not re-implemented."""
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
    rep["shuffle"] = {
        "per_seed": rows,
        "mean_true_bearer_rule_accuracy":
            float(np.mean([r["true_bearer_rule_accuracy"] for r in rows])),
        "floored_arms": STAGE1_TRAINED,
        "note": ("doc 61 SS15.1: THE FLOOR IS PROCEDURE-MATCHED AND THE INIT IS PART "
                 "OF THE PROCEDURE -- a floor measured under one optimisation "
                 "procedure does not bound what another can achieve on severed "
                 "labels, so the two inits are two procedures and EACH CELL GETS ITS "
                 "OWN FLOOR. Bearer designation is reassigned PER PAIR and "
                 "independently of the features, so twins stay on opposite labels "
                 "and V1 stays exactly 0.500. THE DRIFT ANCHOR CARRIES NO FRESH "
                 "FLOOR (SS5.3.1): it is pinned to within 0.002 of a measurement "
                 "that cleared its own matched floor by 8.02 floor-SD units. IF THE "
                 "ANCHOR FAILS ITS GATE, THAT ARGUMENT IS VOID ALONG WITH THE RUN."),
    }
    return rep


def check_trace(fm, verbs, store, rep):
    """doc 61 SS9.3: the traced pass must be THE SAME forward pass -- and for the
    ANCHOR this doubles as the assertion that `apply_intro`'s re-implementation
    reproduces the committed parent's intro BIT-FOR-BIT."""
    worst, ranges = {}, {}
    for arm in ("A3-T1-uniform", "A3-T2-uniform", ANCHOR_ARM):
        items, _m, _p = arm_items(arm, store)
        ang, _t1 = arm_angles(arm, fm)
        data = R69.compile_all([i for i in items if i["split"] == "test"][:6], ang)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        w = 0.0
        lo, hi = 1.0, 0.0
        with torch.no_grad():
            for cs in data:
                v1 = m.story_values(cs)
                v2, tr = forward_traced(m, cs)
                w = max(w, float((v1 - v2).abs().max()))
                nw = qubits_per_referent(m) * cs.K
                assert set(tr) == ({"after_intro", "at_readout"}
                                   | {"after_event%d" % (k + 1)
                                      for k in range(cs.T)}), tr.keys()
                qpr = qubits_per_referent(m)
                for stg, sn in tr.items():
                    assert len(sn["wire"]) == nw, (stg, len(sn["wire"]), nw)
                    assert len(sn["register"]) == cs.K, (stg, len(sn["register"]))
                    for p in sn["wire"]:            # Tr rho^2 of ONE qubit >= 1/2
                        assert 0.4999 <= p <= 1.0001, ("wire", stg, p)
                        lo, hi = min(lo, p), max(hi, p)
                    for p in sn["register"]:        # >= 1/2^q for a q-qubit block
                        assert 0.5 ** qpr - 1e-4 <= p <= 1.0001, ("reg", stg, p)
                        lo, hi = min(lo, p), max(hi, p)
                if is_amplitude(m):
                    ri = tr["after_intro"]["register"]
                    assert max(abs(p - 1.0) for p in ri) < F32_FLOOR, \
                        ("doc 61 SS9.1: the per-REGISTER purity at intro is not 1 "
                         "(%r) -- the referents are not a product" % ri)
        worst[arm] = w
        ranges[arm] = [lo, hi]
        assert w == 0.0, \
            ("doc 61 SS9.3: the traced pass differs from story_values for %s "
             "(%.3e) -- for the ANCHOR this ALSO means apply_intro does not "
             "reproduce the committed parent intro" % (arm, w))
    rep["trace_parity"] = {
        "max_abs_diff_vs_story_values": worst, "exact_zero": True,
        "purity_range_per_arm": ranges,
        "stages": ["after_intro", "after_event1", "after_event2 (T=2)", "at_readout"],
        "quantities": ["per-wire Tr rho^2", "per-register Tr rho_reg^2"],
        "note": ("doc 61 SS9.3 -- TWO TRACES, NOT ONE. (1) the committed per-WIRE "
                 "trace, read against SS9.2's PRE-COMPUTED INVENTORY as its baseline "
                 "and NOT against 1: the amplitude registers are strongly and "
                 "DATA-DETERMINEDLY entangled within themselves (A3 median wire "
                 "purity 0.666, A4 0.530) and NO NOUN IS WITHIN 0.05 OF A PRODUCT "
                 "STATE at A3 or A4 -- the mechanism is the ENCODING, not a gate. "
                 "(2) the per-REGISTER trace, new here, which is 1.0 at intro by "
                 "construction and below 1 after the first verb block IFF the block "
                 "has actually correlated the two referents. DESCRIPTIVE; NO "
                 "CRITERION READS THEM, and doc 61 SS9.5 forbids promoting them onto "
                 "one after results exist (the exp44 no-promotion clause)."),
        "registered_question_doc61_SS9_4": (
            "does the readout restore the per-REGISTER purity to its INTRO value on "
            "correct q = 2 items and fail to on q = 1 items -- the q-asymmetry undo "
            "signature (doc 55 SS20.5), transposed from wires to registers? "
            "Descriptive, no boolean, no criterion."),
    }
    return rep


def check_failing_directions(fm, verbs, store, rep):
    """doc 48 SS10.6 / doc 61 SS7: EVERY ASSERTION HAS A FAILING DIRECTION, AND THE
    FAILING DIRECTIONS ARE THEMSELVES ASSERTED. A gate that cannot fail is not a
    gate (doc 49 SS8)."""
    out = {}
    rung, q, d = "A3", RUNGS["A3"]["q"], RUNGS["A3"]["d"]
    states = fm["amp"]["states"][rung]
    items, _m, _p = arm_items("A3-T1-uniform", store)
    ang, _a1 = arm_angles("A3-T1-uniform", fm)
    cs = R69.compile_all([i for i in items if i["split"] == "test"][:1], ang)[0]
    ref_name = cs.intro[0][1]

    # (1) A SIGN-FLIPPED AMPLITUDE MUST FAIL. The outer-square feature is sign-blind
    #     so the bound would never see it; the STATE assertion must.
    bad = dict(states)
    x = states[ref_name].copy()
    j = int(np.argmax(np.abs(x)))
    x[j] = -x[j]
    bad[ref_name] = x
    torch.manual_seed(0)
    mbad = AmplitudeModel(verbs, 0, bad, q)
    v64 = intro_state_f64(mbad, cs)
    good = states[cs.intro[0][1]]
    for _w, n, _a in cs.intro[1:]:
        good = np.kron(good, states[n])
    dev = float(np.abs(np.real(v64) - good).max())
    assert dev > 0.05, \
        ("the sign-flip failing direction did NOT fire (%.3e) -- SS7.2 item 1 is a "
         "null instrument" % dev)
    # ... and the outer-square feature is IDENTICAL under that flip only if the
    # whole vector flips; a single-coordinate flip changes v(x), so the sharper
    # statement is the GLOBAL one, asserted here:
    v_plus = IAB.amp_features(states[ref_name])
    v_minus = IAB.amp_features(-states[ref_name])
    blind = float(np.abs(v_plus - v_minus).max())
    assert blind == 0.0, \
        "the outer-square feature is NOT sign-blind -- doc 61 SS1.2's sharp limit fails"
    out["sign_flip_fires"] = {"max_abs_state_deviation": dev,
                              "outer_square_global_sign_blindness": blind,
                              "reading": ("x -> -x leaves v(x) EXACTLY invariant "
                                          "(doc 61 SS1.2: the readout cannot "
                                          "distinguish a referent from its "
                                          "antipode), so a sign error is INVISIBLE "
                                          "to the bound. The state assertion sees "
                                          "it, and here it is shown to fire.")}

    # (2) AN UNNORMALISED VECTOR MUST FAIL at construction.
    bad2 = dict(states)
    bad2[ref_name] = states[ref_name] * 1.5
    fired = False
    try:
        AmplitudeModel(verbs, 0, bad2, q)
    except AssertionError as exc:
        fired = True
        out["unnormalised_rejected_message"] = str(exc)[:160]
    assert fired, "an unnormalised amplitude vector was ACCEPTED"
    out["unnormalised_rejected"] = True

    # (3) A WIRE-ORDER SLIP MUST FAIL. Reversing the bit order of x is a different
    #     state on the same register, and the amplitude-by-amplitude gate must see it.
    perm = [int("{:0{}b}".format(i, q)[::-1], 2) for i in range(d)]
    bad3 = dict(states)
    bad3[ref_name] = states[ref_name][perm]
    torch.manual_seed(0)
    m3 = AmplitudeModel(verbs, 0, bad3, q)
    dev3 = float(np.abs(np.real(intro_state_f64(m3, cs)) - good).max())
    assert dev3 > 0.01, \
        ("the wire-order failing direction did NOT fire (%.3e)" % dev3)
    out["wire_order_slip_fires"] = {"max_abs_state_deviation": dev3}

    # (4) A PERTURBED BLOCK MUST BREAK THE q = 1 REDUCTION.
    from exp42_models import QuantumStoryModel
    torch.manual_seed(0)
    committed = QuantumStoryModel("A3", verbs, 0)
    torch.manual_seed(0)
    mono = AmplitudeModel(verbs, 0, {n: np.eye(2)[0] for n in INS.HUMANS}, 1)
    with torch.no_grad():
        mono.blocks[verbs[0]].theta[0, 0] += 0.05
        dev4 = float((committed.verb_matrix(verbs[0])
                      - mono.verb_matrix(verbs[0])).abs().max())
    assert dev4 > 1e-3, \
        "a perturbed block still matched the committed one -- SS7.2 item 5 is inert"
    out["block_reduction_check_fires"] = {"max_abs_matrix_diff_after_perturbation":
                                          dev4}

    # (5) THE WRONG ANSATZ DEPTH MUST BREAK THE PARAMETER COUNT.
    torch.manual_seed(0)
    m5 = AmplitudeModel(verbs, 0, states, q, layers=2)
    n5 = sum(p.numel() for p in m5.parameters())
    assert n5 != RUNGS["A3"]["n_params"], \
        "a 2-layer ansatz reproduced the registered 408-parameter count"
    out["param_count_check_fires"] = {"layers_2_param_count": n5,
                                      "registered_3_layer_count":
                                          RUNGS["A3"]["n_params"]}

    # (6) doc 58 SS6.2.2 / doc 61 SS5.1.2's NULL-INSTRUMENT CHECK: a readout
    #     initialised AT IDENTITY must be REJECTED. This is the scenario the
    #     exclusion is registered against, verbatim -- "if BOTH readout blocks were
    #     initialised at identity they would be IDENTICAL, v_yes == v_no for every
    #     item, and the model would output exactly 0.5 everywhere". The committed
    #     assertion is CALLED, not re-implemented.
    torch.manual_seed(0)
    m6 = AmplitudeModel(verbs, 0, states, q)
    R72.apply_identity_init(m6, 0, verbs, [])
    with torch.no_grad():
        m6.q_pos.theta.zero_()
        m6.q_neg.theta.zero_()
    rejected = False
    try:
        R72.identity_assertions(m6, verbs, [], 0)
    except AssertionError as exc:
        rejected = True
        out["identity_readout_message"] = str(exc)[:200]
    assert rejected, \
        ("doc 58 SS6.2.2's null-instrument check ACCEPTED an identity-initialised "
         "readout -- two identical readout blocks give v_yes == v_no everywhere")
    # and the degeneracy it names is MEASURED, not merely asserted against
    torch.manual_seed(0)
    m6b = AmplitudeModel(verbs, 0, states, q)
    with torch.no_grad():
        m6b.q_pos.theta.zero_()
        m6b.q_neg.theta.zero_()
        vals = m6b.story_values(cs)
    out["identity_readout_rejected"] = True
    out["identity_readout_is_degenerate"] = {
        "v_yes": float(vals[0]), "v_no": float(vals[1]),
        "abs_v_yes_minus_v_no": float((vals[0] - vals[1]).abs()),
        "softmax_p_yes": float(torch.softmax(torch.log(vals + 1e-12), 0)[0]),
        "reading": ("doc 61 SS5.1.2, MEASURED at 6 wires rather than argued: with "
                    "both readout blocks at identity the two effects are THE SAME "
                    "OPERATOR, so v_yes == v_no EXACTLY for every item and the "
                    "softmax over log v is exactly 0.5 -- a NULL INSTRUMENT at "
                    "initialisation in doc 53 SS5's exact sense. This is why sigma "
                    "is applied to the VERB blocks only.")}
    # THE WIDTH NOTE, recorded because it is a real difference from the 2-wire case:
    # a sigma = 0.1 draw on a 51-parameter 6-wire readout is NOT near-identity, so
    # exp72's `max |q - I| > 0.1` branch would not fire on it here. The registered
    # scenario is the identity one above, and that is what is tested.
    torch.manual_seed(0)
    m6c = AmplitudeModel(verbs, 0, states, q)
    with torch.no_grad():
        gq = torch.Generator().manual_seed(IDENTITY_RNG_BASE + 999)
        for blk in (m6c.q_pos, m6c.q_neg):
            blk.theta.copy_(torch.randn(blk.theta.shape, generator=gq)
                            * IDENTITY_SIGMA)
        eye = torch.eye(2 ** (2 * q), dtype=m6c.q_pos.matrix().dtype)
        out["sigma_0.1_readout_distance_from_identity_at_%dwires" % (2 * q)] = float(
            (m6c.q_pos.matrix() - eye).abs().max())

    # (7) THE WRONG SIGMA must be rejected by the sd window.
    torch.manual_seed(0)
    m7 = AmplitudeModel(verbs, 0, states, q)
    g = torch.Generator().manual_seed(IDENTITY_RNG_BASE)
    with torch.no_grad():
        for v in sorted(verbs):
            t = m7.blocks[v].theta
            t.copy_(torch.randn(t.shape, generator=g) * 0.5)
    rejected7 = False
    try:
        R72.identity_assertions(m7, verbs, [], 0)
    except AssertionError:
        rejected7 = True
    assert rejected7, "the identity-init sd window ACCEPTED sigma = 0.5"
    out["wrong_sigma_rejected"] = True

    rep["failing_directions"] = {
        "checks": out,
        "rule": ("doc 48 SS10.6 / doc 49 SS8: an arm that does not implement what it "
                 "is named for is a null instrument, and a gate that cannot fail is "
                 "not a gate. Every SS7.2 assertion above is shown to FIRE on a "
                 "deliberately broken input before any training is launched."),
    }
    return rep


def check_identity_init(fm, verbs, rep):
    """doc 61 SS7.2 item 7 -- the identity-init arms, BOTH DIRECTIONS, through the
    COMMITTED exp72 code objects."""
    out = {}
    for rung, rc in RUNGS.items():
        torch.manual_seed(0)
        m = build_model("%s-T1-identity" % rung, verbs, 0, fm)
        info = apply_init_scheme(m, "%s-T1-identity" % rung, 0)
        out[rung] = {k: v for k, v in info.items() if k != "blocks_reinitialised"}
        out[rung]["n_blocks_reinitialised"] = len(info["blocks_reinitialised"])
    rep["identity_init"] = {
        "per_rung": out,
        "both_inits_are_arms_doc61_SS5_1_1": (
            "doc 59 SS2's ruling, and the reason is recorded rather than inherited "
            "silently: doc 58 SS19.1 measured rescue_partial(I1, R-identity) = TRUE "
            "(0.7293 -> 0.8860 on the product register at T = 2) and doc 58 SS19.6 "
            "registered the consequence that 'the ledger's trainability walls become "
            "partly an initialisation-policy artifact'. UNTIL THAT CONFIRM DRAW "
            "RETURNS, NEITHER INIT IS THE PROGRAMME'S COMMITTED POLICY, and "
            "registering only one would be choosing a policy on an unconfirmed "
            "single draw. doc 61 SS11.3 registers NO DIRECTION on the init arms for "
            "the same reason."),
    }
    return rep


def smoke(fm, verbs, store, tiny_epochs=2):
    t0 = time.time()
    rep = {"experiment": "exp73 smoke / validity assertions (doc 61 SS7, SS9.3)",
           "spec": SPEC,
           "orchestrator_rulings": {
               "1_verb_block": ("MONOLITHIC Circuit4(2q, 3); 408 (A3) / 552 (A4) "
                                "params; the brickwork is NOTED, NOT RUN"),
               "2_compute": "A4 is a CONDITIONAL SECOND STAGE; stage 1 is A3-only",
               "3_batch_hoist": "DECLINED -- no shared committed code is touched",
               "4_directional_call": "amplitude_hurts_T1 = TRUE (bar <= 0.9220)",
               "5_tier_iii_grid": "C in {1e-4,1e-3,1e-2,1e-1,1,1e1,1e2}, FROZEN"},
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "numpy": np.__version__}
    check_dataset(store, rep)
    check_instrument_provenance(rep)
    check_features(fm, rep)
    check_committed_imports(rep)
    check_block_reduction(verbs, rep)
    check_zero_parameter_intro(fm, verbs, store, rep)
    check_intro_states(fm, verbs, store, rep)
    check_closure_not_applicable(fm, rep)
    check_angle_independence(fm, verbs, store, rep)
    ang, _t1 = arm_angles(ANCHOR_ARM, fm)
    items2, _m, _p = arm_items(ANCHOR_ARM, store)
    data2 = {"test": R69.compile_all(
        [i for i in items2 if i["split"] == "test"], ang)}
    R69.check_word_order(fm, verbs, data2, rep)       # doc 61 SS7.1, inherited
    check_copy(fm, verbs, store, rep)
    check_trace(fm, verbs, store, rep)
    check_identity_init(fm, verbs, rep)
    check_shuffle(store, rep)
    check_failing_directions(fm, verbs, store, rep)

    tiny = {}
    for arm in STAGE1_ARMS:
        r = train(arm, 0, store, verbs, tiny_epochs, fm, save=False)
        tiny[arm] = {"val_acc": r["val_acc"],
                     "heldout_test_acc": r["heldout_test_acc"],
                     "wall_sec": r["wall_sec"], "n_params": r["n_params"],
                     "wires": r["wires"], "T": r["T"], "init": r["init"],
                     "sec_per_epoch": r["wall_sec"] / float(tiny_epochs)}
        print("[smoke] %-28s %dep val=%.3f test=%.3f  %.0fs  (%d params, %d wires, "
              "T=%d, %s)" % (arm, tiny_epochs, r["val_acc"], r["heldout_test_acc"],
                             r["wall_sec"], r["n_params"], r["wires"], r["T"],
                             r["init"]), flush=True)
    rep["probe"] = {"epochs": tiny_epochs, "per_arm": tiny}

    # the extrapolated stage-1 wall time, MEASURED on this machine (doc 61 SS16.2's
    # discipline: computed and measured, not estimated from qubit counts).
    est = {}
    serial = 0.0
    for arm in STAGE1_ARMS:
        per_epoch = tiny[arm]["sec_per_epoch"]
        # the held-out + trace evaluation runs once; the probe already includes one.
        secs = per_epoch * MAX_EPOCHS
        est[arm] = {"projected_sec_per_run": secs, "runs": N_SEEDS,
                    "projected_sec_total": secs * N_SEEDS}
        serial += secs * N_SEEDS
    rep["projected_wall_time"] = {
        "per_arm": est,
        "stage1_runs": len(STAGE1_ARMS) * N_SEEDS,
        "stage1_serial_sec": serial, "stage1_serial_hours": serial / 3600.0,
        "stage1_hours_on_5_workers": serial / 3600.0 / 5.0,
        "basis": ("linear extrapolation of the %d-epoch probe to MAX_EPOCHS = %d, "
                  "measured on this machine at the registered depths. doc 61 SS16.2 "
                  "priced the BRICKWORK; ruling 1's monolithic block is ~2.5x "
                  "cheaper at A3, and this is the measured number rather than the "
                  "doc's." % (tiny_epochs, MAX_EPOCHS)),
    }

    npar = {a: tiny[a]["n_params"] for a in STAGE1_ARMS}
    assert npar["A3-T1-uniform"] == npar["A3-T2-uniform"] \
        == npar["A3-T1-identity"] == npar["A3-T2-identity"] \
        == RUNGS["A3"]["n_params"], \
        ("the A3 cells do not all carry the registered 408 parameters: %r" % npar)
    assert npar[ANCHOR_ARM] == 264, \
        ("the drift anchor does not carry the committed register's 264 parameters "
         "(%d) -- it is not the committed model" % npar[ANCHOR_ARM])
    rep["parameter_budget_identity"] = {
        "per_arm": npar,
        "note": ("doc 61 SS6.4 under ruling 1: all four A3 cells carry EXACTLY 408 "
                 "trainable parameters (8 blocks x 51), so T and init are the only "
                 "things that vary across them. The drift anchor carries the "
                 "COMMITTED 264 because it IS the committed model."),
    }
    rep["assertions_passed"] = True
    rep["wall_time_sec"] = time.time() - t0
    json.dump(rep, open(SMOKE_OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[smoke] wrote %s -- ALL doc 61 SS7 / SS9.3 ASSERTIONS PASSED" % SMOKE_OUT)
    return rep


# ==================================================================== training
def train(arm, seed, store, verbs, max_epochs, fm, save=True, tag=""):
    """The exp60 / exp61 / exp65 / exp68 / exp69 / exp70 / exp72 loop, UNCHANGED --
    Adam(lr 0.02), batch 8, 150 epochs, checkpoint selected on in-distribution val
    accuracy, held-out test read ONCE from the selected checkpoint.

    THE CALL ORDER (compile -> manual_seed -> build -> init scheme -> Adam ->
    Random(RNG_BASE + seed)) IS exp70's VERBATIM, because doc 61 SS5.3 expects the
    DRIFT ANCHOR to reproduce exp69's R2-W BIT-EXACTLY and any reordering of RNG
    consumption would break that. The identity init draws from a DEDICATED stream
    (doc 58 SS6.2) and is applied AFTER `build_model`, so not one committed draw
    moves and the uniform arms are bit-identical to what `build_model` alone
    produces.

    doc 61 SS7.2 item 8: THE ANCHOR RUNS THROUGH THIS ENTRY POINT, not a legacy
    path -- a separate path would make SS5.3's bit-exactness prove nothing about
    the amplitude arms' pipeline."""
    cfg = ARM_CFG[base_arm(arm)]
    items, meta, path = arm_items(arm, store)
    data = compile_arm(arm, items, seed, fm)
    torch.manual_seed(seed)
    m = build_model(arm, verbs, seed, fm)
    init_info = apply_init_scheme(m, arm, seed)
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
        if ep % 25 == 0 or ep == max_epochs:
            print("[%s s%d] ep %3d/%d val=%.3f best=%.3f %.0fs"
                  % (arm, seed, ep, max_epochs, va, best["val"],
                     time.time() - t0), flush=True)
    m.load_state_dict(best["state"])
    te, rows = evaluate_with_traces(m, data["test"])
    pt = sha = None
    if save:
        os.makedirs(CKPT_DIR, exist_ok=True)
        pt = os.path.join(CKPT_DIR, "%s%s_s%d.pt" % (tag, arm, seed))
        torch.save(best["state"], pt)                  # doc 45 SS5
        sha = sha256_file(pt)
    return {"arm": arm, "base_arm": base_arm(arm), "family": cfg["family"],
            "is_floor": arm.startswith("S-shuffle"),
            "seed": seed, "part": cfg["part"], "T": cfg["T"],
            "encoding": cfg["enc"], "rung": cfg["rung"], "init": cfg["init"],
            "d_truncated": cfg["d"], "role": cfg["role"],
            "init_scheme": init_info,
            "train_acc": best["train"], "val_acc": best["val"],
            "selected_epoch": best["epoch"], "heldout_test_acc": te,
            "val_minus_heldout": best["val"] - te,
            "gate_met": bool(best["train"] >= TRAIN_GATE and best["val"] >= VAL_GATE),
            "n_params": sum(p.numel() for p in m.parameters()),
            "wires": cfg["qpr"] * 2,
            "qubits_per_referent": cfg["qpr"],
            "n_train": len(data["train"]), "n_val": len(data["val"]),
            "n_test": len(data["test"]),
            "max_epochs": max_epochs, "lr": LR, "batch": BATCH,
            "dataset": path, "dataset_sha256": meta["dataset_sha256"],
            "ckpt_path": pt, "ckpt_sha256": sha,
            "heldout_items": rows,
            "spec": SPEC,
            "wall_sec": time.time() - t0}


# ============================================= doc 61 SS5.3 -- THE DRIFT ANCHOR
def anchor_report(store):
    """doc 61 SS5.3 / SS10.1, BLOCKING and read FIRST.

    `A0-W-T2`'s PER-SEED held-out accuracies must reproduce results_exp69.json ::
    R2-W's per-seed values to |delta| <= 0.002, and the arm mean to |delta| <= 0.002
    of 0.7293. Both runs are deterministic given their seeds, so THE HONEST
    EXPECTATION IS |delta| = 0 EXACTLY, BIT-EXACT, 5/5; the tolerance absorbs a
    documented float/BLAS platform difference and any non-zero delta is reported
    with its cause identified. FOURTH ENFORCED USE of this gate; the first three
    returned bit-exact (doc 55 SS20.1, doc 56 SS18, doc 58 SS19).

    IF THIS FAILS, the exp73 pipeline has drifted, every amplitude number is a
    comparison against a moving baseline, and NOTHING ELSE IN THE RUN IS READ.
    Rule 3: this writes its OWN results file and no exp69/70/72 artefact is touched.
    """
    _items, meta, path = store
    ref = json.load(open(EXP69_RESULTS, encoding="utf-8"))["arms"][ANCHOR_SOURCE_ARM]
    rows = []
    for s in range(N_SEEDS):
        ckj = os.path.join(CKPT_DIR, "%s_s%d.json" % (ANCHOR_ARM, s))
        if not os.path.exists(ckj):
            raise RuntimeError(
                "the drift anchor is incomplete: %s is missing. doc 61 SS17.3 item 8 "
                "-- the anchor runs FIRST and blocks everything." % ckj)
        rows.append(json.load(open(ckj, encoding="utf-8")))
    rows.sort(key=lambda r: r["seed"])
    got = [r["heldout_test_acc"] for r in rows]
    want = ref["per_seed"]
    assert [r["seed"] for r in rows] == ref["seeds"], \
        "seed sets differ between exp73 %s and exp69 %s" % (ANCHOR_ARM,
                                                            ANCHOR_SOURCE_ARM)
    # doc 61 SS5.3's printed per-seed table is asserted against the FILE, not used
    # in its place (SS7.5's discipline applied to the anchor).
    assert all(abs(w - p) < 5e-5 for w, p in zip(want, ANCHOR_PER_SEED_DOC61)), \
        ("doc 61 SS5.3's printed per-seed values do not match %s :: %s"
         % (EXP69_RESULTS, ANCHOR_SOURCE_ARM))
    d = [g - w for g, w in zip(got, want)]
    mean = float(np.mean(got))
    ok_seed = all(abs(x) <= ANCHOR_TOLERANCE for x in d)
    ok_mean = abs(mean - ANCHOR_MEAN_DOC61) <= ANCHOR_TOLERANCE
    exact = all(x == 0.0 for x in d)
    return {
        "component": "exp73 drift anchor -- doc 61 SS5.3, the BLOCKING gate",
        "spec": SPEC,
        "rule3_note": ("the anchor writes its OWN results file (doc 61 SS16.5, "
                       "'separate by Rule 3'). No exp68/69/70/72 artefact is "
                       "touched, read for writing, or overwritten."),
        "dataset": path, "dataset_sha256": meta["dataset_sha256"],
        "exp69_source": EXP69_RESULTS, "exp69_arm": ANCHOR_SOURCE_ARM,
        "arm": ANCHOR_ARM,
        "exp69_per_seed": want, "exp69_mean": ref["mean"], "exp69_sd": ref["sd"],
        "exp73_per_seed": got, "exp73_mean": mean,
        "exp73_sd": float(np.std(got, ddof=1)),
        "per_seed_delta": d,
        "max_abs_per_seed_delta": float(max(abs(x) for x in d)),
        "mean_delta_vs_exp69": mean - ref["mean"],
        "registered_mean_doc61_SS5_3": ANCHOR_MEAN_DOC61,
        "mean_delta_vs_registered": mean - ANCHOR_MEAN_DOC61,
        "tolerance": ANCHOR_TOLERANCE,
        "bit_exact": bool(exact),
        "per_seed_within_tolerance": bool(ok_seed),
        "mean_within_tolerance": bool(ok_mean),
        "anchor_replicates_committed": bool(ok_seed and ok_mean),
        "val_per_seed": [r["val_acc"] for r in rows],
        "selected_epoch_per_seed": [r["selected_epoch"] for r in rows],
        "n_params": rows[0]["n_params"], "wires": rows[0]["wires"],
        "wall_sec_total": float(sum(r["wall_sec"] for r in rows)),
        "same_entry_point_doc61_SS7_2_8": (
            "the anchor is trained by exp73's OWN `train()` with the amplitude intro "
            "and the amplitude block OFF -- `build_model` returns the COMMITTED "
            "`exp68_run.F2Model(verbs, seed, fm['theta_pc2'])` verbatim. A separate "
            "legacy path would make this bit-exactness prove nothing about the "
            "amplitude arms' pipeline."),
        "honest_expectation": (
            "BIT-EXACT, 5/5. Both runs are deterministic given their seeds and exp73 "
            "constructs the COMMITTED parent class verbatim with exp69's exact call "
            "order; the 0.002 tolerance exists only to absorb a documented "
            "float/BLAS platform difference and any non-zero delta is reported with "
            "its cause identified."),
        "consequence_if_failed": (
            "doc 61 SS5.3: THE EXP73 PIPELINE HAS DRIFTED, every amplitude number is "
            "a comparison against a moving baseline, and NOTHING ELSE IN THE RUN IS "
            "READ. SS5.3.1's floor-omission argument for the anchor is VOID along "
            "with the run. Hunt the build; interpret nothing."),
    }


# ======================================================================== main
def job_list(arms, seeds):
    """doc 61 SS17.3 item 8 -- THE DRIFT ANCHOR RUNS FIRST. The anchor's five seeds
    are the first five jobs, so with `--nworkers 5` every worker's job 0 is an
    anchor seed and the blocking gate completes before any amplitude arm finishes.
    The read-time gate lives in `--anchor-report` and in `exp73_aggregate.py`; the
    launch order is what makes it cheap."""
    anchor = [(ANCHOR_ARM, s) for s in range(seeds) if ANCHOR_ARM in arms]
    rest = [(a, s) for a in arms if a != ANCHOR_ARM for s in range(seeds)]
    return anchor + rest


def main():
    global DATA_A, CKPT_DIR, SMOKE_OUT, VSUITE, ANCHOR_OUT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(STAGE1_ARMS),
                    help="default is STAGE 1 (A3 only; A4 is orchestrator ruling 2's "
                         "CONDITIONAL SECOND STAGE and is not launched by default)")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-epochs", type=int, default=2)
    ap.add_argument("--anchor-report", action="store_true",
                    help="read the five anchor checkpoints and write %s" % ANCHOR_OUT)
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
            "doc 61 SS17.1: exp73 runs on exp69's FROZEN Part A and there is no "
            "exp73 datagen. A different dataset is doc 61 SS17.2's CONFIRM DRAW, "
            "which is not registered as part of stage 1.")

    store = load_part()
    verbs = sorted(store[1]["verb_inventory"].keys())
    assert len(verbs) == 6, \
        ("doc 61 SS6.4's parameter arithmetic assumes 6 verbs (6 verb blocks + "
         "q_pos + q_neg = 8 blocks); the dataset carries %d" % len(verbs))
    fm = feature_maps()
    os.makedirs(CKPT_DIR, exist_ok=True)

    if a.smoke:
        smoke(fm, verbs, store, tiny_epochs=a.smoke_epochs)
        return 0

    if a.anchor_report:
        rep = anchor_report(store)
        json.dump(rep, open(ANCHOR_OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n[ANCHOR GATE] doc 61 SS5.3, |delta| <= %.3f, honest expectation "
              "BIT-EXACT" % ANCHOR_TOLERANCE)
        print("  exp69 %s  %s  mean %.6f"
              % (ANCHOR_SOURCE_ARM, [round(x, 6) for x in rep["exp69_per_seed"]],
                 rep["exp69_mean"]))
        print("  exp73 %s  %s  mean %.6f"
              % (ANCHOR_ARM, [round(x, 6) for x in rep["exp73_per_seed"]],
                 rep["exp73_mean"]))
        print("  delta       %s  max |d| %.2e  bit-exact=%s"
              % ([round(x, 6) for x in rep["per_seed_delta"]],
                 rep["max_abs_per_seed_delta"], rep["bit_exact"]))
        print("  anchor_replicates_committed = %s  ->  %s"
              % (rep["anchor_replicates_committed"],
                 "PROCEED" if rep["anchor_replicates_committed"]
                 else "THE RUN STOPS (doc 61 SS5.3)"))
        print("[ANCHOR GATE] wrote %s" % ANCHOR_OUT)
        return 0 if rep["anchor_replicates_committed"] else 1

    if not os.path.exists(SMOKE_OUT):
        raise RuntimeError(
            "%s missing -- doc 61 SS17.3 item 7: every SS7 smoke assertion, including "
            "the ones designed to fail, runs BEFORE any training. Run "
            "`python exp73_run.py --smoke`." % SMOKE_OUT)
    sm = json.load(open(SMOKE_OUT, encoding="utf-8"))
    if not sm.get("assertions_passed"):
        raise RuntimeError("the exp73 smoke gate did not pass -- nothing may train")

    arms = [x for x in a.arms.split(",") if x]
    unknown = [x for x in arms if base_arm(x) not in ARM_CFG]
    assert not unknown, "unknown arms: %r" % unknown
    a4 = [x for x in arms if ARM_CFG[base_arm(x)]["stage"] == 2]
    if a4:
        print("[exp73] WARNING: A4 cells requested (%r). Orchestrator ruling 2 makes "
              "A4 a CONDITIONAL SECOND STAGE -- it runs only if A3 binds at T = 1 or "
              "shows an init effect >= 0.06. Proceeding because it was asked for "
              "explicitly." % a4, flush=True)

    jobs = job_list(arms, a.seeds)
    mine = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    print("[exp73 w%d/%d] arms=%d -> %d of %d runs (anchor first)"
          % (a.worker, a.nworkers, len(arms), len(mine), len(jobs)), flush=True)
    for arm, s in mine:
        ckj = os.path.join(CKPT_DIR, "%s%s_s%d.json" % (a.tag, arm, s))
        if os.path.exists(ckj):
            print("[ckpt] %s%s_s%d already done" % (a.tag, arm, s), flush=True)
            continue
        r = train(arm, s, store, verbs, MAX_EPOCHS, fm, save=True, tag=a.tag)
        r["tag"] = a.tag
        json.dump(r, open(ckj, "w", encoding="utf-8"))
        print("[exp73] %-28s s%d  train=%.3f val=%.3f HELD-OUT=%.4f  (%.0fs)"
              % (arm, s, r["train_acc"], r["val_acc"], r["heldout_test_acc"],
                 r["wall_sec"]), flush=True)
    print("[exp73 w%d] done" % a.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
