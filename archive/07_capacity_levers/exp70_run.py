# -*- coding: utf-8 -*-
"""exp70_run.py -- doc 56: entangled encodings, the ledger's FOURTH LEVER.

THE QUESTION (doc 56 SS0). The first three levers -- higher frequency within a wire
(K), more readout outcomes within a wire (M), and parallel wires (width) -- all moved
supply INSIDE the product-state family. exp70 is the first that leaves it: the
referent's state is no longer a product of single-qubit RYs, and the readable
function space changes as a result.

**THIS IS NOT AN ADVANTAGE EXPERIMENT, AND THAT IS PROVABLE, NOT MODEST** (doc 56
SS0.1). doc 52 SS1 closed the advantage question for the product-state family with an
argument, not a measurement, and doc 52 SS1.3 named the entangled regime and ruled it
out of reach at this scale. exp70 steps into that family and stays FIRMLY on the
simulable side, deliberately: four qubits, every state in closed form, every
classical comparator computed from those closed forms. NO EXP70 NUMBER MAY APPEAR IN
A SENTENCE CONTAINING "ADVANTAGE".

NO DATAGEN. THIS IS A CLOSED GATE, NOT AN OMISSION (doc 56 SS5.1 / SS16.1)
=========================================================================
Every arm runs on exp69's FROZEN Part A dataset `stories_exp69_QA.json`, sha-pinned
and asserted at smoke, with the T = 1 and T = 2 item forms exp69 already generated
and V-suite-cleared. exp70 introduces NO new Arabic surface, NO new noun and NO new
verb, so doc 50 SS5.2's native-speaker gate does NOT reopen (doc 56 SS17.1).
**`exp70_datagen.py` does not exist and may not be written.** The confirm draw at
generator seed + 1000 (doc 56 SS5.1.1) re-runs the COMMITTED `exp69_datagen.py`
unmodified and fires only on a positive (SS17.3 item 8).

ARMS (doc 56 SS5.2), 5 families x 2 T x 5 seeds + 3 floor families x 2 T x 5 seeds

  A0-W-T{1,2}      PC1/PC2 on the OLD map (p+1)pi, 2 qubits/referent, product
                   register.  **BIT-EXACT ANCHOR** for exp69 R1-W / R2-W.
  EA-ctrl-T{1,2}   frozen angle atan2(PC2,PC1), 1 qubit/referent, single RY, NO CZ.
                   **DUAL ROLE** (doc 56 SS5.2, orchestrator ruling 2): bit-exact
                   anchor for exp69 R1-A / R2-A **and** the map-matched contrast
                   baseline for N-across. The tighter |delta| <= 0.002 gate governs.
  P-eps-T{1,2}     PC1/PC2 on the EPS-MAP, product register, NO entangler.
                   **THE MAP-MATCHED PRODUCT TWIN** (doc 56 SS4.3).
  N-in-T{1,2}      PC1/PC2 on the eps-map, RY(theta1) on q0 then CRY(theta2)
                   q0 -> q1.  The intra-register entangled arm.
  N-across-T{1,2}  frozen angle, one qubit per referent, RY (x) RY then ONE CZ
                   between the two referent wires after intro.  A MECHANISM PROBE
                   (doc 56 SS4.4) -- see below.
  S-shuffle-<arm>  matched floors for the THREE NEW families only (30 runs).

  -> 16 arms x 5 seeds = 80 runs. The anchor families carry NO fresh floor, and the
     argument is stated rather than assumed (doc 56 SS5.3): a gate pinning an arm to
     within 0.002 of a measurement that cleared its own matched floor by 8.02-19.08
     SD is transitively stronger than re-measuring the floor. IF EITHER ANCHOR FAILS
     ITS GATE, THAT ARGUMENT IS VOID ALONG WITH THE RUN.

THE EXPERIMENT IS N-in AGAINST P-eps, AND NOTHING ELSE (doc 56 SS4.3). Both arms
carry the same two features under the same angle map on the same wires with the same
parameter budget; the ONLY difference is the CRY. Without P-eps the entanglement
contrast would be confounded with the map change, which is precisely the confound
doc 56 SS2's query-3 literature search found the literature carrying.

E-ACROSS HAS NO DISCOCIRC LICENSE, AND THAT IS REGISTERED, NOT DISCOVERED (SS4.4).
In DisCoCirc, referents are introduced as INDEPENDENT states and correlation between
them is created by EVENTS. Pre-correlating two referents before any event asserts a
relationship the text has not stated. N-across is therefore a MECHANISM PROBE, not a
linguistic hypothesis: the cleanest available way to ask whether the SB readout is
sensitive to cross-wire correlation AT ALL, with the linguistic question deliberately
switched off. **No exp70 outcome on this arm may be reported as a claim about
DisCoCirc, about composition, or about language.**

THE EPS-MAP IS FORCED, NOT CHOSEN (doc 56 SS3). Under exp68's registered map
theta = (p+1)pi the normalisation extremes |p| = 1 land exactly on the concurrence
dead zones sin t1 = 0 and sin(t2/2) = 0: `diagnostic_encoding_entanglement.json`
found FOUR of the 24 nouns below concurrence 0.05, two of them exactly at zero, one
of which sits in the target category. An arm that does not use the structure it
claims to use is a NULL INSTRUMENT (doc 53 SS5). The eps-map
theta = p(pi - 2eps)/2 + pi/2, eps = pi/6, puts the inventory minimum at 0.1898 with
nothing below 0.05.

VALIDITY, BOTH DIRECTIONS (doc 56 SS7, doc 48 SS10.6's discipline)
  SS7.1  Fourier closure per arm, re-derived for the new maps. **The N-in gate is a
         DOUBLE NEGATIVE and it is the one that matters**: N-in must close in the
         6-dim E-in basis AND must NOT close in the 3-term basis of theta1 alone (a
         CRY with an unwired target degenerates there) AND must NOT close in the
         9-dim product basis (an uncontrolled RY would close there). Both failures
         are asserted against explicitly.
  SS7.2  **N-across gets NO Fourier gate, and that is a PROVEN null instrument, not
         an omission.** doc 56 SS4.5: CZ is Clifford, so conjugating the readout by
         it PERMUTES the Pauli coordinates and leaves the reachable function space
         invariant -- closure holds with the CZ present and absent alike. doc 49
         SS8's newly adopted checklist entry (orchestrator ruling 6): **a gate that
         cannot fail is not a gate.** Replaced by three blocking assertions: the
         per-item COMPUTED intro purity against the closed form
         1 - (sin tA sin tB)^2/2 at 1e-6; a circuit-difference check; and the
         control's purity being exactly 1 (the failing direction).
  SS7.3  THE DEGENERATE-ENTANGLEMENT GATE, new and blocking: every N-in item's
         referents must have concurrence >= 0.10 under the eps-map. NOT applied to
         N-across, where 4 of the 80 casts fall below 0.10 and are NOT excluded --
         dropping casts from a frozen dataset to make an arm look entangled is the
         opposite of a validity gate (orchestrator ruling 7).
  SS7.4  Inherited unchanged: word order, the T = 2 copy assertion, V1 = 0.500 with
         twins bit-identical, V8. exp69's V-suite results are RE-USED, not
         re-derived: the dataset is byte-identical and its sha is asserted.

INSTRUMENTATION (doc 56 SS15.1). Every per-run checkpoint records, for every held-out
item, the per-wire purity Tr rho^2 at FOUR named stages -- after intro, after event
1, after event 2 (T = 2 arms) and at readout -- FROM THE SAME FORWARD PASS that
produces the answer. **The INTRO stage is the re-pinning that matters**: for N-in and
N-across the intro purity is no longer 1 and is the quantity SS7.2 and SS7.3 gate on.

    python exp70_predictions.py                       # FIRST, and it already ran
    python exp70_run.py --smoke                       # doc 56 SS7, ALL gates
    python exp70_run.py --anchor-gate                 # doc 56 SS6.1, BLOCKING, FIRST
    python exp70_run.py --worker i --nworkers 5
    python exp70_comparator.py                        # doc 56 SS8, THE ARBITER
    python exp70_aggregate.py                         # the ONLY results writer
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

torch.set_num_threads(1)                              # doc 45 SS6.1, verified inert

import exp42_compiler as comp
import exp42_sim_adapter as sim
import exp68_run as E                                 # F2Model, feature maps
import exp69_run as R69                               # compile path, floors, gates
import instrument_sb_bounds as INS
import instrument_entangled_bounds as IEB             # the committed closed forms
from exp42_models import QuantumStoryModel
from exp42_run import grad_global_norm
from exp60_run import evaluate

SPEC = ("qnlp_private_docs/56_exp70_entangled_encodings.md (566b9ce, REGISTERED "
        "2026-08-25, orchestrator rulings 1-7 in the status header)")
SPEC_COMMIT = "566b9ce"
DATA_A = R69.DATA_A                                   # stories_exp69_QA.json, FROZEN
EXP69_RESULTS = "results_exp69.json"
EXP69_VSUITE = "results_exp69_WN_validation.json"
CKPT_DIR = "exp70_ckpt"
SMOKE_OUT = "results_exp70_smoke_validation.json"
ANCHOR_OUT = "results_exp70_anchors.json"
PREDICTIONS = "exp70_predictions.json"
VSUITE = EXP69_VSUITE

# --------------------------------------------------------------------------
# THE CONFIRM DRAW (doc 56 SS5.1.1 / SS11 / SS17.3 item 8).
#
# doc 47 SS18 is the standing reason: A HEADLINE THAT SURVIVED ONE DRAW DIED ON
# THE SECOND. On any positive, exp70 must re-run at generator seed + 1000 with
# ALL ARMS PRESENT, and doc 56 SS5.1.1 fixes the scope precisely: the draw
# re-runs `exp69_datagen.py` -- THE COMMITTED GENERATOR, UNMODIFIED -- so it
# introduces NO new surface template, NO new noun and NO new verb, only fresh
# casts and pairings from the same inventories the user already confirmed.
# SS17.1's native-speaker gate therefore stays closed, because that gate governs
# NEW ARABIC, not new draws from a cleared generator.
#
# The switches below exist so the confirm draw runs THE SAME CODEPATH rather
# than a copy of it -- writing an `exp70_confirm_run.py` would fork the very
# codepath the anchors exist to prove is single. EVERY DEFAULT IS THE FIRST
# DRAW'S VALUE, so omitting the flags reproduces the first draw bit-identically.
#
# ONE ROLE CHANGES, AND IT MUST BE STATED (doc 56 SS6.1). On a FRESH DRAW the
# anchor arms A0-W and EA-ctrl ARE NOT BIT-EXACT GATES: bit-exactness was a
# property of exp69's FROZEN items, and there is no exp69 measurement on these
# items to compare against. On the confirm draw they are BASELINE ARMS and the
# anchor gate is SKIPPED rather than silently passed.
#
# The consequence is carried rather than hidden: doc 56 SS5.3 omitted the anchor
# families' floors on the argument that a |delta| <= 0.002 gate against
# floor-clearing exp69 measurements is TRANSITIVELY STRONGER than re-measuring
# the floor. **THAT ARGUMENT DOES NOT SURVIVE A FRESH DRAW**, because the gate it
# rests on does not exist here. exp70 does NOT respond by inventing two new floor
# families -- that would add 20 runs the registration never authorised, on the
# draw whose entire job is to re-test ONE registered boolean. It responds by
# DEMOTING the anchor arms on the confirm draw to UN-FLOORED DESCRIPTIVE
# BASELINES: they are reported, they are compared to the first draw, and NO
# BOOLEAN READS THEM. The rider verdict reads `entanglement_hurts_T2`, which is
# N-in-T2 against P-eps-T2, and BOTH of those carry matched floors re-drawn on
# the fresh draw at the same budget.
# --------------------------------------------------------------------------
CONFIRM_MODE = False
FIRST_DRAW_DATA = DATA_A

# Training config: the exp60 / exp61 / exp65 / exp68 / exp69 loop, UNCHANGED.
# Anything else here would put the anchors' bit-exactness out of reach.
LR, BATCH, MAX_EPOCHS = E.LR, E.BATCH, E.MAX_EPOCHS
TRAIN_GATE, VAL_GATE = E.TRAIN_GATE, E.VAL_GATE
RNG_BASE = E.RNG_BASE
N_SEEDS = E.N_SEEDS

ANCHOR_TOLERANCE = 0.002              # doc 56 SS6.1; honest expectation is 0.0
BUG_HUNT_MARGIN = 0.06                # doc 56 SS6.6
CONCURRENCE_GATE = 0.10               # doc 56 SS7.3

# doc 56 SS5.2. `enc` picks the encoding path, `T` the composition depth. Every arm
# runs on exp69's Part A.
FAM_CFG = {
    "A0-W":     {"enc": "F2", "qpr": 2,
                 "role": "bit-exact anchor for exp69 R1-W / R2-W"},
    "EA-ctrl":  {"enc": "FA", "qpr": 1,
                 "role": ("DUAL ROLE: bit-exact anchor for exp69 R1-A / R2-A AND "
                          "the map-matched contrast baseline for N-across")},
    "P-eps":    {"enc": "P-eps", "qpr": 2,
                 "role": "THE MAP-MATCHED PRODUCT TWIN (doc 56 SS4.3)"},
    "N-in":     {"enc": "E-in", "qpr": 2,
                 "role": "the intra-register entangled arm"},
    "N-across": {"enc": "E-across", "qpr": 1,
                 "role": "the mechanism probe (doc 56 SS4.4)"},
}
FAMILIES = ["A0-W", "EA-ctrl", "P-eps", "N-in", "N-across"]
ANCHOR_FAMILIES = ["A0-W", "EA-ctrl"]
NEW_FAMILIES = ["P-eps", "N-in", "N-across"]

BASE_ARMS = ["%s-T%d" % (f, t) for f in FAMILIES for t in (1, 2)]
ARM_CFG = {a: dict(FAM_CFG[a.rsplit("-T", 1)[0]],
                   family=a.rsplit("-T", 1)[0],
                   T=int(a.rsplit("-T", 1)[1]), part="A")
           for a in BASE_ARMS}
ANCHOR_ARMS = [a for a in BASE_ARMS if ARM_CFG[a]["family"] in ANCHOR_FAMILIES]
FLOORED = [a for a in BASE_ARMS if ARM_CFG[a]["family"] in NEW_FAMILIES]
ARMS = BASE_ARMS + ["S-shuffle-%s" % a for a in FLOORED]
FLOOR_OF = {a: "S-shuffle-%s" % a for a in FLOORED}

# doc 56 SS6.1: the exp69 arms each exp70 anchor must reproduce, and the registered
# means the doc quotes. The PER-SEED values are read from results_exp69.json.
ANCHOR_OF = {"A0-W-T1": "R1-W", "A0-W-T2": "R2-W",
             "EA-ctrl-T1": "R1-A", "EA-ctrl-T2": "R2-A"}
ANCHOR_MEAN_DOC56 = {"EA-ctrl-T1": 0.9227, "EA-ctrl-T2": 0.9187,
                     "A0-W-T1": 0.9820, "A0-W-T2": 0.7293}

BOUND_FAMILY = {a: ARM_CFG[a]["family"] for a in BASE_ARMS}


def base_arm(arm):
    return arm[len("S-shuffle-"):] if arm.startswith("S-shuffle-") else arm


def _u(t):
    return (1.0, math.cos(t), math.sin(t))


# =========================================================== the feature layer
def feature_maps():
    """exp68's committed maps, IMPORTED through `exp68_run.feature_maps` (never
    re-derived here -- shared code objects are the strongest guarantee the anchors
    have), plus exp70's eps-map.

    The eps-map is applied to `fm["coords"]`, which is
    `instrument_sb_bounds.load_raw`'s output under the EXACT normalisation divisors
    (doc 56 SS3 provenance note 2: the committed degeneracy diagnostic used the
    six-decimal-ROUNDED divisors; under the exact ones its concurrences shift by at
    most 4.4e-08, seven orders of magnitude below the SS7.3 gate). `IEB.eps_map` is
    imported, never re-typed."""
    fm = E.feature_maps()
    fm["theta_eps1"] = {n: float(IEB.eps_map(fm["coords"][n][0])) for n in INS.HUMANS}
    fm["theta_eps2"] = {n: float(IEB.eps_map(fm["coords"][n][1])) for n in INS.HUMANS}
    fm["eps"] = IEB.EPS
    fm["eps_map_formula"] = ("theta_k = p_k (pi - 2 eps)/2 + pi/2, eps = pi/6 -> "
                             "theta in [pi/6, 5pi/6]; FORCED by doc 56 SS3's "
                             "committed degeneracy check, imported from "
                             "instrument_entangled_bounds.eps_map")
    fm["concurrence_eps"] = {
        n: float(IEB.ein_concurrence(fm["theta_eps1"][n], fm["theta_eps2"][n]))
        for n in INS.HUMANS}
    return fm


def arm_angles(arm, fm):
    """(intro angles consumed by exp42_compiler, second-qubit angles or None)."""
    enc = ARM_CFG[base_arm(arm)]["enc"]
    if enc in ("FA", "E-across"):
        return fm["phi_exp42"], None
    if enc == "F2":
        return fm["theta_pc1"], fm["theta_pc2"]
    if enc in ("P-eps", "E-in"):
        return fm["theta_eps1"], fm["theta_eps2"]
    raise ValueError(arm)


def referent_features(arm, fm):
    """The per-referent vector the arm's circuit is LINEAR in. This is the object
    doc 56 SS8.1's function-class-matched comparator must consume, and the object
    SS7.1's closure test certifies.

      FA / E-across   u(phi) = (1, cos, sin)                          d = 3
      F2 / P-eps      u(theta1) (x) u(theta2)                         d = 9
      E-in            (a^2, b^2, c^2, ab, ac, bc), the outer square
                      of the three E-in state coefficients            d = 6

    **E-across has NO per-referent factorisation when the CZ is present**, so that
    family is handled by `across_features` and is deliberately absent here (doc 56
    SS8.2)."""
    enc = ARM_CFG[base_arm(arm)]["enc"]
    a0, a1 = arm_angles(arm, fm)
    if enc == "E-in":
        return {n: IEB.ein_features(a0[n], a1[n]) for n in INS.HUMANS}
    if a1 is None:
        return {n: np.array(_u(a0[n])) for n in INS.HUMANS}
    return {n: np.outer(np.array(_u(a0[n])), np.array(_u(a1[n]))).ravel()
            for n in INS.HUMANS}


def across_features(arm, fm):
    """doc 56 SS8.2: the JOINT 10-coordinate feature for the E-across families --
    the 10 independent entries of the outer square of the 4-dim real amplitude
    vector, equivalently the 10 real Pauli coordinates. Returned as a callable
    (agent, patient) -> phi because it does NOT factorise."""
    cz = ARM_CFG[base_arm(arm)]["enc"] == "E-across"
    ang = fm["phi_exp42"]
    return (lambda a, p: IEB.across_features(ang[a], ang[p], cz=cz)), cz


# ================================================================== the gates
def cry_matrix(theta):
    """CRY(theta) controlled on the FIRST wire passed to `apply_gate`, targeting
    the second. Real, constant, DATA -- never a Parameter and never trained."""
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    M = torch.zeros((4, 4), dtype=sim.CDTYPE)
    M[0, 0] = 1.0
    M[1, 1] = 1.0
    M[2, 2] = c
    M[2, 3] = -s
    M[3, 2] = s
    M[3, 3] = c
    return M


CZ = torch.diag(torch.tensor([1.0, 1.0, 1.0, -1.0], dtype=sim.CDTYPE))

# --------------------------------------------------------------------------
# THE PRECISION NOTE, and it is load-bearing for doc 56 SS7.2's thresholds.
#
# `torch_sv_sim.CDTYPE` is `complex64`. The simulator is float32 END TO END, and
# has been since exp40 -- the anchors' bit-exactness depends on that being left
# alone, so exp70 does NOT change it. The consequence is a hard floor: the purity
# of an exactly-product two-qubit state evaluates to 1 + 1.19e-07 at runtime, not
# to 1. doc 56 SS7.2's assertions are registered at 1e-6 (per-item purity vs the
# closed form) and 1e-9 (the control's purity is exactly 1), and THE SECOND
# THRESHOLD IS BELOW THE SIMULATOR'S RESOLUTION. Writing the gate at 1e-6 to make
# it pass would be loosening a registered threshold after seeing the platform.
#
# The build instead recomputes the INTRO CIRCUIT -- the same gates, from the same
# model's own angle data, in the same order -- in complex128, and asserts doc 56's
# thresholds as registered against that. The double-precision object is held to
# the runtime one at float32 tolerance first, so it is provably the same circuit
# and not a second implementation that could agree with the closed form while the
# runtime disagreed. BOTH numbers are reported. Nothing about the trained forward
# pass changes: the checkpointed purity traces are the runtime ones, taken from
# the same pass that produces the answer, exactly as doc 56 SS15.1 requires.
# --------------------------------------------------------------------------
F32_PURITY_FLOOR = 1e-6               # measured 1.19e-07; the assertion tolerance
_CZ64 = np.diag(np.array([1.0, 1.0, 1.0, -1.0], dtype=np.complex128))


def _ry64(t):
    c, s = math.cos(t / 2.0), math.sin(t / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _cry64(t):
    c, s = math.cos(t / 2.0), math.sin(t / 2.0)
    M = np.eye(4, dtype=np.complex128)
    M[2, 2], M[2, 3], M[3, 2], M[3, 3] = c, -s, s, c
    return M


def _apply64(vec, U, wires, nw):
    """`torch_sv_sim.apply_gate`'s contraction, in complex128."""
    k = len(wires)
    T = vec.reshape((2,) * nw)
    Ut = U.reshape((2,) * (2 * k))
    T = np.tensordot(Ut, T, axes=(list(range(k, 2 * k)), list(wires)))
    T = np.moveaxis(T, list(range(k)), list(wires))
    return T.reshape(-1)


def intro_state_f64(m, cs):
    """`apply_intro` in complex128, from the model's OWN angle data."""
    reg = is_register(m)
    ein = isinstance(m, EInModel)
    nw = qubits_per_referent(m) * cs.K
    v = np.zeros(2 ** nw, dtype=np.complex128)
    v[0] = 1.0
    for w, noun, angle in cs.intro:
        v = _apply64(v, _ry64(float(angle)), (2 * w,) if reg else (w,), nw)
        if ein:
            v = _apply64(v, _cry64(float(m.theta1[noun])), (2 * w, 2 * w + 1), nw)
        elif reg and m.wide:
            v = _apply64(v, _ry64(float(m.theta1[noun])), (2 * w + 1,), nw)
    if isinstance(m, EAcrossModel) and m.cz:
        v = _apply64(v, _CZ64, (0, 1), nw)
    return v


def wire_purities64(v, nw):
    T = np.asarray(v).reshape((2,) * nw)
    out = []
    for w in range(nw):
        p = np.moveaxis(T, w, 0).reshape(2, -1)
        rho = p @ p.conj().T
        out.append(float((np.abs(rho) ** 2).sum()))
    return out


# ================================================================== the models
class EInModel(E.F2Model):
    """doc 56 SS4.1 -- N-in. `exp68_run.F2Model` with ONE change to the intro:
    the second qubit's independent `RY(theta2)` is replaced by `CRY(theta2)`
    controlled by qubit 0.

        |psi(r)> = cos(t1/2)|00> + sin(t1/2)cos(t2/2)|10> + sin(t1/2)sin(t2/2)|11>
        concurrence = |sin t1| |sin(t2/2)|,  per-wire purity = 1 - C^2/2

    ORIENTATION (doc 56 SS1.3): qubit 0 carries PC1 and CONTROLS; qubit 1 carries
    PC2 and is the target. Registered a priori on the stated principle that PC1
    carries 12.7 % of the AraVec variance against PC2's 8.1 %, BEFORE any number
    was computed. The reverse orientation is NOTED, NOT RUN (doc 56 SS13 item 1),
    and its bound is reported as a sensitivity row so the choice is auditable
    rather than merely asserted -- the bound does NOT discriminate, which is what
    makes it clear the registration could not have been fitted to it.

    NO NEW PARAMETERS AND NO RNG IS CONSUMED beyond `super().__init__`, so the
    parameter budget is byte-identical to P-eps's. The entangling angle is DATA,
    taken from the referent's own AraVec coordinate: entanglement here is a
    SEMANTIC-COORDINATE-DRIVEN QUANTITY with a fixed mechanism, not a tunable
    parameter and not a random draw (doc 56 SS1.1)."""

    def __init__(self, verbs, seed, theta1, layers=3):
        super().__init__(verbs, seed, theta1, layers=layers, wide=True)

    def story_values(self, cs, verb_map=None):
        return _readout(self, cs, verb_map)


class EAcrossModel(QuantumStoryModel):
    """doc 56 SS4.4 -- N-across. `exp42_models.QuantumStoryModel('A3')` -- the
    2-wire model exp60 / exp61 / exp68 / exp69 use -- with ONE `CZ` between the two
    referent wires, immediately after intro and before event 1.

        |psi> = (cAcB, cAsB, sAcB, -sAsB),  c = cos(t/2), s = sin(t/2)
        concurrence = |sin tA| |sin tB|
        per-wire purity = 1 - (sin tA sin tB)^2 / 2   (identical on both wires)

    ONE GATE, ONE PAIR, ONE POSITION. No entangler is placed inside an event block,
    none is parameterised, and no wire pairing is drawn at random at any point
    (doc 56 SS1.3).

    `cz=False` is EA-ctrl, and it MUST reproduce the parent bit-for-bit: no
    generator is consumed after `super().__init__` and the flag gates exactly one
    constant gate. That parity is asserted at smoke, not assumed."""

    def __init__(self, verbs, seed, cz=True):
        super().__init__("A3", verbs, seed)
        self.cz = bool(cz)

    def story_values(self, cs, verb_map=None):
        return _readout(self, cs, verb_map)


def is_register(m):
    """2 qubits per referent?"""
    return isinstance(m, E.F2Model)


def qubits_per_referent(m):
    return 2 if is_register(m) else 1


def apply_intro(m, psi, cs):
    """THE SINGLE SOURCE OF TRUTH for exp70's four intro paths.

    The two ANCHOR families never route their TRAINING through this function --
    they call the committed parents' own `story_values` -- so this is a
    re-implementation for them, and `check_trace()` asserts it reproduces the
    committed intros BIT-FOR-BIT before any anchor number is read."""
    reg = is_register(m)
    ein = isinstance(m, EInModel)
    for w, noun, angle in cs.intro:
        psi = sim.apply_gate(psi, sim.ry(angle), (2 * w,) if reg else (w,))
        if ein:                                   # doc 56 SS4.1: the CRY, q0 -> q1
            psi = sim.apply_gate(psi, cry_matrix(m.theta1[noun]), (2 * w, 2 * w + 1))
        elif reg and m.wide:                      # doc 54 SS6.4: an independent RY
            psi = sim.apply_gate(psi, sim.ry(m.theta1[noun]), (2 * w + 1,))
    if isinstance(m, EAcrossModel) and m.cz:      # doc 56 SS4.4: ONE CZ, after intro
        psi = sim.apply_gate(psi, CZ, (0, 1))
    return psi


def intro_spec(m, cs):
    """The compiled intro as an ordered gate spec, so doc 56 SS7.2's
    'must be DIFFERENT CIRCUITS' is a structural assertion and not only a numeric
    one. Compared by value."""
    reg = is_register(m)
    ein = isinstance(m, EInModel)
    ops = []
    for w, noun, angle in cs.intro:
        ops.append(("RY", (2 * w,) if reg else (w,), round(float(angle), 12)))
        if ein:
            ops.append(("CRY", (2 * w, 2 * w + 1), round(float(m.theta1[noun]), 12)))
        elif reg and m.wide:
            ops.append(("RY", (2 * w + 1,), round(float(m.theta1[noun]), 12)))
    if isinstance(m, EAcrossModel) and m.cz:
        ops.append(("CZ", (0, 1), None))
    return tuple(ops)


def _readout(m, cs, verb_map=None):
    """The events + question path, identical for every exp70 family; only the
    intro differs. Gate order mirrors `exp42_models.QuantumStoryModel.story_values`
    and `exp68_run.F2Model.story_values` exactly."""
    vm = verb_map or {}
    cache = {}

    def block(v):
        v = vm.get(v, v)
        if v not in cache:
            cache[v] = m.verb_matrix(v)
        return cache[v]

    reg = is_register(m)

    def wires_of(w):
        return (2 * w, 2 * w + 1) if reg else (w,)

    psi = apply_intro(m, sim.init_state(qubits_per_referent(m) * cs.K), cs)
    for v, wa, wp in cs.events:
        psi = sim.apply_gate(psi, block(v), wires_of(wa) + wires_of(wp))
    vseq, qa, qp = cs.question
    qw = wires_of(qa) + wires_of(qp)
    for qv in reversed(vseq):
        psi = sim.apply_gate(psi, block(qv).conj().T, qw)
    return torch.stack([sim.prob_all_zero_on(
        sim.apply_gate(psi, eff.matrix().conj().T, qw), qw)
        for eff in (m.q_pos, m.q_neg)])


def build_model(arm, verbs, seed, fm):
    """doc 56 SS4.7-equivalent: NO NEW ANSATZ. The anchors construct the COMMITTED
    parent classes verbatim, with the identical constructor arguments exp69 used,
    because doc 56 SS6.1 expects them to reproduce exp69 BIT-EXACTLY."""
    enc = ARM_CFG[base_arm(arm)]["enc"]
    if enc == "F2":                       # A0-W == exp69's R1-W / R2-W, verbatim
        return E.F2Model(verbs, seed, fm["theta_pc2"])
    if enc == "FA":                       # EA-ctrl == exp69's R1-A / R2-A, verbatim
        return QuantumStoryModel("A3", verbs, seed)
    if enc == "P-eps":                    # the SAME class, a different angle map
        return E.F2Model(verbs, seed, fm["theta_eps2"])
    if enc == "E-in":
        return EInModel(verbs, seed, fm["theta_eps2"])
    if enc == "E-across":
        return EAcrossModel(verbs, seed, cz=True)
    raise ValueError(arm)


def wire_purities(psi, nwires):
    """Tr rho^2 per wire, rho the single-qubit reduced density matrix."""
    out = []
    for w in range(nwires):
        p = torch.movedim(psi, w, 0).reshape(2, -1)
        rho = p @ p.conj().transpose(0, 1)
        out.append(float((rho.abs() ** 2).sum()))
    return out


def forward_traced(m, cs):
    """doc 56 SS15.1: the forward pass with the per-wire purity trace taken FROM THE
    SAME PASS, at FOUR stages. The intro stage is exp70's re-pinning: for N-in and
    N-across it is no longer 1 and is the quantity SS7.2 / SS7.3 gate on."""
    reg = is_register(m)
    nw = qubits_per_referent(m) * cs.K
    cache = {}

    def block(v):
        if v not in cache:
            cache[v] = m.verb_matrix(v)
        return cache[v]

    def wires_of(w):
        return (2 * w, 2 * w + 1) if reg else (w,)

    psi = apply_intro(m, sim.init_state(nw), cs)
    trace = {"after_intro": wire_purities(psi, nw)}
    for k, (v, wa, wp) in enumerate(cs.events):
        psi = sim.apply_gate(psi, block(v), wires_of(wa) + wires_of(wp))
        trace["after_event%d" % (k + 1)] = wire_purities(psi, nw)
    vseq, qa, qp = cs.question
    qw = wires_of(qa) + wires_of(qp)
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


# ================================================== data loading / compilation
def load_part(path=None):
    """Part A. By default exp69's FROZEN draw (doc 56 SS5.1: no exp70 datagen
    exists); on the confirm draw, the seed + 1000 re-draw from the SAME COMMITTED
    generator (SS5.1.1)."""
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


def item_referents(cs):
    return [n for _w, n, _a in cs.intro]


# ================================================ doc 56 SS7 -- the smoke gates
def check_dataset(store, rep):
    """doc 56 SS7.4 / SS17.1: the dataset is exp69's, BYTE-IDENTICAL, and its sha is
    asserted here so 'nothing changed' is checked rather than assumed. exp69's
    V-suite results are RE-USED on the strength of this assertion and are not
    re-derived."""
    _items, meta, path = store
    ref = json.load(open(EXP69_RESULTS, encoding="utf-8"))
    exp69_sha = ref["datasets"]["A"]["sha256"]
    if CONFIRM_MODE:
        # THE FRESH DRAW IS A NEW DATASET AND IS TREATED AS ONE (doc 56 SS5.1.1).
        # Its V-suite is RE-RUN, not inherited, and it must DIFFER from the first
        # draw -- a confirm draw that silently reproduced the first draw's items
        # would be a null instrument in doc 53 SS5's exact sense.
        assert meta["dataset_sha256"] != exp69_sha, \
            ("the confirm draw reproduced the FIRST draw's sha (%s) -- it is not a "
             "fresh draw and confirms nothing" % exp69_sha)
        vs = json.load(open(VSUITE, encoding="utf-8"))
        assert vs.get("kill_gate_passed") is True, \
            ("the confirm draw's OWN V-suite kill gate did not pass (%s) -- no "
             "confirm-draw result is admissible" % VSUITE)
        assert vs["parts"]["A"]["dataset_sha256"] == meta["dataset_sha256"], \
            ("%s validates a different dataset than the one being run" % VSUITE)
    else:
        assert meta["dataset_sha256"] == exp69_sha, \
            ("exp70 is not running on exp69's Part A: %s vs %s -- doc 56 SS5.1's "
             "no-datagen gate and SS17.1's native-speaker closure BOTH depend on "
             "byte identity" % (meta["dataset_sha256"], exp69_sha))
        vs = json.load(open(VSUITE, encoding="utf-8"))
        assert vs.get("kill_gate_passed") is True, \
            "exp69's V-suite kill gate did not pass -- its results may not be re-used"
    rep["dataset"] = {
        "path": path, "sha256": meta["dataset_sha256"],
        "confirm_draw": bool(CONFIRM_MODE),
        "first_draw_sha256": exp69_sha,
        "differs_from_first_draw": bool(meta["dataset_sha256"] != exp69_sha),
        "matches_exp69_results": bool(meta["dataset_sha256"] == exp69_sha),
        "vsuite_source": VSUITE, "vsuite_kill_gate_passed": True,
        "vsuite_status": ("RE-RUN ON THIS DRAW -- it is a new dataset and is treated "
                          "as one (doc 56 SS5.1.1)" if CONFIRM_MODE else
                          "INHERITED from exp69 on the strength of byte identity "
                          "(doc 56 SS7.4)"),
        "anchor_role_on_this_draw": (
            "BASELINE ARMS, NOT BIT-EXACT GATES. Bit-exactness was a property of "
            "exp69's FROZEN items and there is no exp69 measurement on these items. "
            "doc 56 SS5.3's floor-omission argument rested on the |delta| <= 0.002 "
            "gate and DOES NOT SURVIVE A FRESH DRAW, so the anchor arms are demoted "
            "to UN-FLOORED DESCRIPTIVE BASELINES here and NO BOOLEAN READS THEM. "
            "The rider verdict reads entanglement_hurts_T2 = N-in-T2 vs P-eps-T2, "
            "and both of those carry floors re-drawn on this draw."
            if CONFIRM_MODE else
            "BLOCKING BIT-EXACT GATES against exp69 R1-W / R2-W / R1-A / R2-A "
            "(doc 56 SS6.1)"),
        "no_datagen_doc56_SS5_1": (
            "exp70 introduces NO new Arabic surface, no new noun and no new verb. "
            "exp70_datagen.py DOES NOT EXIST AND MAY NOT BE WRITTEN (doc 56 SS5.1 "
            "/ SS16.1). doc 50 SS5.2's native-speaker gate therefore does not "
            "reopen (SS17.1) -- and that is stated explicitly because 'nothing "
            "changed' is exactly the claim that should be checked rather than "
            "assumed. exp69's V-suite results are RE-USED, not re-derived, on the "
            "strength of the sha assertion above."),
        "confirm_draw_obligation_doc56_SS5_1_1": (
            "on any positive, doc 56 SS11 / SS17.3 item 8 mandate a CONFIRM DRAW at "
            "generator seed + 1000 which re-runs the COMMITTED exp69_datagen.py "
            "UNMODIFIED -- no new template, no new noun, no new verb, only fresh "
            "casts and pairings from the same cleared inventories. The full V-suite "
            "and every SS7 assertion re-run on that draw, and the S-shuffle floors "
            "are RE-DRAWN matched to it (carrying them over would be doc 47 SS16.1's "
            "unmatched comparison)."),
    }
    return rep


def check_features(fm, rep):
    """doc 56 SS7.1(3) + SS3. `exp68_run.check_features` is the COMMITTED raw-PCA
    check, called unchanged; exp70 adds the eps-map's own assertions."""
    E.check_features(fm, rep)
    t1 = np.array([fm["theta_eps1"][n] for n in INS.HUMANS])
    t2 = np.array([fm["theta_eps2"][n] for n in INS.HUMANS])
    lo, hi = math.pi / 6.0, 5.0 * math.pi / 6.0
    assert t1.min() >= lo - 1e-12 and t1.max() <= hi + 1e-12, "theta_eps1 out of range"
    assert t2.min() >= lo - 1e-12 and t2.max() <= hi + 1e-12, "theta_eps2 out of range"
    assert len({round(float(x), 9) for x in t1}) >= 20, "theta_eps1 collapses"
    assert len({round(float(x), 9) for x in t2}) >= 20, "theta_eps2 collapses"

    # the eps-map must reproduce the instrument's, element by element
    worst = max(abs(fm["theta_eps1"][n] - IEB.eps_map(fm["coords"][n][0]))
                for n in INS.HUMANS)
    assert worst == 0.0, "the runtime eps-map differs from the instrument's"

    # SS3's cost of the disqualified map, recomputed here rather than quoted
    old_c = {n: IEB.ein_concurrence(IEB.full_map(fm["coords"][n][0]),
                                    IEB.full_map(fm["coords"][n][1]))
             for n in INS.HUMANS}
    eps_c = fm["concurrence_eps"]
    rep["features"]["eps_map"] = {
        "formula": fm["eps_map_formula"], "eps": fm["eps"],
        "range_theta1": [float(t1.min()), float(t1.max())],
        "range_theta2": [float(t2.min()), float(t2.max())],
        "bitmatch_vs_instrument_eps_map": worst,
        "concurrence_min": float(min(eps_c.values())),
        "concurrence_median": float(np.median(list(eps_c.values()))),
        "concurrence_max": float(max(eps_c.values())),
        "n_nouns_below_0.05": int(sum(1 for v in eps_c.values() if v < 0.05)),
        "n_nouns_below_0.10": int(sum(1 for v in eps_c.values() if v < 0.10)),
        "disqualified_map_n_below_0.05":
            int(sum(1 for v in old_c.values() if v < 0.05)),
        "disqualified_map_min": float(min(old_c.values())),
        "why_forced": (
            "doc 56 SS3: THE EPS-MAP IS FORCED BY THE COMMITTED DEGENERACY CHECK, "
            "NOT CHOSEN FOR CONVENIENCE. Under exp68's registered map "
            "theta = (p+1)pi the normalisation extremes |p| = 1 land exactly on the "
            "dead zones sin t1 = 0 and sin(t2/2) = 0: FOUR of the 24 nouns fall "
            "below concurrence 0.05, two of them EXACTLY AT ZERO, and one of those "
            "sits in the target category. Those four would have carried a PRODUCT "
            "state through an arm named 'entangled' -- a null instrument in doc 53 "
            "SS5's exact sense, for a sixth of the inventory. Measured cost: the "
            "E-in per-twin bound is 0.9500 (152/160) under the disqualified map "
            "against 1.0000 (160/160) under the eps-map -- eight twins."),
        "exact_divisor_note": (
            "exp70 applies the eps-map to instrument_sb_bounds.load_raw's EXACT "
            "normalisation divisors, imported and never re-typed. The committed "
            "degeneracy diagnostic used the six-decimal-ROUNDED divisors printed in "
            "the instrument's provenance string; under the exact ones its eps-map "
            "concurrences shift by at most 4.4e-08, seven orders of magnitude below "
            "the SS7.3 gate. The committed finding stands unchanged (doc 56 SS3 note "
            "2, orchestrator ruling 5)."),
    }
    return rep


def check_intro_states(fm, verbs, store, rep):
    """exp70's own addition, and it is the sharpest single gate here: the COMPILED
    intro statevector must equal the committed instrument's CLOSED FORM, coefficient
    by coefficient. A control/target inversion, a wire-order slip or a half-angle
    error all fail here immediately and unambiguously, where the Fourier sweep would
    only localise them slowly."""
    out = {}
    for arm in ("N-in-T1", "N-across-T1", "P-eps-T1", "EA-ctrl-T1", "A0-W-T1"):
        items, _m, _p = arm_items(arm, store)
        ang, a1 = arm_angles(arm, fm)
        data = R69.compile_all([i for i in items if i["split"] == "test"][:8], ang)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        enc = ARM_CFG[arm]["enc"]
        worst, worst64, worst_rt_vs_64 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for cs in data:
                psi = apply_intro(m, sim.init_state(
                    qubits_per_referent(m) * cs.K), cs)
                v = psi.reshape(-1).detach().numpy()
                v64 = intro_state_f64(m, cs)
                worst_rt_vs_64 = max(worst_rt_vs_64,
                                     float(np.abs(v - v64).max()))
                names = item_referents(cs)
                if enc == "E-in":
                    per = [np.array(IEB.ein_coeffs(ang[n], a1[n])) for n in names]
                    ref = np.kron(np.array([per[0][0], 0.0, per[0][1], per[0][2]]),
                                  np.array([per[1][0], 0.0, per[1][1], per[1][2]]))
                elif enc == "E-across":
                    ref = IEB.across_amplitudes(ang[names[0]], ang[names[1]], cz=True)
                elif enc == "FA":
                    ref = IEB.across_amplitudes(ang[names[0]], ang[names[1]], cz=False)
                else:
                    per = [np.kron([math.cos(ang[n] / 2), math.sin(ang[n] / 2)],
                                   [math.cos(a1[n] / 2), math.sin(a1[n] / 2)])
                           for n in names]
                    ref = np.kron(per[0], per[1])
                worst = max(worst, float(np.abs(np.real(v) - ref).max()),
                            float(np.abs(np.imag(v)).max()))
                worst64 = max(worst64, float(np.abs(np.real(v64) - ref).max()),
                              float(np.abs(np.imag(v64)).max()))
        assert worst < F32_PURITY_FLOOR and worst64 < 1e-12, \
            ("%s's compiled intro state does not match the committed closed form "
             "(runtime %.3e, float64 %.3e). For E-in this is the control/target "
             "orientation and the half-angle; for E-across it is the CZ's sign on "
             "|11>." % (arm, worst, worst64))
        assert worst_rt_vs_64 < F32_PURITY_FLOOR, \
            ("%s: the float64 recomputation is not the same circuit as the runtime "
             "one (%.3e)" % (arm, worst_rt_vs_64))
        out[arm] = {"encoding": enc, "max_abs_diff_vs_closed_form": worst,
                    "max_abs_diff_vs_closed_form_float64": worst64,
                    "runtime_vs_float64_max_abs": worst_rt_vs_64,
                    "closed_form_source": "instrument_entangled_bounds"}
    rep["intro_state_closed_forms"] = {
        "per_arm": out,
        "note": ("every exp70 intro state is asserted AMPLITUDE BY AMPLITUDE against "
                 "the committed instrument's closed form: E-in against "
                 "`ein_coeffs` -> (a, 0, b, c) with the |01> amplitude IDENTICALLY "
                 "ZERO, E-across against `across_amplitudes` with the CZ's minus "
                 "sign on |11>, EA-ctrl against the same function with cz=False, and "
                 "the product registers against RY (x) RY. The instrument is the "
                 "authority for the maths and this is where the circuit is held to "
                 "it."),
        "precision_note": (
            "the simulator is complex64 END TO END (torch_sv_sim.CDTYPE, unchanged "
            "since exp40 and left alone here because the anchors' bit-exactness "
            "depends on it), so the runtime agreement floor is ~1.2e-07. Each intro "
            "state is therefore ALSO recomputed in complex128 from the model's own "
            "angle data through the same gate sequence, held to the runtime state at "
            "float32 tolerance so it is provably the SAME circuit, and asserted "
            "against the closed form at 1e-12."),
    }
    return rep


def fourier_resid_1d(m, cs_list, n_terms):
    """doc 56 SS7.1: fit P(yes) against ONE referent's angle in the n_terms basis."""
    ts = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    worst = 0.0
    for cs in cs_list[:4]:
        ys = []
        for t in ts:
            c2 = copy.deepcopy(cs)
            c2.intro[0] = (c2.intro[0][0], c2.intro[0][1], float(t))
            with torch.no_grad():
                ys.append(float(m.story_values(c2)[0]))
        cols = [np.ones_like(ts)]
        for k in range(1, n_terms // 2 + 1):
            cols += [np.cos(k * ts), np.sin(k * ts)]
        A = np.stack(cols, 1)
        coef, *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
        worst = max(worst, float(np.abs(A @ coef - np.array(ys)).max()))
    return worst


def fourier_resid_2d(m, cs_list, grid=12):
    """The 2-D sweep over (theta1 on the intro, theta2 on the second qubit), fitted
    against THREE bases at once so SS7.1's double negative can be read off one sweep:

      resid3   the 3-term basis of theta1 ALONE      -- must FAIL for every register
      resid9   the 9-dim product basis u (x) u       -- must CLOSE for P-eps / A0-W,
                                                        must FAIL for N-in
      resid6   the 6-dim E-in basis                  -- must CLOSE for N-in

    `PRODUCT_BASIS` and `EIN_BASIS` are IMPORTED from the committed instrument and
    never re-typed here: the basis a gate is checked against must be the basis the
    bound was computed over, or the gate is checking something else."""
    g = np.linspace(0, 2 * math.pi, grid, endpoint=False)
    worst = {"3": 0.0, "9": 0.0, "6": 0.0}
    worst_d = 0.0
    for cs in cs_list[:3]:
        noun = cs.intro[0][1]
        keep = m.theta1[noun]
        ys, B3, B9, B6 = [], [], [], []
        for t0 in g:
            c2 = copy.deepcopy(cs)
            c2.intro[0] = (c2.intro[0][0], noun, float(t0))
            for t1 in g:
                m.theta1[noun] = float(t1)
                with torch.no_grad():
                    ys.append(float(m.story_values(c2)[0]))
                B3.append(list(_u(float(t0))))
                B9.append([f(float(t0), float(t1)) for f in IEB.PRODUCT_BASIS])
                B6.append([f(float(t0), float(t1)) for f in IEB.EIN_BASIS])
        m.theta1[noun] = keep
        ys = np.asarray(ys)
        for Bm, key in ((np.asarray(B3), "3"), (np.asarray(B9), "9"),
                        (np.asarray(B6), "6")):
            c, *_ = np.linalg.lstsq(Bm, ys, rcond=None)
            worst[key] = max(worst[key], float(np.abs(Bm @ c - ys).max()))
        with torch.no_grad():
            m.theta1[noun] = keep
            a = float(m.story_values(cs)[0])
            m.theta1[noun] = keep + 0.7
            b = float(m.story_values(cs)[0])
            m.theta1[noun] = keep
        worst_d = max(worst_d, abs(a - b))
    return worst, worst_d


def check_closures(fm, verbs, store, rep):
    """doc 56 SS7.1, per arm, re-derived for the new maps -- and SS7.2's refusal to
    apply a gate that provably cannot fail."""
    out = {}
    for arm in ("EA-ctrl-T1", "EA-ctrl-T2"):
        items, _m, _p = arm_items(arm, store)
        ang, _t1 = arm_angles(arm, fm)
        data = R69.compile_all([i for i in items if i["split"] == "test"], ang)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        r3 = fourier_resid_1d(m, data, 3)
        assert r3 < 1e-6, \
            ("%s does not close at 3 terms (%.3e) -- a second injection has appeared "
             "on a one-RY wire (doc 56 SS7.1)" % (arm, r3))
        out[arm] = {"resid_3term": r3, "closes_at_3": True, "T": ARM_CFG[arm]["T"],
                    "basis": "u(theta) = (1, cos, sin), joint 9"}
    for arm in ("A0-W-T1", "A0-W-T2", "P-eps-T1", "P-eps-T2"):
        items, _m, _p = arm_items(arm, store)
        ang, _t1 = arm_angles(arm, fm)
        data = R69.compile_all([i for i in items if i["split"] == "test"], ang)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        w, wd = fourier_resid_2d(m, data)
        assert w["3"] > 1e-3, \
            ("%s CLOSES in the 3-term basis of theta1 alone (%.3e) -- the second "
             "qubit is unwired and NO result from this arm is admissible" % (arm, w["3"]))
        assert w["9"] < 1e-6, \
            ("%s does not close in the 9-dim product basis (%.3e)" % (arm, w["9"]))
        assert wd > 1e-6, "perturbing theta2 does not move %s's output" % arm
        out[arm] = {"resid_3term_theta1_only": w["3"],
                    "resid_9dim_product_basis": w["9"],
                    "resid_6dim_Ein_basis": w["6"],
                    "output_change_under_theta2_perturbation": wd,
                    "closes_at_3": False, "T": ARM_CFG[arm]["T"],
                    "basis": "u(theta1) (x) u(theta2), 12 x 12 grid over [0, 2pi)"}
    for arm in ("N-in-T1", "N-in-T2"):
        items, _m, _p = arm_items(arm, store)
        ang, _t1 = arm_angles(arm, fm)
        data = R69.compile_all([i for i in items if i["split"] == "test"], ang)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        w, wd = fourier_resid_2d(m, data)
        assert w["6"] < 1e-6, \
            ("%s does not close in the 6-dim E-in basis (%.3e) -- the arm is not "
             "computing what doc 56 SS4.1 says" % (arm, w["6"]))
        assert w["3"] > 1e-3, \
            ("%s CLOSES in the 3-term basis of theta1 alone (%.3e) -- the CRY's "
             "TARGET IS UNWIRED (theta2 == 0 collapses E-in to a single-angle map) "
             "and no N-in result is admissible (doc 56 SS7.1's double negative, "
             "first direction)" % (arm, w["3"]))
        assert w["9"] > 1e-3, \
            ("%s CLOSES in the 9-dim PRODUCT basis (%.3e) -- the half-angle terms "
             "sin(t1)cos(t2/2) and sin(t1)sin(t2/2) are ABSENT, so the CRY has been "
             "replaced by an uncontrolled RY and the entangler is not doing what "
             "doc 56 SS4.1 says (SS7.1's double negative, second direction)"
             % (arm, w["9"]))
        assert wd > 1e-6, "perturbing theta2 does not move %s's output" % arm
        out[arm] = {"resid_6dim_Ein_basis": w["6"],
                    "resid_3term_theta1_only": w["3"],
                    "resid_9dim_product_basis": w["9"],
                    "output_change_under_theta2_perturbation": wd,
                    "closes_at_6": True, "T": ARM_CFG[arm]["T"],
                    "basis": ("instrument_entangled_bounds.EIN_BASIS = {1, cos t1, "
                              "(1-cos t1)cos t2, (1-cos t1)sin t2, sin t1 cos(t2/2), "
                              "sin t1 sin(t2/2)}, 12 x 12 grid over [0, 2pi)")}
    rep["fourier_closure"] = {
        "per_arm": out,
        "N_in_double_negative_THE_ONE_THAT_MATTERS": (
            "doc 56 SS7.1. A CRY with an UNWIRED TARGET degenerates to (a^2, b^2, ab) "
            "in theta1 alone and would pass a naive 'closes in 6 dims' check by "
            "closing in 3. A CRY REPLACED BY AN UNCONTROLLED RY would close in the "
            "9-dim product basis. BOTH failures are asserted against explicitly, and "
            "no N-in result is admissible unless both directions pass. The measured "
            "residuals are %.3e (6-dim, must close), %.3e (3-term, must fail) and "
            "%.3e (9-dim product, must fail) at T = 1."
            % (out["N-in-T1"]["resid_6dim_Ein_basis"],
               out["N-in-T1"]["resid_3term_theta1_only"],
               out["N-in-T1"]["resid_9dim_product_basis"])),
        "N_across_has_NO_fourier_gate": (
            "doc 56 SS4.5 / SS7.2, and this is a FINDING, not an omission. CZ is a "
            "CLIFFORD gate, so conjugating a general readout by it PERMUTES the "
            "Pauli coordinates and leaves the reachable function space INVARIANT: "
            "the ten independent coordinates of the E-across joint state all lie in "
            "the 9-dim product span {1,cos tA,sin tA} (x) {1,cos tB,sin tB} and "
            "together FILL it, exactly as the no-CZ control's do (certified rank "
            "9 = 9 = 9 in diagnostic_entangled_bounds.json). Closure therefore holds "
            "with the CZ present AND absent, so the standard SS7.1 assertion is "
            "VACUOUS for this arm. doc 49 SS8's newly adopted checklist entry, "
            "proposed by doc 56 SS14 and ADOPTED by orchestrator ruling 6: BEFORE "
            "ASSERTING A VALIDITY GATE, CHECK WHETHER THE INTERVENTION IS A SYMMETRY "
            "OF THE QUANTITY THE GATE MEASURES -- A GATE THAT CANNOT FAIL IS NOT A "
            "GATE. Replaced by SS7.2's three assertions; see `across_validity`."),
        "coefficient_budget": {
            "EA-ctrl / N-across per referent": 3, "joint": 9,
            "A0-W / P-eps per referent": 9, "joint_81": 81,
            "N-in per referent": 6, "joint_36": 36,
            "identical_at_T1_and_T2": True,
            "note": ("doc 56 SS4.1's registered reading: the E-in span is "
                     "RESTRUCTURED, NOT ENLARGED. It ACQUIRES the half-frequency "
                     "terms sin(t1)cos(t2/2) and sin(t1)sin(t2/2) that no product "
                     "register can produce, and it LOSES sin(t1) standing alone, "
                     "which every product register has. Six dimensions against nine "
                     "with a four-dimensional intersection -- NEITHER CLASS CONTAINS "
                     "THE OTHER, and nothing about which reaches further follows "
                     "from counting dimensions."),
        },
    }
    return rep


def check_across_validity(fm, verbs, store, rep):
    """doc 56 SS7.2 -- three BLOCKING assertions replacing an instrument that
    provably cannot work."""
    ang, _t1 = arm_angles("N-across-T1", fm)
    res = {}
    for T in (1, 2):
        arm, ctrl = "N-across-T%d" % T, "EA-ctrl-T%d" % T
        items, _m, _p = arm_items(arm, store)
        data = R69.compile_all([i for i in items if i["split"] == "test"], ang)
        torch.manual_seed(0)
        m = EAcrossModel(verbs, 0, cz=True)
        torch.manual_seed(0)
        c = EAcrossModel(verbs, 0, cz=False)
        torch.manual_seed(0)
        parent = QuantumStoryModel("A3", verbs, 0)

        # (1) per-item intro purity == the closed form, COMPUTED, not "< 1"
        w_pur, w_ctrl, n_items = 0.0, 0.0, 0
        w_pur64, w_ctrl64 = 0.0, 0.0
        conc = []
        with torch.no_grad():
            for cs in data:
                _v, tr = forward_traced(m, cs)
                na, nb = item_referents(cs)
                tgt = IEB.across_purity(ang[na], ang[nb])
                nw = qubits_per_referent(m) * cs.K
                for p in tr["after_intro"]:
                    w_pur = max(w_pur, abs(p - tgt))
                for p in wire_purities64(intro_state_f64(m, cs), nw):
                    w_pur64 = max(w_pur64, abs(p - tgt))
                _v2, tr2 = forward_traced(c, cs)
                for p in tr2["after_intro"]:
                    w_ctrl = max(w_ctrl, abs(p - 1.0))
                for p in wire_purities64(intro_state_f64(c, cs), nw):
                    w_ctrl64 = max(w_ctrl64, abs(p - 1.0))
                conc.append(IEB.across_concurrence(ang[na], ang[nb]))
                n_items += 1
        assert w_pur < 1e-6 and w_pur64 < 1e-9, \
            ("SS7.2(1) FAILED for %s: per-item intro purity departs from the closed "
             "form 1 - (sin tA sin tB)^2/2 by %.3e (runtime) / %.3e (float64). This "
             "asserts the COMPUTED value per item, not the vacuous '< 1' -- a '< 1' "
             "check would pass on any stray entangling gate anywhere, which is the "
             "null-instrument failure SS7.2 exists to prevent."
             % (arm, w_pur, w_pur64))
        assert w_ctrl64 < 1e-9, \
            ("SS7.2(3) FAILED for %s: the CONTROL's intro purity is not exactly 1 "
             "(%.3e in float64). This is the failing direction, without which "
             "assertion 1 could be satisfied by a build that entangles everything."
             % (ctrl, w_ctrl64))
        assert w_ctrl < F32_PURITY_FLOOR, \
            ("SS7.2(3) FAILED for %s at RUNTIME precision: %.3e" % (ctrl, w_ctrl))

        # (2) circuit difference -- structural AND numeric
        specs_differ, cz_in, cz_out = 0, 0, 0
        w_state = 0.0
        with torch.no_grad():
            for cs in data[:12]:
                sm, sc = intro_spec(m, cs), intro_spec(c, cs)
                specs_differ += int(sm != sc)
                cz_in += int(any(o[0] == "CZ" for o in sm))
                cz_out += int(any(o[0] == "CZ" for o in sc))
                w_state = max(w_state,
                              float((m.story_values(cs) - c.story_values(cs))
                                    .abs().max()))
        assert specs_differ == len(data[:12]) and cz_in == len(data[:12]) \
            and cz_out == 0, \
            ("SS7.2(2) FAILED: the with-CZ and without-CZ compilations are the SAME "
             "circuit -- the CZ is not in the compiled circuit")
        assert w_state > 1e-9, \
            ("SS7.2(2) FAILED for %s: the with-CZ and without-CZ forward passes agree "
             "to %.3e on every held-out item -- the CZ is inert" % (arm, w_state))

        # the control must reproduce the COMMITTED parent bit-for-bit
        w_parent = 0.0
        with torch.no_grad():
            for cs in data[:12]:
                w_parent = max(w_parent, float((c.story_values(cs)
                                                - parent.story_values(cs))
                                               .abs().max()))
        assert w_parent == 0.0, \
            ("EA-ctrl (cz=False) does not reproduce the committed "
             "QuantumStoryModel('A3') bit-for-bit (%.3e) -- doc 56 SS6.1's anchor "
             "cannot hold" % w_parent)

        cv = np.asarray(conc)
        res["T%d" % T] = {
            "n_heldout_items": n_items,
            "max_abs_intro_purity_vs_closed_form": w_pur,
            "max_abs_intro_purity_vs_closed_form_float64": w_pur64,
            "closed_form": "1 - (sin tA sin tB)^2 / 2, identical on both wires",
            "control_max_abs_intro_purity_minus_1": w_ctrl,
            "control_max_abs_intro_purity_minus_1_float64": w_ctrl64,
            "circuits_differ_structurally": True,
            "CZ_present_in_N_across_only": True,
            "max_abs_forward_difference_with_vs_without_CZ": w_state,
            "control_reproduces_committed_A3_bitwise": bool(w_parent == 0.0),
            "heldout_concurrence": {
                "min": float(cv.min()), "median": float(np.median(cv)),
                "max": float(cv.max()),
                "n_items_below_0.10": int((cv < CONCURRENCE_GATE).sum()),
                "fraction_below_0.10": float((cv < CONCURRENCE_GATE).mean())},
        }
    rep["across_validity"] = {
        "per_T": res,
        "why_this_replaces_fourier_closure": (
            "doc 56 SS4.5 proves the readable span is CZ-INVARIANT, so closure is "
            "VACUOUS here. This is doc 48 SS10.6's both-directions discipline "
            "finding a case where the programme's usual validity instrument fails, "
            "AND SAYING SO BEFORE THE RUN rather than passing a vacuous gate."),
        "three_assertions": (
            "(1) per-item COMPUTED intro purity against the closed form at 1e-6 -- "
            "not the vacuous '< 1'; (2) the with-CZ and without-CZ compilations are "
            "DIFFERENT CIRCUITS and their forward passes differ by more than 1e-9 on "
            "a held-out item; (3) THE FAILING DIRECTION -- the control's purity is "
            "exactly 1 on both wires to 1e-9 on EVERY item, without which (1) could "
            "be satisfied by a build that entangles everything."),
        "precision_note_THRESHOLDS_HELD_AS_REGISTERED": (
            "the simulator is complex64, so an exactly-product two-qubit state has "
            "runtime purity 1 + 1.19e-07 and doc 56 SS7.2's 1e-9 threshold is BELOW "
            "THE SIMULATOR'S RESOLUTION. Loosening the registered threshold to 1e-6 "
            "to make the gate pass would be changing a gate after seeing the "
            "platform, so the build does the other thing: the INTRO CIRCUIT is "
            "recomputed in complex128 from the model's own angle data through the "
            "same gate sequence, held to the runtime state at float32 tolerance so "
            "it is provably the same circuit, and doc 56's 1e-6 and 1e-9 thresholds "
            "are asserted against it AS WRITTEN. Both numbers are reported. The "
            "checkpointed purity traces remain the RUNTIME ones, taken from the same "
            "forward pass that produces the answer, exactly as SS15.1 requires."),
        "honest_note_doc56_SS4_4": (
            "MANY PAIRS ARE NEAR-PRODUCT and this is registered rather than "
            "discovered: over exp70's 80 SB casts the E-across concurrence runs "
            "0.0884-0.8244 with median 0.4019 and purity 0.6602-0.9961 with median "
            "0.9192. A median cast at purity 0.92 is WEAKLY ENTANGLED. E-across "
            "injects real but MODEST mechanism, and the size of any measured effect "
            "must be read against that, not against the maximum the gate could "
            "produce."),
    }
    return rep


def check_concurrence(fm, store, rep):
    """doc 56 SS7.3 -- THE DEGENERATE-ENTANGLEMENT GATE, new and BLOCKING."""
    ang, a1 = arm_angles("N-in-T1", fm)
    per_arm = {}
    for T in (1, 2):
        arm = "N-in-T%d" % T
        items, _m, _p = arm_items(arm, store)
        vals, worst_item = [], None
        for it in items:
            for r in it["referents"]:
                n = r["name"]
                c = IEB.ein_concurrence(ang[n], a1[n])
                vals.append(c)
                if worst_item is None or c < worst_item[1]:
                    worst_item = (it["id"], c, n)
        v = np.asarray(vals)
        assert v.min() >= CONCURRENCE_GATE, \
            ("SS7.3 FAILED for %s: item %s carries referent %s at concurrence %.4f, "
             "below the 0.10 gate. An arm that does not use the structure it claims "
             "to use is a NULL INSTRUMENT (doc 53 SS5)."
             % (arm, worst_item[0], worst_item[2], worst_item[1]))
        per_arm[arm] = {"n_referent_slots": int(v.size), "min": float(v.min()),
                        "median": float(np.median(v)), "max": float(v.max()),
                        "n_below_gate": 0, "gate": CONCURRENCE_GATE,
                        "worst_item": worst_item[0]}
    rep["degenerate_entanglement_gate"] = {
        "per_arm": per_arm,
        "closed_form": "concurrence(r) = |sin theta1(r)| * |sin(theta2(r)/2)|",
        "inventory_minimum": float(min(fm["concurrence_eps"].values())),
        "rule": ("doc 56 SS7.3: EVERY N-in item's referents must have concurrence "
                 ">= 0.10 under the eps-map, computed from the closed form on the "
                 "item's OWN two nouns. The committed diagnostic guarantees the "
                 "inventory supports this (minimum 0.1898, zero nouns below 0.05 and "
                 "zero below 0.10). THE GATE IS ASSERTED PER ITEM ANYWAY, because "
                 "doc 56 SS3's whole lesson is that 'the map should be fine' is not a "
                 "gate."),
        "deliberately_NOT_applied_to_N_across": (
            "doc 56 SS7.3, registered honestly alongside, and RATIFIED by "
            "orchestrator ruling 7: 4 of the 80 E-across casts fall below "
            "concurrence 0.10 and are NOT excluded. Applying the gate there would "
            "mean DROPPING CASTS FROM A FROZEN DATASET TO MAKE AN ARM LOOK "
            "ENTANGLED, which is the opposite of a validity gate. N-across's "
            "weak-pair fraction is REPORTED (see `across_validity`), and doc 56 "
            "SS6.8's per-item entanglement stratification exists so it can be read "
            "post-hoc without a rerun. The per-item PURITY gate is the working "
            "validity instrument for that arm."),
    }
    return rep


def check_copy(fm, verbs, store, rep):
    """doc 56 SS7.4, inherited from doc 55 SS7.4 and run for ALL FIVE families: with
    event 2 suppressed and q fixed to 1, the T = 2 forward pass must reproduce the
    T = 1 forward pass BIT-FOR-BIT. Only possible because referent state prep --
    including the CRY and the CZ -- happens EXACTLY ONCE, before event 1."""
    items2, _m, _p = arm_items("EA-ctrl-T2", store)
    worst = {}
    for arm in ("EA-ctrl-T2", "A0-W-T2", "P-eps-T2", "N-in-T2", "N-across-T2"):
        a, _ = arm_angles(arm, fm)
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
        assert w == 0.0, "SS7.4 COPY-VERIFICATION FAILED for %s: %.3e" % (arm, w)
    rep["copy_verification"] = {
        "max_abs_diff": worst,
        "exact_zero": bool(all(v == 0.0 for v in worst.values())),
        "note": ("doc 56 SS7.4: with event (3 - q) suppressed and the query fixed to "
                 "the surviving event, every T = 2 arm reproduces its T = 1 pipeline "
                 "bit-for-bit. For the ENTANGLED arms this carries extra weight: it "
                 "proves the CRY and the CZ sit in the INTRO and nowhere else, which "
                 "is what keeps SS7.1's closure arithmetic valid at T = 2 and what "
                 "doc 56 SS13 item 2 declines to give up by putting an entangler "
                 "inside an event block."),
    }
    return rep


def check_trace(fm, verbs, store, rep):
    """doc 56 SS15.1: the traced pass must be THE SAME forward pass -- and for the
    two ANCHOR families this doubles as the assertion that `apply_intro`'s
    re-implementation reproduces the committed parents' intros BIT-FOR-BIT."""
    worst, ranges = {}, {}
    for arm in ("EA-ctrl-T1", "EA-ctrl-T2", "A0-W-T1", "A0-W-T2", "P-eps-T2",
                "N-in-T1", "N-in-T2", "N-across-T1", "N-across-T2"):
        items, _m, _p = arm_items(arm, store)
        ang, _t1 = arm_angles(arm, fm)
        data = R69.compile_all([i for i in items if i["split"] == "test"][:6], ang)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        w, lo, hi = 0.0, 1.0, 0.0
        with torch.no_grad():
            for cs in data:
                v1 = m.story_values(cs)
                v2, tr = forward_traced(m, cs)
                w = max(w, float((v1 - v2).abs().max()))
                nw = qubits_per_referent(m) * cs.K
                assert set(tr) == ({"after_intro", "at_readout"}
                                   | {"after_event%d" % (k + 1)
                                      for k in range(cs.T)}), tr.keys()
                for stg, ps in tr.items():
                    assert len(ps) == nw, (stg, len(ps), nw)
                    for p in ps:
                        assert 0.49 <= p <= 1.0001, (stg, p)
                        lo, hi = min(lo, p), max(hi, p)
        worst[arm] = w
        ranges[arm] = [lo, hi]
        assert w == 0.0, \
            ("SS15.1 traced pass differs from story_values for %s: %.3e -- for the "
             "anchor families this ALSO means apply_intro does not reproduce the "
             "committed parent intro" % (arm, w))
    rep["trace_parity"] = {
        "max_abs_diff_vs_story_values": worst,
        "exact_zero": bool(all(v == 0.0 for v in worst.values())),
        "purity_range_per_arm": ranges,
        "stages": ["after_intro", "after_event1", "after_event2 (T=2)", "at_readout"],
        "quantity": "per-wire purity Tr rho^2 of the single-qubit reduced state",
        "note": ("doc 56 SS15.1, doc 55 SS18.3 verbatim plus exp70's re-pinning: THE "
                 "INTRO STAGE IS ADDED AS A FOURTH RECORDED POINT, because for N-in "
                 "and N-across the intro purity is no longer 1 and is the quantity "
                 "SS7.2 and SS7.3 gate on. The E-in per-wire intro purity is "
                 "1 - |sin t1|^2|sin(t2/2)|^2/2 and the E-across value is "
                 "1 - (sin tA sin tB)^2/2; both are closed forms, both are asserted, "
                 "neither is inferred. Descriptive; NO criterion reads the traces. "
                 "SS15.2's amendment: for the entangled arms the readout RESTORATION "
                 "TARGET is the INTRO purity, not 1."),
    }
    return rep


def check_shuffle(store, rep):
    """doc 46 SS10.2 / doc 56 SS5.2: the floor severs features <-> label and changes
    NOTHING else. `exp69_run.relabel_shuffle_q` is IMPORTED, not re-implemented --
    the floors must stay comparable across rungs."""
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
        "floored_arms": FLOORED,
        "note": ("doc 56 SS5.2: matched S-shuffle floors for the THREE NEW families "
                 "only, bearer designation reassigned PER PAIR and independently of "
                 "the features, so twins stay on opposite labels and V1 stays exactly "
                 "0.500. exp69's `relabel_shuffle_q` is IMPORTED, unchanged. The "
                 "ANCHOR families carry NO fresh floor and doc 56 SS5.3 states the "
                 "argument rather than assuming it: they are gated at |delta| <= "
                 "0.002 against exp69 arms that cleared their own matched floors by "
                 "8.02-19.08 SD, and a gate that pins an arm to within 0.002 of a "
                 "floor-clearing measurement is transitively stronger than "
                 "re-measuring the floor. IF EITHER ANCHOR FAILS, THAT ARGUMENT IS "
                 "VOID ALONG WITH THE RUN."),
    }
    return rep


def smoke(fm, verbs, store):
    t0 = time.time()
    rep = {"experiment": ("exp70 smoke / validity assertions (doc 56 SS7.1-SS7.4, "
                          "SS15.1)"),
           "spec": SPEC, "spec_commit": SPEC_COMMIT,
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "numpy": np.__version__}
    check_dataset(store, rep)
    check_features(fm, rep)
    check_intro_states(fm, verbs, store, rep)
    ang, _t1 = arm_angles("EA-ctrl-T2", fm)
    items2, _m, _p = arm_items("EA-ctrl-T2", store)
    data2 = {"test": R69.compile_all(
        [i for i in items2 if i["split"] == "test"], ang)}
    R69.check_word_order(fm, verbs, data2, rep)           # doc 56 SS7.4, inherited
    check_closures(fm, verbs, store, rep)
    check_across_validity(fm, verbs, store, rep)
    check_concurrence(fm, store, rep)
    check_copy(fm, verbs, store, rep)
    check_trace(fm, verbs, store, rep)
    check_shuffle(store, rep)

    tiny = {}
    for arm in ARMS:
        r = train(arm, 0, store, verbs, 2, fm, save=False)
        tiny[arm] = {"val_acc": r["val_acc"], "heldout_test_acc": r["heldout_test_acc"],
                     "wall_sec": r["wall_sec"], "n_params": r["n_params"],
                     "wires": r["wires"], "T": r["T"]}
        print("[smoke] %-22s 2ep val=%.3f test=%.3f  %.0fs  (%d params, %d wires, "
              "T=%d)" % (arm, r["val_acc"], r["heldout_test_acc"], r["wall_sec"],
                         r["n_params"], r["wires"], r["T"]), flush=True)
    rep["two_epoch_probe"] = tiny

    # the parameter-budget identity that makes N-in vs P-eps a clean contrast
    npar = {a: tiny[a]["n_params"] for a in BASE_ARMS}
    assert npar["N-in-T1"] == npar["P-eps-T1"] == npar["A0-W-T1"], \
        ("the entangled and product registers do not carry the SAME parameter "
         "budget -- doc 56 SS4.3's contrast would be confounded: %r" % npar)
    assert npar["N-across-T1"] == npar["EA-ctrl-T1"], \
        "N-across and EA-ctrl do not carry the same parameter budget"
    rep["parameter_budget_identity"] = {
        "per_arm": npar,
        "note": ("doc 56 SS4.3: N-in and P-eps carry the same two features under the "
                 "same angle map on the same wires with the SAME PARAMETER BUDGET; "
                 "the ONLY difference is the CRY. The entangling angle is DATA taken "
                 "from the referent's own AraVec coordinate, never a Parameter, so "
                 "the entangled arm adds ZERO trainable parameters. Asserted here "
                 "because a budget difference would confound the entire experiment."),
    }
    rep["assertions_passed"] = True
    rep["wall_time_sec"] = time.time() - t0
    json.dump(rep, open(SMOKE_OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[smoke] wrote %s -- ALL doc 56 SS7.1-SS7.4 / SS15.1 ASSERTIONS PASSED"
          % SMOKE_OUT)
    return rep


# ==================================================================== training
def train(arm, seed, store, verbs, max_epochs, fm, save=True, tag=""):
    """The exp60 / exp61 / exp65 / exp68 / exp69 loop, UNCHANGED -- Adam(lr 0.02),
    batch 8, 150 epochs, checkpoint selected on in-distribution val accuracy,
    held-out test read ONCE from the selected checkpoint.

    THE CALL ORDER (compile -> manual_seed -> build -> Adam -> Random(RNG_BASE +
    seed)) IS exp69's VERBATIM, because doc 56 SS6.1 expects the anchors to
    reproduce exp69 BIT-EXACTLY and any reordering of RNG consumption would break
    that. Nothing in exp70 consumes the generator that exp69 did not."""
    items, meta, path = arm_items(arm, store)
    data = compile_arm(arm, items, seed, fm)
    torch.manual_seed(seed)
    m = build_model(arm, verbs, seed, fm)
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
        if ep % 50 == 0 or ep == max_epochs:
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
        sha = hashlib.sha256(open(pt, "rb").read()).hexdigest()
    cfg = ARM_CFG[base_arm(arm)]
    return {"arm": arm, "base_arm": base_arm(arm), "family": cfg["family"],
            "is_floor": arm.startswith("S-shuffle"),
            "seed": seed, "part": cfg["part"], "T": cfg["T"], "encoding": cfg["enc"],
            "role": cfg["role"],
            "train_acc": best["train"], "val_acc": best["val"],
            "selected_epoch": best["epoch"], "heldout_test_acc": te,
            "val_minus_heldout": best["val"] - te,
            "gate_met": bool(best["train"] >= TRAIN_GATE and best["val"] >= VAL_GATE),
            "n_params": sum(p.numel() for p in m.parameters()),
            "wires": cfg["qpr"] * 2,
            "n_train": len(data["train"]), "n_val": len(data["val"]),
            "n_test": len(data["test"]),
            "max_epochs": max_epochs, "lr": LR, "batch": BATCH,
            "dataset": path, "dataset_sha256": meta["dataset_sha256"],
            "ckpt_path": pt, "ckpt_sha256": sha,
            "heldout_items": rows,
            "wall_sec": time.time() - t0}


# ============================================== doc 56 SS6.1 -- THE ANCHOR GATE
def anchor_gate(fm, store, verbs, seeds, arms=None):
    """doc 56 SS6.1, BLOCKING, AND IT RUNS FIRST.

    A0-W's and EA-ctrl's PER-SEED held-out accuracies must reproduce
    `results_exp69.json`'s R1-W / R2-W / R1-A / R2-A per-seed values to
    |delta| <= 0.002, and the arm means to |delta| <= 0.002 of doc 56 SS6.1's
    registered 0.9820 / 0.7293 / 0.9227 / 0.9187.

    Both runs are deterministic given their seeds, so THE HONEST EXPECTATION IS
    BIT-EXACT; the tolerance absorbs a documented float/BLAS platform difference
    only, and any non-zero delta is reported with its cause identified.

    IF THIS FAILS, the exp70 pipeline has drifted from exp69's, doc 56 SS5.3's
    floor-omission argument is VOID, and NOTHING ELSE IN THE RUN IS READ.
    Rule 3: this writes its OWN results file and no exp69 artefact is touched."""
    rows = []
    for arm in (arms or ANCHOR_ARMS):
        for s in seeds:
            ckj = os.path.join(CKPT_DIR, "%s_s%d.json" % (arm, s))
            if os.path.exists(ckj):
                r = json.load(open(ckj, encoding="utf-8"))
                print("[anchor] %s s%d already done: %.6f"
                      % (arm, s, r["heldout_test_acc"]), flush=True)
            else:
                r = train(arm, s, store, verbs, MAX_EPOCHS, fm, save=True)
                r["tag"] = ""
                json.dump(r, open(ckj, "w", encoding="utf-8"))
                print("[anchor] %-11s s%d held-out=%.6f  (%.0fs)"
                      % (arm, s, r["heldout_test_acc"], r["wall_sec"]), flush=True)
            rows.append(r)
    return rows


def anchor_report(rows, store):
    ref = json.load(open(EXP69_RESULTS, encoding="utf-8"))["arms"]
    _items, meta, path = store
    per_arm, ok_all, exact_all = {}, True, True
    for arm in ANCHOR_ARMS:
        rr = sorted([r for r in rows if r["arm"] == arm], key=lambda x: x["seed"])
        src = ANCHOR_OF[arm]
        got = [r["heldout_test_acc"] for r in rr]
        want = ref[src]["per_seed"]
        assert [r["seed"] for r in rr] == ref[src]["seeds"], \
            "seed sets differ between exp70 %s and exp69 %s" % (arm, src)
        d = [g - w for g, w in zip(got, want)]
        mean = float(np.mean(got))
        ok_seed = all(abs(x) <= ANCHOR_TOLERANCE for x in d)
        ok_mean = abs(mean - ANCHOR_MEAN_DOC56[arm]) <= ANCHOR_TOLERANCE
        exact = all(x == 0.0 for x in d)
        ok_all = ok_all and ok_seed and ok_mean
        exact_all = exact_all and exact
        per_arm[arm] = {
            "exp69_arm": src, "exp69_per_seed": want, "exp69_mean": ref[src]["mean"],
            "exp69_sd": ref[src]["sd"],
            "exp70_per_seed": got, "exp70_mean": mean,
            "exp70_sd": float(np.std(got, ddof=1)),
            "per_seed_delta": d,
            "max_abs_per_seed_delta": float(max(abs(x) for x in d)),
            "mean_delta_vs_exp69": mean - ref[src]["mean"],
            "registered_mean_doc56_SS6_1": ANCHOR_MEAN_DOC56[arm],
            "mean_delta_vs_registered": mean - ANCHOR_MEAN_DOC56[arm],
            "bit_exact": bool(exact),
            "per_seed_within_tolerance": bool(ok_seed),
            "mean_within_tolerance": bool(ok_mean),
            "replicates": bool(ok_seed and ok_mean),
            "val_per_seed": [r["val_acc"] for r in rr],
            "selected_epoch_per_seed": [r["selected_epoch"] for r in rr],
            "n_params": rr[0]["n_params"], "wires": rr[0]["wires"],
            "wall_sec_total": float(sum(r["wall_sec"] for r in rr)),
        }
    return {
        "component": "exp70 anchors -- doc 56 SS6.1, the BLOCKING replication gate",
        "spec": SPEC, "spec_commit": SPEC_COMMIT,
        "rule3_note": ("the anchors write their OWN results file (doc 56 SS16.1, "
                       "'separate by Rule 3'). No exp68 or exp69 artefact is "
                       "touched, read for writing, or overwritten."),
        "dataset": path, "dataset_sha256": meta["dataset_sha256"],
        "exp69_source": EXP69_RESULTS,
        "per_arm": per_arm,
        "tolerance": ANCHOR_TOLERANCE,
        "anchors_replicate_exp69": bool(ok_all),
        "all_bit_exact": bool(exact_all),
        "merge_note": (
            "doc 56 SS5.2, RATIFIED by orchestrator ruling 2: A0-A and EA-ctrl are "
            "THE SAME RUNS. E-across's map-matched product control IS exp69's R-A "
            "arm -- same map, same wires, no CZ -- so registering them separately "
            "would be running the same arm twice under two names. Codepath purity "
            "still demands an IN-RUN control, which is what this arm is, and the "
            "TIGHTER of the two gates (|delta| <= 0.002 against exp69) governs."),
        "honest_expectation": (
            "BIT-EXACT. Both runs are deterministic given their seeds, and exp70 "
            "constructs the COMMITTED parent classes verbatim with exp69's exact "
            "call order; the 0.002 tolerance exists only to absorb a documented "
            "float/BLAS platform difference and any non-zero delta is reported with "
            "its cause identified (doc 56 SS6.1)."),
        "consequence_if_failed": (
            "doc 56 SS11 row 1: THE RUN STOPS. The exp70 pipeline has drifted from "
            "exp69's, doc 56 SS5.3's floor-omission argument is VOID along with the "
            "run, and NOTHING ELSE IS READ FROM ANY ARM. Hunt the build; interpret "
            "nothing."),
    }


# ======================================================================== main
def main():
    global DATA_A, CKPT_DIR, SMOKE_OUT, VSUITE, CONFIRM_MODE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--tag", default="", help="checkpoint prefix; use for doc 56 "
                                              "SS5.1.1's confirm draw")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--anchor-gate", action="store_true")
    ap.add_argument("--data", default=DATA_A,
                    help="Part A dataset. Default is exp69's FROZEN draw; pass the "
                         "seed+1000 re-draw for doc 56 SS5.1.1's CONFIRM DRAW.")
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--smoke-out", default=SMOKE_OUT)
    ap.add_argument("--vsuite", default=VSUITE,
                    help="the V-suite results for THIS draw. On the confirm draw it "
                         "is RE-RUN, never inherited (doc 56 SS5.1.1).")
    a = ap.parse_args()

    # Every default above is the first draw's value, so omitting the flags
    # reproduces the first draw bit-identically. Only an explicit --data switches
    # the run into confirm mode.
    DATA_A, CKPT_DIR, SMOKE_OUT, VSUITE = (a.data, a.ckpt_dir, a.smoke_out, a.vsuite)
    CONFIRM_MODE = (os.path.abspath(a.data) != os.path.abspath(FIRST_DRAW_DATA))
    if CONFIRM_MODE:
        print("[exp70] CONFIRM DRAW MODE -- data=%s ckpt=%s vsuite=%s"
              % (DATA_A, CKPT_DIR, VSUITE), flush=True)

    if not os.path.exists(PREDICTIONS):
        raise RuntimeError("%s missing -- doc 56 SS17.3's ordering rule: the "
                           "predictions are frozen BEFORE the run" % PREDICTIONS)

    store = load_part()
    verbs = sorted(store[1]["verb_inventory"].keys())
    fm = feature_maps()
    os.makedirs(CKPT_DIR, exist_ok=True)

    if a.smoke:
        smoke(fm, verbs, store)
        return 0

    if a.anchor_gate:
        if CONFIRM_MODE:
            raise RuntimeError(
                "doc 56 SS6.1's anchor gate is INAPPLICABLE to a fresh draw: it "
                "compares per-seed accuracies against exp69's measurements on "
                "EXP69'S OWN ITEMS, and these are different items. Running it here "
                "would manufacture a comparison, not check one. On the confirm draw "
                "the anchor arms are ordinary baseline arms -- run them with the "
                "other 14.")
        seeds = ([s for s in range(a.seeds) if s % a.nworkers == a.worker]
                 if a.nworkers > 1 else list(range(a.seeds)))
        arms = [x for x in a.arms.split(",") if x in ANCHOR_ARMS] or ANCHOR_ARMS
        rows = anchor_gate(fm, store, verbs, seeds, arms=arms)
        if a.nworkers > 1 or set(arms) != set(ANCHOR_ARMS) \
                or len(seeds) != N_SEEDS:
            print("[anchor w%d] done (%d runs); re-run with --nworkers 1 to write %s"
                  % (a.worker, len(rows), ANCHOR_OUT))
            return 0
        rep = anchor_report(rows, store)
        json.dump(rep, open(ANCHOR_OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n[ANCHOR GATE] doc 56 SS6.1, |delta| <= %.3f, honest expectation "
              "BIT-EXACT" % ANCHOR_TOLERANCE)
        for arm in ANCHOR_ARMS:
            p = rep["per_arm"][arm]
            print("  %-11s (exp69 %s)" % (arm, p["exp69_arm"]))
            print("      exp69  %s  mean %.6f"
                  % ([round(x, 6) for x in p["exp69_per_seed"]], p["exp69_mean"]))
            print("      exp70  %s  mean %.6f"
                  % ([round(x, 6) for x in p["exp70_per_seed"]], p["exp70_mean"]))
            print("      delta  %s  max |d| %.2e  bit-exact=%s  -> %s"
                  % ([round(x, 6) for x in p["per_seed_delta"]],
                     p["max_abs_per_seed_delta"], p["bit_exact"],
                     "PASS" if p["replicates"] else "FAIL"))
        print("  anchors_replicate_exp69 = %s  ->  %s"
              % (rep["anchors_replicate_exp69"],
                 "PROCEED" if rep["anchors_replicate_exp69"]
                 else "THE RUN STOPS (doc 56 SS11 row 1)"))
        print("[ANCHOR GATE] wrote %s" % ANCHOR_OUT)
        return 0 if rep["anchors_replicate_exp69"] else 1

    if CONFIRM_MODE:
        # doc 56 SS6.1's gate is NOT SKIPPED, it is INAPPLICABLE, and the difference
        # matters. The gate compares per-seed accuracies against exp69's measurements
        # ON EXP69'S OWN ITEMS; these are different items, so there is nothing to
        # compare against and a "pass" here would be meaningless. What the gate
        # certified -- that exp70's codepath has not drifted from exp69's -- was
        # established on the FIRST draw and is a property of the CODE, which this run
        # shares byte-for-byte. It is asserted here rather than assumed.
        if not os.path.exists(ANCHOR_OUT):
            raise RuntimeError(
                "%s missing -- the confirm draw may only run AFTER the first draw's "
                "anchor gate has certified the codepath (doc 56 SS6.1)" % ANCHOR_OUT)
        anc = json.load(open(ANCHOR_OUT, encoding="utf-8"))
        if not anc.get("anchors_replicate_exp69"):
            raise RuntimeError(
                "the FIRST draw's anchor gate FAILED -- doc 56 SS11 row 1: nothing is "
                "interpreted from any arm, and a confirm draw would confirm nothing")
        print("[exp70] first-draw anchor gate certified the codepath (bit-exact=%s); "
              "on THIS draw the anchor arms are UN-FLOORED DESCRIPTIVE BASELINES and "
              "no boolean reads them" % anc.get("all_bit_exact"), flush=True)
    else:
        if not os.path.exists(ANCHOR_OUT):
            raise RuntimeError(
                "%s missing -- doc 56 SS17.3 item 6 / SS6.1: THE ANCHORS RUN FIRST "
                "and block everything. Run `python exp70_run.py --anchor-gate`."
                % ANCHOR_OUT)
        anc = json.load(open(ANCHOR_OUT, encoding="utf-8"))
        if not anc.get("anchors_replicate_exp69"):
            raise RuntimeError(
                "the anchor gate FAILED -- doc 56 SS11 row 1: the run stops, the "
                "build is hunted, and nothing is interpreted from any arm")

    arms = [x for x in a.arms.split(",") if x]
    jobs = [(arm, s) for arm in arms for s in range(a.seeds)]
    mine = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    print("[exp70 w%d/%d] arms=%d -> %d of %d runs"
          % (a.worker, a.nworkers, len(arms), len(mine), len(jobs)), flush=True)
    for arm, s in mine:
        ckj = os.path.join(CKPT_DIR, "%s%s_s%d.json" % (a.tag, arm, s))
        if os.path.exists(ckj):
            print("[ckpt] %s%s_s%d already done" % (a.tag, arm, s), flush=True)
            continue
        r = train(arm, s, store, verbs, MAX_EPOCHS, fm, save=True, tag=a.tag)
        r["tag"] = a.tag
        json.dump(r, open(ckj, "w", encoding="utf-8"))
        print("[exp70] %-22s s%d  train=%.3f val=%.3f HELD-OUT=%.4f  (%.0fs)"
              % (arm, s, r["train_acc"], r["val_acc"], r["heldout_test_acc"],
                 r["wall_sec"]), flush=True)
    print("[exp70 w%d] done" % a.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
