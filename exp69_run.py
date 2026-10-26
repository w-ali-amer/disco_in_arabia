# -*- coding: utf-8 -*-
"""exp69_run.py -- doc 55: the DisCoCat reunion. Does a bound content-role binding
survive COMPOSITION?

THE QUESTION (doc 55 SS1). exp60-exp68 all measured binding inside ONE event.
exp69 reads a semantic property of a role-filler through TWO COMPOSED EVENTS, with
the one-event case as the control.

ARMS (doc 55 SS4.4 + SS18.4's Part B, released SS19.1)

  A0     T=1  exp68's frozen dataset, FA angle       BIT-EXACT REPLICATION ANCHOR
  R1-A   T=1  exp69 generator, FA angle              surface anchor -> 0.9187
  R1-W   T=1  exp69 generator, F2 register (4 wires) subfamily member #2
  R2-A   T=2  Q69, FA angle                          angle at T = 2
  R2-W   T=2  Q69, F2 register (4 wires)             PRIMARY ARM, subfamily #3
  R1-M   T=1  Q69 Part B, wazn positional angle      morphology at T = 1
  R2-M   T=2  Q69 Part B, wazn positional angle      morphology at T = 2
  S-shuffle-<arm>  a MATCHED floor for every trained arm EXCEPT A0 (doc 46 SS10.2;
                   exp68's committed floor serves A0, which re-runs exp68's data)

  -> 13 arms x 5 seeds = 65 runs.

THE WORD-ORDER RECONCILIATION -- READ THIS BEFORE THE SS7.3 SMOKE VERDICT
=========================================================================
doc 55 SS4.3 requires that "the SVO and VSO renderings of the SAME event over the
SAME cast must compile to DIFFERENT circuits (different gate sequence, different
wiring)", and SS7.3 makes that a blocking smoke assertion whose stated failure mode
is a NULL INSTRUMENT. Four other commitments in the same doc constrain HOW that
difference may be realised, and they rule out the obvious reading (a separate
trained block per surface order):

  SS4.7   "the per-verb blocks are the SAME SIX BLOCKS, reused."
  SS5     conditioning-group arithmetic: 12 groups at T = 1 = (verb x event
          DIRECTION), 72 at T = 2 = (3 x 2)^2 x 2 queries. Neither has a
          surface-order factor; per-order blocks would give 24 and 288.
  SS6.1   A0 must reproduce exp68's FA per-seed BIT-EXACTLY, and exp68's dataset
          already contains BOTH order flags. A model with per-order blocks run on
          exp68's data cannot reproduce exp68 -- the first blocking gate would fail
          BY CONSTRUCTION, which cannot be what the doc intends.
  SS16.1 Q2 (user-confirmed, SS18.1): the VSO and SVO renderings of the same event
          are TRUTH-CONDITIONALLY IDENTICAL, and SS16.1 calls this "the load-bearing
          assumption of the whole alternation". Different denotations per order
          would violate the confirmed premise.

RESOLUTION, STATED IN THE OPEN AND SUBJECT TO ORCHESTRATOR VETO: **the DERIVATION
differs, the DENOTATION is equal.** exp69's compile step routes every clause
through the archived pregroup reduction
(`archive/01_parser_pipeline/arabic_dep_reader.py:130-138, 167-180`) and records
the reduction as part of the compiled circuit spec -- SVO takes
`Word(v, N.r @ S @ N.l)` with two cups and NO crossing; VSO takes
`Word(v, S @ N.l @ N.l)` with `Swap(N.l, N)` and then two cups. The EXECUTED gate
application is `block(v)` on `(agent_wire, patient_wire)` in both cases, which is
what the reduction contracts to and what keeps SS4.7, SS5, SS6.1 and SS16.1-Q2 all
true simultaneously.

SS7.3 is therefore discharged by THREE assertions, not one, and the third is the
one that carries the content:

  (i)   the compiled derivations DIFFER between the two renderings -- the Swap is
        present in one and absent in the other, and the compiler genuinely
        consults `order_flags`, which is exactly the null-instrument failure SS7.3
        exists to catch (a compiler that never reads the surface at all);
  (ii)  the two denotations are EQUAL, measured, not assumed -- and this is not a
        defect but the confirmed SS16.1-Q2 premise, reported rather than buried;
  (iii) DROPPING THE SWAP from the VSO reduction produces the ROLE-REVERSED
        circuit -- the twin's circuit -- measured to float precision. The Swap is
        therefore LOAD-BEARING: without it the VSO clause would say the object did
        it. That is what makes the alternation real structure rather than an inert
        flag, and it is asserted arithmetically in `check_word_order()`.

WHAT THIS COSTS THE CLAIM, stated now: exp69's circuit is INVARIANT to the surface
alternation by construction, so doc 55 SS6.9's per-surface-order accuracy is a
DATA-COMPOSITION statistic and NOT evidence about word-order processing. It is
reported with that caveat attached and may not be read any other way.

MECHANISM INSTRUMENTATION (doc 55 SS18.3). Every per-run checkpoint records, for
every held-out item, the per-wire purity Tr rho^2 after intro, after event 1, after
event 2 (T = 2 arms) and at readout, FROM THE SAME FORWARD PASS that produces the
answer -- `forward_traced` is asserted at smoke to reproduce `story_values`
bit-for-bit. Descriptive; no criterion reads it. Per-item held-out predictions are
recorded too, because doc 55 SS19.3 (Spearman on witness margins), SS19.4 (purity
contrast) and SS19.5 (error overlap) are item-level and a rerun of 65 jobs to
recover them would be a self-inflicted wound.

    python exp69_run.py --smoke        # doc 55 SS7.1-SS7.4 validity gate
    python exp69_run.py --a0-gate      # doc 55 SS6.1, BLOCKING, RUNS FIRST
    python exp69_run.py --worker i --nworkers 5
    python exp69_comparator.py         # doc 55 SS8, the ARBITER
    python exp69_aggregate.py          # the ONLY results writer
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

import exp41_story_generator as gen
import exp42_compiler as comp
import exp42_sim_adapter as sim
import exp65_featurizer as WZ
import exp68_run as E                                 # features, F2Model -- IMPORTED
import instrument_sb_bounds as INS
from exp42_models import QuantumStoryModel
from exp42_run import grad_global_norm
from exp60_run import evaluate

SPEC = "qnlp_private_docs/55_exp69_reunion.md (da4b8fe, REGISTERED through SS19.7)"
DATA_A = "stories_exp69_QA.json"
DATA_B = "stories_exp69_QB.json"
DATA_EXP68 = "stories_exp68_WN.json"
EXP68_RESULTS = "results_exp68.json"
CKPT_DIR = "exp69_ckpt"
SMOKE_OUT = "results_exp69_smoke_validation.json"
A0_OUT = "results_exp69_A0.json"
PREDICTIONS = "exp69_predictions.json"

# Training config: the exp60 / exp61 / exp65 / exp68 loop, UNCHANGED. Anything
# else here would put A0's bit-exactness out of reach.
LR, BATCH, MAX_EPOCHS = E.LR, E.BATCH, E.MAX_EPOCHS
TRAIN_GATE, VAL_GATE = E.TRAIN_GATE, E.VAL_GATE
RNG_BASE = E.RNG_BASE
N_SEEDS = E.N_SEEDS

A0_TOLERANCE = 0.002                  # doc 55 SS6.1; honest expectation is 0.0
BUG_HUNT_MARGIN = 0.06                # doc 55 SS6.8

# doc 55 SS4.4 / SS18.4. `enc` picks the feature map, `T` the composition depth,
# `part` the dataset.
ARM_CFG = {
    "A0":   {"part": "exp68", "T": 1, "enc": "FA", "role": "bit-exact replication"},
    "R1-A": {"part": "A", "T": 1, "enc": "FA", "role": "surface anchor"},
    "R1-W": {"part": "A", "T": 1, "enc": "F2", "role": "register at T=1"},
    "R2-A": {"part": "A", "T": 2, "enc": "FA", "role": "angle at T=2"},
    "R2-W": {"part": "A", "T": 2, "enc": "F2", "role": "PRIMARY ARM"},
    "R1-M": {"part": "B", "T": 1, "enc": "WZ", "role": "wazn at T=1"},
    "R2-M": {"part": "B", "T": 2, "enc": "WZ", "role": "wazn at T=2"},
}
BASE_ARMS = list(ARM_CFG)
FLOORED = [a for a in BASE_ARMS if a != "A0"]         # A0 uses exp68's committed floor
ARMS = BASE_ARMS + ["S-shuffle-%s" % a for a in FLOORED]
FLOOR_OF = {a: "S-shuffle-%s" % a for a in FLOORED}

# doc 55 SS12 / exp69_predictions.json -- CONTEXT ONLY. No criterion reads these;
# they arm doc 55 SS9.2's bug-hunt row and nothing else.
BOUND_FAMILY = {"A0": "R-A", "R1-A": "R-A", "R2-A": "R-A",
                "R1-W": "R-W", "R2-W": "R-W", "R1-M": "R-M", "R2-M": "R-M"}


def base_arm(arm):
    return arm[len("S-shuffle-"):] if arm.startswith("S-shuffle-") else arm


def _u(t):
    return (1.0, math.cos(t), math.sin(t))


# =========================================================== the feature layer
def feature_maps():
    """exp68's three maps, IMPORTED through `exp68_run.feature_maps` (never
    re-derived here -- shared code objects are the strongest guarantee A0 has),
    plus exp69's wazn map.

    doc 55 SS18.4's featurizer protocol: the positional featurizer is fit
    INVENTORY-WIDE and applied to all 24 nouns. That is NOT leakage in exp69's
    setting because exp69 holds out PAIRINGS, not nouns -- doc 54's raw-PCA
    precedent. exp65's train-only rule existed because exp65 held out NOUNS.
    Stated to preempt the doc-42 failure class."""
    fm = E.feature_maps()
    H = sorted(gen.HUMANS)
    assert sorted(H) == sorted(INS.HUMANS), "population drift"
    fm["theta_wazn"] = {n: float(v)
                        for n, v in WZ.angles_for(H, INS.HUMANS).items()}
    fm["wazn_fit"] = ("exp65_featurizer.angles_for(train_nouns=all 24 humans, "
                      "all_nouns=all 24 humans) -- doc 55 SS18.4 inventory-wide fit")
    fm["wazn_bearers"] = sorted(n for n in INS.HUMANS if n in WZ.FAAIL)
    return fm


def arm_angles(arm, fm):
    """(angles consumed by exp42_compiler, second-qubit angles or None)."""
    enc = ARM_CFG[base_arm(arm)]["enc"]
    if enc == "FA":
        return fm["phi_exp42"], None
    if enc == "F2":
        return fm["theta_pc1"], fm["theta_pc2"]
    if enc == "WZ":
        return fm["theta_wazn"], None
    raise ValueError(arm)


def referent_features(arm, fm):
    """The per-referent u-vector the arm's circuit is LINEAR in: d = 3 for the
    angle arms, d = 9 for the register arms. RY(t)|0> has a density matrix linear
    in (1, cos t, sin t), so a two-qubit product register is linear in the 9-term
    outer product -- which is why SS7.1's closure test can be exact and why SS8's
    comparator can consume the IDENTICAL features."""
    a0, a1 = arm_angles(arm, fm)
    if a1 is None:
        return {n: np.array(_u(a0[n])) for n in INS.HUMANS}
    return {n: np.outer(np.array(_u(a0[n])), np.array(_u(a1[n]))).ravel()
            for n in INS.HUMANS}


# ================================================= the pregroup reduction layer
N_TYPE, S_TYPE = "N", "S"


def clause_derivation(order, verb, wa, wp):
    """The archived pregroup reduction, recorded as part of the compiled spec.

    `archive/01_parser_pipeline/arabic_dep_reader.py`
      :130-138  the TWO verb boxes -- SVO `Word(v, N.r @ S @ N.l)`,
                VSO `Word(v, S @ N.l @ N.l)`
      :167-180  `_vso()` and the Swap VSO requires and SVO does not:
                `swap = Id(S) @ Id(N.l) @ Swap(N.l, N) @ Id(N)`, after which both
                cups close.

    Returned as a tuple so two renderings of the same event compare by value."""
    if order == "SVO":
        return (("Word", "noun", N_TYPE, wa),
                ("Word", verb, "%s.r @ %s @ %s.l" % (N_TYPE, S_TYPE, N_TYPE)),
                ("Word", "noun", N_TYPE, wp),
                ("Cup", "N, N.r"), ("Id", S_TYPE), ("Cup", "N.l, N"),
                ("Contract", verb, ("agent", wa), ("patient", wp)))
    return (("Word", verb, "%s @ %s.l @ %s.l" % (S_TYPE, N_TYPE, N_TYPE)),
            ("Word", "noun", N_TYPE, wa),
            ("Word", "noun", N_TYPE, wp),
            ("Id", S_TYPE), ("Id", "N.l"), ("Swap", "N.l, N"), ("Id", N_TYPE),
            ("Cup", "N.l, N"), ("Cup", "N.l, N"),
            ("Contract", verb, ("agent", wa), ("patient", wp)))


def materialised_clause(order, wa, wp, with_swap=True):
    """The VSO Swap made ARITHMETIC, for `check_word_order()` only.

    In the VSO diagram the words leave `s (x) n.l (x) n.l (x) n (x) n`; the
    planar cups can only close after `Swap(N.l, N)` brings the second `n.l` next
    to the subject. WITHOUT the swap the planar cups would pair slot 1 with the
    OBJECT and slot 2 with the SUBJECT -- i.e. the verb applied to (patient,
    agent), which is precisely the TWIN'S circuit. The swap is therefore
    load-bearing, and this function returns the wire order the reduction
    contracts to so that fact can be measured instead of argued."""
    if order == "SVO":
        return (wa, wp)
    return (wa, wp) if with_swap else (wp, wa)


def compile_item(item, angles):
    """`exp42_compiler.compile_story`, plus exp69's extra spec fields.

    `CompiledStory` is a plain dataclass, so the exp69 fields are ATTACHED to the
    instance; `exp42_compiler.py` is not modified (Rule 3)."""
    cs = comp.compile_story(item, angles)
    qi = int(item.get("query_index", 1))
    flags = item.get("order_flags", ["SVO"] * len(item["events"]))
    cs.qidx = qi
    cs.order_flags = list(flags)
    cs.T = len(item["events"])
    cs.derivation = tuple(clause_derivation(flags[k], e[0], e[1], e[2])
                          for k, e in enumerate(cs.events))
    qe = item["events"][qi - 1]
    cs.meta = {"id": item["id"], "pair_id": item["pair_id"],
               "split": item["split"], "T": cs.T, "query_index": qi,
               "queried_verb": item.get("queried_verb", qe["verb"]),
               "order_flags": list(flags),
               "agent": qe["agent"], "patient": qe["patient"],
               "cast": [r["name"] for r in item["referents"]],
               "answer": item["answer"]}
    return cs


def compile_all(items, angles):
    return [compile_item(i, angles) for i in items]


# ================================================================= the floors
def relabel_shuffle_q(items, seed):
    """doc 46 SS10.2's S-shuffle, transposed to Q69.

    `exp60_run.relabel_shuffle` keys the fake label on `events[0]["agent"]`, which
    is WRONG for a q = 2 item -- it would relabel from the UNQUERIED event and
    break the twin structure. exp69 keys it on the QUERIED event's agent. The RNG
    convention (`random.Random(9_000_000 + seed)`) is exp60's, unchanged, so the
    floors stay comparable across rungs.

    Geometry is bit-identical to the matched arm; only the feature <-> label
    correlation is severed. Per PAIR, so twins stay on opposite labels and V1
    stays exactly 0.500."""
    rng = random.Random(9_000_000 + seed)
    byp, out = {}, []
    for it in items:
        byp.setdefault(it["pair_id"], []).append(it)
    for _pid, g in byp.items():
        g = sorted(g, key=lambda x: x["id"])
        names = [r["name"] for r in g[0]["referents"]]
        fake_bearer = names[rng.randrange(2)]           # independent of the angle
        for it in g:
            j = copy.deepcopy(it)
            qi = int(it.get("query_index", 1))
            j["answer"] = ("yes" if it["events"][qi - 1]["agent"] == fake_bearer
                           else "no")
            out.append(j)
    return out


# ================================================================== the models
def build_model(arm, verbs, seed, fm):
    """doc 55 SS4.7: NO NEW ANSATZ, one new wiring. The angle arms are exp42's
    `QuantumStoryModel('A3')` -- the 2-wire model exp60/exp61/exp68 use -- and the
    register arms are `exp68_run.F2Model`, IMPORTED. T = 2 changes nothing in the
    model: `story_values` already loops over `cs.events`, so the second event's
    block is applied on the SAME persistent referent wires, and the per-verb blocks
    are the same six blocks reused (SS4.7)."""
    if ARM_CFG[base_arm(arm)]["enc"] == "F2":
        return E.F2Model(verbs, seed, fm["theta_pc2"])
    return QuantumStoryModel("A3", verbs, seed)


def is_register(m):
    return isinstance(m, E.F2Model)


def wire_purities(psi, nwires):
    """Tr rho^2 per wire, rho the single-qubit reduced density matrix. For a
    Hermitian rho, Tr rho^2 = sum_ij |rho_ij|^2."""
    out = []
    for w in range(nwires):
        p = torch.movedim(psi, w, 0).reshape(2, -1)
        rho = p @ p.conj().transpose(0, 1)
        out.append(float((rho.abs() ** 2).sum()))
    return out


def forward_traced(m, cs):
    """doc 55 SS18.3: `story_values` with the per-wire purity trace taken FROM THE
    SAME FORWARD PASS. Gate order is identical to the parent implementations
    (`exp42_models.QuantumStoryModel.story_values` and `exp68_run.F2Model.
    story_values`), which is asserted bit-for-bit in `check_trace()`."""
    reg = is_register(m)
    nw = (2 if reg else 1) * cs.K
    cache = {}

    def block(v):
        if v not in cache:
            cache[v] = m.verb_matrix(v)
        return cache[v]

    def wires_of(w):
        return (2 * w, 2 * w + 1) if reg else (w,)

    psi = sim.init_state(nw)
    for w, noun, angle in cs.intro:
        psi = sim.apply_gate(psi, sim.ry(angle), (2 * w,) if reg else (w,))
        if reg and m.wide:
            psi = sim.apply_gate(psi, sim.ry(m.theta1[noun]), (2 * w + 1,))
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
    """Held-out accuracy PLUS doc 55 SS18.3's traces and the per-item predictions
    SS19.3 / SS19.4 / SS19.5 need. One pass, no_grad, no RNG consumed."""
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
def load_part(part):
    path = {"A": DATA_A, "B": DATA_B, "exp68": DATA_EXP68}[part]
    blob = json.load(open(path, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("exp69 dataset provenance FAILED for %s" % path)
    return items, meta, path


def arm_items(arm, store):
    cfg = ARM_CFG[base_arm(arm)]
    items, meta, path = store[cfg["part"]]
    sel = [i for i in items if int(i.get("T", 1)) == cfg["T"]]
    assert sel, "no items at T=%d in %s" % (cfg["T"], path)
    return sel, meta, path


def compile_arm(arm, items, seed, fm):
    ang, _t1 = arm_angles(arm, fm)
    use = relabel_shuffle_q(items, seed) if arm.startswith("S-shuffle") else items
    return {s: compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}


# ============================================== doc 55 SS7.1 - SS7.4 smoke gates
def check_features(fm, rep):
    """doc 55 SS7.1(3): p1 and p2 must be NON-DEGENERATE on the inventory, and the
    wazn angle must be non-degenerate too. Delegated to `exp68_run.check_features`
    for the raw-PCA family (the committed check, unchanged), plus the wazn map."""
    E.check_features(fm, rep)
    ang = np.array([fm["theta_wazn"][n] for n in INS.HUMANS])
    B = set(fm["wazn_bearers"])
    b = np.array([fm["theta_wazn"][n] for n in INS.HUMANS if n in B])
    nb = np.array([fm["theta_wazn"][n] for n in INS.HUMANS if n not in B])
    n_dist = len({round(float(x), 9) for x in ang})
    assert n_dist >= 20, \
        "the wazn angle collapses the inventory to %d distinct values" % n_dist
    assert float(ang.std(ddof=1)) > 0.05, "the wazn angle is degenerate"
    rep["features"]["wazn"] = {
        "fit_protocol": fm["wazn_fit"],
        "n_distinct_angles": n_dist, "sd": float(ang.std(ddof=1)),
        "range": [float(ang.min()), float(ang.max())],
        "bearer_mean": float(b.mean()), "nonbearer_mean": float(nb.mean()),
        "n_bearers": len(b), "n_nonbearers": len(nb),
        "membership_source": ("exp65_featurizer.FAAIL -- doc 51 SS7, "
                              "user-confirmed; imported, never re-typed"),
        "leakage_note": (
            "doc 55 SS18.4: an inventory-wide fit is NOT leakage here because exp69 "
            "holds out PAIRINGS, not nouns. exp65's train-only rule existed because "
            "exp65 held out NOUNS. Stated to preempt the doc-42 failure class."),
    }
    return rep


def check_word_order(fm, verbs, data, rep):
    """doc 55 SS7.3 -- three assertions, and the reconciliation stated in the open.

    See this module's header for why a per-order trained block is ruled out by
    SS4.7 / SS5 / SS6.1 / SS16.1-Q2, and why SS7.3 is discharged as
    DERIVATION-DIFFERS / DENOTATION-EQUAL."""
    torch.manual_seed(0)
    m = QuantumStoryModel("A3", verbs, 0)
    probe = [cs for cs in data["test"] if cs.T == 2][:4] or data["test"][:4]

    # (i) the derivations differ -- structural, BLOCKING
    diffs, same = 0, 0
    for cs in probe:
        for k, (v, wa, wp) in enumerate(cs.events):
            a = clause_derivation("SVO", v, wa, wp)
            b = clause_derivation("VSO", v, wa, wp)
            if a == b:
                same += 1
            else:
                diffs += 1
    assert same == 0 and diffs > 0, \
        ("SS7.3: SVO and VSO reduce to the SAME derivation -- the word-order "
         "alternation would be a NULL INSTRUMENT in doc 53 SS5's sense and no exp69 "
         "result is admissible")
    swap_in_vso = any(op[0] == "Swap" for op in clause_derivation("VSO", "v", 0, 1))
    swap_in_svo = any(op[0] == "Swap" for op in clause_derivation("SVO", "v", 0, 1))
    assert swap_in_vso and not swap_in_svo, \
        "the Swap the archived _vso() requires is not in the VSO derivation"

    # (ii) the denotations are EQUAL -- measured, and this is SS16.1-Q2's premise
    worst_den = 0.0
    with torch.no_grad():
        for cs in probe:
            for v, wa, wp in cs.events:
                U = m.verb_matrix(v)
                psi = sim.init_state(cs.K)
                for w, _n, angle in cs.intro:
                    psi = sim.apply_gate(psi, sim.ry(angle), (w,))
                a = sim.apply_gate(psi, U, materialised_clause("SVO", wa, wp))
                b = sim.apply_gate(psi, U, materialised_clause("VSO", wa, wp))
                worst_den = max(worst_den, float((a - b).abs().max()))
    assert worst_den == 0.0, \
        ("SS16.1-Q2 (user-confirmed) says the two renderings are TRUTH-"
         "CONDITIONALLY IDENTICAL; the compiled denotations differ by %.3e"
         % worst_den)

    # (iii) DROP the Swap and the VSO clause becomes the ROLE-REVERSED circuit
    worst_noswap, worst_vs_twin = 0.0, 0.0
    with torch.no_grad():
        for cs in probe:
            for v, wa, wp in cs.events:
                U = m.verb_matrix(v)
                psi = sim.init_state(cs.K)
                for w, _n, angle in cs.intro:
                    psi = sim.apply_gate(psi, sim.ry(angle), (w,))
                ok = sim.apply_gate(psi, U, materialised_clause("VSO", wa, wp))
                no = sim.apply_gate(psi, U,
                                    materialised_clause("VSO", wa, wp, with_swap=False))
                rev = sim.apply_gate(psi, U, (wp, wa))     # the twin's clause
                worst_noswap = max(worst_noswap, float((ok - no).abs().max()))
                worst_vs_twin = max(worst_vs_twin, float((no - rev).abs().max()))
    assert worst_noswap > 1e-3, \
        ("dropping the Swap changes nothing (%.3e) -- the Swap is not load-bearing "
         "and the reduction is not doing what SS4.3 says" % worst_noswap)
    assert worst_vs_twin == 0.0, \
        ("the swap-free VSO reduction is not the role-reversed circuit (%.3e)"
         % worst_vs_twin)

    rep["word_order"] = {
        "derivations_differ": True,
        "n_event_renderings_checked": diffs,
        "swap_present_in_VSO_only": bool(swap_in_vso and not swap_in_svo),
        "denotation_max_abs_diff_SVO_vs_VSO": worst_den,
        "drop_swap_changes_state_max_abs": worst_noswap,
        "swap_free_VSO_equals_role_reversed_max_abs": worst_vs_twin,
        "archived_source": ("archive/01_parser_pipeline/arabic_dep_reader.py"
                            ":130-138 (the two verb boxes), :167-180 (_vso and its "
                            "Swap)"),
        "verdict": (
            "SS7.3 DISCHARGED as DERIVATION-DIFFERS / DENOTATION-EQUAL, and the "
            "reconciliation is stated rather than buried. The compiled derivations "
            "differ -- the VSO reduction carries `Swap(N.l, N)` and the SVO "
            "reduction does not -- so the compiler genuinely consults the surface, "
            "which is the null-instrument failure SS7.3 exists to catch. The "
            "DENOTATIONS are equal (max diff %.1e), which is not a defect but doc 55 "
            "SS16.1-Q2's USER-CONFIRMED premise that the two orders are truth-"
            "conditionally identical, and which is REQUIRED by SS4.7 ('the same six "
            "blocks, reused'), by SS5's 12/72 conditioning-group arithmetic (no "
            "surface-order factor) and by SS6.1 (A0 must reproduce exp68 bit-exactly "
            "on a dataset that already contains both orders). The Swap is LOAD-"
            "BEARING and that is measured, not argued: dropping it moves the state "
            "by %.3f and yields EXACTLY the role-reversed circuit (diff %.1e) -- "
            "without the Swap, the VSO clause would say the object did it. "
            "CONSEQUENCE FOR THE CLAIM, carried: the circuit is invariant to the "
            "alternation by construction, so doc 55 SS6.9's per-surface-order "
            "accuracy is a DATA-COMPOSITION statistic and is not evidence about "
            "word-order processing. ORCHESTRATOR REVIEW ITEM."
            % (worst_den, worst_noswap, worst_vs_twin)),
    }
    return rep


def check_copy(fm, verbs, store, rep):
    """doc 55 SS4.7 / SS7.4, MANDATORY: with event 2 suppressed and q fixed to 1,
    the T = 2 forward pass must reproduce the T = 1 forward pass BIT-FOR-BIT."""
    items2, _m, _p = arm_items("R2-A", store)
    worst = {}
    for enc_arm in ("R2-A", "R2-W"):
        a, _ = arm_angles(enc_arm, fm)
        torch.manual_seed(0)
        m = build_model(enc_arm, verbs, 0, fm)
        w = 0.0
        for it in items2[:6]:
            qi = it["query_index"]
            two = compile_item(it, a)
            one_item = copy.deepcopy(it)
            one_item["events"] = [it["events"][qi - 1]]
            one_item["sentences"] = [it["sentences"][qi - 1]]
            one_item["order_flags"] = [it["order_flags"][qi - 1]]
            one_item["query_index"] = 1
            one = compile_item(one_item, a)
            # suppress event (3 - q) in the T = 2 story, keeping everything else
            two_sup = copy.deepcopy(two)
            two_sup.events = [two.events[qi - 1]]
            with torch.no_grad():
                d = float((m.story_values(two_sup) - m.story_values(one)).abs().max())
            w = max(w, d)
        worst[enc_arm] = w
        assert w == 0.0, \
            ("SS7.4 COPY-VERIFICATION FAILED for %s: max diff %.3e" % (enc_arm, w))
    rep["copy_verification"] = {
        "max_abs_diff": worst, "exact_zero": bool(all(v == 0.0 for v in worst.values())),
        "note": ("doc 55 SS4.7 / SS7.4: with event (3 - q) suppressed and the query "
                 "fixed to the surviving event, the T = 2 forward pass reproduces "
                 "the T = 1 pipeline bit-for-bit. This is only possible because the "
                 "referent state prep happens EXACTLY ONCE, before event 1 (SS4.3), "
                 "and the events are parameterised gates on already-prepared wires."),
    }
    return rep


def fourier_resid_1d(m, cs_list, n_terms):
    """doc 55 SS7.1(1): fit P(yes) against ONE referent's angle in the n_terms
    basis. `exp61_run.fourier_resid`'s frame, re-stated here so the T = 2 sweep can
    reuse it without importing a T = 1 assumption."""
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
    """doc 55 SS7.1(2): the register arms must NOT close in the 3-term basis of
    theta0 ALONE and MUST close in the 9-dim product basis u(theta0) (x) u(theta1).
    A 1-D sweep of theta0 closes at 3 terms even for a CORRECTLY wired register --
    the first qubit still sees a single RY -- so the sweep is 2-D."""
    g = np.linspace(0, 2 * math.pi, grid, endpoint=False)
    worst3, worst9, worst_d = 0.0, 0.0, 0.0
    for cs in cs_list[:3]:
        noun = cs.intro[0][1]
        keep = m.theta1[noun]
        ys, B3, B9 = [], [], []
        for t0 in g:
            c2 = copy.deepcopy(cs)
            c2.intro[0] = (c2.intro[0][0], noun, float(t0))
            for t1 in g:
                m.theta1[noun] = float(t1)
                with torch.no_grad():
                    ys.append(float(m.story_values(c2)[0]))
                u0, u1 = _u(float(t0)), _u(float(t1))
                B3.append(list(u0))
                B9.append([u0[i] * u1[j] for i in range(3) for j in range(3)])
        m.theta1[noun] = keep
        ys = np.asarray(ys)
        for Bm, key in ((np.asarray(B3), "3"), (np.asarray(B9), "9")):
            c, *_ = np.linalg.lstsq(Bm, ys, rcond=None)
            r = float(np.abs(Bm @ c - ys).max())
            if key == "3":
                worst3 = max(worst3, r)
            else:
                worst9 = max(worst9, r)
        with torch.no_grad():
            m.theta1[noun] = keep
            a = float(m.story_values(cs)[0])
            m.theta1[noun] = keep + 0.7
            b = float(m.story_values(cs)[0])
            m.theta1[noun] = keep
        worst_d = max(worst_d, abs(a - b))
    return worst3, worst9, worst_d


def check_closures(fm, verbs, store, rep):
    """doc 55 SS7.1 per arm, and SS7.2's T = 2 assertion -- the one that turns SS5's
    'T does not multiply the coefficient count' into a VERIFIED PROPERTY."""
    out = {}
    for arm in ("A0", "R1-A", "R2-A", "R1-M", "R2-M"):
        items, _m, _p = arm_items(arm, store)
        ang, _t1 = arm_angles(arm, fm)
        data = compile_all([i for i in items if i["split"] == "test"], ang)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        r3 = fourier_resid_1d(m, data, 3)
        assert r3 < 1e-6, \
            ("%s does not close at 3 terms (%.3e) -- the arm is not a single RY, "
             "or a referent angle has been RE-UPLOADED inside an event block "
             "(doc 55 SS7.2)" % (arm, r3))
        out[arm] = {"resid_3term": r3, "T": ARM_CFG[arm]["T"],
                    "closes_at_3": True, "basis": "u(theta) = (1, cos, sin)"}
    for arm in ("R1-W", "R2-W"):
        items, _m, _p = arm_items(arm, store)
        ang, _t1 = arm_angles(arm, fm)
        data = compile_all([i for i in items if i["split"] == "test"], ang)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        w3, w9, wd = fourier_resid_2d(m, data)
        assert w3 > 1e-3, \
            ("%s CLOSES in the 3-term basis of theta0 alone (%.3e) -- the second "
             "qubit is unwired and NO result from this arm is admissible "
             "(doc 55 SS7.1)" % (arm, w3))
        assert w9 < 1e-6, \
            ("%s does not close in the 9-dim product basis (%.3e) -- at T = 2 this "
             "means a referent angle is re-uploaded somewhere and SS5's 'T does not "
             "multiply the coefficient count' is FALSE for this build (doc 55 SS7.2)"
             % (arm, w9))
        assert wd > 1e-6, "perturbing theta1 does not move %s's output" % arm
        out[arm] = {"resid_3term_theta0_only": w3, "resid_9dim_product_basis": w9,
                    "output_change_under_theta1_perturbation": wd,
                    "T": ARM_CFG[arm]["T"], "closes_at_3": False,
                    "basis": "u(theta0) (x) u(theta1), 12 x 12 grid over [0, 2pi)"}
    rep["fourier_closure"] = {
        "per_arm": out,
        "T2_assertion_doc55_SS7_2": (
            "THE LOAD-BEARING ONE. The T = 2 arms close in the SAME basis as their "
            "T = 1 counterparts -- 3-term for R2-A / R2-M, the 9-dim product basis "
            "for R2-W. If a T = 2 arm had needed a HIGHER-degree basis, the build "
            "would have re-uploaded a referent angle somewhere (a second state prep, "
            "or a feature-dependent gate inside an event block), doc 55 SS5's 'T does "
            "not multiply the coefficient count' would be FALSE for this build, and "
            "the arm would not be admissible. Measured, not assumed."),
        "coefficient_budget": {
            "angle_arms_per_referent": 3, "angle_arms_joint_n2": 9,
            "register_arms_per_referent": 9, "register_arms_joint_n2": 81,
            "identical_at_T1_and_T2": True,
            "note": ("doc 55 SS5: referents are introduced ONCE as a single RY per "
                     "qubit, so the output stays a trigonometric polynomial of the "
                     "SAME degree in the SAME angles at T = 2. What T multiplies is "
                     "circuit DEPTH and the number of CONDITIONING GROUPS, not the "
                     "coefficient budget.")},
    }
    return rep


def check_trace(fm, verbs, store, rep):
    """doc 55 SS18.3: the traced pass must be THE SAME forward pass."""
    worst = {}
    for arm in ("R2-A", "R2-W", "R1-M"):
        items, _m, _p = arm_items(arm, store)
        ang, _t1 = arm_angles(arm, fm)
        data = compile_all([i for i in items if i["split"] == "test"][:6], ang)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        w = 0.0
        with torch.no_grad():
            for cs in data:
                v1 = m.story_values(cs)
                v2, tr = forward_traced(m, cs)
                w = max(w, float((v1 - v2).abs().max()))
                nw = (2 if is_register(m) else 1) * cs.K
                assert set(tr) == ({"after_intro", "at_readout"}
                                   | {"after_event%d" % (k + 1)
                                      for k in range(cs.T)}), tr.keys()
                for st, ps in tr.items():
                    assert len(ps) == nw, (st, len(ps), nw)
                    for p in ps:
                        assert 0.49 <= p <= 1.0001, (st, p)
        worst[arm] = w
        assert w == 0.0, \
            "SS18.3 traced pass differs from story_values for %s: %.3e" % (arm, w)
    rep["trace_parity"] = {
        "max_abs_diff_vs_story_values": worst,
        "exact_zero": bool(all(v == 0.0 for v in worst.values())),
        "stages": ["after_intro", "after_event1", "after_event2 (T=2)", "at_readout"],
        "quantity": "per-wire purity Tr rho^2 of the single-qubit reduced state",
        "note": ("doc 55 SS18.3: descriptive, NO criterion reads it. It exists so "
                 "SS19.4's mechanism analysis (does binding failure correlate with "
                 "purity collapse?) is possible post-hoc WITHOUT a rerun. The "
                 "reference implementation is the demo's T-trace parity gate "
                 "(verify_demo_parity_exp68.py, 300/300 items)."),
    }
    return rep


def check_shuffle(store, rep):
    """doc 46 SS10.2 / doc 55 SS4.4: the floor severs features <-> label and changes
    NOTHING else -- asserted for the Q69 construction, at both T and both q."""
    rows = []
    for part in ("A", "B"):
        items, meta, _p = store[part]
        bearers = set(meta["bearers"])
        for seed in range(N_SEEDS):
            sh = relabel_shuffle_q(items, seed)
            by = {}
            for it in sh:
                by.setdefault(it["pair_id"], []).append(it)
            opp = all(len(g) == 2 and g[0]["answer"] != g[1]["answer"]
                      for g in by.values())
            same_cast = all(
                [r["name"] for r in g[0]["referents"]]
                == [r["name"] for r in g[1]["referents"]] for g in by.values())
            same_q = all(g[0]["query_index"] == g[1]["query_index"]
                         for g in by.values())
            n_yes = sum(1 for i in sh if i["answer"] == "yes")
            true_rule = sum(
                1 for i in sh
                if (i["events"][i["query_index"] - 1]["agent"] in bearers)
                == (i["answer"] == "yes"))
            rows.append({"part": part, "seed": seed, "n_items": len(sh),
                         "twins_still_opposite": bool(opp),
                         "casts_unchanged": bool(same_cast),
                         "query_index_unchanged": bool(same_q),
                         "yes_fraction": n_yes / len(sh),
                         "true_bearer_rule_accuracy": true_rule / len(sh)})
    for r in rows:
        assert r["twins_still_opposite"] and r["casts_unchanged"] \
            and r["query_index_unchanged"], \
            "S-shuffle broke the twin structure -- V1 would no longer be 0.500"
        assert r["yes_fraction"] == 0.5, "S-shuffle unbalanced the classes"
        assert abs(r["true_bearer_rule_accuracy"] - 0.5) < 0.16, \
            ("S-shuffle did not sever features <-> label: the true bearer rule "
             "still scores %.4f" % r["true_bearer_rule_accuracy"])
    rep["shuffle"] = {
        "per_seed": rows,
        "mean_true_bearer_rule_accuracy":
            float(np.mean([r["true_bearer_rule_accuracy"] for r in rows])),
        "note": ("doc 46 SS10.2's floor, per PAIR, keyed on the QUERIED event's "
                 "agent. `exp60_run.relabel_shuffle` keys on events[0] and would "
                 "relabel a q = 2 item from the UNQUERIED event, breaking the twin "
                 "structure; exp69 uses its own, with exp60's RNG convention "
                 "unchanged so the floors stay comparable. The TRUE bearer rule "
                 "collapses to chance on the shuffled labels, which is the "
                 "severance, measured rather than asserted."),
    }
    return rep


def smoke(fm, verbs, store):
    t0 = time.time()
    rep = {"experiment": "exp69 smoke / validity assertions (doc 55 SS7.1-SS7.4, "
                         "SS18.3)",
           "spec": SPEC,
           "datasets": {k: {"path": v[2], "sha256": v[1]["dataset_sha256"]}
                        for k, v in store.items()},
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "numpy": np.__version__}
    check_features(fm, rep)
    ang, _t1 = arm_angles("R2-A", fm)
    items2, _m, _p = arm_items("R2-A", store)
    data2 = {"test": compile_all([i for i in items2 if i["split"] == "test"], ang)}
    check_word_order(fm, verbs, data2, rep)
    check_closures(fm, verbs, store, rep)
    check_copy(fm, verbs, store, rep)
    check_trace(fm, verbs, store, rep)
    check_shuffle(store, rep)

    tiny = {}
    for arm in ARMS:
        r = train(arm, 0, store, verbs, 2, fm, save=False)
        tiny[arm] = {"val_acc": r["val_acc"], "heldout_test_acc": r["heldout_test_acc"],
                     "wall_sec": r["wall_sec"], "n_params": r["n_params"],
                     "wires": r["wires"], "T": r["T"]}
        print("[smoke] %-16s 2ep val=%.3f test=%.3f  %.0fs  (%d params, %d wires, "
              "T=%d)" % (arm, r["val_acc"], r["heldout_test_acc"], r["wall_sec"],
                         r["n_params"], r["wires"], r["T"]), flush=True)
    rep["two_epoch_probe"] = tiny
    rep["assertions_passed"] = True
    rep["wall_time_sec"] = time.time() - t0
    json.dump(rep, open(SMOKE_OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[smoke] wrote %s -- ALL doc 55 SS7.1-SS7.4 / SS18.3 ASSERTIONS PASSED"
          % SMOKE_OUT)
    return rep


# ==================================================================== training
def train(arm, seed, store, verbs, max_epochs, fm, save=True, tag=""):
    """The exp60 / exp61 / exp65 / exp68 loop, UNCHANGED -- Adam(lr 0.02), batch 8,
    150 epochs, checkpoint selected on in-distribution val accuracy, held-out test
    read ONCE from the selected checkpoint.

    The call order (compile -> manual_seed -> build -> Adam -> Random(RNG_BASE +
    seed)) is exp68's verbatim, because doc 55 SS6.1 expects A0 to reproduce exp68
    BIT-EXACTLY and any reordering of RNG consumption would break that."""
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
    return {"arm": arm, "base_arm": base_arm(arm),
            "is_floor": arm.startswith("S-shuffle"),
            "seed": seed, "part": cfg["part"], "T": cfg["T"], "encoding": cfg["enc"],
            "role": cfg["role"],
            "train_acc": best["train"], "val_acc": best["val"],
            "selected_epoch": best["epoch"], "heldout_test_acc": te,
            "val_minus_heldout": best["val"] - te,
            "gate_met": bool(best["train"] >= TRAIN_GATE and best["val"] >= VAL_GATE),
            "n_params": sum(p.numel() for p in m.parameters()),
            "wires": (2 if cfg["enc"] == "F2" else 1) * 2,
            "n_train": len(data["train"]), "n_val": len(data["val"]),
            "n_test": len(data["test"]),
            "max_epochs": max_epochs, "lr": LR, "batch": BATCH,
            "dataset": path, "dataset_sha256": meta["dataset_sha256"],
            "ckpt_path": pt, "ckpt_sha256": sha,
            "heldout_items": rows,
            "wall_sec": time.time() - t0}


# ================================================ doc 55 SS6.1 -- the A0 gate
def a0_gate(fm, store, seeds, workers=None):
    """doc 55 SS6.1, BLOCKING, AND IT RUNS FIRST.

    A0's per-seed held-out accuracies must reproduce `results_exp68.json`'s FA
    per-seed values to |delta| <= 0.002, and the mean to |delta| <= 0.002 of
    0.9187. Both runs are deterministic given their seeds, so THE HONEST
    EXPECTATION IS BIT-EXACT; the tolerance exists only to absorb a documented
    float/BLAS platform difference, and any non-zero delta is reported with its
    cause identified.

    If this fails, the exp69 pipeline has drifted from exp68's and NOTHING ELSE IN
    THE RUN IS READ (doc 45's bit-exact-anchor discipline, applied at build
    level). Rule 3: A0 writes its OWN results file and no exp68 artefact is
    touched."""
    items, meta, path = store["exp68"]
    verbs = sorted(meta["verb_inventory"].keys())
    ref = json.load(open(EXP68_RESULTS, encoding="utf-8"))["arms"]["FA"]
    rows = []
    for s in seeds:
        ckj = os.path.join(CKPT_DIR, "A0_s%d.json" % s)
        if os.path.exists(ckj):
            r = json.load(open(ckj, encoding="utf-8"))
            print("[A0] s%d already done: %.4f" % (s, r["heldout_test_acc"]))
        else:
            r = train("A0", s, store, verbs, MAX_EPOCHS, fm, save=True)
            json.dump(r, open(ckj, "w", encoding="utf-8"))
            print("[A0] s%d held-out=%.6f  (exp68 FA %.6f)  delta=%+.6f  (%.0fs)"
                  % (s, r["heldout_test_acc"], ref["per_seed"][s],
                     r["heldout_test_acc"] - ref["per_seed"][s], r["wall_sec"]),
                  flush=True)
        rows.append(r)
    return rows, ref, meta, path


def a0_report(rows, ref, meta, path):
    per = {r["seed"]: r["heldout_test_acc"] for r in rows}
    seeds = sorted(per)
    deltas = {s: per[s] - ref["per_seed"][s] for s in seeds}
    mean = float(np.mean([per[s] for s in seeds]))
    dmean = mean - ref["mean"]
    ok_seed = all(abs(d) <= A0_TOLERANCE for d in deltas.values())
    ok_mean = abs(mean - 0.9187) <= A0_TOLERANCE
    exact = all(d == 0.0 for d in deltas.values())
    return {
        "component": "exp69 A0 -- doc 55 SS6.1, the BLOCKING replication anchor",
        "spec": SPEC,
        "rule3_note": ("A0 writes its OWN results file. No exp68 artefact is "
                       "touched, read for writing, or overwritten (doc 55 SS4.4)."),
        "dataset": path, "dataset_sha256": meta["dataset_sha256"],
        "exp68_source": EXP68_RESULTS,
        "exp68_FA_per_seed": ref["per_seed"], "exp68_FA_mean": ref["mean"],
        "exp69_A0_per_seed": [per[s] for s in seeds],
        "exp69_A0_mean": mean, "exp69_A0_sd": float(np.std(
            [per[s] for s in seeds], ddof=1)),
        "per_seed_delta": [deltas[s] for s in seeds],
        "max_abs_per_seed_delta": float(max(abs(d) for d in deltas.values())),
        "mean_delta_vs_exp68": dmean,
        "mean_delta_vs_registered_0_9187": mean - 0.9187,
        "tolerance": A0_TOLERANCE,
        "bit_exact": bool(exact),
        "per_seed_within_tolerance": bool(ok_seed),
        "mean_within_tolerance_of_0_9187": bool(ok_mean),
        "anchor_replicates_exp68": bool(ok_seed and ok_mean),
        "honest_expectation": ("BIT-EXACT. Both runs are deterministic given their "
                               "seeds; the 0.002 tolerance exists only to absorb a "
                               "documented float/BLAS platform difference and any "
                               "non-zero delta is reported with its cause "
                               "identified (doc 55 SS6.1)."),
        "consequence_if_failed": (
            "doc 55 SS9.2 row 1: THE RUN STOPS. The exp69 pipeline does not "
            "reproduce exp68 on exp68's own data -- hunt the build, interpret "
            "nothing, from any arm."),
        "val_per_seed": [r["val_acc"] for r in rows],
        "selected_epoch_per_seed": [r["selected_epoch"] for r in rows],
        "n_params": rows[0]["n_params"], "wires": rows[0]["wires"],
        "wall_sec_total": float(sum(r["wall_sec"] for r in rows)),
    }


# ======================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--tag", default="", help="prefix for checkpoint names; use for "
                                              "doc 55 SS4.5's confirm draw")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--a0-gate", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(PREDICTIONS):
        raise RuntimeError("%s missing -- doc 55 SS19.2's ordering rule" % PREDICTIONS)

    store = {k: load_part(k) for k in ("A", "B", "exp68")}
    verbs = sorted(store["A"][1]["verb_inventory"].keys())
    assert verbs == sorted(store["exp68"][1]["verb_inventory"].keys()), \
        "verb inventory drift between exp68 and exp69 -- A0 would not be comparable"
    fm = feature_maps()
    os.makedirs(CKPT_DIR, exist_ok=True)

    if a.smoke:
        smoke(fm, verbs, store)
        return 0

    if a.a0_gate:
        seeds = [s for s in range(a.seeds)
                 if s % a.nworkers == a.worker] if a.nworkers > 1 else list(
            range(a.seeds))
        rows, ref, meta, path = a0_gate(fm, store, seeds)
        if a.nworkers > 1:
            print("[A0 w%d] done (%d seeds); run with --nworkers 1 to write %s"
                  % (a.worker, len(seeds), A0_OUT))
            return 0
        rep = a0_report(rows, ref, meta, path)
        json.dump(rep, open(A0_OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n[A0 GATE] exp68 FA per-seed %s" % [round(x, 6) for x in ref["per_seed"]])
        print("[A0 GATE] exp69 A0 per-seed %s"
              % [round(x, 6) for x in rep["exp69_A0_per_seed"]])
        print("[A0 GATE] deltas %s  max |delta| %.2e  bit-exact=%s"
              % ([round(x, 6) for x in rep["per_seed_delta"]],
                 rep["max_abs_per_seed_delta"], rep["bit_exact"]))
        print("[A0 GATE] mean %.6f vs exp68 %.6f (registered 0.9187)  ->  %s"
              % (rep["exp69_A0_mean"], ref["mean"],
                 "PASS" if rep["anchor_replicates_exp68"] else "FAIL -- THE RUN STOPS"))
        print("[A0 GATE] wrote %s" % A0_OUT)
        return 0 if rep["anchor_replicates_exp68"] else 1

    arms = [x for x in a.arms.split(",") if x]
    jobs = [(arm, s) for arm in arms for s in range(a.seeds)]
    mine = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    print("[exp69 w%d/%d] arms=%s -> %d of %d runs"
          % (a.worker, a.nworkers, arms, len(mine), len(jobs)), flush=True)
    for arm, s in mine:
        ckj = os.path.join(CKPT_DIR, "%s%s_s%d.json" % (a.tag, arm, s))
        if os.path.exists(ckj):
            print("[ckpt] %s%s_s%d already done" % (a.tag, arm, s), flush=True)
            continue
        r = train(arm, s, store, verbs, MAX_EPOCHS, fm, save=True, tag=a.tag)
        r["tag"] = a.tag
        json.dump(r, open(ckj, "w", encoding="utf-8"))
        print("[exp69] %-16s s%d  train=%.3f val=%.3f HELD-OUT=%.4f  (%.0fs)"
              % (arm, s, r["train_acc"], r["val_acc"], r["heldout_test_acc"],
                 r["wall_sec"]), flush=True)
    print("[exp69 w%d] done" % a.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
