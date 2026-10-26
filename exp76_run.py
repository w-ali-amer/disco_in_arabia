# -*- coding: utf-8 -*-
"""exp76_run.py -- THE SCALING RUNG (doc 63, D3 of doc 59's six-direction plan).

REGISTRATION: qnlp_private_docs/63_exp76_scaling.md, REGISTERED 2026-08-29;
the orchestrator rulings in the status header are BINDING over the body. 120
runs ACCEPTED. PHASE 1 ONLY (SS0.1) -- no MPS is built; the certificate is.

THE QUESTION (SS1.1). The single-angle family binds a category to a role at 2
referents, 2 events and 24 nouns, and composition costs it 0.004. Does any of
that survive when the cast grows, the story lengthens, and the noun inventory
becomes a real one -- and if it degrades, is the degradation in the TASK or in
the CIRCUIT?

WHAT IS NEW, AND IT IS ALMOST NOTHING (SS7). No new ansatz, no new gate, no new
featurizer, no new readout. The verb blocks are the committed `Circuit4(2, 3)`
at the (agent, patient) wires of each event and the readout effects are the
committed `q_pos` / `q_neg`. 6 x 15 + 2 x 15 = 120 trainable parameters at
n = 2 AND at n = 7. exp76 is the first rung in this programme where the
intervention adds ZERO parameters, so a degradation cannot be a capacity story.
The n-generalisation is a GENERATOR and COMPILER change (SS7.2); the model file
is untouched and SS8.4 asserts it.

INIT: THE COMMITTED UNIFORM INIT ONLY (SS7.4), on three registered grounds --
the drift anchor requires it, a second init doubles a 120-run design for a
variable that has not replicated, and doc 62 SS3.5 forbids merging init results
across rungs anyway.

  ==> CONSEQUENCE FOR THE STANDING BOUNDED-REDRAW PATTERN (doc 62 SS24.3/SS24.6,
      committed 2026-09-02): exp76 has NO identity-init arm. Nothing here calls
      `exp72_run.apply_identity_init` or `exp72_run.identity_assertions`, so the
      committed window that refuses ~0.1-0.3% of draws is never consulted and
      the driver-level bounded redraw is NOT APPLICABLE and is NOT BUILT.
      Recorded as a decision, with its reason, in every run record's
      `init_scheme` block and in the smoke report -- an absent mechanism that is
      absent for a stated reason is not an omission.

NO DIRECTIONAL CALL IS PLACED (SS13). Both branches of every criterion are read
(SS14). The programme's record is 1 held (twice-drawn), 2 missed.

SCOPE, INHERITED AND NON-NEGOTIABLE (SS0.3 / SS16):
  * NO QUANTUM READING OF ANY KIND. The circuit is a 120-parameter bilinear form
    in single-angle features and the comparator is logistic regression on the
    identical features. No exp76 number may appear in a sentence containing
    "advantage".
  * Not a statement about scaling in general -- n_active in {2,3,4,5,7},
    T in {2,4,6}, one qubit per referent, one category, one surface, two
    inventory sizes, 5 seeds, dense simulation.
  * Not a statement about barren plateaus: no gradient-variance-versus-qubit-
    count measurement is made.
  * NOT A WAZN FINDING. SS3.6.2's bands exclude the GEOMETRY correlate for all
    five classes; SS3.6.3 records that they exclude the SEMANTIC-COHERENCE one
    for none, and SS21 item 9's control stays registered-open. No exp76 number
    may be quoted as evidence that an encoding "carries wazn".

THE SPLIT PATTERN (SS20.4): THIS FILE BUILDS, GATES AND TRAINS. It writes no
results file except the smoke report and the anchor gate. `exp76_aggregate.py`
is the ONLY writer of `results_exp76.json`.

    python exp76_datagen.py                        # the 11 cells + the V-suite
    python exp76_run.py --smoke                    # SS8 / SS10.4, ALL gates
    python exp76_run.py --worker i --nworkers 5    # anchor jobs are ordered FIRST
    python exp76_run.py --anchor-report            # writes results_exp76_anchors.json
    python exp76_comparator.py                     # SS10.2, tiers (i)/(ii)/(iii)
    python exp76_certificate.py                    # SS9, from the stored per-cut data
    python exp76_aggregate.py                      # the ONLY results writer
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
import exp76_datagen as D76                           # the Q76 generator
import instrument_scaling_bounds as ISB
from exp42_models import QuantumStoryModel
from exp42_run import grad_global_norm
from exp60_run import evaluate

SPEC = ("qnlp_private_docs/63_exp76_scaling.md (REGISTERED 2026-08-29, "
        "orchestrator rulings in the status header, BINDING over the body)")
SPEC_DOC = "63_exp76_scaling.md"

DATA_EXP69 = R69.DATA_A                               # stories_exp69_QA.json, FROZEN
EXP69_RESULTS = "results_exp69.json"
EXP69_VSUITE = "results_exp69_WN_validation.json"
EXP76_VSUITE = D76.VALIDATION_OUT
BOUNDS_ARTEFACT = D76.BOUNDS_ARTEFACT

CKPT_DIR = "exp76_ckpt"
SMOKE_OUT = "results_exp76_smoke_validation.json"
ANCHOR_OUT = "results_exp76_anchors.json"

# doc 63 SS6.4 -- the frozen exp69 dataset the drift anchor runs on.
DATASET_SHA_DOC63 = ("2b85822b544571969fd40cab12b03a4ac256b08aee914d2b9643b856"
                     "a6308c96")

# doc 63 SS6.1: the committed recipe, IMPORTED and never re-typed.
LR, BATCH, MAX_EPOCHS = E.LR, E.BATCH, E.MAX_EPOCHS    # 0.02, 8, 150
TRAIN_GATE, VAL_GATE = E.TRAIN_GATE, E.VAL_GATE        # recorded, never a criterion
RNG_BASE = E.RNG_BASE                                  # 55000
N_SEEDS = E.N_SEEDS                                    # 5

ANCHOR_TOLERANCE = 0.002                               # doc 63 SS6.4
N_PARAMS = 120                                         # doc 63 SS7.3, at EVERY cell
FLOOR_RNG_BASE = 9_000_000                             # exp60's S-shuffle convention

# doc 63 SS8.3 -- the inert-padding control's registered tolerance. The padded
# state has a different memory layout, so `prob_all_zero_on`'s summation order
# differs; SS4.4.1 measured the residue at 6.0e-08 in complex64, two orders
# inside the bar. A build reading the readout over ALL wires must FAIL this.
PAD_TOL = 1e-6
PAD_TO_WIRES = 8                                       # SS8.3's construction
PAD_SOURCE_CELL = "n3_T2"                              # SS8.3 names the (3,2) cell

# doc 63 SS8.4 -- structural assertion tolerances.
PURITY_TOL_F64 = 1e-9                                  # item 3, in complex128
CLOSE_TOL = 1e-6                                       # item 7, the 3-term basis
OPEN_TOL = 1e-3                                        # item 7's failing direction
SWEEP_POINTS = 48

# doc 63 SS9.2 -- the certificate.
SCHMIDT_EPS = 1e-6
CERT_STAGES = ("after_intro", "after_event1", "after_last_event", "at_readout")
CHI_INERT = 64                                         # SS9.3, registered INERT
AREA_LAW_SLOPE_BAR = 0.10                              # SS9.3.1, bits per wire

# doc 63 SS12 -- the frozen bars, INHERITED and never chosen here.
BINDS_BAR = 0.65                                       # doc 54 SS7
DELTA_BAR = 0.06                                       # 2 x binomial SD at n_test 300
FLOOR_MULTIPLE = 2.0                                   # SS12, floor's own SD PRIMARY

# doc 63 SS6.4's committed per-seed values, quoted so they can be ASSERTED
# against results_exp69.json rather than used in its place (SS8.6).
ANCHOR_ARMS = ("A0-A-T1", "A0-A-T2")
ANCHOR_SOURCE = {"A0-A-T1": "R1-A", "A0-A-T2": "R2-A"}
ANCHOR_T = {"A0-A-T1": 1, "A0-A-T2": 2}
ANCHOR_PER_SEED_DOC63 = {
    "A0-A-T1": [0.903333, 0.936667, 0.926667, 0.920000, 0.926667],
    "A0-A-T2": [0.916667, 0.946667, 0.873333, 0.913333, 0.943333]}
ANCHOR_MEAN_DOC63 = {"A0-A-T1": 0.9226667, "A0-A-T2": 0.9186667}
COMMITTED_CC_A = 0.0040                                # doc 63 SS0.2's ladder row


# ==================================================================== the arms
ARM_CFG = {}
for _c, _cfg in D76.CELL_CFG.items():
    ARM_CFG[_c] = {
        "kind": "cell", "cell": _c, "n": _cfg["n"], "T": _cfg["T"],
        "inventory": _cfg["inv"], "category": _cfg["cat"], "axis": _cfg["axis"],
        "enc": "FA", "init": "uniform", "n_params": N_PARAMS,
        "role": ("doc 63 SS6.2 grid cell" if _cfg["axis"] == "grid" else
                 "doc 63 SS6.2.3 grade-3 target" if _cfg["axis"] == "grade3" else
                 "doc 63 SS6.3 inventory axis")}
for _a in ANCHOR_ARMS:
    ARM_CFG[_a] = {
        "kind": "anchor", "cell": None, "n": 2, "T": ANCHOR_T[_a],
        "inventory": "exp69_QA", "category": "nonoccupational", "axis": "anchor",
        "enc": "FA", "init": "uniform", "n_params": N_PARAMS,
        "role": ("THE DRIFT ANCHOR (doc 63 SS6.4): exp69's %s re-run through "
                 "exp76's own n-generalised codepath at n = 2, on exp69's FROZEN "
                 "dataset with exp69's FROZEN features. BLOCKING."
                 % ANCHOR_SOURCE[_a])}

TRAINED_CELLS = list(D76.CELL_CFG)                     # 11
FLOOR_ARMS = ["S-shuffle-%s" % a for a in TRAINED_CELLS]
FLOOR_OF = {a: "S-shuffle-%s" % a for a in TRAINED_CELLS}
ALL_ARMS = list(ANCHOR_ARMS) + TRAINED_CELLS + FLOOR_ARMS      # 24 arms, 120 runs


def base_arm(arm):
    return arm[len("S-shuffle-"):] if arm.startswith("S-shuffle-") else arm


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def is_anchor(arm):
    return ARM_CFG[base_arm(arm)]["kind"] == "anchor"


# =========================================================== the feature layer
_FM = {}


def feature_maps():
    """Two maps, and doc 63 SS2.9(b) is why they are two.

    `exp69` -- the COMMITTED frozen 44-noun FA plane, through
    `exp68_run.feature_maps`, used by and only by the drift anchor (SS6.4: the
    anchor runs on exp69's frozen dataset with exp69's frozen features).

    `exp76` -- the 519-noun FA angle map of SS2, through the pre-work
    instrument's own code objects. SS2.9(b) registers that this is NOT the
    committed plane: the PCA is fit over 563 vectors, post-pin agreement over
    the 24 humans is 1.643, and the sign pin is a DETERMINISM device, not a
    continuity claim."""
    if _FM:
        return _FM
    fm69 = E.feature_maps()
    f519 = D76.features519()
    _FM.update({"exp69": {n: float(v) for n, v in fm69["phi_exp42"].items()},
                "exp76": f519["angles"],
                "exp69_meta": {"embeddings_sha256": fm69["embeddings_sha256"]},
                "exp76_prov": f519["prov"]})
    return _FM


def arm_angles(arm, fm):
    return fm["exp69"] if is_anchor(arm) else fm["exp76"]


def _u(t):
    return (1.0, math.cos(t), math.sin(t))


# ==================================================================== the data
_STORE = {}


def load_store():
    """Every dataset this run touches, provenance-verified on load."""
    if _STORE:
        return _STORE
    blob = json.load(open(DATA_EXP69, encoding="utf-8"))
    items69, meta69 = blob["items"], blob["meta"]
    if comp.canonical_sha256(items69) != meta69["dataset_sha256"]:
        raise RuntimeError("exp69 Part A provenance FAILED for %s" % DATA_EXP69)
    _STORE["exp69"] = (items69, meta69, DATA_EXP69)
    for cell in TRAINED_CELLS:
        _STORE[cell] = D76.load_cell(cell)
    return _STORE


def arm_items(arm, store):
    cfg = ARM_CFG[base_arm(arm)]
    if cfg["kind"] == "anchor":
        items, meta, path = store["exp69"]
        sel = [i for i in items if int(i.get("T", 1)) == cfg["T"]]
        assert sel, "no items at T=%d in %s" % (cfg["T"], path)
        return sel, meta, path
    return store[cfg["cell"]]


def relabel_shuffle_q76(items, seed):
    """doc 63 SS18.1's floor: the COMMITTED S-shuffle -- bearer designation
    reassigned PER PAIR, independently of the features (doc 55 SS4.4).

    `exp69_run.relabel_shuffle_q` draws the fake bearer from `referents[:2]`,
    which at n = 2 IS the queried pair. At n > 2 it is not, and drawing from a
    distractor would break the twin structure (the twins would carry the SAME
    label, V1 would stop being exactly 0.500 and the floor would stop being
    procedure-matched). The GENERALISATION is therefore: draw from the queried
    event's two participants, IN WIRE ORDER. At n = 2 that is byte-for-byte
    `exp69_run.relabel_shuffle_q`, so the anchor's floor lineage is unbroken.
    The RNG convention (`random.Random(9_000_000 + seed)`) is exp60's,
    unchanged, so the floors stay comparable across rungs."""
    rng = random.Random(FLOOR_RNG_BASE + seed)
    byp, out = {}, []
    for it in items:
        byp.setdefault(it["pair_id"], []).append(it)
    for _pid, g in byp.items():
        g = sorted(g, key=lambda x: x["id"])
        wires = {r["name"]: w for w, r in enumerate(g[0]["referents"])}
        q = int(g[0].get("query_index", 1))
        qe = g[0]["events"][q - 1]
        cand = sorted((qe["agent"], qe["patient"]), key=lambda x: wires[x])
        fake_bearer = cand[rng.randrange(2)]           # independent of the angle
        for it in g:
            j = copy.deepcopy(it)
            qi = int(it.get("query_index", 1))
            j["answer"] = ("yes" if it["events"][qi - 1]["agent"] == fake_bearer
                           else "no")
            out.append(j)
    return out


def compile_item(item, angles):
    """exp76's compile path. `exp69_run.compile_item` is n-GENERIC already --
    `exp42_compiler.compile_story` builds `K = len(item['referents'])` and
    places 2-wire blocks at the event wires -- so the n-generalisation is a
    GENERATOR change, not a compiler rewrite (doc 63 SS7.2). It is CALLED, not
    copied, and SS8.4 item 1 asserts the forward pass through it is bit-for-bit
    the committed one at n = 2."""
    cs = R69.compile_item(item, angles)
    cs.n_active = len(D76.reverse_reachable(cs.events, cs.question[1:])[1])
    cs.cell = item.get("cell")
    return cs


def compile_all(items, angles):
    return [compile_item(i, angles) for i in items]


def compile_arm(arm, items, seed, fm):
    ang = arm_angles(arm, fm)
    use = relabel_shuffle_q76(items, seed) if arm.startswith("S-shuffle") else items
    return {s: compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}


def build_model(arm, verbs, seed, fm):
    """doc 63 SS7.1: NO NEW ANSATZ. The committed `QuantumStoryModel('A3')`,
    byte-for-byte the model exp60 / exp61 / exp68 / exp69 use, unchanged. Its
    `story_values` already builds `sim.init_state(cs.K)` for arbitrary K and
    applies 2-wire blocks at `(wa, wp)`, which is the whole reason SS7.3's
    parameter count is constant at 120."""
    return QuantumStoryModel("A3", verbs, seed)


# ============================================ doc 63 SS9 / SS11.1 -- the traces
def wire_purities(psi, nwires):
    """Tr rho^2 per wire (doc 63 SS11.1). exp69_run's definition, n-generic."""
    out = []
    for w in range(nwires):
        p = torch.movedim(psi, w, 0).reshape(2, -1)
        rho = p @ p.conj().transpose(0, 1)
        out.append(float((rho.abs() ** 2).sum()))
    return out


def schmidt_cuts(psi, nwires, eps=SCHMIDT_EPS):
    """doc 63 SS9.2, exactly as registered. For each of the n - 1 CONTIGUOUS
    wire-cuts {0..k} | {k+1..n-1}:

        lambda_i(k)   the Schmidt coefficients across cut k (SVD of the
                      reshaped state)
        S(k)          = - sum_i lambda_i^2 log2 lambda_i^2   (bits)
        R_eps(k)      = min { r : sum_{i<=r} lambda_i^2 >= 1 - eps }, eps = 1e-6
        S_max_page(k) = min(k+1, n-k-1)

    SS9.6 item 1: this is CIRCUIT-LEVEL entropy of the evolving state, and it
    says so. No function-level claim is made anywhere. The SVD is taken in
    complex128 -- the simulator runs complex64 and a 1e-6 Schmidt tail is right
    at its noise floor."""
    z = psi.detach().to(torch.complex128)
    S, R, P = [], [], []
    for k in range(nwires - 1):
        M = z.reshape(2 ** (k + 1), -1)
        lam = torch.linalg.svdvals(M)
        p = (lam ** 2)
        p = p / p.sum()
        pp = p[p > 1e-30]
        S.append(float(-(pp * torch.log2(pp)).sum()))
        cum = torch.cumsum(p, 0)
        R.append(int(torch.searchsorted(cum, torch.tensor(1.0 - eps,
                                                          dtype=cum.dtype)).item()) + 1)
        P.append(int(min(k + 1, nwires - k - 1)))
    return S, R, P


def forward_traced(m, cs, certificate=True):
    """doc 63 SS11.1 + SS9.2: `story_values` with the per-wire purity trace AND
    the per-cut Schmidt data taken FROM THE SAME FORWARD PASS. Gate order is
    identical to `exp42_models.QuantumStoryModel.story_values`, which
    `check_trace` asserts bit-for-bit."""
    cache = {}

    def block(v):
        if v not in cache:
            cache[v] = m.verb_matrix(v)
        return cache[v]

    nw = cs.K
    psi = sim.init_state(nw)
    for w, _noun, angle in cs.intro:
        psi = sim.apply_gate(psi, sim.ry(angle), (w,))
    pur, sch = {}, {}

    def snap(name):
        pur[name] = wire_purities(psi, nw)
        if certificate:
            S, R, P = schmidt_cuts(psi, nw)
            k = int(np.argmax(S)) if S else 0
            sch[name] = {"S_max": (float(max(S)) if S else 0.0),
                         "R_max": (int(max(R)) if R else 1),
                         "argmax_cut": k, "S": S, "R": R, "S_page": P}

    snap("after_intro")
    for k, (v, wa, wp) in enumerate(cs.events):
        psi = sim.apply_gate(psi, block(v), (wa, wp))
        if k == 0:
            snap("after_event1")
    snap("after_last_event")
    vseq, qa, qp = cs.question
    qw = (qa, qp)
    for qv in reversed(vseq):
        psi = sim.apply_gate(psi, block(qv).conj().T, qw)
    snap("at_readout")
    vals = torch.stack([sim.prob_all_zero_on(
        sim.apply_gate(psi, eff.matrix().conj().T, qw), qw)
        for eff in (m.q_pos, m.q_neg)])
    return vals, pur, sch


def evaluate_with_traces(m, compiled, certificate=True):
    """Held-out accuracy PLUS SS11.1's traces and SS9's per-cut Schmidt data.
    One pass, no_grad, no RNG consumed."""
    rows, correct = [], 0
    with torch.no_grad():
        for cs in compiled:
            vals, pur, sch = forward_traced(m, cs, certificate=certificate)
            pred = int(torch.argmax(vals).item())
            ok = int(pred == cs.answer)
            correct += ok
            r = dict(cs.meta)
            r.update({"pred": pred, "target": int(cs.answer), "correct": bool(ok),
                      "p_yes": float(vals[0]), "p_no": float(vals[1]),
                      "n_active": int(cs.n_active),
                      "queried_wires": [int(cs.question[1]), int(cs.question[2])],
                      "purity": pur})
            if certificate:
                r["schmidt"] = {k: {"S_max": v["S_max"], "R_max": v["R_max"],
                                    "argmax_cut": v["argmax_cut"]}
                                for k, v in sch.items()}
            rows.append(r)
    return correct / len(compiled), rows


def certificate_summary(rows, nwires):
    """SS9.2's aggregation: per-cell max, mean and 95th percentile of the
    per-item max-over-cuts, plus the argmax stage and cut so a hot spot is
    locatable."""
    out = {"n_wires": nwires, "n_cuts": max(nwires - 1, 0),
           "eps": SCHMIDT_EPS, "stages": {}}
    best = {"S": -1.0, "stage": None, "cut": None, "item": None}
    have = [r for r in rows if "schmidt" in r]
    for st in CERT_STAGES:
        vals = [r["schmidt"][st]["S_max"] for r in have if st in r["schmidt"]]
        rs = [r["schmidt"][st]["R_max"] for r in have if st in r["schmidt"]]
        if not vals:
            continue
        out["stages"][st] = {
            "S_max": float(np.max(vals)), "S_mean": float(np.mean(vals)),
            "S_p95": float(np.percentile(vals, 95)),
            "R_max": int(np.max(rs)), "R_mean": float(np.mean(rs)),
            "R_p95": float(np.percentile(rs, 95))}
        i = int(np.argmax(vals))
        if vals[i] > best["S"]:
            best = {"S": float(vals[i]), "stage": st,
                    "cut": int(have[i]["schmidt"][st]["argmax_cut"]),
                    "item": have[i].get("id")}
    out["argmax"] = best
    out["S_max_overall"] = best["S"]
    out["R_max_overall"] = (max(v["R_max"] for v in out["stages"].values())
                            if out["stages"] else 1)
    out["S_page"] = int(nwires // 2)
    out["filling_fraction"] = (best["S"] / float(nwires // 2)
                               if nwires >= 2 else None)
    out["mps_feasible_at_chi64"] = bool(out["R_max_overall"] <= CHI_INERT)
    out["mps_feasible_at_chi64_is_INERT"] = (
        "doc 63 SS9.3: a contiguous cut of an n-wire state has Schmidt rank at "
        "most 2^min(k+1, n-k-1), so across the registered grid (n_active <= 7) "
        "the rank can never exceed 8 and chi = 64 is TRUE BY ARITHMETIC. It is "
        "computed and stored for completeness and REGISTERED AS INERT. NO "
        "READING DEPENDS ON IT. A certificate that cannot fail is not one.")
    return out


# ============================================================== the smoke gates
def check_datasets(store, rep):
    """SS8.5 -- the inventory sha, the frozen exp69 dataset sha, and the exp76
    V-suite's kill gate, all asserted before anything else is believed."""
    items69, meta69, _p = store["exp69"]
    src = json.load(open(EXP69_RESULTS, encoding="utf-8"))
    assert meta69["dataset_sha256"] == DATASET_SHA_DOC63, \
        "doc 63 SS6.4: the frozen exp69 dataset sha does not match"
    assert src["datasets"]["A"]["sha256"] == DATASET_SHA_DOC63, \
        "results_exp69.json records a different Part A sha"
    v69 = json.load(open(EXP69_VSUITE, encoding="utf-8"))
    assert v69["kill_gate_passed"] is True, \
        "exp69's V-suite kill gate did not pass -- no exp69 value may be reused"
    v76 = json.load(open(EXP76_VSUITE, encoding="utf-8"))
    assert v76["kill_gate_passed"] is True, \
        ("%s does not record a passing V-suite -- run `python exp76_datagen.py` "
         "first (doc 63 SS20.3 step 8: the V-suite precedes the runner)"
         % EXP76_VSUITE)
    assert v76["nouns_sha256"] == D76.NOUNS_SHA256_DOC63
    cells = {}
    for cell in TRAINED_CELLS:
        _i, meta, path = store[cell]
        cells[cell] = {"path": path, "sha256": meta["dataset_sha256"],
                       "n": meta["n"], "T": meta["T"],
                       "n_items": meta["n_items"],
                       "split_counts": meta["split_counts"]}
        assert meta["split_counts"] == {"train": 528, "val": 132, "test": 300}, \
            "doc 63 SS6.1's frozen budget moved at %s" % cell
    rep["datasets_doc63_SS8_5"] = {
        "exp69_frozen": {"path": DATA_EXP69, "sha256": meta69["dataset_sha256"],
                         "n_items": len(items69)},
        "inventory_nouns_sha256": D76.NOUNS_SHA256_DOC63,
        "exp76_cells": cells,
        "exp76_vsuite": {"file": EXP76_VSUITE, "kill_gate_passed": True,
                         "flagged_ambiguities": [a["id"] for a in
                                                 v76["flagged_ambiguities"]]}}
    return rep


def check_committed_imports(rep):
    """SS8.6 -- R1-A's and R2-A's per-seed accuracies, means, sds, floors and
    comparator values are READ PROGRAMMATICALLY from results_exp69.json; the
    ladder from results_exp73.json; the memberships and bounds from the
    pre-work artefact. The doc-typed constants exist ONLY to be asserted
    against the files, never to be used in their place."""
    src = json.load(open(EXP69_RESULTS, encoding="utf-8"))
    out = {}
    for arm, key in ANCHOR_SOURCE.items():
        a = src["arms"][key]
        want = ANCHOR_PER_SEED_DOC63[arm]
        got = [float(x) for x in a["per_seed"]]
        assert len(got) == len(want) == 5
        for i, (g, w) in enumerate(zip(got, want)):
            assert abs(g - w) < 5e-5, \
                ("doc 63 SS6.4's %s seed %d is %.6f, results_exp69.json has %.6f"
                 % (arm, i, w, g))
        assert abs(float(a["mean"]) - ANCHOR_MEAN_DOC63[arm]) < 5e-5
        assert int(a["n_params"]) == N_PARAMS, \
            "doc 63 SS7.3: the committed arm is not 120 parameters"
        out[arm] = {"exp69_arm": key, "per_seed": got, "mean": float(a["mean"]),
                    "sd": float(a["sd"]),
                    "floor_mean": float(src["floors"][key]["mean"]),
                    "floor_sd": float(src["floors"][key]["sd"]),
                    "comparator": float(src["comparator"]["arms"][key]["mean"]),
                    "bound": float(src["bounds_context_only_doc55_SS12"][key]
                                   ["bound"]),
                    "wall_sec_total": float(a["wall_sec_total"])}
    cc = out["A0-A-T1"]["mean"] - out["A0-A-T2"]["mean"]
    assert abs(cc - COMMITTED_CC_A) < 5e-4, \
        ("doc 63 SS0.2's cc_A = %.4f does not reproduce from results_exp69.json "
         "(%.4f)" % (COMMITTED_CC_A, cc))
    art = json.load(open(BOUNDS_ARTEFACT, encoding="utf-8"))
    g1 = {k: float(art["contrasts"][k]["value"])
          for k in ("faail_camel", "band_32arc")}
    assert abs(g1["faail_camel"] - 0.6426) < 5e-4
    assert abs(g1["band_32arc"] - 0.5298) < 5e-4
    ladder = {}
    if os.path.exists("results_exp73.json"):
        try:
            r73 = json.load(open("results_exp73.json", encoding="utf-8"))
            ladder = r73.get("composition_cost_doc61_SS10_5", {}) or {}
        except Exception:                              # pragma: no cover
            ladder = {}
    rep["committed_imports_doc63_SS8_6"] = {
        "anchors": out,
        "composition_cost_cc_A_recomputed": cc,
        "grade1_bounds_doc63_SS2_6": g1,
        "binds_bar_doc54_SS7": BINDS_BAR,
        "delta_bar_doc63_SS12": DELTA_BAR,
        "exp73_ladder_row": ladder,
        "SS12_4_1_note": (
            "REGISTERED BEFORE ANY ARM RUNS: faail_camel's pair-level grade-1 "
            "bound is 0.6426 and the inherited `binds` bar is 0.65, so the bar "
            "sits ABOVE the reachable bound. `binds` is retained unchanged "
            "(doc 55 SS6.2 forbids re-choosing a bar) but is registered as "
            "plausibly unreachable by arithmetic across the 519-noun cells, and "
            "NO HEADLINE CRITERION IS GATED ON IT. SS12.9.3 gates grade 3 on "
            "FLOOR CLEARANCE instead.")}
    return rep


def check_features(fm, rep):
    """SS8.4 item 4 -- the encoding angles are the FA map of SS2's featurizer,
    recomputed from the committed convention chain and asserted against
    `diagnostic_scaling_bounds.json :: features`.

    THE ARTEFACT STORES NO PER-NOUN ANGLES (it stores the featurizer's
    provenance block), so the assertion is executed as an equality of that
    block: re-run `instrument_scaling_bounds.build_fa_features` and compare its
    returned provenance dict to the artefact's, which pins the raw-vector shas,
    the PCA fit population, the variance explained, the sign pin and the
    normalisation divisors. FLAGGED as the only executable reading of item 4."""
    art = json.load(open(BOUNDS_ARTEFACT, encoding="utf-8"))
    assert fm["exp76_prov"] == art["features"], \
        ("doc 63 SS8.4 item 4: the re-derived FA feature provenance differs from "
         "diagnostic_scaling_bounds.json :: features -- a re-derived feature "
         "that differs anywhere fails the run")
    ang = np.array(sorted(fm["exp76"].values()))
    assert len({round(float(x), 9) for x in ang}) >= 500, \
        "the 519-noun angle map collapses the inventory"
    assert float(ang.std(ddof=1)) > 0.5, "the exp76 angle map is degenerate"
    # SS2.9(b): the exp76 plane is NOT the committed plane, and the number says so.
    common = sorted(set(fm["exp69"]) & set(fm["exp76"]))
    drift = (float(max(abs(fm["exp69"][n] - fm["exp76"][n]) for n in common))
             if common else None)
    rep["features_doc63_SS8_4_4"] = {
        "provenance_matches_prework_artefact": True,
        "n_nouns": len(fm["exp76"]), "angle_sd": float(ang.std(ddof=1)),
        "n_distinct_angles": len({round(float(x), 9) for x in ang}),
        "nouns_shared_with_the_committed_44": len(common),
        "max_abs_angle_difference_on_shared_nouns": drift,
        "SS2_9b_note": (
            "doc 63 SS2.9(b): the exp76 plane is NOT the committed plane -- the "
            "PCA is fit over 563 vectors and the sign pin is a DETERMINISM "
            "device, not a continuity claim. SS6.4's drift anchor is unaffected "
            "because it runs on exp69's FROZEN dataset with exp69's FROZEN "
            "features, which is a SEPARATE map in this file (`fm['exp69']`).")}
    return rep


def check_anchor_codepath(fm, verbs, store, rep):
    """SS8.4 item 1, THE ONE THAT MAKES SS6.4 MEAN ANYTHING: at n = 2 the
    n-generalised compiler and model produce a forward pass BIT-FOR-BIT
    identical to the committed `exp42_models` path on the committed exp69 items.

    Failing direction: any difference at n = 2 means the generalisation changed
    the arithmetic, and SS6.4's anchor proves nothing."""
    out = {}
    for arm in ANCHOR_ARMS:
        items, _m, _p = arm_items(arm, store)
        ang = arm_angles(arm, fm)
        worst_fw = worst_pw = 0.0
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            mine = build_model(arm, verbs, seed, fm)
            torch.manual_seed(seed)
            ref = QuantumStoryModel("A3", verbs, seed)
            for (ka, pa), (kb, pb) in zip(sorted(mine.state_dict().items()),
                                          sorted(ref.state_dict().items())):
                assert ka == kb
                worst_pw = max(worst_pw, float((pa - pb).abs().max()))
            for it in items[:8]:
                a = compile_item(it, ang)                  # exp76's path
                b = comp.compile_story(it, ang)            # the committed path
                with torch.no_grad():
                    d = float((mine.story_values(a)
                               - ref.story_values(b)).abs().max())
                worst_fw = max(worst_fw, d)
        assert worst_pw == 0.0, \
            "SS8.4 item 1: exp76's model is not the committed model (%.3e)" % worst_pw
        assert worst_fw == 0.0, \
            ("SS8.4 item 1: exp76's n-generalised codepath differs from the "
             "committed exp42_models path at n = 2 by %.3e -- the "
             "generalisation CHANGED THE ARITHMETIC and SS6.4's anchor proves "
             "nothing" % worst_fw)
        out[arm] = {"max_abs_param_diff": worst_pw,
                    "max_abs_forward_diff": worst_fw, "bit_exact": True,
                    "n_items_checked": 8, "n_seeds": N_SEEDS}
    rep["anchor_codepath_doc63_SS8_4_1"] = {
        "per_arm": out, "bit_exact_all": True,
        "honest_expectation": (
            "doc 63 SS6.4: |delta| = 0 EXACTLY, bit-exact, 10/10. The "
            "n-generalisation is not a rewrite -- `exp42_models.story_values` "
            "already builds `sim.init_state(cs.K)` for arbitrary K -- so at "
            "n = 2 the codepath must be the SAME ARITHMETIC, not merely the "
            "same answer. The 0.002 tolerance exists only to absorb a "
            "documented float/BLAS platform difference.")}
    return rep


def check_parameters(fm, verbs, store, rep):
    """SS8.4 item 2 -- the trainable parameter count is EXACTLY 120 at every
    cell; the verb blocks are 2-WIRE and SHARED ACROSS EVENTS; no per-wire and
    no per-referent parameter exists.

    Failing direction: any other count, or a parameter indexed by wire, fails
    the arm."""
    torch.manual_seed(0)
    m = build_model(TRAINED_CELLS[0], verbs, 0, fm)
    total = sum(p.numel() for p in m.parameters())
    assert total == N_PARAMS, \
        "SS8.4 item 2: %d trainable parameters, not 120" % total
    keys = sorted(m.state_dict())
    assert len(keys) == 8, \
        "SS8.4 item 2: %d parameter tensors, expected 6 verb blocks + 2 readout"
    for k, t in m.state_dict().items():
        assert t.numel() == 15, \
            "SS8.4 item 2: %s carries %d parameters, not Circuit4(2,3)'s 15" \
            % (k, t.numel())
    nouns = set()
    for cell in TRAINED_CELLS:
        items, _mm, _p = store[cell]
        for it in items[:50]:
            nouns.update(r["name"] for r in it["referents"])
    bad = [k for k in keys if any(nn in k for nn in nouns)]
    assert not bad, "SS8.4 item 2: parameters indexed by referent: %r" % bad
    per_cell = {}
    for cell in TRAINED_CELLS:
        torch.manual_seed(0)
        mm = build_model(cell, verbs, 0, fm)
        c = sum(p.numel() for p in mm.parameters())
        assert c == N_PARAMS, "SS8.4 item 2: %s has %d parameters" % (cell, c)
        per_cell[cell] = {"n_params": c, "n": ARM_CFG[cell]["n"],
                          "statevector_dim": 2 ** ARM_CFG[cell]["n"]}
    rep["parameters_doc63_SS8_4_2"] = {
        "n_params": total, "n_param_tensors": len(keys),
        "params_per_tensor": 15, "per_cell": per_cell,
        "no_per_wire_or_per_referent_parameter": True,
        "reading": ("doc 63 SS1.3: 6 x 15 + 2 x 15 = 120 at n = 2 and 120 at "
                    "n = 7. exp76 is the first rung in this programme where the "
                    "intervention adds ZERO parameters, so a degradation cannot "
                    "be a capacity story and SS22's sample-complexity row is "
                    "answered by arithmetic.")}
    return rep


def intro_state_f64(cs):
    """The intro state in complex128, for SS8.4 item 3's 1e-9 assertion. The
    simulator runs complex64 and its purity floor is ~1e-7, so the assertion is
    made on a recomputation, exactly as doc 62 SS8.4.4 did."""
    psi = torch.zeros((2,) * cs.K, dtype=torch.complex128)
    psi[(0,) * cs.K] = 1.0
    for w, _n, a in cs.intro:
        h = float(a) * 0.5
        U = torch.tensor([[math.cos(h), -math.sin(h)],
                          [math.sin(h), math.cos(h)]], dtype=torch.complex128)
        psi = torch.movedim(torch.tensordot(U, psi, dims=([1], [w])), 0, w)
    return psi


def check_intro(fm, verbs, store, rep):
    """SS8.4 item 3 -- intro per-wire purity = 1.0 to 1e-9 for every referent at
    every n: the referents are introduced as a PRODUCT STATE.
    SS8.4 item 5 -- the intro consumes NO RNG draws for the data path and
    registers NO data-dependent parameters.

    Failing directions, both asserted below: a value below 1 means an entangler
    has entered the intro, which is a different experiment; a preparation that
    registers per-noun parameters fails the arm."""
    ang = fm["exp76"]
    per_cell, worst = {}, 0.0
    for cell in TRAINED_CELLS:
        items, _m, _p = store[cell]
        w = 0.0
        for it in items[:6]:
            cs = compile_item(it, ang)
            psi = intro_state_f64(cs)
            for k in range(cs.K):
                p = torch.movedim(psi, k, 0).reshape(2, -1)
                rho = p @ p.conj().transpose(0, 1)
                w = max(w, abs(float((rho.abs() ** 2).sum()) - 1.0))
            S, _R, _P = schmidt_cuts(psi, cs.K)
            w = max(w, float(max(S)) if S else 0.0)
        assert w <= PURITY_TOL_F64, \
            ("SS8.4 item 3: intro purity at %s is off 1.0 by %.3e -- an "
             "ENTANGLER has entered the intro and that is a different "
             "experiment" % (cell, w))
        per_cell[cell] = w
        worst = max(worst, w)

    # ---- item 5: the intro consumes no RNG and registers no per-noun state --
    items, _m, _p = store[TRAINED_CELLS[0]]
    cs = compile_item(items[0], ang)
    torch.manual_seed(0)
    m = build_model(TRAINED_CELLS[0], verbs, 0, fm)
    before = torch.get_rng_state().clone()
    pyst = random.getstate()
    with torch.no_grad():
        v1 = m.story_values(cs).clone()
        v2 = m.story_values(cs).clone()
    after = torch.get_rng_state()
    assert torch.equal(before, after), \
        "SS8.4 item 5: the data path CONSUMED a torch RNG draw"
    assert random.getstate() == pyst, \
        "SS8.4 item 5: the data path CONSUMED a python RNG draw"
    assert float((v1 - v2).abs().max()) == 0.0, \
        "SS8.4 item 5: the forward pass is not deterministic"

    # ---- THE FAILING DIRECTION for item 3: an entangled intro must FAIL -----
    # A Bell pair on wires 0 and 1: purity 0.5 per wire, S = 1 bit per cut. If
    # this does NOT fire, the purity/Schmidt instrument cannot see an entangler
    # and SS8.4 item 3 is a null assertion.
    bad = torch.zeros((2,) * cs.K, dtype=torch.complex128)
    idx0 = (0,) * cs.K
    idx1 = (1, 1) + (0,) * (cs.K - 2)
    bad[idx0] = 1.0 / math.sqrt(2.0)
    bad[idx1] = 1.0 / math.sqrt(2.0)
    pb = torch.movedim(bad, 0, 0).reshape(2, -1)
    rb = pb @ pb.conj().transpose(0, 1)
    pur_bad = float((rb.abs() ** 2).sum())
    Sb, Rb, _P = schmidt_cuts(bad, cs.K)
    assert abs(pur_bad - 1.0) > 1e-3 and max(Sb) > 0.9, \
        ("SS8.4 item 3's FAILING DIRECTION DID NOT FIRE: an entangled intro "
         "reports purity %.12f and S_max %.6f" % (pur_bad, max(Sb)))

    rep["intro_doc63_SS8_4_3_and_5"] = {
        "max_abs_purity_deviation_f64": worst, "tolerance": PURITY_TOL_F64,
        "per_cell": per_cell, "product_state_at_every_n": True,
        "no_rng_consumed_on_the_data_path": True,
        "no_data_dependent_parameters": True,
        "failing_direction_entangled_intro": {
            "purity": pur_bad, "S_max": float(max(Sb)),
            "R_max": int(max(Rb)), "fires": True,
            "reading": ("a value below 1 means an entangler has entered the "
                        "intro, which is a different experiment (SS8.4 item 3)")}}
    return rep


def check_copy(fm, verbs, store, rep):
    """SS8.4 item 6 -- THE COPY ASSERTION. At (n, T) = (2, 2) with the committed
    dataset, event 2 suppressed and q = 1 reproduces the T = 1 forward pass
    BIT-FOR-BIT (`exp69_run.py :: check_copy`). Re-run here on exp76's own
    (2,2) cell as well, because the assertion is about exp76's compiler.

    Failing direction: a failure means the intro has leaked into the event
    loop."""
    out = {}
    for tag, arm, ang in (("exp69_committed", "A0-A-T2", fm["exp69"]),
                          ("exp76_n2_T2", "n2_T2", fm["exp76"])):
        items, _m, _p = arm_items(arm, store)
        torch.manual_seed(0)
        m = build_model(arm, verbs, 0, fm)
        worst = 0.0
        for it in items[:6]:
            qi = int(it.get("query_index", 1))
            two = compile_item(it, ang)
            one_item = copy.deepcopy(it)
            one_item["events"] = [it["events"][qi - 1]]
            one_item["sentences"] = [it["sentences"][qi - 1]]
            one_item["order_flags"] = [it["order_flags"][qi - 1]]
            one_item["query_index"] = 1
            one = compile_item(one_item, ang)
            two_sup = copy.deepcopy(two)
            two_sup.events = [two.events[qi - 1]]
            with torch.no_grad():
                worst = max(worst, float((m.story_values(two_sup)
                                          - m.story_values(one)).abs().max()))
        assert worst == 0.0, \
            ("SS8.4 item 6 COPY-VERIFICATION FAILED for %s: max diff %.3e -- "
             "the intro has leaked into the event loop" % (tag, worst))
        out[tag] = {"arm": arm, "max_abs_diff": worst, "exact_zero": True}
    rep["copy_doc63_SS8_4_6"] = {
        "per_source": out,
        "note": ("with the non-queried event suppressed and the query fixed to "
                 "the surviving event, the T-event forward pass reproduces the "
                 "T = 1 pipeline bit-for-bit. This is only possible because the "
                 "referent state prep happens EXACTLY ONCE, before event 1, and "
                 "the events are parameterised gates on already-prepared wires.")}
    return rep


class _SecondRYModel(QuantumStoryModel):
    """SS8.4 item 7's FAILING DIRECTION: a build with a SECOND RY on a wire
    re-uploads a referent angle. It must FAIL the 3-term Fourier closure -- a
    wider basis means a referent has been re-uploaded, and that is exp75, not
    exp76."""

    def story_values(self, cs, verb_map=None):
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)
            return cache[v]

        psi = sim.init_state(cs.K)
        for w, _n, a in cs.intro:
            psi = sim.apply_gate(psi, sim.ry(a), (w,))
        for k, (v, wa, wp) in enumerate(cs.events):
            psi = sim.apply_gate(psi, block(v), (wa, wp))
            if k == 0:                                  # the second upload
                for w, _n, a in cs.intro:
                    psi = sim.apply_gate(psi, sim.ry(a), (w,))
        vseq, qa, qp = cs.question
        qw = (qa, qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)
        return torch.stack([sim.prob_all_zero_on(
            sim.apply_gate(psi, eff.matrix().conj().T, qw), qw)
            for eff in (self.q_pos, self.q_neg)])


def fourier_resid_queried(m, cs_list, n_terms=3, n_items=4):
    """SS8.4 item 7: fit P(yes) against the QUERIED AGENT's angle in the
    committed 3-term basis {1, cos theta, sin theta}.

    exp69's `fourier_resid_1d` sweeps wire 0, which at n > 2 may hold a
    DISTRACTOR whose angle the label provably does not depend on (SS4.5.1) --
    the residual would then be zero for a trivial reason and the gate would be a
    NULL INSTRUMENT. Sweeping the queried agent's wire keeps the gate live, and
    the sweep's own output range is returned so vacuity is measurable."""
    ts = np.linspace(0, 2 * math.pi, SWEEP_POINTS, endpoint=False)
    worst, rng_out = 0.0, 0.0
    for cs in cs_list[:n_items]:
        _v, wa, _wp = cs.events[cs.qidx - 1]
        j = [k for k, (w, _n, _a) in enumerate(cs.intro) if w == wa][0]
        ys = []
        for t in ts:
            c2 = copy.deepcopy(cs)
            c2.intro[j] = (c2.intro[j][0], c2.intro[j][1], float(t))
            with torch.no_grad():
                ys.append(float(m.story_values(c2)[0]))
        cols = [np.ones_like(ts)]
        for k in range(1, n_terms // 2 + 1):
            cols += [np.cos(k * ts), np.sin(k * ts)]
        A = np.stack(cols, 1)
        coef, *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
        worst = max(worst, float(np.abs(A @ coef - np.array(ys)).max()))
        rng_out = max(rng_out, float(np.ptp(np.array(ys))))
    return worst, rng_out


def check_closures(fm, verbs, store, rep):
    """SS8.4 item 7 -- the Fourier closure holds at EVERY cell: the model closes
    in the committed 3-term basis per referent to < 1e-6, and a build with a
    second RY on any wire must FAIL it."""
    ang = fm["exp76"]
    per_cell = {}
    for cell in TRAINED_CELLS:
        items, _m, _p = store[cell]
        data = compile_all([i for i in items if i["split"] == "test"][:8], ang)
        torch.manual_seed(0)
        m = build_model(cell, verbs, 0, fm)
        r3, span = fourier_resid_queried(m, data)
        assert r3 < CLOSE_TOL, \
            ("SS8.4 item 7: %s does not close at 3 terms (%.3e) -- the arm is "
             "not a single RY, or a referent angle has been RE-UPLOADED inside "
             "an event block" % (cell, r3))
        assert span > 1e-3, \
            ("SS8.4 item 7: the closure sweep at %s moves the output by only "
             "%.3e -- the gate is a null instrument at this cell" % (cell, span))
        per_cell[cell] = {"resid_3term": r3, "sweep_output_range": span,
                          "closes_at_3": True}
    # ---- THE FAILING DIRECTION -------------------------------------------
    items, _m, _p = store["n2_T2"]
    data = compile_all([i for i in items if i["split"] == "test"][:6], ang)
    torch.manual_seed(0)
    bad = _SecondRYModel("A3", verbs, 0)
    rb, spanb = fourier_resid_queried(bad, data)
    assert rb > OPEN_TOL, \
        ("SS8.4 item 7's FAILING DIRECTION DID NOT FIRE: a build with a SECOND "
         "RY still closes in the 3-term basis (%.3e) -- the closure gate is a "
         "null instrument" % rb)
    rep["fourier_closure_doc63_SS8_4_7"] = {
        "per_cell": per_cell, "tolerance": CLOSE_TOL,
        "failing_direction_second_RY": {
            "resid_3term": rb, "sweep_output_range": spanb, "fires": True,
            "bar": OPEN_TOL,
            "reading": ("a wider basis means a referent has been re-uploaded -- "
                        "that is exp75, not exp76 (SS8.4 item 7)")},
        "coefficient_budget": {
            "per_referent": 3, "joint_at_n2": 9,
            "identical_at_every_T": True,
            "note": ("doc 55 SS5: referents are introduced ONCE as a single RY "
                     "per qubit, so the output stays a trigonometric polynomial "
                     "of the SAME degree in the SAME angles at every T. What T "
                     "multiplies is circuit DEPTH and the number of "
                     "CONDITIONING GROUPS, not the coefficient budget.")}}
    return rep


class _AllWiresReadoutModel(QuantumStoryModel):
    """SS8.3's FAILING DIRECTION: a build in which the readout is taken over ALL
    wires rather than the queried pair. It must FAIL the inert-padding
    assertion. A pipeline that passes both is not implementing SS4.4's
    theorem."""

    def story_values(self, cs, verb_map=None):
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)
            return cache[v]

        psi = sim.init_state(cs.K)
        for w, _n, a in cs.intro:
            psi = sim.apply_gate(psi, sim.ry(a), (w,))
        for v, wa, wp in cs.events:
            psi = sim.apply_gate(psi, block(v), (wa, wp))
        vseq, qa, qp = cs.question
        qw = (qa, qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)
        vals = []
        for eff in (self.q_pos, self.q_neg):
            phi = sim.apply_gate(psi, eff.matrix().conj().T, qw)
            vals.append(sim.prob_all_zero_on(phi, tuple(range(cs.K))))
        return torch.stack(vals)


def pad_compiled(cs, to_wires, filler):
    """SS8.3's construction: pad a compiled circuit to `to_wires` with referents
    that appear in NO EVENT, drawing their angles from the same inventory.

    THIS IS A CIRCUIT CONSTRUCTION, NOT A DATASET: no story is generated with
    unnamed referents (SS6.2), and no such item is trained on."""
    out = copy.deepcopy(cs)
    out.K = to_wires
    extra = [(w, filler[i][0], filler[i][1])
             for i, w in enumerate(range(cs.K, to_wires))]
    out.intro = list(cs.intro) + extra
    return out


def check_padding(fm, verbs, store, rep):
    """SS8.3 -- THE INERT-PADDING CONTROL, BLOCKING, and it is the failing
    direction of the whole wire-count machinery.

        max over items of | story_values_padded_to_8 - story_values_(3,2) | <= 1e-6

    SS4.4's theorem: a unitary acting entirely on the complement of the readout
    pair cancels under the partial trace by cyclicity, so an event is ACTIVE iff
    it is reverse-reachable from R, and every referent touched only by
    non-reachable events is EXACTLY INERT."""
    ang = fm["exp76"]
    items, _m, _p = store[PAD_SOURCE_CELL]
    data = compile_all([i for i in items if i["split"] == "test"][:24], ang)
    n0 = data[0].K
    pool = sorted(ang)
    filler = [(pool[7 * k % len(pool)], ang[pool[7 * k % len(pool)]])
              for k in range(PAD_TO_WIRES)]
    torch.manual_seed(0)
    m = build_model(PAD_SOURCE_CELL, verbs, 0, fm)
    worst = 0.0
    with torch.no_grad():
        for cs in data:
            a = m.story_values(cs)
            b = m.story_values(pad_compiled(cs, PAD_TO_WIRES, filler))
            worst = max(worst, float((a - b).abs().max()))
    assert worst <= PAD_TOL, \
        ("doc 63 SS8.3 / SS12.2 INERT-PADDING CONTROL FAILED: padding %s from "
         "%d to %d wires moved the forward pass by %.3e (bar %.1e). SS4.4's "
         "theorem is NOT implemented, every n_active label in the run is wrong, "
         "and THE RUN STOPS." % (PAD_SOURCE_CELL, n0, PAD_TO_WIRES, worst,
                                 PAD_TOL))
    # ---- an event that DOES touch a readout wire must move the answer -------
    live = 0.0
    with torch.no_grad():
        for cs in data[:8]:
            c2 = copy.deepcopy(cs)
            v, _wa, _wp = c2.events[0]
            c2.events = list(c2.events) + [(v, c2.question[1], c2.question[2])]
            live = max(live, float((m.story_values(cs)
                                    - m.story_values(c2)).abs().max()))
    assert live > 1e-3, \
        ("SS8.3: an event on the READOUT wires does not move the answer either "
         "-- the padding control is a null instrument (%.3e)" % live)
    # ---- THE FAILING DIRECTION: an all-wires readout must FAIL --------------
    torch.manual_seed(0)
    bad = _AllWiresReadoutModel("A3", verbs, 0)
    worst_bad = 0.0
    with torch.no_grad():
        for cs in data:
            a = bad.story_values(cs)
            b = bad.story_values(pad_compiled(cs, PAD_TO_WIRES, filler))
            worst_bad = max(worst_bad, float((a - b).abs().max()))
    assert worst_bad > PAD_TOL, \
        ("doc 63 SS8.3's FAILING DIRECTION DID NOT FIRE: a build reading the "
         "readout over ALL WIRES still passes the padding assertion (%.3e). A "
         "pipeline that passes both is not implementing SS4.4's theorem."
         % worst_bad)
    rep["inert_padding_doc63_SS8_3"] = {
        "source_cell": PAD_SOURCE_CELL, "from_wires": n0,
        "to_wires": PAD_TO_WIRES, "n_items": len(data),
        "max_abs_diff": worst, "tolerance": PAD_TOL, "holds": True,
        "live_event_on_readout_wires_moves_by": live,
        "failing_direction_all_wires_readout": {
            "max_abs_diff": worst_bad, "fires": True},
        "why_not_bit_exact": (
            "the padded state has a different memory layout, so "
            "`prob_all_zero_on`'s summation order differs; SS4.4.1 measured the "
            "residue at 6.0e-08 in complex64 and 1e-6 is the registered "
            "tolerance."),
        "why_not_trained": (
            "if the forward pass agrees to 1e-6 the gradients do too, and over "
            "150 epochs the two runs would differ only by accumulated float "
            "divergence. A trained padded cell would MEASURE FLOAT NOISE AND "
            "CALL IT A WIRE EFFECT (SS6.2.2). The same argument is why SS6.2's "
            "grid contains no cell with an inert referent.")}
    return rep


def check_active_structure(fm, store, rep):
    """SS8.2 V9 at the runner, plus its FAILING DIRECTION: an item padded with a
    referent that appears in no event must be REJECTED by the reachability
    instrument. An item containing an inert event is a smaller item wearing a
    larger cell's name."""
    ang = fm["exp76"]
    per_cell = {}
    for cell in TRAINED_CELLS:
        items, meta, _p = store[cell]
        want = min(meta["n"], meta["T"] + 1)
        for it in items[:40]:
            cs = compile_item(it, ang)
            active, R = D76.reverse_reachable(cs.events, cs.question[1:])
            assert all(active) and len(R) == want == cs.K, \
                "SS8.2 V9 failed at %s on %s" % (cell, it["id"])
        per_cell[cell] = {"n_active": want, "all_events_active": True}
    items, _m, _p = store["n3_T2"]
    it = copy.deepcopy(items[0])
    pool = sorted(set(ang) - {r["name"] for r in it["referents"]})
    it["referents"] = it["referents"] + [{"name": pool[0], "type": "human"}]
    cs = compile_item(it, ang)
    _a, R = D76.reverse_reachable(cs.events, cs.question[1:])
    assert len(R) < cs.K, \
        ("SS8.2 V9's FAILING DIRECTION DID NOT FIRE: an item padded with a "
         "referent that appears in no event still reports every wire reachable")
    rep["active_structure_doc63_SS8_2_V9"] = {
        "per_cell": per_cell,
        "failing_direction_padded_item": {
            "K": cs.K, "n_reachable": len(R), "fires": True,
            "reading": ("SS0.4: a referent that appears in no event appears in "
                        "NO SENTENCE -- the story never names it. Such a cell "
                        "would be a smaller cell plus unused wires and would "
                        "measure simulation cost and nothing else.")}}
    return rep


def check_trace(fm, verbs, store, rep):
    """The traced forward must equal `story_values` BIT-FOR-BIT (doc 55 SS18.3's
    discipline). If it does not, SS11's traces and SS9's certificate describe a
    circuit the run did not train."""
    ang = fm["exp76"]
    worst, sanity = 0.0, {}
    for cell in TRAINED_CELLS:
        items, _m, _p = store[cell]
        data = compile_all([i for i in items if i["split"] == "test"][:6], ang)
        torch.manual_seed(0)
        m = build_model(cell, verbs, 0, fm)
        for cs in data:
            with torch.no_grad():
                a = m.story_values(cs)
                b, pur, sch = forward_traced(m, cs)
            worst = max(worst, float((a - b).abs().max()))
            for st, v in pur.items():
                for x in v:
                    assert 0.49 <= x <= 1.0001, \
                        "purity %r out of range at %s/%s" % (x, cell, st)
            assert sch["after_intro"]["S_max"] < 1e-5, \
                ("the certificate reports entanglement %.3e at the INTRO of %s "
                 "-- the intro is a product state and the Schmidt code is wrong"
                 % (sch["after_intro"]["S_max"], cell))
        sanity[cell] = {"S_after_last_event":
                        sch["after_last_event"]["S_max"],
                        "R_after_last_event":
                        sch["after_last_event"]["R_max"]}
    assert worst == 0.0, \
        ("the traced forward differs from `story_values` by %.3e -- SS11's "
         "traces and SS9's certificate would describe a circuit the run did not "
         "train" % worst)
    live = max(v["S_after_last_event"] for v in sanity.values())
    assert live > 1e-6, \
        ("the certificate reports ZERO entanglement after every event at every "
         "cell -- the Schmidt instrument is not measuring anything")
    rep["trace_and_certificate_doc63_SS9_2_SS11_1"] = {
        "max_abs_diff_vs_story_values": worst, "bit_exact": True,
        "intro_is_a_product_state": True,
        "per_cell_after_last_event": sanity,
        "stages": list(CERT_STAGES),
        "SS9_6_scoping": [
            "CIRCUIT-LEVEL entropy of the evolving state (arXiv:2307.06937 "
            "distinguishes it from the coefficient-MPS entropy; SS9 measures the "
            "STATE and makes no function-level claim).",
            "THE INFERENCE RUNS ONE WAY ONLY: a low R_1e-6 soundly certifies "
            "FEASIBILITY; a high one certifies nothing about hardness "
            "(arXiv:2607.08396).",
            "TRAINABILITY AND MPS-FEASIBILITY ARE NOT INDEPENDENT CLAIMS "
            "(arXiv:2312.09121): `gap_stable_in_n` TRUE together with "
            "`entanglement_area_law_like` TRUE is ONE finding wearing two "
            "names, and no write-up may present them as mutual support."]}
    return rep


def check_comparator_frame(fm, store, rep):
    """doc 56 SS14's null-instrument-by-symmetry, second live case (SS22): a
    comparator that is flat BECAUSE IT IS BROKEN would pass SS10.4, so the
    comparator's FRAME is asserted too -- WIRE-ORDERED features must be
    bit-identical across twins (provably 0.500, a null instrument) while
    AGENT-ORDERED features separate every twin pair."""
    ang = fm["exp76"]
    out = {}
    for cell in TRAINED_CELLS:
        items, _m, _p = store[cell]
        te = [i for i in items if i["split"] == "test"]
        byp = {}
        for it in te:
            byp.setdefault(it["pair_id"], []).append(it)
        worst_wire, worst_agent = 0.0, None
        for _p2, g in sorted(byp.items())[:60]:
            assert len(g) == 2
            fs_w, fs_a = [], []
            for it in g:
                cs = compile_item(it, ang)
                noun = {w: n for w, n, _a in cs.intro}
                q1, q2 = cs.question[1], cs.question[2]
                _v, wa, wp = cs.events[cs.qidx - 1]
                fs_w.append(np.outer(_u(ang[noun[q1]]),
                                     _u(ang[noun[q2]])).ravel())
                fs_a.append(np.outer(_u(ang[noun[wa]]),
                                     _u(ang[noun[wp]])).ravel())
            worst_wire = max(worst_wire, float(np.abs(fs_w[0] - fs_w[1]).max()))
            d = float(np.abs(fs_a[0] - fs_a[1]).max())
            worst_agent = d if worst_agent is None else min(worst_agent, d)
        assert worst_wire == 0.0, \
            ("the WIRE-ORDERED comparator frame separates twins at %s (%.3e) -- "
             "it must be bit-identical and score exactly 0.500" % (cell,
                                                                   worst_wire))
        assert worst_agent > 1e-9, \
            ("the AGENT-ORDERED comparator frame is identical across some twin "
             "pair at %s -- the arbiter frame cannot see the label" % cell)
        out[cell] = {"wire_frame_max_abs_twin_diff": worst_wire,
                     "agent_frame_min_abs_twin_diff": worst_agent}
    rep["comparator_frame_doc63_SS22"] = {
        "per_cell": out, "wire_frame_is_a_null_instrument": True,
        "agent_frame_separates_every_twin": True}
    return rep


def check_shuffle(store, rep):
    """SS18.1's floor construction: geometry bit-identical to the matched arm,
    only the feature <-> label correlation severed; per PAIR, so twins stay on
    opposite labels and V1 stays exactly 0.500. Plus the n = 2 lineage
    assertion: `relabel_shuffle_q76` must agree BYTE-FOR-BYTE with the committed
    `exp69_run.relabel_shuffle_q` on exp69's own items."""
    out = {}
    for cell in TRAINED_CELLS:
        items, _m, _p = store[cell]
        sh = relabel_shuffle_q76(items, 0)
        assert len(sh) == len(items)
        byp = {}
        for it in sh:
            byp.setdefault(it["pair_id"], []).append(it)
        for _p2, g in byp.items():
            assert {x["answer"] for x in g} == {"yes", "no"}, \
                ("SS18.1: the S-shuffle broke the twin structure at %s -- V1 "
                 "would stop being exactly 0.500" % cell)
        base = {i["id"]: i for i in items}
        for it in sh:
            b = base[it["id"]]
            assert it["events"] == b["events"] and it["referents"] == b["referents"]
        moved = sum(1 for it in sh if it["answer"] != base[it["id"]]["answer"])
        out[cell] = {"n_items": len(sh), "labels_moved": moved,
                     "geometry_bit_identical": True, "twins_opposite": True}
    items69, _m, _p = store["exp69"]
    a = R69.relabel_shuffle_q(items69, 3)
    b = relabel_shuffle_q76(items69, 3)
    assert [x["id"] for x in a] == [x["id"] for x in b] and \
        [x["answer"] for x in a] == [x["answer"] for x in b], \
        ("SS18.1: exp76's generalised S-shuffle is NOT byte-identical to the "
         "committed `exp69_run.relabel_shuffle_q` at n = 2 -- the floor lineage "
         "is broken")
    rep["floors_doc63_SS18_1"] = {
        "per_cell": out, "n_floors": len(TRAINED_CELLS),
        "n2_lineage_byte_identical_to_exp69": True,
        "construction": ("the committed S-shuffle -- bearer designation "
                         "reassigned PER PAIR, independently of the features "
                         "(doc 55 SS4.4) -- same items, same splits, 5 seeds, "
                         "run under the IDENTICAL cell procedure. Eleven cells, "
                         "eleven floors, all in stage 1 (SS18.2).")}
    return rep


def check_grid(rep):
    """SS6.2 / SS4.4.2 -- the registered grid, asserted rather than trusted, and
    SS19.2's run arithmetic checked against the arm table."""
    by_t = {}
    for cell, cfg in D76.CELL_CFG.items():
        if cfg["axis"] == "grid":
            by_t.setdefault(cfg["T"], []).append(cfg["n"])
    got = {t: sorted(v) for t, v in by_t.items()}
    assert got == D76.GRID_BY_T, \
        "doc 63 SS6.2's grid is %r, the arm table gives %r" % (D76.GRID_BY_T, got)
    for cell, cfg in D76.CELL_CFG.items():
        assert cfg["n"] <= cfg["T"] + 1, \
            ("doc 63 SS4.4.2: %s violates n <= T + 1 and would carry a cast "
             "size that is a fiction, not a variable" % cell)
    n_runs = len(ALL_ARMS) * N_SEEDS
    assert n_runs == 120, "doc 63 SS19.2 registers 120 runs; the table gives %d" \
        % n_runs
    rep["grid_doc63_SS6_2"] = {
        "grid_by_T": got, "cast_sizes": sorted({c["n"] for c in
                                                D76.CELL_CFG.values()}),
        "n_trained_cells": len(TRAINED_CELLS), "n_floors": len(FLOOR_ARMS),
        "n_anchor_runs": len(ANCHOR_ARMS) * N_SEEDS, "n_runs": n_runs,
        "run_arithmetic_doc63_SS19_2": {
            "drift_anchor": len(ANCHOR_ARMS) * N_SEEDS,
            "grid_cells": 8 * N_SEEDS, "grade3_target": 1 * N_SEEDS,
            "inventory_axis": 2 * N_SEEDS,
            "matched_floors": len(FLOOR_ARMS) * N_SEEDS,
            "inert_padding": 0, "total": n_runs},
        "deviation_note_doc63_SS19_5": (
            "The specification asked for n in {2,4,8} x T in {2,4,6} -- 9 cells. "
            "SS4.4.2 caps n_active at T + 1, so five of those nine would have "
            "contained referents that appear in NO SENTENCE. This doc registers "
            "8 trained cells at cast sizes 2, 3, 4, 5 and 7 -- same cell count, "
            "same cost, DIFFERENT cast sizes -- and the orchestrator RATIFIED "
            "the deviation in doc 63's status header.")}
    return rep


def check_init_scheme(rep):
    """SS7.4 -- exp76 registers the COMMITTED UNIFORM INIT ONLY, and the
    standing bounded-redraw pattern is therefore NOT APPLICABLE. Recorded as a
    decision with its reason, because an absent mechanism that is absent for a
    stated reason is not an omission."""
    bound = [k for k in globals()
             if k in ("apply_identity_init", "identity_assertions", "R72")]
    assert not bound, \
        ("doc 63 SS7.4: exp76 registers ONE init, but this module has bound the "
         "identity-init API %r -- either the registration changed or an arm was "
         "added without one" % bound)
    inits = {ARM_CFG[base_arm(a)]["init"] for a in ALL_ARMS}
    assert inits == {"uniform"}, \
        "doc 63 SS7.4 registers ONE init and the arm table carries %r" % inits
    rep["init_doc63_SS7_4"] = {
        "schemes": sorted(inits), "n_init_arms": 1,
        "bounded_redraw_doc62_SS24_3": {
            "applicable": False,
            "reason": ("doc 63 SS7.4 registers the COMMITTED UNIFORM INIT ONLY. "
                       "No exp76 arm calls `exp72_run.apply_identity_init` or "
                       "`exp72_run.identity_assertions`, so the committed "
                       "identity window -- which refuses ~0.1-0.3%% of draws on "
                       "new architectures and is the thing doc 62 SS24.3's "
                       "driver-level bounded redraw exists to absorb -- is never "
                       "consulted. The pattern is NOT BUILT, and this record is "
                       "the statement that it was considered and found "
                       "inapplicable rather than forgotten."),
            "attempts_used": None, "refused_attempts": []},
        "why_one_init": (
            "doc 63 SS7.4, three registered grounds: the committed drift anchor "
            "requires the uniform init (SS6.4); a second init doubles a 120-run "
            "design to 220+ for a variable that has not replicated (doc 58 "
            "SS20.1 records the identity-init headline dying on its confirm "
            "draw); and doc 62 SS3.5 forbids merging init results across rungs "
            "anyway, so a fourth unmergeable init point buys nothing. SS21 item "
            "5 offers the reduction that would restore it.")}
    return rep


# ==================================================================== training
def train(arm, seed, store, verbs, max_epochs, fm, save=True, tag="",
          model=None, quiet=False, certificate=True):
    """The exp60 / exp61 / exp65 / exp68 / exp69 loop, UNCHANGED -- Adam(lr
    0.02), batch 8, 150 epochs, checkpoint selected on in-distribution val
    accuracy, held-out read ONCE from the selected checkpoint.

    The call order (compile -> manual_seed -> build -> Adam -> Random(RNG_BASE +
    seed)) is exp69's verbatim, because SS6.4 expects the anchor to reproduce
    exp69 BIT-EXACTLY and any reordering of RNG consumption would break it."""
    cfg = ARM_CFG[base_arm(arm)]
    items, meta, path = arm_items(arm, store)
    data = compile_arm(arm, items, seed, fm)
    torch.manual_seed(seed)
    if model is None:
        m = build_model(arm, verbs, seed, fm)
    else:
        m = model
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    rng = random.Random(RNG_BASE + seed)

    # SS9.4 row 4 -- the certificate at INIT, measured separately, so "training
    # is what makes the state hard" is a comparison and not an assumption. No
    # RNG is consumed (no_grad, no draws).
    cert_init = None
    if certificate and save:
        _acc0, rows0 = evaluate_with_traces(m, data["test"], certificate=True)
        cert_init = certificate_summary(rows0, data["test"][0].K)
        cert_init["heldout_acc_at_init"] = _acc0

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

    m.load_state_dict(best["state"])
    te, rows = evaluate_with_traces(m, data["test"], certificate=certificate)
    cert = certificate_summary(rows, data["test"][0].K) if certificate else None

    pt = sha = None
    if save:
        os.makedirs(CKPT_DIR, exist_ok=True)
        pt = os.path.join(CKPT_DIR, "%s%s_s%d.pt" % (tag, arm, seed))
        torch.save(best["state"], pt)                     # doc 45 SS5
        sha = sha256_file(pt)
    return {
        "arm": arm, "base_arm": base_arm(arm), "kind": cfg["kind"],
        "cell": cfg["cell"], "axis": cfg["axis"],
        "is_floor": arm.startswith("S-shuffle"),
        "seed": seed, "n": cfg["n"], "T": cfg["T"],
        "n_active": min(cfg["n"], cfg["T"] + 1),
        "inventory": cfg["inventory"], "category": cfg["category"],
        "encoding": cfg["enc"], "init": cfg["init"], "role": cfg["role"],
        "init_scheme": {
            "scheme": "uniform",
            "recipe": ("every trainable block ~ Uniform[0, 2pi) via the "
                       "committed `Circuit4` constructor -- doc 63 SS7.4, the "
                       "COMMITTED init and the only one this rung registers"),
            "bounded_redraw_doc62_SS24_3_applicable": False,
            "bounded_redraw_reason": (
                "no exp76 arm calls exp72_run.apply_identity_init, so the "
                "committed identity window is never consulted and the "
                "driver-level bounded redraw has nothing to absorb"),
            "redraw_attempts_used": None, "redraw_refused_attempts": []},
        "train_acc": best["train"], "val_acc": best["val"],
        "selected_epoch": best["epoch"], "heldout_test_acc": te,
        "val_minus_heldout": best["val"] - te,
        "gate_met": bool(best["train"] >= TRAIN_GATE and best["val"] >= VAL_GATE),
        "n_params": sum(p.numel() for p in m.parameters()),
        "wires": data["test"][0].K, "qubits_per_referent": 1,
        "n_train": len(data["train"]), "n_val": len(data["val"]),
        "n_test": len(data["test"]),
        "max_epochs": max_epochs, "lr": LR, "batch": BATCH,
        "dataset": path, "dataset_sha256": meta["dataset_sha256"],
        "ckpt_path": pt, "ckpt_sha256": sha,
        "certificate_doc63_SS9": cert,
        "certificate_at_init_doc63_SS9_4": cert_init,
        "heldout_items": rows,
        "spec": SPEC, "script_sha256": _SELF_SHA,
        "wall_sec": time.time() - t0}


# ==================================================================== the smoke
def smoke(fm, verbs, store, tiny_epochs=2):
    t0 = time.time()
    rep = {"experiment": "exp76 -- scaling the composition-free family (doc 63)",
           "spec": SPEC,
           "orchestrator_rulings": {
               "1": "the four measurement-corrected deviations are RATIFIED",
               "2": ("the inertness-theorem grid at cast sizes {2,3,4,5,7} with "
                     "the padded case as a BLOCKING assertion and n = 16 "
                     "recorded as structurally obstructed"),
               "3": "constructive-with-exact-bridge at 519",
               "4": "the growth-slope certificate a <= 0.10 bits/wire",
               "5": "120 runs ACCEPTED",
               "6": ("SS3.6.2's narrowed conclusion RATIFIED after the bands_c "
                     "correction: arc count excluded for all five classes; the "
                     "semantic-coherence correlate remains OPEN and SS16's wazn "
                     "prohibition holds on those grounds"),
               "7": "no directional call; record 1 held / 2 missed"},
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "numpy": np.__version__,
           "script_sha256": _SELF_SHA}
    check_datasets(store, rep)
    check_committed_imports(rep)
    check_grid(rep)
    check_init_scheme(rep)
    check_features(fm, rep)
    check_anchor_codepath(fm, verbs, store, rep)
    check_parameters(fm, verbs, store, rep)
    check_intro(fm, verbs, store, rep)
    check_copy(fm, verbs, store, rep)
    check_closures(fm, verbs, store, rep)
    check_padding(fm, verbs, store, rep)
    check_active_structure(fm, store, rep)
    check_trace(fm, verbs, store, rep)
    check_comparator_frame(fm, store, rep)
    check_shuffle(store, rep)
    # SS10.4 -- asserted at SMOKE, as a DATASET property, before any training,
    # together with its failing direction (a core re-drawn per n must FAIL).
    rep["comparator_flat_in_n_doc63_SS10_4"] = D76.check_shared_core(fm["exp76"])
    print("[smoke] comparator_flat_in_n: tier (ii) bit-identical across n",
          flush=True)

    probe = {}
    for arm in ALL_ARMS:
        t1 = time.time()
        r = train(arm, 0, store, verbs, tiny_epochs, fm, save=False, quiet=True,
                  certificate=True)
        dt = time.time() - t1
        assert r["n_params"] == N_PARAMS, \
            "SS8.4 item 2: %s trained with %d parameters" % (arm, r["n_params"])
        probe[arm] = {"epochs": tiny_epochs, "sec": dt,
                      "sec_per_epoch": dt / tiny_epochs,
                      "val": r["val_acc"], "heldout": r["heldout_test_acc"],
                      "n_params": r["n_params"], "wires": r["wires"],
                      "n_active": r["n_active"],
                      "S_max": (r["certificate_doc63_SS9"]["S_max_overall"]
                                if r["certificate_doc63_SS9"] else None)}
        print("[smoke] %-24s %dep val=%.3f test=%.3f %.0fs  (%d params, n=%d, "
              "T=%d, S_max=%.3f)"
              % (arm, tiny_epochs, r["val_acc"], r["heldout_test_acc"], dt,
                 r["n_params"], r["n"], r["T"],
                 probe[arm]["S_max"] or 0.0), flush=True)
    rep["probe"] = probe

    # SS19.3 / SS20.3 step 9 -- the realised per-run cost, reported against the
    # registration's measured factors, with SS19.4's contention factor applied.
    ser = sum(p["sec_per_epoch"] * MAX_EPOCHS for p in probe.values()) * N_SEEDS
    rep["projected_wall_time_doc63_SS19"] = {
        "measured_sec_per_epoch": {a: p["sec_per_epoch"] for a, p in probe.items()},
        "projected_sec_per_run": {a: p["sec_per_epoch"] * MAX_EPOCHS
                                  for a, p in probe.items()},
        "serial_hours_single_process": ser / 3600.0,
        "contention_factor_of_record": 2.0,
        "hours_on_5_workers_uncontended": ser / 5.0 / 3600.0,
        "hours_on_5_workers_with_contention": ser / 5.0 / 3600.0 * 2.0,
        "doc63_SS19_3_table_estimate_hours": 5.4,
        "note": ("doc 63 SS19.4: the committed walls were measured UNDER 5 "
                 "PARALLEL WORKERS, so pool contention is already inside them. "
                 "The measured factor of record -- 2.0x, settled on completed "
                 "anchor runs (exp73_LAUNCH_REPORT.md SS3) -- applies to any "
                 "single-process smoke projection exp76 produces.")}

    rep["failing_directions_doc63"] = {
        "all_wires_readout_must_fail_the_padding_control":
            rep["inert_padding_doc63_SS8_3"]
               ["failing_direction_all_wires_readout"]["fires"],
        "second_RY_must_fail_the_3_term_closure":
            rep["fourier_closure_doc63_SS8_4_7"]
               ["failing_direction_second_RY"]["fires"],
        "entangled_intro_must_fail_the_purity_assertion":
            rep["intro_doc63_SS8_4_3_and_5"]
               ["failing_direction_entangled_intro"]["fires"],
        "a_redrawn_core_must_fail_comparator_flat_in_n":
            rep["comparator_flat_in_n_doc63_SS10_4"]["_failing_direction"]["fires"],
        "a_padded_item_must_fail_V9":
            rep["active_structure_doc63_SS8_2_V9"]
               ["failing_direction_padded_item"]["fires"],
        "the_wire_ordered_comparator_frame_is_a_null_instrument":
            rep["comparator_frame_doc63_SS22"]["wire_frame_is_a_null_instrument"],
    }
    assert all(rep["failing_directions_doc63"].values()), \
        "a designed-to-fail build did not fail -- the gate set is not live"

    rep["assertions_passed"] = True
    rep["wall_time_sec"] = time.time() - t0
    json.dump(rep, open(SMOKE_OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[smoke] wrote %s -- ALL doc 63 SS8 / SS10.4 ASSERTIONS PASSED, "
          "INCLUDING THE SIX DESIGNED TO FAIL" % SMOKE_OUT)
    return rep


# ============================================================== the anchor gate
def anchor_report():
    """SS6.4 / SS12.1 -- BLOCKING, READ FIRST. Per-seed held-out accuracies must
    reproduce exp69's R1-A and R2-A to |delta| <= 0.002 on every seed and each
    arm mean to |delta| <= 0.002. The honest expectation is |delta| = 0 EXACTLY.

    IF THIS FAILS, the exp76 pipeline has drifted, every scaling number is a
    comparison against a moving baseline, and NOTHING ELSE IN THE RUN IS READ.
    Hunt the build; interpret nothing."""
    src = json.load(open(EXP69_RESULTS, encoding="utf-8"))
    rep = {"component": "exp76 drift anchor (doc 63 SS6.4), the SIXTH enforced "
                        "use of this gate; the first five returned bit-exact",
           "spec": SPEC,
           "rule3_note": ("its own file by Rule 3; no exp69 artefact is "
                          "modified and no results file is overwritten"),
           "dataset": DATA_EXP69, "dataset_sha256": DATASET_SHA_DOC63,
           "exp69_source": EXP69_RESULTS, "tolerance": ANCHOR_TOLERANCE,
           "per_arm": {}}
    all_ok = all_exact = True
    for arm in ANCHOR_ARMS:
        key = ANCHOR_SOURCE[arm]
        ref = src["arms"][key]
        got, val, tr, ep, wall = [], [], [], [], 0.0
        for s in range(N_SEEDS):
            f = os.path.join(CKPT_DIR, "%s_s%d.json" % (arm, s))
            if not os.path.exists(f):
                raise RuntimeError("the drift anchor is incomplete: %s is "
                                   "missing" % f)
            r = json.load(open(f, encoding="utf-8"))
            got.append(float(r["heldout_test_acc"]))
            val.append(float(r["val_acc"]))
            tr.append(float(r["train_acc"]))
            ep.append(int(r["selected_epoch"]))
            wall += float(r["wall_sec"])
        want = [float(x) for x in ref["per_seed"]]
        d = [g - w for g, w in zip(got, want)]
        mean = float(np.mean(got))
        ok_seed = all(abs(x) <= ANCHOR_TOLERANCE for x in d)
        ok_mean = abs(mean - ANCHOR_MEAN_DOC63[arm]) <= ANCHOR_TOLERANCE
        exact = all(x == 0.0 for x in d)
        all_ok = all_ok and ok_seed and ok_mean
        all_exact = all_exact and exact
        rep["per_arm"][arm] = {
            "exp69_source_arm": key, "exp69_per_seed": want,
            "exp69_mean": float(ref["mean"]), "exp69_sd": float(ref["sd"]),
            "exp76_per_seed": got, "exp76_mean": mean,
            "exp76_sd": float(np.std(got, ddof=1)),
            "per_seed_delta": d, "max_abs_per_seed_delta": float(max(map(abs, d))),
            "mean_delta_vs_exp69": mean - float(ref["mean"]),
            "registered_mean_doc63_SS6_4": ANCHOR_MEAN_DOC63[arm],
            "mean_delta_vs_registered": mean - ANCHOR_MEAN_DOC63[arm],
            "bit_exact": exact, "per_seed_within_tolerance": ok_seed,
            "mean_within_tolerance": ok_mean,
            "replicates": bool(ok_seed and ok_mean),
            "val_per_seed": val, "train_per_seed": tr,
            "selected_epoch_per_seed": ep, "n_params": N_PARAMS,
            "wall_sec_total": wall}
    rep["n_runs"] = len(ANCHOR_ARMS) * N_SEEDS
    rep["bit_exact_all"] = all_exact
    rep["anchor_replicates_exp69"] = bool(all_ok)
    rep["same_entry_point_doc63_SS8_4_1"] = (
        "both arms run through exp76's OWN n-generalised codepath at n = 2, on "
        "exp69's frozen dataset with exp69's frozen features. SS8.4 item 1 "
        "asserts the forward pass bit-for-bit against the committed "
        "exp42_models path at smoke, so the anchor tests the TRAINING loop.")
    rep["honest_expectation"] = (
        "doc 63 SS6.4: |delta| = 0 EXACTLY, bit-exact, 10/10, and here that "
        "expectation is STRONGER than in any previous use of this gate. The "
        "0.002 tolerance exists only to absorb a documented float/BLAS platform "
        "difference, and any non-zero delta is reported with its cause "
        "identified.")
    rep["consequence_if_failed"] = (
        "the exp76 pipeline has drifted, every scaling number is a comparison "
        "against a moving baseline, and NOTHING ELSE IN THE RUN IS READ. "
        "SS6.4.1's floor-omission argument is void along with the run. Hunt the "
        "build; interpret nothing.")
    return rep


def job_list(arms, seeds):
    """SS20.4(a) -- THE DRIFT ANCHOR RUNS FIRST. The two anchor arms' ten seeds
    are the first ten jobs, so with `--nworkers 5` every worker's first two jobs
    are anchor seeds and the blocking gate completes before any grid cell
    finishes."""
    anchor = [(a, s) for a in ANCHOR_ARMS if a in arms for s in range(seeds)]
    rest = [(a, s) for a in arms if a not in ANCHOR_ARMS for s in range(seeds)]
    return anchor + rest


_SELF_SHA = hashlib.sha256(
    open(os.path.abspath(__file__), "rb").read()).hexdigest()


def main():
    global CKPT_DIR, SMOKE_OUT, ANCHOR_OUT
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(ALL_ARMS),
                    help="default is doc 63 SS19.2's full 120-run set: 10 anchor "
                         "runs, 40 grid, 5 grade-3, 10 inventory, 55 floors")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-epochs", type=int, default=2)
    ap.add_argument("--anchor-report", action="store_true",
                    help="read the ten anchor checkpoints and write %s" % ANCHOR_OUT)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--smoke-out", default=SMOKE_OUT)
    ap.add_argument("--anchor-out", default=ANCHOR_OUT)
    a = ap.parse_args()
    CKPT_DIR, SMOKE_OUT, ANCHOR_OUT = a.ckpt_dir, a.smoke_out, a.anchor_out

    store = load_store()
    verbs = sorted(store["exp69"][1]["verb_inventory"].keys())
    assert len(verbs) == 6, \
        ("doc 63 SS7.3's parameter arithmetic assumes 6 verbs and SS4.3's T <= 6 "
         "ceiling is what keeps the native-speaker gate closed; the dataset "
         "carries %d" % len(verbs))
    for cell in TRAINED_CELLS:
        cv = sorted(store[cell][1]["verb_inventory"].keys())
        assert cv == verbs, "verb-inventory drift at %s" % cell
    fm = feature_maps()
    os.makedirs(CKPT_DIR, exist_ok=True)

    if a.smoke:
        smoke(fm, verbs, store, tiny_epochs=a.smoke_epochs)
        return 0

    if a.anchor_report:
        rep = anchor_report()
        json.dump(rep, open(ANCHOR_OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n[ANCHOR GATE] doc 63 SS6.4, sixth enforced use")
        for arm, r in sorted(rep["per_arm"].items()):
            print("  %-10s exp76 %s" % (arm, ["%.6f" % x for x in
                                              r["exp76_per_seed"]]))
            print("  %-10s exp69 %s  max|d|=%.2e  bit_exact=%s"
                  % ("", ["%.6f" % x for x in r["exp69_per_seed"]],
                     r["max_abs_per_seed_delta"], r["bit_exact"]))
        print("  anchor_replicates_exp69 = %s  ->  %s"
              % (rep["anchor_replicates_exp69"],
                 "PROCEED" if rep["anchor_replicates_exp69"]
                 else "THE RUN STOPS (doc 63 SS6.4)"))
        return 0 if rep["anchor_replicates_exp69"] else 1

    if not os.path.exists(SMOKE_OUT):
        raise RuntimeError(
            "%s missing -- doc 63 SS20.3 step 9: every SS8 and SS10.4 assertion, "
            "INCLUDING THE SIX DESIGNED TO FAIL, runs BEFORE any training. Run "
            "`python exp76_run.py --smoke`." % SMOKE_OUT)
    sm = json.load(open(SMOKE_OUT, encoding="utf-8"))
    if not sm.get("assertions_passed"):
        raise RuntimeError("the exp76 smoke gate did not pass -- nothing may train")

    arms = [x for x in a.arms.split(",") if x]
    unknown = [x for x in arms if base_arm(x) not in ARM_CFG]
    assert not unknown, "unknown arms: %r" % unknown
    jobs = job_list(arms, a.seeds)
    mine = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    print("[exp76 w%d/%d] arms=%d -> %d of %d runs (anchor first)"
          % (a.worker, a.nworkers, len(arms), len(mine), len(jobs)), flush=True)
    for arm, s in mine:
        ckj = os.path.join(CKPT_DIR, "%s%s_s%d.json" % (a.tag, arm, s))
        if os.path.exists(ckj):
            print("[ckpt] %s%s_s%d already done" % (a.tag, arm, s), flush=True)
            continue
        r = train(arm, s, store, verbs, MAX_EPOCHS, fm, save=True, tag=a.tag)
        r["tag"] = a.tag
        json.dump(r, open(ckj, "w", encoding="utf-8"))
        cert = r["certificate_doc63_SS9"]
        print("[exp76] %-24s s%d  train=%.3f val=%.3f HELD-OUT=%.4f  "
              "n=%d T=%d S_max=%.3f R_max=%d  (%.0fs)"
              % (arm, s, r["train_acc"], r["val_acc"], r["heldout_test_acc"],
                 r["n"], r["T"], (cert["S_max_overall"] if cert else 0.0),
                 (cert["R_max_overall"] if cert else 0), r["wall_sec"]),
              flush=True)
    print("[exp76 w%d] done" % a.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
