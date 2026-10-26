# -*- coding: utf-8 -*-
"""exp68_run.py -- doc 54: the WIDTH lever, and the close of the three-lever ledger.

THE QUESTION (doc 54 SS1). Frequency K (exp61/62/63) and readout arity M (exp67)
widened the function class over ONE number and neither converted into held-out
accuracy. WIDTH is the only lever that widens WHAT THE MODEL IS TOLD, and until
`aravec_raw_vectors_44.json` arrived the frozen inventory had no second number to
give: on unit-normalised 2D vectors PC1 = cos(phi) and PC2 = sin(phi) EXACTLY, so
a "two-feature register" there is a reparameterisation of one number (doc 54
SS3.1). On the RAW plane it is not -- FA carries the angle, F2 carries angle AND
radius -- so exp68 is width-as-INFORMATION (doc 54 SS3.2).

ARMS (doc 54 SS6.2), all FROZEN LOOKUPS -- no train-time featurizer fit exists:

  F1  theta = (p1 + 1) pi on one qubit                  1 qubit/ref, d = 3
  F2  theta0 = (p1 + 1) pi on qubit 0 of the register,
      theta1 = (p2 + 1) pi on qubit 1                   2 qubits/ref, d = 9
  FA  phi = atan2(PC2, PC1) -- THE EXISTING exp42 MAP   1 qubit/ref, d = 3
      information-restricted control, and doc 50 Part A's never-run T=1 anchor
  S-shuffle-{F1,F2,FA}   the registered floor (doc 46 SS10.2): geometry
      bit-identical to the arm it matches, bearer designation for ANSWERS
      reassigned PER PAIR independently of the features

  p_k(r) = PC_k(r) / max_{r' in the 24 humans} |PC_k(r')|,  PCA over all 44
  referents, centred, PC signs PINNED to the frozen vec2 plane (doc 54 SS3.2 --
  bounds are sign-invariant, a TRAINED CIRCUIT IS NOT).

THREE FLOOR ARMS, NOT ONE -- a deliberate, recorded departure from doc 54 SS13's
"~20 runs (4 arms x 5 seeds)" cost arithmetic. doc 54 SS7 states the floor gate
twice as "its MATCHED S-shuffle floor" and "every arm must exceed ITS MATCHED
S-shuffle mean". The floor is architecture-dependent here in a way it was not in
exp61, where every arm shared one circuit family: F2 is a different circuit with
an 81-dimensional joint class against ~190 constraints, which is precisely the
regime doc 52 SS2.2 showed INTERPOLATES shuffled labels. A single floor borrowed
from the 2-wire arms would be an unmatched comparator -- the exp44 failure class
this programme keeps recording -- and would make F2's gate too easy. SS13's "~20"
is a cost estimate; SS7 is a frozen criterion, so SS7 governs: 6 arms x 5 seeds =
30 runs. Extra floors can only make the gates STRICTER; they cannot manufacture a
positive.

FEATURES ARE IMPORTED FROM THE COMMITTED INSTRUMENT, NEVER RETYPED.
`instrument_sb_bounds.load_raw()` is the module that produced doc 54 SS12's
bounds, so the arms and the bounds cannot drift apart. FA's runtime angles come
from `exp42_noun_embeddings.json` itself, because doc 54 SS6.2 defines FA as
"phi = atan2(PC2, PC1) -- the existing exp42 map" and doc 54 SS3.2 establishes
that those are the same quantity; `--smoke` asserts the agreement rather than
trusting it. See SS6.5 note (2) in `smoke()` for the one place the instrument's
internal FA differs from the doc's formula and why the bound is unaffected.

CIRCUIT (doc 54 SS6.4). F1/FA are `exp42_models.QuantumStoryModel` ("A3"), the
2-wire model exp60/exp61 use. F2 is `exp45_model.TwoQubitReferentModel` ("A4",
doc 28) with an `exp62_c_run.WModel`-shaped `story_values` override -- W-wide
gave the second qubit the SAME angle (frequency, K = 2); F2 gives it a DIFFERENT
FEATURE (width, capacity). doc 54 SS6.4.1 makes copy-verification mandatory:
with the second injection disabled the F2 forward pass must reproduce its parent
BIT-FOR-BIT.

    python exp68_run.py --smoke                       # doc 54 SS6.5 validity gate
    python exp68_run.py --worker i --nworkers 5       # the 30 runs
    python exp68_comparator.py                        # doc 54 SS8, the ARBITER
    python exp68_aggregate.py                         # the ONLY results writer
"""
import argparse, copy, hashlib, json, math, os, platform, random, sys, time

import numpy as np
import torch

torch.set_num_threads(1)                              # doc 45 SS6.1, verified inert

import exp42_compiler as comp
import exp42_sim_adapter as sim
import instrument_sb_bounds as INS
from exp42_models import QuantumStoryModel
from exp45_model import TwoQubitReferentModel
from exp42_run import grad_global_norm
from exp60_run import relabel_shuffle, evaluate
from exp61_run import fourier_resid

DESIGN_DOC = "qnlp_private_docs/54_exp68_width_lever.md (1025d21, FROZEN)"
DATA = "stories_exp68_WN.json"
CKPT_DIR = "exp68_ckpt"
SMOKE_OUT = "results_exp68_smoke_validation.json"

# Training config: the exp60 / exp61 / exp65 line, unchanged.
LR, BATCH, MAX_EPOCHS = 0.02, 8, 150
TRAIN_GATE, VAL_GATE = 0.8, 0.7                       # recorded, never a criterion
RNG_BASE = 55000
N_SEEDS = 5                                           # doc 54 SS6.2: OPTIMISER INIT ONLY

BASE_ARMS = ["F1", "F2", "FA"]
ARMS = BASE_ARMS + ["S-shuffle-%s" % a for a in BASE_ARMS]
FLOOR_OF = {a: "S-shuffle-%s" % a for a in BASE_ARMS}
SCALING = INS.DEFAULT_SCALING                          # "full": theta = (p + 1) pi

# doc 54 SS12, CONTEXT ONLY -- no criterion is read against these (doc 54 SS7).
# Used for one thing: doc 54 SS9.2's bug-hunt row.
PER_TWIN_BOUNDS = {"F1": 0.7625, "F2": 1.0000, "FA": 0.9250}
BUG_HUNT_MARGIN = 0.06


def base_arm(arm):
    return arm[len("S-shuffle-"):] if arm.startswith("S-shuffle-") else arm


def _u(t):
    """K = 1 single-angle span (1, cos t, sin t) -- the instrument's `_u1`."""
    return (1.0, math.cos(t), math.sin(t))


# --------------------------------------------------------------- feature layer
def feature_maps():
    """The three registered per-referent maps, and their provenance.

    Everything numeric here comes from committed artefacts through committed
    code: `instrument_sb_bounds.load_raw` for the raw PCA plane (sign pin,
    max-abs normalisation, both registered in doc 54 SS6.2) and
    `exp42_compiler.load_embeddings` for FA."""
    coords, prov = INS.load_raw(INS.RAW_EMB, fit_on="all44")
    frozen_ang, emb_meta, emb_sha = comp.load_embeddings()
    scale = INS.SCALINGS[SCALING][1]
    th1 = {n: float(scale(coords[n][0])) for n in INS.HUMANS}     # theta0 <- PC1
    th2 = {n: float(scale(coords[n][1])) for n in INS.HUMANS}     # theta1 <- PC2
    fa = {n: float(frozen_ang[n]) for n in INS.HUMANS}
    return {"coords": coords, "prov": prov,
            "theta_pc1": th1, "theta_pc2": th2, "phi_exp42": fa,
            "embeddings_sha256": emb_sha, "embeddings_meta": emb_meta,
            "scaling": {"name": SCALING, "formula": INS.SCALINGS[SCALING][0]}}


def arm_angles(arm, fm):
    """(angles consumed by exp42_compiler, second-qubit angles or None)."""
    b = base_arm(arm)
    if b == "F1":
        return fm["theta_pc1"], None
    if b == "FA":
        return fm["phi_exp42"], None
    if b == "F2":
        return fm["theta_pc1"], fm["theta_pc2"]
    raise ValueError(arm)


def referent_features(arm, fm):
    """The per-referent u-vector each arm's circuit is linear in -- d = 3 for
    F1/FA, d = 9 for F2. These are EXACTLY the vectors doc 54 SS12's bounds were
    computed over, and exactly what doc 54 SS8's comparator must consume.

    WHY d = 9 IS THE RIGHT NUMBER FOR F2, in one line: RY(theta)|0> has density
    matrix linear in (1, cos theta, sin theta), so a two-qubit product register
    has density matrix linear in u(theta0) (x) u(theta1) -- nine terms -- and any
    outcome probability is linear in it. The instrument's feature map IS the
    circuit's function class, which is why SS6.5's closure test can be exact."""
    a0, a1 = arm_angles(arm, fm)
    if a1 is None:
        return {n: np.array(_u(a0[n])) for n in INS.HUMANS}
    return {n: np.outer(np.array(_u(a0[n])), np.array(_u(a1[n]))).ravel()
            for n in INS.HUMANS}


# ---------------------------------------------------------------- the F2 model
class F2Model(TwoQubitReferentModel):
    """doc 54 SS6.4: `exp45_model.TwoQubitReferentModel` ("A4") with an
    `exp62_c_run.WModel`-shaped override. The ONLY change from the parent is the
    second RY on qubit 2w+1, carrying a DIFFERENT feature.

    `theta1` is DATA, not a Parameter, and no generator is consumed after
    `super().__init__`, so `wide=False` must reproduce the parent bit-for-bit
    (doc 54 SS6.4.1). `theta1` is mutable so SS6.5's 2-D closure sweep can drive
    the second qubit; that is the only thing that ever writes to it."""

    def __init__(self, verbs, seed, theta1, layers=3, wide=True):
        super().__init__("A4", verbs, seed, layers=layers)
        self.theta1 = dict(theta1)
        self.wide = wide

    def story_values(self, cs, verb_map=None):
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)
            return cache[v]

        psi = sim.init_state(2 * cs.K)
        for w, noun, angle in cs.intro:
            psi = sim.apply_gate(psi, sim.ry(angle), (2 * w,))
            if self.wide:                     # doc 54 SS6.4: WIDTH, DIFFERENT theta
                psi = sim.apply_gate(psi, sim.ry(self.theta1[noun]), (2 * w + 1,))
        for v, wa, wp in cs.events:
            psi = sim.apply_gate(psi, block(v), self._reg(wa) + self._reg(wp))
        vseq, qa, qp = cs.question
        qw = self._reg(qa) + self._reg(qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)
        return torch.stack([sim.prob_all_zero_on(
            sim.apply_gate(psi, eff.matrix().conj().T, qw), qw)
            for eff in (self.q_pos, self.q_neg)])


def build_model(arm, verbs, seed, fm):
    if base_arm(arm) == "F2":
        return F2Model(verbs, seed, fm["theta_pc2"])
    return QuantumStoryModel("A3", verbs, seed)


def compile_arm(arm, items, seed, fm):
    ang, _t1 = arm_angles(arm, fm)
    use = relabel_shuffle(items, None, seed) if arm.startswith("S-shuffle") else items
    return {s: comp.compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}


# -------------------------------------------------------- doc 54 SS6.5 / SS6.4.1
def _independent_raw_coords():
    """Recompute the raw PCA plane straight from the committed source files, so
    the smoke's bit-match is a check and not a tautology. Same registered recipe
    as doc 54 SS6.2: PCA over all 44 referents, centred, signs pinned to the
    frozen vec2 plane, per-component max-abs normalisation over the 24 humans."""
    blob = json.load(open(INS.RAW_EMB, encoding="utf-8"))
    vecs = blob["vectors"]
    names = sorted(vecs)
    X = np.array([vecs[n] for n in names], dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    _U, _S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Pall = Xc @ Vt[:2].T
    idx = {n: i for i, n in enumerate(names)}
    P = np.array([Pall[idx[n]] for n in INS.HUMANS])
    fz = json.load(open(INS.FROZEN_EMB, encoding="utf-8"))["embeddings"]
    F = np.array([fz[n]["vec2"] for n in INS.HUMANS], dtype=float)
    U2 = P / np.linalg.norm(P, axis=1, keepdims=True)
    best, dmin = None, None
    for sa in (1, -1):
        for sb in (1, -1):
            dd = float(np.abs(U2 * np.array([sa, sb]) - F).max())
            if dmin is None or dd < dmin:
                dmin, best = dd, (sa, sb)
    P = P * np.array(best, dtype=float)
    denom = np.abs(P).max(axis=0)
    ang_frozen = np.array([fz[n]["angle"] for n in INS.HUMANS])
    return {"P_signed": P, "Pn": P / denom, "denom": denom, "sign_pin": list(best),
            "unit_vs_frozen_vec2_max_abs": dmin,
            "atan2_vs_frozen_angle_max_rad": float(np.abs(np.angle(np.exp(
                1j * (np.arctan2(P[:, 1], P[:, 0]) - ang_frozen)))).max()),
            "blob_declared_vectors_sha256": blob["meta"]["vectors_sha256"],
            "recomputed_vectors_sha256": comp.canonical_sha256(vecs)}


def check_features(fm, rep):
    """doc 54 SS6.5 (3) + the task's bit-match obligation, both directions."""
    ind = _independent_raw_coords()
    diag = json.load(open(INS.OUT, encoding="utf-8"))
    Pn_ins = np.array([fm["coords"][n] for n in INS.HUMANS])

    d_coord = float(np.abs(ind["Pn"] - Pn_ins).max())
    assert d_coord == 0.0, \
        ("runtime coords differ from an independent recomputation by %.3e -- "
         "the feature map is not what doc 54 SS6.2 registered" % d_coord)
    assert ind["blob_declared_vectors_sha256"] == ind["recomputed_vectors_sha256"], \
        "aravec_raw_vectors_44.json declared sha != canonical_sha256 of its vectors"

    d_div = float(np.abs(np.array(ind["denom"])
                         - np.array(diag["families"]["raw"]["norm_divisors"])).max())
    assert d_div < 1e-12, "norm divisors drifted from diagnostic_sb_bounds.json"
    assert abs(ind["denom"][0] - 8.669495) < 5e-6 and \
        abs(ind["denom"][1] - 10.160404) < 5e-6, \
        "norm divisors are not doc 54's PC1 8.669495 / PC2 10.160404"
    assert list(ind["sign_pin"]) == diag["families"]["raw"][
        "plane_agreement_with_frozen2d"]["sign_pin_applied"], "sign pin drifted"

    # FA: doc 54 SS6.2 defines it as atan2(PC2, PC1) = "the existing exp42 map".
    # doc 54 SS3.2's load-bearing fact, re-asserted here rather than trusted.
    phi_run = np.array([fm["phi_exp42"][n] for n in INS.HUMANS])
    phi_pc = np.arctan2(ind["P_signed"][:, 1], ind["P_signed"][:, 0])
    d_fa = float(np.abs(np.angle(np.exp(1j * (phi_run - phi_pc)))).max())
    assert d_fa < 1e-12, "FA runtime angle != atan2(PC2, PC1): %.3e" % d_fa
    assert ind["unit_vs_frozen_vec2_max_abs"] < 1e-12, \
        "unit-normalised raw PCA plane does not reproduce exp42_noun_embeddings"

    # SS6.5 (3): the second feature must be NON-DEGENERATE on the inventory.
    p1, p2 = Pn_ins[:, 0], Pn_ins[:, 1]
    rho = np.sqrt(p1 ** 2 + p2 ** 2)
    pear = float(np.corrcoef(p1, p2)[0, 1])
    poly = {}
    for deg in (1, 3, 5):
        V = np.vander(p1, deg + 1)
        c, *_ = np.linalg.lstsq(V, p2, rcond=None)
        poly["deg%d_max_resid" % deg] = float(np.abs(V @ c - p2).max())
    q2 = np.array([INS.load_frozen2d()[0][n] for n in INS.HUMANS])
    rho_frozen = np.sqrt((q2 ** 2).sum(1))
    assert float(rho.std(ddof=1)) > 0.05, "raw radius is degenerate"
    assert abs(pear) < 0.99, "p1 and p2 are collinear"
    assert poly["deg5_max_resid"] > 0.10, \
        "p2 is a polynomial function of p1 -- the second feature adds nothing"
    assert float(rho_frozen.std(ddof=1)) < 1e-9, \
        "frozen2d is not unit-norm -- doc 54 SS3.1's contrast does not hold"

    rep["features"] = {
        "runtime_source": ("instrument_sb_bounds.load_raw(fit_on='all44') for "
                           "PC1/PC2; exp42_compiler.load_embeddings() for FA"),
        "bitmatch_vs_independent_recomputation": d_coord,
        "bitmatch_is_exact": bool(d_coord == 0.0),
        "norm_divisors": [float(x) for x in ind["denom"]],
        "norm_divisors_vs_diagnostic_max_abs": d_div,
        "sign_pin_applied": list(ind["sign_pin"]),
        "FA_runtime_vs_atan2_PC2_PC1_max_rad": d_fa,
        "unit_raw_pca_vs_frozen_vec2_max_abs": ind["unit_vs_frozen_vec2_max_abs"],
        "theta0_range": [float(((p1 + 1) * math.pi).min()),
                         float(((p1 + 1) * math.pi).max())],
        "theta1_range": [float(((p2 + 1) * math.pi).min()),
                         float(((p2 + 1) * math.pi).max())],
        "nondegeneracy": {
            "raw_radius_mean": float(rho.mean()), "raw_radius_sd": float(rho.std(ddof=1)),
            "raw_radius_min": float(rho.min()), "raw_radius_max": float(rho.max()),
            "frozen2d_radius_sd": float(rho_frozen.std(ddof=1)),
            "pearson_p1_p2": pear, "poly_fit_of_p2_on_p1": poly,
            "verdict": ("p2 is NOT a function of p1 on this inventory: a degree-5 "
                        "polynomial leaves a max residual of %.3f on a feature "
                        "whose range is [-1, 1], and the radius has sd %.3f where "
                        "the frozen2d family has sd %.1e (it is exactly unit by "
                        "construction). doc 54 SS3.1's degeneracy -- the one that "
                        "made the frozen2d version of this experiment a "
                        "reparameterisation -- does not hold on the raw family, "
                        "so exp68 is width-as-INFORMATION."
                        % (poly["deg5_max_resid"], rho.std(ddof=1),
                           rho_frozen.std(ddof=1)))},
        "instrument_FA_note": (
            "doc 54 SS6.2 writes FA as atan2(PC2, PC1) -- the UNNORMALISED "
            "components, i.e. the existing exp42 map -- and that is what this "
            "runtime uses, asserted above to %.1e rad. instrument_sb_bounds."
            "features() computes FA as atan2(p2, p1) on its NORMALISED coords, "
            "which on the raw family is a different angle (max difference 7.9e-02 "
            "rad). The BOUND is unaffected and the two agree exactly at 148/160: "
            "the normalisation is a positive-diagonal linear map of the plane, "
            "which preserves the cyclic order of the referents, and a d = 3 "
            "single-angle bound depends only on that order. diagnostic_sb_bounds"
            ".json carries both computations and they match -- raw FA per-twin "
            "163/190 equals frozen2d FA per-twin 163/190 for tertiary, and the "
            "non-occupational target reads 148/160 under the raw-family FA "
            "(target_selection.table) and 148/160 under the frozen2d FA, which IS "
            "the exp42 map (membership_reconstruction). A TRAINED CIRCUIT is not "
            "invariant to the rescaling even though the bound is -- the same "
            "reasoning doc 54 SS3.2 gives for pinning the PC signs -- so the "
            "doc's literal formula governs the runtime." % d_fa),
    }
    return rep


def check_circuit(fm, verbs, data, rep):
    """doc 54 SS6.4.1 copy-verification and SS6.5 (1)-(2) closure, both directions."""
    probe = data["test"][:4]

    # ---- SS6.4.1: with the second injection disabled, F2 == its parent, exactly.
    torch.manual_seed(0)
    child = F2Model(verbs, 0, fm["theta_pc2"], wide=False)
    torch.manual_seed(0)
    parent = TwoQubitReferentModel("A4", verbs, 0)
    worst = 0.0
    with torch.no_grad():
        for cs in probe:
            worst = max(worst, float((child.story_values(cs)
                                      - parent.story_values(cs)).abs().max()))
    assert worst == 0.0, "SS6.4.1 copy-verification FAILED: max diff %.3e" % worst

    # ---- SS6.5 (1): F1 and FA close in the 3-term (K = 1) basis of their angle.
    r13 = {}
    for arm in ("F1", "FA"):
        torch.manual_seed(0)
        m = QuantumStoryModel("A3", verbs, 0)
        r13[arm] = float(fourier_resid(m, data["test"][:4], 3))
        assert r13[arm] < 1e-6, \
            "%s does not close at 3 terms (%.3e) -- the arm is not a single RY" \
            % (arm, r13[arm])

    # ---- SS6.5 (2): F2 must NOT close in the 3-term basis of theta0 ALONE, and
    # MUST close in the 9-dim product basis u(theta0) (x) u(theta1).
    #
    # Sweeping theta0 alone would close at 3 terms even for a CORRECTLY wired F2
    # (the first qubit still sees a single RY), so the sweep is 2-D over
    # (theta0, theta1) of one referent. If the second qubit were unwired, P(yes)
    # would not move with theta1 and the 3-column basis would close -- which is
    # exactly the exp60 S-reup null-instrument failure this check exists to catch.
    torch.manual_seed(0)
    f2 = F2Model(verbs, 0, fm["theta_pc2"])
    g = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    worst3, worst9, worst_dth1 = 0.0, 0.0, 0.0
    for cs in probe[:3]:
        noun = cs.intro[0][1]
        keep = f2.theta1[noun]
        ys, B3, B9 = [], [], []
        for t0 in g:
            c2 = copy.deepcopy(cs)
            c2.intro[0] = (c2.intro[0][0], noun, float(t0))
            for t1 in g:
                f2.theta1[noun] = float(t1)
                with torch.no_grad():
                    ys.append(float(f2.story_values(c2)[0]))
                u0, u1 = _u(float(t0)), _u(float(t1))
                B3.append(list(u0))
                B9.append([u0[i] * u1[j] for i in range(3) for j in range(3)])
        f2.theta1[noun] = keep
        ys = np.asarray(ys)
        for Bm, key in ((np.asarray(B3), "3"), (np.asarray(B9), "9")):
            c, *_ = np.linalg.lstsq(Bm, ys, rcond=None)
            r = float(np.abs(Bm @ c - ys).max())
            if key == "3":
                worst3 = max(worst3, r)
            else:
                worst9 = max(worst9, r)
        # direct wiring check: moving theta1 alone must move the output
        c2 = copy.deepcopy(cs)
        with torch.no_grad():
            f2.theta1[noun] = keep
            a = float(f2.story_values(c2)[0])
            f2.theta1[noun] = keep + 0.7
            b = float(f2.story_values(c2)[0])
            f2.theta1[noun] = keep
        worst_dth1 = max(worst_dth1, abs(a - b))

    assert worst3 > 1e-3, \
        ("F2 CLOSES in the 3-term basis of theta0 alone (%.3e) -- the second "
         "qubit is unwired and NO result from this arm is admissible "
         "(doc 54 SS6.5)" % worst3)
    assert worst9 < 1e-6, \
        "F2 does not close in the 9-dim product basis (%.3e)" % worst9
    assert worst_dth1 > 1e-6, "perturbing theta1 does not move F2's output"

    rep["circuit"] = {
        "copy_verification_max_abs_diff": worst,
        "copy_verification_exact_zero": bool(worst == 0.0),
        "copy_verification_note": (
            "doc 54 SS6.4.1: F2Model(wide=False) vs exp45_model."
            "TwoQubitReferentModel('A4') at the same seed, on 4 compiled test "
            "stories. Bit-for-bit is only possible because theta1 is DATA and no "
            "generator is consumed after super().__init__."),
        "F1_resid_3term": r13["F1"], "FA_resid_3term": r13["FA"],
        "F2_resid_3term_theta0_only": worst3,
        "F2_resid_9dim_product_basis": worst9,
        "F2_output_change_under_theta1_perturbation": worst_dth1,
        "grid": "12 x 12 over (theta0, theta1) in [0, 2pi), 3 test stories",
        "verdict": (
            "F1 and FA close at 3 terms (%.1e, %.1e): a single RY, nothing else "
            "changed. F2 does NOT close in the 3-term basis of theta0 alone "
            "(%.3f, four orders above the bar) and DOES close in the 9-dim "
            "product basis (%.1e). The second qubit is genuinely wired and "
            "genuinely carries a second feature, so the arm is not a null "
            "instrument. The 9-dim closure also confirms the identity the "
            "comparator relies on: the circuit is linear in u(theta0) (x) "
            "u(theta1) per referent, which is exactly doc 54 SS12's F2 feature "
            "map." % (r13["F1"], r13["FA"], worst3, worst9)),
    }
    return rep


def check_shuffle(items, fm, rep):
    """doc 54 SS6.2 / doc 46 SS10.2: the floor must sever features <-> label and
    change NOTHING else."""
    bearers = set(json.load(open(DATA, encoding="utf-8"))["meta"]["bearers"])
    rows = []
    for seed in range(N_SEEDS):
        sh = relabel_shuffle(items, None, seed)
        by = {}
        for it in sh:
            by.setdefault(it["pair_id"], []).append(it)
        opp = all(len(g) == 2 and g[0]["answer"] != g[1]["answer"]
                  for g in by.values())
        casts_same = all(
            [r["name"] for r in g[0]["referents"]] == [r["name"] for r in g[1]["referents"]]
            for g in by.values())
        n_yes = sum(1 for i in sh if i["answer"] == "yes")
        # the TRUE rule's accuracy on the shuffled labels: chance iff severed
        true_rule = sum(1 for i in sh
                        if (i["events"][0]["agent"] in bearers) == (i["answer"] == "yes"))
        rows.append({"seed": seed, "n_items": len(sh),
                     "twins_still_opposite": bool(opp),
                     "casts_unchanged": bool(casts_same),
                     "yes_fraction": n_yes / len(sh),
                     "true_bearer_rule_accuracy": true_rule / len(sh)})
    for r in rows:
        assert r["twins_still_opposite"] and r["casts_unchanged"], \
            "S-shuffle broke the twin structure -- V1 would no longer be 0.500"
        assert r["yes_fraction"] == 0.5, "S-shuffle unbalanced the classes"
        assert abs(r["true_bearer_rule_accuracy"] - 0.5) < 0.16, \
            ("S-shuffle did not sever features <-> label: the true bearer rule "
             "still scores %.4f" % r["true_bearer_rule_accuracy"])
    rep["shuffle"] = {
        "per_seed": rows,
        "mean_true_bearer_rule_accuracy":
            float(np.mean([r["true_bearer_rule_accuracy"] for r in rows])),
        "note": ("doc 46 SS10.2's floor, per PAIR: the bearer designation used for "
                 "ANSWERS is reassigned independently of the features, so the "
                 "geometry every arm sees is bit-identical to its matched arm and "
                 "only the feature <-> label correlation is removed. Twins stay on "
                 "opposite labels, so V1 stays exactly 0.500 and V3 stays intact. "
                 "The TRUE bearer rule collapses to chance on the shuffled labels, "
                 "which is the severance, measured rather than asserted. Each of "
                 "the three floor arms uses the SAME relabelling at a given seed, "
                 "so the floors differ only by architecture -- which is the point "
                 "of matching them (doc 54 SS7).")}
    return rep


def smoke(items, meta, verbs, fm):
    t0 = time.time()
    rep = {"experiment": "exp68 smoke / validity assertions (doc 54 SS6.4.1, SS6.5)",
           "spec": DESIGN_DOC, "dataset": DATA,
           "dataset_sha256": meta["dataset_sha256"],
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "numpy": np.__version__}
    check_features(fm, rep)
    data = compile_arm("F2", items, 0, fm)
    check_circuit(fm, verbs, data, rep)
    check_shuffle(items, fm, rep)

    # a 2-epoch training pass per arm, to prove every arm actually runs
    tiny = {}
    for arm in ARMS:
        r = train(arm, 0, items, verbs, 2, fm, save=False)
        tiny[arm] = {"val_acc": r["val_acc"], "heldout_test_acc": r["heldout_test_acc"],
                     "wall_sec": r["wall_sec"], "n_params": r["n_params"],
                     "wires": r["wires"]}
        print("[smoke] %-14s 2ep val=%.3f test=%.3f  %.0fs  (%d params, %d wires)"
              % (arm, r["val_acc"], r["heldout_test_acc"], r["wall_sec"],
                 r["n_params"], r["wires"]), flush=True)
    rep["two_epoch_probe"] = tiny
    rep["assertions_passed"] = True
    rep["wall_time_sec"] = time.time() - t0
    json.dump(rep, open(SMOKE_OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[smoke] wrote %s -- ALL doc 54 SS6.4.1 / SS6.5 ASSERTIONS PASSED" % SMOKE_OUT)
    return rep


# -------------------------------------------------------------------- training
def train(arm, seed, items, verbs, max_epochs, fm, save=True, tag=""):
    """The exp60 / exp61 / exp65 loop, unchanged: Adam(lr 0.02), batch 8, 150
    epochs, checkpoint selected on in-distribution val accuracy, held-out test
    read once from the selected checkpoint."""
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
    with torch.no_grad():
        te = evaluate(m, data["test"])
    pt = sha = None
    if save:
        os.makedirs(CKPT_DIR, exist_ok=True)
        pt = os.path.join(CKPT_DIR, "%s%s_s%d.pt" % (tag, arm, seed))
        torch.save(best["state"], pt)                      # doc 45 SS5
        sha = hashlib.sha256(open(pt, "rb").read()).hexdigest()
    return {"arm": arm, "base_arm": base_arm(arm), "is_floor": arm.startswith("S-shuffle"),
            "seed": seed, "train_acc": best["train"], "val_acc": best["val"],
            "selected_epoch": best["epoch"], "heldout_test_acc": te,
            "val_minus_heldout": best["val"] - te,
            "gate_met": bool(best["train"] >= TRAIN_GATE and best["val"] >= VAL_GATE),
            "n_params": sum(p.numel() for p in m.parameters()),
            "wires": 2 * (2 if base_arm(arm) == "F2" else 1),
            "n_train": len(data["train"]), "n_val": len(data["val"]),
            "n_test": len(data["test"]),
            "max_epochs": max_epochs, "lr": LR, "batch": BATCH,
            "ckpt_path": pt, "ckpt_sha256": sha,
            "wall_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--tag", default="", help="prefix for checkpoint names; use "
                                              "for the doc 54 SS6.1.1 confirm pass")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    blob = json.load(open(a.data, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("exp68 dataset provenance FAILED")
    verbs = sorted(meta["verb_inventory"].keys())
    fm = feature_maps()

    if a.smoke:
        smoke(items, meta, verbs, fm)
        return 0

    arms = [x for x in a.arms.split(",") if x]
    jobs = [(arm, s) for arm in arms for s in range(a.seeds)]
    mine = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    print("[exp68 w%d/%d] arms=%s -> %d of %d runs"
          % (a.worker, a.nworkers, arms, len(mine), len(jobs)), flush=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    for arm, s in mine:
        ckj = os.path.join(CKPT_DIR, "%s%s_s%d.json" % (a.tag, arm, s))
        if os.path.exists(ckj):
            print("[ckpt] %s%s_s%d already done" % (a.tag, arm, s), flush=True)
            continue
        r = train(arm, s, items, verbs, MAX_EPOCHS, fm, save=True, tag=a.tag)
        r["dataset"] = a.data
        r["dataset_sha256"] = meta["dataset_sha256"]
        r["tag"] = a.tag
        json.dump(r, open(ckj, "w", encoding="utf-8"))
        print("[exp68] %-14s s%d  train=%.3f val=%.3f HELD-OUT=%.4f  (%.0fs)"
              % (arm, s, r["train_acc"], r["val_acc"], r["heldout_test_acc"],
                 r["wall_sec"]), flush=True)
    print("[exp68 w%d] done" % a.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
