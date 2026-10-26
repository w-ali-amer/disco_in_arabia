# -*- coding: utf-8 -*-
"""exp42r_common.py -- shared spine for the exp42 REVIVAL (doc 22 + doc 65).

Owns the four things `exp42r_run.py` and `exp42r_aggregate.py` must agree on
EXACTLY, so they cannot drift apart:

  1. the provenance gate (dataset sha + embedding sha + C5),
  2. the registered arm set, including doc 65 SS2.1's A2-TI,
  3. doc 65 SS2.4's init policy and its per-(arm, seed) dedicated streams,
  4. the on-disk unit-checkpoint layout.

TOUCHES NO COMMITTED ARTEFACT. `exp42_models.py`, `exp42_baselines.py`,
`exp42_controls.py`, `exp42_compiler.py`, `exp42_run.py`,
`results_exp42.json` and `results_exp40b.json` are IMPORTED OR READ, never
written and never modified (doc 65's Rule-3 clause). Every file this
programme writes carries the `exp42r` / `results_exp42r` marker.

------------------------------------------------------------------------
THE DATASET IS NOT THE AUGUST ONE. `exp42_compiler.load_dataset` pins
`EXPECTED_DATASET_SHA256 = 2cb3cc...`, the August dataset that doc 65 SS6.2
ruled VOID. The revival's dataset of record is `stories_exp41r.json`, sha
`2326942f...` (doc 65 SS6.3), so this module supplies its own loader and
its own pin. `exp42_compiler`'s pure functions -- `canonical_sha256`,
`compile_story`, `compile_all`, `split_items`, `load_embeddings`,
`rerender_sentences` -- are reused unmodified.

------------------------------------------------------------------------
doc 65 SS2.1 -- A2-TI, THE ONE NEW REGISTERED ARM

"A2's circuit with the noun-intro rotations TRAINABLE, everything else
identical." `TrainableIntroModel` below subclasses the committed
`QuantumStoryModel` in the "A2" configuration and adds ONE trained angle
per noun (44 nouns: 24 humans + 20 objects), replacing the frozen
Ry(embedding angle) introduction. Everything downstream -- the dressed
symmetric core, the shared two-effect readout, the C1 verb_map protocol --
is inherited verbatim.

FLAGGED DEVIATION, doc 65 SS2.4 vs SS2.1. SS2.4 registers "all trained
parameters init ~ N(0, 0.1^2)". Read literally that would initialise
A2-TI's intro angles from noise, which DELETES THE EMBEDDING SEED -- and
SS2.1's entire defence of the arm is that "the loss is QA, embeddings only
initialise". Under the literal reading A2-TI would differ from A2 in TWO
ways at once (intro trainable AND intro information destroyed), and
`trained_intro_helps` would be confounded beyond interpretation. This
build therefore initialises A2-TI's intro angles AT THE FROZEN EMBEDDING
ANGLE and applies N(0, 0.1^2) to every other trained parameter. The
deviation is recorded in results_exp42r.json, not buried here.

------------------------------------------------------------------------
doc 65 SS2.4 -- INIT POLICY, PER ARM, RECORDED

"All trained parameters init ~ N(0, 0.1^2) from per-(arm, seed) dedicated
streams; recorded per run." Applied as:

  A1 / A2 / A2TI / A3   every angle ~ N(0, 0.1^2)   (blocks near identity)
  A2TI intro angles     = the frozen embedding angle (see above)
  B1                    core angles ~ N(0, 0.1^2); the dressing tensors and
                        the two effect matrices keep their committed
                        near-identity FORM, I + N(0, 0.1^2) -- which is
                        already an N(0, 0.1^2) DRAW, exactly SS2.4's
                        distribution, around the structural identity
  B2 / B2_generous / B3 every trained parameter ~ N(0, 0.1^2)

WHY B1 IS NOT ZEROED, stated before the numbers exist. B1's dressings and
effects are MATRICES, not angles; drawing them from N(0, 0.1^2) about ZERO
makes every verb map a near-null contraction and the classical composition
tree numerically degenerate. B1 is the DECISIVE baseline of doc 22 SS1's
H-Q, so an init that cripples it would manufacture a tier_q pass. SS2.4's
subject matter is the ledger's identity-vs-uniform ANGLE question, and the
committed B1 already draws N(0, 0.1^2); this build changes only its
uniform[0, 2pi) core angles. Recorded as an interpretation, and flagged.

Streams: `init_stream_seed(arm, seed) = 42_000_000 + 1000 * ARM_INDEX[arm]
+ seed` -- dedicated per (arm, seed), disjoint across arms by construction,
and disjoint from the generator's 41000 / 42000 seeds. Each unit records
its stream seed AND the sha256 of its own flattened init vector, so a
re-run is checkable rather than merely repeatable.

No admissibility gate is applied to any drawn init: doc 65 SS2.3 registers
NO bounded-redraw machinery and none is added.
"""

import hashlib
import json
import math
import os

import torch
from torch import nn

import exp42_compiler as comp
from exp42_baselines import BagBaseline, ClassicalDisCoCirc, SeqBaseline
from exp42_models import QuantumStoryModel

# ---------------------------------------------------------------- pins
DATA_PATH = "stories_exp41r.json"
DATASET_SHA = ("2326942ffb6ccf6bca95a7ec0502c82563957fabb159b143608a9fb31f"
               "8f3113")
EMB_SHA = ("00c1ecffcaf05b38df082f8a3cbed1d013bddb3e39bc00adf1c2513b4ea674"
           "ac")

# doc 65 SS7.1: C5 of record for the REVIVAL is results_exp40r.json (the
# Mac re-run that reproduced August bit-exactly). results_exp40b.json is
# the August artefact and is READ-ONLY here -- it is never opened for write.
GATE_FILE = "results_exp40r.json"
AUGUST_GATE_FILE = "results_exp40b.json"

# Directory overrides exist ONLY so the run->aggregate chain can be
# integration-tested end to end into a scratch tree before the real
# launch. Unset in every real run; the defaults are the run of record.
CKPT_DIR = os.environ.get("EXP42R_CKPT_DIR", "exp42r_ckpt")
CTL_DIR = os.environ.get("EXP42R_CTL_DIR", "exp42r_ctl")
LOCK_DIR = os.environ.get("EXP42R_LOCK_DIR", "exp42r_locks")

# doc 22 SS3 A1-A3 + doc 65 SS2.1's A2TI + doc 22 SS4 B1-B3.
ARMS = ("A1", "A2", "A2TI", "A3", "B1", "B2", "B2_generous", "B3")
ARM_INDEX = {a: i for i, a in enumerate(ARMS)}
QUANTUM_ARMS = ("A1", "A2", "A2TI", "A3")
STRUCTURED_ARMS = ("A1", "A2", "A2TI", "A3", "B1")
UNSTRUCTURED_ARMS = ("B2", "B2_generous", "B3")

INIT_STD = 0.1
INIT_STREAM_BASE = 42_000_000

READOUT_AMBIGUITY_NOTE = (
    "doc 22 SS3.4 does not specify how the question's verb identity enters "
    "the two-effect readout (Duneau's task had one fixed question). The "
    "committed August resolution is INHERITED UNCHANGED, not re-decided: "
    "the question verb's OWN trained block is applied as inverse "
    "preparation on the two queried wires, then two SHARED trained "
    "Circuit-4 effects (q_pos/q_neg, 15 params each). Zero per-verb readout "
    "parameters, doc 22 SS3.3's budget preserved, C1 permutes story and "
    "question coherently, and A1's exact-chance twin guarantee preserved. "
    "Q1 bridge questions apply both verbs' blocks in reverse order. See "
    "exp42_models.py's docstring.")

A2TI_INIT_DEVIATION = (
    "doc 65 SS2.4 registers N(0, 0.1^2) for ALL trained parameters; A2-TI's "
    "intro angles are instead initialised AT THE FROZEN EMBEDDING ANGLE. "
    "Reason, recorded before any run: SS2.1's own defence of the arm is "
    "'the loss is QA, embeddings only initialise'. A noise init would "
    "delete the embedding seed, so A2-TI would differ from A2 in two ways "
    "at once (intro trainable AND intro information destroyed) and "
    "trained_intro_helps would be confounded. FLAGGED DEVIATION.")

B1_INIT_INTERPRETATION = (
    "B1's dressing tensors and effect matrices keep the committed "
    "near-identity form I + N(0, 0.1^2) -- already an N(0, 0.1^2) draw, "
    "doc 65 SS2.4's distribution, taken around the structural identity "
    "rather than around zero. Only B1's uniform[0, 2pi) core angles are "
    "changed to N(0, 0.1^2). Reason, recorded before any run: B1's "
    "dressings are MATRICES, not angles; drawing them about zero makes "
    "every classical verb map a near-null contraction, and B1 is doc 22 "
    "SS1's DECISIVE baseline -- an init that cripples it would manufacture "
    "a tier_q pass. FLAGGED INTERPRETATION.")


# ==================================================================
# provenance
# ==================================================================
def load_dataset(path=DATA_PATH):
    """doc 22 SS6 provenance sanity gate, on the REVIVAL dataset of record."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items, meta = data["items"], data["meta"]
    sha = comp.canonical_sha256(items)
    if sha != meta["dataset_sha256"] or sha != DATASET_SHA:
        raise RuntimeError(
            "PROVENANCE GATE FAILED (doc 22 SS6): %s canonical sha256=%s, "
            "manifest=%s, doc 65 SS6.3 pin=%s -- the run is invalidated and "
            "THAT is the report."
            % (path, sha, meta["dataset_sha256"], DATASET_SHA))
    return items, meta, sha


def load_embeddings():
    """Frozen noun embeddings, pinned. doc 22 SS3.1: initialization only."""
    angles, emb_meta, emb_sha = comp.load_embeddings()
    if emb_sha != EMB_SHA:
        raise RuntimeError(
            "PROVENANCE GATE FAILED (doc 22 SS6): %s sha256=%s, pin=%s -- "
            "the frozen embeddings drifted; the run is invalidated and THAT "
            "is the report." % (comp.EMB_PATH, emb_sha, EMB_SHA))
    return angles, emb_meta, emb_sha


def check_c5_gate():
    """doc 22 SS6 C5 / doc 65 SS7.1. Reads results_exp40r.json; never writes."""
    if not os.path.exists(GATE_FILE):
        raise RuntimeError(
            "C5 GATE: %s does not exist. exp42r is blocked until the exp40r "
            "harness-calibration verdict (doc 22 SS6 C5)." % GATE_FILE)
    with open(GATE_FILE, encoding="utf-8") as f:
        v = json.load(f)
    cal = v.get("c5_calibration", {})
    if cal.get("harness_calibrated") is not True:
        raise RuntimeError("C5 GATE: %s has harness_calibrated=%r (must be "
                           "true)." % (GATE_FILE, cal.get("harness_calibrated")))
    return {"file": GATE_FILE, "harness_calibrated": True,
            "qualifying_seeds": cal.get("qualifying_seeds"),
            "verdict_rule": cal.get("verdict_rule"),
            "reproduces_august_exactly":
                v.get("august_comparison", {})
                 .get("max_abs_accuracy_difference") == 0.0,
            "august_reference_read_only": AUGUST_GATE_FILE}


def provenance_gate(path=DATA_PATH, require_c5=True):
    """The single call every entry point makes before doing anything."""
    items, meta, sha = load_dataset(path)
    angles, emb_meta, emb_sha = load_embeddings()
    gate = check_c5_gate() if require_c5 else {"skipped": "smoke"}
    rec = {"dataset": {"path": path, "sha256": sha, "n_items": len(items),
                       "generator_seed": meta.get("seed"),
                       "pin_doc65_SS6_3": DATASET_SHA, "pin_matches": True},
           "embeddings": {"path": comp.EMB_PATH, "sha256": emb_sha,
                          "n": len(angles), "source": emb_meta.get("source")},
           "c5_gate": gate,
           "untouched_committed_artifacts": [
               "exp42_run.py", "results_exp42.json", "results_exp40b.json",
               "exp42_models.py", "exp42_baselines.py", "exp42_controls.py",
               "exp42_compiler.py", "stories_exp41.json"]}
    return items, meta, angles, emb_sha, rec


# ==================================================================
# doc 65 SS2.1 -- A2-TI
# ==================================================================
class TrainableIntroModel(QuantumStoryModel):
    """A2's circuit with the noun-intro rotations TRAINABLE (doc 65 SS2.1).

    One trained angle per noun, seeded from the frozen 2D embedding angle.
    Everything else -- dressed symmetric core, shared two-effect readout,
    verb_map protocol -- is the committed A2 model, inherited."""

    def __init__(self, verbs, seed, angles):
        super().__init__("A2", verbs, seed)
        # `self.arm` STAYS "A2": the committed `verb_matrix` dispatches on
        # it, and A2-TI's block family IS A2's (doc 65 SS2.1: "A2's circuit
        # with the noun-intro rotations TRAINABLE, everything else
        # identical"). The ARM IDENTITY is carried separately -- an earlier
        # build overwrote self.arm and silently routed every verb through
        # A3's `blocks`, which does not exist here.
        self.arm_label = "A2TI"
        self.nouns = sorted(angles)
        self.intro_angle = nn.ParameterDict(
            {n: nn.Parameter(torch.tensor(float(angles[n])))
             for n in self.nouns})

    def story_values(self, cs, verb_map=None):
        """Identical to the parent except the introduction: Ry of a TRAINED
        angle per noun instead of Ry of the frozen embedding angle."""
        import exp42_sim_adapter as sim
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)
            return cache[v]

        psi = sim.init_state(cs.K)
        for w, noun, angle in cs.intro:
            th = self.intro_angle[noun] if noun in self.intro_angle \
                else torch.tensor(float(angle))
            psi = sim.apply_gate(psi, sim.ry(th), (w,))
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

    def param_counts(self):
        """The parent asserts an exact 14/verb + readout accounting, which
        A2-TI's +1/noun intro parameters break by design."""
        total = sum(p.numel() for p in self.parameters())
        readout = sum(p.numel() for p in self.q_pos.parameters()) + \
            sum(p.numel() for p in self.q_neg.parameters())
        intro = len(self.nouns)
        assert total == 14 * len(self.verbs) + readout + intro, \
            "A2TI param accounting drifted: %d" % total
        return {"arm": "A2TI", "per_verb": 14, "n_verbs": len(self.verbs),
                "readout_shared": readout, "trained_intro_angles": intro,
                "total_trained": total,
                "note": "doc 65 SS2.1: A2 + one trained intro angle per noun"}


# ==================================================================
# factory + doc 65 SS2.4's init
# ==================================================================
def make_model(arm, verbs, vocab, seed, angles):
    if arm in ("A1", "A2", "A3"):
        return QuantumStoryModel(arm, verbs, seed)
    if arm == "A2TI":
        return TrainableIntroModel(verbs, seed, angles)
    if arm == "B1":
        return ClassicalDisCoCirc(verbs, seed, d_ref=2)
    if arm == "B2":
        return SeqBaseline(vocab, seed, variant="matched")
    if arm == "B2_generous":
        return SeqBaseline(vocab, seed, variant="generous")
    if arm == "B3":
        return BagBaseline(vocab, seed)
    raise ValueError("unregistered arm %r (registered set: %s)"
                     % (arm, ARMS))


def init_stream_seed(arm, seed):
    return INIT_STREAM_BASE + 1000 * ARM_INDEX[arm] + seed


def apply_init(model, arm, seed, angles):
    """doc 65 SS2.4, applied per the per-arm policy documented above.
    Returns the recordable init spec, including a sha of the drawn vector."""
    stream = init_stream_seed(arm, seed)
    g = torch.Generator().manual_seed(stream)
    notes = []
    with torch.no_grad():
        if arm in QUANTUM_ARMS:
            for name, p in model.named_parameters():
                if arm == "A2TI" and name.startswith("intro_angle."):
                    noun = name.split(".", 1)[1]
                    p.copy_(torch.tensor(float(angles[noun])))
                else:
                    p.copy_(torch.randn(p.shape, generator=g) * INIT_STD)
            policy = "every trained angle ~ N(0, %.2f^2)" % INIT_STD
            if arm == "A2TI":
                policy += ("; intro angles = the frozen embedding angle "
                           "(FLAGGED DEVIATION, see A2TI_INIT_DEVIATION)")
                notes.append(A2TI_INIT_DEVIATION)
        elif arm == "B1":
            d = model.d
            D = d * d
            for name, p in model.named_parameters():
                if name.startswith("core."):
                    p.copy_(torch.randn(p.shape, generator=g) * INIT_STD)
                elif name.startswith("dress."):
                    p.copy_(torch.eye(d).repeat(4, 1, 1)
                            + INIT_STD * torch.randn(p.shape, generator=g))
                else:                                     # q_pos / q_neg
                    p.copy_(torch.eye(D)
                            + INIT_STD * torch.randn(p.shape, generator=g))
            policy = ("core angles ~ N(0, %.2f^2); dressings and effects "
                      "= I + N(0, %.2f^2)" % (INIT_STD, INIT_STD))
            notes.append(B1_INIT_INTERPRETATION)
        else:                                             # B2 / B3
            for _name, p in model.named_parameters():
                p.copy_(torch.randn(p.shape, generator=g) * INIT_STD)
            policy = ("every trained parameter ~ N(0, %.2f^2) (literal "
                      "SS2.4)" % INIT_STD)

        flat = torch.cat([p.detach().reshape(-1).to(torch.float64)
                          for _n, p in sorted(model.named_parameters())])
        h = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
        assert torch.isfinite(flat).all(), "non-finite init for %s s%d" % (arm,
                                                                           seed)
    return {"spec": "doc 65 SS2.4", "arm": arm, "seed": seed,
            "policy": policy, "std": INIT_STD, "stream_seed": stream,
            "stream_rule": "42_000_000 + 1000*ARM_INDEX[arm] + seed "
                           "(dedicated per (arm, seed))",
            "n_params": int(flat.numel()), "init_vector_sha256": h,
            "no_admissibility_gate": "doc 65 SS2.3 registers NO "
                                     "bounded-redraw machinery and none is "
                                     "added",
            "notes": notes}


def build_model(arm, verbs, vocab, seed, angles):
    """Construct + apply the registered init. The ONE way a model is made."""
    torch.manual_seed(seed)
    m = make_model(arm, verbs, vocab, seed, angles)
    spec = apply_init(m, arm, seed, angles)
    return m, spec


def is_seq(arm):
    return arm in UNSTRUCTURED_ARMS


# ==================================================================
# checkpoint layout
# ==================================================================
def unit_paths(arm, seed, ckpt_dir=CKPT_DIR):
    return (os.path.join(ckpt_dir, "%s_s%d.json" % (arm, seed)),
            os.path.join(ckpt_dir, "%s_s%d.pt" % (arm, seed)))


def failed_path(job_id, ctl_dir=CTL_DIR):
    return os.path.join(ctl_dir, "FAILED_%s.json" % job_id)


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def vocab_sha(vocab):
    return comp.canonical_sha256(vocab)


# ==================================================================
# fast permutation evaluation (control C1)
# ==================================================================
def precompute_verb_mats(model, verbs, verb_map=None):
    """Verb matrices for ONE verb_map, built once instead of once per story.

    `exp42_models.story_values` caches verb matrices per STORY; C1 evaluates
    1,000 permutations x 5 seeds x the union set, so hoisting the cache to
    the (model, permutation) level is the difference between hours and
    minutes. It changes no arithmetic -- `assert_fast_path_matches` below
    checks the fast path against the committed forward on real items."""
    vm = verb_map or {}
    with torch.no_grad():
        return {v: model.verb_matrix(vm.get(v, v)) for v in verbs}


def fast_story_values(model, cs, mats):
    """The committed gate order with a pre-built verb-matrix table.
    Quantum arms and A2TI only (B1 has its own real-tensor semantics)."""
    import exp42_sim_adapter as sim
    psi = sim.init_state(cs.K)
    trained_intro = getattr(model, "intro_angle", None)
    for w, noun, angle in cs.intro:
        th = trained_intro[noun] if (trained_intro is not None
                                     and noun in trained_intro) \
            else torch.tensor(float(angle))
        psi = sim.apply_gate(psi, sim.ry(th), (w,))
    for v, wa, wp in cs.events:
        psi = sim.apply_gate(psi, mats[v], (wa, wp))
    vseq, qa, qp = cs.question
    qw = (qa, qp)
    for qv in reversed(vseq):
        psi = sim.apply_gate(psi, mats[qv].conj().T, qw)
    return torch.stack([sim.prob_all_zero_on(
        sim.apply_gate(psi, eff.matrix().conj().T, qw), qw)
        for eff in (model.q_pos, model.q_neg)])


def fast_accuracy(model, compiled, verbs, verb_map=None):
    with torch.no_grad():
        mats = precompute_verb_mats(model, verbs, verb_map)
        return sum(1 for cs in compiled
                   if int(torch.argmax(fast_story_values(model, cs, mats))
                          .item()) == cs.answer) / len(compiled)


def assert_fast_path_matches(model, compiled, verbs, n=12, tol=1e-6):
    """The fast path must reproduce the committed forward bit-for-bit."""
    worst = 0.0
    with torch.no_grad():
        mats = precompute_verb_mats(model, verbs)
        for cs in compiled[:n]:
            a = fast_story_values(model, cs, mats)
            b = model.story_values(cs)
            worst = max(worst, float((a - b).abs().max().item()))
    if not math.isfinite(worst) or worst > tol:
        raise RuntimeError("C1 fast path diverges from the committed "
                           "forward: max|delta| = %r" % worst)
    return worst
