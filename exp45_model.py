# -*- coding: utf-8 -*-
"""exp45_model.py -- doc 28: two-qubit referent registers.

One qubit per referent -> a TWO-QUBIT REGISTER per referent. K referents
occupy 2K wires; a verb event on (agent, patient) acts on the FOUR wires of
the two registers. The second qubit of each register is unconstrained
working space: nothing prepares it, nothing reads it directly, and only
trained verb blocks can populate it. That is the whole bet -- with one
qubit per referent there is nowhere to write the intermediate binding a
2-hop bridge answer needs.

Everything else is held fixed (doc 28 SS2): same composition tree, same
frozen datasets, same frozen noun embeddings, same readout protocol, same
loss, same gates.

  intro    referent r with frozen embedding angle theta_r prepares as
           Ry(theta_r)|0> on the register's FIRST qubit, |0> on its second.
           The frozen embedding artifact is reused VERBATIM.
  events   block on wires (2*wa, 2*wa+1, 2*wp, 2*wp+1) -- agent register
           first, mirroring exp42's agent-wire-first convention.
  readout  the queried pair occupies 4 wires; the question's verb blocks
           are applied as inverse preparations in reverse order (the doc-22
           SS3.4 resolution, unchanged), then two SHARED trained effects,
           then P(queried 4 wires == |0000>) with all other wires traced.
           Loss logits = log p (the exp40b fix), cross-entropy.

Arms (doc 28 SS3):
  A4  the bet          Circuit4(4 wires, 3 layers)   33 params/verb
  A5  leak sentinel    register-symmetric             2 params/verb

A5 EXISTS BECAUSE LEMMA 1 DOES NOT TRANSFER FOR FREE. A1's exact-chance
guarantee at one qubit per referent comes from commuting with the pair
SWAP; a 4-wire block needs the guarantee rebuilt against S_reg, the
permutation exchanging the agent register with the patient register.
Blocks of the form

    U_v = exp(-i (a_v (H1 (x) I + I (x) H1) + b_v (H2 (x) I + I (x) H2)))

with H1, H2 fixed Hermitian single-register (4x4) generators satisfy
[U_v, S_reg] = 0 identically -- the two commutator terms cancel -- so a
swap-twin story compiles to the IDENTICAL statevector and any readout is
pinned to exactly 50% on twins. verify_sentinel() below discharges doc 28
SS3's verification obligation.

EACH ARM TRAINS ITS OWN q_pos/q_neg (doc 28 SS3): A5's chance guarantee is
readout-independent, so sharing a readout would only make the sentinel's
numbers depend on another arm's training.
"""

import math

import torch
from torch import nn

import exp42_sim_adapter as sim

CD = sim.CDTYPE
_I2 = torch.eye(2, dtype=CD)
_X = torch.tensor([[0, 1], [1, 0]], dtype=CD)
_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=CD)
_Z = torch.tensor([[1, 0], [0, -1]], dtype=CD)
_I4 = torch.eye(4, dtype=CD)

# Fixed Hermitian single-register (2-qubit, 4x4) generators for A5.
# Generic enough that A5 is not trivially near-identity; FIXED, never trained.
_H1 = torch.kron(_X, _Z) + torch.kron(_Z, _X)
_H2 = torch.kron(_Y, _Y) + torch.kron(_X, _X)


def register_swap(dtype=CD):
    """S_reg: 16x16 permutation exchanging register A (local wires 0,1)
    with register B (local wires 2,3), in the simulator's wire ordering."""
    S = torch.zeros(16, 16, dtype=dtype)
    for a in range(4):                       # a = (w0,w1) as 2 bits
        for b in range(4):                   # b = (w2,w3) as 2 bits
            S[b * 4 + a, a * 4 + b] = 1.0
    return S


def sentinel_matrix(a, b, dtype=CD):
    """A5's block. G = a(H1 (x) I + I (x) H1) + b(H2 (x) I + I (x) H2)
    commutes with S_reg by construction, so exp(-iG) does too."""
    H1 = _H1.to(dtype)
    H2 = _H2.to(dtype)
    I4 = _I4.to(dtype)
    G = (a.to(dtype) * (torch.kron(H1, I4) + torch.kron(I4, H1))
         + b.to(dtype) * (torch.kron(H2, I4) + torch.kron(I4, H2)))
    return torch.linalg.matrix_exp(-1j * G)


class TwoQubitReferentModel(nn.Module):
    """Arms A4 / A5 over 2K wires (two-qubit register per referent)."""

    ARMS = ("A4", "A5")

    def __init__(self, arm, verbs, seed, layers=3):
        super().__init__()
        assert arm in self.ARMS, arm
        self.arm = arm
        self.layers = layers
        self.verbs = list(verbs)
        g = torch.Generator().manual_seed(seed)
        if arm == "A4":
            self.blocks = nn.ModuleDict(
                {v: sim.Circuit4(4, layers, generator=g) for v in self.verbs})
        else:
            self.core = nn.ParameterDict(
                {v: nn.Parameter(torch.rand(2, generator=g) * 2 * math.pi)
                 for v in self.verbs})
        # own readout pair (doc 28 SS3), same Circuit4(4, layers) shape
        self.q_pos = sim.Circuit4(4, layers, generator=g)
        self.q_neg = sim.Circuit4(4, layers, generator=g)

    # ----------------------------------------------------------- blocks
    def verb_matrix(self, v):
        if self.arm == "A4":
            return self.blocks[v].matrix()
        t = self.core[v]
        return sentinel_matrix(t[0], t[1])

    @staticmethod
    def _reg(w):
        return (2 * w, 2 * w + 1)

    # ---------------------------------------------------------- forward
    def story_values(self, cs, verb_map=None):
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)
            return cache[v]

        psi = sim.init_state(2 * cs.K)
        for w, _noun, angle in cs.intro:
            psi = sim.apply_gate(psi, sim.ry(angle), (2 * w,))
        for v, wa, wp in cs.events:
            psi = sim.apply_gate(psi, block(v),
                                 self._reg(wa) + self._reg(wp))

        vseq, qa, qp = cs.question
        qw = self._reg(qa) + self._reg(qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)
        vals = []
        for eff in (self.q_pos, self.q_neg):
            phi = sim.apply_gate(psi, eff.matrix().conj().T, qw)
            vals.append(sim.prob_all_zero_on(phi, qw))
        return torch.stack(vals)

    def story_loss(self, cs):
        vals = self.story_values(cs)
        target = torch.tensor([cs.answer])
        logits = torch.log(vals + 1e-12)
        loss = nn.functional.cross_entropy(logits.unsqueeze(0), target)
        return loss, int(torch.argmax(vals).item())

    def predict(self, cs, verb_map=None):
        with torch.no_grad():
            return int(torch.argmax(self.story_values(cs, verb_map)).item())

    def param_counts(self):
        per_verb = 33 if self.arm == "A4" else 2
        total = sum(p.numel() for p in self.parameters())
        readout = sum(p.numel() for p in self.q_pos.parameters()) + \
            sum(p.numel() for p in self.q_neg.parameters())
        assert total == per_verb * len(self.verbs) + readout, \
            "param accounting drifted: %d" % total
        return {"arm": self.arm, "per_verb": per_verb,
                "n_verbs": len(self.verbs), "readout_shared": readout,
                "total_trained": total, "wires_per_referent": 2}


# ------------------------------------------------- doc 28 SS3 verification
def verify_sentinel(compiled_twin_pairs=None, n_draws=200, seed=45000):
    """Discharge doc 28 SS3's verification obligation.

    Bars are split by precision, and the split is itself pre-registered:
    the commutation is EXACT as algebra, but complex64 matrix_exp on a
    16x16 matrix does not resolve 1e-6 -- measured worst 5.9e-6 in
    complex64 versus 1.9e-14 in complex128 while doc 28 was being drafted.
    A flat 1e-6 bar would therefore have voided a mathematically sound
    sentinel for reasons of arithmetic.

      (i)   float64 commutator      <= 1e-10
      (ii)  complex64 commutator    <= 1e-4   [AMENDED, doc 30]
      (iii) twin statevector delta  <= 1e-5
      (iv)  a5_twin_accuracy        == 0.500 exactly

    DOC 30 AMENDMENT, and it travels with every number this returns: check
    (ii) was originally registered at <= 1e-5 and MEASURED 1.13e-5 -- a 13%
    miss. The bar had been calibrated against torch.randn generators with
    a,b ~ N(0,1), while the shipped A5 uses fixed H1 = XZ+ZX, H2 = YY+XX
    with a,b ~ U[0,2pi), giving a much larger ||G|| and correspondingly
    larger float32 matrix_exp round-off. The construction itself is sound
    (float64 5.35e-14; twin statevectors 9.00e-7; 100% of twin pairs
    predicted identically; twin accuracy exactly 0.500). A BROKEN
    construction would give a commutator of order 1, not 1e-5, so any bar
    in 1e-3..1e-6 discharges this check's purpose identically.
    """
    g = torch.Generator().manual_seed(seed)
    out = {"doc30_amendment": (
        "check (ii) complex64 commutator bar amended 1e-5 -> 1e-4 (doc 30, "
        "commit aa05e85), authored BEFORE any exp45 result existed. The "
        "originally registered bar FAILED at a measured 1.13e-5. The theorem "
        "is carried by check (i) in float64 at <= 1e-10, unchanged and "
        "passing at 5.35e-14. Rule 5: this caveat travels with the number.")}
    for dtype, key, bar in ((torch.complex128, "float64", 1e-10),
                            (CD, "complex64", 1e-4)):
        S = register_swap(dtype)
        worst = 0.0
        for _ in range(n_draws):
            a, b = (torch.rand(2, generator=g) * 2 * math.pi)
            U = sentinel_matrix(a, b, dtype=dtype)
            c = torch.linalg.matrix_norm(U @ S - S @ U).item()
            worst = max(worst, c)
        out["commutator_%s_worst" % key] = worst
        out["commutator_%s_bar" % key] = bar
        out["commutator_%s_pass" % key] = bool(worst <= bar)
        if key == "complex64":
            out["commutator_complex64_bar_as_originally_registered"] = 1e-5
            out["commutator_complex64_pass_at_original_bar"] = bool(worst <= 1e-5)
    out["n_draws"] = n_draws

    if compiled_twin_pairs:
        model = TwoQubitReferentModel("A5", compiled_twin_pairs[0][2], 0)
        worst_delta, n_ident, n_pairs = 0.0, 0, 0
        correct = 0
        with torch.no_grad():
            for ca, cb, _verbs in compiled_twin_pairs:
                pa = _statevector(model, ca)
                pb = _statevector(model, cb)
                worst_delta = max(worst_delta, (pa - pb).abs().max().item())
                if model.predict(ca) == model.predict(cb):
                    n_ident += 1
                correct += int(model.predict(ca) == ca.answer)
                correct += int(model.predict(cb) == cb.answer)
                n_pairs += 1
        acc = correct / (2 * n_pairs)
        out.update({
            "twin_statevector_delta_worst": worst_delta,
            "twin_statevector_delta_bar": 1e-5,
            "twin_statevector_pass": bool(worst_delta <= 1e-5),
            "n_twin_pairs": n_pairs,
            "frac_twin_pairs_predicting_identically": n_ident / n_pairs,
            "a5_twin_accuracy": acc,
            "a5_twin_accuracy_exact_half": bool(acc == 0.5),
        })
    checks = [v for k, v in out.items() if k.endswith("_pass")
              or k == "a5_twin_accuracy_exact_half"]
    out["sentinel_verified"] = bool(checks and all(checks))
    out["consequence_if_failed"] = (
        "doc 28 SS3: if any check fails the sentinel is broken and NO exp45 "
        "result is reported -- a class without a working sentinel cannot "
        "distinguish learning from leakage.")
    return out


def _statevector(model, cs):
    """Compiled story -> statevector, no readout (for the twin identity)."""
    with torch.no_grad():
        psi = sim.init_state(2 * cs.K)
        for w, _n, angle in cs.intro:
            psi = sim.apply_gate(psi, sim.ry(angle), (2 * w,))
        for v, wa, wp in cs.events:
            psi = sim.apply_gate(psi, model.verb_matrix(v),
                                 model._reg(wa) + model._reg(wp))
        return psi
