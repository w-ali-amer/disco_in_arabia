# -*- coding: utf-8 -*-
"""exp42_models.py -- the three quantum arms (doc 22 SS3) + shared readout.

Arms (per-verb two-qubit blocks; the pre-registered ablation is the WHOLE
search space -- doc 22 SS7):
- A1 exchange-symmetric core only: U_v = CP(theta_v) * XX(phi_v),
  2 params/verb. NEGATIVE CONTROL: [U_v, SWAP] = 0, so a swap-twin story
  compiles to the *identical* statevector and A1 sits at exactly 50% on
  twin pairs. If A1 scores above chance, generator or readout leaks and
  the run is invalidated (doc 22 SS3.2).
- A2 dressed symmetric core: U_v = (A_v (x) P_v) . E_v . (A'_v (x) P'_v),
  E_v the same symmetric core, four single-qubit Euler role dressings.
  12 + 2 = 14 params/verb. The bet.
- A3 unconstrained reference: Sim-et-al Circuit-4, 3 layers on 2 wires,
  15 params/verb (exchange-asymmetric family, analog-incompatible).

Readout (doc 22 SS3.4, mirroring exp40_mini_duneau + its exp40b fix):
two-effect softmax. For question (v, agent_wire, patient_wire):
  1. apply U_v^dagger on the queried wires  (inverse preparation -- the
     verb part of the question effect; see AMBIGUITY note below);
  2. for each trained effect Q_pos, Q_neg (shared Circuit-4, 3 layers,
     2 wires): apply Q^dagger, take P(queried wires == |00>) with every
     other wire traced out;
  3. loss logits = log(p + 1e-12), cross-entropy -- softmax(log p) =
     p/(p_pos+p_neg), the exp40b gradient-scaling fix (raw-p logits bound
     the logit gap to <= 1 and starve gradients ~6x). Prediction = argmax.

AMBIGUITY RESOLVED (FLAGGED, doc 22 SS3.4): the doc specifies the
two-effect readout but not how the QUESTION'S VERB identity enters the
effect (Duneau's task had a single fixed question). Options considered:
(a) per-verb trained effect pairs (+30 params/verb -- busts the SS3.3
budget and lets readout, not blocks, carry verb content); (b) verb one-hot
-> effect-parameter map (same objection); (c) the verb's OWN trained block
applied as inverse preparation on the queried wires, followed by shared
trained effects. (c) is chosen: it is literally "the question applied as
an effect (inverse preparation)", adds zero per-verb readout parameters
(SS3.3 budget preserved: 2/14/15 per verb + 30 shared), keeps C1's
verb->block permutation coherent (question and story permute together),
and preserves A1's exact-chance guarantee. For the two-verb Q1 bridge
questions ("did the one who v1-ed X also v2 Y", doc 22 SS2.2) both verbs'
blocks are applied as inverse preparations on the queried (x, y) wires in
reverse order -- same rule, sequence length 2. This resolution is
recorded in every results JSON.
"""

import math

import torch
from torch import nn

import exp42_sim_adapter as sim


class QuantumStoryModel(nn.Module):
    ARMS = ("A1", "A2", "A3")

    def __init__(self, arm, verbs, seed):
        super().__init__()
        assert arm in self.ARMS, arm
        self.arm = arm
        self.verbs = list(verbs)
        g = torch.Generator().manual_seed(seed)

        def u(shape):
            return nn.Parameter(torch.rand(shape, generator=g) * 2 * math.pi)

        if arm == "A1":
            self.core = nn.ParameterDict({v: u(2) for v in self.verbs})
        elif arm == "A2":
            self.core = nn.ParameterDict({v: u(2) for v in self.verbs})
            # dress[v][slot] = Euler triple; slots 0..3 = A, P, A', P'
            self.dress = nn.ParameterDict({v: u((4, 3)) for v in self.verbs})
        else:                                             # A3
            self.blocks = nn.ModuleDict(
                {v: sim.Circuit4(2, 3, generator=g) for v in self.verbs})

        # Shared question effects (exp40 convention): Circuit-4, 3 layers,
        # 2 wires, 15 params each. Shared across arms for comparability;
        # A1's chance guarantee is independent of readout asymmetry (its
        # twin stories produce the identical statevector).
        self.q_pos = sim.Circuit4(2, 3, generator=g)
        self.q_neg = sim.Circuit4(2, 3, generator=g)

    # ----------------------------------------------------------- blocks
    def _core_matrix(self, v):
        th = self.core[v]
        return sim.cphase(th[0]) @ sim.xx(th[1])

    def verb_matrix(self, v):
        if self.arm == "A1":
            return self._core_matrix(v)
        if self.arm == "A2":
            d = self.dress[v]
            A = sim.euler_1q(d[0, 0], d[0, 1], d[0, 2])
            P = sim.euler_1q(d[1, 0], d[1, 1], d[1, 2])
            A2 = sim.euler_1q(d[2, 0], d[2, 1], d[2, 2])
            P2 = sim.euler_1q(d[3, 0], d[3, 1], d[3, 2])
            return sim.kron2(A, P) @ self._core_matrix(v) @ sim.kron2(A2, P2)
        return self.blocks[v].matrix()                    # A3

    # ---------------------------------------------------------- forward
    def story_values(self, cs, verb_map=None):
        """[v_yes, v_no] for one CompiledStory. `verb_map` (controls C1/C2)
        remaps verb -> block-owner verb, applied consistently to story
        events AND the question (the pre-registered permutation is of the
        verb->block ASSIGNMENT)."""
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)
            return cache[v]

        psi = sim.init_state(cs.K)
        for w, _noun, angle in cs.intro:
            psi = sim.apply_gate(psi, sim.ry(angle), (w,))
        for v, wa, wp in cs.events:
            psi = sim.apply_gate(psi, block(v), (wa, wp))

        vseq, qa, qp = cs.question
        qw = (qa, qp)
        for qv in reversed(vseq):                         # inverse prep
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)
        vals = []
        for eff in (self.q_pos, self.q_neg):
            phi = sim.apply_gate(psi, eff.matrix().conj().T, qw)
            vals.append(sim.prob_all_zero_on(phi, qw))
        return torch.stack(vals)

    def story_loss(self, cs):
        vals = self.story_values(cs)
        target = torch.tensor([cs.answer])
        logits = torch.log(vals + 1e-12)                  # exp40b fix
        loss = nn.functional.cross_entropy(logits.unsqueeze(0), target)
        pred = int(torch.argmax(vals).item())
        return loss, pred

    def predict(self, cs, verb_map=None):
        with torch.no_grad():
            return int(torch.argmax(self.story_values(cs, verb_map)).item())

    # ------------------------------------------------------------ counts
    def param_counts(self):
        per_verb = {"A1": 2, "A2": 14, "A3": 15}[self.arm]
        total = sum(p.numel() for p in self.parameters())
        readout = sum(p.numel() for p in self.q_pos.parameters()) + \
            sum(p.numel() for p in self.q_neg.parameters())
        assert total == per_verb * len(self.verbs) + readout, \
            "param accounting drifted: %d != %d*%d+%d" \
            % (total, per_verb, len(self.verbs), readout)
        return {"arm": self.arm, "per_verb": per_verb,
                "n_verbs": len(self.verbs), "readout_shared": readout,
                "total_trained": total}
