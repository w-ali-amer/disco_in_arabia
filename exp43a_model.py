# -*- coding: utf-8 -*-
"""exp43a_model.py -- provably-universal two-qubit blocks for exp43a.

Doc 23 exp43a numerical certificate: "A3-class (full SU(4)) shared blocks".
exp42's A3 used the Sim-et-al Circuit-4 3-layer ansatz (15 params), whose
coverage of SU(4) is NOT proven. exp43a therefore uses the Cartan/KAK form

    U(theta) = (A1 (x) A2) . N(a,b,c) . (B1 (x) B2),
    N(a,b,c) = exp(i (a XX + b YY + c ZZ)),

with A_i, B_i single-qubit XZX-Euler unitaries (3 params each) and (a,b,c)
the Cartan coordinates: 4*3 + 3 = 15 parameters. By the KAK decomposition
of SU(4) w.r.t. SU(2)(x)SU(2) (Khaneja & Glaser 2001; Kraus & Cirac 2001)
EVERY two-qubit unitary is of this form up to global phase, and a global
phase on any block is irrelevant to every probability this model computes
(each probability is the squared modulus of amplitudes linear in each
block). So this parameterization provably covers the full model class.

N(a,b,c) is built in closed form via its Bell-basis eigendecomposition
(XX, YY, ZZ commute; the magic basis diagonalizes all three):
    |Phi+> : a - b + c      |Psi+> : a + b - c
    |Phi-> : -a + b + c     |Psi-> : -a - b - c

Effects: the exp42 readout uses effect vector Q|00>. For KAK Q this ranges
over ALL two-qubit pure states: any |phi> has a Schmidt form
(A (x) B)(cos t |00> + e^{i s} sin t |11>), and N(t,0,0)|00> =
cos t |00> + i sin t |11> supplies the Schmidt weights (phases absorbed
into A, B). So the trained effect pair is unrestricted, as the doc's
"two shared question effects trainable" requires.

Readout semantics are INHERITED unchanged from
exp42_models.QuantumStoryModel: story_values / story_loss / predict are
the committed exp42 code paths (two-effect log-p softmax, inverse-
preparation question encoding, agent-wire-first convention). Only
__init__ / verb_matrix / param_counts differ. exp42_models is imported,
never modified.
"""

import math

import torch
from torch import nn

import exp42_models
import exp42_sim_adapter as sim

# Bell/magic basis, columns = (|Phi+>, |Psi+>, |Phi->, |Psi->),
# rows = (|00>, |01>, |10>, |11>).
_BELL = (1.0 / math.sqrt(2.0)) * torch.tensor(
    [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [0, 1, 0, -1],
     [1, 0, -1, 0]], dtype=sim.CDTYPE)

# eigenvalue coefficient rows for (a, b, c) per Bell column above
_EIG_SIGNS = torch.tensor([[1.0, -1.0, 1.0],
                           [1.0, 1.0, -1.0],
                           [-1.0, 1.0, 1.0],
                           [-1.0, -1.0, -1.0]])


def cartan_core(a, b, c):
    """N(a,b,c) = exp(i (a XX + b YY + c ZZ)), closed form, differentiable."""
    abc = torch.stack([a, b, c])
    lam = _EIG_SIGNS @ abc                                # (4,)
    ph = torch.exp(1j * lam.to(sim.CDTYPE))
    return _BELL @ torch.diag(ph) @ _BELL.conj().T


class KAK2Q(nn.Module):
    """General two-qubit unitary in Cartan/KAK form, 15 parameters."""

    def __init__(self, generator=None):
        super().__init__()
        init = torch.rand(15, generator=generator) * (2 * math.pi)
        self.theta = nn.Parameter(init)

    def matrix(self):
        t = self.theta
        A = sim.kron2(sim.euler_1q(t[0], t[1], t[2]),
                      sim.euler_1q(t[3], t[4], t[5]))
        B = sim.kron2(sim.euler_1q(t[6], t[7], t[8]),
                      sim.euler_1q(t[9], t[10], t[11]))
        return A @ cartan_core(t[12], t[13], t[14]) @ B


class KakStoryModel(exp42_models.QuantumStoryModel):
    """exp42 story model with provably-universal SU(4) blocks + effects.

    Inherits the exp42 forward/readout verbatim; replaces only the block
    family and the effect parameterization.
    """

    def __init__(self, verbs, seed):
        nn.Module.__init__(self)                          # bypass arm setup
        self.arm = "KAK"
        self.verbs = list(verbs)
        g = torch.Generator().manual_seed(seed)
        self.blocks = nn.ModuleDict({v: KAK2Q(g) for v in self.verbs})
        self.q_pos = KAK2Q(g)
        self.q_neg = KAK2Q(g)

    def verb_matrix(self, v):
        return self.blocks[v].matrix()

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        readout = self.q_pos.theta.numel() + self.q_neg.theta.numel()
        assert total == 15 * len(self.verbs) + readout
        return {"arm": "KAK", "per_verb": 15, "n_verbs": len(self.verbs),
                "readout_shared": readout, "total_trained": total}


def _selftest():
    torch.manual_seed(0)
    # 1) cartan_core matches matrix_exp for random Cartan coordinates
    for _ in range(5):
        a, b, c = torch.rand(3) * 2 * math.pi
        X = torch.tensor([[0, 1], [1, 0]], dtype=sim.CDTYPE)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=sim.CDTYPE)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=sim.CDTYPE)
        H = (a.to(sim.CDTYPE) * torch.kron(X, X)
             + b.to(sim.CDTYPE) * torch.kron(Y, Y)
             + c.to(sim.CDTYPE) * torch.kron(Z, Z))
        ref = torch.linalg.matrix_exp(1j * H)
        got = cartan_core(a, b, c)
        err = (ref - got).abs().max().item()
        assert err < 5e-6, "cartan_core mismatch: %g" % err

    # 2) KAK2Q matrix unitary + gradients flow
    g = torch.Generator().manual_seed(1)
    blk = KAK2Q(generator=g)
    M = blk.matrix()
    err = (M.conj().T @ M - torch.eye(4, dtype=sim.CDTYPE)).abs().max().item()
    assert err < 5e-6, "KAK2Q not unitary: %g" % err
    loss = M.real.sum()
    loss.backward()
    assert blk.theta.grad is not None and blk.theta.grad.norm().item() > 0

    # 3) KAK can express a maximally-asymmetric gate class member (CNOT)
    #    up to phase: check the Cartan point (pi/4,0,0) + local frames can
    #    reach an operator with the CNOT's nonlocal invariants -- here we
    #    just verify N(pi/4,0,0) maximally entangles |00> (sanity, not a
    #    full invariant calculation).
    N = cartan_core(torch.tensor(math.pi / 4), torch.tensor(0.0),
                    torch.tensor(0.0))
    psi = torch.tensor([1, 0, 0, 0], dtype=sim.CDTYPE)
    out = (N @ psi).reshape(2, 2)
    # Schmidt coefficients of the output
    s = torch.linalg.svdvals(out)
    assert abs(s[0].item() - s[1].item()) < 1e-5, \
        "N(pi/4,0,0) should be perfectly entangling on this input"

    # 4) inherited exp42 readout runs end-to-end on a toy compiled story
    from exp42_compiler import CompiledStory
    cs = CompiledStory(
        item_id="toy", pair_id="toy", split="train", qtype="Q1",
        hop_count=2, pro_drop_item=False, order_dependent=False, K=4,
        wires={"A": 0, "X": 1, "Y": 2, "W": 3},
        intro=[(0, "A", 0.3), (1, "X", 0.7), (2, "Y", 1.1), (3, "W", 0.5)],
        events=[("v1", 0, 1), ("v2", 0, 2), ("v1", 3, 2)],
        question=(("v1", "v2"), 1, 2), answer=0)
    m = KakStoryModel(["v1", "v2"], seed=7)
    loss, pred = m.story_loss(cs)
    loss.backward()
    gn = sum(p.grad.norm().item() for p in m.parameters()
             if p.grad is not None)
    assert gn > 0, "no gradient through inherited exp42 readout"
    assert pred in (0, 1)
    pc = m.param_counts()
    assert pc["total_trained"] == 15 * 2 + 30
    print("[exp43a_model] selftest OK (cartan closed form, unitarity, "
          "grads, inherited exp42 readout)")


if __name__ == "__main__":
    _selftest()
