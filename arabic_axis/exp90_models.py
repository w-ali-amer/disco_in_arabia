# -*- coding: utf-8 -*-
"""exp90_models.py -- doc 83 SS4 architectures.

Quantum (all inherit exp42_models.QuantumStoryModel("A3", verbs, seed), so
verb blocks (Circuit-4, 3 layers) and the two shared effects are initialised
exactly as exp60's SBModel at the same seed; readout = exp60's: the question
verb's block inverse on the FIXED cast pair (0,1), then the effects,
P(all-zero on (0,1))):

  ArchM  (SS4.1) 1 id qubit / referent, RY(theta). Case boxes NOM, ACC =
         Euler-3 (exp42 euler_1q: RX-RZ-RX) single-qubit unitaries shared
         across nouns, applied on the noun's id wire after RY per its vowel
         (NONE -> no box). Event V_verb on (slot-1 wire, slot-2 wire).
         inits: 'id' (boxes = I), 'rand' (Haar boxes). boxes=False -> P
         (vowel-blind, no box parameters).
  ArchR  (SS4.2) id wires 0..K-1, morpheme wires K..2K-1. Case box = RY(a)
         on the noun's morpheme wire (NOM: a_nom, ACC: a_acc). Router
         R = controlled-U, control = slot-1 morpheme wire, U in KAK SU(4)
         (exp43a_model.KAK2Q) on (slot-1, slot-2) id wires. Event R^dag V R.
         inits: 'und' (U=SWAP, a_nom=a_acc=pi/2), 'rand', 'pure' (U=SWAP,
         a_nom=a_acc=0), 'construct' (U=SWAP, a_nom=0, a_acc=pi; A0 only).

Classical (SS4.3-4.4, pinned in SS11.1): CLDisco, CLMLP, CLLSTM.
Label convention: exp42 yes = class 0. Quantum predict = argmax(values);
classical predict = 0 (yes) iff logit > 0; BCE target = 1 for yes.
"""
import math

import torch
from torch import nn

import exp42_sim_adapter as sim
from exp42_models import QuantumStoryModel
from exp43a_model import KAK2Q
import exp90_encoder as enc

BOX_SEED_BASE = 90_700          # Haar/rand box + router generator (flagged)
PI = math.pi


def euler_params_haar(g):
    """Haar-distributed SU(2) in exp42's RX(c) RZ(b) RX(a) Euler form:
    outer angles uniform, middle angle density ~ sin(b) on [0, pi]."""
    a = torch.rand((), generator=g) * 2 * PI
    c = torch.rand((), generator=g) * 2 * PI
    b = torch.acos(1 - 2 * torch.rand((), generator=g))
    return torch.stack([a, b, c])


def kak_swap_theta():
    """KAK2Q parameters with identity locals and Cartan (pi/4,pi/4,pi/4):
    N = exp(i pi/4 (XX+YY+ZZ)) = e^{i pi/4} SWAP (global phase irrelevant
    inside R^dag V R; A0 checks this numerically rather than assuming it)."""
    t = torch.zeros(15)
    t[12:15] = PI / 4
    return t


def _p0(state, wires):
    """P(wires == |0..0>), identical maths to torch_sv_sim.prob_all_zero_on but
    |z|^2 computed as re^2 + im^2. Reason (doc 83 §11.3): complex64 abs()
    backward returns NaN at SUBNORMAL amplitudes (verified: z ~ 1.1e-39 ->
    grad nan), which Arch-R reaches when its case boxes converge to exact basis
    states. re^2+im^2 is smooth everywhere; forward agrees with abs()**2 to
    <= 4.8e-7 on normal-range amplitudes. Shared module left untouched."""
    s = state
    for w in sorted(wires, reverse=True):
        s = s.select(w, 0)
    return (s.real ** 2 + s.imag ** 2).sum()


class _QBase(QuantumStoryModel):
    def mats(self):
        """Per-batch matrix cache (identical maths to per-item rebuild)."""
        return {"V": {v: self.verb_matrix(v) for v in self.verbs},
                "Qp": self.q_pos.matrix(), "Qn": self.q_neg.matrix()}

    def _readout(self, psi, c, M):
        vseq, qa, qp = c.question
        qw = (qa, qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, M["V"][qv].conj().T, qw)
        vp = _p0(sim.apply_gate(psi, M["Qp"].conj().T, qw), qw)
        vn = _p0(sim.apply_gate(psi, M["Qn"].conj().T, qw), qw)
        return torch.stack([vp, vn])

    def values(self, c, M=None):
        return self.forward_one(c, M if M is not None else self.mats())

    def predict_many(self, cs):
        with torch.no_grad():
            M = self.mats()
            return [int(torch.argmax(self.forward_one(c, M)).item()) for c in cs]

    def loss_batch(self, cs):
        M = self.mats()
        tot = 0.0
        for c in cs:
            tot = tot + nn.functional.cross_entropy(
                torch.log(self.forward_one(c, M) + 1e-12).unsqueeze(0),
                torch.tensor([c.answer]))
        return tot / len(cs)


class ArchM(_QBase):
    def __init__(self, verbs, seed, init="id", boxes=True):
        super().__init__("A3", verbs, seed)
        self.use_boxes = boxes
        self.init_kind = init if boxes else "none(P)"
        if boxes:
            if init == "id":
                self.nom = nn.Parameter(torch.zeros(3))
                self.acc = nn.Parameter(torch.zeros(3))
            elif init == "rand":
                g = torch.Generator().manual_seed(BOX_SEED_BASE + seed)
                self.nom = nn.Parameter(euler_params_haar(g))
                self.acc = nn.Parameter(euler_params_haar(g))
            else:
                raise ValueError(init)

    def mats(self):
        M = super().mats()
        if self.use_boxes:
            M["NOM"] = sim.euler_1q(self.nom[0], self.nom[1], self.nom[2])
            M["ACC"] = sim.euler_1q(self.acc[0], self.acc[1], self.acc[2])
        return M

    def forward_one(self, c, M):
        psi = sim.init_state(c.K)
        case_of = {c.slots[0]: c.cases[0], c.slots[1]: c.cases[1]}
        for w, _n, ang in c.intro:
            psi = sim.apply_gate(psi, sim.ry(ang), (w,))
            cs = case_of.get(w, enc.NONE)
            if self.use_boxes and cs != enc.NONE:
                psi = sim.apply_gate(psi, M[cs], (w,))
        psi = sim.apply_gate(psi, M["V"][c.verb], c.slots)
        return self._readout(psi, c, M)

    def box_state(self):
        if not self.use_boxes:
            return {}
        return {"nom": self.nom.detach().tolist(),
                "acc": self.acc.detach().tolist()}


class ArchR(_QBase):
    def __init__(self, verbs, seed, init="und"):
        super().__init__("A3", verbs, seed)
        self.init_kind = init
        g = torch.Generator().manual_seed(BOX_SEED_BASE + 50_000 + seed)
        self.router = KAK2Q(generator=g)            # rand default (uniform)
        if init == "rand":
            # control population sin^2(a/2) ~ U(0,1)  (Haar marginal)
            a1 = 2 * torch.asin(torch.sqrt(torch.rand((), generator=g)))
            a2 = 2 * torch.asin(torch.sqrt(torch.rand((), generator=g)))
            self.a_nom = nn.Parameter(a1.reshape(1))
            self.a_acc = nn.Parameter(a2.reshape(1))
        else:
            with torch.no_grad():
                self.router.theta.copy_(kak_swap_theta())
            an, aa = {"und": (PI / 2, PI / 2), "pure": (0.0, 0.0),
                      "construct": (0.0, PI)}[init]
            self.a_nom = nn.Parameter(torch.tensor([an]))
            self.a_acc = nn.Parameter(torch.tensor([aa]))

    def mats(self):
        M = super().mats()
        U = self.router.matrix()
        eye = torch.eye(4, dtype=sim.CDTYPE)
        CU = torch.block_diag(eye, U)                # control = first wire
        M["CU"], M["CUd"] = CU, CU.conj().T
        M["NOM"], M["ACC"] = sim.ry(self.a_nom[0]), sim.ry(self.a_acc[0])
        return M

    def forward_one(self, c, M):
        K = c.K
        psi = sim.init_state(2 * K)
        for w, _n, ang in c.intro:
            psi = sim.apply_gate(psi, sim.ry(ang), (w,))
        for w, cs in zip(c.slots, c.cases):
            if cs != enc.NONE:
                psi = sim.apply_gate(psi, M[cs], (K + w,))
        s1, s2 = c.slots
        psi = sim.apply_gate(psi, M["CU"], (K + s1, s1, s2))
        psi = sim.apply_gate(psi, M["V"][c.verb], (s1, s2))
        psi = sim.apply_gate(psi, M["CUd"], (K + s1, s1, s2))
        return self._readout(psi, c, M)

    def box_state(self):
        return {"a_nom": float(self.a_nom.detach()[0]),
                "a_acc": float(self.a_acc.detach()[0]),
                "router_theta": self.router.theta.detach().tolist()}


# ================================================================ classical
def _case_onehot(cs):
    return {enc.NOM: [1.0, 0.0], enc.ACC: [0.0, 1.0], enc.NONE: [0.0, 0.0]}[cs]


class _CBase(nn.Module):
    def predict_many(self, cs):
        with torch.no_grad():
            z = self.logits(cs)
        return [0 if float(v) > 0 else 1 for v in z]

    def loss_batch(self, cs):
        y = torch.tensor([1.0 if c.answer == 0 else 0.0 for c in cs])
        return nn.functional.binary_cross_entropy_with_logits(self.logits(cs), y)

    def box_state(self):
        return {}


def _theta(c, slot):
    ang = {w: a for w, _n, a in c.intro}
    return ang[c.slots[slot]]


class CLDisco(_CBase):
    """SS11.1: u = (1, cos t, sin t); logit = (Case1 u1)^T C_v (Case2 u2) + b.
    NONE (unvocalised) -> identity case map (the 'no case box' analogue)."""

    def __init__(self, verbs, seed, init="id"):
        super().__init__()
        self.verbs = list(verbs)
        self.init_kind = init
        g = torch.Generator().manual_seed(seed)
        if init == "id":
            self.N = nn.Parameter(torch.eye(3))
            self.A = nn.Parameter(torch.eye(3))
            self.C = nn.Parameter(torch.randn(len(verbs), 3, 3, generator=g) * 0.1)
            self.b = nn.Parameter(torch.zeros(()))
        elif init == "rand":
            self.N = nn.Parameter(torch.randn(3, 3, generator=g) * 0.5)
            self.A = nn.Parameter(torch.randn(3, 3, generator=g) * 0.5)
            self.C = nn.Parameter(torch.randn(len(verbs), 3, 3, generator=g) * 0.5)
            self.b = nn.Parameter(torch.randn((), generator=g) * 0.5)
        else:
            raise ValueError(init)

    def logits(self, cs):
        eye = torch.eye(3)
        out = []
        for c in cs:
            vi = self.verbs.index(c.verb)
            vecs = []
            for s in (0, 1):
                t = _theta(c, s)
                u = torch.tensor([1.0, math.cos(t), math.sin(t)])
                Mx = {enc.NOM: self.N, enc.ACC: self.A, enc.NONE: eye}[c.cases[s]]
                vecs.append(Mx @ u)
            out.append(vecs[0] @ self.C[vi] @ vecs[1] + self.b)
        return torch.stack(out)

    def box_state(self):
        return {"N": self.N.detach().tolist(), "A": self.A.detach().tolist()}


class CLMLP(_CBase):
    """SS11.1: [cos t1, sin t1, case1(2), cos t2, sin t2, case2(2), verb(6)]
    -> 32 ReLU -> 1."""

    def __init__(self, verbs, seed):
        super().__init__()
        self.verbs = list(verbs)
        self.init_kind = "default"
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(8 + len(verbs), 32), nn.ReLU(),
                                 nn.Linear(32, 1))

    def feats(self, c):
        f = []
        for s in (0, 1):
            t = _theta(c, s)
            f += [math.cos(t), math.sin(t)] + _case_onehot(c.cases[s])
        f += [1.0 if v == c.verb else 0.0 for v in self.verbs]
        return f

    def logits(self, cs):
        return self.net(torch.tensor([self.feats(c) for c in cs])).squeeze(-1)


class CLLSTM(_CBase):
    """SS11.1: surface-order token sequence (V, N1, N2). Noun token =
    [cos t, sin t, case(2)] + 6-d zero; verb token = 4-d zero + verb one-hot.
    dim 10, 1 layer, hidden 32, last state -> 1."""

    def __init__(self, verbs, seed):
        super().__init__()
        self.verbs = list(verbs)
        self.init_kind = "default"
        torch.manual_seed(seed)
        self.lstm = nn.LSTM(4 + len(verbs), 32, num_layers=1, batch_first=True)
        self.head = nn.Linear(32, 1)

    def seq(self, c):
        z6 = [0.0] * len(self.verbs)
        vt = [0.0] * 4 + [1.0 if v == c.verb else 0.0 for v in self.verbs]
        toks = [vt]
        for s in (0, 1):
            t = _theta(c, s)
            toks.append([math.cos(t), math.sin(t)] + _case_onehot(c.cases[s]) + z6)
        return toks

    def logits(self, cs):
        x = torch.tensor([self.seq(c) for c in cs])
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


# ================================================================ registry
ARMS = {
    "M-id":      ("ArchM", {"init": "id"}),
    "M-rand":    ("ArchM", {"init": "rand"}),
    "P":         ("ArchM", {"init": "id", "boxes": False}),
    "M-shuf":    ("ArchM", {"init": "id"}),
    "R-und":     ("ArchR", {"init": "und"}),
    "R-rand":    ("ArchR", {"init": "rand"}),
    "R-pure":    ("ArchR", {"init": "pure"}),
    "CLD-id":    ("CLDisco", {"init": "id"}),
    "CLD-rand":  ("CLDisco", {"init": "rand"}),
    "CL-MLP":    ("CLMLP", {}),
    "CL-LSTM":   ("CLLSTM", {}),
}
QUANTUM_ARMS = {"M-id", "M-rand", "P", "M-shuf", "R-und", "R-rand", "R-pure"}


def make_model(arm, verbs, seed):
    cls, kw = ARMS[arm]
    return {"ArchM": ArchM, "ArchR": ArchR, "CLDisco": CLDisco,
            "CLMLP": CLMLP, "CLLSTM": CLLSTM}[cls](verbs, seed, **kw)
