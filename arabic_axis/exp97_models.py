# -*- coding: utf-8 -*-
"""exp97_models.py -- doc 92 (REGISTERED 24c4f39 + §4 amendments 72727a4). GOLD routing, exp95 chain machinery.

Every gate: U = U0 @ expm(G), U0 fixed Haar base point, G trainable generator initialised at ZERO (doc 92 §4.1).
  Q-INV      : U(4) on (r, o).  U0 Haar complex from Generator(97700+seed).
  R-INV      : SO(4) on (r, o). U0 Haar SO(4) from Generator(97700+seed).
  R-8        : SO(8) on (r, o, RB) with ONE shared rebit RB; U0 = REALIFICATION of Q-INV's U0 (same seed) ->
               starts functionally identical to Q-INV (§4.1); strict superset of realified U(4) (§1).
  R-8-LOCAL  : as R-8 but one fresh rebit per event (no cross-event memory); descriptive (§4.2).
Readout INV: V_q^dag (V^T for real) on the event's wires, then shared effects Qp/Qn, P(x, O_q = 00) with rebits traced.
"""
import torch
import torch.nn as nn
from torch_sv_sim import CDTYPE
import exp94_common as C
import exp95_models as M95
from exp94_models import apply_n


def haar_u(g, n=4):
    Z = torch.complex(torch.randn(n, n, generator=g), torch.randn(n, n, generator=g)) / 2 ** 0.5
    Q, R = torch.linalg.qr(Z)
    d = torch.diagonal(R)
    return Q * (d / d.abs())[None, :]


def haar_so(g, n=4):
    Z = torch.randn(n, n, generator=g)
    Q, R = torch.linalg.qr(Z)
    Q = Q * torch.sign(torch.diagonal(R))[None, :]
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def realify(U):
    """complex 4x4 on (r,o) -> real 8x8 on wire order (r, o, rebit): [[A,-B],[B,A]] (rebit-major) then permuted."""
    A, B = U.real, U.imag
    M = torch.cat([torch.cat([A, -B], 1), torch.cat([B, A], 1)], 0)            # index rb*4 + k
    perm = torch.tensor([rb * 4 + k for k in range(4) for rb in range(2)])     # new index k*2 + rb
    return M[perm][:, perm]


class Gate(nn.Module):
    def __init__(self, U0, field):
        super().__init__()
        self.field = field
        n = U0.shape[0]
        self.register_buffer("U0", U0.to(CDTYPE) if field == "C" else U0.to(torch.float32))
        self.re = nn.Parameter(torch.zeros(n, n))
        if field == "C":
            self.im = nn.Parameter(torch.zeros(n, n))

    def matrix(self):
        if self.field == "C":
            A = torch.complex(self.re, self.im)
            return self.U0 @ torch.linalg.matrix_exp(A - A.conj().T).to(CDTYPE)
        return (self.U0 @ torch.linalg.matrix_exp(self.re - self.re.T)).to(CDTYPE)


class QRoom(M95.QChain):
    def __init__(self, seed, kind):
        nn.Module.__init__(self)
        self.mode, self.kind = "GOLD", kind
        g = torch.Generator().manual_seed(97700 + seed)
        n_gates = len(C.VERBS) + 2
        if kind == "Q-INV":
            gates = [Gate(haar_u(g), "C") for _ in range(n_gates)]
        elif kind == "R-INV":
            gates = [Gate(haar_so(g), "R") for _ in range(n_gates)]
        else:                                                   # R-8 / R-8-LOCAL: realified Q-INV base points
            gates = [Gate(realify(haar_u(g)), "R") for _ in range(n_gates)]
        self.verbs = nn.ModuleList(gates[:len(C.VERBS)])
        self.qp, self.qn = gates[len(C.VERBS)], gates[len(C.VERBS) + 1]
        self.morph = None

    def weights(self, b):
        return M95.gold_weights(b["gold"], b["T"])

    def _nreb(self, T):
        return {"Q-INV": 0, "R-INV": 0, "R-8": 1, "R-8-LOCAL": T}[self.kind]

    def _wires(self, r, t, T):
        base = (r, 2 + t)
        if self.kind == "R-8":
            return base + (2 + T,)
        if self.kind == "R-8-LOCAL":
            return base + (2 + T + t,)
        return base

    def scores(self, b):
        B, T = b["cg"].shape[0], b["T"]
        nq = 2 + T + self._nreb(T)
        V = torch.stack([m.matrix() for m in self.verbs])
        qs = [M95.ry_state(b["theta"][:, 0]), M95.ry_state(b["theta"][:, 1])] + \
             [M95.ry_state(b["theta_o"][:, t]) for t in range(T)] + \
             [M95.ry_state(torch.zeros(B)) for _ in range(self._nreb(T))]          # rebits |0>
        psi = qs[0]
        for q in qs[1:]:
            psi = psi[..., None] * q.reshape([B] + [1] * (psi.dim() - 1) + [2])
        states = psi[:, None]
        for t in range(T):
            nb = states.shape[1]
            flat = states.reshape([B * nb] + [2] * nq)
            Vr = V[b["v"][:, t]].repeat_interleave(nb, 0)
            a0 = apply_n(flat, Vr, self._wires(0, t, T), nq)
            s1 = apply_n(flat, Vr, self._wires(1, t, T), nq)
            s01 = apply_n(a0, Vr, self._wires(1, t, T), nq)
            states = torch.stack([x.reshape(states.shape) for x in (flat, a0, s1, s01)], 1).reshape([B, 4 * nb] + [2] * nq)
        W = self.weights(b)
        NS = states.shape[1]
        flat = states.reshape([B * NS] + [2] * nq)
        qverb = b["v"][torch.arange(B), b["q"]]
        pre = V[qverb].conj().transpose(1, 2).repeat_interleave(NS, 0)
        d = pre.shape[-1]
        Ep = self.qp.matrix().conj().T.expand(B * NS, d, d); En = self.qn.matrix().conj().T.expand(B * NS, d, d)
        res = []
        for xw in (0, 1):
            per_q = []
            for qt in range(T):
                w = self._wires(xw, qt, T)
                s = apply_n(flat, pre, w, nq)
                r = []
                for E in (Ep, En):
                    tt = apply_n(s, E, w, nq)
                    idx = [slice(None)] * (nq + 1); idx[1 + xw] = 0; idx[1 + 2 + qt] = 0
                    a = tt[tuple(idx)]
                    r.append(((a.real ** 2 + a.imag ** 2).reshape(B, NS, -1).sum(-1) * W).sum(1))
                per_q.append(torch.stack(r, -1))
            res.append(torch.stack(per_q, 1))
        R = torch.stack(res, 1)
        return R[torch.arange(B), b["x"], b["q"]]


class ClGoldRef(M95.ClChain):
    """R-UNC-GRAM reference (exp96 architecture, gold routing)."""
    def logit(self, b):
        saved = M95.seq_weights
        M95.seq_weights = lambda pc, ps: M95.gold_weights(b["gold"], b["T"])
        try:
            return super().logit(b)
        finally:
            M95.seq_weights = saved


ARMS = {"Q-INV": lambda s: QRoom(s, "Q-INV"), "R-INV": lambda s: QRoom(s, "R-INV"), "R-8": lambda s: QRoom(s, "R-8"),
        "R-8-LOCAL": lambda s: QRoom(s, "R-8-LOCAL"), "R-UNC-GRAM": lambda s: ClGoldRef(s)}
