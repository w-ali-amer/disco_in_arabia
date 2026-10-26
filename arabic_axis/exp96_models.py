# -*- coding: utf-8 -*-
"""exp96_models.py -- doc 91 (REGISTERED 8929d26): field x readout decomposition, GOLD routing, exp95 chain machinery.

Verbs/effects are matrix exponentials: complex U = expm(A - A^dag) (A complex N(0,1)), real O = expm(B - B^T)
(B real N(0,1)) -- the real restriction of the complex generator family. Real arms stay exactly real (G-REAL).
Readout INV: V_q^dag on (x, O_q) then shared effects Qp/Qn; FREE: per-question-verb effects E_p[v], E_n[v] directly.
"""
import torch
import torch.nn as nn
from torch_sv_sim import CDTYPE
import exp94_common as C
import exp95_models as M95
from exp94_models import apply_n


class Expm(nn.Module):
    def __init__(self, g, field):
        super().__init__()
        self.field = field
        self.re = nn.Parameter(torch.randn(4, 4, generator=g))
        if field == "C":
            self.im = nn.Parameter(torch.randn(4, 4, generator=g))

    def matrix(self):
        if self.field == "C":
            A = torch.complex(self.re, self.im)
            return torch.linalg.matrix_exp(A - A.conj().T).to(CDTYPE)
        B = self.re
        return torch.linalg.matrix_exp(B - B.T).to(CDTYPE)          # real orthogonal; imag == 0 exactly


class QField(M95.QChain):
    def __init__(self, seed, field, readout):
        nn.Module.__init__(self)
        self.mode, self.field, self.readout = "GOLD", field, readout
        g = torch.Generator().manual_seed(96700 + seed)
        self.verbs = nn.ModuleList([Expm(g, field) for _ in C.VERBS])
        if readout == "INV":
            self.qp, self.qn = Expm(g, field), Expm(g, field)
        else:
            self.ep = nn.ModuleList([Expm(g, field) for _ in C.VERBS])
            self.en = nn.ModuleList([Expm(g, field) for _ in C.VERBS])
        self.morph = None

    def weights(self, b):
        return M95.gold_weights(b["gold"], b["T"])

    def scores(self, b):
        B, T = b["cg"].shape[0], b["T"]
        nq = 2 + T
        V = torch.stack([m.matrix() for m in self.verbs])
        qs = [M95.ry_state(b["theta"][:, 0]), M95.ry_state(b["theta"][:, 1])] + \
             [M95.ry_state(b["theta_o"][:, t]) for t in range(T)]
        psi = qs[0]
        for q in qs[1:]:
            psi = psi[..., None] * q.reshape([B] + [1] * (psi.dim() - 1) + [2])
        states = psi[:, None]
        for t in range(T):
            nb = states.shape[1]
            flat = states.reshape([B * nb] + [2] * nq)
            Vr = V[b["v"][:, t]].repeat_interleave(nb, 0)
            a0 = apply_n(flat, Vr, (0, 2 + t), nq); s1 = apply_n(flat, Vr, (1, 2 + t), nq); s01 = apply_n(a0, Vr, (1, 2 + t), nq)
            states = torch.stack([x.reshape(states.shape) for x in (flat, a0, s1, s01)], 1).reshape([B, 4 * nb] + [2] * nq)
        W = self.weights(b)
        NS = states.shape[1]
        flat = states.reshape([B * NS] + [2] * nq)
        qverb = b["v"][torch.arange(B), b["q"]]
        if self.readout == "INV":
            pre = V[qverb].conj().transpose(1, 2).repeat_interleave(NS, 0)
            Ep = self.qp.matrix().conj().T; En = self.qn.matrix().conj().T
            Ep = Ep.expand(B * NS, 4, 4); En = En.expand(B * NS, 4, 4)
        else:
            pre = None
            EP = torch.stack([m.matrix() for m in self.ep]).conj().transpose(1, 2)
            EN = torch.stack([m.matrix() for m in self.en]).conj().transpose(1, 2)
            Ep = EP[qverb].repeat_interleave(NS, 0); En = EN[qverb].repeat_interleave(NS, 0)
        res = []
        for xw in (0, 1):
            per_q = []
            for qt in range(T):
                s = apply_n(flat, pre, (xw, 2 + qt), nq) if pre is not None else flat
                r = []
                for E in (Ep, En):
                    tt = apply_n(s, E, (xw, 2 + qt), nq)
                    idx = [slice(None)] * (nq + 1); idx[1 + xw] = 0; idx[1 + 2 + qt] = 0
                    a = tt[tuple(idx)]
                    r.append(((a.real ** 2 + a.imag ** 2).reshape(B, NS, -1).sum(-1) * W).sum(1))
                per_q.append(torch.stack(r, -1))
            res.append(torch.stack(per_q, 1))
        R = torch.stack(res, 1)
        self._last_imag = float(flat.imag.detach().abs().max()) if self.field == "R" else None
        return R[torch.arange(B), b["x"], b["q"]]


class ClGold(M95.ClChain):
    """R-UNC-GRAM reference: exp95's classical twin with GOLD routing."""
    def logit(self, b):
        saved = M95.seq_weights
        M95.seq_weights = lambda pc, ps: M95.gold_weights(b["gold"], b["T"])
        try:
            return super().logit(b)
        finally:
            M95.seq_weights = saved


ARMS = {"Q-INV": lambda s: QField(s, "C", "INV"), "R-INV": lambda s: QField(s, "R", "INV"),
        "Q-FREE": lambda s: QField(s, "C", "FREE"), "R-FREE": lambda s: QField(s, "R", "FREE"),
        "R-UNC-GRAM": lambda s: ClGold(s)}
