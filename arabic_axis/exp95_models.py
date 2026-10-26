# -*- coding: utf-8 -*-
"""exp95_models.py -- doc 90 §3/§4: chain circuits (exact mixture with SHARED candidate features, uncomputed parity),
classical twin, TF/LSTM baselines, full-statevector G-SIM reference (K=2).

W(S_1..S_T) = sum_c P(c) prod_t P(S_t | c); c = 64 candidate-feature configs (2 cands x 3 bits); P(S_t|c) sums over
event t's 8 suffix configs; S_t in {none, {0}, {1}, {0,1}} indexed by bitmask.
"""
import itertools
import torch
import torch.nn as nn
from torch_sv_sim import Circuit4, CDTYPE
import exp94_common as C
from exp94_models import Morph, ry_state, apply_n, NG, NN, VG, VN

CFG8 = torch.tensor([[(c >> j) & 1 for j in range(3)] for c in range(8)], dtype=torch.float32)          # suffix configs
CFG64 = torch.tensor([[(c >> j) & 1 for j in range(6)] for c in range(64)], dtype=torch.float32)       # cand bits
# S(c, s): bitmask of candidates whose 3 bits equal s
SMAP = torch.zeros(64, 8, dtype=torch.long)
for ci in range(64):
    for si in range(8):
        m = 0
        for r in range(2):
            if all(CFG64[ci, 3 * r + j] == CFG8[si, j] for j in range(3)):
                m |= 1 << r
        SMAP[ci, si] = m
SONE = nn.functional.one_hot(SMAP, 4).float()                                                         # [64,8,4]


def seq_weights(pc, ps):
    """pc [B,2,3] cand bit probs; ps [B,T,3] suffix bit probs -> W [B, 4^T] (sequence index = sum S_t * 4^t)."""
    B, T = ps.shape[0], ps.shape[1]
    pcf = pc.reshape(B, 6)
    Pc = torch.prod(pcf[:, None, :] * CFG64 + (1 - pcf[:, None, :]) * (1 - CFG64), -1)                  # [B,64]
    Ps = torch.prod(ps[:, :, None, :] * CFG8 + (1 - ps[:, :, None, :]) * (1 - CFG8), -1)                 # [B,T,8]
    A = torch.einsum("bts,csk->btck", Ps, SONE)                                                         # [B,T,64,4]
    acc = Pc[:, :, None]                                                                                 # [B,64,1]
    for t in range(T):
        acc = (A[:, t, :, :, None] * acc[:, :, None, :]).reshape(B, 64, -1)                              # new digit MOST significant (matches branch stacking)
    return acc.sum(1)                                                                                    # [B,4^T]


def gold_weights(gold, T):
    B = gold.shape[0]
    idx = torch.zeros(B, dtype=torch.long)
    for t in range(T):
        idx = idx + (1 << gold[:, t]) * (4 ** t)
    W = torch.zeros(B, 4 ** T); W[torch.arange(B), idx] = 1.0
    return W


class QChain(nn.Module):
    kind = "quantum"

    def __init__(self, seed, mode="AGR"):
        super().__init__()
        self.mode = mode
        g = torch.Generator().manual_seed(95700 + seed)
        self.verbs = nn.ModuleList([Circuit4(2, 3, generator=g) for _ in C.VERBS])
        self.qp, self.qn = Circuit4(2, 3, generator=g), Circuit4(2, 3, generator=g)
        self.morph = Morph(g, blind=(mode == "BLIND"))

    def weights(self, b):
        T = b["T"]
        if self.mode == "GOLD":
            return gold_weights(b["gold"], T)
        pc = self.morph.noun_p(b["cg"], b["cn"])
        ps = torch.stack([self.morph.verb_p(b["vg"][:, t], b["vn"][:, t]) for t in range(T)], 1)
        return seq_weights(pc, ps)

    def scores(self, b):
        B, T = b["cg"].shape[0], b["T"]
        nq = 2 + T
        V = torch.stack([m.matrix() for m in self.verbs])
        qs = [ry_state(b["theta"][:, 0]), ry_state(b["theta"][:, 1])] + [ry_state(b["theta_o"][:, t]) for t in range(T)]
        psi = qs[0]
        for q in qs[1:]:
            psi = psi[..., None] * q.reshape([B] + [1] * (psi.dim() - 1) + [2])
        states = psi[:, None]
        for t in range(T):
            Vt = V[b["v"][:, t]]
            nb = states.shape[1]
            flat = states.reshape([B * nb] + [2] * nq)
            Vr = Vt.repeat_interleave(nb, 0)
            a0 = apply_n(flat, Vr, (0, 2 + t), nq)
            s1 = apply_n(flat, Vr, (1, 2 + t), nq)
            s01 = apply_n(a0, Vr, (1, 2 + t), nq)
            # subset index S_t = bitmask {0:none, 1:{0}, 2:{1}, 3:{0,1}}; new digit most significant
            states = torch.stack([x.reshape(states.shape) for x in (flat, a0, s1, s01)], 1).reshape([B, 4 * nb] + [2] * nq)
        W = self.weights(b)
        NS = states.shape[1]
        flat = states.reshape([B * NS] + [2] * nq)
        qv = V[b["v"][torch.arange(B), b["q"]]].conj().transpose(1, 2).repeat_interleave(NS, 0)
        out = torch.zeros(B, 2)
        res_all = []
        for xw in (0, 1):
            per_q = []
            for qt in range(T):
                s = apply_n(flat, qv, (xw, 2 + qt), nq)
                r = []
                for E in (self.qp.matrix().conj().T, self.qn.matrix().conj().T):
                    tt = apply_n(s, E, (xw, 2 + qt), nq)
                    idx = [slice(None)] * (nq + 1); idx[1 + xw] = 0; idx[1 + 2 + qt] = 0
                    a = tt[tuple(idx)]
                    r.append(((a.real ** 2 + a.imag ** 2).reshape(B, NS, -1).sum(-1) * W).sum(1))
                per_q.append(torch.stack(r, -1))
            res_all.append(torch.stack(per_q, 1))                                  # [B,T,2]
        R = torch.stack(res_all, 1)                                                # [B,2,T,2]
        return R[torch.arange(B), b["x"], b["q"]]

    def loss(self, b):
        v = self.scores(b)
        return nn.functional.cross_entropy(torch.log(v + 1e-12), 1 - b["label"])

    def predict(self, b):
        v = self.scores(b)
        return (v[:, 0] > v[:, 1]).long(), (v[:, 0] - v[:, 1])


class ClChain(nn.Module):
    kind = "classical"

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(95700 + seed)
        self.morph = Morph(g)
        self.V = nn.Parameter(torch.eye(9).repeat(len(C.VERBS), 1, 1) + 0.1 * torch.randn(len(C.VERBS), 9, 9, generator=g))
        self.p = nn.Parameter(torch.randn(9, generator=g)); self.n = nn.Parameter(torch.randn(9, generator=g))
        self.bias = nn.Parameter(torch.zeros(()))

    def logit(self, b):
        B, T = b["cg"].shape[0], b["T"]
        nq = 2 + T
        u = lambda th: torch.stack([torch.ones_like(th), torch.cos(th), torch.sin(th)], -1)
        vecs = [u(b["theta"][:, 0]), u(b["theta"][:, 1])] + [u(b["theta_o"][:, t]) for t in range(T)]
        X = vecs[0]
        for v in vecs[1:]:
            X = X[..., None] * v.reshape([B] + [1] * (X.dim() - 1) + [3])
        def app(st, M, w, o):
            Mn = st.shape[0]
            perm = [0, 1 + w, 1 + o] + [1 + i for i in range(nq) if i not in (w, o)]
            s = (M @ st.permute(perm).reshape(Mn, 9, -1)).reshape([Mn, 3, 3] + [3] * (nq - 2))
            inv = [0] * (nq + 1)
            for i, p in enumerate(perm):
                inv[p] = i
            return s.permute(inv)
        states = X[:, None]
        for t in range(T):
            nb = states.shape[1]
            flat = states.reshape([B * nb] + [3] * nq)
            Vr = self.V[b["v"][:, t]].repeat_interleave(nb, 0)
            a0 = app(flat, Vr, 0, 2 + t); s1 = app(flat, Vr, 1, 2 + t); s01 = app(a0, Vr, 1, 2 + t)
            states = torch.stack([x.reshape(states.shape) for x in (flat, a0, s1, s01)], 1).reshape([B, 4 * nb] + [3] * nq)
        W = seq_weights(self.morph.noun_p(b["cg"], b["cn"]),
                        torch.stack([self.morph.verb_p(b["vg"][:, t], b["vn"][:, t]) for t in range(T)], 1))
        NS = states.shape[1]
        out = torch.zeros(B)
        Vq = self.V[b["v"][torch.arange(B), b["q"]]]
        res = torch.zeros(B, 2, T)
        for xw in (0, 1):
            for qt in range(T):
                perm = [0, 1, 2 + xw, 2 + 2 + qt] + [2 + i for i in range(nq) if i not in (xw, 2 + qt)]
                Z = states.permute(perm).reshape(B, NS, 9, -1)
                G = Z @ Z.transpose(-1, -2)
                G = G / (G.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] + 1e-9)
                Gm = (W[..., None, None] * G).sum(1)
                H = Vq.transpose(1, 2) @ Gm @ Vq
                res[:, xw, qt] = torch.einsum("i,bij,j->b", self.p, H, self.p) - torch.einsum("i,bij,j->b", self.n, H, self.n)
        return res[torch.arange(B), b["x"], b["q"]] + self.bias

    def loss(self, b):
        return nn.functional.binary_cross_entropy_with_logits(self.logit(b), b["label"].float())

    def predict(self, b):
        z = self.logit(b)
        return (z > 0).long(), z


class _Emb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.stem = nn.Embedding(len(C.STEMS), d); self.obj = nn.Embedding(len(C.OBJECTS), d)
        self.vstem = nn.Embedding(len(C.VERBS), d)
        self.ng = nn.Embedding(NG, d); self.nn_ = nn.Embedding(NN, d); self.vg = nn.Embedding(VG, d); self.vn = nn.Embedding(VN, d)
        self.isx = nn.Embedding(2, d); self.typ = nn.Embedding(4, d)          # 0 cls, 1 cand, 2 event, 3 question

    def tokens(self, b):
        B, T = b["cg"].shape[0], b["T"]
        tt = lambda i, n: self.typ(torch.full((B, n), i, dtype=torch.long))
        isx = (torch.arange(2)[None, :] == b["x"][:, None]).long()
        cand = self.stem(b["cs"]) + self.ng(b["cg"]) + self.nn_(b["cn"]) + self.isx(isx) + tt(1, 2)
        ev = self.vstem(b["v"]) + self.vg(b["vg"]) + self.vn(b["vn"]) + self.obj(b["o"]) + tt(2, T)
        qo = b["o"][torch.arange(B), b["q"]]; qvs = b["v"][torch.arange(B), b["q"]]
        qu = (self.vstem(qvs) + self.obj(qo))[:, None] + tt(3, 1)
        return tt(0, 1), cand, ev, qu


class TFChain(nn.Module):
    kind = "classical"

    def __init__(self, seed, d):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = _Emb(d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True), 2)
        self.out = nn.Linear(d, 1)

    def logit(self, b):
        cls, cand, ev, qu = self.emb.tokens(b)
        return self.out(self.enc(torch.cat([cls, cand, ev, qu], 1))[:, 0]).squeeze(-1)

    def loss(self, b):
        return nn.functional.binary_cross_entropy_with_logits(self.logit(b), b["label"].float())

    def predict(self, b):
        z = self.logit(b)
        return (z > 0).long(), z


class LSTMChain(TFChain):
    def __init__(self, seed, d):
        nn.Module.__init__(self)
        torch.manual_seed(seed)
        self.emb = _Emb(d)
        self.rnn = nn.LSTM(d, d, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * d, 1)

    def logit(self, b):
        _, cand, ev, qu = self.emb.tokens(b)
        _, (h, _) = self.rnn(torch.cat([cand, ev, qu], 1))
        return self.out(torch.cat([h[0], h[1]], -1)).squeeze(-1)


def full_statevector_scores(model, b):
    """K=2 chain with shared feature qubits, per-event suffix qubits, parity compute -> controlled V -> UNCOMPUTE."""
    B, T = b["cg"].shape[0], b["T"]
    nq = 2 + T + 3 * T + 6
    OBJ = lambda t: 2 + t; SUF = lambda t, j: 2 + T + 3 * t + j; FEAT = lambda r, j: 2 + 4 * T + 3 * r + j
    V = torch.stack([m.matrix() for m in model.verbs])
    pc = model.morph.noun_p(b["cg"], b["cn"])
    ang = lambda p: 2 * torch.asin(torch.sqrt(p.clamp(0, 1)))
    qs = [ry_state(b["theta"][:, 0]), ry_state(b["theta"][:, 1])] + [ry_state(b["theta_o"][:, t]) for t in range(T)]
    for t in range(T):
        ps = model.morph.verb_p(b["vg"][:, t], b["vn"][:, t])
        qs += [ry_state(ang(ps[:, j])) for j in range(3)]
    qs += [ry_state(ang(pc[:, r, j])) for r in range(2) for j in range(3)]
    psi = qs[0]
    for q in qs[1:]:
        psi = psi[..., None] * q.reshape([B] + [1] * (psi.dim() - 1) + [2])
    CN = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=CDTYPE)
    X = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE)
    for t in range(T):
        for r in range(2):
            for j in range(3):
                psi = apply_n(psi, CN, (SUF(t, j), FEAT(r, j)), nq); psi = apply_n(psi, X, (FEAT(r, j),), nq)
        Vt = V[b["v"][:, t]]
        for r in range(2):
            Cm = torch.eye(32, dtype=CDTYPE).repeat(B, 1, 1); Cm[:, 28:, 28:] = Vt
            psi = apply_n(psi, Cm, (FEAT(r, 0), FEAT(r, 1), FEAT(r, 2), r, OBJ(t)), nq)
        for r in range(2):                                                       # UNCOMPUTE (doc 90 §3)
            for j in range(3):
                psi = apply_n(psi, X, (FEAT(r, j),), nq); psi = apply_n(psi, CN, (SUF(t, j), FEAT(r, j)), nq)
    qv = V[b["v"][torch.arange(B), b["q"]]].conj().transpose(1, 2)
    out = []
    for i in range(B):
        xw, qt = int(b["x"][i]), int(b["q"][i])
        s = apply_n(psi[i:i + 1], qv[i:i + 1], (xw, OBJ(qt)), nq)
        r = []
        for E in (model.qp.matrix().conj().T, model.qn.matrix().conj().T):
            tt = apply_n(s, E, (xw, OBJ(qt)), nq)
            idx = [slice(None)] * (nq + 1); idx[1 + xw] = 0; idx[1 + OBJ(qt)] = 0
            a = tt[tuple(idx)]
            r.append((a.real ** 2 + a.imag ** 2).sum())
        out.append(torch.stack(r))
    return torch.stack(out)
