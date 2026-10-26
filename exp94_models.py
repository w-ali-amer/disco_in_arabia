# -*- coding: utf-8 -*-
"""exp94_models.py -- doc 89 §4/§5: Q-AGR / Q-BLIND / Q-GOLD (exact-mixture simulation), CL-AGR (classical twin),
TF and LSTM baselines (information parity), and the full-statevector reference used by G-SIM.

Feature bits per entity (g, n1, n2): each morpheme box sets RY angles on the bits it specifies; p(bit=1)=sin^2(a/2).
Noun boxes and verb boxes are separate parameters. Fused number morphemes (UN/AT, UW/NA) also set the gender bit.
Control of candidate r = AND_j [cand_bit_rj == suffix_bit_j]  (CNOT/X parity + 3-controlled V in the circuit).
Exact mixture (doc 89 §4): referent state = sum_S W(S) U_S rho U_S^dag, U_S = prod_{r in S, intro order} V(r, obj),
W(S) = sum_c P(c) prod_r [m_r(c) if r in S else 1 - m_r(c)], c over the 8 suffix-bit configurations.
"""
import math
import torch
import torch.nn as nn
from torch_sv_sim import Circuit4, CDTYPE
import exp94_common as C

NG, NN, VG, VN = len(C.NOUN_G), len(C.NOUN_N), len(C.VERB_G), len(C.VERB_N)
N_FUSED = [C.NOUN_N.index(k) for k in ("UN", "AT")]
V_FUSED = [C.VERB_N.index(k) for k in ("UW", "NA")]
CONFIGS = torch.tensor([[(c >> j) & 1 for j in range(3)] for c in range(8)], dtype=torch.float32)   # [8,3]


def ry_state(theta):
    return torch.stack([torch.cos(theta / 2), torch.sin(theta / 2)], -1).to(CDTYPE)


def apply_n(state, U, wires, nq):
    """state [M, 2]*nq ; U [d,d] or [M,d,d]."""
    M = state.shape[0]; k = len(wires)
    perm = [0] + [1 + w for w in wires] + [1 + w for w in range(nq) if w not in wires]
    s = state.permute(perm).reshape(M, 2 ** k, -1)
    s = (U @ s) if U.dim() == 3 else torch.einsum("ij,mjr->mir", U, s)
    s = s.reshape([M] + [2] * nq)
    inv = [0] * (nq + 1)
    for i, p in enumerate(perm):
        inv[p] = i
    return s.permute(inv)


class Morph(nn.Module):
    """Morpheme boxes -> per-bit probabilities for nouns and for the event verb."""
    def __init__(self, g, blind=False):
        super().__init__()
        r = lambda *s: nn.Parameter(torch.rand(*s, generator=g) * 2 * math.pi)
        self.ng, self.nn_, self.nfg = r(NG), r(NN, 2), r(NN)
        self.blind = blind
        if blind:
            self.vconst = r(3)
        else:
            self.vg, self.vn, self.vfg = r(VG), r(VN, 2), r(VN)

    @staticmethod
    def _angles(gid, nid, g, n, fg, fused):
        is_f = torch.zeros_like(nid, dtype=torch.bool)
        for f in fused:
            is_f |= nid == f
        ag = torch.where(is_f, fg[nid], g[gid])
        return torch.cat([ag[..., None], n[nid]], -1)             # [..., 3]

    def noun_p(self, gid, nid):
        return torch.sin(self._angles(gid, nid, self.ng, self.nn_, self.nfg, N_FUSED) / 2) ** 2

    def verb_p(self, gid, nid):
        if self.blind:
            return (torch.sin(self.vconst / 2) ** 2).expand(gid.shape[0], 3)
        return torch.sin(self._angles(gid, nid, self.vg, self.vn, self.vfg, V_FUSED) / 2) ** 2

    def table(self):
        out = {"noun": {}, "verb": {}}
        with torch.no_grad():
            for cls in C.CLASSES:
                for side, segf, pf, G, N in (("noun", C.segment_noun, self.noun_p, C.NOUN_G, C.NOUN_N),
                                             ("verb", C.segment_verb, self.verb_p, C.VERB_G, C.VERB_N)):
                    tok = C.noun_surface(C.STEMS[0], cls) if side == "noun" else C.verb_surface(C.VERBS[0], cls)
                    _, gm, nm = segf(tok)
                    p = pf(torch.tensor([G.index(gm)]), torch.tensor([N.index(nm)]))[0]
                    out[side]["%s-%s" % cls] = [round(float(v), 4) for v in p]
        return out


def branch_weights(pc, ps):
    """pc [B,K,3] candidate bit probs, ps [B,3] suffix bit probs -> W [B, 2^K], S index = sum 2^r [r in S]."""
    B, K, _ = pc.shape
    cfg = CONFIGS                                                          # [8,3]
    Pc = torch.prod(ps[:, None, :] * cfg + (1 - ps[:, None, :]) * (1 - cfg), -1)          # [B,8]
    m = torch.prod(pc[:, None, :, :] * cfg[None, :, None, :] + (1 - pc[:, None, :, :]) * (1 - cfg[None, :, None, :]),
                   -1)                                                     # [B,8,K]
    S = torch.arange(2 ** K)
    bits = ((S[:, None] >> torch.arange(K)) & 1).float()                  # [NS,K]
    f = m[:, :, None, :] * bits[None, None] + (1 - m[:, :, None, :]) * (1 - bits[None, None])   # [B,8,NS,K]
    return torch.einsum("bc,bcs->bs", Pc, torch.prod(f, -1))


class QAgr(nn.Module):
    kind = "quantum"

    def __init__(self, seed, mode="AGR"):
        super().__init__()
        self.mode = mode
        g = torch.Generator().manual_seed(94700 + seed)
        self.verbs = nn.ModuleList([Circuit4(2, 3, generator=g) for _ in C.VERBS])
        self.qp, self.qn = Circuit4(2, 3, generator=g), Circuit4(2, 3, generator=g)
        self.morph = Morph(g, blind=(mode == "BLIND"))

    def weights(self, b):
        K = b["K"]
        if self.mode == "GOLD":
            W = torch.zeros(b["gold"].shape[0], 2 ** K)
            W[torch.arange(W.shape[0]), 2 ** b["gold"]] = 1.0
            return W
        pc = self.morph.noun_p(b["cg"], b["cn"])                             # [B,K,3]
        ps = self.morph.verb_p(b["vg"], b["vn"])                             # [B,3]
        return branch_weights(pc, ps)

    def scores(self, b):
        B, K = b["cg"].shape[0], b["K"]
        nq = K + 1
        V = torch.stack([m.matrix() for m in self.verbs])
        qs = [ry_state(b["theta"][:, r]) for r in range(K)] + [ry_state(b["theta_o"])]
        psi = qs[0]
        for q in qs[1:]:
            psi = psi[..., None] * q.reshape([B] + [1] * (psi.dim() - 1) + [2])
        states = psi[:, None]                                                # [B,1,2..]
        Vb = V[b["v"]]
        for r in range(K):
            nb = states.shape[1]
            flat = states.reshape([B * nb] + [2] * nq)
            new = apply_n(flat, Vb.repeat_interleave(nb, 0), (r, K), nq).reshape(states.shape)
            states = torch.cat([states, new], 1)
        W = self.weights(b)                                                  # [B, 2^K]
        NS = states.shape[1]
        flat = states.reshape([B * NS] + [2] * nq)
        Vq = V[b["qv"]].conj().transpose(1, 2).repeat_interleave(NS, 0)
        qp, qn = self.qp.matrix().conj().T, self.qn.matrix().conj().T
        out = []
        for xw in range(K):
            s = apply_n(flat, Vq, (xw, K), nq)
            res = []
            for E in (qp, qn):
                t = apply_n(s, E, (xw, K), nq)
                idx = [slice(None)] * (nq + 1); idx[1 + xw] = 0; idx[1 + K] = 0
                a = t[tuple(idx)]
                res.append((a.real ** 2 + a.imag ** 2).reshape(B, NS, -1).sum(-1))
            out.append(torch.stack([(W * res[0]).sum(1), (W * res[1]).sum(1)], -1))
        out = torch.stack(out, 1)                                            # [B,K,2]
        return out[torch.arange(B), b["x"]]

    def loss(self, b):
        v = self.scores(b)
        return nn.functional.cross_entropy(torch.log(v + 1e-12), 1 - b["label"])

    def predict(self, b):
        v = self.scores(b)
        return (v[:, 0] > v[:, 1]).long(), (v[:, 0] - v[:, 1])


class ClAgr(nn.Module):
    """Classical DisCoCirc twin: identical control mixture; real u=(1,cos,sin); real 9x9 verbs; bilinear readout."""
    kind = "classical"

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(94700 + seed)
        self.morph = Morph(g)
        self.V = nn.Parameter(torch.eye(9).repeat(len(C.VERBS), 1, 1) + 0.1 * torch.randn(len(C.VERBS), 9, 9, generator=g))
        self.p = nn.Parameter(torch.randn(9, generator=g)); self.n = nn.Parameter(torch.randn(9, generator=g))
        self.bias = nn.Parameter(torch.zeros(()))

    def logit(self, b):
        B, K = b["cg"].shape[0], b["K"]
        nq = K + 1
        u = lambda th: torch.stack([torch.ones_like(th), torch.cos(th), torch.sin(th)], -1)
        vecs = [u(b["theta"][:, r]) for r in range(K)] + [u(b["theta_o"])]
        T = vecs[0]
        for v in vecs[1:]:
            T = T[..., None] * v.reshape([B] + [1] * (T.dim() - 1) + [3])
        states = T[:, None]
        Vb = self.V[b["v"]]
        def app(st, M, w):                                                   # st [M?, 3..], apply 9x9 on (w, K)
            Mn = st.shape[0]
            perm = [0, 1 + w, 1 + K] + [1 + i for i in range(nq) if i not in (w, K)]
            s = st.permute(perm).reshape(Mn, 9, -1)
            s = M @ s
            s = s.reshape([Mn, 3, 3] + [3] * (nq - 2))
            inv = [0] * (nq + 1)
            for i, p in enumerate(perm):
                inv[p] = i
            return s.permute(inv)
        for r in range(K):
            nb = states.shape[1]
            flat = states.reshape([B * nb] + [3] * nq)
            new = app(flat, Vb.repeat_interleave(nb, 0), r).reshape(states.shape)
            states = torch.cat([states, new], 1)
        W = branch_weights(self.morph.noun_p(b["cg"], b["cn"]), self.morph.verb_p(b["vg"], b["vn"]))
        NS = states.shape[1]
        out = []
        for xw in range(K):
            perm = [0, 1, 2 + xw, 2 + K] + [2 + i for i in range(nq) if i not in (xw, K)]
            Z = states.permute(perm).reshape(B, NS, 9, -1)
            G = Z @ Z.transpose(-1, -2)
            G = G / (G.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] + 1e-9)
            Gm = (W[..., None, None] * G).sum(1)                             # [B,9,9]
            Vq = self.V[b["qv"]]
            H = Vq.transpose(1, 2) @ Gm @ Vq
            out.append(torch.einsum("i,bij,j->b", self.p, H, self.p) - torch.einsum("i,bij,j->b", self.n, H, self.n))
        out = torch.stack(out, 1)
        return out[torch.arange(B), b["x"]] + self.bias

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
        self.isx = nn.Embedding(2, d); self.typ = nn.Embedding(5, d)          # 0 cls,1 cand,2 verb,3 obj,4 qverb

    def tokens(self, b):
        B, K = b["cg"].shape[0], b["K"]
        isx = (torch.arange(K)[None, :] == b["x"][:, None]).long()
        t = lambda i, n: self.typ(torch.full((B, n), i, dtype=torch.long))
        cand = self.stem(b["cs"]) + self.ng(b["cg"]) + self.nn_(b["cn"]) + self.isx(isx) + t(1, K)
        verb = (self.vstem(b["v"]) + self.vg(b["vg"]) + self.vn(b["vn"]))[:, None] + t(2, 1)
        obj = self.obj(b["o"])[:, None] + t(3, 1)
        qv = self.vstem(b["qv"])[:, None] + t(4, 1)
        return t(0, 1), cand, verb, obj, qv


class TFBase(nn.Module):
    """2-layer transformer; NO positional encoding (candidate set order-free; types distinguish roles)."""
    kind = "classical"

    def __init__(self, seed, d):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = _Emb(d)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(d, 1)

    def logit(self, b):
        cls, cand, verb, obj, qv = self.emb.tokens(b)
        h = self.enc(torch.cat([cls, cand, verb, obj, qv], 1))
        return self.out(h[:, 0]).squeeze(-1)

    def loss(self, b):
        return nn.functional.binary_cross_entropy_with_logits(self.logit(b), b["label"].float())

    def predict(self, b):
        z = self.logit(b)
        return (z > 0).long(), z


class LSTMBase(TFBase):
    """Bidirectional LSTM over [cands (intro order), verb, obj, qverb]; same summed token embeddings."""
    def __init__(self, seed, d):
        nn.Module.__init__(self)
        torch.manual_seed(seed)
        self.emb = _Emb(d)
        self.rnn = nn.LSTM(d, d, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * d, 1)

    def logit(self, b):
        _, cand, verb, obj, qv = self.emb.tokens(b)
        _, (h, _) = self.rnn(torch.cat([cand, verb, obj, qv], 1))
        return self.out(torch.cat([h[0], h[1]], -1)).squeeze(-1)


# ---------------------------------------------------------------- G-SIM reference (full statevector)
def full_statevector_scores(model, b):
    """Q-AGR on the full register: K cands + obj + 3 suffix + 3K feature qubits; CNOT/X parity; 3-controlled V."""
    B, K = b["cg"].shape[0], b["K"]
    nq = K + 1 + 3 + 3 * K
    SUF = K + 1; FEAT = lambda r, j: K + 4 + 3 * r + j
    V = torch.stack([m.matrix() for m in model.verbs])
    pc = model.morph.noun_p(b["cg"], b["cn"]); ps = model.morph.verb_p(b["vg"], b["vn"])
    ang = lambda p: 2 * torch.asin(torch.sqrt(p.clamp(0, 1)))
    qs = [ry_state(b["theta"][:, r]) for r in range(K)] + [ry_state(b["theta_o"])]
    qs += [ry_state(ang(ps[:, j])) for j in range(3)]
    qs += [ry_state(ang(pc[:, r, j])) for r in range(K) for j in range(3)]
    psi = qs[0]
    for q in qs[1:]:
        psi = psi[..., None] * q.reshape([B] + [1] * (psi.dim() - 1) + [2])
    CNOT = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=CDTYPE)
    X = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE)
    for r in range(K):
        for j in range(3):
            psi = apply_n(psi, CNOT, (SUF + j, FEAT(r, j)), nq)
            psi = apply_n(psi, X, (FEAT(r, j),), nq)
    Vb = V[b["v"]]
    for r in range(K):
        Cm = torch.eye(32, dtype=CDTYPE).repeat(B, 1, 1)
        Cm[:, 28:, 28:] = Vb
        psi = apply_n(psi, Cm, (FEAT(r, 0), FEAT(r, 1), FEAT(r, 2), r, K), nq)
    Vq = V[b["qv"]].conj().transpose(1, 2)
    out = []
    for xw in range(K):
        s = apply_n(psi, Vq, (xw, K), nq)
        res = []
        for E in (model.qp.matrix().conj().T, model.qn.matrix().conj().T):
            t = apply_n(s, E, (xw, K), nq)
            idx = [slice(None)] * (nq + 1); idx[1 + xw] = 0; idx[1 + K] = 0
            a = t[tuple(idx)]
            res.append((a.real ** 2 + a.imag ** 2).reshape(B, -1).sum(-1))
        out.append(torch.stack(res, -1))
    out = torch.stack(out, 1)
    return out[torch.arange(B), b["x"]]
