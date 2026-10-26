# -*- coding: utf-8 -*-
"""exp93_models.py -- doc 88 §4: Q-AGR, Q-BLIND, Q-GOLD, SURF-MLP on a batched 6-qubit statevector.

Wires: 0,1 = S1 humans (surface order); 2 = S2 object; 3 = suffix box s; 4,5 = gender boxes g0,g1.
Q-AGR:  s=RY(sigma[suffix]); g_r=RY(gamma[fem_r]); CNOT(3->4), CNOT(3->5), X(4), X(5) => g_r=|1> iff match;
        S1: V1 on (0,1); S2: controlled-V2 (control 4, targets 0,2) then (control 5, targets 1,2).
Q-BLIND: s=RY(sigma_const) -- no suffix path (twins compile identically => exactly 0.5 per pair).
Q-GOLD: wires 4,5 prepared |gold>; no s, no parity.
Readout: V2(question)^dag on (x,2); vp=P(x,2=00) after Qp^dag, vn after Qn^dag; re^2+im^2 (doc 83 §11.3).
"""
import math
import torch
import torch.nn as nn
from torch_sv_sim import Circuit4, CDTYPE

NQ = 6
X_ = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE)
CNOT = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=CDTYPE)


def ry_state(theta):
    return torch.stack([torch.cos(theta / 2), torch.sin(theta / 2)], -1).to(CDTYPE)


def apply(state, U, wires):
    """state [B,2,...,2] (NQ), U [d,d] or [B,d,d] on `wires` (in order)."""
    B = state.shape[0]
    k = len(wires)
    perm = [0] + [1 + w for w in wires] + [1 + w for w in range(NQ) if w not in wires]
    s = state.permute(perm).reshape(B, 2 ** k, -1)
    s = (U @ s) if U.dim() == 3 else torch.einsum("ij,bjr->bir", U, s)
    s = s.reshape([B] + [2] * NQ)
    inv = [0] * (NQ + 1)
    for i, p in enumerate(perm):
        inv[p] = i
    return s.permute(inv)


def controlled(V):
    """[B,4,4] -> [B,8,8] = |0><0| (x) I + |1><1| (x) V, control first."""
    B = V.shape[0]
    C = torch.zeros(B, 8, 8, dtype=CDTYPE)
    C[:, :4, :4] = torch.eye(4, dtype=CDTYPE)
    C[:, 4:, 4:] = V
    return C


def p00(state, w0, w1):
    """P(wires w0,w1 = 00), others traced; re^2+im^2."""
    idx = [slice(None)] * (NQ + 1)
    idx[1 + w0] = 0; idx[1 + w1] = 0
    a = state[tuple(idx)]
    return (a.real ** 2 + a.imag ** 2).reshape(state.shape[0], -1).sum(-1)


class QStory(nn.Module):
    kind = "quantum"

    def __init__(self, v1_ids, v2_ids, seed, mode):
        super().__init__()
        assert mode in ("AGR", "BLIND", "GOLD")
        self.mode = mode
        g = torch.Generator().manual_seed(93700 + seed)
        self.v1 = nn.ModuleList([Circuit4(2, 3, generator=g) for _ in v1_ids])
        self.v2 = nn.ModuleList([Circuit4(2, 3, generator=g) for _ in v2_ids])
        self.qp, self.qn = Circuit4(2, 3, generator=g), Circuit4(2, 3, generator=g)
        self.sigma = nn.Parameter(torch.rand(2 if mode == "AGR" else 1, generator=g) * 2 * math.pi)
        self.gamma = nn.Parameter(torch.rand(2, generator=g) * 2 * math.pi)
        if mode == "GOLD":
            self.sigma.requires_grad_(False); self.gamma.requires_grad_(False)

    def scores(self, b):
        """b: dict of tensors: v1,v2,vq,theta0,theta1,theta_o,fem0,fem1,suffix,x_wire,gold."""
        B = b["v1"].shape[0]
        V1 = torch.stack([m.matrix() for m in self.v1]); V2 = torch.stack([m.matrix() for m in self.v2])
        st = torch.ones([B] + [1] * NQ, dtype=CDTYPE)
        qubits = [ry_state(b["theta0"]), ry_state(b["theta1"]), ry_state(b["theta_o"])]
        if self.mode == "AGR":
            qubits.append(ry_state(self.sigma[b["suffix"]]))
        elif self.mode == "BLIND":
            qubits.append(ry_state(self.sigma[0].expand(B)))
        else:
            qubits.append(ry_state(torch.zeros(B)))
        if self.mode == "GOLD":
            g0 = (b["gold"] == 0).float() * math.pi; g1 = (b["gold"] == 1).float() * math.pi
            qubits += [ry_state(g0), ry_state(g1)]
        else:
            qubits += [ry_state(self.gamma[b["fem0"]]), ry_state(self.gamma[b["fem1"]])]
        psi = qubits[0]
        for q in qubits[1:]:
            psi = (psi[..., None] * q.reshape([B] + [1] * (psi.dim() - 1) + [2]))
        st = psi
        if self.mode != "GOLD":
            st = apply(st, CNOT, (3, 4)); st = apply(st, CNOT, (3, 5))
            st = apply(st, X_, (4,)); st = apply(st, X_, (5,))
        st = apply(st, V1[b["v1"]], (0, 1))
        C2 = controlled(V2[b["v2"]])
        st = apply(st, C2, (4, 0, 2))
        st = apply(st, C2, (5, 1, 2))
        Vq = V2[b["vq"]].conj().transpose(1, 2)
        out = []
        for xw in (0, 1):
            s = apply(st, Vq, (xw, 2))
            vp = p00(apply(s, self.qp.matrix().conj().T, (xw, 2)), xw, 2)
            vn = p00(apply(s, self.qn.matrix().conj().T, (xw, 2)), xw, 2)
            out.append(torch.stack([vp, vn], -1))
        return torch.where((b["x_wire"] == 0)[:, None], out[0], out[1])       # [B,2]

    def loss(self, b):
        v = self.scores(b)
        logp = torch.log(v + 1e-12)
        loss = nn.functional.cross_entropy(logp, 1 - b["label"])            # class 0 = yes (vp)
        return loss, (v[:, 0] > v[:, 1]).long()

    def predict(self, b):
        v = self.scores(b)
        return (v[:, 0] > v[:, 1]).long(), (v[:, 0] - v[:, 1])

    def match_table(self):
        """P(control fires) for (suffix, gender) under Q-AGR: |<match>|^2 of the parity of two RY states."""
        if self.mode != "AGR":
            return None
        t = {}
        for sf in (0, 1):
            for gf in (0, 1):
                ps = torch.sin(self.sigma[sf] / 2) ** 2; pg = torch.sin(self.gamma[gf] / 2) ** 2
                t["suffix%d_gender%d" % (sf, gf)] = float((ps * pg + (1 - ps) * (1 - pg)).detach())
        return t


class SurfMLP(nn.Module):
    kind = "classical"

    def __init__(self, n_v1, n_v2, n_stems, n_obj, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.dims = (n_v1, n_v2, n_stems, n_stems, n_obj, 2)
        self.net = nn.Sequential(nn.Linear(sum(self.dims) + 4, 64), nn.ReLU(), nn.Linear(64, 1))

    def logit(self, b):
        parts = [b["v1"], b["v2"], b["stem0"], b["stem1"], b["obj"], b["x_wire"]]
        oh = [nn.functional.one_hot(v, d).float() for v, d in zip(parts, self.dims)]
        xfem = torch.where(b["x_wire"] == 0, b["fem0"], b["fem1"])
        bits = torch.stack([b["suffix"], b["fem0"], b["fem1"], xfem], -1).float()
        return self.net(torch.cat(oh + [bits], -1)).squeeze(-1)

    def loss(self, b):
        z = self.logit(b)
        return nn.functional.binary_cross_entropy_with_logits(z, b["label"].float()), (z > 0).long()

    def predict(self, b):
        z = self.logit(b)
        return (z > 0).long(), z
