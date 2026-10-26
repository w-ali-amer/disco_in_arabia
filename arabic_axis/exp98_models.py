# -*- coding: utf-8 -*-
"""exp98_models.py -- doc 93 (REGISTERED 23d4dfb): per-event register under LEARNED control.

Reuses exp97 QRoom (base point x expm gates; per-event register wiring = "R-8-LOCAL" layout) with the exp95 learned
mixture control (Morph morpheme boxes + seq_weights over shared candidate-feature configs, uncomputed parity).
  Q-AGR-REG  : U(8) per event on (cand, O_t, RB_t), learned control     R-AGR-REG : SO(8), learned control
  Q-AGR-NOREG: U(4) on (cand, O_t), learned control                     Q-GOLD-REG: U(8), gold control (ceiling)
  Q-BLIND-REG: U(8), suffix-blind constant control (floor: identical within twins)
"""
import torch
import torch.nn as nn
import exp94_common as C
import exp95_models as M95
import exp97_models as M97
from exp94_models import Morph, apply_n
from torch_sv_sim import CDTYPE


class QReg(M97.QRoom):
    def __init__(self, seed, field, reg, mode):
        nn.Module.__init__(self)
        self.mode, self.field, self.reg = mode, field, reg
        self.kind = "R-8-LOCAL" if reg else "Q-INV"            # wiring layout only (per-event register or none)
        g = torch.Generator().manual_seed(98700 + seed)
        dim = 8 if reg else 4
        mk = (lambda: M97.Gate(M97.haar_u(g, dim), "C")) if field == "C" else (lambda: M97.Gate(M97.haar_so(g, dim), "R"))
        gates = [mk() for _ in range(len(C.VERBS) + 2)]
        self.verbs = nn.ModuleList(gates[:len(C.VERBS)])
        self.qp, self.qn = gates[len(C.VERBS)], gates[len(C.VERBS) + 1]
        self.morph = None if mode == "GOLD" else Morph(g, blind=(mode == "BLIND"))

    def weights(self, b):
        T = b["T"]
        if self.mode == "GOLD":
            return M95.gold_weights(b["gold"], T)
        pc = self.morph.noun_p(b["cg"], b["cn"])
        ps = torch.stack([self.morph.verb_p(b["vg"][:, t], b["vn"][:, t]) for t in range(T)], 1)
        return M95.seq_weights(pc, ps)


def full_statevector_scores(model, b):
    """G-SIM reference: K=2 chain, shared feature qubits, per-event suffix qubits, parity -> controlled V (on cand, O_t
    [, RB_t]) -> UNCOMPUTE; register wires |0> per event; readout V_q^dag + effects on (x, O_q[, RB_q])."""
    B, T = b["cg"].shape[0], b["T"]
    nr = T if model.reg else 0
    nq = 2 + T + nr + 3 * T + 6
    OBJ = lambda t: 2 + t; RB = lambda t: 2 + T + t; SUF = lambda t, j: 2 + T + nr + 3 * t + j
    FEAT = lambda r, j: 2 + T + nr + 3 * T + 3 * r + j
    V = torch.stack([m.matrix() for m in model.verbs])
    pc = model.morph.noun_p(b["cg"], b["cn"])
    ang = lambda p: 2 * torch.asin(torch.sqrt(p.clamp(0, 1)))
    qs = [M95.ry_state(b["theta"][:, 0]), M95.ry_state(b["theta"][:, 1])] + [M95.ry_state(b["theta_o"][:, t]) for t in range(T)]
    qs += [M95.ry_state(torch.zeros(B)) for _ in range(nr)]
    for t in range(T):
        ps = model.morph.verb_p(b["vg"][:, t], b["vn"][:, t]); qs += [M95.ry_state(ang(ps[:, j])) for j in range(3)]
    qs += [M95.ry_state(ang(pc[:, r, j])) for r in range(2) for j in range(3)]
    psi = qs[0]
    for q in qs[1:]:
        psi = psi[..., None] * q.reshape([B] + [1] * (psi.dim() - 1) + [2])
    CN = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=CDTYPE)
    X = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE)
    d = V.shape[-1]
    for t in range(T):
        for r in range(2):
            for j in range(3):
                psi = apply_n(psi, CN, (SUF(t, j), FEAT(r, j)), nq); psi = apply_n(psi, X, (FEAT(r, j),), nq)
        tgt = lambda r: (r, OBJ(t)) + ((RB(t),) if model.reg else ())
        for r in range(2):
            Cm = torch.eye(8 * d, dtype=CDTYPE).repeat(B, 1, 1); Cm[:, 7 * d:, 7 * d:] = V[b["v"][:, t]]
            psi = apply_n(psi, Cm, (FEAT(r, 0), FEAT(r, 1), FEAT(r, 2)) + tgt(r), nq)
        for r in range(2):
            for j in range(3):
                psi = apply_n(psi, X, (FEAT(r, j),), nq); psi = apply_n(psi, CN, (SUF(t, j), FEAT(r, j)), nq)
    out = []
    for i in range(B):
        xw, qt = int(b["x"][i]), int(b["q"][i])
        w = (xw, OBJ(qt)) + ((RB(qt),) if model.reg else ())
        s = apply_n(psi[i:i + 1], V[b["v"][i, qt]].conj().T, w, nq)
        r = []
        for E in (model.qp.matrix().conj().T, model.qn.matrix().conj().T):
            tt = apply_n(s, E, w, nq)
            idx = [slice(None)] * (nq + 1); idx[1 + xw] = 0; idx[1 + OBJ(qt)] = 0
            a = tt[tuple(idx)]; r.append((a.real ** 2 + a.imag ** 2).sum())
        out.append(torch.stack(r))
    return torch.stack(out)


ARMS = {"Q-AGR-REG": lambda s: QReg(s, "C", True, "AGR"), "R-AGR-REG": lambda s: QReg(s, "R", True, "AGR"),
        "Q-AGR-NOREG": lambda s: QReg(s, "C", False, "AGR"), "Q-GOLD-REG": lambda s: QReg(s, "C", True, "GOLD"),
        "Q-BLIND-REG": lambda s: QReg(s, "C", True, "BLIND")}
