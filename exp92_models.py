# -*- coding: utf-8 -*-
"""exp92_models.py -- doc 87 §4: Q-COMP, Q-CELL, OP-ORACLE, CL-DISCO, CL-MLP-FACT.

Every model maps a compiled batch (pair ids, form ids, noun ids [B,2], q_pos [B], state ids [B]) to a
logit-like score; `prob_yes` returns p in (0,1) for quantum arms (BCE on p) and a logit for classical arms.
Two wires always (doc 87 §3). Quantum readout: p = <e_S| rho_N |e_S>, marginal on the queried wire,
computed as re^2 + im^2 (doc 83 §11.3: complex64 abs() backward is NaN at subnormal amplitudes).
"""
import math
import torch
import torch.nn as nn
from torch_sv_sim import Circuit4, CDTYPE, euler_1q

FORMS = ["I", "VII", "IV"]
SWAP = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=CDTYPE)


def ry_state(theta):
    """RY(theta)|0> for a batch of angles -> [B,2] complex."""
    return torch.stack([torch.cos(theta / 2), torch.sin(theta / 2)], -1).to(CDTYPE)


class _Q(nn.Module):
    kind = "quantum"

    def __init__(self, n_pairs, n_nouns, n_states, seed, blocks):
        super().__init__()
        g = torch.Generator().manual_seed(92700 + seed)
        self.blocks = nn.ModuleList([Circuit4(2, 3, generator=g) for _ in range(blocks)])
        self.noun = nn.Parameter(torch.rand(n_nouns, generator=g) * math.pi)
        self.eff = nn.Parameter(torch.rand(n_states, 3, generator=g) * (2 * math.pi))

    def verb_mats(self, pair, form):
        raise NotImplementedError

    def prob_yes(self, pair, form, nouns, qpos, state):
        B = pair.shape[0]
        V = self.verb_mats(pair, form)                                   # [B,4,4]
        q0, q1 = ry_state(self.noun[nouns[:, 0]]), ry_state(self.noun[nouns[:, 1]])
        psi = (q0[:, :, None] * q1[:, None, :]).reshape(B, 4)            # wire0 major
        phi = torch.einsum("bij,bj->bi", V, psi).reshape(B, 2, 2)        # [b, i0, i1]
        E = self.eff[state]
        U = torch.stack([euler_1q(e[0], e[1], e[2]) for e in E])        # [B,2,2]; e_S = U|0>
        # amplitude of |e_S> on the queried wire: <e_S| = (U^dag)[0,:]  -> row 0 of U^dag = conj(U[:,0])
        bra = U[:, :, 0].conj()                                          # [B,2]
        a0 = torch.einsum("bk,bkj->bj", bra, phi)                        # queried wire 0: contract i0
        a1 = torch.einsum("bk,bjk->bj", bra, phi)                        # queried wire 1: contract i1
        a = torch.where((qpos == 0)[:, None], a0, a1)
        return (a.real ** 2 + a.imag ** 2).sum(-1)

    def loss(self, batch):
        p = self.prob_yes(*batch[:5]).clamp(1e-6, 1 - 1e-6)
        y = batch[5].float()
        return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean(), (p > 0.5).long()


class QComp(_Q):
    """Shared root block R_r; derived form f: F_f^dag R_r F_f with ONE learned F_f per form (doc 87 §4)."""
    def __init__(self, n_pairs, n_nouns, n_states, seed, oracle=False, stopgrad=False):
        super().__init__(n_pairs, n_nouns, n_states, seed, n_pairs)
        self.oracle = oracle
        self.stopgrad = stopgrad   # Q-COMP-SG (doc 87 §12.1): derived items cannot update root blocks
        g = torch.Generator().manual_seed(92800 + seed)
        self.F = nn.ModuleList([Circuit4(2, 3, generator=g) for _ in range(2)])   # VII, IV
        if oracle:
            for p in self.F.parameters():
                p.requires_grad_(False)

    def F_mat(self, fi):
        return SWAP if self.oracle else self.F[fi - 1].matrix()

    def verb_mats(self, pair, form):
        out = []
        cache = {}
        for p, f in zip(pair.tolist(), form.tolist()):
            k = (p, f)
            if k not in cache:
                R = self.blocks[p].matrix()
                if f == 0:
                    cache[k] = R
                else:
                    Fm = self.F_mat(f)
                    Rd = R.detach() if self.stopgrad else R
                    cache[k] = Fm.conj().T @ Rd @ Fm
            out.append(cache[k])
        return torch.stack(out)


class QCell(_Q):
    """Per-(root, form) blocks, no operator: block index = 2*pair + (form != I)."""
    def __init__(self, n_pairs, n_nouns, n_states, seed):
        super().__init__(n_pairs, n_nouns, n_states, seed, 2 * n_pairs)

    def verb_mats(self, pair, form):
        cache = {}
        out = []
        for p, f in zip(pair.tolist(), form.tolist()):
            k = 2 * p + (1 if f else 0)
            if k not in cache:
                cache[k] = self.blocks[k].matrix()
            out.append(cache[k])
        return torch.stack(out)


class CLDisco(nn.Module):
    """Real twin: u_n=(1,cos,sin); x=u0(x)u1; R_r 9x9 (I+N(0,.1^2)); F_f=expm(A-A^T), A~N(0,1); logit=e^T M_w e - b."""
    kind = "classical"

    def __init__(self, n_pairs, n_nouns, n_states, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.noun = nn.Parameter(torch.rand(n_nouns) * math.pi)
        self.R = nn.Parameter(torch.eye(9).repeat(n_pairs, 1, 1) + 0.1 * torch.randn(n_pairs, 9, 9))
        self.A = nn.Parameter(torch.randn(2, 9, 9))
        self.e = nn.Parameter(torch.randn(n_states, 3))
        self.b = nn.Parameter(torch.zeros(n_states))

    def logit(self, pair, form, nouns, qpos, state):
        B = pair.shape[0]
        th = self.noun[nouns]                                             # [B,2]
        u = torch.stack([torch.ones_like(th), torch.cos(th), torch.sin(th)], -1)   # [B,2,3]
        x = (u[:, 0, :, None] * u[:, 1, None, :]).reshape(B, 9)
        F = torch.matrix_exp(self.A - self.A.transpose(1, 2))             # [2,9,9]
        R = self.R[pair]
        Ff = F[(form - 1).clamp(min=0)]
        Rd = torch.where((form > 0)[:, None, None], Ff.transpose(1, 2) @ R @ Ff, R)
        Y = torch.einsum("bij,bj->bi", Rd, x).reshape(B, 3, 3)
        M0 = Y @ Y.transpose(1, 2)
        M1 = Y.transpose(1, 2) @ Y
        M = torch.where((qpos == 0)[:, None, None], M0, M1)
        e = self.e[state]
        return torch.einsum("bi,bij,bj->b", e, M, e) - self.b[state]

    def loss(self, batch):
        z = self.logit(*batch[:5])
        return nn.functional.binary_cross_entropy_with_logits(z, batch[5].float()), (z > 0).long()


class CLMLPFact(nn.Module):
    """One-hots [root; form; noun1; noun2; qpos; state] -> 64 ReLU -> 1."""
    kind = "classical"

    def __init__(self, n_pairs, n_nouns, n_states, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.dims = (n_pairs, 3, n_nouns, n_nouns, 2, n_states)
        self.net = nn.Sequential(nn.Linear(sum(self.dims), 64), nn.ReLU(), nn.Linear(64, 1))

    def logit(self, pair, form, nouns, qpos, state):
        parts = [pair, form, nouns[:, 0], nouns[:, 1], qpos, state]
        x = torch.cat([nn.functional.one_hot(v, d).float() for v, d in zip(parts, self.dims)], -1)
        return self.net(x).squeeze(-1)

    def loss(self, batch):
        z = self.logit(*batch[:5])
        return nn.functional.binary_cross_entropy_with_logits(z, batch[5].float()), (z > 0).long()


ARMS = {
    "Q-COMP": lambda P, N, S, s: QComp(P, N, S, s),
    "Q-CELL": lambda P, N, S, s: QCell(P, N, S, s),
    "OP-ORACLE": lambda P, N, S, s: QComp(P, N, S, s, oracle=True),
    "Q-COMP-SG": lambda P, N, S, s: QComp(P, N, S, s, stopgrad=True),
    "CL-DISCO": lambda P, N, S, s: CLDisco(P, N, S, s),
    "CL-MLP-FACT": lambda P, N, S, s: CLMLPFact(P, N, S, s),
}


def _rho1(phi, wire):
    """phi [B,2,2] (i0,i1) -> reduced density on `wire`, [B,2,2]."""
    if wire == 0:
        return torch.einsum("bij,bkj->bik", phi, phi.conj())
    return torch.einsum("bji,bjk->bik", phi, phi.conj())


def _fid(r, s):
    """Uhlmann fidelity for 2x2 density matrices: Tr(r s) + 2 sqrt(det r det s)."""
    tr = torch.einsum("bij,bji->b", r, s).real
    dr = torch.linalg.det(r).real.clamp(min=0); ds = torch.linalg.det(s).real.clamp(min=0)
    return (tr + 2 * torch.sqrt(dr * ds)).clamp(max=1.0)


def routing_fidelity(model, root_ids, fi, n_inputs=64, seed=92990):
    """doc 87 §7 (registered descriptive): mean fidelity, over random product inputs and the given roots, between
    each wire's reduced state under F_f^dag R F_f and under SWAP R SWAP with the SAME trained R."""
    g = torch.Generator().manual_seed(seed)
    th = torch.rand(n_inputs, 2, generator=g) * math.pi
    psi = (ry_state(th[:, 0])[:, :, None] * ry_state(th[:, 1])[:, None, :]).reshape(n_inputs, 4)
    out = {"wire0": [], "wire1": []}
    with torch.no_grad():
        Fm = model.F_mat(fi)
        for r in root_ids:
            Rm = model.blocks[r].matrix()
            A = (Fm.conj().T @ Rm @ Fm @ psi.T).T.reshape(-1, 2, 2)
            B = (SWAP @ Rm @ SWAP @ psi.T).T.reshape(-1, 2, 2)
            for w in (0, 1):
                out["wire%d" % w].append(_fid(_rho1(A, w), _rho1(B, w)).mean().item())
    return {k: sum(v) / len(v) for k, v in out.items()}
