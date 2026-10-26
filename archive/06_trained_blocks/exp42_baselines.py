# -*- coding: utf-8 -*-
"""exp42_baselines.py -- matched baselines B1/B2/B3 (doc 22 SS4).

B1 ClassicalDisCoCirc -- THE decisive baseline (H-Q):
  identical composition tree, consumed from the same gold events via the
  same CompiledStory objects the quantum arms use. Referent = unit vector
  in R^d (d=2 default, d=4 config). Verb = trained real linear maps with
  the SAME role-dressing structure as A2:
      M_v = (D0 (x) D1) . E . (D2 (x) D3)
  D* = general d x d real dressings (agent-slot / patient-slot), E =
  expm(t*G1 + p*G2) with G1, G2 fixed exchange-symmetric rotation
  generators mirroring A2's 2-param symmetric core. The global state is a
  real rank-K tensor (the honest classical tensor-network semantics of
  DisCoCirc -- exactly what Duneau et al. concede "would not require a
  quantum computer at test time"); renormalized after each event (general
  linear maps are not isometries; the normalization cancels in the
  softmax). Readout mirrors the quantum protocol: adjoint (transpose) of
  the question verb's map on the queried pair, then two trained effect
  maps, P(queried == 00) with other wires traced.
  Param count (d=2): 15 verbs x (4*4 dressing + 2 core) + 2*16 effects
  = 302 vs A2's 240 -> ratio 1.26, within the pre-registered x1.5 (B1
  slightly OVER-parameterized -- conservative for H-Q).

B2 GRU over SURFACE tokens (order variation and pro-drop are real for it)
  with name-augmentation (referent names resampled within type across
  training copies -- the fix to Duneau's baseline asymmetry). Matched
  variant AND ~10k-param generous variant.

B3 bag-of-embeddings + MLP -- must sit at chance (replicates exp41 V1).

All torch, all trained with the same loss family (cross-entropy), same
splits, same optimizer (runner-owned), 5 seeds, selection on ID val only.
"""

import math

import torch
from torch import nn


# =====================================================================
# B1 -- classical DisCoCirc (structure-matched, decisive)
# =====================================================================
def _pair_swap(d):
    """Permutation matrix of the wire exchange on R^d (x) R^d."""
    D = d * d
    S = torch.zeros(D, D)
    for i in range(d):
        for j in range(d):
            S[j * d + i, i * d + j] = 1.0
    return S


def _sym_generators(d):
    """Two fixed antisymmetric generators commuting with the pair swap
    (rotations inside the exchange-symmetric subspace), mirroring the
    2-param symmetric core of A2. [expm(aG1+bG2), SWAP] = 0."""
    D = d * d
    u = torch.zeros(D); u[0] = 1.0                        # e_(0,0)
    s = torch.zeros(D)                                    # (e01+e10)/sqrt2
    s[0 * d + 1] = s[1 * d + 0] = 1.0 / math.sqrt(2.0)
    w = torch.zeros(D); w[1 * d + 1] = 1.0                # e_(1,1)
    G1 = torch.outer(s, u) - torch.outer(u, s)
    G2 = torch.outer(s, w) - torch.outer(w, s)
    return G1, G2


def apply_real(state, M, wires):
    """Real tensordot gate application; NO norm assumption."""
    k = len(wires)
    d = state.shape[wires[0]]
    Mt = M.reshape((d,) * (2 * k))
    out = torch.tensordot(Mt, state, dims=(list(range(k, 2 * k)), list(wires)))
    return torch.movedim(out, list(range(k)), list(wires))


def real_prob_all_zero_on(state, wires):
    """Sum over all other-wire indices of amplitude^2 at queried == 0."""
    s = state
    for w in sorted(wires, reverse=True):
        s = s.select(w, 0)
    return (s ** 2).sum()


class ClassicalDisCoCirc(nn.Module):
    def __init__(self, verbs, seed, d_ref=2):
        super().__init__()
        assert d_ref in (2, 4), "doc 22 SS4: referent = R^2, config R^4"
        self.d = d_ref
        self.verbs = list(verbs)
        g = torch.Generator().manual_seed(seed)
        D = d_ref * d_ref

        def near_eye(n):
            return nn.Parameter(torch.eye(n) +
                                0.1 * torch.randn(n, n, generator=g))

        self.dress = nn.ParameterDict({
            v: nn.Parameter(torch.eye(self.d).repeat(4, 1, 1) +
                            0.1 * torch.randn(4, self.d, self.d, generator=g))
            for v in self.verbs})
        self.core = nn.ParameterDict({
            v: nn.Parameter(torch.rand(2, generator=g) * 2 * math.pi)
            for v in self.verbs})
        self.q_pos = near_eye(D)
        self.q_neg = near_eye(D)
        G1, G2 = _sym_generators(self.d)
        self.register_buffer("G1", G1)
        self.register_buffer("G2", G2)

    def verb_matrix(self, v):
        d0, d1, d2, d3 = self.dress[v]
        t = self.core[v]
        E = torch.linalg.matrix_exp(t[0] * self.G1 + t[1] * self.G2)
        return torch.kron(d0, d1) @ E @ torch.kron(d2, d3)

    def _intro_vec(self, angle):
        vec = torch.zeros(self.d)
        vec[0] = math.cos(angle / 2.0)
        vec[1] = math.sin(angle / 2.0)
        return vec                                        # same Ry|0> amps

    def story_values(self, cs, verb_map=None):
        vm = verb_map or {}
        cache = {}

        def mat(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)
            return cache[v]

        psi = self._intro_vec(cs.intro[0][2])
        for _w, _n, angle in cs.intro[1:]:
            psi = torch.einsum("...,j->...j", psi, self._intro_vec(angle))
        for v, wa, wp in cs.events:
            psi = apply_real(psi, mat(v), (wa, wp))
            psi = psi / (psi.norm() + 1e-12)              # cancels in softmax

        vseq, qa, qp = cs.question
        qw = (qa, qp)
        for qv in reversed(vseq):                         # adjoint effect
            psi = apply_real(psi, mat(qv).T, qw)
        vals = []
        for Q in (self.q_pos, self.q_neg):
            phi = apply_real(psi, Q, qw)
            vals.append(real_prob_all_zero_on(phi, qw))
        return torch.stack(vals)

    def story_loss(self, cs):
        vals = self.story_values(cs)
        target = torch.tensor([cs.answer])
        logits = torch.log(vals + 1e-12)                  # same loss as A*
        loss = nn.functional.cross_entropy(logits.unsqueeze(0), target)
        return loss, int(torch.argmax(vals).item())

    def predict(self, cs, verb_map=None):
        with torch.no_grad():
            return int(torch.argmax(self.story_values(cs, verb_map)).item())

    def param_counts(self):
        D = self.d * self.d
        per_verb = 4 * self.d * self.d + 2
        readout = 2 * D * D
        total = sum(p.numel() for p in self.parameters())
        assert total == per_verb * len(self.verbs) + readout
        return {"arm": "B1", "d_ref": self.d, "per_verb": per_verb,
                "n_verbs": len(self.verbs), "readout_shared": readout,
                "total_trained": total}


# =====================================================================
# B2 -- GRU over surface tokens (+ name augmentation)
# =====================================================================
def build_vocab(all_tokens_iter):
    vocab = {"<unk>": 0}
    for toks in all_tokens_iter:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def sample_name_mapping(humans, objects, rng):
    """Type-preserving referent-name bijection (name augmentation)."""
    hs, os_ = list(humans), list(objects)
    hp, op = hs[:], os_[:]
    rng.shuffle(hp)
    rng.shuffle(op)
    m = dict(zip(hs, hp))
    m.update(zip(os_, op))
    return m


def remap_tokens(tokens, mapping):
    return [mapping.get(t, t) for t in tokens]


class SeqBaseline(nn.Module):
    """B2 GRU. variant='matched' (~within x1.5 of A2's 240 trained params)
    or 'generous' (~10k)."""

    def __init__(self, vocab, seed, variant="matched"):
        super().__init__()
        torch.manual_seed(seed)
        self.vocab = vocab
        self.variant = variant
        emb, hid = (3, 4) if variant == "matched" else (20, 44)
        self.emb = nn.Embedding(len(vocab), emb)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.head = nn.Linear(hid, 2)

    def _idx(self, tokens):
        return torch.tensor([[self.vocab.get(t, 0) for t in tokens]])

    def logits(self, tokens):
        x = self.emb(self._idx(tokens))
        _, h = self.gru(x)
        return self.head(h[-1])                           # (1, 2)

    def story_loss(self, cs, mapping=None):
        toks = remap_tokens(cs.tokens, mapping) if mapping else cs.tokens
        logits = self.logits(toks)
        target = torch.tensor([cs.answer])
        loss = nn.functional.cross_entropy(logits, target)
        return loss, int(torch.argmax(logits[0]).item())

    def predict(self, cs, tokens=None):
        with torch.no_grad():
            lg = self.logits(tokens if tokens is not None else cs.tokens)
            return int(torch.argmax(lg[0]).item())

    def param_counts(self):
        return {"arm": "B2" if self.variant == "matched" else "B2_generous",
                "vocab": len(self.vocab),
                "total_trained": sum(p.numel() for p in self.parameters())}


# =====================================================================
# B3 -- bag-of-embeddings + MLP (must replicate exp41 V1 chance)
# =====================================================================
class BagBaseline(nn.Module):
    def __init__(self, vocab, seed, emb=3, hidden=16):
        super().__init__()
        torch.manual_seed(seed)
        self.vocab = vocab
        self.emb = nn.Embedding(len(vocab), emb)
        self.mlp = nn.Sequential(nn.Linear(emb, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def logits(self, tokens):
        idx = torch.tensor([self.vocab.get(t, 0) for t in tokens])
        return self.mlp(self.emb(idx).mean(dim=0)).unsqueeze(0)

    def story_loss(self, cs, mapping=None):
        toks = remap_tokens(cs.tokens, mapping) if mapping else cs.tokens
        logits = self.logits(toks)
        target = torch.tensor([cs.answer])
        loss = nn.functional.cross_entropy(logits, target)
        return loss, int(torch.argmax(logits[0]).item())

    def predict(self, cs, tokens=None):
        with torch.no_grad():
            lg = self.logits(tokens if tokens is not None else cs.tokens)
            return int(torch.argmax(lg[0]).item())

    def param_counts(self):
        return {"arm": "B3", "vocab": len(self.vocab),
                "total_trained": sum(p.numel() for p in self.parameters())}
