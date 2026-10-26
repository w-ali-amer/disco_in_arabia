# -*- coding: utf-8 -*-
"""exp119_tn.py -- exact tensor-network evaluation of DisCoCat diagrams for docs 119/120 (A0).

A lambeq grammar Diagram is walked layer by layer: a box with empty domain is a STATE on its codomain wires; a box
with non-empty domain is a MAP (e.g. an SRE pattern box n -> n); Cup contracts two wires (Bell effect, no
conjugation); Swap permutes; Id passes.  One open wire (s) remains.  Each diagram becomes an einsum over named
parameter tensors, and diagrams with the same structure are evaluated in one batched einsum.

Model classes (identical diagrams):
  Q -- every state is a unit-norm COMPLEX vector over its wires; every map is a UNITARY (matrix_exp of i*Hermitian);
       readout = Born rule |amp_s|^2, renormalised (post-selection on the cups).  This is the full family of
       circuits at these widths, not one ansatz.
  R -- (C-norm) unit-norm REAL states, ORTHOGONAL maps, readout amp_s^2 renormalised: a real-amplitude (rebit) model.
  CL -- (doc 121) as C, but the s-vector is read as LOGITS: p = softmax(amp) -- the standard classical tensor model.
  C -- (C-free) every state an unconstrained REAL tensor; every map an unconstrained real matrix; readout amp_s^2
       renormalised.  The classical comparator.
"""
import math, string
from collections import defaultdict
import torch
from lambeq.backend.grammar import Box, Cup, Swap, Ty

LETTERS = string.ascii_letters


def wire_dims(ty, dims):
    return [dims[str(t.name if hasattr(t, "name") else t)] for t in ty]


def atom(t):
    """Base name of an atomic type (n.l / n.r -> n)."""
    return t.name


class Structure:
    """Einsum recipe for one diagram shape: list of (box_name, kind, dims) factors and the einsum string."""

    def __init__(self, diagram, dims, shape_keyed=False):
        labels = []          # open wire labels, left to right
        lab_dim = {}
        factors = []         # (name, kind, [labels])
        next_lab = [0]

        def new(dim):
            l = next_lab[0]; next_lab[0] += 1; lab_dim[l] = dim; return l
        for layer in diagram.layers:
            left, box, right = layer.left, layer.box, layer.right
            off = len(left)
            k = len(box.dom)
            if isinstance(box, Cup):
                a, b = labels[off], labels[off + 1]
                factors.append(("__cup__", "cup", [a, b]))
                labels = labels[:off] + labels[off + 2:]
            elif isinstance(box, Swap):
                labels = labels[:off] + [labels[off + 1], labels[off]] + labels[off + 2:]
            elif k == 0:
                ls = [new(dims[atom(t)]) for t in box.cod]
                factors.append((box.name, "state", ls))
                labels = labels[:off] + ls + labels[off:]
            else:
                ins = labels[off:off + k]
                outs = [new(dims[atom(t)]) for t in box.cod]
                factors.append((box.name, "map", outs + ins))
                labels = labels[:off] + outs + labels[off + k:]
        assert len(labels) == 1, "expected one open wire, got %d" % len(labels)
        self.out = labels[0]
        self.factors = factors
        self.lab_dim = lab_dim
        # cups become identity contractions: merge labels instead of a factor
        parent = {l: l for l in lab_dim}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        for name, kind, ls in factors:
            if kind == "cup":
                parent[find(ls[1])] = find(ls[0])
        letter = {}
        for l in lab_dim:
            r = find(l)
            if r not in letter:
                letter[r] = LETTERS[len(letter)]
        self.terms = []
        for name, kind, ls in factors:
            if kind == "cup":
                continue
            ds = [lab_dim[l] for l in ls]
            nm = "%s|%s|%s" % (name, kind, ",".join(map(str, ds))) if shape_keyed else name   # exp119: one word, many types
            self.terms.append((nm, kind, ds, "".join(letter[find(l)] for l in ls)))
        self.out_letter = letter[find(self.out)]
        self.out_dim = lab_dim[self.out]
        self.key = (tuple((kind, tuple(d), s) for _, kind, d, s in self.terms), self.out_letter)


class TNModel(torch.nn.Module):
    def __init__(self, diagrams, dims, kind, seed, init=None, shape_keyed=False):
        super().__init__()
        self.kind = kind
        g = torch.Generator().manual_seed(seed)
        self.structs = [Structure(d, dims, shape_keyed) for d in diagrams]
        shapes = {}
        for st in self.structs:
            for name, k, ds, _ in st.terms:
                shapes[name] = (k, tuple(ds))
        self.names = sorted(shapes)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.params = torch.nn.ParameterList()
        for n in self.names:
            k, ds = shapes[n]
            size = math.prod(ds)
            if k == "state":
                shape = (2, size) if kind == "Q" else (size,)
                p = torch.randn(shape, generator=g, dtype=torch.float64) * (init if (init and kind in ("C", "CL")) else 1 / math.sqrt(size))
            else:
                d_out = math.prod(ds[:len(ds) // 2]); d_in = math.prod(ds[len(ds) // 2:])
                assert d_out == d_in, "pattern maps are square"
                sc = 0.1 if kind in ("Q", "R") else (init if init else 1 / math.sqrt(d_out))
                p = torch.randn((d_out, d_out), generator=g, dtype=torch.float64) * sc
            self.params.append(torch.nn.Parameter(p))
        self.shapes = shapes
        groups = defaultdict(list)
        for i, st in enumerate(self.structs):
            groups[st.key].append(i)
        self.groups = list(groups.items())

    def tensor(self, name):
        k, ds = self.shapes[name]
        p = self.params[self.index[name]]
        if k == "state":
            if self.kind == "Q":
                v = torch.complex(p[0], p[1]); v = v / v.abs().pow(2).sum().sqrt()
            elif self.kind == "R":
                v = p / p.pow(2).sum().sqrt()
            else:
                v = p
            return v.reshape(ds)
        d = p.shape[0]
        if self.kind == "Q":
            H = (p + p.T) / 2 + 1j * (p - p.T) / 2                     # Hermitian
            M = torch.linalg.matrix_exp(1j * H)
        elif self.kind == "R":
            M = torch.linalg.matrix_exp((p - p.T) / 2)                  # orthogonal
        else:
            M = p
        return M.reshape(ds)

    def forward(self, idx):
        """Class-1 probability for diagrams idx (list of ints)."""
        out = torch.zeros(len(idx), dtype=torch.float64)
        pos = {i: j for j, i in enumerate(idx)}
        cache = {}
        for key, members in self.groups:
            mem = [i for i in members if i in pos]
            if not mem:
                continue
            st0 = self.structs[mem[0]]
            ops, subs = [], []
            for t_i, (_, kind, ds, s) in enumerate(st0.terms):
                ts = []
                for i in mem:
                    nm = self.structs[i].terms[t_i][0]
                    if nm not in cache:
                        cache[nm] = self.tensor(nm)
                    ts.append(cache[nm])
                ops.append(torch.stack(ts)); subs.append("Z" + s)
            expr = ",".join(subs) + "->Z" + st0.out_letter
            amp = torch.einsum(expr, *ops)
            if self.kind == "CL":                                       # doc 121: standard classical logit readout
                pr = torch.softmax(amp, dim=1)
            else:
                pr = amp.abs().pow(2) if self.kind == "Q" else amp.pow(2)     # quadratic readout for Q / R / C
                pr = pr / pr.sum(1, keepdim=True).clamp_min(1e-300)
            out[[pos[i] for i in mem]] = pr[:, 1]
        return out

    def n_params(self):
        return sum(p.numel() for p in self.params)


def train_eval(diagrams, labels, tr, te, dims, kind, seed, epochs=300, lr=0.05, init=None, shape_keyed=False):
    m = TNModel(diagrams, dims, kind, seed, init, shape_keyed)
    y = torch.tensor(labels, dtype=torch.float64)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        p = m(tr).clamp(1e-9, 1 - 1e-9)
        loss = torch.nn.functional.binary_cross_entropy(p, y[tr])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pt = m(te) if len(te) else torch.zeros(0, dtype=torch.float64)
        ptr = m(tr)
    return {"test_pred": (pt > 0.5).long().tolist(), "test_acc": float(((pt > 0.5).double() == y[te]).double().mean()),
            "train_acc": float(((ptr > 0.5).double() == y[tr]).double().mean()), "final_loss": float(loss.detach()), "n_params": m.n_params()}
