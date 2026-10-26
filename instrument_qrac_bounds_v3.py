# -*- coding: utf-8 -*-
"""instrument_qrac_bounds_v3.py -- BOUND CALCULATOR v3 for doc 94 (exp99), pre-registration (doc 55 §19.2).
v1 (kept): classical exhaustive 0.75 (n=2,3), free qubit/rebit optima; its compositional check was WRONG (commuting boxes).
v2: KILLED before output (rebit boxes RY-only = SO(2), excluded real reflections -> would underestimate the rebit).
v3: COMPOSITIONAL optima with every morpheme box = base point x expm (exp97 discipline):
  qubit box  B = U0 expm(A - A^dag), U0 Haar U(2) per restart;  rebit box B = O0 expm(B - B^T), O0 Haar O(2) (det +-1),
  codeword(word) = B_n(b_n) ... B_1(b_1) |0>, one box per feature VALUE, fixed order; one measurement axis per query
  (qubit: S^2; rebit: the XZ circle). Metric: average single-shot success, uniform words x uniform queries.
Vectorised over words. Writes diagnostic_qrac_bounds_v3.json (Rule 3)."""
import itertools, json, math, os, hashlib
import torch
torch.set_num_threads(1)
OUT = "diagnostic_qrac_bounds_v3.json"; assert not os.path.exists(OUT)


def haar_u2(g):
    Z = torch.complex(torch.randn(2, 2, generator=g), torch.randn(2, 2, generator=g))
    Q, R = torch.linalg.qr(Z); d = torch.diagonal(R); return Q * (d / d.abs())[None, :]


def haar_o2(g):
    Q, R = torch.linalg.qr(torch.randn(2, 2, generator=g)); return Q * torch.sign(torch.diagonal(R))[None, :]


def run(n, rebit, restarts=300, steps=700, seed=3):
    g = torch.Generator().manual_seed(seed)
    words = torch.tensor(list(itertools.product((0, 1), repeat=n)))           # [W,n]
    sign = (1 - 2 * words).float()
    best = 0.0; best_detail = None
    for _ in range(restarts):
        if rebit:
            base = torch.stack([torch.stack([haar_o2(g) for _ in range(2)]) for _ in range(n)]).to(torch.complex64)  # [n,2,2,2]
            Gp = torch.zeros(n, 2, 2, 2, requires_grad=True)
            A = torch.randn(n, 2, generator=g).requires_grad_(True)             # axes in XZ plane (x,z)
            params = [Gp, A]
        else:
            base = torch.stack([torch.stack([haar_u2(g) for _ in range(2)]) for _ in range(n)])
            Gr = torch.zeros(n, 2, 2, 2, requires_grad=True); Gi = torch.zeros(n, 2, 2, 2, requires_grad=True)
            A = torch.randn(n, 3, generator=g).requires_grad_(True)
            params = [Gr, Gi, A]
        opt = torch.optim.Adam(params, lr=0.05)
        for _ in range(steps):
            if rebit:
                Bx = torch.matrix_exp(Gp - Gp.transpose(-1, -2)).to(torch.complex64)
            else:
                Gc = torch.complex(Gr, Gi); Bx = torch.matrix_exp(Gc - Gc.conj().transpose(-1, -2))
            Box = base @ Bx                                                     # [n,2(value),2,2]
            psi = torch.zeros(words.shape[0], 2, dtype=torch.complex64); psi[:, 0] = 1
            for i in range(n):
                M = Box[i][words[:, i]]                                         # [W,2,2]
                psi = torch.einsum("wij,wj->wi", M, psi)
            a, b = psi[:, 0], psi[:, 1]
            v = torch.stack([2 * (a.conj() * b).real, 2 * (a.conj() * b).imag, a.abs() ** 2 - b.abs() ** 2], 1)   # [W,3]
            if rebit:
                ax = torch.stack([A[:, 0], torch.zeros(n), A[:, 1]], 1)
            else:
                ax = A
            ax = ax / ax.norm(dim=1, keepdim=True).clamp(min=1e-9)
            p = 0.5 * (1 + sign * (v @ ax.T))                                   # [W,n]
            loss = -p.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        val = float(-loss.detach())
        if val > best:
            best = val; best_detail = {"imag_max_rebit": float(psi.imag.abs().max()) if rebit else None}
    return best, best_detail


v1 = json.load(open("diagnostic_qrac_bounds.json"))["results"]
res = {}
for n in (2, 3):
    for rebit in (False, True):
        val, det = run(n, rebit)
        res["compositional_%s_n%d" % ("rebit" if rebit else "qubit", n)] = val
        if det and det["imag_max_rebit"] is not None:
            res["compositional_rebit_n%d_imag_max" % n] = det["imag_max_rebit"]
res["reference_v1"] = {k: v for k, v in v1.items() if not k.startswith("compositional")}
res["analytic_rebit_n3_lit_agent"] = 0.5 + (1 + math.sqrt(5)) / 12
json.dump({"doc": "94 pre-registration bound calculator v3", "results": res,
           "self_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
