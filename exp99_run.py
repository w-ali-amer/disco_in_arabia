# -*- coding: utf-8 -*-
"""exp99_run.py -- doc 94 (REGISTERED 082dc78): one qubit per noun as a random access code for Arabic morphology.

Features (segmented from the SURFACE, nominative): gender (ة/ت), number (dual ان/تان), definiteness (PRESENCE of ال).
Balanced set: every stem x word (2^n) x feature x probe value; label = (feature value == probe value).
Encoder sees ONLY the segmented morphemes (never stem, never query). One qubit (or rebit) per referent, no ancilla.
Loss = -mean single-shot success (linear QRAC objective); metric = exact mean single-shot P(correct) (no argmax).
Arms: Q-COMP-n / R-COMP-n (a box per feature value, fixed order, base point x expm; U(2) / O(2)),
      Q-LOOKUP-3 / R-LOOKUP-3 (free state per word), B-1BIT-n (trained straight-through bit), FLOAT-3 (a real per noun).
Usage: python exp99_run.py --all   -> exp99_ckpt/<arm>_s<seed>.json (Rule 3; os.makedirs)
"""
import argparse, itertools, json, os, time
import torch
import torch.nn as nn
import exp94_common as C94

CK = "exp99_ckpt"
SEEDS = range(32)
STEMS = C94.STEMS
PROBES = {0: ("ذَهَبَ", "ذَهَبَتْ"), 1: ("واحدٌ", "اثنانِ"), 2: ("(صلة بلا موصول)", "الذي")}   # partly constructed (doc 94 §2)
HAR = "ًٌٍَُِّْٰ"


def surface(stem, g, nu, d):
    base = stem[2:]                                                     # stems carry ال
    end = {(0, 0): "", (1, 0): "ة", (0, 1): "ان", (1, 1): "تان"}[(g, nu)]
    case = "ُ" if nu == 0 else "ِ"
    tanwin = "ٌ" if (nu == 0 and not d) else case
    return C94.nfc(("ال" if d else "") + base + end + (tanwin if not d else case))


def segment(tok):
    """SURFACE -> (gender, number, definiteness) from morphemes only."""
    s = "".join(c for c in C94.nfc(tok) if c not in HAR)
    d = int(s.startswith("ال"))
    body = s[2:] if d else s
    if body.endswith("تان"):
        return 1, 1, d
    if body.endswith("ان"):
        return 0, 1, d
    if body.endswith("ة"):
        return 1, 0, d
    return 0, 0, d


def dataset(n):
    items = []
    for stem in STEMS:
        for w in itertools.product((0, 1), repeat=n):
            full = list(w) + [1] * (3 - n)                                  # n=2: definite only
            tok = surface(stem, full[0], full[1], full[2])
            for f in range(n):
                for pv in (0, 1):
                    items.append({"noun": tok, "feature": f, "probe": PROBES[f][pv], "probe_value": pv,
                                  "label": int(w[f] == pv)})
    return items


def encode(items, n):
    words, q, y = [], [], []
    for it in items:
        seg = segment(it["noun"])[:n]
        words.append(int("".join(map(str, seg)), 2))
        q.append(it["feature"] * 2 + it["probe_value"]); y.append(it["label"])
    return torch.tensor(words), torch.tensor(q), torch.tensor(y, dtype=torch.float32)


def haar_u2(g):
    Z = torch.complex(torch.randn(2, 2, generator=g), torch.randn(2, 2, generator=g))
    Q, R = torch.linalg.qr(Z); d = torch.diagonal(R); return Q * (d / d.abs())[None, :]


def haar_o2(g):
    Q, R = torch.linalg.qr(torch.randn(2, 2, generator=g)); return Q * torch.sign(torch.diagonal(R))[None, :]


def bloch(psi):
    a, b = psi[..., 0], psi[..., 1]
    return torch.stack([2 * (a.conj() * b).real, 2 * (a.conj() * b).imag, a.abs() ** 2 - b.abs() ** 2], -1)


class QCode(nn.Module):
    """compositional (box per feature value) or lookup (free state per word); qubit or rebit; learned query axes."""
    def __init__(self, n, rebit, lookup, seed):
        super().__init__()
        g = torch.Generator().manual_seed(99700 + seed)
        self.n, self.rebit, self.lookup = n, rebit, lookup
        nb = (2 ** n) if lookup else (n * 2)
        base = [haar_o2(g).to(torch.complex64) if rebit else haar_u2(g) for _ in range(nb)]
        self.register_buffer("U0", torch.stack(base))
        self.G = nn.Parameter(torch.zeros(nb, 2, 2)) if rebit else None
        if not rebit:
            self.Gr = nn.Parameter(torch.zeros(nb, 2, 2)); self.Gi = nn.Parameter(torch.zeros(nb, 2, 2))
        self.axes = nn.Parameter(torch.randn(2 * n, 3, generator=g))

    def boxes(self):
        if self.rebit:
            E = torch.matrix_exp(self.G - self.G.transpose(-1, -2)).to(torch.complex64)
        else:
            Gc = torch.complex(self.Gr, self.Gi); E = torch.matrix_exp(Gc - Gc.conj().transpose(-1, -2))
        return self.U0 @ E

    def states(self):
        B = self.boxes()
        W = torch.tensor(list(itertools.product((0, 1), repeat=self.n)))
        psi = torch.zeros(len(W), 2, dtype=torch.complex64); psi[:, 0] = 1
        if self.lookup:
            return torch.einsum("wij,wj->wi", B, psi)
        for i in range(self.n):
            psi = torch.einsum("wij,wj->wi", B[i * 2 + W[:, i]], psi)
        return psi

    def p_yes(self, words, q):
        v = bloch(self.states())[words]
        ax = self.axes
        if self.rebit:
            ax = torch.stack([ax[:, 0], torch.zeros_like(ax[:, 0]), ax[:, 2]], 1)
        ax = ax / ax.norm(dim=1, keepdim=True).clamp(min=1e-9)
        return 0.5 * (1 + (v * ax[q]).sum(-1))


class Bit1(nn.Module):
    """trained 1-bit register: encoder word -> hard bit (straight-through); decoder (bit, query) -> P(yes)."""
    def __init__(self, n, seed):
        super().__init__(); torch.manual_seed(seed)
        self.enc = nn.Parameter(torch.randn(2 ** n)); self.dec = nn.Parameter(torch.randn(2, 2 * n))

    def p_yes(self, words, q):
        s = torch.sigmoid(self.enc[words]); hard = (s > 0.5).float()
        bit = hard + s - s.detach()
        return bit * torch.sigmoid(self.dec[1, q]) + (1 - bit) * torch.sigmoid(self.dec[0, q])


class Float1(nn.Module):
    def __init__(self, n, seed):
        super().__init__(); torch.manual_seed(seed)
        self.enc = nn.Parameter(torch.randn(2 ** n)); self.dec = nn.Sequential(nn.Linear(1 + 2 * n, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def p_yes(self, words, q):
        x = torch.cat([self.enc[words][:, None], nn.functional.one_hot(q, self.dec[0].in_features - 1).float()], 1)
        return torch.sigmoid(self.dec(x)).squeeze(-1)


ARMS = {"Q-COMP-2": (2, lambda s: QCode(2, False, False, s)), "R-COMP-2": (2, lambda s: QCode(2, True, False, s)),
        "Q-COMP-3": (3, lambda s: QCode(3, False, False, s)), "R-COMP-3": (3, lambda s: QCode(3, True, False, s)),
        "Q-LOOKUP-3": (3, lambda s: QCode(3, False, True, s)), "R-LOOKUP-3": (3, lambda s: QCode(3, True, True, s)),
        "B-1BIT-2": (2, lambda s: Bit1(2, s)), "B-1BIT-3": (3, lambda s: Bit1(3, s)), "FLOAT-3": (3, lambda s: Float1(3, s))}


def run_unit(arm, seed):
    os.makedirs(CK, exist_ok=True)
    path = os.path.join(CK, "%s_s%d.json" % (arm, seed))
    if os.path.exists(path):
        return
    n, mk = ARMS[arm]
    items = dataset(n); words, q, y = encode(items, n)
    model = mk(seed); opt = torch.optim.Adam(model.parameters(), lr=0.05 if arm != "FLOAT-3" else 0.01)
    t0 = time.time()
    steps = 400 if arm != "FLOAT-3" else 4000          # doc 94 §7 ruling: FLOAT is a descriptive resource demo
    for _ in range(steps):
        p = model.p_yes(words, q)
        if arm == "FLOAT-3":          # doc 94 §7 ruling: descriptive capacity demo trained with BCE; still SCORED single-shot
            loss = nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), y)
        else:
            loss = -(y * p + (1 - y) * (1 - p)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = model.p_yes(words, q); succ = y * p + (1 - y) * (1 - p)
        per_feature = {f: float(succ[q // 2 == f].mean()) for f in range(n)}
        worst = float(min(succ[(words == w) & (q // 2 == f)].mean() for w in range(2 ** n) for f in range(n)))
        argmax = float(((p > 0.5).float() == y).float().mean())
        extra = {}
        if isinstance(model, QCode):
            extra["bloch"] = bloch(model.states()).tolist()
            extra["imag_max"] = float(model.boxes().imag.abs().max()) if model.rebit else None
    out = {"doc": "94 (082dc78)", "arm": arm, "seed": seed, "n": n, "single_shot": float(succ.mean()),
           "per_feature": per_feature, "worst_case": worst, "argmax_acc_descriptive": argmax, "n_items": len(items),
           "final_loss": float(loss.detach()), "wall_sec": time.time() - t0, "extra": extra, "torch": torch.__version__,
           "provenance": C94.provenance()}
    C94.write_new_json(path, out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--all", action="store_true"); a = ap.parse_args()
    torch.set_num_threads(1)
    for arm in ARMS:
        for s in SEEDS:
            run_unit(arm, s)
        print(arm, "done", flush=True)


if __name__ == "__main__":
    main()
