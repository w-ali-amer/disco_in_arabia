# -*- coding: utf-8 -*-
"""exp61_run.py -- doc 47 Stage B: capacity vs spectrum width.

Arms (doc 47 SS7):
  S1        K=1, single RY per referent (exp60's S-base, unchanged)
  S2        K=2, ONE re-upload after the event, kappa PINNED to 1
  S-shuffle THE registered floor -- identical geometry, bearer designation for
            ANSWERS reassigned per PAIR independently of the angle

kappa is PINNED, not trained (doc 46 SS17.3): exp60's arm was inadmissible
because kappa never moved (baseline at ceiling) and a trainable kappa gives
NON-integer frequencies {0,+-1,+-k,+-(1+-k)} that an integer-harmonic validity
basis can never close. Pinning fixes both, and at m>=2 the K=1 ceiling is 0.75
so gradient pressure now exists.

Geometry comes from exp61_bands -- the SAME module the ceiling uses (SS5.1).
"""
import argparse, copy, hashlib, json, math, os, random, sys, time
import torch

torch.set_num_threads(1)

import exp42_compiler as comp
import exp42_sim_adapter as sim
import exp61_bands as BANDS
from exp42_models import QuantumStoryModel
from exp42_run import grad_global_norm
from exp60_run import relabel_shuffle, evaluate

DESIGN_DOC = "qnlp_private_docs/47_exp61_category_capacity.md (0236976, FROZEN)"
DATA = "stories_exp60_SB.json"
CKPT_DIR = "exp61_ckpt"
LR, BATCH, MAX_EPOCHS = 0.02, 8, 150
TRAIN_GATE, VAL_GATE = 0.8, 0.7
RNG_BASE = 55000
ARMS = ["S1", "S2", "S-shuffle"]
CELLS = ["M1", "M2", "M2w", "M3"]
REUP = {"S2": 1}                                    # extra encoding layers, kappa=1
ARM_K = {"S1": 1, "S2": 2, "S-shuffle": 1}


class CapModel(QuantumStoryModel):
    """A3 with R pinned re-uploading layers (kappa = 1 exactly)."""

    def __init__(self, arm, verbs, seed, reup=0):
        super().__init__(arm, verbs, seed)
        self.reup = reup                             # int, NOT a Parameter

    def story_values(self, cs, verb_map=None):
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)
            return cache[v]

        psi = sim.init_state(cs.K)
        for w, noun, angle in cs.intro:
            psi = sim.apply_gate(psi, sim.ry(angle), (w,))
        for v, wa, wp in cs.events:
            psi = sim.apply_gate(psi, block(v), (wa, wp))
            for _ in range(self.reup):               # kappa pinned to 1
                for w, noun, angle in cs.intro:
                    psi = sim.apply_gate(psi, sim.ry(angle), (w,))
        vseq, qa, qp = cs.question
        qw = (qa, qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)
        vp = sim.prob_all_zero_on(
            sim.apply_gate(psi, self.q_pos.matrix().conj().T, qw), qw)
        vn = sim.prob_all_zero_on(
            sim.apply_gate(psi, self.q_neg.matrix().conj().T, qw), qw)
        return torch.stack([vp, vn])


def fourier_resid(m, cs_list, n_terms):
    """doc 47 SS9 validity: fit P(yes) vs one referent's angle in the n_terms
    basis. S1 must close at 3; S2 must close at 5 and NOT at 3."""
    import numpy as np
    ts = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    worst = 0.0
    for cs in cs_list[:4]:
        ys = []
        for t in ts:
            c2 = copy.deepcopy(cs)
            c2.intro[0] = (c2.intro[0][0], c2.intro[0][1], float(t))
            with torch.no_grad():
                ys.append(float(m.story_values(c2)[0]))
        cols = [np.ones_like(ts)]
        for k in range(1, n_terms // 2 + 1):
            cols += [np.cos(k * ts), np.sin(k * ts)]
        A = np.stack(cols, 1)
        coef, *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
        worst = max(worst, float(np.abs(A @ coef - np.array(ys)).max()))
    return worst


def angles_for(arm, cell, seed, meta):
    bearers = [n for n in meta["humans"] if n in set(meta["bearers"])]
    nonb = [n for n in meta["humans"] if n not in set(meta["bearers"])]
    return BANDS.build_angles(cell, seed, bearers, nonb,
                              meta["train_inventory"], meta["heldout_inventory"],
                              flat=(arm == "S-flat"))


def train(arm, cell, seed, items, meta, verbs, max_epochs, save=True):
    ang = angles_for(arm, cell, seed, meta)
    use = relabel_shuffle(items, meta, seed) if arm == "S-shuffle" else items
    data = {s: comp.compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}
    torch.manual_seed(seed)
    m = CapModel("A3", verbs, seed, reup=REUP.get(arm, 0))
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    rng = random.Random(RNG_BASE + seed)
    best = {"val": -1.0, "train": -1.0, "state": None}
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        order = list(range(len(data["train"])))
        rng.shuffle(order)
        for s in range(0, len(order), BATCH):
            opt.zero_grad()
            bl = 0.0
            for i in order[s:s + BATCH]:
                cs = data["train"][i]
                bl = bl + torch.nn.functional.cross_entropy(
                    torch.log(m.story_values(cs) + 1e-12).unsqueeze(0),
                    torch.tensor([cs.answer]))
            (bl / min(BATCH, len(order) - s)).backward()
            assert math.isfinite(grad_global_norm(m))
            opt.step()
        with torch.no_grad():
            va = evaluate(m, data["val"])
        if va > best["val"]:
            with torch.no_grad():
                best = {"val": va, "train": evaluate(m, data["train"]),
                        "state": copy.deepcopy(m.state_dict())}
        if ep % 50 == 0 or ep == max_epochs:
            print("[%s %s s%d] ep %3d/%d val=%.3f best=%.3f %.0fs"
                  % (arm, cell, seed, ep, max_epochs, va, best["val"],
                     time.time() - t0), flush=True)
    m.load_state_dict(best["state"])
    with torch.no_grad():
        te = evaluate(m, data["test"])
    r3 = fourier_resid(m, data["test"], 3)
    r5 = fourier_resid(m, data["test"], 5)
    pt = sha = None
    if save:
        os.makedirs(CKPT_DIR, exist_ok=True)
        pt = os.path.join(CKPT_DIR, "%s%s_%s_s%d.pt"
                          % (os.environ.get("EXP61_TAG", ""), arm, cell, seed))
        torch.save(best["state"], pt)
        sha = hashlib.sha256(open(pt, "rb").read()).hexdigest()
    return {"arm": arm, "cell": cell, "seed": seed, "K": ARM_K[arm],
            "ceiling": BANDS.FROZEN_CEILINGS.get((cell, ARM_K[arm])),
            "train_acc": best["train"], "val_acc": best["val"],
            "heldout_test_acc": te,
            "fourier_resid_3term": r3, "fourier_resid_5term": r5,
            "gate_met": bool(best["train"] >= TRAIN_GATE and best["val"] >= VAL_GATE),
            "ckpt_path": pt, "ckpt_sha256": sha, "wall_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="S1,S2,S-shuffle")
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    assert BANDS.verify_frozen_table(), "ceiling table drift"
    blob = json.load(open(a.data, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("SB provenance FAILED")
    verbs = sorted(meta["verb_inventory"].keys())
    arms = [x for x in a.arms.split(",") if x]
    cells = [c for c in a.cells.split(",") if c]
    maxep = 2 if a.smoke else MAX_EPOCHS
    nseeds = 1 if a.smoke else a.seeds

    pairs = [(ar, c, s) for ar in arms for c in cells for s in range(nseeds)]
    mine = [p for i, p in enumerate(pairs) if i % a.nworkers == a.worker]
    print("[exp61 w%d/%d] %d runs | ceilings %s"
          % (a.worker, a.nworkers, len(mine),
             {c: BANDS.FROZEN_CEILINGS[(c, 1)] for c in cells}), flush=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    for arm, cell, s in mine:
        key = "%s%s_%s_s%d" % (a.tag, arm, cell, s)
        ckj = os.path.join(CKPT_DIR, key + ".json")
        if os.path.exists(ckj) and not a.smoke:
            print("[ckpt] %s done" % key, flush=True); continue
        r = train(arm, cell, s, items, meta, verbs, maxep, save=not a.smoke)
        if not a.smoke:
            json.dump(r, open(ckj, "w", encoding="utf-8"))
        print("[exp61] %-9s %-4s s%d  HELD-OUT=%.4f  ceiling=%s  F3=%.1e F5=%.1e"
              % (arm, cell, s, r["heldout_test_acc"], r["ceiling"],
                 r["fourier_resid_3term"], r["fourier_resid_5term"]), flush=True)
    print("[exp61 w%d] done" % a.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
