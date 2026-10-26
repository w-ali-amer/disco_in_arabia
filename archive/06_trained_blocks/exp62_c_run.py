# -*- coding: utf-8 -*-
"""exp62_c_run.py -- doc 48 SS10: does SATURATION generalise to a two-qubit
referent register?

  W-work   RY(theta) on qubit 0 of the register only  -> K=1 (exp45 as-is)
  W-wide   RY(theta) on BOTH qubits of the register   -> K=2 from WIDTH
           (two parallel injection sites), vs S2's K=2 from DEPTH.

Runs on exp62-A's three-way dataset; A's completed S1/S2/S-shuffle runs are the
comparators (doc 48 SS10.4), so no new floor is needed.

The intro sits mid-function in exp45's forward pass, so story_values must be
overridden rather than wrapped. The copy is therefore VERIFIED: with wide=False
this class must reproduce TwoQubitReferentModel.story_values bit-for-bit, and
that is asserted at --smoke before any real run (doc 48 SS10.6 discipline).
"""
import argparse, copy, hashlib, json, math, os, random, sys, time
import torch
torch.set_num_threads(1)
import exp42_compiler as comp
import exp42_sim_adapter as sim
import exp62_bands as B62
import exp61_run as R
from exp45_model import TwoQubitReferentModel
from exp42_run import grad_global_norm

DESIGN_DOC = ("qnlp_private_docs/48_exp62_selection_optimisation_width.md "
              "(1437f47, FROZEN incl. SS10)")
DATA = "stories_exp62_SB3.json"
CKPT_DIR = "exp62c_ckpt"
CELLS = ["M1", "M2", "M3"]
ARMS = ["W-work", "W-wide"]
WIDE = {"W-wide": True, "W-work": False}


class WModel(TwoQubitReferentModel):
    def __init__(self, verbs, seed, wide=False, layers=3):
        super().__init__("A4", verbs, seed, layers=layers)
        self.wide = wide

    def story_values(self, cs, verb_map=None):
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)
            return cache[v]

        psi = sim.init_state(2 * cs.K)
        for w, _noun, angle in cs.intro:
            psi = sim.apply_gate(psi, sim.ry(angle), (2 * w,))
            if self.wide:                      # doc 48 SS10.2: WIDTH, same theta
                psi = sim.apply_gate(psi, sim.ry(angle), (2 * w + 1,))
        for v, wa, wp in cs.events:
            psi = sim.apply_gate(psi, block(v), self._reg(wa) + self._reg(wp))
        vseq, qa, qp = cs.question
        qw = self._reg(qa) + self._reg(qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)
        vals = []
        for eff in (self.q_pos, self.q_neg):
            vals.append(sim.prob_all_zero_on(
                sim.apply_gate(psi, eff.matrix().conj().T, qw), qw))
        return torch.stack(vals)


def angles(cell, seed, meta):
    bear = [n for n in meta["humans"] if n in set(meta["bearers"])]
    non = [n for n in meta["humans"] if n not in set(meta["bearers"])]
    invs = [meta["train_inventory"], meta["valnovel_inventory"],
            meta["heldout_inventory"]]
    return B62.build_angles_3way(cell, seed, bear, non, invs)


def train(arm, cell, seed, items, meta, verbs, max_epochs, save=True):
    ang = angles(cell, seed, meta)
    data = {s: comp.compile_all([i for i in items if i["split"] == s], ang)
            for s in ("train", "val_indist", "val_novel", "test")}
    torch.manual_seed(seed)
    m = WModel(verbs, seed, wide=WIDE[arm])
    opt = torch.optim.Adam(m.parameters(), lr=R.LR)
    rng = random.Random(R.RNG_BASE + seed)
    best = {"v": -1.0, "state": None}
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        order = list(range(len(data["train"])))
        rng.shuffle(order)
        for s in range(0, len(order), R.BATCH):
            opt.zero_grad()
            bl = 0.0
            for i in order[s:s + R.BATCH]:
                cs = data["train"][i]
                bl = bl + torch.nn.functional.cross_entropy(
                    torch.log(m.story_values(cs) + 1e-12).unsqueeze(0),
                    torch.tensor([cs.answer]))
            (bl / min(R.BATCH, len(order) - s)).backward()
            assert math.isfinite(grad_global_norm(m))
            opt.step()
        with torch.no_grad():
            vi = R.evaluate(m, data["val_indist"])
        if vi > best["v"]:
            best = {"v": vi, "state": copy.deepcopy(m.state_dict())}
        if ep % 50 == 0:
            print("[%s %s s%d] ep %3d val=%.3f best=%.3f %.0fs"
                  % (arm, cell, seed, ep, vi, best["v"], time.time() - t0), flush=True)
    m.load_state_dict(best["state"])
    with torch.no_grad():
        te = R.evaluate(m, data["test"])
    r3 = R.fourier_resid(m, data["test"], 3)
    r5 = R.fourier_resid(m, data["test"], 5)
    pt = sha = None
    if save:
        os.makedirs(CKPT_DIR, exist_ok=True)
        pt = os.path.join(CKPT_DIR, "%s_%s_s%d.pt" % (arm, cell, seed))
        torch.save(best["state"], pt)
        sha = hashlib.sha256(open(pt, "rb").read()).hexdigest()
    return {"arm": arm, "cell": cell, "seed": seed, "K": 2 if WIDE[arm] else 1,
            "val_indist": best["v"], "heldout_test_acc": te,
            "fourier_resid_3term": r3, "fourier_resid_5term": r5,
            "ckpt_path": pt, "ckpt_sha256": sha, "wall_sec": time.time() - t0}


def main():
    global CKPT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--ckptdir", default=CKPT_DIR)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    CKPT_DIR = a.ckptdir
    blob = json.load(open(a.data, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("SB3 provenance FAILED")
    verbs = sorted(meta["verb_inventory"].keys())

    if a.smoke:
        # (i) the copied forward pass must equal the PARENT's when wide=False
        ang = angles("M2", 0, meta)
        cs = comp.compile_all([i for i in items if i["split"] == "test"][:20], ang)
        torch.manual_seed(0); mine = WModel(verbs, 0, wide=False)
        torch.manual_seed(0); ref = TwoQubitReferentModel("A4", verbs, 0, layers=3)
        with torch.no_grad():
            d = max(float((mine.story_values(c) - ref.story_values(c)).abs().max())
                    for c in cs)
        print("[C] copied forward vs exp45 parent (wide=False): max diff %.3e" % d)
        assert d == 0.0, "the override drifted from exp45's forward pass"
        # (ii) BOTH directions of the validity check (doc 48 SS10.6)
        torch.manual_seed(0); w1 = WModel(verbs, 0, wide=False)
        torch.manual_seed(0); w2 = WModel(verbs, 0, wide=True)
        f = R.fourier_resid
        print("[C] W-work  3-term %.2e (must be small)   5-term %.2e"
              % (f(w1, cs, 3), f(w1, cs, 5)))
        print("[C] W-wide  3-term %.2e (must be LARGE)   5-term %.2e (must be small)"
              % (f(w2, cs, 3), f(w2, cs, 5)))
        assert f(w1, cs, 3) < 1e-5, "W-work is not K=1"
        assert f(w2, cs, 3) > 1e-3, "W-wide closes at 3 terms -> second RY unwired"
        assert f(w2, cs, 5) < 1e-5, "W-wide is not K=2"
        print("[C] validity checks PASS in both directions")

    maxep = 2 if a.smoke else R.MAX_EPOCHS
    nseeds = 1 if a.smoke else a.seeds
    jobs = [(ar, c, s) for ar in a.arms.split(",") if ar
            for c in a.cells.split(",") if c for s in range(nseeds)]
    mine_j = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    print("[exp62-C w%d/%d] %d runs" % (a.worker, a.nworkers, len(mine_j)), flush=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    for arm, cell, s in mine_j:
        ckj = os.path.join(CKPT_DIR, "%s_%s_s%d.json" % (arm, cell, s))
        if os.path.exists(ckj) and not a.smoke:
            print("[ckpt] done", flush=True); continue
        r = train(arm, cell, s, items, meta, verbs, maxep, save=not a.smoke)
        if not a.smoke:
            json.dump(r, open(ckj, "w", encoding="utf-8"))
        print("[exp62-C] %-7s %-3s s%d  held-out=%.4f  F3=%.1e F5=%.1e  %.0fs"
              % (arm, cell, s, r["heldout_test_acc"], r["fourier_resid_3term"],
                 r["fourier_resid_5term"], r["wall_sec"]), flush=True)
    print("[exp62-C w%d] done" % a.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
