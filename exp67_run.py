# -*- coding: utf-8 -*-
"""exp67_run.py -- doc 53: does READOUT ARITY convert to trained capacity?

R2 = binary (baseline) | R4 = argmax over 4 outcomes | R8 = argmax over 8
S-shuffle = registered floor. Grouping is FIXED alternating (SS3.1); training
uses a logsumexp surrogate (SS3.2) but ALL reported accuracies use hard argmax.

Validity, both directions, asserted at smoke (SS5):
  (1) each outcome probability still closes in the 3-term basis  -> K unchanged
  (2) hard argmax selects >=3 distinct outcomes                  -> arity USED
An arm failing (2) has degenerated to binary and is inadmissible.
"""
import argparse, copy, hashlib, json, math, os, random, sys, time
import torch
torch.set_num_threads(1)
import exp42_compiler as comp
import exp42_sim_adapter as sim
import exp62_bands as B
import exp61_run as R
from exp42_models import QuantumStoryModel
from exp60_run import relabel_shuffle
from exp42_run import grad_global_norm

DESIGN_DOC = "qnlp_private_docs/53_exp67_readout_arity.md (126f72b, FROZEN)"
DATA = "stories_exp60_SB.json"
DATA_SHA = "ae94ab9c31d62778e1c916aa44566a0ec3c8f74b09f4bf3839bab5f331c87b8c"
CKPT_DIR = "exp67_ckpt"
CELLS = ["M2", "M3"]
ARMS = {"R2": 2, "R4": 4, "R8": 8, "S-shuffle": 2}
TEMP, SCALE = 4.0, 3.0


class ArityModel(QuantumStoryModel):
    """A3 with M readout effects, argmax decision, alternating outcome->class."""

    def __init__(self, arm, verbs, seed, M=2):
        super().__init__(arm, verbs, seed)
        self.M = M
        if M > 2:
            g = torch.Generator().manual_seed(seed + 6700)
            self.effects = torch.nn.ModuleList(
                [sim.Circuit4(2, 3, generator=g) for _ in range(M)])
        self.grp = [i % 2 for i in range(M)]          # SS3.1 FIXED

    def outcome_probs(self, cs, verb_map=None):
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)
            return cache[v]

        psi = sim.init_state(cs.K)
        for w, _n, angle in cs.intro:
            psi = sim.apply_gate(psi, sim.ry(angle), (w,))
        for v, wa, wp in cs.events:
            psi = sim.apply_gate(psi, block(v), (wa, wp))
        vseq, qa, qp = cs.question
        qw = (qa, qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)
        effs = ([self.q_pos, self.q_neg] if self.M == 2 else list(self.effects))
        return torch.stack([sim.prob_all_zero_on(
            sim.apply_gate(psi, e.matrix().conj().T, qw), qw) for e in effs])

    def story_values(self, cs, verb_map=None):
        """Class scores. M=2 reproduces the baseline exactly."""
        p = self.outcome_probs(cs, verb_map)
        if self.M == 2:
            return p
        g = torch.tensor(self.grp)
        return torch.stack([torch.logsumexp(p[g == c] * TEMP, 0) / TEMP
                            for c in (0, 1)])

    def hard_predict(self, cs):
        p = self.outcome_probs(cs)
        if self.M == 2:
            return int(torch.argmax(p).item()), int(torch.argmax(p).item())
        o = int(torch.argmax(p).item())
        return self.grp[o], o


def outcome_fourier_resid(m, cs_list, n_terms=3):
    """doc 53 SS5 direction 1, CORRECTED: fit each OUTCOME PROBABILITY in the
    3-term basis. The earlier version fitted story_values -- the CLASS SCORE --
    which for M>2 is a logsumexp and is non-3-term BY DESIGN. That is the arity
    mechanism, not an encoding change; testing it as direction 1 was wrong."""
    import numpy as np
    ts = np.linspace(0, 2 * math.pi, 40, endpoint=False)
    worst = 0.0
    for cs in cs_list[:4]:
        P = []
        for t in ts:
            c2 = copy.deepcopy(cs)
            c2.intro[0] = (c2.intro[0][0], c2.intro[0][1], float(t))
            with torch.no_grad():
                P.append(m.outcome_probs(c2).numpy())
        P = np.array(P)                                  # (n_t, M)
        A = np.stack([np.ones_like(ts), np.cos(ts), np.sin(ts)], 1)
        for o in range(P.shape[1]):
            coef, *_ = np.linalg.lstsq(A, P[:, o], rcond=None)
            worst = max(worst, float(np.abs(A @ coef - P[:, o]).max()))
    return worst


def classscore_fourier_resid(m, cs_list, n_terms=3):
    """doc 53 SS5 direction 1b: the CLASS SCORE must NOT close at 3 terms for
    M>2 -- that non-closure IS the nonlinearity the arity lever supplies."""
    return R.fourier_resid(m, cs_list, n_terms)


def evaluate_hard(m, data):
    ok = 0; outs = []
    with torch.no_grad():
        for cs in data:
            c, o = m.hard_predict(cs); outs.append(o); ok += int(c == cs.answer)
    return ok / len(data), outs


def train(arm, cell, seed, items, meta, verbs, max_epochs, save=True):
    M = ARMS[arm]
    bear = [n for n in meta["humans"] if n in set(meta["bearers"])]
    non = [n for n in meta["humans"] if n not in set(meta["bearers"])]
    ang = B.build_angles_3way(cell, seed, bear, non,
                              [meta["train_inventory"], meta["heldout_inventory"]])
    use = relabel_shuffle(items, meta, seed) if arm == "S-shuffle" else items
    data = {s: comp.compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}
    torch.manual_seed(seed)
    m = ArityModel("A3", verbs, seed, M=M)
    opt = torch.optim.Adam(m.parameters(), lr=R.LR)
    rng = random.Random(R.RNG_BASE + seed)
    best = {"v": -1.0, "state": None}
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        order = list(range(len(data["train"]))); rng.shuffle(order)
        for s in range(0, len(order), R.BATCH):
            opt.zero_grad(); bl = 0.0
            for i in order[s:s + R.BATCH]:
                cs = data["train"][i]
                bl = bl + torch.nn.functional.cross_entropy(
                    (torch.log(m.story_values(cs) + 1e-12) * SCALE).unsqueeze(0),
                    torch.tensor([cs.answer]))
            (bl / min(R.BATCH, len(order) - s)).backward()
            assert math.isfinite(grad_global_norm(m)); opt.step()
        v, _ = evaluate_hard(m, data["val"])
        if v > best["v"]:
            best = {"v": v, "state": copy.deepcopy(m.state_dict())}
        if ep % 50 == 0:
            print("[%s %s s%d] ep %3d val=%.3f best=%.3f %.0fs"
                  % (arm, cell, seed, ep, v, best["v"], time.time() - t0), flush=True)
    m.load_state_dict(best["state"])
    te, outs = evaluate_hard(m, data["test"])
    distinct = len(set(outs))
    pt = sha = None
    if save:
        os.makedirs(CKPT_DIR, exist_ok=True)
        pt = os.path.join(CKPT_DIR, "%s_%s_s%d.pt" % (arm, cell, seed))
        torch.save(best["state"], pt)
        sha = hashlib.sha256(open(pt, "rb").read()).hexdigest()
    return {"arm": arm, "cell": cell, "seed": seed, "M": M,
            "val_acc": best["v"], "heldout_test_acc": te,
            "distinct_outcomes_used": distinct,
            "arity_used": bool(M == 2 or distinct >= 3),
            "ckpt_path": pt, "ckpt_sha256": sha, "wall_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    blob = json.load(open(DATA, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != DATA_SHA:
        raise RuntimeError("SB provenance FAILED")
    verbs = sorted(meta["verb_inventory"].keys())

    if a.smoke:
        bear = [n for n in meta["humans"] if n in set(meta["bearers"])]
        non = [n for n in meta["humans"] if n not in set(meta["bearers"])]
        ang = B.build_angles_3way("M2", 0, bear, non,
                                  [meta["train_inventory"], meta["heldout_inventory"]])
        cs = comp.compile_all([i for i in items if i["split"] == "test"][:6], ang)
        for arm in ("R2", "R4", "R8"):
            m = ArityModel("A3", verbs, 0, M=ARMS[arm])
            ro = outcome_fourier_resid(m, cs)
            rc = classscore_fourier_resid(m, cs)
            _, outs = evaluate_hard(m, comp.compile_all(
                [i for i in items if i["split"] == "test"], ang))
            d = len(set(outs))
            print("[%-3s] outcomes 3-term %.2e (must be SMALL: K unchanged) | "
                  "class-score 3-term %.2e (must be LARGE for M>2) | distinct %d"
                  % (arm, ro, rc, d), flush=True)
            assert ro < 1e-5, "%s: an OUTCOME PROBABILITY is not 3-term -- K changed" % arm
            if ARMS[arm] > 2:
                assert rc > 1e-3, "%s: class score IS 3-term -- nonlinearity absent" % arm
                assert d >= 3, "%s: only %d outcomes used -- collapsed to binary" % (arm, d)
        print("[smoke] BOTH validity directions PASS for all arms", flush=True)

    maxep = 2 if a.smoke else R.MAX_EPOCHS
    ns = 1 if a.smoke else 5
    jobs = [(ar, c, s) for ar in ARMS for c in CELLS for s in range(ns)]
    mine = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    print("[exp67 w%d/%d] %d runs" % (a.worker, a.nworkers, len(mine)), flush=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    for arm, cell, s in mine:
        ckj = os.path.join(CKPT_DIR, "%s_%s_s%d.json" % (arm, cell, s))
        if os.path.exists(ckj) and not a.smoke:
            continue
        r = train(arm, cell, s, items, meta, verbs, maxep, save=not a.smoke)
        if not a.smoke:
            json.dump(r, open(ckj, "w", encoding="utf-8"))
        print("[exp67] %-9s %-3s s%d  held-out=%.4f  outcomes-used=%d  arity_used=%s"
              % (arm, cell, s, r["heldout_test_acc"], r["distinct_outcomes_used"],
                 r["arity_used"]), flush=True)
    print("[exp67 w%d] done" % a.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
