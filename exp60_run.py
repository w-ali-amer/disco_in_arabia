# -*- coding: utf-8 -*-
"""exp60_run.py -- doc 46 Stage B: arms over the frozen (Delta, w) grid.

Registered arms (doc 46 SS10.2):
  S-base     single RY per referent
  S-reup     data re-uploading: RY(kappa*theta) re-applied after the event,
             kappa trainable, INIT 0 so epoch-0 == S-base exactly. One event
             => R=1 extra layer => spectrum {0,+-1,+-2}, a doubling.
  S-shuffle  THE REGISTERED FLOOR. Angle draws bit-for-bit identical to S-base;
             the bearer designation used for ANSWERS is reassigned per PAIR,
             independently of the angle. Removes ONLY the angle<->category
             correlation. (S-hash was rejected as a floor in doc 46 rev-2: its
             sha256 angles are ~U(0,2pi), so it fails for CONDITIONING reasons
             whether or not binding exists -- far too easy a bar.)
  S-flat     Delta = 0, category absent from the input. Must fail.
  S-hash     secondary destroy-everything arm. Recorded, never the bar.

doc 46 SS10.4 screen -> confirm: screen all 8 cells at 3 seeds; screened
positive = held-out >= 0.60; at most the TOP-2 cells advance to a 5-seed
confirmation on a FRESH generator draw (dataset_seed + 1000), with BOTH arms
run at that cell because reuploading_relieves is a DIFFERENCE.
"""
import argparse, copy, hashlib, json, math, os, platform, random, sys, time
import torch

torch.set_num_threads(1)                  # doc 45 SS6.1, verified inert

import exp42_compiler as comp
import exp42_sim_adapter as sim
from exp42_models import QuantumStoryModel
from exp42_run import fresh_results_path, git_note, grad_global_norm

DESIGN_DOC = ("qnlp_private_docs/46_exp60_semantic_binding_and_reuploading.md "
              "(ed19946, FROZEN incl. SS16 amendment)")
DATA = "stories_exp60_SB.json"
CKPT_DIR = "exp60_ckpt"
LR, BATCH = 0.02, 8
MAX_EPOCHS = 150
TRAIN_GATE, VAL_GATE = 0.8, 0.7
RNG_BASE = 55000
PI = math.pi

# doc 46 SS10.1 -- FROZEN GRID, subject to Delta >= 2w + 0.10
GRID = [(d, w) for w in (0.05, 0.15) for d in (0.4, 0.8, 1.6, 2.4)
        if d >= 2 * w + 0.10 - 1e-12]
ARMS = ["S-base", "S-reup", "S-shuffle", "S-flat", "S-hash"]
ANGLE_SEED = 60100
SCREEN_SEEDS, CONFIRM_SEEDS = 3, 5
SCREEN_BAR = 0.60                          # doc 46 SS10.4
SB_BINDS_ABS = 0.65                        # doc 46 SS10.3


def cell_id(d, w):
    return "D%.2f_w%.2f" % (d, w)


def build_angles(arm, d, w, seed, bearers, nouns):
    """doc 46 SS6. bearers ~ U(pi + d/2 +- w), non-bearers ~ U(pi - d/2 +- w).
    Disjointness d >= 2w + 0.10 is asserted, so bayes_ceiling = 1.0."""
    if arm == "S-flat":
        d = 0.0
    if arm != "S-flat":
        assert d >= 2 * w + 0.10 - 1e-12, "band overlap: d=%.3f w=%.3f" % (d, w)
    rng = random.Random(ANGLE_SEED + seed)
    ang = {}
    for n in nouns:
        if arm == "S-hash":
            h = hashlib.sha256(("exp42-frozen-noun:" + n).encode()).digest()
            ang[n] = (int.from_bytes(h[:8], "big") / 2 ** 64) * 2 * PI
        else:
            c = PI + (d / 2 if n in bearers else -d / 2)
            ang[n] = c + rng.uniform(-w, w)
    return ang


def relabel_shuffle(items, meta, seed):
    """doc 46 SS10.2 S-shuffle: identical geometry, bearer designation for the
    ANSWER reassigned PER PAIR independently of the angle. Per-pair (not
    per-item) keeps twins on opposite labels, so V1 and V3 stay intact."""
    rng = random.Random(9_000_000 + seed)
    out = []
    byp = {}
    for it in items:
        byp.setdefault(it["pair_id"], []).append(it)
    for pid, g in byp.items():
        g = sorted(g, key=lambda x: x["id"])
        names = [r["name"] for r in g[0]["referents"]]
        fake_bearer = names[rng.randrange(2)]      # independent of the angle
        for it in g:
            j = copy.deepcopy(it)
            j["answer"] = "yes" if it["events"][0]["agent"] == fake_bearer else "no"
            out.append(j)
    return out


class SBModel(QuantumStoryModel):
    """A3 with doc 46 SS10.2's re-uploading option, off by default."""

    def __init__(self, arm, verbs, seed, reup=False):
        super().__init__(arm, verbs, seed)
        self.reup = reup
        if reup:
            self.kappa = torch.nn.Parameter(torch.zeros(1))   # identity at init

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
            if self.reup:                       # doc 46 SS10.2: R = 1
                for w, noun, angle in cs.intro:
                    psi = sim.apply_gate(psi, sim.ry(self.kappa[0] * angle), (w,))
        vseq, qa, qp = cs.question
        qw = (qa, qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)
        vp = sim.prob_all_zero_on(
            sim.apply_gate(psi, self.q_pos.matrix().conj().T, qw), qw)
        vn = sim.prob_all_zero_on(
            sim.apply_gate(psi, self.q_neg.matrix().conj().T, qw), qw)
        return torch.stack([vp, vn])


def evaluate(m, compiled):
    return sum(int(torch.argmax(m.story_values(c)).item()) == c.answer
               for c in compiled) / len(compiled)


def fourier_rank(m, cs_list, n_terms):
    """doc 46 SS10.5 arm validity: fit P(yes) vs one referent's angle in the
    n_terms Fourier basis. S-base must close at 3 terms; S-reup must NOT (it
    needs 5), else the arm does not implement the intervention."""
    import numpy as np
    ts = np.linspace(0, 2 * PI, 40, endpoint=False)
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


def train(arm, d, w, seed, items, meta, verbs, max_epochs):
    bearers, nouns = set(meta["bearers"]), meta["humans"]
    ang = build_angles(arm, d, w, seed, bearers, nouns)
    use = relabel_shuffle(items, meta, seed) if arm == "S-shuffle" else items
    data = {s: comp.compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}
    torch.manual_seed(seed)
    m = SBModel("A3", verbs, seed, reup=(arm == "S-reup"))
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
                  % (arm, cell_id(d, w), seed, ep, max_epochs, va, best["val"],
                     time.time() - t0), flush=True)
    m.load_state_dict(best["state"])
    with torch.no_grad():
        te = evaluate(m, data["test"])
    os.makedirs(CKPT_DIR, exist_ok=True)
    pt = os.path.join(CKPT_DIR, "%s%s_%s_s%d.pt"
                      % (os.environ.get("EXP60_TAG", ""), arm,
                         cell_id(d, w), seed))
    torch.save(best["state"], pt)                     # doc 45 SS5
    sha = hashlib.sha256(open(pt, "rb").read()).hexdigest()
    return {"arm": arm, "delta": d, "w": w, "cell": cell_id(d, w), "seed": seed,
            "train_acc": best["train"], "val_acc": best["val"],
            "heldout_test_acc": te, "bayes_ceiling": 1.0,
            "kappa": (float(m.kappa[0].detach()) if arm == "S-reup" else None),
            "gate_met": bool(best["train"] >= TRAIN_GATE
                             and best["val"] >= VAL_GATE),
            "ckpt_path": pt, "ckpt_sha256": sha,
            "wall_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="S-base,S-reup")
    ap.add_argument("--seeds", type=int, default=SCREEN_SEEDS)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--cells", default="",
                    help="restrict to these cell ids; S-flat/S-hash "
                         "angles are cell-invariant so one suffices")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    blob = json.load(open(args.data, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("SB provenance FAILED")
    verbs = sorted(meta["verb_inventory"].keys())
    arms = [a for a in args.arms.split(",") if a]
    maxep = 2 if args.smoke else MAX_EPOCHS
    nseeds = 1 if args.smoke else args.seeds
    grid = GRID[:1] if args.smoke else GRID
    if args.cells:
        want = set(args.cells.split(","))
        grid = [(d, w) for (d, w) in grid if cell_id(d, w) in want]

    pairs = [(a, d, w, s) for a in arms for (d, w) in grid
             for s in range(nseeds)]
    mine = [p for i, p in enumerate(pairs) if i % args.nworkers == args.worker]
    print("[exp60 w%d/%d] grid=%d cells arms=%s -> %d runs"
          % (args.worker, args.nworkers, len(grid), arms, len(mine)), flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    for arm, d, w, s in mine:
        key = "%s%s_%s_s%d" % (args.tag, arm, cell_id(d, w), s)
        ckj = os.path.join(CKPT_DIR, key + ".json")
        if os.path.exists(ckj) and not args.smoke:
            print("[ckpt] %s done" % key, flush=True); continue
        r = train(arm, d, w, s, items, meta, verbs, maxep)
        if not args.smoke:
            json.dump(r, open(ckj, "w", encoding="utf-8"))
        print("[exp60] %-9s %s s%d  train=%.3f val=%.3f HELD-OUT=%.4f%s"
              % (arm, cell_id(d, w), s, r["train_acc"], r["val_acc"],
                 r["heldout_test_acc"],
                 ("  kappa=%+.4f" % r["kappa"]) if r["kappa"] is not None else ""),
              flush=True)
    print("[exp60 w%d] done" % args.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
