# -*- coding: utf-8 -*-
"""exp63_run.py -- doc 49 SS6: is the K=2 shortfall SAMPLE COMPLEXITY?

The reachable space for two referents has (2K+1)^2 coefficients: 9 at K=1, 25 at
K=2. Nearly triple the quantities to estimate, while training data stayed fixed.
Sample complexity => S2's gap to its ceiling shrinks with data.
Optimisation/frequency-decay => it does not.

doc 49 SS6.1: exp60's frozen SB dataset, training subsampled BY PAIRS (twins must
stay together), 3 levels, arms S1/S2, cells M2/M3, 5 seeds, S-shuffle floor at
full data. Band assignment is STRATIFIED (exp62_bands) -- the corrected method;
all levels share it, so the curve is internally consistent.
"""
import argparse, copy, hashlib, json, math, os, random, sys, time
import torch
torch.set_num_threads(1)
import exp42_compiler as comp
import exp62_bands as B62
import exp61_run as R
from exp60_run import relabel_shuffle
from exp42_run import grad_global_norm

DESIGN_DOC = "qnlp_private_docs/49_exp63_bringing_it_home.md (771df00, FROZEN)"
DATA = "stories_exp60_SB.json"
DATA_SHA = "ae94ab9c31d62778e1c916aa44566a0ec3c8f74b09f4bf3839bab5f331c87b8c"
CKPT_DIR = "exp63_ckpt"
LEVELS = [120, 240, 470]                 # doc 49 SS6.1, frozen
SUBSAMPLE_SEED = 63500                   # registered
CELLS = ["M2", "M3"]
ARMS = ["S1", "S2"]


def angles(cell, seed, meta):
    bear = [n for n in meta["humans"] if n in set(meta["bearers"])]
    non = [n for n in meta["humans"] if n not in set(meta["bearers"])]
    return B62.build_angles_3way(cell, seed, bear, non,
                                 [meta["train_inventory"], meta["heldout_inventory"]])


def subsample_by_pairs(items, n_items, seed):
    """doc 49 SS6.1: subsample by PAIR. Twins must stay together or the
    construction's structure breaks inside the training set."""
    tr = [i for i in items if i["split"] == "train"]
    byp = {}
    for i in tr:
        byp.setdefault(i["pair_id"], []).append(i)
    pids = sorted(byp)
    random.Random(seed).shuffle(pids)
    keep, out = set(), []
    for p in pids:
        if len(out) + len(byp[p]) > n_items:
            continue
        keep.add(p); out.extend(byp[p])
        if len(out) >= n_items:
            break
    rest = [i for i in items if i["split"] != "train"]
    return out + rest, len(keep)


def train(arm, cell, level, seed, items, meta, verbs, max_epochs, save=True):
    ang = angles(cell, seed, meta)
    use = relabel_shuffle(items, meta, seed) if arm == "S-shuffle" else items
    use, npairs = subsample_by_pairs(use, level, SUBSAMPLE_SEED + seed)
    data = {s: comp.compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}
    torch.manual_seed(seed)
    m = R.CapModel("A3", verbs, seed, reup=R.REUP.get(arm, 0))
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
            v = R.evaluate(m, data["val"])
        if v > best["v"]:
            best = {"v": v, "state": copy.deepcopy(m.state_dict())}
        if ep % 50 == 0:
            print("[%s %s n=%d s%d] ep %3d val=%.3f best=%.3f %.0fs"
                  % (arm, cell, level, seed, ep, v, best["v"], time.time() - t0), flush=True)
    m.load_state_dict(best["state"])
    with torch.no_grad():
        te = R.evaluate(m, data["test"])
    pt = sha = None
    if save:
        os.makedirs(CKPT_DIR, exist_ok=True)
        pt = os.path.join(CKPT_DIR, "%s_%s_n%d_s%d.pt" % (arm, cell, level, seed))
        torch.save(best["state"], pt)
        sha = hashlib.sha256(open(pt, "rb").read()).hexdigest()
    return {"arm": arm, "cell": cell, "level": level, "n_train_pairs": npairs,
            "seed": seed, "K": 2 if arm == "S2" else 1,
            "val_acc": best["v"], "heldout_test_acc": te,
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
        raise RuntimeError("exp60 SB provenance FAILED")
    verbs = sorted(meta["verb_inventory"].keys())
    maxep = 2 if a.smoke else R.MAX_EPOCHS
    ns = 1 if a.smoke else 5
    jobs = [(ar, c, L, s) for ar in ARMS for c in CELLS for L in LEVELS for s in range(ns)]
    jobs += [("S-shuffle", c, LEVELS[-1], s) for c in CELLS for s in range(ns)]
    if a.smoke:
        jobs = jobs[:2]
    mine = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    print("[exp63 w%d/%d] %d runs | levels %s" % (a.worker, a.nworkers, len(mine), LEVELS), flush=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    for arm, cell, L, s in mine:
        ckj = os.path.join(CKPT_DIR, "%s_%s_n%d_s%d.json" % (arm, cell, L, s))
        if os.path.exists(ckj) and not a.smoke:
            continue
        r = train(arm, cell, L, s, items, meta, verbs, maxep, save=not a.smoke)
        if not a.smoke:
            json.dump(r, open(ckj, "w", encoding="utf-8"))
        print("[exp63] %-9s %-3s n=%-4d s%d  held-out=%.4f (%d pairs)"
              % (arm, cell, L, s, r["heldout_test_acc"], r["n_train_pairs"]), flush=True)
    print("[exp63 w%d] done" % a.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
