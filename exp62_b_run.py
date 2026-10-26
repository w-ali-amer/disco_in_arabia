# -*- coding: utf-8 -*-
"""exp62_b_run.py -- doc 48 SS9.1 B1: was exp61's M3 collapse a BAND-ASSIGNMENT
artifact?

exp61's frozen dataset, M3, arms S1 and S-shuffle, 5 seeds, protocol identical
to exp61 -- changing EXACTLY ONE THING: stratified band assignment
(exp62_bands.assign_bands_3way) instead of exp61's global round-robin, which
gave M3 a band holding a SINGLE training noun (train [1,3,3,2,3,2]).

exp61_bands.assign_bands and exp61_run are NOT modified: exp61's results depend
on them bit-for-bit. CapModel/evaluate are IMPORTED from exp61_run (doc-42
discipline: no re-copy).
"""
import argparse, copy, hashlib, json, math, os, random, sys, time
import torch
torch.set_num_threads(1)
import exp42_compiler as comp
import exp62_bands as B62
import exp61_run as R
from exp60_run import relabel_shuffle
from exp42_run import grad_global_norm

DESIGN_DOC = ("qnlp_private_docs/48_exp62_selection_optimisation_width.md "
              "(13b4795, FROZEN incl. SS9)")
DATA = "stories_exp60_SB.json"
DATA_SHA = "ae94ab9c31d62778e1c916aa44566a0ec3c8f74b09f4bf3839bab5f331c87b8c"
CKPT_DIR = "exp62b_ckpt"
CELL = "M3"
EXP61_M3_S1 = 0.4807          # doc 48 SS9.1 comparison point


def angles(arm, seed, meta):
    bear = [n for n in meta["humans"] if n in set(meta["bearers"])]
    non = [n for n in meta["humans"] if n not in set(meta["bearers"])]
    invs = [meta["train_inventory"], meta["heldout_inventory"]]
    return B62.build_angles_3way(CELL, seed, bear, non, invs,
                                 flat=(arm == "S-flat"))


def train(arm, seed, items, meta, verbs, max_epochs, save=True):
    ang = angles(arm, seed, meta)
    use = relabel_shuffle(items, meta, seed) if arm == "S-shuffle" else items
    data = {s: comp.compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}
    torch.manual_seed(seed)
    m = R.CapModel("A3", verbs, seed, reup=R.REUP.get(arm, 0))
    opt = torch.optim.Adam(m.parameters(), lr=R.LR)
    rng = random.Random(R.RNG_BASE + seed)
    best = {"val": -1.0, "train": -1.0, "state": None}
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
            va = R.evaluate(m, data["val"])
        if va > best["val"]:
            with torch.no_grad():
                best = {"val": va, "train": R.evaluate(m, data["train"]),
                        "state": copy.deepcopy(m.state_dict())}
        if ep % 50 == 0:
            print("[B1 %s s%d] ep %3d val=%.3f best=%.3f %.0fs"
                  % (arm, seed, ep, va, best["val"], time.time() - t0), flush=True)
    m.load_state_dict(best["state"])
    with torch.no_grad():
        te = R.evaluate(m, data["test"])
    pt = sha = None
    if save:
        os.makedirs(CKPT_DIR, exist_ok=True)
        pt = os.path.join(CKPT_DIR, "%s_%s_s%d.pt" % (arm, CELL, seed))
        torch.save(best["state"], pt)
        sha = hashlib.sha256(open(pt, "rb").read()).hexdigest()
    return {"arm": arm, "cell": CELL, "seed": seed, "assignment": "stratified",
            "train_acc": best["train"], "val_acc": best["val"],
            "heldout_test_acc": te, "exp61_reference": EXP61_M3_S1,
            "ckpt_path": pt, "ckpt_sha256": sha, "wall_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="S1,S-shuffle")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    blob = json.load(open(DATA, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != DATA_SHA:
        raise RuntimeError("exp61 dataset provenance FAILED")
    verbs = sorted(meta["verb_inventory"].keys())
    maxep = 2 if a.smoke else R.MAX_EPOCHS
    nseeds = 1 if a.smoke else a.seeds
    jobs = [(ar, s) for ar in a.arms.split(",") if ar for s in range(nseeds)]
    mine = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    if a.worker == 0:
        band = B62.assign_bands_3way(CELL, [n for n in meta["humans"] if n in set(meta["bearers"])],
                                     [n for n in meta["humans"] if n not in set(meta["bearers"])],
                                     [meta["train_inventory"], meta["heldout_inventory"]])
        for lab, inv in (("train", meta["train_inventory"]),
                         ("held-out", meta["heldout_inventory"])):
            print("[B1] stratified M3 occupancy %-9s %s"
                  % (lab, [sum(1 for n in inv if band[n] == c)
                           for c in B62.band_centres(CELL)]), flush=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    for arm, s in mine:
        ckj = os.path.join(CKPT_DIR, "%s_%s_s%d.json" % (arm, CELL, s))
        if os.path.exists(ckj) and not a.smoke:
            print("[ckpt] done", flush=True); continue
        r = train(arm, s, items, meta, verbs, maxep, save=not a.smoke)
        if not a.smoke:
            json.dump(r, open(ckj, "w", encoding="utf-8"))
        print("[B1] %-9s s%d  held-out=%.4f  (exp61 global-assignment: %.4f)"
              % (arm, s, r["heldout_test_acc"], EXP61_M3_S1), flush=True)
    print("[B1 w%d] done" % a.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
