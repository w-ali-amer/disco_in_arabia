# -*- coding: utf-8 -*-
"""exp65_run.py -- doc 51 SS11.4: does the circuit bind a REAL morphological
contrast (فاعِل vs non-فاعِل) and transfer it to unseen words?

  S-real      positional featurizer, fit TRAIN-ONLY, applied to held-out
  S-scramble  same featurizer on character-scrambled forms
  S-shuffle   the registered floor (angle<->label severed, per PAIR)

doc 51 SS11.5: the two featurizers' bounds differ by only 0.067 on the 8-noun test
inventory, so the scramble arm MAY BE STATISTICALLY UNRESOLVABLE. S-real approx
S-scramble within noise reads as "insufficient resolution at n=8" -- NOT as
evidence either way.
"""
import argparse, copy, hashlib, json, math, os, random, sys, time
import torch
torch.set_num_threads(1)
import exp42_compiler as comp
import exp65_featurizer as F
import exp61_run as R
from exp60_run import relabel_shuffle
from exp42_run import grad_global_norm

DESIGN_DOC = ("qnlp_private_docs/51_exp65_wazn_binding_PROPOSAL.md "
              "(d293f42, FROZEN incl. SS11-12)")
DATA = "stories_exp65_WZ.json"
CKPT_DIR = "exp65_ckpt"
ARMS = ["S-real", "S-scramble", "S-shuffle"]


def angles(arm, meta):
    return F.angles_for(meta["train_inventory"], meta["humans"],
                        scramble=(arm == "S-scramble"))


def train(arm, seed, items, meta, verbs, max_epochs, save=True):
    ang = angles(arm, meta)
    use = relabel_shuffle(items, meta, seed) if arm == "S-shuffle" else items
    data = {s: comp.compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}
    torch.manual_seed(seed)
    m = R.CapModel("A3", verbs, seed, reup=0)
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
                    torch.log(m.story_values(cs) + 1e-12).unsqueeze(0),
                    torch.tensor([cs.answer]))
            (bl / min(R.BATCH, len(order) - s)).backward()
            assert math.isfinite(grad_global_norm(m)); opt.step()
        with torch.no_grad():
            v = R.evaluate(m, data["val"])
        if v > best["v"]:
            best = {"v": v, "state": copy.deepcopy(m.state_dict())}
        if ep % 50 == 0:
            print("[%s s%d] ep %3d val=%.3f best=%.3f %.0fs"
                  % (arm, seed, ep, v, best["v"], time.time() - t0), flush=True)
    m.load_state_dict(best["state"])
    with torch.no_grad():
        te = R.evaluate(m, data["test"])
    pt = sha = None
    if save:
        os.makedirs(CKPT_DIR, exist_ok=True)
        pt = os.path.join(CKPT_DIR, "%s_s%d.pt" % (arm, seed))
        torch.save(best["state"], pt)
        sha = hashlib.sha256(open(pt, "rb").read()).hexdigest()
    return {"arm": arm, "seed": seed, "val_acc": best["v"],
            "heldout_test_acc": te, "ckpt_path": pt, "ckpt_sha256": sha,
            "wall_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    blob = json.load(open(DATA, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("WZ provenance FAILED")
    verbs = sorted(meta["verb_inventory"].keys())
    maxep = 2 if a.smoke else R.MAX_EPOCHS
    ns = 1 if a.smoke else a.seeds
    jobs = [(ar, s) for ar in ARMS for s in range(ns)]
    mine = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    print("[exp65 w%d/%d] %d runs" % (a.worker, a.nworkers, len(mine)), flush=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    for arm, s in mine:
        ckj = os.path.join(CKPT_DIR, "%s_s%d.json" % (arm, s))
        if os.path.exists(ckj) and not a.smoke:
            continue
        r = train(arm, s, items, meta, verbs, maxep, save=not a.smoke)
        if not a.smoke:
            json.dump(r, open(ckj, "w", encoding="utf-8"))
        print("[exp65] %-11s s%d  held-out=%.4f  val=%.4f"
              % (arm, s, r["heldout_test_acc"], r["val_acc"]), flush=True)
    print("[exp65 w%d] done" % a.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
