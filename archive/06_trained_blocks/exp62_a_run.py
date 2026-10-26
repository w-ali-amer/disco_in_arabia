# -*- coding: utf-8 -*-
"""exp62_a_run.py -- doc 48 SS8: DUAL-VALIDATION, within-run.

One trajectory, selected two ways:
    selected_test_indist = test[ argmax_epoch val_indist ]
    selected_test_novel  = test[ argmax_epoch val_novel  ]
The difference is PURELY selection -- same run, same weights, no cross-dataset
or seed noise. Tie-break for BOTH rules: the FIRST epoch achieving the max
(SS8.3).

SS8.2: per-epoch test values are DIAGNOSTIC CURVES. They are never a selection
input and are never reported as model performance. The only reported numbers are
the two selected-checkpoint test accuracies. `oracle` stays diagnostic-only.
"""
import argparse, copy, hashlib, json, math, os, random, sys, time
import torch
torch.set_num_threads(1)
import exp42_compiler as comp
import exp62_bands as B
import exp61_run as R                      # CapModel / evaluate imported, not copied
from exp60_run import relabel_shuffle
from exp42_run import grad_global_norm

DESIGN_DOC = ("qnlp_private_docs/48_exp62_selection_optimisation_width.md "
              "(19e69d2, FROZEN incl. SS8)")
DATA = "stories_exp62_SB3.json"
CKPT_DIR = "exp62a_ckpt"
CELLS = ["M1", "M2", "M3"]
ARMS = ["S1", "S2", "S-shuffle"]
SPLITS = ("train", "val_indist", "val_novel", "test")


def angles(arm, cell, seed, meta):
    bear = [n for n in meta["humans"] if n in set(meta["bearers"])]
    non = [n for n in meta["humans"] if n not in set(meta["bearers"])]
    invs = [meta["train_inventory"], meta["valnovel_inventory"],
            meta["heldout_inventory"]]
    return B.build_angles_3way(cell, seed, bear, non, invs,
                               flat=(arm == "S-flat"))


def train(arm, cell, seed, items, meta, verbs, max_epochs, save=True):
    ang = angles(arm, cell, seed, meta)
    use = relabel_shuffle(items, meta, seed) if arm == "S-shuffle" else items
    data = {s: comp.compile_all([i for i in use if i["split"] == s], ang)
            for s in SPLITS}
    torch.manual_seed(seed)
    m = R.CapModel("A3", verbs, seed, reup=R.REUP.get(arm, 0))
    opt = torch.optim.Adam(m.parameters(), lr=R.LR)
    rng = random.Random(R.RNG_BASE + seed)
    curve = []
    best = {"indist": {"v": -1.0, "state": None, "ep": None},
            "novel": {"v": -1.0, "state": None, "ep": None}}
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
            vn = R.evaluate(m, data["val_novel"])
            te = R.evaluate(m, data["test"])        # DIAGNOSTIC curve only
        curve.append({"epoch": ep, "val_indist": vi, "val_novel": vn, "test": te})
        for key, v in (("indist", vi), ("novel", vn)):
            if v > best[key]["v"]:                  # SS8.3: strict > == FIRST max
                best[key] = {"v": v, "ep": ep,
                             "state": copy.deepcopy(m.state_dict())}
        if ep % 50 == 0:
            print("[%s %s s%d] ep %3d vi=%.3f vn=%.3f te=%.3f"
                  % (arm, cell, seed, ep, vi, vn, te), flush=True)
    sel = {}
    for key in ("indist", "novel"):
        m.load_state_dict(best[key]["state"])
        with torch.no_grad():
            sel[key] = R.evaluate(m, data["test"])
        if save:
            os.makedirs(CKPT_DIR, exist_ok=True)
            p = os.path.join(CKPT_DIR, "%s_%s_s%d_%s.pt" % (arm, cell, seed, key))
            torch.save(best[key]["state"], p)
            sel[key + "_ckpt_sha256"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    oracle = max(c["test"] for c in curve)
    return {"arm": arm, "cell": cell, "seed": seed,
            "selected_test_indist": sel["indist"],
            "selected_test_novel": sel["novel"],
            "novel_minus_indist": sel["novel"] - sel["indist"],
            "oracle_test": oracle,
            "epoch_indist": best["indist"]["ep"], "epoch_novel": best["novel"]["ep"],
            "best_val_indist": best["indist"]["v"], "best_val_novel": best["novel"]["v"],
            "ckpt_sha_indist": sel.get("indist_ckpt_sha256"),
            "ckpt_sha_novel": sel.get("novel_ckpt_sha256"),
            "curve": curve, "wall_sec": time.time() - t0}


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
    arms = [x for x in a.arms.split(",") if x]
    cells = [c for c in a.cells.split(",") if c]
    maxep = 2 if a.smoke else R.MAX_EPOCHS
    nseeds = 1 if a.smoke else a.seeds

    # NO tied-verb sentinel here. doc 46 SS11: SB has NO exact 0.5000 theorem --
    # its twins differ by ARGUMENT exchange, so tying verbs does not collapse
    # them (measured 0.4625, near chance but not pinned). Neither exp60 nor
    # exp61 carries one. SB's controls are EMPIRICAL: the S-shuffle floor, which
    # is an arm in this run. Importing VT's instrument here was an error.

    jobs = [(ar, c, s) for ar in arms for c in cells for s in range(nseeds)]
    mine = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    print("[exp62-A w%d/%d] %d runs" % (a.worker, a.nworkers, len(mine)), flush=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    for arm, cell, s in mine:
        key = "%s_%s_s%d" % (arm, cell, s)
        ckj = os.path.join(CKPT_DIR, key + ".json")
        if os.path.exists(ckj) and not a.smoke:
            print("[ckpt] %s done" % key, flush=True); continue
        r = train(arm, cell, s, items, meta, verbs, maxep, save=not a.smoke)
        if not a.smoke:
            json.dump(r, open(ckj, "w", encoding="utf-8"))
        print("[exp62-A] %-9s %-3s s%d  indist=%.4f (ep%s)  novel=%.4f (ep%s)  "
              "diff=%+.4f  oracle=%.4f"
              % (arm, cell, s, r["selected_test_indist"], r["epoch_indist"],
                 r["selected_test_novel"], r["epoch_novel"],
                 r["novel_minus_indist"], r["oracle_test"]), flush=True)
    print("[exp62-A w%d] done" % a.worker, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
