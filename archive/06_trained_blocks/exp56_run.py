# -*- coding: utf-8 -*-
"""exp56_run.py -- doc 41: does a TIGHTENED referent encoding lift RP over the bar?

NOT a third budget. Doc 38 SS4 prohibited extending COMPUTE and this extends
none: epochs, seeds, arms, grid, dataset and every doc 37 SS4.1 threshold are
inherited verbatim. The only change is the referent angle map.

Encodings:
  RP-A  AraVec angles -- the exp54 baseline, loaded from the ORIGINAL frozen
        artifact (exp42_noun_embeddings.json) with its sha asserted. NOT
        re-derived, and nothing here touches the exp55 phi artifact.
  RP-T  random angles, half-width 0.15 rad  (sweep optimum)
  RP-M  random angles, half-width 0.60 rad  (mid-curve guard against the
        optimum being a VT-specific artifact)

Exploratory sweep behind this (VT, random angles, 3 seeds, no verdicts):
  w   0.05 0.15 0.35 0.60 0.90 1.30 1.80 2.40  pi
  acc .852 .863 .807 .809 .772 .730 .616 .513 .517     Spearman -0.950
Consistency: w=pi is uniform and gave .517 vs exp55 E0 .5013; AraVec's
equivalent half-width 1.195 predicts ~.74 vs exp55 E1 .7373. An OFF-POLE
tight band at pi/2 gave .857, so the operative variable is SPREAD, not
pole-proximity.

PROVENANCE (binding after doc 40 SS1): angle maps come from a LITERAL fixed
seed table -- no hash() anywhere in the seeding path -- and every drawn
{noun: angle} map is written into the results JSON.

THE FLOOR IS PER-ENCODING BY DEFINITION. doc 37 SS4.1 defines
novel_block_floor as arm A's held-out distribution, which the encoding shift
moves. exp54's 0.5519 belongs to RP-A's angles ONLY and is never reused. If a
tightened band lifts the consistency mechanism across the board, A's floor
RISES and C must clear a HIGHER bar under RP-T than under RP-A -- that is
correct behaviour, not a fairness problem.
"""

import argparse, copy, json, math, os, platform, random, sys, time
import torch

import exp42_compiler as comp
import exp41_story_generator as gen
import exp42_sim_adapter as sim
from exp42_models import QuantumStoryModel
from exp42_run import fresh_results_path, git_note, grad_global_norm
from exp51_run import predict, accuracy, loss_of, make_arm as _mk51

DESIGN_DOC = ("qnlp_private_docs/41_exp56_tightened_encoding_rp.md (e7081cc), "
              "inheriting 37 (1ba1469+38a9fd4+b52d1f9) criteria UNCHANGED")
DATA = "stories_exp53_RP.json"
EXPECTED_SHA = \
    "2bfdd9f8aa310974e25a200990db9275844bbf87fca19183508ec95d4686b4fd"
CKPT_DIR = "exp56_ckpt"
GRID_CFG = [(0.005, 1), (0.005, 8), (0.02, 1), (0.02, 8)]
N_SEEDS, MAX_EPOCHS = 2, 150   # doc 38: 60 -> 150, the ONLY change
TRAIN_GATE, VAL_GATE = 0.8, 0.7
KS = [0, 1, 2, 4, 8]
N_FLOOR_DRAWS = 5
RNG_BASE = 53000
ENCODINGS = {"RP-A": None, "RP-T": 0.15, "RP-M": 0.60}   # None = frozen AraVec
ANGLE_SEEDS = {("RP-T", 0): 56001, ("RP-T", 1): 56002, ("RP-T", 2): 56003,
               ("RP-T", 3): 56004, ("RP-T", 4): 56005,
               ("RP-M", 0): 56011, ("RP-M", 1): 56012, ("RP-M", 2): 56013,
               ("RP-M", 3): 56014, ("RP-M", 4): 56015}
FLOOR_NOTE = ("doc 37 SS4.1 defines novel_block_floor as arm A's held-out "
              "distribution, so it is PER-ENCODING by definition. exp54's "
              "0.5519 belongs to RP-A only and is never reused. A rising floor "
              "under RP-T is the consistency mechanism inflating, NOT a "
              "fairness problem.")


def angle_map(enc, seed, base):
    """RP-A returns the ORIGINAL frozen artifact map untouched. Others draw a
    band at the pole from the LITERAL seed table -- no hash() in this path."""
    if ENCODINGS[enc] is None:
        return dict(base), None
    w = ENCODINGS[enc]
    rng = random.Random(ANGLE_SEEDS[(enc, seed)])
    ang = dict(base)
    drawn = {h: math.pi + rng.uniform(-w, w) for h in gen.HUMANS}
    ang.update(drawn)
    return ang, drawn

SCOPE = ("doc 37 SS7: not quantum advantage (no arm comparison of that kind is "
         "registered); composition of verb FORM, not of meaning -- generalising "
         "to an unseen root x pattern cell shows the model factorises form, and "
         "whether the block means anything is a separate question; R=4 x P=3 is "
         "a small grid with 4 held-out cells; it does not revive the wazn line, "
         "which closed negative on its own criteria (exp50); simulation only.")

BUILT_IN = ("arm C BUILDS IN the factorisation, so a positive says 'the "
            "morphology is learnable compositionally when the architecture is "
            "given the factorisation', NOT 'models spontaneously discover "
            "wazn'. The passive probes asked the latter and failed.")


class CompositionalModel(QuantumStoryModel):
    """U_v = U_root(root(v)) . U_pat(pattern(v)). Parameters are shared across
    the grid, so a held-out cell is fully determined by parts seen in
    training -- zero new parameters."""

    def __init__(self, verbs, seed, cell_of):
        super().__init__("A3", verbs, seed)
        self.cell_of = {v: tuple(c) for v, c in cell_of.items()}
        roots = sorted({c[0] for c in self.cell_of.values()})
        pats = sorted({c[1] for c in self.cell_of.values()})
        g = torch.Generator().manual_seed(seed + 7717)
        del self.blocks
        # ModuleDict keys cannot contain "."; roots are dotted (خ.ب.ر)
        self._rk = {r: r.replace(".", "_") for r in roots}
        self.root_blocks = torch.nn.ModuleDict(
            {self._rk[r]: sim.Circuit4(2, 3, generator=g) for r in roots})
        self.pat_blocks = torch.nn.ModuleDict(
            {p: sim.Circuit4(2, 3, generator=g) for p in pats})

    def verb_matrix(self, v):
        r, p = self.cell_of[v]
        return self.root_blocks[self._rk[r]].matrix() @ \
            self.pat_blocks[p].matrix()

    def param_counts(self):
        tot = sum(p.numel() for p in self.parameters())
        return {"arm": "C", "n_root_blocks": len(self.root_blocks),
                "n_pat_blocks": len(self.pat_blocks), "total_trained": tot}


def make_arm(arm, verbs, seed, cell_of):
    if arm == "C":
        return CompositionalModel(verbs, seed, cell_of)
    return _mk51("A3" if arm == "A" else "T", verbs, seed)


def train(arm, seed, lr, batch, data, verbs, cell_of, max_epochs, tag=""):
    torch.manual_seed(seed)
    m = make_arm(arm, verbs, seed, cell_of)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    rng = random.Random(RNG_BASE + seed)
    best = {"val_acc": -1.0, "train_acc": -1.0, "state": None}
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        order = list(range(len(data["train"])))
        rng.shuffle(order)
        losses = []
        for s in range(0, len(order), batch):
            bat = [data["train"][i] for i in order[s:s + batch]]
            opt.zero_grad()
            bl = 0.0
            for cs in bat:
                bl = bl + loss_of(m, "A3", cs)
            bl = bl / len(bat)
            bl.backward()
            assert math.isfinite(grad_global_norm(m))
            opt.step()
            losses.append(bl.item())
        va = accuracy(m, "A3", data["val"])
        if va > best["val_acc"]:
            best = {"val_acc": va, "train_acc": accuracy(m, "A3", data["train"]),
                    "state": copy.deepcopy(m.state_dict())}
        print("[%s%s s%d lr%.3f b%d] ep %2d/%d loss=%.4f val=%.3f best=%.3f %.1fs"
              % (tag, arm, seed, lr, batch, ep, max_epochs,
                 sum(losses) / len(losses), va, best["val_acc"],
                 time.time() - t0), flush=True)
    m.load_state_dict(best["state"])
    return m, {"arm": arm, "seed": seed, "lr": lr, "batch": batch,
               "val_acc": best["val_acc"], "train_acc": best["train_acc"],
               "heldout_acc": accuracy(m, "A3", data["test"]),
               "gate_met": bool(best["train_acc"] >= TRAIN_GATE
                                and best["val_acc"] >= VAL_GATE)}


def few_shot(arm, model, data, verbs, cell_of, seed, k, held_verbs):
    """Freeze everything; fine-tune ONLY the held-out cell's representation on
    k examples. Arm C has zero such parameters -- its curve is flat by
    construction, which is the point."""
    if k == 0 or arm == "C":
        return accuracy(model, "A3", data["test"])
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    tune = [p for n, p in m.named_parameters()
            if any(("blocks.%s." % v) in n for v in held_verbs)]
    for p in tune:
        p.requires_grad_(True)
    if not tune:
        return accuracy(m, "A3", data["test"])
    rng = random.Random(RNG_BASE + seed + k)
    shots = rng.sample(data["test"], min(k * 2, len(data["test"])))
    opt = torch.optim.Adam(tune, lr=0.02)
    for _ in range(40):
        opt.zero_grad()
        bl = sum(loss_of(m, "A3", cs) for cs in shots) / len(shots)
        bl.backward()
        opt.step()
    rest = [cs for cs in data["test"] if cs not in shots]
    return accuracy(m, "A3", rest)


def run_encoding(enc, items, meta, base, verbs, cell_of, held, grid,
                 nseeds, maxep, ks, ndraws, smoke):
    ang, drawn = angle_map(enc, 0, base)
    if enc == "RP-A":
        assert ang == dict(base), "RP-A must be the ORIGINAL frozen map"
    data = {sp: comp.compile_all([i for i in items if i["split"] == sp], ang)
            for sp in ("train", "val", "test")}
    # tied sentinel verified PER ENCODING: the theorem is encoding-independent,
    # so a reading != 0.5000 here is an implementation bug, not physics.
    t_acc = accuracy(make_arm("T", verbs, 0, cell_of), "A3", data["test"])
    print("[exp56][%s] tied sentinel = %.4f (must be 0.5000)" % (enc, t_acc),
          flush=True)
    per_arm, models, maps = {}, {}, {}
    for arm in ("C", "A"):
        runs, keep = [], []
        for lr, b in grid:
            for s in range(nseeds):
                a_s, d_s = angle_map(enc, s, base)
                if d_s is not None:
                    maps["%s_s%d" % (enc, s)] = d_s
                dat = data if enc == "RP-A" else {
                    sp: comp.compile_all(
                        [i for i in items if i["split"] == sp], a_s)
                    for sp in ("train", "val", "test")}
                key = "%s_%s_lr%s_b%d_s%d" % (enc, arm, lr, b, s)
                ckj = os.path.join(CKPT_DIR, key + ".json")
                ckp = os.path.join(CKPT_DIR, key + ".pt")
                if not smoke and os.path.exists(ckj) and os.path.exists(ckp):
                    d = json.load(open(ckj, encoding="utf-8"))
                    m = make_arm(arm, verbs, s, cell_of)
                    m.load_state_dict(torch.load(ckp, weights_only=True))
                    print("[ckpt] %s loaded" % key, flush=True)
                else:
                    m, d = train(arm, s, lr, b, dat, verbs, cell_of, maxep,
                                 tag=enc + ":")
                    if not smoke:
                        json.dump(d, open(ckj, "w", encoding="utf-8"),
                                  ensure_ascii=False)
                        torch.save(m.state_dict(), ckp)
                runs.append(d); keep.append(m)
        per_arm[arm] = runs
        bi = max(range(len(runs)), key=lambda i: runs[i]["val_acc"])
        models[arm] = keep[bi]
        print("[exp56][%s] %s: seen-gate %d/%d best val=%.3f held-out=%.4f"
              % (enc, arm, sum(1 for r in runs if r["gate_met"]), len(runs),
                 max(r["val_acc"] for r in runs), runs[bi]["heldout_acc"]),
              flush=True)
    curves = {a: {str(k): few_shot(a, models[a], data, verbs, cell_of, 0, k,
                                   held) for k in ks} for a in ("C", "A")}
    a_hold = [r["heldout_acc"] for r in per_arm["A"]]
    reg_mean = sum(a_hold) / len(a_hold)
    reg_sd = (sum((x - reg_mean) ** 2 for x in a_hold)
              / max(len(a_hold) - 1, 1)) ** .5
    c0, a0 = curves["C"]["0"], curves["A"]["0"]
    valid = {a: any(r["gate_met"] for r in per_arm[a]) for a in ("C", "A")}
    adv = [curves["C"][str(k)] - curves["A"][str(k)] for k in ks if k <= 2]
    print("[exp56][%s] REGISTERED floor (arm A held-out, n=%d) = %.4f +/- %.4f"
          % (enc, len(a_hold), reg_mean, reg_sd), flush=True)
    return {"encoding": enc, "half_width": ENCODINGS[enc],
            "tied_sentinel": t_acc, "drawn_angle_maps": maps,
            "per_arm": per_arm, "few_shot_curves": curves,
            "registered_novel_block_floor": reg_mean,
            "registered_novel_block_floor_sd": reg_sd,
            "C_heldout_k0": c0, "A_heldout_k0": a0,
            "rp_seen_learnable_C": valid["C"], "rp_seen_learnable_A": valid["A"],
            "rp_composes": bool(valid["C"] and c0 > reg_mean + 2 * reg_sd
                                and c0 > a0),
            "rp_composition_advantage": bool(valid["C"] and max(adv) >= 0.05)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--drop", default="")
    args = ap.parse_args()
    t0 = time.time()
    if not args.smoke:
        os.makedirs(CKPT_DIR, exist_ok=True)
    grid = GRID_CFG[-1:] if args.smoke else GRID_CFG
    nseeds = 1 if args.smoke else N_SEEDS
    maxep = 2 if args.smoke else MAX_EPOCHS
    ks = [0, 1] if args.smoke else KS
    ndraws = 2 if args.smoke else N_FLOOR_DRAWS
    encs = [e for e in ("RP-A", "RP-T", "RP-M") if e not in args.drop.split(",")]

    blob = json.load(open(DATA, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    sha = comp.canonical_sha256(items)
    if sha != meta["dataset_sha256"] or sha != EXPECTED_SHA:
        raise RuntimeError("RP provenance FAILED: %s" % sha)
    base, _e, emb_sha = comp.load_embeddings()
    verbs = sorted(meta["verb_inventory"].keys())
    cell_of, held = meta["cell_of"], meta["held_out_verbs"]
    print("[exp56] RP sha ok | frozen emb sha %s | encodings=%s"
          % (emb_sha[:12], encs), flush=True)

    res = {}
    for enc in encs:
        res[enc] = run_encoding(enc, items, meta, base, verbs, cell_of, held,
                                grid, nseeds, maxep, ks, ndraws, args.smoke)

    lifted = []
    if "RP-A" in res:
        for enc in encs:
            if enc == "RP-A":
                continue
            for arm in ("C", "A"):
                if res[enc]["rp_seen_learnable_" + arm] and \
                        not res["RP-A"]["rp_seen_learnable_" + arm]:
                    lifted.append("%s/%s" % (enc, arm))
    verdicts = {"encoding_lifts_rp": bool(lifted), "lifted": lifted}
    for enc in encs:
        for k in ("rp_seen_learnable_C", "rp_seen_learnable_A", "rp_composes",
                  "rp_composition_advantage", "registered_novel_block_floor"):
            verdicts["%s__%s" % (enc, k)] = res[enc][k]

    out = {"experiment": "exp56", "spec": DESIGN_DOC,
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "dataset_sha256": sha,
           "frozen_embeddings_sha256": emb_sha,
           "encodings": {k: v for k, v in ENCODINGS.items() if k in encs},
           "angle_seed_table": {"%s|%d" % k: v for k, v in ANGLE_SEEDS.items()},
           "per_encoding": res, "verdicts": verdicts,
           "meta_notes": {"floor_is_per_encoding": FLOOR_NOTE,
                          "not_a_budget_extension": (
                              "doc 38 prohibited extending COMPUTE; this "
                              "extends none. Epochs, seeds, arms, grid, "
                              "dataset and every doc 37 SS4.1 threshold are "
                              "inherited verbatim; only the angle map changes."),
                          "no_mechanism_claimed": (
                              "doc 41 SS6: a positive shows a conditioned "
                              "encoding lifts the rung; the WHY is not "
                              "claimed here and is a separate question.")},
           "git_note": git_note(), "wall_time_sec": time.time() - t0}
    print("\n[exp56] VERDICTS")
    for k, v in verdicts.items():
        print("   %-38s %s" % (k, v))
    if args.smoke:
        print("[exp56] smoke -- no results file")
        return 0
    p = fresh_results_path("results_exp56")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[exp56] wrote %s (%.1f min)" % (p, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
