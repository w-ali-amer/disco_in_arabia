# -*- coding: utf-8 -*-
"""exp53_run.py -- doc 37: does the model COMPOSE root x pattern?

Arms:
  C  compositional   U_v = U_root(r) . U_pat(p)   4+3 = 7 blocks
  A  atomic          one free block per verb      12 blocks (exp51's A3)
  T  tied            one shared block             sentinel, pinned at 0.500

C has FEWER parameters than A (7 x 15 = 105 vs 12 x 15 = 180) and fine-tuning
gives C ZERO additional ones, so no outcome favouring C can be a capacity
effect. That direction is foreclosed by construction, deliberately.

CRITERIA -- doc 37 SS4.1, which SUPERSEDES SS4's zero-shot bar. That bar was
voided by a measured confound: the readout applies block(v) in the story and
block(v)-dagger in the question, the SAME tensor, so the signal is
story-question CONSISTENCY not content, and a never-trained block answered at
0.6136 on exp51's rung -- above the 0.60 bar. Hence:

  A. novel_block_floor  measured HERE on this grid with randomly
     re-initialised held-out blocks (>=5 draws x all seeds). The n=44
     diagnostic voided a criterion; it must not set one.
  B. rp_composes  iff C's held-out accuracy at k=0 exceeds
     novel_block_floor + 2 SD AND exceeds A's at k=0.
     rp_composition_advantage  iff C beats A by >=5pp at k<=2.
     Few-shot is the robust instrument because the confound is
     k-INDEPENDENT: it lifts both arms equally at every k, so the DIFFERENCE
     between arms is uncontaminated while zero-shot accuracy is not.
  Positive control: both arms must clear train>=0.8 AND val>=0.7 on SEEN
  cells, else that arm is an untrained instrument (doc 24 SS4).
"""

import argparse, copy, json, math, os, platform, random, sys, time
import torch

import exp42_compiler as comp
import exp42_sim_adapter as sim
from exp42_models import QuantumStoryModel
from exp42_run import fresh_results_path, git_note, grad_global_norm
from exp51_run import predict, accuracy, loss_of, make_arm as _mk51

DESIGN_DOC = ("qnlp_private_docs/37_exp53_root_pattern_composition.md "
              "(1ba1469 criteria + 38a9fd4 amendment + b52d1f9 grid, FROZEN)")
DATA = "stories_exp53_RP.json"
EXPECTED_SHA = \
    "2bfdd9f8aa310974e25a200990db9275844bbf87fca19183508ec95d4686b4fd"
CKPT_DIR = "exp53_ckpt"
GRID_CFG = [(0.005, 1), (0.005, 8), (0.02, 1), (0.02, 8)]
N_SEEDS, MAX_EPOCHS = 2, 60
TRAIN_GATE, VAL_GATE = 0.8, 0.7
KS = [0, 1, 2, 4, 8]
N_FLOOR_DRAWS = 5
RNG_BASE = 53000

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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if not args.smoke:
        os.makedirs(CKPT_DIR, exist_ok=True)
    grid = GRID_CFG[-1:] if args.smoke else GRID_CFG
    nseeds = 1 if args.smoke else N_SEEDS
    maxep = 2 if args.smoke else MAX_EPOCHS
    ks = [0, 1] if args.smoke else KS
    ndraws = 2 if args.smoke else N_FLOOR_DRAWS

    blob = json.load(open(DATA, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    sha = comp.canonical_sha256(items)
    if sha != meta["dataset_sha256"] or sha != EXPECTED_SHA:
        raise RuntimeError("RP provenance FAILED: %s" % sha)
    angles, _e, emb_sha = comp.load_embeddings()
    verbs = sorted(meta["verb_inventory"].keys())
    cell_of = meta["cell_of"]
    held = meta["held_out_verbs"]
    data = {sp: comp.compile_all([i for i in items if i["split"] == sp], angles)
            for sp in ("train", "val", "test")}
    print("[exp53] RP sha=%s train/val/held-out=%d/%d/%d | SEEN=%d HELD=%d"
          % (sha[:12], len(data["train"]), len(data["val"]), len(data["test"]),
             len(meta["seen_verbs"]), len(held)), flush=True)

    # ---------- tied sentinel (doc 35 SS3 theorem, inherited)
    t_acc = accuracy(make_arm("T", verbs, 0, cell_of), "A3", data["test"])
    print("[exp53] tied sentinel on held-out: %.4f (must be 0.5000)" % t_acc,
          flush=True)

    # ---------- A. novel-block floor, measured on THIS grid
    floor = []
    for d in range(ndraws):
        m = make_arm("A", verbs, 900 + d, cell_of)
        floor.append(accuracy(m, "A3", data["test"]))
    fl_mean = sum(floor) / len(floor)
    fl_sd = (sum((x - fl_mean) ** 2 for x in floor) / max(len(floor) - 1, 1)) ** .5
    print("[exp53] untrained_floor (descriptive) = %.4f +/- %.4f  %s"
          % (fl_mean, fl_sd, [round(x, 3) for x in floor]), flush=True)
    print("[exp53] NOTE: doc 37 SS4.1's REGISTERED floor is a TRAINED model with "
          "random held-out blocks. In arm A the held-out verbs never appear in "
          "training, so their blocks never receive a gradient -- A's held-out "
          "distribution IS that condition, and it is computed after training "
          "below. The untrained floor above is descriptive only and is NOT the "
          "criterion.", flush=True)

    # ---------- train both arms
    per_arm, models = {}, {}
    for arm in ("C", "A"):
        runs, keep = [], []
        for lr, b in grid:
            for s in range(nseeds):
                key = "RP_%s_lr%s_b%d_s%d" % (arm, lr, b, s)
                ckj, ckp = (os.path.join(CKPT_DIR, key + ".json"),
                            os.path.join(CKPT_DIR, key + ".pt"))
                if not args.smoke and os.path.exists(ckj) and os.path.exists(ckp):
                    d = json.load(open(ckj, encoding="utf-8"))
                    m = make_arm(arm, verbs, s, cell_of)
                    m.load_state_dict(torch.load(ckp, weights_only=True))
                    print("[ckpt] %s loaded" % key, flush=True)
                else:
                    m, d = train(arm, s, lr, b, data, verbs, cell_of, maxep,
                                 tag="RP:")
                    if not args.smoke:
                        json.dump(d, open(ckj, "w", encoding="utf-8"),
                                  ensure_ascii=False)
                        torch.save(m.state_dict(), ckp)
                runs.append(d)
                keep.append(m)
        per_arm[arm] = runs
        bi = max(range(len(runs)), key=lambda i: runs[i]["val_acc"])
        models[arm] = keep[bi]
        print("[exp53] %s: seen-gate %d/%d, best val=%.3f, held-out=%.4f"
              % (arm, sum(1 for r in runs if r["gate_met"]), len(runs),
                 max(r["val_acc"] for r in runs),
                 runs[bi]["heldout_acc"]), flush=True)

    # ---------- B. few-shot curves
    curves = {}
    for arm in ("C", "A"):
        curves[arm] = {str(k): few_shot(arm, models[arm], data, verbs, cell_of,
                                        0, k, held) for k in ks}
        print("[exp53] %s few-shot: %s"
              % (arm, {k: round(v, 4) for k, v in curves[arm].items()}),
              flush=True)

    c0, a0 = curves["C"]["0"], curves["A"]["0"]
    # REGISTERED floor (doc 37 SS4.1 A): trained arm, random held-out blocks.
    a_hold = [r["heldout_acc"] for r in per_arm["A"]]
    reg_mean = sum(a_hold) / len(a_hold)
    reg_sd = (sum((x - reg_mean) ** 2 for x in a_hold)
              / max(len(a_hold) - 1, 1)) ** .5
    print("[exp53] REGISTERED novel_block_floor (arm A held-out, n=%d) = "
          "%.4f +/- %.4f" % (len(a_hold), reg_mean, reg_sd), flush=True)
    valid = {a: any(r["gate_met"] for r in per_arm[a]) for a in ("C", "A")}
    adv = [curves["C"][str(k)] - curves["A"][str(k)] for k in ks if k <= 2]
    verdicts = {
        "tied_sentinel_exact_half": bool(t_acc == 0.5),
        "novel_block_floor_note": ("REGISTERED floor = arm A held-out "
                                   "(trained model, never-trained held-out "
                                   "blocks). The untrained-model floor is "
                                   "descriptive only."),
        "rp_seen_learnable_C": valid["C"], "rp_seen_learnable_A": valid["A"],
        "C_heldout_k0": c0, "A_heldout_k0": a0,
        "untrained_floor_descriptive": fl_mean,
        "registered_novel_block_floor": reg_mean,
        "registered_novel_block_floor_sd": reg_sd,
        "rp_composes": bool(valid["C"] and c0 > reg_mean + 2 * reg_sd
                            and c0 > a0),
        "rp_composition_advantage": bool(valid["C"] and max(adv) >= 0.05),
    }
    out = {"experiment": "exp53", "spec": DESIGN_DOC,
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "dataset_sha256": sha,
           "embeddings_sha256": emb_sha, "grid": meta["grid"],
           "held_out_cells": meta["held_out_cells"],
           "grid_provenance": meta["grid_provenance"],
           "param_counts": {
               "C": make_arm("C", verbs, 0, cell_of).param_counts(),
               "A": make_arm("A", verbs, 0, cell_of).param_counts()},
           "tied_sentinel_heldout_acc": t_acc,
           "novel_block_floor_draws": floor,
           "per_arm": per_arm, "few_shot_curves": curves,
           "verdicts": verdicts,
           "meta_notes": {"scope": SCOPE, "factorisation_is_built_in": BUILT_IN},
           "git_note": git_note(), "wall_time_sec": time.time() - t0}
    print("\n[exp53] VERDICTS")
    for k, v in verdicts.items():
        print("   %-30s %s" % (k, v))
    if args.smoke:
        print("[exp53] smoke -- no results file")
        return 0
    p = fresh_results_path("results_exp53")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[exp53] wrote %s (%.1f min)" % (p, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
