# -*- coding: utf-8 -*-
"""exp90_run.py -- doc 83: run ONE (draw, arm, condition, seed) unit, or a
worker's shard of the registered grid. Each unit writes its own JSON + .pt to
the output dir (default exp90_ckpt/); an existing unit JSON is NEVER
overwritten (grid mode skips it; single-unit mode refuses).

Recipe (doc 83 SS11.1 = exp60 verbatim): Adam, LR 0.02, batch 8, <= 150
epochs, best-val checkpoint (strictly greater val), torch.set_num_threads(1),
torch.manual_seed(seed), batch order from random.Random(55000 + seed).
"converged" iff train >= 0.8 and val >= 0.7; non-converged units are kept.

Angle map: exp60_run.build_angles("S-base", 0.40, 0.05, seed, bearers,
humans) with the run seed and the draw's own bearers (SS11.1).

M-shuf (SS5 floor): train+val vowels re-drawn per item independent of role
(one NOM + one ACC, slot assignment ~ fair coin from
random.Random(9_090_000 + seed), the exp60 S-shuffle per-run-seed precedent);
test uses the true vowels.

Grid (registered_grid()): Rung A (k=0.5) + Rung B doses {k0, knat, k25} x
10 learner arms x seeds 0-11 on the PRIMARY draw, plus R-pure at k0 only;
the CONFIRM draw re-runs all 10 learner arms x 4 conditions x 12 seeds
(doc 70 SS3 rule 5: re-run every arm whose failure would invalidate the
draw -- conservative reading, flagged).
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time

import torch

torch.set_num_threads(1)

import exp60_run as e60                              # noqa: E402
from exp42_run import grad_global_norm              # noqa: E402
import exp90_encoder as enc                         # noqa: E402
from exp90_common import (K_CONDS, CELL, compile_surface, file_sha256,  # noqa: E402
                          provenance, write_new_json)
from exp90_datagen import render_surface            # noqa: E402
from exp90_models import make_model, QUANTUM_ARMS   # noqa: E402

LR, BATCH, MAX_EPOCHS = 0.02, 8, 150
TRAIN_GATE, VAL_GATE = 0.8, 0.7
RNG_BASE = 55000                                    # exp60
SHUF_BASE = 9_090_000
DRAWS = {"primary": "stories_exp90_primary.json",
         "confirm": "stories_exp90_confirm.json"}
LEARNERS = ["M-id", "M-rand", "P", "M-shuf", "R-und", "R-rand",
            "CLD-id", "CLD-rand", "CL-MLP", "CL-LSTM"]
CONDS = ["A", "k0", "knat", "k25"]
SEEDS = list(range(12))


def registered_grid():
    units = []
    for draw in ("primary", "confirm"):
        for cond in CONDS:                           # Rung A first (SS11)
            for s in SEEDS:
                for arm in LEARNERS:
                    units.append((draw, arm, cond, s))
            if draw == "primary" and cond == "k0":
                units += [(draw, "R-pure", "k0", s) for s in SEEDS]
    return units


def unit_key(draw, arm, cond, seed):
    return "%s_%s_%s_s%d" % (draw, arm, cond, seed)


def load_draw(draw):
    path = DRAWS[draw]
    blob = json.load(open(path, encoding="utf-8"))
    from exp90_common import canonical_sha256
    data = {"test": blob["test"], "train_sets": blob["train_sets"]}
    if canonical_sha256(data) != blob["meta"]["data_sha256"]:
        raise RuntimeError("exp90 data provenance FAILED: %s" % path)
    return blob, path


def shuffle_vowels(items, seed):
    rng = random.Random(SHUF_BASE + seed)
    out = []
    for it in items:
        j = copy.deepcopy(it)
        cases = [enc.NOM, enc.ACC] if rng.random() < 0.5 else [enc.ACC, enc.NOM]
        j["slot_cases"] = cases
        j["surface"] = render_surface(j["verb"], j["slots"], cases, j["order"])
        j["surface_unvoc"] = enc.strip_marks(j["surface"])
        out.append(j)
    return out


def acc_of(preds, cs):
    return sum(int(p == c.answer) for p, c in zip(preds, cs)) / len(cs)


def run_unit(draw, arm, cond, seed, max_epochs=MAX_EPOCHS, outdir="exp90_ckpt",
             quiet=False):
    if arm == "R-pure" and cond != "k0":
        raise ValueError("R-pure is registered at k = 0 only (doc 83 SS4.2)")
    blob, path = load_draw(draw)
    meta = blob["meta"]
    verbs = meta["verbs"]
    ang = e60.build_angles("S-base", CELL[0], CELL[1], seed,
                           set(meta["bearers"]), meta["humans"])
    tv = blob["train_sets"][cond]
    if arm == "M-shuf":
        tv = shuffle_vowels(tv, seed)
    data = {s: [compile_surface(i, ang, verbs) for i in tv if i["split"] == s]
            for s in ("train", "val")}
    test_items = blob["test"]
    test = {v: [compile_surface(i, ang, verbs, view=v) for i in test_items]
            for v in ("voc", "unvoc")}
    types = [i["item_type"] for i in test_items]

    torch.manual_seed(seed)
    m = make_model(arm, verbs, seed)
    box0 = m.box_state()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    rng = random.Random(RNG_BASE + seed)
    best = {"val": -1.0, "train": -1.0, "state": None, "epoch": 0}
    t0 = time.time()
    epoch_times = []
    for ep in range(1, max_epochs + 1):
        te0 = time.time()
        order = list(range(len(data["train"])))
        rng.shuffle(order)
        for s in range(0, len(order), BATCH):
            opt.zero_grad()
            loss = m.loss_batch([data["train"][i] for i in order[s:s + BATCH]])
            loss.backward()
            assert math.isfinite(grad_global_norm(m))
            opt.step()
        va = acc_of(m.predict_many(data["val"]), data["val"])
        if va > best["val"]:
            best = {"val": va, "train": acc_of(m.predict_many(data["train"]),
                                               data["train"]),
                    "state": copy.deepcopy(m.state_dict()), "epoch": ep}
        epoch_times.append(time.time() - te0)
        if not quiet and (ep % 25 == 0 or ep == max_epochs):
            print("[%s] ep %3d/%d val=%.3f best=%.3f@%d %.0fs"
                  % (unit_key(draw, arm, cond, seed), ep, max_epochs, va,
                     best["val"], best["epoch"], time.time() - t0), flush=True)
    m.load_state_dict(best["state"])
    preds = {v: m.predict_many(test[v]) for v in ("voc", "unvoc")}
    res_acc = {}
    for v in ("voc", "unvoc"):
        for ty in ("CONSISTENT", "CONFLICT"):
            idx = [i for i, t in enumerate(types) if t == ty]
            res_acc["%s_%s" % (v, ty)] = sum(
                int(preds[v][i] == test[v][i].answer) for i in idx) / len(idx)
    os.makedirs(outdir, exist_ok=True)
    key = unit_key(draw, arm, cond, seed)
    pt = os.path.join(outdir, key + ".pt")
    if os.path.exists(pt):
        raise FileExistsError(pt)
    torch.save(best["state"], pt)
    counts = meta["counts"][cond]
    rec = {
        "unit": key, "draw": draw, "arm": arm, "cond": cond,
        "k": K_CONDS[cond], "seed": seed, "cell": "D0.40_w0.05",
        "init": getattr(m, "init_kind", None),
        "quantum": arm in QUANTUM_ARMS,
        "max_epochs": max_epochs, "best_epoch": best["epoch"],
        "train_acc": best["train"], "val_acc": best["val"],
        "converged": bool(best["train"] >= TRAIN_GATE and best["val"] >= VAL_GATE),
        "consistent_acc": res_acc["voc_CONSISTENT"],
        "conflict_acc": res_acc["voc_CONFLICT"],
        "unvoc_consistent_acc": res_acc["unvoc_CONSISTENT"],
        "unvoc_conflict_acc": res_acc["unvoc_CONFLICT"],
        "test_item_ids": [i["id"] for i in test_items],
        "test_item_types": "".join("S" if t == "CONSISTENT" else "C" for t in types),
        "test_answers": "".join(str(c.answer) for c in test["voc"]),
        "pred_voc": "".join(map(str, preds["voc"])),
        "pred_unvoc": "".join(map(str, preds["unvoc"])),
        "params_init": box0, "params_final": m.box_state(),
        "n_conflict_train": counts["n_conflict_train"],
        "n_conflict_val": counts["n_conflict_val"],
        "m_shuf_vowel_seed": (SHUF_BASE + seed) if arm == "M-shuf" else None,
        "ckpt_path": pt, "ckpt_sha256": file_sha256(pt),
        "wall_sec": time.time() - t0,
        "mean_epoch_sec": sum(epoch_times) / len(epoch_times),
        "data_file": path, "data_payload_sha256": blob.get("payload_sha256"),
        "provenance": provenance(extra_files=[path]),
    }
    write_new_json(os.path.join(outdir, key + ".json"), rec)
    if not quiet:
        print("[exp90] %-34s train=%.3f val=%.3f CONS=%.3f CONF=%.3f "
              "unvoc=%.3f/%.3f %s %.0fs"
              % (key, rec["train_acc"], rec["val_acc"], rec["consistent_acc"],
                 rec["conflict_acc"], rec["unvoc_consistent_acc"],
                 rec["unvoc_conflict_acc"],
                 "CONV" if rec["converged"] else "noconv", rec["wall_sec"]),
              flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draw", choices=sorted(DRAWS))
    ap.add_argument("--arm")
    ap.add_argument("--cond", choices=CONDS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--grid", action="store_true",
                    help="run this worker's shard of registered_grid()")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--outdir", default="exp90_ckpt")
    ap.add_argument("--draws", default="primary,confirm",
                    help="grid mode: restrict to these draws (e.g. primary first)")
    args = ap.parse_args()
    is_ckpt = os.path.abspath(args.outdir) == os.path.abspath("exp90_ckpt")
    if args.grid:
        if args.max_epochs != MAX_EPOCHS and is_ckpt:
            raise SystemExit("non-registered epochs must not write exp90_ckpt/")
        units = [u for u in registered_grid() if u[0] in args.draws.split(",")]
        # balanced shard: quantum and classical units dealt separately (order
        # kept: Rung A first), so every worker gets ~equal quantum load
        q = [u for u in units if u[1] in QUANTUM_ARMS]
        c = [u for u in units if u[1] not in QUANTUM_ARMS]
        mine = q[args.worker::args.nworkers] + c[args.worker::args.nworkers]
        print("[exp90 w%d/%d] %d of %d units" % (args.worker, args.nworkers,
                                                 len(mine), len(units)), flush=True)
        for u in mine:
            base = os.path.join(args.outdir, unit_key(*u))
            if os.path.exists(base + ".json"):
                print("[skip] %s exists" % unit_key(*u), flush=True)
                continue
            if os.path.exists(base + ".pt"):     # partial (crash/sleep before JSON)
                print("[partial] removing orphan %s.pt (no JSON = no result)"
                      % unit_key(*u), flush=True)
                os.remove(base + ".pt")
            run_unit(*u, max_epochs=args.max_epochs, outdir=args.outdir)
        print("[exp90 w%d] done" % args.worker, flush=True)
        return 0
    if None in (args.draw, args.arm, args.cond, args.seed):
        raise SystemExit("single-unit mode needs --draw --arm --cond --seed")
    if args.max_epochs != MAX_EPOCHS and is_ckpt:
        raise SystemExit("non-registered epochs must not write exp90_ckpt/")
    run_unit(args.draw, args.arm, args.cond, args.seed,
             max_epochs=args.max_epochs, outdir=args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
