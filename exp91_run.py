# -*- coding: utf-8 -*-
"""exp91_run.py -- doc 84: run ONE (draw, learner, n, seed) unit, or a worker's
shard of the registered 1,440-unit grid, through exp90's OWN training code.

Wrapper, no edits to exp90 files: exp90_run.run_unit reads exactly two
module-level names that are condition/draw specific --
    exp90_run.DRAWS[draw]   (dataset path, via load_draw)
    exp90_run.K_CONDS[cond] (the unit's recorded "k")
-- plus `write_new_json` for the unit record. This module rebinds those three
names on the imported exp90_run module (DRAWS -> the exp91 datasets, K_CONDS ->
{"n<n>": n/470}, write_new_json -> the same writer after appending an `exp91`
provenance block). Everything else (angles, compiler, models, recipe: Adam LR
0.02, batch 8, 150 epochs, best-val, seeds, batch-order RNG) is exp90's code
object, unchanged.

    python3 exp91_run.py --draw primary --arm M-id --n 12 --seed 0
    python3 exp91_run.py --grid --worker w --nworkers 10
"""
import argparse
import os
import sys

import torch

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)

import exp90_run                                          # noqa: E402
import exp90_common                                       # noqa: E402
from exp90_models import QUANTUM_ARMS                     # noqa: E402
from exp91_datagen import LADDER, DESIGN_DOC_91           # noqa: E402

MAX_EPOCHS = exp90_run.MAX_EPOCHS                         # 150 (doc 84 SS2)
assert MAX_EPOCHS == 150
DRAWS91 = {"primary": "stories_exp91_primary.json",
           "confirm": "stories_exp91_confirm.json"}
LEARNERS = ["M-id", "M-rand", "CLD-id", "CLD-rand", "CL-MLP", "CL-LSTM"]
SEEDS = list(range(12))
N_TRAIN = 470
K91 = {"n%d" % n: n / N_TRAIN for n in LADDER}
CKPT = "exp91_ckpt"

_orig_writer = exp90_common.write_new_json


def _exp91_writer(path, obj):
    obj = dict(obj)
    if "unit" in obj and "cond" in obj:
        cond = obj["cond"]
        obj["exp91"] = {
            "design_doc": DESIGN_DOC_91,
            "n": int(cond[1:]),
            "k_note": "k = n / 470 (effective train dose; exact-count rendering)",
            "runner": "exp91_run.py -> exp90_run.run_unit (exp90 code, "
                      "DRAWS/K_CONDS/write_new_json rebound)",
            "module_sha256_all_repo": exp90_common.module_shas(),
            "exp91_files_sha256": {f: exp90_common.file_sha256(os.path.join(REPO, f))
                                   for f in ("exp91_run.py", "exp91_datagen.py")},
        }
    return _orig_writer(path, obj)


def install():
    """Point exp90_run at the exp91 datasets (idempotent)."""
    exp90_run.DRAWS = dict(DRAWS91)
    exp90_run.K_CONDS = dict(K91)
    exp90_run.write_new_json = _exp91_writer


install()


def unit_key(draw, arm, n, seed):
    return exp90_run.unit_key(draw, arm, "n%d" % n, seed)


def registered_grid():
    """6 learners x 10 doses x 12 seeds x 2 draws = 1,440 (doc 84 SS8)."""
    return [(d, a, n, s) for d in ("primary", "confirm") for n in LADDER
            for s in SEEDS for a in LEARNERS]


def shard(units, worker, nworkers):
    """Quantum and classical units dealt round-robin separately, so every
    worker gets an equal quantum load (exp90's rule)."""
    q = [u for u in units if u[1] in QUANTUM_ARMS]
    c = [u for u in units if u[1] not in QUANTUM_ARMS]
    return q[worker::nworkers] + c[worker::nworkers]


def run_unit(draw, arm, n, seed, max_epochs=MAX_EPOCHS, outdir=CKPT, quiet=False):
    install()
    if arm not in LEARNERS:
        raise ValueError("arm %r not registered for exp91" % arm)
    if n not in LADDER:
        raise ValueError("n=%r not on the ladder" % n)
    if max_epochs != MAX_EPOCHS and os.path.abspath(outdir) == os.path.abspath(CKPT):
        raise SystemExit("non-registered epochs must not write %s/" % CKPT)
    return exp90_run.run_unit(draw, arm, "n%d" % n, seed, max_epochs=max_epochs,
                              outdir=outdir, quiet=quiet)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draw", choices=sorted(DRAWS91))
    ap.add_argument("--arm", choices=LEARNERS)
    ap.add_argument("--n", type=int, choices=LADDER)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--outdir", default=CKPT)
    ap.add_argument("--draws", default="primary,confirm")
    ap.add_argument("--arms", default=",".join(LEARNERS))
    ap.add_argument("--ns", default=",".join(map(str, LADDER)))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = ap.parse_args()
    if args.max_epochs != MAX_EPOCHS and \
            os.path.abspath(args.outdir) == os.path.abspath(CKPT):
        raise SystemExit("non-registered epochs must not write %s/" % CKPT)
    if args.grid:
        sel = (set(args.draws.split(",")), set(args.arms.split(",")),
               {int(x) for x in args.ns.split(",")},
               {int(x) for x in args.seeds.split(",")})
        units = [u for u in registered_grid()
                 if u[0] in sel[0] and u[1] in sel[1] and u[2] in sel[2]
                 and u[3] in sel[3]]
        mine = shard(units, args.worker, args.nworkers)
        print("[exp91 w%d/%d] %d of %d units" % (args.worker, args.nworkers,
                                                 len(mine), len(units)), flush=True)
        for u in mine:
            base = os.path.join(args.outdir, unit_key(*u))
            if os.path.exists(base + ".json"):
                print("[skip] %s exists" % unit_key(*u), flush=True)
                continue
            if os.path.exists(base + ".pt"):
                print("[partial] removing orphan %s.pt (no JSON = no result)"
                      % unit_key(*u), flush=True)
                os.remove(base + ".pt")
            run_unit(*u, max_epochs=args.max_epochs, outdir=args.outdir)
        print("[exp91 w%d] done" % args.worker, flush=True)
        return 0
    if None in (args.draw, args.arm, args.n, args.seed):
        raise SystemExit("single-unit mode needs --draw --arm --n --seed")
    run_unit(args.draw, args.arm, args.n, args.seed,
             max_epochs=args.max_epochs, outdir=args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
