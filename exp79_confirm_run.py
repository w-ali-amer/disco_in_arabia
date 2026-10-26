# -*- coding: utf-8 -*-
"""exp79 CONFIRM driver -- doc 71 §10.2, ordered 2026-09-09.

Fresh generator draw at seed 43000 (stories_exp79_confirm.json, sha-asserted;
strong V-suite gating before any training).
Arms: C-DISCARD (exp79 §9.3 code as-is), A2, B1, A1 (the invalidating control).
5 seeds each. Separate dirs: exp79_confirm_ckpt/, exp79_confirm_ctl/,
exp79_confirm_locks/.
A2/B1/A1 RETRAINED on confirm draw via the committed exp42r machinery --
do NOT reuse first-draw exp42r_ckpt/ or exp79_ckpt/ checkpoints.
C1 dial-swap run on C-DISCARD (1000 perms, seed 4242, 5 shards).

Rebinding idiom: identical to exp42r_confirm_run.py. Six env vars set before
any import so every module-level constant in exp42r_common and exp79_run
resolves to the confirm dirs. Smoke-marker and gate-artifact globals in
exp79_run are rebound AFTER import (Python module-global lookup at call time).

Usage:
  python3 exp79_confirm_run.py --smoke
  python3 exp79_confirm_run.py --worker i --nworkers 5   # i = 0..4
"""

# ====================================================================
# ALL ENV VARS BEFORE ANY IMPORT -- this is the only safe order
# ====================================================================
import os, sys

CONF_CKPT  = "exp79_confirm_ckpt"
CONF_CTL   = "exp79_confirm_ctl"
CONF_LOCKS = "exp79_confirm_locks"

os.environ["EXP79_CKPT_DIR"]  = CONF_CKPT
os.environ["EXP79_CTL_DIR"]   = CONF_CTL
os.environ["EXP79_LOCK_DIR"]  = CONF_LOCKS
os.environ["EXP42R_CKPT_DIR"] = CONF_CKPT
os.environ["EXP42R_CTL_DIR"]  = CONF_CTL
os.environ["EXP42R_LOCK_DIR"] = CONF_LOCKS

# ====================================================================
# Dataset SHA assertion (before imports: fail fast if file missing)
# ====================================================================
CONFIRM_DATA = "stories_exp79_confirm.json"
FIRST_DRAW_SHA = (
    "2326942ffb6ccf6bca95a7ec0502c82563957fabb159b143608a9fb31f8f3113")

import json
try:
    _meta = json.load(open(CONFIRM_DATA, encoding="utf-8"))["meta"]
    CONFIRM_SHA = _meta.get("dataset_sha256") or _meta.get("sha256")
except FileNotFoundError:
    sys.exit("FATAL: %s not found -- run datagen (seed 43000) first."
             % CONFIRM_DATA)
except KeyError:
    sys.exit("FATAL: %s has no 'meta' key." % CONFIRM_DATA)

if CONFIRM_SHA is None:
    sys.exit("FATAL: no dataset_sha256 in %s meta." % CONFIRM_DATA)
if CONFIRM_SHA == FIRST_DRAW_SHA:
    sys.exit("FATAL: confirm draw reproduced the first draw -- STOP.")

print("[conf] dataset=%s  sha=%s..." % (CONFIRM_DATA, CONFIRM_SHA[:16]),
      flush=True)

# ====================================================================
# Imports (after env vars so module-level consts resolve correctly)
# ====================================================================
import argparse
import copy
import math
import random
import time
import traceback

import torch

import exp42r_common as K42
import exp42r_run as R42
import exp79_run as R79
from exp79_run import (
    make_model79, unit_paths79, job_main79,
    N_SEEDS,
)

# ====================================================================
# Rebind exp42r_common data pins (same idiom as exp42r_confirm_run.py)
# ====================================================================
K42.DATA_PATH = CONFIRM_DATA
K42.DATASET_SHA = CONFIRM_SHA
K42.load_dataset.__defaults__ = (CONFIRM_DATA,)
K42.provenance_gate.__defaults__ = (CONFIRM_DATA, True)

# ====================================================================
# Rebind exp79_run artifact names -- avoids overwriting first-draw
# gate artifacts (doc 70 §3.6: each run's artifacts are its own)
# ====================================================================
R79.SMOKE_MARKER = "exp79_confirm_smoke_PASSED.json"
R79.SMOKE_REPORT = "exp79_confirm_smoke.json"
R79.SG_FS_OUT    = "exp79_confirm_gate_SG_FS.json"
R79.SG_ANT_OUT   = "exp79_confirm_gate_SG_ANT.json"
R79.SG_PS_OUT    = "exp79_confirm_gate_SG_PS.json"
R79.ANCHOR_OUT   = "exp79_confirm_anchor.json"   # not used; name kept clean

SMOKE_MARKER = R79.SMOKE_MARKER    # local alias used in run_worker()

# Arms in this confirm run
CONFIRM_ARMS_EXP79 = ["C-DISCARD"]
CONFIRM_ARMS_EXP42R = ["A2", "B1", "A1"]
N_SEEDS_CONFIRM = N_SEEDS          # 5

# C1 parameters (same as exp42r's registered C1 draw)
C1_N_PERM    = 1000
C1_PERM_SEED = 4242
C1_N_SHARDS  = 5


# ====================================================================
# Settled / failure helpers (confirm-aware)
# ====================================================================
# Note: all six dirs are redirected via env vars at import time, so:
#   K42.LOCK_DIR  == CONF_LOCKS  (claim uses this)
#   K42.CTL_DIR   == CONF_CTL    (settled / record_failure for exp42r arms)
#   R79.CTL_DIR   == CONF_CTL    (settled79 / record_failure79 for C-DISCARD)
# We can use R42.claim, R42.settled, R79.settled79 directly.

def _cdiscard_done(seed):
    jp, _ = unit_paths79("C-DISCARD", seed, CONF_CKPT)
    return os.path.exists(jp)


# ====================================================================
# C1 dial-swap on C-DISCARD (confirm data)
# ====================================================================
def _c1_perms(verbs):
    """1000 non-identity permutations of verbs, seed 4242 (same draw as exp42r)."""
    rng = random.Random(C1_PERM_SEED)
    out = []
    for _ in range(C1_N_PERM):
        p = verbs[:]
        while True:
            rng.shuffle(p)
            if p != verbs:
                break
        out.append(list(p))
    return out


def _load_cdiscard_models(verbs):
    """Load all 5 trained C-DISCARD models from exp79_confirm_ckpt/."""
    models = []
    for s in range(N_SEEDS_CONFIRM):
        json_path, pt_path = unit_paths79("C-DISCARD", s, CONF_CKPT)
        if not (os.path.exists(json_path) and os.path.exists(pt_path)):
            raise RuntimeError(
                "C-DISCARD s%d not found in %s -- not trained yet?" % (s, CONF_CKPT))
        m = make_model79("C-DISCARD", verbs)
        m.load_state_dict(torch.load(pt_path, weights_only=True))
        m.eval()
        models.append(m)
    return models


def job_c1_shard_confirm(shard, data):
    """C1 dial-swap on trained C-DISCARD models with confirm data.
    Writes exp79_confirm_ctl/c1_confirm_shard{i}.json.
    """
    out_path = os.path.join(CONF_CTL, "c1_confirm_shard%d.json" % shard)
    if os.path.exists(out_path):
        print("[c1_confirm shard %d] already done, skipping." % shard, flush=True)
        return

    verbs = data["verbs"]
    models = _load_cdiscard_models(verbs)
    union = data["compiled"]["test_union"]

    def seed_mean_acc(vmap=None):
        total = 0.0
        for m in models:
            with torch.no_grad():
                correct = sum(
                    1 if m.predict(cs, verb_map=vmap) == cs.answer else 0
                    for cs in union)
            total += correct / len(union)
        return total / len(models)

    perms = _c1_perms(verbs)
    lo = shard * (C1_N_PERM // C1_N_SHARDS)
    hi = ((shard + 1) * (C1_N_PERM // C1_N_SHARDS)
          if shard < C1_N_SHARDS - 1 else C1_N_PERM)

    orig = seed_mean_acc()
    t0 = time.time()
    perm_accs = []
    for i in range(lo, hi):
        vmap = dict(zip(verbs, perms[i]))
        perm_accs.append(seed_mean_acc(vmap))
        done = i - lo + 1
        if done % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / done * (hi - lo - done)
            print("[c1_confirm shard %d] %d/%d  eta %.0fs"
                  % (shard, done, hi - lo, eta), flush=True)

    out = {
        "control": "C1 trained-dial-swap (doc 22 SS6, exp79 confirm draw)",
        "arm": "C-DISCARD",
        "confirm_dataset_sha256": CONFIRM_SHA,
        "shard": shard, "n_shards": C1_N_SHARDS,
        "perm_index_lo": lo, "perm_index_hi": hi,
        "n_permutations_this_shard": len(perm_accs),
        "n_permutations_total": C1_N_PERM,
        "perm_seed": C1_PERM_SEED,
        "n_seeds": len(models),
        "n_union_items": len(union),
        "original_seed_mean_acc": orig,
        "permuted_seed_mean_accs": perm_accs,
        "wall_time_sec": round(time.time() - t0, 1),
        "note": ("C1 on C-DISCARD (confirm). Seed mean acc for each of 1000 "
                 "non-identity verb->block permutations applied uniformly to "
                 "all 5 seeds; original must exceed 95th percentile."),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("[c1_confirm shard %d] done: orig=%.4f n_perms=%d (%.0fs)"
          % (shard, orig, len(perm_accs), out["wall_time_sec"]), flush=True)


# ====================================================================
# Worker
# ====================================================================
def run_worker(w, args, data, cfg):
    if not os.path.exists(SMOKE_MARKER):
        sys.exit("[w%d] REFUSING: %s absent -- run --smoke first."
                 % (w, SMOKE_MARKER))

    torch.set_num_threads(args.threads)
    K42.ensure_dirs(CONF_CKPT, CONF_CTL, CONF_LOCKS)
    print("[conf w%d] starting (%d threads)" % (w, args.threads), flush=True)

    # ---- Phase 1 training jobs ----------------------------------
    # C-DISCARD via exp79 machinery (§9.3 code as-is)
    for s in range(N_SEEDS_CONFIRM):
        jid = "CONFIRM_CDISCARD:s%d" % s
        outs = list(unit_paths79("C-DISCARD", s, CONF_CKPT))
        if R79.settled79(outs, jid) or not R42.claim(jid):
            continue
        try:
            job_main79("C-DISCARD", s, data, cfg)
        except Exception as exc:
            payload = {
                "job": jid, "tag": "C-DISCARD",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "isolation": "arm-level failure; worker continues",
            }
            with open(os.path.join(CONF_CTL, "FAILED_%s.json" % jid), "w",
                      encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
            print("[FAIL] %s: %r (isolated)" % (jid, exc), flush=True)

    # A2, B1, A1 via exp42r machinery -- RETRAINED on confirm draw
    for arm in CONFIRM_ARMS_EXP42R:
        for s in range(N_SEEDS_CONFIRM):
            jid = "CONFIRM_%s:s%d" % (arm, s)
            outs = list(K42.unit_paths(arm, s))
            if R42.settled(outs, jid) or not R42.claim(jid):
                continue
            try:
                R42.job_main(arm, s, data, cfg)
            except Exception as exc:
                R42.record_failure(jid, exc, arm)

    # ---- Phase 2 C1 shards (requires all C-DISCARD done) --------
    all_cdiscard_done = all(_cdiscard_done(s) for s in range(N_SEEDS_CONFIRM))
    if not all_cdiscard_done:
        print("[conf w%d] C-DISCARD not all complete; skipping C1 shards "
              "this pass." % w, flush=True)
    else:
        for shard in range(C1_N_SHARDS):
            jid = "CONFIRM_c1_shard%d" % shard
            out_path = os.path.join(CONF_CTL, "c1_confirm_shard%d.json" % shard)
            fail_path = os.path.join(CONF_CTL, "FAILED_%s.json" % jid)
            if os.path.exists(out_path) or os.path.exists(fail_path):
                continue
            if not R42.claim(jid):
                continue
            try:
                job_c1_shard_confirm(shard, data)
            except Exception as exc:
                payload = {
                    "job": jid, "tag": "c1_shard%d" % shard,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "isolation": "arm-level failure; worker continues",
                }
                with open(fail_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=1)
                print("[FAIL] %s: %r (isolated)" % (jid, exc), flush=True)

    print("[conf w%d] done." % w, flush=True)


# ====================================================================
# Main
# ====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke",  action="store_true",
                    help="Run smoke battery on confirm data and write marker")
    ap.add_argument("--worker",  type=int, default=None,
                    help="Worker index (0-4)")
    ap.add_argument("--nworkers", type=int, default=5)
    ap.add_argument("--epochs",  type=int, default=30)
    ap.add_argument("--lr",      type=float, default=0.005)
    ap.add_argument("--batch",   type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    K42.ensure_dirs(CONF_CKPT, CONF_CTL, CONF_LOCKS)

    # provenance gate + compile (uses rebound DATA_PATH / DATASET_SHA)
    data = R42.prepare()

    cfg = {"verbs": data["verbs"], "lr": args.lr,
           "batch": args.batch, "epochs": args.epochs}

    if args.smoke:
        # Smoke battery on confirm data with confirm-specific artifact names.
        # No drift anchor (comparing committed A2 s0 against a different dataset
        # is meaningless; the retrained A2 is the confirm's anchor).
        return R79.smoke_battery(data)

    if args.worker is not None:
        run_worker(args.worker, args, data, cfg)
        return 0

    print("Give --smoke or --worker i", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
