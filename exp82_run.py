# -*- coding: utf-8 -*-
"""exp82 driver: A3-T2 identity vs uniform, 15 fresh seeds each (82000..82014).

REGISTRATION: qnlp_private_docs/74_exp82_init_direction.md (REGISTERED 2026-09-10)

Design:
  - exp73 A3-T2 cell exactly (committed machinery, frozen Part A dataset stories_exp69_QA.json)
  - Arms: A3-T2-identity, A3-T2-uniform ONLY (no other arms)
  - Seeds: 82000..82014 (15 fresh seeds; original exp73 seeds 0-4 are NEVER touched)
  - Bounded-redraw (doc 62 SS24.3) on identity-init window:
      attempt-0 bit-identical to pre-amendment behaviour;
      stride 100000; 20-STOP; provenance recorded per run.

Smoke gate: reuses results_exp73_smoke_validation.json (exp73 committed artifact,
READ-ONLY -- this driver NEVER rewrites it).

Checkpoint dir: exp82_ckpt/  (separate from exp73_ckpt/ and all other dirs)
Logs: exp82_w{i}.log (launched externally with nohup / caffeinate)

Usage:
  python exp82_run.py --probe                   # 2-epoch wiring probe (no save)
  python exp82_run.py --worker i --nworkers 5   # production worker (i in 0..4)
"""
import argparse
import json
import os
import sys
import time
import traceback

import exp73_run as R73
import exp72_run as R72

SPEC = "qnlp_private_docs/74_exp82_init_direction.md (REGISTERED 2026-09-10)"

EXP82_CKPT = "exp82_ckpt"
SMOKE_FILE = "results_exp73_smoke_validation.json"  # committed artifact, read-only

ARMS = ["A3-T2-identity", "A3-T2-uniform"]
SEED_BASE = 82000
N_SEEDS_82 = 15
SEEDS = list(range(SEED_BASE, SEED_BASE + N_SEEDS_82))  # 82000..82014

# Original exp73 seeds 0-4: motivating record; NEVER touched or pooled
FORBIDDEN_SEEDS = set(range(5))

# doc 62 SS24.3 bounded-redraw parameters
REDRAW_MAX = 20
REDRAW_STRIDE = 100_000

# Save original apply_init_scheme BEFORE any rebinding (at module import time)
_ORIG_INIT = R73.apply_init_scheme


def apply_init_scheme_82(m, arm, seed):
    """Wraps exp73's apply_init_scheme with doc 62 SS24.3 bounded-redraw for identity.

    Uniform arm: delegates to _ORIG_INIT unchanged -- same output byte-for-byte.
    Identity arm: attempt k=0 is _ORIG_INIT(m, arm, seed), byte-identical to exp73's
    own behaviour. On AssertionError (sd window miss) stride by 100_000 and retry,
    up to REDRAW_MAX attempts. Exhaustion is a STOP (not a result). On success, adds
    'redraw_doc62_SS24_3' provenance key to the returned dict."""
    cfg = R73.ARM_CFG[R73.base_arm(arm)]
    if cfg["init"] == "uniform":
        return _ORIG_INIT(m, arm, seed)

    # identity arm: bounded-redraw (doc 62 SS24.3)
    refusals = []
    for k in range(REDRAW_MAX):
        eff = int(seed) + REDRAW_STRIDE * k
        try:
            rep = _ORIG_INIT(m, arm, eff)
            # success -- augment with redraw provenance
            rep["redraw_doc62_SS24_3"] = {
                "attempts_used": k + 1,
                "refused_attempts": refusals,
                "effective_seed_of_record": eff,
                "stride": REDRAW_STRIDE,
                "max_attempts": REDRAW_MAX,
                "note": ("attempt 0 is byte-identical to the pre-amendment behaviour; "
                         "stride 100000 per refusal, 20-STOP per doc 62 SS24.3"),
            }
            return rep
        except AssertionError as e:
            refusals.append({"attempt": k, "effective_seed": eff,
                             "refusal": str(e)})
    raise RuntimeError(
        "doc 62 SS24.3 STOP: %d consecutive identity draws refused for arm %s "
        "seed %d -- not sampling noise. Refusals: %r"
        % (REDRAW_MAX, arm, seed, refusals))


def check_smoke():
    """Verify exp73 smoke gate file is present and passed. Never rewrites it."""
    if not os.path.exists(SMOKE_FILE):
        sys.exit("FATAL: %s missing. "
                 "If re-running smoke: redirect with --smoke-out; "
                 "do NOT overwrite the committed artifact." % SMOKE_FILE)
    sm = json.load(open(SMOKE_FILE, encoding="utf-8"))
    if not sm.get("assertions_passed"):
        sys.exit("FATAL: %s: assertions_passed is not True -- gate not passed" % SMOKE_FILE)
    return sm


def job_list():
    """(arm, seed) pairs in stable order for modulo-partition dispatch."""
    return [(arm, s) for arm in ARMS for s in SEEDS]


def run_worker(worker, nworkers, max_epochs=None):
    sm = check_smoke()
    print("[exp82 w%d] smoke gate OK (%s)" % (worker, SMOKE_FILE), flush=True)

    # Rebind exp73_run module globals so train() uses exp82 dirs and init scheme
    R73.CKPT_DIR = EXP82_CKPT
    R73.apply_init_scheme = apply_init_scheme_82

    os.makedirs(EXP82_CKPT, exist_ok=True)

    store = R73.load_part()
    verbs = sorted(store[1]["verb_inventory"].keys())
    fm = R73.feature_maps()

    # Contract insurance: assert frozen Part A (doc 74 §1)
    _items, meta, _path = store
    assert meta["dataset_sha256"] == R73.DATASET_SHA_DOC61, (
        "exp82 is not running on exp73's frozen Part A -- "
        "sha mismatch: got %s, expected %s"
        % (meta["dataset_sha256"], R73.DATASET_SHA_DOC61))

    epochs = max_epochs if max_epochs is not None else R73.MAX_EPOCHS

    jobs = job_list()
    mine = [(arm, s) for i, (arm, s) in enumerate(jobs) if i % nworkers == worker]

    print("[exp82 w%d/%d] %d of %d runs  epochs=%d"
          % (worker, nworkers, len(mine), len(jobs), epochs), flush=True)

    for arm, seed in mine:
        assert seed not in FORBIDDEN_SEEDS, (
            "FORBIDDEN: seed %d is in the original exp73 seeds 0-4 (never touched)" % seed)
        ckj = os.path.join(EXP82_CKPT, "%s_s%d.json" % (arm, seed))
        if os.path.exists(ckj):
            print("[ckpt] %s s%d already done" % (arm, seed), flush=True)
            continue
        try:
            r = R73.train(arm, seed, store, verbs, epochs, fm, save=True, tag="")
            r["spec"] = SPEC
            r["exp82_seed_base"] = SEED_BASE
            r["exp82_seed_index"] = seed - SEED_BASE
            json.dump(r, open(ckj, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            rd = r.get("init_scheme", {}).get("redraw_doc62_SS24_3", {})
            print("[exp82] %-28s s%d  train=%.3f val=%.3f HELD-OUT=%.4f  "
                  "redraw=%s  (%.0fs)"
                  % (arm, seed, r["train_acc"], r["val_acc"],
                     r["heldout_test_acc"],
                     rd.get("attempts_used", "n/a"),
                     r["wall_sec"]), flush=True)
        except Exception as exc:
            print("[FAIL] %s s%d: %r" % (arm, seed, exc), flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()

    print("[exp82 w%d] done." % worker, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true",
                    help="2-epoch probe to verify driver wiring; no save")
    ap.add_argument("--worker", type=int, default=None,
                    help="Worker index (0..4)")
    ap.add_argument("--nworkers", type=int, default=5)
    a = ap.parse_args()

    if a.probe:
        sm = check_smoke()
        print("[probe] smoke gate OK", flush=True)
        # Rebind for probe
        R73.CKPT_DIR = EXP82_CKPT
        R73.apply_init_scheme = apply_init_scheme_82
        store = R73.load_part()
        verbs = sorted(store[1]["verb_inventory"].keys())
        fm = R73.feature_maps()
        _items, meta, _path = store
        assert meta["dataset_sha256"] == R73.DATASET_SHA_DOC61, "dataset SHA mismatch"
        print("[probe] dataset SHA ok (%s...)" % meta["dataset_sha256"][:16], flush=True)
        for arm in ARMS:
            seed = SEEDS[0]  # 82000
            r = R73.train(arm, seed, store, verbs, 2, fm, save=False, tag="")
            rd = r.get("init_scheme", {}).get("redraw_doc62_SS24_3", {})
            print("[probe] %-28s s%d  val=%.3f  "
                  "redraw_attempts=%s  eff_seed=%s"
                  % (arm, seed, r["val_acc"],
                     rd.get("attempts_used", "n/a (uniform)"),
                     rd.get("effective_seed_of_record", "n/a")), flush=True)
        print("[probe] OK -- wiring verified.", flush=True)
        return 0

    if a.worker is not None:
        run_worker(a.worker, a.nworkers)
        return 0

    print("Give --probe or --worker i", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
