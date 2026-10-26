# -*- coding: utf-8 -*-
"""exp82c driver: magnitude-confirm for A3-T2 identity vs uniform (doc 74 §4.1).

REGISTRATION: qnlp_private_docs/74_exp82_init_direction.md §4.1
Ordered 2026-09-10 after BOTH exp82 booleans fired.

Design (TWO changes from exp82_run.py only):
  1. Dataset: stories_exp70_confirm_QA.json (seed-70000 confirm draw,
     internal meta sha 59e6ada8...; same file exp73_confirm_run.py used).
  2. Seed base: 82500 (seeds 82500..82514); exp73 seeds 0-4 and exp82
     seeds 82000-82014 are both NEVER touched or pooled.

Everything else is exp82_run.py byte-for-byte:
  - Arms: A3-T2-identity vs A3-T2-uniform
  - exp73 machinery unforked
  - Bounded-redraw (doc 62 SS24.3) on identity-init window:
      attempt-0 bit-identical; stride 100000; 20-STOP; provenance recorded
  - 30 runs, ~6h on 5 workers, detached (exp82c_w*.log)

Dataset patching approach (verbatim from exp73_confirm_run.py):
  R73.R69.DATA_A, R73.DATA_A, R73.DATASET_SHA_DOC61,
  R73.VSUITE, R73.check_dataset -- all rebound before load_part().

Smoke gate: results_exp73_confirm_smoke.json (confirm draw's own machinery
gate, assertions_passed=True, READ-ONLY -- never rewritten).

Checkpoint dir: exp82c_ckpt/  (separate from all other ckpt dirs)
Logs: exp82c_w{i}.log (launched externally with nohup / caffeinate)

Usage:
  python exp82c_run.py --probe                   # 2-epoch wiring probe (no save)
  python exp82c_run.py --worker i --nworkers 5   # production worker (i in 0..4)
"""
import argparse
import json
import os
import sys
import time
import traceback

import exp73_run as R73
import exp72_run as R72

SPEC = "qnlp_private_docs/74_exp82_init_direction.md §4.1 (magnitude-confirm, 2026-09-10)"

EXP82C_CKPT = "exp82c_ckpt"
SMOKE_FILE = "results_exp73_confirm_smoke.json"  # confirm draw's own gate, READ-ONLY

CONFIRM_DATA = "stories_exp70_confirm_QA.json"
CONFIRM_SHA = ("59e6ada8a98b8917a1bd8e927d09249132a9765f7002e40fceb807a4bd51"
               "cd7c")
FIRST_DRAW_SHA = ("2b85822b544571969fd40cab12b03a4ac256b08aee914d2b9643b856"
                  "a6308c96")
CONFIRM_VSUITE = "results_exp70_confirm_WN_validation.json"

ARMS = ["A3-T2-identity", "A3-T2-uniform"]
SEED_BASE = 82500
N_SEEDS_82C = 15
SEEDS = list(range(SEED_BASE, SEED_BASE + N_SEEDS_82C))  # 82500..82514

# Forbidden: original exp73 seeds 0-4 (motivating record) and
# exp82 seeds 82000-82014 (prospective-confirm draw, now prior record)
FORBIDDEN_SEEDS = set(range(5)) | set(range(82000, 82015))

# doc 62 SS24.3 bounded-redraw parameters (identical to exp82_run.py)
REDRAW_MAX = 20
REDRAW_STRIDE = 100_000

# Capture original apply_init_scheme BEFORE any rebinding (at module import time)
_ORIG_INIT = R73.apply_init_scheme


def apply_init_scheme_82c(m, arm, seed):
    """Wraps exp73's apply_init_scheme with doc 62 SS24.3 bounded-redraw for identity.

    Uniform arm: delegates to _ORIG_INIT unchanged -- same output byte-for-byte.
    Identity arm: attempt k=0 is _ORIG_INIT(m, arm, seed), byte-identical to exp73's
    own behaviour. On AssertionError (sd window miss) stride by 100_000 and retry,
    up to REDRAW_MAX attempts. Exhaustion is a STOP (not a result). On success, adds
    'redraw_doc62_SS24_3' provenance key to the returned dict.

    Identical to exp82_run.py's apply_init_scheme_82 -- only the function name differs."""
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


def make_confirm_check_dataset(R):
    """Verbatim from exp73_confirm_run.py: confirm variant of the SS7.3 dataset gate."""
    def confirm_check_dataset(store, rep):
        _items, meta, path = store
        assert meta["dataset_sha256"] == CONFIRM_SHA, \
            ("exp82c confirm dataset sha mismatch: %s vs the registered %s"
             % (meta["dataset_sha256"], CONFIRM_SHA))
        assert meta["dataset_sha256"] != FIRST_DRAW_SHA, \
            "the confirm draw reproduced the FIRST draw -- it is not fresh"
        vs = json.load(open(CONFIRM_VSUITE, encoding="utf-8"))
        assert vs.get("kill_gate_passed") is True, \
            "the confirm draw's V-suite kill gate did not pass"
        rep["dataset"] = {
            "path": path, "sha256": meta["dataset_sha256"],
            "registered_sha_doc74_S4_1": CONFIRM_SHA,
            "is_confirm_draw": True, "generator_seed": meta.get("seed"),
            "differs_from_first_draw": True,
            "vsuite_source": CONFIRM_VSUITE, "vsuite_kill_gate_passed": True,
            "vsuite_status": ("INHERITED from exp72's confirm on the strength "
                              "of byte identity (same sha) -- doc 61 SS21.3 / doc 74 §4.1"),
        }
        return rep
    return confirm_check_dataset


def _patch_r73():
    """Apply all confirm-draw bindings to exp73_run module globals.
    Called before load_part() in both probe and run_worker.
    Verbatim approach from exp73_confirm_run.py."""
    R73.R69.DATA_A = CONFIRM_DATA          # guard at line 1904 compares to R69.DATA_A
    R73.DATA_A = CONFIRM_DATA              # load_part() reads this module-global directly
    R73.DATASET_SHA_DOC61 = CONFIRM_SHA    # SS7.3 sha assertion checks this
    R73.VSUITE = CONFIRM_VSUITE            # V-suite binding
    R73.check_dataset = make_confirm_check_dataset(R73)  # confirm variant
    R73.CKPT_DIR = EXP82C_CKPT
    R73.apply_init_scheme = apply_init_scheme_82c


def check_smoke():
    """Verify confirm smoke gate file is present and passed. Never rewrites it."""
    if not os.path.exists(SMOKE_FILE):
        sys.exit("FATAL: %s missing. This is the confirm draw's machinery gate, "
                 "produced by exp73_confirm_run.py. STOP." % SMOKE_FILE)
    sm = json.load(open(SMOKE_FILE, encoding="utf-8"))
    if not sm.get("assertions_passed"):
        sys.exit("FATAL: %s: assertions_passed is not True -- gate not passed" % SMOKE_FILE)
    return sm


def job_list():
    """(arm, seed) pairs in stable order for modulo-partition dispatch."""
    return [(arm, s) for arm in ARMS for s in SEEDS]


def run_worker(worker, nworkers, max_epochs=None):
    sm = check_smoke()
    print("[exp82c w%d] smoke gate OK (%s)" % (worker, SMOKE_FILE), flush=True)

    # Pre-flight: assert confirm dataset sha on the raw file
    raw_meta = json.load(open(CONFIRM_DATA, encoding="utf-8"))["meta"]
    assert raw_meta["dataset_sha256"] == CONFIRM_SHA, (
        "exp82c pre-flight STOP: confirm dataset sha mismatch -- got %r"
        % raw_meta["dataset_sha256"])
    print("[exp82c w%d] confirm dataset sha OK (%s...)"
          % (worker, CONFIRM_SHA[:16]), flush=True)

    # Apply all confirm-draw + exp82c bindings
    _patch_r73()

    os.makedirs(EXP82C_CKPT, exist_ok=True)

    store = R73.load_part()
    verbs = sorted(store[1]["verb_inventory"].keys())
    fm = R73.feature_maps()

    # Contract insurance: assert confirm sha via rebounded constant
    _items, meta, _path = store
    assert meta["dataset_sha256"] == CONFIRM_SHA, (
        "exp82c is not running on the confirm draw -- "
        "sha mismatch: got %s, expected %s"
        % (meta["dataset_sha256"], CONFIRM_SHA))

    epochs = max_epochs if max_epochs is not None else R73.MAX_EPOCHS

    jobs = job_list()
    mine = [(arm, s) for i, (arm, s) in enumerate(jobs) if i % nworkers == worker]

    print("[exp82c w%d/%d] %d of %d runs  epochs=%d  dataset=%s"
          % (worker, nworkers, len(mine), len(jobs), epochs, CONFIRM_DATA),
          flush=True)

    for arm, seed in mine:
        assert seed not in FORBIDDEN_SEEDS, (
            "FORBIDDEN: seed %d is in exp73 0-4 or exp82 82000-82014 "
            "(never touched or pooled)" % seed)
        ckj = os.path.join(EXP82C_CKPT, "%s_s%d.json" % (arm, seed))
        if os.path.exists(ckj):
            print("[ckpt] %s s%d already done" % (arm, seed), flush=True)
            continue
        try:
            r = R73.train(arm, seed, store, verbs, epochs, fm, save=True, tag="")
            r["spec"] = SPEC
            r["exp82c_seed_base"] = SEED_BASE
            r["exp82c_seed_index"] = seed - SEED_BASE
            json.dump(r, open(ckj, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            rd = r.get("init_scheme", {}).get("redraw_doc62_SS24_3", {})
            print("[exp82c] %-28s s%d  train=%.3f val=%.3f HELD-OUT=%.4f  "
                  "redraw=%s  (%.0fs)"
                  % (arm, seed, r["train_acc"], r["val_acc"],
                     r["heldout_test_acc"],
                     rd.get("attempts_used", "n/a"),
                     r["wall_sec"]), flush=True)
        except Exception as exc:
            print("[FAIL] %s s%d: %r" % (arm, seed, exc), flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()

    print("[exp82c w%d] done." % worker, flush=True)


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

        # Pre-flight sha assertion
        raw_meta = json.load(open(CONFIRM_DATA, encoding="utf-8"))["meta"]
        assert raw_meta["dataset_sha256"] == CONFIRM_SHA, (
            "probe STOP: confirm dataset sha mismatch -- got %r"
            % raw_meta["dataset_sha256"])
        print("[probe] confirm dataset sha OK (%s...)" % CONFIRM_SHA[:16], flush=True)

        # Apply all bindings
        _patch_r73()

        store = R73.load_part()
        verbs = sorted(store[1]["verb_inventory"].keys())
        fm = R73.feature_maps()

        _items, meta, _path = store
        assert meta["dataset_sha256"] == CONFIRM_SHA, "post-load sha mismatch"
        print("[probe] post-load dataset SHA ok (%s...)"
              % meta["dataset_sha256"][:16], flush=True)

        for arm in ARMS:
            seed = SEEDS[0]  # 82500
            r = R73.train(arm, seed, store, verbs, 2, fm, save=False, tag="")
            rd = r.get("init_scheme", {}).get("redraw_doc62_SS24_3", {})
            print("[probe] %-28s s%d  val=%.3f  "
                  "redraw_attempts=%s  eff_seed=%s"
                  % (arm, seed, r["val_acc"],
                     rd.get("attempts_used", "n/a (uniform)"),
                     rd.get("effective_seed_of_record", "n/a")), flush=True)
        print("[probe] OK -- confirm-draw wiring verified.", flush=True)
        return 0

    if a.worker is not None:
        run_worker(a.worker, a.nworkers)
        return 0

    print("Give --probe or --worker i", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
