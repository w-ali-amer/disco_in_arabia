# -*- coding: utf-8 -*-
"""exp79r_run.py -- doc 71 §12 (REGISTERED e790627): exp79 C-RESET CLEAN re-run.

exp79_run.py (code of record) is imported and NOT edited. Changes, all explicit here:
  (iii) |z|^2 = re^2 + im^2 in _prob_all_zero_on_unnorm and sim.prob_all_zero_on
        (identical maths; doc 83 §11.3: complex64 abs() backward -> NaN at subnormals).
  (iv)  train_unit79's §9.3 skip-on-nonfinite block is REPLACED by a raise (the seed
        ABORTS and is recorded), and the §9.3 grad-clipping line is REMOVED.
  §9 scale normalization and the §9.2 1e-6 clamp (inside story_loss) are unchanged.
Instrumentation (descriptive): per epoch, min |amplitude| over readout inputs and the
count of subnormal (< 1.1754944e-38, nonzero) amplitudes.
Output: EXP79_CKPT_DIR=exp79r_ckpt (originals untouched).
Usage: python exp79r_run.py --seed S [--threads 1]
"""
import os, sys, ast, argparse, hashlib, json, math, time
os.environ.setdefault("EXP79_CKPT_DIR", "exp79r_ckpt")
import torch
import exp79_run as E

FLT_MIN = 1.1754944e-38
STATS = {"calls": 0, "min_abs": float("inf"), "n_subnormal": 0}
EPOCH_STATS = []


def _smooth_p0(state, wires):
    s = state
    for w in sorted(wires, reverse=True):
        s = s.select(w, 0)
    with torch.no_grad():
        a = s.detach().abs()
        nz = a[a > 0]
        if nz.numel():
            STATS["min_abs"] = min(STATS["min_abs"], float(nz.min()))
        STATS["n_subnormal"] += int(((a > 0) & (a < FLT_MIN)).sum())
        STATS["calls"] += 1
    return (s.real ** 2 + s.imag ** 2).sum()


E._prob_all_zero_on_unnorm = _smooth_p0
E.sim.prob_all_zero_on = _smooth_p0

# --- rebuild train_unit79 with skip->raise and clip removed (explicit source edit)
SRC_PATH = E.__file__
SRC = open(SRC_PATH, encoding="utf-8").read()
fn = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "train_unit79")
orig_src = ast.get_source_segment(SRC, fn)
SKIP_BLOCK = """                nonfinite_skips.append({"epoch": epoch, "batch_start": start})
                opt.zero_grad()
                continue"""
RAISE_BLOCK = """                nonfinite_skips.append({"epoch": epoch, "batch_start": start})
                raise FloatingPointError("doc71 §12: non-finite gradient (seed aborts) at epoch %d batch %d" % (epoch, start))"""
CLIP_LINE = "            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)\n"
assert orig_src.count(SKIP_BLOCK) == 1 and orig_src.count(CLIP_LINE) == 1, "exp79 source changed; refuse"
new_src = orig_src.replace(SKIP_BLOCK, RAISE_BLOCK).replace(CLIP_LINE, "")
exec(compile(new_src, SRC_PATH + "::train_unit79[doc71§12]", "exec"), E.__dict__)
E.__dict__["train_unit79_ORIGINAL_SHA256"] = hashlib.sha256(orig_src.encode()).hexdigest()

# --- per-epoch instrumentation: snapshot STATS whenever validation evaluate() runs
_orig_eval = E.evaluate
def _eval_snap(model, items, *a, **k):
    EPOCH_STATS.append(dict(STATS)); STATS.update(calls=0, min_abs=float("inf"), n_subnormal=0)
    return _orig_eval(model, items, *a, **k)
E.evaluate = _eval_snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    data = E.prepare()
    cfg = {"verbs": data["verbs"], "lr": 0.005, "batch": 1, "epochs": 30}
    t0 = time.time(); status = "completed"; err = None
    os.makedirs(os.environ["EXP79_CKPT_DIR"], exist_ok=True)   # erratum doc 71 §12.3: job_main79 does not create it
    try:
        E.job_main79("C-RESET", args.seed, data, cfg)
    except FloatingPointError as e:
        status, err = "ABORTED_NONFINITE", str(e)
    rec = {"doc": "71 §12 (e790627)", "seed": args.seed, "status": status, "error": err,
           "threads": args.threads, "torch": torch.__version__, "platform": sys.platform,
           "wall_sec": time.time() - t0,
           "train_unit79_original_sha256": E.__dict__["train_unit79_ORIGINAL_SHA256"],
           "exp79_run_sha256": hashlib.sha256(open(SRC_PATH, "rb").read()).hexdigest(),
           "self_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
           "epoch_readout_stats": EPOCH_STATS}
    os.makedirs(os.environ["EXP79_CKPT_DIR"], exist_ok=True)
    out = os.path.join(os.environ["EXP79_CKPT_DIR"], "CRESET_s%d_clean_meta.json" % args.seed)
    assert not os.path.exists(out), "Rule 3"
    json.dump(rec, open(out, "w"), indent=1)
    print("[exp79r s%d] %s  %.0fs" % (args.seed, status, rec["wall_sec"]), flush=True)


if __name__ == "__main__":
    main()
