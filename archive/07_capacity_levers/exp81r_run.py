# -*- coding: utf-8 -*-
"""exp81r_run.py -- doc 73 §7 (REGISTERED 665656a): exp81 A2q2 + A2q2-FLOOR CLEAN re-run.

exp81_run.py (code of record) is imported and NOT edited. Exactly two changes, explicit:
  (1) |z|^2 = re^2 + im^2 in sim.prob_all_zero_on (gate A81: the NaN source is
      AbsBackward0 in torch_sv_sim.prob_all_zero_on at s1 epoch 7 -- doc 83 §11.3 bug).
  (2) train_unit81's skip-on-nonfinite block -> RAISE (the unit aborts, recorded).
Gradient clipping at 5.0 is KEPT (applied to every exp81 arm; standing A2-WIDE/RETRAIN
comparable). QR unitarization + renorm backstop unchanged. Output: EXP81_CKPT_DIR=exp81r_ckpt.
Usage: python exp81r_run.py --arm A2q2|A2q2-FLOOR --seed S [--threads 2]
"""
import os, sys, ast, argparse, hashlib, json, time
os.environ.setdefault("EXP81_CKPT_DIR", "exp81r_ckpt")
import torch
import exp81_run as E


def _smooth_p0(state, wires):
    s = state
    for w in sorted(wires, reverse=True):
        s = s.select(w, 0)
    return (s.real ** 2 + s.imag ** 2).sum()


E.sim.prob_all_zero_on = _smooth_p0
# erratum doc 73 §7.1: keep renorm incidents out of the COMMITTED exp81_renorm_log.jsonl
_orig_flush = E.sim.flush_renorm_log
E.sim.flush_renorm_log = lambda path="exp81r_renorm_log.jsonl": _orig_flush(path)

SRC_PATH = E.__file__
SRC = open(SRC_PATH, encoding="utf-8").read()
fn = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "train_unit81")
orig_src = ast.get_source_segment(SRC, fn)
SKIP_BLOCK = """                nonfinite_skips.append({"epoch": epoch, "batch_start": start})
                opt.zero_grad()
                continue"""
RAISE_BLOCK = """                nonfinite_skips.append({"epoch": epoch, "batch_start": start})
                raise FloatingPointError("doc73 §7: non-finite gradient (unit aborts) at epoch %d batch %d" % (epoch, start))"""
assert orig_src.count(SKIP_BLOCK) == 1, "exp81 source changed; refuse"
exec(compile(orig_src.replace(SKIP_BLOCK, RAISE_BLOCK), SRC_PATH + "::train_unit81[doc73§7]", "exec"), E.__dict__)
ORIG_SHA = hashlib.sha256(orig_src.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A2q2", "A2q2-FLOOR"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    data = E.prepare()
    cfg = {"lr": 0.005, "batch": 1, "epochs": 30, "verbs": data["verbs"]}
    sm = None
    if a.arm == "A2q2-FLOOR":
        all_ids = sorted({cs.item_id for split in data["compiled"].values() for cs in split})
        sm = E.draw_fixed_scramble(all_ids, E.FLOOR_SCRAMBLE_SEED)
    os.makedirs(os.environ["EXP81_CKPT_DIR"], exist_ok=True)   # erratum doc 73 §7.1: job_main81 does not create it
    t0 = time.time(); status, err = "completed", None
    try:
        E.job_main81(a.arm, a.seed, data, cfg, scramble_map=sm)
    except FloatingPointError as e:
        status, err = "ABORTED_NONFINITE", str(e)
    os.makedirs(os.environ["EXP81_CKPT_DIR"], exist_ok=True)
    out = os.path.join(os.environ["EXP81_CKPT_DIR"], "%s_s%d_clean_meta.json" % (a.arm.replace("-", ""), a.seed))
    assert not os.path.exists(out), "Rule 3"
    json.dump({"doc": "73 §7 (665656a)", "arm": a.arm, "seed": a.seed, "status": status, "error": err,
               "threads": a.threads, "torch": torch.__version__, "wall_sec": time.time() - t0,
               "train_unit81_original_sha256": ORIG_SHA,
               "exp81_run_sha256": hashlib.sha256(open(SRC_PATH, "rb").read()).hexdigest(),
               "self_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()},
              open(out, "w"), indent=1)
    print("[exp81r %s s%d] %s %.0fs" % (a.arm, a.seed, status, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
