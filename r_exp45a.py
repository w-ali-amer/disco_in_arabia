# -*- coding: utf-8 -*-
"""r_exp45a.py -- doc 25 SSR.4(a): exp45a on family-r. Decides
`bridge_representable_2q_r` (train >= 0.99 certifies). Protocol verbatim from
exp45a_representability.py: 40 restarts x Adam lr 0.02, <= 500 epochs,
batch 1, seeds 45000 + r, L-BFGS polish (300 iterations) of the top 3.

WRAPS exp45a_representability.py (never edited). Rebinds:
  DATA / EXPECTED_FAMILY_SHA256  -> stories_exp43a_familyr.json / its pin
  STATE_FMT / REC_FMT / SENTINEL -> exp45a_r_ckpt/exp45a_r_*
  glob (module attribute)        -> a shim that maps the runner's hard-coded
                                    finalize pattern "exp45a_restart_r*.json"
                                    to exp45a_r_ckpt/exp45a_r_restart_r*.json
                                    (any other pattern raises)
  fresh_results_path             -> results_exp45a_r (Rule 3)
  train_restart / lbfgs_polish   -> GUARDED COPIES (below). The originals have
                                    no isfinite check; doc 25 SSR.4 ADDS one.
                                    The copies are line-for-line the committed
                                    functions (source sha256 pinned and
                                    asserted at import) plus the guard lines
                                    marked `# [r] guard`.
Readout: smooth |z|^2 (r_common.patch_readout).

NON-FINITE POLICY (doc 25 SSR.4): a non-finite loss or grad norm ABORTS that
restart (or that polish); it is RECORDED (aborted_nonfinite + detail in the
restart record / r_wrapper block), never skipped over. An aborted restart
keeps its best-so-far FINITE state and measured train accuracy, and stays in
the ranking like any other restart (see FLAGS in the build report). An
aborted polish reports train_acc_after_lbfgs = -1.0 (a sentinel, not a
measurement) and leaves the Adam state untouched.

Usage (sharded; any number of shards, then one finalize):
  python r_exp45a.py --shard 0 --nshards 4      (alias: --worker)
  ...
  python r_exp45a.py --finalize
  python r_exp45a.py --smoke    # 1 restart x 5 epochs + finalize (3 L-BFGS
                                # iterations) -> r_smoke/
"""

import argparse
import glob as _glob
import json
import math
import os
import random
import sys
import time
import types

import torch

import r_common as rc

PATCH = rc.patch_readout()
import exp45a_representability as m                          # noqa: E402
from exp42_run import grad_global_norm                        # noqa: E402
from exp45_model import TwoQubitReferentModel                 # noqa: E402

PINNED_SRC = {
    "train_restart":
        "133e9d3de1ede3801d4d351dfac49668fc560f13f2abd02dd13bf04aefb65f7a",
    "lbfgs_polish":
        "29fc3ed7be937668656250c27bde6d0472d2588b1b85e292b341c77e75c53293",
}
assert rc.source_sha256(m.train_restart) == PINNED_SRC["train_restart"], \
    "exp45a train_restart changed since the guarded copy was made"
assert rc.source_sha256(m.lbfgs_polish) == PINNED_SRC["lbfgs_polish"], \
    "exp45a lbfgs_polish changed since the guarded copy was made"
ORIG_SRC_SHA = {"exp45a_representability.py":
                rc.file_sha256("exp45a_representability.py"),
                "exp45_model.py": rc.file_sha256("exp45_model.py")}

POLISH_NONFINITE = []


class _NonFinite(Exception):
    pass


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------ guarded copies
def train_restart(r, compiled, verbs, max_epochs):
    rc.assert_patched()                                       # [r] guard
    seed = m.RNG_BASE + r
    model = TwoQubitReferentModel("A4", verbs, seed)
    opt = torch.optim.Adam(model.parameters(), lr=m.LR)
    rng = random.Random(seed)
    order = list(range(len(compiled)))
    best_acc, best_epoch, best_state = -1.0, -1, None
    t0 = time.time()
    epochs_run, aborted = 0, False
    nonfinite = None                                          # [r] guard
    for ep in range(max_epochs):
        epochs_run = ep + 1
        rng.shuffle(order)
        tot = 0.0
        for step, i in enumerate(order):
            loss, _p = model.story_loss(compiled[i])
            opt.zero_grad()
            loss.backward()
            gn = grad_global_norm(model)                      # [r] guard
            lv = loss.item()                                  # [r] guard
            if not (math.isfinite(gn) and math.isfinite(lv)):  # [r] guard
                nonfinite = {"epoch": ep, "step_in_epoch": step,  # [r] guard
                             "item_index": i, "loss": repr(lv),   # [r] guard
                             "grad_norm": repr(gn)}               # [r] guard
                break                                         # [r] guard
            opt.step()
            tot += loss.item()
        if nonfinite is not None:                             # [r] guard
            log("[r%02d] NON-FINITE at ep%03d -- restart ABORTED and recorded"
                " (%s)" % (r, ep, nonfinite))                 # [r] guard
            break                                             # [r] guard
        acc = m.fast_eval(model, compiled)
        if acc > best_acc:
            best_acc, best_epoch = acc, ep
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if ep % 10 == 0 or acc >= m.TARGET:
            log("[r%02d ep%03d] train_acc=%.4f best=%.4f loss=%.4f (%.1fs)"
                % (r, ep, acc, best_acc, tot / len(order), time.time() - t0))
        if best_acc >= m.TARGET:
            open(m.SENTINEL, "w").write("restart %d\n" % r)
            break
        if m.certificate_seen():
            aborted = True
            log("[r%02d] aborting at ep%03d -- certificate sentinel seen"
                % (r, ep))
            break
    rec = {"restart": r, "seed": seed, "best_train_acc_adam": best_acc,
           "best_epoch": best_epoch, "epochs_run": epochs_run,
           "aborted_on_certificate": aborted,
           "aborted_nonfinite": nonfinite is not None,        # [r] guard
           "nonfinite_detail": nonfinite,                     # [r] guard
           "wall_sec": round(time.time() - t0, 1)}
    with open(m.REC_FMT % r, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    torch.save({"state_dict": best_state, "restart": r, "stage": "adam",
                "best_train_acc": best_acc}, m.STATE_FMT % r)
    return rec


def lbfgs_polish(state, compiled, verbs, r, max_iter=None):
    max_iter = m.LBFGS_MAX_ITER if max_iter is None else max_iter
    if state is None:                                         # [r] guard
        POLISH_NONFINITE.append({"restart": r, "reason": "no finite Adam "
                                 "state (aborted before first eval)"})
        return -1.0, float("nan"), None                       # [r] guard
    model = TwoQubitReferentModel("A4", verbs, m.RNG_BASE + r)
    model.load_state_dict(state)
    opt = torch.optim.LBFGS(model.parameters(), max_iter=max_iter,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        total = 0.0
        for cs in compiled:
            loss, _p = model.story_loss(cs)
            total = total + loss
        total = total / len(compiled)
        total.backward()
        gn = grad_global_norm(model)                          # [r] guard
        if not (math.isfinite(gn) and math.isfinite(total.item())):  # [r]
            raise _NonFinite({"loss": repr(total.item()),     # [r] guard
                              "grad_norm": repr(gn)})         # [r] guard
        return total

    try:                                                      # [r] guard
        lval = opt.step(closure)
    except _NonFinite as e:                                   # [r] guard
        POLISH_NONFINITE.append({"restart": r, "detail": e.args[0]})
        log("[r%02d] NON-FINITE in L-BFGS polish -- polish ABORTED and "
            "recorded; Adam state kept" % r)
        return -1.0, float("nan"), None                       # [r] guard
    acc = m.fast_eval(model, compiled)
    return acc, float(lval), {k: v.detach().clone()
                              for k, v in model.state_dict().items()}


# ------------------------------------------------------------------ wiring
def install(outdir, smoke):
    ck = os.path.join(outdir, "exp45a_r_ckpt")
    os.makedirs(ck, exist_ok=True)
    m.DATA = rc.FAMR_DATA
    m.EXPECTED_FAMILY_SHA256 = rc.FAMR_SHA256
    m.STATE_FMT = os.path.join(ck, "exp45a_r_state_r%02d.pt")
    m.REC_FMT = os.path.join(ck, "exp45a_r_restart_r%02d.json")
    m.SENTINEL = os.path.join(ck, "exp45a_r_CERTIFIED.flag")
    rec_glob = os.path.join(ck, "exp45a_r_restart_r*.json")

    def shim_glob(pattern):
        if pattern != "exp45a_restart_r*.json":
            raise RuntimeError("unexpected glob %r" % pattern)
        return _glob.glob(rec_glob)
    m.glob = types.SimpleNamespace(glob=shim_glob)
    m.train_restart = train_restart
    m.lbfgs_polish = lbfgs_polish
    state = {"results_path": None}
    fresh = rc.make_fresh_results_path({"results_exp45a": "results_exp45a_r"},
                                       outdir)

    def fresh_results_path(base):
        state["results_path"] = fresh(base)
        return state["results_path"]
    m.fresh_results_path = fresh_results_path
    if smoke:
        m.N_RESTARTS = 1
        m.MAX_EPOCHS = 5
        m.LBFGS_MAX_ITER = 3
    return state, ck


def run_main(argv):
    old = sys.argv
    sys.argv = [old[0]] + argv
    try:
        return m.main()
    finally:
        sys.argv = old


def finalize_provenance(state, ck, smoke):
    p = state["results_path"]
    if not (p and os.path.exists(p)):
        return
    with open(p, encoding="utf-8") as f:
        out = json.load(f)
    recs = out.get("restarts", [])
    rc.inject_provenance(p, {
        "registration": rc.REGISTRATION,
        "wrapped_runner": "exp45a_representability.py (unedited)",
        "wrapped_source_sha256": ORIG_SRC_SHA,
        "guarded_copies_of": PINNED_SRC, "smoke": smoke,
        "readout_patch": PATCH,
        "family_is": "family-r (%s, sha %s)" % (rc.FAMR_DATA, rc.FAMR_SHA256),
        "ckpt_dir": ck,
        "restarts_aborted_nonfinite": [r["restart"] for r in recs
                                       if r.get("aborted_nonfinite")],
        "polish_nonfinite": POLISH_NONFINITE,
        "polish_sentinel_note": "train_acc_after_lbfgs = -1.0 marks an "
                                "aborted polish (sentinel, not a measurement)",
        "verdict_alias": {"bridge_representable_2q_r":
                          out["verdicts"]["bridge_representable_2q"]},
    })
    print("[r_exp45a] provenance added -> %s ; bridge_representable_2q_r=%s"
          % (p, out["verdicts"]["bridge_representable_2q"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", "--worker", dest="shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    outdir = "r_smoke" if args.smoke else "."
    os.makedirs(outdir, exist_ok=True)
    if args.finalize or args.smoke:
        rc.rule3(os.path.join(outdir, "results_exp45a_r.json"))
    state, ck = install(outdir, args.smoke)
    if args.smoke:
        r1 = run_main(["--shard", "0", "--nshards", "1"])
        r2 = run_main(["--finalize"])
        finalize_provenance(state, ck, True)
        return r1 or r2
    if args.finalize:
        # completeness guard (wrapper-side; the runner would finalize on a
        # partial set): all N_RESTARTS records, or a certificate sentinel
        done = [r for r in range(m.N_RESTARTS) if os.path.exists(m.REC_FMT % r)]
        if len(done) < m.N_RESTARTS and not os.path.exists(m.SENTINEL):
            print("[r_exp45a] REFUSING to finalize: %d/%d restart records and "
                  "no certificate sentinel" % (len(done), m.N_RESTARTS))
            return 3
        r = run_main(["--finalize"])
        finalize_provenance(state, ck, False)
        return r
    return run_main(["--shard", str(args.shard), "--nshards", str(args.nshards)])


if __name__ == "__main__":
    sys.exit(main())
