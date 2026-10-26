# -*- coding: utf-8 -*-
"""exp45a_representability.py -- doc 28 SS4: does a 2-hop bridge become
REPRESENTABLE once each referent gets a two-qubit register?

Question R, restated for the new class: does there EXIST a shared assignment
of per-verb 4-wire blocks + two shared question effects that answers Q1
bridge questions correctly across the frozen exp43a family? exp43a asked
this for ONE qubit per referent and returned representability_open at a best
train accuracy of 0.955 against a 0.99 bar, with no obstruction found.

THIS IS A NEW EXPERIMENT, NOT AN exp43a BUDGET EXTENSION. Doc 24 SS9 froze
exp43a's budget and that freeze is respected: results_exp43a.json stands as
computed and is neither recomputed nor amended. A different model class
requires its own certificate, which is what this is.

Method mirrors exp43a's numerical certificate exactly, changing only the
block class: 40 restarts x (Adam lr 0.02, <=500 epochs, batch 1) + L-BFGS
polish on the top 3, target TRAIN accuracy on the family (generalisation is
irrelevant to representability). Family reused VERBATIM under a pinned
sha256; no new data.

Verdicts (doc 28 SS4, frozen):
  bridge_representable_2q       = any restart reaches train >= 0.99
  bridge_unrepresentable_2q_proved = only with a WRITTEN PROOF (else false)
  representability_2q_open      = otherwise; reported as OPEN, NEVER as
                                  "proved impossible"

FROZEN INTERPRETATION GUARD (doc 28 SS4), stated before any number exists:
  * A HIGHER best-train here than exp43a's 0.955 is DESCRIPTIVE ONLY. It is
    not evidence that the two-qubit class is more expressive: the two runs
    differ in optimisation landscape as well as in class, and exp43a's
    number is a budget-limited FLOOR, not a ceiling. Any comparison of the
    two carries this sentence.
  * representability_2q_open is NOT evidence against the class. It
    reproduces exp43a's own ambiguity and is reported as such.

SCOPE LIMIT (recorded because it bounds every negative reading): A4 is
Circuit4(4 wires, 3 layers) = 33 params/verb, which is NOT a universal
parameterisation of SU(16) (255 params). exp43a deliberately used a PROVABLY
universal KAK form for its 1-qubit certificate for exactly this reason. A
failure to certify here is therefore ambiguous between "the two-qubit
register does not make the bridge representable" and "this ansatz is too
small", and must never be reported as the former alone. Doc 28 SS3 froze the
arm as Circuit4(4,3); widening it needs a new numbered doc.

Usage:
  python exp45a_representability.py --shard 0 --nshards 4    # run a slice
  python exp45a_representability.py --finalize               # collect+polish
"""

import argparse
import glob
import json
import os
import platform
import random
import sys
import time

import torch

import exp42_compiler as comp
import exp42_sim_adapter as sim
from exp42_run import fresh_results_path, git_note
from exp45_model import TwoQubitReferentModel

DESIGN_DOC = ("qnlp_private_docs/28_exp45_two_qubit_referent_multihop.md "
              "(commit cb19b4f, FROZEN)")
DATA = "stories_exp43a_family.json"
EXPECTED_FAMILY_SHA256 = \
    "93f52af6daae07f13e48b11d87233d779a21caeb66225571cb85307207b0dfc4"
N_RESTARTS = 40
LR = 0.02
MAX_EPOCHS = 500
TARGET = 0.99
POLISH_TOP = 3
LBFGS_MAX_ITER = 300
RNG_BASE = 45000                       # exp45 stream (exp43a used 43000)
STATE_FMT = "exp45a_state_r%02d.pt"
REC_FMT = "exp45a_restart_r%02d.json"
SENTINEL = "exp45a_CERTIFIED.flag"

EXP43A_COMPARISON_GUARD = (
    "doc 28 SS4, frozen BEFORE any number existed: a higher best-train than "
    "exp43a's 0.955 is DESCRIPTIVE ONLY -- the runs differ in optimisation "
    "landscape as well as model class, and exp43a's 0.955 is a "
    "budget-limited FLOOR, not a ceiling. representability_2q_open is NOT "
    "evidence against the class; it reproduces exp43a's own ambiguity.")

ANSATZ_CAVEAT = (
    "A4 = Circuit4(4 wires, 3 layers) = 33 params/verb is NOT universal on "
    "SU(16) (255 params). exp43a used a provably universal KAK form for its "
    "1-qubit certificate precisely for this reason. A failure to certify is "
    "ambiguous between 'the two-qubit register does not make the bridge "
    "representable' and 'this ansatz is too small', and must never be "
    "reported as the former alone.")


def log(msg):
    print(msg, flush=True)


def load_family():
    with open(DATA, encoding="utf-8") as f:
        data = json.load(f)
    items, meta = data["items"], data["meta"]
    sha = comp.canonical_sha256(items)
    if sha != meta["dataset_sha256"] or sha != EXPECTED_FAMILY_SHA256:
        raise RuntimeError("family provenance FAILED: sha=%s manifest=%s "
                           "pin=%s" % (sha, meta["dataset_sha256"],
                                       EXPECTED_FAMILY_SHA256))
    angles, _m, emb_sha = comp.load_embeddings()
    fam = [it for it in items if it["split"] == "train"]
    compiled = [comp.compile_story(it, angles) for it in fam]
    verbs = sorted(meta["verb_inventory"].keys())
    assert len(compiled) >= 200, "family below pre-registered floor"
    assert all(cs.qtype == "Q1" for cs in compiled)
    return compiled, verbs, meta, sha, emb_sha


def fast_eval(model, compiled):
    """Train accuracy with block/effect matrices built ONCE (no_grad).
    Same math as model.predict on the two-qubit-register wiring; parity
    asserted by check_eval_parity."""
    with torch.no_grad():
        mats = {v: model.verb_matrix(v) for v in model.verbs}
        effs = [model.q_pos.matrix().conj().T, model.q_neg.matrix().conj().T]
        n_ok = 0
        for cs in compiled:
            psi = sim.init_state(2 * cs.K)
            for w, _n, ang in cs.intro:
                psi = sim.apply_gate(psi, sim.ry(ang), (2 * w,))
            for v, wa, wp in cs.events:
                psi = sim.apply_gate(psi, mats[v],
                                     (2 * wa, 2 * wa + 1, 2 * wp, 2 * wp + 1))
            vseq, qa, qp = cs.question
            qw = (2 * qa, 2 * qa + 1, 2 * qp, 2 * qp + 1)
            for qv in reversed(vseq):
                psi = sim.apply_gate(psi, mats[qv].conj().T, qw)
            vals = [sim.prob_all_zero_on(sim.apply_gate(psi, E, qw), qw)
                    for E in effs]
            pred = 0 if vals[0].item() >= vals[1].item() else 1
            n_ok += (pred == cs.answer)
    return n_ok / len(compiled)


def check_eval_parity(compiled, verbs):
    """fast_eval must agree with model.predict item-for-item."""
    m = TwoQubitReferentModel("A4", verbs, 0)
    sub = compiled[:25]
    slow = sum(1 for cs in sub if m.predict(cs) == cs.answer) / len(sub)
    fast = fast_eval(m, sub)
    assert abs(slow - fast) < 1e-12, "eval parity broken: %f vs %f" % (slow, fast)
    return {"n_checked": len(sub), "slow": slow, "fast": fast, "parity": True}


def certificate_seen():
    return os.path.exists(SENTINEL)


def train_restart(r, compiled, verbs, max_epochs):
    seed = RNG_BASE + r
    model = TwoQubitReferentModel("A4", verbs, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = random.Random(seed)
    order = list(range(len(compiled)))
    best_acc, best_epoch, best_state = -1.0, -1, None
    t0 = time.time()
    epochs_run, aborted = 0, False
    for ep in range(max_epochs):
        epochs_run = ep + 1
        rng.shuffle(order)
        tot = 0.0
        for i in order:
            loss, _p = model.story_loss(compiled[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        acc = fast_eval(model, compiled)
        if acc > best_acc:
            best_acc, best_epoch = acc, ep
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if ep % 10 == 0 or acc >= TARGET:
            log("[r%02d ep%03d] train_acc=%.4f best=%.4f loss=%.4f (%.1fs)"
                % (r, ep, acc, best_acc, tot / len(order), time.time() - t0))
        if best_acc >= TARGET:
            open(SENTINEL, "w").write("restart %d\n" % r)
            break
        if certificate_seen():
            aborted = True
            log("[r%02d] aborting at ep%03d -- certificate sentinel seen"
                % (r, ep))
            break
    rec = {"restart": r, "seed": seed, "best_train_acc_adam": best_acc,
           "best_epoch": best_epoch, "epochs_run": epochs_run,
           "aborted_on_certificate": aborted,
           "wall_sec": round(time.time() - t0, 1)}
    with open(REC_FMT % r, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    torch.save({"state_dict": best_state, "restart": r, "stage": "adam",
                "best_train_acc": best_acc}, STATE_FMT % r)
    return rec


def lbfgs_polish(state, compiled, verbs, r, max_iter=LBFGS_MAX_ITER):
    model = TwoQubitReferentModel("A4", verbs, RNG_BASE + r)
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
        return total

    lval = opt.step(closure)
    acc = fast_eval(model, compiled)
    return acc, float(lval), {k: v.detach().clone()
                              for k, v in model.state_dict().items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    compiled, verbs, meta, sha, emb_sha = load_family()
    max_epochs = 2 if args.smoke else MAX_EPOCHS
    n_restarts = 2 if args.smoke else N_RESTARTS

    if not args.finalize:
        parity = check_eval_parity(compiled, verbs)
        log("[exp45a] family n=%d verbs=%d sha=%s | eval parity %s"
            % (len(compiled), len(verbs), sha[:12], parity["parity"]))
        mine = [r for r in range(n_restarts) if r % args.nshards == args.shard]
        log("[exp45a] shard %d/%d -> restarts %s"
            % (args.shard, args.nshards, mine))
        for r in mine:
            if os.path.exists(REC_FMT % r):
                log("[r%02d] already done, skipping" % r)
                continue
            if certificate_seen():
                log("[exp45a] certificate already seen; stopping shard")
                break
            train_restart(r, compiled, verbs, max_epochs)
        log("[exp45a] shard %d complete (%.1f min)"
            % (args.shard, (time.time() - t0) / 60))
        return 0

    # ---------------- finalize: collect, polish top-3, write verdicts
    recs = []
    for p in sorted(glob.glob("exp45a_restart_r*.json")):
        with open(p, encoding="utf-8") as f:
            recs.append(json.load(f))
    if not recs:
        log("[exp45a] no restart records found")
        return 1
    recs.sort(key=lambda x: -x["best_train_acc_adam"])
    log("[exp45a] %d restarts; best Adam %.4f"
        % (len(recs), recs[0]["best_train_acc_adam"]))

    polish = []
    for rec in recs[:POLISH_TOP]:
        r = rec["restart"]
        blob = torch.load(STATE_FMT % r, weights_only=False)
        log("[exp45a] L-BFGS polish on restart %d (adam %.4f)..."
            % (r, rec["best_train_acc_adam"]))
        acc, lval, new_state = lbfgs_polish(
            blob["state_dict"], compiled, verbs, r,
            max_iter=3 if args.smoke else LBFGS_MAX_ITER)
        polish.append({"restart": r, "train_acc_after_lbfgs": acc,
                       "final_loss": lval,
                       "train_acc_before": rec["best_train_acc_adam"]})
        if acc > rec["best_train_acc_adam"]:
            torch.save({"state_dict": new_state, "restart": r,
                        "stage": "lbfgs", "best_train_acc": acc},
                       STATE_FMT % r)
        log("[exp45a]   -> %.4f" % acc)

    best_overall = max([r["best_train_acc_adam"] for r in recs]
                       + [p["train_acc_after_lbfgs"] for p in polish])
    representable = bool(best_overall >= TARGET)
    verdicts = {
        "bridge_representable_2q": representable,
        "bridge_unrepresentable_2q_proved": False,
        "representability_2q_open": bool(not representable),
        "best_train_acc_overall": best_overall,
        "target": TARGET,
    }
    out = {
        "experiment": "exp45a",
        "spec": DESIGN_DOC,
        "question": ("does a 2-hop bridge become representable with a "
                     "two-qubit register per referent?"),
        "not_an_exp43a_extension": (
            "doc 24 SS9 froze exp43a's budget; this is a NEW certificate for "
            "a DIFFERENT model class. results_exp43a.json stands as computed "
            "and is neither recomputed nor amended."),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "family_dataset_sha256": sha,
        "family_n_items": len(compiled),
        "embeddings_sha256": emb_sha,
        "param_counts": TwoQubitReferentModel("A4", verbs, 0).param_counts(),
        "budget": {"n_restarts": N_RESTARTS, "optimizer": "Adam", "lr": LR,
                   "max_epochs": MAX_EPOCHS, "batch_size": 1,
                   "early_stop_train_acc": TARGET,
                   "lbfgs_polish_top": POLISH_TOP,
                   "lbfgs_max_iter": LBFGS_MAX_ITER,
                   "restart_seeds": "%d + restart_index" % RNG_BASE},
        "restarts": recs,
        "lbfgs_polish": polish,
        "verdicts": verdicts,
        "exp43a_reference": {
            "best_train_acc": 0.955, "target": 0.99,
            "verdict": "representability_open = true",
            "comparison_guard": EXP43A_COMPARISON_GUARD},
        "meta_notes": {"ansatz_caveat": ANSATZ_CAVEAT,
                       "comparison_guard": EXP43A_COMPARISON_GUARD},
        "git_note": git_note(),
        "wall_time_sec": time.time() - t0,
    }
    log("\n[exp45a] VERDICTS")
    for k, v in verdicts.items():
        log("   %-34s %s" % (k, v))
    if args.smoke:
        log("[exp45a] smoke -- no results file")
        return 0
    path = fresh_results_path("results_exp45a")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log("[exp45a] wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
