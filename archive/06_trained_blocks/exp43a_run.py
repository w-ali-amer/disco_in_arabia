# -*- coding: utf-8 -*-
"""exp43a_run.py -- the exp43a numerical representability certificate.

Doc 23 exp43a step (ii), budget FROZEN:
  40 restarts x (Adam lr 0.02, <=500 epochs, batch 1) + L-BFGS polish on
  the best restarts; target TRAIN accuracy 1.0 on the canonical family
  (generalization irrelevant); early stop on train acc >= 0.99.

Pre-registered verdicts (doc 23, frozen), written to results_exp43a.json:
  bridge_representable          = any restart reaches train >= 0.99
                                  (Adam or its L-BFGS polish -- the polish
                                  is part of the same restart's budget)
  bridge_unrepresentable_proved = true ONLY with a written proof; the
                                  analytic work (report + analysis JSON)
                                  produced no impossibility proof -> false
  representability_open         = neither of the above

Model: KakStoryModel (exp43a_model) -- per-verb Cartan/KAK two-qubit
blocks, provably covering all of SU(4) up to irrelevant global phase, plus
the two shared trained question effects; readout code paths inherited
verbatim from the committed exp42_models. Family = train split of
stories_exp43a_family.json (Q1 bridge twins, K=4, T=5, 6 verbs).

EXECUTION NOTES (flagged; semantics of the frozen budget unchanged):
  * Restarts are independent by construction (fixed seeds 43000+r), so
    they are sharded over parallel single-threaded worker processes for
    wall-clock only:   --worker W --restarts A B   trains r in [A, B).
    --finalize merges worker records, runs the L-BFGS polish if no
    certificate, and writes the results JSON.
  * Once any restart certifies (>= 0.99) a sentinel file stops the other
    workers: the frozen verdict is an existence claim and further
    restarts cannot change it. Aborted/skipped restarts are recorded.
  * Evaluation uses block matrices cached once per epoch under no_grad
    (identical math; parity with the committed exp42 forward is asserted
    at startup).

Never overwrites existing results files (suffix _b, _c, ...).
"""

import argparse
import datetime
import glob
import json
import os
import random
import time

import torch

import exp42_compiler as comp
import exp42_sim_adapter as sim
from exp43a_model import KakStoryModel

DATA = "stories_exp43a_family.json"
RESULTS_BASE = "results_exp43a"
BEST_CKPT = "exp43a_best.pt"
STATE_FMT = "exp43a_state_r%02d.pt"
PROGRESS_FMT = "exp43a_progress_w%d.json"         # rolling, overwritable
CERT_SENTINEL = "exp43a_CERT.json"

# pinned after datagen (provenance gate, exp42_compiler convention)
EXPECTED_FAMILY_SHA256 = \
    "93f52af6daae07f13e48b11d87233d779a21caeb66225571cb85307207b0dfc4"

N_RESTARTS = 40
LR = 0.02
MAX_EPOCHS = 500
TARGET = 0.99
POLISH_TOP = 3
LBFGS_MAX_ITER = 300


def log(msg):
    print("%s %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg),
          flush=True)


def load_family():
    with open(DATA, encoding="utf-8") as f:
        data = json.load(f)
    items, meta = data["items"], data["meta"]
    sha = comp.canonical_sha256(items)
    if sha != meta["dataset_sha256"] or sha != EXPECTED_FAMILY_SHA256:
        raise RuntimeError(
            "family provenance FAILED: sha=%s manifest=%s pin=%s"
            % (sha, meta["dataset_sha256"], EXPECTED_FAMILY_SHA256))
    angles, _m, emb_sha = comp.load_embeddings()
    fam_items = [it for it in items if it["split"] == "train"]
    compiled = [comp.compile_story(it, angles) for it in fam_items]
    verbs = sorted(meta["verb_inventory"].keys())
    assert len(compiled) >= 200, "family below pre-registered floor"
    assert all(cs.qtype == "Q1" for cs in compiled)
    return compiled, verbs, meta, sha, emb_sha


# ------------------------------------------------------------------ eval
def fast_eval(model, compiled):
    """Train accuracy with block/effect matrices built ONCE (no_grad).
    Same math as model.predict; parity asserted by check_eval_parity."""
    with torch.no_grad():
        mats = {v: model.verb_matrix(v) for v in model.verbs}
        effs = [model.q_pos.matrix().conj().T, model.q_neg.matrix().conj().T]
        n_ok = 0
        for cs in compiled:
            psi = sim.init_state(cs.K)
            for w, _n, ang in cs.intro:
                psi = sim.apply_gate(psi, sim.ry(ang), (w,))
            for v, wa, wp in cs.events:
                psi = sim.apply_gate(psi, mats[v], (wa, wp))
            vseq, qa, qp = cs.question
            qw = (qa, qp)
            for qv in reversed(vseq):
                psi = sim.apply_gate(psi, mats[qv].conj().T, qw)
            vals = [sim.prob_all_zero_on(sim.apply_gate(psi, E, qw), qw)
                    for E in effs]
            pred = 0 if vals[0].item() >= vals[1].item() else 1
            n_ok += (pred == cs.answer)
    return n_ok / len(compiled)


def check_eval_parity(compiled, verbs):
    model = KakStoryModel(verbs, 99)
    with torch.no_grad():
        ref = [model.predict(cs) for cs in compiled[:20]]
        mats = {v: model.verb_matrix(v) for v in model.verbs}
        effs = [model.q_pos.matrix().conj().T, model.q_neg.matrix().conj().T]
        for cs, rp in zip(compiled[:20], ref):
            psi = sim.init_state(cs.K)
            for w, _n, ang in cs.intro:
                psi = sim.apply_gate(psi, sim.ry(ang), (w,))
            for v, wa, wp in cs.events:
                psi = sim.apply_gate(psi, mats[v], (wa, wp))
            vseq, qa, qp = cs.question
            qw = (qa, qp)
            for qv in reversed(vseq):
                psi = sim.apply_gate(psi, mats[qv].conj().T, qw)
            vals = [sim.prob_all_zero_on(sim.apply_gate(psi, E, qw), qw)
                    for E in effs]
            pred = 0 if vals[0].item() >= vals[1].item() else 1
            assert pred == rp, "fast_eval parity broken on %s" % cs.item_id
    log("fast_eval parity vs committed exp42 predict: OK (20 items)")


# ----------------------------------------------------------------- training
def certificate_seen():
    return os.path.exists(CERT_SENTINEL)


def train_restart(r, compiled, verbs, max_epochs):
    seed = 43000 + r
    model = KakStoryModel(verbs, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = random.Random(seed)
    order = list(range(len(compiled)))
    best_acc, best_epoch, best_state = -1.0, -1, None
    t0 = time.time()
    epochs_run, aborted = 0, False
    for ep in range(max_epochs):
        epochs_run = ep + 1
        rng.shuffle(order)
        tot_loss = 0.0
        for i in order:
            loss, _pred = model.story_loss(compiled[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
        acc = fast_eval(model, compiled)
        if acc > best_acc:
            best_acc, best_epoch = acc, ep
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if ep % 10 == 0 or acc >= TARGET:
            log("[r%02d ep%03d] train_acc=%.4f best=%.4f loss=%.4f (%.1fs)"
                % (r, ep, acc, best_acc, tot_loss / len(order),
                   time.time() - t0))
        if best_acc >= TARGET:
            break
        if certificate_seen():
            aborted = True
            log("[r%02d] aborting at ep%03d -- certificate sentinel seen"
                % (r, ep))
            break
    return {"restart": r, "seed": seed, "best_train_acc_adam": best_acc,
            "best_epoch": best_epoch, "epochs_run": epochs_run,
            "aborted_on_certificate": aborted,
            "wall_sec": round(time.time() - t0, 1)}, best_state


def run_worker(worker, r_lo, r_hi, compiled, verbs, max_epochs):
    records = []
    for r in range(r_lo, r_hi):
        if certificate_seen():
            records.append({"restart": r, "seed": 43000 + r,
                            "skipped_after_certificate": True})
            continue
        rec, state = train_restart(r, compiled, verbs, max_epochs)
        torch.save({"state_dict": state, "restart": r, "seed": rec["seed"],
                    "train_acc": rec["best_train_acc_adam"]}, STATE_FMT % r)
        records.append(rec)
        with open(PROGRESS_FMT % worker, "w") as f:
            json.dump(records, f, indent=1)
        if rec["best_train_acc_adam"] >= TARGET and not certificate_seen():
            with open(CERT_SENTINEL, "w") as f:
                json.dump({"restart": r, "acc": rec["best_train_acc_adam"],
                           "stage": "adam"}, f)
            log("CERTIFICATE at restart %d (adam): %.4f"
                % (r, rec["best_train_acc_adam"]))
    with open(PROGRESS_FMT % worker, "w") as f:
        json.dump(records, f, indent=1)
    log("worker %d done (restarts %d..%d)" % (worker, r_lo, r_hi - 1))


# ----------------------------------------------------------------- finalize
def lbfgs_polish(state, compiled, verbs, r):
    model = KakStoryModel(verbs, 0)
    model.load_state_dict(state)
    opt = torch.optim.LBFGS(model.parameters(), max_iter=LBFGS_MAX_ITER,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        total = 0.0
        for cs in compiled:
            loss, _ = model.story_loss(cs)
            total = total + loss
        total = total / len(compiled)
        total.backward()
        return total

    t0 = time.time()
    final = opt.step(closure)
    acc = fast_eval(model, compiled)
    log("[polish r%02d] train_acc=%.4f loss=%.4f (%.1fs)"
        % (r, acc, float(final), time.time() - t0))
    new_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return acc, float(final), new_state


def finalize(compiled, verbs, meta, sha, emb_sha, smoke, t_start,
             n_workers_note):
    records = []
    for path in sorted(glob.glob("exp43a_progress_w*.json")):
        with open(path) as f:
            records.extend(json.load(f))
    records.sort(key=lambda x: x["restart"])
    trained = [x for x in records if "best_train_acc_adam" in x]
    assert trained, "no trained restarts found"

    certified = any(x["best_train_acc_adam"] >= TARGET for x in trained)
    polish = []
    if not certified and not smoke:
        ranked = sorted(trained,
                        key=lambda x: -x["best_train_acc_adam"])[:POLISH_TOP]
        for rec in ranked:
            r = rec["restart"]
            blob = torch.load(STATE_FMT % r, map_location="cpu",
                              weights_only=False)
            acc, lval, new_state = lbfgs_polish(blob["state_dict"],
                                                compiled, verbs, r)
            polish.append({"restart": r, "train_acc_after_lbfgs": acc,
                           "final_loss": lval})
            rec["best_train_acc_after_polish"] = max(
                rec["best_train_acc_adam"], acc)
            if acc > blob["train_acc"]:
                torch.save({"state_dict": new_state, "restart": r,
                            "seed": rec["seed"], "train_acc": acc,
                            "stage": "lbfgs"}, STATE_FMT % r)
            if acc >= TARGET:
                certified = True
                log("CERTIFICATE at restart %d (lbfgs polish): %.4f"
                    % (r, acc))
                break

    def restart_best(x):
        return x.get("best_train_acc_after_polish",
                     x.get("best_train_acc_adam", -1.0))

    best_overall = max(trained, key=restart_best)
    r_best = best_overall["restart"]
    blob = torch.load(STATE_FMT % r_best, map_location="cpu",
                      weights_only=False)
    torch.save(blob, BEST_CKPT)
    log("best checkpoint -> %s (restart %d, acc %.4f)"
        % (BEST_CKPT, r_best, restart_best(best_overall)))

    bridge_representable = any(restart_best(x) >= TARGET for x in trained)
    verdicts = {
        "bridge_representable": bool(bridge_representable),
        "bridge_unrepresentable_proved": False,
        "representability_open": bool(not bridge_representable),
    }

    results = {
        "experiment": "exp43a",
        "design_doc": "qnlp_private_docs/23_exp43_ladder_representability_"
                      "design.md exp43a (commit ef15d76; verdict criteria "
                      "frozen)",
        "timestamp": datetime.datetime.now().isoformat(),
        "smoke": smoke,
        "dataset_file": DATA,
        "family_dataset_sha256": sha,
        "family_seed": meta["seed"],
        "family_definition": meta["family_definition"],
        "family_n_items": len(compiled),
        "embeddings_sha256": emb_sha,
        "verbs": verbs,
        "parameterization": {
            "blocks": "Cartan/KAK two-qubit form (A1xA2)exp(i(aXX+bYY+cZZ))"
                      "(B1xB2), XZX-Euler locals, 15 params/verb; covers ALL "
                      "of SU(4) up to global phase (Khaneja-Glaser/Kraus-"
                      "Cirac), and global phases are irrelevant to every "
                      "model probability -- provably universal, unlike the "
                      "Circuit-4 3-layer ansatz exp42 used for A3",
            "effects": "two shared KAK effects; effect vector Q|00> ranges "
                       "over all two-qubit pure states (Schmidt argument)",
            "readout": "inherited verbatim from committed exp42_models."
                       "QuantumStoryModel (two-effect log-p softmax, "
                       "inverse-preparation question encoding)",
            "intro": "frozen exp42 noun-embedding Ry angles (exp42 model "
                     "class: only blocks + effects are trainable)",
        },
        "budget": {"n_restarts": N_RESTARTS, "optimizer": "Adam",
                   "lr": LR, "max_epochs": MAX_EPOCHS, "batch_size": 1,
                   "early_stop_train_acc": TARGET,
                   "lbfgs_polish_top": POLISH_TOP,
                   "lbfgs_max_iter": LBFGS_MAX_ITER,
                   "restart_seeds": "43000 + restart_index"},
        "execution_notes": [
            "restarts sharded over %s independent single-threaded worker "
            "processes (wall-clock only; restart seeds and per-restart "
            "training identical to the serial protocol)" % n_workers_note,
            "restarts after a certificate are skipped/aborted (the frozen "
            "verdict is an existence claim); recorded per restart",
            "eval uses per-epoch cached block matrices (identical math; "
            "parity with committed exp42 predict asserted at startup)",
        ],
        "restarts_completed": len(trained),
        "restarts_skipped_or_aborted": [
            x["restart"] for x in records
            if x.get("skipped_after_certificate")
            or x.get("aborted_on_certificate")],
        "per_restart": records,
        "lbfgs_polish": polish,
        "best_restart": {"restart": r_best,
                         "best_train_acc": restart_best(best_overall)},
        "verdicts": verdicts,
        "analytic_findings_summary": [
            "Lemma 1 (exact): for blocks commuting with SWAP (A1 class) "
            "twin stories compile to identical statevectors; any readout "
            "is at exactly 50% on twins (verified to float precision, "
            "analysis W1).",
            "Prop 2 (refutes the strong form of doc 23's candidate "
            "obstruction): for generic full-SU(4) assignments the twin "
            "reduced states on the two queried wires DIFFER for every "
            "family pair (analysis W2: trace distances median ~0.5, none "
            "below 1e-3), and the effect-pair readout form is information-"
            "complete on the queried-wire reduced state (a Hermitian "
            "difference vanishing in all rank-1 effects is zero). Any "
            "unrepresentability could therefore only arise from sharing "
            "blocks/effects family-wide, not from per-item information "
            "loss under partial trace.",
            "Prop 3 (reduction): representability == existence of blocks "
            "plus ONE traceless rank-<=2 Hermitian decision functional D "
            "with sign Tr[rho_tilde D] = label across the family; the "
            "unranked relaxation of the best-D fit at fixed random blocks "
            "reaches only ~0.59-0.63 (analysis W3), locating the "
            "difficulty in family-wide sharing.",
            "No impossibility proof found; no exact hand construction "
            "found (obstacles: random temporal order of the two core "
            "events (48% e1-first), filler traffic on the queried wires "
            "by shared verbs (99% of pairs, mean 2.4 touches), "
            "item-dependent frozen intro angles vs fixed effects -- "
            "analysis W4). Decision therefore falls to this numerical "
            "certificate, per the frozen protocol.",
        ],
        "analysis_results_file": "results_exp43a_analysis*.json",
        "torch_version": torch.__version__,
        "wall_time_sec_finalize": round(time.time() - t_start, 1),
        "git_note": "local commit only, author w-ali-amer, no push",
    }

    out = RESULTS_BASE + ".json"
    suffix = ord("b")
    while os.path.exists(out):                    # never overwrite results
        out = "%s_%s.json" % (RESULTS_BASE, chr(suffix))
        suffix += 1
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    log("results -> %s" % out)
    log("VERDICTS: %s" % json.dumps(verdicts))
    accs = sorted((restart_best(x) for x in trained), reverse=True)
    log("restart best accs (desc): %s" % [round(a, 4) for a in accs])
    return out


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--restarts", type=int, nargs=2, metavar=("LO", "HI"))
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)
    t_start = time.time()
    compiled, verbs, meta, sha, emb_sha = load_family()

    if args.finalize:
        log("finalize: merging worker records")
        n_workers = len(glob.glob("exp43a_progress_w*.json"))
        finalize(compiled, verbs, meta, sha, emb_sha, args.smoke, t_start,
                 str(n_workers))
        return

    check_eval_parity(compiled, verbs)
    log("family loaded: %d items (%d pairs), verbs=%s"
        % (len(compiled), len(compiled) // 2, verbs))
    if args.smoke:
        run_worker(0, 0, 2, compiled, verbs, max_epochs=3)
        return
    assert args.worker is not None and args.restarts, \
        "need --worker and --restarts LO HI (or --finalize / --smoke)"
    run_worker(args.worker, args.restarts[0], args.restarts[1],
               compiled, verbs, MAX_EPOCHS)


if __name__ == "__main__":
    main()
