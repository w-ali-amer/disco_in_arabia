# -*- coding: utf-8 -*-
"""exp45b_phase1.py -- doc 28 SS5: learnability of the two-qubit-referent class.

Runs A4 (Circuit4 on 4 wires, 33 params/verb) on the FROZEN L1 and L2 rungs,
verbatim, at the FROZEN Phase-1 grid. No new data: direct comparability with
exp43b Phase 1 is the entire point, and regenerating here would be dataset
shopping.

  a4_learns_L1        = any config meets train >= 0.8 AND val >= 0.7 on L1
  rung_learnable_L2_2q = same gate on L2

L1 IS THE POSITIVE CONTROL AND IT GATES THE L2 READING (doc 28 SS5). The
1-qubit class already learns L1 (A3 4/8 configs, A2 test 0.799). A
two-qubit-referent arm that CANNOT learn L1 has an optimisation or wiring
problem, and its L2 failure would carry NO information about multi-hop
composition. Frozen consequence: if a4_learns_L1 is false, then
rung_learnable_L2_2q is reported as UNINTERPRETABLE whatever its value, and
the finding is the L1 regression (Rule 4: lead with it).

Also here, both gating everything downstream:
  - doc 28 SS3 sentinel verification (as amended by doc 30);
  - doc 24 SS5.2 memorisation control for A4 and A5, with A5 as
    control-of-the-control: A5 is register-swap-commuting by construction so
    it MUST fail. If A5 passes, the control is not measuring what it claims
    and exp45 is reported as inconclusive rather than acted upon.

Compute protocol (doc 28 SS8), registered so scope cannot be shopped after
timings are seen: measure s/epoch on ONE config first, project the full
grid, run it whole if <= 30 h, else drop the two batch-1 configs and record
compute_fallback_applied. No other reduction is permitted.
"""

import argparse
import copy
import json
import math
import os
import platform
import random
import sys
import time

import torch

import exp42_compiler as comp
from exp42_run import fresh_results_path, git_note, grad_global_norm
from exp45_model import TwoQubitReferentModel, verify_sentinel

DESIGN_DOC = ("qnlp_private_docs/28_exp45_two_qubit_referent_multihop.md "
              "(commit cb19b4f) as amended by 30_exp45_sentinel_tolerance_"
              "amendment.md (commit aa05e85), FROZEN")
CKPT_DIR = "exp45b_ckpt"
GRID = [(0.005, 1), (0.005, 8), (0.02, 1), (0.02, 8)]   # FROZEN Phase-1 grid
FALLBACK_GRID = [(0.005, 8), (0.02, 8)]                 # doc 28 SS8 step 4
N_SEEDS = 2                                             # FROZEN (doc 23)
MAX_EPOCHS = 60                                         # FROZEN
TRAIN_GATE, VAL_GATE = 0.8, 0.7                         # FROZEN
MEMO_ITEMS, MEMO_EPOCHS, MEMO_BAR = 64, 300, 0.95       # doc 24 SS5.2
RNG_BASE = 45000
COMPUTE_BUDGET_H = 30.0
AUDIT_PAIRS = 100

L1_GATES_L2 = (
    "doc 28 SS5, frozen: L1 is the positive control. The 1-qubit class "
    "already learns L1 (A3 4/8 configs; A2 test 0.799 at Phase 2). If "
    "a4_learns_L1 is false, rung_learnable_L2_2q is UNINTERPRETABLE whatever "
    "its value -- a wiring/optimisation regression cannot evidence anything "
    "about multi-hop composition -- and the finding is the L1 regression.")

SCOPE_NOTE = (
    "doc 28 SS9: no quantum-advantage reading is available from exp45 in any "
    "outcome. A4 is a full-Circuit4 4-wire class with NO structure-matched "
    "classical counterpart in the arm set (the same limitation doc 24 SS7 "
    "froze for tier-q'). A learnability verdict is a statement about a model "
    "class on a rung, not a comparison. A4's analog-compilability is neither "
    "claimed nor implied, and no hardware reading follows.")

ANSATZ_CAVEAT = (
    "SCOPE LIMIT, recorded because it bounds every negative reading: A4 is "
    "Circuit4(4 wires, 3 layers) = 33 params/verb, which is NOT a universal "
    "parameterisation of SU(16) (255 params). exp43a deliberately used a "
    "PROVABLY universal KAK form for its 1-qubit-referent certificate for "
    "exactly this reason. So an exp45 failure to learn is ambiguous between "
    "'the two-qubit register does not help' and 'this ansatz is too small', "
    "and must never be reported as the former alone. Doc 28 SS3 froze the "
    "arm as Circuit4(4,3); widening it needs a new numbered doc.")


def evaluate(model, compiled):
    correct = [1 if model.predict(cs) == cs.answer else 0 for cs in compiled]
    return sum(correct) / len(correct), correct


def train_run(arm, seed, lr, batch, data, verbs, max_epochs,
              train_items=None, select_on_val=True, tag=""):
    torch.manual_seed(seed)
    model = TwoQubitReferentModel(arm, verbs, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(RNG_BASE + seed)
    train = train_items if train_items is not None else data["train"]
    val = data["val"] if select_on_val else train

    best = {"val_acc": -1.0, "train_acc": -1.0, "epoch": -1, "state": None}
    curves = {"loss": [], "train_run_acc": [], "val_acc": [], "grad_norm": []}
    steps, ep_times = 0, []
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses, gnorms, run_correct = [], [], 0
        for start in range(0, len(order), batch):
            bat = [train[i] for i in order[start:start + batch]]
            opt.zero_grad()
            bloss = 0.0
            for cs in bat:
                loss, pred = model.story_loss(cs)
                bloss = bloss + loss
                if pred == cs.answer:
                    run_correct += 1
            bloss = bloss / len(bat)
            bloss.backward()
            gn = grad_global_norm(model)
            assert math.isfinite(gn), "non-finite grad (%s s%d)" % (arm, seed)
            gnorms.append(gn)
            opt.step()
            steps += 1
            losses.append(bloss.item())

        val_acc, _ = evaluate(model, val)
        ep_times.append(time.time() - t0)
        curves["loss"].append(sum(losses) / len(losses))
        curves["train_run_acc"].append(run_correct / len(train))
        curves["val_acc"].append(val_acc)
        curves["grad_norm"].append(sum(gnorms) / len(gnorms))
        if val_acc > best["val_acc"]:
            clean_train, _ = evaluate(model, train)
            best = {"val_acc": val_acc, "train_acc": clean_train,
                    "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        print("[%s%s s%d lr%.3f b%d] ep %2d/%d loss=%.4f train(run)=%.3f "
              "val=%.3f gn=%.2e best@%d(%.3f) %.1fs"
              % (tag, arm, seed, lr, batch, epoch, max_epochs,
                 curves["loss"][-1], curves["train_run_acc"][-1], val_acc,
                 curves["grad_norm"][-1], best["epoch"], best["val_acc"],
                 ep_times[-1]), flush=True)
        if best["val_acc"] >= 1.0 and best["train_acc"] >= 1.0:
            print("[%s%s s%d] early stop: train and val both 100%%"
                  % (tag, arm, seed), flush=True)
            break

    model.load_state_dict(best["state"])
    test_acc, test_correct = (evaluate(model, data["test"])
                              if select_on_val else (best["train_acc"], []))
    return model, {
        "arm": arm, "seed": seed, "lr": lr, "batch": batch,
        "epochs_run": len(ep_times), "steps_taken": steps,
        "best_epoch": best["epoch"], "val_acc": best["val_acc"],
        "train_acc": best["train_acc"], "test_acc": test_acc,
        "test_correct": test_correct,
        "mean_sec_per_epoch": sum(ep_times) / len(ep_times),
        "gate_met_at_bestval":
            bool(best["train_acc"] >= TRAIN_GATE and best["val_acc"] >= VAL_GATE),
        "curves": curves}


def load_rung(rung):
    with open("stories_exp43b_%s.json" % rung, encoding="utf-8") as f:
        d = json.load(f)
    return d["items"], d["meta"]


def compile_rung(items, angles):
    return {sp: comp.compile_all([i for i in items if i["split"] == sp], angles)
            for sp in ("train", "val", "test")}


def memo_subset(items, angles):
    by = {}
    for it in items:
        if it["split"] == "train":
            by.setdefault(it["pair_id"], []).append(it)
    ids = []
    for p in sorted(by):
        if len(by[p]) == 2:
            ids += [x["id"] for x in sorted(by[p], key=lambda x: x["id"])]
        if len(ids) >= MEMO_ITEMS:
            break
    ids = set(ids[:MEMO_ITEMS])
    return comp.compile_all(
        [i for i in items if i["split"] == "train" and i["id"] in ids], angles)


def twin_pairs(items, angles, verbs, n):
    by = {}
    for it in items:
        if it["split"] == "train":
            by.setdefault(it["pair_id"], []).append(it)
    out = []
    for p in [q for q in sorted(by) if len(by[q]) == 2][:n]:
        a, b = sorted(by[p], key=lambda x: x["id"])
        out.append((comp.compile_story(a, angles),
                    comp.compile_story(b, angles), verbs))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if not args.smoke:
        os.makedirs(CKPT_DIR, exist_ok=True)
    max_epochs = 2 if args.smoke else MAX_EPOCHS
    memo_epochs = 2 if args.smoke else MEMO_EPOCHS
    n_seeds = 1 if args.smoke else N_SEEDS

    angles, _em, emb_sha = comp.load_embeddings()
    rungs = {}
    for r in ("L1", "L2"):
        items, meta = load_rung(r)
        rungs[r] = {"items": items, "meta": meta,
                    "data": compile_rung(items, angles),
                    "sha": meta["dataset_sha256"]}
    verbs = sorted(rungs["L1"]["meta"]["verb_inventory"].keys())
    print("[exp45b] L1 %s | L2 %s | verbs=%d"
          % (rungs["L1"]["sha"][:12], rungs["L2"]["sha"][:12], len(verbs)),
          flush=True)

    # ---------- doc 28 SS3 sentinel verification (as amended by doc 30)
    sentinel = verify_sentinel(
        twin_pairs(rungs["L1"]["items"], angles, verbs, AUDIT_PAIRS),
        n_draws=20 if args.smoke else 200)
    print("[exp45b] sentinel_verified=%s (float64 %.2e, complex64 %.2e, "
          "twin delta %.2e, a5 acc %.4f)"
          % (sentinel["sentinel_verified"],
             sentinel["commutator_float64_worst"],
             sentinel["commutator_complex64_worst"],
             sentinel.get("twin_statevector_delta_worst", float("nan")),
             sentinel.get("a5_twin_accuracy", float("nan"))), flush=True)
    if not sentinel["sentinel_verified"]:
        print("[exp45b] HALT: doc 28 SS3 sentinel broken; no exp45 result is "
              "reported.", flush=True)
        if not args.smoke:
            p = fresh_results_path("results_exp45b_HALTED")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"experiment": "exp45b", "halted": True,
                           "reason": "doc 28 SS3 sentinel verification failed",
                           "sentinel": sentinel}, f, indent=1)
        return 1

    # ---------- doc 28 SS8 compute protocol: measure, project, then decide
    probe_data = rungs["L2"]["data"]
    tprobe = time.time()
    m = TwoQubitReferentModel("A4", verbs, 0)
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    n_probe = 0
    for s in range(0, 160, 8):
        bat = probe_data["train"][s:s + 8]
        if not bat:
            break
        opt.zero_grad()
        L = 0.0
        for cs in bat:
            l, _ = m.story_loss(cs)
            L = L + l
        (L / len(bat)).backward()
        opt.step()
        n_probe += 1
    sec_per_step_b8 = (time.time() - tprobe) / max(n_probe, 1)
    n_train = len(probe_data["train"])
    sec_epoch_b8 = sec_per_step_b8 * math.ceil(n_train / 8)
    sec_epoch_b1 = sec_per_step_b8 / 8.0 * n_train      # per-story cost x N
    # both rungs, both lrs, N_SEEDS seeds, MAX_EPOCHS epochs
    proj_h = (2 * 2 * N_SEEDS * MAX_EPOCHS * (sec_epoch_b8 + sec_epoch_b1)
              / 2.0) / 3600.0
    grid = GRID
    fallback = False
    if not args.smoke and proj_h > COMPUTE_BUDGET_H:
        grid, fallback = FALLBACK_GRID, True
    if args.smoke:
        grid = [(0.02, 8)]
    print("[exp45b] compute probe: %.3f s/step(b8) -> L2 epoch b8 %.1fs / "
          "b1 %.1fs; projected full grid %.1f h; fallback=%s"
          % (sec_per_step_b8, sec_epoch_b8, sec_epoch_b1, proj_h, fallback),
          flush=True)

    # ---------- doc 24 SS5.2 memorisation controls (A5 must FAIL)
    memo = {}
    for arm in ("A4", "A5"):
        subset = memo_subset(rungs["L1"]["items"], angles)
        runs, best = [], 0.0
        for lr, batch in grid:
            for seed in range(2 if not args.smoke else 1):
                _m, d = train_run(arm, seed, lr, batch, rungs["L1"]["data"],
                                  verbs, memo_epochs, train_items=subset,
                                  select_on_val=False, tag="memo:")
                best = max(best, d["train_acc"])
                runs.append({"lr": lr, "batch": batch, "seed": seed,
                             "best_train_acc": d["train_acc"],
                             "meets": bool(d["train_acc"] >= MEMO_BAR)})
        memo[arm] = {"n_items": len(subset), "bar": MEMO_BAR, "runs": runs,
                     "best_train_acc": best,
                     "arm_trainable": bool(best >= MEMO_BAR)}
        print("[exp45b] %s memorisation: best=%.3f trainable=%s"
              % (arm, best, memo[arm]["arm_trainable"]), flush=True)

    a5_fails = not memo["A5"]["arm_trainable"]
    if not a5_fails:
        print("[exp45b] CONTROL-OF-CONTROL FAILED: A5 passed the memorisation "
              "control. exp45 is reported as INCONCLUSIVE, not acted upon.",
              flush=True)

    # ---------- Phase 1 proper
    per_rung = {}
    for rung in ("L1", "L2"):
        runs = []
        for lr, batch in grid:
            for seed in range(n_seeds):
                key = "%s_A4_lr%s_b%d_s%d" % (rung, lr, batch, seed)
                ckj = os.path.join(CKPT_DIR, key + ".json")
                ckp = os.path.join(CKPT_DIR, key + ".pt")
                if not args.smoke and os.path.exists(ckj) \
                        and os.path.exists(ckp):
                    with open(ckj, encoding="utf-8") as f:
                        d = json.load(f)
                    print("[ckpt] %s loaded" % key, flush=True)
                else:
                    mm, d = train_run("A4", seed, lr, batch,
                                      rungs[rung]["data"], verbs, max_epochs,
                                      tag=rung + ":")
                    if not args.smoke:
                        with open(ckj, "w", encoding="utf-8") as f:
                            json.dump(d, f, ensure_ascii=False)
                        torch.save(mm.state_dict(), ckp)
                runs.append({k: v for k, v in d.items() if k != "curves"})
        per_rung[rung] = runs
        print("[exp45b] %s: gate met in %d/%d configs; best val=%.3f "
              "train=%.3f" % (rung, sum(1 for r in runs if r["gate_met_at_bestval"]),
                              len(runs), max(r["val_acc"] for r in runs),
                              max(r["train_acc"] for r in runs)), flush=True)

    a4_learns_L1 = bool(any(r["gate_met_at_bestval"] for r in per_rung["L1"]))
    l2_gate = bool(any(r["gate_met_at_bestval"] for r in per_rung["L2"]))

    verdicts = {
        "sentinel_verified": sentinel["sentinel_verified"],
        "arm_trainable_A4": memo["A4"]["arm_trainable"],
        "a5_fails_memorisation": a5_fails,
        "a4_learns_L1": a4_learns_L1,
    }
    if not memo["A4"]["arm_trainable"]:
        verdicts["rung_learnable_L2_2q"] = "UNDECIDABLE"
        verdicts["note"] = ("doc 24 SS4: A4 failed the memorisation control "
                            "and is an UNTRAINED INSTRUMENT; both rung "
                            "verdicts are UNDECIDABLE.")
        verdicts["a4_learns_L1"] = "UNDECIDABLE"
    elif not a4_learns_L1:
        verdicts["rung_learnable_L2_2q"] = "UNINTERPRETABLE"
        verdicts["note"] = L1_GATES_L2
    else:
        verdicts["rung_learnable_L2_2q"] = l2_gate
    verdicts["l2_gate_raw_value"] = l2_gate

    out = {
        "experiment": "exp45b",
        "spec": DESIGN_DOC,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "embeddings_sha256": emb_sha,
        "dataset_sha256": {r: rungs[r]["sha"] for r in ("L1", "L2")},
        "datasets_reused_verbatim": True,
        "grid": [{"lr": lr, "batch": b} for lr, b in grid],
        "frozen_grid_if_no_fallback": [{"lr": lr, "batch": b} for lr, b in GRID],
        "compute_fallback_applied": fallback,
        "measured_sec_per_step_b8": sec_per_step_b8,
        "measured_sec_per_epoch_L2_b8": sec_epoch_b8,
        "projected_full_grid_hours": proj_h,
        "compute_budget_hours": COMPUTE_BUDGET_H,
        "n_seeds": n_seeds, "max_epochs": max_epochs,
        "gate": {"train": TRAIN_GATE, "val": VAL_GATE},
        "param_counts": TwoQubitReferentModel("A4", verbs, 0).param_counts(),
        "sentinel": sentinel,
        "memorisation_controls": memo,
        "per_rung": per_rung,
        "verdicts": verdicts,
        "meta_notes": {"l1_gates_l2": L1_GATES_L2, "scope": SCOPE_NOTE,
                       "ansatz_caveat": ANSATZ_CAVEAT},
        "git_note": git_note(),
        "wall_time_sec": time.time() - t0,
    }
    print("\n[exp45b] VERDICTS")
    for k, v in verdicts.items():
        print("   %-28s %s" % (k, v))
    if args.smoke:
        print("\n[exp45b] smoke run -- NO results file written")
        return 0
    path = fresh_results_path("results_exp45b_phase1")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[exp45b] wrote %s (%.1f min)" % (path, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
