# -*- coding: utf-8 -*-
"""exp42r_run.py -- the exp42 REVIVAL runner (doc 22 SS3-SS7 + doc 65 SS2/SS7.4).

doc 65 SS7.4: "exp42 RUNS AS REGISTERED -- reviewer-consulted GO. No
amendment, no skip: the registered design with a ratified clock, at Mac
speed, with SS2.4's N(0, 0.1^2) init, all arms INCLUDING A2-TI -- the one
arm that has genuinely never run."

THIS FILE WRITES NO RESULTS FILE. It writes per-unit checkpoints under
exp42r_ckpt/ and control raw numbers under exp42r_ctl/. `exp42r_aggregate.py`
is the SOLE WRITER of results_exp42r.json (doc 65 SS2.3). The committed
August artefacts -- exp42_run.py, results_exp42.json, results_exp40b.json,
exp42_ckpt_*.pt -- are neither written nor deleted; every path this file
touches carries the `exp42r` marker.

ARMS (the whole search space; doc 22 SS7 forbids widening it)
  A1           exchange-symmetric core only             NEGATIVE CONTROL
  A2           dressed symmetric core                   THE BET
  A2TI         A2 + trainable noun intro                doc 65 SS2.1
  A3           unconstrained 2-qubit (Circuit-4)        upper reference
  B1           classical DisCoCirc, structure-matched   DECISIVE (H-Q)
  B2 / B2_generous  GRU over surface tokens + name augmentation
  B3           bag-of-embeddings + MLP                  must sit at chance

PROTOCOL
  5 seeds every arm, no seed selection (doc 22 SS3.3).
  Adam, lr 0.005, batch 1, 30 epochs -- exp40r/August parity. The epoch
  budget is defensible from this programme's own measurement rather than
  from habit: doc 65 SS7.2's H-condition ran 60 epochs and closed at
  honest - scrambled = +0.003, so budget was never the binding constraint.
  EARLY STOPPING / MODEL SELECTION ON THE IN-DISTRIBUTION VAL SPLIT ONLY,
  never on test_a/test_b/test_c -- doc 22 SS4's stated fix to Duneau et
  al.'s model-selection weakness. The compositional splits are touched
  exactly once per unit, at the end, to score the frozen selected model.
  Init: doc 65 SS2.4 via exp42r_common.apply_init (its per-arm policy and
  two flagged readings are documented in that module).

CONTROLS (raw numbers here; BOOLEANS in the aggregator)
  C1 dial-swap        1,000 sampled permutations of the verb->block
                      assignment, applied to story AND question together,
                      SHARDED across workers. The permutations are drawn
                      from Random(4242) in exp42_controls.c1_permutation_
                      test's exact idiom, so the union of the shards IS
                      that function's sequence.
  C2 informative null blocks trained on the verbset-A restricted-inventory
                      dataset, evaluated on verbset-B stories through the
                      fixed kind-matched bijection.
  C3 order-invariance SVO<->VSO re-rendering must change the compiled
                      circuit NOT AT ALL for the structured arms (machine
                      precision) and must move B2.
  C4 sentence-scramble  shuffling event order must destroy accuracy on the
                      generator-labelled order-dependent items.

FAILURE ISOLATION IS ARM-LEVEL, NEVER WORKER-LEVEL (doc 62 SS24's lesson).
Every job runs inside its own try/except; a failure writes
exp42r_ctl/FAILED_<job>.json with the traceback and the worker moves on. A
missing arm never blocks another arm, never kills a worker, and never
silently disappears -- the aggregator reads the FAILED markers and reports
the arm as incomplete WITH ITS REASON.

WORK QUEUE. Jobs are claimed with exclusive-create lock files, so the five
workers self-balance instead of being statically partitioned: an arm that
happens to be slow does not idle four boxes. A worker skips any job whose
output already exists, so a killed worker is relaunched with the identical
command line. Recovery from a crashed worker: delete exp42r_locks/ and
relaunch; finished units are kept, unfinished ones are re-claimed.

SMOKE GATE. `--smoke` runs the designed-to-fail battery of doc 65 SS2.3
and, on a clean pass, writes exp42r_smoke_PASSED.json. Workers REFUSE to
train without it.

Usage:
  python3 exp42r_run.py --smoke
  python3 exp42r_run.py --worker i --nworkers 5      # i = 0..4
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import traceback

import torch

import exp42_compiler as comp
import exp42_controls as ctl
import exp42r_common as K
from exp42_baselines import build_vocab, sample_name_mapping

SMOKE_MARKER = "exp42r_smoke_PASSED.json"
SMOKE_REPORT = "exp42r_smoke.json"

N_PERM = 1000                     # doc 22 SS6 C1: sampled permutations
C1_PERM_SEED = 4242               # exp42_controls.c1_permutation_test's seed
C1_SHARDS = 5
C1_ARM = "A2"                     # doc 22 SS7's gate arm; see c1_scope note

C2_TRAIN_DATA = "stories_exp42r_c2_trainA.json"
C2_EVAL_DATA = "stories_exp42r_c2_evalB.json"
C2_ARM = "A2"

BARRIER_TIMEOUT_SEC = 6 * 3600
BARRIER_POLL_SEC = 20

C1_SCOPE_NOTE = (
    "C1 is run on A2, the arm doc 22 SS7's kill-gate reads "
    "(`tier_q_passed` = quantum beats B1 AND content_earned). doc 22 SS6 "
    "registers one dial-swap control, not one per arm, and doc 65 SS2.1 "
    "gives A2-TI exactly one registered boolean (`trained_intro_helps`) "
    "which does not read C1. Running C1 on further arms would widen the "
    "registered control set after the fact; it is NOT done, and this "
    "sentence is why.")


# ==================================================================
# job queue
# ==================================================================
def claim(job_id):
    """Exclusive-create lock. True = this worker owns the job."""
    path = os.path.join(K.LOCK_DIR, job_id + ".lock")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, ("pid %d %s\n" % (os.getpid(), time.ctime())).encode())
    os.close(fd)
    return True


def settled(outputs, job_id):
    """A job is settled if it produced its outputs OR recorded its failure."""
    return (all(os.path.exists(p) for p in outputs)
            or os.path.exists(K.failed_path(job_id)))


def record_failure(job_id, exc, tag):
    """Arm-level isolation: the failure is DATA, not a dead worker."""
    payload = {"job": job_id, "tag": tag, "error": repr(exc),
               "traceback": traceback.format_exc(),
               "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "isolation": "doc 62 SS24: gate failures are ARM-LEVEL. This "
                            "job is abandoned, the worker continues, and the "
                            "aggregator reports the arm as incomplete WITH "
                            "THIS REASON rather than silently omitting it."}
    with open(K.failed_path(job_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("[FAIL] %s: %r (isolated; worker continues)" % (job_id, exc),
          flush=True)


# ==================================================================
# data
# ==================================================================
def prepare(log=True):
    """Provenance gate + compile everything, identically in every worker."""
    items, meta, angles, emb_sha, prov = K.provenance_gate()
    verbs = sorted(meta["verb_inventory"])
    by = comp.split_items(items)
    compiled = {k: comp.compile_all(v, angles, with_tokens=True)
                for k, v in by.items()}
    vocab = build_vocab(cs.tokens for cs in
                        compiled["train"] + compiled["val"]
                        + compiled["test_union"])
    union_ids = [cs.item_id for cs in compiled["test_union"]]
    if log:
        print("[data] %s sha=%s.. emb=%s.. verbs=%d train=%d val=%d "
              "union=%d vocab=%d"
              % (K.DATA_PATH, prov["dataset"]["sha256"][:16], emb_sha[:16],
                 len(verbs), len(compiled["train"]), len(compiled["val"]),
                 len(union_ids), len(vocab)), flush=True)
    return {"items": items, "meta": meta, "angles": angles, "prov": prov,
            "verbs": verbs, "by": by, "compiled": compiled, "vocab": vocab,
            "union_ids": union_ids, "emb_sha": emb_sha}


# ==================================================================
# training
# ==================================================================
def evaluate(model, compiled):
    correct = [1 if model.predict(cs) == cs.answer else 0 for cs in compiled]
    return sum(correct) / len(correct), correct


def grad_global_norm(model):
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            tot += p.grad.norm().item() ** 2
    return math.sqrt(tot)


def train_unit(arm, seed, data, cfg, splits=("test_a", "test_b", "test_c")):
    """One (arm, seed). Returns (model, detail). Selection on ID val ONLY."""
    t0 = time.time()
    model, init_spec = K.build_model(arm, cfg["verbs"], data["vocab"], seed,
                                     data["angles"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = random.Random(9000 + seed)
    train, val = data["compiled"]["train"], data["compiled"]["val"]
    augment = arm in ("B2", "B2_generous")
    meta = data["meta"]

    best = {"val_acc": -1.0, "train_acc": None, "epoch": -1, "state": None}
    losses_e, gnorms_e, runacc_e, val_e = [], [], [], []
    first = False
    for epoch in range(1, cfg["epochs"] + 1):
        te = time.time()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses, gnorms, run_correct = [], [], 0
        for start in range(0, len(order), cfg["batch"]):
            batch = [train[i] for i in order[start:start + cfg["batch"]]]
            opt.zero_grad()
            bl = 0.0
            for cs in batch:
                if augment:
                    mp = sample_name_mapping(meta["humans"], meta["objects"],
                                             rng)
                    loss, pred = model.story_loss(cs, mapping=mp)
                else:
                    loss, pred = model.story_loss(cs)
                bl = bl + loss
                if pred == cs.answer:
                    run_correct += 1
            bl = bl / len(batch)
            bl.backward()
            gn = grad_global_norm(model)
            if not math.isfinite(gn):
                raise RuntimeError("non-finite gradient norm (%s seed %d "
                                   "epoch %d)" % (arm, seed, epoch))
            if not first:
                if gn <= 0.0:
                    raise RuntimeError("zero gradient on the first batch "
                                       "(%s seed %d)" % (arm, seed))
                first = True
            gnorms.append(gn)
            opt.step()
            losses.append(bl.item())

        val_acc, _ = evaluate(model, val)
        losses_e.append(sum(losses) / len(losses))
        gnorms_e.append(sum(gnorms) / len(gnorms))
        runacc_e.append(run_correct / len(train))
        val_e.append(val_acc)
        if val_acc > best["val_acc"]:                     # ID VAL ONLY
            ctr, _ = evaluate(model, train)
            best = {"val_acc": val_acc, "train_acc": ctr, "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
        print("[%s s%d] epoch %2d/%d loss=%.4f train(run)=%.3f val=%.3f "
              "gn=%.3e best@%d(val=%.3f) %.1fs"
              % (arm, seed, epoch, cfg["epochs"], losses_e[-1], runacc_e[-1],
                 val_acc, gnorms_e[-1], best["epoch"], best["val_acc"],
                 time.time() - te), flush=True)
        if best["val_acc"] >= 1.0 and best["train_acc"] >= 1.0:
            print("[%s s%d] early stop: ID train and val both 1.000"
                  % (arm, seed), flush=True)
            break

    model.load_state_dict(best["state"])
    detail = {
        "arm": arm, "seed": seed, "best_epoch": best["epoch"],
        "epochs_run": len(losses_e), "max_epochs": cfg["epochs"],
        "train_acc_at_best": best["train_acc"], "val_acc": best["val_acc"],
        "epoch_mean_losses": losses_e, "epoch_mean_grad_norms": gnorms_e,
        "epoch_running_train_acc": runacc_e, "epoch_val_acc": val_e,
        "init": init_spec, "param_counts": model.param_counts(),
        "model_selection": "in-distribution val split ONLY (doc 22 SS4); the "
                           "compositional splits are scored once, after "
                           "selection, on the frozen model",
        "name_augmentation": bool(augment),
        "test": {}, "wall_time_sec": round(time.time() - t0, 1),
    }
    for sp in splits:
        acc, corr = evaluate(model, data["compiled"][sp])
        detail["test"][sp] = {"accuracy": acc, "n": len(corr),
                              "per_item_correct": corr,
                              "item_ids": [cs.item_id
                                           for cs in data["compiled"][sp]]}
    ua, uc = evaluate(model, data["compiled"]["test_union"])
    detail["test"]["test_union"] = {"accuracy": ua, "n": len(uc)}
    detail["union_item_ids"] = data["union_ids"]
    detail["union_per_item_correct"] = uc
    detail["vocab_sha256"] = K.vocab_sha(data["vocab"])
    detail["dataset_sha256"] = data["prov"]["dataset"]["sha256"]
    detail["embeddings_sha256"] = data["emb_sha"]
    return model, detail


def job_main(arm, seed, data, cfg):
    ck_json, ck_pt = K.unit_paths(arm, seed)
    model, detail = train_unit(arm, seed, data, cfg)
    torch.save(model.state_dict(), ck_pt)
    with open(ck_json, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False)
    print("[unit] %s s%d done: val=%.3f union=%.3f (%.0fs)"
          % (arm, seed, detail["val_acc"],
             detail["test"]["test_union"]["accuracy"],
             detail["wall_time_sec"]), flush=True)


# ==================================================================
# C1 -- dial-swap, sharded
# ==================================================================
def c1_permutations(verbs, n_perm=N_PERM, seed=C1_PERM_SEED):
    """exp42_controls.c1_permutation_test's draw, reproduced EXACTLY so the
    union of the shards is that function's sequence (same rng, same
    identity-exclusion loop, same order)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n_perm):
        p = verbs[:]
        while True:
            rng.shuffle(p)
            if p != verbs:
                break
        out.append(list(p))
    return out


def load_arm_models(arm, data, seeds=5):
    models, details = [], []
    for s in range(seeds):
        cj, cp = K.unit_paths(arm, s)
        if not (os.path.exists(cj) and os.path.exists(cp)):
            raise RuntimeError("C-stage prerequisite missing: %s / %s -- arm "
                               "%s did not complete" % (cj, cp, arm))
        with open(cj, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("vocab_sha256") != K.vocab_sha(data["vocab"]):
            raise RuntimeError("vocab drift for %s seed %d" % (arm, s))
        if d.get("union_item_ids") != data["union_ids"]:
            raise RuntimeError("union item-id drift for %s seed %d" % (arm, s))
        m, _spec = K.build_model(arm, data["verbs"], data["vocab"], s,
                                 data["angles"])
        m.load_state_dict(torch.load(cp, weights_only=True))
        m.eval()
        models.append(m)
        details.append(d)
    return models, details


def job_c1_shard(shard, data, seeds=5):
    models, _d = load_arm_models(C1_ARM, data, seeds=seeds)
    verbs = data["verbs"]
    union = data["compiled"]["test_union"]
    fast_delta = K.assert_fast_path_matches(models[0], union, verbs)

    perms = c1_permutations(verbs)
    lo = shard * (N_PERM // C1_SHARDS)
    hi = (shard + 1) * (N_PERM // C1_SHARDS) if shard < C1_SHARDS - 1 \
        else N_PERM
    orig = sum(K.fast_accuracy(m, union, verbs) for m in models) / len(models)
    t0 = time.time()
    accs = []
    for i in range(lo, hi):
        vmap = dict(zip(verbs, perms[i]))
        accs.append(sum(K.fast_accuracy(m, union, verbs, vmap)
                        for m in models) / len(models))
        if (i - lo) % 25 == 0:
            done = i - lo + 1
            rate = (time.time() - t0) / done
            print("[c1 shard %d] %d/%d perms  eta %.0fs"
                  % (shard, done, hi - lo, rate * (hi - lo - done)),
                  flush=True)
    out = {"control": "C1 trained-dial-swap (doc 22 SS6)", "arm": C1_ARM,
           "scope_note": C1_SCOPE_NOTE, "shard": shard, "n_shards": C1_SHARDS,
           "perm_index_lo": lo, "perm_index_hi": hi,
           "n_permutations_this_shard": len(accs),
           "n_permutations_total": N_PERM, "perm_seed": C1_PERM_SEED,
           "n_seeds": len(models), "n_union_items": len(union),
           "original_seed_mean_acc": orig,
           "permuted_seed_mean_accs": accs,
           "aggregation": "for every sampled permutation the SAME permutation "
                          "is applied to all 5 seeds and the accuracies are "
                          "averaged; the original seed-mean must STRICTLY "
                          "exceed the 95th percentile of the permuted "
                          "seed-means (exp42_controls.c1_permutation_test's "
                          "pre-run decision, unchanged)",
           "fast_path_max_abs_delta_vs_committed_forward": fast_delta,
           "wall_time_sec": round(time.time() - t0, 1)}
    with open(os.path.join(K.CTL_DIR, "c1_shard%d.json" % shard), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("[c1 shard %d] done: orig=%.4f perms=%d (%.0fs)"
          % (shard, orig, len(accs), out["wall_time_sec"]), flush=True)


# ==================================================================
# C2 -- informative null
# ==================================================================
def load_c2_dataset(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    items, meta = d["items"], d["meta"]
    sha = comp.canonical_sha256(items)
    if sha != meta["dataset_sha256"]:
        raise RuntimeError("C2 dataset %s hash mismatch" % path)
    return items, meta, sha


def job_c2(seed, data, cfg):
    items_a, meta_a, sha_a = load_c2_dataset(C2_TRAIN_DATA)
    items_b, meta_b, sha_b = load_c2_dataset(C2_EVAL_DATA)
    inv = dict(meta_a["verb_inventory"])
    inv.update(meta_b["verb_inventory"])
    va, vb = ctl.c2_verb_split(inv)
    assert sorted(meta_a["verb_inventory"]) == va, "trainA inventory drift"
    assert sorted(meta_b["verb_inventory"]) == vb, "evalB inventory drift"

    angles = data["angles"]
    by_a = comp.split_items(items_a)
    cdata = {"compiled": {k: comp.compile_all(v, angles, with_tokens=True)
                          for k, v in by_a.items()},
             "vocab": {}, "angles": angles, "meta": meta_a,
             "union_ids": [], "emb_sha": data["emb_sha"],
             "prov": {"dataset": {"sha256": sha_a}}}
    cdata["union_ids"] = [cs.item_id
                          for cs in cdata["compiled"]["test_union"]]
    test_b = [it for it in items_b if it["split"].startswith("test_")]
    compiled_b = comp.compile_all(test_b, angles)

    c2cfg = dict(cfg, verbs=va)
    model, detail = train_unit(C2_ARM, seed, cdata, c2cfg)
    ev = ctl.c2_eval_disjoint(model, compiled_b, inv)
    detail.pop("union_per_item_correct", None)
    for sp in detail["test"]:
        detail["test"][sp].pop("per_item_correct", None)
        detail["test"][sp].pop("item_ids", None)
    out = {"control": "C2 informative null (doc 22 SS6)", "seed": seed,
           "arm": C2_ARM,
           "criterion": "disjoint-inventory evaluation accuracy <= chance + "
                        "5 pp = 0.55",
           "train_dataset": {"path": C2_TRAIN_DATA, "sha256": sha_a,
                             "verbset_A": va},
           "eval_dataset": {"path": C2_EVAL_DATA, "sha256": sha_b,
                            "verbset_B": vb, "n_test_items": len(compiled_b)},
           "verb_bijection": ctl.c2_bijection(inv),
           "training": detail, "evaluation": ev}
    with open(os.path.join(K.CTL_DIR, "c2_s%d.json" % seed), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("[c2 s%d] trainA val=%.3f  disjoint-eval acc=%.3f"
          % (seed, detail["val_acc"], ev["accuracy"]), flush=True)


# ==================================================================
# C3 / C4
# ==================================================================
def job_c3(data):
    angles = data["angles"]
    struct, missing = {}, []
    for arm in ("A2", "A2TI", "B1"):
        try:
            m, _d = load_arm_models(arm, data, seeds=1)
            struct["%s_seed0" % arm] = m[0]
        except Exception as e:                            # noqa: BLE001
            missing.append("%s (%r)" % (arm, e))
    seq = None
    try:
        seq = load_arm_models("B2", data, seeds=1)[0][0]
    except Exception as e:                                # noqa: BLE001
        missing.append("B2 (%r)" % e)

    probe = next(it for it in data["by"]["test_a"]
                 if "SVO" in it["order_flags"] and "VSO" in it["order_flags"])
    out = {"control": "C3 order-invariance (doc 22 SS6)",
           "arms_missing": missing,
           "single_item": ctl.c3_order_invariance(struct, seq, probe, angles)}
    # a population check, not a single probe item
    pop = {name: {"n_items": 0, "max_abs_value_delta": 0.0}
           for name in struct}
    sample = [it for it in data["by"]["test_union"]
              if "SVO" in it["order_flags"] and "VSO" in it["order_flags"]][:200]
    with torch.no_grad():
        for it in sample:
            cs = comp.compile_story(it, angles, with_tokens=True)
            csf = comp.compile_story(comp.rerender_sentences(it, flip=True),
                                     angles, with_tokens=True)
            for name, m in struct.items():
                d = float((m.story_values(cs) - m.story_values(csf))
                          .abs().max().item())
                pop[name]["n_items"] += 1
                pop[name]["max_abs_value_delta"] = max(
                    pop[name]["max_abs_value_delta"], d)
    for name in pop:
        pop[name]["machine_precision_identical"] = bool(
            pop[name]["max_abs_value_delta"] == 0.0)
    out["population"] = pop
    out["population_note"] = (
        "the structured arms never see the surface -- they consume the "
        "generator's GOLD events -- so SVO<->VSO re-rendering must leave the "
        "compiled circuit and hence the readout values EXACTLY unchanged. "
        "This is an identity, and it is asserted at machine precision over "
        "%d items rather than on one probe." % len(sample))
    if seq is not None:
        out["B2_behavioral_delta"] = ctl.c3_b2_behavioral_delta(
            seq, data["by"]["test_union"][:200], angles)
    with open(os.path.join(K.CTL_DIR, "c3.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[c3] %s" % {k: v["max_abs_value_delta"] for k, v in pop.items()},
          flush=True)


def job_c4(data, seeds=5):
    out = {"control": "C4 sentence-scramble (doc 22 SS6)",
           "criterion": "scrambled accuracy on the generator-labelled "
                        "order-dependent items falls to <= 0.55",
           "per_arm": {}, "arms_missing": []}
    union = data["compiled"]["test_union"]
    for arm in ("A2", "A2TI", "A3", "B1"):
        try:
            models, _d = load_arm_models(arm, data, seeds=seeds)
        except Exception as e:                            # noqa: BLE001
            out["arms_missing"].append("%s (%r)" % (arm, e))
            continue
        per_seed = [ctl.c4_sentence_scramble(m, union) for m in models]
        for i, r in enumerate(per_seed):
            r["seed"] = i
        n = len(per_seed)
        out["per_arm"][arm] = {
            "per_seed": per_seed,
            "mean_accuracy_original": sum(r["accuracy_original"]
                                          for r in per_seed) / n,
            "mean_accuracy_scrambled": sum(r["accuracy_scrambled"]
                                           for r in per_seed) / n,
            "n_order_dependent_items": per_seed[0]["n_order_dependent_items"],
        }
        print("[c4] %s orig=%.3f scrambled=%.3f"
              % (arm, out["per_arm"][arm]["mean_accuracy_original"],
                 out["per_arm"][arm]["mean_accuracy_scrambled"]), flush=True)
    with open(os.path.join(K.CTL_DIR, "c4.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


# ==================================================================
# smoke battery (doc 65 SS2.3) -- designed-to-fail builds
# ==================================================================
def smoke_battery(args):
    """Every bar is declared IN SOURCE before any number exists.

    S1 mechanical      every registered arm builds under doc 65 SS2.4's init,
                       first-batch gradient is finite and non-zero, and the
                       loss decreases over 3 epochs on 20 items.
    S2 DESIGNED-TO-FAIL: leaking question renderer. exp41_datagen's `qleak`
                       sabotage renders the yes-class and no-class question
                       particles differently. A first-token detector must
                       reach >= 0.95 on it and must sit <= 0.55 on the real
                       dataset. If the leak detector cannot see a planted
                       leak, its silence on the real data means nothing.
    S3 DESIGNED-TO-FAIL: wrong-order compile. A deliberately broken compile
                       that SORTS each event's (agent, patient) wires
                       destroys thematic role; under it a swap-twin pair must
                       become bit-identical, so any model sits at exactly
                       chance. Under the real compile the twins must differ.
                       This is the check that argument order is load-bearing
                       all the way through the pipeline.
    S4 DESIGNED-TO-FAIL: symmetric-arm-above-chance synthetic. doc 22 SS3.2
                       makes "A1 above chance" the run-invalidating alarm, so
                       the alarm must be shown to have POWER: on a synthetic
                       task whose label is a leak A1 CAN see (a property of
                       the cast, not of argument order) A1 must climb above
                       0.65, while on the real swap twins it must sit at
                       EXACTLY 0.5 (its blocks commute with SWAP, so the
                       twins compile to the identical statevector).
    S5 init policy     per-(arm, seed) streams are distinct, every init is
                       finite, the sha is reproducible, and A2-TI's intro
                       angles equal the frozen embedding angles.
    S6 C1 fast path    the hoisted verb-matrix forward reproduces the
                       committed `story_values` to 1e-6.
    S7 alignment       the union item-id vector is deterministic and its
                       length matches doc 22 SS7's ">= 50 items, >= 500
                       preferred".
    """
    t0 = time.time()
    data = prepare()
    verbs, angles = data["verbs"], data["angles"]
    tests = []

    def add(name, passed, bar, **kw):
        rec = {"test": name, "bar": bar, "passed": bool(passed)}
        rec.update(kw)
        tests.append(rec)
        print("[smoke] %-34s %s  %s"
              % (name, "PASS" if passed else "FAIL",
                 {k: v for k, v in kw.items() if k != "detail"}), flush=True)

    # ---- S1 mechanical, under the REGISTERED init -------------------
    rng = random.Random(0)
    pool = data["compiled"]["train"][:]
    rng.shuffle(pool)
    tiny = ([c for c in pool if c.answer == 0][:10]
            + [c for c in pool if c.answer == 1][:10])
    tiny_val = data["compiled"]["val"][:12]
    mech = {}
    ok1 = True
    for arm in K.ARMS:
        sub = {"compiled": {"train": tiny, "val": tiny_val,
                            "test_a": tiny_val[:4], "test_b": tiny_val[4:8],
                            "test_c": tiny_val[8:12],
                            "test_union": tiny_val[:12]},
               "vocab": data["vocab"], "angles": angles, "meta": data["meta"],
               "union_ids": [c.item_id for c in tiny_val[:12]],
               "emb_sha": data["emb_sha"], "prov": data["prov"]}
        _m, d = train_unit(arm, 0, sub,
                           {"verbs": verbs, "lr": 0.005, "batch": 1,
                            "epochs": 3})
        ls, gs = d["epoch_mean_losses"], d["epoch_mean_grad_norms"]
        arm_ok = (all(math.isfinite(g) and g > 0 for g in gs)
                  and ls[-1] < ls[0])
        mech[arm] = {"params": d["param_counts"]["total_trained"],
                     "losses": ls, "grad_norms": gs,
                     "grads_finite_and_nonzero":
                         all(math.isfinite(g) and g > 0 for g in gs),
                     "loss_decreased": bool(ls[-1] < ls[0]), "passed": arm_ok}
        ok1 = ok1 and arm_ok
    add("S1 mechanical (all 8 arms)", ok1,
        "every arm: finite non-zero grads AND loss(epoch 3) < loss(epoch 1), "
        "under doc 65 SS2.4's N(0, 0.1^2) init", per_arm=mech)

    # ---- S2 leaking question renderer (designed-to-fail) ------------
    import exp41_datagen as dgen

    def first_token_detector(items):
        """Counts the question's first surface token per answer class and
        predicts the majority class for that token: the cheapest possible
        surface-leak probe."""
        tab = {}
        for it in items:
            t = it["question"]["surface"].split()[0]
            tab.setdefault(t, [0, 0])[0 if it["answer"] == "yes" else 1] += 1
        ok = sum(max(v) for v in tab.values())
        return ok / len(items)

    tiny_spec = {k: dict(v, Q1=20, Q2=8, Q0=(6 if k in ("train", "val") else 0))
                 for k, v in dgen.SPLIT_SPEC.items()}
    leak_items, _h = dgen.generate(770001, sabotage="qleak", spec=tiny_spec)
    leak_acc = first_token_detector(leak_items)
    real_acc = first_token_detector(data["items"])
    add("S2 leaking-question-renderer (fail-build)",
        leak_acc >= 0.95 and real_acc <= 0.55,
        "planted qleak detected at >= 0.95 AND real dataset <= 0.55",
        qleak_detector_accuracy=leak_acc,
        real_dataset_detector_accuracy=real_acc)

    # ---- twin census: which twins are PURE argument exchanges? -------
    # doc 22 SS3.2's A1 guarantee ("[U, SWAP] = 0, so the twins compile to
    # the identical statevector") holds only where the twin differs from
    # its partner by within-event argument exchange ALONE. The exp41r Q2
    # construction is not of that kind: flipping the previous event's
    # arguments RE-RESOLVES the pro-dropped subject, so the pro-drop event
    # lands on a DIFFERENT WIRE PAIR in the two twins (exp41_datagen lines
    # 326-334). This census measures the split rather than assuming it, and
    # S3/S4 condition their exact-identity bars on the pure subset.
    from dataclasses import replace as _replace

    def order_blind(cs):
        return [(v, min(x, y), max(x, y)) for v, x, y in cs.events]

    pairs = {}
    for cs in data["compiled"]["test_union"]:
        pairs.setdefault(cs.pair_id, []).append(cs)
    all_twins = [v for v in pairs.values() if len(v) == 2]
    eq_twins = [t for t in all_twins if order_blind(t[0]) == order_blind(t[1])]
    neq_twins = [t for t in all_twins
                 if order_blind(t[0]) != order_blind(t[1])]
    by_qtype = {}
    for t in all_twins:
        k = (t[0].qtype, "swap_equivalent"
             if order_blind(t[0]) == order_blind(t[1]) else "NOT_swap_equiv")
        by_qtype[k] = by_qtype.get(k, 0) + 1
    n_eq_items = 2 * len(eq_twins)
    n_neq_items = 2 * len(neq_twins)
    a1_ceiling = (0.5 * n_eq_items + 1.0 * n_neq_items) / \
        (n_eq_items + n_neq_items)
    twins = eq_twins[:60]
    m_a2, _s = K.build_model("A2", verbs, data["vocab"], 0, angles)
    real_delta, broken_delta = 0.0, 0.0
    with torch.no_grad():
        for a, b in twins:
            real_delta = max(real_delta, float(
                (m_a2.story_values(a) - m_a2.story_values(b))
                .abs().max().item()))
            ab = _replace(a, events=[(v, min(x, y), max(x, y))
                                     for v, x, y in a.events])
            bb = _replace(b, events=[(v, min(x, y), max(x, y))
                                     for v, x, y in b.events])
            broken_delta = max(broken_delta, float(
                (m_a2.story_values(ab) - m_a2.story_values(bb))
                .abs().max().item()))
    add("S3 wrong-order compile (fail-build)",
        broken_delta < 1e-6 and real_delta > 1e-3,
        "on SWAP-EQUIVALENT twins: an order-blind compile makes them "
        "identical (< 1e-6) while the real compile separates them (> 1e-3)",
        n_twin_pairs=len(twins), real_compile_max_twin_delta=real_delta,
        order_blind_compile_max_twin_delta=broken_delta,
        conditioned_on="swap-equivalent twins only (see S8)")

    # ---- S4 symmetric-arm-above-chance synthetic (designed-to-fail) --
    import exp40r_trainer as tr
    leaky = []
    for cs in tiny + data["compiled"]["train"][20:120]:
        # label = a cast property A1 CAN read (parity of the queried wires),
        # NOT argument order. A1 must be able to climb here.
        lab = (cs.question[1] + cs.question[2]) % 2
        leaky.append(_replace(cs, answer=lab))
    m_a1, _s = K.build_model("A1", verbs, data["vocab"], 0, angles)
    traj = tr.train_with_trajectory(m_a1, leaky, {}, lambda c: c.answer,
                                    seed=0, epochs=25, lr=0.02, batch_size=1,
                                    log_prefix="[smoke S4] ")
    a1_leaky = traj["max_clean_train_acc"]
    m_a1b, _s = K.build_model("A1", verbs, data["vocab"], 0, angles)
    a1_twin_delta = 0.0
    with torch.no_grad():
        for a, b in twins:
            a1_twin_delta = max(a1_twin_delta, float(
                (m_a1b.story_values(a) - m_a1b.story_values(b))
                .abs().max().item()))
    a1_neq_delta = 0.0
    with torch.no_grad():
        for a, b in neq_twins[:60]:
            a1_neq_delta = max(a1_neq_delta, float(
                (m_a1b.story_values(a) - m_a1b.story_values(b))
                .abs().max().item()))
    add("S4 symmetric-arm alarm has power (fail-build)",
        a1_leaky >= 0.65 and a1_twin_delta < 1e-6,
        "A1 >= 0.65 on a synthetic leak it CAN see, AND A1's SWAP-EQUIVALENT "
        "twins identical to < 1e-6 on the real data",
        a1_max_train_acc_on_planted_leak=a1_leaky,
        a1_max_twin_value_delta_swap_equivalent=a1_twin_delta,
        a1_max_twin_value_delta_NOT_swap_equivalent=a1_neq_delta,
        conditioned_on="swap-equivalent twins only (see S8)")

    # ---- S5 init policy ---------------------------------------------
    seen, finite, specs = {}, True, {}
    for arm in K.ARMS:
        for s in (0, 1):
            m, spec = K.build_model(arm, verbs, data["vocab"], s, angles)
            seen[(arm, s)] = spec["stream_seed"]
            specs["%s_s%d" % (arm, s)] = {
                "stream_seed": spec["stream_seed"],
                "sha": spec["init_vector_sha256"][:16],
                "policy": spec["policy"]}
            finite = finite and all(torch.isfinite(p).all()
                                    for p in m.parameters())
    m_ti, spec_ti = K.build_model("A2TI", verbs, data["vocab"], 0, angles)
    # 1e-6, not 0: the trained intro angles are float32 nn.Parameters and the
    # frozen embedding angles are float64, so the seed is exact only to
    # float32 resolution (~1e-7 here). Bar declared with its reason.
    intro_dev = max(abs(float(m_ti.intro_angle[n]) - float(angles[n]))
                    for n in m_ti.nouns)
    intro_ok = intro_dev < 1e-6
    _m2, spec2 = K.build_model("A2", verbs, data["vocab"], 0, angles)
    _m3, spec3 = K.build_model("A2", verbs, data["vocab"], 0, angles)
    add("S5 init policy (doc 65 SS2.4)",
        len(set(seen.values())) == len(seen) and finite and intro_ok
        and spec2["init_vector_sha256"] == spec3["init_vector_sha256"],
        "streams distinct per (arm, seed); all inits finite; the init sha is "
        "reproducible; A2-TI intro angles == the frozen embedding angles to "
        "float32 resolution (< 1e-6)",
        n_distinct_streams=len(set(seen.values())), n_units=len(seen),
        all_finite=bool(finite), a2ti_intro_equals_embeddings=bool(intro_ok),
        a2ti_intro_max_abs_deviation=intro_dev,
        init_sha_reproducible=bool(spec2["init_vector_sha256"]
                                   == spec3["init_vector_sha256"]),
        detail=specs)

    # ---- S6 C1 fast path --------------------------------------------
    d_a2 = K.assert_fast_path_matches(m_a2, data["compiled"]["test_union"],
                                      verbs)
    d_ti = K.assert_fast_path_matches(m_ti, data["compiled"]["test_union"],
                                      verbs)
    add("S6 C1 fast path == committed forward", max(d_a2, d_ti) < 1e-6,
        "hoisted verb-matrix forward reproduces exp42_models.story_values "
        "to < 1e-6", A2_max_abs_delta=d_a2, A2TI_max_abs_delta=d_ti)

    # ---- S7 alignment ------------------------------------------------
    d2 = prepare(log=False)
    add("S7 union alignment deterministic",
        d2["union_ids"] == data["union_ids"] and len(data["union_ids"]) >= 500,
        "the union item-id vector is reproducible across processes and has "
        ">= 500 items (doc 22 SS7)",
        n_union_items=len(data["union_ids"]),
        deterministic=bool(d2["union_ids"] == data["union_ids"]))

    # ---- S8 twin census + the A1 alarm's real ceiling ----------------
    add("S8 twin census (A1 alarm ceiling)",
        len(eq_twins) + len(neq_twins) == len(all_twins) and len(eq_twins) > 0,
        "the census is complete and non-degenerate. THIS TEST RECORDS A "
        "MEASUREMENT, and the measurement is a FLAG, not a pass/fail on the "
        "experiment",
        n_twin_pairs=len(all_twins), n_swap_equivalent=len(eq_twins),
        n_NOT_swap_equivalent=len(neq_twins), by_qtype={
            "%s/%s" % k: v for k, v in sorted(by_qtype.items())},
        a1_arithmetic_ceiling_on_union=a1_ceiling,
        FLAG=("doc 22 SS3.2 registers 'A1 MUST sit at chance on swap twins' "
              "and makes A1-above-chance a RUN-INVALIDATING alarm, on the "
              "physics argument that [U_A1, SWAP] = 0 makes a twin pair "
              "compile to the identical statevector. That argument is EXACT "
              "for every Q1 and Q0 twin (%d pairs) and DOES NOT APPLY to any "
              "Q2 twin (%d pairs): exp41_datagen's Q2 flips the previous "
              "event's arguments, which RE-RESOLVES the pro-dropped subject, "
              "so the pro-drop event lands on a different WIRE PAIR in the "
              "two twins -- a difference a SWAP-symmetric model can see "
              "without any leak. A1 can therefore reach %.4f on the union "
              "while doc 22 SS3.2's physics still holds exactly where it "
              "claims to. The registered union boolean is computed AS "
              "REGISTERED, and the swap-equivalent-subset boolean is "
              "reported beside it so the alarm can be read correctly rather "
              "than fired mechanically. FLAGGED FOR THE USER; no registered "
              "criterion is changed here."
              % (len(eq_twins), len(neq_twins), a1_ceiling)))

    all_ok = all(t["passed"] for t in tests)
    report = {
        "component": "exp42r smoke battery (doc 65 SS2.3)",
        "twin_census": {
            "n_twin_pairs": len(all_twins),
            "n_swap_equivalent": len(eq_twins),
            "n_NOT_swap_equivalent": len(neq_twins),
            "by_qtype": {"%s/%s" % k: v for k, v in sorted(by_qtype.items())},
            "a1_arithmetic_ceiling_on_union": a1_ceiling,
            "swap_equivalent_item_ids": sorted(
                cs.item_id for t in eq_twins for cs in t)},
        "designed_to_fail_builds": ["S2 leaking question renderer",
                                    "S3 wrong-order compile",
                                    "S4 symmetric-arm-above-chance synthetic"],
        "bars_declared_in_source_before_any_run": True,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "provenance": data["prov"],
        "tests": tests, "all_passed": all_ok,
        "wall_time_sec": round(time.time() - t0, 1),
    }
    with open(SMOKE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("\n[smoke] %d/%d passed -- %s (%.0f s)"
          % (sum(t["passed"] for t in tests), len(tests),
             "SMOKE GATE OPEN" if all_ok else "SMOKE GATE CLOSED",
             report["wall_time_sec"]))
    if all_ok:
        with open(SMOKE_MARKER, "w", encoding="utf-8") as f:
            json.dump({"all_passed": True, "report": SMOKE_REPORT,
                       "when": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime())}, f, indent=1)
        print("[smoke] wrote %s -- workers may train." % SMOKE_MARKER)
        return 0
    print("[smoke] NO marker written: exp42r_run.py refuses to train.")
    return 1


# ==================================================================
# worker
# ==================================================================
def main_jobs(seeds):
    jobs = []
    for arm in K.ARMS:
        for s in range(seeds):
            jobs.append(("main:%s:s%d" % (arm, s), arm, s))
    return jobs


def run_worker(args):
    if not os.path.exists(SMOKE_MARKER):
        sys.exit("[w%d] REFUSING to train: %s absent. Run "
                 "`exp42r_run.py --smoke` first (doc 65 SS2.3: the smoke "
                 "battery runs BEFORE any training)."
                 % (args.worker, SMOKE_MARKER))
    torch.set_num_threads(args.threads)
    K.ensure_dirs(K.CKPT_DIR, K.CTL_DIR, K.LOCK_DIR)
    data = prepare()
    cfg = {"verbs": data["verbs"], "lr": args.lr, "batch": args.batch,
           "epochs": args.epochs}
    w = args.worker
    print("[w%d] queue start (%d threads)" % (w, args.threads), flush=True)

    # ---------------- phase 1: main units + C2 units -----------------
    arms = ([a for a in K.ARMS if a in set(args.arms.split(","))]
            if args.arms else list(K.ARMS))
    main = [j for j in main_jobs(args.seeds) if j[1] in arms]
    c2_ok = os.path.exists(C2_TRAIN_DATA) and os.path.exists(C2_EVAL_DATA)
    if not c2_ok:
        print("[w%d] C2 datasets absent (%s / %s) -- C2 jobs skipped; the "
              "aggregator will report C2 as not run."
              % (w, C2_TRAIN_DATA, C2_EVAL_DATA), flush=True)
    queue = []
    # interleave so no worker takes a whole slow arm before the others start
    for k in range(max(args.seeds, 1)):
        for jid, arm, s in main:
            if s == k:
                queue.append((jid, "main", (arm, s)))
        if c2_ok and k < args.seeds:
            queue.append(("c2:s%d" % k, "c2", (k,)))

    for jid, kind, payload in queue:
        outs = (list(K.unit_paths(*payload)) if kind == "main"
                else [os.path.join(K.CTL_DIR, "c2_s%d.json" % payload[0])])
        if settled(outs, jid) or not claim(jid):
            continue
        try:
            if kind == "main":
                job_main(payload[0], payload[1], data, cfg)
            else:
                job_c2(payload[0], data, cfg)
        except Exception as e:                            # noqa: BLE001
            record_failure(jid, e, kind)

    # ---------------- barrier ----------------------------------------
    print("[w%d] phase 1 exhausted; waiting on the other workers" % w,
          flush=True)
    deadline = time.time() + args.barrier_timeout
    pending = None
    while time.time() < deadline:
        pending = [jid for jid, _a, _s in main
                   if not settled(list(K.unit_paths(_a, _s)), jid)]
        if not pending:
            break
        time.sleep(BARRIER_POLL_SEC)
    if pending:
        print("[w%d] BARRIER TIMEOUT after %.0fs with %d units unsettled: "
              "%s -- proceeding to the control stage anyway; prerequisites "
              "that are genuinely missing become arm-level FAILED markers, "
              "never a hung worker."
              % (w, args.barrier_timeout, len(pending), pending[:8]),
              flush=True)

    # ---------------- phase 2: C1 shards, C3, C4 ---------------------
    phase2 = [("c1:shard%d" % k, "c1", (k,),
               [os.path.join(K.CTL_DIR, "c1_shard%d.json" % k)])
              for k in range(C1_SHARDS)]
    phase2 += [("c3", "c3", (), [os.path.join(K.CTL_DIR, "c3.json")]),
               ("c4", "c4", (), [os.path.join(K.CTL_DIR, "c4.json")])]
    # rotate so the five workers do not all try the same shard first
    phase2 = phase2[w % len(phase2):] + phase2[:w % len(phase2)]
    for jid, kind, payload, outs in phase2:
        if settled(outs, jid) or not claim(jid):
            continue
        try:
            if kind == "c1":
                job_c1_shard(payload[0], data, seeds=args.seeds)
            elif kind == "c3":
                job_c3(data)
            else:
                job_c4(data, seeds=args.seeds)
        except Exception as e:                            # noqa: BLE001
            record_failure(jid, e, kind)

    print("[exp42r w%d] done" % w, flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--nworkers", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--arms", default=None,
                    help="comma-separated arm subset (integration testing "
                         "only; the run of record uses all registered arms)")
    ap.add_argument("--barrier-timeout", type=float,
                    default=BARRIER_TIMEOUT_SEC)
    args = ap.parse_args()

    if args.smoke:
        torch.set_num_threads(args.threads)
        K.ensure_dirs(K.CKPT_DIR, K.CTL_DIR, K.LOCK_DIR)
        return smoke_battery(args)
    if args.worker is None:
        sys.exit("give --smoke or --worker i")
    return run_worker(args)


if __name__ == "__main__":
    sys.exit(main())
