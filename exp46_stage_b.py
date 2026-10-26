# -*- coding: utf-8 -*-
"""exp46_stage_b.py -- doc 29: wazn Stage B.

Does trained verb-block geometry cluster by Arabic morphological pattern?

Stage A (doc 26) tested ONE wazn-mismatched pair against two matched pairs
over 40 restarts and returned inconclusive (k=16/40, p=0.231) -- it had no
degrees of freedom in the verb-sampling direction. Stage B removes exactly
that limit: 4 verbs per pattern x 3 patterns, so within-pattern and
between-pattern distances are both estimable and the question becomes DO
BLOCKS CLUSTER BY PATTERN rather than IS ONE PAIR AN OUTLIER.

Selectional profile is CONSTANT BY CONSTRUCTION (all 12 verbs transitive
with a human direct object), so it cannot contribute to any distance.

Pipeline, all gates blocking:
  1. FIT GATE (doc 29 SS4): A3 on W1, 8 seeds x the frozen Phase-1 grid,
     val-only selection. w1_learnable = gate met (train>=0.8 AND val>=0.7)
     in >= 5 of 8 seeds. Threshold calibrated against the only per-seed
     rates ever measured for this gate (Phase 2 at the selected config:
     A2 5/5, A3 4/5) and frozen. If false, exp46 is VOID -- doc 26 SS4.3:
     structure recovered from poorly-fit solutions is weak evidence.
  2. DISTANCES: behavioural substitution distance, exp39's notion.
     d(v,w) = 1/2[drop(v->w) + drop(w->v)] on the W1 test split, per
     contributing seed, averaged into one 12x12 matrix. 66 pairs: 18
     within-pattern, 48 between.
  3. STATISTIC: Delta = mean(between) - mean(within), tested by EXACT
     enumeration of all 12!/(4!^3 3!) = 5775 partitions of the verbs into
     three unlabeled groups of four. No sampling, no seed in the p-value.
  4. CONTROL (blocking): the whole pipeline on 200 draws of random KAK
     blocks, where verb identity carries no information.
     random_block_null_calibrated iff the fraction of draws with p<0.05
     lies in [0.01, 0.10]. If the metric is miscalibrated, exp46 is VOID.

Verdicts (doc 29 SS5, frozen):
  wazn_clusters            iff p < 0.05
  wazn_clustering_ruled_out iff Delta_observed <= median of the null
  wazn_stage_b_inconclusive otherwise

Per-seed results are DESCRIPTIVE ONLY: the criterion runs on the
seed-averaged matrix, fixed so a favourable seed cannot be promoted later.

Scope (doc 29 SS7): a positive is NOT causal -- wazn cannot be isolated from
meaning on real verbs. It does not revive exp33's specificity claim or
reverse exp39's verdict, which stand as computed against the criteria they
were registered on. No quantum reading whatsoever.
"""

import argparse
import copy
import itertools
import json
import math
import os
import platform
import random
import sys
import time

import torch

import exp42_compiler as comp
from exp42_models import QuantumStoryModel
from exp42_run import fresh_results_path, git_note, grad_global_norm
from exp43a_model import KakStoryModel

DESIGN_DOC = ("qnlp_private_docs/29_exp46_wazn_stage_b.md "
              "(commits 1ac3819 criteria + e9c69d0 sign-off, FROZEN)")
DATA = "stories_exp46_W1.json"
EXPECTED_SHA = \
    "db2f27e510601492968a068ad57e24d0dd77bf64d2bd634982daf8f6bdf45325"
CKPT_DIR = "exp46_ckpt"
ARM = "A3"                                  # doc 29 SS4 primary
GRID = [(0.005, 1), (0.005, 8), (0.02, 1), (0.02, 8)]
N_SEEDS = 8
MAX_EPOCHS = 60
TRAIN_GATE, VAL_GATE = 0.8, 0.7
FIT_GATE_SEEDS = 5                          # >= 5 of 8, frozen
N_CONTROL_DRAWS = 200
CONTROL_BAND = (0.01, 0.10)
ALPHA = 0.05
RNG_BASE = 46000
CONTROL_SEED_BASE = 460000

FIT_GATE_NOTE = (
    "doc 29 SS4, frozen: 5-of-8 is calibrated against the only per-seed rates "
    "ever measured for this gate -- Phase 2 at the selected config on the "
    "frozen 6-verb L1 gave A2 5/5 and A3 4/5. A 62.5% bar sits BELOW A3's "
    "observed 80%, leaving room for a collapsed seed without voiding the run, "
    "while refusing to build geometry on a mostly-failed fit. It is a genuine "
    "VOID risk: the inventory doubles from 6 verbs to 12 and nothing has been "
    "measured at that scale. Accepted knowingly, not hedged after the fact.")

SCOPE_NOTE = (
    "doc 29 SS7: (1) NOT causal -- wazn cannot be isolated from meaning on "
    "real verbs; a positive says block geometry ALIGNS with a morphological "
    "split, not that morphology causes it. (2) One inventory, 12 verbs, one "
    "rung, one task family, simulation only. (3) Does NOT revive exp33's "
    "specificity claim or reverse exp39's verdict -- those stand as computed "
    "against the criteria they were registered on; exp46 tests a question "
    "exp39's all-Form-I design COULD NOT POSE. (4) No quantum reading "
    "whatsoever: this is a geometry probe of trained parameters and bears on "
    "neither tier-q nor hardware. (5) The selectional profile is fixed, so "
    "the result is conditional on it.")


def log(m):
    print(m, flush=True)


def load_w1():
    with open(DATA, encoding="utf-8") as f:
        d = json.load(f)
    items, meta = d["items"], d["meta"]
    sha = comp.canonical_sha256(items)
    if sha != meta["dataset_sha256"] or sha != EXPECTED_SHA:
        raise RuntimeError("W1 provenance FAILED: %s vs %s vs pin %s"
                           % (sha, meta["dataset_sha256"], EXPECTED_SHA))
    return items, meta, sha


def evaluate(model, compiled):
    correct = [1 if model.predict(cs) == cs.answer else 0 for cs in compiled]
    return sum(correct) / len(correct), correct


def train_run(seed, lr, batch, data, verbs, max_epochs, arm=ARM):
    torch.manual_seed(seed)
    model = QuantumStoryModel(arm, verbs, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(RNG_BASE + seed)
    train = data["train"]
    best = {"val_acc": -1.0, "train_acc": -1.0, "epoch": -1, "state": None}
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses, run_correct = [], 0
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
            assert math.isfinite(gn), "non-finite grad s%d" % seed
            opt.step()
            losses.append(bloss.item())
        val_acc, _ = evaluate(model, data["val"])
        if val_acc > best["val_acc"]:
            ta, _ = evaluate(model, train)
            best = {"val_acc": val_acc, "train_acc": ta, "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
        log("[s%d lr%.3f b%d] ep %2d/%d loss=%.4f train(run)=%.3f val=%.3f "
            "best@%d(%.3f) %.1fs"
            % (seed, lr, batch, epoch, max_epochs,
               sum(losses) / len(losses), run_correct / len(train), val_acc,
               best["epoch"], best["val_acc"], time.time() - t0))
        if best["val_acc"] >= 1.0 and best["train_acc"] >= 1.0:
            break
    model.load_state_dict(best["state"])
    test_acc, _ = evaluate(model, data["test"])
    return model, {
        "seed": seed, "lr": lr, "batch": batch, "best_epoch": best["epoch"],
        "val_acc": best["val_acc"], "train_acc": best["train_acc"],
        "test_acc": test_acc,
        "gate_met": bool(best["train_acc"] >= TRAIN_GATE
                         and best["val_acc"] >= VAL_GATE)}


# ------------------------------------------------------------- distances
def _cached_eval(model, compiled, vmap=None):
    """Accuracy with verb matrices built once. Mirrors model.predict."""
    import exp42_sim_adapter as sim
    vm = vmap or {}
    with torch.no_grad():
        mats = {v: model.verb_matrix(vm.get(v, v)) for v in model.verbs}
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


def substitution_matrix(model, compiled, verbs):
    """d(v,w) = 1/2[drop(v->w) + drop(w->v)], exp39's behavioural notion.

    Only stories touching v or w can change under the substitution, so the
    evaluation is restricted to that subset and the drop is rescaled to the
    full split -- exact, not an approximation, because every other story's
    prediction is bitwise unchanged."""
    n = len(verbs)
    idx = {v: i for i, v in enumerate(verbs)}
    D = [[0.0] * n for _ in range(n)]
    base_all = _cached_eval(model, compiled)
    subsets = {}
    for v in verbs:
        subsets[v] = [cs for cs in compiled
                      if any(e[0] == v for e in cs.events)
                      or v in cs.question[0]]
    for a, b in itertools.combinations(verbs, 2):
        sub = [cs for cs in compiled
               if any(e[0] in (a, b) for e in cs.events)
               or a in cs.question[0] or b in cs.question[0]]
        if not sub:
            continue
        base_sub = _cached_eval(model, sub)
        d1 = base_sub - _cached_eval(model, sub, {a: b})
        d2 = base_sub - _cached_eval(model, sub, {b: a})
        scale = len(sub) / len(compiled)
        d = 0.5 * (d1 + d2) * scale
        D[idx[a]][idx[b]] = D[idx[b]][idx[a]] = d
    return D, base_all


# ------------------------------------------------------------- statistic
def all_partitions(n=12, k=4):
    """All 12!/(4!^3 3!) = 5775 partitions into 3 UNLABELED groups of 4.
    Canonical: element 0 leads group A; the smallest unused element leads
    group B. No duplicates, no group-label symmetry."""
    elems = list(range(n))
    out = []
    rest0 = elems[1:]
    for a_rest in itertools.combinations(rest0, k - 1):
        A = (0,) + a_rest
        rem = [e for e in elems if e not in A]
        lead = rem[0]
        for b_rest in itertools.combinations(rem[1:], k - 1):
            B = (lead,) + b_rest
            C = tuple(e for e in rem if e not in B)
            out.append((A, B, C))
    return out


def delta(D, part):
    within, between = [], []
    grp = {}
    for gi, g in enumerate(part):
        for e in g:
            grp[e] = gi
    n = len(D)
    for i in range(n):
        for j in range(i + 1, n):
            (within if grp[i] == grp[j] else between).append(D[i][j])
    return sum(between) / len(between) - sum(within) / len(within)


def permutation_test(D, parts, observed_part):
    obs = delta(D, observed_part)
    nulls = [delta(D, p) for p in parts]
    ge = sum(1 for v in nulls if v >= obs)
    nulls_sorted = sorted(nulls)
    median = nulls_sorted[len(nulls_sorted) // 2]
    return {"delta_observed": obs, "p": ge / len(parts),
            "n_partitions": len(parts), "null_median": median,
            "null_max": nulls_sorted[-1], "null_min": nulls_sorted[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", default=ARM, choices=("A3", "A2"),
                    help="doc 29 SS4: A3 is the primary arm; A2 is the "
                         "SECONDARY, DESCRIPTIVE-ONLY replication and can "
                         "never carry a criterion.")
    args = ap.parse_args()
    arm = args.arm
    t0 = time.time()
    if not args.smoke:
        os.makedirs(CKPT_DIR, exist_ok=True)
    n_seeds = 2 if args.smoke else N_SEEDS
    max_epochs = 2 if args.smoke else MAX_EPOCHS
    grid = GRID[-1:] if args.smoke else GRID
    n_control = 5 if args.smoke else N_CONTROL_DRAWS

    items, meta, sha = load_w1()
    angles, _em, emb_sha = comp.load_embeddings()
    verbs = sorted(meta["verb_inventory"].keys())
    wazn = meta["wazn_groups"]
    pattern_of = {v: p for p, vs in wazn.items() for v in vs}
    data = {sp: comp.compile_all([i for i in items if i["split"] == sp], angles)
            for sp in ("train", "val", "test")}
    log("[exp46] W1 sha=%s verbs=%d train/val/test=%d/%d/%d"
        % (sha[:12], len(verbs), len(data["train"]), len(data["val"]),
           len(data["test"])))

    observed_part = tuple(tuple(sorted(verbs.index(v) for v in wazn[p]))
                          for p in ("I", "II", "III"))
    log("[exp46] observed partition (verb indices): %s" % (observed_part,))

    # ---------- 1. FIT GATE
    per_seed, models = [], []
    for seed in range(n_seeds):
        cands = []
        for lr, batch in grid:
            key = "w1_%s_s%d_lr%s_b%d" % (arm, seed, lr, batch)
            ckj, ckp = (os.path.join(CKPT_DIR, key + ".json"),
                        os.path.join(CKPT_DIR, key + ".pt"))
            if not args.smoke and os.path.exists(ckj) and os.path.exists(ckp):
                with open(ckj, encoding="utf-8") as f:
                    d = json.load(f)
                m = QuantumStoryModel(arm, verbs, seed)
                m.load_state_dict(torch.load(ckp, weights_only=True))
                log("[ckpt] %s loaded" % key)
            else:
                m, d = train_run(seed, lr, batch, data, verbs, max_epochs,
                                 arm=arm)
                if not args.smoke:
                    with open(ckj, "w", encoding="utf-8") as f:
                        json.dump(d, f, ensure_ascii=False)
                    torch.save(m.state_dict(), ckp)
            cands.append((d, m))
        best_d, best_m = min(cands, key=lambda c: (-c[0]["val_acc"],
                                                   c[0]["lr"], c[0]["batch"]))
        per_seed.append(best_d)
        models.append(best_m)
        log("[exp46] seed %d selected lr=%.3f b=%d val=%.3f train=%.3f "
            "test=%.3f gate=%s" % (seed, best_d["lr"], best_d["batch"],
                                   best_d["val_acc"], best_d["train_acc"],
                                   best_d["test_acc"], best_d["gate_met"]))

    n_gate = sum(1 for d in per_seed if d["gate_met"])
    w1_learnable = bool(n_gate >= (1 if args.smoke else FIT_GATE_SEEDS))
    log("[exp46] FIT GATE: %d/%d seeds met -> w1_learnable=%s"
        % (n_gate, n_seeds, w1_learnable))

    base = {
        "experiment": "exp46 wazn Stage B",
        "spec": DESIGN_DOC,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "dataset_sha256": sha, "embeddings_sha256": emb_sha,
        "sign_off": meta["sign_off"], "wazn_groups": wazn,
        "arm": arm,
        "arm_role": ("PRIMARY (doc 29 SS4)" if arm == "A3" else
                     "SECONDARY, DESCRIPTIVE ONLY -- never carries a "
                     "criterion (doc 29 SS4)"),
        "n_seeds": n_seeds, "grid": grid,
        "fit_gate": {"bar_seeds": FIT_GATE_SEEDS, "n_met": n_gate,
                     "w1_learnable": w1_learnable, "note": FIT_GATE_NOTE},
        "per_seed": per_seed,
        "meta_notes": {"scope": SCOPE_NOTE},
        "git_note": git_note(),
    }
    if not w1_learnable:
        base["verdicts"] = {
            "exp46_void": True,
            "void_reason": ("doc 29 SS4/SS6: the fit gate failed (%d/%d seeds "
                            "< %d). No distance is reported and no clustering "
                            "statistic is computed. The finding is that the "
                            "12-verb rung was not learned -- itself "
                            "informative, since the frozen 6-verb L1 was."
                            % (n_gate, n_seeds, FIT_GATE_SEEDS)),
            "wazn_clusters": None, "wazn_clustering_ruled_out": None,
            "wazn_stage_b_inconclusive": None}
        base["wall_time_sec"] = time.time() - t0
        log("[exp46] VOID -- fit gate failed; no geometry computed.")
        if not args.smoke:
            p = fresh_results_path("results_exp46_stage_b_%s" % arm)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(base, f, ensure_ascii=False, indent=1)
            log("[exp46] wrote %s" % p)
        return 1

    # ---------- 2/3. DISTANCES + STATISTIC
    contributing = [m for m, d in zip(models, per_seed) if d["gate_met"]]
    log("[exp46] computing substitution distances over %d contributing seeds"
        % len(contributing))
    mats, per_seed_stats = [], []
    parts = all_partitions()
    assert len(parts) == 5775, len(parts)
    for i, m in enumerate(contributing):
        D, b = substitution_matrix(m, data["test"], verbs)
        mats.append(D)
        st = permutation_test(D, parts, observed_part)
        per_seed_stats.append({"seed_index": i, "base_acc": b, **st})
        log("[exp46]   seed %d: base=%.3f delta=%.4f p=%.4f"
            % (i, b, st["delta_observed"], st["p"]))
    n = len(verbs)
    Davg = [[sum(M[i][j] for M in mats) / len(mats) for j in range(n)]
            for i in range(n)]
    primary = permutation_test(Davg, parts, observed_part)
    log("[exp46] PRIMARY (seed-averaged): delta=%.4f p=%.4f (median null %.4f)"
        % (primary["delta_observed"], primary["p"], primary["null_median"]))

    # ---------- 4. CONTROL (blocking)
    log("[exp46] random-KAK control, %d draws..." % n_control)
    ctrl_p, sig = [], 0
    for k in range(n_control):
        cm = KakStoryModel(verbs, CONTROL_SEED_BASE + k)
        D, _b = substitution_matrix(cm, data["test"], verbs)
        st = permutation_test(D, parts, observed_part)
        ctrl_p.append(st["p"])
        sig += int(st["p"] < ALPHA)
        if (k + 1) % 25 == 0:
            log("[exp46]   control %d/%d, sig so far %d" % (k + 1, n_control, sig))
    frac = sig / n_control
    calibrated = bool(CONTROL_BAND[0] <= frac <= CONTROL_BAND[1])
    log("[exp46] control: %d/%d significant = %.3f, band %s -> calibrated=%s"
        % (sig, n_control, frac, CONTROL_BAND, calibrated))

    if not calibrated:
        verdicts = {
            "exp46_void": True,
            "void_reason": ("doc 29 SS5/SS6: random-block control NOT "
                            "calibrated (fraction %.3f outside %s). The "
                            "distance metric is biased under exchangeability, "
                            "so Stage B is VOID and no clustering number is "
                            "quoted." % (frac, CONTROL_BAND)),
            "random_block_null_calibrated": False,
            "wazn_clusters": None, "wazn_clustering_ruled_out": None,
            "wazn_stage_b_inconclusive": None}
    else:
        clusters = bool(primary["p"] < ALPHA)
        ruled_out = bool(primary["delta_observed"] <= primary["null_median"])
        verdicts = {
            "exp46_void": False,
            "random_block_null_calibrated": True,
            "wazn_clusters": clusters,
            "wazn_clustering_ruled_out": bool(ruled_out and not clusters),
            "wazn_stage_b_inconclusive": bool(not clusters and not ruled_out),
            "p": primary["p"], "delta_observed": primary["delta_observed"]}

    base.update({
        "observed_partition_indices": observed_part,
        "verbs_sorted": verbs,
        "pattern_of": pattern_of,
        "distance_matrix_seed_averaged": Davg,
        "primary_statistic": primary,
        "per_seed_statistics_descriptive": per_seed_stats,
        "control_random_kak": {
            "n_draws": n_control, "n_significant": sig,
            "fraction_significant": frac, "band": list(CONTROL_BAND),
            "calibrated": calibrated, "p_values": ctrl_p},
        "verdicts": verdicts,
        "wall_time_sec": time.time() - t0,
    })
    log("\n[exp46] VERDICTS")
    for k, v in verdicts.items():
        log("   %-34s %s" % (k, v))
    if args.smoke:
        log("[exp46] smoke -- no results file")
        return 0
    p = fresh_results_path("results_exp46_stage_b_%s" % arm)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=1)
    log("[exp46] wrote %s (%.1f min)" % (p, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
