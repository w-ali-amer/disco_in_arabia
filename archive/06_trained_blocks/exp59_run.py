# -*- coding: utf-8 -*-
"""exp59_run.py -- doc 45: weight persistence + a provenance-clean curve.

This run TESTS NO NEW HYPOTHESIS (doc 45 SS1). It closes two blockers:
  B1  no run in the exp55-58 line persists weights (exp57_ckpt: 35 .json, 0 .pt)
  B2  the dose-response curve exists only in spread_sweep.log -- UNTRACKED, 3 seeds

Four anchors must reproduce their committed per-seed test_acc EXACTLY (doc 45
SS2); they are the product, not a fallback, so a miss HALTS assembly. The model
class and angle logic are IMPORTED from exp58_run verbatim (SS2.1) so the exp58
replications hold by construction; drift in this file's train() is what the
anchors detect.

Sweep points are DESCRIPTIVE ONLY. No monotonicity criterion is registered:
exp58 measured a RISE on the left (0.825 -> 0.852 -> 0.863 over w in
{0, 0.05, 0.15}), so a whole-curve bar would fail by construction (SS3.1).

  parallel:  python3 exp59_run.py --worker I --nworkers N   (disjoint pairs)
  assemble:  python3 exp59_run.py --assemble
"""
import argparse, copy, hashlib, json, math, os, platform, random, sys, time
import torch

torch.set_num_threads(1)          # doc 45 SS6.1: verified arithmetically inert

import exp42_compiler as comp
import exp41_story_generator as gen
import exp58_run as e58
from exp58_run import CondModel, build_angles, half_width, std_p1, evaluate
from exp42_run import fresh_results_path, git_note, grad_global_norm

DESIGN_DOC = ("qnlp_private_docs/45_exp59_artifact_and_provenance_run.md "
              "(9b0148c, FROZEN)")
DATA, VT_SHA = e58.DATA, e58.VT_SHA
LR, BATCH = e58.LR, e58.BATCH
N_SEEDS, MAX_EPOCHS = e58.N_SEEDS, e58.MAX_EPOCHS
TRAIN_GATE, VAL_GATE = e58.TRAIN_GATE, e58.VAL_GATE
RNG_BASE = e58.RNG_BASE
CKPT_DIR = "exp59_ckpt"
PI = math.pi

ANCHORS = ["A-tight", "A-wide", "A-zero", "A-scale"]
SWEEP = ["W0.05", "W0.35", "W0.60", "W0.90", "W1.30", "W1.90", "Wpi"]
CONDITIONS = ANCHORS + SWEEP

SPREAD = {"A-tight": 0.15, "A-wide": 2.40, "A-zero": 0.00, "A-scale": 2.40,
          "W0.05": 0.05, "W0.35": 0.35, "W0.60": 0.60, "W0.90": 0.90,
          "W1.30": 1.30, "W1.90": 1.90, "Wpi": PI}
CENTER = {c: PI for c in CONDITIONS}
ANGLE_SEEDS = {"A-tight": 57000, "A-wide": 57010, "A-zero": 57040,
               "A-scale": 57010,                      # doc 45 SS2
               "W0.05": 59010, "W0.35": 59020, "W0.60": 59030,
               "W0.90": 59040, "W1.30": 59050, "W1.90": 59060, "Wpi": 59070}
USE_SCALE = {"A-scale": True}

# doc 45 SS2.1: register exp59's conditions INTO exp58's tables so the imported
# build_angles / CondModel see them. The draw for A-tight/A-wide/A-scale is then
# bit-identical to exp57's M-tight/M-wide and exp58's M-wide-SCALE.
e58.SPREAD.update(SPREAD)
e58.CENTER.update(CENTER)
e58.ANGLE_SEEDS.update(ANGLE_SEEDS)
e58.USE_SCALE.update(USE_SCALE)

# doc 45 SS2, frozen per-seed test_acc targets
TARGETS = {
    "A-tight": [0.8133333333333334, 0.8333333333333334, 0.8733333333333333,
                0.9, 0.8633333333333333],
    "A-wide": [0.5733333333333334, 0.6333333333333333, 0.5566666666666666,
               0.5366666666666666, 0.5866666666666667],
    "A-zero": [0.81, 0.8333333333333334, 0.8033333333333333,
               0.8666666666666667, 0.8133333333333334],
    "A-scale": [0.86, 0.86, 0.8233333333333334, 0.86, 0.8933333333333333],
}
TARGET_SRC = {"A-tight": "exp57 M-tight (results_exp55b.json)",
              "A-wide": "exp57 M-wide (results_exp55b.json) = exp58 M-wide-REPRO",
              "A-zero": "exp58 Z-pole (results_exp58.json)",
              "A-scale": "exp58 M-wide-SCALE (results_exp58.json)"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def train(cond, seed, items, base, verbs, max_epochs, save_ckpt=True):
    """exp58_run.train with doc 45 SS5 weight persistence added. Any drift from
    exp58's loop is caught by the four anchors (SS4)."""
    ang, drawn = build_angles(cond, seed, base)
    data = {sp: comp.compile_all([i for i in items if i["split"] == sp], ang)
            for sp in ("train", "val", "test")}
    torch.manual_seed(seed)
    m = e58.make_model(cond, verbs, seed, ang)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    rng = random.Random(RNG_BASE + seed)
    best = {"val_acc": -1.0, "train_acc": -1.0, "state": None}
    t_start = time.time()
    gnorms = []
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        order = list(range(len(data["train"])))
        rng.shuffle(order)
        losses = []
        for s in range(0, len(order), BATCH):
            bat = [data["train"][i] for i in order[s:s + BATCH]]
            opt.zero_grad()
            bl = 0.0
            for cs in bat:
                vals = m.story_values(cs)
                bl = bl + torch.nn.functional.cross_entropy(
                    torch.log(vals + 1e-12).unsqueeze(0),
                    torch.tensor([cs.answer]))
            bl = bl / len(bat)
            bl.backward()
            gn = grad_global_norm(m)
            assert math.isfinite(gn)
            if ep <= 10:
                gnorms.append(gn)
            opt.step()
            losses.append(bl.item())
        with torch.no_grad():
            va, _ = evaluate(m, data["val"])
        if va > best["val_acc"]:
            with torch.no_grad():
                ta, _ = evaluate(m, data["train"])
            best = {"val_acc": va, "train_acc": ta,
                    "state": copy.deepcopy(m.state_dict())}
        if ep % 25 == 0 or ep == max_epochs:
            print("[%s s%d] ep %3d/%d loss=%.4f val=%.3f best=%.3f %.1fs"
                  % (cond, seed, ep, max_epochs, sum(losses) / len(losses),
                     va, best["val_acc"], time.time() - t0), flush=True)
    m.load_state_dict(best["state"])
    with torch.no_grad():
        te, corr = evaluate(m, data["test"])

    # ---- doc 45 SS5: persist the BEST-VAL state_dict (the selection the quoted
    # test_acc comes from), and hash it -- the .pt is untracked, so this sha256
    # in the committed JSON IS the provenance link the demo cites.
    ckpt_path = ckpt_sha = None
    if save_ckpt:
        ckpt_path = os.path.join(CKPT_DIR, "%s_s%d.pt" % (cond, seed))
        torch.save(best["state"], ckpt_path)
        ckpt_sha = sha256_file(ckpt_path)

    init_vals = [drawn[h] for h in e58.HUMANS]
    post, learned_scale = None, None
    if USE_SCALE.get(cond):
        learned_scale = float(m.scale.detach())
        post = {h: PI + learned_scale * (drawn[h] - PI) for h in e58.HUMANS}
    post_vals = [post[h] for h in e58.HUMANS] if post else init_vals
    return {"condition": cond, "seed": seed, "spread": SPREAD[cond],
            "drawn_angles": drawn, "post_angles": post,
            "learned_scale": learned_scale,
            "init_half_width": half_width(init_vals),
            "init_std_p1": std_p1(init_vals),
            "half_width": half_width(post_vals), "std_p1": std_p1(post_vals),
            "grad_norm_mean_first10": (sum(gnorms) / len(gnorms)
                                       if gnorms else None),
            "val_acc": best["val_acc"], "train_acc": best["train_acc"],
            "test_acc": te, "test_correct": corr,
            "ckpt_path": ckpt_path, "ckpt_sha256": ckpt_sha,
            "wall_sec": time.time() - t_start,
            "gate_met": bool(best["train_acc"] >= TRAIN_GATE
                             and best["val_acc"] >= VAL_GATE)}


def load_corpus():
    blob = json.load(open(DATA, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != VT_SHA:
        raise RuntimeError("VT provenance FAILED")
    base, _m, emb_sha = comp.load_embeddings()
    verbs = sorted(meta["verb_inventory"].keys())
    return items, base, emb_sha, verbs


def sentinel_check(conds, items, base, verbs):
    """doc 45 SS4 precondition: tied blocks -> identical statevectors -> exactly
    0.5000 under every condition. Halts before training."""
    pairs = {}
    for it in items:
        if it["split"] == "test":
            pairs.setdefault(it["pair_id"], []).append(it)
    tp = [sorted(v, key=lambda x: x["id"]) for v in pairs.values()
          if len(v) == 2][:60]
    out = {}
    for cond in conds:
        ang0, _d = build_angles(cond, 0, base)
        tm = e58.make_model(cond, verbs, 0, ang0)
        _canon = sorted(verbs)[0]
        _f = tm.verb_matrix
        tm.verb_matrix = (lambda _v, _c=_canon, _g=_f: _g(_c))
        ok = tot = 0
        with torch.no_grad():
            for a, b in tp:
                for it in (a, b):
                    cs = comp.compile_story(it, ang0)
                    ok += int(int(torch.argmax(
                        tm.story_values(cs)).item()) == cs.answer)
                    tot += 1
        out[cond] = ok / tot
        print("[exp59] tied sentinel %-8s = %.4f (must be 0.5000)"
              % (cond, out[cond]), flush=True)
    bad = [c for c, v in out.items() if abs(v - 0.5) > 1e-9]
    if bad:
        raise RuntimeError("tied sentinel != 0.5000 for %s -- forward-path bug"
                           % bad)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)
    items, base, emb_sha, verbs = load_corpus()

    all_pairs = [(c, s) for c in CONDITIONS for s in range(N_SEEDS)]

    if not args.assemble:
        mine = [p for i, p in enumerate(all_pairs)
                if i % args.nworkers == args.worker]
        if args.smoke:
            mine = [("A-zero", 0), ("A-scale", 0)]
        print("[exp59 w%d/%d] %d pairs: %s"
              % (args.worker, args.nworkers, len(mine),
                 [("%s:s%d" % p) for p in mine[:6]]), flush=True)
        sentinel_check(sorted({c for c, _ in mine}), items, base, verbs)
        maxep = 2 if args.smoke else MAX_EPOCHS
        for cond, seed in mine:
            ckj = os.path.join(CKPT_DIR, "%s_s%d.json" % (cond, seed))
            if os.path.exists(ckj) and not args.smoke:
                print("[ckpt] %s_s%d already done" % (cond, seed), flush=True)
                continue
            r = train(cond, seed, items, base, verbs, maxep,
                      save_ckpt=not args.smoke)
            if not args.smoke:
                json.dump(r, open(ckj, "w", encoding="utf-8"))
            print("[exp59] %s s%d test=%.4f gate=%s"
                  % (cond, seed, r["test_acc"], r["gate_met"]), flush=True)
        print("[exp59 w%d] done (%.1f min)"
              % (args.worker, (time.time() - t0) / 60), flush=True)
        return 0

    # ------------------------------------------------------------ assembly
    per_cond, missing = {}, []
    for cond in CONDITIONS:
        rs = []
        for s in range(N_SEEDS):
            ckj = os.path.join(CKPT_DIR, "%s_s%d.json" % (cond, s))
            if not os.path.exists(ckj):
                missing.append("%s_s%d" % (cond, s))
                continue
            rs.append(json.load(open(ckj, encoding="utf-8")))
        per_cond[cond] = rs
    if missing:
        print("[exp59] INCOMPLETE, %d missing: %s"
              % (len(missing), missing[:8]), flush=True)
        return 1

    anchors = {}
    for c in ANCHORS:
        got = [r["test_acc"] for r in per_cond[c]]
        exact = bool(got == TARGETS[c])
        anchors[c] = {"target": TARGETS[c], "got": got, "exact": exact,
                      "max_abs_diff": max(abs(a - b) for a, b in
                                          zip(got, TARGETS[c])),
                      "source": TARGET_SRC[c]}
        print("[exp59] anchor %-8s exact=%s  maxdiff=%.1e"
              % (c, exact, anchors[c]["max_abs_diff"]), flush=True)
    anchors_reproduce = all(a["exact"] for a in anchors.values())

    weights_ok = all(r.get("ckpt_sha256") and os.path.exists(r["ckpt_path"])
                     for rs in per_cond.values() for r in rs)

    curve = []
    for c in CONDITIONS:
        a = [r["test_acc"] for r in per_cond[c]]
        mu = sum(a) / len(a)
        sd = (sum((x - mu) ** 2 for x in a) / (len(a) - 1)) ** .5
        curve.append({"condition": c, "spread": SPREAD[c],
                      "learnable_scale": bool(USE_SCALE.get(c)),
                      "mean_test": mu, "sd_test": sd, "per_seed": a,
                      "gate_met": sum(1 for r in per_cond[c] if r["gate_met"]),
                      "mean_final_half_width":
                          sum(r["half_width"] for r in per_cond[c]) / len(a)})
    curve.sort(key=lambda x: (x["learnable_scale"], x["spread"]))

    verdicts = {"anchors_reproduce": anchors_reproduce,
                "weights_persisted": bool(weights_ok),
                "sweep_registered_as": "DESCRIPTIVE ONLY -- no verdict (doc 45 SS3)"}
    for c in ANCHORS:
        verdicts["anchor_exact_" + c] = anchors[c]["exact"]

    out = {"experiment": "exp59", "spec": DESIGN_DOC,
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "torch_num_threads": torch.get_num_threads(),
           "dataset_sha256": VT_SHA, "embeddings_sha256": emb_sha,
           "config": {"lr": LR, "batch": BATCH, "epochs": MAX_EPOCHS,
                      "n_seeds": N_SEEDS, "arm": "A3", "spread": SPREAD,
                      "center": CENTER, "angle_seeds": ANGLE_SEEDS},
           "anchors": anchors, "curve": curve,
           "per_condition": {c: [{k: v for k, v in r.items()
                                  if k != "test_correct"} for r in rs]
                             for c, rs in per_cond.items()},
           "verdicts": verdicts,
           "meta_notes": {
               "purpose": ("doc 45 SS1: replication + artifact run. Tests NO new "
                           "hypothesis and may not support any claim exp57/exp58 "
                           "did not already support. Closes B1 (no persisted "
                           "weights) and B2 (untracked 3-seed sweep log)."),
               "no_monotonicity_criterion": (
                   "doc 45 SS3.1: exp58 measured a RISE on the left "
                   "(0.825 -> 0.852 -> 0.863 over w in {0, 0.05, 0.15}), so a "
                   "whole-curve monotonicity bar would fail by construction. Any "
                   "rank correlation quoted from this curve must be scoped to the "
                   "decline region w >= 0.15 and labelled descriptive."),
               "weight_provenance": (
                   "doc 45 SS5: .pt files are untracked artifacts, so the sha256 "
                   "chain in this JSON is the provenance link the demo cites. "
                   "state_dicts are BEST-VAL, matching the quoted test_acc."),
               "thread_pinning": (
                   "doc 45 SS6.1: torch.set_num_threads(1). Verified inert "
                   "2026-08-20 -- 8 epochs at 4 vs 1 thread gave bitwise "
                   "identical parameters, max abs diff 0.0."),
               "scope": ("one rung (VT), one arm (A3), simulation; "
                         "compression_is_learnable remains a one-task-family "
                         "result (doc 44 SS7) and this run does not broaden it")},
           "git_note": git_note(), "wall_time_sec": time.time() - t0}

    print("\n[exp59] VERDICTS")
    for k, v in verdicts.items():
        print("   %-28s %s" % (k, v))
    if not anchors_reproduce:
        print("[exp59] HALT (doc 45 SS4): an anchor did not reproduce. The curve "
              "is NOT published and the weights are NOT licensed.", flush=True)
    print("\n[exp59] CURVE")
    for row in curve:
        print("   w=%-7.4f %-8s test %.4f +/- %.4f  gate %d/5  hw_final %.4f"
              % (row["spread"], row["condition"], row["mean_test"],
                 row["sd_test"], row["gate_met"], row["mean_final_half_width"]))
    p = fresh_results_path("results_exp59")
    assert "exp59" in p, "doc 45 SS6.3 filename guard: %s" % p
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[exp59] wrote %s (%.1f min)" % (p, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
