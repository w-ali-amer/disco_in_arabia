# -*- coding: utf-8 -*-
"""exp58_run.py -- doc 44: is the referent channel USED at all on VT, and can
its conditioning be LEARNED?

doc 44 SS1 (erratum E1): exp57's M-wide-ROT was a NULL INSTRUMENT. One global
Euler rotation is a Bloch-sphere ISOMETRY -- it cannot change pairwise angular
separation, so it cannot compress. Measured: the learned rotation moved pairwise
angles by <= 6e-5 rad. `optimisation_explains = false` stands as computed, but
H-optim was never tested. M-wide-SCALE is the instrument it required.

doc 44 SS2 (erratum E2): Q2 could not have returned TRUE on VT. All 900 twin
pairs share cast and question and have opposite answers, so any referent-only
signal scores exactly 0.500 (V1_bag_of_embeddings = 0.5 confirms it). Z-pole /
Z-equator convert that by-construction argument into a measured one.

Conditions (doc 44 SS4), arm A3, 5 seeds x 150 epochs:
  M-wide-REPRO   w=2.40, exp57's M-wide draw, no model change  -> integrity
  Z-pole         w=0.00 about pi                               -> zero-info PRIMARY
  Z-equator      w=0.00 about pi/2                             -> zero-info control
  M-wide-SCALE   w=2.40 + ONE learnable scalar a (init 1.0)    -> H-optim
  M-wide-LEARN   w=2.40 + 24 learnable per-noun angles         -> learn the encoding
"""
import argparse, copy, json, math, os, platform, random, sys, time
import torch

import exp42_compiler as comp
import exp42_sim_adapter as sim
import exp41_story_generator as gen
from exp42_models import QuantumStoryModel
from exp42_run import (fresh_results_path, git_note, grad_global_norm,
                       paired_permutation_pvalue)

DESIGN_DOC = ("qnlp_private_docs/44_exp58_zero_information_and_learnable_"
              "conditioning.md (4fca1ba, FROZEN)")
DATA = "stories_exp51_VT.json"
VT_SHA = "8aa71d1e286392ad96fe7a5f77dc8afb7998e5594b8abde67bbaef024e9d1064"
PHI_ART = "exp55_noun_embeddings_phi.json"
PHI_SHA = "2cb5b3cb5feef3a41fbcc7249eccb20c0a0ac4d5ef2f43de5f96b83c37b33242"
CKPT_DIR = "exp58_ckpt"
LR, BATCH = 0.02, 8                 # doc 44 SS8, quoted from exp57_run.py
N_SEEDS, MAX_EPOCHS = 5, 150
TRAIN_GATE, VAL_GATE = 0.8, 0.7
RNG_BASE = 55000
RHO_MIN_GAP = 0.15

CONDITIONS = ["M-wide-REPRO", "Z-pole", "Z-equator",
              "M-wide-SCALE", "M-wide-LEARN"]
SPREAD = {"M-wide-REPRO": 2.40, "Z-pole": 0.00, "Z-equator": 0.00,
          "M-wide-SCALE": 2.40, "M-wide-LEARN": 2.40}
CENTER = {"M-wide-REPRO": math.pi, "Z-pole": math.pi,
          "Z-equator": math.pi / 2, "M-wide-SCALE": math.pi,
          "M-wide-LEARN": math.pi}
# doc 44 SS4: REPRO/SCALE/LEARN share exp57's M-wide draw (57010) so a rescue is
# measured against the identical input configuration. Z conditions have zero
# spread, so their seed is immaterial; recorded for the record.
ANGLE_SEEDS = {"M-wide-REPRO": 57010, "M-wide-SCALE": 57010,
               "M-wide-LEARN": 57010, "Z-pole": 57040, "Z-equator": 57050}
USE_SCALE = {"M-wide-SCALE": True}
USE_LEARN = {"M-wide-LEARN": True}

# doc 44 SS4, frozen from results_exp55b.json (exp57, identical protocol)
REF_M_TIGHT = [0.8133333333333334, 0.8333333333333334, 0.8733333333333333,
               0.9, 0.8633333333333333]
REF_M_WIDE = [0.5733333333333334, 0.6333333333333333, 0.5566666666666666,
              0.5366666666666666, 0.5866666666666667]
REF_TIGHT_MEAN = sum(REF_M_TIGHT) / len(REF_M_TIGHT)
REF_WIDE_MEAN = sum(REF_M_WIDE) / len(REF_M_WIDE)

HUMANS = sorted(gen.HUMANS)                      # doc 44 SS4: 24 per-noun params
HIDX = {h: i for i, h in enumerate(HUMANS)}


class CondModel(QuantumStoryModel):
    """A3 with doc-44's two angle-path modifications, both off by default.

    OFF: the intro applies sim.ry(angle) to the raw float, byte-identical to
    exp57_run.py:100 -- this is what makes harness_reproduces a real test.
    SS8.2: new Parameters are deterministic, so no extra global RNG is drawn."""

    def __init__(self, arm, verbs, seed, use_scale=False, use_learn=False,
                 init_angles=None):
        super().__init__(arm, verbs, seed)
        self.use_scale = use_scale
        self.use_learn = use_learn
        if use_scale:
            self.scale = torch.nn.Parameter(torch.tensor(1.0))   # identity
        if use_learn:
            init = [float(init_angles[h]) for h in HUMANS]
            self.theta = torch.nn.Parameter(torch.tensor(init))  # init = drawn

    def _angle(self, noun, angle):
        if self.use_learn:
            if noun not in HIDX:
                raise RuntimeError("LEARN: non-human referent on a wire: %r"
                                   % noun)
            return self.theta[HIDX[noun]]
        if self.use_scale:
            return math.pi + self.scale * (angle - math.pi)
        return angle                                   # raw float, exp57 path

    def story_values(self, cs, verb_map=None):
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)
            return cache[v]

        psi = sim.init_state(cs.K)
        for w, noun, angle in cs.intro:
            psi = sim.apply_gate(psi, sim.ry(self._angle(noun, angle)), (w,))
        for v, wa, wp in cs.events:
            psi = sim.apply_gate(psi, block(v), (wa, wp))
        vseq, qa, qp = cs.question
        qw = (qa, qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)
        vp = sim.prob_all_zero_on(
            sim.apply_gate(psi, self.q_pos.matrix().conj().T, qw), qw)
        vn = sim.prob_all_zero_on(
            sim.apply_gate(psi, self.q_neg.matrix().conj().T, qw), qw)
        return torch.stack([vp, vn])


def circ_mean(vals):
    return math.atan2(sum(math.sin(v) for v in vals) / len(vals),
                      sum(math.cos(v) for v in vals) / len(vals))


def wrapped_dev(t, tbar):
    return math.atan2(math.sin(t - tbar), math.cos(t - tbar))


def half_width(vals):
    """doc 44 SS5: THE compression statistic. Same w convention as SPREAD."""
    tbar = circ_mean(vals)
    return max(abs(wrapped_dev(t, tbar)) for t in vals)


def std_p1(vals):
    """Descriptive only. Fixes doc 43 SS9.3's label: this is NOT angular."""
    p = [math.sin(t / 2) ** 2 for t in vals]
    mu = sum(p) / len(p)
    return (sum((x - mu) ** 2 for x in p) / len(p)) ** .5


def build_angles(cond, seed, base):
    """doc 44 SS8.3: center is a parameter (Z-equator needs pi/2). Z draws
    uniform(-0.0, 0.0) and so lands exactly on center for every referent."""
    ang = dict(base)
    rng = random.Random(ANGLE_SEEDS[cond] + seed)
    w, c = SPREAD[cond], CENTER[cond]
    ang.update({h: c + rng.uniform(-w, w) for h in gen.HUMANS})
    drawn = {h: ang[h] for h in gen.HUMANS}
    return ang, drawn


def evaluate(m, compiled):
    c = [1 if int(torch.argmax(m.story_values(cs)).item()) == cs.answer else 0
         for cs in compiled]
    return sum(c) / len(c), c


def make_model(cond, verbs, seed, ang):
    return CondModel("A3", verbs, seed,
                     use_scale=USE_SCALE.get(cond, False),
                     use_learn=USE_LEARN.get(cond, False), init_angles=ang)


def train(cond, seed, items, base, verbs, max_epochs):
    ang, drawn = build_angles(cond, seed, base)
    data = {sp: comp.compile_all([i for i in items if i["split"] == sp], ang)
            for sp in ("train", "val", "test")}
    torch.manual_seed(seed)
    m = make_model(cond, verbs, seed, ang)
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
        if ep % 10 == 0 or ep == max_epochs:
            print("[%s s%d] ep %3d/%d loss=%.4f val=%.3f best=%.3f %.1fs"
                  % (cond, seed, ep, max_epochs, sum(losses) / len(losses),
                     va, best["val_acc"], time.time() - t0), flush=True)
    m.load_state_dict(best["state"])
    with torch.no_grad():
        te, corr = evaluate(m, data["test"])

    # ---- doc 44 SS8.5: post-training angle map, literal, both statistics
    init_vals = [drawn[h] for h in HUMANS]
    post, learned_scale = None, None
    if USE_LEARN.get(cond):
        post = {h: float(m.theta[HIDX[h]].detach()) for h in HUMANS}
    elif USE_SCALE.get(cond):
        learned_scale = float(m.scale.detach())
        post = {h: math.pi + learned_scale * (drawn[h] - math.pi)
                for h in HUMANS}
    post_vals = [post[h] for h in HUMANS] if post else init_vals
    return {"condition": cond, "seed": seed, "drawn_angles": drawn,
            "post_angles": post, "learned_scale": learned_scale,
            "init_half_width": half_width(init_vals),
            "init_std_p1": std_p1(init_vals),
            "half_width": half_width(post_vals), "std_p1": std_p1(post_vals),
            "grad_norm_mean_first10": (sum(gnorms) / len(gnorms)
                                       if gnorms else None),
            "val_acc": best["val_acc"], "train_acc": best["train_acc"],
            "test_acc": te, "test_correct": corr,
            "wall_sec": time.time() - t_start,
            "gate_met": bool(best["train_acc"] >= TRAIN_GATE
                             and best["val_acc"] >= VAL_GATE)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--drop", default="", help="comma-separated conditions")
    args = ap.parse_args()
    t0 = time.time()
    if not args.smoke:
        os.makedirs(CKPT_DIR, exist_ok=True)
    nseeds = 1 if args.smoke else N_SEEDS
    maxep = 2 if args.smoke else MAX_EPOCHS
    conds = [c for c in CONDITIONS if c not in args.drop.split(",")]

    blob = json.load(open(DATA, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != VT_SHA:
        raise RuntimeError("VT provenance FAILED")
    base, _m, emb_sha = comp.load_embeddings()
    art = json.load(open(PHI_ART, encoding="utf-8"))
    if art["embeddings_sha256"] != PHI_SHA:
        raise RuntimeError("phi artifact provenance FAILED")
    verbs = sorted(meta["verb_inventory"].keys())
    print("[exp58] VT sha ok | %d humans | conditions=%s"
          % (len(HUMANS), conds), flush=True)

    # ---- doc 44 SS5 precondition: tied blocks compile both twins to the
    # IDENTICAL statevector whatever the angles are, so every condition --
    # including SCALE and LEARN -- must read exactly 0.5000. A deviation is a
    # forward-path bug and halts the run BEFORE training.
    vt_pairs = {}
    for it in items:
        if it["split"] == "test":
            vt_pairs.setdefault(it["pair_id"], []).append(it)
    tp = [sorted(v, key=lambda x: x["id"]) for v in vt_pairs.values()
          if len(v) == 2][:60]
    sentinel = {}
    for cond in conds:
        ang0, _d = build_angles(cond, 0, base)
        tm = make_model(cond, verbs, 0, ang0)
        _canon = sorted(verbs)[0]
        _base_vm = tm.verb_matrix
        tm.verb_matrix = (lambda _v, _c=_canon, _f=_base_vm: _f(_c))
        ok = tot = 0
        with torch.no_grad():
            for a, b in tp:
                for it in (a, b):
                    cs = comp.compile_story(it, ang0)
                    ok += int(int(torch.argmax(
                        tm.story_values(cs)).item()) == cs.answer)
                    tot += 1
        sentinel[cond] = ok / tot
        print("[exp58] tied sentinel %-13s = %.4f (must be 0.5000)"
              % (cond, sentinel[cond]), flush=True)
    bad = [c for c, v in sentinel.items() if abs(v - 0.5) > 1e-9]
    if bad:
        print("[exp58] HALT: tied sentinel != 0.5000 for %s -- forward-path bug"
              % bad, flush=True)
        return 1

    per_cond = {}
    for cond in conds:
        rs = []
        for s in range(nseeds):
            key = "%s_s%d" % (cond, s)
            ckj = os.path.join(CKPT_DIR, key + ".json")
            if not args.smoke and os.path.exists(ckj):
                rs.append(json.load(open(ckj, encoding="utf-8")))
                print("[ckpt] %s loaded" % key, flush=True)
                continue
            r = train(cond, s, items, base, verbs, maxep)
            if not args.smoke:
                json.dump(r, open(ckj, "w", encoding="utf-8"))
            rs.append(r)
        per_cond[cond] = rs
        acc = [r["test_acc"] for r in rs]
        mu = sum(acc) / len(acc)
        sd = (sum((x - mu) ** 2 for x in acc) / max(len(acc) - 1, 1)) ** .5
        print("[exp58] %-13s test %.4f +/- %.4f  gate %d/%d  hw %.4f  %s"
              % (cond, mu, sd, sum(1 for r in rs if r["gate_met"]), len(rs),
                 rs[0]["half_width"], [round(x, 3) for x in acc]), flush=True)

    def seedmean(c):
        a = [r["test_acc"] for r in per_cond[c]]
        return sum(a) / len(a)

    # ---- doc 44 SS5: harness integrity decides which references rho uses
    repro = {}
    if "M-wide-REPRO" in per_cond and not args.smoke:
        got = [r["test_acc"] for r in per_cond["M-wide-REPRO"]]
        repro = {"exp57_M_wide_per_seed": REF_M_WIDE, "exp58_repro_per_seed": got,
                 "max_abs_diff": max(abs(a - b) for a, b in zip(got, REF_M_WIDE))
                 if len(got) == len(REF_M_WIDE) else None}
        repro["exact"] = bool(len(got) == len(REF_M_WIDE)
                              and all(a == b for a, b in zip(got, REF_M_WIDE)))
    harness_reproduces = bool(repro.get("exact"))
    wide_mean = (REF_WIDE_MEAN if harness_reproduces
                 else (seedmean("M-wide-REPRO") if "M-wide-REPRO" in per_cond
                       else REF_WIDE_MEAN))
    gap = REF_TIGHT_MEAN - wide_mean
    rho = {"reference_gap": gap, "references_valid": bool(gap >= RHO_MIN_GAP),
           "wide_reference_used": ("exp57 frozen" if harness_reproduces
                                   else "exp58 M-wide-REPRO (SUBSTITUTED -- "
                                        "harness did not reproduce exactly)"),
           "tight_reference_used": "exp57 frozen (M-tight, mean %.6f)"
                                   % REF_TIGHT_MEAN}
    for c in ("Z-pole", "Z-equator", "M-wide-SCALE", "M-wide-LEARN"):
        if c in per_cond:
            rho["rho_" + c] = ((seedmean(c) - wide_mean) / gap
                               if gap >= RHO_MIN_GAP else None)
            rho["half_width_" + c] = per_cond[c][0]["half_width"]

    verdicts = {"harness_reproduces": harness_reproduces}
    if rho["references_valid"]:
        z = [rho.get("rho_Z-pole"), rho.get("rho_Z-equator")]
        verdicts["zero_information_suffices"] = bool(
            any(x is not None and x >= 0.90 for x in z))
        verdicts["zero_information_locations_passing"] = [
            n for n, x in zip(("Z-pole", "Z-equator"), z)
            if x is not None and x >= 0.90]
        for c, name in (("M-wide-SCALE", "compression_is_learnable"),
                        ("M-wide-LEARN", "encoding_is_learnable")):
            if c in per_cond:
                hw = [r["half_width"] for r in per_cond[c]]
                verdicts[name] = bool(rho.get("rho_" + c, 0) >= 0.50
                                      and max(hw) < 0.50)
                verdicts[name + "_max_half_width"] = max(hw)
    else:
        for k in ("zero_information_suffices", "compression_is_learnable",
                  "encoding_is_learnable"):
            verdicts[k] = "UNDECIDABLE-on-references"
    verdicts.update({"gate_rate_%s" % c:
                     "%d/%d" % (sum(1 for r in per_cond[c] if r["gate_met"]),
                                len(per_cond[c])) for c in per_cond})

    out = {"experiment": "exp58", "spec": DESIGN_DOC,
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "dataset_sha256": VT_SHA,
           "embeddings_sha256": emb_sha, "phi_artifact_sha256": PHI_SHA,
           "config": {"lr": LR, "batch": BATCH, "epochs": maxep,
                      "n_seeds": nseeds, "arm": "A3",
                      "angle_seeds": ANGLE_SEEDS, "spread": SPREAD,
                      "center": CENTER},
           "per_condition": {c: [{k: v for k, v in r.items()
                                  if k != "test_correct"} for r in rs]
                             for c, rs in per_cond.items()},
           "tied_sentinel": sentinel, "harness_reproduction": repro,
           "rho": rho, "verdicts": verdicts,
           "meta_notes": {
               "erratum_E1": ("doc 44 SS1: exp57's M-wide-ROT was a NULL "
                              "INSTRUMENT for H-optim -- a global 1q unitary is "
                              "a Bloch isometry and cannot compress; measured "
                              "pairwise-angle change <= 6e-5 rad. "
                              "optimisation_explains=false stands as computed "
                              "but does NOT refute H-optim. M-wide-SCALE is the "
                              "instrument it required."),
               "erratum_E2": ("doc 44 SS2: Q2 could not have returned TRUE on "
                              "VT -- all 900 twin pairs share cast and question "
                              "with opposite answers, so any referent-only "
                              "signal scores exactly 0.500 "
                              "(V1_bag_of_embeddings = 0.5). "
                              "semantics_helps_when_conditioned=false stands, "
                              "but is evidence about the RUNG, not embeddings."),
               "statistic_naming": ("doc 44 SS5: half_width = "
                                    "max|wrapped_dev(theta, circ_mean)| is THE "
                                    "compression statistic all criteria gate "
                                    "on. std_p1 is descriptive only and is NOT "
                                    "angular -- doc 43 SS9.3 conflated them."),
               "scope": ("one rung (VT), one arm (A3), one config, 24 human "
                         "referents, 4 wires, simulation only; no classical "
                         "comparator and no quantum-advantage claim is "
                         "registered; a method claim needs a second task "
                         "family (doc 44 SS7)"),
               "trajectory_gate": ("doc 44 SS6: exp58 reports and STOPS. No "
                                   "further experiment, write-up change, demo "
                                   "build or claim until the user directs.")},
           "git_note": git_note(), "wall_time_sec": time.time() - t0}

    print("\n[exp58] VERDICTS")
    for k, v in verdicts.items():
        print("   %-38s %s" % (k, v))
    for k, v in rho.items():
        if k.startswith("rho_"):
            print("   %-38s %.4f" % (k, v) if v is not None else "   %s None" % k)
    if args.smoke:
        print("[exp58] smoke -- no results file")
        return 0
    p = fresh_results_path("results_exp58")
    assert "exp58" in p, "doc 44 SS8.6 filename guard: %s" % p    # exp57 defect
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[exp58] wrote %s (%.1f min)" % (p, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
