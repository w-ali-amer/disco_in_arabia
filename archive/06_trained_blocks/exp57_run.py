# -*- coding: utf-8 -*-
"""exp57_run.py -- doc 43: WHY does angular spread control learnability (Q1),
and does semantic content help once conditioning is fixed (Q2)?

Q1. Spread controls learnability (Spearman -0.950) and NO mechanism is known;
two were proposed and REFUTED in three days. Two survive and predict opposite
rescues of a spread encoding:
  H-readout  only TWO shared effects must cover the queried state region
             -> MORE EFFECTS rescue spread              (M-wide-R8)
  H-optim    the landscape is ill-conditioned
             -> a TRAINABLE INPUT ROTATION rescues it   (M-wide-ROT)
rho = (acc - M-wide)/(M-tight - M-wide) >= 0.50 decides. BOTH FALSE is a
registered outcome: spread_mechanism_open, reported as OPEN, never explained.

Q2. exp55 concluded semantics carry nothing -- but tested them ONLY at
AraVec's native spread (~1.195 rad) where the rung is hard. C-AraVec
compresses the frozen angles into the 0.15 band, preserving order and
proportional spacing exactly, and must beat BOTH random-at-0.15 and its own
within-human permutation.

doc 43 SS9 resolutions, recorded before this file was written:
  9.1 circular stats pinned (AraVec straddles +-pi; naive mean -> wrong pole)
  9.2 R8 aggregates by MEAN, not max (dense gradients, unbiased at init)
  9.3 ROT = one Euler triple, shared, IDENTITY-INITIALISED
  9.4 rho computed only if M-tight - M-wide >= 0.15, else UNDECIDABLE
  9.5 tied sentinel verified under R8 and ROT
"""
import argparse, copy, json, math, os, platform, random, sys, time
import torch

import exp42_compiler as comp
import exp42_sim_adapter as sim
import exp41_story_generator as gen
from exp42_models import QuantumStoryModel
from exp42_run import (fresh_results_path, git_note, grad_global_norm,
                       paired_permutation_pvalue)

DESIGN_DOC = ("qnlp_private_docs/43_exp57_mechanism_and_compressed_aravec.md "
              "(9c7b586 + SS9-SS10 4ef4041, FROZEN)")
DATA = "stories_exp51_VT.json"
VT_SHA = "8aa71d1e286392ad96fe7a5f77dc8afb7998e5594b8abde67bbaef024e9d1064"
PHI_ART = "exp55_noun_embeddings_phi.json"
PHI_SHA = "2cb5b3cb5feef3a41fbcc7249eccb20c0a0ac4d5ef2f43de5f96b83c37b33242"
CKPT_DIR = "exp57_ckpt"
LR, BATCH = 0.02, 8                 # doc 39 SS3: the config that met VT's gate
N_SEEDS, MAX_EPOCHS = 5, 150
TRAIN_GATE, VAL_GATE = 0.8, 0.7
RNG_BASE = 55000
CONDITIONS = ["M-tight", "M-wide", "M-wide-R8", "M-wide-ROT", "M-tight-R8",
              "C-AraVec", "C-AraVec-Pw"]
SPREAD = {"M-tight": 0.15, "M-wide": 2.40, "M-wide-R8": 2.40,
          "M-wide-ROT": 2.40, "M-tight-R8": 0.15}
N_EFF = {"M-wide-R8": 4, "M-tight-R8": 4}          # per class; else 1
USE_ROT = {"M-wide-ROT": True}
ANGLE_SEEDS = {c: 57000 + 10 * i for i, c in enumerate(CONDITIONS)}
RHO_MIN_GAP = 0.15                                  # doc 43 SS9.4
FALLBACK_ORDER = ["M-tight-R8", "C-AraVec-Pw"]   # doc 43 SS8


class MechModel(QuantumStoryModel):
    """A3 with two doc-43 modifications, both off by default:
      n_eff>1  -> n_eff shared effects PER CLASS, aggregated by MEAN (SS9.2:
                  max gives sparse gradients and is upward-biased at init).
      use_rot  -> ONE Euler triple shared across every referent wire, applied
                  after the intro, IDENTITY-INITIALISED so epoch-0 behaviour
                  equals the no-rotation condition exactly (SS9.3)."""

    def __init__(self, arm, verbs, seed, n_eff=1, use_rot=False):
        super().__init__(arm, verbs, seed)
        self.n_eff = n_eff
        g = torch.Generator().manual_seed(seed + 5701)
        if n_eff > 1:
            self.eff_pos = torch.nn.ModuleList(
                [sim.Circuit4(2, 3, generator=g) for _ in range(n_eff)])
            self.eff_neg = torch.nn.ModuleList(
                [sim.Circuit4(2, 3, generator=g) for _ in range(n_eff)])
        self.use_rot = use_rot
        if use_rot:
            self.rot = torch.nn.Parameter(torch.zeros(3))   # identity at init

    def _effects(self):
        if self.n_eff > 1:
            return list(self.eff_pos), list(self.eff_neg)
        return [self.q_pos], [self.q_neg]

    def story_values(self, cs, verb_map=None):
        vm = verb_map or {}
        cache = {}

        def block(v):
            v = vm.get(v, v)
            if v not in cache:
                cache[v] = self.verb_matrix(v)
            return cache[v]

        psi = sim.init_state(cs.K)
        R = sim.euler_1q(self.rot[0], self.rot[1], self.rot[2]) \
            if self.use_rot else None
        for w, noun, angle in cs.intro:
            psi = sim.apply_gate(psi, sim.ry(angle), (w,))
            if R is not None:
                psi = sim.apply_gate(psi, R, (w,))
        for v, wa, wp in cs.events:
            psi = sim.apply_gate(psi, block(v), (wa, wp))
        vseq, qa, qp = cs.question
        qw = (qa, qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)
        pos, neg = self._effects()
        vp = torch.stack([sim.prob_all_zero_on(
            sim.apply_gate(psi, e.matrix().conj().T, qw), qw) for e in pos])
        vn = torch.stack([sim.prob_all_zero_on(
            sim.apply_gate(psi, e.matrix().conj().T, qw), qw) for e in neg])
        return torch.stack([vp.mean(), vn.mean()])      # SS9.2 MEAN


def wrap_dist_to_pole(t):
    return min(abs(t - math.pi), abs(t + math.pi))


def circ_mean(vals):
    import math as _m
    return _m.atan2(sum(_m.sin(v) for v in vals) / len(vals),
                    sum(_m.cos(v) for v in vals) / len(vals))


def wrapped_dev(t, tbar):
    return math.atan2(math.sin(t - tbar), math.cos(t - tbar))


def build_angles(cond, seed, base, phi_base=None):
    """doc 43 SS9.1: circular statistics are pinned. AraVec humans straddle
    +-pi, so a naive arithmetic mean lands at the OPPOSITE pole."""
    humans = list(gen.HUMANS)
    ang = dict(base)
    rng = random.Random(ANGLE_SEEDS[cond] + seed)
    if cond in SPREAD:
        w = SPREAD[cond]
        ang.update({h: math.pi + rng.uniform(-w, w) for h in humans})
    elif cond in ("C-AraVec", "C-AraVec-Pw"):
        hv = [base[h] for h in humans]
        tbar = circ_mean(hv)
        dev = {h: wrapped_dev(base[h], tbar) for h in humans}
        w_native = max(abs(d) for d in dev.values())
        assert w_native > 0.15, "C-AraVec must COMPRESS: w_native=%.4f" % w_native
        k = 0.15 / w_native
        comp_ang = {h: math.pi + dev[h] * k for h in humans}
        assert max(abs(v - math.pi) for v in comp_ang.values()) <= 0.15 + 1e-9
        order_before = [h for h in sorted(humans, key=lambda x: dev[x])]
        order_after = [h for h in sorted(humans, key=lambda x: comp_ang[x])]
        assert order_before == order_after, "compression scrambled rank order"
        if cond == "C-AraVec-Pw":
            vals = [comp_ang[h] for h in humans]
            rng.shuffle(vals)
            comp_ang = dict(zip(humans, vals))
        ang.update(comp_ang)
    else:
        raise ValueError(cond)
    drawn = {h: ang[h] for h in humans}
    return ang, drawn


def evaluate(m, compiled):
    c = [1 if int(torch.argmax(m.story_values(cs)).item()) == cs.answer else 0
         for cs in compiled]
    return sum(c) / len(c), c


def make_model(cond, verbs, seed):
    return MechModel("A3", verbs, seed, n_eff=N_EFF.get(cond, 1),
                     use_rot=USE_ROT.get(cond, False))


def train(cond, seed, items, base, phi_base, verbs, max_epochs):
    ang, drawn = build_angles(cond, seed, base, phi_base)
    data = {sp: comp.compile_all([i for i in items if i["split"] == sp], ang)
            for sp in ("train", "val", "test")}
    torch.manual_seed(seed)
    m = make_model(cond, verbs, seed)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    rng = random.Random(RNG_BASE + seed)
    best = {"val_acc": -1.0, "train_acc": -1.0, "state": None}
    with torch.no_grad():
        pass
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
            assert math.isfinite(grad_global_norm(m))
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
    rot = [float(x) for x in m.rot.detach()] if getattr(m, "use_rot", False) else None
    post_spread = None
    if rot is not None:
        R = sim.euler_1q(*[torch.tensor(x) for x in rot])
        pv = []
        for h in gen.HUMANS:
            st = sim.apply_gate(sim.apply_gate(sim.init_state(1),
                                sim.ry(ang[h]), (0,)), R, (0,))
            pv.append(float(abs(st[1]) ** 2))          # |<1|psi>|^2 per referent
        mu = sum(pv) / len(pv)
        post_spread = (sum((x - mu) ** 2 for x in pv) / len(pv)) ** .5
    return {"condition": cond, "seed": seed, "drawn_angles": drawn,
            "learned_rot": rot, "post_rotation_spread": post_spread,
            "val_acc": best["val_acc"],
            "train_acc": best["train_acc"], "test_acc": te,
            "test_correct": corr,
            "gate_met": bool(best["train_acc"] >= TRAIN_GATE
                             and best["val_acc"] >= VAL_GATE)}


def seed_mean(rs):
    v = [r["test_correct"] for r in rs]
    return [sum(x[i] for x in v) / len(v) for i in range(len(v[0]))]


def contrast(a, b):
    ma, mb = seed_mean(a), seed_mean(b)
    obs, p = paired_permutation_pvalue([x - y for x, y in zip(ma, mb)])
    return {"mean_x": sum(ma) / len(ma), "mean_y": sum(mb) / len(mb),
            "mean_paired_diff": obs, "one_sided_p": p, "n_items": len(ma)}


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
    assert art["meta"]["theta_reproduces_frozen_exp42_angle"], "theta gate"
    assert art["meta"]["warm_start_hits"] == art["meta"]["n_nouns"], "warm-start gate"
    phi_base = {k: v["phi"] for k, v in art["embeddings"].items()}
    verbs = sorted(meta["verb_inventory"].keys())
    print("[exp57] VT sha ok | phi artifact ok (44/44, theta max diff %.1e) | "
          "conditions=%s" % (art["meta"]["theta_max_circular_diff_rad"], conds),
          flush=True)

    # ---- doc 43 SS9.5: the tied theorem runs through the forward path that R8
    # and ROT modify. Tied blocks give identical statevectors, so ANY readout
    # must return identical outputs -> exactly 0.5000. A deviation is a bug.
    vt_pairs = {}
    for it in items:
        if it["split"] == "test":
            vt_pairs.setdefault(it["pair_id"], []).append(it)
    tp = [sorted(v, key=lambda x: x["id"]) for v in vt_pairs.values()
          if len(v) == 2][:60]
    sentinel = {}
    for cond in conds:
        ang0, _d = build_angles(cond, 0, base, phi_base)
        tm = make_model(cond, verbs, 0)          # same readout/ROT as the arm
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
        print("[exp57] tied sentinel %-12s = %.4f (must be 0.5000)"
              % (cond, sentinel[cond]), flush=True)
    bad = [c for c, v in sentinel.items() if abs(v - 0.5) > 1e-9]
    if bad and not args.smoke:
        print("[exp57] HALT: tied sentinel != 0.5000 for %s -- forward-path bug"
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
            r = train(cond, s, items, base, phi_base, verbs, maxep)
            if not args.smoke:
                json.dump(r, open(ckj, "w", encoding="utf-8"))
            rs.append(r)
        per_cond[cond] = rs
        acc = [r["test_acc"] for r in rs]
        mu = sum(acc) / len(acc)
        sd = (sum((x - mu) ** 2 for x in acc) / max(len(acc) - 1, 1)) ** .5
        print("[exp57] %-5s test %.4f +/- %.4f  gate-rate %d/%d  %s"
              % (cond, mu, sd, sum(1 for r in rs if r["gate_met"]), len(rs),
                 [round(x, 3) for x in acc]), flush=True)

    def seedmean(c):
        a = [r["test_acc"] for r in per_cond[c]]
        return sum(a) / len(a)

    # ---- Q1: rescue fraction, with doc 43 SS9.4's validity precondition
    q1 = {}
    if "M-tight" in per_cond and "M-wide" in per_cond:
        gap = seedmean("M-tight") - seedmean("M-wide")
        q1["reference_gap"] = gap
        q1["references_valid"] = bool(gap >= RHO_MIN_GAP)
        for c in ("M-wide-R8", "M-wide-ROT"):
            if c in per_cond:
                q1["rho_" + c] = ((seedmean(c) - seedmean("M-wide")) / gap
                                  if gap >= RHO_MIN_GAP else None)

    # ---- Q2: semantics at good conditioning
    prim = {}
    if "C-AraVec" in per_cond and "M-tight" in per_cond:
        prim["Q2_CAraVec_vs_Mtight"] = contrast(per_cond["C-AraVec"],
                                                per_cond["M-tight"])
    if "C-AraVec" in per_cond and "C-AraVec-Pw" in per_cond:
        prim["Q2_CAraVec_vs_CAraVecPw"] = contrast(per_cond["C-AraVec"],
                                                   per_cond["C-AraVec-Pw"])
    verdicts = {}
    if q1.get("references_valid"):
        verdicts["readout_capacity_explains"] = bool(
            (q1.get("rho_M-wide-R8") or 0) >= 0.50)
        verdicts["optimisation_explains"] = bool(
            (q1.get("rho_M-wide-ROT") or 0) >= 0.50)
        verdicts["spread_mechanism_open"] = bool(
            not verdicts["readout_capacity_explains"]
            and not verdicts["optimisation_explains"])
    else:
        verdicts["readout_capacity_explains"] = "UNDECIDABLE-on-references"
        verdicts["optimisation_explains"] = "UNDECIDABLE-on-references"
        verdicts["spread_mechanism_open"] = "UNDECIDABLE-on-references"
    if len(prim) == 2:
        verdicts["semantics_helps_when_conditioned"] = bool(
            prim["Q2_CAraVec_vs_Mtight"]["one_sided_p"] < 0.05
            and prim["Q2_CAraVec_vs_CAraVecPw"]["one_sided_p"] < 0.05)
    verdicts.update({"gate_rate_%s" % c:
                     "%d/%d" % (sum(1 for r in per_cond[c] if r["gate_met"]),
                                len(per_cond[c])) for c in per_cond})
    q1_out = q1

    out = {"experiment": "exp57", "spec": DESIGN_DOC,
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "dataset_sha256": VT_SHA,
           "embeddings_sha256": emb_sha, "phi_artifact_sha256": PHI_SHA,
           "phi_artifact_meta": art["meta"],
           "config": {"lr": LR, "batch": BATCH, "epochs": maxep,
                      "n_seeds": nseeds, "arm": "A3"},
           "per_condition": {c: [{k: v for k, v in r.items()
                                  if k != "test_correct"} for r in rs]
                             for c, rs in per_cond.items()},
           "tied_sentinel": sentinel, "Q1_rescue": q1_out, "Q2_contrasts": prim, "verdicts": verdicts,
           "meta_notes": {
               "confound_recorded": ("the pilot permutation permuted over all "
                                     "44 nouns and destroyed the human cluster; "
                                     "EPw here permutes WITHIN humans and is "
                                     "distribution-preserving for this rung"),
               "withdrawn_claim": ("an earlier mechanism claim that referents "
                                   "near |1> suffer weak gradients is WITHDRAWN "
                                   "as backwards -- AraVec is the encoding "
                                   "closest to |1> and it is the one that works"),
               "scope": ("one rung, one arm, one config, 44 nouns, simulation "
                         "only; says nothing about RP, which is out of scope by "
                         "construction; no quantum-advantage comparison is "
                         "registered")},
           "git_note": git_note(), "wall_time_sec": time.time() - t0}
    print("\n[exp57] VERDICTS")
    for k, v in verdicts.items():
        print("   %-34s %s" % (k, v))
    for k, v in prim.items():
        print("   %-34s diff %+.4f  p=%.5f" % (k, v["mean_paired_diff"], v["one_sided_p"]))
    if args.smoke:
        print("[exp57] smoke -- no results file")
        return 0
    p = fresh_results_path("results_exp57")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[exp57] wrote %s (%.1f min)" % (p, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
