# -*- coding: utf-8 -*-
"""exp55_run.py -- doc 39: does the referent ENCODING carry the signal, and is
it SEMANTICS or CONDITIONING?

Motivating observation (n=1, unvalidated): training VT with random referent
angles reaches CHANCE (0.4967) where AraVec reaches 0.7200 and clears the gate.

The two live hypotheses, which NO diagnostic so far separates (doc 39 SS2):
  H-sem   the noun->angle assignment carries information the model uses
  H-cond  what matters is that referents occupy a tight band near a common
          reference state, giving the two shared trained effects a fixed
          signature to key against

The pilot permutation was CONFOUNDED: it permuted over all 44 nouns, and
AraVec puts humans near the |1> pole (mean distance 0.598 rad) and objects
near |0> (2.467 rad). VT uses all-human casts, so that permutation injected
object angles into human slots and destroyed the cluster. EPw here permutes
WITHIN humans, which is genuinely distribution-preserving for this rung.

Conditions, all at ONE qubit per referent (2-qubit registers are exp45's
variable and would confound architecture with encoding):
  E1    AraVec theta, true assignment           cluster + semantics
  EPw   AraVec theta, permuted WITHIN humans    semantics vs conditioning
  EC    random from a cluster-matched band      conditioning, no AraVec values
  EG    24 evenly spaced over the circle        separability, no cluster
  E0    uniform random                          validates the premise at n=5
  E2    AraVec (theta, phi)                     RICHNESS -- the hypothesis
  E2Pw  (theta, phi) permuted within humans     richness without semantics

The richer intro is Rz(phi).Ry(theta)|0>: a pure qubit has exactly 2 real DOF
and the baseline uses one (real amplitudes). Relative phase is physical once
the generic SU(4) blocks entangle.

Primary contrasts (doc 39 SS5), one-sided paired permutation, 10k perms:
  richness_helps        E2  vs E1
  encoding_is_semantic  E1  vs EPw
  random_encoding_fails E1  vs E0
Everything else is descriptive, including per-condition GATE-RATE (n/5 seeds
meeting train>=0.8 AND val>=0.7) -- the "beyond the bar" quantity.
"""

import argparse, copy, json, math, os, platform, random, sys, time
import torch

import exp42_compiler as comp
import exp42_sim_adapter as sim
import exp41_story_generator as gen
from exp42_models import QuantumStoryModel
from exp42_run import (fresh_results_path, git_note, grad_global_norm,
                       paired_permutation_pvalue)

DESIGN_DOC = "qnlp_private_docs/39_exp55_referent_encoding.md (c410977, FROZEN)"
DATA = "stories_exp51_VT.json"
VT_SHA = "8aa71d1e286392ad96fe7a5f77dc8afb7998e5594b8abde67bbaef024e9d1064"
PHI_ART = "exp55_noun_embeddings_phi.json"
PHI_SHA = "2cb5b3cb5feef3a41fbcc7249eccb20c0a0ac4d5ef2f43de5f96b83c37b33242"
CKPT_DIR = "exp55_ckpt"
LR, BATCH = 0.02, 8                 # doc 39 SS3: the config that met VT's gate
N_SEEDS, MAX_EPOCHS = 5, 150
TRAIN_GATE, VAL_GATE = 0.8, 0.7
RNG_BASE = 55000
CONDITIONS = ["E1", "EPw", "EC", "EG", "E0", "E2", "E2Pw"]
FALLBACK_ORDER = ["EG", "EC"]       # doc 39 SS8: drop these first, never seeds


class PhaseIntroModel(QuantumStoryModel):
    """Baseline is Ry(theta)|0>. With a phi map the intro becomes
    Rz(phi).Ry(theta)|0>, using the qubit's second real degree of freedom."""

    def __init__(self, arm, verbs, seed, phi_map=None):
        super().__init__(arm, verbs, seed)
        self.phi_map = phi_map or {}
        # phases are constants, not parameters -- build each gate once
        self._rz = {n: sim.rz(torch.tensor(float(v)))
                    for n, v in (phi_map or {}).items()}

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
            psi = sim.apply_gate(psi, sim.ry(angle), (w,))
            if noun in self._rz:
                psi = sim.apply_gate(psi, self._rz[noun], (w,))
        for v, wa, wp in cs.events:
            psi = sim.apply_gate(psi, block(v), (wa, wp))
        vseq, qa, qp = cs.question
        qw = (qa, qp)
        for qv in reversed(vseq):
            psi = sim.apply_gate(psi, block(qv).conj().T, qw)
        return torch.stack([sim.prob_all_zero_on(
            sim.apply_gate(psi, e.matrix().conj().T, qw), qw)
            for e in (self.q_pos, self.q_neg)])


def wrap_dist_to_pole(t):
    return min(abs(t - math.pi), abs(t + math.pi))


def build_angles(cond, seed, base, phi_base):
    """Return (angle_map, phi_map). Objects keep frozen values throughout --
    VT casts are all-human so object angles are never used, but leaving them
    fixed removes them as a variable."""
    humans = list(gen.HUMANS)
    ang = dict(base)
    rng = random.Random(RNG_BASE + 977 * seed + hash(cond) % 1000)
    hvals = [base[h] for h in humans]
    if cond in ("E1", "E2"):
        pass
    elif cond in ("EPw", "E2Pw"):
        v = hvals[:]
        rng.shuffle(v)
        ang.update(dict(zip(humans, v)))
    elif cond == "EC":
        w = max(wrap_dist_to_pole(t) for t in hvals)     # matched band width
        ang.update({h: math.pi + rng.uniform(-w, w) for h in humans})
    elif cond == "EG":
        step = 2 * math.pi / len(humans)
        ev = [i * step for i in range(len(humans))]
        rng.shuffle(ev)
        ang.update(dict(zip(humans, ev)))
    elif cond == "E0":
        ang.update({h: rng.uniform(0, 2 * math.pi) for h in humans})
    else:
        raise ValueError(cond)
    phi = None
    if cond in ("E2", "E2Pw"):
        if cond == "E2":
            phi = {h: phi_base[h] for h in humans}
        else:
            v = [phi_base[h] for h in humans]
            rng.shuffle(v)
            phi = dict(zip(humans, v))
    return ang, phi


def evaluate(m, compiled):
    c = [1 if int(torch.argmax(m.story_values(cs)).item()) == cs.answer else 0
         for cs in compiled]
    return sum(c) / len(c), c


def train(cond, seed, items, base, phi_base, verbs, max_epochs):
    ang, phi = build_angles(cond, seed, base, phi_base)
    data = {sp: comp.compile_all([i for i in items if i["split"] == sp], ang)
            for sp in ("train", "val", "test")}
    torch.manual_seed(seed)
    m = PhaseIntroModel("A3", verbs, seed, phi_map=phi)
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
    return {"condition": cond, "seed": seed, "val_acc": best["val_acc"],
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
    print("[exp55] VT sha ok | phi artifact ok (44/44, theta max diff %.1e) | "
          "conditions=%s" % (art["meta"]["theta_max_circular_diff_rad"], conds),
          flush=True)

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
        print("[exp55] %-5s test %.4f +/- %.4f  gate-rate %d/%d  %s"
              % (cond, mu, sd, sum(1 for r in rs if r["gate_met"]), len(rs),
                 [round(x, 3) for x in acc]), flush=True)

    prim = {}
    if "E2" in per_cond and "E1" in per_cond:
        prim["richness_helps__E2_vs_E1"] = contrast(per_cond["E2"], per_cond["E1"])
    if "EPw" in per_cond and "E1" in per_cond:
        prim["encoding_is_semantic__E1_vs_EPw"] = contrast(per_cond["E1"], per_cond["EPw"])
    if "E0" in per_cond and "E1" in per_cond:
        prim["random_encoding_fails__E1_vs_E0"] = contrast(per_cond["E1"], per_cond["E0"])
    verdicts = {k: bool(v["one_sided_p"] < 0.05) for k, v in prim.items()}
    verdicts.update({"gate_rate_%s" % c:
                     "%d/%d" % (sum(1 for r in per_cond[c] if r["gate_met"]),
                                len(per_cond[c])) for c in per_cond})

    out = {"experiment": "exp55", "spec": DESIGN_DOC,
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "dataset_sha256": VT_SHA,
           "embeddings_sha256": emb_sha, "phi_artifact_sha256": PHI_SHA,
           "phi_artifact_meta": art["meta"],
           "config": {"lr": LR, "batch": BATCH, "epochs": maxep,
                      "n_seeds": nseeds, "arm": "A3"},
           "per_condition": {c: [{k: v for k, v in r.items()
                                  if k != "test_correct"} for r in rs]
                             for c, rs in per_cond.items()},
           "primary_contrasts": prim, "verdicts": verdicts,
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
    print("\n[exp55] VERDICTS")
    for k, v in verdicts.items():
        print("   %-34s %s" % (k, v))
    for k, v in prim.items():
        print("   %-34s diff %+.4f  p=%.5f" % (k, v["mean_paired_diff"], v["one_sided_p"]))
    if args.smoke:
        print("[exp55] smoke -- no results file")
        return 0
    p = fresh_results_path("results_exp55")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[exp55] wrote %s (%.1f min)" % (p, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
