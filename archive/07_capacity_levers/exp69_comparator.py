# -*- coding: utf-8 -*-
"""exp69_comparator.py -- doc 55 SS8: the classical comparator, and THE ARBITER.

doc 55 SS8 registers this BEFORE any exp69 number exists, for two reasons that must
not be conflated.

  1. A GUARD. "Registered prediction: the classical comparator MATCHES OR EXCEEDS
     every arm." doc 47 SS14 (a linear model on the identical features hits the
     identical bound by construction) and doc 52 SS1.1 (the product-state kernel
     factorises and is classically O(n), verified to 3.3e-16). Registered so it
     cannot become a post-hoc surprise, and SO NO EXP69 NUMBER CAN BE READ AS A
     QUANTUM RESULT. doc 55 SS11: no exp69 number may appear in a sentence
     containing "advantage".

  2. THE ARBITER (SS8.2), which is the load-bearing one. If R2-W sits at its floor,
     the comparator on the IDENTICAL features decides between two very different
     negatives:

       R2-W at floor, comparator CLEARS  -> CIRCUIT-TRAINABILITY failure at depth.
           The information is there and is linearly extractable; the circuit did
           not learn it. Observed precedent: doc 53 SS7.1.3 on the arity lever.
           Because the T = 1 arms passed with the IDENTICAL encoding, features and
           compilation, the failure is located at COMPOSITION, not at the encoding.
       R2-W at floor, comparator AT FLOOR -> INFORMATION / SAMPLE-side. The
           features do not carry the category through a two-event read at this
           sample size, and no circuit claim is available either way.

     SS8.3 extends the same rule to `register_beats_angle_at_T2`: if the circuit
     shows R2-W - R2-A >= 0.06 but LR(register) - LR(angle) < 0.06, the circuit's
     gap EXCEEDS the information gap and is an R2-A-trainability artefact, and the
     boolean may NOT be claimed.

FORM (doc 55 SS8.1), registered exactly. Logistic regression per arm on the
IDENTICAL per-referent features, joint by OUTER PRODUCT, ordered AGENT-FIRST OF THE
QUERIED EVENT:

    phi = flatten( u(theta_agent(q)) (x) u(theta_patient(q)) )
          u(t) = (1, cos t, sin t)                 -> 9-dim   for the angle arms
          u(theta_0) (x) u(theta_1) per referent   -> 81-dim  for the register arms

"OF THE QUERIED EVENT" is what makes this the right frame at T = 2: doc 55 SS12
states that the per-twin constraint system at T = 2 is IDENTICAL to exp68's,
because the label is a function of the QUERIED event's ordered (agent, patient)
pair and the category membership of that agent -- the same antisymmetric form doc
47 SS3.2/SS17 states, with the queried event supplying the ordering.

THE FRAME CHECK IS ASSERTED, NOT ASSUMED (doc 53 SS5's both-directions discipline),
AND AT T = 2 IT RUNS PER q: wire-ordered features are bit-identical across twins (a
NULL INSTRUMENT, provably 0.5) while agent-ordered features separate EVERY twin
pair. A frame that cannot separate the twins cannot be measuring the task.

THE COMPARATOR GETS FLOORS TOO, at the same relabel seeds the circuit floors use,
because SS8.2's rows are of the form "comparator clears floor".

SS8.4's QUALIFIER, CARRIED AND STRENGTHENED AT T = 2. "Identical features" makes
the comparator a RESTRICTION of the circuit's conditional function class: the
circuit carries one form PER CONDITIONING GROUP, the comparator carries one form
TOTAL. doc 55 SS5 counts 12 groups at T = 1 and 72 at T = 2, so the comparator is a
STRICTLY STRONGER restriction here than it was in exp68. A comparator loss on a
T = 2 cell is therefore even less automatically a circuit win. The conditioning
descriptors (q, the unqueried event's direction) are REPORTED alongside so the gap
is interpretable.

Writes `exp69_ckpt/comparator.json`, and NO results file: doc 53 SS7.3's standing
rule gives the run ONE aggregator, and `exp69_aggregate.py` is it. It ships in the
SAME commit as the runner (doc 55 SS8.5, closing doc 53 SS7.1's registration debt).

    python exp69_comparator.py
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)

import exp69_run as R

DESIGN_DOC = ("qnlp_private_docs/55_exp69_reunion.md (da4b8fe, REGISTERED "
              "through SS19.7) SS8")
OUT = os.path.join(R.CKPT_DIR, "comparator.json")
FIT_SEED = 69001                        # registered; per-seed init = FIT_SEED + seed
ADAM_STEPS, ADAM_LR = 3000, 0.05
LBFGS_ROUNDS, LBFGS_ITER = 10, 200
TIE_BAND = 0.02                         # doc 55 SS6.7


# ----------------------------------------------------------------- features
def joint_features(cs, uref):
    """doc 55 SS8.1: outer(u(agent of event q), u(patient of event q)), AGENT-FIRST.

    Twins share the cast, the wire order, the question spec and the UNQUERIED
    event, and differ ONLY in the queried event's agent/patient direction, so the
    ordered pair the decision is a function of is (agent(q), patient(q))."""
    noun = {w: n for w, n, _a in cs.intro}
    _v, wa, wp = cs.events[cs.qidx - 1]
    return np.outer(uref[noun[wa]], uref[noun[wp]]).ravel()


def wire_features(cs, uref):
    """The null-instrument frame, kept only to be machine-refuted."""
    noun = {w: n for w, n, _a in cs.intro}
    return np.outer(uref[noun[0]], uref[noun[1]]).ravel()


def build_xy(cs_list, uref):
    X = torch.tensor(np.asarray([joint_features(c, uref) for c in cs_list]),
                     dtype=torch.float64)
    y = torch.tensor([c.answer for c in cs_list], dtype=torch.long)
    return X, y


def assert_frame(cs_list, uref, dim, key, rep):
    """doc 55 SS8.1, and AT T = 2 THE CHECK RUNS PER q."""
    byq = {}
    for cs in cs_list:
        byq.setdefault(cs.qidx, []).append(cs)
    per_q = {}
    for q, sub in sorted(byq.items()):
        pairs = {}
        for cs in sub:
            pairs.setdefault(cs.pair_id, []).append(cs)
        twins = [v for v in pairs.values() if len(v) == 2]
        n_wire_same = sum(1 for a, b in twins
                          if np.allclose(wire_features(a, uref),
                                         wire_features(b, uref)))
        n_agent_same = sum(1 for a, b in twins
                           if np.allclose(joint_features(a, uref),
                                          joint_features(b, uref)))
        assert n_wire_same == len(twins), \
            ("q=%s: the wire frame separates twins -- the null instrument is not "
             "null" % q)
        assert n_agent_same == 0, \
            "q=%s: the agent frame FAILS to separate %d twin pairs" % (q, n_agent_same)
        per_q[str(q)] = {"twin_pairs_checked": len(twins),
                         "wire_frame_identical_twin_pairs": n_wire_same,
                         "agent_frame_identical_twin_pairs": n_agent_same}
    d = len(joint_features(cs_list[0], uref))
    assert d == dim, "feature dim %d != registered %d" % (d, dim)
    rep[key] = {"dim": d, "per_q": per_q,
                "wire_frame_is_a_null_instrument": True,
                "frame": "agent-first outer product of the QUERIED event"}


# -------------------------------------------------------------------- model
class LogReg(torch.nn.Module):
    """Binary logistic regression, float64. Two logits so the head matches the
    circuit's two-effect readout shape; softmax over two logits is the same model
    as one logit and keeps the loss identical to the circuit's."""

    def __init__(self, d, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = torch.nn.Parameter(
            (torch.rand(2, d, generator=g, dtype=torch.float64) - 0.5) * (2.0 / 3.0))
        self.b = torch.nn.Parameter(
            (torch.rand(2, generator=g, dtype=torch.float64) - 0.5) * (2.0 / 3.0))

    def logits(self, X):
        return X @ self.W.T + self.b

    def loss(self, X, y):
        return torch.nn.functional.cross_entropy(self.logits(X), y)

    def hard_acc(self, X, y):
        with torch.no_grad():
            return float((torch.argmax(self.logits(X), 1) == y).double().mean())


def fit(arm, seed, store, fm, shuffled):
    """One deterministic fit -- `exp67_comparator` / `exp68_comparator`'s frame,
    verbatim: float64, full-batch Adam then LBFGS/strong-Wolfe, checkpoint selected
    on HARD val accuracy, hard argmax accuracy reported on held-out."""
    base = R.base_arm(arm)
    uref = R.referent_features(base, fm)
    items, meta, path = R.arm_items(base, store)
    use = R.relabel_shuffle_q(items, seed) if shuffled else items
    ang, _t1 = R.arm_angles(base, fm)
    data = {s: R.compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}
    Xtr, ytr = build_xy(data["train"], uref)
    Xva, yva = build_xy(data["val"], uref)
    Xte, yte = build_xy(data["test"], uref)

    init = FIT_SEED + seed
    torch.manual_seed(init)
    m = LogReg(Xtr.shape[1], init)
    t0 = time.time()
    with torch.no_grad():
        loss0 = float(m.loss(Xtr, ytr))
    best = {"v": -1.0, "state": None, "where": None}

    def record(where):
        v = m.hard_acc(Xva, yva)
        if v > best["v"]:
            best.update({"v": v, "where": where,
                         "state": {k: t.detach().clone()
                                   for k, t in m.state_dict().items()}})

    opt = torch.optim.Adam(m.parameters(), lr=ADAM_LR)
    for step in range(1, ADAM_STEPS + 1):
        opt.zero_grad()
        L = m.loss(Xtr, ytr)
        L.backward()
        opt.step()
        if step % 10 == 0:
            record("adam%d" % step)
    lb = torch.optim.LBFGS(m.parameters(), max_iter=LBFGS_ITER, history_size=50,
                           tolerance_grad=1e-12, tolerance_change=1e-14,
                           line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        L = m.loss(Xtr, ytr)
        L.backward()
        return L

    for _r in range(LBFGS_ROUNDS):
        lb.step(closure)
        record("lbfgs%d" % _r)
    with torch.no_grad():
        loss_final = float(m.loss(Xtr, ytr))
    m.zero_grad()
    m.loss(Xtr, ytr).backward()
    gnorm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters())))

    m.load_state_dict(best["state"])
    with torch.no_grad():
        pred = torch.argmax(m.logits(Xte), 1).tolist()
    rows = []
    for cs, p in zip(data["test"], pred):
        r = dict(cs.meta)
        r.update({"pred": int(p), "target": int(cs.answer),
                  "correct": bool(int(p) == cs.answer)})
        rows.append(r)

    # SS8.4's conditioning descriptors, REPORTED alongside the gap
    desc = {}
    for keyname, keyfn in (
            ("by_q", lambda r: "q%d" % r["query_index"]),
            ("by_order_pair", lambda r: "/".join(r["order_flags"]))):
        cell = {}
        for r in rows:
            cell.setdefault(keyfn(r), [0, 0])
            cell[keyfn(r)][0] += int(r["correct"])
            cell[keyfn(r)][1] += 1
        desc[keyname] = {k: {"acc": v[0] / v[1], "n": v[1]}
                         for k, v in sorted(cell.items())}

    return {"arm": arm, "base_arm": base, "is_floor": bool(shuffled), "seed": seed,
            "T": R.ARM_CFG[base]["T"], "part": R.ARM_CFG[base]["part"],
            "feature_dim": int(Xtr.shape[1]), "fit_seed": init,
            "dataset": path, "dataset_sha256": meta["dataset_sha256"],
            "train_acc": m.hard_acc(Xtr, ytr), "val_acc": best["v"],
            "heldout_test_acc": m.hard_acc(Xte, yte),
            "train_loss_init": loss0, "train_loss_final": loss_final,
            "grad_norm_final": gnorm, "selected_at": best["where"],
            "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
            "conditioning_descriptors": desc,
            "heldout_items": rows,
            "wall_sec": time.time() - t0}


def agg(rows):
    a = [r["heldout_test_acc"] for r in rows]
    mean = sum(a) / len(a)
    sd = (sum((x - mean) ** 2 for x in a) / (len(a) - 1)) ** 0.5 if len(a) > 1 else 0.0
    return {"mean": mean, "sd": sd, "per_seed": a,
            "seeds": [r["seed"] for r in rows],
            "feature_dim": rows[0]["feature_dim"], "T": rows[0]["T"],
            "val_per_seed": [r["val_acc"] for r in rows],
            "train_per_seed": [r["train_acc"] for r in rows],
            "train_loss_init_per_seed": [r["train_loss_init"] for r in rows],
            "train_loss_final_per_seed": [r["train_loss_final"] for r in rows],
            "grad_norm_final_per_seed": [r["grad_norm_final"] for r in rows],
            "conditioning_descriptors_per_seed":
                [r["conditioning_descriptors"] for r in rows],
            "runs": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=R.N_SEEDS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    t0 = time.time()
    store = {k: R.load_part(k) for k in ("A", "B", "exp68")}
    fm = R.feature_maps()

    frame = {}
    for base in R.BASE_ARMS:
        uref = R.referent_features(base, fm)
        ang, _t1 = R.arm_angles(base, fm)
        items, _m, _p = R.arm_items(base, store)
        probe = R.compile_all([i for i in items if i["split"] == "test"], ang)
        dim = 81 if R.ARM_CFG[base]["enc"] == "F2" else 9
        assert_frame(probe, uref, dim, base, frame)
        print("[frame] %-5s d=%-2d  %s" % (base, dim, frame[base]["per_q"]), flush=True)

    arms, floors = {}, {}
    for base in R.BASE_ARMS:
        arms[base] = agg([fit(base, s, store, fm, False) for s in range(a.seeds)])
        line = ("[exp69-comp] %-5s d=%-2d T=%d held-out %.4f (sd %.4f)"
                % (base, arms[base]["feature_dim"], arms[base]["T"],
                   arms[base]["mean"], arms[base]["sd"]))
        if base in R.FLOOR_OF:
            floors[base] = agg([fit(R.FLOOR_OF[base], s, store, fm, True)
                                for s in range(a.seeds)])
            line += "  floor %.4f (sd %.4f)" % (floors[base]["mean"],
                                                floors[base]["sd"])
        else:
            line += "  [no floor -- A0 uses exp68's committed floor]"
        line += ("  train-loss %.4f -> %.4f  |g|=%.1e"
                 % (float(np.mean(arms[base]["train_loss_init_per_seed"])),
                    float(np.mean(arms[base]["train_loss_final_per_seed"])),
                    max(arms[base]["grad_norm_final_per_seed"])))
        print(line, flush=True)

    out = {
        "component": "exp69_comparator",
        "spec": DESIGN_DOC,
        "role": ("doc 55 SS8: registered GUARD (no exp69 number may be read as a "
                 "quantum result -- doc 52 SS1 closed advantage for this encoding "
                 "family) and registered ARBITER (SS8.2 / SS8.3) between a "
                 "circuit-trainability failure at depth and an information-side "
                 "failure."),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "features": {
            "definition": ("per referent: u = (1, cos t, sin t) for the angle arms "
                           "(d = 3) and u(theta0) (x) u(theta1) for the register "
                           "arms (d = 9); joint by OUTER PRODUCT ordered AGENT-FIRST "
                           "OF THE QUERIED EVENT -> 9 dims for the angle arms, 81 "
                           "for the register arms"),
            "why_the_queried_event": (
                "doc 55 SS12: the per-twin constraint system at T = 2 is IDENTICAL to "
                "exp68's, because the label is a function of the QUERIED event's "
                "ordered (agent, patient) pair and the category membership of that "
                "agent -- doc 47 SS3.2/SS17's antisymmetric form with the queried "
                "event supplying the ordering."),
            "identical_to_circuit": (
                "these are the SAME per-referent vectors the arms' circuits are "
                "linear in. exp69_run.py --smoke measures the closure directly: the "
                "angle arms close at 3 terms and the register arms close in the "
                "9-dim product basis, AT BOTH T (doc 55 SS7.2)."),
            "frame_check": frame},
        "optimiser": {"fit_seed_base": FIT_SEED, "per_seed_init": "FIT_SEED + seed",
                      "adam_steps": ADAM_STEPS, "adam_lr": ADAM_LR,
                      "lbfgs_rounds": LBFGS_ROUNDS, "lbfgs_max_iter": LBFGS_ITER,
                      "line_search": "strong_wolfe", "batching": "FULL batch",
                      "dtype": "float64",
                      "model_selection": "best HARD argmax accuracy on the val split",
                      "reported_accuracy": "HARD argmax on held-out test",
                      "seeds_vary": ("optimiser INITIALISATION ONLY -- doc 55 SS4.4's "
                                     "rule for the circuit arms, applied identically "
                                     "here. The features are frozen constants, so the "
                                     "five fits sit on ONE geometry and the spread is "
                                     "init spread, not data spread.")},
        "arms": arms, "floors": floors,
        "sd_convention": "sample sd, ddof=1", "n_seeds": a.seeds,
        "tie_band": TIE_BAND,
        "function_class_qualifier": (
            "doc 55 SS8.4: 'identical features' makes the comparator a RESTRICTION of "
            "the circuit's conditional function class -- the circuit carries one "
            "9-dim (or 81-dim) form PER CONDITIONING GROUP, the comparator carries "
            "one TOTAL. doc 55 SS5 counts 12 groups at T = 1 and 72 at T = 2, so at "
            "T = 2 the comparator is a STRICTLY STRONGER restriction than it was in "
            "exp68 and a comparator loss on a T = 2 cell is even less automatically a "
            "circuit win. This qualifier travels with every number in this file."),
        "wall_time_sec": time.time() - t0,
    }
    os.makedirs(R.CKPT_DIR, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[exp69-comp] wrote %s (%.0fs)" % (a.out, out["wall_time_sec"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
