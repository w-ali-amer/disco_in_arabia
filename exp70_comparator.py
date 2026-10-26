# -*- coding: utf-8 -*-
"""exp70_comparator.py -- doc 56 SS8: the classical comparator HIERARCHY, and it is
THE ARBITER.

doc 56 SS8 registers this BEFORE any exp70 number exists, and it ships in the SAME
COMMIT as the runner (SS8.6 / SS16.1, closing doc 53 SS7.1's registration debt rather
than repeating it). It does two things that must not be conflated.

  1. A GUARD. "Registered prediction: the comparators MATCH OR EXCEED every arm"
     (doc 47 SS14; doc 52 SS1.1). Registered in advance SO NO EXP70 NUMBER CAN BE
     READ AS A QUANTUM RESULT and so a tie is not a post-hoc surprise. exp67, exp68
     and exp69 all measured comparator >= circuit on every arm. doc 56 SS0.1: no
     exp70 number may appear in a sentence containing "advantage", and this holds
     for the entangled arms MOST of all -- stepping into the family doc 52 SS1.3
     named is not the same as reaching the regime it described.

  2. THE ARBITER, and doc 56 SS8.1 makes it sharper than any previous rung's.

TWO COMPARATORS FOR N-in AND P-eps, AND THE PAIR IS THE INSTRUMENT (SS8.1)
=========================================================================
    (i)  INFORMATION-MATCHED   logistic regression on the RAW (p1, p2) pairs
         phi = (p1(agent), p2(agent), p1(patient), p2(patient))          4-dim
         -- the same four numbers the encoding consumes, with NO MAP APPLIED.

    (ii) FUNCTION-CLASS-MATCHED  logistic regression on the EXPLICIT STATE-
         COEFFICIENT features, agent-first outer product
         P-eps:  phi = vec( u_agent (x) u_patient ), u = u(t1) (x) u(t2)  81-dim
         N-in :  phi = vec( v_agent (x) v_patient ), v = (a^2,b^2,c^2,ab,ac,bc) 36-dim

THE READING RULE, REGISTERED VERBATIM (doc 56 SS8.1):

  * If (i) TIES the circuit, THE ENTANGLED MAP ADDED NOTHING EXTRACTABLE -- the two
    raw coordinates already carry everything the circuit got out of them, and the
    map is a reparameterisation however exotic its span.
  * If (i) FAILS to tie but (ii) TIES, THE MAP GENUINELY RESHAPED THE FUNCTION
    CLASS -- the information needed a nonlinear reorganisation the raw coordinates
    do not offer linearly, and the encoding supplied it.

  This is the sharpest form the arbiter has taken in the programme, because it
  separates "the circuit learned something" from "the ENCODING made something
  learnable", which every previous rung's single comparator conflated.

THE N-across COMPARATOR IS THE JOINT-STATE FORM (SS8.2)
=======================================================
For a genuinely entangled cross-referent state the per-referent factorisation DOES
NOT EXIST, so the comparator feature is the JOINT one: the 10 independent entries of
the outer square of the joint 4-dim real amplitude vector, equivalently the 10 real
Pauli coordinates {<P(x)Q> : P,Q in {I,X,Z}} u {<Y(x)Y>}, agent-first.

**WHY THE OUTER SQUARE AND NOT THE AMPLITUDES.** The 4-dim amplitude vector is the
smaller and more tempting feature, and it is WRONG: readouts are expectation values,
QUADRATIC in amplitudes, so a linear model on amplitudes is a strictly weaker class
than the circuit's and would MANUFACTURE A SPURIOUS CIRCUIT WIN. The outer square is
the function-class-matched object. Stated because choosing the amplitudes would have
been an easy and invisible error.

**AND IT CANNOT SEPARATE N-across FROM EA-ctrl, BY PROOF, REGISTERED IN ADVANCE.**
doc 56 SS4.5: <X(x)X> is identically zero under the CZ map so the EFFECTIVE dimension
is 9, and EA-ctrl's comparator on the same 10 coordinates (with <Y(x)Y> identically
zero instead) is the SAME FUNCTION CLASS. Any measured N-across vs EA-ctrl difference
is therefore TRAINABILITY or INDUCTIVE BIAS, NEVER CAPACITY -- registered here so it
cannot be reached for after the fact.

THE FRAME CHECK IS ASSERTED, NOT ASSUMED (SS8.3), AND AT T = 2 IT RUNS PER q:
wire-ordered features are bit-identical across twins (a NULL INSTRUMENT, provably
0.5) while agent-ordered features separate EVERY twin pair. A frame that cannot
separate the twins cannot be measuring the task.

THE COMPARATORS GET FLOORS TOO, at the same relabel seeds the circuit floors use,
because SS8.4's rows are of the form "comparator clears floor".

SS8.5's QUALIFIER, CARRIED. "Identical features" makes each comparator a RESTRICTION
of the circuit's conditional function class: the circuit carries one form PER
CONDITIONING GROUP (12 at T = 1, 72 at T = 2, doc 55 SS5), the comparator carries one
form TOTAL. A comparator loss on a T = 2 cell is therefore even less automatically a
circuit win, and is read with this qualifier attached.

Writes `exp70_ckpt/comparator.json`, and NO results file: doc 53 SS7.3's standing
rule gives the run ONE aggregator, and `exp70_aggregate.py` is it.

    python exp70_comparator.py
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

import exp69_run as R69
import exp70_run as R
import instrument_sb_bounds as INS

DESIGN_DOC = ("qnlp_private_docs/56_exp70_entangled_encodings.md (566b9ce, "
              "REGISTERED 2026-08-25) SS8")
OUT = os.path.join(R.CKPT_DIR, "comparator.json")
FIT_SEED = 70001                        # registered; per-seed init = FIT_SEED + seed
ADAM_STEPS, ADAM_LR = 3000, 0.05
LBFGS_ROUNDS, LBFGS_ITER = 10, 200
TIE_BAND = 0.02                         # doc 56 SS6.5

# doc 56 SS8.1 / SS8.2: which comparators each family carries, in registration order.
# "raw"   = (i)  INFORMATION-MATCHED, 4-dim, no map applied
# "class" = (ii) FUNCTION-CLASS-MATCHED, the state-coefficient outer product
# "joint" =      the E-across joint 10-coordinate form (SS8.2)
TIERS = {
    "A0-W":     ["class"],
    "EA-ctrl":  ["joint"],
    "P-eps":    ["raw", "class"],
    "N-in":     ["raw", "class"],
    "N-across": ["joint"],
}
TIER_ROLE = {
    "raw": ("(i) INFORMATION-MATCHED -- logistic regression on the RAW (p1, p2) "
            "pairs of the queried event's agent and patient, NO MAP APPLIED. The "
            "same four numbers the encoding consumes."),
    "class": ("(ii) FUNCTION-CLASS-MATCHED -- logistic regression on the EXPLICIT "
              "STATE-COEFFICIENT features, agent-first outer product. These are the "
              "SAME per-referent vectors the arm's circuit is linear in, which "
              "exp70_run.py --smoke certifies directly by Fourier closure."),
    "joint": ("the JOINT 10-coordinate form (doc 56 SS8.2) -- the outer square of "
              "the joint 4-dim real amplitude vector, NOT the amplitudes. There is "
              "no per-referent factorisation to use."),
}


# ----------------------------------------------------------------- features
def feature_fn(arm, tier, fm):
    """(callable(cs) -> phi, dim, description). AGENT-FIRST OF THE QUERIED EVENT."""
    base = R.base_arm(arm)
    fam = R.ARM_CFG[base]["family"]
    if tier == "raw":
        co = fm["coords"]

        def phi_raw(a, p):
            return np.array([co[a][0], co[a][1], co[p][0], co[p][1]], dtype=float)
        return phi_raw, 4, ("(p1(agent), p2(agent), p1(patient), p2(patient)) -- "
                            "the raw normalised PCA coordinates, NO map")
    if tier == "joint":
        fn, cz = R.across_features(base, fm)
        return fn, 10, ("the 10 independent entries of the outer square of the "
                        "joint amplitude vector (cA cB, cA sB, sA cB, %ssA sB), "
                        "agent-first; equivalently the 10 real Pauli coordinates. "
                        "<X(x)X> is identically %s here, so the EFFECTIVE dimension "
                        "is 9 and this class is IDENTICAL to %s's (doc 56 SS4.5)."
                        % ("-" if cz else "", "0" if cz else "sin tA sin tB",
                           "EA-ctrl" if cz else "N-across"))
    uref = R.referent_features(base, fm)
    d = len(next(iter(uref.values())))

    def phi_class(a, p):
        return np.outer(uref[a], uref[p]).ravel()
    return phi_class, d * d, ("vec(v_agent (x) v_patient), agent-first, with "
                              "v the arm's own per-referent state-coefficient "
                              "vector (d_ref = %d, d_joint = %d) for family %s"
                              % (d, d * d, fam))


def queried_pair(cs):
    noun = {w: n for w, n, _a in cs.intro}
    _v, wa, wp = cs.events[cs.qidx - 1]
    return noun[wa], noun[wp]


def wire_pair(cs):
    noun = {w: n for w, n, _a in cs.intro}
    return noun[0], noun[1]


def build_xy(cs_list, fn):
    X = torch.tensor(np.asarray([fn(*queried_pair(c)) for c in cs_list]),
                     dtype=torch.float64)
    y = torch.tensor([c.answer for c in cs_list], dtype=torch.long)
    return X, y


def assert_frame(cs_list, fn, dim, key, rep):
    """doc 56 SS8.3, and AT T = 2 THE CHECK RUNS PER q."""
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
                          if np.allclose(fn(*wire_pair(a)), fn(*wire_pair(b))))
        n_agent_same = sum(1 for a, b in twins
                           if np.allclose(fn(*queried_pair(a)),
                                          fn(*queried_pair(b))))
        assert n_wire_same == len(twins), \
            "q=%s: the wire frame SEPARATES twins -- the null instrument is not null" % q
        assert n_agent_same == 0, \
            "q=%s: the agent frame FAILS to separate %d twin pairs" % (q, n_agent_same)
        per_q[str(q)] = {"twin_pairs_checked": len(twins),
                         "wire_frame_identical_twin_pairs": n_wire_same,
                         "agent_frame_identical_twin_pairs": n_agent_same}
    d = len(fn(*queried_pair(cs_list[0])))
    assert d == dim, "feature dim %d != registered %d" % (d, dim)
    rep[key] = {"dim": d, "per_q": per_q,
                "wire_frame_is_a_null_instrument": True,
                "frame": "agent-first ordering of the QUERIED event"}


# -------------------------------------------------------------------- model
class LogReg(torch.nn.Module):
    """Binary logistic regression, float64. Two logits so the head matches the
    circuit's two-effect readout shape."""

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


def fit(arm, tier, seed, store, fm, shuffled):
    """One deterministic fit -- exp67 / exp68 / exp69's frame, verbatim: float64,
    full-batch Adam then LBFGS/strong-Wolfe, checkpoint selected on HARD val
    accuracy, hard argmax accuracy reported on held-out."""
    base = R.base_arm(arm)
    fn, dim, desc = feature_fn(base, tier, fm)
    items, meta, path = R.arm_items(base, store)
    use = R69.relabel_shuffle_q(items, seed) if shuffled else items
    ang, _t1 = R.arm_angles(base, fm)
    data = {s: R69.compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}
    Xtr, ytr = build_xy(data["train"], fn)
    Xva, yva = build_xy(data["val"], fn)
    Xte, yte = build_xy(data["test"], fn)

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

    desc_cells = {}
    for keyname, keyfn in (
            ("by_q", lambda r: "q%d" % r["query_index"]),
            ("by_order_pair", lambda r: "/".join(r["order_flags"]))):
        cell = {}
        for r in rows:
            cell.setdefault(keyfn(r), [0, 0])
            cell[keyfn(r)][0] += int(r["correct"])
            cell[keyfn(r)][1] += 1
        desc_cells[keyname] = {k: {"acc": v[0] / v[1], "n": v[1]}
                               for k, v in sorted(cell.items())}

    return {"arm": arm, "base_arm": base, "tier": tier,
            "is_floor": bool(shuffled), "seed": seed,
            "T": R.ARM_CFG[base]["T"], "family": R.ARM_CFG[base]["family"],
            "feature_dim": int(Xtr.shape[1]), "feature_definition": desc,
            "fit_seed": init,
            "dataset": path, "dataset_sha256": meta["dataset_sha256"],
            "train_acc": m.hard_acc(Xtr, ytr), "val_acc": best["v"],
            "heldout_test_acc": m.hard_acc(Xte, yte),
            "train_loss_init": loss0, "train_loss_final": loss_final,
            "grad_norm_final": gnorm, "selected_at": best["where"],
            "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
            "conditioning_descriptors": desc_cells,
            "heldout_items": rows,
            "wall_sec": time.time() - t0}


def agg(rows):
    a = [r["heldout_test_acc"] for r in rows]
    mean = sum(a) / len(a)
    sd = (sum((x - mean) ** 2 for x in a) / (len(a) - 1)) ** 0.5 if len(a) > 1 else 0.0
    return {"mean": mean, "sd": sd, "per_seed": a,
            "seeds": [r["seed"] for r in rows],
            "tier": rows[0]["tier"], "tier_role": TIER_ROLE[rows[0]["tier"]],
            "feature_dim": rows[0]["feature_dim"],
            "feature_definition": rows[0]["feature_definition"],
            "T": rows[0]["T"], "family": rows[0]["family"],
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
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default=R.DATA_A,
                    help="Part A dataset; pass the seed+1000 re-draw for doc 56 "
                         "SS5.1.1's CONFIRM DRAW. The comparator features are a "
                         "function of the ITEMS, so a fresh draw needs fresh fits.")
    ap.add_argument("--ckpt-dir", default=R.CKPT_DIR)
    a = ap.parse_args()
    t0 = time.time()
    # Route the shared machinery at the requested draw. Defaults are the first
    # draw's, so omitting the flags reproduces it exactly.
    R.DATA_A, R.CKPT_DIR = a.data, a.ckpt_dir
    R.CONFIRM_MODE = (os.path.abspath(a.data)
                      != os.path.abspath(R.FIRST_DRAW_DATA))
    out_path = a.out or os.path.join(R.CKPT_DIR, "comparator.json")
    store = R.load_part()
    fm = R.feature_maps()
    print("[exp70-comp] draw=%s (confirm=%s) -> %s"
          % (a.data, R.CONFIRM_MODE, out_path), flush=True)

    frame = {}
    for base in R.BASE_ARMS:
        ang, _t1 = R.arm_angles(base, fm)
        items, _m, _p = R.arm_items(base, store)
        probe = R69.compile_all([i for i in items if i["split"] == "test"], ang)
        for tier in TIERS[R.ARM_CFG[base]["family"]]:
            fn, dim, _d = feature_fn(base, tier, fm)
            assert_frame(probe, fn, dim, "%s::%s" % (base, tier), frame)
        print("[frame] %-13s %s" % (base, TIERS[R.ARM_CFG[base]["family"]]),
              flush=True)

    arms, floors = {}, {}
    for base in R.BASE_ARMS:
        fam = R.ARM_CFG[base]["family"]
        arms[base] = {}
        for tier in TIERS[fam]:
            arms[base][tier] = agg([fit(base, tier, s, store, fm, False)
                                    for s in range(a.seeds)])
            line = ("[exp70-comp] %-13s %-5s d=%-3d T=%d held-out %.4f (sd %.4f)"
                    % (base, tier, arms[base][tier]["feature_dim"],
                       arms[base][tier]["T"], arms[base][tier]["mean"],
                       arms[base][tier]["sd"]))
            if base in R.FLOOR_OF:
                floors.setdefault(base, {})[tier] = agg(
                    [fit(R.FLOOR_OF[base], tier, s, store, fm, True)
                     for s in range(a.seeds)])
                line += "  floor %.4f (sd %.4f)" % (floors[base][tier]["mean"],
                                                    floors[base][tier]["sd"])
            else:
                line += "  [no floor -- doc 56 SS5.3's anchor family]"
            line += ("  train-loss %.4f -> %.4f  |g|=%.1e"
                     % (float(np.mean(arms[base][tier]["train_loss_init_per_seed"])),
                        float(np.mean(arms[base][tier]["train_loss_final_per_seed"])),
                        max(arms[base][tier]["grad_norm_final_per_seed"])))
            print(line, flush=True)

    out = {
        "component": "exp70_comparator",
        "spec": DESIGN_DOC, "spec_commit": R.SPEC_COMMIT,
        "role": ("doc 56 SS8: registered GUARD (no exp70 number may be read as a "
                 "quantum result -- doc 52 SS1 closed advantage for the product-state "
                 "family with an argument, and doc 56 SS0.1 refuses the word for the "
                 "entangled arms MOST of all) and registered ARBITER (SS8.1's "
                 "two-tier reading rule and SS8.4's four-row table), which for the "
                 "first time in this programme separates a CIRCUIT-TRAINABILITY "
                 "failure from an ENCODING failure."),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "tiers_per_family": TIERS, "tier_roles": TIER_ROLE,
        "draw": {"dataset": a.data,
                 "dataset_sha256": store[1]["dataset_sha256"],
                 "is_confirm_draw": bool(R.CONFIRM_MODE),
                 "note": ("the comparator features are a function of the ITEMS, so "
                          "a fresh draw requires fresh fits -- these are NOT the "
                          "first draw's numbers carried over (doc 47 SS16.1)")},
        "reading_rule_doc56_SS8_1": (
            "If (i) the RAW comparator TIES the circuit, THE ENTANGLED MAP ADDED "
            "NOTHING EXTRACTABLE -- the two raw coordinates already carry everything "
            "the circuit got out of them, and the map is a reparameterisation "
            "however exotic its span. If (i) FAILS to tie but (ii) the "
            "FUNCTION-CLASS-MATCHED comparator TIES, THE MAP GENUINELY RESHAPED THE "
            "FUNCTION CLASS -- the information needed a nonlinear reorganisation the "
            "raw coordinates do not offer linearly, and the encoding supplied it."),
        "why_the_outer_square_not_the_amplitudes_doc56_SS8_2": (
            "the 4-dim real amplitude vector is the smaller and more tempting "
            "feature, and it is WRONG: readouts are expectation values, QUADRATIC in "
            "amplitudes, so a linear model on amplitudes is a strictly WEAKER class "
            "than the circuit's and would MANUFACTURE A SPURIOUS CIRCUIT WIN. The "
            "outer square is the function-class-matched object. Stated because "
            "choosing the amplitudes would have been an easy and invisible error."),
        "clifford_consequence_doc56_SS4_5": (
            "the joint-state comparator for N-across is the SAME FUNCTION CLASS as "
            "EA-ctrl's -- <X(x)X> is identically zero under the CZ map and <Y(x)Y> is "
            "identically zero without it, and both sets span the same 9 dimensions. "
            "The comparator therefore CANNOT separate N-across from EA-ctrl, and "
            "that is registered as a PROPERTY OF THE DESIGN rather than discovered "
            "as a null. Any measured N-across vs EA-ctrl difference is TRAINABILITY "
            "or INDUCTIVE BIAS, NEVER CAPACITY."),
        "features": {"frame_check": frame,
                     "identical_to_circuit": (
                         "the 'class' tier consumes the SAME per-referent vectors "
                         "the arms' circuits are linear in. exp70_run.py --smoke "
                         "measures the closure directly: EA-ctrl closes at 3 terms, "
                         "A0-W / P-eps close in the 9-dim product basis, and N-in "
                         "closes in the 6-dim E-in basis while FAILING both the "
                         "3-term and the 9-dim product bases (doc 56 SS7.1's double "
                         "negative)."),
                     "why_the_queried_event": (
                         "doc 55 SS12, inherited: the label is a function of the "
                         "QUERIED event's ordered (agent, patient) pair and the "
                         "category membership of that agent -- doc 47 SS3.2/SS17's "
                         "antisymmetric form with the queried event supplying the "
                         "ordering.")},
        "optimiser": {"fit_seed_base": FIT_SEED, "per_seed_init": "FIT_SEED + seed",
                      "adam_steps": ADAM_STEPS, "adam_lr": ADAM_LR,
                      "lbfgs_rounds": LBFGS_ROUNDS, "lbfgs_max_iter": LBFGS_ITER,
                      "line_search": "strong_wolfe", "batching": "FULL batch",
                      "dtype": "float64",
                      "model_selection": "best HARD argmax accuracy on the val split",
                      "reported_accuracy": "HARD argmax on held-out test",
                      "seeds_vary": ("optimiser INITIALISATION ONLY. The features "
                                     "are frozen constants, so the five fits sit on "
                                     "ONE geometry and the spread is init spread, "
                                     "not data spread.")},
        "arms": arms, "floors": floors,
        "sd_convention": "sample sd, ddof=1", "n_seeds": a.seeds,
        "tie_band": TIE_BAND,
        "function_class_qualifier": (
            "doc 56 SS8.5: 'identical features' makes each comparator a RESTRICTION "
            "of the circuit's conditional function class -- the circuit carries one "
            "form PER CONDITIONING GROUP (12 at T = 1, 72 at T = 2, doc 55 SS5), the "
            "comparator carries one TOTAL. A comparator loss on a T = 2 cell is "
            "therefore even less automatically a circuit win. This qualifier travels "
            "with every number in this file."),
        "wall_time_sec": time.time() - t0,
    }
    os.makedirs(R.CKPT_DIR, exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[exp70-comp] wrote %s (%.0fs)" % (out_path, out["wall_time_sec"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
