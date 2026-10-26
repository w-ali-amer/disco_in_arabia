# -*- coding: utf-8 -*-
"""exp73_comparator.py -- doc 61 SS8: the classical comparator, in THREE TIERS, and
it is THE ARBITER.

doc 61 SS8 registers this BEFORE any exp73 number exists and it ships in the SAME
COMMIT as the runner (SS16.5, doc 53 SS7.3's standing rule). It does two things that
must not be conflated.

  1. A GUARD, and at this rung it is the strongest form the guard has taken.
     doc 61 SS0.3: every gate after intro is feature-independent, so any outcome
     probability is p(x) = x^T M x = <W, v(x)> with v(x) = (x_i x_j)_{i<=j}. **THE
     READOUT IS EXACTLY A QUADRATIC FORM IN THE RAW FEATURES AND THE MATCHED
     CLASSICAL COMPARATOR IS A QUADRATIC (KERNEL) MODEL ON THE SAME FEATURES.** For
     the two-referent decision it is a linear model in the degree-(2,2) bi-monomials
     x_a[i] x_a[j] x_p[k] x_p[l], equivalently the product kernel
     (x_a . x_a')^2 (x_p . x_p')^2. Not similar -- THE SAME OBJECT IN ANOTHER BASIS,
     in one line, with no product formula needed, and the equivalence is ALREADY IN
     THE LITERATURE (doc 61 SS4.1: amplitude-encoding feature maps are kernel methods;
     two copies give the quadratic kernel). **SS0.3 IS A DERIVATION FOR THIS SETTING,
     NOT A DISCOVERY, AND NO WRITE-UP SENTENCE MAY CLAIM IT AS NEW.**
     **NO EXP73 NUMBER MAY APPEAR IN A SENTENCE CONTAINING "ADVANTAGE"** (doc 52 SS1;
     doc 56 SS0.1; doc 58 SS12; doc 61 SS13).

  2. THE ARBITER (SS8.1's reading rule), and SS8.2's registered vacuity.

THE THREE TIERS (doc 61 SS8.1 / SS8.2), REGISTERED EXACTLY
==========================================================
  (i)  INFORMATION-MATCHED    logistic regression on the RAW TRUNCATED VECTORS
       phi = ( x(agent), x(patient) )                       2d-dim
       = 16-dim (A3), 32-dim (A4) -- the same numbers the encoding consumes,
       with NO MAP APPLIED. NOT VACUOUS (16 or 32 dims against 660 items).

  (ii) FUNCTION-CLASS-MATCHED logistic regression on the QUADRATIC (kernel)
       features, agent-first outer product
       phi = vec( v(x_agent) (x) v(x_patient) ),  v(x) = (x_i x_j)_{i<=j}
       = 1296-dim (A3), 18496-dim (A4).
       **VACUOUS AT BOTH RUNGS, AND THIS IS ARITHMETIC, NOT A SURPRISE.** An
       unregularised linear model with 1296 or 18496 free coefficients fitted to
       660 train + val items separates every item BY DIMENSION COUNT ALONE. It
       will return 1.0000 and the number will mean nothing. This is the SAME
       arithmetic fact as doc 61 SS2.4's bound saturation -- d_joint >= n in both
       places -- and the doc treats them as one. It is REPORTED anyway, because
       its vacuity is itself the registered demonstration and suppressing it would
       hide the finding.

  (iii) CAPACITY-MATCHED      L2-REGULARISED logistic regression on the SAME
       quadratic features as (ii), regularisation strength selected on the
       IN-DISTRIBUTION VALIDATION SPLIT ONLY, over the FROZEN grid
           C in {1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2}
       (orchestrator ruling 5, which supersedes SS8.2's six-value body text),
       deterministic fit at the committed fit seed, ONE SELECTION PER ARM, and the
       held-out read ONCE from the selected model.
       **SELECTION IS ON VAL, NEVER ON HELD-OUT.**

THE OBJECTIVE, WRITTEN OUT BECAUSE THE GRID IS FROZEN AND THE CONVENTION IS NOT
==============================================================================
Tier (iii) minimises, over W and b, in float64, full batch:

    mean_i CE(W phi_i + b, y_i)  +  ||W||_F^2 / (2 * C * n_train)     (b UNPENALISED)

which is `sklearn.linear_model.LogisticRegression(penalty='l2', C=C)`'s objective
0.5||w||^2 + C sum_i loss_i divided by (C * n_train) -- the same minimiser, the same
grid semantics. Tiers (i) and (ii) are the SAME objective with the penalty term
removed (exp67/exp68/exp69/exp70's committed unregularised frame, verbatim).

THE READING RULE, doc 61 SS8.1 VERBATIM: if (i) TIES the circuit the map added
nothing extractable; if (i) FAILS to tie but (ii) TIES, the map genuinely reshaped
the function class. **Tier (iii) is the comparator every exp73 criterion that
mentions a comparator reads**; (i) and (ii) are REPORTED.

SS8.3's QUALIFIER TRAVELS WITH EVERY TIER-(iii) NUMBER, IN THE SAME SENTENCE. A
regularised comparator is NO LONGER EXACTLY the circuit's function class -- it is a
RESTRICTED member of it. So a circuit that beats tier (iii) has beaten a
CAPACITY-LIMITED CLASSICAL MODEL, not the quadratic class, and THAT MAY NOT BE
REPORTED AS BEATING THE CLASSICAL FORM OF THE ENCODING. doc 55 SS8.4's restriction
qualifier in a new direction: previously the comparator was a restriction because it
carried one form where the circuit carried many; here it is ADDITIONALLY a
restriction because it is regularised and the circuit is not. Both restrictions are
stated together wherever tier (iii) is quoted.

SS8.4's REGISTERED PREDICTION, recorded so a tie is not reported as a surprise: THE
COMPARATORS MATCH OR EXCEED EVERY ARM. exp67, exp68, exp69 and exp70 all measured
comparator >= circuit on every arm.

THE COMPARATORS GET FLOORS TOO, at the same relabel seeds the circuit floors use
(doc 52 SS3: a gain criterion needs a null exactly as much as an absolute one).

THE FRAME CHECK IS ASSERTED, NOT ASSUMED (doc 56 SS8.3, inherited), and at T = 2 it
runs PER q: wire-ordered features are bit-identical across twins (a NULL INSTRUMENT,
provably 0.5) while agent-ordered features separate EVERY twin pair.

Writes `exp73_ckpt/comparator.json`, and NO results file: doc 53 SS7.3's standing
rule gives the run ONE aggregator, and `exp73_aggregate.py` is it.

    python exp73_comparator.py                  # A3 (stage 1)
    python exp73_comparator.py --rungs A3,A4    # after an A4 stage 2
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
import exp73_run as R

DESIGN_DOC = ("qnlp_private_docs/61_exp73_amplitude.md (REGISTERED 2026-08-28, "
              "orchestrator rulings 1-5) SS8")
FIT_SEED = 73001                        # registered; per-seed init = FIT_SEED + seed
ADAM_STEPS, ADAM_LR = 3000, 0.05
LBFGS_ROUNDS, LBFGS_ITER = 10, 200
TIE_BAND = 0.02

# ORCHESTRATOR RULING 5, FROZEN. Seven values; this SUPERSEDES doc 61 SS8.2's
# six-value body text and is not tuned, extended or trimmed after any number exists.
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]

TIERS = ["raw", "class", "l2"]
TIER_ROLE = {
    "raw": ("(i) INFORMATION-MATCHED -- logistic regression on the RAW TRUNCATED "
            "vectors phi = (x(agent), x(patient)), 2d-dim, NO MAP APPLIED. The same "
            "numbers the encoding consumes. NOT vacuous: 16 (A3) or 32 (A4) "
            "dimensions against 660 train+val items."),
    "class": ("(ii) FUNCTION-CLASS-MATCHED -- UNREGULARISED logistic regression on "
              "the QUADRATIC features vec(v(x_a) (x) v(x_p)), v(x) = (x_i x_j)_{i<=j}. "
              "**REGISTERED IN ADVANCE AS VACUOUS** (doc 61 SS8.2): 1296 (A3) or "
              "18496 (A4) free coefficients against 660 items separates every item "
              "by dimension count alone, so it will return 1.0000 and THE NUMBER "
              "WILL MEAN NOTHING. Reported because its vacuity IS the registered "
              "demonstration -- the same arithmetic fact as SS2.4's bound "
              "saturation."),
    "l2": ("(iii) CAPACITY-MATCHED -- L2-REGULARISED logistic regression on the SAME "
           "quadratic features, C selected on the IN-DISTRIBUTION VAL SPLIT ONLY "
           "over the frozen grid, held-out read ONCE from the selected model. **THIS "
           "IS THE TIER EVERY EXP73 CRITERION READS**, and doc 61 SS8.3's "
           "restriction qualifier travels with every number it produces."),
}
RESTRICTION_QUALIFIER = (
    "doc 61 SS8.3, IN THE SAME SENTENCE AS THE NUMBER: tier (iii) is a RESTRICTED "
    "member of the circuit's function class, twice over -- (a) the circuit carries "
    "one form PER CONDITIONING GROUP while the comparator carries one form TOTAL "
    "(doc 55 SS8.4), and (b) the comparator is REGULARISED and the circuit is not. A "
    "circuit that beats tier (iii) has beaten A CAPACITY-LIMITED CLASSICAL MODEL, "
    "NOT THE QUADRATIC CLASS, and that may not be reported as beating the classical "
    "form of the encoding.")


# ----------------------------------------------------------------- features
def feature_fn(rung, tier, fm):
    """(callable(agent, patient) -> phi, dim, description). AGENT-FIRST OF THE
    QUERIED EVENT (doc 55 SS12, inherited: the label is a function of the queried
    event's ordered pair and the category membership of that agent)."""
    st = fm["amp"]["states"][rung]
    d = R.RUNGS[rung]["d"]
    if tier == "raw":
        def phi_raw(a, p):
            return np.concatenate([st[a], st[p]])
        return phi_raw, 2 * d, (
            "(x(agent), x(patient)) -- the RAW truncated, sign-pinned, unit-"
            "normalised principal scores, %d-dim, NO map applied" % (2 * d))
    vref = fm["amp"]["features"][rung]                # v(x) = (x_i x_j)_{i<=j}
    dref = len(next(iter(vref.values())))

    def phi_quad(a, p):
        return np.outer(vref[a], vref[p]).ravel()
    return phi_quad, dref * dref, (
        "vec( v(x_agent) (x) v(x_patient) ), v(x) = (x_i x_j)_{i<=j} the OUTER "
        "SQUARE (d_ref = %d, d_joint = %d); equivalently the product kernel "
        "(x_a . x_a')^2 (x_p . x_p')^2. **THE OUTER SQUARE AND NOT THE AMPLITUDES**: "
        "readouts are expectation values, QUADRATIC in amplitudes, so a linear model "
        "on the amplitudes would be a strictly WEAKER class than the circuit's and "
        "would MANUFACTURE A SPURIOUS CIRCUIT WIN (doc 56 SS8.2; doc 61 SS4.1 records "
        "that this 'would have been an easy and invisible error')." % (dref, dref * dref))


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
    """doc 56 SS8.3, inherited; at T = 2 THE CHECK RUNS PER q."""
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
            ("q=%s: the wire frame SEPARATES twins -- the null instrument is not "
             "null" % q)
        assert n_agent_same == 0, \
            "q=%s: the agent frame FAILS to separate %d twin pairs" % (q, n_agent_same)
        per_q[str(q)] = {"twin_pairs_checked": len(twins),
                         "wire_frame_identical_twin_pairs": n_wire_same,
                         "agent_frame_identical_twin_pairs": n_agent_same}
    got = len(fn(*queried_pair(cs_list[0])))
    assert got == dim, "feature dim %d != registered %d" % (got, dim)
    rep[key] = {"dim": got, "per_q": per_q,
                "wire_frame_is_a_null_instrument": True,
                "frame": "agent-first ordering of the QUERIED event"}


# -------------------------------------------------------------------- model
class LogReg(torch.nn.Module):
    """Binary logistic regression, float64. Two logits so the head matches the
    circuit's two-effect readout shape. `l2_C` None = unregularised (tiers i, ii)."""

    def __init__(self, d, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = torch.nn.Parameter(
            (torch.rand(2, d, generator=g, dtype=torch.float64) - 0.5) * (2.0 / 3.0))
        self.b = torch.nn.Parameter(
            (torch.rand(2, generator=g, dtype=torch.float64) - 0.5) * (2.0 / 3.0))

    def logits(self, X):
        return X @ self.W.T + self.b

    def loss(self, X, y, l2_C=None):
        L = torch.nn.functional.cross_entropy(self.logits(X), y)
        if l2_C is not None:
            # sklearn's 0.5||w||^2 + C sum_i loss_i, divided by (C * n): the SAME
            # minimiser. The BIAS IS UNPENALISED, as sklearn leaves it.
            L = L + (self.W ** 2).sum() / (2.0 * float(l2_C) * float(X.shape[0]))
        return L

    def hard_acc(self, X, y):
        with torch.no_grad():
            return float((torch.argmax(self.logits(X), 1) == y).double().mean())


def _fit_once(Xtr, ytr, Xva, yva, init, l2_C):
    """exp67 / exp68 / exp69 / exp70's committed frame, verbatim: float64, full-batch
    Adam then LBFGS/strong-Wolfe, checkpoint selected on HARD val accuracy."""
    torch.manual_seed(init)
    m = LogReg(Xtr.shape[1], init)
    with torch.no_grad():
        loss0 = float(m.loss(Xtr, ytr, l2_C))
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
        L = m.loss(Xtr, ytr, l2_C)
        L.backward()
        opt.step()
        if step % 10 == 0:
            record("adam%d" % step)
    lb = torch.optim.LBFGS(m.parameters(), max_iter=LBFGS_ITER, history_size=50,
                           tolerance_grad=1e-12, tolerance_change=1e-14,
                           line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        L = m.loss(Xtr, ytr, l2_C)
        L.backward()
        return L

    for _r in range(LBFGS_ROUNDS):
        lb.step(closure)
        record("lbfgs%d" % _r)
    with torch.no_grad():
        loss_final = float(m.loss(Xtr, ytr, l2_C))
    m.zero_grad()
    m.loss(Xtr, ytr, l2_C).backward()
    gnorm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters())))
    m.load_state_dict(best["state"])
    return m, {"val_acc": best["v"], "selected_at": best["where"],
               "train_loss_init": loss0, "train_loss_final": loss_final,
               "grad_norm_final": gnorm}


def fit(rung, T, tier, seed, store, fm, shuffled):
    """One deterministic fit. For tier (iii) the FROZEN C grid is swept and the
    winner is chosen on HARD VAL ACCURACY -- never on held-out -- with ties broken
    towards the SMALLER C, i.e. towards MORE regularisation, registered here so the
    tie-break cannot be chosen after the fact."""
    arm = "%s-T%d-uniform" % (rung, T)               # features do not depend on init
    fn, dim, desc = feature_fn(rung, tier, fm)
    items, meta, path = R.arm_items(arm, store)
    use = R69.relabel_shuffle_q(items, seed) if shuffled else items
    ang, _t1 = R.arm_angles(arm, fm)
    data = {s: R69.compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}
    Xtr, ytr = build_xy(data["train"], fn)
    Xva, yva = build_xy(data["val"], fn)
    Xte, yte = build_xy(data["test"], fn)

    init = FIT_SEED + seed
    t0 = time.time()
    sweep = []
    if tier == "l2":
        best = None
        for C in C_GRID:
            m, info = _fit_once(Xtr, ytr, Xva, yva, init, C)
            row = {"C": C, "val_acc": info["val_acc"],
                   "train_acc": m.hard_acc(Xtr, ytr),
                   "train_loss_final": info["train_loss_final"]}
            sweep.append(row)
            # ties -> SMALLER C (more regularisation). Registered, not chosen later.
            if best is None or info["val_acc"] > best[1]["val_acc"] + 1e-15:
                best = (m, dict(info, C=C))
        m, info = best
        selected_C = info["C"]
    else:
        m, info = _fit_once(Xtr, ytr, Xva, yva, init, None)
        selected_C = None

    with torch.no_grad():
        pred = torch.argmax(m.logits(Xte), 1).tolist()
    rows = []
    for cs, p in zip(data["test"], pred):
        r = dict(cs.meta)
        r.update({"pred": int(p), "target": int(cs.answer),
                  "correct": bool(int(p) == cs.answer)})
        rows.append(r)
    desc_cells = {}
    for keyname, keyfn in (("by_q", lambda r: "q%d" % r["query_index"]),
                           ("by_order_pair", lambda r: "/".join(r["order_flags"]))):
        cell = {}
        for r in rows:
            cell.setdefault(keyfn(r), [0, 0])
            cell[keyfn(r)][0] += int(r["correct"])
            cell[keyfn(r)][1] += 1
        desc_cells[keyname] = {k: {"acc": v[0] / v[1], "n": v[1]}
                               for k, v in sorted(cell.items())}

    return {"rung": rung, "T": T, "tier": tier, "is_floor": bool(shuffled),
            "seed": seed, "feature_dim": int(Xtr.shape[1]),
            "feature_definition": desc, "fit_seed": init,
            "selected_C": selected_C, "C_grid": C_GRID if tier == "l2" else None,
            "C_sweep": sweep or None,
            "C_selection_rule": ("HARD val accuracy over the FROZEN grid, ties -> "
                                 "SMALLER C (more regularisation); selection is on "
                                 "the IN-DISTRIBUTION VAL SPLIT ONLY and the "
                                 "held-out is read ONCE from the selected model"
                                 if tier == "l2" else None),
            "dataset": path, "dataset_sha256": meta["dataset_sha256"],
            "train_acc": m.hard_acc(Xtr, ytr), "val_acc": info["val_acc"],
            "heldout_test_acc": m.hard_acc(Xte, yte),
            "train_loss_init": info["train_loss_init"],
            "train_loss_final": info["train_loss_final"],
            "grad_norm_final": info["grad_norm_final"],
            "selected_at": info["selected_at"],
            "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
            "conditioning_descriptors": desc_cells, "heldout_items": rows,
            "wall_sec": time.time() - t0}


def agg(rows):
    a = [r["heldout_test_acc"] for r in rows]
    mean = sum(a) / len(a)
    sd = (sum((x - mean) ** 2 for x in a) / (len(a) - 1)) ** 0.5 if len(a) > 1 else 0.0
    out = {"mean": mean, "sd": sd, "per_seed": a,
           "seeds": [r["seed"] for r in rows],
           "tier": rows[0]["tier"], "tier_role": TIER_ROLE[rows[0]["tier"]],
           "feature_dim": rows[0]["feature_dim"],
           "feature_definition": rows[0]["feature_definition"],
           "rung": rows[0]["rung"], "T": rows[0]["T"],
           "val_per_seed": [r["val_acc"] for r in rows],
           "train_per_seed": [r["train_acc"] for r in rows],
           "train_loss_init_per_seed": [r["train_loss_init"] for r in rows],
           "train_loss_final_per_seed": [r["train_loss_final"] for r in rows],
           "grad_norm_final_per_seed": [r["grad_norm_final"] for r in rows],
           "conditioning_descriptors_per_seed":
               [r["conditioning_descriptors"] for r in rows],
           "runs": rows}
    if rows[0]["tier"] == "l2":
        out["selected_C_per_seed"] = [r["selected_C"] for r in rows]
        out["restriction_qualifier_doc61_SS8_3"] = RESTRICTION_QUALIFIER
    if rows[0]["tier"] == "class":
        out["vacuity_doc61_SS8_2"] = {
            "d_joint": rows[0]["feature_dim"],
            "n_train_plus_val": rows[0]["n_train"] + rows[0]["n_val"],
            "separates_by_dimension_count": bool(
                rows[0]["feature_dim"] >= rows[0]["n_train"] + rows[0]["n_val"]),
            "reading": ("doc 61 SS8.2: an unregularised linear model with %d free "
                        "coefficients fitted to %d train + val items separates every "
                        "item BY DIMENSION COUNT ALONE, so a value at or near 1.0000 "
                        "MEANS NOTHING. This is the same arithmetic fact as SS2.4's "
                        "bound saturation, and tier (iii) exists because of it."
                        % (rows[0]["feature_dim"],
                           rows[0]["n_train"] + rows[0]["n_val"]))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=R.N_SEEDS)
    ap.add_argument("--rungs", default="A3",
                    help="comma-separated; default A3 (stage 1). A4 is orchestrator "
                         "ruling 2's CONDITIONAL SECOND STAGE.")
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-dir", default=R.CKPT_DIR)
    a = ap.parse_args()
    t0 = time.time()
    R.CKPT_DIR = a.ckpt_dir
    out_path = a.out or os.path.join(a.ckpt_dir, "comparator.json")
    rungs = [x for x in a.rungs.split(",") if x]
    tiers = [x for x in a.tiers.split(",") if x]
    store = R.load_part()
    fm = R.feature_maps()
    os.makedirs(a.ckpt_dir, exist_ok=True)
    print("[exp73-comp] rungs=%r tiers=%r -> %s" % (rungs, tiers, out_path), flush=True)

    frame = {}
    for rung in rungs:
        for T in (1, 2):
            arm = "%s-T%d-uniform" % (rung, T)
            ang, _t1 = R.arm_angles(arm, fm)
            items, _m, _p = R.arm_items(arm, store)
            probe = R69.compile_all([i for i in items if i["split"] == "test"], ang)
            for tier in tiers:
                fn, dim, _d = feature_fn(rung, tier, fm)
                assert_frame(probe, fn, dim, "%s-T%d::%s" % (rung, T, tier), frame)
            print("[frame] %s-T%d %r" % (rung, T, tiers), flush=True)

    arms, floors = {}, {}
    for rung in rungs:
        for T in (1, 2):
            cell = "%s-T%d" % (rung, T)
            arms[cell], floors[cell] = {}, {}
            for tier in tiers:
                arms[cell][tier] = agg([fit(rung, T, tier, s, store, fm, False)
                                        for s in range(a.seeds)])
                floors[cell][tier] = agg([fit(rung, T, tier, s, store, fm, True)
                                          for s in range(a.seeds)])
                print("[exp73-comp] %-8s %-5s d=%-6d held-out %.4f (sd %.4f)  "
                      "floor %.4f (sd %.4f)%s"
                      % (cell, tier, arms[cell][tier]["feature_dim"],
                         arms[cell][tier]["mean"], arms[cell][tier]["sd"],
                         floors[cell][tier]["mean"], floors[cell][tier]["sd"],
                         ("  C=%r" % arms[cell][tier]["selected_C_per_seed"]
                          if tier == "l2" else "")), flush=True)

    out = {
        "component": "exp73_comparator",
        "spec": DESIGN_DOC, "runner_spec": R.SPEC,
        "role": ("doc 61 SS8: registered GUARD and registered ARBITER. THE GUARD IS "
                 "STRONGER HERE THAN AT ANY PREVIOUS RUNG: for the amplitude map the "
                 "matched classical comparator is not a CONSTRUCTION (doc 52 SS1.1's "
                 "factorising kernel) but THE SAME OBJECT IN ANOTHER BASIS -- the "
                 "readout IS a quadratic form and the comparator IS a quadratic "
                 "kernel model on the same features, in one line. doc 52 SS1's "
                 "prohibition applies in its STRONGEST form to date."),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "rungs": rungs, "tiers": tiers, "tier_roles": TIER_ROLE,
        "dataset": store[2], "dataset_sha256": store[1]["dataset_sha256"],
        "quadratic_form_identity_doc61_SS0_3": (
            "p(x) = <psi(x)| O |psi(x)> = x^T M x = <W, v(x)>, v(x) = (x_i x_j)_{i<=j}, "
            "d_ref = d(d+1)/2; for the two-referent decision a linear model in the "
            "degree-(2,2) bi-monomials x_a[i] x_a[j] x_p[k] x_p[l], equivalently the "
            "product kernel (x_a . x_a')^2 (x_p . x_p')^2. Verified to 4.4e-16 on "
            "random symmetric M at d = 2, 4, 8, 16 "
            "(diagnostic_amplitude_bounds.json :: selftest.checks."
            "quadratic_form_identity)."),
        "not_novel_doc61_SS4_1": (
            "THE EQUIVALENCE IS TEXTBOOK AND IS RECORDED FIRST AND PLAINLY. The "
            "literature states directly that amplitude-encoding feature maps are "
            "kernel methods -- with unit data vectors the amplitude feature map "
            "yields a LINEAR kernel and preparing MULTIPLE COPIES implements "
            "POLYNOMIAL kernels, two copies giving exactly the QUADRATIC kernel "
            "(Quantum Sci. Technol. 2019; arXiv:2206.02806; arXiv:2202.02151). doc 61 "
            "SS0.3 IS A DERIVATION FOR THIS SETTING, NOT A DISCOVERY, and NO WRITE-UP "
            "SENTENCE MAY CLAIM IT AS NEW."),
        "reading_rule_doc61_SS8_1": (
            "if (i) the INFORMATION-MATCHED comparator TIES the circuit, THE MAP "
            "ADDED NOTHING EXTRACTABLE; if (i) FAILS to tie but (ii) the "
            "FUNCTION-CLASS-MATCHED comparator TIES, THE MAP GENUINELY RESHAPED THE "
            "FUNCTION CLASS. **Tier (iii) is the comparator every exp73 criterion "
            "that mentions a comparator reads**; (i) and (ii) are REPORTED."),
        "tier_iii_objective": (
            "mean_i CE(W phi_i + b, y_i) + ||W||_F^2 / (2 C n_train), BIAS "
            "UNPENALISED -- sklearn.linear_model.LogisticRegression(penalty='l2', "
            "C=C)'s objective 0.5||w||^2 + C sum_i loss_i divided by (C n_train), "
            "the same minimiser and the same grid semantics. Tiers (i) and (ii) are "
            "the same objective with the penalty removed."),
        "C_grid_FROZEN_ruling5": C_GRID,
        "C_grid_note": ("orchestrator ruling 5 freezes SEVEN values and supersedes "
                        "doc 61 SS8.2's six-value body text. The grid is not tuned, "
                        "extended or trimmed after any number exists; selection is "
                        "on the IN-DISTRIBUTION VAL SPLIT ONLY and the held-out is "
                        "read ONCE from the selected model."),
        "restriction_qualifier_doc61_SS8_3": RESTRICTION_QUALIFIER,
        "registered_prediction_doc61_SS8_4": (
            "THE COMPARATORS MATCH OR EXCEED EVERY ARM. doc 47 SS14 and doc 52 SS1.1's "
            "standing expectation; exp67, exp68, exp69 and exp70 all measured "
            "comparator >= circuit on every arm. Recorded IN ADVANCE so a tie is not "
            "reported as a surprise."),
        "features": {"frame_check": frame,
                     "why_the_queried_event": (
                         "doc 55 SS12, inherited: the label is a function of the "
                         "QUERIED event's ordered (agent, patient) pair and the "
                         "category membership of that agent."),
                     "identical_to_the_circuit": (
                         "the quadratic tiers consume the SAME per-referent vectors "
                         "the amplitude circuit is linear in -- v(x), the outer "
                         "square, which doc 61 SS7.2's intro-state assertions pin the "
                         "circuit's own register to amplitude by amplitude.")},
        "optimiser": {"fit_seed_base": FIT_SEED, "per_seed_init": "FIT_SEED + seed",
                      "adam_steps": ADAM_STEPS, "adam_lr": ADAM_LR,
                      "lbfgs_rounds": LBFGS_ROUNDS, "lbfgs_max_iter": LBFGS_ITER,
                      "line_search": "strong_wolfe", "batching": "FULL batch",
                      "dtype": "float64",
                      "model_selection": "best HARD argmax accuracy on the val split",
                      "reported_accuracy": "HARD argmax on held-out test",
                      "seeds_vary": ("optimiser INITIALISATION ONLY. The features are "
                                     "frozen constants, so the five fits sit on ONE "
                                     "geometry and the spread is init spread, not "
                                     "data spread.")},
        "arms": arms, "floors": floors,
        "floors_note": ("doc 52 SS3: a gain criterion needs a null exactly as much as "
                        "an absolute one, so every comparator tier carries a matched "
                        "S-shuffle floor at the SAME relabel seeds the circuit floors "
                        "use (`exp69_run.relabel_shuffle_q`, imported unchanged)."),
        "sd_convention": "sample sd, ddof=1", "n_seeds": a.seeds,
        "tie_band": TIE_BAND,
        "advantage_prohibition": (
            "NO EXP73 NUMBER MAY APPEAR IN A SENTENCE CONTAINING 'ADVANTAGE' "
            "(doc 52 SS1; doc 56 SS0.1; doc 58 SS12; doc 61 SS0.3 / SS13). For this "
            "encoding the classical form is MORE IMMEDIATE than for any previous "
            "rung, and doc 61 states that as a RESULT, not a caveat."),
        "wall_time_sec": time.time() - t0,
    }
    json.dump(out, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[exp73-comp] wrote %s (%.0fs)" % (out_path, out["wall_time_sec"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
