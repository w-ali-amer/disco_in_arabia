# -*- coding: utf-8 -*-
"""exp76_comparator.py -- doc 63 SS10. THE CLASSICAL COMPARATOR, REFIT AT EVERY
CELL, AND THE GAP IS THE REGISTERED QUANTITY.

SS10.1's load-bearing criteria choice, stated first: raw accuracy versus n
conflates TASK DIFFICULTY with TRAINABILITY, and the programme has no way to
separate them from a single curve. So:

    gap(n, T) = heldout_mean_circuit(n, T) - heldout_mean_comparator(n, T)

and every headline criterion reads the GAP, not raw accuracy.

THE THREE TIERS (SS10.2), agent-first of the queried event:

  (i)   INFORMATION-MATCHED   phi_i = (theta_agent(q), theta_patient(q))
                              2-dim. The raw numbers the encoding consumes.
  (ii)  COMMITTED FORM        phi_ii = vec(u(theta_a) (x) u(theta_p))
                              u(t) = (1, cos t, sin t) -> 9-dim, AT EVERY n.
                              THE ARBITER (doc 61 SS10.5).
  (iii) DISTRACTOR-AWARE      phi_iii = phi_ii (+) mean_d u(theta_d)
                                               (+) elementwise-max_d u(theta_d)
                              over the n-2 non-queried cast members -> 15-dim.
                              PERMUTATION-INVARIANT, because the distractors are
                              exchangeable and an ordered encoding of them would
                              be a wire-index artefact (SS4.5 L3).

SS10.3: tier (ii) is 9-DIMENSIONAL AT EVERY n, deliberately. The naive
function-class-matched comparator at n referents would be the full outer
product -- 3^n = 81 at n = 4, 6561 at n = 8 -- and would be vacuous by dimension
count from n = 8 onward (doc 61 SS8.2's failure mode). It is NOT BUILT (SS17
item 5). Tier (iii)'s 15 dimensions against 660 items is a 44:1 ratio at worst,
so no regularised tier is bought and doc 61 SS8.3's regularisation qualifier does
NOT apply to this doc.

SS10.6 THE RESTRICTION QUALIFIER, AND AT SCALE IT GROWS. The comparator carries
ONE form total while the circuit carries one per conditioning group: 240 groups
at T = 2, 23 040 at T = 4, 276 480 at T = 6. The comparator is therefore a MUCH
stronger restriction at T = 6 than it was at exp69's T = 2, and a comparator loss
is correspondingly LESS a circuit win. The group count is NOT a count of
independent functions to learn -- the six verb blocks are shared parameters
reused across every group -- and that qualifier travels too.

SS10.7 REGISTERED PREDICTION: the comparators MATCH OR EXCEED every arm
(measured on every arm of exp67, exp68, exp69, exp70 and exp73). Recorded so a
tie is not reported as a surprise.

THE FIT is `exp69_comparator.fit`'s frame VERBATIM -- float64, full-batch Adam
then LBFGS/strong-Wolfe, checkpoint selected on HARD val accuracy, hard argmax
accuracy reported on held-out. Only the feature vector changes.

Writes `exp76_ckpt/comparator.json`. IT WRITES NO RESULTS FILE.
"""

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch

import exp76_run as R
import exp76_datagen as D76

OUT = os.path.join(R.CKPT_DIR, "comparator.json")
FIT_SEED = 76001                        # registered; per-seed init = FIT_SEED + seed
ADAM_STEPS, ADAM_LR = 3000, 0.05        # exp67/68/69/75's frame, verbatim
LBFGS_ROUNDS, LBFGS_ITER = 10, 200
TIERS = ("i_raw", "ii_committed", "iii_distractor")
FLAT_TOL = 1e-9                         # doc 63 SS10.4

# doc 63 SS10.6's table, computed from T distinct verbs of 6, 2 surface orders
# per event and T query positions.
CONDITIONING_GROUPS = {2: 240, 4: 23040, 6: 276480}


def _u(t):
    return np.array((1.0, math.cos(t), math.sin(t)))


def features(cs, tier, angles):
    """SS10.2's three tiers, from the QUERIED event, AGENT-FIRST."""
    noun = {w: n for w, n, _a in cs.intro}
    _v, wa, wp = cs.events[cs.qidx - 1]
    ta, tp = angles[noun[wa]], angles[noun[wp]]
    if tier == "i_raw":
        return np.array([float(ta), float(tp)])
    core = np.outer(_u(ta), _u(tp)).ravel()
    if tier == "ii_committed":
        return core
    d = [_u(angles[noun[w]]) for w in sorted(noun)
         if w not in (wa, wp)]
    if d:
        mean_d = np.mean(d, axis=0)
        max_d = np.max(d, axis=0)
    else:
        # n = 2 carries no distractors. Zeros, RECORDED rather than assumed: at
        # n = 2 tier (iii) is tier (ii) padded with two constant blocks, so its
        # held-out value must equal tier (ii)'s up to optimisation noise and the
        # extra dimensions carry no information. SS10.2's tier is defined over
        # "the n-2 non-queried cast members" and at n = 2 that set is empty.
        mean_d = np.zeros(3)
        max_d = np.zeros(3)
    return np.concatenate([core, mean_d, max_d])


def build_xy(cs_list, tier, angles):
    X = torch.tensor(np.asarray([features(c, tier, angles) for c in cs_list]),
                     dtype=torch.float64)
    y = torch.tensor([c.answer for c in cs_list], dtype=torch.long)
    return X, y


class LogReg(torch.nn.Module):
    """exp69_comparator.LogReg, verbatim. Binary logistic regression, float64;
    two logits so the head matches the circuit's two-effect readout shape."""

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


def fit(arm, tier, seed, store, fm):
    """One deterministic fit -- exp69_comparator's frame, verbatim."""
    base = R.base_arm(arm)
    items, _meta, _path = R.arm_items(arm, store)
    ang = R.arm_angles(arm, fm)
    use = (R.relabel_shuffle_q76(items, seed)
           if arm.startswith("S-shuffle") else items)
    data = {s: R.compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}
    Xtr, ytr = build_xy(data["train"], tier, ang)
    Xva, yva = build_xy(data["val"], tier, ang)
    Xte, yte = build_xy(data["test"], tier, ang)

    init = FIT_SEED + seed
    torch.manual_seed(init)
    m = LogReg(Xtr.shape[1], init)
    best = {"v": -1.0, "where": None, "state": None}

    def record(where):
        v = m.hard_acc(Xva, yva)
        if v > best["v"]:
            best.update({"v": v, "where": where,
                         "state": {k: t.detach().clone()
                                   for k, t in m.state_dict().items()}})

    record("init")
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

    m.load_state_dict(best["state"])
    with torch.no_grad():
        pred = torch.argmax(m.logits(Xte), 1).tolist()
    rows = []
    for cs, p in zip(data["test"], pred):
        r = dict(cs.meta)
        r.update({"pred": int(p), "target": int(cs.answer),
                  "correct": bool(int(p) == cs.answer)})
        rows.append(r)
    cfg = R.ARM_CFG[base]
    return {"arm": arm, "base_arm": base, "tier": tier, "seed": seed,
            "n": cfg["n"], "T": cfg["T"], "inventory": cfg["inventory"],
            "feature_dim": int(Xtr.shape[1]),
            "train_acc": m.hard_acc(Xtr, ytr), "val_acc": best["v"],
            "heldout_test_acc": m.hard_acc(Xte, yte),
            "selected_at": best["where"],
            "n_train": len(data["train"]), "n_test": len(data["test"]),
            "rows": rows}


def agg(vals):
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "per_seed": [float(x) for x in a], "n_seeds": len(a)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=R.N_SEEDS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    store = R.load_store()
    fm = R.feature_maps()
    t0 = time.time()
    rep = {"component": "exp76 comparator (doc 63 SS10)", "spec": R.SPEC,
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "numpy": np.__version__,
           "fit_seed": FIT_SEED, "adam": [ADAM_STEPS, ADAM_LR],
           "lbfgs": [LBFGS_ROUNDS, LBFGS_ITER],
           "sd_convention": "sample sd, ddof=1",
           "tiers": {"i_raw": "phi = (theta_agent(q), theta_patient(q)), 2-dim",
                     "ii_committed": ("phi = vec(u(theta_a) (x) u(theta_p)), "
                                      "9-dim AT EVERY n -- THE ARBITER"),
                     "iii_distractor": ("phi_ii (+) mean_d u(theta_d) (+) "
                                        "elementwise-max_d u(theta_d), 15-dim, "
                                        "PERMUTATION-INVARIANT")},
           "arms": {}, "floors": {}}

    arms = list(R.ANCHOR_ARMS) + R.TRAINED_CELLS
    for arm in arms:
        rep["arms"][arm] = {}
        for tier in TIERS:
            vals, dim = [], None
            for s in range(a.seeds):
                r = fit(arm, tier, s, store, fm)
                vals.append(r["heldout_test_acc"])
                dim = r["feature_dim"]
            rep["arms"][arm][tier] = agg(vals)
            rep["arms"][arm][tier]["feature_dim"] = dim
            print("[exp76 comp] %-24s %-16s dim=%2d heldout=%.4f"
                  % (arm, tier, dim, rep["arms"][arm][tier]["mean"]), flush=True)
    for cell in R.TRAINED_CELLS:
        arm = R.FLOOR_OF[cell]
        rep["floors"][arm] = {}
        for tier in TIERS:
            vals = [fit(arm, tier, s, store, fm)["heldout_test_acc"]
                    for s in range(a.seeds)]
            rep["floors"][arm][tier] = agg(vals)
        print("[exp76 comp] %-24s floors done" % arm, flush=True)

    # ---- SS10.4 `comparator_flat_in_n`, from the FITTED tier-(ii) values -----
    # The dataset-level assertion already ran at smoke (SS10.4 is asserted as a
    # DATASET property before any training); this is the same statement read off
    # the fits, which is the form SS12.3's boolean takes.
    groups = {}
    for cell in R.TRAINED_CELLS:
        cfg = D76.CELL_CFG[cell]
        groups.setdefault(D76.core_key(cfg), []).append(cell)
    flat = {}
    for k, cells in sorted(groups.items()):
        if len(cells) < 2:
            flat[k] = {"cells": cells, "assertion": None,
                       "why_null": ("doc 63 SS10.4's scope: the assertion runs "
                                    "across n WITHIN one inventory. This group "
                                    "carries one n, so it is VACUOUS and is "
                                    "stored as null, NOT as TRUE.")}
            continue
        v = [rep["arms"][c]["ii_committed"]["mean"] for c in cells]
        spread = float(max(v) - min(v))
        flat[k] = {"cells": cells, "tier_ii_heldout": v, "spread": spread,
                   "tolerance": FLAT_TOL, "assertion": bool(spread <= FLAT_TOL)}
    rep["comparator_flat_in_n_doc63_SS10_4"] = flat
    rep["comparator_flat_in_n"] = (
        all(v["assertion"] for v in flat.values() if v["assertion"] is not None))
    rep["flat_reading"] = (
        "doc 63 SS10.4: a comparator that MOVES along n has caught a leak -- an "
        "n-dependent information channel the V-suite missed, or a core that was "
        "not shared. BLOCKING: if it fails, the n-axis cells are not read, "
        "SS12.5's booleans are stored as null (NEVER as FALSE), and the finding "
        "is a GENERATOR DEFECT. Why 1e-9 and not 0.02: doc 55 SS6.7's tie band "
        "is a comparator-versus-circuit band ON THE SAME ITEMS; applied across "
        "independently drawn cells it would fire on a healthy generator roughly "
        "half the time. SS6.2.1 removes the noise instead of widening the band.")
    rep["restriction_qualifier_doc63_SS10_6"] = {
        "conditioning_groups_by_T": CONDITIONING_GROUPS,
        "note": ("the comparator carries ONE form total while the circuit "
                 "carries one per conditioning group. At T = 6 that is 276 480 "
                 "groups against exp69's 240 at T = 2, so the comparator is a "
                 "MUCH stronger restriction here and a comparator loss is "
                 "correspondingly LESS a circuit win. The group count is NOT a "
                 "count of independent functions to learn -- the six verb "
                 "blocks are shared parameters reused across every group.")}
    rep["registered_prediction_doc63_SS10_7"] = (
        "the comparators MATCH OR EXCEED every arm (doc 47 SS14; doc 52 SS1.1; "
        "measured on every arm of exp67, exp68, exp69, exp70 and exp73). "
        "Recorded so a tie is not reported as a surprise.")
    rep["wall_time_sec"] = time.time() - t0
    json.dump(rep, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[exp76 comp] wrote %s (%.0fs)" % (a.out, rep["wall_time_sec"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
