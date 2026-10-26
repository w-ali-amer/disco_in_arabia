# -*- coding: utf-8 -*-
"""exp75_comparator.py -- doc 62 SS10, THE THREE TIERS, AND WHICH ONE ARBITRATES.

doc 62 SS10.1's subtlety, stated first: doc 55 SS8.1's comparator is logistic
regression on the IDENTICAL per-referent features, joint by outer product,
agent-first -- `u(theta_a) (x) u(theta_p)` with `u(t) = (1, cos t, sin t)`, 9-dim,
which returned 0.9200 for every FA arm. FOR EXP75 THAT CONSTRUCTION IS AMBIGUOUS,
because the circuit's EFFECTIVE per-referent feature is `U_L(theta)|0>` with
`U_L` TRAINED: the features the circuit ends up using do not exist until after
the run, and a comparator fitted on them would be fitted on an object the circuit
chose. THREE TIERS ARE REGISTERED AND WHICH ONE ARBITRATES IS FIXED IN ADVANCE.

  (i)   INFORMATION-MATCHED   logistic regression on the RAW ANGLE PAIR
        phi = (theta_agent(q), theta_patient(q)), 2-dim -- the same numbers the
        encoding consumes, no map applied. Not vacuous (2 dims, 660 items).

  (ii)  SPAN-MATCHED AT DEGREE L   logistic regression on the joint harmonic
        features u_L(theta_a) (x) u_L(theta_p),
        u_L(t) = (1, cos t, sin t, ..., cos Lt, sin Lt)
        -> 25-dim (L = 2), 49-dim (L = 3).             ***THE ARBITER (SS10.3)***

  (iii) THE COMMITTED SINGLE-ENCODE COMPARATOR, 9-dim, value 0.9200, IMPORTED
        FROM `results_exp69.json` AND NEVER RE-FITTED -- the classical
        counterpart of the baseline the lift criteria read against.

doc 62 SS10.2: TIER (ii) IS NOT VACUOUS HERE AND THE ARITHMETIC IS THE REASON. doc
61 SS8.2 had to register a capacity-matched tier because 1296 and 18496 joint
dimensions against 660 items separate every item by dimension count alone. HERE
THE JOINT DIMENSIONS ARE 25 AND 49 AGAINST 660 ITEMS -- a ratio of 13:1 at worst.
NO REGULARISED TIER IS BOUGHT, the fit is the committed deterministic
UNREGULARISED logistic regression at the committed fit-seed convention, and doc 61
SS8.3's regularisation qualifier DOES NOT APPLY to this doc.

doc 62 SS10.4, THE RESTRICTION QUALIFIER, AND HERE IT CUTS BOTH WAYS IN THE SAME
SENTENCE -- both halves are stated wherever a tier (ii) number is quoted.

doc 62 SS10.5, REGISTERED PREDICTION: THE COMPARATORS MATCH OR EXCEED EVERY ARM
(doc 47 SS14; doc 52 SS1.1; measured on every arm of exp67, exp68, exp69 and
exp70). Recorded so a tie is not reported as a surprise.

NO EXP75 NUMBER MAY APPEAR IN A SENTENCE CONTAINING "ADVANTAGE" (doc 62 SS0.3).

    python exp75_comparator.py            # writes exp75_ckpt/comparator.json
"""
import argparse
import json
import math
import os
import platform
import time

import numpy as np
import torch

torch.set_num_threads(1)

import exp69_run as R69
import exp75_run as R
import instrument_sb_bounds as INS

DESIGN_DOC = ("qnlp_private_docs/62_exp75_reuploading.md (REGISTERED 2026-08-28) "
              "SS10")
OUT = os.path.join(R.CKPT_DIR, "comparator.json")
FIT_SEED = 75001                        # per-seed init = FIT_SEED + seed
ADAM_STEPS, ADAM_LR = 3000, 0.05        # exp67/68/69's frame, verbatim
LBFGS_ROUNDS, LBFGS_ITER = 10, 200
TIE_BAND = 0.02                         # doc 55 SS6.7
TIER_III_COMMITTED = 0.9200             # imported, asserted, NEVER re-fitted


def u_L(t, L):
    """u_L(t) = (1, cos t, sin t, ..., cos Lt, sin Lt) -- 2L+1 real terms, doc 62
    SS2.1's DERIVED span. At L = 1 this is the committed 3-term u(t)."""
    out = [1.0]
    for k in range(1, L + 1):
        out += [math.cos(k * t), math.sin(k * t)]
    return np.array(out)


def features_for(tier, L, angles):
    """Per-referent feature vector, by tier."""
    if tier == "raw":
        return {n: np.array([float(angles[n])]) for n in INS.HUMANS}
    return {n: u_L(float(angles[n]), L) for n in INS.HUMANS}


def joint(cs, uref, tier):
    """doc 55 SS8.1's frame: the QUERIED event's (agent, patient), AGENT-FIRST.

    Twins share the cast, the wire order, the question spec and the unqueried
    event, and differ ONLY in the queried event's agent/patient direction, so the
    ordered pair the decision is a function of is (agent(q), patient(q))."""
    noun = {w: n for w, n, _a in cs.intro}
    _v, wa, wp = cs.events[cs.qidx - 1]
    a, p = uref[noun[wa]], uref[noun[wp]]
    if tier == "raw":
        return np.concatenate([a, p])          # the RAW angle pair, 2-dim
    return np.outer(a, p).ravel()              # the joint harmonic features


def wire_frame(cs, uref, tier):
    """The null-instrument frame, kept only to be machine-refuted."""
    noun = {w: n for w, n, _a in cs.intro}
    a, p = uref[noun[0]], uref[noun[1]]
    if tier == "raw":
        return np.concatenate([a, p])
    return np.outer(a, p).ravel()


def build_xy(cs_list, uref, tier):
    X = torch.tensor(np.asarray([joint(c, uref, tier) for c in cs_list]),
                     dtype=torch.float64)
    y = torch.tensor([c.answer for c in cs_list], dtype=torch.long)
    return X, y


def assert_frame(cs_list, uref, tier, dim, key, rep):
    """doc 55 SS8.1, AND AT T = 2 THE CHECK RUNS PER q. The WIRE frame must NOT
    separate twins (it is the registered null instrument) and the AGENT frame
    must separate every twin pair."""
    byq = {}
    for cs in cs_list:
        byq.setdefault(cs.qidx, []).append(cs)
    per_q = {}
    for q, sub in sorted(byq.items()):
        pairs = {}
        for cs in sub:
            pairs.setdefault(cs.meta["pair_id"], []).append(cs)
        twins = [v for v in pairs.values() if len(v) == 2]
        n_wire_same = sum(1 for x, y in twins
                          if np.allclose(wire_frame(x, uref, tier),
                                         wire_frame(y, uref, tier)))
        n_agent_same = sum(1 for x, y in twins
                           if np.allclose(joint(x, uref, tier),
                                          joint(y, uref, tier)))
        assert n_wire_same == len(twins), \
            "q=%s: the wire frame separates twins -- the null instrument is not " \
            "null" % q
        assert n_agent_same == 0, \
            "q=%s: the agent frame FAILS to separate %d twin pairs" % (q,
                                                                       n_agent_same)
        per_q[str(q)] = {"twin_pairs_checked": len(twins),
                         "wire_frame_identical_twin_pairs": n_wire_same,
                         "agent_frame_identical_twin_pairs": n_agent_same}
    d = len(joint(cs_list[0], uref, tier))
    assert d == dim, "feature dim %d != registered %d" % (d, dim)
    rep[key] = {"dim": d, "per_q": per_q,
                "wire_frame_is_a_null_instrument": True,
                "frame": "agent-first joint feature of the QUERIED event"}


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


def fit(tier, L, T, seed, store, fm, shuffled):
    """ONE deterministic fit -- `exp69_comparator.fit`'s frame VERBATIM: float64,
    full-batch Adam then LBFGS/strong-Wolfe, checkpoint selected on HARD val
    accuracy, hard argmax accuracy reported on held-out. Only the FEATURE VECTOR
    is widened (doc 62 SS10.2: no regularised tier is bought)."""
    arm = "RU-L%d-T%d-uniform" % (max(L, 2), T)      # T selects the items only
    ang, _t1 = R.arm_angles(arm, fm)
    uref = features_for(tier, L, ang)
    items, meta, path = R.arm_items(arm, store)
    use = R69.relabel_shuffle_q(items, seed) if shuffled else items
    data = {s: R69.compile_all([i for i in use if i["split"] == s], ang)
            for s in ("train", "val", "test")}
    Xtr, ytr = build_xy(data["train"], uref, tier)
    Xva, yva = build_xy(data["val"], uref, tier)
    Xte, yte = build_xy(data["test"], uref, tier)

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
        Ls = m.loss(Xtr, ytr)
        Ls.backward()
        opt.step()
        if step % 10 == 0:
            record("adam%d" % step)
    lb = torch.optim.LBFGS(m.parameters(), max_iter=LBFGS_ITER, history_size=50,
                           tolerance_grad=1e-12, tolerance_change=1e-14,
                           line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        Ls = m.loss(Xtr, ytr)
        Ls.backward()
        return Ls

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
    desc = {}
    for keyname, keyfn in (("by_q", lambda r: "q%d" % r["query_index"]),
                           ("by_order_pair", lambda r: "/".join(r["order_flags"]))):
        cell = {}
        for r in rows:
            cell.setdefault(keyfn(r), [0, 0])
            cell[keyfn(r)][0] += int(r["correct"])
            cell[keyfn(r)][1] += 1
        desc[keyname] = {k: {"acc": v[0] / v[1], "n": v[1]}
                         for k, v in sorted(cell.items())}

    return {"tier": tier, "L": L, "T": T, "is_floor": bool(shuffled),
            "seed": seed, "feature_dim": int(Xtr.shape[1]), "fit_seed": init,
            "dataset": path, "dataset_sha256": meta["dataset_sha256"],
            "train_acc": m.hard_acc(Xtr, ytr), "val_acc": best["v"],
            "heldout_test_acc": m.hard_acc(Xte, yte),
            "train_loss_init": loss0, "train_loss_final": loss_final,
            "grad_norm_final": gnorm, "selected_at": best["where"],
            "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
            "conditioning_descriptors": desc,
            "wall_sec": time.time() - t0}


def agg(rows):
    a = [r["heldout_test_acc"] for r in rows]
    mean = float(np.mean(a))
    return {"mean": mean, "sd": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "per_seed": a, "seeds": [r["seed"] for r in rows],
            "feature_dim": rows[0]["feature_dim"], "T": rows[0]["T"],
            "L": rows[0]["L"], "tier": rows[0]["tier"],
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
    ap.add_argument("--floors", action="store_true",
                    help="also fit the S-shuffle floors (REPORTED, not registered)")
    a = ap.parse_args()
    t0 = time.time()
    store = R.load_part()
    fm = R.feature_maps()
    os.makedirs(R.CKPT_DIR, exist_ok=True)

    rep = {"component": "exp75 classical comparators -- doc 62 SS10",
           "design_doc": DESIGN_DOC, "spec": R.SPEC,
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "numpy": np.__version__,
           "dataset": store[2], "dataset_sha256": store[1]["dataset_sha256"],
           "fit": {"seed_base": FIT_SEED, "adam_steps": ADAM_STEPS,
                   "adam_lr": ADAM_LR, "lbfgs_rounds": LBFGS_ROUNDS,
                   "lbfgs_iter": LBFGS_ITER, "dtype": "float64",
                   "regularisation": "NONE -- doc 62 SS10.2",
                   "selection": "hard val accuracy",
                   "frame": "exp69_comparator.fit, verbatim; only the FEATURE "
                            "VECTOR is widened"}}

    # ---- tier (iii): IMPORTED AND ASSERTED, NEVER RE-FITTED -----------------
    src = json.load(open(R.EXP69_RESULTS, encoding="utf-8"))
    t3 = {}
    for arm in ("R1-A", "R2-A"):
        got = float(src["comparator"]["arms"][arm]["mean"])
        assert abs(got - TIER_III_COMMITTED) < 5e-5, \
            ("doc 62 SS10.1's tier (iii) value %.4f does not match %s :: %s = %.6f"
             % (TIER_III_COMMITTED, R.EXP69_RESULTS, arm, got))
        t3["T%d" % R.COMMITTED_DOC62_SS1_1[arm]["T"]] = {
            "mean": got, "sd": float(src["comparator"]["arms"][arm]["sd"]),
            "per_seed": list(src["comparator"]["arms"][arm]["per_seed"]),
            "feature_dim": 9, "source": R.EXP69_RESULTS, "source_arm": arm,
            "refitted": False}

    # ---- the frame assertions, per tier, per T ------------------------------
    frames = {}
    for T in (1, 2):
        arm = "RU-L2-T%d-uniform" % T
        ang, _t = R.arm_angles(arm, fm)
        items, _m, _p = R.arm_items(arm, store)
        data = R69.compile_all([i for i in items if i["split"] == "test"], ang)
        assert_frame(data, features_for("raw", 1, ang), "raw", 2,
                     "tier_i_raw_T%d" % T, frames)
        for L in R.UPLOADS:
            assert_frame(data, features_for("span", L, ang), "span",
                         (2 * L + 1) ** 2, "tier_ii_span_L%d_T%d" % (L, T), frames)
    rep["frame_assertions"] = frames

    # ---- tier (i) and tier (ii) --------------------------------------------
    tiers = {"information_matched": {}, "span_matched": {}, "committed_9dim": t3}
    floors = {"information_matched": {}, "span_matched": {}}
    for T in (1, 2):
        rows = [fit("raw", 1, T, s, store, fm, False) for s in range(a.seeds)]
        tiers["information_matched"]["T%d" % T] = agg(rows)
        print("[comp] tier (i)  T=%d  dim=%d  mean=%.4f"
              % (T, rows[0]["feature_dim"],
                 tiers["information_matched"]["T%d" % T]["mean"]), flush=True)
        if a.floors:
            fr = [fit("raw", 1, T, s, store, fm, True) for s in range(a.seeds)]
            floors["information_matched"]["T%d" % T] = agg(fr)
    for L in R.UPLOADS:
        tiers["span_matched"]["L%d" % L] = {}
        floors["span_matched"]["L%d" % L] = {}
        for T in (1, 2):
            rows = [fit("span", L, T, s, store, fm, False) for s in range(a.seeds)]
            tiers["span_matched"]["L%d" % L]["T%d" % T] = agg(rows)
            print("[comp] tier (ii) L=%d T=%d  dim=%d  mean=%.4f  (tier iii "
                  "%.4f, gap %+.4f)"
                  % (L, T, rows[0]["feature_dim"],
                     tiers["span_matched"]["L%d" % L]["T%d" % T]["mean"],
                     TIER_III_COMMITTED,
                     tiers["span_matched"]["L%d" % L]["T%d" % T]["mean"]
                     - TIER_III_COMMITTED), flush=True)
            if a.floors:
                fr = [fit("span", L, T, s, store, fm, True) for s in range(a.seeds)]
                floors["span_matched"]["L%d" % L]["T%d" % T] = agg(fr)

    rep["tiers"] = tiers
    if a.floors:
        rep["floors_REPORTED_NOT_REGISTERED"] = floors
    rep["information_gap_tier_ii_minus_tier_iii"] = {
        "L%d" % L: {"T%d" % T: (tiers["span_matched"]["L%d" % L]["T%d" % T]["mean"]
                                - TIER_III_COMMITTED)
                    for T in (1, 2)} for L in R.UPLOADS}
    rep["arbiter_doc62_SS10_3"] = (
        "TIER (ii) AT DEGREE L IS THE ARBITER. The registered rule, in full: "
        "(1) the best re-upload cell lifts >= 0.06 over its committed baseline "
        "AND tier (ii) exceeds tier (iii)'s 0.9200 by >= 0.06 -> THE ADDED "
        "FREQUENCIES CARRY INFORMATION AND THE CIRCUIT CONVERTED IT, and this is "
        "the ONLY configuration in which a lift may be reported as a re-uploading "
        "win. (2) the circuit lifts >= 0.06 but tier (ii) - 0.9200 < 0.06 -> THE "
        "CIRCUIT'S GAP EXCEEDS THE INFORMATION GAP, so the lift is a "
        "BASELINE-TRAINABILITY ARTEFACT (the committed single-encode arm "
        "under-trained its own class; doc 62 SS1.4 measured 0.04 of unconverted "
        "supply) and `reuploading_lifts` MAY NOT BE CLAIMED AS A RE-UPLOADING "
        "RESULT. (3) the circuit does not lift but tier (ii) exceeds tier (iii) "
        "by >= 0.06 -> SUPPLY IS REAL AND EXTRACTABLE CLASSICALLY AND THE CIRCUIT "
        "DID NOT CONVERT IT -- doc 55 SS8.2's middle row, the programme's fifth "
        "instance. (4) the circuit does not lift and tier (ii) ties tier (iii) -> "
        "THE EXTRA FREQUENCIES CARRY NO USABLE INFORMATION FOR THIS TASK AT THIS "
        "SAMPLE SIZE, and no circuit claim is available either way. doc 62 "
        "SS12.4.1 makes row (2) BINDING on any TRUE `lifts`.")
    rep["restriction_qualifier_doc62_SS10_4"] = (
        "BOTH HALVES ARE STATED WHEREVER A TIER (ii) NUMBER IS QUOTED, AND HERE "
        "THEY POINT IN OPPOSITE DIRECTIONS. (a) SUPERSET IN COEFFICIENT SPACE: "
        "tier (ii) ranges over ALL coefficient vectors in the degree-L span while "
        "unitarity restricts the circuit to a NON-CONVEX PROPER SUBSET, so the "
        "comparator is an UPPER reference and a circuit below it is NOT evidence "
        "the class is insufficient. (b) RESTRICTION ACROSS CONDITIONING GROUPS: "
        "the circuit carries one form per conditioning group (12 at T = 1, 72 at "
        "T = 2) while the comparator carries ONE FORM TOTAL, so the comparator is "
        "simultaneously a WEAKER model in that dimension. Neither half is new to "
        "this programme; WHAT IS NEW IS THAT THEY POINT IN OPPOSITE DIRECTIONS "
        "AND THE HONEST SUMMARY IS THEREFORE 'TIE IS THE STRONGEST AVAILABLE "
        "STATEMENT', NOT 'THE CIRCUIT BEAT THE CLASS'.")
    rep["not_vacuous_doc62_SS10_2"] = (
        "the joint dimensions are 25 (L = 2) and 49 (L = 3) against 660 train+val "
        "items -- a ratio of 13:1 at worst. NO REGULARISED TIER IS BOUGHT and doc "
        "61 SS8.3's regularisation qualifier DOES NOT APPLY to this rung, which "
        "is a real contrast with doc 61 SS8.2 where 1296 and 18496 joint "
        "dimensions against 660 items separate every item by dimension count "
        "alone.")
    rep["registered_prediction_doc62_SS10_5"] = (
        "THE COMPARATORS MATCH OR EXCEED EVERY ARM (doc 47 SS14; doc 52 SS1.1; "
        "measured on every arm of exp67, exp68, exp69 and exp70). Recorded so a "
        "TIE IS NOT REPORTED AS A SURPRISE.")
    rep["advantage_prohibition"] = (
        "NO EXP75 NUMBER MAY APPEAR IN A SENTENCE CONTAINING 'ADVANTAGE' (doc 52 "
        "SS1; doc 56 SS0.1; doc 58 SS12; doc 61 SS0.3; doc 62 SS0.3 / SS16).")
    rep["tie_band_doc55_SS6_7"] = TIE_BAND
    rep["wall_sec"] = time.time() - t0
    json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[comp] wrote %s (%.0fs)" % (a.out, rep["wall_sec"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
