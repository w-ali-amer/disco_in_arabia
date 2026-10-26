# -*- coding: utf-8 -*-
"""exp68_comparator.py -- doc 54 SS8: the classical comparator, and THE ARBITER.

doc 54 SS8 registers this BEFORE any exp68 number exists, for two reasons that
must not be conflated:

  1. A GUARD. "Registered prediction: the classical comparator MATCHES OR
     EXCEEDS every arm." doc 47 SS14 (a linear model on the identical features
     hits the identical bound by construction) and doc 52 SS1.1 (the
     product-state kernel factorises and is classically O(n), verified to
     3.3e-16). Registered so it cannot become a post-hoc surprise, and so no
     exp68 number can be read as a quantum result.
  2. THE ARBITER (SS8.1), which is the load-bearing one. If F2 sits at its floor,
     the comparator on the IDENTICAL features decides between two very different
     negatives:

       F2 at floor, comparator clears  -> CIRCUIT-TRAINABILITY failure. The
           information is there and is linearly extractable; the circuit did not
           learn it. Observed precedent: doc 53 SS7.1.3 on the arity lever.
       F2 at floor, comparator at floor too -> INFORMATION / SAMPLE-side. The
           second feature does not carry the category at this sample size, and
           no circuit claim is available either way.

     SS8.2 extends the same rule to the `width_adds_information` gap: if the
     circuit shows F2 - FA >= 0.06 but LR(F2) - LR(FA) < 0.06, the circuit's gap
     EXCEEDS the information gap and is an FA-trainability artefact, and
     `width_adds_information` may NOT be claimed.

FORM (doc 54 SS8). Logistic regression per arm on the IDENTICAL per-referent
features, joint by outer product and ordered AGENT-FIRST: 9-dim for F1 and FA
(u = (1, cos t, sin t) per referent), 81-dim for F2 (u(theta0) (x) u(theta1) per
referent). Same dataset, same splits, same 5 seeds, deterministic fit at a
registered fit seed. The optimiser frame is `exp67_comparator.py`'s verbatim --
float64, full-batch Adam then LBFGS/strong-Wolfe, checkpoint selected on hard
val accuracy, hard argmax accuracy reported on held-out.

WHAT "SAME 5 SEEDS" MEANS HERE, stated so it cannot be read as five independent
draws. In exp67 the per-item angles were seed-indexed, so its five fits sat on
five geometries. In exp68 the features are DETERMINISTIC CONSTANTS (doc 54
SS6.2: "seeds vary optimiser initialisation only"), so the five comparator fits
differ ONLY in initialisation -- exactly as the five circuit runs of each arm do.
The spread reported is optimiser-init spread, not data spread, for the circuit
and for the comparator alike, and that is what makes them comparable.

THE COMPARATOR GETS FLOORS TOO. SS8.1's rows are of the form "comparator clears
floor", which needs a floor for the comparator itself, on the same footing as
the circuit's. Each arm's comparator is therefore also fitted on the
S-shuffle-relabelled dataset at the SAME relabel seeds the circuit floors use.
These support the registered arbiter; they add no new criterion.

SS8.3's qualifier travels with every number here: "identical features" means the
comparator is a RESTRICTION of the circuit's conditional function class -- the
circuit carries one 9-dim (or 81-dim) form per verb and event direction, the
comparator carries one. A comparator loss on a cell is not automatically a
circuit win.

Writes `exp68_ckpt/comparator.json`. It writes NO results file: doc 53 SS7.3's
standing rule gives the run ONE aggregator, and `exp68_aggregate.py` is it.

    python exp68_comparator.py
"""
import argparse, json, os, platform, sys, time

import numpy as np
import torch

torch.set_num_threads(1)

import exp42_compiler as comp
import exp68_run as E
from exp60_run import relabel_shuffle

DESIGN_DOC = "qnlp_private_docs/54_exp68_width_lever.md (1025d21, FROZEN) SS8"
OUT = os.path.join(E.CKPT_DIR, "comparator.json")
FIT_SEED = 68001                       # registered; per-seed init = FIT_SEED + seed
ADAM_STEPS, ADAM_LR = 3000, 0.05
LBFGS_ROUNDS, LBFGS_ITER = 10, 200


# ----------------------------------------------------------------- features
def joint_features(cs, uref):
    """doc 54 SS8: outer(u(agent), u(patient)), AGENT-FIRST.

    The frame is doc 47 SS3.2/SS17's twin statement -- twin YES needs
    G(x_i, y_j) > 0 and twin NO needs G(y_j, x_i) < 0, the SAME form with the
    arguments swapped. Twins share cast, wire order and question surface and
    differ ONLY in the event's agent/patient direction, so the ordered pair the
    decision is a function of is (agent, patient)."""
    noun = {w: n for w, n, _a in cs.intro}
    _v, wa, wp = cs.events[0]
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


def assert_frame(cs_list, uref, dim, rep_key, rep):
    """doc 53 SS5's both-directions discipline, transposed to exp68."""
    pairs = {}
    for cs in cs_list:
        pairs.setdefault(cs.pair_id, []).append(cs)
    twins = [v for v in pairs.values() if len(v) == 2]
    n_wire_same = sum(1 for a, b in twins
                      if np.allclose(wire_features(a, uref), wire_features(b, uref)))
    n_agent_same = sum(1 for a, b in twins
                       if np.allclose(joint_features(a, uref), joint_features(b, uref)))
    assert n_wire_same == len(twins), \
        "wire frame unexpectedly separates twins -- the null instrument is not null"
    assert n_agent_same == 0, \
        "agent frame FAILS to separate %d twin pairs" % n_agent_same
    d = len(joint_features(cs_list[0], uref))
    assert d == dim, "feature dim %d != registered %d" % (d, dim)
    rep[rep_key] = {"dim": d, "twin_pairs_checked": len(twins),
                    "wire_frame_identical_twin_pairs": n_wire_same,
                    "agent_frame_identical_twin_pairs": n_agent_same,
                    "wire_frame_is_a_null_instrument": True}


# -------------------------------------------------------------------- model
class LogReg(torch.nn.Module):
    """Binary logistic regression, float64. Two logits so the head matches the
    circuit's two-effect readout shape; softmax over two logits is the same
    model as one logit and keeps the loss identical to the circuit's."""

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


def fit(arm, seed, items, fm, shuffled):
    """One deterministic fit. `shuffled` selects the matched S-shuffle floor."""
    base = E.base_arm(arm)
    uref = E.referent_features(base, fm)
    use = relabel_shuffle(items, None, seed) if shuffled else items
    ang, _t1 = E.arm_angles(base, fm)
    data = {s: comp.compile_all([i for i in use if i["split"] == s], ang)
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
    for step in range(1, ADAM_STEPS + 1):                    # FULL batch
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

    for r in range(LBFGS_ROUNDS):
        lb.step(closure)
        record("lbfgs%d" % r)
    with torch.no_grad():
        loss_final = float(m.loss(Xtr, ytr))
    m.zero_grad()
    m.loss(Xtr, ytr).backward()
    gnorm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters())))

    m.load_state_dict(best["state"])
    return {"arm": arm, "base_arm": base, "is_floor": bool(shuffled), "seed": seed,
            "feature_dim": int(Xtr.shape[1]), "fit_seed": init,
            "train_acc": m.hard_acc(Xtr, ytr), "val_acc": best["v"],
            "heldout_test_acc": m.hard_acc(Xte, yte),
            "train_loss_init": loss0, "train_loss_final": loss_final,
            "grad_norm_final": gnorm, "selected_at": best["where"],
            "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
            "wall_sec": time.time() - t0}


def agg(rows):
    a = [r["heldout_test_acc"] for r in rows]
    mean = sum(a) / len(a)
    sd = (sum((x - mean) ** 2 for x in a) / (len(a) - 1)) ** 0.5 if len(a) > 1 else 0.0
    return {"mean": mean, "sd": sd, "per_seed": a,
            "seeds": [r["seed"] for r in rows],
            "feature_dim": rows[0]["feature_dim"],
            "val_per_seed": [r["val_acc"] for r in rows],
            "train_per_seed": [r["train_acc"] for r in rows],
            "train_loss_init_per_seed": [r["train_loss_init"] for r in rows],
            "train_loss_final_per_seed": [r["train_loss_final"] for r in rows],
            "grad_norm_final_per_seed": [r["grad_norm_final"] for r in rows],
            "runs": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=E.DATA)
    ap.add_argument("--seeds", type=int, default=E.N_SEEDS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    t0 = time.time()
    blob = json.load(open(a.data, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("exp68 dataset provenance FAILED")
    fm = E.feature_maps()

    frame = {}
    for base, dim in (("F1", 9), ("FA", 9), ("F2", 81)):
        uref = E.referent_features(base, fm)
        ang, _t1 = E.arm_angles(base, fm)
        probe = comp.compile_all([i for i in items if i["split"] == "test"], ang)
        assert_frame(probe, uref, dim, base, frame)
        print("[frame] %-2s d=%d  wire-frame identical twins %d/%d (null "
              "instrument), agent-frame identical %d"
              % (base, dim, frame[base]["wire_frame_identical_twin_pairs"],
                 frame[base]["twin_pairs_checked"],
                 frame[base]["agent_frame_identical_twin_pairs"]), flush=True)

    arms, floors = {}, {}
    for base in E.BASE_ARMS:
        arms[base] = agg([fit(base, s, items, fm, False) for s in range(a.seeds)])
        floors[base] = agg([fit("S-shuffle-%s" % base, s, items, fm, True)
                            for s in range(a.seeds)])
        print("[exp68-comp] %-2s d=%-2d held-out %.4f (sd %.4f)  floor %.4f "
              "(sd %.4f)  train-loss %.4f -> %.4f  |g|=%.1e"
              % (base, arms[base]["feature_dim"], arms[base]["mean"],
                 arms[base]["sd"], floors[base]["mean"], floors[base]["sd"],
                 float(np.mean(arms[base]["train_loss_init_per_seed"])),
                 float(np.mean(arms[base]["train_loss_final_per_seed"])),
                 max(arms[base]["grad_norm_final_per_seed"])), flush=True)

    out = {
        "component": "exp68_comparator",
        "spec": DESIGN_DOC,
        "role": ("doc 54 SS8: registered GUARD (no exp68 number may be read as a "
                 "quantum result) and registered ARBITER (SS8.1/SS8.2) between a "
                 "circuit-trainability failure and an information-side failure."),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "dataset": a.data, "dataset_sha256": meta["dataset_sha256"],
        "features": {
            "definition": ("per referent: u = (1, cos t, sin t) for F1/FA (d = 3) "
                           "and u(theta0) (x) u(theta1) for F2 (d = 9); joint by "
                           "OUTER PRODUCT ordered AGENT-FIRST -> 9 dims for F1/FA, "
                           "81 for F2"),
            "identical_to_circuit": (
                "these are the SAME per-referent vectors the arms' circuits are "
                "linear in -- RY(t)|0> has density matrix linear in "
                "(1, cos t, sin t), so a two-qubit product register is linear in "
                "the 9-term outer product. exp68_run.py --smoke measures the "
                "closure directly (F2 residual ~7e-08 in the 9-dim basis)."),
            "frame_check": frame},
        "optimiser": {"fit_seed_base": FIT_SEED,
                      "per_seed_init": "FIT_SEED + seed",
                      "restarts": 0, "adam_steps": ADAM_STEPS, "adam_lr": ADAM_LR,
                      "lbfgs_rounds": LBFGS_ROUNDS, "lbfgs_max_iter": LBFGS_ITER,
                      "line_search": "strong_wolfe", "batching": "FULL batch",
                      "dtype": "float64",
                      "model_selection": "best HARD argmax accuracy on the val split",
                      "reported_accuracy": "HARD argmax on held-out test",
                      "seeds_vary": ("optimiser INITIALISATION ONLY -- doc 54 "
                                     "SS6.2's rule for the circuit arms, applied "
                                     "identically here. The features are frozen "
                                     "constants, so the five fits sit on ONE "
                                     "geometry and the spread is init spread, not "
                                     "data spread. Unlike exp67, whose five fits "
                                     "sat on five seed-indexed angle draws.")},
        "arms": arms, "floors": floors,
        "sd_convention": "sample sd, ddof=1",
        "n_seeds": a.seeds,
        "function_class_qualifier": (
            "doc 54 SS8.3: 'identical features' makes the comparator a RESTRICTION "
            "of the circuit's conditional function class -- the circuit carries one "
            "9-dim (or 81-dim) form PER VERB AND EVENT DIRECTION, the comparator "
            "carries one. A comparator loss on a cell is therefore not "
            "automatically a circuit win, and this qualifier travels with every "
            "number in this file."),
        "wall_time_sec": time.time() - t0,
    }
    os.makedirs(E.CKPT_DIR, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[exp68-comp] wrote %s (%.0fs)" % (a.out, out["wall_time_sec"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
