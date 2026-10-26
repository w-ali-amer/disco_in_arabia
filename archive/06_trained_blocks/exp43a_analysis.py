# -*- coding: utf-8 -*-
"""exp43a_analysis.py -- numerical companions to the exp43a analytic section.

Doc 23 exp43a step (i): analytic attempt -- hand-construct a solving
assignment or identify a structural obstruction. This script computes the
witnesses the write-up relies on. It carries NO pre-registered verdicts;
the frozen verdict booleans come exclusively from exp43a_run.py.

W1  Exchange-symmetric invariance (Lemma 1 witness): under A1-class blocks
    ([U,SWAP]=0) the two members of every swap twin compile to the SAME
    statevector, so ANY readout is at exactly 50% on twins. Verified to
    float precision on the whole family.
W2  Generic distinguishability (Prop 2 witness): under random KAK (full
    SU(4)) assignments, the frame-rotated reduced density matrices on the
    two QUERIED wires differ between twins for every family pair. Reports
    the per-pair trace-distance distribution -- if min > 0 the candidate
    obstruction ("the conjunction leaves no joint signature on the queried
    wires") is FALSE as an information-theoretic claim; the magnitudes
    quantify how much signal survives partial tracing.
W3  Best shared linear readout at fixed blocks (Prop 3 diagnostic): with
    blocks frozen at a random assignment, the exp42 decision rule is
    sign Tr[rho_tilde D] with D = |phi+><phi+| - |phi-><phi-| shared
    across the family. Relaxing D to an ARBITRARY traceless Hermitian
    (upper bound on the rank-<=2 implementable set) turns the best
    achievable train accuracy into a convex-ish fit: logistic regression
    without intercept on the 15 Pauli expectation features of rho_tilde.
W4  Family structure statistics feeding the hand-construction analysis:
    core-event order (e1 before e2), filler traffic on the queried wires.
W5  Affine-collapse audit (Lemma 2 witness): a twin pair defeats EVERY
    effect pair phi+/phi- iff rho_no = alpha*rho_yes + c*I with alpha >= 0
    (then <phi|rho_no|phi> is a monotone function of <phi|rho_yes|phi>, so
    every fixed decision functional gets exactly one twin right). W5
    reports the per-pair least-squares residual of that affine model at
    random KAK blocks -- residuals bounded away from 0 mean per-item
    two-sided separability holds generically and only FAMILY-WIDE sharing
    can obstruct representability.

Optional --ckpt exp43a_best.pt: re-runs W2/W3 at the certificate run's
best trained parameters (post-hoc diagnostic, clearly labeled).

Output: results_exp43a_analysis*.json (never-overwrite suffixing).
"""

import argparse
import datetime
import json
import os

import numpy as np
import torch

import exp42_compiler as comp
import exp42_models
import exp42_sim_adapter as sim
from exp43a_model import KakStoryModel

DATA = "stories_exp43a_family.json"
OUT_BASE = "results_exp43a_analysis"


# ------------------------------------------------------------------ plumbing
def load_family(path=DATA):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items, meta = data["items"], data["meta"]
    sha = comp.canonical_sha256(items)
    assert sha == meta["dataset_sha256"], "family dataset hash mismatch"
    angles, _m, _s = comp.load_embeddings()
    fam = [it for it in items if it["split"] == "train"]
    compiled = [comp.compile_story(it, angles) for it in fam]
    verbs = sorted(meta["verb_inventory"].keys())
    return compiled, verbs, meta, sha


def story_state(model, cs):
    """Statevector after story + inverse question preparation -- mirrors
    exp42_models.QuantumStoryModel.story_values up to (not including) the
    trained effects; parity with the committed code is asserted in
    check_parity()."""
    cache = {}

    def block(v):
        if v not in cache:
            cache[v] = model.verb_matrix(v)
        return cache[v]

    psi = sim.init_state(cs.K)
    for w, _n, ang in cs.intro:
        psi = sim.apply_gate(psi, sim.ry(ang), (w,))
    for v, wa, wp in cs.events:
        psi = sim.apply_gate(psi, block(v), (wa, wp))
    vseq, qa, qp = cs.question
    for qv in reversed(vseq):
        psi = sim.apply_gate(psi, block(qv).conj().T, (qa, qp))
    return psi, (qa, qp)


def rho_queried(psi, wires):
    """4x4 reduced density matrix on the two queried wires (frame already
    applied at state level by story_state's inverse preparation)."""
    K = psi.dim()
    qa, qp = wires
    rest = [w for w in range(K) if w not in (qa, qp)]
    A = psi.permute([qa, qp] + rest).reshape(4, -1)
    return A @ A.conj().T


def check_parity(model, compiled, n=5):
    """The rho-based probabilities must equal the committed exp42 readout."""
    for cs in compiled[:n]:
        with torch.no_grad():
            psi, qw = story_state(model, cs)
            rho = rho_queried(psi, qw)
            ps = []
            for eff in (model.q_pos, model.q_neg):
                phi = eff.matrix()[:, 0]
                ps.append((phi.conj() @ rho @ phi).real.item())
            ref = model.story_values(cs).detach()
        assert np.allclose(ps, ref.numpy(), atol=1e-5), \
            "rho-readout parity broken: %s vs %s" % (ps, ref)


def pairs_of(compiled):
    by = {}
    for cs in compiled:
        by.setdefault(cs.pair_id, {})[cs.answer] = cs
    out = []
    for pid, d in sorted(by.items()):
        assert set(d) == {0, 1}, "incomplete twin pair %s" % pid
        out.append((pid, d[0], d[1]))                     # (pid, yes, no)
    return out


def trace_distance(r1, r2):
    ev = torch.linalg.eigvalsh(r1 - r2)
    return 0.5 * ev.abs().sum().item()


# ----------------------------------------------------------------------- W1
def run_w1(compiled, verbs):
    model = exp42_models.QuantumStoryModel("A1", verbs, seed=0)
    worst = 0.0
    with torch.no_grad():
        for _pid, cy, cn in pairs_of(compiled):
            py, _ = story_state(model, cy)
            pn, _ = story_state(model, cn)
            worst = max(worst, (py - pn).abs().max().item())
    return {"claim": "A1-class ([U,SWAP]=0) twin statevectors identical -> "
                     "any readout exactly 50% on twins (Lemma 1)",
            "max_abs_state_diff_over_family": worst,
            "confirmed": worst < 1e-5}


# ----------------------------------------------------------------------- W2
def run_w2(compiled, verbs, seeds, model=None, label="random_kak"):
    per_seed = {}
    for s in seeds:
        m = model if model is not None else KakStoryModel(verbs, 9000 + s)
        dists = []
        with torch.no_grad():
            for _pid, cy, cn in pairs_of(compiled):
                ry_, qw = story_state(m, cy)
                rn_, qw2 = story_state(m, cn)
                assert qw == qw2
                dists.append(trace_distance(rho_queried(ry_, qw),
                                            rho_queried(rn_, qw2)))
        d = np.array(dists)
        per_seed[str(s)] = {
            "min": float(d.min()), "p25": float(np.percentile(d, 25)),
            "median": float(np.median(d)),
            "p75": float(np.percentile(d, 75)), "max": float(d.max()),
            "n_pairs": len(d),
            "n_pairs_below_1e-3": int((d < 1e-3).sum()),
        }
        if model is not None:
            break
    return {"label": label, "metric": "trace distance of frame-rotated "
            "reduced states on the two queried wires, per twin pair",
            "per_seed": per_seed}


# ----------------------------------------------------------------------- W3
_PAULI_1 = {
    "I": torch.eye(2, dtype=sim.CDTYPE),
    "X": torch.tensor([[0, 1], [1, 0]], dtype=sim.CDTYPE),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=sim.CDTYPE),
    "Z": torch.tensor([[1, 0], [0, -1]], dtype=sim.CDTYPE),
}
PAULI_2 = [(a + b, torch.kron(_PAULI_1[a], _PAULI_1[b]))
           for a in "IXYZ" for b in "IXYZ" if a + b != "II"]


def run_w3(compiled, verbs, seeds, model=None, label="random_kak"):
    from sklearn.linear_model import LogisticRegression
    per_seed = {}
    for s in seeds:
        m = model if model is not None else KakStoryModel(verbs, 9000 + s)
        X, y = [], []
        with torch.no_grad():
            for cs in compiled:
                psi, qw = story_state(m, cs)
                rho = rho_queried(psi, qw)
                X.append([(rho @ P).diagonal().sum().real.item()
                          for _n, P in PAULI_2])
                y.append(1 if cs.answer == 0 else -1)     # yes -> +1
        X, y = np.array(X), np.array(y)
        clf = LogisticRegression(fit_intercept=False, C=1e6, max_iter=50000)
        clf.fit(X, y)
        acc = float((clf.predict(X) == y).mean())
        per_seed[str(s)] = {"best_linear_train_acc": acc}
        if model is not None:
            break
    return {"label": label,
            "meaning": "best UNRANKED traceless-Hermitian shared decision "
                       "functional at fixed blocks (upper bound on the "
                       "rank-2 effect-pair readout at those blocks); "
                       "logistic regression w/o intercept on 15 Pauli "
                       "features of rho_tilde",
            "per_seed": per_seed}


# ----------------------------------------------------------------------- W4
def run_w4(items_train):
    n = 0
    e1_before = 0
    touch_any = 0
    touch_after_core = 0
    touches = []
    for it in items_train:
        if it["answer"] != "yes":
            continue
        n += 1
        q = it["question"]
        i2 = it["flip_idx"]
        cand = [i for i, e in enumerate(it["events"])
                if e["verb"] == q["v1"] and e["patient"] == q["x"]]
        assert len(cand) == 1, "e1 not unique in %s" % it["id"]
        i1 = cand[0]
        if i1 < i2:
            e1_before += 1
        core = {i1, i2}
        t = [i for i, e in enumerate(it["events"]) if i not in core
             and (e["agent"] in (q["x"], q["y"])
                  or e["patient"] in (q["x"], q["y"]))]
        touches.append(len(t))
        if t:
            touch_any += 1
        if any(i > max(core) for i in t):
            touch_after_core += 1
    return {
        "n_pairs": n,
        "frac_e1_before_e2": e1_before / n,
        "frac_pairs_with_filler_on_queried_wires": touch_any / n,
        "frac_pairs_with_filler_on_queried_wires_after_both_core": (
            touch_after_core / n),
        "mean_filler_touches_on_queried_wires": float(np.mean(touches)),
    }


# ----------------------------------------------------------------------- W5
def run_w5(compiled, verbs, seeds, model=None, label="random_kak"):
    eye = torch.eye(4, dtype=sim.CDTYPE)

    def herm_dot(A, B):
        return (A.conj() * B).sum().real.item()

    per_seed = {}
    for s in seeds:
        m = model if model is not None else KakStoryModel(verbs, 9000 + s)
        resid, alphas, collapsed = [], [], 0
        with torch.no_grad():
            for _pid, cy, cn in pairs_of(compiled):
                ry_, qw = story_state(m, cy)
                rn_, qw2 = story_state(m, cn)
                Ry = rho_queried(ry_, qw)
                Rn = rho_queried(rn_, qw2)
                # least squares Rn ~ alpha*Ry + c*I  (real span, Frobenius)
                g = torch.tensor([[herm_dot(Ry, Ry), herm_dot(Ry, eye)],
                                  [herm_dot(eye, Ry), herm_dot(eye, eye)]],
                                 dtype=torch.float64)
                b = torch.tensor([herm_dot(Ry, Rn), herm_dot(eye, Rn)],
                                 dtype=torch.float64)
                sol = torch.linalg.solve(g, b)
                alpha, c = sol[0].item(), sol[1].item()
                E = Rn - alpha * Ry - c * eye
                denom = (Rn - eye / 4).norm().item()
                r = E.norm().item() / max(denom, 1e-12)
                resid.append(r)
                alphas.append(alpha)
                if r < 1e-3 and alpha >= 0:
                    collapsed += 1
        rr = np.array(resid)
        per_seed[str(s)] = {
            "min_residual": float(rr.min()),
            "median_residual": float(np.median(rr)),
            "max_residual": float(rr.max()),
            "n_pairs_affine_collapsed": collapsed,
            "n_pairs": len(rr),
        }
        if model is not None:
            break
    return {"label": label,
            "meaning": "relative Frobenius residual of the best affine "
                       "model rho_no = alpha*rho_yes + c*I per twin pair; "
                       "collapse (residual~0 with alpha>=0) is the ONLY "
                       "way a single pair can defeat every effect pair "
                       "(Lemma 2)",
            "per_seed": per_seed}


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ckpt", default=None,
                    help="exp43a_best.pt for post-run diagnostics")
    args = ap.parse_args()

    torch.set_num_threads(4)
    compiled, verbs, meta, sha = load_family(args.data)
    with open(args.data, encoding="utf-8") as f:
        items_train = [it for it in json.load(f)["items"]
                       if it["split"] == "train"]
    print("[exp43a-analysis] family: %d items, verbs=%s" %
          (len(compiled), verbs))

    check_parity(KakStoryModel(verbs, 1234), compiled)
    print("[exp43a-analysis] rho-readout parity vs committed exp42 code: OK")

    seeds = list(range(args.seeds))
    res = {
        "experiment": "exp43a_analysis",
        "design_doc": "qnlp_private_docs/23_exp43_ladder_representability_"
                      "design.md exp43a step (i) (commit ef15d76)",
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset_file": args.data, "dataset_sha256": sha,
        "family_n_items": len(compiled), "verbs": verbs,
        "note": "witnesses for the analytic write-up; carries NO "
                "pre-registered verdicts (those live in results_exp43a*.json)",
        "W1_symmetric_invariance": run_w1(compiled, verbs),
        "W2_generic_twin_distinguishability": run_w2(compiled, verbs, seeds),
        "W3_best_shared_linear_readout": run_w3(compiled, verbs, seeds),
        "W4_family_structure": run_w4(items_train),
        "W5_affine_collapse_audit": run_w5(compiled, verbs, seeds),
    }

    if args.ckpt:
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        m = KakStoryModel(verbs, 0)
        m.load_state_dict(blob["state_dict"])
        acc = np.mean([m.predict(cs) == cs.answer for cs in compiled])
        res["trained_ckpt_diagnostics"] = {
            "ckpt": args.ckpt, "ckpt_seed": blob.get("seed"),
            "recomputed_train_acc": float(acc),
            "W2_at_trained": run_w2(compiled, verbs, [0], model=m,
                                    label="trained_best"),
            "W3_at_trained": run_w3(compiled, verbs, [0], model=m,
                                    label="trained_best"),
            "W5_at_trained": run_w5(compiled, verbs, [0], model=m,
                                    label="trained_best"),
        }

    out = OUT_BASE + ".json"
    suffix = ord("b")
    while os.path.exists(out):                            # never overwrite
        out = "%s_%s.json" % (OUT_BASE, chr(suffix))
        suffix += 1
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[exp43a-analysis] results -> %s" % out)
    print(json.dumps({k: res[k] for k in
                      ("W1_symmetric_invariance", "W4_family_structure")},
                     indent=1))
    for k in ("W2_generic_twin_distinguishability",
              "W3_best_shared_linear_readout"):
        print(k, json.dumps(res[k]["per_seed"], indent=1))
    if args.ckpt:
        print("trained_ckpt_diagnostics",
              json.dumps(res["trained_ckpt_diagnostics"], indent=1,
                         default=str)[:2000])


if __name__ == "__main__":
    main()
