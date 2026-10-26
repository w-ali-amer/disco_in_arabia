# -*- coding: utf-8 -*-
"""exp43_wazn_screen.py -- Stage A wazn screen (doc 26 SS3, FROZEN).

Do trained verb blocks track morphological pattern (wazn)? The frozen
exp43a family has three selectionally matched verb pairs, exactly one of
which is wazn-mismatched:

    DRB <-> SA'D   (Form I vs Form III)   <- the wazn pair
    HML <-> RF'    (both Form I)
    QT' <-> FTH    (both Form I)

Primary metric (frozen; exp39's notion): behavioural substitution
distance. For verbs v, w: substitute v's block with w's, re-evaluate
train accuracy on the frozen exp43a family (200 items), take the
accuracy drop; symmetrise d(v,w) = 0.5*[drop(v->w) + drop(w->v)].
Training-free -- forward passes only.

Statistic: per restart, is d(wazn pair) the LARGEST of the three
matched-pair distances? (strictly largest; ties recorded and counted as
non-successes). k over the 40 restarts, null k ~ Binomial(40, 1/3).

Pre-registered verdicts (doc 26 SS3):
    wazn_screen_survives     iff k >= 19
    wazn_screen_ruled_out    iff k <= 13
    wazn_screen_inconclusive iff 14 <= k <= 18
There is deliberately NO wazn_confirmed verdict (doc 26 SS3-4).

Control-of-the-control, run FIRST (frozen): the same statistic on 200
sets of random KAK blocks, where the three pair-distances are
exchangeable by construction. random_block_null_calibrated = true iff
the wazn-mismatched pair is largest in 28-39% of draws. If not, Stage A
is VOID and reported as void, not as a result.

Stratification (fixed in advance): the headline k uses all 40 restarts;
the L-BFGS-polished solutions are reported separately, DESCRIPTIVE ONLY.
ARTIFACT NOTE (flagged): doc 26 SS2 describes the 40 committed state
files as Adam solutions (train 0.645-0.705). Inspection shows
exp43a_state_r01/r08/r30.pt contain the L-BFGS-POLISHED states (0.93/
0.955/0.95, metadata stage="lbfgs") -- exp43a's finalize overwrote a
restart's file when polish improved it, and the Adam-only states for
those three restarts no longer exist. The headline k over "all 40
restarts" is therefore computed over the 40 files as committed (37 Adam
+ 3 polished); k over the 37 pure-Adam files is reported alongside as a
descriptive robustness number (the frozen bands are defined for n=40 and
are not applied to it).

Secondary, DESCRIPTIVE ONLY (never a verdict): Euclidean distance
between canonicalised Cartan vectors (a,b,c) per pair.

Output (never overwritten) -> results_exp43_wazn_screen.json.
This script modifies no existing results file and trains nothing.
"""

import glob
import itertools
import json
import math
import os
import platform
import random
import sys
import time

import torch

torch.set_num_threads(1)

import exp42_sim_adapter as sim                               # noqa: E402
import exp43a_run as run43a                                   # noqa: E402
from exp42_run import fresh_results_path, git_note            # noqa: E402
from exp43a_model import KakStoryModel, cartan_core           # noqa: E402

V_DRB, V_SAD = "ضرب", "ساعد"  # ضرب ساعد
V_HML, V_RFE = "حمل", "رفع"        # حمل رفع
V_QTA, V_FTH = "قطع", "فتح"        # قطع فتح

PAIRS = [("wazn_mismatched", V_DRB, V_SAD),   # Form I vs Form III
         ("matched_I_a", V_HML, V_RFE),       # both Form I
         ("matched_I_b", V_QTA, V_FTH)]       # both Form I

N_RESTARTS = 40
STATE_FMT = "exp43a_state_r%02d.pt"
N_RANDOM = 200                                            # FROZEN
RANDOM_SEED_BASE = 260000
K_SURVIVE, K_RULED_OUT = 19, 13                           # FROZEN
CAL_LO, CAL_HI = 0.28, 0.39                               # FROZEN


# ------------------------------------------------------- fast evaluation
def block_matrices(model):
    with torch.no_grad():
        mats = {v: model.verb_matrix(v).detach() for v in model.verbs}
        effs = [model.q_pos.matrix().conj().T.detach(),
                model.q_neg.matrix().conj().T.detach()]
    return mats, effs


def family_acc(mats, effs, compiled, sub=None):
    """Train accuracy with cached block matrices; `sub` remaps
    verb -> block-owner verb, applied to story events AND the question
    (exp42's verb_map semantics)."""
    sub = sub or {}
    with torch.no_grad():
        n_ok = 0
        for cs in compiled:
            psi = sim.init_state(cs.K)
            for w, _n, ang in cs.intro:
                psi = sim.apply_gate(psi, sim.ry(ang), (w,))
            for v, wa, wp in cs.events:
                psi = sim.apply_gate(psi, mats[sub.get(v, v)], (wa, wp))
            vseq, qa, qp = cs.question
            qw = (qa, qp)
            for qv in reversed(vseq):
                psi = sim.apply_gate(
                    psi, mats[sub.get(qv, qv)].conj().T, qw)
            vals = [sim.prob_all_zero_on(sim.apply_gate(psi, E, qw), qw)
                    for E in effs]
            pred = 0 if vals[0].item() >= vals[1].item() else 1
            n_ok += (pred == cs.answer)
    return n_ok / len(compiled)


def pair_distances(model, compiled):
    """Symmetrised behavioural substitution distances for the three
    matched pairs + baseline accuracy."""
    mats, effs = block_matrices(model)
    base = family_acc(mats, effs, compiled)
    out = {"base_acc": base, "d": {}}
    for name, v, w in PAIRS:
        drop_vw = base - family_acc(mats, effs, compiled, sub={v: w})
        drop_wv = base - family_acc(mats, effs, compiled, sub={w: v})
        out["d"][name] = {"d_sym": 0.5 * (drop_vw + drop_wv),
                          "drop_v_to_w": drop_vw, "drop_w_to_v": drop_wv}
    ds = {name: out["d"][name]["d_sym"] for name, _v, _w in PAIRS}
    d_wazn = ds["wazn_mismatched"]
    d_other = max(ds["matched_I_a"], ds["matched_I_b"])
    out["wazn_pair_strictly_largest"] = bool(d_wazn > d_other)
    out["tie_at_max"] = bool(d_wazn == d_other)
    return out


# ------------------------------------- Cartan canonicalisation (descriptive)
_QUARTER = math.pi / 4.0
_HALF = math.pi / 2.0


def chamber_reduce(t, eps=1e-9):
    """Map raw Cartan coordinates (a,b,c) of N(a,b,c)=exp(i(aXX+bYY+cZZ))
    into the canonical Weyl chamber pi/4 >= x >= y >= |z| (z >= 0 unless
    all coords nonzero and the orbit's product sign is negative; z >= 0
    forced on the x=pi/4 wall). Allowed moves: per-coordinate shifts by
    pi/2 (local X(x)X / Y(x)Y / Z(x)Z times a global phase), coordinate
    permutations, and simultaneous sign flips of any two coordinates.
    DESCRIPTIVE ONLY (doc 26 SS3); boundary identifications applied
    approximately."""
    v = [x % _HALF for x in t]                     # [0, pi/2)
    v = [x - _HALF if x > _QUARTER + eps else x for x in v]
    v.sort(key=lambda x: -abs(x))
    if any(abs(x) <= eps for x in v):
        v = [abs(x) for x in v]
    else:
        p = v[0] * v[1] * v[2]
        v = [abs(v[0]), abs(v[1]), abs(v[2]) * (1.0 if p > 0 else -1.0)]
    if abs(v[2] + _QUARTER) <= eps:                # -pi/4 == +pi/4 wall
        v[2] = _QUARTER
    if abs(v[0] - _QUARTER) <= eps and v[2] < 0:   # x = pi/4 wall ident.
        v[2] = abs(v[2])
    v.sort(key=lambda x: -abs(x))
    return v


def makhlin_invariants(U):
    """Local invariants (G1 complex, G2 real) of a two-qubit unitary."""
    U = U.to(torch.complex128)
    det = torch.linalg.det(U)
    U = U / det ** 0.25
    M = (1.0 / math.sqrt(2.0)) * torch.tensor(
        [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
        dtype=torch.complex128)
    V = M.conj().T @ U @ M
    m = V.T @ V
    tr = torch.diagonal(m).sum()
    g1 = (tr * tr) / 16.0
    g2 = ((tr * tr - torch.diagonal(m @ m).sum()) / 4.0).real
    return complex(g1.item()), float(g2)


def _selftest_cartan():
    # known canonical points
    assert max(abs(a - b) for a, b in zip(
        chamber_reduce([0.0, 0.0, 0.0]), [0, 0, 0])) < 1e-9
    got = chamber_reduce([math.pi / 4, 0.0, 0.0])
    assert abs(got[0] - _QUARTER) < 1e-9 and abs(got[1]) < 1e-9
    got = chamber_reduce([math.pi / 4] * 3)                 # SWAP class
    assert all(abs(x - _QUARTER) < 1e-9 for x in got)
    # invariance under the allowed moves
    rng = random.Random(7)
    for _ in range(200):
        t = [rng.uniform(0, 2 * math.pi) for _ in range(3)]
        u = list(t)
        for _m in range(6):
            mv = rng.randrange(3)
            if mv == 0:
                i = rng.randrange(3)
                u[i] += _HALF * rng.choice([-3, -2, -1, 1, 2, 3])
            elif mv == 1:
                rng.shuffle(u)
            else:
                i, j = rng.sample(range(3), 2)
                u[i], u[j] = -u[i], -u[j]
        a, b = chamber_reduce(t), chamber_reduce(u)
        assert max(abs(x - y) for x, y in zip(a, b)) < 1e-7, (t, u, a, b)
    # Makhlin cross-check: N(canonical) is locally equivalent to the
    # full dressed block built from the same raw Cartan coordinates
    g = torch.Generator().manual_seed(11)
    from exp43a_model import KAK2Q
    for _ in range(10):
        blk = KAK2Q(generator=g)
        U = blk.matrix().detach()
        raw = [float(x) for x in blk.theta[12:15]]
        can = chamber_reduce(raw)
        N = cartan_core(torch.tensor(can[0]), torch.tensor(can[1]),
                        torch.tensor(can[2]))
        g1u, g2u = makhlin_invariants(U)
        g1n, g2n = makhlin_invariants(N)
        assert abs(g1u - g1n) < 1e-5 and abs(g2u - g2n) < 1e-5, \
            (raw, can, g1u, g1n, g2u, g2n)
    print("[selftest] cartan canonicalisation OK (known points, 200 "
          "random orbit moves, 10 Makhlin cross-checks)", flush=True)


def cartan_block(state_dict, verb):
    th = state_dict["blocks.%s.theta" % verb]
    return chamber_reduce([float(x) for x in th[12:15]])


def cartan_pair_distances(state_dict):
    out = {}
    for name, v, w in PAIRS:
        cv, cw = cartan_block(state_dict, v), cartan_block(state_dict, w)
        out[name] = math.sqrt(sum((a - b) ** 2 for a, b in zip(cv, cw)))
    return out


# ------------------------------------------------------------- statistics
def binom_tail_ge(k, n, p):
    """P[X >= k], X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


# ------------------------------------------------------------------- main
def main():
    t0 = time.time()
    compiled, verbs, meta, sha, emb_sha = run43a.load_family()
    for _name, v, w in PAIRS:
        assert v in verbs and w in verbs, (v, w, verbs)
    print("[wazn] family loaded: %d items, verbs=%s" % (len(compiled),
                                                        verbs), flush=True)
    _selftest_cartan()

    # eval parity: family_acc must reproduce exp43a_run.fast_eval
    probe = KakStoryModel(verbs, 99)
    mats, effs = block_matrices(probe)
    a_fast = run43a.fast_eval(probe, compiled)
    a_here = family_acc(mats, effs, compiled)
    assert abs(a_fast - a_here) < 1e-12, (a_fast, a_here)
    print("[selftest] family_acc parity with exp43a fast_eval: OK "
          "(%.4f)" % a_here, flush=True)

    # ---------------- control-of-the-control FIRST (frozen)
    print("[wazn] control: %d random KAK-block draws..." % N_RANDOM,
          flush=True)
    ctrl_succ, ctrl_ties, ctrl_rows = 0, 0, []
    for i in range(N_RANDOM):
        model = KakStoryModel(verbs, RANDOM_SEED_BASE + i)
        row = pair_distances(model, compiled)
        ctrl_succ += int(row["wazn_pair_strictly_largest"])
        ctrl_ties += int(row["tie_at_max"])
        ctrl_rows.append({"seed": RANDOM_SEED_BASE + i,
                          "base_acc": row["base_acc"],
                          "d_sym": {k: r["d_sym"]
                                    for k, r in row["d"].items()},
                          "wazn_largest": row["wazn_pair_strictly_largest"],
                          "tie": row["tie_at_max"]})
        if (i + 1) % 50 == 0:
            print("[control] %d/%d wazn-largest so far: %d (ties %d)"
                  % (i + 1, N_RANDOM, ctrl_succ, ctrl_ties), flush=True)
    ctrl_frac = ctrl_succ / N_RANDOM
    calibrated = bool(CAL_LO <= ctrl_frac <= CAL_HI)
    stage_a_void = not calibrated
    print("[control] wazn-pair-largest fraction: %.3f (band %.2f-%.2f) "
          "-> calibrated=%s" % (ctrl_frac, CAL_LO, CAL_HI, calibrated),
          flush=True)

    # ---------------- the 40 committed restarts
    per_restart = []
    for r in range(N_RESTARTS):
        path = STATE_FMT % r
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = KakStoryModel(verbs, 0)
        model.load_state_dict(blob["state_dict"])
        row = pair_distances(model, compiled)
        cart = cartan_pair_distances(blob["state_dict"])
        per_restart.append({
            "restart": r, "file": path, "seed": blob.get("seed"),
            "stage": blob.get("stage", "adam"),
            "recorded_train_acc": blob.get("train_acc"),
            "base_acc_recomputed": row["base_acc"],
            "d_sym": {k: v["d_sym"] for k, v in row["d"].items()},
            "d_detail": row["d"],
            "wazn_pair_strictly_largest": row["wazn_pair_strictly_largest"],
            "tie_at_max": row["tie_at_max"],
            "cartan_pair_dist_descriptive": cart,
        })
        print("[r%02d %s] base=%.3f d=%s largest=%s"
              % (r, per_restart[-1]["stage"], row["base_acc"],
                 {k: round(v["d_sym"], 4) for k, v in row["d"].items()},
                 row["wazn_pair_strictly_largest"]), flush=True)

    k_all = sum(int(x["wazn_pair_strictly_largest"]) for x in per_restart)
    ties_all = sum(int(x["tie_at_max"]) for x in per_restart)
    adam_rows = [x for x in per_restart if x["stage"] == "adam"]
    lbfgs_rows = [x for x in per_restart if x["stage"] == "lbfgs"]
    k_adam = sum(int(x["wazn_pair_strictly_largest"]) for x in adam_rows)

    # best.pt is a duplicate of the best polished restart; descriptive
    best_blob = torch.load("exp43a_best.pt", map_location="cpu",
                           weights_only=False)
    polished_note = {
        "rows": [{k: x[k] for k in ("restart", "stage",
                                    "recorded_train_acc", "d_sym",
                                    "wazn_pair_strictly_largest",
                                    "cartan_pair_dist_descriptive")}
                 for x in lbfgs_rows],
        "exp43a_best_pt": {"restart": best_blob.get("restart"),
                           "train_acc": best_blob.get("train_acc"),
                           "stage": best_blob.get("stage"),
                           "note": "duplicate of that restart's state "
                                   "file; not double-counted"},
        "descriptive_only": "n=%d cannot carry a criterion (doc 26 SS3)"
                            % len(lbfgs_rows),
    }

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    cartan_summary = {
        name: {"median_over_40": med([x["cartan_pair_dist_descriptive"]
                                      [name] for x in per_restart]),
               "median_adam_only": med([x["cartan_pair_dist_descriptive"]
                                        [name] for x in adam_rows])}
        for name, _v, _w in PAIRS}

    if calibrated:
        verdicts = {
            "random_block_null_calibrated": True,
            "stage_a_void": False,
            "wazn_screen_survives": bool(k_all >= K_SURVIVE),
            "wazn_screen_ruled_out": bool(k_all <= K_RULED_OUT),
            "wazn_screen_inconclusive": bool(K_RULED_OUT < k_all
                                             < K_SURVIVE),
        }
    else:
        verdicts = {
            "random_block_null_calibrated": False,
            "stage_a_void": True,
            "wazn_screen_survives": None,
            "wazn_screen_ruled_out": None,
            "wazn_screen_inconclusive": None,
            "void_note": "the metric is biased on random blocks; Stage A "
                         "is reported as VOID, not as a result (doc 26 "
                         "SS3). k is recorded above for the audit trail "
                         "only.",
        }

    results = {
        "experiment": "exp43 wazn screen, Stage A",
        "spec": "doc 26 (qnlp_private_docs/26, FROZEN): behavioural "
                "substitution distance (exp39's notion) on the frozen "
                "exp43a family; per restart, is d(wazn pair) the largest "
                "of the three matched pairs; k over 40 restarts; "
                "survives iff k>=19, ruled_out iff k<=13, else "
                "inconclusive; control first: 200 random KAK draws must "
                "put the wazn pair largest in 28-39% or Stage A is VOID. "
                "There is deliberately NO wazn_confirmed verdict.",
        "reopens_exp39_note": "post-hoc in motivation, pre-registered in "
                "execution; this reopens a question exp39 closed (all-"
                "Form-I design could not separate 'nothing verb-specific' "
                "from 'encodes the shared pattern'). exp39's and exp33's "
                "verdicts stand as computed (doc 26 SS1, SS4).",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "family_dataset_sha256": sha,
        "embeddings_sha256": emb_sha,
        "n_family_items": len(compiled),
        "pairs": {name: [v, w] for name, v, w in PAIRS},
        "tie_convention": "'largest' = strictly larger than both other "
                          "pair distances; ties recorded and counted as "
                          "non-successes",
        "artifact_note_flagged": "doc 26 SS2 describes the 40 state files "
                "as Adam solutions (train 0.645-0.705), but "
                "exp43a_state_r01/r08/r30.pt contain the L-BFGS-polished "
                "states (0.93/0.955/0.95, metadata stage='lbfgs'): "
                "exp43a's finalize overwrote a restart's file when polish "
                "improved it, so the Adam states of those three restarts "
                "no longer exist. The frozen headline ('k uses all 40 "
                "restarts') is computed over the 40 files as committed "
                "(37 adam + 3 lbfgs); k over the 37 pure-Adam files is "
                "reported as a descriptive robustness number only.",
        "control_random_blocks": {
            "n_draws": N_RANDOM, "seed_base": RANDOM_SEED_BASE,
            "wazn_pair_largest_count": ctrl_succ,
            "wazn_pair_largest_fraction": ctrl_frac,
            "ties_at_max": ctrl_ties,
            "calibration_band": [CAL_LO, CAL_HI],
            "per_draw": ctrl_rows,
        },
        "per_restart": per_restart,
        "headline": {
            "k_wazn_largest_over_40": k_all,
            "ties_at_max_over_40": ties_all,
            "one_sided_binomial_p_at_third": binom_tail_ge(k_all,
                                                           N_RESTARTS,
                                                           1.0 / 3.0),
            "k_adam_only_37_descriptive": k_adam,
            "n_adam_only": len(adam_rows),
        },
        "lbfgs_polished_descriptive": polished_note,
        "cartan_descriptive": {
            "note": "Euclidean distance between canonicalised Cartan "
                    "vectors (a,b,c); Weyl-chamber reduction self-tested "
                    "(200 orbit moves, Makhlin invariant cross-checks); "
                    "DESCRIPTIVE ONLY, never a verdict (doc 26 SS3)",
            "summary": cartan_summary,
        },
        "verdicts": verdicts,
        "git_note": git_note(),
        "wall_time_sec": round(time.time() - t0, 1),
    }
    path = fresh_results_path("results_exp43_wazn_screen")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("[wazn] wrote %s" % path)
    print("[wazn] control frac=%.3f calibrated=%s | k=%d/40 (adam-only "
          "%d/%d) | verdicts=%s"
          % (ctrl_frac, calibrated, k_all, k_adam, len(adam_rows),
             json.dumps({k: v for k, v in verdicts.items()
                         if k != "void_note"})))


if __name__ == "__main__":
    main()
