# -*- coding: utf-8 -*-
"""exp85 driver: identity vs uniform at T=3 + FLOOR arm.

REGISTRATION: qnlp_private_docs/77_exp85_identity_T3.md (FROZEN 2026-09-17)

Design:
  - Dataset: stories_exp85_QA.json (exp85_datagen.py, seed 85999, sha pinned)
  - Arms: A3-T3-identity, A3-T3-uniform, FLOOR
  - Seeds: 85000..85014 (15 fresh seeds)
  - T=3 cell: THREE verb blocks composed at depth 3; QR-unitarized (doc 77 §0)
  - Loss: plain CE (best-known config, frozen exp83 verdict)
  - FLOOR: CE + identity underlying + fixed label scramble seed 8599942
  - Unitarity drift threshold: 1e-6 (exp81 invalidation line); hard-fail if exceeded

FORBIDDEN_SEEDS guard: only range(85000, 85015) allowed.

Gates (run BEFORE production):
  python exp85_run.py --gate-datagen      # GATE_DATAGEN
  python exp85_run.py --gate-t3-unitarity # GATE_T3_UNITARITY
  python exp85_run.py --gate-floor        # GATE_FLOOR_NONVACUOUS
  python exp85_run.py --gate-smoke        # GATE_SMOKE (1 seed, 5 epochs)

Production:
  python exp85_run.py --worker i --nworkers 5
  python exp85_run.py --arm A3-T3-identity --seed 85000  # single unit
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time
import traceback

import torch
import torch.nn.functional as F

import exp42_sim_adapter as sim
import exp73_run as R73

SPEC = "qnlp_private_docs/77_exp85_identity_T3.md (FROZEN 2026-09-17)"

EXP85_DATA = "stories_exp85_QA.json"
EXP85_CKPT = "exp85_ckpt"

DATASET_SHA_85 = ("9566a4a5bc305f66c0a9fb8e7e467ec3f22b9d0e3de298cbf7458104"
                  "c1554b00")                  # stories_exp85_QA.json sha

GATE_DATAGEN_FILE        = "exp85_gate_DATAGEN.json"
GATE_T3_UNITARITY_FILE   = "exp85_gate_T3_UNITARITY.json"
GATE_FLOOR_FILE          = "exp85_gate_FLOOR_NONVACUOUS.json"
SMOKE_FILE               = "exp85_smoke.json"
SMOKE_FILE_PASSED        = "exp85_smoke_PASSED.json"

# Arms: T=3 cell only (T=1, T=2 emitted descriptively at T=3 cell scale)
ARMS = ["A3-T3-identity", "A3-T3-uniform", "FLOOR"]
UNDERLYING_ARM = "A3-T3-identity"   # FLOOR uses identity underlying

SEED_BASE = 85000
N_SEEDS = 15
SEEDS = list(range(85000, 85015))

# FLOOR scramble seed (doc 77 §2); outside all experiment seed ranges
FLOOR_SCRAMBLE_SEED = 8_599_942

# doc 62 §24.3 bounded-redraw (identical to exp82/exp83)
REDRAW_MAX    = 20
REDRAW_STRIDE = 100_000

# Unitarity drift threshold (exp81 invalidation line, doc 77 §0)
DRIFT_THRESHOLD = 1e-6

# FORBIDDEN_SEEDS guard: all prior bases + their confirm ranges
FORBIDDEN_SEEDS = (
    set(range(5))               |   # exp73 originals 0-4
    set(range(42000, 44000))    |   # 42xxx and 43xxx (prior experiment bases)
    set(range(69000, 70000))    |   # 69xxx (exp69 generator seeds)
    set(range(79000, 81000))    |   # 79xxx and 80xxx (prior + confirm draws)
    set(range(82000, 82015))    |   # exp82 base draw
    set(range(82500, 82515))    |   # exp82c confirm draw
    set(range(83000, 83015))    |   # exp83 base draw (forbidden per doc 77 §0)
    set(range(83500, 83515))    |   # canonical +1000 of 82500-series
    set(range(84000, 84015))        # exp83's reserved confirm base
)
# Note: 86000-series is exp85's reserved confirm base; NOT listed here (not yet used)

# Save original init scheme BEFORE any rebinding
_ORIG_INIT = R73.apply_init_scheme

# ── Inject T=3 ARM_CFG entries (must happen at module import time) ──────────
# R73.ARM_CFG has no T=3 entries -- inject before any arm_items / build_model call.
_T3_CFG_BASE = {"enc": "amplitude", "rung": "A3", "qpr": 3, "d": 8,
                "part": "A", "stage": 1}
R73.ARM_CFG["A3-T3-identity"] = dict(_T3_CFG_BASE,
    T=3, init="identity", family="A3-identity",
    role="exp85 T=3 identity arm (doc 77 §2)")
R73.ARM_CFG["A3-T3-uniform"] = dict(_T3_CFG_BASE,
    T=3, init="uniform", family="A3-uniform",
    role="exp85 T=3 uniform arm (doc 77 §2)")


# ── doc 62 §24.3 bounded-redraw (copied from exp82/exp83, same code object) ─
def apply_init_scheme_85(m, arm, seed):
    """Wrap R73.apply_init_scheme with bounded-redraw for identity.

    Uniform arm: delegates to _ORIG_INIT unchanged.
    Identity arm: attempt k=0 is _ORIG_INIT(m, arm, seed), byte-identical to
    exp73's own behaviour. On AssertionError stride by REDRAW_STRIDE and retry,
    up to REDRAW_MAX attempts. Exhaustion is a STOP."""
    cfg = R73.ARM_CFG[R73.base_arm(arm)]
    if cfg["init"] == "uniform":
        return _ORIG_INIT(m, arm, seed)

    refusals = []
    for k in range(REDRAW_MAX):
        eff = int(seed) + REDRAW_STRIDE * k
        try:
            rep = _ORIG_INIT(m, arm, eff)
            rep["redraw_doc62_SS24_3"] = {
                "attempts_used": k + 1,
                "refused_attempts": refusals,
                "effective_seed_of_record": eff,
                "stride": REDRAW_STRIDE,
                "max_attempts": REDRAW_MAX,
                "note": ("attempt 0 is byte-identical to the pre-amendment behaviour; "
                         "stride 100000 per refusal, 20-STOP per doc 62 §24.3"),
            }
            return rep
        except AssertionError as e:
            refusals.append({"attempt": k, "effective_seed": eff, "refusal": str(e)})
    raise RuntimeError(
        "doc 62 §24.3 STOP: %d consecutive identity draws refused for arm %s "
        "seed %d -- not sampling noise. Refusals: %r"
        % (REDRAW_MAX, arm, seed, refusals))


# ── Dataset loading ──────────────────────────────────────────────────────────
import exp42_compiler as comp

def load_exp85(path=None):
    """Load stories_exp85_QA.json, verify sha self-consistency."""
    path = path or EXP85_DATA
    if not os.path.exists(path):
        raise RuntimeError(
            "%s not found -- run exp85_datagen.py first" % path)
    blob = json.load(open(path, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    if comp.canonical_sha256(items) != meta["dataset_sha256"]:
        raise RuntimeError("exp85 dataset SHA self-check FAILED for %s" % path)
    assert meta["dataset_sha256"] == DATASET_SHA_85, (
        "exp85 dataset SHA mismatch: loaded %s, registered %s"
        % (meta["dataset_sha256"], DATASET_SHA_85))
    return items, meta, path


# ── FLOOR label scramble ─────────────────────────────────────────────────────
def _apply_floor_scramble_with_seed(items, seed):
    """Deterministic label scramble; same map applied to all splits."""
    rng = random.Random(seed)
    id_to_label = {
        it["id"]: rng.choice(["yes", "no"])
        for it in sorted(items, key=lambda x: x["id"])
    }
    return [dict(it, answer=id_to_label[it["id"]]) for it in items], id_to_label


def apply_floor_scramble(items):
    """Apply the FIXED production FLOOR scramble."""
    return _apply_floor_scramble_with_seed(items, FLOOR_SCRAMBLE_SEED)


# ── Local story forward pass with QR-unitarized verb blocks (doc 77 §0) ──────
import exp69_run as R69


def story_values_85(m, cs):
    """Story values with QR-unitarized verb blocks.

    Mirrors R73._readout exactly but applies sim.unitarize() to each verb
    block matrix before state application.  Autograd flows through
    torch.linalg.qr (differentiable).  Does NOT check drift per item for
    efficiency -- drift is checked once at training startup (see train_85)."""
    cache = {}

    def block(v):
        if v not in cache:
            cache[v] = sim.unitarize(m.verb_matrix(v))
        return cache[v]

    nw = R73.qubits_per_referent(m) * cs.K
    psi = R73.apply_intro(m, sim.init_state(nw), cs)
    for v, wa, wp in cs.events:
        psi = sim.apply_gate(psi, block(v), R73.reg_of(m, wa) + R73.reg_of(m, wp))
    vseq, qa, qp = cs.question
    qw = R73.reg_of(m, qa) + R73.reg_of(m, qp)
    for qv in reversed(vseq):
        psi = sim.apply_gate(psi, block(qv).conj().T, qw)
    return torch.stack([sim.prob_all_zero_on(
        sim.apply_gate(psi, eff.matrix().conj().T, qw), qw)
        for eff in (m.q_pos, m.q_neg)])


def evaluate_85(m, compiled):
    """Evaluate accuracy using story_values_85."""
    with torch.no_grad():
        correct = sum(
            int(torch.argmax(story_values_85(m, cs)).item() == cs.answer)
            for cs in compiled
        )
    return correct / len(compiled)


def evaluate_with_traces_85(m, compiled):
    """Evaluate with per-item traces (for test checkpoint)."""
    rows, correct = [], 0
    with torch.no_grad():
        for cs in compiled:
            vals = story_values_85(m, cs)
            pred = int(torch.argmax(vals).item())
            ok = int(pred == cs.answer)
            correct += ok
            r = dict(cs.meta)
            r.update({"pred": pred, "target": int(cs.answer), "correct": bool(ok),
                      "p_yes": float(vals[0]), "p_no": float(vals[1])})
            rows.append(r)
    return correct / len(compiled), rows


def _startup_drift_check(m, verbs):
    """Check post-QR unitarity drift for all verb blocks at model init.

    Raises RuntimeError if any block's drift exceeds DRIFT_THRESHOLD (1e-6).
    Returns max drift observed.  Called once per unit before training loop.
    (doc 77 §0: 'any unit exceeding it hard-fails with the drift logged')
    """
    n = 2 ** (2 * m.q)   # matrix dimension for A3: 2^6 = 64
    I_mat = torch.eye(n, dtype=sim.CDTYPE)
    max_drift = 0.0
    drift_log = {}
    with torch.no_grad():
        for v in verbs:
            raw = m.verb_matrix(v)
            U = sim.unitarize(raw)
            drift = float((U.conj().T @ U - I_mat).abs().max().item())
            drift_log[v] = drift
            if drift > DRIFT_THRESHOLD:
                raise RuntimeError(
                    "exp85 STOP: post-QR unitarity drift=%.3e > threshold=%.1e "
                    "for verb %r -- post-fix unitarity not achieved at depth-3 "
                    "composition (exp81 invalidation line, doc 77 §0). "
                    "All verb drifts: %r" % (drift, DRIFT_THRESHOLD, v, drift_log))
            max_drift = max(max_drift, drift)
    return max_drift, drift_log


# ── Training loop ────────────────────────────────────────────────────────────
def train_85(arm_85, seed, store, verbs, max_epochs, fm, save=True):
    """exp85 training: T=3 cell with QR-unitarized verb blocks.

    Call order (compile -> manual_seed -> build -> init -> Adam -> Random)
    preserved verbatim from exp73/exp82/exp83 (doc 61 §5.3 discipline)."""
    underlying = UNDERLYING_ARM
    items_all, meta, path = store
    items = [i for i in items_all if int(i.get("T", 1)) == 3]

    if arm_85 == "FLOOR":
        items, floor_id_to_label = apply_floor_scramble(items)
        floor_scramble_info = {
            "scramble_seed": FLOOR_SCRAMBLE_SEED,
            "n_items_scrambled": len(items),
            "note": "same scramble applied to train/val/test via id_to_label map",
        }
    else:
        floor_id_to_label = None
        floor_scramble_info = None

    ang, _ = R73.arm_angles(underlying, fm)
    data = {s: R69.compile_all([i for i in items if i["split"] == s], ang)
            for s in ("train", "val", "test")}

    torch.manual_seed(seed)
    m = R73.build_model(underlying, verbs, seed, fm)
    # FLOOR uses identity underlying; A3-T3-identity and A3-T3-uniform use
    # their own scheme (uniform branch leaves the committed Uniform[0,2pi) intact)
    init_arm = UNDERLYING_ARM if arm_85 == "FLOOR" else arm_85
    init_info = apply_init_scheme_85(m, init_arm, seed)

    # Startup drift check (doc 77 §0: hard-fail if any block exceeds 1e-6)
    max_startup_drift, drift_log = _startup_drift_check(m, verbs)
    print("[%s s%d] startup drift check OK: max_drift=%.2e"
          % (arm_85, seed, max_startup_drift), flush=True)

    opt = torch.optim.Adam(m.parameters(), lr=R73.LR)
    rng = random.Random(R73.RNG_BASE + seed)
    best = {"val": -1.0, "train": -1.0, "epoch": -1, "state": None}
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        order = list(range(len(data["train"])))
        rng.shuffle(order)
        for s in range(0, len(order), R73.BATCH):
            opt.zero_grad()
            bl = 0.0
            for i in order[s:s + R73.BATCH]:
                cs = data["train"][i]
                z = torch.log(story_values_85(m, cs) + 1e-12)
                bl = bl + F.cross_entropy(z.unsqueeze(0), torch.tensor([cs.answer]))
            (bl / min(R73.BATCH, len(order) - s)).backward()
            assert math.isfinite(R73.grad_global_norm(m)), (
                "non-finite gradient at ep %d arm %s seed %d" % (ep, arm_85, seed))
            opt.step()
        with torch.no_grad():
            va = evaluate_85(m, data["val"])
        if va > best["val"]:
            with torch.no_grad():
                best = {"val": va,
                        "train": evaluate_85(m, data["train"]),
                        "epoch": ep,
                        "state": copy.deepcopy(m.state_dict())}
        if ep % 25 == 0 or ep == max_epochs:
            print("[%s s%d] ep %3d/%d val=%.3f best=%.3f %.0fs"
                  % (arm_85, seed, ep, max_epochs, va, best["val"],
                     time.time() - t0), flush=True)

    m.load_state_dict(best["state"])
    te, rows = evaluate_with_traces_85(m, data["test"])

    pt = sha = None
    if save:
        os.makedirs(EXP85_CKPT, exist_ok=True)
        pt = os.path.join(EXP85_CKPT, "%s_s%d.pt" % (arm_85, seed))
        torch.save(best["state"], pt)
        sha = R73.sha256_file(pt)

    rd = init_info.get("redraw_doc62_SS24_3", {})
    return {
        "arm": arm_85,
        "underlying_model_arm": underlying,
        "seed": seed,
        "spec": SPEC,
        "exp85_seed_base": SEED_BASE,
        "exp85_seed_index": seed - SEED_BASE,
        "init_scheme": init_info,
        "floor_scramble_info": floor_scramble_info,
        "unitarity_drift": {
            "startup_max_drift": max_startup_drift,
            "per_verb_drift": drift_log,
            "threshold": DRIFT_THRESHOLD,
            "passed": bool(max_startup_drift <= DRIFT_THRESHOLD),
            "note": ("post-QR drift check at model init (doc 77 §0, exp81 line); "
                     "QR-unitarization applied to all verb blocks in forward pass"),
        },
        "train_acc": best["train"],
        "val_acc": best["val"],
        "selected_epoch": best["epoch"],
        "heldout_test_acc": te,
        "val_minus_heldout": best["val"] - te,
        "gate_met": bool(best["train"] >= R73.TRAIN_GATE and
                         best["val"] >= R73.VAL_GATE),
        "n_params": sum(p.numel() for p in m.parameters()),
        "n_train": len(data["train"]),
        "n_val": len(data["val"]),
        "n_test": len(data["test"]),
        "T": 3,
        "max_epochs": max_epochs,
        "lr": R73.LR,
        "batch": R73.BATCH,
        "dataset": path,
        "dataset_sha256": meta["dataset_sha256"],
        "dataset_sha256_registered": DATASET_SHA_85,
        "dataset_sha256_matches_registered": (
            meta["dataset_sha256"] == DATASET_SHA_85),
        "ckpt_path": pt,
        "ckpt_sha256": sha,
        "heldout_items": rows,
        "redraw_attempts_used": rd.get("attempts_used"),
        "redraw_effective_seed": rd.get("effective_seed_of_record"),
        "wall_sec": time.time() - t0,
    }


# ── Job list ─────────────────────────────────────────────────────────────────
def job_list_85():
    """(arm, seed) pairs; identity arm first for anchor-first dispatch."""
    return [(arm, s) for arm in ARMS for s in SEEDS]


# ── Gate artifacts pre-check ──────────────────────────────────────────────────
def check_gates():
    missing = [f for f in [GATE_DATAGEN_FILE, GATE_T3_UNITARITY_FILE,
                            GATE_FLOOR_FILE, SMOKE_FILE_PASSED]
               if not os.path.exists(f)]
    if missing:
        sys.exit("FATAL: gate artifact(s) missing -- run gates before production:\n  %s"
                 % "\n  ".join(missing))
    for gf, key in [(SMOKE_FILE_PASSED, "assertions_passed"),
                    (GATE_DATAGEN_FILE, "passed"),
                    (GATE_T3_UNITARITY_FILE, "passed"),
                    (GATE_FLOOR_FILE, "passed")]:
        g = json.load(open(gf, encoding="utf-8"))
        if not g.get(key):
            sys.exit("FATAL: %s: %s is not True" % (gf, key))
    return True


# ── GATE_DATAGEN ─────────────────────────────────────────────────────────────
def gate_datagen():
    """GATE_DATAGEN: verify slot disjointness, T=3 event-slot assignments,
    label balance, and sha self-consistency.

    DTF: a deliberately overlapping slot assignment must trip the disjointness check.
    Writes exp85_gate_DATAGEN.json.
    """
    if not os.path.exists(EXP85_DATA):
        raise RuntimeError("%s not found -- run exp85_datagen.py first" % EXP85_DATA)

    items_all, meta, _ = load_exp85()
    V1 = set(meta["slot_assignment"]["V1"])
    V2 = set(meta["slot_assignment"]["V2"])
    V3 = set(meta["slot_assignment"]["V3"])
    all_verbs = set(meta["verb_inventory"].keys())

    # ── 1: Slot disjointness ─────────────────────────────────────────────────
    assert not (V1 & V2), "V1 ∩ V2 not empty: %s" % (V1 & V2)
    assert not (V1 & V3), "V1 ∩ V3 not empty: %s" % (V1 & V3)
    assert not (V2 & V3), "V2 ∩ V3 not empty: %s" % (V2 & V3)
    assert V1 | V2 | V3 == all_verbs, "V1 ∪ V2 ∪ V3 != full inventory"
    assert len(V1) == len(V2) == len(V3) == 2, "slots are not 2/2/2"

    # ── 2: T=3 event-slot assignments ────────────────────────────────────────
    t3_items = [i for i in items_all if i["T"] == 3]
    slot_violations = []
    for i in t3_items:
        if i["events"][0]["verb"] not in V1:
            slot_violations.append("%s ev0 verb not in V1" % i["id"])
        if i["events"][1]["verb"] not in V2:
            slot_violations.append("%s ev1 verb not in V2" % i["id"])
        if i["events"][2]["verb"] not in V3:
            slot_violations.append("%s ev2 verb not in V3" % i["id"])
    assert not slot_violations, \
        "T=3 slot violations (%d): %r" % (len(slot_violations), slot_violations[:4])

    # ── 3: Label balance (exact by construction: each pair has one yes/one no) -
    from collections import Counter
    balance_ok = {}
    for t in (1, 2, 3):
        sub = [i for i in items_all if i["T"] == t]
        for s in ("train", "val", "test"):
            c = Counter(i["answer"] for i in sub if i["split"] == s)
            balance_ok["T%d_%s" % (t, s)] = (c["yes"] == c["no"],
                                              c["yes"], c["no"])
    all_balanced = all(v[0] for v in balance_ok.values())
    assert all_balanced, "label imbalance detected: %r" % balance_ok

    # ── DTF: overlapping slot assignment must trip disjointness check ─────────
    dtf_overlap_msg = "NOT RUN"
    try:
        # V1_bad and V2_bad share a verb
        bad_V1 = list(V1)
        bad_V2 = [bad_V1[0]] + list(V2)[:1]   # first verb of V1 is also in bad_V2
        bad_inter = set(bad_V1) & set(bad_V2)
        assert not bad_inter, \
            "dtf_overlap: overlapping slots detected: V1∩V2=%s" % bad_inter
        dtf_overlap_msg = "ERROR: designed-to-fail did NOT fire (overlap not detected)"
    except AssertionError as e:
        dtf_overlap_msg = "OK: designed-to-fail fired: %s" % str(e)
    assert dtf_overlap_msg.startswith("OK:"), dtf_overlap_msg

    result = {
        "gate": "GATE_DATAGEN",
        "spec": SPEC,
        "passed": True,
        "dataset": EXP85_DATA,
        "dataset_sha256": meta["dataset_sha256"],
        "slot_assignment": {
            "V1": sorted(V1), "V2": sorted(V2), "V3": sorted(V3),
            "all_verbs": sorted(all_verbs),
        },
        "slot_sizes": {"V1": len(V1), "V2": len(V2), "V3": len(V3)},
        "slot_pairwise_disjoint": True,
        "slot_union_equals_inventory": True,
        "n_t3_items_checked": len(t3_items),
        "n_t3_slot_violations": len(slot_violations),
        "all_t3_events_in_correct_slot": True,
        "label_balance": {k: {"balanced": v[0], "n_yes": v[1], "n_no": v[2]}
                          for k, v in balance_ok.items()},
        "all_splits_balanced": all_balanced,
        "dtf_overlapping_slot": dtf_overlap_msg,
        "n_items_total": len(items_all),
        "split_counts": meta["split_counts"],
    }

    json.dump(result, open(GATE_DATAGEN_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[GATE_DATAGEN] PASSED -- wrote %s" % GATE_DATAGEN_FILE, flush=True)
    print("  slots: V1=%s V2=%s V3=%s" % (sorted(V1), sorted(V2), sorted(V3)))
    print("  T=3 items: %d checked, 0 slot violations" % len(t3_items))
    return result


# ── GATE_T3_UNITARITY ─────────────────────────────────────────────────────────
def gate_t3_unitarity():
    """GATE_T3_UNITARITY: compose 3 verb blocks at A3-T3, verify drift ≤ 1e-6.

    1. Build a smoke model (seed 85000, no training).
    2. Unitarize each of 3 verb block matrices via QR.
    3. Check per-block and composed-product drift ≤ DRIFT_THRESHOLD.
    DTF: perturb a block matrix by ×1.01 to guarantee drift > 1e-6.
    Writes exp85_gate_T3_UNITARITY.json.
    """
    items_all, meta, path = load_exp85()
    verbs = sorted(meta["verb_inventory"].keys())
    V1 = meta["slot_assignment"]["V1"]
    fm = R73.feature_maps()

    torch.manual_seed(85000)
    m = R73.build_model(UNDERLYING_ARM, verbs, 85000, fm)
    n = 2 ** (2 * m.q)   # 64 for A3 (q=3, 2 referents -> 6 wires -> 2^6)
    I_mat = torch.eye(n, dtype=sim.CDTYPE)

    per_block_drift = {}
    unitarized_blocks = {}
    raw_blocks = {}
    with torch.no_grad():
        for v in verbs:
            raw = m.verb_matrix(v)
            raw_blocks[v] = raw
            U = sim.unitarize(raw)
            unitarized_blocks[v] = U
            drift = float((U.conj().T @ U - I_mat).abs().max().item())
            per_block_drift[v] = drift
            assert drift <= DRIFT_THRESHOLD, (
                "GATE_T3_UNITARITY FAILED: verb %r drift=%.3e > threshold=%.1e"
                % (v, drift, DRIFT_THRESHOLD))

    # Compose 3 verb blocks (depth-3 composition as in T=3 stories)
    # Use V1[0], V2[0], V3[0] as a representative triple
    V2 = meta["slot_assignment"]["V2"]
    V3 = meta["slot_assignment"]["V3"]
    v1_rep = V1[0]; v2_rep = V2[0]; v3_rep = V3[0]
    U1 = unitarized_blocks[v1_rep]
    U2 = unitarized_blocks[v2_rep]
    U3 = unitarized_blocks[v3_rep]

    # Depth-3 composition: U3 @ U2 @ U1 (applied right-to-left on state)
    U_composed = U3 @ U2 @ U1
    composed_drift = float((U_composed.conj().T @ U_composed - I_mat).abs().max().item())
    assert composed_drift <= DRIFT_THRESHOLD, (
        "GATE_T3_UNITARITY FAILED: composed 3-block drift=%.3e > threshold=%.1e"
        % (composed_drift, DRIFT_THRESHOLD))

    # ── DTF: perturbed (non-unitary) block must have drift > 1e-6 ─────────────
    raw_rep = raw_blocks[v1_rep]
    raw_perturbed = raw_rep * 1.1   # scale breaks unitarity detectably
    drift_perturbed = float((raw_perturbed.conj().T @ raw_perturbed - I_mat).abs().max().item())
    dtf_msg = "NOT RUN"
    try:
        assert drift_perturbed <= DRIFT_THRESHOLD, (
            "dtf_perturbed: drift=%.3e > threshold=%.1e for ×1.1-scaled matrix"
            % (drift_perturbed, DRIFT_THRESHOLD))
        dtf_msg = "ERROR: designed-to-fail did NOT fire (perturbed matrix drift not detected)"
    except AssertionError as e:
        dtf_msg = "OK: designed-to-fail fired: %s" % str(e)
    assert dtf_msg.startswith("OK:"), dtf_msg

    # ── DTF2: raw (non-unitarized) composition ─────────────────────────────────
    R1 = raw_blocks[v1_rep]
    R2 = raw_blocks[v2_rep]
    R3 = raw_blocks[v3_rep]
    R_composed = R3 @ R2 @ R1
    raw_composed_drift = float((R_composed.conj().T @ R_composed - I_mat).abs().max().item())
    # Raw composed may or may not exceed threshold; we record it descriptively
    raw_exceeds = raw_composed_drift > DRIFT_THRESHOLD

    result = {
        "gate": "GATE_T3_UNITARITY",
        "spec": SPEC,
        "passed": True,
        "seed": 85000,
        "arm": UNDERLYING_ARM,
        "rung": "A3",
        "q_per_referent": m.q,
        "matrix_dim": n,
        "drift_threshold": DRIFT_THRESHOLD,
        "per_block_drift": per_block_drift,
        "max_per_block_drift": max(per_block_drift.values()),
        "all_per_block_within_threshold": True,
        "representative_triple": [v1_rep, v2_rep, v3_rep],
        "depth3_composition": {
            "formula": "U3 @ U2 @ U1",
            "composed_drift": composed_drift,
            "within_threshold": bool(composed_drift <= DRIFT_THRESHOLD),
        },
        "raw_composition_descriptive": {
            "raw_composed_drift": raw_composed_drift,
            "exceeds_threshold": raw_exceeds,
            "note": ("raw (non-unitarized) 3-block composition drift, recorded "
                     "descriptively to show what unitarization fixes"),
        },
        "dtf_perturbed_block": dtf_msg,
        "note": (
            "QR-unitarization path exercised at depth 3 (doc 77 §0). "
            "Post-fix drift <= 1e-6 for all 6 verb blocks and the composed product. "
            "sim.unitarize() projects via Q*phase from torch.linalg.qr."),
    }

    json.dump(result, open(GATE_T3_UNITARITY_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[GATE_T3_UNITARITY] PASSED -- wrote %s" % GATE_T3_UNITARITY_FILE,
          flush=True)
    print("  max per-block drift: %.2e  composed-3 drift: %.2e"
          % (max(per_block_drift.values()), composed_drift))
    print("  raw-composed drift (no QR): %.2e  exceeds_threshold=%s"
          % (raw_composed_drift, raw_exceeds))
    return result


# ── GATE_FLOOR_NONVACUOUS ─────────────────────────────────────────────────────
def gate_floor_nonvacuous():
    """GATE_FLOOR_NONVACUOUS: verify the FLOOR scramble changes at least one label.

    Same checks as exp83: mismatch_count > 0, deterministic, same scramble
    for train and eval.
    DTFs: different seed gives different scramble; assert mismatch_count==0 fails.
    Writes exp85_gate_FLOOR_NONVACUOUS.json.
    """
    items_all, meta, _ = load_exp85()
    items = [i for i in items_all if int(i.get("T", 1)) == 3]
    n_total = len(items)
    n_train = sum(1 for i in items if i["split"] == "train")
    n_val   = sum(1 for i in items if i["split"] == "val")
    n_test  = sum(1 for i in items if i["split"] == "test")

    scrambled, id_to_label = apply_floor_scramble(items)
    scrambled2, id_to_label2 = apply_floor_scramble(items)
    assert id_to_label == id_to_label2, "FLOOR scramble is not deterministic"
    assert all(s1["answer"] == s2["answer"]
               for s1, s2 in zip(scrambled, scrambled2)), \
        "FLOOR scramble gives different item outputs on second call"

    orig_by_id = {it["id"]: it["answer"] for it in items}
    mismatch_count = sum(1 for it in scrambled
                         if it["answer"] != orig_by_id[it["id"]])
    mismatch_fraction = mismatch_count / n_total if n_total else 0.0
    assert mismatch_count > 0, \
        "FLOOR scramble did not change any labels (mismatch_count=0 on %d items)" % n_total

    # ── DTF 1: different seeds give different scrambles ───────────────────────
    _, id_to_label_alt = _apply_floor_scramble_with_seed(items, FLOOR_SCRAMBLE_SEED + 1)
    dtf_diff_seed = "NOT RUN"
    try:
        assert id_to_label == id_to_label_alt, "dtf: expected different seeds to match"
        dtf_diff_seed = "ERROR: designed-to-fail did NOT fire"
    except AssertionError as e:
        dtf_diff_seed = "OK: designed-to-fail fired: different seeds give different scrambles"
    assert dtf_diff_seed.startswith("OK:"), dtf_diff_seed

    # ── DTF 2: assert mismatch_count==0 on real scramble ─────────────────────
    dtf_zero_mismatch = "NOT RUN"
    try:
        assert mismatch_count == 0, \
            "dtf_zero_mismatch: mismatch_count=%d" % mismatch_count
        dtf_zero_mismatch = "ERROR: designed-to-fail did NOT fire (mismatch was 0)"
    except AssertionError as e:
        dtf_zero_mismatch = "OK: designed-to-fail fired: %s" % str(e)
    assert dtf_zero_mismatch.startswith("OK:"), dtf_zero_mismatch

    result = {
        "gate": "GATE_FLOOR_NONVACUOUS",
        "spec": SPEC,
        "passed": True,
        "floor_scramble_seed": FLOOR_SCRAMBLE_SEED,
        "T": 3,
        "n_items_total": n_total,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "mismatch_count": mismatch_count,
        "mismatch_fraction": mismatch_fraction,
        "scramble_is_deterministic": True,
        "train_val_test_use_same_scramble": True,
        "note": ("same id_to_label map applied to all T=3 splits before "
                 "split-filtering; per exp79 vacuous-floor lesson"),
        "dtf_different_seed": dtf_diff_seed,
        "dtf_zero_mismatch": dtf_zero_mismatch,
    }

    json.dump(result, open(GATE_FLOOR_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[GATE_FLOOR_NONVACUOUS] PASSED -- wrote %s" % GATE_FLOOR_FILE, flush=True)
    print("  n_items=%d mismatch=%d (%.1f%%)"
          % (n_total, mismatch_count, 100 * mismatch_fraction), flush=True)
    return result


# ── GATE_SMOKE ────────────────────────────────────────────────────────────────
def gate_smoke():
    """GATE_SMOKE: all 3 arms, 1 seed (85000), 5 epochs, finite losses at T=3.

    Writes exp85_smoke.json + exp85_smoke_PASSED.json.
    DTF: NaN finite-check assertion shown failing.
    """
    seed = SEEDS[0]   # 85000
    N_SMOKE_EPOCHS = 5

    items_all, meta, _ = load_exp85()
    verbs = sorted(meta["verb_inventory"].keys())
    fm = R73.feature_maps()

    # DTF: NaN loses the finite check
    dtf_nan_msg = "NOT RUN"
    try:
        assert math.isfinite(float("nan")), "dtf_nan: nan is not finite"
        dtf_nan_msg = "ERROR: designed-to-fail did NOT fire"
    except AssertionError as e:
        dtf_nan_msg = "OK: designed-to-fail fired: %s" % str(e)
    assert dtf_nan_msg.startswith("OK:"), dtf_nan_msg

    store = (items_all, meta, EXP85_DATA)
    arms_results = {}
    for arm_85 in ARMS:
        print("[smoke] running %s seed=%d %d epochs at T=3..."
              % (arm_85, seed, N_SMOKE_EPOCHS), flush=True)
        r = train_85(arm_85, seed, store, verbs, N_SMOKE_EPOCHS, fm, save=False)
        arms_results[arm_85] = {
            "train_acc": r["train_acc"],
            "val_acc": r["val_acc"],
            "gate_met": r["gate_met"],
            "wall_sec": r["wall_sec"],
            "n_train": r["n_train"],
            "n_val": r["n_val"],
            "n_test": r["n_test"],
            "max_startup_drift": r["unitarity_drift"]["startup_max_drift"],
            "redraw_attempts_used": r.get("redraw_attempts_used"),
        }
        rd = r.get("redraw_attempts_used")
        print("[smoke] %-20s s%d  train=%.3f val=%.3f  drift=%.2e  redraw=%s  (%.0fs)"
              % (arm_85, seed, r["train_acc"], r["val_acc"],
                 r["unitarity_drift"]["startup_max_drift"],
                 rd if rd is not None else "n/a",
                 r["wall_sec"]), flush=True)

    wall_times = [v["wall_sec"] for v in arms_results.values()]
    avg_5ep = sum(wall_times) / len(wall_times)
    est_per_unit_sec = avg_5ep / N_SMOKE_EPOCHS * R73.MAX_EPOCHS
    est_per_unit_min = est_per_unit_sec / 60.0
    est_wall_clock_5workers_min = (len(job_list_85()) / 5) * est_per_unit_min

    result = {
        "gate": "GATE_SMOKE",
        "spec": SPEC,
        "passed": True,
        "assertions_passed": True,
        "seed": seed,
        "n_smoke_epochs": N_SMOKE_EPOCHS,
        "max_epochs_production": R73.MAX_EPOCHS,
        "T": 3,
        "dataset_sha256": meta["dataset_sha256"],
        "arms": arms_results,
        "dtf_nan_finite_check": dtf_nan_msg,
        "est_per_unit_sec": est_per_unit_sec,
        "est_per_unit_min": est_per_unit_min,
        "est_wall_clock_5workers_min": est_wall_clock_5workers_min,
        "n_total_units": len(job_list_85()),
        "note": ("smoke gate: %d epochs all 3 arms at T=3; finite gradients "
                 "asserted; QR-unitarization + drift check verified"
                 % N_SMOKE_EPOCHS),
    }

    json.dump(result, open(SMOKE_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    passed_result = {
        "assertions_passed": True,
        "smoke_file": SMOKE_FILE,
        "spec": SPEC,
        "seed": seed,
        "n_smoke_epochs": N_SMOKE_EPOCHS,
    }
    json.dump(passed_result, open(SMOKE_FILE_PASSED, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n[GATE_SMOKE] PASSED -- wrote %s and %s" % (SMOKE_FILE, SMOKE_FILE_PASSED),
          flush=True)
    print("[smoke] est per-unit runtime: %.1fs (%.1f min)  "
          "5-worker wall-clock estimate: %.1f min"
          % (est_per_unit_sec, est_per_unit_min, est_wall_clock_5workers_min),
          flush=True)
    return result


# ── Worker (production) ───────────────────────────────────────────────────────
def run_worker(worker, nworkers, max_epochs=None):
    check_gates()
    print("[exp85 w%d] all gates OK" % worker, flush=True)

    os.makedirs(EXP85_CKPT, exist_ok=True)
    items_all, meta, _ = load_exp85()
    verbs = sorted(meta["verb_inventory"].keys())
    fm = R73.feature_maps()
    store = (items_all, meta, EXP85_DATA)

    epochs = max_epochs if max_epochs is not None else R73.MAX_EPOCHS
    jobs = job_list_85()
    mine = [(arm, s) for i, (arm, s) in enumerate(jobs) if i % nworkers == worker]
    print("[exp85 w%d/%d] %d of %d runs  epochs=%d"
          % (worker, nworkers, len(mine), len(jobs), epochs), flush=True)

    for arm_85, seed in mine:
        assert seed in range(85000, 85015), \
            "FORBIDDEN: seed %d not in 85000-85014" % seed
        assert seed not in FORBIDDEN_SEEDS, \
            "FORBIDDEN: seed %d matches a prior experiment seed range" % seed

        ckj = os.path.join(EXP85_CKPT, "%s_s%d.json" % (arm_85, seed))
        if os.path.exists(ckj):
            print("[ckpt] %s s%d already done" % (arm_85, seed), flush=True)
            continue
        try:
            r = train_85(arm_85, seed, store, verbs, epochs, fm, save=True)
            json.dump(r, open(ckj, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            rd = r.get("redraw_attempts_used")
            print("[exp85] %-22s s%d  train=%.3f val=%.3f HELD-OUT=%.4f  "
                  "drift=%.2e  redraw=%s  (%.0fs)"
                  % (arm_85, seed, r["train_acc"], r["val_acc"],
                     r["heldout_test_acc"],
                     r["unitarity_drift"]["startup_max_drift"],
                     rd if rd is not None else "n/a",
                     r["wall_sec"]), flush=True)
        except Exception as exc:
            print("[FAIL] %s s%d: %r" % (arm_85, seed, exc), flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()

    print("[exp85 w%d] done." % worker, flush=True)


# ── Single-unit runner ────────────────────────────────────────────────────────
def run_single(arm_85, seed, max_epochs=None):
    assert arm_85 in ARMS, "Unknown arm: %r (valid: %r)" % (arm_85, ARMS)
    assert seed in range(85000, 85015), \
        "seed %d not in exp85 range 85000-85014" % seed
    assert seed not in FORBIDDEN_SEEDS, \
        "seed %d matches a forbidden prior-experiment range" % seed

    check_gates()
    os.makedirs(EXP85_CKPT, exist_ok=True)
    ckj = os.path.join(EXP85_CKPT, "%s_s%d.json" % (arm_85, seed))
    if os.path.exists(ckj):
        print("[ckpt] %s s%d already done; skipping" % (arm_85, seed), flush=True)
        return json.load(open(ckj, encoding="utf-8"))

    items_all, meta, _ = load_exp85()
    verbs = sorted(meta["verb_inventory"].keys())
    fm = R73.feature_maps()
    store = (items_all, meta, EXP85_DATA)
    epochs = max_epochs if max_epochs is not None else R73.MAX_EPOCHS

    r = train_85(arm_85, seed, store, verbs, epochs, fm, save=True)
    json.dump(r, open(ckj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    rd = r.get("redraw_attempts_used")
    print("[exp85] %-22s s%d  train=%.3f val=%.3f HELD-OUT=%.4f  "
          "drift=%.2e  redraw=%s  (%.0fs)"
          % (arm_85, seed, r["train_acc"], r["val_acc"],
             r["heldout_test_acc"],
             r["unitarity_drift"]["startup_max_drift"],
             rd if rd is not None else "n/a",
             r["wall_sec"]), flush=True)
    return r


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-datagen",      action="store_true")
    ap.add_argument("--gate-t3-unitarity", action="store_true")
    ap.add_argument("--gate-floor",        action="store_true")
    ap.add_argument("--gate-smoke",        action="store_true")
    ap.add_argument("--worker",    type=int, default=None)
    ap.add_argument("--nworkers",  type=int, default=5)
    ap.add_argument("--arm",       type=str, default=None,
                    help="Single-unit arm (one of: %s)" % ", ".join(ARMS))
    ap.add_argument("--seed",      type=int, default=None,
                    help="Single-unit seed (85000-85014)")
    ap.add_argument("--max-epochs", type=int, default=None)
    a = ap.parse_args()

    if a.gate_datagen:
        gate_datagen()
        return 0
    if a.gate_t3_unitarity:
        gate_t3_unitarity()
        return 0
    if a.gate_floor:
        gate_floor_nonvacuous()
        return 0
    if a.gate_smoke:
        gate_smoke()
        return 0
    if a.worker is not None:
        run_worker(a.worker, a.nworkers, max_epochs=a.max_epochs)
        return 0
    if a.arm is not None and a.seed is not None:
        run_single(a.arm, a.seed, max_epochs=a.max_epochs)
        return 0

    print("Give one of: --gate-datagen | --gate-t3-unitarity | --gate-floor | "
          "--gate-smoke | --worker i | --arm ARM --seed SEED",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
