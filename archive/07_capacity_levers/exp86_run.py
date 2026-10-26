# -*- coding: utf-8 -*-
"""exp86_run.py -- doc 78: WHY IDENTITY INIT HELPS -- MECHANISM STUDY.

REGISTRATION: qnlp_private_docs/78_exp86_identity_mechanism.md (REGISTERED 2026-09-17)

Reruns exp82's two arms (identity init, uniform init) at seeds 82000-82014 with
per-epoch instrumentation of readout purity, per-verb Frobenius drift, gradient norms.

FREEZE PINS (doc 78 SS0, binding):
  - Instrumentation consumes ZERO RNG draws.
  - All instrumentation runs under torch.no_grad() on detached tensors.
  - BIT-IDENTITY CONTROL: heldout_test_acc must EXACTLY equal results_exp82.json's
    per-seed value. Any mismatch invalidates the unit (both values logged).

Gates (run in order; do NOT launch production before orchestrator verifies):
  python exp86_run.py --gate-zero-rng   # GATE_ZERO_RNG (seconds)
  python exp86_run.py --gate-smoke      # GATE_BIT_IDENTITY_SMOKE (~15 min)
  python exp86_run.py --gate-purity     # GATE_PURITY_RANGE (reads smoke unit)

Production unit (arm and seed are required):
  python exp86_run.py --arm A3-T2-identity --seed 82001
  python exp86_run.py --arm A3-T2-uniform  --seed 82000
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

import numpy as np
import torch
import torch.nn.functional as F

import exp73_run as R73
import exp82_run as R82  # imports after R73 so _ORIG_INIT captures R73.apply_init_scheme
import torch_sv_sim as sim

from exp42_run import grad_global_norm

SPEC = ("qnlp_private_docs/78_exp86_identity_mechanism.md (REGISTERED 2026-09-17)")
EXP86_CKPT = "exp86_ckpt"
RESULTS_EXP82 = "results_exp82.json"
SMOKE_FILE_EXP82 = R82.SMOKE_FILE   # results_exp73_smoke_validation.json

# The two arms exp82 ran, verbatim
ARMS = R82.ARMS           # ["A3-T2-identity", "A3-T2-uniform"]
SEEDS = R82.SEEDS         # range 82000..82014
SEED_BASE = R82.SEED_BASE # 82000
N_SEEDS = R82.N_SEEDS_82  # 15

# Gate artifact filenames
GATE_ZERO_RNG_OUT = "exp86_gate_ZERO_RNG.json"
GATE_SMOKE_OUT = "exp86_gate_BIT_IDENTITY_SMOKE.json"
GATE_PURITY_OUT = "exp86_gate_PURITY_RANGE.json"
SMOKE_UNIT_CKPT = os.path.join(EXP86_CKPT, "A3-T2-identity_s82000.json")

# Purity range for A3 (q=3): [1/2^3, 1] = [0.125, 1]
A3_Q = R73.RUNGS["A3"]["q"]  # 3
PURITY_MIN = 1.0 / (2 ** A3_Q)  # 0.125
PURITY_MAX = 1.0 + 1e-6  # small tolerance for float arithmetic


# ================================================================= reference data
def load_exp82_ref():
    """Load results_exp82.json and return a dict keyed by (arm, seed) -> record."""
    ref = json.load(open(RESULTS_EXP82, encoding="utf-8"))
    lookup = {}
    for row in ref["per_seed"]:
        s = row["seed"]
        lookup[("A3-T2-identity", s)] = {
            "heldout_test_acc": row["identity_acc"],
            "val_acc": row["identity_val_acc"],
            "train_acc": row["identity_train_acc"],
            "selected_epoch": row["identity_selected_epoch"],
        }
        lookup[("A3-T2-uniform", s)] = {
            "heldout_test_acc": row["uniform_acc"],
            "val_acc": row["uniform_val_acc"],
            "train_acc": row["uniform_train_acc"],
            "selected_epoch": row["uniform_selected_epoch"],
        }
    return lookup


# ================================================================= instrumentation
def _identity_matrix_torch(n):
    """Return n x n identity in sim.CDTYPE."""
    return torch.eye(n, dtype=sim.CDTYPE)


def _frob(A, B):
    """Frobenius norm ||A - B||_F  (works for complex tensors)."""
    return float((A - B).abs().pow(2).sum().sqrt())


def _opnorm(A, B):
    """Operator norm (max singular value) ||A - B||_2  via numpy SVD."""
    diff = (A - B).detach().cpu().numpy()
    sv = np.linalg.svd(diff, compute_uv=False)
    return float(sv[0]) if len(sv) > 0 else 0.0


def _save_block_matrices(m, verbs):
    """Return dict: verb -> detached clone of current block matrix (for distance-from-init)."""
    with torch.no_grad():
        return {v: m.verb_matrix(v).detach().clone() for v in verbs}


def _init_traces(verbs):
    """Return a fresh traces accumulator dict."""
    return {
        "epoch": [],
        "mean_readout_purity": [],
        "grad_norm_last_batch": [],
        "val_acc_per_epoch": [],
        "per_verb_frob_from_I": {v: [] for v in verbs},
        "per_verb_frob_from_init": {v: [] for v in verbs},
        "per_verb_frob_step": {v: [] for v in verbs},
        "per_verb_opnorm_from_I": {v: [] for v in verbs},
    }


def _instrument_epoch(m, ep, test_data, verbs, init_mats, prev_mats,
                      traces, gnorm, val_acc, _inject_rng=False):
    """Record one epoch of instrumentation. ZERO RNG draws (unless _inject_rng=True
    for the DTF leg of GATE_ZERO_RNG).

    Args:
        m:          the model (read-only; called under no_grad)
        ep:         epoch index (1-based)
        test_data:  list of compiled stories for the test split
        verbs:      sorted verb list
        init_mats:  dict: verb -> detached init block matrix (epoch-0 snapshot)
        prev_mats:  dict: verb -> detached block matrix from previous epoch
        traces:     the accumulator dict (mutated in place)
        gnorm:      gradient norm (last batch of this epoch, already computed by training)
        val_acc:    validation accuracy this epoch (already computed by training)
        _inject_rng: if True, consume one torch RNG draw to test GATE_ZERO_RNG's DTF
    """
    nw = 2 * m.q         # number of wires (6 for A3-T2)
    dim = 2 ** nw        # 64 for A3
    eye = _identity_matrix_torch(dim)

    with torch.no_grad():
        # --- readout-state purity: mean over test items of mean register purity
        #     at the "at_readout" stage (doc 78 SS0 pin (a))
        purity_vals = []
        for cs in test_data:
            _, tr = R73.forward_traced(m, cs)
            reg_pur = tr["at_readout"]["register"]   # list: one purity per referent
            purity_vals.append(float(sum(reg_pur) / len(reg_pur)))
        mean_purity = float(sum(purity_vals) / len(purity_vals))

        if _inject_rng:
            # DTF leg: deliberately consume one draw -- this proves the gate catches it
            _ = torch.rand(1)

        # --- per-verb Frobenius and operator norms (doc 78 SS0 pin (b))
        for v in verbs:
            W = m.verb_matrix(v).detach()
            frob_I = _frob(W, eye)
            frob_init = _frob(W, init_mats[v])
            frob_step = _frob(W, prev_mats[v])
            opnorm_I = _opnorm(W, eye)
            traces["per_verb_frob_from_I"][v].append(frob_I)
            traces["per_verb_frob_from_init"][v].append(frob_init)
            traces["per_verb_frob_step"][v].append(frob_step)
            traces["per_verb_opnorm_from_I"][v].append(opnorm_I)

    traces["mean_readout_purity"].append(mean_purity)
    traces["grad_norm_last_batch"].append(float(gnorm))
    traces["val_acc_per_epoch"].append(float(val_acc))
    traces["epoch"].append(ep)


# ================================================================= training loop
def train_instrumented(arm, seed, store, verbs, max_epochs, fm, save=True):
    """Replicates exp82's train() call path with per-epoch instrumentation hooks.

    The call order, RNG consumption, loss expression, and optimizer step are
    verbatim from R73.train() / exp82_run -- nothing about the TRAINING is changed.
    Instrumentation runs AFTER each epoch's opt.step() and val eval, under
    torch.no_grad() on detached tensors, consuming ZERO RNG draws.

    BIT-IDENTITY CONTROL: after training, heldout_test_acc is compared EXACTLY
    against results_exp82.json. Mismatch hard-fails the unit.
    """
    ref_map = load_exp82_ref()
    ref = ref_map[(arm, seed)]

    # --- setup: IDENTICAL to R73.train() ---
    items, meta, path = R73.arm_items(arm, store)
    data = R73.compile_arm(arm, items, seed, fm)
    torch.manual_seed(seed)
    m = R73.build_model(arm, verbs, seed, fm)
    init_info = R82.apply_init_scheme_82(m, arm, seed)
    opt = torch.optim.Adam(m.parameters(), lr=R73.LR)
    rng = random.Random(R73.RNG_BASE + seed)
    best = {"val": -1.0, "train": -1.0, "epoch": -1, "state": None}
    t0 = time.time()

    # --- epoch-0 snapshot: save initial block matrices BEFORE any training ---
    init_mats = _save_block_matrices(m, verbs)
    prev_mats = {v: init_mats[v].clone() for v in verbs}

    # --- traces accumulator ---
    traces = _init_traces(verbs)

    # Epoch-0 row: Frobenius distances at init (frob_from_init = 0, frob_step = 0)
    with torch.no_grad():
        nw = 2 * m.q
        dim = 2 ** nw
        eye = _identity_matrix_torch(dim)
        for v in verbs:
            W0 = init_mats[v]
            traces["per_verb_frob_from_I"][v].append(_frob(W0, eye))
            traces["per_verb_frob_from_init"][v].append(0.0)
            traces["per_verb_frob_step"][v].append(0.0)
            traces["per_verb_opnorm_from_I"][v].append(_opnorm(W0, eye))
        purity0_vals = []
        for cs in data["test"]:
            _, tr = R73.forward_traced(m, cs)
            reg_pur = tr["at_readout"]["register"]
            purity0_vals.append(float(sum(reg_pur) / len(reg_pur)))
        traces["mean_readout_purity"].append(float(sum(purity0_vals) / len(purity0_vals)))
        traces["grad_norm_last_batch"].append(float("nan"))  # no gradient at epoch 0
        traces["val_acc_per_epoch"].append(float("nan"))
        traces["epoch"].append(0)

    # --- training loop: verbatim from R73.train() ---
    last_gnorm = float("nan")
    for ep in range(1, max_epochs + 1):
        order = list(range(len(data["train"])))
        rng.shuffle(order)
        for s in range(0, len(order), R73.BATCH):
            opt.zero_grad()
            bl = 0.0
            for i in order[s:s + R73.BATCH]:
                cs = data["train"][i]
                bl = bl + F.cross_entropy(
                    torch.log(m.story_values(cs) + 1e-12).unsqueeze(0),
                    torch.tensor([cs.answer]))
            (bl / min(R73.BATCH, len(order) - s)).backward()
            gnorm_val = grad_global_norm(m)
            assert math.isfinite(gnorm_val)
            last_gnorm = gnorm_val
            opt.step()
        with torch.no_grad():
            va = R73.evaluate(m, data["val"])
        if va > best["val"]:
            with torch.no_grad():
                best = {"val": va, "train": R73.evaluate(m, data["train"]),
                        "epoch": ep, "state": copy.deepcopy(m.state_dict())}
        if ep % 25 == 0 or ep == max_epochs:
            print("[exp86] %-28s s%d ep %3d/%d val=%.3f best=%.3f %.0fs"
                  % (arm, seed, ep, max_epochs, va, best["val"], time.time() - t0),
                  flush=True)

        # --- INSTRUMENTATION: zero RNG, no_grad, detached tensors ---
        cur_mats = _save_block_matrices(m, verbs)
        _instrument_epoch(m, ep, data["test"], verbs, init_mats, prev_mats,
                          traces, last_gnorm, va)
        prev_mats = cur_mats

    # --- restore best checkpoint and evaluate (same as R73.train()) ---
    m.load_state_dict(best["state"])
    te, rows = R73.evaluate_with_traces(m, data["test"])

    # --- BIT-IDENTITY CONTROL (doc 78 SS0) ---
    ref_acc = ref["heldout_test_acc"]
    bit_identical = (te == ref_acc)
    if not bit_identical:
        msg = (
            "BIT-IDENTITY CONTROL FAILED for %s seed %d: "
            "exp86 heldout_test_acc = %.15f, "
            "results_exp82.json recorded = %.15f, "
            "diff = %.3e"
            % (arm, seed, te, ref_acc, abs(te - ref_acc))
        )
        # Log secondary diagnostics for divergence reporting
        print("[FAIL-IDENTITY] selected_epoch exp86=%d ref=%d  "
              "val_acc exp86=%.6f ref=%.6f  "
              "train_acc exp86=%.6f ref=%.6f"
              % (best["epoch"], ref["selected_epoch"],
                 best["val"], ref["val_acc"],
                 best["train"], ref["train_acc"]),
              flush=True)
        raise RuntimeError(msg)

    pt = sha = None
    if save:
        os.makedirs(EXP86_CKPT, exist_ok=True)
        pt = os.path.join(EXP86_CKPT, "%s_s%d.pt" % (arm, seed))
        torch.save(best["state"], pt)
        sha = R73.sha256_file(pt)

    # --- build path-length summary ---
    for v in verbs:
        steps = traces["per_verb_frob_step"][v]
        # epoch-0 step is 0; path_length is sum of steps from epoch 1 onwards
        traces["per_verb_frob_step"][v] = steps  # kept per epoch (including epoch-0=0)

    result = {
        "spec": SPEC,
        "arm": arm,
        "seed": seed,
        "bit_identity_passed": True,
        "heldout_test_acc": te,
        "ref_heldout_test_acc": ref_acc,
        "selected_epoch": best["epoch"],
        "ref_selected_epoch": ref["selected_epoch"],
        "val_acc": best["val"],
        "ref_val_acc": ref["val_acc"],
        "train_acc": best["train"],
        "ref_train_acc": ref["train_acc"],
        "init_scheme": init_info,
        "n_epochs_run": max_epochs,
        "n_train": len(data["train"]),
        "n_val": len(data["val"]),
        "n_test": len(data["test"]),
        "ckpt_path": pt,
        "ckpt_sha256": sha,
        "wall_sec": time.time() - t0,
        "traces": traces,
    }
    return result


# ================================================================= gate: ZERO_RNG
def gate_zero_rng(store, verbs, fm):
    """GATE_ZERO_RNG: instrument one epoch and verify zero RNG consumption.

    REAL leg: capture torch/numpy/python global RNG states immediately before and
    after the instrumentation block for one training epoch. Assert all three states
    are IDENTICAL.

    DTF leg: same setup but with _inject_rng=True inside _instrument_epoch.
    Assert that the torch state DIFFERS (proving the gate catches a stray draw).
    """
    print("[GATE_ZERO_RNG] starting...", flush=True)
    arm = "A3-T2-identity"
    seed = 82000

    def _run_one_epoch(inject_rng):
        """Run setup + one training epoch + instrumentation; return (before, after) states."""
        items, _meta, _path = R73.arm_items(arm, store)
        data = R73.compile_arm(arm, items, seed, fm)
        torch.manual_seed(seed)
        m = R73.build_model(arm, verbs, seed, fm)
        R82.apply_init_scheme_82(m, arm, seed)
        opt = torch.optim.Adam(m.parameters(), lr=R73.LR)
        rng_loop = random.Random(R73.RNG_BASE + seed)

        init_mats = _save_block_matrices(m, verbs)
        prev_mats = {v: init_mats[v].clone() for v in verbs}
        traces = _init_traces(verbs)

        # Run one epoch of training
        order = list(range(len(data["train"])))
        rng_loop.shuffle(order)
        last_gnorm = 0.0
        for s in range(0, len(order), R73.BATCH):
            opt.zero_grad()
            bl = 0.0
            for i in order[s:s + R73.BATCH]:
                cs = data["train"][i]
                bl = bl + F.cross_entropy(
                    torch.log(m.story_values(cs) + 1e-12).unsqueeze(0),
                    torch.tensor([cs.answer]))
            (bl / min(R73.BATCH, len(order) - s)).backward()
            last_gnorm = grad_global_norm(m)
            assert math.isfinite(last_gnorm)
            opt.step()
        with torch.no_grad():
            va = R73.evaluate(m, data["val"])

        # --- capture RNG states BEFORE instrumentation ---
        state_torch_before = torch.get_rng_state().clone()
        state_numpy_before = np.random.get_state()
        state_python_before = random.getstate()

        cur_mats = _save_block_matrices(m, verbs)
        _instrument_epoch(m, 1, data["test"], verbs, init_mats, prev_mats,
                          traces, last_gnorm, va, _inject_rng=inject_rng)

        # --- capture RNG states AFTER instrumentation ---
        state_torch_after = torch.get_rng_state().clone()
        state_numpy_after = np.random.get_state()
        state_python_after = random.getstate()

        torch_match = bool(torch.equal(state_torch_before, state_torch_after))
        numpy_match = bool(np.array_equal(state_numpy_before[1], state_numpy_after[1]))
        python_match = bool(state_python_before == state_python_after)
        return torch_match, numpy_match, python_match

    # REAL leg
    t_r, n_r, p_r = _run_one_epoch(inject_rng=False)
    real_passed = t_r and n_r and p_r
    print("[GATE_ZERO_RNG] REAL leg: torch=%s numpy=%s python=%s -> %s"
          % (t_r, n_r, p_r, "PASSED" if real_passed else "FAILED"), flush=True)

    # DTF leg (inject one torch.rand(1))
    t_d, n_d, p_d = _run_one_epoch(inject_rng=True)
    dtf_caught = not t_d  # torch state MUST differ
    print("[GATE_ZERO_RNG] DTF  leg: torch=%s (expected False) numpy=%s python=%s -> caught=%s"
          % (t_d, n_d, p_d, dtf_caught), flush=True)

    overall = real_passed and dtf_caught
    result = {
        "gate": "GATE_ZERO_RNG",
        "passed": overall,
        "arm": arm,
        "seed": seed,
        "real_leg": {
            "torch_states_match": t_r,
            "numpy_states_match": n_r,
            "python_states_match": p_r,
            "passed": real_passed,
        },
        "dtf_leg": {
            "description": ("deliberately called torch.rand(1) inside _instrument_epoch; "
                            "torch RNG state must differ after -- proving the gate catches "
                            "a stray draw"),
            "torch_states_match": t_d,
            "numpy_states_match": n_d,
            "python_states_match": p_d,
            "caught": dtf_caught,
            "passed": dtf_caught,
        },
        "pin_doc78_SS0": ("instrumentation under torch.no_grad() on detached copies "
                          "consumes ZERO RNG draws; all three RNG states (torch global, "
                          "numpy global, python global) are identical before and after "
                          "one full instrumented epoch on the real path"),
    }
    print("[GATE_ZERO_RNG] -> %s" % ("PASSED" if overall else "FAILED"), flush=True)
    return result


# ========================================================= gate: BIT_IDENTITY_SMOKE
def gate_bit_identity_smoke(store, verbs, fm):
    """GATE_BIT_IDENTITY_SMOKE: train one full unit (identity arm, seed 82000) and
    verify that heldout_test_acc EXACTLY equals results_exp82.json's recorded value.

    This doubles as the smoke gate: the checkpoint is saved to exp86_ckpt/ and
    reused by GATE_PURITY_RANGE.

    DTF leg: compare the result against a deliberately wrong reference (0.5) and
    show the bit-identity check trips.
    """
    arm = "A3-T2-identity"
    seed = 82000
    ref_map = load_exp82_ref()
    ref_acc = ref_map[(arm, seed)]["heldout_test_acc"]

    print("[GATE_BIT_IDENTITY_SMOKE] running identity arm seed %d (~15 min)..." % seed,
          flush=True)

    if os.path.exists(SMOKE_UNIT_CKPT):
        print("[GATE_BIT_IDENTITY_SMOKE] checkpoint found: %s  (reusing)" % SMOKE_UNIT_CKPT,
              flush=True)
        r = json.load(open(SMOKE_UNIT_CKPT, encoding="utf-8"))
        te = r["heldout_test_acc"]
    else:
        t0 = time.time()
        r = train_instrumented(arm, seed, store, verbs, R73.MAX_EPOCHS, fm, save=True)
        te = r["heldout_test_acc"]
        json.dump(r, open(SMOKE_UNIT_CKPT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("[GATE_BIT_IDENTITY_SMOKE] unit complete in %.0fs" % (time.time() - t0),
              flush=True)

    exact = (te == ref_acc)
    print("[GATE_BIT_IDENTITY_SMOKE] REAL: heldout=%.15f  ref=%.15f  exact=%s"
          % (te, ref_acc, exact), flush=True)

    # DTF leg: compare against deliberately wrong reference
    wrong_ref = 0.5
    dtf_trips = (te != wrong_ref)
    print("[GATE_BIT_IDENTITY_SMOKE] DTF:  heldout=%.15f  wrong_ref=%.6f  check_trips=%s"
          % (te, wrong_ref, dtf_trips), flush=True)

    overall = exact and dtf_trips
    result = {
        "gate": "GATE_BIT_IDENTITY_SMOKE",
        "passed": overall,
        "arm": arm,
        "seed": seed,
        "heldout_test_acc_reproduced": te,
        "results_exp82_recorded": ref_acc,
        "exact_match": exact,
        "diff": abs(te - ref_acc),
        "dtf_leg": {
            "description": ("compare exp86 heldout against wrong reference 0.5; "
                            "the check must trip (te != wrong_ref)"),
            "wrong_reference": wrong_ref,
            "check_trips": dtf_trips,
            "passed": dtf_trips,
        },
        "smoke_ckpt": SMOKE_UNIT_CKPT,
        "control_doc78_SS0": ("after training, heldout_test_acc must EXACTLY equal "
                              "results_exp82.json per_seed value; any mismatch "
                              "invalidates the whole run (doc 70 style)"),
    }
    print("[GATE_BIT_IDENTITY_SMOKE] -> %s" % ("PASSED" if overall else "FAILED"),
          flush=True)
    return result


# ============================================================ gate: PURITY_RANGE
def gate_purity_range():
    """GATE_PURITY_RANGE: verify purity values in [1/2^k, 1] for the readout
    subsystem on the smoke unit's trace.

    For A3 (q=3): k=3, range = [0.125, 1.0].
    Also verifies that purity > 0 (trivially, for any quantum state).

    DTF leg: inject an unnormalized state (scale by 1.5) and show Tr(rho^2) > 1,
    i.e., outside the valid range.
    """
    print("[GATE_PURITY_RANGE] reading %s..." % SMOKE_UNIT_CKPT, flush=True)
    if not os.path.exists(SMOKE_UNIT_CKPT):
        sys.exit("GATE_PURITY_RANGE: smoke unit checkpoint missing (%s). "
                 "Run --gate-smoke first." % SMOKE_UNIT_CKPT)

    r = json.load(open(SMOKE_UNIT_CKPT, encoding="utf-8"))
    traces = r["traces"]
    purity_trace = traces["mean_readout_purity"]

    # Exclude epoch-0 (nan guard): epoch-0 is index 0 in purity_trace
    # (nan is not expected in purity, but epoch-0 has no gnorm; purity at epoch-0 IS computed)
    all_in_range = all(PURITY_MIN - 1e-6 <= p <= PURITY_MAX for p in purity_trace)
    all_positive = all(p > 0 for p in purity_trace)
    n_epochs_checked = len(purity_trace)
    pmin = min(purity_trace)
    pmax = max(purity_trace)

    print("[GATE_PURITY_RANGE] REAL: %d epochs, purity in [%.4f, %.4f], "
          "valid_range=[%.3f, 1.0], in_range=%s"
          % (n_epochs_checked, pmin, pmax, PURITY_MIN, all_in_range), flush=True)

    # DTF leg: inject an unnormalized state and compute "purity" directly
    q = A3_Q
    nw = 2 * q  # 6 wires for T=2
    dim = 2 ** nw  # 64
    psi_unit = torch.zeros((2,) * nw, dtype=sim.CDTYPE)
    psi_unit.reshape(-1)[0] = 1.0  # |0...0>
    # Scale by 1.5 -> unnormalized
    psi_bad = psi_unit * 1.5
    # Compute register purity for reg=(0,1,2), rest=(3,4,5)
    reg = list(range(q))
    rest = list(range(q, nw))
    T = torch.permute(psi_bad, reg + rest).reshape(2 ** q, -1)
    rho = T @ T.conj().transpose(0, 1)
    bad_purity = float((rho.abs() ** 2).sum())
    dtf_out_of_range = (bad_purity > 1.0 + 1e-6)
    print("[GATE_PURITY_RANGE] DTF:  purity of 1.5*|0> = %.6f, out_of_range=%s"
          % (bad_purity, dtf_out_of_range), flush=True)

    overall = all_in_range and all_positive and dtf_out_of_range
    result = {
        "gate": "GATE_PURITY_RANGE",
        "passed": overall,
        "smoke_unit_ckpt": SMOKE_UNIT_CKPT,
        "n_epochs_checked": n_epochs_checked,
        "purity_min_observed": pmin,
        "purity_max_observed": pmax,
        "valid_range": [PURITY_MIN, 1.0],
        "all_in_range": all_in_range,
        "all_positive": all_positive,
        "k": q,
        "dtf_leg": {
            "description": ("inject unnormalized state (scale |0...0> by 1.5); "
                            "Tr(rho^2) without normalisation exceeds 1.0"),
            "unnormalized_scale": 1.5,
            "bad_purity": bad_purity,
            "out_of_range": dtf_out_of_range,
            "passed": dtf_out_of_range,
        },
        "subsystem": ("per-referent register purity at the at_readout stage, "
                      "averaged over both registers and all test items. "
                      "For A3 (q=%d): purity in [1/2^%d, 1] = [%.3f, 1.0]." % (q, q, PURITY_MIN)),
    }
    print("[GATE_PURITY_RANGE] -> %s" % ("PASSED" if overall else "FAILED"), flush=True)
    return result


# ================================================================== exp82 smoke check
def check_exp82_smoke():
    """Verify exp73 smoke gate (required by exp82 and transitively by exp86)."""
    if not os.path.exists(SMOKE_FILE_EXP82):
        sys.exit("FATAL: %s missing -- exp73 smoke gate not run. "
                 "exp86 inherits exp82's smoke dependency." % SMOKE_FILE_EXP82)
    sm = json.load(open(SMOKE_FILE_EXP82, encoding="utf-8"))
    if not sm.get("assertions_passed"):
        sys.exit("FATAL: %s: assertions_passed is not True -- "
                 "exp73 smoke gate not passed, exp86 cannot proceed." % SMOKE_FILE_EXP82)
    return sm


# ============================================================= job helpers
def job_list():
    """(arm, seed) pairs in the same stable order as exp82."""
    return R82.job_list()


# ================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS,
                    help="Arm to train (required for production units)")
    ap.add_argument("--seed", type=int,
                    help="Seed in 82000..82014 (required for production units)")
    ap.add_argument("--gate-zero-rng", action="store_true",
                    help="Run GATE_ZERO_RNG (seconds)")
    ap.add_argument("--gate-smoke", action="store_true",
                    help="Run GATE_BIT_IDENTITY_SMOKE (identity, seed 82000; ~15 min)")
    ap.add_argument("--gate-purity", action="store_true",
                    help="Run GATE_PURITY_RANGE (reads smoke unit checkpoint)")
    ap.add_argument("--worker", type=int, default=None,
                    help="Worker index for exp86_launch.sh (0..4)")
    ap.add_argument("--nworkers", type=int, default=5,
                    help="Number of workers (for --worker mode)")
    a = ap.parse_args()

    # --- gates don't need full setup for purity (reads file); others do ---
    if a.gate_purity:
        rep = gate_purity_range()
        json.dump(rep, open(GATE_PURITY_OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("[GATE_PURITY_RANGE] wrote %s" % GATE_PURITY_OUT)
        return 0 if rep["passed"] else 1

    # All other modes need the data loaded
    check_exp82_smoke()
    store = R73.load_part()
    verbs = sorted(store[1]["verb_inventory"].keys())
    fm = R73.feature_maps()

    # Dataset SHA check (exp82 assertion)
    _items, meta, _path = store
    assert meta["dataset_sha256"] == R73.DATASET_SHA_DOC61, (
        "exp86 is not running on exp73's frozen Part A -- sha mismatch: got %s"
        % meta["dataset_sha256"])

    if a.gate_zero_rng:
        rep = gate_zero_rng(store, verbs, fm)
        json.dump(rep, open(GATE_ZERO_RNG_OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("[GATE_ZERO_RNG] wrote %s" % GATE_ZERO_RNG_OUT)
        return 0 if rep["passed"] else 1

    if a.gate_smoke:
        rep = gate_bit_identity_smoke(store, verbs, fm)
        json.dump(rep, open(GATE_SMOKE_OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("[GATE_BIT_IDENTITY_SMOKE] wrote %s" % GATE_SMOKE_OUT)
        return 0 if rep["passed"] else 1

    if a.worker is not None:
        # Production worker mode (used by exp86_launch.sh)
        os.makedirs(EXP86_CKPT, exist_ok=True)
        jobs = job_list()
        mine = [(arm, s) for i, (arm, s) in enumerate(jobs) if i % a.nworkers == a.worker]
        print("[exp86 w%d/%d] %d of %d units" % (a.worker, a.nworkers, len(mine), len(jobs)),
              flush=True)
        for arm, seed in mine:
            # Skip the smoke unit (identity, 82000) -- its checkpoint is kept
            if arm == "A3-T2-identity" and seed == 82000:
                ckj = os.path.join(EXP86_CKPT, "%s_s%d.json" % (arm, seed))
                if os.path.exists(ckj):
                    print("[skip-smoke] %s s%d already done" % (arm, seed), flush=True)
                    continue
            ckj = os.path.join(EXP86_CKPT, "%s_s%d.json" % (arm, seed))
            if os.path.exists(ckj):
                print("[ckpt] %s s%d already done" % (arm, seed), flush=True)
                continue
            try:
                t0 = time.time()
                r = train_instrumented(arm, seed, store, verbs, R73.MAX_EPOCHS, fm, save=True)
                json.dump(r, open(ckj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                print("[exp86 w%d] %-28s s%d  heldout=%.4f  bit_ok=%s  (%.0fs)"
                      % (a.worker, arm, seed, r["heldout_test_acc"],
                         r["bit_identity_passed"], time.time() - t0), flush=True)
            except Exception as exc:
                print("[FAIL] %s s%d: %r" % (arm, seed, exc), flush=True)
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()
        print("[exp86 w%d] done." % a.worker, flush=True)
        return 0

    if a.arm and a.seed is not None:
        # Single-unit mode
        if a.seed not in SEEDS:
            sys.exit("--seed must be in 82000..82014, got %d" % a.seed)
        ckj = os.path.join(EXP86_CKPT, "%s_s%d.json" % (a.arm, a.seed))
        if os.path.exists(ckj):
            print("[ckpt] %s s%d already done: %s" % (a.arm, a.seed, ckj))
            return 0
        os.makedirs(EXP86_CKPT, exist_ok=True)
        t0 = time.time()
        r = train_instrumented(a.arm, a.seed, store, verbs, R73.MAX_EPOCHS, fm, save=True)
        json.dump(r, open(ckj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("[exp86] %-28s s%d  heldout=%.4f  bit_ok=%s  selected_epoch=%d  (%.0fs)"
              % (a.arm, a.seed, r["heldout_test_acc"], r["bit_identity_passed"],
                 r["selected_epoch"], time.time() - t0))
        return 0

    print("Give one of: --gate-zero-rng, --gate-smoke, --gate-purity, "
          "--worker i, or --arm ARM --seed SEED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
