# -*- coding: utf-8 -*-
"""exp78b_run.py -- budget-matched scaling (doc 69).

Registration: qnlp_private_docs/69_exp78b_budget_matched_scaling.md.

THE QUESTION (doc 69 §1): does the Q76 scaling wall move when parameters
scale with the system?

Scaling rule (doc 69 §2): params(n) = (120 / wires(2)) * wires(n) = 60n.
Realized as Lv = 2n-1 verb-block layers with readout fixed at Lr = 3.
ALL exact -- rounding residual 0 at every n in {2, 3, 4, 5, 7}.

Grid (doc 69 §3): n in {2,3,4,5,7} at minimum valid T (T=2 for n<=3,
T=4 for n in {4,5}, T=6 for n=7), 5 matched floors, 5 seeds, 50 runs.

T-inclusion decision (computed at module load from measured exp76 wall-times
x param-scaling factors): total on 5 workers ~5.06h < 12h => T>=4 included.

FLAGGED AMBIGUITY F1: doc 69 §3 says "n in {2,3,4,5,7} at T=2 primary"
but n>=4 requires T>=n-1 (SS4.4.2: n<=T+1 for C3). Resolved toward minimum
valid T matching exp76's own committed cells: n=4->T=4, n=5->T=4, n=7->T=6.
Flagged per doc 63 standing rule.

FLAGGED AMBIGUITY F2: doc 69 §3 does not name band32 or I24 cells (different
axes). Resolved toward the five primary n-axis cells only.

Patches to exp76_run (all at module load time):
  1. R.build_model -> scaled_build_model (the ONE new element per doc 69 §3).
  2. R.ARM_CFG    -> exp78b's 10 arms only.
  3. R.ALL_ARMS   -> exp78b's 10 arms only.
  4. R.CKPT_DIR   -> "exp78b_ckpt".
  No dataset patches (frozen exp76 datasets: same STORY_FMT, same shas).
  No EXP76_VSUITE patch (same validation suite carries).

Drift gate: ScaledStoryModel at n=2 (Lv=3) must be byte-for-byte the
committed QuantumStoryModel("A3") -- checked statically (state_dict and
forward) and dynamically (full n2_T2 seed 0 train vs exp76 checkpoint).

Usage:
  python exp78b_run.py --smoke              # all gates + drift gate (~8 min)
  python exp78b_run.py --worker i --nworkers 5
"""

import sys, json, os, platform, time
import torch
from torch import nn

# No D76.STORY_FMT patch needed: frozen exp76 datasets, unchanged paths.
import exp76_datagen as D76
import exp76_run as R
from exp42_models import QuantumStoryModel
import exp42_sim_adapter as sim

# ====================================================================== constants
SCALING_CELLS = ["n2_T2", "n3_T2", "n4_T4", "n5_T4", "n7_T6"]
TRAIN_ARMS    = SCALING_CELLS + ["S-shuffle-" + c for c in SCALING_CELLS]  # 10 arms

CELL_N = {"n2_T2": 2, "n3_T2": 3, "n4_T4": 4, "n5_T4": 5, "n7_T6": 7}

# Scaling table (doc 69 §2): all exact, rounding residual 0 at every n.
SCALING_TABLE = {n: {
    "n": n, "target_60n": 60 * n,
    "lv": 2 * n - 1, "lr": 3,
    "total_params": 6 * 5 * (2 * n - 1) + 2 * 15,   # 6 verb blocks + 2 readout
    "rounding_residual": 0,
    "wires": n,
    "derivation": (
        "params(n) = (120/wires(2))*wires(n) = 60n; realized as "
        "Lv=2n-1 verb-block layers via Circuit4(2, 2n-1), readout "
        "fixed Lr=3 via Circuit4(2,3); exact at every n in {2,3,4,5,7}; "
        "rejected alternative: uniform L=ceil(1.5n) rounds +/-20 params "
        "at odd n in arbitrary direction -- not nearest-realizable; "
        "readout-fixed resolved: scaling readout would split the 60n budget "
        "ambiguously; fixed Lr=3 keeps the committed readout unchanged and "
        "places all scaling in the verb blocks where the composition lives"),
} for n in [2, 3, 4, 5, 7]}

# Measured exp76 mean wall-times per arm (5-seed means from exp76_ckpt/*.json)
EXP76_WALL_S = {
    "n2_T2":           490.6, "n3_T2":           495.7,
    "n4_T4":           761.6, "n5_T4":           761.6, "n7_T6":          1044.3,
    "S-shuffle-n2_T2": 507.8, "S-shuffle-n3_T2": 538.2,
    "S-shuffle-n4_T4": 875.3, "S-shuffle-n5_T4": 908.2,
    "S-shuffle-n7_T6": 1300.4,
}

# Compute table and T-inclusion decision (doc 69 §3).
def _make_compute_table():
    rows, total = {}, 0.0
    for arm, wall in EXP76_WALL_S.items():
        cell = arm[len("S-shuffle-"):] if arm.startswith("S-shuffle-") else arm
        n = CELL_N[cell]; scale = n / 2.0
        est_per_run = wall * scale
        est_5seeds  = est_per_run * 5
        rows[arm] = {"exp76_wall_mean_s": wall, "param_scale_factor": scale,
                     "exp78b_est_per_run_s": round(est_per_run, 1),
                     "exp78b_est_5seeds_s":  round(est_5seeds,  1)}
        total += est_5seeds
    return rows, total

_COMPUTE_ROWS, _SERIAL_S = _make_compute_table()
_HOURS_5W = _SERIAL_S / 5.0 / 3600.0
T_INCLUDE = _HOURS_5W < 12.0             # True -> T>=4 cells included

COMPUTE_TABLE = {
    "rows":                  _COMPUTE_ROWS,
    "total_serial_s":        round(_SERIAL_S, 0),
    "hours_on_5_workers":    round(_HOURS_5W, 2),
    "t_inclusion_decision":  "T>=4 INCLUDED" if T_INCLUDE else "T=2 ONLY",
    "decision_rule":         "doc 69 §3: include T>=4 iff total on 5 workers < 12h",
    "note": (
        "exp76 walls were measured under 5-worker contention (doc 63 SS19.4) "
        "so no additional contention factor applies. For n=7 the minimum valid "
        "T is 6 (n<=T+1), so n7_T6 is the sole n=7 cell -- not a T-axis extension."),
}

# Frozen exp76 dataset shas (doc 69 §3: same files, same shas).
DATASET_SHAS = {
    "n2_T2": "c698aa7346bf330badaf512d646284f69e3c2986bc4899906e185c33b40a0788",
    "n3_T2": "eb46f4e65a8f46c8a6fc0e0013d9dfdabc43200475b9c06b4c1488c5e2c41072",
    "n4_T4": "588e42558c1ecf72abfe0cdbf0d8ae5f65c90b6e512c1a4c74d37e5de075fe29",
    "n5_T4": "8266c30d767c034fe6427052fc16b4c37185c5ebb1db9d7d1e9c8804b09bf0e7",
    "n7_T6": "badcb738a93aac95870a8096922065aecca142364f4ee20ffb2712981eba581e",
}

CKPT_DIR   = "exp78b_ckpt"
SMOKE_OUT  = "results_exp78b_smoke.json"
ANCHOR_OUT = "results_exp78b_anchor.json"


# ============================================================= ScaledStoryModel
class ScaledStoryModel(QuantumStoryModel):
    """QuantumStoryModel A3 with verb-block depth scaled as Lv = 2n-1.

    At n=2, Lv=3 -- byte-for-byte the committed 120-param A3 model.
    Generator consumption order mirrors the committed __init__ exactly:
      - 6 verb blocks Circuit4(2, Lv) in verb order (from g)
      - q_pos  Circuit4(2, 3)          (from g, continuing)
      - q_neg  Circuit4(2, 3)          (from g, continuing)
    At n=2 the parameter tensor sequence is identical to the committed
    model -- the drift gate asserts this bit-exactly.

    story_values / verb_matrix / story_loss / predict are inherited
    unchanged from QuantumStoryModel.
    """

    def __init__(self, n_qubits, verbs, seed):
        nn.Module.__init__(self)           # bypass parent to avoid double init
        self.arm   = "A3"
        self.verbs = list(verbs)
        lv = 2 * n_qubits - 1
        g  = torch.Generator().manual_seed(seed)
        self.blocks = nn.ModuleDict(
            {v: sim.Circuit4(2, lv, generator=g) for v in self.verbs})
        self.q_pos = sim.Circuit4(2, 3, generator=g)
        self.q_neg = sim.Circuit4(2, 3, generator=g)


def scaled_build_model(arm, verbs, seed, fm):
    """Drop-in replacement for R.build_model; routes n -> ScaledStoryModel."""
    n = R.ARM_CFG[R.base_arm(arm)]["n"]
    return ScaledStoryModel(n, verbs, seed)


# ============================================================= exp78b ARM_CFG
def _build_arm_cfg():
    cfg = {}
    for cell in SCALING_CELLS:
        dc = D76.CELL_CFG[cell]
        n  = dc["n"]
        base = {"kind": "cell", "cell": cell, "n": n, "T": dc["T"],
                "inventory": dc["inv"], "category": dc["cat"], "axis": "grid",
                "enc": "FA", "init": "uniform", "n_params": 60 * n}
        cfg[cell] = dict(base,
                         role="doc 69 §3 grid cell, budget-matched scaling (60n params)")
        fa = "S-shuffle-" + cell
        cfg[fa] = dict(base,
                       role=("doc 69 §3 floor, S-shuffle procedure-matched "
                             "at scaled 60n budget"))
    return cfg


# ============================================================= module-level patches
# Patch 1: scaled builder (call-time lookup in R.train() makes this effective)
R.build_model = scaled_build_model

# Patch 2: ARM_CFG narrowed to exp78b's 10 arms (guards the unknown-arms check)
R.ARM_CFG = _build_arm_cfg()

# Patch 3: ALL_ARMS narrowed (default for --arms)
R.ALL_ARMS = list(TRAIN_ARMS)

# Patch 4: CKPT_DIR redirected (so R.train(save=True) writes to exp78b_ckpt)
R.CKPT_DIR = CKPT_DIR


# ============================================================ freshness gate
def assert_frozen_datasets():
    """Doc 69 §3: sha asserted not assumed (doc 63 SS8 lineage rule)."""
    for cell, expected in DATASET_SHAS.items():
        path = D76.STORY_FMT % cell
        try:
            blob = json.load(open(path, encoding="utf-8"))
        except FileNotFoundError:
            raise RuntimeError(
                "Frozen dataset missing: %s -- run exp76_datagen.py first." % path)
        got = blob["meta"]["dataset_sha256"]
        assert got == expected, (
            "doc 69 §3 sha mismatch at %s\n  expected: %s\n  got:      %s"
            % (path, expected, got))


# ================================================================= exp78b smoke
def smoke_exp78b(fm, verbs, store, tiny_epochs=2):
    """All exp78b gates.  Writes SMOKE_OUT with assertions_passed=True on success.

    Gate order:
      0. Dataset freshness (sha asserted)
      1. Scaling table (no rounding, all exact)
      2. Compute table + T-inclusion decision
      3. Wrong-param gate (designed to fail -- must fire at n in {3,4,5,7})
      4. Static bit-exact check (ScaledStoryModel n=2 vs committed A3)
      5. 2-epoch probe all 10 arms, assert 60n params each
      6. Drift gate: full n2_T2 seed 0 train, compare with exp76 checkpoint
    """
    t0 = time.time()
    rep = {
        "experiment": "exp78b -- budget-matched scaling (doc 69)",
        "spec":       "doc 69, registered 2026-09-03",
        "platform":   platform.platform(),
        "python":     platform.python_version(),
        "torch":      torch.__version__,
        "flagged_ambiguities": {
            "F1": ("doc 69 §3 says 'n in {2,3,4,5,7} at T=2 primary' but n>=4 "
                   "requires T>=n-1 (SS4.4.2: n<=T+1). Resolved toward minimum "
                   "valid T matching exp76's committed cells: n=4->T=4, n=5->T=4, "
                   "n=7->T=6. Flagged per doc 63 standing rule."),
            "F2": ("doc 69 §3 does not name band32 or I24 cells (different axes). "
                   "Resolved toward five primary n-axis cells only: "
                   "n2_T2, n3_T2, n4_T4, n5_T4, n7_T6."),
        },
    }

    # ---- Gate 0: dataset freshness
    assert_frozen_datasets()
    rep["dataset_shas"] = {
        cell: {"path": D76.STORY_FMT % cell, "sha256": sha}
        for cell, sha in DATASET_SHAS.items()}
    print("[smoke78b] Gate 0: dataset shas OK", flush=True)

    # ---- Gate 1: scaling table
    for n, row in SCALING_TABLE.items():
        assert row["total_params"] == row["target_60n"], (
            "scaling table error n=%d: %d != %d"
            % (n, row["total_params"], row["target_60n"]))
    rep["scaling_table"] = SCALING_TABLE
    print("[smoke78b] Gate 1: scaling table OK (all exact, zero rounding residual)",
          flush=True)

    # ---- Gate 2: compute table + T-inclusion
    rep["compute_table"] = COMPUTE_TABLE
    print("[smoke78b] Gate 2: compute table %.2fh on 5 workers -> %s"
          % (_HOURS_5W, COMPUTE_TABLE["t_inclusion_decision"]), flush=True)

    # ---- Gate 3: wrong-param gate (designed to FAIL the assertion at n>2)
    wrong_m  = QuantumStoryModel("A3", verbs, 0)     # always 120 params
    wrong_np = sum(p.numel() for p in wrong_m.parameters())
    dtf = {}
    for n in [3, 4, 5, 7]:
        expected  = 60 * n
        gate_fires = (wrong_np != expected)
        assert gate_fires, (
            "WRONG-PARAM GATE DID NOT FIRE at n=%d: committed model "
            "(%d params) == 60*%d=%d -- the 60n assertion cannot distinguish "
            "the unscaled model from the n=%d scaled model"
            % (n, wrong_np, n, expected, n))
        dtf[n] = {"wrong_model_params": wrong_np,
                  "expected_60n": expected, "gate_fired": True}
    rep["designed_to_fail_wrong_param_gate"] = dtf
    print("[smoke78b] Gate 3: wrong-param gate fired correctly at n=3,4,5,7",
          flush=True)

    # ---- Gate 4: static bit-exact check (ScaledStoryModel n=2 vs committed A3)
    m_sc = ScaledStoryModel(2, verbs, 0)
    m_og = QuantumStoryModel("A3", verbs, 0)

    sd_sc, sd_og = m_sc.state_dict(), m_og.state_dict()
    assert set(sd_sc.keys()) == set(sd_og.keys()), (
        "state_dict key mismatch: scaled=%r orig=%r"
        % (sorted(sd_sc.keys()), sorted(sd_og.keys())))
    worst_pw = max(float((sd_sc[k] - sd_og[k]).abs().max()) for k in sd_sc)
    assert worst_pw == 0.0, (
        "BIT-EXACT WEIGHT CHECK FAILED: worst_pw=%.2e -- "
        "ScaledStoryModel(n=2) does not match committed A3 model" % worst_pw)

    items_n2, _m, _p = R.arm_items("n2_T2", store)
    data_n2 = R.compile_arm("n2_T2", items_n2, 0, fm)
    cs = data_n2["test"][0]
    with torch.no_grad():
        sv_sc = m_sc.story_values(cs)
        sv_og = m_og.story_values(cs)
    worst_fw = float((sv_sc - sv_og).abs().max())
    assert worst_fw == 0.0, (
        "BIT-EXACT FORWARD CHECK FAILED: worst_fw=%.2e" % worst_fw)

    rep["static_bit_exact_n2"] = {
        "worst_param_delta": worst_pw, "worst_forward_delta": worst_fw,
        "result": "PASS",
        "assertion": ("ScaledStoryModel(n=2, Lv=3) is byte-for-byte the "
                      "committed QuantumStoryModel('A3') -- same generator "
                      "consumption order, same tensor values"),
    }
    print("[smoke78b] Gate 4: static bit-exact PASS (worst_pw=0, worst_fw=0)",
          flush=True)

    # ---- Gate 5: 2-epoch probe all 10 arms, assert 60n params each
    probe = {}
    for arm in TRAIN_ARMS:
        t1   = time.time()
        r    = R.train(arm, 0, store, verbs, tiny_epochs, fm,
                       save=False, quiet=True, certificate=False)
        dt   = time.time() - t1
        n    = CELL_N[R.base_arm(arm)]
        exp_p = 60 * n
        assert r["n_params"] == exp_p, (
            "PARAM COUNT MISMATCH at %s: got %d, expected 60*%d=%d"
            % (arm, r["n_params"], n, exp_p))
        probe[arm] = {"epochs": tiny_epochs, "sec": round(dt, 1),
                      "val": r["val_acc"], "heldout": r["heldout_test_acc"],
                      "n_params": r["n_params"], "n": r["n"], "T": r["T"]}
        print("[smoke78b] Gate 5: %-28s 2ep val=%.3f heldout=%.3f %.0fs "
              "(%dp n=%d T=%d)"
              % (arm, r["val_acc"], r["heldout_test_acc"], dt,
                 r["n_params"], r["n"], r["T"]), flush=True)
    rep["probe"] = probe
    print("[smoke78b] Gate 5: all 10 arms probed, 60n param counts verified",
          flush=True)

    # ---- Gate 6: drift gate -- full n2_T2 seed 0, compare with exp76 checkpoint
    ref_path = os.path.join("exp76_ckpt", "n2_T2_s0.json")
    if not os.path.exists(ref_path):
        raise RuntimeError(
            "exp76 drift reference missing: %s -- check exp76_ckpt/" % ref_path)
    exp76_heldout = json.load(open(ref_path, encoding="utf-8"))["heldout_test_acc"]

    ckj = os.path.join(CKPT_DIR, "n2_T2_s0.json")
    if os.path.exists(ckj):
        exp78b_heldout = json.load(open(ckj, encoding="utf-8"))["heldout_test_acc"]
        print("[smoke78b] Gate 6: reusing existing %s" % ckj, flush=True)
    else:
        print("[smoke78b] Gate 6: training n2_T2 s0 (150 epochs, ~8 min)...",
              flush=True)
        os.makedirs(CKPT_DIR, exist_ok=True)
        r_d = R.train("n2_T2", 0, store, verbs, R.MAX_EPOCHS, fm,
                      save=True, quiet=False, certificate=True)
        json.dump(r_d, open(ckj, "w", encoding="utf-8"))
        exp78b_heldout = r_d["heldout_test_acc"]

    delta     = abs(exp78b_heldout - exp76_heldout)
    gate_pass = (delta == 0.0)
    rep["drift_gate"] = {
        "exp76_n2_T2_s0_heldout":  exp76_heldout,
        "exp78b_n2_T2_s0_heldout": exp78b_heldout,
        "abs_delta": delta, "bit_exact": gate_pass,
        "verdict": "PASS" if gate_pass else "FAIL -- pipeline drift detected",
        "consequence_if_failed": (
            "The exp78b pipeline has drifted from exp76 at n=2. Every "
            "scaling comparison is against a moving baseline. Interpret "
            "nothing; hunt the build."),
    }
    if not gate_pass:
        rep["assertions_passed"] = False
        json.dump(rep, open(SMOKE_OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        raise RuntimeError(
            "DRIFT GATE FAILED: exp78b n2_T2_s0 heldout=%.6f, "
            "exp76=%.6f, delta=%.2e -- investigate before launch."
            % (exp78b_heldout, exp76_heldout, delta))
    print("[smoke78b] Gate 6: drift gate PASSED (bit-exact at n=2)", flush=True)

    rep["assertions_passed"] = True
    rep["total_smoke_wall_sec"] = round(time.time() - t0, 1)
    json.dump(rep, open(SMOKE_OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[smoke78b] ALL GATES PASSED -> %s" % SMOKE_OUT, flush=True)


# ======================================================================= main
if __name__ == "__main__":
    passthrough = list(sys.argv[1:])

    if "--smoke" in passthrough:
        # Handle smoke directly; do NOT delegate to R.main() (its smoke()
        # checks N_PARAMS=120 on every arm and would fail for n>2).
        assert_frozen_datasets()
        store = R.load_store()
        verbs = sorted(store["exp69"][1]["verb_inventory"].keys())
        assert len(verbs) == 6, "expected 6 verbs, got %d" % len(verbs)
        for cell in SCALING_CELLS:
            cv = sorted(store[cell][1]["verb_inventory"].keys())
            assert cv == verbs, "verb-inventory drift at %s" % cell
        fm = R.feature_maps()
        os.makedirs(CKPT_DIR, exist_ok=True)
        smoke_exp78b(fm, verbs, store)
        sys.exit(0)

    # Training: bake exp78b-scope redirects, delegate to R.main().
    # R.main() reads SMOKE_OUT (patched via --smoke-out) for the smoke gate,
    # uses ARM_CFG/ALL_ARMS/build_model (all patched above), and writes
    # checkpoints to CKPT_DIR (patched via --ckpt-dir).
    sys.argv = (
        [sys.argv[0],
         "--arms",       ",".join(TRAIN_ARMS),
         "--ckpt-dir",   CKPT_DIR,
         "--smoke-out",  SMOKE_OUT,
         "--anchor-out", ANCHOR_OUT]
        + passthrough
    )
    sys.exit(R.main())
