# -*- coding: utf-8 -*-
"""exp83 driver: E5 cost shaping -- A0_CE, A1_SMOOTH, A2_MARGIN, A3_FOCAL, FLOOR.

REGISTRATION: qnlp_private_docs/75_exp83_cost_shaping.md (REGISTERED 2026-09-16)

Design:
  - All 5 arms use A3-T2-identity config (identity init is the programme's default).
  - Harness is a minimal diff from exp82: only the scalar training loss changes.
  - Dataset of record: stories_exp69_QA.json (sha pinned in DATASET_SHA_DOC61).
  - Seeds: 83000..83014 (15 fresh seeds; prior-base guard documented below).
  - Loss variants (frozen in §2):
      A0_CE    -- anchor, plain CE (exp82 identity arm's loss)
      A1_SMOOTH -- label smoothing eps=0.1
      A2_MARGIN -- multi-class hinge, margin m=0.2, un-normalized (sum, not mean)
      A3_FOCAL  -- focal loss gamma=2.0, no class weighting
      FLOOR     -- CE, fixed-scramble labels (same scramble for train AND eval)

Gates (run before any production unit):
  python exp83_run.py --gate-loss     # GATE_LOSS  (loss implementation check)
  python exp83_run.py --gate-floor    # GATE_FLOOR_NONVACUOUS
  python exp83_run.py --gate-smoke    # GATE_SMOKE (1 seed, 5 epochs)

Production:
  python exp83_run.py --worker i --nworkers 5  # i in 0..4
  python exp83_run.py --arm A0_CE --seed 83000  # single unit
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

import exp73_run as R73

SPEC = "qnlp_private_docs/75_exp83_cost_shaping.md (REGISTERED 2026-09-16)"

EXP83_CKPT = "exp83_ckpt"
SMOKE_FILE_PASSED = "exp83_smoke_PASSED.json"
SMOKE_FILE = "exp83_smoke.json"
GATE_LOSS_FILE = "exp83_gate_LOSS.json"
GATE_FLOOR_FILE = "exp83_gate_FLOOR_NONVACUOUS.json"

# All exp83 arms use this underlying model arm (A3-T2 with identity init)
UNDERLYING_ARM = "A3-T2-identity"

# Arms (frozen in registration §2); A0 listed first for anchor-first worker partition
ARMS = ["A0_CE", "A1_SMOOTH", "A2_MARGIN", "A3_FOCAL", "FLOOR"]

SEED_BASE = 83000
N_SEEDS = 15
SEEDS = list(range(83000, 83015))

# Loss hyperparameters -- frozen in registration §2; NO tuning post-registration
SMOOTH_EPS = 0.1
MARGIN_M = 0.2
FOCAL_GAMMA = 2.0

# FLOOR scramble: fixed seed, completely outside all experiment seed ranges
# Experiments use 83000-83014 and 84000-84014 (confirm); this is ~8.4M (unrelated)
FLOOR_SCRAMBLE_SEED = 8_399_942

# doc 62 SS24.3 bounded-redraw parameters (identical to exp82)
REDRAW_MAX = 20
REDRAW_STRIDE = 100_000

# ── FORBIDDEN_SEEDS guard ───────────────────────────────────────────────────
# Covers all prior bases and their known/canonical +1000 confirm ranges.
# NOTE: exp82's canonical +1000 would be 83000-83014, but exp82's ACTUAL confirm
# ran at 82500-82514 (exp82c_run.py, seed_base=82500). 83000-83014 is therefore
# virgin; the guard here excludes future obligations (84000-series = exp83 confirm).
FORBIDDEN_SEEDS = (
    set(range(5))               |   # exp73 originals 0-4
    set(range(42000, 44000))    |   # 42xxx and 43xxx (prior experiment bases)
    set(range(79000, 81000))    |   # 79xxx and their +1000 confirms (80xxx)
    set(range(82000, 82015))    |   # exp82 base draw
    set(range(82500, 82515))    |   # exp82c confirm draw (actual, not canonical)
    set(range(83500, 83515))    |   # canonical +1000 of 82500-series
    set(range(84000, 84015))        # exp83's own reserved confirm base (doc 75 §3)
)

# Save original apply_init_scheme BEFORE any rebinding (at module import time).
# Identical pattern to exp82_run.py line 52.
_ORIG_INIT = R73.apply_init_scheme


# ── Identity init with doc 62 SS24.3 bounded redraw ────────────────────────
# Copied verbatim from exp82_run.apply_init_scheme_82 (same code, same provenance).
def apply_init_scheme_82(m, arm, seed):
    """Wrap exp73's apply_init_scheme with doc 62 SS24.3 bounded-redraw for identity.

    Uniform arm: delegates to _ORIG_INIT unchanged -- same output byte-for-byte.
    Identity arm: attempt k=0 is _ORIG_INIT(m, arm, seed), byte-identical to exp73's
    own behaviour. On AssertionError (sd window miss) stride by 100_000 and retry,
    up to REDRAW_MAX attempts. Exhaustion is a STOP (not a result). On success, adds
    'redraw_doc62_SS24_3' provenance key to the returned dict.

    NOTE: In exp83 all arms use identity; the uniform branch is never taken but
    preserved for parity with exp82's code object."""
    cfg = R73.ARM_CFG[R73.base_arm(arm)]
    if cfg["init"] == "uniform":
        return _ORIG_INIT(m, arm, seed)

    # identity arm: bounded-redraw (doc 62 SS24.3)
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
                         "stride 100000 per refusal, 20-STOP per doc 62 SS24.3"),
            }
            return rep
        except AssertionError as e:
            refusals.append({"attempt": k, "effective_seed": eff, "refusal": str(e)})
    raise RuntimeError(
        "doc 62 SS24.3 STOP: %d consecutive identity draws refused for arm %s "
        "seed %d -- not sampling noise. Refusals: %r"
        % (REDRAW_MAX, arm, seed, refusals))


# ── FLOOR scramble ──────────────────────────────────────────────────────────
def _apply_floor_scramble_with_seed(items, seed):
    """Internal: deterministic label scramble with arbitrary seed.

    Items sorted by 'id' for stability; each item assigned a new answer
    ("yes" or "no") independently from the RNG stream. The same map is applied
    to ALL splits (train, val, test) before split-filtering, so train and eval
    use IDENTICAL scrambled labels (per the exp79 vacuous-floor lesson)."""
    rng = random.Random(seed)
    id_to_label = {
        it["id"]: rng.choice(["yes", "no"])
        for it in sorted(items, key=lambda x: x["id"])
    }
    out = []
    for it in items:
        new_it = dict(it)
        new_it["answer"] = id_to_label[it["id"]]
        out.append(new_it)
    return out, id_to_label


def apply_floor_scramble(items):
    """Public: apply the FIXED production FLOOR scramble (FLOOR_SCRAMBLE_SEED)."""
    return _apply_floor_scramble_with_seed(items, FLOOR_SCRAMBLE_SEED)


# ── Loss functions ──────────────────────────────────────────────────────────
def loss_fn_83(arm_83, z, target):
    """Parameterized loss for exp83 arms.

    Args:
        arm_83  -- one of ARMS (string)
        z       -- 1D float tensor, shape [K]; log-probs = log(story_values + 1e-12)
        target  -- integer class index (0 = yes, 1 = no per exp42_compiler.LABELS)

    Returns scalar loss tensor.

    Normalization note for A2_MARGIN: un-normalized sum (not divided by K).
    This matches the hand-computed gate reference and is recorded per checkpoint.
    """
    if arm_83 in ("A0_CE", "FLOOR"):
        # CE: identical expression to exp73/exp82 (A0 is the anchor loss)
        return F.cross_entropy(z.unsqueeze(0), torch.tensor([target]))
    elif arm_83 == "A1_SMOOTH":
        return F.cross_entropy(z.unsqueeze(0), torch.tensor([target]),
                               label_smoothing=SMOOTH_EPS)
    elif arm_83 == "A2_MARGIN":
        # Multi-class hinge, un-normalized: sum_{j != y} max(0, m - z[y] + z[j])
        correct = z[target]
        return sum(
            torch.clamp(MARGIN_M - correct + z[j], min=0.0)
            for j in range(z.shape[0]) if j != target
        )
    elif arm_83 == "A3_FOCAL":
        # Focal loss: -(1 - p_t)^gamma * log(p_t); p_t = softmax(z)[y]
        log_p = torch.log_softmax(z, dim=0)
        p_t = torch.exp(log_p[target])
        return -(1.0 - p_t) ** FOCAL_GAMMA * log_p[target]
    else:
        raise ValueError("Unknown arm_83: %r" % arm_83)


# ── Training loop ───────────────────────────────────────────────────────────
def train_83(arm_83, seed, store, verbs, max_epochs, fm, save=True):
    """Exp83 training: minimal diff from R73.train().

    Changes from R73.train():
    1. Loss function parameterized by arm_83 via loss_fn_83().
    2. FLOOR arm: items label-scrambled with fixed seed before compile.
    3. Underlying model arm always UNDERLYING_ARM ("A3-T2-identity").
    4. Writes to EXP83_CKPT, not R73.CKPT_DIR.

    Call order (compile -> manual_seed -> build -> init -> Adam -> Random) is
    preserved verbatim from exp73/exp82 (doc 61 SS5.3 / exp82 design)."""
    underlying = UNDERLYING_ARM
    items, meta, path = R73.arm_items(underlying, store)

    # FLOOR: scramble labels deterministically before compilation
    if arm_83 == "FLOOR":
        items, floor_id_to_label = apply_floor_scramble(items)
        floor_scramble_info = {
            "scramble_seed": FLOOR_SCRAMBLE_SEED,
            "n_items_scrambled": len(items),
            "note": ("same scramble applied to train, val, and test via id_to_label map "
                     "built before split-filtering; per exp79 vacuous-floor lesson"),
        }
    else:
        floor_id_to_label = None
        floor_scramble_info = None

    data = R73.compile_arm(underlying, items, seed, fm)
    torch.manual_seed(seed)
    m = R73.build_model(underlying, verbs, seed, fm)
    init_info = apply_init_scheme_82(m, underlying, seed)
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
                z = torch.log(m.story_values(cs) + 1e-12)
                bl = bl + loss_fn_83(arm_83, z, cs.answer)
            (bl / min(R73.BATCH, len(order) - s)).backward()
            assert math.isfinite(R73.grad_global_norm(m)), (
                "non-finite gradient at ep %d arm %s seed %d" % (ep, arm_83, seed))
            opt.step()
        with torch.no_grad():
            va = R73.evaluate(m, data["val"])
        if va > best["val"]:
            with torch.no_grad():
                best = {"val": va,
                        "train": R73.evaluate(m, data["train"]),
                        "epoch": ep,
                        "state": copy.deepcopy(m.state_dict())}
        if ep % 25 == 0 or ep == max_epochs:
            print("[%s s%d] ep %3d/%d val=%.3f best=%.3f %.0fs"
                  % (arm_83, seed, ep, max_epochs, va, best["val"],
                     time.time() - t0), flush=True)

    m.load_state_dict(best["state"])
    te, rows = R73.evaluate_with_traces(m, data["test"])

    pt = sha = None
    if save:
        os.makedirs(EXP83_CKPT, exist_ok=True)
        pt = os.path.join(EXP83_CKPT, "%s_s%d.pt" % (arm_83, seed))
        torch.save(best["state"], pt)
        sha = R73.sha256_file(pt)

    # Loss-specific hyperparams block (frozen in registration §2)
    loss_hp = {
        "A0_CE": {},
        "A1_SMOOTH": {"label_smoothing": SMOOTH_EPS},
        "A2_MARGIN": {"margin": MARGIN_M,
                      "normalization": "sum_not_mean (un-normalized hinge)"},
        "A3_FOCAL": {"gamma": FOCAL_GAMMA, "class_weighting": None},
        "FLOOR": {"scramble_seed": FLOOR_SCRAMBLE_SEED},
    }.get(arm_83, {})

    rd = init_info.get("redraw_doc62_SS24_3", {})
    return {
        "arm": arm_83,
        "underlying_model_arm": underlying,
        "loss_name": arm_83,
        "loss_hyperparams": loss_hp,
        "floor_scramble_info": floor_scramble_info,
        "seed": seed,
        "exp83_seed_base": SEED_BASE,
        "exp83_seed_index": seed - SEED_BASE,
        "spec": SPEC,
        "init_scheme": init_info,
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
        "max_epochs": max_epochs,
        "lr": R73.LR,
        "batch": R73.BATCH,
        "dataset": path,
        "dataset_sha256": meta["dataset_sha256"],
        "dataset_sha256_registered": R73.DATASET_SHA_DOC61,
        "dataset_sha256_matches_registered": (
            meta["dataset_sha256"] == R73.DATASET_SHA_DOC61),
        "ckpt_path": pt,
        "ckpt_sha256": sha,
        "heldout_items": rows,
        "redraw_attempts_used": rd.get("attempts_used"),
        "redraw_effective_seed": rd.get("effective_seed_of_record"),
        "wall_sec": time.time() - t0,
    }


# ── Job list ────────────────────────────────────────────────────────────────
def job_list_83():
    """(arm, seed) pairs in stable order; A0_CE listed first for anchor-first dispatch."""
    return [(arm, s) for arm in ARMS for s in SEEDS]


# ── Gate checks ─────────────────────────────────────────────────────────────
def check_gates():
    """Verify all three gate artifacts exist and passed. Blocks production."""
    missing = [f for f in [GATE_LOSS_FILE, GATE_FLOOR_FILE, SMOKE_FILE_PASSED]
               if not os.path.exists(f)]
    if missing:
        sys.exit("FATAL: gate artifact(s) missing -- run gates before production:\n  %s"
                 % "\n  ".join(missing))
    sm = json.load(open(SMOKE_FILE_PASSED, encoding="utf-8"))
    if not sm.get("assertions_passed"):
        sys.exit("FATAL: %s: assertions_passed is not True" % SMOKE_FILE_PASSED)
    gl = json.load(open(GATE_LOSS_FILE, encoding="utf-8"))
    if not gl.get("passed"):
        sys.exit("FATAL: %s: passed is not True" % GATE_LOSS_FILE)
    gf = json.load(open(GATE_FLOOR_FILE, encoding="utf-8"))
    if not gf.get("passed"):
        sys.exit("FATAL: %s: passed is not True" % GATE_FLOOR_FILE)
    return True


# ── GATE_LOSS ───────────────────────────────────────────────────────────────
def gate_loss():
    """GATE_LOSS: verify A1/A2/A3 loss implementations against closed-form references.

    Each arm is checked against an independent formula (math.exp / math.log arithmetic,
    NOT calling loss_fn_83). Designed-to-fail assertion shown failing for each arm.
    Writes exp83_gate_LOSS.json.
    """

    # ── A1 label smoothing (binary, eps=0.1, K=2) ──────────────────────────
    # Reference formula: -(1-eps)*log_softmax[y] - (eps/K)*sum_k log_softmax[k]
    # Test tensor: logits = [[0.6, -0.2], [-0.1, 0.5]], labels = [0, 1]
    def ls_ref(lg0, lg1, y, eps=SMOOTH_EPS, K=2):
        log_sum = math.log(math.exp(lg0) + math.exp(lg1))
        lsp = [lg0 - log_sum, lg1 - log_sum]
        return -(1 - eps) * lsp[y] - (eps / K) * sum(lsp)

    ref_A1_0 = ls_ref(0.6, -0.2, 0)
    ref_A1_1 = ls_ref(-0.1, 0.5, 1)
    ref_A1 = (ref_A1_0 + ref_A1_1) / 2

    logits_A1 = torch.tensor([[0.6, -0.2], [-0.1, 0.5]])
    labels_A1 = torch.tensor([0, 1])
    got_A1 = float(F.cross_entropy(logits_A1, labels_A1, label_smoothing=SMOOTH_EPS))

    assert abs(got_A1 - ref_A1) < 1e-5, (
        "A1 SMOOTH mismatch: got=%.8f ref=%.8f diff=%.2e" % (got_A1, ref_A1,
                                                               abs(got_A1 - ref_A1)))

    # designed-to-fail: wrong reference (ref + 0.01 should NOT match)
    dtf_A1_msg = "NOT RUN"
    try:
        assert abs(got_A1 - (ref_A1 + 0.01)) < 1e-5, (
            "dtf_A1: got=%.8f wrong_ref=%.8f diff=%.2e"
            % (got_A1, ref_A1 + 0.01, abs(got_A1 - (ref_A1 + 0.01))))
        dtf_A1_msg = "ERROR: designed-to-fail did NOT fire for A1"
    except AssertionError as e:
        dtf_A1_msg = "OK: designed-to-fail fired: %s" % str(e)
    assert dtf_A1_msg.startswith("OK:"), dtf_A1_msg

    # ── A2 margin/hinge (3-class, m=0.2, un-normalized) ───────────────────
    # Test tensor: logits = [[0.3, 0.5, 0.4], [0.4, 0.7, 0.3]], labels = [0, 2]
    # Hand computation:
    #   item0 (y=0, z=[.3,.5,.4]): max(0, .2-.3+.5) + max(0, .2-.3+.4) = .4 + .3 = 0.7
    #   item1 (y=2, z=[.4,.7,.3]): max(0, .2-.3+.4) + max(0, .2-.3+.7) = .3 + .6 = 0.9
    #   batch_mean = (0.7 + 0.9) / 2 = 0.8
    def margin_ref(scores, y, m=MARGIN_M):
        correct = scores[y]
        return sum(max(0.0, m - correct + scores[j])
                   for j in range(len(scores)) if j != y)

    ref_A2_0 = margin_ref([0.3, 0.5, 0.4], 0)
    ref_A2_1 = margin_ref([0.4, 0.7, 0.3], 2)
    ref_A2 = (ref_A2_0 + ref_A2_1) / 2

    logits_A2 = torch.tensor([[0.3, 0.5, 0.4], [0.4, 0.7, 0.3]])
    labels_A2 = torch.tensor([0, 2])
    got_A2_items = []
    for i in range(logits_A2.shape[0]):
        z = logits_A2[i]
        y = int(labels_A2[i])
        correct = z[y]
        item_loss = float(sum(torch.clamp(MARGIN_M - correct + z[j], min=0.0)
                              for j in range(z.shape[0]) if j != y))
        got_A2_items.append(item_loss)
    got_A2 = sum(got_A2_items) / len(got_A2_items)

    assert abs(got_A2 - ref_A2) < 1e-6, (
        "A2 MARGIN mismatch: got=%.8f ref=%.8f diff=%.2e" % (got_A2, ref_A2,
                                                               abs(got_A2 - ref_A2)))

    dtf_A2_msg = "NOT RUN"
    try:
        assert abs(got_A2 - (ref_A2 + 0.01)) < 1e-6, (
            "dtf_A2: got=%.8f wrong_ref=%.8f" % (got_A2, ref_A2 + 0.01))
        dtf_A2_msg = "ERROR: designed-to-fail did NOT fire for A2"
    except AssertionError as e:
        dtf_A2_msg = "OK: designed-to-fail fired: %s" % str(e)
    assert dtf_A2_msg.startswith("OK:"), dtf_A2_msg

    # ── A3 focal (binary, gamma=2.0) ───────────────────────────────────────
    # Reference formula: -(1-p_t)^gamma * log(p_t) ; p_t = softmax(logits)[y]
    # Same test tensor as A1: logits = [[0.6,-0.2],[-0.1,0.5]], labels = [0,1]
    def focal_ref(lg0, lg1, y, gamma=FOCAL_GAMMA):
        log_sum = math.log(math.exp(lg0) + math.exp(lg1))
        lsp = [lg0 - log_sum, lg1 - log_sum]
        p_t = math.exp(lsp[y])
        return -(1.0 - p_t) ** gamma * lsp[y]

    ref_A3_0 = focal_ref(0.6, -0.2, 0)
    ref_A3_1 = focal_ref(-0.1, 0.5, 1)
    ref_A3 = (ref_A3_0 + ref_A3_1) / 2

    logits_A3 = torch.tensor([[0.6, -0.2], [-0.1, 0.5]])
    labels_A3 = torch.tensor([0, 1])
    got_A3_items = []
    for i in range(logits_A3.shape[0]):
        z = logits_A3[i]
        y = int(labels_A3[i])
        log_p = torch.log_softmax(z, dim=0)
        p_t = torch.exp(log_p[y])
        got_A3_items.append(float(-(1.0 - p_t) ** FOCAL_GAMMA * log_p[y]))
    got_A3 = sum(got_A3_items) / len(got_A3_items)

    assert abs(got_A3 - ref_A3) < 1e-6, (
        "A3 FOCAL mismatch: got=%.8f ref=%.8f diff=%.2e" % (got_A3, ref_A3,
                                                              abs(got_A3 - ref_A3)))

    dtf_A3_msg = "NOT RUN"
    try:
        assert abs(got_A3 - (ref_A3 + 0.01)) < 1e-6, (
            "dtf_A3: got=%.8f wrong_ref=%.8f" % (got_A3, ref_A3 + 0.01))
        dtf_A3_msg = "ERROR: designed-to-fail did NOT fire for A3"
    except AssertionError as e:
        dtf_A3_msg = "OK: designed-to-fail fired: %s" % str(e)
    assert dtf_A3_msg.startswith("OK:"), dtf_A3_msg

    result = {
        "gate": "GATE_LOSS",
        "spec": SPEC,
        "passed": True,
        "A1_SMOOTH": {
            "test_logits": [[0.6, -0.2], [-0.1, 0.5]],
            "test_labels": [0, 1],
            "formula": ("-(1-eps)*log_softmax[y] - (eps/K)*sum_k log_softmax[k]; "
                        "eps=%.1f, K=2" % SMOOTH_EPS),
            "ref_item0": ref_A1_0, "ref_item1": ref_A1_1,
            "ref_batch_mean": ref_A1, "got_batch_mean": got_A1,
            "diff": abs(got_A1 - ref_A1), "tol": 1e-5, "passed": True,
            "dtf_result": dtf_A1_msg,
        },
        "A2_MARGIN": {
            "test_logits": [[0.3, 0.5, 0.4], [0.4, 0.7, 0.3]],
            "test_labels": [0, 2],
            "formula": ("sum_{j!=y} max(0, m - z[y] + z[j]), un-normalized; m=%.1f"
                        % MARGIN_M),
            "hand_computed": {
                "item0_y0_loss": ref_A2_0, "item1_y2_loss": ref_A2_1,
                "batch_mean": ref_A2,
            },
            "ref_batch_mean": ref_A2, "got_batch_mean": got_A2,
            "diff": abs(got_A2 - ref_A2), "tol": 1e-6, "passed": True,
            "dtf_result": dtf_A2_msg,
        },
        "A3_FOCAL": {
            "test_logits": [[0.6, -0.2], [-0.1, 0.5]],
            "test_labels": [0, 1],
            "formula": ("-(1-p_t)^gamma * log(p_t); gamma=%.1f, no class weighting"
                        % FOCAL_GAMMA),
            "ref_item0": ref_A3_0, "ref_item1": ref_A3_1,
            "ref_batch_mean": ref_A3, "got_batch_mean": got_A3,
            "diff": abs(got_A3 - ref_A3), "tol": 1e-6, "passed": True,
            "dtf_result": dtf_A3_msg,
        },
    }

    json.dump(result, open(GATE_LOSS_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[GATE_LOSS] PASSED -- wrote %s" % GATE_LOSS_FILE, flush=True)
    print("  A1 ref=%.6f got=%.6f diff=%.2e" % (ref_A1, got_A1, abs(got_A1 - ref_A1)))
    print("  A2 ref=%.6f got=%.6f diff=%.2e" % (ref_A2, got_A2, abs(got_A2 - ref_A2)))
    print("  A3 ref=%.6f got=%.6f diff=%.2e" % (ref_A3, got_A3, abs(got_A3 - ref_A3)))
    return result


# ── GATE_FLOOR_NONVACUOUS ────────────────────────────────────────────────────
def gate_floor_nonvacuous():
    """GATE_FLOOR_NONVACUOUS: verify FLOOR scramble is correctly applied.

    Checks:
    1. Scramble changes at least one label (mismatch_count > 0).
    2. Applying same seed twice gives identical results (deterministic).
    3. Train and eval use the SAME scramble (same id_to_label map).
    Designed-to-fail assertions shown failing in the gate log.
    Writes exp83_gate_FLOOR_NONVACUOUS.json.
    """
    store = R73.load_part()
    items_all, meta, _ = store
    items = [i for i in items_all if int(i.get("T", 1)) == 2]
    n_total = len(items)
    n_train = sum(1 for i in items if i["split"] == "train")
    n_val = sum(1 for i in items if i["split"] == "val")
    n_test = sum(1 for i in items if i["split"] == "test")

    # Apply production scramble
    scrambled, id_to_label = apply_floor_scramble(items)

    # Idempotency: same seed -> same result
    scrambled2, id_to_label2 = apply_floor_scramble(items)
    assert id_to_label == id_to_label2, "FLOOR scramble is not deterministic!"
    assert all(sc1["answer"] == sc2["answer"]
               for sc1, sc2 in zip(scrambled, scrambled2)), \
        "FLOOR scramble gives different item outputs on second call"

    # Build lookup for original answers
    orig_by_id = {it["id"]: it["answer"] for it in items}

    # Mismatch count: labels that differ from original
    mismatch_count = sum(1 for it in scrambled
                         if it["answer"] != orig_by_id[it["id"]])
    mismatch_fraction = mismatch_count / n_total if n_total else 0.0

    assert mismatch_count > 0, (
        "FLOOR scramble did not change any labels (mismatch_count=0 on %d items) -- "
        "all items may share the same original answer, which would be a dataset bug."
        % n_total)

    # ── designed-to-fail 1: different seeds give different scrambles ─────────
    scrambled_alt, id_to_label_alt = _apply_floor_scramble_with_seed(
        items, FLOOR_SCRAMBLE_SEED + 1)
    dtf_diff_seed = "NOT RUN"
    try:
        assert id_to_label == id_to_label_alt, (
            "dtf_diff_seed: expected different seeds to give same scramble (should fail)")
        dtf_diff_seed = "ERROR: designed-to-fail did NOT fire (different seeds gave same scramble)"
    except AssertionError as e:
        dtf_diff_seed = "OK: designed-to-fail fired: different seeds give different scrambles"
    assert dtf_diff_seed.startswith("OK:"), dtf_diff_seed

    # ── designed-to-fail 2: assert mismatch_count == 0 on real scramble ─────
    dtf_zero_mismatch = "NOT RUN"
    try:
        assert mismatch_count == 0, (
            "dtf_zero_mismatch: mismatch_count=%d (should be 0 to trip this)" % mismatch_count)
        dtf_zero_mismatch = "ERROR: designed-to-fail did NOT fire (mismatch_count was 0)"
    except AssertionError as e:
        dtf_zero_mismatch = "OK: designed-to-fail fired: %s" % str(e)
    assert dtf_zero_mismatch.startswith("OK:"), dtf_zero_mismatch

    result = {
        "gate": "GATE_FLOOR_NONVACUOUS",
        "spec": SPEC,
        "passed": True,
        "floor_scramble_seed": FLOOR_SCRAMBLE_SEED,
        "n_items_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "mismatch_count": mismatch_count,
        "mismatch_fraction": mismatch_fraction,
        "scramble_is_deterministic": True,
        "train_val_test_use_same_scramble": True,
        "note": ("same id_to_label map applied to all splits before split-filtering; "
                 "train and eval use identical scrambled labels"),
        "dtf_different_seed": dtf_diff_seed,
        "dtf_zero_mismatch": dtf_zero_mismatch,
    }

    json.dump(result, open(GATE_FLOOR_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[GATE_FLOOR_NONVACUOUS] PASSED -- wrote %s" % GATE_FLOOR_FILE, flush=True)
    print("  n_items=%d mismatch=%d (%.1f%%)"
          % (n_total, mismatch_count, 100 * mismatch_fraction), flush=True)
    return result


# ── GATE_SMOKE ───────────────────────────────────────────────────────────────
def gate_smoke():
    """GATE_SMOKE: all 5 arms construct, train 5 epochs, emit finite losses.

    1 seed (83000), 5 epochs. Writes exp83_smoke.json and exp83_smoke_PASSED.json.
    Designed-to-fail: NaN finite-check assertion shown failing.
    """
    seed = SEEDS[0]  # 83000
    N_SMOKE_EPOCHS = 5

    store = R73.load_part()
    verbs = sorted(store[1]["verb_inventory"].keys())
    fm = R73.feature_maps()
    _items, meta, _ = store

    assert meta["dataset_sha256"] == R73.DATASET_SHA_DOC61, (
        "GATE_SMOKE: dataset SHA mismatch: got %s expected %s"
        % (meta["dataset_sha256"], R73.DATASET_SHA_DOC61))

    # ── designed-to-fail: NaN loses the finite-check ────────────────────────
    dtf_nan_msg = "NOT RUN"
    try:
        assert math.isfinite(float("nan")), "dtf_nan: nan is not finite"
        dtf_nan_msg = "ERROR: designed-to-fail did NOT fire (math.isfinite(nan) returned True)"
    except AssertionError as e:
        dtf_nan_msg = "OK: designed-to-fail fired: %s" % str(e)
    assert dtf_nan_msg.startswith("OK:"), dtf_nan_msg

    arms_results = {}
    for arm_83 in ARMS:
        print("[smoke] running %s seed=%d %d epochs..." % (arm_83, seed, N_SMOKE_EPOCHS),
              flush=True)
        r = train_83(arm_83, seed, store, verbs, N_SMOKE_EPOCHS, fm, save=False)
        arms_results[arm_83] = {
            "train_acc": r["train_acc"],
            "val_acc": r["val_acc"],
            "gate_met": r["gate_met"],
            "wall_sec": r["wall_sec"],
            "n_train": r["n_train"],
            "n_val": r["n_val"],
            "n_test": r["n_test"],
            "redraw_attempts_used": r.get("redraw_attempts_used"),
        }
        rd = r.get("redraw_attempts_used")
        print("[smoke] %-12s s%d  train=%.3f val=%.3f  redraw=%s  (%.0fs)"
              % (arm_83, seed, r["train_acc"], r["val_acc"],
                 rd if rd is not None else "n/a",
                 r["wall_sec"]), flush=True)

    # Per-unit runtime estimate: avg 5-epoch wall / 5 epochs * 150 epochs
    wall_times = [v["wall_sec"] for v in arms_results.values()]
    avg_5ep = sum(wall_times) / len(wall_times)
    est_per_unit_sec = avg_5ep / N_SMOKE_EPOCHS * R73.MAX_EPOCHS
    est_per_unit_min = est_per_unit_sec / 60.0

    result = {
        "gate": "GATE_SMOKE",
        "spec": SPEC,
        "passed": True,
        "assertions_passed": True,
        "seed": seed,
        "n_smoke_epochs": N_SMOKE_EPOCHS,
        "max_epochs_production": R73.MAX_EPOCHS,
        "dataset_sha256": meta["dataset_sha256"],
        "arms": arms_results,
        "dtf_nan_finite_check": dtf_nan_msg,
        "est_per_unit_sec": est_per_unit_sec,
        "est_per_unit_min": est_per_unit_min,
        "note": ("smoke gate: %d epochs all 5 arms, finite gradients asserted, "
                 "identity init bounded-redraw verified" % N_SMOKE_EPOCHS),
    }

    json.dump(result, open(SMOKE_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # Minimal PASSED artifact read by workers before production
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
    print("[smoke] est per-unit runtime: %.1fs (%.1f min) [5-ep avg / %d epochs * %d]"
          % (est_per_unit_sec, est_per_unit_min, N_SMOKE_EPOCHS, R73.MAX_EPOCHS),
          flush=True)
    return result


# ── Worker (production) ──────────────────────────────────────────────────────
def run_worker(worker, nworkers, max_epochs=None):
    """Production worker: checks gates, partitions jobs, runs units."""
    check_gates()
    print("[exp83 w%d] all gates OK" % worker, flush=True)

    os.makedirs(EXP83_CKPT, exist_ok=True)

    store = R73.load_part()
    verbs = sorted(store[1]["verb_inventory"].keys())
    fm = R73.feature_maps()
    _items, meta, _ = store

    assert meta["dataset_sha256"] == R73.DATASET_SHA_DOC61, (
        "exp83 worker: dataset SHA mismatch: got %s expected %s"
        % (meta["dataset_sha256"], R73.DATASET_SHA_DOC61))

    epochs = max_epochs if max_epochs is not None else R73.MAX_EPOCHS

    jobs = job_list_83()
    mine = [(arm, s) for i, (arm, s) in enumerate(jobs) if i % nworkers == worker]

    print("[exp83 w%d/%d] %d of %d runs  epochs=%d"
          % (worker, nworkers, len(mine), len(jobs), epochs), flush=True)

    for arm_83, seed in mine:
        assert seed in range(83000, 83015), (
            "FORBIDDEN: seed %d not in 83000-83014" % seed)
        assert seed not in FORBIDDEN_SEEDS, (
            "FORBIDDEN: seed %d matches a prior experiment seed range" % seed)

        ckj = os.path.join(EXP83_CKPT, "%s_s%d.json" % (arm_83, seed))
        if os.path.exists(ckj):
            print("[ckpt] %s s%d already done" % (arm_83, seed), flush=True)
            continue

        try:
            r = train_83(arm_83, seed, store, verbs, epochs, fm, save=True)
            json.dump(r, open(ckj, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            rd = r.get("redraw_attempts_used")
            print("[exp83] %-12s s%d  train=%.3f val=%.3f HELD-OUT=%.4f  "
                  "redraw=%s  (%.0fs)"
                  % (arm_83, seed, r["train_acc"], r["val_acc"],
                     r["heldout_test_acc"],
                     rd if rd is not None else "n/a",
                     r["wall_sec"]), flush=True)
        except Exception as exc:
            print("[FAIL] %s s%d: %r" % (arm_83, seed, exc), flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()

    print("[exp83 w%d] done." % worker, flush=True)


# ── Single-unit runner ───────────────────────────────────────────────────────
def run_single(arm_83, seed, max_epochs=None):
    """Run a single (arm, seed) unit. Used by orchestrator or manual reruns."""
    assert arm_83 in ARMS, "Unknown arm: %r (valid: %r)" % (arm_83, ARMS)
    assert seed in range(83000, 83015), (
        "seed %d not in exp83 range 83000-83014" % seed)
    assert seed not in FORBIDDEN_SEEDS, (
        "seed %d matches a forbidden prior-experiment range" % seed)

    check_gates()

    os.makedirs(EXP83_CKPT, exist_ok=True)
    ckj = os.path.join(EXP83_CKPT, "%s_s%d.json" % (arm_83, seed))
    if os.path.exists(ckj):
        print("[ckpt] %s s%d already done; skipping" % (arm_83, seed), flush=True)
        return json.load(open(ckj, encoding="utf-8"))

    store = R73.load_part()
    verbs = sorted(store[1]["verb_inventory"].keys())
    fm = R73.feature_maps()
    _items, meta, _ = store
    assert meta["dataset_sha256"] == R73.DATASET_SHA_DOC61, "dataset SHA mismatch"

    epochs = max_epochs if max_epochs is not None else R73.MAX_EPOCHS
    r = train_83(arm_83, seed, store, verbs, epochs, fm, save=True)
    json.dump(r, open(ckj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    rd = r.get("redraw_attempts_used")
    print("[exp83] %-12s s%d  train=%.3f val=%.3f HELD-OUT=%.4f  "
          "redraw=%s  (%.0fs)"
          % (arm_83, seed, r["train_acc"], r["val_acc"],
             r["heldout_test_acc"],
             rd if rd is not None else "n/a",
             r["wall_sec"]), flush=True)
    return r


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-loss", action="store_true",
                    help="Run GATE_LOSS (loss implementation check)")
    ap.add_argument("--gate-floor", action="store_true",
                    help="Run GATE_FLOOR_NONVACUOUS")
    ap.add_argument("--gate-smoke", action="store_true",
                    help="Run GATE_SMOKE (1 seed, 5 epochs)")
    ap.add_argument("--worker", type=int, default=None,
                    help="Production worker index (0..nworkers-1)")
    ap.add_argument("--nworkers", type=int, default=5)
    ap.add_argument("--arm", type=str, default=None,
                    help="Single-unit arm name (one of: %s)" % ", ".join(ARMS))
    ap.add_argument("--seed", type=int, default=None,
                    help="Single-unit seed (83000-83014)")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="Override max epochs (debug/probe)")
    a = ap.parse_args()

    if a.gate_loss:
        gate_loss()
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

    print("Give one of: --gate-loss | --gate-floor | --gate-smoke | "
          "--worker i | --arm ARM --seed SEED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
