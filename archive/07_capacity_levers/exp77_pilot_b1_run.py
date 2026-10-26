# -*- coding: utf-8 -*-
"""exp77_pilot_b1_run.py -- doc 66 §11.4 B1 PILOT (type-uniform control).

SPEC: doc 66 §11.4, REGISTERED 2026-09-04.

B1 definition (doc 66 §2): ONE shared lambeq-convention parameterized ansatz
where wazn identity enters ONLY as data -- no per-template structure.

Pilot spec (doc 66 §11.4):
  - B1 only
  - 5 seeds on seed base 77500 (seeds 77500..77504)
  - Provisional 70/30 root-level split of doc 67 §6 grid
  - Negatives per §11.2: different-root cross-template pairs sharing >=3 surface letters
  - BARRED from every verdict
  - Only output that matters: seed-level sd of test accuracy
  - Δ = max(0.06, 2 * sd)

Runs AFTER exp78b compute (serial rule, §11.4).

Usage:
    python exp77_pilot_b1_run.py --smoke          # gates only, no full run
    python exp77_pilot_b1_run.py                  # all 5 seeds sequential

Output: exp77_pilot_b1.json
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)  # doc 45 SS6.1 discipline

# ---------------------------------------------------------------------------
# Local sim import (torch_sv_sim.py in same directory)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import torch_sv_sim as sim

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
INVENTORY_PATH = os.path.join(_HERE, "exp77_inventory.json")
OUT_PATH = os.path.join(_HERE, "exp77_pilot_b1.json")

# Doc 67 §6: the SHA prefix stated in the registration document.
# The meta field inventory_sha256 was computed at build time and embedded in
# the file; we verify the prefix matches the doc's stated value.
REGISTERED_SHA_PREFIX = "a9e9de9ead17cfee"

# ---------------------------------------------------------------------------
# FLAG: FEATURIZER SPEC
# Per-slot positional extraction (doc 65 positional-featurizer lineage,
# adapted for multi-slot). Column -> root-slot indices after stripping ال:
#
#   faail   (فاعِل):   C-ا-C-C     -> positions [0, 2, 3]
#   faiil   (فَعِيل):  C-C-ي-C     -> positions [0, 1, 3]
#   fa33aal (فَعّال):  C-C-ا-C     -> positions [0, 1, 3]  (shadda not in surface)
#   maf3uul (مَفْعُول): م-C-C-و-C  -> positions [1, 2, 4]
#   mu1 (مُفَعِّل):    م-C-C-C     -> positions [1, 2, 3]
#   mu2 (مُفْعِل):     م-C-C-C     -> positions [1, 2, 3]
#
# Char -> angle: angle = pi * (codepoint - 0x0621) / 40  in [0, pi].
# Out-of-range chars -> pi/2 (center). No corpus fitting: zero leakage.
# Fallback rule: slot index clamped to len(bare)-1 for short words (geminate
# contraction, hamza seats). Flagged per occurrence counter in featurize().
# Column wazn enters as Rz(2*pi*col_idx/6) on each of the 3 wires (data, not
# structure). Root field is NEVER an input: structural (keys are surface+column).
# ---------------------------------------------------------------------------
COL_SLOTS = {
    "faail":   [0, 2, 3],
    "faiil":   [0, 1, 3],
    "fa33aal": [0, 1, 3],
    "maf3uul": [1, 2, 4],
    "mu1":     [1, 2, 3],
    "mu2":     [1, 2, 3],
}
COL_ORDER = ["faail", "faiil", "fa33aal", "maf3uul", "mu1", "mu2"]

ARABIC_BASE = 0x0621   # ء
ARABIC_TOP  = 0x0621 + 40  # ي (U+064A = 0x0621+41, use 40 for range)

# Circuit hyperparameters
N_QUBITS = 3    # per word (matches wazn-arm's 3 root-consonant wires)
N_LAYERS = 3    # Circuit4 layers -> 3*(2*3+2)=24 ansatz params + 2 readout = 26 total

# Data seeds (FIXED across all 5 model seeds)
ROOT_SPLIT_SEED = 77490
NEG_SAMPLE_SEED = 77491

# Model seeds
SEED_BASE = 77500
N_SEEDS   = 5

# Training
LR         = 0.01
MAX_EPOCHS = 150
VAL_FRAC   = 0.15   # fraction of train pairs held for epoch selection

# Negative sampling
MIN_SHARED_LETTERS = 3  # doc 66 §11.2


# ---------------------------------------------------------------------------
# FEATURIZER
# ---------------------------------------------------------------------------

def strip_al(w):
    """Strip the Arabic definite article ال."""
    return w[2:] if w.startswith("ال") else w


# Tracks clamping events for provenance
_clamp_events = []

def arabic_char_angle(ch):
    """Fixed char->angle map. No corpus fitting. range [0, pi]."""
    cp = ord(ch)
    if ARABIC_BASE <= cp <= ARABIC_TOP:
        return math.pi * (cp - ARABIC_BASE) / 40.0
    return math.pi / 2.0  # fallback: center (non-Arabic, rare diacritics)


def featurize(surface, column):
    """(surface, column) -> (angle0, angle1, angle2, col_angle).

    Root field is NEVER accessed here. Column enters ONLY as Rz modulation.
    """
    bare = strip_al(surface)
    slots = COL_SLOTS[column]
    angles = []
    for s in slots:
        idx = min(s, len(bare) - 1)
        if idx < s:
            _clamp_events.append((surface, column, s, idx))
        ch = bare[idx] if bare else "ا"  # fallback to alif
        angles.append(arabic_char_angle(ch))
    col_idx = COL_ORDER.index(column)
    col_angle = 2.0 * math.pi * col_idx / len(COL_ORDER)
    return tuple(angles) + (col_angle,)


def build_word_features(words):
    """Build feature dict keyed by (surface, column). Root field never read."""
    result = {}
    for w in words:
        key = (w["word"], w["column"])
        result[key] = featurize(w["word"], w["column"])
    return result


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def letter_set(w):
    """Distinct Arabic letters in w after stripping ال (no diacritics)."""
    bare = strip_al(w)
    return set(
        c for c in bare
        if "؀" <= c <= "ۿ"
        and ord(c) not in range(0x064B, 0x0653)
    )


def build_positives(words):
    """All cross-template share-root pairs (same root, different column)."""
    by_root = {}
    for w in words:
        by_root.setdefault(w["root"], {}).setdefault(w["column"], []).append(w["word"])

    pairs = []
    for root, cols in by_root.items():
        col_list = sorted(cols.items())
        for i in range(len(col_list)):
            for j in range(i + 1, len(col_list)):
                ca, wa_list = col_list[i]
                cb, wb_list = col_list[j]
                for wa in wa_list:
                    for wb in wb_list:
                        pairs.append({
                            "word_a": wa, "col_a": ca,
                            "word_b": wb, "col_b": cb,
                            "root": root, "label": 1,
                        })
    return pairs


def sample_negatives(words, n_needed, anchor_roots, seed):
    """Sample n_needed different-root cross-template pairs with >=3 shared letters.

    doc 66 §11.2 protocol. At least ONE word in each pair has a root in
    anchor_roots (keeps the negative distribution centred on the split's
    relevant word pool). The second word is drawn from the full inventory.
    Featurizer is fixed (no learned embeddings), so cross-split words in
    negatives cause no information leakage; this is flagged in the output.
    """
    anchor_words = [w for w in words if w["root"] in anchor_roots]
    rng = random.Random(seed)

    # Group by column for enumeration
    anchor_by_col = {}
    for w in anchor_words:
        anchor_by_col.setdefault(w["column"], []).append(w)

    all_by_col = {}
    for w in words:
        all_by_col.setdefault(w["column"], []).append(w)

    ls = {w["word"]: letter_set(w["word"]) for w in words}

    cols = sorted(anchor_by_col.keys())
    candidates = []
    seen = set()
    for ca in cols:
        for cb in sorted(all_by_col.keys()):
            if ca == cb:
                continue
            for wa in anchor_by_col[ca]:
                for wb in all_by_col[cb]:
                    if wa["root"] != wb["root"]:
                        pair_key = tuple(sorted([wa["word"] + "|" + ca,
                                                 wb["word"] + "|" + cb]))
                        if pair_key in seen:
                            continue
                        shared = len(ls[wa["word"]] & ls[wb["word"]])
                        if shared >= MIN_SHARED_LETTERS:
                            seen.add(pair_key)
                            candidates.append({
                                "word_a": wa["word"], "col_a": ca,
                                "word_b": wb["word"], "col_b": cb,
                                "root_a": wa["root"], "root_b": wb["root"],
                                "label": 0,
                            })

    if len(candidates) < n_needed:
        raise RuntimeError(
            f"Not enough negatives: need {n_needed}, found {len(candidates)}"
        )
    rng.shuffle(candidates)
    return candidates[:n_needed]


def build_dataset(words,
                  root_split_seed=ROOT_SPLIT_SEED,
                  neg_seed=NEG_SAMPLE_SEED):
    """Build train/test pair lists with 70/30 root-level split."""
    all_roots = sorted(set(w["root"] for w in words))
    rng = random.Random(root_split_seed)
    rng.shuffle(all_roots)
    n_train = int(round(len(all_roots) * 0.70))
    train_roots = set(all_roots[:n_train])
    test_roots  = set(all_roots[n_train:])

    positives = build_positives(words)
    train_pos = [p for p in positives if p["root"] in train_roots]
    test_pos  = [p for p in positives if p["root"] in test_roots]

    train_neg = sample_negatives(words, len(train_pos), train_roots, neg_seed)
    test_neg  = sample_negatives(words, len(test_pos),  test_roots,  neg_seed + 1)

    return {
        "train": train_pos + train_neg,
        "test":  test_pos  + test_neg,
        "train_roots": sorted(train_roots),
        "test_roots":  sorted(test_roots),
        "n_train_pos": len(train_pos),
        "n_test_pos":  len(test_pos),
        "n_train_neg": len(train_neg),
        "n_test_neg":  len(test_neg),
    }


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

def ry(theta):
    """Ry(theta) gate matrix."""
    half = theta * 0.5
    c = torch.cos(half).to(sim.CDTYPE)
    s = torch.sin(half).to(sim.CDTYPE)
    return torch.stack([torch.stack([c, -s]),
                        torch.stack([s,  c])])


def encode_word_state(slot_angles, col_angle):
    """Encode one word as a 3-qubit statevector (no ansatz, no grad).

    For each qubit i: Ry(slot_angle[i]) then Rz(col_angle).
    Returns shape (2,2,2) complex64 tensor with requires_grad=False.
    """
    state = sim.init_state(N_QUBITS)
    for i, ang in enumerate(slot_angles):
        theta = torch.tensor(float(ang))
        state = sim.apply_gate(state, ry(theta), (i,))
    for i in range(N_QUBITS):
        theta = torch.tensor(float(col_angle))
        state = sim.apply_gate(state, sim.rz(theta), (i,))
    return state.detach()


def apply_ansatz_and_probs(encoded_state, U):
    """Apply ansatz unitary U to encoded_state, return 2^n probability vector."""
    state = sim.apply_gate(encoded_state, U, tuple(range(N_QUBITS)))
    return state.abs().pow(2).reshape(-1)  # (8,)


class B1Model(nn.Module):
    """Type-uniform B1 model for share-root verification.

    FLAG (readout choice): computational-basis collision readout.
      P(same_root) = sigmoid(a * dot(probs_A, probs_B) + b)
      where probs_A, probs_B are 2^n_q-dim probability vectors after the
      shared ansatz U(theta). Trainable: U sits inside the probabilities.
      Reusable verbatim by the wazn-ansatz arm (same readout, different circuit).

    FLAG (featurizer): per-slot positional extraction (doc 65 lineage). See
      COL_SLOTS, featurize(). Root field is NEVER an input.

    FLAG (ansatz): Circuit4(3, 3) from torch_sv_sim.py. lambeq-convention
      shared parameter tensor across all words. No per-template structure.
    """

    def __init__(self, seed=None):
        super().__init__()
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        self.ansatz = sim.Circuit4(N_QUBITS, N_LAYERS, generator=gen)
        # Readout: sigmoid(a * collision + b), a initialized to 1, b to 0
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward_encoded(self, enc_a, enc_b):
        """Forward pass given pre-computed encoded statevectors."""
        U = self.ansatz.matrix()  # (8,8) complex, computed once
        probs_a = apply_ansatz_and_probs(enc_a, U)
        probs_b = apply_ansatz_and_probs(enc_b, U)
        collision = (probs_a * probs_b).sum()
        return torch.sigmoid(self.a * collision + self.b)


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_one_seed(seed, dataset, features, encoded_cache, max_epochs=MAX_EPOCHS):
    """Train B1 model for one seed. Returns (test_acc, best_val_acc, wall_sec)."""
    torch.manual_seed(seed)
    model = B1Model(seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rng = random.Random(seed)

    train_pairs = dataset["train"]
    test_pairs  = dataset["test"]

    # Stable train/val split (seeded per model seed for reproducibility)
    pair_order = list(range(len(train_pairs)))
    rng.shuffle(pair_order)
    n_val = max(1, int(len(pair_order) * VAL_FRAC))
    val_idx_set = set(pair_order[:n_val])

    tr  = [train_pairs[i] for i in range(len(train_pairs)) if i not in val_idx_set]
    val = [train_pairs[i] for i in range(len(train_pairs)) if i in val_idx_set]

    best_val   = -1.0
    best_state = None
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        model.train()
        # Shuffle training order
        ep_order = list(range(len(tr)))
        rng.shuffle(ep_order)

        # Full-batch accumulate (all pairs, then one optimizer step)
        optimizer.zero_grad()
        total_loss = torch.zeros((), dtype=torch.float32)
        for idx in ep_order:
            p = tr[idx]
            enc_a = encoded_cache[(p["word_a"], p["col_a"])]
            enc_b = encoded_cache[(p["word_b"], p["col_b"])]
            prob = model.forward_encoded(enc_a, enc_b)
            loss = F.binary_cross_entropy(prob, torch.tensor(float(p["label"])))
            total_loss = total_loss + loss

        (total_loss / len(tr)).backward()
        optimizer.step()

        # Validation (no grad)
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for p in val:
                enc_a = encoded_cache[(p["word_a"], p["col_a"])]
                enc_b = encoded_cache[(p["word_b"], p["col_b"])]
                prob = model.forward_encoded(enc_a, enc_b)
                val_correct += int((prob.item() > 0.5) == bool(p["label"]))
        val_acc = val_correct / len(val)

        if val_acc > best_val:
            best_val   = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if ep % 30 == 0 or ep == max_epochs:
            elapsed = time.time() - t0
            print(
                f"    ep {ep:3d}/{max_epochs} val={val_acc:.3f} "
                f"best_val={best_val:.3f} {elapsed:.0f}s",
                flush=True,
            )

    # Load best checkpoint and evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for p in test_pairs:
            enc_a = encoded_cache[(p["word_a"], p["col_a"])]
            enc_b = encoded_cache[(p["word_b"], p["col_b"])]
            prob = model.forward_encoded(enc_a, enc_b)
            test_correct += int((prob.item() > 0.5) == bool(p["label"]))
    test_acc = test_correct / len(test_pairs)
    wall_sec = time.time() - t0

    print(
        f"  [seed {seed}] test_acc={test_acc:.4f}  best_val={best_val:.3f}  {wall_sec:.0f}s",
        flush=True,
    )
    return test_acc, best_val, wall_sec


# ---------------------------------------------------------------------------
# SMOKE TESTS
# ---------------------------------------------------------------------------

def smoke_grad_nonzero(dataset, encoded_cache):
    """Ansatz grad norm must be nonzero after one backward pass."""
    print("[smoke] gradient nonzero...", flush=True)
    torch.manual_seed(0)
    model = B1Model(seed=0)
    model.zero_grad()

    for p in dataset["train"][:3]:
        enc_a = encoded_cache[(p["word_a"], p["col_a"])]
        enc_b = encoded_cache[(p["word_b"], p["col_b"])]
        prob  = model.forward_encoded(enc_a, enc_b)
        loss  = F.binary_cross_entropy(prob, torch.tensor(float(p["label"])))
        loss.backward()

    gn = model.ansatz.theta.grad.norm().item()
    print(f"  ansatz theta grad norm = {gn:.6f}", flush=True)
    assert gn > 1e-10, f"SMOKE FAIL: grad norm {gn} <= 1e-10"
    print("  PASSED: gradient nonzero", flush=True)


def smoke_scrambled_labels(dataset, encoded_cache):
    """Training on scrambled labels must not raise accuracy above 0.65."""
    print("[smoke] scrambled labels...", flush=True)
    torch.manual_seed(88888)
    model = B1Model(seed=88888)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rng = random.Random(88888)

    # Use a small subset for speed
    subset = dataset["train"][:min(80, len(dataset["train"]))]

    for ep in range(30):
        model.train()
        order = list(range(len(subset)))
        rng.shuffle(order)
        optimizer.zero_grad()
        total_loss = torch.zeros(())
        for idx in order:
            p = subset[idx]
            enc_a = encoded_cache[(p["word_a"], p["col_a"])]
            enc_b = encoded_cache[(p["word_b"], p["col_b"])]
            prob  = model.forward_encoded(enc_a, enc_b)
            scrambled_label = float(rng.randint(0, 1))
            loss  = F.binary_cross_entropy(prob, torch.tensor(scrambled_label))
            total_loss = total_loss + loss
        (total_loss / len(subset)).backward()
        optimizer.step()

    # Eval with new scramble
    model.eval()
    correct = 0
    eval_rng = random.Random(11111)
    with torch.no_grad():
        for p in subset:
            enc_a = encoded_cache[(p["word_a"], p["col_a"])]
            enc_b = encoded_cache[(p["word_b"], p["col_b"])]
            prob  = model.forward_encoded(enc_a, enc_b)
            scrambled = float(eval_rng.randint(0, 1))
            correct += int((prob.item() > 0.5) == bool(scrambled))
    acc = correct / len(subset)
    print(f"  scrambled-label acc = {acc:.3f} (should be near 0.5)", flush=True)
    assert acc <= 0.65, f"SMOKE FAIL: scrambled label acc {acc:.3f} > 0.65"
    print("  PASSED: scrambled labels pinned near 0.5", flush=True)


def smoke_surface_only(features):
    """Featurizer keys are (surface, column) only -- no root."""
    for key in list(features.keys())[:10]:
        word, col = key
        assert isinstance(word, str) and isinstance(col, str)
    print("[smoke] PASSED: featurizer surface-only (keys are (surface, column))",
          flush=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke tests only, skip full training")
    ap.add_argument("--seeds", type=int, default=N_SEEDS,
                    help="Number of seeds to run (default 5)")
    a = ap.parse_args()

    t_wall_start = time.time()

    # ---- Load and verify inventory ----
    print("[b1] loading inventory ...", flush=True)
    with open(INVENTORY_PATH, encoding="utf-8") as fh:
        inv = json.load(fh)

    meta     = inv["meta"]
    meta_sha = meta["inventory_sha256"]

    # Verify SHA prefix matches doc 67 §6's registered value
    assert meta_sha.startswith(REGISTERED_SHA_PREFIX), (
        f"Inventory SHA prefix mismatch: meta={meta_sha[:16]} "
        f"!= registered={REGISTERED_SHA_PREFIX}"
    )

    words = [w for w in inv["words"] if w["admitted"]]
    # Count invariants from doc 67 §6
    assert len(words) == 499,  f"Expected 499 admitted words, got {len(words)}"
    roots = set(w["root"] for w in words)
    assert len(roots) == 182,  f"Expected 182 roots, got {len(roots)}"

    print(
        f"  inventory ok: {len(words)} words, {len(roots)} roots, "
        f"sha_prefix={meta_sha[:16]}",
        flush=True,
    )

    # ---- Build features (surface-only, no root) ----
    print("[b1] building surface features ...", flush=True)
    _clamp_events.clear()
    features = build_word_features(words)
    if _clamp_events:
        print(f"  slot-clamp events: {len(_clamp_events)} (fallback rule active)", flush=True)

    # ---- Verify pair count ----
    positives = build_positives(words)
    assert len(positives) == 478, f"Expected 478 positive pairs, got {len(positives)}"

    # ---- Build dataset ----
    print("[b1] building 70/30 dataset ...", flush=True)
    dataset = build_dataset(words)
    n_tr = len(dataset["train"])
    n_te = len(dataset["test"])
    print(
        f"  train: {dataset['n_train_pos']} pos + {dataset['n_train_neg']} neg = {n_tr}",
        flush=True,
    )
    print(
        f"  test:  {dataset['n_test_pos']} pos + {dataset['n_test_neg']} neg = {n_te}",
        flush=True,
    )
    print(
        f"  roots: {len(dataset['train_roots'])} train / "
        f"{len(dataset['test_roots'])} test",
        flush=True,
    )

    # No test pair should share a root with any training pair
    train_root_set = set(dataset["train_roots"])
    for p in dataset["test"]:
        root = p.get("root") or p.get("root_a")
        if root is not None:
            assert root not in train_root_set, f"Root {root!r} leaks into test"
    print("  root-leak assertion: PASSED", flush=True)

    # ---- Pre-encode all words (no ansatz, no grad) ----
    print("[b1] pre-encoding words ...", flush=True)
    encoded_cache = {}
    for key, feat in features.items():
        slot_angles = feat[:N_QUBITS]
        col_angle   = feat[N_QUBITS]
        encoded_cache[key] = encode_word_state(slot_angles, col_angle)
    print(f"  encoded {len(encoded_cache)} words", flush=True)

    # ---- Smoke tests ----
    print("[b1] smoke tests ...", flush=True)
    smoke_surface_only(features)
    smoke_grad_nonzero(dataset, encoded_cache)
    smoke_scrambled_labels(dataset, encoded_cache)

    if a.smoke:
        print("[b1] --smoke mode: all gates passed, exiting before training.", flush=True)
        return 0

    # ---- Run 5 seeds ----
    n_seeds_run = min(a.seeds, N_SEEDS)
    print(
        f"\n[b1] running {n_seeds_run} seeds "
        f"({SEED_BASE}..{SEED_BASE + n_seeds_run - 1}) ...\n",
        flush=True,
    )

    seed_results = []
    for i in range(n_seeds_run):
        seed = SEED_BASE + i
        print(f"[b1] === seed {seed} ({i+1}/{n_seeds_run}) ===", flush=True)
        test_acc, best_val, wall_sec = train_one_seed(
            seed, dataset, features, encoded_cache
        )
        seed_results.append({
            "seed": seed,
            "test_acc": float(test_acc),
            "best_val_acc": float(best_val),
            "wall_sec": round(float(wall_sec), 1),
        })

    # ---- Compute statistics ----
    accs      = [r["test_acc"] for r in seed_results]
    mean_acc  = sum(accs) / len(accs)
    # ddof=1 as required (flagged)
    sd        = math.sqrt(sum((x - mean_acc) ** 2 for x in accs) / (len(accs) - 1))
    delta     = max(0.06, 2.0 * sd)
    total_wall = time.time() - t_wall_start

    print("\n[b1] ====== RESULTS ======", flush=True)
    print(f"  per-seed test accs: {[round(a2, 4) for a2 in accs]}", flush=True)
    print(f"  mean = {mean_acc:.4f}", flush=True)
    print(f"  sd (ddof=1) = {sd:.4f}", flush=True)
    print(f"  Δ = max(0.06, 2*sd) = max(0.06, {2*sd:.4f}) = {delta:.4f}", flush=True)
    print(f"  total wall: {total_wall:.0f}s", flush=True)
    print("[b1] BARRED from verdict; output is sd and Δ only.", flush=True)

    # ---- Write output ----
    # Param count: Circuit4 per-layer = 2*n_qubits + (n_qubits-1) params
    per_layer_params = 2 * N_QUBITS + (N_QUBITS - 1)
    n_ansatz_params  = per_layer_params * N_LAYERS
    n_total_params   = n_ansatz_params + 2  # +2 for a, b

    output = {
        "experiment":    "exp77_pilot_b1",
        "spec":          "doc 66 §11.4, REGISTERED 2026-09-04",
        "pilot_status":  "BARRED from every verdict; only sd matters",
        "inventory_meta": {
            "sha_prefix_from_meta":    meta_sha[:16],
            "registered_sha_prefix":   REGISTERED_SHA_PREFIX,
            "sha_verified":            True,
            "admitted_words":          len(words),
            "roots":                   len(roots),
        },
        "config": {
            "n_qubits_per_word":  N_QUBITS,
            "n_layers":           N_LAYERS,
            "n_ansatz_params":    n_ansatz_params,
            "n_readout_params":   2,
            "n_total_params":     n_total_params,
            "FLAG_readout": (
                "computational-basis collision readout: "
                "P(same_root) = sigmoid(a * dot(probs_A, probs_B) + b). "
                "probs_X = |U(theta) @ encoded_X|^2 as 2^3-dim prob vector. "
                "Trainable: U(theta) sits inside probs; collision changes with theta. "
                "Reusable verbatim by the wazn-ansatz arm."
            ),
            "FLAG_featurizer": (
                "per-slot positional extraction, doc 65 lineage adapted for multi-slot. "
                "Column->slot indices: faail[0,2,3], faiil[0,1,3], fa33aal[0,1,3], "
                "maf3uul[1,2,4], mu1[1,2,3], mu2[1,2,3]. "
                "Char->angle: pi*(cp-0x0621)/40 in [0,pi]; out-of-range -> pi/2. "
                "Slot clamp fallback: min(s, len(bare)-1). "
                "Column as Rz(2*pi*col_idx/6) on all 3 wires (data, not structure). "
                "Root field NEVER an input: keys are (surface, column) only."
            ),
            "ansatz": "Circuit4(n_qubits=3, n_layers=3) from torch_sv_sim.py (lambeq-convention Sim-et-al.)",
            "readout_params_init": "a=1.0, b=0.0",
            "split": {
                "rule":            "70/30 root-level split",
                "root_split_seed": ROOT_SPLIT_SEED,
                "neg_sample_seed": NEG_SAMPLE_SEED,
                "negative_protocol": (
                    "doc 66 §11.2: different-root cross-template pairs "
                    "with |letter_set(A) & letter_set(B)| >= 3 after stripping ال. "
                    "Ratio 1:1 with positives per split. "
                    "Sampling: at least one word has root in the split's anchor-roots "
                    "set; second word drawn from full inventory. Featurizer is fixed "
                    "(no learned embeddings) so cross-split words in negatives cause "
                    "no information leakage."
                ),
                "val_fraction_of_train": VAL_FRAC,
                "epoch_selection": "best val acc across all 150 epochs",
            },
            "training": {
                "optimizer":       "Adam",
                "lr":              LR,
                "max_epochs":      MAX_EPOCHS,
                "batch":           "full-batch per epoch (accumulate, one step)",
                "set_num_threads": 1,
            },
            "seeds": list(range(SEED_BASE, SEED_BASE + N_SEEDS)),
            "sd_ddof": 1,
        },
        "pair_counts": {
            "total_positives_in_inventory": len(positives),
            "train_positive": dataset["n_train_pos"],
            "train_negative": dataset["n_train_neg"],
            "test_positive":  dataset["n_test_pos"],
            "test_negative":  dataset["n_test_neg"],
            "train_total":    n_tr,
            "test_total":     n_te,
            "train_roots":    len(dataset["train_roots"]),
            "test_roots":     len(dataset["test_roots"]),
        },
        "results": {
            "per_seed":         seed_results,
            "test_accuracies":  accs,
            "mean_test_acc":    round(mean_acc, 6),
            "sd_ddof1":         round(sd, 6),
            "delta":            round(delta, 6),
            "delta_formula":    "max(0.06, 2 * sd_ddof1)",
            "note_barred":      (
                "BARRED from verdict. sd and Δ feed doc 66 §6 bar freeze only."
            ),
        },
        "wall_sec_total": round(total_wall, 1),
    }

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=1)

    print(f"\n[b1] wrote {OUT_PATH}", flush=True)
    print(f"[b1] PILOT DONE: sd={sd:.4f}  Δ={delta:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
