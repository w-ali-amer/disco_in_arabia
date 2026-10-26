# -*- coding: utf-8 -*-
"""exp77_run.py — exp77 full multi-arm runner.

Arms (5 seeds each, seed base 77000, per-arm offset x100):
  b1         : B1 — pilot's type-uniform model, UNCHANGED (Circuit4 shared ansatz)
  wazn       : WAZN — per-column ansatz per §2/§12.4 (CV-skeleton topology)
  wazn_shuf  : WAZN-SHUFFLED — column→family permuted by seed 777 (§12.5 control)
  floor_wazn : WAZN trained on within-wazn (col-pair stratum) label shuffle; floor
  floor_root : WAZN trained on within-root label shuffle; secondary floor

Seeds:
  b1=[77000..77004], wazn=[77100..77104], wazn_shuf=[77200..77204],
  floor_wazn=[77300..77304], floor_root=[77400..77404]
  (all disjoint from pilot seeds 77490..77504)

Splits:
  primary   : held-out root (30% roots held out; §11.3 primary boolean)
  secondary : root-seen / col-pair-unseen (within train_roots; §11.3 secondary)

Negatives: strictly within-split, ≥3 shared surface letters, balanced 1:1 (§12.3).
Primary test: cap at 139 (pilot census; assert capacity ≥ 139 before launch).
Subsampling seed 77.

MUST-REUSE (§12.4): featurize, arabic_char_angle, strip_al, COL_SLOTS, COL_ORDER,
encode_word_state, apply_ansatz_and_probs, B1Model — byte-identical imports from
exp77_pilot_b1_run.py; 'is' identity asserted in smoke battery.
smoke_scrambled_labels NOT imported (§12.2 vacuous-variant incident; corrected here).

Checkpoints: exp77_ckpt/<arm>_s<seed>_results.json (per-pair records for aggregator).
Workers: 5 nohup workers, each handles units[W::5]; logs to exp77_w<W>.log.

Usage:
  python exp77_run.py --smoke              # smoke battery only
  python exp77_run.py --worker W           # run worker W (0-4); handles 5 of 25 units
  python exp77_run.py --arm ARM --seed_idx I  # run one unit directly
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

torch.set_num_threads(1)   # doc 45 SS6.1 discipline

# ---------------------------------------------------------------------------
# Local sim import
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import torch_sv_sim as sim

# ---------------------------------------------------------------------------
# §12.4 MUST-REUSE: byte-identical imports from exp77_pilot_b1_run.py.
# Identity ('is') asserted in smoke_import_identity().
# ---------------------------------------------------------------------------
import exp77_pilot_b1_run as _pilot

featurize            = _pilot.featurize
arabic_char_angle    = _pilot.arabic_char_angle
strip_al             = _pilot.strip_al
COL_SLOTS            = _pilot.COL_SLOTS
COL_ORDER            = _pilot.COL_ORDER
encode_word_state    = _pilot.encode_word_state
apply_ansatz_and_probs = _pilot.apply_ansatz_and_probs
B1Model              = _pilot.B1Model
build_word_features  = _pilot.build_word_features
build_positives      = _pilot.build_positives
letter_set           = _pilot.letter_set
N_QUBITS             = _pilot.N_QUBITS   # 3
N_LAYERS             = _pilot.N_LAYERS   # 3
INVENTORY_PATH       = _pilot.INVENTORY_PATH
REGISTERED_SHA_PREFIX = _pilot.REGISTERED_SHA_PREFIX
MIN_SHARED_LETTERS   = _pilot.MIN_SHARED_LETTERS   # 3
LR                   = _pilot.LR                   # 0.01
MAX_EPOCHS           = _pilot.MAX_EPOCHS            # 150
VAL_FRAC             = _pilot.VAL_FRAC              # 0.15

# ---------------------------------------------------------------------------
# WAZN TOPOLOGY (§12.4, derived from CV-skeleton slot adjacency, pinned here)
#
# Slot positions per column (COL_SLOTS after strip_al):
#   faail   [0,2,3]: wire0→pos0, wire1→pos2, wire2→pos3
#             adjacent: pos2&3 (diff=1) → wires 1&2 → CRX(ctrl=2, tgt=1)
#             NOT adjacent: pos0&2 (diff=2) → no entangler between wires 0&1
#             FLAG: wire0 is never entangled for faail; circuit factorises as
#             wire0 ⊗ (wire1⊗wire2). A consequence of the rule, not a bug.
#   faiil   [0,1,3]: wire0→pos0, wire1→pos1, wire2→pos3
#             adjacent: pos0&1 (diff=1) → wires 0&1 → CRX(ctrl=1, tgt=0)
#             NOT adjacent: pos1&3 (diff=2) → no entangler wires 1&2
#   fa33aal [0,1,3]: identical slot structure to faiil → CRX(ctrl=1, tgt=0)
#   maf3uul [1,2,4]: wire0→pos1, wire1→pos2, wire2→pos4
#             adjacent: pos1&2 (diff=1) → wires 0&1 → CRX(ctrl=1, tgt=0)
#             NOT adjacent: pos2&4 (diff=2) → no entangler wires 1&2
#   mu1     [1,2,3]: wire0→pos1, wire1→pos2, wire2→pos3
#             adjacent: pos1&2 (diff=1) → wires 0&1 → CRX(ctrl=1, tgt=0)
#             adjacent: pos2&3 (diff=1) → wires 1&2 → CRX(ctrl=2, tgt=1)
#   mu2     [1,2,3]: identical slot structure to mu1 → both entanglers
#
# Three distinct topologies across 6 columns (functions may be non-injective):
#   class A: {faail}                     — CRX(2,1) only
#   class B: {faiil, fa33aal, maf3uul}   — CRX(1,0) only
#   class C: {mu1, mu2}                  — CRX(1,0) + CRX(2,1)
#
# CRX direction convention: control = higher wire index (matches Circuit4).
# ---------------------------------------------------------------------------
WAZN_TOPOLOGY = {
    "faail":   [(2, 1)],
    "faiil":   [(1, 0)],
    "fa33aal": [(1, 0)],
    "maf3uul": [(1, 0)],
    "mu1":     [(1, 0), (2, 1)],
    "mu2":     [(1, 0), (2, 1)],
}

# Per-column parameter counts (N_LAYERS layers; per layer: 2*N_QUBITS + n_entanglers)
WAZN_PARAMS_PER_LAYER = {col: 2 * N_QUBITS + len(WAZN_TOPOLOGY[col]) for col in COL_ORDER}
WAZN_PARAMS_TOTAL = {col: WAZN_PARAMS_PER_LAYER[col] * N_LAYERS for col in COL_ORDER}
WAZN_TOTAL_ANSATZ_PARAMS = sum(WAZN_PARAMS_TOTAL[col] for col in COL_ORDER)
# faail=21, faiil=21, fa33aal=21, maf3uul=21, mu1=24, mu2=24 → 132 + 2 readout = 134

# ---------------------------------------------------------------------------
# SHUFFLED-CONTROL PERMUTATION (seed 777, fixed, §12.5)
# ---------------------------------------------------------------------------
def _compute_shuf_perm():
    """Draw permutation of COL_ORDER with seed 777. Returns {col: mapped_col}."""
    rng = random.Random(777)
    perm_list = list(COL_ORDER)
    rng.shuffle(perm_list)
    return {col: perm_list[i] for i, col in enumerate(COL_ORDER)}

SHUF_PERM = _compute_shuf_perm()

# ---------------------------------------------------------------------------
# ARM / SEED CONSTANTS
# ---------------------------------------------------------------------------
ARMS = ["b1", "wazn", "wazn_shuf", "floor_wazn", "floor_root"]
ARM_OFFSET = {"b1": 0, "wazn": 1, "wazn_shuf": 2, "floor_wazn": 3, "floor_root": 4}
SEED_BASE = 77000
N_SEEDS   = 5

def run_seed(arm, seed_idx):
    return SEED_BASE + ARM_OFFSET[arm] * 100 + seed_idx

# All 25 units, ordered: iterate arms × seeds for round-robin worker assignment
UNITS = [(arm, i) for arm in ARMS for i in range(N_SEEDS)]

# ---------------------------------------------------------------------------
# DATA CONSTANTS
# ---------------------------------------------------------------------------
ROOT_SPLIT_SEED               = 77490   # REUSED from pilot (same root split)
FULL_RUN_TRAIN_NEG_SEED       = 77493   # fresh: strictly-within-split
FULL_RUN_PRIMARY_TEST_NEG_SEED = 77494  # fresh: strictly-within-split
SECONDARY_SPLIT_SEED          = 77495   # pinned; underdetermined by docs
POS_SUBSAMPLE_SEED            = 77      # §12.3
PRIMARY_TEST_NEG_CAPACITY     = 139     # pilot census (strictly-within-split)

CKPT_DIR = os.path.join(_HERE, "exp77_ckpt")

# ---------------------------------------------------------------------------
# WAZN CIRCUIT
# ---------------------------------------------------------------------------

class WaznFamilyCircuit(nn.Module):
    """Per-column parameterized ansatz family for the WAZN arm.

    Each column has its own (N_LAYERS, per_layer_params) parameter tensor.
    Topology (entangling pairs) is determined by the column's CV-skeleton
    slot adjacency (WAZN_TOPOLOGY). Words sharing a column share structure.

    matrix(column) returns the 2^N_QUBITS x 2^N_QUBITS unitary for that column.
    This is the 'A_w' family from doc 66 §2.
    """

    def __init__(self, seed=None):
        super().__init__()
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        self.params = nn.ParameterDict()
        for col in COL_ORDER:
            per_layer = WAZN_PARAMS_PER_LAYER[col]
            init = torch.rand(N_LAYERS, per_layer, generator=gen) * (2 * math.pi)
            # ParameterDict keys must be valid Python identifiers (col names are)
            self.params[col] = nn.Parameter(init)

    def matrix(self, column):
        """Compute unitary for the given column's topology."""
        n = N_QUBITS
        M = torch.eye(2 ** n, dtype=sim.CDTYPE)
        entanglers = WAZN_TOPOLOGY[column]
        p = self.params[column]
        for layer in range(N_LAYERS):
            row = p[layer]
            idx = 0
            for w in range(n):
                M = sim.mat_apply(M, sim.rx(row[idx]), (w,), n)
                idx += 1
            for w in range(n):
                M = sim.mat_apply(M, sim.rz(row[idx]), (w,), n)
                idx += 1
            for (ctrl, tgt) in entanglers:
                M = sim.mat_apply(M, sim.crx(row[idx]), (ctrl, tgt), n)
                idx += 1
        return M


class WaznModel(nn.Module):
    """WAZN arm: per-column parameter tensors and topology.

    Each word is processed by its column's ansatz (A_w).
    Collision readout identical to B1 (§12.4 MUST-REUSE):
      P(same_root) = sigmoid(a * dot(probs_A, probs_B) + b)
    """

    def __init__(self, seed=None):
        super().__init__()
        self.family = WaznFamilyCircuit(seed=seed)
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward_encoded(self, enc_a, col_a, enc_b, col_b, col_matrices=None):
        """Forward pass using per-column ansatz.

        col_matrices: optional dict {col: U} for epoch-level caching.
        """
        if col_matrices is not None:
            if col_a not in col_matrices:
                col_matrices[col_a] = self.family.matrix(col_a)
            if col_b not in col_matrices:
                col_matrices[col_b] = self.family.matrix(col_b)
            U_a = col_matrices[col_a]
            U_b = col_matrices[col_b]
        else:
            U_a = self.family.matrix(col_a)
            U_b = self.family.matrix(col_b)
        probs_a = apply_ansatz_and_probs(enc_a, U_a)
        probs_b = apply_ansatz_and_probs(enc_b, U_b)
        collision = (probs_a * probs_b).sum()
        return torch.sigmoid(self.a * collision + self.b)


class WaznShuffledModel(nn.Module):
    """WAZN-SHUFFLED arm: column→family assignment permuted by seed 777.

    Identical to WAZN in structure and param count; the permutation
    SHUF_PERM maps each column to a (randomly assigned) family column,
    decoupling morphological structure from ansatz topology.

    If WAZN-SHUFFLED beats B1, the effect is capacity, not structure
    (§12.5 invalidation clause).
    """

    def __init__(self, seed=None):
        super().__init__()
        self.family = WaznFamilyCircuit(seed=seed)
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        # SHUF_PERM is global (fixed seed 777)

    def forward_encoded(self, enc_a, col_a, enc_b, col_b, col_matrices=None):
        """Forward pass using permuted column→family mapping."""
        fam_a = SHUF_PERM[col_a]
        fam_b = SHUF_PERM[col_b]
        if col_matrices is not None:
            if fam_a not in col_matrices:
                col_matrices[fam_a] = self.family.matrix(fam_a)
            if fam_b not in col_matrices:
                col_matrices[fam_b] = self.family.matrix(fam_b)
            U_a = col_matrices[fam_a]
            U_b = col_matrices[fam_b]
        else:
            U_a = self.family.matrix(fam_a)
            U_b = self.family.matrix(fam_b)
        probs_a = apply_ansatz_and_probs(enc_a, U_a)
        probs_b = apply_ansatz_and_probs(enc_b, U_b)
        collision = (probs_a * probs_b).sum()
        return torch.sigmoid(self.a * collision + self.b)


# ---------------------------------------------------------------------------
# MODEL FACTORY AND UNIFIED FORWARD
# ---------------------------------------------------------------------------

def make_model(arm, seed):
    if arm == "b1":
        return B1Model(seed=seed)
    elif arm in ("wazn", "floor_wazn", "floor_root"):
        return WaznModel(seed=seed)
    elif arm == "wazn_shuf":
        return WaznShuffledModel(seed=seed)
    else:
        raise ValueError(f"Unknown arm: {arm!r}")


def model_forward(model, arm, enc_a, col_a, enc_b, col_b, col_matrices=None):
    """Unified forward dispatch.

    B1 ignores col_a/col_b (column info already in encoded state via Rz).
    WAZN/SHUFFLED use column to select the ansatz family.
    """
    if arm == "b1":
        # B1Model.forward_encoded(enc_a, enc_b) — column in encoded state
        # Use apply_ansatz_and_probs directly for epoch-level matrix cache
        if col_matrices is not None:
            if "b1" not in col_matrices:
                col_matrices["b1"] = model.ansatz.matrix()
            U = col_matrices["b1"]
            probs_a = apply_ansatz_and_probs(enc_a, U)
            probs_b = apply_ansatz_and_probs(enc_b, U)
            collision = (probs_a * probs_b).sum()
            return torch.sigmoid(model.a * collision + model.b)
        else:
            return model.forward_encoded(enc_a, enc_b)
    elif arm in ("wazn", "floor_wazn", "floor_root"):
        return model.forward_encoded(enc_a, col_a, enc_b, col_b, col_matrices)
    elif arm == "wazn_shuf":
        return model.forward_encoded(enc_a, col_a, enc_b, col_b, col_matrices)
    else:
        raise ValueError(f"Unknown arm: {arm!r}")


# ---------------------------------------------------------------------------
# NEGATIVE SAMPLING: STRICTLY WITHIN-SPLIT (§12.3)
# ---------------------------------------------------------------------------

def sample_negatives_strict_all(words_in_split, seed):
    """Enumerate ALL valid strictly-within-split negatives.

    Both words must be from words_in_split.
    Different-root, cross-template, ≥3 shared surface letters.
    Returns (list_of_pairs_shuffled, total_found).
    """
    rng = random.Random(seed)
    by_col = {}
    for w in words_in_split:
        by_col.setdefault(w["column"], []).append(w)

    ls = {w["word"]: letter_set(w["word"]) for w in words_in_split}
    cols = sorted(by_col.keys())
    candidates = []
    seen = set()
    for i, ca in enumerate(cols):
        for cb in cols:
            if ca == cb:
                continue
            for wa in by_col.get(ca, []):
                for wb in by_col.get(cb, []):
                    if wa["root"] == wb["root"]:
                        continue
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
    rng.shuffle(candidates)
    return candidates, len(candidates)


# ---------------------------------------------------------------------------
# SECONDARY SPLIT (root-seen / col-pair-unseen, §11.3)
# ---------------------------------------------------------------------------

def build_secondary_split(train_pos, secondary_split_seed=SECONDARY_SPLIT_SEED):
    """Split train_pos into (remaining_train_pos, secondary_test_pos).

    Holdout unit: (root, frozenset{col_a, col_b}).
    For each train root with ≥2 column pairs: randomly hold out exactly one
    column-pair triple (ALL word-pairs instantiating it move together).
    Roots with exactly 1 pair keep their pair in training (root-seen constraint).

    Seed: SECONDARY_SPLIT_SEED = 77495 (pinned; underdetermined by docs).
    Returns (train_pos_trimmed, secondary_test_pos, held_out_triples_list).
    """
    rng = random.Random(secondary_split_seed)

    # Group by (root, frozenset{col_a, col_b})
    by_triple = {}
    for p in train_pos:
        key = (p["root"], frozenset([p["col_a"], p["col_b"]]))
        by_triple.setdefault(key, []).append(p)

    # Per root: list of (frozenset) column pairs
    root_to_colpairs = {}
    for (root, colpair) in by_triple:
        root_to_colpairs.setdefault(root, []).append(colpair)

    # For each root with ≥2 col-pairs: hold out one
    holdout_keys = set()
    n_contrib_roots = 0
    for root in sorted(root_to_colpairs.keys()):  # deterministic iteration
        colpairs = root_to_colpairs[root]
        if len(colpairs) >= 2:
            n_contrib_roots += 1
            # Sort for determinism, then rng.choice
            sorted_cps = sorted(colpairs, key=lambda x: tuple(sorted(x)))
            chosen = rng.choice(sorted_cps)
            holdout_keys.add((root, chosen))

    train_pos_trimmed = []
    secondary_test_pos = []
    for p in train_pos:
        key = (p["root"], frozenset([p["col_a"], p["col_b"]]))
        if key in holdout_keys:
            secondary_test_pos.append(p)
        else:
            train_pos_trimmed.append(p)

    return train_pos_trimmed, secondary_test_pos, list(holdout_keys), n_contrib_roots


# ---------------------------------------------------------------------------
# DATASET BUILDER (full run; strictly within-split)
# ---------------------------------------------------------------------------

def build_full_dataset(words):
    """Build train, primary_test, secondary_test splits for all arms.

    Root split: same seed as pilot (77490). Strictly-within-split negatives.
    Primary test: asserts capacity ≥ 139 (pilot census), caps at 139.
    Secondary test: root-seen / col-pair-unseen; size determined by data.
    Train and secondary-test negatives: from train_roots pool, disjoint by
    slicing the same shuffled enumeration.

    Returns dict with full split info and metadata.
    """
    # -- Root split (REUSED from pilot: same 127/55 train/test roots) --
    all_roots = sorted(set(w["root"] for w in words))
    rng = random.Random(ROOT_SPLIT_SEED)
    rng.shuffle(all_roots)
    n_train = int(round(len(all_roots) * 0.70))
    train_roots = set(all_roots[:n_train])
    test_roots  = set(all_roots[n_train:])

    assert len(train_roots) == 127, f"Expected 127 train roots, got {len(train_roots)}"
    assert len(test_roots)  == 55,  f"Expected 55 test roots,  got {len(test_roots)}"

    # -- Positives --
    all_pos = build_positives(words)
    assert len(all_pos) == 478, f"Expected 478 positives, got {len(all_pos)}"

    train_roots_pos = [p for p in all_pos if p["root"] in train_roots]
    primary_test_pos_all = [p for p in all_pos if p["root"] in test_roots]

    # Root-leak sanity
    for p in primary_test_pos_all:
        assert p["root"] not in train_roots, f"Root leak: {p['root']!r}"

    # -- Secondary split --
    train_pos, secondary_test_pos_all, held_out_triples, n_contrib_roots = \
        build_secondary_split(train_roots_pos, SECONDARY_SPLIT_SEED)

    # -- Word pools --
    train_words = [w for w in words if w["root"] in train_roots]
    test_words  = [w for w in words if w["root"] in test_roots]

    # -- Primary test negatives (strictly within test_roots pool) --
    primary_test_neg_all, primary_cap = sample_negatives_strict_all(
        test_words, FULL_RUN_PRIMARY_TEST_NEG_SEED)

    assert primary_cap >= PRIMARY_TEST_NEG_CAPACITY, (
        f"Primary test strictly-within-split negative capacity {primary_cap} "
        f"< {PRIMARY_TEST_NEG_CAPACITY} (pilot census). Root split seed mismatch "
        f"or wrong pool. HALT."
    )

    # Subsample positives to negative capacity (cap at 139), seed 77
    pos_rng = random.Random(POS_SUBSAMPLE_SEED)
    primary_test_pos_shuffled = list(primary_test_pos_all)
    pos_rng.shuffle(primary_test_pos_shuffled)
    n_primary = min(len(primary_test_pos_shuffled), PRIMARY_TEST_NEG_CAPACITY)
    primary_test_pos = primary_test_pos_shuffled[:n_primary]
    primary_test_neg = primary_test_neg_all[:n_primary]
    primary_test = primary_test_pos + primary_test_neg

    # -- Train negatives (strictly within train_roots pool) --
    train_neg_all, train_neg_cap = sample_negatives_strict_all(
        train_words, FULL_RUN_TRAIN_NEG_SEED)

    # Subsample train positives if neg_cap is smaller; balanced 1:1
    n_train_use = min(len(train_pos), train_neg_cap)
    if len(train_pos) > train_neg_cap:
        tr_rng = random.Random(POS_SUBSAMPLE_SEED)
        train_pos_copy = list(train_pos)
        tr_rng.shuffle(train_pos_copy)
        train_pos = train_pos_copy[:train_neg_cap]
    train_neg = train_neg_all[:n_train_use]

    # -- Secondary test negatives: disjoint slice from train_neg_all --
    # Both words are from train_roots pool by construction; pair-disjoint
    # from train_neg by taking the next slice.
    secondary_neg_pool = train_neg_all[n_train_use:]

    # Subsample secondary test positives to negative capacity; seed 77
    sec_rng = random.Random(POS_SUBSAMPLE_SEED)
    secondary_test_pos_shuffled = list(secondary_test_pos_all)
    sec_rng.shuffle(secondary_test_pos_shuffled)
    n_secondary = min(len(secondary_test_pos_shuffled), len(secondary_neg_pool))
    secondary_test_pos = secondary_test_pos_shuffled[:n_secondary]
    secondary_test_neg = secondary_neg_pool[:n_secondary]
    secondary_test = secondary_test_pos + secondary_test_neg

    train_pairs = train_pos + train_neg

    # Pair-level disjointness check (train vs secondary neg)
    train_neg_keys = set(
        tuple(sorted([p["word_a"] + "|" + p["col_a"],
                      p["word_b"] + "|" + p["col_b"]]))
        for p in train_neg)
    secondary_neg_keys = set(
        tuple(sorted([p["word_a"] + "|" + p["col_a"],
                      p["word_b"] + "|" + p["col_b"]]))
        for p in secondary_test_neg)
    n_neg_overlap = len(train_neg_keys & secondary_neg_keys)
    if n_neg_overlap > 0:
        print(f"  [data] WARNING: {n_neg_overlap} negative pairs appear in both "
              f"train and secondary_test negatives. This should not happen with "
              f"the slice approach.", flush=True)

    return {
        "train": train_pairs,
        "primary_test": primary_test,
        "secondary_test": secondary_test,
        "train_roots": sorted(train_roots),
        "test_roots": sorted(test_roots),
        "meta": {
            "root_split_seed": ROOT_SPLIT_SEED,
            "secondary_split_seed": SECONDARY_SPLIT_SEED,
            "pos_subsample_seed": POS_SUBSAMPLE_SEED,
            "train_neg_seed": FULL_RUN_TRAIN_NEG_SEED,
            "primary_test_neg_seed": FULL_RUN_PRIMARY_TEST_NEG_SEED,
            "n_train_pos": len(train_pos),
            "n_train_neg": len(train_neg),
            "n_train_total": len(train_pairs),
            "n_primary_test_pos": len(primary_test_pos),
            "n_primary_test_neg": len(primary_test_neg),
            "n_primary_test_total": len(primary_test),
            "n_secondary_test_pos": len(secondary_test_pos),
            "n_secondary_test_neg": len(secondary_test_neg),
            "n_secondary_test_total": len(secondary_test),
            "primary_neg_capacity_found": primary_cap,
            "primary_neg_capacity_asserted": PRIMARY_TEST_NEG_CAPACITY,
            "train_neg_capacity_found": train_neg_cap,
            "secondary_neg_pool_size": len(secondary_neg_pool),
            "n_secondary_holdout_roots": n_contrib_roots,
            "n_negative_pair_overlap": n_neg_overlap,
            "note_strictly_within_split": (
                "Both words in each negative pair are from the split's root pool. "
                "Train negatives: train_roots pool. Primary test negatives: test_roots pool. "
                "Secondary test negatives: train_roots pool, disjoint from train_neg by slice."
            ),
        },
    }


# ---------------------------------------------------------------------------
# FLOOR LABEL SHUFFLE
# ---------------------------------------------------------------------------

def shuffle_labels_within_wazn(pairs, rng):
    """Shuffle labels within each (col_a, col_b) unordered stratum.

    Stratum key: frozenset{col_a, col_b} — 15 possible column pairs.
    Floor for primary boolean (transfers_across_roots_within_wazn).
    """
    strata = {}
    for i, p in enumerate(pairs):
        key = frozenset([p["col_a"], p["col_b"]])
        strata.setdefault(key, []).append(i)

    shuffled = [dict(p) for p in pairs]
    for key, indices in strata.items():
        labels = [pairs[i]["label"] for i in indices]
        rng.shuffle(labels)
        for j, idx in enumerate(indices):
            shuffled[idx]["label"] = labels[j]
    return shuffled


def shuffle_labels_within_root(pairs, rng):
    """Shuffle labels within each root stratum.

    Positives: stratum key = p['root']. Negatives: stratum key = p['root_a'].
    Some strata may be single-label (shuffle is identity) — counted and flagged.
    Floor for secondary boolean (transfers_across_wazn_within_root).
    Returns (shuffled_pairs, n_degenerate_strata).
    """
    strata = {}
    for i, p in enumerate(pairs):
        root = p.get("root") or p.get("root_a", "UNK")
        strata.setdefault(root, []).append(i)

    shuffled = [dict(p) for p in pairs]
    n_degen = 0
    for root, indices in strata.items():
        labels = [pairs[i]["label"] for i in indices]
        if len(set(labels)) == 1:
            n_degen += 1
        rng.shuffle(labels)
        for j, idx in enumerate(indices):
            shuffled[idx]["label"] = labels[j]
    return shuffled, n_degen


def apply_floor_shuffle(arm, pairs, seed):
    """Apply label shuffle for floor arms; identity for non-floor arms."""
    if arm == "floor_wazn":
        rng = random.Random(seed)
        return shuffle_labels_within_wazn(pairs, rng), 0
    elif arm == "floor_root":
        rng = random.Random(seed)
        shuffled, n_degen = shuffle_labels_within_root(pairs, rng)
        return shuffled, n_degen
    else:
        return pairs, 0


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_one_unit(arm, seed_idx, full_dataset, encoded_cache):
    """Train one (arm, seed_idx) unit. Saves checkpoint. Returns result dict."""
    seed = run_seed(arm, seed_idx)
    print(f"\n[{arm}] === seed {seed} (idx {seed_idx}) ===", flush=True)

    torch.manual_seed(seed)
    model = make_model(arm, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Apply floor shuffle to training pairs (real labels used at eval)
    train_pairs_raw = full_dataset["train"]
    train_pairs, n_degen = apply_floor_shuffle(arm, train_pairs_raw, seed)
    if arm in ("floor_wazn", "floor_root") and n_degen > 0:
        print(f"  [{arm}] {n_degen} degenerate strata (single-label; shuffle is identity)",
              flush=True)

    primary_test  = full_dataset["primary_test"]
    secondary_test = full_dataset["secondary_test"]

    # Train / val split (seeded per model seed)
    rng = random.Random(seed)
    pair_order = list(range(len(train_pairs)))
    rng.shuffle(pair_order)
    n_val = max(1, int(len(pair_order) * VAL_FRAC))
    val_idx_set = set(pair_order[:n_val])
    tr  = [train_pairs[i] for i in range(len(train_pairs)) if i not in val_idx_set]
    val = [train_pairs[i] for i in range(len(train_pairs)) if i in val_idx_set]

    best_val   = -1.0
    best_state = None
    t0 = time.time()

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        ep_order = list(range(len(tr)))
        rng.shuffle(ep_order)

        # Epoch-level matrix cache (recomputed each epoch from updated params)
        col_matrices = {}
        optimizer.zero_grad()
        total_loss = torch.zeros((), dtype=torch.float32)

        for idx in ep_order:
            p = tr[idx]
            enc_a = encoded_cache[(p["word_a"], p["col_a"])]
            enc_b = encoded_cache[(p["word_b"], p["col_b"])]
            prob = model_forward(model, arm, enc_a, p["col_a"], enc_b, p["col_b"],
                                 col_matrices)
            loss = F.binary_cross_entropy(prob, torch.tensor(float(p["label"])))
            total_loss = total_loss + loss

        (total_loss / len(tr)).backward()
        optimizer.step()

        # Val eval (uses floor-shuffled labels for floor arms — epoch selection)
        model.eval()
        val_correct = 0
        with torch.no_grad():
            col_mat_val = {}
            for p in val:
                enc_a = encoded_cache[(p["word_a"], p["col_a"])]
                enc_b = encoded_cache[(p["word_b"], p["col_b"])]
                prob = model_forward(model, arm, enc_a, p["col_a"], enc_b, p["col_b"],
                                     col_mat_val)
                val_correct += int((prob.item() > 0.5) == bool(p["label"]))
        val_acc = val_correct / len(val)

        if val_acc > best_val:
            best_val   = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if ep % 30 == 0 or ep == MAX_EPOCHS:
            elapsed = time.time() - t0
            print(f"    ep {ep:3d}/{MAX_EPOCHS}  val={val_acc:.3f}  "
                  f"best={best_val:.3f}  {elapsed:.0f}s", flush=True)

    # Load best state; evaluate on REAL labels (not shuffled)
    model.load_state_dict(best_state)
    model.eval()

    def eval_split(split_pairs):
        records = []
        correct = 0
        with torch.no_grad():
            col_mat_ev = {}
            for p in split_pairs:
                enc_a = encoded_cache[(p["word_a"], p["col_a"])]
                enc_b = encoded_cache[(p["word_b"], p["col_b"])]
                prob = model_forward(model, arm, enc_a, p["col_a"], enc_b, p["col_b"],
                                     col_mat_ev)
                pv = prob.item()
                pred = pv > 0.5
                is_correct = pred == bool(p["label"])
                correct += int(is_correct)
                records.append({
                    "word_a": p["word_a"], "col_a": p["col_a"],
                    "word_b": p["word_b"], "col_b": p["col_b"],
                    "label": p["label"],
                    "prob": round(pv, 6),
                    "correct": bool(is_correct),
                })
        acc = correct / len(split_pairs) if split_pairs else float("nan")
        return round(acc, 6), records

    primary_acc,   primary_records   = eval_split(primary_test)
    secondary_acc, secondary_records = eval_split(secondary_test)
    wall_sec = time.time() - t0

    print(f"  [{arm} s{seed}] primary={primary_acc:.4f}  "
          f"secondary={secondary_acc:.4f}  {wall_sec:.0f}s", flush=True)

    ckpt = {
        "arm": arm,
        "seed": seed,
        "seed_idx": seed_idx,
        "arm_offset": ARM_OFFSET[arm],
        "split_meta": full_dataset["meta"],
        "primary_test": {
            "accuracy": primary_acc,
            "n_pairs": len(primary_test),
            "per_pair": primary_records,
        },
        "secondary_test": {
            "accuracy": secondary_acc,
            "n_pairs": len(secondary_test),
            "per_pair": secondary_records,
        },
        "best_val_acc": round(best_val, 6),
        "wall_sec": round(wall_sec, 1),
        "n_degen_floor_strata": n_degen,
    }

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{arm}_s{seed}_results.json")
    with open(ckpt_path, "w", encoding="utf-8") as fh:
        json.dump(ckpt, fh, ensure_ascii=False, indent=1)
    print(f"  saved: {ckpt_path}", flush=True)
    return ckpt


# ---------------------------------------------------------------------------
# SMOKE BATTERY
# ---------------------------------------------------------------------------

def smoke_import_identity():
    """Assert byte-identical imports from pilot module (§12.4 MUST-REUSE)."""
    print("[smoke] import identity (is-assertions)...", flush=True)
    checks = [
        ("featurize",          featurize,            _pilot.featurize),
        ("arabic_char_angle",  arabic_char_angle,    _pilot.arabic_char_angle),
        ("strip_al",           strip_al,             _pilot.strip_al),
        ("COL_SLOTS",          COL_SLOTS,            _pilot.COL_SLOTS),
        ("COL_ORDER",          COL_ORDER,            _pilot.COL_ORDER),
        ("encode_word_state",  encode_word_state,    _pilot.encode_word_state),
        ("apply_ansatz_and_probs", apply_ansatz_and_probs, _pilot.apply_ansatz_and_probs),
        ("B1Model",            B1Model,              _pilot.B1Model),
        ("build_word_features",build_word_features,  _pilot.build_word_features),
        ("build_positives",    build_positives,      _pilot.build_positives),
        ("letter_set",         letter_set,           _pilot.letter_set),
    ]
    for name, local, remote in checks:
        assert local is remote, \
            f"IMPORT IDENTITY FAIL: {name} is not the pilot module's object"
    print("  PASSED: all pilot objects byte-identical (is assertions)", flush=True)


def smoke_fixed_scramble(arm_name, dataset, encoded_cache):
    """Fixed-scramble smoke per §12.2 corrected spec.

    ONE fixed scramble drawn at seed 4242; train AND eval on THE SAME labels.
    Bar: memorization accuracy < 0.65.
    """
    print(f"[smoke] fixed-scramble ({arm_name})...", flush=True)
    subset = dataset["train"][:min(80, len(dataset["train"]))]

    # ONE fixed scramble
    scramble_rng = random.Random(4242)
    fixed_labels = [scramble_rng.randint(0, 1) for _ in subset]

    torch.manual_seed(88888)
    model = make_model(arm_name, 88888)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(30):
        model.train()
        col_matrices = {}
        optimizer.zero_grad()
        total_loss = torch.zeros(())
        for i, p in enumerate(subset):
            enc_a = encoded_cache[(p["word_a"], p["col_a"])]
            enc_b = encoded_cache[(p["word_b"], p["col_b"])]
            prob = model_forward(model, arm_name, enc_a, p["col_a"], enc_b, p["col_b"],
                                 col_matrices)
            loss = F.binary_cross_entropy(prob, torch.tensor(float(fixed_labels[i])))
            total_loss = total_loss + loss
        (total_loss / len(subset)).backward()
        optimizer.step()

    # Evaluate on THE SAME scrambled labels
    model.eval()
    correct = 0
    with torch.no_grad():
        col_mat_ev = {}
        for i, p in enumerate(subset):
            enc_a = encoded_cache[(p["word_a"], p["col_a"])]
            enc_b = encoded_cache[(p["word_b"], p["col_b"])]
            prob = model_forward(model, arm_name, enc_a, p["col_a"], enc_b, p["col_b"],
                                 col_mat_ev)
            correct += int((prob.item() > 0.5) == bool(fixed_labels[i]))
    acc = correct / len(subset)
    print(f"  {arm_name}: fixed-scramble acc = {acc:.3f}  (bar < 0.65)", flush=True)
    assert acc < 0.65, \
        f"SMOKE FAIL ({arm_name}): fixed-scramble acc {acc:.3f} >= 0.65"
    print(f"  PASSED: {arm_name} fixed-scramble", flush=True)


def smoke_grad_nonzero_b1(dataset, encoded_cache):
    """Grad nonzero for B1: ansatz.theta grad norm > 0."""
    print("[smoke] grad nonzero (b1)...", flush=True)
    torch.manual_seed(0)
    model = B1Model(seed=0)
    model.zero_grad()
    for p in dataset["train"][:3]:
        enc_a = encoded_cache[(p["word_a"], p["col_a"])]
        enc_b = encoded_cache[(p["word_b"], p["col_b"])]
        prob = model.forward_encoded(enc_a, enc_b)
        F.binary_cross_entropy(prob, torch.tensor(float(p["label"]))).backward()
    gn = model.ansatz.theta.grad.norm().item()
    print(f"  b1 ansatz.theta grad norm = {gn:.6f}", flush=True)
    assert gn > 1e-10, f"SMOKE FAIL: b1 ansatz grad norm {gn} <= 1e-10"
    print("  PASSED: b1 grad nonzero", flush=True)


def smoke_grad_nonzero_wazn(dataset, encoded_cache):
    """Grad nonzero for WAZN: ALL 6 column param tensors must have nonzero grad.

    Batch must cover all 6 columns to catch per-column wiring bugs.
    """
    print("[smoke] grad nonzero (wazn, all 6 columns)...", flush=True)
    torch.manual_seed(1)
    model = WaznModel(seed=1)
    model.zero_grad()

    # Ensure at least one pair per column appears as col_a or col_b
    # Collect pairs, covering all 6 columns
    covered_cols = set()
    pairs_for_smoke = []
    for p in dataset["train"]:
        if p["col_a"] not in covered_cols or p["col_b"] not in covered_cols:
            pairs_for_smoke.append(p)
            covered_cols.add(p["col_a"])
            covered_cols.add(p["col_b"])
        if covered_cols == set(COL_ORDER):
            break

    if covered_cols != set(COL_ORDER):
        print(f"  WARNING: could only cover columns {covered_cols} in training pairs",
              flush=True)

    for p in pairs_for_smoke:
        enc_a = encoded_cache[(p["word_a"], p["col_a"])]
        enc_b = encoded_cache[(p["word_b"], p["col_b"])]
        prob = model.forward_encoded(enc_a, p["col_a"], enc_b, p["col_b"])
        F.binary_cross_entropy(prob, torch.tensor(float(p["label"]))).backward()

    all_pass = True
    for col in COL_ORDER:
        param = model.family.params[col]
        if param.grad is None:
            print(f"  FAIL: {col} grad is None (not covered in batch?)", flush=True)
            all_pass = False
        else:
            gn = param.grad.norm().item()
            active = col in covered_cols
            print(f"  {col:10s}  grad_norm={gn:.6f}  {'[covered]' if active else '[NOT COVERED]'}",
                  flush=True)
            if active and gn <= 1e-10:
                print(f"  FAIL: {col} grad norm {gn} <= 1e-10", flush=True)
                all_pass = False
    assert all_pass, "SMOKE FAIL: wazn — one or more column param tensors have zero/None grad"
    print("  PASSED: wazn grad nonzero (all covered columns)", flush=True)


def smoke_grad_nonzero_wazn_shuf(dataset, encoded_cache):
    """Grad nonzero for WAZN-SHUFFLED."""
    print("[smoke] grad nonzero (wazn_shuf)...", flush=True)
    torch.manual_seed(2)
    model = WaznShuffledModel(seed=2)
    model.zero_grad()
    for p in dataset["train"][:6]:
        enc_a = encoded_cache[(p["word_a"], p["col_a"])]
        enc_b = encoded_cache[(p["word_b"], p["col_b"])]
        prob = model.forward_encoded(enc_a, p["col_a"], enc_b, p["col_b"])
        F.binary_cross_entropy(prob, torch.tensor(float(p["label"]))).backward()
    gn_total = sum(
        model.family.params[col].grad.norm().item()
        for col in COL_ORDER
        if model.family.params[col].grad is not None
    )
    print(f"  wazn_shuf total grad norm across columns = {gn_total:.6f}", flush=True)
    assert gn_total > 1e-10, \
        f"SMOKE FAIL: wazn_shuf total grad norm {gn_total} <= 1e-10"
    print("  PASSED: wazn_shuf grad nonzero", flush=True)


def smoke_param_count_report():
    """Report WAZN vs B1 param counts. Not a gate; informational."""
    print("[smoke] param-count report...", flush=True)
    b1_ansatz = 24
    b1_total  = 26
    wazn_per_col = {col: WAZN_PARAMS_TOTAL[col] for col in COL_ORDER}
    wazn_ansatz  = WAZN_TOTAL_ANSATZ_PARAMS  # 132
    wazn_total   = wazn_ansatz + 2           # 134
    ratio = wazn_total / b1_total

    print(f"  B1:   {b1_ansatz} ansatz + 2 readout = {b1_total} total", flush=True)
    print(f"  WAZN per-column ansatz params:", flush=True)
    for col in COL_ORDER:
        n_ent = len(WAZN_TOPOLOGY[col])
        print(f"    {col:10s}: {WAZN_PARAMS_PER_LAYER[col]} params/layer × {N_LAYERS} = "
              f"{WAZN_PARAMS_TOTAL[col]} (entanglers/layer={n_ent})", flush=True)
    print(f"  WAZN: {wazn_ansatz} ansatz + 2 readout = {wazn_total} total", flush=True)
    print(f"  Ratio WAZN/B1 = {ratio:.2f}× (B1 is NOT matched within 5%; "
          f"WAZN-SHUFFLED is the capacity control, §12.5)", flush=True)
    print(f"  WAZN-SHUFFLED: identical structure to WAZN, {wazn_total} total params.",
          flush=True)
    print("  [informational — not a gate]", flush=True)


def smoke_shuf_perm_report():
    """Report WAZN-SHUFFLED permutation: fixed points, topology-class crossings."""
    print("[smoke] shuffled-control permutation report (seed 777)...", flush=True)
    TOPOLOGY_CLASS = {
        "faail": "A", "faiil": "B", "fa33aal": "B",
        "maf3uul": "B", "mu1": "C", "mu2": "C"
    }
    fixed_points = []
    cross_class  = []
    within_class = []
    for col in COL_ORDER:
        mapped = SHUF_PERM[col]
        is_fixed = (col == mapped)
        same_class = (TOPOLOGY_CLASS[col] == TOPOLOGY_CLASS[mapped])
        print(f"  {col:10s} -> {mapped:10s}  "
              f"{'[FIXED POINT]' if is_fixed else ''}"
              f"{'[same class ' + TOPOLOGY_CLASS[col] + ']' if same_class and not is_fixed else ''}"
              f"{'[CROSS CLASS ' + TOPOLOGY_CLASS[col] + '->' + TOPOLOGY_CLASS[mapped] + ']' if not same_class else ''}",
              flush=True)
        if is_fixed:
            fixed_points.append(col)
        elif same_class:
            within_class.append(col)
        else:
            cross_class.append(col)

    print(f"  Fixed points: {fixed_points or 'none'}", flush=True)
    print(f"  Within-class (topology unchanged): {within_class or 'none'}", flush=True)
    print(f"  Cross-class moves: {cross_class or 'none'}", flush=True)
    if not cross_class:
        print("  FLAG: NO cross-class moves! Permutation is class-preserving. "
              "Columns within a class are distributionally identical (same topology, "
              "different random init). Control's bite comes only from cross-class moves. "
              "If all moves are within-class, this control is vacuous — record loudly, "
              "do NOT reroll the pinned seed (fixed by §12.5).", flush=True)
    else:
        print(f"  Permutation crosses class boundaries ({len(cross_class)} columns) "
              f"— control has structural bite.", flush=True)
    print("  [informational — not a gate]", flush=True)


def run_smoke_battery(full_dataset, encoded_cache):
    """Run the full smoke battery. All gates must pass before launch."""
    print("\n" + "="*60, flush=True)
    print("[smoke] SMOKE BATTERY START", flush=True)
    print("="*60, flush=True)

    smoke_import_identity()
    smoke_param_count_report()
    smoke_shuf_perm_report()
    smoke_grad_nonzero_b1(full_dataset, encoded_cache)
    smoke_grad_nonzero_wazn(full_dataset, encoded_cache)
    smoke_grad_nonzero_wazn_shuf(full_dataset, encoded_cache)
    smoke_fixed_scramble("b1",       full_dataset, encoded_cache)
    smoke_fixed_scramble("wazn",     full_dataset, encoded_cache)
    smoke_fixed_scramble("wazn_shuf", full_dataset, encoded_cache)

    print("\n" + "="*60, flush=True)
    print("[smoke] ALL GATES PASSED", flush=True)
    print("="*60 + "\n", flush=True)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_inventory_and_build():
    """Load inventory, verify SHA, build features, encoded cache, dataset."""
    print("[init] loading inventory ...", flush=True)
    with open(INVENTORY_PATH, encoding="utf-8") as fh:
        inv = json.load(fh)

    meta_sha = inv["meta"]["inventory_sha256"]
    assert meta_sha.startswith(REGISTERED_SHA_PREFIX), (
        f"Inventory SHA prefix mismatch: {meta_sha[:16]} != {REGISTERED_SHA_PREFIX}"
    )
    words = [w for w in inv["words"] if w["admitted"]]
    assert len(words) == 499,  f"Expected 499 admitted words, got {len(words)}"
    assert len(set(w["root"] for w in words)) == 182, "Expected 182 roots"
    print(f"  inventory ok: {len(words)} words, sha_prefix={meta_sha[:16]}", flush=True)

    print("[init] building features and encoded cache ...", flush=True)
    _pilot._clamp_events.clear()
    features = build_word_features(words)
    encoded_cache = {}
    for key, feat in features.items():
        slot_angles = feat[:N_QUBITS]
        col_angle   = feat[N_QUBITS]
        encoded_cache[key] = encode_word_state(slot_angles, col_angle)
    if _pilot._clamp_events:
        print(f"  slot-clamp events: {len(_pilot._clamp_events)}", flush=True)
    print(f"  encoded {len(encoded_cache)} words", flush=True)

    print("[init] building full dataset (strictly within-split) ...", flush=True)
    full_dataset = build_full_dataset(words)
    m = full_dataset["meta"]
    print(f"  train: {m['n_train_pos']} pos + {m['n_train_neg']} neg = {m['n_train_total']}",
          flush=True)
    print(f"  primary_test: {m['n_primary_test_pos']} pos + {m['n_primary_test_neg']} neg "
          f"= {m['n_primary_test_total']} (neg_cap={m['primary_neg_capacity_found']})",
          flush=True)
    print(f"  secondary_test: {m['n_secondary_test_pos']} pos + {m['n_secondary_test_neg']} neg "
          f"= {m['n_secondary_test_total']} (holdout_roots={m['n_secondary_holdout_roots']})",
          flush=True)

    return words, features, encoded_cache, full_dataset


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke battery only, then exit")
    ap.add_argument("--worker", type=int, default=None,
                    help="Worker index (0-4): runs units[W::5]")
    ap.add_argument("--arm", type=str, default=None,
                    help="Arm name for single-unit run")
    ap.add_argument("--seed_idx", type=int, default=None,
                    help="Seed index (0-4) for single-unit run")
    a = ap.parse_args()

    t_wall = time.time()

    # Load data (always needed)
    words, features, encoded_cache, full_dataset = load_inventory_and_build()

    if a.smoke:
        run_smoke_battery(full_dataset, encoded_cache)
        print("[exp77] --smoke mode: battery passed, exiting.", flush=True)
        return 0

    if a.arm is not None and a.seed_idx is not None:
        # Single-unit mode
        assert a.arm in ARMS, f"Unknown arm {a.arm!r}; valid: {ARMS}"
        assert 0 <= a.seed_idx < N_SEEDS, f"seed_idx must be 0-{N_SEEDS-1}"
        train_one_unit(a.arm, a.seed_idx, full_dataset, encoded_cache)
        print(f"[exp77] unit ({a.arm}, seed_idx={a.seed_idx}) done; "
              f"wall={time.time()-t_wall:.0f}s", flush=True)
        return 0

    if a.worker is not None:
        # Worker mode: handle units[W::5]
        W = a.worker
        assert 0 <= W <= 4, f"Worker must be 0-4, got {W}"
        my_units = UNITS[W::5]
        print(f"[worker {W}] units: {my_units}", flush=True)
        for arm, seed_idx in my_units:
            train_one_unit(arm, seed_idx, full_dataset, encoded_cache)
        print(f"[worker {W}] all done; wall={time.time()-t_wall:.0f}s", flush=True)
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
