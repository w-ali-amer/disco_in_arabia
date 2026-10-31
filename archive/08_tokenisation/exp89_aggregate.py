#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp89_aggregate.py — Aggregate exp89 results and write results_exp89.json.

Reads: exp89_anchor_gate.json, exp89_evaluate.json, exp89_gates.json
Sole writer: results_exp89.json

Registration: qnlp_private_docs/81_exp88_89_skeleton_scale_and_allocation.md §3

Bars (§3):
  allocation_cell_occupied = some NEW arm (opt or noback) has:
    - held-out fertility ≤ 1.10 × BPE fertility
    - expanded boundary F1 ≥ BPE F1 + 0.10
    ALL 3 seeds must satisfy both.
  Named outcome if NO arm qualifies: CELL_EMPTY_AT_32K.

Per-arm third outcomes (when only one bar met):
  - fertility_only:  fertility ≤ 1.10×BPE but F1 gap < 0.10
  - boundary_only:   F1 gap ≥ 0.10 but fertility > 1.10×BPE
  - replication_failure: anchor gate missed ±0.03 (voids the run)
"""

import json
import os
import sys
import time

WORK_DIR = os.path.expanduser('~/discocat_arabic_v2')

FERTILITY_BAR = 1.10   # ≤ 1.10 × BPE
F1_GAP_BAR    = 0.10   # ≥ BPE F1 + 0.10

SEEDS = [42, 43, 44]


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def artifact_path(name):
    return os.path.join(WORK_DIR, name)


def check_arm_all_seeds(arm_key, per_seed_results):
    """
    Check whether an arm meets BOTH bars across ALL 3 seeds.

    Returns:
        {
          'arm': arm_key,
          'per_seed': [{'seed', 'fertility_ratio', 'f1_gap', 'fertility_ok', 'f1_ok', 'both_ok'}],
          'fertility_ok_all_seeds': bool,
          'f1_ok_all_seeds': bool,
          'both_ok_all_seeds': bool,
        }
    """
    per_seed = []
    for row in per_seed_results:
        seed = row['seed']
        bpe_fert = row['fertility']['bpe']
        arm_fert = row['fertility'][arm_key]
        bpe_f1   = row['boundary_f1']['bpe']['f1']
        arm_f1   = row['boundary_f1'][arm_key]['f1']

        fert_ratio = arm_fert / bpe_fert if bpe_fert > 0 else float('nan')
        f1_gap     = arm_f1 - bpe_f1

        fert_ok = fert_ratio <= FERTILITY_BAR
        f1_ok   = f1_gap    >= F1_GAP_BAR
        per_seed.append({
            'seed': seed,
            'bpe_fertility': round(bpe_fert, 6),
            'arm_fertility': round(arm_fert, 6),
            'fertility_ratio': round(fert_ratio, 6),
            'bpe_f1': round(bpe_f1, 6),
            'arm_f1': round(arm_f1, 6),
            'f1_gap': round(f1_gap, 6),
            'fertility_ok': fert_ok,
            'f1_ok': f1_ok,
            'both_ok': fert_ok and f1_ok,
        })

    fert_all = all(r['fertility_ok'] for r in per_seed)
    f1_all   = all(r['f1_ok']        for r in per_seed)
    both_all = fert_all and f1_all

    # Determine third outcome
    if both_all:
        third_outcome = None
    elif fert_all and not f1_all:
        third_outcome = 'FERTILITY_ONLY'
    elif not fert_all and f1_all:
        third_outcome = 'BOUNDARY_ONLY'
    else:
        third_outcome = 'NEITHER_BAR_MET'

    return {
        'arm': arm_key,
        'per_seed': per_seed,
        'fertility_ok_all_seeds': fert_all,
        'f1_ok_all_seeds': f1_all,
        'both_ok_all_seeds': both_all,
        'third_outcome': third_outcome,
    }


def main():
    t0 = time.time()
    print('[aggregate] exp89_aggregate.py starting…')

    # ── Load intermediate artifacts ────────────────────────────────────────────
    anchor_path   = artifact_path('exp89_anchor_gate.json')
    evaluate_path = artifact_path('exp89_evaluate.json')
    gates_path    = artifact_path('exp89_gates.json')

    errors = []
    anchor_data   = None
    evaluate_data = None
    gates_data    = None

    for label, path, tgt_var in [
        ('anchor_gate',  anchor_path,   'anchor_data'),
        ('evaluate',     evaluate_path, 'evaluate_data'),
        ('gates',        gates_path,    'gates_data'),
    ]:
        if not os.path.exists(path):
            msg = f'{label}: file not found at {path}'
            print(f'  WARNING: {msg}')
            errors.append(msg)
        else:
            data = load_json(path)
            if label == 'anchor_gate':
                anchor_data = data
            elif label == 'evaluate':
                evaluate_data = data
            else:
                gates_data = data

    # ── Anchor gate summary ────────────────────────────────────────────────────
    anchor_summary = None
    if anchor_data is not None:
        anchor_summary = {
            'all_seeds_pass': anchor_data.get('all_seeds_pass', False),
            'verdict': anchor_data.get('verdict', 'UNKNOWN'),
        }
        if not anchor_summary['all_seeds_pass']:
            errors.append('ANCHOR_GATE_FAIL: run is voided per registration')

    # ── Allocation cell check (requires evaluate data) ─────────────────────────
    allocation_cell_occupied = False
    per_arm_results = {}
    named_outcome = 'INCOMPLETE_DATA'

    if evaluate_data is not None:
        per_seed_results = evaluate_data.get('per_seed', [])

        for arm_key in ('opt', 'noback'):
            per_arm_results[arm_key] = check_arm_all_seeds(arm_key, per_seed_results)

        # Cell is occupied if ANY new arm meets BOTH bars on ALL 3 seeds
        cell_occupied_by = [arm for arm, res in per_arm_results.items()
                            if res['both_ok_all_seeds']]
        allocation_cell_occupied = len(cell_occupied_by) > 0

        if allocation_cell_occupied:
            arms_str = ' and '.join(cell_occupied_by)
            named_outcome = (f'CELL_OCCUPIED: {arms_str} achieves '
                             f'fertility ≤1.10×BPE AND F1≥BPE+0.10 on all 3 seeds. '
                             f'TAX_CUT achieved within 32k vocab.')
        else:
            # Describe per-arm third outcomes
            outcomes = {arm: res['third_outcome']
                        for arm, res in per_arm_results.items()}
            named_outcome = (f'CELL_EMPTY_AT_32K: no new arm achieves both bars on all seeds. '
                             f'Per-arm outcomes: '
                             + ', '.join(f'{arm}={out}' for arm, out in outcomes.items())
                             + '. Calibrated probe-B predicts this outcome: '
                               'fertility and morphology cannot both be had '
                               'within this allocation family at 32k vocab.')

        print(f'[aggregate] allocation_cell_occupied = {allocation_cell_occupied}')
        print(f'[aggregate] named_outcome: {named_outcome}')

    # ── Gates summary ──────────────────────────────────────────────────────────
    gates_summary = None
    if gates_data is not None:
        g = gates_data.get('summary', {})
        gates_summary = {
            'all_vocab_accounting_pass':      g.get('all_vocab_accounting_pass', False),
            'all_expansion_determinism_pass': g.get('all_expansion_determinism_pass', False),
            'all_gates_pass':                 g.get('all_gates_pass', False),
        }

    # ── Write results ──────────────────────────────────────────────────────────
    results = {
        'experiment':  'exp89 — THE ALLOCATION CELL',
        'registration': os.path.expanduser(
            'qnlp_private_docs/81_exp88_89_skeleton_scale_and_allocation.md'),
        'sole_writer': 'exp89_aggregate.py → results_exp89.json',
        'bars': {
            'fertility_bar':  f'arm fertility ≤ {FERTILITY_BAR} × BPE fertility',
            'f1_gap_bar':     f'arm F1 ≥ BPE F1 + {F1_GAP_BAR}',
            'scope':          'all 3 seeds (42/43/44) must satisfy both bars',
        },
        'anchor_gate':            anchor_summary,
        'gates':                  gates_summary,
        'per_arm_results':        per_arm_results,
        'allocation_cell_occupied': allocation_cell_occupied,
        'named_outcome':          named_outcome,
        'errors':                 errors,
        'elapsed_s':              round(time.time() - t0, 1),
    }

    out_path = artifact_path('results_exp89.json')
    save_json(results, out_path)
    print(f'[aggregate] Written → {out_path}')
    print(f'  allocation_cell_occupied = {allocation_cell_occupied}')
    print(f'  named_outcome: {named_outcome}')
    if errors:
        print(f'  errors: {errors}')


if __name__ == '__main__':
    main()
