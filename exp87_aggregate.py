#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp87_aggregate.py — ARABIC WUG TEST, LEG A: SOLE WRITER OF results_exp87.json
Registration: /Users/wajahath/qnlp_private_docs/80_exp87_arabic_wug_test.md §3

Run after exp87_run.py --evaluate (and --gates for anchor status).
Interpreter: ~/venv/bin/python3

Outputs: ~/discocat_arabic_v2/results_exp87.json
"""

import json
import os
import sys
import time

WORK_DIR = os.path.expanduser('~/discocat_arabic_v2')

EXP84_ANCHOR = {
    42: {'bpe': 0.1974, 'wazn': 0.5785},
    43: {'bpe': 0.1899, 'wazn': 0.5624},
    44: {'bpe': 0.2045, 'wazn': 0.5609},
}

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def artifact_path(name):
    return os.path.join(WORK_DIR, name)


def main():
    t0 = time.time()
    print('[aggregate] Loading exp87 artifacts…')

    # ── Load evaluation output ────────────────────────────────────────────────
    eval_path = artifact_path('exp87_evaluate.json')
    if not os.path.exists(eval_path):
        sys.exit('[aggregate] FATAL: exp87_evaluate.json not found — run --evaluate first')
    eval_data = load_json(eval_path)

    # ── Load gate outcomes ────────────────────────────────────────────────────
    gate1_path = artifact_path('exp87_gate_nonce_validity.json')
    gate2_path = artifact_path('exp87_gate_gold_determinism.json')
    gate3_path = artifact_path('exp87_gate_replication_anchor.json')

    gate1 = load_json(gate1_path) if os.path.exists(gate1_path) else None
    gate2 = load_json(gate2_path) if os.path.exists(gate2_path) else None
    gate3 = load_json(gate3_path) if os.path.exists(gate3_path) else None

    anchor_voided = False
    if gate3 is None:
        print('[aggregate] WARN: replication_anchor gate not found — treating as voided')
        anchor_voided = True
    elif not gate3.get('gate_passes', False):
        anchor_voided = True
        print('[aggregate] NOTE: anchor gate FAILED → anchor_voided=True '
              '(science booleans recorded but marked void)')

    # ── Per-seed F1 table ─────────────────────────────────────────────────────
    seed_results = eval_data['seed_results']

    per_seed_table = []
    for sr in seed_results:
        seed = sr['seed']
        nonce = sr['boundary_f1']['nonce']
        real  = sr['boundary_f1']['real']

        # Real-control gap: WAZN F1 − BPE F1
        real_gap_wazn  = real['wazn']['f1']  - real['bpe']['f1']
        real_gap_fused = real['fused']['f1'] - real['bpe']['f1']

        # Nonce gap: WAZN F1 − BPE F1
        nonce_gap_wazn  = nonce['wazn']['f1']  - nonce['bpe']['f1']
        nonce_gap_fused = nonce['fused']['f1'] - nonce['bpe']['f1']

        # Productive headroom: SKELETON nonce − BPE nonce
        skel_headroom = nonce['skel']['f1'] - nonce['bpe']['f1']

        per_seed_table.append({
            'seed': seed,
            # F1 values
            'bpe_real_f1':   real['bpe']['f1'],
            'wazn_real_f1':  real['wazn']['f1'],
            'fused_real_f1': real['fused']['f1'],
            'skel_real_f1':  real['skel']['f1'],
            'bpe_nonce_f1':   nonce['bpe']['f1'],
            'wazn_nonce_f1':  nonce['wazn']['f1'],
            'fused_nonce_f1': nonce['fused']['f1'],
            'skel_nonce_f1':  nonce['skel']['f1'],
            # Gaps
            'real_gap_wazn_over_bpe':   round(real_gap_wazn,  4),
            'real_gap_fused_over_bpe':  round(real_gap_fused, 4),
            'nonce_gap_wazn_over_bpe':  round(nonce_gap_wazn,  4),
            'nonce_gap_fused_over_bpe': round(nonce_gap_fused, 4),
            'skel_headroom_over_bpe':   round(skel_headroom,   4),
            # Per-seed bar checks (doc 80 §3)
            'seed_lexicon_bound_wazn': (
                real_gap_wazn  >= 0.10 and nonce_gap_wazn  < 0.05
            ),
            'seed_lexicon_bound_fused': (
                real_gap_fused >= 0.10 and nonce_gap_fused < 0.05
            ),
            'seed_productive_headroom': skel_headroom >= 0.10,
            # Anchor delta
            'anchor_delta_bpe':  round(real['bpe']['f1']  - EXP84_ANCHOR[seed]['bpe'],  4),
            'anchor_delta_wazn': round(real['wazn']['f1'] - EXP84_ANCHOR[seed]['wazn'], 4),
            # Fertility
            'fertility': sr.get('fertility', {}),
            # Analyzer behavior
            'analyzer_behavior': sr.get('analyzer_behavior', {}),
        })

    # ── doc 80 §3 BARS ───────────────────────────────────────────────────────
    # morphology_is_lexicon_bound: real gap WAZN-BPE ≥ +0.10 AND nonce gap < +0.05
    # in ALL 3 seeds (using WAZN arm as the primary morphological arm)
    morphology_is_lexicon_bound = all(
        r['seed_lexicon_bound_wazn'] for r in per_seed_table
    )

    # productive_headroom_exists: SKELETON nonce F1 ≥ BPE nonce F1 + 0.10, all 3 seeds
    # Interpretation: all-3-seeds (symmetric with morphology_is_lexicon_bound);
    # per-seed values also recorded.
    productive_headroom_exists = all(
        r['seed_productive_headroom'] for r in per_seed_table
    )

    # ── Named outcomes (doc 80 §3) ────────────────────────────────────────────
    # Third outcomes:
    #   NOT_FULLY_GATED: nonce WAZN gap ≥ +0.05 for any seed
    #   INSTRUMENT_FAILURE: real-control gap < +0.10 for any seed (replication fails)
    #   ANCHOR_VOIDED: anchor gate failed → can't trust comparisons

    any_nonce_gap_ge_05 = any(
        r['nonce_gap_wazn_over_bpe'] >= 0.05 for r in per_seed_table
    )
    any_real_gap_lt_10 = any(
        r['real_gap_wazn_over_bpe'] < 0.10 for r in per_seed_table
    )

    if anchor_voided:
        named_outcome = 'ANCHOR_VOIDED: anchor gate failed; science booleans recorded but void'
        science_booleans_valid = False
    elif any_real_gap_lt_10:
        named_outcome = 'INSTRUMENT_FAILURE: real-control WAZN gap < +0.10 — exp84 not replicated; no boolean reading'
        science_booleans_valid = False
    elif any_nonce_gap_ge_05:
        named_outcome = 'NOT_FULLY_GATED: nonce WAZN gap ≥ +0.05 in at least one seed — investigate before claiming'
        science_booleans_valid = False
    elif morphology_is_lexicon_bound and productive_headroom_exists:
        named_outcome = 'LEXICON_BOUND_WITH_HEADROOM: morphology memorized AND productive segmenter is buildable'
        science_booleans_valid = True
    elif morphology_is_lexicon_bound and not productive_headroom_exists:
        named_outcome = 'LEXICON_BOUND_NO_HEADROOM: morphology memorized; template-skeleton approach insufficient at this set'
        science_booleans_valid = True
    elif not morphology_is_lexicon_bound and productive_headroom_exists:
        named_outcome = 'PARTIAL_GENERALIZATION: nonce gap ≥ +0.05 or real gap < +0.10; see per-seed table'
        science_booleans_valid = True
    else:
        named_outcome = 'INCONCLUSIVE: neither bar met; see per-seed table'
        science_booleans_valid = True

    # ── Full per-arm per-condition table ──────────────────────────────────────
    arms = ['bpe', 'wazn', 'fused', 'skel']
    conditions = ['real', 'nonce']
    f1_table = []
    for sr in seed_results:
        seed = sr['seed']
        for cond in conditions:
            row = {'seed': seed, 'condition': cond}
            for arm in arms:
                row[f'f1_{arm}'] = sr['boundary_f1'][cond][arm]['f1']
                row[f'p_{arm}']  = sr['boundary_f1'][cond][arm]['p']
                row[f'r_{arm}']  = sr['boundary_f1'][cond][arm]['r']
            f1_table.append(row)

    # ── Gate summary ──────────────────────────────────────────────────────────
    gates_summary = {
        'nonce_validity': {
            'passes': gate1.get('gate_overall_passes', None) if gate1 else None,
            'verdict': gate1.get('verdict', 'NOT_RUN') if gate1 else 'NOT_RUN',
        },
        'gold_determinism': {
            'passes': gate2.get('gate_overall_passes', None) if gate2 else None,
            'verdict': gate2.get('verdict', 'NOT_RUN') if gate2 else 'NOT_RUN',
        },
        'replication_anchor': {
            'passes': gate3.get('gate_passes', None) if gate3 else None,
            'verdict': gate3.get('verdict', 'NOT_RUN') if gate3 else 'NOT_RUN',
            'anchor_voided': anchor_voided,
        },
    }

    # ── Assemble results_exp87.json ───────────────────────────────────────────
    results = {
        'meta': {
            'experiment':       'exp87',
            'registration':     '80_exp87_arabic_wug_test.md',
            'aggregate_script': 'exp87_aggregate.py',
            'timestamp':        time.strftime('%Y-%m-%dT%H:%M:%S'),
            'n_seeds':          len(seed_results),
            'seeds':            [sr['seed'] for sr in seed_results],
        },
        'science_booleans': {
            'morphology_is_lexicon_bound': morphology_is_lexicon_bound,
            'productive_headroom_exists':  productive_headroom_exists,
            'valid': science_booleans_valid,
        },
        'named_outcome': named_outcome,
        'anchor_voided':  anchor_voided,
        'gates': gates_summary,
        'per_seed': per_seed_table,
        'f1_table': f1_table,
        'interpretation_notes': {
            'all_3_seeds_rule': (
                'Both booleans use all-3-seeds criterion (symmetric). '
                'Per-seed values are in per_seed[*].seed_lexicon_bound_* for alternative reading.'
            ),
            'mu1_mu2_collision': (
                'mu1 and mu2 produce identical undiacritized surface forms for the same root. '
                'SKELETON arm marks them as ambiguous → unmatched → predicted boundary {}. '
                'This limits SKELETON nonce F1 ceiling to ~0.80 (120/180 uniquely matched).'
            ),
            'anchor_context': (
                'Real-control forms are all definite ال+stem forms (gold={2}). '
                'exp84 gold was multi-boundary corpus words. '
                'Anchor failure likely reflects distribution difference, not tokenizer drift.'
            ),
            'fused_arm': (
                'FUSED arm uses predict_boundaries_fused_expanded (exp84 path exactly). '
                'Fallback for unanalyzable words uses fused_tok (own tokenizer), not bpe_tok.'
            ),
        },
    }

    out_path = os.path.join(WORK_DIR, 'results_exp87.json')
    save_json(results, out_path)

    print(f'\n[aggregate] ── RESULTS ──────────────────────────────────────────')
    print(f'  morphology_is_lexicon_bound = {morphology_is_lexicon_bound}')
    print(f'  productive_headroom_exists  = {productive_headroom_exists}')
    print(f'  science_booleans_valid      = {science_booleans_valid}')
    print(f'  anchor_voided               = {anchor_voided}')
    print(f'  named_outcome: {named_outcome}')
    print(f'\n  Per-seed F1 table:')
    for r in per_seed_table:
        print(f'    seed={r["seed"]}:')
        print(f'      real  — BPE={r["bpe_real_f1"]:.4f} WAZN={r["wazn_real_f1"]:.4f} '
              f'FUSED={r["fused_real_f1"]:.4f} SKEL={r["skel_real_f1"]:.4f}')
        print(f'      nonce — BPE={r["bpe_nonce_f1"]:.4f} WAZN={r["wazn_nonce_f1"]:.4f} '
              f'FUSED={r["fused_nonce_f1"]:.4f} SKEL={r["skel_nonce_f1"]:.4f}')
        print(f'      gaps  — real_wazn={r["real_gap_wazn_over_bpe"]:+.4f} '
              f'nonce_wazn={r["nonce_gap_wazn_over_bpe"]:+.4f} '
              f'skel_head={r["skel_headroom_over_bpe"]:+.4f}')
        print(f'      anchor_delta — bpe={r["anchor_delta_bpe"]:+.4f} wazn={r["anchor_delta_wazn"]:+.4f}')
    print(f'\n[aggregate] Written: {out_path}')


if __name__ == '__main__':
    main()
