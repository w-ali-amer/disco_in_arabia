#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp84_aggregate.py — sole writer of results_exp84.json.

Reads all gate and LM artifacts produced by exp84_run.py.
Computes all §3 boolean outcomes and §4's wazn_lm_helps / NO_LM_SIGNAL.
Must be run after all 6 LM units have completed.

Registration: /Users/wajahath/qnlp_private_docs/76_exp84_fused_vocab_tokenizer.md
"""

import json
import math
import os
import sys

WORK_DIR = os.path.expanduser('~/discocat_arabic_v2')
RESULTS_PATH = os.path.join(WORK_DIR, 'results_exp84.json')

ARMS  = ['bpe', 'wazn', 'fused']
LM_SEEDS = [0, 1]


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def mean(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float('nan')


def aggregate():
    errors = []

    # ── corpus meta ───────────────────────────────────────────────────────────
    corpus_meta_path = os.path.join(WORK_DIR, 'exp84_corpus_meta.json')
    if os.path.exists(corpus_meta_path):
        corpus_meta = load_json(corpus_meta_path)
    else:
        corpus_meta = {'error': 'exp84_corpus_meta.json not found'}
        errors.append('corpus_meta missing')

    # ── gate artifacts ────────────────────────────────────────────────────────
    gate_paths = {
        'vocab_accounting':     os.path.join(WORK_DIR, 'exp84_gate_vocab_accounting.json'),
        'derangement':          os.path.join(WORK_DIR, 'exp84_gate_derangement.json'),
        'expansion_determinism': os.path.join(WORK_DIR, 'exp84_gate_expansion_determinism.json'),
    }
    gates = {}
    for name, path in gate_paths.items():
        if os.path.exists(path):
            gates[name] = load_json(path)
        else:
            gates[name] = {'error': f'{path} not found'}
            errors.append(f'gate {name} missing')

    def gate_fires(gate_key, check_key='real_check'):
        g = gates.get(gate_key, {})
        if 'error' in g:
            return None
        return g.get(check_key, {}).get('gate_fires', None)

    gate_summary = {
        'vocab_accounting_fires':      gate_fires('vocab_accounting'),
        'derangement_fires':           gate_fires('derangement', 'real_check'),
        'expansion_determinism_fires':  gate_fires('expansion_determinism'),
        'all_gates_fire': all(
            gate_fires(k) is True
            for k in ['vocab_accounting', 'derangement', 'expansion_determinism']
        ),
        'dtf_vocab_accounting_fires': (
            gates.get('vocab_accounting', {}).get('dtf_leg', {}).get('dtf_gate_fires', None)),
        'dtf_derangement_fires': (
            gates.get('derangement', {}).get('dtf_leg', {}).get('gate_fires', None)),
        'dtf_expansion_determinism_fires': (
            gates.get('expansion_determinism', {}).get('dtf_leg', {}).get('dtf_gate_fires', None)),
    }

    # ── evaluate artifact ─────────────────────────────────────────────────────
    eval_path = os.path.join(WORK_DIR, 'exp84_evaluate.json')
    if os.path.exists(eval_path):
        evaluate = load_json(eval_path)
        agg_eval = evaluate.get('aggregate', {})
    else:
        evaluate = {'error': 'exp84_evaluate.json not found'}
        agg_eval = {}
        errors.append('evaluate missing')

    # §3 boolean outcomes
    fertility_competitive = agg_eval.get('fertility_competitive_all_seeds', None)
    boundaries_kept       = agg_eval.get('boundaries_kept_all_seeds', None)
    tax_cut               = agg_eval.get('TAX_CUT_all_seeds', None)
    named_outcomes        = agg_eval.get('named_outcomes', [])

    # Third outcomes
    if fertility_competitive is True and boundaries_kept is False:
        third_outcome = 'COMPRESSION_WITHOUT_MORPHOLOGY'
    elif fertility_competitive is False and boundaries_kept is True:
        third_outcome = 'TAX_PERSISTS'
    elif fertility_competitive is False and boundaries_kept is False:
        third_outcome = 'ALLOCATION_FAILS'
    elif fertility_competitive is True and boundaries_kept is True:
        third_outcome = 'TAX_CUT'
    else:
        third_outcome = 'INCONCLUSIVE'

    # ── LM artifacts ─────────────────────────────────────────────────────────
    lm_results = {}
    lm_missing = []
    for arm in ARMS:
        lm_results[arm] = {}
        for s in LM_SEEDS:
            path = os.path.join(WORK_DIR, f'exp84_lm_{arm}_s{s}.json')
            if os.path.exists(path):
                lm_results[arm][str(s)] = load_json(path)
            else:
                lm_results[arm][str(s)] = {'error': f'{path} not found', 'bits_per_char': None}
                lm_missing.append(f'exp84_lm_{arm}_s{s}.json')
    if lm_missing:
        errors.append(f'LM files missing: {lm_missing}')

    def get_bpc(arm, seed):
        v = lm_results[arm].get(str(seed), {}).get('evaluation', {})
        if not v:
            return None
        return v.get('bits_per_char', None)

    # Collect bits/char by arm
    bpc = {}
    for arm in ARMS:
        vals = [get_bpc(arm, s) for s in LM_SEEDS]
        bpc[arm] = {
            'per_seed': {str(s): get_bpc(arm, s) for s in LM_SEEDS},
            'mean': round(mean(vals), 6) if all(v is not None for v in vals) else None,
        }

    # §4 bar: wazn_lm_helps = min(WAZN-BPE, FUSED) bpc < BPE bpc in BOTH seeds
    #         with mean gap ≥ 0.02 bits/char
    bpc_bpe   = [get_bpc('bpe',   s) for s in LM_SEEDS]
    bpc_wazn  = [get_bpc('wazn',  s) for s in LM_SEEDS]
    bpc_fused = [get_bpc('fused', s) for s in LM_SEEDS]

    lm_helps_per_seed = []
    gaps = []
    for i, s in enumerate(LM_SEEDS):
        b, w, fu = bpc_bpe[i], bpc_wazn[i], bpc_fused[i]
        if b is None or w is None or fu is None:
            lm_helps_per_seed.append(None)
            continue
        better = min(w, fu) < b
        gap    = b - min(w, fu)
        lm_helps_per_seed.append(better)
        gaps.append(gap)

    wazn_lm_helps = (
        all(v is True for v in lm_helps_per_seed)
        and mean(gaps) >= 0.02
    ) if all(v is not None for v in lm_helps_per_seed) else None

    no_lm_signal = (
        not wazn_lm_helps
    ) if wazn_lm_helps is not None else None

    lm_summary = {
        'bits_per_char': bpc,
        'bpe_mean_bpc':   bpc['bpe']['mean'],
        'wazn_mean_bpc':  bpc['wazn']['mean'],
        'fused_mean_bpc': bpc['fused']['mean'],
        'lm_helps_per_seed': {str(s): v for s, v in zip(LM_SEEDS, lm_helps_per_seed)},
        'mean_gap_min_wazn_fused_over_bpe': round(mean(gaps), 6) if gaps else None,
        'wazn_lm_helps':  wazn_lm_helps,
        'NO_LM_SIGNAL':   no_lm_signal,
        'bar_def': ('wazn_lm_helps = min(WAZN-BPE, FUSED) bpc < BPE bpc in BOTH seeds '
                    'with mean gap ≥ 0.02 bits/char'),
    }

    # ── Overall verdict ───────────────────────────────────────────────────────
    verdict_parts = []
    if tax_cut is True:
        verdict_parts.append('TAX_CUT=PASS')
    elif tax_cut is False:
        verdict_parts.append(f'TAX_CUT=FAIL ({third_outcome})')
    if wazn_lm_helps is True:
        verdict_parts.append('wazn_lm_helps=PASS')
    elif wazn_lm_helps is False:
        verdict_parts.append('NO_LM_SIGNAL')
    verdict = ' | '.join(verdict_parts) if verdict_parts else 'PENDING'

    results = {
        'meta': {
            'experiment':     'exp84 — FUSED-VOCAB WAZN-BPE + stage-2 LM',
            'registration':   os.path.expanduser(
                '~/qnlp_private_docs/76_exp84_fused_vocab_tokenizer.md'),
            'sole_writer':    'exp84_aggregate.py',
            'errors':         errors,
            'lm_units_missing': lm_missing,
        },
        'corpus': corpus_meta,
        'gates':  {
            'per_gate':   gates,
            'summary':    gate_summary,
        },
        'tokenizer_evaluate': evaluate,
        'lm': lm_summary,
        'bars': {
            'fertility_competitive': fertility_competitive,
            'boundaries_kept':       boundaries_kept,
            'TAX_CUT':               tax_cut,
            'wazn_lm_helps':         wazn_lm_helps,
            'NO_LM_SIGNAL':          no_lm_signal,
            'third_outcome':         third_outcome,
        },
        'named_outcomes_from_evaluate': named_outcomes,
        'verdict': verdict,
    }

    save_json(results, RESULTS_PATH)
    print(f'[aggregate] Written → {RESULTS_PATH}')
    print(f'  verdict: {verdict}')
    if errors:
        print(f'  WARNINGS: {errors}')
    return results


if __name__ == '__main__':
    r = aggregate()
    if r['meta']['errors']:
        print('[aggregate] Some artifacts missing — re-run after all stages complete.')
        sys.exit(1)
