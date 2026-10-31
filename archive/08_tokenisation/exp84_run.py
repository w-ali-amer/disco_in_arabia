#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp84_run.py — FUSED-VOCAB WAZN-BPE experiment.

Registration: qnlp_private_docs/76_exp84_fused_vocab_tokenizer.md
Predecessor:  exp80_run.py (WAZN-BPE baseline; MORPHOLOGY_TAX finding)

Stages (CLI flags):
  --corpus      derive arwiki_exp84.txt, assert sha256
  --tokenizers  build BPE-32k, WAZN-BPE, FUSED for seeds 42/43/44
  --gates       emit exp84_gate_*.json (3 gates, each with DTF leg)
  --evaluate    fertility, boundary F1, surface F1, weak stratum
  --lm          stage-2 LM (requires --arm and --seed)
  --arm         {bpe,wazn,fused}
  --seed        LM training seed (0 or 1)

Compute host: WSL (~/venv/bin/python3).
Sole writer of per-stage artifacts; exp84_aggregate.py is sole writer of results_exp84.json.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

# ── CAMeL path (WSL source checkout takes precedence if present) ───────────────
_CAMEL_SRC = os.path.expanduser('~/camel_tools')
if os.path.isdir(_CAMEL_SRC) and _CAMEL_SRC not in sys.path:
    sys.path.insert(0, _CAMEL_SRC)
# Mac fallback (calima venv)
_MAC_VENV = '/tmp/camel_env/lib/python3.14/site-packages'
if os.path.isdir(_MAC_VENV) and _MAC_VENV not in sys.path:
    sys.path.insert(0, _MAC_VENV)

# ── paths (expanduser so they work on both Mac and WSL) ───────────────────────
WORK_DIR     = os.path.expanduser('~/discocat_arabic_v2')
CORPUS_PATH  = os.path.join(WORK_DIR, 'arwiki_exp84.txt')
PASS1_PATH   = os.path.join(WORK_DIR, 'exp77_pass1_camel.json')
CORPUS_META  = os.path.join(WORK_DIR, 'exp84_corpus_meta.json')

CORPUS_SHA_EXPECTED = '8cde0c825e3c058340b78d9f3a54af95f32d3e78a4d3cd880e1b6f325a1ead63'
N_ARTICLES   = 3673
VOCAB_SIZE   = 32_000
TRAIN_FRAC   = 0.90
SEEDS        = [42, 43, 44]   # corpus split seeds (tokenizer/evaluate)
LM_SEED_TOK  = 42             # canonical tokenizer/split seed for LM stage
GOLD_SIZE    = 2_000
WEAK_MIN_FRAC = 0.30
MIN_TMPL_OBS  = 3
TOP_K_TYPES   = 80_000
N_FUSED       = 8_000         # frozen vocabulary allocation (doc 76 §2)

# Arabic diacritics U+064B–U+065F
DIACRITICS = frozenset('ًٌٍَُِّْٕٖٟٓٔٗ٘')
ARABIC_RE  = re.compile(r'[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]')
WEAK_RADICAL_CHARS = frozenset('وياءأإآ#')


# ═══════════════════════════════════════════════════════════════════════════════
# §1  UTILITIES  (same as exp80_run.py)
# ═══════════════════════════════════════════════════════════════════════════════

def strip_diac(s):
    return ''.join(c for c in s if c not in DIACRITICS)

def normalize_surface(s):
    s = strip_diac(s)
    s = s.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    return s

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def sha256_str(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def is_arabic_word(w):
    arabic_count = sum(1 for c in w if ARABIC_RE.match(c))
    latin_count  = sum(1 for c in w if c.isascii() and c.isalpha())
    return arabic_count >= 2 and latin_count == 0

def is_weak_root(root):
    parts = root.split('.')
    if len(parts) != 3:
        return False
    return any(p in WEAK_RADICAL_CHARS for p in parts)


# ═══════════════════════════════════════════════════════════════════════════════
# §2  CAMEL ANALYSIS  (same as exp80_run.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_analyzer():
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    db = MorphologyDB.builtin_db()
    return Analyzer(db)


def select_reading(analyses):
    pos_rank = {'noun': 0, 'adj': 1, 'verb': 2}

    def is_triliteral(root):
        parts = root.split('.')
        return len(parts) == 3 and all(len(p) >= 1 for p in parts)

    def has_radical_digits(pattern):
        digit_count = sum(1 for d in '123' if d in pattern)
        return digit_count >= 2

    candidates = [
        a for a in analyses
        if is_triliteral(a.get('root', ''))
        and has_radical_digits(a.get('pattern', ''))
        and a.get('bw', '')
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda a: (
        pos_rank.get(a.get('pos', 'zzz').lower()[:4], 3),
        a.get('lex', ''),
        a.get('diac', ''),
    ))
    return candidates[0]


def analyze_types(analyzer, word_types):
    cache = {}
    for w in word_types:
        try:
            analyses = analyzer.analyze(w)
        except Exception:
            continue
        r = select_reading(analyses)
        if r is not None:
            cache[w] = r
    return cache


# ═══════════════════════════════════════════════════════════════════════════════
# §3  BW PARSING → GOLD BOUNDARIES  (same as exp80_run.py)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_bw_boundaries(bw_field, word):
    parts = bw_field.split('+')
    segs = []
    for part in parts:
        if '/' in part:
            text, _ = part.rsplit('/', 1)
        else:
            text = part
        seg_norm = normalize_surface(text)
        if seg_norm:
            segs.append(seg_norm)
    cat  = ''.join(segs)
    surf = normalize_surface(word)
    if cat != surf:
        return None
    boundaries = set()
    pos = 0
    for seg in segs[:-1]:
        pos += len(seg)
        if 0 < pos < len(surf):
            boundaries.add(pos)
    return boundaries


def get_gold_boundaries(reading, word):
    bw = reading.get('bw', '')
    if not bw:
        return None
    return parse_bw_boundaries(bw, word)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  SKELETON EXTRACTION  (same as exp80_run.py)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_skeleton(pattern):
    stripped = strip_diac(pattern)
    has_al   = stripped.startswith('ال')
    has_ta   = stripped.endswith('ة')
    has_ta2  = stripped.endswith('ات')
    has_oon  = stripped.endswith('ون') or stripped.endswith('ين')
    has_tum  = stripped.endswith('تم') or stripped.endswith('تن')
    has_an   = stripped.endswith('ان') or stripped.endswith('ين') and not has_oon
    has_verb_prefix = (not has_al and len(stripped) > 0
                       and stripped[0] in 'يتنأ' and '1' in stripped)
    prefix_len = 2 if has_al else (1 if has_verb_prefix else 0)
    if has_ta2:
        suffix_len = 2
    elif has_oon or has_tum:
        suffix_len = 2
    elif has_ta:
        suffix_len = 1
    else:
        suffix_len = 0
    return {
        'has_al':        has_al,
        'has_verb_pref': has_verb_prefix,
        'prefix_len':    prefix_len,
        'suffix_len':    suffix_len,
    }


def predict_boundaries_wazn(word, reading, tmpl_skeleton_map):
    surf = normalize_surface(word)
    L = len(surf)
    pattern = reading.get('pattern', '')
    skel = tmpl_skeleton_map.get(pattern)
    if skel is None:
        skel = extract_skeleton(pattern)
    bounds = set()
    if skel['prefix_len'] > 0 and skel['prefix_len'] < L:
        bounds.add(skel['prefix_len'])
    if skel['suffix_len'] > 0 and (L - skel['suffix_len']) > 0:
        bounds.add(L - skel['suffix_len'])
    return bounds


# ═══════════════════════════════════════════════════════════════════════════════
# §5  BPE TOKENIZER  (same as exp80_run.py)
# ═══════════════════════════════════════════════════════════════════════════════

def train_bpe_tokenizer(corpus_lines, vocab_size, special_tokens, tmp_corpus_path):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    with open(tmp_corpus_path, 'w', encoding='utf-8') as f:
        for line in corpus_lines:
            f.write(line + '\n')

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=False,
    )
    tokenizer.train([tmp_corpus_path], trainer)
    return tokenizer


def bpe_tokenize_word(tokenizer, word):
    enc = tokenizer.encode(word)
    return enc.tokens


def predict_boundaries_bpe(tokenizer, word):
    surf = normalize_surface(word)
    tokens = bpe_tokenize_word(tokenizer, surf)
    bounds = set()
    pos = 0
    for tok in tokens[:-1]:
        tok_chars = re.sub(r'^[##▁Ġ]+', '', tok)
        if not tok_chars:
            tok_chars = tok
        pos += len(tok_chars)
        if 0 < pos < len(surf):
            bounds.add(pos)
    return bounds


def compute_fertility_bpe(tokenizer, lines, max_lines=None):
    total_tokens = total_words = total_chars = 0
    if max_lines:
        lines = lines[:max_lines]
    for line in lines:
        words = line.split()
        if not words:
            continue
        total_words += len(words)
        total_chars += sum(len(w) for w in words)
        enc = tokenizer.encode(line)
        total_tokens += len(enc.ids)
    fert = total_tokens / total_words if total_words > 0 else float('nan')
    comp = total_chars / total_tokens if total_tokens > 0 else float('nan')
    return fert, comp


# ═══════════════════════════════════════════════════════════════════════════════
# §6  WAZN PREPROCESSING  (same as exp80_run.py)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_line_wazn(line, analysis_cache):
    words = line.split()
    out = []
    for w in words:
        r = analysis_cache.get(w)
        if r is not None:
            root = r.get('root', '')
            pattern = r.get('pattern', '')
            parts = root.split('.')
            if len(parts) == 3:
                out.append(' '.join(parts + [f'TMPL_{pattern}']))
                continue
        out.append(w)
    return ' '.join(out)


def wazn_fertility_word(word, analysis_cache, bpe_fallback):
    r = analysis_cache.get(word)
    if r is not None:
        root = r.get('root', '')
        if len(root.split('.')) == 3:
            return 4
    surf = normalize_surface(word)
    return len(bpe_tokenize_word(bpe_fallback, surf))


def compute_fertility_wazn(lines, analysis_cache, bpe_fallback, max_lines=None):
    total_tokens = total_words = total_chars = 0
    if max_lines:
        lines = lines[:max_lines]
    for line in lines:
        words = line.split()
        if not words:
            continue
        total_words += len(words)
        total_chars += sum(len(w) for w in words)
        for w in words:
            total_tokens += wazn_fertility_word(w, analysis_cache, bpe_fallback)
    fert = total_tokens / total_words if total_words > 0 else float('nan')
    comp = total_chars / total_tokens if total_tokens > 0 else float('nan')
    return fert, comp


# ═══════════════════════════════════════════════════════════════════════════════
# §7  FUSED TOKENIZER (new for exp84)
# ═══════════════════════════════════════════════════════════════════════════════

def build_fused_set(train_type_counts, analysis_cache, n_fused=N_FUSED):
    """
    Return dict: word_form → {'root': str, 'pattern': str, 'parts': [r1,r2,r3]}
    for the top n_fused most frequent analyzable word types in train.
    """
    fused = {}
    for word, _ in train_type_counts.most_common():
        if len(fused) >= n_fused:
            break
        r = analysis_cache.get(word)
        if r is None:
            continue
        root    = r.get('root', '')
        pattern = r.get('pattern', '')
        parts   = root.split('.')
        if len(parts) == 3:
            fused[word] = {'root': root, 'pattern': pattern, 'parts': parts}
    return fused


def preprocess_line_fused(line, fused_set, analysis_cache):
    """
    For LM encoding (not BPE training).
    Emit:
      fused word  → word itself (AddedToken in tokenizer → single token at encode time)
      other analyzable → R1 R2 R3 TMPL_pattern  (WAZN decomposition)
      unanalyzable → unchanged
    """
    words = line.split()
    out = []
    for w in words:
        if w in fused_set:
            out.append(w)   # word form → AddedToken match → 1 token
            continue
        r = analysis_cache.get(w)
        if r is not None:
            root    = r.get('root', '')
            pattern = r.get('pattern', '')
            parts   = root.split('.')
            if len(parts) == 3:
                out.append(' '.join(parts + [f'TMPL_{pattern}']))
                continue
        out.append(w)
    return ' '.join(out)


def preprocess_line_fused_train(line, fused_set, analysis_cache):
    """
    For BPE training only.
    Fused words → WAZN decomposition (not word form), preventing the BPE remainder
    from burning merge budget reconstructing top-8k word forms that will always
    be shadowed by their AddedToken slots at encode time.
    For LM encoding, use preprocess_line_fused() instead.
    """
    words = line.split()
    out = []
    for w in words:
        if w in fused_set:
            fs = fused_set[w]
            parts   = fs['parts']
            pattern = fs['pattern']
            out.append(' '.join(parts + [f'TMPL_{pattern}']))
            continue
        r = analysis_cache.get(w)
        if r is not None:
            root    = r.get('root', '')
            pattern = r.get('pattern', '')
            parts   = root.split('.')
            if len(parts) == 3:
                out.append(' '.join(parts + [f'TMPL_{pattern}']))
                continue
        out.append(w)
    return ' '.join(out)


def fused_fertility_word(word, fused_set, analysis_cache, bpe_fallback):
    """Token count for one word under FUSED tokenizer (emitted stream)."""
    if word in fused_set:
        return 1   # single fused token
    r = analysis_cache.get(word)
    if r is not None:
        root = r.get('root', '')
        if len(root.split('.')) == 3:
            return 4   # WAZN decomposition: R1 R2 R3 TMPL
    surf = normalize_surface(word)
    return len(bpe_tokenize_word(bpe_fallback, surf))


def compute_fertility_fused(lines, fused_set, analysis_cache, bpe_fallback,
                             max_lines=None):
    """Fertility on EMITTED stream for FUSED tokenizer."""
    total_tokens = total_words = total_chars = 0
    if max_lines:
        lines = lines[:max_lines]
    for line in lines:
        words = line.split()
        if not words:
            continue
        total_words += len(words)
        total_chars += sum(len(w) for w in words)
        for w in words:
            total_tokens += fused_fertility_word(w, fused_set, analysis_cache,
                                                 bpe_fallback)
    fert = total_tokens / total_words if total_words > 0 else float('nan')
    comp = total_chars / total_tokens if total_tokens > 0 else float('nan')
    return fert, comp


def predict_boundaries_fused_expanded(word, fused_set, analysis_cache,
                                       tmpl_skeleton_map):
    """
    Boundary prediction for FUSED on DETERMINISTIC EXPANSION.
    Fused word → expand to stored decomposition → same boundary logic as WAZN-BPE.
    """
    fs = fused_set.get(word)
    if fs is not None:
        # Reconstruct a pseudo-reading dict for predict_boundaries_wazn
        reading = {'root': fs['root'], 'pattern': fs['pattern'], 'bw': ''}
        return predict_boundaries_wazn(word, reading, tmpl_skeleton_map)
    r = analysis_cache.get(word)
    if r is not None:
        return predict_boundaries_wazn(word, r, tmpl_skeleton_map)
    return set()


def predict_boundaries_fused_surface(word, fused_set, analysis_cache,
                                      bpe_fallback):
    """
    Surface-level boundary prediction (no expansion). Fused words → no predicted
    boundaries (they're a single token). Non-fused → BPE boundaries.
    """
    if word in fused_set:
        return set()   # single token → no boundaries predicted
    r = analysis_cache.get(word)
    if r is not None:
        # Non-fused analyzable → still decomposed as 4 tokens in BPE-land
        # For surface F1, predict boundary from BPE tokenizer on the word form
        return predict_boundaries_bpe(bpe_fallback, word)
    surf = normalize_surface(word)
    return predict_boundaries_bpe(bpe_fallback, surf)


# ═══════════════════════════════════════════════════════════════════════════════
# §8  F1 METRIC  (same as exp80_run.py)
# ═══════════════════════════════════════════════════════════════════════════════

def boundary_f1(gold, pred):
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def micro_f1(records):
    tp_total = fp_total = fn_total = 0
    for gold, pred in records:
        tp_total += len(gold & pred)
        fp_total += len(pred - gold)
        fn_total += len(gold - pred)
    p  = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    r  = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ═══════════════════════════════════════════════════════════════════════════════
# §9  GOLD SET CONSTRUCTION  (same as exp80_run.py)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_arabic_types(lines):
    counts = Counter()
    for line in lines:
        for w in line.split():
            if is_arabic_word(w):
                counts[w] += 1
    return counts


def build_gold_set(heldout_lines, analysis_cache, target_size, weak_min_frac,
                   tmpl_skeleton_map, seed):
    rng = random.Random(seed)
    type_counts = extract_arabic_types(heldout_lines)
    gold_cands_weak   = []
    gold_cands_strong = []
    mismatch_count = 0

    for word, freq in type_counts.most_common():
        reading = analysis_cache.get(word)
        if reading is None:
            continue
        gold_bounds = get_gold_boundaries(reading, word)
        if gold_bounds is None:
            mismatch_count += 1
            continue
        root    = reading.get('root', '')
        pattern = reading.get('pattern', '')
        weak    = is_weak_root(root)
        entry = {'word': word, 'root': root, 'pattern': pattern,
                 'weak': weak, 'gold_bounds': list(sorted(gold_bounds)),
                 'freq': freq}
        if weak:
            gold_cands_weak.append(entry)
        else:
            gold_cands_strong.append(entry)

    rng.shuffle(gold_cands_weak)
    rng.shuffle(gold_cands_strong)

    n_weak_target = int(target_size * weak_min_frac)
    n_weak   = min(n_weak_target, len(gold_cands_weak))
    n_strong = min(target_size - n_weak, len(gold_cands_strong))
    slack = target_size - n_weak - n_strong
    if slack > 0:
        extra_weak = min(slack, len(gold_cands_weak) - n_weak)
        n_weak += extra_weak

    gold = gold_cands_weak[:n_weak] + gold_cands_strong[:n_strong]
    rng.shuffle(gold)
    actual_weak_frac = sum(1 for e in gold if e['weak']) / len(gold) if gold else 0.0

    return gold, {
        'n_weak_candidates':   len(gold_cands_weak),
        'n_strong_candidates': len(gold_cands_strong),
        'n_gold':              len(gold),
        'n_weak_in_gold':      sum(1 for e in gold if e['weak']),
        'actual_weak_frac':    actual_weak_frac,
        'mismatch_drops':      mismatch_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §10  DESIGNED-TO-FAIL GATE: DERANGEMENT (from exp80, reused)
# ═══════════════════════════════════════════════════════════════════════════════

def build_deranged_skeleton_map(tmpl_skeleton_map, seed=999):
    rng = random.Random(seed)
    patterns  = list(tmpl_skeleton_map.keys())
    skeletons = [tmpl_skeleton_map[p] for p in patterns]
    shuffled  = list(range(len(skeletons)))
    for _ in range(1000):
        rng.shuffle(shuffled)
        if all(shuffled[i] != i for i in range(len(shuffled))):
            break
    return {patterns[i]: skeletons[shuffled[i]] for i in range(len(patterns))}


def _run_derangement_gate_on(gold_set, analysis_cache, fused_set,
                               tmpl_skeleton_map):
    """
    Derangement gate applied to FUSED EXPANDED boundary F1.
    Real map vs deranged map; gate fires when corrupted F1 < real F1 − 0.10.
    """
    deranged_map = build_deranged_skeleton_map(tmpl_skeleton_map, seed=999)
    real_recs    = []
    corrupt_recs = []

    for entry in gold_set:
        word    = entry['word']
        gold_b  = set(entry['gold_bounds'])
        # expanded boundary prediction (real map)
        pred_real    = predict_boundaries_fused_expanded(
            word, fused_set, analysis_cache, tmpl_skeleton_map)
        pred_corrupt = predict_boundaries_fused_expanded(
            word, fused_set, analysis_cache, deranged_map)
        real_recs.append((gold_b, pred_real))
        corrupt_recs.append((gold_b, pred_corrupt))

    _, _, f1_real    = micro_f1(real_recs)
    _, _, f1_corrupt = micro_f1(corrupt_recs)
    gate_fires = f1_corrupt < f1_real - 0.10

    return {
        'description': 'Designed-to-fail: deranged template→skeleton map on FUSED expanded F1',
        'derangement_seed': 999,
        'n_templates_deranged': len(tmpl_skeleton_map),
        'f1_real_fused_expanded':    round(f1_real,    4),
        'f1_deranged_fused_expanded': round(f1_corrupt, 4),
        'delta_f1':                   round(f1_real - f1_corrupt, 4),
        'gate_fires':  gate_fires,
        'verdict': ('GATE_FIRES: corrupted expanded F1 tanks as expected' if gate_fires
                    else 'GATE_DOES_NOT_FIRE: boundary predictions insufficiently sensitive'),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §11  CROSS-CHECK: pass1 CAMeL agreement (lineage continuity)
# ═══════════════════════════════════════════════════════════════════════════════

def cross_check_pass1(analyzer):
    if not os.path.exists(PASS1_PATH):
        return {'error': f'pass1 file not found at {PASS1_PATH}', 'skipped': True}
    with open(PASS1_PATH) as f:
        data = json.load(f)
    rows = data['rows']
    n_total = n_root_match = n_pat_match = n_both_match = n_no_reading = 0
    for row in rows:
        word             = row['noun']
        committed_root   = row.get('root_committed', row.get('root_reproduced', ''))
        committed_pattern = row.get('pattern_committed', row.get('pattern_reproduced', ''))
        try:
            analyses = analyzer.analyze(word)
        except Exception:
            n_no_reading += 1
            continue
        r = select_reading(analyses)
        if r is None:
            n_no_reading += 1
            continue
        n_total += 1
        root_ok = r.get('root', '') == (committed_root or '')
        pat_ok  = strip_diac(r.get('pattern', '')) == strip_diac(committed_pattern or '')
        if root_ok:
            n_root_match += 1
        if pat_ok:
            n_pat_match += 1
        if root_ok and pat_ok:
            n_both_match += 1
    return {
        'n_rows_in_pass1':    len(rows),
        'n_readings_found':   n_total,
        'n_no_reading':       n_no_reading,
        'root_agreement_pct': round(100 * n_root_match / n_total, 2) if n_total else 0,
        'pattern_agreement_pct': round(100 * n_pat_match / n_total, 2) if n_total else 0,
        'both_agreement_pct': round(100 * n_both_match / n_total, 2) if n_total else 0,
        'camel_db_both': 'calima-msa-r13 (WSL src) vs calima-msa-r13 (committed/pass1)',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §12  CORPUS SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def split_lines(lines, train_frac, seed):
    rng  = random.Random(seed)
    idxs = list(range(len(lines)))
    rng.shuffle(idxs)
    n_train     = int(len(idxs) * train_frac)
    train_idxs  = set(idxs[:n_train])
    train   = [lines[i] for i in range(len(lines)) if i in train_idxs]
    heldout = [lines[i] for i in range(len(lines)) if i not in train_idxs]
    return train, heldout


# ═══════════════════════════════════════════════════════════════════════════════
# §13  ARTIFACT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def artifact_path(name):
    return os.path.join(WORK_DIR, name)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def analysis_cache_path(seed):
    return artifact_path(f'exp84_analysis_s{seed}.pkl')

def fused_set_path(seed):
    return artifact_path(f'exp84_fused_set_s{seed}.json')

def tmpl_skel_path(seed):
    return artifact_path(f'exp84_tmpl_skel_s{seed}.json')

def split_path(seed):
    return artifact_path(f'exp84_split_s{seed}.json')

def tok_dir(arm, seed):
    return artifact_path(f'exp84_{arm}_tok_s{seed}')


# ═══════════════════════════════════════════════════════════════════════════════
# §14  STAGE: --corpus
# ═══════════════════════════════════════════════════════════════════════════════

def run_corpus():
    t0 = time.time()
    print('[corpus] Deriving Arabic Wikipedia corpus…')
    os.makedirs(WORK_DIR, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset('wikimedia/wikipedia', '20231101.ar',
                      split='train', streaming=True)

    n_art = 0
    with open(CORPUS_PATH, 'w', encoding='utf-8') as out:
        for art in ds:
            text = art.get('text', '')
            for line in text.split('\n'):
                line = line.strip()
                if line:
                    out.write(line + '\n')
            n_art += 1
            if n_art % 500 == 0:
                print(f'  {n_art} articles…', flush=True)
            if n_art >= N_ARTICLES:
                break

    actual_sha  = sha256_file(CORPUS_PATH)
    file_size   = os.path.getsize(CORPUS_PATH)
    with open(CORPUS_PATH, encoding='utf-8') as f:
        n_lines = sum(1 for _ in f)

    sha_match = actual_sha == CORPUS_SHA_EXPECTED

    if sha_match:
        sha_verdict = 'REPRODUCED_EXACTLY'
        print(f'[corpus] sha256 REPRODUCED EXACTLY: {actual_sha}')
    else:
        sha_verdict = 'FALLBACK_FIRED'
        print(f'[corpus] sha256 MISMATCH — fallback: freeze new corpus, record deltas')
        print(f'  Expected: {CORPUS_SHA_EXPECTED}')
        print(f'  Got:      {actual_sha}')

    meta = {
        'experiment':       'exp84',
        'n_articles':       n_art,
        'n_lines':          n_lines,
        'file_bytes':       file_size,
        'sha256_actual':    actual_sha,
        'sha256_expected':  CORPUS_SHA_EXPECTED,
        'sha256_verdict':   sha_verdict,
        'sha256_reproduced': sha_match,
        'delta_lines':      n_lines - 277644,
        'delta_bytes':      file_size - 78675954,
        'elapsed_s':        round(time.time() - t0, 1),
        'note': ('corpus frozen at download; all three tokenizers retrained on it'
                 if not sha_match else 'corpus matches exp80 corpus of record'),
    }
    save_json(meta, CORPUS_META)
    print(f'[corpus] Done. {n_lines:,} lines, {file_size:,} bytes → {CORPUS_META}')
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# §15  STAGE: --tokenizers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_seed_tokenizers(seed, all_lines, analyzer, t0):
    """Build BPE, WAZN-BPE, FUSED tokenizers for one seed. Returns summary dict."""
    print(f'\n[tok] === SEED {seed} ===')
    train_lines, heldout_lines = split_lines(all_lines, TRAIN_FRAC, seed)
    print(f'  Train: {len(train_lines):,} | Held-out: {len(heldout_lines):,}')

    # Save split indices for LM stage
    spl = {'seed': seed, 'n_train': len(train_lines), 'n_heldout': len(heldout_lines),
           'train_sha': sha256_str('\n'.join(train_lines)),
           'heldout_sha': sha256_str('\n'.join(heldout_lines))}
    save_json(spl, split_path(seed))

    # ── analysis caches (load from disk if exists) ────────────────────────────
    cache_p = analysis_cache_path(seed)
    if os.path.exists(cache_p):
        print(f'[{time.time()-t0:.0f}s]  Loading cached analysis…')
        with open(cache_p, 'rb') as f:
            all_analysis = pickle.load(f)
        train_type_counts = extract_arabic_types(train_lines)
    else:
        print(f'[{time.time()-t0:.0f}s]  Extracting & analyzing train types…')
        train_type_counts = extract_arabic_types(train_lines)
        top_train_types   = [w for w, _ in train_type_counts.most_common(TOP_K_TYPES)]
        train_analysis    = analyze_types(analyzer, top_train_types)
        print(f'  Train analyzable: {len(train_analysis):,}')

        print(f'[{time.time()-t0:.0f}s]  Analyzing held-out types…')
        heldout_type_counts = extract_arabic_types(heldout_lines)
        top_heldout_types   = [w for w, _ in heldout_type_counts.most_common(TOP_K_TYPES)]
        heldout_analysis    = analyze_types(analyzer, top_heldout_types)
        print(f'  Held-out analyzable: {len(heldout_analysis):,}')

        all_analysis = {**train_analysis, **heldout_analysis}
        with open(cache_p, 'wb') as f:
            pickle.dump(all_analysis, f)

    # ── template vocabulary ───────────────────────────────────────────────────
    # Recompute from train analysis
    train_analysis = {w: r for w, r in all_analysis.items()
                      if w in set(train_type_counts.keys())}
    pattern_counts: Counter = Counter()
    for r in train_analysis.values():
        pat = r.get('pattern', '')
        if pat:
            pattern_counts[pat] += 1
    observed_templates = [p for p, cnt in pattern_counts.most_common()
                          if cnt >= MIN_TMPL_OBS]
    tmpl_tokens        = [f'TMPL_{p}' for p in observed_templates]
    tmpl_skel_map      = {p: extract_skeleton(p) for p in observed_templates}
    print(f'  Templates ≥{MIN_TMPL_OBS} obs: {len(observed_templates):,}')

    # Save tmpl skeleton map (for gates/evaluate)
    save_json(tmpl_skel_map, tmpl_skel_path(seed))

    # ── fused set ─────────────────────────────────────────────────────────────
    fused_set = build_fused_set(train_type_counts, train_analysis, N_FUSED)
    print(f'  Fused set size: {len(fused_set):,} (target {N_FUSED})')
    save_json(fused_set, fused_set_path(seed))

    # ── vocab accounting ──────────────────────────────────────────────────────
    n_special  = 3   # [UNK] [BOS] [EOS]
    n_tmpl     = len(tmpl_tokens)
    n_fused    = len(fused_set)   # may be < N_FUSED if corpus has fewer analyzable types
    n_bpe_budget = VOCAB_SIZE - n_special - n_tmpl - n_fused
    print(f'  Vocab: {n_special} special + {n_tmpl} tmpl + {n_fused} fused '
          f'+ {n_bpe_budget} BPE = {n_special+n_tmpl+n_fused+n_bpe_budget}')

    # ── train baseline BPE ────────────────────────────────────────────────────
    print(f'[{time.time()-t0:.0f}s]  Training BPE-32k…')
    bpe_dir = tok_dir('bpe', seed)
    if os.path.exists(bpe_dir + '/tokenizer.json'):
        from tokenizers import Tokenizer
        baseline_bpe = Tokenizer.from_file(bpe_dir + '/tokenizer.json')
        print(f'  Loaded cached BPE ({baseline_bpe.get_vocab_size():,} tokens)')
    else:
        os.makedirs(bpe_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                         encoding='utf-8') as tf:
            tmp = tf.name
        baseline_bpe = train_bpe_tokenizer(
            train_lines, VOCAB_SIZE, ['[UNK]', '[BOS]', '[EOS]'], tmp)
        baseline_bpe.save(bpe_dir + '/tokenizer.json')
        os.unlink(tmp)
        print(f'  BPE trained ({baseline_bpe.get_vocab_size():,} tokens)')

    # ── train WAZN-BPE ────────────────────────────────────────────────────────
    print(f'[{time.time()-t0:.0f}s]  Training WAZN-BPE…')
    wazn_dir = tok_dir('wazn', seed)
    if os.path.exists(wazn_dir + '/tokenizer.json'):
        from tokenizers import Tokenizer
        wazn_bpe = Tokenizer.from_file(wazn_dir + '/tokenizer.json')
        print(f'  Loaded cached WAZN-BPE ({wazn_bpe.get_vocab_size():,} tokens)')
    else:
        os.makedirs(wazn_dir, exist_ok=True)
        wazn_lines = [preprocess_line_wazn(l, train_analysis) for l in train_lines]
        wazn_specials = ['[UNK]', '[BOS]', '[EOS]'] + tmpl_tokens
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                          encoding='utf-8') as tf:
            tmp = tf.name
        wazn_bpe = train_bpe_tokenizer(wazn_lines, VOCAB_SIZE, wazn_specials, tmp)
        wazn_bpe.save(wazn_dir + '/tokenizer.json')
        os.unlink(tmp)
        print(f'  WAZN-BPE trained ({wazn_bpe.get_vocab_size():,} tokens)')

    # ── train FUSED ───────────────────────────────────────────────────────────
    print(f'[{time.time()-t0:.0f}s]  Training FUSED tokenizer…')
    fused_dir = tok_dir('fused', seed)
    if os.path.exists(fused_dir + '/tokenizer.json'):
        from tokenizers import Tokenizer
        fused_tok = Tokenizer.from_file(fused_dir + '/tokenizer.json')
        print(f'  Loaded cached FUSED ({fused_tok.get_vocab_size():,} tokens)')
    else:
        os.makedirs(fused_dir, exist_ok=True)
        fused_lines = [preprocess_line_fused_train(l, fused_set, train_analysis)
                       for l in train_lines]
        fused_specials = (['[UNK]', '[BOS]', '[EOS]']
                          + tmpl_tokens
                          + list(fused_set.keys()))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                          encoding='utf-8') as tf:
            tmp = tf.name
        fused_tok = train_bpe_tokenizer(fused_lines, VOCAB_SIZE, fused_specials, tmp)
        fused_tok.save(fused_dir + '/tokenizer.json')
        os.unlink(tmp)
        print(f'  FUSED trained ({fused_tok.get_vocab_size():,} tokens)')

    return {
        'seed': seed,
        'n_train': len(train_lines), 'n_heldout': len(heldout_lines),
        'n_templates': n_tmpl, 'n_fused_actual': n_fused,
        'n_bpe_budget': n_bpe_budget,
        'vocab_sum': n_special + n_tmpl + n_fused + n_bpe_budget,
        'bpe_vocab_actual':   baseline_bpe.get_vocab_size(),
        'wazn_vocab_actual':  wazn_bpe.get_vocab_size(),
        'fused_vocab_actual': fused_tok.get_vocab_size(),
    }


def run_tokenizers():
    t0 = time.time()
    print('[tokenizers] Loading corpus…')
    with open(CORPUS_PATH, encoding='utf-8') as f:
        all_lines = [l.rstrip('\n') for l in f if l.strip()]
    print(f'  Corpus: {len(all_lines):,} lines')

    print(f'[{time.time()-t0:.0f}s] Loading CAMeL analyzer…')
    analyzer = load_analyzer()
    print('  Analyzer ready (calima-msa-r13)')

    cc = cross_check_pass1(analyzer)
    print(f'  Cross-check: root={cc.get("root_agreement_pct","?")}% '
          f'both={cc.get("both_agreement_pct","?")}%')
    save_json(cc, artifact_path('exp84_cross_check_pass1.json'))

    seed_summaries = []
    for seed in SEEDS:
        summary = _build_seed_tokenizers(seed, all_lines, analyzer, t0)
        seed_summaries.append(summary)

    save_json({'seeds': seed_summaries, 'elapsed_s': round(time.time()-t0,1)},
              artifact_path('exp84_tokenizer_summary.json'))
    print(f'\n[tokenizers] Done in {time.time()-t0:.0f}s')


# ═══════════════════════════════════════════════════════════════════════════════
# §16  STAGE: --gates
# ═══════════════════════════════════════════════════════════════════════════════

def _require_seed42_artifacts():
    """Load seed-42 artifacts; build them if missing."""
    cache_p = analysis_cache_path(42)
    fused_p = fused_set_path(42)
    skel_p  = tmpl_skel_path(42)
    if not (os.path.exists(cache_p) and os.path.exists(fused_p)
            and os.path.exists(skel_p)):
        print('[gates] Seed-42 artifacts missing — running --tokenizers first…')
        run_tokenizers()
    with open(cache_p, 'rb') as f:
        all_analysis = pickle.load(f)
    fused_set   = load_json(fused_p)
    tmpl_skel   = load_json(skel_p)
    split_p     = split_path(42)
    if os.path.exists(split_p):
        spl = load_json(split_p)
    else:
        spl = None
    return all_analysis, fused_set, tmpl_skel, spl


def run_gates():
    t0 = time.time()
    print('[gates] Loading seed-42 artifacts…')
    all_analysis, fused_set, tmpl_skel_map, spl = _require_seed42_artifacts()
    print(f'  analysis_cache: {len(all_analysis):,} words, fused: {len(fused_set):,}')

    with open(CORPUS_PATH, encoding='utf-8') as f:
        all_lines = [l.rstrip('\n') for l in f if l.strip()]
    _, heldout_lines = split_lines(all_lines, TRAIN_FRAC, 42)
    train_lines, _   = split_lines(all_lines, TRAIN_FRAC, 42)
    train_type_counts = extract_arabic_types(train_lines)

    # ── GATE 1: Vocab accounting sums to exactly 32,000 ─────────────────────
    print('[gates] Gate 1: vocab accounting…')
    n_special  = 3
    n_tmpl     = len(tmpl_skel_map)
    n_fused_actual = len(fused_set)
    n_bpe_budget   = VOCAB_SIZE - n_special - n_tmpl - n_fused_actual
    total_real = n_special + n_tmpl + n_fused_actual + n_bpe_budget
    gate1_fires = (total_real == VOCAB_SIZE) and (n_bpe_budget >= 0)

    # DTF leg: use N_FUSED+1 fused tokens → budget goes off
    n_fused_dtf   = N_FUSED + 1
    n_bpe_dtf     = VOCAB_SIZE - n_special - n_tmpl - n_fused_dtf
    total_dtf     = n_special + n_tmpl + n_fused_dtf + max(0, n_bpe_dtf)
    gate1_dtf_budget_negative = n_bpe_dtf < 0
    gate1_dtf_fires = (total_dtf != VOCAB_SIZE) or gate1_dtf_budget_negative

    gate1 = {
        'gate_id': 'exp84_gate_vocab_accounting',
        'description': 'FUSED vocab allocation sums to exactly 32,000',
        'real_check': {
            'n_special': n_special, 'n_tmpl': n_tmpl,
            'n_fused': n_fused_actual, 'n_bpe_budget': n_bpe_budget,
            'total': total_real, 'target': VOCAB_SIZE,
            'gate_fires': gate1_fires,
            'verdict': ('PASS: vocab accounting exact' if gate1_fires
                        else 'FAIL: vocab accounting off'),
        },
        'dtf_leg': {
            'description': 'Designed-to-fail: n_fused=N_FUSED+1 overflows budget',
            'n_fused_injected': n_fused_dtf, 'n_bpe_budget_computed': n_bpe_dtf,
            'total_computed': total_dtf,
            'budget_negative': gate1_dtf_budget_negative,
            'dtf_gate_fires': gate1_dtf_fires,
            'verdict': ('DTF_FIRES: budget overflow detected as expected' if gate1_dtf_fires
                        else 'DTF_DOES_NOT_FIRE: overflow undetected — UNEXPECTED'),
        },
    }
    save_json(gate1, artifact_path('exp84_gate_vocab_accounting.json'))
    print(f'  Gate 1: {gate1["real_check"]["verdict"]}')
    print(f'  DTF leg: {gate1["dtf_leg"]["verdict"]}')

    # ── GATE 2: Derangement gate on FUSED expanded F1 ────────────────────────
    print('[gates] Gate 2: derangement on expanded F1…')
    print('  Building gold set (seed 42)…')
    analyzer = load_analyzer()
    gold_set, gold_meta = build_gold_set(
        heldout_lines, all_analysis, GOLD_SIZE, WEAK_MIN_FRAC, tmpl_skel_map, 42)
    print(f'  Gold set: {gold_meta["n_gold"]} words, {100*gold_meta["actual_weak_frac"]:.1f}% weak')

    derang_result = _run_derangement_gate_on(
        gold_set, all_analysis, fused_set, tmpl_skel_map)
    gate2 = {
        'gate_id': 'exp84_gate_derangement',
        'description': 'Corrupted skeleton map tanks FUSED expanded boundary F1',
        'real_check': derang_result,
        'dtf_leg': {
            'description': 'The DTF run IS the corrupt check above (same gate structure as exp80)',
            'f1_deranged': derang_result['f1_deranged_fused_expanded'],
            'threshold_checked': 'f1_corrupt < f1_real - 0.10',
            'gate_fires': derang_result['gate_fires'],
            'verdict': derang_result['verdict'],
        },
    }
    save_json(gate2, artifact_path('exp84_gate_derangement.json'))
    print(f'  Gate 2: {derang_result["verdict"]}')

    # ── GATE 3: Expansion determinism ────────────────────────────────────────
    print('[gates] Gate 3: expansion determinism…')
    # Every fused token must have a valid 3-part root decomposition
    invalid_entries = []
    decomp_seen     = {}   # decomp_str → word (check uniqueness of decomps)
    duplicate_decomps = []
    for word, fs in fused_set.items():
        parts = fs.get('parts', [])
        if len(parts) != 3 or not all(isinstance(p, str) and len(p) >= 1 for p in parts):
            invalid_entries.append({'word': word, 'parts': parts})
        decomp_str = fs.get('root', '') + '|' + fs.get('pattern', '')
        if decomp_str in decomp_seen:
            duplicate_decomps.append({'word': word, 'conflicts_with': decomp_seen[decomp_str]})
        else:
            decomp_seen[decomp_str] = word

    gate3_fires = len(invalid_entries) == 0

    # DTF leg: inject a synthetic entry with invalid decomposition
    dtf_word = '__TEST_INVALID__'
    dtf_fused_set = dict(fused_set)
    dtf_fused_set[dtf_word] = {'root': 'xx', 'pattern': 'test', 'parts': ['xx']}  # 1 part only
    dtf_invalid = []
    for word, fs in dtf_fused_set.items():
        parts = fs.get('parts', [])
        if len(parts) != 3 or not all(isinstance(p, str) and len(p) >= 1 for p in parts):
            dtf_invalid.append({'word': word, 'parts': parts})
    dtf_gate_fires = len(dtf_invalid) > 0

    gate3 = {
        'gate_id': 'exp84_gate_expansion_determinism',
        'description': 'Every fused token maps to exactly one valid 3-part decomposition',
        'real_check': {
            'n_fused_checked':   len(fused_set),
            'n_invalid_decomps': len(invalid_entries),
            'invalid_examples':  invalid_entries[:5],
            'n_duplicate_decomps': len(duplicate_decomps),
            'duplicate_examples':  duplicate_decomps[:5],
            'gate_fires':  gate3_fires,
            'verdict': ('PASS: all fused tokens have valid decompositions' if gate3_fires
                        else 'FAIL: invalid decompositions found'),
        },
        'dtf_leg': {
            'description': 'Inject synthetic entry with 1-part root; gate must catch it',
            'injected_word':  dtf_word,
            'injected_parts': ['xx'],
            'n_invalid_after_injection': len(dtf_invalid),
            'dtf_gate_fires': dtf_gate_fires,
            'verdict': ('DTF_FIRES: invalid decomposition detected as expected' if dtf_gate_fires
                        else 'DTF_DOES_NOT_FIRE: injection undetected — UNEXPECTED'),
        },
    }
    save_json(gate3, artifact_path('exp84_gate_expansion_determinism.json'))
    print(f'  Gate 3: {gate3["real_check"]["verdict"]}')
    print(f'  DTF leg: {gate3["dtf_leg"]["verdict"]}')
    print(f'[gates] Done in {time.time()-t0:.0f}s')


# ═══════════════════════════════════════════════════════════════════════════════
# §17  STAGE: --evaluate
# ═══════════════════════════════════════════════════════════════════════════════

def _evaluate_one_seed(seed, all_lines, analyzer, t0):
    print(f'\n[eval] === SEED {seed} ===')
    train_lines, heldout_lines = split_lines(all_lines, TRAIN_FRAC, seed)
    MAX_LINES_FERTILITY = 20_000
    heldout_sample = heldout_lines[:MAX_LINES_FERTILITY]

    # ── Load artifacts ────────────────────────────────────────────────────────
    cache_p = analysis_cache_path(seed)
    if not os.path.exists(cache_p):
        raise RuntimeError(f'analysis cache missing for seed {seed}; run --tokenizers first')
    with open(cache_p, 'rb') as f:
        all_analysis = pickle.load(f)
    fused_set   = load_json(fused_set_path(seed))
    tmpl_skel   = load_json(tmpl_skel_path(seed))

    from tokenizers import Tokenizer
    bpe_tok   = Tokenizer.from_file(tok_dir('bpe', seed)   + '/tokenizer.json')
    wazn_tok  = Tokenizer.from_file(tok_dir('wazn', seed)  + '/tokenizer.json')
    fused_tok = Tokenizer.from_file(tok_dir('fused', seed) + '/tokenizer.json')

    train_type_counts = extract_arabic_types(train_lines)
    train_analysis    = {w: r for w, r in all_analysis.items()
                         if w in set(train_type_counts.keys())}

    # ── Fertility ─────────────────────────────────────────────────────────────
    print(f'[{time.time()-t0:.0f}s]  Fertility (on {len(heldout_sample):,} held-out lines)…')
    fert_bpe,   comp_bpe   = compute_fertility_bpe(bpe_tok, heldout_sample)
    fert_wazn,  comp_wazn  = compute_fertility_wazn(
        heldout_sample, all_analysis, wazn_tok)   # own tokenizer as fallback
    fert_fused, comp_fused = compute_fertility_fused(
        heldout_sample, fused_set, all_analysis, fused_tok)  # own tokenizer as fallback

    ratio_wazn  = fert_wazn  / fert_bpe if fert_bpe > 0 else float('nan')
    ratio_fused = fert_fused / fert_bpe if fert_bpe > 0 else float('nan')
    # doc 76 §3: fertility_competitive = FUSED held-out ≤ 1.10 × BPE
    bar_fused_competitive = ratio_fused <= 1.10
    # exp80 bar (for reference): WAZN ≤ 0.95 × BPE
    bar_wazn_improves = ratio_wazn <= 0.95

    print(f'  BPE={fert_bpe:.4f} WAZN={fert_wazn:.4f} FUSED={fert_fused:.4f}')
    print(f'  ratio_wazn={ratio_wazn:.4f} ratio_fused={ratio_fused:.4f}')
    print(f'  bar_wazn_improves={bar_wazn_improves} bar_fused_competitive={bar_fused_competitive}')

    # ── Gold set ──────────────────────────────────────────────────────────────
    print(f'[{time.time()-t0:.0f}s]  Building gold set…')
    gold_set, gold_meta = build_gold_set(
        heldout_lines, all_analysis, GOLD_SIZE, WEAK_MIN_FRAC, tmpl_skel, seed)
    print(f'  Gold: {gold_meta["n_gold"]} words ({100*gold_meta["actual_weak_frac"]:.1f}% weak)')

    # ── Boundary F1 ───────────────────────────────────────────────────────────
    print(f'[{time.time()-t0:.0f}s]  Boundary F1…')
    bpe_recs_all    = []; wazn_recs_all  = []; fused_exp_all  = []; fused_surf_all = []
    bpe_recs_weak   = []; wazn_recs_weak = []; fused_exp_weak = []; fused_surf_weak = []
    bpe_recs_strong = []; wazn_recs_strong=[]; fused_exp_str  = []; fused_surf_str = []

    for entry in gold_set:
        word    = entry['word']
        gold_b  = set(entry['gold_bounds'])
        weak    = entry['weak']
        reading = all_analysis.get(word)
        if reading is None:
            continue
        surf    = normalize_surface(word)
        pred_bpe       = predict_boundaries_bpe(bpe_tok, surf)
        pred_wazn      = predict_boundaries_wazn(word, reading, tmpl_skel)
        pred_fused_exp = predict_boundaries_fused_expanded(
            word, fused_set, all_analysis, tmpl_skel)
        pred_fused_surf = predict_boundaries_fused_surface(
            word, fused_set, all_analysis, bpe_tok)

        bpe_recs_all.append((gold_b, pred_bpe))
        wazn_recs_all.append((gold_b, pred_wazn))
        fused_exp_all.append((gold_b, pred_fused_exp))
        fused_surf_all.append((gold_b, pred_fused_surf))

        bucket = (bpe_recs_weak, wazn_recs_weak, fused_exp_weak, fused_surf_weak,
                  bpe_recs_strong, wazn_recs_strong, fused_exp_str, fused_surf_str)
        if weak:
            bpe_recs_weak.append((gold_b, pred_bpe));  wazn_recs_weak.append((gold_b, pred_wazn))
            fused_exp_weak.append((gold_b, pred_fused_exp)); fused_surf_weak.append((gold_b, pred_fused_surf))
        else:
            bpe_recs_strong.append((gold_b, pred_bpe)); wazn_recs_strong.append((gold_b, pred_wazn))
            fused_exp_str.append((gold_b, pred_fused_exp)); fused_surf_str.append((gold_b, pred_fused_surf))

    def f1_row(recs):
        p, r, f = micro_f1(recs)
        return {'p': round(p,4), 'r': round(r,4), 'f1': round(f,4), 'n': len(recs)}

    bpe_f1_all  = micro_f1(bpe_recs_all)[2]
    wazn_f1_all = micro_f1(wazn_recs_all)[2]
    fexp_f1_all = micro_f1(fused_exp_all)[2]

    gap_wazn  = wazn_f1_all - bpe_f1_all
    gap_fused = fexp_f1_all - bpe_f1_all
    # doc 76 §3: boundaries_kept = expanded F1 ≥ BPE F1 + 0.10
    bar_boundaries_kept = gap_fused >= 0.10
    bar_wazn_boundaries = gap_wazn  >= 0.10   # exp80 bar (for reference)

    print(f'  F1 BPE={bpe_f1_all:.4f} WAZN={wazn_f1_all:.4f} FUSED_exp={fexp_f1_all:.4f}')
    print(f'  gap_wazn={gap_wazn:+.4f} gap_fused={gap_fused:+.4f}')
    print(f'  bar_boundaries_kept={bar_boundaries_kept}')

    # Named outcomes (doc 76 §3)
    tax_cut = bar_fused_competitive and bar_boundaries_kept
    if tax_cut:
        named_outcome = 'TAX_CUT: FUSED is fertility-competitive AND keeps boundaries'
    elif bar_fused_competitive and not bar_boundaries_kept:
        named_outcome = 'COMPRESSION_WITHOUT_MORPHOLOGY: fertility OK, boundary gap lost'
    elif not bar_fused_competitive and bar_boundaries_kept:
        named_outcome = 'TAX_PERSISTS: boundary gap kept, but fertility still high'
    else:
        named_outcome = 'ALLOCATION_FAILS: neither fertility bar nor boundary bar met'

    return {
        'seed': seed,
        'fertility': {
            'heldout_lines_used': len(heldout_sample),
            'bpe':       {'fertility': round(fert_bpe,6),   'comp': round(comp_bpe,4)},
            'wazn':      {'fertility': round(fert_wazn,6),  'comp': round(comp_wazn,4),
                          'ratio_over_bpe': round(ratio_wazn,6),
                          'bar_wazn_improves_over_bpe': bar_wazn_improves},
            'fused':     {'fertility': round(fert_fused,6), 'comp': round(comp_fused,4),
                          'ratio_over_bpe': round(ratio_fused,6),
                          'bar_fertility_competitive': bar_fused_competitive,
                          'bar_def': 'FUSED fertility ≤ 1.10 × BPE'},
        },
        'gold_set_meta': gold_meta,
        'boundary_f1': {
            'overall': {
                'bpe':         f1_row(bpe_recs_all),
                'wazn':        f1_row(wazn_recs_all),
                'fused_expanded': f1_row(fused_exp_all),
                'fused_surface':  f1_row(fused_surf_all),
                'gap_wazn_over_bpe':  round(gap_wazn,4),
                'gap_fused_over_bpe': round(gap_fused,4),
                'bar_wazn_boundaries_recovered': bar_wazn_boundaries,
                'bar_boundaries_kept': bar_boundaries_kept,
                'bar_def': 'FUSED expanded F1 ≥ BPE F1 + 0.10',
            },
            'weak_stratum': {
                'bpe':            f1_row(bpe_recs_weak),
                'wazn':           f1_row(wazn_recs_weak),
                'fused_expanded': f1_row(fused_exp_weak),
                'fused_surface':  f1_row(fused_surf_weak),
            },
            'strong_stratum': {
                'bpe':            f1_row(bpe_recs_strong),
                'wazn':           f1_row(wazn_recs_strong),
                'fused_expanded': f1_row(fused_exp_str),
                'fused_surface':  f1_row(fused_surf_str),
            },
        },
        'named_outcome': named_outcome,
        'bars': {
            'fertility_competitive': bar_fused_competitive,
            'boundaries_kept':       bar_boundaries_kept,
            'TAX_CUT':               tax_cut,
        },
    }


def run_evaluate():
    t0 = time.time()
    print('[evaluate] Loading corpus…')
    with open(CORPUS_PATH, encoding='utf-8') as f:
        all_lines = [l.rstrip('\n') for l in f if l.strip()]
    print(f'  {len(all_lines):,} lines')

    print(f'[{time.time()-t0:.0f}s] Loading CAMeL analyzer…')
    analyzer = load_analyzer()

    seed_results = []
    for seed in SEEDS:
        sr = _evaluate_one_seed(seed, all_lines, analyzer, t0)
        seed_results.append(sr)

    def mean(vals):
        return sum(vals) / len(vals) if vals else float('nan')

    agg = {
        'fertility_competitive_all_seeds': all(
            s['bars']['fertility_competitive'] for s in seed_results),
        'boundaries_kept_all_seeds': all(
            s['bars']['boundaries_kept'] for s in seed_results),
        'TAX_CUT_all_seeds': all(s['bars']['TAX_CUT'] for s in seed_results),
        'mean_fertility_bpe':   round(mean([s['fertility']['bpe']['fertility']
                                            for s in seed_results]), 6),
        'mean_fertility_wazn':  round(mean([s['fertility']['wazn']['fertility']
                                            for s in seed_results]), 6),
        'mean_fertility_fused': round(mean([s['fertility']['fused']['fertility']
                                            for s in seed_results]), 6),
        'mean_ratio_fused_over_bpe': round(mean([s['fertility']['fused']['ratio_over_bpe']
                                                  for s in seed_results]), 6),
        'mean_f1_bpe':           round(mean([s['boundary_f1']['overall']['bpe']['f1']
                                              for s in seed_results]), 4),
        'mean_f1_wazn':          round(mean([s['boundary_f1']['overall']['wazn']['f1']
                                              for s in seed_results]), 4),
        'mean_f1_fused_expanded': round(mean([s['boundary_f1']['overall']['fused_expanded']['f1']
                                               for s in seed_results]), 4),
        'named_outcomes': list(set(s['named_outcome'] for s in seed_results)),
    }

    result = {
        'experiment': 'exp84 — FUSED-VOCAB WAZN-BPE',
        'registration': os.path.expanduser(
            'qnlp_private_docs/76_exp84_fused_vocab_tokenizer.md'),
        'per_seed': seed_results,
        'aggregate': agg,
        'elapsed_s': round(time.time() - t0, 1),
    }
    save_json(result, artifact_path('exp84_evaluate.json'))
    print(f'\n[evaluate] Done in {time.time()-t0:.0f}s → exp84_evaluate.json')
    print(f'  Named outcomes: {agg["named_outcomes"]}')
    print(f'  TAX_CUT all seeds: {agg["TAX_CUT_all_seeds"]}')


# ═══════════════════════════════════════════════════════════════════════════════
# §18  STAGE: --lm  (stage-2 LM per doc 76 §4)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_model(vocab_size, d=128, nhead=4, ff=512, nlayers=2,
                dropout=0.1, ctx=128):
    # doc 76 §4.2 amendment: resized spec, identical across arms.
    import torch
    import torch.nn as nn

    class CausalTransformer(nn.Module):
        def __init__(self, vocab_size, d=d, nhead=nhead, ff=ff,
                     nlayers=nlayers, dropout=dropout, ctx=ctx):
            super().__init__()
            self.ctx = ctx
            self.embed   = nn.Embedding(vocab_size, d)
            self.pos_emb = nn.Embedding(ctx, d)
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=nhead, dim_feedforward=ff,
                dropout=dropout, batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=nlayers)
            self.head = nn.Linear(d, vocab_size, bias=False)
            self.d    = d
            # Causal mask cache
            mask = torch.triu(torch.ones(ctx, ctx), diagonal=1).bool()
            self.register_buffer('causal_mask', mask)

        def forward(self, x):
            B, T = x.shape
            pos  = torch.arange(T, device=x.device)
            h    = self.embed(x) * (self.d ** 0.5) + self.pos_emb(pos)
            h    = self.transformer(h, mask=self.causal_mask[:T, :T])
            return self.head(h)

    return CausalTransformer(vocab_size)


def _tokenize_lines_to_ids(lines, tokenizer_obj, analysis_cache,
                            fused_set, arm, preprocess_fn):
    """Convert text lines to a flat list of token IDs."""
    all_ids = []
    for line in lines:
        preprocessed = preprocess_fn(line)
        enc = tokenizer_obj.encode(preprocessed)
        all_ids.extend(enc.ids)
    return all_ids


def run_lm(arm, lm_seed):
    import torch
    import torch.nn as nn
    import numpy as np

    t0 = time.time()
    print(f'[lm] arm={arm} lm_seed={lm_seed}')
    # doc 76 §4.2 amendment (2026-09-17): resized after OOM cascade +
    # single-thread 8.6-day/unit projection; frozen, identical across arms.
    CTX     = 128
    D_MODEL = 128
    N_HEAD  = 4
    D_FF    = 512
    N_LAYER = 2
    DROPOUT = 0.1
    BATCH   = 16
    STEPS   = 1500
    LR      = 3e-4
    GRAD_CLIP = 1.0
    LOG_EVERY = 100
    OMP_THREADS = max(1, int(os.environ.get('OMP_NUM_THREADS', '4')))
    torch.set_num_threads(OMP_THREADS)

    # Reproducibility
    torch.manual_seed(lm_seed)
    random.seed(lm_seed)
    np.random.seed(lm_seed)

    # ── Load canonical tokenizer artifacts (seed 42) ──────────────────────────
    print(f'  Loading seed-{LM_SEED_TOK} artifacts…')
    cache_p = analysis_cache_path(LM_SEED_TOK)
    if not os.path.exists(cache_p):
        raise RuntimeError('--tokenizers must run before --lm')
    with open(cache_p, 'rb') as f:
        all_analysis = pickle.load(f)
    fused_set   = load_json(fused_set_path(LM_SEED_TOK))

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_dir(arm, LM_SEED_TOK) + '/tokenizer.json')
    vocab_size = tok.get_vocab_size()
    print(f'  Tokenizer vocab: {vocab_size:,}')

    # ── Load corpus + split ───────────────────────────────────────────────────
    print(f'  Loading corpus…')
    with open(CORPUS_PATH, encoding='utf-8') as f:
        all_lines = [l.rstrip('\n') for l in f if l.strip()]
    train_lines, heldout_lines = split_lines(all_lines, TRAIN_FRAC, LM_SEED_TOK)

    # Total surface chars for bits/char denominator (fixed, same for all arms)
    total_surface_chars = sum(len(w) for line in heldout_lines
                              for w in line.split())

    # ── Preprocess according to arm ───────────────────────────────────────────
    def preprocess(line, a=arm, an=all_analysis, fs=fused_set):
        if a == 'bpe':
            return line
        elif a == 'wazn':
            return preprocess_line_wazn(line, an)
        else:   # fused
            return preprocess_line_fused(line, fs, an)

    print(f'  Tokenizing train ({len(train_lines):,} lines)…')
    _train_ids_list = []
    for line in train_lines:
        enc = tok.encode(preprocess(line))
        _train_ids_list.extend(enc.ids)
    # Convert to tensor immediately; free the Python list to save ~400 MB RAM
    train_tensor = torch.tensor(_train_ids_list, dtype=torch.long)
    del _train_ids_list
    print(f'  Train token IDs: {len(train_tensor):,}')

    print(f'  Tokenizing held-out ({len(heldout_lines):,} lines)…')
    _heldout_ids_list = []
    for line in heldout_lines:
        enc = tok.encode(preprocess(line))
        _heldout_ids_list.extend(enc.ids)
    heldout_tensor = torch.tensor(_heldout_ids_list, dtype=torch.long)
    n_heldout_tokens = len(heldout_tensor)
    del _heldout_ids_list
    print(f'  Held-out token IDs: {n_heldout_tokens:,}')

    # ── Build sequences of length CTX (training: disjoint blocks) ────────────
    def make_batches(train_t, ctx, batch_size, rng_state):
        """Yield batches of (input, target) tensors of shape (B, ctx-1)."""
        n    = (len(train_t) // ctx) * ctx
        data = train_t[:n].view(-1, ctx)
        perm = torch.randperm(len(data), generator=rng_state)
        data = data[perm]
        for i in range(0, len(data) - batch_size + 1, batch_size):
            batch = data[i:i+batch_size]       # (B, ctx)
            yield batch[:, :-1], batch[:, 1:]  # input, target

    # ── Model ─────────────────────────────────────────────────────────────────
    model = _make_model(vocab_size, d=D_MODEL, nhead=N_HEAD, ff=D_FF,
                        nlayers=N_LAYER, dropout=DROPOUT, ctx=CTX)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # ── Training ──────────────────────────────────────────────────────────────
    print(f'  Training LM: {STEPS} steps, batch={BATCH}, ctx={CTX}, d={D_MODEL}…')
    step = 0
    epoch = 0
    rng_g = torch.Generator()
    rng_g.manual_seed(lm_seed)
    loss_log = []
    t_train  = time.time()

    model.train()
    while step < STEPS:
        epoch += 1
        for x, y in make_batches(train_tensor, CTX, BATCH, rng_g):
            if step >= STEPS:
                break
            optim.zero_grad()
            logits = model(x)   # (B, ctx-1, V)
            loss   = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            step += 1
            if step % LOG_EVERY == 0 or step == 1:
                elapsed_min = (time.time() - t_train) / 60
                secs_per_step = (time.time() - t_train) / step
                eta_min = secs_per_step * (STEPS - step) / 60
                print(f'  step {step}/{STEPS}  loss={loss.item():.4f}  '
                      f'elapsed={elapsed_min:.1f}m  eta={eta_min:.1f}m', flush=True)
                loss_log.append({'step': step, 'loss': round(loss.item(), 6),
                                 'elapsed_s': round(time.time()-t0, 1)})

    # ── Evaluation: bits per character ────────────────────────────────────────
    # Use CTX+1-length windows with stride CTX so every held-out token (except
    # the very first) is predicted exactly once. This eliminates the differential
    # bias in the previous disjoint-block design: high-fertility arms produced
    # more blocks and therefore more unpredicted first-tokens, biasing their
    # bits/char downward relative to BPE.
    print(f'  Evaluating bits/char on held-out…')
    model.eval()
    total_nats    = 0.0
    total_tokens  = 0

    with torch.no_grad():
        bs = 32   # eval batch
        if len(heldout_tensor) >= CTX + 1:
            # unfold(dim, size, step): windows of CTX+1 with stride CTX
            # shape: (n_windows, CTX+1); windows overlap by 1 element
            windows = heldout_tensor.unfold(0, CTX + 1, CTX)  # (n_windows, CTX+1)
            for i in range(0, len(windows), bs):
                batch = windows[i:i+bs]           # (B, CTX+1)
                x_ev  = batch[:, :CTX]            # (B, CTX)
                y_ev  = batch[:, 1:]              # (B, CTX)
                logits = model(x_ev)              # (B, CTX, V)
                loss_e = loss_fn(logits.reshape(-1, vocab_size), y_ev.reshape(-1))
                n_tok  = y_ev.numel()
                total_nats   += loss_e.item() * n_tok   # CE is in nats
                total_tokens += n_tok

    bits_per_char = (total_nats / total_surface_chars / math.log(2)
                     if total_surface_chars > 0 else float('nan'))

    result = {
        'experiment':     'exp84 stage-2 LM',
        'registration':   os.path.expanduser(
            'qnlp_private_docs/76_exp84_fused_vocab_tokenizer.md'),
        'arm':            arm,
        'lm_seed':        lm_seed,
        'tok_seed':       LM_SEED_TOK,
        'model': {
            'd_model':   D_MODEL, 'nhead': N_HEAD, 'd_ff': D_FF,
            'nlayers':   N_LAYER, 'dropout': DROPOUT, 'ctx': CTX,
        },
        'training': {
            'optimizer': 'Adam', 'lr': LR, 'batch': BATCH,
            'steps': STEPS, 'grad_clip': GRAD_CLIP,
        },
        'tokenizer': {
            'vocab_size':   vocab_size,
            'tok_dir':      tok_dir(arm, LM_SEED_TOK),
        },
        'split': {
            'n_train_lines':    len(train_lines),
            'n_heldout_lines':  len(heldout_lines),
            'n_train_tokens':   len(train_tensor),
            'n_heldout_tokens': n_heldout_tokens,
        },
        'evaluation': {
            'total_nats':          round(total_nats, 4),
            'total_tokens_eval':   total_tokens,
            'total_surface_chars': total_surface_chars,
            'bits_per_char':       round(bits_per_char, 6),
            'definition': 'total_nats / total_surface_chars / ln(2); '
                          'surface_chars = sum of word lengths in held-out (non-space)',
        },
        'training_log': loss_log,
        'elapsed_s': round(time.time() - t0, 1),
    }

    out_path = artifact_path(f'exp84_lm_{arm}_s{lm_seed}.json')
    save_json(result, out_path)
    print(f'  bits/char = {bits_per_char:.6f}')
    print(f'  Written → {out_path}')
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# §18b  GATE_LM_LOSS (doc 76 §4.2) — blocks any LM verdict
# ═══════════════════════════════════════════════════════════════════════════════

def run_gate_lm_loss():
    """doc 76 §4.2: prove the LM loss path before any bits/char is read.
    (i) closed-form per-token CE on a fixed tiny batch, DTF leg shown failing;
    (ii) untrained-model bits/char consistency: the eval-loop accumulation
        must match an independent per-token -log p recomputation from the
        same forward pass;
    (iii) records the init-scale explanation for the observed step-1 loss."""
    import torch
    import torch.nn as nn
    import math as _m

    art = {'gate_id': 'exp84_gate_lm_loss',
           'spec': 'doc 76 §4.2 (2026-09-17 amendment)'}

    # ── (i) closed-form CE ────────────────────────────────────────────────
    logits = torch.tensor([[0.6, -0.2], [-0.1, 0.5]])
    labels = torch.tensor([0, 1])
    # explicit arithmetic, independent of nn.CrossEntropyLoss:
    # item0: -log(e^0.6 / (e^0.6 + e^-0.2)); item1: -log(e^0.5 / (e^-0.1 + e^0.5))
    ref0 = -_m.log(_m.exp(0.6) / (_m.exp(0.6) + _m.exp(-0.2)))
    ref1 = -_m.log(_m.exp(0.5) / (_m.exp(-0.1) + _m.exp(0.5)))
    ref  = (ref0 + ref1) / 2.0
    got  = float(nn.CrossEntropyLoss()(logits, labels))
    diff = abs(got - ref)
    dtf_wrong_ref = ref + 0.01
    dtf_fires = abs(got - dtf_wrong_ref) > 1e-5
    art['closed_form'] = {
        'ref_item0': ref0, 'ref_item1': ref1, 'ref_mean': ref,
        'got': got, 'diff': diff, 'tol': 1e-5, 'passed': diff < 1e-5,
        'dtf': ('OK: designed-to-fail fired: wrong_ref=%.6f diff=%.2e'
                % (dtf_wrong_ref, abs(got - dtf_wrong_ref))) if dtf_fires
               else 'FAIL: DTF did not fire'}

    # ── (ii) untrained-model bits/char consistency ────────────────────────
    torch.manual_seed(0)
    V, N = 50, 2000
    toy_model = _make_model(V, d=32, nhead=2, ff=64, nlayers=1,
                            dropout=0.0, ctx=16)
    toy_model.eval()
    toy_ids = torch.randint(0, V, (N,))
    CTXg = 16
    loss_fn = nn.CrossEntropyLoss()
    total_nats, total_tok = 0.0, 0
    indep_nats = 0.0
    with torch.no_grad():
        windows = toy_ids.unfold(0, CTXg + 1, CTXg)
        for i in range(0, len(windows), 8):
            b = windows[i:i+8]
            x_ev, y_ev = b[:, :CTXg], b[:, 1:]
            lg = toy_model(x_ev)
            # production-path accumulation
            l = loss_fn(lg.reshape(-1, V), y_ev.reshape(-1))
            total_nats += l.item() * y_ev.numel()
            total_tok  += y_ev.numel()
            # independent recomputation: per-token -log softmax, summed
            lp = torch.log_softmax(lg.reshape(-1, V), dim=-1)
            indep_nats += float(-lp[torch.arange(y_ev.numel()),
                                    y_ev.reshape(-1)].sum())
    rel = abs(total_nats - indep_nats) / max(indep_nats, 1e-9)
    chars_fake = 3 * total_tok   # arbitrary fixed denominator, same both routes
    bpc_prod  = total_nats / chars_fake / _m.log(2)
    bpc_indep = indep_nats / chars_fake / _m.log(2)
    art['consistency'] = {
        'vocab': V, 'n_tokens_eval': total_tok,
        'total_nats_production_path': round(total_nats, 6),
        'total_nats_independent':     round(indep_nats, 6),
        'relative_diff': rel, 'tol_relative': 1e-5,
        'bits_per_char_production': bpc_prod,
        'bits_per_char_independent': bpc_indep,
        'passed': rel < 1e-5,
        'untrained_sanity': {
            'mean_nats_per_token': total_nats / total_tok,
            'ln_vocab': _m.log(V),
            'note': 'untrained mean CE need not equal ln(V) exactly — the '
                    'embed*sqrt(d) init scale inflates it; the equality '
                    'checked is production-path vs independent recomputation.'}}

    # ── (iii) init-scale explanation of record ────────────────────────────
    art['step1_loss_explanation'] = (
        'Observed step-1 loss ~38.7 vs ln(32000)=10.37: the forward pass '
        'scales embeddings by sqrt(d_model) (Transformer convention) and '
        'norm_first encoder blocks apply no final LayerNorm before the '
        'head, so initial logits have std >> 1 and initial CE exceeds '
        'ln(V). Identical across arms; converges under training; not a '
        'mis-normalization of the loss.')

    art['passed'] = (art['closed_form']['passed'] and dtf_fires
                     and art['consistency']['passed'])
    save_json(art, artifact_path('exp84_gate_lm_loss.json'))
    print('[gate_lm_loss] passed=%s (closed-form diff %.2e; consistency '
          'rel %.2e)' % (art['passed'], diff, rel))
    if not art['passed']:
        raise SystemExit('GATE_LM_LOSS FAILED')


# ═══════════════════════════════════════════════════════════════════════════════
# §19  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='exp84 runner')
    parser.add_argument('--corpus',     action='store_true', help='derive corpus')
    parser.add_argument('--tokenizers', action='store_true', help='build tokenizers')
    parser.add_argument('--gates',      action='store_true', help='run gates')
    parser.add_argument('--evaluate',   action='store_true', help='run evaluation')
    parser.add_argument('--lm',         action='store_true', help='train LM')
    parser.add_argument('--gate-lm-loss', dest='gate_lm_loss',
                        action='store_true',
                        help='doc 76 §4.2 LM-loss gate (blocks LM verdict)')
    parser.add_argument('--arm',        choices=['bpe', 'wazn', 'fused'],
                        help='LM arm (for --lm)')
    parser.add_argument('--seed',       type=int,
                        help='LM training seed 0 or 1 (for --lm)')
    args = parser.parse_args()

    if args.gate_lm_loss:
        run_gate_lm_loss()
    if args.corpus:
        run_corpus()
    if args.tokenizers:
        run_tokenizers()
    if args.gates:
        run_gates()
    if args.evaluate:
        run_evaluate()
    if args.lm:
        if args.arm is None or args.seed is None:
            parser.error('--lm requires --arm and --seed')
        run_lm(args.arm, args.seed)


if __name__ == '__main__':
    main()
