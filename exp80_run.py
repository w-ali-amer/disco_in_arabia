#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp80_run.py — WAZN-aware tokenizer experiment.

Hypothesis: A tokenizer using Arabic root-and-pattern morphology (WAZN-BPE)
beats standard BPE at matched vocab size on:
  (a) token fertility (tokens/word — lower is better)
  (b) morpheme-boundary recovery (F1 vs CAMeL gold segmentation)

Registration: /Users/wajahath/qnlp_private_docs/72_exp80_wazn_tokenizer.md

CAMeL route: LIVE (camel-tools installed in /tmp/camel_env, calima-msa-r13 DB).
WSL tunnel: DOWN — Mac-local CAMeL is the primary route.
Fallback to committed artifacts (exp77_pass1_camel.json etc.) was not needed.

Sole writer: this script.
Output: /Users/wajahath/discocat_arabic_v2/exp80_results.json
"""

import hashlib
import json
import math
import os
import random
import re
import sys
import time
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

# ── venv path ──────────────────────────────────────────────────────────────────
VENV_SITE = '/tmp/camel_env/lib/python3.14/site-packages'
if VENV_SITE not in sys.path:
    sys.path.insert(0, VENV_SITE)

# ── constants ──────────────────────────────────────────────────────────────────
CORPUS_PATH   = '/tmp/arwiki_raw.txt'
RESULTS_PATH  = '/Users/wajahath/discocat_arabic_v2/exp80_results.json'
PASS1_PATH    = '/Users/wajahath/discocat_arabic_v2/exp77_pass1_camel.json'

VOCAB_SIZE    = 32_000
TRAIN_FRAC    = 0.90
SEEDS         = [42, 43, 44]
GOLD_SIZE     = 2_000
WEAK_MIN_FRAC = 0.30
MIN_TMPL_OBS  = 3      # template pattern must appear ≥ this many times to get a symbol
TOP_K_TYPES   = 80_000  # analyze up to this many unique types per split

# Arabic diacritics U+064B–U+065F, shadda U+0651
DIACRITICS = frozenset(
    'ًٌٍَُِّْ'
    'ٕٖٟٓٔٗ٘'
)
# Arabic script code points (basic block + extended)
ARABIC_RE = re.compile(r'[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]')

WEAK_RADICAL_CHARS = frozenset('وياءأإآ#')


# ═══════════════════════════════════════════════════════════════════════════════
# §1  UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def strip_diac(s: str) -> str:
    return ''.join(c for c in s if c not in DIACRITICS)

def normalize_surface(s: str) -> str:
    """Strip diacritics + normalize alef forms → canonical undiacriticised form."""
    s = strip_diac(s)
    s = s.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    return s

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def is_arabic_word(w: str) -> bool:
    """True if word contains ≥ 2 Arabic script characters and no Latin."""
    arabic_count = sum(1 for c in w if ARABIC_RE.match(c))
    latin_count  = sum(1 for c in w if c.isascii() and c.isalpha())
    return arabic_count >= 2 and latin_count == 0

def is_weak_root(root: str) -> bool:
    """True if any of the three root consonants is a weak radical."""
    parts = root.split('.')
    if len(parts) != 3:
        return False
    return any(p in WEAK_RADICAL_CHARS for p in parts)


# ═══════════════════════════════════════════════════════════════════════════════
# §2  CAMEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def load_analyzer():
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    db = MorphologyDB.builtin_db()
    return Analyzer(db)


def select_reading(analyses: list) -> dict | None:
    """
    Select the first reading with a triliteral root and a digit-bearing pattern.
    Deterministic: stable sort by (pos_rank, lex, diac).
    Returns the analysis dict or None.
    """
    pos_rank = {'noun': 0, 'adj': 1, 'verb': 2}

    def is_triliteral(root: str) -> bool:
        parts = root.split('.')
        return len(parts) == 3 and all(len(p) >= 1 for p in parts)

    def has_radical_digits(pattern: str) -> bool:
        # Require at least two numbered slots; hollow/defective roots have '1'+'3' but no '2'
        # (the middle or final radical is absorbed by the template — see exp80 §1 design notes)
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


def analyze_types(analyzer, word_types: list[str]) -> dict:
    """
    Run CAMeL analysis on each word type. Returns dict: word → reading_dict.
    Words with no triliteral reading are absent from the dict.
    """
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
# §3  BW PARSING → GOLD BOUNDARIES
# ═══════════════════════════════════════════════════════════════════════════════

def parse_bw_boundaries(bw_field: str, word: str) -> set | None:
    """
    Parse CAMeL bw field into morpheme boundaries (character positions in surface).
    A boundary position i means "there is a morpheme break BEFORE character i".
    Returns a set of int positions, or None on alignment mismatch.
    Positions at 0 and len(word) are never included.
    """
    parts = bw_field.split('+')
    segs = []
    for part in parts:
        if '/' in part:
            text, _ = part.rsplit('/', 1)
        else:
            text = part
        seg_norm = normalize_surface(text)
        if seg_norm:  # skip empty (pure-diacritic segments)
            segs.append(seg_norm)

    cat = ''.join(segs)
    surf = normalize_surface(word)

    if cat != surf:
        return None  # alignment mismatch → drop

    boundaries = set()
    pos = 0
    for seg in segs[:-1]:
        pos += len(seg)
        if 0 < pos < len(surf):
            boundaries.add(pos)
    return boundaries


def get_gold_boundaries(reading: dict, word: str) -> set | None:
    """Extract gold boundary set from a CAMeL reading dict."""
    bw = reading.get('bw', '')
    if not bw:
        return None
    return parse_bw_boundaries(bw, word)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  SKELETON EXTRACTION (for WAZN-BPE boundary prediction)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_skeleton(pattern: str) -> dict:
    """
    Extract boundary skeleton from a CAMeL pattern string.
    Skeleton = rules for predicting morpheme boundaries on the surface form.
    Returns a dict usable by predict_boundaries_wazn().
    """
    stripped = strip_diac(pattern)  # remove diacritics for structure inspection

    has_al   = stripped.startswith('ال')
    has_ta   = stripped.endswith('ة')
    has_ta2  = stripped.endswith('ات')   # plural suffix
    has_oon  = stripped.endswith('ون') or stripped.endswith('ين')
    has_tum  = stripped.endswith('تم') or stripped.endswith('تن')
    has_an   = stripped.endswith('ان') or stripped.endswith('ين') and not has_oon
    has_verb_prefix = not has_al and len(stripped) > 0 and stripped[0] in 'يتنأ' and '1' in stripped

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


def predict_boundaries_wazn(word: str, reading: dict, tmpl_skeleton_map: dict) -> set:
    """
    Predict boundary positions for WAZN-BPE using the template skeleton.
    tmpl_skeleton_map: pattern_string → skeleton_dict  (can be deranged for gate test).
    """
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
# §5  BPE TOKENIZER TRAINING & INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def train_bpe_tokenizer(corpus_lines: list[str], vocab_size: int,
                        special_tokens: list[str],
                        tmp_corpus_path: str) -> object:
    """Train HF BPE tokenizer. Returns fitted Tokenizer object."""
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


def bpe_tokenize_word(tokenizer, word: str) -> list[str]:
    """Tokenize a single word (no surrounding spaces) with a BPE tokenizer."""
    enc = tokenizer.encode(word)
    return enc.tokens


def predict_boundaries_bpe(tokenizer, word: str) -> set:
    """
    Predict boundary positions for the BPE baseline.
    Each token boundary → character position in normalized surface.
    """
    surf = normalize_surface(word)
    tokens = bpe_tokenize_word(tokenizer, surf)
    bounds = set()
    pos = 0
    for tok in tokens[:-1]:
        # Strip any continuation prefix added by the pre-tokenizer
        tok_chars = re.sub(r'^[##▁Ġ]+', '', tok)
        if not tok_chars:
            tok_chars = tok  # fallback
        pos += len(tok_chars)
        if 0 < pos < len(surf):
            bounds.add(pos)
    return bounds


def compute_fertility(tokenizer, lines: list[str], max_lines: int = None) -> tuple:
    """
    Fertility = total tokens / total whitespace-words on the given lines.
    Compression ratio = total surface chars / total tokens (chars per token).
    Returns (fertility, compression_ratio).
    """
    total_tokens = 0
    total_words  = 0
    total_chars  = 0
    if max_lines:
        lines = lines[:max_lines]
    for line in lines:
        words = line.split()
        if not words:
            continue
        total_words += len(words)
        total_chars += sum(len(w) for w in words)
        enc = tokenizer.encode(line)
        # subtract special tokens (BOS, EOS, UNK are not added by default in HF BPE)
        total_tokens += len(enc.ids)
    fert = total_tokens / total_words if total_words > 0 else float('nan')
    comp = total_chars / total_tokens if total_tokens > 0 else float('nan')
    return fert, comp


# ═══════════════════════════════════════════════════════════════════════════════
# §6  WAZN-BPE CORPUS PRE-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_line_wazn(line: str, analysis_cache: dict) -> str:
    """
    Replace each analyzable word in `line` with its WAZN decomposition.
    Analyzable word: R1 R2 R3 TMPL_<pattern>
    Unanalyzable words are passed through unchanged.
    """
    words = line.split()
    out = []
    for w in words:
        r = analysis_cache.get(w)
        if r is not None:
            root = r.get('root', '')
            pattern = r.get('pattern', '')
            parts = root.split('.')
            if len(parts) == 3:
                tmpl_tok = f'TMPL_{pattern}'
                # emit: R1 R2 R3 TMPL
                out.append(' '.join(parts + [tmpl_tok]))
                continue
        out.append(w)  # fallback
    return ' '.join(out)


def wazn_fertility_word(word: str, analysis_cache: dict, bpe_tokenizer) -> int:
    """
    Return number of WAZN-BPE tokens for a single word.
    Analyzable → 4 tokens; else → BPE token count.
    """
    r = analysis_cache.get(word)
    if r is not None:
        root = r.get('root', '')
        if len(root.split('.')) == 3:
            return 4
    # fallback BPE
    surf = normalize_surface(word)
    tokens = bpe_tokenize_word(bpe_tokenizer, surf)
    return len(tokens)


def compute_fertility_wazn(lines: list[str], analysis_cache: dict,
                           bpe_fallback: object, max_lines: int = None) -> tuple:
    """Fertility and compression ratio for WAZN-BPE on given lines.
    Returns (fertility, compression_ratio).
    """
    total_tokens = 0
    total_words  = 0
    total_chars  = 0
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
# §7  F1 METRIC
# ═══════════════════════════════════════════════════════════════════════════════

def boundary_f1(gold: set, pred: set) -> tuple[float, float, float]:
    """Precision, Recall, F1 on boundary position sets."""
    if not gold and not pred:
        return 1.0, 1.0, 1.0  # both empty → perfect (no boundaries needed)
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def micro_f1(records: list[tuple[set, set]]) -> tuple[float, float, float]:
    """Micro-average P, R, F1 over many (gold, pred) pairs."""
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
# §8  GOLD SET CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_arabic_types(lines: list[str]) -> Counter:
    """Count Arabic word types from a list of text lines."""
    counts = Counter()
    for line in lines:
        for w in line.split():
            if is_arabic_word(w):
                counts[w] += 1
    return counts


def build_gold_set(heldout_lines: list[str], analysis_cache: dict,
                   target_size: int, weak_min_frac: float,
                   tmpl_skeleton_map: dict, seed: int) -> list[dict]:
    """
    Build gold set: up to target_size word types from held-out,
    stratified so weak-radical words ≥ weak_min_frac.

    Each gold entry: {word, root, pattern, weak, gold_bounds, is_analyzable}
    """
    rng = random.Random(seed)
    type_counts = extract_arabic_types(heldout_lines)

    # Only keep types that have a CAMeL reading and parseable gold boundaries
    gold_candidates_weak   = []
    gold_candidates_strong = []
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

        entry = {
            'word':       word,
            'root':       root,
            'pattern':    pattern,
            'weak':       weak,
            'gold_bounds': list(sorted(gold_bounds)),
            'freq':        freq,
        }
        if weak:
            gold_candidates_weak.append(entry)
        else:
            gold_candidates_strong.append(entry)

    rng.shuffle(gold_candidates_weak)
    rng.shuffle(gold_candidates_strong)

    # Strategy: take exactly weak_min_frac × target from weak (or fewer if not available),
    # fill the rest from strong, then top-up with more weak if strong runs short.
    n_weak_target = int(target_size * weak_min_frac)           # floor: 30% of 2000 = 600
    n_weak   = min(n_weak_target, len(gold_candidates_weak))   # take up to 600 weak
    n_strong = min(target_size - n_weak, len(gold_candidates_strong))
    # If strong runs short, allow more weak (up to target_size total)
    slack = target_size - n_weak - n_strong
    if slack > 0:
        extra_weak = min(slack, len(gold_candidates_weak) - n_weak)
        n_weak += extra_weak

    gold = gold_candidates_weak[:n_weak] + gold_candidates_strong[:n_strong]
    rng.shuffle(gold)

    actual_weak_frac = sum(1 for e in gold if e['weak']) / len(gold) if gold else 0.0

    return gold, {
        'n_weak_candidates':   len(gold_candidates_weak),
        'n_strong_candidates': len(gold_candidates_strong),
        'n_gold':              len(gold),
        'n_weak_in_gold':      sum(1 for e in gold if e['weak']),
        'actual_weak_frac':    actual_weak_frac,
        'mismatch_drops':      mismatch_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §9  DESIGNED-TO-FAIL GATE
# ═══════════════════════════════════════════════════════════════════════════════

def build_deranged_skeleton_map(tmpl_skeleton_map: dict, seed: int = 999) -> dict:
    """
    Create a deranged (permuted) template→skeleton map for the gate test.
    Each template maps to a DIFFERENT skeleton (by construction).
    Derangement: no template maps to its own skeleton.
    """
    rng = random.Random(seed)
    patterns = list(tmpl_skeleton_map.keys())
    skeletons = [tmpl_skeleton_map[p] for p in patterns]

    # Shuffle skeletons until no pattern maps to its own
    shuffled = list(range(len(skeletons)))
    max_tries = 1000
    for _ in range(max_tries):
        rng.shuffle(shuffled)
        if all(shuffled[i] != i for i in range(len(shuffled))):
            break

    deranged = {}
    for i, pat in enumerate(patterns):
        deranged[pat] = skeletons[shuffled[i]]
    return deranged


def run_gate(gold_set: list[dict], analysis_cache: dict,
             real_skeleton_map: dict) -> dict:
    """
    Designed-to-fail gate: WAZN-BPE with a deranged template→skeleton map.
    Expected: F1 tanks compared to real WAZN-BPE.
    Gate PASSES (self-check validates) if corrupted F1 < real F1 − 0.10.
    """
    deranged_map = build_deranged_skeleton_map(real_skeleton_map, seed=999)

    real_records    = []
    corrupt_records = []

    for entry in gold_set:
        word    = entry['word']
        reading = analysis_cache.get(word)
        if reading is None:
            continue
        gold_bounds = set(entry['gold_bounds'])

        pred_real    = predict_boundaries_wazn(word, reading, real_skeleton_map)
        pred_corrupt = predict_boundaries_wazn(word, reading, deranged_map)

        real_records.append((gold_bounds, pred_real))
        corrupt_records.append((gold_bounds, pred_corrupt))

    _, _, f1_real    = micro_f1(real_records)
    _, _, f1_corrupt = micro_f1(corrupt_records)

    gate_fires = f1_corrupt < f1_real - 0.10

    return {
        'description': 'Designed-to-fail: deranged template→skeleton map',
        'derangement_seed': 999,
        'n_templates_deranged': len(real_skeleton_map),
        'f1_real_wazn_bpe':    round(f1_real,    4),
        'f1_deranged_wazn_bpe': round(f1_corrupt, 4),
        'delta_f1':            round(f1_real - f1_corrupt, 4),
        'gate_fires':          gate_fires,
        'verdict': 'GATE_FIRES: corrupted F1 tanks as expected' if gate_fires
                   else 'GATE_DOES_NOT_FIRE: boundary predictions insufficiently sensitive',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §10  CROSS-CHECK: Mac CAMeL vs committed pass1 analysis
# ═══════════════════════════════════════════════════════════════════════════════

def cross_check_pass1(analyzer) -> dict:
    """
    Run Mac CAMeL on 543 exp77 pass1 words; compare root + pattern to committed.
    Both use calima-msa-r13. Reports agreement rate.
    """
    with open(PASS1_PATH) as f:
        data = json.load(f)
    rows = data['rows']

    n_total   = 0
    n_root_match = 0
    n_pat_match  = 0
    n_both_match = 0
    n_no_reading = 0

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

        mac_root    = r.get('root', '')
        mac_pattern = r.get('pattern', '')

        n_total      += 1
        root_ok = mac_root == (committed_root or '')
        committed_pattern_s = committed_pattern or ''
        pat_ok  = strip_diac(mac_pattern) == strip_diac(committed_pattern_s)
        if root_ok:       n_root_match += 1
        if pat_ok:        n_pat_match  += 1
        if root_ok and pat_ok: n_both_match += 1

    return {
        'n_rows_in_pass1':    len(rows),
        'n_readings_found':   n_total,
        'n_no_reading':       n_no_reading,
        'root_agreement_pct': round(100 * n_root_match / n_total, 2) if n_total else 0,
        'pattern_agreement_pct': round(100 * n_pat_match / n_total, 2) if n_total else 0,
        'both_agreement_pct': round(100 * n_both_match / n_total, 2) if n_total else 0,
        'camel_db_both': 'calima-msa-r13 (Mac) vs calima-msa-r13 (committed/WSL)',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §11  MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def split_lines(lines: list[str], train_frac: float, seed: int) -> tuple:
    rng = random.Random(seed)
    idxs = list(range(len(lines)))
    rng.shuffle(idxs)
    n_train = int(len(idxs) * train_frac)
    train_idxs = set(idxs[:n_train])
    train = [lines[i] for i in range(len(lines)) if i in train_idxs]
    heldout = [lines[i] for i in range(len(lines)) if i not in train_idxs]
    return train, heldout


def main():
    t0 = time.time()

    # ── Load corpus ────────────────────────────────────────────────────────────
    print(f'[{time.time()-t0:.1f}s] Loading corpus...')
    corpus_sha = sha256_file(CORPUS_PATH)
    assert corpus_sha == '8cde0c825e3c058340b78d9f3a54af95f32d3e78a4d3cd880e1b6f325a1ead63', \
        f'Corpus SHA mismatch: {corpus_sha}'
    with open(CORPUS_PATH, encoding='utf-8') as f:
        all_lines = [l.rstrip('\n') for l in f if l.strip()]
    print(f'  Corpus: {len(all_lines):,} lines, sha256 verified.')

    # ── Load CAMeL analyzer ────────────────────────────────────────────────────
    print(f'[{time.time()-t0:.1f}s] Loading CAMeL analyzer...')
    analyzer = load_analyzer()
    print(f'  Analyzer ready (calima-msa-r13, Mac-local).')

    # ── Cross-check pass1 ──────────────────────────────────────────────────────
    print(f'[{time.time()-t0:.1f}s] Running cross-check on exp77 pass1 words...')
    cc = cross_check_pass1(analyzer)
    print(f'  Cross-check: root={cc["root_agreement_pct"]}% '
          f'pattern={cc["pattern_agreement_pct"]}% '
          f'both={cc["both_agreement_pct"]}% '
          f'(n={cc["n_readings_found"]}, no_reading={cc["n_no_reading"]})')

    # ── Per-seed runs ──────────────────────────────────────────────────────────
    seed_results = []

    for seed in SEEDS:
        print(f'\n[{time.time()-t0:.1f}s] === SEED {seed} ===')

        # Split
        train_lines, heldout_lines = split_lines(all_lines, TRAIN_FRAC, seed)
        train_sha   = sha256_str('\n'.join(train_lines))
        heldout_sha = sha256_str('\n'.join(heldout_lines))
        print(f'  Train: {len(train_lines):,} lines | Held-out: {len(heldout_lines):,} lines')

        # ── Unique Arabic types (train split for WAZN vocab) ──────────────────
        print(f'[{time.time()-t0:.1f}s]  Extracting train types...')
        train_type_counts = extract_arabic_types(train_lines)
        # Analyze top-K most frequent types
        top_train_types = [w for w, _ in train_type_counts.most_common(TOP_K_TYPES)]
        print(f'  Unique Arabic types in train: {len(train_type_counts):,}, '
              f'analyzing top {len(top_train_types):,}...')
        train_analysis = analyze_types(analyzer, top_train_types)
        print(f'  Analyzable train types: {len(train_analysis):,} '
              f'({100*len(train_analysis)/len(top_train_types):.1f}%)')

        # ── Unique Arabic types (heldout split for gold) ──────────────────────
        print(f'[{time.time()-t0:.1f}s]  Extracting held-out types...')
        heldout_type_counts = extract_arabic_types(heldout_lines)
        top_heldout_types = [w for w, _ in heldout_type_counts.most_common(TOP_K_TYPES)]
        print(f'  Unique Arabic types in held-out: {len(heldout_type_counts):,}, '
              f'analyzing top {len(top_heldout_types):,}...')
        heldout_analysis = analyze_types(analyzer, top_heldout_types)
        print(f'  Analyzable held-out types: {len(heldout_analysis):,} '
              f'({100*len(heldout_analysis)/len(top_heldout_types):.1f}%)')

        # Combined cache (for fertility computation on held-out words)
        all_analysis = {**train_analysis, **heldout_analysis}

        # ── Build template vocabulary ──────────────────────────────────────────
        print(f'[{time.time()-t0:.1f}s]  Building template vocabulary...')
        pattern_counts: Counter = Counter()
        for r in train_analysis.values():
            pat = r.get('pattern', '')
            if pat:
                pattern_counts[pat] += 1

        observed_templates = [
            pat for pat, cnt in pattern_counts.most_common()
            if cnt >= MIN_TMPL_OBS
        ]
        tmpl_tokens = [f'TMPL_{pat}' for pat in observed_templates]
        print(f'  Unique patterns in train: {len(pattern_counts):,}, '
              f'templates with ≥{MIN_TMPL_OBS} obs: {len(observed_templates):,}')

        # Template → skeleton map (real)
        tmpl_skeleton_map: dict = {
            pat: extract_skeleton(pat) for pat in observed_templates
        }

        # Vocab accounting
        n_special  = 3   # [UNK], [BOS], [EOS]
        n_tmpl     = len(tmpl_tokens)
        n_rad_approx = 29  # Arabic consonants (determined empirically below)
        bpe_budget = VOCAB_SIZE - n_special - n_tmpl
        wazn_bpe_vocab_accounting = {
            'total_target':        VOCAB_SIZE,
            'n_special_tokens':    n_special,
            'n_template_tokens':   n_tmpl,
            'bpe_budget':          bpe_budget,
            'n_radical_syms_approx': n_rad_approx,
            'note': 'Radical symbols (Arabic consonants) appear in BPE vocab naturally; '
                    'not separately reserved. Template tokens added as special tokens.',
        }
        print(f'  Vocab: {n_tmpl} templates + {n_rad_approx} radical chars + '
              f'{bpe_budget} BPE budget = {VOCAB_SIZE} total target')

        # ── Train baseline BPE ────────────────────────────────────────────────
        print(f'[{time.time()-t0:.1f}s]  Training baseline BPE (vocab={VOCAB_SIZE})...')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                         encoding='utf-8') as tf:
            tmp_bpe_corpus = tf.name
            for line in train_lines:
                tf.write(line + '\n')

        baseline_bpe = train_bpe_tokenizer(
            corpus_lines=train_lines,
            vocab_size=VOCAB_SIZE,
            special_tokens=['[UNK]', '[BOS]', '[EOS]'],
            tmp_corpus_path=tmp_bpe_corpus,
        )
        actual_baseline_vocab = baseline_bpe.get_vocab_size()
        print(f'  Baseline BPE vocab size: {actual_baseline_vocab:,}')

        # ── Train WAZN-BPE ────────────────────────────────────────────────────
        print(f'[{time.time()-t0:.1f}s]  Preprocessing corpus for WAZN-BPE...')
        wazn_train_lines = [
            preprocess_line_wazn(line, train_analysis)
            for line in train_lines
        ]

        print(f'[{time.time()-t0:.1f}s]  Training WAZN-BPE '
              f'(budget={bpe_budget}, {n_tmpl} templates as specials)...')
        wazn_special_tokens = ['[UNK]', '[BOS]', '[EOS]'] + tmpl_tokens
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                         encoding='utf-8') as tf:
            tmp_wazn_corpus = tf.name
        wazn_bpe = train_bpe_tokenizer(
            corpus_lines=wazn_train_lines,
            vocab_size=VOCAB_SIZE,   # matched to baseline per §1 hypothesis
            special_tokens=wazn_special_tokens,
            tmp_corpus_path=tmp_wazn_corpus,
        )
        actual_wazn_vocab_bpe_part = wazn_bpe.get_vocab_size()
        actual_wazn_total_vocab = actual_wazn_vocab_bpe_part  # templates are within specials
        print(f'  WAZN-BPE vocab size (incl. templates): {actual_wazn_total_vocab:,}')

        # Count radical symbols actually observed in wazn train
        radical_chars_observed = set()
        for r in train_analysis.values():
            root = r.get('root', '')
            for part in root.split('.'):
                if part:
                    radical_chars_observed.add(part)
        wazn_bpe_vocab_accounting['n_radical_syms_observed'] = len(radical_chars_observed)
        wazn_bpe_vocab_accounting['actual_wazn_bpe_vocab'] = actual_wazn_total_vocab

        # ── Fertility on held-out ─────────────────────────────────────────────
        print(f'[{time.time()-t0:.1f}s]  Computing fertility on held-out...')
        MAX_LINES_FERTILITY = 20_000  # cap for speed
        heldout_sample = heldout_lines[:MAX_LINES_FERTILITY]

        fert_baseline, comp_baseline = compute_fertility(baseline_bpe, heldout_sample)
        fert_wazn, comp_wazn = compute_fertility_wazn(
            heldout_sample, all_analysis, wazn_bpe   # wazn_bpe is the matched-vocab fallback
        )
        fertility_ratio = fert_wazn / fert_baseline if fert_baseline > 0 else float('nan')
        bar_fertility = fertility_ratio <= 0.95

        print(f'  Fertility — BPE: {fert_baseline:.4f} | WAZN-BPE: {fert_wazn:.4f} '
              f'| ratio: {fertility_ratio:.4f} | bar passes: {bar_fertility}')

        # ── Build gold set ─────────────────────────────────────────────────────
        print(f'[{time.time()-t0:.1f}s]  Building gold set (target={GOLD_SIZE})...')
        gold_set, gold_meta = build_gold_set(
            heldout_lines=heldout_lines,
            analysis_cache=heldout_analysis,
            target_size=GOLD_SIZE,
            weak_min_frac=WEAK_MIN_FRAC,
            tmpl_skeleton_map=tmpl_skeleton_map,
            seed=seed,
        )
        print(f'  Gold set: {gold_meta["n_gold"]} words '
              f'({100*gold_meta["actual_weak_frac"]:.1f}% weak, '
              f'target ≥{100*WEAK_MIN_FRAC:.0f}%)')

        # Check train overlap
        train_types_set = set(train_type_counts.keys())
        n_overlap = sum(1 for e in gold_set if e['word'] in train_types_set)
        train_overlap_frac = n_overlap / len(gold_set) if gold_set else 0.0

        # ── Boundary F1 computation ─────────────────────────────────────────────
        print(f'[{time.time()-t0:.1f}s]  Computing boundary F1...')

        bpe_records_all    = []
        wazn_records_all   = []
        bpe_records_weak   = []
        wazn_records_weak  = []
        bpe_records_strong = []
        wazn_records_strong = []

        n_gold_valid = 0

        for entry in gold_set:
            word        = entry['word']
            gold_bounds = set(entry['gold_bounds'])
            weak        = entry['weak']
            reading     = heldout_analysis.get(word)
            if reading is None:
                continue

            # BPE baseline boundaries
            surf = normalize_surface(word)
            pred_bpe  = predict_boundaries_bpe(baseline_bpe, surf)
            # WAZN boundaries from skeleton
            pred_wazn = predict_boundaries_wazn(word, reading, tmpl_skeleton_map)

            bpe_records_all.append((gold_bounds, pred_bpe))
            wazn_records_all.append((gold_bounds, pred_wazn))

            if weak:
                bpe_records_weak.append((gold_bounds, pred_bpe))
                wazn_records_weak.append((gold_bounds, pred_wazn))
            else:
                bpe_records_strong.append((gold_bounds, pred_bpe))
                wazn_records_strong.append((gold_bounds, pred_wazn))
            n_gold_valid += 1

        # Overall F1
        bpe_p_all, bpe_r_all, bpe_f1_all       = micro_f1(bpe_records_all)
        wazn_p_all, wazn_r_all, wazn_f1_all     = micro_f1(wazn_records_all)

        # Stratum F1
        bpe_p_w, bpe_r_w, bpe_f1_w             = micro_f1(bpe_records_weak)
        wazn_p_w, wazn_r_w, wazn_f1_w           = micro_f1(wazn_records_weak)
        bpe_p_s, bpe_r_s, bpe_f1_s             = micro_f1(bpe_records_strong)
        wazn_p_s, wazn_r_s, wazn_f1_s           = micro_f1(wazn_records_strong)

        boundary_gap = wazn_f1_all - bpe_f1_all
        bar_boundary = boundary_gap >= 0.10

        print(f'  Boundary F1 — BPE: {bpe_f1_all:.4f} | WAZN-BPE: {wazn_f1_all:.4f} '
              f'| gap: {boundary_gap:+.4f} | bar passes (≥0.10): {bar_boundary}')
        print(f'  Weak stratum (n={len(bpe_records_weak)}): '
              f'BPE={bpe_f1_w:.4f} | WAZN={wazn_f1_w:.4f}')
        print(f'  Strong stratum (n={len(bpe_records_strong)}): '
              f'BPE={bpe_f1_s:.4f} | WAZN={wazn_f1_s:.4f}')

        # ── Designed-to-fail gate ──────────────────────────────────────────────
        print(f'[{time.time()-t0:.1f}s]  Running designed-to-fail gate...')
        gate_result = run_gate(gold_set, heldout_analysis, tmpl_skeleton_map)
        print(f'  Gate: real_f1={gate_result["f1_real_wazn_bpe"]:.4f} '
              f'corrupted_f1={gate_result["f1_deranged_wazn_bpe"]:.4f} '
              f'→ {gate_result["verdict"]}')

        # ── Named outcomes (§3) ────────────────────────────────────────────────
        if bar_fertility and bar_boundary:
            third_outcome = 'both bars pass'
        elif not bar_fertility and bar_boundary:
            third_outcome = 'MORPHOLOGY_TAX: fertility worse, boundaries better'
        elif bar_fertility and not bar_boundary:
            third_outcome = 'fertility better but boundaries flat/worse'
        else:
            third_outcome = 'BACKBONE_FAILS: both bars fail at this vocab size'

        seed_result = {
            'seed': seed,
            'corpus_split': {
                'n_train_lines':   len(train_lines),
                'n_heldout_lines': len(heldout_lines),
                'train_sha256':    train_sha,
                'heldout_sha256':  heldout_sha,
            },
            'vocab_accounting': wazn_bpe_vocab_accounting,
            'baseline_bpe_vocab_actual': actual_baseline_vocab,
            'wazn_bpe_vocab_actual': actual_wazn_total_vocab,
            'fertility': {
                'definition': 'total_tokens / total_whitespace_words on held-out sample',
                'held_out_lines_used': len(heldout_sample),
                'bpe_fertility':       round(fert_baseline, 6),
                'wazn_bpe_fertility':  round(fert_wazn, 6),
                'ratio_wazn_over_bpe': round(fertility_ratio, 6),
                'bar_fertility_improves': bar_fertility,
                'bar_definition': 'wazn_fertility ≤ 0.95 × bpe_fertility',
                'compression_ratio': {
                    'definition': 'surface chars / tokens (chars per token; higher = more compression)',
                    'bpe':      round(comp_baseline, 4),
                    'wazn_bpe': round(comp_wazn, 4),
                },
            },
            'gold_set': {
                **gold_meta,
                'train_overlap_frac': round(train_overlap_frac, 4),
                'n_gold_valid_for_f1': n_gold_valid,
            },
            'boundary_f1': {
                'overall': {
                    'bpe':  {'p': round(bpe_p_all,4), 'r': round(bpe_r_all,4), 'f1': round(bpe_f1_all,4)},
                    'wazn': {'p': round(wazn_p_all,4),'r': round(wazn_r_all,4),'f1': round(wazn_f1_all,4)},
                    'gap_wazn_minus_bpe': round(boundary_gap, 4),
                    'bar_boundaries_recovered': bar_boundary,
                    'bar_definition': 'wazn_f1 ≥ bpe_f1 + 0.10',
                },
                'weak_stratum': {
                    'n': len(bpe_records_weak),
                    'bpe':  {'p': round(bpe_p_w,4), 'r': round(bpe_r_w,4), 'f1': round(bpe_f1_w,4)},
                    'wazn': {'p': round(wazn_p_w,4),'r': round(wazn_r_w,4),'f1': round(wazn_f1_w,4)},
                    'gap_wazn_minus_bpe': round(wazn_f1_w - bpe_f1_w, 4),
                },
                'strong_stratum': {
                    'n': len(bpe_records_strong),
                    'bpe':  {'p': round(bpe_p_s,4), 'r': round(bpe_r_s,4), 'f1': round(bpe_f1_s,4)},
                    'wazn': {'p': round(wazn_p_s,4),'r': round(wazn_r_s,4),'f1': round(wazn_f1_s,4)},
                    'gap_wazn_minus_bpe': round(wazn_f1_s - bpe_f1_s, 4),
                },
            },
            'gate': gate_result,
            'named_outcome': third_outcome,
        }
        seed_results.append(seed_result)

        # Cleanup temp files
        try:
            os.unlink(tmp_bpe_corpus)
            os.unlink(tmp_wazn_corpus)
        except Exception:
            pass

    # ── Aggregate across seeds ─────────────────────────────────────────────────
    print(f'\n[{time.time()-t0:.1f}s] Aggregating across {len(SEEDS)} seeds...')

    def mean(vals): return sum(vals) / len(vals) if vals else float('nan')

    agg = {
        'fertility': {
            'bpe_mean':       round(mean([s['fertility']['bpe_fertility'] for s in seed_results]), 6),
            'wazn_bpe_mean':  round(mean([s['fertility']['wazn_bpe_fertility'] for s in seed_results]), 6),
            'ratio_mean':     round(mean([s['fertility']['ratio_wazn_over_bpe'] for s in seed_results]), 6),
            'bar_fertility_improves': all(s['fertility']['bar_fertility_improves'] for s in seed_results),
            'compression_ratio': {
                'bpe_mean':      round(mean([s['fertility']['compression_ratio']['bpe'] for s in seed_results]), 4),
                'wazn_bpe_mean': round(mean([s['fertility']['compression_ratio']['wazn_bpe'] for s in seed_results]), 4),
            },
        },
        'boundary_f1_overall': {
            'bpe_f1_mean':    round(mean([s['boundary_f1']['overall']['bpe']['f1'] for s in seed_results]), 4),
            'wazn_f1_mean':   round(mean([s['boundary_f1']['overall']['wazn']['f1'] for s in seed_results]), 4),
            'gap_mean':       round(mean([s['boundary_f1']['overall']['gap_wazn_minus_bpe'] for s in seed_results]), 4),
            'bar_boundaries_recovered': all(s['boundary_f1']['overall']['bar_boundaries_recovered'] for s in seed_results),
        },
        'weak_stratum': {
            'bpe_f1_mean':    round(mean([s['boundary_f1']['weak_stratum']['bpe']['f1'] for s in seed_results]), 4),
            'wazn_f1_mean':   round(mean([s['boundary_f1']['weak_stratum']['wazn']['f1'] for s in seed_results]), 4),
            'gap_mean':       round(mean([s['boundary_f1']['weak_stratum']['gap_wazn_minus_bpe'] for s in seed_results]), 4),
        },
        'strong_stratum': {
            'bpe_f1_mean':    round(mean([s['boundary_f1']['strong_stratum']['bpe']['f1'] for s in seed_results]), 4),
            'wazn_f1_mean':   round(mean([s['boundary_f1']['strong_stratum']['wazn']['f1'] for s in seed_results]), 4),
            'gap_mean':       round(mean([s['boundary_f1']['strong_stratum']['gap_wazn_minus_bpe'] for s in seed_results]), 4),
        },
        'named_outcomes': list(set(s['named_outcome'] for s in seed_results)),
        'gate': {
            'all_gates_fire': all(s['gate']['gate_fires'] for s in seed_results),
            'f1_real_mean':    round(mean([s['gate']['f1_real_wazn_bpe'] for s in seed_results]), 4),
            'f1_corrupt_mean': round(mean([s['gate']['f1_deranged_wazn_bpe'] for s in seed_results]), 4),
        },
    }

    elapsed = time.time() - t0

    # ── Write results ──────────────────────────────────────────────────────────
    results = {
        'meta': {
            'experiment':     'exp80 — WAZN-aware tokenizer',
            'registration':   '/Users/wajahath/qnlp_private_docs/72_exp80_wazn_tokenizer.md',
            'script':         __file__,
            'camel_route':    'LIVE (Mac, calima-msa-r13)',
            'wsl_status':     'DOWN — Mac-local CAMeL used',
            'corpus': {
                'source':     'wikimedia/wikipedia 20231101.ar (HuggingFace datasets streaming)',
                'path':       CORPUS_PATH,
                'sha256':     corpus_sha,
                'n_lines':    len(all_lines),
                'raw_bytes':  78675954,
                'n_articles': 3673,
            },
            'seeds':          SEEDS,
            'vocab_size':     VOCAB_SIZE,
            'gold_size_target': GOLD_SIZE,
            'weak_min_frac':  WEAK_MIN_FRAC,
            'tokenizer_lib':  'HuggingFace tokenizers (BpeTrainer)',
            'elapsed_seconds': round(elapsed, 1),
        },
        'cross_check_pass1': cc,
        'per_seed': seed_results,
        'aggregate': agg,
        'bars_summary': {
            'fertility_improves': agg['fertility']['bar_fertility_improves'],
            'boundaries_recovered': agg['boundary_f1_overall']['bar_boundaries_recovered'],
            'named_outcome': agg['named_outcomes'],
        },
    }

    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f'\n[{elapsed:.1f}s] Done. Results → {RESULTS_PATH}')

    # ── Summary table ──────────────────────────────────────────────────────────
    print('\n══════════════════════════════════════════════════════════')
    print('                   EXP80 RESULTS TABLE')
    print('══════════════════════════════════════════════════════════')
    print(f'Corpus: arwiki 20231101.ar, {len(all_lines):,} lines, 75MB')
    print(f'CAMeL:  calima-msa-r13 (Mac-local; WSL down)')
    print(f'Cross-check pass1: root={cc["root_agreement_pct"]}% '
          f'pattern={cc["pattern_agreement_pct"]}% '
          f'both={cc["both_agreement_pct"]}%')
    print()
    print(f'{"Metric":<35} {"BPE":>10} {"WAZN-BPE":>10} {"Gap":>8} {"Bar":>6}')
    print(f'{"-"*70}')
    a = agg
    print(f'{"Fertility (mean)":<35} '
          f'{a["fertility"]["bpe_mean"]:>10.4f} '
          f'{a["fertility"]["wazn_bpe_mean"]:>10.4f} '
          f'{a["fertility"]["ratio_mean"]-1:>+8.4f} '
          f'{"PASS" if a["fertility"]["bar_fertility_improves"] else "FAIL":>6}')
    print(f'{"Boundary F1 — overall (mean)":<35} '
          f'{a["boundary_f1_overall"]["bpe_f1_mean"]:>10.4f} '
          f'{a["boundary_f1_overall"]["wazn_f1_mean"]:>10.4f} '
          f'{a["boundary_f1_overall"]["gap_mean"]:>+8.4f} '
          f'{"PASS" if a["boundary_f1_overall"]["bar_boundaries_recovered"] else "FAIL":>6}')
    print(f'{"  — weak stratum":<35} '
          f'{a["weak_stratum"]["bpe_f1_mean"]:>10.4f} '
          f'{a["weak_stratum"]["wazn_f1_mean"]:>10.4f} '
          f'{a["weak_stratum"]["gap_mean"]:>+8.4f} '
          f'{"(informative)":>6}')
    print(f'{"  — strong stratum":<35} '
          f'{a["strong_stratum"]["bpe_f1_mean"]:>10.4f} '
          f'{a["strong_stratum"]["wazn_f1_mean"]:>10.4f} '
          f'{a["strong_stratum"]["gap_mean"]:>+8.4f} '
          f'{"(informative)":>6}')
    print()
    comp = agg['fertility']['compression_ratio']
    print(f'{"Compression (chars/tok, descriptive)":<35} '
          f'{comp["bpe_mean"]:>10.4f} '
          f'{comp["wazn_bpe_mean"]:>10.4f} '
          f'{"":>8} '
          f'{"(desc)":>6}')
    print()
    print(f'Named outcome: {agg["named_outcomes"]}')
    print(f'Gate: all_fires={agg["gate"]["all_gates_fire"]} '
          f'(real={agg["gate"]["f1_real_mean"]:.4f} '
          f'corrupt={agg["gate"]["f1_corrupt_mean"]:.4f})')
    print('══════════════════════════════════════════════════════════')


if __name__ == '__main__':
    main()
