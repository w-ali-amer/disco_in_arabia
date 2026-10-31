#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp89_run.py — THE ALLOCATION CELL (four tokenizer arms, 3 seeds each).

Registration: qnlp_private_docs/81_exp88_89_skeleton_scale_and_allocation.md §3
Intermediate writers: exp89_anchor_gate.json, exp89_evaluate.json, exp89_gates.json
Final results: exp89_aggregate.py → results_exp89.json

Stages
  --anchor-gates   : evaluate exp84 BPE + FUSED-F8000 anchors; run anchor gate
  --build-new      : build FUSED-BACKOFF-OPT and FUSED-NOBACKOFF for all seeds
  --evaluate       : evaluate all 4 arms on all 3 seeds  → exp89_evaluate.json
  --gates          : vocab accounting + expansion-determinism gates → exp89_gates.json
  --all            : build-new + evaluate + gates (for nohup)

Run on WSL:
  ~/venv/bin/python3 exp89_run.py --anchor-gates 2>&1 | tee exp89_anchor.log
  nohup ~/venv/bin/python3 exp89_run.py --all > exp89_wsl.log 2>&1 &
"""

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import sys
import tempfile
import time
from collections import Counter

# ── CAMeL path ────────────────────────────────────────────────────────────────
_CAMEL_SRC = os.path.expanduser('~/camel_tools')
if os.path.isdir(_CAMEL_SRC) and _CAMEL_SRC not in sys.path:
    sys.path.insert(0, _CAMEL_SRC)

WORK_DIR    = os.path.expanduser('~/discocat_arabic_v2')
CORPUS_PATH = os.path.join(WORK_DIR, 'arwiki_exp84.txt')

# ── exp84 lineage constants (must not change) ─────────────────────────────────
VOCAB_SIZE          = 32_000
TRAIN_FRAC          = 0.90
SEEDS               = [42, 43, 44]
GOLD_SIZE           = 2_000
WEAK_MIN_FRAC       = 0.30
MIN_TMPL_OBS        = 3
TOP_K_TYPES         = 80_000
SPECIAL_TOKENS      = ['[UNK]', '[BOS]', '[EOS]']
WEAK_RADICAL_CHARS  = frozenset('وياءأإآ#')   # exact from exp84_run.py line 66

# ── exp89 new arms ────────────────────────────────────────────────────────────
N_FUSED_ANCHOR = 8_000    # anchors: same as exp84
N_FUSED_OPT    = 24_000   # FUSED-BACKOFF-OPT
N_FUSED_NOBACK = 16_000   # FUSED-NOBACKOFF

# ── anchor gate: exp84 reference values (from results_exp84.json) ─────────────
EXP84_REF = {
    42: {'bpe_fertility': 1.381825, 'fused_fertility': 1.824266,
         'bpe_f1': 0.1974,          'fused_f1': 0.5785},
    43: {'bpe_fertility': 1.379707, 'fused_fertility': 1.822217,
         'bpe_f1': 0.1899,          'fused_f1': 0.5624},
    44: {'bpe_fertility': 1.380975, 'fused_fertility': 1.823715,
         'bpe_f1': 0.2045,          'fused_f1': 0.5609},
}
ANCHOR_TOL = 0.03

DIACRITICS = frozenset('ًٌٍَُِّْٕٖٟٓٔٗ٘')
ARABIC_RE  = re.compile(r'[؀-ۿ]')


# ═══════════════════════════════════════════════════════════════════════════════
# §1  UTILITIES  (exact parity with exp84_run.py)
# ═══════════════════════════════════════════════════════════════════════════════

def strip_diac(s):
    return ''.join(c for c in s if c not in DIACRITICS)

def normalize_surface(s):
    s = strip_diac(s)
    return s.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')

def is_arabic_word(w):
    arabic_count = sum(1 for c in w if ARABIC_RE.match(c))
    latin_count  = sum(1 for c in w if c.isascii() and c.isalpha())
    return arabic_count >= 2 and latin_count == 0

def is_weak_root(root):
    """Exact copy from exp84_run.py line 96-100."""
    parts = root.split('.')
    if len(parts) != 3:
        return False
    return any(p in WEAK_RADICAL_CHARS for p in parts)

def sha256_str(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def artifact_path(name):
    return os.path.join(WORK_DIR, name)

def analysis_cache_path(seed):
    return artifact_path(f'exp84_analysis_s{seed}.pkl')

def fused_set_path(seed):
    return artifact_path(f'exp84_fused_set_s{seed}.json')

def tmpl_skel_path(seed):
    return artifact_path(f'exp84_tmpl_skel_s{seed}.json')

def split_path(seed):
    return artifact_path(f'exp84_split_s{seed}.json')

def tok_dir(arm, seed):
    if arm in ('bpe', 'fused', 'wazn'):
        return artifact_path(f'exp84_{arm}_tok_s{seed}')
    return artifact_path(f'exp89_{arm}_tok_s{seed}')

def fused_opt_set_path(seed):
    return artifact_path(f'exp89_fused_opt_set_s{seed}.json')

def fused_noback_set_path(seed):
    return artifact_path(f'exp89_fused_noback_set_s{seed}.json')


def split_lines(lines, train_frac, seed):
    rng = random.Random(seed)
    indices = list(range(len(lines)))
    rng.shuffle(indices)
    n_train = int(len(lines) * train_frac)
    train_idx   = sorted(indices[:n_train])
    heldout_idx = sorted(indices[n_train:])
    return [lines[i] for i in train_idx], [lines[i] for i in heldout_idx]


def extract_arabic_types(lines):
    counts = Counter()
    for line in lines:
        for w in line.split():
            if is_arabic_word(w):
                counts[w] += 1
    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# §2  BW PARSING → GOLD BOUNDARIES  (exact copy from exp84_run.py §3)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_bw_boundaries(bw_field, word):
    """Parse Buckwalter morpheme boundaries into character-position set."""
    parts = bw_field.split('+')
    segs  = []
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
# §3  SKELETON EXTRACTION  (exact copy from exp84_run.py §4)
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


# ═══════════════════════════════════════════════════════════════════════════════
# §4  BPE UTILITIES  (exact parity with exp84_run.py §5)
# ═══════════════════════════════════════════════════════════════════════════════

def bpe_tokenize_word(tokenizer, word):
    return tokenizer.encode(word).tokens


def predict_boundaries_bpe(tokenizer, word):
    """Exact copy from exp84_run.py."""
    surf   = normalize_surface(word)
    tokens = bpe_tokenize_word(tokenizer, surf)
    bounds = set()
    pos    = 0
    for tok in tokens[:-1]:
        tok_chars = re.sub(r'^[##▁Ġ]+', '', tok)
        if not tok_chars:
            tok_chars = tok
        pos += len(tok_chars)
        if 0 < pos < len(surf):
            bounds.add(pos)
    return bounds


def compute_fertility_bpe(tokenizer, lines, max_lines=None):
    """Exact copy from exp84_run.py: encode whole line at once."""
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
# §5  WAZN / FUSED UTILITIES  (exact parity with exp84_run.py §§6-7)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_boundaries_wazn(word, reading, tmpl_skeleton_map):
    """Exact copy from exp84_run.py: prefix + suffix boundary."""
    surf    = normalize_surface(word)
    L       = len(surf)
    pattern = reading.get('pattern', '')
    skel    = (tmpl_skeleton_map.get(pattern) if pattern in tmpl_skeleton_map
               else extract_skeleton(pattern))
    bounds  = set()
    if skel['prefix_len'] > 0 and skel['prefix_len'] < L:
        bounds.add(skel['prefix_len'])
    if skel['suffix_len'] > 0 and (L - skel['suffix_len']) > 0:
        bounds.add(L - skel['suffix_len'])
    return bounds


def predict_boundaries_fused_expanded(word, fused_set, analysis_cache,
                                       tmpl_skeleton_map):
    fs = fused_set.get(word)
    if fs is not None:
        reading = {'root': fs['root'], 'pattern': fs['pattern'], 'bw': ''}
        return predict_boundaries_wazn(word, reading, tmpl_skeleton_map)
    r = analysis_cache.get(word)
    if r is not None:
        return predict_boundaries_wazn(word, r, tmpl_skeleton_map)
    return set()


def fused_fertility_word(word, fused_set, analysis_cache, bpe_fallback):
    """Exact copy from exp84_run.py."""
    if word in fused_set:
        return 1
    r = analysis_cache.get(word)
    if r is not None and len(r.get('root', '').split('.')) == 3:
        return 4   # R1 R2 R3 TMPL
    surf = normalize_surface(word)
    return len(bpe_tokenize_word(bpe_fallback, surf))


def compute_fertility_fused(lines, fused_set, analysis_cache, bpe_fallback,
                             max_lines=None):
    """Exact parity with exp84: count ALL words; bpe_fallback = arm's OWN tokenizer."""
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


def noback_fertility_word(word, fused_set, arm_tok):
    """FUSED-NOBACKOFF: fused→1, non-fused→arm tokenizer (plain BPE portion)."""
    if word in fused_set:
        return 1
    surf = normalize_surface(word)
    return len(bpe_tokenize_word(arm_tok, surf))


def compute_fertility_noback(lines, fused_set, arm_tok, max_lines=None):
    """FUSED-NOBACKOFF fertility: count ALL words; arm_tok = NOBACKOFF tokenizer."""
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
            total_tokens += noback_fertility_word(w, fused_set, arm_tok)
    fert = total_tokens / total_words if total_words > 0 else float('nan')
    comp = total_chars / total_tokens if total_tokens > 0 else float('nan')
    return fert, comp


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
# §6  FUSED-SET CONSTRUCTION  (exact copy from exp84_run.py)
# ═══════════════════════════════════════════════════════════════════════════════

def build_fused_set(train_type_counts, analysis_cache, n_fused):
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


# ═══════════════════════════════════════════════════════════════════════════════
# §7  CORPUS PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_line_fused_train(line, fused_set, analysis_cache):
    words = line.split()
    out   = []
    for w in words:
        if w in fused_set:
            fs = fused_set[w]
            out.append(' '.join(fs['parts'] + [f'TMPL_{fs["pattern"]}']))
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


def preprocess_line_noback_train(line, fused_set):
    """NOBACKOFF: fused → 'R1 R2 R3' (no template); non-fused → surface."""
    words = line.split()
    out   = []
    for w in words:
        if w in fused_set:
            fs = fused_set[w]
            out.append(' '.join(fs['parts']))
            continue
        out.append(w)
    return ' '.join(out)


# ═══════════════════════════════════════════════════════════════════════════════
# §8  BPE TRAINING  (exact copy from exp84)
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


# ═══════════════════════════════════════════════════════════════════════════════
# §9  GOLD SET CONSTRUCTION  (exact parity with exp84_run.py §9)
# ═══════════════════════════════════════════════════════════════════════════════

def build_gold_set(heldout_lines, analysis_cache, target_size, weak_min_frac,
                   tmpl_skeleton_map, seed):
    """Returns (gold_set: list-of-dicts, meta: dict). Exact parity with exp84."""
    rng = random.Random(seed)
    type_counts       = extract_arabic_types(heldout_lines)
    gold_cands_weak   = []
    gold_cands_strong = []
    mismatch_count    = 0

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
            'word': word, 'root': root, 'pattern': pattern,
            'weak': weak, 'gold_bounds': sorted(gold_bounds), 'freq': freq,
        }
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

    actual_weak_frac = (sum(1 for e in gold if e['weak']) / len(gold)) if gold else 0.0

    return gold, {
        'n_weak_candidates':  len(gold_cands_weak),
        'n_strong_candidates': len(gold_cands_strong),
        'n_gold':             len(gold),
        'n_weak_in_gold':     sum(1 for e in gold if e['weak']),
        'actual_weak_frac':   actual_weak_frac,
        'mismatch_drops':     mismatch_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §10  BOUNDARY F1 EVALUATION  (per arm; gold_set = list-of-dicts)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_bpe_boundary_f1(gold_set, bpe_tok):
    records = []
    for entry in gold_set:
        word       = entry['word']
        gold_bounds = set(entry['gold_bounds'])
        surf       = normalize_surface(word)
        pred       = predict_boundaries_bpe(bpe_tok, surf)
        records.append((gold_bounds, pred))
    p, r, f1 = micro_f1(records)
    return {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4), 'n': len(records)}


def evaluate_fused_expanded_f1(gold_set, fused_set, analysis_cache, tmpl_skeleton_map):
    records = []
    for entry in gold_set:
        word        = entry['word']
        gold_bounds = set(entry['gold_bounds'])
        pred        = predict_boundaries_fused_expanded(
            word, fused_set, analysis_cache, tmpl_skeleton_map)
        records.append((gold_bounds, pred))
    p, r, f1 = micro_f1(records)
    return {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4), 'n': len(records)}


def evaluate_noback_expanded_f1(gold_set, fused_set, arm_tok):
    """FUSED-NOBACKOFF: fused→WAZN (stored decomp); non-fused→arm tokenizer BPE."""
    def predict_noback(word):
        fs = fused_set.get(word)
        if fs is not None:
            skel = extract_skeleton(fs['pattern'])
            surf = normalize_surface(word)
            L    = len(surf)
            bounds = set()
            if skel['prefix_len'] > 0 and skel['prefix_len'] < L:
                bounds.add(skel['prefix_len'])
            if skel['suffix_len'] > 0 and (L - skel['suffix_len']) > 0:
                bounds.add(L - skel['suffix_len'])
            return bounds
        return predict_boundaries_bpe(arm_tok, word)

    records = []
    for entry in gold_set:
        word        = entry['word']
        gold_bounds = set(entry['gold_bounds'])
        pred        = predict_noback(word)
        records.append((gold_bounds, pred))
    p, r, f1 = micro_f1(records)
    return {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4), 'n': len(records)}


# ═══════════════════════════════════════════════════════════════════════════════
# §11  ANCHOR GATE
# ═══════════════════════════════════════════════════════════════════════════════

def run_anchor_gates(all_lines):
    """
    Load exp84 BPE + FUSED-F8000 tokenizers. Evaluate on all 3 seeds.
    FUSED fertility uses fused_tok (own tokenizer) as bpe_fallback — exact parity
    with exp84's evaluate stage (fused_tok as fallback in compute_fertility_fused).
    Writes exp89_anchor_gate.json. Exits with code 1 on gate failure.
    """
    from tokenizers import Tokenizer
    t0 = time.time()
    print('[anchor-gates] Starting anchor gate evaluation…')

    per_seed = []
    all_pass = True

    for seed in SEEDS:
        print(f'\n[anchor-gates] Seed {seed}…')

        cache_p = analysis_cache_path(seed)
        if not os.path.exists(cache_p):
            sys.exit(f'FATAL: analysis cache missing: {cache_p}')
        with open(cache_p, 'rb') as f:
            analysis_cache = pickle.load(f)

        fused_set     = load_json(fused_set_path(seed))
        tmpl_skel_map = load_json(tmpl_skel_path(seed))

        bpe_dir   = tok_dir('bpe',   seed)
        fused_dir = tok_dir('fused', seed)
        if not os.path.exists(bpe_dir   + '/tokenizer.json'):
            sys.exit(f'FATAL: BPE tokenizer missing: {bpe_dir}')
        if not os.path.exists(fused_dir + '/tokenizer.json'):
            sys.exit(f'FATAL: FUSED tokenizer missing: {fused_dir}')

        bpe_tok   = Tokenizer.from_file(bpe_dir   + '/tokenizer.json')
        fused_tok = Tokenizer.from_file(fused_dir + '/tokenizer.json')
        print(f'  BPE vocab={bpe_tok.get_vocab_size():,}, '
              f'FUSED vocab={fused_tok.get_vocab_size():,}')

        _, heldout_lines = split_lines(all_lines, TRAIN_FRAC, seed)

        # Fertility: use own tokenizer as fallback (matching exp84's evaluate stage)
        print(f'  [s{seed}] Computing fertility…')
        bpe_fert,    _ = compute_fertility_bpe(bpe_tok, heldout_lines, max_lines=20000)
        fused_fert,  _ = compute_fertility_fused(
            heldout_lines, fused_set, analysis_cache, fused_tok, max_lines=20000)
        print(f'  BPE={bpe_fert:.6f}, FUSED={fused_fert:.6f}')

        # Gold set
        print(f'  [s{seed}] Building gold set…')
        gold_set, gold_meta = build_gold_set(
            heldout_lines, analysis_cache, GOLD_SIZE, WEAK_MIN_FRAC, tmpl_skel_map, seed)
        print(f'  Gold: {gold_meta["n_gold"]} words '
              f'({100*gold_meta["actual_weak_frac"]:.1f}% weak)')

        # Boundary F1
        print(f'  [s{seed}] Computing boundary F1…')
        bpe_f1_res   = evaluate_bpe_boundary_f1(gold_set, bpe_tok)
        fused_f1_res = evaluate_fused_expanded_f1(
            gold_set, fused_set, analysis_cache, tmpl_skel_map)
        print(f'  BPE F1={bpe_f1_res["f1"]:.4f}, FUSED F1={fused_f1_res["f1"]:.4f}')

        # Anchor gate checks
        ref = EXP84_REF[seed]
        checks = {}
        for metric_name, measured, reference in [
            ('bpe_fertility',   bpe_fert,           ref['bpe_fertility']),
            ('fused_fertility', fused_fert,          ref['fused_fertility']),
            ('bpe_f1',          bpe_f1_res['f1'],   ref['bpe_f1']),
            ('fused_f1',        fused_f1_res['f1'],  ref['fused_f1']),
        ]:
            delta = abs(measured - reference)
            ok    = delta <= ANCHOR_TOL
            checks[metric_name] = {
                'measured': round(measured, 6), 'reference': reference,
                'delta': round(delta, 6), 'within_tol': ok,
            }
            print(f'    [{"PASS" if ok else "FAIL"}] {metric_name}: '
                  f'{measured:.6f} ref={reference:.6f} Δ={delta:.6f}')

        seed_pass = all(v['within_tol'] for v in checks.values())
        if not seed_pass:
            all_pass = False

        per_seed.append({
            'seed': seed,
            'bpe_fertility': round(bpe_fert, 6), 'fused_fertility': round(fused_fert, 6),
            'bpe_f1': bpe_f1_res, 'fused_f1': fused_f1_res,
            'gold_meta': gold_meta, 'anchor_checks': checks, 'seed_passes': seed_pass,
        })

    verdict = ('ANCHOR_GATE_PASS: all seeds within ±0.03 — run continues' if all_pass else
               'ANCHOR_GATE_FAIL: at least one seed outside ±0.03 — RUN VOIDED')
    print(f'\n[anchor-gates] {verdict}')

    result = {
        'gate_name': 'EXP89_ANCHOR_GATE',
        'anchor_tolerance': ANCHOR_TOL,
        'per_seed': per_seed,
        'all_seeds_pass': all_pass,
        'verdict': verdict,
        'elapsed_s': round(time.time() - t0, 1),
    }
    save_json(result, artifact_path('exp89_anchor_gate.json'))
    print(f'[anchor-gates] Written → exp89_anchor_gate.json')

    if not all_pass:
        print('[anchor-gates] FATAL: anchor gate failure voids exp89.')
        sys.exit(1)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# §12  BUILD NEW ARMS
# ═══════════════════════════════════════════════════════════════════════════════

def build_new_arms_seed(seed, all_lines, t0):
    from tokenizers import Tokenizer

    print(f'\n[build-new] === SEED {seed} ===')
    train_lines, _ = split_lines(all_lines, TRAIN_FRAC, seed)

    with open(analysis_cache_path(seed), 'rb') as f:
        analysis_cache = pickle.load(f)

    train_type_counts = extract_arabic_types(train_lines)

    # Template vocabulary (same filter as exp84)
    pattern_counts = Counter()
    for r in analysis_cache.values():
        pat = r.get('pattern', '')
        if pat:
            pattern_counts[pat] += 1
    observed_templates = [p for p, cnt in pattern_counts.most_common()
                          if cnt >= MIN_TMPL_OBS]
    tmpl_tokens = [f'TMPL_{p}' for p in observed_templates]
    n_tmpl = len(tmpl_tokens)
    print(f'  Templates ≥{MIN_TMPL_OBS}: {n_tmpl:,}')

    # ── FUSED-BACKOFF-OPT (F=24000) ─────────────────────────────────────────
    opt_dir  = tok_dir('fused_opt', seed)
    opt_path = opt_dir + '/tokenizer.json'
    if os.path.exists(opt_path):
        tok_opt  = Tokenizer.from_file(opt_path)
        fused_opt = load_json(fused_opt_set_path(seed))
        print(f'[{time.time()-t0:.0f}s]  FUSED-BACKOFF-OPT loaded ({tok_opt.get_vocab_size():,})')
    else:
        os.makedirs(opt_dir, exist_ok=True)
        fused_opt    = build_fused_set(train_type_counts, analysis_cache, N_FUSED_OPT)
        save_json(fused_opt, fused_opt_set_path(seed))
        n_fused_opt  = len(fused_opt)
        n_bpe_opt    = VOCAB_SIZE - 3 - n_tmpl - n_fused_opt
        print(f'[{time.time()-t0:.0f}s]  FUSED-BACKOFF-OPT build: '
              f'fused={n_fused_opt}, tmpl={n_tmpl}, bpe_budget={n_bpe_opt}')

        opt_lines    = [preprocess_line_fused_train(l, fused_opt, analysis_cache)
                        for l in train_lines]
        opt_specials = SPECIAL_TOKENS + tmpl_tokens + list(fused_opt.keys())
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                          encoding='utf-8') as tf:
            tmp = tf.name
        tok_opt = train_bpe_tokenizer(opt_lines, VOCAB_SIZE, opt_specials, tmp)
        tok_opt.save(opt_path)
        os.unlink(tmp)
        print(f'[{time.time()-t0:.0f}s]  FUSED-BACKOFF-OPT trained ({tok_opt.get_vocab_size():,})')

    # ── FUSED-NOBACKOFF (F=16000) ────────────────────────────────────────────
    noback_dir  = tok_dir('fused_noback', seed)
    noback_path = noback_dir + '/tokenizer.json'
    if os.path.exists(noback_path):
        tok_noback   = Tokenizer.from_file(noback_path)
        fused_noback = load_json(fused_noback_set_path(seed))
        print(f'[{time.time()-t0:.0f}s]  FUSED-NOBACKOFF loaded ({tok_noback.get_vocab_size():,})')
    else:
        os.makedirs(noback_dir, exist_ok=True)
        fused_noback = build_fused_set(train_type_counts, analysis_cache, N_FUSED_NOBACK)
        save_json(fused_noback, fused_noback_set_path(seed))
        n_fused_nb   = len(fused_noback)
        n_bpe_nb     = VOCAB_SIZE - 3 - n_fused_nb   # NO template tokens
        print(f'[{time.time()-t0:.0f}s]  FUSED-NOBACKOFF build: '
              f'fused={n_fused_nb}, tmpl=0, bpe_budget={n_bpe_nb}')

        noback_lines    = [preprocess_line_noback_train(l, fused_noback)
                           for l in train_lines]
        noback_specials = SPECIAL_TOKENS + list(fused_noback.keys())
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                          encoding='utf-8') as tf:
            tmp = tf.name
        tok_noback = train_bpe_tokenizer(noback_lines, VOCAB_SIZE, noback_specials, tmp)
        tok_noback.save(noback_path)
        os.unlink(tmp)
        print(f'[{time.time()-t0:.0f}s]  FUSED-NOBACKOFF trained ({tok_noback.get_vocab_size():,})')

    return {
        'seed': seed, 'n_tmpl': n_tmpl,
        'fused_opt_size':     len(fused_opt),
        'fused_noback_size':  len(fused_noback),
        'fused_opt_vocab':    tok_opt.get_vocab_size(),
        'fused_noback_vocab': tok_noback.get_vocab_size(),
    }


def run_build_new(all_lines):
    t0 = time.time()
    print('[build-new] Building FUSED-BACKOFF-OPT and FUSED-NOBACKOFF…')
    results = []
    for seed in SEEDS:
        r = build_new_arms_seed(seed, all_lines, t0)
        results.append(r)
    save_json({'stages': results, 'elapsed_s': round(time.time()-t0, 1)},
              artifact_path('exp89_build_new_summary.json'))
    print(f'[build-new] Done in {time.time()-t0:.0f}s')


# ═══════════════════════════════════════════════════════════════════════════════
# §13  FULL EVALUATION (all 4 arms × 3 seeds)
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluate(all_lines):
    from tokenizers import Tokenizer
    t0 = time.time()
    print('[evaluate] Evaluating all 4 arms × 3 seeds…')

    per_seed_results = []

    for seed in SEEDS:
        print(f'\n[evaluate] === SEED {seed} ===')
        _, heldout_lines = split_lines(all_lines, TRAIN_FRAC, seed)

        with open(analysis_cache_path(seed), 'rb') as f:
            analysis_cache = pickle.load(f)
        fused_set     = load_json(fused_set_path(seed))
        tmpl_skel_map = load_json(tmpl_skel_path(seed))

        for p in [fused_opt_set_path(seed), fused_noback_set_path(seed)]:
            if not os.path.exists(p):
                sys.exit(f'FATAL: {p} — run --build-new first')
        fused_opt    = load_json(fused_opt_set_path(seed))
        fused_noback = load_json(fused_noback_set_path(seed))

        bpe_tok    = Tokenizer.from_file(tok_dir('bpe',          seed) + '/tokenizer.json')
        fused_tok  = Tokenizer.from_file(tok_dir('fused',        seed) + '/tokenizer.json')
        opt_tok    = Tokenizer.from_file(tok_dir('fused_opt',    seed) + '/tokenizer.json')
        noback_tok = Tokenizer.from_file(tok_dir('fused_noback', seed) + '/tokenizer.json')

        # Fertility (own tokenizer as fallback for each FUSED arm)
        print(f'  [s{seed}] Fertility (max 20000 heldout lines)…')
        bpe_fert,    _ = compute_fertility_bpe(bpe_tok,    heldout_lines, max_lines=20000)
        fused_fert,  _ = compute_fertility_fused(heldout_lines, fused_set,
                                                  analysis_cache, fused_tok,   max_lines=20000)
        opt_fert,    _ = compute_fertility_fused(heldout_lines, fused_opt,
                                                  analysis_cache, opt_tok,     max_lines=20000)
        noback_fert, _ = compute_fertility_noback(heldout_lines, fused_noback,
                                                   noback_tok,  max_lines=20000)
        print(f'  BPE={bpe_fert:.4f} FUSED8k={fused_fert:.4f} '
              f'OPT24k={opt_fert:.4f} NOBACK16k={noback_fert:.4f}')

        # Gold set
        print(f'  [s{seed}] Building gold set…')
        gold_set, gold_meta = build_gold_set(
            heldout_lines, analysis_cache, GOLD_SIZE, WEAK_MIN_FRAC, tmpl_skel_map, seed)
        print(f'  Gold: {gold_meta["n_gold"]} words '
              f'({100*gold_meta["actual_weak_frac"]:.1f}% weak)')

        # Boundary F1
        print(f'  [s{seed}] Boundary F1…')
        bpe_f1    = evaluate_bpe_boundary_f1(gold_set, bpe_tok)
        fused_f1  = evaluate_fused_expanded_f1(gold_set, fused_set,  analysis_cache, tmpl_skel_map)
        opt_f1    = evaluate_fused_expanded_f1(gold_set, fused_opt,  analysis_cache, tmpl_skel_map)
        noback_f1 = evaluate_noback_expanded_f1(gold_set, fused_noback, noback_tok)
        print(f'  BPE={bpe_f1["f1"]:.4f} FUSED8k={fused_f1["f1"]:.4f} '
              f'OPT={opt_f1["f1"]:.4f} NOBACK={noback_f1["f1"]:.4f}')

        per_seed_results.append({
            'seed': seed, 'n_heldout': len(heldout_lines), 'gold_meta': gold_meta,
            'fertility': {
                'bpe':    round(bpe_fert,    6), 'fused':  round(fused_fert,  6),
                'opt':    round(opt_fert,    6), 'noback': round(noback_fert, 6),
            },
            'vocab_sizes': {
                'bpe': bpe_tok.get_vocab_size(), 'fused': fused_tok.get_vocab_size(),
                'opt': opt_tok.get_vocab_size(), 'noback': noback_tok.get_vocab_size(),
            },
            'boundary_f1': {'bpe': bpe_f1, 'fused': fused_f1, 'opt': opt_f1, 'noback': noback_f1},
        })

    result = {
        'experiment': 'exp89 — THE ALLOCATION CELL',
        'arms': {
            'bpe':    'BPE-32k (anchor, exp84 lineage)',
            'fused':  'FUSED-F8000 (anchor, exp84 lineage)',
            'opt':    'FUSED-BACKOFF-OPT (F=24000)',
            'noback': 'FUSED-NOBACKOFF (F=16000)',
        },
        'per_seed': per_seed_results,
        'elapsed_s': round(time.time() - t0, 1),
    }
    save_json(result, artifact_path('exp89_evaluate.json'))
    print(f'\n[evaluate] Done in {time.time()-t0:.0f}s → exp89_evaluate.json')
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# §14  VOCAB ACCOUNTING + EXPANSION DETERMINISM GATES
# ═══════════════════════════════════════════════════════════════════════════════

def run_gates(all_lines):
    from tokenizers import Tokenizer
    t0 = time.time()
    print('[gates] Running vocab accounting and expansion-determinism gates…')

    gates_results = []

    for seed in SEEDS:
        print(f'\n[gates] === SEED {seed} ===')

        opt_tok    = Tokenizer.from_file(tok_dir('fused_opt',    seed) + '/tokenizer.json')
        noback_tok = Tokenizer.from_file(tok_dir('fused_noback', seed) + '/tokenizer.json')
        fused_opt    = load_json(fused_opt_set_path(seed))
        fused_noback = load_json(fused_noback_set_path(seed))
        tmpl_skel_map = load_json(tmpl_skel_path(seed))
        n_tmpl = len(tmpl_skel_map)

        # ── Gate A: Actual vocab sizes (doc 76 §4.1: measure real, not arithmetic) ──
        opt_vocab_actual    = opt_tok.get_vocab_size()
        noback_vocab_actual = noback_tok.get_vocab_size()
        gate_a_opt    = (opt_vocab_actual    == VOCAB_SIZE)
        gate_a_noback = (noback_vocab_actual == VOCAB_SIZE)

        n_bpe_opt    = VOCAB_SIZE - 3 - n_tmpl - len(fused_opt)
        n_bpe_noback = VOCAB_SIZE - 3 - len(fused_noback)

        # DTF: train tiny tokenizer with vocab_size=100 (deliberately wrong)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                          encoding='utf-8') as tf:
            tmp_dtf = tf.name
        tok_dtf = train_bpe_tokenizer(['ذهب الولد إلى المدرسة'] * 100, 100,
                                       SPECIAL_TOKENS, tmp_dtf)
        os.unlink(tmp_dtf)
        dtf_vocab   = tok_dtf.get_vocab_size()
        dtf_fires_a = (dtf_vocab != VOCAB_SIZE)

        gate_a = {
            'gate_name': 'VOCAB_ACCOUNTING', 'seed': seed,
            'description': 'Measure REAL trained vocab size; must equal 32000',
            'opt':    {'n_fused': len(fused_opt),   'n_tmpl': n_tmpl,
                       'n_bpe_arithmetic': n_bpe_opt,
                       'vocab_actual': opt_vocab_actual, 'passes': gate_a_opt},
            'noback': {'n_fused': len(fused_noback), 'n_tmpl': 0,
                       'n_bpe_arithmetic': n_bpe_noback,
                       'vocab_actual': noback_vocab_actual, 'passes': gate_a_noback},
            'dtf_leg': {
                'dtf_vocab_trained': dtf_vocab, 'expected': VOCAB_SIZE,
                'dtf_fires': dtf_fires_a,
                'verdict': ('DTF_FIRES: wrong vocab caught' if dtf_fires_a
                            else 'DTF_DOES_NOT_FIRE — UNEXPECTED'),
            },
            'seed_passes': gate_a_opt and gate_a_noback,
        }
        print(f'  Gate A OPT={opt_vocab_actual} {"PASS" if gate_a_opt else "FAIL"}; '
              f'NOBACK={noback_vocab_actual} {"PASS" if gate_a_noback else "FAIL"}; '
              f'DTF: {gate_a["dtf_leg"]["verdict"]}')

        # ── Gate B: Expansion-determinism (every entry has 3-part parts) ─────────
        opt_invalid    = [w for w, fs in fused_opt.items()    if len(fs.get('parts',[])) != 3]
        noback_invalid = [w for w, fs in fused_noback.items() if len(fs.get('parts',[])) != 3]

        fused_opt_dtf = dict(fused_opt)
        fused_opt_dtf['__TEST_INVALID__'] = {
            'root': 'x', 'pattern': 'test', 'parts': ['xx'], 'freq': 0}
        dtf_invalid = [w for w, fs in fused_opt_dtf.items() if len(fs.get('parts',[])) != 3]
        dtf_fires_b = len(dtf_invalid) > 0

        gate_b = {
            'gate_name': 'EXPANSION_DETERMINISM', 'seed': seed,
            'opt':    {'n_checked': len(fused_opt),    'n_invalid': len(opt_invalid),
                       'passes': len(opt_invalid) == 0},
            'noback': {'n_checked': len(fused_noback), 'n_invalid': len(noback_invalid),
                       'passes': len(noback_invalid) == 0},
            'dtf_leg': {
                'n_invalid_after_injection': len(dtf_invalid),
                'dtf_fires': dtf_fires_b,
                'verdict': ('DTF_FIRES: invalid detected' if dtf_fires_b
                            else 'DTF_DOES_NOT_FIRE — UNEXPECTED'),
            },
            'seed_passes': len(opt_invalid) == 0 and len(noback_invalid) == 0,
        }
        print(f'  Gate B OPT n_invalid={len(opt_invalid)} {"PASS" if gate_b["opt"]["passes"] else "FAIL"}; '
              f'NOBACK n_invalid={len(noback_invalid)} {"PASS" if gate_b["noback"]["passes"] else "FAIL"}; '
              f'DTF: {gate_b["dtf_leg"]["verdict"]}')

        gates_results.append({'seed': seed, 'vocab_accounting': gate_a,
                              'expansion_determinism': gate_b})

    all_a = all(g['vocab_accounting']['seed_passes']    for g in gates_results)
    all_b = all(g['expansion_determinism']['seed_passes'] for g in gates_results)
    result = {
        'experiment': 'exp89 gates',
        'per_seed': gates_results,
        'summary': {
            'all_vocab_accounting_pass':      all_a,
            'all_expansion_determinism_pass': all_b,
            'all_gates_pass': all_a and all_b,
        },
        'elapsed_s': round(time.time() - t0, 1),
    }
    save_json(result, artifact_path('exp89_gates.json'))
    print(f'\n[gates] Done → exp89_gates.json')
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# §15  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def load_corpus():
    if not os.path.exists(CORPUS_PATH):
        sys.exit(f'FATAL: corpus not found at {CORPUS_PATH}')
    with open(CORPUS_PATH, encoding='utf-8') as f:
        return [l.rstrip('\n') for l in f if l.strip()]


def main():
    parser = argparse.ArgumentParser(description='exp89: The Allocation Cell')
    parser.add_argument('--anchor-gates', action='store_true')
    parser.add_argument('--build-new',    action='store_true')
    parser.add_argument('--evaluate',     action='store_true')
    parser.add_argument('--gates',        action='store_true')
    parser.add_argument('--all',          action='store_true',
                        help='build-new + evaluate + gates (for nohup)')
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    print('[exp89] Loading corpus…')
    all_lines = load_corpus()
    print(f'  {len(all_lines):,} lines')

    if args.anchor_gates:
        run_anchor_gates(all_lines)

    if args.build_new or args.all:
        run_build_new(all_lines)

    if args.evaluate or args.all:
        run_evaluate(all_lines)

    if args.gates or args.all:
        run_gates(all_lines)


if __name__ == '__main__':
    main()
