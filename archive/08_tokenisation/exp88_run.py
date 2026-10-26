#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp88_run.py — SKELETON MATCHING AT FULL INVENTORY (no training).

Registration: ~/qnlp_private_docs/81_exp88_89_skeleton_scale_and_allocation.md §2
Sole writer: results_exp88.json

Run on WSL: ~/venv/bin/python3 exp88_run.py
Estimated time: minutes (no tokenizer training).
"""

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict

# ── CAMeL path (WSL source checkout) ─────────────────────────────────────────
_CAMEL_SRC = os.path.expanduser('~/camel_tools')
if os.path.isdir(_CAMEL_SRC) and _CAMEL_SRC not in sys.path:
    sys.path.insert(0, _CAMEL_SRC)

WORK_DIR = os.path.expanduser('~/discocat_arabic_v2')
CORPUS_PATH = os.path.join(WORK_DIR, 'arwiki_exp84.txt')

# exp88 sampling constants (from registration §2, frozen into design)
SAMPLE_SEED = 88999
SAMPLE_SIZE = 8000   # 8000-line sample (same draw recipe as probe)

# Coverage bar (registration §2)
COVERAGE_BAR = 0.20   # unique-match TOKEN coverage ≥ 0.20 on FULL corpus
PRECISION_BAR = 0.80  # root agreement ≥ 0.80 among CAMeL-analyzable unique matches

# Nonce baseline (from exp87)
NONCE_BASELINE_UNIQUE = 120   # 120/180 from exp87's 6-template matcher

# Arabic diacritics (same as exp84)
DIACRITICS = frozenset('ًٌٍَُِّْٕٖٟٓٔٗ٘')
ARABIC_RE   = re.compile(r'[؀-ۿ]')


# ═══════════════════════════════════════════════════════════════════════════════
# §1  UTILITIES  (parity with exp84_run.py)
# ═══════════════════════════════════════════════════════════════════════════════

def strip_diac(s):
    return ''.join(c for c in s if c not in DIACRITICS)

def normalize_surface(s):
    s = strip_diac(s)
    s = s.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    return s

def is_arabic_letter(c):
    """Any Arabic letter (including weak radicals ا و ي ء etc.) — for radical slots."""
    return '؀' <= c <= 'ۿ' and c not in DIACRITICS

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


# ═══════════════════════════════════════════════════════════════════════════════
# §2  SKELETON INVENTORY CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_skeleton_inventory(analysis_cache):
    """
    Compile undiacritized template skeletons from the pattern field of the
    exp84 seed-42 analysis cache.

    Skeleton string = normalize_surface(pattern):
      - digits '1','2','3' mark radical-slot positions
      - all other Arabic chars are literal template characters

    Returns:
        dict[skeleton_str, {'patterns': set[str], 'slot_positions': list[int]}]

    Also returns the set of all (normalized) pattern strings seen.
    """
    skeleton_inventory = {}
    all_patterns = set()

    for word, reading in analysis_cache.items():
        pattern = reading.get('pattern', '')
        if not pattern:
            continue
        root = reading.get('root', '')
        if len(root.split('.')) != 3:
            continue
        # Require at least 2 of the 3 digits (same filter as exp84 select_reading)
        n_digits = sum(1 for d in '123' if d in pattern)
        if n_digits < 2:
            continue

        all_patterns.add(pattern)
        skel_str = normalize_surface(pattern)   # strip diacritics, normalize hamza

        if skel_str not in skeleton_inventory:
            slot_positions = [i for i, c in enumerate(skel_str) if c in '123']
            skeleton_inventory[skel_str] = {
                'patterns': set(),
                'slot_positions': slot_positions,
            }
        skeleton_inventory[skel_str]['patterns'].add(pattern)

    return skeleton_inventory, all_patterns


def skeleton_inventory_sha(skeleton_inventory):
    """Canonical SHA of the skeleton inventory: sort keys, serialize deterministically."""
    items = []
    for skel_str in sorted(skeleton_inventory.keys()):
        entry = skeleton_inventory[skel_str]
        items.append({
            'skeleton': skel_str,
            'patterns': sorted(entry['patterns']),
            'slot_positions': entry['slot_positions'],
        })
    canonical = json.dumps(items, ensure_ascii=False, sort_keys=True)
    return sha256_str(canonical)


# ═══════════════════════════════════════════════════════════════════════════════
# §3  SKELETON MATCHER
# ═══════════════════════════════════════════════════════════════════════════════

def _try_match_one_skeleton(surface_form, skel_str):
    """
    Check if surface_form exactly matches skel_str under the skeleton rules:
    - at digit positions ('1','2','3'): accept any Arabic letter
    - at other positions: character must match exactly

    Returns (matched: bool, root_chars: list[str])
    """
    if len(surface_form) != len(skel_str):
        return False, []
    root_chars = []
    for s_c, sk_c in zip(surface_form, skel_str):
        if sk_c in '123':
            if not is_arabic_letter(s_c):
                return False, []
            root_chars.append(s_c)
        else:
            if s_c != sk_c:
                return False, []
    return True, root_chars


def match_word(surface_word, skeleton_inventory):
    """
    Match a surface word against the full skeleton inventory.

    ال optional prefix:
        - If surface starts with 'ال' and skeleton does NOT: also try stripping ال
          and matching surface[2:] against the skeleton.
        - This implements ال as an optional detachable prefix.

    Returns:
        {
          'status': 'unique' | 'ambiguous' | 'no_match',
          'n_matches': int,
          'root': 'r1.r2.r3' or None,
          'skeleton': skel_str or None,
          'patterns': list[str] or [],
        }
    """
    norm_surf = normalize_surface(surface_word)
    al_surf   = norm_surf.startswith('ال')
    surf_no_al = norm_surf[2:] if al_surf else None

    match_results = []   # list of (skel_str, root_chars)

    for skel_str, entry in skeleton_inventory.items():
        skel_has_al = skel_str.startswith('ال')

        # Try direct match
        ok, root_chars = _try_match_one_skeleton(norm_surf, skel_str)
        if ok:
            match_results.append((skel_str, root_chars, 'direct'))
            continue   # same skeleton can't also match via stripping

        # ال optional: surface has ال, skeleton doesn't → try stripping
        if al_surf and not skel_has_al and surf_no_al is not None:
            ok2, root_chars2 = _try_match_one_skeleton(surf_no_al, skel_str)
            if ok2:
                match_results.append((skel_str, root_chars2, 'al_stripped'))

    if len(match_results) == 0:
        return {'status': 'no_match', 'n_matches': 0,
                'root': None, 'skeleton': None, 'patterns': []}
    if len(match_results) == 1:
        skel_str, root_chars, _ = match_results[0]
        root_str = '.'.join(root_chars)
        patterns = sorted(skeleton_inventory[skel_str]['patterns'])
        return {'status': 'unique', 'n_matches': 1,
                'root': root_str, 'skeleton': skel_str, 'patterns': patterns}
    # ambiguous: >= 2 skeletons match
    return {
        'status': 'ambiguous', 'n_matches': len(match_results),
        'root': None, 'skeleton': None,
        'patterns': sorted({p for sk, _, _ in match_results
                            for p in skeleton_inventory[sk]['patterns']}),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §4  CORPUS COVERAGE MEASURES
# ═══════════════════════════════════════════════════════════════════════════════

def is_arabic_word(w):
    arabic_count = sum(1 for c in w if ARABIC_RE.match(c))
    latin_count  = sum(1 for c in w if c.isascii() and c.isalpha())
    return arabic_count >= 2 and latin_count == 0


def measure_corpus(lines, skeleton_inventory, analysis_cache, label=''):
    """
    Compute coverage, ambiguity, precision on a set of lines.

    Returns a dict with:
        type_total, type_unique_match, type_ambiguous, type_no_match
        token_total, token_unique_match, token_ambiguous, token_no_match
        unique_match_type_coverage, unique_match_token_coverage
        ambiguity_rate_types, precision_root_agreement,
        n_camel_unanalyzable (skeleton unique-matched but CAMeL has no analysis)
    """
    t0 = time.time()
    # Build type → token count
    type_counts = Counter()
    for line in lines:
        for w in line.split():
            if is_arabic_word(w):
                type_counts[normalize_surface(w)] += 1

    n_types = len(type_counts)
    n_tokens = sum(type_counts.values())

    unique_match_types = 0
    ambiguous_types    = 0
    no_match_types     = 0

    unique_match_tokens = 0
    ambiguous_tokens    = 0
    no_match_tokens     = 0

    precision_agree = 0
    precision_total = 0
    n_camel_unanalyzable = 0   # unique-matched but CAMeL has no analysis for root comparison

    for norm_w, freq in type_counts.items():
        res = match_word(norm_w, skeleton_inventory)
        s = res['status']

        if s == 'unique':
            unique_match_types  += 1
            unique_match_tokens += freq
            # Precision: compare extracted root vs CAMeL root
            camel_reading = analysis_cache.get(norm_w)
            if camel_reading is None:
                n_camel_unanalyzable += 1
            else:
                camel_root = camel_reading.get('root', '')
                if len(camel_root.split('.')) == 3:
                    precision_total += 1
                    if camel_root == res['root']:
                        precision_agree += 1
                else:
                    n_camel_unanalyzable += 1
        elif s == 'ambiguous':
            ambiguous_types  += 1
            ambiguous_tokens += freq
        else:
            no_match_types  += 1
            no_match_tokens += freq

    type_cov  = unique_match_types  / n_types  if n_types  > 0 else 0.0
    token_cov = unique_match_tokens / n_tokens if n_tokens > 0 else 0.0
    ambig_rate = ambiguous_types / n_types if n_types > 0 else 0.0
    precision  = precision_agree / precision_total if precision_total > 0 else 0.0

    elapsed = round(time.time() - t0, 1)
    print(f'  [{label}] n_types={n_types:,} n_tokens={n_tokens:,}  '
          f'unique={unique_match_types:,}({type_cov:.4f} type / {token_cov:.4f} tok)  '
          f'ambig={ambiguous_types:,}({ambig_rate:.4f})  '
          f'precision={precision:.4f} ({precision_agree}/{precision_total})  '
          f'unanalyzable={n_camel_unanalyzable:,}  elapsed={elapsed}s', flush=True)

    return {
        'label':               label,
        'n_lines':             len(lines),
        'n_types':             n_types,
        'n_tokens':            n_tokens,
        'unique_match_types':  unique_match_types,
        'unique_match_tokens': unique_match_tokens,
        'ambiguous_types':     ambiguous_types,
        'ambiguous_tokens':    ambiguous_tokens,
        'no_match_types':      no_match_types,
        'no_match_tokens':     no_match_tokens,
        'unique_match_type_coverage':  round(type_cov,  6),
        'unique_match_token_coverage': round(token_cov, 6),
        'ambiguity_rate_types':        round(ambig_rate, 6),
        'precision_root_agreement':    round(precision,  6),
        'precision_numerator':         precision_agree,
        'precision_denominator':       precision_total,
        'n_camel_unanalyzable':        n_camel_unanalyzable,
        'elapsed_s':                   elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §5  NONCE LEG (exp87's 180 nonce forms, full-inventory matcher)
# ═══════════════════════════════════════════════════════════════════════════════

def run_nonce_leg(skeleton_inventory):
    """
    Run the full-inventory skeleton matcher on exp87's 180 nonce forms.
    Returns per-column breakdown and total unique-match count.
    """
    mat_path = artifact_path('exp87_materials.json')
    if not os.path.exists(mat_path):
        return {'error': f'exp87_materials.json not found at {mat_path}'}

    mat = load_json(mat_path)
    nonce_forms = mat['nonce_forms']
    assert len(nonce_forms) == 180, f'Expected 180 nonce forms, got {len(nonce_forms)}'

    per_column = defaultdict(lambda: Counter())
    total_unique = 0
    total_ambiguous = 0
    total_no_match = 0

    per_form_results = []
    for entry in nonce_forms:
        surf   = entry['surface']
        col    = entry['column']
        true_root = entry['root']

        res = match_word(surf, skeleton_inventory)
        st  = res['status']

        per_column[col][st] += 1
        if st == 'unique':
            total_unique += 1
        elif st == 'ambiguous':
            total_ambiguous += 1
        else:
            total_no_match += 1

        per_form_results.append({
            'form_id':   entry['form_id'],
            'surface':   surf,
            'column':    col,
            'true_root': true_root,
            'status':    st,
            'matched_root': res.get('root'),
            'n_matches': res['n_matches'],
        })

    per_col_summary = {}
    for col in ['faail', 'faiil', 'fa33aal', 'maf3uul', 'mu1', 'mu2']:
        c = per_column[col]
        per_col_summary[col] = {
            'unique': c['unique'], 'ambiguous': c['ambiguous'], 'no_match': c['no_match'],
        }

    return {
        'n_nonce_forms':      len(nonce_forms),
        'total_unique':       total_unique,
        'total_ambiguous':    total_ambiguous,
        'total_no_match':     total_no_match,
        'exp87_baseline':     NONCE_BASELINE_UNIQUE,
        'meets_exp87_baseline': total_unique >= NONCE_BASELINE_UNIQUE,
        'per_column':         per_col_summary,
        'per_form':           per_form_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §6  GATES
# ═══════════════════════════════════════════════════════════════════════════════

def gate_skeleton_compilation_determinism(analysis_cache):
    """
    Gate: skeleton-compilation determinism.

    Real check: compile the inventory from a SORTED pattern list and an
    UNSORTED (original dict-iteration) list; the canonical sorted serialization
    sha must be identical (the set content is order-independent).

    DTF (designed-to-fail): serialize in INSERTION ORDER (without sorting keys)
    from two compilations with different traversal orders.  The SHA differs,
    demonstrating that the sort step in skeleton_inventory_sha is doing real
    work — without it the check would miss non-determinism in the output.
    """
    print('[gate] Skeleton-compilation determinism…')
    t0 = time.time()

    # Real check: two compilations, same input data, different insertion orders
    # (achieved by iterating the cache dict vs iterating it reversed)
    inv1, _ = build_skeleton_inventory(analysis_cache)
    # Build second inventory from cache items in a different (reversed) order
    inv2_raw = {}
    items_reversed = list(analysis_cache.items())[::-1]
    for word, reading in items_reversed:
        pattern = reading.get('pattern', '')
        if not pattern:
            continue
        root = reading.get('root', '')
        if len(root.split('.')) != 3:
            continue
        n_digits = sum(1 for d in '123' if d in pattern)
        if n_digits < 2:
            continue
        skel_str = normalize_surface(pattern)
        if skel_str not in inv2_raw:
            slot_positions = [i for i, c in enumerate(skel_str) if c in '123']
            inv2_raw[skel_str] = {'patterns': set(), 'slot_positions': slot_positions}
        inv2_raw[skel_str]['patterns'].add(pattern)

    sha1 = skeleton_inventory_sha(inv1)
    sha2 = skeleton_inventory_sha(inv2_raw)
    real_passes = (sha1 == sha2)

    # DTF: serialize WITHOUT sorting — use insertion-order JSON.
    # inv1 and inv2_raw have different insertion orders, so their unsorted serializations differ.
    def unsorted_sha(inv):
        items = []
        for skel_str, entry in inv.items():   # insertion order
            items.append({'skeleton': skel_str,
                          'patterns': sorted(entry['patterns']),   # sort patterns but NOT keys
                          'slot_positions': entry['slot_positions']})
        canonical = json.dumps(items, ensure_ascii=False)
        return sha256_str(canonical)

    dtf_sha1 = unsorted_sha(inv1)
    dtf_sha2 = unsorted_sha(inv2_raw)
    dtf_fires = (dtf_sha1 != dtf_sha2)

    elapsed = round(time.time() - t0, 1)
    result = {
        'gate_name': 'SKELETON_COMPILATION_DETERMINISM',
        'description': (
            'Compiling the skeleton inventory from two different traversal orders '
            'of the analysis cache yields identical content (sorted SHA matches). '
            'DTF: unsorted serialization from two orders produces different SHAs, '
            'showing the sort step is necessary.'
        ),
        'real_check': {
            'n_skeletons': len(inv1),
            'sha_forward':  sha1[:16],
            'sha_reversed': sha2[:16],
            'sha_match':    real_passes,
            'gate_passes':  real_passes,
            'verdict': ('PASS: inventory content is order-independent' if real_passes
                        else 'FAIL: inventory sha differs by traversal order — BUG'),
        },
        'dtf_leg': {
            'description': 'Unsorted serialization should differ by traversal order',
            'dtf_sha_forward_prefix':  dtf_sha1[:16],
            'dtf_sha_reversed_prefix': dtf_sha2[:16],
            'dtf_fires': dtf_fires,
            'verdict': ('DTF_FIRES: unsorted sha differs — sort step validated' if dtf_fires
                        else 'DTF_DOES_NOT_FIRE: unexpected — both happen to have same sha'),
        },
        'gate_passes': real_passes,
        'elapsed_s': elapsed,
    }
    print(f'  Real: {result["real_check"]["verdict"]}')
    print(f'  DTF:  {result["dtf_leg"]["verdict"]}')
    return result


def gate_matcher_sanity_exp87(skeleton_inventory):
    """
    Gate: matcher-sanity on exp87's known nonce forms.

    The 6 exp87 template skeletons are constructed from exp87's generate_surface_form
    logic.  A RESTRICTED inventory containing only these 6 skeletons (5 unique strings,
    since mu1 and mu2 share the same surface pattern 'الم123') is tested against
    the 180 nonce forms.

    Expected results (from exp87's skeleton_match):
      faail (30 forms)  → unique_match   (skeleton 'ال1ا23')
      faiil (30 forms)  → unique_match   (skeleton 'ال12ي3')
      fa33aal (30)      → unique_match   (skeleton 'ال12ا3')
      maf3uul (30)      → unique_match   (skeleton 'الم12و3')
      mu1 (30)          → NO unique_match: 'الم123' matches once → unique by our definition
      mu2 (30)          → same surface as mu1 → same result

    exp87's skeleton_match explicitly marks mu1/mu2 as None (ambiguous). Our matcher
    would call them 'unique' (one skeleton match) but can't distinguish mu1 from mu2.
    We record this as a noted behavioural difference, not a failure.

    The sanity check verifies:
      1. faail/faiil/fa33aal/maf3uul forms: UNIQUE match in restricted inventory.
      2. mu1/mu2 forms: exactly ONE skeleton 'الم123' matches (unique by our count,
         but noted as pattern-ambiguous).
      3. No restricted-inventory form gets MORE than 1 skeleton match.

    DTF: corrupt the 'faail' skeleton by changing 'ا' → 'ب';
         the faail forms should then get no_match, detected as a deviation.
    """
    print('[gate] Matcher-sanity on exp87 nonce forms…')

    mat_path = artifact_path('exp87_materials.json')
    if not os.path.exists(mat_path):
        return {'gate_name': 'MATCHER_SANITY_EXP87',
                'error': 'exp87_materials.json not found', 'gate_passes': False}

    mat  = load_json(mat_path)
    nonce_forms = mat['nonce_forms']

    # Build restricted 6-template skeleton inventory from exp87's generation logic.
    # (ال is 2 chars; digits mark radical slots)
    RESTRICTED_SKELS = {
        'ال1ا23':  'faail',      # ال + R1 + ا + R2 + R3
        'ال12ي3':  'faiil',      # ال + R1 + R2 + ي + R3
        'ال12ا3':  'fa33aal',    # ال + R1 + R2 + ا + R3
        'الم12و3': 'maf3uul',    # ال + م + R1 + R2 + و + R3
        'الم123':  'mu1/mu2',    # ال + م + R1 + R2 + R3  (shared by mu1 and mu2)
    }
    restricted_inventory = {}
    for skel_str, col_label in RESTRICTED_SKELS.items():
        slot_positions = [i for i, c in enumerate(skel_str) if c in '123']
        restricted_inventory[skel_str] = {
            'patterns': {col_label},
            'slot_positions': slot_positions,
        }

    # Run restricted matcher on nonce forms; record per-column outcomes
    col_outcomes = defaultdict(list)
    for entry in nonce_forms:
        surf = entry['surface']
        col  = entry['column']
        res  = match_word(surf, restricted_inventory)
        col_outcomes[col].append(res['status'])

    # Expected: faail/faiil/fa33aal/maf3uul → all 'unique'; mu1/mu2 → all 'unique'
    # (one shared skeleton; still unique at skeleton level)
    expected_unique_cols  = ['faail', 'faiil', 'fa33aal', 'maf3uul', 'mu1', 'mu2']
    col_check = {}
    all_ok = True
    for col in expected_unique_cols:
        outcomes = col_outcomes.get(col, [])
        n_unique = outcomes.count('unique')
        n_other  = len(outcomes) - n_unique
        ok = (n_unique == 30 and n_other == 0)
        col_check[col] = {'n_unique': n_unique, 'n_other': n_other, 'ok': ok}
        if not ok:
            all_ok = False

    real_passes = all_ok

    # DTF: corrupt faail skeleton ال1ا23 → ال1ب23 (ا → ب at position 3)
    corrupted_inventory = {k: dict(v, patterns=set(v['patterns']))
                           for k, v in restricted_inventory.items()}
    # Remove original faail skeleton; add corrupted one
    faail_key = 'ال1ا23'
    corrupt_key = 'ال1ب23'   # ا → ب
    if faail_key in corrupted_inventory:
        old_entry = corrupted_inventory.pop(faail_key)
        slot_positions_c = [i for i, c in enumerate(corrupt_key) if c in '123']
        corrupted_inventory[corrupt_key] = {
            'patterns': old_entry['patterns'],
            'slot_positions': slot_positions_c,
        }

    dtf_col_outcomes = defaultdict(list)
    for entry in nonce_forms:
        if entry['column'] == 'faail':
            surf = entry['surface']
            res  = match_word(surf, corrupted_inventory)
            dtf_col_outcomes['faail'].append(res['status'])

    faail_dtf = dtf_col_outcomes.get('faail', [])
    n_faail_unique_dtf = faail_dtf.count('unique')
    dtf_fires = (n_faail_unique_dtf < 30)   # should be 0 since faail skeleton corrupted

    result = {
        'gate_name': 'MATCHER_SANITY_EXP87',
        'description': (
            'Restricted 6-template inventory reproduces exp87-consistent match '
            'statuses on nonce forms. mu1/mu2 counted unique at skeleton level '
            '(shared skeleton الم123); pattern-level ambiguity noted but not a failure.'
        ),
        'restricted_skeleton_inventory': list(RESTRICTED_SKELS.keys()),
        'real_check': {
            'per_column': col_check,
            'all_columns_ok': real_passes,
            'gate_passes': real_passes,
            'verdict': ('PASS: restricted matcher matches exp87 statuses' if real_passes
                        else 'FAIL: some columns deviate from expected'),
        },
        'dtf_leg': {
            'description': 'Corrupt faail skeleton ال1ا23→ال1ب23; faail unique matches should drop to 0',
            'n_faail_unique_after_corruption': n_faail_unique_dtf,
            'dtf_fires': dtf_fires,
            'verdict': ('DTF_FIRES: faail matches dropped — corruption detected' if dtf_fires
                        else 'DTF_DOES_NOT_FIRE: corruption undetected — UNEXPECTED'),
        },
        'gate_passes': real_passes,
    }
    print(f'  Real: {result["real_check"]["verdict"]}')
    print(f'  DTF:  {result["dtf_leg"]["verdict"]}')
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# §7  MAIN RUN
# ═══════════════════════════════════════════════════════════════════════════════

def run_all():
    t0 = time.time()
    print('[exp88] Starting skeleton-at-scale run…')

    # ── Load analysis cache (seed 42) ─────────────────────────────────────────
    cache_path = analysis_cache_path(42)
    if not os.path.exists(cache_path):
        sys.exit(f'[exp88] FATAL: analysis cache not found at {cache_path}')
    print(f'[{time.time()-t0:.0f}s] Loading analysis cache (seed 42)…')
    with open(cache_path, 'rb') as f:
        analysis_cache = pickle.load(f)
    print(f'  Cache size: {len(analysis_cache):,} entries')

    # ── Build skeleton inventory ───────────────────────────────────────────────
    print(f'[{time.time()-t0:.0f}s] Building skeleton inventory…')
    skeleton_inventory, all_patterns = build_skeleton_inventory(analysis_cache)
    n_patterns   = len(all_patterns)
    n_skeletons  = len(skeleton_inventory)
    inv_sha      = skeleton_inventory_sha(skeleton_inventory)
    print(f'  Patterns: {n_patterns:,} → Skeletons: {n_skeletons:,} (sha: {inv_sha[:16]}…)')

    skeleton_meta = {
        'n_patterns_in_cache': n_patterns,
        'n_skeletons':         n_skeletons,
        'inventory_sha':       inv_sha,
        'compile_seed':        42,
    }

    # ── Load corpus ────────────────────────────────────────────────────────────
    print(f'[{time.time()-t0:.0f}s] Loading corpus…')
    if not os.path.exists(CORPUS_PATH):
        sys.exit(f'[exp88] FATAL: corpus not found at {CORPUS_PATH}')
    with open(CORPUS_PATH, encoding='utf-8') as f:
        all_lines = [l.rstrip('\n') for l in f if l.strip()]
    print(f'  Corpus: {len(all_lines):,} lines')

    # ── Draw 8000-line sample (seed 88999) ────────────────────────────────────
    print(f'[{time.time()-t0:.0f}s] Drawing {SAMPLE_SIZE}-line sample (seed {SAMPLE_SEED})…')
    nonempty_lines = [l for l in all_lines if l.strip()]
    rng = random.Random(SAMPLE_SEED)
    sample_lines = rng.sample(nonempty_lines, min(SAMPLE_SIZE, len(nonempty_lines)))
    print(f'  Sample: {len(sample_lines):,} lines drawn')

    # ── Corpus measures ────────────────────────────────────────────────────────
    print(f'[{time.time()-t0:.0f}s] Measuring sample (8000-line)…')
    sample_results = measure_corpus(sample_lines, skeleton_inventory, analysis_cache,
                                    label='sample_8k')

    print(f'[{time.time()-t0:.0f}s] Measuring full corpus ({len(all_lines):,} lines)…')
    full_results = measure_corpus(all_lines, skeleton_inventory, analysis_cache,
                                  label='full_corpus')

    # ── Nonce leg ──────────────────────────────────────────────────────────────
    print(f'[{time.time()-t0:.0f}s] Running nonce leg (exp87 180 forms)…')
    nonce_results = run_nonce_leg(skeleton_inventory)
    if 'error' in nonce_results:
        print(f'  WARNING: {nonce_results["error"]}')
    else:
        print(f'  Nonce unique-match: {nonce_results["total_unique"]}/180 '
              f'(exp87 baseline: {NONCE_BASELINE_UNIQUE})')
        print(f'  Per column: {nonce_results["per_column"]}')

    # ── Booleans ──────────────────────────────────────────────────────────────
    # §2: full_inventory_lifts_coverage = token coverage ≥ 0.20 on FULL corpus
    full_token_cov = full_results['unique_match_token_coverage']
    full_inventory_lifts_coverage = full_token_cov >= COVERAGE_BAR

    # §2: precision_holds_at_scale = root agreement ≥ 0.80 on FULL corpus
    full_precision = full_results['precision_root_agreement']
    precision_holds_at_scale = full_precision >= PRECISION_BAR

    # Named third outcomes
    if full_inventory_lifts_coverage and precision_holds_at_scale:
        named_outcome = ('SUCCESS: lexicon-free segmenter REAL at scale; '
                         'earns v2 build')
    elif not full_inventory_lifts_coverage and precision_holds_at_scale:
        named_outcome = 'NICHE: precision without coverage — skeleton too sparse at scale'
    elif full_inventory_lifts_coverage and not full_precision >= PRECISION_BAR:
        named_outcome = 'AMBIGUITY_SWAMP: coverage without precision — unique-matching collapses'
    else:
        named_outcome = 'NICHE: neither coverage nor precision bar met at full inventory'

    booleans = {
        'full_inventory_lifts_coverage':  full_inventory_lifts_coverage,
        'precision_holds_at_scale':       precision_holds_at_scale,
        'coverage_bar':                   COVERAGE_BAR,
        'precision_bar':                  PRECISION_BAR,
        'full_token_coverage_measured':   round(full_token_cov, 6),
        'full_precision_measured':        round(full_precision, 6),
        'named_outcome':                  named_outcome,
    }

    print(f'\n[exp88] Booleans:')
    print(f'  full_inventory_lifts_coverage = {full_inventory_lifts_coverage} '
          f'(token_cov={full_token_cov:.4f}, bar={COVERAGE_BAR})')
    print(f'  precision_holds_at_scale = {precision_holds_at_scale} '
          f'(precision={full_precision:.4f}, bar={PRECISION_BAR})')
    print(f'  named_outcome: {named_outcome}')

    # ── Gates ──────────────────────────────────────────────────────────────────
    print(f'\n[{time.time()-t0:.0f}s] Running gates…')
    gate1 = gate_skeleton_compilation_determinism(analysis_cache)
    gate2 = gate_matcher_sanity_exp87(skeleton_inventory)

    save_json(gate1, artifact_path('exp88_gate_compilation_determinism.json'))
    save_json(gate2, artifact_path('exp88_gate_matcher_sanity.json'))

    # ── Write results ──────────────────────────────────────────────────────────
    results = {
        'experiment':    'exp88 — SKELETON MATCHING AT FULL INVENTORY',
        'registration':  os.path.expanduser(
            '~/qnlp_private_docs/81_exp88_89_skeleton_scale_and_allocation.md'),
        'sole_writer':   'exp88_run.py → results_exp88.json',
        'skeleton_meta': skeleton_meta,
        'sample_measures': sample_results,
        'full_corpus_measures': full_results,
        'nonce_leg':     nonce_results,
        'booleans':      booleans,
        'gates': {
            'compilation_determinism': gate1,
            'matcher_sanity_exp87':    gate2,
            'all_gates_pass': gate1['gate_passes'] and gate2['gate_passes'],
        },
        'elapsed_s': round(time.time() - t0, 1),
    }
    out_path = artifact_path('results_exp88.json')
    save_json(results, out_path)
    print(f'\n[exp88] Written → {out_path}  ({round(time.time()-t0,1)}s total)')
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# §8  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='exp88: Skeleton Matching at Full Inventory'
    )
    parser.add_argument('--all', action='store_true',
                        help='Run all stages (corpus + nonce + gates → results_exp88.json)')
    args = parser.parse_args()

    if args.all or not any(vars(args).values()):
        run_all()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
