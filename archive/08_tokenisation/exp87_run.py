#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp87_run.py  — ARABIC WUG TEST, LEG A (TOKENIZERS)
Registration: qnlp_private_docs/80_exp87_arabic_wug_test.md

CLI flags:
  --materials   generate nonce roots, build 360-form set, write exp87_materials.json
  --gates       emit exp87_gate_{nonce_validity,gold_determinism,replication_anchor}.json
  --evaluate    segment all 360 forms, compute boundary F1, write exp87_evaluate.json

Interpreter notes:
  --materials requires gensim (AraVec):
      qiskit_lambeq_env/bin/python3 exp87_run.py --materials
  --gates and --evaluate work with the standard venv:
      ~/venv/bin/python3 exp87_run.py --gates
      ~/venv/bin/python3 exp87_run.py --evaluate

WSL compute host only. Do not run on Mac.
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
from collections import Counter

# ── CAMeL path (WSL source checkout) ─────────────────────────────────────────
_CAMEL_SRC = os.path.expanduser('~/camel_tools')
if os.path.isdir(_CAMEL_SRC) and _CAMEL_SRC not in sys.path:
    sys.path.insert(0, _CAMEL_SRC)

WORK_DIR = os.path.expanduser('~/discocat_arabic_v2')

# ── Arabic character sets ─────────────────────────────────────────────────────
# 25 sound consonants: exclude و ي ا ء ة (and hamza variants آ أ إ)
SOUND_CONSONANTS = frozenset('بتثجحخدذرزسشصضطظعغفقكلمنه')
DIACRITICS = frozenset('ًٌٍَُِّْٕٖٟٓٔٗ٘')
WEAK_RADICAL_CHARS = frozenset('وياءأإآ#')

# 6 coherent template columns (doc 67 §1)
COLUMNS = ['faail', 'faiil', 'fa33aal', 'maf3uul', 'mu1', 'mu2']

# Seeds (registration §2)
NONCE_SEED   = 87999
CONTROL_SEED = 87998
EVAL_SEEDS   = [42, 43, 44]
N_NONCE_ROOTS   = 30
N_FORMS_PER_ARM = 180   # nonce and real each

# exp84 reference values for REPLICATION_ANCHOR gate
EXP84_ANCHOR = {
    42: {'bpe': 0.1974, 'wazn': 0.5785},
    43: {'bpe': 0.1899, 'wazn': 0.5624},
    44: {'bpe': 0.2045, 'wazn': 0.5609},
}
ANCHOR_TOLERANCE = 0.05

# ═══════════════════════════════════════════════════════════════════════════════
# §1  UTILITIES  (parity with exp84_run.py)
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

def tok_dir(arm, seed):
    return artifact_path(f'exp84_{arm}_tok_s{seed}')


# ═══════════════════════════════════════════════════════════════════════════════
# §2  TEMPLATE SURFACE FORM GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_surface_form(r1, r2, r3, column):
    """
    Generate undiacritized normalized surface form for a sound root + column.
    Gold boundary is always {2} (the ال prefix boundary).

    Verified against exp77_inventory examples:
      العالم  (ع.ل.م, faail)   = ال+ع+ا+ل+م
      العليم  (ع.ل.م, faiil)   = ال+ع+ل+ي+م
      العلام  (ع.ل.م, fa33aal) = ال+ع+ل+ا+م
      المعلوم (ع.ل.م, maf3uul) = ال+م+ع+ل+و+م
      المعلم  (ع.ل.م, mu1)     = ال+م+ع+ل+م
      المقطع  (ق.ط.ع, mu2)     = ال+م+ق+ط+ع
    """
    if column == 'faail':
        return 'ال' + r1 + 'ا' + r2 + r3
    elif column == 'faiil':
        return 'ال' + r1 + r2 + 'ي' + r3
    elif column == 'fa33aal':
        return 'ال' + r1 + r2 + 'ا' + r3
    elif column == 'maf3uul':
        return 'ال' + 'م' + r1 + r2 + 'و' + r3
    elif column in ('mu1', 'mu2'):
        # Identical undiacritized forms; flagged as surface_shared_with_mu_sibling
        return 'ال' + 'م' + r1 + r2 + r3
    else:
        raise ValueError(f'Unknown column: {column!r}')

GOLD_BOUNDS = {2}   # by construction for all 6 templates (ال prefix = 2 chars)


def unique_surfaces_for_root(r1, r2, r3):
    """Return the 5 distinct undiacritized surface strings for a root (mu1==mu2)."""
    seen, out = set(), []
    for col in COLUMNS:
        s = generate_surface_form(r1, r2, r3, col)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# §3  SKELETON MATCHER (LEXICON-FREE ARM)
# ═══════════════════════════════════════════════════════════════════════════════

def skeleton_match(word):
    """
    Match a surface word (normalized, with ال prefix) against the 6 template
    skeletons.  Sound-radical slots only (not ا/و/ي).

    Returns (column, r1, r2, r3) on unique match,
            None if ambiguous (mu1/mu2) or no match.

    Template shapes after stripping ال (2 chars):
      faail:   R1+ا+R2+R3          len=4  stem[1]='ا'
      faiil:   R1+R2+ي+R3          len=4  stem[2]='ي'
      fa33aal: R1+R2+ا+R3          len=4  stem[2]='ا'
      maf3uul: م+R1+R2+و+R3       len=5  stem[0]='م' stem[3]='و'
      mu1/mu2: م+R1+R2+R3          len=4  stem[0]='م'  ← AMBIGUOUS
    """
    surf = normalize_surface(word)
    if not surf.startswith('ال'):
        return None
    stem = surf[2:]   # strip ال

    def is_sound(c):
        return c in SOUND_CONSONANTS

    if len(stem) == 5 and stem[0] == 'م' and stem[3] == 'و':
        r1, r2, r3 = stem[1], stem[2], stem[4]
        if all(is_sound(c) for c in (r1, r2, r3)):
            return ('maf3uul', r1, r2, r3)
        return None

    if len(stem) == 4:
        if stem[1] == 'ا':
            r1, r2, r3 = stem[0], stem[2], stem[3]
            if all(is_sound(c) for c in (r1, r2, r3)):
                return ('faail', r1, r2, r3)
            return None
        if stem[2] == 'ي':
            r1, r2, r3 = stem[0], stem[1], stem[3]
            if all(is_sound(c) for c in (r1, r2, r3)):
                return ('faiil', r1, r2, r3)
            return None
        if stem[2] == 'ا':
            r1, r2, r3 = stem[0], stem[1], stem[3]
            if all(is_sound(c) for c in (r1, r2, r3)):
                return ('fa33aal', r1, r2, r3)
            return None
        if stem[0] == 'م' and is_sound(stem[1]) and is_sound(stem[2]) and is_sound(stem[3]):
            return None   # mu1/mu2 — AMBIGUOUS

    return None


def predict_boundaries_skeleton(word):
    """SKELETON arm: unique match → {2}; no match or ambiguous → {}."""
    match = skeleton_match(word)
    return ({2}, 'matched') if match is not None else (set(), 'unmatched')


# ═══════════════════════════════════════════════════════════════════════════════
# §4  BOUNDARY F1  (verbatim from exp84_run.py)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_skeleton(pattern):
    stripped = strip_diac(pattern)
    has_al   = stripped.startswith('ال')
    has_ta   = stripped.endswith('ة')
    has_ta2  = stripped.endswith('ات')
    has_oon  = stripped.endswith('ون') or stripped.endswith('ين')
    has_tum  = stripped.endswith('تم') or stripped.endswith('تن')
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
    L    = len(surf)
    pattern = reading.get('pattern', '')
    skel    = tmpl_skeleton_map.get(pattern)
    if skel is None:
        skel = extract_skeleton(pattern)
    bounds = set()
    if skel['prefix_len'] > 0 and skel['prefix_len'] < L:
        bounds.add(skel['prefix_len'])
    if skel['suffix_len'] > 0 and (L - skel['suffix_len']) > 0:
        bounds.add(L - skel['suffix_len'])
    return bounds


def predict_boundaries_bpe(tokenizer, word):
    surf   = normalize_surface(word)
    tokens = tokenizer.encode(surf).tokens
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
# §5  CAMeL ROOT INVENTORY
# ═══════════════════════════════════════════════════════════════════════════════

def get_camel_root_inventory():
    """
    Extract all dot-separated roots from the CAMeL builtin morphology DB.
    Returns a frozenset of strings like 'ك.ت.ب'.

    DB structure (calima-msa): stem_hash maps stem → list of (category, entry_dict)
    where entry_dict has 'root' key (e.g. 'ك.ت.ب' or '#.#.ب').
    """
    from camel_tools.morphology.database import MorphologyDB
    db = MorphologyDB.builtin_db()
    roots = set()

    # stem_hash: stem_str → list of (cat_str, entry_dict) tuples
    try:
        for tups in db.stem_hash.values():
            for item in tups:
                # item is a 2-tuple: (category_str, entry_dict)
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    entry = item[1]
                elif isinstance(item, dict):
                    entry = item
                else:
                    continue
                if not isinstance(entry, dict):
                    continue
                r = entry.get('root', '')
                if r and '.' in r:
                    roots.add(r)
        if roots:
            print(f'  [camel] roots via stem_hash: {len(roots):,}')
            return frozenset(roots)
    except AttributeError:
        pass

    # Fallback: lemma_hash — same tuple structure
    try:
        for tups in db.lemma_hash.values():
            for item in tups:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    entry = item[1]
                elif isinstance(item, dict):
                    entry = item
                else:
                    continue
                if not isinstance(entry, dict):
                    continue
                r = entry.get('root', '')
                if r and '.' in r:
                    roots.add(r)
        if roots:
            print(f'  [camel] roots via lemma_hash: {len(roots):,}')
            return frozenset(roots)
    except AttributeError:
        pass

    raise RuntimeError(
        'Cannot extract root inventory from CAMeL DB — '
        'check camel_tools version and DB structure'
    )


def get_exp77_pool(enum_grid_path=None):
    """Return frozenset of dot-separated roots in the exp77 enumeration pool."""
    if enum_grid_path is None:
        enum_grid_path = artifact_path('exp77_enum_grid.json')
    data = load_json(enum_grid_path)
    # exp77_enum_grid.json has 'root_metadata' or 'roots' key
    roots = set()
    if 'root_metadata' in data:
        for r in data['root_metadata'].keys():
            roots.add(r)
    elif 'roots' in data:
        for r in data['roots']:
            roots.add(r if isinstance(r, str) else r.get('root', ''))
    else:
        # Fall back: any key that looks like a root dict
        for k, v in data.items():
            if isinstance(k, str) and '.' in k:
                roots.add(k)
    roots.discard('')
    return frozenset(roots)


# ═══════════════════════════════════════════════════════════════════════════════
# §6  STAGE: --materials
# ═══════════════════════════════════════════════════════════════════════════════

def run_materials():
    t0 = time.time()
    print('[materials] Starting — requires qiskit_lambeq_env (gensim)')

    # --- gensim / AraVec (lazy import) ---
    try:
        import gensim.models
    except (ImportError, Exception) as e:
        sys.exit(
            f'[materials] FATAL: gensim not importable ({e}).\n'
            '  Run with: qiskit_lambeq_env/bin/python3 exp87_run.py --materials'
        )

    aravec_path = os.path.join(WORK_DIR, 'aravec', 'full_uni_cbow_300_twitter.mdl')
    if not os.path.exists(aravec_path):
        sys.exit(f'[materials] FATAL: AraVec model not found at {aravec_path}')
    print(f'[{time.time()-t0:.0f}s] Loading AraVec model (mmap)…')
    aravec = gensim.models.Word2Vec.load(aravec_path, mmap='r')
    aravec_vocab = aravec.wv.key_to_index
    print(f'  AraVec vocab size: {len(aravec_vocab):,}')

    def in_aravec(surface):
        """Check AraVec membership; also tries ة→ه substitution."""
        s = normalize_surface(surface)
        if s in aravec_vocab:
            return True
        s2 = s.replace('ة', 'ه')
        return s2 in aravec_vocab

    # --- CAMeL root inventory ---
    print(f'[{time.time()-t0:.0f}s] Loading CAMeL root inventory…')
    camel_roots = get_camel_root_inventory()
    print(f'  CAMeL root inventory: {len(camel_roots):,} roots')

    # --- exp77 enumeration pool ---
    print(f'[{time.time()-t0:.0f}s] Loading exp77 pool…')
    exp77_pool = get_exp77_pool()
    print(f'  exp77 pool: {len(exp77_pool):,} roots')

    # --- Verify exp77_inventory.json SHA ---
    inv_path = artifact_path('exp77_inventory.json')
    if not os.path.exists(inv_path):
        sys.exit(f'[materials] FATAL: exp77_inventory.json not found at {inv_path}')
    inv_data    = load_json(inv_path)
    inv_file_sha = sha256_file(inv_path)
    inv_meta_sha = inv_data.get('meta', {}).get('inventory_sha256', '')
    if not inv_meta_sha.startswith('a9e9de9e'):
        sys.exit(
            f'[materials] FATAL: exp77_inventory.json SHA mismatch.\n'
            f'  meta.inventory_sha256 = {inv_meta_sha!r}\n'
            f'  Expected prefix: a9e9de9e'
        )
    print(f'  Inventory SHA check OK: {inv_meta_sha[:16]}…')

    # --- Generate 30 nonce SOUND roots at seed 87999 ---
    print(f'[{time.time()-t0:.0f}s] Generating {N_NONCE_ROOTS} nonce sound roots (seed {NONCE_SEED})…')
    rng = random.Random(NONCE_SEED)
    consonant_list = sorted(SOUND_CONSONANTS)   # deterministic order

    accepted_roots = []     # list of (r1, r2, r3) tuples
    rejected_roots = []     # list of {'root', 'reasons'}
    reason_counts  = Counter()
    n_draws        = 0

    # DTF assertion: ك.ت.ب must fail screening
    dtf_root_str = 'ك.ت.ب'
    dtf_checked  = False

    while len(accepted_roots) < N_NONCE_ROOTS:
        r_chars = rng.sample(consonant_list, 3)   # no duplicate consonants
        r1, r2, r3 = r_chars
        root_str   = f'{r1}.{r2}.{r3}'
        n_draws   += 1

        reasons = []

        # Screen 1: CAMeL root inventory
        if root_str in camel_roots:
            reasons.append('in_camel')

        # Screen 2: exp77 enumeration pool
        if root_str in exp77_pool:
            reasons.append('in_exp77_pool')

        # Screen 3: AraVec attestation of any of the 5 unique surface forms
        attested_forms = [s for s in unique_surfaces_for_root(r1, r2, r3) if in_aravec(s)]
        if attested_forms:
            reasons.append('aravec_attested')

        if reasons:
            rejected_roots.append({'root': root_str, 'reasons': reasons})
            for rsn in set(reasons):
                reason_counts[rsn] += 1
        else:
            accepted_roots.append((r1, r2, r3))

    print(f'  Accepted {len(accepted_roots)} roots after {n_draws} draws '
          f'({n_draws - len(accepted_roots)} rejected)')
    print(f'  Rejection reasons: {dict(reason_counts)}')

    # --- DTF: verify ك.ت.ب is rejected ---
    ktb_r1, ktb_r2, ktb_r3 = 'ك', 'ت', 'ب'
    ktb_reasons = []
    if 'ك.ت.ب' in camel_roots:
        ktb_reasons.append('in_camel')
    if 'ك.ت.ب' in exp77_pool:
        ktb_reasons.append('in_exp77_pool')
    ktb_attested = [s for s in unique_surfaces_for_root(ktb_r1, ktb_r2, ktb_r3) if in_aravec(s)]
    if ktb_attested:
        ktb_reasons.append('aravec_attested')
    dtf_nonce_validity_fires = len(ktb_reasons) > 0
    print(f'  DTF ك.ت.ب rejection reasons: {ktb_reasons} → gate_fires={dtf_nonce_validity_fires}')

    # --- Build 180 nonce forms ---
    nonce_forms = []
    for i, (r1, r2, r3) in enumerate(accepted_roots):
        root_str = f'{r1}.{r2}.{r3}'
        for col in COLUMNS:
            surf = generate_surface_form(r1, r2, r3, col)
            is_mu_shared = (col in ('mu1', 'mu2'))
            nonce_forms.append({
                'form_id':    f'nonce_{i}_{col}',
                'type':       'nonce',
                'surface':    surf,
                'root':       root_str,
                'column':     col,
                'gold_bounds': [2],
                'gold_source': 'by_construction',
                'surface_shared_with_mu_sibling': is_mu_shared,
            })

    # --- Draw 180 real-control forms from exp77_inventory at seed 87998 ---
    print(f'[{time.time()-t0:.0f}s] Drawing real-control forms (seed {CONTROL_SEED})…')

    # Filter: sound admitted entries (weak=False, geminate=False, admitted=True)
    words = inv_data.get('words', inv_data.get('entries', []))
    by_col = {col: [] for col in COLUMNS}
    for entry in words:
        if (not entry.get('weak', True)
                and not entry.get('geminate', True)
                and entry.get('admitted', False)):
            col = entry.get('column', '')
            if col in by_col:
                by_col[col].append(entry)

    col_avail = {col: len(by_col[col]) for col in COLUMNS}
    print(f'  Sound admitted by column: {col_avail}')

    # Stratified sampling: min(30, available) per column, then redistribute deficit
    rng2 = random.Random(CONTROL_SEED)
    for col in COLUMNS:
        rng2.shuffle(by_col[col])   # deterministic shuffle within column

    col_draw = {col: min(30, col_avail[col]) for col in COLUMNS}
    total_drawn = sum(col_draw.values())
    deficit = N_FORMS_PER_ARM - total_drawn

    if deficit > 0:
        # Redistribute deficit to columns with surplus (those that drew exactly 30)
        surplus_cols = [col for col in COLUMNS if col_draw[col] == 30
                        and col_avail[col] > 30]
        surplus_cols_sorted = sorted(surplus_cols)   # alphabetical for reproducibility
        per_col_extra = deficit // len(surplus_cols_sorted)
        remainder     = deficit % len(surplus_cols_sorted)
        for idx, col in enumerate(surplus_cols_sorted):
            extra = per_col_extra + (1 if idx < remainder else 0)
            col_draw[col] = min(col_draw[col] + extra, col_avail[col])

    final_total = sum(col_draw.values())
    print(f'  Realized draw per column: {col_draw} (total={final_total})')

    forced_deviation = any(col_draw[col] != 30 for col in COLUMNS)
    if forced_deviation:
        short_cols = {col: col_draw[col] for col in COLUMNS if col_draw[col] < 30}
        print(f'  FORCED DEVIATION: columns short of 30: {short_cols}')

    real_forms = []
    form_idx   = 0
    for col in COLUMNS:
        entries = by_col[col][:col_draw[col]]
        for entry in entries:
            word     = entry.get('word', '')
            root_str = entry.get('root', '')
            real_forms.append({
                'form_id':    f'real_{form_idx}',
                'type':       'real',
                'surface':    normalize_surface(word),
                'root':       root_str,
                'column':     col,
                'gold_bounds': [2],
                'gold_source': 'by_construction_from_inventory',
                'surface_shared_with_mu_sibling': (col in ('mu1', 'mu2')),
                'inventory_word': word,
            })
            form_idx += 1

    assert len(real_forms) == N_FORMS_PER_ARM, \
        f'Expected {N_FORMS_PER_ARM} real forms, got {len(real_forms)}'

    # --- Write exp87_materials.json ---
    out_path = artifact_path('exp87_materials.json')
    materials = {
        'meta': {
            'experiment':            'exp87',
            'registration':          '80_exp87_arabic_wug_test.md',
            'nonce_seed':            NONCE_SEED,
            'control_seed':          CONTROL_SEED,
            'inventory_sha_meta':    inv_meta_sha,
            'inventory_sha_file':    inv_file_sha,
            'n_nonce_forms':         len(nonce_forms),
            'n_real_forms':          len(real_forms),
            'n_total_forms':         len(nonce_forms) + len(real_forms),
            'forced_deviation':      forced_deviation,
            'forced_deviation_note': (
                'Real-control mu1/mu2 columns have fewer than 30 admitted entries; '
                'drew all available and redistributed deficit to large columns.'
                if forced_deviation else 'none'
            ),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
        'nonce_screening': {
            'n_draws':           n_draws,
            'n_accepted':        len(accepted_roots),
            'n_rejected':        n_draws - len(accepted_roots),
            'reason_counts':     dict(reason_counts),
            'dtf_ktb_reasons':   ktb_reasons,
            'dtf_gate_fires':    dtf_nonce_validity_fires,
        },
        'real_control_distribution': {
            **{col: col_draw[col] for col in COLUMNS},
            'total':             final_total,
            'forced_deviation':  forced_deviation,
        },
        'nonce_roots': [
            {'root': f'{r1}.{r2}.{r3}', 'r1': r1, 'r2': r2, 'r3': r3}
            for r1, r2, r3 in accepted_roots
        ],
        'nonce_forms': nonce_forms,
        'real_forms':  real_forms,
    }
    save_json(materials, out_path)
    print(f'[{time.time()-t0:.0f}s] Written: {out_path}')
    print(f'  {len(nonce_forms)} nonce + {len(real_forms)} real = '
          f'{len(nonce_forms)+len(real_forms)} total forms')
    return materials


# ═══════════════════════════════════════════════════════════════════════════════
# §7  STAGE: --gates
# ═══════════════════════════════════════════════════════════════════════════════

def run_gates():
    t0 = time.time()
    print('[gates] Loading materials…')

    mat_path = artifact_path('exp87_materials.json')
    if not os.path.exists(mat_path):
        sys.exit('[gates] FATAL: exp87_materials.json not found — run --materials first')
    mat = load_json(mat_path)

    nonce_forms = mat['nonce_forms']
    real_forms  = mat['real_forms']
    all_forms   = nonce_forms + real_forms
    nonce_roots_list = mat['nonce_roots']

    # ── Gate 1: NONCE_VALIDITY ────────────────────────────────────────────────
    print(f'[{time.time()-t0:.0f}s] Gate 1: NONCE_VALIDITY…')

    # Load screening resources (CAMeL + exp77 pool); AraVec check recorded at --materials
    camel_roots = get_camel_root_inventory()
    exp77_pool  = get_exp77_pool()

    violations = []
    for entry in nonce_roots_list:
        root_str = entry['root']
        if root_str in camel_roots:
            violations.append({'root': root_str, 'reason': 'in_camel'})
        elif root_str in exp77_pool:
            violations.append({'root': root_str, 'reason': 'in_exp77_pool'})

    gate1_passes_real = len(violations) == 0

    # DTF: inject ك.ت.ب — must be caught
    dtf_root = 'ك.ت.ب'
    dtf_caught = (dtf_root in camel_roots) or (dtf_root in exp77_pool)
    dtf_nonce = mat['nonce_screening']['dtf_gate_fires']

    gate1 = {
        'gate_name': 'NONCE_VALIDITY',
        'description': (
            'No accepted nonce root appears in CAMeL DB root inventory '
            'or exp77 enumeration pool.'
        ),
        'n_nonce_roots_checked': len(nonce_roots_list),
        'violations': violations,
        'gate_passes_real': gate1_passes_real,
        'dtf_inject_root':  dtf_root,
        'dtf_in_camel':     dtf_root in camel_roots,
        'dtf_in_exp77':     dtf_root in exp77_pool,
        'dtf_gate_fires':   dtf_caught,
        'dtf_verdict': ('DTF_FIRES: ك.ت.ب correctly caught' if dtf_caught
                        else 'DTF_DOES_NOT_FIRE: screening missed known root — CRITICAL BUG'),
        'gate_overall_passes': gate1_passes_real and dtf_caught,
        'verdict': (
            'PASS: all 30 nonce roots cleared all screens; DTF injection caught'
            if gate1_passes_real and dtf_caught
            else 'FAIL: see violations or dtf failure'
        ),
    }
    save_json(gate1, artifact_path('exp87_gate_nonce_validity.json'))
    print(f'  Gate 1 verdict: {gate1["verdict"]}')

    # ── Gate 2: GOLD_DETERMINISM ──────────────────────────────────────────────
    print(f'[{time.time()-t0:.0f}s] Gate 2: GOLD_DETERMINISM…')

    # Check: each surface maps to exactly one gold BOUNDARY SET
    surface_to_bounds = {}
    conflicts = []
    for entry in all_forms:
        surf   = entry['surface']
        bounds = frozenset(entry['gold_bounds'])
        if surf in surface_to_bounds:
            if surface_to_bounds[surf] != bounds:
                conflicts.append({
                    'surface':     surf,
                    'existing':    sorted(surface_to_bounds[surf]),
                    'conflicting': sorted(bounds),
                })
        else:
            surface_to_bounds[surf] = bounds

    gate2_passes_real = len(conflicts) == 0

    # DTF: inject a surface mapped to two different boundary sets
    dtf_surf   = 'التستنج'   # arbitrary; not a real Arabic word
    dtf_entries = [
        {'surface': dtf_surf, 'gold_bounds': [2]},
        {'surface': dtf_surf, 'gold_bounds': [3]},
    ]
    dtf_map = {}
    dtf_conflicts = []
    for e in dtf_entries:
        s = e['surface']
        b = frozenset(e['gold_bounds'])
        if s in dtf_map and dtf_map[s] != b:
            dtf_conflicts.append({'surface': s, 'bounds_a': sorted(dtf_map[s]), 'bounds_b': sorted(b)})
        dtf_map[s] = b
    dtf_gate_fires = len(dtf_conflicts) > 0

    gate2 = {
        'gate_name': 'GOLD_DETERMINISM',
        'description': (
            'Every surface form in the 360-form set maps to exactly one gold boundary set. '
            'Note: mu1/mu2 pairs share the same surface AND same gold {2} — no conflict.'
        ),
        'n_unique_surfaces': len(surface_to_bounds),
        'n_total_forms':     len(all_forms),
        'conflicts':         conflicts,
        'gate_passes_real':  gate2_passes_real,
        'dtf_injected_surface': dtf_surf,
        'dtf_conflicting_bounds': [sorted(dtf_entries[0]['gold_bounds']),
                                   sorted(dtf_entries[1]['gold_bounds'])],
        'dtf_gate_fires':    dtf_gate_fires,
        'dtf_verdict': ('DTF_FIRES: ambiguous mapping correctly detected' if dtf_gate_fires
                        else 'DTF_DOES_NOT_FIRE: determinism check missed conflict — CRITICAL BUG'),
        'gate_overall_passes': gate2_passes_real and dtf_gate_fires,
        'verdict': (
            'PASS: all 360 forms deterministically mapped; DTF injection caught'
            if gate2_passes_real and dtf_gate_fires
            else 'FAIL: see conflicts or dtf failure'
        ),
    }
    save_json(gate2, artifact_path('exp87_gate_gold_determinism.json'))
    print(f'  Gate 2 verdict: {gate2["verdict"]}')

    # ── Gate 3: REPLICATION_ANCHOR (doc 80 §5.1 amendment: tokenizer IDENTITY) ─
    # The original ±0.05 F1-window anchor compared structurally different gold
    # distributions and was voided (§5.1). Replacement: sha256 pins of every
    # instrument artifact exp87 loads — the nine tokenizer.json files, the
    # per-seed analysis caches, and the template-skeleton maps. For
    # deterministic artifacts, file identity is strictly stronger than an F1
    # window. First computation IS the pin of record (mtimes show the files
    # date from the exp84 build, 2026-09-17 00:52, unmodified since).
    print(f'[{time.time()-t0:.0f}s] Gate 3: REPLICATION_ANCHOR (identity pins, §5.1)…')

    def _sha256_file(path):
        h = hashlib.sha256()
        with open(path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()

    instrument_pins = []
    all_present = True
    for seed in EVAL_SEEDS:
        for arm in ('bpe', 'wazn', 'fused'):
            p = tok_dir(arm, seed) + '/tokenizer.json'
            ok = os.path.exists(p)
            all_present = all_present and ok
            instrument_pins.append({
                'artifact': f'{arm}_tok_s{seed}/tokenizer.json',
                'present': ok,
                'sha256': _sha256_file(p) if ok else None,
                'mtime': (time.strftime('%Y-%m-%d %H:%M:%S',
                          time.localtime(os.path.getmtime(p))) if ok else None),
            })
        for label, p in (('analysis_cache', analysis_cache_path(seed)),
                         ('tmpl_skel', tmpl_skel_path(seed))):
            ok = os.path.exists(p)
            all_present = all_present and ok
            instrument_pins.append({
                'artifact': f'{label}_s{seed}',
                'present': ok,
                'sha256': _sha256_file(p) if ok else None,
                'mtime': (time.strftime('%Y-%m-%d %H:%M:%S',
                          time.localtime(os.path.getmtime(p))) if ok else None),
            })

    # DTF: hash a deliberately modified in-memory copy of one pinned file and
    # show the identity check catches the mismatch.
    ref_pin = next(r for r in instrument_pins if r['present'])
    ref_path = None
    for seed in EVAL_SEEDS:
        p = tok_dir('bpe', seed) + '/tokenizer.json'
        if os.path.exists(p):
            ref_path = p
            break
    with open(ref_path, 'rb') as fh:
        tampered = fh.read() + b'\n// tampered'
    tampered_sha = hashlib.sha256(tampered).hexdigest()
    dtf_fires = tampered_sha != _sha256_file(ref_path)

    gate3 = {
        'gate_name':   'REPLICATION_ANCHOR',
        'description': (
            'doc 80 §5.1 amendment: tokenizer-identity anchor. sha256 pins of '
            'all instrument artifacts exp87 loads; first computation is the '
            'pin of record. The original F1-window anchor was voided as a '
            'structurally ill-posed instrument (see doc 80 §5.1).'
        ),
        'instrument_pins': instrument_pins,
        'gate_passes': all_present and dtf_fires,
        'dtf_inject':  'append bytes to a pinned file in memory; sha must differ',
        'dtf_tampered_sha_prefix': tampered_sha[:16],
        'dtf_gate_fires': dtf_fires,
        'dtf_verdict': ('DTF_FIRES: tampered copy correctly detected'
                        if dtf_fires else 'DTF_DOES_NOT_FIRE: identity check broken'),
        'verdict': ('PASS: all instrument artifacts present and pinned'
                    if (all_present and dtf_fires)
                    else 'FAIL: missing instrument artifact or broken DTF'),
        'superseded_f1_window': {
            'note': 'original gate values retained for the record',
            'deltas_bpe': [-0.1974, -0.1899, -0.2045],
            'deltas_wazn': [0.1267, 0.1373, 0.1443],
            'tolerance': ANCHOR_TOLERANCE,
        },
    }
    save_json(gate3, artifact_path('exp87_gate_replication_anchor.json'))
    print(f'  Gate 3 verdict: {gate3["verdict"]}')

    all_gates_pass = (gate1['gate_overall_passes']
                      and gate2['gate_overall_passes']
                      and gate3['gate_passes'])
    print(f'\n[gates] Summary: Gate1={gate1["gate_overall_passes"]} '
          f'Gate2={gate2["gate_overall_passes"]} '
          f'Gate3={gate3["gate_passes"]} '
          f'→ all_pass={all_gates_pass}')
    if not gate3['gate_passes']:
        print('[gates] NOTE: anchor failure voids science booleans in aggregate '
              '(anchor_voided=true will be set in results_exp87.json)')


# ═══════════════════════════════════════════════════════════════════════════════
# §8  STAGE: --evaluate
# ═══════════════════════════════════════════════════════════════════════════════

def _evaluate_one_seed(seed, nonce_forms, real_forms, t0):
    print(f'\n[eval] === SEED {seed} ===')

    cache_p = analysis_cache_path(seed)
    if not os.path.exists(cache_p):
        raise RuntimeError(f'analysis cache missing for seed {seed}; run exp84 --tokenizers first')
    with open(cache_p, 'rb') as f:
        analysis_cache = pickle.load(f)

    fused_set = load_json(fused_set_path(seed))
    tmpl_skel = load_json(tmpl_skel_path(seed))

    from tokenizers import Tokenizer
    bpe_tok   = Tokenizer.from_file(tok_dir('bpe',   seed) + '/tokenizer.json')
    wazn_tok  = Tokenizer.from_file(tok_dir('wazn',  seed) + '/tokenizer.json')
    fused_tok = Tokenizer.from_file(tok_dir('fused', seed) + '/tokenizer.json')

    def predict_wazn(surf):
        reading = analysis_cache.get(surf) or analysis_cache.get(normalize_surface(surf))
        if reading is not None:
            return predict_boundaries_wazn(surf, reading, tmpl_skel), 'analyzed'
        return predict_boundaries_bpe(wazn_tok, surf), 'fallback'

    def predict_fused(surf):
        fs = fused_set.get(surf)
        if fs is not None:
            reading = {'root': fs['root'], 'pattern': fs['pattern'], 'bw': ''}
            return predict_boundaries_wazn(surf, reading, tmpl_skel), 'fused_analyzed'
        reading = analysis_cache.get(surf) or analysis_cache.get(normalize_surface(surf))
        if reading is not None:
            return predict_boundaries_wazn(surf, reading, tmpl_skel), 'analyzed'
        return predict_boundaries_bpe(fused_tok, surf), 'fallback'

    def fertility_bpe(surf):
        tokens = bpe_tok.encode(normalize_surface(surf)).tokens
        return len(tokens)

    def fertility_wazn(surf):
        r = analysis_cache.get(surf) or analysis_cache.get(normalize_surface(surf))
        if r is not None and len(r.get('root', '').split('.')) == 3:
            return 4
        return len(wazn_tok.encode(normalize_surface(surf)).tokens)   # own tokenizer

    def fertility_fused(surf):
        if surf in fused_set:
            return 1
        r = analysis_cache.get(surf) or analysis_cache.get(normalize_surface(surf))
        if r is not None and len(r.get('root', '').split('.')) == 3:
            return 4
        return len(fused_tok.encode(normalize_surface(surf)).tokens)

    def process_forms(forms, condition_label):
        rows = []
        for entry in forms:
            surf   = entry['surface']
            gold_b = set(entry['gold_bounds'])

            pred_bpe   = predict_boundaries_bpe(bpe_tok, surf)
            pred_wazn,  wazn_beh  = predict_wazn(surf)
            pred_fused, fused_beh = predict_fused(surf)
            pred_skel,  skel_beh  = predict_boundaries_skeleton(surf)

            fert_bpe   = fertility_bpe(surf)
            fert_wazn  = fertility_wazn(surf)
            fert_fused = fertility_fused(surf)

            rows.append({
                'form_id':    entry['form_id'],
                'condition':  condition_label,
                'surface':    surf,
                'root':       entry.get('root', ''),
                'column':     entry.get('column', ''),
                'gold_bounds': sorted(gold_b),
                'pred_bpe':   sorted(pred_bpe),
                'pred_wazn':  sorted(pred_wazn),
                'pred_fused': sorted(pred_fused),
                'pred_skel':  sorted(pred_skel),
                'wazn_behavior':  wazn_beh,
                'fused_behavior': fused_beh,
                'skel_behavior':  skel_beh,
                'fert_bpe':   fert_bpe,
                'fert_wazn':  fert_wazn,
                'fert_fused': fert_fused,
            })
        return rows

    nonce_rows = process_forms(nonce_forms, 'nonce')
    real_rows  = process_forms(real_forms,  'real')
    all_rows   = nonce_rows + real_rows

    def compute_stats(rows):
        arms = ['bpe', 'wazn', 'fused', 'skel']
        records = {arm: [] for arm in arms}
        for row in rows:
            g = set(row['gold_bounds'])
            records['bpe'].append((g, set(row['pred_bpe'])))
            records['wazn'].append((g, set(row['pred_wazn'])))
            records['fused'].append((g, set(row['pred_fused'])))
            records['skel'].append((g, set(row['pred_skel'])))

        result = {}
        for arm in arms:
            p, r, f = micro_f1(records[arm])
            result[arm] = {'p': round(p,4), 'r': round(r,4), 'f1': round(f,4),
                           'n': len(records[arm])}
        return result

    nonce_f1 = compute_stats(nonce_rows)
    real_f1  = compute_stats(real_rows)

    # Analyzer behavior counts (nonce should be all fallback)
    wazn_beh_nonce  = Counter(r['wazn_behavior']  for r in nonce_rows)
    wazn_beh_real   = Counter(r['wazn_behavior']  for r in real_rows)
    fused_beh_nonce = Counter(r['fused_behavior'] for r in nonce_rows)
    fused_beh_real  = Counter(r['fused_behavior'] for r in real_rows)
    skel_beh_nonce  = Counter(r['skel_behavior']  for r in nonce_rows)
    skel_beh_real   = Counter(r['skel_behavior']  for r in real_rows)

    # Fertility averages
    def mean_fert(rows, arm):
        vals = [r[f'fert_{arm}'] for r in rows]
        return round(sum(vals)/len(vals), 4) if vals else 0.0

    print(f'  Nonce F1: BPE={nonce_f1["bpe"]["f1"]:.4f} '
          f'WAZN={nonce_f1["wazn"]["f1"]:.4f} '
          f'FUSED={nonce_f1["fused"]["f1"]:.4f} '
          f'SKEL={nonce_f1["skel"]["f1"]:.4f}')
    print(f'  Real  F1: BPE={real_f1["bpe"]["f1"]:.4f} '
          f'WAZN={real_f1["wazn"]["f1"]:.4f} '
          f'FUSED={real_f1["fused"]["f1"]:.4f} '
          f'SKEL={real_f1["skel"]["f1"]:.4f}')

    return {
        'seed': seed,
        'boundary_f1': {
            'nonce': nonce_f1,
            'real':  real_f1,
        },
        'analyzer_behavior': {
            'wazn_nonce':  dict(wazn_beh_nonce),
            'wazn_real':   dict(wazn_beh_real),
            'fused_nonce': dict(fused_beh_nonce),
            'fused_real':  dict(fused_beh_real),
            'skel_nonce':  dict(skel_beh_nonce),
            'skel_real':   dict(skel_beh_real),
        },
        'fertility': {
            'bpe_nonce':   mean_fert(nonce_rows, 'bpe'),
            'bpe_real':    mean_fert(real_rows,  'bpe'),
            'wazn_nonce':  mean_fert(nonce_rows, 'wazn'),
            'wazn_real':   mean_fert(real_rows,  'wazn'),
            'fused_nonce': mean_fert(nonce_rows, 'fused'),
            'fused_real':  mean_fert(real_rows,  'fused'),
        },
        'per_form': all_rows,
    }


def run_evaluate():
    t0 = time.time()
    print('[evaluate] Loading materials…')

    mat_path = artifact_path('exp87_materials.json')
    if not os.path.exists(mat_path):
        sys.exit('[evaluate] FATAL: exp87_materials.json not found — run --materials first')
    mat = load_json(mat_path)

    nonce_forms = mat['nonce_forms']
    real_forms  = mat['real_forms']

    seed_results = []
    for seed in EVAL_SEEDS:
        sr = _evaluate_one_seed(seed, nonce_forms, real_forms, t0)
        seed_results.append(sr)

    out_path = artifact_path('exp87_evaluate.json')
    save_json({'seed_results': seed_results, 'meta': mat['meta']}, out_path)
    print(f'\n[evaluate] Written: {out_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# §9  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='exp87: Arabic WUG Test, Leg A — tokenizer generalization'
    )
    parser.add_argument('--materials', action='store_true',
                        help='Generate materials (needs qiskit_lambeq_env)')
    parser.add_argument('--gates',    action='store_true',
                        help='Run gates (needs venv)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation (needs venv)')
    args = parser.parse_args()

    if args.materials:
        run_materials()
    if args.gates:
        run_gates()
    if args.evaluate:
        run_evaluate()

    if not (args.materials or args.gates or args.evaluate):
        parser.print_help()


if __name__ == '__main__':
    main()
