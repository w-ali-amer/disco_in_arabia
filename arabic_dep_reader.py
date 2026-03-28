# -*- coding: utf-8 -*-
"""
arabic_dep_reader.py
--------------------
Converts Arabic sentences (via Stanza dependency parses) into lambeq Grammar
Diagrams that *always* reduce to the sentence type s.

Handles:
  SVO  – subject before transitive verb before object
  VSO  – verb first, then subject, then object  (most common Arabic)
  SV   – subject before intransitive verb
  VS   – verb first, then subject (intransitive)
  NOM  – nominal sentence: noun/pronoun subject + adj/noun predicate
  FALLBACK – any other structure; guaranteed cod == s

All output Diagrams satisfy:  diagram.cod == Ty('s')
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ── lambeq grammar primitives ─────────────────────────────────────────────
from lambeq.backend.grammar import Ty, Cup, Id, Word, Diagram, Swap

N = Ty('n')   # noun type
S = Ty('s')   # sentence type

# ── Analysis backend  (reuse camel_test2 which already works) ─────────────
try:
    from camel_test2 import analyze_arabic_sentence_with_morph
    _ANALYSIS_OK = True
    logger.info("arabic_dep_reader: analysis backend loaded from camel_test2.")
except ImportError as _e:
    _ANALYSIS_OK = False
    logger.error(f"arabic_dep_reader: could not import camel_test2 ({_e}). "
                  "Call sentence_to_diagram_from_parse() directly.")
    def analyze_arabic_sentence_with_morph(s, debug=False):
        return [], [], "ERROR", {}


# ═══════════════════════════════════════════════════════════════════════════
#  MORPHOLOGICAL TAG HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _morph_tag(analysis: Dict) -> str:
    """
    Build a short morphological tag from CAMeL analysis fields.

    Uses:
      asp — aspect: 'p' (perfect/past), 'i' (imperfect/present), 'c' (command)
      per — person:  '1', '2', '3', 'na'
      num — number:  's' (singular), 'd' (dual), 'p' (plural), 'na'
      gen — gender:  'm' (masc), 'f' (fem), 'na'

    Returns '' when no useful features are present (e.g. for determiners).
    """
    ca = analysis.get('camel_analysis', {}) or {}
    parts = []
    asp = ca.get('asp', 'na')
    if asp and asp != 'na':
        parts.append(f"ASP-{asp}")
    per = ca.get('per', 'na')
    if per and per != 'na':
        parts.append(f"PER-{per}")
    num = ca.get('num', 'na')
    if num and num != 'na':
        parts.append(f"NUM-{num}")
    gen = ca.get('gen', 'na')
    if gen and gen != 'na':
        parts.append(f"GEN-{gen}")
    return ('_' + '_'.join(parts)) if parts else ''


def _enriched(word: str, analysis: Optional[Dict]) -> str:
    """Return word text enriched with morphological tag when analysis is available."""
    if analysis is None:
        return word
    tag = _morph_tag(analysis)
    return word + tag if tag else word


# ═══════════════════════════════════════════════════════════════════════════
#  WORD-BOX FACTORIES  (each returns a lambeq Word with the correct type)
# ═══════════════════════════════════════════════════════════════════════════

def _noun(word: str) -> Word:
    """Plain noun: type  n"""
    return Word(word, N)

def _verb_trans_svo(word: str) -> Word:
    """Transitive verb (SVO order): n.r @ s @ n.l
       Cancels with subject on left, object on right."""
    return Word(word, N.r @ S @ N.l)

def _verb_trans_vso(word: str) -> Word:
    """Transitive verb (VSO order): s @ n.l @ n.l
       Produces s, then cancels two nouns to the right via cups+swap."""
    return Word(word, S @ N.l @ N.l)

def _verb_intrans_sv(word: str) -> Word:
    """Intransitive verb (SV order): n.r @ s"""
    return Word(word, N.r @ S)

def _verb_intrans_vs(word: str) -> Word:
    """Intransitive verb (VS / verb-first order): s @ n.l"""
    return Word(word, S @ N.l)

def _predicate(word: str) -> Word:
    """Adjectival / nominal predicate in a nominal sentence: n.r @ s"""
    return Word(word, N.r @ S)


# ═══════════════════════════════════════════════════════════════════════════
#  DIAGRAM BUILDERS  (each returns a Diagram with cod == S)
# ═══════════════════════════════════════════════════════════════════════════

def _svo(subj: str, verb: str, obj: str) -> Diagram:
    """
    SVO:  n  ⊗  (n.r ⊗ s ⊗ n.l)  ⊗  n   →   s
    Cups: Cup(n, n.r) on the left  +  Cup(n.l, n) on the right.
    """
    words = _noun(subj) @ _verb_trans_svo(verb) @ _noun(obj)
    cups  = Cup(N, N.r) @ Id(S) @ Cup(N.l, N)
    return words >> cups


def _vso(verb: str, subj: str, obj: str) -> Diagram:
    """
    VSO:  (s ⊗ n.l ⊗ n.l)  ⊗  n  ⊗  n   →   s
    Requires a Swap to make the two n.l types adjacent to their partner n:
      After tensor:       s  n.l  n.l  n  n
      After swap pos 2,3: s  n.l  n    n.l  n
      Cup both pairs:     s
    """
    words = _verb_trans_vso(verb) @ _noun(subj) @ _noun(obj)
    # words.cod = s @ n.l @ n.l @ n @ n
    swap  = Id(S) @ Id(N.l) @ Swap(N.l, N) @ Id(N)
    # after swap: s @ n.l @ n @ n.l @ n
    cups  = Id(S) @ Cup(N.l, N) @ Cup(N.l, N)
    return words >> swap >> cups


def _sv(subj: str, verb: str) -> Diagram:
    """
    SV:  n  ⊗  (n.r ⊗ s)   →   s
    """
    words = _noun(subj) @ _verb_intrans_sv(verb)
    return words >> (Cup(N, N.r) @ Id(S))


def _vs(verb: str, subj: str) -> Diagram:
    """
    VS:  (s ⊗ n.l)  ⊗  n   →   s
    """
    words = _verb_intrans_vs(verb) @ _noun(subj)
    return words >> (Id(S) @ Cup(N.l, N))


def _nominal(subj: str, pred: str) -> Diagram:
    """
    Nominal:  n  ⊗  (n.r ⊗ s)   →   s   (same topology as SV)
    Used for:  الجو  جميل   /   الطالبة  مجتهدة
    """
    words = _noun(subj) @ _predicate(pred)
    return words >> (Cup(N, N.r) @ Id(S))


def _fallback(tokens: List[str], analyses: List[Dict]) -> Diagram:
    """
    Robust fallback guaranteed to produce cod == s.

    Tries to extract 1–2 content words and compose a minimal valid diagram.
    If everything fails, wraps the whole sentence in a single s-typed box.
    """
    content = [_enriched(a['text'], a) for a in analyses
               if a.get('upos') in ('NOUN', 'VERB', 'PROPN', 'ADJ', 'NUM')]
    if not content:
        content = [_enriched(a['text'], a) for a in analyses[:2]]
    content = content[:3]

    try:
        if len(content) >= 2:
            return _nominal(content[0], content[1])
        elif len(content) == 1:
            return _nominal(content[0], '_pred')
    except Exception:
        pass

    # Last resort: one sentence-level box
    key = '_'.join(t for t in tokens[:3] if t) or 'sentence'
    return Word(key, S)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def sentence_to_diagram_from_parse(
    tokens: List[str],
    analyses: List[Dict[str, Any]],
    structure: str,
    roles: Dict[str, Any],
    debug: bool = False,
) -> Diagram:
    """
    Build a lambeq Diagram from the output of analyze_arabic_sentence_with_morph.

    Args:
        tokens:    list of token strings
        analyses:  list of analysis dicts (one per token)
        structure: structure label from camel_test2, e.g. 'SVO', 'VSO', 'SV', ...
        roles:     roles dict with keys 'subject', 'verb', 'object', 'predicate_idx', ...
        debug:     enable verbose logging

    Returns:
        Diagram with cod == Ty('s')
    """
    def _tok(idx) -> Optional[str]:
        if idx is not None and 0 <= idx < len(analyses):
            return analyses[idx]['text']
        return None

    def _etok(idx) -> Optional[str]:
        """Return morphology-enriched word text for token at idx."""
        if idx is not None and 0 <= idx < len(analyses):
            return _enriched(analyses[idx]['text'], analyses[idx])
        return None

    subj_idx = roles.get('subject')
    verb_idx = roles.get('verb')
    obj_idx  = roles.get('object')
    pred_idx = roles.get('predicate_idx')

    # ── Verb-rescue: if Stanza missed the verb as root, find first VERB-tagged word ──
    if verb_idx is None:
        for i, ana in enumerate(analyses):
            if ana.get('upos') == 'VERB':
                verb_idx = i
                # If subj is also None, look for nsubj in dependency graph
                if subj_idx is None:
                    dep_graph = roles.get('dependency_graph', {})
                    for dep_i, dep_r in dep_graph.get(i, []):
                        if dep_r in ('nsubj', 'nsubj:pass', 'csubj'):
                            subj_idx = dep_i
                            break
                    # fallback: first NOUN/PROPN that isn't the verb itself
                    if subj_idx is None:
                        for j, a2 in enumerate(analyses):
                            if j != i and a2.get('upos') in ('NOUN', 'PROPN', 'PRON'):
                                subj_idx = j
                                break
                logger.debug(f"Verb-rescue: using idx={verb_idx} ({analyses[verb_idx]['text']})")
                break

    subj_str = _tok(subj_idx)
    verb_str = _tok(verb_idx)
    obj_str  = _tok(obj_idx)
    pred_str = _tok(pred_idx)

    # enriched versions (with morphological tags appended)
    e_subj = _etok(subj_idx)
    e_verb = _etok(verb_idx)
    e_obj  = _etok(obj_idx)
    e_pred = _etok(pred_idx)

    if debug:
        logger.debug(f"Structure={structure!r}  subj={subj_str!r}→{e_subj!r}  "
                     f"verb={verb_str!r}→{e_verb!r}  obj={obj_str!r}  pred={pred_str!r}")

    diagram = None
    try:
        # ── Transitive verbal ────────────────────────────────────────────
        if subj_str and verb_str and obj_str:
            if subj_idx < verb_idx:               # subject BEFORE verb → SVO
                diagram = _svo(e_subj, e_verb, e_obj)
                logger.debug("→ SVO")
            else:                                  # verb BEFORE subject  → VSO
                diagram = _vso(e_verb, e_subj, e_obj)
                logger.debug("→ VSO")

        # ── Intransitive verbal ──────────────────────────────────────────
        elif subj_str and verb_str:
            if subj_idx < verb_idx:
                diagram = _sv(e_subj, e_verb)
                logger.debug("→ SV")
            else:
                diagram = _vs(e_verb, e_subj)
                logger.debug("→ VS")

        # ── Nominal (subject + predicate, no verb) ───────────────────────
        elif subj_str and pred_str:
            diagram = _nominal(e_subj, e_pred)
            logger.debug("→ Nominal")

        # ── Verb + object only (no explicit subject) ────────────────────
        elif verb_str and obj_str:
            diagram = _vs(e_verb, e_obj)
            logger.debug("→ VO-as-VS")

        # ── Single verb ──────────────────────────────────────────────────
        elif verb_str:
            root_idx = roles.get('root')
            root_str = _tok(root_idx)
            fake_subj = root_str or (tokens[0] if tokens else 'subj')
            diagram = _vs(e_verb, fake_subj)
            logger.debug("→ V-only-as-VS")

    except Exception as exc:
        logger.warning(f"Diagram build raised {exc!r}, using fallback.")
        diagram = None

    if diagram is None:
        diagram = _fallback(tokens, analyses)
        logger.debug("→ fallback")

    # ── Final safety ────────────────────────────────────────────────────
    if diagram.cod != S:
        logger.error(f"Diagram cod={diagram.cod}, expected s. Replacing with sentence box.")
        key = '_'.join(tokens[:3]) if tokens else 'sentence'
        diagram = Word(key, S)

    return diagram


def sentence_to_diagram(sentence: str, debug: bool = False) -> Diagram:
    """
    Full pipeline: parse Arabic sentence and return a lambeq Diagram with cod == s.
    """
    if not sentence or not sentence.strip():
        return Word('empty', S)

    try:
        tokens, analyses, structure, roles = analyze_arabic_sentence_with_morph(
            sentence, debug=debug
        )
    except Exception as exc:
        logger.error(f"Parse error for '{sentence[:30]}': {exc}")
        return Word(sentence[:15], S)

    if not tokens:
        return Word(sentence[:15], S)

    return sentence_to_diagram_from_parse(tokens, analyses, structure, roles, debug=debug)


def sentences_to_diagrams(
    sentences: List[str],
    debug: bool = False,
    log_interval: int = 25,
) -> List[Diagram]:
    """Convert a list of Arabic sentences to lambeq Diagrams (all cod == s)."""
    diagrams = []
    for i, sent in enumerate(sentences):
        try:
            d = sentence_to_diagram(sent, debug=debug)
        except Exception as exc:
            logger.error(f"[{i}] Unhandled error for '{sent[:30]}': {exc}")
            d = Word(sent[:15], S)
        diagrams.append(d)
        if log_interval and (i + 1) % log_interval == 0:
            logger.info(f"  Built {i+1}/{len(sentences)} diagrams.")
    return diagrams
