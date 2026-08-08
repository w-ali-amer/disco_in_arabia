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

# ── Modifier-extension switch ─────────────────────────────────────────────
# When True, the four modifier rules (attributive adjective, idafa, post-verbal
# adverb, prepositional adjunct) also *extend* the core SVO/VSO/SV/VS/nominal
# patterns (i.e. they fire on sentences a core rule already handles, adding the
# previously-dropped modifier wiring).  When False (default) the core dispatch
# is byte-identical to the original reader — the modifier rules then only fire
# on sentences that would otherwise reach _fallback (pure promotions, so they
# can never regress a diagram a core rule already produced).  This flag exists
# because the greedy dispatch already handles modifier-bearing sentences via
# core rules (dropping the modifier); enabling enrichment therefore changes
# those diagrams, which the regression gate forbids for the live pipeline.
ENRICH_MODIFIERS = False

# ── CAMeL-POS fusion switch ───────────────────────────────────────────────
# Stanza systematically reads masdar-capable sentence-initial verbs (فتح,
# رفع, قطع …) as nouns, parsing VSO clauses as iḍāfa NPs ("the man's opening
# of the door") — no token gets upos VERB, so verb-rescue never fires and the
# sentence falls through to nominal/fallback.  When this flag is on (env
# ARABIC_POS_FUSION=1, read at call time) and Stanza found NO verb anywhere,
# a conservative rescue asks the CAMeL analyzer whether the sentence-initial
# token has a verb reading; if it does and at least two nominals follow, the
# clause is rebuilt as VSO.  Verbs never carry the definite article, so
# ال-initial subjects can never trigger a spurious verb reading.  Off by
# default: the live pipeline and all published numbers are byte-identical.
import os

def _camel_has_verb_reading(token_text: str) -> bool:
    try:
        import camel_test2
        if getattr(camel_test2, 'CAMEL_ANALYZER', None) is None:
            return False
        for ana in camel_test2.CAMEL_ANALYZER.analyze(token_text) or []:
            if isinstance(ana, dict) and ana.get('pos') == 'verb':
                return True
    except Exception as exc:
        logger.debug(f"POS-fusion CAMeL query failed for {token_text!r}: {exc!r}")
    return False

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


# ═══════════════════════════════════════════════════════════════════════════
#  MODIFIER RULES  (attributive adjective, idafa, adverb, prepositional adjunct)
#  ---------------------------------------------------------------------------
#  Word-box types (dual conventions match the existing core builders):
#     adjective  n.r @ n       [noun][adj]                → Cup(n,n.r)      → n
#     mudaf N1   n  @ n.l       [mudaf N1][mudaf-ilayh N2] → Cup(n.l,n)      → n
#     adverb     s.r @ s        [clause][adverb]           → Cup(s,s.r)      → s
#     prep       s.r @ s @ n.l  [clause][prep][obj]        → Cup(s,s.r)+Cup(n.l,n) → s
# ═══════════════════════════════════════════════════════════════════════════

def _adj(word: str) -> Word:
    """Attributive adjective (noun-then-adjective order): n.r @ n"""
    return Word(word, N.r @ N)

def _mudaf(word: str) -> Word:
    """Idafa head (mudaf), first noun of a construct: n @ n.l"""
    return Word(word, N @ N.l)

def _adverb(word: str) -> Word:
    """Post-verbal adverb, modifies the whole clause: s.r @ s"""
    return Word(word, S.r @ S)

def _prep(word: str) -> Word:
    """Preposition of a clause-modifying PP adjunct: s.r @ s @ n.l"""
    return Word(word, S.r @ S @ N.l)


def _np_idafa(n1: str, n2: str) -> Diagram:
    """
    Idafa construct  N1 N2  →  n
    (mudaf) n @ n.l  ⊗  (mudaf-ilayh) n   →   n   via Cup(n.l, n).
    """
    words = _mudaf(n1) @ _noun(n2)
    return words >> (Id(N) @ Cup(N.l, N))


# ── NP-aware core builders: identical topology to _svo/_vso/_sv/_vs/_nominal,
#    but accept a pre-built noun-phrase Diagram (cod == n) in the subject/object
#    slot, so an adjective- or idafa-enriched NP can be plugged straight in.
#    When the NP slots are plain _noun(...) states these produce byte-identical
#    diagrams to the original string builders. ────────────────────────────────

def _svo_np(subj_np: Diagram, verb: str, obj_np: Diagram) -> Diagram:
    words = subj_np @ _verb_trans_svo(verb) @ obj_np
    cups  = Cup(N, N.r) @ Id(S) @ Cup(N.l, N)
    return words >> cups

def _vso_np(verb: str, subj_np: Diagram, obj_np: Diagram) -> Diagram:
    words = _verb_trans_vso(verb) @ subj_np @ obj_np
    swap  = Id(S) @ Id(N.l) @ Swap(N.l, N) @ Id(N)
    cups  = Id(S) @ Cup(N.l, N) @ Cup(N.l, N)
    return words >> swap >> cups

def _sv_np(subj_np: Diagram, verb: str) -> Diagram:
    words = subj_np @ _verb_intrans_sv(verb)
    return words >> (Cup(N, N.r) @ Id(S))

def _vs_np(verb: str, subj_np: Diagram) -> Diagram:
    words = _verb_intrans_vs(verb) @ subj_np
    return words >> (Id(S) @ Cup(N.l, N))

def _nominal_np(subj_np: Diagram, pred: str) -> Diagram:
    words = subj_np @ _predicate(pred)
    return words >> (Cup(N, N.r) @ Id(S))


def _attach_adverb(clause: Diagram, adverb: str) -> Diagram:
    """clause(s) ⊗ adverb(s.r @ s)  →  s   via Cup(s, s.r)."""
    d = clause @ _adverb(adverb)
    return d >> (Cup(S, S.r) @ Id(S))

def _attach_pp(clause: Diagram, prep: str, obj: str) -> Diagram:
    """clause(s) ⊗ prep(s.r @ s @ n.l) ⊗ obj(n) → s  via Cup(s,s.r)+Cup(n.l,n)."""
    d = clause @ _prep(prep) @ _noun(obj)
    return d >> (Cup(S, S.r) @ Id(S) @ Cup(N.l, N))


def _build_extended(tokens: List[str], analyses: List[Dict], roles: Dict):
    """
    Try to build a modifier-enriched diagram (cod == s) directly from the
    Stanza dependency tree.  Returns (Diagram, tag) when at least one in-scope
    modifier — attributive adjective (amod), idafa (adjacent nmod noun),
    post-verbal adverb (obl/advmod ADJ/ADV, no case child) or prepositional
    adjunct (obl noun WITH a case child) — is detected and wired; otherwise
    None (so the caller keeps the original behaviour).

    Conservative by design: only fires when the core skeleton (verb+subject,
    or subject+predicate) is confidently identifiable, so it does not mangle
    Stanza mis-parses into spurious constructions.
    """
    n = len(analyses)
    if n == 0:
        return None
    dg = roles.get('dependency_graph', {}) or {}

    def deps_of(i):
        return dg.get(i, [])
    def has_case_child(i):
        return any(r == 'case' for _, r in deps_of(i))
    def enr(i):
        return _enriched(analyses[i]['text'], analyses[i])

    used = set()

    def build_np(head_idx):
        """Noun phrase (cod n) for head_idx, optionally enriched with idafa
        (adjacent nmod noun) and/or an attributive adjective (amod)."""
        used.add(head_idx)
        cons = []
        # idafa: first nmod NOUN/PROPN dependent immediately following the head
        idafa_idx = None
        for di, dr in deps_of(head_idx):
            if di in used:
                continue
            da = analyses[di]
            if dr in ('nmod', 'nmod:poss') and da['upos'] in ('NOUN', 'PROPN') \
                    and di == head_idx + 1 \
                    and not tokens[head_idx].startswith('\u0627\u0644'):
                # true idafa: the mudaf never carries the definite article
                # (reviewer tightening #1; adjectival idafa is the documented
                # exception, absent from current datasets)
                idafa_idx = di
                break
        if idafa_idx is not None:
            used.add(idafa_idx)
            dia = _np_idafa(enr(head_idx), enr(idafa_idx))
            cons.append('idafa')
        else:
            dia = _noun(enr(head_idx))
        # attributive adjective: an amod ADJ dependent that follows the head
        adj_idx = None
        for di, dr in deps_of(head_idx):
            if di in used:
                continue
            if dr == 'amod' and analyses[di]['upos'] == 'ADJ' and di > head_idx:
                adj_idx = di
                break
        if adj_idx is not None:
            used.add(adj_idx)
            dia = (dia @ _adj(enr(adj_idx))) >> (Cup(N, N.r) @ Id(N))
            cons.append('amod')
        return dia, cons

    constructions = []

    # ── Verb-headed clause ────────────────────────────────────────────────
    verb_idx = roles.get('verb')
    if verb_idx is None or analyses[verb_idx]['upos'] != 'VERB':
        verb_idx = None
        for i, a in enumerate(analyses):
            if a['upos'] == 'VERB':
                verb_idx = i
                break

    if verb_idx is not None and analyses[verb_idx]['upos'] == 'VERB':
        subj_idx = None
        obj_idx = None
        for di, dr in deps_of(verb_idx):
            if dr in ('nsubj', 'nsubj:pass', 'csubj') and subj_idx is None \
                    and analyses[di]['upos'] in ('NOUN', 'PROPN', 'PRON'):
                subj_idx = di
            elif dr in ('obj', 'iobj') and obj_idx is None \
                    and analyses[di]['upos'] in ('NOUN', 'PROPN'):
                obj_idx = di
        if subj_idx is None:
            return None  # no confident subject → leave to existing dispatch
        vw = enr(verb_idx)
        if obj_idx is not None:
            snp, sc = build_np(subj_idx)
            onp, oc = build_np(obj_idx)
            constructions += sc + oc
            if subj_idx < verb_idx:
                clause, core = _svo_np(snp, vw, onp), 'SVO'
            else:
                clause, core = _vso_np(vw, snp, onp), 'VSO'
        else:
            snp, sc = build_np(subj_idx)
            constructions += sc
            if subj_idx < verb_idx:
                clause, core = _sv_np(snp, vw), 'SV'
            else:
                clause, core = _vs_np(vw, snp), 'VS'

        # post-verbal adverb (obl/advmod ADJ/ADV, no preposition child)
        for di, dr in deps_of(verb_idx):
            if di in used or di in (subj_idx, obj_idx):
                continue
            da = analyses[di]
            if dr in ('advmod', 'obl') and da['upos'] in ('ADJ', 'ADV') \
                    and not has_case_child(di):
                used.add(di)
                clause = _attach_adverb(clause, enr(di))
                constructions.append('adverb')
                break

        # prepositional adjunct (obl noun WITH a case/preposition child)
        for di, dr in deps_of(verb_idx):
            if di in used or di in (subj_idx, obj_idx):
                continue
            da = analyses[di]
            if dr in ('obl', 'obl:arg', 'nmod') and da['upos'] in ('NOUN', 'PROPN') \
                    and has_case_child(di):
                prep_i = None
                for ci, cr in deps_of(di):
                    if cr == 'case':
                        prep_i = ci
                        break
                if prep_i is not None:
                    used.add(di)
                    used.add(prep_i)
                    clause = _attach_pp(clause, enr(prep_i), enr(di))
                    constructions.append('pp')
                    break

        if not constructions:
            return None
        return clause, core + '+' + '+'.join(constructions)

    # ── Nominal clause (no verb): subject + predicate, enrich the subject ──
    subj_idx = roles.get('subject')
    if subj_idx is None:
        subj_idx = roles.get('root')
    pred_idx = roles.get('predicate_idx')
    if subj_idx is None or pred_idx is None or pred_idx == subj_idx:
        return None
    if not (0 <= subj_idx < n and 0 <= pred_idx < n):
        return None
    used.add(pred_idx)   # protect the predicate from being consumed as a modifier
    snp, sc = build_np(subj_idx)
    constructions += sc
    if not constructions:
        return None
    clause = _nominal_np(snp, enr(pred_idx))
    return clause, 'NOM+' + '+'.join(constructions)


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

    # ── CAMeL-POS fusion rescue (opt-in, see flag comment at top) ────────
    if os.environ.get("ARABIC_POS_FUSION", "0") == "1" and len(analyses) >= 3:
        def _nominal_candidate(i):
            # NOUN/PROPN/PRON, plus ال-definite ADJ/X (Stanza tags
            # profession nouns like النجار as ADJ or X); exclude
            # prepositional objects.
            a = analyses[i]
            if a.get('upos') not in ('NOUN', 'PROPN', 'PRON', 'ADJ', 'X'):
                return False
            if a.get('upos') in ('ADJ', 'X') and not a['text'].startswith('ال'):
                return False
            if i > 0 and analyses[i - 1].get('upos') == 'ADP':
                return False
            # fused-preposition tokens (بالميدالية = ب+ال…) are PP heads,
            # not bare nominals — promoting them fabricates transitivity
            if a['text'].startswith(('بال', 'كال', 'لل')):
                return False
            return True
        # Rule 1: no verb anywhere — masdar mis-parse ("قطع النجار الخشب"
        # read as an NP).  CAMeL confirms a verb reading for token 0.
        if (verb_idx is None
                and analyses[0].get('upos') in ('NOUN', 'PROPN', 'X', 'ADJ')
                and _camel_has_verb_reading(analyses[0]['text'])):
            nominals = [i for i in range(1, len(analyses))
                        if _nominal_candidate(i)]
            if len(nominals) >= 2:
                verb_idx, subj_idx, obj_idx = 0, nominals[0], nominals[1]
                pred_idx = None
                logger.debug(f"POS-fusion rule 1: VSO "
                             f"verb={analyses[0]['text']!r} "
                             f"subj_idx={subj_idx} obj_idx={obj_idx}")
        # Rule 2: verb + subject found but the object was swallowed as an
        # iḍāfa modifier of the subject ("فتح الرجل الباب" parsed VS with
        # الباب attached to الرجل) — promote the first free nominal after
        # both back to object.
        elif (verb_idx is not None and subj_idx is not None
                and obj_idx is None):
            for k in range(max(verb_idx, subj_idx) + 1, len(analyses)):
                if k not in (verb_idx, subj_idx) and _nominal_candidate(k):
                    obj_idx = k
                    logger.debug(f"POS-fusion rule 2: promoted obj_idx={k} "
                                 f"({analyses[k]['text']!r})")
                    break
        # Rule 3: VO_NO_SUBJ with two free post-verbal nominals is VSO —
        # Stanza read "فتح الرجل الباب" as pro-drop "he opened the man's
        # door"; post-verbal NP1 is the subject, NP2 the object.  Genuine
        # pro-drop clauses have no second nominal and stay untouched.
        elif (verb_idx is not None and subj_idx is None
                and obj_idx is not None and obj_idx > verb_idx):
            for k in range(obj_idx + 1, len(analyses)):
                if k != verb_idx and _nominal_candidate(k):
                    subj_idx, obj_idx = obj_idx, k
                    logger.debug(f"POS-fusion rule 3: VSO subj_idx={subj_idx} "
                                 f"obj_idx={obj_idx}")
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

    # ── Modifier-enrichment layer (opt-in via ENRICH_MODIFIERS) ──────────
    #    Extends the core patterns with attributive adjectives / idafa /
    #    adverbs / PP adjuncts.  Off by default → core dispatch below runs
    #    unchanged and every non-fallback diagram stays byte-identical.
    if ENRICH_MODIFIERS:
        try:
            _ext = _build_extended(tokens, analyses, roles)
        except Exception as _exc:
            logger.warning(f"Extended build raised {_exc!r}; ignoring.")
            _ext = None
        if _ext is not None:
            diagram, _tag = _ext
            logger.debug(f"→ {_tag}")

    if diagram is None:
        try:
            # ── Transitive verbal ────────────────────────────────────────
            if subj_str and verb_str and obj_str:
                if subj_idx < verb_idx:               # subject BEFORE verb → SVO
                    diagram = _svo(e_subj, e_verb, e_obj)
                    logger.debug("→ SVO")
                else:                                  # verb BEFORE subject  → VSO
                    diagram = _vso(e_verb, e_subj, e_obj)
                    logger.debug("→ VSO")

            # ── Intransitive verbal ──────────────────────────────────────
            elif subj_str and verb_str:
                if subj_idx < verb_idx:
                    diagram = _sv(e_subj, e_verb)
                    logger.debug("→ SV")
                else:
                    diagram = _vs(e_verb, e_subj)
                    logger.debug("→ VS")

            # ── Nominal (subject + predicate, no verb) ───────────────────
            elif subj_str and pred_str:
                diagram = _nominal(e_subj, e_pred)
                logger.debug("→ Nominal")

            # ── Verb + object only (no explicit subject) ────────────────
            elif verb_str and obj_str:
                diagram = _vs(e_verb, e_obj)
                logger.debug("→ VO-as-VS")

            # ── Single verb ──────────────────────────────────────────────
            elif verb_str:
                root_idx = roles.get('root')
                root_str = _tok(root_idx)
                fake_subj = root_str or (tokens[0] if tokens else 'subj')
                diagram = _vs(e_verb, fake_subj)
                logger.debug("→ V-only-as-VS")

        except Exception as exc:
            logger.warning(f"Diagram build raised {exc!r}, using fallback.")
            diagram = None

    # ── Modifier-rescue layer (always on) ────────────────────────────────
    #    Runs only when the core dispatch produced nothing, i.e. exactly the
    #    sentences that would otherwise reach _fallback.  It can therefore
    #    only PROMOTE a fallback to a real modifier rule, never regress a
    #    diagram a core rule already produced.
    if diagram is None:
        try:
            _resc = _build_extended(tokens, analyses, roles)
        except Exception as _exc:
            logger.warning(f"Rescue build raised {_exc!r}; ignoring.")
            _resc = None
        if _resc is not None:
            diagram, _tag = _resc
            logger.debug(f"→ {_tag}")

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
