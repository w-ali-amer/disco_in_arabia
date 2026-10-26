# -*- coding: utf-8 -*-
"""exp90_encoder.py -- doc 83 SS4.5 (G11): the case-vowel extractor.

The vowel is read from the noun's FINAL diacritic cluster after NFC
normalisation: U+064F (damma) -> NOM, U+064E (fatha) -> ACC, none -> NONE.

Trap handled (and used as the designed-to-fail case in the tests): Unicode
canonical ordering puts fatha (ccc 30) and damma (ccc 31) BEFORE shadda
(ccc 33), so after NFC the nisba form  ـيُّ  ends in the SHADDA, not the vowel.
A "last codepoint" reader therefore returns NONE for all three nisba nouns.
This extractor reads the whole trailing run of Arabic combining marks,
ignores shadda, and requires at most one case vowel.

Out-of-domain marks (kasra, tanwin, sukun, ...) raise, so the case channel
can never silently absorb a mark the design does not define.
"""
import unicodedata

NOM, ACC, NONE = "NOM", "ACC", "NONE"
DAMMA, FATHA, SHADDA = "ُ", "َ", "ّ"
# Arabic harakat block U+064B..U+0652 plus superscript alef U+0670
ARABIC_MARKS = {chr(c) for c in range(0x064B, 0x0653)} | {"ٰ"}
CASE_OF = {DAMMA: NOM, FATHA: ACC}
VOWEL_OF = {NOM: DAMMA, ACC: FATHA}


class EncoderError(ValueError):
    pass


def nfc(s):
    return unicodedata.normalize("NFC", s)


def strip_marks(tok):
    """Base letters only (NFC, all Arabic combining marks removed)."""
    return "".join(ch for ch in nfc(tok) if ch not in ARABIC_MARKS)


def trailing_marks(tok):
    t = nfc(tok)
    i = len(t)
    while i > 0 and t[i - 1] in ARABIC_MARKS:
        i -= 1
    return t[i:]


def extract_case(tok):
    """NOM / ACC / NONE from the token's final diacritic cluster."""
    t = nfc(tok)
    if any(ch in ARABIC_MARKS for ch in t[:len(t) - len(trailing_marks(t))]):
        raise EncoderError("internal diacritic outside the design: %r" % tok)
    marks = [m for m in trailing_marks(t) if m != SHADDA]
    bad = [m for m in marks if m not in CASE_OF]
    if bad:
        raise EncoderError("out-of-domain mark(s) %s in %r"
                           % ([hex(ord(b)) for b in bad], tok))
    if len(marks) > 1:
        raise EncoderError("more than one case vowel in %r" % tok)
    return CASE_OF[marks[0]] if marks else NONE


def vocalise_noun(noun, case, shadda_first=True):
    """Rule-based vocalisation (doc 83 SS3.1). Definite nouns, no tanwin.
    Nisba nouns (final ي) take shadda + vowel: ـيُّ / ـيَّ. Returned NFC."""
    if case == NONE:
        return nfc(noun)
    v = VOWEL_OF[case]
    if noun.endswith("ي"):                       # nisba -yy
        s = noun + (SHADDA + v if shadda_first else v + SHADDA)
    else:
        s = noun + v
    return nfc(s)


def vocalise_verb(verb):
    """Past-tense verb with final fatha (doc 83 SS3.1)."""
    return nfc(verb + FATHA)


def naive_last_char_case(tok):
    """DESIGNED-TO-FAIL reference: reads only the last codepoint after NFC.
    Correct for plain nouns, wrong for NFC nisba forms (last char = shadda)."""
    t = nfc(tok)
    return CASE_OF.get(t[-1], NONE) if t else NONE
