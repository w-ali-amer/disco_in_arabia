# -*- coding: utf-8 -*-
"""exp92_common.py -- doc 87 (REGISTERED 59f6621): shared lexicon, surface realisation, S-T compiler, IO.

The compiler reads SURFACE tokens only: noun 1 -> wire 0, noun 2 (object or PP noun) -> wire 1; the verb
token -> (root, form) by lookup of its vocalised surface. Case vowels and the preposition are NOT compiled
(doc 87 §1 compiler choice). Role is never assigned.
"""
import os, json, hashlib, unicodedata, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "wazn_b1", "frozen_lexicon_v1.json")
FROZEN_SHA = "d4bf1bccf4c1bf81766225f5e56fbe2f215362447bdfaec0384d0545680caf5d"

NOM, ACC, GEN, FATHA = "ُ", "َ", "ِ", "َ"
HARAKAT = "ًٌٍَُِّْٰ"
PREP = "في"

INAN = ["الحجر", "الريح", "الماء", "النار", "الشمس", "الغصن", "الباب", "الزجاج", "الحبل", "الثلج", "الخشب", "الجدار"]
HUMAN = ["الطالب", "الرجل", "الولد", "المدير", "المعلم", "الطبيب", "العامل", "المهندس", "الشرطي", "الكاتب", "الطباخ", "الجندي"]
NOUNS = INAN + HUMAN
HUMAN_ROOTS = {"هزم/انهزم", "شغل/انشغل", "فرح/أفرح", "غضب/أغضب", "خجل/أخجل", "تعب/أتعب", "قلق/أقلق", "خاف/أخاف"}

# State surfaces (doc 87 §2; build ruling §12.1): regular pattern-derived participle/adjective of the INTRANSITIVE
# member -- VII active participle (munfaʕil) for ANTI; Form I faaʕil participle or faʕil adjective for CAUS.
# Cosmetic only (the circuit sees a state ID). AVN participles (exp92_state_surfaces.json) are a cross-check:
# bare-letter matching there picked Form II participles and AVN typos for several roots.
STATE_SURFACE = {
    "ثنى/انثنى": "مُنْثَنٍ", "حجب/انحجب": "مُنْحَجِب", "خلع/انخلع": "مُنْخَلِع", "سكب/انسكب": "مُنْسَكِب",
    "شغل/انشغل": "مُنْشَغِل", "شق/انشق": "مُنْشَقّ", "عزل/انعزل": "مُنْعَزِل", "عكس/انعكس": "مُنْعَكِس",
    "فتح/انفتح": "مُنْفَتِح", "فصل/انفصل": "مُنْفَصِل", "فك/انفك": "مُنْفَكّ", "فلق/انفلق": "مُنْفَلِق",
    "قطع/انقطع": "مُنْقَطِع", "كسر/انكسر": "مُنْكَسِر", "كشف/انكشف": "مُنْكَشِف", "هزم/انهزم": "مُنْهَزِم",
    "برز/أبرز": "بارِز", "تعب/أتعب": "تَعِب", "تلف/أتلف": "تالِف", "خاف/أخاف": "خائِف", "خجل/أخجل": "خَجِل",
    "ذاب/أذاب": "ذائِب", "ظهر/أظهر": "ظاهِر", "غرق/أغرق": "غارِق", "غضب/أغضب": "غاضِب", "فرح/أفرح": "فَرِح",
    "فسد/أفسد": "فاسِد", "قلق/أقلق": "قَلِق", "نبت/أنبت": "نابِت",
}

# undergoer position (0 = noun after the verb, 1 = object / PP noun) per (alternation, form)
UNDERGOER_POS = {("ANTI", "I"): 1, ("ANTI", "VII"): 0, ("CAUS", "I"): 0, ("CAUS", "IV"): 1}
TRANSITIVE = {("ANTI", "I"): True, ("ANTI", "VII"): False, ("CAUS", "I"): False, ("CAUS", "IV"): True}


def nfc(s):
    return unicodedata.normalize("NFC", s)


def verb_surface(lex):
    """CAMeL citation diac -> surface: wasla ٱ -> ا, final fatha unless the stem ends in a long vowel."""
    s = nfc(lex).replace("ٱ", "ا")
    if s[-1] in "ىا":
        return s
    return nfc(s + FATHA)


def noun_surface(noun, case):
    return nfc(noun + case)


def load_lexicon():
    raw = open(FROZEN, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == FROZEN_SHA, "frozen lexicon sha mismatch"
    fz = json.loads(raw.decode("utf-8"))
    roots = []   # ordered list of dicts
    for alt_key, alt, dform in (("I-VII", "ANTI", "VII"), ("I-IV", "CAUS", "IV")):
        for x in sorted(fz["pairs"][alt_key], key=lambda r: r["pair"]):
            roots.append({"pair": x["pair"], "alt": alt, "derived": dform,
                          "lex": {"I": nfc(x["I"]), dform: nfc(x[dform])},
                          "cls": "HUMAN" if x["pair"] in HUMAN_ROOTS else "INAN",
                          "state": STATE_SURFACE[x["pair"]]})
    assert set(r["pair"] for r in roots) == set(STATE_SURFACE)
    return roots


def verb_table(roots):
    """vocalised verb surface -> (pair, form); must be injective."""
    tab = {}
    for r in roots:
        for f, lex in r["lex"].items():
            s = verb_surface(lex)
            assert s not in tab, ("verb surface collision", s)
            tab[s] = (r["pair"], f)
    return tab


def render(root, form, n1, n2):
    """Fixed VSO; intransitives carry the PP adjunct (doc 87 §2)."""
    v = verb_surface(root["lex"][form])
    if TRANSITIVE[(root["alt"], form)]:
        return [v, noun_surface(n1, NOM), noun_surface(n2, ACC)]
    return [v, noun_surface(n1, NOM), PREP, noun_surface(n2, GEN)]


def strip_marks(tok):
    return "".join(c for c in nfc(tok) if c not in HARAKAT)


def compile_surface(tokens, vtab, noun_index):
    """S-T compiler: returns (pair, form, [noun_id_wire0, noun_id_wire1]). Reads surface only."""
    assert tokens[0] in vtab, ("unknown verb surface", tokens[0])
    pair, form = vtab[tokens[0]]
    nouns = [noun_index[strip_marks(t)] for t in tokens[1:] if t != PREP]
    assert len(nouns) == 2
    return pair, form, nouns


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def provenance():
    mods = {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f or not f.endswith(".py"):
            continue
        fa = os.path.abspath(f)
        if os.path.dirname(fa) == HERE and os.path.exists(fa):   # repo modules only
            mods[os.path.basename(fa)] = sha_file(fa)
    return {"modules_sha256": dict(sorted(mods.items())), "frozen_lexicon_sha256": FROZEN_SHA}


def write_new_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    assert not os.path.exists(path), "Rule 3: refuse to overwrite %s" % path
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1)
    obj2 = dict(obj)
    obj2["payload_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj2, f, ensure_ascii=False, sort_keys=True, indent=1)
