# -*- coding: utf-8 -*-
"""exp94_common.py -- doc 89 (REGISTERED 7fb995a): paradigm lexicon, surfaces, the ONE segmentation function
(shared by the circuit compiler and the baselines -- doc 89 §5 information parity), IO.

Classes: (gender, number) in {m,f} x {sg,du,pl}; held-out cell f-du.
Noun morphemes (nominative only; doc 89 §2): gender {G0, T}, number {N0, AN, UN*, AT*} (* fused: carry gender).
Verb morphemes: gender {G0, T}, number {N0, A, UW*, NA*}.  ة and the ت of ـتان are ONE id (T); تْ and the ت of ـتا
are ONE verb id (T).
"""
import os, json, hashlib, unicodedata, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARAKAT = "ًٌٍَُِّْٰ"
CLASSES = [("m", "sg"), ("f", "sg"), ("m", "du"), ("f", "du"), ("m", "pl"), ("f", "pl")]
HELD = ("f", "du")
STEMS = ["المعلم", "المهندس", "الموظف", "اللاعب", "السائق", "المسافر", "المدرب", "الممرض",
         "المصور", "المترجم", "المبرمج", "المراسل", "المحاسب", "المشرف", "المفتش", "المنظم"]
OBJECTS = ["الكتاب", "الباب", "الصندوق", "البيت", "الجدار", "الملف", "التقرير", "الخبز"]
VERBS = ["كتب", "فتح", "حمل", "رفع", "كسر", "رسم"]
NOUN_END = {("m", "sg"): "ُ", ("f", "sg"): "ةُ", ("m", "du"): "انِ", ("f", "du"): "تانِ",
            ("m", "pl"): "ونَ", ("f", "pl"): "اتُ"}
VERB_END = {("m", "sg"): "َ", ("f", "sg"): "َتْ", ("m", "du"): "َا", ("f", "du"): "َتَا",
            ("m", "pl"): "ُوا", ("f", "pl"): "ْنَ"}
INTRO, THEN, Q, AND = ["في", "الغرفةِ"], "ثمّ", "هل", "و"

NOUN_G = ["G0", "T"]; NOUN_N = ["N0", "AN", "UN", "AT"]
VERB_G = ["G0", "T"]; VERB_N = ["N0", "A", "UW", "NA"]
FUSED = {"UN": "m", "AT": "f", "UW": "m", "NA": "f"}


def nfc(s):
    return unicodedata.normalize("NFC", s)


def strip_marks(tok):
    return "".join(c for c in nfc(tok) if c not in HARAKAT)


def noun_surface(stem, cls, conj=False):
    return nfc((AND if conj else "") + stem + NOUN_END[cls])


def verb_surface(v, cls):
    a, b, c = v
    return nfc(a + "َ" + b + "َ" + c + VERB_END[cls])


def segment_noun(tok):
    """-> (stem, gender morpheme, number morpheme). The SAME function feeds circuit and baselines."""
    s = strip_marks(tok)
    if s.startswith(AND + "ال"):
        s = s[1:]
    for end, g, n in (("تان", "T", "AN"), ("ان", "G0", "AN"), ("ون", "G0", "UN"), ("ات", "G0", "AT"),
                      ("ة", "T", "N0")):
        if s.endswith(end) and s[:-len(end)] in STEMS:
            return s[:-len(end)], g, n
    if s in STEMS or s in OBJECTS:
        return s, "G0", "N0"
    raise ValueError("noun segmentation failed: %r" % tok)


def segment_verb(tok):
    s = strip_marks(tok)
    for end, g, n in (("تا", "T", "A"), ("وا", "G0", "UW"), ("ت", "T", "N0"), ("ا", "G0", "A"), ("ن", "G0", "NA")):
        if s.endswith(end) and s[:-len(end)] in VERBS:
            return s[:-len(end)], g, n
    if s in VERBS:
        return s, "G0", "N0"
    raise ValueError("verb segmentation failed: %r" % tok)


def features(g, n):
    """morpheme ids -> (gender, number) class the morphemes denote (used only by gates/validation)."""
    gender = FUSED.get(n, "f" if g == "T" else "m")
    number = {"N0": "sg", "AN": "du", "A": "du", "UN": "pl", "AT": "pl", "UW": "pl", "NA": "pl"}[n]
    return gender, number


def compile_story(st):
    """S-T compiler (doc 89 §4). Surface only. Returns candidates [(stem, g, n)], verb (stem, g, n), object, x index."""
    intro, ev, q = st["intro"], st["event"], st["q"]
    assert intro[:2] == INTRO
    cands = [segment_noun(t) for t in intro[2:]]
    assert ev[0] == THEN
    verb = segment_verb(ev[1])
    obj = segment_noun(ev[2])[0]
    assert q[0] == Q
    qverb = segment_verb(q[1])[0]                   # question verb: STEM only; its suffix is NOT compiled
    x_stem = segment_noun(q[2])[0]
    stems = [c[0] for c in cands]
    return {"cands": cands, "verb": verb, "obj": obj, "qverb": qverb, "x": stems.index(x_stem)}


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def provenance():
    mods = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f or not f.endswith(".py"):
            continue
        fa = os.path.abspath(f)
        if os.path.dirname(fa) == HERE and os.path.exists(fa):
            mods[os.path.basename(fa)] = sha_file(fa)
    return {"modules_sha256": dict(sorted(mods.items()))}


def write_new_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    assert not os.path.exists(path), "Rule 3: refuse to overwrite %s" % path
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1)
    obj2 = dict(obj)
    obj2["payload_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj2, f, ensure_ascii=False, sort_keys=True)
