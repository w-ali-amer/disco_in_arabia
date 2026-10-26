# -*- coding: utf-8 -*-
"""exp93_common.py -- doc 88 (REGISTERED c1caeb4): lexicon, surfaces, S-T surface compiler, IO.

Compiler reads SURFACE tokens only. Morphemes: V2 suffix (token ends in تْ), noun gender (stripped token ends in ة).
S1 suffix and the question verb's suffix are NOT compiled (doc 88 §2, §3). No agent is ever given.
"""
import os, json, hashlib, unicodedata, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NOM, ACC = "ُ", "َ"
FEM_SUFFIX = "تْ"
HARAKAT = "ًٌٍَُِّْٰ"
TA = "ة"

STEMS = ["الطالب", "المدير", "المعلم", "الطبيب", "العامل", "المهندس", "الكاتب", "الطباخ",
         "اللاعب", "الموظف", "الصحفي", "السائق", "المسافر", "الباحث", "المريض", "الحارس"]
OBJECTS = ["الكتاب", "الباب", "الصندوق", "البيت", "الجدار", "الملف", "التقرير", "الخبز"]
V1 = {"ساعد": "ساعَدَ", "ضرب": "ضَرَبَ", "سأل": "سَأَلَ", "عالج": "عالَجَ"}
V2 = {"كتب": "كَتَبَ", "فتح": "فَتَحَ", "حمل": "حَمَلَ", "رفع": "رَفَعَ", "كسر": "كَسَرَ", "رسم": "رَسَمَ"}
THEN, Q = "ثمّ", "هل"


def nfc(s):
    return unicodedata.normalize("NFC", s)


def verb_surface(vocal, fem):
    return nfc(vocal + (FEM_SUFFIX if fem else ""))


def noun_surface(stem, fem, case):
    return nfc(stem + (TA if fem else "") + case)


def strip_marks(tok):
    return "".join(c for c in nfc(tok) if c not in HARAKAT)


def read_verb(tok, table):
    """-> (verb id, fem suffix bit). The suffix bit is returned; callers decide whether to compile it."""
    t = nfc(tok)
    fem = t.endswith(nfc(FEM_SUFFIX))
    base = strip_marks(t[:-len(nfc(FEM_SUFFIX))] if fem else t)
    ids = [k for k in table if strip_marks(table[k]) == base]
    assert len(ids) == 1, ("verb", tok)
    return ids[0], int(fem)


def read_noun(tok):
    """-> (stem, fem bit) from the surface: strip marks, then a final ة is the gender morpheme."""
    s = strip_marks(tok)
    fem = s.endswith(TA)
    stem = s[:-1] if fem else s
    assert stem in STEMS or stem in OBJECTS, ("noun", tok)
    return stem, int(fem)


def compile_story(story):
    """S-T compiler (doc 88 §3). Returns a dict of ids/bits; wire 0/1 = S1 nouns in surface order."""
    s1, s2, q = story["s1"], story["s2"], story["q"]
    v1, _s1suffix_not_compiled = read_verb(s1[0], V1)
    (a_stem, a_fem), (b_stem, b_fem) = read_noun(s1[1]), read_noun(s1[2])
    assert s2[0] == THEN
    v2, suffix_fem = read_verb(s2[1], V2)
    o_stem, o_fem = read_noun(s2[2])
    assert o_fem == 0 and o_stem in OBJECTS
    assert q[0] == Q
    vq, _qsuffix_not_compiled = read_verb(q[1], V2)
    x_stem, _ = read_noun(q[2])
    x_wire = [a_stem, b_stem].index(x_stem)          # X's STEM finds its wire
    return {"v1": v1, "v2": v2, "vq": vq, "stem0": a_stem, "stem1": b_stem, "fem0": a_fem, "fem1": b_fem,
            "obj": o_stem, "suffix_fem": suffix_fem, "x_wire": x_wire}


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
        json.dump(obj2, f, ensure_ascii=False, sort_keys=True, indent=1)
