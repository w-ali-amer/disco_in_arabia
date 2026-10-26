#!/usr/bin/env python3
"""wazn_b1_build_lexicon.py -- doc 86 B1: candidate lexicon for wazn -> valency.

A CANDIDATE SUPERSET WITH FLAGS. It is NOT a frozen lexicon, and it never
emits "CLEAN". Freezing is a separate decision (reviewer, then user).

Sources:
  (1) CAMeL Tools morphology DB (calima-msa-r13). Every perfect, active-voice
      verb stem whose diac == lex (the citation stem) is enumerated from
      db.stem_hash. We keep root, pattern, gloss and lex_logprob (a lexeme
      log-probability from the ATB; -99 = unseen). The verb FORM (I..X, Q) is
      derived from the pattern string by form_of(); CAMeL has no form feature.
  (2) Arabic VerbNet XML (Mousser 2013; github JaouadMousser/Arabic-Verbnet),
      cloned at ~/ext/Arabic-Verbnet, commit recorded.
      - AVN members are joined to CAMeL lexemes by analysing the member through
        the SAME stem table (one form classifier, one join key).
      - Class-level transitivity comes from the FRAME primary descriptions,
        with subclass frames inheriting their ancestors' frames.
      - SIBLINGCLASSES links (e.g. kasara-1 <-> Iinokasara-1) are recorded as
        pair corroboration.

Run on WSL:  PYTHONPATH=~/camel_tools ~/venv/bin/python3 wazn_b1_build_lexicon.py
(run from the repo dir, NOT ~ -- ~/camel_tools shadows the package from ~).
Outputs: wazn_b1/candidates.json, wazn_b1/candidates.md, wazn_b1/form_classifier_check.json
"""
import os, sys, json, re, glob, hashlib, subprocess, unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wazn_b1")
AVN_DIR = os.path.expanduser("~/ext/Arabic-Verbnet")

HARAKAT = "ًٌٍَُِْٰ"   # tanwin, short vowels, sukun, dagger alif
SHADDA = "ّ"


def strip_h(s):
    """Remove short vowels / sukun (keep shadda and letters)."""
    return "".join(c for c in s if c not in HARAKAT)


def strip_all(s):
    return "".join(c for c in strip_h(s) if c != SHADDA)


def norm_letters(s):
    """For joining: no diacritics, alif variants unified, final vowel ignored."""
    s = strip_all(s)
    for a in "ٱإآأ":                     # CAMeL surface keys are hamza-normalised
        s = s.replace(a, "ا")
    return s


def nfc(s):
    """Canonical mark order (shadda/fatha order differs between sources)."""
    return unicodedata.normalize("NFC", s)


# ----------------------------------------------------------------- form classifier
def form_of(pattern):
    """Verb form from a CAMeL stem pattern (digits = radicals; weak radicals may
    appear as letters). Returns 'I'..'X', 'Q' (quadriliteral) or '?'."""
    p = strip_h(pattern).replace("ٱ", "ا")          # wasla -> alif
    if "4" in p:
        return "Q"
    if p.startswith("است"):                     # است
        return "X"
    if p.startswith("ان") and len(p) > 2 and (p[2].isdigit() or p[2] in "اوي"):
        return "VII"                                          # ان + R1
    if p.startswith("ا"):                                # ٱِ...
        body = p[1:]
        # VIII: R1 + infix t (or assimilated t/T/d), or assimilated ٱِتّ
        if re.match(r"^(\d|[^\d])[تطد]", body) and not body.startswith("ن"):
            return "VIII"
        if body.startswith("ت" + SHADDA):
            return "VIII"
        if re.match(r"^1\d" + SHADDA + "$", body) or re.match(r"^12\d?" + SHADDA, body):
            return "IX"
        return "?"
    if p.startswith("أ"):                                # أ + R1 -> IV
        # IV needs R1 after the prefix (digit 1, or a weak R1 as و/ي + digit).
        # أ directly followed by R2 means أ IS R1 (hamza-initial Form I; CAMeL root #.x.y).
        if len(p) > 1 and (p[1] == "1" or (p[1] in "وي" and len(p) > 2 and p[2].isdigit())):
            return "IV"
        p = "1" + p[1:]                                     # hamza is R1: classify the rest as unprefixed
    elif p.startswith("ت"):                              # ت prefix
        if SHADDA in p:
            return "V"
        if re.match(r"^ت(\d|[^\d])ا", p):
            return "VI"
        return "?"
    # unprefixed stems: I, II, III
    # shadda on the 2nd skeleton slot (R2 as digit OR weak letter و/ي):
    # followed by more -> II (1َ2َّ3, 1َيَّ3, 1َ2َّى); at the end -> geminate Form I (1َ2ّ)
    if len(p) >= 3 and p[2] == SHADDA:
        return "II" if len(p) > 3 else "I"
    if re.match(r"^(\d|[^\d])ا\d", p) and "2" in p:      # R1 + ا + R2 -> III
        return "III"
    return "I"


# gold set for the classifier: lex -> expected form (reviewer-rated pairs + control verbs)
GOLD = {
    "كَسَر": "I", "ٱِنْكَسَر": "VII", "قَطَع": "I", "ٱِنْقَطَع": "VII", "فَتَح": "I", "ٱِنْفَتَح": "VII",
    "حَلّ": "I", "ٱِنْحَلّ": "VII", "فَكّ": "I", "ذاب": "I", "أَذاب": "IV", "فَرِح": "I", "أَفْرَح": "IV",
    "فَرَّح": "II", "كَسَّر": "II", "يَقِظ": "I", "يَقَّظ": "II", "قاتَل": "III", "تَقاتَل": "VI",
    "تَكَسَّر": "V", "ٱِتَّصَل": "VIII", "ٱِخْتار": "VIII", "ٱِزْدَهَر": "VIII", "ٱِضْطَرّ": "VIII",
    "ٱِتَّخَذ": "VIII", "ٱِسْتَقام": "X", "ٱِحْمَرّ": "IX", "دَحْرَج": "Q", "ٱِنْقاد": "VII",
    "قال": "I", "رَضِي": "I", "رَضَّى": "II", "نَسِي": "I", "طَفِئ": "I", "أَنْحَل": "IV",
    "غَلَق": "I", "كَتَم": "I", "بَتَل": "I", "ٱِنْبَتَل": "VII", "قاتَل": "III",
    "ضَيَّف": "II", "عَوَّد": "II", "سَيَّل": "II", "أَرِب": "I", "أَعاد": "IV", "أَثار": "IV", "أَقْلَق": "IV",
}


# ----------------------------------------------------------------- CAMeL enumeration
def load_camel():
    from camel_tools.morphology.database import MorphologyDB
    import camel_tools
    db = MorphologyDB.builtin_db()
    lex = {}          # lex -> record (citation stem)
    by_surface = defaultdict(set)   # undiacritized normalized surface -> set(lex)
    for surf, entries in db.stem_hash.items():
        for cat, f in entries:
            if f.get("pos") != "verb" or f.get("asp") != "p" or f.get("vox") != "a":
                continue
            L = nfc(f["lex"])
            by_surface[norm_letters(surf)].add(L)
            rec = {"lex": L, "root": f["root"], "pattern": f["pattern"], "diac": nfc(f["diac"]),
                   "gloss": f.get("gloss", ""), "lex_logprob": float(f.get("lex_logprob", -99))}
            cur = lex.get(L)
            # prefer the citation stem (diac == lex); keep all glosses for the lexeme
            if cur is None or (nfc(f["diac"]) == L and cur["diac"] != L):
                if cur is not None:
                    rec["glosses"] = cur.get("glosses", [cur["gloss"]])
                lex[L] = rec
            glist = lex[L].setdefault("glosses", [lex[L]["gloss"]])
            if rec["gloss"] not in glist:
                glist.append(rec["gloss"])
    for L, r in lex.items():
        r["form"] = form_of(r["pattern"])
        r["glosses"] = sorted(set(r["glosses"]))
    meta = {"camel_tools_version": camel_tools.__version__, "db": "calima-msa-r13",
            "db_version": json.load(open(os.path.expanduser("~/.camel_tools/versions.json")))
                          .get("morphology-db-msa-r13")}
    return lex, by_surface, meta


# ----------------------------------------------------------------- AVN
def frame_valency(primary, secondary):
    """'trans' / 'intrans' / None from an AVN frame description."""
    p = (primary or "").strip()
    toks = p.split()
    if toks and toks[0] == "V":
        nps = 0
        for t in toks[1:]:
            if t.startswith("NP"):
                nps += 1
            elif t.startswith("PP") or t.startswith("S") or t.startswith("Oa"):
                continue
        return "trans" if nps >= 2 else ("intrans" if nps == 1 else None)
    if re.match(r"^NP(\.\w+)? V NP", p):
        return "trans"
    if re.match(r"^NP(\.\w+)? V$", p):
        return "intrans"
    txt = (p + " " + (secondary or "")).lower()
    if "intransitive" in txt or "inchoative" in txt:
        return "intrans"
    if "transitive" in txt:
        return "trans"
    return None


def load_avn():
    commit = subprocess.check_output(["git", "-C", AVN_DIR, "rev-parse", "HEAD"], text=True).strip()
    classes = {}   # class id -> {"members":[(name, transname, root)], "val": set, "siblings": set}

    def walk(node, cid, inherited_val, siblings):
        frames = node.find("FRAMES")
        val = set(inherited_val)
        if frames is not None:
            for fr in frames.findall("FRAME"):
                d = fr.find("DESCRIPTION")
                if d is not None:
                    v = frame_valency(d.get("primary"), d.get("secondary"))
                    if v:
                        val.add(v)
        mem = []
        ms = node.find("MEMBERS")
        if ms is not None:
            for m in ms.findall("MEMBER"):
                r = m.find("Root")
                mem.append((m.get("name"), m.get("transname"), r.get("name") if r is not None else None))
        classes[cid] = {"members": mem, "val": val, "siblings": set(siblings)}
        subs = node.find("SUBCLASSES")
        if subs is not None:
            for sc in subs.findall("AVNSUBCLASS"):
                walk(sc, sc.get("ID"), val, siblings)

    for fp in sorted(glob.glob(os.path.join(AVN_DIR, "*.xml"))):
        root = ET.parse(fp).getroot()
        sib = set()
        for tag in ("SIBLINGCLASSES",):
            s = root.find(tag)
            if s is not None:
                for c in s:
                    if c.get("ID"):
                        sib.add(c.get("ID"))
        walk(root, root.get("ID"), set(), sib)
    return classes, commit


def camel_root_matches(camel_root, avn_root):
    if not avn_root:
        return True
    cr = camel_root.split(".")
    ar = list(norm_letters(avn_root))
    if len(cr) != len(ar):
        return False
    WEAK = set("وياىءأئؤإآ")
    return all((c == "#" and a in WEAK) or norm_letters(c) == a or (c in "أءئؤ" and a in "أءئؤا")
               for c, a in zip(cr, ar))


def norm_join(s):
    """Join key that KEEPS non-initial hamza (أطفأ 'extinguish' != أطفا 'make float'):
    no diacritics, initial alif variants unified, ى -> ي."""
    s = strip_all(nfc(s))
    if s and s[0] in "ٱإآأ":
        s = "ا" + s[1:]
    return s.replace("ى", "ي")


def join_avn(classes, lex, by_surface):
    """AVN member -> CAMeL lex. Returns lex -> list of class ids, plus a join report."""
    lex2cls = defaultdict(set)
    lex_index = defaultdict(set)
    for L in lex:
        lex_index[norm_letters(L).replace("ى", "ي")].add(L)
    report = {"members": 0, "joined_exact": 0, "joined_root_form": 0, "unjoined": []}
    for cid, c in classes.items():
        for name, transname, aroot in c["members"]:
            report["members"] += 1
            key = norm_letters(transname or name or "")
            cands = [L for L in by_surface.get(key, ()) if camel_root_matches(lex[L]["root"], aroot)]
            if not cands:   # final-weak stems are split in the stem table: fall back to the citation form
                cands = [L for L in lex_index.get(key.replace("ى", "ي"), ())
                         if camel_root_matches(lex[L]["root"], aroot)]
            want = norm_letters(name or "")
            # exact: diacritized AVN name (final vowel dropped) == CAMeL lex, modulo wasla/hamza
            nm = (norm_join(name or ""), SHADDA in (name or ""))
            exact = [L for L in cands if (norm_join(L), SHADDA in L) == nm]
            if exact:
                picked, how = exact, "joined_exact"
            elif cands and all("#" not in lex[L]["root"] for L in cands):   # no fuzzy join on weak roots
                # same letters: disambiguate by shadda agreement (I vs II)
                sh = SHADDA in (name or "")
                picked = [L for L in cands if (SHADDA in L) == sh] or cands
                how = "joined_root_form"
            elif cands:
                report["unjoined"].append({"class": cid, "name": name, "root": aroot, "why": "weak_root_no_exact"})
                continue
            else:
                report["unjoined"].append({"class": cid, "name": name, "root": aroot})
                continue
            report[how] += 1
            for L in picked:
                lex2cls[L].add(cid)
    report["unjoined_n"] = len(report["unjoined"])
    return lex2cls, report


# ----------------------------------------------------------------- flags / pairing
VII_BAD_R1 = set("رلويءأنؤئ")      # Form VII phonotactic restriction (confirm in Wright/Ryding)

# doc 86 model ratings, keyed by the LEXEME PAIR (I letters, derived letters; norm_letters), never by root:
# CAMeL '#' collapses distinct weak/hamza roots (reviewer, 2026-09-26). B2' = rated pairs; B2'-ext = the
# reviewer's PROVISIONAL rating of B1's new candidates, re-rated where the CAMeL row gloss disagreed
# (row wins). Model judgement, NOT native-speaker validation, NOT lexicon corroboration.
# RECORD OF RATINGS = doc 86 §5.1; this dict mirrors it -- any user override must land in BOTH.
_R = {
    "B2'": {
        "VII": {"clean": "كسر/انكسر قطع/انقطع فصل/انفصل عزل/انعزل حل/انحل فك/انفك خلع/انخلع سكب/انسكب فلق/انفلق حجب/انحجب",
                "pending_lexicon": "فتح/انفتح كشف/انكشف",
                "excluded": "كتم/انكتم بتل/انبتل غلق/انغلق"},
        "IV": {"clean": "ذاب/أذاب فرح/أفرح شحب/أشحب تلف/أتلف غرق/أغرق فسد/أفسد ضاع/أضاع طفئ/أطفأ",
               "conditional": "خرج/أخرج", "downgraded": "يقظ/أيقظ"},
    },
    "B2'-ext(provisional)": {
        "VII": {"likely": "عكس/انعكس هزم/انهزم شق/انشق شغل/انشغل",
                "likely_homonymous": "قلب/انقلب",   # reviewer flagged gloss_mismatch on the FIRST gloss (overthrow);
                                                     # the merged row also carries 'turn_around;reverse' -> row wins;
                                                     # items must use the physical sense
                "likely_register_flag": "خفض/انخفض", "likely_pending_hash_check": "ثنى/انثنى",
                "excluded": "سحب/انسحب دفع/اندفع زرع/انزرع بعث/انبعث طوى/انطوى قض/انقض جلا/انجلى"},
        "IV": {"likely": "غضب/أغضب خجل/أخجل تعب/أتعب قلق/أقلق نبت/أنبت ظهر/أظهر برز/أبرز خاف/أخاف زال/أزال",
               "gloss_mismatch": "عجب/أعجب",     # row gloss 'admire' = passive أُعجب بـ usage; active CAUS unattested
               "register_flag": "مات/أمات",      # literary/religious; قتل is the everyday causative
               "semi": "بعد/أبعد كثر/أكثر صلح/أصلح",   # drifted glosses (remove; increase/do_frequently; repair)
               "excluded": "قام/أقام دار/أدار جاز/أجاز حل/أحل قر/أقر باح/أباح طل/أطل شاع/أشاع طلع/أطلع سار/أسار فسح/أفسح",
               "excluded_conflation": "سأل/أسال طفا/أطفأ",
               "excluded_motion_posture": "دخل/أدخل نزل/أنزل وقف/أوقف قعد/أقعد"},
    },
}
B2P = {}
for _src, _by_form in _R.items():
    for _form, _by_rating in _by_form.items():
        for _rating, _pairs in _by_rating.items():
            for _pair in _pairs.split():
                _a, _b = _pair.split("/")
                B2P[(_form, norm_letters(_a), norm_letters(_b))] = {"rating": _rating, "source": _src, "pair": _pair}


def b2p_rating(form, lex_i, lex_d):
    return B2P.get((form, norm_letters(lex_i), norm_letters(lex_d)))


def valency_contrast(li, ld, derived, classes, lex2cls):
    """INFORMATION, not a gate: some sibling-linked class pair where the I-class has ONLY the expected
    valency and the derived class ONLY its expected one (many AVN classes are both-both)."""
    base = lambda c: c.split(".")[0]
    want_i, want_d = ("trans", "intrans") if derived == "VII" else ("intrans", "trans")
    for ci in lex2cls.get(li, ()):
        for cd in lex2cls.get(ld, ()):
            linked = base(cd) in {base(s) for s in classes[ci]["siblings"]} or \
                     base(ci) in {base(s) for s in classes[cd]["siblings"]}
            if linked and classes[ci]["val"] == {want_i} and classes[cd]["val"] == {want_d}:
                return True
    return False


def derived_form_sibling(li, ld, derived, classes, lex2cls):
    """Strong corroboration: some AVN class of the Form I lexeme and some class of the derived lexeme
    are SIBLINGS (either direction) AND their frame valency runs the expected way:
    VII (ANTI): I-class has a transitive frame, VII-class an intransitive one;
    IV (CAUS): I-class intransitive, IV-class transitive.
    (AVN encodes most caus/inch pairs as other_cos-1 <-> other_cos-2, not by Iino*/Oa* names.)"""
    base = lambda c: c.split(".")[0]
    want_i, want_d = ("trans", "intrans") if derived == "VII" else ("intrans", "trans")
    for ci in lex2cls.get(li, ()):
        for cd in lex2cls.get(ld, ()):
            linked = base(cd) in {base(s) for s in classes[ci]["siblings"]} or \
                     base(ci) in {base(s) for s in classes[cd]["siblings"]}
            if linked and want_i in classes[ci]["val"] and want_d in classes[cd]["val"]:
                return True
    return False


def main():
    os.makedirs(OUT, exist_ok=True)
    lex, by_surface, cmeta = load_camel()

    # classifier check against the gold set
    check = {"rows": [], "errors": []}
    for L, want in sorted((nfc(k), v) for k, v in GOLD.items()):
        if L not in lex:
            check["rows"].append({"lex": L, "want": want, "got": None, "note": "lex not in DB"})
            continue
        got = lex[L]["form"]
        check["rows"].append({"lex": L, "pattern": lex[L]["pattern"], "want": want, "got": got})
        if got != want:
            check["errors"].append(L)
    fdist = defaultdict(int)
    for r in lex.values():
        fdist[r["form"]] += 1
    check["form_distribution"] = dict(sorted(fdist.items()))
    check["unclassified"] = sorted({(r["pattern"], r["lex"], r["gloss"]) for r in lex.values() if r["form"] == "?"})
    json.dump(check, open(os.path.join(OUT, "form_classifier_check.json"), "w"), ensure_ascii=False, indent=1)
    print("[classifier] gold %d, errors %d: %s" % (len(GOLD), len(check["errors"]), check["errors"]))
    print("[classifier] form distribution:", dict(fdist))
    if check["errors"]:
        print("REFUSING to build pairs: classifier disagrees with the gold set", file=sys.stderr)
        sys.exit(2)

    classes, avn_commit = load_avn()
    lex2cls, jrep = join_avn(classes, lex, by_surface)
    print("[avn] classes %d members %d exact %d root_form %d unjoined %d"
          % (len(classes), jrep["members"], jrep["joined_exact"], jrep["joined_root_form"], jrep["unjoined_n"]))

    def avn_info(L):
        cids = sorted(lex2cls.get(L, ()))
        val = set()
        sib = set()
        for c in cids:
            val |= classes[c]["val"]
            sib |= classes[c]["siblings"]
        return {"classes": cids, "valency": sorted(val), "siblings": sorted(sib)}

    roots = defaultdict(lambda: defaultdict(list))
    for L, r in lex.items():
        roots[r["root"]][r["form"]].append(L)

    rows = []
    for rt, forms in roots.items():
        for derived in ("VII", "IV"):
            if "I" not in forms or derived not in forms:
                continue
            for Li in sorted(forms["I"]):
                for Ld in sorted(forms[derived]):
                    ai, ad = avn_info(Li), avn_info(Ld)
                    _b = lambda s: {c.split(".")[0] for c in s}
                    sib_link = bool(_b(ai["classes"]) & _b(ad["siblings"]) or _b(ad["classes"]) & _b(ai["siblings"]))
                    r1 = rt.split(".")[0]
                    flags = []
                    if derived == "VII":
                        if r1 in VII_BAD_R1:
                            flags.append("VII_phonotactic_R1")
                        if ai["valency"] and "trans" not in ai["valency"]:
                            flags.append("I_not_transitive_in_AVN")
                    else:
                        if "trans" in ai["valency"] and "intrans" in ai["valency"]:
                            flags.append("I_labile_in_AVN")
                        elif ai["valency"] == ["trans"]:
                            flags.append("I_transitive_in_AVN")
                        if not any(g.startswith(("be_", "become")) for g in lex[Li]["glosses"]) \
                                and not ai["valency"]:
                            flags.append("I_gloss_not_stative")
                    if "II" in forms:
                        flags.append("form_II_exists")
                    stat = lambda L: all(g.split(";")[0].startswith(("be_", "become", "get_", "subside"))
                                         for g in lex[L]["glosses"])
                    I_kinds = {stat(x) for x in forms["I"]}
                    if len(I_kinds) == 2:   # the root has a stative/intransitive Form I AND a non-stative one
                        flags.append("I_labile_in_CAMeL")
                    if len(lex[Li]["glosses"]) > 1:   # one lex key merged several DB entries (e.g. حلّ)
                        flags.append("I_homonymous")
                    if "#" in rt:   # CAMeL '#' = any weak/hamza radical: distinct roots can collide
                        flags.append("hash_root_conflation_risk")        # (e.g. طفا 'float' vs أطفأ 'extinguish')
                    if not ai["classes"] and not ad["classes"]:
                        flags.append("no_AVN_evidence")
                    rows.append({
                        "alternation": "I-" + derived, "root": rt,
                        "I": {"lex": Li, "gloss": lex[Li]["glosses"], "logprob": lex[Li]["lex_logprob"], "avn": ai},
                        derived: {"lex": Ld, "gloss": lex[Ld]["glosses"], "logprob": lex[Ld]["lex_logprob"], "avn": ad},
                        "form_II": [{"lex": x, "gloss": lex[x]["glosses"]} for x in sorted(forms.get("II", []))],
                        "avn_sibling_link": sib_link,
                        "flags": flags,
                        "avn_derived_form_sibling": derived_form_sibling(Li, Ld, derived, classes, lex2cls),
                        "avn_valency_contrast_info": valency_contrast(Li, Ld, derived, classes, lex2cls),
                        "b2prime_model_rating": b2p_rating(derived, Li, Ld),
                    })
    rows.sort(key=lambda x: (x["alternation"], not x["avn_sibling_link"],
                             -max(x["I"]["logprob"], x[x["alternation"][2:]]["logprob"]), x["root"]))

    b2_coverage = {}
    for (f, a, b), v in sorted(B2P.items()):
        hit = [x for x in rows if x["alternation"] == "I-" + f
               and norm_letters(x["I"]["lex"]) == a and norm_letters(x[f]["lex"]) == b]
        b2_coverage["%s:%s" % (f, v["pair"])] = {
            "rating": v["rating"], "source": v["source"],
            "rows": [{"I": h["I"]["lex"], "I_gloss": h["I"]["gloss"], "I_logprob": h["I"]["logprob"],
                      f: h[f]["lex"], f + "_gloss": h[f]["gloss"], f + "_logprob": h[f]["logprob"],
                      "root": h["root"], "avn_sibling_link": h["avn_sibling_link"],
                      "avn_derived_form_sibling": h["avn_derived_form_sibling"],
                      "avn_valency_contrast_info": h["avn_valency_contrast_info"], "flags": h["flags"]}
                     for h in hit]}

    header = {
        "doc": "86 B1 (candidate superset with flags; NOT frozen; no CLEAN labels)",
        "camel": cmeta, "avn_repo": "github.com/JaouadMousser/Arabic-Verbnet", "avn_commit": avn_commit,
        "builder_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "frequency_proxy": "CAMeL lex_logprob (ATB-derived; -99 = unseen in ATB). Crude; do not over-read.",
        "avn_valency_note": "class-level (frames incl. inherited), not per member; noisy -- labile/transitive flags are weak evidence",
        "avn_sibling_note": "avn_sibling_link: the I lexeme's AVN class lists the derived lexeme's class (or vice versa) in SIBLINGCLASSES -- derivational relatedness only. avn_derived_form_sibling (STRONG): a sibling-linked class pair whose frame valency runs the alternation's way",
        "join_note": "AVN member -> CAMeL lex: exact on a hamza-preserving key; fuzzy root+form join only for roots without '#'; '#' matches weak/hamza letters only",
        "avn_join": {k: v for k, v in jrep.items() if k != "unjoined"},
        "counts": {a: sum(1 for x in rows if x["alternation"] == a) for a in ("I-VII", "I-IV")},
        "counts_avn_sibling_link": {a: sum(1 for x in rows if x["alternation"] == a and x["avn_sibling_link"])
                                    for a in ("I-VII", "I-IV")},
        "counts_avn_derived_form_sibling": {a: sum(1 for x in rows if x["alternation"] == a and x["avn_derived_form_sibling"])
                                            for a in ("I-VII", "I-IV")},
        "b2prime_coverage": b2_coverage,
    }
    json.dump({"header": header, "rows": rows, "avn_unjoined": jrep["unjoined"]},
              open(os.path.join(OUT, "candidates.json"), "w"), ensure_ascii=False, indent=1, sort_keys=True)

    with open(os.path.join(OUT, "candidates.md"), "w", encoding="utf-8") as f:
        f.write("# wazn B1 candidate lexicon (doc 86) -- superset with flags, NOT frozen\n\n")
        f.write("```\n" + json.dumps(header, ensure_ascii=False, indent=1) + "\n```\n\n")
        for alt in ("I-VII", "I-IV"):
            d = alt[2:]
            f.write("## %s\n\n| root | I | gloss I | lp I | %s | gloss %s | lp %s | AVN sib | AVN val I/%s | II? | flags | B2' |\n"
                    "|---|---|---|---|---|---|---|---|---|---|---|---|\n" % (alt, d, d, d, d))
            for x in rows:
                if x["alternation"] != alt:
                    continue
                f.write("| %s | %s | %s | %.1f | %s | %s | %.1f | %s | %s / %s | %s | %s | %s |\n" % (
                    x["root"], x["I"]["lex"], "; ".join(x["I"]["gloss"])[:40], x["I"]["logprob"],
                    x[d]["lex"], "; ".join(x[d]["gloss"])[:40], x[d]["logprob"],
                    "Y" if x["avn_sibling_link"] else "", ",".join(x["I"]["avn"]["valency"]) or "-",
                    ",".join(x[d]["avn"]["valency"]) or "-", "Y" if x["form_II"] else "",
                    " ".join(x["flags"] + (["AVN_DERIVED_FORM_SIB"] if x["avn_derived_form_sibling"] else [])),
                    ("%s:%s" % (x["b2prime_model_rating"]["source"], x["b2prime_model_rating"]["rating"]))
                    if x["b2prime_model_rating"] else ""))
            f.write("\n")
    print("[out] %s  I-VII %d (AVN-sib %d)  I-IV %d (AVN-sib %d)" % (
        OUT, header["counts"]["I-VII"], header["counts_avn_sibling_link"]["I-VII"],
        header["counts"]["I-IV"], header["counts_avn_sibling_link"]["I-IV"]))


if __name__ == "__main__":
    main()
