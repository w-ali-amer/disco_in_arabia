# -*- coding: utf-8 -*-
"""exp90_encoder_tests.py -- doc 83 SS4.5 / G11 unit tests for the vowel extractor.

run_suite(extract) runs every test against an extractor function and returns
a list of {name, passed, detail}. The gate (exp90_gates.py encoder) requires
  * the real extractor (exp90_encoder.extract_case) passes EVERY test, and
  * the designed-to-fail naive last-codepoint extractor FAILS (it must fail
    the nisba tests), proving the suite can detect the NFC shadda trap.
Run directly:  python3 exp90_encoder_tests.py
"""
import sys
import unicodedata

import exp41_story_generator as gen
import exp90_encoder as enc

NISBA_EXPECTED = {"الجندي", "الشرطي", "الصحفي"}
CASES = (enc.NOM, enc.ACC, enc.NONE)


def _forms(noun, case):
    """Every surface form the extractor must read: NFC and NFD, and for nisba
    nouns both typed orders of shadda + vowel."""
    out = []
    nisba = noun.endswith("ي")
    for sf in ((True, False) if nisba else (True,)):
        if case == enc.NONE:
            tail = ""
        elif not nisba:
            tail = enc.VOWEL_OF[case]
        elif sf:
            tail = enc.SHADDA + enc.VOWEL_OF[case]
        else:
            tail = enc.VOWEL_OF[case] + enc.SHADDA
        raw = noun + tail
        out += [("raw_shadda_first=%s" % sf, raw),
                ("NFC_shadda_first=%s" % sf, unicodedata.normalize("NFC", raw)),
                ("NFD_shadda_first=%s" % sf, unicodedata.normalize("NFD", raw))]
    return out


def run_suite(extract):
    res = []

    def rec(name, ok, detail=""):
        res.append({"name": name, "passed": bool(ok), "detail": detail})

    nouns = sorted(gen.HUMANS)
    rec("inventory_is_24_nouns", len(nouns) == 24, "n=%d" % len(nouns))
    nis = {n for n in nouns if n.endswith("ي")}
    rec("nisba_set_exact", nis == NISBA_EXPECTED, "found=%s" % sorted(nis))

    # 24 nouns x {NOM, ACC, NONE}, via the generator's own vocaliser
    bad = []
    for n in nouns:
        for c in CASES:
            s = enc.vocalise_noun(n, c)
            try:
                got = extract(s)
            except Exception as e:                        # noqa: BLE001
                got = "RAISE:%s" % e
            if got != c:
                bad.append((n, c, got))
    rec("all_24x3_vocalise_roundtrip", not bad,
        "%d/72 wrong; first: %s" % (len(bad), bad[:3]))

    # nisba: both shadda orders, raw / NFC / NFD
    bad = []
    for n in sorted(nis):
        for c in CASES:
            for tag, s in _forms(n, c):
                try:
                    got = extract(s)
                except Exception as e:                    # noqa: BLE001
                    got = "RAISE:%s" % e
                if got != c:
                    bad.append((n, c, tag, got))
    rec("nisba_shadda_orders_nfc_nfd", not bad,
        "%d wrong; first: %s" % (len(bad), bad[:3]))

    # NFD of every plain noun too
    bad = []
    for n in nouns:
        for c in CASES:
            s = unicodedata.normalize("NFD", enc.vocalise_noun(n, c))
            try:
                got = extract(s)
            except Exception as e:                        # noqa: BLE001
                got = "RAISE:%s" % e
            if got != c:
                bad.append((n, c, got))
    rec("all_nouns_nfd", not bad, "%d wrong" % len(bad))

    # injectivity: (noun, case) -> surface is injective and inverted exactly
    surf = {}
    inv_bad = []
    for n in nouns:
        for c in CASES:
            s = enc.vocalise_noun(n, c)
            surf.setdefault(s, []).append((n, c))
            try:
                back = (enc.strip_marks(s), extract(s))
            except Exception as e:                        # noqa: BLE001
                back = ("RAISE", str(e))
            if back != (n, c):
                inv_bad.append((n, c, back))
    coll = [k for k, v in surf.items() if len(v) > 1]
    rec("injective_72_distinct_surfaces", len(surf) == 72 and not coll,
        "distinct=%d collisions=%d" % (len(surf), len(coll)))
    rec("inverse_exact", not inv_bad, "%d wrong" % len(inv_bad))

    # verbs: final fatha, base recoverable
    import json
    verbs = sorted(json.load(open("stories_exp51_VT.json", encoding="utf-8"))
                   ["meta"]["verb_inventory"].keys())
    vb = [v for v in verbs if enc.strip_marks(enc.vocalise_verb(v)) != v
          or not enc.vocalise_verb(v).endswith(enc.FATHA)]
    rec("verbs_final_fatha_roundtrip", len(verbs) == 6 and not vb,
        "verbs=%s bad=%s" % (verbs, vb))

    # out-of-domain marks must raise (never silently map)
    ood = {"kasra": "الطالبِ", "tanwin_damm": "الطالبٌ",
           "two_vowels": "الطالبَُ", "sukun": "الطالبْ"}
    silent = []
    for k, s in ood.items():
        try:
            got = extract(s)
            silent.append((k, got))
        except enc.EncoderError:
            pass
        except Exception as e:                            # noqa: BLE001
            silent.append((k, "wrong-exception:%s" % type(e).__name__))
    rec("out_of_domain_raises", not silent, "silent=%s" % silent)
    return res


def gate():
    real = run_suite(enc.extract_case)
    naive = run_suite(enc.naive_last_char_case)
    real_ok = all(r["passed"] for r in real)
    naive_fails = [r["name"] for r in naive if not r["passed"]]
    dtf_ok = ("nisba_shadda_orders_nfc_nfd" in naive_fails
              or "all_24x3_vocalise_roundtrip" in naive_fails)
    return {"real_extractor": real, "real_all_pass": real_ok,
            "designed_to_fail_extractor": "exp90_encoder.naive_last_char_case",
            "designed_to_fail_results": naive,
            "designed_to_fail_failed_tests": naive_fails,
            "designed_to_fail_shown_failing": bool(dtf_ok),
            "passed": bool(real_ok and dtf_ok)}


if __name__ == "__main__":
    g = gate()
    for r in g["real_extractor"]:
        print("[real ] %-34s %s  %s" % (r["name"], "PASS" if r["passed"] else "FAIL", r["detail"]))
    for r in g["designed_to_fail_results"]:
        print("[naive] %-34s %s  %s" % (r["name"], "PASS" if r["passed"] else "FAIL", r["detail"]))
    print("GATE", "PASS" if g["passed"] else "FAIL")
    sys.exit(0 if g["passed"] else 1)
