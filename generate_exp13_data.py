# -*- coding: utf-8 -*-
"""
generate_exp13_data.py
----------------------
Generates additional matched SVO/VSO pairs and augmented Tense data
for exp13. Appends to sentences.json under new keys:
  - "WordOrderMatched": list of {sentence, label, pair_id}
      all 60 SVO and 60 VSO sentences share word triples (matched pairs)
  - "TenseBinary": list of {sentence, label}
      50 Past + 50 Present (morphologically unambiguous)

Run:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 generate_exp13_data.py
"""

import json
from pathlib import Path

# ─── Matched SVO/VSO triples ─────────────────────────────────────────────────
# Format: (subject, past_verb, present_verb, object)
# All are MSA / Gulf colloquial that AraVec covers.
# The existing 40 triples come from sentences.json['WordOrder'].
# We add 20 more for a total of 60 matched pairs.

NEW_TRIPLES = [
    # subject          past-verb    pres-verb     object
    ("الرجل",          "فتح",       "يفتح",       "الباب"),
    ("المراة",         "غسلت",      "تغسل",       "الثياب"),
    ("الولد",          "كسر",       "يكسر",       "الزجاج"),
    ("البنت",          "رسمت",      "ترسم",       "الصورة"),
    ("الاستاذ",        "صحح",       "يصحح",       "الواجب"),
    ("الطالبة",        "حلت",       "تحل",        "المسالة"),
    ("الشرطي",         "امسك",      "يمسك",       "اللص"),
    ("الطفلة",         "لبست",      "تلبس",       "الفستان"),
    ("المدير",         "وقع",       "يوقع",       "العقد"),
    ("الممرضة",        "ضمدت",      "تضمد",       "الجرح"),
    ("الصياد",         "صاد",       "يصيد",       "السمك"),
    ("الطباخ",         "طبخ",       "يطبخ",       "الطعام"),
    ("الحارس",         "فتش",       "يفتش",       "الحقيبة"),
    ("العالم",         "اكتشف",     "يكتشف",      "الحقيقة"),
    ("الراعي",         "رعى",       "يرعى",       "الغنم"),
    ("الجندي",         "حمى",       "يحمي",       "المدينة"),
    ("السائق",         "اوقف",      "يوقف",       "السيارة"),
    ("المحامي",        "دافع",      "يدافع",      "الموكل"),
    ("الكاتب",         "نشر",       "ينشر",       "الكتاب"),
    ("الطيار",         "قاد",       "يقود",       "الطائرة"),
]

# ─── Tense binary sentences (morphologically unambiguous) ────────────────────
# Format: (past_sentence, present_sentence)
# These are simple subject-[object] sentences where the verb form
# unambiguously marks tense: suffix (past) vs prefix (present).

TENSE_PAIRS = [
    # past                              present
    ("كتب الطالب الدرس",               "يكتب الطالب الدرس"),
    ("قرات البنت القصة",               "تقرا البنت القصة"),
    ("سافر الرجل الى المدينة",         "يسافر الرجل الى المدينة"),
    ("شرب الطفل الحليب",               "يشرب الطفل الحليب"),
    ("اكلت القطة السمك",               "تاكل القطة السمك"),
    ("فتح المعلم الكتاب",              "يفتح المعلم الكتاب"),
    ("رسم الولد الصورة",               "يرسم الولد الصورة"),
    ("غسلت المراة الملابس",            "تغسل المراة الملابس"),
    ("بنى المهندس البيت",              "يبني المهندس البيت"),
    ("فحص الطبيب المريض",              "يفحص الطبيب المريض"),
    ("شرح الاستاذ القاعدة",            "يشرح الاستاذ القاعدة"),
    ("لعبت البنت بالدمية",             "تلعب البنت بالدمية"),
    ("ركب الفارس الحصان",              "يركب الفارس الحصان"),
    ("صاد الصياد السمك",               "يصيد الصياد السمك"),
    ("زرع الفلاح القمح",               "يزرع الفلاح القمح"),
    ("ضرب اللاعب الكرة",               "يضرب اللاعب الكرة"),
    ("حمل العامل الصندوق",             "يحمل العامل الصندوق"),
    ("قادت المراة السيارة",            "تقود المراة السيارة"),
    ("دخل الطالب الفصل",               "يدخل الطالب الفصل"),
    ("خرجت المعلمة من الغرفة",         "تخرج المعلمة من الغرفة"),
    ("اشترى الرجل الخبز",              "يشتري الرجل الخبز"),
    ("باعت المراة الثياب",             "تبيع المراة الثياب"),
    ("حل الطالب المسالة",              "يحل الطالب المسالة"),
    ("نظفت الخادمة البيت",             "تنظف الخادمة البيت"),
    ("طبخ الطباخ الطعام",              "يطبخ الطباخ الطعام"),
    # 25 more for 50/class total
    ("كسر الولد الزجاج",               "يكسر الولد الزجاج"),
    ("فتشت الشرطية الحقيبة",          "تفتش الشرطية الحقيبة"),
    ("رفع الجندي العلم",               "يرفع الجندي العلم"),
    ("خاط الخياط الثوب",               "يخيط الخياط الثوب"),
    ("مسح الولد السبورة",              "يمسح الولد السبورة"),
    ("فتحت البنت النافذة",             "تفتح البنت النافذة"),
    ("اغلق الحارس الباب",              "يغلق الحارس الباب"),
    ("حمت الامهات الاطفال",            "تحمي الامهات الاطفال"),
    ("اوقف السائق السيارة",            "يوقف السائق السيارة"),
    ("نشر الكاتب المقال",              "ينشر الكاتب المقال"),
    ("حضر المدير الاجتماع",            "يحضر المدير الاجتماع"),
    ("غادر الطيار المطار",             "يغادر الطيار المطار"),
    ("درست الطالبة الفيزياء",          "تدرس الطالبة الفيزياء"),
    ("ربح الفريق المباراة",            "يربح الفريق المباراة"),
    ("خسرت البنت اللعبة",              "تخسر البنت اللعبة"),
    ("وصل القطار المحطة",              "يصل القطار المحطة"),
    ("غادرت الطائرة المطار",           "تغادر الطائرة المطار"),
    ("فاز الرياضي بالميدالية",         "يفوز الرياضي بالميدالية"),
    ("اكمل الطالب الواجب",             "يكمل الطالب الواجب"),
    ("تركت البنت المدرسة",             "تترك البنت المدرسة"),
    ("زار الرجل العائلة",              "يزور الرجل العائلة"),
    ("استقبلت المراة الضيوف",          "تستقبل المراة الضيوف"),
    ("ادار المدير الشركة",             "يدير المدير الشركة"),
    ("اعد الطباخ العشاء",              "يعد الطباخ العشاء"),
    ("حكم القاضي بالبراءة",            "يحكم القاضي بالبراءة"),
]


def build_wordorder_matched():
    """Build 60 matched SVO/VSO pairs from the 20 new triples."""
    entries = []
    for pair_id, (subj, past_v, pres_v, obj) in enumerate(NEW_TRIPLES, start=41):
        # Use present-tense verb for the SVO/VSO sentences
        # (past-tense also fine; we alternate to add variety)
        verb = past_v if pair_id % 2 == 0 else pres_v
        svo = f"{subj} {verb} {obj}"
        vso = f"{verb} {subj} {obj}"
        entries.append({"sentence": svo, "label": "WordOrder_SVO",
                        "pair_id": pair_id, "source": "generated"})
        entries.append({"sentence": vso, "label": "WordOrder_VSO",
                        "pair_id": pair_id, "source": "generated"})
    return entries


def build_tense_binary():
    """50 past + 50 present from TENSE_PAIRS."""
    entries = []
    for past_s, pres_s in TENSE_PAIRS:
        entries.append({"sentence": past_s,  "label": "Tense_Past"})
        entries.append({"sentence": pres_s,  "label": "Tense_Pres"})
    return entries


def main():
    path = Path("sentences.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    # ── WordOrderMatched ──────────────────────────────────────────────────────
    # Combine existing 40+40 from WordOrder with 20 new matched pairs
    existing_wo = [
        {**d, "pair_id": i % 40, "source": "original"}
        for i, d in enumerate(data["WordOrder"])
        if d["label"] in ("WordOrder_SVO", "WordOrder_VSO")
    ]
    new_wo = build_wordorder_matched()
    matched = existing_wo + new_wo

    from collections import Counter
    c = Counter(d["label"] for d in matched)
    print(f"WordOrderMatched: {dict(c)}  total={len(matched)}")

    data["WordOrderMatched"] = matched

    # ── TenseBinary ───────────────────────────────────────────────────────────
    tense = build_tense_binary()
    c2 = Counter(d["label"] for d in tense)
    print(f"TenseBinary: {dict(c2)}  total={len(tense)}")

    data["TenseBinary"] = tense

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved → sentences.json")

    # ── Quick sanity check ────────────────────────────────────────────────────
    print("\nSample WordOrderMatched:")
    for d in matched[38:44]:
        print(f"  [{d['pair_id']}] {d['label']}: {d['sentence']}")
    print("\nSample TenseBinary:")
    for d in tense[:4]:
        print(f"  {d['label']}: {d['sentence']}")


if __name__ == "__main__":
    main()
