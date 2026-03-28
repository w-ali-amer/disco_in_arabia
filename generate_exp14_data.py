# -*- coding: utf-8 -*-
"""
generate_exp14_data.py
----------------------
Generates the structural word-sense disambiguation dataset for exp14.

Three Arabic verbs where sense is determined by the ARGUMENT STRUCTURE
(semantic class of the object noun), not by surrounding vocabulary:

  فتح (fataḥa): OPEN (physical container) vs CONQUER (territory)
  رفع (rafaʿa): LIFT  (physical object)   vs FILE    (institutional document)
  قطع (qaṭaʿa): CUT   (physical material)  vs SEVER   (abstract relation/distance)

Design principles:
  - Subject is always a human agent (animate) — held approximately constant
  - Object is either CONCRETE (physical) or ABSTRACT (institutional/relational)
  - Vocabulary deliberately varied across sentences within each sense
  - 20 sentences per sense per verb = 120 total
  - Adds key "WordSenseDisambiguation" to sentences.json

Run:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 generate_exp14_data.py
"""

import json
from pathlib import Path
from collections import Counter

# ─── فتح: open (physical) vs conquer (territory) ──────────────────────────────
FATAHA_OPEN = [
    ("الرجل فتح الباب",         "فتح", "open"),   # the man opened the door
    ("الولد فتح الصندوق",       "فتح", "open"),   # the boy opened the box
    ("الطالب فتح الكتاب",       "فتح", "open"),   # the student opened the book
    ("المدير فتح الملف",        "فتح", "open"),   # the director opened the folder
    ("الطبيب فتح الحقيبة",      "فتح", "open"),   # the doctor opened the bag
    ("البائع فتح المتجر",       "فتح", "open"),   # the seller opened the shop
    ("العامل فتح النافذة",      "فتح", "open"),   # the worker opened the window
    ("الطفل فتح العلبة",        "فتح", "open"),   # the child opened the can
    ("المراة فتحت الباب",       "فتح", "open"),   # the woman opened the door
    ("الاستاذ فتح الدفتر",      "فتح", "open"),   # the teacher opened the notebook
    ("السائق فتح الشنطة",       "فتح", "open"),   # the driver opened the suitcase
    ("المهندس فتح الحاسوب",     "فتح", "open"),   # the engineer opened the computer
    ("الجندي فتح الخزانة",      "فتح", "open"),   # the soldier opened the cabinet
    ("الطباخ فتح الوعاء",       "فتح", "open"),   # the cook opened the pot
    ("التاجر فتح الصندوق",      "فتح", "open"),   # the merchant opened the crate
    ("الرجل يفتح الباب",        "فتح", "open"),
    ("الطالب يفتح الكتاب",      "فتح", "open"),
    ("المراة تفتح النافذة",     "فتح", "open"),
    ("العامل يفتح العلبة",      "فتح", "open"),
    ("الولد يفتح الحقيبة",      "فتح", "open"),   # the boy opens the bag
]

FATAHA_CONQUER = [
    ("الجيش فتح المدينة",       "فتح", "conquer"),  # the army conquered the city
    ("القائد فتح البلاد",       "فتح", "conquer"),  # the leader conquered the country
    ("المحارب فتح الحصن",       "فتح", "conquer"),  # the warrior conquered the fortress
    ("الملك فتح المنطقة",       "فتح", "conquer"),  # the king conquered the region
    ("الجنود فتحوا الارض",      "فتح", "conquer"),  # the soldiers conquered the land
    ("الفاتح فتح القلعة",       "فتح", "conquer"),  # the conqueror took the castle
    ("الامير فتح الاقليم",      "فتح", "conquer"),  # the prince conquered the province
    ("الجيش فتح البلدة",        "فتح", "conquer"),  # the army took the town
    ("القائد فتح الميناء",      "فتح", "conquer"),  # the leader seized the port
    ("الغازي فتح الاقليم",      "فتح", "conquer"),  # the invader took the province
    ("العسكر فتح القلعة",       "فتح", "conquer"),  # the military took the castle
    ("المقاتل فتح المدينة",     "فتح", "conquer"),  # the fighter took the city
    ("السلطان فتح البلاد",      "فتح", "conquer"),  # the sultan conquered the land
    ("الجيش يفتح المدينة",      "فتح", "conquer"),
    ("القائد يفتح البلاد",      "فتح", "conquer"),
    ("المحارب يفتح الحصن",      "فتح", "conquer"),
    ("الملك يفتح الارض",        "فتح", "conquer"),
    ("الجنود يفتحون الحصن",     "فتح", "conquer"),  # the soldiers conquer the fortress
    ("الامير يفتح المنطقة",     "فتح", "conquer"),  # the prince conquers the region
    ("القوات فتحت الارض",       "فتح", "conquer"),  # the forces conquered the land
]

# ─── رفع: lift (physical) vs file (institutional) ─────────────────────────────
RAFAA_LIFT = [
    ("العامل رفع الصندوق",       "رفع", "lift"),    # the worker lifted the box
    ("الولد رفع الحجر",          "رفع", "lift"),    # the boy lifted the stone
    ("الجندي رفع السلاح",        "رفع", "lift"),    # the soldier raised the weapon
    ("الرياضي رفع الثقل",        "رفع", "lift"),    # the athlete lifted the weight
    ("البناء رفع الطوب",         "رفع", "lift"),    # the builder lifted the brick
    ("الطفل رفع الكرة",          "رفع", "lift"),    # the child lifted the ball
    ("السائق رفع الحقيبة",       "رفع", "lift"),    # the driver lifted the bag
    ("الطباخ رفع القدر",          "رفع", "lift"),   # the cook lifted the pot
    ("الحارس رفع السلاح",        "رفع", "lift"),    # the guard raised the weapon
    ("المهندس رفع المعدات",      "رفع", "lift"),    # the engineer raised the equipment
    ("المزارع رفع الحمل",        "رفع", "lift"),    # the farmer raised the load
    ("الطالب رفع الكتاب",        "رفع", "lift"),    # the student raised the book
    ("العمال رفعوا العمود",      "رفع", "lift"),    # the workers raised the pillar
    ("الفلاح رفع المحصول",       "رفع", "lift"),    # the farmer lifted the harvest
    ("العامل يرفع الصندوق",      "رفع", "lift"),
    ("الولد يرفع الكتاب",        "رفع", "lift"),
    ("الجندي يرفع الحجر",        "رفع", "lift"),
    ("الرياضي يرفع الوزن",       "رفع", "lift"),
    ("الطفل يرفع اللعبة",        "رفع", "lift"),    # the child lifts the toy
    ("البناء يرفع الاعمدة",      "رفع", "lift"),    # the builder raises the pillars
]

RAFAA_FILE = [
    ("المحامي رفع الشكوى",       "رفع", "file"),    # the lawyer filed the complaint
    ("المواطن رفع القضية",       "رفع", "file"),    # the citizen filed the case
    ("الموظف رفع التقرير",       "رفع", "file"),    # the employee submitted the report
    ("المدير رفع الطلب",         "رفع", "file"),    # the director submitted the request
    ("الطالب رفع الاعتراض",      "رفع", "file"),    # the student filed the objection
    ("المحكوم رفع الاستئناف",    "رفع", "file"),    # the convict filed the appeal
    ("المسؤول رفع التوصية",      "رفع", "file"),    # the official submitted the recommendation
    ("الصحفي رفع البلاغ",        "رفع", "file"),    # the journalist filed the report
    ("النائب رفع الاقتراح",      "رفع", "file"),    # the deputy submitted the proposal
    ("المستثمر رفع العريضة",     "رفع", "file"),    # the investor submitted the petition
    ("الشركة رفعت الدعوى",       "رفع", "file"),    # the company filed the lawsuit
    ("المعلم رفع الشكوى",        "رفع", "file"),    # the teacher filed the complaint
    ("المحامي يرفع الشكوى",      "رفع", "file"),
    ("المواطن يرفع القضية",      "رفع", "file"),
    ("الموظف يرفع التقرير",      "رفع", "file"),
    ("المدير يرفع الطلب",        "رفع", "file"),
    ("الطالب يرفع الطعن",        "رفع", "file"),    # the student files the appeal
    ("المحكوم يرفع الطعن",       "رفع", "file"),    # the convict files the appeal
    ("المسؤول يرفع الاقتراح",    "رفع", "file"),    # the official submits the proposal
    ("المدعي رفع الاتهام",       "رفع", "file"),    # the prosecutor filed the charge
]

# ─── قطع: cut (physical) vs sever/traverse (abstract) ───────────────────────
QATAA_CUT = [
    ("النجار قطع الخشب",         "قطع", "cut"),     # the carpenter cut the wood
    ("الطباخ قطع اللحم",         "قطع", "cut"),     # the cook cut the meat
    ("الحلاق قطع الشعر",         "قطع", "cut"),     # the barber cut the hair
    ("الجراح قطع الورم",         "قطع", "cut"),     # the surgeon cut the tumor
    ("الفلاح قطع الزرع",         "قطع", "cut"),     # the farmer cut the crops
    ("العامل قطع الحبل",         "قطع", "cut"),     # the worker cut the rope
    ("الحداد قطع المعدن",        "قطع", "cut"),     # the blacksmith cut the metal
    ("الخياط قطع القماش",        "قطع", "cut"),     # the tailor cut the fabric
    ("الطفل قطع الورقة",         "قطع", "cut"),     # the child cut the paper
    ("المزارع قطع الاشجار",      "قطع", "cut"),     # the farmer cut the trees
    ("الصياد قطع الخيط",         "قطع", "cut"),     # the fisherman cut the line
    ("البناء قطع الاسمنت",       "قطع", "cut"),     # the builder cut the concrete
    ("النجار يقطع الخشب",        "قطع", "cut"),
    ("الطباخ يقطع اللحم",        "قطع", "cut"),
    ("الحلاق يقطع الشعر",        "قطع", "cut"),
    ("الجراح يقطع الجلد",        "قطع", "cut"),     # the surgeon cuts the skin
    ("العامل يقطع الاسلاك",      "قطع", "cut"),     # the worker cuts the wires
    ("الفلاح يقطع الحشائش",      "قطع", "cut"),     # the farmer cuts the grass
    ("الخياط يقطع الجلد",        "قطع", "cut"),     # the tailor cuts the leather
    ("الحداد يقطع الاسلاك",      "قطع", "cut"),     # the blacksmith cuts the wires
]

QATAA_SEVER = [
    ("الرجل قطع العلاقة",        "قطع", "sever"),   # the man severed the relationship
    ("المسافر قطع المسافة",      "قطع", "sever"),   # the traveler crossed the distance
    ("الرياضي قطع الشوط",        "قطع", "sever"),   # the athlete completed the round
    ("الطالب قطع التعهد",        "قطع", "sever"),   # the student broke the pledge
    ("المدير قطع التواصل",       "قطع", "sever"),   # the director severed communication
    ("المسؤول قطع العقد",        "قطع", "sever"),   # the official severed the contract
    ("السياسي قطع الصمت",        "قطع", "sever"),   # the politician broke the silence
    ("الجيش قطع الامداد",        "قطع", "sever"),   # the army cut the supply line
    ("الزعيم قطع الصلة",         "قطع", "sever"),   # the leader severed the connection
    ("المواطن قطع الصمت",        "قطع", "sever"),   # the citizen broke the silence
    ("النائب قطع التواصل",       "قطع", "sever"),   # the deputy severed communication
    ("اللاعب قطع الشوط",         "قطع", "sever"),   # the player completed the round
    ("الرجل يقطع العلاقة",       "قطع", "sever"),
    ("المسافر يقطع المسافة",     "قطع", "sever"),
    ("الطالب يقطع التعهد",       "قطع", "sever"),
    ("المدير يقطع التواصل",      "قطع", "sever"),
    ("السياسي يقطع الصلة",       "قطع", "sever"),   # the politician severs connection
    ("المسؤول يقطع العقد",       "قطع", "sever"),   # the official severs the contract
    ("الزعيم يقطع الصمت",        "قطع", "sever"),   # the leader breaks the silence
    ("الشركة قطعت العلاقة",      "قطع", "sever"),   # the company severed the relationship
]

# ─── ضرب: strike (physical) vs exemplify/set (abstract event-nominal) ────────
# Cleanest structural binary: object is [+concrete ±animate] vs [-concrete +event-nominal]
# The abstract objects (مثلاً، رقماً، موعداً) are event-nominals, not nameable entities;
# this is the hardest test for vocabulary-based disambiguation (AraVec/BERT).
DARABA_STRIKE = [
    ("الجندي ضرب العدو",          "ضرب", "strike"),   # the soldier struck the enemy
    ("الرجل ضرب الكلب",           "ضرب", "strike"),   # the man hit the dog
    ("الطفل ضرب الكرة",           "ضرب", "strike"),   # the child hit the ball
    ("اللاعب ضرب الكرة",          "ضرب", "strike"),   # the player struck the ball
    ("الحارس ضرب المهاجم",        "ضرب", "strike"),   # the guard struck the attacker
    ("الطباخ ضرب اللحم",          "ضرب", "strike"),   # the cook beat the meat
    ("الحداد ضرب المعدن",         "ضرب", "strike"),   # the blacksmith struck the metal
    ("المزارع ضرب الارض",         "ضرب", "strike"),   # the farmer struck the ground
    ("الولد ضرب الباب",           "ضرب", "strike"),   # the boy struck the door
    ("الرياضي ضرب الهدف",         "ضرب", "strike"),   # the athlete struck the target
    ("الجندي يضرب العدو",         "ضرب", "strike"),
    ("الرجل يضرب الكلب",          "ضرب", "strike"),
    ("الطفل يضرب الكرة",          "ضرب", "strike"),
    ("اللاعب يضرب الكرة",         "ضرب", "strike"),
    ("الحداد يضرب المعدن",        "ضرب", "strike"),
    ("المزارع يضرب الارض",        "ضرب", "strike"),
    ("الولد يضرب الباب",          "ضرب", "strike"),
    ("الملاكم ضرب الخصم",         "ضرب", "strike"),   # the boxer struck the opponent
    ("الفارس ضرب الجواد",         "ضرب", "strike"),   # the knight struck the horse
    ("العامل ضرب المسمار",        "ضرب", "strike"),   # the worker struck the nail
]

DARABA_EXEMPLIFY = [
    ("المعلم ضرب مثلا",           "ضرب", "exemplify"),  # the teacher gave an example
    ("الخطيب ضرب مثلا",           "ضرب", "exemplify"),  # the orator set an example
    ("الزعيم ضرب موعدا",          "ضرب", "exemplify"),  # the leader set an appointment
    ("المدير ضرب موعدا",          "ضرب", "exemplify"),  # the director set a date
    ("الرياضي ضرب رقما",          "ضرب", "exemplify"),  # the athlete set a record
    ("الفريق ضرب رقما",           "ضرب", "exemplify"),  # the team set a record
    ("العالم ضرب مثلا",           "ضرب", "exemplify"),  # the scientist set an example
    ("المستثمر ضرب موعدا",        "ضرب", "exemplify"),  # the investor set a meeting
    ("السياسي ضرب موعدا",         "ضرب", "exemplify"),  # the politician set a date
    ("الكاتب ضرب مثلا",           "ضرب", "exemplify"),  # the writer set an example
    ("المعلم يضرب مثلا",          "ضرب", "exemplify"),
    ("الخطيب يضرب مثلا",          "ضرب", "exemplify"),
    ("المدير يضرب موعدا",         "ضرب", "exemplify"),
    ("الرياضي يضرب رقما",         "ضرب", "exemplify"),
    ("الفريق يضرب رقما",          "ضرب", "exemplify"),
    ("العالم يضرب مثلا",          "ضرب", "exemplify"),
    ("المستثمر يضرب موعدا",       "ضرب", "exemplify"),
    ("السياسي يضرب موعدا",        "ضرب", "exemplify"),
    ("الكاتب يضرب مثلا",          "ضرب", "exemplify"),
    ("الملاكم ضرب رقما",          "ضرب", "exemplify"),  # the boxer set a record
]


ALL_DATA = (
    [(s, "WSD_فتح_open")       for s, _, _ in FATAHA_OPEN]      +
    [(s, "WSD_فتح_conquer")    for s, _, _ in FATAHA_CONQUER]   +
    [(s, "WSD_رفع_lift")       for s, _, _ in RAFAA_LIFT]       +
    [(s, "WSD_رفع_file")       for s, _, _ in RAFAA_FILE]       +
    [(s, "WSD_قطع_cut")        for s, _, _ in QATAA_CUT]        +
    [(s, "WSD_قطع_sever")      for s, _, _ in QATAA_SEVER]      +
    [(s, "WSD_ضرب_strike")     for s, _, _ in DARABA_STRIKE]    +
    [(s, "WSD_ضرب_exemplify")  for s, _, _ in DARABA_EXEMPLIFY]
)


def main():
    path = Path("sentences.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    wsd_entries = [
        {
            "sentence": sent,
            "label": label,
            "verb": label.split("_")[1],
            "sense": label.split("_")[2],
        }
        for sent, label in ALL_DATA
    ]

    c = Counter(e["label"] for e in wsd_entries)
    print("WordSenseDisambiguation label distribution:")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")
    print(f"  TOTAL: {len(wsd_entries)}")

    data["WordSenseDisambiguation"] = wsd_entries
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved → sentences.json")

    print("\nSample sentences:")
    for verb in ["فتح", "رفع", "قطع", "ضرب"]:
        for sense in ["open", "conquer"]     if verb == "فتح" else \
                     ["lift", "file"]        if verb == "رفع" else \
                     ["cut", "sever"]        if verb == "قطع" else \
                     ["strike", "exemplify"]:
            ex = next(e for e in wsd_entries if e["verb"] == verb and e["sense"] == sense)
            print(f"  [{verb}/{sense}] {ex['sentence']}")


if __name__ == "__main__":
    main()
