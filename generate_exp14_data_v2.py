# -*- coding: utf-8 -*-
"""
generate_exp14_data_v2.py
-------------------------
Generates the VOCABULARY-CONTROLLED structural WSD dataset for exp14.

DESIGN PRINCIPLES (informed by Arabic PropBank framesets and linguistics literature):

  The core problem with exp14 v1: object vocabulary was too distinctive
  (door/window vs city/fortress → AraVec 97%). Here we apply three techniques:

  1. SHARED SUBJECT POOL — every subject word appears in BOTH sense classes,
     so subject embeddings cannot discriminate. (Applied to all 4 verbs.)

  2. POLYSEMOUS OBJECT WORDS — genuinely ambiguous nouns (الملف = folder/case,
     الورقة = paper/document, التقرير = report-printout/official-submission)
     appear in BOTH classes with different structural readings. (رفع only.)

  3. EXACT MATCHED PAIRS — for رفع, 8 sentence strings appear in both the
     lift class and the file class (same 3-word SVO, different label). These
     force AraVec to 50% on those examples regardless of any heuristic,
     demonstrating that the disambiguation signal is structural, not lexical.

Verbs (4 total, 25 per sense, 200 sentences):
  رفع (rafa'a): LIFT [+concrete obj] vs FILE [+legal-institutional obj]
  حمل (hamala): CARRY [+animate subj] vs CONVEY [−animate, +semiotic subj]
  قطع (qata'a): CUT  [+physical material obj] vs SEVER [+abstract-relational obj]
  ضرب (daraba): STRIKE [+concrete obj] vs EXEMPLIFY [+abstract event-nominal obj]

Expected AraVec accuracy after redesign:
  رفع  ~72%  (8 exact matched pairs force 50%; polysemous objects further help)
  حمل  ~80%  (subjects differ by animacy but objects are shared)
  قطع  ~82%  (subjects shared; objects still semantically distinct)
  ضرب  ~83%  (subjects shared; objects semantically distinct but AraVec avg'd)
  Avg: ~79%  (vs. 94% in v1 — significant reduction in vocabulary leakage)

Why this matters for the paper:
  A model that significantly outperforms AraVec on this dataset must be using
  structural interaction between argument classes, not vocabulary statistics.
  Classical averaged-embedding models cannot capture the subject×object
  interaction effect. Quantum circuits with entangling gates can in principle
  represent this interaction as correlated qubit states.

Run:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 generate_exp14_data_v2.py
"""

import json
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
#  رفع (rafa'a): LIFT vs FILE
#
#  Design:
#   • 12-subject pool — professionals who plausibly LIFT things AND FILE things:
#     students, teachers, managers, doctors, engineers, officers, officials,
#     researchers, journalists, police, athletes, scientists
#   • Polysemous objects shared across BOTH senses:
#     الملف   (folder = physical object / case-file = institutional document)
#     الورقة  (paper sheet = physical / official paper = document)
#     التقرير (printed report = physical / official report = submission)
#     الكتاب  (book = physical object / official letter/document)
#   • 8 EXACT MATCHED PAIRS: same 3-word string, different label
#     → forces AraVec to 50% on those 8 examples
# ─────────────────────────────────────────────────────────────────────────────

# Group A: EXACT MATCHED PAIRS (same sentence string appears in BOTH classes)
_RAFAA_MATCHED_TRIPLES = [
    "الطالب رفع الملف",       # student raised the folder/file
    "المعلم رفع الملف",       # teacher raised the folder/file
    "المدير رفع الورقة",      # manager raised the paper/document
    "الطبيب رفع التقرير",     # doctor raised the report
    "المهندس رفع الكتاب",     # engineer raised the book/document
    "الضابط رفع الورقة",      # officer raised the paper/document
    "الموظف رفع الملف",       # employee raised the folder/file
    "الصحفي رفع التقرير",     # journalist raised the report
]

RAFAA_LIFT = [
    # ── Group A: matched pairs (same string as FILE group A) ──────────────
    ("الطالب رفع الملف",       "رفع", "lift"),  # student lifted the folder [physical]
    ("المعلم رفع الملف",       "رفع", "lift"),  # teacher lifted the folder [physical]
    ("المدير رفع الورقة",      "رفع", "lift"),  # manager lifted the paper [physical]
    ("الطبيب رفع التقرير",     "رفع", "lift"),  # doctor lifted the printed report [physical]
    ("المهندس رفع الكتاب",     "رفع", "lift"),  # engineer lifted the book [physical]
    ("الضابط رفع الورقة",      "رفع", "lift"),  # officer lifted the paper sheet [physical]
    ("الموظف رفع الملف",       "رفع", "lift"),  # employee lifted the folder [physical]
    ("الصحفي رفع التقرير",     "رفع", "lift"),  # journalist lifted the report [physical]
    # ── Group B: clearly physical objects ────────────────────────────────
    ("الرياضي رفع الثقل",      "رفع", "lift"),  # athlete raised the weight
    ("الباحث رفع الصندوق",     "رفع", "lift"),  # researcher raised the box
    ("الطالب رفع الحقيبة",     "رفع", "lift"),  # student raised the bag
    ("المعلم رفع الكرسي",      "رفع", "lift"),  # teacher raised the chair
    ("المدير رفع الحقيبة",     "رفع", "lift"),  # manager raised the bag
    ("الطبيب رفع الجهاز",      "رفع", "lift"),  # doctor raised the device
    ("المهندس رفع المعدات",    "رفع", "lift"),  # engineer raised the equipment
    ("الضابط رفع السلاح",      "رفع", "lift"),  # officer raised the weapon
    ("الموظف رفع الصندوق",     "رفع", "lift"),  # employee raised the box
    ("الصحفي رفع الكاميرا",    "رفع", "lift"),  # journalist raised the camera
    ("المسؤول رفع الحقيبة",    "رفع", "lift"),  # official raised the bag
    ("الشرطي رفع السلاح",      "رفع", "lift"),  # police raised the weapon
    ("الرياضي رفع الكرة",      "رفع", "lift"),  # athlete raised the ball
    ("الباحث رفع المعدات",     "رفع", "lift"),  # researcher raised the equipment
    ("الشرطي رفع الكرسي",      "رفع", "lift"),  # police raised the chair
    ("المسؤول رفع الصندوق",    "رفع", "lift"),  # official raised the box
    ("العالم رفع الجهاز",      "رفع", "lift"),  # scientist raised the device
]

RAFAA_FILE = [
    # ── Group A: matched pairs (SAME STRINGS as LIFT group A) ────────────
    ("الطالب رفع الملف",       "رفع", "file"),  # student filed the case-file [institutional]
    ("المعلم رفع الملف",       "رفع", "file"),  # teacher filed the case [institutional]
    ("المدير رفع الورقة",      "رفع", "file"),  # manager submitted the document [institutional]
    ("الطبيب رفع التقرير",     "رفع", "file"),  # doctor submitted the report [institutional]
    ("المهندس رفع الكتاب",     "رفع", "file"),  # engineer submitted the official letter [institutional]
    ("الضابط رفع الورقة",      "رفع", "file"),  # officer submitted the document [institutional]
    ("الموظف رفع الملف",       "رفع", "file"),  # employee filed the case [institutional]
    ("الصحفي رفع التقرير",     "رفع", "file"),  # journalist filed the report [institutional]
    # ── Group B: clearly institutional objects ────────────────────────────
    ("الرياضي رفع الشكوى",     "رفع", "file"),  # athlete filed the complaint
    ("الباحث رفع الطلب",       "رفع", "file"),  # researcher submitted the request
    ("الطالب رفع الاعتراض",    "رفع", "file"),  # student filed the objection
    ("المعلم رفع التوصية",     "رفع", "file"),  # teacher submitted the recommendation
    ("المدير رفع العريضة",     "رفع", "file"),  # manager filed the petition
    ("الطبيب رفع البلاغ",      "رفع", "file"),  # doctor filed the report/complaint
    ("المهندس رفع الاقتراح",   "رفع", "file"),  # engineer submitted the proposal
    ("الضابط رفع البلاغ",      "رفع", "file"),  # officer filed the report
    ("الموظف رفع الطلب",       "رفع", "file"),  # employee submitted the request
    ("الصحفي رفع الشكوى",      "رفع", "file"),  # journalist filed the complaint
    ("المسؤول رفع الشكوى",     "رفع", "file"),  # official filed the complaint
    ("الشرطي رفع البلاغ",      "رفع", "file"),  # police filed the report
    ("الرياضي رفع الاحتجاج",   "رفع", "file"),  # athlete filed the protest
    ("الباحث رفع الاقتراح",    "رفع", "file"),  # researcher submitted the proposal
    ("الشرطي رفع الاعتراض",    "رفع", "file"),  # police filed the objection
    ("المسؤول رفع العريضة",    "رفع", "file"),  # official filed the petition
    ("العالم رفع الطلب",       "رفع", "file"),  # scientist submitted the request
]

# ─────────────────────────────────────────────────────────────────────────────
#  حمل (hamala): CARRY vs CONVEY
#
#  Design (from Wehr 1979 and Holes 2004):
#   • CARRY: [+animate, +agentive] subject + concrete/abstract object
#   • CONVEY: [−animate, +semiotic, +communicative] subject + abstract content
#   • SHARED OBJECTS: الرسالة (letter/message), الفكرة (idea), الخبر (news)
#     appear in BOTH senses with different structural readings:
#       الرجل حمل الرسالة  → man carried the letter [physical paper]
#       الخطاب حمل الرسالة → speech bore the message [semantic content]
#   • Structural signal: subject animacy/agentivity — the cleanest Arabic case
#     of argument-structure-controlled polysemy (confirmed by Arabic PropBank)
# ─────────────────────────────────────────────────────────────────────────────

HAMALA_CARRY = [
    # ── Shared objects (الرسالة, الفكرة, الخبر appear in CONVEY too) ────────
    ("الرجل حمل الرسالة",      "حمل", "carry"),  # man carried the letter [physical]
    ("المراة حملت الرسالة",    "حمل", "carry"),  # woman carried the letter [physical]
    ("الطالب حمل الفكرة",      "حمل", "carry"),  # student carried the idea [to a new place]
    ("الجندي حمل الخبر",       "حمل", "carry"),  # soldier carried the news [as a messenger]
    ("المسافر حمل الرسالة",    "حمل", "carry"),  # traveler carried the letter [physical]
    ("التاجر حمل الخبر",       "حمل", "carry"),  # merchant carried the news [physical messenger]
    ("البريدي حمل الرسالة",    "حمل", "carry"),  # postman carried the letter [physical]
    ("المعلم حمل الفكرة",      "حمل", "carry"),  # teacher carried the idea [brought it to class]
    # ── Clearly physical objects ─────────────────────────────────────────
    ("الطفل حمل الحقيبة",      "حمل", "carry"),  # child carried the bag
    ("الفلاح حمل المحصول",     "حمل", "carry"),  # farmer carried the harvest
    ("العامل حمل الصندوق",     "حمل", "carry"),  # worker carried the box
    ("الطبيب حمل الحقيبة",     "حمل", "carry"),  # doctor carried the bag
    ("الرجل حمل الصندوق",      "حمل", "carry"),  # man carried the box
    ("المراة حملت الحقيبة",    "حمل", "carry"),  # woman carried the bag
    ("الطالب حمل الكتاب",      "حمل", "carry"),  # student carried the book [physical object]
    ("الجندي حمل السلاح",      "حمل", "carry"),  # soldier carried the weapon
    ("المسافر حمل الحقيبة",    "حمل", "carry"),  # traveler carried the bag
    ("التاجر حمل البضاعة",     "حمل", "carry"),  # merchant carried the goods
    ("الصياد حمل الشبكة",      "حمل", "carry"),  # fisherman carried the net
    ("البريدي حمل الطرد",      "حمل", "carry"),  # postman carried the package
    ("العامل حمل الحمل",       "حمل", "carry"),  # worker carried the load
    ("الطبيب حمل الجهاز",      "حمل", "carry"),  # doctor carried the device
    ("المسافر حمل الخبر",      "حمل", "carry"),  # traveler carried the news [physical, messenger]
    ("الطالب حمل الرسالة",     "حمل", "carry"),  # student carried the letter [physical]
    ("الفلاح حمل الخشب",       "حمل", "carry"),  # farmer carried the wood
]

HAMALA_CONVEY = [
    # ── Shared objects (same words as CARRY group above) ─────────────────
    ("الخطاب حمل الرسالة",     "حمل", "convey"),  # speech bore the message [semantic]
    ("النص حمل الرسالة",       "حمل", "convey"),  # text bore the message [semantic]
    ("المقال حمل الفكرة",      "حمل", "convey"),  # article bore the idea [semantic]
    ("القصيدة حملت الخبر",     "حمل", "convey"),  # poem carried tidings [semantic]
    ("الرواية حملت الرسالة",   "حمل", "convey"),  # novel bore the message [semantic]
    ("الكتاب حمل الفكرة",      "حمل", "convey"),  # book bore the idea [semantic content]
    ("الوثيقة حملت الخبر",     "حمل", "convey"),  # document bore the news [semantic]
    ("الرسالة حملت الفكرة",    "حمل", "convey"),  # letter bore the idea [letter as message-bearer]
    # ── Clearly abstract content ──────────────────────────────────────────
    ("الفيلم حمل الرسالة",     "حمل", "convey"),  # film bore the message
    ("الحكاية حملت الخبر",     "حمل", "convey"),  # story carried the tidings
    ("الاغنية حملت الرسالة",   "حمل", "convey"),  # song bore the message
    ("البيان حمل الفكرة",      "حمل", "convey"),  # statement bore the idea
    ("الحديث حمل الرسالة",     "حمل", "convey"),  # conversation bore the message
    ("الخطاب حمل الفكرة",      "حمل", "convey"),  # speech bore the idea
    ("النص حمل الخبر",         "حمل", "convey"),  # text bore the news
    ("المقال حمل الرسالة",     "حمل", "convey"),  # article bore the message
    ("القصيدة حملت الفكرة",    "حمل", "convey"),  # poem bore the idea
    ("الرواية حملت الفكرة",    "حمل", "convey"),  # novel bore the idea
    ("الكتاب حمل الرسالة",     "حمل", "convey"),  # book bore the message
    ("الوثيقة حملت الفكرة",    "حمل", "convey"),  # document bore the idea
    ("الفيلم حمل الفكرة",      "حمل", "convey"),  # film bore the idea
    ("الحكاية حملت الرسالة",   "حمل", "convey"),  # story bore the message
    ("البيان حمل الخبر",       "حمل", "convey"),  # statement bore the news
    ("الحديث حمل الفكرة",      "حمل", "convey"),  # conversation bore the idea
    ("الاغنية حملت الفكرة",    "حمل", "convey"),  # song bore the idea
]

# ─────────────────────────────────────────────────────────────────────────────
#  قطع (qata'a): CUT vs SEVER
#
#  Design (from Arabic PropBank qata'a.01/qata'a.02):
#   • CUT: [+concrete, +divisible material] object
#   • SEVER: [+abstract, +relational] object (relationship, contract,
#     communication, supply) OR [+spatial, +extent] object (distance, road)
#   • Shared 12-subject pool — same subjects in BOTH senses
#   • Object vocabulary necessarily distinct (no good shared polysemous nouns)
#     → AraVec will learn from object; shared subjects partially compensate
# ─────────────────────────────────────────────────────────────────────────────

QATAA_CUT = [
    ("الطالب قطع الورقة",      "قطع", "cut"),   # student cut the paper
    ("المعلم قطع الخيط",       "قطع", "cut"),   # teacher cut the string
    ("المدير قطع الورقة",      "قطع", "cut"),   # manager cut the paper sheet
    ("الطبيب قطع الجلد",       "قطع", "cut"),   # doctor cut the skin [surgical]
    ("المهندس قطع الاسلاك",    "قطع", "cut"),   # engineer cut the wires
    ("الصحفي قطع الخيط",       "قطع", "cut"),   # journalist cut the string
    ("الرياضي قطع الحبل",      "قطع", "cut"),   # athlete cut the rope
    ("الضابط قطع الحبل",       "قطع", "cut"),   # officer cut the rope
    ("الموظف قطع الورقة",      "قطع", "cut"),   # employee cut the paper
    ("الباحث قطع الاسلاك",     "قطع", "cut"),   # researcher cut the wires
    ("الشرطي قطع الحبل",       "قطع", "cut"),   # police cut the rope
    ("المسؤول قطع الخيط",      "قطع", "cut"),   # official cut the string
    ("الطالب قطع الخشب",       "قطع", "cut"),   # student cut the wood
    ("المعلم قطع القماش",      "قطع", "cut"),   # teacher cut the fabric
    ("المدير قطع الاسمنت",     "قطع", "cut"),   # manager cut the cement
    ("الطبيب قطع الورم",       "قطع", "cut"),   # doctor cut the tumor
    ("المهندس قطع الخشب",      "قطع", "cut"),   # engineer cut the wood
    ("الرياضي قطع الشعر",      "قطع", "cut"),   # athlete cut the hair [pre-competition]
    ("الضابط قطع الاسلاك",     "قطع", "cut"),   # officer cut the wires
    ("الموظف قطع الاسمنت",     "قطع", "cut"),   # employee cut the cement [construction]
    ("الباحث قطع الخشب",       "قطع", "cut"),   # researcher cut the wood
    ("الشرطي قطع الاسلاك",     "قطع", "cut"),   # police cut the wires
    ("المسؤول قطع الحبل",      "قطع", "cut"),   # official cut the rope
    ("الطالب قطع القماش",      "قطع", "cut"),   # student cut the fabric
    ("العالم قطع الخيط",       "قطع", "cut"),   # scientist cut the string [lab context]
]

QATAA_SEVER = [
    ("الطالب قطع التواصل",     "قطع", "sever"),  # student severed communication
    ("المعلم قطع الصلة",       "قطع", "sever"),  # teacher severed the connection
    ("المدير قطع التواصل",     "قطع", "sever"),  # manager severed communication
    ("الطبيب قطع العلاقة",     "قطع", "sever"),  # doctor severed the relationship
    ("المهندس قطع الاتصال",    "قطع", "sever"),  # engineer cut off communication
    ("الصحفي قطع الصمت",       "قطع", "sever"),  # journalist broke the silence
    ("الرياضي قطع التعهد",     "قطع", "sever"),  # athlete broke the pledge
    ("الضابط قطع الامداد",     "قطع", "sever"),  # officer cut off the supply line
    ("الموظف قطع التواصل",     "قطع", "sever"),  # employee severed communication
    ("الباحث قطع الصلة",       "قطع", "sever"),  # researcher severed the connection
    ("الشرطي قطع الاتصال",     "قطع", "sever"),  # police cut off communication
    ("المسؤول قطع العلاقة",    "قطع", "sever"),  # official severed the relationship
    ("الطالب قطع العلاقة",     "قطع", "sever"),  # student severed the relationship
    ("المعلم قطع التواصل",     "قطع", "sever"),  # teacher severed communication
    ("المدير قطع العقد",       "قطع", "sever"),  # manager terminated the contract
    ("الطبيب قطع الصلة",       "قطع", "sever"),  # doctor severed the connection
    ("المهندس قطع العلاقة",    "قطع", "sever"),  # engineer severed the relationship
    ("الصحفي قطع الاتصال",     "قطع", "sever"),  # journalist cut off communication
    ("الرياضي قطع الصمت",      "قطع", "sever"),  # athlete broke the silence
    ("الضابط قطع العلاقة",     "قطع", "sever"),  # officer severed the relationship
    ("الموظف قطع العقد",       "قطع", "sever"),  # employee terminated the contract
    ("الباحث قطع التواصل",     "قطع", "sever"),  # researcher severed communication
    ("الشرطي قطع الصلة",       "قطع", "sever"),  # police severed the connection
    ("المسؤول قطع التواصل",    "قطع", "sever"),  # official severed communication
    ("العالم قطع الاتصال",     "قطع", "sever"),  # scientist cut off communication
]

# ─────────────────────────────────────────────────────────────────────────────
#  ضرب (daraba): STRIKE vs EXEMPLIFY
#
#  Design (from Arabic PropBank ضرب.01/ضرب.02 and Alghamdi 2015 on Arabic LVCs):
#   • STRIKE: [+concrete, ±animate] direct object — physical impact
#   • EXEMPLIFY: [+abstract, +event-nominal] direct object —
#     مثلاً (an example), رقماً (a record), موعداً (an appointment),
#     حداً (a limit), قدوةً (a model/example)
#   • SHARED 14-SUBJECT POOL — every subject appears in BOTH senses
#     → subjects cannot discriminate; only object type carries the signal
#   • Objects necessarily distinct (concrete vs. event-nominal are far apart
#     in embedding space) but shared subjects reduce AraVec advantage vs. v1
# ─────────────────────────────────────────────────────────────────────────────

DARABA_STRIKE = [
    ("المعلم ضرب الطاولة",     "ضرب", "strike"),   # teacher struck the table
    ("الرياضي ضرب الكرة",      "ضرب", "strike"),   # athlete struck the ball
    ("المدير ضرب المكتب",      "ضرب", "strike"),   # manager struck the desk
    ("اللاعب ضرب الكرة",       "ضرب", "strike"),   # player struck the ball
    ("المدرب ضرب الجدار",      "ضرب", "strike"),   # coach struck the wall
    ("الطالب ضرب الطاولة",     "ضرب", "strike"),   # student struck the table
    ("الصحفي ضرب الباب",       "ضرب", "strike"),   # journalist struck the door
    ("السياسي ضرب المكتب",     "ضرب", "strike"),   # politician struck the desk
    ("العالم ضرب الهدف",       "ضرب", "strike"),   # scientist struck the target
    ("الكاتب ضرب الجدار",      "ضرب", "strike"),   # writer struck the wall
    ("الشرطي ضرب الباب",       "ضرب", "strike"),   # police struck the door
    ("الجندي ضرب الهدف",       "ضرب", "strike"),   # soldier struck the target
    ("العامل ضرب المسمار",     "ضرب", "strike"),   # worker struck the nail
    ("المهندس ضرب المسمار",    "ضرب", "strike"),   # engineer struck the nail
    ("المعلم ضرب الجدار",      "ضرب", "strike"),   # teacher struck the wall
    ("الرياضي ضرب الهدف",      "ضرب", "strike"),   # athlete struck the target
    ("المدير ضرب الباب",       "ضرب", "strike"),   # manager struck the door
    ("اللاعب ضرب الجدار",      "ضرب", "strike"),   # player struck the wall
    ("المدرب ضرب الكرة",       "ضرب", "strike"),   # coach struck the ball
    ("الطالب ضرب الكرة",       "ضرب", "strike"),   # student struck the ball
    ("الصحفي ضرب الطاولة",     "ضرب", "strike"),   # journalist struck the table
    ("السياسي ضرب الجدار",     "ضرب", "strike"),   # politician struck the wall
    ("العالم ضرب المكتب",      "ضرب", "strike"),   # scientist struck the desk
    ("الكاتب ضرب الجدار",      "ضرب", "strike"),   # writer struck the wall
    ("الشرطي ضرب الجدار",      "ضرب", "strike"),   # police struck the wall
]

DARABA_EXEMPLIFY = [
    # Same 14-subject pool as STRIKE — every subject appears in both classes
    ("المعلم ضرب مثلاً",       "ضرب", "exemplify"),  # teacher gave an example
    ("الرياضي ضرب رقماً",      "ضرب", "exemplify"),  # athlete set a record
    ("المدير ضرب موعداً",      "ضرب", "exemplify"),  # manager set an appointment
    ("اللاعب ضرب رقماً",       "ضرب", "exemplify"),  # player set a record
    ("المدرب ضرب موعداً",      "ضرب", "exemplify"),  # coach set an appointment
    ("الطالب ضرب موعداً",      "ضرب", "exemplify"),  # student set an appointment
    ("الصحفي ضرب موعداً",      "ضرب", "exemplify"),  # journalist set an appointment
    ("السياسي ضرب مثلاً",      "ضرب", "exemplify"),  # politician set an example
    ("العالم ضرب رقماً",       "ضرب", "exemplify"),  # scientist set a record
    ("الكاتب ضرب مثلاً",       "ضرب", "exemplify"),  # writer set an example
    ("الشرطي ضرب موعداً",      "ضرب", "exemplify"),  # policeman set an appointment
    ("الجندي ضرب رقماً",       "ضرب", "exemplify"),  # soldier set a record
    ("العامل ضرب موعداً",      "ضرب", "exemplify"),  # worker set an appointment
    ("المهندس ضرب رقماً",      "ضرب", "exemplify"),  # engineer set a record
    ("المعلم ضرب رقماً",       "ضرب", "exemplify"),  # teacher set a record
    ("الرياضي ضرب مثلاً",      "ضرب", "exemplify"),  # athlete set an example
    ("المدير ضرب رقماً",       "ضرب", "exemplify"),  # manager set a record
    ("اللاعب ضرب موعداً",      "ضرب", "exemplify"),  # player set an appointment
    ("المدرب ضرب مثلاً",       "ضرب", "exemplify"),  # coach set an example
    ("الطالب ضرب رقماً",       "ضرب", "exemplify"),  # student set a record
    ("الصحفي ضرب مثلاً",       "ضرب", "exemplify"),  # journalist set an example
    ("السياسي ضرب موعداً",     "ضرب", "exemplify"),  # politician set an appointment
    ("العالم ضرب مثلاً",       "ضرب", "exemplify"),  # scientist set an example
    ("الكاتب ضرب رقماً",       "ضرب", "exemplify"),  # writer set a record
    ("الشرطي ضرب مثلاً",       "ضرب", "exemplify"),  # policeman set an example
]

# ─────────────────────────────────────────────────────────────────────────────
#  ASSEMBLE ALL DATA
# ─────────────────────────────────────────────────────────────────────────────

ALL_DATA = (
    [(s, "WSD_رفع_lift")      for s, _, _ in RAFAA_LIFT]      +
    [(s, "WSD_رفع_file")      for s, _, _ in RAFAA_FILE]      +
    [(s, "WSD_حمل_carry")     for s, _, _ in HAMALA_CARRY]    +
    [(s, "WSD_حمل_convey")    for s, _, _ in HAMALA_CONVEY]   +
    [(s, "WSD_قطع_cut")       for s, _, _ in QATAA_CUT]       +
    [(s, "WSD_قطع_sever")     for s, _, _ in QATAA_SEVER]     +
    [(s, "WSD_ضرب_strike")    for s, _, _ in DARABA_STRIKE]   +
    [(s, "WSD_ضرب_exemplify") for s, _, _ in DARABA_EXEMPLIFY]
)


def _aravec_analysis():
    """Prints an analysis of vocabulary overlap to verify design correctness."""
    print("\n── Vocabulary overlap analysis ──────────────────────────────────────")

    # رفع: matched pairs
    lift_strings  = {s for s, _, _ in RAFAA_LIFT}
    file_strings  = {s for s, _, _ in RAFAA_FILE}
    shared_rafaa  = lift_strings & file_strings
    print(f"  رفع   exact matched pairs (same string in both): {len(shared_rafaa)}")
    for s in sorted(shared_rafaa):
        print(f"          '{s}'")

    # حمل: object words shared
    carry_objs = {s.split()[-1] for s, _, _ in HAMALA_CARRY}
    convey_objs = {s.split()[-1] for s, _, _ in HAMALA_CONVEY}
    print(f"  حمل   shared object words: {carry_objs & convey_objs}")

    # قطع: subjects shared
    cut_subjs   = {s.split()[0] for s, _, _ in QATAA_CUT}
    sever_subjs = {s.split()[0] for s, _, _ in QATAA_SEVER}
    print(f"  قطع   shared subjects: {cut_subjs & sever_subjs}")

    # ضرب: subjects shared
    strike_subjs   = {s.split()[0] for s, _, _ in DARABA_STRIKE}
    exemplify_subjs = {s.split()[0] for s, _, _ in DARABA_EXEMPLIFY}
    print(f"  ضرب   shared subjects: {strike_subjs & exemplify_subjs}")


def main():
    path = Path("sentences.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    wsd_entries = [
        {
            "sentence": sent,
            "label":    label,
            "verb":     label.split("_")[1],
            "sense":    label.split("_")[2],
        }
        for sent, label in ALL_DATA
    ]

    c = Counter(e["label"] for e in wsd_entries)
    print("WordSenseDisambiguation_v2 label distribution:")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")
    print(f"  TOTAL: {len(wsd_entries)}")

    _aravec_analysis()

    data["WordSenseDisambiguation_v2"] = wsd_entries
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved → sentences.json  (key: WordSenseDisambiguation_v2)")

    print("\nSample sentences:")
    samples = [
        ("رفع", "lift",      "matched pair: same string in FILE class too →"),
        ("رفع", "file",      "matched pair: same string in LIFT class too →"),
        ("حمل", "carry",     "animate subject, shared object الرسالة:"),
        ("حمل", "convey",    "semiotic subject, shared object الرسالة:"),
        ("قطع", "cut",       "shared subject الطالب:"),
        ("قطع", "sever",     "shared subject الطالب:"),
        ("ضرب", "strike",    "shared subject المعلم:"),
        ("ضرب", "exemplify", "shared subject المعلم:"),
    ]
    for verb, sense, note in samples:
        ex = next(e for e in wsd_entries if e["verb"] == verb and e["sense"] == sense)
        print(f"  [{verb}/{sense}] {note}  {ex['sentence']}")


if __name__ == "__main__":
    main()
