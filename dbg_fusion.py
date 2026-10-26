# -*- coding: utf-8 -*-
import os
import arabic_dep_reader as adr
ref = adr.sentence_to_diagram("حمل الولد الحقيبة")   # clean Stanza VSO
print("REF VSO repr:", repr(ref)[:300], flush=True)
for s in ["فتح الرجل الباب", "قطع النجار الخشب", "رفع الطالب الملف"]:
    os.environ["ARABIC_POS_FUSION"] = "0"
    d0 = adr.sentence_to_diagram(s)
    os.environ["ARABIC_POS_FUSION"] = "1"
    d1 = adr.sentence_to_diagram(s)
    print("===", s, flush=True)
    print("  off:", repr(d0)[:220], flush=True)
    print("  on :", repr(d1)[:220], flush=True)
