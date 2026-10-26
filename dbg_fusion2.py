# -*- coding: utf-8 -*-
import os, logging
os.environ["ARABIC_POS_FUSION"] = "1"
logging.basicConfig(level=logging.DEBUG)
import camel_test2, arabic_dep_reader as adr
for w in ["قطع", "فتح", "رفع"]:
    poses = set()
    try:
        for a in camel_test2.CAMEL_ANALYZER.analyze(w) or []:
            poses.add(a.get("pos"))
    except Exception as e:
        poses = {f"ERR:{e!r}"}
    print(f"[dbg2] CAMeL pos set for {w!r}: {poses}", flush=True)
for s in ["فتح الرجل الباب", "قطع النجار الخشب"]:
    toks, ana, struct, roles = camel_test2.analyze_arabic_sentence_with_morph(s)
    print(f"[dbg2] {s!r}: struct={struct} verb={roles.get(chr(118)+chr(101)+chr(114)+chr(98))} subj={roles.get(chr(115)+chr(117)+chr(98)+chr(106)+chr(101)+chr(99)+chr(116))} obj={roles.get(chr(111)+chr(98)+chr(106)+chr(101)+chr(99)+chr(116))} upos={[a[chr(117)+chr(112)+chr(111)+chr(115)] for a in ana]}", flush=True)
print("[dbg2] DONE", flush=True)
