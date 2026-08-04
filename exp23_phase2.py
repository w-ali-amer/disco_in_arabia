# -*- coding: utf-8 -*-
"""Exp 23 Phase 2: sense-splitting + expanded controls + sentence dose-response.

Pre-registered:
S1  under sense-split anchors, WITHIN-lexeme pairs (جميل/جميلة) stay at
    sibling proximity levels
S2  CROSS-lexeme pairs (جمل/جملة) drop to random level under split anchors
    (vs sibling-level under Phase-1 merged anchors — replicating R4b)
S3  coherent-family R1 statistics unchanged by the split (sanity)
R4a-XL expanded pattern-mate set (18 pairs, same-pattern/different-root):
    indistinguishable from random at eps=0.175/0.30
R2-sweep sentence-level sibling effect shows dose-response across
    eps in {0.10, 0.175, 0.30} (6 frames, parse once, re-evaluate per eps)
Sense probe (descriptive): sentence fidelity جمل-vs-جملة high under merged,
    drops under split.
"""
import json, time, hashlib
from difflib import SequenceMatcher
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

t0 = time.time()
RNG = np.random.default_rng(42)

FAMILIES = {
 "ك.ت.ب": ["كتاب","الكتاب","الكتب","كتب","يكتب","تكتب","اكتب","الكاتب","المكتب","مكتبة"],
 "د.ر.س": ["الدرس","الدروس","المدرسة","تدرس","درست","يدرسون","يدرسان"],
 "ل.ع.ب": ["لعب","يلعب","تلعب","اللاعب","اللعبة","الملعب","لعبة"],
 "ف.ت.ح": ["فتح","يفتح","تفتح","الفتح","مفتوح","الفاتح"],
 "ط.ب.ب": ["الطبيب","الطبيبة","الطبيبات"],
 "ق.ل.م": ["القلم","الاقلام","القلمان","قلما"],
 "ش.ر.ح": ["شرح","يشرح","تشرح","الشرح"],
 "ض.ر.ب": ["ضرب","يضرب","ضربت","الضرب"],
 "ك.س.ر": ["كسر","يكسر","مكسورة","الكسور"],
 "ط.ب.خ": ["طبخ","يطبخ","تطبخ","الطباخ"],
 "س.ف.ر": ["سافر","يسافر","تسافر","المسافر"],
 "ن.ج.ح": ["نجح","ينجح","النجاح","الناجحة"],
 "ر.ف.ع": ["رفع","يرفع","رفعت","رفعنا"],
 "ف.ه.م": ["فهم","يفهم","فهمت","مفهوم"],
 "ز.ر.ع": ["زرع","يزرع","تزرع","المزارع","زراعية"],
 "م.ر.ض": ["المريض","المرضى","الممرضة","ممرضات","المريضات"],
 "ن.ظ.ف": ["نظف","نظيف","نظيفة","تنظف"],
 "ع.ل.ج": ["عالج","يعالج","يعالجون"],
 "م.ط.ر": ["المطر","الامطار","ممطر"],
 "ه.ن.د.س": ["المهندس","المهندسة","مهندسون","المهندسون"],
}
NEG_CONTROLS = {
 "ج.م.ل": {"beauty": ["الجميل","الجميلة","جميلة","جمال"],
           "camel": ["الجمل","جمل"], "sentence": ["الجملة","جملة"]},
 "ر.ج.ل": {"man": ["الرجل","رجال","رجل"], "leg": ["ارجل","رجله"]},
 "ح.ص.ن": {"horse": ["الحصان"], "fort": ["الحصن","الحصون"]},
 "ش.ر.ط": {"police": ["الشرطي","الشرطية"], "condition": ["شرط"]},
 "ب.ر.د": {"cold": ["البرد","بارد"], "mail": ["البريدي"]},
}
PATTERN_PAIRS = [
 ("الكاتب","اللاعب"), ("الكاتب","الفاتح"), ("اللاعب","الفاتح"),
 ("المكتب","الملعب"), ("الطبيب","المريض"), ("الدرس","الضرب"),
 ("الشرح","الفتح"), ("الكتب","الدروس"), ("القلم","المطر"),
 ("اللعبة","الطبيبة"), ("يكتب","يلعب"), ("يفتح","يشرح"),
 ("يضرب","يكسر"), ("يطبخ","يزرع"), ("يرفع","يفهم"),
 ("تكتب","تدرس"), ("تفتح","تطبخ"), ("مفتوح","مفهوم")]

def h01(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

def params_for(word, anchor, eps):
    return [2.0 * h01(f"{anchor}|{i}") +
            eps * (2.0 * h01(f"{word}|{i}") - 1.0) for i in range(3)]

def rx(t):
    c, s = np.cos(np.pi * t), np.sin(np.pi * t)
    return np.array([[c, -1j * s], [-1j * s, c]])

def rz(t):
    return np.diag([np.exp(-1j * np.pi * t), np.exp(1j * np.pi * t)])

def wstate(p):
    v = (rx(p[0]) @ rz(p[1]) @ rx(p[2]))[0, :]
    return v / np.linalg.norm(v)

def fid(a, b):
    return float(abs(np.vdot(a, b)) ** 2)

def anchors(scheme):
    """word -> anchor string. merged: root; split: root::lexeme for controls."""
    amap = {}
    for r, ws in FAMILIES.items():
        for w in ws:
            amap[w] = r
    for r, lex in NEG_CONTROLS.items():
        for lx, ws in lex.items():
            for w in ws:
                amap[w] = r if scheme == "merged" else f"{r}::{lx}"
    return amap

OUT = {}
EPS = 0.175
for scheme in ("merged", "split"):
    amap = anchors(scheme)
    st = {w: wstate(params_for(w, a, EPS)) for w, a in amap.items()}
    SIB = [fid(st[a], st[b]) for r, ws in FAMILIES.items()
           for i, a in enumerate(ws) for b in ws[i + 1:]]
    WITHIN, CROSS = [], []
    for r, lex in NEG_CONTROLS.items():
        keys = list(lex)
        for lx in keys:
            ws = lex[lx]
            WITHIN += [fid(st[a], st[b]) for i, a in enumerate(ws)
                       for b in ws[i + 1:]]
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                CROSS += [fid(st[a], st[b]) for a in lex[keys[x]]
                          for b in lex[keys[y]]]
    fam_words = [w for ws in FAMILIES.values() for w in ws]
    amap_fam = {w: amap[w] for w in fam_words}
    RND = []
    while len(RND) < 400:
        a, b = RNG.choice(fam_words, 2, replace=False)
        if amap_fam[a] == amap_fam[b]:
            continue
        RND.append(fid(st[a], st[b]))
    res = {"medians": {"SIB": float(np.median(SIB)),
                       "NEG_within": float(np.median(WITHIN)),
                       "NEG_cross": float(np.median(CROSS)),
                       "RND": float(np.median(RND))},
           "S2_cross_vs_rnd_p": float(mannwhitneyu(
               CROSS, RND, alternative="greater")[1]),
           "S1_within_vs_rnd_p": float(mannwhitneyu(
               WITHIN, RND, alternative="greater")[1]) if WITHIN else None}
    OUT[f"word_{scheme}"] = res
    print(f"[23p2] {scheme}: SIB={res['medians']['SIB']:.4f} "
          f"NEGwithin={res['medians']['NEG_within']:.4f} "
          f"NEGcross={res['medians']['NEG_cross']:.4f} "
          f"RND={res['medians']['RND']:.4f} | S2 cross>rnd p="
          f"{res['S2_cross_vs_rnd_p']:.3f} S1 within>rnd p="
          f"{res['S1_within_vs_rnd_p']:.2e}", flush=True)

# R4a-XL at two eps values
for eps in (0.175, 0.30):
    amap = anchors("split")
    st = {w: wstate(params_for(w, a, eps)) for w, a in amap.items()}
    PAT = [fid(st[a], st[b]) for a, b in PATTERN_PAIRS]
    fam_words = [w for ws in FAMILIES.values() for w in ws]
    RND = []
    while len(RND) < 400:
        a, b = RNG.choice(fam_words, 2, replace=False)
        if amap[a] == amap[b] or frozenset((a, b)) in {
                frozenset(p) for p in PATTERN_PAIRS}:
            continue
        RND.append(fid(st[a], st[b]))
    y = np.r_[np.ones(len(PAT)), np.zeros(len(RND))]
    auc = float(roc_auc_score(y, np.r_[PAT, RND]))
    p = float(mannwhitneyu(PAT, RND, alternative="two-sided")[1])
    OUT[f"R4aXL_eps_{eps}"] = {"n_pat": len(PAT), "auc": auc, "p": p,
                               "median_pat": float(np.median(PAT)),
                               "median_rnd": float(np.median(RND))}
    print(f"[23p2] R4a-XL eps={eps}: PAT={np.median(PAT):.4f} (n={len(PAT)}) "
          f"vs RND={np.median(RND):.4f} | AUC={auc:.3f} p={p:.3f}", flush=True)

# ── sentence level: eps sweep + sense probe ─────────────────────────────────
print("[23p2] sentence part...", flush=True)
import exp13_arabert_comparison as exp13

DEF_NOUNS = {"ك.ت.ب": ["الكتاب","الكاتب","المكتب"],
             "د.ر.س": ["الدرس","المدرسة"],
             "ل.ع.ب": ["اللاعب","الملعب","اللعبة"],
             "ط.ب.ب": ["الطبيب","الطبيبة"],
             "م.ر.ض": ["المريض","الممرضة"],
             "ه.ن.د.س": ["المهندس","المهندسة"],
             "ق.ل.م": ["القلم"], "م.ط.ر": ["المطر"],
             "س.ف.ر": ["المسافر"], "ف.ت.ح": ["الفاتح"]}
SENSE_SUBJ = ["الرجل","الجمل","الجملة","الحصان","الحصن"]
FRAMES = [("SVO1", "{X} اكل الطعام"), ("VSO1", "اكل {X} الطعام"),
          ("SVO2", "{X} شاهد الولد"), ("VSO2", "شاهد {X} الولد"),
          ("SVO3", "{X} وجد الباب"), ("VSO3", "وجد {X} الباب")]
subjects = [(r, w) for r, ws in DEF_NOUNS.items() for w in ws] + \
           [("SENSE", w) for w in SENSE_SUBJ]
sent_list, meta = [], []
for fr_name, fr in FRAMES:
    for r, w in subjects:
        sent_list.append(fr.format(X=w))
        meta.append((fr_name, r, w))
diagrams = exp13.sentences_to_diagrams(sent_list, log_interval=999)
ansatz = exp13.make_ansatz(1, 1)
circs, cmeta = [], []
for k, d in enumerate(diagrams):
    if d is None:
        continue
    try:
        c = ansatz(exp13._remove_cups(d))
        circs.append((c, sorted(c.free_symbols, key=str)))
        cmeta.append(meta[k])
    except Exception:
        pass
print(f"[23p2] parsed+built {len(circs)}/{len(sent_list)}", flush=True)

def sent_states(eps, scheme):
    amap = anchors(scheme)
    def wgt(name):
        base = name.split("__")[0].split("_")[0]
        try:
            idx = int(name.rsplit("_", 1)[-1])
        except ValueError:
            idx = 0
        if base in amap:
            return params_for(base, amap[base], eps)[idx % 3]
        return 2.0 * h01(name)
    S = []
    for c, syms in circs:
        vals = [wgt(str(s)) for s in syms]
        amps = np.asarray(c.lambdify(*syms)(*vals).eval()).flatten()
        p = float(np.sum(np.abs(amps) ** 2))
        S.append(amps / np.sqrt(p) if p > 1e-12 else None)
    return S

for eps in (0.10, 0.175, 0.30):
    S = sent_states(eps, "split")
    SIB2, RND2 = [], []
    for fr_name, _ in FRAMES:
        idxs = [k for k in range(len(circs))
                if cmeta[k][0] == fr_name and S[k] is not None
                and cmeta[k][1] != "SENSE"]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ka, kb = idxs[a], idxs[b]
                f = fid(S[ka], S[kb])
                (SIB2 if cmeta[ka][1] == cmeta[kb][1] else RND2).append(f)
    p2 = float(mannwhitneyu(SIB2, RND2, alternative="greater")[1])
    y = np.r_[np.ones(len(SIB2)), np.zeros(len(RND2))]
    auc2 = float(roc_auc_score(y, np.r_[SIB2, RND2]))
    OUT[f"R2_eps_{eps}"] = {"n_sib": len(SIB2),
                            "median_sib": float(np.median(SIB2)),
                            "median_rnd": float(np.median(RND2)),
                            "auc": auc2, "p": p2}
    print(f"[23p2] R2 eps={eps}: SIB={np.median(SIB2):.4f} "
          f"RND={np.median(RND2):.4f} AUC={auc2:.4f} p={p2:.2e}", flush=True)

# sense probe: جمل vs جملة sentences under merged vs split
for scheme in ("merged", "split"):
    S = sent_states(EPS, scheme)
    pairs = {}
    for tgt_a, tgt_b, tag in (("الجمل", "الجملة", "camel_vs_sentence"),
                              ("الحصان", "الحصن", "horse_vs_fort")):
        fids = []
        for fr_name, _ in FRAMES:
            ka = [k for k in range(len(circs)) if cmeta[k][0] == fr_name
                  and cmeta[k][2] == tgt_a and S[k] is not None]
            kb = [k for k in range(len(circs)) if cmeta[k][0] == fr_name
                  and cmeta[k][2] == tgt_b and S[k] is not None]
            if ka and kb:
                fids.append(fid(S[ka[0]], S[kb[0]]))
        pairs[tag] = float(np.median(fids)) if fids else None
    OUT[f"sense_probe_{scheme}"] = pairs
    print(f"[23p2] sense probe ({scheme}): {pairs}", flush=True)

OUT["runtime_sec"] = round(time.time() - t0, 1)
json.dump(OUT, open("results_exp23p2.json", "w"), indent=2, ensure_ascii=False)
print(f"[23p2] DONE in {OUT['runtime_sec']}s", flush=True)
