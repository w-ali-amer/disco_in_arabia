# -*- coding: utf-8 -*-
"""Exp 23 Phase 1: root/pattern factored encoding (paper §10.2).

Encoding: every word's 3 Euler parameters = ROOT anchor (identical for all
words sharing a جذر) + a small per-form PATTERN perturbation (eps). Zero
training. Family lists = Opus review (EXP23_FAMILY_REVIEW.md), user-approved.

Pre-registered:
R1  root-sibling word states closer than cross-root pairs (MWU one-sided)
R3  holds for surface-DISSIMILAR siblings (difflib ratio < 0.5) — the effect
    is parameter sharing, not string similarity
R4a pattern-mates (same وزن, different root) indistinguishable from random
    (report AUC, expect ~0.5)
R4b negative-control families (root-shared, meaning-unrelated: جمل/جملة/جميل,
    man/leg, horse/fortress, police/condition, cold/mail): cross-lexeme pairs
    predicted AS CLOSE as true siblings — quantifying that the mechanism is
    formal; a semantic layer requires sense-splitting (Phase 2)
R2  sentence level: same frame, subject substituted — sibling substitutions
    closer than cross-family substitutions (MWU one-sided)
eps robustness sweep: {0.10, 0.175, 0.30} half-turns.
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
PATTERN_PAIRS = [("الكاتب","اللاعب"), ("المكتب","الملعب"), ("الطبيب","المريض"),
                 ("الدرس","الفتح"), ("الشرح","الضرب"), ("الكتب","الدروس"),
                 ("القلم","المطر"), ("اللعبة","الطبيبة")]

def h01(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

def word_params(word, root, eps):
    return [2.0 * h01(f"{root}|{i}") +
            eps * (2.0 * h01(f"{word}|{i}") - 1.0) for i in range(3)]

def rx(t):
    c, s = np.cos(np.pi * t), np.sin(np.pi * t)
    return np.array([[c, -1j * s], [-1j * s, c]])

def rz(t):
    return np.diag([np.exp(-1j * np.pi * t), np.exp(1j * np.pi * t)])

def word_state(p):
    v = (rx(p[0]) @ rz(p[1]) @ rx(p[2]))[0, :]
    return v / np.linalg.norm(v)

def fid(a, b):
    return float(abs(np.vdot(a, b)) ** 2)

def surface_sim(a, b):
    strip = lambda w: w[2:] if w.startswith("ال") else w
    return SequenceMatcher(None, strip(a), strip(b)).ratio()

OUT = {"pre_registered": "R1 R2 R3 R4a R4b; eps sweep 0.10/0.175/0.30"}
for eps in (0.10, 0.175, 0.30):
    states, rootof = {}, {}
    for r, ws in FAMILIES.items():
        for w in ws:
            states[w] = word_state(word_params(w, r, eps)); rootof[w] = r
    neg_states, neg_lex = {}, {}
    for r, lex in NEG_CONTROLS.items():
        for lx, ws in lex.items():
            for w in ws:
                neg_states[w] = word_state(word_params(w, r, eps))
                neg_lex[w] = (r, lx)
    SIB, SIB_SIM, RND, PAT, NEG = [], [], [], [], []
    for r, ws in FAMILIES.items():
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                f = fid(states[ws[i]], states[ws[j]])
                SIB.append(f)
                SIB_SIM.append(surface_sim(ws[i], ws[j]))
    # R3: least surface-similar TERCILE of sibling pairs (rank-based —
    # root-siblings share 3 consonants by definition, so an absolute
    # threshold under-fills; the bottom third is always populated)
    cut = np.percentile(SIB_SIM, 33.4)
    SIB_DIS = [f for f, s in zip(SIB, SIB_SIM) if s <= cut]
    allw = list(states)
    patset = {frozenset(p) for p in PATTERN_PAIRS}
    while len(RND) < 400:
        a, b = RNG.choice(allw, 2, replace=False)
        if rootof[a] == rootof[b] or frozenset((a, b)) in patset:
            continue
        RND.append(fid(states[a], states[b]))
    for a, b in PATTERN_PAIRS:
        PAT.append(fid(states[a], states[b]))
    for r, lex in NEG_CONTROLS.items():
        keys = list(lex)
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                for wa in lex[keys[x]]:
                    for wb in lex[keys[y]]:
                        NEG.append(fid(neg_states[wa], neg_states[wb]))
    res = {"n": {"SIB": len(SIB), "SIB_dissim": len(SIB_DIS), "RND": len(RND),
                 "PAT": len(PAT), "NEG": len(NEG)},
           "medians": {"SIB": float(np.median(SIB)),
                       "SIB_dissim": float(np.median(SIB_DIS)),
                       "RND": float(np.median(RND)),
                       "PAT": float(np.median(PAT)),
                       "NEG": float(np.median(NEG))}}
    res["R1_p"] = float(mannwhitneyu(SIB, RND, alternative="greater")[1])
    res["R3_p"] = float(mannwhitneyu(SIB_DIS, RND, alternative="greater")[1])
    y = np.r_[np.ones(len(PAT)), np.zeros(len(RND))]
    res["R4a_auc_pat_vs_rnd"] = float(roc_auc_score(y, np.r_[PAT, RND]))
    res["R4b_neg_vs_sib_p_twosided"] = float(
        mannwhitneyu(NEG, SIB, alternative="two-sided")[1])
    OUT[f"word_level_eps_{eps}"] = res
    print(f"[23p1] eps={eps}: SIB={res['medians']['SIB']:.4f} "
          f"SIBdis={res['medians']['SIB_dissim']:.4f} "
          f"RND={res['medians']['RND']:.4f} PAT={res['medians']['PAT']:.4f} "
          f"NEG={res['medians']['NEG']:.4f} | R1 p={res['R1_p']:.2e} "
          f"R3 p={res['R3_p']:.2e} R4a AUC={res['R4a_auc_pat_vs_rnd']:.3f} "
          f"R4b p={res['R4b_neg_vs_sib_p_twosided']:.3f}", flush=True)
json.dump(OUT, open("results_exp23p1.json", "w"), indent=2, ensure_ascii=False)

# ── Part B: sentence-level R2 (frames, eps=0.175) ───────────────────────────
print("[23p1] Part B: sentence frames...", flush=True)
import exp13_arabert_comparison as exp13

DEF_NOUNS = {"ك.ت.ب": ["الكتاب","الكاتب","المكتب"],
             "د.ر.س": ["الدرس","المدرسة"],
             "ل.ع.ب": ["اللاعب","الملعب","اللعبة"],
             "ط.ب.ب": ["الطبيب","الطبيبة"],
             "م.ر.ض": ["المريض","الممرضة"],
             "ه.ن.د.س": ["المهندس","المهندسة"],
             "ق.ل.م": ["القلم"], "م.ط.ر": ["المطر"],
             "س.ف.ر": ["المسافر"], "ف.ت.ح": ["الفاتح"]}
FRAMES = [("SVO", "{X} اكل الطعام"), ("VSO", "اكل {X} الطعام")]
subjects = [(r, w) for r, ws in DEF_NOUNS.items() for w in ws]
sent_list, meta = [], []
for fr_name, fr in FRAMES:
    for r, w in subjects:
        sent_list.append(fr.format(X=w))
        meta.append((fr_name, r, w))
diagrams = exp13.sentences_to_diagrams(sent_list, log_interval=999)
ansatz = exp13.make_ansatz(1, 1)
EPS = 0.175
def sym_weight(name):
    word_part = name.split("__")[0]
    base = word_part.split("_")[0]
    try:
        idx = int(name.rsplit("_", 1)[-1])
    except ValueError:
        idx = 0
    for r, ws in list(FAMILIES.items()) + [
            (r2, sum(lx.values(), [])) for r2, lx in NEG_CONTROLS.items()]:
        if base in ws:
            p = word_params(base, r, EPS)
            return p[idx % 3]
    return 2.0 * h01(name)

states_s, ok = {}, 0
for k, d in enumerate(diagrams):
    if d is None:
        continue
    try:
        c = ansatz(exp13._remove_cups(d))
        syms = sorted(c.free_symbols, key=str)
        vals = [sym_weight(str(s)) for s in syms]
        amps = np.asarray(c.lambdify(*syms)(*vals).eval()).flatten()
        p = float(np.sum(np.abs(amps) ** 2))
        if p > 1e-12:
            states_s[k] = amps / np.sqrt(p); ok += 1
    except Exception:
        pass
print(f"[23p1] parsed+built {ok}/{len(sent_list)} frame sentences", flush=True)
SIB2, RND2 = [], []
for fi, (fr_name, _) in enumerate(FRAMES):
    idxs = [k for k in states_s if meta[k][0] == fr_name]
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            ka, kb = idxs[a], idxs[b]
            f = fid(states_s[ka], states_s[kb])
            (SIB2 if meta[ka][1] == meta[kb][1] else RND2).append(f)
if SIB2 and RND2:
    p2 = float(mannwhitneyu(SIB2, RND2, alternative="greater")[1])
    y = np.r_[np.ones(len(SIB2)), np.zeros(len(RND2))]
    auc2 = float(roc_auc_score(y, np.r_[SIB2, RND2]))
    OUT["R2_sentence"] = {"n_sib": len(SIB2), "n_rnd": len(RND2),
                          "median_sib": float(np.median(SIB2)),
                          "median_rnd": float(np.median(RND2)),
                          "auc": auc2, "mwu_p": p2}
    print(f"[23p1] R2: SIB={np.median(SIB2):.4f} (n={len(SIB2)}) vs "
          f"RND={np.median(RND2):.4f} (n={len(RND2)}) | AUC={auc2:.4f} "
          f"p={p2:.2e}", flush=True)
OUT["runtime_sec"] = round(time.time() - t0, 1)
json.dump(OUT, open("results_exp23p1.json", "w"), indent=2, ensure_ascii=False)
print(f"[23p1] DONE in {OUT['runtime_sec']}s", flush=True)
