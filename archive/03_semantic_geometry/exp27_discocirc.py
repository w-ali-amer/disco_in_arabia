# -*- coding: utf-8 -*-
"""Exp 27: DisCoCirc referent-wire skeleton. Design doc: 16_exp27.

Text = persistent referent wires + sense registers; sentences = verb-
directional filtered gates; readout = MAP interpretation + entropy per
referent. Pre-registered D1-D6 (see design doc)."""
import json, hashlib
import numpy as np

def h01(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

def rx(t):
    c, s = np.cos(np.pi * t), np.sin(np.pi * t)
    return np.array([[c, -1j * s], [-1j * s, c]])

def rz(t):
    return np.diag([np.exp(-1j * np.pi * t), np.exp(1j * np.pi * t)])

def ry(t):
    c, s = np.cos(np.pi * t), np.sin(np.pi * t)
    return np.array([[c, -s], [s, c]], dtype=complex)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def euler(p):
    return rx(p[0]) @ rz(p[1]) @ rx(p[2])

def apply1(st, U, q):
    st = np.tensordot(U, st, ([1], [q]))
    return np.moveaxis(st, 0, q)

def apply2(st, U4, qa, qb):
    T = U4.reshape(2, 2, 2, 2)
    st = np.tensordot(T, st, ([2, 3], [qa, qb]))
    return np.moveaxis(st, [0, 1], [qa, qb])

def blockdiag(UA, UB):
    M = np.zeros((4, 4), dtype=complex)
    M[:2, :2] = UA
    M[2:, 2:] = UB
    return M

def crz(t):
    return np.diag([1, 1, np.exp(-1j * np.pi * t),
                    np.exp(1j * np.pi * t)]).astype(complex)

def entropy_rho(rho):
    ev = np.clip(np.real(np.linalg.eigvalsh((rho + rho.conj().T) / 2)),
                 1e-15, None)
    ev = ev / ev.sum()
    return float(-np.sum(ev * np.log2(ev)))

def reg_entropy(psi, q):
    n = psi.ndim
    p = np.moveaxis(psi, q, n - 1).reshape(-1, 2)
    rho = p.T @ p.conj()
    rho = rho / np.trace(rho)
    return entropy_rho(rho), float(np.real(rho[1, 1]))

def anchor(root, lex):
    return [2 * h01(f"{root}::{lex}|{i}") for i in range(3)]

def verb_axis_strength(verb_root):
    if verb_root.startswith("INTRO"):
        return 0.0, 0.0          # introduction is NOT evidence: identity gate
    return 2 * h01(f"{verb_root}|axis"), 0.30 + 0.15 * h01(f"{verb_root}|str")

class Text:
    """State over [reg_1..reg_R, wire_1..wire_R]; sequential s-ancillas."""
    def __init__(self, referents):
        self.R = len(referents)
        self.psi = np.zeros((2,) * (2 * self.R), dtype=complex)
        self.psi[(0,) * (2 * self.R)] = 1.0
        self.p_post = 1.0
        for i, (root, lexA, lexB) in enumerate(referents):
            self.psi = apply1(self.psi, ry(0.25), i)          # equal prior
            U = blockdiag(euler(anchor(root, lexA)),
                          euler(anchor(root, lexB)))
            self.psi = apply2(self.psi, U, i, self.R + i)     # ctrl prep

    def sentence(self, ref_idx, verb_root):
        ax, g = verb_axis_strength(verb_root)
        w = self.R + ref_idx
        self.psi = apply1(self.psi, rx(ax), w)
        self.psi = np.tensordot(self.psi,
                                np.array([1, 1], dtype=complex) / np.sqrt(2), 0)
        s = self.psi.ndim - 1
        self.psi = apply2(self.psi, crz(g), w, s)
        self.psi = apply1(self.psi, H, s)
        self.psi = np.take(self.psi, 0, axis=s)
        self.psi = apply1(self.psi, rx(-ax), w)
        n2 = float(np.sum(np.abs(self.psi) ** 2))
        self.psi = self.psi / np.sqrt(n2)
        self.p_post *= n2
        return n2

    def readout(self, reg_idx):
        return reg_entropy(self.psi, reg_idx)

REFS = {"رجل(man/leg)": ("ر.ج.ل", "man", "leg"),
        "جمل(camel/sentence)": ("ج.م.ل", "camel", "sentence"),
        "جمل(camel/beauty)": ("ج.م.ل", "camel", "beauty"),
        "عين(eye/spring)": ("ع.#.ن", "eye", "spring")}
VERB_POOL = ["ك.ت.ب", "د.ر.س", "ل.ع.ب", "ف.ت.ح", "ش.ر.ح", "ض.ر.ب", "ك.س.ر",
             "ط.ب.خ", "س.ف.ر", "ن.ج.ح", "ر.ف.ع", "ف.ه.م", "ز.ر.ع", "ن.ظ.ف",
             "ع.ل.ج", "م.ط.ر", "ق.ب.ل", "ح.م.ل", "خ.ر.ج", "ر.ك.ب", "ق.ط.ع",
             "ن.ش.ر", "س.ع.#", "ح.ل.ل", "ج.ه.د", "ص.ح.ح", "غ.د.ر", "ب.د.ع",
             "خ.ل.ص", "ك.ش.ف"]

def selectivity_spectrum(ref):
    """Per verb: (P(sense=B | S1+S2_v), pass rate). Selection step (limitation #1)."""
    out = {}
    for v in VERB_POOL:
        t = Text([REFS[ref]])
        t.sentence(0, "INTRO")           # S1: neutral introducing event
        e1, pb1 = t.readout(0)
        n2 = t.sentence(0, v)
        e2, pb2 = t.readout(0)
        out[v] = {"dPB": pb2 - pb1, "pass": n2, "S_after": e2}
    return out

OUT = {"selection_step": {}}
texts_json = []
results = []
for ref in REFS:
    spec = selectivity_spectrum(ref)
    # disambig = strongest purifier from equal prior; neutral = weakest
    ranked = sorted(spec.items(), key=lambda kv: kv[1]["S_after"])
    dis_v, dis_info = ranked[0]
    neu_v, neu_info = ranked[-1]
    OUT["selection_step"][ref] = {
        "disambig_verb": dis_v, "dPB": dis_info["dPB"],
        "pass_dis": dis_info["pass"],
        "neutral_verb": neu_v, "dPB_neu": neu_info["dPB"],
        "pass_neu": neu_info["pass"]}
    row = {"referent": ref}
    t = Text([REFS[ref]]); t.sentence(0, "INTRO")
    row["S_after_S1"], row["PB_after_S1"] = t.readout(0)
    td = Text([REFS[ref]]); td.sentence(0, "INTRO"); td.sentence(0, dis_v)
    row["S_dis"], row["PB_dis"] = td.readout(0)
    row["p_post_dis"] = td.p_post
    tn = Text([REFS[ref]]); tn.sentence(0, "INTRO"); tn.sentence(0, neu_v)
    row["S_neu"], row["PB_neu"] = tn.readout(0)
    row["p_post_neu"] = tn.p_post
    # D5 order swap
    ts = Text([REFS[ref]]); ts.sentence(0, dis_v); ts.sentence(0, "INTRO")
    row["S_dis_swapped"], _ = ts.readout(0)
    # D3: MAP sense after disambiguation = direction the verb pushed
    row["map_flip_ok"] = bool((row["PB_dis"] > 0.5) == (dis_info["dPB"] > 0))
    results.append(row)
    sel = "B" if dis_info["dPB"] > 0 else "A"
    texts_json.append({
        "referent": ref, "senses": REFS[ref][1:],
        "S1": f"دخل {ref.split('(')[0]}",
        "S2_disambig": f"[verb {dis_v} selecting sense {sel}]",
        "S2_neutral": f"[verb {neu_v}]",
        "note": "surfaces are documentation; gates run on anchors; "
                "native review pending"})
    print(f"[27] {ref}: S1={row['S_after_S1']:.3f} -> "
          f"dis={row['S_dis']:.3f} (v={dis_v}, dPB={dis_info['dPB']:+.3f}, "
          f"pass={dis_info['pass']:.3f}) | neu={row['S_neu']:.3f} "
          f"(v={neu_v}, pass={neu_info['pass']:.3f}) | "
          f"swap={row['S_dis_swapped']:.3f} | MAPflip={row['map_flip_ok']}",
          flush=True)

# D1/D2 verdicts (S1 entropy is exactly 1.0 now: equal prior, identity intro)
d1 = all(r["S_dis"] < 0.7 for r in results)
d2 = all(r["S_neu"] > 0.9 for r in results)
d3 = all(r["map_flip_ok"] for r in results)
print(f"[27] D1 purification (all refs): {d1}", flush=True)
print(f"[27] D2 neutral-control (all refs): {d2}", flush=True)
print(f"[27] D3 MAP flips with verb direction: {d3}", flush=True)

# D4: 3-sentence trajectory + stacking
t3 = Text([REFS["رجل(man/leg)"]])
traj, pps = [], []
t3.sentence(0, "INTRO"); traj.append(t3.readout(0)[0]); pps.append(t3.p_post)
dv = OUT["selection_step"]["رجل(man/leg)"]["disambig_verb"]
nv = OUT["selection_step"]["رجل(man/leg)"]["neutral_verb"]
t3.sentence(0, nv); traj.append(t3.readout(0)[0]); pps.append(t3.p_post)
t3.sentence(0, dv); traj.append(t3.readout(0)[0]); pps.append(t3.p_post)
OUT["D4_trajectory"] = {"entropy": traj, "p_post_cum": pps}
print(f"[27] D4 3-sentence trajectory: S={['%.3f' % x for x in traj]} "
      f"p_post={['%.3f' % x for x in pps]}", flush=True)

# D6: two referents, one text; disambiguate only referent 0
t2 = Text([REFS["رجل(man/leg)"], REFS["عين(eye/spring)"]])
t2.sentence(0, "INTRO"); t2.sentence(1, "INTRO2")
e_r0_before, e_r1_before = t2.readout(0)[0], t2.readout(1)[0]
t2.sentence(0, dv)
e_r0_after, e_r1_after = t2.readout(0)[0], t2.readout(1)[0]
OUT["D6_two_referents"] = {
    "before": [e_r0_before, e_r1_before],
    "after_disambig_ref0": [e_r0_after, e_r1_after],
    "p_post": t2.p_post}
print(f"[27] D6: ref0 {e_r0_before:.3f}->{e_r0_after:.3f} "
      f"(disambiguated), ref1 {e_r1_before:.3f}->{e_r1_after:.3f} "
      f"(untouched, should be ~unchanged); joint p_post={t2.p_post:.3f}",
      flush=True)

# D7: evidence-vs-prior conflict — prior 0.75 toward sense A, verb selects B
d7_rows = []
for ref in REFS:
    dis_v = OUT["selection_step"][ref]["disambig_verb"]
    spec_dir = OUT["selection_step"][ref]["dPB"]
    t = Text([REFS[ref]])
    # re-prepare register with skewed prior AGAINST the verb's direction
    t.psi = np.zeros_like(t.psi); t.psi[(0,) * t.psi.ndim] = 1.0
    p_target = 0.25 if spec_dir > 0 else 0.75   # prior favors the OTHER sense
    t.psi = apply1(t.psi, ry(np.arcsin(np.sqrt(p_target)) / np.pi), 0)
    U = blockdiag(euler(anchor(*REFS[ref][:1], ) if False else anchor(REFS[ref][0], REFS[ref][1])),
                  euler(anchor(REFS[ref][0], REFS[ref][2])))
    t.psi = apply2(t.psi, U, 0, 1)
    _, pb0 = t.readout(0)
    t.sentence(0, dis_v)
    _, pb1 = t.readout(0)
    flipped = bool((pb1 > 0.5) == (spec_dir > 0))
    d7_rows.append({"referent": ref, "PB_prior": pb0, "PB_after": pb1,
                    "evidence_beats_prior": flipped})
    print(f"[27] D7 {ref}: PB {pb0:.3f} -> {pb1:.3f} "
          f"(verb dir {'+' if spec_dir>0 else '-'}) "
          f"evidence beats prior: {flipped}", flush=True)
OUT["D7_prior_conflict"] = d7_rows

# D7b: evidence ACCUMULATION across sentences vs a hostile 0.75 prior
d7b = []
for ref in REFS:
    dis_v = OUT["selection_step"][ref]["disambig_verb"]
    spec_dir = OUT["selection_step"][ref]["dPB"]
    t = Text([REFS[ref]])
    t.psi = np.zeros_like(t.psi); t.psi[(0,) * t.psi.ndim] = 1.0
    p_target = 0.25 if spec_dir > 0 else 0.75
    t.psi = apply1(t.psi, ry(np.arcsin(np.sqrt(p_target)) / np.pi), 0)
    U = blockdiag(euler(anchor(REFS[ref][0], REFS[ref][1])),
                  euler(anchor(REFS[ref][0], REFS[ref][2])))
    t.psi = apply2(t.psi, U, 0, 1)
    traj = [t.readout(0)[1]]
    cross = None
    for n in range(1, 6):
        t.sentence(0, dis_v)
        pb = t.readout(0)[1]
        traj.append(pb)
        if cross is None and ((pb > 0.5) == (spec_dir > 0)):
            cross = n
    d7b.append({"referent": ref, "PB_trajectory": traj,
                "crossed_at_sentence": cross, "p_post_total": t.p_post})
    print(f"[27] D7b {ref}: PB " +
          " -> ".join(f"{x:.3f}" for x in traj) +
          f" | crossed at n={cross}, cum p_post={t.p_post:.3f}", flush=True)
print(f"[27] D7b: evidence accumulation beats prior in "
      f"{sum(1 for r in d7b if r['crossed_at_sentence'])}/4 referents",
      flush=True)
OUT["D7b_accumulation"] = d7b

# D7c: fresh token wire per mention, re-grounded from the sense register
# (the DisCoCirc-correct architecture: register = identity thread, each
# mention re-accesses the lexicon). Does evidence now accumulate?
d7c = []
for ref in REFS:
    dis_v = OUT["selection_step"][ref]["disambig_verb"]
    spec_dir = OUT["selection_step"][ref]["dPB"]
    ax, g = verb_axis_strength(dis_v)
    p_target = 0.25 if spec_dir > 0 else 0.75
    psi = np.array([1, 0], dtype=complex).reshape(2)   # register only
    psi = apply1(psi.reshape((2,)), ry(np.arcsin(np.sqrt(p_target)) / np.pi), 0)
    p_post = 1.0
    traj = [float(np.abs(psi[1]) ** 2)]
    cross = None
    U = blockdiag(euler(anchor(REFS[ref][0], REFS[ref][1])),
                  euler(anchor(REFS[ref][0], REFS[ref][2])))
    for n in range(1, 6):
        # fresh token wire, controlled-prep from register
        psi = np.tensordot(psi, np.array([1, 0], dtype=complex), 0)
        w = psi.ndim - 1
        psi = apply2(psi, U, 0, w)
        # sentence gate on the fresh wire
        psi = apply1(psi, rx(ax), w)
        psi = np.tensordot(psi, np.array([1, 1], dtype=complex) / np.sqrt(2), 0)
        s = psi.ndim - 1
        psi = apply2(psi, crz(g), w, s)
        psi = apply1(psi, H, s)
        psi = np.take(psi, 0, axis=s)
        n2 = float(np.sum(np.abs(psi) ** 2))
        psi = psi / np.sqrt(n2)
        p_post *= n2
        rho = np.moveaxis(psi, 0, psi.ndim - 1).reshape(-1, 2)
        rho = rho.T @ rho.conj()
        pb = float(np.real(rho[1, 1] / np.trace(rho)))
        traj.append(pb)
        if cross is None and ((pb > 0.5) == (spec_dir > 0)):
            cross = n
    d7c.append({"referent": ref, "PB_trajectory": traj,
                "crossed_at": cross, "p_post_total": p_post})
    print(f"[27] D7c {ref}: PB " +
          " -> ".join(f"{x:.3f}" for x in traj) +
          f" | crossed at n={cross}, cum p_post={p_post:.3f}", flush=True)
print(f"[27] D7c fresh-wire accumulation beats prior in "
      f"{sum(1 for r in d7c if r['crossed_at'])}/4", flush=True)
OUT["D7c_fresh_wire"] = d7c
print(f"[27] D7 overall: {sum(r['evidence_beats_prior'] for r in d7_rows)}/4",
      flush=True)

OUT["per_referent"] = results
json.dump(OUT, open("results_exp27.json", "w"), indent=2, ensure_ascii=False)
json.dump(texts_json, open("texts_exp27.json", "w"), indent=2,
          ensure_ascii=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
labels = [r["referent"].split("(")[0] + "\n" + r["referent"].split("(")[1][:-1]
          for r in results]
x = np.arange(len(results))
a1.bar(x - 0.25, [r["S_after_S1"] for r in results], 0.25, label="after S1")
a1.bar(x, [r["S_neu"] for r in results], 0.25, label="+neutral S2")
a1.bar(x + 0.25, [r["S_dis"] for r in results], 0.25, label="+disambig S2")
a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=7)
a1.set_ylabel("sense-register entropy (bits)")
a1.set_title("D1/D2: reading further purifies (or doesn't)")
a1.legend(fontsize=8)
a2.plot([1, 2, 3], OUT["D4_trajectory"]["entropy"], "o-")
a2.set_xticks([1, 2, 3])
a2.set_xticklabels(["S1 intro", "S2 neutral", "S3 disambig"])
a2.set_ylabel("entropy (bits)")
a2.set_title("D4: entropy trajectory across a 3-sentence text")
fig.suptitle("Exp27 — DisCoCirc referent-wire skeleton: "
             "disambiguation as purification across sentences")
fig.tight_layout()
fig.savefig("fig_exp27.png", dpi=200, bbox_inches="tight")
print("[27] DONE — results_exp27.json, texts_exp27.json, fig_exp27.png",
      flush=True)
