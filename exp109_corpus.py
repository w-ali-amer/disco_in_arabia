# -*- coding: utf-8 -*-
"""exp109_corpus.py -- doc 109 (REGISTERED b9c4aff; A1 030e0f3): Tashkeela 0.3 (original) -> lexical-homograph readings,
bigram 2x2 contexts, reservoir-sampled instances for the LM arm.  Corpus NOT committed (gitignored); derived counts only.

Pass 1: complete tokens -> (form, melody) counts -> H (>= 200 complete tokens, 2nd melody >= 15%).
Pass 2: adjacent (X, Y), X != Y, both in H with top-2 melodies -> 2x2 tables; exclusion rates.
Pass 3: for contexts in any rank-4 system at n_min = 50: <= 50 instances per context, chosen by the 50 smallest
        sha256(seed 109, file, line, pos) keys (deterministic); each instance = 15 preceding undiacritised words + gold.
Run with ~/camel_mac_env (stdlib + numpy).  Writes counts_exp109.json (Rule 3).
"""
import collections, glob, hashlib, heapq, itertools, json, os, re, sys
from multiprocessing import Pool
import exp94_common as C94

ROOT = "data/tashkeela/Tashkeela-arabic-diacritized-text-utf8-0.3/texts.txt"
MARKS = set("ًٌٍَُِّْ")
VOWELS = set("ًٌٍَُِْ")         # sukun counts; shadda alone does not
LETTER = re.compile("^[ء-غف-يٱً-ْ]+$")
EXEMPT = set("اويىٱ")
EDGE = re.compile("^[^ء-يٱ]+|[^ء-يٱً-ْ]+$")
N_FORM, P2, CAP = 200, 0.15, 50


def files():
    return sorted(glob.glob(os.path.join(ROOT, "**", "*.txt"), recursive=True))


def clean(tok):
    t = EDGE.sub("", tok)
    return t if t and LETTER.match(t) else None


def analyse(tok):
    """-> (form, melody, complete)."""
    form = "".join(ch for ch in tok if ch not in MARKS)
    i = len(tok)
    while i > 0 and tok[i - 1] in MARKS:
        i -= 1
    mel = tok[:i] + ("\u0651" if "\u0651" in tok[i:] else "")         # A2: keep a final shadda (lexical)
    letters = []                                                         # (letter, marks) over the melody string
    for ch in mel:
        if ch in MARKS:
            if letters:
                letters[-1][1].add(ch)
        else:
            letters.append([ch, set()])
    ok = True
    for k, (ch, mk) in enumerate(letters[:-1]):                          # last letter exempt (its mark = ending)
        if ch in EXEMPT:
            continue
        if ch == "ل" and k >= 1 and letters[k - 1][0] in "اٱ" and k - 1 <= 2:
            continue                                                     # article lam (sun letters)
        if ch == "ل" and k == 1 and letters[0][0] == "ل":
            continue                                                     # لل = ل + ال
        if not (mk & VOWELS):
            ok = False; break
    return form, mel, ok


def lines(f):
    with open(f, encoding="utf-8", errors="ignore") as fh:
        for ln, line in enumerate(fh):
            toks = [clean(t) for t in line.split()]
            yield ln, toks


def p1(f):
    c = collections.Counter(); st = collections.Counter()
    for _, toks in lines(f):
        for t in toks:
            if t is None:
                continue
            form, mel, ok = analyse(t); st["tokens"] += 1
            if ok:
                c[(form, mel)] += 1
            else:
                st["incomplete"] += 1
    return c, st


def p2(args):
    f, top2 = args
    tab = collections.Counter(); st = collections.Counter()
    for _, toks in lines(f):
        prev = None
        for t in toks:
            cur = None
            if t is not None:
                form, mel, ok = analyse(t)
                if ok and form in top2:
                    if mel in top2[form]:
                        cur = (form, top2[form].index(mel))
                    else:
                        st["H_other_melody"] += 1
                if form in top2:
                    st["H_tokens"] += 1
            if prev and cur and prev[0] != cur[0]:
                tab[(prev[0], cur[0], prev[1], cur[1])] += 1
            prev = cur
    return tab, st


def key(f, ln, pos):
    return hashlib.sha256(("109|%s|%d|%d" % (os.path.basename(f), ln, pos)).encode()).hexdigest()


def p3(args):
    f, top2, ctxs = args
    heaps = collections.defaultdict(list)
    for ln, toks in lines(f):
        prev = None; plain = []
        for pos, t in enumerate(toks):
            cur = None; form = None
            if t is not None:
                form, mel, ok = analyse(t)
                if ok and form in top2 and mel in top2[form]:
                    cur = (form, top2[form].index(mel))
            if prev and cur and (prev[0], cur[0]) in ctxs:
                k = key(f, ln, pos); h = heaps[(prev[0], cur[0])]
                item = (k, {"left": " ".join(plain[-16:-1]) if len(plain) > 1 else "", "rx": prev[1], "ry": cur[1]})
                heapq.heappush(h, (-int(k[:15], 16), item))
                if len(h) > CAP:
                    heapq.heappop(h)
            plain.append(form if form else "")
            prev = cur
    return {c: [it for _, it in h] for c, h in heaps.items()}


def systems(tables, n_min):
    tot = collections.Counter()
    for (x, y, a, b), n in tables.items():
        tot[(x, y)] += n
    ok = {c for c, n in tot.items() if n >= n_min}
    right = collections.defaultdict(set)
    for x, y in ok:
        right[x].add(y)
    out = []
    for x1, x2 in itertools.combinations(sorted(right), 2):
        common = sorted(right[x1] & right[x2])
        for y1, y2 in itertools.combinations(common, 2):
            out.append((x1, y1, x2, y2))
    return out


def main():
    fs = files(); print("files", len(fs), flush=True)
    with Pool(8) as pool:
        c1 = collections.Counter(); st1 = collections.Counter()
        for c, st in pool.imap_unordered(p1, fs):
            c1.update(c); st1.update(st)
        forms = collections.defaultdict(collections.Counter)
        for (f, m), n in c1.items():
            forms[f][m] += n
        top2 = {}
        for f, c in forms.items():
            tot = sum(c.values())
            if tot >= N_FORM and len(c) >= 2:
                (m1, n1), (m2, n2) = c.most_common(2)
                if n2 / tot >= P2:
                    top2[f] = [m1, m2]
        print("pass1", dict(st1), "H", len(top2), flush=True)
        tab = collections.Counter(); st2 = collections.Counter()
        for t, st in pool.imap_unordered(p2, [(f, top2) for f in fs]):
            tab.update(t); st2.update(st)
        sys50, sys100 = systems(tab, 50), systems(tab, 100)
        print("pass2", dict(st2), "contexts", len({(k[0], k[1]) for k in tab}), "systems n50", len(sys50), "n100", len(sys100), flush=True)
        ctxs = {(s[0], s[1]) for s in sys50} | {(s[2], s[1]) for s in sys50} | {(s[2], s[3]) for s in sys50} | {(s[0], s[3]) for s in sys50}
        inst = collections.defaultdict(list)
        for d in pool.imap_unordered(p3, [(f, top2, ctxs) for f in fs]):
            for cx, items in d.items():
                inst[cx].extend(items)
    inst = {cx: [it for _, it in sorted(v, key=lambda z: z[0])[:CAP]] for cx, v in inst.items()}
    tables = collections.defaultdict(lambda: [[0, 0], [0, 0]])
    for (x, y, a, b), n in tab.items():
        tables[(x, y)][a][b] += n
    need = ctxs
    out = {"doc": "109 (b9c4aff; A1 030e0f3)", "stats_pass1": dict(st1), "stats_pass2": dict(st2),
           "incomplete_rate": st1["incomplete"] / max(1, st1["tokens"]),
           "H": {f: {"top2": m, "counts": [forms[f][m[0]], forms[f][m[1]]], "total": sum(forms[f].values())} for f, m in top2.items()},
           "tables": {"%s|%s" % cx: tables[cx] for cx in need},
           "systems_n50": [list(s) for s in sys50], "systems_n100": [list(s) for s in sys100],
           "instances": {"%s|%s" % cx: v for cx, v in inst.items()},
           "corpus_sha256_tar": "ef41cee7d5f9b284a5c5e2a482657969d5a6b61883368ea63cdb50a2190648a3", "n_files": len(fs),
           "provenance": C94.provenance()}
    C94.write_new_json("counts_exp109.json", out)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
