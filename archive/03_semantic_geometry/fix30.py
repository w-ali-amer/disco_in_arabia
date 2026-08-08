import io
src = io.open("exp30_corpus_verbs.py", encoding="utf-8").read()
old = """VERBS = ["قرأ", "حمل", "فتح", "كتب"]
subj = defaultdict(list)
for split in data_all.values():
    for d in split:
        t = d["sentence"].split()
        if len(t) >= 3:
            if t[0] in VERBS:
                subj[t[0]].append(t[1])
            elif t[1] in VERBS:
                subj[t[1]].append(t[0])"""
new = """VERBS = ["قرأ", "حمل", "فتح", "كتب"]
def hnorm(w):
    return (w.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا"))
VN = {hnorm(v): v for v in VERBS}
subj = defaultdict(list)
for split in data_all.values():
    for d in split:
        t = d["sentence"].split()
        if len(t) >= 3:
            if hnorm(t[0]) in VN:
                subj[VN[hnorm(t[0])]].append(t[1])
            elif hnorm(t[1]) in VN:
                subj[VN[hnorm(t[1])]].append(t[0])"""
assert src.count(old) == 1
src = src.replace(old, new)
old2 = """def vec(wd):
    v = exp13._aravec_vec(wd)
    return None if v is None else np.asarray(v, float)"""
new2 = """def vec(wd):
    cands = [wd]
    if wd.startswith("ال"):
        cands.append(wd[2:])
    cands += [c.replace("ة", "ه").replace("ى", "ي") for c in list(cands)]
    for c in dict.fromkeys(cands):
        v = exp13._aravec_vec(c)
        if v is not None:
            return np.asarray(v, float)
    return None"""
assert src.count(old2) == 1
src = src.replace(old2, new2)
old3 = """pref = {}
for v in VERBS:
    vs = [vec(w) for w in subj[v]]
    vs = [x for x in vs if x is not None]
    m = np.mean(vs, axis=0)
    pref[v] = enc(m)"""
new3 = """pref = {}
for v in VERBS:
    vs = [vec(w) for w in subj[v]]
    vs = [x for x in vs if x is not None]
    if len(vs) < 3:
        print(f"[30] {v}: only {len(vs)} attested-subject vectors — skipped",
              flush=True)
        continue
    pref[v] = enc(np.mean(vs, axis=0))
VERBS = [v for v in VERBS if v in pref]"""
assert src.count(old3) == 1
src = src.replace(old3, new3)
io.open("exp30_corpus_verbs.py", "w", encoding="utf-8").write(src)
print("PATCHED")
