# -*- coding: utf-8 -*-
"""exp138_run.py -- doc 138 (REGISTERED 570c0c3): Arabic lossless-compression benchmark.  Run with qnlp_mac_env (pyarrow).
  prep     payloads (FLORES-200 A1/A4), Wikipedia training samples, G-DISJOINT, G-COUNT, dictionaries trained twice
  a        short-payload dictionaries: zstd -19 / brotli -q 11 -w 24 x {none, D_ar, D_en} x {16 KB, 112 KB}, every payload alone, twice
  b        Tashkeela diacritized vs stripped, whole-file, six compressors
  c        FLORES devtest ar vs en whole-file context
  combine  gates, predictions -> results_exp138.json (Rule 3)
All compression goes through the CLI binaries over stdin/stdout pipes.
"""
import hashlib, json, os, random, re, subprocess, sys, time, unicodedata
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "exp138"); WORK = os.path.join(DATA, "work"); LOGS = os.path.join(ROOT, "logs")
OUT = os.path.join(ROOT, "results_exp138.json")
FLORES = os.path.join(DATA, "flores", "flores200_dataset")
TASH = os.path.join(ROOT, "data", "tashkeela", "Tashkeela-arabic-diacritized-text-utf8-0.3", "texts.txt")
LANGS = {"ar": "arb_Arab", "en": "eng_Latn"}; WIKI = {"ar": "wiki_ar_00004.parquet", "en": "wiki_en_00028.parquet"}
DICT_SIZES = {"16k": 16384, "112k": 112640}; N_TRAIN = 20000; SEED = 13800; WIN = 64
MARKS = set(range(0x064B, 0x0653)) | {0x0670, 0x0640}
POOL = 9


def part_path(name):
    return os.path.join(LOGS, "exp138_%s.json" % name)


def write_part(name, obj):
    os.makedirs(LOGS, exist_ok=True)
    with open(part_path(name), "x") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def nfc(s):
    return unicodedata.normalize("NFC", s)


def norm(s):
    return re.sub(r"\s+", " ", nfc(s)).strip()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def run(cmd, data):
    return subprocess.run(cmd, input=data, capture_output=True, check=True).stdout


def flores(lang, split):
    with open(os.path.join(FLORES, split, "%s.%s" % (LANGS[lang], split)), encoding="utf-8") as f:
        return [nfc(l.rstrip("\n")) for l in f]


def payload_dir(lang, size):
    return os.path.join(WORK, "payloads", "%s_%s" % (lang, size))


def dict_path(lang, size, rep=1):
    return os.path.join(WORK, "dicts", "D_%s_%s%s" % (lang, size, "" if rep == 1 else "_rep%d" % rep))


def versions():
    v = {}
    for name, cmd in (("zstd", ["zstd", "--version"]), ("brotli", ["brotli", "--version"]), ("xz", ["xz", "--version"]), ("gzip", ["gzip", "--version"])):
        r = subprocess.run(cmd, capture_output=True, text=True); v[name] = (r.stdout + r.stderr).splitlines()[0]
    r = subprocess.run(["bzip2", "--help"], capture_output=True, text=True); v["bzip2"] = (r.stdout + r.stderr).splitlines()[0]
    return v


def prep():
    import pyarrow.parquet as pq
    out = {"versions": versions(), "payloads": {}, "train": {}, "dicts": {}}
    sents = {l: flores(l, "dev") + flores(l, "devtest") for l in LANGS}
    for l, ss in sents.items():
        for size, items in (("A1", ss), ("A4", ["\n".join(ss[i:i + 4]) for i in range(0, 4 * (len(ss) // 4), 4)])):
            d = payload_dir(l, size); os.makedirs(d)
            for i, s in enumerate(items):
                with open(os.path.join(d, "%04d" % i), "wb") as f:
                    f.write(s.encode("utf-8"))
            raw = sum(len(s.encode("utf-8")) for s in items)
            out["payloads"]["%s_%s" % (l, size)] = {"n": len(items), "raw_bytes": raw, "code_points": sum(len(s) for s in items)}
    samples = {}
    for l, fn in WIKI.items():
        cands = []
        for t in pq.read_table(os.path.join(DATA, fn), columns=["text"]).column("text").to_pylist():
            for line in nfc(t).split("\n"):
                line = line.strip()
                if len(line) >= 40:
                    cands.append(line)
        pick = random.Random(SEED).sample(cands, N_TRAIN); samples[l] = pick
        d = os.path.join(WORK, "train", l); os.makedirs(d)
        for i, s in enumerate(pick):
            with open(os.path.join(d, "%05d" % i), "wb") as f:
                f.write(s.encode("utf-8"))
        out["train"][l] = {"candidates": len(cands), "n": N_TRAIN, "bytes": sum(len(s.encode("utf-8")) for s in pick)}
    # G-DISJOINT: exact sentence (substring) and any 64-code-point window, every FLORES sentence vs both samples
    dis = {}
    for sl, pick in samples.items():
        big = "\n".join(norm(s) for s in pick)
        wins = {}
        for fl, ss in sents.items():
            for s in ss:
                s = norm(s)
                for k in range(len(s) - WIN + 1):
                    wins[s[k:k + WIN]] = fl
        exact = {fl: sum(norm(s) in big for s in ss) for fl, ss in sents.items()}
        hits = set()
        for k in range(len(big) - WIN + 1):
            w = big[k:k + WIN]
            if w in wins:
                hits.add(w)
        dis[sl] = {"exact_hits": exact, "window_hits": len(hits), "examples": sorted(hits)[:5]}
    out["disjoint"] = dis
    out["G_DISJOINT"] = all(sum(v["exact_hits"].values()) == 0 and v["window_hits"] == 0 for v in dis.values())
    os.makedirs(os.path.join(WORK, "dicts"))
    for l in LANGS:
        for size, mx in DICT_SIZES.items():
            for rep in (1, 2):
                cmd = ["zstd", "--train", "-q", "-r", os.path.join(WORK, "train", l), "--maxdict=%d" % mx, "-o", dict_path(l, size, rep)]
                subprocess.run(cmd, check=True, capture_output=True)
            b1, b2 = (open(dict_path(l, size, r), "rb").read() for r in (1, 2))
            out["dicts"]["%s_%s" % (l, size)] = {"bytes": len(b1), "sha256": sha(b1), "rep_identical": b1 == b2, "cmd": " ".join(cmd[:-2])}
    out["G_COUNT_flores"] = all(len(ss) == 2009 for ss in sents.values()) and all(out["payloads"]["%s_A4" % l]["n"] == 502 for l in LANGS)
    write_part("prep", out); print(json.dumps({k: out[k] for k in ("payloads", "train", "disjoint", "G_DISJOINT", "dicts", "G_COUNT_flores")}, indent=1, ensure_ascii=False))


def codec(comp, dic):
    if comp == "zstd19":
        c, d = ["zstd", "-19", "-q", "-c"], ["zstd", "-d", "-q", "-c"]
    else:
        c, d = ["brotli", "-q", "11", "-w", "24", "-c"], ["brotli", "-d", "-c"]
    if dic:
        c, d = c + ["-D", dic], d + ["-D", dic]
    return c, d


CONDS = [(c, dl, ds) for c in ("zstd19", "brotli11") for dl, ds in [(None, None)] + [(l, s) for l in LANGS for s in DICT_SIZES]]


def a_job(job):
    lang, size, comp, dl, ds = job
    dic = dict_path(dl, ds) if dl else None; c, d = codec(comp, dic)
    pd = payload_dir(lang, size); tot = 0; per = []; ok = True
    for fn in sorted(os.listdir(pd)):
        raw = open(os.path.join(pd, fn), "rb").read(); z = run(c, raw)
        ok &= run(d, z) == raw; tot += len(z); per.append(len(z))
    return job, {"bytes": tot, "roundtrip": ok, "per_sha256": sha(json.dumps(per).encode())}


def a():
    jobs = [(l, s, c, dl, ds) for l in LANGS for s in ("A1", "A4") for c, dl, ds in CONDS]
    out = {"runs": [], "t0": time.time()}
    with Pool(POOL) as pool:
        for rep in (1, 2):
            res = {}
            for job, r in pool.imap_unordered(a_job, jobs):
                res["|".join(str(x) for x in job)] = r; print(rep, job, r["bytes"], r["roundtrip"], flush=True)
            out["runs"].append(res)
    out["wall_s"] = time.time() - out.pop("t0"); write_part("a", out)


WHOLE = {"gzip9": (["gzip", "-9", "-n", "-c"], ["gzip", "-d", "-c"]), "bzip2_9": (["bzip2", "-9", "-c"], ["bzip2", "-d", "-c"]),
         "xz9e": (["xz", "-9e", "-T1", "-c"], ["xz", "-d", "-c"]), "zstd19_long27": (["zstd", "-19", "--long=27", "-q", "-c"], ["zstd", "-d", "--long=27", "-q", "-c"]),
         "zstd22": (["zstd", "--ultra", "-22", "-q", "-c"], ["zstd", "-d", "-q", "-c"]), "brotli11": (["brotli", "-q", "11", "-w", "24", "-c"], ["brotli", "-d", "-c"])}


def whole_job(item):
    key, name, raw = item; c, d = WHOLE[name]; z = run(c, raw)
    return key, name, {"bytes": len(z), "roundtrip": run(d, z) == raw}


def whole(texts):
    """texts: {key: str}.  Returns {key: {compressor: {...}}} plus raw counts."""
    raws = {k: t.encode("utf-8") for k, t in texts.items()}
    res = {k: {"raw_bytes": len(r), "code_points": len(texts[k])} for k, r in raws.items()}
    with Pool(min(POOL, len(raws) * len(WHOLE))) as pool:
        for k, name, r in pool.imap_unordered(whole_job, [(k, n, r) for k, r in raws.items() for n in WHOLE]):
            res[k][name] = r; print(k, name, r, flush=True)
    return res


def b():
    files = sorted(os.listdir(TASH), key=lambda fn: sha(fn.encode("utf-8")))[:10]
    diac = "\n".join(nfc(open(os.path.join(TASH, fn), encoding="utf-8").read()) for fn in files)
    counts = {}
    for ch in diac:
        if ord(ch) in MARKS:
            counts["U+%04X" % ord(ch)] = counts.get("U+%04X" % ord(ch), 0) + 1
    strip = "".join(ch for ch in diac if ord(ch) not in MARKS)
    marks = sum(counts.values()); letters = sum(0x0621 <= ord(ch) <= 0x064A for ch in strip)
    res = whole({"diacritized": diac, "stripped": strip})
    out = {"files": files, "mark_counts": counts, "marks": marks, "base_letters": letters,
           "G_COUNT_marks": marks == len(diac) - len(strip), "whole": res}
    write_part("b", out); print(json.dumps({k: out[k] for k in ("marks", "base_letters", "G_COUNT_marks", "mark_counts")}, ensure_ascii=False))


def c():
    texts = {l: "\n".join(flores(l, "devtest")) + "\n" for l in LANGS}
    write_part("c", {"whole": whole(texts)})


def combine():
    assert not os.path.exists(OUT), "Rule 3"
    P, A, B, C = (json.load(open(part_path(n))) for n in ("prep", "a", "b", "c"))
    g = {"G_DISJOINT": P["G_DISJOINT"], "G_COUNT": P["G_COUNT_flores"] and B["G_COUNT_marks"]}
    r1, r2 = A["runs"]
    g["G_DETERMINISM"] = all(v["rep_identical"] for v in P["dicts"].values()) and all(r1[k]["bytes"] == r2[k]["bytes"] and r1[k]["per_sha256"] == r2[k]["per_sha256"] for k in r1)
    rt = [v["roundtrip"] for run_ in A["runs"] for v in run_.values()]
    rt += [v[n]["roundtrip"] for W in (B["whole"], C["whole"]) for v in W.values() for n in WHOLE]
    g["G_ROUNDTRIP"] = all(rt)

    def by(lang, size, comp, dl=None, ds=None):
        return r1["|".join(str(x) for x in (lang, size, comp, dl, ds))]["bytes"]

    def gain(lang, size, comp, dl, ds):
        return 1 - by(lang, size, comp, dl, ds) / by(lang, size, comp)

    tab = {}
    for l in LANGS:
        for s in ("A1", "A4"):
            raw = P["payloads"]["%s_%s" % (l, s)]["raw_bytes"]; n = P["payloads"]["%s_%s" % (l, s)]["n"]
            for comp, dl, ds in CONDS:
                key = "%s|%s|%s|%s" % (l, s, comp, "none" if dl is None else "D_%s_%s" % (dl, ds)); bb = by(l, s, comp, dl, ds)
                hdr = 0 if dl is None else (40 if comp == "zstd19" else 36)   # RFC 9842 dcz / dcb header per response (secondary)
                tab[key] = {"bytes": bb, "ratio": bb / raw, "gain": None if dl is None else gain(l, s, comp, dl, ds),
                            "gain_with_rfc9842_header": None if dl is None else 1 - (bb + hdr * n) / by(l, s, comp)}
    pred = {"PA1": gain("ar", "A1", "zstd19", "ar", "112k") >= 0.30,
            "PA2": gain("ar", "A1", "zstd19", "en", "112k") < 0.05,
            "PA3": gain("ar", "A1", "zstd19", "ar", "112k") >= gain("en", "A1", "zstd19", "en", "112k"),
            "PA4": all(gain("ar", "A4", cp, "ar", "112k") < gain("ar", "A1", cp, "ar", "112k") for cp in ("zstd19", "brotli11")),
            "PA5": gain("ar", "A1", "brotli11", "ar", "112k") >= 0.20}
    W = B["whole"]; bt = {}
    for n in WHOLE:
        extra = 8 * (W["diacritized"][n]["bytes"] - W["stripped"][n]["bytes"]) / B["marks"]
        per_letter = 8 * W["stripped"][n]["bytes"] / B["base_letters"]
        bt[n] = {"extra_bits_per_mark": extra, "bits_per_base_letter_stripped": per_letter,
                 "bpc_diacritized": 8 * W["diacritized"][n]["bytes"] / W["diacritized"]["code_points"], "bpc_stripped": 8 * W["stripped"][n]["bytes"] / W["stripped"]["code_points"],
                 "bpb_diacritized": 8 * W["diacritized"][n]["bytes"] / W["diacritized"]["raw_bytes"], "bpb_stripped": 8 * W["stripped"][n]["bytes"] / W["stripped"]["raw_bytes"]}
    pred["PB1"] = all(v["extra_bits_per_mark"] < v["bits_per_base_letter_stripped"] for v in bt.values())
    pred["PB2"] = bt["xz9e"]["extra_bits_per_mark"] < 2.0
    Cw = C["whole"]; ct = {}
    for n in WHOLE:
        ct[n] = {l: {"bpc": 8 * Cw[l][n]["bytes"] / Cw[l]["code_points"], "bpb": 8 * Cw[l][n]["bytes"] / Cw[l]["raw_bytes"]} for l in LANGS}
        ct[n]["compressed_premium"] = Cw["ar"][n]["bytes"] / Cw["en"][n]["bytes"]
    raw_prem = Cw["ar"]["raw_bytes"] / Cw["en"]["raw_bytes"]
    pred["PC1"] = ct["xz9e"]["compressed_premium"] < raw_prem and ct["xz9e"]["compressed_premium"] < 1.3
    res = {"doc": "138", "registered": "570c0c3", "versions": P["versions"], "gates": g, "predictions": pred, "arm_a": tab, "arm_b": bt, "arm_b_counts": {k: B[k] for k in ("marks", "base_letters", "mark_counts")},
           "arm_c": ct, "raw_byte_premium": raw_prem, "dicts": P["dicts"], "train": P["train"], "payloads": P["payloads"],
           "runner_sha256": sha(open(__file__, "rb").read())}
    with open(OUT, "x") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps({"gates": g, "predictions": pred}, indent=1))
    for k, v in tab.items():
        print("%-32s bytes %8d ratio %.4f gain %s hdr %s" % (k, v["bytes"], v["ratio"], "%.4f" % v["gain"] if v["gain"] is not None else "-",
                                                               "%.4f" % v["gain_with_rfc9842_header"] if v["gain_with_rfc9842_header"] is not None else "-"))
    for n, v in bt.items():
        print("B %-14s extra/mark %.3f  bits/letter %.3f  bpc %.3f/%.3f" % (n, v["extra_bits_per_mark"], v["bits_per_base_letter_stripped"], v["bpc_diacritized"], v["bpc_stripped"]))
    for n, v in ct.items():
        print("C %-14s ar bpc %.3f en bpc %.3f premium %.3f (raw %.3f)" % (n, v["ar"]["bpc"], v["en"]["bpc"], v["compressed_premium"], raw_prem))


if __name__ == "__main__":
    {"prep": prep, "a": a, "b": b, "c": c, "combine": combine}[sys.argv[1]]()
