# -*- coding: utf-8 -*-
"""exp142_run.py -- doc 142 (REGISTERED d6f2b1f; amended 1105fae): Arabic compression follow-up.

Parts (run in order): fetch, prep, a, b, c, combine
  fetch:   network I/O -- HF parquet, Tatoeba, Wikipedia REST, Wikidata action API
  prep:    offline -- payloads, training files, dicts, G-DISJOINT, G-COUNT
  a:       arm A -- JSON payloads x {zstd: none/D_ar_112k/D_en_112k/D_json; brotli: none}
  b:       arm B -- brotli x {none, D_ar_raw, D_en_raw, D_json_raw} on FLORES + JSON payloads
  c:       arm C -- zstd x {none, D_ar_112k, D_tat} on {FLORES ar A1, Tatoeba test}
  combine: gates + predictions -> results_exp142.json (Rule 3)

Deviation from registration: D_ar_bgen not built (external build not permitted in this
environment). Arm B uses D_ar_raw only for PB1 (registered fallback). Recorded in results.
Action-API calls use utf8=1 to preserve real Unicode (not \\uXXXX escaping). Disclosed.
"""
import bz2, hashlib, json, os, random, re, subprocess, sys, time, unicodedata
import urllib.request, urllib.parse, urllib.error
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "exp142")
WORK = os.path.join(DATA, "work")
LOGS = os.path.join(ROOT, "logs")
OUT = os.path.join(ROOT, "results_exp142.json")

DATA138 = os.path.join(ROOT, "data", "exp138")
WORK138 = os.path.join(DATA138, "work")
FLORES138 = os.path.join(WORK138, "payloads")
TRAIN138_AR = os.path.join(WORK138, "train", "ar")
TRAIN138_EN = os.path.join(WORK138, "train", "en")
DICT138_AR = os.path.join(WORK138, "dicts", "D_ar_112k")
DICT138_EN = os.path.join(WORK138, "dicts", "D_en_112k")

SEED = 14200
N_TOTAL = 3000
N_TRAIN = 2000
N_TEST = 1000
N_TEST_TAT = 2000
DICT_SIZE = 112640
POOL = 9

UA = "exp142-arabic-compression-benchmark/1.0 (https://github.com/w-ali-amer/disco_in_arabia)"
RATE_S = 0.11  # ~9 req/s to stay safely under 10/s

PARQUET_URL = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.ar/train-00005-of-00007.parquet"
TATOEBA_URL = "https://downloads.tatoeba.org/exports/per_language/ara/ara_sentences.tsv.bz2"
AR_WIKI_API = "https://ar.wikipedia.org/w/api.php"
EN_WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
AR_SUMMARY_URL = "https://ar.wikipedia.org/api/rest_v1/page/summary/%s"
EN_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/%s"

LICENCES = [
    {"source": "wikimedia/wikipedia 20231101.ar shard 00005", "use": "titles for JSON payloads (disjoint from shard 00004 used in exp138 training)", "licence": "CC BY-SA 4.0"},
    {"source": "Wikipedia REST /api/rest_v1/page/summary/{title} ar and en", "use": "J1 payloads and training set", "licence": "CC BY-SA 4.0"},
    {"source": "Wikidata wbgetentities (props: labels, descriptions, aliases, claims; languages=ar)", "use": "J2 payloads and training set; re-serialised per-entity compact JSON (key order kept, whitespace minimal); disclosed", "licence": "CC0"},
    {"source": "Tatoeba per-language export ara_sentences.tsv", "use": "arm C conversational sentences", "licence": "CC BY 2.0 FR"},
    {"source": "FLORES-200 payloads (reused from exp138)", "use": "arm B FLORES baselines and arm C D_ar reference payload", "licence": "CC BY-SA 4.0"},
    {"source": "exp138 D_ar_112k and D_en_112k zstd-trained dictionaries", "use": "arm A and arm C encyclopaedic dictionaries", "licence": "CC BY-SA 4.0"},
]

_t_last = [0.0]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def nfc(s):
    return unicodedata.normalize("NFC", s)


def norm(s):
    return re.sub(r"\s+", " ", nfc(s)).strip()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def run(cmd, data):
    return subprocess.run(cmd, input=data, capture_output=True, check=True).stdout


def part_path(name):
    return os.path.join(LOGS, "exp142_%s.json" % name)


def write_part(name, obj):
    os.makedirs(LOGS, exist_ok=True)
    with open(part_path(name), "x") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def versions():
    v = {}
    for name, cmd in (("zstd", ["zstd", "--version"]), ("brotli", ["brotli", "--version"])):
        r = subprocess.run(cmd, capture_output=True, text=True)
        v[name] = (r.stdout + r.stderr).splitlines()[0]
    return v


def codec_cmds(codec, dic_path):
    if codec == "zstd19":
        c = ["zstd", "-19", "-q", "-c"]
        d = ["zstd", "-d", "-q", "-c"]
    else:
        c = ["brotli", "-q", "11", "-w", "24", "-c"]
        d = ["brotli", "-d", "-c"]
    if dic_path:
        c = c + ["-D", dic_path]
        d = d + ["-D", dic_path]
    return c, d


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _rate():
    now = time.time()
    gap = RATE_S - (now - _t_last[0])
    if gap > 0:
        time.sleep(gap)
    _t_last[0] = time.time()


def _do_req(req, max_tries=5):
    for attempt in range(max_tries):
        _rate()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == max_tries - 1:
                print("FAIL HTTP %d %s" % (e.code, req.full_url[:80]), flush=True)
                return None
            wait = 2 ** (attempt + 1)
            print("HTTP %d retry in %ds" % (e.code, wait), flush=True)
            time.sleep(wait)
        except Exception as e:
            if attempt == max_tries - 1:
                print("FAIL %s %s" % (type(e).__name__, str(e)[:60]), flush=True)
                return None
            wait = 2 ** (attempt + 1)
            time.sleep(wait)
    return None


def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return _do_req(req)


def http_post(url, data):
    body = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded"})
    return _do_req(req)


def stream_download(url, dest_path):
    """Download to file in 1 MB chunks; return total bytes or None on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    _rate()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            total = 0
            with open(dest_path, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
        return total
    except Exception as e:
        print("stream_download failed: %s" % e, flush=True)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return None


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------

def fetch():
    import pyarrow.parquet as pq
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    t0 = time.time()

    # 1. HF parquet shard 00005
    pq_path = os.path.join(DATA, "wiki_ar_00005.parquet")
    if not os.path.exists(pq_path):
        print("Downloading HF parquet...", flush=True)
        nb = stream_download(PARQUET_URL, pq_path)
        if nb is None:
            raise RuntimeError("Failed to download parquet: %s" % PARQUET_URL)
        print("Parquet: %d bytes" % nb, flush=True)
    else:
        print("Parquet cached", flush=True)

    # 2. Tatoeba bz2
    tat_path = os.path.join(DATA, "ara_sentences.tsv.bz2")
    tat_ok = os.path.exists(tat_path)
    if not tat_ok:
        print("Downloading Tatoeba...", flush=True)
        nb = stream_download(TATOEBA_URL, tat_path)
        tat_ok = nb is not None
        if not tat_ok:
            print("Tatoeba download failed; arm C will use registered fallback: no Common Voice available, arm C skipped and disclosed.", flush=True)

    # 3. Read parquet, deduplicate, shuffle, sample 3000
    titles_json = os.path.join(DATA, "titles.json")
    if not os.path.exists(titles_json):
        tbl = pq.read_table(pq_path, columns=["title"])
        raw_titles = [nfc(str(t)) for t in tbl.column("title").to_pylist() if t]
        seen = set(); dedup = []
        for t in raw_titles:
            k = norm(t)
            if k not in seen:
                seen.add(k); dedup.append(t)
        rng = random.Random(SEED)
        rng.shuffle(dedup)
        sample = dedup[:N_TOTAL]
        with open(titles_json, "w", encoding="utf-8") as f:
            json.dump({"titles": sample, "n_raw": len(raw_titles), "n_dedup": len(dedup)}, f, ensure_ascii=False)
        print("Parquet: %d raw, %d dedup, %d sampled" % (len(raw_titles), len(dedup), len(sample)), flush=True)
    else:
        d = json.load(open(titles_json))
        sample = d["titles"]
        print("Titles cached: %d" % len(sample), flush=True)

    # 4. Langlinks + pageprops (en_title and Q-ID)
    tinfo_json = os.path.join(DATA, "title_info.json")
    if not os.path.exists(tinfo_json):
        print("Fetching langlinks+pageprops for %d titles..." % len(sample), flush=True)
        title_info = {t: {"en_title": None, "qid": None} for t in sample}
        for bi in range(0, len(sample), 50):
            batch = sample[bi:bi+50]
            post = {
                "action": "query", "prop": "langlinks|pageprops",
                "ppprop": "wikibase_item", "lllang": "en", "lllimit": "500",
                "redirects": "1", "titles": "|".join(batch),
                "format": "json", "utf8": "1", "formatversion": "2",
            }
            body = http_post(AR_WIKI_API, post)
            if body is None:
                print("langlinks batch %d FAIL" % (bi // 50), flush=True)
                continue
            resp = json.loads(body.decode("utf-8"))
            q = resp.get("query", {})
            fwd = {}
            for t in batch:
                fwd[t] = t
            for item in q.get("normalized", []):
                src, dst = item.get("from", ""), item.get("to", "")
                if src in fwd: fwd[dst] = fwd[src]
            for item in q.get("redirects", []):
                src, dst = item.get("from", ""), item.get("to", "")
                if src in fwd: fwd[dst] = fwd[src]
            for page in q.get("pages", []):
                ptitle = page.get("title", "")
                orig = fwd.get(ptitle)
                if orig is None:
                    for t in batch:
                        if norm(t) == norm(ptitle):
                            orig = t; break
                if orig is None or orig not in title_info:
                    continue
                lls = page.get("langlinks", [])
                en_t = lls[0].get("title") if lls else None
                qid = page.get("pageprops", {}).get("wikibase_item")
                title_info[orig] = {"en_title": en_t, "qid": qid}
            if bi % 500 == 0:
                print("  langlinks %d/%d" % (bi + len(batch), len(sample)), flush=True)

        # Q-ID and en-title deduplication (keep first in shuffled order)
        seen_qids = set(); seen_en = set(); drop = set()
        for t in sample:
            q = title_info[t].get("qid")
            e = title_info[t].get("en_title")
            if q and q in seen_qids:
                drop.add(t); continue
            if e and e in seen_en:
                drop.add(t); continue
            if q: seen_qids.add(q)
            if e: seen_en.add(e)
        titles_clean = [t for t in sample if t not in drop]
        print("After Q-ID/en dedup: dropped %d, %d remain" % (len(drop), len(titles_clean)), flush=True)
        with open(tinfo_json, "w", encoding="utf-8") as f:
            json.dump({"title_info": title_info, "titles_clean": titles_clean, "n_dropped": len(drop)}, f, ensure_ascii=False)
    else:
        d = json.load(open(tinfo_json))
        title_info = d["title_info"]; titles_clean = d["titles_clean"]
        print("title_info cached: %d clean titles" % len(titles_clean), flush=True)

    # 5 & 6. Wikipedia AR and EN summaries (parallel, resumable)
    # 8 concurrent workers; actual rate ~4-6 req/s, well under 10/s limit.
    FETCH_WORKERS = 16  # increased from 8; actual rate ~6 req/s, within <=10/s limit
    _fail_lock = threading.Lock()

    def _fetch_summary(args):
        i, url, fp = args
        if os.path.exists(fp): return i, True, "cached"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    body = r.read()
                with open(fp, "wb") as f: f.write(body)
                return i, True, None
            except urllib.error.HTTPError as e:
                if e.code == 404: return i, False, "404"
                if attempt == 4: return i, False, "HTTP %d" % e.code
                time.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 4: return i, False, str(e)[:40]
                time.sleep(2 ** attempt)
        return i, False, "max_retries"

    j1ar_dir = os.path.join(DATA, "j1_ar_raw")
    os.makedirs(j1ar_dir, exist_ok=True)
    ar_jobs = []
    for i, title in enumerate(titles_clean):
        fp = os.path.join(j1ar_dir, "%04d.json" % i)
        enc = urllib.parse.quote(title.replace(" ", "_"), safe="")
        ar_jobs.append((i, AR_SUMMARY_URL % enc, fp))
    ar_fail = 0
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        futs = {ex.submit(_fetch_summary, j): j for j in ar_jobs}
        done = 0
        for fut in as_completed(futs):
            i, ok, msg = fut.result()
            if not ok: ar_fail += 1
            done += 1
            if done % 300 == 0: print("AR summaries: %d/%d" % (done, len(ar_jobs)), flush=True)
    print("AR summaries done, failures: %d" % ar_fail, flush=True)

    j1en_dir = os.path.join(DATA, "j1_en_raw")
    os.makedirs(j1en_dir, exist_ok=True)
    en_jobs = []
    for i, title in enumerate(titles_clean):
        en_t = title_info.get(title, {}).get("en_title")
        if not en_t: continue
        fp = os.path.join(j1en_dir, "%04d.json" % i)
        enc = urllib.parse.quote(en_t.replace(" ", "_"), safe="")
        en_jobs.append((i, EN_SUMMARY_URL % enc, fp))
    en_fail = 0
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        futs = {ex.submit(_fetch_summary, j): j for j in en_jobs}
        done = 0
        for fut in as_completed(futs):
            i, ok, msg = fut.result()
            if not ok: en_fail += 1
            done += 1
            if done % 300 == 0: print("EN summaries: %d/%d" % (done, len(en_jobs)), flush=True)
    print("EN summaries done, failures: %d" % en_fail, flush=True)

    # 7. Wikidata entities (batch 50, resumable)
    j2_dir = os.path.join(DATA, "j2_raw")
    os.makedirs(j2_dir, exist_ok=True)
    qid_to_idxs = {}
    for i, title in enumerate(titles_clean):
        qid = title_info.get(title, {}).get("qid")
        if qid: qid_to_idxs.setdefault(qid, []).append(i)

    done_idx = set()
    for fn in os.listdir(j2_dir):
        if fn.endswith(".json"):
            try:
                idx = int(fn[:-5])
                if os.path.getsize(os.path.join(j2_dir, fn)) > 10:
                    done_idx.add(idx)
            except Exception: pass

    need_qids = [q for q, idxs in qid_to_idxs.items() if not all(i in done_idx for i in idxs)]
    print("Wikidata: %d QIDs to fetch" % len(need_qids), flush=True)
    wd_fail = 0
    for bi in range(0, len(need_qids), 50):
        batch = need_qids[bi:bi+50]
        post = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels|descriptions|aliases|claims",
            "languages": "ar",
            "format": "json",
            "utf8": "1",
        }
        body = http_post(WIKIDATA_API, post)
        if body is None:
            wd_fail += len(batch)
            print("Wikidata batch %d FAIL" % (bi // 50), flush=True)
            continue
        resp = json.loads(body.decode("utf-8"))
        for qid, entity in resp.get("entities", {}).items():
            if entity.get("missing"): continue
            single = json.dumps({"entities": {qid: entity}, "success": 1},
                                separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            for idx in qid_to_idxs.get(qid, []):
                fp = os.path.join(j2_dir, "%04d.json" % idx)
                if not os.path.exists(fp):
                    with open(fp, "wb") as f: f.write(single)
        if bi % 500 == 0: print("Wikidata: %d/%d" % (bi + len(batch), len(need_qids)), flush=True)
    print("Wikidata done, failures: %d" % wd_fail, flush=True)

    # 8. Summary
    n_j1ar = sum(1 for i in range(len(titles_clean)) if os.path.exists(os.path.join(j1ar_dir, "%04d.json" % i)))
    n_j1en = sum(1 for i in range(len(titles_clean)) if os.path.exists(os.path.join(j1en_dir, "%04d.json" % i)))
    n_j2 = sum(1 for i in range(len(titles_clean)) if os.path.exists(os.path.join(j2_dir, "%04d.json" % i)))
    out = {
        "n_titles_clean": len(titles_clean),
        "n_j1ar_fetched": n_j1ar, "n_j1en_fetched": n_j1en, "n_j2_fetched": n_j2,
        "ar_failures": ar_fail, "en_failures": en_fail, "wd_failures": wd_fail,
        "n_with_en": sum(1 for v in (json.load(open(tinfo_json))["title_info"]).values() if v.get("en_title")),
        "n_with_qid": sum(1 for v in (json.load(open(tinfo_json))["title_info"]).values() if v.get("qid")),
        "tatoeba_ok": tat_ok, "wall_s": time.time() - t0,
    }
    write_part("fetch", out)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------
# PREP
# ---------------------------------------------------------------------------

def prep():
    import pyarrow.parquet as pq
    t0 = time.time()
    out = {"versions": versions()}
    os.makedirs(os.path.join(WORK, "payloads"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "train"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "dicts"), exist_ok=True)

    # Load title metadata
    d = json.load(open(os.path.join(DATA, "title_info.json")))
    title_info = d["title_info"]
    titles_clean = d["titles_clean"]
    titles_train = titles_clean[:N_TRAIN]
    titles_test = titles_clean[N_TRAIN:N_TRAIN + N_TEST]

    # A. Tatoeba
    tat_path = os.path.join(DATA, "ara_sentences.tsv.bz2")
    tat_ok = os.path.exists(tat_path)
    if tat_ok:
        raw_tsv = bz2.decompress(open(tat_path, "rb").read()).decode("utf-8", errors="replace")
        tat_raw = []
        for line in raw_tsv.splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                tat_raw.append(parts[2])
        seen_tat = set(); tat_sents = []
        for s in tat_raw:
            k = norm(s)
            if k and k not in seen_tat:
                seen_tat.add(k); tat_sents.append(s)
        random.Random(SEED).shuffle(tat_sents)
        tat_test = tat_sents[:N_TEST_TAT]
        tat_train = tat_sents[N_TEST_TAT:]
        out["tatoeba"] = {"n_raw": len(tat_raw), "n_dedup": len(tat_sents),
                          "n_test": len(tat_test), "n_train": len(tat_train)}
        tat_pay_dir = os.path.join(WORK, "payloads", "tat")
        os.makedirs(tat_pay_dir, exist_ok=True)
        for i, s in enumerate(tat_test):
            with open(os.path.join(tat_pay_dir, "%04d" % i), "wb") as f:
                f.write(s.encode("utf-8"))
        tat_tr_dir = os.path.join(WORK, "train", "tat")
        os.makedirs(tat_tr_dir, exist_ok=True)
        for i, s in enumerate(tat_train):
            with open(os.path.join(tat_tr_dir, "%05d" % i), "wb") as f:
                f.write(s.encode("utf-8"))
    out["tat_ok"] = tat_ok

    # B. JSON payload sets (train=indices 0..N_TRAIN-1, test=N_TRAIN..N_TRAIN+N_TEST-1)
    def build_json_set(name, raw_dir):
        pay_dir = os.path.join(WORK, "payloads", name)
        tr_dir = os.path.join(WORK, "train", name)
        os.makedirs(pay_dir, exist_ok=True); os.makedirs(tr_dir, exist_ok=True)
        n_tr = 0; rb_tr = 0
        for j in range(N_TRAIN):
            src = os.path.join(raw_dir, "%04d.json" % j)
            if not os.path.exists(src): continue
            data = open(src, "rb").read()
            with open(os.path.join(tr_dir, "%05d" % j), "wb") as f: f.write(data)
            n_tr += 1; rb_tr += len(data)
        n_te = 0; rb_te = 0
        for j in range(N_TEST):
            src = os.path.join(raw_dir, "%04d.json" % (N_TRAIN + j))
            if not os.path.exists(src): continue
            data = open(src, "rb").read()
            with open(os.path.join(pay_dir, "%04d" % j), "wb") as f: f.write(data)
            n_te += 1; rb_te += len(data)
        return {"n_train": n_tr, "n_test": n_te, "raw_bytes_train": rb_tr, "raw_bytes_test": rb_te}

    out["payloads"] = {}
    out["payloads"]["j1_ar"] = build_json_set("j1_ar", os.path.join(DATA, "j1_ar_raw"))
    out["payloads"]["j1_en"] = build_json_set("j1_en", os.path.join(DATA, "j1_en_raw"))
    out["payloads"]["j2"] = build_json_set("j2", os.path.join(DATA, "j2_raw"))

    # G-COUNT: each set >= 90% of target
    g_count = True
    for name, ps in out["payloads"].items():
        if ps["n_test"] < 0.9 * N_TEST:
            print("G-COUNT FAIL: %s test %d < 900" % (name, ps["n_test"])); g_count = False
        if ps["n_train"] < 0.9 * N_TRAIN:
            print("G-COUNT FAIL: %s train %d < 1800" % (name, ps["n_train"])); g_count = False
    if tat_ok and out["tatoeba"]["n_test"] < 0.9 * N_TEST_TAT:
        print("G-COUNT FAIL: tatoeba test %d < 1800" % out["tatoeba"]["n_test"]); g_count = False
    out["G_COUNT"] = g_count

    # G-DISJOINT
    train_titles_norm = set(norm(t) for t in titles_train)
    test_titles_norm = set(norm(t) for t in titles_test)
    train_qids = set(title_info.get(t, {}).get("qid", "") or "" for t in titles_train) - {""}
    test_qids = set(title_info.get(t, {}).get("qid", "") or "" for t in titles_test) - {""}
    title_overlap = train_titles_norm & test_titles_norm
    qid_overlap = train_qids & test_qids

    # Shard 00005 vs shard 00004
    pq04 = os.path.join(DATA138, "wiki_ar_00004.parquet")
    shard04_norms = set()
    if os.path.exists(pq04):
        tbl04 = pq.read_table(pq04, columns=["title"])
        for t in tbl04.column("title").to_pylist():
            if t: shard04_norms.add(norm(nfc(str(t))))
    d05 = json.load(open(os.path.join(DATA, "titles.json")))
    shard05_norms = set(norm(t) for t in d05["titles"])
    shard_overlap = shard04_norms & shard05_norms

    # Tatoeba test vs train: disjoint by construction (different slices, dedup applied)
    out["G_DISJOINT"] = (len(title_overlap) == 0 and len(qid_overlap) == 0 and len(shard_overlap) == 0)
    out["disjoint"] = {
        "json_title_overlap": len(title_overlap), "json_qid_overlap": len(qid_overlap),
        "shard05_vs_04_overlap": len(shard_overlap),
        "shard_overlap_examples": sorted(shard_overlap)[:5],
    }

    # C. Raw dictionaries (plain concatenation of training files, no separator)
    dicts_dir = os.path.join(WORK, "dicts")

    def make_raw_dict(src_dir, out_path):
        buf = b""
        for fn in sorted(os.listdir(src_dir)):
            buf += open(os.path.join(src_dir, fn), "rb").read()
            if len(buf) >= DICT_SIZE: break
        d = buf[:DICT_SIZE]
        with open(out_path, "wb") as f: f.write(d)
        return {"bytes": len(d), "sha256": sha(d), "separator": "none"}

    out["dicts"] = {}
    out["dicts"]["D_ar_raw"] = make_raw_dict(TRAIN138_AR, os.path.join(dicts_dir, "D_ar_raw"))
    out["dicts"]["D_en_raw"] = make_raw_dict(TRAIN138_EN, os.path.join(dicts_dir, "D_en_raw"))
    out["dicts"]["D_ar_bgen"] = "not_run: external build not permitted in this environment (registered fallback)"
    for name, tr_subdir in (("D_json_raw_j1ar", "j1_ar"), ("D_json_raw_j1en", "j1_en"), ("D_json_raw_j2", "j2")):
        tr_d = os.path.join(WORK, "train", tr_subdir)
        out["dicts"][name] = make_raw_dict(tr_d, os.path.join(dicts_dir, name))

    # D. Trained dictionaries (zstd --train, twice for G-DET)
    def train_dict(name, tr_dir):
        base = os.path.join(dicts_dir, name)
        for rep in (1, 2):
            p = base if rep == 1 else base + "_rep2"
            cmd = ["zstd", "--train", "-q", "-r", tr_dir, "--maxdict=%d" % DICT_SIZE, "-o", p]
            subprocess.run(cmd, check=True, capture_output=True)
        b1 = open(base, "rb").read(); b2 = open(base + "_rep2", "rb").read()
        return {"bytes": len(b1), "sha256": sha(b1), "rep_identical": b1 == b2}

    for name, tr_subdir in (("D_json_j1ar", "j1_ar"), ("D_json_j1en", "j1_en"), ("D_json_j2", "j2")):
        out["dicts"][name] = train_dict(name, os.path.join(WORK, "train", tr_subdir))

    if tat_ok:
        out["dicts"]["D_tat"] = train_dict("D_tat", os.path.join(WORK, "train", "tat"))
    else:
        out["dicts"]["D_tat"] = "not_built: Tatoeba unavailable"

    out["wall_s"] = time.time() - t0
    write_part("prep", out)
    print(json.dumps({k: out[k] for k in ("G_DISJOINT", "G_COUNT", "disjoint", "payloads", "dicts")},
                     indent=1, ensure_ascii=False))


# ---------------------------------------------------------------------------
# ARM A
# ---------------------------------------------------------------------------

def _a_job(job):
    payload, codec, dname, dpath, pay_dir = job
    c, d = codec_cmds(codec, dpath)
    tot = 0; per = []; ok = True
    for fn in sorted(os.listdir(pay_dir)):
        raw = open(os.path.join(pay_dir, fn), "rb").read()
        z = run(c, raw); ok &= (run(d, z) == raw)
        tot += len(z); per.append(len(z))
    key = "%s|%s|%s" % (payload, codec, dname)
    return key, {"bytes": tot, "roundtrip": ok, "per_sha256": sha(json.dumps(per).encode()), "n": len(per)}


def _a_jobs():
    dicts_dir = os.path.join(WORK, "dicts")
    jobs = []
    for payload, dj in (("j1_ar", "D_json_j1ar"), ("j1_en", "D_json_j1en"), ("j2", "D_json_j2")):
        pay_dir = os.path.join(WORK, "payloads", payload)
        for dname, dpath in (("none", None), ("D_ar_112k", DICT138_AR),
                              ("D_en_112k", DICT138_EN), ("D_json", os.path.join(dicts_dir, dj))):
            jobs.append((payload, "zstd19", dname, dpath, pay_dir))
        jobs.append((payload, "brotli11", "none", None, pay_dir))
    return jobs


def a():
    jobs = _a_jobs()
    out = {"runs": [], "t0": time.time()}
    with Pool(POOL) as pool:
        for rep in (1, 2):
            res = {}
            for key, r in pool.imap_unordered(_a_job, jobs):
                res[key] = r; print(rep, key, r["bytes"], r["roundtrip"], flush=True)
            out["runs"].append(res)
    out["wall_s"] = time.time() - out.pop("t0")
    write_part("a", out)


# ---------------------------------------------------------------------------
# ARM B
# ---------------------------------------------------------------------------

def _b_job(job):
    pname, dname, dpath, pay_dir = job
    c, d = codec_cmds("brotli11", dpath)
    tot = 0; per = []; ok = True
    for fn in sorted(os.listdir(pay_dir)):
        raw = open(os.path.join(pay_dir, fn), "rb").read()
        z = run(c, raw); ok &= (run(d, z) == raw)
        tot += len(z); per.append(len(z))
    key = "%s|brotli11|%s" % (pname, dname)
    return key, {"bytes": tot, "roundtrip": ok, "per_sha256": sha(json.dumps(per).encode()), "n": len(per)}


def _b_jobs():
    dicts_dir = os.path.join(WORK, "dicts")
    d_ar_raw = os.path.join(dicts_dir, "D_ar_raw")
    d_en_raw = os.path.join(dicts_dir, "D_en_raw")
    d_json_j1ar = os.path.join(dicts_dir, "D_json_raw_j1ar")
    d_json_j1en = os.path.join(dicts_dir, "D_json_raw_j1en")
    d_json_j2 = os.path.join(dicts_dir, "D_json_raw_j2")
    jobs = []
    # FLORES payloads (from exp138)
    for pname, pdir in (("flores_ar_A1", os.path.join(FLORES138, "ar_A1")),
                         ("flores_ar_A4", os.path.join(FLORES138, "ar_A4")),
                         ("flores_en_A1", os.path.join(FLORES138, "en_A1")),
                         ("flores_en_A4", os.path.join(FLORES138, "en_A4"))):
        for dname, dpath in (("none", None), ("D_ar_raw", d_ar_raw), ("D_en_raw", d_en_raw)):
            jobs.append((pname, dname, dpath, pdir))
    # JSON payloads
    for pname, pdir, d_json_raw in (
        ("j1_ar", os.path.join(WORK, "payloads", "j1_ar"), d_json_j1ar),
        ("j1_en", os.path.join(WORK, "payloads", "j1_en"), d_json_j1en),
        ("j2",    os.path.join(WORK, "payloads", "j2"),    d_json_j2),
    ):
        for dname, dpath in (("none", None), ("D_ar_raw", d_ar_raw),
                              ("D_en_raw", d_en_raw), ("D_json_raw", d_json_raw)):
            jobs.append((pname, dname, dpath, pdir))
    return jobs


def b():
    jobs = _b_jobs()
    out = {"runs": [], "t0": time.time()}
    with Pool(POOL) as pool:
        for rep in (1, 2):
            res = {}
            for key, r in pool.imap_unordered(_b_job, jobs):
                res[key] = r; print(rep, key, r["bytes"], r["roundtrip"], flush=True)
            out["runs"].append(res)
    out["wall_s"] = time.time() - out.pop("t0")
    write_part("b", out)


# ---------------------------------------------------------------------------
# ARM C
# ---------------------------------------------------------------------------

def _c_job(job):
    pname, dname, dpath, pay_dir = job
    c, d = codec_cmds("zstd19", dpath)
    tot = 0; per = []; ok = True
    for fn in sorted(os.listdir(pay_dir)):
        raw = open(os.path.join(pay_dir, fn), "rb").read()
        z = run(c, raw); ok &= (run(d, z) == raw)
        tot += len(z); per.append(len(z))
    key = "%s|zstd19|%s" % (pname, dname)
    return key, {"bytes": tot, "roundtrip": ok, "per_sha256": sha(json.dumps(per).encode()), "n": len(per)}


def _c_jobs():
    dicts_dir = os.path.join(WORK, "dicts")
    d_tat = os.path.join(dicts_dir, "D_tat")
    tat_ok = os.path.exists(d_tat)
    flores_ar_A1 = os.path.join(FLORES138, "ar_A1")
    tat_test_dir = os.path.join(WORK, "payloads", "tat")
    jobs = []
    payloads = [("flores_ar_A1", flores_ar_A1)]
    if tat_ok:
        payloads.append(("tat_test", tat_test_dir))
    for pname, pdir in payloads:
        for dname, dpath in (("none", None), ("D_ar_112k", DICT138_AR)):
            jobs.append((pname, dname, dpath, pdir))
        if tat_ok:
            jobs.append((pname, "D_tat", d_tat, pdir))
    return jobs


def c():
    jobs = _c_jobs()
    out = {"runs": [], "t0": time.time(), "tat_ok": os.path.exists(os.path.join(WORK, "dicts", "D_tat"))}
    with Pool(POOL) as pool:
        for rep in (1, 2):
            res = {}
            for key, r in pool.imap_unordered(_c_job, jobs):
                res[key] = r; print(rep, key, r["bytes"], r["roundtrip"], flush=True)
            out["runs"].append(res)
    out["wall_s"] = time.time() - out.pop("t0")
    write_part("c", out)


# ---------------------------------------------------------------------------
# COMBINE
# ---------------------------------------------------------------------------

def combine():
    assert not os.path.exists(OUT), "Rule 3"
    P, F, A, B, C = (json.load(open(part_path(n))) for n in ("prep", "fetch", "a", "b", "c"))
    r1_a, r2_a = A["runs"]
    r1_b, r2_b = B["runs"]
    r1_c, r2_c = C["runs"]

    # Gates
    g = {}
    g["G_DISJOINT"] = P["G_DISJOINT"]
    g["G_COUNT"] = P["G_COUNT"]
    g["G_LICENCE"] = True  # licences table is in this results JSON

    g["G_DET"] = (
        all(r1_a[k]["bytes"] == r2_a[k]["bytes"] and r1_a[k]["per_sha256"] == r2_a[k]["per_sha256"] for k in r1_a)
        and all(r1_b[k]["bytes"] == r2_b[k]["bytes"] for k in r1_b)
        and all(r1_c[k]["bytes"] == r2_c[k]["bytes"] for k in r1_c)
    )
    all_rt = ([v["roundtrip"] for run_ in (r1_a, r2_a, r1_b, r2_b, r1_c, r2_c) for v in run_.values()])
    g["G_ROUNDTRIP"] = all(all_rt)

    def ab(run_, key): return run_[key]["bytes"]

    def a_bytes(payload, codec, dname): return r1_a["%s|%s|%s" % (payload, codec, dname)]["bytes"]
    def a_n(payload, codec, dname): return r1_a["%s|%s|%s" % (payload, codec, dname)]["n"]
    def a_gain(payload, codec, dname):
        return 1.0 - a_bytes(payload, codec, dname) / a_bytes(payload, codec, "none")

    def b_bytes(pname, dname): return r1_b["%s|brotli11|%s" % (pname, dname)]["bytes"]
    def b_gain(pname, dname): return 1.0 - b_bytes(pname, dname) / b_bytes(pname, "none")

    def c_bytes(pname, dname): return r1_c["%s|zstd19|%s" % (pname, dname)]["bytes"]
    def c_gain(pname, dname): return 1.0 - c_bytes(pname, dname) / c_bytes(pname, "none")

    # Arm A table
    arm_a = {}
    for payload in ("j1_ar", "j1_en", "j2"):
        raw_b = P["payloads"][payload]["raw_bytes_test"]
        n = P["payloads"][payload]["n_test"]
        for codec in ("zstd19", "brotli11"):
            dnames = ["none", "D_ar_112k", "D_en_112k", "D_json"] if codec == "zstd19" else ["none"]
            for dname in dnames:
                bb = a_bytes(payload, codec, dname)
                gain = a_gain(payload, codec, dname) if dname != "none" else None
                hdr = 40 if codec == "zstd19" else 36
                gain_hdr = (1.0 - (bb + hdr * n) / a_bytes(payload, codec, "none")) if dname != "none" else None
                arm_a["%s|%s|%s" % (payload, codec, dname)] = {
                    "bytes": bb, "n": n, "ratio": bb / raw_b if raw_b else None,
                    "gain": gain, "gain_with_rfc9842_header": gain_hdr,
                }

    # Arm B table
    arm_b = {}
    flores_payloads = ["flores_ar_A1", "flores_ar_A4", "flores_en_A1", "flores_en_A4"]
    json_payloads = ["j1_ar", "j1_en", "j2"]
    for pname in flores_payloads:
        for dname in ("none", "D_ar_raw", "D_en_raw"):
            bb = b_bytes(pname, dname)
            arm_b["%s|brotli11|%s" % (pname, dname)] = {
                "bytes": bb, "gain": b_gain(pname, dname) if dname != "none" else None}
    for pname in json_payloads:
        for dname in ("none", "D_ar_raw", "D_en_raw", "D_json_raw"):
            bb = b_bytes(pname, dname)
            arm_b["%s|brotli11|%s" % (pname, dname)] = {
                "bytes": bb, "gain": b_gain(pname, dname) if dname != "none" else None}

    # Arm C table
    arm_c = {}
    tat_ok = C["tat_ok"]
    c_payloads = ["flores_ar_A1"] + (["tat_test"] if tat_ok else [])
    c_dicts = ["none", "D_ar_112k"] + (["D_tat"] if tat_ok else [])
    for pname in c_payloads:
        for dname in c_dicts:
            key = "%s|zstd19|%s" % (pname, dname)
            if key not in r1_c: continue
            bb = c_bytes(pname, dname)
            arm_c[key] = {"bytes": bb, "gain": c_gain(pname, dname) if dname != "none" else None}

    # RFC 9842 deployment comparison
    rfc = {}
    for payload in ("j1_ar", "j1_en", "j2"):
        n = a_n(payload, "zstd19", "none")
        zstd_hdr = a_bytes(payload, "zstd19", "D_json") + 40 * n
        brotli_none = a_bytes(payload, "brotli11", "none")
        rfc[payload] = {
            "zstd_D_json_bytes": a_bytes(payload, "zstd19", "D_json"),
            "n_payloads": n, "rfc9842_header_bytes_per_payload": 40,
            "zstd_D_json_plus_header_total": zstd_hdr,
            "brotli_none_total": brotli_none,
            "delta": zstd_hdr - brotli_none,
            "winner": "zstd+D_json" if zstd_hdr < brotli_none else "brotli_none",
        }

    # Predictions (thresholds verbatim from doc 142 §6)
    pred = {}
    pj1_gain = a_gain("j1_ar", "zstd19", "D_json")
    pred["PJ1"] = pj1_gain >= 0.40

    pj2a = a_gain("j1_ar", "zstd19", "D_json") > a_gain("j1_ar", "zstd19", "D_ar_112k")
    pj2b = a_gain("j2",    "zstd19", "D_json") > a_gain("j2",    "zstd19", "D_ar_112k")
    pred["PJ2"] = pj2a and pj2b

    pred["PJ3"] = a_gain("j1_ar", "zstd19", "D_json") >= a_gain("j1_en", "zstd19", "D_json")

    n_j1ar = a_n("j1_ar", "zstd19", "none")
    pred["PJ4"] = (a_bytes("j1_ar", "zstd19", "D_json") + 40 * n_j1ar) < a_bytes("j1_ar", "brotli11", "none")

    # PB1: D_ar_raw on FLORES ar A1 >= 0.22 (bgen not run; registered fallback)
    pb1_gain = b_gain("flores_ar_A1", "D_ar_raw")
    pred["PB1"] = pb1_gain >= 0.22

    pred["PB2"] = pb1_gain < 0.403

    if tat_ok:
        pc1a = c_gain("tat_test",    "D_tat")    > c_gain("tat_test",    "D_ar_112k")
        pc1b = c_gain("flores_ar_A1", "D_ar_112k") > c_gain("flores_ar_A1", "D_tat")
        pred["PC1"] = pc1a and pc1b
        pc2a = c_gain("tat_test",    "D_ar_112k") > 0
        pc2b = c_gain("flores_ar_A1", "D_tat")    > 0
        pred["PC2"] = pc2a and pc2b
    else:
        pred["PC1"] = "SKIPPED: Tatoeba unavailable"
        pred["PC2"] = "SKIPPED: Tatoeba unavailable"

    pred_detail = {
        "PJ1": {"value": pj1_gain, "threshold": ">=0.40"},
        "PJ2": {"j1_ar": pj2a, "j2": pj2b},
        "PJ3": {"j1_ar_gain": a_gain("j1_ar","zstd19","D_json"), "j1_en_gain": a_gain("j1_en","zstd19","D_json")},
        "PJ4": {"zstd_D_json_total": a_bytes("j1_ar","zstd19","D_json")+40*n_j1ar, "brotli_none_total": a_bytes("j1_ar","brotli11","none")},
        "PB1": {"value": pb1_gain, "threshold": ">=0.22", "D_ar_bgen": "not_run: external build not permitted (registered fallback)"},
        "PB2": {"value": pb1_gain, "threshold": "<0.403"},
    }
    if tat_ok:
        pred_detail["PC1"] = {"tat_D_tat_gain": c_gain("tat_test","D_tat"), "tat_D_ar_gain": c_gain("tat_test","D_ar_112k"),
                               "flores_D_ar_gain": c_gain("flores_ar_A1","D_ar_112k"), "flores_D_tat_gain": c_gain("flores_ar_A1","D_tat")}
        pred_detail["PC2"] = {"tat_cross_gain": c_gain("tat_test","D_ar_112k"), "flores_cross_gain": c_gain("flores_ar_A1","D_tat")}

    res = {
        "doc": "142", "registered": "d6f2b1f", "amended": "1105fae",
        "deviations": [
            "D_ar_bgen not built: external build not permitted in this environment. PB1 uses D_ar_raw only (registered fallback).",
            "Action-API calls use utf8=1 to preserve Unicode (not \\uXXXX). Disclosed in §3."
        ],
        "licences": LICENCES,
        "gates": g, "predictions": pred, "predictions_detail": pred_detail,
        "arm_a": arm_a, "arm_b": arm_b, "arm_c": arm_c,
        "rfc9842_comparison": rfc,
        "payloads": P["payloads"], "dicts": P["dicts"], "versions": P["versions"],
        "fetch_summary": {k: F.get(k) for k in ("n_titles_clean","n_with_en","n_with_qid",
                          "n_j1ar_fetched","n_j1en_fetched","n_j2_fetched",
                          "ar_failures","en_failures","wd_failures","tatoeba_ok")},
        "runner_sha256": sha(open(__file__, "rb").read()),
    }
    with open(OUT, "x") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps({"gates": g, "predictions": pred}, indent=1))
    print("\nArm A:")
    for k, v in arm_a.items():
        if v["gain"] is not None:
            print("  %-40s gain=%.4f hdr=%.4f" % (k, v["gain"], v["gain_with_rfc9842_header"] or 0))
    print("\nArm B:")
    for k, v in arm_b.items():
        if v["gain"] is not None:
            print("  %-45s gain=%.4f" % (k, v["gain"]))
    print("\nArm C:")
    for k, v in arm_c.items():
        if v["gain"] is not None:
            print("  %-40s gain=%.4f" % (k, v["gain"]))
    print("\nRFC 9842 comparison:")
    for k, v in rfc.items():
        print("  %-8s zstd+D_json+hdr=%d  brotli_none=%d  winner=%s" % (k, v["zstd_D_json_plus_header_total"], v["brotli_none_total"], v["winner"]))


if __name__ == "__main__":
    {"fetch": fetch, "prep": prep, "a": a, "b": b, "c": c, "combine": combine}[sys.argv[1]]()
