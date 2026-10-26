#!/usr/bin/env python3
"""exp77_tier2_gemma.py -- §2 Tier-2: gemma3:12b per-word MSA validation.

Reads exp77_enum_grid.json (produced by exp77_inventory_enum_wsl.py on WSL,
scp'd to Mac). For each candidate word, queries gemma3:12b via ollama local
API (temperature 0, JSON format) and records accept/reject.

ACCEPTANCE RULE (pinned, §2):
    A word is admitted iff AraVec-attested AND at least one model accepts.
    Since all grid words ARE AraVec-attested (by construction of the scan),
    the relevant split is:
        gemma=yes  -> admitted
        gemma=no   -> tier3_pending (awaiting haiku cross-check)
    The haiku pass (NOT run here) is the tiebreaker for gemma=no words.

PROMPT DESIGN:
    Per-word calls (not batched) to avoid JSON drift.
    Temperature 0, response_format: {"type": "json_object"}.
    Root '#' radicals are explained as weak (و/ي/ء absorbed), never shown raw
    to avoid spurious rejection of the weak stratum the quota protects.
    Column names given as Arabic wazn names (morphological labels).
    Exact prompt template recorded in meta.

Run:  python exp77_tier2_gemma.py --grid exp77_enum_grid.json \
                                   --out exp77_tier2_verdicts.json
"""
import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
import urllib.error
import time

HERE = os.path.dirname(os.path.abspath(__file__))

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:12b"

COLUMN_ARABIC = {
    "faail":   ("فاعِل", "active participle Form I"),
    "faiil":   ("فَعِيل", "adjective/noun pattern"),
    "fa33aal": ("فَعّال", "intensive agent noun"),
    "maf3uul": ("مَفعُول", "passive participle Form I"),
    "mu1":     ("مُفعِّل", "Form II active participle"),
    "mu2":     ("مُفعِل", "Form IV active participle"),
}


def root_human(root_str):
    """Convert CAMeL root string to human-readable form for the prompt.
    Replaces '#' with a description of the weak radical position.
    """
    if not root_str:
        return root_str
    parts = root_str.split(".")
    if len(parts) != 3:
        return root_str
    positions = ["first", "second", "third"]
    weak_positions = [positions[i] for i, r in enumerate(parts) if r == "#"]
    concrete = [r for r in parts if r != "#"]
    if not weak_positions:
        return root_str
    root_concrete = ".".join(r if r != "#" else "?" for r in parts)
    note = " (the %s radical is weak: و/ي/ء, absorbed into the template)" % \
           "/".join(weak_positions)
    return root_concrete + note


PROMPT_TEMPLATE = """\
You are an Arabic morphology expert. Answer ONLY with valid JSON.

Question: Is the Arabic word "{word}" a real, reasonably common Modern Standard Arabic word
whose root is {root_human} and whose morphological pattern is {wazn_ar} ({wazn_name}, {col_desc})?

Answer with exactly this JSON structure (no other text):
{{"accept": true or false, "reason": "one sentence"}}

Rules:
- "accept": true only if the word is genuinely attested MSA, has the stated root, and fits the stated pattern.
- "accept": false if the word is rare/dialectal/misspelled/does not have this root or pattern.
- Keep "reason" under 20 words.
"""


def make_prompt(word, root, column):
    wazn_ar, col_desc = COLUMN_ARABIC.get(column, (column, column))
    return PROMPT_TEMPLATE.format(
        word=word,
        root_human=root_human(root),
        wazn_ar=wazn_ar,
        wazn_name=column,
        col_desc=col_desc,
    )


def get_model_digest():
    try:
        req = urllib.request.Request("http://localhost:11434/api/show",
                                     data=json.dumps({"name": MODEL}).encode(),
                                     method="POST",
                                     headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
        return resp.get("digest", "unknown")
    except Exception as e:
        return "error:%s" % str(e)[:40]


def query_gemma(prompt, retries=3):
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 80}
    }).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(OLLAMA_URL, data=payload, method="POST",
                                         headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=60).read()
            resp_json = json.loads(resp)
            raw = resp_json.get("response", "")
            parsed = json.loads(raw)
            return parsed.get("accept"), parsed.get("reason", ""), raw
        except (json.JSONDecodeError, KeyError) as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            return None, "parse_error:%s" % str(e)[:50], ""
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            return None, "url_error:%s" % str(e)[:50], ""
    return None, "max_retries", ""


CKPT_PATH = os.path.join(HERE, "exp77_tier2_checkpoint.jsonl")


def load_checkpoint():
    """Load previously completed (word, root, column) verdicts from checkpoint."""
    done = {}
    if not os.path.exists(CKPT_PATH):
        return done
    with open(CKPT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = (r["word"], r["root"], r["column"])
                done[key] = r
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default=os.path.join(HERE, "exp77_enum_grid.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "exp77_tier2_verdicts.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Test prompt generation without calling ollama")
    args = ap.parse_args()

    grid_doc = json.load(open(args.grid, encoding="utf-8"))
    grid_rows = grid_doc["grid"]
    print("[tier2] loaded %d word-rows from %s" % (len(grid_rows), args.grid), flush=True)

    # Load checkpoint (resume from prior interrupted run)
    done = load_checkpoint()
    print("[tier2] checkpoint: %d rows already completed" % len(done), flush=True)

    digest = get_model_digest()
    print("[tier2] model=%s digest=%s" % (MODEL, digest), flush=True)

    results = list(done.values())   # start with completed results
    n_accept = sum(1 for r in results if r.get("gemma_verdict") is True)
    n_reject = sum(1 for r in results if r.get("gemma_verdict") is False)
    n_error = sum(1 for r in results if r.get("gemma_verdict") is None)

    ckpt_f = open(CKPT_PATH, "a", encoding="utf-8")

    for i, row in enumerate(grid_rows):
        word = row["word"]
        root = row["root"]
        col = row["column"]
        key = (word, root, col)

        # Skip if already done
        if key in done:
            continue

        prompt = make_prompt(word, root, col)

        if args.dry_run:
            print("[dry-run] word=%s root=%s col=%s" % (word, root, col))
            if i >= 3:
                print("[dry-run] stopping at 3 rows")
                break
            continue

        accept, reason, raw = query_gemma(prompt)
        if accept is True:
            n_accept += 1
        elif accept is False:
            n_reject += 1
        else:
            n_error += 1

        row_out = dict(row)
        row_out["gemma_verdict"] = accept
        row_out["gemma_reason"] = reason
        row_out["gemma_raw"] = raw[:200] if raw else ""
        # Tier3 pending = gemma said no (awaiting haiku cross-check)
        row_out["tier3_pending"] = (accept is False) or (accept is None)
        results.append(row_out)

        # Write checkpoint immediately (survives interruption)
        ckpt_f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
        ckpt_f.flush()

        total_done = len(results)
        if total_done % 20 == 0 or total_done == len(grid_rows):
            print("[tier2] %d/%d accept=%d reject=%d error=%d" %
                  (total_done, len(grid_rows), n_accept, n_reject, n_error), flush=True)

    ckpt_f.close()

    if args.dry_run:
        print("[dry-run] done - prompts look correct, no ollama calls made")
        return

    # Collect disagreements: words where gemma rejected (tier3_pending=True)
    disagreements = [r for r in results if r["tier3_pending"]]
    disa_path = os.path.join(HERE, "exp77_tier3_disagreements.json")
    json.dump({
        "meta": {
            "purpose": "exp77 §2 tier-3 pending: words where gemma3:12b rejected, awaiting haiku cross-check",
            "acceptance_rule": ("admitted iff AraVec-attested AND at least one model accepts; "
                                "these words have gemma=False/None, require haiku tiebreaker"),
            "n_disagreements": len(disagreements),
            "note": "DO NOT run haiku yourself; leave this file for the orchestrator",
        },
        "words": disagreements,
    }, open(disa_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[tier2] wrote tier-3 disagreements file: %s (%d words)" %
          (disa_path, len(disagreements)))

    out_doc = {
        "meta": {
            "purpose": "exp77 §2 tier-2 verdicts: gemma3:12b per-word MSA validation",
            "model": MODEL,
            "model_digest": digest,
            "ollama_url": OLLAMA_URL,
            "temperature": 0,
            "format": "json",
            "prompt_template": PROMPT_TEMPLATE,
            "root_hash_handling": (
                "CAMeL '#' radicals explained in natural language as 'weak: و/ي/ء' "
                "to avoid spurious rejection of the weak stratum"),
            "acceptance_rule": (
                "admitted iff AraVec-attested (all=True by construction) AND "
                "at least one model accepts. Here: gemma=True -> admitted; "
                "gemma=False -> tier3_pending (awaiting haiku orchestrator pass)"),
            "n_total": len(results),
            "n_accept": n_accept,
            "n_reject": n_reject,
            "n_error": n_error,
            "n_tier3_pending": len(disagreements),
            "grid_sha256": hashlib.sha256(
                open(args.grid, "rb").read()).hexdigest(),
        },
        "words": results,
    }
    json.dump(out_doc, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[tier2] wrote %s" % args.out)
    print("[tier2] SUMMARY: total=%d accept=%d reject=%d error=%d tier3_pending=%d" %
          (len(results), n_accept, n_reject, n_error, len(disagreements)))


if __name__ == "__main__":
    main()
