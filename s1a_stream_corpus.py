# -*- coding: utf-8 -*-
"""S1a: stream Arabic Wikipedia (HF wikimedia/wikipedia, no full download)
into a flat file of short, clean MSA sentences for verb-preference harvest.

Runs in pulser_env (has network-happy modern stack). Output:
arwiki_sents.txt — one sentence per line, 3–14 tokens, Arabic-script only.
"""
import os, re, sys

N_ARTICLES = int(os.environ.get("S1A_ARTICLES", "120000"))
OUT = "arwiki_sents.txt"

from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train",
                  streaming=True)
ar_only = re.compile(r"^[؀-ۿ\s]+$")
splitter = re.compile(r"[.!؟?،;\n\r]+")

n_art = n_kept = 0
with open(OUT, "w", encoding="utf-8") as f:
    for art in ds:
        for raw in splitter.split(art.get("text", "")):
            toks = raw.split()
            if 3 <= len(toks) <= 14:
                s = " ".join(toks)
                if ar_only.match(s):
                    f.write(s + "\n")
                    n_kept += 1
        n_art += 1
        if n_art % 5000 == 0:
            print(f"[s1a] {n_art} articles → {n_kept} sentences", flush=True)
        if n_art >= N_ARTICLES:
            break
print(f"[s1a] FINAL: {n_art} articles → {n_kept} sentences in {OUT}",
      flush=True)
print("[s1a] DONE", flush=True)
