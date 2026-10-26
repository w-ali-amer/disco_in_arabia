# -*- coding: utf-8 -*-
"""exp100_census.py -- doc 97 (REGISTERED a06fb33): PADT agreement-reader census (frozen classifier, doc 97 §2).

Reads data/UD_Arabic-PADT/ar_padt-ud-{train,dev,test}.conllu; writes census_exp100.json (Rule 3).
Noun tuple x = (Gender, Number, Definite, Case); missing -> "∅". Reader classes applied in order, first match wins.
"""
import collections, json, os
import exp94_common as C94

DATA = "data/UD_Arabic-PADT"
FILES = ["ar_padt-ud-%s.conllu" % s for s in ("train", "dev", "test")]
NOUN = {"NOUN", "PROPN"}
SUBJ = {"nsubj", "nsubj:pass"}
NOSUBJ = {"nsubj", "nsubj:pass", "csubj", "csubj:pass", "expl"}
READS = {"R-VSO": ("Gender",), "R-SVO": ("Gender", "Number"), "R-PRODROP": ("Gender", "Number"),
         "R-PRON": ("Gender", "Number"), "R-ADJ": ("Gender", "Number", "Definite", "Case"),
         "R-DEM": ("Gender", "Number"), "R-REL": ("Gender", "Number", "Case")}
REMOTE = {"R-PRODROP", "R-PRON"}
XF = ("Gender", "Number", "Definite", "Case")


def sentences(path):
    sent, n_words = [], 0
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if sent:
                yield sent
            sent = []
            continue
        if line.startswith("#"):
            continue
        c = line.split("\t")
        assert len(c) == 10, line
        if "-" in c[0] or "." in c[0]:
            continue
        feats = dict(kv.split("=", 1) for kv in c[5].split("|")) if c[5] != "_" else {}
        sent.append({"id": int(c[0]), "upos": c[3], "feats": feats, "head": int(c[6]), "deprel": c[7]})
    if sent:
        yield sent


def fv(tok, keys):
    return tuple(tok["feats"].get(k, "∅") for k in keys)


def classify(tok, kids, sent):
    f, up, dr = tok["feats"], tok["upos"], tok["deprel"]
    if up == "VERB" and f.get("Person") == "3":
        subj = [k for k in kids if k["deprel"] in SUBJ and k["upos"] in NOUN]
        if subj:
            s = subj[0]
            return ("R-VSO" if s["id"] > tok["id"] else "R-SVO"), s
    if up == "VERB" and f.get("VerbForm") == "Fin" and f.get("Person") == "3" and not any(k["deprel"] in NOSUBJ for k in kids):
        return "R-PRODROP", None
    if up == "PRON" and f.get("PronType") == "Prs" and f.get("Person") == "3":
        return "R-PRON", None
    head = sent[tok["head"] - 1] if tok["head"] > 0 else None
    if up == "ADJ" and dr == "amod" and head and head["upos"] in NOUN:
        return "R-ADJ", head
    if f.get("PronType") == "Dem" and dr == "det" and head and head["upos"] in NOUN:
        return "R-DEM", head
    if f.get("PronType") == "Rel":
        a = tok
        while a["head"] > 0 and a["deprel"] != "acl:relcl":
            a = sent[a["head"] - 1]
        tgt = sent[a["head"] - 1] if (a["deprel"] == "acl:relcl" and a["head"] > 0) else None
        return "R-REL", (tgt if tgt and tgt["upos"] in NOUN else None)
    return None, None


def main():
    counts, noun_x, answers, joint = collections.Counter(), collections.Counter(), {r: collections.Counter() for r in READS}, collections.Counter()
    missing = collections.Counter(); n_words = n_sent = n_nouns = 0; unlinked_rel = 0
    defl = collections.Counter()
    for fn in FILES:
        for sent in sentences(os.path.join(DATA, fn)):
            n_sent += 1; n_words += len(sent)
            assert [t["id"] for t in sent] == list(range(1, len(sent) + 1))
            kids = collections.defaultdict(list)
            for t in sent:
                kids[t["head"]].append(t)
            for t in sent:
                if t["upos"] in NOUN:
                    n_nouns += 1; x = fv(t, XF); noun_x[x] += 1
                    for k, v in zip(XF, x):
                        missing[k] += v == "∅"
                r, tgt = classify(t, kids[t["id"]], sent)
                if not r:
                    continue
                counts[r] += 1
                a = fv(t, READS[r]); answers[r][a] += 1
                if r == "R-REL" and tgt is None:
                    unlinked_rel += 1
                if tgt is not None:
                    x = fv(tgt, XF)
                    joint[(x, r, a)] += 1
                    if x[1] == "Plur":
                        defl["plural_links"] += 1
                        if t["feats"].get("Number") == "Sing" and t["feats"].get("Gender") == "Fem":
                            defl["deflected"] += 1
    wl = sum(1 for fn in FILES for l in open(os.path.join(DATA, fn), encoding="utf-8")
             if l.strip() and not l.startswith("#") and "-" not in l.split("\t")[0] and "." not in l.split("\t")[0])
    assert wl == n_words, (wl, n_words)                                              # G-PARSE
    sets = sorted({READS[r] for r in counts if counts[r]}, key=len)
    is_chain = all(set(a) <= set(b) for a, b in zip(sets, sets[1:])) and len({len(s) for s in sets}) == len(sets)
    nnum = collections.Counter()
    for x, c in noun_x.items():
        nnum[x[1]] += c
    out = {"doc": "97 (a06fb33)", "gate_G_PARSE": True, "n_sentences": n_sent, "n_words": n_words, "n_nouns": n_nouns,
           "missing_rate": {k: missing[k] / n_nouns for k in XF},
           "noun_number_dist": {k: v / n_nouns for k, v in nnum.items()},
           "reader_counts": dict(counts), "reads": {r: list(v) for r, v in READS.items()},
           "remote_classes": sorted(REMOTE), "remote_share": sum(counts[r] for r in REMOTE) / sum(counts.values()),
           "read_sets_present": [list(s) for s in sets], "is_chain": is_chain,
           "answer_dist": {r: {"|".join(a): c for a, c in answers[r].items()} for r in READS},
           "noun_x_dist": {"|".join(x): c for x, c in noun_x.items()},
           "linked_joint": [{"x": list(x), "r": r, "a": list(a), "n": c} for (x, r, a), c in sorted(joint.items())],
           "rel_unlinked": unlinked_rel, "deflection": dict(defl),
           "data_sha256": {fn: C94.sha_file(os.path.join(DATA, fn)) for fn in FILES},
           "provenance": C94.provenance()}
    C94.write_new_json("census_exp100.json", out)
    print(json.dumps({k: out[k] for k in ("n_sentences", "n_words", "n_nouns", "reader_counts", "remote_share", "is_chain",
                                         "read_sets_present", "noun_number_dist", "deflection", "rel_unlinked", "missing_rate")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
