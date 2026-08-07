# -*- coding: utf-8 -*-
"""exp42_compiler.py -- exp41 story JSON -> circuit spec (doc 22 SS3.1).

Pure-python compilation, no torch. One qubit (wire) per referent, in the
item's referent-list order. Referent introduction = Ry(angle) from the
FROZEN 2D noun embedding (exp42_noun_embeddings.json, built once by
exp42_make_embeddings.py; source + hash recorded there and in results).
Each event applies the verb's two-wire block on (agent_wire, patient_wire)
-- agent first.

PRO-DROP (doc 22 SS2.2): the compiler consumes the dataset's GOLD events,
whose agent field is already the resolved referent even when the surface
sentence drops it. That IS the pre-registered design: structured models get
gold structure; the honest scope of the claim ("grammar-informed structure
buys order-robustness") is stated wherever results are reported.

Provenance: the dataset's canonical sha256 (json.dumps(items,
ensure_ascii=False, sort_keys=True) -- the exp41 generator/validator
convention) is recomputed here and must equal BOTH the manifest value and
the pre-registered pin below, else compilation refuses (doc 22 SS6
provenance sanity gate).
"""

import hashlib
import json
from dataclasses import dataclass, field

DATA_PATH = "stories_exp41.json"
EMB_PATH = "exp42_noun_embeddings.json"

# Pre-registered pin: hash of the validated exp41 dataset
# (results_exp41_validation_b.json, all V1-V5 passed).
EXPECTED_DATASET_SHA256 = \
    "2cb3ccd67ac0759c8d76dca432be0214325b1036556326a305a94a0cc2b23013"

LABELS = {"yes": 0, "no": 1}          # class index convention (= exp40)


def canonical_sha256(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CompiledStory:
    item_id: str
    pair_id: str
    split: str
    qtype: str
    hop_count: int
    pro_drop_item: bool
    order_dependent: bool
    K: int
    wires: dict                       # referent name -> wire index
    intro: list                       # [(wire, noun, angle)] in wire order
    events: list                      # [(verb, agent_wire, patient_wire)]
    question: tuple                   # (verb_seq, wire_1, wire_2); see below
    answer: int                       # 0 = yes, 1 = no
    tokens: list = field(default_factory=list)   # surface tokens (B2/B3 only)


def load_dataset(path=DATA_PATH):
    """Load + provenance-check the exp41 dataset. Returns (items, meta, sha)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items, meta = data["items"], data["meta"]
    sha = canonical_sha256(items)
    if sha != meta["dataset_sha256"] or sha != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "dataset provenance FAILED: canonical sha256=%s, manifest=%s, "
            "pre-registered pin=%s -- refusing to compile (doc 22 SS6)."
            % (sha, meta["dataset_sha256"], EXPECTED_DATASET_SHA256))
    return items, meta, sha


def load_embeddings(path=EMB_PATH):
    """Load the frozen noun embeddings; verify internal hash. Returns
    (angles: {noun: angle}, meta, sha256)."""
    with open(path, encoding="utf-8") as f:
        art = json.load(f)
    sha = canonical_sha256(art["embeddings"])
    if sha != art["embeddings_sha256"]:
        raise RuntimeError("embedding artifact self-hash mismatch: %s vs %s"
                           % (sha, art["embeddings_sha256"]))
    angles = {n: e["angle"] for n, e in art["embeddings"].items()}
    return angles, art["meta"], sha


def surface_tokens(item):
    """Tokens the sequence baselines (B2) see: raw surface words, sentence
    separators, then the question surface. Order variation (SVO/VSO) and
    pro-drop are REAL in this view -- by design."""
    toks = []
    for s in item["sentences"]:
        toks.extend(s.split())
        toks.append("<s>")
    toks.append("<q>")
    toks.extend(item["question"]["surface"].split())
    return toks


def compile_story(item, angles, with_tokens=False):
    wires = {r["name"]: i for i, r in enumerate(item["referents"])}
    K = len(wires)
    intro = []
    for name, w in sorted(wires.items(), key=lambda kv: kv[1]):
        if name not in angles:
            raise KeyError("no frozen embedding for referent %r" % name)
        intro.append((w, name, angles[name]))

    events = []
    for ev in item["events"]:
        a, p = ev["agent"], ev["patient"]
        assert a in wires and p in wires, \
            "event referent outside cast: %s" % (ev,)
        assert a != p, "reflexive event not in the design: %s" % (ev,)
        events.append((ev["verb"], wires[a], wires[p]))

    # Question spec = (verb_seq, wire_1, wire_2). Two schemas in exp41:
    #   Q0/Q2 "did A v P":            verb_seq (v,),     wires (a, p)
    #   Q1 "did the one who v1-ed X also v2 Y" (the two-hop bridge of doc
    #      22 SS2.2):                 verb_seq (v1, v2), wires (x, y)
    # Readout applies the verbs' own trained blocks as inverse
    # preparations on the two queried wires, in reverse sequence order,
    # then the shared trained effects (see exp42_models docstring --
    # flagged resolution of the SS3.4 question-encoding ambiguity).
    q = item["question"]
    if "a" in q:                                          # Q0 / Q2
        r1, r2, vseq = q["a"], q["p"], (q["v"],)
    else:                                                 # Q1 bridge
        r1, r2, vseq = q["x"], q["y"], (q["v1"], q["v2"])
    assert r1 in wires and r2 in wires, \
        "question referent outside cast: %s" % (q,)
    assert r1 != r2, "degenerate question pair: %s" % (q,)
    question = (vseq, wires[r1], wires[r2])

    return CompiledStory(
        item_id=item["id"], pair_id=item["pair_id"], split=item["split"],
        qtype=item["qtype"], hop_count=item["hop_count"],
        pro_drop_item=item["pro_drop_item"],
        order_dependent=item["order_dependent"],
        K=K, wires=wires, intro=intro, events=events, question=question,
        answer=LABELS[item["answer"]],
        tokens=surface_tokens(item) if with_tokens else [],
    )


def compile_all(items, angles, with_tokens=False):
    return [compile_story(it, angles, with_tokens) for it in items]


def split_items(items):
    """Standard split dict; 'test_union' is the gate set (doc 22 SS7)."""
    by = {"train": [], "val": [], "test_a": [], "test_b": [], "test_c": []}
    for it in items:
        by[it["split"]].append(it)
    by["test_union"] = by["test_a"] + by["test_b"] + by["test_c"]
    return by


def rerender_sentences(item, flip=True):
    """Re-render an item's surface from its GOLD events with SVO<->VSO
    flipped (or as-is), using the generator's 3-word frames:
      SVO: 'agent verb patient' ; VSO: 'verb agent patient' ;
      pro-drop (always verb-initial): 'verb patient'.
    Used by control C3: the compiled circuit must not change (structured
    models never see the surface), while B2's input does change.
    """
    out = dict(item)
    sents, flags = [], []
    for ev, flag in zip(item["events"], item["order_flags"]):
        v, a, p = ev["verb"], ev["agent"], ev["patient"]
        if ev.get("pro_drop", False):
            sents.append("%s %s" % (v, p))
            flags.append(flag)
            continue
        new = ("VSO" if flag == "SVO" else "SVO") if flip else flag
        sents.append("%s %s %s" % (a, v, p) if new == "SVO"
                     else "%s %s %s" % (v, a, p))
        flags.append(new)
    out["sentences"], out["order_flags"] = sents, flags
    return out
