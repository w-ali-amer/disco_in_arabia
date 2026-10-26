# -*- coding: utf-8 -*-
"""exp90_common.py -- shared plumbing for exp90 (doc 83): provenance, the
no-overwrite writer, and the SURFACE -> ENCODER -> circuit-spec compiler.

The compiler is the only path by which any exp90 learner sees case: the case
of each noun is read from its vocalised surface token by
exp90_encoder.extract_case (G11), never from stored metadata. Metadata (gold
agent/patient, intended slot cases) is used only to CROSS-CHECK the compile.
"""
import hashlib
import json
import os
import sys
from dataclasses import dataclass

import exp90_encoder as enc

REPO = os.path.dirname(os.path.abspath(__file__))
DESIGN_DOC = ("qnlp_private_docs/83_exp90_case_twins.md (REGISTERED 3446428; "
              "SS11.1 execution pins 03e5b98)")
LABELS = {"yes": 0, "no": 1}                   # exp42 convention: yes = class 0
K_CONDS = {"A": 0.5, "k0": 0.0, "knat": 0.021, "k25": 0.25}   # SS7, SS11.1
CELL = (0.40, 0.05)                             # D0.40_w0.05 (SS11.1)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True)
                          .encode("utf-8")).hexdigest()


def module_shas():
    """sha256 of every imported module whose source lives in this repo."""
    out = {}
    for name, m in sorted(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f or not f.endswith(".py") or not os.path.isabs(f):
            continue
        if os.path.dirname(f) == REPO and os.path.isfile(f):
            out[os.path.basename(f)] = file_sha256(f)
    return out


def provenance(extra_files=()):
    import platform
    import torch
    d = {"design_doc": DESIGN_DOC, "module_sha256": module_shas(),
         "python": sys.version.split()[0], "torch": torch.__version__,
         "platform": platform.platform(),
         "note_exp42_sim_adapter": ("exp42_sim_adapter.py has uncommitted "
                                    "working-tree changes (exp81-era "
                                    "unitarize path); its sha256 is recorded "
                                    "in module_sha256 (doc 83 SS11.1)")}
    d["input_file_sha256"] = {p: file_sha256(os.path.join(REPO, p))
                              for p in extra_files}
    return d


def write_new_json(path, obj):
    """Rule 3: never overwrite. Adds payload_sha256 (hash of obj without it)."""
    if os.path.exists(path):
        raise FileExistsError("refusing to overwrite existing file: %s" % path)
    obj = dict(obj)
    obj["payload_sha256"] = canonical_sha256(obj)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    if os.path.exists(path):
        os.remove(tmp)
        raise FileExistsError("refusing to overwrite existing file: %s" % path)
    os.rename(tmp, path)
    return obj["payload_sha256"]


# --------------------------------------------------------------- compiler
@dataclass
class C90:
    """Compiled exp90 item. Wires = exp60 cast order (question on (0,1))."""
    item_id: str
    pair_id: str
    item_type: str            # CONSISTENT / CONFLICT
    view: str                 # voc / unvoc
    K: int
    intro: list               # [(wire, noun, angle)]
    verb: str
    slots: tuple              # (wire of slot-1 noun, wire of slot-2 noun)
    cases: tuple              # (case of slot-1, case of slot-2) NOM/ACC/NONE
    gold: tuple               # (agent wire, patient wire) -- cross-check only
    question: tuple           # ((verb,), 0, 1)
    answer: int               # 0 yes / 1 no


def compile_surface(item, angles, verbs, view="voc"):
    """Parse the item's surface token by token: a token whose base letters are
    a verb is the verb; every other token is a noun whose wire is its cast
    index and whose case is extract_case(token). view='unvoc' parses the
    unvocalised surface (every case NONE)."""
    surface = item["surface"] if view == "voc" else item["surface_unvoc"]
    wires = {n: i for i, n in enumerate(item["referents"])}
    verb, nouns = None, []
    for tok in surface.split():
        base = enc.strip_marks(tok)
        if base in verbs:
            assert verb is None, "two verbs in %r" % surface
            verb = base
        else:
            assert base in wires, "token %r not in cast %s" % (tok, wires)
            nouns.append((wires[base], enc.extract_case(tok)))
    assert verb is not None and len(nouns) == 2, surface
    slots = (nouns[0][0], nouns[1][0])
    cases = (nouns[0][1], nouns[1][1])
    # ---- cross-checks against metadata (never the source of case)
    assert verb == item["verb"]
    assert slots == (wires[item["slots"][0]], wires[item["slots"][1]]), item["id"]
    if view == "voc":
        assert list(cases) == item["slot_cases"], (item["id"], cases)
    else:
        assert cases == (enc.NONE, enc.NONE), item["id"]
    ga, gp = wires[item["agent"]], wires[item["patient"]]
    if item["item_type"] == "CONSISTENT":
        assert slots == (ga, gp), item["id"]
    else:
        assert slots == (gp, ga), item["id"]
    intro = [(w, n, float(angles[n])) for n, w in
             sorted(wires.items(), key=lambda kv: kv[1])]
    return C90(item_id=item["id"], pair_id=item["pair_id"],
               item_type=item["item_type"], view=view, K=len(wires),
               intro=intro, verb=verb, slots=slots, cases=cases,
               gold=(ga, gp), question=((verb,), 0, 1),
               answer=LABELS[item["answer"]])
