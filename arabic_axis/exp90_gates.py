# -*- coding: utf-8 -*-
"""exp90_gates.py -- doc 83 SS11 / doc 70 SS3 rule 6: self-emitting,
hash-pinned gate artifacts, each with a designed-to-fail case shown FAILING.

    python3 exp90_gates.py {vsuite,encoder,a0,t4,t5,all} [--outdir exp90_gates]

Each gate writes <outdir>/<gate>.json (refuses to overwrite; the orchestrator
re-executes with a fresh --outdir). A gate PASSES iff its real assertions all
hold AND its designed-to-fail assertion FAILS.
"""
import argparse
import copy
import json
import math
import os
import random
import sys

import torch

torch.set_num_threads(1)

import exp60_run as e60                                   # noqa: E402
import exp42_compiler as comp                             # noqa: E402
import exp90_encoder as enc                               # noqa: E402
import exp90_encoder_tests as enc_tests                   # noqa: E402
from exp90_common import (CELL, K_CONDS, compile_surface, file_sha256,  # noqa: E402
                          provenance, write_new_json)
from exp90_models import ArchM, ArchR                     # noqa: E402

DRAW_FILES = ["stories_exp90_primary.json", "stories_exp90_confirm.json",
              "stories_exp90_A0_61000.json"]
A0_BASE = "stories_exp60_SB_confirm.json"
A0_RENDER = "stories_exp90_A0_61000.json"
A0_TOL = 1e-6
ZERO_TOL = 1e-9          # "exactly 0" in float32 autograd (raw values recorded)
NONZERO_MIN = 1e-4


def _load(p):
    return json.load(open(p, encoding="utf-8"))


def _angles(meta, seed=0):
    return e60.build_angles("S-base", CELL[0], CELL[1], seed,
                            set(meta["bearers"]), meta["humans"])


# ======================================================================= V
def p_circuit(c):
    """What the vowel-blind P arm consumes: everything except the cases."""
    return (tuple(c.intro), c.verb, tuple(c.slots), c.question, c.K)


def p_circuit_case_leak(c):
    """DESIGNED-TO-FAIL circuit view: a P that (wrongly) also sees the cases."""
    return p_circuit(c) + (tuple(c.cases),)


def t1_twin_identity(blob, seeds=(0, 1, 11), circuit=p_circuit):
    """For every test pair: P's compiled circuit for the CONFLICT twin equals
    its CONSISTENT twin's exactly, and labels are opposite."""
    meta, test = blob["meta"], blob["test"]
    byid = {i["id"]: i for i in test}
    fails, n = [], 0
    for s in seeds:
        ang = _angles(meta, s)
        for it in test:
            if it["item_type"] != "CONFLICT":
                continue
            tw = byid.get(it.get("twin_of"))
            if tw is None or tw["item_type"] != "CONSISTENT":
                fails.append((it["id"], "no CONSISTENT twin")); continue
            a = compile_surface(it, ang, meta["verbs"])
            b = compile_surface(tw, ang, meta["verbs"])
            n += 1
            if circuit(a) != circuit(b):
                fails.append((it["id"], "circuit differs"))
            if a.answer == b.answer:
                fails.append((it["id"], "label not flipped"))
            if a.pair_id != b.pair_id:
                fails.append((it["id"], "pair_id differs"))
    n_conf = sum(1 for i in test if i["item_type"] == "CONFLICT")
    n_cons = sum(1 for i in test if i["item_type"] == "CONSISTENT")
    twins = [i.get("twin_of") for i in test if i["item_type"] == "CONFLICT"]
    bijective = len(set(twins)) == len(twins) == n_cons
    return {"n_pairs_checked": n, "n_conflict": n_conf, "n_consistent": n_cons,
            "twin_map_bijective": bijective, "n_fail": len(fails),
            "first_fails": fails[:5],
            "passed": bool(not fails and bijective and n_conf == n_cons)}


def role_balance(blob):
    meta, test = blob["meta"], blob["test"]
    bearers = set(meta["bearers"])
    out, ok = {}, True
    for ty in ("CONSISTENT", "CONFLICT"):
        its = [i for i in test if i["item_type"] == ty]
        yes = sum(1 for i in its if i["answer"] == "yes")
        b_slot1 = sum(1 for i in its if i["slots"][0] in bearers)
        b_wire0 = sum(1 for i in its if i["referents"][0] in bearers)
        roles = {}
        for i in its:
            roles.setdefault((frozenset(i["referents"]), i["verb"]), set()).add(
                (i["agent"], i["patient"]))
        both = all(len(v) == 2 for v in roles.values())
        verbs = {}
        for i in its:
            verbs.setdefault(i["verb"], [0, 0])[i["answer"] == "yes"] += 1
        vbal = all(a == b for a, b in verbs.values())
        good = (yes * 2 == len(its) and b_slot1 * 2 == len(its) and both and vbal)
        ok &= good
        out[ty] = {"n": len(its), "n_yes": yes, "bearer_in_slot1": b_slot1,
                   "bearer_on_wire0": b_wire0,
                   "every_cast_verb_in_both_role_assignments": both,
                   "verb_answer_balanced": vbal, "passed": good}
    tr = {}
    for cond, its in blob["train_sets"].items():
        t = [i for i in its if i["split"] == "train"]
        tr[cond] = {ty: {"n": sum(1 for i in t if i["item_type"] == ty),
                         "n_yes": sum(1 for i in t if i["item_type"] == ty
                                      and i["answer"] == "yes")}
                    for ty in ("CONSISTENT", "CONFLICT")}
        tr[cond]["n_yes_total"] = sum(1 for i in t if i["answer"] == "yes")
        ok &= tr[cond]["n_yes_total"] * 2 == len(t)
    return {"test": out, "train_descriptive": tr, "passed": bool(ok)}


def no_event_both_renderings(blob):
    fails = []
    base_ids = None
    for cond, its in blob["train_sets"].items():
        seen = {}
        for i in its:
            seen.setdefault(i["base_id"], []).append(i["item_type"])
        dup = [b for b, v in seen.items() if len(v) != 1]
        if dup:
            fails.append((cond, "event rendered %d times" % len(dup), dup[:3]))
        if base_ids is None:
            base_ids = set(seen)
        elif set(seen) != base_ids:
            fails.append((cond, "event set differs across doses"))
        # the same (event) must not appear as both CONS and CONF
        both = [b for b, v in seen.items() if set(v) == {"CONSISTENT", "CONFLICT"}]
        if both:
            fails.append((cond, "both renderings", both[:3]))
    # nesting (common random numbers) -- descriptive invariant of the rule
    conf = {c: {i["base_id"] for i in its if i["item_type"] == "CONFLICT"}
            for c, its in blob["train_sets"].items()}
    nested = conf["k0"] <= conf["knat"] <= conf["k25"] <= conf["A"]
    if not nested:
        fails.append(("nesting", "doses not nested"))
    # test events never in train (held-out inventory)
    held = set(blob["meta"]["heldout_inventory"])
    leak = [i["id"] for its in blob["train_sets"].values() for i in its
            if set(i["referents"]) & held]
    if leak:
        fails.append(("heldout_leak", leak[:3]))
    return {"n_fail": len(fails), "fails": fails[:5], "doses_nested": nested,
            "passed": not fails}


def conf_equals_other_twin(blob):
    """Each test CONF item is the CONFLICT rendering of the other exp60 twin
    of its pair (same event, patient first): a cross-check of construction."""
    test = blob["test"]
    cons = {(i["pair_id"], i["agent"], i["patient"]): i for i in test
            if i["item_type"] == "CONSISTENT"}
    bad = 0
    for i in test:
        if i["item_type"] != "CONFLICT":
            continue
        o = cons.get((i["pair_id"], i["agent"], i["patient"]))
        if o is None or o["answer"] != i["answer"] or \
                i["slots"] != [i["patient"], i["agent"]]:
            bad += 1
    return {"n_bad": bad, "passed": bad == 0}


def encoder_crosscheck_all(blob):
    """Every surface (voc + unvoc, test + every train set) compiles through the
    encoder and matches the intended slots/cases (compile_surface asserts)."""
    meta = blob["meta"]
    ang = _angles(meta, 0)
    n = 0
    for i in blob["test"]:
        compile_surface(i, ang, meta["verbs"], "voc")
        compile_surface(i, ang, meta["verbs"], "unvoc")
        n += 2
    for its in blob["train_sets"].values():
        for i in its:
            compile_surface(i, ang, meta["verbs"], "voc"); n += 1
    return {"n_compiled": n, "passed": True}


def gate_vsuite():
    per = {}
    ok = True
    for f in DRAW_FILES:
        blob = _load(f)
        r = {"T1_twin_circuit_identity_and_label_flip": t1_twin_identity(blob),
             "role_balance": role_balance(blob),
             "no_event_in_both_renderings": no_event_both_renderings(blob),
             "conf_is_conflict_rendering_of_other_twin": conf_equals_other_twin(blob),
             "encoder_crosscheck_all_surfaces": encoder_crosscheck_all(blob),
             "counts": blob["meta"]["counts"]}
        r["passed"] = all(v["passed"] for k, v in r.items() if k != "counts")
        ok &= r["passed"]
        per[f] = r
    # ---------------- designed-to-fail cases (primary draw)
    blob = _load(DRAW_FILES[0])
    # (a) twin built from the WRONG pair: re-point twin_of cyclically
    bad = copy.deepcopy(blob)
    confs = [i for i in bad["test"] if i["item_type"] == "CONFLICT"]
    tw = [i["twin_of"] for i in confs]
    for i, t in zip(confs, tw[1:] + tw[:1]):
        i["twin_of"] = t
    dtf_a = t1_twin_identity(bad, seeds=(0,))
    # (b) a P arm that leaks the case vowels: T1 identity must break
    dtf_b = t1_twin_identity(blob, seeds=(0,), circuit=p_circuit_case_leak)
    # (c) one event rendered both ways in the k25 set
    bad3 = copy.deepcopy(blob)
    it = next(i for i in bad3["train_sets"]["k25"] if i["item_type"] == "CONSISTENT")
    dup = copy.deepcopy(it); dup["item_type"] = "CONFLICT"
    bad3["train_sets"]["k25"].append(dup)
    dtf_c = no_event_both_renderings(bad3)
    dtf = {"wrong_pair_twins_T1": dtf_a, "P_with_case_leak_T1": dtf_b,
           "event_in_both_renderings": dtf_c}
    dtf_shown = all(not v["passed"] for v in dtf.values())
    return {"gate": "V-suite", "per_draw": per, "real_all_pass": bool(ok),
            "designed_to_fail": dtf, "designed_to_fail_shown_failing": dtf_shown,
            "passed": bool(ok and dtf_shown),
            "provenance": provenance(extra_files=DRAW_FILES)}


def gate_encoder():
    g = enc_tests.gate()
    g["gate"] = "encoder"
    g["provenance"] = provenance()
    return g


# ====================================================================== A0
def a0_compare(a_nom, a_acc, seeds=range(5)):
    base = _load(A0_BASE)
    bmeta = base["meta"]
    rblob = _load(A0_RENDER)
    verbs = sorted(bmeta["verb_inventory"])
    bt = {i["id"]: i for i in base["items"] if i["split"] == "test"}
    gold_of = {(v["pair_id"], v["events"][0]["agent"]): k for k, v in bt.items()}
    per = {}
    for s in seeds:
        pt = "exp60_ckpt/CONF_S-base_D0.40_w0.05_s%d.pt" % s
        side = _load(pt.replace(".pt", ".json"))
        sha = file_sha256(pt)
        assert sha == side["ckpt_sha256"], "ckpt sha mismatch %s" % pt
        sd = torch.load(pt)
        ang = e60.build_angles("S-base", CELL[0], CELL[1], s,
                               set(bmeta["bearers"]), bmeta["humans"])
        ref = e60.SBModel("A3", verbs, s)
        ref.load_state_dict(sd)
        cref = {c.item_id: c for c in comp.compile_all(list(bt.values()), ang)}
        m = ArchR(verbs, s, init="construct")
        r = m.load_state_dict(sd, strict=False)
        assert sorted(r.missing_keys) == ["a_acc", "a_nom", "router.theta"], r
        assert not r.unexpected_keys, r
        with torch.no_grad():
            m.a_nom.fill_(a_nom); m.a_acc.fill_(a_acc)
            M = m.mats()
            mx = {"CONSISTENT": 0.0, "CONFLICT": 0.0}
            acc = {"CONSISTENT": 0, "CONFLICT": 0}
            n = {"CONSISTENT": 0, "CONFLICT": 0}
            for it in rblob["test"]:
                c = compile_surface(it, ang, verbs)
                g = gold_of[(it["pair_id"], it["agent"])]
                vr = ref.story_values(cref[g])
                vm = m.forward_one(c, M)
                ty = it["item_type"]
                mx[ty] = max(mx[ty], float((vm - vr).abs().max()))
                acc[ty] += int(int(torch.argmax(vm)) == c.answer)
                n[ty] += 1
        per["s%d" % s] = {"ckpt": pt, "ckpt_sha256": sha,
                          "max_abs_delta": mx,
                          "max_abs_delta_both": max(mx.values()),
                          "n_items": n,
                          "archR_acc": {k: acc[k] / n[k] for k in acc},
                          "exp60_recorded_heldout": side["heldout_test_acc"],
                          "passed": max(mx.values()) <= A0_TOL}
    return per


def gate_a0():
    per = a0_compare(0.0, math.pi)
    ok = all(v["passed"] for v in per.values())
    # designed-to-fail: ACC box at the NOM point (router never fires on
    # CONFLICT items) must break the anchor
    bad = a0_compare(0.0, 0.0, seeds=[0])
    dtf = {"construction": "a_nom=0, a_acc=0 (router never fires)",
           "per_ckpt": bad, "passed": all(v["passed"] for v in bad.values())}
    return {"gate": "A0", "tolerance": A0_TOL,
            "construction_point": {"a_nom": 0.0, "a_acc": math.pi,
                                   "router": "KAK identity locals, Cartan (pi/4,pi/4,pi/4) = e^{i pi/4} SWAP"},
            "per_checkpoint": per, "real_all_pass": ok,
            "anchor_bit_exact": ok,
            "designed_to_fail": dtf,
            "designed_to_fail_shown_failing": not dtf["passed"],
            "passed": bool(ok and not dtf["passed"]),
            "provenance": provenance(extra_files=[A0_BASE, A0_RENDER] + [
                "exp60_ckpt/CONF_S-base_D0.40_w0.05_s%d.pt" % s for s in range(5)])}


# ================================================================== T4 / T5
def _batch(cond, draw="stories_exp90_primary.json", seed=0, first_batch=False):
    blob = _load(draw)
    meta = blob["meta"]
    ang = _angles(meta, seed)
    tr = [compile_surface(i, ang, meta["verbs"]) for i in blob["train_sets"][cond]
          if i["split"] == "train"]
    if first_batch:                      # the exact step-0 batch of a run
        order = list(range(len(tr)))
        random.Random(55000 + seed).shuffle(order)
        tr = [tr[i] for i in order[:8]]
    return tr, meta["verbs"]


def _grads(model, batch):
    model.zero_grad()
    model.loss_batch(batch).backward()
    return {n: (0.0 if p.grad is None else float(p.grad.norm()))
            for n, p in model.named_parameters()}


def _box_router(g):
    return {"router": g.get("router.theta", None), "nom": g.get("a_nom", g.get("nom")),
            "acc": g.get("a_acc", g.get("acc"))}


def gate_t4(seeds=range(12)):
    rows, ok = [], True
    for s in seeds:
        for cond in ("k0", "A"):                 # k0 registered; A = "every dose"
            for fb in (True, False):
                batch, verbs = _batch(cond, seed=s, first_batch=fb)
                torch.manual_seed(s)
                g = _box_router(_grads(ArchR(verbs, s, init="pure"), batch))
                good = all(abs(v) <= ZERO_TOL for v in g.values())
                ok &= good
                rows.append({"arm": "R-pure", "seed": s, "cond": cond,
                             "batch": "first_step0_batch" if fb else "full_train",
                             "grad_norms": g, "all_zero": good})
    # R-und on CONSISTENT items: slot-1 box (NOM) != 0, slot-2 box (ACC) == 0
    und = []
    for s in seeds:
        batch, verbs = _batch("k0", seed=s)
        torch.manual_seed(s)
        g = _box_router(_grads(ArchR(verbs, s, init="und"), batch))
        good = g["nom"] >= NONZERO_MIN and abs(g["acc"]) <= ZERO_TOL
        ok &= good
        und.append({"arm": "R-und", "seed": s, "cond": "k0 (all CONSISTENT)",
                    "grad_norms": g, "slot1_nonzero_slot2_zero": good})
    # also through the REAL exp60 readout (A0 checkpoint s0 blocks/effects)
    batch, verbs = _batch("k0", seed=0)
    m = ArchR(verbs, 0, init="und")
    m.load_state_dict(torch.load("exp60_ckpt/CONF_S-base_D0.40_w0.05_s0.pt"),
                      strict=False)
    g_real = _box_router(_grads(m, batch))
    real_ok = g_real["nom"] >= NONZERO_MIN and abs(g_real["acc"]) <= ZERO_TOL
    ok &= real_ok
    # designed-to-fail: run the "all zero" assertion on R-und's slot-1 box
    batch, verbs = _batch("k0", seed=0)
    g_d = _box_router(_grads(ArchR(verbs, 0, init="und"), batch))
    dtf_pass = abs(g_d["nom"]) <= ZERO_TOL
    return {"gate": "T4", "zero_tol": ZERO_TOL, "nonzero_min": NONZERO_MIN,
            "r_pure_rows": rows, "r_und_rows": und,
            "r_und_exp60_ckpt_s0_readout": {"grad_norms": g_real, "passed": real_ok,
                                           "note": "design preview reported 0.268 for the slot-1 box under its parameterisation; checked != 0 only"},
            "real_all_pass": bool(ok), "t4_stationary_R": bool(all(r["all_zero"] for r in rows)),
            "designed_to_fail": {"assertion": "R-und slot-1 (NOM) box grad == 0",
                                 "value": g_d["nom"], "passed": dtf_pass},
            "designed_to_fail_shown_failing": not dtf_pass,
            "passed": bool(ok and not dtf_pass),
            "provenance": provenance(extra_files=[DRAW_FILES[0]])}


def gate_t5(seeds=range(12)):
    rows, ok = [], True
    for s in seeds:
        batch, verbs = _batch("k0", seed=s)
        for init in ("id", "rand"):
            torch.manual_seed(s)
            g = _box_router(_grads(ArchM(verbs, s, init=init), batch))
            good = g["acc"] >= NONZERO_MIN
            ok &= good
            rows.append({"arm": "M-" + init, "seed": s, "grad_norms": g,
                         "acc_grad_nonzero": good})
    batch, verbs = _batch("k0", seed=0)
    g_d = _box_router(_grads(ArchR(verbs, 0, init="und"), batch))
    dtf_pass = g_d["acc"] >= NONZERO_MIN
    return {"gate": "T5", "nonzero_min": NONZERO_MIN, "rows": rows,
            "real_all_pass": bool(ok),
            "designed_to_fail": {"assertion": "Arch-R (R-und) ACC grad != 0 at k=0 (T4 says it is 0)",
                                 "value": g_d["acc"], "passed": dtf_pass},
            "designed_to_fail_shown_failing": not dtf_pass,
            "passed": bool(ok and not dtf_pass),
            "provenance": provenance(extra_files=[DRAW_FILES[0]])}


GATES = {"vsuite": gate_vsuite, "encoder": gate_encoder, "a0": gate_a0,
         "t4": gate_t4, "t5": gate_t5}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gate", choices=sorted(GATES) + ["all"])
    ap.add_argument("--outdir", default="exp90_gates")
    args = ap.parse_args()
    names = ["vsuite", "encoder", "a0", "t4", "t5"] if args.gate == "all" else [args.gate]
    rc = 0
    for n in names:
        path = os.path.join(args.outdir, "%s.json" % n)
        if os.path.exists(path):
            raise SystemExit("refusing to overwrite %s (use a fresh --outdir)" % path)
        r = GATES[n]()
        sha = write_new_json(path, r)
        print("[gate %-7s] %s  real=%s  designed-to-fail shown failing=%s  -> %s (%s)"
              % (n, "PASS" if r["passed"] else "FAIL", r.get("real_all_pass", r.get("real_all_pass")),
                 r["designed_to_fail_shown_failing"], path, sha[:12]), flush=True)
        rc |= (not r["passed"])
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
