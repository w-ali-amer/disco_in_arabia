# -*- coding: utf-8 -*-
"""r_gates.py -- doc 25 SSR.3 data gates (BLOCKING; self-emitting with
designed-to-fail). New file; the validators are imported, never edited.

  g1  strong V5 (validate_exp41r_data.run_v5_strong, called directly as in the
      D8/A1 drivers) passes the validator's own chance bound on BOTH the
      max-aggregation and best-sentence readings:
        L2r       two-hop test subset (hop_count >= 2; n = 150)
        family-r  all test items (test_a+b+c; n = 504)
  g2  designed-to-fail: `nodistract` builds of each spec must FAIL g1; a
      `qleak` build of each spec must FAIL V2; the August datasets are re-run
      live as positive controls (recorded values 0.9933 / 0.9901).
  g3  determinism: three processes with different PYTHONHASHSEED produce the
      identical canonical sha for both datasets (== the pins).
  g4  V1-V4 via the committed runners (validate_exp41_data.run_v1..v4), the
      exp43b_datagen.vsuite_rung wiring for the single-test-split rung.

Each gate writes r_gates/<gate>.json (Rule 3: refuses to overwrite) carrying
the input shas (datasets + code); r_gates/PINS.json records each artifact's
file sha256. If any gate fails: exit 1, and no training may run (SSR.3).
"""

import datetime
import json
import os
import subprocess
import sys

import torch

import validate_exp41r_data as v41r      # imported BEFORE any verb rebinding
import validate_exp41_data as val
import r_common as rc
import r_datagen as D

OUTDIR = "r_gates"
PY = sys.executable
CODE = ["r_gates.py", "r_datagen.py", "r_common.py", "validate_exp41r_data.py",
        "validate_exp41_data.py", "exp41_datagen.py", "exp41_story_generator.py",
        "exp43b_datagen.py", "exp43a_family_datagen.py"]
AUG_L2, AUG_FAM = "stories_exp43b_L2.json", "stories_exp43a_family.json"
DETERMINISM_SEEDS = (1, 987, 31337)


def load(path, pin=None):
    if pin:
        return rc.load_pinned(path, pin)
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["items"], d["meta"], rc.canonical_sha256(d["items"])


def v5_sets(which, items):
    tr = [i for i in items if i["split"] == "train"]
    va = [i for i in items if i["split"] == "val"]
    if which == "L2":
        te = [i for i in items if i["split"] == "test"]
        gated = [i for i in te if i["hop_count"] >= 2]
    else:
        te = [i for i in items if i["split"] not in ("train", "val")]
        gated = te
    return tr, va, te, gated


def strong_v5(which, items):
    tr, va, te, gated = v5_sets(which, items)
    r = v41r.run_v5_strong(tr, va, gated)
    return {"n_gated": len(gated), "gated_subset":
            ("two-hop test subset" if which == "L2" else "all test items"),
            "acc_max": r["test_accuracy_max_aggregation"],
            "ci95_max": r["ci95_max_aggregation"],
            "acc_best_sentence": r["test_accuracy_best_sentence"],
            "ci95_best_sentence": r["ci95_best_sentence"],
            "passed": r["passed"], "full": r}


def v12_test(which, items):
    """vsuite_rung wiring: the rung's single test split is round-robined over
    test_a/b/c for run_v1/run_v2 only (their per-split loops hard-code those
    labels); the family has real test_a/b/c."""
    tr = [i for i in items if i["split"] == "train"]
    va = [i for i in items if i["split"] == "val"]
    if which == "L2":
        te = [i for i in items if i["split"] == "test"]
        te = [dict(i, split="test_" + "abc"[n % 3]) for n, i in enumerate(te)]
    else:
        te = [i for i in items if i["split"] not in ("train", "val")]
    return tr, va, te


def emit(name, payload, pins):
    payload = dict(payload)
    payload["gate_artifact"] = name
    payload["registration"] = rc.REGISTRATION
    payload["timestamp"] = datetime.datetime.now().isoformat()
    payload["code_sha256"] = {p: rc.file_sha256(p) for p in CODE}
    payload["dataset_pins"] = {"L2r": rc.L2R_SHA256, "familyr": rc.FAMR_SHA256}
    path = rc.rule3(os.path.join(OUTDIR, name + ".json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    pins[name] = {"path": path, "file_sha256": rc.file_sha256(path),
                  "passed": payload["passed"]}
    print("[r_gates] %-4s passed=%s  %s  sha256=%s"
          % (name, payload["passed"], path, pins[name]["file_sha256"]),
          flush=True)


def main():
    torch.set_num_threads(1)
    os.makedirs(OUTDIR, exist_ok=True)
    rc.rule3(os.path.join(OUTDIR, "PINS.json"))
    pins = {}
    l2, _ml2, _ = load(rc.L2R_DATA, rc.L2R_SHA256)
    fam, _mf, _ = load(rc.FAMR_DATA, rc.FAMR_SHA256)
    real = {"L2r": ("L2", l2), "familyr": ("family", fam)}

    # ------------------------------------------------------------- g1
    g1 = {w: strong_v5(k, its) for w, (k, its) in real.items()}
    emit("g1", {"criterion": "strong V5 chance bound (acc <= 0.55 AND 95% CI "
                             "contains 0.5) on BOTH max-aggregation and "
                             "best-sentence",
                "results": g1,
                "passed": bool(all(r["passed"] for r in g1.values()))}, pins)

    # ------------------------------------------------------------- g2
    g2 = {"nodistract_must_fail_g1": {}, "qleak_must_fail_V2": {},
          "august_positive_controls": {}}
    for w, (k, _its) in real.items():
        nd, _h = D.build(w, sabotage="nodistract")
        r = strong_v5(k, nd)
        r["dataset_sha256_in_memory"] = rc.canonical_sha256(nd)
        r["fired"] = not r["passed"]
        g2["nodistract_must_fail_g1"][w] = r
        ql, _h = D.build(w, sabotage="qleak")
        tr, va, te = v12_test(k, ql)
        v2 = val.run_v2(tr, va, te)
        g2["qleak_must_fail_V2"][w] = {
            "dataset_sha256_in_memory": rc.canonical_sha256(ql),
            "V2_test_accuracy": v2["test_accuracy"], "V2_ci95": v2["ci95"],
            "V2_passed": v2["passed"], "fired": not v2["passed"]}
    for w, (k, path, recorded) in {
            "august_L2": ("L2", AUG_L2, 0.9933),
            "august_family": ("family", AUG_FAM, 0.9901)}.items():
        its, _m, sha = load(path)
        r = strong_v5(k, its)
        r["dataset_sha256"] = sha
        r["recorded_value_doc82"] = recorded
        r["fired"] = not r["passed"]
        g2["august_positive_controls"][w] = r
    fired = ([r["fired"] for r in g2["nodistract_must_fail_g1"].values()]
             + [r["fired"] for r in g2["qleak_must_fail_V2"].values()]
             + [r["fired"] for r in g2["august_positive_controls"].values()])
    g2["passed"] = bool(all(fired))
    g2["criterion"] = ("every designed-to-fail build and every August positive "
                       "control must FAIL its check; a build that passes "
                       "indicts the validator")
    emit("g2", g2, pins)

    # ------------------------------------------------------------- g3
    runs = []
    for hs in DETERMINISM_SEEDS:
        env = dict(os.environ, PYTHONHASHSEED=str(hs))
        out = subprocess.run([PY, "r_datagen.py", "--sha-only"], env=env,
                             capture_output=True, text=True, check=True)
        runs.append({"PYTHONHASHSEED": hs,
                     "shas": json.loads(out.stdout.strip().splitlines()[-1])})
    ok = all(r["shas"] == {"L2r": rc.L2R_SHA256, "familyr": rc.FAMR_SHA256}
             for r in runs)
    emit("g3", {"criterion": "3 processes, different PYTHONHASHSEED, identical "
                             "canonical sha == pin",
                "runs": runs, "passed": bool(ok)}, pins)

    # ------------------------------------------------------------- g4
    g4 = {}
    for w, (k, its) in real.items():
        meta = _ml2 if w == "L2r" else _mf
        verbs = sorted(meta["verb_inventory"].keys())
        referents = meta["humans"] + meta["objects"]
        tr, va, te = v12_test(k, its)
        v1 = val.run_v1(tr, va, te)
        v2 = val.run_v2(tr, va, te)
        v3 = val.run_v3(its, verbs, referents)
        v4 = val.run_v4(its, set(referents))
        for r in (v1, v2):
            r["per_split_note"] = ("L2r: single test split round-robined over "
                                   "test_a/b/c (vsuite_rung wiring); thirds "
                                   "are arbitrary partitions" if k == "L2"
                                   else "family: real test_a/b/c")
        g4[w] = {"V1": v1, "V2": v2, "V3": v3, "V4": v4,
                 "passed": bool(all(x["passed"] for x in (v1, v2, v3, v4)))}
    g4_all = bool(all(r["passed"] for r in g4.values()))
    emit("g4", {"criterion": "V1-V4 pass via the committed runners",
                "results": g4, "passed": g4_all}, pins)

    all_ok = all(p["passed"] for p in pins.values())
    with open(rc.rule3(os.path.join(OUTDIR, "PINS.json")), "w",
              encoding="utf-8") as f:
        json.dump({"artifacts": pins, "all_gates_passed": bool(all_ok),
                   "consequence": ("training may proceed (Stage 2, separately "
                                   "authorised)" if all_ok else
                                   "doc 25 SSR.3: a gate failed -- NO training; "
                                   "the failure is the finding")},
                  f, ensure_ascii=False, indent=1)
    print("[r_gates] ALL GATES PASSED = %s" % all_ok, flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
