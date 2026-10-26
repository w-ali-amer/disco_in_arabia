# -*- coding: utf-8 -*-
"""exp60_assemble.py -- doc 46 SS10.3/10.4/10.5 verdicts.

  sb_learnable          train >= 0.8 and val >= 0.7            (descriptive, per cell)
  sb_binds              held-out >= 0.65 AND exceeds S-shuffle's held-out mean
                        at the SAME cell by >= 2 SD of its seed distribution
  reuploading_relieves  S-reup binds at a cell where S-base does not
  arm validity (SS10.5) S-base must close in the 3-term Fourier basis;
                        S-reup must NOT (it needs 5), else the re-uploading
                        arm did not implement the intervention
"""
import glob, json, math, os, platform, sys, time
from collections import defaultdict
import torch

import exp42_compiler as comp
import exp60_run as R
from exp42_run import fresh_results_path, git_note

SB_ABS = R.SB_BINDS_ABS


def load(tag):
    out = defaultdict(list)
    for f in sorted(glob.glob("exp60_ckpt/*.json")):
        base = os.path.basename(f)
        if tag and not base.startswith(tag):
            continue
        if not tag and base.startswith("CONF_"):
            continue
        r = json.load(open(f, encoding="utf-8"))
        out[(r["cell"], r["arm"])].append(r)
    return out


def stats(rs, key="heldout_test_acc"):
    v = [r[key] for r in rs]
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)) ** .5
    return m, sd, v


def verdicts_for(by, stage):
    cells = sorted({c for c, _ in by}, key=lambda c: (float(c.split("_w")[1]),
                                                      float(c[1:].split("_")[0])))
    rows, v = [], {}
    for c in cells:
        row = {"cell": c}
        fl = by.get((c, "S-shuffle"))
        fm, fsd, _ = stats(fl) if fl else (None, None, None)
        row["floor_mean"], row["floor_sd"] = fm, fsd
        for arm in ("S-base", "S-reup"):
            rs = by.get((c, arm))
            if not rs:
                continue
            m, sd, vals = stats(rs)
            gate = sum(1 for r in rs if r["gate_met"])
            binds = (fm is not None and m >= SB_ABS
                     and m >= fm + 2 * (fsd if fsd else 0.0))
            row[arm] = {"heldout_mean": m, "heldout_sd": sd, "per_seed": vals,
                        "n_seeds": len(rs), "gate_met": "%d/%d" % (gate, len(rs)),
                        "exceeds_floor_by_sd": (None if fm is None or not fsd
                                                else (m - fm) / fsd),
                        "sb_binds": bool(binds)}
        rows.append(row)
    base_ok = [r["cell"] for r in rows if r.get("S-base", {}).get("sb_binds")]
    reup_ok = [r["cell"] for r in rows if r.get("S-reup", {}).get("sb_binds")]
    v["%s_sb_binds_S-base_cells" % stage] = base_ok
    v["%s_sb_binds_S-reup_cells" % stage] = reup_ok
    v["%s_sb_binds" % stage] = bool(base_ok or reup_ok)
    v["%s_reuploading_relieves" % stage] = bool(set(reup_ok) - set(base_ok))
    v["%s_n_cells_tested" % stage] = len(rows)
    return rows, v


def fourier_validity():
    """doc 46 SS10.5: S-base closes at 3 terms; S-reup must not."""
    blob = json.load(open(R.DATA, encoding="utf-8"))
    items, meta = blob["items"], blob["meta"]
    verbs = sorted(meta["verb_inventory"].keys())
    ang = R.build_angles("S-base", 0.4, 0.05, 0, set(meta["bearers"]),
                         meta["humans"])
    cs = comp.compile_all([i for i in items if i["split"] == "test"][:4], ang)
    out = {}
    for arm, pt in (("S-base", "exp60_ckpt/S-base_D0.40_w0.05_s0.pt"),
                    ("S-reup", "exp60_ckpt/S-reup_D0.40_w0.05_s0.pt")):
        if not os.path.exists(pt):
            continue
        m = R.SBModel("A3", verbs, 0, reup=(arm == "S-reup"))
        m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
        out[arm] = {"resid_3term": R.fourier_rank(m, cs, 3),
                    "resid_5term": R.fourier_rank(m, cs, 5),
                    "kappa": (float(m.kappa[0].detach())
                              if arm == "S-reup" else None)}
    if "S-base" in out and "S-reup" in out:
        out["S-base_closes_at_3"] = bool(out["S-base"]["resid_3term"] < 1e-5)
        out["S-reup_needs_more_than_3"] = bool(out["S-reup"]["resid_3term"] > 1e-5)
        out["S-reup_closes_at_5"] = bool(out["S-reup"]["resid_5term"] < 1e-5)
        out["arm_implements_intervention"] = bool(
            out["S-reup_needs_more_than_3"] and out["S-reup_closes_at_5"])
    return out


def main():
    t0 = time.time()
    screen_rows, sv = verdicts_for(load(""), "screen")
    conf_by = load("CONF_")
    conf_rows, cv = verdicts_for(conf_by, "confirm") if conf_by else ([], {})
    fv = fourier_validity()

    ctrl = {}
    for arm in ("S-flat", "S-hash"):
        rs = [r for (c, a), v in load("").items() if a == arm for r in v]
        if rs:
            m, sd, vals = stats(rs)
            ctrl[arm] = {"heldout_mean": m, "heldout_sd": sd, "per_seed": vals,
                         "must_fail": True, "fails": bool(m < SB_ABS)}

    verdicts = {}
    verdicts.update(sv)
    verdicts.update(cv)
    verdicts["fourier_arm_validity"] = fv.get("arm_implements_intervention")
    verdicts["controls_fail_as_required"] = bool(
        all(c["fails"] for c in ctrl.values())) if ctrl else None
    # doc 46 SS10.4: a screened-positive cell is CLAIMED only if it replicates
    verdicts["sb_binds_CONFIRMED"] = bool(cv.get("confirm_sb_binds")) if cv else None
    verdicts["reuploading_relieves_CONFIRMED"] = (
        bool(cv.get("confirm_reuploading_relieves")) if cv else None)

    out = {"experiment": "exp60", "spec": R.DESIGN_DOC,
           "platform": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "torch_num_threads": torch.get_num_threads(),
           "screen_dataset_sha256": json.load(open(R.DATA, encoding="utf-8"))
               ["meta"]["dataset_sha256"],
           "confirm_dataset_sha256": (json.load(open(
               "stories_exp60_SB_confirm.json", encoding="utf-8"))["meta"]
               ["dataset_sha256"] if os.path.exists(
               "stories_exp60_SB_confirm.json") else None),
           "grid": [R.cell_id(d, w) for d, w in R.GRID],
           "config": {"lr": R.LR, "batch": R.BATCH, "epochs": R.MAX_EPOCHS,
                      "screen_seeds": R.SCREEN_SEEDS,
                      "confirm_seeds": R.CONFIRM_SEEDS,
                      "screen_bar": R.SCREEN_BAR, "sb_binds_abs": SB_ABS},
           "screen": screen_rows, "confirm": conf_rows,
           "controls": ctrl, "fourier_validity": fv, "verdicts": verdicts,
           "meta_notes": {
               "floor": ("S-shuffle is THE registered floor (doc 46 SS10.2): "
                         "angle draws bit-for-bit identical to S-base, bearer "
                         "designation reassigned PER PAIR independently of the "
                         "angle, so only the angle<->category correlation is "
                         "removed. S-hash was REJECTED as a floor -- its "
                         "sha256 angles are ~U(0,2pi), so it would fail for "
                         "conditioning reasons regardless of binding."),
               "tie_break_disclosed": (
                   "All 8 cells screened positive, so doc 46 SS10.4's top-2 rule "
                   "ties. DISCLOSED TIE-BREAK, chosen before the confirm ran: "
                   "advance the two SMALLEST-Delta cells (smallest category "
                   "margin 2*sin(Delta/2)) -- the theorem's tightest test, and "
                   "the only cells that were not perfect in the screen."),
               "prediction_status": (
                   "doc 46 SS4 predicted S-base would FAIL across the grid. If "
                   "it binds, that prediction is REFUTED and the reason is "
                   "instructive: the theorem supplies ~3 Fourier coefficients "
                   "per referent and a 1-BIT category demands one. The registered "
                   "prediction mis-estimated the DEMAND, not the supply."),
               "scope": ("doc 46 SS13: one task family, one arm family, "
                         "simulation, NO classical comparator registered. "
                         "Classical models already fail novel-filler binding "
                         "(arXiv:1902.09006), so nothing here is quantum-"
                         "specific. A positive binds a 1-BIT CONSTRUCTED "
                         "category to a role -- the honest minimum viable "
                         "semantics, NOT 'meaning'.")},
           "git_note": git_note(), "wall_time_sec": time.time() - t0}
    p = fresh_results_path("results_exp60")
    assert "exp60" in p
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[exp60] VERDICTS")
    for k, v in verdicts.items():
        print("   %-42s %s" % (k, v))
    print("\n[exp60] wrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
