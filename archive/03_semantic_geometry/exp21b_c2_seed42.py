"""Driver: run exp21b classical controls C2 for seed 42.

Seed 42 crashed pre-fix with a matmul shape error (size 24 vs 18 in
feats_concat).  The fix ([:3] truncation) is already in exp21b_robust.py.
This script replays only the classical-control path — quantum SPSA is
skipped (best_theta loaded from exp21b_encoder_seed42.npz).

Code reuse: exec(compile(head)) pattern taken from exp33_mixed_ansatz.py.
Two slices are exec'd from exp21b_robust.py:
  1. head  — everything before "ITERS = 6000" (data load, split, model build,
             encoder init, pair-sets).  This takes ~60–120 s (circuit build).
  2. cc    — the "# classical controls" block (sent_z + classical_controls).
The quantum SPSA loop (lines 337–356) and evaluation functions are never
reached; best_theta is not needed by classical_controls().
"""

import json
import os
import time

import numpy as np

os.environ.setdefault("SPLIT_SEED", "42")

PROJ = "/home/waj/discocat_arabic_v2"
os.chdir(PROJ)

src = open(f"{PROJ}/exp21b_robust.py").read()

# ── 1. exec the header (stops before the 6000-iter quantum SPSA loop) ────────
_head_marker = "ITERS = 6000"
head = src[: src.index(_head_marker)]
ns: dict = {}
print("[driver] exec-ing exp21b header (data load + model build)…", flush=True)
t_head = time.time()
exec(compile(head, "exp21b_robust.py:head", "exec"), ns)  # noqa: S102
print(f"[driver] header done in {time.time()-t_head:.1f}s", flush=True)

# ── 2. exec the classical-controls block (sent_z + classical_controls def) ───
_cc_start = src.index("# classical controls")
_cc_end   = src.index("\nw_init = weights_from(theta0)")
cc_block  = src[_cc_start:_cc_end]
exec(compile(cc_block, "exp21b_robust.py:cc", "exec"), ns)  # noqa: S102
print("[driver] classical_controls function defined", flush=True)

# ── 3. run C1 + C2 ───────────────────────────────────────────────────────────
print("[driver] running classical_controls() …", flush=True)
t_cc = time.time()
cc = ns["classical_controls"]()
elapsed = round(time.time() - t_cc, 1)
print(f"[driver] classical_controls done in {elapsed}s", flush=True)

c1_auc = cc["C1_bow"]["meaning_auc"]
c2_auc = cc["C2_matched_classical"]["meaning_auc_test"]
print(f"[driver] RESULT  C1 AUC={c1_auc:.4f}  C2 AUC={c2_auc:.4f}", flush=True)

# ── 4. sanity check ──────────────────────────────────────────────────────────
assert 0.0 <= c1_auc <= 1.0, f"C1 AUC out of range: {c1_auc}"
assert 0.0 <= c2_auc <= 1.0, f"C2 AUC out of range: {c2_auc}"

# ── 5. save JSON ─────────────────────────────────────────────────────────────
out = {
    "seed": 42,
    "C2_auc": c2_auc,
    "C1_auc": c1_auc,
    "C2_params": cc["C2_matched_classical"]["params"],
    "C2_val_margin": cc["C2_matched_classical"]["val_margin"],
    "C1_note": cc["C1_bow"]["note"],
    "runtime_sec": elapsed,
}
out_path = f"{PROJ}/results_exp21b_seed42_c2.json"
json.dump(out, open(out_path, "w"), indent=2)
print(f"[driver] wrote {out_path}", flush=True)
