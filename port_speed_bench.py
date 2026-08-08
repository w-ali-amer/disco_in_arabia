"""Per-epoch throughput benchmark, reusing exp43b_phase1.train_run verbatim.

Runs 2 epochs of a config already completed on the WSL box so the wall-clock is
directly comparable. Training-free of any claim: this measures the machine, not
the science. No results file is written.
"""
import sys
import time

import torch

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 1)

import exp42_compiler as comp                     # noqa: E402
import exp43b_phase1 as p1                        # noqa: E402

p1.MAX_EPOCHS = 2

angles, _, _ = comp.load_embeddings()
items, meta, _ = p1.load_rung("L1")
verbs = sorted(meta["verb_inventory"].keys())
data = {s: comp.compile_all([i for i in items if i["split"] == s], angles)
        for s in ("train", "val", "test")}
print("threads=%d  train=%d val=%d test=%d"
      % (torch.get_num_threads(), len(data["train"]), len(data["val"]),
         len(data["test"])), flush=True)

for arm, lr, batch, wsl_sec in (("A3", 0.02, 8, 35.3), ("B1", 0.02, 8, 5.3),
                                ("A3", 0.005, 1, 90.0)):
    t0 = time.time()
    p1.train_run("L1", arm, lr, batch, 1, data, verbs)
    mac_sec = (time.time() - t0) / p1.MAX_EPOCHS
    print("%s lr%g b%d : mac %.1fs/epoch vs wsl %.1fs/epoch -> %.1fx"
          % (arm, lr, batch, mac_sec, wsl_sec, wsl_sec / mac_sec), flush=True)
