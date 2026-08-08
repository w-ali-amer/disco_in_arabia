"""Cross-machine parity witness for the exp41-43 line (WSL <-> Mac).

Training-free. Prints a canonical digest of every trainable tensor at a fixed
seed, plus dataset digests, so the two boxes can be compared exactly. No verdict:
the acceptance criteria live in qnlp_private_docs/24.
"""
import hashlib
import json
import platform
import sys

import torch

torch.set_num_threads(1)

import exp42_compiler as comp                                # noqa: E402
from exp42_baselines import ClassicalDisCoCirc               # noqa: E402
from exp42_models import QuantumStoryModel                   # noqa: E402


def digest(model):
    h = hashlib.sha256()
    for name, prm in sorted(model.named_parameters()):
        h.update(name.encode())
        t = prm.detach()
        t = torch.view_as_real(t) if t.is_complex() else t
        h.update(t.to(torch.float64).numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    out = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    with open("stories_exp43b_L1.json", encoding="utf-8") as f:
        d = json.load(f)
    out["L1_sha256"] = comp.canonical_sha256(d["items"])
    out["L1_sha256_matches_meta"] = (
        out["L1_sha256"] == d["meta"]["dataset_sha256"])
    verbs = sorted(d["meta"]["verb_inventory"].keys())
    _, _, emb_sha = comp.load_embeddings()
    out["embeddings_sha256"] = emb_sha

    out["init_digests"] = {}
    for seed in (0, 1):
        torch.manual_seed(seed)
        out["init_digests"]["B1_s%d" % seed] = digest(
            ClassicalDisCoCirc(verbs, seed, d_ref=2))
        torch.manual_seed(seed)
        out["init_digests"]["A3_s%d" % seed] = digest(
            QuantumStoryModel("A3", verbs, seed))
        torch.manual_seed(seed)
        out["init_digests"]["A2_s%d" % seed] = digest(
            QuantumStoryModel("A2", verbs, seed))

    # one deterministic forward pass, for numerical-kernel parity
    angles, _, _ = comp.load_embeddings()
    train = [i for i in d["items"] if i["split"] == "train"][:20]
    cs = [comp.compile_story(i, angles) for i in train]
    torch.manual_seed(0)
    m = QuantumStoryModel("A3", verbs, 0)
    with torch.no_grad():
        vals = torch.stack([m.story_values(c) for c in cs])
    out["A3_s0_first20_values_sum"] = float(vals.sum())
    out["A3_s0_first20_values_digest"] = hashlib.sha256(
        vals.to(torch.float64).numpy().tobytes()).hexdigest()[:16]

    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
