"""Bloch map v2 (final): 120 Arabic sentence states with TRUE twin pairing.

Left panel:  exp15/exp15c-W1 states (hash parameters — the published-run
             regime), twin chords rebuilt from word multisets (the original
             fig5 used the scrambled pair_id column; see ERRATUM.md).
Right panel: exp15c-W2 states (repaired AraVec-tied parameters).

Reading: states spread over the sphere (word-order information is present in
the geometry) but twin chords are long and unstructured under both parameter
schemes (meaning proximity is absent) — the Rung-2 negative. The black star
is L0: without entanglement all 120 sentences collapse to a single point.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict


def bloch(psi):
    a, b = psi[0], psi[1]
    return np.array([2 * np.real(np.conj(a) * b),
                     2 * np.imag(np.conj(a) * b),
                     np.abs(a) ** 2 - np.abs(b) ** 2])


def twin_pairs_by_multiset(sentences, labels):
    groups = defaultdict(list)
    for i, s in enumerate(sentences):
        groups[tuple(sorted(str(s).split()))].append(i)
    pairs = []
    for idxs in sorted(groups.values(), key=lambda g: g[0]):
        svo = [i for i in idxs if str(labels[i]).endswith("SVO")]
        vso = [i for i in idxs if str(labels[i]).endswith("VSO")]
        pairs += list(zip(svo, vso))
    return pairs


d1 = np.load("states_L1.npz", allow_pickle=True)
d2 = np.load("states_L1_tied.npz", allow_pickle=True)
d0 = np.load("states_L0.npz", allow_pickle=True)
l0 = bloch(d0["states"][0])

panels = [
    (d1, twin_pairs_by_multiset(d1["sentences"], d1["labels"]),
     "W1: hash parameters (published-run regime)\ntwin-vs-nontwin AUC = 0.580 (p = 0.064)"),
    (d2, [tuple(t) for t in d2["twin_pairs"]],
     "W2: repaired AraVec-tied parameters\ntwin-vs-nontwin AUC = 0.499 (p = 0.51)"),
]

fig = plt.figure(figsize=(16, 8))
u = np.linspace(0, 2 * np.pi, 24)
v = np.linspace(0, np.pi, 16)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))

for k, (d, pairs, subtitle) in enumerate(panels):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    coords = np.array([bloch(s) for s in d["states"]])
    labels = d["labels"]
    ax.plot_wireframe(xs, ys, zs, color="lightgray", linewidth=0.4, alpha=0.6)
    for a_i, b_i in pairs:
        ax.plot([coords[a_i, 0], coords[b_i, 0]],
                [coords[a_i, 1], coords[b_i, 1]],
                [coords[a_i, 2], coords[b_i, 2]],
                color="gray", linewidth=0.5, alpha=0.5)
    is_svo = np.array([str(l).endswith("SVO") for l in labels])
    ax.scatter(coords[is_svo, 0], coords[is_svo, 1], coords[is_svo, 2],
               c="blue", marker="o", s=36, depthshade=True, label="SVO")
    ax.scatter(coords[~is_svo, 0], coords[~is_svo, 1], coords[~is_svo, 2],
               c="red", marker="^", s=40, depthshade=True, label="VSO")
    ax.scatter([l0[0]], [l0[1]], [l0[2]], c="black", marker="*", s=300,
               label="L0 (all 120 sentences)")
    ax.set_title(subtitle, fontsize=10, pad=8)
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=8)
    print(f"panel {k}: {len(pairs)} twin chords, {int(is_svo.sum())} SVO / "
          f"{int((~is_svo).sum())} VSO")

fig.suptitle("Arabic sentence states on the Bloch sphere — true twin pairs connected\n"
             "Order information present (states spread); meaning proximity absent "
             "(twin chords ~ random) — Rung-2 result", fontsize=12, y=0.99)
fig.savefig("figures/fig5_bloch_map_v2.png", dpi=220, bbox_inches="tight")
print("SAVED figures/fig5_bloch_map_v2.png")
