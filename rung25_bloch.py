import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

AUC_A_VS_B = 0.5462

d1 = np.load("states_L1.npz", allow_pickle=True)
d0 = np.load("states_L0.npz", allow_pickle=True)

states = d1["states"]
labels = d1["labels"]
pair_ids = d1["pair_ids"]
n = states.shape[0]

def bloch(psi):
    a, b = psi[0], psi[1]
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = np.abs(a) ** 2 - np.abs(b) ** 2
    return x, y, z

coords = np.array([bloch(s) for s in states])

# L0: all identical -> single point
s0 = d0["states"][0]
x0, y0, z0 = bloch(s0)

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection="3d")

# light wireframe unit sphere
u = np.linspace(0, 2 * np.pi, 24)
v = np.linspace(0, np.pi, 16)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(xs, ys, zs, color="lightgray", linewidth=0.4, alpha=0.6)

# thin gray lines connecting matched pairs (same pair_id)
from collections import defaultdict
groups = defaultdict(list)
for i, pid in enumerate(pair_ids):
    groups[int(pid)].append(i)
for pid, idxs in groups.items():
    for j in range(len(idxs)):
        for k in range(j + 1, len(idxs)):
            a_i, b_i = idxs[j], idxs[k]
            ax.plot(
                [coords[a_i, 0], coords[b_i, 0]],
                [coords[a_i, 1], coords[b_i, 1]],
                [coords[a_i, 2], coords[b_i, 2]],
                color="gray", linewidth=0.5, alpha=0.5,
            )

is_svo = np.array([str(l).endswith("SVO") for l in labels])
is_vso = np.array([str(l).endswith("VSO") for l in labels])

ax.scatter(coords[is_svo, 0], coords[is_svo, 1], coords[is_svo, 2],
           c="blue", marker="o", s=40, depthshade=True, label="SVO sentences")
ax.scatter(coords[is_vso, 0], coords[is_vso, 1], coords[is_vso, 2],
           c="red", marker="^", s=45, depthshade=True, label="VSO sentences")

# L0 single point (black star)
ax.scatter([x0], [y0], [z0], c="black", marker="*", s=320,
           label="L0 (all identical)")
ax.text(x0, y0, z0 + 0.12,
        "L0: no entanglement -- every sentence collapses to this single point",
        color="black", fontsize=8, ha="center")

ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

fig.suptitle("Arabic sentences as points in quantum state space (first map of its kind)",
             fontsize=13, y=0.97)
ax.set_title("n = %d states   |   exp15 AUC (A vs B) = %.4f" % (n, AUC_A_VS_B),
             fontsize=10, pad=10)
ax.legend(loc="upper left", fontsize=8)

fig.savefig("figures/fig5_bloch_map.png", dpi=220, bbox_inches="tight")
print("SAVED figures/fig5_bloch_map.png")
print("n_states:", n, "SVO:", int(is_svo.sum()), "VSO:", int(is_vso.sum()))
print("n_pairs(groups):", len(groups))
