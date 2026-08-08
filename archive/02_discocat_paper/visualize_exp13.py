# -*- coding: utf-8 -*-
"""
visualize_exp13.py
------------------
Paper-ready figures for exp13 (Structural Inductive Bias vs AraBERT).

Figures:
  fig7_exp13_main.png      — Main comparison table: all methods × 3 tasks
  fig8_ablation.png        — L0 vs L1 vs L2 ablation (the core causal claim)
  fig9_learning_curves.png — Learning curves on word order: AraVec vs Frozen BERT vs QFM L1/L2
  fig10_mechanisms.png     — Schematic: 3 mechanisms side-by-side (AraVec / BERT / Quantum)

Usage:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 visualize_exp13.py
"""

import json, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

EXP13_DIR = Path("qnlp_experiment_outputs_per_set_v2/exp13_arabert")

# ── colour palette ─────────────────────────────────────────────────────────────
C_CLASSICAL = "#888888"
C_BERT_FRZ  = "#4E79A7"
C_BERT_FT   = "#2C5F8A"
C_TOPO      = "#F28E2B"
C_QFM_L0    = "#E8E8E8"
C_QFM_L1    = "#59A14F"
C_QFM_L2    = "#76B7B2"
C_SPSA      = "#B07AA1"
C_CHANCE    = "#CC0000"


# ═══════════════════════════════════════════════════════════════════════════════
#  Load results
# ═══════════════════════════════════════════════════════════════════════════════

def load():
    main = json.loads((EXP13_DIR / "exp13_summary.json").read_text(encoding="utf-8"))
    ft   = json.loads((EXP13_DIR / "arabert_finetuned_results.json").read_text(encoding="utf-8"))
    lc   = json.loads((EXP13_DIR / "learning_curves.json").read_text(encoding="utf-8"))
    return main, ft, lc

main_results, ft_results, lc_results = load()

TASKS = {
    "TaskA_WordOrder": ("Word Order\n(SVO vs VSO)", 0.50),
    "TaskB_Tense":     ("Tense\n(Past vs Present)", 0.50),
    "TaskC_King_Possess": ("Lexical Structural\n(King vs Possess)", 0.50),
}

METHODS = [
    ("AraVec",            "AraVec\n(bag-of-words)",   C_CLASSICAL),
    ("AraBERT_frozen",    "Frozen AraBERT\nCLS + SVM", C_BERT_FRZ),
    ("AraBERT_finetuned", "Fine-tuned\nAraBERT",       C_BERT_FT),
    ("topology_only",     "Topology-only\n(zero params)", C_TOPO),
    ("QFM_IQP_L0",        "QFM L0\n(product ablation)", C_QFM_L0),
    ("QFM_IQP_L1",        "QFM IQP L1",               C_QFM_L1),
    ("QFM_IQP_L2",        "QFM IQP L2",               C_QFM_L2),
    ("SPSA_IQP_L1",       "SPSA IQP L1",              C_SPSA),
]


def get_mean_std(task_key, method_key):
    """Pull mean/std from main_results, with ft override for finetuned."""
    if method_key == "AraBERT_finetuned":
        ft_key_map = {
            "TaskA_WordOrder":     "WordOrder_SVO_vs_VSO",
            "TaskB_Tense":        "Tense_Past_vs_Pres",
            "TaskC_King_Possess": "King_vs_Possess",
        }
        r = ft_results.get(ft_key_map[task_key], {})
        return r.get("mean", 0.0), r.get("std", 0.0)
    r = main_results.get(task_key, {}).get(method_key, {})
    return r.get("mean", 0.0), r.get("std", 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Fig 7 — Main comparison bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def fig7_main():
    task_keys  = list(TASKS.keys())
    task_labels = [TASKS[k][0] for k in task_keys]
    n_tasks    = len(task_keys)
    n_methods  = len(METHODS)

    x     = np.arange(n_tasks)
    width = 0.09
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

    fig, ax = plt.subplots(figsize=(14, 6))

    for j, (mkey, mlabel, colour) in enumerate(METHODS):
        means, stds = [], []
        for tk in task_keys:
            mu, sd = get_mean_std(tk, mkey)
            means.append(mu)
            stds.append(sd)
        bars = ax.bar(x + offsets[j], means, width * 0.9, yerr=stds,
                      color=colour, label=mlabel, capsize=3,
                      edgecolor="white" if colour != C_QFM_L0 else "#aaaaaa",
                      linewidth=0.5, error_kw={"elinewidth": 1})

    # Chance line
    ax.axhline(0.50, color=C_CHANCE, ls="--", lw=1.2, label="Chance (50%)")

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("exp13: Structural Inductive Bias — All Methods × All Tasks", fontsize=13, pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=8.5,
              framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / "fig7_exp13_main.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Fig 8 — Ablation: L0 vs L1 vs L2 (Word Order only — the causal claim)
# ═══════════════════════════════════════════════════════════════════════════════

def fig8_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    ablation_methods = [
        ("QFM_IQP_L0", "QFM L0\n(product\nno entanglement)", C_QFM_L0),
        ("QFM_IQP_L1", "QFM L1\n(1 entangling\nlayer)", C_QFM_L1),
        ("QFM_IQP_L2", "QFM L2\n(2 entangling\nlayers)", C_QFM_L2),
    ]
    context_methods = [
        ("AraVec",            "AraVec",          C_CLASSICAL),
        ("AraBERT_frozen",    "Frozen\nAraBERT",  C_BERT_FRZ),
        ("AraBERT_finetuned", "Fine-tuned\nBERT", C_BERT_FT),
    ]

    for ax, (tk, (tlabel, chance)) in zip(axes, TASKS.items()):
        # Context bars (greyed background)
        for i, (mkey, mlabel, colour) in enumerate(context_methods):
            mu, sd = get_mean_std(tk, mkey)
            ax.bar(i, mu, 0.6, color=colour, alpha=0.35, edgecolor="grey", lw=0.5)
            ax.text(i, mu + 0.015, f"{mu:.0%}", ha="center", va="bottom",
                    fontsize=8, color="grey")

        # Ablation bars (vivid)
        for i, (mkey, mlabel, colour) in enumerate(ablation_methods):
            xi = i + len(context_methods) + 0.5
            mu, sd = get_mean_std(tk, mkey)
            ec = "#888" if colour == C_QFM_L0 else "white"
            ax.bar(xi, mu, 0.6, color=colour, edgecolor=ec, lw=0.8,
                   zorder=3)
            ax.text(xi, mu + 0.015, f"{mu:.0%}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color="black" if colour == C_QFM_L0 else colour)

        # Vertical separator
        ax.axvline(len(context_methods) + 0.1, color="#cccccc", lw=1, ls=":")
        ax.axhline(chance, color=C_CHANCE, ls="--", lw=1.2, alpha=0.7)

        # Highlight L0→L1 jump arrow
        mu_l0, _ = get_mean_std(tk, "QFM_IQP_L0")
        mu_l1, _ = get_mean_std(tk, "QFM_IQP_L1")
        xi_l0 = len(context_methods) + 0.5
        xi_l1 = xi_l0 + 1
        if mu_l1 > mu_l0 + 0.02:
            ax.annotate("", xy=(xi_l1, mu_l1 - 0.01),
                        xytext=(xi_l0, mu_l0 + 0.01),
                        arrowprops=dict(arrowstyle="->", color=C_QFM_L1, lw=1.5))
            delta = mu_l1 - mu_l0
            ax.text((xi_l0 + xi_l1) / 2 + 0.15, (mu_l0 + mu_l1) / 2,
                    f"+{delta:.0%}", color=C_QFM_L1, fontsize=9, fontweight="bold")

        all_labels = [m[1] for m in context_methods] + [""] + [m[1] for m in ablation_methods]
        all_x      = list(range(len(context_methods))) + [len(context_methods) + 0.1] + \
                     [len(context_methods) + 0.5 + i for i in range(len(ablation_methods))]
        ax.set_xticks([len(context_methods) + 0.5 + i for i in range(len(ablation_methods))] +
                      list(range(len(context_methods))))
        ax.set_xticklabels([m[1] for m in ablation_methods] +
                           [m[1] for m in context_methods], fontsize=8)
        ax.set_title(tlabel, fontsize=11, pad=8)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Accuracy", fontsize=11)
    fig.suptitle("QFM Entanglement Ablation: L0 (product) → L1 → L2\n"
                 "Grey = classical baselines  |  Vivid = quantum QFM",
                 fontsize=12, y=1.02)

    # Legend
    patches = [
        mpatches.Patch(color=C_QFM_L0, ec="#888", label="QFM L0 — grammar topology only (no entanglement)"),
        mpatches.Patch(color=C_QFM_L1, label="QFM L1 — grammar + 1 IQP entangling layer"),
        mpatches.Patch(color=C_QFM_L2, label="QFM L2 — grammar + 2 IQP entangling layers"),
        mpatches.Patch(color=C_CLASSICAL, alpha=0.4, label="AraVec (bag-of-words)"),
        mpatches.Patch(color=C_BERT_FRZ, alpha=0.4, label="Frozen AraBERT CLS"),
        mpatches.Patch(color=C_BERT_FT, alpha=0.4, label="Fine-tuned AraBERT"),
        mpatches.Patch(color=C_CHANCE, label="Chance (50%)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.08), framealpha=0.9)

    fig.tight_layout()
    out = FIG_DIR / "fig8_ablation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Fig 9 — Learning curves (Word Order)
# ═══════════════════════════════════════════════════════════════════════════════

def fig9_learning_curves():
    fig, ax = plt.subplots(figsize=(8, 5))

    lc_methods = [
        ("AraVec",        "AraVec (bag-of-words)",    C_CLASSICAL, "o--"),
        ("AraBERT_frozen","Frozen AraBERT CLS + SVM", C_BERT_FRZ,  "s-"),
        ("QFM_IQP_L1",    "QFM IQP L1",              C_QFM_L1,    "^-"),
        ("QFM_IQP_L2",    "QFM IQP L2",              C_QFM_L2,    "D-"),
    ]

    for mkey, mlabel, colour, fmt in lc_methods:
        mdata = lc_results.get(mkey, {})
        Ns    = sorted(int(k) for k in mdata.keys())
        means = [mdata[str(N)]["mean"] for N in Ns]
        stds  = [mdata[str(N)]["std"]  for N in Ns]
        ax.plot(Ns, means, fmt, color=colour, label=mlabel,
                lw=2, ms=7, markerfacecolor="white", markeredgewidth=2)
        ax.fill_between(Ns,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=colour, alpha=0.12)

    ax.axhline(0.50, color=C_CHANCE, ls="--", lw=1.2, label="Chance (50%)")
    ax.set_xlabel("Training samples per class (N)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Learning Curves — Word Order SVO vs VSO (matched pairs)\n"
                 "AraVec deteriorates; QFM improves monotonically", fontsize=11)
    ax.set_xticks([5, 10, 20, 40])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotation: AraVec getting worse
    ax.annotate("AraVec learns\nspurious anti-correlations\n(matched pairs)",
                xy=(40, 0.185), xytext=(22, 0.08),
                fontsize=8, color=C_CLASSICAL,
                arrowprops=dict(arrowstyle="->", color=C_CLASSICAL, lw=1))

    fig.tight_layout()
    out = FIG_DIR / "fig9_learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Fig 10 — Mechanism schematic (text/diagram figure)
# ═══════════════════════════════════════════════════════════════════════════════

def fig10_mechanisms():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#f8f8f8")

    panels = [
        {
            "title": "AraVec\n(Bag-of-Words)",
            "colour": C_CLASSICAL,
            "acc": "12.8%\non word order",
            "mechanism": (
                "Each word → 300-dim vector\n"
                "Sentence → mean of word vectors\n\n"
                "الولد  →  v₁\n"
                "يقرأ   →  v₂\n"
                "الكتاب →  v₃\n\n"
                "mean(v₁, v₂, v₃)\n\n"
                "ORDER LOST:\nmean is commutative\n"
                "SVO ≡ VSO ≡ OVS"
            ),
            "verdict": "Provably order-blind\non matched pairs",
            "verdict_colour": "#cc0000",
        },
        {
            "title": "Fine-tuned AraBERT\n(Positional Encoding)",
            "colour": C_BERT_FT,
            "acc": "100%\non word order",
            "mechanism": (
                "Token embeddings + position IDs\n"
                "Self-attention over all pairs\n\n"
                "[CLS] الولد[1] يقرأ[2] الكتاب[3]\n\n"
                "Attention: position 1 before 2\n"
                "before 3 → subject-verb-object\n\n"
                "ORDER PRESERVED:\npositional encoding + attention\n"
                "distinguish SVO vs VSO"
            ),
            "verdict": "Perfect accuracy\nRequires fine-tuning",
            "verdict_colour": "#2C5F8A",
        },
        {
            "title": "Quantum QFM IQP L1\n(Circuit Topology)",
            "colour": C_QFM_L1,
            "acc": "64.9%\non word order",
            "mechanism": (
                "Grammar → pregroup diagram\n"
                "→ IQP quantum circuit\n\n"
                "SVO:  |n⟩|n.r⊗s⊗n.l⟩|n⟩\n"
                "      Cup₁ ——————— Cup₂\n\n"
                "VSO:  |s⊗n.l⊗n.l⟩|n⟩|n⟩\n"
                "      Swap → Cup → Cup\n\n"
                "ORDER ENCODED:\ndifferent cup-wiring topology\n"
                "→ different probability output"
            ),
            "verdict": "65% accuracy\nNo fine-tuning needed\nInterpretable mechanism",
            "verdict_colour": "#59A14F",
        },
    ]

    for ax, panel in zip(axes, panels):
        ax.set_facecolor("white")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")

        # Header bar
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.0, 0.85), 1.0, 0.15,
            boxstyle="round,pad=0.01", fc=panel["colour"], ec="none", alpha=0.9,
            transform=ax.transAxes, clip_on=False))
        ax.text(0.5, 0.925, panel["title"], transform=ax.transAxes,
                ha="center", va="center", fontsize=11, fontweight="bold", color="white")

        # Accuracy badge
        ax.text(0.5, 0.78, panel["acc"], transform=ax.transAxes,
                ha="center", va="center", fontsize=13, fontweight="bold",
                color=panel["colour"])

        # Mechanism text
        ax.text(0.5, 0.44, panel["mechanism"], transform=ax.transAxes,
                ha="center", va="center", fontsize=8.5,
                fontfamily="monospace", color="#333333",
                bbox=dict(fc="#f5f5f5", ec="#dddddd", boxstyle="round,pad=0.4"))

        # Verdict
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.05, 0.01), 0.90, 0.12,
            boxstyle="round,pad=0.01",
            fc=panel["verdict_colour"], ec="none", alpha=0.15,
            transform=ax.transAxes))
        ax.text(0.5, 0.07, panel["verdict"], transform=ax.transAxes,
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=panel["verdict_colour"])

    fig.suptitle("Three Mechanisms for Word Order Sensitivity in NLP",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(pad=1.5)
    out = FIG_DIR / "fig10_mechanisms.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating exp13 figures...")
    fig7_main()
    fig8_ablation()
    fig9_learning_curves()
    fig10_mechanisms()
    print(f"\nAll figures saved to {FIG_DIR}/")
