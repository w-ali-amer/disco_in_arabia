# -*- coding: utf-8 -*-
"""
visualize_results_v2.py
-----------------------
Paper-ready figures covering all experiments (exp8 through exp12).

Figures generated:
  fig1 — Three Quantum Advantage Scenarios (the key paper figure)
  fig2 — Tense Deep Ablation: n_layers × ansatz (exp9)
  fig3 — Word Order: Classical vs Quantum (exp10 3-class + exp12 binary QFM)
  fig4 — Sense-Switch: exp8 vs exp11 oracle vs exp11 auto per pair
  fig5 — Binary Pairs Heatmap: all methods × all pairs
  fig6 — PCA of AraVec embeddings (Tense, Word Order, King/Possess)

Usage:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 visualize_results_v2.py
"""

import json, warnings, sys
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

BASE = Path("qnlp_experiment_outputs_per_set_v2")

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi":     150,
})

C = {
    "quantum":   "#4C72B0",
    "qfm":       "#4C72B0",
    "spsa":      "#6A9ECC",
    "classical": "#2CA02C",
    "oracle":    "#DD8452",
    "auto":      "#E8A87C",
    "exp8":      "#8172B2",
    "chance":    "#AAAAAA",
    "sim14":     "#DD8452",
    "iqp":       "#4C72B0",
}

pct = mticker.PercentFormatter(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _mean(path, key="mean"):
    try:
        return load_json(path).get(key, None)
    except Exception:
        return None

def load_exp9():
    p = BASE / "exp9_tense_deep"
    results = defaultdict(list)
    for f in sorted(p.glob("results_*.json")):
        d = load_json(f)
        stem = f.stem.replace("results_Morphology_", "")
        task, ans, layer = stem.split("_")[:3]
        results[(task, ans, layer)].append(d.get("mean", 0))
    return results  # key: (task, ansatz, L#)

def load_exp10():
    p = BASE / "exp10_wordorder"
    cl = load_json(p / "classical_baseline.json")
    quantum = defaultdict(list)
    for f in sorted(p.glob("results_*.json")):
        d = load_json(f)
        stem = f.stem.replace("results_", "")
        parts = stem.split("_")
        key = (parts[0], parts[1])   # (IQP, L1)
        quantum[key].append(d.get("mean", 0))
    return cl, quantum

def load_exp11():
    p = BASE / "exp11_sense_switch"
    summary = load_json(p / "exp11_summary.json")
    return summary  # has exp11, exp8_iqp, classical

def load_exp12():
    p = BASE / "exp12_quantum_advantage"
    return load_json(p / "exp12_summary.json")

def load_exp8():
    p = BASE / "exp8_binary_lexico"
    results = {}
    for f in p.glob("results_Binary_*_IQP.json"):
        pair = f.stem.replace("results_", "").replace("_IQP", "")
        results[pair] = load_json(f).get("mean", 0)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 1 — THREE QUANTUM ADVANTAGE SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_three_scenarios(exp9, exp10_cl, exp10_q, exp11, exp12):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # ── S1: Tense ─────────────────────────────────────────────────────────────
    ax = axes[0]
    # Best from exp5 (trained, 150ep): 75.3%; exp9 Sim14 L2: 64.0%; classical: 54.7%
    tense_quantum = np.mean(exp9[("Tense", "Sim14", "L2")])
    tense_exp5    = 0.753
    tense_cl      = 0.547

    bars = ax.bar(["Classical\n(AraVec SVM)", "Quantum\n(Sim14 L2\nexp9)",
                   "Quantum\n(IQP exp5)"],
                  [tense_cl, tense_quantum, tense_exp5],
                  color=[C["classical"], C["sim14"], C["iqp"]],
                  alpha=0.85, width=0.55, zorder=3,
                  yerr=[0.04, np.std(exp9[("Tense", "Sim14", "L2")]), 0.04],
                  capsize=5, error_kw={"linewidth": 1.5})
    ax.axhline(0.5, color=C["chance"], linestyle="--", linewidth=1.5, label="Chance (50%)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title("S1: Morphology / Tense", fontweight="bold")
    ax.set_ylabel("5-fold CV Accuracy")
    ax.text(2, tense_exp5 + 0.02, f"+{tense_exp5 - tense_cl:+.1%}", ha="center",
            fontsize=9, color="darkgreen", fontweight="bold")
    ax.text(0.5, 0.05, "Mechanism: past كتب → nominal circuit\npres يكتب → verbal circuit",
            transform=ax.transAxes, fontsize=8, ha="center", color="gray",
            style="italic")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(fontsize=8)

    # ── S2: Word Order (controlled matched pairs) ─────────────────────────────
    ax = axes[1]
    s2 = exp12["scenario2_wordorder"]
    cl_wo  = s2["classical"]["mean"]
    qfm_l1 = s2["qfm_iqp_l1"]["mean"]
    qfm_l2 = s2["qfm_iqp_l2"]["mean"]
    spsa   = s2["spsa_iqp_l1"]["mean"]

    vals   = [cl_wo, qfm_l1, qfm_l2, spsa]
    errs   = [s2["classical"]["std"], s2["qfm_iqp_l1"]["std"],
              s2["qfm_iqp_l2"]["std"], s2["spsa_iqp_l1"]["std"]]
    colors = [C["classical"], C["qfm"], C["qfm"], C["spsa"]]
    labels = ["Classical\n(AraVec SVM)", "Quantum\nQFM L1", "Quantum\nQFM L2",
              "Quantum\nSPSA L1"]
    alphas = [0.85, 0.5, 0.85, 0.65]

    for i, (v, e, col, lbl, alp) in enumerate(zip(vals, errs, colors, labels, alphas)):
        ax.bar(i, v, color=col, alpha=alp, width=0.55, zorder=3,
               yerr=e, capsize=5, error_kw={"linewidth": 1.5})
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0.5, color=C["chance"], linestyle="--", linewidth=1.5, label="Chance (50%)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title("S2: Word Order / SVO vs VSO\n(Matched Pairs, Same Words)", fontweight="bold")
    ax.text(2, qfm_l2 + 0.03, f"+{qfm_l2 - 0.5:+.1%} vs chance", ha="center",
            fontsize=9, color="darkgreen", fontweight="bold")
    ax.text(0.5, 0.05, "Classical: identical word bags → BLIND to order\nQuantum: cup wiring topology → ORDER-SENSITIVE",
            transform=ax.transAxes, fontsize=8, ha="center", color="gray",
            style="italic")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(fontsize=8)

    # ── S3: Lexical Ambiguity (sense-switch) ──────────────────────────────────
    ax = axes[2]
    e11 = exp11["exp11"]
    e8_map = exp11["exp8_iqp"]
    cl_map = exp11["classical"]

    # Average across IQP pairs
    oracle_means = [r["oracle_mean"] for r in e11 if r.get("ansatz") == "IQP" and "error" not in r]
    auto_means   = [r["auto_mean"]   for r in e11 if r.get("ansatz") == "IQP" and "error" not in r]
    e8_vals      = [v for v in e8_map.values() if v is not None]
    cl_vals      = [v for v in cl_map.values() if v is not None]

    mu_oracle = np.mean(oracle_means)
    mu_auto   = np.mean(auto_means)
    mu_e8     = np.mean(e8_vals)
    mu_cl     = np.mean(cl_vals)

    vals   = [mu_e8, mu_oracle, mu_auto, mu_cl]
    errs   = [np.std(e8_vals), np.std(oracle_means), np.std(auto_means), np.std(cl_vals)]
    colors = [C["exp8"], C["oracle"], C["auto"], C["classical"]]
    labels = ["Exp8\n(baseline\nIQP)", "Exp11\nOracle\n(upper bound)",
              "Exp11\nAuto\n(dual-circuit)", "Classical\n(AraVec SVM)"]

    for i, (v, e, col, lbl) in enumerate(zip(vals, errs, colors, labels)):
        ax.bar(i, v, color=col, alpha=0.85, width=0.55, zorder=3,
               yerr=e, capsize=5, error_kw={"linewidth": 1.5})
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0.5, color=C["chance"], linestyle="--", linewidth=1.5, label="Chance (50%)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title("S3: Lexical Ambiguity\n(Sense-Switch, 7 Binary Pairs, avg)", fontweight="bold")
    ax.annotate("", xy=(1, mu_oracle), xytext=(0, mu_e8),
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5))
    ax.text(0.5, mu_e8 + (mu_oracle - mu_e8)/2, f"+{mu_oracle - mu_e8:.1%}",
            ha="center", fontsize=9, color="darkgreen", fontweight="bold")
    ax.text(0.5, 0.05, "Mechanism: sense-tagged params (SA/SB)\ngive polysemous word two quantum states",
            transform=ax.transAxes, fontsize=8, ha="center", color="gray",
            style="italic")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(fontsize=8)

    fig.suptitle("Arabic QNLP — Quantum Advantage Across Three NLP Tasks",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = FIG_DIR / "fig1_three_scenarios.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 2 — TENSE ABLATION (exp9)
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_tense_ablation(exp9):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    layers = ["L1", "L2", "L3"]
    x = np.arange(len(layers))
    w = 0.35

    classical_tense = 0.547  # SVM_rbf

    for ax, ansatz, color in [(axes[0], "IQP", C["iqp"]), (axes[1], "Sim14", C["sim14"])]:
        means = [np.mean(exp9.get(("Tense", ansatz, l), [0])) for l in layers]
        stds  = [np.std(exp9.get(("Tense", ansatz, l), [0]))  for l in layers]
        seed_vals = [exp9.get(("Tense", ansatz, l), []) for l in layers]

        bars = ax.bar(x, means, w*2, label=f"{ansatz} (mean±std)", color=color,
                      alpha=0.7, zorder=3, yerr=stds, capsize=6,
                      error_kw={"linewidth": 1.5})

        # Individual seed dots
        for i, sv in enumerate(seed_vals):
            jitter = np.linspace(-0.12, 0.12, len(sv))
            ax.scatter(np.full(len(sv), i) + jitter, sv,
                       color=color, s=50, zorder=5, alpha=0.9,
                       label=("Seeds" if i == 0 else None))

        ax.axhline(classical_tense, color=C["classical"], linestyle="--",
                   linewidth=1.8, label=f"Classical best ({classical_tense:.1%})")
        ax.axhline(0.5, color=C["chance"], linestyle=":", linewidth=1.2, label="Chance (50%)")
        ax.axhline(0.753, color="darkgreen", linestyle="-.", linewidth=1.5,
                   label="Exp5 best (75.3%)", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([f"L={i+1}" for i in range(len(layers))], fontsize=11)
        ax.set_title(f"{ansatz} Ansatz — Tense Classification\n(500 epochs, 3 seeds)")
        ax.set_ylabel("5-fold CV Accuracy")
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(pct)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.25, zorder=0)

        # Annotate best
        best_i = int(np.argmax(means))
        ax.annotate(f"Best: {means[best_i]:.1%}",
                    xy=(best_i, means[best_i] + stds[best_i]),
                    xytext=(best_i + 0.3, means[best_i] + 0.06),
                    fontsize=8, color="darkblue",
                    arrowprops=dict(arrowstyle="->", color="darkblue", lw=1.0))

    fig.suptitle("exp9 — Tense Classification: Layer Depth Ablation (IQP vs Sim14)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIG_DIR / "fig2_tense_ablation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 3 — WORD ORDER: exp10 (3-class) + exp12 (binary QFM)
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_word_order(exp10_cl, exp10_q, exp12):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: 3-class SPSA (exp10) ────────────────────────────────────────────
    ax = axes[0]
    methods = ["SVM_linear", "SVM_rbf", "RF"]
    cl_vals = [exp10_cl[m]["mean"] for m in methods]

    q_configs = [("IQP","L1"), ("IQP","L2"), ("Sim14","L1"), ("Sim14","L2")]
    q_means   = [np.mean(exp10_q.get(cfg, [0])) for cfg in q_configs]
    q_stds    = [np.std(exp10_q.get(cfg, [0]))  for cfg in q_configs]
    q_labels  = ["IQP L1", "IQP L2", "Sim14 L1", "Sim14 L2"]

    all_means  = cl_vals + q_means
    all_colors = [C["classical"]]*3 + [C["iqp"], C["iqp"], C["sim14"], C["sim14"]]
    all_alphas = [0.85]*3 + [0.85, 0.6, 0.85, 0.6]
    all_labels = ["SVM-lin", "SVM-rbf", "RF"] + q_labels
    all_errs   = [exp10_cl[m]["std"] for m in methods] + q_stds

    for i, (v, e, col, lbl, alp) in enumerate(
            zip(all_means, all_errs, all_colors, all_labels, all_alphas)):
        ax.bar(i, v, color=col, alpha=alp, width=0.65, zorder=3,
               yerr=e, capsize=4, error_kw={"linewidth": 1.2})

    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=25, ha="right", fontsize=9)
    ax.axhline(1/3, color=C["chance"], linestyle="--", linewidth=1.5, label="Chance (33.3%)")
    ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title("exp10 — Word Order 3-Class\n(SVO/VSO/Nominal, SPSA training)")
    ax.set_ylabel("5-fold CV Accuracy")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    cl_patch = mpatches.Patch(color=C["classical"], alpha=0.85, label="Classical")
    q_patch  = mpatches.Patch(color=C["iqp"],       alpha=0.85, label="Quantum")
    ax.legend(handles=[cl_patch, q_patch,
                       mpatches.Patch(color=C["chance"], label="Chance")],
              fontsize=8, loc="upper right")
    ax.text(0.5, 0.9, "Both classical & quantum near chance:\naveraged embeddings are order-blind",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray", style="italic")

    # ── Right: binary SVO/VSO QFM (exp12) ─────────────────────────────────────
    ax = axes[1]
    s2 = exp12["scenario2_wordorder"]
    items = [
        ("Classical\nSVM",       s2["classical"]["mean"],   s2["classical"]["std"],   C["classical"], 0.85),
        ("QFM\nIQP L1",          s2["qfm_iqp_l1"]["mean"],  s2["qfm_iqp_l1"]["std"],  C["iqp"],       0.5),
        ("QFM\nIQP L2",          s2["qfm_iqp_l2"]["mean"],  s2["qfm_iqp_l2"]["std"],  C["iqp"],       0.85),
        ("SPSA\nIQP L1",         s2["spsa_iqp_l1"]["mean"], s2["spsa_iqp_l1"]["std"], C["spsa"],      0.75),
    ]
    for i, (lbl, v, e, col, alp) in enumerate(items):
        ax.bar(i, v, color=col, alpha=alp, width=0.6, zorder=3,
               yerr=e, capsize=5, error_kw={"linewidth": 1.5})

    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([x[0] for x in items], fontsize=10)
    ax.axhline(0.5, color=C["chance"], linestyle="--", linewidth=1.5, label="Chance (50%)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title("exp12 — Word Order SVO/VSO Binary\n(Matched Pairs, Same Words, QFM)")
    ax.set_ylabel("5-fold CV Accuracy")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    qfm_l2 = s2["qfm_iqp_l2"]["mean"]
    ax.annotate(f"QFM L2: {qfm_l2:.1%}\n+{qfm_l2-0.5:.1%} vs chance",
                xy=(2, qfm_l2), xytext=(2.3, qfm_l2 + 0.1),
                fontsize=8, color="darkgreen", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkgreen"))
    ax.text(0.5, 0.05,
            "Classical: same word bag → can't encode order\nQuantum topology: cup-wiring ≠ swap+cups",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray", style="italic")

    fig.suptitle("Word Order Classification: Classical vs Quantum", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIG_DIR / "fig3_word_order.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 4 — SENSE-SWITCH: exp8 vs exp11 oracle vs exp11 auto
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_sense_switch(exp11):
    e11_results = [r for r in exp11["exp11"] if r.get("ansatz") == "IQP" and "error" not in r]
    e8_map  = exp11["exp8_iqp"]
    cl_map  = exp11["classical"]

    pairs  = [r["pair_name"] for r in e11_results]
    labels = [p.replace("Binary_", "").replace("_", "/") for p in pairs]
    n = len(pairs)
    x = np.arange(n)
    w = 0.22

    e8_vals     = [e8_map.get(p, 0) or 0     for p in pairs]
    oracle_vals = [r["oracle_mean"]           for r in e11_results]
    auto_vals   = [r["auto_mean"]             for r in e11_results]
    cl_vals     = [cl_map.get(p, 0) or 0      for p in pairs]
    oracle_errs = [r["oracle_std"]            for r in e11_results]
    auto_errs   = [r["auto_std"]              for r in e11_results]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - 1.5*w, e8_vals,     w, label="Exp8 IQP (baseline)",     color=C["exp8"],      alpha=0.85, zorder=3)
    ax.bar(x - 0.5*w, oracle_vals, w, label="Exp11 Oracle (upper bound)", color=C["oracle"],  alpha=0.85, zorder=3,
           yerr=oracle_errs, capsize=4, error_kw={"linewidth": 1.2})
    ax.bar(x + 0.5*w, auto_vals,   w, label="Exp11 Auto (dual-circuit)", color=C["auto"],    alpha=0.85, zorder=3,
           yerr=auto_errs,   capsize=4, error_kw={"linewidth": 1.2})
    ax.bar(x + 1.5*w, cl_vals,     w, label="Classical best (SVM/RF/MLP)", color=C["classical"], alpha=0.7, zorder=3)

    ax.axhline(0.5, color=C["chance"], linestyle="--", linewidth=1.5, label="Chance (50%)")

    # Annotate improvements
    for i, (e8, ora, pair) in enumerate(zip(e8_vals, oracle_vals, pairs)):
        delta = ora - e8
        if delta > 0.02:
            ax.annotate(f"+{delta:.0%}", xy=(i - 0.5*w, ora + 0.01),
                        xytext=(i - 0.5*w, ora + 0.05),
                        ha="center", fontsize=7.5, color="darkgreen", fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color="darkgreen", lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("5-fold CV Accuracy")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title("exp11 — Sense-Switch: Polysemous Word Gets Two Parameter Sets (SA / SB)\n"
                 "Oracle = correct sense tag at test time (upper bound); "
                 "Auto = dual-circuit confidence selection", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=9)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    fig.tight_layout()
    path = FIG_DIR / "fig4_sense_switch.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 5 — HEATMAP: ALL BINARY PAIRS × ALL METHODS
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_heatmap(exp11):
    e11_results = [r for r in exp11["exp11"] if r.get("ansatz") == "IQP" and "error" not in r]
    e8_map  = exp11["exp8_iqp"]
    cl_map  = exp11["classical"]

    pairs  = [r["pair_name"] for r in e11_results]
    ylabels = [p.replace("Binary_", "").replace("_", "/") for p in pairs]

    # Load classical breakdown
    cl_dir  = BASE / "classical_baseline_binary"
    cl_full = {}
    for p in pairs:
        fpath = cl_dir / f"results_{p}.json"
        if fpath.exists():
            d = load_json(fpath)["classifiers"]
            cl_full[p] = {k: d[k]["mean"] for k in d}

    cols = ["Exp8\nIQP", "Exp11\nOracle", "Exp11\nAuto", "SVM\nlinear", "SVM\nrbf", "RF", "MLP", "Chance"]

    mat = np.full((len(pairs), len(cols)), np.nan)
    for i, (pair, r) in enumerate(zip(pairs, e11_results)):
        mat[i, 0] = e8_map.get(pair) or 0
        mat[i, 1] = r["oracle_mean"]
        mat[i, 2] = r["auto_mean"]
        cf = cl_full.get(pair, {})
        mat[i, 3] = cf.get("SVM_linear", np.nan)
        mat[i, 4] = cf.get("SVM_rbf",    np.nan)
        mat[i, 5] = cf.get("RF",          np.nan)
        mat[i, 6] = cf.get("MLP",         np.nan)
        mat[i, 7] = 0.5

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=10)

    for i in range(len(ylabels)):
        for j in range(len(cols)):
            v = mat[i, j]
            if not np.isnan(v):
                color = "black" if 0.4 < v < 0.82 else "white"
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Accuracy", format=pct, shrink=0.8)

    # Vertical separator: quantum | classical
    ax.axvline(2.5, color="white", linewidth=3)
    ax.text(1.0, -0.7, "← Quantum →", ha="center", fontsize=9, color=C["iqp"],
            transform=ax.transData)
    ax.text(5.0, -0.7, "← Classical →", ha="center", fontsize=9, color=C["classical"],
            transform=ax.transData)

    ax.set_title("exp8 vs exp11 vs Classical — Binary Polysemous Pairs\n"
                 "(quantum methods left; classical right; green = high accuracy)",
                 fontsize=11)
    fig.tight_layout()
    path = FIG_DIR / "fig5_heatmap_all_methods.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 6 — PCA EMBEDDINGS (Tense, Word Order matched pairs, King/Possess)
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_pca(embed_fn):
    from sklearn.decomposition import PCA
    data = json.load(open("sentences.json", encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Tense ─────────────────────────────────────────────────────────────────
    morph = data.get("Morphology", [])
    tense_map = {"Morph_Past": "Past (كتب)", "Morph_Pres": "Pres (يكتب)"}
    trows = [(d["sentence"], tense_map[d["label"]]) for d in morph if d["label"] in tense_map]
    _draw_pca(axes[0], trows, embed_fn, "Tense — Low AraVec Separability\n→ Quantum topology wins")

    # ── Word Order (SVO vs VSO matched pairs) ─────────────────────────────────
    wo = data.get("WordOrder", [])
    wo_rows = [(d["sentence"], d["label"].replace("WordOrder_", ""))
               for d in wo if d["label"] in ("WordOrder_SVO", "WordOrder_VSO")]
    _draw_pca(axes[1], wo_rows, embed_fn,
              "Word Order SVO/VSO — Identical Word Bags\n→ PCA collapses both to same region")

    # ── King/Possess ──────────────────────────────────────────────────────────
    la = data.get("LexicalAmbiguity", [])
    kp = [(d["sentence"], d["label"].replace("Ambiguity_", ""))
          for d in la if d["label"] in ("Ambiguity_King", "Ambiguity_Possess")]
    _draw_pca(axes[2], kp, embed_fn,
              "King/Possess — Classical separable via context\nbut quantum topology also discriminates")

    fig.suptitle("PCA of AraVec Sentence Embeddings — Task Separability Analysis",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIG_DIR / "fig6_pca_embeddings.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _draw_pca(ax, rows, embed_fn, title):
    from sklearn.decomposition import PCA
    sents  = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    unique = sorted(set(labels))
    print(f"  PCA: {title.split(chr(10))[0]} ({len(sents)} samples)...")
    X  = np.stack([embed_fn(s) for s in sents])
    pca = PCA(n_components=2, random_state=42)
    X2  = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    cmap = plt.get_cmap("tab10")
    for ci, cls in enumerate(unique):
        idx = [i for i, l in enumerate(labels) if l == cls]
        ax.scatter(X2[idx, 0], X2[idx, 1], label=cls,
                   color=cmap(ci), alpha=0.75, s=45, edgecolors="none")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(f"PC1 ({var[0]:.1%})", fontsize=8)
    ax.set_ylabel(f"PC2 ({var[1]:.1%})", fontsize=8)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading experiment results...")
    exp9     = load_exp9()
    exp10_cl, exp10_q = load_exp10()
    exp11    = load_exp11()
    exp12    = load_exp12()

    # ── Text summary ──────────────────────────────────────────────────────────
    print("\n=== RESULTS SUMMARY ===")
    print(f"  Tense (Sim14 L2 exp9): {np.mean(exp9[('Tense','Sim14','L2')]):.1%}  vs classical 54.7%")
    print(f"  Tense (IQP exp5 ref):   75.3%")
    print(f"  WO 3-class SVM_linear: {exp10_cl['SVM_linear']['mean']:.1%}  (chance 33.3%)")
    print(f"  WO 3-class IQP L1:     {np.mean(exp10_q[('IQP','L1')]):.1%}")
    print(f"  WO binary QFM L2:       {exp12['scenario2_wordorder']['qfm_iqp_l2']['mean']:.1%}")
    e11 = [r for r in exp11["exp11"] if r.get("ansatz")=="IQP" and "error" not in r]
    print(f"  Lexico oracle avg:      {np.mean([r['oracle_mean'] for r in e11]):.1%}")
    print(f"  Lexico exp8 avg:        {np.mean([v for v in exp11['exp8_iqp'].values() if v]):.1%}")
    print()

    print("=== Generating figures ===")

    print("Fig 1: Three quantum advantage scenarios...")
    fig1_three_scenarios(exp9, exp10_cl, exp10_q, exp11, exp12)

    print("Fig 2: Tense ablation (exp9)...")
    fig2_tense_ablation(exp9)

    print("Fig 3: Word order comparison...")
    fig3_word_order(exp10_cl, exp10_q, exp12)

    print("Fig 4: Sense-switch (exp11)...")
    fig4_sense_switch(exp11)

    print("Fig 5: Full heatmap...")
    fig5_heatmap(exp11)

    print("Fig 6: PCA embeddings (AraVec)...")
    try:
        from gensim.models import Word2Vec
        kv = Word2Vec.load("aravec/full_uni_cbow_300_twitter.mdl").wv
        try:
            import stanza
            nlp = stanza.Pipeline("ar", processors="tokenize", verbose=False, download_method=None)
            def tok(s): return [w.text for sent in nlp(s).sentences for w in sent.words]
        except Exception:
            def tok(s): return s.strip().split()
        def embed(sentence):
            tokens = tok(sentence)
            vecs = []
            for t in tokens:
                cands = [t, t[2:]] if t.startswith("ال") else [t]
                for c in cands:
                    if kv.has_index_for(c):
                        vecs.append(kv.get_vector(c)); break
            if not vecs: return np.zeros(kv.vector_size)
            v = np.mean(vecs, axis=0); n = np.linalg.norm(v)
            return v / n if n > 0 else v
        fig6_pca(embed)
    except Exception as e:
        print(f"  Fig 6 skipped: {e}")

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
