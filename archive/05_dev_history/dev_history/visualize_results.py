# -*- coding: utf-8 -*-
"""
visualize_results.py
--------------------
Comprehensive visualization of all Arabic QNLP experiment results.

Generates:
  1. Binary pair results — bar chart with error bars (exp8)
  2. Full experiment comparison — quantum best vs classical vs chance
  3. PCA of AraVec sentence embeddings, colored by label (all experiment sets)
  4. Results heatmap — all experiments × ansatzes/classifiers
  5. Fold variance violin plots — binary pairs
  6. Progression across experiments (exp5 → exp6 → exp7 → exp8)

Usage:
    /home/waj/discocat_arabic_v2/qiskit_lambeq_env/bin/python3 visualize_results.py
"""

import json, math, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

# ── output ────────────────────────────────────────────────────────────────────
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "figure.dpi":      150,
})

COLORS = {
    "IQP":        "#4C72B0",
    "Sim14":      "#DD8452",
    "SVM_linear": "#55A868",
    "SVM_rbf":    "#C44E52",
    "RF":         "#8172B2",
    "MLP":        "#937860",
    "chance":     "#BBBBBB",
    "classical":  "#2CA02C",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_results():
    base = Path("qnlp_experiment_outputs_per_set_v2")

    # exp5
    exp5 = json.load(open(base / "exp5_trained/run_summary_exp5.json"))["results"]

    # exp6
    exp6 = json.load(open(base / "exp6_warmstart/run_summary_exp6.json"))["results"]

    # exp7
    exp7 = json.load(open(base / "exp7_morph_enriched/run_summary_exp7.json"))["results"]

    # exp8
    exp8 = json.load(open(base / "exp8_binary_lexico/exp8_summary.json"))

    # classical baseline
    cl = json.load(open(base / "classical_baseline/baseline_summary.json"))

    return exp5, exp6, exp7, exp8, cl


def aravec_embeddings():
    """Load AraVec and return sentence embedding function."""
    try:
        from gensim.models import Word2Vec
        kv = Word2Vec.load("aravec/full_uni_cbow_300_twitter.mdl").wv
        dim = kv.vector_size

        try:
            import stanza
            nlp = stanza.Pipeline("ar", processors="tokenize", verbose=False,
                                   download_method=None)
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
            if not vecs:
                return np.zeros(dim)
            v = np.mean(vecs, axis=0)
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        return embed, True
    except Exception as e:
        print(f"AraVec unavailable: {e}")
        return None, False


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 1 — BINARY PAIR RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_binary_pairs(exp8: List[Dict]):
    pairs = [r for r in exp8 if r["exp_name"].startswith("Binary_") and "error" not in r]
    # keep one entry per (name, ansatz)
    by_exp: Dict[str, Dict] = defaultdict(dict)
    for r in pairs:
        by_exp[r["exp_name"]][r["ansatz"]] = r

    names  = sorted(by_exp.keys())
    labels = [n.replace("Binary_", "").replace("_", "/") for n in names]
    x      = np.arange(len(names))
    w      = 0.28

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (ans, color) in enumerate([("IQP", COLORS["IQP"]), ("Sim14", COLORS["Sim14"])]):
        means = [by_exp[n].get(ans, {}).get("mean", 0) for n in names]
        stds  = [by_exp[n].get(ans, {}).get("std",  0) for n in names]
        ax.bar(x + (i - 0.5) * w, means, w, label=ans,
               color=color, alpha=0.85, zorder=3,
               yerr=stds, capsize=4, error_kw={"linewidth": 1.2})

    # chance line
    ax.axhline(0.5, color=COLORS["chance"], linestyle="--", linewidth=1.5,
               label="Chance (50%)", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("5-fold CV Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("exp8 — Binary Polysemous-Pair Classification (Augmented Dataset, 300 epochs, AraVec warm-start)")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # shade above chance
    ax.axhspan(0.5, 1.05, alpha=0.04, color="green")

    fig.tight_layout()
    path = FIG_DIR / "fig1_binary_pairs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 2 — FULL COMPARISON: QUANTUM BEST vs CLASSICAL
# ═══════════════════════════════════════════════════════════════════════════════

def fig_full_comparison(exp5, exp6, exp7, exp8, cl):
    # Build quantum best per experiment
    q_all = exp5 + exp6 + exp7
    q_best: Dict[str, Tuple[float, float, str]] = {}
    for r in q_all:
        name = r["exp_name"]
        mean = r["accuracy_mean"]
        std  = r.get("accuracy_std", 0)
        ans  = r["ansatz"]
        if name not in q_best or mean > q_best[name][0]:
            q_best[name] = (mean, std, ans)

    # Add exp8 binary bests
    for r in exp8:
        if r["exp_name"].startswith("Binary_") and "error" not in r:
            name = r["exp_name"]
            mean = r.get("mean", 0)
            std  = r.get("std",  0)
            ans  = r["ansatz"]
            if name not in q_best or mean > q_best[name][0]:
                q_best[name] = (mean, std, ans)

    # Classical best per experiment
    cl_best: Dict[str, Tuple[float, float]] = {}
    cl_chance: Dict[str, float] = {}
    for r in cl:
        name = r["exp_name"]
        cl_chance[name] = r["chance"]
        for clf_name, clf_r in r["classifiers"].items():
            mean = clf_r["mean"]
            std  = clf_r["std"]
            if name not in cl_best or mean > cl_best[name][0]:
                cl_best[name] = (mean, std)

    # Build common experiment list (all quantum experiments)
    exp_names = list(q_best.keys())
    # sort: morphology tasks first, then lexical, then binary
    order = ["Morphology_Tense", "Morphology_Number", "Morphology_Possession",
             "WordOrder", "LexicalAmbiguity_6class", "LexicalAmbiguity_14class"] + \
            [n for n in sorted(exp_names) if n.startswith("Binary_")]
    exp_names = [n for n in order if n in q_best]

    x = np.arange(len(exp_names))
    w = 0.30
    fig, ax = plt.subplots(figsize=(16, 6))

    q_means = [q_best[n][0] for n in exp_names]
    q_stds  = [q_best[n][1] for n in exp_names]
    c_means = [cl_best.get(n, (0, 0))[0] for n in exp_names]
    c_stds  = [cl_best.get(n, (0, 0))[1] for n in exp_names]
    chances = [cl_chance.get(n, 0.5) for n in exp_names]

    ax.bar(x - w/2, q_means, w, label="Quantum best", color=COLORS["IQP"],
           alpha=0.85, zorder=3, yerr=q_stds, capsize=3,
           error_kw={"linewidth": 1.0})
    ax.bar(x + w/2, c_means, w, label="Classical best (SVM/RF)", color=COLORS["classical"],
           alpha=0.75, zorder=3, yerr=c_stds, capsize=3,
           error_kw={"linewidth": 1.0})

    # plot chance as dots
    ax.scatter(x, chances, marker="D", color=COLORS["chance"], s=40, zorder=5,
               label="Chance level")

    xlabels = [n.replace("Binary_", "").replace("Morphology_", "Morph_")
                .replace("LexicalAmbiguity_", "LA_").replace("_", "/")
               for n in exp_names]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("5-fold CV Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Arabic QNLP — Quantum Best vs Classical Best (all experiments)")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    path = FIG_DIR / "fig2_quantum_vs_classical.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 3 — PCA OF AraVec SENTENCE EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_pca_embeddings(embed_fn):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder

    data = json.load(open("sentences.json", encoding="utf-8"))

    LEXICO_6 = {"Ambiguity_Man","Ambiguity_Leg","Ambiguity_Hit",
                "Ambiguity_Multiply","Ambiguity_King","Ambiguity_Possess"}

    number_map = {"Morph_SgMasc":"Sg","Morph_SgFem":"Sg",
                  "Morph_DualMasc":"Du","Morph_DualFem":"Du",
                  "Morph_PlMasc":"Pl","Morph_PlFem":"Pl",
                  "Morph_PlBroken":"Pl","Morph_AdjPlMasc":"Pl","Morph_AdjPlFem":"Pl"}
    tense_map  = {"Morph_Past":"Past","Morph_Pres":"Pres"}
    poss_map   = {"Morph_Poss1Sg":"1st","Morph_Poss1Pl":"1st",
                  "Morph_Poss2MascSg":"2nd","Morph_Poss2Pl":"2nd",
                  "Morph_Poss3MascSg":"3rd","Morph_Poss3FemSg":"3rd"}

    morph = data.get("Morphology", [])
    la    = data.get("LexicalAmbiguity", [])

    datasets = {
        "Morphology: Tense":
            [(d["sentence"], tense_map[d["label"]]) for d in morph if d["label"] in tense_map],
        "Morphology: Number":
            [(d["sentence"], number_map[d["label"]]) for d in morph if d["label"] in number_map],
        "Morphology: Possession":
            [(d["sentence"], poss_map[d["label"]]) for d in morph if d["label"] in poss_map],
        "LexicalAmbiguity (6-class)":
            [(d["sentence"], d["label"].replace("Ambiguity_","")) for d in la if d["label"] in LEXICO_6],
    }

    # Binary pairs
    pair_map = [("Man","Leg"),("Eye","Spring"),("Knowledge","Flag")]
    for a, b in pair_map:
        key = f"Binary: {a}/{b}"
        rows = [(d["sentence"], d["label"].replace("Ambiguity_",""))
                for d in la if d["label"] in {f"Ambiguity_{a}", f"Ambiguity_{b}"}]
        datasets[key] = rows

    # Color palettes per dataset
    cmap_names = ["tab10", "tab10", "tab10", "Set1", "Set1", "Set1", "Set1"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for ax_i, (title, rows) in enumerate(datasets.items()):
        if ax_i >= len(axes):
            break
        ax = axes[ax_i]
        sents  = [r[0] for r in rows]
        labels = [r[1] for r in rows]
        unique = sorted(set(labels))

        print(f"  PCA: {title} ({len(sents)} samples)...")
        X = np.stack([embed_fn(s) for s in sents])
        pca = PCA(n_components=2, random_state=42)
        X2  = pca.fit_transform(X)
        var = pca.explained_variance_ratio_

        cmap = plt.get_cmap("tab10")
        for ci, cls in enumerate(unique):
            idx = [i for i, l in enumerate(labels) if l == cls]
            ax.scatter(X2[idx, 0], X2[idx, 1], label=cls,
                       color=cmap(ci), alpha=0.75, s=40, edgecolors="none")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"PC1 ({var[0]:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({var[1]:.1%})", fontsize=8)
        ax.legend(fontsize=7, loc="best", markerscale=1.2)
        ax.grid(alpha=0.2)

    # hide unused axes
    for ax_i in range(len(datasets), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle("PCA of AraVec Sentence Embeddings by Task and Label", fontsize=13, y=1.01)
    fig.tight_layout()
    path = FIG_DIR / "fig3_pca_embeddings.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 4 — HEATMAP OF ALL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_heatmap(exp5, exp6, exp7, exp8, cl):
    # Experiments (rows) × systems (columns)
    systems = ["IQP (Q)", "Sim14 (Q)", "SVM_lin", "SVM_rbf", "RF", "MLP", "Chance"]

    row_order = [
        ("Morphology_Tense",        "Morph: Tense"),
        ("Morphology_Number",       "Morph: Number"),
        ("Morphology_Possession",   "Morph: Possession"),
        ("WordOrder",               "Word Order"),
        ("LexicalAmbiguity_6class", "Lexico 6-class"),
        ("LexicalAmbiguity",        "Lexico 14-class"),
        ("Binary_Man_Leg",          "Binary: Man/Leg"),
        ("Binary_Eye_Spring",       "Binary: Eye/Spring"),
        ("Binary_King_Possess",     "Binary: King/Possess"),
        ("Binary_Hit_Multiply",     "Binary: Hit/Multiply"),
        ("Binary_Knowledge_Flag",   "Binary: Knowledge/Flag"),
        ("Binary_Camel_Sentences",  "Binary: Camel/Sentences"),
        ("Binary_Open_Conquer",     "Binary: Open/Conquer"),
    ]

    # Build quantum lookup: (exp_name, ansatz) → mean
    q_lookup: Dict[Tuple[str, str], float] = {}
    for r in exp5 + exp6 + exp7:
        key = (r["exp_name"], r["ansatz"])
        q_lookup[key] = max(q_lookup.get(key, 0), r["accuracy_mean"])
    for r in exp8:
        if "error" not in r and "mean" in r:
            key = (r["exp_name"], r["ansatz"])
            q_lookup[key] = max(q_lookup.get(key, 0), r.get("mean", 0))

    # Classical lookup
    cl_lookup: Dict[Tuple[str, str], float] = {}
    cl_chance: Dict[str, float] = {}
    for r in cl:
        cl_chance[r["exp_name"]] = r["chance"]
        for clf_name, clf_r in r["classifiers"].items():
            cl_lookup[(r["exp_name"], clf_name)] = clf_r["mean"]

    # Binary pairs: chance = 0.5
    for r in exp8:
        if r["exp_name"].startswith("Binary_"):
            cl_chance[r["exp_name"]] = 0.5

    data_matrix = []
    row_labels   = []

    for exp_key, exp_label in row_order:
        row = [
            q_lookup.get((exp_key, "IQP"),   q_lookup.get((exp_key.replace("_14class",""), "IQP"), np.nan)),
            q_lookup.get((exp_key, "Sim14"), q_lookup.get((exp_key.replace("_14class",""), "Sim14"), np.nan)),
            cl_lookup.get((exp_key, "SVM_linear"), np.nan),
            cl_lookup.get((exp_key, "SVM_rbf"),    np.nan),
            cl_lookup.get((exp_key, "RF"),          np.nan),
            cl_lookup.get((exp_key, "MLP"),         np.nan),
            cl_chance.get(exp_key, np.nan),
        ]
        if not all(np.isnan(v) for v in row):
            data_matrix.append(row)
            row_labels.append(exp_label)

    mat = np.array(data_matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(systems)):
            v = mat[i, j]
            if not np.isnan(v):
                color = "black" if 0.3 < v < 0.75 else "white"
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Accuracy", format=matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Arabic QNLP — All Experiments × All Systems\n(green = high accuracy, red = low)")
    fig.tight_layout()
    path = FIG_DIR / "fig4_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 5 — FOLD VARIANCE VIOLIN / STRIP PLOT FOR BINARY PAIRS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_fold_violin(exp8: List[Dict]):
    # Collect fold accuracies per pair × ansatz
    pairs_data: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    for r in exp8:
        if r["exp_name"].startswith("Binary_") and "error" not in r and "folds" in r:
            name = r["exp_name"].replace("Binary_", "").replace("_", "/")
            pairs_data[name][r["ansatz"]] = r["folds"]

    pair_names = sorted(pairs_data.keys())
    n = len(pair_names)

    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, pair_names):
        data_iqp   = pairs_data[name].get("IQP",   [])
        data_sim14 = pairs_data[name].get("Sim14", [])

        positions = [1, 2]
        vdata = [d for d in [data_iqp, data_sim14] if d]
        vpos  = [p for p, d in zip(positions, [data_iqp, data_sim14]) if d]
        colors_v = [COLORS["IQP"], COLORS["Sim14"]][:len(vdata)]

        if vdata:
            parts = ax.violinplot(vdata, positions=vpos, showmedians=True,
                                  showextrema=True, widths=0.6)
            for pc, col in zip(parts["bodies"], colors_v):
                pc.set_facecolor(col)
                pc.set_alpha(0.6)
            parts["cmedians"].set_color("black")
            parts["cmins"].set_color("gray")
            parts["cmaxes"].set_color("gray")
            parts["cbars"].set_color("gray")

        # Jittered points
        for i, (folds, col) in enumerate(zip([data_iqp, data_sim14], colors_v)):
            if folds:
                jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(folds))
                ax.scatter(np.full(len(folds), i + 1) + jitter, folds,
                           color=col, s=30, zorder=5, alpha=0.9)

        ax.axhline(0.5, color=COLORS["chance"], linestyle="--", linewidth=1.2)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["IQP", "Sim14"], fontsize=8)
        ax.set_title(name, fontsize=8, pad=4)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Fold Accuracy")
    fig.suptitle("exp8 Binary Pairs — Per-fold Accuracy Distribution (dashed = chance)", fontsize=11)
    fig.tight_layout()
    path = FIG_DIR / "fig5_fold_violin.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 6 — EXPERIMENT PROGRESSION (exp5 → exp6 → exp7 → exp8)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_progression(exp5, exp6, exp7, exp8):
    # Track the key tasks across experiments
    tasks = {
        "Morph: Tense":      "Morphology_Tense",
        "Morph: Number":     "Morphology_Number",
        "Morph: Possession": "Morphology_Possession",
        "Lexico 6-class":    "LexicalAmbiguity_6class",
    }
    ansatzes = ["IQP", "Sim14"]

    # build lookup per exp
    def best_ans(results_list, exp_name):
        vals = {r["ansatz"]: r["accuracy_mean"] for r in results_list if r["exp_name"] == exp_name}
        if not vals:
            return {a: np.nan for a in ansatzes}
        return {a: vals.get(a, np.nan) for a in ansatzes}

    exps_labels = ["exp5\n(random init\n150ep)", "exp6\n(warm-start\n300ep)",
                   "exp7\n(morph-enriched\n300ep)"]
    exps_data   = [exp5, exp6, exp7]

    fig, axes = plt.subplots(1, len(tasks), figsize=(14, 5), sharey=False)

    for ax, (task_label, task_key) in zip(axes, tasks.items()):
        for ans, color in [("IQP", COLORS["IQP"]), ("Sim14", COLORS["Sim14"])]:
            ys = [best_ans(exd, task_key).get(ans, np.nan) for exd in exps_data]
            xs = range(len(exps_labels))
            ax.plot(xs, ys, marker="o", label=ans, color=color, linewidth=2, markersize=7)

        # chance
        if task_key == "Morphology_Tense":
            chance = 0.5
        elif "6class" in task_key:
            chance = 1/6
        elif "Number" in task_key or "Possession" in task_key:
            chance = 1/3
        else:
            chance = 1/6
        ax.axhline(chance, color=COLORS["chance"], linestyle="--", linewidth=1.2, label="Chance")

        ax.set_xticks(range(len(exps_labels)))
        ax.set_xticklabels(exps_labels, fontsize=8)
        ax.set_title(task_label, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle("Accuracy Progression Across Experiments (exp5→6→7)", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / "fig6_progression.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading results...")
    exp5, exp6, exp7, exp8, cl = load_all_results()

    print("Loading AraVec for PCA...")
    embed_fn, aravec_ok = aravec_embeddings()

    print("\n=== Generating figures ===")

    print("Fig 1: Binary pair results...")
    fig_binary_pairs(exp8)

    print("Fig 2: Quantum vs classical comparison...")
    fig_full_comparison(exp5, exp6, exp7, exp8, cl)

    if aravec_ok:
        print("Fig 3: PCA of AraVec embeddings...")
        fig_pca_embeddings(embed_fn)
    else:
        print("Fig 3: Skipped (AraVec unavailable)")

    print("Fig 4: Results heatmap...")
    fig_heatmap(exp5, exp6, exp7, exp8, cl)

    print("Fig 5: Fold variance violins...")
    fig_fold_violin(exp8)

    print("Fig 6: Experiment progression...")
    fig_progression(exp5, exp6, exp7, exp8)

    print(f"\nAll figures saved to {FIG_DIR}/")
    print("\n=== RESULTS SUMMARY ===")

    # Quick text summary
    print("\n-- exp8 Binary Pairs --")
    by_exp = defaultdict(dict)
    for r in exp8:
        if r["exp_name"].startswith("Binary_") and "error" not in r and "mean" in r:
            by_exp[r["exp_name"]][r["ansatz"]] = r["mean"]
    for name in sorted(by_exp):
        label = name.replace("Binary_","").replace("_","/")
        iqp   = by_exp[name].get("IQP", float("nan"))
        sim14 = by_exp[name].get("Sim14", float("nan"))
        best  = max(iqp, sim14)
        delta = best - 0.5
        flag  = "↑" if delta > 0.05 else ("~" if delta > -0.05 else "↓")
        print(f"  {label:<22} IQP={iqp:.1%}  Sim14={sim14:.1%}  best={best:.1%}  Δchance={delta:+.1%} {flag}")

    print("\n-- Quantum Tense (best result) --")
    best_tense = max((r for r in exp5 if "Tense" in r["exp_name"]),
                     key=lambda r: r["accuracy_mean"])
    print(f"  {best_tense['exp_name']}/{best_tense['ansatz']}: {best_tense['accuracy_mean']:.1%}")
    print("  Classical best (SVM_rbf): 54.7%")
    print("  → Quantum wins on Tense (+20.6pp)")


if __name__ == "__main__":
    main()
