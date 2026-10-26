# -*- coding: utf-8 -*-
"""readout_arity_figure.py -- Fig 3 candidate: the readout-arity budget, and
what it buys on real semantic categories.

Rendered from `diagnostic_readout_arity_bounds.json`; kept in its own module so
the assembly code and the drawing code can be read separately, exactly as
`capacity_validation_figure.py` is kept apart from `capacity_validation.py`.

FORM (dataviz step 1). Two panels, two different jobs.
  Panel 1 -- "how many arcs does each readout arity actually buy, against the
      analytic bound?" That is MAGNITUDE over an ordered discrete x (M), so:
      bars for the two measured/found values, and REFERENCE MARKS (not bars)
      for the two analytic bounds. Bounds are not a series to be compared by
      area -- they are a ceiling the bars sit under, which is what a reference
      rule reads as.
  Panel 2 -- "how does each category's reachable accuracy move with M?" That is
      CHANGE OVER AN ORDERED VARIABLE for four entities, so: lines with
      markers, one hue per category. M is drawn on EVENLY SPACED category
      positions (2, 4, 8, 12), never on a linear numeric axis -- the spacing is
      a design choice about arity steps, not a measured interval, and a linear
      axis would invent a slope.

COLOR (dataviz steps 2-3). Categorical -- identity, not magnitude, so no ramp.
The four slots are the SAME four this programme already ships in
`fig_capacity_validation`, in the same order, so the two figures read as one
system. Re-validated on the light surface #fcfcfb with the skill's validator:

    node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#4a3aa7" \
         --mode light --pairs all
    [PASS] lightness band | [PASS] chroma floor
    [PASS] CVD separation      worst all-pairs #1baf7a<->#eb6834 dE 9.2 (deutan)
    [PASS] normal-vision floor worst all-pairs #4a3aa7<->#2a78d6 dE 16.3
    [WARN] contrast vs surface #1baf7a 2.74 -- RELIEF SHIPPED: every category
           line is DIRECT-LABELLED at its right end and the complete table view
           is emitted as fig_arc_budget_data.json.

Panel 1 needs only two slots and takes slot 1 (this file) against neutral ink
with a HOLLOW face (doc 50 SSC2, a REFERENCE series, not a lever) -- the same
composite hue x fill encoding the capacity figure uses for its comparator,
rather than a generated hue.

SECONDARY ENCODING carries the source, so it is never colour-alone:
    filled marker, solid line   this file (recomputed)
    hollow marker, no line      doc 50 SSC3 as published

MODE. A paper figure: one selected light theme painted explicitly (opaque
surface, no transparency) so it renders identically on a light page, a dark page
and in print. Not auto-flipped for dark mode.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402

# ---- design tokens (dataviz reference palette, light surface) -------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

CAT_ORDER = ["tertiary", "uniformed", "manual", "nonoccupational"]
CAT_COLOR = dict(zip(CAT_ORDER, SLOT))
CAT_LABEL = {
    "tertiary": "tertiary-educated profession",
    "uniformed": "uniformed / disciplined service",
    "manual": "manual / skilled trade",
    "nonoccupational": "non-occupational state",
}
# right-end direct labels, nudged apart by hand where lines converge
CAT_LABEL_DY = {"tertiary": 0.0, "uniformed": 0.0, "manual": 0.0,
                "nonoccupational": 0.0}

M_LIST = [2, 4, 8, 12]


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.8, ls="-", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def render(blob, svg_path, png_path, data_path):
    A = blob["part_A_found_arcs"]["table"]
    Bt = blob["part_B_category_ceilings"]["table"]
    k2 = blob["part_B_category_ceilings"]["k2_cross_term"]

    Ms = sorted(int(k) for k in A)
    found = [A[str(m)]["best_found_arcs"] for m in Ms]
    pub = [A[str(m)]["doc50_C2_published"] for m in Ms]
    ds = [A[str(m)]["ds_bound_2M_minus_1"] for m in Ms]
    mx = [A[str(m)]["exact_maximum_2M_minus_2"] for m in Ms]

    fig = plt.figure(figsize=(13.4, 7.6))
    fig.patch.set_facecolor(SURFACE)
    ax1 = fig.add_axes((0.052, 0.285, 0.372, 0.495))
    ax2 = fig.add_axes((0.545, 0.285, 0.252, 0.495))

    # ------------------------------------------------ panel 1: the arc budget
    x = list(range(len(Ms)))
    w = 0.30
    gap = 0.02                       # 2px-equivalent surface gap between fills
    for i, xx in enumerate(x):
        ax1.bar(xx - w / 2 - gap / 2, pub[i], width=w, color=SURFACE,
                edgecolor=INK, linewidth=1.4, zorder=3)
        ax1.bar(xx + w / 2 + gap / 2, found[i], width=w, color=SLOT[0],
                edgecolor=SURFACE, linewidth=1.1, zorder=3)
    for i, xx in enumerate(x):
        ax1.plot([xx - 0.42, xx + 0.42], [mx[i]] * 2, color=INK_2, lw=2.0,
                 solid_capstyle="round", zorder=5)
        ax1.plot([xx - 0.42, xx + 0.42], [ds[i]] * 2, color=MUTED, lw=1.6,
                 ls=(0, (3.5, 2.5)), dash_capstyle="round", zorder=4)
        # selective direct labels: the two values the reading turns on
        ax1.text(xx + w / 2 + gap / 2, found[i] + 0.28, "%d" % found[i],
                 color=INK, fontsize=9.0, ha="center", va="bottom", zorder=6)
        if pub[i] != found[i]:
            ax1.text(xx - w / 2 - gap / 2, pub[i] + 0.28, "%d" % pub[i],
                     color=INK_2, fontsize=9.0, ha="center", va="bottom",
                     zorder=6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["M = %d" % m for m in Ms])
    ax1.set_xlim(-0.62, len(Ms) - 0.38)
    ax1.set_ylim(0, 18.4)
    ax1.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
    _style(ax1)
    ax1.set_ylabel("decision arcs carved on the circle at K = 1",
                   color=INK_2, fontsize=10.0, labelpad=8)
    ax1.set_xlabel("readout arity   (argmax over M outcomes, alternating grouping)",
                   color=INK_2, fontsize=10.0, labelpad=8)

    h1 = [
        Patch(facecolor=SLOT[0], edgecolor=SURFACE, linewidth=1.1,
              label="best found here (search + construction)"),
        Patch(facecolor=SURFACE, edgecolor=INK, linewidth=1.4,
              label="doc 50 §C2 as published"),
        Line2D([], [], color=INK_2, lw=2.0,
               label="achievable maximum 2M − 2  (proved here)"),
        Line2D([], [], color=MUTED, lw=1.6, ls=(0, (3.5, 2.5)),
               label="Davenport–Schinzel bound 2M − 1"),
    ]
    lg1 = ax1.legend(handles=h1, loc="upper left", bbox_to_anchor=(0.006, 0.995),
                     frameon=False, fontsize=8.6, labelspacing=0.55,
                     handletextpad=0.7, borderpad=0.15, labelcolor=INK_2)
    lg1._legend_box.align = "left"

    # -------------------------------------- panel 2: what the arity buys
    xs = list(range(len(M_LIST)))
    for key in CAT_ORDER:
        col = CAT_COLOR[key]
        cells = Bt[key]["M_ary_per_twin"]
        ys = [cells[str(m)]["value"] for m in M_LIST]
        yp = [cells[str(m)]["doc50_C3_published"] for m in M_LIST]
        ax2.plot(xs, ys, color=col, lw=2.0, zorder=4, solid_capstyle="round")
        ax2.scatter(xs, ys, s=46, facecolors=col, edgecolors=SURFACE,
                    linewidths=1.4, zorder=5)
        # the M=2 column of doc 50 SSC3 IS doc 47 SS13's pair-level anchor, and
        # is drawn as the tick below -- not repeated as an M-ary cell it is not
        ax2.scatter(xs[1:], yp[1:], s=40, facecolors=SURFACE, edgecolors=col,
                    linewidths=1.5, zorder=5)
        # the doc 47 §13 binary anchor: where the category started
        ax2.scatter([xs[0]], [Bt[key]["binary_pair_level"]["value"]], s=34,
                    marker="_", color=col, linewidths=2.0, zorder=6)
        CAT_LABEL_DY[key] = ys[-1]

    # right-end direct labels, pushed apart deterministically where lines
    # converge (relief for the aqua contrast WARN: every line is labelled)
    lab = sorted(CAT_ORDER, key=lambda k: CAT_LABEL_DY[k])
    place, MINSEP = {}, 0.0135
    for i, key in enumerate(lab):
        yv = CAT_LABEL_DY[key]
        if i and yv - place[lab[i - 1]] < MINSEP:
            yv = place[lab[i - 1]] + MINSEP
        place[key] = yv
    for key in CAT_ORDER:
        ax2.plot([len(M_LIST) - 1, len(M_LIST) - 1 + 0.11],
                 [CAT_LABEL_DY[key], place[key]], color=CAT_COLOR[key],
                 lw=0.9, zorder=6, clip_on=False)
        ax2.text(len(M_LIST) - 1 + 0.15, place[key], CAT_LABEL[key],
                 color=CAT_COLOR[key], fontsize=8.7, ha="left", va="center",
                 zorder=7)

    # the one K=2 cross-term
    kx = M_LIST.index(k2["M"])
    ax2.scatter([kx], [k2["value"]], s=64, marker="D",
                facecolors=CAT_COLOR[k2["category"]], edgecolors=INK,
                linewidths=1.2, zorder=8)
    ax2.annotate("K = 2, same readout\n%.4f" % k2["value"],
                 xy=(kx, k2["value"]), xytext=(kx + 0.10, 0.926),
                 color=INK, fontsize=8.4, ha="left", va="center",
                 linespacing=1.35, zorder=9,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9,
                                 shrinkA=3, shrinkB=6))

    ax2.set_xticks(xs)
    ax2.set_xticklabels(["M = %d" % m for m in M_LIST])
    ax2.set_xlim(-0.30, len(M_LIST) - 0.88)
    ax2.set_ylim(0.815, 1.045)
    ax2.set_yticks([0.85, 0.90, 0.95, 1.00])
    _style(ax2)
    ax2.set_ylabel("reachable per-twin accuracy   (constructive lower bound)",
                   color=INK_2, fontsize=10.0, labelpad=8)
    ax2.set_xlabel("readout arity   (encoding UNCHANGED, K = 1 unless marked)",
                   color=INK_2, fontsize=10.0, labelpad=8)

    h2 = [
        Line2D([], [], marker="o", linestyle="-", color=MUTED, lw=2.0,
               markersize=7.5, markerfacecolor=MUTED, markeredgecolor=SURFACE,
               markeredgewidth=1.4, label="recomputed here"),
        Line2D([], [], marker="o", linestyle="none", markersize=7.5,
               markerfacecolor=SURFACE, markeredgecolor=MUTED,
               markeredgewidth=1.5, label="doc 50 §C3 as published"),
        Line2D([], [], marker="_", linestyle="none", markersize=8,
               color=MUTED, markeredgewidth=2.0,
               label="doc 47 §13 binary anchor (pair level)"),
        Line2D([], [], marker="D", linestyle="none", markersize=7.0,
               markerfacecolor=MUTED, markeredgecolor=INK, markeredgewidth=1.2,
               label="K = 2 cross-term"),
    ]
    lg2 = fig.legend(handles=h2, loc="lower left", ncol=2,
                     bbox_to_anchor=(0.545, 0.150), frameon=False, fontsize=8.6,
                     columnspacing=2.2, labelspacing=0.55, handletextpad=0.7,
                     borderpad=0.0, labelcolor=INK_2)
    lg2._legend_box.align = "left"

    # ------------------------------------------------------------- furniture
    fig.text(0.008, 0.977, "The bottleneck was the readout, and its budget is exactly 2M \u2212 2",
             color=INK, fontsize=15.5, fontweight="bold", va="top", ha="left")
    fig.text(0.008, 0.938,
             "LEFT: a hard argmax over M outcomes carves at most 2M \u2212 2 decision arcs on the K = 1 circle. The Davenport\u2013Schinzel envelope gives 2M \u2212 1; a closed curve forces an\n"
             "even count, which removes the odd value; and a parallel-gradient construction attains 2M \u2212 2. So this is a MAXIMUM, not a best-found \u2014 which is what doc 50 \u00a7C2 said it did\n"
             "not have. RIGHT: what that budget reaches on doc 47 \u00a713's four real AraVec categories, encoding untouched \u2014 constructive lower bounds over all ordered twins of the 24 nouns.",
             color=INK_2, fontsize=9.2, va="top", ha="left", linespacing=1.5)

    fig.text(0.008, 0.108,
             "Left panel counts arcs EXACTLY \u2014 closed-form roots of every pairwise difference, class read at the midpoint of consecutive roots \u2014 never on a grid, because a grid can only miss arcs. Right panel: hard argmax throughout, outcome\u2192class\n"
             "grouping FIXED alternating (doc 53 \u00a73.1); the M-ladder is nested by construction (each M is seeded with the previous M\u2019s witness plus dead outcomes), so a cell can never dip below its predecessor \u2014 doc 50 \u00a7C3\u2019s asterisked uniformed / M = 4 cell\n"
             "is exactly that failure mode and it is unreachable here. Binary anchors are doc 47 \u00a713\u2019s EXACT pair-level ceilings over CASTS; the M-ary column is per-twin over 2|B||N| twins, the denominator doc 50 \u00a7C3\u2019s own arithmetic forces (0.9105 = 173/190).\n"
             "Two different populations, both stated (doc 47 \u00a718.5). These are REPRESENTATIONAL bounds \u2014 what a readout can express, never what an optimiser will find; exp67 measured conversion and it FAILED for the circuit. Simulation, one task family, one inventory.\n"
             "Table view: fig_arc_budget_data.json   \u00b7   full instrument output: diagnostic_readout_arity_bounds.json",
             color=MUTED, fontsize=7.2, va="top", ha="left", linespacing=1.6)

    fig.savefig(svg_path, format="svg", facecolor=SURFACE, dpi=300)
    fig.savefig(png_path, format="png", facecolor=SURFACE, dpi=300)
    plt.close(fig)

    payload = {
        "figure": os.path.basename(svg_path),
        "source": "diagnostic_readout_arity_bounds.json",
        "what_is_plotted": (
            "panel 1: the Part A found-arc table (best found here, doc 50 SSC2 as "
            "published, the proved maximum 2M-2 and the Davenport-Schinzel bound "
            "2M-1) for every M in the table. panel 2: the Part B per-twin M-ary "
            "ceilings for all four doc 47 SS13 categories at M in {2,4,8,12}, the "
            "doc 50 SSC3 published value of each cell, the doc 47 SS13 pair-level "
            "binary anchor, and the one K=2 cross-term."),
        "encoding": {
            "panel1_x": "readout arity M, evenly spaced categorical positions",
            "panel1_y": "decision arcs carved on the circle at K=1 (integer count)",
            "panel1_bars": "filled = recomputed here; hollow with ink edge = doc 50 SSC2",
            "panel1_rules": "solid mid-ink = achievable maximum 2M-2; dashed muted = DS 2M-1",
            "panel2_x": ("readout arity M in {2,4,8,12} on EVENLY SPACED categorical "
                         "positions -- arity steps are a design choice, not a measured "
                         "interval, so a linear numeric axis would invent a slope"),
            "panel2_y": "reachable per-twin accuracy (constructive lower bound)",
            "colour": ("categorical, one slot per doc 47 SS13 category, in the same "
                       "four validated slots and the same order as "
                       "fig_capacity_validation, so the two figures read as one system"),
            "secondary_encoding": ("filled marker + solid line = recomputed here; "
                                   "hollow marker, no line = doc 50 SSC3 as published; "
                                   "tick = doc 47 SS13 pair-level binary anchor; "
                                   "diamond = K=2 cross-term. Source is therefore never "
                                   "carried by colour alone."),
        },
        "palette": {
            "surface": SURFACE,
            "categories": CAT_COLOR,
            "reference_series": {"doc50_C2_bars": "hollow, ink edge %s" % INK,
                                 "exact_maximum_rule": INK_2,
                                 "ds_bound_rule": MUTED},
            "validator": ("node scripts/validate_palette.js "
                          "\"#2a78d6,#eb6834,#1baf7a,#4a3aa7\" --mode light "
                          "--pairs all -> ALL CHECKS PASS (CVD dE 9.2 deutan, "
                          "normal-vision dE 16.3); aqua contrast WARN 2.74, relief "
                          "= every line direct-labelled + this table view"),
        },
        "panel1_points": [
            {"M": m,
             "arcs_best_found_here": A[str(m)]["best_found_arcs"],
             "arcs_from_construction": A[str(m)]["arcs_from_analytic_construction"],
             "arcs_from_random_search": A[str(m)]["arcs_from_random_search"],
             "doc50_C2_published": A[str(m)]["doc50_C2_published"],
             "exact_maximum_2M_minus_2": A[str(m)]["exact_maximum_2M_minus_2"],
             "ds_bound_2M_minus_1": A[str(m)]["ds_bound_2M_minus_1"],
             "reconciliation": A[str(m)]["reconciliation"]}
            for m in Ms],
        "panel2_points": [
            {"category": key, "gloss": CAT_LABEL[key], "colour": CAT_COLOR[key],
             "category_arcs": Bt[key]["arcs"],
             "binary_pair_level_anchor": Bt[key]["binary_pair_level"]["value"],
             "binary_pair_level_fraction": Bt[key]["binary_pair_level"]["fraction"],
             "M": m,
             "y_recomputed": Bt[key]["M_ary_per_twin"][str(m)]["value"],
             "fraction": Bt[key]["M_ary_per_twin"][str(m)]["fraction"],
             "y_doc50_C3": Bt[key]["M_ary_per_twin"][str(m)]["doc50_C3_published"],
             "reconciliation": Bt[key]["M_ary_per_twin"][str(m)]["reconciliation"]}
            for key in CAT_ORDER for m in M_LIST],
        "k2_cross_term_point": {
            "category": k2["category"], "M": k2["M"], "K": 2,
            "y_recomputed": k2["value"], "fraction": k2["fraction"],
            "y_doc50_C3": k2["doc50_C3_published"],
            "reconciliation": k2["reconciliation"]},
        "caveats_shipped_on_the_figure": [
            "arcs counted exactly, never on a grid",
            "hard argmax, fixed alternating grouping (doc 53 SS3.1)",
            "M-ladder nested by construction, so no cell can dip",
            "binary anchors are PAIR-LEVEL over casts; M-ary cells are PER-TWIN "
            "over 2|B||N| -- two populations, both stated (doc 47 SS18.5)",
            "representational bounds only; exp67 measured conversion and it failed "
            "for the circuit",
        ],
    }
    with open(data_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return payload


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    blob = json.load(open(os.path.join(here, "diagnostic_readout_arity_bounds.json"),
                          encoding="utf-8"))
    p = render(blob,
               os.path.join(here, "fig_arc_budget.svg"),
               os.path.join(here, "fig_arc_budget.png"),
               os.path.join(here, "fig_arc_budget_data.json"))
    print("wrote fig_arc_budget.svg / .png / _data.json (%d + %d points)"
          % (len(p["panel1_points"]), len(p["panel2_points"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
