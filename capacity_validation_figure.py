# -*- coding: utf-8 -*-
"""capacity_validation_figure.py -- the flagship prediction-vs-measurement figure.

Rendered by `capacity_validation.py`; kept in its own module so the assembly
code and the drawing code can be read separately.

FORM (dataviz step 1). The data's job is "does a predicted quantity match a
measured one?" over ~80 independent (prediction, measurement) pairs. That is a
scatter against the identity line -- the canonical form for this job. Not a bar
chart (no magnitude ranking), not a line (no ordered x).

COLOR (dataviz steps 2-3). Categorical, by lever family -- identity, not
magnitude, so no ramp. FOUR categorical slots carry the four CIRCUIT levers.
Validated with the skill's own validator, all-pairs (the scatter pairlist), on
the light surface #fcfcfb:

    node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#4a3aa7" \
         --mode light --pairs all
    [PASS] lightness band | [PASS] chroma floor
    [PASS] CVD separation      worst all-pairs #1baf7a<->#eb6834 dE 9.2 (deutan)
    [PASS] normal-vision floor worst all-pairs #4a3aa7<->#2a78d6 dE 16.3
    [WARN] contrast vs surface #1baf7a 2.74 -- RELIEF SHIPPED: the called-out
           points are direct-labelled and the complete table view is emitted as
           fig_capacity_validation_data.json (and diagnostic_capacity_validation.json).

The classical comparator is deliberately NOT a fifth categorical hue -- a fifth
slot fails the all-pairs gate, and a comparator is a REFERENCE series, not a
lever. It is drawn in neutral ink with a HOLLOW face, which is the strongest
non-hue distinction available and is colourblind- and print-safe. Floors are
recessive muted grey. This is the skill's sanctioned composite encoding
(hue x shape x fill) rather than a generated hue.

SHAPE carries the honest caveat, so it cannot be missed (doc 47 SS18.5):
    circle  the bound is over the POPULATION THAT WAS TRAINED ON
    square  the bound is over BAND CENTRES -- doc 47 SS18.1: NOT a bound over a
            finite sampled noun inventory, so a square below the diagonal may be
            a population mismatch rather than a conversion failure.

MODE. This is a paper figure: one selected light theme, painted explicitly (an
opaque surface, no transparency) so it renders identically on a light page, a
dark page and in print. It is not auto-flipped for dark mode.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402

# ---- design tokens (dataviz reference palette, light surface) -------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

FAMILY_COLOR = {
    "K1-single-wire": "#2a78d6",      # slot 1 blue
    "K2-within-wire": "#eb6834",      # slot 2 orange
    "width-register": "#1baf7a",      # slot 3 aqua
    "M-ary-readout": "#4a3aa7",       # slot 4 violet
    "classical-comparator": INK,      # reference series, neutral ink, hollow
    "floor": MUTED,                   # recessive control
}
FAMILY_ORDER = ["K1-single-wire", "width-register", "K2-within-wire",
                "M-ary-readout", "classical-comparator", "floor"]
FAMILY_LABEL = {
    "K1-single-wire": "K = 1, single wire",
    "width-register": "width register (one RY per wire)",
    "K2-within-wire": "K = 2, within-wire spectrum",
    "M-ary-readout": "M-ary readout",
    "classical-comparator": "classical comparator (same features)",
    "floor": "label-shuffled floor",
}

BAND_POP = "band centres"


def _shape(r):
    """circle = bound over the trained population; square = band centres."""
    return "s" if r["bound_provenance"]["population"].startswith(BAND_POP) else "o"


# points called out by name (the ones the reading turns on). Labelled
# SELECTIVELY -- a value on every point is an anti-pattern.
LABELS = [
    {"experiment": "exp68", "arm": "F2",
     "text": "exp68 F2  width register\n0.959 vs bound 1.000",
     "at": (1.010, 0.972), "ha": "left"},
    {"experiment": "exp67", "arm": "R8", "cell": "M2",
     "text": "exp67 R8  M-ary readout\n0.739 vs bound 1.000",
     "at": (1.010, 0.772), "ha": "left"},
    {"experiment": "exp67", "arm": "R8", "cell": "M3",
     "text": "exp67 R8 (M3)\n0.697 vs bound 1.000",
     "at": (1.010, 0.664), "ha": "left"},
]

DODGE_STEP = 0.0042      # data units; disclosed in the caption and the data JSON
DODGE_SPAN = 0.011       # a cluster is never spread wider than this, so the
                         # offset can never be read as a difference in bound
DODGE_Y_TOL = 0.016      # two points closer than this in y are dodged apart
X_CEILING = 1.0          # no mark is ever drawn to the right of a bound of 1.0


def _dodge(pts):
    """Deterministic horizontal offset for points that would otherwise land on
    top of each other. The bounds are highly discrete (0.6875, 0.75, 0.875,
    1.0, ...), so many rows share an x. Only points whose y also nearly
    coincides are moved; `x_bound` is preserved verbatim in the emitted data."""
    for r in pts:
        r["_x"] = r["bound"]
    groups = {}
    for r in pts:
        # floors are never dodged -- they are a recessive cloud, and sorting a
        # cloud by y while offsetting x would draw a trend that is not there
        if r["role"] == "floor":
            continue
        groups.setdefault(round(r["bound"], 6), []).append(r)
    for x0, g in groups.items():
        g.sort(key=lambda r: r["measured_mean"])
        cluster = [g[0]]
        for r in g[1:]:
            if r["measured_mean"] - cluster[-1]["measured_mean"] < DODGE_Y_TOL:
                cluster.append(r)
            else:
                _spread(cluster, x0)
                cluster = [r]
        _spread(cluster, x0)


def _spread(cluster, x0):
    n = len(cluster)
    if n < 2:
        return
    step = min(DODGE_STEP, DODGE_SPAN / (n - 1))
    xs = [x0 + (i - (n - 1) / 2.0) * step for i in range(n)]
    over = max(xs) - X_CEILING
    if over > 0:                       # translate, never draw past the ceiling
        xs = [x - over for x in xs]
    for r, x in zip(cluster, xs):
        r["_x"] = x


def _plotted(rows):
    """Primary rows plus the floors. `context` rows (exp60's 3-seed screen,
    exp63's subsampled levels, exp62-C's sd-less S1/S2 means) are in the JSON
    but are not plotted: they duplicate or under-specify a plotted point."""
    return [r for r in rows
            if r["role"] in ("primary", "floor")
            and r["bound"] is not None and r["measured_mean"] is not None]


def render(rows, svg_path, png_path, data_path):
    pts = _plotted(rows)
    _dodge(pts)

    fig = plt.figure(figsize=(8.8, 8.6))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes((0.082, 0.253, 0.902, 0.548))
    ax.set_facecolor(SURFACE)

    x0, x1, y0, y1 = 0.655, 1.098, 0.448, 1.030
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # the region above the identity line -- nothing may sit far into it
    ax.fill_between([0.0, 1.0], [0.0, 1.0], [1.06, 1.06],
                    color=INK, alpha=0.030, lw=0, zorder=0)
    ax.plot([0.0, 1.0], [0.0, 1.0], color=INK_2, lw=1.4, alpha=0.6,
            zorder=1, solid_capstyle="round")

    ax.set_xticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(True, which="major", color=GRID, lw=0.8, ls="-", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)

    # ---- marks. Floors first (recessive), then levers, comparators on top.
    order = {"floor": 0, "K1-single-wire": 2, "width-register": 3,
             "K2-within-wire": 2, "M-ary-readout": 3,
             "classical-comparator": 4}
    for fam in sorted({r["family"] for r in pts},
                      key=lambda f: (order[f], f)):
        sel = [r for r in pts if r["family"] == fam]
        col = FAMILY_COLOR[fam]
        hollow = (fam == "classical-comparator")
        isfloor = (fam == "floor")
        for mk in ("o", "s"):
            g = [r for r in sel if _shape(r) == mk]
            if not g:
                continue
            x = [r["_x"] for r in g]
            y = [r["measured_mean"] for r in g]
            e = [r["measured_sd"] or 0.0 for r in g]
            if not isfloor and any(e):
                ax.errorbar(x, y, yerr=e, fmt="none", ecolor=col, alpha=0.30,
                            elinewidth=0.8, capsize=0,
                            zorder=order[fam] + 0.5)
            ax.scatter(x, y, marker=mk,
                       s=15 if isfloor else 58,
                       facecolors=SURFACE if hollow else col,
                       edgecolors=col if hollow else SURFACE,
                       linewidths=1.5 if hollow else 1.1,
                       alpha=0.5 if isfloor else 0.95,
                       zorder=order[fam] + 1)

    # ---- y = x label, laid along the line in DISPLAY space
    p0 = ax.transData.transform((0.80, 0.80))
    p1 = ax.transData.transform((0.84, 0.84))
    import math as _m
    ang = _m.degrees(_m.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(0.868, 0.876, "y = x   the bound met exactly", color=INK_2,
            fontsize=8.6, ha="left", va="bottom", rotation=ang,
            rotation_mode="anchor", zorder=6)

    # ---- selective direct labels (never a number on every point)
    for spec in LABELS:
        hit = [r for r in pts
               if r["experiment"] == spec["experiment"]
               and r["arm"] == spec["arm"]
               and ("cell" not in spec or r["cell"].startswith(spec["cell"]))]
        for r in hit[:1]:
            ax.annotate(spec["text"], xy=(r["_x"], r["measured_mean"]),
                        xytext=spec["at"], color=INK, fontsize=8.6,
                        ha=spec["ha"], va="center", linespacing=1.35, zorder=7,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9,
                                        shrinkA=3, shrinkB=6))

    # ---- legends: identity (colour) and bound population (shape)
    fam_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=(SURFACE if f == "classical-comparator"
                                else FAMILY_COLOR[f]),
               markeredgecolor=(FAMILY_COLOR[f] if f == "classical-comparator"
                                else SURFACE),
               markeredgewidth=1.5 if f == "classical-comparator" else 1.1,
               alpha=0.55 if f == "floor" else 0.95,
               label=FAMILY_LABEL[f])
        for f in FAMILY_ORDER if any(r["family"] == f for r in pts)]
    shp_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=MUTED, markeredgecolor=SURFACE,
               markeredgewidth=1.1,
               label="over the trained population"),
        Line2D([], [], marker="s", linestyle="none", markersize=8,
               markerfacecolor=MUTED, markeredgecolor=SURFACE,
               markeredgewidth=1.1,
               label="over BAND CENTRES (doc 47 §18.1)"),
    ]
    l1 = ax.legend(handles=fam_handles, loc="upper left",
                   bbox_to_anchor=(0.004, 0.998), frameon=False, fontsize=8.7,
                   labelspacing=0.60, handletextpad=0.7, borderpad=0.15,
                   labelcolor=INK_2, title="lever family",
                   title_fontproperties={"size": 8.9, "weight": "bold"})
    l1.get_title().set_color(INK)
    l1._legend_box.align = "left"
    ax.add_artist(l1)
    # the shape key sits BELOW the plot, where it can collide with nothing
    l2 = fig.legend(handles=shp_handles, loc="lower left", ncol=2,
                    bbox_to_anchor=(0.082, 0.120), frameon=False, fontsize=8.7,
                    columnspacing=2.6, handletextpad=0.7, borderpad=0.0,
                    labelcolor=INK_2, title="what the bound bounds",
                    title_fontproperties={"size": 8.9, "weight": "bold"})
    l2.get_title().set_color(INK)
    l2._legend_box.align = "left"

    ax.set_xlabel("computed capacity bound   (reachable held-out accuracy)",
                  color=INK_2, fontsize=10.5, labelpad=9)
    ax.set_ylabel("trained held-out accuracy   (mean over seeds, ±1 sd)",
                  color=INK_2, fontsize=10.5, labelpad=9)

    fig.text(0.010, 0.978, "Supply is not conversion", color=INK,
             fontsize=15.5, fontweight="bold", va="top", ha="left")
    fig.text(0.010, 0.941,
             "Every committed (computed bound, trained held-out accuracy) pair "
             "the programme has produced.\n"
             "Arms that add a wire at one RY per wire reach their bound; arms "
             "that add spectrum or readout arity\n"
             "inside the same wire do not. Classical comparators on the "
             "identical features convert what is there.",
             color=INK_2, fontsize=9.4, va="top", ha="left", linespacing=1.5)

    fig.text(0.010, 0.100,
             "Bounds are LP-exact over band geometry (exp60 / exp61 / exp63 / "
             "exp67) or constructive lower bounds and per-twin MAX-FS over the "
             "inventory actually\n"
             "trained on (exp62 / exp65 / exp68), so a square below the line "
             "may be a population mismatch rather than a conversion failure "
             "(doc 47 §18.1). Points\n"
             "sharing an x carry a deterministic offset (≤ ±0.006, never past "
             "1.0) so they do not overlap; exact and offset x are both in the "
             "data file. Floors are not\n"
             "offset. Simulation throughout, product-state encodings, one task "
             "family — no quantum-advantage reading is licensed.\n"
             "Table view: fig_capacity_validation_data.json   ·   all rows "
             "including context: diagnostic_capacity_validation.json",
             color=MUTED, fontsize=7.3, va="top", ha="left", linespacing=1.6)

    fig.savefig(svg_path, format="svg", facecolor=SURFACE, dpi=300)
    fig.savefig(png_path, format="png", facecolor=SURFACE, dpi=300)
    plt.close(fig)

    payload = {
        "figure": os.path.basename(svg_path),
        "what_is_plotted": ("rows with role 'primary' or 'floor' from "
                            "diagnostic_capacity_validation.json; 'context' rows "
                            "are excluded (they duplicate or under-specify a "
                            "plotted point)"),
        "encoding": {
            "x": "bound", "y": "measured_mean", "yerr": "measured_sd (1 sd)",
            "colour": "lever family (4 validated categorical slots; the "
                      "classical comparator is neutral ink + hollow face, not a "
                      "5th hue)",
            "shape": "circle = bound over the trained population; square = "
                     "bound over band centres (doc 47 SS18.1)",
            "reference": "identity line y = x",
            "x_offset": ("points sharing an x are given a deterministic "
                         "horizontal offset of +/- %.4f so they do not overlap; "
                         "`x_bound` is the true value, `x_plotted` is where the "
                         "mark was drawn" % DODGE_STEP),
        },
        "palette": {"surface": SURFACE, "families": FAMILY_COLOR,
                    "validator": "node scripts/validate_palette.js "
                                 "\"#2a78d6,#eb6834,#1baf7a,#4a3aa7\" "
                                 "--mode light --pairs all -> PASS "
                                 "(CVD 9.2, normal-vision 16.3); aqua contrast "
                                 "WARN 2.74, relief = direct labels + table view"},
        "n_points": len(pts),
        "labelled_points": [s["text"].replace("\n", " ") for s in LABELS],
        "points": [
            {"experiment": r["experiment"], "arm": r["arm"], "cell": r["cell"],
             "family": r["family"], "role": r["role"],
             "x_bound": r["bound"], "x_plotted": r["_x"],
             "y_measured": r["measured_mean"],
             "yerr_sd": r["measured_sd"], "n_seeds": r["n_seeds"],
             "conversion_ratio": r["conversion_ratio"], "gap": r["gap"],
             "marker": _shape(r), "colour": FAMILY_COLOR[r["family"]],
             "bound_population": r["bound_provenance"]["population"],
             "bound_kind": r["bound_provenance"]["kind"],
             "bound_source": "%s :: %s" % (r["bound_provenance"]["file"],
                                           r["bound_provenance"]["field"]),
             "measured_source": "%s :: %s" % (r["source_results_file"],
                                              r["measured_field"])}
            for r in pts],
    }
    with open(data_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return payload
