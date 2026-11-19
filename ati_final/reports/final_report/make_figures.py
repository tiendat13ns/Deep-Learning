"""Figures for the final report, in two styles with identical layout.

Run from ati_final:
    python reports/final_report/make_figures.py                 # black and white -> results/figures/final_report_bw/
    python reports/final_report/make_figures.py --style color   # colour          -> results/figures/final_report_color/
Reads only saved tables in results/; every figure is drawn at its printed width (16 cm) so text prints at
its true size.

Colour style (validated with the dataviz skill's validate_palette.js on a white surface; OKLab ΔE ×100,
CVD simulated with Machado et al. 2009, protan/deutan):
  decoders  fixed #d0d0cb (neutral grey = the control), global #5598e7, per-label #f0905f
            blue/orange: CVD ΔE 20.8, normal-vision ΔE 27.2; grey vs hues all-pairs CVD ≥ 14.1
  seeds     #8b7fe0 / #3aa58f / #eda100, all pairs: CVD ΔE 13.4, normal-vision ΔE 20.2
  flips     corrected #3aa58f, broken #eda100 (not red/green)
  tiers     ordinal one-hue violet ramp head #5a4fc4, mid #7f76d9, tail #a9a2ea (passes --ordinal)
Fills below 3:1 contrast on white carry a dark hairline edge and direct labels or a legend (relief rule);
all text stays black (Nature figure guide: no coloured text, keys inside the figure).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RQ1 = ROOT / "results" / "rq1"
HERE = Path(__file__).resolve().parent
W = 16 / 2.54  # printed width in inches
STRATEGIES = ("fixed", "global", "per_label")
NAMES = {"fixed": "Fixed (0.5)", "global": "Global tuned", "per_label": "Per-label tuned"}

STYLES = {
    "bw": dict(
        out="final_report_bw", edge="black", ink="black", grid="#d9d9d9",
        fill={"fixed": "white", "global": "#bdbdbd", "per_label": "#4d4d4d"},
        hatch={"fixed": "", "global": "///", "per_label": ""},
        bar_text={"fixed": "black", "global": "black", "per_label": "white"},
        tier_fill={"head": "#4d4d4d", "mid": "#a6a6a6", "tail": "white"},
        tier_hatch={"head": "", "mid": "", "tail": "////"},
        seed_fc=("black", "white", "#9e9e9e"), range_line="#7f7f7f",
        flip_gain="#bdbdbd", flip_loss="#4d4d4d", band="#e0e0e0", diag="black",
        box={"model": "white", "val": "white", "test": "#e6e6e6", "calib": "white", "freeze": "white",
             "decode": "white", "robust": "white"},
    ),
    "color": dict(
        out="final_report_color", edge="#52514e", ink="#3a3936", grid="#e1e0d9",
        fill={"fixed": "#d0d0cb", "global": "#5598e7", "per_label": "#f0905f"},
        hatch={"fixed": "", "global": "", "per_label": ""},
        bar_text={"fixed": "black", "global": "black", "per_label": "black"},
        tier_fill={"head": "#5a4fc4", "mid": "#7f76d9", "tail": "#a9a2ea"},
        tier_hatch={"head": "", "mid": "", "tail": ""},
        seed_fc=("#8b7fe0", "#3aa58f", "#eda100"), range_line="#b9b8b1",
        flip_gain="#3aa58f", flip_loss="#eda100", band="#dce9fb", diag="#898781",
        box={"model": "#f2f2ef", "val": "#e3eefb", "test": "#fde9df", "calib": "#e3eefb", "freeze": "#ece9fb",
             "decode": "#f2f2ef", "robust": "white"},
    ),
}
S = STYLES["bw"]
OUT = ROOT / "results" / "figures" / S["out"]

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "hatch.linewidth": 0.5,
                     "savefig.dpi": 300, "mathtext.fontset": "stix", "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=S["grid"], lw=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Figure: label distribution
def label_distribution():
    d = pd.read_csv(ROOT / "results" / "tables" / "label_distribution.csv", index_col=0).sort_values("train")
    fig, ax = plt.subplots(figsize=(W, 4.6))
    y = np.arange(len(d))
    for tier in ("head", "mid", "tail"):
        m = (d.tier == tier).to_numpy()
        ax.barh(y[m], d.train[m], height=0.72, color=S["tier_fill"][tier], hatch=S["tier_hatch"][tier],
                edgecolor=S["edge"], lw=0.5, label=f"{tier} ({m.sum()} labels)")
    for v, txt in ((750, "750"), (1000, "1,000")):
        ax.axvline(v, color=S["ink"], ls=":", lw=0.8)
        ax.text(v, len(d) - 0.2, txt, ha="center", va="bottom", fontsize=7.5)
    for yi, v in zip(y, d.train):
        ax.text(v + 25, yi, f"{v:,}", va="center", fontsize=6.5, bbox=dict(fc="white", ec="none", pad=0.2))
    ax.set_yticks(y, d.index, fontsize=7.5)
    ax.set_ylim(-0.6, len(d) + 0.4)
    ax.set_xlabel("Positive training comments (16,531 comments in total)")
    ax.legend(loc="lower right", frameon=False, title="Frequency tier (train)", title_fontsize=8)
    grid(ax, "x")
    save(fig, "fig_label_distribution.png")


# ---------------------------------------------------------------- Figure: protocol diagram
def protocol():
    fig, ax = plt.subplots(figsize=(W, 2.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 31); ax.axis("off")
    fs = 7

    def box(x, y, w, h, text, kind, ls="-"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=S["box"][kind], ec=S["edge"], lw=0.8, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=S["ink"],
                                                             shrinkA=0, shrinkB=0, mutation_scale=8))

    box(0, 10.5, 15, 10, "PhoBERT-base\nMidterm model\n(not retrained)", "model")
    box(19, 19, 16, 10, "Validation scores\n2,066 × 28", "val")
    box(19, 1, 16, 10, "Test scores\n2,067 × 28", "test")
    box(39, 19, 25, 10, "Calibrate on the grid\n0.05–0.90, step 0.025:\nglobal $\\tau^{*}$, per-label $\\tau^{*}_{k}$", "calib")
    box(68, 19, 15, 10, "Freeze\ncalibration.json\n+ source hashes", "freeze")
    box(39, 1, 44, 10, "Decode the same scores with $\\tau=0.5$, $\\tau^{*}$ and $\\tau^{*}_{k}$;\n"
                       "evaluate once: metrics, tiers, cardinality, gap", "decode")
    box(87, 1, 13, 10, "Robustness:\nbootstrap,\nseeds 43, 44", "robust", ls="--")
    arrow(15, 18, 19, 24); arrow(15, 13, 19, 6)
    arrow(35, 24, 39, 24); arrow(64, 24, 68, 24)
    arrow(75.5, 19, 75.5, 11); arrow(35, 6, 39, 6); arrow(83, 6, 87, 6)
    ax.text(27, 15, "not opened during\ncalibration", ha="center", va="center", fontsize=6.5, style="italic")
    save(fig, "fig_protocol.png")


# ---------------------------------------------------------------- Figure: thresholds across seeds
def thresholds():
    stab = pd.read_csv(RQ1 / "tables" / "seed_threshold_stability.csv")
    counts = pd.read_csv(ROOT / "results" / "tables" / "label_distribution.csv", index_col=0).train
    stab["train"] = stab.label.map(counts)
    d = stab.sort_values("train", ascending=True).reset_index(drop=True)
    seeds = [c for c in d.columns if c.startswith("seed")]
    glob = pd.read_csv(RQ1 / "tables" / "seed_results.csv").groupby("seed").global_threshold.first()
    fig, ax = plt.subplots(figsize=(W, 4.9))
    y = np.arange(len(d))
    ax.hlines(y, d[seeds].min(axis=1), d[seeds].max(axis=1), color=S["range_line"], lw=1.2, zorder=1)
    for c, mk, fc in zip(seeds, ("o", "^", "s"), S["seed_fc"]):
        ax.scatter(d[c], y, marker=mk, s=19, facecolor=fc, edgecolor=S["edge"], lw=0.6, zorder=3,
                   label=f"seed {c[4:]}")
    ax.axvline(0.5, color=S["ink"], ls=":", lw=0.9)
    ax.axvspan(glob.min(), glob.max(), color=S["band"], zorder=0)
    ax.text(0.5, len(d) - 0.3, "fixed 0.5", ha="right", va="bottom", fontsize=7)
    ax.text(glob.max() + 0.005, len(d) - 0.3, f"global τ* {glob.min():.3f}–{glob.max():.3f}", ha="left",
            va="bottom", fontsize=7)
    ax.set_yticks(y, [f"{l} ({t[0].upper()})" for l, t in zip(d.label, d.tier)], fontsize=7.5)
    ax.set_ylim(-0.6, len(d) + 0.6)
    ax.set_xlim(0.2, 0.92)
    ax.set_xlabel("Per-label threshold $\\tau^{*}_{k}$ selected on validation")
    ax.legend(loc="lower left", frameon=True, framealpha=1, edgecolor="none")
    grid(ax, "x")
    save(fig, "fig_thresholds_seeds.png")


# ---------------------------------------------------------------- Figure: Macro-F1 by tier (3 seeds)
def tiers():
    r = pd.read_csv(RQ1 / "tables" / "seed_results.csv")
    cols = [("test_macro_f1", "All 28 labels"), ("test_macro_f1_head", "Head (10)"),
            ("test_macro_f1_mid", "Mid (10)"), ("test_macro_f1_tail", "Tail (8)")]
    fig, ax = plt.subplots(figsize=(W, 2.7))
    w = 0.26
    for i, s in enumerate(STRATEGIES):
        sub = r[r.strategy == s]
        means = [sub[c].mean() for c, _ in cols]
        stds = [sub[c].std() for c, _ in cols]
        x = np.arange(len(cols)) + (i - 1) * w
        ax.bar(x, means, w * 0.92, color=S["fill"][s], hatch=S["hatch"][s], edgecolor=S["edge"], lw=0.5,
               label=NAMES[s], yerr=stds, error_kw=dict(lw=0.7, capsize=2, capthick=0.7, ecolor=S["ink"]))
        for xi, m in zip(x, means):
            ax.text(xi, 0.505, f"{m:.3f}", ha="center", va="bottom", fontsize=6.3, rotation=90,
                    color=S["bar_text"][s])
    ax.set_xticks(range(len(cols)), [c[1] for c in cols])
    ax.set_ylim(0.5, 0.67)
    ax.set_ylabel("Test Macro-F1")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    grid(ax)
    save(fig, "fig_tier_f1.png")


# ---------------------------------------------------------------- Figure: does validation gain transfer to test?
def gain_transfer():
    runs = [(42, RQ1), (43, RQ1 / "seeds" / "phobert_text_pw_sqrt_ep12_seed43"),
            (44, RQ1 / "seeds" / "phobert_text_pw_sqrt_ep12_seed44")]
    tr = pd.read_csv(RQ1 / "tables" / "seed_gain_transfer.csv").set_index("seed")
    fig, ax = plt.subplots(figsize=(W * 0.62, 3.3))
    for (s, folder), mk, fc in zip(runs, ("o", "^", "s"), S["seed_fc"]):
        c = pd.read_csv(folder / "tables" / "per_label_comparison.csv")
        ax.scatter(c.val_gain_per_label_vs_fixed, c.test_gain_per_label_vs_fixed, marker=mk, s=18, facecolor=fc,
                   edgecolor=S["edge"], lw=0.6, label=f"seed {s} (ρ = {tr.loc[s, 'spearman_val_gain_vs_test_gain']:.2f})")
        if s == 42:
            for _, row in c.iterrows():
                if row.label in ("relief", "curiosity", "love", "grief", "remorse"):
                    off = (-24, 3) if row.label == "grief" else (3, 2)  # grief sits next to other markers
                    ax.annotate(row.label, (row.val_gain_per_label_vs_fixed, row.test_gain_per_label_vs_fixed),
                                xytext=off, textcoords="offset points", fontsize=7)
    lim = (-0.01, 0.13)
    ax.plot(lim, lim, color=S["diag"], ls=":", lw=0.8)
    ax.text(0.095, 0.1, "test gain = validation gain", fontsize=7, ha="right")
    ax.axhline(0, color=S["ink"], lw=0.6)
    ax.set_xlim(lim)
    ax.set_xlabel("F1 gain on validation (per-label vs 0.5)")
    ax.set_ylabel("F1 gain on test")
    ax.legend(loc="upper left", frameon=False, fontsize=7.3)
    grid(ax, "both")
    save(fig, "fig_gain_transfer.png")


# ---------------------------------------------------------------- Figure: decisions flipped per label
def flips():
    f = pd.read_csv(RQ1 / "error_analysis" / "decision_flips.csv")
    d = f[f.comparison == "per_label vs fixed"].sort_values("threshold_a").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(W, 4.6))
    y = np.arange(len(d))
    gain = d.fn_recovered + d.fp_removed
    loss = d.fn_introduced + d.fp_introduced
    ax.barh(y, gain, height=0.72, color=S["flip_gain"], edgecolor=S["edge"], lw=0.4,
            label="corrected (FN recovered + FP removed)")
    ax.barh(y, -loss, height=0.72, color=S["flip_loss"], edgecolor=S["edge"], lw=0.4,
            label="broken (FN introduced + FP introduced)")
    ax.scatter(gain - loss, y, marker="D", s=12, facecolor="white", edgecolor="black", lw=0.6, zorder=3,
               label="net change")
    ax.axvline(0, color=S["ink"], lw=0.7)
    ax.set_yticks(y, [f"{l} ({t:.3f})" for l, t in zip(d.label, d.threshold_a)], fontsize=7.5)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlabel("Test decisions changed when moving from τ = 0.5 to the per-label threshold (in brackets)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7.3)
    grid(ax, "x")
    save(fig, "fig_decision_flips.png")


# ---------------------------------------------------------------- Figure: under/over-prediction by cardinality
def cardinality():
    c = pd.read_csv(RQ1 / "error_analysis" / "cardinality_errors.csv")
    groups = ["1", "2", "3", ">=4"]
    n = c[c.strategy == "fixed"].set_index("cardinality").loc[groups].n_samples
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True)
    w = 0.26
    for i, s in enumerate(STRATEGIES):
        sub = c[c.strategy == s].set_index("cardinality").loc[groups]
        x = np.arange(4) + (i - 1) * w
        for ax, col in zip(axes, ("share_under", "share_over")):
            ax.bar(x, sub[col], w * 0.92, color=S["fill"][s], hatch=S["hatch"][s], edgecolor=S["edge"], lw=0.5,
                   label=NAMES[s])
    labels = [f"|Y|={g}\n(n={k})" for g, k in zip(["1", "2", "3"], n.iloc[:3])] + [f"|Y|≥4\n(n={n.iloc[3]})"]
    for ax, title in zip(axes, ("(a) Fewer labels predicted than gold", "(b) More labels predicted than gold")):
        ax.set_xticks(range(4), labels)
        ax.set_title(title)
        ax.set_ylim(0, 0.85)
        grid(ax)
    axes[0].set_ylabel("Share of test comments")
    axes[1].legend(loc="upper right", frameon=False)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_cardinality_errors.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=sorted(STYLES), default="bw")
    S = STYLES[ap.parse_args().style]
    OUT = ROOT / "results" / "figures" / S["out"]
    OUT.mkdir(exist_ok=True)
    for fn in (label_distribution, protocol, thresholds, tiers, gain_transfer, flips, cardinality):
        fn()
        print("ok", fn.__name__, "->", OUT.name)
