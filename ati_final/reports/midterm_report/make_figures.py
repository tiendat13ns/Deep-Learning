"""Figures for the midterm report (English, same lavender palette as the slides).

Only derived statistics are plotted; no comment text is stored. Run from ati_final:
    python reports/midterm_report/make_figures.py
The four figures that are shared with the slides (label frequency, PhoBERT vs BiLSTM,
threshold sweep) come from slides/make_figures.py.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.data import LABELS, load_all, label_matrix  # noqa: E402

TABLES = ROOT / "results" / "tables"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

INK, DEEP, PRIMARY, MID, LIGHT, CORAL, GREY = (
    "#2B2140", "#4A2C82", "#7C4DCC", "#A98BE6", "#D9C6F5", "#E5587F", "#9A93AD")
have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(f for f in ("Calibri", "Segoe UI", "DejaVu Sans") if f in have)
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": GREY})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.savefig(OUT / name, dpi=250, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


data = load_all()
Y = {s: label_matrix(data[s]).astype(np.int64) for s in data}

# 1. Label cardinality by split (share of comments)
fig, ax = plt.subplots(figsize=(6.0, 2.9))
groups = ["1", "2", "3", "≥ 4"]
width = 0.26
for k, (split, color) in enumerate(zip(("train", "val", "test"), (DEEP, PRIMARY, MID))):
    card = Y[split].sum(1)
    share = [(card == 1).mean(), (card == 2).mean(), (card == 3).mean(), (card >= 4).mean()]
    bars = ax.bar(np.arange(4) + (k - 1) * width, np.array(share) * 100, width, color=color, label=split)
    for b, v in zip(bars, share):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.8, f"{v * 100:.1f}", ha="center", fontsize=8)
ax.set_xticks(range(4))
ax.set_xticklabels(groups)
ax.set_xlabel("Number of labels per comment |Y|")
ax.set_ylabel("% of comments")
ax.set_ylim(0, 56)
ax.yaxis.grid(True, color=LIGHT, linewidth=0.7)
ax.set_axisbelow(True)
clean(ax)
ax.legend(frameon=False, ncol=3, loc="upper right")
save(fig, "fig_cardinality.png")

# 2. Phi correlation matrix (train labels only), clustered for readability
phi = np.corrcoef(Y["train"].T.astype(float))
dist = 1 - phi
np.fill_diagonal(dist, 0)
order = leaves_list(linkage(squareform(dist, checks=False), "average"))
shown = phi[np.ix_(order, order)].copy()
np.fill_diagonal(shown, np.nan)
cmap = LinearSegmentedColormap.from_list("lav", [CORAL, "#FFFFFF", DEEP])
fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(shown, cmap=cmap, vmin=-0.45, vmax=0.45)
names = [LABELS[i] for i in order]
ax.set_xticks(range(28))
ax.set_yticks(range(28))
ax.set_xticklabels(names, rotation=90, fontsize=7.5)
ax.set_yticklabels(names, fontsize=7.5)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("phi coefficient (train)", fontsize=9)
save(fig, "fig_phi_heatmap.png")

# 3. Text length (train): whitespace tokens and characters
texts = data["train"].text.astype(str)
n_tok, n_chr = texts.str.split().str.len(), texts.str.len()
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
for ax, s, title in ((axes[0], n_tok, "Whitespace-delimited tokens"), (axes[1], n_chr, "Characters")):
    ax.hist(s.clip(upper=s.quantile(0.995)), bins=45, color=PRIMARY, edgecolor="white", linewidth=0.3)
    med = s.median()
    ax.axvline(med, color=CORAL, ls="--", lw=1.4)
    ax.text(med * 1.08, ax.get_ylim()[1] * 0.9, f"median {med:.0f}", color=CORAL, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("per comment (train, clipped at the 99.5th percentile)", fontsize=8)
    clean(ax)
axes[0].set_ylabel("comments")
fig.tight_layout()
save(fig, "fig_text_length.png")

# 4. Per-label F1 (PhoBERT, test) against training frequency
f = pd.read_csv(TABLES / "q12_f1_by_label.csv", index_col=0)
sp = pd.read_csv(TABLES / "q12_spearman.csv").set_index("model").loc["PhoBERT"]
colors = {"head": DEEP, "mid": MID, "tail": CORAL}
fig, ax = plt.subplots(figsize=(5.3, 3.4))
for tier, label in (("head", "frequent"), ("mid", "mid"), ("tail", "rare")):
    d = f[f.tier == tier]
    ax.scatter(d.train_count, d.PhoBERT, s=42, color=colors[tier], label=label, edgecolor="white", linewidth=0.8)
offsets = {"gratitude": (-6, -2, "right"), "embarrassment": (7, 2, "left"), "amusement": (-6, 6, "right"),
           "disapproval": (7, -3, "left"), "neutral": (7, 3, "left"), "desire": (7, -4, "left")}
for name, (dx, dy, ha) in offsets.items():
    r = f.loc[name]
    ax.annotate(name, (r.train_count, r.PhoBERT), xytext=(dx, dy), textcoords="offset points", fontsize=8, ha=ha)
ax.set_xlabel("positive examples in train")
ax.set_ylabel("test F1 (PhoBERT)")
ax.text(0.97, 0.04, f"Spearman ρ = {sp.spearman:.2f} (p = {sp.p_value:.2f})", transform=ax.transAxes,
        fontsize=9, ha="right")
ax.yaxis.grid(True, color=LIGHT, linewidth=0.7)
ax.set_axisbelow(True)
clean(ax)
ax.legend(frameon=False, title="train frequency", loc="center right", fontsize=8, title_fontsize=8)
save(fig, "fig_f1_vs_frequency.png")

# 5. Predicted vs true labels per comment, by gold cardinality (PhoBERT, test, tau = 0.5)
e = pd.read_csv(TABLES / "error_by_cardinality_phobert.csv")
yt = Y["test"]
card = yt.sum(1)
true_mean = [1.0, 2.0, 3.0, card[card >= 4].mean()]
pred_mean = e["nhãn_dự_đoán"].to_numpy()
fig, ax = plt.subplots(figsize=(5.3, 2.9))
x = np.arange(4)
b1 = ax.bar(x - 0.18, true_mean, 0.36, color=LIGHT, label="gold labels")
b2 = ax.bar(x + 0.18, pred_mean, 0.36, color=DEEP, label="predicted labels")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.06, f"{b.get_height():.2f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(["1", "2", "3", "≥ 4"])
ax.set_xlabel("Number of gold labels |Y| of the comment")
ax.set_ylabel("labels per comment")
ax.set_ylim(0, 4.9)
clean(ax)
ax.legend(frameon=False, loc="upper left")
save(fig, "fig_pred_vs_true.png")

print("OK", sorted(p.name for p in OUT.glob("*.png")))
