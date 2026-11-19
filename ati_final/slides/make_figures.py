"""Vẽ lại các figure cho slide giữa kì theo tông lavender của deck.

Chỉ đọc bảng/dự đoán đã có trong results/; không sửa results/figures/. Số liệu giữ nguyên,
chỉ đổi màu, cỡ chữ và cách trình bày để dễ đọc khi chiếu.
Chạy từ thư mục ati_final: python slides/make_figures.py
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
PREDS = ROOT / "results" / "preds"
OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)

INK, DEEP, PRIMARY, MID, LIGHT, CORAL, GREY = (
    "#2B2140", "#4A2C82", "#7C4DCC", "#A98BE6", "#D9C6F5", "#E5587F", "#9A93AD")
TIER_COLOR = {"head": DEEP, "mid": MID, "tail": LIGHT}
TIER_EN = {"head": "Frequent", "mid": "Mid", "tail": "Rare"}

have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(f for f in ("Calibri", "Segoe UI", "DejaVu Sans") if f in have)
plt.rcParams.update({"font.size": 13, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": GREY})


def vi(x, nd=2):
    return f"{x:.{nd}f}"


def clean(ax, left=False):
    for s in ("top", "right") + (() if left else ("left",)):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# 1. Tần suất 28 nhãn (train), tô theo nhóm head/mid/tail
d = pd.read_csv(TABLES / "label_distribution.csv", index_col=0).sort_values("train", ascending=False)
n_train = len(pd.read_csv(ROOT / "splits" / "train.csv"))
d["pct_train"] = d["train"] / n_train * 100  # tính lại từ số đếm để khỏi làm tròn hai lần
fig, ax = plt.subplots(figsize=(6.6, 4.1))
x = np.arange(len(d))
ax.bar(x, d["pct_train"], color=[TIER_COLOR[t] for t in d["tier"]], width=0.78)
ax.set_xticks([])
ax.set_yticks([0, 5, 10, 15])
ax.set_yticklabels(["0%", "5%", "10%", "15%"])
ax.yaxis.grid(True, color=LIGHT, linewidth=0.8)
ax.set_axisbelow(True)
clean(ax)
top, bot = d.index[0], d.index[-1]
ax.annotate(f"{top}\n{vi(d['pct_train'].iloc[0], 1)}%", (0, d["pct_train"].iloc[0]), xytext=(3.3, 15.6),
            fontsize=12, color=INK, va="center")
ax.annotate(f"{bot}\n{vi(d['pct_train'].iloc[-1], 1)}%", (len(d) - 1, d["pct_train"].iloc[-1]),
            xytext=(len(d) - 6.2, 8.5), fontsize=12, color=INK, va="center",
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
ax.set_ylim(0, 19)
for tier in ("head", "mid", "tail"):
    idx = np.flatnonzero(d["tier"].to_numpy() == tier)
    ax.plot([idx.min() - 0.4, idx.max() + 0.4], [-1.6, -1.6], color=TIER_COLOR[tier], lw=3,
            clip_on=False, solid_capstyle="round")
    ax.text(idx.mean(), -3.4, f"{TIER_EN[tier]} ({len(idx)})", ha="center", va="top", fontsize=12.5,
            color=INK)
save(fig, "fig_label_frequency.png")

# 2. F1 PhoBERT: 5 nhãn dễ nhất và 5 nhãn khó nhất
f = pd.read_csv(TABLES / "q12_f1_by_label.csv", index_col=0).sort_values("PhoBERT", ascending=False)
best, worst = f.head(5), f.tail(5)
rows = pd.concat([best, worst])
fig, ax = plt.subplots(figsize=(5.8, 4.1))
y = np.r_[np.arange(5), np.arange(5) + 6][::-1].astype(float)
ax.barh(y, rows["PhoBERT"], color=[DEEP] * 5 + [CORAL] * 5, height=0.7)
ax.set_yticks(y)
ax.set_yticklabels(rows.index, fontsize=13)
for yi, v in zip(y, rows["PhoBERT"]):
    ax.text(v + 0.012, yi, vi(v), va="center", fontsize=13, color=INK)
ax.text(0, 5.15 - 0.05, "· · ·", ha="left", va="center", color=GREY, fontsize=12)
ax.set_xlim(0, 0.95)
ax.set_xticks([])
clean(ax, left=True)
ax.spines["left"].set_color(LIGHT)
save(fig, "fig_f1_easy_hard.png")

# 3. PhoBERT hơn BiLSTM bao nhiêu (Micro-F1) theo kiểu câu, kèm khoảng tin cậy 95% (Câu 4)
q = pd.read_csv(TABLES / "q4_subset_results.csv")
names = {"Tất cả": "All comments", "Câu ngắn (<= 5 token)": "Short (≤ 5 words)",
         "Câu dài (>= 30 token)": "Long (≥ 30 words)", "Có emoji": "With emoji",
         "Nhiều emoji (>= 2)": "Many emoji (≥ 2)", "Có teencode": "With teencode",
         "Chữ kéo dài (caooo)": "Stretched letters", "Dấu câu lặp (..., !!!)": "Repeated punctuation",
         "Văn bản thường": "Plain text"}
q = q.assign(label=q["nhóm"].map(names) + " (" + q["số câu"].astype(str) + ")", delta=q["PhoBERT - BiLSTM"])
ci = q["CI95 PhoBERT-BiLSTM"].map(lambda s: [float(v) for v in re.findall(r"[+-]?\d+\.\d+", s)])
q["lo"], q["hi"] = ci.str[0], ci.str[1]
q["diff_plain"] = q["CI95 trừ nhóm thường (* = khác 0)"].str.contains(r"\*")
plain = float(q.loc[q["nhóm"] == "Văn bản thường", "delta"].iloc[0])
q = q.sort_values("delta")
fig, ax = plt.subplots(figsize=(5.6, 3.05))
y = np.arange(len(q))
sig = q["lo"] > 0
ax.barh(y, q["delta"], color=np.where(sig, PRIMARY, LIGHT), height=0.62)
ax.errorbar(q["delta"], y, xerr=[q["delta"] - q["lo"], q["hi"] - q["delta"]], fmt="none",
            ecolor=INK, elinewidth=1.1, capsize=3)
ax.axvline(0, color=GREY, lw=1)
ax.axvline(plain, color=CORAL, lw=1.4, ls=(0, (4, 3)))
ax.text(plain, len(q) - 0.35, f"plain text +{plain:.2f}", color=CORAL, fontsize=11, ha="center", va="bottom")
ax.set_yticks(y)
ax.set_yticklabels([l + (" *" if f else "") for l, f in zip(q["label"], q["diff_plain"])], fontsize=11.5)
for yi, v in zip(y, q["delta"]):
    ax.text(0.222, yi, f"{v:+.2f}", va="center", ha="right", fontsize=12, fontweight="bold", color=DEEP)
ax.set_xlim(-0.05, 0.225)
ax.set_ylim(-0.6, len(q) + 0.25)
ax.set_xticks([])
clean(ax, left=True)
ax.spines["left"].set_visible(False)
save(fig, "fig_phobert_vs_bilstm.png")

# 4. Macro-F1 theo ngưỡng chung (val) cho 3 mô hình
grid = np.round(np.linspace(0.05, 0.90, 35), 3)
runs = {"TF-IDF + LR": ("tfidf_lr", GREY, 2.2), "BiLSTM": ("bilstm", MID, 2.2),
        "PhoBERT": ("phobert_text_pw_sqrt_ep12", DEEP, 3.4)}
tau_ref = pd.read_csv(TABLES / "threshold_global_all_models.csv").set_index("run")["tau*_val"]
fig, ax = plt.subplots(figsize=(5.6, 3.1))
for label, (run, color, lw) in runs.items():
    z = np.load(PREDS / f"{run}_val.npz")
    y_true, p = z["y"], z["p"]
    curve = np.array([f1_score(y_true, p >= t, average="macro", zero_division=0) for t in grid])
    best = grid[curve.argmax()]
    assert abs(best - tau_ref[run]) < 1e-6, (run, best, tau_ref[run])
    ax.plot(grid, curve, color=color, lw=lw, label=label, solid_capstyle="round")
    ax.scatter([best], [curve.max()], s=70, color=color, zorder=5, edgecolor="white", linewidth=1.5)
    if label == "PhoBERT":
        ax.annotate(f"τ* = {vi(best, 3)}", (best, curve.max()), xytext=(best + 0.02, curve.max() + 0.045),
                    fontsize=12.5, color=DEEP)
ax.axvline(0.5, color=CORAL, lw=1.6, ls=(0, (4, 3)))
ax.text(0.505, 0.255, "τ = 0.5", color=CORAL, fontsize=12.5, va="bottom")
ax.set_xlabel("one shared threshold τ (all labels)", fontsize=13)
ax.set_ylabel("Macro-F1 (val)", fontsize=13)
ax.set_xlim(0.05, 0.9)
ax.set_ylim(0.25, 0.68)
ax.yaxis.grid(True, color=LIGHT, linewidth=0.8)
ax.set_axisbelow(True)
clean(ax, left=True)
ax.spines["left"].set_visible(False)
ax.legend(frameon=False, loc="lower center", ncol=3, fontsize=12.5, bbox_to_anchor=(0.5, -0.42))
save(fig, "fig_threshold_sweep.png")
# 5. Câu 2: F1 của PhoBERT theo số mẫu dương trong train
f2 = pd.read_csv(TABLES / "q12_f1_by_label.csv", index_col=0)
rho = pd.read_csv(TABLES / "q12_spearman.csv").set_index("model").loc["PhoBERT"]
fig, ax = plt.subplots(figsize=(4.9, 2.4))
ax.scatter(f2["train_count"], f2["PhoBERT"], c=[TIER_COLOR[t] for t in f2["tier"]], s=46, edgecolor="white", linewidth=0.8, zorder=3)
for lab, dx, dy, ha in (("embarrassment", 25, 0.0, "left"), ("gratitude", 25, -0.03, "left"), ("amusement", -30, 0.045, "right"),
                        ("disapproval", 25, -0.012, "left"), ("desire", 25, -0.012, "left"), ("neutral", 25, 0.015, "left")):
    r = f2.loc[lab]
    ax.text(r["train_count"] + dx, r["PhoBERT"] + dy, lab, fontsize=10.5, color=INK, ha=ha, va="center")
ax.set_xlabel("positive training comments of the label", fontsize=11.5)
ax.set_ylabel("PhoBERT F1 (test)", fontsize=11.5)
ax.set_ylim(0.35, 0.9)
ax.yaxis.grid(True, color=LIGHT, linewidth=0.8)
ax.set_axisbelow(True)
clean(ax, left=True)
ax.spines["left"].set_visible(False)
ax.text(0.98, 0.04, f"Spearman ρ = {rho['spearman']:.2f}  (p = {rho['p_value']:.2f})", transform=ax.transAxes,
        ha="right", fontsize=11.5, color=DEEP, fontweight="bold")
save(fig, "fig_q2_scatter.png")

# 6. Câu 3: số cặp đồng xuất hiện (thật / PhoBERT dự đoán / kỳ vọng nếu độc lập) cho 6 cặp có phi train cao nhất
pq = pd.read_csv(TABLES / "q3_pair_quality.csv").head(6)
fig, ax = plt.subplots(figsize=(7.0, 3.5))
xx = np.arange(len(pq))
w = 0.27
ax.bar(xx - w, pq["thật cùng có"], w, color=DEEP, label="true pairs (test)")
ax.bar(xx, pq["PhoBERT: dự đoán"], w, color=CORAL, label="predicted by PhoBERT")
ax.bar(xx + w, pq["PhoBERT: kỳ vọng nếu độc lập"], w, color=LIGHT, label="expected if labels were independent")
for xi, a, b in zip(xx, pq["thật cùng có"], pq["PhoBERT: dự đoán"]):
    ax.text(xi - w, a + 3, str(int(a)), ha="center", fontsize=10.5)
    ax.text(xi, b + 3, str(int(b)), ha="center", fontsize=10.5)
ax.set_xticks(xx)
ax.set_xticklabels([c.replace(" - ", "\n+ ").replace("disappointment", "disappoint-\nment") for c in pq["cặp"]], fontsize=10.5)
ax.set_ylim(0, 330)
ax.set_yticks([])
clean(ax)
ax.spines["left"].set_visible(False)
ax.legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(0.0, 1.04))
save(fig, "fig_q3_pairs.png")

# 7. Câu 3: phi của nhãn thật (test) so với phi của nhãn PhoBERT dự đoán, 378 cặp nhãn
from scipy.stats import spearmanr
zt, zp = np.load(PREDS / "phobert_text_pw_sqrt_ep12_test.npz"), np.load(PREDS / "phobert_text_pw_sqrt_ep12_test.npz")
Yg, Yp = zt["y"].astype(float), (zp["p"] >= 0.5).astype(float)
iu = np.triu_indices(Yg.shape[1], 1)
with np.errstate(invalid="ignore", divide="ignore"):
    pg, pp = np.corrcoef(Yg.T)[iu], np.corrcoef(Yp.T)[iu]
ok = ~(np.isnan(pg) | np.isnan(pp))
r_s = spearmanr(pg[ok], pp[ok])[0]
fig, ax = plt.subplots(figsize=(4.6, 3.5))
ax.plot([-0.25, 0.5], [-0.25, 0.5], color=GREY, lw=1, ls=(0, (4, 3)))
ax.scatter(pg[ok], pp[ok], s=14, color=PRIMARY, alpha=0.55, edgecolor="none")
ax.set_xlabel("φ of true labels (test)", fontsize=11.5)
ax.set_ylabel("φ of PhoBERT predictions", fontsize=11.5)
ax.grid(True, color=LIGHT, linewidth=0.7)
ax.set_axisbelow(True)
clean(ax, left=True)
ax.spines["left"].set_visible(False)
ax.text(0.04, 0.95, f"Spearman ρ = {r_s:.2f}\n378 label pairs", transform=ax.transAxes, va="top", fontsize=11.5,
        color=DEEP, fontweight="bold")
save(fig, "fig_q3_phi_scatter.png")

# 8. Câu 5 (cho báo cáo): Macro-F1 theo τ, số nhãn dự đoán theo τ, độ hiệu chuẩn - 4 lần chạy, bản tiếng Anh
sw = pd.read_csv(TABLES / "q5_threshold_sweep.csv")
rel = pd.read_csv(TABLES / "q5_reliability_bins.csv")
gold_lpc = float(np.load(PREDS / "phobert_text_pw_sqrt_ep12_test.npz")["y"].sum(1).mean())
runs5 = {"TF-IDF+LR": ("TF-IDF + LR", GREY), "BiLSTM": ("BiLSTM", MID),
         "PhoBERT": ("PhoBERT (sqrt weight)", DEEP), "PhoBERT (BCE thường)": ("PhoBERT (plain BCE)", CORAL)}
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
for key, (lab, col) in runs5.items():
    v = sw[(sw.model == key) & (sw.split == "val")]
    axes[0].plot(v.tau, v.macro_f1, color=col, lw=2.2, label=lab)
    axes[1].plot(v.tau, v.labels_per_comment, color=col, lw=2.2)
    r = rel[rel.model == key]
    axes[2].plot(r.p_mean, r.freq_true, "o-", color=col, lw=2, ms=4)
for a in axes[:2]:
    a.axvline(0.5, color=CORAL, lw=1.2, ls=(0, (4, 3)), alpha=0.6)
axes[1].axhline(gold_lpc, color=INK, lw=1, ls=":")
axes[1].text(0.88, gold_lpc * 1.07, "gold 1.91", ha="right", fontsize=10.5)
axes[2].plot([0, 1], [0, 1], color=INK, lw=1, ls=":")
axes[0].set(xlabel="shared threshold τ", ylabel="Macro-F1 (validation)", title="Macro-F1", ylim=(0.25, 0.65))
axes[1].set(xlabel="shared threshold τ", ylabel="labels per comment", title="Predicted labels", yscale="log")
axes[2].set(xlabel="predicted probability", ylabel="observed frequency", title="Calibration (validation)")
axes[0].legend(frameon=False, fontsize=9.5, loc="lower center")
for a in axes:
    a.grid(True, color=LIGHT, linewidth=0.7)
    a.set_axisbelow(True)
    clean(a, left=True)
    a.spines["left"].set_visible(False)
    a.title.set_fontsize(12.5)
save(fig, "fig_q5_sensitivity.png")

print("OK", sorted(p.name for p in OUT.glob("*.png")))
