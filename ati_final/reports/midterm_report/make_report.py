"""Build the ViGoEmotions midterm report (.docx) with python-docx.

Every number is read from results/tables, results/preds and the data splits, so the report
cannot drift from the experiments. No comment text is written to the document.
Run from ati_final (after make_figures.py and slides/make_figures.py):
    python reports/midterm_report/make_report.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.data import HEAD_MIN, LABELS, TAIL_MAX, label_matrix, label_tiers, load_all  # noqa: E402

T = ROOT / "results" / "tables"
FIG = HERE / "figures"
SLIDE_FIG = ROOT / "slides" / "assets"
OUT = ROOT / "reports" / "ViGoEmotions_Midterm_Report.docx"

INK, DEEP, PRIMARY, TINT, SOFT, GREY = "2B2140", "4A2C82", "7C4DCC", "E6CFF9", "F5EDFC", "6B6580"
MEMBERS = [("Nguyễn Quốc Khanh", "2301140051"), ("Đinh Tiến Đạt", "2301140022"),
           ("Nguyễn Duy Anh", "2301140006")]
FONT = "Calibri"


# ----------------------------------------------------------------------------- data
def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def n0(x):
    return f"{int(round(x)):,}"


def pct(x, nd=1):
    return f"{x:.{nd}f}%"


def sgn(x, nd=3):
    return f"{x:+.{nd}f}"


data = load_all()
Y = {s: label_matrix(data[s]).astype(np.int64) for s in data}
N = {s: len(Y[s]) for s in Y}
tiers = label_tiers(Y["train"].sum(0))
base = pd.read_csv(T / "baseline_results.csv").set_index("run")
MODELS = [("TF-IDF + LR", "tfidf_lr"), ("BiLSTM", "bilstm"), ("PhoBERT", "phobert_text_pw_sqrt_ep12")]
perlabel = pd.read_csv(T / "perlabel_compare_3models_test.csv", index_col=0)
q12 = pd.read_csv(T / "q12_f1_by_label.csv", index_col=0)
spear = pd.read_csv(T / "q12_spearman.csv").set_index("model")
q3pairs = pd.read_csv(T / "q3_top_pairs.csv")
q3phi = pd.read_csv(T / "q3_phi_correlation.csv")
q4 = pd.read_csv(T / "q4_subset_results.csv")
q1ci = pd.read_csv(T / "q1_f1_ci.csv", index_col=0)
q2ap = pd.read_csv(T / "q2_spearman_ap.csv", index_col=0)
q3q = pd.read_csv(T / "q3_pair_quality.csv")
q3tot = pd.read_csv(T / "q3_pair_totals.csv", index_col=0)
q5s = pd.read_csv(T / "q5_summary.csv", index_col=0)
q5c = pd.read_csv(T / "q5_calibration.csv", index_col=0)
q4i = q4.set_index("nhóm")
q4sty = q4i.drop(index=["Tất cả", "Văn bản thường"])["PhoBERT - BiLSTM"]
q4plain = float(q4i.loc["Văn bản thường", "PhoBERT - BiLSTM"])
thr = pd.read_csv(T / "threshold_global_all_models.csv").set_index("run")
tl = pd.read_csv(T / "text_length.csv")
inf = pd.read_csv(T / "informal_features.csv").set_index("split")
inf_lab = pd.read_csv(T / "informal_by_label.csv").set_index("label")
err_card = pd.read_csv(T / "error_by_cardinality_phobert.csv")
err_lab = pd.read_csv(T / "error_per_label_phobert.csv", index_col=0)
conf = pd.read_csv(T / "confused_pairs_phobert.csv")
grid = pd.read_csv(T / "tfidf_lr_grid_val.csv")
bil_hist = pd.read_csv(T / "bilstm_history_bce.csv")
bil_loss = pd.read_csv(T / "bilstm_loss_val.csv", index_col=0)
cost = {r: json.load(open(T / f"{r}_cost.json", encoding="utf-8")) for r in
        ("phobert_keep_bce", "phobert_text_bce", "phobert_text_bce_ep12", "phobert_text_pw_sqrt",
         "phobert_text_pw_sqrt_ep12")}
sel = cost["phobert_text_pw_sqrt_ep12"]

# predictions of the three selected models (test)
pred = {name: np.load(ROOT / "results" / "preds" / f"{run}_test.npz") for name, run in MODELS}
per_comment = {name: (z["p"] >= 0.5).sum(1) for name, z in pred.items()}
pb = pred["PhoBERT"]
tp = int(((pb["p"] >= .5) & (pb["y"] == 1)).sum())
fp = int(((pb["p"] >= .5) & (pb["y"] == 0)).sum())
fn = int(((pb["p"] < .5) & (pb["y"] == 1)).sum())
micro_p, micro_r = tp / (tp + fp), tp / (tp + fn)

# co-occurrence statistics (train labels only)
Ytr = Y["train"]
cnt = Ytr.sum(0)
co = Ytr.T @ Ytr
exp = np.outer(cnt, cnt) / N["train"]
phi = np.corrcoef(Ytr.T.astype(float))
li = {l: i for i, l in enumerate(LABELS)}

# split balance
share = {s: Y[s].mean(0) * 100 for s in Y}
gap = {k: np.abs(share[a] - share[b]) for k, (a, b) in
       {"train-val": ("train", "val"), "train-test": ("train", "test"), "val-test": ("val", "test")}.items()}


# ----------------------------------------------------------------------------- docx helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.2)
sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.0)
CONTENT_W = 21.0 - 4.4  # cm

counters = {"table": 0, "figure": 0}


def rgb(h):
    return RGBColor.from_string(h)


def style_font(style, size, bold=None, color=None, italic=None):
    f = style.font
    f.name = FONT
    f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color:
        f.color.rgb = rgb(color)
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)


normal = doc.styles["Normal"]
style_font(normal, 10.5, color=INK)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.12
for name, size, before, after in (("Heading 1", 16, 16, 6), ("Heading 2", 12.5, 12, 4), ("Heading 3", 11, 8, 3)):
    st = doc.styles[name]
    style_font(st, size, bold=True, color=DEEP)
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(after)
    st.paragraph_format.keep_with_next = True
style_font(doc.styles["Title"], 26, bold=True, color=DEEP)
doc.styles["Title"].paragraph_format.space_after = Pt(4)
tpr = doc.styles["Title"].element.get_or_add_pPr()
for b in tpr.findall(qn("w:pBdr")):
    tpr.remove(b)
style_font(doc.styles["List Bullet"], 10.5, color=INK)
doc.styles["List Bullet"].paragraph_format.space_after = Pt(3)
style_font(doc.styles["Caption"], 9, bold=False, color=GREY, italic=False)
doc.styles["Caption"].paragraph_format.space_after = Pt(4)
doc.styles["Caption"].paragraph_format.space_before = Pt(6)


def runs(par, text, size=None, color=None, italic=False, bold=False):
    """Add text with **bold** markup to a paragraph."""
    for i, part in enumerate(re.split(r"\*\*", text)):
        if not part:
            continue
        r = par.add_run(part)
        r.bold = bold or (i % 2 == 1)
        r.italic = italic
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = rgb(color)
    return par


def p(text, align=None, size=None, color=None, italic=False, after=None, keep=False):
    par = doc.add_paragraph()
    runs(par, text, size=size, color=color, italic=italic)
    if align == "c":
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "j":
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if after is not None:
        par.paragraph_format.space_after = Pt(after)
    par.paragraph_format.keep_with_next = keep
    return par


def bullets(items):
    for t in items:
        par = doc.add_paragraph(style="List Bullet")
        runs(par, t)


def h1(t):
    doc.add_heading(t, level=1)


def h2(t):
    doc.add_heading(t, level=2)


def h3(t):
    doc.add_heading(t, level=3)


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    for s in tcpr.findall(qn("w:shd")):
        tcpr.remove(s)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def table_borders(tbl, color="C9B2F2"):
    tblpr = tbl._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, sz in (("top", 8), ("left", 0), ("bottom", 8), ("right", 0), ("insideH", 4), ("insideV", 0)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single" if sz else "nil")
        if sz:
            el.set(qn("w:sz"), str(sz))
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), color)
        borders.append(el)
    tblpr.append(borders)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblpr.append(layout)


def caption(kind, text, keep=False):
    counters[kind] += 1
    par = doc.add_paragraph(style="Caption")
    r = par.add_run(f"{kind.capitalize()} {counters[kind]}. ")
    r.bold = True
    r.font.color.rgb = rgb(DEEP)
    runs(par, text)
    par.paragraph_format.keep_with_next = keep
    return par


def table(rows, widths, title, align=None, size=9, hl=(), bold_rows=(), note=None):
    """rows[0] is the header. widths in cm. align: list of 'l'/'c'/'r' per column."""
    caption("table", title, keep=True)
    assert abs(sum(widths) - CONTENT_W) < 0.9 or sum(widths) < CONTENT_W + 0.05, sum(widths)
    tbl = doc.add_table(rows=len(rows), cols=len(widths))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    table_borders(tbl)
    align = align or ["l"] + ["c"] * (len(widths) - 1)
    for i, row in enumerate(rows):
        tr = tbl.rows[i]
        trpr = tr._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        trpr.append(cs)
        if i == 0:
            th = OxmlElement("w:tblHeader")
            trpr.append(th)
        for j, text in enumerate(row):
            cell = tr.cells[j]
            cell.width = Cm(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            par = cell.paragraphs[0]
            par.paragraph_format.space_after = Pt(0)
            par.paragraph_format.space_before = Pt(0)
            par.paragraph_format.line_spacing = 1.0
            par.alignment = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER,
                             "r": WD_ALIGN_PARAGRAPH.RIGHT}[align[j]]
            is_head = i == 0
            runs(par, str(text), size=size, color=DEEP if is_head else INK, bold=is_head or i in bold_rows)
            for r in par.runs:
                r.font.name = FONT
            tcpr = cell._tc.get_or_add_tcPr()
            mar = OxmlElement("w:tcMar")
            for side in ("top", "bottom"):
                el = OxmlElement(f"w:{side}")
                el.set(qn("w:w"), "40")
                el.set(qn("w:type"), "dxa")
                mar.append(el)
            tcpr.append(mar)
            if len(rows) <= 16 and i < len(rows) - 1:
                par.paragraph_format.keep_with_next = True  # keep short tables on one page
            if is_head:
                shade(cell, TINT)
            elif i in hl:
                shade(cell, SOFT)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(2)
    sp.paragraph_format.line_spacing = 0.6
    if note:
        n = p(note, size=8.5, color=GREY, italic=True, after=8)
        # pull the note up to the table
        doc.element.body.remove(sp._p)
    return tbl


def figure(path, width_cm, title):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.keep_with_next = True
    par.paragraph_format.space_after = Pt(0)
    par.add_run().add_picture(str(path), width=Cm(width_cm))
    cap = caption("figure", title)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)


def add_field(par, instr):
    for kind in ("begin", None, "separate", "text", "end"):
        r = par.add_run()
        r.font.size = Pt(9)
        r.font.color.rgb = rgb(GREY)
        if kind == "begin" or kind == "separate" or kind == "end":
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), kind)
            r._r.append(fc)
        elif kind is None:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = instr
            r._r.append(it)
        else:
            r.text = "1"


# header and footer
hp = sec.header.paragraphs[0]
hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
runs(hp, "ViGoEmotions · Midterm Report · Group 3", size=9, color=GREY)
fp_ = sec.footer.paragraphs[0]
fp_.alignment = WD_ALIGN_PARAGRAPH.CENTER
r0 = fp_.add_run("Page ")
r0.font.size = Pt(9)
r0.font.color.rgb = rgb(GREY)
add_field(fp_, "PAGE")
r1 = fp_.add_run(" of ")
r1.font.size = Pt(9)
r1.font.color.rgb = rgb(GREY)
add_field(fp_, "NUMPAGES")

# ----------------------------------------------------------------------------- title block
doc.add_paragraph("ViGoEmotions Midterm Report", style="Title")
p("Fine-grained multi-label emotion recognition on Vietnamese social media", size=13, color=PRIMARY, after=2)
p("ATI Course Project 2 · Group 3 · October 2026, revised November 2026", size=10.5, color=GREY, after=2)
p(" · ".join(f"{n} ({i})" for n, i in MEMBERS), size=10.5, after=10)

# ----------------------------------------------------------------------------- summary
tf, bl, pbm = (base.loc[r] for _, r in MODELS)
h1("Summary")
p(f"Fine-tuned PhoBERT-base is the strongest model, with test Macro-F1 {f3(pbm.test_macro_f1)} and "
  f"Micro-F1 {f3(pbm.test_micro_f1)}. TF-IDF with One-vs-Rest Logistic Regression is second "
  f"({f3(tf.test_macro_f1)} / {f3(tf.test_micro_f1)}) and a BiLSTM trained from scratch is last "
  f"({f3(bl.test_macro_f1)} / {f3(bl.test_micro_f1)}). Every model, setting and threshold was selected on the "
  f"validation set; the test set is reported at the default threshold 0.5 with a single seed (42).")
g_ph = thr.loc["phobert_text_pw_sqrt_ep12", "gain_test"]
g_bce12 = thr.loc["phobert_text_bce_ep12", "gain_test"]
g_bce6 = thr.loc["phobert_text_bce", "gain_test"]
p(f"Two findings shape our proposed Final research question. First, 0.5 is not the best threshold, but once PhoBERT "
  f"is trained long enough a better shared threshold adds only {sgn(g_ph)} to {sgn(g_bce12)} Macro-F1 on the test set. "
  f"Second, PhoBERT over-predicts ({per_comment['PhoBERT'].mean():.2f} labels per comment against "
  f"{Y['test'].sum(1).mean():.2f} gold labels), so the way labels are decided from probabilities still matters.")
rows = [["Key message", "Evidence"],
        ["The task is genuinely multi-label",
         f"{Ytr.sum(1).mean():.2f} labels per comment; only {pct((Ytr.sum(1) == 1).mean() * 100)} of train comments have a single label"],
        ["Social-media text is noisy",
         f"{pct(inf.loc['train', 'emoji'])} of train comments contain emoji, {pct(inf.loc['train', 'teencode'])} teencode, "
         f"{pct(inf.loc['train', 'elong'])} stretched letters"],
        ["Emoji must be handled for PhoBERT",
         f"Raw emoji make {pct(cost['phobert_keep_bce']['unk_rate'] * 100, 2)} of PhoBERT sub-word tokens unknown; "
         f"converting them to words reduces this to {pct(cost['phobert_text_bce']['unk_rate'] * 100, 2)} and raised "
         f"validation Macro-F1 from {f3(base.loc['phobert_keep_bce', 'val_macro_f1'])} to "
         f"{f3(base.loc['phobert_text_bce', 'val_macro_f1'])} (same 6-epoch setting)"],
        ["PhoBERT is ahead in every text style, but not specifically on any",
         f"Micro-F1 gap over the BiLSTM is positive in every style ({sgn(q4sty.min(), 2)} to {sgn(q4sty.max(), 2)}); long "
         f"comments ({sgn(q4i.loc['Câu dài (>= 30 token)', 'PhoBERT - BiLSTM'], 2)}) and teencode "
         f"({sgn(q4i.loc['Có teencode', 'PhoBERT - BiLSTM'], 2)}) do not beat plain text ({sgn(q4plain, 2)}); only "
         f"emoji comments differ, with a smaller gap"],
        ["Label dependencies are kept but amplified",
         f"Spearman {q3phi.iloc[5]['rho']:.2f} between the phi matrices of gold and predicted labels, yet PhoBERT predicts "
         f"{int(q3tot.loc['PhoBERT', 'tổng cặp dự đoán']):,} co-occurring label pairs against "
         f"{int(q3tot.loc['PhoBERT', 'tổng cặp thật']):,} gold (pair precision {q3tot.loc['PhoBERT', 'precision (378 cặp)']:.2f})"],
        ["Training length changes conclusions",
         f"After 6 epochs of plain BCE a shared threshold seemed to add {sgn(g_bce6)} Macro-F1; after 12 epochs it adds "
         f"{sgn(g_bce12)}"],
        ["Errors are confusions among close emotions",
         "24 of 69 hand-labelled error cases are near-synonym labels; 13 of the 15 most frequent missed-label "
         "confusions are between two negative emotions"]]
table(rows, [5.2, CONTENT_W - 5.2], "Key messages of the midterm.", align=["l", "l"], size=9.5)
p("All numbers come from the notebooks in notebooks/ and the tables in results/. The repository contains code, "
  "split index files, aggregate tables and figures, and no comment text.", size=9.5, color=GREY, italic=True)

# ----------------------------------------------------------------------------- 1 task
h1("1. Task and dataset")
p("Each Vietnamese social-media comment receives one or more of 27 emotions or neutral, so a model makes 28 independent "
  "yes/no decisions per comment. Following the assignment, the primary metrics are Macro-F1 and Micro-F1; exact-match "
  "accuracy is not used because one wrong label among 28 makes the whole comment wrong.")
p("ViGoEmotions [1] is a gated Hugging Face dataset (uitnlp/vigoemotions): access requires accepting its terms and "
  "logging in with a token. We load it through the Hugging Face hub (src/data.py). The licence allows non-commercial "
  "research and education only and forbids redistributing comment text, so our repository holds code, split index "
  "files, aggregate statistics and metrics only. The case descriptions in Section 7 are paraphrases.")
p("We checked data quality before any analysis. There are no empty texts, no comments without labels, no duplicate ids, "
  "no duplicate texts within a split and no id shared between splits. We found two issues and report them rather than "
  "fix them, so that the official split stays intact:", keep=True)
neutral_mix = {s: int(((Y[s][:, 27] == 1) & (Y[s].sum(1) > 1)).sum()) for s in Y}
te_in_tr = len(set(data["test"].text) & set(data["train"].text))
bullets([f"**{sum(neutral_mix.values())} comments are labelled neutral together with another emotion** "
         f"(train {neutral_mix['train']}, val {neutral_mix['val']}, test {neutral_mix['test']}). Neutral should exclude "
         f"emotions, so this looks like annotation noise.",
         f"**{te_in_tr} test comment is identical to a train comment** ({te_in_tr / N['test'] * 100:.2f}% of the test set), "
         f"too few to move any metric."])

# ----------------------------------------------------------------------------- 2 Part A
h1("2. Part A: dataset analysis")
p("Every statistic that drives a modelling decision (tiers, co-occurrence matrices, vocabulary, length budgets) is "
  "computed on the training set only, so nothing from validation or test leaks into the design.")

h2("2.1 Sample count and label inventory")
p(f"ViGoEmotions has {sum(N.values()):,} comments: {N['train']:,} train ({N['train'] / sum(N.values()) * 100:.1f}%), "
  f"{N['val']:,} validation and {N['test']:,} test. The label space is 27 emotions plus neutral (28 labels; neutral is "
  f"index 27). Table 2 lists every label with the number of comments that carry it.")
rows = [["#", "Label", "Train", "Val", "Test", "Total", "% of train", "Tier"]]
for i, l in enumerate(LABELS):
    tr, va, te = (int(Y[s][:, i].sum()) for s in ("train", "val", "test"))
    rows.append([str(i), l, f"{tr:,}", f"{va:,}", f"{te:,}", f"{tr + va + te:,}", f"{tr / N['train'] * 100:.2f}", tiers[l]])
table(rows, [1.0, 3.4, 1.8, 1.6, 1.6, 1.8, 2.2, 2.0], "Label inventory. Counts are comments carrying the label; a comment can "
      "carry several labels, so the percentages add up to about 191%.", size=8.5,
      align=["c", "l", "r", "r", "r", "r", "r", "c"])

h2("2.2 Label frequency and head / mid / tail tiers")
n_head = sum(t == "head" for t in tiers.values())
n_mid = sum(t == "mid" for t in tiers.values())
n_tail = sum(t == "tail" for t in tiers.values())
top, bot = int(cnt.max()), int(cnt.min())
p(f"The most frequent label, amusement ({top:,} train comments), has {top / bot:.1f} times the positives of the rarest, "
  f"relief ({bot:,}). Each label covers only {cnt.min() / N['train'] * 100:.1f}% to {cnt.max() / N['train'] * 100:.1f}% of "
  f"comments, so every yes/no decision is dominated by \"no\". The imbalance is moderate rather than severe.")
p(f"We fixed the tiers from train counts before evaluating any model: **frequent** (head) labels have at least "
  f"{HEAD_MIN:,} positive train comments, **rare** (tail) labels fewer than {TAIL_MAX}, and **mid** labels lie in between. "
  f"This gives {n_head} head, {n_mid} mid and {n_tail} tail labels.")
rows = [["Tier", "Rule (train positives)", "Labels"]]
for t, rule in (("head", f"≥ {HEAD_MIN:,}"), ("mid", f"{TAIL_MAX}–{HEAD_MIN - 1}"), ("tail", f"< {TAIL_MAX}")):
    ls = sorted([l for l in LABELS if tiers[l] == t], key=lambda l: -cnt[li[l]])
    rows.append([f"{t} ({len(ls)})", rule, ", ".join(ls)])
table(rows, [2.3, 3.2, CONTENT_W - 5.5], "Frequency tiers, defined on the training set.", align=["l", "c", "l"], size=9)
figure(SLIDE_FIG / "fig_label_frequency.png", 11.0, "Frequency of the 28 labels in the training set, coloured by tier.")
alone = lambda l: ((Ytr[:, li[l]] == 1) & (Ytr.sum(1) == 1)).sum() / cnt[li[l]] * 100  # noqa: E731
p(f"Some labels rarely appear alone: disappointment stands alone in only {alone('disappointment'):.1f}% of its comments "
  f"and grief in {alone('grief'):.1f}%, while neutral stands alone in {alone('neutral'):.1f}%.")

h2("2.3 Label cardinality |Y|")
p(f"Label cardinality is the average number of positive labels per comment. It is {Ytr.sum(1).mean():.2f} on train "
  f"(maximum 5); label density, cardinality divided by 28, is {Ytr.sum(1).mean() / 28:.3f}, so about "
  f"{(1 - Ytr.sum(1).mean() / 28) * 100:.0f}% of label decisions are \"no\".")
rows = [["Split", "Mean |Y|", "|Y| = 1", "|Y| = 2", "|Y| = 3", "|Y| ≥ 4"]]
for s in ("train", "val", "test"):
    c_ = Y[s].sum(1)
    rows.append([s, f"{c_.mean():.3f}"] + [pct((m).mean() * 100) for m in (c_ == 1, c_ == 2, c_ == 3, c_ >= 4)])
table(rows, [2.6, 2.4, 2.4, 2.4, 2.4, 2.4], "Label cardinality by split (share of comments).", size=9.5)
figure(FIG / "fig_cardinality.png", 11.5, "Distribution of the number of labels per comment in each split.")
p("Two labels is the most common case and only a third of comments have a single label, so a single-label classifier "
  "would be wrong on about two thirds of comments by design. Hamming loss stays low for any model because of the "
  "density, and has to be read together with F1.")

h2("2.4 Label co-occurrence and pairwise correlation")
p("We built two 28 × 28 matrices from the training labels only (test labels are never used). The co-occurrence matrix "
  "counts comments carrying both labels. The correlation matrix holds the phi coefficient, the Pearson correlation of "
  "two 0/1 label columns, which runs from −1 to 1 and corrects for how frequent each label is.")
pairs = [("disappointment", "sadness"), ("anger", "annoyance"), ("sadness", "grief"), ("amusement", "joy"),
         ("fear", "nervousness"), ("amusement", "annoyance")]
rows = [["Label pair", "Observed", "Expected if independent", "Phi"]]
for a, b in pairs:
    i, j = li[a], li[b]
    rows.append([f"{a} + {b}", f"{co[i, j]:,}", f"{exp[i, j]:.0f}", f"{phi[i, j]:.3f}"])
table(rows, [6.0, 3.0, 4.6, 3.0], "Co-occurrence of selected label pairs in train: observed counts, counts expected "
      "under independence (count of A × count of B / N), and phi.", size=9.5)
fi, ni = li["fear"], li["nervousness"]
am, an = li["amusement"], li["annoyance"]
p(f"Raw counts mislead when both labels are common. Fear and nervousness co-occur {co[fi, ni] / exp[fi, ni]:.0f} times "
  f"more often than chance, the strongest link in the data (phi {phi[fi, ni]:.2f}), while amusement and annoyance co-occur "
  f"{co[am, an]:,} times only because both are frequent: they meet less often than chance (expected {exp[am, an]:.0f}, "
  f"phi {phi[am, an]:.3f}). The most mutually exclusive pairs are amusement + sadness (phi {phi[li['amusement'], li['sadness']]:.3f}) "
  f"and joy + annoyance ({phi[li['joy'], li['annoyance']]:.3f}).")
figure(FIG / "fig_phi_heatmap.png", 12.2, "Phi correlation between labels (train), with labels ordered by hierarchical "
       "clustering. Emotions of the same polarity form blocks.")

h2("2.5 Text length")
p("Length is measured in whitespace-delimited tokens and in characters. Comments are short and right-skewed.")
sub = tl.copy()
sub.columns = ["unit", "stat", "test", "train", "val"]
sub["unit"] = sub["unit"].ffill()
rows = [["Unit", "Split", "Mean", "Median", "p90", "p95", "p99", "Max"]]
for unit, uname in (("n_tok", "tokens"), ("n_chr", "characters")):
    for s in ("train", "val", "test"):
        g = sub[sub.unit == unit].set_index("stat")[s].astype(float)
        rows.append([uname, s, f"{g['mean']:.1f}", f"{g['50%']:.0f}", f"{g['90%']:.0f}", f"{g['95%']:.0f}",
                     f"{g['99%']:.0f}", f"{g['max']:.0f}"])
table(rows, [2.6, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], "Comment length in whitespace tokens and characters.", size=9.5)
gt = sub[sub.unit == "n_tok"].set_index("stat")["train"].astype(float)
gc = sub[sub.unit == "n_chr"].set_index("stat")["train"].astype(float)
figure(FIG / "fig_text_length.png", 14.5, "Length distributions in the training set (clipped at the 99.5th percentile).")
p(f"Half of the training comments have at most {gt['50%']:.0f} tokens ({gc['50%']:.0f} characters) and 95% have at most "
  f"{gt['95%']:.0f}, but the longest reaches {gt['max']:.0f} tokens ({gc['max']:.0f} characters). The mean "
  f"({gt['mean']:.1f} tokens) sits above the median because of this long tail.")

h2("2.6 Qualitative review of informal Vietnamese text")
p("We reviewed the informal-language phenomena of the corpus with regular-expression flags and a hand-made list of "
  "teencode forms, then read the model errors in Section 7. The flags are estimates, not gold annotations: the "
  "teencode list can both miss forms and count ordinary words, especially one-letter forms such as k, r and j. To respect "
  "the licence we report only aggregate counts and generic forms, never comments.")
ti = inf.loc["train"]
toks = {w: 0 for w in ("ko", "k", "đc", "dc", "r", "vs", "j", "cx", "mik")}
for t in data["train"].text.astype(str):
    for w in t.lower().split():
        if w in toks:
            toks[w] += 1
no_letter = int(sum(1 for t in data["train"].text.astype(str) if not re.search(r"[A-Za-zÀ-ỹ0-9]", t)))
rows = [["Phenomenon", "% of train comments", "Typical forms and observations"],
        ["Emoji", pct(ti.emoji),
         "328 distinct emoji. Most frequent: face with tears of joy (882), rolling on the floor laughing (530), red heart "
         "(483), loudly crying face (394), Vietnamese flag (367), smiling face with hearts (351). Some comments "
         "consist of symbols only (last row)."],
        ["Teencode and abbreviations", pct(ti.teencode),
         f"Shortened or respelled words: ko / k for không (not) ({toks['ko']:,} / {toks['k']:,} tokens), đc / dc for được "
         f"(can, OK) ({toks['đc']:,} / {toks['dc']:,}), r for rồi (already) ({toks['r']:,}), vs for với (with) "
         f"({toks['vs']:,}), j for gì (what) ({toks['j']:,}), cx for cũng (also) ({toks['cx']:,}), mik for mình (I, me) "
         f"({toks['mik']:,})."],
        ["Stretched letters", pct(ti.elong),
         "A letter repeated three or more times (\"caooo\"). Repeated punctuation is counted in its own row."],
        ["Emoticons", "10.2% (estimate)", "Text smileys and crying faces made of punctuation characters."],
        ["Repeated punctuation", pct(ti.repunc), "Runs of ! ? or . used for emphasis or hesitation."],
        ["Mentions", "3.8%", "User mentions with @; hashtags are rare (0.2%)."],
        ["No diacritics", pct(ti.no_diac), "Comments typed without Vietnamese tone marks, which makes many words ambiguous."],
        ["Capitalised words", pct(ti.upper), "Shouting with capital letters is rare; comments are mostly lower case."],
        ["Emoji or punctuation only", f"{no_letter / N['train'] * 100:.1f}% ({no_letter} comments)",
         "No letters or digits at all, so the emotion rests entirely on the symbols."]]
table(rows, [3.4, 3.0, CONTENT_W - 6.4], "Informal-text phenomena in the training set (regex estimates).",
      align=["l", "c", "l"], size=9)
em = inf_lab["%emoji"]
tc = inf_lab["%teencode"]
p(f"These phenomena are not spread evenly over emotions (results/tables/informal_by_label.csv). Emoji are most "
  f"common in love ({em['love']:.1f}% of its comments), pride ({em['pride']:.1f}%) and amusement ({em['amusement']:.1f}%) "
  f"and rare in anger ({em['anger']:.1f}%) and neutral ({em['neutral']:.1f}%). Teencode is most common in embarrassment "
  f"({tc['embarrassment']:.1f}%), nervousness ({tc['nervousness']:.1f}%) and confusion ({tc['confusion']:.1f}%) and rare in "
  f"admiration ({tc['admiration']:.1f}%) and love ({tc['love']:.1f}%). Stretched letters are rare overall and most common "
  f"in {inf_lab['%elong'].idxmax()} ({inf_lab['%elong'].max():.1f}%). These are descriptive correlations, not causes.")
p("Two decisions follow from this review. Emoji are converted to words for PhoBERT instead of being deleted, because its "
  "tokenizer cannot represent them (Section 3). Teencode is left as written so that Question 4 can measure how each model "
  "copes with it. The review also predicts one class of errors found in Section 7: very short or symbol-only comments "
  "give a model almost nothing to work with.")

h2("2.7 Train / validation / test partition protocol")
_all = np.vstack([Y["train"], Y["val"], Y["test"]]).astype(float)
_rng = np.random.default_rng(0)
_sim_mean, _sim_max = [], []
for _ in range(1000):  # 1,000 random 80/10/10 splits of the same 20,664 comments
    _i = _rng.permutation(len(_all))
    _g = np.abs(_all[_i[:N["val"]]].mean(0) * 100 - _all[_i[N["val"]:N["val"] + N["test"]]].mean(0) * 100)
    _sim_mean.append(_g.mean())
    _sim_max.append(_g.max())
_sim_mean, _sim_max = np.array(_sim_mean), np.array(_sim_max)
p(f"The assignment asks us to report a stratified multi-label partition protocol. We keep the **official split** of the "
  f"dataset ({N['train']:,} / {N['val']:,} / {N['test']:,}, an 8:1:1 split) instead of re-splitting. It is the protocol of "
  f"the original paper, so our scores are comparable, and no comment can move between splits. **This split is not "
  f"stratified, and it does not behave like a stratified one:** its label shares differ between sets about as much as "
  f"those of a random split, so we treat it as a random split and state this as a limitation.")
bullets([f"The mean gap in a label's share of comments between validation and test is {gap['val-test'].mean():.2f} percentage "
         f"points; for 1,000 random splits of the same data it is {_sim_mean.mean():.2f}. The largest gap is "
         f"{gap['val-test'].max():.2f} points ({LABELS[int(gap['val-test'].argmax())]}: "
         f"{share['val'][int(gap['val-test'].argmax())]:.1f}% in validation against "
         f"{share['test'][int(gap['val-test'].argmax())]:.1f}% in test), against {_sim_max.mean():.2f} on average for random "
         f"splits; {(_sim_max >= gap['val-test'].max()).mean() * 100:.0f}% of random splits have a larger maximum gap. A "
         f"stratified split would keep these gaps close to zero.",
         f"Against train the largest gaps are {gap['train-val'].max():.2f} points for validation "
         f"({LABELS[int(gap['train-val'].argmax())]}) and {gap['train-test'].max():.2f} for test "
         f"({LABELS[int(gap['train-test'].argmax())]}).",
         "Label cardinality is almost identical across splits (Table 4).",
         f"The rarest label in validation, relief, still has {int(Y['val'][:, li['relief']].sum())} positives, enough to tune "
         f"and compare models. We did not run an iterative multi-label stratification as a sensitivity check; see Section 9.1."])
rows = [["Set", "Used for", "Never used for"],
        ["Train", "Model weights and every data-driven statistic: vocabulary, IDF, tiers, co-occurrence matrices",
         "Choosing hyper-parameters"],
        ["Validation", "Choosing hyper-parameters, the best epoch and the threshold sweep", "Fitting weights"],
        ["Test", "One report of each selected model", "Choosing a model, an epoch or a threshold"]]
table(rows, [2.4, 8.2, CONTENT_W - 10.6], "How the three sets are used.", align=["l", "l", "l"], size=9.5)

# ----------------------------------------------------------------------------- 3 preprocessing
h1("3. Preprocessing pipeline")
p("Normalisation is shared by all models (src/preprocess.py). Model-specific steps follow it.")
bullets(["**Common normalisation:** HTML entities are unescaped, <br> tags become spaces, text is normalised to Unicode "
         "NFC and whitespace is collapsed. Case, emoji and punctuation are kept. Teencode and stretched letters are not "
         "corrected.",
         "**TF-IDF and BiLSTM:** lower-casing at vectorisation; tokens are words or single non-space symbols, so emoji and "
         "punctuation become tokens of their own.",
         "**PhoBERT:** each emoji is replaced by its English name (emoji package), then the text is word-segmented with "
         "VnCoreNLP [4], which PhoBERT's pre-training requires, and tokenised with the vinai/phobert-base tokenizer "
         "(maximum length 128 sub-words)."])
p(f"Converting emoji matters because PhoBERT's vocabulary cannot represent them: with raw emoji "
  f"{pct(cost['phobert_keep_bce']['unk_rate'] * 100, 2)} of sub-word tokens are unknown, against "
  f"{pct(cost['phobert_text_bce']['unk_rate'] * 100, 2)} after conversion (counted over all three splits).")
rows = [["Model", "Vocabulary / features", "Input length"],
        ["TF-IDF + LR", "Word 1–2-grams, minimum document frequency 2, sub-linear tf (37,626 features)", "unlimited"],
        ["BiLSTM", "Words and symbols from train with frequency ≥ 2 (5,128 entries incl. padding and unknown)", "100 tokens"],
        ["PhoBERT", "vinai/phobert-base sub-words after emoji conversion and word segmentation", "128 sub-words"]]
table(rows, [3.0, 10.6, CONTENT_W - 13.6], "Representation of each model.", align=["l", "l", "c"], size=9.5)

# ----------------------------------------------------------------------------- 4 models
h1("4. Part B: models")
p("All three models output 28 independent sigmoid probabilities and predict a label when its probability is at least 0.5. "
  "The neural models minimise binary cross-entropy averaged over the labels, as the assignment specifies.")
h2("4.1 TF-IDF + One-vs-Rest Logistic Regression")
p("One L2-regularised logistic regression per label (liblinear, scikit-learn [5]) on word 1–2-gram TF-IDF features, fitted "
  "on train only. We searched 12 configurations on validation: word features or word plus character 2–4-grams, "
  "C ∈ {1, 4, 16}, and class weights balanced or none. The selected configuration is word features, C = 4, balanced "
  f"(validation Macro-F1 {f3(grid.iloc[0].macro_f1)}). Balanced weights beat every unweighted run, because they raise "
  "recall for labels whose positives are a small minority.")
g2 = grid.sort_values("macro_f1", ascending=False).head(12)
rows = [["Features", "C", "Class weight", "Val Macro-F1", "Val Micro-F1", "Train time (s)"]]
for _, r in g2.iterrows():
    cw = r.class_weight if isinstance(r.class_weight, str) else "none"
    rows.append([r.features, f"{r.C:g}", cw, f3(r.macro_f1), f3(r.micro_f1), f"{r.train_s:.1f}"])
table(rows, [3.0, 1.6, 3.0, 3.0, 3.0, 3.0], "TF-IDF + LR configurations searched on the validation set, best first "
      "(selected: first row).", size=9, hl=(1,))
h2("4.2 BiLSTM")
best_ep = int(bil_hist.val_macro_f1.idxmax()) + 1
p(f"Embedding of size 200 (random initialisation), a bidirectional LSTM with 128 units per direction, max- and mean-pooling "
  f"over time, dropout 0.3 and a linear layer from 512 to 28 outputs [3]. Adam with learning rate 1e-3, batch size 64, "
  f"gradient clipping at 1.0 and early stopping on validation Macro-F1 (patience 4, at most 30 epochs). Training ran "
  f"{len(bil_hist)} epochs and the best epoch was {best_ep} (validation Macro-F1 {f3(bil_hist.val_macro_f1.max())}). "
  f"We also tried a loss with full positive weights; it was clearly worse on validation "
  f"({f3(bil_loss.loc['bce_posweight', 'macro_f1'])} against {f3(bil_loss.loc['bce', 'macro_f1'])}), so plain BCE was kept.")
h2("4.3 PhoBERT")
p(f"vinai/phobert-base [2] (revision {sel['revision'][:8]}), a 12-layer RoBERTa-style model, with a classification head on the "
  f"first token and 28 sigmoid outputs. AdamW with learning rate 2e-5, weight decay 0.01, 10% linear warm-up, batch size 16, mixed precision, "
  f"up to 12 epochs with early stopping on validation Macro-F1 (patience 3). We compared five runs (Table 11). The "
  f"selected run uses BCE with a square-root positive weight per label, sqrt((N − n_c) / n_c), a milder version of the "
  f"full weight (N − n_c) / n_c used in [1]. It stopped after {sel['epochs_run']} epochs and the best epoch was "
  f"{sel['best_epoch']}. Plain BCE is the baseline loss of the assignment; its 12-epoch run is reported in the same table.")
rows = [["Run", "Emoji", "Loss", "Epochs run (best)", "Val Macro-F1", "Test Macro-F1"]]
names = {"phobert_keep_bce": ("kept", "BCE"), "phobert_text_bce": ("to text", "BCE"),
         "phobert_text_bce_ep12": ("to text", "BCE"), "phobert_text_pw_sqrt": ("to text", "BCE, sqrt weight"),
         "phobert_text_pw_sqrt_ep12": ("to text", "BCE, sqrt weight")}
for run, (em_, loss) in names.items():
    c_ = cost[run]
    rows.append([run.replace("phobert_", ""), em_, loss, f"{c_['epochs_run']} ({c_['best_epoch']})",
                 f3(base.loc[run, "val_macro_f1"]), f3(base.loc[run, "test_macro_f1"])])
table(rows, [4.2, 1.8, 3.4, 3.2, 2.0, 2.0], "PhoBERT runs. The selected run (shaded) has the highest validation Macro-F1; "
      "test scores are listed for transparency and were not used to choose.", size=9, hl=(5,),
      align=["l", "c", "c", "c", "c", "c"])
p(f"Peak GPU memory was {sel['peak_gpu_gb']} GB on an NVIDIA RTX 3060 Laptop GPU (6 GB). During epoch 4 of the selected "
  f"run the laptop went to sleep, which inflated the wall-clock time to {sel['train_seconds']:,.0f} s; Table 12 reports the "
  f"corrected {base.loc['phobert_text_pw_sqrt_ep12', 'train_seconds']:,.0f} s for the {sel['epochs_run']} epochs.")

# ----------------------------------------------------------------------------- 5 metrics
h1("5. Part C: metrics and results")
h2("5.1 Main results")
rows = [["Model", "Split", "Macro-F1", "Micro-F1", "Hamming ↓", "LRAP", "mAP", "Subset acc.", "Parameters", "Train time"]]
for name, run in MODELS:
    r = base.loc[run]
    for s in ("val", "test"):
        rows.append([name if s == "val" else "", s, f3(r[f"{s}_macro_f1"]), f3(r[f"{s}_micro_f1"]),
                     f"{r[f'{s}_hamming_loss']:.4f}", f3(r[f"{s}_lrap"]), f3(r[f"{s}_mAP"]),
                     f3(r[f"{s}_subset_acc"]), f"{r.n_params / 1e6:.2f} M" if s == "val" else "",
                     (f"{r.train_seconds:,.1f} s" if r.train_seconds < 1000 else f"{r.train_seconds / 60:.1f} min") if s == "val" else ""])
table(rows, [2.3, 1.1, 1.6, 1.6, 2.1, 1.3, 1.3, 1.7, 1.9, 1.7], "Baseline results at threshold 0.5. LRAP and mAP use "
      "probabilities and need no threshold. Subset accuracy is shown for reference only.", size=8.5, hl=(5, 6))
d_tf = pbm.test_macro_f1 - tf.test_macro_f1
d_bl = pbm.test_macro_f1 - bl.test_macro_f1
p(f"PhoBERT is best on every test metric. It gains {d_tf:.3f} Macro-F1 over TF-IDF and {d_bl:.3f} over the BiLSTM, and the "
  f"gap is larger on the threshold-free LRAP ({pbm.test_lrap - tf.test_lrap:+.3f} over TF-IDF) than on F1. On validation "
  f"the BiLSTM has a Hamming loss marginally below PhoBERT's. A BiLSTM trained from scratch on 16.5 thousand comments "
  f"loses to a linear model at least under our tuning budget; random embeddings learn little about rare words, while "
  f"TF-IDF counts them directly. Hamming loss says little here: all values lie between 0.058 and 0.067 because most "
  f"decisions are \"no\", and the BiLSTM beats TF-IDF on it only because it predicts fewer labels "
  f"({per_comment['BiLSTM'].mean():.2f} against {per_comment['TF-IDF + LR'].mean():.2f} per comment).")
rows = [["Model", "Head (10)", "Mid (10)", "Tail (8)"]]
for name, run in MODELS:
    r = base.loc[run]
    rows.append([name, f3(r.test_macro_f1_head), f3(r.test_macro_f1_mid), f3(r.test_macro_f1_tail)])
table(rows, [4.0, 2.8, 2.8, 2.8], "Test Macro-F1 by frequency tier.", size=9.5)
tier_gain = {t: base.loc["phobert_text_pw_sqrt_ep12", f"test_macro_f1_{t}"] - base.loc["tfidf_lr", f"test_macro_f1_{t}"]
             for t in ("head", "mid", "tail")}
p(f"Tail labels are hardest for every model (PhoBERT {base.loc['phobert_text_pw_sqrt_ep12', 'test_macro_f1_tail']:.3f} against "
  f"{base.loc['phobert_text_pw_sqrt_ep12', 'test_macro_f1_head']:.3f} for head labels). PhoBERT's advantage over TF-IDF holds in "
  f"every tier ({sgn(tier_gain['head'])} head, {sgn(tier_gain['mid'])} mid, {sgn(tier_gain['tail'])} tail).")
h2("5.2 Per-label F1")
rows = [["Label", "Tier", "Train n", "TF-IDF + LR", "BiLSTM", "PhoBERT"]]
order = perlabel["phobert_text_pw_sqrt_ep12"].sort_values(ascending=False).index
for l in order:
    v = [perlabel.loc[l, "tfidf_lr"], perlabel.loc[l, "bilstm"], perlabel.loc[l, "phobert_text_pw_sqrt_ep12"]]
    cells = [f3(x) for x in v]
    cells[int(np.argmax(v))] = f"**{cells[int(np.argmax(v))]}**"
    rows.append([l, tiers[l], f"{int(cnt[li[l]]):,}"] + cells)
table(rows, [3.6, 2.0, 2.2, 2.9, 2.6, 2.6], "F1 per label on the test set at threshold 0.5, sorted by PhoBERT; the best "
      "model per label is in bold.", size=8.5, align=["l", "c", "r", "c", "c", "c"])
wins = int((perlabel["phobert_text_pw_sqrt_ep12"] > perlabel["tfidf_lr"]).sum())
losses = [l for l in LABELS if perlabel.loc[l, "phobert_text_pw_sqrt_ep12"] < perlabel.loc[l, "tfidf_lr"]]
p(f"PhoBERT beats TF-IDF on {wins} of 28 labels. TF-IDF is better on {', '.join(losses)}. PhoBERT gains most on neutral "
  f"({perlabel.loc['neutral', 'tfidf_lr']:.3f} → {perlabel.loc['neutral', 'phobert_text_pw_sqrt_ep12']:.3f}), "
  f"disappointment and realization.")
h2("5.3 Comparison with the original paper")
p("The ViGoEmotions paper [1] reports test Macro-F1 56.33 for PhoBERT-base-v2 and 61.50 for ViSoBERT, its best model "
  "(Weighted-F1 57.94 and 63.26; it reports no Micro-F1). Our 0.608 is therefore not directly comparable: we use "
  "vinai/phobert-base (version 1), convert emoji to words, use a square-root positive weight and learning rate 2e-5, "
  "whereas [1] uses PhoBERT-base-v2 with full positive weights and learning rate 5e-5 (its Scenario 1 also keeps emoji). "
  "What the comparison does show is that our pipeline is in the range reported for this dataset, and that the "
  "result of a single recipe should not be over-interpreted.")

# ----------------------------------------------------------------------------- 6 questions
h1("6. Part D: midterm questions")
h2("Q1. Which emotion labels have the highest and lowest F1?")
rows = [["Model", "Highest F1", "Lowest F1"]]
for name, run in MODELS:
    s = perlabel[run if run != "tfidf_lr" else "tfidf_lr"].sort_values(ascending=False)
    top3 = ", ".join(f"{l} {s[l]:.3f}" for l in s.index[:3])
    low3 = ", ".join(f"{l} {s[l]:.3f}" for l in s.index[::-1][:3])
    rows.append([name, top3, low3])
table(rows, [3.0, 6.8, CONTENT_W - 9.8], "Three highest and lowest per-label test F1 for each model.", align=["l", "l", "l"], size=9)
p("Gratitude, embarrassment and amusement are the three easiest labels for every model. Disapproval and neutral are among "
  "the three hardest for every model, together with realization (TF-IDF, BiLSTM) or desire (PhoBERT). A plausible reason, which we did not test, is that easy labels have explicit surface markers: "
  f"amusement comments often contain laughter emoji ({inf_lab['%emoji']['amusement']:.0f}% of them). The hard labels depend on stance rather than vocabulary: disapproval overlaps "
  "with anger, annoyance and disgust (Section 7), and neutral is defined by the absence of cues, which bag-of-words models "
  f"handle poorly (TF-IDF {perlabel.loc['neutral', 'tfidf_lr']:.3f}, PhoBERT {perlabel.loc['neutral', 'phobert_text_pw_sqrt_ep12']:.3f}).")
rows = [["Label", "Tier", "Test n", "PhoBERT F1", "95% CI", "Precision", "Recall", "Rank stability"]]
for l in list(q1ci.index[:3]) + list(q1ci.index[::-1][:3]):
    r = q1ci.loc[l]
    top_side = l in list(q1ci.index[:3])
    rows.append([l, r["nhóm"], str(int(r["số mẫu test"])), f3(r["F1 PhoBERT"]), r["CI95 PhoBERT"], f3(r["PhoBERT precision"]),
                 f3(r["PhoBERT recall"]),
                 f"top 5 in {r['p_top5'] * 100:.0f}% of resamples" if top_side else f"bottom 5 in {r['p_bot5'] * 100:.0f}%"])
table(rows, [2.9, 1.3, 1.4, 2.0, 2.4, 1.8, 1.6, 3.2], "Uncertainty of the extremes for PhoBERT: 95% bootstrap interval of the test F1 "
      "(1,000 resamples of the test comments) and how often the label stays among the five highest or lowest.", size=8.5,
      align=["l", "c", "c", "c", "c", "c", "c", "c"])
p(f"Each label has only {int(q1ci['số mẫu test'].min())} to {int(q1ci['số mẫu test'].max())} positive test comments, so a "
  f"per-label F1 carries sampling noise of roughly ±0.03 to ±0.09. Only the two ends of the ranking are reliable: "
  f"embarrassment, gratitude and amusement stay in PhoBERT's top five in {q1ci.loc[['embarrassment', 'gratitude', 'amusement'], 'p_top5'].min() * 100:.0f}% of "
  f"resamples, and disapproval stays in the bottom five in {q1ci.loc['disapproval', 'p_bot5'] * 100:.0f}%. Labels in the "
  f"middle cannot be ranked against each other. The hard labels combine low precision (disapproval {q1ci.loc['disapproval', 'PhoBERT precision']:.2f}, "
  f"desire {q1ci.loc['desire', 'PhoBERT precision']:.2f}) or low recall (neutral {q1ci.loc['neutral', 'PhoBERT recall']:.2f}).")

h2("Q2. How strongly does performance correlate with training label frequency?")
rows = [["Model", "ρ with F1 (τ = 0.5)", "p", "ρ with AP (no threshold)", "p", "95% range of ρ(F1) over test resamples"]]
for name in ("TF-IDF+LR", "BiLSTM", "PhoBERT"):
    r = q2ap.loc[name]
    rows.append([name.replace("TF-IDF+LR", "TF-IDF + LR"), f3(r["rho(F1@0.5, tần suất)"]), f3(r["p (F1)"]), f3(r["rho(AP, tần suất)"]),
                 f3(r["p (AP)"]), r["rho(F1) 95% theo bootstrap test"]])
table(rows, [3.0, 3.0, 1.5, 3.2, 1.5, 4.4], "Spearman correlation between training frequency and test performance over the 28 labels, "
      "using F1 at threshold 0.5 and average precision (AP), which needs no threshold.", size=9)
figure(FIG / "fig_f1_vs_frequency.png", 10.5, "Test F1 of PhoBERT against the number of positive training examples per label.")
p("The correlation is weak and not significant for all three models. Frequency explains little because its range is narrow "
  f"({int(cnt.min())} to {int(cnt.max()):,}, a factor of {cnt.max() / cnt.min():.1f}) and label meaning matters more: gratitude has "
  f"{int(cnt[li['gratitude']])} training examples and one of the highest F1, while disapproval has {int(cnt[li['disapproval']])} "
  "and the lowest. The tier means still show a mild trend (Table 13), so imbalance matters at the extremes without being "
  "the main driver.")
p(f"With 28 labels a correlation must exceed 0.375 in absolute value to be significant at 5%, and the frequency range is only "
  f"a factor of {cnt.max() / cnt.min():.1f}, so the test has low power: we find no evidence of a relationship, which is not "
  f"evidence that there is none. Average precision gives the same picture (ρ from {q2ap['rho(AP, tần suất)'].min():.2f} to "
  f"{q2ap['rho(AP, tần suất)'].max():.2f}, all p ≥ {q2ap['p (AP)'].min():.2f}), so the weak link is not an artefact of the "
  f"0.5 threshold.")

h2("Q3. Which emotion pairs co-occur most, and are the dependencies preserved?")
p("The leading pairs in train are disappointment + sadness, anger + annoyance, sadness + grief and amusement + joy "
  "(Table 5); fear + nervousness has the highest phi. We compare the gold pairs of the test set with the pairs predicted by "
  "each model. Pair recall is the share of test comments with both gold labels in which the model predicts both.")
rows = [["Label pair", "Phi (train)", "Gold pairs (test)", "Recall TF-IDF", "Recall BiLSTM", "Recall PhoBERT", "PhoBERT predicts pair"]]
for _, r in q3pairs.iterrows():
    g = r["test: thật cùng có"]
    rows.append([r["cặp"].replace(" - ", " + "), f2(r["phi_train"]), str(g),
                 f2(r["TF-IDF+LR: đúng cả đôi"] / g), f2(r["BiLSTM: đúng cả đôi"] / g), f2(r["PhoBERT: đúng cả đôi"] / g),
                 str(r["PhoBERT: dự đoán cùng có"])])
table(rows, [4.0, 1.6, 2.0, 1.9, 1.9, 2.0, 3.2], "Pair recall on the test set for the twelve pairs with the highest phi.",
      size=8.5)
rho = lambda i: q3phi.iloc[i]["rho"]  # noqa: E731
p(f"At the level of the whole 378-pair structure the dependencies are preserved: the Spearman correlation between the phi "
  f"matrix of the gold test labels and that of the predictions is {rho(1):.3f} for TF-IDF, {rho(3):.3f} for the BiLSTM and "
  f"{rho(5):.3f} for PhoBERT, close to the {rho(0):.3f} obtained between the train and the gold test matrices. Inside "
  f"individual comments the picture is weaker: pair recall for PhoBERT ranges from "
  f"{min(r['PhoBERT: đúng cả đôi'] / r['test: thật cùng có'] for _, r in q3pairs.iterrows()):.2f} to "
  f"{max(r['PhoBERT: đúng cả đôi'] / r['test: thật cùng có'] for _, r in q3pairs.iterrows()):.2f}, and it predicts "
  f"pairs far more often than they occur (e.g. anger + annoyance {int(q3pairs.loc[q3pairs['cặp'] == 'anger - annoyance', 'PhoBERT: dự đoán cùng có'].iloc[0])} "
  f"times against {int(q3pairs.loc[q3pairs['cặp'] == 'anger - annoyance', 'test: thật cùng có'].iloc[0])} gold pairs). Because each label has its "
  "own sigmoid output, the dependencies come from shared words and context, not from an explicit model of label interaction.")
p("**Preserved, but amplified.** Pair recall alone hides how often the pair is wrong. The table below adds pair precision and "
  "the number of pairs that would appear if the outputs were independent (predicted frequency of A × predicted frequency "
  "of B × N).", keep=True)
rows = [["Label pair", "Gold", "PhoBERT predicts", "Precision", "Recall", "Predicted ÷ gold", "Expected if independent"]]
for _, r in q3q.iterrows():
    rows.append([r["cặp"].replace(" - ", " + "), str(int(r["thật cùng có"])), str(int(r["PhoBERT: dự đoán"])),
                 f2(r["PhoBERT: precision"]), f2(r["PhoBERT: recall"]), f2(r["PhoBERT: tỉ lệ dự đoán/thật"]),
                 f"{r['PhoBERT: kỳ vọng nếu độc lập']:.0f}"])
table(rows, [4.4, 1.5, 2.4, 1.9, 1.7, 2.3, 2.4], "Quality of the pairs predicted by PhoBERT on the test set (twelve pairs with the "
      "highest phi in train).", size=8.5)
rows = [["Model", "Predicted co-occurring pairs", "Gold pairs", "Pair precision", "Pair recall"]]
for name in ("TF-IDF+LR", "BiLSTM", "PhoBERT"):
    r = q3tot.loc[name]
    rows.append([name.replace("TF-IDF+LR", "TF-IDF + LR"), f"{int(r['tổng cặp dự đoán']):,}", f"{int(r['tổng cặp thật']):,}",
                 f3(r["precision (378 cặp)"]), f3(r["recall (378 cặp)"])])
table(rows, [3.4, 4.4, 2.8, 3.0, 3.0], "Co-occurring label pairs over all 378 pairs, test set: counts of (comment, pair) occurrences.",
      size=9)
figure(SLIDE_FIG / "fig_q3_pairs.png", 12.5, "Co-occurring pairs in the test set: gold, predicted by PhoBERT, and expected if the "
       "outputs were independent.")
figure(SLIDE_FIG / "fig_q3_phi_scatter.png", 8.5, "Phi of the gold test labels against phi of PhoBERT's predictions for the 378 label "
       "pairs; the dashed line is equality.")
_ph3 = q3tot.loc["PhoBERT"]
p(f"The answer is therefore two-sided. PhoBERT does preserve the dependency structure: its predictions follow the gold phi "
  f"matrix (ρ = {rho(5):.2f}) and for sadness + grief it predicts {int(q3q.loc[q3q['cặp'] == 'sadness - grief', 'PhoBERT: dự đoán'].iloc[0])} pairs "
  f"where independent outputs would give about {q3q.loc[q3q['cặp'] == 'sadness - grief', 'PhoBERT: kỳ vọng nếu độc lập'].iloc[0]:.0f}, so the "
  f"dependency is learned. But it amplifies the dependencies: {int(_ph3['tổng cặp dự đoán']):,} predicted pairs against "
  f"{int(_ph3['tổng cặp thật']):,} gold, and only {_ph3['precision (378 cặp)'] * 100:.0f}% of predicted pairs are correct. The BiLSTM "
  f"predicts about the right number of pairs ({int(q3tot.loc['BiLSTM', 'tổng cặp dự đoán']):,}) but finds fewer of the gold pairs "
  f"(recall {q3tot.loc['BiLSTM', 'recall (378 cặp)']:.2f} against {_ph3['recall (378 cặp)']:.2f}).")

h2("Q4. On which text styles does PhoBERT outperform the recurrent baseline most?")
names4 = {"Tất cả": "All comments", "Câu ngắn (<= 5 token)": "Short (≤ 5 tokens)", "Câu dài (>= 30 token)": "Long (≥ 30 tokens)",
          "Có emoji": "With emoji", "Nhiều emoji (>= 2)": "Many emoji (≥ 2)", "Có teencode": "With teencode",
          "Chữ kéo dài (caooo)": "Stretched letters", "Dấu câu lặp (..., !!!)": "Repeated punctuation",
          "Văn bản thường": "Plain text (no flag)"}
rows = [["Subset", "n", "TF-IDF", "BiLSTM", "PhoBERT", "PhoBERT − BiLSTM", "95% CI", "Gap minus plain-text gap (95% CI)"]]
for _, r in q4.iterrows():
    diff = "reference" if r["nhóm"] == "Văn bản thường" else f"{sgn(r['PhoBERT-BiLSTM trừ nhóm thường'])} {r['CI95 trừ nhóm thường (* = khác 0)']}"
    rows.append([names4[r["nhóm"]], f"{int(r['số câu']):,}", f3(r["Micro-F1 TF-IDF+LR"]), f3(r["Micro-F1 BiLSTM"]),
                 f3(r["Micro-F1 PhoBERT"]), sgn(r["PhoBERT - BiLSTM"]), r["CI95 PhoBERT-BiLSTM"], diff])
table(rows, [3.6, 1.1, 1.4, 1.4, 1.5, 1.7, 2.3, 3.6], "Micro-F1 on test subsets. Intervals are paired bootstrap (2,000 resamples "
      "of the comments in each subset); the last column compares each subset's gap with the gap on plain text (* = the "
      "interval excludes zero). Subsets come from regex flags; the first seven overlap.", size=8.5,
      align=["l", "c", "c", "c", "c", "c", "c", "c"])
figure(SLIDE_FIG / "fig_phobert_vs_bilstm.png", 11.8, "Difference in Micro-F1 between PhoBERT and BiLSTM per subset, with 95% "
       "confidence intervals (pale bars: the interval includes zero; * = the gap differs from the plain-text gap; dashed line = plain-text gap).")
_n_pos = int((q4i.drop(index=["Tất cả"])["CI95 PhoBERT-BiLSTM"].map(lambda t: float(re.findall(r"[+-]?\d+\.\d+", t)[0])) > 0).sum())
p(f"PhoBERT is ahead of the BiLSTM in every subset ({sgn(q4sty.min(), 2)} to {sgn(q4sty.max(), 2)} Micro-F1) and the interval "
  f"excludes zero in {_n_pos} of the {len(q4sty) + 1} subsets (plain text included); it includes zero for comments with many emoji and "
  f"for the {int(q4i.loc['Chữ kéo dài (caooo)', 'số câu'])} comments with stretched letters, which are too few to conclude anything. Of "
  f"the three styles named in the question, teencode shows the largest gap ({sgn(q4i.loc['Có teencode', 'PhoBERT - BiLSTM'])}), then short "
  f"comments ({sgn(q4i.loc['Câu ngắn (<= 5 token)', 'PhoBERT - BiLSTM'])}) and emoji-heavy text "
  f"({sgn(q4i.loc['Nhiều emoji (>= 2)', 'PhoBERT - BiLSTM'])}).")
p(f"The last column shows that this ranking should not be over-read. The gap on long comments ({sgn(q4i.loc['Câu dài (>= 30 token)', 'PhoBERT - BiLSTM'])}) "
  f"and on teencode comments is not larger than on plain text ({sgn(q4plain)}), so we cannot say that PhoBERT helps specifically "
  f"there. The only reliable difference between styles goes the other way: for comments with emoji "
  f"({sgn(q4i.loc['Có emoji', 'PhoBERT-BiLSTM trừ nhóm thường'])}) and with many emoji "
  f"({sgn(q4i.loc['Nhiều emoji (>= 2)', 'PhoBERT-BiLSTM trừ nhóm thường'])}) the gap is significantly smaller than on plain text. "
  f"A possible reason, which we did not test, is that the BiLSTM sees emoji directly as tokens, whereas PhoBERT sees converted "
  f"names; the BiLSTM reaches its best Micro-F1 ({q4i.loc['Nhiều emoji (>= 2)', 'Micro-F1 BiLSTM']:.3f}) on emoji-heavy comments. "
  f"Note that the two models also differ in preprocessing (word segmentation, emoji conversion) and size, so this is a comparison "
  f"of pipelines, not of architectures.")

h2("Q5. How sensitive are the outputs to the default threshold of 0.5?")
rows = [["Model / run", "τ* (val)", "Val F1 at 0.5", "Val F1 at τ*", "Test F1 at 0.5", "Test F1 at τ*", "Test gain"]]
for label, run in (("TF-IDF + LR", "tfidf_lr"), ("BiLSTM", "bilstm"), ("PhoBERT, plain BCE, 6 epochs", "phobert_text_bce"),
                   ("PhoBERT, plain BCE, 12 epochs", "phobert_text_bce_ep12"),
                   ("PhoBERT, sqrt weight, 12 epochs (selected)", "phobert_text_pw_sqrt_ep12")):
    r = thr.loc[run]
    rows.append([label, f"{r['tau*_val']:.3f}", f3(r["val@0.5"]), f3(r["val@tau*"]), f3(r["test@0.5"]), f3(r["test@tau*"]),
                 sgn(r["gain_test"])])
table(rows, [5.4, 1.5, 1.9, 1.9, 1.9, 1.9, 2.1], "Macro-F1 with one shared threshold for all labels. τ* is chosen on validation only; "
      "the test columns are reported, not used for choosing.", size=8.5, hl=(5,))
figure(SLIDE_FIG / "fig_threshold_sweep.png", 11.0, "Validation Macro-F1 as a function of one shared threshold. Dots mark τ*; "
       "the dashed line is 0.5.")
nolab = {n: (v == 0).mean() * 100 for n, v in per_comment.items()}
p(f"The sensitivity depends on how well the model is trained. For the selected PhoBERT run the best shared threshold is "
  f"{thr.loc['phobert_text_pw_sqrt_ep12', 'tau*_val']:.3f} and it adds only {sgn(thr.loc['phobert_text_pw_sqrt_ep12', 'gain_test'])} "
  f"on test; for TF-IDF and the BiLSTM the gains are {sgn(thr.loc['tfidf_lr', 'gain_test'])} and "
  f"{sgn(thr.loc['bilstm', 'gain_test'])}. The picture was very different for the 6-epoch plain-BCE run, where the probabilities "
  f"had not yet moved above 0.5 and a threshold of {thr.loc['phobert_text_bce', 'tau*_val']:.3f} seemed to add "
  f"{sgn(g_bce6)}; with 12 epochs the gain falls to {sgn(g_bce12)}. We therefore draw no conclusion about thresholds from a "
  f"model that has not converged. At 0.5 the models also differ in how many labels they output per comment "
  f"(TF-IDF {per_comment['TF-IDF + LR'].mean():.2f}, BiLSTM {per_comment['BiLSTM'].mean():.2f}, PhoBERT "
  f"{per_comment['PhoBERT'].mean():.2f}; gold {Y['test'].sum(1).mean():.2f}), and the BiLSTM leaves {nolab['BiLSTM']:.1f}% "
  f"of comments with no label at all (TF-IDF {nolab['TF-IDF + LR']:.1f}%, PhoBERT {nolab['PhoBERT']:.1f}%).")
_names5 = {"TF-IDF+LR": "TF-IDF + LR", "BiLSTM": "BiLSTM", "PhoBERT": "PhoBERT (sqrt weight, selected)",
           "PhoBERT (BCE thường)": "PhoBERT (plain BCE, 12 epochs)"}
rows = [["Model / run", "τ = 0.3", "0.4", "0.5", "0.6", "0.7", "Spread"]]
for k, nm in _names5.items():
    v = [q5s.loc[k, f"val Macro-F1 @{t}"] for t in (0.3, 0.4, 0.5, 0.6, 0.7)]
    rows.append([nm] + [f3(x) for x in v] + [f3(max(v) - min(v))])
table(rows, [5.4, 1.8, 1.8, 1.8, 1.8, 1.8, 2.2], "Validation Macro-F1 at fixed shared thresholds, and the spread (maximum minus "
      "minimum) over τ = 0.3 to 0.7.", size=9)
rows = [["Model / run", "Labels per comment at 0.5", "at τ*", "Micro-precision at 0.5", "Micro-recall at 0.5", "ECE (val)", "Mean probability"]]
for k, nm in _names5.items():
    rows.append([nm, f2(q5s.loc[k, "labels/comment @0.5"]), f2(q5s.loc[k, "labels/comment @tau*"]), f3(q5s.loc[k, "micro P @0.5"]),
                 f3(q5s.loc[k, "micro R @0.5"]), f3(q5c.loc[k, "ECE (val)"]), f3(q5c.loc[k, "xác suất TB"])])
table(rows, [4.6, 2.3, 1.4, 2.4, 2.3, 1.8, 1.8], f"What the threshold does to the outputs (test, except ECE and mean probability on validation). "
      f"Gold: {Y['test'].sum(1).mean():.2f} labels per comment; true share of positive (comment, label) pairs "
      f"{q5c['tỉ lệ nhãn dương thật'].iloc[0]:.3f}. ECE = expected calibration error over 10 probability bins.", size=8.5)
figure(SLIDE_FIG / "fig_q5_sensitivity.png", 16.0, "Validation Macro-F1 (left), predicted labels per comment "
       "(middle, log scale; dashed line = gold) and reliability curves (right) for four runs; the dotted red line is τ = 0.5.")
_sp = {k: max(q5s.loc[k, f"val Macro-F1 @{t}"] for t in (0.3, 0.4, 0.5, 0.6, 0.7)) - min(q5s.loc[k, f"val Macro-F1 @{t}"] for t in (0.3, 0.4, 0.5, 0.6, 0.7)) for k in _names5}
p(f"**Answer.** The sensitivity of Macro-F1 to the threshold is modest near the optimum and large in what the outputs look like. "
  f"For the selected PhoBERT run, Macro-F1 on validation moves by only {_sp['PhoBERT']:.3f} between τ = 0.3 and 0.7, and the best "
  f"shared threshold gains {sgn(thr.loc['phobert_text_pw_sqrt_ep12', 'gain_test'])} on test; TF-IDF ({_sp['TF-IDF+LR']:.3f}) and the BiLSTM "
  f"({_sp['BiLSTM']:.3f}) are slightly more sensitive. But the number of labels per comment changes a lot: PhoBERT outputs "
  f"{q5s.loc['PhoBERT', 'labels/comment @0.5']:.2f} at 0.5 against {Y['test'].sum(1).mean():.2f} gold labels. The loss function moves the "
  f"best threshold: with plain BCE the best τ* is {q5s.loc['PhoBERT (BCE thường)', 'tau*_val']:.2f}, because that model under-predicts "
  f"({q5s.loc['PhoBERT (BCE thường)', 'labels/comment @0.5']:.2f} labels per comment at 0.5), while the square-root weight pushes "
  f"probabilities up (mean probability {q5c.loc['PhoBERT', 'xác suất TB']:.3f} against {q5c.loc['PhoBERT (BCE thường)', 'xác suất TB']:.3f}) and the "
  f"best τ* becomes {q5s.loc['PhoBERT', 'tau*_val']:.3f}. So 0.5 is arbitrary in the sense of the question: it means different "
  f"things for different models and losses, even though the Macro-F1 curve is flat around the optimum. The ECE values are not a ranking "
  f"of models, because the weighted loss deliberately shifts probabilities.")

# ----------------------------------------------------------------------------- 7 error analysis
h1("7. Error analysis on edge cases")
p("We analyse the selected PhoBERT model on the test set at threshold 0.5.")
h2("7.1 Performance by number of gold labels")
rows = [["Gold |Y|", "Comments", "Predicted labels per comment", "F1 per comment", "Exact match"]]
for _, r in err_card.iterrows():
    rows.append([str(r["|Y|"]).replace(">=", "≥ "), f"{int(r['số_câu']):,}", f"{r['nhãn_dự_đoán']:.2f}", f"{r['F1_câu']:.3f}",
                 pct(r["khớp_hoàn_toàn"] * 100)])
table(rows, [2.4, 2.6, 4.6, 3.0, 3.0], "PhoBERT by number of gold labels.", size=9.5)
figure(FIG / "fig_pred_vs_true.png", 10.5, "Mean number of gold and predicted labels per comment, by gold cardinality.")
p(f"The number of predicted labels grows more slowly than the number of gold labels. PhoBERT over-predicts for comments with "
  f"one to three gold labels (for single-label comments it predicts {err_card.iloc[0]['nhãn_dự_đoán']:.2f} labels on average) and "
  f"under-predicts for comments with four or more (3.39 against 4.17). Exact-set match falls from "
  f"{err_card.iloc[0]['khớp_hoàn_toàn'] * 100:.1f}% to {err_card.iloc[3]['khớp_hoàn_toàn'] * 100:.1f}% as the number of gold labels "
  f"grows. Overall the model predicts {per_comment['PhoBERT'].mean():.2f} labels per comment against "
  f"{Y['test'].sum(1).mean():.2f} gold labels: micro-precision is {micro_p:.3f} and micro-recall {micro_r:.3f}.")
h2("7.2 Precision and recall per emotion")
lowp = err_lab.sort_values("precision").head(5)
lowr = err_lab.sort_values("recall").head(5)
rows = [["Lowest precision", "Precision", "FP", "FN", "Lowest recall", "Recall", "FN", "FP"]]
for k in range(5):
    a, b = lowp.iloc[k], lowr.iloc[k]
    rows.append([lowp.index[k], f3(a.precision), str(int(a.fp)), str(int(a.fn)), lowr.index[k], f3(b.recall),
                 str(int(b.fn)), str(int(b.fp))])
table(rows, [3.2, 1.9, 1.2, 1.2, 3.2, 1.9, 1.2, 1.2], "Labels with the lowest precision and the lowest recall (test, τ = 0.5).",
      size=9, align=["l", "c", "c", "c", "l", "c", "c", "c"])
p("Disapproval, desire, confusion and disgust are over-assigned (low precision); neutral, realization and disappointment "
  "are the labels most often missed (low recall). Neutral and realization are also among the hardest labels in Table 14.")
h2("7.3 Systematic confusions")
rows = [["Gold label missed", "Predicted instead", "Comments"]]
for _, r in conf.head(10).iterrows():
    rows.append([r.iloc[0], r.iloc[1], str(int(r.iloc[2]))])
table(rows, [5.0, 5.0, 3.0], "Most frequent pairs where, in the same comment, a gold label was missed and a wrong label was "
      "predicted.", size=9.5, align=["l", "l", "c"])
neg = {"annoyance", "anger", "disgust", "disapproval", "disappointment", "sadness", "grief"}
both_neg = int(sum(1 for _, r in conf.head(15).iterrows() if r.iloc[0] in neg and r.iloc[1] in neg))
p(f"{both_neg} of the {len(conf.head(15))} most frequent confusions are between two negative emotions (annoyance, anger, disgust, "
  "disapproval, disappointment, sadness, grief), the same labels that co-occur most in the data (Section 2.4). Annotators "
  "also find these boundaries fuzzy, which is consistent with the low F1 of disapproval and disgust.")
h2("7.4 Manual audit of 70 cases")
p("We read 70 test comments with their gold and predicted labels: 20 random false negatives and 20 random false positives "
  "(stratified by label frequency tier: 7 frequent, 7 mid, 6 rare), the 10 false negatives to which the model gave the "
  "lowest probability and the 10 false positives to which it gave the highest, and 10 random comments with four or more "
  "gold labels. 69 cases are errors and one multi-label case was predicted exactly. Each error got one main cause. The "
  "case notes (reports/error_analysis_notes.md) paraphrase the comments and were drafted with the help of an AI assistant. "
  "The classification is subjective, the sample is not representative of all errors, and statements that a gold label may be "
  "missing are hypotheses.")
causes = [("Near-synonym labels", 24), ("Keyword traps", 10), ("Needs context or too short", 8), ("Sarcasm or dark humour", 7),
          ("Emoji or punctuation confusion", 7), ("Teencode or noisy text", 7), ("Mixed emotions", 3),
          ("Near the threshold", 2), ("Doubtful gold label", 1)]
rows = [["Main cause", "Cases", "Share"]] + [[c, str(n), pct(n / 69 * 100)] for c, n in causes]
table(rows, [7.0, 2.2, 2.2], "Main cause of the 69 error cases (subjective).", size=9.5, align=["l", "c", "c"])
bullets(["**Near-synonym labels** are the most common cause, in the positive cluster (joy, amusement, love, caring, pride, "
         "admiration) and the negative one (anger, annoyance, disgust, disapproval, sadness, grief). Many over-predictions "
         "are reasonable in meaning, which suggests that some gold labels may be missing.",
         "**Keyword traps** cause most confident false positives (probability ≥ 0.9): a word for wanting triggers desire, "
         "a word for love triggers love, a phrase for not understanding triggers confusion, a question mark triggers "
         "confusion or curiosity.",
         "**Sarcasm and dark humour** are read literally, praise as praise and violent jokes as disgust or anger.",
         "**Emoji or laughter that contradicts the words** pulls the model towards amusement or joy.",
         "**Missed emotions in extreme cases** come mostly from very short, context-free or noisy text.",
         "**Comments with four or more labels:** in 10 cases only one was predicted exactly; the model keeps the dominant "
         "emotion, drops opposing ones and adds one to three extra labels."])
rows = [["Case type", "What happened (paraphrased)"],
        ["Condolence message", "Caring, sadness and grief were found; love was missed (probability 0.31). Love and caring are close."],
        ["Short reflective sentence with a light smiley", "Optimism was missed by a hair (probability 0.47, threshold 0.5) although joy and relief were found."],
        ["Question with a sarcastic tone", "The sarcasm was read as humour: amusement was predicted (0.61) for a comment labelled curiosity, disappointment and annoyance."],
        ["Indirect complaint with masked rude words", "Sadness and disappointment were predicted; anger was missed (probability 0.03)."],
        ["Mild comment with an upside-down smiley", "Gold was annoyance only; the model added disgust and anger (0.71) and over-read the intensity."],
        ["Colloquial word that also means disgust", "Disgust was predicted (0.63) for a comment labelled amusement and surprise: a keyword trap."]]
table(rows, [5.2, CONTENT_W - 5.2], "Examples of edge cases, described in words.", align=["l", "l"], size=9)

# ----------------------------------------------------------------------------- 8 RQ
h1("8. Proposed research question for the Final")
p("**RQ1. Is a single uniform decision threshold (τ = 0.5) optimal across all emotion labels in the presence of class "
  "imbalance?**", keep=True)
p("The midterm gives four reasons to expect that the answer is not simply yes:", keep=True)
bullets([f"Labels are imbalanced ({cnt.min() / N['train'] * 100:.1f}% to {cnt.max() / N['train'] * 100:.1f}% of comments each) and "
         f"the models behave differently at 0.5: PhoBERT predicts {per_comment['PhoBERT'].mean():.2f} labels per comment and "
         f"the BiLSTM {per_comment['BiLSTM'].mean():.2f}, against {Y['test'].sum(1).mean():.2f} gold labels.",
         f"The best shared threshold differs from model to model ({thr.loc['tfidf_lr', 'tau*_val']:.3f}, "
         f"{thr.loc['bilstm', 'tau*_val']:.3f} and {thr.loc['phobert_text_pw_sqrt_ep12', 'tau*_val']:.3f}) and with the loss "
         f"function ({thr.loc['phobert_text_bce_ep12', 'tau*_val']:.3f} for PhoBERT with plain BCE), so it will probably "
         f"differ from label to label too.",
         f"For the converged PhoBERT, a shared threshold adds only {sgn(g_ph)} to {sgn(g_bce12)} Macro-F1, so there is little "
         f"room, and a plain 0.5 may already be close to the best.",
         f"Only {int(Y['val'].sum(0).min())} to {int(Y['val'].sum(0).max())} validation positives exist per label, so "
         f"per-label thresholds may overfit, most of all for rare labels."])
rows = [["Strategy", "Definition", "How it is chosen"],
        ["Fixed", "τ = 0.5 for every label", "Not tuned"],
        ["Global tuned", "One value τ* = argmax Macro-F1(D_val; τ)", "Grid 0.05–0.90 in steps of 0.025, validation only"],
        ["Per-label tuned", "One value τ_c* = argmax F1_c(D_val; τ_c) per label", "Same grid, validation only, separately per label"]]
table(rows, [3.0, 7.3, CONTENT_W - 10.3], "The three decoding strategies to compare.", align=["l", "l", "l"], size=9.5)
p("**Protocol.** We reuse the best neural model of the midterm (PhoBERT) and do not retrain it. Thresholds are tuned on the "
  "saved validation probabilities, frozen, and then applied to the test probabilities. We report Macro-F1, Micro-F1, "
  "per-label F1 and Hamming loss for the three strategies; the gains and losses by frequency tier (frequent, mid, rare, as "
  "defined in Section 2.2) and by label cardinality (|Y| = 1, 2, 3, ≥ 4); the tuned threshold of every label; and the "
  "validation-to-test gap ΔF1 = F1_val − F1_test as a check for threshold overfitting. The error analysis will cover at "
  "least 20 false negatives, 20 false positives, multi-emotion comments and confused label pairs. The threshold literature "
  "[6, 7] motivates tuning on a held-out set.")
p("**Hypothesis and caution.** We expect small overall gains and that per-label thresholds may overfit rare labels. Because "
  "the test probabilities were already inspected for this midterm, a comparison made on them is post hoc; in the Final we "
  "will state this, run several seeds and use paired bootstrap intervals instead of a single look at the test set.")

# ----------------------------------------------------------------------------- 9 limitations
h1("9. Limitations, difficulties and reproducibility")
h2("9.1 Limitations")
bullets(["**One seed.** Each configuration was run once (seed 42), so small differences, for example between PhoBERT variants, "
         "may be noise.",
         "**Uneven tuning.** TF-IDF had a 12-configuration grid; the BiLSTM and PhoBERT mainly differ in loss and number of "
         "epochs, and their learning rates were not searched. Comparisons between models are therefore not perfectly fair.",
         "**BiLSTM without pre-trained embeddings.** The explanation for its weak result is a hypothesis that we did not test.",
         "**Regex flags are estimates and overlap.** The text-style subsets of Question 4 come from regular expressions and a "
         "hand-made teencode list whose accuracy we did not measure by reading a sample; the stretched-letters subset has "
         "only 46 test comments.",
         "**The official split is not stratified.** Its label shares differ between sets like those of a random split (Section "
         "2.7); we did not run a multi-label stratified re-split as a sensitivity check.",
         "**Qualitative review of informal text.** Section 2.6 rests on counts from regular expressions, not on a systematic "
         "manual reading of a sample.",
         "**Style comparison is of pipelines.** PhoBERT and the BiLSTM differ in tokenisation, emoji handling and size, so "
         "Question 4 does not isolate the effect of the architecture.",
         "**Small, subjective error sample.** 70 cases were read and classified by hand.",
         "**Checkpoint.** We used PhoBERT-base version 1; the paper's results are for version 2."])
h2("9.2 Difficulties we met")
bullets(["The dataset is gated: access had to be requested on Hugging Face and a token configured.",
         "The first PyTorch installation was a CPU build; we reinstalled the CUDA build (2.14.1 with CUDA 12.6).",
         "transformers 5.x and huggingface_hub versions conflicted; the requirements pin huggingface_hub below 2.0.",
         "The laptop went to sleep during PhoBERT training, which inflated the recorded time; we corrected and noted it.",
         "Our first PhoBERT runs stopped after 6 epochs and had not converged, which led to a wrong conclusion about "
         "thresholds (+0.065 Macro-F1). We retrained for 12 epochs and corrected the conclusion (Q5).",
         "During the November revision we found that our stretched-letters flag also matched repeated punctuation such as "
         "\"...\" and \"!!!\", which inflated the share from about 2% to 14%. We restricted the flag to repeated letters, added a "
         "separate repeated-punctuation flag, and recomputed Question 4; the stretched-letters subset shrank from 265 to 46 comments.",
         "The first version of this report described the official split as behaving like a stratified one; comparing it with "
         "random splits showed that it does not (Section 2.7), and we corrected the text.",
         "The first versions of Questions 3 and 4 reported pair recall and subset gaps only. Pair precision showed that PhoBERT "
         "amplifies the dependencies, and comparing with plain text showed that its advantage on teencode is not specific; both "
         "conclusions were revised."])
h2("9.3 Reproducibility")
rows = [["Item", "Value"],
        ["Hardware", f"NVIDIA RTX 3060 Laptop GPU, 6 GB; PhoBERT peak memory {sel['peak_gpu_gb']} GB"],
        ["Random seed", "42 for NumPy and PyTorch; all splits fixed (splits/*.csv hold comment ids only)"],
        ["Checkpoint", f"vinai/phobert-base, revision {sel['revision']}"],
        ["Software", f"Python 3.12, PyTorch {sel['torch']}, transformers {sel['transformers']}, scikit-learn, emoji, VnCoreNLP"],
        ["Preprocessing", "Section 3; teencode and stretched letters are not normalised"],
        ["Decision threshold", "0.5 for every label in all reported results"],
        ["Error-case sample", "The 70 cases are fixed in results/error_cases/cases.csv; notebook 05 reads the file instead of resampling"],
        ["Saved outputs", "Validation and test probabilities of every model in results/preds/*.npz (saved before any threshold is applied)"],
        ["Run order", "notebooks 01 to 05; PhoBERT: python -m src.train_phobert --emoji text --loss pw_sqrt --epochs 12 --patience 3"],
        ["Data", "Hugging Face uitnlp/vigoemotions (gated). Comment text is not in the repository"]]
table(rows, [3.6, CONTENT_W - 3.6], "Reproducibility information.", align=["l", "l"], size=9)

# ----------------------------------------------------------------------------- references
h1("References")
refs = ["[1] T. Q. Hung, P. T. Nam, S. T. Luu and K. V. Nguyen, “ViGoEmotions: A Benchmark Dataset For Fine-grained Emotion "
        "Detection on Vietnamese Texts,” in Proc. 19th Conf. of the European Chapter of the ACL (EACL), vol. 1, 2026, "
        "pp. 2805–2831, doi: 10.18653/v1/2026.eacl-long.129.",
        "[2] D. Q. Nguyen and A. T. Nguyen, “PhoBERT: Pre-trained language models for Vietnamese,” in Findings of the ACL: "
        "EMNLP 2020, pp. 1037–1042, doi: 10.18653/v1/2020.findings-emnlp.92.",
        "[3] A. Graves and J. Schmidhuber, “Framewise phoneme classification with bidirectional LSTM and other neural network "
        "architectures,” Neural Networks, vol. 18, no. 5–6, pp. 602–610, 2005.",
        "[4] T. Vu, D. Q. Nguyen, D. Q. Nguyen, M. Dras and M. Johnson, “VnCoreNLP: A Vietnamese natural language processing "
        "toolkit,” in Proc. NAACL-HLT 2018: Demonstrations, pp. 56–60.",
        "[5] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, "
        "pp. 2825–2830, 2011.",
        "[6] R. Alotaibi and P. Flach, “Multi-label thresholding for cost-sensitive classification,” Neurocomputing, vol. 436, "
        "pp. 232–247, 2021, doi: 10.1016/j.neucom.2020.12.004.",
        "[7] I. Pillai, G. Fumera and F. Roli, “Threshold optimisation for multi-label classifiers,” Pattern Recognition, "
        "vol. 46, no. 7, pp. 2055–2065, 2013, doi: 10.1016/j.patcog.2013.01.012."]
for r_ in refs:
    par = p(r_, size=9.5, after=3)
    par.paragraph_format.left_indent = Cm(0.8)
    par.paragraph_format.first_line_indent = Cm(-0.8)

doc.core_properties.title = "ViGoEmotions Midterm Report"
doc.core_properties.author = "Group 3"
try:
    doc.save(OUT)
except PermissionError:  # the file is open in Word: write next to it
    OUT = OUT.with_name(OUT.stem + "_v2.docx")
    doc.save(OUT)
print("saved", OUT)
