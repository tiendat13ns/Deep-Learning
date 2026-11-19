"""Build the final report on ATI_Final_Report_Template.docx.

Run from ati_final (after notebooks 06 and 07 and reports/final_report/make_figures.py):
    python reports/final_report/make_report.py                  # black-and-white figures
    python reports/final_report/make_report.py --style color    # colour figures -> ..._report_color.docx
The two versions differ only in their figures (results/figures/final_report_bw/ or final_report_color/). Then open the .docx in Word once (or run update_fields.ps1) so the table of contents is filled.

Every table and every key number in the text is read from results/ at build time; nothing is typed by
hand except the qualitative prose. Comment text is never printed: failure examples are paraphrased.
Inline markup in the strings below: **bold**, ~italic~, _{subscript}, ^{superscript}.
"""
import argparse
import json
import sys
import re
from pathlib import Path

import latex2mathml.converter
import lxml.etree as ET
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser()
_ap.add_argument("--style", choices=("bw", "color"), default="bw")
STYLE = _ap.parse_args().style
FIG = ROOT / "results" / "figures" / ("final_report_color" if STYLE == "color" else "final_report_bw")
RQ1 = ROOT / "results" / "rq1"
TAB = ROOT / "results" / "tables"
TEMPLATE = ROOT / "ATI_Final_Report_Template.docx"
PREFIX = "62FIT4ATI-CLC02_G03_P02"
OUT = HERE / f"{PREFIX}_report{'_color' if STYLE == 'color' else ''}.docx"
MML2OMML = Path(r"C:/Program Files/Microsoft Office/root/Office16/MML2OMML.XSL")
STRAT = ("fixed", "global", "per_label")
SNAME = {"fixed": "Fixed (τ = 0.5)", "global": "Global tuned (τ^{*})", "per_label": "Per-label tuned (τ^{*}_{k})"}

# ============================================================================ numbers from results/
f4 = lambda x: f"{x:.4f}"
f3 = lambda x: f"{x:.3f}"
s4 = lambda x: f"{x:+.4f}".replace("-", "−")
m4 = lambda x: f"{x:.4f}".replace("-", "−")
pm = lambda m, s: f"{m:.4f} ± {s:.4f}"
pct = lambda x, d=1: f"{100 * x:.{d}f}%"

base = pd.read_csv(TAB / "baseline_results.csv").set_index("run")
cost = pd.read_csv(TAB / "model_cost.csv").set_index("run")
dist = pd.read_csv(TAB / "label_distribution.csv", index_col=0)
informal = pd.read_csv(TAB / "informal_features.csv").set_index("split")
tlen = pd.read_csv(TAB / "text_length.csv")
seed_res = pd.read_csv(RQ1 / "tables" / "seed_results.csv")
agg = pd.read_csv(RQ1 / "tables" / "seed_mean_std.csv").set_index("strategy")
sdelta = pd.read_csv(RQ1 / "tables" / "seed_macro_f1_deltas.csv", index_col=0)
sdelta.index = sdelta.index.astype(str)
transfer = pd.read_csv(RQ1 / "tables" / "seed_gain_transfer.csv").set_index("seed")
stab = pd.read_csv(RQ1 / "tables" / "seed_threshold_stability.csv").set_index("label")
boot = pd.read_csv(RQ1 / "tables" / "bootstrap_deltas.csv")
summ = pd.read_csv(RQ1 / "tables" / "strategy_metrics.csv")
tiers = pd.read_csv(RQ1 / "tables" / "tier_metrics.csv")
card = pd.read_csv(RQ1 / "tables" / "cardinality_metrics.csv")
gaps = pd.read_csv(RQ1 / "tables" / "generalization_gaps.csv").set_index("strategy")
comp = pd.read_csv(RQ1 / "tables" / "per_label_comparison.csv").set_index("label")
calib = json.loads((RQ1 / "calibration.json").read_text(encoding="utf-8"))
EA = RQ1 / "error_analysis"
errc = pd.read_csv(EA / "error_counts.csv")
flips = pd.read_csv(EA / "decision_flips.csv")
pairs = pd.read_csv(EA / "confused_pairs.csv")
carderr = pd.read_csv(EA / "cardinality_errors.csv")
causes = pd.read_csv(EA / "cause_summary.csv", index_col=0)
ann = pd.read_csv(EA / "annotations.csv", dtype=str, keep_default_na=False)
defens = pd.read_csv(EA / "defensible_by_group.csv", index_col=0)
q5 = pd.read_csv(TAB / "q5_summary.csv")
q5 = q5[q5.model.isin(["PhoBERT", "PhoBERT (BCE thường)"])].reset_index(drop=True)  # row 0 selected, row 1 plain BCE
assert q5.model.tolist() == ["PhoBERT", "PhoBERT (BCE thường)"]
costs = {s: json.loads((TAB / f"phobert_text_pw_sqrt_ep12{'' if s == 42 else f'_seed{s}'}_cost.json").read_text(encoding="utf-8"))
         for s in (42, 43, 44)}

T42 = summ[summ.split == "test"].set_index("strategy")
V42 = summ[summ.split == "val"].set_index("strategy")
MAIN = "phobert_text_pw_sqrt_ep12"


def bt(comparison, scope="all", metric="macro_f1"):
    r = boot[(boot.comparison == comparison) & (boot.scope == scope) & (boot.metric == metric)].iloc[0]
    return r


def ci(comparison, scope="all", metric="macro_f1"):
    r = bt(comparison, scope, metric)
    return f"{s4(r.observed_delta)} [{s4(r.ci95_low)}, {s4(r.ci95_high)}]"


def tier42(strategy, tier, col="macro_f1"):
    return tiers[(tiers.split == "test") & (tiers.strategy == strategy) & (tiers.tier == tier)][col].iloc[0]


def seedmean(strategy, col):
    return agg.loc[strategy, f"{col}_mean"], agg.loc[strategy, f"{col}_std"]


n_train, n_val, n_test = 16531, 2066, 2067
th = np.asarray(calib["per_label_thresholds"])
val_pos = np.asarray(calib["validation_positive_counts"])
glob_th = seed_res.groupby("seed").global_threshold.first()
flip_tot = flips.groupby("comparison")[["fn_recovered", "fn_introduced", "fp_removed", "fp_introduced",
                                        "net_correct_decisions"]].sum()
pl_vs_fixed = flips[flips.comparison == "per_label vs fixed"].set_index("label")
gpu_seconds = float(cost.loc[[r for r in cost.index if r.startswith("phobert")], "train_seconds"].sum()
                    + cost.loc["bilstm", "train_seconds"] + costs[43]["train_seconds"] + costs[44]["train_seconds"])

# ============================================================================ document plumbing
doc = Document(TEMPLATE)
body = doc.element.body
FINAL_SECT = body[-1]
assert FINAL_SECT.tag == qn("w:sectPr")
XSLT = ET.XSLT(ET.parse(str(MML2OMML)))


def text_of(e):
    return "".join(t.text or "" for t in e.iter(qn("w:t")))


def style_of(e):
    ps = e.find(qn("w:pPr") + "/" + qn("w:pStyle"))
    return ps.get(qn("w:val")) if ps is not None else None


def mk(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k), str(v))
    return e


TOKEN = re.compile(r"(\*\*.+?\*\*|~.+?~|_\{[^}]*\}|\^\{[^}]*\})")


def add_runs(p, text, size=None, bold=False, italic=False):
    """Append runs to a w:p element, honouring the inline markup."""
    for part in TOKEN.split(text):
        if not part:
            continue
        b, i, va = bold, italic, None
        if part.startswith("**"):
            part, b = part[2:-2], True
        elif part.startswith("~") and part.endswith("~") and len(part) > 1:
            part, i = part[1:-1], True
        elif part.startswith("_{"):
            part, va = part[2:-1], "subscript"
        elif part.startswith("^{"):
            part, va = part[2:-1], "superscript"
        # nested markup inside bold (e.g. **τ^{*}**)
        if (b and not bold) and TOKEN.search(part):
            add_runs(p, part, size, True, i)
            continue
        r = mk("w:r")
        rpr = mk("w:rPr")
        if b:
            rpr.append(mk("w:b")); rpr.append(mk("w:bCs"))
        if i:
            rpr.append(mk("w:i")); rpr.append(mk("w:iCs"))
        if size:
            rpr.append(mk("w:sz", **{"w:val": int(size * 2)})); rpr.append(mk("w:szCs", **{"w:val": int(size * 2)}))
        if va:
            rpr.append(mk("w:vertAlign", **{"w:val": va}))
        if len(rpr):
            r.append(rpr)
        t = mk("w:t")
        t.text = part
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        r.append(t)
        p.append(r)


def new_p(style=None, align=None, keep_next=False, num=False, tabs=None, page_break_before=False):
    p = mk("w:p")
    ppr = mk("w:pPr")
    if style:
        ppr.append(mk("w:pStyle", **{"w:val": style}))
    if keep_next:
        ppr.append(mk("w:keepNext"))
    if page_break_before:
        ppr.append(mk("w:pageBreakBefore"))
    if num:
        numpr = mk("w:numPr")
        numpr.append(mk("w:ilvl", **{"w:val": 0}))
        numpr.append(mk("w:numId", **{"w:val": 2}))
        ppr.append(numpr)
    if tabs:
        t = mk("w:tabs")
        for kind, pos in tabs:
            t.append(mk("w:tab", **{"w:val": kind, "w:pos": pos}))
        ppr.append(t)
    if align:
        ppr.append(mk("w:jc", **{"w:val": align}))
    p.append(ppr)
    return p


def put(e):
    FINAL_SECT.addprevious(e)
    return e


def P(text, style=None, align=None, keep_next=False):
    p = new_p(style, align, keep_next)
    add_runs(p, text)
    return put(p)


def H1(text, page_break=False):
    p = new_p("Heading1", page_break_before=page_break)
    add_runs(p, text)
    return put(p)


def H2(text):
    return P(text, "Heading2")


def BUL(items):
    for it in items:
        p = new_p("ListParagraph", align="both", num=True)
        add_runs(p, it)
        put(p)


# ---- numbered objects. Cross-references use the keys below; the order is asserted on insertion.
TABLE_KEYS = ["rq", "data", "train", "compute", "baseline", "main", "deltas", "card", "gap", "errors", "pairs",
              "causes", "examples", "contrib", "ai", "perlabel"]
FIG_KEYS = ["dist", "protocol", "tiers", "thresholds", "transfer", "flips", "carderr"]
_next = {"Table": 0, "Figure": 0}


def T(key):
    return f"Table {TABLE_KEYS.index(key) + 1}"


def F(key):
    return f"Figure {FIG_KEYS.index(key) + 1}"


def caption(kind, key, text):
    keys = TABLE_KEYS if kind == "Table" else FIG_KEYS
    assert keys[_next[kind]] == key, f"{kind} order: expected {keys[_next[kind]]}, got {key}"
    _next[kind] += 1
    p = new_p("Caption", keep_next=(kind == "Table"))
    add_runs(p, f"{kind} ", bold=True)
    fld = mk("w:fldSimple", **{"w:instr": f"SEQ {kind} \\* ARABIC"})
    r = mk("w:r"); t = mk("w:t"); t.text = str(_next[kind]); r.append(t); fld.append(r)
    p.append(fld)
    add_runs(p, ": ", bold=True)
    add_runs(p, text)
    put(p)


def figure(path, key, text, width_cm=16.0):
    p = doc.add_paragraph()  # python-docx inserts before the final sectPr
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.keep_with_next = True
    pf.space_before = Pt(6)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    caption("Figure", key, text)


BORDER = '<w:{0} w:val="single" w:color="000000" w:sz="4"/>'


def table(key, title, rows, widths, align=None, size=10, note=None, header_rows=1, keep=None):
    caption("Table", key, title)
    total = 9070
    dxa = [int(round(w / sum(widths) * total)) for w in widths]
    dxa[-1] += total - sum(dxa)
    tbl = mk("w:tbl")
    tpr = mk("w:tblPr")
    tpr.append(mk("w:tblW", **{"w:type": "dxa", "w:w": total}))
    tpr.append(mk("w:jc", **{"w:val": "center"}))
    b = mk("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b.append(mk(f"w:{side}", **{"w:val": "single", "w:color": "auto", "w:sz": 4}))
    tpr.append(b)
    tpr.append(mk("w:tblLayout", **{"w:type": "fixed"}))
    mar = mk("w:tblCellMar")
    for side, v in (("top", 30), ("left", 80), ("bottom", 30), ("right", 80)):
        mar.append(mk(f"w:{side}", **{"w:type": "dxa", "w:w": v}))
    tpr.append(mar)
    tbl.append(tpr)
    grid = mk("w:tblGrid")
    for w in dxa:
        grid.append(mk("w:gridCol", **{"w:w": w}))
    tbl.append(grid)
    align = align or (["left"] + ["center"] * (len(widths) - 1))
    keep = len(rows) <= 12 if keep is None else keep  # short tables stay on one page
    for ri, row in enumerate(rows):
        tr = mk("w:tr")
        trpr = mk("w:trPr")
        trpr.append(mk("w:cantSplit"))
        if ri < header_rows:
            trpr.append(mk("w:tblHeader"))
        tr.append(trpr)
        for ci_, (cell, w) in enumerate(zip(row, dxa)):
            tc = mk("w:tc")
            tcpr = mk("w:tcPr")
            tcpr.append(mk("w:tcW", **{"w:type": "dxa", "w:w": w}))
            tcpr.append(mk("w:vAlign", **{"w:val": "center"}))
            tc.append(tcpr)
            for li, line in enumerate(str(cell).split("\n")):
                p = new_p("TableText", align=align[ci_] if ri >= header_rows else ("left" if ci_ == 0 else "center"),
                          keep_next=keep and ri < len(rows) - 1)
                add_runs(p, line, size=size, bold=(ri < header_rows))
                tc.append(p)
            tr.append(tc)
        tbl.append(tr)
    put(tbl)
    if note:
        p = new_p("TableText", align="left")
        add_runs(p, note, size=9)
        put(p)
    sp = new_p()  # breathing space after the table
    sp_ppr = sp.find(qn("w:pPr"))
    sp_ppr.append(mk("w:spacing", **{"w:after": 0, "w:line": 120, "w:lineRule": "auto"}))
    put(sp)


_eq = [0]


def equation(latex):
    _eq[0] += 1
    mml = latex2mathml.converter.convert(latex)
    omml = XSLT(ET.fromstring(mml)).getroot()
    p = new_p(tabs=[("center", 4535), ("right", 9070)], align="left")
    add_runs(p, "\t")
    from docx.oxml import parse_xml
    p.append(parse_xml(ET.tostring(omml)))
    add_runs(p, f"\t({_eq[0]})")
    put(p)
    return _eq[0]


# ============================================================================ cover, front matter, header
def replace_in_paragraph(p_el, old, new):
    ts = list(p_el.iter(qn("w:t")))
    for t in ts:  # prefer one run, so the formatting of the other runs is kept
        if t.text and old in t.text:
            t.text = t.text.replace(old, new)
            return True
    full = "".join(t.text or "" for t in ts)
    if old not in full:
        return False
    ts[0].text = full.replace(old, new)
    for t in ts[1:]:
        t.text = ""
    return True


COVER = {
    "[PROJECT TITLE]": "Is a Single 0.5 Threshold Enough? Decision-Threshold Calibration for Multi-Label Vietnamese Emotion Recognition on ViGoEmotions",
    "Project [P00]  ·  Group [G00]  ·  Class [CLASS CODE]": "Project P02  ·  Group G03  ·  Class 62FIT4ATI-CLC02",
    "Hanoi, [Month YYYY]": "Hanoi, November 2026",
}
# The group representative (submitter) is marked with *, as the template asks.
MEMBERS = [("Nguyễn Quốc Khanh", "2301140051"), ("Đinh Tiến Đạt*", "2301140022"), ("Nguyễn Duy Anh", "2301140006")]
for p_el in body.iter(qn("w:p")):
    for old, new in COVER.items():
        replace_in_paragraph(p_el, old, new)
member_table = doc.tables[0]
for (name, sid), row in zip(MEMBERS, member_table.rows[1:]):
    replace_in_paragraph(row.cells[1].paragraphs[0]._p, text_of(row.cells[1].paragraphs[0]._p), name)
    replace_in_paragraph(row.cells[2].paragraphs[0]._p, "[Student ID]", sid)
for sec in doc.sections:
    for hp in sec.header.paragraphs:
        replace_in_paragraph(hp._p, "[PREFIX]", PREFIX)

# remove the instruction page: from "Template Instructions" up to the "Abstract" heading
children = list(body)
start = next(i for i, e in enumerate(children) if e.tag == qn("w:p") and text_of(e) == "Template Instructions")
stop = next(i for i, e in enumerate(children) if e.tag == qn("w:p") and text_of(e) == "Abstract")
for e in children[start:stop]:
    body.remove(e)
# remove guidance paragraphs in the front matter (never the paragraph that carries a sectPr)
for e in list(body):
    if e.tag == qn("w:p") and style_of(e) == "Guidance" and e.find(".//" + qn("w:sectPr")) is None:
        if text_of(e).startswith("["):
            intro_seen = any(text_of(x).startswith("1  Introduction") for x in e.itersiblings(preceding=True))
            if not intro_seen:
                body.remove(e)

# remove the template body (Section 1 heading .. end) and rebuild it
children = list(body)
first = next(i for i, e in enumerate(children) if e.tag == qn("w:p") and text_of(e).startswith("1  Introduction"))
for e in children[first:-1]:
    body.remove(e)

ABSTRACT = (
    "Multi-label emotion classifiers output one probability per emotion and almost always turn them into "
    "labels with the same cut-off, 0.5. We ask whether this single threshold is optimal on ViGoEmotions, a "
    "corpus of 20,664 Vietnamese social-media comments annotated with 27 emotions plus neutral and a "
    f"4.5-fold imbalance between its most and least frequent labels. Keeping the strongest Midterm model "
    f"(PhoBERT-base fine-tuned with square-root positive weights) fixed, we decode the same validation and "
    f"test probabilities three ways: a fixed 0.5, one global threshold, and one threshold per label, both tuned "
    f"on validation only and frozen before test inference. Over three training seeds, test Macro-F1 is "
    f"{pm(*seedmean('fixed', 'test_macro_f1'))} at 0.5, {pm(*seedmean('global', 'test_macro_f1'))} with a "
    f"global threshold and {pm(*seedmean('per_label', 'test_macro_f1'))} with per-label thresholds. "
    f"A global threshold above 0.5 helps in every seed and reliably lowers Hamming loss by curbing "
    f"over-prediction, but per-label thresholds add only {s4(sdelta.loc['mean', 'per_label - global'])} ± "
    f"{f4(sdelta.loc['std', 'per_label - global'])} Macro-F1 over it, change sign across seeds, have a paired "
    f"bootstrap interval that includes zero, overfit validation (validation-to-test gap of about +0.016) and "
    f"do not help tail labels. An error audit of 80 cases shows that tuning trades false positives for false "
    f"negatives and cannot fix the model's main failure: too many labels on single-emotion comments and too few "
    f"on multi-emotion ones. The main limitation is that the test set had already been inspected during the "
    f"Midterm, so the comparison is post hoc.")
for e in body.iter(qn("w:p")):
    if text_of(e) == "[Write the abstract here.]":
        replace_in_paragraph(e, "[Write the abstract here.]", ABSTRACT)
    if text_of(e).startswith("Keywords:"):
        replace_in_paragraph(e, "[3–5 keywords separated by commas]",
                             "multi-label emotion recognition, decision threshold calibration, class imbalance, "
                             "Vietnamese, PhoBERT")
assert len(ABSTRACT.split()) <= 250, len(ABSTRACT.split())

# ============================================================================ 1 Introduction
H1("1  Introduction")
H2("1.1  Motivation and Task")
P("Detecting the emotions expressed in social-media comments supports content moderation, customer-feedback "
  "analysis and the study of public opinion. A single comment often carries several emotions at once, so the "
  "task is naturally ~multi-label~: each comment receives any subset of a fixed emotion inventory. ViGoEmotions "
  "[1] brings the 27-emotion taxonomy of GoEmotions [2] to Vietnamese. It contains 20,664 comments from "
  "Facebook, Reddit, Threads, TikTok, X and YouTube, each annotated with one or more of 27 emotions or neutral.")
P("Three properties make the corpus hard. First, the text is informal: about a quarter of the comments contain "
  f"emojis and about {informal.loc['train', 'teencode']:.0f}% contain teencode or abbreviations. Second, the labels "
  "overlap in meaning (annoyance, anger, disgust and disapproval; joy, amusement and excitement), and the dataset "
  "authors report a final average inter-annotator Cohen's κ of only about 0.56 [1]. Third, the labels are "
  f"imbalanced: amusement has {dist.loc['amusement', 'train']:,} positive training comments and relief only "
  f"{dist.loc['relief', 'train']}, a ratio of {dist.train.max() / dist.train.min():.1f}.")
P("A multi-label network outputs an independent sigmoid probability ~p~_{k} for every label ~k~, and a decision "
  "rule turns these probabilities into labels. The default rule predicts a label when ~p~_{k} ≥ 0.5. Under "
  "imbalance and weighted training losses there is no reason for 0.5 to be the best operating point for every "
  "label, and threshold selection is a long-studied step of multi-label learning [3]–[6]. In the Midterm we "
  f"benchmarked TF-IDF with logistic regression, a BiLSTM and PhoBERT-base at τ = 0.5; PhoBERT was best with test "
  f"Macro-F1 {f4(base.loc[MAIN, 'test_macro_f1'])}. The Final stage studies the decision rule of that model.")
H2("1.2  Research Questions and Hypotheses")
P(f"The track brief fixes the research question. We state a falsifiable hypothesis and its control in {T('rq')}. "
  "The hypothesis is rejected if the Macro-F1 difference between per-label and global thresholds is not larger "
  "than the variation across training seeds and its paired bootstrap confidence interval includes zero.")
table("rq", "Research question, hypothesis and control.", [
    ["RQ", "Question", "Hypothesis", "Control (Midterm)"],
    ["RQ1", "Is a single uniform decision threshold (τ = 0.5) optimal across all 27 emotion labels in the presence "
            "of severe class imbalance?",
     "H1: thresholds tuned on validation beat τ = 0.5 on test Macro-F1, and per-label thresholds beat one global "
     "threshold, with the largest gain on tail labels. H0: the differences are within seed and resampling noise.",
     f"PhoBERT-base (emoji → text, square-root positive weights, seed 42) decoded at τ = 0.5: test Macro-F1 "
     f"{f4(base.loc[MAIN, 'test_macro_f1'])}."],
], [1.2, 4.6, 5.6, 4.6], align=["left"] * 4, size=10)
H2("1.3  Contributions")
BUL([
    "**A frozen, controlled comparison.** The same PhoBERT probabilities are decoded with a fixed, a global and a "
    "per-label threshold. Thresholds are fitted on validation only and written to disk, with hashes of their inputs, "
    "before any test file is opened; the code checks the split ids and refuses changed inputs.",
    "**Evidence on whether the differences are real.** A paired bootstrap over test comments, two extra training "
    "seeds with their own calibration, the stability of each label's threshold across seeds, and a test of whether a "
    "label's validation gain transfers to test.",
    "**An error audit of what threshold tuning changes.** Decisions flipped per label, confused label pairs, "
    "under- and over-prediction by the number of gold labels, and a manual reading of 80 new cases, including every "
    "type of error that tuning introduces or removes.",
])

# ============================================================================ 2 Background
H1("2  Background and Related Work")
H2("2.1  Dataset Paper and Prior Results")
P("Tran et al. [1] built ViGoEmotions from an existing Vietnamese dataset and 53,890 newly crawled comments, "
  "filtered for duplicates, off-topic content and spam. Large language models (Gemini Flash, Llama 3 and Gemma) "
  "proposed labels under a written guideline, and three human annotators reviewed them; the final label set is "
  "the annotators' intersection or majority. Average pairwise Cohen's κ rose from about 0.4 in early rounds to "
  "about 0.56 on the full corpus, and disapproval had the lowest agreement. The corpus is split 8:1:1 into "
  "training, development and test sets.")
P("The authors fine-tuned eight Transformer encoders and sequence-to-sequence models with BCE loss for 12 epochs. "
  "Under their emoji-preserving setting, ViSoBERT reached the best test Macro-F1 (61.50%), followed by CafeBERT "
  "(61.17%); PhoBERT-base-v2 reached 56.33%. Per-label F1 was highest for gratitude, embarrassment, amusement and "
  "sadness and lowest for disapproval (35.78%), neutral and realization. The paper does not describe how the 28 "
  "sigmoid outputs are turned into labels, so the effect of the decision threshold on this corpus is open. Our "
  f"Midterm PhoBERT-base reached {f4(base.loc[MAIN, 'test_macro_f1'])} Macro-F1 at 0.5; the setups differ "
  "(model version, preprocessing, loss weighting, learning rate), so the numbers are not directly comparable.")
H2("2.2  Model Families")
P("**Linear bag-of-n-grams.** TF-IDF features with one-vs-rest logistic regression [7] are a fast, strong "
  "baseline for short texts and need no GPU. **Recurrent networks.** A bidirectional LSTM [8], [9] reads the "
  "comment in both directions and pools the hidden states; it learns its own embeddings from the training set. "
  "**Pre-trained Transformers.** BERT [10] introduced masked-language-model pre-training of Transformer encoders; "
  "RoBERTa [11] refined its training recipe. PhoBERT [12] is a RoBERTa model pre-trained on 20 GB of "
  "word-segmented Vietnamese text and expects input segmented by VnCoreNLP [13]. We fine-tune "
  "PhoBERT-base with a 28-way sigmoid output layer, so each label is an independent binary decision.")
H2("2.3  Related Studies on the Research Question")
P("Yang [3] compared rank-based (RCut), proportion-based (PCut) and score-based (SCut) thresholding for text "
  "categorisation and found that the choice of strategy changes results considerably, with per-category score "
  "thresholds tuned on held-out data being effective but sensitive to rare categories. Fan and Lin [4] studied "
  "per-label threshold selection for multi-label classification and showed that thresholds tuned for labels with "
  "few positives can overfit, proposing safeguards for such labels. Lipton et al. [5] analysed the threshold that "
  "maximises F1 and showed that it depends on the attainable F1 of the classifier, so a single default such as 0.5 "
  "is rarely optimal for F1. Pillai et al. [6] studied threshold optimisation for multi-label classifiers under "
  "macro- and micro-averaged F-measures, comparing one global threshold with label-specific thresholds, and "
  "Alotaibi and Flach [14] framed multi-label thresholding as a cost-sensitive decision per label.")
P("Our study differs in three ways: it isolates the decision rule by decoding one set of saved probabilities "
  "three ways, it fixes the grid, tie rule and data boundary before evaluation, and it measures whether gains "
  "survive test resampling and retraining, which single-run comparisons cannot show. It is, to our knowledge, "
  "the first such analysis on Vietnamese multi-label emotion data.")

# ============================================================================ 3 Dataset
H1("3  Dataset and Task Framing")
H2("3.1  Dataset Description")
from src.data import label_matrix, load_split  # noqa: E402  (train labels are needed for the split table)
Ys = {s: label_matrix(load_split(s)) for s in ("train", "val", "test")}
data_rows = [["Split", "Comments", "Mean labels", "|Y|=1", "|Y|=2", "|Y|=3", "|Y|≥4", "Median tokens", "Emoji", "Teencode"]]
for s, name in (("train", "Train"), ("val", "Validation"), ("test", "Test")):
    c = Ys[s].sum(1)
    med = tlen[(tlen.iloc[:, 0] == "n_tok") & (tlen.iloc[:, 1] == "50%")][s].iloc[0]
    data_rows.append([name, f"{len(c):,}", f"{c.mean():.2f}", pct((c == 1).mean()), pct((c == 2).mean()),
                      pct((c == 3).mean()), pct((c >= 4).mean()), f"{med:.0f}",
                      f"{informal.loc[s, 'emoji']:.1f}%", f"{informal.loc[s, 'teencode']:.1f}%"])
P(f"We downloaded the corpus from its gated Hugging Face page [15] after accepting its terms, which allow "
  "non-commercial research and education and forbid redistribution; the repository therefore stores only comment "
  "ids, labels and model outputs, never comment text. Each comment has one to five labels out of 28 (27 emotions "
  f"and neutral). {T('data')} summarises the three official splits; label cardinality is counted from the "
  "multi-hot vectors, and the emoji and teencode rates come from our regular-expression flags.")
table("data", "Dataset statistics per split (official split of [1]).", data_rows,
      [2.2, 1.9, 1.7, 1.4, 1.4, 1.4, 1.4, 1.7, 1.5, 1.7], size=9.5)
P("The splits are similar in every respect that matters here: about 1.9 labels per comment, about four fifths "
  "of comments with one or two labels and fewer than 4% with four or more. The ≥4 group of the test set contains only "
  f"{int((Ys['test'].sum(1) >= 4).sum())} comments, so results for it are noisy.")
H2("3.2  Label Distribution and Imbalance")
P(f"{F('dist')} shows the number of positive training comments per label. We split labels into frequency tiers on "
  "the training set before any test result was seen (Midterm decision): **head** ≥ 1,000 positives (10 labels), "
  "**mid** 750–999 (10 labels) and **tail** < 750 (8 labels). The imbalance is moderate rather than extreme: the "
  f"rarest label still has {dist.train.min()} training positives. For threshold calibration, however, what matters "
  f"is the validation set, where a label has only {val_pos.min()} to {val_pos.max()} positives "
  f"(relief has {val_pos[list(calib['labels']).index('relief')]}). A per-label threshold is therefore fitted on a "
  "few dozen to a few hundred positive examples.")
figure(FIG / "fig_label_distribution.png", "dist",
       "Positive training comments per label, shaded by frequency tier. Dotted lines mark the tier boundaries.")
H2("3.3  Leakage and Shortcut Risks; Split Protocol")
P("We use the official split of [1] and store its comment ids in ~splits/~. Every evaluation script checks that "
  "prediction files list exactly these ids in this order and that train, validation and test do not share an id. "
  "The official split is not stratified: label proportions differ between validation and test by 0.65 percentage "
  "points on average (2.27 at most), similar to a random 1:1 split in our simulations. Label co-occurrence "
  "statistics, frequency tiers and positive weights are computed from training labels only. We did not run our own "
  "near-duplicate audit across splits; the dataset authors report removing duplicates [1].")
P("Two shortcut risks are relevant. Emojis and single keywords are strong surface cues, and the error audit in "
  "Section 6 shows the model relying on them. The comment ids encode the source platform, but ids are never model "
  "input. All randomness uses seed 42 for the Midterm model; the two extra seeds of Section 4.4 use 43 and 44.")
P("Finally, the test set was evaluated, and its errors read, during the Midterm. Every Final decision (grid, tie "
  "rule, metrics, tiers) was fixed before the RQ1 test evaluation, but the test set is not untouched, so RQ1 is a "
  "post-hoc comparison on held-out predictions. We return to this in Section 7.1.")

# ============================================================================ 4 Methods
H1("4  Methods and Experimental Protocol")
H2("4.1  Preprocessing Pipeline")
P("All models receive the same normalised text: HTML entities are unescaped, line-break tags become spaces, "
  "Unicode is normalised to NFC and whitespace is collapsed. Emojis, teencode and punctuation are kept. For PhoBERT "
  "two more steps follow: each emoji is replaced by its English name with the ~emoji~ package (PhoBERT's vocabulary "
  "maps every emoji to the unknown token), and the text is word-segmented with VnCoreNLP [13]. Sequences are "
  f"truncated at 128 sub-word tokens, which affects {pct(0.0027, 2)} of training comments; "
  f"{pct(costs[42]['unk_rate'], 2)} of tokens are unknown. No step depends on the label and none is applied only "
  "during training; there is no data augmentation.")
H2("4.2  Models and Training")
P(f"{T('train')} lists the three Midterm models. Hyper-parameters and the number of epochs were chosen on "
  "validation Macro-F1 at τ = 0.5. For PhoBERT we compared five configurations (emoji kept or converted, plain or "
  "square-root-weighted BCE, 6 or 12 epochs); the selected one converts emojis to text and weights the positive "
  "term of label ~k~ by √(~N~_{neg,k} / ~N~_{pos,k}), computed on training labels. PhoBERT uses AdamW [16] with "
  "weight decay 0.01, 10% linear warm-up, mixed precision and early stopping on validation Macro-F1 (patience 3); "
  "PyTorch [17] and Hugging Face Transformers [18] are used throughout.")
table("train", "Training configuration of the Midterm models (selected settings).", [
    ["Model", "Checkpoint / features", "Input size", "Epochs", "Batch", "LR", "Params"],
    ["TF-IDF + LR", "word 1–2-grams, min_df 2, sublinear TF; liblinear, C = 4, balanced", "full text", "–", "–",
     "–", f"{cost.loc['tfidf_lr', 'n_params'] / 1e6:.2f} M"],
    ["BiLSTM", "trained embeddings (200), 128 units per direction, max + mean pooling, dropout 0.3",
     "100 tokens", "≤ 30 (patience 4)", "64", "1e-3", f"{cost.loc['bilstm', 'n_params'] / 1e6:.2f} M"],
    ["PhoBERT-base", "vinai/phobert-base @ 01daacda [12], sigmoid head, √-weighted BCE", "128 tokens",
     "≤ 12 (patience 3)", "16", "2e-5", f"{cost.loc[MAIN, 'n_params'] / 1e6:.1f} M"],
], [2.4, 6.0, 1.8, 2.2, 1.2, 1.2, 1.4], align=["left", "left", "center", "center", "center", "center", "center"],
      size=9.5)
H2("4.3  Evaluation Metrics")
P("Let ~y~_{ik} ∈ {0, 1} be the gold label and ŷ_{ik} the prediction for comment ~i~ and label ~k~, with ~N~ "
  "comments and ~K~ = 28 labels. Per-label F1 is computed from true positives, false positives and false negatives "
  "(an F1 of 0 is used when a label is never predicted nor present). The primary metric is Macro-F1, the "
  "unweighted mean over labels, because it gives a rare label the same weight as a frequent one; this is exactly "
  "where per-label thresholds are expected to help:")
eq_macro = equation(r"\text{Macro-F1}=\frac{1}{K}\sum_{k=1}^{K}\frac{2\,TP_k}{2\,TP_k+FP_k+FN_k}")
P("Micro-F1 pools the counts over labels and so weights frequent labels more. Hamming loss is the share of wrong "
  "label decisions and penalises over-prediction directly:")
equation(r"\text{Micro-F1}=\frac{2\sum_k TP_k}{\sum_k\left(2\,TP_k+FP_k+FN_k\right)}")
equation(r"\text{HL}=\frac{1}{NK}\sum_{i=1}^{N}\sum_{k=1}^{K}\left[\hat{y}_{ik}\neq y_{ik}\right]")
P("We also report per-label F1, Macro-F1 within each frequency tier, metrics by label cardinality and the mean "
  "number of predicted labels per comment. LRAP and mAP, used in the Midterm, depend only on the ranking of "
  "probabilities and are therefore identical for all three decoders.")
H2("4.4  RQ1 Protocol")
P(f"**What is varied and what is held fixed.** The backbone, its weights, the probability matrices, the label set, "
  "the splits and the metrics are fixed; only the threshold vector changes. A label is predicted when its "
  "probability reaches its threshold:")
equation(r"\hat{y}_{ik}=\left[p_{ik}\geq\tau_k\right],\qquad \tau_k=0.5\ \text{(fixed)},\quad \tau_k=\tau^{*}\ "
         r"\text{(global)},\quad \tau_k=\tau_k^{*}\ \text{(per-label)}")
P("Both tuned decoders search the same grid ~G~ = {0.050, 0.075, …, 0.900} (35 values), the grid already used for "
  "the Midterm threshold sweep. The global threshold maximises validation Macro-F1; each per-label threshold "
  "maximises that label's validation F1:")
equation(r"\tau^{*}=\mathrm{arg\,max}_{\tau\in G}\ \text{Macro-F1}_{\mathrm{val}}(\tau),\qquad "
         r"\tau_k^{*}=\mathrm{arg\,max}_{\tau\in G}\ F1_{k,\mathrm{val}}(\tau)")
P("Ties go to the value closest to 0.5 and then to the smaller value; a label without validation positives would "
  "keep 0.5 (none occurs). By construction per-label tuning cannot lower validation Macro-F1 below the global "
  "optimum, so validation scores say nothing about which decoder is better; only test scores do.")
P(f"**Freezing.** {F('protocol')} shows the protocol. Calibration reads only validation probabilities and "
  "training-derived label counts and writes ~calibration.json~ with the thresholds, the grid, the tie rule and "
  "SHA-256 hashes of its inputs. Evaluation refuses to run if any input has changed, loads each probability matrix "
  "once, applies the three frozen threshold vectors and writes the decoded predictions next to the probabilities. "
  "Ten unit tests cover the tie rule, the validation/test boundary and id alignment.")
figure(FIG / "fig_protocol.png", "protocol",
       "RQ1 protocol. Test probabilities are opened only after the thresholds are frozen.")
P("**Samples and runs.** Calibration uses the 2,066 validation comments and evaluation the 2,067 test comments. "
  "The main run is the Midterm model (seed 42). To measure training noise we retrained the identical configuration "
  "with seeds 43 and 44; each seed calibrates its own thresholds on its own validation scores and is evaluated "
  "once. These seeds were trained after the seed-42 results were known and are used only to measure variation, "
  "never to select anything.")
P(f"**Statistics.** For seed 42 we run a paired bootstrap over test comments [19], [20]: 10,000 resamples with "
  "replacement (seed 42), scoring all three decoders on the same resample, and report the observed difference with "
  "a percentile 95% interval. Across seeds we report mean ± standard deviation (ddof = 1) and the paired difference "
  "per seed. To probe threshold overfitting we compare each label's F1 gain on validation with its gain on test "
  "(Spearman ρ) and measure how much each label's threshold moves between seeds.")
H2("4.5  Compute Budget")
P(f"All training ran on one NVIDIA RTX 3060 Laptop GPU (6 GB) under Windows with Python 3.12, PyTorch 2.14.1 + "
  f"CUDA 12.6 and Transformers 5.17.0; threshold calibration, evaluation and all analyses run on the CPU from saved "
  f"probabilities. {T('compute')} lists the cost per run. All GPU training in this project took about "
  f"{gpu_seconds / 3600:.1f} hours.")
table("compute", "Compute cost per run (GPU training time excludes data loading and segmentation).", [
    ["Run", "Hardware", "Epochs run (best)", "Training time", "Peak GPU memory"],
    ["TF-IDF + LR", "CPU", "–", f"{cost.loc['tfidf_lr', 'train_seconds']:.1f} s", "–"],
    ["BiLSTM", "RTX 3060 Laptop", "early-stopped", f"{cost.loc['bilstm', 'train_seconds']:.0f} s", "–"],
    *[[f"PhoBERT-base, seed {s}", "RTX 3060 Laptop", f"{costs[s]['epochs_run']} ({costs[s]['best_epoch']})",
       f"{(cost.loc[MAIN, 'train_seconds'] if s == 42 else costs[s]['train_seconds']) / 60:.1f} min",
       f"{costs[s]['peak_gpu_gb']:.2f} GB"] for s in (42, 43, 44)],
    ["Calibration + evaluation (per seed)", "CPU", "–", "< 10 s", "–"],
    ["Paired bootstrap, 10,000 resamples", "CPU", "–", "≈ 12 s", "–"],
], [5.2, 3.0, 2.8, 2.6, 2.6], size=9.5,
      note="The seed-42 time is the corrected value recorded in the Midterm (the raw wall clock included a laptop sleep).")

# ============================================================================ 5 Results
H1("5  Results")
H2("5.1  Baseline Recap")
brow = [["Model (Midterm, τ = 0.5)", "Val Macro-F1", "Test Macro-F1", "Test Micro-F1", "Hamming loss", "LRAP", "mAP"]]
for run, name in (("tfidf_lr", "TF-IDF + LR"), ("bilstm", "BiLSTM"), ("phobert_keep_bce", "PhoBERT, emoji kept, BCE, 6 ep."),
                  ("phobert_text_bce", "PhoBERT, emoji → text, BCE, 6 ep."),
                  ("phobert_text_bce_ep12", "PhoBERT, emoji → text, BCE, 12 ep."),
                  ("phobert_text_pw_sqrt", "PhoBERT, emoji → text, √-BCE, 6 ep."),
                  (MAIN, "**PhoBERT, emoji → text, √-BCE, 12 ep. (selected)**")):
    r = base.loc[run]
    brow.append([name, f3(r.val_macro_f1), f3(r.test_macro_f1), f3(r.test_micro_f1), f"{r.test_hamming_loss:.4f}",
                 f3(r.test_lrap), f3(r.test_mAP)])
P(f"{T('baseline')} recalls the Midterm results that serve as the control. PhoBERT beats both baselines on every "
  "metric. Among the PhoBERT runs the selected configuration has the best validation Macro-F1; the plain-BCE "
  f"12-epoch run has a better Hamming loss because it predicts fewer labels ({q5.loc[1, 'labels/comment @0.5']:.2f} "
  f"per comment against {q5.loc[0, 'labels/comment @0.5']:.2f}), a first sign that the decision rule, not only the "
  "model, shapes these metrics. In the Midterm threshold sweep (question 5) validation Macro-F1 of the selected "
  f"model stayed between {min(q5.loc[0, f'val Macro-F1 @0.{d}'] for d in range(3, 8)):.3f} and "
  f"{V42.loc['global', 'macro_f1']:.3f} for τ = 0.3–0.7, with its maximum at {q5.loc[0, 'tau*_val']:.3f}.")
table("baseline", "Midterm results at τ = 0.5 (seed 42). Selection used validation Macro-F1 only.", brow,
      [5.6, 1.7, 1.7, 1.7, 1.9, 1.3, 1.3], size=9.5)
H2("5.2  RQ1 Results")
P(f"**Overall.** {T('main')} gives the test results over three seeds. Both tuned decoders beat τ = 0.5 on Macro-F1, "
  "Micro-F1 and Hamming loss, and both cut the mean number of predicted labels from about 2.44 to about 2.2 per "
  f"comment (the gold mean is {Ys['test'].sum(1).mean():.2f}). The selected global thresholds were "
  f"{', '.join(f'{v:.3f}' for v in glob_th)} for seeds 42, 43 and 44: always above 0.5, because the square-root "
  "positive weights push probabilities upwards. Per-label thresholds have the best mean Macro-F1, but the global "
  "threshold has the best Micro-F1 and Hamming loss, and the per-label validation-to-test gap is the only one "
  "clearly above zero.")
mrow = [["Decoder", "Macro-F1", "Micro-F1", "Hamming loss", "Labels / comment", "Val − test gap (Macro-F1)"]]
best = {c: (agg[f"{c}_mean"].idxmin() if c == "test_hamming_loss" else agg[f"{c}_mean"].idxmax())
        for c in ("test_macro_f1", "test_micro_f1", "test_hamming_loss")}
for s in STRAT:
    cells = [SNAME[s]]
    for c in ("test_macro_f1", "test_micro_f1", "test_hamming_loss"):
        txt = pm(*seedmean(s, c))
        cells.append(f"**{txt}**" if best[c] == s else txt)
    cells.append(f"{seedmean(s, 'test_mean_predicted_cardinality')[0]:.2f} ± {seedmean(s, 'test_mean_predicted_cardinality')[1]:.2f}")
    cells.append(f"{s4(seedmean(s, 'gap_val_minus_test_macro_f1')[0])} ± {f4(seedmean(s, 'gap_val_minus_test_macro_f1')[1])}")
    mrow.append(cells)
table("main", "RQ1 comparison on the test set (mean ± std over 3 seeds). Best values in bold.", mrow,
      [4.4, 2.8, 2.8, 2.8, 2.0, 2.8], size=9.5)
P(f"**Are the differences real?** {T('deltas')} pairs the decoders within each seed and adds the paired bootstrap "
  f"for seed 42. Tuning a global threshold improves Macro-F1 in all three seeds "
  f"({s4(sdelta.loc['mean', 'global - fixed'])} ± {f4(sdelta.loc['std', 'global - fixed'])}), and its Hamming-loss "
  f"reduction is certain within seed 42 ({ci('global - fixed', metric='hamming_loss')}), although its Macro-F1 "
  f"interval just includes zero. Per-label versus global is different: the difference changes sign (seed 43 is "
  f"negative), its mean {s4(sdelta.loc['mean', 'per_label - global'])} is smaller than its standard deviation, and "
  f"per-label thresholds win in only {pct(bt('per_label - global').share_resamples_a_better, 0)} of bootstrap "
  "resamples, close to a coin flip.")
drow = [["Comparison (Macro-F1)", "Seed 42", "Seed 43", "Seed 44", "Mean ± std", "Bootstrap 95% CI (seed 42)", "Resamples won"]]
for c in ("global - fixed", "per_label - fixed", "per_label - global"):
    b = bt(c)
    drow.append([c.replace("per_label", "per-label").replace(" - ", " − "), s4(sdelta.loc["42", c]), s4(sdelta.loc["43", c]),
                 s4(sdelta.loc["44", c]), f"{s4(sdelta.loc['mean', c])} ± {f4(sdelta.loc['std', c])}",
                 f"[{s4(b.ci95_low)}, {s4(b.ci95_high)}]", pct(b.share_resamples_a_better, 0)])
table("deltas", "Paired test Macro-F1 differences between decoders.", drow, [3.6, 1.5, 1.5, 1.5, 2.6, 3.4, 1.9], size=9.5)
P(f"**Frequency tiers.** {F('tiers')} breaks Macro-F1 down by tier. The hypothesis expected the largest per-label "
  f"gain on tail labels; the data show the opposite. Per-label thresholds help most on **head** labels "
  f"({tier42('per_label', 'head'):.3f} versus {tier42('global', 'head'):.3f} for global in seed 42; bootstrap "
  f"difference {ci('per_label - global', 'head')}), are slightly worse on **mid** labels, and on **tail** labels "
  f"match the global threshold ({seedmean('per_label', 'test_macro_f1_tail')[0]:.4f} versus "
  f"{seedmean('global', 'test_macro_f1_tail')[0]:.4f} over seeds). Within seed 42, per-label versus 0.5 on the "
  f"tail is {ci('per_label - fixed', 'tail')}: no evidence of a benefit for rare labels. Head labels have the most "
  "validation positives, so their thresholds are estimated most reliably.")
figure(FIG / "fig_tier_f1.png", "tiers", "Test Macro-F1 by frequency tier (mean over 3 seeds; error bars: ±1 std).")
crow = [["|Y|", "n", "Micro-F1: fixed / global / per-label", "Samples-F1: fixed / global / per-label", "Exact set match: fixed / per-label"]]
for g in ("1", "2", "3", ">=4"):
    sub = card[(card.split == "test") & (card.cardinality == g)].set_index("strategy")
    ce = carderr[carderr.cardinality == g].set_index("strategy")
    crow.append(["≥4" if g == ">=4" else g, f"{int(sub.loc['fixed', 'n_samples'])}",
                 " / ".join(f3(sub.loc[s, "micro_f1"]) for s in STRAT),
                 " / ".join(f3(sub.loc[s, "samples_f1"]) for s in STRAT),
                 f"{pct(ce.loc['fixed', 'share_exact_set'])} / {pct(ce.loc['per_label', 'share_exact_set'])}"])
P(f"**Label cardinality.** {T('card')} stratifies seed 42 by the number of gold labels. Tuning helps comments with "
  "one or two labels, where the model over-predicts, and hurts comments with three or more, where it already "
  "under-predicts. For comments with at least four labels the exact-match rate falls from "
  f"{pct(carderr[(carderr.cardinality == '>=4') & (carderr.strategy == 'fixed')].share_exact_set.iloc[0])} at 0.5 to "
  f"{pct(carderr[(carderr.cardinality == '>=4') & (carderr.strategy == 'per_label')].share_exact_set.iloc[0])} with "
  "per-label thresholds (n = 66, so the estimate is rough).")
table("card", "Test results by label cardinality (seed 42).", crow, [1.0, 1.0, 4.6, 4.6, 3.6], size=9.5)
P(f"**Threshold profile and stability.** {F('thresholds')} plots the per-label thresholds of all three seeds. In seed "
  f"42 they range from {th.min():.3f} to {th.max():.3f}: high for grief, gratitude, love and embarrassment, low for "
  "annoyance, disapproval, realization and desire. The profile is not a property of the emotions, however: the "
  f"threshold of a label moves by {stab['range'].mean():.3f} on average between seeds, and by "
  f"{stab.loc['grief', 'range']:.3f} for grief ({', '.join(f'{v:.3f}' for v in stab.loc['grief', ['seed42', 'seed43', 'seed44']])}). "
  f"Only remorse kept the same threshold. The full per-label table is in Appendix D ({T('perlabel')}).")
figure(FIG / "fig_thresholds_seeds.png", "thresholds",
       "Per-label thresholds selected on validation for three training seeds; labels ordered by training frequency "
       "(H/M/T = head/mid/tail). Shaded band: global thresholds of the three seeds.")
grow = [["Decoder (seed 42)", "Val Macro-F1", "Test Macro-F1", "Gap (val − test)", "Gap, 3 seeds (mean ± std)"]]
for s in STRAT:
    m, sd = seedmean(s, "gap_val_minus_test_macro_f1")
    grow.append([SNAME[s], f4(gaps.loc[s, "val_macro_f1"]), f4(gaps.loc[s, "test_macro_f1"]),
                 s4(gaps.loc[s, "gap_val_minus_test_macro_f1"]), f"{s4(m)} ± {f4(sd)}"])
P(f"**Validation-to-test generalisation.** {T('gap')} compares validation and test Macro-F1. Fixed and global "
  "decoding generalise with a gap near zero; per-label decoding loses about 0.016 between validation and test in "
  f"every seed. {F('transfer')} looks at the same effect label by label: by construction 26–27 labels gain on "
  f"validation in each seed, but only {transfer.loc[42, 'labels_test_gain_positive']}, "
  f"{transfer.loc[43, 'labels_test_gain_positive']} and {transfer.loc[44, 'labels_test_gain_positive']} gain on "
  f"test, and the rank correlation between a label's validation gain and its test gain is "
  f"{', '.join(f'{v:.2f}'.replace('-', '−') for v in transfer.spearman_val_gain_vs_test_gain)} (|ρ| ≥ 0.375 would "
  f"be needed for significance with 28 labels). Relief, the rarest label, gains "
  f"{s4(comp.loc['relief', 'val_gain_per_label_vs_fixed'])} on validation and loses "
  f"{m4(comp.loc['relief', 'test_gain_per_label_vs_fixed'])} on test in seed 42.")
table("gap", "Validation-to-test generalisation gap of Macro-F1.", grow, [5.0, 2.4, 2.4, 2.6, 3.6], size=9.5)
figure(FIG / "fig_gain_transfer.png", "transfer",
       "Per-label F1 gain of per-label thresholds over τ = 0.5 on validation versus test (28 labels × 3 seeds).",
       width_cm=10.5)
H2("5.3  Answer to the Research Question")
P("**RQ1: is a single uniform threshold of 0.5 optimal?** No. For this model a single threshold tuned on "
  f"validation ({glob_th.min():.3f}–{glob_th.max():.3f}) is better than 0.5 in every seed "
  f"({s4(sdelta.loc['mean', 'global - fixed'])} Macro-F1) and reliably reduces Hamming loss, because 0.5 lets the "
  "model predict about half a label too many per comment. The evidence for this part is moderate: consistent across "
  "seeds, but within one seed the Macro-F1 interval just touches zero.")
P("**H1 is not supported for per-label thresholds.** They add "
  f"{s4(sdelta.loc['mean', 'per_label - global'])} ± {f4(sdelta.loc['std', 'per_label - global'])} Macro-F1 over a "
  "global threshold, a difference that changes sign across seeds and lies inside the bootstrap interval; they are "
  "worse on Micro-F1 and Hamming loss, overfit validation, do not help tail labels, and select thresholds that are "
  "unstable across retraining. On ViGoEmotions the answer to the imbalance is therefore one better-placed threshold, "
  "not 27 separate ones.")

# ============================================================================ 6 Error analysis
H1("6  Error Analysis")
P("The audit uses the seed-42 test predictions. It focuses on what the tuned thresholds change, because the "
  "Midterm audit already read 70 errors at τ = 0.5.")
H2("6.1  Confusion and Per-Class Errors")
erow = [["Decoder", "TP", "FP", "FN", "FP head / mid / tail", "FN head / mid / tail"]]
for s in STRAT:
    e = errc[errc.strategy == s]
    tt = e.groupby("tier")[["fp", "fn"]].sum()
    erow.append([SNAME[s], f"{e.tp.sum():,}", f"{e.fp.sum():,}", f"{e.fn.sum():,}",
                 " / ".join(f"{tt.loc[t, 'fp']:,}" for t in ("head", "mid", "tail")),
                 " / ".join(f"{tt.loc[t, 'fn']:,}" for t in ("head", "mid", "tail"))])
ft = flip_tot.loc["per_label vs fixed"]
gt = flip_tot.loc["per_label vs global"]
P(f"{T('errors')} counts errors over all 28 labels. Both tuned decoders trade about 350 false positives for about "
  "130 false negatives relative to 0.5, which explains why Hamming loss falls clearly while F1 barely moves. Moving "
  f"from 0.5 to per-label thresholds flips {int(ft[['fn_recovered', 'fn_introduced', 'fp_removed', 'fp_introduced']].sum()):,} "
  f"decisions: {int(ft.fp_removed)} false positives removed and {int(ft.fn_recovered)} misses recovered, against "
  f"{int(ft.fn_introduced)} new misses and {int(ft.fp_introduced)} new false positives. Relative to the global "
  f"threshold, per-label decoding flips {int(gt[['fn_recovered', 'fn_introduced', 'fp_removed', 'fp_introduced']].sum())} "
  f"decisions with a net loss of {abs(int(gt.net_correct_decisions))} correct decisions; Macro-F1 still rises "
  "slightly because the changes fall on low-support labels, which weigh more in a macro average.")
table("errors", "Error counts on the test set (seed 42, all 28 labels).", erow, [4.4, 1.4, 1.4, 1.4, 3.6, 3.6], size=9.5)
P(f"{F('flips')} shows where the decisions change. High thresholds on the positive and bereavement labels remove "
  f"many false positives: love (net {int(pl_vs_fixed.loc['love', 'net_correct_decisions']):+d}), caring "
  f"({int(pl_vs_fixed.loc['caring', 'net_correct_decisions']):+d}) and grief "
  f"({int(pl_vs_fixed.loc['grief', 'net_correct_decisions']):+d}). Low thresholds on the negative cluster mostly add "
  f"false positives: annoyance at {pl_vs_fixed.loc['annoyance', 'threshold_a']:.3f} gains "
  f"{int(pl_vs_fixed.loc['annoyance', 'fn_recovered'])} hits but adds {int(pl_vs_fixed.loc['annoyance', 'fp_introduced'])} "
  f"false positives (net {int(pl_vs_fixed.loc['annoyance', 'net_correct_decisions']):+d}), and disapproval nets "
  f"{int(pl_vs_fixed.loc['disapproval', 'net_correct_decisions']):+d}. In that cluster the model already confuses "
  "neighbouring labels, so lowering a threshold mainly adds a near-synonym.")
figure(FIG / "fig_decision_flips.png", "flips",
       "Test decisions changed per label when moving from τ = 0.5 to per-label thresholds (seed 42).")
prow = [["Gold label missed (i)", "Predicted instead (j)", "n", "φ_{ij} (train)", "", "Correct label (i)",
         "Spurious companion (j)", "n"]]
mis = pairs[(pairs.strategy == "per_label") & (pairs.kind == "missed_i_predicted_j")].head(6).reset_index(drop=True)
spu = pairs[(pairs.strategy == "per_label") & (pairs.kind == "correct_i_spurious_j")].head(6).reset_index(drop=True)
for a, b in zip(mis.itertuples(), spu.itertuples()):
    prow.append([a.label_i, a.label_j, str(a.n_comments), f"{a.phi_train:+.2f}".replace("-", "−"), "", b.label_i,
                 b.label_j, str(b.n_comments)])
sg = pairs[(pairs.kind == "correct_i_spurious_j") & (pairs.label_i == "sadness") & (pairs.label_j == "grief")].set_index("strategy").n_comments
ad = pairs[(pairs.kind == "correct_i_spurious_j") & (pairs.label_i == "annoyance") & (pairs.label_j == "disapproval")].set_index("strategy").n_comments
P(f"{T('pairs')} lists the most frequent confusions under per-label thresholds. Almost all involve the negative "
  "cluster (disgust, disapproval, anger, annoyance, disappointment, sadness). Some confused pairs rarely co-occur "
  "in training (φ < 0), so the model is not simply copying label co-occurrence. Threshold tuning reshapes these "
  f"pairs: the spurious sadness + grief pair falls from {sg['fixed']} comments at 0.5 to {sg['per_label']}, while "
  f"annoyance + spurious disapproval rises from {ad['fixed']} to {ad['per_label']}.")
table("pairs", "Most frequent label confusions under per-label thresholds (seed 42, test).", prow,
      [2.8, 2.6, 1.0, 1.6, 0.3, 2.6, 2.8, 1.0], size=9)
P(f"{F('carderr')} shows the structural error that thresholds cannot remove. At 0.5 the model predicts more labels "
  f"than the gold set for {pct(carderr[(carderr.cardinality == '1') & (carderr.strategy == 'fixed')].share_over.iloc[0], 0)} "
  "of single-label comments and fewer for "
  f"{pct(carderr[(carderr.cardinality == '>=4') & (carderr.strategy == 'fixed')].share_under.iloc[0], 0)} of comments "
  "with four or more labels. Any label-wise threshold moves both errors in the same direction, so tuning lowers "
  "over-prediction at the price of more under-prediction.")
figure(FIG / "fig_cardinality_errors.png", "carderr",
       "Share of test comments with fewer (a) or more (b) predicted labels than gold labels, by gold cardinality (seed 42).")
H2("6.2  Failure Examples")
P("We sampled 80 test cases with seed 42 before reading any text, excluding the 70 comments read in the Midterm: 20 "
  "false negatives and 20 false positives that remain under per-label thresholds (stratified 7/7/6 over head, mid "
  "and tail), 10 misses and 10 false positives introduced by per-label tuning, 5 misses it recovered and 5 false "
  "positives it removed, and 10 comments with at least four gold labels. Each case received one primary and an "
  f"optional secondary cause; {T('causes')} tallies the primary causes. The comments are paraphrased, never quoted, "
  "because the licence forbids redistributing the text.")
group_cols = [("fn_per_label", "FN left"), ("fp_per_label", "FP left"), ("fn_introduced", "FN new"),
              ("fp_introduced", "FP new"), ("multi_label", "|Y|≥4")]
crow2 = [["Primary cause"] + [g[1] for g in group_cols] + ["Fixed by tuning", "Total"]]
for cause, r in causes.iterrows():
    if cause == "total":
        continue
    crow2.append([cause] + [str(int(r.get(g, 0))) for g, _ in group_cols]
                 + [str(int(r.get("fn_recovered", 0) + r.get("fp_removed", 0))), str(int(r["total"]))])
tot = causes.loc["total"]
crow2.append(["**Total**"] + [f"**{int(tot.get(g, 0))}**" for g, _ in group_cols]
             + [f"**{int(tot.get('fn_recovered', 0) + tot.get('fp_removed', 0))}**", f"**{int(tot['total'])}**"])
table("causes", "Primary cause of the 80 audited cases (seed 42, per-label thresholds).", crow2,
      [5.6, 1.4, 1.4, 1.4, 1.4, 1.4, 1.8, 1.4], size=9.5,
      note="Cases in the four tuning groups were selected because a threshold flipped them, so their primary cause is "
           "the threshold; their secondary causes are discussed in the text.")
nd = lambda g, v: int(defens.loc[g, v]) if v in defens.columns else 0
P(f"Four findings stand out. (1) **Every miss introduced by tuning is a real error** ({nd('fn_introduced', 'no')}/10 "
  "not defensible): each comment states the emotion plainly (crying over a lost partner, pity for children, "
  "advice to let go and find peace) and the model scored it 0.51–0.83, enough at 0.5 but below thresholds of "
  f"0.725–0.875. (2) **Half of the new false positives are defensible** ({nd('fp_introduced', 'yes')}/10): they add "
  "annoyance or disgust next to a gold anger or disapproval, labels the annotators themselves apply "
  "inconsistently. (3) **The errors that remain after tuning are the Midterm errors**: near-synonymous labels, "
  "keyword traps, teencode and dialect, and sarcasm read literally. Their scores are far from any threshold "
  "(misses at 0.00–0.16, false positives at 0.70–0.96), so no decoder reaches them. (4) **Multi-emotion comments "
  "lose their secondary emotions**: the model keeps the dominant one (love and caring in a condolence) and drops "
  f"the rest (grief, sadness). {T('examples')} gives representative cases.")
EX = [("2411", "grief", "FN introduced by tuning"), ("tik018661", "annoyance", "FP introduced by tuning"),
      ("1493", "sadness", "FN left after tuning"), ("6107", "remorse", "FP left after tuning"),
      ("18288", "", "|Y| ≥ 4"), ("you001290", "gratitude", "FN left after tuning")]
cases = pd.read_csv(EA / "cases.csv", dtype=str, keep_default_na=False)
PARA = {
    "2411": "A short cry of regret over losing a partner.",
    "tik018661": "A single 'see-no-evil' monkey emoji and nothing else.",
    "1493": "A lament that people do not know what to do as illnesses pile up.",
    "6107": "A bitter, sarcastic agreement about trusting people until they betray you.",
    "18288": "A condolence for a child who died, wishing the child rest.",
    "you001290": "A list of three self-help tips that ends by thanking the teacher.",
}
CAUSE = {
    "2411": "Raised threshold (0.875) cuts an explicit, well-scored grief label.",
    "tik018661": "Nothing passes 0.5; annoyance passes its lowered 0.325 threshold. Emoji cue.",
    "1493": "Keyword trap: the phrase for 'what can one do' triggers confusion; sadness scored 0.28.",
    "6107": "Sarcasm read literally as remorse/sadness; gold (approval, nervousness) is itself doubtful.",
    "18288": "Dominant emotion kept, explicit grief and sadness dropped by every decoder.",
    "you001290": "Signal diluted: the explicit thanks at the end of a long, formatted text scores only 0.08.",
}
xrow = [["Case (id)", "Paraphrase", "Gold labels", "Per-label prediction", "Likely cause"]]
for sid, lab, grp in EX:
    r = cases[(cases.sample_id == sid) & (cases.label == lab)].iloc[0]
    focus = f" ({lab}: p = {float(r.prob):.2f}, τ_{{k}} = {float(r.threshold_per_label):.3f})" if lab else ""
    xrow.append([f"{grp}\n{sid}", PARA[sid], r.true_labels.replace("; ", ", "),
                 (r.pred_per_label.replace("; ", ", ") or "(none)") + focus, CAUSE[sid]])
table("examples", "Representative failure cases (comments paraphrased in English; seed 42).", xrow,
      [2.4, 3.8, 2.6, 3.6, 4.0], align=["left"] * 5, size=9)

# ============================================================================ 7 Limitations, ethics, conclusion
H1("7  Limitations, Ethics and Conclusion")
H2("7.1  Limitations and Threats to Validity")
BUL([
    "**Post-hoc test use.** The test predictions were evaluated and read during the Midterm. The Final grid, tie "
    "rule and metrics were fixed before the RQ1 evaluation and thresholds never saw test labels, but a fresh test set "
    "would be stronger evidence.",
    "**Few seeds, one model.** Three seeds are enough to see that per-label versus global changes sign, not to test "
    "small differences formally. Seeds 43 and 44 were trained after seed 42 was analysed; seed 44 stopped early "
    "and is the weakest model. Results may differ for other backbones (ViSoBERT, CafeBERT) or unweighted BCE, whose "
    "probabilities sit lower and benefit more from tuning (Midterm question 5).",
    "**Small validation set per label.** 58–354 validation positives per label limit any per-label estimate. "
    "Smoothing thresholds towards the global value, or cross-validating them on training folds, might reduce the "
    "overfitting we observe; we did not test these variants.",
    "**Annotation noise.** With an average κ of about 0.56 and inconsistent use of the negative labels, part of the "
    "measured error is label noise; nine of our 80 cases name a doubtful gold label.",
    "**Subjective audit.** The 80 cases were coded by one reader, with the help of an AI assistant (Appendix C), "
    "without a second coder or agreement measure.",
])
H2("7.2  Ethical Considerations")
P("ViGoEmotions is licensed for non-commercial research and education only. We stored no comment text in the "
  "repository, figures or this report and paraphrased every example. One caveat must be stated: the 80 audited "
  "comments, like the 70 of the Midterm audit, were displayed in a session of the AI assistant listed in Appendix C, "
  "so their text was processed by that provider's service; we stored or published none of it there, and future "
  "audits should be read locally only. "
  "Comments are public posts but may still identify their authors or third parties, which is another reason not to "
  "quote them. A deployed emotion classifier can misread sarcasm, dialect and teencode, as our audit shows, so "
  "outputs should not drive decisions about individuals (for example moderation or mental-health flags) without "
  "human review. Lowering thresholds for negative emotions, as per-label tuning did here, would increase false "
  "accusations of anger or disgust.")
H2("7.3  Conclusion")
P("We asked whether one decision threshold of 0.5 is optimal for 28 imbalanced emotion labels in Vietnamese "
  "social-media comments. Holding the best Midterm PhoBERT model fixed and freezing all thresholds on validation, we "
  "found that 0.5 is not optimal: one global threshold around 0.58–0.60 consistently improves Macro-F1 and Hamming "
  "loss by curbing over-prediction. Per-label thresholds, the textbook remedy for imbalance, do not improve on it: "
  "their small gain changes sign across seeds, overfits a validation set with few positives per label and does not "
  "reach the rare labels. The main lesson is that a gain on validation is not evidence by itself; the "
  "remaining errors are about how many emotions a comment carries and which near-synonym it uses, problems for "
  "cardinality-aware decoding or better models rather than for thresholds.")

# ============================================================================ references
H1("References")
REFS = [
    "H. Q. Tran, N. T. Pham, S. T. Luu, and K. V. Nguyen, “ViGoEmotions: A benchmark dataset for fine-grained "
    "emotion detection on Vietnamese texts,” in Proc. 19th Conf. Eur. Chapter Assoc. Comput. Linguistics (EACL), "
    "vol. 1, 2026, pp. 2805–2831, doi: 10.18653/v1/2026.eacl-long.129.",
    "D. Demszky, D. Movshovitz-Attias, J. Ko, A. Cowen, G. Nemade, and S. Ravi, “GoEmotions: A dataset of "
    "fine-grained emotions,” in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2020, pp. 4040–4054, "
    "doi: 10.18653/v1/2020.acl-main.372.",
    "Y. Yang, “A study of thresholding strategies for text categorization,” in Proc. 24th Annu. Int. ACM SIGIR "
    "Conf. Res. Develop. Inf. Retrieval, 2001, pp. 137–145, doi: 10.1145/383952.383975.",
    "R.-E. Fan and C.-J. Lin, “A study on threshold selection for multi-label classification,” Dept. Comput. Sci. "
    "Inf. Eng., National Taiwan Univ., Taipei, Taiwan, Tech. Rep., 2007.",
    "Z. C. Lipton, C. Elkan, and B. Naryanaswamy, “Optimal thresholding of classifiers to maximize F1 measure,” in "
    "Machine Learning and Knowledge Discovery in Databases (ECML PKDD), LNCS, vol. 8725, 2014, pp. 225–239, "
    "doi: 10.1007/978-3-662-44851-9_15.",
    "I. Pillai, G. Fumera, and F. Roli, “Threshold optimisation for multi-label classifiers,” Pattern Recognit., "
    "vol. 46, no. 7, pp. 2055–2065, Jul. 2013, doi: 10.1016/j.patcog.2013.01.012.",
    "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, "
    "pp. 2825–2830, 2011.",
    "S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Comput., vol. 9, no. 8, pp. 1735–1780, "
    "1997, doi: 10.1162/neco.1997.9.8.1735.",
    "A. Graves and J. Schmidhuber, “Framewise phoneme classification with bidirectional LSTM and other neural "
    "network architectures,” Neural Netw., vol. 18, no. 5–6, pp. 602–610, 2005, doi: 10.1016/j.neunet.2005.06.042.",
    "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for "
    "language understanding,” in Proc. NAACL-HLT, Minneapolis, MN, USA, 2019, pp. 4171–4186.",
    "Y. Liu et al., “RoBERTa: A robustly optimized BERT pretraining approach,” arXiv:1907.11692, 2019.",
    "D. Q. Nguyen and A. T. Nguyen, “PhoBERT: Pre-trained language models for Vietnamese,” in Findings Assoc. "
    "Comput. Linguistics: EMNLP 2020, 2020, pp. 1037–1042, doi: 10.18653/v1/2020.findings-emnlp.92.",
    "T. Vu, D. Q. Nguyen, D. Q. Nguyen, M. Dras, and M. Johnson, “VnCoreNLP: A Vietnamese natural language "
    "processing toolkit,” in Proc. NAACL-HLT: Demonstrations, New Orleans, LA, USA, 2018, pp. 56–60, "
    "doi: 10.18653/v1/N18-5012.",
    "R. Alotaibi and P. Flach, “Multi-label thresholding for cost-sensitive classification,” Neurocomputing, "
    "vol. 436, pp. 232–247, May 2021, doi: 10.1016/j.neucom.2020.12.004.",
    "UIT NLP Group, “uitnlp/vigoemotions,” Hugging Face Datasets, 2026. [Online]. Available: "
    "https://huggingface.co/datasets/uitnlp/vigoemotions (accessed Oct. 1, 2026).",
    "I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. Int. Conf. Learn. "
    "Representations (ICLR), 2019.",
    "A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Adv. Neural Inf. "
    "Process. Syst. (NeurIPS), vol. 32, 2019, pp. 8024–8035.",
    "T. Wolf et al., “Transformers: State-of-the-art natural language processing,” in Proc. Conf. Empirical "
    "Methods Natural Lang. Process.: System Demonstrations (EMNLP), 2020, pp. 38–45, "
    "doi: 10.18653/v1/2020.emnlp-demos.6.",
    "B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman & Hall, 1993.",
    "T. Berg-Kirkpatrick, D. Burkett, and D. Klein, “An empirical investigation of statistical significance in "
    "NLP,” in Proc. Joint Conf. Empirical Methods Natural Lang. Process. Comput. Natural Lang. Learn. "
    "(EMNLP-CoNLL), 2012, pp. 995–1005.",
]
for i, ref in enumerate(REFS, 1):
    P(f"[{i}]\t{ref}", "Reference")

# ============================================================================ appendices
H1("Appendix A  Reproducibility", page_break=True)
P("Code: https://github.com/tiendat13ns/Deep-Learning (folder ~ati_final~). The data must be downloaded from "
  "Hugging Face after accepting the dataset terms [15]; it is never redistributed. The commands below, run from "
  "~ati_final~, reproduce every table and figure of this report; they match ~README.md~.")
for line in [
    "# Python 3.12, CUDA GPU for training (RTX 3060 Laptop 6 GB used); Java 8+ for VnCoreNLP",
    "pip install -r requirements.txt",
    "hf auth login                                  # gated dataset uitnlp/vigoemotions",
    f"# Midterm baselines: {T('data')}, {T('baseline')}, {F('dist')} (notebooks keep their outputs)",
    "python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb",
    "python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_tfidf_lr.ipynb",
    "python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_bilstm.ipynb",
    "python -m src.train_phobert --emoji text --loss pw_sqrt --epochs 12 --patience 3 --name phobert_text_pw_sqrt_ep12",
    "python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_phobert.ipynb",
    f"# RQ1 seed 42: calibration and evaluation ({T('card')}, {T('perlabel')})",
    "python -m jupyter nbconvert --to notebook --execute --inplace notebooks/06_thresholds.ipynb",
    f"# seeds 43/44, bootstrap, error audit ({T('main')}, {T('deltas')}, {T('gap')}, {T('errors')}-{T('examples')})",
    "python -m src.train_phobert --emoji text --loss pw_sqrt --epochs 12 --patience 3 --seed 43 --name phobert_text_pw_sqrt_ep12_seed43",
    "python -m src.train_phobert --emoji text --loss pw_sqrt --epochs 12 --patience 3 --seed 44 --name phobert_text_pw_sqrt_ep12_seed44",
    "python -m jupyter nbconvert --to notebook --execute --inplace notebooks/07_rq1_robustness_errors.ipynb",
    "python -m unittest discover -s tests          # 13 tests",
    f"# all figures ({F('dist')}-{F('carderr')}) and this report",
    *([f"python reports/final_report/make_figures.py --style color && python reports/final_report/make_report.py --style color"]
      if STYLE == "color" else
      ["python reports/final_report/make_figures.py && python reports/final_report/make_report.py"]),
    "powershell -File reports/final_report/update_fields.ps1 <report.docx>   # Word: table of contents and captions",
    "# Seeds: 42 (all Midterm runs), 43 and 44 (RQ1 robustness). Split ids: splits/. On Windows set PYTHONIOENCODING=utf-8.",
]:
    P(line, "Code")
H1("Appendix B  Member Contributions")
table("contrib", "Contribution of each member.", [
    ["Student ID", "Full name", "Main tasks", "%"],
    ["2301140022", "Đinh Tiến Đạt*", "Running the notebooks and models: TF-IDF + LR, BiLSTM and PhoBERT training "
                                    "and the Midterm analyses (notebooks 02–05); extra seeds, bootstrap and error "
                                    "audit (notebook 07); Final report writing; submission package", "34"],
    ["2301140051", "Nguyễn Quốc Khanh", "Outputs and evaluation: metrics (src/metrics.py), result tables, saved "
                                       "predictions and figures; RQ1 threshold calibration and frozen evaluation "
                                       "(notebook 06, unit tests); Midterm presentation", "33"],
    ["2301140006", "Nguyễn Duy Anh", "Research: dataset paper and related work; dataset analysis (EDA, notebook 01); "
                                    "Midterm report; Midterm slides", "33"],
    ["", "**Total**", "", "**100**"],
], [2.4, 3.2, 9.2, 1.2], align=["left", "left", "left", "center"], size=10,
      note="* Group representative (submitter). The group agreed on equal contributions; as percentages must be "
           "whole numbers summing to 100, they are recorded as 34 / 33 / 33.")
H1("Appendix C  Use of Generative AI Tools")
table("ai", "Generative AI tools used in this project.", [
    ["Tool", "Purpose", "Sections or files affected"],
    ["Claude Code (Anthropic), Claude Opus models, 2026",
     "Writing and debugging analysis and training code; drafting notebooks; running the RQ1 robustness analysis; "
     "a first coding of the error-audit cases (paraphrases and causes), for which the sampled comment texts were "
     "shown to the assistant; drafting figures and the text of this report.",
     "src/, notebooks/ 05–07, reports/, results/rq1/error_analysis/annotations.csv, all sections of this report"],
], [4.0, 7.0, 5.0], align=["left"] * 3, size=10,
      note="The group remains responsible for every claim, number and citation. All numbers are produced by the "
           "submitted code.")
H1("Appendix D  Per-Label Results", page_break=True)
lrow = [["Label", "Tier", "Train pos.", "Val pos.", "τ^{*}_{k}", "F1 fixed", "F1 global", "F1 per-label", "AP"]]
plm = pd.read_csv(RQ1 / "tables" / "per_label_metrics.csv")
ap = plm[(plm.split == "test") & (plm.strategy == "fixed")].set_index("label").ap
for lab in dist.sort_values("train", ascending=False).index:
    c = comp.loc[lab]
    vals = [c.test_f1_fixed, c.test_f1_global, c.test_f1_per_label]
    bestv = max(vals)
    cells = [f"**{f3(v)}**" if v == bestv else f3(v) for v in vals]
    lrow.append([lab, c.tier, f"{int(c.train_count):,}", str(int(c.val_support)), f3(c.per_label_threshold), *cells,
                 f3(ap[lab])])
table("perlabel", "Per-label test F1 of the three decoders and the per-label thresholds (seed 42; global τ^{*} = "
      f"{calib['global_threshold']:.3f}). Best F1 per label in bold; AP is threshold-free.", lrow,
      [3.0, 1.2, 1.6, 1.4, 1.4, 1.6, 1.6, 1.9, 1.3], size=9)

# ============================================================================ checks and save
assert _next["Table"] == len(TABLE_KEYS) and _next["Figure"] == len(FIG_KEYS), _next
full = "\n".join(text_of(p) for p in body.iter(qn("w:p")))
left = re.findall(r"\[(?!\d+\])(?!\s\])[^\]\n]{0,40}\]", full)
print("bracketed text left:", left)
doc.save(OUT)
print("saved", OUT)
