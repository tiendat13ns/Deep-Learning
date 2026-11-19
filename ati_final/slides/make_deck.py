"""Build the ViGoEmotions midterm deck (10 slides, English) with python-pptx.

Colour tone: white -> lavender (#E6CFF9) gradient background. Numbers come from results/tables;
figures come from slides/assets (run make_figures.py first).
Run from ati_final: python slides/make_deck.py
"""
from pathlib import Path

import pandas as pd
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
OUT = HERE.parent / "ViGoEmotions_midterm_v2.pptx"

INK, DEEP, PRIMARY, MID, LIGHT, TINT, CORAL, GREY, WHITE = (
    "2B2140", "4A2C82", "7C4DCC", "A98BE6", "D9C6F5", "E6CFF9", "E5587F", "7A7392", "FFFFFF")
FONT = "Calibri"
LM, CW = 0.7, 11.93  # left margin, content width (inches)
N_SLIDES = 10
MEMBERS = [("Nguyễn Quốc Khanh", "2301140051"), ("Đinh Tiến Đạt", "2301140022"),
           ("Nguyễn Duy Anh", "2301140006")]


def rgb(h):
    return RGBColor.from_string(h)


class Ctx:
    """Keeps the (shape id, animation group) list of one slide."""

    def __init__(self, slide):
        self.slide, self.items = slide, []

    def reg(self, shape, g):
        if g is not None:
            self.items.append((shape.shape_id, g))
        return shape


def tb(c, x, y, w, h, paras, size=18, color=INK, bold=False, align="l", anchor="t",
       line=1.0, gap=0, g=None):
    s = c.slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(paras, str):
        paras = paras.split("\n")
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        para.line_spacing = line
        if gap and i < len(paras) - 1:
            para.space_after = Pt(gap)
        for r in (p if isinstance(p, list) else [p]):
            text, opt = (r, {}) if isinstance(r, str) else r
            run = para.add_run()
            run.text = text
            f = run.font
            f.name = FONT
            f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold)
            f.italic = opt.get("italic", False)
            f.color.rgb = rgb(opt.get("color", color))
    return c.reg(s, g)


def shape(c, kind, x, y, w, h, fill=None, line=None, lw=1.0, radius=None, g=None):
    s = c.slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.shadow.inherit = False
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(lw)
    else:
        s.line.fill.background()
    if radius is not None and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return c.reg(s, g)


def card(c, x, y, w, h, fill=WHITE, line=LIGHT, g=None, radius=0.05):
    return shape(c, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill, line, 1.0, radius, g)


def pill(c, x, y, w, h, text, fill, color=WHITE, size=16, bold=True, line=None, g=None):
    s = shape(c, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill, line, 1.0, 0.5, g)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.name, r.font.size, r.font.bold = FONT, Pt(size), bold
    r.font.color.rgb = rgb(color)
    return s


def badge(c, x, y, d, text, fill=PRIMARY, g=None):
    s = shape(c, MSO_SHAPE.OVAL, x, y, d, d, fill, g=g)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.name, r.font.size, r.font.bold = FONT, Pt(d * 36), True
    r.font.color.rgb = rgb(WHITE)
    return s


def picture(c, name, x, y, w, h, g=None, center=True):
    path = ASSETS / name
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    px = x + (w - pw) / 2 if center else x
    return c.reg(c.slide.shapes.add_picture(str(path), Inches(px), Inches(y), Inches(pw), Inches(ph)), g)


def new_slide(prs, n, kicker=None, title=None, notes=""):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    fill = s.background.fill
    fill.gradient()
    fill.gradient_angle = 0  # left -> right
    fill.gradient_stops[0].color.rgb = rgb(WHITE)
    fill.gradient_stops[0].position = 0.0
    fill.gradient_stops[1].color.rgb = rgb(TINT)
    fill.gradient_stops[1].position = 1.0
    c = Ctx(s)
    if kicker:
        tb(c, LM, 0.45, 9, 0.3, kicker, size=14, color=PRIMARY, bold=True, g=0)
    if title:
        tb(c, LM, 0.78, CW, 0.8, title, size=36, color=DEEP, bold=True, g=0)
    tb(c, 11.6, 7.0, 1.03, 0.3, f"{n} / {N_SLIDES}", size=11, color=GREY, align="r")
    s.notes_slide.notes_text_frame.text = notes.strip()
    return c


P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def add_motion(c, step=260, dur=520):
    """Fade transition + groups fading in order (automatic, no clicks needed)."""
    items = sorted(c.items, key=lambda t: t[1])
    nid = [4]

    def n():
        nid[0] += 1
        return nid[0]

    effects = []
    for i, (spid, g) in enumerate(items):
        a, b, d = n(), n(), n()
        node = "afterEffect" if i == 0 else "withEffect"
        effects.append(
            f'<p:par><p:cTn id="{a}" presetID="10" presetClass="entr" presetSubtype="0" fill="hold" '
            f'nodeType="{node}"><p:stCondLst><p:cond delay="{g * step}"/></p:stCondLst><p:childTnLst>'
            f'<p:set><p:cBhvr><p:cTn id="{b}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/>'
            f'</p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl><p:attrNameLst>'
            f'<p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr><p:to>'
            f'<p:strVal val="visible"/></p:to></p:set>'
            f'<p:animEffect transition="in" filter="fade"><p:cBhvr><p:cTn id="{d}" dur="{dur}"/>'
            f'<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cBhvr></p:animEffect>'
            f'</p:childTnLst></p:cTn></p:par>')
    timing = (
        f'<p:timing xmlns:p="{P_NS}"><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" '
        f'nodeType="tmRoot"><p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" '
        f'dur="indefinite" nodeType="mainSeq"><p:childTnLst><p:par><p:cTn id="3" fill="hold">'
        f'<p:stCondLst><p:cond delay="indefinite"/><p:cond evt="onBegin" delay="0"><p:tn val="2"/>'
        f'</p:cond></p:stCondLst><p:childTnLst><p:par><p:cTn id="4" fill="hold"><p:stCondLst>'
        f'<p:cond delay="0"/></p:stCondLst><p:childTnLst>{"".join(effects)}</p:childTnLst></p:cTn>'
        f'</p:par></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn><p:prevCondLst>'
        f'<p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
        f'<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond>'
        f'</p:nextCondLst></p:seq></p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>')
    sld = c.slide._element
    sld.append(etree.fromstring(f'<p:transition xmlns:p="{P_NS}" spd="med"><p:fade/></p:transition>'))
    sld.append(etree.fromstring(timing))




# ------------------------------------------------------------------ data tables used on the slides
T = HERE.parent / "results" / "tables"
base = pd.read_csv(T / "baseline_results.csv", index_col=0)
q1 = pd.read_csv(T / "q1_f1_ci.csv", index_col=0)
q2 = pd.read_csv(T / "q2_spearman_ap.csv", index_col=0)
q5 = pd.read_csv(T / "q5_summary.csv", index_col=0)
q3t = pd.read_csv(T / "q3_pair_totals.csv", index_col=0)
q3q = pd.read_csv(T / "q3_pair_quality.csv", index_col=0)
PH = "phobert_text_pw_sqrt_ep12"


def table(c, x, y, w, colw, header, rows, row_h=0.42, font=13, fills=None, bold_rows=(), g=None, head_font=None):
    """Simple styled table. rows: list of lists of str. fills: {row index (0-based, data rows): hex}."""
    n_r, n_c = len(rows) + 1, len(header)
    gf = c.slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w), Inches(row_h * n_r))
    c.reg(gf, g)
    t = gf.table
    t.horz_banding = False
    for j, cw in enumerate(colw):
        t.columns[j].width = Inches(cw)
    for i in range(n_r):
        t.rows[i].height = Inches(row_h)
        for j in range(n_c):
            cell = t.cell(i, j)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.09)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            head = i == 0
            hero = (i - 1) in bold_rows
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(PRIMARY if head else (fills or {}).get(i - 1, DEEP if hero else WHITE))
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = header[j] if head else rows[i - 1][j]
            r.font.name, r.font.size = FONT, Pt((head_font or font) if head else font)
            r.font.bold = head or hero or j == 0
            r.font.color.rgb = rgb(WHITE if (head or (hero and (fills or {}).get(i - 1) is None)) else INK)
    return gf


def qstrip(c, x, y, w, h, tag, text, g=None, size=13):
    """Question strip: the assignment question, verbatim."""
    card(c, x, y, w, h, fill=DEEP, line=DEEP, g=g)
    pill(c, x + 0.18, y + h / 2 - 0.19, 0.62, 0.38, tag, TINT, color=DEEP, size=14, g=g)
    tb(c, x + 0.95, y + 0.05, w - 1.1, h - 0.1, text, size=size, color=WHITE, line=1.0, anchor="m", g=g)


def bullets(c, x, y, w, items, size=13, gap=5, color=INK, dot=PRIMARY, line_h=0.5, g=None):
    """items: list of rich paragraphs; one dot + text box each; line_h = vertical step."""
    for i, it in enumerate(items):
        yy = y + i * line_h
        shape(c, MSO_SHAPE.OVAL, x, yy + 0.055, 0.11, 0.11, dot, g=g)
        tb(c, x + 0.25, yy, w - 0.25, line_h, [it], size=size, color=color, line=1.0, g=g)


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

# ================================================================ 1. Title
c = new_slide(prs, 1, notes="""
[~15 s] Hello everyone. We are Group 3. Our project is fine-grained multi-label emotion recognition in Vietnamese social-media comments, on the ViGoEmotions benchmark. One comment can carry several emotions at once, so every model makes 28 yes/no decisions per comment.
""")
tb(c, LM, 0.95, 9, 0.35, "ATI · PROJECT 2 · GROUP 3 · MIDTERM PRESENTATION", size=15, color=PRIMARY, bold=True, g=0)
tb(c, LM, 1.55, 11.5, 2.4, ["Fine-Grained Multi-Label", "Vietnamese Emotion Recognition"], size=50, color=DEEP, bold=True, line=0.98, g=0)
tb(c, LM, 3.85, 9, 0.5, "ViGoEmotions benchmark (Hung et al., EACL 2026)", size=22, color=INK, g=1)
tb(c, LM, 4.65, 1.9, 0.4, "One comment  →", size=18, color=GREY, anchor="m", g=2)
pill(c, 2.7, 4.6, 1.6, 0.5, "sadness", DEEP, size=17, g=2)
pill(c, 4.45, 4.6, 1.6, 0.5, "anger", PRIMARY, size=17, g=2)
shape(c, MSO_SHAPE.RECTANGLE, LM, 5.75, CW, 0.02, LIGHT, g=3)
tb(c, LM, 5.9, 3, 0.3, "GROUP 3", size=13, color=PRIMARY, bold=True, g=3)
for i, (name, sid) in enumerate(MEMBERS):
    tb(c, LM + i * 3.98, 6.25, 3.8, 0.7,
       [[(name, {"size": 19, "bold": True, "color": DEEP})], [(sid, {"size": 14, "color": GREY})]], g=3)
add_motion(c)

# ================================================================ 2. Data
c = new_slide(prs, 2, "01 · DATA", "The data: what is interesting, what we did", notes="""
[~45 s] ViGoEmotions has twenty thousand six hundred sixty-four comments and twenty-eight labels, with 1.91 labels per comment on average. Three things are interesting. The labels are only moderately imbalanced: amusement is 4.5 times as frequent as relief. Emotions come in clusters like sadness with grief. And it is social-media text: short, a quarter with emoji, one in six with teencode. We used the official split, which is not stratified; we checked the label shares. We checked data quality, defined the frequency tiers and co-occurrence on train only, and never store comment text because of the licence.
""")
tiles = [("20,664", "comments"), ("28", "labels: 27 emotions + neutral"), ("1.91", "labels per comment (max 5)"),
         ("4.5×", "most vs. least frequent label")]
for i, (num, cap) in enumerate(tiles):
    x = LM + i * 3.02
    card(c, x, 1.68, 2.9, 0.98, g=1 + i % 2)
    tb(c, x + 0.2, 1.74, 1.35, 0.85, num, size=30, color=DEEP, bold=True, anchor="m", g=1 + i % 2)
    tb(c, x + 1.5 if i != 1 else x + 0.85, 1.74, 1.35 if i != 1 else 1.95, 0.85, cap, size=12, color=INK, anchor="m", line=1.0, g=1 + i % 2)
card(c, LM, 2.85, 5.85, 4.0, g=2)
tb(c, LM + 0.25, 2.97, 5.4, 0.3, "Share of train comments carrying each label", size=13, color=GREY, bold=True, g=2)
picture(c, "fig_label_frequency.png", LM + 0.2, 3.35, 5.45, 3.4, g=2)
X2, W2 = 6.8, 5.83
card(c, X2, 2.85, W2, 1.9, g=3)
tb(c, X2 + 0.25, 2.95, 5.3, 0.35, "What is interesting", size=17, color=DEEP, bold=True, g=3)
bullets(c, X2 + 0.25, 3.35, W2 - 0.4, [
    [("Moderate imbalance: ", {"bold": True}), "amusement 17.3% vs. relief 3.8% → 10 frequent, 10 mid, 8 rare labels."],
    [("Emotions cluster: ", {"bold": True}), "sadness–grief, fear–nervousness, anger–annoyance."],
    [("Social-media text: ", {"bold": True}), "median 11 words, ~25% emoji, ~16% teencode."]], size=12.5, line_h=0.45, g=3)
card(c, X2, 4.9, W2, 1.95, g=4)
tb(c, X2 + 0.25, 5.0, 5.3, 0.35, "What we did with it", size=17, color=DEEP, bold=True, g=4)
bullets(c, X2 + 0.25, 5.4, W2 - 0.4, [
    [("Split: ", {"bold": True}), "official 16,531 / 2,066 / 2,067 (paper's split; not stratified, we checked label shares)."],
    [("Checks: ", {"bold": True}), "no duplicate ids; 306 comments are neutral + another emotion (reported, not fixed)."],
    [("Train only: ", {"bold": True}), "tiers, co-occurrence and vocabulary; no comment text is stored (licence)."]], size=12.5, line_h=0.45, g=4)
add_motion(c)

# ================================================================ 3. Pipeline
c = new_slide(prs, 3, "02 · PIPELINE", "From text to labels: before and after the model", notes="""
[~45 s] This is the whole pipeline. Every comment first gets the same light cleaning: HTML entities, line-break tags, Unicode normalisation. We keep emoji and teencode on purpose. Then each model has its own input. TF-IDF counts word and symbol n-grams. The BiLSTM builds a vocabulary from train only and pads to one hundred tokens. For PhoBERT we convert emoji to words, because its tokenizer maps every emoji to unknown, then segment words with VnCoreNLP, then tokenize. After the model, we get twenty-eight sigmoid probabilities, which we save before applying any threshold. The decision is probability at least zero point five, and then we compute the metrics.
""")
flow = [("Raw comment", "Vietnamese social-media text"),
        ("Common cleaning", "HTML unescape · <br> → space · Unicode NFC · trim spaces. Emoji, case and teencode are kept.")]
card(c, LM, 1.65, 3.0, 0.95, g=1)
tb(c, LM + 0.2, 1.72, 2.6, 0.35, flow[0][0], size=16, color=DEEP, bold=True, g=1)
tb(c, LM + 0.2, 2.08, 2.6, 0.5, flow[0][1], size=12, color=INK, g=1)
tb(c, 3.8, 1.8, 0.5, 0.6, "→", size=28, color=PRIMARY, bold=True, align="c", g=1)
card(c, 4.35, 1.65, 8.28, 0.95, g=1)
tb(c, 4.55, 1.72, 3.5, 0.35, flow[1][0] + " (all models)", size=16, color=DEEP, bold=True, g=1)
tb(c, 4.55, 2.08, 7.9, 0.5, flow[1][1], size=12.5, color=INK, g=1)
lanes = [("TF-IDF + LR", ["Lower-case; words and symbols (emoji, punctuation) become tokens",
                          "1–2-grams, min. document frequency 2, sub-linear tf",
                          "37,626 features, fitted on train only"]),
         ("BiLSTM", ["Same tokens, lower-cased",
                     "Vocabulary from train (frequency ≥ 2): 5,128 entries; unknown → <unk>",
                     "Cut or pad to 100 tokens; embeddings learned from scratch"]),
         ("PhoBERT", ["Emoji → English words (<unk> rate 2.98% → 0.08%)",
                      "VnCoreNLP word segmentation (what PhoBERT was pretrained on)",
                      "PhoBERT tokenizer, max 128 sub-words"])]
for i, (name, steps) in enumerate(lanes):
    x = LM + i * 4.07
    hero = i == 2
    card(c, x, 2.82, 3.8, 2.38, fill=DEEP if hero else WHITE, line=DEEP if hero else LIGHT, g=2)
    tb(c, x + 0.25, 2.92, 3.3, 0.35, name, size=18, color=WHITE if hero else DEEP, bold=True, g=2)
    for k, st in enumerate(steps):
        yy = 3.38 + k * 0.58
        badge(c, x + 0.25, yy + 0.04, 0.3, str(k + 1), fill=WHITE if hero else PRIMARY, g=2)
        if hero:
            for r in c.slide.shapes[-1].text_frame.paragraphs[0].runs:
                r.font.color.rgb = rgb(DEEP)
        tb(c, x + 0.68, yy, 2.95, 0.58, st, size=11.5, color=WHITE if hero else INK, line=1.0, g=2)
tb(c, LM, 5.33, 4, 0.3, "AFTER THE MODEL", size=12, color=PRIMARY, bold=True, g=3)
after = [("28 probabilities", "one sigmoid per label; saved before any threshold"),
         ("Decision", "label is predicted if p ≥ τ = 0.5"),
         ("Label set", "0 to 28 labels per comment"),
         ("Metrics", "Macro/Micro-F1 · per-label F1 · Hamming · LRAP · mAP")]
for i, (h, body) in enumerate(after):
    x = LM + i * 3.04
    card(c, x, 5.65, 2.75, 1.2, g=3)
    tb(c, x + 0.18, 5.72, 2.45, 0.35, h, size=15, color=DEEP, bold=True, g=3)
    tb(c, x + 0.18, 6.08, 2.45, 0.75, body, size=11.5, color=INK, line=1.0, g=3)
    if i < 3:
        tb(c, x + 2.72, 6.0, 0.35, 0.5, "→", size=20, color=PRIMARY, bold=True, align="c", g=3)
add_motion(c)

# ================================================================ 4. Models
c = new_slide(prs, 4, "03 · MODELS", "Three models: PhoBERT is the best", notes="""
[~45 s] We compare three models. TF-IDF with logistic regression counts words and is our simplest reference. The BiLSTM reads the sentence in order and learns everything from scratch. PhoBERT is pretrained on a lot of Vietnamese and we fine-tune it. On the test set, PhoBERT is best on every metric: Macro-F1 0.608, which is 0.06 above TF-IDF and about 0.10 above the BiLSTM. The cost is 135 million parameters and about twenty minutes of training, against 2.7 seconds for TF-IDF. Surprisingly the BiLSTM is below TF-IDF; our guess is that sixteen thousand comments are too few to learn embeddings from scratch, but we have not tested it.
""")
rows_m = []
for name, run, par, tm in (("TF-IDF + Logistic Regression", "tfidf_lr", "1.05 M", "2.7 s (CPU)"),
                           ("BiLSTM", "bilstm", "1.38 M", "151 s"),
                           ("PhoBERT-base", PH, "135 M", "~20 min")):
    r = base.loc[run]
    rows_m.append([name, f"{r.test_macro_f1:.3f}", f"{r.test_micro_f1:.3f}", f"{r.test_hamming_loss:.3f}",
                   f"{r.test_lrap:.3f}", f"{r.test_mAP:.3f}", par, tm])
table(c, LM, 1.7, CW, [2.7, 1.2, 1.2, 1.5, 1.0, 1.0, 1.4, 1.93],
      ["Model", "Macro-F1", "Micro-F1", "Hamming ↓", "LRAP", "mAP", "Parameters", "Training time"],
      rows_m, row_h=0.55, font=16, bold_rows=(2,), g=1)
tb(c, LM, 4.0, CW, 0.3, "Test set · threshold 0.5 · seed 42 · one run per model · source: results/tables/baseline_results.csv",
   size=11, color=GREY, g=1)
cards = [("TF-IDF + LR", "Counts words and word pairs; 28 independent logistic regressions. 12-configuration grid chosen on validation."),
         ("BiLSTM", "Embedding 200 → BiLSTM (128 per direction) → max + mean pooling → 28 sigmoids. Learns from scratch."),
         ("PhoBERT-base", "vinai/phobert-base fine-tuned; emoji → words; BCE with √-positive weight; up to 12 epochs, early stopping on val.")]
for i, (h, body) in enumerate(cards):
    x = LM + i * 4.07
    hero = i == 2
    card(c, x, 4.45, 3.8, 1.5, fill=DEEP if hero else WHITE, line=DEEP if hero else LIGHT, g=2)
    tb(c, x + 0.22, 4.53, 3.4, 0.35, h, size=16, color=WHITE if hero else DEEP, bold=True, g=2)
    tb(c, x + 0.22, 4.92, 3.4, 1.0, body, size=12, color=WHITE if hero else INK, line=1.0, g=2)
card(c, LM, 6.12, CW, 0.75, fill=TINT, line=TINT, g=3)
tb(c, LM + 0.3, 6.12, CW - 0.6, 0.75,
   [[("PhoBERT wins everywhere: ", {"bold": True, "color": DEEP}),
     "+0.060 Macro-F1 over TF-IDF, +0.097 over BiLSTM, and better on 24 of 28 labels. ",
     ("BiLSTM < TF-IDF ", {"bold": True, "color": DEEP}), "(guess: too little data to learn from scratch; not tested)."]],
   size=13, anchor="m", g=3)
add_motion(c)

# ================================================================ 5. Q1-Q2
c = new_slide(prs, 5, "04 · MIDTERM QUESTIONS 1–2", "Easy and hard labels, and frequency", notes="""
[~60 s] Question one. The easiest labels are embarrassment, gratitude and amusement, at 0.78 to 0.82 F1; the hardest are disapproval, desire and neutral, around 0.41 to 0.46. The intervals are wide, plus or minus about 0.06 to 0.09, so only the extremes are reliable: disapproval is in the bottom five in 98 percent of resamples, but labels in the middle cannot be ranked. Question two. The correlation between F1 and training frequency is about 0.2, not significant; with 28 labels we would need 0.375. AP, which does not depend on the threshold, tells the same story. The frequency range is only 4.5 times, so we can say there is no evidence, not that there is no relationship. Hard labels are hard because their meaning is vague.
""")
LW, RX, RW = 6.1, 7.0, 5.63
qstrip(c, LM, 1.6, LW, 0.85, "Q1", "Which emotion labels exhibit the highest and lowest predictive F1-scores?", g=1, size=14)
hi, lo = ["embarrassment", "gratitude", "amusement"], ["disapproval", "desire", "neutral"]
rows1 = []
for lab in hi + lo:
    r = q1.loc[lab]
    rows1.append([lab, f"{r['F1 PhoBERT']:.2f}", r["CI95 PhoBERT"], f"{r['F1 TF-IDF+LR']:.2f}", f"{r['F1 BiLSTM']:.2f}", str(int(r["số mẫu test"]))])
table(c, LM, 2.6, LW, [1.7, 0.95, 1.2, 0.85, 0.75, 0.65],
      ["Label", "PhoBERT", "95% CI", "TF-IDF", "BiLSTM", "n"], rows1, row_h=0.42, font=13, head_font=12,
      fills={0: "EFE7FB", 1: "EFE7FB", 2: "EFE7FB", 3: "FBE3EA", 4: "FBE3EA", 5: "FBE3EA"}, g=1)
tb(c, LM, 5.6, LW, 1.3,
   [[("Highest: ", {"bold": True, "color": DEEP}), "embarrassment, gratitude, amusement (top 3 for every model)."],
    [("Lowest: ", {"bold": True, "color": CORAL}), "disapproval, desire, neutral (confidence intervals are wide: only the extremes are reliable)."]],
   size=13, line=1.0, gap=5, g=1)
qstrip(c, RX, 1.6, RW, 0.85, "Q2", "How strongly does model performance correlate with training label frequency?", g=2, size=14)
card(c, RX, 2.6, RW, 2.5, g=2)
picture(c, "fig_q2_scatter.png", RX + 0.1, 2.68, RW - 0.2, 2.35, g=2)
rows2 = [[m, f"{r['rho(F1@0.5, tần suất)']:.2f}", f"{r['p (F1)']:.2f}", f"{r['rho(AP, tần suất)']:.2f}", f"{r['p (AP)']:.2f}"]
         for m, r in q2.iterrows()]
table(c, RX, 5.22, RW, [1.65, 1.0, 0.9, 1.1, 0.98], ["Model", "ρ (F1)", "p", "ρ (AP)", "p"], rows2, row_h=0.32, font=11.5, head_font=11, g=2)
tb(c, RX, 6.55, RW, 0.4,
   [[("Weak, not significant ", {"bold": True, "color": DEEP}), "(28 labels need |ρ| ≥ 0.375). Meaning matters more than counts."]],
   size=12, line=1.0, g=2)
add_motion(c)

# ================================================================ 6. Q3
c = new_slide(prs, 6, "05 · MIDTERM QUESTION 3", "Label co-occurrence: preserved, but amplified", notes="""
[~60 s] Question three. In train the most frequent pairs are disappointment with sadness, anger with annoyance, sadness with grief, and amusement with joy. Are these dependencies preserved? At the level of all 378 label pairs, yes: the correlation of PhoBERT's predictions follows the correlation of the true labels, with Spearman 0.90, close to the 0.93 ceiling between train and test. The pairs are not predicted by chance either: for sadness with grief, 152 pairs are predicted against 32 expected if the outputs were independent. But the model amplifies them: 4,741 co-occurring pairs predicted against 2,466 true, so only a quarter of predicted pairs are right. The BiLSTM predicts about the right number, 2,414, but misses more.
""")
qstrip(c, LM, 1.6, CW, 0.75, "Q3", "Which emotion pairs co-occur most frequently, and does the baseline model preserve these dependencies?", g=1, size=14)
card(c, LM, 2.5, 7.0, 3.4, g=2)
tb(c, LM + 0.25, 2.58, 6.5, 0.3, "Co-occurring pairs in test: true, predicted by PhoBERT, expected if independent", size=12, color=GREY, bold=True, g=2)
picture(c, "fig_q3_pairs.png", LM + 0.15, 2.9, 6.7, 2.95, g=2)
card(c, 7.9, 2.5, 4.73, 3.4, g=3)
tb(c, 8.15, 2.58, 4.3, 0.3, "φ of predictions vs. φ of true labels", size=12, color=GREY, bold=True, g=3)
picture(c, "fig_q3_phi_scatter.png", 8.05, 2.9, 4.43, 2.95, g=3)
pt = q3t.loc["PhoBERT"]
tk = [("Most frequent (train)", "disappointment–sadness 904 · anger–annoyance 699 · sadness–grief 620 · amusement–joy 568. Highest φ: fear–nervousness 0.41."),
      ("Preserved", "Spearman ρ = 0.90 between the φ of true and predicted labels (train-vs-test ceiling 0.93); BiLSTM 0.88, TF-IDF 0.84."),
      ("But amplified", f"{int(pt['tổng cặp dự đoán']):,} predicted vs. {int(pt['tổng cặp thật']):,} true pairs; pair precision {pt['precision (378 cặp)']:.2f}, recall {pt['recall (378 cặp)']:.2f}. Not independent: sadness–grief 152 vs. 32 expected.")]
for i, (h, body) in enumerate(tk):
    x = LM + i * 4.07
    card(c, x, 6.02, 3.8, 0.95, g=4)
    tb(c, x + 0.15, 6.05, 3.5, 0.95,
       [[(h + ": ", {"bold": True, "color": DEEP if i < 2 else CORAL}), body]], size=10.5, line=1.0, anchor="m", g=4)
add_motion(c)

# ================================================================ 7. Q4-Q5
c = new_slide(prs, 7, "06 · MIDTERM QUESTIONS 4–5", "Text styles and the 0.5 threshold", notes="""
[~75 s] Question four. PhoBERT is ahead of the BiLSTM in every style, from plus 0.04 to plus 0.13 Micro-F1, although two of the eight intervals include zero. The largest gaps are on long comments and teencode, but both are about the same as for plain text, plus 0.10, so there is no style where PhoBERT gains specifically. The one real difference is the other direction: on emoji comments the gap is smaller, and for emoji-heavy comments it is not certain, perhaps because the BiLSTM sees emoji directly. The stretched-letters group has only 46 comments, so it tells us nothing. Question five. Around the optimum the curve is flat: PhoBERT's Macro-F1 stays within 0.03 between thresholds 0.3 and 0.7, and the best shared threshold, 0.575, gains only 0.004 on test. But 0.5 is still arbitrary: the best threshold is 0.525, 0.35 and 0.575 for the three models, and 0.25 when PhoBERT is trained with plain BCE. At 0.5 PhoBERT predicts 2.4 labels per comment against 1.9 true.
""")
HW = 5.85
qstrip(c, LM, 1.55, HW, 1.05, "Q4", "On which text styles (short comments, emoji-heavy text, teencode) does PhoBERT most significantly outperform the recurrent baseline?", g=1, size=12)
card(c, LM, 2.72, HW, 2.95, g=1)
tb(c, LM + 0.2, 2.78, HW - 0.4, 0.3, "PhoBERT − BiLSTM, Micro-F1 on test (bar = gap, whisker = 95% CI)", size=11, color=GREY, bold=True, g=1)
picture(c, "fig_phobert_vs_bilstm.png", LM + 0.1, 3.1, HW - 0.2, 2.5, g=1)
bullets(c, LM + 0.05, 5.78, HW - 0.1, [
    [("Ahead in every style (+0.04 to +0.13; CI above 0 in 6 of 8), ", {"bold": True, "color": DEEP}), "but long text and teencode do not beat plain text (+0.10): no style-specific gain."],
    [("Emoji-heavy is the exception: ", {"bold": True, "color": DEEP}), "smaller gap (+0.04, CI touches 0). * = differs from plain text."]],
    size=11.5, line_h=0.6, g=1)
RX2 = LM + HW + 0.23
qstrip(c, RX2, 1.55, HW, 1.05, "Q5", "How sensitive are baseline classification outputs to the arbitrary default decision threshold of 0.5?", g=2, size=12)
card(c, RX2, 2.72, HW, 2.95, g=2)
tb(c, RX2 + 0.2, 2.78, HW - 0.4, 0.3, "Macro-F1 vs. one shared threshold (validation)", size=11, color=GREY, bold=True, g=2)
picture(c, "fig_threshold_sweep.png", RX2 + 0.1, 3.1, HW - 0.2, 2.5, g=2)
ph = q5.loc["PhoBERT"]
bullets(c, RX2 + 0.05, 5.78, HW - 0.1, [
    [("Flat near the optimum: ", {"bold": True, "color": DEEP}),
     f"PhoBERT stays within 0.03 for τ = 0.3–0.7; best τ* = {ph['tau*_val']:.3f} gains only +{ph['test gain tau*']:.3f} (test)."],
    [("Yet 0.5 is arbitrary: ", {"bold": True, "color": DEEP}),
     f"τ* = {q5.loc['TF-IDF+LR','tau*_val']:.3f} / {q5.loc['BiLSTM','tau*_val']:.2f} / {ph['tau*_val']:.3f} (TF-IDF / BiLSTM / PhoBERT), "
     f"{q5.loc['PhoBERT (BCE thường)','tau*_val']:.2f} with plain BCE; at 0.5 PhoBERT predicts {ph['labels/comment @0.5']:.1f} labels per comment vs. 1.9 true."]],
    size=11.5, line_h=0.6, g=2)
add_motion(c)

# ================================================================ 8. Error analysis
c = new_slide(prs, 8, "07 · ERROR ANALYSIS", "Error analysis: where does PhoBERT fail?", notes="""
[~45 s] We read 70 test cases by hand: false negatives, false positives, and comments with four or more labels. The most common cause, about a third, is confusing similar labels like anger and annoyance. Next are keyword traps: the word for love makes the model predict love, even when the meaning is different. Sarcasm is read literally, and laughing emoji pull the model towards amusement. Very short comments give too little information. For comments with four or more labels, only one of ten was predicted exactly. Overall PhoBERT predicts too many labels: 2.4 per comment against 1.9 true.
""")
card(c, LM, 1.65, 6.4, 5.2, g=1)
tb(c, LM + 0.28, 1.78, 5.9, 0.3, "Main cause of 69 error cases", size=14, color=GREY, bold=True, g=1)
causes = [("Near-synonym labels", 24), ("Keyword traps", 10), ("Needs context / too short", 8),
          ("Sarcasm, dark humor", 7), ("Emoji, punctuation confusion", 7), ("Teencode, noisy text", 7),
          ("Mixed emotions", 3), ("Near the threshold", 2), ("Doubtful gold label", 1)]
cd = CategoryChartData()
cd.categories = [k for k, _ in causes]
cd.add_series("Cases", [v for _, v in causes])
gf = c.slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(LM + 0.12), Inches(2.1), Inches(6.15), Inches(4.2), cd)
c.reg(gf, 1)
ch = gf.chart
ch.has_legend = False
ch.has_title = False
ch.font.name, ch.font.size = FONT, Pt(13)
ch.font.color.rgb = rgb(INK)
plot = ch.plots[0]
plot.gap_width = 40
plot.has_data_labels = True
dl = plot.data_labels
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size, dl.font.bold = Pt(14), True
dl.font.color.rgb = rgb(DEEP)
ca = ch.category_axis
ca.reverse_order = True
ca.major_tick_mark = XL_TICK_MARK.NONE
ca.format.line.fill.background()
ca.tick_labels.font.size = Pt(13)
va = ch.value_axis
va.visible = False
va.has_major_gridlines = False
va.maximum_scale = 28
va.minimum_scale = 0
for idx in range(len(causes)):
    pt_ = plot.series[0].points[idx]
    pt_.format.fill.solid()
    pt_.format.fill.fore_color.rgb = rgb(DEEP if idx == 0 else PRIMARY if idx < 6 else LIGHT)
tb(c, LM + 0.28, 6.4, 5.9, 0.35, "Method: 70 test cases read by hand (false negatives, false positives, comments with ≥ 4 labels).",
   size=11, color=GREY, line=1.0, g=1)
LX = 7.3
card(c, LX, 1.65, 5.33, 1.45, fill=DEEP, line=DEEP, g=2)
tb(c, LX + 0.3, 1.78, 4.7, 0.3, "LABELS PER COMMENT", size=12, color=LIGHT, bold=True, g=2)
tb(c, LX + 0.3, 2.1, 1.7, 0.9, "2.4", size=48, color=WHITE, bold=True, g=2)
tb(c, LX + 1.95, 2.35, 1.2, 0.6, "predicted", size=16, color=LIGHT, g=2)
tb(c, LX + 3.15, 2.1, 1.0, 0.9, "1.9", size=36, color=TINT, bold=True, g=2)
tb(c, LX + 4.0, 2.35, 1.2, 0.6, "true", size=16, color=LIGHT, g=2)
card(c, LX, 3.25, 5.33, 3.6, g=3)
tb(c, LX + 0.28, 3.35, 4.8, 0.4, "What goes wrong", size=19, color=DEEP, bold=True, g=3)
bullets(c, LX + 0.28, 3.85, 4.85, [
    [("Similar labels: ", {"bold": True}), "anger ↔ annoyance, joy ↔ amusement."],
    [("Keyword traps: ", {"bold": True}), "“yêu” (love) triggers love even when the meaning differs."],
    [("Sarcasm and emoji: ", {"bold": True}), "sarcasm is read literally; laughing emoji pull towards amusement."],
    [("Short or noisy text: ", {"bold": True}), "too little information to decide."],
    [("≥ 4 labels: ", {"bold": True}), "only 1 of 10 predicted exactly; the main emotion is kept, 1–3 extra labels added."]],
    size=12.5, line_h=0.6, g=3)
add_motion(c)

# ================================================================ 9. Final research question
c = new_slide(prs, 9, "08 · PROPOSED RESEARCH QUESTION FOR THE FINAL", "Proposed research question for the Final", notes="""
[~45 s] Our proposed research question for the final: is a single uniform threshold of 0.5 optimal for every emotion label under class imbalance? The midterm gives three reasons to ask: the labels are imbalanced, the best shared threshold differs between models and between training recipes, and the model over-predicts. We will compare three decoding strategies: fixed 0.5, one shared threshold, and one threshold per label. We keep the same PhoBERT with no retraining, tune the thresholds on validation only, freeze them, and run the test once. We will analyze by label frequency and by labels per comment, and check the validation-to-test gap for overfitting. We expect small gains, with per-label thresholds possibly overfitting rare labels.
""")
card(c, LM, 1.65, CW, 1.65, fill=DEEP, line=DEEP, g=1)
pill(c, LM + 0.35, 2.0, 0.95, 0.4, "RQ1", TINT, color=DEEP, size=15, g=1)
tb(c, LM + 1.6, 1.8, CW - 1.95, 0.95,
   "Is a single uniform decision threshold (τ = 0.5) optimal across all emotion labels in the presence of severe class imbalance?",
   size=24, color=WHITE, bold=True, line=1.0, anchor="m", g=1)
tb(c, LM + 1.6, 2.8, CW - 1.95, 0.4,
   "Hypothesis: gains will be small overall, and per-label thresholds may overfit rare labels (only 58–354 validation positives each).",
   size=14, color=TINT, line=1.0, g=1)
card(c, LM, 3.45, 5.85, 2.35, g=2)
tb(c, LM + 0.35, 3.6, 5.2, 0.4, "Why we ask: midterm evidence", size=18, color=DEEP, bold=True, g=2)
why = ["Labels are imbalanced: 3.8% to 17.3% of comments each.",
       "Best shared τ differs: 0.35 to 0.575 across models, 0.25 with plain BCE.",
       "PhoBERT predicts 2.4 labels per comment vs. 1.9 true."]
for i, t in enumerate(why):
    y = 4.15 + i * 0.52
    shape(c, MSO_SHAPE.OVAL, LM + 0.38, y + 0.08, 0.14, 0.14, PRIMARY, g=2)
    tb(c, LM + 0.7, y, 4.9, 0.5, t, size=14, line=1.0, g=2)
card(c, 6.78, 3.45, 5.85, 2.35, g=3)
tb(c, 7.13, 3.6, 5.2, 0.4, "Three strategies to compare", size=18, color=DEEP, bold=True, g=3)
strategies = [("Fixed", "τ = 0.5 for every label"),
              ("Global", "one τ* tuned on val to maximize Macro-F1"),
              ("Per-label", "one τc* per label, each tuned on val")]
for i, (name, desc) in enumerate(strategies):
    y = 4.15 + i * 0.52
    pill(c, 7.13, y, 1.3, 0.38, name, PRIMARY if i else MID, size=13, g=3)
    tb(c, 8.6, y + 0.02, 3.9, 0.5, desc, size=14, line=1.0, anchor="m", g=3)
card(c, LM, 5.95, CW, 0.95, g=4)
tb(c, LM + 0.35, 5.95, 5.4, 0.95,
   [[("Protocol: ", {"bold": True, "color": DEEP}),
     "same PhoBERT, no retraining; tune on val only, freeze, run test once."]],
   size=14, line=1.0, anchor="m", g=4)
tb(c, 6.78, 5.95, 5.6, 0.95,
   [[("Analysis: ", {"bold": True, "color": DEEP}),
     "by label frequency, by labels per comment (1, 2, 3, ≥4), and the val→test gap."]],
   size=14, line=1.0, anchor="m", g=4)
add_motion(c)

# ================================================================ 10. Closing
c = new_slide(prs, 10, notes="""
[~5 s] Thank you for listening. We are happy to answer your questions.
""")
tb(c, LM, 1.9, CW, 1.2, "Thank you for listening", size=60, color=DEEP, bold=True, align="c", g=0)
tb(c, LM, 3.2, CW, 0.6, "Questions and feedback are welcome", size=26, color=INK, align="c", g=1)
shape(c, MSO_SHAPE.RECTANGLE, LM + CW / 2 - 0.6, 4.2, 1.2, 0.05, PRIMARY, g=2)
tb(c, LM, 4.5, CW, 0.35, "GROUP 3", size=14, color=PRIMARY, bold=True, align="c", g=2)
for i, (name, sid) in enumerate(MEMBERS):
    tb(c, LM + i * 3.98, 4.95, 3.8, 0.7,
       [[(name, {"size": 18, "bold": True, "color": DEEP})], [(sid, {"size": 14, "color": GREY})]],
       align="c", g=2)
tb(c, LM, 6.6, CW, 0.4,
   "Sources: ViGoEmotions (Hung et al., EACL 2026) · PhoBERT (vinai/phobert-base) · VnCoreNLP (word segmentation)",
   size=12, color=GREY, align="c", g=3)
add_motion(c)

prs.core_properties.title = "ViGoEmotions - Midterm"
prs.save(OUT)
print("saved", OUT)
