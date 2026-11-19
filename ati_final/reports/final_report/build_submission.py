"""Assemble the Final submission folder (no zip) as ATI_Submission_Guide.pdf describes.

Run from ati_final:  python reports/final_report/build_submission.py
Writes ../submission/62FIT4ATI-CLC02_G03_P02/ (outside ati_final, git-ignored) with exactly:
    slides/<PREFIX>_midterm_slides.pdf
    code/   README, requirements, notebooks, src, tests, configs, splits, results/{tables, figures,
            predictions, preds, rq1, error_cases, environment.md}, report scripts
    report/<PREFIX>_report.docx   (the colour-figure version)
Then checks: required files present, no cache/checkpoint/data/secret files, total size, and that no
comment text from the local dataset copy appears in any text file (needs data/*.csv locally).
"""
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # ati_final
PREFIX = "62FIT4ATI-CLC02_G03_P02"
DEST = ROOT.parent / "submission" / PREFIX
REPORT = ROOT / "reports" / "final_report" / f"{PREFIX}_report_color.docx"
SLIDES = ROOT / "slides" / f"{PREFIX}_midterm_slides.pdf"

CODE = {  # destination (relative to code/) -> source (relative to ati_final), files or folders
    "README.md": "README.md",
    "requirements.txt": "requirements.txt",
    "requirements-analysis.txt": "requirements-analysis.txt",
    "notebooks": "notebooks",
    "src": "src",
    "tests": "tests",
    "configs": "configs",
    "splits": "splits",
    "results/tables": "results/tables",
    "results/figures": "results/figures",
    "results/figures/midterm_slides": "slides/assets",
    "results/predictions": "results/predictions",
    "results/preds": "results/preds",
    "results/rq1": "results/rq1",
    "results/error_cases": "results/error_cases",
    "results/environment.md": "results/environment.md",
    "reports/final_report/make_figures.py": "reports/final_report/make_figures.py",
    "reports/final_report/make_report.py": "reports/final_report/make_report.py",
    "reports/final_report/update_fields.ps1": "reports/final_report/update_fields.ps1",
    "reports/final_report/build_submission.py": "reports/final_report/build_submission.py",
    "reports/rq1_threshold_comparison.md": "reports/rq1_threshold_comparison.md",
    "reports/rq1_final_analysis.md": "reports/rq1_final_analysis.md",
}
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".git", ".venv", "venv", "env", "data"}
FORBIDDEN_EXT = {".pt", ".pth", ".ckpt", ".safetensors", ".h5", ".pyc", ".env", ".bak"}
TEXT_EXT = {".ipynb", ".md", ".py", ".json", ".txt", ".csv", ".ps1"}


def ignore(_dir, names):
    return [n for n in names if n in SKIP_DIRS or Path(n).suffix in FORBIDDEN_EXT or n.startswith("~$")]


def build():
    if DEST.exists():
        shutil.rmtree(DEST)
    (DEST / "slides").mkdir(parents=True)
    (DEST / "report").mkdir()
    shutil.copy2(SLIDES, DEST / "slides" / f"{PREFIX}_midterm_slides.pdf")
    shutil.copy2(REPORT, DEST / "report" / f"{PREFIX}_report.docx")
    for dst, src in CODE.items():
        s, d = ROOT / src, DEST / "code" / dst
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d, ignore=ignore, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    # notebooks: only the seven project notebooks
    for nb in (DEST / "code" / "notebooks").glob("*"):
        if not re.match(r"0[1-7]_.*\.ipynb$", nb.name):
            nb.unlink() if nb.is_file() else shutil.rmtree(nb)


def notebook_text(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        out.append("".join(c.get("source", [])))
        for o in c.get("outputs", []):
            out.append("".join(o.get("text", [])))
            for v in o.get("data", {}).values():
                if isinstance(v, list):
                    out.append("".join(v))
                elif isinstance(v, str) and len(v) < 2_000_000 and not v.startswith("iVBOR"):
                    out.append(v)
    return "\n".join(out)


def check():
    problems = []
    files = [p for p in DEST.rglob("*") if p.is_file()]
    must = [DEST / "slides" / f"{PREFIX}_midterm_slides.pdf", DEST / "report" / f"{PREFIX}_report.docx",
            DEST / "code" / "README.md", DEST / "code" / "requirements.txt", DEST / "code" / "results" / "environment.md"]
    must += [DEST / "code" / d for d in ("notebooks", "splits", "configs", "results/tables", "results/figures",
                                         "results/predictions")]
    problems += [f"missing: {m.relative_to(DEST)}" for m in must if not m.exists()]
    top = sorted(p.name for p in DEST.iterdir())
    if top != ["code", "report", "slides"]:
        problems.append(f"top level must be code/, report/, slides/: {top}")
    for p in files:
        rel = p.relative_to(DEST)
        if p.suffix in FORBIDDEN_EXT or set(rel.parts) & SKIP_DIRS:
            problems.append(f"forbidden file: {rel}")
        if re.search(r"(hf_[A-Za-z0-9]{30,}|HF_TOKEN\s*=\s*['\"]hf_)", p.read_text(encoding="utf-8", errors="ignore")
                     if p.suffix in TEXT_EXT else ""):
            problems.append(f"possible token in {rel}")
    notebooks = sorted(p.name for p in (DEST / "code" / "notebooks").glob("*.ipynb"))
    for nb in notebooks:
        cells = json.loads((DEST / "code" / "notebooks" / nb).read_text(encoding="utf-8"))["cells"]
        code = [c for c in cells if c["cell_type"] == "code"]
        if sum(bool(c.get("outputs")) for c in code) < len(code) // 2:
            problems.append(f"notebook without outputs: {nb}")
    # comment-text leak check: 30-character windows of every long comment, and every short comment
    # (12-29 characters) verbatim, against every text file
    shingles, short = set(), set()
    for split in ("train", "val", "test"):
        f = ROOT / "data" / f"{split}.csv"
        if not f.exists():
            problems.append(f"leak check skipped: {f} not found")
            continue
        import pandas as pd
        for t in pd.read_csv(f).text.astype(str):
            t = " ".join(t.split())
            if len(t) >= 30:
                shingles.update({t[:30], t[len(t) // 2 - 15: len(t) // 2 + 15], t[-30:]})
            elif len(t) >= 12:
                short.add(t)
    scanned, leaks = 0, []
    for p in files:
        if p.suffix not in TEXT_EXT or "predictions" in p.parts or "preds" in p.parts:
            continue  # prediction files hold ids and numbers only
        text = notebook_text(p) if p.suffix == ".ipynb" else p.read_text(encoding="utf-8", errors="ignore")
        text = " ".join(text.split())
        scanned += 1
        if any(text[i:i + 30] in shingles for i in range(len(text) - 29)) or any(t in text for t in short):
            leaks.append(str(p.relative_to(DEST)))
    problems += [f"comment text found in {l}" for l in leaks]
    size = sum(p.stat().st_size for p in files)
    return problems, files, size, scanned, notebooks


if __name__ == "__main__":
    build()
    problems, files, size, scanned, notebooks = check()
    print(f"{DEST}\n{len(files)} files, {size / 1e6:.1f} MB (guide: < 50 MB); notebooks: {', '.join(notebooks)}")
    print(f"leak check: {scanned} text files scanned against the local dataset")
    for top in ("slides", "report", "code"):
        for p in sorted((DEST / top).iterdir()):
            n = sum(1 for q in p.rglob("*") if q.is_file()) if p.is_dir() else 1
            mb = (sum(q.stat().st_size for q in p.rglob("*") if q.is_file()) if p.is_dir() else p.stat().st_size) / 1e6
            print(f"  {top}/{p.name}{'/' if p.is_dir() else ''}  ({n} files, {mb:.1f} MB)")
    print("PROBLEMS:" if problems else "all checks passed", *problems, sep="\n  ")
