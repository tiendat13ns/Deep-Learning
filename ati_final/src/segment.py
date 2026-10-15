"""Tách từ tiếng Việt bằng VnCoreNLP (RDRSegmenter) - đúng bộ tách từ mà PhoBERT được huấn luyện.

Cần Java (>= 8). Jar + model được tải về data/vncorenlp/ (không commit).
Kết quả tách từ của từng split được cache ở data/seg/<split>_<tag>.txt (tag = biến thể tiền xử lý, ví dụ keep / emojitext) (mỗi dòng ứng với một mẫu theo thứ tự split).
"""
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VN_DIR = ROOT / "data" / "vncorenlp"
SEG_DIR = ROOT / "data" / "seg"
_BASE = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master"
_FILES = ["VnCoreNLP-1.2.jar", "models/wordsegmenter/vi-vocab", "models/wordsegmenter/wordsegmenter.rdr"]
_SENT = "ZZENDZZ"  # dòng đánh dấu hết mẫu: một dòng có thể bị tách thành nhiều câu


def ensure_vncorenlp() -> Path:
    for f in _FILES:
        p = VN_DIR / f
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{_BASE}/{f}", p)
    return VN_DIR / "VnCoreNLP-1.2.jar"


def segment_texts(texts) -> list:
    """Trả về danh sách chuỗi đã tách từ (từ ghép nối bằng '_'), cùng thứ tự và độ dài với `texts`."""
    jar = ensure_vncorenlp()
    texts = [" ".join(str(t).split()) or "." for t in texts]  # không để dòng rỗng / chứa xuống dòng
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = Path(tmp) / "in.txt", Path(tmp) / "out.txt"
        fin.write_text("\n".join(f"{t}\n{_SENT}" for t in texts) + "\n", encoding="utf-8")
        subprocess.run(["java", "-Xmx2g", "-jar", str(jar), "-fin", str(fin), "-fout", str(fout), "-annotators", "wseg"],
                       cwd=VN_DIR, check=True, capture_output=True)
        blocks = fout.read_text(encoding="utf-8").strip().split("\n\n")
    out, cur = [], []
    for b in blocks:
        toks = [ln.split("\t")[1] for ln in b.splitlines() if ln.strip()]
        if toks == [_SENT]:
            out.append(" ".join(cur)); cur = []
        else:
            cur.extend(toks)
    assert len(out) == len(texts), f"lệch số mẫu: {len(out)} != {len(texts)}"
    return out


def segmented_split(split: str, texts, tag: str) -> list:
    """Tách từ có cache theo split và biến thể tiền xử lý `tag`."""
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    path = SEG_DIR / f"{split}_{tag}.txt"
    if path.exists():
        lines = path.read_text(encoding="utf-8").split("\n")[:-1]
        if len(lines) == len(texts):
            return lines
    seg = segment_texts(texts)
    path.write_text("\n".join(seg) + "\n", encoding="utf-8")
    return seg
