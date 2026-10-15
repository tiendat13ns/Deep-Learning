"""Cờ kiểu văn bản mạng xã hội (heuristic regex) dùng cho phân tích theo nhóm câu.

Định nghĩa trùng với notebooks/01_eda.ipynb. Đây là ước lượng, không phải nhãn chuẩn:
danh sách teencode tự lập, có thể vừa bỏ sót vừa đếm dư.
"""
import re

import pandas as pd

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF⭐❤]")
ELONG = re.compile(r"(.)\1{2,}")  # ký tự lặp >= 3 lần ("caoooo", "hahaha")
TEEN = {'ko', 'k', 'kh', 'hok', 'hong', 'dc', 'đc', 'dk', 'vs', 'vl', 'vcl', 'cx', 'cg', 'bt', 'bik', 'biet', 'wa', 'r', 'rùi',
        'ntn', 'mn', 'ae', 'ok', 'oke', 'j', 'z', 'zậy', 'trc', 'sg', 'nc', 'nhìu', 'nhiu', 'thik', 'iu', 'cmt', 'sv', 'hnay',
        'mik', 'mìk'}
SHORT_MAX = 5   # câu ngắn: <= 5 token (tách theo khoảng trắng)
LONG_MIN = 30   # câu dài: >= 30 token


def text_flags(texts) -> pd.DataFrame:
    """Trả về DataFrame theo thứ tự `texts`: n_tok, n_emoji, emoji, elong, teencode, short, long."""
    rows = []
    for t in texts:
        t = str(t)
        toks = t.lower().split()
        n_tok = len(toks)
        rows.append({
            "n_tok": n_tok,
            "n_emoji": len(EMOJI.findall(t)),
            "elong": bool(ELONG.search(t)),
            "teencode": any(w in TEEN for w in toks),
        })
    df = pd.DataFrame(rows)
    df["emoji"] = df.n_emoji > 0
    df["short"] = df.n_tok <= SHORT_MAX
    df["long"] = df.n_tok >= LONG_MIN
    return df
