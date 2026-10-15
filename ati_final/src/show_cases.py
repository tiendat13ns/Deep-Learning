"""In ra màn hình các ca lỗi đã chọn (kèm văn bản gốc) để đọc cục bộ. KHÔNG ghi văn bản ra file.

    python -m src.show_cases --group fn_random --n 20
    python -m src.show_cases --group fp_extreme
    python -m src.show_cases --group multi_label

Nhóm hợp lệ: xem cột `group` trong results/error_cases/cases.csv.
Văn bản gốc chỉ để đọc và diễn đạt lại trong ghi chú; không dán nguyên văn vào repo (license dataset).
"""
import argparse
from pathlib import Path

import pandas as pd

from .data import load_split

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--n", type=int, default=0, help="chỉ in n ca đầu (0 = tất cả)")
    a = ap.parse_args()
    # đọc mọi cột là chuỗi và không đổi ô rỗng thành NaN (ca không dự đoán nhãn nào có pred_labels rỗng)
    cases = pd.read_csv(ROOT / "results" / "error_cases" / "cases.csv", dtype=str, keep_default_na=False)
    sub = cases[cases.group == a.group]
    if sub.empty:
        raise SystemExit(f"Không có nhóm '{a.group}'. Các nhóm: {sorted(cases.group.unique())}")
    if a.n:
        sub = sub.head(a.n)
    text = load_split("test").set_index("id").text
    for k, r in enumerate(sub.itertuples(), 1):
        focus = f"{r.label} (p={float(r.prob):.2f})" if r.label else "-"  # nhóm multi_label không có nhãn đang xét
        print(f"[{k}] id={r.sample_id} | nhãn đang xét: {focus}")
        print(f"    thật : {r.true_labels}")
        print(f"    đoán : {r.pred_labels or '(không dự đoán nhãn nào)'}")
        print(f"    text : {text[r.sample_id]}\n")


if __name__ == "__main__":
    main()
