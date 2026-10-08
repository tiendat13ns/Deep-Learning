"""Chỉ số đánh giá multi-label dùng chung cho mọi mô hình (đề: mục C).

Quy ước: Y là ma trận nhị phân (N, C); P là xác suất (N, C) trước khi áp ngưỡng.
Ngưỡng có thể là số (một ngưỡng chung) hoặc vector độ dài C (ngưỡng từng nhãn).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, hamming_loss,
                             label_ranking_average_precision_score)

from .data import LABELS


def binarize(P, thr=0.5) -> np.ndarray:
    return (np.asarray(P) >= np.asarray(thr)).astype(np.int8)


def per_label_f1(Y, Yhat) -> np.ndarray:
    return f1_score(Y, Yhat, average=None, zero_division=0)


def compute_metrics(Y, P, thr=0.5) -> dict:
    """Các chỉ số tổng hợp. LRAP/mAP tính trên xác suất nên không phụ thuộc ngưỡng."""
    Y = np.asarray(Y)
    Yhat = binarize(P, thr)
    return {
        "macro_f1": f1_score(Y, Yhat, average="macro", zero_division=0),
        "micro_f1": f1_score(Y, Yhat, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(Y, Yhat),
        "lrap": label_ranking_average_precision_score(Y, P),
        "mAP": average_precision_score(Y, P, average="macro"),
        "subset_acc": float((Y == Yhat).all(axis=1).mean()),  # chỉ để tham khảo, không phải chỉ số chính
    }


def per_label_table(Y, P, thr=0.5, train_counts=None, tiers=None) -> pd.DataFrame:
    """F1, precision/recall, AP từng nhãn; kèm tần suất train và nhóm head/mid/tail nếu có."""
    from sklearn.metrics import precision_score, recall_score

    Y = np.asarray(Y)
    Yhat = binarize(P, thr)
    df = pd.DataFrame({
        "f1": per_label_f1(Y, Yhat),
        "precision": precision_score(Y, Yhat, average=None, zero_division=0),
        "recall": recall_score(Y, Yhat, average=None, zero_division=0),
        "ap": average_precision_score(Y, P, average=None),
        "support": Y.sum(0),
    }, index=LABELS)
    if train_counts is not None:
        df["train_count"] = np.asarray(train_counts)
    if tiers is not None:
        df["tier"] = [tiers[l] for l in LABELS]
    return df


def f1_by_tier(Y, P, tiers, thr=0.5) -> dict:
    """Macro-F1 trung bình theo nhóm head/mid/tail."""
    f1 = per_label_f1(Y, binarize(P, thr))
    out = {}
    for t in ("head", "mid", "tail"):
        idx = [i for i, l in enumerate(LABELS) if tiers[l] == t]
        out[f"macro_f1_{t}"] = float(f1[idx].mean())
    return out


def save_preds(path, ids, Y, P) -> None:
    """Lưu xác suất liên tục (trước khi nhị phân hoá) để dùng lại ở cuối kì."""
    np.savez_compressed(path, ids=np.asarray(ids), y=np.asarray(Y, dtype=np.int8),
                        p=np.asarray(P, dtype=np.float32))


def load_preds(path):
    z = np.load(path, allow_pickle=True)
    return z["ids"], z["y"], z["p"]
