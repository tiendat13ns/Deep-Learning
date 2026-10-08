"""Nạp ViGoEmotions, chuyển nhãn sang multi-hot. Dùng chung cho mọi notebook/mô hình.

Chia dữ liệu: dùng split chính thức của dataset (train/val/test = 16531/2066/2067).
Văn bản gốc KHÔNG được lưu vào repo (xem license) - chỉ lưu id trong splits/.
"""
import ast
import os

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SEED = 42
REPO_ID = "uitnlp/vigoemotions"
# Theo README của dataset: 27 cảm xúc + neutral (index 27)
LABELS = [
    "amusement", "excitement", "joy", "love", "desire", "optimism", "caring",
    "pride", "admiration", "gratitude", "relief", "approval", "realization",
    "surprise", "curiosity", "confusion", "fear", "nervousness", "remorse",
    "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger",
    "annoyance", "disapproval", "neutral",
]
NUM_LABELS = len(LABELS)
SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}


def _to_multihot(label_str: str) -> np.ndarray:
    y = np.zeros(NUM_LABELS, dtype=np.int8)
    y[ast.literal_eval(label_str)] = 1
    return y


def load_split(split: str) -> pd.DataFrame:
    """Trả về DataFrame cột: id, text, labels (list[int]), y (multi-hot), n_labels."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO_ID, SPLIT_FILES[split], repo_type="dataset")
    df = pd.read_csv(path)
    df["labels"] = df["labels"].apply(ast.literal_eval)
    df["y"] = df["labels"].apply(lambda l: np.isin(np.arange(NUM_LABELS), l).astype(np.int8))
    df["n_labels"] = df["labels"].apply(len)
    return df


def load_all() -> dict:
    return {s: load_split(s) for s in SPLIT_FILES}


def label_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["y"].values)


# Phân nhóm tần suất, định nghĩa TRƯỚC khi đánh giá test, dựa trên số mẫu dương ở tập train.
HEAD_MIN = 1000   # head: >= 1000 mẫu dương trong train
TAIL_MAX = 750    # tail: < 750; mid: [750, 1000)


def label_tiers(train_counts) -> dict:
    """Trả về {tên nhãn: 'head'|'mid'|'tail'}."""
    return {
        name: "head" if n >= HEAD_MIN else ("tail" if n < TAIL_MAX else "mid")
        for name, n in zip(LABELS, train_counts)
    }
