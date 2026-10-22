"""RQ1: compare fixed, global and per-label decoding of saved PhoBERT scores.

Run from ati_final: python -m src.analyze_thresholds
No raw comment text, Hugging Face login, GPU or model checkpoint is needed.
Calibration is a separate phase and writes its frozen specification before any
test probability file is opened. Existing Midterm artifacts are left intact.
"""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

from .data import HEAD_MIN, LABELS, SEED, TAIL_MAX, label_tiers
from .metrics import compute_metrics, load_preds, per_label_table
from .thresholds import DEFAULT_GRID, TIE_RULE, calibrate_thresholds, validate_predictions

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = "phobert_text_pw_sqrt_ep12"
STRATEGIES = ("fixed", "global", "per_label")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _check_ids(root, split, ids):
    expected = pd.read_csv(root / "splits" / f"{split}.csv", dtype=str).sample_id.to_numpy()
    if not np.array_equal(np.asarray(ids, dtype=str), expected):
        raise ValueError(f"{split}: prediction IDs differ from the fixed split or its row order")
    if len(set(expected)) != len(expected):
        raise ValueError(f"{split}: duplicate sample IDs")
    return expected


def calibrate(root=ROOT, output_dir=None, run=DEFAULT_RUN):
    """Use only validation scores and train-derived counts; never read test scores."""
    root = Path(root)
    output_dir = Path(output_dir) if output_dir is not None else root / "results" / "rq1"
    if Path(run).name != run or not run.startswith("phobert_"):
        raise ValueError("run must be the name of an existing PhoBERT prediction run")
    val_path = root / "results" / "preds" / f"{run}_val.npz"
    counts_path = root / "results" / "tables" / "label_distribution.csv"
    tiers_path = root / "splits" / "label_tiers.json"
    ids, y, p = load_preds(val_path)
    y, p = validate_predictions(y, p)
    if y.shape[1] != len(LABELS):
        raise ValueError("Prediction columns do not match the documented 28-label order")
    _check_ids(root, "val", ids)
    counts = pd.read_csv(counts_path, index_col=0)["train"].reindex(LABELS)
    if counts.isna().any() or (counts <= 0).any():
        raise ValueError("Missing or invalid train label frequencies")
    tiers = label_tiers(counts.to_numpy())
    tier_spec = json.loads(tiers_path.read_text(encoding="utf-8"))
    if (tier_spec["labels"] != LABELS or tier_spec["tiers"] != tiers
            or tier_spec["head_min"] != HEAD_MIN or tier_spec["tail_max"] != TAIL_MAX):
        raise ValueError("Train frequency tiers disagree with the fixed split metadata")
    fitted = calibrate_thresholds(y, p)
    spec = {
        "schema_version": 1,
        "run": run,
        "model_selection": "Reuse the Midterm neural model selected by validation Macro-F1 at 0.5",
        "labels": LABELS,
        "seed": SEED,
        "backbone_retrained": False,
        "calibration_split": "val",
        "objective_global": "maximum validation Macro-F1 on the declared grid",
        "objective_per_label": "maximum validation F1 for each label on the same grid",
        "grid": fitted["grid"].tolist(),
        "tie_rule": TIE_RULE,
        "fixed_threshold": 0.5,
        "global_threshold": fitted["global_threshold"],
        "per_label_thresholds": fitted["per_label_thresholds"].tolist(),
        "validation_samples": len(y),
        "validation_positive_counts": y.sum(axis=0).astype(int).tolist(),
        "train_positive_counts": counts.astype(int).tolist(),
        "tiers": tiers,
        "head_min": HEAD_MIN,
        "tail_max": TAIL_MAX,
        "source_hashes": {
            "validation_predictions": sha256(val_path),
            "train_distribution_table": sha256(counts_path),
            "tier_specification": sha256(tiers_path),
            "validation_split_ids": sha256(root / "splits" / "val.csv"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    # This is the freeze point. Test scores have not been opened.
    _write_json(output_dir / "calibration.json", spec)
    sweep = pd.DataFrame(fitted["per_label_f1_grid"], columns=LABELS)
    sweep.insert(0, "threshold", fitted["grid"])
    sweep.insert(1, "macro_f1", fitted["global_macro_f1_grid"])
    sweep.to_csv(output_dir / "tables" / "validation_sweep.csv", index=False)
    return spec


def _decision_metrics(y, predicted):
    return {
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, predicted, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y, predicted)),
        "micro_precision": float(precision_score(y, predicted, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y, predicted, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y, predicted, average="samples", zero_division=0)),
        "mean_predicted_cardinality": float(predicted.sum(axis=1).mean()),
        "mean_true_cardinality": float(y.sum(axis=1).mean()),
        "false_positives": int(((y == 0) & (predicted == 1)).sum()),
        "false_negatives": int(((y == 1) & (predicted == 0)).sum()),
    }


def evaluate(root=ROOT, output_dir=None, report_date=None):
    """Evaluate frozen thresholds. This function never fits or changes thresholds."""
    root = Path(root)
    output_dir = Path(output_dir) if output_dir is not None else root / "results" / "rq1"
    calibration_path = output_dir / "calibration.json"
    spec = json.loads(calibration_path.read_text(encoding="utf-8"))
    frozen_hash = sha256(calibration_path)
    run = spec["run"]
    if spec["schema_version"] != 1 or spec["labels"] != LABELS or spec["calibration_split"] != "val":
        raise ValueError("Unrecognized or incompatible frozen threshold specification")
    val_path = root / "results" / "preds" / f"{run}_val.npz"
    checks = {
        "validation_predictions": val_path,
        "train_distribution_table": root / "results" / "tables" / "label_distribution.csv",
        "tier_specification": root / "splits" / "label_tiers.json",
        "validation_split_ids": root / "splits" / "val.csv",
    }
    for key, path in checks.items():
        if sha256(path) != spec["source_hashes"][key]:
            raise ValueError(f"Calibration source changed after threshold freeze: {key}")
    thresholds = {
        "fixed": np.full(len(LABELS), spec["fixed_threshold"]),
        "global": np.full(len(LABELS), spec["global_threshold"]),
        "per_label": np.asarray(spec["per_label_thresholds"], dtype=float),
    }
    for values in thresholds.values():
        if values.shape != (len(LABELS),) or not np.isfinite(values).all() or ((values <= 0) | (values >= 1)).any():
            raise ValueError("Invalid frozen thresholds")
    for folder in ("tables", "figures", "preds"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    train_ids = set(pd.read_csv(root / "splits" / "train.csv", dtype=str).sample_id)
    seen_ids = set(train_ids)
    tiers = spec["tiers"]
    counts = np.asarray(spec["train_positive_counts"])
    cost = json.loads((root / "results" / "tables" / f"{run}_cost.json").read_text(encoding="utf-8"))
    cost_table = pd.read_csv(root / "results" / "tables" / "model_cost.csv", index_col=0)
    summary_rows, per_label_rows, tier_rows, cardinality_rows = [], [], [], []
    input_hashes = {}
    for split in ("val", "test"):
        # One load of each saved probability matrix, shared by all three decoders.
        path = root / "results" / "preds" / f"{run}_{split}.npz"
        ids, y, p = load_preds(path)
        y, p = validate_predictions(y, p)
        y = y.astype(np.int8)
        ids = _check_ids(root, split, ids)
        if seen_ids.intersection(ids):
            raise ValueError(f"{split}: overlapping IDs across train/validation/test")
        seen_ids.update(ids)
        if y.shape[1] != len(LABELS):
            raise ValueError("Prediction columns do not match the frozen label order")
        input_hashes[split] = sha256(path)
        cardinality = y.sum(axis=1)  # unique positive labels, not raw list length
        decoded = {s: (p >= threshold).astype(np.int8) for s, threshold in thresholds.items()}
        for strategy, predicted in decoded.items():
            base = {"split": split, "strategy": strategy}
            metrics = {**compute_metrics(y, p, thresholds[strategy]), **_decision_metrics(y, predicted)}
            summary_rows.append({**base, "n_samples": len(y), **metrics,
                                 "n_params": cost["n_params"],
                                 "backbone_train_seconds_raw": cost["train_seconds"],
                                 "backbone_train_seconds_reported": float(cost_table.loc[run, "train_seconds"]),
                                 "additional_backbone_train_seconds": 0.0})
            table = per_label_table(y, p, thresholds[strategy], counts, tiers)
            table.index.name = "label"
            table = table.reset_index().assign(split=split, strategy=strategy,
                                               threshold=thresholds[strategy])
            per_label_rows.extend(table.to_dict("records"))
            for tier in ("head", "mid", "tail"):
                mask = np.array([tiers[label] == tier for label in LABELS])
                tier_rows.append({**base, "tier": tier, "n_labels": int(mask.sum()),
                                  "n_samples": len(y), "positive_support": int(y[:, mask].sum()),
                                  **_decision_metrics(y[:, mask], predicted[:, mask])})
            for group, mask in [("1", cardinality == 1), ("2", cardinality == 2),
                                ("3", cardinality == 3), (">=4", cardinality >= 4)]:
                if mask.any():
                    cardinality_rows.append({**base, "cardinality": group, "n_samples": int(mask.sum()),
                                             **_decision_metrics(y[mask], predicted[mask])})
        # IDs are Unicode, so new artifacts can be read with allow_pickle=False.
        np.savez_compressed(output_dir / "preds" / f"{run}_{split}_decoded.npz",
                            sample_id=np.asarray(ids, dtype=str), true_label_vector=y,
                            probability_vector=p.astype(np.float32),
                            pred_fixed=decoded["fixed"], pred_global=decoded["global"],
                            pred_per_label=decoded["per_label"], label_cardinality=cardinality,
                            labels=np.asarray(LABELS),
                            thresholds_fixed=thresholds["fixed"], thresholds_global=thresholds["global"],
                            thresholds_per_label=thresholds["per_label"],
                            calibration_sha256=np.asarray(frozen_hash))
    summary = pd.DataFrame(summary_rows)
    per_label = pd.DataFrame(per_label_rows)
    tier_results = pd.DataFrame(tier_rows)
    cardinality_results = pd.DataFrame(cardinality_rows)
    for frame, keys in ((summary, ["split"]), (tier_results, ["split", "tier"]),
                        (cardinality_results, ["split", "cardinality"])):
        reference = frame[frame.strategy == "fixed"].set_index(keys)
        for metric in ("macro_f1", "micro_f1", "hamming_loss"):
            baseline = frame[keys].merge(reference[[metric]].reset_index(), on=keys, how="left")[metric]
            frame[f"delta_{metric}_vs_fixed"] = frame[metric].to_numpy() - baseline.to_numpy()
    gaps = []
    for strategy in STRATEGIES:
        d = summary[summary.strategy == strategy].set_index("split")
        row = {"strategy": strategy}
        for metric in ("macro_f1", "micro_f1", "hamming_loss"):
            row.update({f"val_{metric}": d.loc["val", metric], f"test_{metric}": d.loc["test", metric],
                        f"gap_val_minus_test_{metric}": d.loc["val", metric] - d.loc["test", metric]})
        gaps.append(row)
    gaps = pd.DataFrame(gaps)
    profile = pd.DataFrame({"label": LABELS, "tier": [tiers[l] for l in LABELS],
                            "train_count": counts, "val_support": spec["validation_positive_counts"],
                            "fixed_threshold": thresholds["fixed"], "global_threshold": thresholds["global"],
                            "per_label_threshold": thresholds["per_label"]})
    comparison = profile.copy().set_index("label")
    for split in ("val", "test"):
        pivot = per_label[per_label.split == split].pivot(index="label", columns="strategy", values="f1")
        for strategy in STRATEGIES:
            comparison[f"{split}_f1_{strategy}"] = pivot[strategy]
        comparison[f"{split}_gain_per_label_vs_fixed"] = pivot["per_label"] - pivot["fixed"]
        comparison[f"{split}_gain_per_label_vs_global"] = pivot["per_label"] - pivot["global"]
    comparison = comparison.reset_index()
    # Macro-F1 can only improve on validation when independently optimizing each
    # label over the same grid. Test performance has no such guarantee.
    val_macro = summary[summary.split == "val"].set_index("strategy").macro_f1
    if not (val_macro.per_label + 1e-12 >= val_macro["global"] >= val_macro.fixed - 1e-12):
        raise AssertionError("Validation calibration invariant failed")
    for name, frame in {"strategy_metrics": summary, "per_label_metrics": per_label,
                        "tier_metrics": tier_results, "cardinality_metrics": cardinality_results,
                        "generalization_gaps": gaps, "threshold_profile": profile,
                        "per_label_comparison": comparison}.items():
        frame.to_csv(output_dir / "tables" / f"{name}.csv", index=False)
    metadata = {
        "run": run, "calibration_sha256": frozen_hash, "prediction_sha256": input_hashes,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "platform": platform.platform(),
        "evaluation_device": "CPU", "backbone_retrained": False,
        "checkpoint": cost["checkpoint"], "checkpoint_revision": cost["revision"],
        "training_args": cost["args"],
        "test_caveat": "These held-out predictions were already inspected during Midterm; this is a post-hoc comparison, not a newly untouched test set.",
    }
    if sha256(calibration_path) != frozen_hash:
        raise RuntimeError("Frozen threshold specification changed during evaluation")
    _write_json(output_dir / "evaluation.json", metadata)
    _plot(output_dir, summary, profile, tier_results)
    _report(root, output_dir, spec, summary, gaps, tier_results, cardinality_results, comparison, report_date)
    return summary


def _plot(output_dir, summary, profile, tier_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4c78a8", "#f58518", "#54a24b"]
    test = summary[summary.split == "test"].set_index("strategy").loc[list(STRATEGIES)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ("macro_f1", "micro_f1", "hamming_loss")):
        bars = ax.bar(STRATEGIES, test[metric], color=colors)
        for bar, value in zip(bars, test[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(metric + (" (lower is better)" if metric == "hamming_loss" else ""))
        ax.set_ylim(0, test[metric].max() * 1.2)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Same PhoBERT probabilities; thresholds selected using validation only")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "strategy_metrics.png", dpi=160)
    plt.close(fig)
    d = profile.sort_values("train_count")
    fig, ax = plt.subplots(figsize=(9, 9))
    tier_colors = {"head": "#4c78a8", "mid": "#f58518", "tail": "#54a24b"}
    ax.barh(d.label, d.per_label_threshold, color=[tier_colors[t] for t in d.tier])
    ax.axvline(0.5, color="gray", ls=":", label="fixed = 0.5")
    ax.axvline(float(profile.global_threshold.iloc[0]), color="black", ls="--", label="global tuned")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Threshold maximizing validation F1 on the shared grid")
    ax.set_title("Per-label thresholds (ordered by train frequency)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "threshold_profile.png", dpi=160)
    plt.close(fig)
    d = tier_results[tier_results.split == "test"].pivot(index="tier", columns="strategy", values="macro_f1")
    d = d.loc[["head", "mid", "tail"], list(STRATEGIES)]
    ax = d.plot.bar(figsize=(8, 4), color=colors, rot=0)
    ax.set_ylabel("Macro-F1 (test)")
    ax.set_ylim(0, 0.8)
    ax.set_title("Threshold comparison by train-frequency tier")
    ax.legend(title=None, ncol=3, loc="upper center")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "figures" / "f1_by_tier.png", dpi=160)
    plt.close(ax.figure)


def _markdown_table(frame):
    def value(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.4f}"
        return str(x).replace("|", "\\|")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines += ["| " + " | ".join(value(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def _report(root, output_dir, spec, summary, gaps, tier_results, cardinality_results, comparison, report_date):
    day = report_date or datetime.now(timezone(timedelta(hours=7))).date().isoformat()
    test = summary[summary.split == "test"].set_index("strategy").loc[list(STRATEGIES)]
    val = summary[summary.split == "val"].set_index("strategy").loc[list(STRATEGIES)]
    gain_fixed = test.loc["per_label", "macro_f1"] - test.loc["fixed", "macro_f1"]
    gain_global = test.loc["per_label", "macro_f1"] - test.loc["global", "macro_f1"]
    better = int((comparison.test_gain_per_label_vs_fixed > 1e-12).sum())
    worse = int((comparison.test_gain_per_label_vs_fixed < -1e-12).sum())
    best = test.macro_f1.idxmax()
    threshold_min = min(spec["per_label_thresholds"])
    threshold_max = max(spec["per_label_thresholds"])
    tail = tier_results[(tier_results.split == "test") & (tier_results.tier == "tail")].set_index("strategy")
    tail_gain = tail.loc["per_label", "macro_f1"] - tail.loc["fixed", "macro_f1"]
    lines = ["# RQ1 - So sánh ngưỡng cố định, ngưỡng chung và ngưỡng riêng từng nhãn", "",
             f"Ngày cập nhật: {day}. Mô hình: `{spec['run']}`. Không huấn luyện lại backbone.", "",
             "## Giao thức", "",
             "- Giữ nguyên mô hình PhoBERT đã chọn ở giữa kì theo Macro-F1 validation tại 0,5; dùng cùng xác suất cho cả ba cách giải mã.",
             "- Giữ đủ 28 cột của dữ liệu: 27 cảm xúc và neutral. Split cố định: train 16.531, val 2.066, test 2.067.",
             "- Lưới đặt trước, giống giữa kì: 0,05 đến 0,90, bước 0,025. Ngưỡng chung tối đa Macro-F1 val; ngưỡng riêng tối đa F1 val của từng nhãn trên cùng lưới.",
             "- Khi hòa điểm: chọn ngưỡng gần 0,5 nhất, sau đó chọn ngưỡng nhỏ hơn. Nhãn không có mẫu dương trên val giữ 0,5.",
             f"- Ngưỡng chung đã chốt: **{spec['global_threshold']:g}**. Tất cả ngưỡng được ghi vào `results/rq1/calibration.json` trước khi mở file xác suất test.",
             "- Mỗi ma trận val/test được nạp một lần trong pha đánh giá, rồi áp ba bộ ngưỡng đã cố định; không quét hoặc chọn ngưỡng bằng nhãn test.",
             "- Chỉ đọc ID, nhãn và xác suất đã lưu; không tải dataset và không xuất văn bản gốc. Cardinality tính từ multi-hot nên không đếm trùng nhãn.", "",
             "## Kết quả test", "",
             _markdown_table(test.reset_index()[["strategy", "macro_f1", "micro_f1", "hamming_loss", "lrap", "mAP", "subset_acc"]]), "",
             f"Ngưỡng riêng so với 0,5: {gain_fixed:+.4f} Macro-F1 ({100 * gain_fixed:+.2f} điểm phần trăm). So với ngưỡng chung: {gain_global:+.4f} ({100 * gain_global:+.2f} điểm phần trăm).",
             f"Trong lần chạy này, `{best}` có Macro-F1 test cao nhất. Đây là kết quả quan sát để báo cáo, không dùng để chọn lại ngưỡng.",
             f"So với ngưỡng 0,5: F1 của {better}/28 nhãn tăng, {worse}/28 nhãn giảm, {28 - better - worse}/28 nhãn giữ nguyên.", "",
             "LRAP và mAP giống nhau giữa ba cách vì cùng ma trận xác suất, chỉ khác bước nhị phân hóa. Subset accuracy chỉ là chỉ số phụ.", "",
             "## Trả lời RQ1", "",
             f"Ngưỡng 0,5 không đạt Macro-F1 validation cao nhất trên lưới của mô hình này. Ngưỡng riêng đã chọn trải từ {threshold_min:g} đến {threshold_max:g}; cả global và per-label đều tăng Macro-F1 test so với 0,5.",
             f"Tuy vậy lợi ích bổ sung của per-label so với global chỉ {gain_global:+.4f} Macro-F1. Global có Micro-F1 {test.loc['global', 'micro_f1']:.4f} và Hamming Loss {test.loc['global', 'hamming_loss']:.4f}, so với {test.loc['per_label', 'micro_f1']:.4f} và {test.loc['per_label', 'hamming_loss']:.4f} của per-label. Không có một cách giải mã tốt nhất ở mọi chỉ số.",
             f"Đặc biệt nhóm tail thay đổi {tail_gain:+.4f} Macro-F1 khi dùng per-label thay cho 0,5; kết quả này không ủng hộ kết luận ngưỡng riêng luôn giúp nhãn hiếm. Cần giữ kết luận trong phạm vi một mô hình và một seed.", "",
             "## Validation và khả năng tổng quát hóa", "",
             _markdown_table(val.reset_index()[["strategy", "macro_f1", "micro_f1", "hamming_loss"]]), "",
             _markdown_table(gaps[["strategy", "gap_val_minus_test_macro_f1", "gap_val_minus_test_micro_f1"]]), "",
             "Gap = F1 val trừ F1 test. Việc F1 val tăng là hệ quả của tối ưu trên val; không bảo đảm tăng tương ứng trên test. Gap dương lớn hơn gợi ý khả năng quá khớp, nhưng riêng gap không chứng minh nguyên nhân.", "",
             "## Kết quả theo nhóm tần suất train", "",
             _markdown_table(tier_results[tier_results.split == "test"][["strategy", "tier", "n_labels", "macro_f1", "micro_f1", "hamming_loss"]]), "",
             "Head >= 1.000 mẫu dương train, mid 750-999, tail < 750. Định nghĩa giữ nguyên từ giữa kì, không dùng nhãn test để chia nhóm.", "",
             "## Kết quả theo số nhãn thật mỗi câu", "",
             _markdown_table(cardinality_results[cardinality_results.split == "test"][["strategy", "cardinality", "n_samples", "macro_f1", "micro_f1", "hamming_loss", "samples_f1"]]), "",
             "Các CSV tổng hợp và phân nhóm còn có delta Macro-F1, Micro-F1 và Hamming Loss so với fixed. Delta Hamming âm là cải thiện.", "",
             "## Các nhãn thay đổi nhiều nhất", "",
             _markdown_table(comparison.sort_values("test_gain_per_label_vs_fixed", ascending=False).head(5)[["label", "per_label_threshold", "test_f1_fixed", "test_f1_per_label", "test_gain_per_label_vs_fixed"]]), "",
             _markdown_table(comparison.sort_values("test_gain_per_label_vs_fixed").head(5)[["label", "per_label_threshold", "test_f1_fixed", "test_f1_per_label", "test_gain_per_label_vs_fixed"]]), "",
             "## Giới hạn", "",
             "- Test đã được xem và phân tích trong giữa kì. Đây là so sánh hậu nghiệm trên dự đoán giữ lại, không phải một tập test mới hoàn toàn chưa được xem. Thuật toán và lưới lần này được cố định trước khi đánh giá.",
             "- Một mô hình, một seed 42; chưa có sai số giữa nhiều lần huấn luyện hoặc kiểm định ý nghĩa thống kê. Không kết luận chắc về chênh lệch nhỏ.",
             "- Ngưỡng chỉ tốt nhất trên lưới đã công bố, không phải tối ưu liên tục. Mỗi nhãn thêm một ngưỡng có thể quá khớp val, nhất là nhãn ít mẫu dương.",
             "- Không áp quy tắc bắt buộc ít nhất một nhãn hay loại trừ neutral: giữ cùng giao thức sigmoid độc lập cho cả ba cách.",
             "- Các nhóm cardinality có ít mẫu: đặc biệt nhóm >=4 chỉ có 66 câu test; Macro-F1 nhóm tính đủ 28 nhãn (zero_division=0), nên cần đọc cùng Micro-F1 và support.", "",
             "## Sản phẩm và cách chạy lại", "",
             "```bash", "cd ati_final", "pip install -r requirements-analysis.txt",
             "python -m src.analyze_thresholds --mode calibrate",
             "python -m src.analyze_thresholds --mode evaluate", "```", "",
             "Có thể chạy gộp bằng `python -m src.analyze_thresholds` hoặc notebook `06_thresholds.ipynb`.", "",
             "- `results/rq1/calibration.json`: ngưỡng đã cố định, lưới, tie rule, nhãn, tần suất và hash nguồn val.",
             "- `results/rq1/evaluation.json`: hash nguồn, checkpoint/revision, cấu hình gốc và phiên bản thư viện phân tích.",
             "- `results/rq1/tables/`: chỉ số tổng hợp, F1/precision/recall/AP từng nhãn, so sánh từng nhãn, head/mid/tail, cardinality, profile ngưỡng và gap val-test.",
             "- `results/rq1/preds/`: hai NPZ chứa ID, nhãn thật, xác suất, pred_fixed/pred_global/pred_per_label, cardinality và ba bộ ngưỡng; mở được với allow_pickle=False.",
             "- `results/rq1/figures/`: biểu đồ chỉ số tổng hợp, ngưỡng từng nhãn và F1 theo nhóm tần suất.",
             "- Thời gian/số tham số gốc giữ trong `strategy_metrics.csv`: mọi cách giải mã dùng cùng backbone; thời gian train gốc và thời gian đã hiệu chỉnh từ giữa kì được ghi riêng.", ""]
    report_path = root / "reports" / "rq1_threshold_comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "evaluate", "run"), default="run")
    parser.add_argument("--run", default=DEFAULT_RUN, help="PhoBERT run already selected using Midterm validation")
    args = parser.parse_args()
    if args.mode in ("calibrate", "run"):
        spec = calibrate(run=args.run)
        print(f"Frozen global threshold: {spec['global_threshold']:g}")
    if args.mode in ("evaluate", "run"):
        summary = evaluate()
        print(summary[["split", "strategy", "macro_f1", "micro_f1", "hamming_loss", "lrap", "mAP"]].to_string(index=False))
        print("Artifacts: results/rq1/; report: reports/rq1_threshold_comparison.md")


if __name__ == "__main__":
    main()
