"""Export every saved prediction file to CSV (results/predictions/), as the submission guide asks.

Run from ati_final:  python -m src.export_predictions
The .npz files in results/preds/ and results/rq1/ remain the inputs of the notebooks; the CSVs are a
readable copy. No comment text is written. Vectors are space-separated in the label order of
results/predictions/labels.json (27 emotions + neutral).

  <run>_<split>.csv                columns: sample_id, true_label_vector, probability_vector, label_cardinality
  rq1_<run>_<split>.csv            the RQ1 brief's columns: sample_id, true_label_vector, probability_vector,
                                   pred_fixed, pred_global, pred_per_label, label_cardinality
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABELS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "predictions"


def vec(m, fmt):
    return [" ".join(fmt.format(v) for v in row) for row in m]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "labels.json").write_text(json.dumps({"label_order": LABELS}, indent=1), encoding="utf-8")
    n = 0
    for f in sorted((ROOT / "results" / "preds").glob("*.npz")):
        z = np.load(f, allow_pickle=True)  # Midterm files store ids as an object array
        y, p = z["y"].astype(int), z["p"]
        pd.DataFrame({"sample_id": z["ids"].astype(str), "true_label_vector": vec(y, "{:d}"),
                      "probability_vector": vec(p, "{:.6f}"), "label_cardinality": y.sum(1)}
                     ).to_csv(OUT / f"{f.stem}.csv", index=False)
        n += 1
    rq1 = [ROOT / "results" / "rq1" / "preds"] + sorted((ROOT / "results" / "rq1" / "seeds").glob("*/preds"))
    for folder in rq1:
        for f in sorted(folder.glob("*_decoded.npz")):
            z = np.load(f, allow_pickle=False)
            assert list(z["labels"]) == LABELS
            pd.DataFrame({"sample_id": z["sample_id"], "true_label_vector": vec(z["true_label_vector"], "{:d}"),
                          "probability_vector": vec(z["probability_vector"], "{:.6f}"),
                          "pred_fixed": vec(z["pred_fixed"], "{:d}"), "pred_global": vec(z["pred_global"], "{:d}"),
                          "pred_per_label": vec(z["pred_per_label"], "{:d}"),
                          "label_cardinality": z["label_cardinality"]}
                         ).to_csv(OUT / f"rq1_{f.stem.replace('_decoded', '')}.csv", index=False)
            n += 1
    print(f"{n} CSV files in {OUT}")


if __name__ == "__main__":
    main()
