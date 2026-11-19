# Environment

All experiments of the Midterm and the Final (RQ1) were run on one machine.

## Hardware

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB (PhoBERT peak memory 3.26–3.27 GB) |
| OS | Windows 11 (10.0.26300) |
| GPU time | about 2.1 hours in total (5 Midterm PhoBERT runs, BiLSTM, 2 extra PhoBERT seeds); see `results/tables/model_cost.csv` and `results/tables/*_cost.json` |
| CPU-only parts | TF-IDF + LR, threshold calibration and evaluation, bootstrap, error audit, figures |

## Software

| Package | Version |
|---|---|
| Python | 3.12.4 |
| PyTorch | 2.14.1 + CUDA 12.6 |
| transformers | 5.17.0 |
| huggingface_hub | 1.33.0 |
| datasets | 5.1.0 |
| scikit-learn | 1.9.0 |
| numpy | 2.5.0 |
| pandas | 3.0.6 |
| scipy | 1.18.0 |
| matplotlib | 3.11.0 |
| emoji | 2.16.0 |
| VnCoreNLP | 1.2 (jar, RDRSegmenter word segmentation; downloaded by `src/segment.py`) |
| Java | 26 (any Java 8+ works for VnCoreNLP) |
| Report build only | python-docx 1.2.0, latex2mathml 3.81.1, lxml 6.1.1, Microsoft Word (equations via `MML2OMML.XSL`, field update via `update_fields.ps1`) |

The RQ1 threshold analysis was first run on Python 3.10.5, numpy 2.2.6, pandas 2.3.3 and scikit-learn 1.7.2 (Windows 10). Re-running it on the versions above reproduced every table, threshold and decoded prediction exactly.

## Pretrained checkpoint

`vinai/phobert-base`, Hugging Face revision `01daacda68afe13d83023d16ec647239e344a1e6`.

## Seeds

| Seed | Used for |
|---|---|
| 42 | Every Midterm run (`SEED` in `src/data.py`: Python, NumPy, PyTorch), the selected PhoBERT model, the RQ1 bootstrap and the error-audit case sample |
| 43, 44 | Two extra trainings of the selected PhoBERT configuration, used only to measure seed variation in RQ1 |

## Notes

- On Windows set `PYTHONIOENCODING=utf-8` when redirecting training logs to a file (the scripts print Vietnamese text).
- Run notebooks with `python -m jupyter nbconvert --to notebook --execute --inplace <notebook>` if `jupyter` is not on the PATH.
- GPU kernels are not bit-for-bit deterministic, so retraining PhoBERT can differ slightly from the saved predictions; every number in the report is computed from the saved predictions in `results/preds/`.
