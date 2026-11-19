# ViGoEmotions: Multi-label Vietnamese Emotion Recognition (Midterm + Final RQ1)

Advanced Topics in Information Technology 2026 · Class 62FIT4ATI-CLC02 · Group 03 · Project P02

| Member | Student ID |
|---|---|
| Nguyễn Quốc Khanh | 2301140051 |
| Đinh Tiến Đạt | 2301140022 |
| Nguyễn Duy Anh | 2301140006 |

GitHub repository: https://github.com/tiendat13ns/Deep-Learning (folder `ati_final/`)

**Task.** Fine-grained multi-label emotion recognition on ViGoEmotions (Tran et al., EACL 2026): 20,664 Vietnamese social-media comments, 27 emotions plus `neutral`, 28 labels.
**Midterm.** Three baselines: TF-IDF + logistic regression, BiLSTM and PhoBERT-base.
**Final (RQ1).** *Is a single uniform decision threshold (τ = 0.5) optimal across all labels under class imbalance?* We decode the same PhoBERT probabilities with a fixed 0.5, one global threshold and one threshold per label (both tuned on validation only and frozen before test inference), check the differences with a paired bootstrap and two extra training seeds, and audit the errors.

## Main results (test set)

| Midterm, τ = 0.5, seed 42 | Macro-F1 | Micro-F1 | Hamming loss | LRAP | mAP | Parameters | Train time |
|---|---|---|---|---|---|---|---|
| TF-IDF + One-vs-Rest Logistic Regression | 0.548 | 0.563 | 0.0666 | 0.694 | 0.561 | 1.05 M | 2.7 s |
| BiLSTM | 0.511 | 0.541 | 0.0590 | 0.675 | 0.539 | 1.38 M | 151 s |
| PhoBERT-base (selected) | **0.608** | **0.624** | **0.0583** | **0.753** | **0.630** | 135.0 M | ~20 min |

| Final RQ1, PhoBERT-base, 3 seeds (mean ± std) | Macro-F1 | Micro-F1 | Hamming loss | Tail Macro-F1 | Val − test gap |
|---|---|---|---|---|---|
| Fixed τ = 0.5 | 0.6036 ± 0.0088 | 0.6190 ± 0.0087 | 0.0591 ± 0.0017 | 0.5673 ± 0.0159 | +0.0002 |
| Global τ* (0.575–0.600) | 0.6098 ± 0.0058 | **0.6254 ± 0.0058** | **0.0544 ± 0.0010** | 0.5741 ± 0.0128 | −0.0009 |
| Per-label τ*_k | **0.6115 ± 0.0031** | 0.6241 ± 0.0030 | 0.0551 ± 0.0015 | 0.5751 ± 0.0026 | +0.0157 |

Answer to RQ1: 0.5 is not optimal; a single validation-tuned threshold is better in every seed and reliably lowers Hamming loss. Per-label thresholds add no reliable gain over it (+0.0017 ± 0.0032 Macro-F1, sign changes across seeds, bootstrap 95% CI includes 0), overfit validation and do not help tail labels.

## Contents

| Path | Content |
|---|---|
| `notebooks/` | Executed notebooks, outputs kept: `01` dataset analysis · `02` TF-IDF + LR · `03` BiLSTM · `04` PhoBERT results and merged table · `05` Midterm questions 1–5 and error analysis · `06` RQ1 threshold calibration and evaluation · `07` RQ1 bootstrap, three seeds and error audit |
| `src/` | Data loading (`data.py`), preprocessing, VnCoreNLP segmentation, metrics, PhoBERT training (`train_phobert.py`), threshold calibration (`thresholds.py`, `analyze_thresholds.py`), RQ1 robustness and error audit (`rq1_audit.py`), CSV export of predictions (`export_predictions.py`), local case viewer (`show_cases.py`) |
| `configs/` | Hyper-parameters of every experiment (one JSON per run) and the RQ1 protocol (`rq1_thresholds.json`) |
| `splits/` | Comment ids of train / val / test (16,531 / 2,066 / 2,067) and `label_tiers.json` (head / mid / tail, from training counts) |
| `results/tables/` | Result tables (CSV) and per-run training cost (`*_cost.json`, `model_cost.csv`) |
| `results/figures/` | Midterm analysis figures; `midterm_slides/` the figures of the Midterm slides (in the repository: `slides/assets/`); `final_report_color/` and `final_report_bw/` the figures of the Final report (colour and black-and-white versions) |
| `results/predictions/` | Per-comment labels and probabilities of every run as CSV; `rq1_*.csv` add `pred_fixed`, `pred_global`, `pred_per_label` and `label_cardinality` (label order in `labels.json`) |
| `results/preds/` | The same predictions as `.npz`, read by the notebooks |
| `results/rq1/` | Frozen thresholds (`calibration.json`), RQ1 tables and figures, decoded predictions, the extra seeds (`seeds/`) and the error audit (`error_analysis/`: case ids, probabilities and paraphrased notes) |
| `results/error_cases/` | Midterm error-case sample (ids and probabilities only) |
| `results/environment.md` | Hardware, library versions and seeds |
| `reports/final_report/` | Scripts that draw the report figures and build the report from the course template |
| `reports/rq1_*.md` | RQ1 analysis notes (in Vietnamese) |
| `tests/` | 13 unit tests: threshold calibration, the validation/test boundary, the bootstrap scorer |

No file contains comment text, in line with the dataset licence.

## Data

- **Source:** Hugging Face dataset [uitnlp/vigoemotions](https://huggingface.co/datasets/uitnlp/vigoemotions). The dataset is gated: accept its terms on the page, then run `hf auth login` or set `HF_TOKEN`. `src/data.py` downloads it to `data/` on first use.
- **Terms:** non-commercial research and education only; the data must not be redistributed. `data/`, model checkpoints and tokens are never included in this submission.
- **Split:** the official split; the ids are in `splits/`.

## Installation

Python 3.10+ (developed with 3.12.4); Java 8+ for VnCoreNLP word segmentation (PhoBERT only); a CUDA GPU for the BiLSTM and PhoBERT (6 GB is enough).

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA build of PyTorch
python -c "from src.data import load_all; print({k: len(v) for k, v in load_all().items()})"   # {'train': 16531, 'val': 2066, 'test': 2067}
```

Everything after model training (notebooks `04`–`07`, RQ1, figures) runs on the CPU from the saved predictions in `results/`; `requirements-analysis.txt` is enough for that part and needs no data download.

## Running order

Run from this folder (`code/`, i.e. `ati_final/` in the repository). On Windows set `PYTHONIOENCODING=utf-8`; if `jupyter` is not on the PATH use `python -m jupyter`.

```bash
# 1. Midterm baselines
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_tfidf_lr.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_bilstm.ipynb
python -m src.train_phobert --emoji text --loss pw_sqrt --epochs 12 --patience 3 --name phobert_text_pw_sqrt_ep12   # selected model, ~20 min on GPU
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_phobert.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/05_analysis.ipynb

# 2. Final RQ1 (seed 42): calibrate on validation, freeze, evaluate once
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/06_thresholds.ipynb   # = python -m src.analyze_thresholds

# 3. RQ1 robustness: two extra seeds of the same configuration, bootstrap, error audit
python -m src.train_phobert --emoji text --loss pw_sqrt --epochs 12 --patience 3 --seed 43 --name phobert_text_pw_sqrt_ep12_seed43
python -m src.train_phobert --emoji text --loss pw_sqrt --epochs 12 --patience 3 --seed 44 --name phobert_text_pw_sqrt_ep12_seed44
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/07_rq1_robustness_errors.ipynb   # = python -m src.rq1_audit
python -m unittest discover -s tests

# 4. CSV predictions, report figures and the report (the report build needs Microsoft Word on Windows
#    and ATI_Final_Report_Template.docx placed next to this README)
python -m src.export_predictions
python reports/final_report/make_figures.py --style color
python reports/final_report/make_report.py --style color
powershell -File reports/final_report/update_fields.ps1 reports/final_report/62FIT4ATI-CLC02_G03_P02_report_color.docx
```

The other four PhoBERT runs of the Midterm ablation (emoji kept or converted, plain or square-root-weighted BCE, 6 or 12 epochs) use the same command with the arguments in `configs/phobert_*.json`. The error audit's case sample is drawn with seed 42 before any text is read; `python -m src.show_cases --cases results/rq1/error_analysis/cases.csv --group <group>` prints those comments on screen to read them locally and never saves them.

## Where each table and figure of the report comes from

| Report item | Produced by | File |
|---|---|---|
| Table 2 (dataset statistics) | notebook `01`; report script (cardinality from `splits/` labels) | `results/tables/text_length.csv`, `informal_features.csv` |
| Table 3 (training configuration) | – | `configs/*.json` |
| Table 4 (compute cost) | training runs | `results/tables/model_cost.csv`, `results/tables/*_cost.json` |
| Table 5 (Midterm baselines) | notebook `04` | `results/tables/baseline_results.csv` |
| Table 6 (RQ1, 3 seeds) | notebook `07` | `results/rq1/tables/seed_mean_std.csv`, `seed_results.csv` |
| Table 7 (paired differences, bootstrap) | notebook `07` | `results/rq1/tables/seed_macro_f1_deltas.csv`, `bootstrap_deltas.csv` |
| Table 8 (label cardinality) | notebooks `06`, `07` | `results/rq1/tables/cardinality_metrics.csv`, `results/rq1/error_analysis/cardinality_errors.csv` |
| Table 9 (validation-to-test gap) | notebooks `06`, `07` | `results/rq1/tables/generalization_gaps.csv`, `seed_mean_std.csv` |
| Tables 10–13 (error audit) | notebook `07` | `results/rq1/error_analysis/error_counts.csv`, `confused_pairs.csv`, `cause_summary.csv`, `cases.csv`, `annotations.csv` |
| Table 16 (per-label results) | notebook `06` | `results/rq1/tables/per_label_comparison.csv`, `per_label_metrics.csv` |
| Figures 1–7 | `reports/final_report/make_figures.py` | `results/figures/final_report_color/` (inputs: `label_distribution.csv` from notebook `01`; `seed_results.csv`, `seed_threshold_stability.csv`, `seed_gain_transfer.csv`, `decision_flips.csv`, `cardinality_errors.csv` from notebook `07`) |

## Reproducibility

| Item | Value |
|---|---|
| Hardware | NVIDIA RTX 3060 Laptop GPU, 6 GB (PhoBERT peak memory 3.3 GB); details in `results/environment.md` |
| Seeds | 42 for every Midterm run (`SEED` in `src/data.py`: Python, NumPy, PyTorch), the bootstrap and the audit sample; 43 and 44 for the RQ1 seed check |
| Pretrained checkpoint | `vinai/phobert-base`, revision `01daacda68afe13d83023d16ec647239e344a1e6` |
| Preprocessing | HTML unescape, `<br>` → space, Unicode NFC, whitespace collapse; emoji and teencode kept. PhoBERT only: emoji → English names, then VnCoreNLP word segmentation, max 128 tokens |
| PhoBERT settings | AdamW, lr 2e-5, weight decay 0.01, 10% warm-up, batch 16, fp16, up to 12 epochs, early stopping (patience 3) on validation Macro-F1; BCE with √(neg/pos) positive weights |
| Decision thresholds | 0.5 in the Midterm; RQ1 grid 0.05–0.90 (step 0.025), tuned on validation only, frozen in `results/rq1/calibration.json` before test inference |

## Reference

H. Q. Tran, N. T. Pham, S. T. Luu, and K. V. Nguyen, "ViGoEmotions: A benchmark dataset for fine-grained emotion detection on Vietnamese texts," in *Proc. EACL*, vol. 1, 2026, pp. 2805–2831, doi: 10.18653/v1/2026.eacl-long.129.
