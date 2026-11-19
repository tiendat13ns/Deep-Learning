# Pipeline chuẩn bị giữa kì - ViGoEmotions

> Cập nhật 22/10/2026: đã bổ sung RQ1 ở `notebooks/06_thresholds.ipynb` và `src/analyze_thresholds.py`: hiệu chỉnh ngưỡng riêng trên val, so sánh fixed/global/per-label với đủ chỉ số, head/mid/tail, cardinality và gap val–test. Kết quả và giới hạn xem [reports/rq1_threshold_comparison.md](reports/rq1_threshold_comparison.md). Các mục slide, kiểm tra chéo lỗi và báo cáo cuối kì dạng bài báo vẫn chưa hoàn tất.

Mục tiêu: nộp đủ mục E của đề (slide, notebook, bảng/hình, bảng kết quả baseline, phân tích lỗi, câu hỏi nghiên cứu cho cuối kì, README).
Trạng thái: ✅ xong · 🟡 làm một phần · ⬜ chưa làm.

## 1. Tổng quan luồng

```
Dữ liệu (HF) → splits cố định → [TF-IDF+LR] [BiLSTM/GRU] [PhoBERT]
      → lưu xác suất val/test (.npz) → metrics.py → bảng kết quả
      → phân tích 5 câu hỏi + phân tích lỗi → slide + README
```

## 2. Các bước

| Tuần | Bước | Việc cần làm | Đầu ra | Trạng thái |
|------|------|--------------|--------|-----------|
| 1 | 1.1 | Môi trường: PyTorch CUDA, đăng nhập HF | Máy nhận GPU | ✅ |
| 1 | 1.2 | Tải dữ liệu, kiểm tra split, chốt giao thức chia | `src/data.py`, `splits/*.csv` | ✅ |
| 1 | 1.3 | Phân tích dữ liệu (mục A) | `notebooks/01_eda.ipynb`, `results/tables`, `results/figures` | ✅ |
| 1 | 1.4 | Viết `src/metrics.py` dùng chung: Macro/Micro-F1, F1 từng nhãn, Hamming Loss, LRAP/mAP; hàm lưu xác suất ra `.npz` | `src/metrics.py`, `src/preprocess.py` | ✅ |
| 1 | 1.5 | Baseline TF-IDF + OvR Logistic Regression | `notebooks/02_tfidf_lr.ipynb`, `results/preds/tfidf_lr_*.npz` | ✅ |
| 2 | 2.1 | BiLSTM hoặc GRU (embedding khởi tạo ngẫu nhiên hoặc fastText nếu có), BCE + sigmoid | `notebooks/03_bilstm.ipynb`, `results/preds/bilstm_*.npz` | ✅ |
| 2 | 2.2 | PhoBERT-base: tách từ (VnCoreNLP/underthesea), `max_length` 128, fp16, batch 16, sigmoid + BCE; lưu checkpoint mỗi epoch | `src/train_phobert.py`, `notebooks/04_phobert.ipynb`, `results/preds/phobert_*.npz` | ✅ |
| 2 | 2.3 | Ghi cho mỗi mô hình: số tham số, thời gian train, seed, tên checkpoint, siêu tham số | `results/tables/model_cost.csv` | ✅ |
| 2 | 2.4 | Gộp bảng kết quả baseline (mỗi mô hình một dòng, đủ chỉ số) | `results/tables/baseline_results.csv` | ✅ |
| 3 | 3.1 | Câu 1–2: F1 từng nhãn cao/thấp nhất; tương quan Spearman giữa F1 và tần suất train | hình + bảng | ✅ (`05_analysis.ipynb`: `q12_f1_by_label.png`, `q12_f1_vs_frequency.png`) |
| 3 | 3.2 | Câu 3: so ma trận đồng xuất hiện thật (train) với dự đoán của mô hình | hình + bảng | ✅ (`05_analysis.ipynb`: `q3_*`) |
| 3 | 3.3 | Câu 4: PhoBERT vs BiLSTM trên nhóm câu ngắn / nhiều emoji / có teencode (dùng cột cờ của EDA) | bảng theo nhóm | ✅ (`05_analysis.ipynb`: `q4_*`, bootstrap CI) |
| 3 | 3.4 | Câu 5: quét τ từ 0,1 đến 0,9, vẽ Macro-F1 theo τ (**chỉ dùng val để chọn**, test chỉ để báo cáo) | hình + bảng | ✅ (`results/figures/threshold_sweep_all_models.png`, `threshold_sweep_final3.png`, `threshold_sweep_6_vs_12_epoch.png`; chọn τ* trên val) |
| 3 | 3.5 | Phân tích lỗi: chọn ca FN/FP tiêu biểu và câu nhiều nhãn (đọc trên máy, diễn đạt lại, không đưa văn bản gốc vào repo) | `notebooks/05_analysis.ipynb`, ghi chú | ✅ (`reports/error_analysis_notes.md`: 70 ca, Claude soạn, **cần nhóm kiểm tra chéo**) |
| 4 | 4.1 | Đề xuất câu hỏi nghiên cứu cho cuối kì (RQ1: ngưỡng chung vs ngưỡng từng nhãn) kèm bằng chứng từ câu 5 | đoạn trong slide/README | ⬜ |
| 4 | 4.2 | Làm slide | file slide | ⬜ |
| 4 | 4.3 | Viết README: cấu trúc thư mục, cách chạy, seed, checkpoint, license | `README.md` | ✅ (đã có; cập nhật lại ở cuối) |
| 4 | 4.4 | Dự phòng 2–3 ngày: chạy lại nếu lỗi, kiểm tra tái lập | - | ⬜ |

## 3. Quy tắc bắt buộc (từ đề)

| Quy tắc | Cách tuân thủ |
|---------|---------------|
| Không chỉnh ngưỡng trên test | Mọi lựa chọn ngưỡng/siêu tham số dùng val; test chạy một lần cuối |
| Không dựng đồ thị/ma trận nhãn từ nhãn test | Chỉ dùng train (đã làm ở EDA) |
| Lưu xác suất trước khi nhị phân hoá | Mỗi mô hình lưu `.npz` (val + test) trước khi áp ngưỡng |
| Cố định seed, ghi checkpoint | `SEED = 42`; ghi tên checkpoint PhoBERT và phiên bản thư viện |
| Không đưa văn bản gốc lên repo | `data/` nằm trong `.gitignore`; chỉ commit mã, chỉ số split, bảng tổng hợp |
| Không dùng subset accuracy làm chỉ số chính | Báo cáo Macro/Micro-F1 là chính |

## 4. Gợi ý cấu hình ban đầu (RTX 3060 Laptop, 6 GB)

| Mô hình | Thiết bị | Thiết lập đề xuất |
|---------|----------|-------------------|
| TF-IDF + LR | CPU | n-gram (1,2); `class_weight` thử cả có và không; vài phút |
| BiLSTM/GRU | GPU | Hidden 128–256; dropout 0,3; early stopping theo Macro-F1 trên val |
| PhoBERT-base | GPU | fp16, `max_length` 128, batch 16 (giảm xuống 8 + gradient accumulation nếu hết VRAM), lr 2e-5, 3–5 epoch |

## 5. Phân công gợi ý (3 người)

| Người | Phần việc |
|-------|-----------|
| Bạn | Toàn bộ code và chạy thực nghiệm |
| Thành viên 2 | Làm slide, soạn bảng/hình từ kết quả |
| Thành viên 3 | Ghi chú phân tích lỗi (đọc ca FN/FP), viết README |

## 6. Cấu trúc thư mục đích

```
ati_final/
  notebooks/  01_eda · 02_tfidf_lr · 03_bilstm · 04_phobert · 05_analysis
  src/        data.py · metrics.py
  splits/     train/val/test.csv · label_tiers.json
  results/    figures/ · tables/ · preds/
  PROGRESS.md · PIPELINE.md · README.md
```

## 7. Lịch theo ngày (cập nhật 12/10)

Mốc cố định: **báo cáo tiến độ thứ Sáu 23/10**, **thuyết trình giữa kì thứ Sáu 30/10**.

| Thời gian | Việc | Trạng thái |
|---|---|---|
| 12–14/10 | Đánh giá định tính mục A (teencode, chữ lóng, emoji, cú pháp); viết mục RQ1; push GitHub | ⬜ |
| 15–19/10 | Thử lr cho PhoBERT (2e-5, 3e-5, 5e-5) và hidden/lr cho BiLSTM, chọn theo val; chạy lại notebook 04 và 05; cập nhật báo cáo | ⬜ |
| 19–22/10 | Chuẩn bị báo cáo tiến độ; nhóm kiểm tra chéo `reports/error_analysis_notes.md` | ⬜ |
| **23/10** | **Báo cáo tiến độ** | ⬜ |
| 24–27/10 | Làm slide; kiểm tra tái lập trên môi trường sạch (4.4) | ⬜ |
| **26/10** | **Đóng băng thực nghiệm**: không chạy thêm gì làm đổi con số | ⬜ |
| 28–29/10 | Tập thuyết trình; chốt README | ⬜ |
| **30/10** | **Thuyết trình giữa kì** | ⬜ |

