# Tổng hợp tiến độ - ViGoEmotions (Midterm)

Cập nhật: 2026-10-08

## 1. Việc đã làm

| # | Hạng mục | Kết quả | File |
|---|----------|---------|------|
| 1 | Đọc đề, paper, bản tóm tắt docx | Nắm yêu cầu giữa kì (A–E) và cuối kì (RQ1) | `02-vigoemotions-vietnamese-emotion.pdf`, `2026.eacl-long.129.pdf`, `Tom_tat_ViGoEmotions.docx` |
| 2 | Môi trường | PyTorch 2.14.1+cu126, nhận RTX 3060 Laptop 6 GB | - |
| 3 | Đăng nhập Hugging Face | Viết lại cell login chạy được ở máy (không phụ thuộc `google.colab`, sửa lỗi sai tên biến `HF_TOKEN`) | `draft.ipynb` |
| 4 | Tải dữ liệu | Dataset `uitnlp/vigoemotions` (gated) tải thành công | `src/data.py` |
| 5 | Module dữ liệu | Nạp split, chuyển nhãn sang multi-hot 28 chiều, hàm chia head/mid/tail | `src/data.py` |
| 6 | Phân tích dữ liệu (mục A) | Notebook chạy xong, xuất bảng và hình | `notebooks/01_eda.ipynb` |
| 7 | Giao thức chia tập | Dùng split chính thức, lưu chỉ số mẫu | `splits/train.csv`, `val.csv`, `test.csv`, `label_tiers.json` |
| 8 | Module chỉ số + tiền xử lý | Macro/Micro-F1, F1 từng nhãn, Hamming, LRAP, mAP; chuẩn hoá `<br>`/NFC | `src/metrics.py`, `src/preprocess.py` |
| 9 | Baseline TF-IDF + OvR LR | Test Macro-F1 0,548, Micro-F1 0,563, Hamming 0,067, LRAP 0,694; train 2,7 giây | `notebooks/02_tfidf_lr.ipynb`, `results/preds/tfidf_lr_*.npz` |

## 2. Các quyết định đã chốt

| Vấn đề | Quyết định | Lý do |
|--------|-----------|-------|
| Split | Dùng split chính thức của dataset: 16.531 / 2.066 / 2.067 | Cùng split với paper nên so sánh được; không trùng id; phân bố nhãn ổn định |
| Số nhãn | 28 cột (27 cảm xúc + `neutral`, index 27) | Theo README của dataset |
| Nhóm tần suất | Dựa trên số mẫu dương ở **train**: head ≥ 1000, mid 750–999, tail < 750 → 10 / 10 / 8 nhãn | Định nghĩa trước khi nhìn test, đúng yêu cầu cuối kì |
| Seed | 42 (`SEED` trong `src/data.py`) | Yêu cầu tái lập |
| Ma trận đồng xuất hiện | Chỉ tính trên train | Đề cấm dùng nhãn test |
| Văn bản gốc | Không in/lưu trong notebook và repo | License dataset |

## 3. Kết quả phân tích dữ liệu

| Nội dung | Số liệu |
|----------|---------|
| Tổng số mẫu | 20.664 (train 16.531, val 2.066, test 2.067) |
| Tần suất nhãn (train) | Từ 3,8% (relief) đến 17,3% (amusement); tỷ lệ max/min ≈ 4,5 |
| Head (10 nhãn) | amusement, sadness, annoyance, joy, anger, disappointment, disgust, love, caring, optimism |
| Tail (8 nhãn) | grief, fear, nervousness, remorse, realization, confusion, surprise, relief |
| Số nhãn mỗi mẫu \|Y\| (train) | 1: 5.431, 2: 7.706, 3: 2.915, 4: 438, 5: 41 |
| Cặp đi cùng nhiều nhất | disappointment–sadness (904), anger–annoyance (699), sadness–grief (620), amusement–joy (568) |
| Tương quan phi cao nhất | fear–nervousness 0,41; sadness–grief 0,38; disappointment–sadness 0,36 |
| Tương quan âm rõ nhất | amusement–sadness −0,17 |
| Độ dài văn bản | Trung vị 11 token; trung bình ≈ 16; 99% mẫu dưới ≈ 90 token |
| Đặc trưng mạng xã hội (ước lượng bằng regex) | Emoji ≈ 25%; teencode ≈ 16–17%; kéo dài ký tự ≈ 14%; không dấu < 1% |

## 4. Lưu ý và rủi ro

| Vấn đề | Ghi chú |
|--------|---------|
| Nhóm \|Y\| ≥ 4 rất nhỏ | Test chỉ có 54 mẫu 4 nhãn và 12 mẫu 5 nhãn → số liệu theo nhóm này nhiễu, cần nêu rõ khi báo cáo |
| Chỉ số teencode/emoji | Là heuristic regex và danh sách từ viết tắt tự lập, không phải nhãn chuẩn |
| `draft.ipynb` | Mới sửa cell login, chưa chạy lại; các cell sau còn bản nháp |
| Chưa commit | Các file mới chưa được commit vào git |

## 5. Chưa làm

Xem [PIPELINE.md](PIPELINE.md): BiLSTM/GRU, PhoBERT, bảng gộp kết quả, trả lời 5 câu hỏi giữa kì, phân tích lỗi, slide, README.
