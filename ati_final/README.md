# ViGoEmotions – Nhận diện cảm xúc tiếng Việt đa nhãn (Project 2)

Bài giữa kì và cuối kì của môn ATI: phân loại đa nhãn 27 cảm xúc + `neutral` trên bộ dữ liệu **ViGoEmotions** (20.664 bình luận mạng xã hội, EACL 2026).
Chi tiết đề bài nằm ở file PDF của môn (không đưa lên repo). Trạng thái công việc xem [PROGRESS.md](PROGRESS.md), kế hoạch xem [PIPELINE.md](PIPELINE.md), báo cáo tuần ở [reports/](reports/).

## ⚠️ Quy tắc về dữ liệu (license)

- Dataset chỉ dùng cho nghiên cứu phi thương mại và giáo dục, **không được phân phối lại**.
- **Không commit** dữ liệu gốc (`data/` đã nằm trong `.gitignore`), không dán câu bình luận gốc vào README, báo cáo, issue, hay output notebook.
- Repo chỉ chứa: mã nguồn, file chỉ số split (chỉ có id), bảng thống kê tổng hợp, và xác suất dự đoán của mô hình.
- Mỗi thành viên **tự tải** dataset bằng tài khoản Hugging Face của mình (xem bước 3).

## Cấu trúc thư mục

```
ati_final/
  notebooks/
    01_eda.ipynb          Phân tích dữ liệu (mục A giữa kì)
    02_tfidf_lr.ipynb     Baseline TF-IDF + One-vs-Rest Logistic Regression
    03_bilstm.ipynb       Baseline BiLSTM (sigmoid + BCE)
    04_phobert.ipynb      Bảng kết quả gộp 3 mô hình, chi phí, quét ngưỡng (đọc kết quả PhoBERT đã lưu)
    05_analysis.ipynb     Trả lời câu hỏi giữa kì 1-4 và phân tích lỗi (chỉ đọc xác suất đã lưu)
  src/
    data.py               Nạp dữ liệu từ Hugging Face, nhãn multi-hot 28 chiều, nhóm head/mid/tail
    preprocess.py         Chuẩn hoá văn bản (<br>, HTML entity, Unicode NFC)
    metrics.py            Macro/Micro-F1, F1 từng nhãn, Hamming Loss, LRAP, mAP; lưu/đọc xác suất
    segment.py            Tách từ tiếng Việt bằng VnCoreNLP (cho PhoBERT)
    train_phobert.py      Fine-tune PhoBERT-base (chạy bằng dòng lệnh, xem mục PhoBERT)
    features.py           Cờ kiểu văn bản (emoji, teencode, câu ngắn/dài) dùng cho phân tích theo nhóm
    show_cases.py         In ca lỗi (kèm văn bản gốc) ra màn hình để đọc cục bộ, KHÔNG ghi file
  splits/                 id của train/val/test (không có văn bản) + label_tiers.json
  results/
    tables/               Bảng thống kê và kết quả (.csv)
    figures/              Biểu đồ (.png)
    preds/                Xác suất dự đoán val/test của từng mô hình (.npz)
  reports/                Báo cáo tuần
  data/                   (tự sinh/tự tải, KHÔNG commit)
```

## Cài đặt

Yêu cầu: Python 3.10+ (đã chạy với 3.12).

**1. Clone repo và tạo môi trường**
```bash
git clone https://github.com/tiendat13ns/Deep-Learning.git
cd Deep-Learning/ati_final
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
```

**2. Cài thư viện**
```bash
pip install -r requirements.txt
```
Nếu có GPU NVIDIA, cài PyTorch bản CUDA thay cho bản CPU mặc định (chỉ cần cho BiLSTM/PhoBERT):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"   # phải in True
```
GPU đã dùng để phát triển: RTX 3060 Laptop 6 GB (đủ cho PhoBERT-base với fp16, `max_length` 128, batch 16).

**3. Truy cập dataset (gated)**
1. Vào https://huggingface.co/datasets/uitnlp/vigoemotions, đăng nhập và đồng ý điều khoản để xin quyền.
2. Tạo token loại *Read* tại https://huggingface.co/settings/tokens.
3. Chạy `hf auth login` rồi dán token (hoặc đặt biến môi trường `HF_TOKEN`). **Không commit token.**

Kiểm tra:
```bash
python -c "from src.data import load_all; d = load_all(); print({k: len(v) for k, v in d.items()})"
# {'train': 16531, 'val': 2066, 'test': 2067}
```
Dữ liệu được lưu trong cache của Hugging Face; `data/` chỉ là bản sao tuỳ chọn để mở xem.

## Chạy lại thực nghiệm

Chạy theo thứ tự từ thư mục `notebooks/` (notebook dùng đường dẫn tương đối `../`):

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_tfidf_lr.ipynb
```
Hoặc mở trong VS Code/Jupyter và chạy lần lượt. Các notebook tự ghi kết quả vào `results/` và `splits/`.

### PhoBERT

Cần thêm **Java (>= 8)** để chạy VnCoreNLP (`java -version`). Lần đầu chạy, `src/segment.py` tự tải jar và model tách từ (~28 MB) vào `data/vncorenlp/`; PhoBERT (`vinai/phobert-base`, ~500 MB) tự tải từ Hugging Face.
```bash
# chạy từ thư mục ati_final; mỗi lần ~12-15 phút trên RTX 3060 Laptop 6 GB (đỉnh ~3,3 GB VRAM)
python -m src.train_phobert --emoji text --loss bce        # cấu hình chính
python -m src.train_phobert --emoji keep --loss bce        # đối chứng: giữ emoji (bị <unk> trong PhoBERT)
python -m src.train_phobert --emoji text --loss pw_sqrt    # BCE có pos_weight = sqrt(#âm/#dương)
# cấu hình đã chọn: thêm --epochs 12 --patience 3 --name phobert_text_pw_sqrt_ep12 (6 epoch chưa hội tụ)
```
Tokenizer của PhoBERT đưa **mọi emoji về `<unk>`**, nên cấu hình chính thay emoji bằng tên của nó (thư viện `emoji`). Kết quả: `results/preds/phobert_*.npz`, `results/tables/phobert_*_history.csv` và `phobert_*_cost.json` (số tham số, thời gian, revision checkpoint, siêu tham số). Checkpoint lưu ở `data/ckpt/` (không commit).

## Giao thức thực nghiệm (bắt buộc tuân thủ)

| Quy tắc | Cách làm |
|---|---|
| Split | Dùng split chính thức của dataset: train 16.531 / val 2.066 / test 2.067 |
| Chọn siêu tham số và ngưỡng | **Chỉ trên val**. Test chỉ chạy một lần với cấu hình đã chốt |
| Ma trận đồng xuất hiện | Chỉ tính từ nhãn train |
| Nhóm head/mid/tail | Theo số mẫu dương ở train: head ≥ 1000, mid 750–999, tail < 750 (`src/data.py`) |
| Xác suất dự đoán | Lưu `.npz` (trước khi áp ngưỡng) vào `results/preds/` |
| Seed | `SEED = 42` trong `src/data.py` |
| Chỉ số chính | Macro-F1 và Micro-F1 (không dùng subset accuracy làm chỉ số chính) |
| Chuẩn hoá văn bản | `src/preprocess.py`; giữ emoji; không sửa teencode ở cấu hình chính |

## Quy trình làm việc nhóm

```bash
git pull                       # luôn pull trước khi làm
git checkout -b ten-ban/chu-de # làm trên nhánh riêng, ví dụ: an/bilstm
git add <file cụ thể>          # không dùng `git add .` khi chưa kiểm tra git status
git commit -m "Mô tả ngắn"
git push -u origin ten-ban/chu-de
```
- Trước khi commit, chạy `git status` và chắc chắn **không có** `data/`, file `.csv` chứa văn bản gốc, token, hay checkpoint mô hình lớn (`*.pt`, `*.pth`).
- Notebook: xoá output nào có chứa câu bình luận gốc trước khi commit.
- Mỗi mô hình mới cần: notebook riêng trong `notebooks/`, file `.npz` xác suất trong `results/preds/`, và một dòng trong bảng kết quả.

## Xử lý sự cố thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `Dataset ... is a gated dataset` | Chưa xin quyền hoặc chưa đăng nhập: làm lại bước 3 |
| `torch.cuda.is_available()` là `False` | Đang cài bản CPU: cài lại bằng lệnh có `--index-url` ở bước 2 |
| Cảnh báo symlink của `huggingface_hub` trên Windows | Chỉ là cảnh báo; có thể bỏ qua hoặc đặt `HF_HUB_DISABLE_SYMLINKS_WARNING=1` |
| `WRITEBACKIFCOPY base is read-only` khi fit LR | Không dùng `n_jobs > 1` với liblinear (notebook đã để chạy tuần tự) |
| Hết VRAM khi fine-tune PhoBERT | Giảm batch xuống 8 và dùng gradient accumulation, giữ fp16 |
