# Lý thuyết Deep Learning — Tổng hợp theo các bài lab ATI

Bộ tài liệu này giải thích chi tiết phần lý thuyết đằng sau 5 bài lab (TUT01–TUT05). Mỗi kiến trúc hoặc mảng kiến thức lớn nằm trong một file riêng. Các ví dụ và số liệu được lấy từ kết quả chạy thật của các notebook.

## Mục lục

| # | File | Chủ đề | Lab |
|---|---|---|---|
| 01 | [Mạng neural & Forward pass](01_mang_neural_forward_pass.md) | neuron, lớp Linear, hàm kích hoạt, MLP, tham số và activation | TUT01 |
| 02 | [Huấn luyện mạng neural](02_huan_luyen_mang_neural.md) | dữ liệu, loss, gradient descent, backprop, Adam, vòng lặp training, overfitting | TUT02 |
| 03 | [CNN](03_cnn.md) | convolution, padding/stride, pooling, receptive field, BatchNorm, các kiến trúc CNN | TUT03 |
| 04 | [Transfer learning & Fine-tuning](04_transfer_learning.md) | pretrained model, đóng băng/mở băng, learning rate theo nhóm, BatchNorm khi đóng băng | TUT03 |
| 05 | [Đánh giá mô hình & thực nghiệm](05_danh_gia_mo_hinh.md) | precision/recall/F1, macro/weighted, MAE/RMSE/R², baseline, nhiều seed, FLOPs và latency | TUT02–04 |
| 06 | [RNN, LSTM, GRU](06_rnn_lstm_gru.md) | recurrence, BPTT, vanishing/exploding gradient, các cổng, số tham số | TUT04 |
| 07 | [Dự báo chuỗi thời gian](07_du_bao_chuoi_thoi_gian.md) | lấy sai phân, chia dữ liệu theo thời gian, cửa sổ trượt, persistence, dự báo nhiều bước | TUT04 |
| 08 | [Attention](08_attention.md) | Q/K/V, scaled dot-product, causal mask, multi-head | TUT05 |
| 09 | [Transformer & Hugging Face](09_transformer_hugging_face.md) | khối Transformer, BERT và GPT, tokenization, distillation, decoding, Hugging Face API | TUT05 |

## Thứ tự đọc đề xuất

```text
01 Mạng neural ──► 02 Huấn luyện ──┬──► 03 CNN ──► 04 Transfer learning
                                   │
                                   ├──► 05 Đánh giá (đọc song song, dùng cho mọi chương)
                                   │
                                   └──► 06 RNN/LSTM/GRU ──► 07 Chuỗi thời gian
                                                    │
                                                    └──► 08 Attention ──► 09 Transformer
```

## Các ý tưởng xuyên suốt

| Ý tưởng | Xuất hiện ở |
|---|---|
| **Chia sẻ trọng số** | CNN chia sẻ theo không gian (03); RNN chia sẻ theo thời gian (06); Transformer dùng cùng ma trận chiếu cho mọi token (08) |
| **Phi tuyến là bắt buộc** | ReLU trong MLP và CNN (01, 03); tanh/sigmoid trong RNN (06); GELU trong Transformer (09) |
| **Đường tắt cho gradient** | residual của ResNet (03); cell state của LSTM (06); residual của Transformer (09) |
| **Softmax biến điểm số thành phân phối** | lớp đầu ra khi phân loại (02); trọng số attention (08) |
| **Không rò rỉ dữ liệu** | chia trước rồi mới chuẩn hóa (02); không dùng test để chọn model (05); chia theo thời gian (07); causal mask (08) |
| **Kiến trúc và tham số là hai thứ khác nhau** | training chỉ thay đổi tham số (01, 02); transfer learning tái sử dụng tham số (04) |
| **Một con số là không đủ** | macro-F1 so với accuracy (05); nhiều seed (05); FLOPs không phải latency (05) |

## Tra cứu nhanh công thức

| Công thức | Ý nghĩa |
|---|---|
| $n_{out}\cdot n_{in} + n_{out}$ | số tham số của lớp Linear |
| $C_{out}\cdot C_{in}\cdot K^2 + C_{out}$ | số tham số của lớp Conv2d |
| $\lfloor (N + 2P - K)/S\rfloor + 1$ | kích thước output của conv/pool |
| $k\cdot(hd + h^2 + 2h)$, với $k$ = 1 / 3 / 4 | số tham số của RNN / GRU / LSTM (PyTorch) |
| $4d^2 + 4d$ | số tham số của multi-head attention |
| $\partial\,\text{BCE}/\partial z = p - y$ | gradient của BCE theo logit |
| $\text{softmax}(QK^\top/\sqrt{d_k})V$ | scaled dot-product attention |
| $F_1 = 2PR/(P+R)$ | F1-score |
| FLOPs ≈ 2 × MACs | quy ước đếm phép toán |
