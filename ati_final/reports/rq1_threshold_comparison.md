# RQ1 - So sánh ngưỡng cố định, ngưỡng chung và ngưỡng riêng từng nhãn

Ngày cập nhật: 2026-10-22. Mô hình: `phobert_text_pw_sqrt_ep12`. Không huấn luyện lại backbone.

## Giao thức

- Giữ nguyên mô hình PhoBERT đã chọn ở giữa kì theo Macro-F1 validation tại 0,5; dùng cùng xác suất cho cả ba cách giải mã.
- Giữ đủ 28 cột của dữ liệu: 27 cảm xúc và neutral. Split cố định: train 16.531, val 2.066, test 2.067.
- Lưới đặt trước, giống giữa kì: 0,05 đến 0,90, bước 0,025. Ngưỡng chung tối đa Macro-F1 val; ngưỡng riêng tối đa F1 val của từng nhãn trên cùng lưới.
- Khi hòa điểm: chọn ngưỡng gần 0,5 nhất, sau đó chọn ngưỡng nhỏ hơn. Nhãn không có mẫu dương trên val giữ 0,5.
- Ngưỡng chung đã chốt: **0.575**. Tất cả ngưỡng được ghi vào `results/rq1/calibration.json` trước khi mở file xác suất test.
- Mỗi ma trận val/test được nạp một lần trong pha đánh giá, rồi áp ba bộ ngưỡng đã cố định; không quét hoặc chọn ngưỡng bằng nhãn test.
- Chỉ đọc ID, nhãn và xác suất đã lưu; không tải dataset và không xuất văn bản gốc. Cardinality tính từ multi-hot nên không đếm trùng nhãn.

## Kết quả test

| strategy | macro_f1 | micro_f1 | hamming_loss | lrap | mAP | subset_acc |
| --- | --- | --- | --- | --- | --- | --- |
| fixed | 0.6082 | 0.6240 | 0.0583 | 0.7530 | 0.6304 | 0.2303 |
| global | 0.6125 | 0.6276 | 0.0546 | 0.7530 | 0.6304 | 0.2530 |
| per_label | 0.6142 | 0.6261 | 0.0552 | 0.7530 | 0.6304 | 0.2482 |

Ngưỡng riêng so với 0,5: +0.0060 Macro-F1 (+0.60 điểm phần trăm). So với ngưỡng chung: +0.0017 (+0.17 điểm phần trăm).
Trong lần chạy này, `per_label` có Macro-F1 test cao nhất. Đây là kết quả quan sát để báo cáo, không dùng để chọn lại ngưỡng.
So với ngưỡng 0,5: F1 của 14/28 nhãn tăng, 12/28 nhãn giảm, 2/28 nhãn giữ nguyên.

LRAP và mAP giống nhau giữa ba cách vì cùng ma trận xác suất, chỉ khác bước nhị phân hóa. Subset accuracy chỉ là chỉ số phụ.

## Trả lời RQ1

Ngưỡng 0,5 không đạt Macro-F1 validation cao nhất trên lưới của mô hình này. Ngưỡng riêng đã chọn trải từ 0.325 đến 0.875; cả global và per-label đều tăng Macro-F1 test so với 0,5.
Tuy vậy lợi ích bổ sung của per-label so với global chỉ +0.0017 Macro-F1. Global có Micro-F1 0.6276 và Hamming Loss 0.0546, so với 0.6261 và 0.0552 của per-label. Không có một cách giải mã tốt nhất ở mọi chỉ số.
Đặc biệt nhóm tail thay đổi -0.0041 Macro-F1 khi dùng per-label thay cho 0,5; kết quả này không ủng hộ kết luận ngưỡng riêng luôn giúp nhãn hiếm. Cần giữ kết luận trong phạm vi một mô hình và một seed.

## Validation và khả năng tổng quát hóa

| strategy | macro_f1 | micro_f1 | hamming_loss |
| --- | --- | --- | --- |
| fixed | 0.6030 | 0.6124 | 0.0593 |
| global | 0.6077 | 0.6173 | 0.0552 |
| per_label | 0.6258 | 0.6279 | 0.0543 |

| strategy | gap_val_minus_test_macro_f1 | gap_val_minus_test_micro_f1 |
| --- | --- | --- |
| fixed | -0.0051 | -0.0116 |
| global | -0.0048 | -0.0103 |
| per_label | 0.0116 | 0.0018 |

Gap = F1 val trừ F1 test. Việc F1 val tăng là hệ quả của tối ưu trên val; không bảo đảm tăng tương ứng trên test. Gap dương lớn hơn gợi ý khả năng quá khớp, nhưng riêng gap không chứng minh nguyên nhân.

## Kết quả theo nhóm tần suất train

| strategy | tier | n_labels | macro_f1 | micro_f1 | hamming_loss |
| --- | --- | --- | --- | --- | --- |
| fixed | head | 10 | 0.6345 | 0.6510 | 0.0861 |
| fixed | mid | 10 | 0.6030 | 0.5958 | 0.0451 |
| fixed | tail | 8 | 0.5817 | 0.5781 | 0.0402 |
| global | head | 10 | 0.6349 | 0.6498 | 0.0818 |
| global | mid | 10 | 0.6110 | 0.6078 | 0.0413 |
| global | tail | 8 | 0.5864 | 0.5841 | 0.0374 |
| per_label | head | 10 | 0.6484 | 0.6599 | 0.0799 |
| per_label | mid | 10 | 0.6091 | 0.5877 | 0.0449 |
| per_label | tail | 8 | 0.5777 | 0.5720 | 0.0372 |

Head >= 1.000 mẫu dương train, mid 750-999, tail < 750. Định nghĩa giữ nguyên từ giữa kì, không dùng nhãn test để chia nhóm.

## Kết quả theo số nhãn thật mỗi câu

| strategy | cardinality | n_samples | macro_f1 | micro_f1 | hamming_loss | samples_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed | 1 | 695 | 0.4581 | 0.5131 | 0.0525 | 0.5718 |
| fixed | 2 | 946 | 0.6060 | 0.6357 | 0.0586 | 0.6416 |
| fixed | 3 | 360 | 0.6489 | 0.6922 | 0.0647 | 0.6862 |
| fixed | >=4 | 66 | 0.6329 | 0.7014 | 0.0806 | 0.6898 |
| global | 1 | 695 | 0.4696 | 0.5321 | 0.0468 | 0.5796 |
| global | 2 | 946 | 0.6136 | 0.6396 | 0.0550 | 0.6386 |
| global | 3 | 360 | 0.6383 | 0.6851 | 0.0634 | 0.6752 |
| global | >=4 | 66 | 0.5953 | 0.6709 | 0.0839 | 0.6578 |
| per_label | 1 | 695 | 0.4835 | 0.5303 | 0.0483 | 0.5879 |
| per_label | 2 | 946 | 0.6164 | 0.6406 | 0.0553 | 0.6400 |
| per_label | 3 | 360 | 0.6379 | 0.6800 | 0.0637 | 0.6719 |
| per_label | >=4 | 66 | 0.6016 | 0.6781 | 0.0812 | 0.6672 |

Các CSV tổng hợp và phân nhóm còn có delta Macro-F1, Micro-F1 và Hamming Loss so với fixed. Delta Hamming âm là cải thiện.

## Các nhãn thay đổi nhiều nhất

| label | per_label_threshold | test_f1_fixed | test_f1_per_label | test_gain_per_label_vs_fixed |
| --- | --- | --- | --- | --- |
| love | 0.8000 | 0.6573 | 0.7405 | 0.0832 |
| caring | 0.7500 | 0.6121 | 0.6774 | 0.0653 |
| desire | 0.4000 | 0.4410 | 0.4955 | 0.0545 |
| grief | 0.8750 | 0.5844 | 0.6347 | 0.0504 |
| approval | 0.7000 | 0.5985 | 0.6356 | 0.0371 |

| label | per_label_threshold | test_f1_fixed | test_f1_per_label | test_gain_per_label_vs_fixed |
| --- | --- | --- | --- | --- |
| curiosity | 0.7750 | 0.6522 | 0.6054 | -0.0468 |
| remorse | 0.7250 | 0.7135 | 0.6748 | -0.0387 |
| pride | 0.6250 | 0.7250 | 0.6933 | -0.0317 |
| relief | 0.7750 | 0.6154 | 0.5841 | -0.0313 |
| disappointment | 0.6000 | 0.5073 | 0.4839 | -0.0234 |

## Giới hạn

- Test đã được xem và phân tích trong giữa kì. Đây là so sánh hậu nghiệm trên dự đoán giữ lại, không phải một tập test mới hoàn toàn chưa được xem. Thuật toán và lưới lần này được cố định trước khi đánh giá.
- Một mô hình, một seed 42; chưa có sai số giữa nhiều lần huấn luyện hoặc kiểm định ý nghĩa thống kê. Không kết luận chắc về chênh lệch nhỏ.
- Ngưỡng chỉ tốt nhất trên lưới đã công bố, không phải tối ưu liên tục. Mỗi nhãn thêm một ngưỡng có thể quá khớp val, nhất là nhãn ít mẫu dương.
- Không áp quy tắc bắt buộc ít nhất một nhãn hay loại trừ neutral: giữ cùng giao thức sigmoid độc lập cho cả ba cách.
- Các nhóm cardinality có ít mẫu: đặc biệt nhóm >=4 chỉ có 66 câu test; Macro-F1 nhóm tính đủ 28 nhãn (zero_division=0), nên cần đọc cùng Micro-F1 và support.

## Sản phẩm và cách chạy lại

```bash
cd ati_final
pip install -r requirements-analysis.txt
python -m src.analyze_thresholds --mode calibrate
python -m src.analyze_thresholds --mode evaluate
```

Có thể chạy gộp bằng `python -m src.analyze_thresholds` hoặc notebook `06_thresholds.ipynb`.

- `results/rq1/calibration.json`: ngưỡng đã cố định, lưới, tie rule, nhãn, tần suất và hash nguồn val.
- `results/rq1/evaluation.json`: hash nguồn, checkpoint/revision, cấu hình gốc và phiên bản thư viện phân tích.
- `results/rq1/tables/`: chỉ số tổng hợp, F1/precision/recall/AP từng nhãn, so sánh từng nhãn, head/mid/tail, cardinality, profile ngưỡng và gap val-test.
- `results/rq1/preds/`: hai NPZ chứa ID, nhãn thật, xác suất, pred_fixed/pred_global/pred_per_label, cardinality và ba bộ ngưỡng; mở được với allow_pickle=False.
- `results/rq1/figures/`: biểu đồ chỉ số tổng hợp, ngưỡng từng nhãn và F1 theo nhóm tần suất.
- Thời gian/số tham số gốc giữ trong `strategy_metrics.csv`: mọi cách giải mã dùng cùng backbone; thời gian train gốc và thời gian đã hiệu chỉnh từ giữa kì được ghi riêng.
