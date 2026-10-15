# Ý chính báo cáo miệng tiến độ – 23/10/2026

Số liệu cập nhật ngày 12/10. **Trước 23/10 cần sửa lại** các mục có dấu ★ nếu có thay đổi (kết quả thử lr, đánh giá định tính, nhóm kiểm tra chéo ghi chú lỗi). Nói khoảng 5–7 phút.

## 1. Mở đầu (30 giây)
- Bài toán: nhận diện cảm xúc tiếng Việt **đa nhãn** trên ViGoEmotions: 20.664 bình luận mạng xã hội, 27 cảm xúc + neutral (28 nhãn). Một câu có thể có nhiều cảm xúc cùng lúc.
- Vì đa nhãn nên chỉ số chính là **Macro-F1 và Micro-F1**, không dùng độ chính xác "đúng cả 28 nhãn".

## 2. Em đã làm những gì, đến đâu
| Hạng mục của đề giữa kì | Trạng thái |
|---|---|
| A. Phân tích dữ liệu (phân bố nhãn, đồng xuất hiện nhãn, độ dài, emoji/teencode, cách chia tập) | Xong phần thống kê; ★ phần đánh giá định tính đang bổ sung |
| B. Ba mô hình: TF-IDF + Logistic Regression, BiLSTM, PhoBERT | Xong cả ba |
| C. Chỉ số: Macro/Micro-F1, F1 từng nhãn, Hamming Loss, LRAP/mAP, số tham số, thời gian train | Xong |
| D. Năm câu hỏi | Đã có câu trả lời cho cả năm |
| E. Phân tích lỗi, README, notebook chạy lại được | Xong; ★ ghi chú lỗi còn cần kiểm tra chéo |
| Còn lại | Slide thuyết trình 30/10, mục câu hỏi nghiên cứu cuối kì (RQ1 có sẵn trong đề) |

## 3. Kết quả chính (test, ngưỡng 0,5, một seed)
| Mô hình | Macro-F1 | Micro-F1 | Số tham số | Thời gian train |
|---|---|---|---|---|
| TF-IDF + LR | 0,548 | 0,563 | 1,05 triệu | 2,7 giây (CPU) |
| BiLSTM | 0,512 | 0,541 | 1,38 triệu | 151 giây |
| **PhoBERT-base** | **0,608** | **0,624** | 135 triệu | ~20 phút |

Các phát hiện nên nói:
1. **PhoBERT tốt nhất**, hơn TF-IDF 0,06 Macro-F1 và tốt hơn ở 24/28 nhãn. Hơn BiLSTM ở cả 8 nhóm kiểu câu (câu ngắn, dài, có emoji, teencode...), rõ nhất ở **câu dài và câu có teencode**; riêng nhóm nhiều emoji (188 câu) khoảng tin cậy chứa 0 nên chưa chắc chắn.
2. **BiLSTM thua cả TF-IDF.** Giả thuyết: phải học embedding từ đầu trên chỉ 16,5 nghìn câu (chưa kiểm chứng).
3. **Nhãn khó nhất:** disapproval, desire, neutral, realization. **F1 không phụ thuộc rõ vào tần suất nhãn** (Spearman ≈ 0,2, không có ý nghĩa thống kê).
4. **Mô hình giữ được cấu trúc phụ thuộc giữa các nhãn** (cặp hay đi cùng nhau như sadness–grief, fear–nervousness), nhưng **dự đoán thừa nhãn** (khoảng 2,4 nhãn mỗi câu, trong khi thật là 1,91).
5. **Về ngưỡng 0,5:** với mô hình huấn luyện đủ lâu, chọn một ngưỡng chung chỉ tăng thêm khoảng +0,004 đến +0,009 Macro-F1. Ngưỡng riêng từng nhãn chưa thử (đó là câu hỏi RQ1 của cuối kì).
6. **Đọc 70 ca lỗi:** phần lớn là nhầm giữa các nhãn gần nghĩa, nhiều ca dự đoán thừa thực ra hợp lý về nghĩa. Mô hình còn bị đánh lừa bởi từ khoá ("muốn" → desire, "yêu" → love) và bị mỉa mai, emoji gây nhầm.

## 4. Khó khăn gặp phải (nói thẳng)
- Dataset bị khoá, phải xin quyền Hugging Face; PyTorch ban đầu là bản CPU nên cài lại bản GPU; vài lỗi tương thích thư viện (xung đột phiên bản `transformers` và `huggingface_hub`, lỗi chạy song song của Logistic Regression).
- **PhoBERT đưa mọi emoji về ký hiệu từ lạ (`<unk>`)**, trong khi emoji xuất hiện ở khoảng 25% câu. Em đã chuyển emoji thành chữ, so với giữ nguyên thì tốt hơn khoảng 0,017 Macro-F1 (chỉ đo ở một cấu hình, một seed).
- Laptop vào chế độ ngủ giữa lúc train, làm thời gian ghi nhận sai; em đã hiệu chỉnh và ghi chú rõ.

## 5. Sai sót và những chỗ đã đổi hướng (phần thầy quan tâm)
- **Sai sót lớn nhất:** ban đầu train PhoBERT 6 epoch và rút ra "ngưỡng chung giúp +0,065 Macro-F1". Sau đó phát hiện **cả ba lần chạy chưa hội tụ**. Tăng lên 12 epoch thì mức tăng chỉ còn +0,009, và BCE thường tự tăng từ 0,508 lên 0,599. Bài học: không kết luận về ngưỡng khi mô hình chưa hội tụ.
- **Đổi hướng ở loss:** `pos_weight` đầy đủ làm BiLSTM tệ đi, nên đổi sang trọng số nhẹ hơn (căn bậc hai) cho PhoBERT. Điều này chỉ khác biệt nhỏ so với BCE thường ở 12 epoch (+0,009) nên chưa kết luận chắc loại nào tốt hơn.
- **Mức tinh chỉnh không đồng đều:** TF-IDF có lưới 12 cấu hình, còn BiLSTM và PhoBERT chủ yếu chỉ đổi loss và số epoch. Vì vậy so sánh giữa mô hình chưa phải "mỗi mô hình ở trạng thái tối ưu". ★ Em dự định thử lr trong 15–19/10 để bù.
- **Mất cân bằng nhãn chỉ ở mức vừa** (từ 3,8% đến 17,3%, tỉ lệ khoảng 4,5 lần), nên em kỳ vọng ngưỡng riêng từng nhãn ở cuối kì sẽ không cải thiện quá lớn.

## 6. Hạn chế thành thật
- Mỗi cấu hình mới chạy **một seed**, chưa có độ lệch chuẩn; chênh lệch nhỏ giữa các cấu hình có thể nằm trong nhiễu.
- Đánh giá theo kiểu văn bản dùng cờ regex (ước lượng), các nhóm chồng lấn nhau.
- Phân tích lỗi mới đọc 70 ca và là nhận định chủ quan.

## 7. Kế hoạch đến 30/10
- 15–19/10: thử lr cho PhoBERT và BiLSTM (chọn theo val), chạy lại phân tích.
- ★ Hoàn thiện đánh giá định tính; viết mục RQ1 (có sẵn trong đề) kèm bằng chứng sơ bộ.
- 24–27/10: làm slide, kiểm tra tái lập trên môi trường sạch. **26/10 đóng băng thực nghiệm.**

## 8. Câu thầy có thể hỏi, gợi ý trả lời
| Câu hỏi | Gợi ý trả lời |
|---|---|
| Sao không dùng độ chính xác "đúng cả 28 nhãn"? | Quá khắt khe cho đa nhãn; một nhãn sai là tính sai cả câu. Dùng Macro/Micro-F1 như đề yêu cầu. Subset accuracy vẫn tính để tham khảo. |
| Sao BiLSTM thua TF-IDF? | Giả thuyết: ít dữ liệu để học embedding từ đầu. Chưa kiểm chứng; chưa thử embedding tiền huấn luyện hay GRU. |
| Sao dùng split chính thức, không tự chia? | Trùng split của paper nên so sánh được; đã kiểm tra không trùng id và phân bố nhãn giữa các tập ổn định (chênh tối đa ~2,3 điểm phần trăm). |
| Kết quả có tin cậy không? | Một seed, nên chênh lệch nhỏ có thể là nhiễu. Chênh PhoBERT–BiLSTM đã có khoảng tin cậy bootstrap (không chứa 0 ở hầu hết nhóm). Nếu còn thời gian sẽ chạy thêm seed. |
| Ngưỡng 0,5 có tối ưu không? | Với một ngưỡng chung, cải thiện rất nhỏ ở mô hình hội tụ. Ngưỡng riêng từng nhãn chưa thử, đó là RQ1 ở cuối kì. |
| Có dùng test để chọn gì không? | Không. Mọi lựa chọn (siêu tham số, ngưỡng, mô hình) chỉ dựa trên val; test chạy một lần. |
