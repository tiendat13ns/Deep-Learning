# 05 — Đánh giá mô hình, chi phí tính toán và phương pháp thực nghiệm

> **Lab tương ứng:** TUT02 (threshold), TUT03 Part 5 và Exercise 4–5 (metrics, FLOPs, latency), TUT04 (MAE/RMSE/R², baseline, nhiều seed, so sánh có kiểm soát)
> **Ý tưởng chính:** Một con số accuracy là **không đủ**. Cần chọn đúng metric, có baseline để so sánh, thí nghiệm có kiểm soát, và báo cáo cả **chi phí tính toán**.

---

## Phần A — Metric cho bài toán phân loại

### 1. Confusion matrix (ma trận nhầm lẫn)

Với bài toán nhị phân, chọn một lớp làm "dương tính" (positive):

| | Dự đoán **Dương** | Dự đoán **Âm** |
|---|---|---|
| **Thật Dương** | TP (True Positive) | FN (False Negative): bỏ sót |
| **Thật Âm** | FP (False Positive): báo nhầm | TN (True Negative) |

Với $K$ lớp, confusion matrix có kích thước $K\times K$: hàng là lớp thật, cột là lớp dự đoán. Đường chéo là các dự đoán đúng, còn các ô ngoài đường chéo cho biết **lớp nào bị nhầm sang lớp nào**. Ví dụ trong TUT03, cat và dog hay bị nhầm lẫn với nhau.

### 2. Các metric cơ bản

$$
\text{Accuracy} = \frac{TP + TN}{\text{tổng số mẫu}}
$$

$$
\text{Precision} = \frac{TP}{TP + FP} \qquad \text{(trong các mẫu đoán là dương, bao nhiêu mẫu đúng?)}
$$

$$
\text{Recall} = \frac{TP}{TP + FN} \qquad \text{(trong các mẫu thật sự dương, tìm ra được bao nhiêu?)}
$$

$$
F_1 = 2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision} + \text{Recall}} \qquad \text{(trung bình điều hòa)}
$$

**Ví dụ:** TP = 40, FP = 10, FN = 20, TN = 130 (tổng 200 mẫu):
- Accuracy = 170/200 = **0,85**
- Precision = 40/50 = **0,80**
- Recall = 40/60 = **0,667**
- F1 = 2(0,8)(0,667)/(0,8 + 0,667) = **0,727**

F1 dùng **trung bình điều hòa** nên bị kéo về phía giá trị nhỏ hơn. Một model có precision 1,0 nhưng recall 0,1 sẽ chỉ có F1 ≈ 0,18. Muốn F1 cao thì **cả hai** đều phải cao.

| Khi nào ưu tiên | Ví dụ |
|---|---|
| **Recall** (bỏ sót thì tốn kém) | chẩn đoán ung thư, phát hiện gian lận |
| **Precision** (báo nhầm thì tốn kém) | lọc spam (không muốn email quan trọng bị chuyển vào spam), gợi ý sản phẩm |

### 3. Trung bình metric cho nhiều lớp

| Cách | Công thức | Ý nghĩa |
|---|---|---|
| **Macro** | $\frac{1}{K}\sum_k F_1^{(k)}$ | mỗi lớp có trọng số **như nhau**, nên lớp nhỏ được coi trọng ngang lớp lớn |
| **Weighted** | $\sum_k \frac{n_k}{N}F_1^{(k)}$ | trọng số theo **support** (số mẫu) của từng lớp, nên bị lớp đông chi phối |
| **Micro** | tính TP, FP, FN trên toàn bộ rồi mới tính metric | với single-label, micro-F1 = accuracy |

Khi các lớp **cân bằng**, macro bằng weighted. Ví dụ TUT03 có 200 ảnh test mỗi lớp, macro-F1 = weighted-F1 = 0,855.

**Classification report của TUT03 (test set cân bằng):**

| Lớp | Precision | Recall | F1 |
|---|---|---|---|
| airplane | 0,909 | 0,895 | 0,902 |
| automobile | 0,950 | 0,945 | 0,947 |
| cat | 0,752 | 0,805 | 0,778 |
| dog | 0,816 | 0,775 | 0,795 |

Accuracy 0,855 che giấu một điều: model phân biệt **cat và dog** kém hơn nhiều so với phân biệt máy bay và ô tô.

### 4. Accuracy che giấu lớp thiểu số (TUT03, Exercise 4)

Train với dữ liệu mất cân bằng (chỉ có 80 ảnh chó, so với 500 ảnh cho mỗi lớp khác), rồi test trên tập cũng mất cân bằng (32 ảnh chó, 200 ảnh cho mỗi lớp khác):

| Train | Test | Accuracy | Dog recall | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|---|
| cân bằng | mất cân bằng | 0,864 | 0,844 | 0,798 | 0,874 |
| **dog = 80** | **mất cân bằng** | **0,896** ↑ | **0,563** ↓↓ | 0,824 | **0,895** ↑ |

Model train lệch có **accuracy và weighted-F1 cao nhất**, nhưng chỉ nhận đúng 18/32 ảnh chó. Chó chỉ chiếm 5% tập test, nên đoán sai ảnh chó gần như không ảnh hưởng tới accuracy.

Cái để lộ vấn đề rõ nhất là **recall của từng lớp** (dog recall giảm từ 0,844 xuống 0,563). Một tín hiệu khác là khoảng cách giữa các metric **trong cùng một model**: macro-F1 0,824 thấp hơn hẳn accuracy 0,896, nghĩa là có lớp đang làm kém.

Tuy nhiên, nếu so sánh **giữa hai model**, macro-F1 của model train lệch (0,824) lại *cao hơn* model train cân bằng (0,798). Lý do: model train cân bằng đoán "dog" khá nhiều, nên khi tập test chỉ có 32 ảnh chó, precision của lớp dog bị kéo xuống. Bài học là **không có một con số tổng hợp nào đủ**: phải xem precision và recall của **từng lớp** cùng với confusion matrix.

> **Quy tắc:** với dữ liệu mất cân bằng, phải báo cáo precision/recall/F1 của **từng lớp**, macro-F1, và confusion matrix, không chỉ accuracy.

Một **baseline ngây thơ** cũng có thể đạt accuracy cao trên dữ liệu lệch. Ví dụ trong TUT02, luôn đoán PASS đã đạt 63,6%. Model với lr = 0,0001 cũng chỉ đạt 0,64, tức là **chưa học được gì**.

### 5. Threshold và sự đánh đổi precision–recall

Với model xuất ra xác suất, nhãn dự đoán phụ thuộc vào ngưỡng $t$:
- giảm $t$ thì nhiều mẫu được đoán là dương hơn: **recall tăng, precision giảm**;
- tăng $t$ thì ngược lại.

TUT02: ngưỡng 0,3 / 0,5 / 0,7 cho lần lượt 176 / 156 / 137 dự đoán PASS. Đổi ngưỡng **không làm thay đổi model**.

- **Đường Precision–Recall:** vẽ (recall, precision) khi quét $t$ từ 0 tới 1.
- **ROC curve:** vẽ TPR (= recall) theo FPR $= FP/(FP+TN)$. **AUC** (diện tích dưới đường ROC) bằng xác suất model xếp hạng một mẫu dương ngẫu nhiên cao hơn một mẫu âm ngẫu nhiên. AUC không phụ thuộc vào việc chọn ngưỡng.

---

## Phần B — Metric cho bài toán hồi quy (TUT04)

| Metric | Công thức | Đặc điểm |
|---|---|---|
| **MAE** | $\frac{1}{N}\sum\lvert y - \hat y\rvert$ | cùng đơn vị với $y$ (ppm), dễ hiểu, ít nhạy với outlier |
| **MSE** | $\frac{1}{N}\sum(y-\hat y)^2$ | phạt nặng sai số lớn, khả vi, thường dùng làm loss |
| **RMSE** | $\sqrt{\text{MSE}}$ | cùng đơn vị với $y$, nhạy với sai số lớn hơn MAE; luôn có RMSE ≥ MAE |
| **R²** | $1 - \dfrac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}$ | tỉ lệ phương sai được giải thích; bằng 1 là hoàn hảo, bằng 0 là ngang với việc đoán trung bình |

**Lưu ý về R² trong TUT04:** R² trên **mức CO₂** đạt khoảng 0,98–0,99 cho **mọi** model, kể cả baseline đơn giản (0,980). Lý do là chuỗi có xu hướng tăng mạnh, nên chỉ cần "theo xu hướng" là đã giải thích được gần hết phương sai. R² cao ở đây **không** có nghĩa là model tốt. Metric có giá trị chẩn đoán hơn là **RMSE trên mức thay đổi** (Delta RMSE).

---

## Phần C — Baseline

Một model chỉ có ý nghĩa khi **so sánh với một cách làm đơn giản**:

| Bài toán | Baseline |
|---|---|
| Phân loại | luôn đoán lớp chiếm đa số; đoán ngẫu nhiên theo tỉ lệ các lớp |
| Hồi quy | luôn đoán giá trị trung bình |
| Chuỗi thời gian | **persistence**: giá trị tuần sau bằng giá trị tuần này |

Kết quả TUT04: Delta RMSE của RNN/LSTM/GRU là 0,428 / 0,405 / 0,400, so với **0,495 của persistence**. Cả ba model đều thắng baseline, và GRU giảm sai số khoảng 19%.

---

## Phần D — Phương pháp thực nghiệm

### 6. Train / validation / test và data leakage

- Mọi quyết định (chọn epoch, learning rate, số block mở băng, kích thước ảnh, checkpoint) đều dựa trên **validation**.
- Test chỉ dùng **một lần** ở cuối. Nếu xem test sau mỗi epoch để chọn model, test đã trở thành một validation set thứ hai, và điểm báo cáo sẽ **lạc quan giả tạo**.
- Các thống kê tiền xử lý (mean/std của scaler) chỉ được tính trên tập train.

### 7. Chỉ thay đổi một biến mỗi lần

Nếu đổi cùng lúc learning rate và kiến trúc, ta sẽ không biết thay đổi nào gây ra khác biệt. TUT02, TUT03 và TUT04 đều yêu cầu thay đổi đúng **một** biến và giữ nguyên mọi thứ còn lại: dữ liệu, seed, số epoch.

Nên viết **giả thuyết trước khi chạy**, theo mẫu: "Tôi dự đoán X sẽ tăng/giảm vì Y". Cách này tránh việc giải thích kết quả theo hướng có lợi sau khi đã thấy số liệu.

### 8. Nhiều seed: tách tín hiệu khỏi nhiễu

Kết quả của một lần chạy phụ thuộc vào cách khởi tạo trọng số và thứ tự dữ liệu. Nên chạy nhiều seed và báo cáo **mean ± std**.

**TUT04, Exercise 3 (5 seed):**

| | Delta RMSE (mean ± std) |
|---|---|
| RNN | 0,4202 ± 0,0024 |
| LSTM | **0,4060 ± 0,0051** |
| GRU | 0,4114 ± 0,0112 |

Ở seed 42, GRU thắng (0,3998). Nhưng tính trung bình thì **LSTM tốt hơn**. Chênh lệch giữa LSTM và GRU (≈ 0,005) **nhỏ hơn** độ dao động của GRU giữa các lần chạy (≈ 0,011). Vì vậy **không thể kết luận** kiến trúc nào tốt hơn. Kết luận duy nhất đủ vững là "RNN kém hơn cả hai".

> Nếu chênh lệch giữa hai model nhỏ hơn độ dao động giữa các seed, thì đó có thể chỉ là nhiễu.

### 9. So sánh công bằng: cùng ngân sách tham số

So sánh các kiến trúc ở **cùng hidden size** không công bằng, vì LSTM (2.617 tham số) lớn hơn GRU (1.969) và RNN (673). TUT04, Exercise 2 chọn hidden size sao cho cả ba có khoảng 2.000 tham số. Khi đó RNN cải thiện rõ (0,428 → 0,419) và khoảng cách giữa các model thu hẹp lại.

### 10. Chẩn đoán underfit và overfit qua loss

| Dấu hiệu | Chẩn đoán |
|---|---|
| train loss cao **và** val loss cao | underfit |
| train loss thấp, val loss thấp và gần train loss | vừa khớp |
| train loss tiếp tục giảm, val loss đi ngang rồi **tăng** | overfit, nên dừng ở điểm val loss thấp nhất |

**TUT04, Exercise 5 (thay đổi hidden size):**
- hidden 8: cả hai loss đều cao, nên underfit;
- hidden 32: hợp lý;
- RNN-128: val loss tốt nhất ở epoch 5 rồi không cải thiện thêm, nên early stopping dừng ở epoch 8; có dấu hiệu overfit hoặc mất ổn định.

**TUT03, Exercise 3:** mở 6 block thì val loss tăng, trong khi train loss vẫn giảm, nên overfit.

---

## Phần E — Chi phí tính toán

### 11. Các đại lượng đo chi phí

| Đại lượng | Đo cái gì | Ghi chú |
|---|---|---|
| **Parameters** | số giá trị học được mà model lưu trữ | quyết định dung lượng model và bộ nhớ chứa trọng số |
| **Model size** | dung lượng file checkpoint | ≈ số tham số × 4 byte (float32), cộng thêm buffer như running stats của BN. TUT03: 5,93 MB |
| **MACs** | số phép nhân-cộng (multiply–accumulate) cho **một** input | phụ thuộc vào kích thước input |
| **FLOPs** | số phép toán dấu phẩy động | theo một quy ước phổ biến, FLOPs ≈ 2 × MACs, vì mỗi MAC gồm 1 phép nhân và 1 phép cộng |
| **Latency** | thời gian xử lý một input (batch 1) | phụ thuộc phần cứng và phần mềm |
| **Throughput** | số mẫu xử lý được mỗi giây (batch lớn) | |

> Các công cụ khác nhau dùng các quy ước đếm FLOPs khác nhau. Chỉ so sánh khi **cùng công cụ đo, cùng kích thước input và cùng quy ước**. TUT03 dùng `thop`, công cụ này báo MACs.

### 12. Vì sao MACs tỉ lệ với độ phân giải ảnh

Mỗi kernel conv được tính tại **mọi vị trí** $(h, w)$, nên MACs của một lớp conv là $K^2 \cdot C_{in}\cdot C_{out}\cdot H_{out}\cdot W_{out}$, **tỉ lệ thuận với $H\times W$**. Số tham số thì **không đổi**.

**TUT03, Exercise 5 (MobileNetV3-Small, GPU T4):**

| Kích thước ảnh | Số pixel (so với 128) | GMACs | GMACs (so với 128) | Median latency |
|---|---|---|---|---|
| 96 | 0,5625× | 0,0121 | 0,585× | 5,2 ms |
| 128 | 1× | 0,0208 | 1× | 15,6 ms |
| 160 | 1,5625× | 0,0319 | 1,534× | 16,8 ms |

Tỉ lệ GMACs gần đúng bằng tỉ lệ số pixel. Sai lệch nhỏ đến từ những phần không phụ thuộc vào độ phân giải (classifier sau GAP) và từ việc làm tròn kích thước sau các lớp có stride.

### 13. FLOPs không phải là latency

Từ 128 lên 160 px, GMACs tăng 53% nhưng latency chỉ tăng khoảng 8%. Cùng kích thước 128, latency đo ở Part 5 là 7,3 ms nhưng ở Exercise 5 lại là 15,6 ms. Lý do:

- **Băng thông bộ nhớ:** nhiều lớp (như depthwise conv) có ít FLOPs nhưng bị giới hạn bởi tốc độ đọc/ghi bộ nhớ.
- **Overhead cố định:** mỗi lớp cần một lần khởi chạy kernel trên GPU và cần đồng bộ CPU–GPU. Với batch 1, các overhead này chiếm phần lớn thời gian.
- **Mức độ song song:** GPU chỉ được tận dụng hết khi có đủ việc để làm song song.
- Các phép toán không được tính vào FLOPs: activation, reshape, concat.
- Mức tối ưu của thư viện (cuDNN), kiểu dữ liệu (FP32/FP16), và nhiễu do dùng chung GPU trên Colab.

Vì vậy **hai model có GFLOPs giống nhau vẫn có thể có latency rất khác nhau**.

**Cách đo latency đúng** (như trong TUT03):
1. `model.eval()` và dùng `torch.no_grad()`;
2. chạy vài lần **warm-up** để khởi tạo cache và cuDNN;
3. gọi `torch.cuda.synchronize()` trước và sau mỗi lần đo, vì GPU chạy bất đồng bộ;
4. lặp lại nhiều lần và báo cáo **median** và **p90** (median ít bị ảnh hưởng bởi các lần đo bất thường).

### 14. Cân bằng giữa độ chính xác và hiệu năng

- 160 px: tốn thêm khoảng 53% phép tính, tăng accuracy khoảng 4 điểm.
- 96 px: tiết kiệm khoảng 42% phép tính, giảm accuracy khoảng 2 điểm.

Lựa chọn tốt nhất phụ thuộc vào ràng buộc triển khai: chạy trên điện thoại, chạy thời gian thực, hay chạy trên server.

---

## 15. Checklist trước khi kết luận "model A tốt hơn model B"

- [ ] Cách chia dữ liệu đúng (với chuỗi thời gian thì phải theo thứ tự thời gian); không có leakage.
- [ ] Tiền xử lý chỉ được fit trên tập train.
- [ ] Hai model nhận cùng thông tin đầu vào và cùng cài đặt tối ưu.
- [ ] Có baseline đơn giản để so sánh.
- [ ] Dùng metric phù hợp (macro-F1 hoặc per-class khi dữ liệu lệch; Delta RMSE thay vì R² khi chuỗi có xu hướng).
- [ ] Báo cáo cả độ chính xác **và** chi phí (tham số, FLOPs, latency).
- [ ] Chạy nhiều seed và báo cáo mean ± std.
- [ ] Giới hạn kết luận trong phạm vi dữ liệu và bài toán đã thử.

---

## 16. Câu hỏi ôn tập

<details>
<summary>1. Macro-F1 cho biết điều gì mà accuracy có thể che giấu?</summary>

Hiệu năng của **từng lớp**, kể cả lớp ít mẫu. Accuracy bị lớp đông mẫu chi phối.
</details>

<details>
<summary>2. Vì sao không được xem test set sau mỗi epoch?</summary>

Vì khi dùng test để ra quyết định, ta đã gián tiếp fit theo test. Điểm báo cáo khi đó không còn là ước lượng không thiên lệch cho dữ liệu mới.
</details>

<details>
<summary>3. Hai model có GFLOPs gần nhau, vì sao latency lại khác nhau?</summary>

Vì latency còn phụ thuộc vào truy cập bộ nhớ, overhead mỗi lớp, mức song song, các phép toán không được đếm vào FLOPs, phần cứng và mức tối ưu của thư viện.
</details>

<details>
<summary>4. LSTM hơn GRU 0,005 RMSE ở một seed. Có thể kết luận LSTM tốt hơn không?</summary>

Chưa thể. Cần chạy nhiều seed. Nếu chênh lệch nhỏ hơn độ lệch chuẩn giữa các lần chạy thì đó có thể chỉ là nhiễu.
</details>

<details>
<summary>5. Vì sao R² ≈ 0,98 trên mức CO₂ không chứng minh model tốt?</summary>

Vì chuỗi có xu hướng mạnh. Baseline persistence cũng đạt R² ≈ 0,98. Cần so sánh Delta RMSE với baseline.
</details>

---

**Trước:** [04 — Transfer learning](04_transfer_learning.md) · **Tiếp theo:** [06 — RNN, LSTM, GRU](06_rnn_lstm_gru.md)
