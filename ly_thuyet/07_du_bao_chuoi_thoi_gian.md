# 07 — Dự báo chuỗi thời gian (Time-series Forecasting)

> **Lab tương ứng:** TUT04, Part B (dự báo nồng độ CO₂ hằng tuần ở Mauna Loa bằng RNN/LSTM/GRU) và Exercise 1, 4
> **Ý tưởng chính:** Dự báo chuỗi thời gian có những quy tắc riêng. Quan trọng nhất là **tôn trọng thứ tự thời gian**: không được để thông tin từ tương lai lọt vào quá trình train.

---

## 1. Chuỗi thời gian là gì?

Chuỗi thời gian là dãy quan sát $y_1, y_2, \dots, y_T$ được đo theo thời gian, thường cách đều nhau (hằng ngày, hằng tuần, ...). Khác với dữ liệu dạng bảng thông thường, **các mẫu không độc lập với nhau**: giá trị hôm nay liên quan chặt chẽ tới hôm qua. Hiện tượng này gọi là **tự tương quan (autocorrelation)**.

### Các thành phần

$$
y_t = \underbrace{T_t}_{\text{xu hướng}} + \underbrace{S_t}_{\text{mùa vụ}} + \underbrace{\varepsilon_t}_{\text{nhiễu}}
$$

| Thành phần | Ví dụ với CO₂ ở Mauna Loa |
|---|---|
| **Trend (xu hướng)** | tăng dần từ ≈ 316 ppm (1958) lên ≈ 370 ppm (2001) |
| **Seasonality (mùa vụ)** | dao động theo năm (chu kỳ 52 tuần): thực vật ở Bắc bán cầu hấp thụ CO₂ vào mùa hè |
| **Noise (nhiễu)** | biến động ngẫu nhiên giữa các tuần |

**Dữ liệu của lab:** 2.284 quan sát hằng tuần (1958-03-29 → 2001-12-29) lấy từ `statsmodels`. Dữ liệu gốc có 59 tuần bị thiếu, được điền bằng **nội suy theo thời gian**.

> Trong một hệ thống thật, cách điền giá trị thiếu phải chỉ dùng thông tin **có sẵn tại thời điểm dự báo**. Nội suy dùng cả điểm phía sau, nên chỉ chấp nhận được cho một bài lab trên dữ liệu lịch sử cố định.

---

## 2. Tính dừng và lấy sai phân (differencing)

Một chuỗi được gọi là **dừng (stationary)** nếu các tính chất thống kê của nó (trung bình, phương sai) không đổi theo thời gian. Chuỗi CO₂ có xu hướng tăng nên **không dừng**.

Nếu dự báo **trực tiếp mức CO₂**, model có thể trông rất chính xác chỉ vì nó "đi theo xu hướng". Ví dụ, đoán "tuần sau bằng tuần này" đã cho R² ≈ 0,98. Để so sánh các kiến trúc một cách có ý nghĩa, lab dự báo **mức thay đổi**:

$$
\Delta y_t = y_t - y_{t-1}
$$

Sau khi lấy sai phân, chuỗi gần như dừng (dao động quanh một mức trung bình nhỏ và dương). Sau khi dự báo, ta **khôi phục** lại mức CO₂:

$$
\hat{y}_{t+1} = y_t + \widehat{\Delta y}_{t+1}
$$

Như vậy có hai cách đánh giá:
1. **Sai số trên mức thay đổi** (Delta MAE/RMSE): khó hơn và **có giá trị chẩn đoán** hơn;
2. **Sai số trên mức CO₂** (Level MAE/RMSE/R²): dễ hiểu theo đơn vị ppm.

> Với dự báo 1 bước, Level MAE bằng đúng Delta MAE, vì $\hat y_{t+1} - y_{t+1} = \widehat{\Delta y}_{t+1} - \Delta y_{t+1}$ (mức nền $y_t$ đã biết).

---

## 3. Chia dữ liệu theo thời gian

> **Không bao giờ xáo trộn ngẫu nhiên chuỗi thời gian trước khi chia.**

Nếu xáo trộn, tập train sẽ chứa các tuần **nằm sau** các tuần trong tập test. Model khi đó học được từ "tương lai", và kết quả đánh giá lạc quan giả tạo (leakage).

```text
|──────── train 70% ────────|── val 15% ──|── test 15% ──|
1958                                                   2001
```

Trong TUT04, các tập lần lượt có 1.598 / 342 / 343 giá trị thay đổi hằng tuần.

- `StandardScaler` chỉ được **fit trên tập train**, sau đó dùng cùng tham số đó cho val và test.
- Tập test là giai đoạn **mới nhất**. Đây đúng là tình huống thực tế: dự báo tương lai bằng model học từ quá khứ.
- Phân phối dữ liệu có thể **thay đổi theo thời gian** (distribution shift). Đó là một lý do khiến val loss cao hơn train loss trong TUT04.

**Mở rộng:** phương pháp **walk-forward / rolling-origin validation** lặp lại nhiều lần: train trên $[1, t]$, đánh giá trên $[t+1, t+h]$, rồi dịch $t$ về phía trước. Cách này cho ước lượng ổn định hơn so với một lần chia duy nhất.

---

## 4. Cửa sổ trượt (Sliding windows)

Mạng cần các cặp (input, target). Với `SEQ_LEN = 52` (khoảng 1 năm):

```text
input: 52 thay đổi trong quá khứ          target: thay đổi tuần tiếp theo
[x_(t-51), ..., x_t]               →     x_(t+1)
```

Một chuỗi có độ dài $N$ tạo ra $N - L$ cửa sổ (với $L$ = `SEQ_LEN`). Ví dụ tập train có $1598 - 52 = 1546$ cửa sổ.

**Cửa sổ được tạo riêng bên trong từng tập**, nên không có cửa sổ nào bắc cầu từ train sang val hoặc test. Các cửa sổ chồng lấn nhau rất nhiều (cửa sổ $i$ và $i+1$ chung 51 giá trị), nên chúng **không độc lập**.

**Shape đưa vào RNN:** `(batch, 52, 1)` (52 bước, mỗi bước 1 feature). Target có shape `(batch, 1)`.

### Chọn độ dài cửa sổ (TUT04, Exercise 1)

| SEQ_LEN | RNN | LSTM | GRU |
|---|---|---|---|
| 13 (~1 quý) | 0,4271 | 0,4248 | 0,4241 |
| 26 (~nửa năm) | 0,4204 | 0,4037 | **0,3994** |
| 52 (~1 năm) | 0,4283 | 0,4048 | 0,3998 |
| 104 (~2 năm) | 0,4484 | 0,4161 | 0,4138 |

(Delta RMSE, đơn vị ppm)

- Cửa sổ quá ngắn (13) không đủ thấy cả chu kỳ mùa.
- Cửa sổ quá dài (104) làm **cả ba** model tệ đi. **RNN tệ đi nhiều nhất**, phù hợp với vấn đề vanishing gradient. Ngoài ra, cửa sổ dài hơn cũng làm giảm số mẫu train.
- **Nhiều quá khứ hơn không có nghĩa là dự báo tốt hơn.** `SEQ_LEN` là một hyperparameter cần chọn bằng validation.

---

## 5. Baseline cho chuỗi thời gian

| Baseline | Dự báo | Khi nào mạnh |
|---|---|---|
| **Persistence (naive)** | $\hat y_{t+1} = y_t$, tức $\widehat{\Delta y} = 0$ | chuỗi biến đổi chậm |
| **Seasonal naive** | $\hat y_{t+1} = y_{t+1-52}$ (cùng tuần năm trước) | chuỗi có mùa vụ rõ |
| **Drift** | $\hat y_{t+1} = y_t + \overline{\Delta y}_{train}$ | chuỗi có xu hướng tuyến tính |
| Trung bình trượt | trung bình $k$ bước gần nhất | |

Persistence của TUT04 có Delta RMSE **0,4945**. Mọi model học được phải **thắng** con số này thì mới có giá trị.

---

## 6. Huấn luyện cho bài toán dự báo

Giao thức chung của TUT04, áp dụng giống nhau cho cả ba kiến trúc:
- hidden size 24, 1 lớp recurrent;
- Adam, lr = $10^{-3}$, loss **MSE**;
- **gradient clipping** với norm 1,0, để tránh exploding gradient;
- tối đa 12 epoch;
- **early stopping** theo val MSE (patience = 3), và khôi phục checkpoint tốt nhất.

**Đọc đường loss:** khi train ổn, train loss giảm dần, còn val loss giảm rồi chững lại. Early stopping giữ lại checkpoint có val loss thấp nhất.

> DataLoader có thể `shuffle=True` **thứ tự các cửa sổ** trong tập train. Việc này không vi phạm thứ tự thời gian, vì mỗi cửa sổ vẫn giữ nguyên thứ tự bên trong và vẫn nằm hoàn toàn trong tập train. Điều bị cấm là xáo trộn **trước khi chia** tập.

---

## 7. Dự báo nhiều bước (Multi-step forecasting)

Muốn dự báo $H$ bước tới (horizon $H$), có ba chiến lược:

| Chiến lược | Cách làm | Ưu và nhược điểm |
|---|---|---|
| **Recursive (đệ quy)** | một model 1 bước; dùng dự báo của bước trước làm input cho bước sau | chỉ cần một model; nhưng **sai số tích lũy** qua từng bước |
| **Direct** | $H$ model riêng, model thứ $h$ dự báo bước thứ $h$ | không tích lũy sai số qua input; nhưng tốn kém, và các bước dự báo không nhất quán với nhau |
| **Multi-output (MIMO)** | một model xuất ra **cả vector** $H$ giá trị | là cách TUT04 dùng: head `Linear(hidden, H)` |

**TUT04, Exercise 4 (multi-output):**
1. Target là vector $[\Delta y_{t+1}, \dots, \Delta y_{t+H}]$.
2. Head là `Linear(24, H)`.
3. Khôi phục **quỹ đạo** mức CO₂ bằng tổng cộng dồn:

$$
\hat y_{t+h} = y_t + \sum_{j=1}^{h}\widehat{\Delta y}_{t+j}
$$

Level RMSE tại bước cuối cùng của horizon:

| Horizon | RNN | LSTM | GRU | Persistence |
|---|---|---|---|---|
| 1 | 0,417 | 0,401 | 0,420 | 0,495 |
| 4 | 0,799 | **0,610** | 0,781 | 1,273 |
| 12 | 0,957 | 0,808 | 0,822 | 3,063 |

- Sai số **tăng theo horizon**, vì sai số của các thay đổi dự báo **cộng dồn** vào mức CO₂, và tương lai càng xa càng bất định.
- **RNN suy giảm nhiều nhất**, còn LSTM giữ ưu thế rõ nhất ở horizon 4.
- Persistence rất tệ ở horizon dài (3,06 ppm), vì nó giả định CO₂ không đổi trong 12 tuần, bỏ qua cả xu hướng lẫn mùa vụ.

---

## 8. Đọc kết quả cho đúng

- **Delta RMSE** là metric chính. **Level R²** rất cao với mọi model (≈ 0,98) nên ít giá trị chẩn đoán.
- **Thời gian train** chỉ là con số tương đối trên phần cứng hiện tại, không phải benchmark độc lập với phần cứng.
- **Một dataset, một cách chia, một seed** chỉ là thí nghiệm để học, chưa phải benchmark. Không kết luận "kiến trúc X luôn tốt nhất".

---

## 9. Checklist cho bài toán dự báo

- [ ] Chia dữ liệu theo **thứ tự thời gian**; không xáo trộn trước khi chia.
- [ ] Scaler và mọi tiền xử lý chỉ fit trên tập train.
- [ ] Cửa sổ trượt nằm hoàn toàn trong từng tập.
- [ ] Có baseline (persistence hoặc seasonal naive).
- [ ] Biến đổi target khi chuỗi có xu hướng (lấy sai phân), và khôi phục lại khi đánh giá.
- [ ] Đánh giá trên metric có giá trị chẩn đoán, không chỉ R².
- [ ] Với dự báo nhiều bước, báo cáo sai số theo từng horizon.

---

## 10. Câu hỏi ôn tập

<details>
<summary>1. Vì sao không được xáo trộn ngẫu nhiên chuỗi thời gian trước khi chia train/test?</summary>

Vì tập train sẽ chứa dữ liệu ở thời điểm sau tập test. Model được "nhìn thấy tương lai", dẫn đến leakage và kết quả lạc quan giả tạo.
</details>

<details>
<summary>2. Vì sao dự báo mức thay đổi Δy thay vì dự báo trực tiếp mức CO₂?</summary>

Vì chuỗi có xu hướng mạnh: chỉ cần đi theo xu hướng là đã đạt sai số thấp và R² cao. Lấy sai phân loại bỏ xu hướng, nên việc so sánh model tập trung vào biến động thực sự giữa các tuần.
</details>

<details>
<summary>3. Chuỗi train có 1.598 giá trị, SEQ_LEN = 52. Có bao nhiêu cửa sổ?</summary>

$1598 - 52 = 1546$.
</details>

<details>
<summary>4. Baseline persistence dự báo gì? Tương đương với việc dự báo Δy bằng bao nhiêu?</summary>

Mức CO₂ tuần sau bằng tuần này, tương đương dự báo $\Delta y = 0$.
</details>

<details>
<summary>5. Vì sao sai số tăng khi horizon dài hơn?</summary>

Vì mức CO₂ được khôi phục bằng cách cộng dồn các thay đổi dự báo, nên sai số cũng tích lũy. Tương lai càng xa thì càng bất định.
</details>

---

**Trước:** [06 — RNN, LSTM, GRU](06_rnn_lstm_gru.md) · **Tiếp theo:** [08 — Attention](08_attention.md)
