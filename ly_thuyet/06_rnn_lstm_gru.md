# 06 — Mạng hồi quy: RNN, LSTM, GRU

> **Lab tương ứng:** TUT04 — *RNN, LSTM, and GRU: From Cell Mechanics to Time-Series Forecasting* (Part A và mục 6)
> **Ý tưởng chính:** Xử lý dữ liệu **tuần tự** bằng một **trạng thái ẩn (hidden state)** được cập nhật qua từng bước thời gian. Trạng thái này đóng vai trò "bộ nhớ" về những gì đã thấy trước đó.

---

## 1. Dữ liệu tuần tự và vì sao MLP/CNN chưa đủ

**Dữ liệu tuần tự** là dữ liệu mà **thứ tự** mang thông tin: chuỗi thời gian (CO₂, giá cổ phiếu), văn bản, âm thanh, video.

Với dữ liệu tuần tự, MLP có ba hạn chế:
- cần input có **độ dài cố định**;
- không chia sẻ những gì đã học giữa các vị trí trong chuỗi;
- không có khái niệm "trước" và "sau".

Ta cần một mô hình:
1. nhận được chuỗi có **độ dài tùy ý**;
2. **dùng chung tham số** ở mọi bước thời gian (giống weight sharing của CNN, nhưng theo trục thời gian);
3. **nhớ** thông tin từ các bước trước.

---

## 2. Vanilla RNN (Elman, 1990)

### 2.1 Công thức

$$
\mathbf{h}_t = \tanh\big(W_x\mathbf{x}_t + W_h\mathbf{h}_{t-1} + \mathbf{b}\big)
$$

- $\mathbf{x}_t$: input tại bước $t$.
- $\mathbf{h}_{t-1}$: hidden state của bước trước, là "bộ nhớ" mang theo thông tin từ quá khứ. Thường khởi tạo $\mathbf{h}_0 = \mathbf{0}$.
- $W_x, W_h, \mathbf{b}$: **cùng một bộ tham số** được dùng ở **mọi** bước thời gian.
- **Recurrence (hồi quy):** $\mathbf{h}_t$ phụ thuộc vào cả input hiện tại **và** trạng thái trước.

```text
x_t ────────┐
            ├── tổ hợp tuyến tính ── tanh ──> h_t ──> (dùng cho output, và
h_(t-1) ────┘                                        chuyển sang bước t+1)
```

### 2.2 "Unroll" theo thời gian

Có thể vẽ RNN như một mạng sâu, mỗi "lớp" là một bước thời gian, và các lớp **dùng chung trọng số**:

```text
h_0 → [cell] → h_1 → [cell] → h_2 → [cell] → h_3 → ...
        ↑               ↑               ↑
       x_1             x_2             x_3
```

Chuỗi dài 52 bước tương đương một mạng sâu 52 lớp. Đây chính là nguồn gốc của vấn đề gradient ở mục 3.

### 2.3 Các kiểu bài toán chuỗi

| Kiểu | Ví dụ |
|---|---|
| many-to-one | phân loại cảm xúc một câu; **dự báo giá trị tiếp theo từ 52 bước trước (TUT04)** |
| one-to-many | sinh mô tả cho một ảnh |
| many-to-many (đồng bộ) | gán nhãn từ loại cho từng từ |
| many-to-many (seq2seq) | dịch máy: encoder → decoder |

### 2.4 Nhận xét về Vanilla RNN

- Chỉ có **một** trạng thái $\mathbf{h}_t$. Thông tin mới và cũ bị trộn lẫn qua **một** phép biến đổi phi tuyến.
- **Không có cổng (gate)** nào để quyết định "giữ lại cái này" hay "quên cái kia".
- Rất khó học được **phụ thuộc dài hạn** (long-term dependency).

---

## 3. Backpropagation Through Time (BPTT) và vấn đề gradient

**BPTT** là backpropagation áp dụng trên mạng đã được unroll. Gradient của loss theo $\mathbf{h}_k$ ở một bước xa trong quá khứ phải đi qua tích của nhiều Jacobian:

$$
\frac{\partial\mathbf{h}_t}{\partial\mathbf{h}_k} = \prod_{j=k+1}^{t}\frac{\partial\mathbf{h}_j}{\partial\mathbf{h}_{j-1}} = \prod_{j=k+1}^{t}\text{diag}\big(1 - \mathbf{h}_j^2\big)\,W_h
$$

Đây là tích của $t - k$ ma trận:
- Nếu các thừa số "nhỏ hơn 1" (đạo hàm tanh ≤ 1, và giá trị riêng lớn nhất của $W_h$ nhỏ hơn 1), tích **co về 0 theo hàm mũ**. Đó là **vanishing gradient**: tín hiệu từ quá khứ xa gần như không còn, nên mạng không học được phụ thuộc dài hạn.
- Nếu các thừa số "lớn hơn 1", tích **bùng nổ theo hàm mũ**. Đó là **exploding gradient**: loss nhảy vọt, trọng số có thể thành NaN.

### Cách xử lý

| Vấn đề | Giải pháp |
|---|---|
| Exploding | **Gradient clipping**: nếu $\lVert g\rVert > c$ thì co lại $g \leftarrow g\cdot c/\lVert g\rVert$. TUT04 dùng `clip_grad_norm_(max_norm=1.0)`. |
| Vanishing | **Kiến trúc có cổng (LSTM, GRU)**; khởi tạo trọng số hợp lý; dùng chuỗi ngắn hơn (truncated BPTT); attention ([file 08](08_attention.md)) |

---

## 4. LSTM — Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)

### 4.1 Ý tưởng

Tách bộ nhớ thành **hai** trạng thái:
- **cell state $\mathbf{c}_t$:** "băng chuyền" bộ nhớ dài hạn, được cập nhật bằng phép **cộng**;
- **hidden state $\mathbf{h}_t$:** đầu ra ở mỗi bước, là phần bộ nhớ được "hiển thị" ra ngoài.

Các **cổng (gate)** dùng sigmoid (cho giá trị trong $(0, 1)$) để điều khiển dòng thông tin: 0 là chặn hoàn toàn, 1 là cho qua hoàn toàn.

### 4.2 Công thức

$$
\begin{aligned}
\mathbf{f}_t &= \sigma(W_f\mathbf{x}_t + U_f\mathbf{h}_{t-1} + \mathbf{b}_f) && \text{forget gate}\\
\mathbf{i}_t &= \sigma(W_i\mathbf{x}_t + U_i\mathbf{h}_{t-1} + \mathbf{b}_i) && \text{input gate}\\
\mathbf{g}_t &= \tanh(W_g\mathbf{x}_t + U_g\mathbf{h}_{t-1} + \mathbf{b}_g) && \text{candidate (nội dung mới)}\\
\mathbf{o}_t &= \sigma(W_o\mathbf{x}_t + U_o\mathbf{h}_{t-1} + \mathbf{b}_o) && \text{output gate}\\[4pt]
\mathbf{c}_t &= \mathbf{f}_t\odot\mathbf{c}_{t-1} + \mathbf{i}_t\odot\mathbf{g}_t && \text{cập nhật cell state}\\
\mathbf{h}_t &= \mathbf{o}_t\odot\tanh(\mathbf{c}_t) && \text{hidden state}
\end{aligned}
$$

| Thành phần | Vai trò |
|---|---|
| **Forget gate $f_t$** | giữ lại bao nhiêu phần cell state cũ ($f = 1$: giữ toàn bộ, $f = 0$: xóa hết) |
| **Input gate $i_t$** | ghi bao nhiêu phần nội dung mới vào bộ nhớ |
| **Candidate $g_t$** | nội dung mới được đề xuất, nằm trong $(-1, 1)$ |
| **Output gate $o_t$** | phần nào của bộ nhớ được đưa ra thành $h_t$ |

> **Ghi chú lịch sử:** LSTM gốc năm 1997 **chưa có** forget gate. Forget gate được Gers và cộng sự bổ sung sau này (khoảng năm 2000).

### 4.3 Vì sao LSTM giữ thông tin tốt hơn?

Cell state được cập nhật theo kiểu **cộng**:

$$
\mathbf{c}_t = \mathbf{f}_t\odot\mathbf{c}_{t-1} + \mathbf{i}_t\odot\mathbf{g}_t
\quad\Rightarrow\quad
\frac{\partial\mathbf{c}_t}{\partial\mathbf{c}_{t-1}} \approx \text{diag}(\mathbf{f}_t)
$$

Khi $\mathbf{f}_t \approx 1$, thông tin (và gradient) đi qua nhiều bước mà **hầu như không bị co lại**. Ở RNN thường, gradient bị nhân với $W_h$ và đạo hàm tanh ở mỗi bước. Con đường này gọi là **"constant error carousel"**, có ý tưởng giống với residual connection trong ResNet.

> LSTM **không** loại bỏ hoàn toàn vanishing gradient. Nó tạo ra một con đường cho thông tin và gradient mà **mạng có thể học để điều khiển**, dễ kiểm soát hơn nhiều so với recurrence tanh thuần.

---

## 5. GRU — Gated Recurrent Unit (Cho et al., 2014)

### 5.1 Ý tưởng

GRU đơn giản hơn LSTM: **bỏ cell state riêng**, chỉ giữ $\mathbf{h}_t$, và dùng **hai** cổng.

### 5.2 Công thức (quy ước của PyTorch)

$$
\begin{aligned}
\mathbf{r}_t &= \sigma(W_r\mathbf{x}_t + U_r\mathbf{h}_{t-1} + \mathbf{b}_r) && \text{reset gate}\\
\mathbf{z}_t &= \sigma(W_z\mathbf{x}_t + U_z\mathbf{h}_{t-1} + \mathbf{b}_z) && \text{update gate}\\
\mathbf{n}_t &= \tanh\big(W_n\mathbf{x}_t + \mathbf{r}_t\odot(U_n\mathbf{h}_{t-1}) + \mathbf{b}_n\big) && \text{candidate}\\
\mathbf{h}_t &= (1-\mathbf{z}_t)\odot\mathbf{n}_t + \mathbf{z}_t\odot\mathbf{h}_{t-1}
\end{aligned}
$$

| Cổng | Vai trò |
|---|---|
| **Reset $r_t$** | trạng thái cũ đóng góp bao nhiêu khi tạo candidate. $r \approx 0$ nghĩa là "bỏ qua quá khứ, bắt đầu lại". |
| **Update $z_t$** | **nội suy** giữa trạng thái cũ và candidate. Với quy ước này, **$z \approx 1$ nghĩa là giữ lại nhiều trạng thái cũ**. |

Update gate của GRU đảm nhiệm cùng lúc vai trò của forget gate và input gate trong LSTM, vì hai hệ số $(1 - z)$ và $z$ có tổng bằng 1. Phép nội suy này cũng tạo ra một con đường gần như cộng, giúp gradient đi xa.

> Bài báo gốc của GRU đặt reset gate ở vị trí hơi khác: $r_t\odot\mathbf{h}_{t-1}$ được tính **trước** khi nhân với $U_n$. Một số tài liệu cũng đổi vai trò của $z$ và $(1 - z)$. Khi đọc tài liệu cần để ý quy ước đang dùng.

---

## 6. So sánh ba kiến trúc

| Thuộc tính | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| Biến trạng thái | $h_t$ | $h_t$ **và** $c_t$ | $h_t$ |
| Cổng | không có | forget, input, output | reset, update |
| Số phép biến đổi (gate/candidate) | 1 | 4 | 3 |
| Số tham số (cùng hidden size) | **ít nhất** | **nhiều nhất** | ở giữa |
| Kiểm soát thông tin dài hạn | yếu, ngầm định | tường minh | tường minh |
| Đặc điểm | đơn giản, rẻ nhất | cơ chế cổng linh hoạt nhất | gọn hơn LSTM, thường cho kết quả tương đương |

> **Không có kiến trúc nào luôn thắng.** Kết quả phụ thuộc vào dữ liệu, độ dài chuỗi, cách tối ưu, hidden size, ngân sách tham số và bài toán cụ thể.

### 6.1 Công thức số tham số (PyTorch)

Với input size $d$ và hidden size $h$, mỗi "phép biến đổi" có $W$ ($h\times d$), $U$ ($h\times h$) và **hai** vector bias ($b_{ih}$ và $b_{hh}$, mỗi cái kích thước $h$):

$$
\text{một phép biến đổi} = h\cdot d + h\cdot h + 2h
$$

| | Số tham số của lớp recurrent | TUT04 ($d = 1$, $h = 24$) | Cộng head `Linear(24→1)` (+25) |
|---|---|---|---|
| RNN | $1\times(hd + h^2 + 2h)$ | 648 | **673** |
| GRU | $3\times(hd + h^2 + 2h)$ | 1.944 | **1.969** |
| LSTM | $4\times(hd + h^2 + 2h)$ | 2.592 | **2.617** |

LSTM lớn hơn GRU đúng $4/3$ lần ở phần recurrent (+648 tham số, khoảng 33%). Vì vậy so sánh các model **cùng hidden size** chỉ là so sánh thực dụng, **không** phải so sánh cùng ngân sách tham số (xem [file 05](05_danh_gia_mo_hinh.md), mục 9).

---

## 7. RNN, LSTM, GRU trong PyTorch

```python
rnn  = nn.RNN(input_size=1,  hidden_size=24, num_layers=1, batch_first=True)
lstm = nn.LSTM(input_size=1, hidden_size=24, num_layers=1, batch_first=True)
gru  = nn.GRU(input_size=1,  hidden_size=24, num_layers=1, batch_first=True)
```

| | Shape |
|---|---|
| Input (`batch_first=True`) | `(batch, seq_len, input_size)`, ví dụ `(64, 52, 1)` |
| `output` | `(batch, seq_len, hidden_size)`: $h_t$ ở **mọi** bước |
| `h_n` | `(num_layers, batch, hidden_size)`: $h$ ở **bước cuối cùng** |
| `c_n` (chỉ LSTM) | `(num_layers, batch, hidden_size)` |

- `RNN` và `GRU` trả về `(output, h_n)`.
- `LSTM` trả về `(output, (h_n, c_n))`.
- Không đặt `batch_first=True` thì mặc định là `(seq_len, batch, input_size)`. Đây là nguồn bug rất phổ biến.

**Model dự báo trong TUT04 (many-to-one):**

```python
output, _ = self.recurrent(x)       # (B, 52, 24)
last = output[:, -1, :]             # (B, 24): hidden state ở bước cuối, đã "tóm tắt" 52 tuần
y_hat = self.head(last)             # Linear(24 → 1): thay đổi CO₂ của tuần tới
```

Ba kiến trúc dùng **cùng một wrapper**, chỉ khác lớp recurrent. Nhờ vậy việc so sánh mới công bằng.

### Các biến thể

- **Stacked RNN** (`num_layers > 1`): output của lớp dưới là input của lớp trên.
- **Bidirectional** (`bidirectional=True`): một RNN đọc xuôi, một RNN đọc ngược, rồi ghép kết quả. Hữu ích khi có toàn bộ chuỗi, ví dụ phân loại văn bản. **Không** dùng được cho dự báo, vì sẽ "nhìn thấy tương lai".

---

## 8. Kết quả trong TUT04

| | Delta RMSE (ppm) | Tham số | Thời gian train |
|---|---|---|---|
| RNN | 0,4283 | 673 | nhanh nhất |
| LSTM | 0,4048 | 2.617 | |
| GRU | **0,3998** | 1.969 | chậm nhất |
| Persistence | 0,4945 | 0 | — |

Kết quả các bài tập:
- **Exercise 1 (độ dài chuỗi 13, 26, 52, 104):** GRU tốt nhất ở mọi độ dài. **RNN nhạy với độ dài chuỗi nhất**, và tệ đi rõ khi chuỗi dài 104 bước. Điều này phù hợp với lý thuyết về vanishing gradient. Chuỗi dài hơn **không** phải lúc nào cũng tốt hơn.
- **Exercise 3 (5 seed):** tính trung bình thì **LSTM** tốt hơn GRU, nhưng chênh lệch nhỏ hơn độ dao động giữa các seed. Không kết luận được kiến trúc nào tốt hơn.
- **Exercise 4 (dự báo 1, 4, 12 tuần):** RNN suy giảm mạnh nhất khi horizon dài hơn.

---

## 9. Hạn chế của RNN và hướng phát triển

- **Xử lý tuần tự:** $h_t$ phải chờ $h_{t-1}$ tính xong, nên **không song song hóa** được theo trục thời gian, và train chậm với chuỗi dài.
- **Nút thắt cổ chai:** toàn bộ quá khứ bị nén vào một vector có kích thước cố định.
- **Phụ thuộc rất xa** vẫn khó học, kể cả với LSTM.

**Attention** ([file 08](08_attention.md)) cho phép mỗi vị trí **truy cập trực tiếp** mọi vị trí khác. **Transformer** ([file 09](09_transformer_hugging_face.md)) bỏ hẳn recurrence và đã thay thế RNN trong phần lớn bài toán NLP.

---

## 10. Câu hỏi ôn tập

<details>
<summary>1. Vì sao vanilla RNN khó học phụ thuộc dài hạn?</summary>

Gradient phải đi qua tích của nhiều Jacobian $\text{diag}(1-h^2)W_h$, nên co về 0 theo hàm mũ (vanishing) hoặc bùng nổ theo hàm mũ (exploding).
</details>

<details>
<summary>2. Thành phần nào giúp LSTM giữ thông tin qua nhiều bước?</summary>

Cập nhật cộng của cell state $c_t = f_t c_{t-1} + i_t g_t$. Khi $f_t \approx 1$, thông tin và gradient đi qua gần như nguyên vẹn.
</details>

<details>
<summary>3. Trong GRU (quy ước PyTorch), z ≈ 1 nghĩa là gì?</summary>

Giữ gần như nguyên trạng thái cũ: $h_t \approx h_{t-1}$, và bỏ qua candidate mới.
</details>

<details>
<summary>4. Với hidden = 24 và input = 1, vì sao LSTM có 2.592 tham số ở phần recurrent?</summary>

$4\times(24\cdot1 + 24\cdot24 + 2\cdot24) = 4\times648 = 2592$, vì LSTM có 4 phép biến đổi (gate/candidate), mỗi phép có 2 vector bias.
</details>

<details>
<summary>5. Gradient clipping giải quyết vấn đề nào? Có giải quyết vanishing gradient không?</summary>

Giải quyết exploding gradient, bằng cách giới hạn chuẩn của gradient. **Không** giải quyết vanishing gradient.
</details>

<details>
<summary>6. LSTM trả về những gì? Shape của chúng là gì khi <code>batch_first=True</code>?</summary>

`output`: `(B, L, H)`. `(h_n, c_n)`: mỗi cái có shape `(num_layers, B, H)`.
</details>

---

**Trước:** [05 — Đánh giá mô hình](05_danh_gia_mo_hinh.md) · **Tiếp theo:** [07 — Dự báo chuỗi thời gian](07_du_bao_chuoi_thoi_gian.md)
