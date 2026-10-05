# 01 — Mạng neural và Forward Pass

> **Lab tương ứng:** TUT01 — *A Neural Network Forward Pass, Under the Hood*
> **Câu hỏi trung tâm:** Vài con số đầu vào đi qua một mạng neural và trở thành một con số đầu ra như thế nào?

---

## 1. Bản đồ: AI → Machine Learning → Deep Learning

| Khái niệm | Ý nghĩa | Ví dụ |
|---|---|---|
| **Artificial Intelligence (AI)** | Mọi kỹ thuật giúp máy thực hiện việc "có vẻ thông minh" | Hệ chuyên gia dùng luật if–else, tìm đường A* |
| **Machine Learning (ML)** | Máy **học quy luật từ dữ liệu** thay vì được lập trình sẵn từng luật | Hồi quy tuyến tính, cây quyết định, SVM |
| **Deep Learning (DL)** | Nhánh của ML dùng **mạng neural nhiều lớp**, tự học biểu diễn (representation) từ dữ liệu thô | CNN cho ảnh, RNN cho chuỗi, Transformer cho văn bản |

Điểm khác biệt cốt lõi của Deep Learning: thay vì con người tự thiết kế đặc trưng (feature engineering), mạng **tự học các biểu diễn trung gian** qua nhiều lớp biến đổi liên tiếp. Mỗi lớp nhận biểu diễn của lớp trước và tạo ra một biểu diễn mới, trừu tượng hơn.

---

## 2. Neuron nhân tạo

Một neuron nhận vector đầu vào $\mathbf{x} = [x_1, \dots, x_n]$ và tính:

$$
z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b = \mathbf{w}^\top\mathbf{x} + b
$$

$$
a = f(z)
$$

- $\mathbf{w}$ — **trọng số (weights)**: mức độ quan trọng và dấu (tăng hay giảm) của từng đầu vào.
- $b$ — **bias (độ lệch)**: dịch ngưỡng kích hoạt, cho phép neuron cho ra giá trị khác 0 ngay cả khi mọi đầu vào bằng 0.
- $z$ — **pre-activation**: tổng có trọng số, *trước* khi qua hàm kích hoạt.
- $f$ — **hàm kích hoạt (activation function)**.
- $a$ — **activation**: đầu ra của neuron.

> **Lưu ý:** hình vẽ "neuron và các mũi tên" là **đồ thị tính toán (computation graph)**, không phải mô phỏng tế bào thần kinh thật. Bên trong chỉ là phép nhân, cộng và một hàm phi tuyến.

### Ví dụ từ TUT01

Neuron 1 của lớp ẩn thứ nhất, với $\mathbf{x} = [0.75, 0.60, 0.55]$, $\mathbf{w} = [0.8, -0.4, 0.5]$, $b = 0.1$:

$$
z = 0.8(0.75) - 0.4(0.60) + 0.5(0.55) + 0.1 = 0.6 - 0.24 + 0.275 + 0.1 = 0.735
$$

Mỗi tích $w_ix_i$ được gọi là **đóng góp (contribution)** của đầu vào $i$. Trọng số âm ($-0.4$) làm giảm $z$, nghĩa là trong mạng này, tỉ lệ nộp bài cao lại *kéo giảm* neuron 1. Đây chỉ là trọng số tự chọn, không mang ý nghĩa thực tế.

---

## 3. Lớp tuyến tính (Linear / Fully-connected / Dense layer)

Khi có nhiều neuron cùng nhận một đầu vào, ta gộp chúng thành phép nhân ma trận:

$$
\mathbf{z} = W\mathbf{x} + \mathbf{b}
$$

| Ký hiệu | Shape | Ý nghĩa |
|---|---|---|
| $\mathbf{x}$ | $(n_{in},)$ | vector đầu vào |
| $W$ | $(n_{out}, n_{in})$ | mỗi **hàng** là trọng số của một neuron |
| $\mathbf{b}$ | $(n_{out},)$ | mỗi neuron có một bias |
| $\mathbf{z}$ | $(n_{out},)$ | đầu ra của lớp |

**Cách đọc shape:** `W1.shape == (3, 3)` nghĩa là có 3 hàng ứng với 3 neuron ở lớp sau, và 3 cột vì mỗi neuron nhận 3 giá trị. `W3.shape == (1, 2)` nghĩa là 1 neuron đầu ra nhận 2 giá trị.

> **Quy ước PyTorch:** `nn.Linear(in_features, out_features)` lưu `weight` với shape `(out_features, in_features)` và tính $y = xW^\top + b$, trong đó $x$ có shape `(batch, in_features)`.

Một phép biến đổi dạng $W\mathbf{x} + \mathbf{b}$ được gọi là **affine** (tuyến tính cộng thêm một phép dịch).

### Đếm số tham số

$$
\text{số tham số của một lớp Linear} = n_{out} \times n_{in} + n_{out}
$$

(số trọng số + số bias)

Mạng `3 → 3 → 2 → 1` của TUT01:

| Lớp | Tính | Số tham số |
|---|---|---|
| Linear 1 (3→3) | $3\times3 + 3$ | 12 |
| Linear 2 (3→2) | $2\times3 + 2$ | 8 |
| Linear 3 (2→1) | $1\times2 + 1$ | 3 |
| **Tổng** | | **23** |

ReLU và Sigmoid **không có tham số**: chúng là hàm cố định.

---

## 4. Hàm kích hoạt (Activation functions)

### 4.1 Vì sao cần phi tuyến?

Nếu bỏ hàm kích hoạt, hai lớp tuyến tính liên tiếp sẽ gộp lại thành **một** lớp tuyến tính:

$$
W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = \underbrace{(W_2W_1)}_{W'}\mathbf{x} + \underbrace{(W_2\mathbf{b}_1 + \mathbf{b}_2)}_{\mathbf{b}'}
$$

Chồng 100 lớp tuyến tính cũng chỉ tương đương 1 lớp, và chỉ biểu diễn được quan hệ tuyến tính (đường thẳng, siêu phẳng). Hàm phi tuyến xen giữa các lớp mới khiến "độ sâu" có ý nghĩa: mạng có thể biểu diễn các ranh giới quyết định cong và phức tạp.

> **Universal Approximation Theorem:** một mạng có ít nhất một lớp ẩn phi tuyến, với đủ neuron, có thể xấp xỉ bất kỳ hàm liên tục nào trên một miền đóng và bị chặn với độ chính xác tùy ý. Định lý chỉ nói rằng **tồn tại** bộ trọng số như vậy, không nói cách **tìm** ra nó. Việc tìm trọng số là nhiệm vụ của training (file 02).

**Quan sát từ TUT01, Exercise 5:** bỏ ReLU nhưng score vẫn bằng đúng 0.5686. Đây là **trùng hợp**. Với input này, ReLU chỉ cắt neuron 3 (có $z = -0.025$), và neuron này không ảnh hưởng tới đầu ra vì $W_3 \cdot W_2[:, 2] = 1.2(0.4) - 0.8(0.6) = 0$. Với input khác, ví dụ `[0.1, 0.9, 0.2]`, score là 0.360 khi có ReLU và 0.263 khi không có ReLU. Bài học: **đừng kết luận từ một mẫu duy nhất**.

### 4.2 Các hàm kích hoạt phổ biến

| Hàm | Công thức | Miền giá trị | Đạo hàm | Dùng ở đâu |
|---|---|---|---|---|
| **ReLU** | $\max(0, z)$ | $[0, \infty)$ | 1 nếu $z>0$, 0 nếu $z<0$ | Lớp ẩn: mặc định cho MLP và CNN |
| **Sigmoid** | $\sigma(z) = \dfrac{1}{1+e^{-z}}$ | $(0, 1)$ | $\sigma(z)(1-\sigma(z))$ | Đầu ra phân loại nhị phân; các cổng (gate) trong LSTM/GRU |
| **Tanh** | $\tanh(z) = \dfrac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | $1 - \tanh^2(z)$ | Hidden state của RNN, LSTM |
| **Softmax** | $\dfrac{e^{z_i}}{\sum_j e^{z_j}}$ | mỗi phần tử $\in (0,1)$, tổng bằng 1 | — | Đầu ra phân loại nhiều lớp; trọng số attention |
| Leaky ReLU | $\max(\alpha z, z)$, với $\alpha \approx 0.01$ | $\mathbb{R}$ | $\alpha$ hoặc 1 | Tránh "neuron chết" |
| GELU | $z\,\Phi(z)$ | $\approx(-0.17, \infty)$ | mượt | Transformer (BERT, GPT) |
| Hardswish | $z \cdot \dfrac{\text{ReLU6}(z+3)}{6}$ | | | MobileNetV3 |

**ReLU — "cánh cổng" (gate):**
- giữ nguyên giá trị dương, đưa giá trị âm về **đúng 0**;
- rẻ để tính, và đạo hàm bằng 1 khi $z > 0$ nên gradient không bị co lại khi đi qua (khác với sigmoid/tanh bị bão hòa);
- nhược điểm: neuron có $z < 0$ với mọi input sẽ có gradient bằng 0 mãi mãi (**dying ReLU**).

Ví dụ trong TUT01: $\mathbf{z}^{(1)} = [0.735, 0.225, -0.025] \xrightarrow{\text{ReLU}} \mathbf{a}^{(1)} = [0.735, 0.225, 0]$.

ReLU thay đổi **activation**, nhưng **không** thay đổi tham số và **không** thay đổi kiến trúc.

**Sigmoid:** ép một số thực bất kỳ vào khoảng $(0, 1)$. Nó có tính đối xứng $\sigma(-z) = 1 - \sigma(z)$, và $\sigma(0) = 0.5$. Khi $|z|$ lớn, sigmoid bị **bão hòa**: đạo hàm gần 0, gây ra vấn đề vanishing gradient.

**Softmax:** biến một vector điểm số thành **phân phối xác suất**. Ví dụ $[2, 1, 0.1] \to [0.659, 0.242, 0.099]$. Softmax chỉ phụ thuộc vào **chênh lệch** giữa các điểm số: cộng cùng một hằng số vào mọi phần tử thì kết quả không đổi.

---

## 5. Multi-Layer Perceptron (MLP) và Forward Pass

**MLP (Multi-Layer Perceptron)**, còn gọi là **feedforward network**, là chuỗi các lớp Linear xen kẽ với hàm kích hoạt. Thông tin chỉ đi **một chiều**, từ đầu vào tới đầu ra, không có vòng lặp.

Mạng của TUT01:

$$
\mathbf{x} \xrightarrow{\text{Linear}_1} \mathbf{z}^{(1)} \xrightarrow{\text{ReLU}} \mathbf{a}^{(1)} \xrightarrow{\text{Linear}_2} \mathbf{z}^{(2)} \xrightarrow{\text{ReLU}} \mathbf{a}^{(2)} \xrightarrow{\text{Linear}_3} z^{(3)} \xrightarrow{\sigma} \hat{y}
$$

Mạng neural là **một hàm hợp (composition) của nhiều hàm nhỏ**:

$$
\hat{y} = f(\mathbf{x}) = \sigma \circ L_3 \circ \text{ReLU} \circ L_2 \circ \text{ReLU} \circ L_1(\mathbf{x})
$$

**Forward pass (lan truyền xuôi)** là quá trình tính lần lượt từng hàm, từ trong ra ngoài.

### Trace đầy đủ của TUT01

| Giai đoạn | Giá trị |
|---|---|
| Input $\mathbf{x}$ | $[0.75,\ 0.60,\ 0.55]$ |
| $\mathbf{z}^{(1)} = W_1\mathbf{x} + \mathbf{b}_1$ | $[0.735,\ 0.225,\ -0.025]$ |
| $\mathbf{a}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)})$ | $[0.735,\ 0.225,\ 0]$ |
| $\mathbf{a}^{(2)}$ | $\approx [0.402,\ 0.133]$ |
| logit $z^{(3)}$ | $\approx 0.276$ |
| score $\sigma(z^{(3)})$ | $\approx 0.569$ |

Mỗi lớp tạo ra một **biểu diễn (representation)** mới. Sau lớp 1, mạng không còn làm việc trực tiếp với "chuyên cần, nộp bài, điểm giữa kỳ" nữa, mà với 3 giá trị nội bộ do chính nó tạo ra.

### Logit

**Logit** là đầu ra thô của lớp cuối, có thể là bất kỳ số thực nào trong $(-\infty, +\infty)$. Logit chưa phải xác suất: phải qua sigmoid (bài toán nhị phân) hoặc softmax (bài toán nhiều lớp) mới thành xác suất. Về mặt toán, logit chính là log-odds:

$$
z = \log\frac{p}{1-p} \iff p = \sigma(z)
$$

### "Deep" nghĩa là gì?

Một mạng được gọi là **deep** khi đầu vào được biến đổi qua **nhiều lớp** liên tiếp. Thêm một lớp (Exercise 6 của TUT01) nghĩa là thêm một phép biến đổi nữa vào chuỗi hàm hợp.

---

## 6. Phân biệt bốn loại giá trị

Đây là nội dung **quan trọng nhất** của TUT01.

| Loại | Ví dụ trong TUT01 | Thuộc về | Có thay đổi khi đổi input? | Có thay đổi khi training? |
|---|---|---|---|---|
| **Input (dữ liệu)** | `0.75, 0.60, 0.55` | mẫu dữ liệu hiện tại | — | không |
| **Parameters (tham số)** | các phần tử của $W_1, W_2, W_3, \mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$ | model, được lưu bên trong | **không** | **có**: đây là thứ được học |
| **Intermediate activations** | $\mathbf{z}^{(1)}, \mathbf{a}^{(1)}, \mathbf{z}^{(2)}, \mathbf{a}^{(2)}$ | **một lần chạy** model | **có** | — (tạm thời, tính lại mỗi lần) |
| **Output** | `0.569` | một lần chạy model | **có** | — |

Ngoài ra còn hai khái niệm cần phân biệt:
- **Architecture (kiến trúc):** *cấu trúc*, gồm số lớp, số neuron, loại hàm kích hoạt và cách kết nối giữa các lớp.
- **Hyperparameters (siêu tham số):** các lựa chọn do người thiết kế đặt ra, **không** được học bằng gradient. Ví dụ: số lớp, số neuron, learning rate, batch size.

> **Model** = kiến trúc + bộ tham số. **Một lần thực thi model** = một forward pass trên một input cụ thể.

TUT01, Exercise 1: đổi input sang `[0.95, 0.90, 0.80]` thì activation và output thay đổi (score 0.5296), còn tham số giữ nguyên.

---

## 7. Model chưa được train vẫn cho ra con số

Với trọng số tự chọn, mạng vẫn cho ra `0.569`, một con số trông giống xác suất. Nhưng:

- trọng số không được học từ dữ liệu;
- kết quả chưa từng được so sánh với nhãn thật;
- vì vậy `0.569` **không phải bằng chứng** model hữu ích.

> **Chạy được ≠ dự đoán đúng.** Một mạng với trọng số vô nghĩa vẫn thực hiện forward pass bình thường.

**Training** là quá trình điều chỉnh tham số, dựa trên dữ liệu và một hàm mục tiêu (loss), để đầu ra trở nên hữu ích. Đó là chủ đề của file [02](02_huan_luyen_mang_neural.md).

| | Inference (suy luận) | Training (huấn luyện) |
|---|---|---|
| Forward pass | ✔ | ✔ |
| Tính loss | ✘ | ✔ |
| Backward (tính gradient) | ✘ | ✔ |
| Cập nhật tham số | ✘ | ✔ |

---

## 8. Forward pass trong PyTorch

```python
model = nn.Sequential(
    nn.Linear(3, 3), nn.ReLU(),
    nn.Linear(3, 2), nn.ReLU(),
    nn.Linear(2, 1), nn.Sigmoid(),
)
with torch.inference_mode():          # chỉ suy luận: không lưu đồ thị gradient
    y = model(x)                       # gọi model(x), KHÔNG gọi model.forward(x)
```

- `nn.Linear` cài đặt phép biến đổi affine $xW^\top + b$.
- `torch.inference_mode()` (hoặc `torch.no_grad()`) tắt việc theo dõi gradient, nên nhanh hơn và tốn ít bộ nhớ hơn khi chỉ dự đoán.
- **Forward hook** (`module.register_forward_hook(fn)`) là một "người quan sát" gắn vào một module. Hook được gọi sau mỗi lần module chạy forward, nên có thể ghi lại output trung gian mà không cần sửa code của model. Rất hữu ích để debug và trực quan hóa activation (dùng lại ở CNN, file 03).
- Phải gọi `model(x)` thay vì `model.forward(x)`, vì `__call__` mới kích hoạt các hook và các cơ chế khác của module.

---

## 9. Câu hỏi ôn tập

<details>
<summary>1. Một lớp <code>nn.Linear(128, 64)</code> có bao nhiêu tham số?</summary>

$64 \times 128 + 64 = 8256$.
</details>

<details>
<summary>2. Vì sao xếp chồng các lớp Linear mà không có activation thì không tăng khả năng biểu diễn?</summary>

Vì hợp của các hàm affine vẫn là một hàm affine: $W_2(W_1x + b_1) + b_2 = W'x + b'$.
</details>

<details>
<summary>3. ReLU có tham số học được không? Nó thay đổi cái gì?</summary>

Không có tham số. ReLU chỉ thay đổi **activation** (đưa giá trị âm về 0). Nó không thay đổi tham số và không thay đổi kiến trúc.
</details>

<details>
<summary>4. Đưa cùng một model hai input khác nhau. Cái gì thay đổi, cái gì giữ nguyên?</summary>

Activation trung gian và output thay đổi. Tham số và kiến trúc giữ nguyên.
</details>

<details>
<summary>5. Logit là gì? Vì sao không dùng logit trực tiếp làm xác suất?</summary>

Logit là đầu ra thô, nằm trong $(-\infty, +\infty)$. Nó cần qua sigmoid hoặc softmax mới thành xác suất nằm trong $(0, 1)$.
</details>

<details>
<summary>6. Training sẽ thay đổi gì mà notebook TUT01 không bao giờ thay đổi?</summary>

Các giá trị tham số $W$ và $b$, được cập nhật bằng gradient của loss trên dữ liệu có nhãn.
</details>

---

## 10. Tóm tắt

- Mạng neural là **hàm hợp** của các phép biến đổi affine ($W\mathbf{x} + \mathbf{b}$) xen kẽ với hàm kích hoạt phi tuyến.
- **Forward pass** tính các phép biến đổi đó lần lượt, từ input tới output.
- **Tham số** ($W$, $b$) thuộc về model. **Activation** là giá trị tạm thời của một lần chạy.
- Số tham số của một lớp Linear là $n_{out} \cdot n_{in} + n_{out}$.
- Phi tuyến là bắt buộc. Nếu không có nó, mọi mạng sâu đều suy biến thành một lớp tuyến tính.
- Model chưa train vẫn cho ra số, nhưng con số đó vô nghĩa. Training là bước làm cho tham số trở nên hữu ích.

**Tiếp theo:** [02 — Huấn luyện mạng neural](02_huan_luyen_mang_neural.md)
