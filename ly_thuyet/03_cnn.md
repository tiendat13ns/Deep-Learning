# 03 — Mạng neural tích chập (CNN — Convolutional Neural Network)

> **Lab tương ứng:** TUT03 — *Practical CNN: From Convolution Mechanics to Fine-Tuning* (Part 1–3, Part 6)
> **Ý tưởng chính:** Một mẫu hình cục bộ (cạnh, góc, texture) có thể xuất hiện ở **nhiều vị trí** trong ảnh, nên ta dùng **cùng một bộ lọc nhỏ** trượt khắp ảnh.

---

## 1. Vì sao MLP không phù hợp với ảnh?

Một ảnh RGB $224 \times 224$ có $224 \times 224 \times 3 = 150.528$ giá trị.

**MLP (Dense):** nếu nối mọi pixel tới 256 neuron, chỉ riêng lớp đầu đã cần

$$
224 \times 224 \times 3 \times 256 = 38.535.168 \text{ trọng số}
$$

**Conv:** từ 3 kênh tới 32 kênh với kernel $3\times3$ chỉ cần

$$
32 \times 3 \times 3 \times 3 + 32 = 896 \text{ tham số}
$$

MLP có ba vấn đề với ảnh:
1. **Quá nhiều tham số**, nên dễ overfit và tốn bộ nhớ.
2. **Mất cấu trúc không gian:** ảnh bị duỗi thành vector, nên mạng không biết hai pixel nào nằm cạnh nhau.
3. **Không tái sử dụng được:** một con mèo ở góc trái và ở góc phải phải được học **riêng biệt**, vì chúng đi qua các trọng số khác nhau.

CNN giải quyết cả ba vấn đề bằng hai **inductive bias** (giả định có sẵn trong kiến trúc):

| Nguyên lý | Ý nghĩa |
|---|---|
| **Local connectivity (kết nối cục bộ)** | mỗi đầu ra chỉ nhìn một vùng nhỏ (patch) của input |
| **Weight sharing (chia sẻ trọng số)** | cùng một kernel được dùng ở **mọi vị trí** |

---

## 2. Phép tích chập (Convolution)

### 2.1 Một kênh

Với ảnh 1 kênh $x$ và kernel $k$ kích thước $K \times K$:

$$
y_{i,j} = \sum_{u=0}^{K-1}\sum_{v=0}^{K-1} x_{i+u,\,j+v}\;k_{u,v} + b
$$

Nói cách khác: **output = tổng của (patch ⊙ kernel)**, với ⊙ là phép nhân từng phần tử.

> Về mặt toán, công thức trên là **cross-correlation**: kernel không bị lật như trong định nghĩa convolution chuẩn của xử lý tín hiệu. Mọi framework deep learning đều dùng phép này và vẫn gọi là "convolution". Vì kernel được học từ dữ liệu, việc có lật hay không không quan trọng.

### 2.2 Ví dụ tính tay: kernel phát hiện cạnh dọc

Input (3×5), bên trái tối, bên phải sáng:

$$
x = \begin{bmatrix} 1&1&1&8&8\\ 1&1&1&8&8\\ 1&1&1&8&8 \end{bmatrix},
\qquad
K = \begin{bmatrix} -1&0&1\\ -1&0&1\\ -1&0&1 \end{bmatrix}
$$

Trượt kernel (không padding, stride 1) cho output kích thước 1×3:

| Vị trí | Patch (mỗi hàng) | Tính mỗi hàng | Output |
|---|---|---|---|
| cột 0–2 | `1 1 1` | $-1 + 0 + 1 = 0$ | **0** (vùng phẳng) |
| cột 1–3 | `1 1 8` | $-1 + 0 + 8 = 7$ | **21** (có cạnh) |
| cột 2–4 | `1 8 8` | $-1 + 0 + 8 = 7$ | **21** (có cạnh) |

Kernel phản ứng **mạnh ở chỗ cường độ thay đổi theo chiều ngang** (cạnh dọc), và cho 0 ở vùng đồng nhất.

### 2.3 Các kernel kinh điển (TUT03, Part 2)

| Kernel | Ma trận | Phát hiện |
|---|---|---|
| Sobel dọc | $\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}$ | cạnh dọc (thay đổi trái ↔ phải) |
| Sobel ngang | $\begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}$ | cạnh ngang (thay đổi trên ↔ dưới) |
| Blur trung bình | $\frac{1}{9}\mathbf{1}_{3\times3}$ | làm mờ, khử nhiễu |

**TUT03, Exercise 1:** kernel Prewitt ngang $\begin{bmatrix}1&1&1\\0&0&0\\-1&-1&-1\end{bmatrix}$ cho kết quả:
- **−3** ở cạnh trên của hình chữ nhật (đi xuống: tối → sáng);
- **+3** ở cạnh dưới (sáng → tối);
- **0** ở cạnh trái/phải và ở vùng phẳng.

Dấu của response cho biết **hướng** thay đổi độ sáng.

> Trong CNN thật, kernel **được học từ dữ liệu**. Các kernel viết tay ở đây chỉ để dễ hiểu cơ chế.

### 2.4 Nhiều kênh đầu vào và đầu ra

Với input có $C_{in}$ kênh, ví dụ RGB có $C_{in} = 3$:
- **mỗi kernel** có kích thước $C_{in} \times K \times K$, quét qua **toàn bộ chiều sâu** và cộng tất cả lại thành **một** feature map;
- dùng $C_{out}$ kernel thì có $C_{out}$ feature map, chính là $C_{out}$ kênh đầu ra.

$$
y_{c_{out},i,j} = \sum_{c=1}^{C_{in}}\sum_{u,v} x_{c,\,i+u,\,j+v}\;k_{c_{out},c,u,v} + b_{c_{out}}
$$

**Số tham số của một lớp Conv2d:**

$$
C_{out} \times C_{in} \times K \times K + C_{out}
$$

Con số này **không phụ thuộc vào kích thước ảnh**. Ví dụ `Conv2d(3, 32, kernel_size=5)` có $32\cdot3\cdot5\cdot5 + 32 = 2432$ tham số, bất kể ảnh lớn hay nhỏ.

### 2.5 Kernel, feature map, channel

| Thuật ngữ | Định nghĩa |
|---|---|
| **Kernel / filter** | tập trọng số học được, shape $C_{in}\times K\times K$ |
| **Feature map** | output 2D khi áp **một** kernel lên toàn bộ input |
| **Channel** | một "lát" theo chiều sâu của tensor. Ở input, channel là R, G, B. Ở các lớp sau, mỗi channel chính là một feature map. |

---

## 3. Padding và Stride

- **Padding $P$:** thêm $P$ hàng/cột (thường là số 0) quanh viền ảnh, để giữ kích thước output và để pixel ở viền cũng được "nhìn" đầy đủ.
- **Stride $S$:** bước nhảy của kernel. $S = 2$ làm chiều cao và chiều rộng giảm khoảng một nửa.

**Công thức kích thước output** (áp dụng cho mỗi chiều không gian):

$$
N_{out} = \left\lfloor\frac{N + 2P - K}{S}\right\rfloor + 1
$$

| $N$ | $K$ | $P$ | $S$ | $N_{out}$ | Ghi chú |
|---|---|---|---|---|---|
| 32 | 3 | 0 | 1 | 30 | "valid": co lại $K - 1$ |
| 32 | 3 | 1 | 1 | 32 | "same": giữ kích thước |
| 32 | 3 | 1 | 2 | 16 | giảm một nửa |
| 64 | 5 | 2 | 1 | 64 | "same" với $P = (K-1)/2$ |
| 64 | 5 | 2 | 2 | 32 | TUT03 Exercise 2 |

**TUT03, Exercise 2:** input $8\times3\times64\times64$ qua `Conv2d(3, 32, kernel_size=5, stride=2, padding=2)` cho ra **$8\times32\times32\times32$**:
- batch giữ nguyên là 8;
- số kênh bằng số kernel, là 32;
- kích thước không gian $\lfloor(64+4-5)/2\rfloor + 1 = 32$.

> Mẹo: với stride 1, chọn $P = (K-1)/2$ (với $K$ lẻ) thì kích thước output bằng input.

---

## 4. ReLU trong CNN

Convolution là phép **tuyến tính** theo input. Sau mỗi conv ta đặt một hàm phi tuyến, thường là ReLU, vì nếu không có nó, nhiều lớp conv chồng lên nhau vẫn chỉ tương đương một phép tuyến tính (xem [file 01](01_mang_neural_forward_pass.md)).

Trên feature map, ReLU giữ lại các response dương (ví dụ: "có cạnh theo hướng này") và xóa các response âm.

---

## 5. Pooling

**Max pooling** lấy giá trị lớn nhất trong mỗi cửa sổ:

$$
y_{i,j} = \max(\text{cửa sổ cục bộ})
$$

Ví dụ max pooling $2\times2$, stride 2:

$$
\begin{bmatrix}1&3&2&0\\5&6&1&2\\0&2&4&4\\1&1&3&8\end{bmatrix}
\;\xrightarrow{\text{max-pool }2\times2}\;
\begin{bmatrix}6&2\\2&8\end{bmatrix}
$$

| Loại | Phép toán | Ghi chú |
|---|---|---|
| Max pooling | max | giữ tín hiệu mạnh nhất: "đặc trưng có xuất hiện trong vùng này không" |
| Average pooling | trung bình | mượt hơn |
| **Global Average Pooling (GAP)** | trung bình **toàn bộ** $H\times W$, mỗi kênh còn 1 số | thay cho flatten trước classifier; cho phép input có kích thước bất kỳ |

Tác dụng của pooling:
- giảm $H, W$ nên giảm lượng tính toán;
- tăng receptive field của các lớp sau;
- tạo ra một chút **bất biến với dịch chuyển nhỏ**.

Pooling **không có tham số**, và cũng **không phải định nghĩa của CNN**. Các CNN hiện đại (như MobileNet, ResNet) thường dùng **conv với stride 2** để giảm kích thước.

---

## 6. Receptive field (vùng tiếp nhận)

**Receptive field** của một neuron là vùng của **ảnh gốc** có thể ảnh hưởng tới giá trị của neuron đó.

- Một conv $3\times3$ có receptive field $3\times3$.
- Hai conv $3\times3$ liên tiếp (stride 1) có receptive field $5\times5$. Ba conv thì $7\times7$.
- Stride và pooling làm receptive field **tăng nhanh hơn** theo độ sâu.

Do đó các lớp sâu "nhìn" được vùng ngữ cảnh rộng hơn và tổng hợp được các mẫu hình phức tạp hơn. Đây là ý tưởng **phân cấp đặc trưng (feature hierarchy)**:

```text
pixel → cạnh, màu (lớp đầu) → texture, góc (lớp giữa) → bộ phận (mắt, bánh xe)
      → toàn vật thể (lớp sâu) → global pooling → logit của từng lớp
```

> Hai conv $3\times3$ có cùng receptive field với một conv $5\times5$, nhưng ít tham số hơn ($2\cdot9C^2 = 18C^2$ so với $25C^2$) và có thêm một lớp phi tuyến. Đây là lý do VGG chỉ dùng kernel $3\times3$.

---

## 7. Kiến trúc CNN điển hình và cách theo dõi shape

Tensor ảnh trong PyTorch có dạng **$(N, C, H, W)$**: batch, kênh, chiều cao, chiều rộng.

**TinyCNN trong TUT03, Part 3:**

```text
Input                          2 × 3  × 32 × 32
Conv2d(3→8, k3, p1)  + ReLU →  2 × 8  × 32 × 32      (224 tham số)
MaxPool2d(2)               →  2 × 8  × 16 × 16
Conv2d(8→16, k3, p1) + ReLU →  2 × 16 × 16 × 16      (1168 tham số)
MaxPool2d(2)               →  2 × 16 × 8  × 8
AdaptiveAvgPool2d(1)       →  2 × 16 × 1  × 1      (GAP)
Flatten                    →  2 × 16
Linear(16→4)               →  2 × 4  logits        (68 tham số)
```

Tổng cộng $224 + 1168 + 68 = 1460$ tham số.

**Quy luật chung:** càng vào sâu, **kích thước không gian giảm** còn **số kênh tăng**. Mạng đánh đổi thông tin "ở đâu" để lấy thông tin "là cái gì".

Cấu trúc tổng quát:

$$
\underbrace{[\text{Conv} \to \text{(BN)} \to \text{ReLU} \to \text{(Pool)}] \times n}_{\text{feature extractor / backbone}} \;\to\; \text{GAP / Flatten} \;\to\; \underbrace{\text{Linear}}_{\text{classifier / head}}
$$

---

## 8. Batch Normalization

BatchNorm chuẩn hóa activation của mỗi kênh theo batch, rồi co giãn và dịch lại bằng hai tham số học được $\gamma$, $\beta$:

$$
\hat{x} = \frac{x - \mu_{batch}}{\sqrt{\sigma^2_{batch} + \epsilon}}, \qquad y = \gamma\hat{x} + \beta
$$

- **Lúc train:** dùng mean/var của batch hiện tại, đồng thời cập nhật **running mean/var** (trung bình trượt).
- **Lúc eval:** dùng running mean/var đã lưu, nên kết quả không phụ thuộc vào batch.

Lợi ích: train ổn định hơn, dùng được learning rate lớn hơn, và có tác dụng regularization nhẹ. Cần lưu ý cách xử lý BatchNorm khi fine-tune với backbone bị đóng băng (xem [file 04](04_transfer_learning.md)).

---

## 9. Các kiến trúc CNN tiêu biểu

| Kiến trúc | Năm | Ý tưởng chính |
|---|---|---|
| **LeNet-5** | 1998 | CNN đầu tiên thành công (nhận dạng chữ số viết tay): conv → pool → FC |
| **AlexNet** | 2012 | CNN sâu trên GPU, dùng ReLU và Dropout, thắng ImageNet và mở đầu kỷ nguyên deep learning |
| **VGG** | 2014 | chỉ dùng conv $3\times3$ xếp chồng, sâu 16–19 lớp |
| **GoogLeNet / Inception** | 2014 | nhiều nhánh conv với kernel khác nhau chạy song song |
| **ResNet** | 2015 | **kết nối tắt (residual)** $y = F(x) + x$, cho phép train mạng sâu 100+ lớp vì gradient đi thẳng qua đường tắt |
| **MobileNet (V1–V3)** | 2017–2019 | **depthwise separable convolution**, nhẹ, phù hợp cho thiết bị di động |

### Depthwise separable convolution (nền tảng của MobileNet)

Tách một conv chuẩn thành hai bước:
1. **Depthwise:** mỗi kênh có một kernel $K\times K$ riêng (không trộn các kênh với nhau);
2. **Pointwise:** conv $1\times1$ để trộn các kênh.

| | Chi phí (MACs) |
|---|---|
| Conv chuẩn | $K^2 \cdot C_{in}\cdot C_{out}\cdot H\cdot W$ |
| Depthwise separable | $K^2\cdot C_{in}\cdot H\cdot W + C_{in}\cdot C_{out}\cdot H\cdot W$ |
| **Tỉ lệ** | $\dfrac{1}{C_{out}} + \dfrac{1}{K^2}$ |

Với $K = 3$ và $C_{out} = 64$, tỉ lệ là $\approx 0.127$, tức **giảm gần 8 lần** lượng tính toán.

**MobileNetV3-Small** (model dùng trong TUT03) còn kết hợp thêm:
- inverted residual block;
- squeeze-and-excitation (SE);
- hàm kích hoạt Hardswish;
- kiến trúc được tìm tự động bằng NAS.

Phiên bản trong lab có 1,52M tham số và ≈ 0,021 GMACs ở ảnh $128\times128$.

---

## 10. CNN thực sự học được gì? (TUT03, Part 6)

Dùng forward hook để lấy activation ở lớp đầu (`features[0]`) và lớp sâu (`features[-3]`) của MobileNetV3, với ảnh $128\times128$:

| | Lớp đầu | Lớp sâu |
|---|---|---|
| Shape | $1\times16\times64\times64$ | $1\times96\times4\times4$ |
| Độ phân giải không gian | cao, còn thấy rõ hình dạng vật thể | rất thấp |
| Số kênh | ít | nhiều |
| Receptive field | nhỏ | gần như cả ảnh |
| Nội dung | cạnh, màu, tương phản | tổ hợp trừu tượng, gần với quyết định phân loại |

> Đừng diễn giải mỗi kênh như một khái niệm rõ ràng mà con người hiểu được. Biểu diễn của mạng thường **phân tán (distributed)**: một khái niệm được mã hóa bởi nhiều kênh, và một kênh có thể tham gia vào nhiều khái niệm.

---

## 11. Tính chất của convolution

- **Translation equivariance:** dịch input bao nhiêu thì feature map dịch theo bấy nhiêu. Đó là hệ quả của weight sharing.
- **Translation invariance (xấp xỉ):** nhờ pooling và GAP, **đầu ra phân loại** gần như không đổi khi vật thể dịch chuyển trong ảnh.
- CNN **không** tự nhiên bất biến với phép xoay hay thay đổi tỉ lệ. Muốn có tính chất đó cần data augmentation hoặc kiến trúc đặc biệt.

---

## 12. Câu hỏi ôn tập

<details>
<summary>1. Vì sao cùng một kernel phát hiện được một mẫu hình ở nhiều vị trí khác nhau?</summary>

Nhờ weight sharing: cùng một bộ trọng số được áp ở mọi vị trí, với cùng một phép nhân-cộng. Vì vậy cùng một mẫu hình cho cùng một response, bất kể nó nằm ở đâu trong ảnh.
</details>

<details>
<summary>2. Input 32×32, kernel 3×3, padding 1, stride 2. Output có kích thước bao nhiêu?</summary>

$\lfloor(32+2-3)/2\rfloor + 1 = 16$, tức $16\times16$.
</details>

<details>
<summary>3. <code>Conv2d(64, 128, kernel_size=3)</code> có bao nhiêu tham số?</summary>

$128\cdot64\cdot3\cdot3 + 128 = 73.856$.
</details>

<details>
<summary>4. Vì sao số tham số của Conv không phụ thuộc vào kích thước ảnh, còn số MACs thì có?</summary>

Kernel được dùng chung cho mọi vị trí, nên số tham số cố định. Nhưng kernel phải được **tính** tại mọi vị trí $(h, w)$, nên lượng tính toán tỉ lệ với $H\times W$.
</details>

<details>
<summary>5. Global Average Pooling có lợi gì so với Flatten?</summary>

GAP không có tham số, giảm mạnh số tham số của lớp FC phía sau, giảm overfit, và cho phép input có kích thước bất kỳ. MobileNetV3 nhận được ảnh 96, 128 hay 160 px là nhờ GAP.
</details>

<details>
<summary>6. Trong một CNN, khi đi sâu hơn thì H, W và C thay đổi thế nào? Vì sao?</summary>

H và W giảm (do pooling/stride), còn C tăng. Mạng tổng hợp ngữ cảnh rộng hơn thành nhiều loại đặc trưng trừu tượng hơn, và đánh đổi độ chính xác về vị trí lấy thông tin ngữ nghĩa.
</details>

---

## 13. Tóm tắt (theo TUT03, Part 10)

1. **Convolution là phép toán cục bộ.**
2. **Trọng số được chia sẻ** giữa mọi vị trí.
3. **Nhiều kernel tạo ra nhiều feature map.**
4. **ReLU thêm tính phi tuyến.**
5. **Pooling hoặc stride giảm độ phân giải không gian.**
6. **Các lớp sâu tổng hợp ngữ cảnh ngày càng rộng.**

**Trước:** [02 — Huấn luyện](02_huan_luyen_mang_neural.md) · **Tiếp theo:** [04 — Transfer learning](04_transfer_learning.md)
