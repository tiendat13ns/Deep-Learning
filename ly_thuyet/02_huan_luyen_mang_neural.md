# 02 — Huấn luyện mạng neural (Training)

> **Lab tương ứng:** TUT02 — *Train Your First Neural Network with PyTorch* (phân loại PASS/FAIL cho sinh viên)
> **Câu hỏi trung tâm:** Làm sao để mạng tự tìm ra bộ tham số hữu ích từ dữ liệu?

Ở [file 01](01_mang_neural_forward_pass.md), tham số được chọn bằng tay. Training là quá trình **tự động điều chỉnh tham số** để giảm sai số trên dữ liệu có nhãn.

---

## 1. Học có giám sát (Supervised learning)

Dữ liệu gồm các cặp $(\mathbf{x}_i, y_i)$:

- $\mathbf{x}$ — **features** (đặc trưng đầu vào). TUT02 dùng `[attendance, study_hours, midterm_score]`.
- $y$ — **label / target** (nhãn). TUT02 dùng `0 = FAIL`, `1 = PASS`.

Mục tiêu là tìm hàm $f_\theta$ (mạng với tham số $\theta$) sao cho $f_\theta(\mathbf{x}) \approx y$, kể cả trên **dữ liệu mới chưa từng thấy**. Khả năng này gọi là **tổng quát hóa (generalization)**.

| Loại bài toán | Ví dụ | Đầu ra mạng | Hàm loss điển hình |
|---|---|---|---|
| Phân loại nhị phân | PASS/FAIL | 1 logit | Binary Cross-Entropy |
| Phân loại nhiều lớp | 4 lớp CIFAR (TUT03) | $K$ logit | Cross-Entropy |
| Hồi quy | Dự báo CO₂ (TUT04) | số thực | MSE / MAE |

Trong thực tế ta **không biết quy luật thật** sinh ra nhãn. TUT02 tự sinh dữ liệu bằng một quy luật ẩn cộng nhiễu, gồm 1.200 sinh viên với tỉ lệ pass 63.6%. Model phải tự suy ra quy luật từ các ví dụ. Nhiễu khiến bài toán không thể đạt 100%, giống như dữ liệu thật.

---

## 2. Chuẩn bị dữ liệu

### 2.1 Chia train / validation / test

| Tập | Dùng để | Có được dùng để cập nhật trọng số không? |
|---|---|---|
| **Train** | tính gradient, cập nhật tham số | có |
| **Validation** | chọn hyperparameter, chọn epoch hoặc checkpoint, early stopping | không |
| **Test** | đánh giá **một lần** ở cuối, ước lượng hiệu năng trên dữ liệu mới | không, và **không** dùng để ra quyết định |

TUT02 chỉ có train/test (960/240, tỉ lệ 80/20). Các lab TUT03 và TUT04 dùng đủ ba tập.

### 2.2 Data leakage (rò rỉ dữ liệu)

**Leakage** là khi thông tin từ tập test (hoặc từ tương lai) lọt vào quá trình train. Hậu quả là kết quả đánh giá **lạc quan giả tạo**.

> **Quy tắc vàng:** *chia dữ liệu trước, rồi mới tính các thống kê tiền xử lý (mean, std, ...) trên **tập train**.*

Một số dạng leakage thường gặp:
- chuẩn hóa bằng mean/std tính trên **toàn bộ** dữ liệu;
- xáo trộn ngẫu nhiên chuỗi thời gian trước khi chia (xem [file 07](07_du_bao_chuoi_thoi_gian.md));
- chọn hyperparameter dựa trên điểm của tập test.

### 2.3 Chuẩn hóa (Standardization)

Các feature trong TUT02 có thang đo rất khác nhau: attendance 45–100, study hours 0–15, midterm 0–10. Ta chuẩn hóa về mean ≈ 0 và std ≈ 1:

$$
x_{scaled} = \frac{x - \mu_{train}}{\sigma_{train}}
$$

**Vì sao giúp training?**
- Feature nào có thang đo lớn thì gradient theo trọng số của nó cũng lớn, làm bề mặt loss bị "kéo dài" theo một hướng. Gradient descent khi đó đi zig-zag và hội tụ chậm.
- Sau khi chuẩn hóa, các hướng có độ cong tương đương nhau, nên một learning rate chung phù hợp cho mọi trọng số.
- Activation ban đầu nằm trong vùng hợp lý, tránh làm sigmoid hay tanh bão hòa.

**Kết quả TUT02, Exercise 4:** dữ liệu thô hội tụ chậm hơn rõ rệt (sau 10 epoch loss ≈ 0.55, so với 0.37 khi đã chuẩn hóa) và cho test accuracy thấp hơn (0.808 so với 0.821). Model vẫn học được, chỉ kém hiệu quả hơn. Không nên kết luận rằng chuẩn hóa *luôn* bắt buộc, nhưng **thang đo của input có ảnh hưởng tới quá trình tối ưu**.

> Phải dùng **cùng** $\mu_{train}$, $\sigma_{train}$ cho tập test và cho mọi input mới khi inference. Train với một pipeline tiền xử lý nhưng inference với pipeline khác là một bug rất phổ biến trong thực tế.

---

## 3. Tensor, Dataset, DataLoader

| Đối tượng | Vai trò |
|---|---|
| **Tensor** | mảng số nhiều chiều (giống NumPy array) mà PyTorch đặt được lên GPU và tính được gradient |
| **Dataset** (`TensorDataset`) | lưu các cặp (mẫu, nhãn), lấy được phần tử thứ $i$ |
| **DataLoader** | chia dữ liệu thành **mini-batch**, xáo trộn (shuffle) mỗi epoch |

**Shape:** một sinh viên có shape `(3,)`. Một batch 32 sinh viên có shape `(32, 3)`, tức `(batch, features)`. Chiều đầu tiên **luôn là batch**.

### Batch, iteration, epoch

- **Batch size** $B$: số mẫu được xử lý trong một bước cập nhật (TUT02 dùng 32).
- **Iteration / step**: một lần cập nhật tham số, ứng với một batch.
- **Epoch**: một lượt duyệt qua **toàn bộ** tập train. Với 960 mẫu và $B = 32$, mỗi epoch có 30 iteration.

| Kiểu | Batch size | Đặc điểm |
|---|---|---|
| Batch gradient descent | toàn bộ dữ liệu | gradient chính xác nhưng chậm, tốn bộ nhớ |
| Stochastic GD (SGD) | 1 | rất nhiễu, cập nhật nhiều lần |
| **Mini-batch GD** | 16–512 | cân bằng: tận dụng được GPU, nhiễu vừa phải giúp thoát khỏi điểm tồi |

---

## 4. Kiến trúc và logit

MLP trong TUT02: `3 → Linear(3,8) → ReLU → Linear(8,4) → ReLU → Linear(4,1) → 1 logit`.

Số tham số: $(3\cdot8+8) + (8\cdot4+4) + (4\cdot1+1) = 32 + 36 + 5 = 73$.

Những nguyên tắc thiết kế rút ra từ lab:
1. Kích thước input phải bằng số feature.
2. Bắt đầu với mạng nhỏ: bài toán dạng bảng đơn giản không cần mạng lớn.
3. Xen hàm phi tuyến (ReLU) giữa các lớp Linear.
4. Phân loại nhị phân chỉ cần **1 đầu ra**, vì $P(\text{FAIL}) = 1 - P(\text{PASS})$.
5. **Không** đặt Sigmoid trong model khi train bằng `BCEWithLogitsLoss` (xem mục 5).
6. Kiểm tra shape bằng một forward pass trước khi train.
7. Tham số được khởi tạo ngẫu nhiên. Đây không phải model pretrained.

---

## 5. Hàm mất mát (Loss function)

Loss $\mathcal{L}(\theta)$ là **một con số** đo mức độ sai của dự đoán so với nhãn. Training chính là bài toán **tối thiểu hóa loss**:

$$
\theta^* = \arg\min_\theta \frac{1}{N}\sum_{i=1}^N \ell\big(f_\theta(\mathbf{x}_i), y_i\big)
$$

> Optimizer đi theo gradient của **loss**, không phải của accuracy. Accuracy không khả vi: nó là hàm bậc thang, nên gradient bằng 0 hầu như ở mọi nơi. Vì vậy accuracy chỉ dùng để **theo dõi**.

### 5.1 Binary Cross-Entropy (BCE)

Với $p = \sigma(z)$ là xác suất dự đoán của lớp 1:

$$
\ell(p, y) = -\big[\,y\log p + (1-y)\log(1-p)\,\big]
$$

| Nhãn thật $y = 1$, dự đoán $p$ | Loss $-\log p$ |
|---|---|
| 0.9 (đúng, tự tin) | 0.105 |
| 0.6 (đúng, chưa chắc) | 0.511 |
| 0.1 (sai, tự tin) | 2.303 |

BCE **phạt rất nặng khi model sai mà lại tự tin**. Về mặt thống kê, tối thiểu BCE tương đương với **maximum likelihood** khi giả định nhãn tuân theo phân phối Bernoulli.

**Gradient theo logit rất gọn:**

$$
\frac{\partial \ell}{\partial z} = p - y
$$

Model dự đoán càng sai thì gradient càng lớn.

### 5.2 Vì sao dùng `BCEWithLogitsLoss`?

`BCEWithLogitsLoss` nhận **logit**, rồi gộp sigmoid và BCE thành một công thức ổn định về mặt số học (dùng mẹo log-sum-exp). Nếu tự tính `sigmoid` rồi `log`, khi $z$ rất lớn hoặc rất nhỏ thì $\sigma(z)$ tròn thành 1 hoặc 0, và $\log 0 = -\infty$ gây ra NaN.

```
train:      logits = model(x)        →  BCEWithLogitsLoss(logits, y)
inference:  p = torch.sigmoid(logits) →  pred = (p >= threshold)
```

### 5.3 Cross-Entropy cho nhiều lớp

Với $K$ lớp, logit $\mathbf{z} \in \mathbb{R}^K$ và $\mathbf{p} = \text{softmax}(\mathbf{z})$:

$$
\ell = -\log p_{y} = -z_y + \log\sum_j e^{z_j}
$$

`nn.CrossEntropyLoss` nhận logit và chỉ số lớp đúng, đã bao gồm softmax bên trong. Lab TUT03 dùng hàm này với 4 lớp.

### 5.4 MSE cho hồi quy

$$
\text{MSE} = \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2
$$

Lab TUT04 dùng MSE để dự báo mức thay đổi CO₂.

---

## 6. Gradient Descent

Gradient $\nabla_\theta\mathcal{L}$ chỉ hướng làm loss **tăng nhanh nhất**. Vì vậy ta đi ngược hướng gradient:

$$
\theta \leftarrow \theta - \eta\,\nabla_\theta\mathcal{L}
$$

$\eta$ là **learning rate (tốc độ học)**, quyết định độ dài mỗi bước.

| Learning rate | Hiện tượng |
|---|---|
| Quá nhỏ | học rất chậm, có thể chưa hội tụ khi hết số epoch |
| Vừa phải | loss giảm đều đặn |
| Quá lớn | loss dao động hoặc tăng vọt, bước "nhảy qua" điểm cực tiểu, có thể ra NaN |

**Kết quả TUT02, Exercise 3** (model `3→16→8→1`, Adam):

| lr | Train acc cuối | Test acc | Nhận xét |
|---|---|---|---|
| 0.0001 | 0.640 | 0.621 | Học quá chậm: accuracy chỉ ngang tỉ lệ pass (63.6%), tức model gần như đoán PASS cho tất cả |
| 0.01 | 0.820 | 0.821 | Tốt nhất |
| 0.1 | 0.814 | 0.817 | Chưa mất ổn định, nhưng test loss tệ hơn (0.470 so với 0.414) |

Learning rate là **hyperparameter quan trọng nhất**, và không có giá trị nào đúng cho mọi bài toán.

---

## 7. Backpropagation (lan truyền ngược)

Backpropagation là thuật toán tính $\nabla_\theta\mathcal{L}$ cho **mọi** tham số một cách hiệu quả bằng **quy tắc chuỗi (chain rule)**. Nó đi ngược từ loss về input, và tái sử dụng các giá trị trung gian đã lưu trong forward pass.

### 7.1 Quy tắc chuỗi

Nếu $\mathcal{L} = \ell(a)$, $a = \sigma(z)$, $z = wx + b$, thì:

$$
\frac{\partial\mathcal{L}}{\partial w} = \frac{\partial\mathcal{L}}{\partial a}\cdot\frac{\partial a}{\partial z}\cdot\frac{\partial z}{\partial w}
$$

### 7.2 Ví dụ tính tay: một neuron sigmoid với BCE

Cho $\mathbf{x} = [1, 2]$, $\mathbf{w} = [0.5, -0.3]$, $b = 0.1$, nhãn $y = 1$, learning rate $\eta = 0.1$.

**Forward:**
- $z = 0.5(1) - 0.3(2) + 0.1 = 0$
- $p = \sigma(0) = 0.5$
- $\mathcal{L} = -\log 0.5 = 0.693$

**Backward** (dùng $\partial\mathcal{L}/\partial z = p - y$):
- $\partial\mathcal{L}/\partial z = 0.5 - 1 = -0.5$
- $\partial\mathcal{L}/\partial\mathbf{w} = (p - y)\,\mathbf{x} = [-0.5,\ -1.0]$
- $\partial\mathcal{L}/\partial b = -0.5$

**Update:**
- $\mathbf{w} \leftarrow [0.5, -0.3] - 0.1\cdot[-0.5, -1.0] = [0.55,\ -0.2]$
- $b \leftarrow 0.1 - 0.1(-0.5) = 0.15$

**Kiểm tra:** $z_{mới} = 0.55 - 0.4 + 0.15 = 0.3$, nên $p = 0.574$ và $\mathcal{L} = 0.554$. Loss đã giảm từ 0.693 xuống 0.554.

### 7.3 Dạng tổng quát cho MLP

Với lớp $l$: $\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ và $\mathbf{a}^{(l)} = f(\mathbf{z}^{(l)})$. Đặt $\boldsymbol{\delta}^{(l)} = \partial\mathcal{L}/\partial\mathbf{z}^{(l)}$. Khi đó:

$$
\boldsymbol{\delta}^{(l)} = \big(W^{(l+1)\top}\boldsymbol{\delta}^{(l+1)}\big) \odot f'(\mathbf{z}^{(l)})
$$

$$
\frac{\partial\mathcal{L}}{\partial W^{(l)}} = \boldsymbol{\delta}^{(l)}\,\mathbf{a}^{(l-1)\top}, \qquad \frac{\partial\mathcal{L}}{\partial\mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}
$$

Hai điểm cần nhớ:
- Gradient của một trọng số tỉ lệ với **activation đầu vào** của nó. Nếu activation đó bằng 0 (ví dụ bị ReLU cắt), trọng số đó không được cập nhật trong bước này.
- Gradient bị nhân với $f'(z)$ ở mỗi lớp. Nếu $f'$ thường nhỏ (sigmoid có $f' \le 0.25$), gradient ở các lớp đầu sẽ co lại rất nhanh. Đó là **vanishing gradient**, chi tiết ở [file 06](06_rnn_lstm_gru.md).

### 7.4 Autograd trong PyTorch

PyTorch ghi lại đồ thị tính toán trong lúc chạy forward. `loss.backward()` đi ngược đồ thị đó và **cộng dồn** gradient vào thuộc tính `.grad` của mọi tensor có `requires_grad=True`. Đó chính là các tham số trainable.

---

## 8. Optimizer

**SGD:** $\theta \leftarrow \theta - \eta\,g$.

**SGD + Momentum:** tích lũy "vận tốc" để đi nhanh hơn theo hướng ổn định và giảm dao động:

$$
v \leftarrow \beta v + g, \qquad \theta \leftarrow \theta - \eta v
$$

**Adam** (lab dùng Adam với lr = 0.01): kết hợp momentum (moment bậc 1) với việc chia cho độ lớn trung bình của gradient (moment bậc 2), nên **mỗi tham số có một bước nhảy thích nghi riêng**:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \\
\hat m_t &= \frac{m_t}{1-\beta_1^t}, \quad \hat v_t = \frac{v_t}{1-\beta_2^t} \quad \text{(bias correction)} \\
\theta_t &= \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
\end{aligned}
$$

Giá trị mặc định thường dùng: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

**AdamW** (dùng trong TUT03) là Adam với **weight decay tách rời**: mỗi bước kéo trọng số về 0 một lượng tỉ lệ thuận với trọng số. Đây là một dạng regularization.

---

## 9. Vòng lặp training

Năm thao tác cốt lõi của **một bước train**:

```python
model.train()
for X_batch, y_batch in train_loader:
    optimizer.zero_grad()              # 1. xóa gradient cũ
    logits = model(X_batch)            # 2. forward
    loss = loss_fn(logits, y_batch)    # 3. tính loss
    loss.backward()                    # 4. backward: tính gradient
    optimizer.step()                   # 5. cập nhật tham số
```

| Bước | Vì sao cần |
|---|---|
| `zero_grad()` | PyTorch **cộng dồn** gradient vào `.grad`. Không xóa thì gradient của batch trước sẽ cộng vào batch hiện tại. Cơ chế cộng dồn này đôi khi được dùng có chủ đích (gradient accumulation), nhưng trong vòng lặp thông thường thì phải xóa. |
| `backward()` | tính $\partial\mathcal{L}/\partial\theta$ |
| `step()` | áp dụng quy tắc cập nhật của optimizer. Đây là bước **duy nhất** thực sự thay đổi tham số. |

TUT02, mục 11, kiểm chứng điều này: sau một bước, trọng số `network.0.weight[0,0]` thực sự thay đổi. Điểm khác biệt so với TUT01 nằm ở bước 4 và 5.

**Đọc log training.** Một lần chạy bình thường có các dấu hiệu: loss giảm dần, accuracy tăng dần, và các epoch sau thay đổi ít hơn các epoch đầu. Không cần nhớ con số cụ thể, vì chúng có thể khác nhau giữa các môi trường. Điều cần chú ý là **xu hướng** có đi đúng hướng hay không.

---

## 10. Chế độ train và eval, inference

| | `model.train()` | `model.eval()` |
|---|---|---|
| Dropout | bật (tắt ngẫu nhiên một số neuron) | tắt |
| BatchNorm | dùng thống kê của batch hiện tại, cập nhật running mean/var | dùng running mean/var đã lưu |

`model.eval()` **không** tắt gradient. Muốn tắt gradient khi inference, phải dùng thêm `torch.inference_mode()` hoặc `torch.no_grad()`.

**Quy trình inference:**
1. `model.eval()`
2. `with torch.inference_mode():`
3. tiền xử lý input mới bằng **cùng thứ tự feature và cùng $\mu$, $\sigma$ của tập train**
4. forward → logit → sigmoid → xác suất
5. so sánh với **ngưỡng (threshold)** để ra nhãn

### Threshold không phải là training

Đổi ngưỡng **không** thay đổi tham số. Nó chỉ thay đổi **quy tắc quyết định** áp lên cùng một tập xác suất.

TUT02, Exercise 6, trên 240 mẫu test:

| Threshold | Dự đoán PASS | Test accuracy |
|---|---|---|
| 0.3 | 176 | 0.804 |
| 0.5 | 156 | 0.821 |
| 0.7 | 137 | 0.825 |

Ngưỡng thấp thì nhiều mẫu được dự đoán là dương tính hơn, nên **recall tăng nhưng precision giảm**. Ngưỡng cao thì ngược lại. Chọn ngưỡng nào phụ thuộc vào chi phí của từng loại lỗi (xem [file 05](05_danh_gia_mo_hinh.md)).

---

## 11. Capacity, underfitting, overfitting

**Capacity** (năng lực biểu diễn) là độ phức tạp của các hàm mà model có thể học. Capacity tăng theo số lớp, số neuron và số tham số.

| Tình trạng | Train loss | Val/Test loss | Nguyên nhân | Cách xử lý |
|---|---|---|---|---|
| **Underfitting** | cao | cao | model quá đơn giản, train chưa đủ lâu, lr không phù hợp | tăng capacity, train lâu hơn, chỉnh lr |
| **Vừa khớp** | thấp | thấp, gần train | | |
| **Overfitting** | rất thấp | cao hơn hẳn train, hoặc đang **tăng** | model "học thuộc" cả nhiễu của tập train | thêm dữ liệu, regularization, early stopping, giảm capacity |

**Kết quả TUT02, Exercise 2:** model `3→16→8→1` có 209 tham số, model gốc có 73. Hai model có test accuracy như nhau (0.821), nhưng model lớn có test loss tệ hơn (0.414 so với 0.396). Bài toán chỉ có 3 feature và quy luật đơn giản, nên capacity thêm vào không có ích. **Nhiều tham số hơn không tự động tốt hơn.**

### Các kỹ thuật regularization

- **Weight decay (L2):** cộng thêm $\lambda\|\theta\|^2$ vào loss, để trọng số không quá lớn.
- **Dropout:** khi train, đặt ngẫu nhiên một tỉ lệ $p$ activation bằng 0, buộc mạng không phụ thuộc vào một neuron nào. Head của MobileNetV3 trong TUT03 có `Dropout(p=0.2)`.
- **Early stopping:** dừng train khi validation loss không cải thiện thêm sau một số epoch (patience), và giữ lại checkpoint tốt nhất. TUT04 dùng patience = 3.
- **Data augmentation:** tạo biến thể của dữ liệu train, ví dụ lật hoặc cắt ảnh (TUT03).
- **Transfer learning:** bắt đầu từ trọng số đã học trên dữ liệu lớn ([file 04](04_transfer_learning.md)).

---

## 12. Tái lập kết quả (Reproducibility)

Mạng khởi tạo tham số ngẫu nhiên, và DataLoader cũng xáo trộn dữ liệu ngẫu nhiên. Đặt **seed** (`random.seed`, `np.random.seed`, `torch.manual_seed`) giúp loại bỏ các nguồn ngẫu nhiên chính, nên kết quả chạy lại sẽ giống nhau. Seed **không** đảm bảo kết quả giống hệt nhau giữa các phần cứng hoặc phiên bản thư viện khác nhau, đặc biệt trên GPU.

Một model đạt kết quả tốt ở một seed chưa chắc tốt ở seed khác. Xem phần thực nghiệm nhiều seed ở [file 05](05_danh_gia_mo_hinh.md).

---

## 13. Toàn bộ pipeline

```text
dữ liệu → X, y → chia train/test → chuẩn hóa bằng thống kê TRAIN
→ Dataset + DataLoader → MLP → logit → loss → backward → optimizer.step
→ lặp qua nhiều batch/epoch → model.eval() + inference_mode()
→ input mới → xác suất → threshold → PASS/FAIL
```

| Đối tượng | Trách nhiệm |
|---|---|
| Tensor | dữ liệu số |
| Dataset | lưu mẫu và nhãn |
| DataLoader | tạo batch |
| Model | ánh xạ input → logit |
| Loss | đo sai số |
| Optimizer | cập nhật tham số bằng gradient |
| Training loop | lặp các bước học |
| Inference code | dùng tham số đã học (cố định) cho input mới |

> **architecture = cấu trúc** · **parameters = các con số đã học bên trong cấu trúc đó**
> Training chỉ thay đổi **giá trị tham số**, không thay đổi kiến trúc.

---

## 14. Câu hỏi ôn tập

<details>
<summary>1. Trong quá trình training, kiến trúc hay giá trị tham số thay đổi?</summary>

Chỉ giá trị tham số thay đổi. Kiến trúc, ví dụ `3→8→4→1`, giữ nguyên.
</details>

<details>
<summary>2. Vì sao phải gọi <code>optimizer.zero_grad()</code> trước <code>backward()</code>?</summary>

Vì PyTorch cộng dồn gradient. Nếu không xóa, gradient của batch trước sẽ cộng vào batch hiện tại, khiến bước cập nhật bị sai.
</details>

<details>
<summary>3. Vì sao model trả về logit thay vì chữ PASS/FAIL?</summary>

Mạng chỉ tạo ra số. Logit khả vi nên dùng được để tính loss và gradient. Sigmoid chuyển logit thành xác suất, và threshold chuyển xác suất thành nhãn. `BCEWithLogitsLoss` làm việc trực tiếp với logit và ổn định về mặt số học.
</details>

<details>
<summary>4. Vì sao input mới phải có cùng thứ tự feature và cùng cách chuẩn hóa như lúc train?</summary>

Trọng số được học cho một thứ tự và một thang đo cụ thể. Nếu đổi thứ tự hoặc thang đo, các con số mang ý nghĩa khác, và dự đoán sẽ sai.
</details>

<details>
<summary>5. <code>model.train()</code> và <code>model.eval()</code> khác nhau thế nào?</summary>

Hai chế độ này thay đổi hành vi của các layer như Dropout và BatchNorm. Chúng không tự tính gradient và không tự cập nhật trọng số.
</details>

<details>
<summary>6. Train accuracy 0.99, test accuracy 0.70 là dấu hiệu gì?</summary>

Overfitting. Có thể xử lý bằng regularization, thêm dữ liệu, early stopping hoặc giảm capacity.
</details>

<details>
<summary>7. Vì sao không dùng accuracy làm loss?</summary>

Accuracy không khả vi: nó là hàm bậc thang, gradient bằng 0 hầu như ở mọi nơi, nên không cho tín hiệu để cập nhật trọng số.
</details>

---

**Trước:** [01 — Mạng neural & forward pass](01_mang_neural_forward_pass.md) · **Tiếp theo:** [03 — CNN](03_cnn.md)
