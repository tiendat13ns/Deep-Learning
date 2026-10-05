# 08 — Cơ chế Attention

> **Lab tương ứng:** TUT05, Part I — *Attention from first principles* và Exercise 1–3
> **Ý tưởng chính:** Attention là **truy xuất thông tin theo mức độ liên quan**. Mỗi vị trí tự "hỏi" xem nó cần thông tin gì, so sánh câu hỏi đó với mọi vị trí khác, rồi lấy **trung bình có trọng số** thông tin của các vị trí đó.

---

## 1. Động lực: nút thắt của RNN

Trong mô hình seq2seq dùng RNN (ví dụ dịch máy), encoder phải nén **toàn bộ câu nguồn** vào **một vector** $h_T$ có kích thước cố định. Câu càng dài thì thông tin bị mất càng nhiều. Ngoài ra, thông tin ở đầu câu phải đi qua nhiều bước mới tới được decoder (xem [file 06](06_rnn_lstm_gru.md)).

**Attention** (Bahdanau và cộng sự, 2014) cho phép decoder ở mỗi bước **nhìn lại trực tiếp mọi vị trí** của encoder và tự chọn vị trí nào quan trọng. Năm 2017, *"Attention Is All You Need"* (Vaswani và cộng sự) bỏ hẳn RNN, chỉ dùng attention, và tạo ra **Transformer**.

---

## 2. Attention với một query

### 2.1 Phép so sánh với tra cứu dữ liệu

Hãy hình dung một "từ điển mềm":

| Thành phần | Ý nghĩa | Ví dụ khi tìm kiếm thông tin |
|---|---|---|
| **Query (Q)** | "Tôi đang cần thông tin gì?" | câu bạn gõ vào ô tìm kiếm |
| **Key (K)** | "Tôi có thể được tìm thấy bằng từ khóa gì?" | tiêu đề hoặc nhãn của mỗi tài liệu |
| **Value (V)** | "Đây là nội dung tôi mang theo" | nội dung của tài liệu |

Một từ điển thường trả về **đúng một** value có key khớp. Attention thì trả về **tổ hợp của mọi value**, có trọng số theo mức độ khớp giữa query và từng key.

### 2.2 Ba bước

Với một query $\mathbf{q}$, các key $\mathbf{k}_i$ và value $\mathbf{v}_i$:

$$
\underbrace{s_i = \mathbf{q}^\top\mathbf{k}_i}_{\text{1. điểm tương đồng}}
\qquad
\underbrace{\alpha_i = \text{softmax}(\mathbf{s})_i = \frac{e^{s_i}}{\sum_j e^{s_j}}}_{\text{2. chuẩn hóa thành trọng số}}
\qquad
\underbrace{\mathbf{c} = \sum_i\alpha_i\mathbf{v}_i}_{\text{3. tổng có trọng số}}
$$

$\mathbf{c}$ được gọi là **context vector**.

### 2.3 Ví dụ trong TUT05

Bốn token với embedding viết tay (ở ví dụ này Q = K = V = X):

| Token | Vector |
|---|---|
| I | [1, 0, 1, 0] |
| love | [0, 1, 0, 1] |
| deep | [1, 1, 0, 0] |
| learning | [1, 0, 0, 1] |

Query là "learning", $\mathbf{q} = [1, 0, 0, 1]$:
- Score $= X\mathbf{q} = [1, 1, 1, 2]$. "learning" khớp với chính nó nhiều nhất.
- Trọng số $= \text{softmax}([1, 1, 1, 2]) = [0.175,\ 0.175,\ 0.175,\ 0.475]$, tổng bằng 1.
- Context $= \sum\alpha_i\mathbf{x}_i = [0.825,\ 0.350,\ 0.175,\ 0.650]$.

Những điểm cần để ý:
- Trọng số **không âm** và có **tổng bằng 1**, nhờ softmax.
- Context **không phải bản sao** của một token, mà là **hỗn hợp** của nhiều token.
- Đổi query thì phân phối attention thay đổi, nên context cũng thay đổi, dù $X$ giữ nguyên. Attention **phụ thuộc vào ngữ cảnh**.

---

## 3. Vai trò của softmax (TUT05, Exercise 1)

Nếu bỏ softmax và dùng score thô làm trọng số:

| | Có softmax | Không có softmax |
|---|---|---|
| Trọng số âm? | không bao giờ | **có** (4/16 score trong ví dụ Q/K) |
| Tổng bằng 1? | có | **không** (ví dụ toy: tổng = 5; ví dụ Q/K có hàng tổng âm) |
| Độ lớn của output | ổn định: là trung bình có trọng số (convex combination) của các value | **trôi theo độ lớn của score**: norm tăng từ 1,12 lên 5,48 trong ví dụ toy |
| Diễn giải được | "phần trăm chú ý" | không |

Softmax biến mỗi hàng thành một **phân phối xác suất**, giữ output nằm trong "vùng" của các value, và làm việc train ổn định hơn.

---

## 4. Self-attention và Scaled Dot-Product Attention

### 4.1 Self-attention

Trong **self-attention**, Q, K, V đều được tạo ra **từ cùng một chuỗi**, và **mọi** token đều đóng vai trò query. Với input $X \in \mathbb{R}^{L\times d_{model}}$ ($L$ token), ta có ba ma trận chiếu **học được**:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

Ba ma trận chiếu này cho phép một token có "câu hỏi", "từ khóa" và "nội dung" **khác nhau**.

$$
\boxed{\;\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\;}
$$

| Ma trận | Shape |
|---|---|
| $Q, K$ | $L\times d_k$ |
| $V$ | $L\times d_v$ |
| $QK^\top$ (score) | $L\times L$ |
| $A = \text{softmax}(\cdot)$ (attention matrix) | $L\times L$, **mỗi hàng có tổng bằng 1** |
| Output $AV$ | $L\times d_v$ |

**Cách đọc attention matrix:** ô ở hàng $i$, cột $j$ cho biết *"khi cập nhật token $i$, lấy bao nhiêu thông tin từ token $j$?"*.

### 4.2 Vì sao chia cho $\sqrt{d_k}$?

Giả sử các thành phần của $\mathbf{q}$ và $\mathbf{k}$ độc lập, có mean 0 và variance 1. Khi đó:

$$
\mathbf{q}^\top\mathbf{k} = \sum_{i=1}^{d_k}q_ik_i \quad\Rightarrow\quad \text{Var}(\mathbf{q}^\top\mathbf{k}) = d_k, \quad \text{std} = \sqrt{d_k}
$$

$d_k$ càng lớn thì score có độ lớn càng lớn, và softmax tiến gần tới **argmax** (one-hot):
- attention chỉ tập trung vào một token duy nhất;
- softmax bị **bão hòa**, nên gradient gần bằng 0 và model khó học.

Chia cho $\sqrt{d_k}$ đưa variance của score về khoảng 1, bất kể $d_k$ là bao nhiêu.

**TUT05, Exercise 2** (10 token, $\log 10 \approx 2{,}30$ là entropy tối đa ứng với attention đều):

| $d_k$ | Std của score (không scale) | Entropy (không scale) | Max weight (không scale) | Entropy (có scale) |
|---|---|---|---|---|
| 4 | 2,0 | 1,43 | 0,50 | 1,95 |
| 16 | 4,0 | 0,75 | 0,72 | 1,93 |
| 64 | 8,0 | 0,34 | 0,87 | 1,92 |
| 256 | 15,9 | **0,16** | **0,93** | 1,92 |

Không scale thì attention ngày càng "nhọn". Có scale thì entropy gần như không đổi.

**Entropy** đo độ "phân tán" của phân phối attention: $H(p) = -\sum_i p_i\log p_i$. $H = 0$ khi toàn bộ trọng số dồn vào một token, $H = \log L$ khi trọng số chia đều.

### 4.3 Kiểm tra với PyTorch

`torch.nn.functional.scaled_dot_product_attention(Q, K, V)` nhận tensor có shape `[batch, heads, seq, d_k]`. Trong TUT05, kết quả cài đặt thủ công khớp với PyTorch tới sai số $\sim 10^{-7}$. Hàm này có các backend tối ưu (như FlashAttention) **không lưu lại** toàn bộ ma trận $L\times L$.

---

## 5. Masking

### 5.1 Causal mask (mặt nạ nhân quả)

Một mô hình ngôn ngữ kiểu **decoder** (như GPT) dự đoán token tiếp theo chỉ từ các token **trước** nó. Vì vậy token $i$ **không được** nhìn thấy các vị trí $j > i$:

$$
M = \begin{bmatrix}1&0&0&0\\1&1&0&0\\1&1&1&0\\1&1&1&1\end{bmatrix} \quad \text{(tam giác dưới)}
$$

Các vị trí bị che được gán $-\infty$ **trước** khi qua softmax. Vì $e^{-\infty} = 0$, trọng số của chúng bằng đúng 0, và các trọng số còn lại tự chuẩn hóa để có tổng bằng 1.

$$
\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M_{-\infty}\right)V
$$

**Causal attention weights trong TUT05:**

```text
[[1.000, 0,     0,     0    ],     token 0 chỉ thấy chính nó
 [0.424, 0.576, 0,     0    ],
 [0.248, 0.438, 0.314, 0    ],
 [0.180, 0.255, 0.324, 0.241]]     token cuối thấy toàn bộ chuỗi
```

> Che **sau** softmax (nhân với 0) là sai: các hàng sẽ không còn tổng bằng 1. Phải gán $-\infty$ **trước** softmax.

### 5.2 Phá causal mask: rò rỉ thông tin (TUT05, Exercise 3)

Cho phép token 0 nhìn thấy token 3:
- hàng 0 thay đổi từ $[1, 0, 0, 0]$ thành $[0.456, 0, 0, 0.544]$;
- khi chỉ thay đổi token **tương lai** (token 3), output ở vị trí 0 không đổi nếu mask đúng (sai khác 0,0), nhưng thay đổi tới **4,29** nếu mask bị hỏng.

Đây là **information leakage**. Khi train một mô hình dự đoán token tiếp theo, nếu vị trí $i$ thấy được token $i+1$ (chính là đáp án), model chỉ cần "chép đáp án". Loss khi train thấp một cách giả tạo, nhưng khi **sinh văn bản** thì tương lai chưa tồn tại, nên model hoạt động rất kém.

### 5.3 Padding mask

Khi gộp nhiều câu có độ dài khác nhau vào một batch, các câu ngắn được thêm token `[PAD]`. **Padding mask** che các vị trí `[PAD]` để không token nào chú ý vào chúng. Trong Hugging Face, `attention_mask` có giá trị 1 cho token thật và 0 cho padding.

### 5.4 Cẩn thận: mỗi API có quy ước mask khác nhau

| API | `True` trong mask boolean nghĩa là |
|---|---|
| `F.scaled_dot_product_attention` (`attn_mask`) | vị trí này **được phép** tham gia attention |
| `nn.MultiheadAttention` (`key_padding_mask`, `attn_mask`) | vị trí này **bị che / bỏ qua** |

Phải đọc tài liệu của từng API trước khi dùng lại một mask.

---

## 6. Multi-Head Attention

Một attention matrix chỉ thể hiện **một** kiểu quan hệ giữa các token. **Multi-head** chạy $h$ phép attention **song song**, mỗi head có ma trận chiếu riêng, nên có thể học các kiểu quan hệ khác nhau. Ví dụ, một head chú ý tới token liền trước, một head khác chú ý tới token liền sau.

$$
\text{head}_i = \text{Attention}(XW_Q^{(i)},\ XW_K^{(i)},\ XW_V^{(i)})
$$

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_O
$$

- Thường chọn $d_{head} = d_{model}/h$, nên tổng chi phí tương đương một head có kích thước đầy đủ.
- Số tham số: $W_Q, W_K, W_V, W_O$ mỗi cái $d_{model}\times d_{model}$, cộng bias, tổng là $4d_{model}^2 + 4d_{model}$. Với $d_{model} = 768$ thì có ≈ 2,36M tham số.
- Shape của attention weights khi lấy riêng từng head: `[batch, heads, L_query, L_key]`. TUT05 dùng `nn.MultiheadAttention(embed_dim=8, num_heads=2)`, nên weights có shape `(1, 2, 4, 4)`.

---

## 7. Các tính chất quan trọng

### 7.1 Attention không biết thứ tự

Nếu hoán vị các token đầu vào, output của self-attention cũng bị hoán vị theo đúng cách đó (**permutation equivariant**). Self-attention **không tự biết thứ tự** của các token. Vì vậy Transformer phải cộng thêm **positional encoding** (xem [file 09](09_transformer_hugging_face.md)).

### 7.2 Chi phí

| | Self-attention | RNN |
|---|---|---|
| Chi phí tính toán mỗi lớp | $O(L^2\cdot d)$ | $O(L\cdot d^2)$ |
| Số bước tuần tự | $O(1)$, **song song hoàn toàn** | $O(L)$ |
| Đường đi giữa 2 vị trí bất kỳ | $O(1)$ | $O(L)$ |

Attention nối trực tiếp hai vị trí bất kỳ và song song hóa được, nhưng chi phí tăng theo **$L^2$**. Đây là lý do các mô hình có giới hạn độ dài ngữ cảnh.

### 7.3 Các loại attention

| Loại | Q đến từ | K, V đến từ | Ví dụ |
|---|---|---|---|
| **Self-attention** | chuỗi X | chính X | encoder của BERT |
| **Causal (masked) self-attention** | X | X, nhưng chỉ các vị trí ≤ i | GPT |
| **Cross-attention** | decoder | output của encoder | dịch máy (decoder nhìn câu nguồn) |

Về cách tính score: **dot-product** (Luong, Transformer) dùng $\mathbf{q}^\top\mathbf{k}$; **additive** (Bahdanau) dùng $\mathbf{v}^\top\tanh(W_1\mathbf{q} + W_2\mathbf{k})$.

### 7.4 Attention không phải là lời giải thích

Heatmap attention cho biết **head đó lấy thông tin từ đâu**. Nó **không** chứng minh rằng một token đã **gây ra** dự đoán. Lý do:
- model có nhiều lớp, nhiều head;
- còn có residual connection và FFN;
- giá trị của các vector value cũng ảnh hưởng tới kết quả, không chỉ trọng số.

Đừng kết luận một head "hiểu ngữ pháp" nếu chưa có thí nghiệm kiểm chứng (xem TUT05, Exercise 4, ở [file 09](09_transformer_hugging_face.md)).

---

## 8. Tóm tắt

$$
\boxed{QK^\top \;\rightarrow\; \text{chia }\sqrt{d_k}\text{ / áp mask} \;\rightarrow\; \text{softmax} \;\rightarrow\; \text{tổng có trọng số của }V}
$$

- **Q** hỏi "cần thông tin gì". **K** mô tả "có thể được tìm thấy bằng gì". **V** chứa "thông tin để lấy".
- **Self-attention:** Q, K, V lấy từ cùng một chuỗi.
- **Chia cho $\sqrt{d_k}$** giữ cho softmax không bị bão hòa khi số chiều lớn.
- **Causal mask:** không được nhìn tương lai. Gán $-\infty$ trước softmax.
- **Multi-head:** nhiều kiểu truy xuất thông tin chạy song song.

---

## 9. Câu hỏi ôn tập

<details>
<summary>1. Vì sao context vector thay đổi khi đổi query, dù X không đổi?</summary>

Vì score $\mathbf{q}^\top\mathbf{k}_i$ thay đổi, nên phân phối trọng số thay đổi, và tổng có trọng số của các value cũng thay đổi.
</details>

<details>
<summary>2. Điều gì xảy ra nếu bỏ hệ số $1/\sqrt{d_k}$ khi $d_k$ lớn?</summary>

Score có variance $d_k$, nên softmax gần như one-hot. Attention dồn vào một token, và gradient gần bằng 0 do softmax bão hòa.
</details>

<details>
<summary>3. Vì sao gán $-\infty$ trước softmax chứ không nhân 0 sau softmax?</summary>

Gán $-\infty$ trước softmax cho trọng số bằng đúng 0, và các trọng số còn lại vẫn có tổng bằng 1. Nhân 0 sau softmax làm hàng không còn tổng bằng 1.
</details>

<details>
<summary>4. Self-attention có biết thứ tự token không?</summary>

Không. Self-attention là permutation-equivariant, nên cần positional encoding để đưa thông tin về vị trí vào.
</details>

<details>
<summary>5. Attention matrix với L = 512 token có bao nhiêu phần tử mỗi head? Chi phí tăng thế nào khi L tăng gấp đôi?</summary>

$512^2 = 262.144$ phần tử. Khi $L$ tăng gấp đôi, chi phí tăng khoảng 4 lần, vì tỉ lệ với $L^2$.
</details>

<details>
<summary>6. Self-attention, causal attention và cross-attention khác nhau thế nào?</summary>

- Self-attention: Q, K, V lấy từ cùng một chuỗi.
- Causal attention: giống self-attention nhưng chỉ nhìn các vị trí phía trước.
- Cross-attention: Q lấy từ decoder, còn K, V lấy từ encoder.
</details>

---

**Trước:** [07 — Dự báo chuỗi thời gian](07_du_bao_chuoi_thoi_gian.md) · **Tiếp theo:** [09 — Transformer & Hugging Face](09_transformer_hugging_face.md)
