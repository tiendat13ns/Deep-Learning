# 09 — Transformer, BERT/GPT và Hugging Face

> **Lab tương ứng:** TUT05, Part II (DistilBERT phân loại cảm xúc, DistilGPT2 sinh văn bản) và Exercise 4–5
> **Ý tưởng chính:** Transformer xếp chồng nhiều **khối Transformer**. Mỗi khối gồm multi-head attention, mạng feed-forward, residual connection và LayerNorm. Từ kiến trúc này sinh ra hai họ mô hình chính: **encoder (BERT)** và **decoder (GPT)**.

Đọc [file 08 — Attention](08_attention.md) trước file này.

---

## 1. Từ attention tới Transformer

TUT05 kết thúc bằng câu: *"bước tiếp theo là đặt multi-head attention vào một **khối Transformer** đầy đủ, với LayerNorm, residual connection và mạng feed-forward"*. Mục này giải thích khối đó.

### 1.1 Đầu vào: embedding và vị trí

1. **Tokenization:** văn bản được tách thành các token, rồi chuyển thành `input_ids` (xem mục 4).
2. **Token embedding:** một bảng tra cứu $E\in\mathbb{R}^{V\times d_{model}}$ (với $V$ là kích thước từ vựng) biến mỗi id thành một vector.
3. **Positional encoding:** self-attention không biết thứ tự token (xem file 08, mục 7.1), nên phải cộng thêm thông tin vị trí:

$$
\mathbf{x}_i = \text{TokenEmb}(t_i) + \text{PosEmb}(i)
$$

- **Sinusoidal** (bài báo Transformer gốc), cố định, không học:

$$
PE_{(pos,2k)} = \sin\!\left(\frac{pos}{10000^{2k/d}}\right), \quad PE_{(pos,2k+1)} = \cos\!\left(\frac{pos}{10000^{2k/d}}\right)
$$

- **Learned:** BERT, DistilBERT và GPT-2 học một vector riêng cho mỗi vị trí, nên có độ dài tối đa (512 với BERT, 1024 với GPT-2).
- Các mô hình mới thường dùng RoPE hoặc ALiBi (mã hóa vị trí tương đối).

### 1.2 Khối Transformer

```text
x ──► Multi-Head Self-Attention ──► (+) ──► LayerNorm ──► Feed-Forward ──► (+) ──► LayerNorm ──► ra
│                                    ▲                │                    ▲
└────────── residual ────────────────┘                └───── residual ─────┘
```

$$
\begin{aligned}
\mathbf{h} &= \text{LayerNorm}\big(\mathbf{x} + \text{MHA}(\mathbf{x})\big)\\
\mathbf{y} &= \text{LayerNorm}\big(\mathbf{h} + \text{FFN}(\mathbf{h})\big)
\end{aligned}
$$

Đây là kiểu **post-LN**, dùng trong Transformer gốc và BERT. GPT-2 và hầu hết mô hình mới dùng **pre-LN**: $\mathbf{x} + \text{MHA}(\text{LN}(\mathbf{x}))$, train ổn định hơn khi mạng sâu.

| Thành phần | Vai trò |
|---|---|
| **Multi-head attention** | **trộn thông tin giữa các token** (giao tiếp giữa các vị trí) |
| **FFN (feed-forward)** | MLP 2 lớp $\text{FFN}(\mathbf{h}) = W_2\,\text{GELU}(W_1\mathbf{h} + b_1) + b_2$, thường có $d_{ff} = 4d_{model}$. Áp **riêng cho từng token** (xử lý nội dung tại mỗi vị trí). |
| **Residual** $x + F(x)$ | gradient đi thẳng qua đường tắt, nên train được mạng sâu (cùng ý tưởng với ResNet và cell state của LSTM) |
| **LayerNorm** | chuẩn hóa **theo chiều feature của từng token**, không theo batch như BatchNorm. Không phụ thuộc batch size hay độ dài chuỗi, nên phù hợp với dữ liệu chuỗi. |

**Số tham số mỗi khối** ≈ $4d^2$ (attention) $+ 8d^2$ (FFN với $d_{ff} = 4d$) $= 12d^2$. Với $d = 768$ là ≈ 7,1M tham số mỗi khối. BERT-base có 12 khối, cộng embedding, tổng ≈ 110M tham số.

### 1.3 Đầu ra

- **Encoder:** cho ra một vector ngữ cảnh cho **mỗi** token, rồi gắn head phù hợp với bài toán (phân loại, gán nhãn, ...).
- **Decoder (mô hình ngôn ngữ):** `Linear(d_model → V)` cho ra logit trên toàn bộ từ vựng, rồi softmax để có phân phối của token tiếp theo.

---

## 2. Ba họ mô hình Transformer

| Họ | Attention | Mục tiêu pretrain | Thế mạnh | Ví dụ |
|---|---|---|---|---|
| **Encoder-only** | **hai chiều** (thấy cả trái lẫn phải) | **Masked LM**: che ≈ 15% token và đoán lại | **hiểu** văn bản: phân loại, NER, tìm kiếm | BERT, **DistilBERT**, RoBERTa |
| **Decoder-only** | **causal** (chỉ thấy bên trái) | **Causal LM**: đoán token tiếp theo | **sinh** văn bản, chatbot | GPT-2, **DistilGPT2**, GPT-3/4, LLaMA |
| **Encoder–decoder** | encoder hai chiều + decoder causal + **cross-attention** | khôi phục hoặc biến đổi văn bản | dịch máy, tóm tắt | Transformer gốc, T5, BART |

### Mục tiêu pretrain

- **Masked Language Modeling (MLM):** `"The movie was [MASK] good"` → đoán "not". Mô hình dùng ngữ cảnh **hai phía**. BERT còn có thêm mục tiêu Next Sentence Prediction (đoán hai câu có liền nhau không). RoBERTa và DistilBERT bỏ mục tiêu này.
- **Causal Language Modeling (CLM):**

$$
P(x_1,\dots,x_T) = \prod_t P(x_t\mid x_{<t})
$$

  Bắt buộc phải dùng causal mask, nếu không sẽ rò rỉ đáp án (xem file 08, mục 5.2).

**Pretrain rồi fine-tune:** pretrain không cần nhãn, trên lượng văn bản khổng lồ; sau đó fine-tune với ít dữ liệu có nhãn cho từng bài toán. Đây chính là transfer learning ([file 04](04_transfer_learning.md)) áp dụng cho NLP. Ví dụ, DistilBERT trong lab được fine-tune trên **SST-2** (Stanford Sentiment Treebank, phân loại câu thành tích cực/tiêu cực).

---

## 3. Knowledge distillation: chữ "Distil" trong DistilBERT

**Knowledge distillation** (Hinton và cộng sự, 2015): train một mô hình **học trò (student)** nhỏ để bắt chước **phân phối đầu ra** của một mô hình **thầy (teacher)** lớn, chứ không chỉ học từ nhãn cứng.

$$
\mathcal{L} = \alpha\,\mathcal{L}_{\text{CE}}(y, p_s) + (1-\alpha)\,T^2\,\text{KL}\big(p_t^{(T)}\,\|\,p_s^{(T)}\big), \qquad p^{(T)} = \text{softmax}(z/T)
$$

Nhiệt độ $T > 1$ làm phân phối "mềm" hơn, để lộ ra kiến thức ẩn của teacher, ví dụ "ảnh này giống mèo 70% nhưng cũng hơi giống chó".

| | BERT-base | **DistilBERT** | GPT-2 small | **DistilGPT2** |
|---|---|---|---|---|
| Số lớp | 12 | **6** | 12 | **6** |
| Hidden / số head | 768 / 12 | 768 / 12 | 768 / 12 | 768 / 12 |
| Tham số | ≈ 110M | ≈ 66M | ≈ 124M | ≈ 82M |

DistilBERT nhỏ hơn khoảng 40%, nhanh hơn khoảng 60%, và giữ được khoảng 97% khả năng hiểu ngôn ngữ của BERT (theo Sanh và cộng sự, 2019).

---

## 4. Tokenization

Mạng chỉ xử lý số. **Tokenizer** chuyển văn bản thành dãy id. Các mô hình hiện đại dùng **subword tokenization**: từ phổ biến giữ nguyên, từ hiếm được tách thành các mảnh nhỏ. Cách này tránh được từ ngoài từ vựng, mà từ vựng vẫn có kích thước vừa phải.

| | WordPiece (BERT, DistilBERT) | Byte-level BPE (GPT-2) |
|---|---|---|
| Từ vựng | ≈ 30.522 (bản *uncased*: chuyển hết thành chữ thường) | 50.257 |
| Mảnh tiếp nối | tiền tố `##`, ví dụ `play ##ing` | dấu cách được mã hóa vào token, ví dụ `Ġthe` |
| Token đặc biệt | `[CLS]` (đầu câu), `[SEP]` (cuối câu), `[PAD]`, `[MASK]`, `[UNK]` | `<\|endoftext\|>` |

Ví dụ trong TUT05:

```text
"The lecture was clear and surprisingly useful."
→ ['[CLS]', 'the', 'lecture', 'was', 'clear', 'and', 'surprisingly', 'useful', '.', '[SEP]']
```

Kết quả của tokenizer:
- `input_ids`: id của từng token;
- `attention_mask`: 1 cho token thật, 0 cho padding (xem file 08, mục 5.3).

**Vector của `[CLS]`** ở lớp cuối được dùng làm biểu diễn của **cả câu** để phân loại: `[CLS]` → pre-classifier → classifier → 2 logit (NEGATIVE, POSITIVE).

---

## 5. Sinh văn bản với mô hình decoder

### 5.1 Vòng lặp autoregressive

```text
prompt → model → phân phối token tiếp theo → chọn 1 token → nối vào chuỗi → lặp lại
```

Mỗi bước model chỉ dùng tiền tố $x_1,\dots,x_t$, vì attention của nó là causal. **KV cache** lưu lại K và V của các token đã xử lý, để không phải tính lại chúng ở mỗi bước mới.

### 5.2 Các chiến lược giải mã (decoding)

| Chiến lược | Cách làm | Đặc điểm |
|---|---|---|
| **Greedy** (`do_sample=False`, như TUT05) | luôn chọn token có xác suất cao nhất | tất định; dễ **lặp lại** hoặc đơn điệu |
| **Beam search** | giữ $k$ chuỗi tốt nhất ở mỗi bước | tốt cho dịch máy; vẫn dễ lặp với văn bản tự do |
| **Sampling + temperature** | lấy mẫu từ $\text{softmax}(z/T)$ | $T < 1$: an toàn hơn; $T > 1$: đa dạng hơn nhưng dễ lạc đề |
| **Top-k** | chỉ lấy mẫu trong $k$ token có xác suất cao nhất | cắt bỏ phần đuôi ít khả năng |
| **Top-p (nucleus)** | lấy mẫu trong tập token nhỏ nhất có tổng xác suất ≥ $p$ | kích thước tập ứng viên tự thích nghi |

**Quan sát trong TUT05:** DistilGPT2 với greedy decoding sinh ra:

> *"Attention in neural networks is useful because it allows us to see the neural networks that are involved in the processing of information."*

Sau câu đó là khoảng 20 dòng trống. Hiện tượng này gọi là **degeneration**: greedy decoding bị kẹt trong một chuỗi có xác suất cao nhưng lặp lại. Có thể khắc phục bằng `repetition_penalty`, `no_repeat_ngram_size`, hoặc dùng sampling với top-p.

---

## 6. Hugging Face

### 6.1 Quy trình

1. Chọn **model ID** trên Hugging Face Hub, ví dụ `distilbert/distilbert-base-uncased-finetuned-sst-2-english`.
2. Gọi `from_pretrained(...)` hoặc `pipeline(...)`. Config, tokenizer và trọng số được **tải về rồi cache lại** (trong `~/.cache/huggingface`), nên lần sau không phải tải lại.
3. Chạy inference ngay trên máy (Colab hoặc laptop).

### 6.2 Hai mức API

**Mức cao: `pipeline`**

```python
classifier = pipeline("text-classification", model=CLASSIFIER_ID, device=-1)  # -1 = CPU, 0 = GPU đầu tiên
classifier(["The lecture was clear and surprisingly useful.",
            "The explanation was confusing and I learned almost nothing."])
# → POSITIVE 0.9997 ; NEGATIVE 0.9992
```

**Mức thấp: `AutoTokenizer` và `AutoModel...`**

```python
tokenizer = AutoTokenizer.from_pretrained(ID)
model = AutoModelForSequenceClassification.from_pretrained(ID, attn_implementation="eager")
batch = tokenizer(sentence, return_tensors="pt")
with torch.no_grad():
    out = model(**batch, output_attentions=True)
probs = out.logits.softmax(-1)
label = model.config.id2label[int(probs.argmax())]
```

**Những gì `pipeline` làm thay ta:**

```text
text → tokenizer → input_ids + attention_mask → Transformer → logits → softmax → argmax → id2label
```

**TUT05, Exercise 5:** gọi model trực tiếp cho logits $[-3{,}88;\ 4{,}10]$, softmax ra $[0{,}000343;\ 0{,}999657]$, tức POSITIVE với score 0,999657. Kết quả này **trùng khớp** với `pipeline`.

### 6.3 Lấy ma trận attention ra để xem

- `output_attentions=True` trả về một tuple, **mỗi lớp một tensor**, mỗi tensor có shape `[batch, heads, seq, seq]`. DistilBERT có 6 lớp × 12 head. Câu trong lab có 14 token, nên mỗi tensor có shape `(1, 12, 14, 14)`.
- Phải dùng `attn_implementation="eager"`. Các backend tối ưu như SDPA hay FlashAttention **không tạo ra** toàn bộ ma trận attention để tiết kiệm bộ nhớ, nên không trả về được trọng số.
- Mỗi hàng của ma trận attention có tổng bằng 1 (sai số khoảng $10^{-7}$).

### 6.4 Phân tích các head (TUT05, Exercise 4)

Câu: *"The movie was not good, but the ending was excellent."*

**Layer 0:**

| Head | Entropy | Đặc điểm |
|---|---|---|
| 0 | 2,49 | gần như **đều** (tối đa là $\log 14 \approx 2{,}64$) |
| 2 | 0,57 | 0,70 trọng số dồn vào **token đứng trước** |
| 10 | 0,70 | 0,67 trọng số dồn vào **token đứng sau** |
| 9 | 1,14 | 0,70 trọng số dồn vào `[CLS]` |

**Các lớp giữa (2–4):** nhiều head dồn 50–88% trọng số vào `[SEP]`. Các nghiên cứu (Clark và cộng sự, 2019, *"What Does BERT Look At?"*) cho rằng đây thường là hành vi **"no-op"**: khi head không có gì hữu ích để lấy, nó "đỗ" attention vào một token đặc biệt.

Nhận xét:
- Các head ở lớp đầu thường có **mẫu hình theo vị trí** (token trước, token sau), không phụ thuộc vào nghĩa.
- Mỗi head học một kiểu quan hệ khác nhau. Đây chính là lý do dùng multi-head.
- Chỉ quan sát trên một câu thì **không thể** kết luận một head "hiểu" một khái niệm ngữ pháp. Attention không phải là lời giải thích nhân quả (xem file 08, mục 7.4).

---

## 7. So sánh với các kiến trúc trước

| | CNN | RNN/LSTM | Transformer |
|---|---|---|---|
| Inductive bias | cục bộ, chia sẻ trọng số theo không gian | tuần tự, chia sẻ trọng số theo thời gian | rất ít: mọi token nhìn mọi token |
| Song song hóa | cao | thấp (tuần tự) | **cao** |
| Phụ thuộc xa | cần mạng sâu | khó (vanishing gradient) | trực tiếp, đường đi $O(1)$ |
| Chi phí theo độ dài chuỗi | tuyến tính | tuyến tính | **bậc hai** ($L^2$) |
| Nhu cầu dữ liệu | vừa | vừa | **lớn**, vì ít inductive bias, nên thường cần pretrain |

Transformer cũng được dùng cho ảnh: **Vision Transformer (ViT)** chia ảnh thành các patch $16\times16$ và xử lý chúng như các token.

---

## 8. Câu hỏi ôn tập

<details>
<summary>1. BERT và GPT khác nhau thế nào về attention và mục tiêu pretrain?</summary>

BERT là encoder, dùng attention hai chiều, pretrain bằng Masked LM, mạnh về hiểu văn bản. GPT là decoder, dùng causal attention, pretrain bằng Causal LM, mạnh về sinh văn bản.
</details>

<details>
<summary>2. Residual connection và LayerNorm giúp gì?</summary>

Residual tạo đường đi thẳng cho gradient, nên train được mạng sâu. LayerNorm ổn định phân phối activation của từng token, và không phụ thuộc batch size.
</details>

<details>
<summary>3. Vì sao cần positional encoding?</summary>

Vì self-attention không biết thứ tự token (permutation-equivariant). Nếu không có thông tin vị trí, "chó cắn người" và "người cắn chó" sẽ có cùng tập biểu diễn.
</details>

<details>
<summary>4. Vì sao phải dùng <code>attn_implementation="eager"</code> khi muốn xem attention?</summary>

Vì các backend tối ưu (SDPA, FlashAttention) không tạo ra toàn bộ ma trận attention, nên không trả về được trọng số.
</details>

<details>
<summary>5. <code>pipeline</code> và gọi model trực tiếp có cho cùng kết quả không? <code>pipeline</code> làm những bước nào?</summary>

Có, cùng kết quả. `pipeline` thực hiện: tokenizer → model → logits → softmax → argmax → ánh xạ sang tên nhãn bằng `id2label`.
</details>

<details>
<summary>6. Vì sao greedy decoding dễ sinh ra văn bản lặp lại?</summary>

Vì greedy luôn chọn token có xác suất cao nhất, nên dễ rơi vào vòng lặp có xác suất cao. Sampling với top-p/top-k, hoặc dùng repetition penalty, giúp giảm hiện tượng này.
</details>

<details>
<summary>7. Knowledge distillation là gì?</summary>

Train một mô hình nhỏ (student) bắt chước phân phối xác suất "mềm" của một mô hình lớn (teacher), để được mô hình nhỏ và nhanh mà vẫn giữ phần lớn chất lượng.
</details>

---

**Trước:** [08 — Attention](08_attention.md) · **Về mục lục:** [README](README.md)
