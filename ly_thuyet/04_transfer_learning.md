# 04 — Transfer Learning và Fine-tuning

> **Lab tương ứng:** TUT03, Part 4 (fine-tune MobileNetV3-Small trên 4 lớp của CIFAR-10) và Exercise 3, 5
> **Ý tưởng chính:** Thay vì học từ đầu, ta **tái sử dụng** các đặc trưng đã được học trên một tập dữ liệu lớn, rồi điều chỉnh chúng cho bài toán mới.

---

## 1. Vì sao cần transfer learning?

Train một CNN mạnh từ trọng số ngẫu nhiên cần:
- **rất nhiều dữ liệu có nhãn** (ImageNet-1K có ≈ 1,28 triệu ảnh thuộc 1000 lớp);
- **rất nhiều tài nguyên tính toán** (nhiều ngày chạy GPU).

Lab TUT03 chỉ có **2.000 ảnh train** (500 ảnh/lớp × 4 lớp). Train từ đầu với lượng dữ liệu này gần như chắc chắn sẽ overfit.

**Quan sát then chốt:** các lớp đầu của CNN học những đặc trưng **tổng quát**, như cạnh, góc, màu, texture. Những đặc trưng này hữu ích cho hầu hết bài toán ảnh, không riêng gì ImageNet. Chỉ các lớp cuối mới chuyên biệt cho từng tập dữ liệu.

| Thuật ngữ | Nghĩa |
|---|---|
| **Source task / dataset** | bài toán và dữ liệu gốc dùng để pretrain (ImageNet-1K, 1000 lớp) |
| **Target task** | bài toán mới (4 lớp: airplane, automobile, cat, dog) |
| **Pretrained weights** | trọng số đã học trên source |
| **Backbone / feature extractor** | phần conv của mạng, tạo ra đặc trưng |
| **Head / classifier** | các lớp cuối, ánh xạ đặc trưng thành logit của từng lớp |

---

## 2. Hai chiến lược

### 2.1 Feature extraction (đóng băng backbone)

- **Đóng băng (freeze)** toàn bộ backbone bằng `requires_grad = False`.
- Thay head bằng một lớp mới có số đầu ra bằng số lớp của bài toán mới, rồi **chỉ train head**.
- Cách này nhanh, ít tham số cần học, và khó overfit.

### 2.2 Fine-tuning (mở băng một phần)

- Mở băng một số block **cuối** của backbone và train tiếp với **learning rate nhỏ**.
- Nhờ đó model điều chỉnh được các đặc trưng cấp cao cho dữ liệu mới.

### 2.3 Chọn chiến lược theo dữ liệu

| | Target **giống** source | Target **khác** source |
|---|---|---|
| **Ít dữ liệu** | feature extraction (chỉ train head) | train head, cẩn thận khi mở băng; có thể dùng đặc trưng ở các lớp giữa |
| **Nhiều dữ liệu** | fine-tune nhiều lớp | fine-tune sâu, hoặc train lại gần như toàn bộ |

---

## 3. Quy trình hai pha trong TUT03

### Bước 0 — Chuẩn bị model

```python
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)  # pretrained ImageNet
for p in model.parameters():
    p.requires_grad = False                                   # đóng băng tất cả
model.classifier[-1] = nn.Linear(1024, 4)                     # head mới: 1000 lớp → 4 lớp
```

Lớp Linear mới được tạo sau khi đóng băng, nên mặc định có `requires_grad = True`.

| | Số tham số |
|---|---|
| Toàn bộ model (với head 4 lớp) | 1.521.956 |
| **Trainable ở Phase A** (chỉ head mới: $1024\times4 + 4$) | **4.100** (0,27%) |
| Trainable ở Phase B (3 block cuối + toàn bộ classifier) | 1.239.588 |
| Trainable khi mở 6 block (Exercise 3) | 1.383.204 |

Phần lớn số tham số trainable ở Phase B nằm trong **classifier block** (`Linear(576→1024)` có ≈ 590k tham số), chứ không phải trong các block conv.

### Phase A — Chỉ train head

- Optimizer: `AdamW(lr=3e-3, weight_decay=1e-4)`, chỉ trên các tham số trainable.
- Chạy 1 epoch.
- **Lý do:** head mới được khởi tạo **ngẫu nhiên**. Nếu mở băng backbone ngay lúc này, gradient lớn và nhiễu từ head ngẫu nhiên sẽ truyền ngược vào backbone và **phá hỏng** các đặc trưng pretrained. Hiện tượng này gọi là **catastrophic forgetting**.

### Phase B — Mở băng các block cuối

- Mở băng `model.features[-3:]` (3 block conv cuối) và toàn bộ `model.classifier`.
- Dùng **learning rate khác nhau cho từng nhóm tham số** (discriminative learning rates):

```python
AdamW([
    {"params": backbone_params, "lr": 1e-4},   # nhỏ: chỉ chỉnh nhẹ đặc trưng đã tốt
    {"params": head_params,     "lr": 5e-4},   # lớn hơn: phần đặc thù cho bài toán
], weight_decay=1e-4)
```

- Chạy 2 epoch.

**Vì sao mở các block CUỐI?** Các block đầu học đặc trưng tổng quát nên giữ nguyên. Các block cuối học đặc trưng đặc thù của ImageNet, nên cần được điều chỉnh nhiều nhất.

### Chọn checkpoint

Sau mỗi epoch, đánh giá **validation macro-F1** và giữ lại trọng số (`state_dict`) tốt nhất. Test set chỉ được dùng **một lần** ở cuối.

---

## 4. Chi tiết quan trọng: BatchNorm khi đóng băng

`model.train()` chuyển **mọi** lớp BatchNorm sang chế độ train, kể cả khi trọng số của chúng đã bị đóng băng. Trong chế độ đó, BatchNorm vẫn **cập nhật running mean/var** theo dữ liệu mới. Như vậy, "đóng băng" bằng `requires_grad = False` **không** đủ để giữ nguyên BatchNorm.

Với dữ liệu nhỏ và batch nhỏ, thống kê của batch rất nhiễu, nên việc này có thể làm hỏng backbone. TUT03 xử lý bằng một hàm phụ:

```python
def keep_frozen_batchnorm_in_eval(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            params = list(m.parameters())
            if params and not any(p.requires_grad for p in params):
                m.eval()     # BN trong block đóng băng: dùng running stats cũ, không cập nhật
```

Hàm này được gọi ngay sau `model.train()` trong mỗi epoch train. Khi một block được mở băng, BN của nó lại hoạt động bình thường.

---

## 5. Tiền xử lý dữ liệu

### 5.1 Khớp với điều kiện lúc pretrain

- **Kích thước ảnh:** ảnh CIFAR-10 có kích thước $32\times32$, còn ImageNet thường là $224\times224$. Lab resize lên **$128\times128$** để cân bằng giữa độ chính xác và chi phí tính toán.
- **Chuẩn hóa ImageNet:** phải dùng **cùng mean/std** như lúc pretrain, vì các trọng số được học trên input đã chuẩn hóa theo cách đó:

$$
\mu = [0.485,\ 0.456,\ 0.406], \qquad \sigma = [0.229,\ 0.224,\ 0.225] \quad \text{(theo kênh R, G, B)}
$$

### 5.2 Data augmentation

Data augmentation chỉ áp dụng cho **tập train**:
- `RandomHorizontalFlip(p=0.5)`: lật ngang ngẫu nhiên. Máy bay hay con mèo lật ngang thì vẫn là máy bay, con mèo.
- `RandomCrop(128, padding=8, padding_mode="reflect")`: dịch ảnh ngẫu nhiên vài pixel.

Validation và test **không** augmentation, để kết quả đánh giá ổn định và tái lập được.

Augmentation tạo ra các biến thể **giữ nguyên nhãn**, giúp model bất biến với những thay đổi không quan trọng, và đóng vai trò regularization.

### 5.3 Chia dữ liệu cân bằng

Mỗi lớp lấy 500 ảnh train, 100 ảnh validation và 200 ảnh test. Train và validation được lấy từ tập train gốc của CIFAR và **không trùng nhau** (overlap = 0). Test lấy từ tập test gốc. Nhãn gốc (0, 1, 3, 5) được ánh xạ lại thành (0, 1, 2, 3).

---

## 6. Kết quả và bài học từ lab

### Kết quả chính (GPU T4)

| Giai đoạn | Val macro-F1 |
|---|---|
| Phase A (chỉ head), epoch 1 | **0,858** |
| Phase B, epoch 1 | 0,840 |
| Phase B, epoch 2 | 0,834 |

Test (dùng checkpoint tốt nhất): **accuracy 0,855, macro-F1 0,855**.

Hai nhận xét:
- **Chỉ train head (4.100 tham số) đã đạt 85%**. Đây là sức mạnh của đặc trưng pretrained.
- Ở lần chạy này, fine-tune **không cải thiện thêm**: train loss giảm (0,50 → 0,24) nhưng val loss tăng (0,38 → 0,42). Đó là dấu hiệu bắt đầu overfit với 2.000 ảnh.

### Exercise 3 — Mở 3 block so với 6 block

| | Val F1 sau fine-tune epoch 1 | Epoch 2 | Val loss |
|---|---|---|---|
| Mở 3 block | 0,826 | 0,824 | 0,42 / 0,41 |
| Mở 6 block | 0,775 | 0,805 | 0,52 / 0,51 |

Mở **nhiều** block hơn cho kết quả **tệ hơn**: có thêm ≈ 144k tham số được train nhưng dữ liệu vẫn chỉ có 2.000 ảnh, nên overfit nặng hơn. Giả thuyết ban đầu "mở thêm block sẽ tốt hơn" đã bị bác bỏ. Bài học rút ra:
- **Mở băng nhiều hơn không phải lúc nào cũng tốt hơn**, nhất là khi ít dữ liệu.
- Phải so sánh trên **validation**, chỉ thay đổi **một biến** mỗi lần, và viết giả thuyết **trước khi chạy**.
- Nếu checkpoint tốt nhất của cả hai cấu hình đều là sau Phase A, thì con số "best" không phân biệt được hai cấu hình. Phải nhìn vào từng epoch.

### Exercise 5 — Đổi độ phân giải ảnh khi inference

Model fine-tune ở 128 px, nhưng nhờ **Global Average Pooling** vẫn nhận được ảnh 96 px hoặc 160 px:

| IMAGE_SIZE | Test accuracy |
|---|---|
| 96 | 0,838 |
| 128 | 0,855 |
| 160 | 0,896 |

Ảnh lớn hơn cho accuracy cao hơn. Lý do có thể là feature map lớn hơn (5×5 thay vì 4×4 ở lớp cuối) gần hơn với tỉ lệ ảnh 224 px mà backbone đã được pretrain. Muốn **chọn** độ phân giải thì phải so sánh trên validation, rồi train lại ở kích thước đã chọn. Về chi phí tính toán, xem [file 05](05_danh_gia_mo_hinh.md).

---

## 7. Mẹo thực hành khi fine-tune

1. Luôn **train head trước**, rồi mới mở băng backbone.
2. Learning rate của backbone nên **nhỏ hơn 5–10 lần** so với head.
3. Mở băng **từ cuối lên**, mỗi lần thêm vài block.
4. Giữ BatchNorm của phần đóng băng ở chế độ eval, và cẩn thận khi batch size nhỏ.
5. Dùng **đúng tiền xử lý** của model pretrained (mean/std, kích thước ảnh).
6. Theo dõi val loss và val metric; dùng early stopping hoặc chọn checkpoint tốt nhất.
7. Weight decay và augmentation giúp giảm overfit.

---

## 8. Transfer learning ngoài thị giác máy tính

Ý tưởng này là nền tảng của NLP hiện đại: **pretrain** một Transformer trên lượng văn bản khổng lồ, sau đó **fine-tune** cho từng bài toán cụ thể. Ví dụ, DistilBERT trong TUT05 được fine-tune cho bài toán phân loại cảm xúc SST-2. Xem [file 09](09_transformer_hugging_face.md).

---

## 9. Câu hỏi ôn tập

<details>
<summary>1. Vì sao đóng băng phần lớn MobileNetV3 trước khi fine-tune?</summary>

Vì dữ liệu ít. Backbone đã có đặc trưng tổng quát tốt. Đóng băng nó giúp tránh phá hỏng các đặc trưng đó bằng gradient nhiễu từ head còn ngẫu nhiên, giảm overfit, và train nhanh hơn, tốn ít bộ nhớ hơn.
</details>

<details>
<summary>2. Vì sao backbone dùng learning rate nhỏ hơn head?</summary>

Trọng số backbone đã gần tối ưu nên chỉ cần chỉnh nhẹ. Trọng số head còn mới và đặc thù cho bài toán nên cần học nhiều hơn.
</details>

<details>
<summary>3. Đặt <code>requires_grad=False</code> có đủ để BatchNorm không thay đổi không?</summary>

Không. Ở chế độ `train()`, BatchNorm vẫn cập nhật running mean/var. Phải đặt riêng các lớp BN đó về `eval()`.
</details>

<details>
<summary>4. Vì sao phải dùng mean/std của ImageNet chứ không dùng mean/std của CIFAR?</summary>

Vì trọng số pretrained được học trên input đã chuẩn hóa theo ImageNet. Giữ cùng phân phối input thì các đặc trưng đã học mới phát huy đúng tác dụng.
</details>

<details>
<summary>5. Fine-tune làm train loss giảm nhưng val loss tăng. Nên làm gì?</summary>

Đây là dấu hiệu overfit. Có thể giảm số block được mở băng hoặc giảm learning rate của backbone, tăng augmentation hoặc weight decay, dùng early stopping, và giữ checkpoint tốt nhất trên validation.
</details>

---

**Trước:** [03 — CNN](03_cnn.md) · **Tiếp theo:** [05 — Đánh giá mô hình](05_danh_gia_mo_hinh.md)
