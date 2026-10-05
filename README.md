# ATI — Lab Deep Learning

Các bài lab (tutorial) của môn **ATI**, học về Deep Learning bằng PyTorch: từ mạng neural cơ bản tới CNN, RNN/LSTM/GRU, Attention và Transformer.

## Nội dung

| Lab | Chủ đề | Nội dung chính |
|---|---|---|
| [TUT01](62FIT4ATI_CLC02_2301140022_TUT01.ipynb) | Forward pass của mạng neural | Neuron, lớp Linear, ReLU/Sigmoid, theo dõi giá trị qua từng lớp |
| [TUT02](62FIT4ATI_CLC02_2301140022_TUT02.ipynb) | Train mạng neural đầu tiên | MLP phân loại PASS/FAIL, loss, backprop, Adam, đánh giá trên tập test |
| [TUT03](62FIT4ATI_CLC02_2301140022_TUT03.ipynb) | CNN | Convolution, pooling, fine-tune MobileNetV3 trên CIFAR-10, F1, FLOPs, latency |
| [TUT04](62FIT4ATI_CLC02_2301140022_TUT04.ipynb) | RNN, LSTM, GRU | Dự báo nồng độ CO₂ hằng tuần, so sánh ba kiến trúc với baseline |
| [TUT05](62FIT4ATI_CLC02_2301140022_TUT05.ipynb) | Attention & Transformer | Self-attention, causal mask, multi-head, DistilBERT và DistilGPT2 trên Hugging Face |

Lý thuyết chi tiết của từng chủ đề nằm trong thư mục [`ly_thuyet/`](ly_thuyet/README.md).

## Chạy notebook

- Môi trường: Python 3.10 trở lên, cùng `torch`, `torchvision`, `transformers`, `scikit-learn`, `pandas`, `matplotlib`, `statsmodels`.
- Có thể upload thẳng lên **Google Colab** (*File → Upload notebook*). TUT03 nên chọn runtime GPU (T4); các lab còn lại chạy được bằng CPU.
- Dữ liệu (CIFAR-10, CO₂) và các model pretrained sẽ được tự động tải về khi chạy lần đầu.
