# Presentation script / Kịch bản thuyết trình (EN | VI)

ViGoEmotions midterm · Group 3 · Nguyễn Quốc Khanh · Đinh Tiến Đạt · Nguyễn Duy Anh

Planned time: about 8:25 (English about 1106 words, about 7.9 min at a calm 140 words per minute). Thời gian dự kiến: khoảng 8 đến 9 phút. Nói chậm, dừng một nhịp ở mỗi câu hỏi Q1–Q5.

| Slide | Time |
|---|---|
| 1. Title | 0:00 – 0:20 |
| 2. The data | 0:20 – 1:20 |
| 3. Pipeline | 1:20 – 2:20 |
| 4. Models | 2:20 – 3:10 |
| 5. Questions 1 and 2 | 3:10 – 4:10 |
| 6. Question 3 | 4:10 – 5:10 |
| 7. Questions 4 and 5 | 5:10 – 6:40 |
| 8. Error analysis | 6:40 – 7:25 |
| 9. Proposed research question for the Final | 7:25 – 8:15 |
| 10. Thank you | 8:15 – 8:25 |

## Slide 1. Title (~20 s)

**English**

Hello everyone. We are Group 3. Our project is about recognizing emotions in Vietnamese social-media comments. One comment can have more than one emotion. For example, it can be sad and angry at the same time. We use the ViGoEmotions dataset.

**Tiếng Việt**

Xin chào mọi người. Chúng em là nhóm 3. Đề tài của nhóm là nhận diện cảm xúc trong bình luận mạng xã hội tiếng Việt. Một bình luận có thể mang nhiều cảm xúc cùng lúc, ví dụ vừa buồn vừa tức giận. Nhóm dùng bộ dữ liệu ViGoEmotions.

## Slide 2. The data (~60 s)

**English**

Let us start with the data. There are 20,664 comments and 28 labels: 27 emotions and one neutral label. On average, a comment has about two labels.
Three things are interesting. First, the labels are not balanced, but not extreme: the most common label, amusement, is 4.5 times more frequent than the rarest one, relief. Second, emotions come in groups. Sadness often comes with grief, and anger with annoyance. Third, this is social-media text. It is short, a quarter of the comments have emoji, and one in six uses teencode, which is slang shorthand.
What did we do? We used the official split of the dataset, the same as in the paper: about 16,500 comments for training and about 2,000 each for validation and test. This split is not stratified. We also checked data quality. We found 306 comments tagged neutral together with another emotion, which may be noise. We report it but do not change it. And we never store comment text, because of the dataset licence.

**Tiếng Việt**

Chúng ta bắt đầu với dữ liệu. Có 20.664 bình luận và 28 nhãn: 27 cảm xúc và một nhãn neutral (trung tính). Trung bình mỗi bình luận có khoảng hai nhãn.
Có ba điểm thú vị. Thứ nhất, các nhãn không cân bằng nhưng chưa quá lệch: nhãn nhiều nhất là amusement, gấp 4,5 lần nhãn ít nhất là relief. Thứ hai, cảm xúc đi thành nhóm. Buồn thường đi cùng đau buồn sâu sắc (grief), giận dữ đi cùng khó chịu. Thứ ba, đây là văn bản mạng xã hội: câu ngắn, một phần tư có emoji, và một phần sáu dùng teencode, tức là cách viết tắt kiểu giới trẻ.
Nhóm đã làm gì? Nhóm dùng cách chia dữ liệu chính thức của bộ dữ liệu, giống bài báo gốc: khoảng 16.500 bình luận để huấn luyện, và khoảng 2.000 cho mỗi tập validation và test. Cách chia này không phân tầng. Nhóm cũng kiểm tra chất lượng dữ liệu. Có 306 bình luận vừa gắn nhãn neutral vừa gắn cảm xúc khác, có thể là nhiễu. Nhóm ghi lại nhưng không sửa. Và nhóm không lưu văn bản bình luận, vì giấy phép của bộ dữ liệu.

## Slide 3. Pipeline (~60 s)

**English**

This slide shows what happens to a comment. First, all models get the same light cleaning. We fix HTML codes, remove line-break tags, and normalize Unicode. We keep emoji and teencode on purpose.
Then each model prepares its own input. TF-IDF counts words and word pairs. The BiLSTM builds a vocabulary from the training set only, and cuts or pads each comment to 100 tokens. For PhoBERT, we first turn emoji into English words, because PhoBERT cannot read emoji. Then we segment Vietnamese words with VnCoreNLP, because PhoBERT was pretrained on segmented text.
After the model, we get 28 probabilities, one for each label. We save them before using any threshold. Then a label is predicted if its probability is at least 0.5. Finally, we compute the metrics.

**Tiếng Việt**

Slide này cho thấy một bình luận được xử lý thế nào. Đầu tiên, mọi mô hình đều qua cùng một bước làm sạch nhẹ: đổi mã HTML, bỏ thẻ xuống dòng và chuẩn hóa Unicode. Nhóm cố ý giữ lại emoji và teencode.
Sau đó mỗi mô hình có đầu vào riêng. TF-IDF đếm từ và cặp từ. BiLSTM tạo từ điển chỉ từ tập huấn luyện, rồi cắt hoặc đệm mỗi bình luận về 100 token. Với PhoBERT, nhóm đổi emoji thành từ tiếng Anh, vì PhoBERT không đọc được emoji. Sau đó tách từ tiếng Việt bằng VnCoreNLP, vì PhoBERT được tiền huấn luyện trên văn bản đã tách từ.
Sau mô hình, ta có 28 xác suất, mỗi nhãn một xác suất. Nhóm lưu lại trước khi áp ngưỡng. Một nhãn được dự đoán nếu xác suất từ 0,5 trở lên. Cuối cùng là tính các chỉ số.

## Slide 4. Models (~50 s)

**English**

We compare three models. TF-IDF with logistic regression just counts words. It is the simplest. The BiLSTM reads a sentence in order and learns everything from scratch. PhoBERT is already pretrained on a lot of Vietnamese, and we fine-tune it.
The table shows the test results. PhoBERT is the best on every metric. Its Macro-F1 is 0.608, which is 0.06 higher than TF-IDF and about 0.10 higher than the BiLSTM. The price is size and time: 135 million parameters and about 20 minutes, while TF-IDF takes under 3 seconds. One surprise: the BiLSTM is worse than TF-IDF. We think 16 thousand comments are too few to learn from scratch, but we have not tested this.

**Tiếng Việt**

Nhóm so sánh ba mô hình. TF-IDF với hồi quy logistic chỉ đếm từ, đây là mô hình đơn giản nhất. BiLSTM đọc câu theo thứ tự và học mọi thứ từ đầu. PhoBERT đã được tiền huấn luyện trên rất nhiều văn bản tiếng Việt, và nhóm chỉ tinh chỉnh (fine-tune) nó.
Bảng cho thấy kết quả trên tập test. PhoBERT tốt nhất ở mọi chỉ số. Macro-F1 của PhoBERT là 0,608, cao hơn TF-IDF 0,06 và cao hơn BiLSTM khoảng 0,10. Cái giá là kích thước và thời gian: 135 triệu tham số và khoảng 20 phút huấn luyện, trong khi TF-IDF chưa tới 3 giây. Một điều bất ngờ: BiLSTM kém hơn TF-IDF. Nhóm đoán 16 nghìn bình luận là quá ít để học từ đầu, nhưng chưa kiểm chứng.

## Slide 5. Questions 1 and 2 (~60 s)

**English**

Now the five midterm questions. Question 1: which labels are the easiest and the hardest? The easiest are embarrassment, gratitude and amusement, with F1 around 0.8. The hardest are disapproval, desire and neutral, with F1 around 0.4 to 0.5. The intervals in the table are wide, because each label has few test examples. So only the two ends are reliable. We cannot rank the labels in the middle.
Question 2: does performance depend on how frequent a label is? The correlation is about 0.2, and it is not significant. So we find no clear link. Hard labels are hard because their meaning is vague, not because they are rare. We should be careful: the frequency range is small, so we say "no evidence", not "no relationship".

**Tiếng Việt**

Bây giờ là năm câu hỏi giữa kì. Câu 1: nhãn nào dễ nhất và khó nhất? Dễ nhất là embarrassment, gratitude và amusement, với F1 khoảng 0,8. Khó nhất là disapproval, desire và neutral, với F1 khoảng 0,4 đến 0,5. Các khoảng tin cậy trong bảng khá rộng, vì mỗi nhãn có ít mẫu test. Vì vậy chỉ hai đầu bảng xếp hạng là đáng tin. Nhóm không thể xếp hạng các nhãn ở giữa.
Câu 2: hiệu quả có phụ thuộc vào độ phổ biến của nhãn không? Tương quan khoảng 0,2 và không có ý nghĩa thống kê. Vậy nhóm chưa thấy mối liên hệ rõ ràng. Nhãn khó là do ý nghĩa mơ hồ, không phải do hiếm. Cần thận trọng: khoảng tần suất khá hẹp, nên nhóm nói là "chưa có bằng chứng", không phải "không có quan hệ".

## Slide 6. Question 3 (~60 s)

**English**

Question 3: which emotion pairs appear together, and does the model keep this? In training, the most frequent pairs are disappointment with sadness, anger with annoyance, sadness with grief, and amusement with joy.
The right plot compares the relations between labels in the true data and in the predictions. The points follow the line, with correlation 0.90. So the model keeps the structure. But the bar chart shows something else. PhoBERT predicts these pairs much more often than they really happen. For example, sadness and grief: 71 real pairs, but 152 predicted. Over all pairs, the model predicts 4,741 against 2,466 real ones, and only a quarter of the predicted pairs are correct. So the answer is: the dependencies are kept, but exaggerated.

**Tiếng Việt**

Câu 3: những cặp cảm xúc nào hay xuất hiện cùng nhau, và mô hình có giữ được điều này không? Trong tập huấn luyện, các cặp nhiều nhất là disappointment với sadness, anger với annoyance, sadness với grief, và amusement với joy.
Hình bên phải so sánh mối quan hệ giữa các nhãn trong dữ liệu thật và trong dự đoán. Các điểm nằm sát đường chéo, tương quan 0,90. Vậy mô hình giữ được cấu trúc. Nhưng biểu đồ cột cho thấy điều khác. PhoBERT dự đoán các cặp này nhiều hơn thực tế rất nhiều. Ví dụ sadness và grief: thật có 71 cặp, nhưng mô hình dự đoán 152. Trên tất cả các cặp, mô hình dự đoán 4.741 so với 2.466 cặp thật, và chỉ một phần tư số cặp dự đoán là đúng. Kết luận: sự phụ thuộc được giữ lại, nhưng bị phóng đại.

## Slide 7. Questions 4 and 5 (~90 s)

**English**

Question 4: on which text styles is PhoBERT better than the BiLSTM? Each bar shows the gap in Micro-F1. PhoBERT is ahead in every style, from 0.04 to 0.13. Among the three styles in the question, teencode has the biggest gap, then short comments, then many emoji. But look at the plain-text bar: it is also about 0.10. So long text and teencode are not special. The only real difference is for emoji: there, the gap is smaller. Maybe the BiLSTM sees emoji directly, but we did not test this.
Question 5: how sensitive are the results to the threshold of 0.5? The curve is quite flat. For PhoBERT, the score changes by only 0.03 when the threshold moves from 0.3 to 0.7, and the best shared threshold gains only 0.004. But 0.5 is still arbitrary. The best threshold is different for each model: 0.525, 0.35 and 0.575. It is even 0.25 if we train PhoBERT with the plain loss. And at 0.5, PhoBERT predicts 2.4 labels per comment, but the truth is 1.9.

**Tiếng Việt**

Câu 4: PhoBERT hơn BiLSTM nhiều nhất ở kiểu văn bản nào? Mỗi thanh cho thấy chênh lệch Micro-F1. PhoBERT dẫn trước ở mọi kiểu, từ 0,04 đến 0,13. Trong ba kiểu mà câu hỏi nêu, teencode có chênh lệch lớn nhất, rồi đến câu ngắn, rồi câu nhiều emoji. Nhưng hãy nhìn thanh văn bản thường: cũng khoảng 0,10. Vậy câu dài và teencode không đặc biệt. Khác biệt thật sự duy nhất nằm ở emoji: ở đó chênh lệch nhỏ hơn. Có thể vì BiLSTM nhìn thấy emoji trực tiếp, nhưng nhóm chưa kiểm chứng.
Câu 5: kết quả nhạy với ngưỡng 0,5 đến mức nào? Đường cong khá phẳng. Với PhoBERT, điểm chỉ đổi 0,03 khi ngưỡng chạy từ 0,3 đến 0,7, và ngưỡng chung tốt nhất chỉ tăng 0,004. Nhưng 0,5 vẫn là một lựa chọn tùy ý. Ngưỡng tốt nhất khác nhau theo từng mô hình: 0,525, 0,35 và 0,575. Thậm chí là 0,25 nếu huấn luyện PhoBERT với hàm loss thường. Và ở ngưỡng 0,5, PhoBERT dự đoán 2,4 nhãn mỗi bình luận, trong khi thực tế là 1,9.

## Slide 8. Error analysis (~45 s)

**English**

We read 70 error cases by hand. The main cause, about one third, is confusing similar labels, like anger and annoyance. Next are keyword traps. For example, the word for love makes the model predict love, even when the meaning is different. The model also reads sarcasm literally, and laughing emoji push it towards amusement. Very short comments give too little information.
For comments with four or more labels, only one out of ten was predicted exactly. The model keeps the main emotion and adds one to three extra labels. Overall, PhoBERT predicts too many labels: 2.4 per comment, but the truth is 1.9.

**Tiếng Việt**

Nhóm đã đọc tay 70 ca lỗi. Nguyên nhân chính, khoảng một phần ba, là nhầm giữa các nhãn gần nghĩa, như anger và annoyance. Tiếp theo là bẫy từ khóa. Ví dụ, từ "yêu" khiến mô hình dự đoán love, dù ý nghĩa câu khác. Mô hình cũng hiểu câu mỉa mai theo nghĩa đen, và emoji cười kéo mô hình về amusement. Câu quá ngắn thì không đủ thông tin.
Với các bình luận có từ bốn nhãn trở lên, chỉ một trong mười ca được dự đoán đúng hoàn toàn. Mô hình giữ cảm xúc chính và thêm một đến ba nhãn thừa. Nhìn chung, PhoBERT dự đoán thừa nhãn: 2,4 nhãn mỗi bình luận, trong khi thực tế là 1,9.

## Slide 9. Proposed research question for the Final (~50 s)

**English**

Our proposed research question for the final is: is a single threshold of 0.5 best for all emotion labels, when the labels are imbalanced? The midterm gives three reasons. Labels are imbalanced. The best threshold changes with the model and the loss. And the model predicts too many labels.
We will compare three ways to decide labels: a fixed 0.5, one shared threshold, and one threshold per label. We keep the same PhoBERT and do not retrain it. We tune thresholds only on the validation set, freeze them, and test once. We will look at results by label frequency and by number of labels. We expect small gains, and per-label thresholds may overfit the rare labels.

**Tiếng Việt**

Câu hỏi nghiên cứu nhóm đề xuất cho cuối kì là: một ngưỡng duy nhất 0,5 có tốt nhất cho mọi nhãn cảm xúc khi các nhãn mất cân bằng không? Giữa kì cho ba lý do. Các nhãn mất cân bằng. Ngưỡng tốt nhất thay đổi theo mô hình và hàm loss. Và mô hình dự đoán thừa nhãn.
Nhóm sẽ so sánh ba cách quyết định nhãn: ngưỡng cố định 0,5, một ngưỡng chung, và một ngưỡng riêng cho từng nhãn. Nhóm giữ nguyên PhoBERT và không huấn luyện lại. Ngưỡng chỉ được chọn trên tập validation, sau đó cố định và chỉ test một lần. Nhóm sẽ phân tích theo độ phổ biến của nhãn và theo số nhãn mỗi câu. Nhóm kỳ vọng mức cải thiện nhỏ, và ngưỡng riêng từng nhãn có thể bị quá khớp ở các nhãn hiếm.

## Slide 10. Thank you (~10 s)

**English**

Thank you for listening. We are happy to answer your questions.

**Tiếng Việt**

Cảm ơn thầy cô và các bạn đã lắng nghe. Nhóm sẵn sàng trả lời câu hỏi.

## Tips / Lưu ý

- Read the question on the slide out loud before answering it. / Đọc to câu hỏi trên slide trước khi trả lời.
- Say "no evidence", not "no relationship" (Q2). / Nói "chưa có bằng chứng", không nói "không có quan hệ" (Câu 2).
- If asked about RQ1 results: they belong to the Final; the midterm only proposes the question. / Nếu bị hỏi kết quả RQ1: đó là việc của cuối kì, giữa kì chỉ đề xuất câu hỏi.
