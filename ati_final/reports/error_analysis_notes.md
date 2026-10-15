# Ghi chú phân tích lỗi (PhoBERT, test, ngưỡng 0.5)

Mô hình: `phobert_text_pw_sqrt_ep12` (emoji chuyển thành chữ, BCE + pos_weight căn bậc hai, 12 epoch). Đọc ca bằng `python -m src.show_cases --group <nhóm>`.

> **Lưu ý về cách làm:** ghi chú này do Claude đọc từng ca và soạn, diễn đạt lại bằng lời (không chép nguyên văn bình luận, theo license dataset). Phân loại nguyên nhân là **nhận định chủ quan** dựa trên 70 ca; các chỗ ghi "có thể nhãn gốc thiếu/nhiễu" là giả thuyết, chưa kiểm chứng. Nhóm nên đọc lại và chỉnh sửa trước khi đưa vào báo cáo.

## Tóm tắt

Trong 70 ca đã đọc (60 ca FN/FP và 10 câu nhiều nhãn), có 69 ca lỗi được gán nguyên nhân chính như sau (mỗi ca chọn một nhóm chính; ca có thêm nhóm phụ ghi ở bảng chi tiết). Số lượng chỉ để so sánh thô giữa các nhóm vì mẫu nhỏ và cách chọn ca không phải mẫu đại diện cho toàn bộ lỗi.

| Nguyên nhân chính | FN ngẫu nhiên | FP ngẫu nhiên | FN cực đoan | FP cực đoan | Nhiều nhãn | Tổng |
|---|---|---|---|---|---|---|
| Nhãn gần nghĩa | 8 | 10 | 0 | 2 | 4 | 24 |
| Bẫy từ khoá | 0 | 5 | 0 | 5 | 0 | 10 |
| Mỉa mai / hài đen | 0 | 2 | 3 | 2 | 0 | 7 |
| Emoji / dấu câu gây nhầm | 3 | 2 | 0 | 1 | 1 | 7 |
| Cần ngữ cảnh / câu quá ngắn | 4 | 0 | 4 | 0 | 0 | 8 |
| Teencode / văn bản nhiễu | 2 | 1 | 3 | 0 | 1 | 7 |
| Cảm xúc hỗn hợp | 1 | 0 | 0 | 0 | 2 | 3 |
| Nghi ngờ nhãn gốc | 0 | 0 | 0 | 0 | 1 | 1 |
| Sát ngưỡng | 2 | 0 | 0 | 0 | 0 | 2 |
| (Dự đoán đúng hoàn toàn) | 0 | 0 | 0 | 0 | 1 | 1 |

**Các nhận xét chung**

1. **Nhãn gần nghĩa là nguyên nhân phổ biến nhất**, nhất là trong cụm tích cực (joy, amusement, love, caring, pride, admiration) và cụm tiêu cực (anger, annoyance, disgust, disapproval, sadness, grief). Nhiều ca FP thực ra hợp lý về nghĩa (ví dụ disapproval cho một câu phê phán, relief cho một câu nói về việc thoát khỏi sự phiền toái), gợi ý nhãn gốc có thể thiếu. Điều này cũng khớp với thống kê cặp nhãn hay bị nhầm ở báo cáo tuần 2.
2. **Bẫy từ khoá** gây phần lớn FP mà mô hình rất tự tin (p ≥ 0,9): từ "muốn" → desire, "yêu" → love, "không hiểu" → confusion, từ gợi nguy hiểm → fear, dấu hỏi → confusion/curiosity. Mô hình dựa nhiều vào từ bề mặt hơn là nghĩa cả câu.
3. **Mỉa mai, hài đen, giọng đe ngầm** làm mô hình đọc theo nghĩa đen (khen thành khen thật, chuyện bạo lực đùa thành disgust/anger), gây cả FN cực đoan lẫn FP cực đoan.
4. **Emoji và tiếng cười mâu thuẫn với nội dung**: chuỗi emoji cười/"haha" kéo mô hình sang amusement dù nhãn gốc là nervousness hay disapproval; ngược lại emoji cười nhẹ làm xuất hiện nhãn joy mà chữ không thể hiện. Cách chuyển emoji thành tên tiếng Anh cũng có thể làm mất sắc thái.
5. **FN cực đoan chủ yếu do câu quá ngắn, thiếu ngữ cảnh hoặc văn bản nhiễu** (thiếu dấu, dấu thanh tách rời, hashtag/quảng cáo, viết tắt): mô hình chấm xác suất gần 0 hoặc đoán hẳn nhóm khác.
6. **Câu nhiều cảm xúc (≥ 4 nhãn)**: mô hình thường giữ phần cảm xúc tích cực/chủ đạo và bỏ cảm xúc đối lập cùng câu (ví dụ buồn xen lạc quan), đồng thời hay thêm 1-3 nhãn thừa. Trong 10 ca chỉ có 1 ca đúng hoàn toàn.
7. **Có dấu hiệu nhiễu ở nhãn gốc** (một vài ca nhãn gốc khó giải thích từ nội dung chữ). Cần kiểm tra thêm trước khi kết luận; nếu đúng, một phần "lỗi" của mô hình là do nhãn.

## Bỏ sót nhãn (FN) - ngẫu nhiên phân tầng theo nhóm tần suất

| # | sample_id | nhãn đang xét (p) | nhãn thật | nhãn dự đoán | nhóm lỗi | nhận xét (diễn đạt lại) |
|---|---|---|---|---|---|---|
| 1 | 2912 | joy (0.28) | amusement; joy | amusement | Nhãn gần nghĩa | Câu đùa vui; mô hình nhận ra yếu tố hài hước nhưng không gán thêm joy (p = 0,28). Hai nhãn amusement và joy cùng cụm tích cực nên ranh giới mờ. |
| 2 | 18285 | love (0.31) | love; caring; sadness; grief | caring; sadness; grief | Nhãn gần nghĩa | Lời chia buồn, thương tiếc; mô hình bắt đúng caring, sadness, grief nhưng bỏ love. Love và caring rất gần nhau. |
| 3 | 26598 | anger (0.03) | sadness; anger | disappointment; sadness | Teencode / văn bản nhiễu; Mỉa mai | Than phiền gián tiếp, che chữ thô tục bằng dấu chấm lửng và dùng ẩn dụ; mô hình chỉ thấy nỗi buồn/thất vọng, bỏ qua sự tức giận (p = 0,03). |
| 4 | x000690 | optimism (0.47) | joy; optimism; relief | joy; relief | Sát ngưỡng; Nhãn gần nghĩa | Câu triết lý ngắn về sự đủ đầy, kèm emoji cười nhẹ; optimism chỉ thiếu chút nữa là vượt ngưỡng (p = 0,47), mô hình đã đoán joy và relief. |
| 5 | tik012648 | joy (0.01) | joy; sadness | disappointment; sadness | Emoji / dấu câu gây nhầm; Nghi ngờ nhãn gốc | Tâm sự buồn nhưng kết thúc bằng emoji cười nhẹ; nhãn gốc có joy, trong khi nội dung chữ gần như toàn buồn nên mô hình không thấy (p = 0,01). Có thể nhãn joy chủ yếu đến từ emoji. |
| 6 | x000619 | caring (0.40) | optimism; caring; realization | optimism; realization | Cần ngữ cảnh / câu quá ngắn; Nhãn gần nghĩa | Lời khuyên triết lý về nhân duyên và trưởng thành; "caring" ở đây là sự quan tâm gián tiếp qua lời khuyên, mô hình chỉ bắt optimism và realization (p = 0,40). |
| 7 | tik024068 | joy (0.24) | joy; love; pride | love; pride | Nhãn gần nghĩa; Emoji / dấu câu gây nhầm | Câu tự hào kèm nhiều emoji yêu mến; mô hình giữ love và pride nhưng bỏ joy. Pride và joy gần nhau. |
| 8 | 4010 | curiosity (0.03) | amusement; curiosity | amusement; joy | Cần ngữ cảnh / câu quá ngắn | Câu rất ngắn, gọi một người thân kèm tiếng cười; curiosity chỉ ngầm trong hàm ý hỏi, không có dấu hiệu từ vựng nên mô hình đoán joy. |
| 9 | tik003321 | desire (0.21) | desire; caring | love; caring | Nhãn gần nghĩa; Teencode / văn bản nhiễu | Lời kêu gọi giúp đỡ một cặp đôi, có từ viết tắt; nhãn gốc là desire (mong muốn giúp) nhưng mô hình đoán love. Desire và love dễ lẫn. |
| 10 | 2771 | desire (0.06) | amusement; desire | amusement; excitement; joy | Nhãn gần nghĩa | Lời rủ rê vui kèm biểu tượng nháy mắt dạng chữ; mô hình đoán thêm excitement, joy và bỏ desire. Desire trong lời rủ rê khó nhận ra. |
| 11 | 2189 | desire (0.44) | amusement; excitement; desire | amusement; excitement; curiosity | Nhãn gần nghĩa | Lời thách/hứa hẹn gây ấn tượng; mô hình đoán curiosity thay cho desire (p = 0,44, gần ngưỡng). |
| 12 | 33707 | neutral (0.34) | neutral | realization | Nhãn gần nghĩa; Nghi ngờ nhãn gốc | Câu cực ngắn nêu việc một người nhận ra sai lầm; mô hình đoán realization, nghe hợp lý về nghĩa hơn neutral. Có thể nhãn neutral gốc gây tranh cãi. |
| 13 | tik023873 | pride (0.10) | pride; disappointment | disapproval; neutral | Cần ngữ cảnh / câu quá ngắn; Emoji / dấu câu gây nhầm | Nhận xét về sự phân biệt vùng miền kèm emoji mặt tự mãn; cần biết ngữ cảnh (nói về ai) mới thấy pride và disappointment, mô hình đoán disapproval, neutral. |
| 14 | 23475 | neutral (0.33) | neutral | (không dự đoán nhãn nào) | Sát ngưỡng; Cần ngữ cảnh / câu quá ngắn | Câu ngắn nêu thông tin về hai người/địa điểm, không có từ cảm xúc; mô hình không dự đoán nhãn nào vì không nhãn nào vượt 0,5 (neutral p = 0,33). |
| 15 | thr000339 | remorse (0.04) | remorse; sadness | annoyance; disapproval | Cần ngữ cảnh / câu quá ngắn | Lời xin lỗi chân thành rồi im lặng; mô hình hiểu thành nhóm bực bội/phản đối và hoàn toàn bỏ remorse (p = 0,04). Cần hiểu hàm ý hối lỗi trong lời xin lỗi. |
| 16 | 307 | fear (0.16) | fear; disgust; anger | grief; disgust; anger | Nhãn gần nghĩa | Than khóc vì bị chặn và bị mắng; mô hình đoán grief từ chi tiết khóc, bỏ fear nhưng giữ đúng disgust và anger. |
| 17 | 32486 | nervousness (0.02) | fear; nervousness; disappointment; annoyance | anger; annoyance | Teencode / văn bản nhiễu; Cảm xúc hỗn hợp | Câu dài, nhiều viết tắt và phương ngữ, kể tình huống hài hước nhưng có lo lắng; thật có 4 nhãn, mô hình chỉ giữ annoyance và đoán anger. |
| 18 | 21476 | confusion (0.23) | surprise; confusion; annoyance | amusement; annoyance; disapproval | Emoji / dấu câu gây nhầm | Chuỗi dấu hỏi liên tiếp kèm emoji cười, hỏi về một người lạ kết bạn; emoji cười đẩy mô hình sang amusement, bỏ surprise và confusion dù dấu hỏi đã thể hiện sự bối rối. |
| 19 | 23751 | confusion (0.10) | admiration; confusion; sadness | amusement; joy; admiration; surprise | Cảm xúc hỗn hợp | Vừa khen phim hay vừa thắc mắc vì sao không được đề cử; mô hình chỉ giữ vế khen (admiration, joy, amusement) và bỏ confusion, sadness. |
| 20 | 3272 | fear (0.46) | fear; disgust; anger | amusement; disgust; anger; annoyance | Emoji / dấu câu gây nhầm; Sát ngưỡng | Kể công việc kinh hãi nhưng giọng văn đùa, kèm emoji ma; mô hình đoán amusement thay cho fear (p = 0,46, sát ngưỡng). |

## Dự đoán thừa (FP) - ngẫu nhiên phân tầng

| # | sample_id | nhãn đang xét (p) | nhãn thật | nhãn dự đoán | nhóm lỗi | nhận xét (diễn đạt lại) |
|---|---|---|---|---|---|---|
| 1 | 6454 | love (0.69) | gratitude; realization; sadness | love; gratitude; relief | Nhãn gần nghĩa | Tâm sự biết ơn người cha; mô hình thêm love và relief ngoài gratitude. Love là nhãn gần nghĩa hợp lý với nội dung. |
| 2 | 4766 | amusement (0.61) | curiosity; disappointment; annoyance | amusement; annoyance; disapproval | Mỉa mai / hài đen | Câu hỏi gợi ý cách nấu; giọng châm chọc/mỉa mai khó phân biệt với hài hước nên mô hình đoán amusement (p = 0,61). |
| 3 | 32504 | caring (0.88) | optimism; approval; neutral | love; optimism; caring; approval | Nhãn gần nghĩa | Bàn luận về cách sống có trách nhiệm trong tình yêu; mô hình thêm love, caring (p = 0,88) trong khi nhãn gốc là optimism, approval, neutral. Dự đoán thừa cảm xúc ấm áp. |
| 4 | 31813 | anger (0.71) | annoyance | disgust; anger; annoyance | Nhãn gần nghĩa; Emoji / dấu câu gây nhầm | Bình luận ngắn nhẹ nhàng kèm emoji mặt cười ngược; nhãn gốc chỉ có annoyance, mô hình đẩy lên disgust và anger (p = 0,71), cường độ bị đánh giá quá mức. |
| 5 | 3021 | disgust (0.63) | amusement; surprise | surprise; disgust | Bẫy từ khoá | Từ mang nghĩa kinh ngạc/khen trong khẩu ngữ nhưng cũng là từ chỉ sự ghê tởm; mô hình đoán disgust (p = 0,63) trong khi nhãn gốc là amusement, surprise. |
| 6 | 31784 | disappointment (0.53) | disgust | disappointment; annoyance; disapproval | Mỉa mai / hài đen; Cần ngữ cảnh / câu quá ngắn | Câu so sánh ngầm, mỉa mai ngắn; nhãn gốc là disgust, mô hình chia ra disappointment, annoyance, disapproval (nhầm trong cụm tiêu cực). |
| 7 | 1940 | sadness (0.52) | confusion; remorse | curiosity; confusion; disappointment; sadness | Nhãn gần nghĩa; Nghi ngờ nhãn gốc | Tự vấn vì sao mình chịu đựng được; mô hình thêm sadness (p = 0,52), nghe hợp lý với nội dung nên có thể nhãn gốc thiếu. |
| 8 | 4935 | excitement (0.55) | amusement; joy; love | amusement; excitement; joy; love | Nhãn gần nghĩa | Khen đáng yêu kèm tiếng cười; mô hình chỉ thêm đúng một nhãn thừa là excitement, gần với joy. |
| 9 | 13329 | disapproval (0.68) | anger | anger; annoyance; disapproval | Nhãn gần nghĩa; Nghi ngờ nhãn gốc | Bức xúc đòi xử lý người phá đơn hàng; mô hình thêm annoyance và disapproval (p = 0,68). Lời phản đối hàm ý disapproval nên có thể nhãn gốc thiếu. |
| 10 | 6895 | disapproval (0.85) | disgust; annoyance | anger; annoyance; disapproval | Nhãn gần nghĩa; Nghi ngờ nhãn gốc | Ý kiến phê phán cách xử lý nước thải; mô hình thêm disapproval (p = 0,85) rất hợp về nghĩa, nhãn gốc chỉ có disgust, annoyance. |
| 11 | tik015891 | desire (0.61) | sadness | desire; embarrassment; sadness | Bẫy từ khoá | Có từ "muốn" khiến mô hình đoán desire (p = 0,61) dù nội dung là muốn tránh đám đông vì buồn, nhãn gốc chỉ sadness. |
| 12 | tik023347 | approval (0.66) | pride; admiration | optimism; pride; approval | Emoji / dấu câu gây nhầm; Nhãn gần nghĩa | Khen thành tựu đất nước kèm emoji đồng ý; emoji khiến mô hình thêm approval; nhãn gốc là pride, admiration. |
| 13 | tik005419 | desire (0.94) | amusement; excitement | excitement; desire | Bẫy từ khoá | Từ khẩu ngữ chỉ sự hóng chuyện bị hiểu là mong muốn (desire, p = 0,94 rất chắc); nhãn gốc là amusement, excitement. |
| 14 | 1887 | disapproval (0.66) | disgust; anger | disgust; anger; annoyance; disapproval | Nhãn gần nghĩa; Nghi ngờ nhãn gốc | Chửi gay gắt người có sức khoẻ nhưng lấy cơm từ thiện; mô hình thêm annoyance và disapproval, hợp về nghĩa nhưng nhãn gốc chỉ có disgust, anger. |
| 15 | 1044 | nervousness (0.56) | disgust | nervousness | Teencode / văn bản nhiễu | Suy đoán về một người, có từ viết tắt dễ hiểu nhầm; mô hình đoán nervousness (p = 0,56), nhãn gốc là disgust. |
| 16 | 12472 | grief (0.82) | joy; admiration; sadness | love; sadness; grief | Nhãn gần nghĩa | Khen màn diễn cảm động đến rơi nước mắt; mô hình thêm love và grief (p = 0,82), nhầm giữa khóc vì cảm động và đau buồn. |
| 17 | 3488 | confusion (0.76) | approval | curiosity; confusion | Emoji / dấu câu gây nhầm | Câu hỏi đuôi ngắn mang nghĩa đồng tình; dấu hỏi khiến mô hình đoán curiosity và confusion (p = 0,76), nhãn gốc là approval. |
| 18 | 4237 | confusion (0.83) | amusement | amusement; confusion; annoyance; disapproval | Bẫy từ khoá | Cụm "không hiểu" kích hoạt confusion (p = 0,83) dù thực chất đang khen mỉa nhẹ; nhãn gốc chỉ amusement. |
| 19 | 1890 | grief (0.84) | fear; nervousness; sadness | disappointment; sadness; grief | Nhãn gần nghĩa | Than về thời bình yên không còn; mô hình thêm grief (p = 0,84) và disappointment, bỏ fear, nervousness. Sadness và grief rất gần nhau. |
| 20 | x000024 | relief (0.88) | optimism | optimism; relief | Bẫy từ khoá; Nhãn gần nghĩa | Ý nghĩa thoát khỏi sự phiền toái gợi relief (p = 0,88), rất hợp về nghĩa; nhãn gốc chỉ có optimism nên có thể thiếu nhãn. |

## Bỏ sót nhãn (FN) - mô hình rất chắc là KHÔNG có

| # | sample_id | nhãn đang xét (p) | nhãn thật | nhãn dự đoán | nhóm lỗi | nhận xét (diễn đạt lại) |
|---|---|---|---|---|---|---|
| 1 | 3919 | fear (0.00) | confusion; fear | (không dự đoán nhãn nào) | Cần ngữ cảnh / câu quá ngắn | Kể đang họp bỗng nhớ ra bếp; nỗi lo ngầm không có từ cảm xúc nên mô hình không dự đoán nhãn nào, bỏ cả confusion và fear. |
| 2 | 15999 | optimism (0.00) | amusement; optimism | annoyance; disapproval | Mỉa mai / hài đen | Câu ngắn nhận xét một người lạc quan, vừa khen vừa châm; mô hình hiểu theo hướng chê (annoyance, disapproval) và bỏ optimism, amusement. |
| 3 | 480 | optimism (0.00) | amusement; optimism | amusement; surprise | Cần ngữ cảnh / câu quá ngắn | Bình luận vui về một em nhỏ và tương lai; optimism ẩn trong lời dự đoán, mô hình chỉ thấy amusement và surprise. |
| 4 | 3006 | annoyance (0.00) | annoyance | joy; love; optimism; caring; admiration | Mỉa mai / hài đen | Câu nhắn ngắn mang giọng đe/mỉa nhẹ; mô hình hiểu thành thân mật và đoán toàn nhãn tích cực, sai hoàn toàn so với annoyance. |
| 5 | you001415 | gratitude (0.00) | gratitude; relief | neutral | Cần ngữ cảnh / câu quá ngắn | Đoạn kiến thức dài, lời cảm ơn chỉ ở câu cuối; phần thông tin lấn át, mô hình đoán neutral và bỏ gratitude, relief. |
| 6 | 3145 | surprise (0.00) | surprise | amusement; anger; annoyance | Teencode / văn bản nhiễu | Câu chửi thề thân mật kèm tiếng cười; từ thô tục kéo mô hình sang anger/annoyance, trong khi nhãn gốc là surprise. |
| 7 | tik018096 | relief (0.00) | relief | excitement; approval | Cần ngữ cảnh / câu quá ngắn; Teencode / văn bản nhiễu | Câu ngắn có viết tắt và emoji ly đá; thiếu ngữ cảnh nên mô hình đoán excitement, approval thay vì relief. |
| 8 | 219 | neutral (0.00) | neutral | disgust; anger | Teencode / văn bản nhiễu; Nghi ngờ nhãn gốc | Văn bản bị lỗi dấu (dấu thanh tách rời khỏi chữ) nên khó đọc; có từ xưng hô thô khiến mô hình đoán disgust, anger. Nhãn gốc neutral cũng khó giải thích, có thể là nhiễu nhãn. |
| 9 | 2202 | pride (0.00) | amusement; pride | disgust; anger; annoyance | Mỉa mai / hài đen | Hài đen kể chuyện bắn một con vật; nội dung bạo lực kéo mô hình sang disgust, anger, annoyance, bỏ amusement và pride. |
| 10 | tik000433 | nervousness (0.00) | nervousness | disgust; anger; annoyance | Teencode / văn bản nhiễu | Viết tắt, thiếu dấu, kể suýt bị lộ; mô hình hiểu thành chửi (disgust, anger, annoyance) thay vì nervousness. |

## Dự đoán thừa (FP) - mô hình rất chắc là CÓ

| # | sample_id | nhãn đang xét (p) | nhãn thật | nhãn dự đoán | nhóm lỗi | nhận xét (diễn đạt lại) |
|---|---|---|---|---|---|---|
| 1 | you002775 | admiration (1.00) | joy; approval | joy; admiration | Nhãn gần nghĩa; Nghi ngờ nhãn gốc | Lời khen chân thành (p = 1,00 cho admiration); nhãn gốc là joy, approval. Hợp lý về nghĩa nên có thể thiếu nhãn. |
| 2 | you005917 | remorse (0.99) | confusion; sadness | remorse; sadness | Nhãn gần nghĩa | Tâm sự dài, day dứt về quá khứ; mô hình rất chắc remorse (p = 0,99), nhãn gốc là confusion, sadness. |
| 3 | 30774 | gratitude (0.99) | joy; love; caring; admiration | joy; gratitude; relief | Bẫy từ khoá; Nhãn gần nghĩa | Khen người mẹ; từ "may" gợi relief và mô hình thêm gratitude dù không có lời cảm ơn nào, nhãn gốc là joy, love, caring, admiration. |
| 4 | 32505 | fear (0.99) | disgust | fear; nervousness; disgust | Bẫy từ khoá | Câu cực ngắn dùng từ gợi sự nguy hiểm/độc theo nghĩa lóng; mô hình đoán fear, nervousness (p = 0,99), nhãn gốc chỉ disgust. |
| 5 | tik013968 | desire (0.99) | realization | desire | Bẫy từ khoá | Có từ "muốn" khiến mô hình rất chắc desire (p = 0,99); nhãn gốc là realization (ý thức được điều mình cần). |
| 6 | you000108 | pride (0.99) | disgust | pride; admiration | Mỉa mai / hài đen | Lời khen mỉa mai (nghĩa đen khen, ý chê); mô hình hiểu theo nghĩa đen nên đoán pride, admiration, nhãn gốc là disgust. |
| 7 | 2618 | fear (0.99) | annoyance; disapproval | amusement; fear; nervousness; disgust; annoyance | Bẫy từ khoá; Mỉa mai / hài đen | Chê người che kín mặt khi đi xe và dự đoán hài hước đen về hậu quả; từ chỉ cái chết kích hoạt fear, nervousness (p = 0,99). |
| 8 | 33650 | love (0.99) | admiration | love; caring | Bẫy từ khoá | Có từ "yêu" trong lời đùa thân thiết khiến mô hình đoán love, caring (p = 0,99); nhãn gốc chỉ admiration. |
| 9 | you005813 | amusement (0.99) | nervousness | amusement | Emoji / dấu câu gây nhầm | Nội dung lo lắng về việc tiêm cho trẻ khiến trẻ khóc; chuỗi emoji cười ra mồ hôi kéo mô hình sang amusement (p = 0,99), nhãn gốc nervousness. Nội dung và emoji mâu thuẫn. |
| 10 | 3328 | amusement (0.99) | disapproval | amusement; excitement; joy | Mỉa mai / hài đen; Emoji / dấu câu gây nhầm | Châm biếm chuyện đóng thuế nhiều kèm tiếng cười; dấu hiệu cười làm mô hình chắc là amusement (p = 0,99), nhãn gốc là disapproval. |

## Câu nhiều cảm xúc (|Y| >= 4)

| # | sample_id | nhãn đang xét (p) | nhãn thật | nhãn dự đoán | nhóm lỗi | nhận xét (diễn đạt lại) |
|---|---|---|---|---|---|---|
| 1 | tik013311 | - | confusion; nervousness; remorse; sadness | confusion; nervousness; remorse; sadness | Dự đoán đúng | Dự đoán đúng cả 4 nhãn; văn bản dài nêu rõ cảm xúc. |
| 2 | 2491 | - | joy; pride; admiration; approval | joy; love; optimism; caring; approval | Nhãn gần nghĩa | Khen học sinh làm việc tốt kèm emoji; trúng 2/4 (joy, approval), thừa love, optimism, caring, bỏ pride, admiration. |
| 3 | 2964 | - | joy; optimism; caring; sadness | excitement; joy; optimism | Cảm xúc hỗn hợp | Câu triết lý đón nhận cả nỗi buồn; trúng 2/4, giữ phần tích cực (joy, optimism) và bỏ caring, sadness, thừa excitement. |
| 4 | 563 | - | surprise; curiosity; confusion; disappointment; annoyance | surprise; curiosity; confusion; annoyance | Cảm xúc hỗn hợp | Câu hỏi nghi ngờ; trúng 4/5, chỉ thiếu disappointment. |
| 5 | tik017356 | - | embarrassment; disappointment; sadness; disgust | embarrassment; sadness | Nhãn gần nghĩa | Liệt kê mặc cảm ngoại hình; trúng 2/4 (embarrassment, sadness), bỏ disappointment và disgust (hướng vào bản thân). |
| 6 | 30114 | - | fear; nervousness; anger; annoyance | disappointment; annoyance; disapproval | Nghi ngờ nhãn gốc; Cần ngữ cảnh / câu quá ngắn | Bình luận phân tích kỹ thuật về sự cố hàng không; chỉ trúng 1/4 (annoyance). Nhãn gốc có fear, nervousness, anger nhưng không thấy rõ trong chữ, có thể nhãn mang tính suy diễn. |
| 7 | you004215 | - | amusement; excitement; joy; love | amusement; excitement; joy | Emoji / dấu câu gây nhầm | Khen hay kèm chuỗi emoji cười và một emoji tim; trúng 3/4, bỏ love vì tín hiệu tim bị lấn át. |
| 8 | x001522 | - | love; desire; optimism; caring | optimism; caring; relief | Teencode / văn bản nhiễu; Nhãn gần nghĩa | Bình luận quan tâm người nhà, kèm đoạn quảng cáo và hashtag; trúng 2/4, bỏ love, desire, thừa relief. |
| 9 | 16663 | - | joy; love; caring; admiration | joy; love; caring; admiration; gratitude; approval | Nhãn gần nghĩa | Câu ngắn khen câu chuyện ý nghĩa; trúng cả 4 nhãn thật nhưng thừa gratitude, approval. |
| 10 | 18176 | - | optimism; caring; admiration; sadness | love; desire; optimism; caring; sadness; grief | Nhãn gần nghĩa | Lời động viên gia đình đang gặp chuyện đau lòng; trúng 3/4, thừa love, desire, grief, bỏ admiration. |
