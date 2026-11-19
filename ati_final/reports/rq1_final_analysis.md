# RQ1 cuối kì – độ tin cậy của kết quả và phân tích lỗi

Cập nhật: 2026-11-19. Bổ sung cho [rq1_threshold_comparison.md](rq1_threshold_comparison.md) (mục A–B của đề). Tài liệu này bổ sung phần còn thiếu: chênh lệch giữa ba cách giải mã có vượt nhiễu không (bootstrap, 3 seed), và mục C (phân tích lỗi). Mã: `src/rq1_audit.py`, notebook `notebooks/07_rq1_robustness_errors.ipynb`. Kết quả: `results/rq1/tables/`, `results/rq1/error_analysis/`, `results/rq1/figures/`.

Không thay đổi ngưỡng đã chốt của mô hình chính (`results/rq1/calibration.json`). Không ghi văn bản bình luận ra file: các ca chỉ lưu id, nhãn, xác suất và nhận xét diễn đạt lại.

## 1. Bootstrap ghép cặp trên test (seed 42)

10.000 lần lấy mẫu lại 2.067 câu test (seed 42); ba cách giải mã được chấm trên cùng một mẫu nên chênh lệch là ghép cặp. Bảng: `results/rq1/tables/bootstrap_deltas.csv`.

| So sánh | Macro-F1 Δ [KTC 95%] | Micro-F1 Δ [KTC 95%] | Hamming Δ [KTC 95%] |
|---|---|---|---|
| global − fixed | +0,0043 [−0,0007; +0,0091] | +0,0035 [−0,0010; +0,0081] | **−0,0037 [−0,0044; −0,0029]** |
| per-label − fixed | +0,0060 [−0,0017; +0,0134] | +0,0021 [−0,0042; +0,0082] | **−0,0031 [−0,0041; −0,0020]** |
| per-label − global | +0,0017 [−0,0056; +0,0090] | −0,0015 [−0,0076; +0,0047] | +0,0006 [−0,0004; +0,0016] |

- Không chênh lệch Macro-F1 hay Micro-F1 nào có khoảng tin cậy loại trừ 0. Per-label hơn global ở 67% số lần lấy mẫu lại, gần như tung đồng xu.
- Thứ duy nhất chắc chắn là **cả hai cách tinh chỉnh đều giảm Hamming Loss so với 0,5** (giảm dự đoán thừa).
- Theo nhóm tần suất, per-label chỉ hơn rõ ở **head** (Macro-F1 +0,0139 [+0,0048; +0,0226] so với 0,5; +0,0135 so với global). Ở **tail**, per-label − fixed = −0,0041 [−0,0221; +0,0130] và per-label − global = −0,0087 [−0,0255; +0,0078]: không có bằng chứng ngưỡng riêng giúp nhãn hiếm, hướng ước lượng còn là âm. Ở **mid**, per-label kém global về Micro-F1 (−0,0202 [−0,0343; −0,0061]) và Hamming (+0,0037 [+0,0020; +0,0054]).

## 2. Ba seed huấn luyện

Train lại đúng cấu hình của mô hình chính (emoji → chữ, BCE + pos_weight căn bậc hai, 12 epoch, patience 3) với seed 43 và 44. Mỗi seed có bộ ngưỡng riêng: fit trên val của chính nó, ghi `calibration.json` trước khi mở file test (`results/rq1/seeds/<run>/`). Seed 43 chạy đủ 12 epoch (val Macro-F1 tốt nhất 0,6097). Seed 44 dừng sớm ở epoch 11 và là mô hình yếu nhất. Bảng: `tables/seed_results.csv`, `seed_mean_std.csv`, `seed_macro_f1_deltas.csv`; hình `figures/seed_robustness.png`.

**Trung bình ± độ lệch chuẩn trên test (3 seed, ddof = 1)**

| Chỉ số | fixed 0,5 | global | per-label |
|---|---|---|---|
| Macro-F1 | 0,6036 ± 0,0088 | 0,6098 ± 0,0058 | **0,6115 ± 0,0031** |
| Micro-F1 | 0,6190 ± 0,0087 | **0,6254 ± 0,0058** | 0,6241 ± 0,0030 |
| Hamming Loss | 0,0591 ± 0,0017 | **0,0544 ± 0,0010** | 0,0551 ± 0,0015 |
| Số nhãn dự đoán TB / câu (thật: 1,91) | 2,44 ± 0,03 | 2,16 ± 0,04 | 2,20 ± 0,09 |
| Macro-F1 head | 0,6346 ± 0,0066 | 0,6391 ± 0,0067 | **0,6472 ± 0,0017** |
| Macro-F1 mid | 0,6017 ± 0,0078 | **0,6091 ± 0,0038** | 0,6050 ± 0,0049 |
| Macro-F1 tail | 0,5673 ± 0,0159 | 0,5741 ± 0,0128 | 0,5751 ± 0,0026 |
| Gap val − test (Macro-F1) | +0,0002 ± 0,0052 | −0,0009 ± 0,0034 | **+0,0157 ± 0,0035** |

**Chênh lệch Macro-F1 ghép cặp theo từng seed**

| Seed | global − fixed | per-label − fixed | per-label − global | Ngưỡng chung |
|---|---|---|---|---|
| 42 | +0,0043 | +0,0060 | +0,0017 | 0,575 |
| 43 | +0,0045 | +0,0030 | **−0,0015** | 0,600 |
| 44 | +0,0098 | +0,0147 | +0,0049 | 0,600 |
| TB ± std | +0,0062 ± 0,0031 | +0,0079 ± 0,0061 | +0,0017 ± 0,0032 |  |

- **Tinh chỉnh so với 0,5 lặp lại ở cả 3 seed:** cả global và per-label đều tăng Macro-F1, giảm Hamming và giảm số nhãn đoán thừa. Ngưỡng chung tối ưu ổn định (0,575–0,600, luôn trên 0,5).
- **Per-label so với global đổi dấu giữa các seed** (+0,0017; −0,0015; +0,0049). Trung bình +0,0017 nhỏ hơn độ lệch chuẩn 0,0032, khớp với bootstrap ở mục 1.
- **Gap val − test của per-label dương ở cả 3 seed** (+0,012; +0,018; +0,017), trong khi fixed và global xấp xỉ 0. Đây là tín hiệu quá khớp ngưỡng lặp lại được.
- **Ở tail, per-label ngang global** (0,5751 so với 0,5741). Lợi ích ổn định duy nhất của per-label là ở head (+0,008 so với global; độ lệch chuẩn của per-label chỉ 0,0017).
- **Bản thân các ngưỡng không ổn định:** biên độ trung bình giữa 3 seed là 0,175 (`tables/seed_threshold_stability.csv`). Chỉ remorse có cùng ngưỡng; grief 0,875 / 0,750 / 0,550; fear 0,425 / 0,525 / 0,750; disgust 0,450 / 0,750 / 0,475. Ngưỡng riêng phản ánh mô hình và nhiễu của val, không phải đặc tính của từng cảm xúc.
- Mô hình yếu nhất (seed 44, dừng sớm) được lợi nhiều nhất từ tinh chỉnh (+0,0098 / +0,0147). Điều này khớp với câu 5 giữa kì: tinh chỉnh ngưỡng bù được nhiều hơn khi xác suất của mô hình lệch xa 0,5. Với n = 3 đây chỉ là quan sát, chưa phải kết luận.

## 3. Ngưỡng riêng có thật sự "học" được gì từ val?

Theo cách tối ưu, ngưỡng riêng không thể làm F1 val của nhãn nào giảm. Câu hỏi là mức tăng trên val có chuyển sang test không. Với seed 42: 27/28 nhãn tăng trên val (amusement giữ ngưỡng 0,5), nhưng chỉ 14/28 nhãn tăng trên test, 12/28 giảm. **Tương quan Spearman giữa mức tăng val và mức tăng test của từng nhãn chỉ là 0,09.** Ví dụ: relief (tail, 58 mẫu dương val) tăng +0,076 trên val nhưng giảm −0,031 trên test; curiosity tăng +0,017 trên val nhưng giảm −0,047 trên test. Đây là bằng chứng trực tiếp nhất cho việc ngưỡng riêng khớp nhiễu của val (khoảng 58–354 mẫu dương mỗi nhãn).

Lặp lại trên 3 seed (`tables/seed_gain_transfer.csv`):

| Seed | Spearman (tăng val, tăng test) | Nhãn tăng trên val | Nhãn tăng / giảm trên test | Tăng TB val | Tăng TB test |
|---|---|---|---|---|---|
| 42 | 0,09 | 27 | 14 / 12 | +0,023 | +0,006 |
| 43 | −0,18 | 26 | 13 / 13 | +0,021 | +0,003 |
| 44 | 0,31 | 26 | 20 / 6 | +0,026 | +0,015 |

Ở cả ba seed, mức tăng trung bình trên test chỉ bằng 14–56% mức tăng trên val, và tương quan giữa hai mức tăng không vượt ngưỡng có ý nghĩa (với 28 nhãn cần |ρ| ≥ 0,375).

## 4. Phân tích lỗi (mục C)

### 4.1 Định lượng

**Số lỗi trên test (cộng 28 nhãn).** Bảng: `error_analysis/error_counts.csv`.

| Cách giải mã | TP | FP | FN | Số nhãn dự đoán TB / câu |
|---|---|---|---|---|
| fixed 0,5 | 2.801 | 2.234 | 1.141 | 2,44 |
| global 0,575 | 2.664 | 1.884 | 1.278 | 2,20 |
| per-label | 2.675 | 1.928 | 1.267 | 2,23 |

Số nhãn thật trung bình là 1,91. Cả hai cách tinh chỉnh đều đổi khoảng 350 FP lấy khoảng 130 FN. Đây là lý do Hamming giảm chắc chắn còn F1 gần như không đổi.

**Quyết định bị lật khi chuyển từ 0,5 sang ngưỡng riêng** (`error_analysis/decision_flips.csv`, hình `figures/decision_flips.png`):

| So sánh | FN được cứu | FN mới sinh | FP được bỏ | FP mới sinh | Ròng |
|---|---|---|---|---|---|
| global vs fixed | 0 | 137 | 350 | 0 | +213 |
| per-label vs fixed | 68 | 194 | 491 | 185 | +180 |
| per-label vs global | 137 | 126 | 279 | 323 | **−33** |

- Per-label vs global lật 865 quyết định nhưng ròng còn **tệ hơn 33 quyết định**. Macro-F1 vẫn nhích +0,0017 vì các thay đổi rơi vào nhãn ít mẫu, nơi mỗi quyết định nặng ký hơn trong trung bình macro.
- **Ngưỡng cao** ở cụm tích cực/thương tiếc (grief 0,875, love 0,80, caring 0,75, approval 0,70) bỏ nhiều FP: love ròng +47, caring +47, grief +40.
- **Ngưỡng thấp** ở cụm tiêu cực (annoyance 0,325, disapproval 0,375, realization 0,40, disgust 0,45) chỉ sinh thêm FP: annoyance ròng −41 (70 FP mới, 29 FN cứu được), disapproval −44. Ở cụm này mô hình vốn lẫn nhãn với nhau; hạ ngưỡng chỉ làm nó gán thêm nhãn gần nghĩa.

**Cặp nhãn bị nhầm** (`error_analysis/confused_pairs.csv`; φ tính từ nhãn train):

- Bỏ sót i nhưng đoán j trong cùng câu (per-label): disgust→disapproval 21 câu, disappointment→disapproval 18, anger→annoyance 17, sadness→annoyance 16, disappointment→annoyance 15. Gần như toàn bộ nằm trong cụm tiêu cực. Một số cặp có φ train âm (sadness–annoyance −0,12), tức mô hình nhầm cả những nhãn ít khi đi cùng nhau.
- Đúng i nhưng thêm j thừa: annoyance+disapproval 86 câu (fixed: 58), annoyance+disgust 62, anger+annoyance 58. Ngưỡng riêng làm cặp annoyance–disapproval tăng mạnh. Ngược lại, sadness+grief thừa giảm từ 57 câu (fixed) và 49 (global) xuống 19 nhờ ngưỡng grief 0,875.

**Theo số nhãn thật** (`error_analysis/cardinality_errors.csv`, hình `figures/cardinality_errors.png`):

| \|Y\| | n | Dự đoán thiếu: fixed / global / per-label | Dự đoán thừa: fixed / global / per-label |
|---|---|---|---|
| 1 | 695 | 1% / 2% / 1% | 64% / 57% / 57% |
| 2 | 946 | 12% / 17% / 19% | 47% / 35% / 39% |
| 3 | 360 | 33% / 43% / 48% | 23% / 14% / 14% |
| ≥4 | 66 | 64% / 79% / 79% | 14% / 5% / 2% |

Mô hình đoán thừa ở câu một nhãn và đoán thiếu ở câu nhiều nhãn. Ngưỡng nào cũng chỉ là một con số cho mỗi nhãn, không biết câu đang xét có mấy cảm xúc, nên tinh chỉnh ngưỡng **làm giảm lỗi thừa nhưng tăng lỗi thiếu**. Với câu ≥4 nhãn, tỉ lệ đúng toàn bộ tập nhãn giảm từ 7,6% (fixed) xuống 6,1% (per-label) và 3,0% (global).

### 4.2 Đọc tay 80 ca

Mẫu chọn bằng seed 42 trước khi đọc văn bản, loại 70 câu đã đọc ở giữa kì (`error_analysis/cases.csv`):

- 20 FN và 20 FP còn lại dưới ngưỡng riêng (7 head / 7 mid / 6 tail);
- 10 FN mới sinh và 10 FP mới sinh do ngưỡng riêng; 5 FN được cứu và 5 FP được bỏ;
- 10 câu có ít nhất 4 nhãn thật.

Mỗi ca có một nguyên nhân chính và có thể thêm một nguyên nhân phụ (`error_analysis/annotations.csv`). Đây là nhận định chủ quan do Claude đọc và gán; nhóm cần đọc lại trước khi nộp.

| Nguyên nhân chính | FN còn lại | FP còn lại | FN mới | FP mới | Cứu/bỏ | ≥4 nhãn | Tổng |
|---|---|---|---|---|---|---|---|
| Quyết định do ngưỡng tinh chỉnh lật | 3 | 2 | 10 | 10 | 10 | 0 | 35 |
| Nhãn gần nghĩa | 7 | 5 | 0 | 0 | 0 | 0 | 12 |
| Cảm xúc hỗn hợp / phụ | 0 | 0 | 0 | 0 | 0 | 7 | 7 |
| Bẫy từ khoá | 2 | 4 | 0 | 0 | 0 | 0 | 6 |
| Teencode, phương ngữ, văn bản nhiễu | 4 | 1 | 0 | 0 | 0 | 1 | 6 |
| Mỉa mai / hài đen | 1 | 3 | 0 | 0 | 0 | 1 | 5 |
| Thiếu ngữ cảnh / câu quá ngắn | 2 | 2 | 0 | 0 | 0 | 0 | 4 |
| Emoji / emoticon | 0 | 2 | 0 | 0 | 0 | 0 | 2 |
| Nhãn gốc đáng ngờ / thiếu | 0 | 1 | 0 | 0 | 0 | 1 | 2 |
| Điểm sát 0,5 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |

Các nhóm "mới/cứu/bỏ" được chọn theo định nghĩa là quyết định bị ngưỡng lật, nên nguyên nhân chính luôn là "ngưỡng". Phần đáng xem ở các nhóm này là nguyên nhân phụ và việc dự đoán có hợp lý không.

**Nhận xét chính**

1. **FN mới sinh đều là lỗi thật (10/10 không hợp lý).** Cả 10 ca đều có tín hiệu rõ trong chữ: khóc vì mất người yêu → grief; "thương các con" → caring; lời khuyên buông bỏ để thanh thản → relief; nhận lỗi "không nên nói ra" → remorse. Mô hình đã chấm 0,51–0,83, đúng ở ngưỡng 0,5, nhưng ngưỡng riêng (0,725–0,875) cắt mất. Ngưỡng cao "trả giá" cho nhãn đúng có xác suất vừa phải.
2. **Một nửa số FP mới sinh là hợp lý (5/10).** Chúng nằm trong cụm annoyance/disgust/disapproval: gán annoyance cho câu đòi bỏ tù kẻ xả thải là chấp nhận được dù nhãn gốc là disgust/disapproval. Các nhãn trong cụm này được gán không nhất quán trong dữ liệu, nên hạ ngưỡng chỉ làm số liệu tệ đi mà không chắc là mô hình sai. Ca tệ nhất: câu chỉ có một emoji khỉ che mắt, không nhãn nào vượt 0,5, nhưng annoyance vượt 0,325.
3. **Lỗi còn lại sau khi tinh chỉnh vẫn giống giữa kì.** Nhãn gần nghĩa nhiều nhất (12/40 ca FN/FP), tiếp theo là bẫy từ khoá ("biết đâu" → confusion; "may quá" → relief; "lãng mạn" → love/desire), teencode và phương ngữ (FN cực đoan, p < 0,1), mỉa mai bị đọc theo nghĩa đen. Ngưỡng không chạm được tới các lỗi này vì xác suất ở rất xa ngưỡng (p = 0,00–0,16 với FN; 0,70–0,96 với FP).
4. **Có ca thiếu tín hiệu hiển nhiên:** một bài liệt kê mẹo tâm lý kết thúc bằng "Em cảm ơn thầy" nhưng gratitude chỉ được 0,08. Câu dài, có ngắt dòng (HTML `<br>`) làm tín hiệu cuối câu bị lấn át.
5. **Câu ≥4 nhãn: không cách giải mã nào cứu được.** Trong 10 ca, mô hình thường giữ cảm xúc chủ đạo (love/caring, fear/nervousness) và bỏ cảm xúc phụ hoặc đối nghịch (grief trong lời chia buồn, remorse/sadness khi sợ bị mẹ mắng). Ngưỡng riêng còn bỏ thêm gratitude ở một ca (ngưỡng 0,85).
6. **Nhãn gốc nhiễu vẫn xuất hiện** (9/80 ca có nhắc tới làm nguyên nhân chính hoặc phụ): ví dụ một câu hai từ chỉ bày tỏ sự thương hại lại mang nhãn caring/nervousness/annoyance; một câu mỉa mai về lòng tin mang nhãn approval/nervousness.

## 5. Trả lời RQ1

> *Một ngưỡng chung 0,5 có tối ưu cho cả 27 cảm xúc khi mất cân bằng mạnh không?*

**Không, nhưng phần lợi nằm ở một ngưỡng chung cao hơn, không nằm ở 27 ngưỡng riêng.**

1. **0,5 không tối ưu.** Với mô hình PhoBERT đã chọn (BCE + pos_weight căn bậc hai), xác suất bị đẩy lên nên 0,5 dự đoán thừa: 2,44 nhãn mỗi câu so với 1,91 nhãn thật. Một ngưỡng chung chọn trên val (0,575–0,600) tăng Macro-F1 test ở cả 3 seed (+0,0062 ± 0,0031). Nó cũng giảm Hamming Loss một cách chắc chắn (bootstrap seed 42: −0,0037 [−0,0044; −0,0029]). Riêng mức tăng Macro-F1 trong một seed thì chưa vượt nhiễu của mẫu test (KTC [−0,0007; +0,0091]).
2. **Ngưỡng riêng từng nhãn không mang lại lợi ích đáng tin so với ngưỡng chung:** +0,0017 ± 0,0032 qua 3 seed, đổi dấu giữa các seed, KTC bootstrap chứa 0. Kèm theo đó là ba dấu hiệu quá khớp val: gap val − test +0,016 lặp lại ở mọi seed; mức tăng val của từng nhãn không chuyển sang test; ngưỡng của cùng một nhãn dao động tới 0,3 giữa các seed.
3. **Mất cân bằng không phải lý do ngưỡng riêng có ích:** nhãn tail không được lợi (per-label 0,5751 so với global 0,5741). Lợi ích nhỏ và ổn định duy nhất là ở nhãn head, nơi val có nhiều mẫu dương nhất. Mất cân bằng của ViGoEmotions chỉ ở mức vừa (nhãn hiếm nhất vẫn có 635 mẫu train), và pos_weight đã bù một phần.
4. **Lỗi chính không sửa được bằng ngưỡng:** mô hình đoán thừa ở câu một nhãn và đoán thiếu ở câu nhiều nhãn. Ngưỡng cố định theo nhãn chỉ dịch được điểm cân bằng giữa hai lỗi này. Phần lỗi còn lại (nhãn gần nghĩa, bẫy từ khoá, mỉa mai, teencode) có xác suất ở rất xa ngưỡng.

**Khuyến nghị:** dùng một ngưỡng chung chọn trên val thay cho 0,5. Chỉ cân nhắc ngưỡng riêng cho nhãn có nhiều mẫu dương ở val, hoặc chuyển sang cách giải mã theo số nhãn của câu (hướng mở rộng, chưa thử).

## 6. Hạn chế

- Test đã được xem ở giữa kì; đây là so sánh hậu nghiệm. Thuật toán, lưới và tie rule được cố định trước khi đánh giá, ngưỡng chỉ fit trên val.
- Seed 43/44 được train sau khi đã thấy kết quả seed 42 trên test. Chúng dùng đúng cấu hình đã chốt và không được dùng để chọn lại gì, chỉ để đo độ biến thiên.
- Bootstrap chỉ đo nhiễu do mẫu test, không đo nhiễu do huấn luyện. Ba seed đo nhiễu huấn luyện nhưng n = 3 quá nhỏ để kiểm định.
- Phân loại nguyên nhân trong 80 ca là nhận định chủ quan, mỗi ca một người đọc (Claude), chưa đo độ đồng thuận giữa người gán.
- Val chỉ có 58–354 mẫu dương mỗi nhãn, nên ngưỡng riêng dễ khớp nhiễu. Đây vừa là hạn chế vừa là một phần câu trả lời.
