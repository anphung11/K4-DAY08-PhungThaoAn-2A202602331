# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Phùng Thảo An
Công cụ gán nhãn đã dùng: CVAT

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian với vùng đệm ở giữa nhằm ngăn chặn hiện tượng rò rỉ dữ liệu (data leakage). Trong video giao thông, các frame ảnh sát nhau có độ tương đồng cực kỳ cao về bối cảnh, ánh sáng và vị trí xe. Vùng đệm (buffer) giúp đảm bảo tập test thực sự là những dữ liệu "mới lạ" đối với mô hình. 

Nếu chia ngẫu nhiên, các frame gần như giống hệt nhau sẽ bị chia đều vào cả tập train và tập test. Khi đó, số đo trên tập kiểm thử sẽ bị lệch theo hướng **lạc quan thái quá (ảo tưởng sức mạnh)**, khiến AP tăng vọt nhưng mô hình lại không có khả năng tổng quát hóa khi gặp dữ liệu ở một thời điểm hay điều kiện ánh sáng khác trong thực tế.

## 2. Mô hình khởi đầu lạnh (cold start)

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào `outputs/compare_round0.jpg`, mô hình khởi đầu lạnh thường không khớp với nhãn tham chiếu ở các loại xe ở khoảng cách rất xa (kích thước pixel quá nhỏ) hoặc các xe đi ngược chiều bị lóa đèn pha. Độ phủ (recall) theo kích thước xe cho thấy mô hình làm khá tốt với xe to/gần (0.561) và xe trung bình (0.547), nhưng bỏ sót rất nhiều xe nhỏ (0.182). 

Trường hợp cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai là khi mô hình dự đoán đúng một chiếc xe lấp ló trong bóng tối hoặc bị che khuất, nhưng nhãn tham chiếu (do cũng được tạo tự động) lại bỏ sót. Lúc này, dự đoán (prediction) thực chất là đúng (False Positive giả), còn nhãn tham chiếu (ground truth) mới là sai.

## 3. Chiến lược chọn mẫu

Công thức `score = W_U·U + W_A·A + W_D·D` là hàm đánh giá tổng hợp để chọn ra những frame "có giá trị học hỏi cao nhất":
*   **U (Uncertainty):** Độ bất định. Ưu tiên những ảnh mà mô hình đang bối rối, dự đoán với độ tự tin thấp.
*   **A (Area/Size):** Kích thước vật thể. Giúp điều chỉnh ưu tiên chọn ảnh có nhiều xe to rõ ràng hoặc ảnh có nhiều xe nhỏ tùy mục tiêu.
*   **D (Diversity/Distance):** Độ đa dạng. Ưu tiên những ảnh có bối cảnh khác biệt so với những ảnh đã chọn.
*   **MIN_GAP_S:** Khoảng cách thời gian tối thiểu giữa các frame được chọn. Tham số này ép hệ thống không được chọn 2 ảnh quá sát nhau (ví dụ cách nhau chỉ 0.1 giây), giúp tiết kiệm công gán nhãn cho những ảnh trùng lặp thông tin.

*Chứng minh qua frame:*
Dựa trên danh sách các ảnh được chọn, thuật toán đã lấy `frame_0099.jpg`, `frame_0107.jpg`, và `frame_0182.jpg` vì chúng có điểm số kết hợp cao. Tuy nhiên, các frame nằm sát ngay sau `frame_0099.jpg` (như `frame_0100` hay `frame_0101`) dù có thể có độ bất định cao tương đương nhưng chắc chắn đã bị loại bỏ vì vi phạm khoảng cách thời gian `MIN_GAP_S`, nhường suất chọn cho một ảnh đa dạng hơn ở thời điểm xa hơn.

Điểm bất định cao **không phải lúc nào cũng chứng minh** ảnh đó sẽ cải thiện mô hình. Nếu mô hình bối rối do nhiễu vật lý (như ánh sáng đèn pha chói lóa tạo quầng sáng ảo, sương mù dày đặc), việc nhồi nhét thêm ảnh đó chỉ làm tốn công gán nhãn mà không giải quyết được giới hạn vật lý của camera (Aleatoric uncertainty).

## 4. Các vòng học chủ động (active learning)

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 279 | 0.508 | -0.263 | 1.000 | 0.045 | 0.086 | 0.000 | 0.054 | 0.049 |

**Mức độ chỉnh sửa nhãn (Vòng 1 - 12 ảnh):**
Tôi đã sửa lại rất nhiều nhãn do mô hình đề xuất. Cụ thể: mô hình đề xuất 169 box, tôi đã giữ nguyên 127 box, chỉnh sửa 22 box, xóa đi 20 box sai (False Positives) và phải tự vẽ thêm tới 130 box mới (False Negatives), đạt tỷ lệ chấp nhận (accept rate) là 75%.

**Biến động hiệu suất:**
* AP50 giảm mạnh từ 0.771 xuống chỉ còn 0.508 (giảm 0.263) so với vòng khởi đầu lạnh.
* Tất cả các nhóm xe đều xấu đi nghiêm trọng sau khi fine-tune. Đặc biệt, nhóm xe nhỏ (small) giảm độ phủ từ 0.182 về 0.000 (không nhận diện được xe nhỏ nào), nhóm xe trung bình (medium) giảm từ 0.547 về 0.054.

**Phân tích ca kiểm thử (Ca frame_0107):**
Ở bước quét mù (Blind Scan) trên `frame_0107.jpg`, quan sát độc lập cho thấy khu vực cụm đèn nối đuôi xa và các xe đi ngược chiều bật đèn pha chói là vị trí dễ sai nhất. Trong quá trình rà soát trên CVAT, tôi đã phải giữ nguyên 11 box, xóa 2 box sai, và vẽ thêm tới 10 box mới cho ảnh này. Mặc dù đã gán nhãn rất kỹ, nhưng sau khi train lại, điểm số lại giảm, cho thấy mô hình có dấu hiệu bị "học vẹt" (overfitting) các đặc trưng cục bộ của 12 ảnh này và quên đi kiến thức chung.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start **giảm sút nghiêm trọng** (-0.263 AP50). Việc AP50 giảm mạnh kèm theo Precision tuyệt đối (1.0) nhưng Recall cực kỳ thấp (0.0447) là minh chứng rõ ràng cho việc mô hình đã trở nên quá thận trọng hoặc bị "quên vĩnh viễn" (catastrophic forgetting) kiến thức cũ khi chỉ được fine-tune đơn độc trên 12 tấm ảnh mới suốt 50 epochs. 

**Đề xuất cho vòng sau:**
Cần tập trung vào các frame có điều kiện ánh sáng giao thoa hoặc các xe ở làn xa cùng chiều. Chi phí rà nhãn cho các ca này sẽ cao do phải zoom kỹ, và có nguy cơ trùng lặp bối cảnh nếu không kiểm soát tốt `MIN_GAP_S`. Quan trọng hơn, cần trộn lẫn thêm một lượng dữ liệu từ COCO hoặc vòng trước để mô hình không bị quên kiến thức cũ.

**Ảnh hưởng của các giới hạn hiện tại:**
1. Tập kiểm thử 20 ảnh là quá nhỏ, không đủ đại diện cho toàn bộ phân phối dữ liệu. Một thay đổi nhỏ có thể làm biến động lớn điểm AP, dẫn đến đánh giá sai lệch năng lực thực tế.
2. Nhãn tham chiếu do mô hình tạo chưa được rà thủ công nghĩa là ta đang dùng "thước đo có sai số" để đo lường. Nếu mô hình đoán đúng một chiếc xe mà nhãn tham chiếu sai, nó lại bị phạt điểm.

**Xử lý khi AP50 giảm:**
Với tình trạng AP50 giảm mạnh như hiện tại, trước khi train thêm vòng 2, tôi cần kiểm tra lại:
1. File `round1_diff.md` xem mình có lỡ tay gán nhầm class ID (ví dụ gán xe thành người/biển báo) trong CVAT không.
2. Thông số huấn luyện: 50 epochs cho bộ dữ liệu 12 ảnh mới thêm vào là quá cao, gây overfit cục bộ. Cần giảm số epoch hoặc trộn chung 12 ảnh này với dữ liệu nền để fine-tune.
