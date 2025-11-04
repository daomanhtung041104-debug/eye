# Eye Tracking Application

Ứng dụng theo dõi vị trí lòng đen (iris) của mắt sử dụng MediaPipe và ByteTrack.
---------------

| Test | Mục tiêu | Bước thực hiện | Kết quả mong đợi |
|------|----------|----------------|-------------------|
| Test 1 | Phát hiện một người, một khuôn mặt | Đứng trước webcam, quan sát panel và màn hình | Phát hiện được 2 mắt (trái và phải), mỗi mắt có track ID riêng, vị trí lòng đen hiển thị chính xác |
| Test 2 | Phát hiện nhiều người, nhiều mắt | 2-3 người cùng đứng trước webcam | Phát hiện được tất cả mắt (4-6 mắt), mỗi mắt có track ID riêng và ổn định |
| Test 3 | ByteTrack tracking ổn định | Di chuyển đầu từ trái sang phải, lên xuống, quan sát track ID | Track ID giữ nguyên khi di chuyển nhẹ, có thể thay đổi khi di chuyển quá nhanh |
| Test 4 | Focus mode - tập trung vào một mắt | Nhấn `f`, nhập track ID, quan sát màn hình và panel | Chỉ mắt được chọn hiển thị, panel hiển thị FOCUS: Track X, reset focus hoạt động đúng |
| Test 5 | MediaPipe Stream Mode | Quan sát panel, di chuyển đầu nhanh | Stream Mode = TRUE, tracking cập nhật realtime không lag |
| Test 6 | Che khuất tạm thời (Occlusion) | Che một mắt bằng tay 1-2 giây rồi bỏ tay ra | Track bị mất khi che, có thể được khôi phục khi bỏ tay ra |

## Bảng đánh giá

| Tiêu chí | Yêu cầu | Kết quả | Ghi chú |
|----------|---------|---------|---------|
| Phát hiện nhiều mắt | Trích xuất vị trí của nhiều mắt trên màn hình | Đạt | MediaPipe phát hiện được nhiều khuôn mặt, mỗi mắt được track riêng |
| Vị trí lòng đen | Trích xuất vị trí lòng đen (iris) | Đạt | Sử dụng MediaPipe iris landmarks (468-477), hiển thị tọa độ X, Y |
| MediaPipe Stream Mode | stream = True cho MediaPipe | Đạt | static_image_mode=False, xử lý video stream realtime |
| ByteTrack | Sử dụng ByteTrack để tập trung vào một mắt | Đạt | ByteTrack tracking với track ID, focus mode cho phép chọn một mắt |
| Real-time | Xử lý realtime từ webcam | Đạt | FPS ~30, xử lý frame từ cv2.VideoCapture(0) |
| Ổn định tracking | Track ID ổn định khi di chuyển | Khá | Track ID giữ nguyên khi di chuyển nhẹ, có thể đổi khi di chuyển nhanh |

