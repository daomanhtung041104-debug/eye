# Eye Tracking Application

Ứng dụng theo dõi vị trí lòng đen (iris) của mắt sử dụng MediaPipe và ByteTrack.

### 1. MediaPipe FaceMesh

Sử dụng `mp.solutions.face_mesh.FaceMesh` với `static_image_mode=False` để xử lý luồng video liên tục.

Trích xuất các vùng mắt và mống mắt thông qua tập chỉ số landmark:

- Mắt trái: `LEFT_EYE = [33, 7, 163, 144, ..., 246]`
- Mắt phải: `RIGHT_EYE = [362, 382, ..., 398]`
- Mống mắt (iris): `LEFT_IRIS = [468-472]`, `RIGHT_IRIS = [473-477]`

Từ các điểm này tính bounding box và trung tâm iris theo công thức trung bình tọa độ.

### 2. ByteTrack Tracking

Mỗi mắt được coi là một detection box: `[x1, y1, x2, y2, score]`.

ByteTrack chia detection thành hai nhóm:

- High score (≥0.6): tin cậy, ưu tiên match
- Low score (<0.6): dùng để khôi phục track tạm mất

Sử dụng IoU matching (Hungarian hoặc greedy) với ngưỡng `match_thresh=0.5`.

Mỗi track được quản lý bởi `KalmanTracker`, lưu trạng thái `[cx, cy, s, r]` (tâm, diện tích, tỉ lệ).

Khi không có detection mới, Kalman dự đoán vị trí kế tiếp để duy trì ID.

Track được giữ tối đa `track_buffer=30` frame trước khi loại bỏ.


## 

| Trường hợp kiểm thử | Kết quả |
|---------------------|---------|
| Một người, một khuôn mặt | Hai mắt được phát hiện, mỗi mắt có Track ID riêng |
| Nhiều người (2-3) | Tất cả mắt được gán ID riêng, hoạt động song song |
| Di chuyển đầu nhẹ | Track ID giữ ổn định |
| Di chuyển nhanh | ID có thể thay đổi do IoU giảm |
| Che mắt 1-2 giây | Track bị "lost" tạm thời, khôi phục khi mắt xuất hiện lại |
| Focus mode | Hoạt động đúng, hiển thị riêng mắt được chọn |

## 

| Tiêu chí | Kết quả |
|----------|---------|
| Phát hiện nhiều mắt | Đạt |
| Vị trí lòng đen | Đạt |
| MediaPipe Stream Mode | Đạt |
| ByteTrack | Đạt |
| Real-time | Đạt |
| Ổn định tracking | Khá |

