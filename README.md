# Eye Tracking Application

## 1. `tracking_utils.py`
- Mô hình sử dụng: **MediaPipe Face Mesh** ở chế độ streaming (`static_image_mode=False`, `refine_landmarks=True`) để tính full 468 landmarks và tâm iris (indices 468–477); OpenCV lo phần hậu kỳ (bbox, mesh, vẽ).
- Vai trò: chuẩn bị dữ liệu cho từng khung hình – `get_detections` trả về bbox + landmarks + offset (camera/PNG), `draw_track_on` vẽ mesh/bounding box/iris, nhưng **chưa** gán ID theo thời gian.

## 2. `bytetrack.py`
- Thuật toán theo dõi: **ByteTrack**. Trong `ByteTracker.update`, xây ma trận chi phí `1 - IoU`, giải gán tối ưu bằng `lap.lapjv`, cập nhật `bbox/landmarks/source/offset/image_shape` và bộ đếm `lost` của từng track.
- Nhiệm vụ: duy trì ID ổn định cho mỗi khuôn mặt–một cặp mắt qua các khung hình; detection mới → track mới, track mất quá `track_buffer=30` khung hình sẽ bị loại bỏ.

```powershell
.\venv\Scripts\activate
python eye.py
```