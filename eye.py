from collections import defaultdict

import cv2
import mediapipe as mp
import numpy as np
from bytetrack import ByteTracker
from tracking_utils import BoxSmoother, PointSmoother, draw_track_on, get_detections

mp_face_mesh = mp.solutions.face_mesh

white_connection_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

tracker = ByteTracker(match_thresh=0.7, track_buffer=15)
selected_display_id = None

bbox_smoothers = defaultdict(lambda: BoxSmoother(alpha=0.35))
left_iris_smoothers = defaultdict(lambda: PointSmoother(alpha=0.5))
right_iris_smoothers = defaultdict(lambda: PointSmoother(alpha=0.5))

with mp_face_mesh.FaceMesh(
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as face_mesh:
    
    while cap.isOpened():
        success, camera_img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        camera_img = cv2.flip(camera_img, 1)
        camera_out = camera_img.copy()

        camera_detections, camera_face_count = get_detections(
            camera_out, face_mesh, source='camera', offset_x=0, offset_y=0)

        tracked_faces = tracker.update(camera_detections)
        active_tracks = [track for track in tracked_faces if track['lost'] == 0]
        ordered_tracks = sorted(active_tracks, key=lambda t: t['bbox'][0])
        for i, track in enumerate(ordered_tracks):
            track['display_id'] = i + 1

        current_ids = {track['id'] for track in ordered_tracks}
        for smoother_map in (bbox_smoothers, left_iris_smoothers, right_iris_smoothers):
            for track_id in list(smoother_map.keys()):
                if track_id not in current_ids:
                    smoother_map[track_id].reset()
                    del smoother_map[track_id]

        for track in ordered_tracks:
            track_id = track['id']
            draw_track_on(
                camera_out,
                track,
                selected_display_id,
                bbox_smoother=bbox_smoothers[track_id],
                iris_smoothers={
                    'left': left_iris_smoothers[track_id],
                    'right': right_iris_smoothers[track_id],
                },
            )

        combined = camera_out

        total_faces = len(ordered_tracks)
        summary_text = f'Total Faces: {total_faces} (Camera: {camera_face_count})'
        cv2.putText(combined, summary_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if selected_display_id is not None:
            cv2.putText(combined, f'Selected Face ID: {selected_display_id} | Press 0 to deselect', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(combined, "Press 1-9 to select face", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('MediaPipe Face Mesh - ByteTrack', combined)

        key = cv2.waitKey(5) & 0xFF

        if key >= ord('1') and key <= ord('9'):
            index = key - ord('1')
            if index < len(ordered_tracks):
                selected_display_id = ordered_tracks[index]['display_id']
                print(f"Selected face ID: {selected_display_id}")
            else:
                print(f"No face at position {index + 1}")

        if key == ord('0'):
            selected_display_id = None
            print("Deselected face - Showing all faces")

        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()