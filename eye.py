import cv2
import mediapipe as mp
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from bytetrack import Bytetrack
from info_panel import InfoPanel

class EyeTracker:
    def __init__(self, use_byte_track: bool = True, focus_track_id: Optional[int] = None):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.use_byte_track = use_byte_track
        self.focus_track_id = focus_track_id
        self.tracker = Bytetrack(track_thresh=0.5, high_thresh=0.6, match_thresh=0.5, 
                                 frame_rate=30, track_buffer=30) if use_byte_track else None
        
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        self.fps_counter = []
        self.prev_time = time.time()
        self.info_panel = InfoPanel()
        
    def get_eye_bbox(self, landmarks, eye_indices: List[int], 
                     img_width: int, img_height: int) -> Optional[np.ndarray]:
        if landmarks is None or not hasattr(landmarks, 'landmark') or len(landmarks.landmark) == 0:
            return None
        
        valid_points = [(landmarks.landmark[idx].x * img_width, landmarks.landmark[idx].y * img_height)
                       for idx in eye_indices if idx < len(landmarks.landmark)]
        
        if len(valid_points) == 0:
            return None
        
        xs, ys = zip(*valid_points)
        margin = 15
        return np.array([
            max(0, int(min(xs) - margin)),
            max(0, int(min(ys) - margin)),
            min(img_width, int(max(xs) + margin)),
            min(img_height, int(max(ys) + margin))
        ])
    
    def get_iris_center(self, landmarks, iris_indices: List[int], 
                       img_width: int, img_height: int) -> Optional[Tuple[int, int]]:
        if landmarks is None or not hasattr(landmarks, 'landmark') or len(landmarks.landmark) == 0:
            return None
        
        iris_points = [(int(landmarks.landmark[idx].x * img_width), 
                        int(landmarks.landmark[idx].y * img_height))
                       for idx in iris_indices if idx < len(landmarks.landmark)]
        
        if len(iris_points) == 0:
            return None
        
        return (int(np.mean([p[0] for p in iris_points])), 
                int(np.mean([p[1] for p in iris_points])))
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        
        h, w = frame.shape[:2]
        track_id_to_eye_info, eye_data, tracking_stats = self._track_eyes(results, w, h)
        
        if self.focus_track_id and self.use_byte_track and self.tracker:
            active_tracks = [t for t in self.tracker.tracks if t.state == 'tracked']
            active_ids = [t.track_id for t in active_tracks]
            if self.focus_track_id not in active_ids and active_ids:
                self.focus_track_id = active_ids[0]  
        
        self._draw_all_landmarks(frame, results, w, h)
        
        self._draw_eyes(frame, results, track_id_to_eye_info, eye_data, w, h)
        
        debug_info = {
            'stream_mode': True,
            'byte_track_active': self.use_byte_track
        }
        return self.info_panel.draw(frame, eye_data, self.focus_track_id, w, h, 
                                   self.tracker, debug_info,
                                   tracking_stats.get('detections_count', 0),
                                   tracking_stats.get('matched_count', 0))
    
    def _track_eyes(self, results, w: int, h: int) -> Tuple[Dict, List, Dict]:

        track_id_to_eye_info = {}
        eye_data = []
        tracking_stats = {'detections_count': 0, 'matched_count': 0}
        
        if not results.multi_face_landmarks:
            return track_id_to_eye_info, eye_data, tracking_stats
        
        if self.use_byte_track:
            detections, detections_info = [], []
            for face_landmarks in results.multi_face_landmarks:
                for eye_type, eye_indices in [('left', self.LEFT_EYE), ('right', self.RIGHT_EYE)]:
                    eye_bbox = self.get_eye_bbox(face_landmarks, eye_indices, w, h)
                    if eye_bbox is not None and len(eye_bbox) == 4:
                        detections.append([float(eye_bbox[0]), float(eye_bbox[1]), 
                                          float(eye_bbox[2]), float(eye_bbox[3]), 0.9])
                        detections_info.append((eye_bbox.copy(), face_landmarks, eye_type))
            
            tracking_stats['detections_count'] = len(detections)
            
            if detections:
                active_tracks = self.tracker.update(np.array(detections, dtype=np.float32))
                
                matched = 0
                for track in active_tracks:
                    best_match = max(enumerate(detections_info), 
                                    key=lambda x: self._calculate_iou(track.bbox, x[1][0]),
                                    default=(None, None))
                    if best_match[0] is not None and self._calculate_iou(track.bbox, best_match[1][0]) > 0.3:
                        _, face_landmarks, eye_type = best_match[1]
                        track_id_to_eye_info[track.track_id] = (face_landmarks, eye_type)
                        matched += 1
                
                tracking_stats['matched_count'] = matched
        
        eye_info_to_track_id = {(id(landmarks), eye_type): track_id 
                                for track_id, (landmarks, eye_type) in track_id_to_eye_info.items()}
        
        for face_landmarks in results.multi_face_landmarks:
            for eye_type, eye_indices, iris_indices in [
                ('left', self.LEFT_EYE, self.LEFT_IRIS),
                ('right', self.RIGHT_EYE, self.RIGHT_IRIS)
            ]:
                track_id = eye_info_to_track_id.get((id(face_landmarks), eye_type))
                if self.use_byte_track and not track_id:
                    continue
                if self.focus_track_id and track_id != self.focus_track_id:
                    continue
                
                eye_bbox = self.get_eye_bbox(face_landmarks, eye_indices, w, h)
                iris_center = self.get_iris_center(face_landmarks, iris_indices, w, h)
                
                if eye_bbox is None or iris_center is None:
                    continue
                
                if track_id:
                    eye_data.append({
                        'track_id': track_id,
                        'eye_type': eye_type,
                        'iris_x': iris_center[0],
                        'iris_y': iris_center[1]
                    })
        
        return track_id_to_eye_info, eye_data, tracking_stats
    
    def _draw_all_landmarks(self, frame, results, w: int, h: int):
        if not results.multi_face_landmarks:
            return
        
        for face_landmarks in results.multi_face_landmarks:
            if face_landmarks and face_landmarks.landmark:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (100, 200, 255), -1)
                
                # Face contour
                face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                for i in range(len(face_oval) - 1):
                    pt1 = (int(face_landmarks.landmark[face_oval[i]].x * w),
                           int(face_landmarks.landmark[face_oval[i]].y * h))
                    pt2 = (int(face_landmarks.landmark[face_oval[i+1]].x * w),
                           int(face_landmarks.landmark[face_oval[i+1]].y * h))
                    cv2.line(frame, pt1, pt2, (200, 200, 200), 1)
                
                # Left eyebrow
                left_eyebrow = [107, 55, 65, 52, 53, 46]
                for i in range(len(left_eyebrow) - 1):
                    pt1 = (int(face_landmarks.landmark[left_eyebrow[i]].x * w),
                           int(face_landmarks.landmark[left_eyebrow[i]].y * h))
                    pt2 = (int(face_landmarks.landmark[left_eyebrow[i+1]].x * w),
                           int(face_landmarks.landmark[left_eyebrow[i+1]].y * h))
                    cv2.line(frame, pt1, pt2, (150, 150, 150), 1)
                
                # Right eyebrow
                right_eyebrow = [336, 285, 295, 282, 283, 276]
                for i in range(len(right_eyebrow) - 1):
                    pt1 = (int(face_landmarks.landmark[right_eyebrow[i]].x * w),
                           int(face_landmarks.landmark[right_eyebrow[i]].y * h))
                    pt2 = (int(face_landmarks.landmark[right_eyebrow[i+1]].x * w),
                           int(face_landmarks.landmark[right_eyebrow[i+1]].y * h))
                    cv2.line(frame, pt1, pt2, (150, 150, 150), 1)
                
                # Nose
                nose = [6, 98, 327, 2, 328, 326, 2, 97, 326, 2, 4, 1, 4, 2]
                for i in range(len(nose) - 1):
                    if nose[i] < len(face_landmarks.landmark) and nose[i+1] < len(face_landmarks.landmark):
                        pt1 = (int(face_landmarks.landmark[nose[i]].x * w),
                               int(face_landmarks.landmark[nose[i]].y * h))
                        pt2 = (int(face_landmarks.landmark[nose[i+1]].x * w),
                               int(face_landmarks.landmark[nose[i+1]].y * h))
                        cv2.line(frame, pt1, pt2, (150, 150, 150), 1)
                
                # Mouth
                mouth = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
                for i in range(len(mouth) - 1):
                    if mouth[i] < len(face_landmarks.landmark) and mouth[i+1] < len(face_landmarks.landmark):
                        pt1 = (int(face_landmarks.landmark[mouth[i]].x * w),
                               int(face_landmarks.landmark[mouth[i]].y * h))
                        pt2 = (int(face_landmarks.landmark[mouth[i+1]].x * w),
                               int(face_landmarks.landmark[mouth[i+1]].y * h))
                        cv2.line(frame, pt1, pt2, (150, 150, 150), 1)
    
    def _draw_eyes(self, frame, results, track_id_to_eye_info: Dict, eye_data: List, w: int, h: int):
        if not results.multi_face_landmarks:
            return
        
        eye_info_to_track_id = {(id(landmarks), eye_type): track_id 
                                for track_id, (landmarks, eye_type) in track_id_to_eye_info.items()}
        
        for face_landmarks in results.multi_face_landmarks:
            for eye_type, eye_indices, iris_indices in [
                ('left', self.LEFT_EYE, self.LEFT_IRIS),
                ('right', self.RIGHT_EYE, self.RIGHT_IRIS)
            ]:
                track_id = eye_info_to_track_id.get((id(face_landmarks), eye_type))
                if self.use_byte_track and not track_id:
                    continue
                if self.focus_track_id and track_id != self.focus_track_id:
                    continue
                
                eye_bbox = self.get_eye_bbox(face_landmarks, eye_indices, w, h)
                iris_center = self.get_iris_center(face_landmarks, iris_indices, w, h)
                
                if eye_bbox is None:
                    continue
                
                if track_id:
                    color = (0, 255, 0) if self.focus_track_id == track_id else (200, 200, 200)
                    cv2.rectangle(frame, tuple(eye_bbox[:2].astype(int)), 
                                tuple(eye_bbox[2:].astype(int)), color, 2)
                    
                    label = f"ID:{track_id} {eye_type[0].upper()}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (int(eye_bbox[0]), int(eye_bbox[1]) - text_h - 5),
                                (int(eye_bbox[0]) + text_w, int(eye_bbox[1])), (40, 40, 40), -1)
                    cv2.putText(frame, label, (int(eye_bbox[0]), int(eye_bbox[1]) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if iris_center:
                    iris_color = (255, 200, 0) if eye_type == 'left' else (255, 100, 255)
                    cv2.circle(frame, iris_center, 6, iris_color, 2)
                    cv2.circle(frame, iris_center, 2, iris_color, -1)
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _update_fps(self):
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time) if current_time - self.prev_time > 0 else 0
        self.prev_time = current_time
        self.fps_counter.append(fps)
        if len(self.fps_counter) > 30:
            self.fps_counter.pop(0)
        return int(sum(self.fps_counter) / len(self.fps_counter)) if self.fps_counter else 0
    
    def _handle_keypress(self, key: int):
        if key == ord('q'):
            return False
        elif key == ord('s'):
            cv2.imwrite(f'frame_{int(time.time())}.jpg', self.last_frame)
            print(f"Đã lưu frame")
        elif key == ord('f') and self.use_byte_track:
            active_tracks = [t for t in self.tracker.tracks if t.state == 'tracked']
            if active_tracks:
                track_ids = [t.track_id for t in active_tracks]
                if self.focus_track_id and self.focus_track_id not in track_ids:
                    self.focus_track_id = track_ids[0]
                else:
                    try:
                        user_input = input("Nhập track ID để focus (Enter = tự động): ").strip()
                        if user_input == "":
                            self.focus_track_id = track_ids[0]
                        else:
                            track_id = int(user_input)
                            if any(t.track_id == track_id for t in active_tracks):
                                self.focus_track_id = track_id
                    except (ValueError, EOFError):
                        if track_ids:
                            self.focus_track_id = track_ids[0]
        elif key == ord('r') and self.use_byte_track:
            self.focus_track_id = None
            print("Reset focus")
        return True
    
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở webcam!")
            return
        
        print("Eye Tracking - Nhấn 'q' để thoát, 's' để lưu, 'f' để focus, 'r' để reset")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame)
            self.last_frame = processed_frame
            
            fps = self._update_fps()
            cv2.putText(processed_frame, f"FPS: {fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            cv2.imshow('Eye Tracking', processed_frame)
            
            if not self._handle_keypress(cv2.waitKey(1) & 0xFF):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Đã thoát!")


def main():
    tracker = EyeTracker()
    tracker.run()


if __name__ == "__main__":
    main()

