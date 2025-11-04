
import numpy as np
from typing import List, Tuple, Optional
from collections import deque

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class KalmanTracker:
    def __init__(self, bbox: np.ndarray):
        self.bbox = bbox.astype(np.float32) 
        
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  
        r = (bbox[2] - bbox[0]) / max(bbox[3] - bbox[1], 1e-6) 
        
        self.mean = np.array([cx, cy, s, r], dtype=np.float32)
        self.velocity = np.zeros(4, dtype=np.float32)
        
    def predict(self):
        self.mean[:4] += self.velocity
        
        cx, cy, s, r = self.mean
        w = np.sqrt(s * r)
        h = s / max(w, 1e-6)
        self.bbox = np.array([
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2
        ], dtype=np.float32)
        return self.bbox.copy()
    
    def update(self, bbox: np.ndarray):
        old_bbox = self.bbox.copy()
        
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        r = (bbox[2] - bbox[0]) / max(bbox[3] - bbox[1], 1e-6)
        
        new_state = np.array([cx, cy, s, r], dtype=np.float32)
        self.velocity = 0.9 * self.velocity + 0.1 * (new_state - self.mean)
        
        self.mean = new_state
        self.bbox = bbox.astype(np.float32)


class Track:
    def __init__(self, bbox: np.ndarray, track_id: int, frame_id: int, score: float = 1.0):
        self.track_id = track_id
        self.bbox = bbox.astype(np.float32)
        self.score = score
        self.frame_id = frame_id
        self.state = 'tracked'  # 'tracked', 'lost', 'removed'
        self.time_since_update = 0
        self.history = deque(maxlen=30)  
        self.history.append(bbox.copy())
        
        self.kalman = KalmanTracker(bbox)
        
    def update(self, bbox: np.ndarray, frame_id: int, score: float = 1.0):
        self.bbox = bbox.astype(np.float32)
        self.score = score
        self.frame_id = frame_id
        self.time_since_update = 0
        self.history.append(bbox.copy())
        self.state = 'tracked'
        
        self.kalman.update(bbox)
        
    def predict(self):
        self.bbox = self.kalman.predict()
        self.time_since_update += 1
        
    def mark_lost(self):
        if self.state == 'tracked':
            self.state = 'lost'
        self.time_since_update += 1
        
    def mark_removed(self):
        self.state = 'removed'


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
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
    
    if union <= 0:
        return 0.0
    
    return intersection / union


class Bytetrack:
    def __init__(self, 
                 track_thresh: float = 0.5,
                 high_thresh: float = 0.6,
                 match_thresh: float = 0.8,
                 frame_rate: int = 30,
                 track_buffer: int = 30):
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.track_buffer = track_buffer
        
        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []  
        self.frame_count = 0
        self.next_id = 1
        
    def update(self, detections: np.ndarray) -> List[Track]:

        self.frame_count += 1
        
        if len(detections) == 0:
            for track in self.tracks:
                if track.state == 'tracked':
                    track.predict()
                    track.mark_lost()
            return []
        
        high_score_dets = detections[detections[:, 4] >= self.high_thresh]
        low_score_dets = detections[detections[:, 4] < self.high_thresh]
        
        tracked_tracks = [t for t in self.tracks if t.state == 'tracked']
        for track in tracked_tracks:
            track.predict()
        
        matched, unmatched_tracks, unmatched_dets_high = self._match(
            tracked_tracks, high_score_dets, self.match_thresh
        )
        
        for track_idx, det_idx in matched:
            tracked_tracks[track_idx].update(
                high_score_dets[det_idx, :4],
                self.frame_count,
                high_score_dets[det_idx, 4]
            )
        
        unmatched_track_objects = [tracked_tracks[i] for i in unmatched_tracks]
        if len(unmatched_track_objects) > 0 and len(low_score_dets) > 0:
            rematched, rematched_unmatched_tracks, rematched_unmatched_dets = self._match(
                unmatched_track_objects, low_score_dets, 0.5  # Lower threshold cho low-score
            )
            
            for local_track_idx, det_idx in rematched:
                original_track_idx = unmatched_tracks[local_track_idx]
                tracked_tracks[original_track_idx].update(
                    low_score_dets[det_idx, :4],
                    self.frame_count,
                    low_score_dets[det_idx, 4]
                )
            
            rematched_track_indices = {unmatched_tracks[local_idx] for local_idx, _ in rematched}
            unmatched_tracks = [i for i in unmatched_tracks if i not in rematched_track_indices]
            unmatched_dets_high = list(set(range(len(high_score_dets))) - 
                                      {det_idx for _, det_idx in matched})
        
        for track_idx in unmatched_tracks:
            tracked_tracks[track_idx].mark_lost()
        
        for det_idx in unmatched_dets_high:
            if high_score_dets[det_idx, 4] >= self.track_thresh:
                new_track = Track(
                    high_score_dets[det_idx, :4],
                    self.next_id,
                    self.frame_count,
                    high_score_dets[det_idx, 4]
                )
                self.tracks.append(new_track)
                self.next_id += 1
        
        if len(self.lost_tracks) > 0:
            for track in self.lost_tracks:
                track.predict()
            
            rebirth_matched, rebirth_unmatched_tracks, _ = self._match(
                self.lost_tracks, high_score_dets, self.match_thresh
            )
            
            for track_idx, det_idx in rebirth_matched:
                self.lost_tracks[track_idx].update(
                    high_score_dets[det_idx, :4],
                    self.frame_count,
                    high_score_dets[det_idx, 4]
                )
                self.tracks.append(self.lost_tracks[track_idx])
            
            self.lost_tracks = [self.lost_tracks[i] for i in rebirth_unmatched_tracks]
        
        for track in self.tracks:
            if track.state == 'lost':
                self.lost_tracks.append(track)
        
        self.tracks = [t for t in self.tracks if t.state == 'tracked']
        
        self.tracks = [
            track for track in self.tracks 
            if track.time_since_update <= self.track_buffer
        ]
        self.lost_tracks = [
            track for track in self.lost_tracks
            if track.time_since_update <= self.track_buffer
        ]
        
        return [track for track in self.tracks if track.state == 'tracked']
    
    def _match(self, tracks: List[Track], detections: np.ndarray, 
               threshold: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], list(range(len(tracks))), []
        
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = iou(track.bbox, det[:4])
        
        if SCIPY_AVAILABLE:
            cost_matrix = 1.0 - iou_matrix  
            matched_indices = linear_sum_assignment(cost_matrix)
            
            matched = []
            unmatched_tracks = list(range(len(tracks)))
            unmatched_dets = list(range(len(detections)))
            
            for track_idx, det_idx in zip(matched_indices[0], matched_indices[1]):
                if iou_matrix[track_idx, det_idx] >= threshold:
                    matched.append((track_idx, det_idx))
                    unmatched_tracks.remove(track_idx)
                    unmatched_dets.remove(det_idx)
            
            return matched, unmatched_tracks, unmatched_dets
        else:
            matched = []
            unmatched_tracks = list(range(len(tracks)))
            unmatched_dets = list(range(len(detections)))
            
            while True:
                max_iou = -1
                best_track = -1
                best_det = -1
                
                for i in unmatched_tracks:
                    for j in unmatched_dets:
                        if iou_matrix[i, j] > max_iou and iou_matrix[i, j] >= threshold:
                            max_iou = iou_matrix[i, j]
                            best_track = i
                            best_det = j
                
                if max_iou < threshold:
                    break
                
                matched.append((best_track, best_det))
                unmatched_tracks.remove(best_track)
                unmatched_dets.remove(best_det)
            
            return matched, unmatched_tracks, unmatched_dets

