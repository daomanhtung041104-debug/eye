import numpy as np
from collections import deque

class ByteTracker:
    def __init__(self, match_thresh=0.5, track_buffer=30, frame_rate=30):
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_id = 0
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.next_id = 1
        
    def iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def linear_assignment(self, cost_matrix):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        
        try:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=1e10)
            matches = [[i, j] for i, j in zip(x, y) if i >= 0 and j >= 0]
            unmatched_a = [i for i, j in enumerate(x) if j < 0]
            unmatched_b = [i for i, j in enumerate(y) if j < 0]
            return matches, unmatched_a, unmatched_b
        except ImportError:
            matches = []
            unmatched_a = list(range(cost_matrix.shape[0]))
            unmatched_b = list(range(cost_matrix.shape[1]))
            
            for i in range(cost_matrix.shape[0]):
                if i not in [m[0] for m in matches]:
                    best_j = -1
                    best_iou = self.match_thresh
                    for j in range(cost_matrix.shape[1]):
                        if j not in [m[1] for m in matches]:
                            iou = 1 - cost_matrix[i, j]
                            if iou > best_iou:
                                best_iou = iou
                                best_j = j
                    if best_j != -1:
                        matches.append([i, best_j])
                        unmatched_a.remove(i)
                        unmatched_b.remove(best_j)
            
            return matches, unmatched_a, unmatched_b
    
    def update(self, detections):
        self.frame_id += 1
        
        if len(detections) == 0:
            for track in self.tracked_tracks:
                track['lost'] += 1
                if track['lost'] > self.track_buffer:
                    self.removed_tracks.append(track)
            self.tracked_tracks = [t for t in self.tracked_tracks if t['lost'] <= self.track_buffer]
            return self.tracked_tracks
        
        tracked_boxes = [t['bbox'] for t in self.tracked_tracks]
        det_boxes = [d['bbox'] for d in detections]
        
        if len(tracked_boxes) > 0 and len(det_boxes) > 0:
            cost_matrix = np.zeros((len(tracked_boxes), len(det_boxes)))
            for i, tb in enumerate(tracked_boxes):
                for j, db in enumerate(det_boxes):
                    cost_matrix[i, j] = 1 - self.iou(tb, db)
            
            matches, unmatched_tracks, unmatched_dets = self.linear_assignment(cost_matrix)
            
            for match in matches:
                track_idx, det_idx = match
                self.tracked_tracks[track_idx]['bbox'] = detections[det_idx]['bbox']
                self.tracked_tracks[track_idx]['lost'] = 0
                self.tracked_tracks[track_idx]['landmarks'] = detections[det_idx].get('landmarks', None)
        else:
            unmatched_tracks = list(range(len(tracked_boxes)))
            unmatched_dets = list(range(len(det_boxes)))
        
        for i in unmatched_tracks:
            self.tracked_tracks[i]['lost'] += 1
        
        for i in unmatched_dets:
            new_track = {
                'id': self.next_id,
                'bbox': detections[i]['bbox'],
                'lost': 0,
                'landmarks': detections[i].get('landmarks', None)
            }
            self.next_id += 1
            self.tracked_tracks.append(new_track)
        
        self.tracked_tracks = [t for t in self.tracked_tracks if t['lost'] <= self.track_buffer]
        
        active_tracks = [t for t in self.tracked_tracks if t['lost'] == 0]
        for idx, track in enumerate(active_tracks, 1):
            track['id'] = idx
        
        return self.tracked_tracks

