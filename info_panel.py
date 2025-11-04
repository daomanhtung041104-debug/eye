import cv2
import numpy as np
from typing import List, Dict, Optional

class InfoPanel:
    def __init__(self):
        self.panel_x = 0
        self.panel_y = 10
        self.panel_width = 250
        self.line_height = 22
        
    def draw(self, frame, eye_data: List[Dict], focus_track_id: Optional[int] = None, 
             img_width: int = 640, img_height: int = 480, 
             tracker=None, debug_info: Dict = None, 
             detections_count: int = 0, matched_count: int = 0):
        panel_x = img_width - 280
        panel_y = 10
        panel_width = 270
        line_height = 22
        
        num_eyes = len(eye_data)
        
        lines_count = 0
        
        lines_count += 1  
        lines_count += 1  
        
        if tracker:
            lines_count += 1 
            lines_count += 1  
            lines_count += 1  
            if detections_count > 0:
                lines_count += 1  
            if matched_count > 0:
                lines_count += 1  
            active_tracks = [t for t in tracker.tracks if t.state == 'tracked']
            if active_tracks:
                lines_count += 1  
        else:
            lines_count += 1  
        
        lines_count += 1  
        
        lines_count += 1  
        lines_count += 1  
        lines_count += num_eyes  
        
        total_height = lines_count * line_height + 40 
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + total_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + total_height), (150, 150, 150), 1)
        
        y_pos = panel_y + 25
        
        cv2.putText(frame, "=== DEBUG INFO ===", (panel_x + 10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y_pos += line_height
        cv2.putText(frame, "Stream Mode: TRUE", (panel_x + 10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y_pos += line_height
        
        if tracker:
            active_tracks = [t for t in tracker.tracks if t.state == 'tracked']
            active_count = len(active_tracks)
            lost_count = len([t for t in tracker.tracks if t.state == 'lost'])
            total_tracks = len(tracker.tracks)
            
            cv2.putText(frame, f"ByteTrack: ACTIVE", (panel_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_pos += line_height
            cv2.putText(frame, f"Total Tracks: {total_tracks}", (panel_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += line_height
            cv2.putText(frame, f"Active: {active_count} | Lost: {lost_count}", (panel_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += line_height
            
            # ByteTrack process details
            if detections_count > 0:
                cv2.putText(frame, f"Detections: {detections_count}", (panel_x + 10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
                y_pos += line_height
            if matched_count > 0:
                cv2.putText(frame, f"Matched: {matched_count}", (panel_x + 10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
                y_pos += line_height
            
            # Show all track IDs
            if active_tracks:
                track_ids_str = ",".join([str(t.track_id) for t in active_tracks[:5]])
                if len(active_tracks) > 5:
                    track_ids_str += "..."
                cv2.putText(frame, f"Track IDs: [{track_ids_str}]", (panel_x + 10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1)
                y_pos += line_height
        else:
            cv2.putText(frame, "ByteTrack: OFF", (panel_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 100), 1)
            y_pos += line_height
        
        if focus_track_id is not None:
            cv2.putText(frame, f"FOCUS: Track {focus_track_id}", (panel_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "FOCUS: ALL (None)", (panel_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        y_pos += line_height + 5
        
        cv2.putText(frame, "=== EYE DATA ===", (panel_x + 10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y_pos += line_height
        cv2.putText(frame, f"Eyes: {num_eyes}", (panel_x + 10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_pos += line_height + 5
        
        for eye in eye_data:
            text_color = (0, 255, 0) if focus_track_id == eye['track_id'] else (220, 220, 220)
            focus_mark = " [FOCUS]" if focus_track_id == eye['track_id'] else ""
            label = f"[{eye['track_id']}] {eye['eye_type'][0].upper()}  X:{eye['iris_x']}  Y:{eye['iris_y']}{focus_mark}"
            cv2.putText(frame, label, (panel_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            y_pos += line_height
        
        return frame

