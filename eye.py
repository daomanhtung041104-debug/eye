import cv2
import mediapipe as mp
import numpy as np
from bytetrack import ByteTracker

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

def get_iris_position(landmarks, iris_indices, w, h):
    points = []
    for idx in iris_indices:
        if idx < len(landmarks.landmark):
            x = int(landmarks.landmark[idx].x * w)
            y = int(landmarks.landmark[idx].y * h)
            points.append((x, y))
    
    if not points:
        return None
    
    return (int(np.mean([p[0] for p in points])), int(np.mean([p[1] for p in points])))

def get_face_bbox(landmarks, w, h):
    xs = [landmarks.landmark[i].x * w for i in range(len(landmarks.landmark))]
    ys = [landmarks.landmark[i].y * h for i in range(len(landmarks.landmark))]
    
    x_min = int(min(xs))
    x_max = int(max(xs))
    y_min = int(min(ys))
    y_max = int(max(ys))
    
    return (x_min, y_min, x_max, y_max)

def process_face_mesh(image, face_mesh, tracker=None, selected_face_id=None):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image_out = image.copy()
    h, w = image_out.shape[:2]
    
    detections = []
    face_count = 0
    
    if results.multi_face_landmarks:
        face_count = len(results.multi_face_landmarks)
        for face_landmarks in results.multi_face_landmarks:
            bbox = get_face_bbox(face_landmarks, w, h)
            detections.append({
                'bbox': bbox,
                'landmarks': face_landmarks
            })
    
    tracked_faces = []
    if tracker is not None:
        tracked_faces = tracker.update(detections)
    elif len(detections) > 0:
        for i, det in enumerate(detections):
            tracked_faces.append({
                'id': i + 1,
                'bbox': det['bbox'],
                'landmarks': det['landmarks']
            })
    
    for track in tracked_faces:
        track_id = track['id']
        bbox = track['bbox']
        x_min, y_min, x_max, y_max = bbox
        landmarks = track.get('landmarks')
        
        is_selected = (selected_face_id is not None and track_id == selected_face_id)
        color = (0, 255, 0) if is_selected else (255, 0, 0)
        thickness = 3 if is_selected else 2
        
        cv2.rectangle(image_out, (x_min, y_min), (x_max, y_max), color, thickness)
        cv2.putText(image_out, f'ID: {track_id}', (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if landmarks:
            if is_selected or selected_face_id is None:
                mp_drawing.draw_landmarks(
                    image=image_out,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=white_connection_spec)
                mp_drawing.draw_landmarks(
                    image=image_out,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=white_connection_spec)
                mp_drawing.draw_landmarks(
                    image=image_out,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=white_connection_spec)
                
                left_iris = get_iris_position(landmarks, LEFT_IRIS, w, h)
                right_iris = get_iris_position(landmarks, RIGHT_IRIS, w, h)
                
                if left_iris:
                    cv2.circle(image_out, left_iris, 5, (0, 255, 0), -1)
                if right_iris:
                    cv2.circle(image_out, right_iris, 5, (255, 0, 255), -1)
    
    cv2.putText(image_out, f'Faces: {face_count} | Tracks: {len(tracked_faces)}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if selected_face_id is not None:
        cv2.putText(image_out, f'Selected Face ID: {selected_face_id} | Press 0 to deselect', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(image_out, 'Press 1-9 to select face', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image_out, face_count, tracked_faces

white_connection_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
phone_image = cv2.imread('test.png')

show_phone_image = False
phone_processed_cache = None
phone_resized_cache = None
phone_faces_cache = 0
camera_height_cache = 0

tracker = ByteTracker(match_thresh=0.5, track_buffer=30)
selected_face_id = None

with mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, camera_img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        camera_img = cv2.flip(camera_img, 1)
        camera_processed, camera_faces, camera_tracks = process_face_mesh(
            camera_img, face_mesh, tracker, selected_face_id)
        
        key = cv2.waitKey(5) & 0xFF
        
        if key == ord('a'):
            show_phone_image = not show_phone_image
            if show_phone_image and phone_image is not None:
                phone_processed_cache, phone_faces_cache, _ = process_face_mesh(
                    phone_image.copy(), face_mesh, None, None)
                phone_resized_cache = None
                print(f"Loaded and processed phone image - Detect {phone_faces_cache} faces")
            elif not show_phone_image:
                print("Phone image turned off")
        
        if key >= ord('1') and key <= ord('9'):
            face_num = key - ord('0')
            if camera_tracks and face_num <= len(camera_tracks):
                selected_face_id = camera_tracks[face_num - 1]['id']
                print(f"Selected face ID: {selected_face_id}")
            else:
                print(f"No face at position {face_num}")
        
        if key == ord('0'):
            selected_face_id = None
            print("Deselected face - Showing all faces")
        
        if show_phone_image and phone_processed_cache is not None:
            h1, w1 = camera_processed.shape[:2]
            
            if phone_resized_cache is None or camera_height_cache != h1:
                h2, w2 = phone_processed_cache.shape[:2]
                scale = h1 / h2
                phone_resized_cache = cv2.resize(phone_processed_cache, (int(w2 * scale), h1))
                camera_height_cache = h1
            
            combined = np.hstack([camera_processed, phone_resized_cache])
            total_faces = camera_faces + phone_faces_cache
            cv2.putText(combined, f'Total: {total_faces} (Camera: {camera_faces}, Phone: {phone_faces_cache})', 
                       (10, h1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            combined = camera_processed
        
        cv2.imshow('MediaPipe Face Mesh - ByteTrack', combined)
        
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
