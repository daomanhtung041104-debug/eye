import cv2
import mediapipe as mp
import numpy as np

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

def process_face_mesh(image, face_mesh):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image_out = image.copy()
    h, w = image_out.shape[:2]
    
    face_count = 0
    if results.multi_face_landmarks:
        face_count = len(results.multi_face_landmarks)
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image_out,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=white_connection_spec)
            mp_drawing.draw_landmarks(
                image=image_out,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=white_connection_spec)
            mp_drawing.draw_landmarks(
                image=image_out,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=white_connection_spec)
            
            left_iris = get_iris_position(face_landmarks, LEFT_IRIS, w, h)
            right_iris = get_iris_position(face_landmarks, RIGHT_IRIS, w, h)
            
            if left_iris:
                cv2.circle(image_out, left_iris, 5, (0, 255, 0), -1)
            if right_iris:
                cv2.circle(image_out, right_iris, 5, (255, 0, 255), -1)
    
    cv2.putText(image_out, f'Faces: {face_count}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image_out, face_count

white_connection_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
phone_image = cv2.imread('test.png')

show_phone_image = False
phone_processed_cache = None
phone_resized_cache = None
phone_faces_cache = 0
camera_height_cache = 0

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
        camera_processed, camera_faces = process_face_mesh(camera_img, face_mesh)
        
        key = cv2.waitKey(5) & 0xFF
        
        if key == ord('a'):
            show_phone_image = not show_phone_image
            if show_phone_image and phone_image is not None:
                phone_processed_cache, phone_faces_cache = process_face_mesh(phone_image.copy(), face_mesh)
                phone_resized_cache = None
                print(f"Loaded and processed phone image - Detect {phone_faces_cache} faces")
            elif not show_phone_image:
                print("Phone image turned off")
        
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
        
        cv2.imshow('MediaPipe Face Mesh - Detect & Track 2 Faces', combined)
        
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
