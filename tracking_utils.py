from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
WHITE_SPEC = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)


class _ExponentialSmoother:
    def __init__(self, alpha: float = 0.4):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._prev: Optional[Tuple[float, ...]] = None

    def reset(self) -> None:
        self._prev = None

    def _smooth(self, values: Tuple[float, ...]) -> Tuple[float, ...]:
        if self._prev is None:
            self._prev = values
        else:
            self._prev = tuple(
                self.alpha * current + (1 - self.alpha) * previous
                for current, previous in zip(values, self._prev)
            )
        return self._prev


class BoxSmoother(_ExponentialSmoother):
    def __call__(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        smoothed = self._smooth(tuple(float(v) for v in bbox))
        return tuple(int(round(v)) for v in smoothed)


class PointSmoother(_ExponentialSmoother):
    def __call__(self, point: Tuple[int, int]) -> Tuple[int, int]:
        smoothed = self._smooth(tuple(float(v) for v in point))
        return tuple(int(round(v)) for v in smoothed)


def get_iris_position(landmarks, iris_indices, w: int, h: int) -> Optional[Tuple[int, int]]:
    points = []
    for idx in iris_indices:
        if idx < len(landmarks.landmark):
            x = int(landmarks.landmark[idx].x * w)
            y = int(landmarks.landmark[idx].y * h)
            points.append((x, y))

    if not points:
        return None

    return (
        int(np.mean([p[0] for p in points])),
        int(np.mean([p[1] for p in points])),
    )


def get_face_bbox(landmarks, w: int, h: int) -> Tuple[int, int, int, int]:
    xs = [landmarks.landmark[i].x * w for i in range(len(landmarks.landmark))]
    ys = [landmarks.landmark[i].y * h for i in range(len(landmarks.landmark))]

    x_min = int(min(xs))
    x_max = int(max(xs))
    y_min = int(min(ys))
    y_max = int(max(ys))

    return (x_min, y_min, x_max, y_max)


def get_detections(image, face_mesh, source="camera", offset_x=0, offset_y=0):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    h, w = image.shape[:2]
    detections = []

    if results and results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            x_min, y_min, x_max, y_max = get_face_bbox(face_landmarks, w, h)
            detections.append(
                {
                    "bbox": (x_min + offset_x, y_min + offset_y, x_max + offset_x, y_max + offset_y),
                    "landmarks": face_landmarks,
                    "source": source,
                    "offset": (offset_x, offset_y),
                    "image_shape": (h, w),
                }
            )

    return detections, len(detections)


def draw_track_on(
    image,
    track,
    selected_display_id: Optional[int],
    bbox_smoother: Optional[BoxSmoother] = None,
    iris_smoothers: Optional[Dict[str, PointSmoother]] = None,
):
    offset_x, offset_y = track.get("offset", (0, 0))
    bbox = track["bbox"]

    if bbox_smoother is not None:
        bbox = bbox_smoother(bbox)

    x_min, y_min, x_max, y_max = bbox
    x_min -= offset_x
    x_max -= offset_x
    y_min -= offset_y
    y_max -= offset_y

    h, w = image.shape[:2]
    x_min = max(0, min(int(x_min), w - 1))
    x_max = max(0, min(int(x_max), w - 1))
    y_min = max(0, min(int(y_min), h - 1))
    y_max = max(0, min(int(y_max), h - 1))

    display_id = track.get("display_id")
    is_selected = selected_display_id is not None and display_id == selected_display_id
    color = (0, 255, 0) if is_selected else (255, 0, 0)
    thickness = 3 if is_selected else 2

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    label_y = y_min - 10 if y_min - 10 > 10 else y_min + 20
    cv2.putText(
        image,
        f"ID: {display_id if display_id is not None else '-'}",
        (x_min, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    landmarks = track.get("landmarks")
    highlight_mesh = landmarks and (is_selected or selected_display_id is None)

    if highlight_mesh:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=WHITE_SPEC,
        )
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=WHITE_SPEC,
        )
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=WHITE_SPEC,
        )

        left_iris = get_iris_position(landmarks, LEFT_IRIS, w, h)
        right_iris = get_iris_position(landmarks, RIGHT_IRIS, w, h)

        if left_iris and iris_smoothers and "left" in iris_smoothers:
            left_iris = iris_smoothers["left"](left_iris)
        if left_iris:
            cv2.circle(image, left_iris, 5, (0, 255, 0), -1)

        if right_iris and iris_smoothers and "right" in iris_smoothers:
            right_iris = iris_smoothers["right"](right_iris)
        if right_iris:
            cv2.circle(image, right_iris, 5, (255, 0, 255), -1)

    return image


