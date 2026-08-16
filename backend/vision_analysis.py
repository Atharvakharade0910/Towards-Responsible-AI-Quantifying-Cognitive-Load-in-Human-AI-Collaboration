from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any


LEFT_EAR_POINTS = (362, 385, 387, 263, 373, 380)
RIGHT_EAR_POINTS = (33, 160, 158, 133, 153, 144)
LEFT_IRIS = (474, 475, 476, 477)
RIGHT_IRIS = (469, 470, 471, 472)
HEAD_POSE_LANDMARKS = (1, 152, 226, 446, 57, 287)


@dataclass
class VisionState:
    calibrated_left: Any = None
    calibrated_right: Any = None
    calibration_left: dict[int, list[Any]] = field(default_factory=dict)
    calibration_right: dict[int, list[Any]] = field(default_factory=dict)
    ear_history: deque[float] = field(default_factory=lambda: deque(maxlen=600))
    calibration_ears: list[float] = field(default_factory=list)
    ear_threshold: float = 0.21
    blink_count: int = 0
    eyes_closed: bool = False
    closed_frames: int = 0
    closed_started_at: float = 0.0
    last_blink_latency_ms: float = 0.0
    closure_history: deque[tuple[float, bool]] = field(default_factory=deque)
    valid_sample_count: int = 0
    last_gaze: Any = None
    last_gaze_at: float = 0.0
    directions: deque[str] = field(default_factory=lambda: deque(maxlen=5))


class VisionAnalyzer:
    """Web-frame adapter for the reusable eye-tracking algorithms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._states: dict[str, VisionState] = {}
        self._face_mesh = None

    @staticmethod
    def _decode(data_url: str):
        import cv2
        import numpy as np

        encoded = data_url.split(",", 1)[-1]
        raw = base64.b64decode(encoded, validate=True)
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid camera frame")
        return frame

    def _mesh(self):
        if self._face_mesh is None:
            import mediapipe as mp

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        return self._face_mesh

    def forget_participant(self, participant_id: str) -> None:
        """Release calibration and rolling eye state after a session ends."""
        with self._lock:
            self._states.pop(participant_id, None)

    @staticmethod
    def _point(landmarks, index: int, width: int, height: int):
        import numpy as np

        item = landmarks[index]
        return np.array((item.x * width, item.y * height), dtype=float)

    @classmethod
    def _ear(cls, landmarks, points, width: int, height: int) -> float:
        import numpy as np

        p1, p2, p3, p4, p5, p6 = [cls._point(landmarks, i, width, height) for i in points]
        return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4) + 1e-6))

    @classmethod
    def _iris(cls, landmarks, points, width: int, height: int):
        import numpy as np

        return np.mean([cls._point(landmarks, i, width, height) for i in points], axis=0)

    @staticmethod
    def _direction(vector) -> str:
        dx, dy = vector
        horizontal = "Left" if dx < -0.018 else "Right" if dx > 0.018 else "Centre"
        vertical = "Up" if dy < -0.018 else "Down" if dy > 0.018 else "Centre"
        if horizontal == "Centre":
            return vertical
        if vertical == "Centre":
            return horizontal
        return f"{vertical}-{horizontal}"

    @classmethod
    def _head_pose(cls, landmarks, width: int, height: int) -> tuple[float, float]:
        import cv2
        import numpy as np

        model_points = np.array(
            ((0, 0, 0), (0, -330, -65), (-225, 170, -135),
             (225, 170, -135), (-150, -150, -125), (150, -150, -125)),
            dtype=np.float64,
        )
        image_points = np.array(
            [cls._point(landmarks, index, width, height) for index in HEAD_POSE_LANDMARKS],
            dtype=np.float64,
        )
        camera = np.array(
            ((width, 0, width / 2), (0, width, height / 2), (0, 0, 1)),
            dtype=np.float64,
        )
        solved, rotation_vector, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not solved:
            return 0.0, 0.0
        rotation, _ = cv2.Rodrigues(rotation_vector)
        sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
        pitch = math.degrees(math.atan2(-rotation[2, 0], sy))
        yaw = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        return pitch, yaw

    @staticmethod
    def _update_blink_state(state: VisionState, ear: float, now: float) -> bool:
        """Update one participant's blink state and return whether a blink ended.

        Eye frames are sampled asynchronously, so requiring a fixed number of
        closed frames makes normal short blinks impossible to detect when the
        sampling interval is longer than the blink. Duration-based detection
        works with both fast and slower camera/API sampling. A small hysteresis
        prevents EAR noise near the threshold from repeatedly opening an eye.
        """
        reopen_threshold = state.ear_threshold + 0.015
        if not state.eyes_closed:
            if ear < state.ear_threshold:
                state.eyes_closed = True
                state.closed_started_at = now
                state.closed_frames = 1
            return False

        state.closed_frames += 1
        if ear < reopen_threshold:
            return False

        closure_duration = now - state.closed_started_at
        blinked = 0.04 <= closure_duration <= 0.8
        if blinked:
            state.blink_count += 1
            state.last_blink_latency_ms = round(closure_duration * 1000, 1)
        state.eyes_closed = False
        state.closed_frames = 0
        state.closed_started_at = 0.0
        return blinked

    def analyze(self, participant_id: str, data_url: str, calibration_point: int | None = None) -> dict[str, Any]:
        import cv2
        import numpy as np

        frame = self._decode(data_url)
        height, width = frame.shape[:2]
        with self._inference_lock:
            results = self._mesh().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return {
                "face_detected": False,
                "pupils_detected": False,
                "calibrating": False,
                "calibration_complete": False,
                "tracking_confidence": 0.0,
            }

        face_landmark_sets = [face.landmark for face in results.multi_face_landmarks[:5]]
        landmarks = face_landmark_sets[0]
        faces_detected = len(face_landmark_sets)
        left_iris = self._iris(landmarks, LEFT_IRIS, width, height)
        right_iris = self._iris(landmarks, RIGHT_IRIS, width, height)
        ear = (self._ear(landmarks, LEFT_EAR_POINTS, width, height) + self._ear(landmarks, RIGHT_EAR_POINTS, width, height)) / 2
        try:
            head_pitch, head_yaw = self._head_pose(landmarks, width, height)
        except Exception:
            head_pitch, head_yaw = 0.0, 0.0
        now = time.time()

        with self._lock:
            state = self._states.setdefault(participant_id, VisionState())
            state.valid_sample_count += 1
            if state.calibrated_left is None:
                state.calibration_ears.append(ear)
                if calibration_point is None:
                    left_samples = state.calibration_left.setdefault(-1, [])
                    right_samples = state.calibration_right.setdefault(-1, [])
                    if len(left_samples) < 10:
                        left_samples.append(left_iris)
                        right_samples.append(right_iris)
                    calibration = min(100, len(left_samples) * 10)
                    if len(left_samples) >= 10:
                        state.calibrated_left = np.median(left_samples, axis=0)
                        state.calibrated_right = np.median(right_samples, axis=0)
                        state.ear_threshold = min(0.25, max(0.16, float(np.median(state.calibration_ears)) * 0.72))
                    return {
                        "face_detected": True,
                        "faces_detected": faces_detected,
                        "face_signatures": self._face_signatures(face_landmark_sets),
                        "pupils_detected": True,
                        "calibrating": state.calibrated_left is None,
                        "calibration_complete": state.calibrated_left is not None,
                        "calibration_percent": calibration,
                        "left_pupil_x": round(float(left_iris[0]), 3),
                        "left_pupil_y": round(float(left_iris[1]), 3),
                        "right_pupil_x": round(float(right_iris[0]), 3),
                        "right_pupil_y": round(float(right_iris[1]), 3),
                    }
                if not 0 <= calibration_point < 16:
                    raise ValueError("calibration_point must be between 0 and 15")
                left_samples = state.calibration_left.setdefault(calibration_point, [])
                right_samples = state.calibration_right.setdefault(calibration_point, [])
                if len(left_samples) < 2:
                    left_samples.append(left_iris)
                    right_samples.append(right_iris)
                completed_points = sum(len(samples) >= 2 for samples in state.calibration_left.values())
                calibration = round(completed_points / 16 * 100)
                point_complete = len(left_samples) >= 2
                calibration_complete = completed_points == 16
                if calibration_complete:
                    central_points = (5, 6, 9, 10)
                    state.calibrated_left = np.mean([sample for point in central_points for sample in state.calibration_left[point]], axis=0)
                    state.calibrated_right = np.mean([sample for point in central_points for sample in state.calibration_right[point]], axis=0)
                    state.ear_threshold = min(0.25, max(0.16, float(np.median(state.calibration_ears)) * 0.72))
                return {
                    "face_detected": True,
                    "faces_detected": faces_detected,
                    "face_signatures": self._face_signatures(face_landmark_sets),
                    "pupils_detected": True,
                    "calibrating": not calibration_complete,
                    "calibration_point": calibration_point,
                    "calibration_point_complete": point_complete,
                    "calibration_complete": calibration_complete,
                    "calibration_percent": calibration,
                    "left_pupil_x": round(float(left_iris[0]), 3),
                    "left_pupil_y": round(float(left_iris[1]), 3),
                    "right_pupil_x": round(float(right_iris[0]), 3),
                    "right_pupil_y": round(float(right_iris[1]), 3),
                }

            interocular_distance = max(1.0, float(np.linalg.norm(left_iris - right_iris)))
            gaze = (((left_iris - state.calibrated_left) + (right_iris - state.calibrated_right)) / 2) / interocular_distance
            direction = self._direction(gaze)
            state.directions.append(direction)
            direction = Counter(state.directions).most_common(1)[0][0]
            speed = 0.0
            if state.last_gaze is not None and now > state.last_gaze_at:
                speed = float(np.linalg.norm(gaze - state.last_gaze) / (now - state.last_gaze_at))
            state.last_gaze, state.last_gaze_at = gaze, now
            state.ear_history.append(ear)
            blinked = self._update_blink_state(state, ear, now)
            closed = state.eyes_closed
            state.closure_history.append((now, closed))
            while state.closure_history and now - state.closure_history[0][0] > 60:
                state.closure_history.popleft()
            perclos = sum(item[1] for item in state.closure_history) / len(state.closure_history)
            fatigue_score = min(1.0, max(0.0, (0.30 - ear) / 0.30) * 0.5 + min(1.0, perclos / 0.3) * 0.5)
            pupil_symmetry = max(0.0, 1.0 - abs(left_iris[1] - right_iris[1]) / max(1.0, interocular_distance * 0.2))
            pose_confidence = max(0.0, 1.0 - (abs(head_yaw) / 35 + abs(head_pitch) / 35) / 2)
            tracking_confidence = min(1.0, pupil_symmetry * 0.65 + pose_confidence * 0.35)

        output: dict[str, Any] = {
            "face_detected": True,
            "faces_detected": faces_detected,
            "face_signatures": self._face_signatures(face_landmark_sets),
            "pupils_detected": True,
            "calibrating": False,
            "direction": direction,
            "saccade": "fast" if speed > 0.35 else "focused",
            "blinked": blinked,
            "blink_count": state.blink_count,
            "blink_latency_ms": state.last_blink_latency_ms if blinked else 0.0,
            "ear": round(ear, 4),
            "ear_threshold": round(state.ear_threshold, 4),
            "perclos": round(perclos, 4),
            "fatigue_score": round(fatigue_score, 4),
            "fatigue": "high" if fatigue_score >= 0.6 else "moderate" if fatigue_score >= 0.3 else "low",
            "head_pitch": round(head_pitch, 2),
            "head_yaw": round(head_yaw, 2),
            "left_pupil_x": round(float(left_iris[0]), 3),
            "left_pupil_y": round(float(left_iris[1]), 3),
            "right_pupil_x": round(float(right_iris[0]), 3),
            "right_pupil_y": round(float(right_iris[1]), 3),
            "tracking_confidence": round(tracking_confidence, 4),
            "valid_sample_count": state.valid_sample_count,
        }
        return output

    @staticmethod
    def _face_signatures(face_landmark_sets) -> list[str]:
        """Return short anonymized signatures for up to five detected faces.

        The signatures are derived from normalized, quantized landmarks rather
        than storing camera images. They are useful for counting/enrollment and
        are not intended as biometric authentication credentials.
        """
        import numpy as np

        signatures: list[str] = []
        anchor_indices = (1, 33, 61, 152, 263, 291, 10, 234, 454, 168)
        for landmarks in face_landmark_sets[:5]:
            points = np.array([[landmarks[index].x, landmarks[index].y, landmarks[index].z] for index in anchor_indices], dtype=float)
            minimum = points[:, :2].min(axis=0)
            maximum = points[:, :2].max(axis=0)
            scale = np.maximum(maximum - minimum, 1e-6)
            normalized = points.copy()
            normalized[:, :2] = (normalized[:, :2] - minimum) / scale
            normalized[:, 2] = normalized[:, 2] / max(float(scale.max()), 1e-6)
            quantized = np.round(normalized, 3).tolist()
            signatures.append(hashlib.sha256(json.dumps(quantized, separators=(",", ":")).encode("utf-8")).hexdigest()[:32])
        return signatures


vision_analyzer = VisionAnalyzer()
