"""Independent real-time pupil and eye-tracking pipeline.

It uses MediaPipe's refined iris landmarks.
"""
from __future__ import annotations

from vision_analysis import VisionAnalyzer


class EyeTrackingAnalyzer(VisionAnalyzer):
    """Eye-only adapter reused by the browser/API eye stream.

    ``VisionAnalyzer`` already implements the calibrated iris, EAR, blink,
    PERCLOS, gaze, saccade, and head-pose algorithm from the supplied pupil
    tracker.
    """

    def analyze_eye_frame(self, participant_id: str, data_url: str, calibration_point: int | None = None):
        return self.analyze(participant_id, data_url, calibration_point)


eye_tracking_analyzer = EyeTrackingAnalyzer()
