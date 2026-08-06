from __future__ import annotations

import base64
import threading
from typing import Any

try:
    import cv2
    import numpy as np
    from facial_expression import FacialExpressionAnalyzer
    FACIAL_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # Keep the rest of the API available if FER is unavailable.
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    FacialExpressionAnalyzer = Any  # type: ignore[assignment,misc]
    FACIAL_IMPORT_ERROR = exc


class FacialExpressionService:
    """Adapter for the ZIP facial-expression model.

    This service is intentionally separate from the eye-tracking analyzer and
    receives the same camera frame through its own API endpoint.
    """

    def __init__(self) -> None:
        self._analyzer: FacialExpressionAnalyzer | None = None
        self._lock = threading.Lock()

    def _get_analyzer(self) -> FacialExpressionAnalyzer:
        if FACIAL_IMPORT_ERROR is not None:
            raise ImportError("Facial-expression dependencies are not installed") from FACIAL_IMPORT_ERROR
        if self._analyzer is None:
            self._analyzer = FacialExpressionAnalyzer()
        return self._analyzer

    @staticmethod
    def _decode(data_url: str) -> np.ndarray:
        if cv2 is None or np is None:
            raise ImportError("Facial-expression dependencies are not installed")
        encoded = data_url.split(",", 1)[-1]
        raw = base64.b64decode(encoded, validate=True)
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid facial-expression camera frame")
        return frame

    def analyze(self, data_url: str) -> dict[str, Any]:
        frame = self._decode(data_url)
        with self._lock:
            return self._get_analyzer().analyze_frame(frame)


facial_expression_service = FacialExpressionService()
