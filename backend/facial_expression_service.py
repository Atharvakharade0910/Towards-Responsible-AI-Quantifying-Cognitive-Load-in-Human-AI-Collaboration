from __future__ import annotations

import base64
import binascii
import threading
from dataclasses import dataclass, field
from typing import Any

try:
    import cv2
    import numpy as np
    from facial_expression_processor import FacialExpressionProcessor

    FACIAL_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # Keep the rest of the API available if FER is unavailable.
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    FacialExpressionProcessor = Any  # type: ignore[assignment,misc]
    FACIAL_IMPORT_ERROR = exc


@dataclass
class _ParticipantFacialState:
    """Temporal facial-expression state for one participant."""

    question_id: str
    task_number: int
    processor: Any
    segment_start_second: int | None = None
    question_closed: bool = False
    lock: Any = field(default_factory=threading.Lock, repr=False)


class FacialExpressionService:
    """Participant-aware facial-expression processing service.

    The browser sends approximately six selected frames per second from the
    same webcam stream used by eye tracking. This service keeps those frames
    separated by participant and question, runs ONNX inference, aggregates the
    frame predictions into one result per elapsed second, and emits sparse
    storage events instead of storing every frame.
    """

    def __init__(
        self,
        expected_frames_per_second: int = 6,
        minimum_valid_frames: int = 3,
        confidence_gap_threshold: float = 0.15,
    ) -> None:
        self.expected_frames_per_second = expected_frames_per_second
        self.minimum_valid_frames = minimum_valid_frames
        self.confidence_gap_threshold = confidence_gap_threshold
        self._states: dict[str, _ParticipantFacialState] = {}
        self._registry_lock = threading.RLock()

    def _new_processor(self) -> Any:
        if FACIAL_IMPORT_ERROR is not None:
            raise ImportError(
                "Facial-expression dependencies are not installed"
            ) from FACIAL_IMPORT_ERROR

        return FacialExpressionProcessor(
            expected_frames_per_second=self.expected_frames_per_second,
            minimum_valid_frames=self.minimum_valid_frames,
            confidence_gap_threshold=self.confidence_gap_threshold,
        )

    @staticmethod
    def _decode(data_url: str) -> Any:
        if cv2 is None or np is None:
            raise ImportError("Facial-expression dependencies are not installed")

        if not isinstance(data_url, str) or not data_url.strip():
            raise ValueError("The facial-expression image is empty")

        encoded = data_url.split(",", 1)[-1]
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("The facial-expression image is not valid base64") from exc

        frame = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is None:
            raise ValueError("Invalid facial-expression camera frame")
        return frame

    def _get_or_create_state(
        self,
        participant_id: str,
        question_id: str,
        task_number: int,
    ) -> _ParticipantFacialState:
        with self._registry_lock:
            state = self._states.get(participant_id)
            if state is None:
                state = _ParticipantFacialState(
                    question_id=question_id,
                    task_number=task_number,
                    processor=self._new_processor(),
                )
                self._states[participant_id] = state
            return state

    @staticmethod
    def _attach_question(
        result: dict[str, Any] | None,
        question_id: str,
        task_number: int,
    ) -> dict[str, Any] | None:
        if result is None:
            return None
        return {
            "question_id": question_id,
            "task_number": task_number,
            **result,
        }

    @staticmethod
    def _ignored_frame(reason: str) -> dict[str, Any]:
        return {
            "face_detected": False,
            "emotion": "unknown",
            "model_confidence": 0.0,
            "scores": {},
            "inference_ms": 0.0,
            "processing_ms": 0.0,
            "reason": reason,
        }

    @staticmethod
    def _build_storage_event(
        state: _ParticipantFacialState,
        second_result: dict[str, Any] | None,
        terminal_reason: str | None = None,
    ) -> dict[str, Any] | None:
        """Convert one completed second into a sparse storage event."""

        if not second_result:
            return None

        result_status = str(second_result.get("result_status", "")).strip().lower()
        emotion = str(second_result.get("emotion", "unknown")).strip().lower() or "unknown"

        elapsed_second = int(second_result.get("elapsed_second", 0) or 0)
        if elapsed_second < 1:
            return None

        previous_value = second_result.get("previous_emotion")
        previous_emotion = (
            str(previous_value).strip().lower() if previous_value else ""
        )
        emotion_changed = bool(second_result.get("emotion_changed", False))

        previous_segment_start_second: int | None = None
        previous_segment_end_second: int | None = None

        if state.segment_start_second is None:
            state.segment_start_second = elapsed_second

        if emotion_changed:
            previous_segment_start_second = state.segment_start_second
            previous_segment_end_second = max(1, elapsed_second - 1)
            state.segment_start_second = elapsed_second

        if terminal_reason:
            record_reason: str | None = terminal_reason
        elif not previous_emotion:
            record_reason = "initial"
        elif emotion_changed:
            record_reason = "emotion_change"
        elif elapsed_second % 10 == 0:
            record_reason = "interval_checkpoint"
        else:
            record_reason = None

        if record_reason is None:
            if result_status != "success" or emotion == "unknown":
                record_reason = result_status or "unknown_emotion"
            else:
                record_reason = "second_complete"

        return {
            "question_id": state.question_id,
            "task_number": state.task_number,
            "elapsed_second": elapsed_second,
            "emotion": emotion,
            "confidence": float(second_result.get("majority_confidence", 0.0) or 0.0),
            "average_model_confidence": float(
                second_result.get("average_model_confidence", 0.0) or 0.0
            ),
            "valid_frames": int(second_result.get("valid_frames", 0) or 0),
            "total_frames": int(second_result.get("total_frames", 0) or 0),
            "expected_frames": int(
                second_result.get("expected_frames", 6) or 6
            ),
            "record_reason": record_reason,
            "previous_emotion": previous_emotion,
            "segment_start_second": state.segment_start_second,
            "segment_end_second": elapsed_second,
            "previous_segment_start_second": (
                previous_segment_start_second
                if previous_segment_start_second is not None
                else ""
            ),
            "previous_segment_end_second": (
                previous_segment_end_second
                if previous_segment_end_second is not None
                else ""
            ),
            "emotion_changed": emotion_changed,
            "tie_break_reason": str(second_result.get("tie_break_reason") or ""),
            "result_status": result_status,
        }

    def _finalize_state(
        self,
        state: _ParticipantFacialState,
        terminal_reason: str,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Finalize and reset the active question for a participant."""

        completed = state.processor.reset()
        attached = self._attach_question(
            completed,
            state.question_id,
            state.task_number,
        )
        event = self._build_storage_event(
            state=state,
            second_result=attached,
            terminal_reason=terminal_reason,
        )
        state.segment_start_second = None
        state.question_closed = True
        return attached, [event] if event else []

    def process_frame(
        self,
        participant_id: str,
        question_id: str,
        task_number: int,
        frame_id: int,
        elapsed_ms: int,
        elapsed_second: int,
        image_data_url: str,
    ) -> dict[str, Any]:
        """Process one selected webcam frame for a participant question."""

        participant_id = participant_id.strip()
        question_id = question_id.strip()
        if not participant_id:
            raise ValueError("participant_id is required")
        if not question_id:
            raise ValueError("question_id is required")
        if task_number < 1:
            raise ValueError("task_number must start from 1")
        if frame_id < 1:
            raise ValueError(
                "frame_id must start from 1"
        )

        if elapsed_ms < 0:
            raise ValueError(
                "elapsed_ms cannot be negative"
        )
        if elapsed_second < 1:
            raise ValueError("elapsed_second must start from 1")

        frame = self._decode(image_data_url)
        state = self._get_or_create_state(
            participant_id=participant_id,
            question_id=question_id,
            task_number=task_number,
        )

        with state.lock:
            storage_events: list[dict[str, Any]] = []
            completed_previous_question: dict[str, Any] | None = None

            if state.question_id != question_id:
                if not state.question_closed:
                    completed_previous_question, old_events = self._finalize_state(
                        state,
                        terminal_reason="question_end",
                    )
                    storage_events.extend(old_events)

                state.question_id = question_id
                state.task_number = task_number
                state.segment_start_second = None
                state.question_closed = False

            elif state.question_closed:
                # A request that was already in flight when the answer was
                # submitted must not reopen the completed question.
                return {
                    "active_question_id": question_id,
                    "frame_result": self._ignored_frame("question_closed"),
                    "completed_second": None,
                    "completed_previous_question": None,
                    "storage_events": [],
                }

            state.task_number = task_number
            # ---------------------------------------------------------
            # FRAME-LEVEL FACIAL ANALYSIS
            # ---------------------------------------------------------

            processed = state.processor.process_selected_frame(
                frame_bgr=frame,
                elapsed_second=elapsed_second,
            )
            frame_result = processed.get("frame_result") or {}
            completed_second = processed.get("completed_second")
            if completed_second:
                completed_second = self._attach_question(
                    completed_second,
                    state.question_id,
                    state.task_number,
                )
                event = self._build_storage_event(
                    state=state,
                    second_result=completed_second,
                )
                if event:
                    storage_events.append(event)

            return {
                "active_question_id": question_id,

                # Synchronization metadata.
                "frame_id": frame_id,
                "elapsed_ms": elapsed_ms,
                "elapsed_second": elapsed_second,

                # Raw prediction for THIS frame.
                "frame_result": frame_result,

                "completed_second": completed_second,

                "completed_previous_question": (
                    completed_previous_question
                ),

                "storage_events": storage_events,
            }
            

            

    def flush_question(
        self,
        participant_id: str,
        question_id: str,
    ) -> dict[str, Any] | None:
        """Finalize the current second when an answer is submitted."""

        with self._registry_lock:
            state = self._states.get(participant_id)
        if state is None:
            return None

        with state.lock:
            if state.question_id != question_id or state.question_closed:
                return None

            completed, storage_events = self._finalize_state(
                state,
                terminal_reason="question_end",
            )
            return {
                "question_id": state.question_id,
                "task_number": state.task_number,
                "completed_second": completed,
                "storage_events": storage_events,
            }

    def forget_participant(
        self,
        participant_id: str,
    ) -> dict[str, Any] | None:
        """Finalize the active question, close resources, and remove state."""

        with self._registry_lock:
            state = self._states.pop(participant_id, None)
        if state is None:
            return None

        with state.lock:
            completed: dict[str, Any] | None = None
            storage_events: list[dict[str, Any]] = []
            try:
                if not state.question_closed:
                    completed, storage_events = self._finalize_state(
                        state,
                        terminal_reason="session_end",
                    )
            finally:
                state.processor.close()

            return {
                "question_id": state.question_id,
                "task_number": state.task_number,
                "completed_second": completed,
                "storage_events": storage_events,
            }


facial_expression_service = FacialExpressionService(
    expected_frames_per_second=6,
    minimum_valid_frames=3,
    confidence_gap_threshold=0.15,
)
