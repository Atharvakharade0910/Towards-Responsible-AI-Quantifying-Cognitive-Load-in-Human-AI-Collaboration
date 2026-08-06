from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from facial_expression import (
    DEFAULT_MODEL_PATH,
    FacialExpressionAnalyzer,
)
from facial_expression_aggregator import (
    SecondEmotionAggregator,
    VALID_EMOTIONS,
)


class FacialExpressionProcessor:
    """
    Connect frame-level facial-expression recognition with
    one-second majority aggregation.

    This class does not control the webcam frame rate.

    The caller is responsible for selecting approximately
    5–6 frames per second and passing them to this processor.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        expected_frames_per_second: int = 6,
        minimum_valid_frames: int = 3,
        confidence_gap_threshold: float = 0.15,
    ) -> None:
        if expected_frames_per_second < 1:
            raise ValueError(
                "expected_frames_per_second must be at least 1."
            )

        if minimum_valid_frames < 1:
            raise ValueError(
                "minimum_valid_frames must be at least 1."
            )

        if minimum_valid_frames > expected_frames_per_second:
            raise ValueError(
                "minimum_valid_frames cannot be greater than "
                "expected_frames_per_second."
            )

        if confidence_gap_threshold < 0:
            raise ValueError(
                "confidence_gap_threshold cannot be negative."
            )

        self.expected_frames_per_second = (
            expected_frames_per_second
        )

        self.minimum_valid_frames = minimum_valid_frames

        self.confidence_gap_threshold = (
            confidence_gap_threshold
        )

        self._analyzer = FacialExpressionAnalyzer(
            model_path=model_path
        )

        self._current_second: int | None = None

        self._current_aggregator: (
            SecondEmotionAggregator | None
        ) = None

        self._previous_emotion: str | None = None

        self._closed = False

    @property
    def previous_emotion(self) -> str | None:
        """
        Return the most recently confirmed per-second emotion.
        """

        return self._previous_emotion

    @property
    def current_second(self) -> int | None:
        """
        Return the elapsed second currently being collected.
        """

        return self._current_second

    def process_selected_frame(
        self,
        frame_bgr: np.ndarray,
        elapsed_second: int,
    ) -> dict[str, Any]:
        """
        Process one selected webcam frame.

        Args:
            frame_bgr:
                OpenCV BGR image.

            elapsed_second:
                One-based elapsed second.

                Example:
                    1 = first second
                    2 = second second
                    10 = tenth second

        Returns:
            {
                "frame_result": {...},
                "completed_second": {...} | None
            }

        completed_second is returned when the incoming frame belongs
        to a newer second than the previously collected frames.
        """

        self._ensure_open()

        if elapsed_second < 1:
            raise ValueError(
                "elapsed_second must start from 1."
            )

        completed_second: dict[str, Any] | None = None

        if self._current_second is None:
            self._start_second(elapsed_second)

        elif elapsed_second < self._current_second:
            raise ValueError(
                "Frames must be supplied in chronological order. "
                f"Current second is {self._current_second}, "
                f"but received second {elapsed_second}."
            )

        elif elapsed_second > self._current_second:
            completed_second = self._finalize_current_second()
            self._start_second(elapsed_second)

        frame_result = self._analyzer.analyze_frame(
            frame_bgr
        )

        if self._current_aggregator is None:
            raise RuntimeError(
                "The per-second aggregator was not initialized."
            )

        self._current_aggregator.add_result(
            frame_result
        )

        return {
            "frame_result": frame_result,
            "completed_second": completed_second,
        }

    def _start_second(
        self,
        elapsed_second: int,
    ) -> None:
        """
        Create a fresh aggregator for a new elapsed second.
        """

        self._current_second = elapsed_second

        self._current_aggregator = (
            SecondEmotionAggregator(
                elapsed_second=elapsed_second,
                expected_frames=(
                    self.expected_frames_per_second
                ),
                confidence_gap_threshold=(
                    self.confidence_gap_threshold
                ),
            )
        )

    def _finalize_current_second(
        self,
    ) -> dict[str, Any] | None:
        """
        Finalize the current second and update temporal state.
        """

        if (
            self._current_aggregator is None
            or self._current_second is None
        ):
            return None

        previous_before_decision = self._previous_emotion

        second_result = (
            self._current_aggregator.finalize(
                previous_emotion=(
                    previous_before_decision
                ),
                minimum_valid_frames=(
                    self.minimum_valid_frames
                ),
            )
        )

        selected_emotion = str(
            second_result.get(
                "emotion",
                "unknown",
            )
        ).strip().lower()

        expression_is_valid = (
            selected_emotion in VALID_EMOTIONS
        )

        emotion_changed = (
            expression_is_valid
            and previous_before_decision is not None
            and selected_emotion
            != previous_before_decision
        )

        second_result["previous_emotion"] = (
            previous_before_decision
        )

        second_result["emotion_changed"] = (
            emotion_changed
        )

        if expression_is_valid:
            self._previous_emotion = (
                selected_emotion
            )

        return second_result

    def flush(
        self,
    ) -> dict[str, Any] | None:
        """
        Finalize the currently collected second.

        Use this when:
        - testing stops,
        - a question ends,
        - a session ends,
        - or the processor is reset.
        """

        self._ensure_open()

        completed_second = (
            self._finalize_current_second()
        )

        self._current_second = None
        self._current_aggregator = None

        return completed_second

    def reset(
        self,
    ) -> dict[str, Any] | None:
        """
        Finalize the current second and clear temporal state.

        This will later be used when moving to a new question.
        """

        completed_second = self.flush()

        self._previous_emotion = None

        return completed_second

    def close(
        self,
    ) -> None:
        """
        Release MediaPipe and model-related resources.
        """

        if self._closed:
            return

        self._analyzer.close()

        self._current_second = None
        self._current_aggregator = None
        self._previous_emotion = None
        self._closed = True

    def _ensure_open(
        self,
    ) -> None:
        """
        Prevent use after the processor has been closed.
        """

        if self._closed:
            raise RuntimeError(
                "FacialExpressionProcessor is closed."
            )

    def __enter__(
        self,
    ) -> "FacialExpressionProcessor":
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        self.close()