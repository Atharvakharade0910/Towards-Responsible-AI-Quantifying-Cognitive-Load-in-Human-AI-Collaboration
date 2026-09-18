from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import isfinite
from typing import Any


VALID_EMOTIONS = {
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
}


@dataclass
class SecondEmotionAggregator:
    """
    Collect facial-expression predictions from selected frames
    belonging to one second and calculate one final expression.

    Tie-handling priority:
    1. Clear majority
    2. Last two valid frames
    3. Meaningful confidence-total difference
    4. Previous second's confirmed emotion
    5. Unknown
    """

    elapsed_second: int
    expected_frames: int = 6
    confidence_gap_threshold: float = 0.15

    _emotions: list[str] = field(default_factory=list)
    _model_confidences: list[float] = field(default_factory=list)
    _total_frames: int = 0
    _invalid_reasons: Counter[str] = field(
        default_factory=Counter
    )

    def __post_init__(self) -> None:
        if self.elapsed_second < 0:
            raise ValueError(
                "elapsed_second cannot be negative."
            )

        if self.expected_frames < 1:
            raise ValueError(
                "expected_frames must be at least 1."
            )

        if self.confidence_gap_threshold < 0:
            raise ValueError(
                "confidence_gap_threshold cannot be negative."
            )

    def add_result(
        self,
        result: dict[str, Any],
    ) -> None:
        """
        Add the result from one selected webcam frame.

        Valid frames participate in expression voting.

        No-face and invalid frames are counted but do not vote.
        """

        self._total_frames += 1

        face_detected = bool(
            result.get("face_detected", False)
        )

        emotion = str(
            result.get("emotion", "unknown")
        ).strip().lower()

        if not face_detected or emotion not in VALID_EMOTIONS:
            reason = str(
                result.get("reason", "invalid_frame")
            ).strip()

            if not reason:
                reason = "invalid_frame"

            self._invalid_reasons[reason] += 1
            return

        confidence = self._safe_confidence(
            result.get("model_confidence", 0.0)
        )

        self._emotions.append(emotion)
        self._model_confidences.append(confidence)

    @staticmethod
    def _safe_confidence(value: Any) -> float:
        """
        Convert model confidence to a safe value between 0 and 1.
        """

        try:
            confidence = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0

        if not isfinite(confidence):
            return 0.0

        return max(
            0.0,
            min(1.0, confidence),
        )

    @property
    def valid_frames(self) -> int:
        """
        Number of frames that produced a valid expression.
        """

        return len(self._emotions)

    @property
    def total_frames(self) -> int:
        """
        Total number of selected frames received.
        """

        return self._total_frames

    def finalize(
        self,
        previous_emotion: str | None = None,
        minimum_valid_frames: int = 3,
    ) -> dict[str, Any]:
        """
        Calculate the final expression for this second.

        Args:
            previous_emotion:
                Confirmed expression from the previous second.

            minimum_valid_frames:
                Minimum number of valid facial predictions required
                to assign an expression to the current second.
        """

        if minimum_valid_frames < 1:
            raise ValueError(
                "minimum_valid_frames must be at least 1."
            )

        normalized_previous = self._normalize_previous_emotion(
            previous_emotion
        )

        vote_counts = Counter(self._emotions)

        if self.valid_frames < minimum_valid_frames:
            return {
                "elapsed_second": self.elapsed_second,
                "emotion": "unknown",
                "majority_confidence": 0.0,
                "average_model_confidence": 0.0,
                "valid_frames": self.valid_frames,
                "total_frames": self.total_frames,
                "expected_frames": self.expected_frames,
                "vote_counts": dict(vote_counts),
                "invalid_reasons": dict(
                    self._invalid_reasons
                ),
                "result_status": (
                    "insufficient_valid_frames"
                ),
                "tie_break_reason": None,
            }

        (
            selected_emotion,
            selected_votes,
            tie_break_reason,
        ) = self._select_emotion(
            vote_counts=vote_counts,
            previous_emotion=normalized_previous,
        )

        majority_confidence = (
            selected_votes / self.valid_frames
            if selected_emotion != "unknown"
            else 0.0
        )

        selected_confidences = [
            confidence
            for emotion, confidence in zip(
                self._emotions,
                self._model_confidences,
            )
            if emotion == selected_emotion
        ]

        average_model_confidence = (
            sum(selected_confidences)
            / len(selected_confidences)
            if selected_confidences
            else 0.0
        )

        return {
            "elapsed_second": self.elapsed_second,
            "emotion": selected_emotion,
            "majority_confidence": round(
                majority_confidence,
                4,
            ),
            "average_model_confidence": round(
                average_model_confidence,
                6,
            ),
            "valid_frames": self.valid_frames,
            "total_frames": self.total_frames,
            "expected_frames": self.expected_frames,
            "vote_counts": dict(vote_counts),
            "invalid_reasons": dict(
                self._invalid_reasons
            ),
            "result_status": (
                "success"
                if selected_emotion != "unknown"
                else "ambiguous_tie"
            ),
            "tie_break_reason": tie_break_reason,
        }

    @staticmethod
    def _normalize_previous_emotion(
        previous_emotion: str | None,
    ) -> str | None:
        """
        Validate and normalize the previous second's expression.
        """

        if not isinstance(previous_emotion, str):
            return None

        normalized = previous_emotion.strip().lower()

        if normalized not in VALID_EMOTIONS:
            return None

        return normalized

    def _select_emotion(
        self,
        vote_counts: Counter[str],
        previous_emotion: str | None,
    ) -> tuple[str, int, str]:
        """
        Select the final expression for one second.

        Priority:
        1. Clear majority
        2. Last two valid frames
        3. Meaningful confidence-total difference
        4. Previous second's confirmed expression
        5. Unknown
        """

        highest_vote_count = max(
            vote_counts.values()
        )

        tied_emotions = [
            emotion
            for emotion, votes in vote_counts.items()
            if votes == highest_vote_count
        ]

        # 1. A clear majority exists.
        if len(tied_emotions) == 1:
            return (
                tied_emotions[0],
                highest_vote_count,
                "clear_majority",
            )

        # 2. In a tie, use the last two valid frames
        # only when both frames agree.
        if len(self._emotions) >= 2:
            second_last_emotion = self._emotions[-2]
            last_emotion = self._emotions[-1]

            if (
                second_last_emotion == last_emotion
                and last_emotion in tied_emotions
            ):
                return (
                    last_emotion,
                    highest_vote_count,
                    "last_two_valid_frames",
                )

        # 3. Calculate confidence totals for tied expressions.
        confidence_totals = {
            emotion: sum(
                confidence
                for frame_emotion, confidence in zip(
                    self._emotions,
                    self._model_confidences,
                )
                if frame_emotion == emotion
            )
            for emotion in tied_emotions
        }

        ranked_emotions = sorted(
            confidence_totals.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        highest_emotion, highest_confidence = (
            ranked_emotions[0]
        )

        second_highest_confidence = (
            ranked_emotions[1][1]
        )

        confidence_gap = (
            highest_confidence
            - second_highest_confidence
        )

        # Select the confidence winner only when
        # the difference is meaningful.
        if (
            confidence_gap > 0
            and confidence_gap >= self.confidence_gap_threshold
        ):
            return (
                highest_emotion,
                highest_vote_count,
                "confidence_total",
            )

        # 4. Confidence difference is small.
        # Retain the previous expression for stability.
        if previous_emotion in tied_emotions:
            return (
                previous_emotion,
                highest_vote_count,
                "previous_second_emotion",
            )

        # 5. No reliable decision can be made.
        return (
            "unknown",
            highest_vote_count,
            "unresolved_tie",
        )
