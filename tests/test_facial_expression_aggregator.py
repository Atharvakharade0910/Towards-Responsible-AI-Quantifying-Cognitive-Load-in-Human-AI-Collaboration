import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from facial_expression_aggregator import SecondEmotionAggregator


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf"), -0.1])
def test_confidence_gap_threshold_rejects_invalid_values(threshold):
    with pytest.raises(ValueError, match="confidence_gap_threshold"):
        SecondEmotionAggregator(elapsed_second=1, confidence_gap_threshold=threshold)


@pytest.mark.parametrize("threshold", [0.0, 0.15, 2.0])
def test_confidence_gap_threshold_accepts_finite_nonnegative_values(threshold):
    aggregator = SecondEmotionAggregator(elapsed_second=1, confidence_gap_threshold=threshold)
    assert aggregator.confidence_gap_threshold == threshold


@pytest.mark.parametrize("previous, expected", [(None, "unknown"), ("sad", "sad")])
def test_zero_threshold_does_not_choose_an_equal_confidence_winner(previous, expected):
    for emotions in [("happy", "sad"), ("sad", "happy")]:
        aggregator = SecondEmotionAggregator(elapsed_second=1, confidence_gap_threshold=0)
        for emotion in emotions * 2:
            aggregator.add_result({
                "face_detected": True,
                "emotion": emotion,
                "model_confidence": 0.5,
            })

        result = aggregator.finalize(previous_emotion=previous)

        assert result["emotion"] == expected
        assert result["tie_break_reason"] == (
            "previous_second_emotion" if previous else "unresolved_tie"
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), 10 ** 400])
def test_invalid_confidence_cannot_win_a_tied_vote(invalid):
    aggregator = SecondEmotionAggregator(elapsed_second=1)
    for emotion, confidence in [("happy", invalid), ("sad", 0.5)] * 2:
        aggregator.add_result({
            "face_detected": True,
            "emotion": emotion,
            "model_confidence": confidence,
        })

    result = aggregator.finalize()

    assert result["emotion"] == "sad"
    assert result["tie_break_reason"] == "confidence_total"
    assert result["average_model_confidence"] == 0.5
    assert result["valid_frames"] == 4


@pytest.mark.parametrize("confidence, expected", [(-0.5, 0.0), (0.75, 0.75), (1.5, 1.0)])
def test_finite_confidence_remains_clamped(confidence, expected):
    aggregator = SecondEmotionAggregator(elapsed_second=1)
    for _ in range(3):
        aggregator.add_result({
            "face_detected": True,
            "emotion": "happy",
            "model_confidence": confidence,
        })

    result = aggregator.finalize()

    assert result["emotion"] == "happy"
    assert result["average_model_confidence"] == expected
