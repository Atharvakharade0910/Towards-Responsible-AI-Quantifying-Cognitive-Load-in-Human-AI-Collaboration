import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from facial_expression_aggregator import SecondEmotionAggregator


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
