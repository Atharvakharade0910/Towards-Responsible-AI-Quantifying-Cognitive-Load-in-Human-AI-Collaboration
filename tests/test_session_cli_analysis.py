import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from session_cli_analysis import build_session_cli_summary


def test_percentage_change_uses_unrounded_scores():
    for end, expected in [(1.00004, 0.003), (0.99998, -0.003)]:
        summary = build_session_cli_summary(
            user_id="test-user",
            session_id="test-session",
            rows=[
                {"elapsed_second": 1, "combined_cli": 1.00001},
                {"elapsed_second": 2, "combined_cli": end},
            ],
            created_at="now",
        )

        assert summary["absolute_change"] == 0.0
        assert summary["percentage_change"] == expected


def test_session_cli_summary_skips_numbers_too_large_for_float():
    for oversized in (10 ** 400, -(10 ** 400)):
        summary = build_session_cli_summary(
            user_id="test-user",
            session_id="test-session",
            rows=[
                {"task_number": 1, "combined_cli": oversized},
                {"task_number": 1, "combined_cli": oversized, "cli_score": 30},
                {"task_number": 1, "combined_cli": 50},
            ],
            baseline_cli=oversized,
            post_cli=oversized,
            created_at="now",
        )

        assert summary["sample_count"] == 2
        assert summary["average_cli"] == 40
        assert summary["baseline_cli"] is None
        assert summary["post_cli"] is None
        json.dumps(summary, allow_nan=False)


def test_session_cli_summary_ignores_nonfinite_reference_values():
    for invalid in (float("nan"), float("inf"), float("-inf")):
        summary = build_session_cli_summary(
            user_id="test-user",
            session_id="test-session",
            rows=[{"task_number": 1, "combined_cli": 30}],
            baseline_cli=invalid,
            post_cli=invalid,
            created_at="now",
        )

        assert summary["average_cli"] == 30
        for field in (
            "baseline_cli", "post_cli", "task_induced_change",
            "recovery_change", "peak_change", "exposure_above_baseline",
        ):
            assert summary[field] is None, field
        json.dumps(summary, allow_nan=False)


def test_session_cli_summary_is_keyed_and_calculates_change():
    summary = build_session_cli_summary(
        user_id="COG0001",
        session_id="session-1",
        rows=[
            {"task_number": 1, "elapsed_second": 1, "combined_cli": "30"},
            {"task_number": 1, "elapsed_second": 2, "combined_cli": 40},
            {"task_number": 2, "elapsed_second": 3, "cli_score": 60},
        ],
        baseline_cli=20,
        post_cli=45,
        created_at="2026-08-13T00:00:00+00:00",
    )

    assert summary["user_id"] == "COG0001"
    assert summary["participant_id"] == "COG0001"
    assert summary["session_id"] == "session-1"
    assert summary["start_cli"] == 30.0
    assert summary["end_cli"] == 60.0
    assert summary["absolute_change"] == 30.0
    assert summary["average_cli"] == 43.3333
    assert summary["baseline_cli"] == 20
    assert summary["post_cli"] == 45
    assert summary["task_cli"] == {"1": 35.0, "2": 60.0}


def test_session_cli_summary_is_safe_without_model_rows():
    summary = build_session_cli_summary(
        user_id="COG0002",
        session_id="session-2",
        rows=[{"task_number": 1, "combined_cli": None}],
        created_at="now",
    )

    assert summary["sample_count"] == 0
    assert summary["absolute_change"] is None
    assert summary["task_cli"] == {}
