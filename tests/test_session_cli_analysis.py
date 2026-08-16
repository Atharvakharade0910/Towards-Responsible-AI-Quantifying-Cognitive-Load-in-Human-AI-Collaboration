import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from session_cli_analysis import build_session_cli_summary


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

