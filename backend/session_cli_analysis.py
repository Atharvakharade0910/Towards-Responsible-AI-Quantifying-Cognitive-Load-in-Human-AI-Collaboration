"""Session-level analysis of CLI values emitted by the existing ML pipeline.

This module deliberately does not calculate or retrain the model.  It consumes
the model's persisted per-second CLI output and produces an auditable summary
for one participant/session pair.
"""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Any, Iterable


CLI_FIELD_ALIASES = ("combined_cli", "cli_score", "cli", "predicted_cli")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) else None


def _cli_value(row: dict[str, Any]) -> float | None:
    for field in CLI_FIELD_ALIASES:
        value = _number(row.get(field))
        if value is not None:
            return value
    return None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else None


def _slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    x_mean = (len(values) - 1) / 2
    y_mean = sum(values) / len(values)
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return round(sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values)) / denominator, 4)


def build_session_cli_summary(
    *,
    user_id: str,
    session_id: str,
    rows: Iterable[dict[str, Any]],
    baseline_cli: float | None = None,
    post_cli: float | None = None,
    created_at: str,
) -> dict[str, Any]:
    """Build a deterministic summary from existing model output rows."""
    baseline_cli = _number(baseline_cli)
    post_cli = _number(post_cli)
    samples: list[tuple[dict[str, Any], float]] = []
    for row in rows:
        value = _cli_value(row)
        if value is not None:
            samples.append((row, value))

    samples.sort(key=lambda item: (_number(item[0].get("elapsed_second")) or 0, str(item[0].get("captured_at", ""))))
    values = [value for _, value in samples]
    task_values: dict[str, list[float]] = defaultdict(list)
    for row, value in samples:
        task = str(row.get("task_number") or "unknown")
        task_values[task].append(value)

    task_cli = {task: _mean(task_values[task]) for task in sorted(task_values, key=lambda item: (item == "unknown", item))}
    task_means = [value for value in task_cli.values() if value is not None]
    start_cli = values[0] if values else None
    end_cli = values[-1] if values else None
    average_cli = _mean(values)
    peak_cli = round(max(values), 4) if values else None
    minimum_cli = round(min(values), 4) if values else None
    absolute_change = round(end_cli - start_cli, 4) if start_cli is not None and end_cli is not None else None
    percentage_change = round(absolute_change / start_cli * 100, 4) if absolute_change is not None and start_cli else None
    exposure = None
    if baseline_cli is not None and values:
        exposure = round(sum(max(0.0, value - baseline_cli) for value in values) / len(values), 4)

    return {
        "user_id": user_id,
        "participant_id": user_id,
        "session_id": session_id,
        "start_cli": start_cli,
        "end_cli": end_cli,
        "average_cli": average_cli,
        "minimum_cli": minimum_cli,
        "maximum_cli": peak_cli,
        "baseline_cli": baseline_cli,
        "post_cli": post_cli,
        "absolute_change": absolute_change,
        "percentage_change": percentage_change,
        "task_induced_change": round(average_cli - baseline_cli, 4) if average_cli is not None and baseline_cli is not None else None,
        "recovery_change": round(post_cli - average_cli, 4) if post_cli is not None and average_cli is not None else None,
        "peak_change": round(peak_cli - baseline_cli, 4) if peak_cli is not None and baseline_cli is not None else None,
        "trend_slope": _slope(task_means),
        "exposure_above_baseline": exposure,
        "sample_count": len(values),
        "task_count": len(task_cli),
        "task_cli": dict(task_cli),
        "calculation_version": "session-cli-change-v1",
        "created_at": created_at,
    }
