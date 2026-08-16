"""Random Forest feature-importance calculation for the group dashboard."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier


BASE_FEATURES = (
    "seconds_recorded", "duration_seconds", "mean_ear", "min_ear",
    "mean_perclos", "max_perclos", "total_blink_events", "total_keypresses",
    "total_backspaces", "backspace_rate", "mean_thinking_pause_seconds",
    "total_mouse_moves", "total_cursor_distance_px", "total_prompts",
    "average_prompt_length_words",
)


def _number(row: dict[str, Any], field: str) -> float:
    try:
        value = float(row.get(field) or 0)
        return value if np.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _features(row: dict[str, Any]) -> dict[str, float]:
    duration = max(_number(row, "duration_seconds"), 1.0)
    keypresses = _number(row, "total_keypresses")
    backspaces = _number(row, "total_backspaces")
    pause = _number(row, "mean_thinking_pause_seconds")
    perclos = _number(row, "mean_perclos")
    mean_ear = _number(row, "mean_ear")
    max_perclos = _number(row, "max_perclos")
    min_perclos = _number(row, "mean_perclos")
    min_ear = _number(row, "min_ear")
    mouse_moves = _number(row, "total_mouse_moves")
    cursor_distance = _number(row, "total_cursor_distance_px")
    values = {field: _number(row, field) for field in BASE_FEATURES}
    values.update({
        "pause_x_perclos": pause * perclos,
        "perclos_x_backspace": perclos * backspaces,
        "backspace_per_keypress": backspaces / max(keypresses, 1.0),
        "pause_ratio": pause / duration,
        "typing_speed": keypresses / duration,
        "perclos_range": max(0.0, max_perclos - min_perclos),
        "ear_range": max(0.0, mean_ear - min_ear),
        "blink_rate": _number(row, "total_blink_events") / duration,
        "mouse_move_rate": mouse_moves / duration,
        "cursor_speed": cursor_distance / duration,
    })
    return values


def calculate_feature_importances(rows: Iterable[dict[str, Any]], top_n: int = 20) -> dict[str, Any]:
    rows = list(rows)
    usable = [row for row in rows if str(row.get("cli_label") or "").strip()]
    feature_names = [*BASE_FEATURES, "pause_x_perclos", "perclos_x_backspace", "backspace_per_keypress", "pause_ratio", "typing_speed", "perclos_range", "ear_range", "blink_rate", "mouse_move_rate", "cursor_speed"]
    if len(usable) < 2 or len({str(row.get("cli_label")).strip() for row in usable}) < 2:
        return {"available": False, "reason": "At least two CLI label classes and two rows are required.", "features": [], "sample_count": len(usable)}

    matrix = np.asarray([[values[name] for name in feature_names] for values in (_features(row) for row in usable)], dtype=float)
    labels = np.asarray([str(row.get("cli_label")).strip() for row in usable])
    model = RandomForestClassifier(
        n_estimators=160,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
    )
    model.fit(matrix, labels)
    ranked = sorted(zip(feature_names, model.feature_importances_), key=lambda item: item[1], reverse=True)[:top_n]
    return {
        "available": True,
        "title": "Top 20 Feature Importances — Random Forest",
        "target": "CLI_label",
        "features": [{"feature": name, "importance": round(float(value), 6)} for name, value in ranked],
        "sample_count": len(usable),
        "class_count": len(model.classes_),
        "classes": [str(value) for value in model.classes_],
        "model": "Random Forest",
        "random_state": 42,
    }

