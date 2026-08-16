"""Notebook-aligned held-out accuracy for the Group User dashboard."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def _notebook_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Map PostgreSQL's imported CSV headers to the notebook's feature contract."""
    frame = pd.DataFrame(rows).rename(columns={
        "participant_id": "participant id",
        "question_id": "question id",
        "seconds_recorded": "seconds recorded",
        "duration_seconds": "duration seconds",
        "mean_ear": "mean ear",
        "min_ear": "min ear",
        "mean_perclos": "mean perclos",
        "max_perclos": "max perclos",
        "dominant_gaze": "dominant gaze",
        "total_blink_events": "total blink events",
        "dominant_fatigue": "dominant fatigue",
        "dominant_emotion": "dominant emotion",
        "total_keypresses": "total keypresses",
        "total_backspaces": "total backspaces",
        "backspace_rate": "backspace rate",
        "mean_thinking_pause_seconds": "mean thinking pause seconds",
        "total_mouse_moves": "total mouse moves",
        "total_cursor_distance_px": "total cursor distance px",
        "total_prompts": "total prompts",
        "average_prompt_length_words": "average prompt length words",
        "cli_score": "CLI_score",
        "cli_label": "CLI_label",
    })
    if "CLI_label" not in frame or "participant id" not in frame:
        raise ValueError("final_dataset is missing CLI_label or participant_id")
    return frame


def calculate_model_accuracies(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reproduce the notebook's grouped split and four GridSearchCV models."""
    frame = _notebook_dataframe(rows)
    target = "CLI_label"
    group = "participant id"
    categorical = ["dominant gaze", "dominant fatigue", "dominant emotion"]
    excluded = {"CLI_score", "question id", target, group, "id"}
    feature_columns = [column for column in frame.columns if column not in excluded]
    numeric = [column for column in feature_columns if column not in categorical]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    X = frame[feature_columns]
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame[target].astype(str))
    groups = frame[group].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups.iloc[train_idx]
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    models = [
        # The notebook's grids contain 200/400-tree candidates. A compact,
        # deterministic representative keeps the dashboard responsive while
        # preserving the same four model families and grouped test split.
        ("Random Forest", RandomForestClassifier(n_estimators=80, random_state=42, class_weight="balanced", n_jobs=1)),
        ("XGBoost", XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.05, objective="multi:softprob", eval_metric="mlogloss", random_state=42, n_jobs=1, tree_method="hist")),
        ("LightGBM", LGBMClassifier(n_estimators=80, num_leaves=31, learning_rate=0.05, random_state=42, n_jobs=1, verbose=-1)),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=80, random_state=42, class_weight="balanced", n_jobs=1)),
    ]
    results = []
    started = time.time()
    for name, estimator in models:
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        results.append({"name": name, "accuracy": round(float(accuracy_score(y_test, prediction)), 4)})
    results.sort(key=lambda item: item["accuracy"], reverse=True)
    return {
        "available": True,
        "source": "CognativeLoad_calculation_(1)_(1).ipynb",
        "source_type": "notebook_aligned_grouped_held_out_test",
        "sample_count": int(len(frame)),
        "test_count": int(len(y_test)),
        "target": target,
        "models": results,
        "best_model": results[0]["name"] if results else None,
        "runtime_seconds": round(time.time() - started, 1),
    }
