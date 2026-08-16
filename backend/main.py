from __future__ import annotations

import asyncio
import base64
import csv
from contextlib import asynccontextmanager
import hmac
import io
import os
import secrets
import hashlib
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ElementTree
from html.parser import HTMLParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
try:
    from groq import Groq
except ImportError:  # Allows local startup and tests when Groq is not configured.
    Groq = None  # type: ignore[assignment]
from pydantic import BaseModel, ConfigDict, Field

from storage import create_storage
from eye_tracking import eye_tracking_analyzer
from facial_expression_service import facial_expression_service
from rag_pipeline import chunk_text, classify_direction, embed_text, retrieve, cosine_similarity
from session_cli_analysis import build_session_cli_summary
from feature_importance import calculate_feature_importances
from model_accuracy import calculate_model_accuracies

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
logger = logging.getLogger(__name__)

def load_participant_token_secret() -> str:
    configured = os.getenv("PARTICIPANT_TOKEN_SECRET", "").strip()
    if configured and configured != "replace-with-a-long-random-secret":
        return configured
    secret_path = PROJECT_ROOT / "data" / ".participant_token_secret"
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    if secret_path.exists():
        stored = secret_path.read_text(encoding="utf-8").strip()
        if stored:
            return stored
    generated = secrets.token_urlsafe(48)
    secret_path.write_text(generated, encoding="utf-8")
    return generated

load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(PROJECT_ROOT / ".env.local", override=True)

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
storage = create_storage(DATABASE_URL)


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2").strip()
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "500"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip()
PAIZA_API_BASE_URL = os.getenv("PAIZA_API_BASE_URL", "https://api.paiza.io").rstrip("/")
SUPPORTED_COMPILER_LANGUAGES = {"java": "java", "python": "python3"}
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
HOST_USERNAME = os.getenv("HOST_USERNAME", "").strip()
HOST_PASSWORD = os.getenv("HOST_PASSWORD", "")
DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "").strip()
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")
TOKEN_TTL_MINUTES = int(os.getenv("TOKEN_TTL_MINUTES", "60"))
AUTH_TOKENS: dict[str, tuple[str, datetime]] = {}
AUTH_LOCK = threading.Lock()
LOGIN_FAILURES: dict[str, list[float]] = {}
LOGIN_FAILURE_WINDOW_SECONDS = 300
LOGIN_FAILURE_LIMIT = 10
PARTICIPANT_TOKEN_SECRET = load_participant_token_secret()
SESSION_STATE: dict[str, dict[str, Any]] = {}
SESSION_LOCK = threading.Lock()
FEATURE_IMPORTANCE_CACHE: dict[str, Any] | None = None
FEATURE_IMPORTANCE_LOCK = threading.Lock()
MODEL_ACCURACY_CACHE: dict[str, Any] | None = None
MODEL_ACCURACY_LOCK = threading.Lock()
COMPILER_MAX_CONCURRENT_RUNS = 4
COMPILER_SEMAPHORE = asyncio.Semaphore(COMPILER_MAX_CONCURRENT_RUNS)
TRACKING_COMPONENT_ORDER = {
    "eye_tracking": 1,
    "mouse_tracking": 2,
    "keyboard_tracking": 3,
    "facial_expression": 4,
    "camera_tracking": 5,
    "behavior_tracking": 6,
    "multimodal_tracking": 7,
    "question_tracking": 8,
    "prompt_tracking": 9,
}
EYE_TRACKING_STORAGE_FIELDS = (
    "face_detected",
    "direction",
    "saccade",
    "blinked",
    "blink_count",
    "blink_latency_ms",
    "ear",
    "perclos",
    "fatigue",
    "head_pitch",
    "head_yaw",
    "tracking_confidence",
)
FACIAL_EXPRESSION_STORAGE_FIELDS = (
    "elapsed_second",
    "emotion",
    "confidence",
    "average_model_confidence",
    "valid_frames",
    "total_frames",
    "expected_frames",
    "record_reason",
    "previous_emotion",
    "segment_start_second",
    "segment_end_second",
    "previous_segment_start_second",
    "previous_segment_end_second",
    "emotion_changed",
    "tie_break_reason",
    "result_status",
)
FACIAL_EXPRESSION_TABLE_FIELDS = (
    "participant_id", "question_id", "task_number", "elapsed_second",
    "emotion", "confidence", "average_model_confidence", "valid_frames",
    "total_frames", "expected_frames", "record_reason", "previous_emotion",
    "segment_start_second", "segment_end_second",
    "previous_segment_start_second", "previous_segment_end_second",
    "emotion_changed", "tie_break_reason", "result_status",
    "captured_at", "received_at",
)


MULTIMODAL_SECOND_EXPORT_FIELDS = (
    "participant_id", "question_id", "task_number", "elapsed_second",
    "camera_available", "behavior_available",
    "first_frame_id", "last_frame_id", "frame_count", "paired_frame_count",
    "first_captured_at", "last_captured_at",

    # Eye
    "eye_sample_count", "eye_face_detected_ratio",
    "mean_ear", "min_ear", "max_ear",
    "mean_perclos", "max_perclos",
    "dominant_gaze", "gaze_change_count", "saccade_count",
    "blink_events", "blink_count_end", "mean_blink_latency_ms",
    "mean_head_pitch", "mean_head_yaw",
    "dominant_fatigue", "mean_tracking_confidence",

    # Facial expression
    "facial_sample_count", "facial_face_detected_ratio",
    "dominant_emotion", "mean_emotion_confidence",
    "emotion_change_count",
    "angry_ratio", "disgust_ratio", "fearful_ratio",
    "happy_ratio", "neutral_ratio", "sad_ratio", "surprised_ratio",

    # Keyboard
    "keypress_count", "backspace_count",
    "typing_active", "thinking_pause_seconds",

    # Mouse
    "mouse_move_count", "cursor_distance_px",
    "scroll_up_count", "scroll_down_count", "scroll_count",
    "mouse_active",

    # Prompt
    "prompt_sent", "prompt_length_words",
    "prompt_count_so_far", "time_since_last_prompt_seconds",
    "prompt_sentiment",

    "received_at",
)


QUESTION_LEVEL_EXPORT_FIELDS = (
    "participant_id", "question_id", "task_number",
    "seconds_recorded", "duration_seconds",
    "camera_available_ratio", "behavior_available_ratio",

    # Eye
    "total_eye_samples", "mean_ear", "min_ear",
    "mean_perclos", "max_perclos", "dominant_gaze",
    "total_gaze_changes", "total_saccades",
    # Question-specific blink count comes from summed blink_events.
    # blink_count_end is cumulative in the Eye analyzer and remains only
    # in the second-level/debug dataset.
    "total_blink_events",
    "mean_blink_latency_ms", "mean_head_pitch", "mean_head_yaw",
    "dominant_fatigue", "mean_tracking_confidence",

    # Facial expression
    "total_facial_samples", "face_detected_ratio",
    "dominant_emotion", "mean_emotion_confidence",
    # total_emotion_changes = changes between sampled frames inside seconds.
    # dominant_emotion_transition_count = changes in second-level dominant
    # expression across the question (for example neutral -> sad).
    "total_emotion_changes", "dominant_emotion_transition_count",
    "angry_ratio", "disgust_ratio", "fearful_ratio",
    "happy_ratio", "neutral_ratio", "sad_ratio", "surprised_ratio",

    # Keyboard
    "total_keypresses", "total_backspaces", "backspace_rate",
    "typing_active_seconds", "typing_active_ratio",
    "thinking_pause_count", "mean_thinking_pause_seconds",
    "max_thinking_pause_seconds",

    # Mouse
    "total_mouse_moves", "total_cursor_distance_px",
    "total_scroll_up", "total_scroll_down", "total_scroll_count",
    "mouse_active_seconds", "mouse_active_ratio",

    # Prompt
    "total_prompts", "prompt_active_seconds",
    "average_prompt_length_words", "max_prompt_length_words",
    "time_to_first_prompt_seconds",
    "average_time_between_prompts_seconds",
    "positive_prompt_count", "neutral_prompt_count",
    "negative_prompt_count",

    "received_at",
)

# One durable, question-level measurement record containing the complete
# feature set requested for PostgreSQL/Admin/CSV consumers.
MEASUREMENT_RECORD_FIELDS = QUESTION_LEVEL_EXPORT_FIELDS
MEASUREMENT_PARAMETER_FIELDS = (
    "parameter_name", "display_name", "data_type", "created_at",
)
SESSION_CLI_SUMMARY_FIELDS = (
    "user_id", "participant_id", "session_id", "start_cli", "end_cli",
    "average_cli", "minimum_cli", "maximum_cli", "baseline_cli", "post_cli",
    "absolute_change", "percentage_change", "task_induced_change",
    "recovery_change", "peak_change", "trend_slope", "exposure_above_baseline",
    "sample_count", "task_count", "task_cli", "calculation_version", "created_at",
)
FACE_PROFILE_LOCK = threading.Lock()
FACE_PROFILE_CACHE: dict[str, set[str]] = {}
bearer_scheme = HTTPBearer(auto_error=False)
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:8002,http://localhost:8002",
    ).split(",")
    if origin.strip()
]

@asynccontextmanager
async def app_lifespan(_: FastAPI):
    ensure_measurement_tables()
    yield


app = FastAPI(title="CogniTrack AI Backend", version="1.0.0", lifespan=app_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_measurement_tables() -> None:
    storage.ensure_table("measurement_records", MEASUREMENT_RECORD_FIELDS)
    storage.ensure_table("measurement_parameters", MEASUREMENT_PARAMETER_FIELDS)
    storage.ensure_table("rag_prompt_evaluations", RAG_PROMPT_EVALUATION_FIELDS)
    storage.ensure_table(
        "nasa_tlx",
        (
            "participant_id", "assessment_phase", "session_started_at",
            "session_ended_at", "mental_demand", "physical_demand",
            "temporal_demand", "performance", "effort", "frustration",
            "overall_score", "received_at",
        ),
    )
    storage.ensure_table(
        "rag_documents",
        (
            "document_id", "exam_id", "filename", "title", "page_count",
            "chunk_count", "status", "uploaded_at", "source_path",
        ),
    )
    storage.ensure_table(
        "rag_chunks",
        (
            "chunk_id", "document_id", "exam_id", "page_number",
            "chunk_index", "chunk_text", "embedding", "created_at",
        ),
    )
    storage.ensure_table(
        "rag_pipeline",
        (
            "pipeline_id", "prompt_id", "participant_id", "question_id",
            "task_number", "prompt", "query_embedding", "retrieved_chunk_ids",
            "retrieved_context", "retrieved_similarity", "generated_answer",
            "answer_faithfulness", "goal_direction", "direction_score",
            "direction", "feedback", "status", "recorded_at",
        ),
    )
    storage.ensure_table("session_cli_summary", SESSION_CLI_SUMMARY_FIELDS)


def unified_tracking_record(component_type: str, record_kind: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "component_sequence": TRACKING_COMPONENT_ORDER[component_type],
        "component_type": component_type,
        "record_kind": record_kind,
        **record,
    }


def append_tracking(component_type: str, record_kind: str, record: dict[str, Any]) -> None:
    participant_id = str(record.get("participant_id") or "")
    session_id = str(SESSION_STATE.get(participant_id, {}).get("session_id") or "")
    unified = {
        **unified_tracking_record(component_type, record_kind, record),
    }
    if session_id and not unified.get("session_id"):
        unified["session_id"] = session_id
    storage.append("tracking_data", list(unified), unified)


def upsert_tracking(component_type: str, record_kind: str, record: dict[str, Any]) -> None:
    participant_id = str(record.get("participant_id") or "")
    session_id = str(SESSION_STATE.get(participant_id, {}).get("session_id") or "")
    unified = {
        **unified_tracking_record(component_type, record_kind, record),
    }
    if session_id and not unified.get("session_id"):
        unified["session_id"] = session_id
    conflict_fields = ["component_type", "record_kind", "participant_id", "question_id"]
    if record_kind == "second" and component_type in {
        "camera_tracking",
        "behavior_tracking",
        "multimodal_tracking",
    }:
        conflict_fields.extend(("task_number", "elapsed_second"))
    storage.upsert("tracking_data", list(unified), unified, conflict_fields)


def persist_measurement_record(summary: dict[str, Any]) -> None:
    """Materialize the complete question summary in its dedicated table."""
    storage.upsert(
        "measurement_records",
        MEASUREMENT_RECORD_FIELDS,
        summary,
        ("participant_id", "question_id", "task_number"),
    )


def persist_session_cli_summary(
    participant_id: str,
    session_id: str,
    post_cli: float | None = None,
) -> dict[str, Any]:
    """Materialize one session-level summary from existing ML CLI outputs."""
    tracking_rows = [
        row for row in storage.list_records("tracking_data", 100_000)
        if str(row.get("participant_id") or "") == participant_id
        and str(row.get("session_id") or "") == session_id
        and row.get("record_kind") == "second"
    ]
    baseline_rows = [
        row for row in storage.list_records("nasa_tlx", 5000)
        if str(row.get("participant_id") or "") == participant_id
        and str(row.get("session_id") or "") == session_id
        and row.get("assessment_phase") == "baseline"
    ]
    baseline_cli = None
    if baseline_rows:
        try:
            baseline_cli = float(baseline_rows[-1].get("overall_score"))
        except (TypeError, ValueError):
            baseline_cli = None
    summary = build_session_cli_summary(
        user_id=participant_id,
        session_id=session_id,
        rows=tracking_rows,
        baseline_cli=baseline_cli,
        post_cli=post_cli,
        created_at=utc_now(),
    )
    storage.upsert(
        "session_cli_summary",
        SESSION_CLI_SUMMARY_FIELDS,
        summary,
        ("user_id", "session_id"),
    )
    return summary


def tracking_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely normalize a stored tracking value to float."""
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def tracking_int(
    value: Any,
    default: int = 0,
) -> int:
    """Safely normalize a stored tracking value to int."""
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def tracking_bool(value: Any) -> bool:
    """Safely normalize bool-like tracking values."""
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def tracking_mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return round(
        sum(values) / len(values),
        6,
    )


def tracking_weighted_mean(
    rows: list[dict[str, Any]],
    value_field: str,
    weight_field: str,
) -> float:
    weighted_total = 0.0
    total_weight = 0.0

    for row in rows:
        weight = tracking_float(
            row.get(weight_field)
        )
        value = tracking_float(
            row.get(value_field)
        )

        if weight <= 0:
            continue

        weighted_total += value * weight
        total_weight += weight

    if total_weight <= 0:
        return 0.0

    return round(
        weighted_total / total_weight,
        6,
    )


def tracking_dominant(
    values: list[Any],
    default: str = "unknown",
) -> str:
    counts: dict[str, int] = {}

    for value in values:
        normalized = str(
            value or ""
        ).strip().lower()

        if normalized in {
            "",
            "unknown",
            "none",
        }:
            continue

        counts[normalized] = (
            counts.get(normalized, 0) + 1
        )

    if not counts:
        return default

    return max(
        counts,
        key=counts.get,
    )


def build_question_tracking_summary(
    participant_id: str,
    question_id: str,
    task_number: int,
) -> dict[str, Any] | None:
    """Aggregate all multimodal second rows into one question-level feature row."""

    # ---------------------------------------------------------
    # ALL SECOND-LEVEL MULTIMODAL ROWS FOR THIS QUESTION
    # ---------------------------------------------------------
    rows = [
        row
        for row in storage.list_records(
            "tracking_data",
            50_000,
        )
        if (
            row.get("component_type") == "multimodal_tracking"
            and row.get("record_kind") == "second"
            and str(row.get("participant_id", "")) == participant_id
            and str(row.get("question_id", "")) == question_id
            and tracking_int(row.get("task_number")) == task_number
        )
    ]

    if not rows:
        return None

    rows.sort(
        key=lambda row: tracking_int(
            row.get("elapsed_second")
        )
    )

    # ---------------------------------------------------------
    # COMMON
    # ---------------------------------------------------------
    elapsed_seconds = [
        tracking_int(
            row.get("elapsed_second")
        )
        for row in rows
    ]

    seconds_recorded = len(rows)

    duration_seconds = (
        max(elapsed_seconds)
        if elapsed_seconds
        else 0
    )

    camera_seconds = sum(
        tracking_bool(
            row.get("camera_available")
        )
        for row in rows
    )

    behavior_seconds = sum(
        tracking_bool(
            row.get("behavior_available")
        )
        for row in rows
    )

    # ---------------------------------------------------------
    # EYE
    # ---------------------------------------------------------
    eye_rows = [
        row
        for row in rows
        if tracking_int(
            row.get("eye_sample_count")
        ) > 0
    ]

    total_eye_samples = sum(
        tracking_int(
            row.get("eye_sample_count")
        )
        for row in eye_rows
    )

    total_blink_events = sum(
        tracking_int(
            row.get("blink_events")
        )
        for row in eye_rows
    )

    total_saccades = sum(
        tracking_int(
            row.get("saccade_count")
        )
        for row in eye_rows
    )

    total_gaze_changes = sum(
        tracking_int(
            row.get("gaze_change_count")
        )
        for row in eye_rows
    )

    min_ear_values = [
        tracking_float(
            row.get("min_ear")
        )
        for row in eye_rows
    ]

    max_perclos_values = [
        tracking_float(
            row.get("max_perclos")
        )
        for row in eye_rows
    ]

    # ---------------------------------------------------------
    # FACIAL
    # ---------------------------------------------------------
    facial_rows = [
        row
        for row in rows
        if tracking_int(
            row.get("facial_sample_count")
        ) > 0
    ]

    total_facial_samples = sum(
        tracking_int(
            row.get("facial_sample_count")
        )
        for row in facial_rows
    )

    total_emotion_changes = sum(
        tracking_int(
            row.get("emotion_change_count")
        )
        for row in facial_rows
    )

    emotion_names = [
        "angry",
        "disgust",
        "fearful",
        "happy",
        "neutral",
        "sad",
        "surprised",
    ]

    # Count transitions between the dominant facial expression of consecutive
    # valid facial seconds. Missing/no-face seconds are skipped rather than
    # creating artificial expression changes.
    dominant_emotion_transition_count = 0
    previous_dominant_emotion: str | None = None

    for row in facial_rows:
        current_dominant_emotion = str(
            row.get("dominant_emotion") or ""
        ).strip().lower()

        if current_dominant_emotion not in emotion_names:
            continue

        if (
            previous_dominant_emotion is not None
            and current_dominant_emotion != previous_dominant_emotion
        ):
            dominant_emotion_transition_count += 1

        previous_dominant_emotion = current_dominant_emotion

    emotion_totals = {
        emotion: 0.0
        for emotion in emotion_names
    }

    for row in facial_rows:
        sample_count = tracking_float(
            row.get("facial_sample_count")
        )

        detected_ratio = tracking_float(
            row.get("facial_face_detected_ratio")
        )

        valid_weight = (
            sample_count * detected_ratio
        )

        for emotion in emotion_names:
            ratio = tracking_float(
                row.get(f"{emotion}_ratio")
            )

            emotion_totals[emotion] += (
                valid_weight * ratio
            )

    if any(
        value > 0
        for value in emotion_totals.values()
    ):
        dominant_emotion = max(
            emotion_totals,
            key=emotion_totals.get,
        )
    else:
        dominant_emotion = "unknown"

    emotion_total_count = sum(
        emotion_totals.values()
    )

    def question_emotion_ratio(
        emotion: str,
    ) -> float:
        if emotion_total_count <= 0:
            return 0.0

        return round(
            emotion_totals[emotion]
            / emotion_total_count,
            6,
        )

    # ---------------------------------------------------------
    # KEYBOARD
    # ---------------------------------------------------------
    total_keypresses = sum(
        tracking_int(
            row.get("keypress_count")
        )
        for row in rows
    )

    total_backspaces = sum(
        tracking_int(
            row.get("backspace_count")
        )
        for row in rows
    )

    typing_active_seconds = sum(
        tracking_bool(
            row.get("typing_active")
        )
        for row in rows
    )

    thinking_pauses = [
        tracking_float(
            row.get("thinking_pause_seconds")
        )
        for row in rows
        if tracking_float(
            row.get("thinking_pause_seconds")
        ) > 0
    ]

    # ---------------------------------------------------------
    # MOUSE
    # ---------------------------------------------------------
    total_mouse_moves = sum(
        tracking_int(
            row.get("mouse_move_count")
        )
        for row in rows
    )

    total_cursor_distance = round(
        sum(
            tracking_float(
                row.get("cursor_distance_px")
            )
            for row in rows
        ),
        2,
    )

    total_scroll_up = sum(
        tracking_int(
            row.get("scroll_up_count")
        )
        for row in rows
    )

    total_scroll_down = sum(
        tracking_int(
            row.get("scroll_down_count")
        )
        for row in rows
    )

    mouse_active_seconds = sum(
        tracking_bool(
            row.get("mouse_active")
        )
        for row in rows
    )

    # ---------------------------------------------------------
    # PROMPT
    # ---------------------------------------------------------
    prompt_rows = [
        row
        for row in rows
        if tracking_bool(
            row.get("prompt_sent")
        )
    ]

    total_prompts = max(
        (
            tracking_int(
                row.get("prompt_count_so_far")
            )
            for row in rows
        ),
        default=0,
    )

    prompt_lengths = [
        tracking_int(
            row.get("prompt_length_words")
        )
        for row in prompt_rows
        if tracking_int(
            row.get("prompt_length_words")
        ) > 0
    ]

    prompt_intervals = [
        tracking_float(
            row.get("time_since_last_prompt_seconds")
        )
        for row in prompt_rows
        if tracking_float(
            row.get("time_since_last_prompt_seconds")
        ) > 0
    ]

    positive_prompts = sum(
        str(
            row.get("prompt_sentiment", "")
        ).strip().lower() == "positive"
        for row in prompt_rows
    )

    neutral_prompts = sum(
        str(
            row.get("prompt_sentiment", "")
        ).strip().lower() == "neutral"
        for row in prompt_rows
    )

    negative_prompts = sum(
        str(
            row.get("prompt_sentiment", "")
        ).strip().lower() == "negative"
        for row in prompt_rows
    )

    time_to_first_prompt = (
        min(
            tracking_int(
                row.get("elapsed_second")
            )
            for row in prompt_rows
        )
        if prompt_rows
        else 0
    )

    # ---------------------------------------------------------
    # FINAL QUESTION-LEVEL ROW
    # ---------------------------------------------------------
    return {
        "participant_id": participant_id,
        "question_id": question_id,
        "task_number": task_number,

        "seconds_recorded": seconds_recorded,
        "duration_seconds": duration_seconds,
        "camera_available_ratio": round(
            camera_seconds / seconds_recorded,
            6,
        ),
        "behavior_available_ratio": round(
            behavior_seconds / seconds_recorded,
            6,
        ),

        # Eye
        "total_eye_samples": total_eye_samples,
        "mean_ear": tracking_weighted_mean(
            eye_rows,
            "mean_ear",
            "eye_sample_count",
        ),
        "min_ear": (
            round(min(min_ear_values), 6)
            if min_ear_values
            else 0.0
        ),
        "mean_perclos": tracking_weighted_mean(
            eye_rows,
            "mean_perclos",
            "eye_sample_count",
        ),
        "max_perclos": (
            round(max(max_perclos_values), 6)
            if max_perclos_values
            else 0.0
        ),
        "dominant_gaze": tracking_dominant([
            row.get("dominant_gaze")
            for row in eye_rows
        ]),
        "total_gaze_changes": total_gaze_changes,
        "total_saccades": total_saccades,
        # This is the true question-specific blink count. The Eye analyzer's
        # blink_count_end is session-cumulative, so it is intentionally not
        # included in question-level ML features.
        "total_blink_events": total_blink_events,
        "mean_blink_latency_ms": tracking_weighted_mean(
            eye_rows,
            "mean_blink_latency_ms",
            "eye_sample_count",
        ),
        "mean_head_pitch": tracking_weighted_mean(
            eye_rows,
            "mean_head_pitch",
            "eye_sample_count",
        ),
        "mean_head_yaw": tracking_weighted_mean(
            eye_rows,
            "mean_head_yaw",
            "eye_sample_count",
        ),
        "dominant_fatigue": tracking_dominant([
            row.get("dominant_fatigue")
            for row in eye_rows
        ]),
        "mean_tracking_confidence": tracking_weighted_mean(
            eye_rows,
            "mean_tracking_confidence",
            "eye_sample_count",
        ),

        # Facial
        "total_facial_samples": total_facial_samples,
        "face_detected_ratio": tracking_weighted_mean(
            facial_rows,
            "facial_face_detected_ratio",
            "facial_sample_count",
        ),
        "dominant_emotion": dominant_emotion,
        "mean_emotion_confidence": tracking_weighted_mean(
            facial_rows,
            "mean_emotion_confidence",
            "facial_sample_count",
        ),
        "total_emotion_changes": total_emotion_changes,
        "dominant_emotion_transition_count": (
            dominant_emotion_transition_count
        ),
        "angry_ratio": question_emotion_ratio("angry"),
        "disgust_ratio": question_emotion_ratio("disgust"),
        "fearful_ratio": question_emotion_ratio("fearful"),
        "happy_ratio": question_emotion_ratio("happy"),
        "neutral_ratio": question_emotion_ratio("neutral"),
        "sad_ratio": question_emotion_ratio("sad"),
        "surprised_ratio": question_emotion_ratio("surprised"),

        # Keyboard
        "total_keypresses": total_keypresses,
        "total_backspaces": total_backspaces,
        "backspace_rate": (
            round(
                total_backspaces / total_keypresses,
                6,
            )
            if total_keypresses
            else 0.0
        ),
        "typing_active_seconds": typing_active_seconds,
        "typing_active_ratio": round(
            typing_active_seconds / seconds_recorded,
            6,
        ),
        "thinking_pause_count": len(thinking_pauses),
        "mean_thinking_pause_seconds": tracking_mean(
            thinking_pauses
        ),
        "max_thinking_pause_seconds": (
            round(max(thinking_pauses), 6)
            if thinking_pauses
            else 0.0
        ),

        # Mouse
        "total_mouse_moves": total_mouse_moves,
        "total_cursor_distance_px": total_cursor_distance,
        "total_scroll_up": total_scroll_up,
        "total_scroll_down": total_scroll_down,
        "total_scroll_count": (
            total_scroll_up + total_scroll_down
        ),
        "mouse_active_seconds": mouse_active_seconds,
        "mouse_active_ratio": round(
            mouse_active_seconds / seconds_recorded,
            6,
        ),

        # Prompt
        "total_prompts": total_prompts,
        "prompt_active_seconds": len(prompt_rows),
        "average_prompt_length_words": tracking_mean(
            prompt_lengths
        ),
        "max_prompt_length_words": max(
            prompt_lengths,
            default=0,
        ),
        "time_to_first_prompt_seconds": time_to_first_prompt,
        "average_time_between_prompts_seconds": tracking_mean(
            prompt_intervals
        ),
        "positive_prompt_count": positive_prompts,
        "neutral_prompt_count": neutral_prompts,
        "negative_prompt_count": negative_prompts,

        "received_at": utc_now(),
    }


def persist_question_tracking_summaries(participant_id: str | None = None) -> int:
    """Materialize one Question-Level Feature row for each saved question."""
    rows = [
        row
        for row in storage.list_records("tracking_data", 100_000)
        if row.get("component_type") == "multimodal_tracking"
        and row.get("record_kind") == "second"
        and (participant_id is None or str(row.get("participant_id", "")) == participant_id)
    ]
    question_keys = {
        (
            str(row.get("participant_id", "")),
            str(row.get("question_id", "")),
            tracking_int(row.get("task_number")),
        )
        for row in rows
        if row.get("participant_id") and row.get("question_id")
    }
    saved = 0
    for saved_participant, question_id, task_number in question_keys:
        summary = build_question_tracking_summary(
            participant_id=saved_participant,
            question_id=question_id,
            task_number=task_number,
        )
        if summary:
            upsert_tracking("question_tracking", "summary", summary)
            persist_measurement_record(summary)
            saved += 1
    return saved


def persist_facial_events(
    participant_id: str,
    question_id: str,
    task_number: int,
    captured_at: str,
    events: list[dict[str, Any]],
) -> int:
    """Persist sparse, aggregated facial-expression events."""

    saved = 0
    for event in events:
        record = {
            "participant_id": participant_id,
            "question_id": str(event.get("question_id") or question_id),
            "task_number": int(event.get("task_number") or task_number),
            "captured_at": captured_at,
            "received_at": utc_now(),
        }
        for field in FACIAL_EXPRESSION_STORAGE_FIELDS:
            record[field] = event.get(field, "")
        append_tracking("facial_expression", "event", record)
        storage.upsert(
            "facial_expression",
            FACIAL_EXPRESSION_TABLE_FIELDS,
            record,
            ("participant_id", "question_id", "task_number", "elapsed_second", "captured_at"),
        )
        saved += 1
    return saved


def backfill_facial_expression_table() -> int:
    """Copy unified facial rows into the dedicated PostgreSQL table."""
    storage.ensure_table("facial_expression", FACIAL_EXPRESSION_TABLE_FIELDS)
    facial_unified_count = storage.count_records("facial_expression")
    source_count = sum(
        1
        for row in storage.list_records("tracking_data", 100_000)
        if row.get("component_type") == "facial_expression"
    )
    if source_count == facial_unified_count:
        return 0
    saved = 0
    for row in storage.list_records("tracking_data", 100_000):
        if row.get("component_type") != "facial_expression":
            continue
        storage.upsert(
            "facial_expression",
            FACIAL_EXPRESSION_TABLE_FIELDS,
            row,
            ("participant_id", "question_id", "task_number", "elapsed_second", "captured_at"),
        )
        saved += 1
    return saved


def persist_face_profiles(participant_id: str, signatures: Any, captured_at: str) -> int:
    """Store up to five anonymized face signatures for one participant."""
    if not isinstance(signatures, list):
        return 0
    with FACE_PROFILE_LOCK:
        known = FACE_PROFILE_CACHE.get(participant_id)
        if known is None:
            known = {
                str(row.get("face_signature"))
                for row in storage.list_records("face_profiles", 20)
                if row.get("participant_id") == participant_id and row.get("face_signature")
            }
            FACE_PROFILE_CACHE[participant_id] = known
        for signature in signatures:
            signature = str(signature).strip()
            if not signature or signature in known or len(known) >= 5:
                continue
            sample_number = len(known) + 1
            storage.upsert(
                "face_profiles",
                ["participant_id", "sample_number", "face_signature", "captured_at"],
                {
                    "participant_id": participant_id,
                    "sample_number": sample_number,
                    "face_signature": signature,
                    "captured_at": captured_at,
                },
                ("participant_id", "sample_number"),
            )
            known.add(signature)
    return len(known)


if not all((ADMIN_USERNAME, ADMIN_PASSWORD, HOST_USERNAME, HOST_PASSWORD)):
    raise RuntimeError(
        "ADMIN_USERNAME, ADMIN_PASSWORD, HOST_USERNAME, and HOST_PASSWORD must be set in .env"
    )


def password_digest(password: str) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=b"cognitrack-role-auth-v1", n=2**14, r=8, p=1)


def analyze_prompt_sentiment(message: str) -> str:
    """Return a deterministic sentiment label for participant prompts.

    Prompts are short and domain-specific, so phrase matching is more useful
    than relying on an external sentiment service. The score is intentionally
    conservative: questions without an emotional signal remain Neutral.
    """
    normalized = re.sub(r"[^a-z0-9']+", " ", str(message).lower()).strip()
    words = normalized.split()
    positive = {
        "good", "great", "excellent", "helpful", "easy", "clear", "thanks", "thank",
        "love", "happy", "correct", "works", "working", "solved", "confident", "appreciate",
    }
    negative = {
        "bad", "wrong", "difficult", "hard", "confused", "confusing", "unclear", "stuck",
        "hate", "angry", "frustrated", "frustrating", "fail", "failing", "error", "problem",
        "issue", "unsure", "lost", "cannot", "can't",
    }
    positive_phrases = {
        "thank you": 2, "very helpful": 2, "makes sense": 2, "works well": 2,
        "i understand": 1, "i got it": 1,
    }
    negative_phrases = {
        "do not understand": 2, "don't understand": 2, "not clear": 2,
        "not working": 2, "doesn't work": 2, "cannot solve": 2, "can't solve": 2,
        "i am confused": 2, "i'm confused": 2, "no idea": 2, "need help": 1,
    }
    score = sum(word in positive for word in words) - sum(word in negative for word in words)
    score += sum(weight for phrase, weight in positive_phrases.items() if phrase in normalized)
    score -= sum(weight for phrase, weight in negative_phrases.items() if phrase in normalized)
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"


ADMIN_PASSWORD_DIGEST = password_digest(ADMIN_PASSWORD)
HOST_PASSWORD_DIGEST = password_digest(HOST_PASSWORD)
DASHBOARD_PASSWORD_DIGEST = password_digest(DASHBOARD_PASSWORD)
ADMIN_PASSWORD = ""
HOST_PASSWORD = ""


class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class Participant(FlexibleModel):
    participant_id: str = ""
    full_name: str = Field(min_length=1, max_length=200)
    age_group: str
    domain: str
    ai_familiarity: str
    test_mode: bool = False


class QuestionStart(FlexibleModel):
    participant_id: str
    topic_id: str = ""
    question_id: str
    question_number: int = Field(default=1, ge=1)
    subquestion_number: int = Field(default=1, ge=1, le=3)
    question_part: str = "Main Question"
    llm: str
    trial_number: int = Field(default=1, ge=1)
    started_at: str


class ChatMessage(FlexibleModel):
    role: str
    content: str = Field(max_length=20_000)


class ChatRequest(FlexibleModel):
    participant_id: str
    topic_id: str = ""
    question_id: str
    task_name: str = ""
    question_number: int = Field(default=1, ge=1)
    subquestion_number: int = Field(default=1, ge=1, le=3)
    question_part: str = "Main Question"
    trial_number: int = Field(default=1, ge=1)
    provider: str = "Groq"
    elapsed_second: int = Field(default=1, ge=1)
    message: str = Field(min_length=1, max_length=20_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=50)


class AnswerSubmission(FlexibleModel):
    participant_id: str
    topic_id: str = ""
    question_id: str
    question_number: int
    subquestion_number: int = Field(default=1, ge=1, le=3)
    question_part: str = "Main Question"
    llm: str
    trial_number: int = Field(default=1, ge=1)
    answer: str = Field(max_length=20_000)
    paas_rating: int | None = None
    started_at: str
    submitted_at: str
    duration_seconds: int
    chat_history: list[dict[str, Any]] = Field(default_factory=list, max_length=50)
    interaction_summary: dict[str, Any] = Field(default_factory=dict, max_length=100)


class MeasurementParameterRequest(BaseModel):
    parameter_name: str = Field(min_length=2, max_length=63, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    display_name: str = Field(default="", max_length=120)
    data_type: str = Field(default="TEXT", max_length=30)


class CodeRunRequest(BaseModel):
    language: str = Field(default="java", max_length=20)
    source_code: str = Field(min_length=1, max_length=30_000)
    stdin: str = Field(default="", max_length=10_000)


class AssessmentStart(BaseModel):
    participant_id: str = Field(min_length=1, max_length=100)
    assessment_started_at: str
    camera_permission: str = Field(default="granted", max_length=30)
    calibration_status: str = Field(default="started", max_length=30)


class NasaTlxRatings(BaseModel):
    mental_demand: int = Field(ge=0, le=10)
    physical_demand: int = Field(ge=0, le=10)
    temporal_demand: int = Field(ge=0, le=10)
    performance: int = Field(ge=0, le=10)
    effort: int = Field(ge=0, le=10)
    frustration: int = Field(ge=0, le=10)


class BaselineNasaTlxRatings(BaseModel):
    mental_demand: int = Field(ge=0, le=10)
    physical_demand: int = Field(ge=0, le=10)
    temporal_demand: int = Field(ge=0, le=10)
    performance: int = Field(ge=0, le=10)
    effort: int = Field(ge=0, le=10)
    frustration: int = Field(ge=0, le=10)


class BaselineNasaTlxSubmission(BaseModel):
    participant_id: str = Field(min_length=1, max_length=100)
    ratings: BaselineNasaTlxRatings


class QuestionWorkloadRating(BaseModel):
    question_id: str = Field(min_length=1, max_length=100)
    question_number: int = Field(ge=1, le=3)
    rating: int = Field(ge=1, le=10)


class SessionCompletion(FlexibleModel):
    participant: Participant
    session_id: str = ""
    session_started_at: str
    session_ended_at: str
    ended_early: bool
    overall_paas_rating: int | None = Field(default=None, ge=1, le=10)
    nasa_tlx: NasaTlxRatings | None = None
    question_ratings: list[QuestionWorkloadRating] = Field(default_factory=list, max_length=3)
    answers: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    interaction_summary: dict[str, Any] = Field(default_factory=dict, max_length=100)


class VisionFrame(BaseModel):
    participant_id: str = Field(min_length=1, max_length=100)
    question_id: str = Field(default="", max_length=100)
    task_number: int = Field(default=1, ge=1)

    # Shared camera-frame synchronization fields.
    # They are optional for now so the existing frontend continues working
    # while we migrate the tracking pipeline step by step.
    frame_id: int | None = Field(default=None, ge=1)
    elapsed_ms: int | None = Field(default=None, ge=0)
    elapsed_second: int | None = Field(default=None, ge=1)

    captured_at: str
    image: str = Field(min_length=100, max_length=2_000_000)

    calibration_point: int | None = Field(default=None, ge=0, le=15)
    calibration_target_x: float | None = Field(default=None, ge=0, le=1)
    calibration_target_y: float | None = Field(default=None, ge=0, le=1)

    persist: bool = True


class FacialExpressionFrame(BaseModel):
    participant_id: str = Field(min_length=1, max_length=100)
    question_id: str = Field(min_length=1, max_length=100)
    task_number: int = Field(default=1, ge=1)

    # These fields will eventually be identical to the values
    # sent to eye tracking for the same webcam frame.
    frame_id: int = Field(ge=1)
    elapsed_ms: int = Field(ge=0)
    elapsed_second: int = Field(ge=1)

    captured_at: str
    image: str = Field(min_length=100, max_length=2_000_000)

class CameraSecondSummary(BaseModel):
    participant_id: str = Field(min_length=1, max_length=100)
    question_id: str = Field(min_length=1, max_length=100)
    task_number: int = Field(ge=1)
    elapsed_second: int = Field(ge=1)

    first_frame_id: int = Field(ge=1)
    last_frame_id: int = Field(ge=1)
    frame_count: int = Field(ge=1)
    paired_frame_count: int = Field(default=0, ge=0)

    first_captured_at: str
    last_captured_at: str

    # Eye features aggregated from the synchronized frame results.
    eye_sample_count: int = Field(default=0, ge=0)
    eye_face_detected_ratio: float = Field(default=0.0, ge=0, le=1)
    mean_ear: float = 0.0
    min_ear: float = 0.0
    max_ear: float = 0.0
    mean_perclos: float = 0.0
    max_perclos: float = 0.0
    dominant_gaze: str = Field(default="unknown", max_length=50)
    gaze_change_count: int = Field(default=0, ge=0)
    saccade_count: int = Field(default=0, ge=0)
    blink_events: int = Field(default=0, ge=0)
    blink_count_end: int = Field(default=0, ge=0)
    mean_blink_latency_ms: float = 0.0
    mean_head_pitch: float = 0.0
    mean_head_yaw: float = 0.0
    dominant_fatigue: str = Field(default="unknown", max_length=50)
    mean_tracking_confidence: float = 0.0

    # Facial-expression features aggregated from the synchronized frames.
    facial_sample_count: int = Field(default=0, ge=0)
    facial_face_detected_ratio: float = Field(default=0.0, ge=0, le=1)
    dominant_emotion: str = Field(default="unknown", max_length=50)
    mean_emotion_confidence: float = 0.0
    emotion_change_count: int = Field(default=0, ge=0)
    angry_ratio: float = Field(default=0.0, ge=0, le=1)
    disgust_ratio: float = Field(default=0.0, ge=0, le=1)
    fearful_ratio: float = Field(default=0.0, ge=0, le=1)
    happy_ratio: float = Field(default=0.0, ge=0, le=1)
    neutral_ratio: float = Field(default=0.0, ge=0, le=1)
    sad_ratio: float = Field(default=0.0, ge=0, le=1)
    surprised_ratio: float = Field(default=0.0, ge=0, le=1)


class BehaviorSecondSummary(BaseModel):
    participant_id: str = Field(min_length=1, max_length=100)
    question_id: str = Field(min_length=1, max_length=100)
    task_number: int = Field(ge=1)
    elapsed_second: int = Field(ge=1)
    captured_at: str

    # Keyboard features aggregated within this elapsed second.
    keypress_count: int = Field(default=0, ge=0)
    backspace_count: int = Field(default=0, ge=0)
    typing_active: bool = False
    thinking_pause_seconds: float = Field(default=0.0, ge=0)

    # Mouse features aggregated within this elapsed second.
    mouse_move_count: int = Field(default=0, ge=0)
    cursor_distance_px: float = Field(default=0.0, ge=0)
    scroll_up_count: int = Field(default=0, ge=0)
    scroll_down_count: int = Field(default=0, ge=0)
    scroll_count: int = Field(default=0, ge=0)
    mouse_active: bool = False

    # Prompt features aligned to the same elapsed second.
    prompt_sent: bool = False
    prompt_length_words: int = Field(default=0, ge=0)
    prompt_count_so_far: int = Field(default=0, ge=0)
    time_since_last_prompt_seconds: float = Field(default=0.0, ge=0)
    prompt_sentiment: str = Field(default="none", max_length=20)


class MultimodalSecondSummary(FlexibleModel):
    participant_id: str = Field(min_length=1, max_length=100)
    question_id: str = Field(min_length=1, max_length=100)
    task_number: int = Field(ge=1)
    elapsed_second: int = Field(ge=1)
    # Existing ML pipeline output. The backend only persists/aggregates it.
    combined_cli: float | None = None

    # Normally both summaries are present. Optional fields preserve a second
    # even if one tracking component temporarily fails.
    camera: CameraSecondSummary | None = None
    behavior: BehaviorSecondSummary | None = None


class KeyboardMeasurement(BaseModel):
    participant_id: str = Field(min_length=1, max_length=100)
    question_id: str = Field(min_length=1, max_length=100)
    question_category: str = Field(default="", max_length=100)
    backspace_count: int = Field(default=0, ge=0)
    backspace_time_seconds: float = Field(default=0, ge=0)
    thinking_pause_seconds: int = Field(default=0, ge=0)
    is_final: bool = False


class MouseCursorMeasurement(BaseModel):
    participant_id: str = Field(min_length=1, max_length=100)
    question_id: str = Field(min_length=1, max_length=100)
    question_category: str = Field(default="", max_length=100)
    scroll_up_count: int = Field(default=0, ge=0)
    scroll_down_count: int = Field(default=0, ge=0)
    scroll_timestep_count: int = Field(default=0, ge=0)
    scroll_event_timestamps: list[Any] = Field(default_factory=list, max_length=1000)
    mouse_move_count: int = Field(default=0, ge=0)
    cursor_distance_px: int = Field(default=0, ge=0)


class MonitoringHeartbeat(BaseModel):
    participant_id: str = Field(min_length=1, max_length=100)
    status: str = Field(default="active", pattern="^(active|completed|ended)$")
    question_id: str = Field(default="", max_length=100)
    question_number: int = Field(default=1, ge=1)
    subquestion_number: int = Field(default=1, ge=1)
    progress_percent: float = Field(default=0, ge=0, le=100)
    question_started: bool = False
    inactivity_seconds: int = Field(default=0, ge=0)
    session_duration_seconds: int = Field(default=0, ge=0)
    tab_switches: int = Field(default=0, ge=0)
    vision_status: str = Field(default="", max_length=200)
    fatigue: str = Field(default="unknown", max_length=20)
    captured_at: str


HOST_AGENT_INSTRUCTIONS = [
    "Dispatch every participant heartbeat to all specialist agents.",
    "Keep each specialist within its assigned monitoring responsibility.",
    "Combine findings into an explainable participant status for the administrator.",
    "Escalate warnings for human review; never make assessment or disciplinary decisions.",
]

SPECIALIST_AGENTS = {
    "progress_agent": "Track current question, completion progress, and whether work has started.",
    "activity_agent": "Detect inactivity and repeated tab switching without reading key values or answer text.",
    "technical_agent": "Monitor camera, face telemetry, and connection-quality signals as technical conditions.",
    "wellbeing_agent": "Report high fatigue signals as wellbeing observations requiring human review.",
    "summary_agent": "Summarize the other agents' findings without scoring or judging answer quality.",
}


def run_specialist_agents(payload: MonitoringHeartbeat) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    alerts: list[dict[str, str]] = []
    reports: list[dict[str, Any]] = []
    progress_findings = [f"Progress is {payload.progress_percent:.1f}% at {payload.question_id or 'pre-test' }."]
    if payload.status == "active" and not payload.question_started:
        progress_findings.append("The current question has not been started yet.")
    reports.append({"agent": "progress_agent", "status": "observing", "instruction": SPECIALIST_AGENTS["progress_agent"], "findings": progress_findings})

    activity_findings: list[str] = []
    if payload.status == "active" and payload.inactivity_seconds >= 60:
        message = f"No interaction for {payload.inactivity_seconds} seconds."
        alerts.append({"level": "warning", "code": "inactive", "agent": "activity_agent", "message": message})
        activity_findings.append(message)
    if payload.tab_switches >= 3:
        message = f"Tab switched {payload.tab_switches} times; human review recommended."
        alerts.append({"level": "warning", "code": "tab_switches", "agent": "activity_agent", "message": message})
        activity_findings.append(message)
    reports.append({"agent": "activity_agent", "status": "attention" if activity_findings else "clear", "instruction": SPECIALIST_AGENTS["activity_agent"], "findings": activity_findings or ["No activity alert."]})

    vision = payload.vision_status.lower()
    technical_findings: list[str] = []
    if any(term in vision for term in ("unavailable", "no face", "requires https", "paused")):
        message = "Camera/face telemetry is unavailable; check technical conditions."
        alerts.append({"level": "info", "code": "vision_unavailable", "agent": "technical_agent", "message": message})
        technical_findings.append(message)
    reports.append({"agent": "technical_agent", "status": "attention" if technical_findings else "clear", "instruction": SPECIALIST_AGENTS["technical_agent"], "findings": technical_findings or ["Vision telemetry has no technical alert."]})

    wellbeing_findings: list[str] = []
    if payload.fatigue.lower() == "high":
        message = "High fatigue signal detected; human wellbeing check recommended."
        alerts.append({"level": "warning", "code": "high_fatigue", "agent": "wellbeing_agent", "message": message})
        wellbeing_findings.append(message)
    reports.append({"agent": "wellbeing_agent", "status": "attention" if wellbeing_findings else "clear", "instruction": SPECIALIST_AGENTS["wellbeing_agent"], "findings": wellbeing_findings or ["No wellbeing alert."]})
    reports.append({"agent": "summary_agent", "status": "attention" if alerts else "clear", "instruction": SPECIALIST_AGENTS["summary_agent"], "findings": [f"{len(alerts)} specialist alert(s) require human review." if alerts else "All specialist agents report normal monitoring conditions."]})
    return reports, alerts


class RoleLogin(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def groq_client() -> Any:
    if Groq is None:
        raise HTTPException(
            status_code=503,
            detail="Groq support is unavailable. Install backend requirements before using this provider.",
        )
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is missing. Copy .env.example to .env and add your key.",
        )
    return Groq(api_key=GROQ_API_KEY)


def json_request(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"Language model provider rejected the request: {detail[:300]}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Could not connect to the language model provider: {exc.reason}") from exc


class DuckDuckGoResultsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self.current: dict[str, str] | None = None
        self.capture: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag == "a" and "result__a" in classes and len(self.results) < 5:
            href = attributes.get("href") or ""
            parsed = urllib.parse.urlparse(href)
            target = urllib.parse.parse_qs(parsed.query).get("uddg", [href])[0]
            self.current = {"title": "", "url": target, "snippet": ""}
            self.capture = "title"
        elif self.current is not None and "result__snippet" in classes:
            self.capture = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.current is not None and self.capture == "title":
            self.capture = None
        elif self.current is not None and self.capture == "snippet" and tag in {"a", "div", "span"}:
            if self.current["title"] and self.current["url"]:
                self.results.append(self.current)
            self.current = None
            self.capture = None

    def handle_data(self, data: str) -> None:
        if self.current is not None and self.capture:
            self.current[self.capture] += data.strip() + " "


def web_search(query: str) -> list[dict[str, str]]:
    bing_url = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": query, "format": "rss"})
    bing_request = urllib.request.Request(bing_url, headers={"User-Agent": "Mozilla/5.0 CogniTrack/1.0"})
    try:
        with urllib.request.urlopen(bing_request, timeout=12) as response:
            root = ElementTree.fromstring(response.read())
        results = []
        for item in root.findall("./channel/item")[:4]:
            title = (item.findtext("title") or "").strip()
            url = (item.findtext("link") or "").strip()
            snippet = re.sub(r"<[^>]+>", " ", item.findtext("description") or "")
            if title and url:
                results.append({"title": title, "url": url, "snippet": " ".join(snippet.split())})
        if results:
            return results
    except (ElementTree.ParseError, urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        pass

    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 CogniTrack/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            parser = DuckDuckGoResultsParser()
            parser.feed(response.read().decode("utf-8", errors="replace"))
            return parser.results[:4]
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return []


def messages_with_web_context(messages: list[dict[str, str]], results: list[dict[str, str]]) -> list[dict[str, str]]:
    if not results:
        return messages
    sources = "\n".join(
        f"[{index}] {item['title'].strip()}\nURL: {item['url']}\nSummary: {item['snippet'].strip()}"
        for index, item in enumerate(results, 1)
    )
    instruction = {
        "role": "system",
        "content": (
            "Use the following live web-search results when relevant. Cite factual web claims inline "
            "with [1], [2], etc., and finish with a Sources section containing the corresponding URLs. "
            "Do not claim you opened pages beyond these results.\n\n" + sources
        ),
    }
    return [messages[0], instruction, *messages[1:]] if messages and messages[0]["role"] == "system" else [instruction, *messages]


def stream_llm(provider: str, messages: list[dict[str, str]]):
    normalized = provider.strip().lower()
    if normalized == "groq":
        completion = groq_client().chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=0.3, max_tokens=1200, stream=True
        )
        for part in completion:
            content = part.choices[0].delta.content
            if content:
                yield content
        return
    if normalized == "ollama":
        request = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/chat",
            data=json.dumps({
                "model": OLLAMA_MODEL, "messages": messages, "stream": True,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {
                    "temperature": 0.3,
                    "num_predict": OLLAMA_NUM_PREDICT,
                    "num_ctx": OLLAMA_NUM_CTX,
                },
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                for line in response:
                    if not line.strip():
                        continue
                    item = json.loads(line.decode("utf-8"))
                    content = item.get("message", {}).get("content", "")
                    if content:
                        yield content
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise HTTPException(status_code=502, detail=f"Ollama rejected the request: {detail[:300]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise HTTPException(status_code=502, detail=f"Could not connect to Ollama: {getattr(exc, 'reason', str(exc))}") from exc
        
        
        return
    if normalized == "chatgpt":
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({
                "model": OPENAI_MODEL, "messages": messages, "temperature": 0.3,
                "max_tokens": 1200, "stream": True,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            for line in response:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: ") or decoded == "data: [DONE]":
                    continue
                item = json.loads(decoded[6:])
                content = item.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    yield content
        return
    raise HTTPException(status_code=400, detail="Provider must be ChatGPT, Ollama, or Groq.")


def request_llm(provider: str, messages: list[dict[str, str]]) -> tuple[str, str, str]:
    normalized = provider.strip().lower()
    if normalized == "chatgpt":
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")
        result = json_request(
            "https://api.openai.com/v1/chat/completions",
            {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 1200},
            {"Authorization": f"Bearer {OPENAI_API_KEY}"},
        )
        return result["choices"][0]["message"]["content"], "ChatGPT", OPENAI_MODEL
    if normalized == "ollama":
        if not OLLAMA_BASE_URL or not OLLAMA_MODEL:
            raise HTTPException(status_code=503, detail="OLLAMA_BASE_URL and OLLAMA_MODEL must be configured.")
        result = json_request(
            f"{OLLAMA_BASE_URL}/api/chat",
            {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {
                    "temperature": 0.3,
                    "num_predict": OLLAMA_NUM_PREDICT,
                    "num_ctx": OLLAMA_NUM_CTX,
                },
            },
        )
        try:
            return result["message"]["content"], "Ollama", OLLAMA_MODEL
        except (KeyError, TypeError) as exc:
            raise HTTPException(status_code=502, detail="Ollama returned an invalid chat response.") from exc
    if normalized == "groq":
        completion = groq_client().chat.completions.create(model=GROQ_MODEL, messages=messages, temperature=0.3, max_tokens=1200)
        return completion.choices[0].message.content or "", "Groq", GROQ_MODEL
    raise HTTPException(status_code=400, detail="Provider must be ChatGPT, Ollama, or Groq.")


def authenticate_role(payload: RoleLogin, role: str) -> dict[str, Any]:
    if role == "admin":
        expected_username = ADMIN_USERNAME
        expected_password = ADMIN_PASSWORD_DIGEST
    elif role == "host":
        expected_username = HOST_USERNAME
        expected_password = HOST_PASSWORD_DIGEST
    else:
        expected_username = DASHBOARD_USERNAME
        expected_password = DASHBOARD_PASSWORD_DIGEST
    username_ok = secrets.compare_digest(payload.username, expected_username)
    password_ok = secrets.compare_digest(password_digest(payload.password), expected_password)
    if not (username_ok and password_ok):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_TTL_MINUTES)
    AUTH_TOKENS[token] = (role, expires_at)
    return {
        "success": True,
        "role": role,
        "access_token": token,
        "token_type": "bearer",
        "expires_at": expires_at.isoformat(),
    }


def prune_expired_auth_tokens() -> None:
    now = datetime.now(timezone.utc)
    with AUTH_LOCK:
        expired = [token for token, (_, expires_at) in AUTH_TOKENS.items() if now >= expires_at]
        for token in expired:
            AUTH_TOKENS.pop(token, None)


def login_key(request: Request, username: str) -> str:
    client_host = request.client.host if request.client else "unknown"
    return f"{client_host}:{username.strip().lower()}"


def login_is_rate_limited(key: str) -> bool:
    now = time.monotonic()
    with AUTH_LOCK:
        attempts = [timestamp for timestamp in LOGIN_FAILURES.get(key, []) if now - timestamp < LOGIN_FAILURE_WINDOW_SECONDS]
        LOGIN_FAILURES[key] = attempts
        return len(attempts) >= LOGIN_FAILURE_LIMIT


def record_login_failure(key: str) -> None:
    now = time.monotonic()
    with AUTH_LOCK:
        attempts = [timestamp for timestamp in LOGIN_FAILURES.get(key, []) if now - timestamp < LOGIN_FAILURE_WINDOW_SECONDS]
        attempts.append(now)
        LOGIN_FAILURES[key] = attempts


def clear_login_failures(key: str) -> None:
    with AUTH_LOCK:
        LOGIN_FAILURES.pop(key, None)


def authenticated_role(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    prune_expired_auth_tokens()
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    token_data = AUTH_TOKENS.get(credentials.credentials)
    if token_data is None:
        raise HTTPException(status_code=401, detail="Invalid access token")
    role, expires_at = token_data
    if datetime.now(timezone.utc) >= expires_at:
        AUTH_TOKENS.pop(credentials.credentials, None)
        raise HTTPException(status_code=401, detail="Access token expired")
    return role


def create_participant_token(participant_id: str, expires_at: datetime) -> str:
    payload = json.dumps(
        {"participant_id": participant_id, "exp": int(expires_at.timestamp())},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).rstrip(b"=")
    signature = hmac.new(PARTICIPANT_TOKEN_SECRET.encode("utf-8"), encoded, hashlib.sha256).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=")
    return f"{encoded.decode()}.{encoded_signature.decode()}"


def authenticated_participant(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Participant session token required")
    try:
        encoded, encoded_signature = credentials.credentials.split(".", 1)
        expected_signature = hmac.new(
            PARTICIPANT_TOKEN_SECRET.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
        ).digest()
        supplied_signature = base64.urlsafe_b64decode(encoded_signature + "=" * (-len(encoded_signature) % 4))
        if not hmac.compare_digest(expected_signature, supplied_signature):
            raise ValueError("signature mismatch")
        payload_bytes = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        payload = json.loads(payload_bytes.decode("utf-8"))
        participant_id = str(payload["participant_id"])
        expires_at = datetime.fromtimestamp(int(payload["exp"]), timezone.utc)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=401, detail="Invalid participant session token")
    if datetime.now(timezone.utc) >= expires_at:
        raise HTTPException(status_code=401, detail="Participant session token expired")
    return participant_id


def active_participant(
    participant_id: str = Depends(authenticated_participant),
) -> str:
    """Require a valid token whose participant session has not ended."""
    with SESSION_LOCK:
        session = SESSION_STATE.get(participant_id)
        if session is None:
            if not storage.participant_exists(participant_id):
                raise HTTPException(status_code=401, detail="Invalid participant session token")
            completed = any(
                row.get("participant_id") == participant_id and row.get("event") == "completed"
                for row in storage.list_records("sessions", 50_000)
            )
            session = {
                "started_at": "",
                "status": "completed" if completed else "active",
            }
            SESSION_STATE[participant_id] = session
        if session.get("status") != "active":
            raise HTTPException(status_code=409, detail="Participant session has ended")
    return participant_id


@app.post("/api/auth/admin/login")
def admin_login(payload: RoleLogin, request: Request) -> dict[str, Any]:
    key = login_key(request, payload.username)
    if login_is_rate_limited(key):
        raise HTTPException(status_code=429, detail="Too many failed login attempts. Try again later.")
    try:
        result = authenticate_role(payload, "admin")
    except HTTPException:
        record_login_failure(key)
        raise
    clear_login_failures(key)
    return result


@app.post("/api/auth/host/login")
def host_login(payload: RoleLogin, request: Request) -> dict[str, Any]:
    key = login_key(request, payload.username)
    if login_is_rate_limited(key):
        raise HTTPException(status_code=429, detail="Too many failed login attempts. Try again later.")
    try:
        result = authenticate_role(payload, "host")
    except HTTPException:
        record_login_failure(key)
        raise
    clear_login_failures(key)
    return result


@app.post("/api/auth/dashboard/login")
def dashboard_login(payload: RoleLogin, request: Request) -> dict[str, Any]:
    key = login_key(request, payload.username)
    if login_is_rate_limited(key):
        raise HTTPException(status_code=429, detail="Too many failed login attempts. Try again later.")
    try:
        result = authenticate_role(payload, "dashboard")
    except HTTPException:
        record_login_failure(key)
        raise
    clear_login_failures(key)
    return result


@app.get("/api/auth/me")
def auth_me(role: str = Depends(authenticated_role)) -> dict[str, Any]:
    return {"success": True, "role": role}


@app.post("/api/auth/logout")
def auth_logout(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    _role: str = Depends(authenticated_role),
) -> dict[str, Any]:
    AUTH_TOKENS.pop(credentials.credentials, None)
    return {"success": True}


@app.get("/", include_in_schema=False)
def frontend_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.get("/app.js", include_in_schema=False)
def frontend_javascript() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript", headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.get("/styles.css", include_in_schema=False)
def frontend_styles() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css", headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.get("/keyboard_tracker.js", include_in_schema=False)
def keyboard_tracker_javascript() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "keyboard_tracker.js", media_type="application/javascript", headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.get("/mouse_tracker.js", include_in_schema=False)
def mouse_tracker_javascript() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "mouse_tracker.js", media_type="application/javascript", headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.get("/assets/brain.glb", include_in_schema=False)
def brain_model_asset() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "assets" / "brain.glb", media_type="model/gltf-binary", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        database_connected = storage.health_check()
    except Exception:
        database_connected = False
    return {
        "status": "ok" if database_connected else "degraded",
        "llm_providers": {
            "chatgpt_configured": bool(OPENAI_API_KEY),
            "ollama_configured": bool(OLLAMA_BASE_URL and OLLAMA_MODEL),
            "groq_configured": bool(GROQ_API_KEY),
        },
        "storage": storage.backend,
        "database_connected": database_connected,
        "timestamp": utc_now(),
    }


@app.post("/api/sessions/start")
def start_session(participant: Participant) -> dict[str, Any]:
    if not participant.participant_id:
        participant = participant.model_copy(update={"participant_id": storage.next_participant_id()})
    if storage.participant_exists(participant.participant_id):
        raise HTTPException(status_code=409, detail="Participant ID already exists")
    received_at = utc_now()
    session_id = str(uuid.uuid4())
    storage.append(
        "participants.csv",
        [
            "participant_id",
            "session_id",
            "full_name",
            "age_group",
            "domain",
            "ai_familiarity",
            "test_mode",
            "received_at",
        ],
        {**participant.model_dump(), "session_id": session_id, "received_at": received_at},
    )
    storage.append(
        "sessions.csv",
        [
            "participant_id",
            "session_id",
            "event",
            "session_started_at",
            "session_ended_at",
            "ended_early",
            "overall_paas_rating",
            "answer_count",
            "duration_seconds",
            "interaction_summary",
            "received_at",
        ],
        {
            "participant_id": participant.participant_id,
            "session_id": session_id,
            "event": "started",
            "session_started_at": received_at,
            "duration_seconds": 0,
            "received_at": received_at,
        },
    )
    with SESSION_LOCK:
        SESSION_STATE[participant.participant_id] = {
            "started_at": received_at,
            "session_id": session_id,
            "status": "active",
        }
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_TTL_MINUTES)
    token = create_participant_token(participant.participant_id, expires_at)
    return {
        "success": True,
        "participant_id": participant.participant_id,
        "session_id": session_id,
        "participant_session_token": token,
        "expires_at": expires_at.isoformat(),
    }


@app.post("/api/sessions/assessment-start")
def start_assessment(
    payload: AssessmentStart,
    authenticated_id: str = Depends(active_participant),
) -> dict[str, Any]:
    if payload.participant_id != authenticated_id:
        raise HTTPException(status_code=403, detail="Participant token does not match the requested assessment")
    storage.append(
        "assessment_starts.csv",
        [
            "participant_id",
            "assessment_started_at",
            "camera_permission",
            "calibration_status",
            "received_at",
        ],
        {**payload.model_dump(), "received_at": utc_now()},
    )
    return {"success": True, "assessment_started_at": payload.assessment_started_at}


@app.post("/api/sessions/nasa-tlx-baseline")
def save_baseline_nasa_tlx(
    payload: BaselineNasaTlxSubmission,
    authenticated_id: str = Depends(active_participant),
) -> dict[str, Any]:
    if payload.participant_id != authenticated_id:
        raise HTTPException(status_code=403, detail="Participant token does not match the baseline survey")
    ratings = payload.ratings.model_dump()
    received_at = utc_now()
    storage.append(
        "nasa_tlx",
        [
            "participant_id", "assessment_phase", "session_started_at",
            "session_ended_at", "mental_demand", "physical_demand",
            "temporal_demand", "performance", "effort", "frustration",
            "overall_score", "session_id", "received_at",
        ],
        {
            "participant_id": authenticated_id,
            "session_id": SESSION_STATE.get(authenticated_id, {}).get("session_id", ""),
            "assessment_phase": "baseline",
            "session_started_at": received_at,
            "session_ended_at": "",
            **ratings,
            "overall_score": round(sum(ratings.values()) / len(ratings), 2),
            "received_at": received_at,
        },
    )
    return {"success": True, "assessment_phase": "baseline", "overall_score": round(sum(ratings.values()) / len(ratings), 2)}


@app.post("/api/questions/start")
def start_question(
    payload: QuestionStart,
    authenticated_id: str = Depends(active_participant),
) -> dict[str, Any]:
    if payload.participant_id != authenticated_id:
        raise HTTPException(status_code=403, detail="Participant token does not match the requested question")
    storage.append(
        "question_starts.csv",
        [
            "participant_id",
            "topic_id",
            "question_id",
            "task_name",
            "question_number",
            "subquestion_number",
            "question_part",
            "llm",
            "trial_number",
            "started_at",
            "received_at",
        ],
        {**payload.model_dump(), "received_at": utc_now()},
    )
    return {"success": True, "question_id": payload.question_id}


@app.post("/api/eye-tracking/frame")
def analyze_eye_tracking(
    payload: VisionFrame,
    authenticated_id: str = Depends(active_participant),
) -> dict[str, Any]:
    """Persist pupil and eye metrics."""
    if payload.participant_id != authenticated_id:
        raise HTTPException(status_code=403, detail="Participant token does not match the requested record")
    try:
        metrics = eye_tracking_analyzer.analyze_eye_frame(
            payload.participant_id, payload.image, payload.calibration_point,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Eye-tracking dependencies are not installed") from exc
    record = {
        "participant_id": payload.participant_id,
        "question_id": payload.question_id,
        "task_number": payload.task_number,
        "captured_at": payload.captured_at,
        "received_at": utc_now(),
    }
    # Keep live metrics in the response, but persist only the factors exposed
    # by the Admin Eye Tracking table. Raw pupil coordinates, calibration
    # state, and internal analyzer counters are not stored.
    for field in EYE_TRACKING_STORAGE_FIELDS:
        record[field] = metrics.get(field, "")
    # Preserve compatibility with older calibration clients without allowing
    # the current frontend to persist calibration data.
    if payload.calibration_point is not None:
        record["pupils_detected"] = metrics.get("pupils_detected", "")
        record["calibration_point"] = payload.calibration_point
        record["calibration_target_x"] = payload.calibration_target_x if payload.calibration_target_x is not None else ""
        record["calibration_target_y"] = payload.calibration_target_y if payload.calibration_target_y is not None else ""
    # Frame-level Eye metrics are returned to the browser. The synchronized
    # assessment pipeline sends persist=False so raw Eye frames are not stored;
    # they are aggregated with Facial results into one camera row per second.
    # persist=True remains available for older/calibration clients.
    if payload.persist:
        append_tracking("eye_tracking", "sample", record)

    return {"success": True, "metrics": metrics}


@app.post("/api/facial-expression/frame")
def analyze_facial_expression(
    payload: FacialExpressionFrame,
    authenticated_id: str = Depends(active_participant),
) -> dict[str, Any]:
    """Process one selected frame and persist completed aggregate events."""

    if payload.participant_id != authenticated_id:
        raise HTTPException(
            status_code=403,
            detail="Participant token does not match the requested record",
        )

    try:
        result = facial_expression_service.process_frame(
            participant_id=payload.participant_id,
            question_id=payload.question_id,
            task_number=payload.task_number,

            frame_id=payload.frame_id,
            elapsed_ms=payload.elapsed_ms,
            elapsed_second=payload.elapsed_second,

            image_data_url=payload.image,
        )
        saved_events = persist_facial_events(
            participant_id=payload.participant_id,
            question_id=payload.question_id,
            task_number=payload.task_number,
            captured_at=payload.captured_at,
            events=result.get("storage_events", []),
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Facial-expression dependencies are not installed",
        ) from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Facial-expression model is unavailable",
        ) from exc

    return {
    "success": True,

    "participant_id":
        payload.participant_id,

    "question_id":
        payload.question_id,

    "frame_id":
        payload.frame_id,

    "elapsed_ms":
        payload.elapsed_ms,

    "elapsed_second":
        payload.elapsed_second,

    "captured_at":
        payload.captured_at,

    "frame_result":
        result.get(
            "frame_result",
            {},
        ),

    "completed_second": result.get("completed_second"),

    "saved_events": saved_events,
     }


@app.post("/api/camera-tracking/second")
def save_camera_second_summary(
    payload: CameraSecondSummary,
    authenticated_id: str = Depends(active_participant),
) -> dict[str, Any]:
    """Persist one synchronized Eye + Facial summary row per elapsed second."""

    if payload.participant_id != authenticated_id:
        raise HTTPException(
            status_code=403,
            detail="Participant token does not match the requested record",
        )

    record = {
        **payload.model_dump(),
        "received_at": utc_now(),
    }

    append_tracking(
        "camera_tracking",
        "second",
        record,
    )

    return {
        "success": True,
        "participant_id": payload.participant_id,
        "question_id": payload.question_id,
        "elapsed_second": payload.elapsed_second,
        "frame_count": payload.frame_count,
    }


@app.post("/api/behavior-tracking/second")
def save_behavior_second_summary(
    payload: BehaviorSecondSummary,
    authenticated_id: str = Depends(active_participant),
) -> dict[str, Any]:
    """Persist one keyboard + mouse + prompt summary row per elapsed second."""

    if payload.participant_id != authenticated_id:
        raise HTTPException(
            status_code=403,
            detail="Participant token does not match the requested record",
        )

    record = {
        **payload.model_dump(),
        "received_at": utc_now(),
    }

    append_tracking(
        "behavior_tracking",
        "second",
        record,
    )

    return {
        "success": True,
        "participant_id": payload.participant_id,
        "question_id": payload.question_id,
        "elapsed_second": payload.elapsed_second,
    }


@app.post("/api/multimodal-tracking/second")
def save_multimodal_second_summary(
    payload: MultimodalSecondSummary,
    authenticated_id: str = Depends(active_participant),
) -> dict[str, Any]:
    """Persist one final multimodal row for an elapsed second."""

    if payload.participant_id != authenticated_id:
        raise HTTPException(
            status_code=403,
            detail="Participant token does not match the requested record",
        )

    if payload.camera is None and payload.behavior is None:
        raise HTTPException(
            status_code=400,
            detail="At least one second-level tracking summary is required",
        )

    if payload.camera is not None:
        camera = payload.camera
        if (
            camera.participant_id != payload.participant_id
            or camera.question_id != payload.question_id
            or camera.task_number != payload.task_number
            or camera.elapsed_second != payload.elapsed_second
        ):
            raise HTTPException(
                status_code=400,
                detail="Camera summary does not match the multimodal second key",
            )

    if payload.behavior is not None:
        behavior = payload.behavior
        if (
            behavior.participant_id != payload.participant_id
            or behavior.question_id != payload.question_id
            or behavior.task_number != payload.task_number
            or behavior.elapsed_second != payload.elapsed_second
        ):
            raise HTTPException(
                status_code=400,
                detail="Behavior summary does not match the multimodal second key",
            )

    record: dict[str, Any] = {
        "participant_id": payload.participant_id,
        "question_id": payload.question_id,
        "task_number": payload.task_number,
        "elapsed_second": payload.elapsed_second,
        "camera_available": payload.camera is not None,
        "behavior_available": payload.behavior is not None,
    }
    payload_values = payload.model_dump()
    for cli_field in ("combined_cli", "cli_score", "cli", "predicted_cli"):
        if payload_values.get(cli_field) is not None:
            record[cli_field] = payload_values[cli_field]

    if payload.camera is not None:
        for key, value in payload.camera.model_dump().items():
            if key not in {
                "participant_id",
                "question_id",
                "task_number",
                "elapsed_second",
            }:
                record[key] = value

    if payload.behavior is not None:
        for key, value in payload.behavior.model_dump().items():
            if key not in {
                "participant_id",
                "question_id",
                "task_number",
                "elapsed_second",
            }:
                record[key] = value

    record["received_at"] = utc_now()

    append_tracking(
        "multimodal_tracking",
        "second",
        record,
    )

    return {
        "success": True,
        "participant_id": payload.participant_id,
        "question_id": payload.question_id,
        "elapsed_second": payload.elapsed_second,
        "camera_available": payload.camera is not None,
        "behavior_available": payload.behavior is not None,
    }


@app.put("/api/keyboard")
def save_keyboard_measurement(
    payload: KeyboardMeasurement,
    authenticated_id: str = Depends(active_participant),
) -> dict[str, Any]:
    if payload.participant_id != authenticated_id:
        raise HTTPException(status_code=403, detail="Participant token does not match the requested record")
    record = payload.model_dump()
    # Remove the legacy high-volume cursor coordinate payload even if an old
    # browser sends it as an extra field.
    record.pop("cursor_samples", None)
    upsert_tracking(
        "keyboard_tracking", "summary",
        record,
    )
    return {
        "success": True,
        "participant_id": payload.participant_id,
        "question_id": payload.question_id,
    }


@app.put("/api/mouse")
def save_mouse_cursor_measurement(
    payload: MouseCursorMeasurement,
    authenticated_id: str = Depends(active_participant),
) -> dict[str, Any]:
    if payload.participant_id != authenticated_id:
        raise HTTPException(status_code=403, detail="Participant token does not match the requested record")
    record = payload.model_dump()
    upsert_tracking(
        "mouse_tracking", "summary",
        record,
    )
    return {
        "success": True,
        "participant_id": payload.participant_id,
        "question_id": payload.question_id,
    }


@app.put("/api/monitoring/heartbeat")
def monitoring_heartbeat(
    payload: MonitoringHeartbeat,
    authenticated_id: str = Depends(active_participant),
) -> dict[str, Any]:
    if payload.participant_id != authenticated_id:
        raise HTTPException(status_code=403, detail="Participant token does not match the requested heartbeat")
    agent_reports, alerts = run_specialist_agents(payload)
    watcher_state = "completed" if payload.status != "active" else "attention" if alerts else "watching"
    record = {
        **payload.model_dump(),
        "watcher_state": watcher_state,
        "alerts": alerts,
        "agent_reports": agent_reports,
        "updated_at": utc_now(),
    }
    storage.upsert(
        "participant_monitoring",
        list(record),
        record,
        ("participant_id",),
    )
    return {"success": True, "host_agent": "orchestrating", "watcher_state": watcher_state, "alerts": alerts, "agent_reports": agent_reports}


@app.get("/api/dashboard/overview")
def dashboard_overview(role: str = Depends(authenticated_role)) -> dict[str, Any]:
    participants = [
        row for row in storage.list_records("participants", 2000)
        if str(row.get("test_mode", "false")).lower() not in {"true", "1", "yes"}
    ]
    participant_ids = {str(row.get("participant_id", "")) for row in participants}
    monitoring = [row for row in storage.list_records("participant_monitoring", 2000) if str(row.get("participant_id", "")) in participant_ids]
    answers = [row for row in storage.list_records("answers", 5000) if str(row.get("participant_id", "")) in participant_ids]
    sessions = [row for row in storage.list_records("sessions", 5000) if str(row.get("participant_id", "")) in participant_ids]
    chats = [row for row in storage.list_records("chat_logs", 5000) if str(row.get("participant_id", "")) in participant_ids]
    tracking = [row for row in storage.list_records("tracking_data", 20_000) if str(row.get("participant_id", "")) in participant_ids]
    nasa_rows = storage.list_records("nasa_tlx", 5000)
    nasa_scores = []
    for row in nasa_rows:
        if str(row.get("participant_id", "")) not in participant_ids:
            continue
        try:
            nasa_scores.append(float(row.get("overall_score", 0) or 0))
        except (TypeError, ValueError):
            pass
    keyboard = [row for row in tracking if row.get("component_type") == "keyboard_tracking"]
    eye = [row for row in tracking if row.get("component_type") == "eye_tracking"]
    facial_expression = [row for row in tracking if row.get("component_type") == "facial_expression"]
    mouse_cursor = [row for row in tracking if row.get("component_type") == "mouse_tracking"]
    multimodal_seconds = [
        row for row in tracking
        if (
            row.get("component_type") == "multimodal_tracking"
            and row.get("record_kind") == "second"
        )
    ]
    question_summaries = [
        row for row in tracking
        if (
            row.get("component_type") == "question_tracking"
            and row.get("record_kind") == "summary"
        )
    ]
    participant_map = {row.get("participant_id"): row for row in participants}
    answer_counts: dict[str, int] = {}
    answer_duration_totals: dict[str, float] = {}
    paas_totals: dict[str, float] = {}
    llm_usage: dict[str, int] = {}
    for answer in answers:
        participant_id = str(answer.get("participant_id", ""))
        answer_counts[participant_id] = answer_counts.get(participant_id, 0) + 1
        try:
            answer_duration_totals[participant_id] = answer_duration_totals.get(participant_id, 0) + float(answer.get("duration_seconds", 0) or 0)
            paas_totals[participant_id] = paas_totals.get(participant_id, 0) + float(answer.get("paas_rating", 0) or 0)
        except (TypeError, ValueError):
            pass
        llm = str(answer.get("llm", "Unknown") or "Unknown")
        llm_usage[llm] = llm_usage.get(llm, 0) + 1

    live = []
    for row in monitoring:
        participant_id = str(row.get("participant_id", ""))
        profile = participant_map.get(participant_id, {})
        alerts = row.get("alerts", "[]")
        if isinstance(alerts, str):
            try:
                alerts = json.loads(alerts)
            except ValueError:
                alerts = []
        agent_reports = row.get("agent_reports", "[]")
        if isinstance(agent_reports, str):
            try:
                agent_reports = json.loads(agent_reports)
            except ValueError:
                agent_reports = []
        live.append({
            **row,
            "full_name": profile.get("full_name", "Unknown participant"),
            "domain": profile.get("domain", ""),
            "session_started_at": profile.get("received_at", row.get("captured_at", row.get("updated_at", ""))),
            "answers_submitted": answer_counts.get(participant_id, 0),
            "alerts": alerts,
            "agent_reports": agent_reports,
        })
    monitoring_by_participant = {str(row.get("participant_id", "")): row for row in monitoring}
    completed_session_by_participant: dict[str, dict[str, Any]] = {}
    for session in sessions:
        if session.get("event") == "completed":
            completed_session_by_participant[str(session.get("participant_id", ""))] = session

    def elapsed_seconds(start_value: Any, end_value: Any) -> int:
        try:
            start = datetime.fromisoformat(str(start_value).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(end_value).replace("Z", "+00:00"))
            return max(0, int((end - start).total_seconds()))
        except (TypeError, ValueError):
            return 0

    stored_results = []
    for participant in participants:
        participant_id = str(participant.get("participant_id", ""))
        count = answer_counts.get(participant_id, 0)
        monitoring_record = monitoring_by_participant.get(participant_id, {})
        completed_session = completed_session_by_participant.get(participant_id, {})
        session_start = (
            completed_session.get("session_started_at")
            or monitoring_record.get("session_started_at")
            or participant.get("received_at", "")
        )
        session_end = completed_session.get("session_ended_at") or monitoring_record.get("updated_at", "")
        monitoring_status = str(monitoring_record.get("status", "") or "").lower()
        session_status = (
            "ended"
            if completed_session or monitoring_status in {"completed", "ended"}
            else "online"
            if monitoring_status == "active"
            else "not started"
        )
        stored_duration = completed_session.get("duration_seconds")
        time_spent_seconds = (
            int(stored_duration)
            if stored_duration not in (None, "")
            else elapsed_seconds(session_start, session_end)
            if session_end
            else 0
        )
        stored_results.append({
            "participant_id": participant_id,
            "full_name": participant.get("full_name", "Unknown participant"),
            "age_group": participant.get("age_group", ""),
            "domain": participant.get("domain", ""),
            "ai_familiarity": participant.get("ai_familiarity", ""),
            "session_status": session_status,
            # A durable completed session is the source of truth for progress.
            # This keeps Admin/Dashboard at 100% even if the final heartbeat
            # arrives late or is interrupted after the completion save.
            "progress_percent": 100.0 if completed_session else monitoring_record.get("progress_percent", 0),
            "answers_submitted": count,
            "time_spent_seconds": time_spent_seconds,
            "time_spent_minutes": round(time_spent_seconds / 60, 1),
        })
    return {
        "success": True,
        "role": role,
        "host_agent": {
            "name": "CogniTrack Host Agent",
            "status": "orchestrating",
            "instructions": HOST_AGENT_INSTRUCTIONS,
            "specialist_agents": [{"name": name, "instruction": instruction} for name, instruction in SPECIALIST_AGENTS.items()],
        },
        "summary": {
            "total_participants": len(participants),
            "active_participants": sum(item.get("status") == "active" for item in live),
            "participants_needing_attention": sum(item.get("watcher_state") == "attention" for item in live),
            "answers_submitted": len(answers),
            "sessions_recorded": len(sessions),
            "completed_sessions": len(completed_session_by_participant),
            "chat_messages": len(chats),
            "keyboard_records": len(keyboard),
            "eye_tracking_records": len(eye),
            "facial_expression_records": len(facial_expression),
            "mouse_cursor_records": len(mouse_cursor),
            "multimodal_second_rows": len(multimodal_seconds),
            "question_summary_rows": len(question_summaries),
            "completion_rate_percent": round(len(completed_session_by_participant) / len(participants) * 100, 1) if participants else 0,
            "average_nasa_tlx": round(sum(nasa_scores) / len(nasa_scores), 1) if nasa_scores else 0,
            "tracking_availability_percent": round((len(multimodal_seconds) + len(eye) + len(facial_expression)) / max(1, len(answers)) * 100, 1),
        },
        "participants": live if role == "admin" else [
            {
                "participant_id": item.get("participant_id", ""),
                "question_id": item.get("question_id", ""),
                "status": item.get("status", ""),
                "progress_percent": item.get("progress_percent", 0),
                "session_duration_seconds": item.get("session_duration_seconds", 0),
                "watcher_state": item.get("watcher_state", "watching"),
            }
            for item in live
        ],
        "stored_results": stored_results if role in {"admin", "dashboard"} else [],
        "analytics": {
            "llm_usage": llm_usage,
            "component_totals": {
                "Eye tracking": len(eye),
                "Facial expression": len(facial_expression),
                "Mouse & cursor tracking": len(mouse_cursor),
                "Keyboard tracking": len(keyboard),
                "Multimodal seconds": len(multimodal_seconds),
                "Question summaries": len(question_summaries),
            },
        } if role in {"admin", "dashboard"} else {},
    }


@app.get("/api/dashboard/group/feature-importances")
def group_feature_importances(role: str = Depends(authenticated_role)) -> dict[str, Any]:
    """Return the cached Random Forest feature-importance chart data for Group User."""
    if role not in {"dashboard", "admin"}:
        raise HTTPException(status_code=403, detail="Dashboard access required")
    global FEATURE_IMPORTANCE_CACHE
    with FEATURE_IMPORTANCE_LOCK:
        if FEATURE_IMPORTANCE_CACHE is None:
            rows = storage.list_records("final_dataset", 100_000)
            FEATURE_IMPORTANCE_CACHE = calculate_feature_importances(rows, top_n=20)
        return {"success": True, **FEATURE_IMPORTANCE_CACHE}


@app.get("/api/dashboard/group/model-accuracies")
def group_model_accuracies(role: str = Depends(authenticated_role)) -> dict[str, Any]:
    """Return cached held-out accuracies calculated using the supplied notebook pipeline."""
    if role not in {"dashboard", "admin"}:
        raise HTTPException(status_code=403, detail="Dashboard access required")
    global MODEL_ACCURACY_CACHE
    with MODEL_ACCURACY_LOCK:
        if MODEL_ACCURACY_CACHE is None:
            rows = storage.list_records("final_dataset", 100_000)
            rows = sorted(rows, key=lambda row: str(row.get("id", "")))
            MODEL_ACCURACY_CACHE = calculate_model_accuracies(rows)
        return {"success": True, **MODEL_ACCURACY_CACHE}


@app.get("/api/dashboard/ml-analytics-legacy")
def dashboard_ml_analytics(role: str = Depends(authenticated_role)) -> dict[str, Any]:
    """Return the verified ML/EDA results from CognitiveLoad_Final_Dataset_v2.ipynb.

    These are notebook evaluation results, not predictions generated from live
    participant rows. Keeping that distinction explicit prevents the dashboard
    from presenting offline test accuracy as real-time production accuracy.
    """
    if role not in {"admin", "dashboard"}:
        raise HTTPException(status_code=403, detail="Dashboard access required")
    return {
        "success": True,
        "source": "CognitiveLoad_Final_Dataset_v2.ipynb",
        "source_type": "offline_notebook_evaluation",
        "dataset": {
            "rows": 12000,
            "original_columns": 23,
            "features_used": 30,
            "participant_grouped_split": True,
            "target": "CLI_tier",
            "target_definition": "Low-Moderate / High / Overload",
            "class_distribution": [
                {"label": "Low-Moderate", "count": 5350, "percent": 44.6},
                {"label": "High", "count": 3814, "percent": 31.8},
                {"label": "Overload", "count": 2836, "percent": 23.6},
            ],
        },
        "best_model": {
            "name": "Stacking Ensemble",
            "accuracy": 0.7539,
            "precision_macro": 0.7378,
            "recall_macro": 0.7351,
            "f1_macro": 0.7378,
            "test_rows": 2397,
        },
        "model_comparison": [
            {"name": "Stacking Ensemble", "accuracy": 0.7539, "f1_macro": 0.7378},
            {"name": "Random Forest", "accuracy": 0.7514, "f1_macro": 0.7377},
            {"name": "Extra Trees", "accuracy": 0.7501, "f1_macro": 0.7348},
            {"name": "HistGradientBoosting", "accuracy": 0.7251, "f1_macro": 0.7141},
            {"name": "Random Forest - original 4 classes", "accuracy": 0.6529, "f1_macro": 0.6537},
        ],
        "eda": {
            "diagnostic_r2": 0.801,
            "diagnostic_label": "Linear R²: CLI_score vs behavioral features",
            "charts": [
                {"name": "PERCLOS by cognitive-load tier", "feature": "mean perclos", "kind": "boxplot"},
                {"name": "Backspace rate by cognitive-load tier", "feature": "backspace rate", "kind": "boxplot"},
                {"name": "Thinking pause by cognitive-load tier", "feature": "mean thinking pause seconds", "kind": "boxplot"},
                {"name": "Typing speed by cognitive-load tier", "feature": "typing_speed", "kind": "boxplot"},
            ],
            "top_signals": [
                "mean perclos", "backspace rate", "mean thinking pause seconds",
                "typing_speed", "blink_rate", "cursor_speed",
            ],
        },
        "interpretation": "The 75.39% score is held-out offline test accuracy from the notebook. It should not be interpreted as live accuracy for current participants until live rows are scored and monitored separately.",
    }


@app.get("/api/dashboard/participant/{participant_id}")
def dashboard_participant_detail(
    participant_id: str,
    role: str = Depends(authenticated_role),
) -> dict[str, Any]:
    if role not in {"admin", "dashboard"}:
        raise HTTPException(status_code=403, detail="Dashboard access required")

    participant = next(
        (
            row
            for row in storage.list_records("participants", 50_000)
            if str(row.get("participant_id", "")) == participant_id
        ),
        None,
    )
    if participant is None:
        raise HTTPException(status_code=404, detail="Participant not found")

    sessions = [
        row for row in storage.list_records("sessions", 50_000)
        if str(row.get("participant_id", "")) == participant_id
    ]
    answers = [
        row for row in storage.list_records("answers", 50_000)
        if str(row.get("participant_id", "")) == participant_id
    ]
    tracking = [
        row for row in storage.list_records("tracking_data", 100_000)
        if str(row.get("participant_id", "")) == participant_id
    ]
    nasa_tlx = [
        row for row in storage.list_records("nasa_tlx", 100)
        if str(row.get("participant_id", "")) == participant_id
    ]
    question_ratings = [
        row for row in storage.list_records("question_ratings", 100)
        if str(row.get("participant_id", "")) == participant_id
    ]
    cli_summaries = [
        row for row in storage.list_records("session_cli_summary", 5000)
        if str(row.get("user_id") or row.get("participant_id") or "") == participant_id
    ]
    component_counts: dict[str, int] = {}
    for row in tracking:
        component = str(row.get("component_type", "unknown") or "unknown")
        component_counts[component] = component_counts.get(component, 0) + 1

    return {
        "success": True,
        "participant": participant,
        "sessions": sessions,
        "answers": answers,
        "nasa_tlx": nasa_tlx,
        "question_ratings": question_ratings,
        "session_cli_summaries": cli_summaries,
        "tracking_summary": component_counts,
    }


def build_dataset_validation_report() -> dict[str, Any]:
    """Validate the final second-level and question-level ML datasets."""

    participants = storage.list_records(
        "participants",
        50_000,
    )

    tracking_rows = storage.list_records(
        "tracking_data",
        100_000,
    )

    second_rows = [
        row
        for row in tracking_rows
        if (
            row.get("component_type") == "multimodal_tracking"
            and row.get("record_kind") == "second"
        )
    ]

    question_rows = [
        row
        for row in tracking_rows
        if (
            row.get("component_type") == "question_tracking"
            and row.get("record_kind") == "summary"
        )
    ]

    # Duplicate second key:
    # participant_id + question_id + elapsed_second.
    second_key_counts: dict[tuple[str, str, int], int] = {}

    for row in second_rows:
        key = (
            str(row.get("participant_id") or ""),
            str(row.get("question_id") or ""),
            tracking_int(row.get("elapsed_second")),
        )
        second_key_counts[key] = second_key_counts.get(key, 0) + 1

    duplicate_second_rows = sum(
        count - 1
        for count in second_key_counts.values()
        if count > 1
    )

    # One question-level row per participant + question.
    question_key_counts: dict[tuple[str, str], int] = {}

    for row in question_rows:
        key = (
            str(row.get("participant_id") or ""),
            str(row.get("question_id") or ""),
        )
        question_key_counts[key] = question_key_counts.get(key, 0) + 1

    duplicate_question_rows = sum(
        count - 1
        for count in question_key_counts.values()
        if count > 1
    )

    all_final_rows = second_rows + question_rows

    missing_participant_ids = sum(
        not str(row.get("participant_id") or "").strip()
        for row in all_final_rows
    )

    missing_question_ids = sum(
        not str(row.get("question_id") or "").strip()
        for row in all_final_rows
    )

    invalid_elapsed_seconds = sum(
        tracking_int(row.get("elapsed_second")) < 1
        for row in second_rows
    )

    second_ratio_fields = (
        "eye_face_detected_ratio",
        "facial_face_detected_ratio",
        "angry_ratio",
        "disgust_ratio",
        "fearful_ratio",
        "happy_ratio",
        "neutral_ratio",
        "sad_ratio",
        "surprised_ratio",
    )

    question_ratio_fields = (
        "camera_available_ratio",
        "behavior_available_ratio",
        "face_detected_ratio",
        "angry_ratio",
        "disgust_ratio",
        "fearful_ratio",
        "happy_ratio",
        "neutral_ratio",
        "sad_ratio",
        "surprised_ratio",
        "backspace_rate",
        "typing_active_ratio",
        "mouse_active_ratio",
    )

    invalid_ratios = 0

    for row in second_rows:
        for field in second_ratio_fields:
            value = tracking_float(row.get(field))
            if value < 0.0 or value > 1.0:
                invalid_ratios += 1

    for row in question_rows:
        for field in question_ratio_fields:
            value = tracking_float(row.get(field))
            if value < 0.0 or value > 1.0:
                invalid_ratios += 1

    # The seven facial class ratios should sum to ~1 whenever
    # at least one valid facial-expression result exists.
    emotion_ratio_fields = (
        "angry_ratio",
        "disgust_ratio",
        "fearful_ratio",
        "happy_ratio",
        "neutral_ratio",
        "sad_ratio",
        "surprised_ratio",
    )

    invalid_emotion_ratio_sums = 0

    for row in second_rows + question_rows:
        ratios = [
            tracking_float(row.get(field))
            for field in emotion_ratio_fields
        ]
        ratio_sum = sum(ratios)

        if ratio_sum > 0 and abs(ratio_sum - 1.0) > 0.05:
            invalid_emotion_ratio_sums += 1

    nonnegative_second_fields = (
        "frame_count", "paired_frame_count",
        "eye_sample_count", "gaze_change_count",
        "saccade_count", "blink_events", "blink_count_end",
        "mean_blink_latency_ms",
        "facial_sample_count", "emotion_change_count",
        "keypress_count", "backspace_count",
        "thinking_pause_seconds",
        "mouse_move_count", "cursor_distance_px",
        "scroll_up_count", "scroll_down_count", "scroll_count",
        "prompt_length_words", "prompt_count_so_far",
        "time_since_last_prompt_seconds",
    )

    nonnegative_question_fields = (
        "seconds_recorded", "duration_seconds",
        "total_eye_samples", "total_gaze_changes",
        "total_saccades", "total_blink_events",
        "mean_blink_latency_ms",
        "total_facial_samples", "total_emotion_changes",
        "dominant_emotion_transition_count",
        "total_keypresses", "total_backspaces",
        "typing_active_seconds", "thinking_pause_count",
        "mean_thinking_pause_seconds", "max_thinking_pause_seconds",
        "total_mouse_moves", "total_cursor_distance_px",
        "total_scroll_up", "total_scroll_down", "total_scroll_count",
        "mouse_active_seconds",
        "total_prompts", "prompt_active_seconds",
        "average_prompt_length_words", "max_prompt_length_words",
        "time_to_first_prompt_seconds",
        "average_time_between_prompts_seconds",
        "positive_prompt_count", "neutral_prompt_count",
        "negative_prompt_count",
    )

    negative_feature_values = 0

    for row in second_rows:
        for field in nonnegative_second_fields:
            if tracking_float(row.get(field)) < 0:
                negative_feature_values += 1

    for row in question_rows:
        for field in nonnegative_question_fields:
            if tracking_float(row.get(field)) < 0:
                negative_feature_values += 1

    # Prompt consistency:
    # no prompt -> sentiment must be none/blank;
    # prompt -> non-empty length and valid sentiment.
    invalid_prompt_rows = 0

    for row in second_rows:
        prompt_sent = tracking_bool(
            row.get("prompt_sent")
        )

        prompt_length = tracking_int(
            row.get("prompt_length_words")
        )

        sentiment = str(
            row.get("prompt_sentiment") or "none"
        ).strip().lower()

        if prompt_sent:
            if (
                prompt_length < 1
                or sentiment not in {
                    "positive",
                    "neutral",
                    "negative",
                }
            ):
                invalid_prompt_rows += 1
        elif sentiment not in {"", "none"}:
            invalid_prompt_rows += 1

    camera_missing_seconds = sum(
        not tracking_bool(row.get("camera_available"))
        for row in second_rows
    )

    behavior_missing_seconds = sum(
        not tracking_bool(row.get("behavior_available"))
        for row in second_rows
    )

    critical_error_count = sum((
        duplicate_second_rows,
        duplicate_question_rows,
        missing_participant_ids,
        missing_question_ids,
        invalid_elapsed_seconds,
        invalid_ratios,
        invalid_emotion_ratio_sums,
        negative_feature_values,
        invalid_prompt_rows,
    ))

    return {
        "status": "PASS" if critical_error_count == 0 else "FAIL",
        "participants": len(participants),
        "second_level_rows": len(second_rows),
        "question_level_rows": len(question_rows),

        "duplicate_second_rows": duplicate_second_rows,
        "duplicate_question_rows": duplicate_question_rows,
        "missing_participant_ids": missing_participant_ids,
        "missing_question_ids": missing_question_ids,
        "invalid_elapsed_seconds": invalid_elapsed_seconds,
        "invalid_ratios": invalid_ratios,
        "invalid_emotion_ratio_sums": invalid_emotion_ratio_sums,
        "negative_feature_values": negative_feature_values,
        "invalid_prompt_rows": invalid_prompt_rows,

        # Availability is useful as a warning but does not fail validation.
        "camera_missing_seconds": camera_missing_seconds,
        "behavior_missing_seconds": behavior_missing_seconds,

        "critical_error_count": critical_error_count,
    }


@app.get("/api/admin/validation")
def admin_dataset_validation(
    role: str = Depends(authenticated_role),
) -> dict[str, Any]:
    if role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Administrator access required",
        )

    return {
        "success": True,
        "validation": build_dataset_validation_report(),
    }


def build_prompt_tracking_summary() -> list[dict[str, Any]]:
    participants = {
        str(row.get("participant_id")): str(row.get("full_name") or row.get("participant_id") or "Unknown")
        for row in storage.list_records("participants", 50_000)
    }
    rows = [
        row for row in storage.list_records("chat_logs", 50_000)
        if str(row.get("role", "")).lower() == "user"
    ]
    assistants = {
        str(row.get("prompt_id")): row
        for row in storage.list_records("chat_logs", 50_000)
        if str(row.get("role", "")).lower() == "assistant" and row.get("prompt_id")
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("participant_id", "")), str(row.get("task_name") or row.get("question_id") or "Unknown task"))
        grouped.setdefault(key, []).append(row)

    summary = []
    for (participant_id, task_name), prompts in grouped.items():
        prompts.sort(key=lambda row: str(row.get("prompt_timestamp") or row.get("timestamp") or ""))
        # Recalculate from the stored prompt text so older rows created by the
        # previous tiny exact-word scorer are corrected in the Admin view.
        sentiments = [analyze_prompt_sentiment(str(row.get("message") or "")) for row in prompts]
        sentiment_counts = {label: sentiments.count(label) for label in ("Positive", "Neutral", "Negative")}
        lengths = [len(str(row.get("message") or "").split()) for row in prompts]
        successful = sum(1 for row in prompts if row.get("prompt_id") and str(row.get("prompt_id")) in assistants)
        summary.append({
            "participant_name": participants.get(participant_id, participant_id or "Unknown"),
            "task_name": task_name,
            "total_prompt_attempt": len(prompts),
            "successful_attempt_number": successful,
            "sentiment_progress": f"Positive {sentiment_counts['Positive']} | Neutral {sentiment_counts['Neutral']} | Negative {sentiment_counts['Negative']}",
            "average_prompt_length_words": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "final_prompt": str(prompts[-1].get("message") or ""),
        })
    return summary


PROMPT_TRACKING_FIELDS = (
    "participant_name",
    "task_name",
    "total_prompt_attempt",
    "successful_attempt_number",
    "sentiment_progress",
    "average_prompt_length_words",
    "final_prompt",
)

RAG_PROMPT_EVALUATION_FIELDS = (
    "participant_id", "question_id", "task_number", "elapsed_second",
    "prompt_id", "prompt", "prompt_text", "selected_provider", "prompt_sentiment",
    "retrieved_document_ids", "retrieved_context", "semantic_similarity",
    "retrieved_similarity", "query_relevance", "answer_faithfulness",
    "direction_score", "goal_direction", "direction_label", "direction",
    "feedback", "evaluation_status", "evaluator_model",
    "generated_answer", "recorded_at",
)

RAG_PIPELINE_FIELDS = (
    "pipeline_id", "prompt_id", "participant_id", "question_id", "task_number",
    "prompt", "query_embedding", "retrieved_chunk_ids", "retrieved_context",
    "retrieved_similarity", "generated_answer", "answer_faithfulness",
    "goal_direction", "direction_score", "direction", "feedback", "status",
    "recorded_at",
)

RAG_DOCUMENT_FIELDS = (
    "document_id", "exam_id", "filename", "title", "page_count", "chunk_count",
    "status", "uploaded_at", "source_path",
)

RAG_CHUNK_FIELDS = (
    "chunk_id", "document_id", "exam_id", "page_number", "chunk_index",
    "chunk_text", "embedding", "created_at",
)


def rag_chunks_for_exam(exam_id: str = "default") -> list[dict[str, Any]]:
    return [
        row for row in storage.list_records("rag_chunks", 50_000)
        if str(row.get("exam_id") or "default") == exam_id
    ]


def build_rag_context(prompt: str, exam_id: str = "default") -> dict[str, Any]:
    matches = retrieve(prompt, rag_chunks_for_exam(exam_id), limit=4)
    usable = [row for row in matches if float(row.get("similarity") or 0) >= 0.08]
    context = "\n\n".join(
        f"[Passage {index} | page {row.get('page_number') or '?'}]\n{row.get('chunk_text') or ''}"
        for index, row in enumerate(usable, 1)
    )
    top_similarity = float(usable[0].get("similarity") or 0) if usable else 0.0
    return {
        "matches": usable,
        "context": context,
        "top_similarity": round(top_similarity, 4),
        "document_ids": sorted({str(row.get("document_id")) for row in usable if row.get("document_id")}),
    }


def evaluate_rag_answer(prompt: str, answer: str, context: str, top_similarity: float) -> dict[str, Any]:
    faithfulness = round(max(0.0, cosine_similarity(embed_text(answer), embed_text(context))), 4) if context else 0.0
    direction_score, direction = classify_direction(top_similarity, faithfulness)
    feedback = (
        "The answer is grounded in the uploaded assessment knowledge base."
        if direction == "correct"
        else "The answer is partially related; compare it with the retrieved passages."
        if direction == "partial"
        else "The answer is weakly grounded; retrieve more relevant context or ask the participant to clarify."
    )
    return {
        "retrieved_similarity": round(top_similarity, 4),
        "semantic_similarity": round(top_similarity, 4),
        "query_relevance": round(top_similarity, 4),
        "answer_faithfulness": faithfulness,
        "goal_direction": float(direction_score),
        "direction_score": direction_score,
        "direction": direction,
        "direction_label": direction,
        "feedback": feedback,
    }


@app.post("/api/admin/rag/documents/upload")
async def upload_rag_document(
    request: Request,
    role: str = Depends(authenticated_role),
) -> dict[str, Any]:
    if role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    content_length = int(request.headers.get("content-length") or 0)
    if content_length > 25_000_000:
        raise HTTPException(status_code=413, detail="PDF must be smaller than 25 MB")
    payload = await request.body()
    if not payload or not payload.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Upload a valid PDF document")
    filename = (request.query_params.get("filename") or "assessment-knowledge.pdf").strip()[:200]
    document_id = f"DOC-{uuid.uuid4().hex}"
    uploaded_at = utc_now()
    try:
        from pypdf import PdfReader
        import io as pdf_io
        reader = PdfReader(pdf_io.BytesIO(payload))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not extract PDF text: {exc}") from exc
    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    for page_number, page_text in enumerate(pages, 1):
        for text_chunk in chunk_text(page_text):
            chunk_index += 1
            chunks.append({
                "chunk_id": f"{document_id}-CHUNK-{chunk_index}",
                "document_id": document_id,
                "exam_id": "default",
                "page_number": page_number,
                "chunk_index": chunk_index,
                "chunk_text": text_chunk,
                "embedding": json.dumps(embed_text(text_chunk)),
                "created_at": uploaded_at,
            })
    if not chunks:
        raise HTTPException(status_code=400, detail="The PDF contains no extractable text")
    storage.append("rag_documents", RAG_DOCUMENT_FIELDS, {
        "document_id": document_id, "exam_id": "default", "filename": filename,
        "title": Path(filename).stem, "page_count": len(pages), "chunk_count": len(chunks),
        "status": "ready", "uploaded_at": uploaded_at, "source_path": "uploaded",
    })
    for chunk in chunks:
        storage.append("rag_chunks", RAG_CHUNK_FIELDS, chunk)
    return {
        "success": True, "document_id": document_id, "filename": filename,
        "page_count": len(pages), "chunk_count": len(chunks), "status": "ready",
    }


@app.get("/api/admin/rag/documents")
def list_rag_documents(role: str = Depends(authenticated_role)) -> dict[str, Any]:
    if role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    return {"success": True, "documents": storage.list_records("rag_documents", 100)}


@app.get("/api/rag/status")
def rag_status(_: str = Depends(active_participant)) -> dict[str, Any]:
    documents = storage.list_records("rag_documents", 100)
    ready = [row for row in documents if str(row.get("status")) == "ready"]
    return {"ready": bool(ready), "documents": len(ready), "chunks": sum(int(row.get("chunk_count") or 0) for row in ready)}


def persist_prompt_tracking_summaries() -> list[dict[str, Any]]:
    """Materialize prompt summaries in PostgreSQL for Admin and CSV use."""
    summaries = build_prompt_tracking_summary()
    for summary in summaries:
        unified = unified_tracking_record("prompt_tracking", "summary", summary)
        storage.upsert(
            "tracking_data",
            list(unified),
            unified,
            ("component_type", "record_kind", "participant_name", "task_name"),
        )
    return summaries


@app.get("/api/admin/data/{dataset}")
def admin_tracking_data(
    dataset: str,
    page: int = 1,
    page_size: int = 50,
    role: str = Depends(authenticated_role),
) -> dict[str, Any]:
    if role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    config = {
        "measurement-records": ("Complete Measurement Records", "measurement_records", MEASUREMENT_RECORD_FIELDS),
        "rag-prompt-evaluations": ("RAG Prompt Evaluations", "rag_prompt_evaluations", RAG_PROMPT_EVALUATION_FIELDS),
        "rag-pipeline": ("RAG Pipeline Runs", "rag_pipeline", RAG_PIPELINE_FIELDS),
        "rag-documents": ("RAG Knowledge Documents", "rag_documents", RAG_DOCUMENT_FIELDS),
        "participants": ("Participant Profiles", "participants", ("participant_id", "full_name", "age_group", "domain", "ai_familiarity", "test_mode", "received_at")),
        "tracking": ("Unified Tracking Data", "tracking_data", ()),
        "eye-tracking": ("Eye Tracking", "eye_tracking", ("participant_id", "task_number", "captured_at", "direction", "saccade", "blinked", "blink_count", "blink_latency_ms", "ear", "perclos", "fatigue", "head_pitch", "head_yaw", "tracking_confidence")),
        "facial-expression": ("Facial Expression", "facial_expression", ("participant_id", "question_id", "task_number", "elapsed_second", "emotion", "confidence", "average_model_confidence", "valid_frames", "total_frames", "expected_frames", "record_reason", "previous_emotion", "segment_start_second", "segment_end_second", "previous_segment_start_second", "previous_segment_end_second", "emotion_changed", "tie_break_reason", "result_status", "captured_at", "received_at")),
        "keyboard": ("Keyboard Tracking", "keyboard_tracking", ("participant_id", "question_id", "question_category", "backspace_count", "backspace_time_seconds", "thinking_pause_seconds", "is_final")),
        "mouse": ("Mouse & Cursor Tracking", "mouse_tracking", ("participant_id", "question_id", "question_category", "scroll_up_count", "scroll_down_count", "scroll_timestep_count", "scroll_event_timestamps", "mouse_move_count", "cursor_distance_px")),
        "prompt-tracking": ("Prompt Tracking", "prompt_tracking", ("participant_name", "task_name", "total_prompt_attempt", "successful_attempt_number", "sentiment_progress", "average_prompt_length_words", "final_prompt")),
        "multimodal-second": (
            "Multimodal Second-Level Dataset",
            "multimodal_tracking",
            MULTIMODAL_SECOND_EXPORT_FIELDS,
        ),
        "question-features": (
            "Question-Level Feature Dataset",
            "question_tracking",
            QUESTION_LEVEL_EXPORT_FIELDS,
        ),
        "nasa-tlx": (
            "NASA-TLX Workload Ratings",
            "nasa_tlx",
            ("participant_id", "session_id", "assessment_phase", "session_started_at", "session_ended_at", "mental_demand", "physical_demand", "temporal_demand", "performance", "effort", "frustration", "overall_score", "received_at"),
        ),
        "session-cli-change": (
            "Session CLI Change Summaries",
            "session_cli_summary",
            SESSION_CLI_SUMMARY_FIELDS,
        ),
    }
    selected = config.get(dataset)
    if selected is None:
        raise HTTPException(status_code=404, detail="Unknown administrator dataset")
    title, component_type, fields = selected
    if dataset == "measurement-records":
        custom_fields = [
            field for field in storage.table_columns("measurement_records")
            if field not in MEASUREMENT_RECORD_FIELDS and field != "id"
        ]
        all_fields = [*MEASUREMENT_RECORD_FIELDS, *custom_fields]
        bounded_page_size = max(1, min(page_size, 100))
        bounded_page = max(1, page)
        rows, total = storage.list_records_page(
            "measurement_records",
            bounded_page_size,
            (bounded_page - 1) * bounded_page_size,
        )
        records = [
            {field: row.get(field, "") for field in all_fields}
            for row in rows
        ]
        return {
            "success": True,
            "dataset": dataset,
            "title": title,
            "fields": all_fields,
            "records": records,
            "pagination": {
                "page": bounded_page,
                "page_size": bounded_page_size,
                "total_records": total,
                "total_pages": max(1, (total + bounded_page_size - 1) // bounded_page_size),
            },
        }
    if dataset in {"rag-prompt-evaluations", "rag-pipeline", "rag-documents", "multimodal-second", "question-features", "facial-expression", "eye-tracking", "keyboard", "mouse", "session-cli-change"}:
        # These are tracking views. Keep filtering, counting, and paging in
        # PostgreSQL; never materialize the full tracking table in Python.
        bounded_page_size = max(1, min(page_size, 100))
        bounded_page = max(1, page)
        if dataset == "rag-prompt-evaluations":
            source_table = "rag_prompt_evaluations"
            source_component = None
            source_fields = RAG_PROMPT_EVALUATION_FIELDS
        elif dataset == "rag-pipeline":
            source_table = "rag_pipeline"
            source_component = None
            source_fields = RAG_PIPELINE_FIELDS
        elif dataset == "rag-documents":
            source_table = "rag_documents"
            source_component = None
            source_fields = RAG_DOCUMENT_FIELDS
        elif dataset == "question-features":
            source_table = "question_tracking"
            source_component = None
            source_fields = QUESTION_LEVEL_EXPORT_FIELDS
        elif dataset == "facial-expression":
            source_table = "facial_expression"
            source_component = None
            source_fields = fields
        elif dataset == "session-cli-change":
            source_table = "session_cli_summary"
            source_component = None
            source_fields = SESSION_CLI_SUMMARY_FIELDS
        else:
            source_table = "tracking_data"
            source_component = component_type
            source_fields = fields
        rows, total = storage.list_records_page(
            source_table,
            bounded_page_size,
            (bounded_page - 1) * bounded_page_size,
            component_type=source_component,
            record_kind="second" if dataset == "multimodal-second" else None,
        )
        records = [
            {field: row.get(field, "") for field in source_fields}
            for row in rows
        ]
        return {
            "success": True,
            "dataset": dataset,
            "title": title,
            "fields": list(source_fields),
            "records": records,
            "pagination": {
                "page": bounded_page,
                "page_size": bounded_page_size,
                "total_records": total,
                "total_pages": max(1, (total + bounded_page_size - 1) // bounded_page_size),
            },
        }
    if dataset == "nasa-tlx":
        records = [
            {field: row.get(field, "") for field in fields}
            for row in storage.list_records(component_type, 50_000)
        ]
        return {"success": True, "dataset": dataset, "title": title, "records": records}
    if dataset == "prompt-tracking":
        persist_prompt_tracking_summaries()
        records = [
            {field: row.get(field, "") for field in PROMPT_TRACKING_FIELDS}
            for row in storage.list_records("tracking_data", 50_000)
            if row.get("component_type") == "prompt_tracking"
        ]
        return {
            "success": True,
            "dataset": dataset,
            "title": title,
            "records": records,
        }
    records = []
    for row in storage.list_records("tracking_data", 50_000):
        if row.get("component_type") == component_type:
            records.append({field: row.get(field, "") for field in fields})
    if dataset == "participants":
        records = [
            {field: row.get(field, "") for field in fields}
            for row in storage.list_records("participants", 50_000)
        ]
        for record in records:
            digits = "".join(character for character in str(record.get("participant_id", "")) if character.isdigit())
            if digits:
                record["participant_number"] = int(digits)
        records.sort(key=lambda record: record.get("participant_number", 0))
    if dataset == "tracking":
        excluded = {"id", "tracking_key", "question_number", "left_click_count", "right_click_count", "middle_click_count", "save_reason"}
        bounded_page_size = max(1, min(page_size, 100))
        bounded_page = max(1, page)
        rows, total = storage.list_records_page(
            "tracking_data",
            bounded_page_size,
            (bounded_page - 1) * bounded_page_size,
        )
        records = [{key: value for key, value in row.items() if key not in excluded} for row in rows]
        fields = [key for key in storage.table_columns("tracking_data") if key not in excluded]
        return {
            "success": True,
            "dataset": dataset,
            "title": title,
            "fields": fields,
            "records": records,
            "pagination": {
                "page": bounded_page,
                "page_size": bounded_page_size,
                "total_records": total,
                "total_pages": max(1, (total + bounded_page_size - 1) // bounded_page_size),
            },
        }
    return {"success": True, "dataset": dataset, "title": title, "records": records}


@app.get("/api/admin/measurement-parameters")
def list_measurement_parameters(role: str = Depends(authenticated_role)) -> dict[str, Any]:
    if role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    rows = storage.list_records("measurement_parameters", 500)
    columns = [
        field for field in storage.table_columns("measurement_records")
        if field not in MEASUREMENT_RECORD_FIELDS and field != "id"
    ]
    return {"success": True, "parameters": rows, "custom_columns": columns}


@app.post("/api/admin/measurement-parameters")
def create_measurement_parameter(
    payload: MeasurementParameterRequest,
    role: str = Depends(authenticated_role),
) -> dict[str, Any]:
    if role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    allowed_types = {"TEXT", "DOUBLE PRECISION", "BOOLEAN", "TIMESTAMPTZ"}
    data_type = payload.data_type.strip().upper()
    if data_type not in allowed_types:
        raise HTTPException(status_code=422, detail="data_type must be TEXT, DOUBLE PRECISION, BOOLEAN, or TIMESTAMPTZ")
    parameter_name = payload.parameter_name.strip()
    if parameter_name in set(MEASUREMENT_RECORD_FIELDS) or parameter_name in {"id", "created_at"}:
        raise HTTPException(status_code=409, detail="That parameter already exists in the measurement table")
    existing = set(storage.table_columns("measurement_records"))
    if parameter_name in existing:
        raise HTTPException(status_code=409, detail="That parameter already exists in the measurement table")
    storage.add_column("measurement_records", parameter_name, data_type)
    storage.append(
        "measurement_parameters",
        MEASUREMENT_PARAMETER_FIELDS,
        {
            "parameter_name": parameter_name,
            "display_name": payload.display_name.strip() or parameter_name.replace("_", " ").title(),
            "data_type": data_type,
            "created_at": utc_now(),
        },
    )
    return {
        "success": True,
        "parameter_name": parameter_name,
        "data_type": data_type,
        "message": f"Parameter {parameter_name} was added to measurement_records",
    }


@app.get("/api/admin/exports/{export_name}.csv")
def download_tracking_export(
    export_name: str,
    role: str = Depends(authenticated_role),
    participant_id: str = "",
    date_from: str = "",
    date_to: str = "",
    completed: str = "",
) -> StreamingResponse:
    if role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    test_ids = {
        str(row.get("participant_id", ""))
        for row in storage.list_records("participants", 100_000)
        if str(row.get("test_mode", "false")).lower() in {"true", "1", "yes"}
    }
    test_names = {
        str(row.get("full_name", ""))
        for row in storage.list_records("participants", 100_000)
        if str(row.get("test_mode", "false")).lower() in {"true", "1", "yes"}
    }
    completed_ids = {
        str(row.get("participant_id", ""))
        for row in storage.list_records("sessions", 100_000)
        if row.get("event") == "completed"
    }

    def include_export_row(row: dict[str, Any]) -> bool:
        row_participant = str(row.get("participant_id", ""))
        if row_participant in test_ids or str(row.get("participant_name", "")) in test_names:
            return False
        if participant_id and row_participant != participant_id:
            return False
        if completed == "true" and row_participant not in completed_ids:
            return False
        if completed == "false" and row_participant in completed_ids:
            return False
        row_date = str(row.get("captured_at") or row.get("submitted_at") or row.get("received_at") or row.get("session_ended_at") or "")[:10]
        if date_from and row_date and row_date < date_from:
            return False
        if date_to and row_date and row_date > date_to:
            return False
        return True
    component_types = {
        "mouse": "mouse_tracking",
        "keyboard": "keyboard_tracking",
        "eye": "eye_tracking",
        "facial": "facial_expression",
        "multimodal-second-level": "multimodal_tracking",
        "question-level-features": "question_tracking",
    }
    direct_exports = {
        "rag-prompt-evaluations": ("rag_prompt_evaluations", RAG_PROMPT_EVALUATION_FIELDS),
        "nasa-tlx": ("nasa_tlx", ("participant_id", "assessment_phase", "session_started_at", "session_ended_at", "mental_demand", "physical_demand", "temporal_demand", "performance", "effort", "frustration", "overall_score", "received_at")),
        "session-cli-change": ("session_cli_summary", SESSION_CLI_SUMMARY_FIELDS),
    }
    if export_name in direct_exports:
        table, fields = direct_exports[export_name]
        rows = [{field: row.get(field, "") for field in fields} for row in storage.list_records(table, 100_000) if include_export_row(row)]
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        response = StreamingResponse(iter([output.getvalue()]), media_type="text/csv; charset=utf-8")
        response.headers["Content-Disposition"] = f'attachment; filename="cognitrack-{export_name}.csv"'
        return response
    component_type = component_types.get(export_name)
    if export_name == "prompt":
        persist_prompt_tracking_summaries()
        source_rows = [
            row
            for row in storage.list_records("tracking_data", 100_000)
            if row.get("component_type") == "prompt_tracking"
        ]
    elif component_type is None:
        raise HTTPException(status_code=404, detail="Unknown tracking export")
    else:
        if export_name == "facial":
            backfill_facial_expression_table()
        if export_name == "question-level-features":
            persist_question_tracking_summaries()
        source_rows = (
            storage.list_records("facial_expression", 100_000)
            if export_name == "facial"
            else [
                row
                for row in storage.list_records("tracking_data", 100_000)
                if (
                    row.get("component_type") == component_type
                    and (
                        export_name not in {
                            "multimodal-second-level",
                            "question-level-features",
                        }
                        or (
                            export_name == "multimodal-second-level"
                            and row.get("record_kind") == "second"
                        )
                        or (
                            export_name == "question-level-features"
                            and row.get("record_kind") == "summary"
                        )
                    )
                )
            ]
        )
    field_map = {
        "keyboard": ("participant_id", "question_id", "question_category", "backspace_count", "backspace_time_seconds", "thinking_pause_seconds", "is_final"),
        "eye": ("participant_id", "task_number", "captured_at", "direction", "saccade", "blinked", "blink_count", "blink_latency_ms", "ear", "perclos", "fatigue", "head_pitch", "head_yaw", "tracking_confidence"),
        "facial": ("participant_id", "question_id", "task_number", "elapsed_second", "emotion", "confidence", "average_model_confidence", "valid_frames", "total_frames", "expected_frames", "record_reason", "previous_emotion", "segment_start_second", "segment_end_second", "previous_segment_start_second", "previous_segment_end_second", "emotion_changed", "tie_break_reason", "result_status", "captured_at", "received_at"),
        "mouse": ("participant_id", "question_id", "question_category", "scroll_up_count", "scroll_down_count", "scroll_timestep_count", "scroll_event_timestamps", "mouse_move_count", "cursor_distance_px"),
        "prompt": ("participant_name", "task_name", "total_prompt_attempt", "successful_attempt_number", "sentiment_progress", "average_prompt_length_words", "final_prompt"),
        "multimodal-second-level": MULTIMODAL_SECOND_EXPORT_FIELDS,
        "question-level-features": QUESTION_LEVEL_EXPORT_FIELDS,
    }
    rows = [row for row in source_rows if include_export_row(row)]
    fields = field_map[export_name]
    rows = [{field: row.get(field, "") for field in fields} for row in rows]
    fieldnames = list(fields)
    output = io.StringIO(newline="")
    if fieldnames:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    response = StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
    )

    export_filenames = {
        "rag-prompt-evaluations": "rag_prompt_evaluations.csv",
        "multimodal-second-level": "multimodal_second_level.csv",
        "question-level-features": "question_level_features.csv",
    }

    filename = export_filenames.get(
        export_name,
        f"cognitrack-{export_name}-tracking.csv",
    )

    response.headers["Content-Disposition"] = (
        f'attachment; filename="{filename}"'
    )

    return response


@app.post("/api/llm/chat/stream")
def chat_stream(
    payload: ChatRequest,
    authenticated_id: str = Depends(active_participant),
) -> StreamingResponse:
    if payload.participant_id != authenticated_id:
        raise HTTPException(status_code=403, detail="Participant token does not match the requested chat")
    request_time = utc_now()
    prompt_id = f"PROMPT-{uuid.uuid4().hex}"
    prompt_sentiment = analyze_prompt_sentiment(payload.message)
    model_name = {"chatgpt": OPENAI_MODEL, "ollama": OLLAMA_MODEL, "groq": GROQ_MODEL}.get(
        payload.provider.lower(), "unknown"
    )
    storage.append(
        "chat_logs.csv",
        [
            "participant_id", "topic_id", "question_id", "elapsed_second", "task_name", "question_number", "subquestion_number",
            "question_part", "trial_number", "selected_provider", "prompt_id", "sentiment_analysis",
            "prompt_timestamp", "role", "message", "model", "timestamp",
        ],
        {
            "participant_id": payload.participant_id, "topic_id": payload.topic_id,
            "question_id": payload.question_id, "elapsed_second": payload.elapsed_second, "task_name": payload.task_name, "question_number": payload.question_number,
            "subquestion_number": payload.subquestion_number, "question_part": payload.question_part,
            "trial_number": payload.trial_number, "selected_provider": payload.provider,
            "prompt_id": prompt_id, "sentiment_analysis": prompt_sentiment,
            "prompt_timestamp": request_time, "role": "user", "message": payload.message,
            "model": model_name, "timestamp": request_time,
        },
    )
    # Create an evaluation ledger row immediately. A future local RAG
    # evaluator can update the pending score/context fields after retrieval.
    storage.append(
        "rag_prompt_evaluations",
        RAG_PROMPT_EVALUATION_FIELDS,
        {
            "participant_id": payload.participant_id,
            "question_id": payload.question_id,
            "task_number": payload.question_number,
            "elapsed_second": payload.elapsed_second,
            "prompt_id": prompt_id,
            "prompt": payload.message,
            "prompt_text": payload.message,
            "selected_provider": payload.provider,
            "prompt_sentiment": prompt_sentiment,
            "evaluation_status": "pending",
            "direction_label": "pending",
            "direction": "pending",
            "recorded_at": request_time,
        },
    )
    rag = build_rag_context(payload.message)
    pipeline_id = f"RAG-{uuid.uuid4().hex}"
    storage.append(
        "rag_pipeline", RAG_PIPELINE_FIELDS,
        {
            "pipeline_id": pipeline_id,
            "prompt_id": prompt_id,
            "participant_id": payload.participant_id,
            "question_id": payload.question_id,
            "task_number": payload.question_number,
            "prompt": payload.message,
            "query_embedding": json.dumps(embed_text(payload.message)),
            "retrieved_chunk_ids": [row.get("chunk_id") for row in rag["matches"]],
            "retrieved_context": rag["context"],
            "retrieved_similarity": rag["top_similarity"],
            "status": "retrieved",
            "recorded_at": request_time,
        },
    )
    messages = [{
        "role": "system",
        "content": (
            "You are an assessment tutor. Help the participant understand and solve the current task clearly. "
            "Do not fabricate facts. Keep the response structured and concise. "
            + ("Use the uploaded assessment passages as the primary source. If the passages do not answer the question, say so clearly.\n\n" + rag["context"] if rag["context"] else "No uploaded assessment passage was found for this prompt.")
        ),
    }]
    for item in payload.history[-12:]:
        if item.role in {"user", "assistant"}:
            messages.append({"role": item.role, "content": item.content})
    if messages[-1].get("content") != payload.message:
        messages.append({"role": "user", "content": payload.message})

    def event_stream():
        answer_parts: list[str] = []
        try:
            use_web_search = not rag["context"] and payload.provider.strip().lower() != "ollama"
            results: list[dict[str, str]] = []
            if use_web_search:
                yield json.dumps({"type": "status", "message": "Searching the web…"}) + "\n"
                results = web_search(payload.message)
            yield json.dumps({"type": "sources", "sources": results}) + "\n"
            enriched_messages = messages_with_web_context(messages, results)
            yield json.dumps({"type": "status", "message": f"{payload.provider} is answering…"}) + "\n"
            for chunk in stream_llm(payload.provider, enriched_messages):
                answer_parts.append(chunk)
                yield json.dumps({"type": "token", "content": chunk}) + "\n"
            answer = "".join(answer_parts)
            evaluation = evaluate_rag_answer(payload.message, answer, rag["context"], rag["top_similarity"])
            storage.update_where(
                "rag_prompt_evaluations",
                {
                    **evaluation,
                    "retrieved_document_ids": rag["document_ids"],
                    "retrieved_context": rag["context"],
                    "generated_answer": answer,
                    "evaluation_status": "completed",
                    "evaluator_model": "local-rag-heuristic-v1",
                },
                {"prompt_id": prompt_id},
            )
            storage.update_where(
                "rag_pipeline",
                {**evaluation, "generated_answer": answer, "status": "completed"},
                {"pipeline_id": pipeline_id},
            )
            storage.append(
                "chat_logs.csv",
                [
                    "participant_id", "topic_id", "question_id", "elapsed_second", "task_name", "question_number", "subquestion_number",
                    "question_part", "trial_number", "selected_provider", "prompt_id", "sentiment_analysis",
                    "prompt_timestamp", "role", "message", "model", "timestamp",
                ],
                {
                    "participant_id": payload.participant_id, "topic_id": payload.topic_id,
                    "question_id": payload.question_id, "elapsed_second": payload.elapsed_second, "task_name": payload.task_name, "question_number": payload.question_number,
                    "subquestion_number": payload.subquestion_number, "question_part": payload.question_part,
                    "trial_number": payload.trial_number, "selected_provider": payload.provider,
                    "prompt_id": prompt_id, "sentiment_analysis": "",
                    "prompt_timestamp": request_time, "role": "assistant", "message": answer,
                    "model": model_name, "timestamp": utc_now(),
                },
            )
            persist_prompt_tracking_summaries()
            yield json.dumps({"type": "done", "provider": payload.provider, "model": model_name, "rag": evaluation}) + "\n"
        except HTTPException as exc:
            storage.update_where("rag_prompt_evaluations", {"evaluation_status": "error", "feedback": str(exc.detail)}, {"prompt_id": prompt_id})
            storage.update_where("rag_pipeline", {"status": "error", "feedback": str(exc.detail)}, {"pipeline_id": pipeline_id})
            yield json.dumps({"type": "error", "message": str(exc.detail)}) + "\n"
        except Exception:
            storage.update_where("rag_prompt_evaluations", {"evaluation_status": "error", "feedback": "Language model streaming failed."}, {"prompt_id": prompt_id})
            storage.update_where("rag_pipeline", {"status": "error", "feedback": "Language model streaming failed."}, {"pipeline_id": pipeline_id})
            yield json.dumps({"type": "error", "message": "Language model streaming failed."}) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


async def compiler_json_request(request: urllib.request.Request | str) -> dict[str, Any]:
    def read_response() -> dict[str, Any]:
        with urllib.request.urlopen(request, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("The online compiler returned an invalid response.")
        return result

    return await asyncio.to_thread(read_response)


@app.post("/api/code/run")
async def run_code(
    payload: CodeRunRequest,
    participant_id: str = Depends(active_participant),
) -> dict[str, Any]:
    """Run assessment code through an external sandbox, never on this server."""
    del participant_id  # Authentication is required even though the compiler stores no participant data.
    language = payload.language.strip().lower()
    compiler_language = SUPPORTED_COMPILER_LANGUAGES.get(language)
    if not compiler_language:
        raise HTTPException(status_code=400, detail="Choose either Java or Python.")
    if not PAIZA_API_BASE_URL:
        raise HTTPException(status_code=503, detail="The online compiler is not configured.")

    request_body = {
        "source_code": payload.source_code,
        "language": compiler_language,
        "input": payload.stdin,
        "api_key": "guest",
    }
    request = urllib.request.Request(
        f"{PAIZA_API_BASE_URL}/runners/create",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        await asyncio.wait_for(COMPILER_SEMAPHORE.acquire(), timeout=0.1)
    except TimeoutError as exc:
        raise HTTPException(status_code=429, detail="The compiler is busy. Please try again shortly.") from exc

    try:
        created = await compiler_json_request(request)
        runner_id = str(created.get("id", ""))
        if not runner_id:
            raise HTTPException(status_code=502, detail="The online compiler did not create a run.")
        result: dict[str, Any] = {}
        for _ in range(12):
            await asyncio.sleep(0.5)
            query = urllib.parse.urlencode({"id": runner_id, "api_key": "guest"})
            result = await compiler_json_request(f"{PAIZA_API_BASE_URL}/runners/get_details?{query}")
            if result.get("status") == "completed":
                break
        if result.get("status") != "completed":
            raise HTTPException(status_code=504, detail="The online compiler took too long. Please try again.")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise HTTPException(status_code=502, detail=f"The online compiler rejected the request: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="The online compiler is unavailable. Please try again.") from exc
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=502, detail="The online compiler returned an invalid response.") from exc
    finally:
        COMPILER_SEMAPHORE.release()

    return {
        "success": True,
        "compile_output": f"{result.get('build_stdout', '')}{result.get('build_stderr', '')}"[:20_000],
        "output": f"{result.get('stdout', '')}{result.get('stderr', '')}"[:20_000],
        "exit_code": result.get("exit_code"),
        "signal": None,
    }


@app.post("/api/answers/submit")
def submit_answer(
    payload: AnswerSubmission,
    authenticated_id: str = Depends(active_participant),
) -> dict[str, Any]:
    if payload.participant_id != authenticated_id:
        raise HTTPException(status_code=403, detail="Participant token does not match the submitted answer")
    storage.append(
        "answers.csv",
        [
            "participant_id",
            "topic_id",
            "question_id",
            "question_number",
            "subquestion_number",
            "question_part",
            "llm",
            "trial_number",
            "answer",
            "paas_rating",
            "started_at",
            "submitted_at",
            "duration_seconds",
            "chat_history",
            "interaction_summary",
            "received_at",
        ],
        {**payload.model_dump(), "received_at": utc_now()},
    )

    question_tracking_saved = False

    try:
        question_summary = build_question_tracking_summary(
            participant_id=authenticated_id,
            question_id=payload.question_id,
            task_number=payload.question_number,
        )

        if question_summary:
            upsert_tracking(
                "question_tracking",
                "summary",
                question_summary,
            )
            persist_measurement_record(question_summary)
            question_tracking_saved = True

    except Exception:
        # The submitted answer must remain durable even if the optional
        # question-level analytics aggregation fails.
        logger.exception(
            "Question-level tracking aggregation failed for participant %s, question %s",
            authenticated_id,
            payload.question_id,
        )

    facial_events_saved = 0
    try:
        final_facial = facial_expression_service.flush_question(
            participant_id=authenticated_id,
            question_id=payload.question_id,
        )
        if final_facial:
            facial_events_saved = persist_facial_events(
                participant_id=authenticated_id,
                question_id=str(final_facial.get("question_id") or payload.question_id),
                task_number=int(final_facial.get("task_number") or payload.question_number),
                captured_at=payload.submitted_at,
                events=final_facial.get("storage_events", []),
            )
    except Exception:
        # Answer persistence must never fail because an optional camera model
        # could not finalize its last partial second.
        logger.exception(
            "Facial-expression question finalization failed for %s",
            authenticated_id,
        )

    return {
        "success": True,
        "question_id": payload.question_id,
        "facial_events_saved": facial_events_saved,
        "question_tracking_saved": question_tracking_saved,
    }


@app.post("/api/sessions/complete")
def complete_session(
    payload: SessionCompletion,
    authenticated_id: str = Depends(authenticated_participant),
) -> dict[str, Any]:
    if payload.participant.participant_id != authenticated_id:
        raise HTTPException(status_code=403, detail="Participant token does not match the completed session")
    if not payload.ended_early:
        if payload.nasa_tlx is None:
            raise HTTPException(status_code=422, detail="NASA-TLX ratings are required before completing the assessment")
    session_id = payload.session_id or str(SESSION_STATE.get(authenticated_id, {}).get("session_id") or "")
    if not session_id:
        session_id = next(
            (str(row.get("session_id")) for row in storage.list_records("sessions", 50_000)
             if str(row.get("participant_id")) == authenticated_id and row.get("event") == "started" and row.get("session_id")),
            "",
        )
    if not session_id:
        raise HTTPException(status_code=409, detail="No session ID is available for CLI analysis")
    with SESSION_LOCK:
        session = SESSION_STATE.get(authenticated_id)
        if session is None:
            # Reconstruct the in-memory state after a normal backend restart.
            # The participant and completed-session records are durable, while
            # SESSION_STATE is intentionally only a concurrency guard.
            if not storage.participant_exists(authenticated_id):
                raise HTTPException(status_code=409, detail="No active session exists for this participant")
            completed = any(
                row.get("participant_id") == authenticated_id and row.get("event") == "completed"
                for row in storage.list_records("sessions", 50_000)
            )
            if completed:
                SESSION_STATE[authenticated_id] = {"started_at": "", "status": "completed"}
                return {
                    "success": True,
                    "participant_id": authenticated_id,
                    "answers_saved": len(payload.answers),
                    "already_completed": True,
                }
            session = {"started_at": "", "status": "active"}
            SESSION_STATE[authenticated_id] = session
        if session["status"] == "completed":
            return {
                "success": True,
                "participant_id": authenticated_id,
                "answers_saved": len(payload.answers),
                "already_completed": True,
            }
        if session["status"] == "completing":
            # A prior request may have been interrupted after claiming the
            # session but before the durable completion row was written.
            # Return success if the write finished; otherwise allow retry.
            completed = any(
                row.get("participant_id") == authenticated_id and row.get("event") == "completed"
                for row in storage.list_records("sessions", 50_000)
            )
            if completed:
                session["status"] = "completed"
                return {
                    "success": True,
                    "participant_id": authenticated_id,
                    "answers_saved": len(payload.answers),
                    "already_completed": True,
                }
            session["status"] = "active"
        session["status"] = "completing"
    try:
        duration_seconds = max(
            0,
            int((
                datetime.fromisoformat(payload.session_ended_at.replace("Z", "+00:00"))
                - datetime.fromisoformat(payload.session_started_at.replace("Z", "+00:00"))
            ).total_seconds()),
        )
    except ValueError:
        duration_seconds = 0
    persist_question_tracking_summaries(authenticated_id)
    storage.append(
        "sessions.csv",
        [
            "participant_id",
            "session_id",
            "event",
            "session_started_at",
            "session_ended_at",
            "ended_early",
            "overall_paas_rating",
            "nasa_tlx",
            "question_ratings",
            "answer_count",
            "duration_seconds",
            "interaction_summary",
            "received_at",
            ],
            {
                "participant_id": payload.participant.participant_id,
                "session_id": session_id,
                "event": "completed",
                "session_started_at": payload.session_started_at,
                "session_ended_at": payload.session_ended_at,
                "ended_early": payload.ended_early,
                "overall_paas_rating": payload.overall_paas_rating,
                "nasa_tlx": payload.nasa_tlx.model_dump() if payload.nasa_tlx else {},
                "question_ratings": [item.model_dump() for item in payload.question_ratings],
                "answer_count": len(payload.answers),
                "duration_seconds": duration_seconds,
                "interaction_summary": payload.interaction_summary,
                "received_at": utc_now(),
            },
        )
    if payload.nasa_tlx:
        nasa = payload.nasa_tlx.model_dump()
        storage.append(
            "nasa_tlx.csv",
            [
                "participant_id",
                "assessment_phase",
                "session_started_at",
                "session_ended_at",
                "mental_demand",
                "physical_demand",
                "temporal_demand",
                "performance",
                "effort",
                "frustration",
                "overall_score",
                "session_id",
                "received_at",
            ],
            {
                "participant_id": authenticated_id,
                "session_id": session_id,
                "assessment_phase": "post_assessment",
                "session_started_at": payload.session_started_at,
                "session_ended_at": payload.session_ended_at,
                **nasa,
                "overall_score": round(sum(nasa.values()) / len(nasa), 2),
                "received_at": utc_now(),
            },
        )
    cli_summary = persist_session_cli_summary(authenticated_id, session_id)
    for rating in payload.question_ratings:
        storage.append(
            "question_ratings.csv",
            [
                "participant_id",
                "question_id",
                "question_number",
                "rating",
                "received_at",
            ],
            {"participant_id": authenticated_id, **rating.model_dump(), "received_at": utc_now()},
        )
    if not payload.ended_early:
        final_question_id = ""
        if payload.question_ratings:
            final_question_id = payload.question_ratings[-1].question_id
        completion_heartbeat = MonitoringHeartbeat(
            participant_id=authenticated_id,
            status="completed",
            question_id=final_question_id or "completed",
            question_number=3,
            subquestion_number=2,
            progress_percent=100,
            question_started=False,
            session_duration_seconds=duration_seconds,
            captured_at=payload.session_ended_at,
        )
        agent_reports, alerts = run_specialist_agents(completion_heartbeat)
        monitoring_record = {
            **completion_heartbeat.model_dump(),
            "watcher_state": "completed",
            "alerts": alerts,
            "agent_reports": agent_reports,
            "updated_at": utc_now(),
        }
        storage.upsert(
            "participant_monitoring",
            list(monitoring_record),
            monitoring_record,
            ("participant_id",),
        )
    facial_session_result = None
    try:
        facial_session_result = facial_expression_service.forget_participant(
            authenticated_id
        )
        if facial_session_result:
            persist_facial_events(
                participant_id=authenticated_id,
                question_id=str(facial_session_result.get("question_id") or ""),
                task_number=int(facial_session_result.get("task_number") or 1),
                captured_at=payload.session_ended_at,
                events=facial_session_result.get("storage_events", []),
            )
    except Exception:
        logger.exception(
            "Facial-expression session finalization failed for %s",
            authenticated_id,
        )

    with SESSION_LOCK:
        SESSION_STATE[authenticated_id]["status"] = "completed"
    eye_tracking_analyzer.forget_participant(authenticated_id)
    return {
        "success": True,
        "participant_id": payload.participant.participant_id,
        "session_id": session_id,
        "cli_summary": cli_summary,
        "answers_saved": len(payload.answers),
        "already_completed": False,
    }
