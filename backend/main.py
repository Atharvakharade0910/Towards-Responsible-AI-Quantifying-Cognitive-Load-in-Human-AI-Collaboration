from __future__ import annotations

import asyncio
import base64
import csv
import hmac
import io
import os
import secrets
import hashlib
import json
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

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

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
storage = create_storage(DATABASE_URL, PROJECT_ROOT / "data" / "cognitrack.sqlite3")
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
TOKEN_TTL_MINUTES = int(os.getenv("TOKEN_TTL_MINUTES", "60"))
AUTH_TOKENS: dict[str, tuple[str, datetime]] = {}
AUTH_LOCK = threading.Lock()
LOGIN_FAILURES: dict[str, list[float]] = {}
LOGIN_FAILURE_WINDOW_SECONDS = 300
LOGIN_FAILURE_LIMIT = 10
PARTICIPANT_TOKEN_SECRET = load_participant_token_secret()
SESSION_STATE: dict[str, dict[str, Any]] = {}
SESSION_LOCK = threading.Lock()
COMPILER_MAX_CONCURRENT_RUNS = 4
COMPILER_SEMAPHORE = asyncio.Semaphore(COMPILER_MAX_CONCURRENT_RUNS)
TRACKING_COMPONENT_ORDER = {
    "eye_tracking": 1,
    "mouse_tracking": 2,
    "keyboard_tracking": 3,
    "facial_expression": 4,
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
    "face_detected",
    "emotion",
    "model_confidence",
    "angry_score",
    "disgust_score",
    "fearful_score",
    "happy_score",
    "neutral_score",
    "sad_score",
    "surprised_score",
    "inference_ms",
    "processing_ms",
    "reason",
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

app = FastAPI(title="CogniTrack AI Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def unified_tracking_record(component_type: str, record_kind: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "component_sequence": TRACKING_COMPONENT_ORDER[component_type],
        "component_type": component_type,
        "record_kind": record_kind,
        **record,
    }


def append_tracking(component_type: str, record_kind: str, record: dict[str, Any]) -> None:
    unified = {
        **unified_tracking_record(component_type, record_kind, record),
    }
    storage.append("tracking_data", list(unified), unified)


def upsert_tracking(component_type: str, record_kind: str, record: dict[str, Any]) -> None:
    unified = {
        **unified_tracking_record(component_type, record_kind, record),
    }
    storage.upsert(
        "tracking_data", list(unified), unified,
        ("component_type", "record_kind", "participant_id", "question_id"),
    )


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


class CodeRunRequest(BaseModel):
    language: str = Field(default="java", max_length=20)
    source_code: str = Field(min_length=1, max_length=30_000)
    stdin: str = Field(default="", max_length=10_000)


class AssessmentStart(BaseModel):
    participant_id: str = Field(min_length=1, max_length=100)
    assessment_started_at: str
    camera_permission: str = Field(default="granted", max_length=30)
    calibration_status: str = Field(default="started", max_length=30)


class SessionCompletion(FlexibleModel):
    participant: Participant
    session_started_at: str
    session_ended_at: str
    ended_early: bool
    overall_paas_rating: int | None = Field(default=None, ge=1, le=10)
    answers: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    interaction_summary: dict[str, Any] = Field(default_factory=dict, max_length=100)


class VisionFrame(BaseModel):
    participant_id: str = Field(min_length=1, max_length=100)
    question_id: str = Field(default="", max_length=100)
    task_number: int = Field(default=1, ge=1)
    captured_at: str
    image: str = Field(min_length=100, max_length=2_000_000)
    calibration_point: int | None = Field(default=None, ge=0, le=15)
    calibration_target_x: float | None = Field(default=None, ge=0, le=1)
    calibration_target_y: float | None = Field(default=None, ge=0, le=1)
    persist: bool = True


class FacialExpressionFrame(BaseModel):
    participant_id: str = Field(min_length=1, max_length=100)
    question_id: str = Field(default="", max_length=100)
    task_number: int = Field(default=1, ge=1)
    captured_at: str
    elapsed_second: int = Field(default=1, ge=1)
    image: str = Field(min_length=100, max_length=2_000_000)
    persist: bool = True


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
        except urllib.error.URLError as exc:
            raise HTTPException(status_code=502, detail=f"Could not connect to Ollama: {exc.reason}") from exc
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
    expected_username = ADMIN_USERNAME if role == "admin" else HOST_USERNAME
    expected_password = ADMIN_PASSWORD_DIGEST if role == "admin" else HOST_PASSWORD_DIGEST
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
    storage.append(
        "participants.csv",
        [
            "participant_id",
            "full_name",
            "age_group",
            "domain",
            "ai_familiarity",
            "received_at",
        ],
        {**participant.model_dump(), "received_at": received_at},
    )
    storage.append(
        "sessions.csv",
        [
            "participant_id",
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
            "event": "started",
            "session_started_at": received_at,
            "duration_seconds": 0,
            "received_at": received_at,
        },
    )
    with SESSION_LOCK:
        SESSION_STATE[participant.participant_id] = {
            "started_at": received_at,
            "status": "active",
        }
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_TTL_MINUTES)
    token = create_participant_token(participant.participant_id, expires_at)
    return {
        "success": True,
        "participant_id": participant.participant_id,
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
    # Persist blink events immediately so a short blink is not lost between
    # the frontend's regular one-second persistence samples.
    if payload.persist or bool(metrics.get("blinked")):
        append_tracking("eye_tracking", "sample", record)

    return {"success": True, "metrics": metrics}


@app.post("/api/facial-expression/frame")
def analyze_facial_expression(
    payload: FacialExpressionFrame,
    authenticated_id: str = Depends(active_participant),
) -> dict[str, Any]:
    """Analyze and optionally persist one facial-expression frame."""
    if payload.participant_id != authenticated_id:
        raise HTTPException(status_code=403, detail="Participant token does not match the requested record")
    try:
        metrics = facial_expression_service.analyze(payload.image)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Facial-expression dependencies are not installed") from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="Facial-expression model is unavailable") from exc

    result = dict(metrics)
    scores = result.pop("scores", {})
    for label in ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"):
        result[f"{label}_score"] = scores.get(label, 0.0)
    result.update({
        "participant_id": payload.participant_id,
        "question_id": payload.question_id,
        "task_number": payload.task_number,
        "elapsed_second": payload.elapsed_second,
        "captured_at": payload.captured_at,
        "received_at": utc_now(),
    })
    if payload.persist:
        record = {field: result.get(field, "") for field in FACIAL_EXPRESSION_STORAGE_FIELDS}
        record.update({
            "participant_id": payload.participant_id,
            "question_id": payload.question_id,
            "task_number": payload.task_number,
            "elapsed_second": payload.elapsed_second,
            "captured_at": payload.captured_at,
            "received_at": result["received_at"],
        })
        append_tracking("facial_expression", "sample", record)
    return {"success": True, "metrics": result}


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
    participants = storage.list_records("participants", 2000)
    monitoring = storage.list_records("participant_monitoring", 2000)
    answers = storage.list_records("answers", 5000)
    sessions = storage.list_records("sessions", 5000)
    chats = storage.list_records("chat_logs", 5000)
    tracking = storage.list_records("tracking_data", 20_000)
    keyboard = [row for row in tracking if row.get("component_type") == "keyboard_tracking"]
    eye = [row for row in tracking if row.get("component_type") == "eye_tracking"]
    facial_expression = [row for row in tracking if row.get("component_type") == "facial_expression"]
    mouse_cursor = [row for row in tracking if row.get("component_type") == "mouse_tracking"]
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
        session_start = completed_session.get("session_started_at") or participant.get("received_at", "")
        session_end = completed_session.get("session_ended_at") or monitoring_record.get("updated_at") or utc_now()
        stored_results.append({
            "participant_id": participant_id,
            "full_name": participant.get("full_name", "Unknown participant"),
            "time_spent_seconds": elapsed_seconds(session_start, session_end),
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
        "stored_results": stored_results if role == "admin" else [],
        "analytics": {
            "llm_usage": llm_usage,
            "component_totals": {
                "Eye tracking": len(eye),
                "Facial expression": len(facial_expression),
                "Mouse & cursor tracking": len(mouse_cursor),
                "Keyboard tracking": len(keyboard),
            },
        } if role == "admin" else {},
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


@app.get("/api/admin/data/{dataset}")
def admin_tracking_data(dataset: str, role: str = Depends(authenticated_role)) -> dict[str, Any]:
    if role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    config = {
        "participants": ("Participant Profiles", "participants", ("participant_id", "full_name", "age_group", "domain", "ai_familiarity", "received_at")),
        "tracking": ("Unified Tracking Data", "tracking_data", ()),
        "eye-tracking": ("Eye Tracking", "eye_tracking", ("participant_id", "task_number", "captured_at", "direction", "saccade", "blinked", "blink_count", "blink_latency_ms", "ear", "perclos", "fatigue", "head_pitch", "head_yaw", "tracking_confidence")),
        "facial-expression": ("Facial Expression", "facial_expression", ("participant_id", "task_number", "elapsed_second", "captured_at", "face_detected", "emotion", "model_confidence", "angry_score", "disgust_score", "fearful_score", "happy_score", "neutral_score", "sad_score", "surprised_score", "inference_ms", "processing_ms", "reason")),
        "keyboard": ("Keyboard Tracking", "keyboard_tracking", ("participant_id", "question_id", "question_category", "backspace_count", "backspace_time_seconds", "thinking_pause_seconds", "is_final")),
        "mouse": ("Mouse & Cursor Tracking", "mouse_tracking", ("participant_id", "question_id", "question_category", "scroll_up_count", "scroll_down_count", "scroll_timestep_count", "scroll_event_timestamps", "mouse_move_count", "cursor_distance_px")),
        "prompt-tracking": ("Prompt Tracking", "prompt_tracking", ("participant_name", "task_name", "total_prompt_attempt", "successful_attempt_number", "sentiment_progress", "average_prompt_length_words", "final_prompt")),
    }
    selected = config.get(dataset)
    if selected is None:
        raise HTTPException(status_code=404, detail="Unknown administrator dataset")
    title, component_type, fields = selected
    if dataset == "prompt-tracking":
        return {"success": True, "dataset": dataset, "title": title, "records": build_prompt_tracking_summary()}
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
        records = [{key: value for key, value in row.items() if key not in excluded} for row in storage.list_records("tracking_data", 50_000)]
    return {"success": True, "dataset": dataset, "title": title, "records": records}


@app.get("/api/admin/exports/{export_name}.csv")
def download_tracking_export(
    export_name: str,
    role: str = Depends(authenticated_role),
) -> StreamingResponse:
    if role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    component_types = {"mouse": "mouse_tracking", "keyboard": "keyboard_tracking", "eye": "eye_tracking", "facial": "facial_expression"}
    component_type = component_types.get(export_name)
    if export_name == "prompt":
        source_rows = build_prompt_tracking_summary()
    elif component_type is None:
        raise HTTPException(status_code=404, detail="Unknown tracking export")
    else:
        source_rows = [
            row for row in storage.list_records("tracking_data", 50_000)
            if row.get("component_type") == component_type
        ]
    field_map = {
        "keyboard": ("participant_id", "question_id", "question_category", "backspace_count", "backspace_time_seconds", "thinking_pause_seconds", "is_final"),
        "eye": ("participant_id", "task_number", "captured_at", "direction", "saccade", "blinked", "blink_count", "blink_latency_ms", "ear", "perclos", "fatigue", "head_pitch", "head_yaw", "tracking_confidence"),
        "facial": ("participant_id", "task_number", "elapsed_second", "captured_at", "face_detected", "emotion", "model_confidence", "angry_score", "disgust_score", "fearful_score", "happy_score", "neutral_score", "sad_score", "surprised_score", "inference_ms", "processing_ms", "reason"),
        "mouse": ("participant_id", "question_id", "question_category", "scroll_up_count", "scroll_down_count", "scroll_timestep_count", "scroll_event_timestamps", "mouse_move_count", "cursor_distance_px"),
        "prompt": ("participant_name", "task_name", "total_prompt_attempt", "successful_attempt_number", "sentiment_progress", "average_prompt_length_words", "final_prompt"),
    }
    rows = source_rows
    fields = field_map[export_name]
    rows = [{field: row.get(field, "") for field in fields} for row in rows]
    fieldnames = list(fields)
    output = io.StringIO(newline="")
    if fieldnames:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    response = StreamingResponse(iter([output.getvalue()]), media_type="text/csv; charset=utf-8")
    response.headers["Content-Disposition"] = f'attachment; filename="cognitrack-{export_name}-tracking.csv"'
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
            "participant_id", "topic_id", "question_id", "task_name", "question_number", "subquestion_number",
            "question_part", "trial_number", "selected_provider", "prompt_id", "sentiment_analysis",
            "prompt_timestamp", "role", "message", "model", "timestamp",
        ],
        {
            "participant_id": payload.participant_id, "topic_id": payload.topic_id,
            "question_id": payload.question_id, "task_name": payload.task_name, "question_number": payload.question_number,
            "subquestion_number": payload.subquestion_number, "question_part": payload.question_part,
            "trial_number": payload.trial_number, "selected_provider": payload.provider,
            "prompt_id": prompt_id, "sentiment_analysis": prompt_sentiment,
            "prompt_timestamp": request_time, "role": "user", "message": payload.message,
            "model": model_name, "timestamp": request_time,
        },
    )
    messages = [{
        "role": "system",
        "content": (
            "You are an assessment tutor. Help the participant understand and solve the current task clearly. "
            "Do not fabricate facts. Keep the response structured and concise."
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
            use_web_search = payload.provider.strip().lower() != "ollama"
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
            storage.append(
                "chat_logs.csv",
                [
                    "participant_id", "topic_id", "question_id", "task_name", "question_number", "subquestion_number",
                    "question_part", "trial_number", "selected_provider", "prompt_id", "sentiment_analysis",
                    "prompt_timestamp", "role", "message", "model", "timestamp",
                ],
                {
                    "participant_id": payload.participant_id, "topic_id": payload.topic_id,
                    "question_id": payload.question_id, "task_name": payload.task_name, "question_number": payload.question_number,
                    "subquestion_number": payload.subquestion_number, "question_part": payload.question_part,
                    "trial_number": payload.trial_number, "selected_provider": payload.provider,
                    "prompt_id": prompt_id, "sentiment_analysis": "",
                    "prompt_timestamp": request_time, "role": "assistant", "message": answer,
                    "model": model_name, "timestamp": utc_now(),
                },
            )
            yield json.dumps({"type": "done", "provider": payload.provider, "model": model_name}) + "\n"
        except HTTPException as exc:
            yield json.dumps({"type": "error", "message": str(exc.detail)}) + "\n"
        except Exception:
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
    return {"success": True, "question_id": payload.question_id}


@app.post("/api/sessions/complete")
def complete_session(
    payload: SessionCompletion,
    authenticated_id: str = Depends(authenticated_participant),
) -> dict[str, Any]:
    if payload.participant.participant_id != authenticated_id:
        raise HTTPException(status_code=403, detail="Participant token does not match the completed session")
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
    storage.append(
        "sessions.csv",
        [
            "participant_id",
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
                "participant_id": payload.participant.participant_id,
                "event": "completed",
                "session_started_at": payload.session_started_at,
                "session_ended_at": payload.session_ended_at,
                "ended_early": payload.ended_early,
                "overall_paas_rating": payload.overall_paas_rating,
                "answer_count": len(payload.answers),
                "duration_seconds": duration_seconds,
                "interaction_summary": payload.interaction_summary,
                "received_at": utc_now(),
            },
        )
    with SESSION_LOCK:
        SESSION_STATE[authenticated_id]["status"] = "completed"
    eye_tracking_analyzer.forget_participant(authenticated_id)
    return {
        "success": True,
        "participant_id": payload.participant.participant_id,
        "answers_saved": len(payload.answers),
        "already_completed": False,
    }
