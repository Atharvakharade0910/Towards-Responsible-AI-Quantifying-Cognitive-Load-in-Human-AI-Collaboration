# CogniTrack AI

CogniTrack is organized as a FastAPI backend with a browser frontend.

## Project structure

- `frontend/` — HTML, CSS, browser JavaScript, and tracking modules.
- `backend/` — FastAPI application, PostgreSQL/SQLite storage adapter, eye-tracking modules, the separate facial-expression ZIP module, and its ONNX model.
- `backend/models/` — the facial-expression ONNX model required by the separate facial endpoint.
- `tests/` — API and analysis tests.
- `docs/` — project and testing documentation.
- `scripts/` — Windows setup and backend launch scripts.
- `data/` — local runtime data and migration backups; do not share this folder.
- `.env` — private local secrets; do not share this file.

## Current storage

Set `DATABASE_URL` to store assessment and camera-analysis data in PostgreSQL:

```env
DATABASE_URL=postgresql://cognitrack:your-password@127.0.0.1:5432/cognitrack
```

If `DATABASE_URL` is omitted, development runs fall back to
`data/cognitrack.sqlite3`. Tables are created automatically on first use when PostgreSQL is not configured.

On the included Windows/PostgreSQL 18 setup, run `./scripts/setup_postgres.ps1` from the
project folder. It securely prompts for the local `postgres` administrator password,
creates the application database and role with a generated password, and writes only
the application connection URL to the ignored `.env` file.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

Also replace both example role passwords. The backend refuses to start when any
admin or host credential is missing. Role access tokens expire after
`TOKEN_TTL_MINUTES` (60 by default).

## Run backend

```powershell
uvicorn main:app --app-dir backend --reload --host 127.0.0.1 --port 8002
```

API health check:

```text
http://127.0.0.1:8002/api/health
```

Swagger documentation:

```text
http://127.0.0.1:8002/docs
```

## Run the combined app

From the project folder, run `start.bat`. FastAPI serves both the frontend and
the API at `http://127.0.0.1:8002/`.

## Notes

- Keep `.env` private and never upload it to GitHub.
- Admin and Host credentials stay in `.env`, access tokens stay only in server memory,
  and their login, logout, dashboard views, and page activity are never persisted to
  PostgreSQL. Their dashboards are read-only views of participant assessment data.
- The Host page is a live operational view of participant progress, activity, alerts,
  prompt-writing status/length/AI-request count, and specialist-agent reports. The Admin page is a stored-results view with participant
  comparisons, tracking-coverage and LLM-usage graphs, and protected CSV downloads
  for participant, answer, eye, and keyboard data.
- ChatGPT, Ollama, and Groq are supported by `/api/llm/chat/stream`. ChatGPT and Groq require
  their corresponding API keys in `.env`; Ollama uses the local `OLLAMA_BASE_URL` and
  `OLLAMA_MODEL` settings. Run `ollama pull llama3.2` before selecting Ollama.
- Coding & Programming tasks include Java and Python editors with a Run Code button. Code is sent
  to the external Paiza.IO sandbox through `/api/code/run`; it is never executed on
  the CogniTrack server. Set `PAIZA_API_BASE_URL` in `.env` only if using a compatible
  self-hosted or alternative Paiza-compatible service.
- Browser camera frames are sampled continuously. MediaPipe eye, gaze, blink,
  fatigue, and head-pose measurements are saved in `eye_tracking`.
- The separate facial-expression endpoint uses the supplied ONNX model and saves
  one-second facial-expression samples in the `facial_expression` component.
  Its emotion scores are shown only in the Admin Facial Expression table and CSV.
- Eye tracking starts automatically after the participant clicks Start Assessment and
  grants camera permission. There is no eye-calibration screen or calibration step;
  pupil coordinates and continuing eye measurements are saved in `eye_tracking`.
  Camera permission is mandatory; non-local devices require HTTPS.
- Keyboard behavior is measured without recording actual key values or answer text.
  Idle autosaves and final submission upsert one `keyboard_tracking` row per
  participant and question, including speed, pauses, corrections, paste count,
  response latency, and answer length metrics.
- Active participants send a progress heartbeat every ten seconds. The authenticated
  Admin dashboard provides platform-wide participant, session, answer, chat, keyboard,
  vision and monitoring information.
- Host is an orchestrator agent, not a second administrator. It dispatches each
  heartbeat to Progress, Activity, Technical, Wellbeing, and Summary specialist agents,
  then consolidates their explainable findings. Agents may flag inactivity, repeated
  tab switching, missing vision telemetry, and high fatigue, but cannot score, accuse,
  reject, terminate, or penalize participants.
- Camera access from another device requires HTTPS; browsers allow HTTP camera access
  only on localhost. Assessment entry still works when camera permission is unavailable.

## Tests

From the project folder:

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -v
```
