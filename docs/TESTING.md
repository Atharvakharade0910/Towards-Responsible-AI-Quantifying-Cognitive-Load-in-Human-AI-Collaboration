# CogniTrack AI – Validation Guide

## Automated backend tests

From the project root:

```powershell
cd backend
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The suite validates role authentication, participant signed tokens, protected tracking routes, eye-metric persistence, keyboard summaries, mouse summaries, monitoring dashboards, LLM request handling, code execution integration, and database schema evolution.

## JavaScript syntax validation

When Node.js is installed:

```powershell
node --check frontend\app.js
node --check frontend\keyboard_tracker.js
node --check frontend\mouse_tracker.js
```

## Manual runtime checks

1. Copy the project-root `.env.example` to `.env` and replace all placeholder credentials.
2. Start the backend and frontend using the supplied scripts.
3. Grant browser camera permission.
4. Confirm automatic eye baseline calibration completes while the participant keeps their face visible, then verify eye samples appear in the Admin tracking view.
5. Type and use Backspace, then confirm the keyboard summary contains `backspace_count` and `thinking_pause_seconds`.
6. Scroll up and down, then confirm the mouse summary contains the correct direction counts and timestamps.
7. Confirm Host monitoring updates while the participant session is active.
8. Test the configured LLM provider separately; external providers require valid keys or a running local Ollama service.

## Mandatory webcam and automatic eye-baseline flow

1. Sign in as a participant and open the instructions page.
2. Click **Start Assessment**.
3. The browser immediately requests webcam permission. The assessment does not open if permission is denied.
4. Keep the face visible for the automatic baseline collection; there is no 16-point calibration screen in the current frontend.
5. Confirm the assessment page opens and eye tracking reports live metrics.
6. Verify the database contains:
   - an `assessment_starts` record with the exact click/start timestamp and camera permission status;
   - `tracking_data` eye records with live eye metrics for the active question.

Webcam permission can only be requested by the browser. It cannot be silently granted by JavaScript; the participant must approve the browser permission prompt once.
