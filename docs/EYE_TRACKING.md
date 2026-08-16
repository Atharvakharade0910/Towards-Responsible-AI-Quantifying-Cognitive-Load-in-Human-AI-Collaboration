# Eye Tracking

## Purpose

CogniTrack uses webcam frames to monitor eye direction, blinks, gaze movement, head pose, attention-related signals, and fatigue during an active assessment. It does not store camera images; frames are analyzed and then discarded.

## Capture flow

1. The browser requests webcam access on localhost.
2. Video frames are resized to approximately 192 pixels wide and encoded as JPEG data.
3. A frame is analyzed about every 200 ms (approximately five attempts per second).
4. The in-flight guard prevents overlapping requests when analysis takes longer than 200 ms.
5. Frames are sent to `POST /api/eye-tracking/frame` with the participant token, question ID, task number, and UTC `captured_at` timestamp.
6. Tracking stops when the question or assessment ends.

## Face and eye analysis

The backend uses MediaPipe Face Mesh with refined iris landmarks. The first detected face is analyzed. If no face is found, the endpoint returns safe neutral metrics instead of failing.

The system calculates:

- Iris centers for the left and right eyes.
- Eye Aspect Ratio (EAR) from six landmarks per eye.
- Gaze direction: Centre, Left, Right, Up, Down, or a diagonal direction.
- Saccade state: `fast` when normalized gaze speed exceeds `0.35`; otherwise `focused`.
- Head pitch and yaw using facial landmarks and `solvePnP`.
- Tracking confidence from pupil symmetry and head-pose confidence.

## Calibration

During initial tracking, the system collects ten normal iris samples and uses their median as the participant’s eye-position baseline. EAR calibration creates a blink threshold equal to 72% of the median normal EAR, constrained between `0.16` and `0.25`.

The API also supports an older 16-point calibration mode. Four central calibration points are used for the final iris baseline. Calibration metadata is accepted for compatibility but is not included in the normal Admin Eye Tracking fields.

## Blink detection

Blink detection uses EAR and participant-specific thresholds:

- An eye closes when EAR falls below the calibrated threshold.
- A small reopening margin prevents noise near the threshold from repeatedly opening and closing the state.
- A blink is counted when the eye reopens after a closure lasting between `0.04` and `0.8` seconds.
- `blinked` is true on the frame that completes the blink.
- `blink_count` is cumulative for the participant’s active session.
- `blink_latency_ms` records the completed eye-closure duration in milliseconds on the frame where the blink is detected; non-blink frames contain `0`.

Blink events are persisted immediately so a short blink is not lost between regular database saves.

## PERCLOS and fatigue

PERCLOS is the proportion of recent valid samples in which the eyes were closed. The rolling history covers approximately 60 seconds.

The fatigue score combines low EAR and PERCLOS:

```text
fatigue_score = 50% low-EAR signal + 50% PERCLOS signal
```

The result is limited to `0.0–1.0` and mapped to:

- `low`
- `moderate`
- `high`

Fatigue is an interaction signal, not a medical diagnosis.

## Database persistence

Frames are analyzed frequently, but normal eye records are saved approximately once per second to reduce database load. The `persist` flag controls regular saving; blink events bypass that interval and save immediately.

Eye Tracking records contain:

- `participant_id`
- `task_number`
- `captured_at`
- `direction`
- `saccade`
- `blinked`
- `blink_count`
- `blink_latency_ms`
- `ear`
- `perclos`
- `fatigue`
- `head_pitch`
- `head_yaw`
- `tracking_confidence`

Records are stored in PostgreSQL in the unified `tracking_data` table with `component_type = eye_tracking` and `record_kind = sample`.

## Admin and CSV output

The Admin Eye Tracking table reads protected records from `/api/admin/data/eye-tracking`. The Eye Tracking CSV export reads the same PostgreSQL dataset through `/api/admin/exports/eye.csv`.

Both outputs include the `fatigue` column along with blink, gaze, EAR, PERCLOS, pose, and confidence values.

## Security and cleanup

- Participant authentication is required for frame submission.
- A participant token can write only its own records.
- Activity endpoints reject requests after the session ends.
- Camera streams and timers stop when the assessment finishes.
- In-memory calibration, blink, gaze, and rolling fatigue state is removed at session completion.
- Raw camera frames and pupil coordinates are not persisted.

## Time format

`captured_at` uses ISO 8601 UTC timestamps, for example `2026-08-05T19:58:55.533Z`. The `Z` means UTC. India Standard Time is UTC+05:30.
