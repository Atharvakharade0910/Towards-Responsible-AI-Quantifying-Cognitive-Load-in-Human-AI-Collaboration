from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

TEST_ADMIN_PASSWORD = "test-admin-password"
TEST_HOST_PASSWORD = "test-host-password"
os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", TEST_ADMIN_PASSWORD)
os.environ.setdefault("HOST_USERNAME", "host")
os.environ.setdefault("HOST_PASSWORD", TEST_HOST_PASSWORD)
os.environ.setdefault("PARTICIPANT_TOKEN_SECRET", "test-participant-secret")

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import main
from storage import SqliteStorage
from vision_analysis import VisionState


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        main.storage = SqliteStorage(Path(self.temp_dir.name) / "test.sqlite3")
        main.AUTH_TOKENS.clear()
        main.SESSION_STATE.clear()
        main.LOGIN_FAILURES.clear()
        self.client = TestClient(main.app)

    def tearDown(self) -> None:
        self.client.close()
        self.temp_dir.cleanup()

    def start_participant(self, name: str = "Test Participant") -> tuple[dict, dict[str, str]]:
        response = self.client.post(
            "/api/sessions/start",
            json={
                "participant_id": "",
                "full_name": name,
                "age_group": "18-22",
                "domain": "Computer Science",
                "ai_familiarity": "Beginner",
            },
        )
        self.assertEqual(response.status_code, 200)
        session = response.json()
        return session, {"Authorization": f"Bearer {session['participant_session_token']}"}

    def test_role_token_is_required_and_logout_revokes_it(self) -> None:
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
        response = self.client.post(
            "/api/auth/admin/login",
            json={"username": main.ADMIN_USERNAME, "password": self._admin_password()},
        )
        self.assertEqual(response.status_code, 200)
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        self.assertEqual(self.client.get("/api/auth/me", headers=headers).json()["role"], "admin")
        self.assertEqual(self.client.post("/api/auth/logout", headers=headers).status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me", headers=headers).status_code, 401)

    def test_prompt_sentiment_recognizes_normal_positive_and_negative_phrases(self) -> None:
        self.assertEqual(main.analyze_prompt_sentiment("Thank you, that was very helpful."), "Positive")
        self.assertEqual(main.analyze_prompt_sentiment("I don't understand; this is not clear."), "Negative")
        self.assertEqual(main.analyze_prompt_sentiment("Can you explain the algorithm?"), "Neutral")

    def test_blink_detection_counts_a_short_closure_without_two_frames(self) -> None:
        state = VisionState(ear_threshold=0.20)
        analyzer = main.eye_tracking_analyzer

        self.assertFalse(analyzer._update_blink_state(state, 0.27, 10.00))
        self.assertFalse(analyzer._update_blink_state(state, 0.14, 10.10))
        self.assertTrue(analyzer._update_blink_state(state, 0.27, 10.24))
        self.assertEqual(state.blink_count, 1)
        self.assertEqual(state.last_blink_latency_ms, 140.0)

        long_closure = VisionState(ear_threshold=0.20)
        analyzer._update_blink_state(long_closure, 0.14, 20.00)
        self.assertFalse(analyzer._update_blink_state(long_closure, 0.27, 20.90))
        self.assertEqual(long_closure.blink_count, 0)

    def test_admin_and_host_activity_is_not_persisted(self) -> None:
        for role, username, password in (
            ("admin", main.ADMIN_USERNAME, self._admin_password()),
            ("host", main.HOST_USERNAME, self._host_password()),
        ):
            login = self.client.post(
                f"/api/auth/{role}/login",
                json={"username": username, "password": password},
            )
            self.assertEqual(login.status_code, 200)
            headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
            self.assertEqual(self.client.get("/api/auth/me", headers=headers).status_code, 200)
            self.assertEqual(self.client.get("/api/dashboard/overview", headers=headers).status_code, 200)
            self.assertEqual(self.client.post("/api/auth/logout", headers=headers).status_code, 200)
        with main.storage._connect() as connection:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        self.assertEqual(tables, [])

    def test_admin_eye_tracking_page_projects_only_eye_columns(self) -> None:
        main.append_tracking(
            "eye_tracking",
            "sample",
            {
                "participant_id": "COG0007",
                "question_id": "Q1",
                "captured_at": "2026-08-04T12:00:00Z",
                "direction": "Center",
                "blink_count": 2,
            },
        )
        login = self.client.post(
            "/api/auth/admin/login",
            json={"username": main.ADMIN_USERNAME, "password": self._admin_password()},
        )
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        eye_response = self.client.get("/api/admin/data/eye-tracking", headers=headers)
        self.assertEqual(eye_response.status_code, 200)
        eye_record = eye_response.json()["records"][0]
        self.assertEqual(eye_record["direction"], "Center")
        self.assertEqual(eye_record["blink_count"], "2")
        self.assertIn("blink_latency_ms", eye_response.json()["records"][0])
        self.assertIn("fatigue", eye_response.json()["records"][0])
        for excluded in ("id", "tracking_key", "question_id"):
            self.assertNotIn(excluded, eye_record)

        unified_record = self.client.get("/api/admin/data/tracking", headers=headers).json()["records"][0]
        for excluded in (
            "id", "tracking_key", "question_number", "left_click_count",
            "right_click_count", "middle_click_count", "save_reason",
        ):
            self.assertNotIn(excluded, unified_record)

    def test_session_start_assigns_unique_server_id(self) -> None:
        payload = {
            "participant_id": "",
            "full_name": "Test Participant",
            "age_group": "18-22",
            "domain": "Computer Science",
            "ai_familiarity": "Beginner",
        }
        first = self.client.post("/api/sessions/start", json=payload)
        second = self.client.post("/api/sessions/start", json=payload)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["participant_id"], "COG0001")
        self.assertEqual(second.json()["participant_id"], "COG0002")
        self.assertTrue(first.json()["participant_session_token"])

    def test_session_completion_is_idempotent(self) -> None:
        session, headers = self.start_participant("End Session Test")
        participant_id = session["participant_id"]
        payload = {
            "participant": {
                "participant_id": participant_id,
                "full_name": "End Session Test",
                "age_group": "18-22",
                "domain": "Computer Science",
                "ai_familiarity": "Beginner",
            },
            "session_started_at": "2026-08-05T12:00:00Z",
            "session_ended_at": "2026-08-05T12:01:00Z",
            "ended_early": True,
            "answers": [],
            "interaction_summary": {},
        }
        first = self.client.post("/api/sessions/complete", headers=headers, json=payload)
        second = self.client.post("/api/sessions/complete", headers=headers, json=payload)
        self.assertEqual(first.status_code, 200)
        self.assertFalse(first.json()["already_completed"])
        self.assertEqual(second.status_code, 200)
        self.assertTrue(second.json()["already_completed"])
        completed_rows = [
            row for row in main.storage.list_records("sessions", 10)
            if row.get("participant_id") == participant_id and row.get("event") == "completed"
        ]
        self.assertEqual(len(completed_rows), 1)


    def test_assessment_start_is_authenticated_and_persisted(self) -> None:
        session, headers = self.start_participant("Calibration Test")
        participant_id = session["participant_id"]
        denied = self.client.post(
            "/api/sessions/assessment-start",
            json={
                "participant_id": participant_id,
                "assessment_started_at": "2026-08-05T04:30:00Z",
                "camera_permission": "granted",
                "calibration_status": "started",
            },
        )
        self.assertEqual(denied.status_code, 401)
        response = self.client.post(
            "/api/sessions/assessment-start",
            headers=headers,
            json={
                "participant_id": participant_id,
                "assessment_started_at": "2026-08-05T04:30:00Z",
                "camera_permission": "granted",
                "calibration_status": "started",
            },
        )
        self.assertEqual(response.status_code, 200)
        records = main.storage.list_records("assessment_starts", limit=10)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["participant_id"], participant_id)
        self.assertEqual(records[0]["camera_permission"], "granted")

    def test_code_run_requires_session_and_keeps_diagnostics_and_output(self) -> None:
        denied = self.client.post(
            "/api/code/run",
            json={"language": "python", "source_code": "print('ok')"},
        )
        self.assertEqual(denied.status_code, 401)

        session = self.client.post(
            "/api/sessions/start",
            json={
                "participant_id": "",
                "full_name": "Compiler Test",
                "age_group": "18-22",
                "domain": "Computer Science",
                "ai_familiarity": "Beginner",
            },
        ).json()
        original = main.compiler_json_request
        main.compiler_json_request = AsyncMock(side_effect=[
            {"id": "run-1"},
            {"status": "completed", "build_stderr": "warning\\n", "stdout": "ok\\n", "exit_code": 0},
        ])
        try:
            response = self.client.post(
                "/api/code/run",
                headers={"Authorization": f"Bearer {session['participant_session_token']}"},
                json={"language": " Python ", "source_code": "print('ok')"},
            )
        finally:
            main.compiler_json_request = original
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["compile_output"], "warning\\n")
        self.assertEqual(response.json()["output"], "ok\\n")

    def test_ollama_uses_local_chat_api(self) -> None:
        captured: dict[str, object] = {}
        original = main.json_request

        def fake_json_request(url, payload, headers=None):
            captured.update({"url": url, "payload": payload, "headers": headers})
            return {"message": {"role": "assistant", "content": "local answer"}}

        main.json_request = fake_json_request
        try:
            answer, provider, model = main.request_llm(
                "Ollama", [{"role": "user", "content": "Help"}]
            )
        finally:
            main.json_request = original

        self.assertEqual(answer, "local answer")
        self.assertEqual(provider, "Ollama")
        self.assertEqual(model, main.OLLAMA_MODEL)
        self.assertEqual(captured["url"], f"{main.OLLAMA_BASE_URL}/api/chat")
        self.assertFalse(captured["payload"]["stream"])
        self.assertEqual(captured["payload"]["keep_alive"], main.OLLAMA_KEEP_ALIVE)
        self.assertEqual(captured["payload"]["options"]["num_predict"], main.OLLAMA_NUM_PREDICT)
        self.assertEqual(captured["payload"]["options"]["num_ctx"], main.OLLAMA_NUM_CTX)

    def test_streaming_chat_searches_web_and_emits_tokens(self) -> None:
        original_search = main.web_search
        original_stream = main.stream_llm
        captured: dict[str, object] = {}
        main.web_search = lambda query: [{
            "title": "Example result", "url": "https://example.com/fact", "snippet": "A useful fact"
        }]

        def fake_stream(provider, messages):
            captured["provider"] = provider
            captured["messages"] = messages
            yield "streamed "
            yield "answer"

        main.stream_llm = fake_stream
        session, headers = self.start_participant("Streaming Test")
        try:
            response = self.client.post(
                "/api/llm/chat/stream",
                headers=headers,
                json={
                    "participant_id": session["participant_id"], "question_id": "Q1",
                    "provider": "Groq", "message": "Find this fact", "history": [],
                },
            )
        finally:
            main.web_search = original_search
            main.stream_llm = original_stream

        self.assertEqual(response.status_code, 200)
        events = [main.json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([event["content"] for event in events if event["type"] == "token"], ["streamed ", "answer"])
        self.assertEqual(captured["provider"], "Groq")
        self.assertIn("https://example.com/fact", captured["messages"][1]["content"])
        self.assertEqual(events[-1]["type"], "done")

    def test_streaming_ollama_skips_web_search(self) -> None:
        original_search = main.web_search
        original_stream = main.stream_llm
        main.web_search = lambda _query: self.fail("Ollama should answer without web-search latency")
        main.stream_llm = lambda _provider, _messages: iter(("fast ", "answer"))
        session, headers = self.start_participant("Ollama Streaming Test")
        try:
            response = self.client.post(
                "/api/llm/chat/stream",
                headers=headers,
                json={
                    "participant_id": session["participant_id"], "question_id": "Q1",
                    "provider": "Ollama", "message": "Explain this code", "history": [],
                },
            )
        finally:
            main.web_search = original_search
            main.stream_llm = original_stream

        self.assertEqual(response.status_code, 200)
        events = [main.json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([event["content"] for event in events if event["type"] == "token"], ["fast ", "answer"])
        self.assertEqual([event for event in events if event["type"] == "sources"][0]["sources"], [])

    def test_eye_metrics_are_persisted_in_unified_tracking_table(self) -> None:
        original = main.eye_tracking_analyzer.analyze_eye_frame
        main.eye_tracking_analyzer.analyze_eye_frame = lambda *_args: {
            "face_detected": True,
            "pupils_detected": True,
            "calibration_point": 0,
            "calibration_point_complete": True,
            "calibration_percent": 6,
            "direction": "Centre",
            "fatigue": "low",
        }
        session, headers = self.start_participant("Vision Persistence Test")
        try:
            response = self.client.post(
                "/api/eye-tracking/frame",
                headers=headers,
                json={
                    "participant_id": session["participant_id"],
                    "question_id": "C1-MAIN",
                    "task_number": 1,
                    "captured_at": "2026-08-01T00:00:00Z",
                    "image": "data:image/jpeg;base64," + "A" * 100,
                    "calibration_point": 0,
                    "calibration_target_x": 0.125,
                    "calibration_target_y": 0.125,
                },
            )
        finally:
            main.eye_tracking_analyzer.analyze_eye_frame = original
        self.assertEqual(response.status_code, 200)
        with main.storage._connect() as connection:
            eye_count = connection.execute("SELECT COUNT(*) FROM tracking_data WHERE component_type='eye_tracking'").fetchone()[0]
            calibration = connection.execute("SELECT pupils_detected, calibration_point FROM tracking_data WHERE component_type='eye_tracking'").fetchone()
            legacy_table = connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='vision_metrics'"
            ).fetchone()[0]
        self.assertEqual(eye_count, 1)
        self.assertEqual(calibration, ("1", "0"))
        self.assertEqual(legacy_table, 0)

    def test_high_frequency_eye_frame_can_skip_database_persistence(self) -> None:
        original = main.eye_tracking_analyzer.analyze_eye_frame
        main.eye_tracking_analyzer.analyze_eye_frame = lambda *_args: {
            "face_detected": True, "pupils_detected": True, "direction": "Centre",
            "tracking_confidence": 0.95, "valid_sample_count": 25,
        }
        try:
            session = self.client.post(
                "/api/sessions/start",
                json={
                    "participant_id": "", "full_name": "Eye Test", "age_group": "18-22",
                    "domain": "Computer Science", "ai_familiarity": "Beginner",
                },
            ).json()
            response = self.client.post(
                "/api/eye-tracking/frame",
                headers={"Authorization": f"Bearer {session['participant_session_token']}"},
                json={
                    "participant_id": session["participant_id"], "question_id": "C1-MAIN",
                    "task_number": 1, "captured_at": "2026-08-01T00:00:00Z",
                    "image": "data:image/jpeg;base64," + "A" * 100, "persist": False,
                },
            )
        finally:
            main.eye_tracking_analyzer.analyze_eye_frame = original
        self.assertEqual(response.status_code, 200)
        with main.storage._connect() as connection:
            table_count = connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tracking_data'"
            ).fetchone()[0]
        self.assertEqual(table_count, 0)

    def test_blink_event_is_persisted_even_between_regular_eye_samples(self) -> None:
        original = main.eye_tracking_analyzer.analyze_eye_frame
        main.eye_tracking_analyzer.analyze_eye_frame = lambda *_args: {
            "face_detected": True, "pupils_detected": True, "direction": "Centre",
            "blinked": True, "blink_count": 1, "ear": 0.12,
        }
        try:
            session, headers = self.start_participant("Blink Persistence Test")
            response = self.client.post(
                "/api/eye-tracking/frame",
                headers=headers,
                json={
                    "participant_id": session["participant_id"], "question_id": "C1-MAIN",
                    "task_number": 1, "captured_at": "2026-08-01T00:00:00Z",
                    "image": "data:image/jpeg;base64," + "A" * 100, "persist": False,
                },
            )
        finally:
            main.eye_tracking_analyzer.analyze_eye_frame = original
        self.assertEqual(response.status_code, 200)
        with main.storage._connect() as connection:
            eye_count = connection.execute(
                "SELECT COUNT(*) FROM tracking_data WHERE component_type='eye_tracking'"
            ).fetchone()[0]
        self.assertEqual(eye_count, 1)

    def test_no_face_frame_returns_safe_eye_metrics(self) -> None:
        session, headers = self.start_participant("No Face Test")
        original_eye = main.eye_tracking_analyzer.analyze_eye_frame
        main.eye_tracking_analyzer.analyze_eye_frame = lambda *_args: {
            "face_detected": False,
            "pupils_detected": False,
            "tracking_confidence": 0.0,
        }
        try:
            response = self.client.post(
                "/api/eye-tracking/frame",
                headers=headers,
                json={
                    "participant_id": session["participant_id"],
                    "question_id": "C1-MAIN",
                    "task_number": 1,
                    "captured_at": "2026-08-01T00:00:00Z",
                    "image": "data:image/jpeg;base64," + "A" * 100,
                    "persist": False,
                },
            )
        finally:
            main.eye_tracking_analyzer.analyze_eye_frame = original_eye
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["metrics"]["face_detected"])
        self.assertEqual(response.json()["metrics"]["tracking_confidence"], 0.0)

    def test_eye_endpoint_requires_matching_participant_token(self) -> None:
        session = self.client.post(
            "/api/sessions/start",
            json={"participant_id": "", "full_name": "Vision Test", "age_group": "18-22", "domain": "CS", "ai_familiarity": "Beginner"},
        ).json()
        payload = {
            "participant_id": session["participant_id"], "question_id": "C1-MAIN", "task_number": 1,
            "captured_at": "2026-08-01T00:00:00Z", "image": "data:image/jpeg;base64," + "A" * 100,
        }
        self.assertEqual(self.client.post("/api/eye-tracking/frame", json=payload).status_code, 401)
        payload["participant_id"] = "another-participant"
        headers = {"Authorization": f"Bearer {session['participant_session_token']}"}
        self.assertEqual(self.client.post("/api/eye-tracking/frame", json=payload, headers=headers).status_code, 403)

        payload["participant_id"] = session["participant_id"]
        original_eye = main.eye_tracking_analyzer.analyze_eye_frame
        main.eye_tracking_analyzer.analyze_eye_frame = lambda *_args: {"face_detected": True, "pupils_detected": True, "direction": "Centre"}
        try:
            self.assertEqual(self.client.post("/api/eye-tracking/frame", json=payload, headers=headers).status_code, 200)
        finally:
            main.eye_tracking_analyzer.analyze_eye_frame = original_eye

    def test_keyboard_saves_only_backspaces_and_thinking_pause_time(self) -> None:
        session, headers = self.start_participant("Keyboard Test")
        payload = {
            "participant_id": session["participant_id"],
            "question_id": "C1-MAIN",
            "question_number": 1,
            "question_category": "Coding & Programming",
            "backspace_count": 1,
            "thinking_pause_seconds": 3,
            "save_reason": "idle_autosave",
            "is_final": False,
        }
        first = self.client.put("/api/keyboard", json=payload, headers=headers)
        payload.update(backspace_count=3, thinking_pause_seconds=5, save_reason="question_submit", is_final=True)
        second = self.client.put("/api/keyboard", json=payload, headers=headers)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        with main.storage._connect() as connection:
            rows = connection.execute(
                "SELECT backspace_count, thinking_pause_seconds, is_final FROM tracking_data WHERE component_type='keyboard_tracking' AND record_kind='summary'"
            ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "3")
        self.assertEqual(rows[0][1], "5")
        self.assertEqual(rows[0][2], "1")

    def test_mouse_cursor_scroll_summary_is_persisted(self) -> None:
        session = self.client.post(
            "/api/sessions/start",
            json={"participant_id": "", "full_name": "Mouse Test", "age_group": "18-22", "domain": "CS", "ai_familiarity": "Beginner"},
        ).json()
        response = self.client.put(
            "/api/mouse",
            headers={"Authorization": f"Bearer {session['participant_session_token']}"},
            json={
                "participant_id": session["participant_id"], "question_id": "C1-MAIN", "question_number": 1,
                "scroll_up_count": 2, "scroll_down_count": 3,
                "scroll_event_timestamps": ["2026-08-04T12:00:00Z"], "cursor_samples": [{"x": 1, "y": 2}],
                "save_reason": "question_submit", "is_final": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        with main.storage._connect() as connection:
            row = connection.execute(
                "SELECT scroll_up_count, scroll_down_count FROM tracking_data WHERE component_type='mouse_tracking'"
            ).fetchone()
        self.assertEqual(row, ("2", "3"))
        with main.storage._connect() as connection:
            columns = {item[1] for item in connection.execute("PRAGMA table_info(tracking_data)").fetchall()}
        self.assertNotIn("cursor_samples", columns)

    def test_monitoring_dashboard_is_protected_and_reports_watcher_alerts(self) -> None:
        participant, participant_headers = self.start_participant("Dashboard Participant")
        heartbeat = self.client.put(
            "/api/monitoring/heartbeat",
            headers=participant_headers,
            json={
                "participant_id": participant["participant_id"],
                "status": "active",
                "question_id": "C1-MAIN",
                "question_number": 1,
                "progress_percent": 10,
                "question_started": True,
                "inactivity_seconds": 75,
                "tab_switches": 3,
                "vision_status": "No face detected",
                "fatigue": "high",
                "captured_at": "2026-08-01T00:00:00Z",
            },
        )
        self.assertEqual(heartbeat.status_code, 200)
        self.assertEqual(heartbeat.json()["watcher_state"], "attention")
        self.assertEqual(len(heartbeat.json()["alerts"]), 4)
        self.assertEqual(len(heartbeat.json()["agent_reports"]), 5)
        self.assertEqual(self.client.get("/api/dashboard/overview").status_code, 401)
        login = self.client.post(
            "/api/auth/host/login",
            json={"username": main.HOST_USERNAME, "password": self._host_password()},
        )
        token = login.json()["access_token"]
        dashboard = self.client.get(
            "/api/dashboard/overview",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(dashboard.status_code, 200)
        self.assertEqual(dashboard.json()["participants"][0]["participant_id"], participant["participant_id"])
        self.assertEqual(dashboard.json()["summary"]["participants_needing_attention"], 1)
        self.assertEqual(dashboard.json()["role"], "host")
        self.assertEqual(len(dashboard.json()["host_agent"]["specialist_agents"]), 5)
        host_participant = dashboard.json()["participants"][0]
        self.assertEqual(host_participant["participant_id"], participant["participant_id"])
        self.assertEqual(host_participant["session_duration_seconds"], "0")
        self.assertNotIn("agent_reports", host_participant)

        admin_login = self.client.post(
            "/api/auth/admin/login",
            json={"username": main.ADMIN_USERNAME, "password": self._admin_password()},
        )
        admin_dashboard = self.client.get(
            "/api/dashboard/overview",
            headers={"Authorization": f"Bearer {admin_login.json()['access_token']}"},
        ).json()
        self.assertEqual(admin_dashboard["role"], "admin")
        self.assertIn("keyboard_records", admin_dashboard["summary"])
        self.assertIn("eye_tracking_records", admin_dashboard["summary"])
        self.assertIn("stored_results", admin_dashboard)
        self.assertIn("component_totals", admin_dashboard["analytics"])

        admin_headers = {"Authorization": f"Bearer {admin_login.json()['access_token']}"}
        self.client.post(
            "/api/sessions/start",
            json={
                "participant_id": "",
                "full_name": "First Participant",
                "age_group": "18-22",
                "domain": "Computer Science",
                "ai_familiarity": "Beginner",
            },
        )
        participant_page = self.client.get("/api/admin/data/participants", headers=admin_headers)
        self.assertEqual(participant_page.status_code, 200)
        self.assertEqual(participant_page.json()["dataset"], "participants")
        self.assertIn("records", participant_page.json())
        self.assertEqual(participant_page.json()["records"][0]["participant_number"], 1)
        self.assertEqual(
            self.client.get("/api/admin/data/tracking", headers={"Authorization": f"Bearer {token}"}).status_code,
            403,
        )

    def test_completed_session_rejects_further_participant_activity(self) -> None:
        session, headers = self.start_participant("Completed Session")
        participant_id = session["participant_id"]
        participant = {
            "participant_id": participant_id,
            "full_name": "Completed Session",
            "age_group": "18-22",
            "domain": "Computer Science",
            "ai_familiarity": "Beginner",
        }
        completion = self.client.post(
            "/api/sessions/complete",
            headers=headers,
            json={
                "participant": participant,
                "session_started_at": "2026-08-05T12:00:00Z",
                "session_ended_at": "2026-08-05T12:01:00Z",
                "ended_early": True,
                "overall_paas_rating": None,
                "answers": [],
                "interaction_summary": {},
            },
        )
        self.assertEqual(completion.status_code, 200)

        requests = (
            ("post", "/api/sessions/assessment-start", {
                "participant_id": participant_id,
                "assessment_started_at": "2026-08-05T12:00:00Z",
                "camera_permission": "granted",
                "calibration_status": "not-required",
            }),
            ("post", "/api/questions/start", {
                "participant_id": participant_id,
                "question_id": "C1-MAIN",
                "llm": "Ollama",
                "started_at": "2026-08-05T12:00:00Z",
            }),
            ("put", "/api/keyboard", {
                "participant_id": participant_id,
                "question_id": "C1-MAIN",
            }),
            ("put", "/api/mouse", {
                "participant_id": participant_id,
                "question_id": "C1-MAIN",
            }),
            ("put", "/api/monitoring/heartbeat", {
                "participant_id": participant_id,
                "captured_at": "2026-08-05T12:01:00Z",
            }),
            ("post", "/api/llm/chat/stream", {
                "participant_id": participant_id,
                "question_id": "C1-MAIN",
                "provider": "Ollama",
                "message": "This must not be streamed after completion.",
            }),
            ("post", "/api/code/run", {
                "language": "python",
                "source_code": "print(1)",
            }),
            ("post", "/api/answers/submit", {
                "participant_id": participant_id,
                "question_id": "C1-MAIN",
                "question_number": 1,
                "llm": "Ollama",
                "answer": "This answer must not be saved after completion.",
                "started_at": "2026-08-05T12:00:00Z",
                "submitted_at": "2026-08-05T12:01:00Z",
                "duration_seconds": 60,
            }),
        )
        for method, path, payload in requests:
            with self.subTest(path=path):
                response = getattr(self.client, method)(path, headers=headers, json=payload)
                self.assertEqual(response.status_code, 409)
                self.assertEqual(response.json()["detail"], "Participant session has ended")

        # The durable completion record must still block activity after a
        # backend restart clears the in-memory concurrency state.
        main.SESSION_STATE.clear()
        response = self.client.put(
            "/api/keyboard",
            headers=headers,
            json={"participant_id": participant_id, "question_id": "C1-MAIN"},
        )
        self.assertEqual(response.status_code, 409)

    def test_static_health_and_common_endpoint_errors(self) -> None:
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/app.js").status_code, 200)
        self.assertEqual(self.client.get("/styles.css").status_code, 200)
        self.assertEqual(self.client.get("/api/health").json()["database_connected"], True)
        self.assertEqual(self.client.post("/api/sessions/start", json={}).status_code, 422)

        session, headers = self.start_participant("Error Contract Test")
        self.assertEqual(
            self.client.post(
                "/api/code/run",
                headers=headers,
                json={"language": "javascript", "source_code": "console.log(1)"},
            ).status_code,
            400,
        )
        self.assertEqual(
            self.client.put(
                "/api/keyboard",
                json={"participant_id": session["participant_id"], "question_id": "C1-MAIN"},
            ).status_code,
            401,
        )

    def test_admin_unknown_routes_return_not_found(self) -> None:
        login = self.client.post(
            "/api/auth/admin/login",
            json={"username": main.ADMIN_USERNAME, "password": self._admin_password()},
        )
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        self.assertEqual(self.client.get("/api/admin/data/unknown", headers=headers).status_code, 404)
        self.assertEqual(self.client.get("/api/admin/exports/unknown.csv", headers=headers).status_code, 404)

    def test_removed_unused_routes_are_not_available(self) -> None:
        self.assertEqual(self.client.get("/api/auth/status").status_code, 404)
        self.assertEqual(self.client.post("/api/session-recordings/start").status_code, 404)
        self.assertEqual(self.client.get("/api/admin/reports/answers.csv").status_code, 404)
        self.assertEqual(self.client.post("/api/llm/chat").status_code, 404)

    def test_storage_adds_new_columns_to_existing_tracking_table(self) -> None:
        main.storage.upsert("schema_evolution", ["participant_id", "status"], {"participant_id": "P1", "status": "active"}, ("participant_id",))
        main.storage.upsert("schema_evolution", ["participant_id", "status", "detail"], {"participant_id": "P1", "status": "active", "detail": "Ready"}, ("participant_id",))
        with main.storage._connect() as connection:
            value = connection.execute("SELECT detail FROM schema_evolution WHERE participant_id='P1'").fetchone()[0]
        self.assertEqual(value, "Ready")

    @staticmethod
    def _admin_password() -> str:
        return TEST_ADMIN_PASSWORD

    @staticmethod
    def _host_password() -> str:
        return TEST_HOST_PASSWORD


if __name__ == "__main__":
    unittest.main()
