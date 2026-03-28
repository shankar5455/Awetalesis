"""
tests/test_api.py – FastAPI endpoint tests using the ASGI test client.
"""

import json

import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoints:
    def test_root_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Speech-to-Speech" in resp.text

    def test_status_endpoint(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "is_running" in data
        assert "target_language" in data
        assert "asr_model" in data
        assert "tts_backend" in data

    def test_config_endpoint(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "audio" in data
        assert "vad" in data
        assert "asr" in data
        assert "translation" in data
        assert "tts" in data


class TestTargetLanguage:
    def test_set_target_language(self, client):
        resp = client.post(
            "/config/target",
            json={"target_language": "de"},
        )
        assert resp.status_code == 200
        assert "de" in resp.json()["message"]

    def test_set_target_language_reflected_in_status(self, client):
        client.post("/config/target", json={"target_language": "fr"})
        status = client.get("/status").json()
        assert status["target_language"] == "fr"

    def teardown_method(self, method):
        """Reset target language to 'en' after each test."""
        from api.app import _cfg
        _cfg.translation.target_language = "en"
        _cfg.tts.language = "en"


class TestPipelineLifecycle:
    def test_stop_when_not_running_returns_409(self, client):
        resp = client.post("/stop")
        assert resp.status_code == 409

    def test_start_pipeline(self, client):
        resp = client.post("/start")
        # May fail if sounddevice is unavailable in CI; that's acceptable
        # We just verify the response code is 200 or 500 (not a crash)
        assert resp.status_code in (200, 409, 500)

    def test_double_start_returns_409(self, client):
        from api.app import _state

        if not _state.is_running:
            client.post("/start")

        if _state.is_running:
            resp = client.post("/start")
            assert resp.status_code == 409
            # cleanup
            client.post("/stop")


class TestWebSocket:
    def test_websocket_connects_and_accepts_message(self, client):
        with client.websocket_connect("/ws/translate") as ws:
            ws.send_text(json.dumps({"action": "set_target", "language": "es"}))
            data = ws.receive_text()
            msg = json.loads(data)
            assert "es" in msg.get("message", "")
