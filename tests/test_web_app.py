"""Integration tests for the Flask web application endpoints."""

import json
from unittest.mock import patch

import pytest

import web_vers


@pytest.fixture
def client():
    """Create a Flask test client."""
    web_vers.app.config["TESTING"] = True
    with web_vers.app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def reset_controller():
    """Reset the global controller before each test."""
    web_vers.controller.reset()
    web_vers.controller.set_mode("adaptive")
    yield


# ---------------------------------------------------------------------------
# Index page tests
# ---------------------------------------------------------------------------

class TestIndexPage:
    def test_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_returns_html(self, client):
        resp = client.get("/")
        assert b"<!DOCTYPE html>" in resp.data

    def test_contains_stream_img(self, client):
        resp = client.get("/")
        assert b"/stream" in resp.data

    def test_contains_controls(self, client):
        resp = client.get("/")
        assert b"Adaptive" in resp.data
        assert b"Manual" in resp.data
        assert b"Apply" in resp.data

    def test_contains_metrics_section(self, client):
        resp = client.get("/")
        assert b"Metrics" in resp.data


# ---------------------------------------------------------------------------
# API metrics endpoint tests
# ---------------------------------------------------------------------------

class TestApiMetrics:
    def test_returns_json(self, client):
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "current_fps" in data
        assert "target_fps" in data
        assert "resolution" in data
        assert "mode" in data

    def test_default_metrics(self, client):
        data = client.get("/api/metrics").get_json()
        assert data["target_fps"] == 11.0
        assert data["current_quality"] == 70
        assert data["resolution"] == "1280x920"
        assert data["mode"] == "adaptive"


# ---------------------------------------------------------------------------
# API mode endpoint tests
# ---------------------------------------------------------------------------

class TestApiMode:
    def test_switch_to_manual(self, client):
        resp = client.post(
            "/api/mode",
            data=json.dumps({"mode": "manual"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["mode"] == "manual"

    def test_switch_to_adaptive(self, client):
        client.post(
            "/api/mode",
            data=json.dumps({"mode": "manual"}),
            content_type="application/json",
        )
        resp = client.post(
            "/api/mode",
            data=json.dumps({"mode": "adaptive"}),
            content_type="application/json",
        )
        assert resp.get_json()["mode"] == "adaptive"

    def test_invalid_mode_returns_400(self, client):
        resp = client.post(
            "/api/mode",
            data=json.dumps({"mode": "invalid"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()


# ---------------------------------------------------------------------------
# API settings endpoint tests
# ---------------------------------------------------------------------------

class TestApiSettings:
    def test_set_fps(self, client):
        resp = client.post(
            "/api/settings",
            data=json.dumps({"fps": 20}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["applied"]["fps"] == 20

    def test_set_quality(self, client):
        resp = client.post(
            "/api/settings",
            data=json.dumps({"quality": 85}),
            content_type="application/json",
        )
        assert resp.get_json()["applied"]["quality"] == 85

    def test_set_resolution(self, client):
        resp = client.post(
            "/api/settings",
            data=json.dumps({"width": 640, "height": 480}),
            content_type="application/json",
        )
        data = resp.get_json()["applied"]
        assert data["width"] == 640
        assert data["height"] == 480

    def test_clamping(self, client):
        resp = client.post(
            "/api/settings",
            data=json.dumps({"fps": 9999, "quality": -5}),
            content_type="application/json",
        )
        data = resp.get_json()["applied"]
        assert data["fps"] == 30
        assert data["quality"] == 10

    def test_settings_reflected_in_metrics(self, client):
        client.post(
            "/api/settings",
            data=json.dumps({"fps": 25, "quality": 40}),
            content_type="application/json",
        )
        metrics = client.get("/api/metrics").get_json()
        assert metrics["target_fps"] == 25
        assert metrics["current_quality"] == 40


# ---------------------------------------------------------------------------
# API reset endpoint tests
# ---------------------------------------------------------------------------

class TestApiReset:
    def test_reset_returns_status(self, client):
        resp = client.post("/api/reset")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "reset"

    def test_reset_restores_defaults(self, client):
        client.post(
            "/api/settings",
            data=json.dumps({"fps": 25, "quality": 40, "width": 640, "height": 480}),
            content_type="application/json",
        )
        client.post("/api/reset")
        metrics = client.get("/api/metrics").get_json()
        assert metrics["target_fps"] == 11.0
        assert metrics["current_quality"] == 70
        assert metrics["resolution"] == "1280x920"


# ---------------------------------------------------------------------------
# Stream endpoint tests
# ---------------------------------------------------------------------------

class TestStreamEndpoint:
    def test_stream_mimetype(self, client):
        with patch("web_vers.generate") as mock_gen:
            mock_gen.return_value = iter([
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfakedata\r\n"
            ])
            resp = client.get("/stream")
            assert "multipart/x-mixed-replace" in resp.content_type

    def test_stream_returns_data(self, client):
        with patch("web_vers.generate") as mock_gen:
            mock_gen.return_value = iter([
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfakedata\r\n"
            ])
            resp = client.get("/stream")
            assert resp.status_code == 200
            assert len(resp.data) > 0
