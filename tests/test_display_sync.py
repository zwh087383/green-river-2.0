from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import rag_system.display_state as display_module
import rag_system.web.app as web_app
from rag_system.display_state import DisplayStateStore, WEB_DISPLAY_LEASE_SECONDS


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(display_module.time, "time", lambda: now[0])
    return now


def test_refresh_resets_previous_conversation_and_rejects_old_messages(tmp_path, clock):
    store = DisplayStateStore(tmp_path / "display.json")
    store.open_web_session("tab", "old", 1)
    store.sync_web_session("old", 1, "speaking")
    store.set_mode("web")

    refreshed = store.open_web_session("tab", "new", 2)
    assert refreshed["state"] == "idle"
    assert refreshed["mode"] == "web"
    assert store.sync_web_session("old", 2, "speaking", claim=True)["session_status"] == "retired"
    store.sync_web_session("new", 1, "listening")
    assert store.release_web_session("old")["state"] == "listening"
    assert store.open_web_session("tab", "late-open", 1)["session_status"] == "retired"
    assert store.snapshot()["state"] == "listening"


def test_heartbeat_preserves_animation_and_lost_browser_expires(tmp_path, clock):
    store = DisplayStateStore(tmp_path / "display.json")
    store.open_web_session("tab", "page")
    first = store.sync_web_session("page", 1, "thinking")
    clock[0] += 5
    heartbeat = store.sync_web_session("page", 2, "thinking")
    assert heartbeat["updated_at"] == first["updated_at"]
    assert heartbeat["lease_expires_at"] > first["lease_expires_at"]
    # Simulate a new server process reading the persisted lease.
    restarted = DisplayStateStore(store.path)
    clock[0] += WEB_DISPLAY_LEASE_SECONDS + 1
    expired = restarted.snapshot()
    assert expired["state"] == "idle"
    assert expired["lease_expires_at"] == 0
    assert restarted.snapshot()["updated_at"] == expired["updated_at"]


def test_passive_tab_and_out_of_order_heartbeat_cannot_override_active_tab(tmp_path, clock):
    store = DisplayStateStore(tmp_path / "display.json")
    store.open_web_session("tab-a", "a")
    store.sync_web_session("a", 1, "speaking")
    assert not store.open_web_session("tab-b", "b")["accepted"]
    assert store.sync_web_session("b", 1, "idle")["state"] == "speaking"
    assert store.sync_web_session("b", 2, "listening", claim=True)["accepted"]
    assert not store.sync_web_session("a", 2, "speaking")["accepted"]
    assert store.sync_web_session("b", 4, "thinking")["accepted"]
    assert not store.sync_web_session("b", 3, "listening", claim=True)["accepted"]
    assert store.snapshot()["state"] == "thinking"


def test_manual_and_ros_states_are_not_replaced_by_background_heartbeats(tmp_path, clock):
    store = DisplayStateStore(tmp_path / "display.json")
    store.open_web_session("tab", "page")
    store.set_emotion("shy")
    assert not store.sync_web_session("page", 1, "idle")["accepted"]
    clock[0] += 100
    assert store.snapshot()["state"] == "shy"
    store.set_emotion("speaking", source="ros")
    assert not store.open_web_session("admin", "admin-page")["accepted"]
    assert store.snapshot()["state"] == "speaking"


def test_release_before_delayed_open_cannot_resurrect_closed_page(tmp_path):
    store = DisplayStateStore(tmp_path / "display.json")
    store.release_web_session("late")
    assert store.open_web_session("tab", "late")["session_status"] == "retired"
    assert store.snapshot()["state"] == "idle"


def test_session_api_validates_and_recovers_after_server_restart(tmp_path, monkeypatch):
    store = DisplayStateStore(tmp_path / "display.json")
    monkeypatch.setattr(web_app, "display_state_store", store)
    client = TestClient(web_app.app)
    identity = {"client_id": "tab", "session_id": "page", "generation": 1}
    assert client.post("/api/display/session", json=identity).json()["accepted"]
    state = {"session_id": "page", "sequence": 1, "state": "speaking"}
    assert client.post("/api/display/session/state", json=state).json()["accepted"]
    assert client.post("/api/display/session/state", json={**state, "state": "bad"}).status_code == 400
    assert client.post("/api/display/session/state", json={**state, "sequence": -1}).status_code == 422
    monkeypatch.setattr(web_app, "display_state_store", DisplayStateStore(store.path))
    assert client.post("/api/display/session/state", json=state).json()["session_status"] == "unknown"
    assert client.post("/api/display/session", json=identity).json()["accepted"]
    assert client.post("/api/display/session/state", json=state).json()["state"] == "speaking"
    assert client.post("/api/display/session/release", json={"session_id": "page"}).json()["state"] == "idle"
    assert client.get("/api/display/state").headers["cache-control"] == "no-store"
    assert client.get("/api/display/client.js").status_code == 200
