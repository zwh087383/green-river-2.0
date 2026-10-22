from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

import rag_system.display_emotion as emotion
import rag_system.web.app as web_app
from rag_system.display_state import DisplayStateStore
from rag_system.settings import AppSettings, SettingsStore


@pytest.mark.parametrize(("text", "tone"), [
    ("太好了，志愿者成功救助了受伤动物！", "happy"),
    ("塑料垃圾正在危害江河生态，这令人担忧。", "concerned"),
    ("这次损失令人痛心，我们深感遗憾。", "sad"),
    ("真没想到，恢复速度如此快！", "surprised"),
    ("我有点疑惑，你说的是哪一条河？", "curious"),
    ("你过奖啦，我都有点不好意思了。", "shy"),
    ("大家多年的守护成果令人自豪。", "proud"),
    ("加油，你也可以成为守护江河的一员！", "encouraging"),
    ("这并非值得骄傲的事情。", "neutral"),
    ("我并不是有点害羞。", "neutral"),
    ("江河是水循环的重要组成部分。", "neutral"),
    ("这条河流存在污染，调查正在进行。", "neutral"),
    ("这不是很高兴的事情，也不是好消息。", "neutral"),
    ("这并非令人担忧的变化。", "neutral"),
])
def test_offline_emotion_is_conservative(text, tone):
    assert emotion.infer_emotion(text).tone == tone


@pytest.mark.parametrize('query,tone', [
    ('展示开心的表情','happy'), ('展示关切的表情','concerned'),
    ('展示难过的表情','sad'), ('展示惊讶的表情','surprised'),
    ('展示疑惑的表情','curious'), ('展示害羞的表情','shy'),
    ('展示自豪的表情','proud'), ('展示鼓励的表情','encouraging'),
])
def test_explicit_expression_requests_survive_unrelated_reply_tone(query, tone):
    assert emotion.infer_reply_emotion(query, '太好了，我很开心能帮到你。').tone == tone


def test_ros_display_accepts_streamed_tone_without_waiting_for_keywords(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from types import ModuleType, SimpleNamespace
    for name in ('rclpy', 'rclpy.node', 'std_msgs', 'std_msgs.msg', 'rag_interfaces', 'rag_interfaces.srv'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['rclpy.node'].Node = object
    sys.modules['std_msgs.msg'].String = object
    sys.modules['rag_interfaces.srv'].SetDisplayMode = SimpleNamespace(Request=object, Response=object)
    spec = importlib.util.spec_from_file_location('display_node_test', Path(__file__).parents[1] / 'ros2/voice/voice/display_node.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    node = module.DisplayNode.__new__(module.DisplayNode)
    node._store = DisplayStateStore(tmp_path / 'display.json')
    node._answer_text = node._query_text = ''
    node._tone = 'neutral'
    node._on_display_text(SimpleNamespace(data='[query] 小绿你真可爱'))
    node._on_display_text(SimpleNamespace(data='[tone] shy'))
    node._on_display_text(SimpleNamespace(data='[delta] 我会继续认真为你讲解。'))
    assert node._store.snapshot()['tone'] == 'shy'
    node._on_display_text(SimpleNamespace(data='[answer] 我会继续认真为你讲解。'))
    assert node._store.snapshot()['tone'] == 'shy'
    node._on_display_text(SimpleNamespace(data='[query] 新的问题'))
    assert node._store.snapshot()['tone'] == 'neutral'


@pytest.mark.parametrize("content", [
    '{"tone":"angry","confidence":0.9}',
    '{"tone":"happy","confidence":NaN}',
    '{"tone":"happy","confidence":2}',
    '[]',
    'invalid',
])
def test_model_result_rejects_invalid_data(content):
    with pytest.raises((ValueError, AttributeError)):
        emotion.parse_emotion(content)


@pytest.mark.parametrize("tone", sorted(emotion.EMOTION_TONES))
def test_all_tones_are_accepted_by_analysis_and_session(tmp_path, tone):
    result = emotion.parse_emotion('{"tone":"' + tone + '","confidence":0.9}')
    store = DisplayStateStore(tmp_path / "display.json")
    store.open_web_session("tab", "page")
    assert store.sync_web_session("page", 1, "speaking", tone=result.tone)["tone"] == tone


def test_gallery_is_a_read_only_page_until_preview_clicked(tmp_path, monkeypatch):
    store = DisplayStateStore(tmp_path / "display.json")
    store.set_emotion("thinking")
    monkeypatch.setattr(web_app, "display_state_store", store)
    before = store.snapshot()
    response = TestClient(web_app.app).get("/display/preview")
    assert response.status_code == 200
    assert "动态表情" in response.text
    assert store.snapshot() == before


def test_low_confidence_and_fenced_json():
    result = emotion.parse_emotion('```json\n{"tone":"sad","confidence":0.3}\n```')
    assert result.tone == "neutral"
    assert result.source == "model"


def test_no_key_does_not_contact_model(monkeypatch):
    async def unexpected(*args):
        pytest.fail("Offline analysis contacted the model")

    monkeypatch.setattr(emotion, "_model_emotion", unexpected)
    result = asyncio.run(emotion.analyze_emotion("太好了！", AppSettings()))
    assert result.tone == "happy"
    assert result.source == "rules"


def test_model_timeout_falls_back_without_blocking_other_work(monkeypatch):
    async def slow_model(*args):
        await asyncio.sleep(10)

    monkeypatch.setattr(emotion, "_model_emotion", slow_model)
    monkeypatch.setattr(emotion, "MODEL_TIMEOUT_SECONDS", 0.02)

    async def run():
        pending = asyncio.create_task(emotion.analyze_emotion("请小心，保护好自己。", AppSettings(deepseek_api_key="test")))
        await asyncio.sleep(0)
        assert not pending.done()
        result = await asyncio.wait_for(pending, 0.5)
        assert result.tone == "concerned"
        assert result.source == "rules"

    asyncio.run(run())


def test_provider_errors_fall_back_and_success_uses_model(monkeypatch):
    async def broken(*args):
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr(emotion, "_model_emotion", broken)
    settings = AppSettings(deepseek_api_key="test")
    assert asyncio.run(emotion.analyze_emotion("太好了！", settings)).source == "rules"

    async def classified(*args):
        return emotion.EmotionAnalysis("concerned", 0.9, "model")

    monkeypatch.setattr(emotion, "_model_emotion", classified)
    assert asyncio.run(emotion.analyze_emotion("待分析回答", settings)).tone == "concerned"


def test_analysis_endpoint_returns_suggestion_without_mutating_display(tmp_path, monkeypatch):
    store = DisplayStateStore(tmp_path / "display.json")
    store.set_emotion("listening")
    monkeypatch.setattr(web_app, "display_state_store", store)
    monkeypatch.setattr(web_app, "settings_store", SettingsStore(tmp_path / "settings.json"))
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = TestClient(web_app.app)
    before = store.snapshot()
    result = client.post("/api/display/analyze", json={"text": "太好了！"})
    assert result.status_code == 200
    assert result.json()["tone"] == "happy"
    assert store.snapshot() == before
    assert client.post("/api/display/analyze", json={"text": "x" * 1201}).status_code == 422


def test_tone_shares_session_ordering_and_resets_on_release(tmp_path):
    store = DisplayStateStore(tmp_path / "display.json")
    store.open_web_session("tab-a", "a")
    store.open_web_session("tab-b", "b")
    store.sync_web_session("a", 1, "speaking", tone="happy", claim=True)
    state = store.sync_web_session("a", 2, "speaking", tone="happy")
    assert state["tone"] == "happy"
    assert store.sync_web_session("a", 3, "speaking", tone="happy")["updated_at"] == state["updated_at"]
    assert not store.sync_web_session("b", 1, "speaking", tone="sad")["accepted"]
    assert not store.sync_web_session("a", 2, "speaking", tone="sad")["accepted"]
    assert store.snapshot()["tone"] == "happy"
    assert store.release_web_session("a")["tone"] == "neutral"


def test_tone_clears_on_expiry_refresh_and_manual_state(tmp_path, monkeypatch):
    import rag_system.display_state as display_module

    now = [100.0]
    monkeypatch.setattr(display_module.time, "time", lambda: now[0])
    store = DisplayStateStore(tmp_path / "display.json")
    store.open_web_session("tab", "a", 1)
    store.sync_web_session("a", 1, "speaking", tone="sad")
    now[0] += 20
    assert store.snapshot()["tone"] == "neutral"
    store.sync_web_session("a", 2, "speaking", tone="sad", claim=True)
    assert store.open_web_session("tab", "b", 2)["tone"] == "neutral"
    store.set_emotion("speaking", tone="happy")
    assert store.set_emotion("thinking")["tone"] == "neutral"
    with pytest.raises(ValueError):
        store.sync_web_session("b", 1, "speaking", tone="bad")


def test_shared_keyword_cases():
    import json
    from pathlib import Path
    from rag_system.emotion_rules import score_emotion

    cases = json.loads(Path(__file__).with_name("emotion_rule_cases.json").read_text(encoding="utf-8"))
    for text, tone in cases:
        actual, score = score_emotion(text)
        assert actual == tone, text
        assert score <= 6
    assert score_emotion("非常开心") == ("happy", 2)
    assert score_emotion("你真棒", interaction=True)[0] == "shy"
    assert score_emotion("你不太可爱", interaction=True)[0] == "neutral"


def test_shared_browser_rules_route():
    from rag_system.emotion_rules import RULES
    response = TestClient(web_app.app).get("/api/display/emotion-rules.js")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "window.DisplayEmotionRules" in response.text
    import json
    assert json.loads(response.text.removeprefix("window.DisplayEmotionRules = ").rstrip(";\n")) == RULES
