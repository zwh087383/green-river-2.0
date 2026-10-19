from __future__ import annotations

import asyncio
import json
import threading

from fastapi.testclient import TestClient
from fastapi import WebSocketDisconnect
from PIL import Image
from dataclasses import asdict
import pytest

from rag_system.generation.models import AnswerResult, Citation, EvidenceChunk
from rag_system.display_state import DisplayStateStore
from rag_system.indexing import BM25Index, ImageIndexingService, IndexingService, InMemoryVectorStore
from rag_system.indexing.manifest import IndexManifest
from rag_system.parsing.models import ParsedAsset, ParsedDocument
from rag_system.settings import SettingsStore
from rag_system.voice import VoiceStatus
import rag_system.web.app as web_app


app = web_app.app


def test_web_health_endpoint() -> None:
    response = web_app.health()

    assert response.status_code == 200
    assert json.loads(response.body) == {"ok": True, "service": "rag-system"}


def test_web_display_emotion_endpoint_updates_shared_state(tmp_path, monkeypatch) -> None:
    store = DisplayStateStore(tmp_path / "display.json")
    monkeypatch.setattr(web_app, "display_state_store", store)
    client = TestClient(app)

    response = client.post(
        "/api/display/emotion",
        json={"state": "thinking", "reason": "web conversation"},
    )

    assert response.status_code == 200
    assert response.json()["state"] == "thinking"
    assert client.get("/api/display/state").json()["state"] == "thinking"
    assert client.post("/api/display/emotion", json={"state": "unknown"}).status_code == 400


def test_web_page_uses_green_river_volunteer_persona() -> None:
    html = web_app.index()

    assert "小绿 · 绿色江河志愿讲解员" in html
    assert "你好，我是小绿" in html
    assert "用科技助力环保工作" in html
    assert web_app.WAKEUP_ACKNOWLEDGEMENT_TEXT == "嗨，我是小绿，有什么环保问题随时问我哦！"
    assert web_app.WAKEUP_ACKNOWLEDGEMENT_URL == "/api/voice/wakeup-acknowledgement"
    assert web_app.INTERRUPT_ACKNOWLEDGEMENT_TEXT == "我在"
    assert web_app.INTERRUPT_ACKNOWLEDGEMENT_URL == "/api/voice/interrupt-acknowledgement"


def test_web_upload_parse_job_returns_sections() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/parse",
        files=[("files", ("note.txt", b"hello parser", "text/plain"))],
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    job_response = client.get(f"/api/jobs/{job_id}")
    assert job_response.status_code == 200
    payload = job_response.json()

    assert payload["status"] == "completed"
    assert payload["progress"] == 1
    assert payload["results"][0]["document"]["sections"][0]["text"] == "hello parser"


def test_web_page_has_default_auto_index_toggle() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert 'id="autoIndexToggle" type="checkbox" checked' in response.text
    assert "上传解析后自动全部入库" in response.text
    assert "rag.autoIndexUploads" in response.text
    assert "maybeStartAutoIndexing" in response.text
    assert "DeepSeek API Key" in response.text
    assert "Query Rewrite" in response.text
    assert "chunkSizeInput" in response.text
    assert "chunkOverlapInput" in response.text
    assert "conversationRoundsInput" in response.text
    assert "saveConversationHistoryBtn" in response.text
    assert "/api/settings/conversation-history" in response.text
    assert "生成回复模型上下文" in response.text
    assert "上下文注入轮数" in response.text
    assert "一问一答计 1 轮" in response.text
    assert 'id="conversationRoundsInput" type="number" min="1" max="100" step="1" value="40"' in response.text
    assert "settings.conversation_max_rounds || 40" in response.text
    assert "conversationRoundsInput.value || 40" in response.text
    assert "判断模型" not in response.text
    assert "deepseekTurnPromptInput" in response.text
    assert "saveDeepseekPromptBtn" in response.text
    assert "resetDeepseekPromptBtn" in response.text
    assert "/api/settings/deepseek-prompt" in response.text
    assert 'id="llmPromptCatalog"' in response.text
    assert 'id="llmPromptCatalogStatus"' in response.text
    assert "/api/settings/llm-prompts" in response.text
    assert "loadLlmPromptCatalog" in response.text
    assert "renderLlmPromptCatalog" in response.text
    assert "检索调试" not in response.text
    assert "searchInput" not in response.text
    assert "searchBtn" not in response.text
    assert "answerBtn" not in response.text
    assert "/api/answer/stream" in response.text
    assert 'id="page-chat"' in response.text
    assert 'class="chat-workspace"' in response.text
    assert 'id="llmTraceStatus"' in response.text
    assert 'id="llmTraceFeed"' in response.text
    assert "function beginLiveLlmTrace" in response.text
    assert "function appendLiveLlmInput" in response.text
    assert "event.type === 'llm_input'" in response.text
    assert "escapeHtml(String(message?.content || ''))" in response.text
    assert 'id="page-docs"' in response.text
    assert 'id="page-manage"' in response.text
    assert 'id="page-settings"' in response.text
    assert 'data-page="chat"' in response.text
    assert 'data-page="docs"' in response.text
    assert 'data-page="manage"' in response.text
    assert 'data-page="settings"' in response.text
    assert "/api/documents" in response.text
    assert "loadDocuments" in response.text
    assert "chatInput" in response.text
    assert "chatSendBtn" in response.text
    assert "小绿 · 绿色江河志愿讲解员" in response.text
    assert "你好，我是小绿" in response.text
    assert "renderAnswerCitations" in response.text
    assert "renderAnswerProvenance" not in response.text
    assert "renderRetrievalQuality" not in response.text
    assert "retrieval.quality" not in response.text
    assert "second_pass" not in response.text
    assert "inline-citation" in response.text
    assert "voiceMicBtn" in response.text
    assert "autoSpeakBtn" in response.text
    assert "resolveAutoSpeakPreference" in response.text
    assert "storedValue === null || storedValue === '1'" in response.text
    assert "localStorage.getItem(AUTO_SPEAK_STORAGE_KEY)" in response.text
    assert "describeTtsRuntime" in response.text
    assert "describeAsrRuntime" in response.text
    assert "Qwen Realtime${modelText}" in response.text
    assert "asr_provider_active" in response.text
    assert "asr_model" in response.text
    assert "tts_provider_active" in response.text
    assert "tts_connection_state" in response.text
    assert "tts_model" in response.text
    assert "DashScope Realtime ${stateText}${modelText}" in response.text
    assert "Edge 在线回退${modelText}" in response.text
    assert "/api/voice/status" in response.text
    assert "/api/voice/stt" in response.text
    assert "/api/voice/stt/stream" in response.text
    assert "/api/voice/tts" in response.text
    assert "/api/voice/tts/stream" in response.text
    assert "vadEnabledInput" in response.text
    assert "vadEndSilenceInput" in response.text
    assert "AudioWorkletNode" in response.text
    assert "vad-pcm-processor" in response.text
    assert "end_silence_ms: endSilenceMs" in response.text
    assert "等待说话" in response.text
    assert "正在聆听" in response.text
    assert "检测到停顿，识别中" in response.text
    assert "payload.type === 'partial'" in response.text
    assert "实时识别（尚未发送）" in response.text
    assert "armVoiceStopTimer(10000)" in response.text
    assert "armVoiceStopTimer(30000)" in response.text
    assert "payload.text || recorder.partialText" in response.text
    assert "clearStreamingPartial" in response.text
    assert "restoreStreamingDraft" in response.text
    assert "流式识别不可用，已回退到整段录音" in response.text
    assert ".chat-composer textarea.stt-partial" in response.text
    assert "encodeWav" in response.text
    assert "startVoiceRecording" in response.text
    assert "playTTS" in response.text
    assert "splitTtsSentences" in response.text
    assert "enqueueTtsSentences" in response.text
    assert "/[\\p{L}\\p{N}]/u.test(cleaned) ? cleaned : ''" in response.text
    assert ".replace(/\\s+([，。！？!?、,.;；:：])/g, '$1')" in response.text
    assert "if (cleaned) ttsSentenceQueue.push" in response.text
    assert "MediaSource.isTypeSupported('audio/mpeg')" in response.text
    assert "response.body.getReader()" in response.text
    assert "playPromise = audio.play()" in response.text
    assert "void playPromise.catch(() => {})" in response.text
    assert "await playPromise" in response.text
    assert "prefetchTtsSentenceBlob" in response.text
    assert "playPrefetchedTtsBlob" in response.text
    assert "if (!ttsActiveItem || ttsPrefetchState || !ttsSentenceQueue.length) return" in response.text
    assert "const item = ttsSentenceQueue[0]" in response.text
    assert "stalePrefetch.controller.abort()" in response.text
    assert "URL.revokeObjectURL(stalePrefetch.objectUrl)" in response.text
    assert "ttsPrefetchState?.item === item" in response.text
    assert "void state.promise.catch(() => {})" in response.text
    assert "new QwenPcmPlayer" in response.text
    assert "await player.drain()" in response.text
    delta_handler = response.text.split("onDelta: text => {", 1)[1].split("onFinal: result => {", 1)[0]
    assert "enqueueTtsSentences" not in delta_handler
    assert "new AbortController()" in response.text
    assert "let wakeupStartToken = 0" in response.text
    assert "const wakeupStartingSockets = new Set()" in response.text
    assert "const startToken = ++wakeupStartToken" in response.text
    assert "const isCurrentStart = () => startToken === wakeupStartToken && wakeupEnabled" in response.text
    assert "const opened = await openWakeupSocket(candidate => {" in response.text
    assert "wakeupStartingSockets.add(candidate)" in response.text
    assert "wakeupStartToken += 1" in response.text
    assert "wakeupStartingSockets.clear()" in response.text
    assert "let currentAnswerController = null" in response.text
    assert "let answerGeneration = 0" in response.text
    assert "signal: controller.signal" in response.text
    assert "async function interruptActiveTurn" in response.text
    assert "currentAnswerController.abort()" in response.text
    assert "answerGeneration += 1" in response.text
    assert "payload.reason === 'interrupt'" in response.text
    assert "prompt_text: '我在'" in response.text
    assert "stopCurrentTTS()" in response.text
    assert "ttsPlaybackVersion" in response.text
    assert "getVisibleAssistantText" in response.text
    assert "assistant-answer-text" in response.text
    assert "isAnswering" in response.text
    assert "流式语音回复" in response.text
    assert ".app-page.chat-page.active" in response.text
    assert "overscroll-behavior: contain" in response.text
    assert "scrollbar-gutter: stable" in response.text

    partial_handler = response.text.split("payload.type === 'partial'", 1)[1].split(
        "payload.type === 'speech_end'", 1
    )[0]
    final_handler = response.text.split("payload.type === 'final'", 1)[1].split(
        "payload.type === 'no_speech'", 1
    )[0]
    assert "chatInput.value = text" in partial_handler
    assert "sendChatMessage" not in partial_handler
    assert "await sendChatMessage()" in final_handler


def test_web_page_exposes_voice_configuration_controls_and_dynamic_wakeup_vad() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert 'id="wakeupWordInput"' in response.text
    assert 'id="vadEndSilenceInput"' in response.text
    assert 'id="asrCorrectionRulesInput"' in response.text
    assert "async function saveWakeupSettings()" in response.text
    assert "async function saveVadSettings()" in response.text
    assert "async function saveAsrCorrections()" in response.text
    assert "/api/settings/wakeup" in response.text
    assert "/api/settings/vad" in response.text
    assert "/api/settings/asr-corrections" in response.text

    correction_parser = response.text.split(
        "function parseAsrCorrectionRules", 1
    )[1].split("async function saveAsrCorrections", 1)[0]
    assert "line.match(/^(.+?)\\s*(?:=>|->)" in correction_parser
    assert "split(/\\r?\\n/)" in correction_parser
    assert "index + 1" in correction_parser
    assert "rules.push({ wrong, correct })" in correction_parser

    wakeup_socket = response.text.split(
        "async function openWakeupSocket", 1
    )[1].split("async function startWakeupListening", 1)[0]
    assert "currentSettings?.vad_end_silence_ms" in wakeup_socket
    assert "voiceStatusPayload?.vad_end_silence_ms" in wakeup_socket
    assert "end_silence_ms: 400" not in wakeup_socket
    assert "voiceStatusPayload?.vad_enabled === false" in response.text
    assert "浏览器唤醒偏好已保留" in response.text


def test_web_settings_save_reload_and_mask_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(web_app, "settings_store", SettingsStore(settings_path))
    client = TestClient(app)
    full_key = "sk-webtest-1234567890"

    save_response = client.post("/api/settings/api-key", json={"api_key": full_key})
    assert save_response.status_code == 200
    assert save_response.json()["has_api_key"] is True
    assert save_response.json()["api_key_source"] == "file"
    assert save_response.json()["masked_api_key"] == "sk-****7890"
    assert full_key not in save_response.text

    monkeypatch.setattr(web_app, "settings_store", SettingsStore(settings_path))
    get_response = client.get("/api/settings")
    assert get_response.status_code == 200
    assert get_response.json()["has_api_key"] is True
    assert get_response.json()["masked_api_key"] == "sk-****7890"
    assert full_key not in get_response.text


def test_web_settings_rejects_empty_or_short_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(web_app, "settings_store", SettingsStore(tmp_path / "settings.json"))
    client = TestClient(app)

    empty_response = client.post("/api/settings/api-key", json={"api_key": " "})
    short_response = client.post("/api/settings/api-key", json={"api_key": "sk-1"})

    assert empty_response.status_code == 400
    assert short_response.status_code == 400


def test_web_settings_delete_saved_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    settings_path = tmp_path / "settings.json"
    full_key = "sk-delete-web-12345678"
    store = SettingsStore(settings_path)
    store.save_api_key(full_key)
    monkeypatch.setattr(web_app, "settings_store", store)
    client = TestClient(app)

    delete_response = client.delete("/api/settings/api-key")

    assert delete_response.status_code == 200
    assert delete_response.json()["has_api_key"] is False
    assert full_key not in delete_response.text
    assert "dashscope_api_key" not in settings_path.read_text(encoding="utf-8")


def test_web_deepseek_settings_save_reload_and_mask_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(web_app, "settings_store", SettingsStore(settings_path))
    client = TestClient(app)
    full_key = "sk-deepseek-web-12345678"

    save_response = client.post("/api/settings/deepseek-api-key", json={"api_key": full_key})
    assert save_response.status_code == 200
    assert save_response.json()["has_deepseek_api_key"] is True
    assert save_response.json()["deepseek_api_key_source"] == "file"
    assert save_response.json()["masked_deepseek_api_key"] == "sk-****5678"
    assert save_response.json()["answer_model"] == "qwen-audio-3.1-realtime-plus"
    assert full_key not in save_response.text

    monkeypatch.setattr(web_app, "settings_store", SettingsStore(settings_path))
    get_response = client.get("/api/settings")
    assert get_response.status_code == 200
    assert get_response.json()["has_deepseek_api_key"] is True
    assert get_response.json()["masked_deepseek_api_key"] == "sk-****5678"
    assert full_key not in get_response.text

    delete_response = client.delete("/api/settings/deepseek-api-key")
    assert delete_response.status_code == 200
    assert delete_response.json()["has_deepseek_api_key"] is False
    assert "deepseek_api_key" not in settings_path.read_text(encoding="utf-8")


def test_web_deepseek_prompt_save_validate_and_reset(tmp_path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(web_app, "settings_store", SettingsStore(settings_path))
    client = TestClient(app)
    custom_prompt = """轮次规则：
[[IGNORE_TURN]] 继续聆听。
[[END_SESSION]] 回到待唤醒。
[[ANSWER]] 正常回答。"""

    save_response = client.post(
        "/api/settings/deepseek-prompt",
        json={"prompt": custom_prompt},
    )

    assert save_response.status_code == 200
    assert save_response.json()["deepseek_turn_prompt"] == custom_prompt
    assert SettingsStore(settings_path).load().deepseek_turn_prompt == custom_prompt

    invalid_response = client.post(
        "/api/settings/deepseek-prompt",
        json={"prompt": "仅保留 [[ANSWER]]"},
    )
    assert invalid_response.status_code == 400
    assert "缺少必要标记" in invalid_response.json()["detail"]

    reset_response = client.delete("/api/settings/deepseek-prompt")
    assert reset_response.status_code == 200
    assert reset_response.json()["deepseek_turn_prompt"] != custom_prompt
    assert "deepseek_turn_prompt" not in settings_path.read_text(encoding="utf-8")


def test_web_llm_prompt_catalog_exposes_complete_runtime_templates(tmp_path, monkeypatch) -> None:
    secret = "sk-catalog-must-not-leak-12345678"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    settings_path = tmp_path / "settings.json"
    store = SettingsStore(settings_path)
    custom_prompt = """自定义轮次规则：
[[IGNORE_TURN]] 忽略无效输入。
[[END_SESSION]] 结束对话。
[[ANSWER]] 继续回答。"""
    store.save_deepseek_turn_prompt(custom_prompt)
    monkeypatch.setattr(web_app, "settings_store", store)

    response = TestClient(app).get("/api/settings/llm-prompts")

    assert response.status_code == 200
    assert secret not in response.text
    prompts = response.json()["prompts"]
    by_id = {prompt["id"]: prompt for prompt in prompts}
    assert set(by_id) == {"direct_audio_reply"}
    assert by_id["direct_audio_reply"]["model"] == "qwen-audio-3.1-realtime-plus"
    messages = by_id["direct_audio_reply"]["messages"]
    assert messages[0]["role"] == "system"
    assert "小绿" in messages[0]["content"]
    assert custom_prompt not in str(messages)
    assert "{{ current_user_message }}" in messages[-1]["content"]


def test_web_settings_test_api_key_does_not_expose_full_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(web_app, "settings_store", SettingsStore(tmp_path / "settings.json"))
    client = TestClient(app)
    full_key = "sk-format-web-12345678"

    response = client.post("/api/settings/test-api-key", json={"api_key": full_key})

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["live_test"] is False
    assert response.json()["checked_api_key"] == "sk-****5678"
    assert full_key not in response.text


def test_web_chunking_settings_save_reload_and_parse_uses_config(tmp_path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(web_app, "settings_store", SettingsStore(settings_path))
    client = TestClient(app)

    save_response = client.post("/api/settings/chunking", json={"chunk_size": 800, "chunk_overlap": 120})
    assert save_response.status_code == 200
    assert save_response.json()["chunk_size"] == 800
    assert save_response.json()["chunk_overlap"] == 120

    reload_response = client.get("/api/settings")
    assert reload_response.status_code == 200
    assert reload_response.json()["chunk_size"] == 800
    assert reload_response.json()["chunk_overlap"] == 120

    long_text = ("alpha beta gamma delta " * 80).encode("utf-8")
    parse_response = client.post(
        "/api/parse",
        data={"chunk_size": "500", "chunk_overlap": "50"},
        files=[("files", ("long.txt", long_text, "text/plain"))],
    )
    assert parse_response.status_code == 200
    job_id = parse_response.json()["job_id"]

    job_response = client.get(f"/api/jobs/{job_id}")
    assert job_response.status_code == 200
    job = job_response.json()
    chunks = job["results"][0]["document"]["chunks"]

    assert job["chunking"] == {"chunk_size": 500, "chunk_overlap": 50}
    assert len(chunks) > 1
    assert all(chunk["metadata"]["chunk_size"] == 500 for chunk in chunks)
    assert all(chunk["metadata"]["chunk_overlap"] == 50 for chunk in chunks)


def test_web_conversation_history_settings_save_reload(tmp_path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(web_app, "settings_store", SettingsStore(settings_path))
    client = TestClient(app)

    default_response = client.get("/api/settings")
    assert default_response.status_code == 200
    assert default_response.json()["conversation_max_rounds"] == 40

    save_response = client.post("/api/settings/conversation-history", json={"max_rounds": 12})
    assert save_response.status_code == 200
    assert save_response.json()["conversation_max_rounds"] == 12

    reload_response = client.get("/api/settings")
    assert reload_response.status_code == 200
    assert reload_response.json()["conversation_max_rounds"] == 12

    bad_response = client.post("/api/settings/conversation-history", json={"max_rounds": 0})
    assert bad_response.status_code == 400


def test_web_vad_settings_save_reload_and_validate(tmp_path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(web_app, "settings_store", SettingsStore(settings_path))
    client = TestClient(app)

    save_response = client.post(
        "/api/settings/vad",
        json={"enabled": True, "end_silence_ms": 400},
    )
    assert save_response.status_code == 200
    assert save_response.json()["vad_enabled"] is True
    assert save_response.json()["vad_end_silence_ms"] == 400

    reload_response = client.get("/api/settings")
    assert reload_response.json()["vad_end_silence_ms"] == 400
    assert client.post(
        "/api/settings/vad",
        json={"enabled": True, "end_silence_ms": 350},
    ).status_code == 400
    assert client.post(
        "/api/settings/vad",
        json={"enabled": True, "end_silence_ms": 3100},
    ).status_code == 400


def test_web_wakeup_settings_preflight_save_and_preserve_previous_on_failure(
    tmp_path, monkeypatch
) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    monkeypatch.setattr(web_app, "settings_store", store)
    checked_words: list[str] = []

    def validate_keyword(word: str) -> str:
        checked_words.append(word)
        if "不可用" in word:
            raise ValueError("token 不受模型支持")
        return f"keyword-spec:{word}"

    monkeypatch.setattr(web_app, "keyword_spec_for_text", validate_keyword)
    client = TestClient(app)

    save_response = client.post(
        "/api/settings/wakeup",
        json={"word": " 你好小绿 "},
    )

    assert save_response.status_code == 200
    assert save_response.json()["wakeup_word"] == "你好小绿"
    assert checked_words == [" 你好小绿 "]
    assert store.load().wakeup_word == "你好小绿"

    rejected_response = client.post(
        "/api/settings/wakeup",
        json={"word": "不可用唤醒词"},
    )

    assert rejected_response.status_code == 400
    assert "token 不受模型支持" in rejected_response.json()["detail"]
    assert checked_words == [" 你好小绿 ", "不可用唤醒词"]
    assert store.load().wakeup_word == "你好小绿"


def test_web_asr_corrections_persist_and_refresh_voice_service(tmp_path, monkeypatch) -> None:
    store = SettingsStore(tmp_path / "settings.json")

    class RefreshableVoiceService:
        def __init__(self) -> None:
            self.refresh_calls = 0

        def refresh_asr_corrections(self) -> None:
            self.refresh_calls += 1

    service = RefreshableVoiceService()
    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(web_app, "voice_service_factory", lambda: service)
    client = TestClient(app)
    rules = [
        {"wrong": "绿色江和", "correct": "绿色江河"},
        {"wrong": "索兰达杰", "correct": "索南达杰"},
    ]

    save_response = client.post(
        "/api/settings/asr-corrections",
        json={"enabled": True, "rules": rules},
    )

    assert save_response.status_code == 200
    assert save_response.json()["asr_correction_enabled"] is True
    assert save_response.json()["asr_correction_rules"] == rules
    assert service.refresh_calls == 1

    reload_response = client.get("/api/settings")
    assert reload_response.status_code == 200
    assert reload_response.json()["asr_correction_enabled"] is True
    assert reload_response.json()["asr_correction_rules"] == rules


def test_web_asr_settings_switch_provider_and_validate(tmp_path, monkeypatch) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    monkeypatch.setattr(web_app, "settings_store", store)
    client = TestClient(app)

    response = client.post(
        "/api/settings/asr",
        json={
            "provider": "dashscope_realtime",
            "model": "qwen3-asr-flash-realtime",
        },
    )

    assert response.status_code == 200
    assert response.json()["asr_provider"] == "dashscope_realtime"
    assert response.json()["dashscope_asr_model"] == "qwen3-asr-flash-realtime"
    assert client.post(
        "/api/settings/asr",
        json={"provider": "unknown"},
    ).status_code == 400


def test_web_settings_test_deepseek_api_key_does_not_expose_full_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(web_app, "settings_store", SettingsStore(tmp_path / "settings.json"))
    client = TestClient(app)
    full_key = "sk-deepseek-format-12345678"

    response = client.post("/api/settings/test-deepseek-api-key", json={"api_key": full_key})

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["live_test"] is False
    assert response.json()["checked_api_key"] == "sk-****5678"
    assert full_key not in response.text


def test_web_voice_status_stt_and_tts_endpoints(monkeypatch, tmp_path) -> None:
    fake_voice = FakeVoiceService()
    monkeypatch.setattr(web_app, "settings_store", SettingsStore(tmp_path / "settings.json"))
    monkeypatch.setattr(web_app, "voice_service_factory", lambda: fake_voice)
    monkeypatch.setattr(web_app, "vad_engine_factory", lambda: FakeVadEngine())
    client = TestClient(app)

    status_response = client.get("/api/voice/status")
    stt_response = client.post(
        "/api/voice/stt",
        files={"audio": ("recording.wav", b"fake-wav", "audio/wav")},
    )
    tts_response = client.post("/api/voice/tts", json={"text": "朗读这句话"})
    streaming_tts_response = client.post("/api/voice/tts/stream", json={"text": "流式朗读"})

    assert status_response.status_code == 200
    assert status_response.json()["stt_available"] is True
    assert status_response.json()["vad_available"] is True
    assert status_response.json()["streaming_stt_available"] is True
    assert status_response.json()["vad_end_silence_ms"] == 400
    assert stt_response.status_code == 200
    assert stt_response.json()["text"] == "语音问题"
    assert fake_voice.last_stt_filename == "recording.wav"
    assert tts_response.status_code == 200
    assert tts_response.json() == {"audio": "ZmFrZS1tcDM=", "format": "mp3"}
    assert streaming_tts_response.status_code == 200
    assert streaming_tts_response.headers["content-type"].startswith("audio/mpeg")
    assert streaming_tts_response.headers["cache-control"] == "no-store"
    assert streaming_tts_response.content == b"fake-stream-mp3"
    assert fake_voice.last_tts_text == "流式朗读"


def test_web_voice_status_uses_qwen_server_vad_without_loading_local_vad(monkeypatch) -> None:
    class CloudStatusVoice(FakeVoiceService):
        def status(self):
            return VoiceStatus(
                voice_available=True,
                stt_available=True,
                tts_available=True,
                stt_model="base",
                tts_voice="Cherry",
                tts_rate="+0%",
                errors=[],
                asr_provider="dashscope_realtime",
                asr_provider_active="dashscope_realtime",
                asr_model="qwen3-asr-flash-realtime",
                asr_streaming=True,
            )

    monkeypatch.setattr(web_app, "voice_service_factory", lambda: CloudStatusVoice())
    monkeypatch.setattr(
        web_app,
        "vad_engine_factory",
        lambda: (_ for _ in ()).throw(AssertionError("local VAD must not load")),
    )

    payload = TestClient(app).get("/api/voice/status").json()

    assert payload["vad_available"] is True
    assert payload["vad_provider"] == "dashscope_server_vad"
    assert payload["vad_model"] == "qwen3-asr-flash-realtime"
    assert payload["streaming_stt_available"] is True


def test_web_lifespan_warms_and_closes_realtime_tts(monkeypatch) -> None:
    monkeypatch.setattr(web_app, "_warm_retrieval", lambda: None)
    class LifecycleVoice:
        def __init__(self) -> None:
            self.warm_calls = 0
            self.close_calls = 0
            self.asr_close_calls = 0

        async def warm_tts_connection(self) -> bool:
            self.warm_calls += 1
            return True

        async def close_tts_connection(self) -> None:
            self.close_calls += 1

        async def close_asr_connections(self) -> None:
            self.asr_close_calls += 1

    voice = LifecycleVoice()
    monkeypatch.setattr(web_app, "voice_service_factory", lambda: voice)

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert voice.warm_calls == 0
        assert voice.close_calls == 0
        assert voice.asr_close_calls == 0

    assert voice.close_calls == 1
    assert voice.asr_close_calls == 1


def test_web_lifespan_keeps_app_available_when_tts_warm_fails(monkeypatch) -> None:
    monkeypatch.setattr(web_app, "_warm_retrieval", lambda: None)
    class FailingLifecycleVoice:
        def __init__(self) -> None:
            self.close_calls = 0

        async def warm_tts_connection(self) -> bool:
            raise RuntimeError("provider unavailable")

        async def close_tts_connection(self) -> None:
            self.close_calls += 1

    voice = FailingLifecycleVoice()
    monkeypatch.setattr(web_app, "voice_service_factory", lambda: voice)

    with TestClient(app) as client:
        assert client.get("/").status_code == 200

    assert voice.close_calls == 1


def test_web_streaming_tts_surfaces_setup_error_before_response(monkeypatch) -> None:
    class BrokenStreamingVoice:
        async def stream_text_to_speech(self, text: str):
            del text
            raise web_app.VoiceDependencyError("streaming TTS unavailable")
            yield b""  # pragma: no cover - keeps this an async generator

    monkeypatch.setattr(web_app, "voice_service_factory", lambda: BrokenStreamingVoice())
    response = TestClient(app).post("/api/voice/tts/stream", json={"text": "hello"})

    assert response.status_code == 503
    assert response.json()["detail"] == "streaming TTS unavailable"


def test_web_streaming_tts_surfaces_provider_error_before_response(monkeypatch) -> None:
    class BrokenProviderVoice:
        async def stream_text_to_speech(self, text: str):
            del text
            raise RuntimeError("provider connection failed")
            yield b""  # pragma: no cover - keeps this an async generator

    monkeypatch.setattr(web_app, "voice_service_factory", lambda: BrokenProviderVoice())
    response = TestClient(app).post("/api/voice/tts/stream", json={"text": "hello"})

    assert response.status_code == 502
    assert response.json()["detail"] == "TTS provider failed: provider connection failed"


def test_web_streaming_tts_rejects_empty_text_before_response(monkeypatch, tmp_path) -> None:
    service = web_app.VoiceService(settings_store=SettingsStore(tmp_path / "settings.json"))
    monkeypatch.setattr(web_app, "voice_service_factory", lambda: service)

    response = TestClient(app).post("/api/voice/tts/stream", json={"text": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "TTS text cannot be empty"


class FakeEmbeddingClient:
    model = "text-embedding-v4"
    dimensions = 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "alpha" in lower else 0.0,
                    1.0 if "beta" in lower else 0.0,
                    1.0 if "gamma" in lower else 0.0,
                    1.0,
                ]
            )
        return vectors


class FakeImageEmbeddingClient:
    model = "qwen3-vl-embedding"
    dimensions = 2048

    def embed_images(self, image_paths: list) -> list[list[float]]:
        return [[1.0, 0.0] + [0.0] * 2046 for _ in image_paths]


class FakeVoiceService:
    def __init__(self) -> None:
        self.last_tts_text = ""
        self.last_stt_filename = ""

    def status(self):
        return VoiceStatus(
            voice_available=True,
            stt_available=True,
            tts_available=True,
            stt_model="base",
            tts_voice="zh-CN-XiaoxiaoNeural",
            tts_rate="+0%",
            errors=[],
        )

    def speech_to_text(self, audio_bytes: bytes, *, filename: str):
        self.last_stt_filename = filename
        assert audio_bytes
        return "语音问题"

    def text_to_speech(self, text: str):
        self.last_tts_text = text
        return b"fake-mp3"

    async def stream_text_to_speech(self, text: str):
        self.last_tts_text = text
        yield b"fake-"
        yield b"stream-mp3"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] + [0.0] * 2046 for _ in texts]


class FakeVadEvent:
    def __init__(
        self,
        event_type: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        reason: str | None = None,
        audio_pcm: bytes | None = None,
    ) -> None:
        self.type = event_type
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.reason = reason
        self.audio_pcm = audio_pcm

    def to_dict(self) -> dict:
        payload = {"type": self.type}
        for key in ("start_ms", "end_ms", "reason"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


class FakeVadSession:
    def __init__(self) -> None:
        self.feed_count = 0

    def feed_pcm16(self, data: bytes, *, is_final: bool = False):
        self.feed_count += 1
        if is_final:
            return [FakeVadEvent("no_speech", reason="final")]
        if self.feed_count == 1:
            return [FakeVadEvent("speech_start", start_ms=0)]
        return [
            FakeVadEvent(
                "speech_end",
                start_ms=0,
                end_ms=400,
                reason="silence",
                audio_pcm=b"\x01\x00" * 1600,
            )
        ]


class FakeVadEngine:
    def __init__(self) -> None:
        self.config = None

    def create_session(self, config=None):
        self.config = config
        return FakeVadSession()


def test_web_realtime_asr_streams_partial_and_final_without_local_vad(tmp_path, monkeypatch) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save_vad(True, 400)

    class Event:
        def __init__(self, event_type: str, text: str) -> None:
            self.type = event_type
            self.text = text

    class FakeRealtimeASR:
        def __init__(self) -> None:
            self.audio = []
            self.close_calls = 0

        async def append_audio(self, chunk: bytes) -> None:
            self.audio.append(chunk)

        async def events(self):
            import asyncio

            while not self.audio:
                await asyncio.sleep(0)
            yield Event("partial", "綠色江河")
            yield Event("completed", "綠色江河知識庫")

        async def finish(self) -> None:
            return None

        async def close(self) -> None:
            self.close_calls += 1

    transport = FakeRealtimeASR()

    class CloudVoice(FakeVoiceService):
        async def open_realtime_asr(self, *, end_silence_ms: int):
            assert end_silence_ms == 400
            return transport

        async def release_realtime_asr(self, candidate) -> None:
            assert candidate is transport
            await candidate.close()

        def postprocess_asr_text(self, text: str) -> str:
            return text.replace("綠", "绿")

    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(web_app, "voice_service_factory", lambda: CloudVoice())
    monkeypatch.setattr(
        web_app,
        "vad_engine_factory",
        lambda: (_ for _ in ()).throw(AssertionError("local VAD must not be loaded")),
    )

    with TestClient(app).websocket_connect("/api/voice/stt/stream") as websocket:
        websocket.send_json({"type": "start"})
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_bytes(b"\x00\x00" * 1600)
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'speech_start', 'reason': 'server_vad'}
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'partial', 'text': '绿色江河', 'provider': 'dashscope_realtime'}
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'speech_end', 'reason': 'server_vad'}
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'final', 'text': '绿色江河知識庫', 'reason': 'server_vad', 'provider': 'dashscope_realtime'}

    assert transport.audio == [b"\x00\x00" * 1600]
    assert transport.close_calls == 1


def test_web_realtime_asr_disconnect_continues_same_audio_with_local_whisper(
    tmp_path, monkeypatch
) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save_vad(True, 400)
    first_chunk = b"\x11\x00" * 1600
    second_chunk = b"\x22\x00" * 1600

    class Event:
        type = "partial"
        text = "绿色江河"

    class DisconnectingTransport:
        def __init__(self) -> None:
            self.audio: list[bytes] = []
            self.closed = False

        async def append_audio(self, chunk: bytes) -> None:
            self.audio.append(chunk)

        async def events(self):
            while not self.audio:
                await asyncio.sleep(0)
            yield Event()
            raise RuntimeError("Qwen connection dropped")

        async def finish(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    class TrackingVadSession:
        def __init__(self) -> None:
            self.audio = bytearray()
            self.started = False

        def feed_pcm16(self, data: bytes, *, is_final: bool = False):
            self.audio.extend(data)
            if not self.started and data:
                self.started = True
                return [FakeVadEvent("speech_start", start_ms=0)]
            if len(self.audio) >= len(first_chunk) + len(second_chunk):
                return [
                    FakeVadEvent(
                        "speech_end",
                        start_ms=0,
                        end_ms=200,
                        reason="silence",
                        audio_pcm=bytes(self.audio),
                    )
                ]
            if is_final:
                return [FakeVadEvent("no_speech", reason="final")]
            return []

    class TrackingVadEngine:
        def __init__(self) -> None:
            self.session = TrackingVadSession()

        def create_session(self, _config=None):
            return self.session

    transport = DisconnectingTransport()
    vad_engine = TrackingVadEngine()

    class CloudVoice(FakeVoiceService):
        def __init__(self) -> None:
            super().__init__()
            self.failure = ""

        async def open_realtime_asr(self, *, end_silence_ms: int):
            assert end_silence_ms == 400
            return transport

        async def release_realtime_asr(self, candidate) -> None:
            await candidate.close()

        def postprocess_asr_text(self, text: str) -> str:
            return text

        def note_realtime_asr_failure(self, error) -> None:
            self.failure = str(error)

    service = CloudVoice()
    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(web_app, "voice_service_factory", lambda: service)
    monkeypatch.setattr(web_app, "vad_engine_factory", lambda: vad_engine)

    with TestClient(app).websocket_connect("/api/voice/stt/stream") as websocket:
        websocket.send_json({"type": "start"})
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_bytes(first_chunk)
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'speech_start', 'reason': 'server_vad'}
        assert websocket.receive_json()["type"] == "partial"
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'asr_fallback', 'provider': 'local_whisper', 'reason': 'cloud_disconnected'}
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'speech_start', 'start_ms': 0}
        websocket.send_bytes(second_chunk)
        assert websocket.receive_json()["type"] == "speech_end"
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'final', 'text': '语音问题', 'reason': 'silence', 'provider': 'local_whisper', 'start_ms': 0, 'end_ms': 200}

    assert bytes(vad_engine.session.audio) == first_chunk + second_chunk
    assert transport.audio == [first_chunk]
    assert transport.closed is True
    assert service.failure == "Qwen connection dropped"


def test_web_streaming_stt_websocket_emits_vad_and_final(tmp_path, monkeypatch) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save_vad(True, 400)
    fake_voice = FakeVoiceService()
    fake_vad = FakeVadEngine()
    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(web_app, "voice_service_factory", lambda: fake_voice)
    monkeypatch.setattr(web_app, "vad_engine_factory", lambda: fake_vad)
    client = TestClient(app)

    with client.websocket_connect("/api/voice/stt/stream") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "sample_rate": 16000,
                "channels": 1,
                "encoding": "pcm_s16le",
                "end_silence_ms": 400,
            }
        )
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'ready', 'frame_ms': 100, 'sample_rate': 16000, 'end_silence_ms': 400}
        websocket.send_bytes(b"\x00\x00" * 1600)
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'speech_start', 'start_ms': 0}
        websocket.send_bytes(b"\x00\x00" * 1600)
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'speech_end', 'start_ms': 0, 'end_ms': 400, 'reason': 'silence'}
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'final', 'text': '语音问题', 'reason': 'silence', 'start_ms': 0, 'end_ms': 400}

    assert fake_vad.config.end_silence_ms == 400
    assert fake_voice.last_stt_filename == "vad-utterance.wav"


def test_web_streaming_stt_manual_stop_without_speech(tmp_path, monkeypatch) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save_vad(True, 400)
    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(web_app, "vad_engine_factory", lambda: FakeVadEngine())
    client = TestClient(app)

    with client.websocket_connect("/api/voice/stt/stream") as websocket:
        websocket.send_json({"type": "start"})
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_json({"type": "stop"})
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'no_speech', 'reason': 'final'}


def test_web_wakeup_session_keeps_audio_local_until_keyword_then_streams_asr(
    tmp_path, monkeypatch
) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save_vad(True, 800)
    service_state = {"open_calls": 0}
    connection_started = threading.Event()
    allow_connection = threading.Event()

    class WakeEvent:
        keyword = "小绿小绿"

    class WakeSession:
        def __init__(self) -> None:
            self.audio: list[bytes] = []
            self.reset_calls = 0

        def accept_pcm16(self, chunk: bytes):
            self.audio.append(chunk)
            if service_state["open_calls"] == 0:
                return WakeEvent() if len(self.audio) == 2 else None
            return WakeEvent() if chunk == interrupt_wakeup_chunk else None

        def reset(self) -> None:
            self.reset_calls += 1

    class WakeEngine:
        def __init__(self) -> None:
            self.sessions: list[WakeSession] = []

        def create_session(self):
            session = WakeSession()
            self.sessions.append(session)
            return session

    class AsrEvent:
        def __init__(self, event_type: str, text: str) -> None:
            self.type = event_type
            self.text = text

    class RealtimeAsr:
        def __init__(self) -> None:
            self.audio: list[bytes] = []
            self.closed = False

        async def append_audio(self, chunk: bytes) -> None:
            self.audio.append(chunk)

        async def events(self):
            import asyncio

            expected_question_bytes = len(question_chunk_1) + len(question_chunk_2)
            while sum(len(chunk) for chunk in self.audio) < expected_question_bytes:
                await asyncio.sleep(0)
            yield AsrEvent("partial", "小绿，小绿，绿色江河是什么")
            yield AsrEvent("completed", "小绿小绿，绿色江河是什么组织")

        async def finish(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    transport = RealtimeAsr()

    class SilentRealtimeAsr:
        def __init__(self) -> None:
            self.audio: list[bytes] = []
            self.closed = False

        async def append_audio(self, chunk: bytes) -> None:
            self.audio.append(chunk)

        async def events(self):
            if False:
                yield None

        async def finish(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    follow_up_transport = SilentRealtimeAsr()

    class CloudVoice:
        async def open_realtime_asr(self, *, end_silence_ms: int):
            import asyncio

            assert end_silence_ms == 800
            service_state["open_calls"] += 1
            if service_state["open_calls"] == 1:
                connection_started.set()
                while not allow_connection.is_set():
                    await asyncio.sleep(0)
                return transport
            return follow_up_transport

        async def release_realtime_asr(self, candidate) -> None:
            assert candidate in {transport, follow_up_transport}
            await candidate.close()

        def postprocess_asr_text(self, text: str) -> str:
            return text

    wake_engine = WakeEngine()
    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(web_app, "wakeword_engine_factory", lambda: wake_engine)
    monkeypatch.setattr(web_app, "voice_service_factory", lambda: CloudVoice())

    first_chunk = b"\x01\x00" * 1600
    wakeup_chunk = b"\x02\x00" * 1600
    question_chunk_1 = b"\x03\x00" * 1600
    question_chunk_2 = b"\x04\x00" * 1600
    interrupt_wakeup_chunk = b"\x05\x00" * 1600
    ignored_during_interrupt_ack = b"\x06\x00" * 1600
    with TestClient(app).websocket_connect("/api/voice/session/stream") as websocket:
        websocket.send_json({"type": "start"})
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'armed', 'keyword': '小绿小绿', 'sample_rate': 16000, 'end_silence_ms': 800, 'cooldown_ms': 800}
        websocket.send_bytes(first_chunk)
        websocket.send_bytes(wakeup_chunk)
        wake_payload = websocket.receive_json()
        assert wake_payload["type"] == "wake_detected"
        assert wake_payload["keyword"] == "小绿小绿"
        assert wake_payload["prompt_text"] == web_app.WAKEUP_ACKNOWLEDGEMENT_TEXT
        assert wake_payload["prompt_audio_url"] == web_app.WAKEUP_ACKNOWLEDGEMENT_URL
        assert not connection_started.is_set()

        # Microphone frames emitted while the acknowledgement is playing are
        # discarded and must never reach cloud ASR.
        websocket.send_bytes(b"\x09\x00" * 1600)
        websocket.send_json({"type": "prompt_finished"})
        assert connection_started.wait(timeout=1)

        # The user starts talking while DashScope is still connecting. These
        # frames must remain queued and reach the transport in their original
        # order once the connection is ready.
        websocket.send_bytes(question_chunk_1)
        websocket.send_bytes(question_chunk_2)
        allow_connection.set()
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'listening', 'keyword': '小绿小绿', 'end_silence_ms': 800}
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'speech_start', 'reason': 'server_vad'}
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'partial', 'text': '绿色江河是什么', 'provider': 'dashscope_realtime'}
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'speech_end', 'reason': 'server_vad'}
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'final', 'text': '绿色江河是什么组织', 'reason': 'server_vad', 'provider': 'dashscope_realtime'}

        # Once ASR is complete, PCM is routed to the independent wake-word
        # monitor while answer generation/TTS proceeds. A hit interrupts that
        # turn and always requests the fixed "我在" acknowledgement.
        websocket.send_bytes(interrupt_wakeup_chunk)
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'wake_detected', 'keyword': '小绿小绿', 'prompt_text': web_app.INTERRUPT_ACKNOWLEDGEMENT_TEXT, 'prompt_audio_url': web_app.INTERRUPT_ACKNOWLEDGEMENT_URL, 'reason': 'interrupt'}

        # Microphone frames remain discarded until the acknowledgement has
        # actually finished. Only prompt_finished may open the next VAD turn.
        websocket.send_bytes(ignored_during_interrupt_ack)
        websocket.send_json({"type": "prompt_finished"})
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'listening', 'keyword': '小绿小绿', 'end_silence_ms': 800}
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'no_speech', 'reason': 'session_finished'}

        # An empty follow-up explicitly returns the session to local wake-word
        # detection instead of opening another ASR turn.
        websocket.send_json({"type": "resume"})
        assert websocket.receive_json()["type"] == "armed"

    assert service_state["open_calls"] == 2
    assert b"".join(transport.audio) == question_chunk_1 + question_chunk_2
    assert transport.closed is True
    assert follow_up_transport.audio == []
    assert follow_up_transport.closed is True
    assert len(wake_engine.sessions) == 1
    assert wake_engine.sessions[0].reset_calls == 3


def test_web_wakeup_qwen_connect_failure_uses_buffered_audio_locally(
    tmp_path, monkeypatch
) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save_vad(True, 400)
    connection_started = threading.Event()
    allow_failure = threading.Event()
    first_question_chunk = b"\x31\x00" * 1600
    second_question_chunk = b"\x32\x00" * 1600

    class WakeEvent:
        keyword = "小绿小绿"

    class WakeSession:
        def __init__(self) -> None:
            self.reset_calls = 0

        def accept_pcm16(self, _chunk: bytes):
            return WakeEvent()

        def reset(self) -> None:
            self.reset_calls += 1

    class WakeEngine:
        def __init__(self) -> None:
            self.session = WakeSession()

        def create_session(self):
            return self.session

    class TrackingVadSession:
        def __init__(self) -> None:
            self.audio = bytearray()
            self.feed_count = 0

        def feed_pcm16(self, data: bytes, *, is_final: bool = False):
            self.audio.extend(data)
            self.feed_count += 1
            if self.feed_count == 1:
                return [FakeVadEvent("speech_start", start_ms=0)]
            return [
                FakeVadEvent(
                    "speech_end",
                    start_ms=0,
                    end_ms=200,
                    reason="silence",
                    audio_pcm=bytes(self.audio),
                )
            ]

    class TrackingVadEngine:
        def __init__(self) -> None:
            self.session = TrackingVadSession()

        def create_session(self, _config=None):
            return self.session

    class CloudVoice(FakeVoiceService):
        async def open_realtime_asr(self, *, end_silence_ms: int):
            assert end_silence_ms == 400
            connection_started.set()
            while not allow_failure.is_set():
                await asyncio.sleep(0)
            return None

        def postprocess_asr_text(self, text: str) -> str:
            return text

    wake_engine = WakeEngine()
    vad_engine = TrackingVadEngine()
    service = CloudVoice()
    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(web_app, "wakeword_engine_factory", lambda: wake_engine)
    monkeypatch.setattr(web_app, "vad_engine_factory", lambda: vad_engine)
    monkeypatch.setattr(web_app, "voice_service_factory", lambda: service)

    with TestClient(app).websocket_connect("/api/voice/session/stream") as websocket:
        websocket.send_json({"type": "start"})
        assert websocket.receive_json()["type"] == "armed"
        websocket.send_bytes(b"\x01\x00" * 1600)
        assert websocket.receive_json()["type"] == "wake_detected"
        websocket.send_json({"type": "prompt_finished"})
        assert connection_started.wait(timeout=1)
        websocket.send_bytes(first_question_chunk)
        websocket.send_bytes(second_question_chunk)
        allow_failure.set()
        assert websocket.receive_json()["type"] == "listening"
        assert _without_retrieval_metadata(websocket.receive_json()) == {'type': 'asr_fallback', 'provider': 'local_whisper', 'reason': 'cloud_connect_failed'}
        assert websocket.receive_json()["type"] == "speech_start"
        assert websocket.receive_json()["type"] == "speech_end"
        assert websocket.receive_json()["provider"] == "local_whisper"
        websocket.send_json({"type": "resume"})
        assert websocket.receive_json()["type"] == "armed"

    assert bytes(vad_engine.session.audio) == first_question_chunk + second_question_chunk
    assert wake_engine.session.reset_calls == 2


def test_web_answer_wait_detects_interrupt_and_waits_for_acknowledgement() -> None:
    wake_chunk = b"\x21\x00" * 1600
    ignored_during_prompt = b"\x22\x00" * 1600

    class WakeEvent:
        keyword = "小绿小绿"
        score = 0.99

    class WakeSession:
        def __init__(self) -> None:
            self.reset_calls = 0

        def accept_pcm16(self, chunk: bytes):
            return WakeEvent() if chunk == wake_chunk else None

        def reset(self) -> None:
            self.reset_calls += 1

    class WakeEngine:
        def create_session(self):
            return WakeSession()

    class WebSocket:
        def __init__(self) -> None:
            self.incoming = [
                {"type": "websocket.receive", "bytes": wake_chunk},
                {"type": "websocket.receive", "bytes": ignored_during_prompt},
                {
                    "type": "websocket.receive",
                    "text": json.dumps({"type": "prompt_finished"}),
                },
            ]
            self.sent: list[dict[str, object]] = []

        async def receive(self):
            return self.incoming.pop(0)

        async def send_json(self, payload: dict[str, object]) -> None:
            self.sent.append(payload)

    session = WakeSession()
    holder = [session]
    socket = WebSocket()
    action = asyncio.run(
        web_app._wait_for_voice_next_action(
            socket,
            wake_session=session,
            wake_engine=WakeEngine(),
            wakeup_word="小绿小绿",
            wake_session_holder=holder,
        )
    )

    assert action == "interrupt"
    assert holder[0] is session
    assert session.reset_calls == 1
    assert socket.sent == [
        {
            "type": "wake_detected",
            "keyword": "小绿小绿",
            "prompt_text": web_app.INTERRUPT_ACKNOWLEDGEMENT_TEXT,
            "prompt_audio_url": web_app.INTERRUPT_ACKNOWLEDGEMENT_URL,
            "reason": "interrupt",
            "score": 0.99,
        }
    ]


def test_web_asr_helpers_propagate_disconnect_instead_of_receiving_again(
    monkeypatch,
) -> None:
    class DisconnectSocket:
        async def receive(self):
            return {"type": "websocket.disconnect"}

        async def send_json(self, _payload):
            return None

    class VadSession:
        def feed_pcm16(self, _audio, *, is_final: bool):
            del is_final
            return []

    class VadEngine:
        def create_session(self, _config):
            return VadSession()

    class RealtimeTransport:
        async def append_audio(self, _audio):
            return None

        async def events(self):
            await asyncio.Event().wait()
            if False:
                yield None

    monkeypatch.setattr(web_app, "vad_engine_factory", lambda: VadEngine())
    config = web_app.VadConfig()
    with pytest.raises(WebSocketDisconnect):
        asyncio.run(
            web_app._run_local_asr_stream(
                DisconnectSocket(),
                object(),
                config,
            )
        )
    with pytest.raises(WebSocketDisconnect):
        asyncio.run(
            web_app._run_realtime_asr_stream(
                DisconnectSocket(),
                RealtimeTransport(),
                object(),
            )
        )


def test_web_armed_resume_rearms_detector_and_confirms_armed(
    tmp_path, monkeypatch
) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save_vad(True, 400)

    class WakeSession:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.closed = False

        def accept_pcm16(self, _chunk: bytes):
            return None

        def reset(self) -> None:
            self.reset_calls += 1

        def close(self) -> None:
            self.closed = True

    class WakeEngine:
        def __init__(self) -> None:
            self.session = WakeSession()

        def create_session(self):
            return self.session

    engine = WakeEngine()
    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(web_app, "wakeword_engine_factory", lambda: engine)

    with TestClient(app).websocket_connect("/api/voice/session/stream") as websocket:
        websocket.send_json({"type": "start"})
        initial = websocket.receive_json()
        assert initial["type"] == "armed"

        # Typed/manual answers finish while this socket is still armed.  The
        # resume command must reset KWS and explicitly release the browser's
        # suspended AudioWorklet with another armed event.
        websocket.send_json({"type": "resume"})
        resumed = websocket.receive_json()
        assert resumed == initial
        websocket.send_json({"type": "stop"})

    assert engine.session.reset_calls == 1
    assert engine.session.closed is True


def test_web_answer_wait_replaces_wake_session_when_reset_fails() -> None:
    wake_chunk = b"\x23\x00" * 1600

    class WakeEvent:
        keyword = "小绿小绿"

    class BrokenWakeSession:
        def __init__(self) -> None:
            self.closed = False

        def accept_pcm16(self, chunk: bytes):
            return WakeEvent() if chunk == wake_chunk else None

        def reset(self) -> None:
            raise RuntimeError("reset unsupported")

        def close(self) -> None:
            self.closed = True

    class ReplacementWakeSession:
        pass

    replacement = ReplacementWakeSession()

    class WakeEngine:
        def create_session(self):
            return replacement

    class WebSocket:
        def __init__(self) -> None:
            self.incoming = [
                {"type": "websocket.receive", "bytes": wake_chunk},
                {
                    "type": "websocket.receive",
                    "text": json.dumps({"type": "prompt_finished"}),
                },
            ]

        async def receive(self):
            return self.incoming.pop(0)

        async def send_json(self, _payload: dict[str, object]) -> None:
            return None

    session = BrokenWakeSession()
    holder = [session]
    action = asyncio.run(
        web_app._wait_for_voice_next_action(
            WebSocket(),
            wake_session=session,
            wake_engine=WakeEngine(),
            wakeup_word="小绿小绿",
            wake_session_holder=holder,
        )
    )

    assert action == "interrupt"
    assert session.closed is True
    assert holder[0] is replacement


def test_web_wakeup_asr_ignores_completed_wakeup_word_before_question() -> None:
    class Event:
        def __init__(self, text: str) -> None:
            self.type = "completed"
            self.text = text

    class Transport:
        async def events(self):
            yield Event("小绿小绿")
            yield Event("介绍一下绿色江河")

    class VoiceService:
        def postprocess_asr_text(self, text: str) -> str:
            return text

    class WebSocket:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send_json(self, payload: dict[str, object]) -> None:
            self.messages.append(payload)

    websocket = WebSocket()
    asyncio.run(
        web_app._forward_realtime_asr_events(
            websocket,
            Transport(),
            VoiceService(),
            wakeup_word="小绿小绿",
        )
    )

    assert [_without_retrieval_metadata(e) for e in websocket.messages] == [
        {"type": "speech_start", "reason": "server_vad"},
        {"type": "speech_end", "reason": "server_vad"},
        {
            "type": "final",
            "text": "介绍一下绿色江河",
            "reason": "server_vad",
            "provider": "dashscope_realtime",
        },
    ]


def test_web_voice_status_degrades_wakeup_without_breaking_manual_asr(
    tmp_path, monkeypatch
) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    fake_voice = FakeVoiceService()
    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(web_app, "voice_service_factory", lambda: fake_voice)
    monkeypatch.setattr(
        web_app,
        "wakeword_engine_factory",
        lambda: (_ for _ in ()).throw(RuntimeError("wakeup model missing")),
    )

    client = TestClient(app)
    status = client.get("/api/voice/status")
    assert status.status_code == 200
    assert status.json()["wakeup_available"] is False
    assert status.json()["wakeup_error"] == "wakeup model missing"
    assert status.json()["stt_available"] is True

    response = client.post(
        "/api/voice/stt",
        files={"audio": ("manual.wav", b"fake-wave", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "语音问题"}


def test_web_page_exposes_persistent_offline_wakeup_controls() -> None:
    html = web_app.index()

    assert 'id="wakeupToggle" type="checkbox" checked' in html
    assert 'id="wakeupStatus"' in html
    assert "/api/voice/session/stream" in html
    assert "startWakeupListening" in html
    assert "wake_detected" in html
    assert "playWakeupAcknowledgement" in html
    assert "prompt_finished" in html
    assert "/api/voice/wakeup-acknowledgement" in html
    assert "/api/voice/interrupt-acknowledgement" in html
    assert "scheduleWakeupResume" in html
    assert "scheduleWakeupContinuation" in html
    assert "scheduleWakeupAction('continue')" in html
    assert "function isConversationClosingUtterance(text)" not in html
    assert "closeContinuousConversation" not in html
    assert "function finishWakeupConversation(recorder, text)" not in html
    assert "event.reason ||" in html
    assert "!wakeupRecorder.pendingWakeupForce" in html
    assert "正在返回待唤醒状态" in html
    assert "has reset the KWS session and emitted the next `armed` event" in html
    assert "仍可点击麦克风提问" in html

    wakeup_handler = html.split(
        "async function handleWakeupSocketMessage", 1
    )[1].split("async function toggleVoiceRecording", 1)[0]
    assert "recorder.partialText = text" in wakeup_handler
    assert "payload.text || recorder.partialText || ''" in wakeup_handler
    assert "采用最后识别结果" in wakeup_handler
    assert "const submitted = await sendChatMessage(true)" in wakeup_handler
    assert "isConversationClosingUtterance" not in wakeup_handler
    assert "finishWakeupConversation" not in wakeup_handler
    assert "scheduleWakeupResume()" in wakeup_handler
    assert "stripWakeupWord" not in wakeup_handler
    assert "submitWakeupQuestion(recorder, text)" in wakeup_handler
    assert "await sendChatMessage(true)" not in wakeup_handler.replace(
        "`const submitted = await sendChatMessage(true)`", ""
    )
    assert "await interruptActiveTurn(recorder, payload)" in wakeup_handler
    assert "const activeTurn = Boolean(" in wakeup_handler
    assert "isAnswering ||" in wakeup_handler
    assert "ttsQueueRunPromise ||" in wakeup_handler
    assert "currentVoiceAudio" in wakeup_handler
    interrupt_handler = html.split(
        "async function interruptActiveTurn", 1
    )[1].split("function submitWakeupQuestion", 1)[0]
    assert "if (!promptCompleted)" in interrupt_handler
    assert "await stopWakeupListening('“我在”播放失败，正在恢复待唤醒')" in interrupt_handler


def test_web_serves_wakeup_acknowledgement_audio() -> None:
    response = TestClient(app).get("/api/voice/wakeup-acknowledgement")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content.startswith((b"ID3", b"\xff"))


def test_web_serves_interrupt_acknowledgement_audio() -> None:
    response = TestClient(app).get("/api/voice/interrupt-acknowledgement")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content.startswith((b"ID3", b"\xff"))


def test_web_index_job_stats_and_search(tmp_path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    store = SettingsStore(settings_path)
    store.save_api_key("sk-web-index-12345678")
    index_service = IndexingService(
        settings_store=store,
        vector_store=InMemoryVectorStore(),
        manifest=IndexManifest(tmp_path / "index.sqlite3"),
        embedding_client=FakeEmbeddingClient(),
        bm25_index=BM25Index(tmp_path / "bm25.sqlite3"),
    )
    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(web_app, "indexing_service_factory", lambda: index_service)
    client = TestClient(app)

    parse_response = client.post(
        "/api/parse",
        files=[("files", ("note.txt", b"alpha searchable content", "text/plain"))],
    )
    assert parse_response.status_code == 200
    parse_job_id = parse_response.json()["job_id"]

    index_response = client.post(f"/api/index/jobs/{parse_job_id}")
    assert index_response.status_code == 200
    index_job_id = index_response.json()["index_job_id"]

    job_response = client.get(f"/api/index/jobs/{index_job_id}")
    assert job_response.status_code == 200
    job = job_response.json()
    assert job["status"] == "completed"
    assert job["results"][0]["vector_count"] == 1

    stats_response = client.get("/api/index/stats")
    assert stats_response.status_code == 200
    assert stats_response.json()["document_count"] == 1
    assert stats_response.json()["vector_count"] == 1
    assert stats_response.json()["bm25_chunk_count"] == 1
    assert "sparse_chunk_count" in stats_response.json()

    search_response = client.post("/api/search", json={"query": "alpha", "top_k": 1})
    assert search_response.status_code == 200
    search_payload = search_response.json()
    assert search_payload["query_tokens"] == ["alpha"]
    assert "rewrite" not in search_payload
    assert "alpha searchable content" in search_payload["results"][0]["text"]
    assert search_payload["results"][0]["metadata"]["retrieval_source"] in {"vector", "hybrid"}
    assert search_payload["results"][0]["explanation"]["query_tokens"] == ["alpha"]


def test_web_document_management_list_disable_enable_and_delete(tmp_path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    store = SettingsStore(settings_path)
    store.save_api_key("sk-web-documents-12345678")
    index_service = IndexingService(
        settings_store=store,
        vector_store=InMemoryVectorStore(),
        manifest=IndexManifest(tmp_path / "index.sqlite3"),
        embedding_client=FakeEmbeddingClient(),
        bm25_index=BM25Index(tmp_path / "bm25.sqlite3"),
    )

    class FakeImageManagementService:
        def list_document_images(self, document_id: str) -> list[dict]:
            return []

        def delete_document_images(self, document_id: str) -> dict:
            return {"document_id": document_id, "image_vectors": 0}

    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(web_app, "indexing_service_factory", lambda: index_service)
    monkeypatch.setattr(web_app, "image_indexing_service_factory", lambda: FakeImageManagementService())
    client = TestClient(app)

    parse_response = client.post(
        "/api/parse",
        files=[("files", ("managed.txt", b"alpha managed document", "text/plain"))],
    )
    assert parse_response.status_code == 200
    parse_job_id = parse_response.json()["job_id"]
    index_response = client.post(f"/api/index/jobs/{parse_job_id}")
    assert index_response.status_code == 200

    documents_response = client.get("/api/documents")
    assert documents_response.status_code == 200
    documents = documents_response.json()["documents"]
    assert len(documents) == 1
    document_id = documents[0]["document_id"]
    assert documents[0]["enabled"] is True
    assert documents[0]["index_status"] == "success"

    detail_response = client.get(f"/api/documents/{document_id}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["document"]["source_name"] == "managed.txt"
    assert detail["chunks"][0]["text"] == "alpha managed document"

    disabled_response = client.patch(f"/api/documents/{document_id}", json={"enabled": False})
    assert disabled_response.status_code == 200
    assert disabled_response.json()["document"]["enabled"] is False
    disabled_search = client.post("/api/search", json={"query": "alpha", "top_k": 1})
    assert disabled_search.status_code == 200
    assert disabled_search.json()["results"] == []

    enabled_response = client.patch(f"/api/documents/{document_id}", json={"enabled": True})
    assert enabled_response.status_code == 200
    enabled_search = client.post("/api/search", json={"query": "alpha", "top_k": 1})
    assert enabled_search.status_code == 200
    assert enabled_search.json()["results"]

    delete_response = client.delete(f"/api/documents/{document_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"]["text_vectors"] == 1
    assert client.get("/api/documents").json()["documents"] == []


def test_web_answer_endpoint_returns_answer_and_grounded_citations(tmp_path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    store = SettingsStore(settings_path)
    store.save_api_key("sk-web-answer-12345678")
    store.save_deepseek_api_key("sk-web-answer-deepseek-12345678")

    class FakeAnswerService:
        def answer(self, query: str, *, top_k: int = 4) -> AnswerResult:
            assert query == "alpha?"
            assert top_k == 4
            return AnswerResult(
                query=query,
                answer="alpha 来自文档证据 [1]。",
                citations=[
                    Citation(
                        id=1,
                        source_name="note.txt",
                        chunk_id="note.txt:s0:c0:abc",
                        heading_path="Intro",
                        quote="alpha searchable content",
                    )
                ],
                evidence=[
                    EvidenceChunk(
                        id=1,
                        search_result_id="vector-1",
                        text="alpha searchable content",
                        source_name="note.txt",
                        chunk_id="note.txt:s0:c0:abc",
                        heading_path="Intro",
                    )
                ],
                model="deepseek-v4-flash",
                confidence="high",
                retrieval={"context_top_k": 4, "rewrite": {"method": "fallback"}, "results": []},
            )

    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(web_app, "answer_generation_service_factory", lambda: FakeAnswerService())
    client = TestClient(app)

    response = client.post("/api/answer", json={"query": "alpha?", "top_k": 9})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "alpha 来自文档证据 [1]。"
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["citations"][0]["source_name"] == "note.txt"
    assert payload["citations"][0]["chunk_id"] == "note.txt:s0:c0:abc"
    assert payload["retrieval"]["context_top_k"] == 4


def test_web_answer_stream_endpoint_returns_ndjson_events(tmp_path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    store = SettingsStore(settings_path)
    store.save_api_key("sk-web-stream-12345678")
    store.save_deepseek_api_key("sk-web-stream-deepseek-12345678")

    cancellation_callbacks = []

    class FakeStreamAnswerService:
        def stream_answer_events(
            self,
            query: str,
            *,
            top_k: int = 4,
            should_cancel=None,
        ):
            assert query == "alpha stream?"
            assert top_k == 4
            cancellation_callbacks.append(should_cancel)
            yield {"type": "meta", "model": "deepseek-v4-flash", "evidence": [], "retrieval": {"context_top_k": 4}}
            yield {"type": "delta", "text": "alpha "}
            yield {"type": "delta", "text": "answer [1]"}
            yield {
                "type": "final",
                "result": {
                    "query": query,
                    "answer": "alpha answer [1]",
                    "citations": [{"id": 1, "source_name": "note.txt", "chunk_id": "chunk-1"}],
                    "evidence": [],
                    "model": "deepseek-v4-flash",
                    "confidence": "medium",
                    "retrieval": {"context_top_k": 4},
                },
            }

    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(web_app, "answer_generation_service_factory", lambda: FakeStreamAnswerService())
    client = TestClient(app)

    response = client.post("/api/answer/stream", json={"query": "alpha stream?", "top_k": 9})

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-accel-buffering"] == "no"
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert [event["type"] for event in events] == ["meta", "delta", "delta", "final"]
    assert events[2]["text"] == "answer [1]"
    assert events[-1]["result"]["citations"][0]["source_name"] == "note.txt"
    assert len(cancellation_callbacks) == 1
    assert callable(cancellation_callbacks[0])


def test_web_answer_stream_opts_in_to_local_llm_input_events(tmp_path, monkeypatch) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save_api_key("sk-web-trace-deepseek-12345678")
    received_options = []

    class FakeTracedAnswerService:
        def stream_answer_events(
            self,
            query: str,
            *,
            top_k: int = 4,
            should_cancel=None,
            include_llm_inputs: bool = False,
        ):
            received_options.append((query, top_k, should_cancel, include_llm_inputs))
            yield {
                "type": "llm_input",
                "stage": "rag_answer",
                "title": "RAG 回答",
                "model": "deepseek-v4-flash",
                "messages": [
                    {"role": "system", "content": "system prompt"},
                    {"role": "user", "content": "context and question"},
                ],
            }
            yield {"type": "delta", "text": "answer"}

    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(
        web_app,
        "answer_generation_service_factory",
        lambda: FakeTracedAnswerService(),
    )

    response = TestClient(app).post(
        "/api/answer/stream",
        json={"query": "trace question", "top_k": 4},
    )
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]

    assert response.status_code == 200
    assert [event["type"] for event in events] == ["llm_input", "delta"]
    assert events[0]["messages"][-1]["content"] == "context and question"
    assert "sk-web-trace" not in response.text
    assert len(received_options) == 1
    assert received_options[0][0:2] == ("trace question", 4)
    assert callable(received_options[0][2])
    assert received_options[0][3] is True


def test_web_answer_stream_ignores_nonsemantic_turn(tmp_path, monkeypatch) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save_api_key("sk-web-gate-12345678")

    class FakeIgnoredService:
        def stream_answer_events(self, query: str, *, top_k: int = 4):
            assert query == "嗯"
            assert top_k == 4
            yield {
                "type": "ignored",
                "signal": "[[IGNORE_TURN]]",
                "reason": "只有拟声词。",
                "model": "deepseek-v4-flash",
            }

    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(
        web_app,
        "answer_generation_service_factory",
        lambda: FakeIgnoredService(),
    )
    client = TestClient(app)

    response = client.post("/api/answer/stream", json={"query": "嗯", "top_k": 4})

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events == [
        {
            "type": "ignored",
            "signal": "[[IGNORE_TURN]]",
            "reason": "只有拟声词。",
            "model": "deepseek-v4-flash",
        }
    ]


def test_web_answer_stream_uses_farewell_for_session_end(tmp_path, monkeypatch) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save_api_key("sk-web-farewell-deepseek-12345678")

    class FakeFarewellService:
        def stream_answer_events(self, query: str, *, top_k: int = 4):
            yield {
                "type": "meta",
                "model": "deepseek-v4-flash",
                "retrieval": {
                    "bypassed": True,
                    "reason": "end_session",
                    "turn_signal": "[[END_SESSION]]",
                },
            }
            yield {"type": "delta", "text": "感谢交流，再见！"}
            yield {
                "type": "final",
                "result": {
                    "query": query,
                    "answer": "感谢交流，再见！",
                    "citations": [],
                    "evidence": [],
                    "model": "deepseek-v4-flash",
                    "confidence": "medium",
                    "retrieval": {"bypassed": True, "reason": "end_session"},
                },
            }

    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(
        web_app,
        "answer_generation_service_factory",
        lambda: FakeFarewellService(),
    )
    client = TestClient(app)

    response = client.post(
        "/api/answer/stream",
        json={"query": "没有别的问题了", "top_k": 4},
    )

    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert response.status_code == 200
    assert [event["type"] for event in events] == ["meta", "delta", "final"]
    assert events[-1]["result"]["retrieval"]["reason"] == "end_session"


def test_web_image_index_job_stats_and_image_search(tmp_path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    store = SettingsStore(settings_path)
    store.save_api_key("sk-web-image-index-12345678")
    image_path = tmp_path / "embedded.png"
    Image.new("RGB", (32, 32), "red").save(image_path)
    image_service = ImageIndexingService(
        settings_store=store,
        vector_store=InMemoryVectorStore(collection_name="rag_images"),
        embedding_client=FakeImageEmbeddingClient(),
    )
    monkeypatch.setattr(web_app, "settings_store", store)
    monkeypatch.setattr(web_app, "image_indexing_service_factory", lambda: image_service)
    parse_job_id = "image-parse-job"
    document = ParsedDocument(
        source=str(tmp_path / "demo.docx"),
        source_name="demo.docx",
        backend="auto",
        assets=[ParsedAsset(asset_id="asset-1", kind="image", path=str(image_path))],
    )
    with web_app._jobs_lock:
        web_app._jobs[parse_job_id] = {
            "job_id": parse_job_id,
            "status": "completed",
            "progress": 1,
            "total": 1,
            "results": [{"source": str(document.source), "status": "success", "document": asdict(document)}],
            "errors": [],
        }
    client = TestClient(app)

    index_response = client.post(f"/api/image-index/jobs/{parse_job_id}")
    assert index_response.status_code == 200
    image_index_job_id = index_response.json()["image_index_job_id"]

    job_response = client.get(f"/api/image-index/jobs/{image_index_job_id}")
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "completed"
    assert job_response.json()["results"][0]["vector_count"] == 1

    stats_response = client.get("/api/image-index/stats")
    assert stats_response.status_code == 200
    assert stats_response.json()["vector_count"] == 1
    assert stats_response.json()["model"] == "qwen3-vl-embedding"

    with image_path.open("rb") as handle:
        search_response = client.post(
            "/api/image-search",
            files={"file": ("query.png", handle, "image/png")},
        )

    assert search_response.status_code == 200
    result = search_response.json()["results"][0]
    assert result["metadata"]["source_name"] == "demo.docx"
    assert result["image_data_url"].startswith("data:image/png;base64,")


def _without_retrieval_metadata(event):
    event = dict(event)
    if event.get("provider") == "dashscope_realtime":
        if event.get("type") == "final":
            assert len(event.pop("retrieval_token")) == 32
            event.pop("item_id")
        elif event.get("type") == "partial":
            assert isinstance(event.pop("confirmed_text"), str)
            assert isinstance(event.pop("stash"), str)
            event.pop("item_id")
    return event
