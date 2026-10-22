import asyncio
import base64
import io
import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag_system.generation.reply_input import ReplyAudioStore, MAX_AUDIO_BYTES
from rag_system.generation.service import AnswerGenerationService
from rag_system.settings import SettingsStore
from rag_system.vad import pcm16_to_wav_bytes
import rag_system.web.app as web
from test_direct_voice import Socket, Index, client

PCM = b'\x01\x02' * 8000
QUERY = '本轮专用转录，不应发给回复模型'


class AudioSocket(Socket):
    def send(self, raw):
        event = json.loads(raw)
        if event['type'] == 'input_audio_buffer.commit':
            self.sent.append(event)
            self.events.append({'type': 'input_audio_buffer.committed', 'item_id': 'audio1'})
        else:
            super().send(raw)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    store = SettingsStore(tmp_path/'settings.json')
    store.save_api_key('sk-offline-test')
    store.save_reply_input(True)
    monkeypatch.setattr(web, 'settings_store', store)
    monkeypatch.setattr(web, 'reply_audio_store', ReplyAudioStore())
    monkeypatch.setattr(web, 'indexing_service_factory', Index)
    return store


def test_audio_store_single_use_mismatch_expiry_and_bounds():
    cache = ReplyAudioStore(ttl=.02, capacity=1)
    first = cache.put('a', PCM)
    second = cache.put('b', PCM)
    with pytest.raises(ValueError):
        cache.take(first, 'a')
    with pytest.raises(ValueError):
        cache.take(second, 'wrong')
    with pytest.raises(ValueError):
        cache.take(second, 'b')
    third = cache.put('c', PCM)
    assert cache.take(third, 'c') == PCM
    with pytest.raises(ValueError):
        cache.take(third, 'c')
    fourth = cache.put('d', PCM)
    time.sleep(.04)
    assert fourth not in cache._items
    for bad in [b'', b'1', b'00' * (MAX_AUDIO_BYTES//2+1)]:
        with pytest.raises(ValueError):
            cache.put('x', bad)


@pytest.mark.parametrize('enabled', [True, False])
def test_web_capture_to_provider_audio_or_text(configured, monkeypatch, enabled):
    configured.save_reply_input(enabled)
    socket, index = AudioSocket(), Index()
    service = AnswerGenerationService(settings_store=configured, indexing_service=index,
                                      answer_client=client(socket))
    service.conversation_history.append('历史问题', '历史回答')
    monkeypatch.setattr(web, 'answer_generation_service_factory', lambda: service)
    messages = []
    class Transport:
        async def events(self):
            yield SimpleNamespace(type='completed', text=QUERY, item_id='turn1')
    class WS:
        async def send_json(self, data):
            messages.append(data)
    asyncio.run(web._forward_realtime_asr_events(WS(), Transport(), SimpleNamespace(),
                                                captured_audio=[PCM[:3000], PCM[3000:]]))
    final = messages[-1]
    assert final['type'] == 'final'
    payload = {'query': QUERY, 'input_audio_token': final.get('input_audio_token')}
    response = TestClient(web.app).post('/api/answer/stream', json=payload)
    events = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200 and events[-1]['type'] == 'final', events
    wire = json.dumps(socket.sent, ensure_ascii=False)
    assert (QUERY in wire) is (not enabled)
    assert '历史问题' in wire and '历史回答' in wire and '保护河流' in wire
    assert index.queries == [QUERY]
    chunks = [e for e in socket.sent if e['type'] == 'input_audio_buffer.append']
    if enabled:
        assert b''.join(base64.b64decode(e['audio']) for e in chunks) == PCM
        kinds = [e['type'] for e in socket.sent]
        assert kinds.index('input_audio_buffer.commit') < kinds.index('response.create')
        assert TestClient(web.app).post('/api/answer/stream', json=payload).status_code == 400
    else:
        assert not chunks and 'input_audio_token' not in final
    trace = next(e for e in events if e['type'] == 'llm_input')
    assert trace['input_mode'] == ('original_audio' if enabled else 'text')


def test_audio_cancellation_during_upload_never_commits_or_generates():
    stop = threading.Event()
    class CancelSocket(AudioSocket):
        def send(self, raw):
            super().send(raw)
            if json.loads(raw)['type'] == 'input_audio_buffer.append':
                stop.set()
    socket = CancelSocket()
    assert list(client(socket).stream([], input_pcm=PCM, should_cancel=stop.is_set)) == []
    assert not any(e['type'] in ['response.create', 'input_audio_buffer.commit'] for e in socket.sent)
    assert socket.closed


def test_audio_requires_commit_ack_and_closes_on_failure():
    socket = Socket()  # Does not acknowledge commit.
    with pytest.raises(ConnectionError):
        list(client(socket).stream([], input_pcm=PCM))
    assert not any(e['type'] == 'response.create' for e in socket.sent)
    assert socket.closed


def test_settings_persist_and_typed_input_is_unchanged(configured, monkeypatch):
    http = TestClient(web.app)
    for value in [False, True]:
        result = http.post('/api/settings/reply-input', json={'enabled': value})
        assert result.status_code == 200
        assert result.json()['reply_use_original_audio'] is value
        assert configured.load().reply_use_original_audio is value
    socket = AudioSocket()
    service = AnswerGenerationService(settings_store=configured, indexing_service=Index(),
                                      answer_client=client(socket))
    monkeypatch.setattr(web, 'answer_generation_service_factory', lambda: service)
    response = http.post('/api/answer/stream', json={'query': QUERY})
    assert json.loads(response.text.splitlines()[-1])['type'] == 'final'
    assert QUERY in json.dumps(socket.sent, ensure_ascii=False)
    assert not any(e['type'] == 'input_audio_buffer.append' for e in socket.sent)


def test_manual_wav_and_local_vad_retain_exact_pcm(configured, monkeypatch):
    monkeypatch.setattr(web, '_run_stt_inference', lambda *a: QUERY)
    http = TestClient(web.app)
    result = http.post('/api/voice/stt', files={'audio': ('recording.wav', pcm16_to_wav_bytes(PCM, sample_rate=16000), 'audio/wav')})
    assert result.status_code == 200
    assert web.reply_audio_store.take(result.json()['input_audio_token'], QUERY) == PCM
    result = http.post('/api/voice/stt', files={'audio': ('bad.wav', b'bad', 'audio/wav')})
    assert result.status_code == 400
    messages = []
    class Event:
        type, audio_pcm = 'speech_end', PCM
        def to_dict(self):
            return {'type': self.type}
    class WS:
        async def send_json(self, value):
            messages.append(value)
    asyncio.run(web._send_vad_events(WS(), [Event()]))
    assert web.reply_audio_store.take(messages[-1]['input_audio_token'], QUERY) == PCM


def test_missing_audio_does_not_silently_use_text(configured):
    with pytest.raises(ValueError):
        web._reply_audio_fields(QUERY, b'')
    result = TestClient(web.app).post('/api/answer/stream', json={'query': QUERY, 'input_audio_token': 'expired'})
    assert result.status_code == 400
