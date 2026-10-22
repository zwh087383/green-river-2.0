from __future__ import annotations

import base64
import json
import threading
import time

import httpx
import pytest

from rag_system.generation.history import ConversationHistory
from rag_system.generation.live_retrieval import RetrievalPrefetch, register, consume
from rag_system.generation.qwen_audio import QwenAudioReplyClient, AudioReplyError
from rag_system.generation.service import AnswerGenerationService, local_turn_control
from rag_system.indexing.embeddings import EmbeddingGemmaClient
from rag_system.indexing.models import SearchResult
from rag_system.settings import SettingsStore


class Socket:
    def __init__(self, *, wrong_owner=False, incomplete=False, interrupt=None):
        self.events = [{"type": "session.created"}]
        self.sent = []
        self.closed = False
        self.wrong_owner = wrong_owner
        self.incomplete = incomplete
        self.interrupt = interrupt

    def send(self, raw):
        event = json.loads(raw)
        self.sent.append(event)
        kind = event["type"]
        if kind == "session.update":
            self.events.append({"type": "session.updated", "session": {
                "modalities": event["session"]["modalities"], "output_audio_format": "pcm"}})
        elif kind == "conversation.item.create":
            self.events.append({"type": "conversation.item.created", "item": event["item"]})
        elif kind == "response.create":
            owner = "stale" if self.wrong_owner else "reply1"
            self.events.extend([
                {"type": "response.created", "response": {"id": "reply1"}},
                {"type": "response.audio_transcript.delta", "response_id": owner, "delta": "环保从身边做起。"},
                {"type": "response.audio.delta", "response_id": owner, "delta": base64.b64encode(b'\0\0' * 10).decode()},
            ])
            if not self.incomplete:
                self.events.append({"type": "response.done", "response": {"id": owner, "status": "completed", "usage": {"total_tokens": 10}}})

    def recv(self, timeout):
        if self.events:
            event = self.events.pop(0)
            if self.interrupt and event["type"] == "response.audio_transcript.delta":
                self.interrupt.set()
            return json.dumps(event)
        raise ConnectionError("stream disconnected")

    def close(self):
        self.closed = True


def client(socket):
    return QwenAudioReplyClient(api_key="test", connect=lambda *a, **k: socket)


def test_qwen_context_acked_before_response_and_pcm_streamed():
    socket = Socket()
    events = list(client(socket).stream([{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}]))
    assert [e["type"] for e in events] == ["delta", "audio", "reply_done"]
    assert events[1]["sample_rate"] == 24000
    assert [e["type"] for e in socket.sent] == ["session.update", "conversation.item.create", "response.create"]
    assert socket.sent[0]["session"]["turn_detection"] is None
    assert socket.closed


@pytest.mark.parametrize("options", [{"wrong_owner": True}, {"incomplete": True}])
def test_qwen_rejects_wrong_owner_or_truncated_stream(options):
    socket = Socket(**options)
    with pytest.raises((AudioReplyError, ConnectionError)):
        list(client(socket).stream([{"role": "user", "content": "question"}]))
    assert socket.sent[-1]["type"] == "response.cancel"
    assert socket.closed


def test_qwen_cancellation_closes_provider_without_completion():
    cancelled = threading.Event()
    socket = Socket(interrupt=cancelled)
    events = list(client(socket).stream([{"role": "user", "content": "q"}], should_cancel=cancelled.is_set))
    assert not any(e["type"] == "reply_done" for e in events)
    assert socket.sent[-1]["type"] == "response.cancel"
    assert socket.closed


class ToolSocket(Socket):
    def __init__(self, *, arguments='{"reason":"明确道别"}', tool_name='end_conversation',
                 wrong_call=False, incomplete_call=False, repeat=False):
        super().__init__()
        self.arguments, self.tool_name = arguments, tool_name
        self.wrong_call, self.incomplete_call, self.repeat = wrong_call, incomplete_call, repeat
        self.responses = 0

    def send(self, raw):
        event = json.loads(raw)
        if event['type'] != 'response.create':
            return super().send(raw)
        self.sent.append(event)
        self.responses += 1
        owner = 'r' + str(self.responses)
        self.events.append({'type': 'response.created', 'response': {'id': owner}})
        if self.responses == 1 or self.repeat:
            self.events.append({'type':'response.output_item.added','response_id':owner,
                'item':{'id':'fc1','type':'function_call','call_id':'call1','name':self.tool_name}})
            if not self.incomplete_call:
                self.events.append({'type':'response.function_call_arguments.done','response_id':owner,
                    'item_id':'fc1','call_id':'stale' if self.wrong_call else 'call1',
                    'name':self.tool_name,'arguments':self.arguments})
        else:
            self.events.extend([
                {'type':'response.audio_transcript.delta','response_id':owner,'delta':'再见！'},
                {'type':'response.audio.delta','response_id':owner,'delta':'AAA='},
            ])
        self.events.append({'type':'response.done','response':{'id':owner,'status':'completed'}})


def test_end_tool_returns_result_then_continues_same_session_for_farewell():
    socket = ToolSocket()
    events = list(client(socket).stream([{'role':'user','content':'再见'}]))
    assert [e['type'] for e in events] == ['tool_result','delta','audio','reply_done']
    assert events[0]['result']['end_session'] is True
    assert socket.sent[0]['session']['tools'][0]['function']['name'] == 'end_conversation'
    output = next(e['item'] for e in socket.sent if e.get('item',{}).get('type') == 'function_call_output')
    assert output['call_id'] == 'call1'
    assert json.loads(output['output'])['status'] == 'finish_after_playback'
    assert socket.responses == 2 and socket.closed


@pytest.mark.parametrize('options', [
    {'arguments':'{}'}, {'tool_name':'unknown'}, {'wrong_call':True}, {'incomplete_call':True},
])
def test_invalid_or_incomplete_tool_never_signals_ending(options):
    socket = ToolSocket(**options)
    events = []
    with pytest.raises(AudioReplyError):
        for event in client(socket).stream([{'role':'user','content':'再见'}]):
            events.append(event)
    assert not any(e['type'] == 'tool_result' for e in events)
    assert not any(e.get('item',{}).get('type') == 'function_call_output' for e in socket.sent)
    assert socket.closed


def test_repeated_end_tool_is_bounded_and_cannot_loop():
    socket = ToolSocket(repeat=True)
    with pytest.raises(AudioReplyError, match='Repeated'):
        list(client(socket).stream([{'role':'user','content':'再见'}]))
    assert socket.responses == 2


def test_prepared_qwen_only_configures_until_final_context_is_acknowledged():
    socket = Socket()
    reply = client(socket)
    prepared = reply.prepare('rules')
    assert prepared._ready.wait(2)
    assert [e['type'] for e in socket.sent] == ['session.update']
    events = list(reply.stream([{'role':'system','content':'rules'}, {'role':'user','content':'final evidence'}], prepared=prepared))
    assert events[-1]['type'] == 'reply_done'
    assert [e['type'] for e in socket.sent] == ['session.update', 'conversation.item.create', 'response.create']
    assert socket.closed
    with pytest.raises(AudioReplyError, match='ownership'):
        prepared.claim(reply, 'rules', True, lambda: False)


def test_prepared_qwen_cancel_during_connect_closes_late_socket_without_generating():
    socket = Socket()
    entered, release = threading.Event(), threading.Event()
    def connect(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return socket
    reply = QwenAudioReplyClient(api_key='test', connect=connect)
    prepared = reply.prepare('rules')
    assert entered.wait(2)
    prepared.close()
    release.set()
    assert prepared._ready.wait(2)
    assert socket.closed and not socket.sent


def test_prepared_qwen_propagates_connect_timeout_instead_of_waiting_forever():
    def fail(*args, **kwargs):
        raise TimeoutError('connect timed out')
    reply = QwenAudioReplyClient(api_key='test', connect=fail)
    prepared = reply.prepare('rules')
    assert prepared._ready.wait(2)
    with pytest.raises(TimeoutError, match='connect timed out'):
        list(reply.stream([{'role':'system','content':'rules'}], prepared=prepared))


def test_prepared_qwen_capacity_is_held_until_connection_is_closed(monkeypatch):
    import rag_system.generation.qwen_audio as qwen
    monkeypatch.setattr(qwen, '_PREPARE_SLOTS', threading.BoundedSemaphore(1))
    first = client(Socket()).prepare('rules')
    assert first._ready.wait(2)
    assert client(Socket()).prepare('rules') is None
    first.close()
    second = client(Socket()).prepare('rules')
    assert second is not None and second._ready.wait(2)
    second.close()


@pytest.mark.parametrize('fail_retrieval', [False, True])
def test_qwen_preparation_overlaps_retrieval_and_closes_on_retrieval_failure(tmp_path, fail_retrieval):
    configured = threading.Event()
    class ConfiguredSocket(Socket):
        def recv(self, timeout):
            raw = super().recv(timeout)
            if json.loads(raw)['type'] == 'session.updated':
                configured.set()
            return raw
    socket = ConfiguredSocket()
    index = Index()
    original = index.search
    def search(query, top_k):
        assert configured.wait(2), 'Connection preparation must overlap retrieval'
        assert [e['type'] for e in socket.sent] == ['session.update']
        if fail_retrieval:
            raise RuntimeError('retrieval unavailable')
        return original(query, top_k)
    index.search = search
    service = AnswerGenerationService(settings_store=SettingsStore(tmp_path/'s.json'),
        indexing_service=index, answer_client=client(socket))
    if fail_retrieval:
        with pytest.raises(RuntimeError, match='retrieval unavailable'):
            list(service.stream_answer_events('question'))
        assert len(service.conversation_history) == 0
        assert [e['type'] for e in socket.sent] == ['session.update']
    else:
        assert list(service.stream_answer_events('question'))[-1]['type'] == 'final'
    assert socket.closed


@pytest.mark.parametrize('fail', [False, True])
def test_startup_runs_complete_retrieval_before_serving_and_reports_failure(monkeypatch, fail):
    import asyncio
    from types import SimpleNamespace
    from rag_system.web import app as web
    calls = []
    class Voice:
        async def close_asr_connections(self): pass
        async def close_tts_connection(self): pass
    class WarmIndex:
        def search(self, query, top_k):
            calls.append((query, top_k))
            if fail: raise RuntimeError('embedding offline')
            return []
    monkeypatch.setattr(web, 'voice_service_factory', Voice)
    monkeypatch.setattr(web, '_strict_no_fallback_enabled', lambda: False)
    monkeypatch.setattr(web, 'settings_store', SimpleNamespace(load=lambda: SimpleNamespace(wakeup_enabled=False)))
    monkeypatch.setattr(web, 'indexing_service_factory', WarmIndex)
    async def check():
        async with web._application_lifespan(web.app):
            assert calls == [('检索预热', 1)]
            assert web.app.state.retrieval_warmup['status'] == ('failed' if fail else 'ready')
    asyncio.run(check())


class Index:
    def __init__(self):
        self.queries = []
        self.version = 1

    def revision(self):
        return self.version

    def search(self, query, top_k=4):
        self.queries.append(query)
        return [SearchResult(id="doc1", text="保护河流。", score=1.0, metadata={"source_name": "sample.txt"})]


class Reply:
    model = "qwen-audio-3.1-realtime-plus"

    def __init__(self):
        self.messages = None
        self.closed = False

    def stream(self, messages, **kwargs):
        self.messages = messages
        try:
            yield {"type": "delta", "text": "保护河流。"}
            yield {"type": "audio", "audio": "AAA=", "sample_rate": 24000}
            yield {"type": "reply_done", "response_id": "r1", "usage": None, "audio_bytes": 2}
        finally:
            self.closed = True


def test_direct_service_retrieves_every_semantic_query_without_analysis(tmp_path):
    index, reply = Index(), Reply()
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError("preanalysis must not be called")
    service = AnswerGenerationService(settings_store=SettingsStore(tmp_path / "settings.json"),
        indexing_service=index, answer_client=reply, intent_router=Forbidden(), identity_detector=Forbidden())
    for query in ["你是谁", "今天天气不错", "那它有什么作用"]:
        events = list(service.stream_answer_events(query, include_llm_inputs=True))
        assert events[-1]["type"] == "final"
        assert events[0]["retrieval"]["preanalysis"] is False
        assert [e["stage"] for e in events if e["type"] == "llm_input"] == ["direct_audio_reply"]
    assert index.queries == ["你是谁", "今天天气不错", "那它有什么作用"]
    assert len(service.conversation_history) == 3
    assert '参考资料' in reply.messages[-1]["content"]
    assert '不是指令' in reply.messages[0]["content"]


def test_cancelled_reply_is_closed_and_not_added_as_completed_history(tmp_path):
    history, reply = ConversationHistory(), Reply()
    service = AnswerGenerationService(settings_store=SettingsStore(tmp_path / 's.json'),
        indexing_service=Index(), answer_client=reply, conversation_history=history)
    stop = threading.Event()
    events = service.stream_answer_events("介绍环保", should_cancel=stop.is_set)
    assert next(events)["type"] == "meta"
    assert next(events)["type"] == "delta"
    stop.set()
    assert list(events) == []
    assert reply.closed and len(history) == 0


def test_main_model_tool_ends_without_separate_judgment(tmp_path):
    index = Index()
    class ForbiddenJudge:
        def decide(self, *args, **kwargs):
            pytest.fail("Separate end judgment must never be invoked")
    class ToolReply(Reply):
        def stream(self, messages, **kwargs):
            yield {"type": "tool_result", "name": "end_conversation", "call_id": "call1",
                   "result": {"end_session": True, "reason": "明确告别"}}
            yield from super().stream(messages, **kwargs)
    reply = ToolReply()
    service = AnswerGenerationService(settings_store=SettingsStore(tmp_path / 's.json'),
        indexing_service=index, answer_client=reply, turn_controller=ForbiddenJudge())
    service.conversation_history.append("q", "a")
    assert list(service.stream_answer_events("嗯。"))[0]["type"] == "ignored"
    events = list(service.stream_answer_events("我已经了解清楚了，今天先不打扰你了", include_llm_inputs=True))
    metas = [e for e in events if e["type"] == "meta"]
    assert metas[0]["retrieval"]["reason"] == "direct_retrieval"
    assert metas[-1]["retrieval"]["reason"] == "end_session"
    assert metas[-1]["retrieval"]["turn_control"]["source"] == "main_model_tool"
    assert all(e["retrieval"]["preanalysis"] is False for e in metas)
    assert len(index.queries) == 1 and len(service.conversation_history) == 0
    assert [e["stage"] for e in events if e["type"] == "llm_input"] == ["direct_audio_reply"]
    assert 'end_conversation' in reply.messages[0]["content"]
    assert local_turn_control("再见") == "answer"


def test_no_tool_call_keeps_history_even_when_farewell_words_are_present(tmp_path):
    index, reply = Index(), Reply()
    service = AnswerGenerationService(settings_store=SettingsStore(tmp_path / 's.json'),
        indexing_service=index, answer_client=reply)
    service.conversation_history.append("q", "a")
    events = list(service.stream_answer_events("再见是什么意思", audio=False))
    assert events[0]["retrieval"]["reason"] == "direct_retrieval"
    assert len(index.queries) == 1 and len(service.conversation_history) == 2
    assert events[-1]["result"]["retrieval"]["turn_control"]["tool_called"] is False


def test_cancel_during_retrieval_closes_prepared_reply_without_response(tmp_path):
    stop = threading.Event()
    class Prepared:
        closed = False
        def close(self): self.closed = True
    prepared = Prepared()
    class PreparingReply(Reply):
        def prepare(self, *args, **kwargs): return prepared
    index = Index()
    def search(*args, **kwargs):
        stop.set()
        return []
    index.search = search
    service = AnswerGenerationService(settings_store=SettingsStore(tmp_path / 's.json'),
        indexing_service=index, answer_client=PreparingReply())
    assert list(service.stream_answer_events("介绍环保", should_cancel=stop.is_set)) == []
    assert prepared.closed


@pytest.mark.parametrize('query,tone', [
    ('小绿你真可爱', 'shy'), ('鼓励我', 'encouraging'), ('我很难过', 'concerned'),
    ('你有什么想问的', 'curious'), ('我做到了', 'proud'),
])
def test_stream_carries_interaction_emotion_to_all_clients(tmp_path, query, tone):
    service = AnswerGenerationService(settings_store=SettingsStore(tmp_path / 's.json'),
        indexing_service=Index(), answer_client=Reply())
    events = list(service.stream_answer_events(query))
    tones = [e['tone'] for e in events if e['type'] == 'emotion']
    assert tones and tones[-1] == tone
    assert events[-1]['result']['retrieval']['emotion']['tone'] == tone


def wait_snapshot(prefetch):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with prefetch._lock:
            if not prefetch._running:
                return
        time.sleep(0.005)
    raise AssertionError("prefetch did not settle")


@pytest.mark.parametrize("query,revision,reuse", [("河流", 1, True), ("河流污染", 1, False), ("河流", 2, False)])
def test_prefetch_only_reuses_exact_final_query_and_corpus_version(query, revision, reuse):
    prefetch = RetrievalPrefetch(Index(), interval=0)
    token = register(prefetch)
    prefetch.offer("河流")
    wait_snapshot(prefetch)
    assert (consume(token, query, revision) is not None) == reuse
    assert consume(token, query, revision) is None


def test_prefetch_replaces_pending_query_and_discards_stale_results():
    entered, release = threading.Event(), threading.Event()
    index = Index()
    original = index.search
    def slow(query, top_k):
        if query == "最初":
            entered.set()
            assert release.wait(2)
        return original(query, top_k)
    index.search = slow
    prefetch = RetrievalPrefetch(index, interval=0)
    prefetch.offer("最初")
    assert entered.wait(2)
    prefetch.offer("中间")
    prefetch.offer("最终")
    release.set()
    wait_snapshot(prefetch)
    assert index.queries == ["最初", "最终"]
    assert prefetch.take("最终", 1) is not None


@pytest.mark.parametrize('pending_final', [False, True])
def test_final_claim_joins_running_or_pending_retrieval_once(pending_final):
    from concurrent.futures import ThreadPoolExecutor
    entered, release = threading.Event(), threading.Event()
    index = Index()
    original = index.search
    first = '较早文字' if pending_final else '最终问题'
    def slow(query, top_k):
        if query == first:
            entered.set()
            assert release.wait(2)
        return original(query, top_k)
    index.search = slow
    prefetch = RetrievalPrefetch(index, interval=0)
    token = register(prefetch)
    prefetch.offer(first)
    assert entered.wait(2)
    if pending_final:
        prefetch.offer('最终问题')
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(consume, token, '最终问题', 1)
        time.sleep(.03)
        assert not future.done()
        prefetch.offer('迟到旧轮次')
        release.set()
        assert future.result(timeout=2) is not None
    wait_snapshot(prefetch)
    assert index.queries == ([first, '最终问题'] if pending_final else ['最终问题'])
    assert consume(token, '最终问题', 1) is None


@pytest.mark.parametrize('mode', ['cancel', 'timeout', 'revision'])
def test_inflight_retrieval_never_returns_cancelled_timed_out_or_stale_result(mode):
    entered, release = threading.Event(), threading.Event()
    index = Index()
    original = index.search
    def slow(query, top_k):
        entered.set()
        assert release.wait(2)
        return original(query, top_k)
    index.search = slow
    prefetch = RetrievalPrefetch(index, interval=0)
    prefetch.offer('最终问题')
    assert entered.wait(2)
    try:
        if mode == 'cancel':
            assert prefetch.take('最终问题', 1, should_cancel=lambda: True) is None
        elif mode == 'timeout':
            with pytest.raises(TimeoutError):
                prefetch.take('最终问题', 1, timeout=.01)
        else:
            index.version = 2
            release.set()
            assert prefetch.take('最终问题', 1) is None
    finally:
        release.set()
        wait_snapshot(prefetch)
    assert index.queries == ['最终问题']


def test_cancel_while_joining_prefetch_closes_prepared_reply_without_new_search(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
    index, socket = Index(), Socket()
    original = index.search
    def slow(query, top_k):
        entered.set()
        assert release.wait(3)
        return original(query, top_k)
    index.search = slow
    prefetch = RetrievalPrefetch(index, interval=0)
    token = register(prefetch)
    prefetch.offer('最终问题')
    assert entered.wait(2)
    service = AnswerGenerationService(settings_store=SettingsStore(tmp_path/'s.json'),
        indexing_service=index, answer_client=client(socket))
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: list(service.stream_answer_events(
                '最终问题', retrieval_token=token, should_cancel=cancelled.is_set)))
            for _ in range(200):
                with prefetch._lock:
                    claimed = prefetch._claimed
                if claimed and socket.sent:
                    break
                time.sleep(.005)
            assert claimed and socket.sent
            cancelled.set()
            assert future.result(timeout=2) == []
    finally:
        release.set()
        wait_snapshot(prefetch)
    assert socket.closed
    assert index.queries == ['最终问题']
    assert not any(e['type'] == 'response.create' for e in socket.sent)


def test_embedding_document_and_query_use_distinct_prompts_without_cloud_key(monkeypatch):
    calls = []
    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={"model": "google/embeddinggemma-2", "data": [
            {"index": i, "embedding": [2.0] + [0.0] * 767} for i in range(len(payload['input']))]})
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    client = EmbeddingGemmaClient()
    assert client.embed_texts(["文档"])[0][0] == 1
    client.embed_queries(["问题"])
    assert [call["prompt_name"] for call in calls] == ["Document", "SearchQuery"]


@pytest.mark.parametrize("vector,model", [([0.0] * 768, "google/embeddinggemma-2"), ([1.0] * 1024, "google/embeddinggemma-2"), ([1.0] * 768, "old-model")])
def test_embedding_fails_closed_on_wrong_space(monkeypatch, vector, model):
    original = httpx.Client
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"model": model, "data": [{"index": 0, "embedding": vector}]}))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    with pytest.raises(RuntimeError):
        EmbeddingGemmaClient().embed_queries(["问题"])


def test_short_asr_stash_triggers_speculation_but_preserves_final_text():
    import asyncio
    from types import SimpleNamespace
    from rag_system.web.app import _forward_realtime_asr_with_retrieval
    offered, sent = [], []
    class Transport:
        async def events(self):
            for kind, text in [("partial", "绿色江河"), ("partial", "绿色江河是哪里"), ("completed", "绿色江河是什么组织")]:
                yield SimpleNamespace(type=kind, text=text, confirmed_text="", stash=text, item_id="asr1")
    class Sink:
        async def send_json(self, event): sent.append(event)
    state = {"final": False}
    asyncio.run(_forward_realtime_asr_with_retrieval(Sink(), Transport(), SimpleNamespace(), None,
        SimpleNamespace(offer=offered.append), "turn-token", state))
    assert offered == ["绿色江河是哪里"]
    assert sent[-1]["text"] == "绿色江河是什么组织"
    assert sent[-1]["retrieval_token"] == "turn-token" and state["final"]


def test_web_conversations_do_not_share_history(monkeypatch, tmp_path):
    from rag_system.web import app as web
    monkeypatch.setattr(web, 'answer_generation_service_factory', lambda: AnswerGenerationService(
        settings_store=SettingsStore(tmp_path / 'settings.json'), indexing_service=Index(), answer_client=Reply()))
    a = web._answer_service_for(web.AnswerPayload(query='a', conversation_id='test-a'))
    b = web._answer_service_for(web.AnswerPayload(query='b', conversation_id='test-b'))
    a.conversation_history.append('private-a', 'reply-a')
    assert len(b.conversation_history) == 0
    again = web._answer_service_for(web.AnswerPayload(query='c', conversation_id='test-a'))
    assert again.conversation_history.snapshot()[0].user == 'private-a'


def test_migration_preserves_old_vectors_ids_sources_and_disabled_state(tmp_path, monkeypatch):
    from rag_system.indexing.migrate_embeddinggemma import migrate
    from rag_system.indexing.store import ChromaVectorStore
    from rag_system.indexing.manifest import IndexManifest
    from rag_system.indexing.models import IndexedDocument
    from rag_system.indexing.service import IndexingService
    legacy = ChromaVectorStore(tmp_path / 'chroma', 'rag_chunks')
    legacy.upsert(['old-id'], ['环境保护资料'], [[1.0, 0.0, 0.0]], [{'document_id':'doc1','source_name':'source.txt'}])
    manifest = IndexManifest(tmp_path / 'index.sqlite3')
    manifest.upsert_document(IndexedDocument(document_id='doc1',source='source.txt',source_name='source.txt',
        file_sha256='abc',chunk_count=1,vector_count=1,model='legacy',dimensions=3,indexed_at='old',enabled=False))
    monkeypatch.setattr(EmbeddingGemmaClient,'embed_texts',lambda self,texts:[[1.0]+[0.0]*767 for _ in texts])
    report = migrate(tmp_path, tmp_path / 'settings.json')
    assert report['vector_count'] == 1 and report['disabled_document_count'] == 1
    assert legacy.query([1.0,0.0,0.0])[0].id == 'old-id'
    target = IndexingService(index_dir=tmp_path,settings_store=SettingsStore(tmp_path / 'settings.json'))
    assert target._vector_store().list_all()[0].metadata['source_name'] == 'source.txt'
    assert target.manifest.disabled_document_ids() == {'doc1'}
    assert target.manifest.list_documents()[0].dimensions == 768
    assert migrate(tmp_path,tmp_path / 'settings.json') == report
