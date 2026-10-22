"""Regression: a spoken farewell must not reopen ASR for an explicit end request."""
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from rag_system.generation.history import ConversationHistory
from rag_system.generation.service import AnswerGenerationService
from rag_system.settings import SettingsStore
from test_realtime_gateway import gateway


class Index:
    def __init__(self): self.queries = []
    def search(self, query, top_k=4):
        self.queries.append(query)
        return []


class FarewellWithoutTool:
    model = 'qwen-audio-3.1-realtime-plus'
    def stream(self, messages, **kwargs):
        self.messages = messages
        yield {'type':'delta','text':'好的，那我先去忙了。'}
        yield {'type':'audio','audio':'AAA=','sample_rate':24000,'channels':1,'format':'pcm16'}
        yield {'type':'reply_done','response_id':'test-reply','audio_bytes':2}


def service_with_history(tmp_path, rounds=5, reply=None):
    history = ConversationHistory()
    for i in range(rounds):
        history.append(f'环保问题{i}', f'介绍环保知识{i}，你还想了解什么？')
    index = Index()
    service = AnswerGenerationService(settings_store=SettingsStore(tmp_path/'settings.json'),
        indexing_service=index, answer_client=reply or FarewellWithoutTool(), conversation_history=history)
    return service, history, index


@pytest.mark.parametrize('rounds', [5,10])
@pytest.mark.parametrize('query', ['没别的事情了','没你事情了','不用你了','先这样吧'])
def test_farewell_without_tool_still_ends_after_multiple_rounds(tmp_path, rounds, query):
    service, history, index = service_with_history(tmp_path, rounds)
    events = list(service.stream_answer_events(query, include_llm_inputs=True))
    assert events[0]['retrieval']['reason'] == 'end_session'
    final = events[-1]['result']
    assert final['answer'] == '好的，那我先去忙了。'
    assert final['retrieval']['turn_control'] == {
        'end_session':True, 'source':'explicit_user_end', 'explicit_end_request':True,
        'model':service.answer_client.model, 'tool_called':False,
    }
    assert not index.queries and not history.snapshot()
    # No old assistant continuation or retrieved passage can dilute this turn.
    assert len(service.answer_client.messages) == 2
    assert '用户已明确结束对话' in service.answer_client.messages[-1]['content']


@pytest.mark.parametrize('query', [
    '没别的事情了', '没有其他问题了。', '小绿，没别的事了，谢谢你。',
    '好的，那我没有其它需求了。', '没有别的事儿了',
    '没你事情了', '没你的事情了', '没你事了', '没有你的事了', '没有你事儿了',
    '不用你了', '我不用你帮忙了', '不需要你了', '不需要你帮忙了', '不再需要你的帮助了',
    '你先退下吧', '退下吧', '不用了，你忙吧', '不用了，你先退下吧', '你先去忙吧',
    '没事了', '没事儿了', '先这样', '先这样吧', '好的，那先这样吧，谢谢你。',
    '我们先聊到这儿', '今天就聊到这里吧', '本次交流到此为止', '结束对话', '退出本次对话吧',
    '再见', '拜拜', '回头见', '下次再聊', '今天不聊了', '我先去忙了', '我先走了',
    '小绿小绿，没你事情了，再见，谢谢你。', '不用你了，先这样吧，辛苦了。',
])
def test_complete_no_further_needs_requests(query):
    from rag_system.generation.turn_control import explicit_end_request
    assert explicit_end_request(query)


@pytest.mark.parametrize('query', [
    '没别的事情了，不过我还想问怎么报名',
    '没有其他问题了，告诉我地址', '别结束，我还有问题',
    '他说没别的事情了是什么意思', '“没别的事情了”',
    '没别的事情了吗', '我先去忙了是什么意思', '谢谢', '好', '是', '对', '不',
    '没你事情了，不过我还想问怎么报名', '不用你了，不过告诉我地址',
    '先这样吧，但别退出，我还有问题', '没事了，测一下吧',
    '不用解释那么多，直接告诉我地址', '我没事，继续讲', '不需要你帮忙搬东西了',
    '不用你介绍了，唱首歌吧', '你先退下吧这句话是什么意思',
    '他说没你事情了是什么意思', '“不用你了”', '他说先这样吧', '再见是什么意思',
    '怎么结束对话', '别结束对话', '不要退出本次对话', '先这样吧？', '不用你了?',
    '不用了', '收到', '辛苦了',
])
def test_other_requests_are_not_ended_by_assistant_farewell(tmp_path, query):
    service, history, index = service_with_history(tmp_path)
    events = list(service.stream_answer_events(query))
    final = events[-1]['result']
    assert final['retrieval']['reason'] == 'direct_retrieval'
    assert final['retrieval']['turn_control']['end_session'] is False
    assert len(history) == 6 and index.queries == [query]


def test_main_model_end_tool_can_confirm_existing_explicit_end(tmp_path):
    class ToolReply(FarewellWithoutTool):
        def stream(self, messages, **kwargs):
            yield {'type':'tool_result','name':'end_conversation','call_id':'c1',
                   'result':{'end_session':True,'status':'finish_after_playback','reason':'没有其他需求'}}
            yield from super().stream(messages, **kwargs)
    service, history, _ = service_with_history(tmp_path, reply=ToolReply())
    events = list(service.stream_answer_events('没别的事情了'))
    control = events[-1]['result']['retrieval']['turn_control']
    assert control['tool_called'] and control['explicit_end_request']
    assert control['source'] == 'main_model_tool' and not history.snapshot()


def test_interrupted_end_reply_does_not_complete_or_clear_history(tmp_path):
    service, history, _ = service_with_history(tmp_path)
    cancel = threading.Event()
    events = service.stream_answer_events('没别的事情了', should_cancel=cancel.is_set)
    assert next(events)['retrieval']['reason'] == 'end_session'
    cancel.set()
    assert list(events) == [] and len(history) == 5


@pytest.mark.parametrize('query', ['没别的事情了','没你事情了','不用你了','先这样吧'])
def test_gateway_returns_to_armed_only_after_farewell_audio_drains(tmp_path, gateway, monkeypatch, query):
    service, _, _ = service_with_history(tmp_path)
    events = list(service.stream_answer_events(query))
    module, node, actions = gateway
    node._gesture = lambda generation, purpose: actions.append(('gesture',purpose))
    class Output:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def write(self, pcm): actions.append(('device','write'))
        def stop(self): actions.append(('device','drained'))
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_lines(self): return (json.dumps(e) for e in events)
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def stream(self, *args, **kwargs): return Response()
    monkeypatch.setitem(sys.modules,'sounddevice',SimpleNamespace(RawOutputStream=Output))
    monkeypatch.setattr(module.httpx,'Client',Client)
    node._reply({'text':query},4,node._cancel)
    assert actions[-1] == ('control','resume')
    assert ('control','continue') not in actions
    assert actions.index(('device','drained')) < actions.index(('playback','completed')) < actions.index(('control','resume'))
    assert [a for a in actions if a[0]=='gesture'] == [('gesture','closing_gesture')]
    node._hardware = None
    node._handle_voice_event({'type':'armed'})
    assert node._wake_phase == 'armed' and not node._capture.is_set()
