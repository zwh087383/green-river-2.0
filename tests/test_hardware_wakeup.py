from __future__ import annotations

import asyncio
import json
import os
import select
import shutil
import subprocess
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from rag_system.hardware_wakeup import FrameParser, HardwareWakeSession, SerialWakeSource, encode_frame, wake_metadata
from rag_system.settings import SettingsStore
import rag_system.web.app as web_app


def packet(message, ident=8, kind=4):
    return encode_frame(kind, json.dumps(message).encode(), ident)


def old_wake():
    return {'type': 'aiui_event', 'content': {'eventType': '4', 'info': json.dumps({'ivw': {'angle': 90}})}}


def test_parser_resynchronizes_after_noise_bad_checksum_and_split_packets():
    good = packet(old_wake())
    corrupt = good[:-1] + bytes([good[-1] ^ 1])
    parser = FrameParser()
    assert parser.feed(b'junk\xa5\x01\x04\xff\xff\0\0\0' + corrupt + good[:6]) == []
    result = parser.feed(good[6:] + good)
    assert len(result) == 2 and result[0][0:2] == (4, 8)
    assert wake_metadata(json.loads(result[0][2])) == {'angle': 90}


@pytest.mark.parametrize('message', [
    {'type': 'version', 'code': 0, 'content': 'version: fake'},
    {'type': 'wakeup_keywords', 'code': 107, 'content': 'Change keyword failed!'},
    {'type': 'aiui_event', 'content': {'eventType': '5'}},
    {'type': 'aiui_event', 'content': 'bad'}, {},
])
def test_only_actual_wake_events_are_accepted(message):
    assert wake_metadata(message) is None


def test_new_firmware_wake_event_and_untrusted_numeric_metadata():
    assert wake_metadata({'type': 'aiui_event', 'content': {'beam': 0, 'angle': 20, 'confidence': 950}}) == {
        'beam': 0, 'angle': 20, 'confidence': 950}
    assert wake_metadata({'type': 'aiui_event', 'content': {'eventType': 4, 'info': 'invalid'}}) == {}


def test_session_rejects_pcm_replays_wrong_connections_and_cooldown(monkeypatch):
    now = [100.]
    monkeypatch.setattr('rag_system.hardware_wakeup.time.monotonic', lambda: now[0])
    session = HardwareWakeSession('小绿小绿', 'connection', 800)
    assert session.accept_pcm16(b'\0\0' * 100) is None
    event = {'type': 'hardware_wake', 'connection_id': 'connection', 'sequence': 1}
    assert session.accept_control(event).keyword == '小绿小绿'
    session.reset()
    for invalid in [{**event}, {**event, 'sequence': 2}, {**event, 'sequence': True},
                    {**event, 'sequence': 100, 'connection_id': 'old'}]:
        assert session.accept_control(invalid) is None
    now[0] += 1
    assert session.accept_control({**event, 'sequence': 2}) is None
    assert session.accept_control({**event, 'sequence': 3}) is not None


@pytest.mark.skipif(os.name != 'posix', reason='Linux serial PTY integration')
def test_serial_source_handshake_version_wake_dedup_and_disconnect(tmp_path):
    import pty
    master, slave = pty.openpty()
    path = os.ttyname(slave)
    events, reports = [], []
    status = tmp_path/'status.json'
    source = SerialWakeSource(path, events.append, keyword='小绿小绿', status_path=status,
                              report=reports.append, cooldown_ms=100)
    parser = FrameParser()
    commands = []
    def wait(predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if select.select([master], [], [], .01)[0]:
                commands.extend(parser.feed(os.read(master, 4096)))
            if predicate(): return
        raise AssertionError((events, reports))
    source.start()
    try:
        wait(lambda: bool(commands))
        os.write(master, encode_frame(1, b'\xa5\0\0\0', 3))
        os.write(master, packet({'type': 'version', 'code': 0, 'content': 'version: circle6:test'}, 9))
        wait(lambda: source.state['connected'])
        os.write(master, packet(old_wake(), 10))
        wait(lambda: len(events) == 1)
        os.write(master, packet(old_wake(), 10))
        os.write(master, packet({'type': 'wakeup_keywords', 'code': 107}, 11))
        time.sleep(.15)
        assert len(events) == 1
        os.write(master, packet(old_wake(), 12))
        wait(lambda: len(events) == 2)
        assert [event['sequence'] for event in events] == [1, 2]
        assert any(kind == 255 and ident == 3 for kind, ident, _ in commands)
        # Runtime sends only version queries and handshake ACKs; no audio,
        # firmware, keyword-write or forced-wake commands.
        assert all(kind == 255 or json.loads(payload) == {'type': 'version'} for kind, _, payload in commands)
        os.close(master)
        master = None
        deadline = time.monotonic() + 2
        while source.state['connected'] and time.monotonic() < deadline: time.sleep(.01)
        assert not source.state['connected']
    finally:
        source.close()
        if master is not None: os.close(master)
        os.close(slave)


@pytest.mark.parametrize('interrupt_source', ['hardware', 'software'])
def test_hardware_websocket_keeps_initial_wake_and_selects_interrupt_source(tmp_path, monkeypatch, interrupt_source):
    store = SettingsStore(tmp_path/'settings.json')
    store.save_wakeup(True, '小绿小绿', 1., .25, 0)
    opened, transports = [], []
    class ASR:
        def __init__(self): self.audio = []; self.closed = False
        async def append_audio(self, audio): self.audio.append(audio)
        async def events(self):
            while not self.audio: await asyncio.sleep(0)
            yield SimpleNamespace(type='completed', text='绿色江河是什么')
        async def close(self): self.closed = True
        async def finish(self): pass
    class Voice:
        async def open_realtime_asr(self, **kwargs):
            opened.append(kwargs)
            asr = ASR(); transports.append(asr)
            return asr
        async def release_realtime_asr(self, transport): await transport.close()
        def postprocess_asr_text(self, text): return text
    monkeypatch.setattr(web_app, 'settings_store', store)
    monkeypatch.setattr(web_app, 'voice_service_factory', Voice)
    kws_audio, kws_closed = [], []
    class KWS:
        def accept_pcm16(self, pcm):
            kws_audio.append(pcm)
            return SimpleNamespace(keyword='小绿小绿') if pcm == b'\x03\0' * 1600 else None
        def reset(self): pass
        def close(self): kws_closed.append(True)
    def factory():
        assert interrupt_source == 'software', 'hardware-only session must not load KWS'
        return SimpleNamespace(create_session=KWS)
    monkeypatch.setattr(web_app, 'wakeword_engine_factory', factory)
    connection = '0123456789abcdef'
    wake = {'type': 'hardware_wake', 'connection_id': connection, 'sequence': 1}
    with TestClient(web_app.app).websocket_connect('/api/voice/session/stream') as ws:
        ws.send_json({'type': 'start', 'wake_source': 'hardware',
                      'interrupt_source': interrupt_source, 'connection_id': connection})
        assert ws.receive_json()['type'] == 'armed'
        ws.send_bytes(b'\x01\0' * 1600)
        ws.send_json({**wake, 'connection_id': 'stale'})
        ws.send_json({'type': 'resume'})
        assert ws.receive_json()['type'] == 'armed'
        assert not opened
        assert not kws_audio  # PCM cannot replace the initial serial wake.
        ws.send_json(wake)
        event = ws.receive_json()
        assert event['type'] == 'wake_detected' and 'reason' not in event
        assert not opened
        ws.send_json({**wake, 'sequence': 2})  # ignored while prompt is playing
        ws.send_json({'type': 'prompt_finished'})
        assert ws.receive_json()['type'] == 'listening'
        ws.send_json({**wake, 'sequence': 3})  # ignored while collecting question
        ws.send_bytes(b'\x02\0' * 1600)
        while ws.receive_json()['type'] != 'final': pass
        ws.send_json(wake)  # replay ignored even after a detector reset
        ws.send_json({**wake, 'sequence': 4})
        if interrupt_source == 'software':
            ws.send_bytes(b'\x03\0' * 1600)
        event = ws.receive_json()
        assert event['type'] == 'wake_detected' and event['reason'] == 'interrupt'
        assert event['prompt_text'] == '我在'
        assert event['prompt_audio_url'] == web_app.INTERRUPT_ACKNOWLEDGEMENT_URL
        if interrupt_source == 'software':
            assert kws_audio == [b'\x03\0' * 1600]
        ws.send_json({'type': 'prompt_finished'})
        assert ws.receive_json()['type'] == 'listening'
        ws.send_bytes(b'\x02\0' * 1600)
        while ws.receive_json()['type'] != 'final': pass
        ws.send_json({'type': 'resume'})
        assert ws.receive_json()['type'] == 'armed'
        ws.send_bytes(b'\x03\0' * 1600)
        ws.send_json({**wake, 'sequence': 5})
        event = ws.receive_json()
        assert event['type']=='wake_detected' and 'reason' not in event
        ws.send_json({'type': 'stop'})
    assert len(opened) == 2
    assert transports[0].audio == [b'\x02\0' * 1600]
    assert all(asr.closed for asr in transports)
    if interrupt_source == 'software':
        assert kws_audio == [b'\x03\0' * 1600] and kws_closed == [True]


def test_hardware_status_reports_expired_gateway_without_opening_serial(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, 'DATA_DIR', tmp_path)
    file = tmp_path/'runtime/hardware-wake.json'
    file.parent.mkdir()
    file.write_text(json.dumps({'source': 'hardware', 'connected': True, 'updated_at': 0}))
    response = TestClient(web_app.app).get('/api/robot/voice/wakeup')
    assert not response.json()['connected'] and '过期' in response.json()['error']


def test_control_page_does_not_auto_start_second_microphone_when_robot_uses_hardware(tmp_path):
    node = shutil.which('node')
    if not node: pytest.skip('node required')
    html = web_app.index()
    source = 'let robotHardwareWakeActive' + html.split('let robotHardwareWakeActive', 1)[1].split('    function describeAsrRuntime', 1)[0]
    file = tmp_path/'voice-status.js'
    file.write_text(source)
    script = r'''
const vm=require('vm'),fs=require('fs'),assert=require('assert');
(async()=>{
 let starts=0,stops=0;
 let payload={robot_wakeup_source:'hardware',wakeup_available:true,vad_enabled:true,wakeup_word:'小绿小绿'};
 const c={wakeupEnabled:true,wakeupRecorder:{},wakeupAutoStartAttempted:false,
   voiceStatusPayload:null,wakeupToggle:{checked:true},wakeupLabel:{},wakeupStatus:{},voiceMicBtn:{},autoSpeakBtn:{},chatStatus:{},autoSpeakEnabled:true,
   fetch:async()=>({json:async()=>payload}),readJsonResponse:async r=>r.json(),describeAsrRuntime:()=>'',
   stopWakeupListening:async()=>{stops++;c.wakeupRecorder=null;},startWakeupListening:async()=>{starts++;},
   renderVoiceConfigSummary:()=>{},applyAutoSpeakState:()=>{}};
 vm.createContext(c);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),c);
 await c.loadVoiceStatus();
 assert.equal(starts,0);assert.equal(stops,1);assert.equal(c.wakeupEnabled,false);assert.equal(c.wakeupToggle.checked,false);
 assert(c.wakeupStatus.textContent.includes('不自动监听'));
 // An explicit user opt-in is retained when settings reload.
 vm.runInContext('browserWakeupExplicitlyEnabled=true; wakeupEnabled=true;',c);
 await c.loadVoiceStatus();assert.equal(stops,1);assert.equal(c.wakeupEnabled,true);
 payload={...payload,robot_wakeup_source:'software'};
 await c.loadVoiceStatus();assert.equal(starts,1);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    subprocess.run([node, '-e', script, str(file)], check=True, capture_output=True, text=True)


def test_hardware_status_label_distinguishes_armed_from_listening(tmp_path):
    node = shutil.which('node')
    if not node: pytest.skip('node required')
    from pathlib import Path
    asset = Path(web_app.__file__).parent/'assets/hardware-wake.js'
    script = r'''
const vm=require('vm'),fs=require('fs'),assert=require('assert');
(async()=>{
 const source=fs.readFileSync(process.argv[1],'utf8');
 for(const [phase,text] of [['armed','待唤醒'],['listening','正在聆听'],['answer','正在回复'],['disconnected','语音连接中断']]){
  const label={}; const c={document:{getElementById:()=>label},AbortSignal,
   fetch:async()=>({ok:true,json:async()=>({source:'hardware',connected:true,keyword:'小绿小绿',gateway_phase:phase})}),setInterval:()=>{}};
  vm.runInNewContext(source,c); await new Promise(setImmediate);
  assert(label.textContent.endsWith(text),label.textContent);
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    subprocess.run([node,'-e',script,str(asset)],check=True,capture_output=True,text=True)
