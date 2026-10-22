import json
import os
from rag_system.voice.echo_cancel import prepare_echo_cancel, SOURCE, SINK


def test_aec_routes_both_streams_and_preserves_system_defaults(monkeypatch):
    commands=[]
    monkeypatch.setenv('PULSE_SOURCE','physical_iflytek')
    monkeypatch.setenv('PULSE_SINK','physical_speaker')
    def output(cmd, **kwargs):
        commands.append(cmd[1:])
        if cmd[1:]==['--format=json','list','modules']: return '[]'
        assert cmd[1]=='load-module'
        return '25'
    monkeypatch.setattr('subprocess.check_output',output)
    result=prepare_echo_cancel()
    assert result['source_master']=='physical_iflytek' and result['sink_master']=='physical_speaker'
    assert os.environ['PULSE_SOURCE']==SOURCE and os.environ['PULSE_SINK']==SINK
    assert all('set-default-source' not in cmd and 'set-default-sink' not in cmd for cmd in commands)
    assert 'source_master=physical_iflytek' in commands[-1] and 'sink_master=physical_speaker' in commands[-1]


def test_existing_aec_module_is_reused_without_creating_a_second_owner(monkeypatch):
    commands=[]
    def output(cmd,**kwargs):
        commands.append(cmd)
        return json.dumps([{'name':'module-echo-cancel',
                            'argument':f'source_name={SOURCE} sink_name={SINK}'}])
    monkeypatch.setattr('subprocess.check_output',output)
    monkeypatch.setenv('PULSE_SOURCE','old')
    monkeypatch.setenv('PULSE_SINK','old')
    assert prepare_echo_cancel()['reused'] is True
    assert len(commands)==1 and os.environ['PULSE_SOURCE']==SOURCE and os.environ['PULSE_SINK']==SINK
