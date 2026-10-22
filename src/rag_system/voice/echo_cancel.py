"""Process-scoped PulseAudio AEC pairing the robot microphone and speaker."""
from __future__ import annotations
import json
import os
import subprocess

SOURCE = 'greenriver_voice_aec_source'
SINK = 'greenriver_voice_aec_sink'


def prepare_echo_cancel() -> dict:
    def call(*args):
        return subprocess.check_output(['pactl', *args], text=True, timeout=5,
                                       env=dict(os.environ, LC_ALL='C')).strip()

    modules = json.loads(call('--format=json', 'list', 'modules'))
    for module in modules:
        if module['name'] == 'module-echo-cancel':
            args = module.get('argument', '')
            if f'source_name={SOURCE}' in args and f'sink_name={SINK}' in args:
                os.environ.update(PULSE_SOURCE=SOURCE, PULSE_SINK=SINK)
                return {'module':module.get('index'), 'source':SOURCE, 'sink':SINK, 'reused':True}
    source_master = os.environ.get('RAG_ROBOT_PULSE_SOURCE') or (
        os.environ.get('PULSE_SOURCE') if os.environ.get('PULSE_SOURCE') != SOURCE else None
    ) or call('get-default-source')
    sink_master = os.environ.get('RAG_ROBOT_PULSE_SINK') or (
        os.environ.get('PULSE_SINK') if os.environ.get('PULSE_SINK') != SINK else None
    ) or call('get-default-sink')
    if SOURCE in source_master or SINK in sink_master:
        raise RuntimeError('AEC physical source/sink must be explicit')
    index = call('load-module', 'module-echo-cancel',
                 f'source_master={source_master}', f'sink_master={sink_master}',
                 f'source_name={SOURCE}', f'sink_name={SINK}', 'aec_method=webrtc',
                 'aec_args="analog_gain_control=0 digital_gain_control=0 noise_suppression=1"',
                 'rate=16000', 'channels=1')
    os.environ.update(PULSE_SOURCE=SOURCE, PULSE_SINK=SINK)
    return {'module':int(index), 'source':SOURCE, 'sink':SINK,
            'source_master':source_master, 'sink_master':sink_master, 'reused':False}
