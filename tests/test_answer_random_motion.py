import copy

import pytest

from rag_system import robot_motion as motion
from rag_system.motion_presets import load_catalog


@pytest.mark.parametrize('purpose',sorted(motion.ANSWER_PURPOSES))
@pytest.mark.parametrize('selected',['open_palms','raise_hand','point_far','flex_muscles','speech'])
def test_every_answer_purpose_uses_only_current_custom_answer_pool(monkeypatch,purpose,selected):
    def choose(options):
        assert set(options)=={'open_palms','raise_hand','point_far','flex_muscles','speech'}
        assert not set(options)&{'fly_kiss','heart','wave_goodbye','scratch_head','invitation'}
        return selected
    commands=[]
    monkeypatch.setattr(motion.random,'choice',choose)
    monkeypatch.setattr(motion,'exchange',lambda command,**kwargs:commands.append(command) or {'status':'accepted'})
    client=motion.MotionClient()
    assert client.request('answer-1',1,purpose)
    client._thread.join(1);client.close()
    assert len(commands)==1 and commands[0].split()[::2]==['PRESET',selected]


def test_new_custom_presets_join_pool_and_removal_applies_next_request(monkeypatch):
    catalog=copy.deepcopy(load_catalog())
    new=copy.deepcopy(next(p for p in catalog['presets'] if p['id']=='open_palms'))
    new['id']='new_custom';catalog['presets'].append(new)
    monkeypatch.setattr(motion,'load_catalog',lambda:catalog)
    choices=[]
    monkeypatch.setattr(motion.random,'choice',lambda options:choices.append(options) or options[0])
    motion.select_motion_preset('answer_gesture')
    assert 'new_custom' in choices[-1]
    catalog['presets']=[p for p in catalog['presets'] if p['id']!='new_custom']
    motion.select_motion_preset('answer_gesture')
    assert 'new_custom' not in choices[-1]


def test_empty_pool_skips_motion_without_builtin_fallback_or_retry(monkeypatch):
    catalog=load_catalog()
    catalog['presets']=[p for p in catalog['presets'] if (p['type']=='builtin' and p['id']!='speech') or p['id'] in motion.ANSWER_EXCLUDED_PRESETS]
    monkeypatch.setattr(motion,'load_catalog',lambda:catalog)
    monkeypatch.setattr(motion,'exchange',lambda *a,**k:pytest.fail('No eligible motion may be sent'))
    reports=[];client=motion.MotionClient(report=reports.append)
    assert client.request('answer-1',1,'answer_gesture')
    client._thread.join(1);client.close()
    assert len(reports)==1 and 'No eligible custom answer motions' in reports[0]


def test_wake_and_closing_pools_remain_separate(monkeypatch):
    monkeypatch.setattr(motion,'load_catalog',lambda:pytest.fail('Wake and closing do not use answer pool'))
    monkeypatch.setattr(motion.random,'choice',lambda options:options[-1] if options==motion.CLOSING_PRESETS else pytest.fail('wrong pool'))
    assert motion.select_motion_preset('wake_wave')=='wave_goodbye'
    assert motion.select_motion_preset('closing_wave')=='wave_goodbye'
    assert motion.select_motion_preset('closing_gesture')=='fly_kiss'
