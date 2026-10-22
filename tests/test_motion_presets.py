from __future__ import annotations
import copy
import json
import math
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from rag_system.motion_presets import load_catalog, validate_vendor_limits
import rag_system.web.app as web_app

ROOT = Path(__file__).parents[1]


def test_fly_kiss_matches_both_images_and_requested_three_speeds():
    catalog = load_catalog(ROOT/'config/motion_presets.json')
    preset = next(p for p in catalog['presets'] if p['id'] == 'fly_kiss')
    assert [f['scale'] for f in preset['frames']] == [.6, .7, .6]
    assert preset['frames'][0]['left_arm'] == [-14.77,67.54,-43.46,27.83,3.33,-3.25,-2.61]
    assert preset['frames'][0]['right_arm'] == [94.54,38.76,5.,110.69,22.29,24.79,-14.47]
    assert preset['frames'][1]['left_arm'] == [-15.04,69.48,-27.90,25.41,-24.08,-3.29,1.32]
    assert preset['frames'][1]['right_arm'] == [107.37,40.55,5.,24.84,86.83,-.93,-14.48]
    assert preset['frames'][2]['home'] is True


@pytest.mark.parametrize('change', [
    lambda p: p['frames'][0].update(scale=True),
    lambda p: p['frames'][0].update(scale=1.1),
    lambda p: p['frames'][0].update(scale=float('nan')),
    lambda p: p['frames'][0].update(left_arm=[1]*6),
    lambda p: p['frames'][0].update(left_arm=[False]*7),
    lambda p: p['frames'][0].update(wait_ms=-1),
    lambda p: p['frames'][-1].update(home=False),
    lambda p: p.update(id='../unsafe'),
])
def test_invalid_catalog_is_rejected_before_hardware(tmp_path, change):
    catalog = load_catalog(ROOT/'config/motion_presets.json')
    preset = next(p for p in catalog['presets'] if p['id']=='fly_kiss')
    change(preset)
    path=tmp_path/'catalog.json';path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError): load_catalog(path)


def test_catalog_discovers_future_presets_without_code_changes(tmp_path):
    data=load_catalog(ROOT/'config/motion_presets.json')
    path=tmp_path/'catalog.json';path.write_text(json.dumps(data))
    before=load_catalog(path)
    new=copy.deepcopy(data['presets'][-1]);new.update(id='next_action',name='后续动作')
    data['presets'].append(new);path.write_text(json.dumps(data))
    after=load_catalog(path)
    assert len(after['presets'])==len(before['presets'])+1 and after['revision']!=before['revision']


def test_scratch_head_removed_and_other_presets_unchanged():
    current = load_catalog(ROOT/'config/motion_presets.json')
    assert 'scratch_head' not in {p['id'] for p in current['presets']}
    assert {'wave_goodbye', 'heart', 'fly_kiss'} <= {p['id'] for p in current['presets']}


def test_vendor_limits_checked_for_all_joint_frames(tmp_path):
    sdk=tmp_path/'sdk';model=sdk/'install/share/robot_description/urdf/wheel_robot.urdf'
    model.parent.mkdir(parents=True)
    model.write_text('<robot>'+''.join(
        f'<joint name="AR5_{side}joint_{i}"><limit lower="-3" upper="3"/></joint>'
        for side in ('L','R') for i in range(1,8))+'</robot>')
    preset=next(p for p in load_catalog(ROOT/'config/motion_presets.json')['presets'] if p['id']=='fly_kiss')
    validate_vendor_limits(preset,sdk)
    preset['frames'][1]['right_arm'][5]=179
    with pytest.raises(ValueError,match='第 6 关节'): validate_vendor_limits(preset,sdk)


@pytest.fixture
def client(monkeypatch):
    catalog=load_catalog(ROOT/'config/motion_presets.json')
    monkeypatch.setattr(web_app,'load_motion_catalog',lambda:catalog)
    monkeypatch.setattr(web_app,'validate_vendor_limits',lambda _:None)
    return TestClient(web_app.app),catalog


def test_catalog_and_page_gets_never_dispatch_motion(client,monkeypatch):
    api,catalog=client
    def forbidden(*args):raise AssertionError('GET must not move arms')
    monkeypatch.setattr(web_app,'request_robot_preset',forbidden)
    assert api.get('/api/robot/motion/presets').json()==catalog
    html=api.get('/').text
    assert 'id="page-motions"' in html and 'data-page="motions"' in html
    assert api.get('/api/robot/motion/presets-client.js').status_code==200
    assert api.get('/api/robot/motion/presets-style.css').status_code==200


@pytest.mark.parametrize('status,code',[('accepted',202),('duplicate',200),('busy',409),('not_at_home',409),('unavailable',503),('invalid_preset',503)])
def test_preset_endpoint_reports_acceptance_not_completion(client,monkeypatch,status,code):
    api,catalog=client;calls=[]
    monkeypatch.setattr(web_app,'request_robot_preset',lambda token,preset:calls.append((token,preset)) or {'status':status,'state':'running'})
    response=api.post('/api/robot/motion/presets/fly_kiss/run',json={'request_id':'trial-1','catalog_revision':catalog['revision']})
    assert response.status_code==code and calls==[('trial-1','fly_kiss')]


def test_stale_catalog_unknown_actions_and_cross_origin_never_move(client,monkeypatch):
    api,catalog=client
    def forbidden(*args):raise AssertionError('must not dispatch hardware')
    monkeypatch.setattr(web_app,'request_robot_preset',forbidden)
    payload={'request_id':'trial-1','catalog_revision':catalog['revision']}
    route='/api/robot/motion/presets/fly_kiss/run'
    assert api.post(route,json=payload,headers={'Origin':'https://outside.test'}).status_code==403
    assert api.post(route,json=payload,headers={'Sec-Fetch-Site':'cross-site'}).status_code==403
    assert api.post(route,json={**payload,'catalog_revision':'0'*64}).status_code==409
    assert api.post('/api/robot/motion/presets/no_such_action/run',json=payload).status_code==404
    assert api.post('/api/robot/motion/presets/scratch_head/run',json=payload).status_code==404
    assert api.post(route,json={**payload,'request_id':'x\nPRESET bad'}).status_code==422
    assert api.post(route,json={'request_id':'trial-1'}).status_code==422


def test_bad_vendor_joint_target_prevents_dispatch(client,monkeypatch):
    api,catalog=client
    def invalid(_):raise ValueError('关节超出范围')
    monkeypatch.setattr(web_app,'validate_vendor_limits',invalid)
    monkeypatch.setattr(web_app,'request_robot_preset',lambda *a:pytest.fail('must not dispatch'))
    response=api.post('/api/robot/motion/presets/fly_kiss/run',json={'request_id':'trial-1','catalog_revision':catalog['revision']})
    assert response.status_code==422
