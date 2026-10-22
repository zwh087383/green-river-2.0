from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
import rag_system.web.app as web_app

@pytest.mark.parametrize('status',['accepted','queued','coalesced','duplicate'])
def test_refresh_endpoint_acknowledges_native_home_without_waiting_for_motion(monkeypatch,status):
    calls=[]
    def home(token):calls.append(token);return {'status':status,'state':'homing'}
    monkeypatch.setattr(web_app,'request_robot_home',home)
    response=TestClient(web_app.app).post('/api/robot/motion/home',json={'request_id':'page-refresh-1'})
    assert response.status_code==202 and response.json()['status']==status
    assert calls==['page-refresh-1']


def test_refresh_never_clears_sdk_fault_or_retries_unavailable_worker(monkeypatch):
    monkeypatch.setattr(web_app,'request_robot_home',lambda token:{'status':'unavailable','state':'fault'})
    client=TestClient(web_app.app)
    assert client.post('/api/robot/motion/home',json={'request_id':'page-1'}).status_code==503
    def missing(token):raise FileNotFoundError('socket absent')
    monkeypatch.setattr(web_app,'request_robot_home',missing)
    assert client.post('/api/robot/motion/home',json={'request_id':'page-2'}).status_code==503


def test_cross_origin_refresh_is_rejected_before_hardware_request(monkeypatch):
    def forbidden(token):raise AssertionError('must not call hardware')
    monkeypatch.setattr(web_app,'request_robot_home',forbidden)
    client=TestClient(web_app.app)
    assert client.post('/api/robot/motion/home',json={'request_id':'page-1'},headers={'Origin':'https://other.example'}).status_code==403
    assert client.post('/api/robot/motion/home',json={'request_id':'bad\nPLAY motion'}).status_code==422
    assert client.post('/api/robot/motion/home',data={'request_id':'form'}).status_code==422


@pytest.mark.parametrize('power', ['on', 'off'])
@pytest.mark.parametrize('status,code', [('accepted', 202), ('duplicate', 200), ('already_on', 200),
                                      ('already_off', 200), ('busy', 409), ('unavailable', 503)])
def test_power_endpoint_reports_acceptance_separately_from_completion(monkeypatch, power, status, code):
    calls = []
    def change(token, target):
        calls.append((token, target))
        return {'status': status, 'state': 'powering_' + power}
    monkeypatch.setattr(web_app, 'request_robot_power', change)
    response = TestClient(web_app.app).post('/api/robot/motion/power', json={'request_id': 'power-1', 'power': power})
    assert response.status_code == code
    assert response.json()['status'] == status
    assert calls == [('power-1', power)]


def test_power_endpoint_rejects_cross_origin_and_malformed_requests_without_hardware(monkeypatch):
    def forbidden(*args): raise AssertionError('must not call hardware')
    monkeypatch.setattr(web_app, 'request_robot_power', forbidden)
    client = TestClient(web_app.app)
    payload = {'request_id': 'power-1', 'power': 'off'}
    for headers in [{'Origin': 'https://other.example'}, {'Sec-Fetch-Site': 'cross-site'}]:
        assert client.post('/api/robot/motion/power', json=payload, headers=headers).status_code == 403
    for invalid in [True, False, 1, 'restart', 'OFF', None]:
        assert client.post('/api/robot/motion/power', json={**payload, 'power': invalid}).status_code == 422
    assert client.post('/api/robot/motion/power', json={**payload, 'request_id': 'bad\nPOWER'}).status_code == 422
    assert client.post('/api/robot/motion/power', data=payload).status_code == 422
    assert client.get('/api/robot/motion/power').status_code == 405
    assert client.get('/').text.count('id="armPowerOnBtn"') == 1
    assert client.get('/api/robot/arm-power.js').status_code == 200


def test_power_transport_failure_does_not_retry(monkeypatch):
    calls = []
    def missing(*args):
        calls.append(args)
        raise TimeoutError('no acknowledgement')
    monkeypatch.setattr(web_app, 'request_robot_power', missing)
    response = TestClient(web_app.app).post('/api/robot/motion/power', json={'request_id': 'power-1', 'power': 'off'})
    assert response.status_code == 503 and len(calls) == 1


@pytest.mark.parametrize('status,code',[('accepted',202),('duplicate',200),('busy',409),('unavailable',503)])
def test_arm_restart_endpoint_is_separate_and_checks_origin(monkeypatch,status,code):
    calls=[]
    monkeypatch.setattr(web_app,'request_robot_arm_restart',lambda token:calls.append(token) or {'status':status})
    monkeypatch.setattr(web_app,'request_robot_all_power',lambda *a:pytest.fail('must not change body'))
    client=TestClient(web_app.app)
    payload={'request_id':'restart-1'}
    for headers in ({'Origin':'https://outside.example'},{'Sec-Fetch-Site':'cross-site'}):
        assert client.post('/api/robot/motion/restart',json=payload,headers=headers).status_code==403
    assert not calls
    assert client.post('/api/robot/motion/restart',json=payload).status_code==code
    assert calls==['restart-1']
    assert client.get('/').text.count('id="armRestartBtn"')==1


def test_restart_endpoint_validates_and_never_retries_unknown_receipt(monkeypatch):
    calls=[]
    def missing(token):calls.append(token);raise TimeoutError('unknown receipt')
    monkeypatch.setattr(web_app,'request_robot_arm_restart',missing)
    client=TestClient(web_app.app)
    assert client.post('/api/robot/motion/restart',json={'request_id':'bad\nPOWER'}).status_code==422
    assert not calls
    assert client.post('/api/robot/motion/restart',json={'request_id':'restart-1'}).status_code==503
    assert calls==['restart-1']


def test_power_browser_reads_only_until_clicked_and_waits_for_completion():
    node = shutil.which('node')
    if not node: pytest.skip('node required')
    source = Path(__file__).resolve().parents[1]/'src/rag_system/web/assets/arm-power.js'
    script = r'''
const vm=require('vm'),fs=require('fs'),assert=require('assert');
(async()=>{
 const elements=Object.fromEntries(['armPowerOnBtn','armPowerOffBtn','armRestartBtn','armPowerStatus','armPowerMessage'].map(id=>[id,{disabled:true,textContent:'',addEventListener(_,fn){this.click=fn;}}]));
 let hardware={status:'ok',state:'ready',power:{left:0,right:0},purpose:'startup_home',power_feedback_age_ms:10}, posts=[], interval;
 const context={document:{hidden:false,getElementById:id=>elements[id]},setInterval:fn=>{interval=fn;},AbortSignal:{timeout:()=>null},crypto:{randomUUID:()=> 'power-1'},
 fetch:async(url,options={})=>{
   if(options.method==='POST'){
     posts.push(JSON.parse(options.body));
     const restart=url==='/api/robot/motion/restart';
     hardware={status:'accepted',state:restart?'restarting_arms':posts.at(-1).power==='off'?'powering_off':'powering_on',request_id:'token',purpose:'power_'+(restart?'restart':posts.at(-1).power)+'_home',power:{left:0,right:0},power_feedback_age_ms:10};
     return {ok:true,json:async()=>hardware};
   }
   return {ok:true,json:async()=>hardware};
 }};
 vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);
 const tick=()=>new Promise(resolve=>setImmediate(resolve));
 await tick(); assert.equal(posts.length,0); assert(elements.armPowerOnBtn.disabled); assert(!elements.armPowerOffBtn.disabled);
 elements.armPowerOffBtn.click(); await tick();
 assert.equal(posts.length,1);assert.equal(posts[0].power,'off');assert(elements.armPowerOnBtn.disabled&&elements.armPowerOffBtn.disabled);
 elements.armPowerOffBtn.click();await tick();assert.equal(posts.length,1);
 hardware={...hardware,status:'ok',state:'powered_off',power:{left:-1,right:-1}};
 interval();await tick();assert(elements.armPowerStatus.textContent.includes('未确认'));assert(!elements.armPowerOnBtn.disabled);
 elements.armPowerOnBtn.click();await tick();assert.equal(posts.length,2);assert.equal(posts[1].power,'on');
 hardware={...hardware,status:'ok',state:'ready',at_preset:true,power:{left:0,right:0}};
 interval();await tick();assert(elements.armPowerMessage.textContent.includes('上电及归位完成'));assert(!elements.armPowerOffBtn.disabled);
 assert(!elements.armRestartBtn.disabled);elements.armRestartBtn.click();await tick();
 assert.equal(posts.length,3);assert(!('power' in posts[2]));assert.equal(hardware.state,'restarting_arms');
 assert(elements.armRestartBtn.disabled&&elements.armPowerOnBtn.disabled&&elements.armPowerOffBtn.disabled);
 elements.armRestartBtn.click();await tick();assert.equal(posts.length,3);
 hardware={...hardware,status:'ok',state:'ready',at_preset:true};interval();await tick();
 assert(elements.armPowerMessage.textContent.includes('手臂重启及归位完成'));
 hardware={...hardware,power_feedback_age_ms:3000};interval();await tick();assert(elements.armRestartBtn.disabled);
 hardware={...hardware,state:'fault',detail:'Preset failed'};interval();await tick();
 assert(elements.armPowerOnBtn.disabled&&elements.armPowerOffBtn.disabled);assert(elements.armPowerMessage.textContent.includes('Preset failed'));
 hardware={...hardware,power_feedback_age_ms:60000};interval();await tick();
 assert(!elements.armRestartBtn.disabled);elements.armRestartBtn.click();await tick();
 assert.equal(posts.length,4);assert(elements.armRestartBtn.disabled);
 hardware={...hardware,state:'fault',restart_pending:true,request_id:'token',purpose:'power_restart_home'};
 interval();await tick();assert(elements.armRestartBtn.disabled);assert(elements.armPowerMessage.textContent.includes('等待旧指令退出'));
 hardware={...hardware,state:'restarting_arms',restart_pending:false};interval();await tick();assert(elements.armRestartBtn.disabled);
 hardware={...hardware,state:'ready',at_preset:true,power_feedback_age_ms:10};interval();await tick();
 assert(!elements.armRestartBtn.disabled);assert(elements.armPowerMessage.textContent.includes('手臂重启及归位完成'));
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    subprocess.run([node, '-e', script, str(source)], check=True, capture_output=True, text=True)


def test_reload_hook_is_present_on_chat_display_and_preview_without_get_side_effect(monkeypatch):
    def forbidden(token):raise AssertionError('GET must not move the robot')
    monkeypatch.setattr(web_app,'request_robot_home',forbidden)
    client=TestClient(web_app.app)
    for page in ['/','/display','/display/preview']:
        response=client.get(page)
        assert response.status_code==200
        assert response.text.count('src="/api/robot/motion-client.js"')==1
    assert client.get('/api/robot/motion-client.js').status_code==200


def test_reload_javascript_only_fires_once_for_top_level_reload():
    node=shutil.which('node')
    if not node:pytest.skip('node is required for the reload hook test')
    source=Path(__file__).resolve().parents[1]/'src/rag_system/web/assets/motion-client.js'
    script=r'''
const vm = require('vm');
const fs = require('fs');
const source = fs.readFileSync(process.argv[1], 'utf8');
(async () => {
  for (const [kind, embedded, expected] of [['reload',false,1],['navigate',false,0],['reload',true,0],['back_forward',false,0]]) {
    const calls=[]; const self={};
    vm.runInNewContext(source, {
      window:{self,top:embedded?{}:self},
      performance:{getEntriesByType:()=>[{type:kind}]},
      crypto:{randomUUID:()=> 'page-1'},
      AbortSignal:{timeout:()=>null},
      fetch:(url,options)=>{calls.push([url,options]);return Promise.resolve({ok:true});},
      console:{debug:()=>{}}
    });
    await Promise.resolve();
    if(calls.length!==expected) throw new Error(kind+' count '+calls.length);
    if(expected && (calls[0][0]!=='/api/robot/motion/home' || calls[0][1].method!=='POST' || JSON.parse(calls[0][1].body).request_id!=='page-1')) throw new Error('bad reload request');
  }
})();
'''
    subprocess.run([node,'-e',script,str(source)],check=True,capture_output=True,text=True)


@pytest.mark.parametrize('power',['on','off'])
@pytest.mark.parametrize('status,code',[('accepted',202),('duplicate',200),('busy',409),('unavailable',503)])
def test_all_power_endpoint_is_separate_and_checks_origin(monkeypatch,power,status,code):
    calls=[]
    monkeypatch.setattr(web_app,'request_robot_all_power',lambda token,value:calls.append((token,value)) or {'status':status})
    monkeypatch.setattr(web_app,'request_robot_power',lambda *args:pytest.fail('arm-only route called'))
    client=TestClient(web_app.app)
    payload={'request_id':'all-1','power':power}
    assert client.post('/api/robot/motion/all-power',json=payload,headers={'Origin':'https://outside.example'}).status_code==403
    assert not calls
    assert client.post('/api/robot/motion/all-power',json=payload).status_code==code
    assert calls==[('all-1',power)]
    html=client.get('/').text
    assert html.count('id="allPowerOnBtn"')==1 and html.count('id="armPowerOnBtn"')==1
    assert client.get('/api/robot/all-power.js').status_code==200


def test_all_power_browser_waits_for_both_groups_and_brakes():
    node=shutil.which('node')
    if not node:pytest.skip('node required')
    source=Path(__file__).resolve().parents[1]/'src/rag_system/web/assets/all-power.js'
    script=r'''
const vm=require('vm'),fs=require('fs'),assert=require('assert');
(async()=>{
 const elements=Object.fromEntries(['allPowerOnBtn','allPowerOffBtn','allPowerStatus','allPowerMessage'].map(id=>[id,{disabled:true,textContent:'',addEventListener(_,fn){this.click=fn;}}]));
 let hardware={status:'ok',state:'ready',power:{left:0,right:0},trunk:{power:[0,0,0,0,0,0],at_home:true,brakes:[1,1,1,1,1,1]},power_feedback_age_ms:10,at_preset:true},posts=[],interval;
 const context={document:{hidden:false,getElementById:id=>elements[id]},setInterval:fn=>interval=fn,AbortSignal:{timeout:()=>null},crypto:{randomUUID:()=> 'full-1'},fetch:async(url,opt={})=>{
   if(opt.method==='POST'){
     assert.equal(url,'/api/robot/motion/all-power');posts.push(JSON.parse(opt.body));
     hardware={...hardware,status:'accepted',state:'powering_off_all',purpose:'all_power_off_home',request_id:'token'};
     return {ok:true,json:async()=>hardware};
   }return {ok:true,json:async()=>hardware};
 }};
 vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);
 const tick=()=>new Promise(r=>setImmediate(r));await tick();assert.equal(posts.length,0);
 assert(!elements.allPowerOffBtn.disabled&&elements.allPowerOnBtn.disabled);
 elements.allPowerOffBtn.click();await tick();elements.allPowerOffBtn.click();await tick();assert.equal(posts.length,1);
 assert(elements.allPowerOffBtn.disabled&&elements.allPowerOnBtn.disabled);
 hardware={...hardware,state:'powered_off',power:{left:1,right:1},trunk:{...hardware.trunk,power:[1,1,1,1,1,1],brakes:[0,0,0,0,0,0]}};
 interval();await tick();assert(elements.allPowerMessage.textContent.includes('抱闸已确认'));
 assert(!elements.allPowerOnBtn.disabled&&elements.allPowerOffBtn.disabled);
 hardware={...hardware,power_feedback_age_ms:3000};interval();await tick();assert(elements.allPowerOnBtn.disabled);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    subprocess.run([node,'-e',script,str(source)],check=True,capture_output=True,text=True)
