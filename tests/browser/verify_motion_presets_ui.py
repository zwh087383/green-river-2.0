"""Headless UI check against a local mock server. Never connects to robot APIs."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from threading import Thread
from urllib.parse import urlparse,parse_qs
import json,subprocess,sys,tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from rag_system.web.app import index
from rag_system.motion_presets import load_catalog
from rag_system.emotion_rules import RULES
catalog=load_catalog(ROOT/'config/motion_presets.json')
state={'status':'ok','state':'ready','at_preset':True,'power':{'left':0,'right':0},'power_feedback_age_ms':20,'completed':0,'purpose':'startup_home','request_id':'startup'}
posts=[]
state_reads=0
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def reply(self,content,kind='application/json'):
  body=content.encode() if isinstance(content,str) else json.dumps(content,ensure_ascii=False).encode()
  self.send_response(200);self.send_header('Content-Type',kind+'; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def do_GET(self):
  global state_reads,state
  route=urlparse(self.path)
  if route.path=='/':
   html=index()
   html=html.replace('</head>','''<script>window.addEventListener('error', e => document.documentElement.dataset.scriptError = e.message);</script></head>''')
   html=html.replace('</body>', '''<script>setInterval(()=>{const p=document.getElementById('page-motions');if(p?.classList.contains('active')&&p.getBoundingClientRect().height>0)document.body.dataset.motionVisible='yes';},100);</script></body>''')
   if parse_qs(route.query).get('test'):
    script='''<script>
let trial=false;
setInterval(()=>{
 const button=document.querySelector('[data-preset-id="fly_kiss"] button');
 if(!trial&&document.body.dataset.motionVisible==='yes'&&button&&button.getBoundingClientRect().height>0&&!button.disabled){trial=true;button.click();button.click();}
 const message=document.getElementById('motionDebugFeedback');
 if(trial&&message?.textContent.includes('已完成'))document.body.dataset.motionTest='passed';
},100);
</script>'''
    html=html.replace('</body>',script+'</body>')
   return self.reply(html,'text/html')
  if route.path=='/api/robot/motion/presets':return self.reply(catalog)
  if route.path=='/api/robot/motion/state':
   if posts:
    state_reads+=1
    if state_reads>=2:state={**state,'state':'ready','at_preset':True,'completed':1,'step_index':3,'step_count':3}
   return self.reply(state)
  if route.path=='/api/voice/status':return self.reply({'robot_wakeup_source':'hardware','wakeup_available':True,'vad_enabled':True,'wakeup_word':'小绿小绿'})
  if route.path=='/api/robot/voice/wakeup':return self.reply({'source':'hardware','connected':True,'keyword':'小绿小绿','gateway_phase':'armed'})
  if route.path=='/api/display/emotion-rules.js':return self.reply('window.DisplayEmotionRules = '+json.dumps(RULES)+';','text/javascript')
  assets={
   '/api/display/client.js':'display_sync.js','/api/display/emotion-client.js':'display_emotion.js',
   '/api/robot/motion/presets-client.js':'motion-presets.js',
   '/api/robot/motion/presets-style.css':'motion-presets.css',
   '/api/robot/arm-power.js':'arm-power.js','/api/robot/all-power.js':'all-power.js','/api/robot/hardware-wake.js':'hardware-wake.js',
   '/api/robot/motion-client.js':'motion-client.js','/api/voice/pcm-player.js':'pcm-player.js',
  }
  if route.path in assets:
   file=ROOT/'src/rag_system/web/assets'/assets[route.path]
   if file.exists():return self.reply(file.read_text(),'text/css' if file.suffix=='.css' else 'text/javascript')
  if route.path.endswith('.js'):return self.reply('','text/javascript')
  return self.reply({})
 def do_POST(self):
  global state
  body=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
  if self.path.startswith('/api/display/'):return self.reply({'session_status':'active'})
  posts.append({'path':self.path,'body':body})
  assert self.path=='/api/robot/motion/presets/fly_kiss/run',posts
  assert body['catalog_revision']==catalog['revision']
  state={**state,'status':'accepted','state':'running','at_preset':False,'purpose':'preset:fly_kiss','request_id':'ui-test-token','step_index':1,'step_count':3}
  self.reply(state)
server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
Thread(target=server.serve_forever,daemon=True).start()
out=ROOT/'data/runtime/motion-ui-verification';out.mkdir(parents=True,exist_ok=True)
try:
 for kind,query in [('preview',''),('interaction','?test=1')]:
  with tempfile.TemporaryDirectory(prefix='greenriver-motion-ui-') as profile:
   result=subprocess.run(['/usr/bin/google-chrome','--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking',f'--user-data-dir={profile}','--window-size=1440,1200','--virtual-time-budget=7000',f'--screenshot={out/(kind+".png")}','--dump-dom',f'http://127.0.0.1:{server.server_port}/{query}#motions'],capture_output=True,text=True,timeout=45)
   if result.returncode:raise RuntimeError(result.stderr[-2000:])
   (out/(kind+'.html')).write_text(result.stdout)
   assert 'data-script-error=' not in result.stdout,result.stdout[-3000:]
   assert 'data-motion-visible="yes"' in result.stdout
   assert 'class="app-page active" id="page-motions"' in result.stdout
   assert result.stdout.count('class="motion-preset-card"')==len(catalog['presets'])
   assert '速度倍率 0.5 → 0.6 → 0.5' in result.stdout
   if kind=='preview':assert not posts
   else:
    assert 'data-motion-test="passed"' in result.stdout
    assert len(posts)==1,posts
 print(json.dumps({'rendered_presets':len(catalog['presets']),'initial_gets_never_move':True,'double_click_dispatched_once':True,'completion_requires_matching_state':True,'posts':posts,'screenshots':str(out)},ensure_ascii=False))
finally:server.shutdown()
