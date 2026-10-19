import shutil
import subprocess
from pathlib import Path

import pytest


def test_pcm_player_drains_all_buffers_and_drops_stale_audio():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for browser audio contract')
    source = (Path(__file__).parents[1] / 'src/rag_system/web/assets/qwen_pcm.js').read_text(encoding='utf-8')
    probe = r'''
const assert = require('node:assert/strict');
(async () => {
  const scheduled = [];
  const context = { state: 'running', currentTime: 1, destination: {},
    createBuffer: (c,n,r) => ({duration:n/r,getChannelData:()=>new Float32Array(n)}),
    createBufferSource: () => { const s = {connect(){},disconnect(){},start(t){s.when=t;scheduled.push(s)},stop(){s.stopped=true}}; return s; }
  };
  const player = new QwenPcmPlayer(context);
  player.push('AAA='); player.push('AAA=');
  assert(scheduled[1].when > scheduled[0].when);
  let drained = false;
  player.drain().then(()=>drained=true);
  scheduled[0].onended(); await Promise.resolve(); assert.equal(drained,false);
  scheduled[1].onended(); await Promise.resolve(); assert.equal(drained,true);
  player.push('AAA=');
  const before = player.frames;
  player.stop(); player.push('AAA=');
  assert.equal(player.frames,before);
  assert.equal(scheduled[2].stopped,true);
  await player.drain(); assert.equal(player.sources.size,0);
})().catch(error=>{console.error(error);process.exitCode=1});
'''
    result = subprocess.run([node], input=source+probe, encoding='utf-8', capture_output=True)
    assert result.returncode == 0, result.stderr
