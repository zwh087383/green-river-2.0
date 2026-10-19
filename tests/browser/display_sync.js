// Run against the local server with:
// playwright-cli -s=display-sync run-code --filename=tests/browser/display_sync.js
async page => {
  const root = 'http://127.0.0.1:8000';
  const results = [];
  const waitState = async expected => {
    for (let attempt = 0; attempt < 80; attempt++) {
      const response = await page.request.get(root + '/api/display/state');
      if ((await response.json()).state === expected) return;
      await page.waitForTimeout(100);
    }
    throw new Error('Display did not reach ' + expected);
  };
  const state = async () => (await (await page.request.get(root + '/api/display/state')).json());
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  const display = await page.context().newPage();
  try {
    await page.goto(root);
    await waitState('idle');
    await display.setViewportSize({ width: 800, height: 480 });
    await display.goto(root + '/display');
    for (const emotion of ['thinking', 'speaking', 'listening']) {
      await page.evaluate(emotion => setDisplayEmotion(emotion, 0, true), emotion);
      await waitState(emotion);
      await display.waitForFunction(value => document.querySelector('#eyes').dataset.state === value, emotion);
      await page.reload();
      await waitState('idle');
      await display.waitForFunction(() => document.querySelector('#eyes').dataset.state === 'idle');
      results.push('refresh from ' + emotion + ' -> idle');
    }

    await page.evaluate(() => setDisplayEmotion('thinking', 0, true));
    await waitState('thinking');
    const passive = await page.context().newPage();
    await passive.goto(root);
    await passive.waitForFunction(() => displaySync.registered && !displaySync.inFlight);
    check((await state()).state === 'thinking', 'Passive tab overwrote the active conversation');
    await passive.close();
    check((await state()).state === 'thinking', 'Closing passive tab reset the active conversation');
    results.push('passive tab leaves active conversation unchanged');

    const audioResponse = await page.request.get(root + '/api/voice/wakeup-acknowledgement');
    check(audioResponse.ok(), 'Missing audio fixture');
    const audio = await audioResponse.body();
    await page.route('**/api/voice/tts/stream', async route => {
      await page.waitForTimeout(700);
      try {
        await route.fulfill({ status: 200, contentType: 'audio/mpeg', body: audio });
      } catch (error) {
        // The interruption test deliberately aborts an in-flight audio request.
        if (!/already handled|has been closed|Invalid InterceptionId/i.test(String(error))) throw error;
      }
    });
    await page.evaluate(() => playTTS('表情同步回归测试。'));
    await waitState('thinking');
    check(!await page.evaluate(() => Boolean(displayPlayingAudio)), 'Speaking started before audio playback');
    await waitState('speaking');
    await page.evaluate(() => stopCurrentTTS());
    await waitState('idle');
    results.push('TTS loading -> thinking; actual playback -> speaking; stop -> idle');

    await page.evaluate(() => { playTTS('第一段。'); stopCurrentTTS(); playTTS('第二段。'); });
    await waitState('speaking');
    check(await page.evaluate(() => Boolean(ttsQueueRunPromise && ttsActiveItem)), 'Old queue cleared new playback');
    await waitState('success');
    await waitState('idle');
    results.push('interrupted old TTS cannot clear new playback; completion returns to idle');

    await page.unroute('**/api/voice/tts/stream');
    await page.route('**/api/voice/tts/stream', route => route.fulfill({
      status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'Regression test failure' })
    }));
    await page.evaluate(() => playTTS('合成失败测试。'));
    await waitState('error');
    await waitState('idle');
    await page.unroute('**/api/voice/tts/stream');
    results.push('TTS failure -> error -> idle');

    await display.waitForFunction(() => document.querySelector('#eyes').dataset.state === 'idle');
    await display.screenshot({ path: 'data/runtime/display-sync-refresh.png' });
    return { passed: results };
  } finally {
    await page.unroute('**/api/voice/tts/stream');
    await page.evaluate(() => { stopCurrentTTS(); setDisplayEmotion('idle'); }).catch(() => {});
    await display.close();
  }
}
