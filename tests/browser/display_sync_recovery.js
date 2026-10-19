// playwright-cli -s=display-sync run-code --filename=tests/browser/display_sync_recovery.js
async page => {
  const root = 'http://127.0.0.1:8000';
  const results = [];
  let step = 'initial manifest retry';
  const display = await page.context().newPage();
  const waitState = async (target, expected, timeout = 8000) => {
    for (let attempt = 0; attempt < timeout / 100; attempt++) {
      const response = await target.request.get(root + '/api/display/state');
      if ((await response.json()).state === expected) return;
      await target.waitForTimeout(100);
    }
    throw new Error('Display did not reach ' + expected);
  };
  try {
    await page.goto(root);
    await waitState(page, 'idle');
    let manifestAttempts = 0;
    await display.route('**/api/display/emotions/manifest.json', route => {
      if (++manifestAttempts === 1) return route.fulfill({ status: 503, body: 'retry' });
      return route.continue();
    });
    await display.goto(root + '/display');
    await display.waitForFunction(() => Boolean(currentFrame) && activeState === 'idle');
    results.push('display recovers after initial manifest download failure');
    step = 'typed streaming reply';

    await page.evaluate(() => {
      window.displayTestTrace = [];
      const originalSetDisplayEmotion = setDisplayEmotion;
      setDisplayEmotion = (...args) => {
        window.displayTestTrace.push({ at: Date.now(), args, answering: isAnswering });
        return originalSetDisplayEmotion(...args);
      };
      autoSpeakEnabled = false;
      window.displayTestOriginalFetch = window.fetch;
      window.fetch = async (url, options) => {
        if (url !== '/api/answer/stream') return window.displayTestOriginalFetch(url, options);
        const encoder = new TextEncoder();
        return new Response(new ReadableStream({ start(controller) {
          setTimeout(() => controller.enqueue(encoder.encode(JSON.stringify({ type: 'delta', text: '同步测试回答。' }) + '\n')), 700);
          setTimeout(() => {
            controller.enqueue(encoder.encode(JSON.stringify({ type: 'final', result: { answer: '同步测试回答。' } }) + '\n'));
            controller.close();
          }, 1400);
        }}), { headers: { 'Content-Type': 'application/x-ndjson' } });
      };
    });
    await page.locator('#chatInput').fill('同步测试');
    await page.locator('#chatSendBtn').click();
    await waitState(page, 'thinking');
    await waitState(page, 'speaking');
    await waitState(page, 'success');
    await waitState(page, 'idle');
    await page.evaluate(() => { window.fetch = window.displayTestOriginalFetch; });
    results.push('typed streaming reply: thinking -> speaking -> success -> idle');
    step = 'display reconnect';

    await page.evaluate(() => setDisplayEmotion('thinking', 0, true));
    await waitState(page, 'thinking');
    await display.waitForFunction(() => activeState === 'thinking');
    await display.route('**/api/display/state', route => route.abort());
    await display.waitForFunction(() => activeState === 'idle', null, { timeout: 7000 });
    await display.unroute('**/api/display/state');
    step = 'display state restored after offline';
    await display.waitForFunction(() => activeState === 'thinking', null, { timeout: 7000 });
    results.push('display network loss -> idle; reconnect -> current conversation state');
    step = 'browser lease expiry';

    await page.waitForFunction(() => !displaySync.inFlight);
    await page.evaluate(() => { clearInterval(displaySync.timer); displaySync.running = false; });
    await waitState(display, 'idle', 13000);
    await display.waitForFunction(() => activeState === 'idle');
    results.push('browser disappears without release: lease expires to idle');
    step = 'page release';

    await page.evaluate(() => { displaySync.state = 'idle'; displaySync.start(); setDisplayEmotion('listening', 0, true); });
    await waitState(page, 'listening');
    await page.goto('about:blank');
    await waitState(display, 'idle');
    results.push('leaving the admin page immediately releases its expression');
    return { passed: results };
  } catch (error) {
    const debug = {
      server: await (await page.request.get(root + '/api/display/state')).json(),
      browser: await page.evaluate(() => ({ desired: displaySync.state, state: displayEmotionState,
        transient: displayTransient, sequence: displaySync.sequence, sessionId: displaySync.sessionId,
        trace: window.displayTestTrace })),
      display: await display.evaluate(() => ({ activeState, lastServerVersion, snapshot, polling }))
    };
    throw new Error(step + ': ' + error.message + '; passed=' + JSON.stringify(results) + '; debug=' + JSON.stringify(debug));
  } finally {
    await display.close();
    await page.goto(root);
  }
}
