"""Local animation gallery; viewing it never takes over the robot display."""
from __future__ import annotations


DISPLAY_PREVIEW_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="data:,"><title>小绿 · 动态表情</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #101613; color: #edf4ef; }
    main { max-width: 1240px; margin: auto; padding: 30px; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-bottom: 24px; }
    h1 { font-size: 26px; margin: 0 0 9px; font-weight: 650; }
    p { margin: 0; color: #9fb2a5; font-size: 14px; line-height: 1.6; }
    a { color: #b5edc8; text-decoration: none; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    button { color: #c3f2d3; border: 1px solid #365241; background: #17271d;
      padding: 9px 13px; border-radius: 8px; font: inherit; font-size: 13px; cursor: pointer; }
    button:hover { background: #244130; }
    button:disabled { opacity: .5; cursor: wait; }
    #gallery { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
    article { overflow: hidden; border: 1px solid #29392e; border-radius: 12px; background: #141f18; }
    canvas { display: block; width: 100%; aspect-ratio: 2 / 1; background: #000; image-rendering: pixelated; }
    .meta { padding: 13px 15px; display: flex; justify-content: space-between; align-items: center; gap: 10px; }
    h2 { margin: 0 0 4px; font-size: 16px; font-weight: 600; }
    .detail { font-size: 12px; color: #9fb2a5; }
    #status { min-height: 24px; margin-top: 18px; color: #b5edc8; }
    @media (max-width: 900px) { #gallery { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 560px) { main { padding: 18px; } header { align-items: flex-start; flex-direction: column; }
      #gallery { grid-template-columns: 1fr; } }
  </style>
</head>
<body><main>
  <header>
    <div><h1>小绿 · 动态表情</h1><p>黑底白眼，像素动画。开心腮红沿用参考图，其余表情按讲解场景补充。</p></div>
    <div class="actions"><a href="/">返回对话</a><button id="reset">恢复待机</button></div>
  </header>
  <section id="gallery" aria-label="动态表情预览"></section>
  <p id="status" role="status">这里的动画仅供预览；点击“投到屏幕”才会切换外接屏。开始新一轮对话后自动接管。</p>
</main>
<script src="/api/robot/motion-client.js"></script>
<script src="/api/display/tone-renderer.js"></script>
<script>
  const renderer = new DisplayToneRenderer();
  const gallery = document.getElementById('gallery');
  const status = document.getElementById('status');
  const cards = [];
  const startedAt = performance.now();
  for (const [tone, config] of Object.entries(DisplayToneRenderer.tones)) {
    const card = document.createElement('article');
    const canvas = document.createElement('canvas');
    canvas.width = 384; canvas.height = 192;
    canvas.dataset.tone = tone;
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', config.label + '动态表情');
    const meta = document.createElement('div'); meta.className = 'meta';
    const text = document.createElement('div');
    const title = document.createElement('h2'); title.textContent = config.label;
    const detail = document.createElement('span'); detail.className = 'detail'; detail.textContent = config.detail;
    text.append(title, detail);
    const button = document.createElement('button'); button.textContent = '投到屏幕';
    button.setAttribute('aria-label', '预览' + config.label + '到外接屏');
    button.addEventListener('click', () => showOnScreen(tone, config.label, button));
    meta.append(text, button); card.append(canvas, meta); gallery.append(card);
    cards.push({ tone, canvas, context: canvas.getContext('2d') });
  }
  function drawGallery(frameIndex) {
    for (const { tone, canvas, context } of cards) {
      context.imageSmoothingEnabled = false;
      context.drawImage(renderer.render(tone, frameIndex), 0, 0, canvas.width, canvas.height);
      canvas.dataset.frame = String(frameIndex % 40);
    }
  }
  drawGallery(0);
  const galleryTimer = setInterval(() => drawGallery(Math.floor((performance.now() - startedAt) / 100)), 100);

  async function post(path, payload) {
    const response = await fetch(path, { method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload), signal: AbortSignal.timeout(4000) });
    if (!response.ok) throw new Error('显示屏暂不可用，请稍后重试');
  }
  let switching = false;
  async function showOnScreen(tone, label, button) {
    if (switching) return;
    switching = true; button.disabled = true;
    try {
      await post('/api/display/mode', {mode: 'emotion', reason: 'animation preview'});
      await post('/api/display/emotion', {state: 'speaking', tone, reason: 'animation preview'});
      status.textContent = '外接屏正在预览“' + label + '”；点击“恢复待机”结束，或开始新一轮对话。';
    } catch (error) { status.textContent = error.message; }
    finally { switching = false; button.disabled = false; }
  }
  document.getElementById('reset').addEventListener('click', async event => {
    if (switching) return;
    switching = true; event.currentTarget.disabled = true;
    try {
      await post('/api/display/emotion', {state: 'idle', tone: 'neutral', reason: 'preview ended'});
      status.textContent = '外接屏已恢复待机，下一轮对话会自动切换表情。';
    } catch (error) { status.textContent = error.message; }
    finally { switching = false; document.getElementById('reset').disabled = false; }
  });
</script></body></html>'''
