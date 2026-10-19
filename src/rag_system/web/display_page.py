"""The kiosk display shell shown on the robot's dedicated screen."""
from __future__ import annotations

DISPLAY_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="data:,">
  <title>小绿显示屏</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #000; }
    body { font-family: system-ui, sans-serif; }
    #emotion, #web { position: fixed; inset: 0; width: 100%; height: 100%; border: 0; }
    #emotion { display: block; background: #000; }
    #eyes { display: block; width: 100%; height: 100%; image-rendering: pixelated; }
    #web { display: none; background: #101214; }
    #caption { position: fixed; left: 50%; bottom: 5vh; transform: translateX(-50%); max-width: 80vw;
      color: #d7f9df; font-size: clamp(16px, 2vw, 30px); text-align: center; opacity: 0; transition: opacity .25s; }
    #caption.visible { opacity: .9; }
  </style>
</head>
<body>
  <main id="emotion"><canvas id="eyes" role="img" aria-label="小绿表情"></canvas><div id="caption"></div></main>
  <iframe id="web" title="小绿管理网页"></iframe>
  <script>
    const eyes = document.getElementById('eyes');
    const emotionPanel = document.getElementById('emotion');
    const webPanel = document.getElementById('web');
    const caption = document.getElementById('caption');
    let manifest = null, lastVersion = null, lastServerVersion = null, timer = null, frameIndex = 0, activeState = '';
    let snapshot = null, lastCaptionVersion = null;
    let activeFrames = null, currentFrame = null, animationToken = 0;
    let polling = false, pollTimer = null, lastContact = Date.now();
    const stateCache = new Map();

    function fitCanvas() {
      const ratio = window.devicePixelRatio || 1;
      eyes.width = Math.round(window.innerWidth * ratio);
      eyes.height = Math.round(window.innerHeight * ratio);
      if (currentFrame && manifest) drawFrame(currentFrame);
    }

    async function loadState(state) {
      if (!stateCache.has(state)) {
        const config = manifest.states[state];
        const pending = Promise.all(config.frames.map(file => new Promise((resolve, reject) => {
          const frame = new Image();
          const timeout = setTimeout(() => reject(new Error(`表情帧加载超时：${file}`)), 5000);
          frame.onload = () => { clearTimeout(timeout); resolve(frame); };
          frame.onerror = () => { clearTimeout(timeout); reject(new Error(`表情帧加载失败：${file}`)); };
          frame.src = `/api/display/emotions/${encodeURIComponent(file)}`;
        })));
        stateCache.set(state, pending.catch(error => {
          stateCache.delete(state);
          throw error;
        }));
      }
      return stateCache.get(state);
    }

    function displayScale(width, height) {
      const layout = manifest.layout;
      const padding = Math.min(width, height) * layout.edge_padding_fraction;
      let scale = Infinity;
      for (let side = 0; side < 2; side++) {
        const [anchorX, anchorY] = layout.eye_anchors[side];
        const [left, top, right, bottom] = layout.content_bounds[side];
        const targetX = width * layout.target_x_fraction[side];
        const targetY = height * layout.target_y_fraction;
        scale = Math.min(
          scale,
          (targetX - padding) / (anchorX - left),
          (width - padding - targetX) / (right + 1 - anchorX),
          (targetY - padding) / (anchorY - top),
          (height - padding - targetY) / (bottom + 1 - anchorY)
        );
      }
      return scale;
    }

    function drawFrame(frame) {
      const context = eyes.getContext('2d');
      const width = eyes.width;
      const height = eyes.height;
      context.fillStyle = '#000';
      context.fillRect(0, 0, width, height);
      context.imageSmoothingEnabled = false;
      const sourceHalfWidth = manifest.canvas.width / 2;
      const sourceHeight = manifest.canvas.height;
      const scale = displayScale(width, height);
      for (let side = 0; side < 2; side++) {
        const sourceX = side * sourceHalfWidth;
        const stateAnchors = manifest.states[activeState]?.eye_anchors || manifest.layout.eye_anchors;
        const [anchorX, anchorY] = stateAnchors[side];
        const targetX = width * manifest.layout.target_x_fraction[side];
        const targetY = height * manifest.layout.target_y_fraction;
        const x = targetX - (anchorX - sourceX) * scale;
        const y = targetY - anchorY * scale;
        context.drawImage(
          frame, sourceX, 0, sourceHalfWidth, sourceHeight,
          x, y, sourceHalfWidth * scale, sourceHeight * scale
        );
      }
    }

    async function loadManifest() {
      const response = await fetch('/api/display/emotions/manifest.json', {
        cache: 'no-store', signal: AbortSignal.timeout(3000)
      });
      if (!response.ok) throw new Error('emotion manifest unavailable');
      manifest = await response.json();
    }

    function showMode(mode) {
      const web = mode === 'web';
      emotionPanel.style.display = web ? 'none' : 'block';
      webPanel.style.display = web ? 'block' : 'none';
      if (web) { clearInterval(timer); animationToken++; activeState = ''; }
      if (web && !webPanel.src) webPanel.src = '/';
    }

    async function playState(state, version) {
      if (!manifest || !manifest.states[state]) state = 'idle';
      if (activeState === state && lastVersion === version) return;
      activeState = state; lastVersion = version; frameIndex = 0;
      eyes.dataset.state = state;
      const config = manifest.states[state];
      clearInterval(timer);
      const token = ++animationToken;
      let loaded;
      try { loaded = await loadState(state); }
      catch (error) {
        if (token !== animationToken) return;
        activeState = ''; lastVersion = null; lastServerVersion = null;
        caption.textContent = error.message; caption.classList.add('visible'); return;
      }
      if (token !== animationToken) return;
      activeFrames = loaded;
      const render = () => {
        currentFrame = activeFrames[Math.min(frameIndex, activeFrames.length - 1)];
        drawFrame(currentFrame);
        frameIndex += 1;
        if (frameIndex >= config.frames.length) {
          if (config.loop) frameIndex = 0;
          else {
            clearInterval(timer); timer = null;
            if (config.fallback_state) setTimeout(() => {
              if (token === animationToken) playState(config.fallback_state, version + ':fallback');
            }, config.frame_duration_ms);
          }
        }
      };
      render();
      if (config.loop || config.frames.length > 1) timer = setInterval(render, config.frame_duration_ms);
    }

    function applyState(next) {
      const previousMode = snapshot && snapshot.mode;
      snapshot = next;
      showMode(next.mode);
      if (next.mode !== 'web') {
        const serverVersion = String(next.updated_at || 0);
        if (serverVersion !== lastServerVersion || previousMode !== next.mode) {
          lastServerVersion = serverVersion;
          playState(next.state || 'idle', serverVersion);
        }
        if (lastCaptionVersion !== serverVersion) {
          lastCaptionVersion = serverVersion;
          caption.textContent = next.text || '';
          caption.classList.toggle('visible', Boolean(next.text));
          if (next.text) setTimeout(() => {
            if (lastCaptionVersion === serverVersion) caption.classList.remove('visible');
          }, 5000);
        }
      }
    }

    async function poll() {
      if (polling) return;
      polling = true;
      clearTimeout(pollTimer);
      try {
        if (!manifest) { await loadManifest(); await loadState('idle'); }
        const response = await fetch('/api/display/state', {
          cache: 'no-store', signal: AbortSignal.timeout(3000)
        });
        if (!response.ok) throw new Error('display state unavailable');
        applyState(await response.json());
        lastContact = Date.now();
      } catch (_) {
        if (manifest && Date.now() - lastContact > 3000) {
          applyState({ mode: snapshot?.mode || 'emotion', state: 'idle', updated_at: 'offline', text: '' });
        }
      } finally {
        polling = false;
        pollTimer = setTimeout(poll, 500);
      }
    }

    window.addEventListener('resize', fitCanvas);
    fitCanvas();
    window.addEventListener('online', () => void poll());
    void poll();
  </script>
</body>
</html>'''
