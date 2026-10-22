(() => {
  const root = document.getElementById('motionDebugRoot');
  if (!root) return;
  const list = document.getElementById('motionPresetList');
  const status = document.getElementById('motionDebugState');
  const detail = document.getElementById('motionDebugDetail');
  const dot = document.getElementById('motionStateDot');
  const feedback = document.getElementById('motionDebugFeedback');
  const home = document.getElementById('motionDebugHome');
  const reload = document.getElementById('motionCatalogRefresh');
  const phases = {ready:'已上电 · 待机', powered_off:'双臂已断电', running:'动作执行中', finishing:'动作收尾归位中', homing:'正在归位',
    initializing:'动作服务初始化中', powering_on:'正在上电并归位', powering_off:'正在归位并断电', fault:'动作服务故障'};
  let catalog = [], catalogRevision = '', state = null, pending = null, sending = false, reading = false, revision = 0;
  const cards = new Map();
  const id = () => globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  function say(text, tone = '') { feedback.textContent = text; feedback.dataset.tone = tone; }
  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function available() {
    return !sending && !pending && state?.state === 'ready' && state?.at_preset === true
      && state?.power?.left === 0 && state?.power?.right === 0 && state.power_feedback_age_ms <= 2000;
  }
  function render() {
    status.textContent = state ? phases[state.state] || '动作状态未知' : '动作服务未连接';
    dot.dataset.state = state?.state || 'unavailable';
    home.disabled = sending || !!pending || state?.state !== 'ready';
    if (!state) detail.textContent = '无法读取机械臂状态，请检查动作服务。测试按钮保持禁用。';
    else if (state.state === 'fault') detail.textContent = state.detail || '请检查机器人后恢复动作服务。';
    else if (state.state === 'powered_off') detail.textContent = '请先到对话页点击“上电并归位”，再返回测试。';
    else if (state.state === 'ready' && !state.at_preset) detail.textContent = '请先点击“回到起始位”，归位确认后才能测试。';
    else if (state.state === 'running' && state.purpose?.startsWith('preset:')) {
      const preset = catalog.find(item => item.id === state.purpose.slice(7));
      detail.textContent = `${preset?.name || '预设动作'} · 步骤 ${state.step_index || 0} / ${state.step_count || 0} · 请等待完成`;
    } else detail.textContent = '请保持双臂活动区域畅通；每个动作只执行一次，完成后回到起始位。';
    for (const [presetId, card] of cards) {
      const running = state?.state === 'running' && state?.purpose === `preset:${presetId}`;
      card.button.disabled = !available();
      card.button.textContent = running ? '正在执行…' : '测试一次';
      card.progress.hidden = !running;
      card.progress.max = state?.step_count || 1;
      card.progress.value = state?.step_index || 0;
    }
  }
  function observe() {
    if (!pending || !state) return;
    if (!['ready','powered_off','fault'].includes(state.state)) return;
    if (state.request_id !== pending.token) {
      say('动作状态已变化，请核对当前姿态；系统不会自动重试。', 'error');
    } else if (state.state === 'fault') {
      say(`动作未完成：${state.detail || '请检查机器人'}。后续步骤已停止。`, 'error');
    } else if (state.state === 'ready' && state.at_preset) {
      say(`${pending.name}已完成，双臂已回到起始位。`, 'success');
    } else say('测试结果未确认，请核对机械臂状态。', 'error');
    pending = null;
  }
  async function refresh() {
    if (reading || sending) return;
    reading = true;
    const version = revision;
    try {
      const response = await fetch('/api/robot/motion/state', {cache:'no-store', signal:AbortSignal.timeout(2000)});
      if (!response.ok) throw new Error('unavailable');
      const result = await response.json();
      if (version !== revision) return;
      state = result; observe();
    } catch (_) { if (version === revision) state = null; }
    finally { reading = false; render(); }
  }
  async function submit(preset) {
    if (sending || pending || (preset ? !available() : home.disabled)) return;
    sending = true; revision += 1; render();
    const name = preset ? preset.name : '归位';
    say(`正在提交${name}…`);
    try {
      const url = preset ? `/api/robot/motion/presets/${encodeURIComponent(preset.id)}/run` : '/api/robot/motion/home';
      const response = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({request_id:id(), ...(preset ? {catalog_revision:catalogRevision} : {})}), signal:AbortSignal.timeout(4000)});
      const result = await response.json();
      if (!response.ok) {
        const messages = {not_at_home:'请先回到起始位。', busy:'机械臂正在执行动作，请等待完成。',
          unavailable:'动作服务不可用或双臂尚未上电。', invalid_preset:'预设配置无效，未执行动作。'};
        say(messages[result.status] || (typeof result.detail === 'string' ? result.detail : '动作未被接受，请检查状态。'), 'error');
      } else {
        state = result;
        pending = {name, token:result.request_id};
        say(`${name}已接收，正在等待执行结果…`);
      }
    } catch (_) {
      say('未确认动作回执。请先核对当前状态，系统不会自动重试。', 'error');
      state = null;
    } finally { sending = false; render(); await refresh(); }
  }
  function renderCatalog() {
    cards.clear(); list.replaceChildren();
    const ordered = [...catalog].sort((a,b) => Number(b.type === 'sequence') - Number(a.type === 'sequence'));
    for (const preset of ordered) {
      const card = element('article', undefined, 'motion-preset-card');
      card.dataset.kind = preset.type; card.dataset.presetId = preset.id;
      const top = element('div', undefined, 'motion-card-top');
      top.append(element('h2', preset.name), element('span', preset.type === 'sequence' ? '自定义 · 待实测' : 'SDK 预设', 'motion-kind'));
      card.append(top, element('p', preset.description || '测试完成后回到起始位。', 'motion-description'));
      if (preset.type === 'sequence') {
        card.append(element('div', `速度倍率 ${preset.frames.map(frame => String(frame.scale)).join(' → ')}`, 'motion-speeds'));
        const details = element('details'); details.append(element('summary', '查看关键帧与关节角'));
        preset.frames.forEach((frame,index) => {
          const row = element('div', undefined, 'motion-frame');
          row.append(element('strong', `${index + 1}. ${frame.name || (frame.home ? '回到起始位' : '关键帧')} · 速度 ${frame.scale}`));
          if (frame.home) row.append(element('p', '使用 SDK 当前归位姿态。', 'motion-joints'));
          else for (const [key,label] of [['left_arm','左臂'],['right_arm','右臂']])
            row.append(element('p', `${label}（°）：${frame[key].map(v => v.toFixed(2)).join('，')}`, 'motion-joints'));
          if (frame.wait_ms) row.append(element('p', `到位后停留 ${frame.wait_ms} 毫秒`));
          details.append(row);
        });
        card.append(details);
      } else card.append(element('div', '保留 SDK 已配置的速度倍率', 'motion-speeds'));
      const progress = element('progress'); progress.hidden = true;
      progress.setAttribute('aria-label', `${preset.name}执行进度`);
      const button = element('button', '测试一次'); button.disabled = true;
      button.addEventListener('click', () => submit(preset));
      card.append(progress,button); list.append(card); cards.set(preset.id,{button,progress});
    }
    render();
  }
  async function loadCatalog() {
    reload.disabled = true;
    try {
      const response = await fetch('/api/robot/motion/presets', {cache:'no-store',signal:AbortSignal.timeout(3000)});
      if (!response.ok) throw new Error('catalog');
      const result = await response.json(); catalog = result.presets; catalogRevision = result.revision; renderCatalog();
      if (!pending) say(`已载入 ${catalog.length} 个预设。选择动作后点击“测试一次”。`);
    } catch (_) {
      catalog = []; renderCatalog(); say('动作目录读取失败，请检查配置后刷新。', 'error');
    } finally { reload.disabled = false; }
  }
  home.addEventListener('click', () => submit(null));
  reload.addEventListener('click', loadCatalog);
  void loadCatalog(); void refresh();
  setInterval(() => { if (!document.hidden) void refresh(); }, 1000);
})();
