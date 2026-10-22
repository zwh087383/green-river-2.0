// Read-only polling; power changes happen only after an explicit button click.
(() => {
  const on = document.getElementById('armPowerOnBtn');
  const off = document.getElementById('armPowerOffBtn');
  const restart = document.getElementById('armRestartBtn');
  const status = document.getElementById('armPowerStatus');
  const message = document.getElementById('armPowerMessage');
  if (!on || !off || !status || !message) return;
  let state = null;
  let sending = false;
  let reading = false;
  let operation = null;
  let revision = 0;
  const labels = { 0: '已上电', 1: '已断电', 2: '急停', 3: '安全门打开', '-1': '未知' };
  const phases = {
    initializing: '初始化中', ready: '待机', running: '动作执行中', finishing: '动作收尾归位中', homing: '归位中',
    powering_on: '上电并归位中', powering_off: '归位并断电中',
    powering_on_all: '全部上电归位中', powering_off_all: '全部归位断电中',
    restarting_arms: '手臂重启中',
    powered_off: '断电操作已完成', fault: '故障锁定',
  };
  function render() {
    on.disabled = sending || operation !== null || state?.state !== 'powered_off';
    off.disabled = sending || operation !== null || state?.state !== 'ready';
    if (restart) restart.disabled = sending || operation !== null
      || state?.restart_pending || !['ready', 'powered_off', 'fault'].includes(state?.state)
      || (state?.state !== 'fault' && (!(state?.power_feedback_age_ms <= 2000)
        || ![state?.power?.left, state?.power?.right].every(v => v === 0 || v === 1)));
    if (!state?.power) {
      on.disabled = off.disabled = true;
      if (restart) restart.disabled = true;
      status.textContent = '状态不可用';
      return;
    }
    const stale = state.power_feedback_age_ms > 2000;
    const left = stale ? '反馈过期' : labels[state.power.left] || '未知';
    const right = stale ? '反馈过期' : labels[state.power.right] || '未知';
    const phase = state.state === 'powered_off' && (state.power.left !== 1 || state.power.right !== 1)
      ? '断电已请求，未确认双臂断电' : phases[state.state] || '状态未知';
    status.textContent = `${phase} · 左臂${left} / 右臂${right}`;
    if (state.restart_pending) message.textContent = '重启已接收，等待旧指令退出；随后检查双臂停止状态并恢复连接。';
    else if (state.state === 'fault') message.textContent = `操作故障：${state.detail || '请检查机器人'}。可点击“重启手臂”尝试恢复。`;
  }
  async function observeOperation() {
    if (!operation) return;
    if (operation.instance) {
      const response = await fetch(`/api/robot/motion/requests/${encodeURIComponent(operation.token)}?instance_id=${encodeURIComponent(operation.instance)}`, { cache:'no-store', signal:AbortSignal.timeout(2000) });
      const result = await response.json();
      if (!response.ok || !result.result) {
        message.textContent = '操作结果未确认，请核对状态；系统不会自动重发。';
        operation = null; return;
      }
      const record = result.result;
      if (['accepted','running'].includes(record.result_state)) return;
      message.textContent = record.result_state === 'verified_completed'
        ? operation.power === 'off' ? '归位及断电反馈已确认。' : operation.power === 'restart' ? '手臂重启及归位反馈已确认。' : '上电及归位反馈已确认。'
        : `操作结果${record.result_state === 'failed' ? '失败' : '未确认'}：${record.detail || '请现场核对'}`;
      operation = null; return;
    }
    if (state?.restart_pending) return;
    const purpose = `power_${operation.power}_home`;
    const same = state?.purpose === purpose && state?.request_id === operation.token;
    if (state.state === 'ready' || state.state === 'powered_off' || state.state === 'fault') {
      if (!same) message.textContent = '操作结果未确认，当前机械臂状态已更新，请核对状态后操作。';
      else if (state.state === 'ready') message.textContent = state.at_preset && state.power?.left === 0 && state.power?.right === 0
        ? operation.power === 'restart' ? '手臂重启及归位完成，双臂上电和位置反馈已确认。' : '上电及归位完成，已核对双臂关节位置。'
        : '操作返回，但双臂上电或归位反馈未确认，请现场核对。';
      else if (state.state === 'powered_off') message.textContent = state.power?.left === 1 && state.power?.right === 1
        ? '归位完成，双臂断电反馈已确认。' : '归位完成，SDK 断电调用已返回；当前断电反馈尚未确认。';
      operation = null;
    }
  }
  async function refresh() {
    if (reading || sending) return;
    reading = true;
    const version = revision;
    try {
      const response = await fetch('/api/robot/motion/state', { cache: 'no-store', signal: AbortSignal.timeout(2000) });
      if (!response.ok) throw new Error('unavailable');
      const result = await response.json();
      if (version !== revision) return;
      state = result;
      await observeOperation();
    } catch (_) {
      if (version === revision) state = null;
    } finally {
      reading = false;
      render();
    }
  }
  async function setPower(power) {
    if (sending || operation || (power === 'restart' ? !restart || restart.disabled : power === 'on' ? on.disabled : off.disabled)) return;
    sending = true;
    revision += 1;
    message.textContent = power === 'restart' ? '正在提交手臂重启及归位；故障时先检查停止状态再恢复连接…' : power === 'on' ? '正在提交上电并归位操作…' : '正在提交归位并断电操作…';
    render();
    const requestId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    try {
      const response = await fetch(power === 'restart' ? '/api/robot/motion/restart' : '/api/robot/motion/power', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(power === 'restart' ? { request_id: requestId } : { request_id: requestId, power }), signal: AbortSignal.timeout(2000),
      });
      const result = await response.json();
      if (!response.ok) {
        message.textContent = result.status === 'busy' ? '机械臂正在执行动作，请待机后再操作。'
          : typeof result.detail === 'string' && result.detail ? result.detail : '操作未被接受，请检查机械臂状态。';
      } else if (result.status === 'accepted' || result.status === 'duplicate') {
        state = result;
        operation = { power, token: result.result?.request_id || result.request_id, instance: result.instance_id };
        message.textContent = power === 'restart' ? '手臂重启已接收，等待双臂重新上电及归位结果。' : power === 'on' ? '操作已接收，等待上电及归位结果。' : '操作已接收，等待归位及断电结果。';
      } else {
        state = result;
        message.textContent = result.status === 'already_on' ? '机械臂已经上电。' : '机械臂已处于断电流程完成状态，请查看左右臂反馈。';
      }
    } catch (_) {
      // An ambiguous response is never automatically retried.
      message.textContent = '未收到操作回执，请先核对机械臂状态；系统不会自动重试。';
      state = null;
    } finally {
      sending = false;
      render();
      await refresh();
    }
  }
  on.addEventListener('click', () => setPower('on'));
  off.addEventListener('click', () => setPower('off'));
  restart?.addEventListener('click', () => setPower('restart'));
  void refresh();
  setInterval(() => { if (!document.hidden) void refresh(); }, 1000);
})();
