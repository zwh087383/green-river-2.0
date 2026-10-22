(() => {
  const label = document.getElementById('robotWakeStatus');
  if (!label) return;
  let reading = false;
  async function refresh() {
    if (reading) return;
    reading = true;
    try {
      const response = await fetch('/api/robot/voice/wakeup', { cache: 'no-store', signal: AbortSignal.timeout(2000) });
      if (!response.ok) throw new Error('unavailable');
      const state = await response.json();
      const phase = { armed: '待唤醒', listening: '正在聆听', answer: '正在回复',
        prompt: '正在播放提示音', connecting: '语音连接中', disconnected: '语音连接中断' }[state.gateway_phase] || '已连接';
      label.textContent = state.source === 'software' ? '机器人唤醒：软件模式'
        : state.connected ? `机器人唤醒：麦克风硬件 · ${state.keyword} · ${phase}`
        : `机器人唤醒：硬件未就绪 · ${state.error || '等待设备连接'}`;
    } catch (_) {
      label.textContent = '机器人唤醒：状态暂不可用';
    } finally { reading = false; }
  }
  void refresh();
  setInterval(() => { if (!document.hidden) void refresh(); }, 2000);
})();
