// A top-level browser reload requests one return to the vendor preset pose.
// Initial process startup homes in the native service; iframe navigation and
// bfcache restoration do not create additional physical movements.
(() => {
  if (window.self !== window.top) return;
  const navigation = performance.getEntriesByType('navigation')[0];
  if (navigation?.type !== 'reload') return;
  const requestId = globalThis.crypto?.randomUUID?.() ||
    `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  fetch('/api/robot/motion/home', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request_id: requestId }),
    signal: AbortSignal.timeout(1500),
  }).then(response => {
    if (!response.ok) console.debug('刷新归位请求暂未被接受');
  }).catch(() => console.debug('机械臂动作服务暂不可用'));
})();
