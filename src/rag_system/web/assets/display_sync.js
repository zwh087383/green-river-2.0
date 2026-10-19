/* Keep one browser conversation in sync with the robot display. */
class DisplaySessionClient {
  constructor() {
    const key = 'rag.displayClientId';
    try {
      this.clientId = sessionStorage.getItem(key) || crypto.randomUUID();
      sessionStorage.setItem(key, this.clientId);
    } catch (_) { this.clientId = crypto.randomUUID(); }
    this.state = 'idle';
    this.version = 0;
    this.claimVersion = 0;
    this.claimAcknowledged = 0;
    this.running = false;
    this.inFlight = false;
    window.addEventListener('pagehide', () => this.stop());
    window.addEventListener('pageshow', event => {
      if (event.persisted) { this.state = 'idle'; this.start(); }
    });
    window.addEventListener('online', () => void this.synchronize());
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) void this.synchronize();
    });
  }

  start() {
    if (this.running) return;
    this.sessionId = crypto.randomUUID();
    try {
      this.generation = Number(sessionStorage.getItem('rag.displayGeneration') || 0) + 1;
      sessionStorage.setItem('rag.displayGeneration', String(this.generation));
    } catch (_) { this.generation = Date.now(); }
    this.sequence = 0;
    this.registered = false;
    this.retired = false;
    this.running = true;
    this.timer = setInterval(() => void this.synchronize(), 2000);
    void this.synchronize();
  }

  setState(state, claim = false) {
    if (this.state === state && !claim) return;
    if (this.state !== state || claim) this.version++;
    this.state = state;
    if (claim) {
      this.claimVersion++;
      // Only new user activity can reactivate a page superseded by another one.
      if (this.retired) { this.stop(); this.start(); }
    }
    void this.synchronize();
  }

  async post(path, body) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 4000);
    try {
      const response = await fetch(path, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body), signal: controller.signal, cache: 'no-store'
      });
      if (!response.ok) throw new Error('Display synchronization failed');
      return await response.json();
    } finally { clearTimeout(timeout); }
  }

  async synchronize() {
    if (!this.running || this.inFlight || this.retired) return;
    this.inFlight = true;
    const sessionId = this.sessionId;
    const version = this.version;
    let retry = false;
    try {
      if (!this.registered) {
        const opened = await this.post('/api/display/session', {
          client_id: this.clientId, session_id: sessionId, generation: this.generation
        });
        if (!this.running || this.sessionId !== sessionId) return;
        if (opened.session_status === 'retired') { this.retired = true; return; }
        this.registered = true;
      }
      const claimVersion = this.claimVersion;
      const result = await this.post('/api/display/session/state', {
        session_id: sessionId, sequence: ++this.sequence, state: this.state,
        claim: claimVersion > this.claimAcknowledged
      });
      if (!this.running || this.sessionId !== sessionId) return;
      if (result.session_status === 'unknown') {
        this.registered = false; // Server restarted; reopen on the next heartbeat.
      } else if (result.session_status === 'retired') {
        this.retired = true;
      } else {
        this.claimAcknowledged = claimVersion;
      }
      retry = this.version !== version;
    } catch (_) {
      // Keep the desired state. The next heartbeat retries after a network failure.
    } finally {
      this.inFlight = false;
      if (this.running && (retry || this.sessionId !== sessionId)) void this.synchronize();
    }
  }

  stop() {
    if (!this.running) return;
    this.running = false;
    clearInterval(this.timer);
    const payload = JSON.stringify({ session_id: this.sessionId });
    const body = new Blob([payload], { type: 'application/json' });
    if (!navigator.sendBeacon?.('/api/display/session/release', body)) {
      void fetch('/api/display/session/release', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: payload, keepalive: true
      }).catch(() => {});
    }
  }
}
window.DisplaySessionClient = DisplaySessionClient;
