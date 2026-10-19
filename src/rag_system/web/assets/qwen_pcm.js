/* One continuous PCM timeline. Completion means physical buffers have ended. */
(function (root) {
  class QwenPcmPlayer {
    constructor(context) {
      this.context = context;
      this.sources = new Set();
      this.nextTime = 0;
      this.stopped = false;
      this.waiters = [];
      this.frames = 0;
    }
    push(base64, sampleRate = 24000) {
      if (this.stopped) return;
      if (sampleRate !== 24000) throw new Error('不支持的回复采样率');
      const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
      if (!bytes.length || bytes.length % 2) throw new Error('回复 PCM 数据无效');
      if (this.context.state !== 'running') throw new Error('请先点击页面启用音频播放');
      const view = new DataView(bytes.buffer);
      const buffer = this.context.createBuffer(1, bytes.length / 2, sampleRate);
      const channel = buffer.getChannelData(0);
      for (let i = 0; i < channel.length; i++) channel[i] = view.getInt16(i * 2, true) / 32768;
      const source = this.context.createBufferSource();
      source.buffer = buffer;
      source.connect(this.context.destination);
      this.sources.add(source);
      source.onended = () => {
        source.disconnect();
        this.sources.delete(source);
        this.settle();
      };
      this.nextTime = Math.max(this.nextTime, this.context.currentTime + 0.03);
      // Bound queued audio; stop the turn instead of accumulating unbounded output.
      if (this.nextTime - this.context.currentTime > 120) {
        this.sources.delete(source);
        source.disconnect();
        throw new Error('回复音频超出播放缓冲上限');
      }
      source.start(this.nextTime);
      this.nextTime += buffer.duration;
      this.frames += channel.length;
    }
    settle() {
      if (!this.sources.size) this.waiters.splice(0).forEach(resolve => resolve());
    }
    drain() {
      if (!this.sources.size) return Promise.resolve();
      return new Promise(resolve => this.waiters.push(resolve));
    }
    stop() {
      this.stopped = true;
      for (const source of this.sources) {
        source.onended = null;
        try { source.stop(); } catch (_) {}
        source.disconnect();
      }
      this.sources.clear();
      this.settle();
    }
  }
  root.QwenPcmPlayer = QwenPcmPlayer;
})(globalThis);
