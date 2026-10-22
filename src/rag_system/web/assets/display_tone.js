/* Native pixel animation: 40 frames / 4 seconds, independent of PNG playback. */
class DisplayToneRenderer {
  static tones = {
    happy: { label: '开心', detail: '笑眼轻跳 · 腮红呼吸', example: '太好了，志愿者成功救助了受伤动物！' },
    concerned: { label: '关切', detail: '抬眉环顾 · 温和眨眼', example: '请小心，户外活动一定要保护好自己。' },
    sad: { label: '难过', detail: '垂眼低望 · 泪滴滑落', example: '这次损失令人痛心，我们深感遗憾。' },
    surprised: { label: '惊讶', detail: '放大回弹 · 感叹闪动', example: '真没想到，这片湿地恢复得这么快！' },
    curious: { label: '疑惑', detail: '歪眉侧看 · 问号浮动', example: '我有点疑惑，你说的是哪一条河呢？' },
    shy: { label: '害羞', detail: '低头偷看 · 腮红渐深', example: '你过奖啦，我都有点不好意思了。' },
    proud: { label: '自豪', detail: '扬眉眯眼 · 星光闪烁', example: '多年的守护成果令人自豪！' },
    encouraging: { label: '鼓励', detail: '俏皮眨眼 · 亮点跳动', example: '加油，你也可以成为守护江河的一员！' }
  };

  constructor() { this.frames = new Map(); }

  render(tone, frameIndex) {
    if (!Object.hasOwn(DisplayToneRenderer.tones, tone)) return null;
    const phase = ((Math.floor(frameIndex) % 40) + 40) % 40;
    const key = tone + ':' + phase;
    if (this.frames.has(key)) return this.frames.get(key);
    const canvas = document.createElement('canvas');
    canvas.width = 96; canvas.height = 48;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 96, 48);
    const rect = (x, y, w, h, color = '#fff') => {
      ctx.fillStyle = color; ctx.fillRect(Math.round(x), Math.round(y), w, h);
    };
    const blink = phase === 30 || phase === 31;
    const gaze = phase >= 8 && phase < 17 ? 1 : phase >= 21 && phase < 27 ? -1 : 0;
    const sparkle = (x, y, bright = true, color = '#ffe6a0') => {
      rect(x, y, 1, 5, color); rect(x - 2, y + 2, 5, 1, color);
      if (bright) rect(x - 1, y + 1, 3, 3, color);
    };
    for (const [side, anchorX] of [28, 68].entries()) {
      const inward = side === 0 ? 1 : -1;
      let x = anchorX;
      if (tone === 'happy') {
        // Phase zero deliberately matches the supplied smiling/blushing image.
        const lift = (phase >= 4 && phase <= 6) || (phase >= 12 && phase <= 14) ? -1 : 0;
        if (blink) rect(x - 5, 24, 10, 2);
        else {
          rect(x - 2, 20 + lift, 4, 2);
          rect(x - 4, 22 + lift, 3, 2); rect(x + 1, 22 + lift, 3, 2);
          rect(x - 6, 24 + lift, 3, 3); rect(x + 3, 24 + lift, 3, 3);
        }
        const warm = phase >= 8 && phase < 24;
        rect(x - 8, 30, warm ? 5 : 4, 2, warm ? '#ffb1b1' : '#f5a5a5');
        rect(x + 4, 30, warm ? 5 : 4, 2, warm ? '#ffb1b1' : '#f5a5a5');
      } else if (tone === 'surprised') {
        const grow = phase >= 4 && phase < 10 ? 1 : 0;
        const y = phase >= 10 && phase < 13 ? 1 : 0;
        if (blink) rect(x - 5, 24, 10, 2);
        else {
          rect(x - 3, 17 - grow + y, 6, 2);
          rect(x - 5 - grow, 19 - grow + y, 10 + grow * 2, 10 + grow * 2);
          rect(x - 3, 29 + grow + y, 6, 2);
          rect(x - 2 - grow, 20 - grow + y, 4 + grow * 2, 8 + grow * 2, '#000');
        }
        if (phase >= 3 && phase < 18) {
          rect(x + 7, 11, 2, 4, '#beeaff'); rect(x + 7, 16, 2, 1, '#beeaff');
        }
      } else if (tone === 'concerned' || tone === 'sad') {
        x += gaze;
        if (blink) rect(x - 3, 26, 6, 2);
        else rect(x - 3, 22 + (tone === 'sad' ? 2 : 0), 6, 7);
        const lift = phase >= 4 && phase < 14 ? -1 : 0;
        rect(x - 4, 18 + (inward === 1 ? 1 : 0) + lift, 4, 2);
        rect(x, 18 + (inward === -1 ? 1 : 0) + lift, 4, 2);
        if (tone === 'sad') {
          const fall = Math.floor((phase % 20) / 4);
          const dropX = anchorX + (side === 0 ? 4 : -6);
          if (phase % 20 < 17) {
            rect(dropX, 28 + fall, 2, 2, '#8cd9f5');
            if (fall < 3) rect(dropX, 30 + fall, 2, 1, '#8cd9f5');
          }
        }
      } else if (tone === 'curious') {
        x += gaze;
        if (blink) rect(x - 4, 24, 8, 2);
        else {
          rect(x - 3, side === 0 ? 22 : 20, 6, side === 0 ? 7 : 10);
          rect(x - 4, side === 0 ? 18 : 15, 8, 2);
        }
        if (side === 1) {
          const y = phase >= 8 && phase < 20 ? 5 : 6;
          const color = phase < 28 ? '#beeaff' : '#7fb8d0';
          rect(77, y, 4, 1, color); rect(80, y + 1, 2, 2, color);
          rect(78, y + 3, 3, 1, color); rect(78, y + 4, 1, 2, color);
          rect(78, y + 7, 1, 1, color);
        }
      } else if (tone === 'shy') {
        x += phase >= 8 && phase < 22 ? inward : 0;
        const closed = blink || (phase >= 13 && phase < 18);
        if (closed) rect(x - 4, 25, 8, 2);
        else { rect(x - 3, 24, 6, 6); rect(x - 4, 22, 8, 2); }
        const deep = phase >= 8 && phase < 26;
        rect(anchorX - 8, 31, deep ? 6 : 4, 2, deep ? '#ff9eae' : '#e8a1ac');
        rect(anchorX + 3, 31, deep ? 6 : 4, 2, deep ? '#ff9eae' : '#e8a1ac');
      } else if (tone === 'proud') {
        const lift = phase >= 5 && phase < 17 ? -1 : 0;
        rect(x - 4, 19 + lift, 8, 2);
        if (blink) rect(x - 5, 25, 10, 2);
        else { rect(x - 4, 23 + lift, 8, 4); rect(x - 2, 27 + lift, 5, 2); }
        if (side === 1 && phase % 20 < 15) sparkle(80, 8, phase % 20 < 8);
        if (side === 0 && phase >= 18 && phase < 29) sparkle(18, 10, false);
      } else if (tone === 'encouraging') {
        const wink = side === 0 && phase >= 7 && phase < 20;
        const lift = phase >= 7 && phase < 13 ? -1 : 0;
        if (blink || wink) {
          rect(x - 5, 24 + lift, 10, 2);
          if (wink) rect(x + 3, 22 + lift, 2, 2);
        } else { rect(x - 3, 20 + lift, 6, 10); rect(x - 1, 19 + lift, 2, 1); }
        if (side === 1 && phase >= 5 && phase < 25) sparkle(80, 10 + (phase < 13 ? 0 : 1), phase < 17, '#b9f5ce');
      }
    }
    this.frames.set(key, canvas);
    return canvas;
  }
}
window.DisplayToneRenderer = DisplayToneRenderer;
