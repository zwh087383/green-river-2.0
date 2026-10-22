/* Keyword scoring/negation adapted from Project N.E.K.O., Apache-2.0.
 * Copyright 2025-2026 Project N.E.K.O. Team; Copyright (c) 2025 Hongzhi Wen.
 * See licenses/neko/. Green River changes: shared rules, 8 tones, streaming first.
 * Optional model refinement is never awaited by answer streaming/audio playback.
 */
class AnswerEmotionClient {
  constructor(onTone) {
    this.onTone = onTone;
    this.generation = 0;
    this.requestVersion = 0;
    this.reset();
    window.addEventListener('pagehide', () => this.reset());
  }

  reset() {
    this.generation++;
    this.requestVersion++;
    this.controller?.abort();
    this.controller = null;
    this.previewSent = false;
    this.finalSent = false;
    this.lastText = '';
    this.currentText = '';
    this.interactionTone = 'neutral';
    this.interactionScore = 0;
    this.modelAccepted = false;
    this.modelEvidence = '';
    this.onTone('neutral');
  }

  begin(query) {
    this.reset();
    // Explicit expression requests and interaction context survive a factual reply.
    // User distress maps to concern, never to a mirrored sad face.
    const interaction = this.inferLocal(query, true);
    this.interactionTone = interaction.tone;
    this.interactionScore = interaction.score;
    if (this.interactionTone !== 'neutral') this.onTone(this.interactionTone);
  }

  inferLocal(value, interaction = false) {
    const rules = window.DisplayEmotionRules;
    if (!rules) return { tone: 'neutral', score: 0 };
    const text = String(value || '').replace(/\[[^\]]*\]/g, '').toLowerCase().slice(0, 1200);
    const escape = word => word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const negation = new RegExp('(?:' + rules.negations.map(escape).join('|') +
      ')(?:真的|特别|非常|有点|很|太|那么|这么|感到|觉得|让人|令人|值得|需要|会|能|\\s)*$');
    const negated = (start, end) => {
      let head = text.slice(Math.max(0, start - 24), start).split(/[，。！？；,.!?;\n：:]/).at(-1);
      for (const contrast of rules.contrasts) head = head.split(contrast).at(-1);
      for (const phrase of rules.negation_exceptions) head = head.split(phrase).join('');
      return negation.test(head) || rules.negative_suffixes.some(word => text.slice(end).startsWith(word));
    };
    let best = { tone: 'neutral', score: 0 };
    for (const [tone, cues] of Object.entries(interaction ? rules.interactions : rules.tones)) {
      const hits = [];
      for (const [cue, weight] of Object.entries(cues)) {
        const pattern = /^[a-zA-Z ]+$/.test(cue)
          ? '(?<![a-zA-Z])' + escape(cue) + '(?![a-zA-Z])' : escape(cue);
        for (const match of text.matchAll(new RegExp(pattern, 'g'))) {
          if (!negated(match.index, match.index + cue.length)) hits.push([match.index, match.index + cue.length, weight]);
        }
      }
      hits.sort((a, b) => b[2] - a[2] || a[0] - b[0] || (b[1] - b[0]) - (a[1] - a[0]));
      const occupied = [];
      let score = 0;
      for (const [start, end, weight] of hits) {
        if (occupied.some(([a, b]) => start < b && end > a)) continue;
        occupied.push([start, end]); score += weight;
      }
      score = Math.min(score, 6);
      if (score > best.score) best = { tone, score };
    }
    return best;
  }

  observe(text, final = false) {
    const cleaned = String(text || '').trim();
    if (!cleaned || this.finalSent) return;
    // Run on every delta, including before the first sentence is complete.
    this.currentText = cleaned;
    const local = this.inferLocal(cleaned);
    if (!this.modelAccepted || `${local.tone}:${local.score}` !== this.modelEvidence) {
      this.onTone(this.interactionScore >= 3 || local.tone === 'neutral' ? this.interactionTone : local.tone);
    }
    if (!final && this.previewSent) return;
    if (!final && !/[。！？!?\n]/.test(cleaned) && cleaned.length < 24) return;
    if (final) this.finalSent = true;
    else this.previewSent = true;
    const sample = cleaned.slice(0, 1200);
    if (sample === this.lastText) return;
    this.lastText = sample;
    void this.analyze(sample);
  }

  acceptTone(tone) {
    const tones = ['neutral', ...Object.keys(window.DisplayEmotionRules?.tones || {})];
    if (tones.includes(tone)) this.onTone(this.interactionScore >= 3 || tone === 'neutral' ? this.interactionTone : tone);
  }

  async analyze(text) {
    this.controller?.abort();
    const controller = new AbortController();
    this.controller = controller;
    const generation = this.generation;
    const version = ++this.requestVersion;
    const timeout = setTimeout(() => controller.abort(), 2200);
    try {
      const response = await fetch('/api/display/analyze', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }), signal: controller.signal, cache: 'no-store'
      });
      if (!response.ok) return;
      const result = await response.json();
      if (controller.signal.aborted || generation !== this.generation || version !== this.requestVersion) return;
      const tones = ['neutral', 'happy', 'concerned', 'sad', 'surprised', 'curious', 'shy', 'proud', 'encouraging'];
      const sampleEvidence = this.inferLocal(text);
      const currentEvidence = this.inferLocal(this.currentText);
      // A preview of an earlier clause cannot erase new streamed evidence.
      if (sampleEvidence.tone !== currentEvidence.tone || sampleEvidence.score !== currentEvidence.score) return;
      // Weak/uncertain neutral must not erase clear local evidence.
      if (result.tone === 'neutral' && this.inferLocal(this.lastText).score >= 2 && Number(result.confidence || 0) < 0.85) return;
      if (tones.includes(result.tone)) {
        const evidence = this.inferLocal(text);
        this.modelEvidence = `${evidence.tone}:${evidence.score}`;
        this.modelAccepted = true;
        // Neutral factual prose must not erase a compliment/request's expression.
        this.onTone(this.interactionScore >= 3 || result.tone === 'neutral' ? this.interactionTone : result.tone);
      }
    } catch (_) {
      // Keep the synchronous local expression on timeout/provider errors.
    } finally {
      clearTimeout(timeout);
      if (this.controller === controller) this.controller = null;
    }
  }
}
window.AnswerEmotionClient = AnswerEmotionClient;
