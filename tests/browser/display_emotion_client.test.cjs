const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../../src/rag_system/web/assets/display_emotion.js'), 'utf8');

function fixture() {
  const pending = [], tones = [];
  const window = { addEventListener() {}, DisplayEmotionRules: JSON.parse(fs.readFileSync(path.join(__dirname, '../../src/rag_system/web/assets/emotion_rules.json'), 'utf8')) };
  const context = vm.createContext({ window, AbortController, setTimeout, clearTimeout,
    fetch: (url, options) => new Promise(resolve => pending.push({ url, options, resolve })) });
  vm.runInContext(source, context);
  const client = new window.AnswerEmotionClient(tone => tones.push(tone));
  const deliver = async (index, tone, confidence = 0.95) => {
    pending[index].resolve({ ok: true, json: async () => ({ tone, confidence }) });
    await new Promise(resolve => setImmediate(resolve));
  };
  return { client, tones, pending, deliver };
}

test('streaming proceeds while analysis is pending; one preview and one final at most', async () => {
  const { client, tones, pending, deliver } = fixture();
  client.observe('太好了，动物成功获救！');
  assert.equal(pending.length, 1);
  client.observe('太好了，动物成功获救！接下来继续讲解。');
  assert.equal(pending.length, 1);
  client.observe('请小心，接下来的任务仍有风险。', true);
  assert.equal(pending.length, 2);
  client.observe('额外的重复结束通知。', true);
  assert.equal(pending.length, 2);
  await deliver(1, 'concerned');
  await deliver(0, 'happy'); // A transport ignoring abort still cannot overwrite newer analysis.
  assert.equal(tones[1], 'happy');
  assert.equal(tones.at(-1), 'concerned');
});

test('interruption/new turn rejects old tone even if the transport ignores cancellation', async () => {
  const { client, tones, pending, deliver } = fixture();
  client.observe('旧回答太好了！', true);
  client.reset();
  client.observe('新回答令人痛心。', true);
  assert.equal(pending[0].options.signal.aborted, true);
  await deliver(0, 'happy');
  assert.equal(tones.at(-1), 'sad');
  await deliver(1, 'sad');
  assert.equal(tones.at(-1), 'sad');
});

test('same preview/final text reuses analysis, and an invalid result is ignored', async () => {
  const { client, pending, tones, deliver } = fixture();
  client.observe('太好了！');
  client.observe('太好了！', true);
  assert.equal(pending.length, 1);
  await deliver(0, 'unknown');
  assert.equal(tones.at(-1), 'happy');
});

test('new outward emotions pass through the browser session client', async () => {
  const { client, tones, deliver } = fixture();
  const added = ['curious', 'shy', 'proud', 'encouraging'];
  for (const [index, tone] of added.entries()) {
    client.reset();
    client.observe('本轮回答内容。', true);
    await deliver(index, tone);
    assert.equal(tones.at(-1), tone);
  }
});


test('keywords fire synchronously before punctuation or any model request', () => {
  const { client, pending, tones } = fixture();
  client.observe('太好了');
  assert.equal(tones.at(-1), 'happy');
  assert.equal(pending.length, 0);
  client.reset();
});

test('explicit expression stays visible when reply or optional classifier has another tone', async () => {
  const { client, tones, deliver } = fixture();
  client.begin('展示害羞的表情');
  client.observe('太好了，我很开心！', true);
  assert.equal(tones.at(-1), 'shy');
  await deliver(0, 'happy', 0.99);
  assert.equal(tones.at(-1), 'shy');
  client.reset();
});

test('streamed negation clears a provisional expression without waiting for cloud', () => {
  const { client, tones } = fixture();
  client.observe('我高兴');
  assert.equal(tones.at(-1), 'happy');
  client.observe('我高兴不起来');
  assert.equal(tones.at(-1), 'neutral');
  client.reset();
});

test('compliments use shy context, distress is not mirrored, new turns clear context', () => {
  const { client, tones } = fixture();
  client.begin('小绿你真可爱');
  assert.equal(tones.at(-1), 'shy');
  client.observe('能帮到你我也很开心');
  assert.equal(tones.at(-1), 'happy');
  client.begin('我很难过');
  assert.equal(tones.at(-1), 'concerned');
  client.begin('普通知识问题');
  assert.equal(tones.at(-1), 'neutral');
});

test('neutral model prose preserves requested expression and compliment context', async () => {
  const { client, tones, deliver } = fixture();
  client.begin('小绿你真可爱');
  client.observe('我会继续认真为你讲解。', true);
  await deliver(0, 'neutral', 0.99);
  assert.equal(tones.at(-1), 'shy');
  client.begin('展示惊讶的表情');
  client.observe('好的，可以这样展示。', true);
  await deliver(1, 'neutral', 0.99);
  assert.equal(tones.at(-1), 'surprised');
  client.reset();
});

test('weak model neutral preserves keywords; confident correction can clear them', async () => {
  const { client, tones, deliver } = fixture();
  client.observe('一起守护江河。');
  await deliver(0, 'neutral', 0.7);
  assert.equal(tones.at(-1), 'encouraging');
  client.observe('一起守护江河。我们可以从身边的小事做起。', true);
  await deliver(1, 'neutral', 0.95);
  assert.equal(tones.at(-1), 'neutral');
});

test('shared cases cover negation, contrast, overlap, repetition and all eight tones', () => {
  const { client } = fixture();
  const cases = JSON.parse(fs.readFileSync(path.join(__dirname, '../emotion_rule_cases.json'), 'utf8'));
  for (const [text, tone] of cases) {
    assert.equal(client.inferLocal(text).tone, tone, text);
    assert.ok(client.inferLocal(text).score <= 6);
  }
  assert.equal(client.inferLocal('非常开心').score, 2); // no overlap amplification
});


test('late preview cannot overwrite a newer streamed emotion', async () => {
  const { client, tones, deliver } = fixture();
  client.observe('太好了。');
  client.observe('太好了。但这个消息令人痛心，我们深感遗憾');
  assert.equal(tones.at(-1), 'sad');
  await deliver(0, 'happy');
  assert.equal(tones.at(-1), 'sad');
  client.reset();
});

test('new keyword evidence still fires after an earlier model-neutral result', async () => {
  const { client, tones, deliver } = fixture();
  client.observe('这条河是长江的重要支流。');
  await deliver(0, 'neutral');
  client.observe('这条河是长江的重要支流。一起守护江河');
  assert.equal(tones.at(-1), 'encouraging');
  client.reset();
});
