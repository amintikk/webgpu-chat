// Drives the real inference worker (src/engine/worker.ts) in Node on the CPU through a multi-turn chat:
// first message, follow-up (cache reuse), regenerate, new conversation, reasoning on. Catches errors in the
// KV-cache reuse logic without a GPU.   node tools/e2e/worker-turns.mjs [model] [dtype]
import { env } from '@huggingface/transformers';

env.cacheDir = process.env.HF_CACHE ?? `${process.env.HOME}/scratch/webgpu-chat/hf-cache`;
const model = process.argv[2] ?? 'lfm3-1.7b';
const dtype = process.argv[3] ?? 'q4f16';
const pending = new Map();
let waiter = null;
globalThis.self = globalThis;
globalThis.postMessage = (m) => {
  if (m.type === 'delta' || m.type === 'progress' || m.type === 'status') return;
  if (m.type === 'error') console.log('ERROR', m.stage, m.message);
  const r = waiter;
  waiter = null;
  r?.(m);
};
await import('../../src/engine/worker.ts');
env.useBrowserCache = false; // the worker enables Cache Storage, which Node lacks
env.useFSCache = true;
const send = (data) => new Promise((resolve) => ((waiter = resolve), globalThis.onmessage({ data })));
const opts = { temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1, maxTokens: 24, reasoning: false };

const t0 = Date.now();
console.log('load', (await send({ type: 'load', model, lang: 'es', dtype, device: 'cpu' })).type, `${Date.now() - t0} ms`);
const chat = [{ role: 'user', content: 'Hola, ¿cómo te llamas? Responde en una frase.' }];
let ok = true;
async function turn(label, messages, o = opts) {
  const r = await send({ type: 'generate', id: Math.random(), messages, options: o });
  if (r.type !== 'done') {
    ok = false;
    console.log(label, 'FAILED');
    return null;
  }
  if (/<\|[a-z_]+\|>|<\/?think>/.test(r.content)) {
    ok = false;
    console.log(label, 'LEAKED SPECIAL TOKEN', JSON.stringify(r.content));
  }
  console.log(label, `prompt=${r.stats.promptTokens} cached=${r.stats.cachedTokens} new=${r.stats.newTokens}`, JSON.stringify(r.content.slice(0, 70)));
  return r;
}
let r = await turn('turn 1', chat);
chat.push({ role: 'assistant', content: r?.content ?? '' }, { role: 'user', content: '¿Y cuánto es 2 + 2?' });
r = await turn('turn 2', chat);
r = await turn('regenerate', chat);
chat.push({ role: 'assistant', content: r?.content ?? '' }, { role: 'user', content: 'Gracias.' });
await turn('turn 3', chat);
globalThis.onmessage({ data: { type: 'reset' } }); // posts no reply
await turn('new conversation', [{ role: 'user', content: 'Di "hola".' }]);
await turn('reasoning on', [{ role: 'user', content: '¿Cuánto es 3 × 7?' }], { ...opts, reasoning: true, maxTokens: 40 });
// Thinking budget: with 200 tokens the reasoning is closed after 72 and the model must still answer.
const budget = await turn('thinking budget', [{ role: 'user', content: 'Explícame qué es WebGPU en tres frases' }], { ...opts, reasoning: true, maxTokens: 200 });
if (!budget?.content.trim() || !budget.reasoning) {
  ok = false;
  console.log('thinking budget: NO ANSWER', JSON.stringify(budget?.reasoning?.slice(-80)));
} else console.log('thinking budget: reasoning', budget.reasoning.length, 'chars, then answer');
// The conversation continues normally after a budget-closed answer (cache reuse).
await turn('after budget', [
  { role: 'user', content: 'Explícame qué es WebGPU en tres frases' },
  { role: 'assistant', content: budget?.content ?? '', reasoning: budget?.reasoning },
  { role: 'user', content: 'Gracias' },
]);
console.log(ok ? 'ALL OK' : 'SOME TURNS FAILED');
process.exit(ok ? 0 : 1);
