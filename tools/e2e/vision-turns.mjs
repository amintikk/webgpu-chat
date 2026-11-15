// Drives the real worker (src/engine/worker.ts) in Node on the CPU through a chat with pictures: a picture and a
// question, a text follow-up (cache reuse with the picture inside), then a second picture. WGC_JOB
//   node tools/e2e/vision-turns.mjs [model] <image1> <image2>
import { env } from '@huggingface/transformers';
import { readFileSync } from 'node:fs';

env.cacheDir = process.env.HF_CACHE ?? `${process.env.HOME}/scratch/webgpu-chat/hf-cache`;
const [model = 'qwen3.5-4b', img1, img2] = process.argv.slice(2);
let waiter = null;
globalThis.self = globalThis;
globalThis.postMessage = (m) => {
  if (m.type === 'log') return console.log(m.text);
  if (m.type === 'vision') return console.log(`vision ${m.loaded}/${m.total}`);
  if (m.type === 'delta' || m.type === 'progress' || m.type === 'status') return;
  if (m.type === 'error') console.log('ERROR', m.stage, m.message);
  const r = waiter;
  waiter = null;
  r?.(m);
};
await import('../../src/engine/worker.ts');
env.useBrowserCache = false;
env.useFSCache = true;
const send = (data) => new Promise((resolve) => ((waiter = resolve), globalThis.onmessage({ data })));
const opts = { temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1, maxTokens: 40, reasoning: false };
const pic = (path, id) => ({ id, blob: new Blob([readFileSync(path)], { type: 'image/png' }) });
console.log('load', (await send({ type: 'load', model, lang: 'es', dtype: 'q4f16', device: 'cpu' })).type);
let ok = true;
async function turn(label, messages) {
  const t0 = Date.now();
  const r = await send({ type: 'generate', id: Math.random(), messages, options: opts });
  if (r.type !== 'done') { ok = false; console.log(label, 'FAILED'); return ''; }
  console.log(label, `prompt=${r.stats.promptTokens} cached=${r.stats.cachedTokens} ${Date.now() - t0} ms`, JSON.stringify(r.content));
  return r.content;
}
const chat = [{ role: 'user', content: 'What shapes and colours are in this picture? One sentence.', images: [pic(img1, 'a')] }];
chat.push({ role: 'assistant', content: await turn('picture', chat) });
chat.push({ role: 'user', content: 'Which shape is on the right?' });
chat.push({ role: 'assistant', content: await turn('follow-up', chat) });
chat.push({ role: 'user', content: 'And what does this second picture show? One sentence.', images: [pic(img2, 'b')] });
await turn('second picture', chat);
const n = await send({ type: 'count', id: 9, messages: chat, reasoning: false });
console.log('count', n.tokens);
process.exit(ok ? 0 : 1);
