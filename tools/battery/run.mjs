// Runs the prompt battery (prompts.json) through a model with Transformers.js in Node, using the same
// ONNX files and official chat template as the web app. Greedy decoding, reasoning off, one prompt per
// conversation. Writes the raw answers to results/battery/<name>.json; grade.mjs scores them.
//
//   node tools/battery/run.mjs <name> [--dtype q4f16|q4] [--limit N]
//
// The server that produced the published results has no GPU, so this runs on the CPU execution
// provider; answers match the WebGPU ones except for floating-point rounding.
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { AutoTokenizer, Lfm3ForCausalLM, Qwen3_5ForCausalLM, Gemma4ForCausalLM, env } from '@huggingface/transformers';

const MODELS = {
  'lfm3-1.7b': { repo: 'onnx-community/LFM3-1.7B-ONNX', revision: 'ab9fec4c681a25bec3b48b0b97a62fc2660a2985', cls: Lfm3ForCausalLM, single: true },
  'qwen3.5-4b': { repo: 'onnx-community/Qwen3.5-4B-ONNX', revision: '23ac3352f288351eddcdf141b32f4f66587f7e2d', cls: Qwen3_5ForCausalLM },
  'qwen3.5-2b': { repo: 'onnx-community/Qwen3.5-2B-ONNX', revision: '315216e893038beefcc6eba41ac168913dd0d879', cls: Qwen3_5ForCausalLM },
  'gemma4-12b': { repo: process.env.GEMMA_REPO ?? '', revision: process.env.GEMMA_REVISION ?? 'main', cls: Gemma4ForCausalLM, local: process.env.GEMMA_DIR },
};

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const name = args[0];
const opt = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const dtypeName = opt('--dtype', 'q4f16');
const limit = Number(opt('--limit', '0'));
const maxNew = Number(opt('--max-new', '768'));
const m = MODELS[name];
if (!m) throw new Error(`unknown model ${name}`);

env.cacheDir = process.env.HF_CACHE ?? join(process.env.HOME, 'scratch/webgpu-chat/hf-cache');
if (m.local) {
  env.allowRemoteModels = false;
  env.localModelPath = dirname(m.local);
}
const id = m.local ? m.local.split('/').pop() : m.repo;
const dtype = m.single ? dtypeName : { embed_tokens: dtypeName, decoder_model_merged: dtypeName };

const prompts = JSON.parse(readFileSync(join(here, 'prompts.json'), 'utf8')).slice(0, limit || undefined);
const outDir = join(here, '../../results/battery');
mkdirSync(outDir, { recursive: true });
const outFile = join(outDir, `${name}${opt('--suffix', '')}.json`);
const prev = existsSync(outFile) ? JSON.parse(readFileSync(outFile, 'utf8')) : null;
const done = new Map((prev?.answers ?? []).map((a) => [a.id, a]));

const t0 = Date.now();
const tokenizer = await AutoTokenizer.from_pretrained(id, { revision: m.revision });
// --mmap: map the weight files instead of copying them into ONNX Runtime's memory (slower; for big models on a
// machine without spare RAM).
const session_options = args.includes('--mmap') ? { extra: { session: { disable_prepacking: '1' } } } : undefined;
const model = await m.cls.from_pretrained(id, { revision: m.revision, dtype, device: 'cpu', session_options });
console.log(`${name} loaded in ${((Date.now() - t0) / 1000).toFixed(0)} s`);

const answers = [];
for (const p of prompts) {
  if (done.has(p.id)) {
    answers.push(done.get(p.id));
    continue;
  }
  const inputs = tokenizer.apply_chat_template([{ role: 'user', content: p.prompt }], { add_generation_prompt: true, return_dict: true, enable_thinking: false });
  const start = Date.now();
  const out = await model.generate({ ...inputs, max_new_tokens: maxNew, do_sample: false, repetition_penalty: 1.0 });
  const ms = Date.now() - start;
  const n = inputs.input_ids.dims[1];
  const gen = Array.from(out.slice(null, [n, null]).data, Number);
  const text = tokenizer.decode(gen, { skip_special_tokens: true }).replace(/^\s*<think>\s*<\/think>\s*/, '').trim();
  answers.push({ id: p.id, answer: text, tokens: gen.length, ms, truncated: gen.length >= maxNew });
  console.log(`[${p.id}] ${gen.length} tok, ${(gen.length / (ms / 1000)).toFixed(1)} tok/s: ${JSON.stringify(text.slice(0, 120))}`);
  writeFileSync(outFile, JSON.stringify({ model: name, repo: m.repo || id, revision: m.revision, dtype: dtypeName, device: 'cpu (onnxruntime-node)', decoding: 'greedy, reasoning off', max_new_tokens: maxNew, answers }, null, 2));
}
writeFileSync(outFile, JSON.stringify({ model: name, repo: m.repo || id, revision: m.revision, dtype: dtypeName, device: 'cpu (onnxruntime-node)', decoding: 'greedy, reasoning off', max_new_tokens: maxNew, answers }, null, 2));
console.log('done', outFile);
