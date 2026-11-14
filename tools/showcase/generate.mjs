// Generates the landing-page examples with the recommended model (same ONNX files and chat template as the
// app; greedy decoding, reasoning off) and writes src/site/showcase.json. Answers are kept verbatim.
//   node tools/showcase/generate.mjs [model]
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { AutoTokenizer, Qwen3_5ForCausalLM, env } from '@huggingface/transformers';

env.cacheDir = process.env.HF_CACHE ?? `${process.env.HOME}/scratch/webgpu-chat/hf-cache`;
const repo = 'onnx-community/Qwen3.5-4B-ONNX';
const revision = '23ac3352f288351eddcdf141b32f4f66587f7e2d';
const prompts = JSON.parse(readFileSync('tools/showcase/prompts.json', 'utf8'));
const out = 'src/site/showcase.json';
const done = existsSync(out) ? JSON.parse(readFileSync(out, 'utf8')) : { model: 'Qwen3.5 4B', es: [], en: [] };
const tok = await AutoTokenizer.from_pretrained(repo, { revision });
const model = await Qwen3_5ForCausalLM.from_pretrained(repo, { revision, dtype: { embed_tokens: 'q4f16', decoder_model_merged: 'q4f16' }, device: 'cpu' });
for (const lang of ['es', 'en']) {
  for (const [i, p] of prompts[lang].entries()) {
    if (done[lang][i]?.prompt === p.prompt) continue;
    const inputs = tok.apply_chat_template([{ role: 'user', content: p.prompt }], { add_generation_prompt: true, return_dict: true, enable_thinking: false });
    const n = inputs.input_ids.dims[1];
    const t0 = Date.now();
    const o = await model.generate({ ...inputs, max_new_tokens: 260, do_sample: false });
    const ids = Array.from(o.slice(null, [n, null]).data, Number);
    const answer = tok.decode(ids, { skip_special_tokens: true }).replace(/^\s*<think>\s*<\/think>\s*/, '').trim();
    done[lang][i] = { ...p, answer, tokens: ids.length, truncated: ids.length >= 260 };
    writeFileSync(out, JSON.stringify(done, null, 2));
    console.log(lang, i, ids.length, `${((Date.now() - t0) / 1000).toFixed(0)}s`, JSON.stringify(answer.slice(0, 60)));
  }
}
console.log('done');
