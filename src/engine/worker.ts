/// <reference lib="webworker" />
// Inference worker: Transformers.js v4 on WebGPU. The page never touches the model; it sends chat
// messages and receives text deltas. Nothing here talks to any server other than the Hugging Face
// file downloads (pinned revisions), which Transformers.js keeps in Cache Storage.
import {
  AutoTokenizer,
  BaseStreamer,
  DynamicCache,
  Gemma4ForCausalLM,
  InterruptableStoppingCriteria,
  Lfm3ForCausalLM,
  Qwen3_5ForCausalLM,
  StoppingCriteriaList,
  Tensor,
  env,
} from '@huggingface/transformers';
import { modelById, type ModelInfo } from './models.ts';
import type { ChatMessage, FromWorker, GenOptions, ToWorker } from './protocol.ts';
import { splitReasoning } from './reasoning.ts';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Any = any;

env.allowLocalModels = false;
env.useBrowserCache = true;
// Load the ONNX Runtime WebAssembly glue straight from /ort/ (same origin): the default wraps it in a blob: URL,
// which the site's Content-Security-Policy does not allow.
(env as unknown as { useWasmCache: boolean }).useWasmCache = false;
const onnx = (env.backends as Any).onnx;
// The WebGPU bundle of ONNX Runtime already contains its JavaScript glue; only the .wasm binary is fetched, from
// this site (the default points to a CDN, which the Content-Security-Policy blocks).
if (onnx?.wasm) {
  onnx.wasm.wasmPaths = { wasm: '/ort/ort-wasm-simd-threaded.asyncify.wasm' };
  onnx.wasm.numThreads = 1;
}

const CLASSES: Record<string, Any> = { Lfm3ForCausalLM, Qwen3_5ForCausalLM, Gemma4ForCausalLM };

const post = (m: FromWorker) => (self as unknown as Worker).postMessage(m);

// ---------------------------------------------------------------------------------------------
// Cache reuse between turns.
//
// Transformers.js replaces (and frees) GPU tensors in a DynamicCache when it advances. To keep a
// snapshot of the cache at the end of the conversation history, while generation continues from it,
// tensors referenced by a snapshot are protected from that disposal.
const protectedTensors = new WeakSet<object>();
const freed = new WeakSet<object>(); // tensors already released by Transformers.js
const origUpdate = (DynamicCache as Any).prototype.update;
(DynamicCache as Any).prototype.update = function (entries: Record<string, Any>) {
  for (const k in entries) {
    const old = this[k];
    if (!old || old === entries[k]) continue;
    if (protectedTensors.has(old)) this[k] = undefined; // keep it alive for the snapshot
    else if (old.location === 'gpu-buffer') freed.add(old);
  }
  return origUpdate.call(this, entries);
};

interface CacheState {
  cache: Any; // DynamicCache
  tokens: number[]; // token ids whose keys/values the cache holds
}

let model: Any = null;
let tokenizer: Any = null;
let info: ModelInfo | null = null;
let live: CacheState | null = null; // where the last generation stopped
let snap: CacheState | null = null; // end of the history before the last answer
const stopper = new InterruptableStoppingCriteria();

function entries(c: Any): Any[] {
  return Object.values(c).filter(Boolean);
}

async function disposeState(s: CacheState | null, keep: CacheState | null = null) {
  if (!s) return;
  const kept = new Set(keep ? entries(keep.cache) : []);
  for (const t of entries(s.cache)) {
    if (!kept.has(t) && !freed.has(t) && t.location === 'gpu-buffer') {
      protectedTensors.delete(t);
      freed.add(t);
      await t.dispose();
    }
  }
}

function copyCache(c: Any): Any {
  const out: Record<string, Any> = {};
  for (const [k, v] of Object.entries(c)) if (v) out[k] = v;
  return new DynamicCache(out);
}

const isPrefix = (a: number[], b: number[]) => a.length <= b.length && a.every((x, i) => x === b[i]);

async function resetCaches() {
  await disposeState(snap, live);
  await disposeState(live);
  snap = live = null;
}

// ---------------------------------------------------------------------------------------------

async function load(id: string, dtypeOverride?: string, device = 'webgpu') {
  const m = modelById(id);
  if (!m) throw new Error(`Unknown model ${id}`);
  if (info?.id === m.id && model) {
    post({ type: 'ready', model: m.id, loadMs: 0 });
    return;
  }
  await unload();
  const t0 = performance.now();
  const totals = new Map<string, { loaded: number; total: number }>();
  const expected = new Map(m.files.map((f) => [f.path, f.bytes]));
  const progress = (p: Any) => {
    if (p.status === 'progress' && p.file) {
      totals.set(p.file, { loaded: p.loaded ?? 0, total: p.total || expected.get(p.file) || 0 });
      // Files not started yet still count towards the total, so the bar never jumps backwards.
      let loaded = 0;
      let total = 0;
      for (const f of m.files) {
        const s = totals.get(f.path);
        loaded += s?.loaded ?? 0;
        total += s?.total ?? f.bytes;
      }
      post({ type: 'progress', loaded, total, file: p.file });
    } else if (p.status === 'done' && p.file) {
      const s = totals.get(p.file);
      if (s) s.loaded = s.total;
    }
  };
  post({ type: 'status', stage: 'download' });
  const opts = { revision: m.revision, progress_callback: progress };
  tokenizer = await AutoTokenizer.from_pretrained(m.repo, opts);
  const cls = CLASSES[m.cls];
  const dtype = dtypeOverride ? (typeof m.dtype === 'string' ? dtypeOverride : Object.fromEntries(Object.keys(m.dtype).map((k) => [k, dtypeOverride]))) : m.dtype;
  model = await cls.from_pretrained(m.repo, { ...opts, dtype, device });
  // Transformers.js 4.3 (Qwen3-VL family): continuing from a cache sets `pixel_values = null`, and a
  // multi-token continuation then treats null as an image and calls the vision encoder, which a text-only
  // load does not have. Drop the null so the turn is processed as text.
  const prepare = model.prepare_inputs_for_generation?.bind(model);
  if (prepare)
    model.prepare_inputs_for_generation = (...a: Any[]) => {
      const inputs = prepare(...a);
      if (inputs && inputs.pixel_values == null) delete inputs.pixel_values;
      return inputs;
    };
  info = m;
  // One short forward pass compiles the GPU kernels now rather than on the first message.
  post({ type: 'status', stage: 'warmup' });
  const ids = tokenizer.apply_chat_template([{ role: 'user', content: 'Hi' }], { add_generation_prompt: true, return_dict: true });
  await model.generate({ ...ids, max_new_tokens: 1, do_sample: false });
  post({ type: 'ready', model: m.id, loadMs: performance.now() - t0 });
}

async function unload() {
  await resetCaches();
  if (model) await model.dispose();
  model = tokenizer = info = null;
}

function template(messages: ChatMessage[], generationPrompt: boolean, reasoning: boolean): number[] {
  const msgs = messages.map((m) =>
    m.role === 'assistant' && m.reasoning ? { role: m.role, content: m.content, reasoning_content: m.reasoning } : { role: m.role, content: m.content },
  );
  const out = tokenizer.apply_chat_template(msgs, {
    add_generation_prompt: generationPrompt,
    tokenize: true,
    return_tensor: false,
    enable_thinking: reasoning,
  });
  // Transformers.js 4.3 returns { input_ids, attention_mask }; older versions a plain or nested array.
  let ids: Any = out && !Array.isArray(out) && 'input_ids' in out ? out.input_ids : out;
  if (Array.isArray(ids) && Array.isArray(ids[0])) ids = ids[0];
  const list = Array.from(ids as ArrayLike<number>, Number);
  if (!list.length) throw new Error('The chat template produced no tokens');
  return list;
}

async function prefill(state: CacheState | null, ids: number[]): Promise<CacheState> {
  // Run the model over `ids` (continuing `state` if given) and return the cache that covers them.
  const input_ids = new Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
  const pkv = state ? copyCache(state.cache) : undefined;
  const out = await model.generate({
    input_ids,
    attention_mask: new Tensor('int64', new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
    past_key_values: pkv,
    max_new_tokens: 1,
    do_sample: false,
    return_dict_in_generate: true,
  });
  return { cache: out.past_key_values, tokens: ids.slice() };
}

class DeltaStreamer extends BaseStreamer {
  ids: number[] = [];
  first = 0;
  private onToken: (ids: number[]) => void;
  constructor(onToken: (ids: number[]) => void) {
    super();
    this.onToken = onToken;
  }
  put(value: Any) {
    const row = value[0] ?? value;
    const list: number[] = Array.from(row as ArrayLike<bigint | number>, Number);
    if (!this.first) {
      this.first = performance.now(); // the first call carries the prompt
      return;
    }
    this.ids.push(...list);
    this.onToken(this.ids);
  }
  end() {}
}

async function generate(id: number, messages: ChatMessage[], o: GenOptions) {
  if (!model || !info) throw new Error('No model loaded');
  const m = info;
  stopper.reset();
  const full = template(messages, true, o.reasoning);
  const hist = template(messages, false, o.reasoning);
  const boundary = isPrefix(hist, full) && hist.length < full.length ? hist.length : 0;

  // Longest cached prefix of this prompt (strictly shorter, so at least one token is fed).
  const candidates = [live, snap].filter((s): s is CacheState => !!s && s.tokens.length < full.length && isPrefix(s.tokens, full));
  candidates.sort((a, b) => b.tokens.length - a.tokens.length);
  let base: CacheState | null = candidates[0] ?? null;
  const t0 = performance.now();
  const cached = base?.tokens.length ?? 0;

  // Snapshot at the end of the history (before the generation prompt), reused by the next turn
  // even though chat templates rewrite earlier answers (e.g. dropping their reasoning).
  let newSnap: CacheState | null = null;
  if (boundary > cached) {
    newSnap = await prefill(base, full.slice(0, boundary));
    base = newSnap;
  } else if (base === snap && snap) {
    newSnap = snap;
  }
  if (newSnap) for (const t of entries(newSnap.cache)) protectedTensors.add(t);
  // Free what is no longer reachable (tensors consumed by generation are already released).
  for (const s of [live, snap]) if (s && s !== base && s !== newSnap) await disposeState(s, base);
  live = null;
  snap = newSnap;

  const prefillEnd = { t: 0 };
  let lastPost = 0;
  const streamer = new DeltaStreamer((ids) => {
    if (!prefillEnd.t) prefillEnd.t = performance.now();
    const now = performance.now();
    if (now - lastPost < 40 && ids.length > 1) return; // ~25 updates/s is plenty
    lastPost = now;
    const text = tokenizer.decode(ids, { skip_special_tokens: false });
    const r = splitReasoning(text, m.think, o.reasoning, specials());
    post({ type: 'delta', id, content: r.content, reasoning: r.reasoning, thinking: r.thinking, tokens: ids.length });
  });

  const input_ids = new Tensor('int64', BigInt64Array.from(full.map(BigInt)), [1, full.length]);
  const criteria = new StoppingCriteriaList();
  criteria.push(stopper);
  const out = await model.generate({
    input_ids,
    attention_mask: new Tensor('int64', new BigInt64Array(full.length).fill(1n), [1, full.length]),
    past_key_values: base ? copyCache(base.cache) : undefined,
    max_new_tokens: o.maxTokens,
    do_sample: o.temperature > 0,
    temperature: Math.max(o.temperature, 1e-3),
    top_p: o.topP,
    top_k: o.topK,
    repetition_penalty: o.repetitionPenalty,
    streamer,
    stopping_criteria: criteria,
    return_dict_in_generate: true,
  });
  const end = performance.now();
  const seq: number[] = Array.from(out.sequences.data as ArrayLike<bigint>, Number);
  const gen = seq.slice(full.length);
  // The cache holds the prompt and every generated token except the last one (never fed back).
  live = { cache: out.past_key_values, tokens: full.concat(gen.slice(0, -1)) };
  const text = tokenizer.decode(gen, { skip_special_tokens: false });
  const r = splitReasoning(text, m.think, o.reasoning, specials());
  const eos = new Set<number>([].concat(model.generation_config?.eos_token_id ?? []).map(Number));
  const pf = (prefillEnd.t || end) - t0;
  const decodeMs = end - (prefillEnd.t || end);
  post({
    type: 'done',
    id,
    content: r.content,
    reasoning: r.reasoning,
    stats: {
      promptTokens: full.length,
      cachedTokens: cached,
      newTokens: gen.length,
      prefillMs: pf,
      decodeMs,
      tokensPerSecond: gen.length > 1 && decodeMs > 0 ? ((gen.length - 1) * 1000) / decodeMs : 0,
      stopped: stopper.interrupted ? 'user' : eos.has(gen.at(-1) ?? -1) ? 'eos' : 'length',
    },
  });
}

/** Fixed benchmark: same prompt, greedy decoding, exactly 128 new tokens; the second of two runs is reported. */
async function bench(id: number) {
  if (!model) throw new Error('No model loaded');
  await resetCaches();
  const ids = template([{ role: 'user', content: 'Write a detailed paragraph about the history of the Internet.' }], true, false);
  let result: Any = null;
  for (let run = 0; run < 2; run++) {
    const input_ids = new Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
    let first = 0;
    const streamer = new DeltaStreamer(() => {
      if (!first) first = performance.now();
    });
    const t0 = performance.now();
    const out = await model.generate({
      input_ids,
      attention_mask: new Tensor('int64', new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
      max_new_tokens: 128,
      min_new_tokens: 128,
      do_sample: false,
      streamer,
      return_dict_in_generate: true,
    });
    const end = performance.now();
    await disposeState({ cache: out.past_key_values, tokens: [] });
    const n = Number(out.sequences.dims.at(-1)) - ids.length;
    const prefillMs = (first || end) - t0;
    const decodeMs = end - (first || end);
    result = { promptTokens: ids.length, newTokens: n, prefillMs, decodeMs, tokensPerSecond: ((n - 1) * 1000) / decodeMs, prefillTokensPerSecond: (ids.length * 1000) / prefillMs, runs: run + 1 };
  }
  post({ type: 'bench', id, result });
}

let specialCache: string[] | null = null;
function specials(): string[] {
  if (!specialCache) {
    // Removed from the text the user sees, after the reasoning markers have been used to split it.
    const list: Any[] = tokenizer?.all_special_tokens ?? tokenizer?._tokenizer?.added_tokens?.filter((t: Any) => t.special).map((t: Any) => t.content) ?? [];
    specialCache = list.map(String).filter(Boolean);
  }
  return specialCache;
}

let queue = Promise.resolve();
self.onmessage = (e: MessageEvent<ToWorker>) => {
  const msg = e.data;
  if (msg.type === 'stop') {
    stopper.interrupt();
    return;
  }
  queue = queue.then(async () => {
    try {
      if (msg.type === 'load') {
        specialCache = null;
        await load(msg.model, msg.dtype, msg.device);
      } else if (msg.type === 'generate') {
        await generate(msg.id, msg.messages, msg.options);
      } else if (msg.type === 'bench') {
        await bench(msg.id);
      } else if (msg.type === 'count') {
        post({ type: 'count', id: msg.id, tokens: tokenizer ? template(msg.messages, true, msg.reasoning).length : 0 });
      } else if (msg.type === 'reset') {
        await resetCaches();
      } else if (msg.type === 'unload') {
        await unload();
        post({ type: 'unloaded' });
      }
    } catch (err) {
      console.error(err);
      const message = err instanceof Error ? err.message : String(err);
      post({ type: 'error', id: 'id' in msg ? msg.id : undefined, message, stage: msg.type === 'load' ? 'load' : 'generate' });
      if (msg.type === 'generate') await resetCaches().catch(() => {});
    }
  });
};
