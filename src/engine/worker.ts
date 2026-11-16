/// <reference lib="webworker" />
// Inference worker: Transformers.js v4 on WebGPU. The page never touches the model; it sends chat
// messages and receives text deltas. Nothing here talks to any server other than the Hugging Face
// file downloads (pinned revisions), which Transformers.js keeps in Cache Storage.
import { tapConsole } from './console-tap.ts'; // first: ONNX Runtime must bind the wrapped console.log
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
import { MODELS, fileUrl, modelById, sourceUrl, visionFiles, type ModelInfo } from './models.ts';
import type { ChatMessage, FromWorker, GenOptions, ToWorker } from './protocol.ts';
import { GpuProfile, OrtProfile } from './profiling.ts';
import { splitReasoning } from './reasoning.ts';
import { matmulBench } from './mmbench.ts';
import { legacyUrls, moveLegacyCache } from './cachemove.ts';
import { expandImageTokens, forgetImages, processImage, visionInputs } from './vision.ts';
import type { ImageRef } from './protocol.ts';

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

// Transformers.js 4.3 checks that a tokenizer exists by requesting tokenizer_config.json from the `main`
// branch, ignoring the pinned revision. Requests for `main` of our models go to the pinned revision instead,
// served from the browser cache when already downloaded, so loading is pinned and works offline.
const realFetch = globalThis.fetch.bind(globalThis);
const inFlight = new Map<string, Promise<Response | null>>();
const pinnedFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
  // Our models' files, from `main` or from the pinned revision.
  const m = /^https:\/\/huggingface\.co\/([^/]+\/[^/]+)\/resolve\/([^/]+)\/(.+)$/.exec(url);
  const model = m && MODELS.find((x) => x.repo === m[1] && x.revision && (m[2] === 'main' || m[2] === x.revision));
  if (m && model) {
    const pinned = fileUrl(model, m[3]!);
    let cache: Cache | undefined;
    try {
      cache = 'caches' in globalThis ? await caches.open('transformers-cache') : undefined;
      const hit = await cache?.match(pinned);
      if (hit) {
        dbg(`${m[3]}: desde la caché`);
        return hit;
      }
    } catch {
      /* cache unavailable: fall through to the network */
    }
    const partial = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined)).has('range');
    const method = (init?.method ?? (input instanceof Request ? input.method : 'GET')).toUpperCase();
    if (cache && method === 'GET' && !partial && /\.onnx(_data(_\d+)?|\.data)$/.test(m[3]!)) {
      // Two sessions may ask for the same file at once (Gemma's decoder also reads the embeddings file):
      // the second waits for the first download and reads the stored copy.
      const running = inFlight.get(pinned);
      if (running) {
        await running.catch(() => null);
        const hit = await cache.match(pinned);
        if (hit) return hit;
      }
      const job = downloadToCache(cache, pinned, m[3]!, init, sourceUrl(model, m[3]!));
      inFlight.set(pinned, job);
      const stored = await job.finally(() => inFlight.delete(pinned));
      if (stored) return stored;
    }
    const src = sourceUrl(model, m[3]!);
    if (src !== pinned) dbg(`${m[3]}: desde ${src.replace('https://huggingface.co/', '')}`);
    return realFetch(src, init);
  }
  return realFetch(input, init).catch((e) => {
    console.warn('[worker] request failed:', url, e);
    throw e;
  });
}) as typeof fetch;
/**
 * A file that now comes from another source (e.g. a WebGPU-tuned graph) has the same cache key as before, so an
 * old copy would be used forever. Drop cached copies whose size is not the expected one.
 */
async function dropOutdatedFiles(m: ModelInfo) {
  if (!('caches' in globalThis)) return;
  try {
    const cache = await caches.open('transformers-cache');
    for (const f of m.files) {
      if (!f.src) continue;
      const r = await cache.match(fileUrl(m, f.path));
      const size = r ? (await r.clone().blob()).size : 0; // graph files are small (about 1 MB)
      if (r && size !== f.bytes) {
        await cache.delete(fileUrl(m, f.path));
        dbg(`${f.path}: copia en caché desactualizada (${size} bytes), se descarga la nueva`);
      }
    }
  } catch {
    /* cache unavailable */
  }
}

/**
 * Files of older versions of the models (another repo or revision: no current model uses them) are deleted
 * before loading, so a new version of a model does not run out of disk space next to the old one.
 */
async function dropOrphans() {
  if (!('caches' in globalThis)) return;
  try {
    const cache = await caches.open('transformers-cache');
    const known = new Set([...MODELS.flatMap((x) => [...x.files, ...visionFiles(x)].map((f) => fileUrl(x, f.path))), ...legacyUrls()]);
    let bytes = 0;
    for (const req of await cache.keys()) {
      if (known.has(req.url) || !req.url.startsWith('https://huggingface.co/')) continue;
      bytes += Number((await cache.match(req))?.headers.get('content-length')) || 0;
      await cache.delete(req);
    }
    if (bytes) dbg(`archivos de versiones antiguas borrados: ${(bytes / 1e9).toFixed(2)} GB`);
  } catch {
    /* cache unavailable */
  }
}

/** Network progress of weight files streamed to the cache (see downloadToCache); set by load(). */
let onNetProgress: ((file: string, loaded: number, total: number) => void) | null = null;

/**
 * Weight files (1-2 GB each) are streamed straight to the browser cache and then read back from it.
 * Left to itself, Transformers.js reads the whole file into memory and then stores a second, in-memory copy
 * in the cache, which on phones is enough for the system to kill the tab. Returns null on failure (e.g. no
 * space), and the caller downloads it the usual way.
 */
async function downloadToCache(cache: Cache, url: string, file: string, init?: RequestInit, from = url): Promise<Response | null> {
  try {
    // Downloaded from the file's source (`src` in models.ts), stored under the model's own address.
    const res = await realFetch(from, init);
    if (!res.ok) return res;
    const total = Number(res.headers.get('content-length')) || 0;
    dbg(`${file}: descarga a la caché en streaming (${(total / 1e6).toFixed(0)} MB)`);
    if (res.status !== 200 || !res.body || !total) {
      void res.body?.cancel();
      return null;
    }
    let loaded = 0;
    const counted = res.body.pipeThrough(
      new TransformStream<Uint8Array, Uint8Array>({
        transform(chunk, ctl) {
          loaded += chunk.byteLength;
          onNetProgress?.(file, loaded, total);
          ctl.enqueue(chunk);
        },
      }),
    );
    const headers = new Headers(res.headers);
    try {
      await cache.put(url, new Response(counted, { status: 200, headers }));
    } catch (e) {
      // The browser could not store the file (almost always: no free disk space). Downloading it into memory
      // instead would need gigabytes of RAM in the tab and crash it, so stop with a clear error.
      await cache.delete(url).catch(() => {});
      dbg(`${file}: no se pudo guardar en la caché: ${e instanceof Error ? `${e.name}: ${e.message}` : String(e)}`);
      throw new Error(`STORAGE_FULL ${file} (${(total / 1e9).toFixed(2)} GB): ${e instanceof Error ? e.message : String(e)}`);
    }
    if (loaded !== total) {
      // Cut short: never leave a partial file in the cache.
      await cache.delete(url);
      return null;
    }
    return (await cache.match(url)) ?? null;
  } catch (e) {
    if (e instanceof Error && e.message.startsWith('STORAGE_FULL')) throw e;
    console.warn('[worker] streaming to the cache failed, downloading in memory:', file, e);
    dbg(`${file}: el streaming falló, se descarga en memoria: ${e instanceof Error ? `${e.name}: ${e.message}` : String(e)}`);
    return null;
  }
}

// Transformers.js downloads through env.fetch (captured when it loads), so install it there.
(env as unknown as { fetch: typeof fetch }).fetch = pinnedFetch;
globalThis.fetch = pinnedFetch;

const post = (m: FromWorker) => (self as unknown as Worker).postMessage(m);
/** Full text of an error, stack included, for the debug record. */
const errText = (e: unknown): string => (e instanceof Error ? `${e.name}: ${e.message}\n${e.stack ?? ''}` : String(e));
/** Debug mode: a line for the page's record (it keeps only what the page asked for). */
const dbg = (text: string) => post({ type: 'log', text });

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
  images: string[]; // ids of the pictures inside those tokens (image tokens are alike for every picture)
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

async function load(id: string, dtypeOverride?: string, device = 'webgpu', profile = false) {
  const m = modelById(id);
  if (!m) throw new Error(`Unknown model ${id}`);
  if (info?.id === m.id && model) {
    post({ type: 'ready', model: m.id, loadMs: 0 });
    return;
  }
  await unload();
  chunkValidated = null;
  loadedDevice = device;
  const t0 = performance.now();
  const totals = new Map<string, { loaded: number; total: number }>();
  const expected = new Map(m.files.map((f) => [f.path, f.bytes]));
  const report = (file: string, loadedNow: number, totalNow: number) => {
    // The same file may report twice (network, then read back from the cache): keep the furthest point.
    const prev = totals.get(file);
    totals.set(file, { loaded: Math.max(prev?.loaded ?? 0, loadedNow), total: totalNow || prev?.total || expected.get(file) || 0 });
    // Files not started yet still count towards the total, so the bar never jumps backwards.
    let loaded = 0;
    let total = 0;
    for (const f of m.files) {
      const s = totals.get(f.path);
      loaded += s?.loaded ?? 0;
      total += s?.total ?? f.bytes;
    }
    post({ type: 'progress', loaded, total, file });
  };
  onNetProgress = report;
  const progress = (p: Any) => {
    if (p.status === 'initiate' && p.file) dbg(`${p.file}: inicio`);
    if (p.status === 'progress' && p.file) {
      report(p.file, p.loaded ?? 0, p.total);
    } else if (p.status === 'done' && p.file) {
      dbg(`${p.file}: listo`);
      const s = totals.get(p.file);
      if (s) s.loaded = s.total;
    }
  };
  await moveLegacyCache(m); // this model's files cached under its previous repository
  await dropOutdatedFiles(m);
  await dropOrphans();
  dbg(`cargar ${m.id} (${m.cls}) · dispositivo ${device} · dtype ${JSON.stringify(dtypeOverride ?? m.dtype)}`);
  post({ type: 'status', stage: 'download' });
  const opts = { revision: m.revision, progress_callback: progress };
  tokenizer = await AutoTokenizer.from_pretrained(m.repo, opts);
  const cls = CLASSES[m.cls];
  const dtype = dtypeOverride ? (typeof m.dtype === 'string' ? dtypeOverride : Object.fromEntries(Object.keys(m.dtype).map((k) => [k, dtypeOverride]))) : m.dtype;
  dbg('carga del modelo: lectura/descarga de pesos y creación de sesiones ONNX');
  const tSes = performance.now();
  try {
    // Tests in Node run on the CPU: skip weight prepacking and the memory arena so a 12B model fits in less RAM.
    const session_options: Any = device === 'cpu' ? { enableCpuMemArena: false, extra: { session: { disable_prepacking: '1' } } } : {};
    const shared = m.sharedData ? { use_external_data_format: m.sharedData.formats } : {};
    if (profile) session_options.enableProfiling = true;
    if (device === 'webgpu' && safeMode) session_options.executionProviders = [await webgpuProvider()];
    profiling = profile;
    if (m.sharedData) session_options.externalData = m.sharedData.files.map((f) => ({ path: f, data: `onnx/${f}` }));
    model = await cls.from_pretrained(m.repo, { ...opts, dtype, device, ...shared, session_options });
  } catch (e) {
    dbg(`creación de sesiones falló: ${errText(e)}`);
    throw e;
  }
  dbg(`pesos y sesiones listos en ${Math.round(performance.now() - tSes)} ms (incluye E/S; no es tiempo exclusivo de GPU)`);
  if (device === 'webgpu') {
    try {
      const gpuDevice = safeMode ? safeDevice : await onnx.webgpu.device;
      const a = gpuDevice?.adapterInfo;
      dbg(`GPU del motor: ${JSON.stringify(a ? { vendor: a.vendor, architecture: a.architecture, device: a.device, description: a.description, subgroupMinSize: a.subgroupMinSize, subgroupMaxSize: a.subgroupMaxSize } : 'identidad no disponible')}${safeMode ? ' · modo compatible (sin subgrupos)' : ''}; memoria GPU y ocupación: no medidas`);
    } catch { dbg('GPU del motor: identidad no disponible'); }
  }
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
  pictureBlocks = null;
  if (m.vision?.kind === 'gemma') maskedAttention(model.sessions.decoder_model_merged);
  // A few greedy tokens compile the GPU kernels now rather than on the first message, and check the answer.
  post({ type: 'status', stage: 'warmup' });
  const ids = tokenizer.apply_chat_template([{ role: 'user', content: 'Hi' }], { add_generation_prompt: true, return_dict: true });
  const tWarm = performance.now();
  const warm = await model.generate({ ...ids, max_new_tokens: WARMUP_TOKENS, do_sample: false, return_dict_in_generate: true });
  const said: number[] = Array.from(warm.sequences.data as ArrayLike<bigint>, Number).slice(ids.input_ids.dims.at(-1));
  await disposeState({ cache: warm.past_key_values ?? {}, tokens: [], images: [] });
  dbg(`primera pasada (compilar kernels) en ${Math.round(performance.now() - tWarm)} ms: ${JSON.stringify(tokenizer.decode(said, { skip_special_tokens: false }))}`);
  if (device === 'webgpu' && degenerate(said)) {
    if (!safeMode) {
      // Seen on Linux (Brave): Qwen answered "aaaa…" while LFM3 was fine. The GPU's subgroup fast paths are the
      // part that depends on the driver; load again without them.
      dbg('la prueba tras la carga devolvió un token repetido: la GPU calcula mal con este modelo; se vuelve a cargar en modo compatible (sin subgrupos)');
      safeMode = true;
      await unload();
      return load(id, dtypeOverride, device, profile);
    }
    await unload();
    throw new Error('gpu-garbage'); // the page explains it (loadErrorText)
  }
  post({ type: 'ready', model: m.id, loadMs: performance.now() - t0 });
}

const WARMUP_TOKENS = 8;
/** A broken GPU computation answers one token (or two) over and over: "aaaa…". */
const degenerate = (ids: number[]) => ids.length >= WARMUP_TOKENS && new Set(ids).size <= 2;

/**
 * Compatible mode: a GPU device without subgroups, so ONNX Runtime uses its plain kernels (reductions through
 * workgroup memory). Everything else the adapter offers is kept: other features and its largest limits.
 */
let safeMode = false;
let safeDevice: GPUDevice | null = null;
async function webgpuProvider(): Promise<Any> {
  if (!safeDevice) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('WebGPU: sin adaptador');
    const requiredFeatures = [...adapter.features].filter((f) => !/subgroup/.test(f)) as GPUFeatureName[];
    const requiredLimits: Record<string, number> = {};
    for (const k in adapter.limits) {
      const v = (adapter.limits as Any)[k];
      if (typeof v === 'number') requiredLimits[k] = v;
    }
    safeDevice = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  }
  return { name: 'webgpu', device: safeDevice };
}

async function unload() {
  await resetCaches();
  forgetImages();
  if (model) await model.dispose();
  model = tokenizer = info = null;
}

function template(messages: ChatMessage[], generationPrompt: boolean, reasoning: boolean): number[] {
  const msgs = messages.map((m) =>
    m.role === 'assistant' && m.reasoning
      ? { role: m.role, content: m.content, reasoning_content: m.reasoning }
      : m.images?.length
        ? { role: m.role, content: [...m.images.map(() => ({ type: 'image' })), { type: 'text', text: m.content }] }
        : { role: m.role, content: m.content },
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

/** The pictures of a conversation, in prompt order. */
const imagesOf = (messages: ChatMessage[]): ImageRef[] => messages.flatMap((m) => (m.role === 'user' ? (m.images ?? []) : []));

/** Prompt tokens with each picture's placeholder expanded to its image tokens. */
async function promptIds(messages: ChatMessage[], generationPrompt: boolean, reasoning: boolean): Promise<number[]> {
  const ids = template(messages, generationPrompt, reasoning);
  const images = imagesOf(messages);
  if (!images.length) return ids;
  const m = info!;
  if (!m.vision) throw new Error('NO_VISION');
  const counts: number[] = [];
  for (const img of images) counts.push((await processImage(m, img)).tokens);
  return expandImageTokens(ids, counts, m.vision, model.config);
}

/** The image encoder, downloaded and started the first time a picture is read. */
async function ensureVision(id: number) {
  const m = info!;
  const v = m.vision!;
  if (model.sessions.vision_encoder) return;
  const tStart = performance.now();
  dbg(`lector de imágenes: preparar ${v.data.path} (${(v.data.bytes / 1e6).toFixed(0)} MB)`);
  post({ type: 'vision', id, loaded: 0, total: v.data.bytes });
  const previous = onNetProgress;
  onNetProgress = (_file, loaded, total) => post({ type: 'vision', id, loaded, total: total || v.data.bytes });
  try {
    const [graph, data] = await Promise.all([fetchBytes(m, v.graph.path), fetchBytes(m, v.data.path)]);
    const ortModule: Any = loadedDevice === 'cpu' ? await import(/* @vite-ignore */ NODE_ORT) : await import('onnxruntime-web/webgpu');
    model.sessions.vision_encoder = await ortModule.InferenceSession.create(graph, {
      executionProviders: [loadedDevice === 'cpu' ? 'cpu' : safeMode ? await webgpuProvider() : 'webgpu'],
      externalData: [{ path: v.data.name, data }],
    });
  } finally {
    onNetProgress = previous;
  }
  post({ type: 'vision', id, loaded: v.data.bytes, total: v.data.bytes });
  dbg(`lector de imágenes listo en ${Math.round(performance.now() - tStart)} ms`);
}
const NODE_ORT = 'onnxruntime-node';
/** 'webgpu' in the browser; 'cpu' in the Node tests. */
let loadedDevice = 'webgpu';

/** A model file's bytes, through the pinned, cached download path (small files are stored here too). */
async function fetchBytes(m: ModelInfo, path: string): Promise<Uint8Array> {
  const url = fileUrl(m, path);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  if ('caches' in globalThis && !/\.onnx(_data(_\d+)?|\.data)$/.test(path)) {
    try {
      const cache = await caches.open('transformers-cache');
      if (!(await cache.match(url))) await cache.put(url, new Response(bytes, { headers: { 'content-length': String(bytes.length) } }));
    } catch {
      /* not stored: downloaded again next time */
    }
  }
  return bytes;
}

async function prefill(state: CacheState | null, ids: number[], chunkSize = 0, extra: Record<string, Any> = {}, images: string[] = []): Promise<CacheState> {
  if (chunkSize && ids.length - (state?.tokens.length ?? 0) > chunkSize) {
    let next = state;
    try {
      while ((next?.tokens.length ?? 0) < ids.length) {
        if (stopper.interrupted) throw new Error('Prefill interrupted');
        const end = Math.min(ids.length, (next?.tokens.length ?? 0) + chunkSize);
        const previous = next;
        next = await prefill(previous, ids.slice(0, end), 0, extra, images);
        if (previous && previous !== state) await disposeState(previous, next);
      }
      return next!;
    } catch (e) {
      if (next && next !== state) await disposeState(next, state);
      throw e;
    }
  }
  // Run the model over `ids` (continuing `state` if given) and return the cache that covers them.
  const input_ids = new Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
  const pkv = state ? copyCache(state.cache) : undefined;
  const out = await model.generate({
    input_ids,
    attention_mask: new Tensor('int64', new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
    past_key_values: pkv,
    ...extra,
    max_new_tokens: 1,
    do_sample: false,
    return_dict_in_generate: true,
  });
  return { cache: out.past_key_values, tokens: ids.slice(), images };
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
  const start = performance.now();
  stopper.reset();
  if (o.chunkedPrefill && chunkValidated === null) await validateChunkedPrefill();
  const validationMs = performance.now() - start;
  if (stopper.interrupted) throw new Error('Prefill interrupted');
  const templateStart = performance.now();
  const full = await promptIds(messages, true, o.reasoning);
  const hist = await promptIds(messages, false, o.reasoning);
  const boundary = isPrefix(hist, full) && hist.length < full.length ? hist.length : 0;
  const pictures = imagesOf(messages);
  const pictureIds = pictures.map((p) => p.id);
  const imageToken = Number(model.config.image_token_id);

  // Longest cached prefix of this prompt (strictly shorter, so at least one token is fed). Image tokens are the
  // same for every picture, so a cache also has to hold the same pictures.
  const samePictures = (s: CacheState) => s.images.every((x, i) => x === pictureIds[i]);
  const candidates = [live, snap].filter((s): s is CacheState => !!s && s.tokens.length < full.length && isPrefix(s.tokens, full) && samePictures(s));
  candidates.sort((a, b) => b.tokens.length - a.tokens.length);
  let base: CacheState | null = candidates[0] ?? null;
  // Transformers.js reads pictures only in a pass that starts from the beginning: a turn that adds a picture
  // is read again from the start (later turns reuse the cache as usual).
  const readsPictures = pictures.length > 0 && full.slice(base?.tokens.length ?? 0).includes(imageToken);
  // Gemma: the graph's masked attention, with both directions inside each picture, for the pass that reads them.
  pictureBlocks = readsPictures && m.vision?.kind === 'gemma' ? pictureBlocksOf(full, imageToken) : null;
  if (readsPictures) {
    base = null;
    await ensureVision(id);
  }
  const processedPictures = await Promise.all(pictures.map((p) => processImage(m, p)));
  const firstPass = m.vision ? visionInputs(m.vision, processedPictures, readsPictures) : {};
  const laterPass = m.vision ? visionInputs(m.vision, processedPictures, false) : {};
  const t0 = performance.now();
  const cached = base?.tokens.length ?? 0;
  const templateMs = t0 - templateStart;
  const snapshotStart = performance.now();

  // Snapshot at the end of the history (before the generation prompt), reused by the next turn
  // even though chat templates rewrite earlier answers (e.g. dropping their reasoning).
  let newSnap: CacheState | null = null;
  if (boundary > cached) {
    newSnap = await prefill(base, full.slice(0, boundary), readsPictures ? 0 : o.chunkedPrefill && chunkValidated ? 128 : 0, firstPass, pictureIds);
    base = newSnap;
  } else if (base === snap && snap) {
    newSnap = snap;
  }
  if (newSnap) for (const t of entries(newSnap.cache)) protectedTensors.add(t);
  // Free what is no longer reachable (tensors consumed by generation are already released).
  for (const s of [live, snap]) if (s && s !== base && s !== newSnap) await disposeState(s, base);
  live = null;
  snap = newSnap;

  const snapshotMs = performance.now() - snapshotStart;
  let streamMs = 0;
  let streamUpdates = 0;
  const prefillEnd = { t: 0 };
  let lastPost = 0;
  // Thinking budget: small models can reason until the token limit and never answer. When the reasoning is
  // still open after this many tokens, it is closed and the model answers with the tokens that are left.
  const reserve = Math.min(512, Math.max(128, Math.round(o.maxTokens * 0.35)));
  let budgetAt = o.reasoning && o.maxTokens > reserve * 1.5 ? o.maxTokens - reserve : 0;
  const budget = new InterruptableStoppingCriteria();
  const onIds = (ids: number[]) => {
    if (!prefillEnd.t) prefillEnd.t = performance.now();
    if (budgetAt && ids.length >= budgetAt && !budget.interrupted) {
      const open = splitReasoning(tokenizer.decode(ids, { skip_special_tokens: false }), m.think, o.reasoning, specials()).thinking;
      if (open) budget.interrupt();
      budgetAt = 0;
    }
    const now = performance.now();
    if (now - lastPost < 40 && ids.length > 1) return; // ~25 updates/s is plenty
    lastPost = now;
    const streamStart = performance.now();
    const text = tokenizer.decode(ids, { skip_special_tokens: false });
    const r = splitReasoning(text, m.think, o.reasoning, specials());
    post({ type: 'delta', id, content: r.content, reasoning: r.reasoning, thinking: r.thinking, tokens: ids.length });
    streamMs += performance.now() - streamStart;
    streamUpdates++;
  };
  const sampling = {
    do_sample: o.temperature > 0,
    temperature: Math.max(o.temperature, 1e-3),
    top_p: o.topP,
    top_k: o.topK,
    repetition_penalty: o.repetitionPenalty,
    return_dict_in_generate: true,
  };
  const ids64 = (ids: number[]) => ({
    input_ids: new Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    attention_mask: new Tensor('int64', new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
  });

  const criteria = new StoppingCriteriaList();
  criteria.push(stopper);
  criteria.push(budget);
  let out = await model.generate({
    ...ids64(full),
    ...sampling,
    past_key_values: base ? copyCache(base.cache) : undefined,
    ...(readsPictures && !(boundary > 0) ? firstPass : laterPass),
    max_new_tokens: o.maxTokens,
    streamer: new DeltaStreamer(onIds),
    stopping_criteria: criteria,
  });
  if (budget.interrupted && !stopper.interrupted) {
    // Close the reasoning and let the model answer, continuing from the cache it just built.
    const thought: number[] = Array.from(out.sequences.data as ArrayLike<bigint>, Number).slice(full.length);
    const close: number[] = Array.from(tokenizer.encode(`\n${m.think.close}\n\n`, { add_special_tokens: false }) as ArrayLike<number>, Number);
    const head = thought.concat(close);
    const rest = new StoppingCriteriaList();
    rest.push(stopper);
    out = await model.generate({
      ...ids64(full.concat(head)),
      ...sampling,
      past_key_values: out.past_key_values,
      ...laterPass,
      max_new_tokens: Math.max(64, o.maxTokens - head.length),
      streamer: new DeltaStreamer((ids) => onIds(head.concat(ids))),
      stopping_criteria: rest,
    });
  }
  const end = performance.now();
  const seq: number[] = Array.from(out.sequences.data as ArrayLike<bigint>, Number);
  const gen = seq.slice(full.length);
  // The cache holds the prompt and every generated token except the last one (never fed back).
  live = { cache: out.past_key_values, tokens: full.concat(gen.slice(0, -1)), images: pictureIds };
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
      templateMs, snapshotMs, streamMs, streamUpdates, validationMs,
      chunkedPrefill: Boolean(o.chunkedPrefill && chunkValidated),
      totalMs: end - start,
      promptTokens: full.length,
      cachedTokens: cached,
      newTokens: gen.length,
      prefillMs: pf,
      decodeMs,
      tokensPerSecond: gen.length > 1 && decodeMs > 0 ? ((gen.length - 1) * 1000) / decodeMs : 0,
      stopped: stopper.interrupted ? 'user' : eos.has(gen.at(-1) ?? -1) ? 'eos' : 'length',
    },
  });
  if (profiling) reportProfile();
}

// ---------------------------------------------------------------------------------------------
// Gemma 4 pictures: two-way attention inside each picture.
//
// Gemma 4 12B reads a picture's patches inside the language model and lets them attend to each other in both
// directions. Its graph (tools/gemma4/bidir.py) has, per attention layer, If(use_bias): a masked attention fed
// with the masks built here, or the fused causal GQA. The extra inputs are hidden from Transformers.js (which
// does not know them) and added to every run of the decoder: the mask only for a pass that starts from the
// beginning and reads pictures; every other step runs the fused attention.

/** Picture block of each prompt position (-1 for text) for the pass that reads pictures, else null. */
let pictureBlocks: Int32Array | null = null;
const LOCAL_WINDOW = 1024; // sliding window of Gemma 4's local layers
const F16_ZERO = 0x0000;
const F16_MIN = 0xfbff; // -65504: masked

function pictureBlocksOf(ids: number[], imageToken: number): Int32Array {
  const out = new Int32Array(ids.length).fill(-1);
  let block = -1;
  for (let i = 0; i < ids.length; i++) {
    if (ids[i] === imageToken) {
      if (i === 0 || ids[i - 1] !== imageToken) block++;
      out[i] = block;
    }
  }
  return out;
}

function maskedAttention(session: Any) {
  const extra = ['use_bias', 'attention_bias_global', 'attention_bias_local'];
  const names: string[] = session?.inputNames ?? [];
  if (!extra.every((n) => names.includes(n))) return;
  Object.defineProperty(session, 'inputNames', { value: names.filter((n) => !extra.includes(n)), configurable: true });
  const run = session.run.bind(session);
  session.run = (feeds: Any, ...rest: Any[]) => {
    const Tensor = feeds.attention_mask.constructor;
    const S = Number(feeds.inputs_embeds?.dims?.[1] ?? 1);
    const T = Number(feeds.attention_mask.dims[1]);
    const blocks = pictureBlocks;
    if (blocks && S > 1 && S === T && S <= blocks.length && blocks.subarray(0, S).some((b) => b >= 0)) {
      const global = new Uint16Array(S * S);
      const local = new Uint16Array(S * S);
      for (let i = 0; i < S; i++) {
        const bi = blocks[i]!;
        for (let j = 0; j < S; j++) {
          const sameBlock = bi >= 0 && blocks[j] === bi;
          const causal = j <= i;
          global[i * S + j] = causal || sameBlock ? F16_ZERO : F16_MIN;
          local[i * S + j] = (causal && i - j < LOCAL_WINDOW) || sameBlock ? F16_ZERO : F16_MIN;
        }
      }
      const pictures = new Set(Array.from(blocks.subarray(0, S)).filter((x) => x >= 0)).size;
      dbg(`atención en ambos sentidos dentro de ${pictures} ${pictures === 1 ? 'imagen' : 'imágenes'} (${S} tokens en esta pasada)`);
      return run({ ...feeds, use_bias: new Tensor('bool', new Uint8Array([1]), []), attention_bias_global: new Tensor('float16', global, [1, 1, S, S]), attention_bias_local: new Tensor('float16', local, [1, 1, S, S]) }, ...rest);
    }
    const none = new Tensor('float16', new Uint16Array(1), [1, 1, 1, 1]);
    return run({ ...feeds, use_bias: new Tensor('bool', new Uint8Array([0]), []), attention_bias_global: none, attention_bias_local: none }, ...rest);
  };
}

/** Set when the model was loaded with ONNX Runtime's profiler on: the first answer ends it and reports. */
let profiling = false;
function reportProfile() {
  profiling = false;
  for (const [name, session] of Object.entries((model?.sessions ?? {}) as Record<string, Any>)) {
    const p = new OrtProfile();
    tapConsole((line) => p.line(line));
    try {
      session.endProfiling?.();
    } catch (e) {
      dbg(`perfil ${name}: ${errText(e)}`);
    } finally {
      tapConsole(null);
    }
    dbg(p.summary(name).join('\n'));
  }
}

/** Test equivalence in this model/device session before allowing experimental chunked prefill. */
async function validateChunkedPrefill() {
  await resetCaches();
  const ids = template([{ role: 'user', content: 'Read this text and summarize it briefly: ' + 'The river crosses a village with gardens and a stone bridge. '.repeat(8) }], true, false);
  let state: CacheState | null = null;
  const run = async (chunked: boolean) => {
    if (chunked) state = await prefill(null, ids.slice(0, -1), 32);
    const out = await model.generate({
      input_ids: new Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
      attention_mask: new Tensor('int64', new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
      ...(state ? { past_key_values: copyCache(state.cache) } : {}),
      max_new_tokens: 8, min_new_tokens: 8, do_sample: false, stopping_criteria: [stopper], return_dict_in_generate: true,
    });
    const tokens = Array.from(out.sequences.data as ArrayLike<bigint>, String).slice(ids.length);
    await disposeState(state, { cache: out.past_key_values, tokens: [], images: [] });
    state = null;
    await disposeState({ cache: out.past_key_values, tokens: [], images: [] });
    return tokens;
  };
  try {
    const baseline = await run(false);
    const chunked = await run(true);
    chunkValidated = baseline.length === 8 && JSON.stringify(baseline) === JSON.stringify(chunked);
    dbg(`prefill por bloques: prueba local de equivalencia ${chunkValidated ? 'superada; bloques de 128 habilitados bajo petición' : 'fallida; se conserva prefill completo'}. Prueba corta, no garantiza equivalencia universal.`);
  } catch (e) {
    chunkValidated = false;
    dbg(`prefill por bloques no disponible: ${errText(e)}`);
  } finally { await disposeState(state); }
}

/** Two warmed runs per mode; comparison is end-to-end, not an isolated sampler microbenchmark. */
async function bench(id: number, compare = false, profile = false) {
  if (!model) throw new Error('No model loaded');
  pictureBlocks = null;
  await resetCaches();
  stopper.reset();
  const ids = template([{ role: 'user', content: 'Write a detailed paragraph about the history of the Internet.' }], true, false);
  const aggregate = new GpuProfile();
  const previousProfile = onnx.webgpu?.profiling;
  if (profile && onnx.webgpu) onnx.webgpu.profiling = { mode: 'default', ondata: (d: Any) => aggregate.add(d) };
  const results: Any[] = [];
  try {
    for (const sampling of compare ? [false, true] : [false]) {
      let result: Any;
      for (let run = 0; run < 2; run++) {
        let first = 0;
        const streamer = new DeltaStreamer(() => { if (!first) first = performance.now(); });
        const t0 = performance.now();
        const out = await model.generate({
          input_ids: new Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
          attention_mask: new Tensor('int64', new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
          max_new_tokens: 128, min_new_tokens: 128,
          do_sample: sampling, temperature: 0.7, top_p: 0.9, top_k: 40, repetition_penalty: 1,
          streamer, stopping_criteria: [stopper], return_dict_in_generate: true,
        });
        const end = performance.now();
        await disposeState({ cache: out.past_key_values, tokens: [], images: [] });
        if (stopper.interrupted) throw new Error('Benchmark interrupted');
        const n = Number(out.sequences.dims.at(-1)) - ids.length;
        const prefillMs = (first || end) - t0;
        const decodeMs = end - (first || end);
        result = { promptTokens: ids.length, newTokens: n, prefillMs, decodeMs, tokensPerSecond: decodeMs > 0 ? ((n - 1) * 1000) / decodeMs : 0, prefillTokensPerSecond: prefillMs > 0 ? ids.length * 1000 / prefillMs : 0, runs: run + 1 };
      }
      results.push(result);
    }
    const result = { ...results[0], profiled: profile, ...(compare ? { comparison: { greedy: results[0], sampling: results[1] } } : {}) };
    dbg(`benchmark: ${JSON.stringify(result)}; mismo prompt y longitud; sampling temperature=0.7 top_p=0.9 top_k=40 repetition_penalty=1; el muestreo cambia la secuencia; no mide utilización del hardware`);
    if (profile) {
      // ORT delivers timestamp readbacks asynchronously; report the samples received so far.
      dbg(`perfil GPU (muestras recibidas, puede ser parcial; incluye calentamiento; instrumentación altera tiempos): ${JSON.stringify(aggregate.summary())}`);
      if (!aggregate.summary().samples) dbg('Sin muestras GPU: timestamp-query puede no estar disponible o la lectura seguir pendiente. No significa coste cero.');
    }
    post({ type: 'bench', id, result });
  } finally {
    if (profile && onnx.webgpu) onnx.webgpu.profiling = previousProfile ?? { mode: 'off' };
  }
}

let specialCache: string[] | null = null;
let chunkValidated: boolean | null = null;
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
        await load(msg.model, msg.dtype, msg.device, msg.profile);
      } else if (msg.type === 'generate') {
        await generate(msg.id, msg.messages, msg.options);
      } else if (msg.type === 'mmbench') {
        await matmulBench((line) => dbg(line));
        post({ type: 'bench', id: msg.id, result: { promptTokens: 0, newTokens: 0 } as Any });
      } else if (msg.type === 'bench') {
        await bench(msg.id, msg.compare, msg.profile);
      } else if (msg.type === 'count') {
        // Some templates (Qwen) refuse a conversation without a user message, e.g. an empty chat with a system
        // prompt: count it as if an empty user message were there, which is what the answer will add anyway.
        const hasUser = msg.messages.some((m) => m.role === 'user');
        const withUser: ChatMessage[] = hasUser ? msg.messages : [...msg.messages, { role: 'user', content: ' ' }];
        // Pictures count as the image tokens they take (a model that cannot read them ignores them here).
        const counted = info?.vision ? withUser : withUser.map((m) => ({ ...m, images: undefined }));
        post({ type: 'count', id: msg.id, tokens: tokenizer ? (await promptIds(counted, true, msg.reasoning)).length : 0 });
      } else if (msg.type === 'reset') {
        await resetCaches();
      } else if (msg.type === 'unload') {
        await unload();
        post({ type: 'unloaded' });
      }
    } catch (err) {
      console.error(err);
      dbg(`ERROR en ${msg.type}: ${errText(err)}`);
      const message = err instanceof Error ? err.message : String(err);
      post({ type: 'error', id: 'id' in msg ? msg.id : undefined, message, stage: msg.type === 'load' ? 'load' : 'generate' });
      if (msg.type === 'generate') await resetCaches().catch(() => {});
    }
  });
};
