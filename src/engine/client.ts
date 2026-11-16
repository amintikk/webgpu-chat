// Page-side handle on the inference worker.
import type { ModelId } from './models.ts';
import type { BenchResult, ChatMessage, FromWorker, GenOptions, GenStats, GpuTest, ToWorker } from './protocol.ts';
import { lang } from '../i18n.ts';
import { dlog } from '../ui/debug.ts';

export interface LoadEvents {
  onProgress?: (loaded: number, total: number) => void;
  onStage?: (stage: 'download' | 'compile' | 'warmup') => void;
}

export interface GenEvents {
  onDelta: (d: { content: string; reasoning: string; thinking: boolean; tokens: number }) => void;
  /** The image encoder is downloading or starting (first picture for this model). */
  onVision?: (loaded: number, total: number) => void;
}

export interface GenResult {
  content: string;
  reasoning: string;
  stats: GenStats;
}

export class Engine {
  private worker: Worker;
  private seq = 0;
  private loadWaiter: { resolve: (ms: number) => void; reject: (e: Error) => void; ev: LoadEvents } | null = null;
  private gens = new Map<number, { resolve: (r: GenResult) => void; reject: (e: Error) => void; ev: GenEvents }>();
  private counts = new Map<number, (n: number) => void>();
  private benches = new Map<number, { resolve: (r: BenchResult) => void; reject: (e: Error) => void }>();
  loaded: ModelId | null = null;
  busy = false;

  constructor() {
    this.worker = this.spawn();
  }

  private spawn(): Worker {
    const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    w.onmessage = (e: MessageEvent<FromWorker>) => this.handle(e.data);
    w.onerror = (e) => {
      const err = new Error(e.message || 'Worker error');
      this.loadWaiter?.reject(err);
      this.loadWaiter = null;
      for (const g of this.gens.values()) g.reject(err);
      this.gens.clear();
      for (const b of this.benches.values()) b.reject(err);
      this.benches.clear();
      this.busy = false;
      this.loaded = null;
    };
    return w;
  }

  /**
   * Stops the worker and starts a new one. Ending a worker releases everything it held, including GPU memory
   * left by a load that failed part-way (the next model would otherwise run out of memory).
   */
  restart(): void {
    this.worker.terminate();
    this.loaded = null;
    this.busy = false;
    const err = new Error('restarted');
    this.loadWaiter?.reject(err);
    this.loadWaiter = null;
    for (const g of this.gens.values()) g.reject(err);
    this.gens.clear();
    for (const b of this.benches.values()) b.reject(err);
    this.benches.clear();
    this.counts.clear();
    this.worker = this.spawn();
  }

  private send(m: ToWorker) {
    this.worker.postMessage(m);
  }

  private handle(m: FromWorker) {
    switch (m.type) {
      case 'progress':
        this.loadWaiter?.ev.onProgress?.(m.loaded, m.total);
        break;
      case 'status':
        this.loadWaiter?.ev.onStage?.(m.stage);
        break;
      case 'ready':
        this.loaded = m.model;
        this.loadWaiter?.resolve(m.loadMs);
        this.loadWaiter = null;
        break;
      case 'delta':
        this.gens.get(m.id)?.ev.onDelta(m);
        break;
      case 'vision':
        this.gens.get(m.id)?.ev.onVision?.(m.loaded, m.total);
        break;
      case 'done': {
        const g = this.gens.get(m.id);
        this.gens.delete(m.id);
        this.busy = false;
        g?.resolve({ content: m.content, reasoning: m.reasoning, stats: m.stats });
        break;
      }
      case 'bench':
        this.busy = false;
        this.benches.get(m.id)?.resolve(m.result);
        this.benches.delete(m.id);
        break;
      case 'count':
        this.counts.get(m.id)?.(m.tokens);
        this.counts.delete(m.id);
        break;
      case 'error': {
        const err = new Error(m.message);
        if (m.stage === 'load') {
          this.loaded = null;
          this.loadWaiter?.reject(err);
          this.loadWaiter = null;
        } else if (m.id !== undefined && this.benches.has(m.id)) {
          this.busy = false;
          this.benches.get(m.id)!.reject(err);
          this.benches.delete(m.id);
        } else if (m.id !== undefined) {
          this.busy = false;
          this.gens.get(m.id)?.reject(err);
          this.gens.delete(m.id);
        }
        break;
      }
      case 'unloaded':
        this.loaded = null;
        break;
      case 'log':
        dlog(`[worker] ${m.text}`);
        break;
    }
  }

  load(model: ModelId, ev: LoadEvents = {}, dtype?: string, profile = false, gpu: GpuTest = 'auto'): Promise<number> {
    this.loadWaiter?.reject(new Error('superseded'));
    return new Promise((resolve, reject) => {
      this.loadWaiter = { resolve, reject, ev };
      this.send({ type: 'load', model, lang, dtype, profile, gpu });
    });
  }

  generate(messages: ChatMessage[], options: GenOptions, ev: GenEvents): Promise<GenResult> {
    if (this.busy) return Promise.reject(new Error('Engine busy'));
    const id = ++this.seq;
    // Chunked prefill stays off: its self-check cost 30-160 s before the first answer and failed for Qwen3.5.
    const chunkedPrefill = false;
    try { localStorage.removeItem('webgpu-chat:chunked-prefill'); } catch { /* unavailable storage */ }
    this.busy = true;
    return new Promise((resolve, reject) => {
      this.gens.set(id, { resolve, reject, ev });
      this.send({ type: 'generate', id, messages, options: { ...options, chunkedPrefill } });
    });
  }

  count(messages: ChatMessage[], reasoning: boolean): Promise<number> {
    const id = ++this.seq;
    return new Promise((resolve) => {
      this.counts.set(id, resolve);
      this.send({ type: 'count', id, messages, reasoning });
    });
  }

  bench(compare = false, profile = false): Promise<BenchResult> {
    if (this.busy) return Promise.reject(new Error('Engine busy'));
    this.busy = true;
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.benches.set(id, { resolve, reject });
      this.send({ type: 'bench', id, compare, profile });
    });
  }

  /** GPU cost of one 4-bit matmul for 1-32 tokens at once; the numbers go to the debug record. */
  matmulBench(): Promise<BenchResult> {
    if (this.busy) return Promise.reject(new Error('Engine busy'));
    this.busy = true;
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.benches.set(id, { resolve, reject });
      this.send({ type: 'mmbench', id });
    });
  }

  stop() {
    this.send({ type: 'stop' });
  }

  /** Forget the cached conversation (switching conversations). */
  reset() {
    this.send({ type: 'reset' });
  }

  unload() {
    this.send({ type: 'unload' });
    this.loaded = null;
  }
}
