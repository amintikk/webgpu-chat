// Page-side handle on the inference worker.
import type { ModelId } from './models.ts';
import type { BenchResult, ChatMessage, FromWorker, GenOptions, GenStats, ToWorker } from './protocol.ts';
import { lang } from '../i18n.ts';

export interface LoadEvents {
  onProgress?: (loaded: number, total: number) => void;
  onStage?: (stage: 'download' | 'compile' | 'warmup') => void;
}

export interface GenEvents {
  onDelta: (d: { content: string; reasoning: string; thinking: boolean; tokens: number }) => void;
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
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<FromWorker>) => this.handle(e.data);
    this.worker.onerror = (e) => {
      const err = new Error(e.message || 'Worker error');
      this.loadWaiter?.reject(err);
      this.loadWaiter = null;
      for (const g of this.gens.values()) g.reject(err);
      this.gens.clear();
    };
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
      case 'done': {
        const g = this.gens.get(m.id);
        this.gens.delete(m.id);
        this.busy = false;
        g?.resolve({ content: m.content, reasoning: m.reasoning, stats: m.stats });
        break;
      }
      case 'bench':
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
    }
  }

  load(model: ModelId, ev: LoadEvents = {}, dtype?: string): Promise<number> {
    this.loadWaiter?.reject(new Error('superseded'));
    return new Promise((resolve, reject) => {
      this.loadWaiter = { resolve, reject, ev };
      this.send({ type: 'load', model, lang, dtype });
    });
  }

  generate(messages: ChatMessage[], options: GenOptions, ev: GenEvents): Promise<GenResult> {
    const id = ++this.seq;
    this.busy = true;
    return new Promise((resolve, reject) => {
      this.gens.set(id, { resolve, reject, ev });
      this.send({ type: 'generate', id, messages, options });
    });
  }

  count(messages: ChatMessage[], reasoning: boolean): Promise<number> {
    const id = ++this.seq;
    return new Promise((resolve) => {
      this.counts.set(id, resolve);
      this.send({ type: 'count', id, messages, reasoning });
    });
  }

  bench(): Promise<BenchResult> {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.benches.set(id, { resolve, reject });
      this.send({ type: 'bench', id });
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
