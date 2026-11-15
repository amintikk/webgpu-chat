import type { ModelId } from './models.ts';

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  /** Reasoning text produced before the answer, if any. */
  reasoning?: string;
  /** Pictures attached to a user message (models that read images). */
  images?: ImageRef[];
}

/** A picture in the conversation: `id` stays the same across turns, so it is processed once. */
export interface ImageRef {
  id: string;
  blob: Blob;
}

export interface GenOptions {
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  maxTokens: number;
  reasoning: boolean;
  /** Opt-in, validated against unchunked greedy decoding before first use. */
  chunkedPrefill?: boolean;
}

export interface GenStats {
  /** CPU wall times; snapshotMs includes model execution, not exclusive GPU time. */
  validationMs?: number;
  chunkedPrefill?: boolean;
  templateMs?: number;
  snapshotMs?: number;
  streamMs?: number;
  streamUpdates?: number;
  totalMs?: number;
  promptTokens: number;
  /** Prompt tokens served from the cache of earlier turns. */
  cachedTokens: number;
  newTokens: number;
  prefillMs: number;
  decodeMs: number;
  tokensPerSecond: number;
  stopped: 'eos' | 'length' | 'user';
}

export interface BenchResult {
  comparison?: { greedy: BenchResult; sampling: BenchResult };
  profiled?: boolean;
  promptTokens: number;
  newTokens: number;
  /** Time to first token for the prompt (prefill), ms. */
  prefillMs: number;
  decodeMs: number;
  tokensPerSecond: number;
  prefillTokensPerSecond: number;
  runs: number;
}

export type ToWorker =
  | { type: 'load'; model: ModelId; lang: 'es' | 'en'; dtype?: string; /** Tests only (Node): 'cpu'. */ device?: string; /** ONNX Runtime's profiler on; the first answer reports it. */ profile?: boolean }
  | { type: 'generate'; id: number; messages: ChatMessage[]; options: GenOptions }
  | { type: 'count'; id: number; messages: ChatMessage[]; reasoning: boolean }
  | { type: 'bench'; id: number; compare?: boolean; profile?: boolean }
  | { type: 'mmbench'; id: number }
  | { type: 'stop' }
  | { type: 'reset' }
  | { type: 'unload' };

export type FromWorker =
  | { type: 'progress'; loaded: number; total: number; file: string }
  | { type: 'status'; stage: 'download' | 'compile' | 'warmup' }
  | { type: 'ready'; model: ModelId; loadMs: number }
  | { type: 'delta'; id: number; content: string; reasoning: string; thinking: boolean; tokens: number }
  /** The image encoder is being downloaded or prepared (first picture sent to this model). */
  | { type: 'vision'; id: number; loaded: number; total: number }
  | { type: 'done'; id: number; content: string; reasoning: string; stats: GenStats }
  | { type: 'count'; id: number; tokens: number }
  | { type: 'bench'; id: number; result: BenchResult }
  | { type: 'error'; id?: number; message: string; stage: 'load' | 'generate' }
  | { type: 'unloaded' }
  /** Debug mode only: one line for the page's record. */
  | { type: 'log'; text: string };
