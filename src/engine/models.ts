import type { ThinkMarkers } from './reasoning.ts';

// Model catalogue. Files are downloaded straight from Hugging Face at a pinned commit, so what runs
// in the browser is always the exact revision that was measured (see results/ in the repository).

export type ModelId = 'lfm3-1.7b' | 'qwen3.5-4b' | 'gemma4-12b';
export type ModelClass = 'Lfm3ForCausalLM' | 'Qwen3_5ForCausalLM' | 'Gemma4ForCausalLM';

export interface ModelFile {
  path: string;
  bytes: number;
}

export interface ModelInfo {
  id: ModelId;
  name: string;
  tier: 'light' | 'recommended' | 'max';
  repo: string;
  revision: string;
  cls: ModelClass;
  /** Transformers.js dtype option: one string, or one per session for multi-session models. */
  dtype: string | Record<string, string>;
  files: ModelFile[];
  /** Largest single weight tensor (bytes): the adapter must allow a buffer at least this big. */
  largestBuffer: number;
  /** Rough GPU memory needed at the context limit (weights + cache + activations), bytes. */
  gpuMemory: number;
  /** Context window offered in the app (tokens). The models support more; the cap keeps memory bounded. */
  context: number;
  /** Generation defaults recommended by the model authors. */
  defaults: { temperature: number; topP: number; topK: number; repetitionPenalty: number; reasoning: boolean };
  /** Reasoning markers as they appear in the decoded output. */
  think: ThinkMarkers;
  license: { name: string; url: string };
  source: string;
  params: string;
}

export const MODELS: ModelInfo[] = [
  {
    id: 'lfm3-1.7b',
    name: 'LFM3 1.7B',
    tier: 'light',
    repo: 'onnx-community/LFM3-1.7B-ONNX',
    revision: 'ab9fec4c681a25bec3b48b0b97a62fc2660a2985',
    cls: 'Lfm3ForCausalLM',
    dtype: 'q4f16',
    files: [
      { path: 'config.json', bytes: 1_549 },
      { path: 'generation_config.json', bytes: 241 },
      { path: 'tokenizer.json', bytes: 17_908_635 },
      { path: 'tokenizer_config.json', bytes: 14_262 },
      { path: 'onnx/model_q4f16.onnx', bytes: 296_714 },
      { path: 'onnx/model_q4f16.onnx_data', bytes: 1_087_789_056 },
    ],
    largestBuffer: 131_072_000,
    gpuMemory: 1.6e9,
    context: 8192,
    defaults: { temperature: 0.3, topP: 0.95, topK: 50, repetitionPenalty: 1.05, reasoning: false },
    think: { open: '<think>', close: '</think>', promptOpens: true },
    license: { name: 'LFM Open License v1.0', url: 'https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE' },
    source: 'https://huggingface.co/LiquidAI/LFM3-1.7B',
    params: '1,7B',
  },
  {
    id: 'qwen3.5-4b',
    name: 'Qwen3.5 4B',
    tier: 'recommended',
    // The -OPT export fuses the linear-attention layers (LinearAttention, CausalConvWithState) and computes logits
    // only for the last position. The original repo ran each prompt position in a loop driven by the CPU (Scan inside
    // If), which WebGPU cannot run on its own: about 14 s before the first word for a short prompt.
    repo: 'onnx-community/Qwen3.5-4B-ONNX-OPT',
    revision: 'bb072268d0912bdc5cf4f41c8c6f53a03fd68624',
    cls: 'Qwen3_5ForCausalLM',
    dtype: { embed_tokens: 'q4f16', decoder_model_merged: 'q4f16' },
    files: [
      { path: 'config.json', bytes: 3_198 },
      { path: 'generation_config.json', bytes: 248 },
      { path: 'tokenizer.json', bytes: 19_226_111 },
      { path: 'tokenizer_config.json', bytes: 10_070 },
      { path: 'onnx/embed_tokens_q4f16.onnx', bytes: 1_064 },
      { path: 'onnx/embed_tokens_q4f16.onnx_data', bytes: 367_513_600 },
      { path: 'onnx/decoder_model_merged_q4f16.onnx', bytes: 933_554 },
      { path: 'onnx/decoder_model_merged_q4f16.onnx_data', bytes: 2_065_635_328 },
      { path: 'onnx/decoder_model_merged_q4f16.onnx_data_1', bytes: 367_513_600 },
    ],
    largestBuffer: 317_849_600,
    gpuMemory: 3.6e9,
    context: 8192,
    defaults: { temperature: 0.7, topP: 0.8, topK: 20, repetitionPenalty: 1.0, reasoning: false },
    think: { open: '<think>', close: '</think>', promptOpens: true },
    license: { name: 'Apache 2.0', url: 'https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE' },
    source: 'https://huggingface.co/Qwen/Qwen3.5-4B',
    params: '4B',
  },
  {
    id: 'gemma4-12b',
    name: 'Gemma 4 12B',
    tier: 'max',
    repo: 'AMINOXDEV/gemma-4-12B-it-ONNX',
    revision: '90e0a27bf539717b6055d7e434b9faa1f0dbd9ff',
    cls: 'Gemma4ForCausalLM',
    dtype: { embed_tokens: 'q4f16', decoder_model_merged: 'q4f16' },
    files: [
      { path: 'config.json', bytes: 3_555 },
      { path: 'generation_config.json', bytes: 260 },
      { path: 'tokenizer.json', bytes: 32_157_161 },
      { path: 'tokenizer_config.json', bytes: 22_222 },
      { path: 'onnx/embed_tokens_q4f16.onnx', bytes: 856 },
      { path: 'onnx/embed_tokens_q4f16.onnx_data', bytes: 581_959_680 },
      { path: 'onnx/decoder_model_merged_q4f16.onnx', bytes: 477_584 },
      { path: 'onnx/decoder_model_merged_q4f16.onnx_data', bytes: 1_873_099_776 },
      { path: 'onnx/decoder_model_merged_q4f16.onnx_data_1', bytes: 1_845_312_512 },
      { path: 'onnx/decoder_model_merged_q4f16.onnx_data_2', bytes: 1_855_536_128 },
      { path: 'onnx/decoder_model_merged_q4f16.onnx_data_3', bytes: 1_346_568_192 },
    ],
    largestBuffer: 503_316_480,
    gpuMemory: 8.2e9,
    context: 4096,
    defaults: { temperature: 1.0, topP: 0.95, topK: 64, repetitionPenalty: 1.0, reasoning: false },
    think: { open: '<|channel>thought', close: '<channel|>', promptOpens: false },
    license: { name: 'Apache 2.0', url: 'https://ai.google.dev/gemma/docs/gemma_4_license' },
    source: 'https://huggingface.co/google/gemma-4-12B-it',
    params: '12B',
  },
];

/** Models that can actually be offered (the export of the largest one may not be published). */
export const AVAILABLE = MODELS.filter((m) => m.repo && m.revision && m.files.length);

export const modelById = (id: string): ModelInfo | undefined => AVAILABLE.find((m) => m.id === id);

export const downloadBytes = (m: ModelInfo): number => m.files.reduce((a, f) => a + f.bytes, 0);

/** URL prefix of the pinned revision; Transformers.js builds the same URLs. */
export const fileUrl = (m: ModelInfo, path: string): string => `https://huggingface.co/${m.repo}/resolve/${m.revision}/${path}`;
