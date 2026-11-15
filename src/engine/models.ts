import type { ThinkMarkers } from './reasoning.ts';

// Model catalogue. Files are downloaded straight from Hugging Face at a pinned commit, so what runs
// in the browser is always the exact revision that was measured (see results/ in the repository).

export type ModelId = 'lfm3-1.7b' | 'qwen3.5-4b' | 'gemma4-12b';
export type ModelClass = 'Lfm3ForCausalLM' | 'Qwen3_5ForCausalLM' | 'Gemma4ForCausalLM';

export interface ModelFile {
  path: string;
  bytes: number;
  /**
   * Downloaded from another repository instead of the model's own, e.g. a graph tuned for WebGPU that uses the
   * original weights. It is still cached under the model's own URL (the key Transformers.js looks up).
   */
  src?: { repo: string; revision: string; path: string };
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
  /**
   * Sessions whose weights are not just their own `<name>_data, _1…` files. `formats` replaces the chunk counts
   * of config.json (0 = take `files`), and `files` are the data files those sessions get, in this order.
   */
  sharedData?: { formats: Record<string, number>; files: string[] };
  /**
   * Image reading, when the model has it. The encoder is downloaded the first time an image is sent, not with
   * the model. `data` is the encoder's weights file as the graph names it.
   */
  vision?: VisionInfo;
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

export interface VisionInfo {
  kind: 'qwen' | 'gemma';
  graph: ModelFile;
  data: ModelFile & { name: string };
  /** Image processor settings (Transformers.js Qwen2VLImageProcessor or Gemma4ImageProcessor). */
  processor: Record<string, unknown>;
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
      // Same graph with accuracy_level=4 on MatMulNBits (int8 DP4A path for prompts); see AMINOXDEV/webgpu-chat-graphs.
      { path: 'onnx/decoder_model_merged_q4f16.onnx', bytes: 939_281, src: { repo: 'AMINOXDEV/webgpu-chat-graphs', revision: 'd1b60fb60d2ff1d6520de073d9824d71cec35267', path: 'qwen3.5-4b-opt/decoder_model_merged_q4f16.onnx' } },
      { path: 'onnx/decoder_model_merged_q4f16.onnx_data', bytes: 2_065_635_328 },
      { path: 'onnx/decoder_model_merged_q4f16.onnx_data_1', bytes: 367_513_600 },
    ],
    // Qwen3.5 reads images with its own encoder (in fp16, the precision it was trained in: 672 MB).
    vision: {
      kind: 'qwen',
      graph: { path: 'onnx/vision_encoder_fp16.onnx', bytes: 345_836 },
      data: { path: 'onnx/vision_encoder_fp16.onnx_data', bytes: 671_943_680, name: 'vision_encoder_fp16.onnx_data' },
      // Up to 640 image tokens (one per 32×32 pixels): enough to read text in a screenshot.
      processor: {
        size: { shortest_edge: 65_536, longest_edge: 655_360 },
        patch_size: 16, temporal_patch_size: 2, merge_size: 2,
        image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5],
        do_resize: true, do_rescale: true, do_normalize: true, do_convert_rgb: true, rescale_factor: 1 / 255,
      },
    },
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
    // Community export of Google's QAT checkpoint (gemma-4-12B-it-qat-q4_0): 4-bit weights trained for 4 bits,
    // fused GroupQueryAttention (1359 nodes instead of 3350 in our own export, 8 casts instead of 570). Its
    // config.json declares a model type Transformers.js 4.3 does not know (gemma4_unified_text), so the config of
    // our export (same architecture, as model_type gemma4) is used. The graph has accuracy_level=4 (DP4A prompts).
    repo: 'james58899/gemma-4-12B-it-qat-ONNX',
    revision: '60555e645e7c456b4a013cc2e93e9bd6379e3006',
    cls: 'Gemma4ForCausalLM',
    dtype: { embed_tokens: 'q4f16', decoder_model_merged: 'q4f16' },
    files: [
      { path: 'config.json', bytes: 3_555, src: { repo: 'AMINOXDEV/gemma-4-12B-it-ONNX', revision: '90e0a27bf539717b6055d7e434b9faa1f0dbd9ff', path: 'config.json' } },
      { path: 'generation_config.json', bytes: 260 },
      // Tokenizer and chat template (embedded in tokenizer_config.json) from our export: same Google tokenizer.
      { path: 'tokenizer.json', bytes: 32_157_161, src: { repo: 'AMINOXDEV/gemma-4-12B-it-ONNX', revision: '90e0a27bf539717b6055d7e434b9faa1f0dbd9ff', path: 'tokenizer.json' } },
      { path: 'tokenizer_config.json', bytes: 22_222, src: { repo: 'AMINOXDEV/gemma-4-12B-it-ONNX', revision: '90e0a27bf539717b6055d7e434b9faa1f0dbd9ff', path: 'tokenizer_config.json' } },
      { path: 'onnx/embed_tokens_q4f16.onnx', bytes: 585 },
      { path: 'onnx/embed_tokens_q4f16.onnx_data', bytes: 566_231_040 },
      { path: 'onnx/decoder_model_merged_q4f16.onnx', bytes: 455_338, src: { repo: 'AMINOXDEV/webgpu-chat-graphs', revision: 'fedd41d62b7427f8dbedc7819ee5bed5d0890832', path: 'gemma-4-12b-qat/decoder_model_merged_q4f16.onnx' } },
      { path: 'onnx/decoder_model_merged_q4f16.onnx_data', bytes: 1_997_070_336 },
      { path: 'onnx/decoder_model_merged_q4f16.onnx_data_1', bytes: 1_988_599_296 },
      { path: 'onnx/decoder_model_merged_q4f16.onnx_data_2', bytes: 1_977_876_480 },
      { path: 'onnx/decoder_model_merged_q4f16.onnx_data_3', bytes: 232_169_472 },
    ],
    // The decoder reads lm_head.qweight from the embeddings file (tied weights), so it gets that file too.
    sharedData: {
      formats: { 'embed_tokens_q4f16.onnx': 1, 'decoder_model_merged_q4f16.onnx': 0 },
      files: ['decoder_model_merged_q4f16.onnx_data', 'decoder_model_merged_q4f16.onnx_data_1', 'decoder_model_merged_q4f16.onnx_data_2', 'decoder_model_merged_q4f16.onnx_data_3', 'embed_tokens_q4f16.onnx_data'],
    },
    // Gemma 4 12B sees images as 48×48-pixel patches projected straight into the language model ("unified"):
    // the encoder is that projection, exported in fp16 by the ONNX Runtime team.
    vision: {
      kind: 'gemma',
      graph: { path: 'onnx/vision_encoder_fp16.onnx', bytes: 19_002, src: { repo: 'justinchuby/gemma-4-12b-onnx', revision: '52a1c4fa78a69f30f96c8fe20f6368621bedbb57', path: 'f16/default/vision_encoder/model.onnx' } },
      data: { path: 'onnx/vision_encoder_fp16.onnx.data', bytes: 100_007_936, name: 'model.onnx.data', src: { repo: 'justinchuby/gemma-4-12b-onnx', revision: '52a1c4fa78a69f30f96c8fe20f6368621bedbb57', path: 'f16/default/vision_encoder/model.onnx.data' } },
      processor: { patch_size: 48, pooling_kernel_size: 1, max_soft_tokens: 280, resample: 3, do_rescale: true, rescale_factor: 1 / 255, do_resize: true, do_convert_rgb: true },
    },
    largestBuffer: 503_316_480,
    gpuMemory: 8.7e9, // 6.8 GB weights + KV cache at 4096 tokens (sliding layers keep their full cache here)
    context: 4096,
    defaults: { temperature: 1.0, topP: 0.95, topK: 64, repetitionPenalty: 1.0, reasoning: false },
    think: { open: '<|channel>thought', close: '<channel|>', promptOpens: false },
    license: { name: 'Apache 2.0', url: 'https://ai.google.dev/gemma/docs/gemma_4_license' },
    source: 'https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized',
    params: '12B',
  },
];

/** Models that can actually be offered (the export of the largest one may not be published). */
export const AVAILABLE = MODELS.filter((m) => m.repo && m.revision && m.files.length);

export const modelById = (id: string): ModelInfo | undefined => AVAILABLE.find((m) => m.id === id);

export const downloadBytes = (m: ModelInfo): number => m.files.reduce((a, f) => a + f.bytes, 0);

/** URL prefix of the pinned revision; Transformers.js builds the same URLs. */
export const fileUrl = (m: ModelInfo, path: string): string => `https://huggingface.co/${m.repo}/resolve/${m.revision}/${path}`;

/** Where a file is actually downloaded from (its `src`, or the model's own pinned URL). */
export const sourceUrl = (m: ModelInfo, path: string): string => {
  const src = [...m.files, ...visionFiles(m)].find((f) => f.path === path)?.src;
  return src ? `https://huggingface.co/${src.repo}/resolve/${src.revision}/${src.path}` : fileUrl(m, path);
};

/** The image encoder's files (empty for text-only models). */
export const visionFiles = (m: ModelInfo): ModelFile[] => (m.vision ? [m.vision.graph, m.vision.data] : []);
