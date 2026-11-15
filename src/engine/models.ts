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
  /** The model's previous repository and revision: files cached under it are moved to the new address. */
  legacy?: { repo: string; revision: string };
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
    repo: 'AMINOXDEV/LFM3-1.7B-ONNX',
    revision: '1ef64c82a7e6d3187a74f9103f70d733ddb9ed52',
    // Where these files were cached before they moved to our own repository (same bytes).
    legacy: { repo: 'onnx-community/LFM3-1.7B-ONNX', revision: 'ab9fec4c681a25bec3b48b0b97a62fc2660a2985' },
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
    repo: 'AMINOXDEV/Qwen3.5-4B-ONNX',
    revision: 'b63dbd33b9d3b9b96b711f060e4dddfe66993ef5',
    // Where these files were cached before they moved to our own repository (same bytes).
    legacy: { repo: 'onnx-community/Qwen3.5-4B-ONNX-OPT', revision: 'bb072268d0912bdc5cf4f41c8c6f53a03fd68624' },
    cls: 'Qwen3_5ForCausalLM',
    dtype: { embed_tokens: 'q4f16', decoder_model_merged: 'q4f16' },
    files: [
      { path: 'config.json', bytes: 3_198 },
      { path: 'generation_config.json', bytes: 248 },
      { path: 'tokenizer.json', bytes: 19_226_111 },
      { path: 'tokenizer_config.json', bytes: 10_070 },
      { path: 'onnx/embed_tokens_q4f16.onnx', bytes: 1_064 },
      { path: 'onnx/embed_tokens_q4f16.onnx_data', bytes: 367_513_600 },
      // Graph with accuracy_level=4 on MatMulNBits (int8 DP4A path for prompts); weights unchanged.
      { path: 'onnx/decoder_model_merged_q4f16.onnx', bytes: 939_281 },
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
    // Our repository gathers: the community export of Google's QAT checkpoint (gemma-4-12B-it-qat-q4_0, by
    // james58899: 4-bit weights trained for 4 bits, fused GroupQueryAttention), its graph with accuracy_level=4
    // (DP4A prompts), a config.json and tokenizer Transformers.js 4.3 understands (model_type gemma4), and the fp16
    // vision encoder of the ONNX Runtime team. Sources and revisions are listed in the repository's README.
    repo: 'AMINOXDEV/gemma-4-12B-it-qat-ONNX',
    revision: '9847447e160486f74b4fb4a90472b5da0c5bf647',
    // Where these files were cached before they moved to our own repository (same bytes).
    legacy: { repo: 'james58899/gemma-4-12B-it-qat-ONNX', revision: '60555e645e7c456b4a013cc2e93e9bd6379e3006' },
    cls: 'Gemma4ForCausalLM',
    dtype: { embed_tokens: 'q4f16', decoder_model_merged: 'q4f16' },
    files: [
      { path: 'config.json', bytes: 3_555 },
      { path: 'generation_config.json', bytes: 260 },
      { path: 'tokenizer.json', bytes: 32_157_161 },
      { path: 'tokenizer_config.json', bytes: 22_222 },
      { path: 'onnx/embed_tokens_q4f16.onnx', bytes: 585 },
      { path: 'onnx/embed_tokens_q4f16.onnx_data', bytes: 566_231_040 },
      // Same graph with an If per attention layer: a masked attention when a picture is read (both directions
      // inside each picture, as Gemma 4 does), the fused GQA otherwise. Weights unchanged (tools/gemma4/bidir.py).
      { path: 'onnx/decoder_model_merged_q4f16.onnx', bytes: 571_104, src: { repo: 'AMINOXDEV/gemma-4-12B-it-qat-ONNX', revision: '4720d58a2adc8f9e8d17e0bbabbfa3917a5ea82b', path: 'onnx/decoder_model_merged_q4f16_vision.onnx' } },
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
    // Gemma 4 12B sees pictures as 48×48-pixel patches projected straight into the language model ("unified"),
    // which reads each picture's patches with attention in both directions: the worker passes that mask to the
    // graph's masked branch while a picture is read. The projection is the ONNX Runtime team's fp16 export graph.
    vision: {
      kind: 'gemma',
      // With the vision weights of the QAT checkpoint, which the QAT language model was trained with.
      graph: { path: 'onnx/vision_encoder_qat_fp16.onnx', bytes: 19_231, src: { repo: 'AMINOXDEV/gemma-4-12B-it-qat-ONNX', revision: '94deb8d0999a37e0b555da6b81c77bb480d1736f', path: 'onnx/vision_encoder_qat_fp16.onnx' } },
      data: { path: 'onnx/vision_encoder_qat_fp16.onnx.data', bytes: 99_852_288, name: 'vision_encoder_qat_fp16.onnx.data', src: { repo: 'AMINOXDEV/gemma-4-12B-it-qat-ONNX', revision: '94deb8d0999a37e0b555da6b81c77bb480d1736f', path: 'onnx/vision_encoder_qat_fp16.onnx.data' } },
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
