// Images in the chat: the model's image processor turns each picture into patches, the model's image encoder
// (downloaded the first time an image is sent) turns them into "image tokens", and Transformers.js merges
// those into the prompt where the chat template placed the image. Everything runs in this worker.
import { Gemma4ImageProcessor, Qwen2VLImageProcessor, RawImage, Tensor, cat } from '@huggingface/transformers';
import type { ImageRef } from './protocol.ts';
import type { ModelInfo, VisionInfo } from './models.ts';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Any = any;

interface Processed {
  /** Image tokens this picture takes in the prompt. */
  tokens: number;
  pixel_values: Tensor;
  /** Qwen: [t, h, w] patch grid. Gemma: the patch positions. */
  grid?: Tensor;
  positions?: Tensor;
}

const processed = new Map<string, Processed>();
let processor: Any = null;
let processorFor = '';

export function forgetImages(): void {
  processed.clear();
  processor = null;
  processorFor = '';
}

function imageProcessor(m: ModelInfo): Any {
  if (processorFor !== m.id) {
    const v = m.vision!;
    processor = v.kind === 'qwen' ? new Qwen2VLImageProcessor(v.processor as Any) : new Gemma4ImageProcessor(v.processor as Any);
    processorFor = m.id;
  }
  return processor;
}

/** Patches of one picture (kept by its id, so a conversation's pictures are processed once). */
export async function processImage(m: ModelInfo, img: ImageRef): Promise<Processed> {
  const hit = processed.get(img.id);
  if (hit) return hit;
  const raw = onWhite(await RawImage.fromBlob(img.blob));
  const proc = imageProcessor(m);
  let p: Processed;
  if (m.vision!.kind === 'qwen') {
    const { pixel_values, image_grid_thw } = await proc([raw]);
    const [t, h, w] = (image_grid_thw.tolist()[0] as (number | bigint)[]).map(Number);
    const merge = Number((m.vision!.processor as Any).merge_size ?? 2);
    p = { tokens: (t! * h! * w!) / (merge * merge), pixel_values, grid: image_grid_thw };
  } else {
    const { pixel_values, image_position_ids, num_soft_tokens_per_image } = await proc([raw]);
    p = { tokens: num_soft_tokens_per_image[0], pixel_values: toFloat16(pixel_values), positions: image_position_ids };
  }
  processed.set(img.id, p);
  return p;
}

/**
 * The chat template writes one placeholder per picture; each becomes as many image tokens as the picture
 * takes (Qwen: <|image_pad|> × n; Gemma: blank lines, <start_of_image>, the image tokens, <end_of_image>).
 */
export function expandImageTokens(ids: number[], counts: number[], v: VisionInfo, cfg: Any, newline: number[]): number[] {
  const imageId = Number(cfg.image_token_id);
  let k = 0;
  return ids.flatMap((x) => {
    if (x !== imageId || k >= counts.length) return [x];
    const n = counts[k++]!;
    if (v.kind === 'qwen') return Array<number>(n).fill(imageId);
    return [...newline, Number(cfg.boi_token_id), ...Array<number>(n).fill(imageId), Number(cfg.eoi_token_id), ...newline];
  });
}

/**
 * Extra inputs for generation. Qwen places text after a picture by its patch grid, so the grids of every
 * picture in the conversation go with every turn; the pixels only when the pictures are read in this pass.
 */
export function visionInputs(v: VisionInfo, images: Processed[], withPixels: boolean): Record<string, Tensor> {
  if (!images.length) return {};
  if (v.kind === 'qwen') {
    const grid = cat(images.map((p) => p.grid!), 0);
    return withPixels ? { image_grid_thw: grid, pixel_values: cat(images.map((p) => p.pixel_values), 0) } : { image_grid_thw: grid };
  }
  if (!withPixels) return {};
  return { pixel_values: cat(images.map((p) => p.pixel_values), 0), image_position_ids: cat(images.map((p) => p.positions!), 0) };
}

/**
 * Transparent pictures (PNG, WebP, GIF) on white, as any viewer shows them. The image processors drop the alpha
 * channel and keep whatever colours lie under transparent pixels, often noise or leftovers, which the model then
 * reads as part of the picture.
 */
function onWhite(raw: RawImage): RawImage {
  if (raw.channels !== 4) return raw;
  const src = raw.data;
  const out = new Uint8ClampedArray((src.length / 4) * 3);
  for (let i = 0, j = 0; i < src.length; i += 4, j += 3) {
    const a = src[i + 3]! / 255;
    const w = 255 * (1 - a);
    out[j] = src[i]! * a + w;
    out[j + 1] = src[i + 1]! * a + w;
    out[j + 2] = src[i + 2]! * a + w;
  }
  return new RawImage(out, raw.width, raw.height, 3);
}

/** The Gemma encoder takes half-precision pixels. */
function toFloat16(t: Tensor): Tensor {
  const src = t.data as Float32Array;
  const F16 = (globalThis as Any).Float16Array;
  if (F16) return new Tensor('float16', F16.from(src), t.dims);
  const out = new Uint16Array(src.length);
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  for (let i = 0; i < src.length; i++) {
    f[0] = src[i]!;
    const x = u[0]!;
    const sign = (x >>> 16) & 0x8000;
    const e = ((x >>> 23) & 0xff) - 112;
    const mant = x & 0x7fffff;
    out[i] = e <= 0 ? sign : e >= 31 ? sign | 0x7c00 : (sign | (e << 10) | (mant >>> 13)) + ((mant & 0x1000) ? 1 : 0);
  }
  return new Tensor('float16', out, t.dims);
}
