// WebGPU capability check, done on the page before any download. A model is blocked when the
// adapter cannot hold its largest weight buffer, lacks 16-bit float shaders, or the device clearly has
// too little memory; otherwise the heaviest model that fits comfortably is recommended.
import { AVAILABLE, type ModelId, type ModelInfo } from './models.ts';

export interface GpuReport {
  ok: boolean;
  /** Why WebGPU is unusable (when !ok). */
  reason?: 'no-webgpu' | 'no-adapter' | 'no-f16' | 'error';
  error?: string;
  vendor: string;
  architecture: string;
  description: string;
  software: boolean;
  f16: boolean;
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  /** navigator.deviceMemory (GB, capped at 8 by browsers) or 0 if unknown. */
  deviceMemory: number;
  mobile: boolean;
}

export type Fit = { ok: true; warn?: 'memory-unknown' | 'memory-low' | 'software' } | { ok: false; why: 'no-f16' | 'buffer' | 'memory' | 'mobile' | 'no-webgpu' };

export async function probeGpu(): Promise<GpuReport> {
  const nav = navigator as Navigator & { deviceMemory?: number; userAgentData?: { mobile?: boolean } };
  const base: GpuReport = {
    ok: false,
    vendor: '',
    architecture: '',
    description: '',
    software: false,
    f16: false,
    maxBufferSize: 0,
    maxStorageBufferBindingSize: 0,
    deviceMemory: nav.deviceMemory ?? 0,
    mobile: nav.userAgentData?.mobile ?? /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent),
  };
  if (!('gpu' in navigator) || !navigator.gpu) return { ...base, reason: 'no-webgpu' };
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { ...base, reason: 'no-adapter' };
    const info = adapter.info ?? ({} as GPUAdapterInfo);
    const desc = `${info.vendor ?? ''} ${info.architecture ?? ''} ${info.description ?? ''}`;
    const r: GpuReport = {
      ...base,
      ok: true,
      vendor: info.vendor ?? '',
      architecture: info.architecture ?? '',
      description: info.description ?? '',
      software: Boolean((info as GPUAdapterInfo & { isFallbackAdapter?: boolean }).isFallbackAdapter) || /swiftshader|llvmpipe|lavapipe|software/i.test(desc),
      f16: adapter.features.has('shader-f16'),
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    };
    if (!r.f16) return { ...r, ok: false, reason: 'no-f16' };
    return r;
  } catch (e) {
    return { ...base, reason: 'error', error: e instanceof Error ? e.message : String(e) };
  }
}

/** Minimum device memory (GB, as reported by navigator.deviceMemory) per model tier. */
const MIN_MEMORY: Record<ModelInfo['tier'], number> = { light: 4, recommended: 8, max: 8 };

export function fit(m: ModelInfo, g: GpuReport): Fit {
  if (!g.ok && g.reason !== 'no-f16') return { ok: false, why: 'no-webgpu' };
  if (!g.f16) return { ok: false, why: 'no-f16' };
  const limit = Math.min(g.maxBufferSize, g.maxStorageBufferBindingSize);
  if (limit < m.largestBuffer) return { ok: false, why: 'buffer' };
  // Phones: only the light model. Qwen3.5 4B closes the browser even on a 12 GB flagship (tested on a recent
  // Snapdragon): while loading, the weights sit in memory and on the GPU at the same time.
  if (m.tier !== 'light' && g.mobile) return { ok: false, why: 'mobile' };
  // navigator.deviceMemory is capped at 8 GB and is only a hint (some browsers report less than the machine has),
  // so on computers it warns instead of blocking. Phones are blocked above, on their own rule.
  if (g.software) return { ok: true, warn: 'software' };
  if (g.deviceMemory && g.deviceMemory < MIN_MEMORY[m.tier]) return { ok: true, warn: 'memory-low' };
  if (m.tier === 'max') return { ok: true, warn: 'memory-unknown' };
  return { ok: true };
}

/** Heaviest model that fits without warnings, never the largest one (it is always an explicit choice). */
export function recommend(g: GpuReport): ModelId | null {
  const order: ModelInfo['tier'][] = g.mobile || (g.deviceMemory && g.deviceMemory < 8) ? ['light'] : ['recommended', 'light'];
  for (const tier of order) {
    const m = AVAILABLE.find((x) => x.tier === tier);
    if (m && fit(m, g).ok) return m.id;
  }
  return null;
}
