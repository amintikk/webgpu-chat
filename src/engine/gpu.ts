// Capability detection and advisory warnings. Model selection is never restricted.
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

export type ModelWarning = 'no-webgpu' | 'no-f16' | 'buffer' | 'mobile' | 'software' | 'memory-low' | 'memory-unknown';

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

export function modelWarning(m: ModelInfo, g: GpuReport): ModelWarning | null {
  if (!g.ok && g.reason !== 'no-f16') return 'no-webgpu';
  if (!g.f16) return 'no-f16';
  if (Math.min(g.maxBufferSize, g.maxStorageBufferBindingSize) < m.largestBuffer) return 'buffer';
  if (g.mobile && m.tier !== 'light') return 'mobile';
  if (g.software) return 'software';
  if (g.deviceMemory && g.deviceMemory < MIN_MEMORY[m.tier]) return 'memory-low';
  return m.tier === 'max' ? 'memory-unknown' : null;
}

/** A suggestion only; saved and explicit choices always take precedence. */
export function recommend(g: GpuReport): ModelId | null {
  if (!g.ok) return null;
  const tier = g.mobile || (g.deviceMemory && g.deviceMemory < 8) ? 'light' : 'recommended';
  const candidate = AVAILABLE.find((m) => m.tier === tier);
  if (candidate && !modelWarning(candidate, g)) return candidate.id;
  return AVAILABLE.find((m) => m.tier === 'light')?.id ?? null;
}
