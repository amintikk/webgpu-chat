// Debug mode: a step-by-step record of what happens when a model loads (download, cache, GPU session, memory),
// kept only in this page and copied by hand as a text report. Off by default; turned on in Settings or with
// ?debug=1. Nothing is sent anywhere.

const KEY = 'webgpu-chat:debug';
const LOG_KEY = 'webgpu-chat:debug-log'; // this session's record, saved as it grows (survives a reload)
const PREV_KEY = 'webgpu-chat:debug-log-prev'; // the record of the session before this one
const MAX_LINES = 800;
const lines: string[] = [];
let previous = '';
let memTimer = 0;
let saveTimer = 0;

function readStored(key: string): string {
  try {
    return localStorage.getItem(key) ?? '';
  } catch {
    return '';
  }
}

function writeStored(key: string, value: string) {
  try {
    if (value) localStorage.setItem(key, value);
    else localStorage.removeItem(key);
  } catch {
    /* storage full or blocked: the record stays in memory only */
  }
}

function save(now = false) {
  clearTimeout(saveTimer);
  const write = () => writeStored(LOG_KEY, lines.join('\n'));
  if (now) write();
  else saveTimer = window.setTimeout(write, 300);
}

/** Starts a session: the record of the one before (if it was left behind) is kept for the report. */
function startSession() {
  const left = readStored(LOG_KEY);
  if (left) {
    previous = left;
    writeStored(PREV_KEY, left);
    writeStored(LOG_KEY, '');
  } else {
    previous = readStored(PREV_KEY);
  }
  lines.length = 0;
  lines.push(`${new Date().toISOString().slice(11, 23)} — sesión nueva —`);
  save(true);
}

export function debugOn(): boolean {
  try {
    return localStorage.getItem(KEY) === '1';
  } catch {
    return false;
  }
}

export function setDebug(on: boolean): void {
  try {
    if (on) localStorage.setItem(KEY, '1');
    else localStorage.removeItem(KEY);
  } catch {
    /* storage unavailable: the mode lasts until the page is closed */
  }
  if (!on) stopMemoryWatch();
}

/** `?debug=1` in the address turns the mode on for this browser. Call once at start-up. */
export function applyUrlDebug(): void {
  if (new URLSearchParams(location.search).get('debug') === '1') setDebug(true);
  if (debugOn()) {
    startSession();
    // The record is saved when the page goes to the background or is closed, not only on a timer.
    addEventListener('pagehide', () => save(true));
    document.addEventListener('visibilitychange', () => document.hidden && save(true));
  }
}

/** Adds one line to the record (only while the mode is on). */
export function dlog(text: string): void {
  if (!debugOn()) return;
  const stamp = new Date().toISOString().slice(11, 23);
  lines.push(`${stamp} ${text}`);
  if (lines.length > MAX_LINES) lines.splice(0, lines.length - MAX_LINES);
  save();
}

export function debugLines(): string {
  return lines.join('\n');
}

export function clearDebug(): void {
  lines.length = 0;
  previous = '';
  writeStored(PREV_KEY, '');
  writeStored(LOG_KEY, '');
}

/** Heap numbers (Chromium only: performance.memory), as a short text. */
export function memoryText(): string {
  const pm = (performance as Performance & { memory?: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
  if (!pm) return 'heap: no disponible en este navegador';
  const mb = (n: number) => `${Math.round(n / 1048576)} MB`;
  return `heap usado ${mb(pm.usedJSHeapSize)} · reservado ${mb(pm.totalJSHeapSize)} · límite ${mb(pm.jsHeapSizeLimit)}`;
}

/** Samples the heap every second while a model loads, so the record shows where memory climbs. */
export function startMemoryWatch(label: string): void {
  if (!debugOn()) return;
  stopMemoryWatch();
  memTimer = window.setInterval(() => dlog(`${label} · ${memoryText()}`), 1000);
}

export function stopMemoryWatch(): void {
  if (memTimer) clearInterval(memTimer);
  memTimer = 0;
}

/** Errors that escape every handler are recorded too (with their stack). */
export function watchErrors(): void {
  addEventListener('error', (e) => dlog(`ERROR no capturado: ${e.message} @ ${e.filename}:${e.lineno}\n${e.error?.stack ?? ''}`));
  addEventListener('unhandledrejection', (e) => {
    const r = e.reason;
    dlog(`PROMISA RECHAZADA: ${r instanceof Error ? `${r.name}: ${r.message}\n${r.stack ?? ''}` : String(r)}`);
  });
}

/** Full text report: the environment, then the record. */
export async function reportText(extra: string[] = []): Promise<string> {
  const nav = navigator as Navigator & { deviceMemory?: number; userAgentData?: { platform?: string; brands?: { brand: string; version: string }[] } };
  const out: string[] = [];
  out.push('== WebGPU Chat · informe de depuración ==');
  out.push(`fecha: ${new Date().toString()}`);
  out.push(`navegador: ${navigator.userAgent}`);
  out.push(`marcas: ${(nav.userAgentData?.brands ?? []).map((b) => `${b.brand} ${b.version}`).join(', ') || 'n/d'}`);
  out.push(`plataforma: ${nav.userAgentData?.platform ?? navigator.platform}`);
  out.push(`RAM según el navegador (deviceMemory, que suele topar en 8 GB): ${nav.deviceMemory ?? 'n/d'} GB`);
  out.push(`núcleos: ${navigator.hardwareConcurrency}`);
  out.push(`${memoryText()}`);
  try {
    const est = await navigator.storage?.estimate?.();
    if (est) out.push(`almacenamiento: usado ${Math.round((est.usage ?? 0) / 1e6)} MB de cuota ${Math.round((est.quota ?? 0) / 1e6)} MB`);
  } catch {
    /* not available */
  }
  try {
    const gpu = navigator.gpu ? await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }) : null;
    if (!gpu) out.push('WebGPU: sin adaptador');
    else {
      const i = gpu.info ?? ({} as GPUAdapterInfo);
      out.push(`GPU: ${i.vendor || 'n/d'} ${i.architecture || ''} ${i.description || ''}`.trim());
      out.push(`GPU: shader-f16 ${gpu.features.has('shader-f16') ? 'sí' : 'no'} · maxBufferSize ${fmtBig(gpu.limits.maxBufferSize)} · maxStorageBufferBindingSize ${fmtBig(gpu.limits.maxStorageBufferBindingSize)} · maxComputeWorkgroupStorageSize ${gpu.limits.maxComputeWorkgroupStorageSize}`);
    }
  } catch (e) {
    out.push(`WebGPU: error al pedir el adaptador: ${e instanceof Error ? e.message : String(e)}`);
  }
  out.push(...extra);
  out.push('');
  if (previous) {
    out.push('== Sesión anterior (la página se recargó o se cerró; su registro se conserva aquí) ==');
    out.push(previous);
    out.push('');
  }
  out.push('== Registro ==');
  out.push(lines.length ? lines.join('\n') : '(vacío: activa el modo depuración y vuelve a cargar el modelo)');
  return out.join('\n');
}

function fmtBig(n: number): string {
  return n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${Math.round(n / 1e6)} MB`;
}
