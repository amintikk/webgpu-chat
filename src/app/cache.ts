// Model files downloaded by Transformers.js live in Cache Storage under the file URL at the pinned
// revision. These helpers report what is stored for each model and delete it.
import { MODELS, fileUrl, visionFiles, type ModelInfo } from '../engine/models.ts';

const CACHE = 'transformers-cache';

export interface CacheEntry {
  /** Files present / files expected. */
  present: number;
  expected: number;
  bytes: number;
}

async function openCache(): Promise<Cache | null> {
  try {
    return 'caches' in self ? await caches.open(CACHE) : null;
  } catch {
    return null; // e.g. storage blocked in a private window
  }
}

export async function cacheStatus(m: ModelInfo): Promise<CacheEntry> {
  const c = await openCache();
  const out: CacheEntry = { present: 0, expected: m.files.length, bytes: 0 };
  if (!c) return out;
  for (const f of m.files) {
    const r = await c.match(fileUrl(m, f.path));
    if (!r) continue;
    out.present++;
    out.bytes += Number(r.headers.get('content-length')) || f.bytes;
  }
  return out;
}

export const isComplete = (e: CacheEntry) => e.expected > 0 && e.present === e.expected;

/** Delete every cached file of this model (any revision of the same repository). */
export async function deleteModel(m: ModelInfo): Promise<void> {
  const c = await openCache();
  if (!c) return;
  const prefix = `https://huggingface.co/${m.repo}/`;
  for (const req of await c.keys()) if (req.url.startsWith(prefix)) await c.delete(req);
}

/** Cached files of older revisions or of models no longer offered (left behind by updates). */
export async function orphanBytes(): Promise<number> {
  const c = await openCache();
  if (!c) return 0;
  const known = new Set(MODELS.flatMap((m) => [...m.files, ...visionFiles(m)].map((f) => fileUrl(m, f.path))));
  let bytes = 0;
  for (const req of await c.keys()) {
    if (known.has(req.url)) continue;
    const r = await c.match(req);
    bytes += Number(r?.headers.get('content-length')) || 0;
  }
  return bytes;
}

export async function deleteOrphans(): Promise<void> {
  const c = await openCache();
  if (!c) return;
  const known = new Set(MODELS.flatMap((m) => [...m.files, ...visionFiles(m)].map((f) => fileUrl(m, f.path))));
  for (const req of await c.keys()) if (!known.has(req.url) && req.url.startsWith('https://huggingface.co/')) await c.delete(req);
}

export async function storageEstimate(): Promise<{ usage: number; quota: number } | null> {
  try {
    const e = await navigator.storage?.estimate();
    return e ? { usage: e.usage ?? 0, quota: e.quota ?? 0 } : null;
  } catch {
    return null;
  }
}
