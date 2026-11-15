// When a model moves to another repository (same files, new address), what the browser already downloaded is
// moved to the new address instead of being downloaded again. Runs once per page and once per worker.
import { MODELS, fileUrl, visionFiles } from './models.ts';

let done: Promise<void> | null = null;

export function moveLegacyCache(): Promise<void> {
  return (done ??= run());
}

async function run() {
  if (!('caches' in globalThis)) return;
  let cache: Cache;
  try {
    cache = await caches.open('transformers-cache');
  } catch {
    return;
  }
  for (const m of MODELS) {
    if (!m.legacy) continue;
    for (const f of [...m.files, ...visionFiles(m)]) {
      try {
        const to = fileUrl(m, f.path);
        const from = `https://huggingface.co/${m.legacy.repo}/resolve/${m.legacy.revision}/${f.path}`;
        if (await cache.match(to)) continue;
        const old = await cache.match(from);
        if (!old) continue;
        const length = Number(old.headers.get('content-length'));
        if (length && length !== f.bytes) continue; // not the same file: it is downloaded as usual
        await cache.put(to, old);
        await cache.delete(from);
      } catch {
        /* e.g. no room for the copy: this file is downloaded again instead */
      }
    }
  }
}
