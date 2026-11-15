// When a model moves to another repository (same files, new address), what the browser already downloaded is
// moved to the new address instead of being downloaded again. It runs in the background on the page, and for
// the model being loaded in the worker; a Web Lock keeps the two from copying at the same time.
import { MODELS, fileUrl, visionFiles, type ModelInfo } from './models.ts';

/** The address a file had in the model's previous repository (null if the model did not move). */
export const legacyUrl = (m: ModelInfo, path: string): string | null =>
  m.legacy ? `https://huggingface.co/${m.legacy.repo}/resolve/${m.legacy.revision}/${path}` : null;

/** Every previous address still worth keeping in the cache (not leftovers to delete). */
export const legacyUrls = (): string[] => MODELS.flatMap((m) => [...m.files, ...visionFiles(m)].map((f) => legacyUrl(m, f.path)).filter((u): u is string => !!u));

export async function moveLegacyCache(only?: ModelInfo): Promise<void> {
  if (!('caches' in globalThis)) return;
  const run = () => move(only ? [only] : MODELS);
  const locks = (globalThis.navigator as Navigator | undefined)?.locks;
  try {
    if (locks) await locks.request('webgpu-chat:cache-move', run);
    else await run();
  } catch {
    /* storage unavailable: files are downloaded as usual */
  }
}

async function move(models: ModelInfo[]) {
  const cache = await caches.open('transformers-cache');
  for (const m of models) {
    for (const f of [...m.files, ...visionFiles(m)]) {
      const from = legacyUrl(m, f.path);
      if (!from) continue;
      try {
        const old = await cache.match(from);
        if (!old) continue;
        const to = fileUrl(m, f.path);
        if (!(await cache.match(to))) {
          const length = Number(old.headers.get('content-length'));
          if (length && length !== f.bytes) continue; // not the same file: it is downloaded as usual
          await cache.put(to, old);
        }
        await cache.delete(from);
      } catch {
        /* e.g. no room for the copy: this file is downloaded again instead */
      }
    }
  }
}
