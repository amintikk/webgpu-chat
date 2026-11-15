// Settings → Tus datos: every conversation in one JSON file (pictures included as data: URLs), and back.
// The file is made and read in the browser; nothing is uploaded.
import type { Conversation, StoredMessage } from './db.ts';

const FORMAT = 'webgpu-chat-backup';

const toDataUrl = (b: Blob) =>
  new Promise<string>((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(r.error);
    r.readAsDataURL(b);
  });

export async function exportBackup(conversations: Conversation[]): Promise<Blob> {
  const out = [];
  for (const c of conversations) {
    const messages = [];
    for (const m of c.messages) {
      messages.push(m.images?.length ? { ...m, images: await Promise.all(m.images.map(async (i) => ({ ...i, blob: await toDataUrl(i.blob) }))) } : m);
    }
    out.push({ ...c, messages });
  }
  return new Blob([JSON.stringify({ format: FORMAT, version: 1, exportedAt: new Date().toISOString(), conversations: out })], { type: 'application/json' });
}

/** Conversations from a backup file; throws if the file is not one. */
export async function readBackup(file: File): Promise<Conversation[]> {
  let data: { format?: string; conversations?: unknown };
  try {
    data = JSON.parse(await file.text());
  } catch {
    throw new Error('not-json');
  }
  if (data.format !== FORMAT || !Array.isArray(data.conversations)) throw new Error('not-backup');
  const list: Conversation[] = [];
  for (const raw of data.conversations as Conversation[]) {
    if (!raw || typeof raw.id !== 'string' || !Array.isArray(raw.messages)) continue;
    const messages: StoredMessage[] = [];
    for (const m of raw.messages) {
      if (m.role !== 'user' && m.role !== 'assistant') continue;
      const images = m.images?.length
        ? m.images.map((i) => ({ ...i, blob: fromDataUrl(String(i.blob as unknown)) }))
        : undefined;
      messages.push({ ...m, content: String(m.content ?? ''), ...(images ? { images } : {}) });
    }
    list.push({ ...raw, title: String(raw.title ?? ''), messages });
  }
  return list;
}

/** data: URL to Blob without fetch (the site's Content-Security-Policy does not allow fetching data: URLs). */
function fromDataUrl(url: string): Blob {
  const [head = '', body = ''] = url.split(',', 2);
  const type = /^data:([^;,]+)/.exec(head)?.[1] ?? 'application/octet-stream';
  const bin = atob(body);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type });
}
