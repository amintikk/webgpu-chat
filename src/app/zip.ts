// ZIP archives read in the browser, with no library: the central directory is parsed here and each entry is
// inflated with the platform's DecompressionStream('deflate-raw'). Only "stored" and "deflate" entries are read
// (what every common zip tool writes); encrypted entries are listed but not opened. Nothing is uploaded.

export interface ZipEntry {
  /** Path inside the archive, with forward slashes. Folders end in "/". */
  path: string;
  /** Uncompressed size in bytes (as the archive declares it). */
  size: number;
  method: number;
  encrypted: boolean;
  /** Where the entry's local header starts. */
  offset: number;
  compressedSize: number;
}

const u16 = (b: DataView, o: number) => b.getUint16(o, true);
const u32 = (b: DataView, o: number) => b.getUint32(o, true);
const u64 = (b: DataView, o: number) => Number(b.getBigUint64(o, true));

// Code page 437, upper half: names in zips written without the UTF-8 flag (older Windows "Send to → zip").
const CP437 =
  'ÇüéâäàåçêëèïîìÄÅÉæÆôöòûùÿÖÜ¢£¥₧ƒáíóúñÑªº¿⌐¬½¼¡«»░▒▓│┤╡╢╖╕╣║╗╝╜╛┐└┴┬├─┼╞╟╚╔╩╦╠═╬╧╨╤╥╙╘╒╓╫╪┘┌█▄▌▐▀αßΓπΣσµτΦΘΩδ∞φε∩≡±≥≤⌠⌡÷≈°∙·√ⁿ²■ ';

function decodeName(bytes: Uint8Array, utf8Flag: boolean): string {
  if (utf8Flag) return new TextDecoder().decode(bytes);
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } catch {
    return Array.from(bytes, (c) => (c < 128 ? String.fromCharCode(c) : CP437[c - 128])).join('');
  }
}

/** Lists the entries of a zip archive. Throws an Error (with an English message) when it is not a zip. */
export function listZip(bytes: Uint8Array): ZipEntry[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  // End of central directory: 22 bytes plus a comment of up to 65535 bytes, at the very end.
  let eocd = -1;
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 22 - 65535); i--) {
    if (u32(view, i) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error('not a zip archive');
  let count = u16(view, eocd + 10);
  let cdOffset = u32(view, eocd + 16);
  // Zip64: the real numbers are in a second record, pointed at by a locator just before this one.
  if ((count === 0xffff || cdOffset === 0xffffffff) && eocd >= 20 && u32(view, eocd - 20) === 0x07064b50) {
    const z = u64(view, eocd - 20 + 8);
    if (z + 56 <= bytes.length && u32(view, z) === 0x06064b50) {
      count = u64(view, z + 32);
      cdOffset = u64(view, z + 48);
    }
  }
  const out: ZipEntry[] = [];
  let p = cdOffset;
  for (let n = 0; n < count; n++) {
    if (p + 46 > bytes.length || u32(view, p) !== 0x02014b50) throw new Error('damaged zip archive');
    const flags = u16(view, p + 8);
    const method = u16(view, p + 10);
    let compressedSize = u32(view, p + 20);
    let size = u32(view, p + 24);
    const nameLen = u16(view, p + 28);
    const extraLen = u16(view, p + 30);
    const commentLen = u16(view, p + 32);
    let offset = u32(view, p + 42);
    const name = decodeName(bytes.subarray(p + 46, p + 46 + nameLen), (flags & 0x800) !== 0);
    // Zip64 extra field: 64-bit values for whichever of the three 32-bit ones were saturated, in this order.
    let e = p + 46 + nameLen;
    const end = e + extraLen;
    while (e + 4 <= end) {
      const id = u16(view, e);
      const len = u16(view, e + 2);
      if (id === 0x0001) {
        let q = e + 4;
        if (size === 0xffffffff) (size = u64(view, q)), (q += 8);
        if (compressedSize === 0xffffffff) (compressedSize = u64(view, q)), (q += 8);
        if (offset === 0xffffffff) offset = u64(view, q);
      }
      e += 4 + len;
    }
    out.push({ path: name.replace(/\\/g, '/').replace(/^\/+/, ''), size, method, encrypted: (flags & 1) !== 0, offset, compressedSize });
    p = end + commentLen;
  }
  return out;
}

/**
 * The entry's uncompressed bytes. `limit` caps how much is inflated, so a "zip bomb" (a tiny entry that
 * claims or expands to gigabytes) stops early: past it, this throws.
 */
export async function readEntry(bytes: Uint8Array, entry: ZipEntry, limit = Infinity): Promise<Uint8Array> {
  if (entry.encrypted) throw new Error('encrypted');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const h = entry.offset;
  if (h + 30 > bytes.length || u32(view, h) !== 0x04034b50) throw new Error('damaged zip archive');
  const start = h + 30 + u16(view, h + 26) + u16(view, h + 28);
  const data = bytes.subarray(start, start + entry.compressedSize);
  if (entry.method === 0) {
    if (data.length > limit) throw new Error('too large');
    return data;
  }
  if (entry.method !== 8) throw new Error('unsupported compression');
  const stream = new Blob([data as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  const reader = stream.getReader();
  const parts: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.length;
    if (total > limit) {
      void reader.cancel();
      throw new Error('too large');
    }
    parts.push(value);
  }
  const out = new Uint8Array(total);
  let o = 0;
  for (const part of parts) {
    out.set(part, o);
    o += part.length;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// What the model reads for an archive: a tree of its contents, then each readable file.

/** Folders whose contents are never read (tooling, dependencies, version control): listed once, collapsed. */
const SKIP_DIRS: Record<string, string> = {
  '.git': 'version control',
  '.svn': 'version control',
  '.hg': 'version control',
  node_modules: 'dependencies',
  vendor: 'dependencies',
  bower_components: 'dependencies',
  __pycache__: 'compiled',
  '.venv': 'virtual environment',
  venv: 'virtual environment',
  '.idea': 'editor settings',
  '.vs': 'editor settings',
  __MACOSX: 'macOS metadata',
  '.next': 'build output',
  '.nuxt': 'build output',
  '.cache': 'cache',
};

/** Files listed without their contents: generated, huge for what they say, or system clutter. */
const SKIP_FILES = /(^|\/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|composer\.lock|Cargo\.lock|poetry\.lock|Gemfile\.lock|\.DS_Store|Thumbs\.db|desktop\.ini)$|\.min\.(js|css)$|\.map$/i;

/** Why the folder this path is in is not read, or '' when it is read. Returns the folder too. */
export function skippedDir(path: string): { dir: string; why: string } | null {
  const parts = path.split('/');
  for (let i = 0; i < parts.length - 1; i++) {
    const why = SKIP_DIRS[parts[i]!];
    if (why) return { dir: parts.slice(0, i + 1).join('/') + '/', why };
  }
  return null;
}

export const skippedFile = (path: string) => SKIP_FILES.test(path);

export interface ArchiveItem {
  path: string;
  /** Read as text: shown in full after the tree. */
  text?: string;
  /** Why it is listed without its text, or what was done with it ("image, attached"). */
  note?: string;
}

/** README files first (they explain the rest), then everything in path order. */
const readmeFirst = (a: ArchiveItem, b: ArchiveItem) => {
  const ra = /(^|\/)readme(\.[a-z]+)?$/i.test(a.path) ? a.path.split('/').length : 99;
  const rb = /(^|\/)readme(\.[a-z]+)?$/i.test(b.path) ? b.path.split('/').length : 99;
  return ra - rb || a.path.localeCompare(b.path);
};

/** An indented tree of the archive's paths, each with its note. */
export function treeText(items: ArchiveItem[]): string {
  const sorted = [...items].sort((a, b) => a.path.localeCompare(b.path));
  const lines: string[] = [];
  const shown = new Set<string>();
  for (const it of sorted) {
    const isDir = it.path.endsWith('/');
    const parts = it.path.replace(/\/$/, '').split('/');
    for (let i = 0; i < parts.length - 1; i++) {
      const dir = parts.slice(0, i + 1).join('/') + '/';
      if (shown.has(dir)) continue;
      shown.add(dir);
      lines.push(`${'  '.repeat(i)}${parts[i]}/`);
    }
    if (isDir && shown.has(it.path)) continue;
    shown.add(it.path);
    lines.push(`${'  '.repeat(parts.length - 1)}${parts.at(-1)}${isDir ? '/' : ''}${it.note ? `  [${it.note}]` : ''}`);
  }
  return lines.join('\n');
}

/** The whole archive as the model reads it. */
export function archiveText(name: string, items: ArchiveItem[]): string {
  const files = items.filter((i) => !i.path.endsWith('/'));
  const blocks = items
    .filter((i) => i.text != null)
    .sort(readmeFirst)
    .map((i) => `<file path="${i.path.replace(/"/g, "'")}">\n${i.text!.trim()}\n</file>`);
  const head = `<archive name="${name.replace(/"/g, "'")}" files="${files.length}">\nContents:\n${treeText(items)}`;
  return [head, ...blocks, '</archive>'].join('\n\n');
}

// ---------------------------------------------------------------------------------------------
// Writing: a zip of text files (the model's answer as a project), deflated with CompressionStream.

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(data: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]!) & 0xff]! ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

async function deflate(data: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([data as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new CompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** A zip with these files (UTF-8 names, today's date). Small files are stored as they are. */
export async function makeZip(files: { path: string; data: Uint8Array }[], now = new Date()): Promise<Blob> {
  const enc = new TextEncoder();
  const time = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
  const date = ((Math.max(1980, now.getFullYear()) - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
  const parts: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;
  for (const f of files) {
    const name = enc.encode(f.path);
    const packed = f.data.length > 64 ? await deflate(f.data) : f.data;
    const method = packed === f.data || packed.length >= f.data.length ? 0 : 8;
    const body = method ? packed : f.data;
    const crc = crc32(f.data);
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    local.setUint16(4, 20, true);
    local.setUint16(6, 0x800, true); // names in UTF-8
    local.setUint16(8, method, true);
    local.setUint16(10, time, true);
    local.setUint16(12, date, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, body.length, true);
    local.setUint32(22, f.data.length, true);
    local.setUint16(26, name.length, true);
    const head = new DataView(new ArrayBuffer(46));
    head.setUint32(0, 0x02014b50, true);
    head.setUint16(4, 20, true);
    head.setUint16(6, 20, true);
    head.setUint16(8, 0x800, true);
    head.setUint16(10, method, true);
    head.setUint16(12, time, true);
    head.setUint16(14, date, true);
    head.setUint32(16, crc, true);
    head.setUint32(20, body.length, true);
    head.setUint32(24, f.data.length, true);
    head.setUint16(28, name.length, true);
    head.setUint32(42, offset, true);
    parts.push(new Uint8Array(local.buffer), name, body);
    central.push(new Uint8Array(head.buffer), name);
    offset += 30 + name.length + body.length;
  }
  const cdSize = central.reduce((n, c) => n + c.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, files.length, true);
  end.setUint16(10, files.length, true);
  end.setUint32(12, cdSize, true);
  end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, new Uint8Array(end.buffer)] as Uint8Array<ArrayBuffer>[], { type: 'application/zip' });
}
