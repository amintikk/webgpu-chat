// Attached files: read in the browser (text files as they are, PDFs with pdf.js) and added to the message as
// plain text; pictures are kept as images for the models that read them. Nothing is uploaded; pdf.js is
// loaded only the first time a PDF is attached.
import { t } from '../i18n.ts';
import type { StoredImage } from './db.ts';
import { archiveText, listZip, readEntry, skippedDir, skippedFile, type ArchiveItem } from './zip.ts';

export interface Attachment {
  name: string;
  size: number;
  text: string;
  /** Set for a zip archive: how many files it holds (its text is already the whole <archive> block). */
  entries?: number;
}

const TEXT_EXT =
  'txt md markdown csv tsv json jsonl xml html htm css scss js mjs cjs ts tsx jsx py ipynb java kt c h cpp hpp cs go rs rb php swift sh bash zsh ps1 bat yml yaml toml ini cfg conf env sql log tex srt vtt rst org r m lua pl dart vue svelte'.split(' ');

const IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif', 'image/bmp', 'image/avif'];

/** Value for the file input's `accept`. */
export const ACCEPT = ['.pdf', '.zip', 'application/zip', 'application/x-zip-compressed', ...TEXT_EXT.map((e) => '.' + e), 'text/*', 'application/json', 'application/pdf', ...IMAGE_TYPES, '.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.avif'].join(',');

export const isImageFile = (f: File) => IMAGE_TYPES.includes(f.type) || /\.(png|jpe?g|webp|gif|bmp|avif)$/i.test(f.name);

const MAX_IMAGE_BYTES = 40 * 1024 * 1024;
/** Longest side kept: the models resize further (Qwen to at most ~0.65 MP, Gemma to 280 patches of 48 px). */
const MAX_IMAGE_SIDE = 2048;

/** Reads a picture: decoded here, downscaled if very large, and kept as a PNG or JPEG. */
export async function readImage(file: File): Promise<StoredImage> {
  if (file.size > MAX_IMAGE_BYTES) throw new Error(t('«{f}» pesa demasiado (máximo {n} MB).', { f: file.name, n: MAX_IMAGE_BYTES >> 20 }));
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    throw new Error(t('«{f}» no es una imagen que el navegador pueda abrir.', { f: file.name }));
  }
  const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(bitmap.width, bitmap.height));
  let blob: Blob = file;
  // Re-encode when it must shrink, or when the format is one the model's reader may not decode (e.g. AVIF, GIF).
  if (scale < 1 || !['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
    const w = Math.max(1, Math.round(bitmap.width * scale));
    const hgt = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = new OffscreenCanvas(w, hgt);
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(bitmap, 0, 0, w, hgt);
    blob = await canvas.convertToBlob(file.type === 'image/png' ? { type: 'image/png' } : { type: 'image/jpeg', quality: 0.92 });
  }
  bitmap.close();
  const name = file.name && file.name !== 'image.png' ? file.name : t('Imagen pegada');
  return { id: crypto.randomUUID(), name, size: blob.size, blob };
}

const MAX_TEXT_BYTES = 2 * 1024 * 1024;
const MAX_PDF_BYTES = 30 * 1024 * 1024;
const MAX_PDF_PAGES = 300;

const ext = (name: string) => name.toLowerCase().split('.').pop() ?? '';
const isPdf = (f: File) => f.type === 'application/pdf' || ext(f.name) === 'pdf';
export const isZipFile = (f: File) => /zip/.test(f.type) || ext(f.name) === 'zip';
const isText = (f: File) => f.type.startsWith('text/') || /json|xml|javascript|typescript|x-sh|x-python|yaml|toml|sql/.test(f.type) || TEXT_EXT.includes(ext(f.name));

/** Reads a file into text, or throws an Error whose message can be shown as is. */
export async function readAttachment(file: File): Promise<Attachment> {
  if (isPdf(file)) {
    if (file.size > MAX_PDF_BYTES) throw new Error(t('«{f}» pesa demasiado (máximo {n} MB).', { f: file.name, n: MAX_PDF_BYTES >> 20 }));
    const text = await pdfText(file);
    if (!text.trim()) throw new Error(t('«{f}» no tiene texto que se pueda leer (¿es un escaneo?).', { f: file.name }));
    return { name: file.name, size: file.size, text };
  }
  if (!isText(file)) throw new Error(t('«{f}»: solo se pueden adjuntar imágenes, PDF, ZIP y archivos de texto o código.', { f: file.name }));
  if (file.size > MAX_TEXT_BYTES) throw new Error(t('«{f}» pesa demasiado (máximo {n} MB).', { f: file.name, n: MAX_TEXT_BYTES >> 20 }));
  const text = await file.text();
  if (text.includes('\u0000')) throw new Error(t('«{f}» no es un archivo de texto.', { f: file.name }));
  return { name: file.name, size: file.size, text: text.replace(/\r\n?/g, '\n') };
}

const MAX_ZIP_BYTES = 100 * 1024 * 1024;
/** A text file inside a zip larger than this is listed without its contents. */
const MAX_ZIP_FILE_TEXT = 300 * 1024;
/** Pictures inside a zip that are attached for the models that see (each one costs hundreds of tokens). */
const MAX_ZIP_IMAGES = 4;
const MAX_ZIP_PDFS = 3;
const IMAGE_EXT = /\.(png|jpe?g|webp|gif|bmp|avif)$/i;
/** Extensions that are never text, skipped without opening them. */
const BINARY_EXT =
  /\.(ico|tiff?|psd|ai|heic|mp3|wav|ogg|flac|m4a|mp4|mov|avi|mkv|webm|woff2?|ttf|otf|eot|exe|dll|so|dylib|bin|o|a|class|jar|war|pyc|wasm|onnx|zip|rar|7z|gz|tgz|bz2|xz|tar|iso|dmg|apk|docx?|xlsx?|pptx?|odt|ods|sqlite|db|mdb)$/i;

/**
 * Reads a zip: its folder tree plus every text and code file, as one attachment; PDFs inside are read as text
 * too, and up to a few pictures are returned separately (only when `withImages`, for models that see them).
 */
export async function readZip(file: File, withImages: boolean): Promise<{ file: Attachment; images: StoredImage[] }> {
  if (file.size > MAX_ZIP_BYTES) throw new Error(t('«{f}» pesa demasiado (máximo {n} MB).', { f: file.name, n: MAX_ZIP_BYTES >> 20 }));
  const bytes = new Uint8Array(await file.arrayBuffer());
  let entries;
  try {
    entries = listZip(bytes);
  } catch {
    throw new Error(t('«{f}» no es un ZIP válido o está dañado.', { f: file.name }));
  }
  const items: ArchiveItem[] = [];
  const images: StoredImage[] = [];
  const collapsed = new Set<string>();
  const utf8 = new TextDecoder('utf-8', { fatal: true });
  let textBytes = 0;
  let pdfs = 0;
  for (const e of entries) {
    if (e.path.endsWith('/')) continue;
    const skip = skippedDir(e.path);
    if (skip) {
      if (!collapsed.has(skip.dir)) items.push({ path: skip.dir, note: `${skip.why}, not read` });
      collapsed.add(skip.dir);
      continue;
    }
    const item: ArchiveItem = { path: e.path };
    items.push(item);
    const name = e.path.split('/').at(-1)!;
    try {
      if (e.encrypted) item.note = 'encrypted, not read';
      else if (skippedFile(e.path)) item.note = 'generated, not read';
      else if (IMAGE_EXT.test(name)) {
        if (withImages && images.length < MAX_ZIP_IMAGES && e.size <= MAX_IMAGE_BYTES) {
          const data = await readEntry(bytes, e, MAX_IMAGE_BYTES);
          images.push(await readImage(new File([data as Uint8Array<ArrayBuffer>], name, { type: imageType(name) })));
          item.note = 'image, attached';
        } else item.note = withImages ? 'image, not attached (limit)' : 'image';
      } else if (/\.pdf$/i.test(name)) {
        if (pdfs < MAX_ZIP_PDFS && e.size <= MAX_PDF_BYTES && textBytes < MAX_TEXT_BYTES) {
          const text = await pdfText(new File([(await readEntry(bytes, e, MAX_PDF_BYTES)) as Uint8Array<ArrayBuffer>], name, { type: 'application/pdf' }));
          pdfs++;
          if (text.trim()) {
            item.text = text;
            textBytes += text.length;
          } else item.note = 'PDF without text';
        } else item.note = 'PDF, not read (limit)';
      } else if (BINARY_EXT.test(name)) item.note = 'binary';
      else if (e.size > MAX_ZIP_FILE_TEXT) item.note = `${Math.round(e.size / 1024)} KB, too large to read`;
      else if (textBytes + e.size > MAX_TEXT_BYTES) item.note = 'not read (attachment limit)';
      else {
        const data = await readEntry(bytes, e, MAX_ZIP_FILE_TEXT);
        // Text is valid UTF-8 with no NUL bytes; anything else is a binary file whatever its name says.
        if (data.includes(0)) item.note = 'binary';
        else {
          item.text = utf8.decode(data).replace(/\r\n?/g, '\n');
          textBytes += data.length;
        }
      }
    } catch {
      // Not valid UTF-8, a picture the browser cannot decode, a damaged or oversized entry…
      if (item.text == null) item.note = 'binary or unreadable';
    }
  }
  const files = entries.filter((e) => !e.path.endsWith('/')).length;
  if (!files) throw new Error(t('«{f}» está vacío.', { f: file.name }));
  // Counted as the model sees them: a collapsed folder such as .git/ is one line, not its thousands of files.
  const shown = items.filter((i) => !i.path.endsWith('/')).length;
  return { file: { name: file.name, size: file.size, text: archiveText(file.name, items), entries: shown }, images };
}

function imageType(name: string): string {
  const e = ext(name);
  return e === 'jpg' ? 'image/jpeg' : `image/${e}`;
}

async function pdfText(file: File): Promise<string> {
  const [pdfjs, worker] = await Promise.all([import('pdfjs-dist'), import('pdfjs-dist/build/pdf.worker.min.mjs?url')]);
  pdfjs.GlobalWorkerOptions.workerSrc = worker.default;
  const doc = await pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()), useSystemFonts: true }).promise;
  try {
    const pages: string[] = [];
    for (let i = 1; i <= Math.min(doc.numPages, MAX_PDF_PAGES); i++) {
      const page = await doc.getPage(i);
      const content = await page.getTextContent();
      let line = '';
      const lines: string[] = [];
      for (const item of content.items) {
        if (!('str' in item)) continue;
        line += item.str;
        if (item.hasEOL) {
          lines.push(line.trimEnd());
          line = '';
        }
      }
      if (line) lines.push(line.trimEnd());
      pages.push(lines.join('\n').replace(/\n{3,}/g, '\n\n').trim());
      page.cleanup();
    }
    return pages.filter(Boolean).join('\n\n');
  } finally {
    void doc.destroy();
  }
}

/** The text the model reads for a user message: each file, then what the user wrote. */
export function withAttachments(content: string, files: Attachment[] | undefined): string {
  if (!files?.length) return content;
  const blocks = files.map((f) => f.entries != null ? f.text.trim() : `<file name="${f.name.replace(/"/g, "'")}">\n${f.text.trim()}\n</file>`);
  return [...blocks, content].filter(Boolean).join('\n\n');
}
