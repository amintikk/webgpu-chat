// Attached files: read in the browser (text files as they are, PDFs with pdf.js) and added to the message as
// plain text. Nothing is uploaded; pdf.js is loaded only the first time a PDF is attached.
import { t } from '../i18n.ts';

export interface Attachment {
  name: string;
  size: number;
  text: string;
}

const TEXT_EXT =
  'txt md markdown csv tsv json jsonl xml html htm css scss js mjs cjs ts tsx jsx py ipynb java kt c h cpp hpp cs go rs rb php swift sh bash zsh ps1 bat yml yaml toml ini cfg conf env sql log tex srt vtt rst org r m lua pl dart vue svelte'.split(' ');

/** Value for the file input's `accept`. */
export const ACCEPT = ['.pdf', ...TEXT_EXT.map((e) => '.' + e), 'text/*', 'application/json', 'application/pdf'].join(',');

const MAX_TEXT_BYTES = 2 * 1024 * 1024;
const MAX_PDF_BYTES = 30 * 1024 * 1024;
const MAX_PDF_PAGES = 300;

const ext = (name: string) => name.toLowerCase().split('.').pop() ?? '';
const isPdf = (f: File) => f.type === 'application/pdf' || ext(f.name) === 'pdf';
const isText = (f: File) => f.type.startsWith('text/') || /json|xml|javascript|typescript|x-sh|x-python|yaml|toml|sql/.test(f.type) || TEXT_EXT.includes(ext(f.name));

/** Reads a file into text, or throws an Error whose message can be shown as is. */
export async function readAttachment(file: File): Promise<Attachment> {
  if (isPdf(file)) {
    if (file.size > MAX_PDF_BYTES) throw new Error(t('«{f}» pesa demasiado (máximo {n} MB).', { f: file.name, n: MAX_PDF_BYTES >> 20 }));
    const text = await pdfText(file);
    if (!text.trim()) throw new Error(t('«{f}» no tiene texto que se pueda leer (¿es un escaneo?).', { f: file.name }));
    return { name: file.name, size: file.size, text };
  }
  if (!isText(file)) throw new Error(t('«{f}»: solo se pueden adjuntar PDF y archivos de texto o código.', { f: file.name }));
  if (file.size > MAX_TEXT_BYTES) throw new Error(t('«{f}» pesa demasiado (máximo {n} MB).', { f: file.name, n: MAX_TEXT_BYTES >> 20 }));
  const text = await file.text();
  if (text.includes('\u0000')) throw new Error(t('«{f}» no es un archivo de texto.', { f: file.name }));
  return { name: file.name, size: file.size, text: text.replace(/\r\n?/g, '\n') };
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
  const blocks = files.map((f) => `<file name="${f.name.replace(/"/g, "'")}">\n${f.text.trim()}\n</file>`);
  return [...blocks, content].filter(Boolean).join('\n\n');
}
