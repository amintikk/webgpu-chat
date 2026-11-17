// Files written by the model: each code block of an answer can be saved as a file, and all of them as one zip.
// The file's path is taken from where models usually write it: the fence (```php config/db.php), the line just
// before the block (**config/db.php**, `index.html`, ### styles.css, "Archivo: app.js:"), or a comment on the
// block's first line (// app.js, <!-- index.html -->). Without one, the name comes from the language.
import { parse, type Block, type Inline } from './markdown.ts';

export interface CodeFile {
  /** Path inside the zip / name of the download. */
  path: string;
  /** The path was written by the model (not made up from the language). */
  named: boolean;
  text: string;
  lang: string;
}

const EXT: Record<string, string> = {
  php: 'php', javascript: 'js', js: 'js', mjs: 'mjs', jsx: 'jsx', typescript: 'ts', ts: 'ts', tsx: 'tsx', html: 'html', htm: 'html',
  xml: 'xml', svg: 'svg', css: 'css', scss: 'scss', sass: 'sass', less: 'less', python: 'py', py: 'py', json: 'json', jsonc: 'json',
  sql: 'sql', bash: 'sh', sh: 'sh', shell: 'sh', zsh: 'sh', powershell: 'ps1', ps1: 'ps1', bat: 'bat', yaml: 'yml', yml: 'yml',
  toml: 'toml', ini: 'ini', markdown: 'md', md: 'md', java: 'java', kotlin: 'kt', kt: 'kt', c: 'c', h: 'h', cpp: 'cpp',
  'c++': 'cpp', hpp: 'hpp', csharp: 'cs', cs: 'cs', go: 'go', rust: 'rs', rs: 'rs', ruby: 'rb', rb: 'rb', swift: 'swift',
  dart: 'dart', lua: 'lua', r: 'r', vue: 'vue', svelte: 'svelte', twig: 'twig', blade: 'blade.php', env: 'env', csv: 'csv',
  dockerfile: 'Dockerfile', makefile: 'Makefile', diff: 'diff', txt: 'txt', text: 'txt', plaintext: 'txt',
};
const KNOWN_EXT = new Set([...Object.values(EXT), 'htaccess', 'gitignore', 'lock', 'cjs', 'cfg', 'conf', 'properties', 'gradle', 'pl', 'tex', 'srt', 'log']);
/** Blocks that are commands to run, not files to save (left out of the zip unless the model named them). */
const SHELL = new Set(['bash', 'sh', 'shell', 'zsh', 'console', 'powershell', 'ps1', 'cmd', 'bat', 'terminal']);

/** A file path with no spaces and a known extension (or a well-known extensionless name). */
const PATH = /^(?:\.{0,2}\/)?(?:[\w@.+-]+\/)*(?:[\w@+-][\w@.+-]*\.([A-Za-z][\w.]{0,9})|Dockerfile|Makefile|\.env|\.htaccess|\.gitignore)$/;

function asPath(s: string): string | null {
  const v = s.trim().replace(/^["'`*]+|["'`*:.,;)]+$/g, '');
  const m = PATH.exec(v);
  if (!m) return null;
  if (m[1] && !KNOWN_EXT.has(m[1].toLowerCase().split('.').pop()!)) return null;
  return v;
}

const plain = (list: Inline[]): string => list.map((n) => ('v' in n ? n.v : 'c' in n ? plain(n.c) : ' ')).join('');

/** A path written in a paragraph or heading: inline code or bold first (the last one), then the last word. */
function pathIn(list: Inline[]): string | null {
  const marked: string[] = [];
  const walk = (l: Inline[]) => {
    for (const n of l) {
      if (n.t === 'code') marked.push(n.v);
      else if (n.t === 'strong' || n.t === 'em') {
        marked.push(plain(n.c));
        walk(n.c);
      }
    }
  };
  walk(list);
  for (const c of marked.reverse()) {
    const p = asPath(c);
    if (p) return p;
  }
  const words = plain(list).trim().split(/\s+/);
  // Only a short line naming the file ("Archivo config/db.php:"), not any sentence that mentions one.
  return words.length <= 6 ? asPath(words.at(-1) ?? '') : null;
}

function pathInFence(info: string | undefined): string | null {
  if (!info) return null;
  const m = /(?:title|file|filename|name)=["']?([^"'\s]+)/i.exec(info);
  return asPath(m ? m[1]! : info.replace(/^:/, '').split(/\s+/)[0]!);
}

const COMMENT = /^\s*(?:\/\/|#|--|;|\/\*+|<!--|\{\{--)\s*(?:(?:file(?:name)?|archivo|ruta|path)\s*:\s*)?(\S+)\s*(?:\*\/|-->|--\}\})?\s*$/i;
function pathInComment(code: string): string | null {
  const first = code.split('\n', 1)[0] ?? '';
  // A PHP file starts with <?php; its name is often on the next line.
  const line = /^<\?php\s*$/.test(first) ? (code.split('\n', 2)[1] ?? '') : first;
  const m = COMMENT.exec(line);
  return m ? asPath(m[1]!) : null;
}

/** Strips what would escape the zip or break a file name: leading "/", "..", and characters Windows refuses. */
export function safePath(p: string): string {
  return p
    .replace(/\\/g, '/')
    .split('/')
    .filter((s) => s && s !== '.' && s !== '..')
    .map((s) => s.replace(/[<>:"|?*\u0000-\u001f]/g, '_'))
    .join('/');
}

/** Every finished code block of an answer, in order, with its file path. The same path twice: the last wins. */
export function codeFiles(blocks: Block[]): Map<Extract<Block, { t: 'code' }>, CodeFile> {
  const out = new Map<Extract<Block, { t: 'code' }>, CodeFile>();
  let unnamed = 0;
  const walk = (list: Block[]) => {
    let prev: Block | undefined;
    for (const b of list) {
      if (b.t === 'code') {
        const fromPrev = prev && (prev.t === 'p' || prev.t === 'h') ? pathIn(prev.c) : null;
        const named = pathInFence(b.info) ?? fromPrev ?? pathInComment(b.v);
        const ext = EXT[b.lang] ?? (b.lang || 'txt');
        const path = named ? safePath(named) : /^[A-Z]/.test(ext) ? ext : `archivo-${++unnamed}.${ext}`;
        out.set(b, { path: path || `archivo-${++unnamed}.txt`, named: !!named, text: b.v.endsWith('\n') ? b.v : b.v + '\n', lang: b.lang });
      } else if (b.t === 'list') b.items.forEach(walk);
      else if (b.t === 'quote') walk(b.c);
      prev = b;
    }
  };
  walk(blocks);
  return out;
}

/** The files that go in an answer's zip: the named ones, plus unnamed code that is not a shell command. */
export function zipFiles(markdown: string): CodeFile[] {
  const byPath = new Map<string, CodeFile>();
  for (const [b, f] of codeFiles(parse(markdown))) {
    if (b.open || !f.text.trim()) continue;
    if (!f.named && SHELL.has(f.lang)) continue;
    byPath.set(f.path, f);
  }
  return [...byPath.values()];
}
