// Markdown parser for model answers. Produces a small AST (testable without a DOM) that render.ts
// turns into DOM nodes. It must cope with text that is still streaming: an unclosed code fence is
// rendered as an open code block, an unclosed `**` as plain text.

export type Inline =
  | { t: 'text'; v: string }
  | { t: 'code'; v: string }
  | { t: 'strong' | 'em' | 'del'; c: Inline[] }
  | { t: 'link'; href: string; c: Inline[] }
  | { t: 'br' };

export type Block =
  | { t: 'p'; c: Inline[] }
  | { t: 'h'; level: number; c: Inline[] }
  | { t: 'code'; lang: string; v: string; open: boolean; info?: string }
  | { t: 'list'; ordered: boolean; start: number; items: Block[][] }
  | { t: 'quote'; c: Block[] }
  | { t: 'table'; align: ('left' | 'center' | 'right' | null)[]; head: Inline[][]; rows: Inline[][][] }
  | { t: 'hr' };

const FENCE = /^ {0,3}(`{3,}|~{3,})\s*([\w#+.-]*)(.*)$/;
const HEADING = /^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const HR = /^ {0,3}([-*_])(\s*\1){2,}\s*$/;
const BULLET = /^(\s*)([-*+])\s+(.*)$/;
const ORDERED = /^(\s*)(\d{1,9})[.)]\s+(.*)$/;
const QUOTE = /^ {0,3}>\s?(.*)$/;
const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

export function parse(src: string): Block[] {
  return parseLines(src.replace(/\r\n?/g, '\n').split('\n'));
}

function parseLines(lines: string[]): Block[] {
  const out: Block[] = [];
  let para: string[] = [];
  const flush = () => {
    if (para.length) out.push({ t: 'p', c: inline(para.join('\n').trim()) });
    para = [];
  };
  let i = 0;
  while (i < lines.length) {
    const line = lines[i]!;
    let m: RegExpExecArray | null;
    if ((m = FENCE.exec(line))) {
      flush();
      const fence = m[1]!;
      const indent = line.length - line.trimStart().length;
      const body: string[] = [];
      i++;
      let open = true;
      while (i < lines.length) {
        const l = lines[i]!;
        if (l.trim().startsWith(fence[0]!.repeat(fence.length)) && l.trim().replace(new RegExp(`^\\${fence[0]}+`), '').trim() === '') {
          open = false;
          i++;
          break;
        }
        body.push(indent ? l.replace(new RegExp(`^ {0,${indent}}`), '') : l);
        i++;
      }
      // What follows the language (```php config/db.php, ```js title="app.js") can name the file.
      const info = m[3]!.trim();
      out.push({ t: 'code', lang: m[2]!.toLowerCase(), v: body.join('\n'), open, ...(info ? { info } : {}) });
      continue;
    }
    if (!line.trim()) {
      flush();
      i++;
      continue;
    }
    if ((m = HEADING.exec(line))) {
      flush();
      out.push({ t: 'h', level: m[1]!.length, c: inline(m[2]!) });
      i++;
      continue;
    }
    if (HR.test(line) && !para.length) {
      flush();
      out.push({ t: 'hr' });
      i++;
      continue;
    }
    if (QUOTE.test(line)) {
      flush();
      const body: string[] = [];
      while (i < lines.length && (m = QUOTE.exec(lines[i]!))) {
        body.push(m[1]!);
        i++;
      }
      out.push({ t: 'quote', c: parseLines(body) });
      continue;
    }
    if (line.includes('|') && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1]!) && lines[i + 1]!.includes('-')) {
      flush();
      const head = cells(line);
      const align = cells(lines[i + 1]!).map((c) => {
        const l = c.startsWith(':');
        const r = c.endsWith(':');
        return l && r ? 'center' : r ? 'right' : l ? 'left' : null;
      });
      i += 2;
      const rows: Inline[][][] = [];
      while (i < lines.length && lines[i]!.includes('|') && lines[i]!.trim()) {
        rows.push(cells(lines[i]!).map(inline));
        i++;
      }
      out.push({ t: 'table', align, head: head.map(inline), rows });
      continue;
    }
    if (BULLET.test(line) || ORDERED.test(line)) {
      flush();
      const r = parseList(lines, i);
      out.push(r.block);
      i = r.next;
      continue;
    }
    para.push(line);
    i++;
  }
  flush();
  return out;
}

function cells(line: string): string[] {
  let s = line.trim();
  if (s.startsWith('|')) s = s.slice(1);
  if (s.endsWith('|') && !s.endsWith('\\|')) s = s.slice(0, -1);
  const out: string[] = [];
  let cur = '';
  let code = false;
  for (let k = 0; k < s.length; k++) {
    const ch = s[k]!;
    if (ch === '`') code = !code;
    if (ch === '\\' && s[k + 1] === '|') {
      cur += '|';
      k++;
    } else if (ch === '|' && !code) {
      out.push(cur.trim());
      cur = '';
    } else cur += ch;
  }
  out.push(cur.trim());
  return out;
}

function parseList(lines: string[], start: number): { block: Block; next: number } {
  const first = (BULLET.exec(lines[start]!) ?? ORDERED.exec(lines[start]!))!;
  const ordered = !BULLET.test(lines[start]!);
  const baseIndent = first[1]!.length;
  const items: string[][] = [];
  let i = start;
  while (i < lines.length) {
    const line = lines[i]!;
    const m = ordered ? ORDERED.exec(line) : BULLET.exec(line);
    if (m && m[1]!.length === baseIndent) {
      items.push([m[3]!]);
      i++;
      continue;
    }
    if (!line.trim()) {
      // A blank line continues the list only if the next line is indented or another item.
      const nxt = lines[i + 1];
      if (nxt !== undefined && (indentOf(nxt) > baseIndent || (ordered ? ORDERED : BULLET).exec(nxt)?.[1]?.length === baseIndent)) {
        items.at(-1)?.push('');
        i++;
        continue;
      }
      break;
    }
    if (indentOf(line) > baseIndent && items.length) {
      items.at(-1)!.push(line.slice(Math.min(indentOf(line), baseIndent + (ordered ? 3 : 2))));
      i++;
      continue;
    }
    // Lazy continuation of a paragraph inside the item.
    if (items.length && !FENCE.test(line) && !HEADING.test(line) && !BULLET.test(line) && !ORDERED.test(line) && !QUOTE.test(line)) {
      items.at(-1)!.push(line);
      i++;
      continue;
    }
    break;
  }
  const startNum = ordered ? Number(first[2]) : 1;
  return { block: { t: 'list', ordered, start: startNum, items: items.map((it) => parseLines(it)) }, next: i };
}

const indentOf = (l: string) => l.length - l.trimStart().length;

// Inline ---------------------------------------------------------------------------------------

const SAFE_URL = /^(https?:\/\/|mailto:)/i;

export function inline(src: string): Inline[] {
  const out: Inline[] = [];
  let text = '';
  const push = (n: Inline) => {
    if (text) out.push({ t: 'text', v: text });
    text = '';
    out.push(n);
  };
  let i = 0;
  while (i < src.length) {
    const ch = src[i]!;
    const rest = src.slice(i);
    if (ch === '\\' && i + 1 < src.length && /[\\`*_{}[\]()#+\-.!|~<>]/.test(src[i + 1]!)) {
      text += src[i + 1];
      i += 2;
      continue;
    }
    if (ch === '`') {
      const n = /^`+/.exec(rest)![0].length;
      const end = src.indexOf('`'.repeat(n), i + n);
      if (end > 0) {
        push({ t: 'code', v: src.slice(i + n, end).replace(/^ (.+) $/, '$1') });
        i = end + n;
        continue;
      }
    }
    if (ch === '\n') {
      push({ t: 'br' });
      i++;
      continue;
    }
    if ((ch === '*' || ch === '_' || ch === '~') && (rest.startsWith(ch + ch) || (ch !== '~' && rest.length > 1))) {
      const dbl = rest.startsWith(ch + ch);
      if (ch === '~' && !dbl) {
        text += ch;
        i++;
        continue;
      }
      const mark = dbl ? ch + ch : ch;
      // Intraword underscores (snake_case) are not emphasis.
      const prev = src[i - 1];
      if (ch === '_' && prev && /\w/.test(prev)) {
        text += mark;
        i += mark.length;
        continue;
      }
      const end = findClose(src, i + mark.length, mark);
      if (end > i + mark.length && !/\s/.test(src[i + mark.length]!)) {
        const kind = ch === '~' ? 'del' : dbl ? 'strong' : 'em';
        push({ t: kind, c: inline(src.slice(i + mark.length, end)) });
        i = end + mark.length;
        continue;
      }
      text += mark;
      i += mark.length;
      continue;
    }
    if (ch === '[') {
      const m = /^\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)/.exec(rest);
      if (m && SAFE_URL.test(m[2]!)) {
        push({ t: 'link', href: m[2]!, c: inline(m[1]!) });
        i += m[0].length;
        continue;
      }
    }
    if (ch === 'h' && (rest.startsWith('http://') || rest.startsWith('https://')) && !/\w/.test(src[i - 1] ?? '')) {
      const m = /^https?:\/\/[^\s<>()]+[^\s<>().,;:!?'"]/.exec(rest);
      if (m) {
        push({ t: 'link', href: m[0], c: [{ t: 'text', v: m[0] }] });
        i += m[0].length;
        continue;
      }
    }
    text += ch;
    i++;
  }
  if (text) out.push({ t: 'text', v: text });
  return out;
}

function findClose(src: string, from: number, mark: string): number {
  let i = from;
  while (i < src.length) {
    if (src[i] === '`') {
      const end = src.indexOf('`', i + 1);
      if (end < 0) return -1;
      i = end + 1;
      continue;
    }
    if (src.startsWith(mark, i) && !/\s/.test(src[i - 1]!)) {
      // `**` must not close a single `*`, and vice versa.
      if (mark.length === 1 && src[i + 1] === mark) {
        i += 2;
        continue;
      }
      return i;
    }
    i++;
  }
  return -1;
}
