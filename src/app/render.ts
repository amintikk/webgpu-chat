// Turns the Markdown AST into DOM. Model text only ever becomes text nodes; the one exception is
// highlight.js output, which is built from escaped text and only adds <span class="hljs-…">.
import hljs from 'highlight.js/lib/core';
import bash from 'highlight.js/lib/languages/bash';
import c from 'highlight.js/lib/languages/c';
import cpp from 'highlight.js/lib/languages/cpp';
import csharp from 'highlight.js/lib/languages/csharp';
import css from 'highlight.js/lib/languages/css';
import diff from 'highlight.js/lib/languages/diff';
import go from 'highlight.js/lib/languages/go';
import java from 'highlight.js/lib/languages/java';
import javascript from 'highlight.js/lib/languages/javascript';
import json from 'highlight.js/lib/languages/json';
import kotlin from 'highlight.js/lib/languages/kotlin';
import markdown from 'highlight.js/lib/languages/markdown';
import php from 'highlight.js/lib/languages/php';
import python from 'highlight.js/lib/languages/python';
import ruby from 'highlight.js/lib/languages/ruby';
import rust from 'highlight.js/lib/languages/rust';
import sql from 'highlight.js/lib/languages/sql';
import swift from 'highlight.js/lib/languages/swift';
import typescript from 'highlight.js/lib/languages/typescript';
import xml from 'highlight.js/lib/languages/xml';
import yaml from 'highlight.js/lib/languages/yaml';
import { t } from '../i18n.ts';
import { h } from '../ui/dom.ts';
import { icon } from '../ui/icons.ts';
import { parse, type Block, type Inline } from './markdown.ts';

const LANGS = { bash, c, cpp, csharp, css, diff, go, java, javascript, json, kotlin, markdown, php, python, ruby, rust, sql, swift, typescript, xml, yaml };
for (const [name, def] of Object.entries(LANGS)) hljs.registerLanguage(name, def);
hljs.registerAliases(['sh', 'shell', 'zsh', 'console'], { languageName: 'bash' });
hljs.registerAliases(['js', 'jsx', 'mjs'], { languageName: 'javascript' });
hljs.registerAliases(['ts', 'tsx'], { languageName: 'typescript' });
hljs.registerAliases(['py'], { languageName: 'python' });
hljs.registerAliases(['html', 'svg'], { languageName: 'xml' });
hljs.registerAliases(['yml'], { languageName: 'yaml' });
hljs.registerAliases(['cs'], { languageName: 'csharp' });
hljs.registerAliases(['rs'], { languageName: 'rust' });
hljs.registerAliases(['kt'], { languageName: 'kotlin' });
hljs.registerAliases(['c++', 'h', 'hpp'], { languageName: 'cpp' });

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const ta = h('textarea', { value: text, class: 'offscreen' });
    document.body.append(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  }
}

export function copyButton(get: () => string, label: string, cls = 'icon-btn'): HTMLButtonElement {
  const btn = h('button', { type: 'button', class: cls, title: label, 'aria-label': label }, icon('copy', 16));
  btn.addEventListener('click', async () => {
    if (await copyText(get())) {
      btn.replaceChildren(icon('check', 16));
      btn.classList.add('done');
      setTimeout(() => {
        btn.replaceChildren(icon('copy', 16));
        btn.classList.remove('done');
      }, 1400);
    }
  });
  return btn;
}

function inlines(list: Inline[]): Node[] {
  return list.map((n): Node => {
    switch (n.t) {
      case 'text':
        return document.createTextNode(n.v);
      case 'code':
        return h('code', {}, n.v);
      case 'br':
        return h('br');
      case 'strong':
        return h('strong', {}, ...inlines(n.c));
      case 'em':
        return h('em', {}, ...inlines(n.c));
      case 'del':
        return h('del', {}, ...inlines(n.c));
      case 'link':
        return h('a', { href: n.href, target: '_blank', rel: 'noopener noreferrer nofollow' }, ...inlines(n.c));
    }
  });
}

function codeBlock(b: Extract<Block, { t: 'code' }>): HTMLElement {
  const code = h('code', { class: 'hljs' });
  const lang = b.lang && hljs.getLanguage(b.lang) ? b.lang : '';
  if (lang && b.v.length < 60_000) {
    code.innerHTML = hljs.highlight(b.v, { language: lang, ignoreIllegals: true }).value;
  } else code.textContent = b.v;
  return h(
    'div',
    { class: 'code-block' },
    h('div', { class: 'code-head' }, h('span', {}, b.lang || t('texto')), copyButton(() => b.v, t('Copiar código'), 'code-copy')),
    h('pre', {}, code),
  );
}

function blocks(list: Block[]): Node[] {
  return list.map((b): Node => {
    switch (b.t) {
      case 'p':
        return h('p', {}, ...inlines(b.c));
      case 'h':
        return h(`h${Math.min(b.level + 1, 6)}` as 'h2', {}, ...inlines(b.c));
      case 'code':
        return codeBlock(b);
      case 'hr':
        return h('hr');
      case 'quote':
        return h('blockquote', {}, ...blocks(b.c));
      case 'list': {
        const el = b.ordered ? h('ol', { start: b.start !== 1 ? b.start : null }) : h('ul');
        for (const it of b.items) {
          // Tight items: a lone paragraph renders without <p> margins.
          const kids = it.length === 1 && it[0]!.t === 'p' ? inlines(it[0]!.c) : blocks(it);
          el.append(h('li', {}, ...kids));
        }
        return el;
      }
      case 'table': {
        const al = (i: number) => (b.align[i] ? `al-${b.align[i]}` : null);
        return h(
          'div',
          { class: 'table-wrap' },
          h(
            'table',
            {},
            h('thead', {}, h('tr', {}, ...b.head.map((c, i) => h('th', { class: al(i) }, ...inlines(c))))),
            h('tbody', {}, ...b.rows.map((r) => h('tr', {}, ...b.head.map((_, i) => h('td', { class: al(i) }, ...inlines(r[i] ?? [])))))),
          ),
        );
      }
    }
  });
}

export function renderMarkdown(src: string): DocumentFragment {
  const f = document.createDocumentFragment();
  f.append(...blocks(parse(src)));
  return f;
}
