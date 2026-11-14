// Writes the examples gallery of the landing page (src/site/partials/ejemplos*.html) from the real answers
// in src/site/showcase.json. Model text is HTML-escaped; Markdown is rendered with the app's own parser.
//   node tools/showcase/partial.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { parse, inline } from '../../src/app/markdown.ts';

const data = JSON.parse(readFileSync('src/site/showcase.json', 'utf8'));
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

function inl(list) {
  return list
    .map((n) => {
      switch (n.t) {
        case 'text': return esc(n.v);
        case 'code': return `<code>${esc(n.v)}</code>`;
        case 'br': return '<br />';
        case 'strong': return `<strong>${inl(n.c)}</strong>`;
        case 'em': return `<em>${inl(n.c)}</em>`;
        case 'del': return `<del>${inl(n.c)}</del>`;
        case 'link': return inl(n.c);
      }
      return '';
    })
    .join('');
}
function blocks(list) {
  return list
    .map((b) => {
      switch (b.t) {
        case 'p': return `<p>${inl(b.c)}</p>`;
        case 'h': return `<p><strong>${inl(b.c)}</strong></p>`;
        case 'code': return `<pre><code>${esc(b.v)}</code></pre>`;
        case 'hr': return '';
        case 'quote': return blocks(b.c);
        case 'list': return `<${b.ordered ? 'ol' : 'ul'}>${b.items.map((it) => `<li>${it.length === 1 && it[0].t === 'p' ? inl(it[0].c) : blocks(it)}</li>`).join('')}</${b.ordered ? 'ol' : 'ul'}>`;
        case 'table': return `<table><thead><tr>${b.head.map((c) => `<th>${inl(c)}</th>`).join('')}</tr></thead><tbody>${b.rows.map((r) => `<tr>${b.head.map((_, i) => `<td>${inl(r[i] ?? [])}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
      }
      return '';
    })
    .join('');
}

for (const lang of ['es', 'en']) {
  const app = lang === 'es' ? '/app/' : '/en/app/';
  const go = lang === 'es' ? 'Probar en el chat' : 'Try it in the chat';
  const cards = data[lang]
    .map(
      (e) => `      <a class="ex glass" data-glow href="${app}?q=${encodeURIComponent(e.prompt)}">
        <span class="ex-kind"><i data-icon="${e.icon}" data-size="16"></i>${esc(e.title)}</span>
        <p class="ex-q">${esc(e.prompt)}</p>
        <div class="ex-a">${blocks(parse(e.answer))}</div>
        <span class="ex-go">${go}<i data-icon="chevronRight" data-size="15"></i></span>
      </a>`,
    )
    .join('\n');
  const prev = lang === 'es' ? 'Anterior' : 'Previous';
  const next = lang === 'es' ? 'Siguiente' : 'Next';
  const html = `<div class="examples" data-reveal>
    <div class="ex-track" id="ex-track">
${cards}
    </div>
    <div class="ex-nav">
      <button class="btn" type="button" data-ex="-1" aria-label="${prev}"><i data-icon="chevronLeft" data-size="18"></i></button>
      <button class="btn" type="button" data-ex="1" aria-label="${next}"><i data-icon="chevronRight" data-size="18"></i></button>
    </div>
  </div>
`;
  writeFileSync(`src/site/partials/ejemplos${lang === 'es' ? '' : '-en'}.html`, html);
}
console.log('ok');
