// Generates the guide and model pages from tools/seo/content.mjs, the list of pages for Vite
// (tools/seo/pages.json) and public/sitemap.xml (every page with its es/en alternates).
//   node tools/seo/build-pages.mjs
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PAGES } from './content.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '../..');
const SITE = 'https://llm.aminox.dev';
const TODAY = new Date().toISOString().slice(0, 10);
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const strip = (s) => s.replace(/<[^>]+>/g, '');
const byId = Object.fromEntries(PAGES.map((p) => [p.id, p]));

function page(p, lang) {
  const c = p[lang];
  const other = p[lang === 'es' ? 'en' : 'es'];
  const es = p.es.path;
  const en = p.en.path;
  const home = lang === 'es' ? '/' : '/en/';
  const T = lang === 'es'
    ? { home: 'Inicio', facts: 'Datos', faq: 'Preguntas frecuentes', more: 'Sigue leyendo', note: 'Son modelos pequeños y pueden equivocarse. Comprueba lo importante.' }
    : { home: 'Home', facts: 'Facts', faq: 'Frequently asked questions', more: 'Keep reading', note: 'These are small models and can be wrong. Check what matters.' };
  const ld = [
    {
      '@context': 'https://schema.org',
      '@type': 'WebPage',
      name: c.title,
      description: c.description,
      url: SITE + c.path,
      inLanguage: lang,
      isPartOf: { '@type': 'WebSite', name: 'WebGPU Chat', url: SITE + home },
      dateModified: TODAY,
      author: { '@type': 'Person', name: 'Amin Tahtah Toumi' },
    },
    {
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'WebGPU Chat', item: SITE + home },
        { '@type': 'ListItem', position: 2, name: c.crumb, item: SITE + c.path },
      ],
    },
    {
      '@context': 'https://schema.org',
      '@type': 'FAQPage',
      mainEntity: c.faq.map(([q, a]) => ({ '@type': 'Question', name: q, acceptedAnswer: { '@type': 'Answer', text: strip(a) } })),
    },
  ];
  const sections = c.sections
    .map((s) => {
      const list = s.ul ? `<ul>${s.ul.map((x) => `<li>${x}</li>`).join('')}</ul>` : s.ol ? `<ol>${s.ol.map((x) => `<li>${x}</li>`).join('')}</ol>` : '';
      const para = s.p ? `<p>${s.p}</p>` : '';
      return `        <section data-reveal>
          <h2>${s.h}</h2>
          ${s.ul || s.ol ? list + para : para}
        </section>`;
    })
    .join('\n');
  const facts = c.facts
    ? `        <dl class="facts glass" data-reveal>${c.facts.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('')}</dl>\n`
    : '';
  const related = c.related
    .map((id) => byId[id][lang])
    .map((r) => `<a class="rel glass" data-glow href="${r.path}"><strong>${r.title}</strong><span>${r.description.split('. ')[0]}.</span></a>`)
    .join('');
  return `<!doctype html>
<html lang="${lang}">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${esc(c.metaTitle)}</title>
    <meta name="description" content="${esc(c.description)}" />
    <link rel="canonical" href="${SITE}${c.path}" />
    <link rel="alternate" hreflang="es" href="${SITE}${es}" />
    <link rel="alternate" hreflang="en" href="${SITE}${en}" />
    <link rel="alternate" hreflang="x-default" href="${SITE}${en}" />
    <meta property="og:type" content="article" />
    <meta property="og:site_name" content="WebGPU Chat" />
    <meta property="og:title" content="${esc(c.title)}" />
    <meta property="og:description" content="${esc(c.description)}" />
    <meta property="og:url" content="${SITE}${c.path}" />
    <meta property="og:image" content="${SITE}/og-${lang}.png" />
    <meta property="og:image:width" content="1200" />
    <meta property="og:image:height" content="630" />
    <meta property="og:locale" content="${lang === 'es' ? 'es_ES' : 'en_US'}" />
    <meta property="og:locale:alternate" content="${lang === 'es' ? 'en_US' : 'es_ES'}" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="${esc(c.title)}" />
    <meta name="twitter:description" content="${esc(c.description)}" />
    <meta name="twitter:image" content="${SITE}/og-${lang}.png" />
    <meta name="color-scheme" content="light dark" />
    <link rel="icon" href="/favicon.ico" sizes="any" />
    <link rel="icon" type="image/svg+xml" href="/icon.svg" />
    <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png" />
    <link rel="manifest" href="/site.webmanifest" />
    <script src="/theme.js"></script>
    <script type="application/ld+json">${JSON.stringify(ld)}</script>
  </head>
  <body class="site">
    <!--#header${lang === 'en' ? '-en' : ''}-->
    <main class="doc article">
      <div class="wrap narrow">
        <nav class="crumbs" aria-label="${lang === 'es' ? 'Ruta' : 'Breadcrumb'}"><a href="${home}">${T.home}</a><span aria-hidden="true">›</span><span>${c.crumb}</span></nav>
        <h1 data-reveal>${c.title}</h1>
        <p class="lead" data-reveal>${c.lead}</p>
${facts}${sections}
        <div class="cta-box glass" data-reveal>
          <a class="btn primary" href="${c.cta.href}">${c.cta.text}</a>
          <p>${T.note}</p>
        </div>
        <section class="faq" data-reveal>
          <h2>${T.faq}</h2>
${c.faq.map(([q, a]) => `          <details><summary>${q}</summary><p>${a}</p></details>`).join('\n')}
        </section>
        <section data-reveal>
          <h2>${T.more}</h2>
          <div class="related">${related}</div>
        </section>
      </div>
    </main>
    <!--#footer${lang === 'en' ? '-en' : ''}-->
    <script type="module" src="/src/site/site.ts"></script>
  </body>
</html>
`;
}

const generated = [];
for (const p of PAGES) {
  for (const lang of ['es', 'en']) {
    const rel = p[lang].path.replace(/^\//, '') + 'index.html';
    mkdirSync(join(root, dirname(rel)), { recursive: true });
    writeFileSync(join(root, rel), page(p, lang));
    generated.push(rel);
  }
}

// Every page of the site with its alternate, for Vite and the sitemap.
const core = [
  ['index.html', 'en/index.html', '/', '/en/', '1.0'],
  ['app/index.html', 'en/app/index.html', '/app/', '/en/app/', '0.9'],
  ...PAGES.map((p) => [p.es.path.slice(1) + 'index.html', p.en.path.slice(1) + 'index.html', p.es.path, p.en.path, p.model ? '0.8' : '0.8']),
  ['aviso-legal/index.html', 'en/legal-notice/index.html', '/aviso-legal/', '/en/legal-notice/', '0.2'],
  ['privacidad/index.html', 'en/privacy/index.html', '/privacidad/', '/en/privacy/', '0.3'],
  ['terminos/index.html', 'en/terms/index.html', '/terminos/', '/en/terms/', '0.2'],
  ['licencias/index.html', 'en/licenses/index.html', '/licencias/', '/en/licenses/', '0.3'],
];
writeFileSync(join(root, 'tools/seo/pages.json'), JSON.stringify(core.flatMap(([a, b]) => [a, b]), null, 2) + '\n');
const urls = core.flatMap(([, , es, en, prio]) =>
  [es, en].map(
    (u) => `  <url>
    <loc>${SITE}${u}</loc>
    <lastmod>${TODAY}</lastmod>
    <priority>${prio}</priority>
    <xhtml:link rel="alternate" hreflang="es" href="${SITE}${es}" />
    <xhtml:link rel="alternate" hreflang="en" href="${SITE}${en}" />
    <xhtml:link rel="alternate" hreflang="x-default" href="${SITE}${en}" />
  </url>`,
  ),
);
writeFileSync(
  join(root, 'public/sitemap.xml'),
  `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">\n${urls.join('\n')}\n</urlset>\n`,
);
console.log(`${generated.length} pages, ${urls.length} sitemap URLs`);
