import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { defineConfig, type Plugin } from 'vite';

const SITE = 'https://llm.aminox.dev';

/** Static pages share a header and a footer: `<!--#header-->` and `<!--#footer-->` are replaced at build time. */
function partials(): Plugin {
  const dir = resolve(__dirname, 'src/site/partials');
  return {
    name: 'partials',
    transformIndexHtml(html) {
      // The language switch links to this page's alternate, read from its own hreflang tags.
      const lang = /<html lang="(\w+)"/.exec(html)?.[1] ?? 'es';
      const other = lang === 'en' ? 'es' : 'en';
      const href = new RegExp(`<link rel="alternate" hreflang="${other}" href="${SITE.replace(/\./g, '\\.')}([^"]*)"`).exec(html)?.[1];
      return html.replace(/<!--#([\w-]+)-->/g, (_, name: string) =>
        readFileSync(join(dir, `${name}.html`), 'utf8')
          .replaceAll('{{alt-href}}', href ?? (other === 'en' ? '/en/' : '/'))
          .replaceAll('{{alt-lang}}', other)
          .replaceAll('{{alt-label}}', other === 'en' ? 'EN' : 'ES')
          .replaceAll('{{alt-title}}', other === 'en' ? 'Read in English' : 'Leer en español'),
      );
    },
  };
}

/**
 * Files in public/ keep their names, and Cloudflare tells browsers to cache them for hours, so an updated
 * icon or screenshot would not show. Every reference to one of them in the HTML gets `?v=<content hash>`.
 */
function versionPublic(): Plugin {
  const pub = resolve(__dirname, 'public');
  const cache = new Map<string, string>();
  const hash = (file: string) => {
    if (!cache.has(file)) cache.set(file, createHash('sha256').update(readFileSync(file)).digest('hex').slice(0, 10));
    return cache.get(file)!;
  };
  return {
    name: 'version-public',
    transformIndexHtml(html) {
      return html.replace(/(src|href)="(\/[^"?#]+\.(?:svg|png|ico|js|webmanifest|css))"/g, (m, attr: string, path: string) => {
        const file = join(pub, path);
        return existsSync(file) ? `${attr}="${path}?v=${hash(file)}"` : m;
      });
    },
  };
}

const pages = [
  'index.html', 'app/index.html', 'aviso-legal/index.html', 'privacidad/index.html', 'terminos/index.html', 'licencias/index.html',
  'en/index.html', 'en/app/index.html', 'en/legal-notice/index.html', 'en/privacy/index.html', 'en/terms/index.html', 'en/licenses/index.html',
];

export default defineConfig({
  plugins: [partials(), versionPublic()],
  // cssTarget: modern engines, so the minifier keeps the standard backdrop-filter (Chrome ignores the -webkit- one).
  build: { target: 'es2022', cssTarget: ['chrome120', 'edge120', 'firefox121', 'safari17'], chunkSizeWarningLimit: 1200, rollupOptions: { input: pages.map((p) => resolve(__dirname, p)) } },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['@huggingface/transformers', 'onnxruntime-web'] },
});
