import { readFileSync } from 'node:fs';
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

const pages = [
  'index.html', 'app/index.html', 'aviso-legal/index.html', 'privacidad/index.html', 'terminos/index.html', 'licencias/index.html',
  'en/index.html', 'en/app/index.html', 'en/legal-notice/index.html', 'en/privacy/index.html', 'en/terms/index.html', 'en/licenses/index.html',
];

export default defineConfig({
  plugins: [partials()],
  build: { target: 'es2022', chunkSizeWarningLimit: 1200, rollupOptions: { input: pages.map((p) => resolve(__dirname, p)) } },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['@huggingface/transformers', 'onnxruntime-web'] },
});
