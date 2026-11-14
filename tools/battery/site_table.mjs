// Writes the "Pruebas y medidas" table of the landing page (src/site/partials/medidas*.html) from
// results/battery/scores.json and results/measurements.json.
//
//   node tools/battery/site_table.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '../..');
const scores = JSON.parse(readFileSync(join(root, 'results/battery/scores.json'), 'utf8')).table;
const meas = JSON.parse(readFileSync(join(root, 'results/measurements.json'), 'utf8'));

const rows = meas.models.filter((m) => m.published);
const gb = (b, lang) => (b / 1e9).toLocaleString(lang === 'es' ? 'es-ES' : 'en-US', { maximumFractionDigits: 2 }) + ' GB';
const num = (x, lang, d = 1) => x.toLocaleString(lang === 'es' ? 'es-ES' : 'en-US', { maximumFractionDigits: d });
const frac = (s, lang) => (lang === 'es' ? String(s).replace('.', ',') : String(s));

function table(lang) {
  const es = lang === 'es';
  const head = es
    ? ['Modelo', 'Descarga', 'Pesos en la GPU', 'Conversación', 'Código', 'Razonamiento', 'Total', 'Velocidad en CPU*']
    : ['Model', 'Download', 'Weights on the GPU', 'Conversation', 'Code', 'Reasoning', 'Total', 'Speed on CPU*'];
  const body = rows
    .map((m) => {
      const s = scores[m.id];
      const c = (k) => (s ? frac(s.cats[k], lang) : '—');
      return `<tr><td>${m.name}</td><td>${gb(m.downloadBytes, lang)}</td><td>${gb(m.weightBytes, lang)}</td><td>${c('chat')}</td><td>${c('code')}</td><td>${c('reason')}</td><td><strong>${s ? frac(s.total, lang) : '—'}</strong></td><td>${s ? num(s.tps, lang) + ' tok/s' : '—'}</td></tr>`;
    })
    .join('\n                ');
  const note = es
    ? `* Velocidad de referencia en el procesador ARM de 4 núcleos del servidor (ONNX Runtime, mismos archivos), no en una GPU: en tu navegador irá bastante más rápido. Este proyecto no ha podido medirse en una GPU real; mide la tuya en la app con «Modelos → Medir rendimiento». Batería: 10 preguntas de conversación, 10 de código (con tests ejecutados) y 10 de razonamiento, en español e inglés, con decodificación voraz.`
    : `* Reference speed on the server's 4-core ARM processor (ONNX Runtime, same files), not on a GPU: it will be considerably faster in your browser. This project could not be measured on a real GPU; measure yours in the app with “Models → Measure performance”. Battery: 10 conversation, 10 code (with executed tests) and 10 reasoning prompts, in Spanish and English, with greedy decoding.`;
  return `<div class="table-wrap">
            <table class="data">
              <thead><tr>${head.map((x) => `<th>${x}</th>`).join('')}</tr></thead>
              <tbody>
                ${body}
              </tbody>
            </table>
          </div>
          <p class="note">${note}</p>
`;
}

writeFileSync(join(root, 'src/site/partials/medidas.html'), table('es'));
writeFileSync(join(root, 'src/site/partials/medidas-en.html'), table('en'));
console.log('ok', rows.map((r) => r.id).join(', '));
