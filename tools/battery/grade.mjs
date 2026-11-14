// Scores the answers in results/battery/<model>.json and writes results/battery/summary.md.
//
// - `check`: regular expressions the answer must contain (all / any), JSON validity, or a pattern that
//   must appear near the end (the final answer).
// - `test`: the code block of the answer is run with the test appended (python3 / node, 10 s limit,
//   in a temporary directory).
// - `regex_test`: the regex in the answer must accept and reject the given strings.
// - Prompts with only a `rubric` are graded by hand in manual.json (0, 0.5 or 1 per model and prompt).
//
//   node tools/battery/grade.mjs lfm3-1.7b qwen3.5-2b qwen3.5-4b [gemma4-12b]
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const dir = join(here, '../../results/battery');
const prompts = JSON.parse(readFileSync(join(here, 'prompts.json'), 'utf8'));
const manualFile = join(dir, 'manual.json');
const manual = existsSync(manualFile) ? JSON.parse(readFileSync(manualFile, 'utf8')) : {};
const models = process.argv.slice(2);

function codeOf(answer, lang) {
  const blocks = [...answer.matchAll(/```([\w+-]*)\n([\s\S]*?)```/g)];
  const pick = blocks.find((b) => !b[1] || b[1].toLowerCase().startsWith(lang === 'python' ? 'py' : 'j')) ?? blocks[0];
  return pick ? pick[2] : answer;
}

function runTest(answer, test) {
  const work = mkdtempSync(join(tmpdir(), 'battery-'));
  try {
    const code = codeOf(answer, test.lang) + '\n\n' + test.code + '\n';
    const file = join(work, test.lang === 'python' ? 'main.py' : 'main.mjs');
    writeFileSync(file, code);
    const out = execFileSync(test.lang === 'python' ? 'python3' : 'node', [file], { cwd: work, timeout: 10_000, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
    return { pass: out.includes('OK'), note: '' };
  } catch (e) {
    return { pass: false, note: String(e.stderr || e.message).split('\n').filter(Boolean).slice(-1)[0]?.slice(0, 160) ?? 'error' };
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

function regexOf(answer) {
  const m = /`\^?([^`]+?)\$?`/.exec(answer) ?? /\/(\^[^/\n]+\$)\//.exec(answer) ?? /(\^[^\s`]+\$)/.exec(answer);
  if (!m) return null;
  let src = m[1].trim();
  if (src.startsWith('/') && src.lastIndexOf('/') > 0) src = src.slice(1, src.lastIndexOf('/'));
  if (!src.startsWith('^')) src = '^' + src;
  if (!src.endsWith('$')) src += '$';
  try {
    return new RegExp(src);
  } catch {
    return null;
  }
}

/** What the app shows: the text after the reasoning block (same rule as src/engine/reasoning.ts). */
export function visible(answer) {
  const close = answer.lastIndexOf('</think>');
  if (close >= 0) return answer.slice(close + '</think>'.length).trim();
  return answer.trimStart().startsWith('<think>') ? '' : answer;
}

function grade(p, a, model) {
  if (!a) return { score: 0, how: 'missing' };
  const text = visible(a.answer);
  if (p.test) {
    const r = runTest(text, p.test);
    return { score: r.pass ? 1 : 0, how: 'test', note: r.note };
  }
  if (p.regex_test) {
    const re = regexOf(text);
    if (!re) return { score: 0, how: 'regex', note: 'no regex found' };
    const ok = p.regex_test.accept.every((s) => re.test(s)) && p.regex_test.reject.every((s) => !re.test(s));
    return { score: ok ? 1 : 0, how: 'regex', note: String(re) };
  }
  if (p.check) {
    const c = p.check;
    let ok = true;
    const notes = [];
    if (c.all) for (const r of c.all) if (!new RegExp(r, 'i').test(text)) (ok = false), notes.push(`missing /${r}/`);
    if (c.any && !c.any.some((r) => new RegExp(r, 'i').test(text))) (ok = false), notes.push('none of the expected answers');
    if (c.last && !new RegExp(c.last, 'i').test(text.slice(-160))) (ok = false), notes.push('final answer differs');
    if (c.json) {
      try {
        JSON.parse((/\{[\s\S]*\}/.exec(text) ?? [''])[0]);
      } catch {
        ok = false;
        notes.push('invalid JSON');
      }
    }
    const auto = ok ? 1 : 0;
    const man = manual[model]?.[p.id];
    // Manual review may lower an automatic pass (e.g. right number, wrong reasoning) but never raise a fail.
    return { score: man !== undefined ? Math.min(auto, man) : auto, how: man !== undefined ? 'check+manual' : 'check', note: notes.join('; ') };
  }
  const man = manual[model]?.[p.id];
  return man === undefined ? { score: null, how: 'manual (pending)' } : { score: man, how: 'manual' };
}

const table = {};
const detail = [];
for (const model of models) {
  const f = join(dir, `${model}.json`);
  if (!existsSync(f)) continue;
  const data = JSON.parse(readFileSync(f, 'utf8'));
  const byId = new Map(data.answers.map((a) => [a.id, a]));
  const cats = {};
  let tok = 0;
  let ms = 0;
  for (const p of prompts) {
    const a = byId.get(p.id);
    const g = grade(p, a, model);
    (cats[p.cat] ??= []).push(g.score);
    if (a) (tok += a.tokens), (ms += a.ms);
    detail.push({ model, id: p.id, ...g });
  }
  const sum = (xs) => xs.filter((x) => x !== null).reduce((s, x) => s + x, 0);
  const n = (xs) => xs.filter((x) => x !== null).length;
  table[model] = { cats: Object.fromEntries(Object.entries(cats).map(([k, v]) => [k, `${sum(v)}/${n(v)}`])), total: `${sum(Object.values(cats).flat())}/${n(Object.values(cats).flat())}`, pending: Object.values(cats).flat().filter((x) => x === null).length, tps: ms ? tok / (ms / 1000) : 0, answers: data.answers.length };
}

const lines = [
  '# Batería de prompts: resultados',
  '',
  'Generado por `tools/battery/grade.mjs`. Respuestas completas en `results/battery/<modelo>.json`; notas manuales en `manual.json`.',
  'Decodificación voraz, razonamiento desactivado, un prompt por conversación, mismos archivos ONNX (q4f16) que la web, ejecutados en CPU.',
  '',
  '| Modelo | Conversación | Código | Razonamiento | Total | Pendientes | tok/s (CPU, orientativo) |',
  '|---|---|---|---|---|---|---|',
  ...Object.entries(table).map(([m, r]) => `| ${m} | ${r.cats.chat ?? '-'} | ${r.cats.code ?? '-'} | ${r.cats.reason ?? '-'} | ${r.total} | ${r.pending} | ${r.tps.toFixed(1)} |`),
  '',
  '## Detalle',
  '',
  '| Prompt | ' + Object.keys(table).join(' | ') + ' |',
  '|---|' + Object.keys(table).map(() => '---').join('|') + '|',
  ...prompts.map((p) => `| ${p.id} | ` + Object.keys(table).map((m) => { const d = detail.find((x) => x.model === m && x.id === p.id); return d?.score === null ? '?' : `${d?.score}${d?.note ? ` (${d.note.replace(/\|/g, '\\|')})` : ''}`; }).join(' | ') + ' |'),
  '',
];
writeFileSync(join(dir, 'summary.md'), lines.join('\n'));
writeFileSync(join(dir, 'scores.json'), JSON.stringify({ table, detail }, null, 2));
console.log(lines.slice(5, 6 + Object.keys(table).length + 1).join('\n'));
