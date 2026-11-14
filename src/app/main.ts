import '@fontsource-variable/inter';
import '../tokens.css';
import './app.css';
import { Engine } from '../engine/client.ts';
import { fit, probeGpu, recommend, type GpuReport } from '../engine/gpu.ts';
import { AVAILABLE, MODELS, downloadBytes, modelById, type ModelId, type ModelInfo } from '../engine/models.ts';
import type { ChatMessage, GenOptions, GenStats } from '../engine/protocol.ts';
import { lang, locale, localPath, t } from '../i18n.ts';
import { $, fmtBytes, fmtDuration, fmtNum, h, toast } from '../ui/dom.ts';
import { mountAurora, pulseAurora, setMood, wireGlow } from '../ui/aurora.ts';
import { hydrateIcons, icon } from '../ui/icons.ts';
import { cacheStatus, deleteModel, deleteOrphans, isComplete, orphanBytes, storageEstimate, type CacheEntry } from './cache.ts';
import { deleteConversation, listConversations, newId, saveConversation, titleFrom, type Conversation, type StoredMessage } from './db.ts';
import { card, tierLabel } from './modelcards.ts';
import { level, score, wait, type Level, type Wait } from './score.ts';
import { copyButton, renderMarkdown } from './render.ts';

// ---------------------------------------------------------------------------------------------
// Settings (localStorage: preferences only; conversations are in IndexedDB)

interface ModelSettings {
  temperature: number;
  topP: number;
  maxTokens: number;
  reasoning: boolean;
}
interface Settings {
  theme: 'auto' | 'light' | 'dark';
  model: ModelId | null;
  system: string;
  perModel: Partial<Record<ModelId, ModelSettings>>;
  panel: boolean;
}
const KEY = 'webgpu-chat-settings';
const settings: Settings = (() => {
  const d: Settings = { theme: 'auto', model: null, system: '', perModel: {}, panel: false };
  try {
    return { ...d, ...JSON.parse(localStorage.getItem(KEY) ?? '{}') };
  } catch {
    return d;
  }
})();
const saveSettings = () => {
  try {
    localStorage.setItem(KEY, JSON.stringify(settings));
  } catch {
    /* storage blocked */
  }
};
const defaultsFor = (m: ModelInfo): ModelSettings => ({
  temperature: m.defaults.temperature,
  topP: m.defaults.topP,
  maxTokens: 1024,
  reasoning: m.defaults.reasoning,
});
const ms = (m: ModelInfo): ModelSettings => (settings.perModel[m.id] ??= defaultsFor(m));

// ---------------------------------------------------------------------------------------------
// State

const engine = new Engine();

/** The aurora's resting mood: 'typing' while the user is composing a message, otherwise 'idle'. */
let typingTimer = 0;
function restMood(): void {
  if (generating || loading) return;
  const composing = document.activeElement === el.input && el.input.value.trim().length > 0;
  setMood(composing ? 'typing' : 'idle');
}
/** Development only (stripped from production builds): 4-bit weights with float32 math, for GPUs without shader-f16
 *  such as the SwiftShader software adapter used in automated tests. */
const DEV_Q4 = import.meta.env.DEV && new URLSearchParams(location.search).has('q4');
let gpu: GpuReport | null = null;
let conversations: Conversation[] = [];
let current: Conversation | null = null;
let selected: ModelInfo | null = null;
const cacheInfo = new Map<ModelId, CacheEntry>();
let loading: { model: ModelId; loaded: number; total: number; stage: string } | null = null;
let generating = false;
let query = '';
let ctxTokens = 0;

const el = {
  shell: $('shell'),
  list: $('conv-list'),
  thread: $('thread'),
  inner: $('thread-inner'),
  input: $<HTMLTextAreaElement>('input'),
  send: $<HTMLButtonElement>('btn-send'),
  status: $('status-line'),
  picker: $('model-picker'),
  panel: $('settings'),
  reasoningChip: $<HTMLButtonElement>('btn-reasoning'),
};

// ---------------------------------------------------------------------------------------------
// Theme

function isDark(): boolean {
  return settings.theme === 'dark' || (settings.theme === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches);
}
function applyTheme() {
  if (settings.theme === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = settings.theme;
  $('btn-theme').replaceChildren(icon(isDark() ? 'sun' : 'moon', 18));
  $('btn-theme').title = isDark() ? t('Tema claro') : t('Tema oscuro');
}

// ---------------------------------------------------------------------------------------------
// Sidebar

function groupLabel(ts: number): string {
  const day = 86_400_000;
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  const d = start.getTime() - ts;
  if (ts >= start.getTime()) return t('Hoy');
  if (d < day) return t('Ayer');
  if (d < 7 * day) return t('Últimos 7 días');
  if (d < 30 * day) return t('Últimos 30 días');
  return new Date(ts).toLocaleDateString(locale, { month: 'long', year: 'numeric' });
}

function matches(c: Conversation, q: string): boolean {
  if (!q) return true;
  const n = (s: string) => s.toLocaleLowerCase(locale).normalize('NFD').replace(/\p{M}/gu, '');
  const qq = n(q);
  return n(c.title).includes(qq) || c.messages.some((m) => n(m.content).includes(qq));
}

function renderList() {
  const items = conversations.filter((c) => matches(c, query));
  el.list.replaceChildren();
  if (!items.length) {
    el.list.append(h('p', { class: 'list-empty' }, query ? t('Sin resultados') : t('Tus conversaciones aparecerán aquí. Se guardan solo en este navegador.')));
    return;
  }
  let group = '';
  for (const c of items) {
    const g = groupLabel(c.updatedAt);
    if (g !== group) {
      group = g;
      el.list.append(h('div', { class: 'list-group' }, g));
    }
    el.list.append(convItem(c));
  }
}

function convItem(c: Conversation): HTMLElement {
  const active = current?.id === c.id;
  const title = h('span', { class: 'conv-title' }, c.title);
  const row = h(
    'div',
    { class: `conv${active ? ' active' : ''}`, role: 'listitem' },
    h('button', { type: 'button', class: 'conv-open', onclick: () => openConversation(c.id), 'aria-current': active ? 'page' : null }, title),
    h(
      'div',
      { class: 'conv-actions' },
      h('button', { type: 'button', class: 'icon-btn sm', title: t('Renombrar'), 'aria-label': t('Renombrar'), onclick: () => rename(c, row) }, icon('pencil', 15)),
      h('button', { type: 'button', class: 'icon-btn sm', title: t('Borrar'), 'aria-label': t('Borrar'), onclick: () => remove(c) }, icon('trash', 15)),
    ),
  );
  return row;
}

function rename(c: Conversation, row: HTMLElement) {
  const input = h('input', { class: 'conv-rename', value: c.title, maxLength: 120, 'aria-label': t('Nuevo nombre') });
  row.replaceChildren(input);
  input.focus();
  input.select();
  let done = false;
  const finish = async (save: boolean) => {
    if (done) return;
    done = true;
    const v = input.value.trim();
    if (save && v && v !== c.title) {
      c.title = v;
      c.customTitle = true;
      await saveConversation(c);
    }
    renderList();
  };
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') void finish(true);
    if (e.key === 'Escape') void finish(false);
  });
  input.addEventListener('blur', () => void finish(true));
}

async function remove(c: Conversation) {
  if (!(await confirmDialog(t('¿Borrar esta conversación?'), t('Se eliminará «{t}» de este navegador. No se puede deshacer.', { t: c.title }), t('Borrar')))) return;
  await deleteConversation(c.id);
  conversations = conversations.filter((x) => x.id !== c.id);
  if (current?.id === c.id) newConversation();
  renderList();
}

function setDrawer(open: boolean) {
  el.shell.classList.toggle('drawer-open', open);
}

// ---------------------------------------------------------------------------------------------
// Conversations

function newConversation() {
  if (generating) return;
  current = null;
  engine.reset();
  renderThread();
  renderList();
  updateContext();
  el.input.focus();
  setDrawer(false);
}

function openConversation(id: string) {
  if (generating) {
    toast(t('Espera a que termine la respuesta o párala.'));
    return;
  }
  current = conversations.find((c) => c.id === id) ?? null;
  engine.reset();
  renderThread();
  renderList();
  updateContext();
  setDrawer(false);
}

async function persist(c: Conversation) {
  c.updatedAt = Date.now();
  await saveConversation(c);
  conversations = [c, ...conversations.filter((x) => x.id !== c.id)];
  renderList();
}

// ---------------------------------------------------------------------------------------------
// Thread rendering

function renderThread() {
  el.inner.replaceChildren();
  if (!current || !current.messages.length) {
    el.inner.append(emptyState());
    el.thread.classList.add('is-empty');
    return;
  }
  el.thread.classList.remove('is-empty');
  current.messages.forEach((m, i) => el.inner.append(messageView(m, i)));
  scrollToEnd(true);
}

let emptyShown = false;
function emptyState(): HTMLElement {
  // The entrance animation plays once per page view, not on every repaint.
  const box = h('div', { class: emptyShown ? 'empty' : 'empty enter' });
  emptyShown = true;
  if (gpu && !gpu.ok) {
    box.append(noWebGpu(gpu));
    return box;
  }
  box.append(h('h1', { class: 'empty-title' }, h('span', {}, t('¿En qué te ayudo?'))));
  box.append(
    h(
      'p',
      { class: 'empty-sub' },
      icon('lock', 15),
      h('span', {}, t('El modelo se ejecuta en tu equipo con WebGPU. Lo que escribas no sale de este navegador.')),
    ),
  );
  if (selected) {
    const c = cacheInfo.get(selected.id);
    if (engine.loaded !== selected.id && !loading && !(c && isComplete(c))) {
      box.append(
        h(
          'div',
          { class: 'download-cta', 'data-glow': '' },
          h('div', {}, h('strong', {}, t('Descarga {m} para empezar', { m: selected.name })), h('span', {}, t('{s} desde Hugging Face, una sola vez. Después funciona sin conexión.', { s: fmtBytes(downloadBytes(selected)) }))),
          h('button', { type: 'button', class: 'btn primary', onclick: () => void loadModel(selected!.id) }, icon('download', 16), t('Descargar')),
        ),
      );
    }
  }
  const sugg = [
    t('Explícame qué es WebGPU en tres frases'),
    t('Escribe una función en Python que invierta una cadena'),
    t('Ideas para una cena rápida con lo que suele haber en la nevera'),
    t('Corrige este texto: «haber si nos vemos mañana y hablamos»'),
  ];
  box.append(
    h(
      'div',
      { class: 'suggestions' },
      ...sugg.map((s) =>
        h('button', { type: 'button', class: 'suggestion', 'data-glow': '', onclick: () => ((el.input.value = s), autosize(), el.input.focus(), updateContext()) }, s),
      ),
    ),
  );
  return box;
}

function noWebGpu(g: GpuReport): HTMLElement {
  const why =
    g.reason === 'no-f16'
      ? t('Tu navegador tiene WebGPU, pero tu GPU (o su controlador) no admite números de 16 bits en los shaders (shader-f16). Los modelos de esta web los necesitan.')
      : g.reason === 'no-adapter'
        ? t('Tu navegador tiene WebGPU, pero no ha encontrado una GPU utilizable. Suele pasar con controladores antiguos, en máquinas virtuales o si la aceleración por hardware está desactivada.')
        : g.reason === 'error'
          ? t('WebGPU ha fallado al iniciarse: {e}', { e: g.error ?? '' })
          : t('Tu navegador no tiene WebGPU, la tecnología que permite ejecutar los modelos en tu GPU. Sin ella no es posible usar esta web (no hay modo de CPU: sería demasiado lento).');
  return h(
    'div',
    { class: 'nogpu' },
    h('div', { class: 'nogpu-icon' }, icon('alert', 22)),
    h('h1', {}, t('Este navegador no puede ejecutar los modelos')),
    h('p', {}, why),
    h('h2', {}, t('Qué puedes hacer')),
    h(
      'ul',
      {},
      h('li', {}, t('Usa Chrome o Edge 113 o posterior en Windows, macOS o ChromeOS, o Chrome 121+ en Android.')),
      h('li', {}, t('En Safari, WebGPU viene activado desde Safari 26 (macOS, iOS y iPadOS 26).')),
      h('li', {}, t('En Firefox, WebGPU está activado en Windows desde la versión 141; en otros sistemas, prueba Chrome.')),
      h('li', {}, t('Comprueba que la aceleración por hardware del navegador está activada y que los controladores de la GPU están al día.')),
    ),
  );
}

function statsLine(s: GenStats, model?: string): string {
  const parts = [
    model ? (MODELS.find((m) => m.id === model)?.name ?? model) : '',
    t('{n} tokens', { n: fmtNum(s.newTokens) }),
    s.tokensPerSecond ? t('{v} tokens/s', { v: fmtNum(s.tokensPerSecond, 1) }) : '',
    t('primera palabra en {d}', { d: fmtDuration(s.prefillMs) }),
    s.cachedTokens ? t('{n} tokens reutilizados de la caché', { n: fmtNum(s.cachedTokens) }) : '',
    s.stopped === 'user' ? t('detenida') : s.stopped === 'length' ? t('cortada por el límite de tokens') : '',
  ];
  return parts.filter(Boolean).join(' · ');
}

function reasoningBox(text: string, thinking: boolean, ms?: number): HTMLDetailsElement {
  const d = h('details', { class: 'reasoning' }) as HTMLDetailsElement;
  d.open = thinking;
  const label = thinking ? t('Razonando…') : ms ? t('Razonamiento ({d})', { d: fmtDuration(ms) }) : t('Razonamiento');
  d.append(h('summary', {}, icon('brain', 15), h('span', {}, label), icon('chevronDown', 14)), h('div', { class: 'reasoning-body' }, text));
  return d;
}

function messageView(m: StoredMessage, index: number): HTMLElement {
  if (m.role === 'user') {
    return h(
      'div',
      { class: 'msg user' },
      h('div', { class: 'bubble' }, m.content),
      h('div', { class: 'msg-tools' }, copyButton(() => m.content, t('Copiar'))),
    );
  }
  const body = h('div', { class: 'md' });
  body.append(renderMarkdown(m.content));
  const view = h('div', { class: 'msg assistant' });
  if (m.reasoning) view.append(reasoningBox(m.reasoning, false));
  view.append(body);
  if (m.error) view.append(h('div', { class: 'msg-error' }, icon('alert', 15), h('span', {}, m.error)));
  const isLast = current && index === current.messages.length - 1;
  view.append(
    h(
      'div',
      { class: 'msg-tools' },
      copyButton(() => m.content, t('Copiar respuesta')),
      isLast ? h('button', { type: 'button', class: 'icon-btn', title: t('Regenerar'), 'aria-label': t('Regenerar'), onclick: () => void regenerate() }, icon('refresh', 16)) : null,
      m.stats ? h('span', { class: 'msg-stats' }, statsLine(m.stats, m.model)) : null,
    ),
  );
  return view;
}

let stick = true;
function scrollToEnd(force = false) {
  if (force || stick) el.thread.scrollTop = el.thread.scrollHeight;
}
el.thread.addEventListener('scroll', () => {
  stick = el.thread.scrollHeight - el.thread.scrollTop - el.thread.clientHeight < 80;
});

// ---------------------------------------------------------------------------------------------
// Model picker

function statusOf(m: ModelInfo): { label: string; cls: string } {
  if (engine.loaded === m.id) return { label: t('Cargado'), cls: 'ok' };
  if (loading?.model === m.id) return { label: t('Cargando'), cls: 'busy' };
  const c = cacheInfo.get(m.id);
  if (c && isComplete(c)) return { label: t('Descargado'), cls: 'idle' };
  return { label: fmtBytes(downloadBytes(m)), cls: 'none' };
}

function renderPicker() {
  const m = selected;
  const st = m ? statusOf(m) : null;
  const btn = h(
    'button',
    { type: 'button', class: 'picker-btn', 'aria-haspopup': 'listbox', 'aria-expanded': 'false', id: 'picker-btn' },
    h('span', { class: `dot ${st?.cls ?? 'none'}` }),
    h('span', { class: 'picker-name' }, m ? m.name : t('Elige un modelo')),
    icon('chevronDown', 16),
  );
  const menu = h('div', { class: 'picker-menu', role: 'listbox', id: 'picker-menu', hidden: true });
  const rec = gpu ? recommend(gpu) : null;
  for (const x of AVAILABLE) {
    const f = gpu ? fit(x, gpu) : { ok: true as const };
    const s = statusOf(x);
    const opt = h(
      'div',
      { class: `picker-opt${x.id === m?.id ? ' selected' : ''}${f.ok ? '' : ' blocked'}`, role: 'option', 'aria-selected': x.id === m?.id ? 'true' : 'false', 'aria-disabled': f.ok ? null : 'true', tabIndex: 0 },
      h(
        'div',
        { class: 'opt-main' },
        h('div', { class: 'opt-title' }, h('strong', {}, x.name), h('span', { class: `tier ${x.tier}` }, tierLabel(x.tier)), x.id === rec ? h('span', { class: 'tag-rec' }, t('Para tu equipo')) : null),
        h('div', { class: 'opt-meta' }, h('span', { class: `dot ${s.cls}` }), h('span', {}, s.label)),
        f.ok ? (f.warn === 'software' ? h('div', { class: 'opt-warn' }, t('Tu GPU es un adaptador por software: irá muy lento.')) : f.warn ? h('div', { class: 'opt-warn' }, t('Necesita una GPU potente; puede no caber en la tuya.')) : null) : h('div', { class: 'opt-warn' }, blockReason(f.why)),
      ),
      h('button', { type: 'button', class: 'icon-btn sm', title: t('Ficha del modelo'), 'aria-label': t('Ficha del modelo'), onclick: (e: Event) => (e.stopPropagation(), closePicker(), showCard(x)) }, icon('info', 16)),
    );
    if (f.ok)
      opt.addEventListener('click', () => {
        closePicker();
        void selectModel(x.id, true);
      });
    opt.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        opt.click();
      }
    });
    menu.append(opt);
  }
  menu.append(h('button', { type: 'button', class: 'picker-foot', onclick: () => (closePicker(), void openModels()) }, icon('hardDrive', 15), t('Gestionar modelos descargados')));
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (menu.hidden) openPicker();
    else closePicker();
  });
  el.picker.replaceChildren(btn, menu);
}

function blockReason(why: string): string {
  switch (why) {
    case 'no-f16':
      return t('Tu GPU no admite shader-f16.');
    case 'buffer':
      return t('Tu GPU no admite búferes tan grandes como los de este modelo.');
    case 'memory':
      return t('Tu equipo tiene poca memoria para este modelo.');
    case 'mobile':
      return t('Demasiado grande para un móvil.');
    default:
      return t('Necesita WebGPU.');
  }
}

function openPicker() {
  const menu = $('picker-menu');
  menu.hidden = false;
  $('picker-btn').setAttribute('aria-expanded', 'true');
  (menu.querySelector('.picker-opt.selected') ?? menu.querySelector('.picker-opt'))?.dispatchEvent(new Event('focus'));
}
function closePicker() {
  const menu = document.getElementById('picker-menu');
  if (menu) menu.hidden = true;
  document.getElementById('picker-btn')?.setAttribute('aria-expanded', 'false');
}
document.addEventListener('click', (e) => {
  if (!(e.target as HTMLElement).closest('#model-picker')) closePicker();
});

async function selectModel(id: ModelId, userChoice: boolean) {
  const m = modelById(id);
  if (!m) return;
  if (generating) {
    toast(t('Espera a que termine la respuesta o párala.'));
    return;
  }
  selected = m;
  settings.model = m.id;
  saveSettings();
  renderPicker();
  renderSettings();
  const c = cacheInfo.get(m.id);
  if (engine.loaded === m.id) return renderComposerState();
  if (c && isComplete(c)) await loadModel(m.id);
  else if (userChoice) await loadModel(m.id);
  else renderThreadIfEmpty();
  renderComposerState();
}

function renderThreadIfEmpty() {
  if (!current || !current.messages.length) renderThread();
}

async function loadModel(id: ModelId) {
  const m = modelById(id);
  if (!m || loading?.model === id) return;
  try {
    await navigator.storage?.persist?.();
  } catch {
    /* best effort */
  }
  loading = { model: id, loaded: 0, total: downloadBytes(m), stage: 'download' };
  setMood('loading');
  renderPicker();
  renderThreadIfEmpty();
  renderLoad();
  renderComposerState();
  try {
    const t0 = performance.now();
    await engine.load(id, {
      onProgress: (l, tot) => {
        if (loading) Object.assign(loading, { loaded: l, total: tot, stage: 'download' });
        renderLoad();
      },
      onStage: (s) => {
        if (loading) loading.stage = s;
        renderLoad();
      },
    }, DEV_Q4 ? 'q4' : undefined);
    loading = null;
    cacheInfo.set(id, await cacheStatus(m));
    toast(t('{m} listo en {d}', { m: m.name, d: fmtDuration(performance.now() - t0) }));
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg === 'superseded') return;
    loading = null;
    showError(loadErrorText(msg, m));
  }
  restMood();
  renderLoad();
  renderPicker();
  renderThreadIfEmpty();
  renderComposerState();
  updateContext();
}

function loadErrorText(msg: string, m: ModelInfo): string {
  if (/memory|allocat|OOM|device.*lost|GPUBuffer|out of/i.test(msg))
    return t('No hay memoria de GPU suficiente para {m}. Prueba un modelo más pequeño o cierra otras pestañas y aplicaciones.', { m: m.name }) + ` (${msg})`;
  if (/fetch|network|Failed to fetch|NetworkError|quota/i.test(msg))
    return t('La descarga ha fallado. Revisa la conexión y el espacio libre en disco, y vuelve a intentarlo: lo ya descargado se conserva.') + ` (${msg})`;
  return t('No se ha podido cargar {m}: {e}', { m: m.name, e: msg });
}

function renderLoad() {
  const box = $('load-banner');
  if (!loading) {
    // Model chosen but not in memory: offer to download or load it right above the composer.
    const m = selected;
    const show = m && engine.loaded !== m.id && !(gpu && !gpu.ok) && current?.messages.length;
    box.hidden = !show;
    if (show) {
      const c = cacheInfo.get(m.id);
      const cached = !!c && isComplete(c);
      box.replaceChildren(
        h(
          'div',
          { class: 'load-row cta' },
          h('span', {}, cached ? t('{m} está descargado pero no cargado.', { m: m.name }) : t('{m} no está descargado ({s}).', { m: m.name, s: fmtBytes(downloadBytes(m)) })),
          h('button', { type: 'button', class: 'btn small primary', onclick: () => void loadModel(m.id) }, icon(cached ? 'cpu' : 'download', 15), cached ? t('Cargar') : t('Descargar')),
        ),
      );
    }
    return;
  }
  const m = modelById(loading.model)!;
  const pct = loading.total ? Math.min(100, (loading.loaded / loading.total) * 100) : 0;
  const label =
    loading.stage === 'warmup'
      ? t('Preparando {m} en la GPU…', { m: m.name })
      : loading.stage === 'compile' || pct >= 99.9
        ? t('Cargando {m} en la GPU…', { m: m.name })
        : t('Descargando {m}: {a} de {b}', { m: m.name, a: fmtBytes(loading.loaded), b: fmtBytes(loading.total) });
  box.hidden = false;
  box.replaceChildren(
    h('div', { class: 'load-row' }, h('span', {}, label), h('span', { class: 'load-pct' }, loading.stage === 'download' && pct < 99.9 ? `${fmtNum(pct, 0)} %` : '')),
    h('div', { class: `bar${loading.stage !== 'download' || pct >= 99.9 ? ' indeterminate' : ''}` }, h('i', { style: null, 'data-w': pct.toFixed(1) })),
  );
  const bar = box.querySelector<HTMLElement>('.bar i');
  if (bar) bar.style.width = `${pct}%`;
}

// ---------------------------------------------------------------------------------------------
// Composer and generation

function autosize() {
  el.input.style.height = 'auto';
  el.input.style.height = `${Math.min(el.input.scrollHeight, 220)}px`;
}

function renderComposerState() {
  renderLoad();
  const ready = !!selected && engine.loaded === selected.id && !loading;
  el.send.replaceChildren(icon(generating ? 'stop' : 'arrowUp', 18));
  el.send.title = generating ? t('Parar (Esc)') : t('Enviar (Intro)');
  el.send.setAttribute('aria-label', el.send.title);
  el.send.classList.toggle('stop', generating);
  el.send.disabled = !generating && (!ready || !el.input.value.trim());
  el.input.disabled = !!gpu && !gpu.ok;
  el.input.placeholder = gpu && !gpu.ok ? t('WebGPU no disponible') : ready || generating ? t('Escribe un mensaje…') : loading ? t('Cargando el modelo…') : t('Descarga un modelo para empezar');
  const r = selected ? ms(selected).reasoning : false;
  el.reasoningChip.classList.toggle('on', r);
  el.reasoningChip.setAttribute('aria-pressed', String(r));
}

function genOptions(m: ModelInfo): GenOptions {
  const s = ms(m);
  return { temperature: s.temperature, topP: s.topP, topK: m.defaults.topK, repetitionPenalty: m.defaults.repetitionPenalty, maxTokens: s.maxTokens, reasoning: s.reasoning };
}

function toChat(msgs: StoredMessage[]): ChatMessage[] {
  const out: ChatMessage[] = [];
  if (settings.system.trim()) out.push({ role: 'system', content: settings.system.trim() });
  for (const m of msgs) {
    if (m.role === 'assistant' && !m.content && !m.reasoning) continue;
    out.push(m.role === 'assistant' ? { role: 'assistant', content: m.content, reasoning: m.reasoning } : { role: 'user', content: m.content });
  }
  return out;
}

/** Drop the oldest exchanges until prompt + answer budget fits the model's context window. */
async function fitContext(m: ModelInfo, msgs: StoredMessage[], maxTokens: number): Promise<{ chat: ChatMessage[]; dropped: number }> {
  let start = 0;
  for (;;) {
    const chat = toChat(msgs.slice(start));
    const n = await engine.count(chat, ms(m).reasoning);
    if (n + maxTokens <= m.context || start >= msgs.length - 1) return { chat, dropped: start };
    start += 2; // one user + one assistant message
    while (start < msgs.length - 1 && msgs[start]!.role !== 'user') start++;
  }
}

async function send() {
  const text = el.input.value.trim();
  if (!text || generating || !selected || engine.loaded !== selected.id) return;
  el.input.value = '';
  autosize();
  if (!current) {
    current = { id: newId(), title: titleFrom(text), createdAt: Date.now(), updatedAt: Date.now(), model: selected.id, messages: [] };
  }
  current.messages.push({ role: 'user', content: text });
  await persist(current);
  renderThread();
  await respond();
}

async function regenerate() {
  if (!current || generating) return;
  if (current.messages.at(-1)?.role === 'assistant') current.messages.pop();
  if (current.messages.at(-1)?.role !== 'user') return;
  renderThread();
  await respond();
}

async function respond() {
  const c = current!;
  const m = selected!;
  const opts = genOptions(m);
  generating = true;
  setMood('thinking');
  renderComposerState();
  const { chat, dropped } = await fitContext(m, c.messages, opts.maxTokens);
  if (dropped) toast(t('Se han omitido {n} mensajes antiguos para no superar el contexto del modelo.', { n: dropped }));

  const view = h('div', { class: 'msg assistant streaming' });
  const rbox = { el: null as HTMLDetailsElement | null, start: 0, end: 0 };
  const body = h('div', { class: 'md' }, h('span', { class: 'cursor' }));
  view.append(body);
  el.inner.append(view);
  el.thread.classList.remove('is-empty');
  stick = true;
  scrollToEnd(true);

  let pending: { content: string; reasoning: string; thinking: boolean } | null = null;
  let raf = 0;
  const paint = () => {
    raf = 0;
    if (!pending) return;
    const p = pending;
    pending = null;
    if (p.reasoning) {
      if (!rbox.el) {
        rbox.start = performance.now();
        rbox.el = reasoningBox(p.reasoning, true);
        view.prepend(rbox.el);
      }
      rbox.el.querySelector('.reasoning-body')!.textContent = p.reasoning;
      if (!p.thinking && !rbox.end) {
        rbox.end = performance.now();
        rbox.el.open = false;
        rbox.el.querySelector('summary span')!.textContent = t('Razonamiento ({d})', { d: fmtDuration(rbox.end - rbox.start) });
      }
    }
    body.replaceChildren(renderMarkdown(p.content));
    if (!p.thinking) body.append(h('span', { class: 'cursor' }));
    scrollToEnd();
  };

  const msg: StoredMessage = { role: 'assistant', content: '', model: m.id };
  try {
    const r = await engine.generate(chat, opts, {
      onDelta: (d) => {
        setMood(d.thinking || !d.content ? 'thinking' : 'writing');
        pulseAurora(0.02);
        pending = d;
        if (!raf) raf = requestAnimationFrame(paint);
      },
    });
    msg.content = r.content;
    if (r.reasoning) msg.reasoning = r.reasoning;
    msg.stats = r.stats;
    msg.partial = r.stats.stopped !== 'eos';
  } catch (e) {
    msg.error = t('Error al generar: {e}', { e: e instanceof Error ? e.message : String(e) });
    if (/device.*lost|GPU|memory/i.test(msg.error)) msg.error += ' ' + t('Si el problema sigue, recarga la página o usa un modelo más pequeño.');
  }
  cancelAnimationFrame(raf);
  generating = false;
  restMood();
  c.messages.push(msg);
  await persist(c);
  if (current?.id === c.id) {
    view.remove();
    el.inner.append(messageView(msg, c.messages.length - 1));
    // The previous answer loses its regenerate button.
    const views = el.inner.querySelectorAll('.msg.assistant');
    views.forEach((v, i) => i < views.length - 1 && v.querySelector('.msg-tools button[title="' + t('Regenerar') + '"]')?.remove());
    scrollToEnd();
  }
  renderComposerState();
  updateContext();
}

el.input.addEventListener('input', () => {
  if (!generating && !loading) {
    setMood(el.input.value.trim() ? 'typing' : 'idle');
    pulseAurora(0.05);
    // A pause in typing lets the light settle back.
    clearTimeout(typingTimer);
    typingTimer = window.setTimeout(() => !generating && !loading && setMood('idle'), 2500);
  }
  autosize();
  renderComposerState();
  scheduleContext();
});
el.input.addEventListener('blur', () => restMood());
el.input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    void send();
  }
});
$('composer').addEventListener('submit', (e) => {
  e.preventDefault();
  if (generating) engine.stop();
  else void send();
});
el.reasoningChip.addEventListener('click', () => {
  if (!selected) return;
  ms(selected).reasoning = !ms(selected).reasoning;
  saveSettings();
  renderComposerState();
  renderSettings();
  updateContext();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && generating) engine.stop();
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'o') {
    e.preventDefault();
    newConversation();
  }
});

// ---------------------------------------------------------------------------------------------
// Context usage

let ctxTimer = 0;
function scheduleContext() {
  clearTimeout(ctxTimer);
  ctxTimer = window.setTimeout(updateContext, 300);
}
async function updateContext() {
  if (!selected || engine.loaded !== selected.id) {
    ctxTokens = 0;
    renderContext();
    return;
  }
  const msgs: StoredMessage[] = [...(current?.messages ?? [])];
  const draft = el.input.value.trim();
  if (draft) msgs.push({ role: 'user', content: draft });
  ctxTokens = await engine.count(toChat(msgs), ms(selected).reasoning);
  renderContext();
}
function renderContext() {
  const m = selected;
  const box = document.getElementById('ctx');
  if (!box || !m) return;
  const max = m.context;
  const reserve = ms(m).maxTokens;
  const used = Math.min(ctxTokens, max);
  const pct = (used / max) * 100;
  const pr = Math.min(100 - pct, (reserve / max) * 100);
  box.replaceChildren(
    h('div', { class: 'ctx-row' }, h('span', {}, t('Uso del contexto')), h('span', { class: 'mono' }, `${fmtNum(used)} / ${fmtNum(max)}`)),
    h('div', { class: 'ctx-bar' }, h('i', { class: 'used' }), h('i', { class: 'reserve' })),
    h('p', { class: 'hint' }, engine.loaded === m.id ? t('Conversación y mensaje en curso, más {n} tokens reservados para la respuesta. Si no caben, se omiten los mensajes más antiguos.', { n: fmtNum(reserve) }) : t('Carga el modelo para medirlo.')),
  );
  box.querySelector<HTMLElement>('.used')!.style.width = `${pct}%`;
  box.querySelector<HTMLElement>('.reserve')!.style.width = `${pr}%`;
  box.classList.toggle('warn', ctxTokens + reserve > max);
  const mini = $('ctx-mini');
  mini.textContent = engine.loaded === m.id ? `${fmtNum((ctxTokens / max) * 100, 0)} %` : '';
  mini.title = t('Uso del contexto: {a} de {b} tokens', { a: fmtNum(ctxTokens), b: fmtNum(max) });
}

// ---------------------------------------------------------------------------------------------
// Settings panel

function slider(label: string, min: number, max: number, step: number, value: number, fmt: (v: number) => string, on: (v: number) => void, hint: string): HTMLElement {
  const out = h('output', { class: 'mono' }, fmt(value));
  const input = h('input', { type: 'range', min, max, step, value, 'aria-label': label });
  input.addEventListener('input', () => {
    out.textContent = fmt(Number(input.value));
    on(Number(input.value));
    saveSettings();
  });
  return h('div', { class: 'field' }, h('div', { class: 'field-row' }, h('label', {}, label), out), input, h('p', { class: 'hint' }, hint));
}

function renderSettings() {
  const body = $('settings-body');
  body.replaceChildren();
  const sys = h('textarea', { rows: 4, value: settings.system, placeholder: t('Por ejemplo: «Responde siempre en español y de forma breve».'), 'aria-label': t('Prompt de sistema') });
  sys.addEventListener('input', () => {
    settings.system = sys.value;
    saveSettings();
    scheduleContext();
  });
  body.append(h('div', { class: 'field' }, h('label', {}, t('Prompt de sistema')), sys, h('p', { class: 'hint' }, t('Instrucciones que el modelo tiene en cuenta en toda la conversación.'))));
  if (!selected) return;
  const m = selected;
  const s = ms(m);
  const dec = (v: number) => v.toLocaleString(locale, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  body.append(
    h('div', { class: 'settings-model' }, t('Ajustes para {m}', { m: m.name })),
    slider(t('Temperatura'), 0, 1.5, 0.05, s.temperature, dec, (v) => (s.temperature = v), t('Más baja: respuestas más previsibles. Más alta: más variadas. 0 elige siempre la palabra más probable.')),
    slider(t('Top-p'), 0.05, 1, 0.05, s.topP, dec, (v) => (s.topP = v), t('Solo se eligen palabras entre las más probables hasta sumar esta probabilidad.')),
    slider(t('Máximo de tokens'), 64, Math.min(4096, m.context / 2), 64, s.maxTokens, (v) => fmtNum(v), (v) => ((s.maxTokens = v), renderContext()), t('Longitud máxima de cada respuesta, razonamiento incluido.')),
  );
  const tog = h('button', { type: 'button', class: `switch${s.reasoning ? ' on' : ''}`, role: 'switch', 'aria-checked': String(s.reasoning), 'aria-label': t('Razonamiento') }, h('i'));
  tog.addEventListener('click', () => el.reasoningChip.click());
  body.append(
    h('div', { class: 'field' }, h('div', { class: 'field-row' }, h('label', {}, t('Razonamiento')), tog), h('p', { class: 'hint' }, t('El modelo piensa antes de responder. Mejora problemas de lógica, matemáticas y código, pero tarda bastante más.'))),
    h('div', { class: 'field ctx', id: 'ctx' }),
    h(
      'button',
      {
        type: 'button',
        class: 'btn ghost small',
        onclick: () => {
          settings.perModel[m.id] = defaultsFor(m);
          saveSettings();
          renderSettings();
          renderComposerState();
        },
      },
      icon('refresh', 15),
      t('Valores recomendados'),
    ),
  );
  renderContext();
}

function setPanel(open: boolean) {
  settings.panel = open;
  saveSettings();
  el.shell.classList.toggle('panel-open', open);
  $('btn-settings').setAttribute('aria-expanded', String(open));
}

// ---------------------------------------------------------------------------------------------
// Dialogs

function dialog(id: string): HTMLDialogElement {
  return $<HTMLDialogElement>(id);
}

function confirmDialog(title: string, text: string, okLabel: string): Promise<boolean> {
  const d = dialog('dlg-confirm');
  return new Promise((resolve) => {
    const ok = h('button', { type: 'button', class: 'btn danger' }, okLabel);
    const cancel = h('button', { type: 'button', class: 'btn ghost' }, t('Cancelar'));
    d.replaceChildren(h('div', { class: 'dlg-body' }, h('h2', {}, title), h('p', {}, text), h('div', { class: 'dlg-actions' }, cancel, ok)));
    const done = (v: boolean) => {
      d.close();
      resolve(v);
    };
    ok.onclick = () => done(true);
    cancel.onclick = () => done(false);
    d.onclose = () => resolve(false);
    d.showModal();
    cancel.focus();
  });
}

function showError(text: string) {
  const d = dialog('dlg-confirm');
  const ok = h('button', { type: 'button', class: 'btn primary', onclick: () => d.close() }, t('Entendido'));
  d.replaceChildren(h('div', { class: 'dlg-body' }, h('h2', {}, t('Algo ha fallado')), h('p', {}, text), h('div', { class: 'dlg-actions' }, ok)));
  d.showModal();
}

function cardView(m: ModelInfo, compact = false): HTMLElement {
  const c = card(m.id);
  return h(
    'div',
    { class: 'card-sheet' },
    h('div', { class: 'card-head' }, h('h2', {}, m.name), h('span', { class: `tier ${m.tier}` }, tierLabel(m.tier))),
    h(
      'dl',
      { class: 'card-facts' },
      h('div', {}, h('dt', {}, t('Parámetros')), h('dd', {}, m.params)),
      h('div', {}, h('dt', {}, t('Descarga')), h('dd', {}, fmtBytes(downloadBytes(m)))),
      h('div', {}, h('dt', {}, t('Contexto en la web')), h('dd', {}, t('{n} tokens', { n: fmtNum(m.context) }))),
      h('div', {}, h('dt', {}, t('Licencia')), h('dd', {}, h('a', { href: m.license.url, target: '_blank', rel: 'noopener' }, m.license.name))),
    ),
    h('h3', {}, t('Qué hace bien')),
    h('ul', { class: 'good' }, ...c.good.map((x) => h('li', {}, x))),
    h('h3', {}, t('Límites')),
    h('ul', { class: 'limits' }, ...c.limits.map((x) => h('li', {}, x))),
    h('h3', {}, t('Equipo necesario')),
    h('p', {}, c.hardware),
    h('h3', {}, t('Licencia')),
    h('p', {}, c.licenseNote),
    compact
      ? null
      : h(
          'p',
          { class: 'card-src' },
          h('a', { href: m.source, target: '_blank', rel: 'noopener' }, t('Modelo original'), icon('external', 13)),
          h('a', { href: `https://huggingface.co/${m.repo}/tree/${m.revision}`, target: '_blank', rel: 'noopener' }, t('Archivos ONNX usados'), icon('external', 13)),
        ),
  );
}

function showCard(m: ModelInfo) {
  const d = dialog('dlg-card');
  d.replaceChildren(
    h('button', { type: 'button', class: 'icon-btn dlg-close', 'aria-label': t('Cerrar'), onclick: () => d.close() }, icon('x', 18)),
    h('div', { class: 'dlg-body' }, cardView(m)),
  );
  d.showModal();
}

async function refreshCache() {
  await Promise.all(AVAILABLE.map(async (m) => cacheInfo.set(m.id, await cacheStatus(m))));
}

async function openModels() {
  const d = dialog('dlg-models');
  const body = h('div', { class: 'dlg-body' });
  d.replaceChildren(h('button', { type: 'button', class: 'icon-btn dlg-close', 'aria-label': t('Cerrar'), onclick: () => d.close() }, icon('x', 18)), body);
  d.showModal();
  const draw = async () => {
    await refreshCache();
    const est = await storageEstimate();
    const orphans = await orphanBytes();
    body.replaceChildren(h('h2', {}, t('Modelos')));
    body.append(h('p', { class: 'muted' }, t('Los modelos se guardan en el almacenamiento de este navegador para no descargarlos otra vez. Puedes borrarlos cuando quieras.')));
    const list = h('div', { class: 'model-rows' });
    for (const m of AVAILABLE) {
      const c = cacheInfo.get(m.id)!;
      const f = gpu ? fit(m, gpu) : { ok: true as const };
      const complete = isComplete(c);
      const partial = !complete && c.present > 0;
      const actions = h('div', { class: 'row-actions' });
      if (complete || partial)
        actions.append(
          h(
            'button',
            {
              type: 'button',
              class: 'btn ghost small',
              onclick: async () => {
                if (!(await confirmDialogNested(t('¿Borrar {m} de este navegador?', { m: m.name }), t('Se liberarán {s}. Si lo vuelves a usar, se descargará otra vez.', { s: fmtBytes(c.bytes) })))) return;
                if (engine.loaded === m.id) engine.unload();
                await deleteModel(m);
                toast(t('{m} borrado', { m: m.name }));
                await draw();
                renderPicker();
                renderComposerState();
                renderThreadIfEmpty();
              },
            },
            icon('trash', 15),
            t('Borrar'),
          ),
        );
      if (!complete && f.ok)
        actions.append(h('button', { type: 'button', class: 'btn small', onclick: () => (d.close(), void selectModel(m.id, true)) }, icon('download', 15), partial ? t('Completar descarga') : t('Descargar')));
      list.append(
        h(
          'div',
          { class: 'model-row' },
          h(
            'div',
            { class: 'row-info' },
            h('div', { class: 'row-title' }, h('strong', {}, m.name), h('span', { class: `tier ${m.tier}` }, tierLabel(m.tier)), engine.loaded === m.id ? h('span', { class: 'tag-ok' }, t('En uso')) : null),
            h(
              'div',
              { class: 'row-meta' },
              complete ? t('Descargado · ocupa {s}', { s: fmtBytes(c.bytes) }) : partial ? t('Descarga incompleta · {s} guardados', { s: fmtBytes(c.bytes) }) : t('No descargado · {s}', { s: fmtBytes(downloadBytes(m)) }),
            ),
            f.ok ? null : h('div', { class: 'opt-warn' }, blockReason(f.why)),
          ),
          actions,
        ),
      );
    }
    body.append(list);
    if (orphans > 0)
      body.append(
        h(
          'div',
          { class: 'orphans' },
          h('span', {}, t('Archivos de versiones antiguas: {s}', { s: fmtBytes(orphans) })),
          h('button', { type: 'button', class: 'btn ghost small', onclick: async () => (await deleteOrphans(), await draw()) }, icon('trash', 15), t('Borrar')),
        ),
      );
    if (est) body.append(h('p', { class: 'hint' }, t('Este sitio ocupa {u} en el navegador (espacio disponible: {q}).', { u: fmtBytes(est.usage), q: fmtBytes(est.quota) })));
    body.append(gpuSummary());
    if (engine.loaded) body.append(benchBox(modelById(engine.loaded)!));
  };
  await draw();
}

/** Semicircular gauge (inline SVG); `set(fraction)` animates the arc. */
function gauge(): { el: SVGSVGElement; set: (f: number) => void; running: (on: boolean) => void } {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 160 92');
  svg.classList.add('gauge');
  const d = 'M 14 82 A 66 66 0 0 1 146 82';
  const track = document.createElementNS(ns, 'path');
  track.setAttribute('d', d);
  track.classList.add('gauge-track');
  const arc = document.createElementNS(ns, 'path');
  arc.setAttribute('d', d);
  arc.setAttribute('pathLength', '100');
  arc.classList.add('gauge-arc');
  svg.append(track, arc);
  return {
    el: svg,
    set: (f) => requestAnimationFrame(() => (arc.style.strokeDashoffset = String(100 - Math.max(0.02, Math.min(1, f)) * 100))),
    running: (on) => svg.classList.toggle('running', on),
  };
}

const LEVEL_TEXT: Record<Level, [string, string]> = {
  excellent: ['Excelente', 'Escribe mucho más rápido de lo que puedes leer.'],
  great: ['Muy buena', 'Escribe más rápido de lo que lees: las respuestas fluyen sin esperas.'],
  good: ['Buena', 'Escribe más o menos a tu ritmo de lectura.'],
  fair: ['Aceptable', 'Escribe algo más despacio de lo que lees. Se puede usar con un poco de paciencia.'],
  slow: ['Lenta', 'Tendrás que esperar bastante en cada respuesta.'],
};
const WAIT_TEXT: Record<Wait, string> = {
  instant: 'Empieza a responder al instante.',
  quick: 'Empieza a responder en un momento.',
  noticeable: 'Tarda unos segundos en empezar a responder.',
  long: 'Tarda bastante en empezar a responder.',
};

function benchBox(m: ModelInfo): HTMLElement {
  const g = gauge();
  const value = h('div', { class: 'bench-score' }, h('strong', {}, '—'), h('span', {}, '/ 10'));
  const verdict = h('div', { class: 'bench-verdict' }, t('Mide cómo de rápido va {m} en tu equipo.', { m: m.name }));
  const details = h('div', { class: 'bench-details' });
  const btn = h('button', { type: 'button', class: 'btn primary small' }, icon('gauge', 15), t('Medir rendimiento'));
  btn.addEventListener('click', async () => {
    if (generating) return toast(t('Espera a que termine la respuesta o párala.'));
    btn.disabled = true;
    g.running(true);
    g.set(0.02);
    value.firstElementChild!.textContent = '…';
    verdict.textContent = t('Midiendo… el modelo escribe un texto de prueba dos veces.');
    details.replaceChildren();
    try {
      const r = await engine.bench();
      const sc = score(r.tokensPerSecond);
      const lv = level(r.tokensPerSecond);
      const wt = wait(r.prefillMs);
      g.running(false);
      g.set(sc / 10);
      value.firstElementChild!.textContent = fmtNum(sc, 1);
      verdict.replaceChildren(h('strong', {}, t(LEVEL_TEXT[lv][0])), h('span', {}, t(LEVEL_TEXT[lv][1])));
      const lighter = AVAILABLE.filter((x) => x.gpuMemory < m.gpuMemory && (!gpu || fit(x, gpu).ok)).sort((a, b) => b.gpuMemory - a.gpuMemory)[0];
      const weights = m.files.filter((f) => f.path.includes('.onnx')).reduce((a, f) => a + f.bytes, 0);
      const report = {
        model: m.id,
        revision: m.revision,
        gpu: gpu && { vendor: gpu.vendor, architecture: gpu.architecture, description: gpu.description, maxBufferSize: gpu.maxBufferSize, software: gpu.software },
        userAgent: navigator.userAgent,
        downloadBytes: downloadBytes(m),
        weightBytesOnGpu: weights,
        score: sc,
        ...r,
        date: new Date().toISOString(),
      };
      details.replaceChildren(
        h('ul', { class: 'bench-facts' },
          h('li', {}, icon('zap', 15), h('span', {}, t('{v} palabras por segundo, aproximadamente', { v: fmtNum(r.tokensPerSecond * 0.7, 0) }))),
          h('li', {}, icon('message', 15), h('span', {}, t(WAIT_TEXT[wt]))),
        ),
        ...((lv === 'fair' || lv === 'slow') && lighter ? [h('p', { class: 'bench-tip' }, t('Para que vaya más rápido, prueba {m}.', { m: lighter.name }))] : []),
        ...(gpu?.software ? [h('p', { class: 'bench-tip' }, t('Tu navegador usa la CPU en lugar de la GPU: activa la aceleración por hardware.'))] : []),
        h(
          'details',
          { class: 'bench-more' },
          h('summary', {}, t('Datos técnicos')),
          h(
            'dl',
            {},
            h('div', {}, h('dt', {}, t('Generación')), h('dd', {}, t('{v} tokens/s', { v: fmtNum(r.tokensPerSecond, 1) }))),
            h('div', {}, h('dt', {}, t('Lectura del prompt')), h('dd', {}, t('{v} tokens/s ({n} tokens en {d})', { v: fmtNum(r.prefillTokensPerSecond, 0), n: r.promptTokens, d: fmtDuration(r.prefillMs) }))),
            h('div', {}, h('dt', {}, t('Pesos en la GPU')), h('dd', {}, fmtBytes(weights))),
          ),
          h('div', { class: 'bench-copy' }, copyButton(() => JSON.stringify(report, null, 2), t('Copiar resultado (JSON)')), h('span', { class: 'hint' }, t('Copiar resultado (JSON)'))),
        ),
      );
      btn.replaceChildren(icon('refresh', 15), t('Repetir'));
    } catch (e) {
      g.running(false);
      g.set(0);
      value.firstElementChild!.textContent = '—';
      verdict.replaceChildren(h('span', { class: 'opt-warn' }, e instanceof Error ? e.message : String(e)));
    }
    btn.disabled = false;
    engine.reset();
  });
  return h(
    'div',
    { class: 'bench' },
    h('div', { class: 'bench-head' }, h('strong', {}, t('Rendimiento en tu equipo')), btn),
    h('div', { class: 'bench-main' }, h('div', { class: 'bench-gauge' }, g.el, value), verdict),
    details,
  );
}

/** A confirmation shown above another open dialog. */
function confirmDialogNested(title: string, text: string): Promise<boolean> {
  return confirmDialog(title, text, t('Borrar'));
}

function gpuSummary(): HTMLElement {
  const g = gpu;
  if (!g) return h('div');
  const rows: [string, string][] = [
    [t('WebGPU'), g.ok ? t('Disponible') : t('No disponible')],
    [t('GPU'), [g.vendor, g.architecture, g.description].filter(Boolean).join(' · ') || t('El navegador no lo indica')],
    ['shader-f16', g.f16 ? t('Sí') : t('No')],
    [t('Búfer máximo'), g.maxBufferSize ? fmtBytes(Math.min(g.maxBufferSize, g.maxStorageBufferBindingSize)) : '—'],
    [t('Memoria del equipo'), g.deviceMemory ? (g.deviceMemory >= 8 ? t('8 GB o más') : `${g.deviceMemory} GB`) : t('El navegador no lo indica')],
  ];
  return h(
    'details',
    { class: 'gpu-info' },
    h('summary', {}, icon('cpu', 15), t('Tu equipo')),
    h('dl', {}, ...rows.map(([k, v]) => h('div', {}, h('dt', {}, k), h('dd', {}, v)))),
    g.software ? h('p', { class: 'opt-warn' }, t('El navegador usa un adaptador por software: los modelos irán muy lentos.')) : null,
  );
}

// ---------------------------------------------------------------------------------------------
// Boot

async function boot() {
  mountAurora();
  wireGlow();
  applyTheme();
  hydrateIcons();
  $('btn-theme').addEventListener('click', () => {
    settings.theme = isDark() ? 'light' : 'dark';
    saveSettings();
    const d = document as Document & { startViewTransition?: (cb: () => void) => unknown };
    if (d.startViewTransition && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
      document.documentElement.classList.add('vt-theme');
      const vt = d.startViewTransition(applyTheme) as { finished?: Promise<void> };
      void vt.finished?.finally(() => document.documentElement.classList.remove('vt-theme'));
    } else applyTheme();
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
  $('btn-new').addEventListener('click', newConversation);
  $('btn-menu').addEventListener('click', () => setDrawer(true));
  $('scrim').addEventListener('click', () => (setDrawer(false), setPanel(false)));
  $('btn-close-drawer').addEventListener('click', () => setDrawer(false));
  $('btn-settings').addEventListener('click', () => setPanel(!el.shell.classList.contains('panel-open')));
  $('btn-close-settings').addEventListener('click', () => setPanel(false));
  $('btn-models').addEventListener('click', () => void openModels());
  const search = $<HTMLInputElement>('search');
  search.addEventListener('input', () => {
    query = search.value.trim();
    renderList();
  });
  for (const d of document.querySelectorAll('dialog')) d.addEventListener('click', (e) => e.target === d && d.close());
  setPanel(settings.panel && matchMedia('(min-width: 1100px)').matches);
  renderComposerState();

  conversations = await listConversations().catch(() => []);
  renderList();
  gpu = await probeGpu();
  if (DEV_Q4 && gpu.reason === 'no-f16') gpu = { ...gpu, ok: true, f16: true, reason: undefined };
  await refreshCache();
  const rec = recommend(gpu);
  const saved = settings.model && modelById(settings.model);
  const initial = saved && fit(saved, gpu).ok ? saved.id : rec;
  if (initial) {
    selected = modelById(initial)!;
    settings.model = initial;
  }
  renderPicker();
  renderSettings();
  renderThread();
  renderComposerState();
  if (selected) {
    const c = cacheInfo.get(selected.id);
    if (c && isComplete(c) && gpu.ok) void loadModel(selected.id);
  }
  // Examples on the landing page open the chat with their prompt ready to send.
  const q = new URLSearchParams(location.search).get('q');
  if (q) {
    el.input.value = q.slice(0, 2000);
    autosize();
    renderComposerState();
    history.replaceState(null, '', location.pathname);
  }
  el.input.focus();
  document.documentElement.classList.add('ready');
}

void boot();

// Expose a tiny hook for the automated benchmark (tools/bench): only reads timings, never data.
(window as unknown as Record<string, unknown>).__chat = { engine, lang, localPath, setMood };
