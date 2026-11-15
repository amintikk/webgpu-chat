import '@fontsource-variable/inter';
import '../tokens.css';
import './app.css';
import { Engine } from '../engine/client.ts';
import { modelWarning, probeGpu, recommend, type GpuReport } from '../engine/gpu.ts';
import { AVAILABLE, MODELS, downloadBytes, modelById, type ModelId, type ModelInfo } from '../engine/models.ts';
import type { ChatMessage, GenOptions, GenStats } from '../engine/protocol.ts';
import { lang, locale, localPath, t } from '../i18n.ts';
import { $, fmtBytes, fmtDuration, fmtNum, h, toast } from '../ui/dom.ts';
import { mountAurora, pulseAurora, setAuroraProgress, setMood, wireGlow } from '../ui/aurora.ts';
import { hydrateIcons, icon } from '../ui/icons.ts';
import { cacheStatus, deleteModel, deleteOrphans, isComplete, orphanBytes, storageEstimate, type CacheEntry } from './cache.ts';
import { clearConversations, deleteConversation, listConversations, newId, saveConversation, titleFrom, type Conversation, type StoredImage, type StoredMessage } from './db.ts';
import { card, tierLabel } from './modelcards.ts';
import { level, score, wait, type Level, type Wait } from './score.ts';
import { copyButton, renderMarkdown } from './render.ts';
import { canSpeak, localVoices, speakButton, speakSample, voicePrefs } from './speak.ts';
import { exportBackup, readBackup } from './backup.ts';
import { moveLegacyCache } from '../engine/cachemove.ts';
import { ACCEPT, isImageFile, readAttachment, readImage, withAttachments, type Attachment } from './attach.ts';
import { applyUrlDebug, clearDebug, debugOn, dlog, memoryText, reportText, setDebug, startMemoryWatch, stopMemoryWatch, watchErrors } from '../ui/debug.ts';

// ---------------------------------------------------------------------------------------------
// Settings (localStorage: preferences only; conversations are in IndexedDB)

interface ModelSettings {
  temperature: number;
  topP: number;
  maxTokens: number;
  reasoning: boolean;
  /** Advanced; the model authors' values when unset. */
  topK?: number;
  repetitionPenalty?: number;
}
interface Settings {
  theme: 'auto' | 'light' | 'dark';
  model: ModelId | null;
  system: string;
  perModel: Partial<Record<ModelId, ModelSettings>>;
  panel: boolean;
  /** Enter sends (Shift+Enter: new line). Off: Ctrl/Cmd+Enter sends and Enter adds a line. */
  enterSends: boolean;
  /** Read aloud: a local voice (its voiceURI, '' = the system's default for the language) and its speed. */
  voice: string;
  rate: number;
}
const KEY = 'webgpu-chat-settings';
const settings: Settings = (() => {
  const d: Settings = { theme: 'auto', model: null, system: '', perModel: {}, panel: false, enterSends: true, voice: '', rate: 1 };
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
/** fromCache: every file is already on disk, so the bar reads the cache rather than downloading. */
let loading: { model: ModelId; loaded: number; total: number; stage: string; fromCache?: boolean } | null = null;
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

/** Files attached in the composer, waiting to be sent (read in attach.ts). */
interface Pending {
  id: number;
  name: string;
  size: number;
  file?: Attachment; // set once read (text and PDF)
  image?: StoredImage; // set once read (pictures)
}
const isRead = (p: Pending) => !!(p.file || p.image);

/** Object URLs for picture thumbnails, one per stored picture. */
const thumbUrls = new WeakMap<Blob, string>();
function thumbUrl(b: Blob): string {
  let u = thumbUrls.get(b);
  if (!u) thumbUrls.set(b, (u = URL.createObjectURL(b)));
  return u;
}
let pending: Pending[] = [];
let pendingSeq = 0;
const MAX_FILES = 5;

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
    { class: `conv${active ? ' active' : ''}` },
    h('button', { type: 'button', class: 'conv-open', onclick: () => openConversation(c.id), 'aria-current': active ? 'page' : null }, title),
    h(
      'div',
      { class: 'conv-actions' },
      h('button', { type: 'button', class: 'icon-btn sm', title: t('Renombrar'), 'aria-label': t('Renombrar'), onclick: () => rename(c, row) }, icon('pencil', 15)),
      h('button', { type: 'button', class: 'icon-btn sm', title: t('Descargar (Markdown)'), 'aria-label': t('Descargar (Markdown)'), onclick: () => exportConversation(c) }, icon('download', 15)),
      h('button', { type: 'button', class: 'icon-btn sm', title: t('Borrar'), 'aria-label': t('Borrar'), onclick: () => remove(c) }, icon('trash', 15)),
    ),
  );
  return row;
}

/** The conversation as a Markdown file, saved by the browser (nothing is uploaded). */
function exportConversation(c: Conversation) {
  const lines = [`# ${c.title}`, '', `_${new Date(c.createdAt).toLocaleString(locale)} · WebGPU Chat_`, ''];
  for (const m of c.messages) {
    if (m.role === 'user') {
      lines.push(`## ${t('Tú')}`, '');
      for (const i of m.images ?? []) lines.push(`_[${t('imagen')}: ${i.name}]_`);
      for (const f of m.files ?? []) lines.push(`_[${t('archivo')}: ${f.name}]_`);
      if (m.images?.length || m.files?.length) lines.push('');
      lines.push(m.content, '');
    } else {
      lines.push(`## ${modelById(m.model ?? '')?.name ?? t('Asistente')}`, '');
      if (m.reasoning) lines.push(`<details><summary>${t('Razonamiento')}</summary>`, '', m.reasoning, '', '</details>', '');
      lines.push(m.error ? `> ${m.error}` : m.content, '');
    }
  }
  const a = h('a', { href: URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/markdown' })), download: `${c.title.replace(/[\\/:*?"<>|]+/g, ' ').trim().slice(0, 80) || 'chat'}.md` });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
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
  if (generating || engine.busy) {
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
      h('span', {}, t('Funciona en tu equipo, sin internet una vez descargado. Lo que escribas no sale de él.')),
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
        h('button', { type: 'button', class: 'suggestion', 'data-glow': '', onclick: () => pickSuggestion(s) }, s),
      ),
    ),
  );
  return box;
}

/** A suggestion fills the composer and leaves it ready to send (button enabled, context measured). */
function pickSuggestion(text: string) {
  el.input.value = text;
  autosize();
  renderComposerState();
  el.input.focus();
  el.input.setSelectionRange(text.length, text.length);
  if (!generating && !loading) setMood('typing');
  void updateContext();
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
      m.images?.length ? h('div', { class: 'msg-images' }, ...m.images.map((i) => h('img', { src: thumbUrl(i.blob), alt: i.name, title: i.name, loading: 'lazy' }))) : null,
      m.files?.length ? h('div', { class: 'msg-files' }, ...m.files.map((f) => fileChip(f.name, `${fmtBytes(f.size)} · ${t('{n} caracteres', { n: fmtNum(f.text.length) })}`))) : null,
      m.content ? h('div', { class: 'bubble' }, m.content) : null,
      m.content ? h('div', { class: 'msg-tools' }, copyButton(() => m.content, t('Copiar'))) : null,
    );
  }
  const body = h('div', { class: 'md' });
  body.append(renderMarkdown(m.content));
  const view = h('div', { class: 'msg assistant' });
  if (m.reasoning) view.append(reasoningBox(m.reasoning, false));
  view.append(body);
  if (m.error) view.append(h('div', { class: 'msg-error' }, icon('alert', 15), h('span', {}, m.error)));
  else if (!m.content.trim() && m.stats?.stopped === 'length')
    view.append(
      h(
        'div',
        { class: 'msg-error' },
        icon('info', 15),
        h('span', {}, m.reasoning ? t('Se agotó el máximo de tokens mientras razonaba, antes de responder. Sube el máximo en Ajustes o desactiva Razonar y pulsa Regenerar.') : t('Se agotó el máximo de tokens antes de responder. Súbelo en Ajustes y pulsa Regenerar.')),
      ),
    );
  const isLast = current && index === current.messages.length - 1;
  view.append(
    h(
      'div',
      { class: 'msg-tools' },
      copyButton(() => m.content, t('Copiar respuesta')),
      speakButton(() => m.content, (msg) => toast(msg)),
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
    const warning = gpu ? modelWarning(x, gpu) : null;
    const s = statusOf(x);
    const opt = h(
      'div',
      { class: `picker-opt${x.id === m?.id ? ' selected' : ''}`, role: 'option', 'aria-selected': x.id === m?.id ? 'true' : 'false', tabIndex: 0 },
      h(
        'div',
        { class: 'opt-main' },
        h('div', { class: 'opt-title' }, h('strong', {}, x.name), h('span', { class: `tier ${x.tier}` }, tierLabel(x.tier)), x.id === rec ? h('span', { class: 'tag-rec' }, t('Para tu equipo')) : null),
        h('div', { class: 'opt-meta' }, h('span', { class: `dot ${s.cls}` }), h('span', {}, s.label)),
        warning ? h('div', { class: 'opt-warn' }, warningText(warning)) : null,
      ),
      h('button', { type: 'button', class: 'icon-btn sm', title: t('Ficha del modelo'), 'aria-label': t('Ficha del modelo'), onclick: (e: Event) => (e.stopPropagation(), closePicker(), showCard(x)) }, icon('info', 16)),
    );
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

function warningText(why: string): string {
  switch (why) {
    case 'no-f16':
      return t('Tu GPU no admite shader-f16.');
    case 'buffer':
      return t('Tu GPU no admite búferes tan grandes como los de este modelo.');
    case 'memory-low':
      return t('El navegador indica poca memoria; puede no caber.');
    case 'software':
      return t('Tu GPU es un adaptador por software: irá muy lento.');
    case 'memory-unknown':
      return t('Necesita una GPU potente; puede no caber en la tuya.');
    case 'mobile':
      return t('En móvil puede faltar memoria y cerrarse la pestaña. Puedes probarlo igualmente.');
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
  if (generating || engine.busy) {
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

// While a model loads, its id is kept in localStorage. If the page opens and finds it, the previous load never
// finished: almost always the system closed the tab for lack of memory. Then the model is not loaded again on
// its own (that would close the tab again) and the user is told why.
const LOADING_KEY = 'webgpu-chat:loading';
const markLoading = (id: ModelId | null) => {
  try {
    if (id) localStorage.setItem(LOADING_KEY, id);
    else localStorage.removeItem(LOADING_KEY);
  } catch {
    /* storage unavailable */
  }
};
function takeCrashedLoad(): ModelInfo | null {
  try {
    const id = localStorage.getItem(LOADING_KEY);
    localStorage.removeItem(LOADING_KEY);
    return id ? (modelById(id) ?? null) : null;
  } catch {
    return null;
  }
}

async function loadModel(id: ModelId, profile = false) {
  const m = modelById(id);
  if (!m || loading?.model === id) return;
  try {
    await navigator.storage?.persist?.();
  } catch {
    /* best effort */
  }
  const cached = cacheInfo.get(id);
  loading = { model: id, loaded: 0, total: downloadBytes(m), stage: 'download', fromCache: !!cached && isComplete(cached) };
  setAuroraProgress(0);
  setMood('loading');
  renderPicker();
  renderThreadIfEmpty();
  renderLoad();
  renderComposerState();
  markLoading(id);
  dlog(`carga de ${m.name} (${id}) · ${memoryText()}`);
  startMemoryWatch(`${m.name}`);
  let lastDecile = -1;
  try {
    const t0 = performance.now();
    await engine.load(id, {
      onProgress: (l, tot) => {
        if (loading) Object.assign(loading, { loaded: l, total: tot, stage: 'download' });
        setAuroraProgress(tot ? l / tot : 0);
        const decile = tot ? Math.floor((l / tot) * 10) : -1;
        if (decile !== lastDecile) {
          lastDecile = decile;
          dlog(`descarga ${decile * 10}% (${fmtBytes(l)} de ${fmtBytes(tot)})`);
        }
        renderLoad();
      },
      onStage: (s) => {
        dlog(`etapa: ${s}`);
        if (loading) loading.stage = s;
        if (s !== 'download') setAuroraProgress(1);
        renderLoad();
      },
    }, DEV_Q4 ? 'q4' : undefined, profile);
    loading = null;
    markLoading(null);
    dlog(`CARGADO en ${fmtDuration(performance.now() - t0)} · ${memoryText()}`);
    stopMemoryWatch();
    cacheInfo.set(id, await cacheStatus(m));
    toast(t('{m} listo en {d}', { m: m.name, d: fmtDuration(performance.now() - t0) }));
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg === 'superseded') return;
    dlog(`ERROR al cargar: ${e instanceof Error ? `${e.name}: ${e.message}\n${e.stack ?? ''}` : msg}`);
    dlog(`memoria en el error · ${memoryText()}`);
    stopMemoryWatch();
    // A load that failed part-way can keep GPU memory in the worker: start a fresh worker before the next try.
    engine.restart();
    dlog('motor reiniciado tras el error');
    loading = null;
    markLoading(null);
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
  // The browser could not reserve memory for the weights in the page (not the GPU's own memory): "Array buffer
  // allocation failed" and similar. Says so, instead of blaming the GPU.
  if (msg.includes('STORAGE_FULL'))
    return (
      t('No se ha podido guardar {m} en el disco: lo más probable es que falte espacio libre (necesita unos {s}). Libera espacio y vuelve a intentarlo; lo ya descargado se conserva.', { m: m.name, s: fmtBytes(downloadBytes(m)) }) +
      ` (${msg})`
    );
  if (/array buffer|allocation failed|ArrayBuffer|RangeError|Invalid array length/i.test(msg))
    return (
      t('El navegador no ha podido reservar memoria para cargar {m} (sus pesos ocupan unos {s} en el navegador). Cierra otras pestañas y aplicaciones y vuelve a intentarlo. Si sigue igual, prueba LFM3 1.7B, que es mucho más ligero.', { m: m.name, s: fmtBytes(downloadBytes(m)) }) +
      ` (${msg})`
    );
  if (/memory|OOM|device.*lost|GPUBuffer|out of/i.test(msg))
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
        : loading.fromCache
          ? t('Leyendo {m} del disco · {a} de {b}', { m: m.name, a: fmtBytes(loading.loaded), b: fmtBytes(loading.total) })
          : t('Descargando {m} · {a} de {b}', { m: m.name, a: fmtBytes(loading.loaded), b: fmtBytes(loading.total) });
  box.hidden = false;
  box.replaceChildren(
    h('div', { class: 'load-row' }, h('span', {}, label), h('span', { class: 'load-pct' }, loading.stage === 'download' && pct < 99.9 ? `${fmtNum(pct, 0)}\u00a0%` : '')),
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
  document.querySelectorAll<HTMLButtonElement>('button[data-engine-test]').forEach((b) => { b.disabled = !engine.loaded || engine.busy; });
  renderLoad();
  const ready = !!selected && engine.loaded === selected.id && !loading && !engine.busy;
  el.send.replaceChildren(icon(generating ? 'stop' : 'arrowUp', 18));
  el.send.title = generating ? t('Parar (Esc)') : t('Enviar (Intro)');
  el.send.setAttribute('aria-label', el.send.title);
  el.send.classList.toggle('stop', generating);
  const hasInput = !!el.input.value.trim() || readyFiles().length > 0 || readyImages().length > 0;
  el.send.disabled = !generating && (!ready || !hasInput || !pending.every(isRead));
  el.input.disabled = !!gpu && !gpu.ok;
  el.input.placeholder = gpu && !gpu.ok ? t('WebGPU no disponible') : ready || generating ? t('Escribe un mensaje…') : loading ? t('Cargando el modelo…') : t('Descarga un modelo para empezar');
  const r = selected ? ms(selected).reasoning : false;
  el.reasoningChip.classList.toggle('on', r);
  el.reasoningChip.setAttribute('aria-pressed', String(r));
}

function genOptions(m: ModelInfo): GenOptions {
  const s = ms(m);
  return { temperature: s.temperature, topP: s.topP, topK: s.topK ?? m.defaults.topK, repetitionPenalty: s.repetitionPenalty ?? m.defaults.repetitionPenalty, maxTokens: s.maxTokens, reasoning: s.reasoning };
}

function toChat(msgs: StoredMessage[], model: ModelInfo | null = selected): ChatMessage[] {
  const out: ChatMessage[] = [];
  if (settings.system.trim()) out.push({ role: 'system', content: settings.system.trim() });
  const sees = !!model?.vision;
  for (const m of msgs) {
    if (m.role === 'assistant' && !m.content && !m.reasoning) continue;
    if (m.role === 'assistant') {
      out.push({ role: 'assistant', content: m.content, reasoning: m.reasoning });
      continue;
    }
    // A model that cannot see pictures is told they were there, so the conversation still makes sense.
    const unseen = !sees && m.images?.length ? m.images.map((i) => `[${t('Imagen adjunta que este modelo no puede ver')}: ${i.name}]`).join('\n') : '';
    const content = [unseen, withAttachments(m.content, m.files)].filter(Boolean).join('\n\n');
    out.push(sees && m.images?.length ? { role: 'user', content, images: m.images.map((i) => ({ id: i.id, blob: i.blob })) } : { role: 'user', content });
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

// ---------------------------------------------------------------------------------------------
// Attachments waiting in the composer


const readyFiles = (): Attachment[] => pending.flatMap((p) => (p.file ? [p.file] : []));
const readyImages = (): StoredImage[] => pending.flatMap((p) => (p.image ? [p.image] : []));

function addFiles(files: File[]) {
  if (gpu && !gpu.ok) return;
  for (const f of files) {
    if (pending.length >= MAX_FILES) {
      toast(t('Como mucho {n} archivos por mensaje.', { n: MAX_FILES }));
      break;
    }
    const p: Pending = { id: ++pendingSeq, name: f.name, size: f.size };
    pending.push(p);
    const image = isImageFile(f);
    if (image && selected && !selected.vision) toast(t('{m} no puede ver imágenes. Qwen3.5 4B y Gemma 4 12B sí.', { m: selected.name }));
    (image
      ? readImage(f).then((img) => {
          p.image = img;
          p.name = img.name;
          p.size = img.size;
        })
      : readAttachment(f).then((a) => {
          p.file = a;
        })
    )
      .catch((e: unknown) => {
        pending = pending.filter((x) => x !== p);
        toast(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        renderAttachments();
        renderComposerState();
        void updateContext();
      });
  }
  renderAttachments();
  renderComposerState();
  el.input.focus();
}

function fileChip(name: string, detail: string, onRemove?: () => void, thumb?: string): HTMLElement {
  return h(
    'div',
    { class: 'file-chip' },
    thumb ? h('img', { class: 'file-thumb', src: thumb, alt: '' }) : icon('fileText', 16),
    h('span', { class: 'file-meta' }, h('strong', {}, name), h('span', {}, detail)),
    onRemove ? h('button', { type: 'button', class: 'file-x', 'aria-label': t('Quitar {f}', { f: name }), title: t('Quitar'), onclick: onRemove }, icon('x', 14)) : null,
  );
}

function renderAttachments() {
  const box = $('attachments');
  box.hidden = pending.length === 0;
  box.replaceChildren(
    ...pending.map((p) =>
      fileChip(
        p.name,
        p.image ? `${fmtBytes(p.size)} · ${t('imagen')}` : p.file ? `${fmtBytes(p.size)} · ${t('{n} caracteres', { n: fmtNum(p.file.text.length) })}` : t('Leyendo…'),
        () => {
          pending = pending.filter((x) => x !== p);
          renderAttachments();
          renderComposerState();
          void updateContext();
        },
        p.image ? thumbUrl(p.image.blob) : undefined,
      ),
    ),
  );
}

async function send() {
  const text = el.input.value.trim();
  const files = readyFiles();
  const images = readyImages();
  if ((!text && !files.length && !images.length) || !pending.every(isRead) || generating || engine.busy || !selected || engine.loaded !== selected.id) return;
  if (images.length && !selected.vision) {
    showError(t('{m} no puede ver imágenes. Cambia a Qwen3.5 4B o Gemma 4 12B, o quita la imagen.', { m: selected.name }));
    return;
  }
  const msg: StoredMessage = { role: 'user', content: text, ...(files.length ? { files } : {}), ...(images.length ? { images } : {}) };
  if (files.length || images.length) {
    // The new message alone (with its files) must leave room for the answer.
    const m = selected;
    const n = await engine.count(toChat([msg]), ms(m).reasoning);
    if (n + ms(m).maxTokens > m.context) {
      showError(
        t('Lo adjunto es demasiado largo para {m}: ocupa unos {n} tokens y el modelo admite {c} en total, contando los {r} reservados para la respuesta. Adjunta un fragmento más corto o reduce el máximo de tokens en Ajustes.', {
          m: m.name,
          n: fmtNum(n),
          c: fmtNum(m.context),
          r: fmtNum(ms(m).maxTokens),
        }),
      );
      return;
    }
  }
  el.input.value = '';
  pending = [];
  renderAttachments();
  autosize();
  if (!current) {
    current = { id: newId(), title: titleFrom(text || (files[0] ?? images[0])!.name), createdAt: Date.now(), updatedAt: Date.now(), model: selected.id, messages: [] };
  }
  current.messages.push(msg);
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
  let renderMs = 0;
  let paintedContent: string | null = null;
  const paint = () => {
    raf = 0;
    if (!pending) return;
    const paintStart = performance.now();
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
    if (p.content !== paintedContent) {
      body.replaceChildren(renderMarkdown(p.content));
      if (!p.thinking) body.append(h('span', { class: 'cursor' }));
      paintedContent = p.content;
    }
    scrollToEnd();
    renderMs += performance.now() - paintStart;
  };

  const msg: StoredMessage = { role: 'assistant', content: '', model: m.id };
  try {
    const r = await engine.generate(chat, opts, {
      onVision: (loaded, total) => {
        const note = loaded >= total ? t('Leyendo la imagen…') : t('Descargando el lector de imágenes de {m} (solo la primera vez): {p} %', { m: m.name, p: Math.floor((loaded / Math.max(1, total)) * 100) });
        body.replaceChildren(h('p', { class: 'vision-note' }, note), h('span', { class: 'cursor' }));
      },
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
    dlog(`CHAT ${m.id}: ${JSON.stringify({ ...r.stats, renderMs, temperature: opts.temperature, topP: opts.topP, reasoning: opts.reasoning })}`);
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
  const sends = settings.enterSends ? !e.shiftKey && !e.ctrlKey && !e.metaKey : e.ctrlKey || e.metaKey;
  if (e.key === 'Enter' && sends && !e.isComposing) {
    e.preventDefault();
    void send();
  }
});
// Attachments: the paperclip, dropping files on the chat, or pasting them.
const fileInput = $<HTMLInputElement>('file-input');
fileInput.accept = ACCEPT;
$('btn-attach').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  addFiles([...(fileInput.files ?? [])]);
  fileInput.value = '';
});
el.input.addEventListener('paste', (e) => {
  const files = [...(e.clipboardData?.files ?? [])];
  if (files.length) {
    e.preventDefault();
    addFiles(files);
  }
});
let dragDepth = 0;
const shellEl = $('shell');
shellEl.addEventListener('dragenter', (e) => {
  if (!e.dataTransfer?.types.includes('Files')) return;
  dragDepth++;
  shellEl.classList.add('dropping');
});
shellEl.addEventListener('dragleave', () => {
  if (--dragDepth <= 0) {
    dragDepth = 0;
    shellEl.classList.remove('dropping');
  }
});
shellEl.addEventListener('dragover', (e) => {
  if (e.dataTransfer?.types.includes('Files')) e.preventDefault();
});
shellEl.addEventListener('drop', (e) => {
  if (!e.dataTransfer?.files.length) return;
  e.preventDefault();
  dragDepth = 0;
  shellEl.classList.remove('dropping');
  addFiles([...e.dataTransfer.files]);
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
  const files = readyFiles();
  const images = readyImages();
  if (draft || files.length || images.length) msgs.push({ role: 'user', content: draft, files, images });
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
  const out = h('output', { class: 'set-value' }, fmt(value));
  const input = h('input', { type: 'range', min, max, step, value, 'aria-label': label });
  // The filled part of the track (CSS reads --p).
  const fill = () => input.style.setProperty('--p', `${((Number(input.value) - min) / (max - min)) * 100}%`);
  fill();
  input.addEventListener('input', () => {
    out.textContent = fmt(Number(input.value));
    fill();
    on(Number(input.value));
    saveSettings();
  });
  return h('div', { class: 'field' }, h('div', { class: 'field-row' }, h('label', {}, label), out), input, h('p', { class: 'hint' }, hint));
}

/** A group in the settings panel: a small heading over a card whose rows are separated by hairlines. */
function section(title: string, ...children: (Node | null)[]): HTMLElement {
  return h('section', { class: 'set-sec' }, h('h3', { class: 'set-head' }, title), h('div', { class: 'set-card' }, ...children));
}

/** A labelled on/off switch. */
function toggle(label: string, value: boolean, on: (v: boolean) => void, hint: string): HTMLElement {
  const sw = h('button', { type: 'button', class: `switch${value ? ' on' : ''}`, role: 'switch', 'aria-checked': String(value), 'aria-label': label }, h('i'));
  sw.addEventListener('click', () => {
    const v = sw.getAttribute('aria-checked') !== 'true';
    sw.classList.toggle('on', v);
    sw.setAttribute('aria-checked', String(v));
    on(v);
    saveSettings();
  });
  return h('div', { class: 'field' }, h('div', { class: 'field-row' }, h('label', {}, label), sw), h('p', { class: 'hint' }, hint));
}

/** The read-aloud voice and speed: applied at start-up and whenever they change. */
function applyPrefs() {
  voicePrefs.voice = settings.voice;
  voicePrefs.rate = settings.rate;
}

const PRESETS: [string, string][] = [
  ['Breve', 'Responde de forma breve y directa, sin rodeos.'],
  ['Detallado', 'Responde con detalle, paso a paso y con ejemplos cuando ayuden.'],
  ['Profesor', 'Actúa como un profesor paciente: explica con palabras sencillas y comprueba que se entiende.'],
  ['Programador', 'Eres un programador experto. Da código correcto y comentado, y explica brevemente las decisiones.'],
  ['Corrector', 'Corrige la ortografía, la gramática y el estilo del texto que te dé, y explica los cambios importantes.'],
  ['Traductor', 'Traduce el texto que te dé al español (o al inglés si ya está en español), manteniendo el tono.'],
];

function renderSettings() {
  const body = $('settings-body');
  const scroll = body.scrollTop;
  body.replaceChildren();
  const dec = (v: number) => v.toLocaleString(locale, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  // Model -------------------------------------------------------------------------------------
  if (selected) {
    const m = selected;
    const s = ms(m);
    const tog = h('button', { type: 'button', class: `switch${s.reasoning ? ' on' : ''}`, role: 'switch', 'aria-checked': String(s.reasoning), 'aria-label': t('Razonamiento') }, h('i'));
    tog.addEventListener('click', () => el.reasoningChip.click());
    const advanced = h(
      'details',
      { class: 'set-adv' },
      h('summary', {}, t('Avanzado'), icon('chevronDown', 14)),
      slider(t('Top-k'), 1, 100, 1, s.topK ?? m.defaults.topK, (v) => fmtNum(v), (v) => (s.topK = v), t('Solo se elige entre estas palabras más probables. Valor del modelo: {v}.', { v: fmtNum(m.defaults.topK) })),
      slider(t('Penalización por repetición'), 1, 1.5, 0.01, s.repetitionPenalty ?? m.defaults.repetitionPenalty, dec, (v) => (s.repetitionPenalty = v), t('Por encima de 1 evita que repita palabras y frases; demasiado alta empeora las respuestas. Valor del modelo: {v}.', { v: dec(m.defaults.repetitionPenalty) })),
    );
    body.append(
      section(m.name,
        slider(t('Temperatura'), 0, 1.5, 0.05, s.temperature, dec, (v) => (s.temperature = v), t('Más baja: respuestas más previsibles. Más alta: más variadas. 0 elige siempre la palabra más probable.')),
        slider(t('Top-p'), 0.05, 1, 0.05, s.topP, dec, (v) => (s.topP = v), t('Solo se eligen palabras entre las más probables hasta sumar esta probabilidad.')),
        slider(t('Máximo de tokens'), 64, Math.min(4096, m.context / 2), 64, s.maxTokens, (v) => fmtNum(v), (v) => ((s.maxTokens = v), renderContext()), t('Longitud máxima de cada respuesta, razonamiento incluido.')),
        h('div', { class: 'field' }, h('div', { class: 'field-row' }, h('label', {}, t('Razonamiento')), tog), h('p', { class: 'hint' }, t('El modelo piensa antes de responder. Mejora problemas de lógica, matemáticas y código, pero tarda bastante más.'))),
        advanced,
        h('div', { class: 'field ctx', id: 'ctx' }),
        h(
          'div',
          { class: 'set-actions' },
          h(
            'button',
            {
              type: 'button',
              class: 'btn ghost small',
              title: t('Volver a los valores recomendados para {m}', { m: m.name }),
              onclick: () => {
                settings.perModel[m.id] = defaultsFor(m);
                saveSettings();
                renderSettings();
                renderComposerState();
                toast(t('Valores recomendados de {m} restaurados.', { m: m.name }));
              },
            },
            icon('refresh', 15),
            t('Restablecer'),
          ),
          h('button', { type: 'button', class: 'btn ghost small', onclick: () => void openModels() }, icon('hardDrive', 15), t('Modelos')),
        ),
      ),
    );
  }

  // Instructions ------------------------------------------------------------------------------
  const sys = h('textarea', { rows: 4, value: settings.system, placeholder: t('Por ejemplo: «Responde siempre en español y de forma breve».'), 'aria-label': t('Instrucciones para el modelo') });
  const setSystem = (v: string) => {
    settings.system = v;
    saveSettings();
    scheduleContext();
  };
  sys.addEventListener('input', () => setSystem(sys.value));
  const presets = h(
    'div',
    { class: 'presets' },
    ...PRESETS.map(([name, text]) =>
      h('button', { type: 'button', class: 'preset', title: t(text), onclick: () => ((sys.value = t(text)), setSystem(sys.value), sys.focus()) }, t(name)),
    ),
    h('button', { type: 'button', class: 'preset ghost', onclick: () => ((sys.value = ''), setSystem(''), sys.focus()) }, t('Quitar')),
  );
  body.append(
    section(t('Instrucciones'),
      h('div', { class: 'field' }, sys, h('p', { class: 'hint' }, t('El modelo las tiene en cuenta en toda la conversación, con cualquier modelo. Empieza desde un ejemplo:')), presets),
    ),
  );

  // Writing and voice -------------------------------------------------------------------------
  const writing: (Node | null)[] = [toggle(t('Intro envía el mensaje'), settings.enterSends, (v) => (settings.enterSends = v), settings.enterSends ? t('Mayús+Intro añade una línea.') : t('Ctrl+Intro (Cmd+Intro en Mac) envía; Intro añade una línea.'))];
  if (canSpeak()) {
    const select = h('select', { 'aria-label': t('Voz') }) as HTMLSelectElement;
    const fill = () => {
      const voices = localVoices().sort((x, y) => Number(!x.lang.toLowerCase().startsWith(lang)) - Number(!y.lang.toLowerCase().startsWith(lang)) || x.name.localeCompare(y.name));
      select.replaceChildren(h('option', { value: '' }, t('Automática')), ...voices.map((v) => h('option', { value: v.voiceURI }, `${v.name} · ${v.lang}`)));
      select.value = voices.some((v) => v.voiceURI === settings.voice) ? settings.voice : '';
    };
    fill();
    speechSynthesis.addEventListener?.('voiceschanged', fill, { once: true });
    select.addEventListener('change', () => {
      settings.voice = select.value;
      saveSettings();
      applyPrefs();
    });
    writing.push(
      h(
        'div',
        { class: 'field' },
        h('label', {}, t('Voz para leer en voz alta')),
        h(
          'div',
          { class: 'set-inline' },
          select,
          h('button', { type: 'button', class: 'btn ghost small', title: t('Probar la voz'), onclick: () => speakSample(t('Hola. Así sonarán las respuestas leídas en voz alta.')) || toast(t('Este equipo no tiene voces instaladas para leer en voz alta en este idioma (se usan solo voces locales, por privacidad).')) }, icon('volume', 15), t('Probar')),
        ),
        h('p', { class: 'hint' }, t('Solo voces instaladas en este equipo: el texto no sale del navegador.')),
      ),
      slider(t('Velocidad de lectura'), 0.6, 1.8, 0.1, settings.rate, (v) => `${v.toLocaleString(locale, { maximumFractionDigits: 1 })}×`, (v) => ((settings.rate = v), applyPrefs()), t('1× es la velocidad normal.')),
    );
  }
  body.append(section(t('Escritura y voz'), ...writing));

  // Data --------------------------------------------------------------------------------------
  const usage = h('p', { class: 'hint set-lead' }, t('Calculando el espacio ocupado…'));
  void storageEstimate().then((e) => {
    usage.textContent = e
      ? t('{n} conversaciones · esta web ocupa {u} en el navegador (sobre todo los modelos descargados).', { n: fmtNum(conversations.length), u: fmtBytes(e.usage) })
      : t('{n} conversaciones guardadas en este navegador.', { n: fmtNum(conversations.length) });
  });
  const importInput = h('input', { type: 'file', accept: 'application/json,.json', hidden: true }) as HTMLInputElement;
  importInput.addEventListener('change', () => void importConversations(importInput));
  body.append(
    section(t('Tus datos'),
      usage,
      h(
        'div',
        { class: 'field' },
        h('label', {}, t('Copia de seguridad')),
        h(
          'div',
          { class: 'set-actions' },
          h('button', { type: 'button', class: 'btn ghost small', onclick: () => void exportConversations() }, icon('download', 15), t('Exportar')),
          h('button', { type: 'button', class: 'btn ghost small', onclick: () => importInput.click() }, icon('upload', 15), t('Importar')),
          importInput,
        ),
        h('p', { class: 'hint' }, t('Un archivo JSON con todas las conversaciones, imágenes incluidas. Se crea en tu equipo; no se sube a ningún sitio.')),
      ),
      h(
        'div',
        { class: 'field' },
        h('div', { class: 'field-row' }, h('label', {}, t('Borrar conversaciones')), h('button', { type: 'button', class: 'btn ghost small danger-text', onclick: () => void clearAllConversations() }, icon('trash', 15), t('Borrar'))),
        h('p', { class: 'hint' }, t('Los modelos descargados se conservan.')),
      ),
    ),
  );

  body.append(section(t('Depuración'), debugBlock()));
  body.scrollTop = scroll;
  renderContext();
}

async function exportConversations() {
  if (!conversations.length) return toast(t('No hay conversaciones que exportar.'));
  const blob = await exportBackup(conversations);
  const a = h('a', { href: URL.createObjectURL(blob), download: `webgpu-chat-${new Date().toISOString().slice(0, 10)}.json` });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  toast(t('{n} conversaciones exportadas.', { n: fmtNum(conversations.length) }));
}

async function importConversations(input: HTMLInputElement) {
  const file = input.files?.[0];
  input.value = '';
  if (!file) return;
  try {
    const list = await readBackup(file);
    for (const c of list) await saveConversation(c);
    conversations = await listConversations();
    renderList();
    renderSettings();
    toast(t('{n} conversaciones importadas.', { n: fmtNum(list.length) }));
  } catch {
    toast(t('Ese archivo no es una copia de WebGPU Chat.'));
  }
}

async function clearAllConversations() {
  if (!conversations.length) return toast(t('No hay conversaciones que borrar.'));
  if (generating) return toast(t('Espera a que termine la respuesta o párala.'));
  if (!(await confirmDialog(t('¿Borrar todas las conversaciones?'), t('Se eliminarán las {n} conversaciones de este navegador. Los modelos descargados se conservan. No se puede deshacer.', { n: fmtNum(conversations.length) }), t('Borrar todas')))) return;
  await clearConversations();
  conversations = [];
  newConversation();
  renderSettings();
  toast(t('Conversaciones borradas.'));
}

/** Settings block for the debug mode: a switch, the record and a copyable report. */
function debugBlock(): HTMLElement {
  let on = debugOn();
  const sw = h('button', { type: 'button', class: `switch${on ? ' on' : ''}`, role: 'switch', 'aria-checked': String(on), 'aria-label': t('Modo depuración') }, h('i'));
  sw.addEventListener('click', () => {
    on = !on;
    sw.classList.toggle('on', on);
    sw.setAttribute('aria-checked', String(on));
    setDebug(on);
    if (on) {
      dlog(`modo depuración activado · ${memoryText()}`);
      watchErrors();
      toast(t('Modo depuración activado. Carga el modelo y luego copia el informe.'));
    } else {
      toast(t('Modo depuración desactivado.'));
    }
    view.hidden = !on;
  });
  // Speed tests live in Modelos → Prueba de rendimiento (with progress and a score); every chat answer is also
  // recorded here with its timings.
  const view = h(
    'div',
    { class: 'debug-tools', hidden: !on },
    h('button', { type: 'button', class: 'btn ghost small', onclick: () => void showDebugReport() }, icon('code', 15), t('Ver informe')),
    h('button', { type: 'button', class: 'btn small ghost', onclick: () => void copyDebugReport() }, icon('copy', 15), t('Copiar informe')),
    h('button', { type: 'button', class: 'btn small ghost', onclick: () => (clearDebug(), toast(t('Registro borrado.'))) }, t('Borrar registro')),
    h('button', { type: 'button', class: 'btn small ghost', onclick: () => void profileGpu() }, t('Medir la GPU')),
    h('button', { type: 'button', class: 'btn small ghost', onclick: () => void matmulBench() }, t('Medir multiplicaciones')),
  );
  return h(
    'div',
    { class: 'field debug-block' },
    h('div', { class: 'field-row' }, h('label', {}, t('Modo depuración')), sw),
    h('p', { class: 'hint' }, t('Anota cada paso de la carga del modelo (descarga, caché, memoria, errores) en esta página. No se envía nada: copia el informe y pásalo a quien te ayude.')),
    view,
  );
}

/**
 * Loads the current model again with ONNX Runtime's profiler on. The next answer is slower (every GPU kernel is
 * timed) and its summary goes to the record: which operators run where and how long each kernel takes.
 */
async function profileGpu() {
  const id = engine.loaded ?? settings.model;
  if (!id || loading) return toast(t('Carga un modelo primero.'));
  dlog('medir la GPU: se vuelve a cargar el modelo con el perfilador de ONNX Runtime');
  engine.restart();
  await loadModel(id, true);
  if (engine.loaded === id) toast(t('Envía un mensaje: al terminar, la medición queda en el informe.'));
}

async function matmulBench() {
  toast(t('Midiendo… (unos segundos)'));
  dlog('medir multiplicaciones: pulsado');
  try {
    await engine.matmulBench();
    toast(t('Hecho: copia el informe.'));
  } catch (e) {
    dlog(`medir multiplicaciones: ${e instanceof Error ? e.message : String(e)}`);
    toast(t('No se pudo medir: mira el informe.'));
  }
}

async function copyDebugReport() {
  const text = await reportText();
  try {
    await navigator.clipboard.writeText(text);
    toast(t('Informe copiado.'));
  } catch {
    void showDebugReport();
    toast(t('No se pudo copiar: selecciona el texto del informe.'));
  }
}

async function showDebugReport() {
  const text = await reportText();
  const d = h('dialog', { class: 'debug-dialog' }) as HTMLDialogElement;
  const pre = h('pre', { class: 'debug-pre' }, text);
  const close = () => d.close();
  d.replaceChildren(
    h(
      'div',
      { class: 'dlg-body' },
      h('h2', {}, t('Informe de depuración')),
      h('p', { class: 'hint' }, t('Cópialo tal cual y pégalo en el mensaje. Así se ve qué ha pasado en cada paso.')),
      pre,
      h(
        'div',
        { class: 'dlg-actions' },
        h('button', { type: 'button', class: 'btn ghost', onclick: close }, t('Cerrar')),
        h('button', { type: 'button', class: 'btn primary', onclick: () => void copyDebugReport() }, t('Copiar informe')),
      ),
    ),
  );
  d.addEventListener('close', () => d.remove());
  document.body.append(d);
  d.showModal();
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
  await moveLegacyCache();
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
      const warning = gpu ? modelWarning(m, gpu) : null;
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
      if (!complete)
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
            warning ? h('div', { class: 'opt-warn' }, warningText(warning)) : null,
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
      const pendingBench = engine.bench();
      renderComposerState();
      const r = await pendingBench;
      const sc = score(r.tokensPerSecond);
      const lv = level(r.tokensPerSecond);
      const wt = wait(r.prefillMs);
      g.running(false);
      g.set(sc / 10);
      value.firstElementChild!.textContent = fmtNum(sc, 1);
      verdict.replaceChildren(h('strong', {}, t(LEVEL_TEXT[lv][0])), h('span', {}, t(LEVEL_TEXT[lv][1])));
      const lighter = AVAILABLE.filter((x) => x.gpuMemory < m.gpuMemory).sort((a, b) => b.gpuMemory - a.gpuMemory)[0];
      const weights = m.files.filter((f) => f.path.includes('.onnx')).reduce((a, f) => a + f.bytes, 0);
      const report = {
        model: m.id,
        revision: m.revision,
        gpu: gpu && { vendor: gpu.vendor, architecture: gpu.architecture, description: gpu.description, maxBufferSize: gpu.maxBufferSize, software: gpu.software },
        userAgent: navigator.userAgent,
        downloadBytes: downloadBytes(m),
        weightFileBytes: weights,
        gpuMemoryMeasured: false,
        benchmarkKind: 'synthetic-greedy-second-run',
        score: sc,
        ...r,
        date: new Date().toISOString(),
      };
      details.replaceChildren(
        h('p', { class: 'hint' }, t('La nota mide rapidez, no el porcentaje de potencia aprovechada. Prueba sintética sin muestreo; el chat puede rendir distinto.')),
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
            h('div', {}, h('dt', {}, t('Tamaño de los archivos de pesos')), h('dd', {}, fmtBytes(weights))),
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
    renderComposerState();
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

/** A quiet note when the device is offline: the chat keeps working with a downloaded model. */
function renderOnline() {
  const pill = $('offline');
  pill.hidden = navigator.onLine;
  pill.replaceChildren(icon('wifiOff', 14), document.createTextNode(t('Sin conexión · la IA sigue funcionando')));
}

async function boot() {
  // Calmer than on the landing page: the chat is for reading.
  mountAurora({ strength: 0.62 });
  renderOnline();
  addEventListener('online', renderOnline);
  addEventListener('offline', renderOnline);
  wireGlow();
  applyTheme();
  applyPrefs();
  hydrateIcons();
  $('btn-theme').addEventListener('click', () => {
    settings.theme = isDark() ? 'light' : 'dark';
    saveSettings();
    if (el.shell.classList.contains('panel-open')) renderSettings();
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
  applyUrlDebug();
  if (debugOn()) watchErrors();
  gpu = await probeGpu();
  if (DEV_Q4 && gpu.reason === 'no-f16') gpu = { ...gpu, ok: true, f16: true, reason: undefined };
  if (debugOn()) {
    dlog(`página abierta · ${memoryText()}`);
    dlog(
      gpu.ok
        ? `GPU: ${gpu.vendor} ${gpu.architecture} ${gpu.description} · f16 ${gpu.f16 ? 'sí' : 'no'} · maxBuffer ${fmtBytes(gpu.maxBufferSize)} · maxStorage ${fmtBytes(gpu.maxStorageBufferBindingSize)} · RAM ${gpu.deviceMemory || 'n/d'} GB · móvil ${gpu.mobile ? 'sí' : 'no'}`
        : `GPU: no disponible (${gpu.reason ?? 'desconocido'}) ${gpu.error ?? ''}`,
    );
  }
  await refreshCache();
  const rec = recommend(gpu);
  // Explicit and saved choices are preserved on every device.
  const asked = modelById(new URLSearchParams(location.search).get('model') ?? '');
  const saved = settings.model && modelById(settings.model);
  const initial = asked?.id ?? (saved ? saved.id : rec);
  if (initial) {
    selected = modelById(initial)!;
    settings.model = initial;
  }
  renderPicker();
  renderSettings();
  renderThread();
  renderComposerState();
  const crashed = takeCrashedLoad();
  if (crashed) {
    dlog(`la página se recargó mientras cargaba ${crashed.name}: la carga anterior no terminó`);
    showError(
      t('La última vez la página se cerró mientras cargaba {m}. Lo más probable es que al dispositivo le faltara memoria. Cierra otras pestañas y aplicaciones y vuelve a intentarlo, o elige un modelo más pequeño.', { m: crashed.name }),
    );
  } else if (selected) {
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
(window as unknown as Record<string, unknown>).__chat = { engine, lang, localPath, setMood, setAuroraProgress };

// Offline support (production build only): the service worker keeps the site available with no connection.
// Install as an app. Chrome and Edge (desktop and Android) offer it through `beforeinstallprompt`; Safari on
// iPhone and iPad has no such event, so the button explains how to add it from the share menu.
type InstallEvent = Event & { prompt(): Promise<void>; userChoice: Promise<{ outcome: string }> };
let installEvt: InstallEvent | null = null;
const installBtn = $<HTMLButtonElement>('btn-install');
const standalone = () =>
  ['standalone', 'fullscreen', 'minimal-ui', 'window-controls-overlay'].some((m) => matchMedia(`(display-mode: ${m})`).matches) ||
  (navigator as Navigator & { standalone?: boolean }).standalone === true;
const iosSafari = /iPhone|iPad|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
const renderInstall = () => {
  installBtn.hidden = standalone() || !(installEvt || iosSafari);
};
addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  installEvt = e as InstallEvent;
  renderInstall();
});
addEventListener('appinstalled', () => {
  installEvt = null;
  renderInstall();
  toast(t('App instalada. La tienes en tu pantalla de inicio.'));
});
installBtn.addEventListener('click', async () => {
  if (installEvt) {
    const evt = installEvt;
    installEvt = null;
    await evt.prompt();
    await evt.userChoice.catch(() => null);
    renderInstall();
    return;
  }
  const d = dialog('dlg-confirm');
  const ok = h('button', { type: 'button', class: 'btn primary', onclick: () => d.close() }, t('Entendido'));
  d.replaceChildren(
    h(
      'div',
      { class: 'dlg-body' },
      h('h2', {}, t('Instalar la app')),
      h('p', {}, t('En Safari, toca el botón Compartir y luego «Añadir a pantalla de inicio». Se abrirá como una app y funcionará sin conexión.')),
      h('div', { class: 'dlg-actions' }, ok),
    ),
  );
  d.showModal();
});
renderInstall();

if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  addEventListener('load', () => navigator.serviceWorker.register('/sw.js').catch(() => {}));
}
