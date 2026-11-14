import '@fontsource-variable/inter';
import '../tokens.css';
import './site.css';
import { mountAurora, wireGlow, wireReveal } from '../ui/aurora.ts';
import { hydrateIcons, icon } from '../ui/icons.ts';

const KEY = 'webgpu-chat-settings';

function readTheme(): string {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? '{}').theme ?? 'auto';
  } catch {
    return 'auto';
  }
}
const isDark = (theme: string) => theme === 'dark' || (theme !== 'light' && matchMedia('(prefers-color-scheme: dark)').matches);

function applyTheme(theme: string): void {
  if (theme === 'dark' || theme === 'light') document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  document.getElementById('btn-theme')?.replaceChildren(icon(isDark(theme) ? 'sun' : 'moon', 18));
}

/** Cross-fades a visual change where the browser supports view transitions. */
function smoothly(fn: () => void): void {
  const d = document as Document & { startViewTransition?: (cb: () => void) => unknown };
  if (d.startViewTransition && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
    document.documentElement.classList.add('vt-theme');
    const vt = d.startViewTransition(fn) as { finished?: Promise<void> };
    void vt.finished?.finally(() => document.documentElement.classList.remove('vt-theme'));
  } else fn();
}

let theme = readTheme();
applyTheme(theme);
document.getElementById('btn-theme')?.addEventListener('click', () => {
  theme = isDark(theme) ? 'light' : 'dark';
  try {
    // Shared with the app, so the choice carries over.
    const s = JSON.parse(localStorage.getItem(KEY) ?? '{}');
    localStorage.setItem(KEY, JSON.stringify({ ...s, theme }));
  } catch {
    /* storage blocked: still switch for this page */
  }
  smoothly(() => applyTheme(theme));
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => applyTheme(theme));

const toggle = document.getElementById('nav-toggle');
const nav = document.getElementById('site-nav');
if (toggle && nav) {
  const set = (open: boolean) => {
    document.body.classList.toggle('nav-open', open);
    toggle.setAttribute('aria-expanded', String(open));
  };
  toggle.addEventListener('click', () => set(!document.body.classList.contains('nav-open')));
  nav.addEventListener('click', (e) => (e.target as HTMLElement).closest('a') && set(false));
  // Tapping outside the menu (on the dimmed page) closes it.
  document.addEventListener('click', (e) => {
    if (document.body.classList.contains('nav-open') && !(e.target as HTMLElement).closest('#site-nav, #nav-toggle')) set(false);
  });
}

// Tell the visitor up front whether their browser can run the models.
const gpuNote = document.getElementById('gpu-check');
if (gpuNote) {
  const lang = document.documentElement.lang;
  const say = (ok: boolean, es: string, en: string) => {
    gpuNote.classList.add(ok ? 'ok' : 'bad');
    gpuNote.replaceChildren(icon(ok ? 'check' : 'alert', 15), document.createTextNode(lang === 'en' ? en : es));
  };
  (async () => {
    if (!('gpu' in navigator) || !navigator.gpu) return say(false, 'Tu navegador no tiene WebGPU: usa Chrome o Edge recientes.', 'Your browser has no WebGPU: use a recent Chrome or Edge.');
    const a = await navigator.gpu.requestAdapter().catch(() => null);
    if (!a) return say(false, 'Tu navegador tiene WebGPU, pero no encuentra una GPU utilizable.', 'Your browser has WebGPU but found no usable GPU.');
    if (!a.features.has('shader-f16')) return say(false, 'Tu GPU no admite shader-f16, que necesitan los modelos.', 'Your GPU lacks shader-f16, which the models need.');
    say(true, 'Tu navegador puede ejecutar los modelos.', 'Your browser can run the models.');
  })();
}

// Devices on the landing lean gently towards the pointer.
const stage = document.querySelector<HTMLElement>('.stage');
if (stage && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const devices = stage.querySelectorAll<HTMLElement>('[data-tilt]');
  stage.addEventListener('pointermove', (e) => {
    const r = stage.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width - 0.5;
    const y = (e.clientY - r.top) / r.height - 0.5;
    devices.forEach((d, i) => {
      const k = i ? 1.6 : 1;
      d.style.setProperty('--ry', `${(x * 6 * k).toFixed(2)}deg`);
      d.style.setProperty('--rx', `${(-y * 5 * k).toFixed(2)}deg`);
    });
  });
  stage.addEventListener('pointerleave', () => devices.forEach((d) => (d.style.setProperty('--ry', '0deg'), d.style.setProperty('--rx', '0deg'))));
}

// Examples gallery: arrow buttons scroll one card at a time.
const track = document.getElementById('ex-track');
document.querySelectorAll<HTMLButtonElement>('[data-ex]').forEach((b) =>
  b.addEventListener('click', () => {
    const card = track?.querySelector<HTMLElement>('.ex');
    if (track && card) track.scrollBy({ left: Number(b.dataset.ex) * (card.offsetWidth + 18), behavior: 'smooth' });
  }),
);

hydrateIcons();
mountAurora();
wireGlow();
wireReveal();
// The inline safety net in theme.js shows everything if this script never runs.
(window as unknown as { siteReady?: boolean }).siteReady = true;
