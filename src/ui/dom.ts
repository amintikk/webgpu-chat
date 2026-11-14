import { locale } from '../i18n.ts';

type Child = Node | string | number | null | undefined | false;

/** Tiny DOM builder. Text always goes through text nodes, never innerHTML. */
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, unknown> = {}, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v as EventListener);
    else if (k === 'class') el.className = String(v);
    else if (k in el && k !== 'list' && k !== 'style') (el as unknown as Record<string, unknown>)[k] = v;
    else el.setAttribute(k, String(v));
  }
  for (const c of children) if (c !== null && c !== undefined && c !== false) el.append(c instanceof Node ? c : String(c));
  return el;
}

export const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;

export function fmtNum(n: number, digits = 0): string {
  return Number.isFinite(n) ? n.toLocaleString(locale, { maximumFractionDigits: digits }) : '—';
}

export function fmtBytes(b: number): string {
  if (!Number.isFinite(b) || b <= 0) return '0 MB';
  return b >= 1e9 ? `${(b / 1e9).toLocaleString(locale, { maximumFractionDigits: 2 })} GB` : `${(b / 1e6).toLocaleString(locale, { maximumFractionDigits: b >= 1e8 ? 0 : 1 })} MB`;
}

export function fmtDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const s = ms / 1000;
  return s < 60 ? `${s.toLocaleString(locale, { maximumFractionDigits: 1 })} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

let toastTimer = 0;
export function toast(text: string): void {
  const el = $('toast');
  el.textContent = text;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => el.classList.remove('show'), 2600);
}
