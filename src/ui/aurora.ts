// A living aurora behind every page: three soft fields of light (blue, violet and a blend) animated by a
// tiny loop that only moves and fades them (transform and opacity, composited by the browser), so it costs
// almost nothing next to the model on the GPU.
//
// Each mood has its own behaviour; every parameter eases towards the new mood's values over about a second
// and the drift speed is integrated into a phase, so changing mood never makes the light jump.
//   idle      slow, calm drift
//   typing    the light gathers above the composer and pulses softly with each key
//   loading   a steady sweep while the model downloads or compiles
//   thinking  the fields draw together, violet leads, slow breathing
//   writing   faster flow, blue leads, a small pulse as text arrives
// With "reduce motion" the aurora stays still.

export type Mood = 'idle' | 'typing' | 'loading' | 'thinking' | 'writing';

interface Params {
  speed: number; // drift speed multiplier
  energy: number; // overall brightness multiplier
  blue: number; // weight of the blue field
  violet: number; // weight of the violet field
  gather: number; // 0 = spread over the page, 1 = drawn to the focus point
  fx: number; // focus point, fraction of the viewport
  fy: number;
  breathe: number; // amplitude of a slow brightness swell
  sweep: number; // amplitude of a horizontal sweep
}

const MOODS: Record<Mood, Params> = {
  idle: { speed: 1, energy: 1, blue: 1, violet: 1, gather: 0, fx: 0.5, fy: 0.4, breathe: 0.04, sweep: 0 },
  typing: { speed: 1.25, energy: 1.12, blue: 1.1, violet: 0.95, gather: 0.42, fx: 0.55, fy: 0.92, breathe: 0.05, sweep: 0 },
  loading: { speed: 1.6, energy: 1.05, blue: 1.05, violet: 1, gather: 0.12, fx: 0.5, fy: 0.5, breathe: 0.06, sweep: 1 },
  thinking: { speed: 0.55, energy: 1.25, blue: 0.85, violet: 1.4, gather: 0.55, fx: 0.55, fy: 0.55, breathe: 0.22, sweep: 0 },
  writing: { speed: 2.3, energy: 1.3, blue: 1.4, violet: 1, gather: 0.25, fx: 0.55, fy: 0.75, breathe: 0.08, sweep: 0 },
};

let root: HTMLElement | null = null;
let target: Params = { ...MOODS.idle };
let mood: Mood = 'idle';
let pulse = 0; // short-lived extra brightness from keystrokes or arriving text

export function mountAurora(): void {
  if (root) return;
  root = document.createElement('div');
  root.className = 'aurora';
  root.setAttribute('aria-hidden', 'true');
  const bands = ['a', 'b', 'c'].map((k) => {
    const band = document.createElement('i');
    band.className = `band band-${k}`;
    root!.append(band);
    return band;
  });
  const grain = document.createElement('i');
  grain.className = 'grain';
  root.append(grain);
  document.body.prepend(root);
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  root.classList.add('live');

  // Pointer parallax, eased so the light follows like something alive rather than snapping.
  let px = 0.5;
  let py = 0.35;
  addEventListener(
    'pointermove',
    (e: PointerEvent) => {
      px = e.clientX / innerWidth;
      py = e.clientY / innerHeight;
    },
    { passive: true },
  );

  const cur: Params = { ...MOODS.idle };
  let mx = 0.5;
  let my = 0.35;
  let phase = Math.random() * 100;
  let last = performance.now();
  // Each field: resting place (fraction of viewport), drift radius and frequencies, parallax depth.
  const fields = [
    { x: 0.18, y: 0.05, rx: 0.16, ry: 0.09, fx: 0.11, fy: 0.07, depth: 0.07, hue: 'blue' as const },
    { x: 0.82, y: 0.12, rx: 0.14, ry: 0.11, fx: 0.08, fy: 0.1, depth: -0.06, hue: 'violet' as const },
    { x: 0.55, y: 0.95, rx: 0.18, ry: 0.07, fx: 0.13, fy: 0.06, depth: 0.09, hue: 'mix' as const },
  ];

  const frame = (now: number) => {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (!document.hidden) {
      // Ease every parameter towards the target mood (about one second), and the pointer a bit faster.
      const k = 1 - Math.exp(-dt / 0.9);
      for (const key of Object.keys(cur) as (keyof Params)[]) cur[key] += (target[key] - cur[key]) * k;
      const kp = 1 - Math.exp(-dt / 0.35);
      mx += (px - mx) * kp;
      my += (py - my) * kp;
      pulse *= Math.exp(-dt / 0.35);
      phase += dt * cur.speed;

      const vw = innerWidth;
      const vh = innerHeight;
      const swell = 1 + cur.breathe * Math.sin(phase * 1.3) + pulse;
      fields.forEach((f, i) => {
        const t = phase + i * 2.1;
        let x = f.x + f.rx * Math.sin(t * f.fx * 6.283) + (mx - 0.5) * f.depth * 2;
        let y = f.y + f.ry * Math.cos(t * f.fy * 6.283) + (my - 0.5) * f.depth * 1.5;
        x += cur.sweep * 0.22 * Math.sin(phase * 0.9 + i * 1.7);
        // Drawn towards the focus point (e.g. the composer) by `gather`.
        x += (cur.fx - x) * cur.gather * (i === 2 ? 0.9 : 0.55);
        y += (cur.fy - y) * cur.gather * (i === 2 ? 0.9 : 0.55);
        const scale = 1 + 0.12 * Math.sin(t * 0.5) + cur.gather * 0.1;
        const weight = f.hue === 'blue' ? cur.blue : f.hue === 'violet' ? cur.violet : (cur.blue + cur.violet) / 2;
        const el = bands[i]!;
        el.style.transform = `translate3d(${(x * vw).toFixed(1)}px, ${(y * vh).toFixed(1)}px, 0) translate(-50%, -50%) scale(${scale.toFixed(3)})`;
        el.style.opacity = Math.min(1, 0.55 * cur.energy * weight * swell * (i === 2 ? 0.7 : 1)).toFixed(3);
      });
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

export function setMood(next: Mood): void {
  if (next === mood) return;
  mood = next;
  target = { ...MOODS[next] };
  document.documentElement.dataset.mood = next;
}

export function getMood(): Mood {
  return mood;
}

/** A brief brightening: a keystroke, or a piece of the answer arriving. */
export function pulseAurora(amount = 0.08): void {
  pulse = Math.min(0.3, pulse + amount);
}

/** Cards and buttons with [data-glow] get a light that follows the pointer across them. */
export function wireGlow(scope: ParentNode = document): void {
  scope.addEventListener(
    'pointermove',
    (ev) => {
      const e = ev as PointerEvent;
      const el = (e.target as HTMLElement).closest?.<HTMLElement>('[data-glow]');
      if (!el) return;
      const r = el.getBoundingClientRect();
      el.style.setProperty('--gx', `${e.clientX - r.left}px`);
      el.style.setProperty('--gy', `${e.clientY - r.top}px`);
    },
    { passive: true },
  );
}

/** Elements with [data-reveal] fade and rise into place the first time they enter the viewport. */
export function wireReveal(): void {
  const els = document.querySelectorAll<HTMLElement>('[data-reveal]');
  if (!('IntersectionObserver' in window) || matchMedia('(prefers-reduced-motion: reduce)').matches) {
    els.forEach((el) => el.classList.add('revealed'));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      for (const en of entries) {
        if (!en.isIntersecting) continue;
        en.target.classList.add('revealed');
        io.unobserve(en.target);
      }
    },
    { rootMargin: '0px 0px -8% 0px', threshold: 0.08 },
  );
  els.forEach((el, i) => {
    el.style.setProperty('--d', `${(i % 4) * 70}ms`);
    io.observe(el);
  });
}
