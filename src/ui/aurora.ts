// A living aurora behind every page: three soft fields of light (blue, violet and a blend) moved by a small
// loop that only changes transform and opacity, so the browser composites it almost for free next to the
// model on the GPU.
//
// In the chat each state has its own choreography, not just a different speed:
//   idle      the light rests at the top corners and drifts very slowly
//   typing    it gathers into a low glow under the composer; each key sends a small ripple
//   loading   two ribbons sweep across the top while a band at the bottom grows with the download
//   thinking  blue and violet orbit each other around the centre and breathe slowly
//   writing   the light flows upwards from the composer, a little brighter as text arrives
// Every state's pose is computed every frame and the states are cross-blended with eased weights, so
// moving from one to another is a smooth morph, never a jump. With "reduce motion" the aurora stays still.

export type Mood = 'idle' | 'typing' | 'loading' | 'thinking' | 'writing';

const MOODS: Mood[] = ['idle', 'typing', 'loading', 'thinking', 'writing'];

interface Pose {
  x: number; // centre, fraction of the viewport
  y: number;
  sx: number; // scale
  sy: number;
  a: number; // opacity, 0-1
}

let root: HTMLElement | null = null;
let mood: Mood = 'idle';
let pulse = 0; // short-lived extra brightness from keystrokes or arriving text
let progress = 0; // download progress, 0-1 (loading)

const TAU = Math.PI * 2;
const wave = (t: number, hz: number, off = 0) => Math.sin(t * hz * TAU + off);
/** 0 at the ends of a 0-1 cycle, 1 in the middle: hides the jump when a flowing field wraps around. */
const hump = (u: number) => Math.sin(Math.PI * Math.min(1, Math.max(0, u)));

// Pose of field i (0 blue, 1 violet, 2 blend) in each state at time t (seconds of that state's own clock).
const POSES: Record<Mood, (i: number, t: number) => Pose> = {
  idle: (i, t) => {
    const base = [
      [0.16, 0.02],
      [0.86, 0.06],
      [0.5, 1.04],
    ][i]!;
    return {
      x: base[0]! + 0.05 * wave(t, 0.021, i * 2),
      y: base[1]! + 0.035 * wave(t, 0.017, i * 3 + 1),
      sx: 1 + 0.06 * wave(t, 0.013, i),
      sy: 1 + 0.06 * wave(t, 0.011, i + 2),
      a: [0.55, 0.5, 0.28][i]!,
    };
  },
  typing: (i, t) => {
    // A pool of light under the composer, swaying gently from side to side.
    const sway = 0.05 * wave(t, 0.09, i * 1.7);
    const base = [
      [0.34, 0.97],
      [0.68, 0.97],
      [0.5, 1.03],
    ][i]!;
    return { x: base[0]! + sway, y: base[1]! + 0.012 * wave(t, 0.2, i), sx: i === 2 ? 1.5 : 1.05, sy: i === 2 ? 0.55 : 0.7, a: [0.5, 0.45, 0.55][i]! };
  },
  loading: (i, t) => {
    if (i === 2) {
      // Grows from the left with the download.
      const p = Math.max(0.04, progress);
      return { x: p * 0.5, y: 1.0, sx: 0.35 + 1.5 * p, sy: 0.5, a: 0.55 };
    }
    // Two ribbons crossing the top, one after the other, faded in and out at the edges.
    const u = (t * 0.11 + i * 0.5) % 1;
    return { x: -0.25 + u * 1.5, y: 0.04 + 0.05 * i, sx: 1.25, sy: 0.42, a: 0.6 * hump(u) };
  },
  thinking: (i, t) => {
    if (i === 2) return { x: 0.5, y: 0.42, sx: 0.8 + 0.12 * wave(t, 0.18), sy: 0.8 + 0.12 * wave(t, 0.18), a: 0.22 + 0.1 * wave(t, 0.18) };
    // Blue and violet orbit each other around the centre, breathing in and out.
    const ang = t * 0.42 + i * Math.PI;
    const breathe = 1 + 0.14 * wave(t, 0.18, i * Math.PI);
    return { x: 0.5 + 0.2 * Math.cos(ang), y: 0.36 + 0.09 * Math.sin(ang), sx: 0.78 * breathe, sy: 0.78 * breathe, a: i === 1 ? 0.6 : 0.42 };
  },
  writing: (i, t) => {
    // Rising from the composer to the top, one field after another.
    const u = (t * 0.075 + i / 3) % 1;
    return { x: 0.5 + 0.28 * wave(t, 0.05, i * 2.1) * (0.4 + u), y: 1.12 - u * 1.25, sx: 0.95, sy: 1.15, a: (i === 0 ? 0.58 : 0.46) * hump(u) };
  },
};

export interface AuroraOptions {
  /** Overall brightness multiplier (the chat uses a calmer aurora than the landing pages). */
  strength?: number;
}

export function mountAurora(opts: AuroraOptions = {}): void {
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
  const strength = opts.strength ?? 1;

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

  const weight: Record<Mood, number> = { idle: 1, typing: 0, loading: 0, thinking: 0, writing: 0 };
  const clock: Record<Mood, number> = { idle: Math.random() * 50, typing: 0, loading: 0, thinking: 0, writing: 0 };
  let mx = 0.5;
  let my = 0.35;
  let last = performance.now();
  const depth = [0.06, -0.05, 0.04];

  const frame = (now: number) => {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (!document.hidden) {
      // Cross-blend the states (about 1.1 s), and advance each state's own clock while it is visible.
      const k = 1 - Math.exp(-dt / 1.1);
      for (const m of MOODS) {
        weight[m] += ((m === mood ? 1 : 0) - weight[m]) * k;
        if (weight[m] > 0.001) clock[m] += dt;
      }
      const kp = 1 - Math.exp(-dt / 0.4);
      mx += (px - mx) * kp;
      my += (py - my) * kp;
      pulse *= Math.exp(-dt / 0.4);

      const vw = innerWidth;
      const vh = innerHeight;
      bands.forEach((el, i) => {
        let x = 0;
        let y = 0;
        let sx = 0;
        let sy = 0;
        let a = 0;
        let total = 0;
        for (const m of MOODS) {
          const w = weight[m];
          if (w < 0.001) continue;
          const p = POSES[m](i, clock[m]);
          x += p.x * w;
          y += p.y * w;
          sx += p.sx * w;
          sy += p.sy * w;
          a += p.a * w;
          total += w;
        }
        x = x / total + (mx - 0.5) * depth[i]!;
        y = y / total + (my - 0.5) * depth[i]! * 0.8;
        const glow = 1 + pulse * (i === 2 || mood === 'typing' ? 1.4 : 0.8);
        el.style.transform = `translate3d(${(x * vw).toFixed(1)}px, ${(y * vh).toFixed(1)}px, 0) translate(-50%, -50%) scale(${(sx / total).toFixed(3)}, ${(sy / total).toFixed(3)})`;
        el.style.opacity = Math.min(1, (a / total) * glow * strength).toFixed(3);
      });
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

export function setMood(next: Mood): void {
  if (next === mood) return;
  mood = next;
  document.documentElement.dataset.mood = next;
}

export function getMood(): Mood {
  return mood;
}

/** Download progress (0-1), shown by the loading state. */
export function setAuroraProgress(p: number): void {
  progress = Number.isFinite(p) ? Math.min(1, Math.max(0, p)) : 0;
}

/** A brief brightening: a keystroke, or a piece of the answer arriving. */
export function pulseAurora(amount = 0.08): void {
  pulse = Math.min(0.25, pulse + amount);
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
