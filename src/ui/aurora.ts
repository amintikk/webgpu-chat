// A living aurora behind every page. Four soft colour fields drift on their own (pure CSS transforms,
// composited by the browser so they cost almost nothing next to the model on the GPU), lean towards the
// pointer, and change mood with the app: calm while idle, brighter while the model reads, flowing while it
// writes. With "reduce motion" the aurora stays still.

export type Mood = 'idle' | 'loading' | 'thinking' | 'writing';

let root: HTMLElement | null = null;

export function mountAurora(): void {
  if (root) return;
  root = document.createElement('div');
  root.className = 'aurora';
  root.setAttribute('aria-hidden', 'true');
  for (const k of ['a', 'b', 'c', 'd']) {
    const band = document.createElement('i');
    band.className = `band band-${k}`;
    root.append(band);
  }
  const grain = document.createElement('i');
  grain.className = 'grain';
  root.append(grain);
  document.body.prepend(root);

  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  // Pointer parallax, eased so the light follows like something alive rather than snapping.
  let tx = 0.5;
  let ty = 0.3;
  let x = tx;
  let y = ty;
  let raf = 0;
  const tick = () => {
    x += (tx - x) * 0.06;
    y += (ty - y) * 0.06;
    root!.style.setProperty('--mx', x.toFixed(4));
    root!.style.setProperty('--my', y.toFixed(4));
    raf = Math.abs(tx - x) + Math.abs(ty - y) > 0.001 ? requestAnimationFrame(tick) : 0;
  };
  window.addEventListener(
    'pointermove',
    (e: PointerEvent) => {
      tx = e.clientX / innerWidth;
      ty = e.clientY / innerHeight;
      if (!raf) raf = requestAnimationFrame(tick);
    },
    { passive: true },
  );
  // Hidden tabs do not animate at all.
  document.addEventListener('visibilitychange', () => root!.classList.toggle('paused', document.hidden));
}

export function setMood(mood: Mood): void {
  if (root) root.dataset.mood = mood;
  document.documentElement.dataset.mood = mood;
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
