// "Read aloud" for answers, with the voices installed on this device only (SpeechSynthesis voices whose
// localService is true): the text never leaves the browser. Hidden where the browser has no speech synthesis.
import { h } from '../ui/dom.ts';
import { icon } from '../ui/icons.ts';
import { lang, t } from '../i18n.ts';

const synth: SpeechSynthesis | null = 'speechSynthesis' in globalThis ? speechSynthesis : null;
let speaking: HTMLButtonElement | null = null;

/** Markdown to plain sentences: code blocks are skipped, symbols dropped. */
function plain(md: string): string {
  return md
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/^\s{0,3}(#{1,6}|[-*+]|\d+[.)]|>)\s+/gm, '')
    .replace(/[*_~|]+/g, '')
    .replace(/\n{2,}/g, '.\n')
    .trim();
}

function localVoice(): SpeechSynthesisVoice | null {
  const voices = synth!.getVoices().filter((v) => v.localService);
  return voices.find((v) => v.lang.toLowerCase().startsWith(lang) && v.default) ?? voices.find((v) => v.lang.toLowerCase().startsWith(lang)) ?? null;
}

function stop() {
  synth?.cancel();
  if (speaking) speaking.replaceChildren(icon('volume', 16));
  speaking = null;
}

export function speakButton(get: () => string, onMissing: (msg: string) => void): HTMLButtonElement | null {
  if (!synth) return null;
  const label = t('Leer en voz alta');
  const btn = h('button', { type: 'button', class: 'icon-btn', title: label, 'aria-label': label }, icon('volume', 16)) as HTMLButtonElement;
  btn.addEventListener('click', () => {
    if (speaking === btn) return stop();
    stop();
    const voice = localVoice();
    if (!voice) return onMissing(t('Este equipo no tiene voces instaladas para leer en voz alta en este idioma (se usan solo voces locales, por privacidad).'));
    const u = new SpeechSynthesisUtterance(plain(get()));
    u.voice = voice;
    u.lang = voice.lang;
    u.onend = u.onerror = () => speaking === btn && stop();
    speaking = btn;
    btn.replaceChildren(icon('stop', 16));
    synth.speak(u);
  });
  return btn;
}

// Voices load asynchronously in Chromium: ask once so they are ready by the first click.
synth?.getVoices();
