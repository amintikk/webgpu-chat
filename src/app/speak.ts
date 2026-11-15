// "Read aloud" for answers, with the voices installed on this device only (SpeechSynthesis voices whose
// localService is true): the text never leaves the browser. Hidden where the browser has no speech synthesis.
import { h } from '../ui/dom.ts';
import { icon } from '../ui/icons.ts';
import { lang, t } from '../i18n.ts';

const synth: SpeechSynthesis | null = 'speechSynthesis' in globalThis ? speechSynthesis : null;
let speaking: HTMLButtonElement | null = null;
/** Chosen in Settings: a voice (voiceURI; '' = the default for the page's language) and its speed. */
export const voicePrefs = { voice: '', rate: 1 };

/** Voices installed on this device (never the browser's online voices). */
export function localVoices(): SpeechSynthesisVoice[] {
  return synth ? synth.getVoices().filter((v) => v.localService) : [];
}
export const canSpeak = (): boolean => !!synth;

/** Says a short sample with the chosen voice (Settings → Probar). */
export function speakSample(text: string): boolean {
  const voice = localVoice();
  if (!synth || !voice) return false;
  stop();
  const u = new SpeechSynthesisUtterance(text);
  u.voice = voice;
  u.lang = voice.lang;
  u.rate = voicePrefs.rate;
  synth.speak(u);
  return true;
}

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
  const voices = localVoices();
  const chosen = voicePrefs.voice && voices.find((v) => v.voiceURI === voicePrefs.voice);
  if (chosen) return chosen;
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
    u.rate = voicePrefs.rate;
    u.onend = u.onerror = () => speaking === btn && stop();
    speaking = btn;
    btn.replaceChildren(icon('stop', 16));
    synth.speak(u);
  });
  return btn;
}

// Voices load asynchronously in Chromium: ask once so they are ready by the first click.
synth?.getVoices();
