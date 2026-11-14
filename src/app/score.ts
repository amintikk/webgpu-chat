// Turns benchmark numbers into something anyone can read: a 0-10 score and a verdict in words, measured
// against reading speed (people read roughly 4-5 words, about 6-7 tokens, per second).

export type Level = 'excellent' | 'great' | 'good' | 'fair' | 'slow';

/** 0-10 on a logarithmic scale: 2 tokens/s is 0, 60 tokens/s or more is 10. */
export function score(tokensPerSecond: number): number {
  if (!(tokensPerSecond > 0)) return 0;
  const s = (10 * Math.log(tokensPerSecond / 2)) / Math.log(60 / 2);
  return Math.round(Math.min(10, Math.max(0, s)) * 10) / 10;
}

export function level(tokensPerSecond: number): Level {
  if (tokensPerSecond >= 35) return 'excellent';
  if (tokensPerSecond >= 18) return 'great';
  if (tokensPerSecond >= 9) return 'good';
  if (tokensPerSecond >= 4) return 'fair';
  return 'slow';
}

export type Wait = 'instant' | 'quick' | 'noticeable' | 'long';

/** How long the first word of a short message takes. */
export function wait(firstWordMs: number): Wait {
  if (firstWordMs < 800) return 'instant';
  if (firstWordMs < 2000) return 'quick';
  if (firstWordMs < 5000) return 'noticeable';
  return 'long';
}
