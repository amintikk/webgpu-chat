// Splits a decoded model output into its reasoning part and its answer.
//
// Qwen3.5 and LFM3 open the reasoning block in the prompt itself (`<think>` at the end of the
// generation prompt), so the output starts inside it and only the closing tag appears. Gemma 4 writes
// its own opening marker (`<|channel>thought`). Both cases end at the closing marker; everything after
// it is the answer. Special tokens (end of turn, etc.) are removed from what the user sees.

export interface ThinkMarkers {
  open: string;
  close: string;
  /** The chat template already opens the reasoning block when reasoning is enabled. */
  promptOpens: boolean;
}

export interface Split {
  reasoning: string;
  content: string;
  /** Still inside the reasoning block (the answer has not started). */
  thinking: boolean;
}

export function splitReasoning(text: string, think: ThinkMarkers, reasoningOn: boolean, specials: string[] = []): Split {
  let t = text;
  let inside = reasoningOn && think.promptOpens;
  const lead = t.trimStart();
  if (lead.startsWith(think.open)) {
    t = lead.slice(think.open.length);
    inside = true;
  }
  let reasoning = '';
  let content = t;
  let thinking = false;
  if (inside) {
    const end = t.indexOf(think.close);
    if (end < 0) {
      reasoning = t;
      content = '';
      thinking = true;
    } else {
      reasoning = t.slice(0, end);
      content = t.slice(end + think.close.length);
    }
  }
  const clean = (s: string) => {
    for (const sp of specials) if (sp && s.includes(sp)) s = s.split(sp).join('');
    return s;
  };
  return { reasoning: clean(reasoning).trim(), content: clean(content).replace(/^\s+/, ''), thinking };
}
