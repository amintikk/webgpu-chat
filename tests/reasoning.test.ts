import assert from 'node:assert/strict';
import { test } from 'node:test';
import { splitReasoning } from '../src/engine/reasoning.ts';

const qwen = { open: '<think>', close: '</think>', promptOpens: true };
const gemma = { open: '<|channel>thought', close: '<channel|>', promptOpens: false };

test('reasoning opened by the prompt (Qwen3.5, LFM3)', () => {
  assert.deepEqual(splitReasoning('Pienso…', qwen, true), { reasoning: 'Pienso…', content: '', thinking: true });
  assert.deepEqual(splitReasoning('Pienso.\n</think>\n\nRespuesta<|im_end|>', qwen, true, ['<|im_end|>']), { reasoning: 'Pienso.', content: 'Respuesta', thinking: false });
});

test('reasoning off: the answer is everything (empty think blocks are dropped)', () => {
  assert.deepEqual(splitReasoning('Hola', qwen, false), { reasoning: '', content: 'Hola', thinking: false });
  assert.deepEqual(splitReasoning('<think></think>Hola', qwen, false), { reasoning: '', content: 'Hola', thinking: false });
});

test('reasoning opened by the model (Gemma 4)', () => {
  assert.deepEqual(splitReasoning('<|channel>thought\nA ver', gemma, true), { reasoning: 'A ver', content: '', thinking: true });
  assert.deepEqual(splitReasoning('<|channel>thought\nA ver<channel|>Sí.<turn|>', gemma, true, ['<turn|>']), { reasoning: 'A ver', content: 'Sí.', thinking: false });
});

test('a stray closing tag with reasoning off splits off the first copy', () => {
  assert.deepEqual(splitReasoning('Respuesta.</think>Respuesta.', qwen, false), { reasoning: 'Respuesta.', content: 'Respuesta.', thinking: false });
});
