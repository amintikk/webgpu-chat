import assert from 'node:assert/strict';
import { test } from 'node:test';
import { inline, parse } from '../src/app/markdown.ts';

test('fenced code keeps its language and content; an unclosed fence stays open while streaming', () => {
  const b = parse('Texto\n\n```python\nprint("hi")\n```\n\nFin');
  assert.deepEqual(b.map((x) => x.t), ['p', 'code', 'p']);
  assert.deepEqual(b[1], { t: 'code', lang: 'python', v: 'print("hi")', open: false });
  const s = parse('```js\nconst a = 1;');
  assert.deepEqual(s[0], { t: 'code', lang: 'js', v: 'const a = 1;', open: true });
});

test('headings, lists (nested and ordered), quotes, rules and tables', () => {
  const b = parse('# Título\n\n- uno\n- dos\n  - dos.a\n\n3. tres\n4. cuatro\n\n> cita\n\n---\n\n| a | b |\n|:--|--:|\n| 1 | `x|y` |');
  assert.deepEqual(b.map((x) => x.t), ['h', 'list', 'list', 'quote', 'hr', 'table']);
  const ul = b[1] as Extract<ReturnType<typeof parse>[number], { t: 'list' }>;
  assert.equal(ul.items.length, 2);
  assert.equal(ul.items[1]!.at(-1)!.t, 'list');
  const ol = b[2] as Extract<ReturnType<typeof parse>[number], { t: 'list' }>;
  assert.equal(ol.ordered, true);
  assert.equal(ol.start, 3);
  const tb = b[5] as Extract<ReturnType<typeof parse>[number], { t: 'table' }>;
  assert.deepEqual(tb.align, ['left', 'right']);
  assert.deepEqual(tb.rows[0]![1], [{ t: 'code', v: 'x|y' }]);
});

test('inline emphasis, code, links; unsafe links and snake_case stay text', () => {
  assert.deepEqual(inline('**negrita** y *cursiva*'), [
    { t: 'strong', c: [{ t: 'text', v: 'negrita' }] },
    { t: 'text', v: ' y ' },
    { t: 'em', c: [{ t: 'text', v: 'cursiva' }] },
  ]);
  assert.deepEqual(inline('`a*b*`'), [{ t: 'code', v: 'a*b*' }]);
  assert.deepEqual(inline('[x](https://e.com)'), [{ t: 'link', href: 'https://e.com', c: [{ t: 'text', v: 'x' }] }]);
  assert.deepEqual(inline('[x](javascript:alert(1))'), [{ t: 'text', v: '[x](javascript:alert(1))' }]);
  assert.deepEqual(inline('mi_variable_larga'), [{ t: 'text', v: 'mi_variable_larga' }]);
  assert.deepEqual(inline('**sin cerrar'), [{ t: 'text', v: '**sin cerrar' }]);
});
