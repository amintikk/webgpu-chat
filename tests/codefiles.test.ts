import assert from 'node:assert/strict';
import { test } from 'node:test';
import { codeFiles, safePath, zipFiles } from '../src/app/codefiles.ts';
import { parse } from '../src/app/markdown.ts';
import { listZip, makeZip, readEntry } from '../src/app/zip.ts';

const paths = (md: string) => [...codeFiles(parse(md)).values()].map((f) => `${f.path}${f.named ? '' : ' (?)'}`);

test('the file name is found before the block, in the fence or in a first-line comment', () => {
  assert.deepEqual(paths('**config/db.php**\n```php\n<?php\n```'), ['config/db.php']);
  assert.deepEqual(paths('### `views/index.php`\n```php\nx\n```'), ['views/index.php']);
  assert.deepEqual(paths('Archivo config/db.php:\n\n```php\nx\n```'), ['config/db.php']);
  assert.deepEqual(paths('```css style.css\nbody{}\n```'), ['style.css']);
  assert.deepEqual(paths('```js title="src/app.js"\nx\n```'), ['src/app.js']);
  assert.deepEqual(paths('```html\n<!-- index.html -->\n<p>\n```'), ['index.html']);
  assert.deepEqual(paths('```php\n<?php\n// models/Course.php\n```'), ['models/Course.php']);
  assert.deepEqual(paths('1. **a.js**\n\n   ```js\n   x\n   ```'), ['a.js']);
});

test('a sentence that mentions a file does not name the block; unnamed blocks get a name from the language', () => {
  assert.deepEqual(paths('Este código va dentro de tu fichero principal, que suele llamarse index.php en estos proyectos.\n```php\nx\n```'), ['archivo-1.php (?)']);
  assert.deepEqual(paths('Por ejemplo:\n```python\nx\n```\n```\ny\n```'), ['archivo-1.py (?)', 'archivo-2.txt (?)']);
  assert.deepEqual(paths('Mira, etc.\n```js\nx\n```'), ['archivo-1.js (?)']);
});

test('the zip leaves out shell commands and unfinished blocks; a repeated path keeps the last version', () => {
  const md = '**a.js**\n```js\nv1\n```\n```bash\nnpm i\n```\n**a.js**\n```js\nv2\n```\n**b/c.css**\n```css\nx{}\n```\n```js\nopen';
  assert.deepEqual(zipFiles(md).map((f) => [f.path, f.text]), [['a.js', 'v2\n'], ['b/c.css', 'x{}\n']]);
});

test('paths cannot leave the zip', () => {
  assert.equal(safePath('../../etc/passwd'), 'etc/passwd');
  assert.equal(safePath('/abs/x:y.js'), 'abs/x_y.js');
});

test('makeZip writes a zip that the reader opens back, deflated and stored', async () => {
  const enc = new TextEncoder();
  const big = 'línea de código\n'.repeat(200);
  const blob = await makeZip([{ path: 'p/año.php', data: enc.encode(big) }, { path: 'x.txt', data: enc.encode('hi') }]);
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const entries = listZip(bytes);
  assert.deepEqual(entries.map((e) => [e.path, e.method]), [['p/año.php', 8], ['x.txt', 0]]);
  assert.equal(new TextDecoder().decode(await readEntry(bytes, entries[0]!)), big);
  assert.equal(new TextDecoder().decode(await readEntry(bytes, entries[1]!)), 'hi');
});
