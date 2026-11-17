import assert from 'node:assert/strict';
import { test } from 'node:test';
import { deflateRawSync } from 'node:zlib';
import { archiveText, listZip, readEntry, skippedDir, treeText } from '../src/app/zip.ts';

/** A minimal zip writer (stored or deflate entries), enough to exercise the reader. */
function makeZip(files: { name: string; data: string | Uint8Array; deflate?: boolean }[]): Uint8Array {
  const enc = new TextEncoder();
  const locals: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;
  for (const f of files) {
    const raw = typeof f.data === 'string' ? enc.encode(f.data) : f.data;
    const body = f.deflate ? new Uint8Array(deflateRawSync(raw)) : raw;
    const name = enc.encode(f.name);
    const lh = new DataView(new ArrayBuffer(30));
    lh.setUint32(0, 0x04034b50, true);
    lh.setUint16(6, 0x800, true);
    lh.setUint16(8, f.deflate ? 8 : 0, true);
    lh.setUint32(18, body.length, true);
    lh.setUint32(22, raw.length, true);
    lh.setUint16(26, name.length, true);
    const local = new Uint8Array([...new Uint8Array(lh.buffer), ...name, ...body]);
    const ch = new DataView(new ArrayBuffer(46));
    ch.setUint32(0, 0x02014b50, true);
    ch.setUint16(8, 0x800, true);
    ch.setUint16(10, f.deflate ? 8 : 0, true);
    ch.setUint32(20, body.length, true);
    ch.setUint32(24, raw.length, true);
    ch.setUint16(28, name.length, true);
    ch.setUint32(42, offset, true);
    central.push(new Uint8Array([...new Uint8Array(ch.buffer), ...name]));
    locals.push(local);
    offset += local.length;
  }
  const cd = central.flatMap((c) => [...c]);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, files.length, true);
  end.setUint16(10, files.length, true);
  end.setUint32(12, cd.length, true);
  end.setUint32(16, offset, true);
  return new Uint8Array([...locals.flatMap((l) => [...l]), ...cd, ...new Uint8Array(end.buffer)]);
}

test('lists and reads stored and deflated entries', async () => {
  const php = '<?php echo "hola"; ?>\n'.repeat(50);
  const zip = makeZip([
    { name: 'proyecto/', data: '' },
    { name: 'proyecto/index.php', data: php, deflate: true },
    { name: 'proyecto/año.txt', data: 'ñ' },
  ]);
  const entries = listZip(zip);
  assert.deepEqual(entries.map((e) => e.path), ['proyecto/', 'proyecto/index.php', 'proyecto/año.txt']);
  assert.equal(new TextDecoder().decode(await readEntry(zip, entries[1]!)), php);
  assert.equal(new TextDecoder().decode(await readEntry(zip, entries[2]!)), 'ñ');
  await assert.rejects(readEntry(zip, entries[1]!, 100), /too large/);
});

test('rejects what is not a zip', () => {
  assert.throws(() => listZip(new TextEncoder().encode('not a zip at all, just some text here')), /not a zip/);
});

test('tooling folders are collapsed', () => {
  assert.deepEqual(skippedDir('p/.git/objects/00/ab'), { dir: 'p/.git/', why: 'version control' });
  assert.deepEqual(skippedDir('node_modules/x/index.js'), { dir: 'node_modules/', why: 'dependencies' });
  assert.equal(skippedDir('p/src/git.ts'), null);
});

test('the archive text has a tree, then README first, then the files', () => {
  const items = [
    { path: 'p/.git/', note: 'version control, not read' },
    { path: 'p/b.css', text: 'b{}' },
    { path: 'p/img/logo.png', note: 'image, attached' },
    { path: 'p/README.md', text: '# P' },
  ];
  assert.equal(treeText(items), 'p/\n  .git/  [version control, not read]\n  b.css\n  img/\n    logo.png  [image, attached]\n  README.md');
  const text = archiveText('p.zip', items);
  assert.ok(text.startsWith('<archive name="p.zip" files="3">\nContents:\np/'));
  assert.ok(text.indexOf('path="p/README.md"') < text.indexOf('path="p/b.css"'));
  assert.ok(text.endsWith('</archive>'));
});
