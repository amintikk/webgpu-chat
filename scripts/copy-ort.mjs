// Copies the ONNX Runtime Web WebAssembly runtime used by Transformers.js to public/ort/, so the app
// serves it itself instead of loading it from a CDN (keeps the CSP to 'self' + Hugging Face).
import { copyFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

// The copy of onnxruntime-web that Transformers.js itself depends on (nested if versions differ).
const nested = 'node_modules/@huggingface/transformers/node_modules/onnxruntime-web';
const pkg = existsSync(nested) ? nested : 'node_modules/onnxruntime-web';
const version = JSON.parse(readFileSync(join(pkg, 'package.json'), 'utf8')).version;
const out = 'public/ort';
mkdirSync(out, { recursive: true });
for (const f of ['ort-wasm-simd-threaded.asyncify.wasm']) copyFileSync(join(pkg, 'dist', f), join(out, f));
console.log(`ONNX Runtime Web ${version} copied to ${out}`);
