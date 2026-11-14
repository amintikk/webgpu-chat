---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-12B-it
base_model_relation: quantized
library_name: transformers.js
pipeline_tag: text-generation
language:
- multilingual
tags:
- gemma4
- onnx
- webgpu
- q4f16
---

# Gemma 4 12B IT: ONNX q4f16 for Transformers.js / WebGPU (text only)

4-bit ONNX export of the text decoder of [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it)
(revision `43c9d014b231474fa7212dbb48a762566c2b255d`), made to run in the browser with
[Transformers.js](https://github.com/huggingface/transformers.js) v4 on WebGPU. It powers the “Maximum”
model of [WebGPU Chat](https://llm.aminox.dev).

**This is a modified version of the original model** (Apache 2.0, see `LICENSE` and NOTICE below):
the weights of the linear layers and the embedding table were quantized to 4 bits, and the vision and
audio towers were removed. It is not an official Google release.

## Files

| File | Content |
|---|---|
| `onnx/embed_tokens_q4f16.onnx` (+ `.onnx_data`) | token ids → embeddings × √hidden (GatherBlockQuantized, 4-bit) |
| `onnx/decoder_model_merged_q4f16.onnx` (+ `.onnx_data` … `_3`) | 48 decoder layers with KV cache → logits |
| `config.json`, `generation_config.json`, `tokenizer*.json`, `chat_template.jinja` | from the original repository; `config.json` adapted for Transformers.js (`gemma4`, text only) |

- Linear layers: `com.microsoft.MatMulNBits`, 4 bits, block 32, asymmetric (zero points), fp16 scales,
  round-to-nearest. The tied LM head is split in 4 chunks (≤ 130 MB per GPU buffer).
- Activations, attention and KV cache in float16; RMSNorm, the residual stream and softmax in float32.
- Sliding-window layers keep only the last 1,024 positions in their cache; global layers keep everything.
- Every external data file is smaller than 2 GB. Total download ≈ 7.1 GB.

## Usage (Transformers.js ≥ 4.3.2, WebGPU, shader-f16)

```js
import { AutoTokenizer, Gemma4ForCausalLM } from '@huggingface/transformers';

const id = 'AMINOXDEV/gemma-4-12B-it-ONNX';
const tokenizer = await AutoTokenizer.from_pretrained(id);
const model = await Gemma4ForCausalLM.from_pretrained(id, {
  dtype: { embed_tokens: 'q4f16', decoder_model_merged: 'q4f16' },
  device: 'webgpu',
});
const inputs = tokenizer.apply_chat_template([{ role: 'user', content: 'Hello!' }], { add_generation_prompt: true, return_dict: true });
const out = await model.generate({ ...inputs, max_new_tokens: 128 });
console.log(tokenizer.decode(out.slice(null, [inputs.input_ids.dims[1], null]), { skip_special_tokens: true }));
```

Needs a GPU with roughly 10 GB of free memory.

## How it was made and checked

The graph is written node by node by [`tools/gemma4/export.py`](https://github.com/amintikk/webgpu-chat/tree/main/tools/gemma4)
from the official safetensors, read tensor by tensor with HTTP range requests (the math follows
`transformers` 5.23 `modeling_gemma4_unified.py`). [`verify.py`](https://github.com/amintikk/webgpu-chat/tree/main/tools/gemma4)
runs this ONNX model with greedy decoding and compares every generated token with the official PyTorch
modules (teacher forcing, equivalent to comparing greedy outputs), both against the original bf16
weights and against the same weights after this 4-bit quantization.

Result ([`parity.json`](https://github.com/amintikk/webgpu-chat/blob/main/results/gemma4/parity.json), 6 prompts in Spanish and English, 274 greedy steps):

- **Same q4 weights in PyTorch: 97.8 % of steps agree**, and every disagreement is a near-tie (top-1/top-2 margin below 0.7 logits), i.e. float16 rounding, not a graph error. Criterion: ≥ 97 % and margins < 1.0 → **pass**.
- **Original bf16 weights: 92.3 % of steps agree** (the effect of 4-bit quantization). Criterion: ≥ 90 % → **pass**.

| Prompt | Steps | Agreement with original bf16 | Identical greedy prefix | Agreement with same q4 weights | Identical greedy prefix | Mismatch margins (q4 track) |
|---|---|---|---|---|---|---|
| es-chat | 48 | 91.7 % | 10 | 100.0 % | 48 | — |
| en-chat | 34 | 82.3 % | 21 | 94.1 % | 24 | 0.68, 0.279 |
| es-code | 48 | 95.8 % | 0 | 97.9 % | 15 | 0.454 |
| en-code | 48 | 93.8 % | 8 | 95.8 % | 34 | 0.187, 0.101 |
| es-reason | 48 | 89.6 % | 0 | 97.9 % | 1 | 0.057 |
| en-reason | 48 | 97.9 % | 32 | 100.0 % | 48 | — |

“Identical greedy prefix” is how many tokens the greedy outputs share before the first disagreement.

## Limitations

Quantization makes the model somewhat less accurate than the original. Text only (no images or audio).
Like any language model it can produce wrong or invented information. See the original
[model card](https://huggingface.co/google/gemma-4-12B-it) for intended use and limitations.

## NOTICE

Gemma 4 is provided by Google under the Apache License 2.0. Copyright Google LLC.
Modifications in this repository (ONNX conversion, 4-bit quantization, removal of the vision and audio
towers, Transformers.js configuration) by Amin Tahtah Toumi, also under the Apache License 2.0.
