"""Copies the files WebGPU Chat uses into AMINOXDEV repositories, byte for byte, with attribution.
python mirror.py <name>   (lfm3 | qwen | gemma)"""
import json, os, shutil, sys
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
W = os.path.expanduser(os.environ.get('MIRROR_WORK', '~/scratch/hf-mirror/work'))
QWEN_LICENSE = ('Qwen/Qwen3.5-4B', 'main', 'LICENSE')

def F(repo, rev, path, dest=None):
    return {'repo': repo, 'rev': rev, 'path': path, 'dest': dest or path}

LFM = ('onnx-community/LFM3-1.7B-ONNX', 'ab9fec4c681a25bec3b48b0b97a62fc2660a2985')
QW = ('onnx-community/Qwen3.5-4B-ONNX-OPT', 'bb072268d0912bdc5cf4f41c8c6f53a03fd68624')
GM = ('james58899/gemma-4-12B-it-qat-ONNX', '60555e645e7c456b4a013cc2e93e9bd6379e3006')
G_OWN = ('AMINOXDEV/gemma-4-12B-it-ONNX', '90e0a27bf539717b6055d7e434b9faa1f0dbd9ff')
G_JC = ('justinchuby/gemma-4-12b-onnx', '52a1c4fa78a69f30f96c8fe20f6368621bedbb57')
GRAPHS_Q = ('AMINOXDEV/webgpu-chat-graphs', 'd1b60fb60d2ff1d6520de073d9824d71cec35267')
GRAPHS_G = ('AMINOXDEV/webgpu-chat-graphs', 'fedd41d62b7427f8dbedc7819ee5bed5d0890832')

SPECS = {
  'lfm3': {
    'repo': 'AMINOXDEV/LFM3-1.7B-ONNX',
    'files': [F(*LFM, p) for p in ['LICENSE', 'chat_template.jinja', 'config.json', 'generation_config.json', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/model_q4f16.onnx', 'onnx/model_q4f16.onnx_data']],
    'card': {
      'license': 'other', 'license_name': 'lfm1.0', 'license_link': 'LICENSE', 'base_model': 'LiquidAI/LFM3-1.7B',
      'title': 'LFM3 1.7B (ONNX, q4f16) for WebGPU Chat',
      'body': 'Copy of the files [WebGPU Chat](https://llm.aminox.dev) uses from [onnx-community/LFM3-1.7B-ONNX](https://huggingface.co/onnx-community/LFM3-1.7B-ONNX/tree/{rev}) (revision `{rev}`), the ONNX conversion of [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B). Files are unchanged.\n\nLicence: LFM Open License v1.0 by Liquid AI, Inc. (see `LICENSE`).',
      'rev': LFM[1],
    },
  },
  'qwen': {
    'repo': 'AMINOXDEV/Qwen3.5-4B-ONNX',
    'files': [F(*QW, p) for p in ['chat_template.jinja', 'config.json', 'generation_config.json', 'preprocessor_config.json', 'processor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/embed_tokens_q4f16.onnx', 'onnx/embed_tokens_q4f16.onnx_data', 'onnx/decoder_model_merged_q4f16.onnx_data', 'onnx/decoder_model_merged_q4f16.onnx_data_1', 'onnx/vision_encoder_fp16.onnx', 'onnx/vision_encoder_fp16.onnx_data']]
      + [F(*GRAPHS_Q, 'qwen3.5-4b-opt/decoder_model_merged_q4f16.onnx', 'onnx/decoder_model_merged_q4f16.onnx'), F(*QWEN_LICENSE)],
    'card': {
      'license': 'apache-2.0', 'base_model': 'Qwen/Qwen3.5-4B',
      'title': 'Qwen3.5 4B (ONNX, q4f16, with fp16 vision encoder) for WebGPU Chat',
      'body': 'Copy of the files [WebGPU Chat](https://llm.aminox.dev) uses from [onnx-community/Qwen3.5-4B-ONNX-OPT](https://huggingface.co/onnx-community/Qwen3.5-4B-ONNX-OPT/tree/{rev}) (revision `{rev}`), the ONNX conversion of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).\n\n**Modified:** `onnx/decoder_model_merged_q4f16.onnx` (graph only) sets `accuracy_level=4` on its `MatMulNBits` nodes, so ONNX Runtime WebGPU can use int8 DP4A kernels for prompts. The weights (`.onnx_data`) are unchanged. All other files are unchanged.\n\nLicence: Apache 2.0, Qwen (Alibaba Cloud) (see `LICENSE`).',
      'rev': QW[1],
    },
  },
  'gemma': {
    'repo': 'AMINOXDEV/gemma-4-12B-it-qat-ONNX',
    'files': [F(*GM, p) for p in ['chat_template.jinja', 'generation_config.json', 'onnx/embed_tokens_q4f16.onnx', 'onnx/embed_tokens_q4f16.onnx_data', 'onnx/decoder_model_merged_q4f16.onnx_data', 'onnx/decoder_model_merged_q4f16.onnx_data_1', 'onnx/decoder_model_merged_q4f16.onnx_data_2', 'onnx/decoder_model_merged_q4f16.onnx_data_3']]
      + [F(*G_OWN, p) for p in ['config.json', 'tokenizer.json', 'tokenizer_config.json']]
      + [F(*GRAPHS_G, 'gemma-4-12b-qat/decoder_model_merged_q4f16.onnx', 'onnx/decoder_model_merged_q4f16.onnx'),
         F(*G_JC, 'f16/default/vision_encoder/model.onnx', 'onnx/vision_encoder_fp16.onnx'),
         F(*G_JC, 'f16/default/vision_encoder/model.onnx.data', 'onnx/vision_encoder_fp16.onnx.data'),
         F(*QWEN_LICENSE)],
    'card': {
      'license': 'apache-2.0', 'base_model': 'google/gemma-4-12B-it-qat-q4_0-unquantized',
      'title': 'Gemma 4 12B it QAT (ONNX, q4f16, with fp16 vision encoder) for WebGPU Chat',
      'body': 'Files [WebGPU Chat](https://llm.aminox.dev) uses for Gemma 4 12B, a derivative of [google/gemma-4-12B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized) (Gemma 4, Apache 2.0, Google DeepMind: https://ai.google.dev/gemma/docs/gemma_4_license).\n\nSources:\n- Weights, embeddings, `chat_template.jinja`, `generation_config.json`: [james58899/gemma-4-12B-it-qat-ONNX](https://huggingface.co/james58899/gemma-4-12B-it-qat-ONNX/tree/{rev}) (revision `{rev}`), unchanged.\n- **Modified** `onnx/decoder_model_merged_q4f16.onnx` (graph only): `accuracy_level=4` on its `MatMulNBits` nodes (int8 DP4A kernels for prompts in ONNX Runtime WebGPU). Weights unchanged.\n- `config.json`, `tokenizer.json`, `tokenizer_config.json`: written by us for Transformers.js (`model_type` `gemma4`; the same Google tokenizer and chat template).\n- `onnx/vision_encoder_fp16.onnx` and `onnx/vision_encoder_fp16.onnx.data`: [justinchuby/gemma-4-12b-onnx](https://huggingface.co/justinchuby/gemma-4-12b-onnx/tree/52a1c4fa78a69f30f96c8fe20f6368621bedbb57) `f16/default/vision_encoder/` (revision `52a1c4fa78a69f30f96c8fe20f6368621bedbb57`), renamed; the graph still names its weights file `model.onnx.data`.\n\nLicence: Apache 2.0 (see `LICENSE`).',
      'rev': GM[1],
    },
  },
}

def card(c):
    head = ['---', f"license: {c['license']}"]
    if 'license_name' in c: head += [f"license_name: {c['license_name']}", f"license_link: {c['license_link']}"]
    head += [f"base_model: {c['base_model']}", 'library_name: transformers.js', 'tags:', '- onnx', '- webgpu', '---', '', f"# {c['title']}", '', c['body'].format(rev=c['rev']), '']
    return '\n'.join(head)

name = sys.argv[1]
spec = SPECS[name]
out = os.path.join(W, name)
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)
cache = os.path.join(W, '_dl')
for f in spec['files']:
    p = hf_hub_download(f['repo'], f['path'], revision=f['rev'], local_dir=os.path.join(cache, f['repo'].replace('/', '__'), f['rev'][:8]))
    dest = os.path.join(out, f['dest'])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(p, dest)
    print('got', f['dest'], os.path.getsize(dest), flush=True)
shutil.rmtree(cache, ignore_errors=True)
open(os.path.join(out, 'README.md'), 'w').write(card(spec['card']))
api.create_repo(spec['repo'], repo_type='model', exist_ok=True, private=False)
info = api.upload_folder(repo_id=spec['repo'], folder_path=out, commit_message='Files used by WebGPU Chat (https://llm.aminox.dev), with attribution')
sha = info.oid
sizes = {f.path: f.size for f in api.get_paths_info(spec['repo'], [x['dest'] for x in spec['files']], revision=sha)}
bad = [x['dest'] for x in spec['files'] if sizes.get(x['dest']) != os.path.getsize(os.path.join(out, x['dest']))]
print(json.dumps({'repo': spec['repo'], 'revision': sha, 'sizes': sizes, 'mismatch': bad}))
shutil.rmtree(out)
