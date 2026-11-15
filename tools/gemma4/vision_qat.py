"""Gemma 4 12B picture encoder with the QAT checkpoint's vision weights.

Takes the ONNX Runtime team's fp16 encoder graph (justinchuby/gemma-4-12b-onnx, f16/default/vision_encoder, whose
weights are those of google/gemma-4-12B-it) and replaces every weight with the one of
google/gemma-4-12B-it-qat-q4_0-unquantized, read tensor by tensor from its safetensors (common.py).
Layouts were checked against the original: patches (y, x, c), pos_emb_x = pos_embedding[:, 0], weights transposed.
    python vision_qat.py <encoder.onnx dir with model.onnx.data> <out dir>
"""
import os, sys
import numpy as np, onnx
from onnx import numpy_helper as nh
from onnx.external_data_helper import load_external_data_for_model, convert_model_to_external_data
from common import RemoteSafetensors

src_dir, out_dir = sys.argv[1], sys.argv[2]
q = RemoteSafetensors('google/gemma-4-12B-it-qat-q4_0-unquantized', 'b6ed86275a6a5735884e208bfed95b445a684ca2')
m = onnx.load(os.path.join(src_dir, 'model.onnx'), load_external_data=False)
load_external_data_for_model(m, src_dir)
V = 'model.vision_embedder.'
pe = q.get(V + 'pos_embedding')
src = {
    'vision_encoder.patch_ln1.weight': q.get(V + 'patch_ln1.weight'), 'vision_encoder.patch_ln1.bias': q.get(V + 'patch_ln1.bias'),
    'vision_encoder.patch_dense.weight_t': q.get(V + 'patch_dense.weight').T, 'vision_encoder.patch_dense.bias': q.get(V + 'patch_dense.bias'),
    'vision_encoder.patch_ln2.weight': q.get(V + 'patch_ln2.weight'), 'vision_encoder.patch_ln2.bias': q.get(V + 'patch_ln2.bias'),
    'vision_encoder.pos_emb_x.weight': pe[:, 0], 'vision_encoder.pos_emb_y.weight': pe[:, 1],
    'vision_encoder.pos_norm.weight': q.get(V + 'pos_norm.weight'), 'vision_encoder.pos_norm.bias': q.get(V + 'pos_norm.bias'),
    'vision_encoder.projector.weight_t': q.get('model.embed_vision.embedding_projection.weight').T,
}
done = 0
for init in m.graph.initializer:
    if init.name in src:
        old = nh.to_array(init)
        new = src[init.name].astype(old.dtype)
        assert new.shape == old.shape, (init.name, new.shape, old.shape)
        init.CopyFrom(nh.from_array(np.ascontiguousarray(new), init.name))
        done += 1
assert done == len(src)
os.makedirs(out_dir, exist_ok=True)
convert_model_to_external_data(m, all_tensors_to_one_file=True, location='vision_encoder_qat_fp16.onnx.data', size_threshold=1024)
onnx.save(m, os.path.join(out_dir, 'vision_encoder_qat_fp16.onnx'))
print('replaced', done, 'weights')
