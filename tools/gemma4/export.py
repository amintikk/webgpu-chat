"""Export the text decoder of google/gemma-4-12B-it to ONNX (q4f16) for Transformers.js on WebGPU.

Produces, in --out:
  onnx/embed_tokens_q4f16.onnx (+ .onnx_data)            input_ids -> inputs_embeds
  onnx/decoder_model_merged_q4f16.onnx (+ .onnx_data_N)  inputs_embeds + KV cache -> logits + KV cache
  config.json, generation_config.json, tokenizer files, chat_template.jinja

The graph is written node by node (no torch.onnx tracing), so the 24 GB checkpoint is streamed one
tensor at a time. Linear layers use com.microsoft MatMulNBits (4-bit, block 32, asymmetric, fp16
scales), the same layout as the official onnx-community Gemma 4 exports. Norms and the residual stream
run in float32; matmuls, attention and the KV cache in float16. Sliding-window layers keep only the
last `sliding_window` positions in their cache.

    python tools/gemma4/export.py --out /path/to/gemma-4-12B-it-ONNX [--layers N]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, os.path.dirname(__file__))
from common import BLOCK, PREFIX, REPO, REVISION, RemoteSafetensors, inv_freqs, quantize_q4, text_config  # noqa: E402

F32, F16, I64, U8 = TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.INT64, TensorProto.UINT8
MAX_FILE = 1_900_000_000  # every external data file stays under 2 GB
ALIGN = 1 << 16  # page-aligned offsets let ONNX Runtime map the weights instead of copying them
LM_HEAD_CHUNKS = 4  # 262144 rows split in 4 so no single GPU buffer exceeds ~130 MB
ATTN_F32 = os.environ.get("ATTN_F32", "0") == "1"
NP = {F32: np.float32, F16: np.float16, I64: np.int64, U8: np.uint8}


class DataWriter:
    """Writes initializers to <name>.onnx_data, <name>.onnx_data_1, ... (each < 2 GB)."""

    def __init__(self, folder: str, model_name: str):
        self.folder, self.model_name = folder, model_name
        self.index, self.offset, self.f = -1, 0, None
        self.files: list[str] = []
        self._next()

    def _next(self):
        if self.f:
            self.f.close()
        self.index += 1
        name = f"{self.model_name}.onnx_data" + (f"_{self.index}" if self.index else "")
        self.files.append(name)
        self.f = open(os.path.join(self.folder, name), "wb")
        self.offset = 0

    def add(self, name: str, arr: np.ndarray) -> TensorProto:
        arr = np.ascontiguousarray(arr)
        data = arr.tobytes()
        pad = (-self.offset) % ALIGN
        if self.offset + pad + len(data) > MAX_FILE and self.offset > 0:
            self._next()
            pad = 0
        if pad:
            self.f.write(b"\0" * pad)
            self.offset += pad
        t = TensorProto()
        t.name = name
        t.data_type = helper.np_dtype_to_tensor_dtype(arr.dtype)
        t.dims.extend(arr.shape)
        t.data_location = TensorProto.EXTERNAL
        for k, v in (("location", self.files[-1]), ("offset", str(self.offset)), ("length", str(len(data)))):
            e = t.external_data.add()
            e.key, e.value = k, v
        self.f.write(data)
        self.offset += len(data)
        return t

    def close(self):
        self.f.close()


class Graph:
    """Minimal ONNX graph builder with automatic value names."""

    def __init__(self, writer: DataWriter):
        self.nodes, self.inits, self.inputs, self.outputs = [], [], [], []
        self.writer = writer
        self.n = 0

    def _name(self, hint: str) -> str:
        self.n += 1
        return f"{hint}_{self.n}"

    def op(self, op_type: str, inputs: list[str], hint: str | None = None, n_out: int = 1, domain: str = "", **attrs):
        outs = [self._name(hint or op_type) for _ in range(n_out)]
        self.nodes.append(helper.make_node(op_type, inputs, outs, name=self._name("n_" + op_type), domain=domain, **attrs))
        return outs[0] if n_out == 1 else outs

    def const(self, value, dtype=I64, hint: str = "c") -> str:
        name = self._name(hint)
        arr = np.asarray(value, dtype=NP[dtype])
        self.inits.append(helper.make_tensor(name, dtype, arr.shape, arr.flatten().tolist()))
        return name

    def weight(self, name: str, arr: np.ndarray) -> str:
        self.inits.append(self.writer.add(name, arr))
        return name

    # Common patterns --------------------------------------------------------------------------
    def cast(self, x, to):
        return self.op("Cast", [x], "cast", to=to)

    def rms(self, x, w_name: str, w: np.ndarray | None, dim: int, eps: float):
        """SimplifiedLayerNormalization in float32 (x float32). w=None means no scale (all ones)."""
        scale = self.weight(w_name, (w if w is not None else np.ones(dim)).astype(np.float32))
        return self.op("SimplifiedLayerNormalization", [x, scale], "rms", axis=-1, epsilon=eps, stash_type=1)

    def linear_q4(self, x16, w_name: str, w: np.ndarray):
        n, k = w.shape
        packed, scale, zp, _ = quantize_q4(w)
        ins = [
            x16,
            self.weight(f"{w_name}.qweight", packed),
            self.weight(f"{w_name}.scales", scale.reshape(n, k // BLOCK)),
            self.weight(f"{w_name}.zp", zp),
        ]
        return self.op("MatMulNBits", ins, "mm", domain="com.microsoft", K=k, N=n, bits=4, block_size=BLOCK)


def shape_dim(g: Graph, x: str, i: int) -> str:
    """Dimension i of x as a 1-D int64 tensor of length 1."""
    return g.op("Shape", [x], "shape", start=i, end=i + 1)


def rotate_half(g: Graph, x: str, hd: int) -> str:
    x1 = g.op("Slice", [x, g.const([0]), g.const([hd // 2]), g.const([-1])], "x1")
    x2 = g.op("Slice", [x, g.const([hd // 2]), g.const([hd]), g.const([-1])], "x2")
    return g.op("Concat", [g.op("Neg", [x2]), x1], "rot", axis=-1)


def build_decoder(st: RemoteSafetensors, tc, folder: str, n_layers: int):
    name = "decoder_model_merged_q4f16"
    writer = DataWriter(folder, name)
    g = Graph(writer)
    D, H, eps, W = tc.hidden_size, tc.num_attention_heads, tc.rms_norm_eps, tc.sliding_window
    freqs = inv_freqs(tc)
    types = tc.layer_types[:n_layers]

    def layer_dims(i):
        lc = tc.per_layer_config[i]
        return lc.head_dim, lc.num_key_value_heads

    g.inputs.append(helper.make_tensor_value_info("inputs_embeds", F32, ["batch_size", "sequence_length", D]))
    g.inputs.append(helper.make_tensor_value_info("attention_mask", I64, ["batch_size", "total_sequence_length"]))
    g.inputs.append(helper.make_tensor_value_info("position_ids", I64, ["batch_size", "sequence_length"]))
    g.inputs.append(helper.make_tensor_value_info("num_logits_to_keep", I64, []))
    g.outputs.append(helper.make_tensor_value_info("logits", F16, ["batch_size", "num_logits_to_keep", tc.vocab_size]))
    # Transformers.js reads the cached length from the first past_key_values entry, which must therefore be a
    # full-attention layer (sliding layers keep a truncated cache). List full layers first.
    order = sorted(range(n_layers), key=lambda i: (types[i] != "full_attention", i))
    for i in order:
        hd, kvh = layer_dims(i)
        past = "past_sequence_length" if types[i] == "full_attention" else "past_sliding_length"
        pres = "total_sequence_length" if types[i] == "full_attention" else "sliding_length"
        for kv in ("key", "value"):
            g.inputs.append(helper.make_tensor_value_info(f"past_key_values.{i}.{kv}", F16, ["batch_size", kvh, past, hd]))
            g.outputs.append(helper.make_tensor_value_info(f"present.{i}.{kv}", F16, ["batch_size", kvh, pres, hd]))

    # Rotary tables per layer type: cos/sin [B, S, 1, hd] in float32.
    pos = g.op("Unsqueeze", [g.cast("position_ids", F32), g.const([-1])], "pos")
    rope = {}
    for t, inv in freqs.items():
        f = g.op("Mul", [pos, g.const(inv, F32, f"inv_freq_{t}")], "freqs")
        emb = g.op("Concat", [f, f], "emb", axis=-1)
        rope[t] = (
            g.op("Unsqueeze", [g.op("Cos", [emb]), g.const([2])], "cos"),
            g.op("Unsqueeze", [g.op("Sin", [emb]), g.const([2])], "sin"),
        )

    # Attention bias per layer type: [B, 1, 1, S, L] float32 (0 or -1e9), L = cached + new positions.
    T = shape_dim(g, "attention_mask", 1)
    S = shape_dim(g, "inputs_embeds", 1)
    qpos = g.op("Unsqueeze", ["position_ids", g.const([-1])], "qpos")  # [B, S, 1]
    mask_cache = {}

    def attention_bias(t: str, past_name: str):
        if t in mask_cache:
            return mask_cache[t]
        L = g.op("Add", [shape_dim(g, past_name, 2), S], "L")
        Ls = g.op("Squeeze", [L, g.const([0])], "Ls")
        first = g.op("Sub", [T, L], "first")  # absolute position of the first cached key
        kpos = g.op("Add", [g.op("Range", [g.const(0), Ls, g.const(1)], "ar"), first], "kpos")  # [L]
        ok = g.op("LessOrEqual", [kpos, qpos], "causal")  # [B, S, L]
        if t == "sliding_attention":
            ok = g.op("And", [ok, g.op("Less", [g.op("Sub", [qpos, kpos]), g.const(W)], "inwin")], "ok")
        am = g.op("Slice", ["attention_mask", first, g.const([2**62]), g.const([1])], "am")  # [B, L]
        am = g.op("Unsqueeze", [g.op("Equal", [am, g.const(1)]), g.const([1])], "am1")  # [B, 1, L]
        ok = g.op("And", [ok, am], "ok2")
        bias = g.op("Where", [ok, g.const(0.0, F32), g.const(-1e9, F32)], "bias")
        mask_cache[t] = g.op("Unsqueeze", [bias, g.const([1, 2])], "bias5")
        return mask_cache[t]

    def lw(i, n):
        return st.get(f"{PREFIX}layers.{i}.{n}")

    x = "inputs_embeds"
    t0 = time.time()
    for i in range(n_layers):
        t = types[i]
        hd, kvh = layer_dims(i)
        rep = H // kvh
        p = f"layers.{i}"
        cos, sin = rope[t]

        def heads(v, nh):
            return g.op("Reshape", [v, g.const([0, 0, nh, hd])], "heads")

        def rope_apply(v):
            return g.op("Add", [g.op("Mul", [v, cos]), g.op("Mul", [rotate_half(g, v, hd), sin])], "roped")

        h16 = g.cast(g.rms(x, f"{p}.input_layernorm", lw(i, "input_layernorm.weight"), D, eps), F16)
        q = g.cast(heads(g.linear_q4(h16, f"{p}.q_proj", lw(i, "self_attn.q_proj.weight")), H), F32)
        q = rope_apply(g.rms(q, f"{p}.q_norm", lw(i, "self_attn.q_norm.weight"), hd, eps))
        k_raw = g.cast(heads(g.linear_q4(h16, f"{p}.k_proj", lw(i, "self_attn.k_proj.weight")), kvh), F32)
        if t == "full_attention" and tc.attention_k_eq_v:
            v_raw = k_raw
        else:
            v_raw = g.cast(heads(g.linear_q4(h16, f"{p}.v_proj", lw(i, "self_attn.v_proj.weight")), kvh), F32)
        k = rope_apply(g.rms(k_raw, f"{p}.k_norm", lw(i, "self_attn.k_norm.weight"), hd, eps))
        v = g.rms(v_raw, f"{p}.v_norm", None, hd, eps)
        perm = [0, 2, 1, 3]
        # Attention math runs in float16 (default, like the official q4f16 exports) or float32 with ATTN_F32=1;
        # the KV cache is float16 either way. Softmax always runs in float32.
        q = g.op("Transpose", [q if ATTN_F32 else g.cast(q, F16)], "qT", perm=perm)  # [B, H, S, hd]
        k = g.op("Transpose", [g.cast(k, F16)], "kT", perm=perm)  # [B, kvh, S, hd]
        v = g.op("Transpose", [g.cast(v, F16)], "vT", perm=perm)
        pk, pv = f"past_key_values.{i}.key", f"past_key_values.{i}.value"
        K = g.op("Concat", [pk, k], "K", axis=2)
        V = g.op("Concat", [pv, v], "V", axis=2)
        if t == "sliding_attention":
            keep = [g.const([-W]), g.const([2**62]), g.const([2])]
            g.nodes.append(helper.make_node("Slice", [K, *keep], [f"present.{i}.key"]))
            g.nodes.append(helper.make_node("Slice", [V, *keep], [f"present.{i}.value"]))
        else:
            g.nodes.append(helper.make_node("Identity", [K], [f"present.{i}.key"]))
            g.nodes.append(helper.make_node("Identity", [V], [f"present.{i}.value"]))
        # Grouped heads without repeating K/V: [B, kvh, rep*S, hd] x [B, kvh, hd, L].
        Ka, Va = (g.cast(K, F32), g.cast(V, F32)) if ATTN_F32 else (K, V)
        qg = g.op("Reshape", [q, g.const([0, kvh, -1, hd])], "qg")
        sc = g.op("MatMul", [qg, g.op("Transpose", [Ka], "Kt", perm=[0, 1, 3, 2])], "scores")  # scaling = 1.0
        sc = g.op("Reshape", [sc if ATTN_F32 else g.cast(sc, F32), g.op("Concat", [g.const([0, kvh, rep]), S, g.const([-1])], "s5", axis=0)], "sc5")
        pr = g.op("Softmax", [g.op("Add", [sc, attention_bias(t, pk)], "masked")], "probs", axis=-1)
        pr = g.op("Reshape", [pr if ATTN_F32 else g.cast(pr, F16), g.op("Concat", [g.const([0, kvh, -1]), shape_dim(g, K, 2)], "s4", axis=0)], "pr4")
        ctx = g.op("MatMul", [pr, Va], "ctx")  # [B, kvh, rep*S, hd]
        ctx = g.op("Reshape", [ctx, g.const([0, H, -1, hd])], "ctx4")
        ctx = g.op("Reshape", [g.op("Transpose", [ctx], "ctxT", perm=perm), g.const([0, 0, H * hd])], "ctx3")
        ctx = g.cast(ctx, F16) if ATTN_F32 else ctx
        o = g.cast(g.linear_q4(ctx, f"{p}.o_proj", lw(i, "self_attn.o_proj.weight")), F32)
        x = g.op("Add", [x, g.rms(o, f"{p}.post_attention_layernorm", lw(i, "post_attention_layernorm.weight"), D, eps)], "resid")

        h16 = g.cast(g.rms(x, f"{p}.pre_feedforward_layernorm", lw(i, "pre_feedforward_layernorm.weight"), D, eps), F16)
        gate_up = np.concatenate([lw(i, "mlp.gate_proj.weight"), lw(i, "mlp.up_proj.weight")], axis=0)
        gu = g.linear_q4(h16, f"{p}.gate_up_proj", gate_up)
        del gate_up
        gate, up = g.op("Split", [gu], "gu", n_out=2, axis=-1, num_outputs=2)
        act = g.op("Mul", [g.op("Gelu", [gate], "gelu", approximate="tanh"), up], "act")
        m = g.cast(g.linear_q4(act, f"{p}.down_proj", lw(i, "mlp.down_proj.weight")), F32)
        x = g.op("Add", [x, g.rms(m, f"{p}.post_feedforward_layernorm", lw(i, "post_feedforward_layernorm.weight"), D, eps)], "resid")
        scalar = float(lw(i, "layer_scalar")[0])
        if scalar != 1.0:
            x = g.op("Mul", [x, g.const(scalar, F32)], "scaled")
        print(f"layer {i + 1}/{n_layers} ({t}) {time.time() - t0:.0f}s", flush=True)

    # Final norm, keep the last num_logits_to_keep positions (0 = all), tied LM head in chunks, soft-capping.
    x = g.rms(x, "norm", st.get(PREFIX + "norm.weight"), D, eps)
    n = "num_logits_to_keep"
    start = g.op("Where", [g.op("Greater", [n, g.const(0)]), g.op("Neg", [n]), g.const(0)], "start")
    x16 = g.cast(g.op("Slice", [x, g.op("Unsqueeze", [start, g.const([0])]), g.const([2**62]), g.const([1])], "kept"), F16)
    V_ = tc.vocab_size
    rows = V_ // LM_HEAD_CHUNKS
    parts = []
    for c in range(LM_HEAD_CHUNKS):
        w = st.get(PREFIX + "embed_tokens.weight", slice(c * rows, (c + 1) * rows))
        parts.append(g.linear_q4(x16, f"lm_head.{c}", w))
        del w
    logits = g.op("Concat", parts, "logits_raw", axis=-1)
    cap = tc.final_logit_softcapping
    logits = g.op("Mul", [g.op("Tanh", [g.op("Div", [logits, g.const(cap, F16)])]), g.const(cap, F16)], "capped")
    g.nodes.append(helper.make_node("Identity", [logits], ["logits"]))

    writer.close()
    save(g, folder, name)
    return writer.files


def build_embed(st: RemoteSafetensors, tc, folder: str):
    name = "embed_tokens_q4f16"
    writer = DataWriter(folder, name)
    g = Graph(writer)
    D, V_ = tc.hidden_size, tc.vocab_size
    packed, scales, zps = [], [], []
    step = 16384
    for r in range(0, V_, step):
        p, s, z, _ = quantize_q4(st.get(PREFIX + "embed_tokens.weight", slice(r, r + step)))
        packed.append(p.reshape(p.shape[0], -1))
        scales.append(s)
        zps.append(z)
    g.inputs.append(helper.make_tensor_value_info("input_ids", I64, ["batch_size", "sequence_length"]))
    g.outputs.append(helper.make_tensor_value_info("inputs_embeds", F32, ["batch_size", "sequence_length", D]))
    ins = [
        g.weight("embed_tokens.qweight", np.concatenate(packed)),
        "input_ids",
        g.weight("embed_tokens.scales", np.concatenate(scales)),
        g.weight("embed_tokens.zp", np.concatenate(zps)),
    ]
    e = g.op("GatherBlockQuantized", ins, "emb", domain="com.microsoft", bits=4, block_size=BLOCK, gather_axis=0, quantize_axis=1)
    # The official module multiplies by sqrt(hidden_size) in the weights' dtype.
    scale = float(np.float32(D**0.5))
    g.nodes.append(helper.make_node("Mul", [g.cast(e, F32), g.const(scale, F32)], ["inputs_embeds"]))
    writer.close()
    save(g, folder, name)
    return writer.files


def save(g: Graph, folder: str, name: str):
    graph = helper.make_graph(g.nodes, name, g.inputs, g.outputs, g.inits)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        producer_name="webgpu-chat tools/gemma4/export.py",
    )
    model.ir_version = 10
    onnx.save(model, os.path.join(folder, f"{name}.onnx"))
    # onnx.checker does not know the ORT-specific ops (SimplifiedLayerNormalization); ORT validates on load.


def write_configs(out: str, src: str, files: dict[str, list[str]], n_layers: int):
    from huggingface_hub import hf_hub_download

    cfg = json.load(open(hf_hub_download(REPO, "config.json", revision=REVISION, local_dir=src)))
    # Transformers.js runs the text decoder through its Gemma 4 classes (text-only sessions).
    cfg["model_type"] = "gemma4"
    cfg["architectures"] = ["Gemma4ForConditionalGeneration"]
    cfg["text_config"]["model_type"] = "gemma4_text"
    cfg["text_config"]["num_hidden_layers"] = n_layers
    cfg["text_config"]["layer_types"] = cfg["text_config"]["layer_types"][:n_layers]
    for k in ("audio_config", "vision_config"):
        cfg.pop(k, None)
    cfg["transformers.js_config"] = {
        "use_external_data_format": {f"{k}.onnx": len(v) for k, v in files.items()},
        "kv_cache_dtype": {"q4f16": "float16"},
    }
    cfg["_export"] = {
        "source": f"https://huggingface.co/{REPO}/tree/{REVISION}",
        "tool": "https://github.com/amintikk/webgpu-chat/tree/main/tools/gemma4",
        "note": "Text-only decoder; vision and audio towers are not included.",
    }
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    for f in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(hf_hub_download(REPO, f, revision=REVISION, local_dir=src), os.path.join(out, f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=0, help="export only the first N layers (smoke tests)")
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--tag", default="", help="free text; lets a watchdog find this process")
    args = ap.parse_args()
    tc = text_config()
    n_layers = args.layers or tc.num_hidden_layers
    onnx_dir = os.path.join(args.out, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    st = RemoteSafetensors()
    files = {}
    if not args.skip_embed:
        files["embed_tokens_q4f16"] = build_embed(st, tc, onnx_dir)
        print("embed done", flush=True)
    files["decoder_model_merged_q4f16"] = build_decoder(st, tc, onnx_dir, n_layers)
    if args.skip_embed:
        files["embed_tokens_q4f16"] = [f for f in os.listdir(onnx_dir) if f.startswith("embed_tokens_q4f16.onnx_data")]
    write_configs(args.out, os.path.join(args.out, ".src"), files, n_layers)
    shutil.rmtree(os.path.join(args.out, ".src"), ignore_errors=True)
    print("done", files, flush=True)


if __name__ == "__main__":
    main()
