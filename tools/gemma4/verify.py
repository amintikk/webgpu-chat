"""Greedy-decoding parity of the ONNX export against the official PyTorch Gemma 4 modeling code.

1. ONNX Runtime (CPU) runs the exported embed + decoder graphs with the KV cache and greedy decoding,
   exactly as Transformers.js does in the browser, for every prompt in PROMPTS.
2. PyTorch runs the official `Gemma4UnifiedTextDecoderLayer` modules, one layer at a time (weights are
   streamed from Hugging Face), over prompt + ONNX continuation in a single teacher-forced pass, on two
   tracks:
     - original:   the bf16 checkpoint (computed in float32);
     - quantized:  the same weights after the 4-bit quantization used by the export (dequantized).
   At every generated position, greedy decoding of that track would have chosen argmax(logits). If it
   equals the ONNX token at every step, the track's greedy output is identical to the ONNX one; the
   first disagreement is where the two greedy outputs would diverge.

Pass criteria (decided before running, written to the report):
  - export correctness: quantized track agrees with ONNX on >= 97 % of steps and every disagreement is
    a near-tie (reference top-1/top-2 margin < 1.0 logit);
  - fidelity to the original: original track agrees with ONNX on >= 90 % of steps.

    python tools/gemma4/verify.py --model /path/to/export --report tools/gemma4/parity.json [--layers N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import PREFIX, REPO, REVISION, RemoteSafetensors, quantize_q4, text_config  # noqa: E402

PROMPTS = [
    ("es-chat", "¿Qué tres consejos darías a alguien que empieza a correr?"),
    ("en-chat", "Write a two-sentence thank-you note to a colleague who helped me move house."),
    ("es-code", "Escribe una función en Python que devuelva los números primos menores que n."),
    ("en-code", "Explain what this JavaScript does: const s = [...new Set(arr)].sort((a, b) => a - b);"),
    ("es-reason", "Si un tren sale a las 9:40 y el viaje dura 2 h 35 min, ¿a qué hora llega? Explica el cálculo."),
    ("en-reason", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"),
]


def protect_server():
    """If memory runs out, the kernel should kill this verification, never a server process."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass


class NumpyEmbed:
    """Same computation as embed_tokens_q4f16.onnx, reading only the rows it needs from the mapped file.

    Saves loading the 0.6 GB embedding session next to the 7 GB decoder; `check()` compares both.
    """

    def __init__(self, model_dir: str):
        import onnx

        path = os.path.join(model_dir, "onnx/embed_tokens_q4f16.onnx")
        m = onnx.load(path, load_external_data=False)
        self.t = {}
        for init in m.graph.initializer:
            if not init.external_data:
                continue
            ext = {e.key: e.value for e in init.external_data}
            dtype = {onnx.TensorProto.UINT8: np.uint8, onnx.TensorProto.FLOAT16: np.float16}[init.data_type]
            self.t[init.name] = np.memmap(os.path.join(model_dir, "onnx", ext["location"]), dtype=dtype, mode="r",
                                          offset=int(ext["offset"]), shape=tuple(init.dims))
        consts = [n for n in m.graph.initializer if not n.external_data]
        self.mul = float(onnx.numpy_helper.to_array(consts[0])) if consts else 1.0
        self.path = path

    def __call__(self, ids):
        q = self.t["embed_tokens.qweight"][ids].astype(np.uint8)
        sc = self.t["embed_tokens.scales"][ids].astype(np.float32)
        zp = self.t["embed_tokens.zp"][ids]
        vals = np.stack([q & 15, q >> 4], -1).reshape(len(ids), -1).astype(np.float32)
        z = np.stack([zp & 15, zp >> 4], -1).reshape(len(ids), -1).astype(np.float32)
        w = (vals.reshape(len(ids), -1, 32) - z[..., None]) * sc[..., None]
        return (w.reshape(len(ids), -1).astype(np.float16).astype(np.float32) * np.float32(self.mul))[None]

    def check(self, ids):
        import onnxruntime as ort

        sess = ort.InferenceSession(self.path, providers=["CPUExecutionProvider"])
        a = sess.run(None, {"input_ids": np.array([ids], np.int64)})[0]
        return float(np.max(np.abs(a - self(ids))))


def onnx_greedy(dec, embed, ids: list[int], n_new: int, eos: set[int]):
    past = {}
    for i in dec.get_inputs():
        if i.name.startswith("past_key_values"):
            past[i.name] = np.zeros([1 if d == "batch_size" else (0 if isinstance(d, str) else d) for d in i.shape], np.float16)
    names = [o.name for o in dec.get_outputs()]
    seq, out, step_logits = list(ids), [], []
    new = list(ids)
    for _ in range(n_new):
        total = len(seq)
        feeds = {
            "inputs_embeds": embed(new),
            "attention_mask": np.ones((1, total), np.int64),
            "position_ids": np.arange(total - len(new), total, dtype=np.int64)[None],
            "num_logits_to_keep": np.array(1, np.int64),
            **past,
        }
        res = dec.run(None, feeds)
        logits = res[0][0, -1].astype(np.float32)
        for n, v in zip(names[1:], res[1:]):
            past[n.replace("present", "past_key_values")] = v
        tok = int(np.argmax(logits))
        step_logits.append(logits)
        out.append(tok)
        seq.append(tok)
        new = [tok]
        if tok in eos:
            break
    return out, step_logits


def reference(tc, n_layers: int, seqs: list[list[int]], st: RemoteSafetensors):
    """Teacher-forced logits [len(seq), vocab] for both tracks, streaming one layer at a time."""
    import torch
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedRMSNorm,
        Gemma4UnifiedTextDecoderLayer,
        Gemma4UnifiedTextRotaryEmbedding,
    )

    torch.set_grad_enabled(False)
    tc._attn_implementation = "eager"
    D, V = tc.hidden_size, tc.vocab_size
    tracks = ("original", "quantized")
    rows = 65536
    # Input embeddings (the official module multiplies by sqrt(hidden_size) in the weights' dtype).
    hs = {t: [np.zeros((len(s), D), np.float32) for s in seqs] for t in tracks}
    for r in range(0, V, rows):
        w = st.get(PREFIX + "embed_tokens.weight", slice(r, r + rows))
        wq = quantize_q4(w)[3]
        for si, s in enumerate(seqs):
            for pi, tok in enumerate(s):
                if r <= tok < r + rows:
                    hs["original"][si][pi] = w[tok - r]
                    hs["quantized"][si][pi] = wq[tok - r]
        del w, wq
    scale = np.float32(D**0.5)
    h = {t: [torch.from_numpy(x * scale)[None] for x in hs[t]] for t in tracks}
    rot = Gemma4UnifiedTextRotaryEmbedding(tc)

    def masks(n):
        i = torch.arange(n)
        causal = i[None, :] <= i[:, None]
        window = (i[:, None] - i[None, :]) < tc.sliding_window
        f = lambda ok: torch.where(ok, 0.0, float("-inf"))[None, None]
        return {"full_attention": f(causal), "sliding_attention": f(causal & window)}

    aux = []
    for s in seqs:
        pos = torch.arange(len(s))[None]
        dummy = torch.zeros(1, len(s), D)
        pe = {t: rot(dummy, pos, t) for t in set(tc.layer_types)}
        aux.append((pos, pe, masks(len(s))))

    linear = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    t0 = time.time()
    for li in range(n_layers):
        layer = Gemma4UnifiedTextDecoderLayer(tc, li).eval()
        sd = {}
        for k in layer.state_dict():
            sd[k] = torch.from_numpy(st.get(f"{PREFIX}layers.{li}.{k}"))
        for track in tracks:
            if track == "quantized":
                for k in sd:
                    if k.endswith(".weight") and k.split(".")[-2] in linear:
                        sd[k] = torch.from_numpy(quantize_q4(sd[k].numpy())[3])
            layer.load_state_dict(sd)
            for si in range(len(seqs)):
                pos, pe, mk = aux[si]
                t = tc.layer_types[li]
                h[track][si] = layer(h[track][si], position_embeddings=pe[t], attention_mask=mk[t], position_ids=pos)
        print(f"  reference layer {li + 1}/{n_layers} {time.time() - t0:.0f}s", flush=True)

    norm = Gemma4UnifiedRMSNorm(D, eps=tc.rms_norm_eps)
    norm.weight.data = torch.from_numpy(st.get(PREFIX + "norm.weight"))
    xs = {t: [norm(x)[0] for x in h[t]] for t in tracks}
    logits = {t: [np.zeros((len(s), V), np.float32) for s in seqs] for t in tracks}
    for r in range(0, V, rows):
        w = st.get(PREFIX + "embed_tokens.weight", slice(r, r + rows))
        ws = {"original": torch.from_numpy(w), "quantized": torch.from_numpy(quantize_q4(w)[3])}
        for t in tracks:
            for si in range(len(seqs)):
                logits[t][si][:, r : r + rows] = (xs[t][si] @ ws[t].T).numpy()
        del w, ws
    cap = tc.final_logit_softcapping
    return {t: [np.tanh(l / cap) * cap for l in logits[t]] for t in tracks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--tag", default="")
    ap.add_argument("--prepack", action="store_true", help="faster, but copies all weights into RAM")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tc = text_config()
    n_layers = args.layers or tc.num_hidden_layers
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    eos = {1, 106, 50}
    st = RemoteSafetensors()

    import onnxruntime as ort

    protect_server()
    embed = NumpyEmbed(args.model)
    probe = tok.encode("Hola, ¿qué tal? Hello world 123")
    print("embed check max abs diff (numpy vs onnx):", embed.check(probe), flush=True)
    so = ort.SessionOptions()
    # Without prepacking ONNX Runtime maps the weight files instead of copying them into its own memory:
    # ~15x slower on CPU, but the 7 GB of weights stay in reclaimable page cache on a shared server.
    if not args.prepack:
        so.add_session_config_entry("session.disable_prepacking", "1")
    so.intra_op_num_threads = 3
    dec = ort.InferenceSession(os.path.join(args.model, "onnx/decoder_model_merged_q4f16.onnx"), so, providers=["CPUExecutionProvider"])
    cases = []
    for name, text in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False, tokenize=True)
        ids = list(ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids)
        t0 = time.time()
        out, onnx_logits = onnx_greedy(dec, embed, ids, args.new_tokens, eos)
        dt = time.time() - t0
        print(f"[onnx] {name}: {len(out)} tokens in {dt:.0f}s: {tok.decode(out)!r}", flush=True)
        cases.append({"name": name, "prompt": text, "ids": ids, "out": out, "onnx_logits": onnx_logits})

    del dec
    seqs = [c["ids"] + c["out"] for c in cases]
    ref = reference(tc, n_layers, seqs, st)

    report = {"model": args.model, "layers": n_layers, "source": f"{REPO}@{REVISION}", "cases": []}
    totals = {"original": [0, 0], "quantized": [0, 0]}
    near_tie_only = True
    for ci, c in enumerate(cases):
        p = len(c["ids"])
        entry = {"name": c["name"], "prompt": c["prompt"], "onnx_output": tok.decode(c["out"]), "steps": len(c["out"])}
        for track in ("original", "quantized"):
            lg = ref[track][ci][p - 1 : p - 1 + len(c["out"])]  # logits that predict each generated token
            agree, first, mism, diffs = 0, None, [], []
            for s, t in enumerate(c["out"]):
                srt = np.argsort(lg[s])[::-1]
                margin = float(lg[s][srt[0]] - lg[s][srt[1]])
                diffs.append(float(np.max(np.abs(lg[s] - c["onnx_logits"][s]))))
                if srt[0] == t:
                    agree += 1
                else:
                    first = s if first is None else first
                    mism.append({"step": s, "onnx": tok.decode([t]), "reference": tok.decode([int(srt[0])]), "margin": round(margin, 3)})
                    if track == "quantized" and margin >= 1.0:
                        near_tie_only = False
            totals[track][0] += agree
            totals[track][1] += len(c["out"])
            entry[track] = {
                "agreement": round(agree / len(c["out"]), 4),
                "identical_greedy_prefix": first if first is not None else len(c["out"]),
                "mismatches": mism,
                "max_abs_logit_diff_mean": round(float(np.mean(diffs)), 3),
            }
        report["cases"].append(entry)
        print(json.dumps({k: entry[k] for k in ("name", "original", "quantized")}, ensure_ascii=False), flush=True)
    agg = {t: round(a / max(n, 1), 4) for t, (a, n) in totals.items()}
    report["agreement"] = agg
    report["criteria"] = {
        "export_correctness": {"rule": "quantized >= 0.97 and every mismatch has margin < 1.0", "pass": agg["quantized"] >= 0.97 and near_tie_only},
        "fidelity_to_original": {"rule": "original >= 0.90", "pass": agg["original"] >= 0.90},
    }
    report["pass"] = all(v["pass"] for v in report["criteria"].values())
    json.dump(report, open(args.report, "w"), indent=2, ensure_ascii=False)
    print("RESULT", json.dumps({"agreement": agg, "pass": report["pass"]}), flush=True)


if __name__ == "__main__":
    main()
