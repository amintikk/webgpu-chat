"""Shared pieces of the Gemma 4 12B export and its verification.

Weights are read tensor by tensor from the official safetensors file on Hugging Face with HTTP
range requests, so the 24 GB checkpoint never has to be stored on disk or held in memory.
"""

from __future__ import annotations

import json
import struct
import time

import numpy as np

REPO = "google/gemma-4-12B-it"
REVISION = "43c9d014b231474fa7212dbb48a762566c2b255d"
PREFIX = "model.language_model."
BLOCK = 32  # 4-bit block size along the input dimension


class RemoteSafetensors:
    """Random access to the tensors of a remote .safetensors file."""

    def __init__(self, repo: str = REPO, revision: str = REVISION, filename: str = "model.safetensors"):
        self.url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        n = struct.unpack("<Q", self._read(0, 8))[0]
        self.header = json.loads(self._read(8, n))
        self.base = 8 + n
        self.header.pop("__metadata__", None)

    def keys(self):
        return self.header.keys()

    def _read(self, start: int, length: int) -> bytes:
        import urllib.request

        for attempt in range(8):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + length - 1}", "User-Agent": "webgpu-chat-export"})
                out = bytearray()
                with urllib.request.urlopen(req, timeout=120) as r:
                    while chunk := r.read(16 << 20):
                        out += chunk
                if len(out) != length:
                    raise IOError(f"short read {len(out)} != {length}")
                return bytes(out)
            except Exception as e:  # network hiccups: retry with backoff
                if attempt == 7:
                    raise
                print(f"  read retry {attempt + 1}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        raise RuntimeError("unreachable")

    def get(self, name: str, rows: slice | None = None) -> np.ndarray:
        """Tensor as float32. `rows` reads only a slice of the first dimension."""
        meta = self.header[name]
        assert meta["dtype"] == "BF16", meta["dtype"]
        shape = list(meta["shape"])
        start, end = meta["data_offsets"]
        row_bytes = (end - start) // shape[0] if shape else end - start
        if rows is not None:
            r0, r1 = rows.start or 0, min(rows.stop, shape[0])
            start, end = start + r0 * row_bytes, start + r1 * row_bytes
            shape[0] = r1 - r0
        raw = np.frombuffer(self._read(self.base + start, end - start), dtype=np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def quantize_q4(w: np.ndarray):
    """Asymmetric 4-bit round-to-nearest per block of 32 along the last axis (the MatMulNBits layout).

    Returns packed weights [N, K/32, 16] uint8, fp16 scales [N, K/32], packed zero points [N, K/64] uint8,
    and the dequantized weights that those values represent (float32, same shape as `w`).
    """
    n, k = w.shape
    nb = k // BLOCK
    wb = w.reshape(n, nb, BLOCK).astype(np.float32)
    mn = np.minimum(wb.min(-1), 0.0)
    mx = np.maximum(wb.max(-1), 0.0)
    scale = ((mx - mn) / 15.0).astype(np.float16)
    scale[scale == 0] = np.float16(1.0)
    s32 = scale.astype(np.float32)
    zp = np.clip(np.round(-mn / s32), 0, 15).astype(np.uint8)
    q = np.clip(np.round(wb / s32[..., None]) + zp[..., None], 0, 15).astype(np.uint8)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)
    zp_packed = (zp[:, 0::2] | (zp[:, 1::2] << 4)).astype(np.uint8)
    deq = ((q.astype(np.float32) - zp[..., None].astype(np.float32)) * s32[..., None]).reshape(n, k)
    return packed, scale, zp_packed, deq


def text_config():
    """The text part of the Hugging Face config, as the official modeling code sees it."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(REPO, revision=REVISION)
    return cfg.text_config


def inv_freqs(tc) -> dict[str, np.ndarray]:
    """Inverse RoPE frequencies per layer type, computed by the official rotary module."""
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextRotaryEmbedding

    rot = Gemma4UnifiedTextRotaryEmbedding(tc)
    return {t: getattr(rot, f"{t}_inv_freq").numpy().astype(np.float32) for t in set(tc.layer_types)}
