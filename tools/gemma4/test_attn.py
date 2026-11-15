"""Masked branch vs fused GQA on the CPU: with a causal (+window) mask they must agree; with a picture block the
masked branch must let tokens of the block see later tokens of the block."""
import numpy as np, onnx, onnxruntime as ort
from onnx import helper as h, TensorProto as T
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import bidir
NEG = np.float16(-65504)

def model(num_heads, kv, H, window):
    gqa = h.make_node('GroupQueryAttention', ['q', 'k', 'v', 'pk', 'pv', 'seqlens', 'total'], ['out', 'presk', 'presv'], domain='com.microsoft', name='gqa',
                      num_heads=num_heads, kv_num_heads=kv, local_window_size=window, scale=1.0, do_rotary=0)
    ins = [h.make_tensor_value_info('q', T.FLOAT16, [1, 'S', num_heads * H]), h.make_tensor_value_info('k', T.FLOAT16, [1, 'S', kv * H]),
           h.make_tensor_value_info('v', T.FLOAT16, [1, 'S', kv * H]), h.make_tensor_value_info('pk', T.FLOAT16, [1, kv, 'P', H]),
           h.make_tensor_value_info('pv', T.FLOAT16, [1, kv, 'P', H]), h.make_tensor_value_info('seqlens', T.INT32, [1]), h.make_tensor_value_info('total', T.INT32, [])]
    outs = [h.make_tensor_value_info(o, T.FLOAT16, None) for o in ('out', 'presk', 'presv')]
    m = h.make_model(h.make_graph([gqa], 't', ins, outs), opset_imports=[h.make_opsetid('', 21), h.make_opsetid('com.microsoft', 1)])
    m.ir_version = 10
    bidir.transform(m)
    return ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])

def bias(P, S, window, blocks):
    Tn = P + S
    b = np.full((1, 1, S, Tn), NEG, np.float16)
    for i in range(S):
        qi = P + i
        for j in range(Tn):
            ok = j <= qi and (window < 0 or qi - j < window)
            if blocks is not None and blocks[qi] >= 0 and blocks[qi] == blocks[j]:
                ok = True
            if ok: b[0, 0, i, j] = 0
    return b

def exact(feeds, num_heads, kv, H, maskbias):
    q = feeds['q'].astype(np.float64).reshape(1, -1, num_heads, H).transpose(0, 2, 1, 3)
    K = np.concatenate([feeds['pk'], feeds['k'].reshape(1, -1, kv, H).transpose(0, 2, 1, 3)], 2).astype(np.float64)
    V = np.concatenate([feeds['pv'], feeds['v'].reshape(1, -1, kv, H).transpose(0, 2, 1, 3)], 2).astype(np.float64)
    g = num_heads // kv
    K = np.repeat(K, g, axis=1); V = np.repeat(V, g, axis=1)
    sc = q @ K.transpose(0, 1, 3, 2) + np.where(maskbias[0, 0] < -1, -1e30, 0)
    sc -= sc.max(-1, keepdims=True); pr = np.exp(sc); pr /= pr.sum(-1, keepdims=True)
    return (pr @ V).transpose(0, 2, 1, 3).reshape(1, q.shape[2], num_heads * H)

if __name__ == '__main__':
  rng = np.random.default_rng(0)
  for amp in (1.0, 0.3):
    print('input scale', amp, '(logit spread ~', round(16 * amp * amp, 1), ')')
    for num_heads, kv, H, window in [(16, 8, 256, 1024), (16, 1, 512, -1)]:
      s = model(num_heads, kv, H, window)
      for P, S in [(0, 7), (5, 6), (9, 1)]:
        f = lambda *sh: (rng.standard_normal(sh) * amp).astype(np.float16)
        feeds = {'q': f(1, S, num_heads * H), 'k': f(1, S, kv * H), 'v': f(1, S, kv * H), 'pk': f(1, kv, P, H), 'pv': f(1, kv, P, H),
                 'seqlens': np.array([P + S - 1], np.int32), 'total': np.array(P + S, np.int32)}
        d = np.zeros((1, 1, 1, 1), np.float16)
        b = bias(P, S, window, None)
        ref = s.run(None, {**feeds, 'use_bias': np.array(False), 'attention_bias_global': d, 'attention_bias_local': d})[0].astype(np.float64)
        got = s.run(None, {**feeds, 'use_bias': np.array(True), 'attention_bias_global': b, 'attention_bias_local': b})[0].astype(np.float64)
        ex = exact(feeds, num_heads, kv, H, b)
        print(f'  kv={kv} past={P} new={S}: error fused GQA {np.abs(ref-ex).max():.4f} | masked branch {np.abs(got-ex).max():.4f}')
