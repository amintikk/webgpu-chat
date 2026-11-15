"""Gemma 4 12B decoder: each GroupQueryAttention becomes If(use_bias): a masked attention when a picture is read
(the caller passes the mask: causal, plus both directions inside each picture, plus the sliding window on local
layers), or the original fused GQA otherwise (every other step, decoding included). Weights are untouched."""
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
import numpy as np

def const(name, arr):
    return h.make_node('Constant', [], [name], value=nh.from_array(np.array(arr, dtype=np.int64), name))

def attn_then(n, num_heads, kv, scale, bias, tag):
    q, k, v, pk_in, pv_in = list(n.input)[:5]
    g = num_heads // kv
    p = lambda s: f'{tag}/{s}'
    nodes = [
        const(p('shq'), [0, 0, num_heads, -1]), const(p('shkv'), [0, 0, kv, -1]),
        h.make_node('Reshape', [q, p('shq')], [p('q4')]),
        h.make_node('Transpose', [p('q4')], [p('qt')], perm=[0, 2, 1, 3]),            # B,16,S,H
        h.make_node('Reshape', [k, p('shkv')], [p('k4')]),
        h.make_node('Transpose', [p('k4')], [p('kt4')], perm=[0, 2, 1, 3]),           # B,kv,S,H
        h.make_node('Reshape', [v, p('shkv')], [p('v4')]),
        h.make_node('Transpose', [p('v4')], [p('vt4')], perm=[0, 2, 1, 3]),
        h.make_node('Concat', [pk_in, p('kt4')], [p('present_k')], axis=2),            # B,kv,T,H
        h.make_node('Concat', [pv_in, p('vt4')], [p('present_v')], axis=2),
        const(p('shg'), [0, kv, -1, 0]),
        h.make_node('Reshape', [p('qt'), p('shg')], [p('qg')]),                        # B,kv,g*S,H (head h -> kv head h//g)
        h.make_node('Transpose', [p('present_k')], [p('kT')], perm=[0, 1, 3, 2]),      # B,kv,H,T
        # Scores and softmax in fp32, like the fused kernel (fp16 logits lose too much on peaky rows).
        h.make_node('Cast', [p('qg')], [p('qg32')], to=T.FLOAT),
        h.make_node('Cast', [p('kT')], [p('kT32')], to=T.FLOAT),
        h.make_node('MatMul', [p('qg32'), p('kT32')], [p('sc0')]),                      # B,kv,g*S,T
    ]
    sc = p('sc0')
    if abs(scale - 1.0) > 1e-9 and scale != 0:
        nodes += [h.make_node('Constant', [], [p('scale')], value=nh.from_array(np.array(scale, dtype=np.float32), p('scale'))),
                  h.make_node('Mul', [sc, p('scale')], [p('sc1')])]
        sc = p('sc1')
    nodes += [
        h.make_node('Shape', [bias], [p('bshape')]),
        const(p('two'), [2]), const(p('four'), [4]), const(p('m1'), [-1]),
        h.make_node('Slice', [p('bshape'), p('two'), p('four')], [p('st')]),            # [S, T]
        h.make_node('Concat', [p('m1'), p('st')], [p('nst')], axis=0),                   # [-1, S, T]
        h.make_node('Reshape', [sc, p('nst')], [p('scn')]),                              # kv*g, S, T
        h.make_node('Cast', [bias], [p('bias32')], to=T.FLOAT),
        h.make_node('Add', [p('scn'), p('bias32')], [p('scb')]),                          # 1, kv*g, S, T
        h.make_node('Softmax', [p('scb')], [p('pr32')], axis=-1),
        h.make_node('Cast', [p('pr32')], [p('pr')], to=T.FLOAT16),
        h.make_node('Shape', [sc], [p('scshape')]),
        h.make_node('Reshape', [p('pr'), p('scshape')], [p('pr4')]),                     # B,kv,g*S,T
        h.make_node('MatMul', [p('pr4'), p('present_v')], [p('o')]),                     # B,kv,g*S,H
        const(p('sho'), [0, num_heads, -1, 0]),
        h.make_node('Reshape', [p('o'), p('sho')], [p('o4')]),                           # B,16,S,H
        h.make_node('Transpose', [p('o4')], [p('ot')], perm=[0, 2, 1, 3]),               # B,S,16,H
        const(p('shout'), [0, 0, -1]),
        h.make_node('Reshape', [p('ot'), p('shout')], [p('out')]),
    ]
    outs = [h.make_tensor_value_info(p(x), T.FLOAT16, None) for x in ('out', 'present_k', 'present_v')]
    return h.make_graph(nodes, tag + '_then', [], outs)

def transform(model, bias_global='attention_bias_global', bias_local='attention_bias_local', cond='use_bias'):
    g = model.graph
    new_nodes = []
    count = 0
    for n in g.node:
        if n.op_type != 'GroupQueryAttention':
            new_nodes.append(n)
            continue
        a = {x.name: h.get_attribute_value(x) for x in n.attribute}
        num_heads, kv = a['num_heads'], a['kv_num_heads']
        local = a.get('local_window_size', -1) != -1
        scale = a.get('scale', 0.0)
        tag = f'bidir{count}'
        then_g = attn_then(n, num_heads, kv, scale, bias_local if local else bias_global, tag)
        e_outs = [f'{tag}/e_{i}' for i in range(len(n.output))]
        gqa = h.make_node('GroupQueryAttention', list(n.input), e_outs, name=n.name, domain=n.domain)
        gqa.attribute.extend(n.attribute)
        else_g = h.make_graph([gqa], tag + '_else', [], [h.make_tensor_value_info(o, T.FLOAT16, None) for o in e_outs])
        new_nodes.append(h.make_node('If', [cond], list(n.output), name=n.name + '/if', then_branch=then_g, else_branch=else_g))
        count += 1
    del g.node[:]
    g.node.extend(new_nodes)
    g.input.extend([
        h.make_tensor_value_info(cond, T.BOOL, []),
        h.make_tensor_value_info(bias_global, T.FLOAT16, [1, 1, 'bias_q', 'bias_kv']),
        h.make_tensor_value_info(bias_local, T.FLOAT16, [1, 1, 'bias_q', 'bias_kv']),
    ])
    return count

if __name__ == '__main__':
    import sys
    m = onnx.load(sys.argv[1], load_external_data=False)
    n = transform(m)
    onnx.save(m, sys.argv[2])
    print('replaced', n, 'attention nodes')
