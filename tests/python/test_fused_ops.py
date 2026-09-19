"""The decoder step's fused kernels, against the ops they replace.

Tolerance (CLAUDE.md §9 case (a): grilly against grilly, same kernels, and
for these three the same arithmetic in the same order):
- ``_C.swiglu`` vs ``F.silu(gate) * up``: **bit-exact**. The kernel calls
  the same sigmoid and multiply functions, in that order.
- ``_C.rope`` vs ``x * cos + rotate_half(x) * sin``: **bit-exact**. Two
  multiplies and an add on the same float32 values; the shader is
  ``precise``, so no fused multiply-add reassociates it.
- a fused int8 q/k/v weight vs the three separate ones: **bit-exact**. The
  scales are per row, so no row's dot product changes.

``_C.rms_norm`` is the one op of the step whose arithmetic differs, so it
is measured against float64 under a bound derived from what it does:
a compensated sum of squares (~1 ulp), the mean, ``+ eps``, a square root,
a reciprocal and two multiplies (~0.5 ulp each) — **4 ulp**. The composed
``x * rsqrt(mean(x^2) + eps) * w`` is held to the same bound, and both
print their worst measured ratio to it.

Whole logits are *not* compared between the two paths against a fixed
epsilon: the norm's ~1 ulp difference is amplified by the network's own
conditioning (measured: 8e-8 at the first norm becomes 1e-4 at the logits
of a random 3-layer model), so any such number would be a property of the
weights, not of the kernels. Both paths are compared against a float64
reference of the same model instead, and the fused path must be no worse
than the composed one — a ratio, as §9 asks for. Greedy token ids must be
identical, which is the property a user sees.
"""

import numpy as np
import pytest

import grilly as torch
from grilly import _C, nn
from grilly.infer.models.llama import LlamaConfig, LlamaForCausalLM, _apply_rotary
from grilly.infer.quantization import Int8Tensor, Int8WeightOnlyConfig, PerRow, quantize_
from grilly.nn import functional as F

RMS_NORM_ULPS = 4.0  # the sum, mean, eps, sqrt, reciprocal and two multiplies


def rand(*shape, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).standard_normal(shape).astype(np.float32))


def wrap(inner):
    return torch.Tensor._wrap(inner)


def bits(t):
    return t.detach().numpy().view(np.uint32)


def worst_ulps(got, want):
    """The worst distance to the float64 value, in float32 ulps of it."""
    g = got.astype(np.float64)
    return float((np.abs(g - want) / np.spacing(np.abs(want).astype(np.float32)).astype(np.float64)).max())


@pytest.mark.parametrize("fused_source", [False, True])
def test_swiglu_is_silu_times_up_bit_for_bit(fused_source):
    rows, width = 5, 128
    if fused_source:  # one gate_up projection: gate in the first half, up in the second
        both = rand(rows, 2 * width, seed=1)
        got = wrap(_C.swiglu(both._t, 0, both._t, width, width))
        gate, up = both[:, :width], both[:, width:]
    else:
        gate, up = rand(rows, width, seed=2), rand(rows, width, seed=3)
        got = wrap(_C.swiglu(gate._t, 0, up._t, 0, width))
    assert got.shape == (rows, width)
    np.testing.assert_array_equal(bits(got), bits(F.silu(gate) * up))


@pytest.mark.parametrize("batch,t,heads,dim,col,extra,pos0", [
    (1, 1, 14, 64, 0, 256, 0),      # a decode step's q, from a fused q/k/v projection
    (1, 1, 2, 64, 896, 128, 23),    # the same projection's k
    (2, 7, 4, 32, 0, 0, 5),         # a prompt, several positions
])
def test_rope_is_the_composed_rotary_bit_for_bit(batch, t, heads, dim, col, extra, pos0):
    row = col + heads * dim + extra
    src = rand(batch, t, row, seed=col + t)
    tables = rand(pos0 + t + 3, dim, seed=7)
    cos, sin = tables.cos(), tables.sin()
    got = wrap(_C.rope(src._t, col, heads, dim, cos._t, sin._t, pos0))
    assert got.shape == (batch, heads, t, dim)

    x = src[..., col:col + heads * dim].reshape(batch, t, heads, dim).transpose(1, 2)
    want = _apply_rotary(x, cos.narrow(0, pos0, t), sin.narrow(0, pos0, t))
    np.testing.assert_array_equal(bits(got), bits(want))


def test_a_fused_int8_projection_equals_the_separate_ones():
    k, rows = 96, 4
    parts = [nn.Linear(k, n, bias=False) for n in (28, 8, 8)]
    for p in parts:
        quantize_(p, Int8WeightOnlyConfig())
    x = rand(rows, k, seed=11)
    fused = Int8Tensor._cat([p.weight for p in parts])
    assert fused.shape == (44, k)
    got, col = F.linear(x, fused), 0
    for p in parts:
        n = p.weight.shape[0]
        np.testing.assert_array_equal(bits(got[:, col:col + n]), bits(F.linear(x, p.weight)))
        col += n


def tiny(**kwargs):
    config = LlamaConfig(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6,
                         qkv_bias=True, tie_word_embeddings=True, **kwargs)
    model = LlamaForCausalLM(config)
    rng = np.random.default_rng(0)
    state = {}
    for name, p in model.state_dict().items():
        values = rng.standard_normal(tuple(p.shape)).astype(np.float32)
        state[name] = torch.from_numpy(values * 0.5 + 1.0 if "norm" in name else values * 0.08)
    model.load_state_dict(state, assign=True)
    model.tie_weights()  # a checkpoint of a tied model carries no lm_head.weight
    return model


def test_rms_norm_is_within_its_error_bound_of_float64():
    x = rand(3, 5, 64, seed=21) * 4.0
    weight = rand(64, seed=22)
    fused = wrap(_C.rms_norm(x._t, 64, weight._t, 1e-6)).numpy()
    composed = (x * (x.pow(2).mean((2,), keepdim=True) + 1e-6).rsqrt() * weight).detach().numpy()
    xd, wd = x.numpy().astype(np.float64), weight.numpy().astype(np.float64)
    want = xd / np.sqrt((xd ** 2).mean(-1, keepdims=True) + 1e-6) * wd
    for name, got in (("fused", fused), ("composed", composed)):
        worst = worst_ulps(got, want)
        print(f"rms_norm {name}: worst {worst:.2f} of the {RMS_NORM_ULPS} ulp bound")
        assert worst <= RMS_NORM_ULPS


def reference_logits(model, ids):
    """The same model in float64 numpy, written from transformers' Llama,
    independently of grilly's ops."""
    p = {k: v.detach().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    c = model.config
    heads, kv, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
    t = ids.shape[1]
    cos, sin = (v.numpy().astype(np.float64)[:t] for v in model._rotary.tables(t))

    def rms(x, w, eps=c.rms_norm_eps):
        return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * w

    def rotate(x):  # (heads, t, d)
        half = d // 2
        return x * cos + np.concatenate([-x[..., half:], x[..., :half]], -1) * sin

    x = p["model.embed_tokens.weight"][ids.numpy()[0]]
    mask = np.triu(np.full((t, t), -np.inf), 1)
    for i in range(c.num_hidden_layers):
        q = f"model.layers.{i}."
        y = rms(x, p[q + "input_layernorm.weight"])

        def proj(name, n, y=y, q=q):
            out = y @ p[q + name + ".weight"].T
            return (out + p[q + name + ".bias"] if q + name + ".bias" in p else out).reshape(t, n, d).transpose(1, 0, 2)

        qh, kh, vh = rotate(proj("self_attn.q_proj", heads)), rotate(proj("self_attn.k_proj", kv)), proj("self_attn.v_proj", kv)
        kh, vh = (np.repeat(h, heads // kv, axis=0) for h in (kh, vh))
        scores = qh @ kh.transpose(0, 2, 1) / np.sqrt(d) + mask
        scores = np.exp(scores - scores.max(-1, keepdims=True))
        attn = (scores / scores.sum(-1, keepdims=True)) @ vh
        x = x + attn.transpose(1, 0, 2).reshape(t, heads * d) @ p[q + "self_attn.o_proj.weight"].T

        y = rms(x, p[q + "post_attention_layernorm.weight"])
        g, u = y @ p[q + "mlp.gate_proj.weight"].T, y @ p[q + "mlp.up_proj.weight"].T
        x = x + (g / (1.0 + np.exp(-g)) * u) @ p[q + "mlp.down_proj.weight"].T
    return rms(x, p["model.norm.weight"]) @ p["lm_head.weight"].T


def both_paths(model, ids):
    composed = model(ids).logits.detach().numpy()[0]  # autograd on: plain ops
    with torch.no_grad():
        fused = model(ids).logits.numpy()[0]
    want = reference_logits(model, ids)
    scale = np.abs(want).max()
    errors = [float(np.abs(v.astype(np.float64) - want).max() / scale) for v in (fused, composed)]
    print(f"float64 reference: fused {errors[0]:.3e}, composed {errors[1]:.3e}")
    return fused, composed, errors


def test_the_fused_path_is_no_less_accurate_than_the_composed_one():
    model = tiny()
    ids = torch.from_numpy(np.array([[5, 17, 42, 3, 60]], dtype=np.int64))
    fused, composed, (fused_err, composed_err) = both_paths(model, ids)
    assert fused_err <= 2 * composed_err, "the fused kernels lost accuracy the composed ops keep"
    assert composed_err < 1e-4, "the composed path itself drifted: a reference or an op changed"
    assert (fused.argmax(-1) == composed.argmax(-1)).all()


def test_fused_and_composed_decode_generate_the_same_tokens():
    model = tiny()
    prompt = torch.from_numpy(np.array([[5, 17, 42]], dtype=np.int64))
    runs = []
    for _ in range(2):
        cache = model.make_cache(1, 32)
        ids, step = list(prompt.numpy()[0]), prompt
        for _ in range(12):
            logits = model(step, past_key_values=cache, logits_to_keep=1).logits
            nxt = int(logits.detach().numpy()[0, -1].argmax())
            ids.append(nxt)
            step = torch.from_numpy(np.array([[nxt]], dtype=np.int64))
        runs.append(ids)

    with torch.no_grad():  # the same, through the fused kernels and the cache
        cache = model.make_cache(1, 32)
        ids, step = list(prompt.numpy()[0]), prompt
        for _ in range(12):
            nxt = int(model(step, past_key_values=cache, logits_to_keep=1).logits.numpy()[0, -1].argmax())
            ids.append(nxt)
            step = torch.from_numpy(np.array([[nxt]], dtype=np.int64))
    assert runs[0] == runs[1], "the composed path is not deterministic"
    assert ids == runs[0], f"fused {ids} != composed {runs[0]}"


def test_quantized_fused_and_separate_projections_agree():
    """Once quantized, the fused path runs one int8 linear for q/k/v and one
    for gate/up. Those are bit-exact per row, so the logits differ only by
    the norms, as in the float32 case."""
    model = tiny()
    quantize_(model, Int8WeightOnlyConfig(granularity=PerRow()))
    ids = torch.from_numpy(np.array([[5, 17, 42, 3]], dtype=np.int64))
    with torch.no_grad():
        first = model(ids).logits.numpy()
        again = model(ids).logits.numpy()  # the fused weights are built once and reused
    np.testing.assert_array_equal(first.view(np.uint32), again.view(np.uint32))
    fused, composed, (fused_err, composed_err) = both_paths(model, ids)
    assert fused_err <= 2 * composed_err
    assert (fused.argmax(-1) == composed.argmax(-1)).all()

