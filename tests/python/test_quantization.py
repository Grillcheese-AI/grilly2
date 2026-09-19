"""grilly.infer.quantization: int8 weight-only, against torchao's math.

The reference is torchao's formula (version 0.18: Int8Tensor.from_hp,
choose_qparams_affine, quantize_affine and the weight-only linear) written
out in torch below. The transcription was checked once against torchao 0.18's
own code: 0 of 10,019,412 int8 values and 0 scales differed, over six weight
shapes up to 4864x896.

Tolerances:
- int8 values, scales and dequantized weights: bit-exact. Both divisions are
  correctly rounded on the GPU, as on the CPU.
- F.linear: against a float64 evaluation of ``(x @ q.T) * scale + bias``
  within the bound the kernel's summation implies, as in
  crates/grilly-infer/tests/quant.rs:
  (16 * ceil(ceil(k / 16) / 64) + 8) * 2^-24 * (scale * sum|x * q| + |bias|).
- A quantized model against torch running torchao's weight-only math on the
  same int8 weights: rtol 1e-5, atol 1e-6, as for single forward passes in
  test_nn.py. That is case (b) in CLAUDE.md §9: torch's embedding,
  layer_norm, GELU and matmuls are other implementations.
- Gradients against torch: rtol 1e-5, atol 1e-6, for the same reason.
"""

import gc
import math

import numpy as np
import pytest
import torch

import grilly
import grilly.nn as gnn
from grilly.infer.quantization import Int8Tensor, Int8WeightOnlyConfig, PerRow, quantize_

RNG = np.random.default_rng(3)


def torchao_quantize(w: torch.Tensor):
    """torchao 0.18's per-row symmetric int8 for a float32 weight: (q, scale)."""
    amax = torch.maximum(-torch.clamp(w.amin(dim=1), max=0), torch.clamp(w.amax(dim=1), min=0))
    scale = torch.clamp(amax / 127.5, min=torch.finfo(torch.float32).eps)
    q = torch.clamp(torch.round(w * (1.0 / scale[:, None])), -128, 127).to(torch.int8)
    return q, scale


def qdata(t: Int8Tensor) -> np.ndarray:
    return np.frombuffer(t._q.qdata(), dtype=np.int8).reshape(tuple(t.shape))


def weights(n, k):
    w = (RNG.standard_normal((n, k)) * RNG.uniform(0.01, 5, (n, 1))).astype(np.float32)
    w[0] = 0.0  # an all-zero row: the scale is float32 eps
    return w


@pytest.mark.parametrize("n,k", [(1, 1), (3, 5), (33, 17), (64, 896), (40, 4864)])
def test_quantized_values_are_torchaos(n, k):
    w = weights(n, k)
    q, scale = torchao_quantize(torch.from_numpy(w))
    t = Int8Tensor.from_hp(grilly.from_numpy(w))
    np.testing.assert_array_equal(qdata(t), q.numpy())
    np.testing.assert_array_equal(t.scale.numpy().view(np.uint32), scale.numpy().view(np.uint32))
    deq = (q.float() * scale[:, None]).numpy()
    np.testing.assert_array_equal(t.dequantize().numpy().view(np.uint32), deq.view(np.uint32))


@pytest.mark.parametrize("m,n,k", [(1, 5, 17), (8, 7, 33), (9, 6, 100), (1, 64, 896), (20, 13, 896), (1, 16, 4864)])
@pytest.mark.parametrize("with_bias", [False, True], ids=["nobias", "bias"])
def test_linear_is_within_its_float64_bound(m, n, k, with_bias):
    w, x = weights(n, k), RNG.uniform(-2, 2, (m, k)).astype(np.float32)
    b = RNG.uniform(-0.5, 0.5, n).astype(np.float32) if with_bias else None
    t = Int8Tensor.from_hp(grilly.from_numpy(w))
    q, scale = (a.numpy().astype(np.float64) for a in torchao_quantize(torch.from_numpy(w)))
    with grilly.no_grad():
        y = gnn.functional.linear(grilly.from_numpy(x), t, None if b is None else grilly.from_numpy(b)).numpy()
    bias = np.zeros(n) if b is None else b.astype(np.float64)
    products = x.astype(np.float64)[:, None, :] * q[None, :, :]
    want = products.sum(-1) * scale + bias
    terms = 16 * math.ceil(math.ceil(k / 16) / 64)
    bound = (terms + 8) * 2.0**-24 * (np.abs(products).sum(-1) * scale + np.abs(bias)) + np.finfo(np.float64).tiny
    ratio = np.abs(y - want) / bound
    print(f"({m}, {n}, {k}) worst error / bound = {ratio.max():.3f}")
    assert (ratio <= 1).all()


def make(nn):
    return nn.Sequential(nn.Embedding(50, 16), nn.LayerNorm(16), nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 50))


def test_a_quantized_model_matches_torchaos_math_in_torch():
    torch.manual_seed(0)
    ref = make(torch.nn)
    model = make(gnn)
    model.load_state_dict({k: grilly.from_numpy(v.numpy()) for k, v in ref.state_dict().items()})
    quantize_(model, Int8WeightOnlyConfig())
    assert isinstance(model[2].weight, Int8Tensor) and isinstance(model[4].weight, Int8Tensor)
    assert not isinstance(model[0].weight, Int8Tensor)  # only nn.Linear, as torchao's default filter

    def ao_linear(x, layer):
        q, scale = torchao_quantize(layer.weight.detach())
        return torch.mm(x.reshape(-1, x.shape[-1]), q.t().float()).mul(scale).reshape(*x.shape[:-1], -1) + layer.bias

    ids = [[1, 2, 3, 4], [5, 6, 7, 49]]
    with torch.no_grad(), grilly.no_grad():
        h = ref[1](ref[0](torch.tensor(ids)))
        want = ao_linear(ref[3](ao_linear(h, ref[2])), ref[4]).numpy()
        got = model(grilly.tensor(ids)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_gradients_reach_input_and_bias_not_the_weight():
    k, n = 24, 10
    w = weights(n, k)
    x, b, g = (RNG.standard_normal(s).astype(np.float32) for s in [(3, k), (n,), (3, n)])
    lin = gnn.Linear(k, n)
    lin.load_state_dict({"weight": grilly.from_numpy(w), "bias": grilly.from_numpy(b)})
    quantize_(lin, Int8WeightOnlyConfig())
    gx = grilly.tensor(x, requires_grad=True)
    lin(gx).backward(grilly.from_numpy(g))

    q, scale = torchao_quantize(torch.from_numpy(w))
    tx, tb = torch.tensor(x, requires_grad=True), torch.tensor(b, requires_grad=True)
    (torch.mm(tx, q.t().float()) * scale + tb).backward(torch.from_numpy(g))
    np.testing.assert_allclose(gx.grad.numpy(), tx.grad.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(lin.bias.grad.numpy(), tb.grad.numpy(), rtol=1e-5, atol=1e-6)
    assert lin.weight.grad is None and not lin.weight.requires_grad
    with pytest.raises(RuntimeError, match="frozen"):
        lin.weight.requires_grad_(True)


def test_quantize_api_as_torchaos():
    model = gnn.Sequential(gnn.Linear(8, 8), gnn.Linear(8, 4), gnn.LayerNorm(4))
    quantize_(model, Int8WeightOnlyConfig(), filter_fn=lambda m, name: name == "1")
    assert not isinstance(model[0].weight, Int8Tensor) and isinstance(model[1].weight, Int8Tensor)
    first = model[1].weight
    quantize_(model, Int8WeightOnlyConfig())  # the default filter skips weights already quantized
    assert model[1].weight is first and isinstance(model[0].weight, Int8Tensor)
    assert tuple(first.shape) == (4, 8) and first.dtype == grilly.float32 and first.numel() == 32
    assert "Int8Tensor(shape=(4, 8)" in repr(first)
    assert sum(p.numel() for p in model.parameters()) == 8 * 8 + 8 + 4 * 8 + 4 + 4 + 4

    with pytest.raises(AssertionError, match="group_size"):
        Int8WeightOnlyConfig(group_size=32)
    with pytest.raises(ValueError, match="version 1"):
        Int8WeightOnlyConfig(version=1)
    with pytest.raises(NotImplementedError, match="granularity"):
        Int8WeightOnlyConfig(granularity=PerRow(dim=0))
    with pytest.raises(NotImplementedError, match="quantize_ with"):
        quantize_(model, object())
    with pytest.raises(NotImplementedError, match="device 'cuda'"):
        quantize_(model, Int8WeightOnlyConfig(), device="cuda")
    with pytest.raises(TypeError, match="from_hp"):
        Int8Tensor(grilly.zeros(2, 2))


def test_quantized_weights_are_int8_in_persistent_memory():
    def persistent():
        gc.collect()
        return grilly._C.submit_stats()["persistent_bytes"]

    before = persistent()
    lin = gnn.Linear(900, 70)  # in_features pads to 912 bytes per row
    assert persistent() - before == 4 * (900 * 70 + 70)
    quantize_(lin, Int8WeightOnlyConfig())
    assert lin.weight.nbytes == 70 * 912 + 70 * 4
    assert persistent() - before == lin.weight.nbytes + 4 * 70  # the float weight is freed; the bias stays
