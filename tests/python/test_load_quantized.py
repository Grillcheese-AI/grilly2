"""``load_quantized_``: a safetensors checkpoint quantized as it loads.

Tolerance: **bit-exact**. Quantizing a weight after loading it and
quantizing it while loading it run the same kernel on the same float32
values, so every int8 value and every scale must match bit for bit
(CLAUDE.md §9 case (a)). The point of the exercise is memory, not numbers,
so the peak is asserted too, from the shim's own high-water mark of
persistent bytes rather than from a claim in a comment.
"""

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import grilly
from grilly import nn
from grilly.infer.quantization import Int8Tensor, Int8WeightOnlyConfig, load_quantized_, quantize_
from grilly.infer.safetensors import load_file


class Small(nn.Module):
    """Linears big enough that the float32 copies dominate the peak."""

    def __init__(self, width=512, depth=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=(i % 2 == 0)) for i in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 3 * width, bias=False)


def checkpoint(tmp_path, model, seed=0):
    """Write the model's shapes as a safetensors file of random weights."""
    rng = np.random.default_rng(seed)
    state = {k: torch.from_numpy(rng.standard_normal(tuple(v.shape)).astype(np.float32) * 0.05)
             for k, v in model.state_dict().items()}
    path = tmp_path / "model.safetensors"
    save_file(state, str(path))
    return path


def q_bits(model):
    """Every quantized weight's int8 values and scales, as bits."""
    out = {}
    for name, module in model.named_modules():
        if isinstance(getattr(module, "weight", None), Int8Tensor):
            w = module.weight
            out[name] = (np.frombuffer(w._q.qdata(), dtype=np.int8), w.scale.numpy().view(np.uint32).copy())
    return out


def persistent():
    s = grilly._C.submit_stats()
    return s["persistent_bytes"], s["persistent_peak"]


def test_streaming_gives_the_same_int8_as_loading_then_quantizing(tmp_path):
    reference, streamed = Small(), Small()
    path = checkpoint(tmp_path, reference)

    grilly._C.reset_persistent_peak()
    base, _ = persistent()
    state = load_file(path)
    reference.load_state_dict(state, assign=True)
    quantize_(reference, Int8WeightOnlyConfig())
    two_step_peak = persistent()[1] - base  # the float32 dict is still alive: this is the peak
    del state

    grilly._C.reset_persistent_peak()
    after_two_step, _ = persistent()
    load_quantized_(streamed, path, Int8WeightOnlyConfig())
    held, streaming_peak = persistent()
    streaming_peak -= after_two_step

    want, got = q_bits(reference), q_bits(streamed)
    assert set(want) == set(got) != set()
    for name in want:
        np.testing.assert_array_equal(got[name][0], want[name][0], f"{name}: int8 values")
        np.testing.assert_array_equal(got[name][1], want[name][1], f"{name}: scales")

    quantized = sum(m.weight.nbytes for m in streamed.modules() if isinstance(getattr(m, "weight", None), Int8Tensor))
    biggest = max(v.numel() * 4 for v in load_file(path).values())
    print(f"peak over the quantized weights: two-step {two_step_peak / 2**20:.1f} MiB, "
          f"streaming {streaming_peak / 2**20:.1f} MiB, one float32 tensor is {biggest / 2**20:.1f} MiB")
    # streaming holds the int8 weights plus at most one float32 tensor;
    # the two-step path holds every float32 weight as well
    assert streaming_peak <= quantized + 2 * biggest
    assert two_step_peak > streaming_peak + biggest


def test_streaming_reports_missing_and_unexpected_keys(tmp_path):
    model = Small()
    path = checkpoint(tmp_path, model)
    state = {k: torch.zeros(tuple(v.shape)) for k, v in model.state_dict().items()}
    state.pop("layers.0.weight")
    state["layers.9.weight"] = torch.zeros(4, 4)
    odd = tmp_path / "odd.safetensors"
    save_file(state, str(odd))

    with pytest.raises(RuntimeError, match='Missing key.*"layers.0.weight"'):
        load_quantized_(Small(), odd, Int8WeightOnlyConfig())
    with pytest.raises(RuntimeError, match='Unexpected key.*"layers.9.weight"'):
        load_quantized_(Small(), odd, Int8WeightOnlyConfig())
    load_quantized_(Small(), odd, Int8WeightOnlyConfig(), strict=False)  # both tolerated

    with pytest.raises(RuntimeError, match="size mismatch for layers.1.weight"):
        wrong = {k: v for k, v in state.items()}
        wrong["layers.1.weight"] = torch.zeros(8, 8)
        bad = tmp_path / "bad.safetensors"
        save_file(wrong, str(bad))
        load_quantized_(Small(), bad, Int8WeightOnlyConfig(), strict=False)


def test_streaming_leaves_unquantized_weights_alone(tmp_path):
    """Only what ``filter_fn`` picks is quantized; the rest loads as float32,
    bit for bit."""
    model = Small()
    path = checkpoint(tmp_path, model)
    load_quantized_(model, path, Int8WeightOnlyConfig())
    want = load_file(path)
    assert isinstance(model.layers[0].weight, Int8Tensor)
    for name in ("norm.weight", "norm.bias", "layers.0.bias"):
        got = dict(model.named_parameters())[name].detach().numpy()
        np.testing.assert_array_equal(got.view(np.uint32), want[name].numpy().view(np.uint32), name)


def test_streaming_with_a_filter(tmp_path):
    model = Small()
    path = checkpoint(tmp_path, model)
    load_quantized_(model, path, Int8WeightOnlyConfig(), filter_fn=lambda m, name: name == "head")
    assert isinstance(model.head.weight, Int8Tensor)
    assert not isinstance(model.layers[0].weight, Int8Tensor)
    np.testing.assert_array_equal(model.layers[0].weight.detach().numpy().view(np.uint32),
                                  load_file(path)["layers.0.weight"].numpy().view(np.uint32))
