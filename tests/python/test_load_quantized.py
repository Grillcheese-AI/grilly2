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
    """Equal-sized linears, so the bound on the peak is tight: holding one
    float32 tensor is a small fraction of holding them all, which is what
    the memory assertions below turn on."""

    def __init__(self, width=512, depth=16):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=(i % 2 == 0)) for i in range(depth)])
        self.norm = nn.LayerNorm(width)


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
    before_streaming, _ = persistent()
    load_quantized_(streamed, path, Int8WeightOnlyConfig())
    held, streaming_peak = persistent()
    streaming_peak -= before_streaming  # over what was already alive, the model included

    want, got = q_bits(reference), q_bits(streamed)
    assert set(want) == set(got) != set()
    for name in want:
        np.testing.assert_array_equal(got[name][0], want[name][0], f"{name}: int8 values")
        np.testing.assert_array_equal(got[name][1], want[name][1], f"{name}: scales")

    quantized = sum(m.weight.nbytes for m in streamed.modules() if isinstance(getattr(m, "weight", None), Int8Tensor))
    tensors = load_file(path)
    biggest = max(v.numel() * 4 for v in tensors.values())
    whole = sum(v.numel() * 4 for v in tensors.values())
    del tensors
    print(f"streaming rose {streaming_peak / 2**20:.1f} MiB above what was alive, two-step "
          f"{two_step_peak / 2**20:.1f} MiB; the checkpoint is {whole / 2**20:.1f} MiB, its largest tensor "
          f"{biggest / 2**20:.1f} MiB, quantized {quantized / 2**20:.1f} MiB")
    # Each weight is replaced by a quarter of its size, so a streaming load's
    # alive bytes *fall*: the peak can only rise by about one float32 tensor.
    # Keeping anything per weight — a state_dict held across the loop, say —
    # leaves the old float32 weights alive and shows up here immediately.
    assert streaming_peak <= 2 * biggest, "the load is holding on to weights it has replaced"
    assert two_step_peak >= whole, "the two-step path should hold the whole checkpoint at once"
    assert streaming_peak < two_step_peak / 4


def test_streaming_reports_missing_and_unexpected_keys(tmp_path):
    model = Small()
    path = checkpoint(tmp_path, model)
    state = {k: torch.zeros(tuple(v.shape)) for k, v in model.state_dict().items()}
    state.pop("layers.0.weight")
    state["layers.99.weight"] = torch.zeros(4, 4)  # no such layer
    odd = tmp_path / "odd.safetensors"
    save_file(state, str(odd))

    with pytest.raises(RuntimeError, match='Missing key.*"layers.0.weight"'):
        load_quantized_(Small(), odd, Int8WeightOnlyConfig())
    with pytest.raises(RuntimeError, match='Unexpected key.*"layers.99.weight"'):
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
    load_quantized_(model, path, Int8WeightOnlyConfig(), filter_fn=lambda m, name: name == "layers.3")
    assert isinstance(model.layers[3].weight, Int8Tensor)
    assert not isinstance(model.layers[0].weight, Int8Tensor)
    np.testing.assert_array_equal(model.layers[0].weight.detach().numpy().view(np.uint32),
                                  load_file(path)["layers.0.weight"].numpy().view(np.uint32))


def test_keep_vars_does_not_dequantize_the_model(tmp_path):
    """``state_dict()`` returns dense tensors, so on a quantized model it
    dequantizes every weight — the whole model in float32, at once. Loading
    code must ask for ``keep_vars=True`` instead. Dequantizing is a kernel,
    so the dispatch counter says whether it happened."""
    model = Small()
    load_quantized_(model, checkpoint(tmp_path, model), Int8WeightOnlyConfig())

    before = grilly._C.submit_stats()["dispatches"]
    keys = list(model.state_dict(keep_vars=True))
    assert grilly._C.submit_stats()["dispatches"] == before, "keep_vars dequantized something"
    assert "layers.0.weight" in keys

    dense = model.state_dict()
    assert grilly._C.submit_stats()["dispatches"] > before, "state_dict() should give dense tensors"
    assert dense["layers.0.weight"].dtype is grilly.float32
