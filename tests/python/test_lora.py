"""LoRA factors beside a frozen base, and the peft files that fill them.

Reference: **the definition** — ``y = W x + (alpha/r) B (A x)`` computed
with plain grilly ops — under case (a)'s composed-chain bound, because the
wrapper and the definition are the same ops on the same backend in a
different grouping.

Three things here are not about arithmetic and matter more:

- **nothing is merged.** The base weight must be the same object and the
  same bytes after wrapping, because evolution strategies perturb the
  factors and a merged adapter has nothing left to perturb;
- **the attention fusion must not eat it.** q/k/v are concatenated into one
  int8 weight when they can be; doing that to a wrapped module would run
  the base and silently drop the adapter. That is a wrong *answer*, not a
  crash, so it gets its own test;
- **an adapter for a different base is refused**, at load, rather than
  producing a plausible wrong model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import grilly
import grilly.nn as nn
from grilly.infer.lora import LoRALinear, apply_lora_, lora_parameters


def rand(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def host(t):
    """A tensor's values. Parameters carry ``requires_grad``, and so do the
    outputs computed from them, so reading either needs a detach."""
    return t.detach().numpy()


def close(a, b, rtol):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-30) <= rtol


class Tiny(nn.Module):
    def __init__(self, din=8, dout=6):
        super().__init__()
        self.proj = nn.Linear(din, dout, bias=False)


def test_it_computes_the_definition():
    """``W x + (alpha/r) B A x``, against the same thing written out."""
    model = Tiny()
    with grilly.no_grad():
        model.proj.weight.copy_(grilly.from_numpy(rand((6, 8), 1)))
    wrapped = apply_lora_(model, rank=2, alpha=4.0, targets=["proj"])["proj"]
    with grilly.no_grad():
        wrapped.lora_A.weight.copy_(grilly.from_numpy(rand((2, 8), 2)))
        wrapped.lora_B.weight.copy_(grilly.from_numpy(rand((6, 2), 3)))
    x = grilly.from_numpy(rand((3, 8), 4))
    got = host(model.proj(x))
    w, a, b = (host(t) for t in (wrapped.weight, wrapped.lora_A.weight, wrapped.lora_B.weight))
    xn = host(x)
    want = xn @ w.T + (4.0 / 2) * ((xn @ a.T) @ b.T)
    assert close(got, want, 1e-7)


def test_the_base_is_untouched():
    """Frozen means frozen: the same tensor object, byte for byte."""
    model = Tiny()
    with grilly.no_grad():
        model.proj.weight.copy_(grilly.from_numpy(rand((6, 8), 5)))
    before_object = model.proj.weight
    before_bytes = host(before_object).copy()
    wrapped = apply_lora_(model, rank=2, alpha=4.0, targets=["proj"])["proj"]
    assert wrapped.weight is before_object, "the base weight was replaced"
    assert (host(wrapped.weight) == before_bytes).all(), "the base weight was modified"


def test_a_fresh_adapter_is_the_identity():
    """peft initializes B to zero, so attaching an untrained adapter must
    not move a single logit — otherwise every ES run starts from a model
    that is already different from the one that was measured."""
    model = Tiny()
    with grilly.no_grad():
        model.proj.weight.copy_(grilly.from_numpy(rand((6, 8), 6)))
    x = grilly.from_numpy(rand((3, 8), 7))
    before = host(model.proj(x)).copy()
    apply_lora_(model, rank=4, alpha=8.0, targets=["proj"])
    assert (host(model.proj(x)) == before).all()


def test_the_checkpoints_names_still_resolve():
    """The wrapper adopts the base's parameters rather than nesting it, so
    a base checkpoint and a peft adapter both load by their own names."""
    model = Tiny()
    apply_lora_(model, rank=2, alpha=4.0, targets=["proj"])
    names = {n for n, _ in model.named_parameters()}
    assert "proj.weight" in names, "the base checkpoint's name stopped resolving"
    assert {"proj.lora_A.weight", "proj.lora_B.weight"} <= names


def test_the_factors_are_what_an_optimizer_gets():
    model = Tiny()
    apply_lora_(model, rank=3, alpha=6.0, targets=["proj"])
    factors = lora_parameters(model)
    assert len(factors) == 2
    assert [tuple(f.shape) for f in factors] == [(3, 8), (6, 3)]
    assert all(f.dim() == 2 for f in factors), "ES only perturbs matrices"


def test_evolution_strategies_moves_the_factors_and_not_the_base():
    """The whole reason the adapter is unmerged."""
    from grilly.learning import EvolutionStrategy

    model = Tiny()
    with grilly.no_grad():
        model.proj.weight.copy_(grilly.from_numpy(rand((6, 8), 8)))
    apply_lora_(model, rank=2, alpha=4.0, targets=["proj"])
    base_before = host(model.proj.weight).copy()
    factors = lora_parameters(model)
    es = EvolutionStrategy(factors, sigma=0.05, lr=0.1, population=4, seed=1)
    x = grilly.from_numpy(rand((2, 8), 9))
    scores = []
    for member in range(es.population):
        with es.member(member):
            scores.append(float(host(model.proj(x)).sum()))
    es.step(scores)
    assert (host(model.proj.weight) == base_before).all(), "ES moved the frozen base"
    assert np.abs(host(factors[1])).max() > 0, "ES did not move the factors"


# ── the fusion trap ──────────────────────────────────────────────────────


def test_the_attention_fusion_refuses_a_wrapped_module():
    """Concatenating q/k/v weights runs the base and drops the adapter —
    a wrong answer with no error. ``fusable = False`` is what stops it.

    The weights are quantized first **so that fusion would otherwise
    happen**: float32 weights are not fused at all, so a version of this
    test without the quantization passes whether or not the guard exists.
    """
    from grilly.infer.models.llama import _fused_weight
    from grilly.infer.quantization import Int8Tensor

    def int8_linear(din, dout, seed):
        layer = nn.Linear(din, dout, bias=False)
        with grilly.no_grad():
            layer.weight.copy_(grilly.from_numpy(rand((dout, din), seed)))
        layer.weight = Int8Tensor.from_hp(layer.weight)
        return layer

    plain = [int8_linear(8, 8, 30 + i) for i in range(3)]
    fused, _ = _fused_weight({}, plain)
    assert fused is not None, "three int8 linears should fuse; the guard below proves nothing otherwise"

    model = Tiny(8, 8)
    with grilly.no_grad():
        model.proj.weight.copy_(grilly.from_numpy(rand((8, 8), 40)))
    model.proj.weight = Int8Tensor.from_hp(model.proj.weight)
    wrapped = apply_lora_(model, rank=2, alpha=4.0, targets=["proj"])["proj"]
    assert wrapped.fusable is False
    assert _fused_weight({}, [wrapped, plain[0], plain[1]]) == (None, None)


def test_a_lora_over_int8_keeps_the_quantized_base():
    """The base stays int8 — the adapter is beside it, not folded in, so
    nothing is dequantized and requantized."""
    from grilly.infer.quantization import Int8Tensor

    if not hasattr(Int8Tensor, "from_hp"):
        pytest.skip("no int8 on this backend")
    model = Tiny(16, 8)
    with grilly.no_grad():
        model.proj.weight.copy_(grilly.from_numpy(rand((8, 16), 10)))
    model.proj.weight = Int8Tensor.from_hp(model.proj.weight)
    quantized = model.proj.weight
    wrapped = apply_lora_(model, rank=2, alpha=4.0, targets=["proj"])["proj"]
    assert wrapped.weight is quantized
    assert isinstance(wrapped.weight, Int8Tensor)
    x = grilly.from_numpy(rand((1, 16), 11))
    assert tuple(model.proj(x).shape) == (1, 8)


# ── the peft file format ─────────────────────────────────────────────────


def write_adapter(tmp_path, paths, rank, alpha, shapes, extra=None):
    from safetensors.numpy import save_file

    config = {"peft_type": "LORA", "r": rank, "lora_alpha": alpha, "lora_dropout": 0.0,
              "target_modules": list(paths)}
    config.update(extra or {})
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {}
    for i, path in enumerate(paths):
        din, dout = shapes[path]
        tensors[f"base_model.model.{path}.lora_A.weight"] = rand((rank, din), 100 + i)
        tensors[f"base_model.model.{path}.lora_B.weight"] = rand((dout, rank), 200 + i)
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    return tmp_path


def test_a_peft_adapter_loads_and_changes_the_output(tmp_path):
    from grilly.huggingface import load_lora_adapter

    model = Tiny(8, 6)
    with grilly.no_grad():
        model.proj.weight.copy_(grilly.from_numpy(rand((6, 8), 12)))
    x = grilly.from_numpy(rand((2, 8), 13))
    before = host(model.proj(x)).copy()
    write_adapter(tmp_path, ["proj"], 4, 8.0, {"proj": (8, 6)})
    wrapped = load_lora_adapter(model, tmp_path)
    assert set(wrapped) == {"proj"}
    assert wrapped["proj"].scaling == 2.0
    assert not (host(model.proj(x)) == before).all(), "a trained adapter changed nothing"


def test_an_adapter_for_a_different_base_is_refused(tmp_path):
    """Caught at load, where it is one message, rather than as a model
    that runs and is quietly wrong."""
    from grilly.huggingface import load_lora_adapter
    from grilly.huggingface.peft import PeftConfigError

    model = Tiny()
    write_adapter(tmp_path, ["not_a_module"], 2, 4.0, {"not_a_module": (8, 6)})
    with pytest.raises(PeftConfigError, match="different base"):
        load_lora_adapter(model, tmp_path)


@pytest.mark.parametrize("field,value", [("use_dora", True), ("use_rslora", True),
                                         ("lora_dropout", 0.1)])
def test_peft_features_that_change_the_arithmetic_are_refused(tmp_path, field, value):
    """Each of these would make the loaded adapter compute something other
    than what it was trained as, so none is ignored."""
    from grilly.huggingface import load_lora_adapter

    model = Tiny()
    write_adapter(tmp_path, ["proj"], 2, 4.0, {"proj": (8, 6)}, extra={field: value})
    with pytest.raises(NotImplementedError):
        load_lora_adapter(model, tmp_path)


# ── a whole population as one batch ──────────────────────────────────────


class TwoLayer(nn.Module):
    def __init__(self, din=12, hidden=10, dout=6):
        super().__init__()
        self.up = nn.Linear(din, hidden, bias=False)
        self.down = nn.Linear(hidden, dout, bias=False)

    def forward(self, x):
        return self.down(self.up(x))


def two_layer_with_lora(rank=2, alpha=4.0, seed=60):
    model = TwoLayer()
    with grilly.no_grad():
        model.up.weight.copy_(grilly.from_numpy(rand((10, 12), seed)))
        model.down.weight.copy_(grilly.from_numpy(rand((6, 10), seed + 1)))
    apply_lora_(model, rank=rank, alpha=alpha, targets=["up", "down"])
    with grilly.no_grad():                    # a trained adapter, not peft's zero B
        for i, f in enumerate(lora_parameters(model)):
            f.copy_(grilly.from_numpy(rand(tuple(f.shape), seed + 10 + i)))
    return model


def test_a_batched_population_equals_running_each_member_alone():
    """The claim the whole batched path rests on, and the only one that
    matters: scoring 32 members as one batch has to give what scoring them
    one at a time gives.

    A composed chain on one backend, but the two sides group their products
    differently (the batched one never builds a perturbed factor), so this
    is bounded rather than bit-exact — the same reason
    ``test_the_population_lora_equals_n_materialised_adapters`` is.
    """
    from grilly.infer.lora import population
    from grilly.learning import EvolutionStrategy

    model = two_layer_with_lora()
    es = EvolutionStrategy(lora_parameters(model), sigma=0.05, lr=0.0, population=6, seed=5)
    x = grilly.from_numpy(rand((1, 3, 12), 70))

    serial = []
    for member in range(es.population):
        with es.member(member):
            serial.append(host(model(x))[0])
    serial = np.stack(serial)

    with population(model, es, repeat=1):
        batched = host(model(grilly.from_numpy(np.repeat(host(x), es.population, axis=0))))

    scale = max(np.abs(serial).max(), 1e-30)
    worst = float(np.abs(batched - serial).max() / scale)
    print(f"  batched vs one-at-a-time: {worst:.2e} relative")
    assert worst < 1e-5, f"the batched population disagrees by {worst:.2e}"


def test_the_batch_lays_members_out_in_the_order_the_caller_must_use():
    """``repeat`` gives each member consecutive rows, so a batch of
    (member, prompt) pairs lines up. Getting this backwards would score
    every member on the wrong prompts and still run."""
    from grilly.infer.lora import population
    from grilly.learning import EvolutionStrategy

    model = two_layer_with_lora()
    es = EvolutionStrategy(lora_parameters(model), sigma=0.05, lr=0.0, population=4, seed=6)
    prompts = 3
    x = grilly.from_numpy(rand((1, 2, 12), 71))
    wide = grilly.from_numpy(np.repeat(host(x), es.population * prompts, axis=0))

    with population(model, es, repeat=prompts):
        batched = host(model(wide))
    # rows within one member are the same input under the same perturbation
    for member in range(es.population):
        block = batched[member * prompts:(member + 1) * prompts]
        assert np.abs(block - block[0]).max() < 1e-6, f"member {member}'s rows differ"
    # and different members really are different
    firsts = batched[::prompts]
    assert np.abs(firsts[0] - firsts[1]).max() > 1e-6


def test_the_model_is_itself_again_after_the_block():
    from grilly.infer.lora import population
    from grilly.learning import EvolutionStrategy

    model = two_layer_with_lora()
    es = EvolutionStrategy(lora_parameters(model), sigma=0.05, lr=0.0, population=4, seed=7)
    x = grilly.from_numpy(rand((1, 2, 12), 72))
    before = host(model(x)).copy()
    try:
        with population(model, es, repeat=1):
            raise RuntimeError("something went wrong mid-batch")
    except RuntimeError:
        pass
    assert (host(model(x)) == before).all(), "an exception left the model perturbed"
