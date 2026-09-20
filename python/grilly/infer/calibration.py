"""Run a model on calibration text and quantize it layer by layer with GPTQ.

GPTQ needs each linear's own inputs (`grilly.infer.gptq`), so the model is
walked one decoder layer at a time:

1. run the layer, capturing what reaches each of its seven linears, and
   accumulate `H = 2 X^T X` for each on the device;
2. quantize those seven against their Hessians;
3. run the layer **again**, now quantized, to produce the next layer's
   input — so every layer is calibrated against what it will actually
   receive, errors included.

That third step is the part that matters and the reason this cannot be a
per-tensor transform like round-to-nearest.

`lm_head` is left to the caller. It is 152K rows here, and GPTQ walks the
columns of a weight one at a time, so it is minutes of work for the layer
that is least sensitive to rounding; AutoGPTQ and torchao leave it out too.
"""

from __future__ import annotations

import numpy as np

from .. import Tensor, matmul, no_grad
from ..nn import Module
from .gptq import gptq_quantize
from .quantization import Int4Tensor

__all__ = ["LINEARS", "calibration_ids", "quantize_gptq_", "wikitext_texts"]

# the seven linears of a decoder layer, in the order they are reached
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


class _Capture(Module):
    """Wraps a linear to accumulate `2 X^T X` over everything it is given."""

    def __init__(self, inner: Module) -> None:
        super().__init__()
        self.inner = inner
        self.hessian = None
        self.rows = 0

    def forward(self, x: Tensor) -> Tensor:
        flat = x.reshape(-1, x.shape[-1])
        gram = matmul(flat.transpose(0, 1), flat)
        self.hessian = gram if self.hessian is None else self.hessian + gram
        self.rows += flat.shape[0]
        return self.inner(x)


def _get(module: Module, path: str) -> Module:
    for part in path.split("."):
        module = getattr(module, part)
    return module


def _set(module: Module, path: str, value) -> None:
    owner, _, name = path.rpartition(".")
    setattr(_get(module, owner) if owner else module, name, value)


def wikitext_texts(samples: int = 128, seed: int = 0) -> list[str]:
    """Paragraphs of wikitext-2, the corpus GPTQ's own experiments use.
    Downloads it (about 5 MB) the first time."""
    from datasets import load_dataset

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    rng = np.random.default_rng(seed)
    long_enough = [t for t in data["text"] if len(t) > 600]
    picked = rng.choice(len(long_enough), size=min(samples, len(long_enough)), replace=False)
    return [long_enough[i] for i in picked]


def calibration_ids(tokenizer, texts, samples: int = 16, length: int = 512, seed: int = 0) -> Tensor:
    """`(samples, length)` token ids: the texts joined into one stream and
    cut into windows at random offsets, as GPTQ's own recipe does. A
    paragraph is usually shorter than a window, which is why they are
    concatenated rather than taken one per row."""
    from .. import tensor

    stream: list[int] = []
    for text in texts:
        stream.extend(tokenizer.encode(text))
        if len(stream) >= samples * length * 2:
            break
    if len(stream) < length + 1:
        raise ValueError(f"the calibration text is {len(stream)} tokens, shorter than one "
                         f"window of {length}; pass more text")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max(1, len(stream) - length), size=samples)
    return tensor([stream[start:start + length] for start in starts])


def quantize_gptq_(model, input_ids: Tensor, group_size: int = 128, percdamp: float = 0.01,
                   blocksize: int = 128, act_order: bool = False, report=None) -> None:
    """Quantize every decoder layer's linears to int4 with GPTQ, in place,
    calibrated on `input_ids` (batch, length)."""
    say = report or (lambda *a: None)
    with no_grad():
        hidden = model.model.embed_tokens(input_ids)
        cos, sin = model._rotary.tables(input_ids.shape[1])
        for index, layer in enumerate(model.model.layers):
            captures = {}
            for path in LINEARS:
                captures[path] = _Capture(_get(layer, path))
                _set(layer, path, captures[path])
            layer(hidden, cos, sin, None, 0)  # pass 1: collect

            for path, capture in captures.items():
                linear = capture.inner
                weight = linear.weight.detach().numpy()
                hessian = capture.hessian.numpy().astype(np.float64)
                levels, scales, zeros = gptq_quantize(
                    weight, hessian, group_size=group_size, blocksize=blocksize,
                    percdamp=percdamp, act_order=act_order)
                _set(layer, path, linear)  # unwrap first: the capture holds a Hessian
                linear.weight = Int4Tensor.from_parts(levels, scales, zeros, group_size)
                del hessian, weight
            hidden = layer(hidden, cos, sin, None, 0)  # pass 2: quantized output
            say(f"  layer {index + 1}/{len(model.model.layers)} quantized")
