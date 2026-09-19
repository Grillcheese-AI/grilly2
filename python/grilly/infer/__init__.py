"""grilly's inference machinery (CLAUDE.md §11).

- ``grilly.infer.safetensors`` loads checkpoints;
- ``grilly.infer.quantization`` quantizes linear weights to int8;
- ``grilly.infer.sampling`` picks next tokens on the GPU.

The decode loop follows. Nothing here depends on a model-zoo adapter: a
safetensors file loads, and a model runs, without one.
"""

from . import quantization, safetensors, sampling

__all__ = ["quantization", "safetensors", "sampling"]
