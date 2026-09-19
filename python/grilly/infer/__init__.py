"""grilly's inference machinery (CLAUDE.md §11).

- ``grilly.infer.safetensors`` loads checkpoints;
- ``grilly.infer.quantization`` quantizes linear weights to int8;
- ``grilly.infer.sampling`` picks next tokens on the GPU;
- ``grilly.infer.cache`` holds keys and values on the GPU for decoding.

The decode loop follows. Nothing here depends on a model-zoo adapter: a
safetensors file loads, and a model runs, without one.
"""

from . import cache, quantization, safetensors, sampling

__all__ = ["cache", "quantization", "safetensors", "sampling"]
