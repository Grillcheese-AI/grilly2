"""grilly's inference machinery (CLAUDE.md §11).

``grilly.infer.safetensors`` loads checkpoints, and
``grilly.infer.quantization`` quantizes linear weights to int8. GPU
sampling and the decode loop follow. Nothing here depends on a model-zoo
adapter: a safetensors file loads, and a model runs, without one.
"""

from . import quantization, safetensors

__all__ = ["quantization", "safetensors"]
