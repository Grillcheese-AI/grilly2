"""grilly's inference machinery (CLAUDE.md §11).

``grilly.infer.safetensors`` loads checkpoints. Quantization, GPU sampling
and the decode loop follow. Nothing here depends on a model-zoo adapter:
a safetensors file loads, and a model runs, without one.
"""

from . import safetensors

__all__ = ["safetensors"]
