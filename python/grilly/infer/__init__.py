"""grilly's inference machinery (CLAUDE.md §11).

- ``grilly.infer.safetensors`` loads checkpoints;
- ``grilly.infer.quantization`` quantizes linear weights to int8;
- ``grilly.infer.lora`` puts low-rank factors beside a frozen weight;
- ``grilly.infer.sampling`` picks next tokens on the GPU;
- ``grilly.infer.cache`` holds keys and values on the GPU for decoding;
- ``grilly.infer.generation`` is the decode loop, with HF's ``generate()``;
- ``grilly.infer.models`` has a Llama-family decoder (Llama, Qwen2).

Nothing here depends on a model-zoo adapter: a
safetensors file loads, and a model runs, without one.
"""

from . import cache, generation, models, quantization, safetensors, sampling

__all__ = ["cache", "generation", "models", "quantization", "safetensors", "sampling"]
