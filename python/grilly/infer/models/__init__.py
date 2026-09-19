"""Models built from grilly's machinery, usable with no model-zoo adapter.
Their parameter names are Hugging Face's, so checkpoints load as they are."""

from .llama import LlamaConfig, LlamaForCausalLM

__all__ = ["LlamaConfig", "LlamaForCausalLM"]
