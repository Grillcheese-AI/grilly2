"""Hugging Face's entry points, over grilly's inference machinery.

    from grilly.huggingface import AutoModelForCausalLM, AutoTokenizer
    from grilly.infer.quantization import Int8WeightOnlyConfig

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                                 quantization_config=Int8WeightOnlyConfig())
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    ids = tokenizer.apply_chat_template([{"role": "user", "content": "Why is the sky blue?"}],
                                        add_generation_prompt=True, return_tensors="pt")
    print(tokenizer.decode(model.generate(ids, max_new_tokens=128)[0]))

**This package is an adapter and nothing else** (CLAUDE.md §11). It reads
``config.json``, finds a repository's files, and hands them to
``grilly.infer``; every mechanism it uses — the models, safetensors
loading, quantization, the KV cache, sampling, ``generate`` — lives there
and works with this package uninstalled. ``tests/python/test_huggingface.py``
holds that line: one test loads a checkpoint and generates with
``grilly.huggingface`` absent from ``sys.modules``.

What it does *not* do: download or run arbitrary modelling code
(``trust_remote_code`` is refused), convert ``.bin`` checkpoints, or
implement a tokenizer. ``AutoTokenizer`` here is a thin wrapper over
Hugging Face's ``tokenizers``, which grilly depends on rather than
reimplementing.
"""

from __future__ import annotations

from .configuration import AutoConfig, config_from_dict, model_class_for
from .modeling import AutoModel, AutoModelForCausalLM, from_pretrained
from .peft import load_lora_adapter
from .tokenization import AutoTokenizer

__all__ = [
    "AutoConfig",
    "AutoModel",
    "AutoModelForCausalLM",
    "AutoTokenizer",
    "config_from_dict",
    "from_pretrained",
    "load_lora_adapter",
    "model_class_for",
]
