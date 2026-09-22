"""``config.json`` to a grilly model config, and the architecture to the
class that runs it.

This is the only place that knows Hugging Face's field names. The models in
``grilly.infer.models`` take a config object and nothing else, so a model
can be built without this package (CLAUDE.md §11).
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import _not_yet
from ..infer.models import LlamaConfig, LlamaForCausalLM

__all__ = ["AutoConfig", "config_from_dict", "model_class_for"]

# model_type -> the grilly model that runs it. Both families are the same
# decoder; Qwen2 differs only in having q/k/v biases.
_FAMILIES = {"llama": LlamaForCausalLM, "qwen2": LlamaForCausalLM, "qwen3": LlamaForCausalLM}

# fields grilly's decoder does not implement, and what they would change
_UNSUPPORTED = {
    "rope_scaling": "scaled rotary embeddings (linear, dynamic, llama3, yarn)",
    "use_sliding_window": "sliding-window attention",
    "attention_dropout": "attention dropout at inference is a no-op only when 0",
    "num_experts": "mixture of experts",
    "num_local_experts": "mixture of experts",
}


def _rope_theta(c: dict) -> float:
    """``rope_theta``, from wherever this config's version keeps it.
    transformers 5 moved it into ``rope_parameters`` alongside a
    ``rope_type``; anything but the plain type changes the positions, so it
    is refused rather than ignored."""
    parameters = c.get("rope_parameters") or {}
    rope_type = parameters.get("rope_type", parameters.get("type", "default"))
    if rope_type != "default":
        raise _not_yet(f"rope_parameters {parameters!r} (grilly has the default rotary embedding)")
    for key in ("rope_theta", "theta"):
        if key in parameters:
            return float(parameters[key])
    return float(c.get("rope_theta", 10000.0))


def _refuse_unsupported(c: dict) -> None:
    for key, what in _UNSUPPORTED.items():
        value = c.get(key)
        if value and not (key == "attention_dropout" and value == 0.0):
            raise _not_yet(f"{c.get('model_type', 'this model')}: {key}={value!r} ({what})")


def config_from_dict(c: dict) -> LlamaConfig:
    """A parsed ``config.json`` as a grilly config."""
    model_type = c.get("model_type")
    if model_type not in _FAMILIES:
        raise _not_yet(f"model_type={model_type!r} (grilly has {', '.join(sorted(_FAMILIES))})")
    _refuse_unsupported(c)
    qwen2 = model_type == "qwen2"
    bias = c.get("attention_bias", False)
    return LlamaConfig(
        vocab_size=c["vocab_size"],
        hidden_size=c["hidden_size"],
        intermediate_size=c["intermediate_size"],
        num_hidden_layers=c["num_hidden_layers"],
        num_attention_heads=c["num_attention_heads"],
        num_key_value_heads=c.get("num_key_value_heads"),
        head_dim=c.get("head_dim"),
        rms_norm_eps=c.get("rms_norm_eps", 1e-6),
        rope_theta=_rope_theta(c),
        # Qwen2 biases q, k and v and never the output projection; Llama's
        # attention_bias covers all four; Qwen3 biases nothing by default
        qkv_bias=qwen2 or bias,
        o_bias=bias and not qwen2,
        qk_norm=model_type == "qwen3",  # an RMSNorm per head on q and k
        tie_word_embeddings=c.get("tie_word_embeddings", False),
        hidden_act=c.get("hidden_act", "silu"),
    )


def model_class_for(c: dict):
    """The grilly class that runs this config (``architectures`` decides when
    it names something grilly knows; otherwise ``model_type``)."""
    for name in c.get("architectures") or []:
        if name in ("LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"):
            return LlamaForCausalLM
        if name.endswith(("ForSequenceClassification", "ForTokenClassification", "ForQuestionAnswering")):
            raise _not_yet(f"architecture {name} (grilly.huggingface loads causal language models)")
    model_type = c.get("model_type")
    if model_type not in _FAMILIES:
        raise _not_yet(f"model_type={model_type!r} (grilly has {', '.join(sorted(_FAMILIES))})")
    return _FAMILIES[model_type]


class AutoConfig:
    """``transformers.AutoConfig``, for the configs grilly can run."""

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs) -> LlamaConfig:
        from .loading import repo_files

        files = repo_files(pretrained_model_name_or_path, ("config.json",), **kwargs)
        return config_from_dict(json.loads(Path(files["config.json"]).read_text(encoding="utf-8")))
