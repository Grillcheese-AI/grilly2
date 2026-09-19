"""``from_pretrained``: a repository or directory to a loaded grilly model."""

from __future__ import annotations

import json
from pathlib import Path

from dataclasses import fields

from .. import _not_yet, float32
from ..infer.generation import GenerationConfig
from .configuration import config_from_dict, model_class_for
from .loading import load_weights, quantization_config_of, repo_files, shard_files

__all__ = ["AutoModel", "AutoModelForCausalLM", "from_pretrained", "generation_config_from"]

CONFIG, GENERATION = "config.json", "generation_config.json"

# transformers arguments that make no difference here, and why
_IGNORED = {
    "low_cpu_mem_usage": "grilly always loads one tensor at a time",
    "use_safetensors": "safetensors is the only format grilly reads",
    "attn_implementation": "grilly has one attention implementation",
    "_fast_init": "grilly does not initialize weights it is about to overwrite",
    "use_cache": "a cache is passed to generate(), not configured here",
}


def _check_dtype(dtype) -> None:
    if dtype in (None, "auto", float32, "float32", "float"):
        return
    raise _not_yet(f"dtype={dtype!r} (grilly computes in float32; quantize with quantization_config=)")


def _check_device_map(device_map) -> None:
    if device_map in (None, "auto", "balanced", "cpu", 0, "cuda", "cuda:0"):
        return  # grilly has one device; "auto" on one device means that device
    raise _not_yet(f"device_map={device_map!r} (grilly has one device; use None or 'auto')")


def from_pretrained(pretrained_model_name_or_path, *, config=None, dtype=None, torch_dtype=None,
                    quantization_config=None, device_map=None, trust_remote_code=False,
                    revision=None, cache_dir=None, local_files_only=False, token=None, **kwargs):
    """Build and load a model from a Hugging Face repository id or a local
    directory, as ``transformers`` does.

    ``quantization_config=Int8WeightOnlyConfig()`` (or transformers'
    ``TorchAoConfig`` of one) quantizes each weight as it is read, which is
    how a checkpoint larger than free VRAM gets loaded at all.
    """
    if trust_remote_code:
        raise _not_yet("trust_remote_code=True (grilly will not download and run modelling code)")
    _check_dtype(dtype if dtype is not None else torch_dtype)
    _check_device_map(device_map)
    for name in kwargs:
        if name not in _IGNORED:
            raise TypeError(f"from_pretrained() got an unexpected keyword argument {name!r}")
    where = dict(revision=revision, cache_dir=cache_dir, local_files_only=local_files_only, token=token)

    files = repo_files(pretrained_model_name_or_path, (CONFIG, GENERATION), optional=(GENERATION,), **where)
    raw = json.loads(Path(files[CONFIG]).read_text(encoding="utf-8"))
    model = (model_class_for(raw))(config if config is not None else config_from_dict(raw))
    load_weights(model, shard_files(pretrained_model_name_or_path, **where),
                 quantization_config=quantization_config_of(quantization_config))
    if GENERATION in files:
        model.generation_config = generation_config_from(json.loads(Path(files[GENERATION]).read_text(encoding="utf-8")))
    return model


def generation_config_from(raw: dict) -> GenerationConfig:
    """``generation_config.json`` as grilly's ``GenerationConfig``. Fields
    grilly's loop does not have (``bos_token_id``, ``transformers_version``,
    beam settings left at their defaults) are dropped; a field that would
    change what is generated is refused rather than ignored."""
    known = {f.name for f in fields(GenerationConfig)}
    config = GenerationConfig()
    for key, value in raw.items():
        if key in known:
            setattr(config, key, value)
        elif key in _GENERATION_DEFAULTS and value not in (None, _GENERATION_DEFAULTS[key]):
            raise _not_yet(f"generation_config.json: {key}={value!r}")
    return config


# generation_config.json fields grilly's loop does not implement: harmless at
# these values, and refused at any other
_GENERATION_DEFAULTS = {
    "num_beam_groups": 1, "penalty_alpha": None, "typical_p": 1.0, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
    "diversity_penalty": 0.0, "length_penalty": 1.0, "no_repeat_ngram_size": 0, "encoder_repetition_penalty": 1.0,
    "bad_words_ids": None, "force_words_ids": None, "min_length": 0, "min_new_tokens": None,
    "early_stopping": False, "num_return_sequences": 1, "output_scores": False, "renormalize_logits": False,
    "suppress_tokens": None, "begin_suppress_tokens": None, "forced_bos_token_id": None,
    "forced_eos_token_id": None, "exponential_decay_length_penalty": None, "guidance_scale": None,
    "low_memory": None, "watermarking_config": None, "assistant_confidence_threshold": None,
}


class AutoModelForCausalLM:
    """``transformers.AutoModelForCausalLM``, for the architectures grilly
    runs (Llama and Qwen2 decoders)."""

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        return from_pretrained(pretrained_model_name_or_path, **kwargs)

    @staticmethod
    def from_config(config):
        """An unloaded model from a config object or a parsed ``config.json``."""
        raw = config if isinstance(config, dict) else {"model_type": "llama"}
        return (model_class_for(raw))(config_from_dict(config) if isinstance(config, dict) else config)


AutoModel = AutoModelForCausalLM
