"""Finding a model's files, and reading its weights into a grilly model.

A repository id (``"Qwen/Qwen2.5-0.5B-Instruct"``) is resolved through
``huggingface_hub`` when it is installed; a path to a directory is used as
it is, and needs nothing installed. Weights are read shard by shard and
tensor by tensor, so the peak is the model, not the model twice.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .. import _not_yet
from ..infer.quantization import Int8WeightOnlyConfig, load_quantized_, quantize_
from ..infer.safetensors import stream_into

__all__ = ["load_weights", "repo_files", "shard_files"]

WEIGHTS = "model.safetensors"
INDEX = "model.safetensors.index.json"


class _Missing(Exception):
    """A file the repository does not have (or is not cached)."""


def _local(path: Path, name: str) -> str:
    f = path / name
    if not f.is_file():
        raise _Missing(name)
    return os.fspath(f)


def _download(repo: str, name: str, *, revision=None, cache_dir=None, local_files_only=False, token=None) -> str:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    except ImportError:  # pragma: no cover - depends on the environment
        raise _not_yet(f"{repo!r} is not a local directory, and huggingface_hub is not installed "
                       "(pip install huggingface_hub, or pass a directory)") from None
    try:
        return hf_hub_download(repo, name, revision=revision, cache_dir=cache_dir,
                               local_files_only=local_files_only, token=token)
    except (EntryNotFoundError, LocalEntryNotFoundError) as e:
        raise _Missing(name) from e


def repo_files(name_or_path, filenames, optional=(), **kwargs) -> dict[str, str]:
    """Local paths for ``filenames`` from a directory or a Hub repository.
    Names in ``optional`` are left out when the repository has no such file;
    anything else missing is an error."""
    path = Path(name_or_path)
    directory = path.is_dir()
    out = {}
    for name in filenames:
        try:
            out[name] = _local(path, name) if directory else _download(str(name_or_path), name, **kwargs)
        except _Missing:
            if name not in optional:
                where = f"directory {path}" if directory else f"repository {name_or_path!r}"
                raise FileNotFoundError(f"{where} has no {name}") from None
    return out


def shard_files(name_or_path, **kwargs) -> list[str]:
    """The checkpoint's safetensors files, in order: one ``model.safetensors``
    or the shards its index names."""
    found = repo_files(name_or_path, (WEIGHTS, INDEX), optional=(WEIGHTS, INDEX), **kwargs)
    if INDEX in found:
        index = json.loads(Path(found[INDEX]).read_text(encoding="utf-8"))
        names = sorted(set(index["weight_map"].values()))
        return [repo_files(name_or_path, (n,), **kwargs)[n] for n in names]
    if WEIGHTS in found:
        return [found[WEIGHTS]]
    raise FileNotFoundError(f"{name_or_path} has neither {WEIGHTS} nor {INDEX} "
                            "(grilly loads safetensors checkpoints; convert a .bin with safetensors)")


def load_weights(model, shards, quantization_config=None, strict: bool = True) -> None:
    """Read ``shards`` into ``model``, quantizing weights as they arrive when
    a config is given, then tie and quantize whatever the checkpoint did not
    carry (a tied output projection)."""
    loaded: set[str] = set()
    for shard in shards:
        if quantization_config is None:
            loaded |= stream_into(model, shard, strict=False)
        else:
            loaded |= load_quantized_(model, shard, quantization_config, strict=False)
    tie = getattr(model, "tie_weights", None)
    if tie is not None:
        tie()
    # a weight tied to one the checkpoint carried is not missing. This is read
    # before quantizing, which gives the tied output projection its own tensor.
    own = model.state_dict(keep_vars=True)  # the parameters themselves: tying is identity
    carried = {id(v) for k, v in own.items() if k in loaded}
    accounted = loaded | {k for k, v in own.items() if id(v) in carried}
    if quantization_config is not None:
        quantize_(model, quantization_config)  # only what the checkpoint did not carry
    # a model built on the meta device must be fully replaced by now: a
    # parameter still without data would fail much later, in a kernel
    left = sorted(k for k, v in model.state_dict(keep_vars=True).items() if v.is_meta)
    if left:
        names = ", ".join(f'"{k}"' for k in left)
        raise RuntimeError(f"{type(model).__name__} has parameters with no data after loading "
                           f"(built on the meta device, and the checkpoint did not carry them): {names}")
    if strict:
        # keep_vars again: a plain state_dict() of a quantized model dequantizes
        # every weight at once, which is the whole model in float32
        missing = [k for k in model.state_dict(keep_vars=True) if k not in accounted]
        if missing:
            names = ", ".join(f'"{k}"' for k in missing)
            raise RuntimeError(f"Error(s) in loading state_dict for {type(model).__name__}:"
                               f"\n\tMissing key(s) in state_dict: {names}")


def quantization_config_of(config) -> Int8WeightOnlyConfig | None:
    """What ``quantization_config=`` means, accepting torchao's wrapper
    (``TorchAoConfig(Int8WeightOnlyConfig())``) as transformers does."""
    if config is None:
        return None
    inner = getattr(config, "quant_type", config)  # transformers' TorchAoConfig
    if isinstance(inner, Int8WeightOnlyConfig):
        return inner
    if inner in ("int8_weight_only", "int8"):
        return Int8WeightOnlyConfig()
    raise _not_yet(f"quantization_config={config!r} (grilly has Int8WeightOnlyConfig and TorchAoConfig of it)")
