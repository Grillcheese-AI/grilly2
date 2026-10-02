"""peft LoRA adapters: ``adapter_config.json`` + ``adapter_model.safetensors``.

An adapter over the adapter — this reads peft's file layout and hands the
factors to :mod:`grilly.infer.lora`, which is where the arithmetic lives
and works with this package uninstalled (CLAUDE.md §11).

peft names a wrapped weight ``base_model.model.<path>.lora_A.weight``. The
prefix is peft's own wrapper chain and carries no information, so it is
stripped; what is left is the module path in the model, which is why a
grilly model whose parameter names are the checkpoint's can take an
adapter with no name table anywhere.

**Nothing is merged.** The factors go in beside a frozen base — see
``grilly.infer.lora`` for why that is the point rather than a shortcut.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .. import _not_yet, from_numpy, no_grad
from ..infer.lora import apply_lora_

__all__ = ["load_lora_adapter", "PeftConfigError"]

CONFIG, WEIGHTS = "adapter_config.json", "adapter_model.safetensors"
_PREFIX = re.compile(r"^base_model\.model\.")
_SUFFIX = re.compile(r"\.(lora_A|lora_B)\.(?:default\.)?weight$")

# peft features that change the arithmetic, and what each would need
_UNSUPPORTED = {
    "use_dora": "DoRA (a learned magnitude beside the direction)",
    "use_rslora": "rank-stabilized scaling (alpha / sqrt(r), not alpha / r)",
    "lora_bias": "a trainable bias on the adapter",
    "megatron_config": "megatron tensor parallelism",
    "loftq_config": "LoftQ initialization",
    "alora_invocation_tokens": "activated LoRA",
}


class PeftConfigError(ValueError):
    """An adapter this loader will not apply, said out loud."""


def _check(config: dict) -> None:
    if config.get("peft_type", "LORA") != "LORA":
        raise _not_yet(f"peft_type={config.get('peft_type')!r} (grilly has LORA)")
    for key, what in _UNSUPPORTED.items():
        value = config.get(key)
        if value:
            raise _not_yet(f"{key}={value!r} ({what})")
    if config.get("lora_dropout"):
        # dropout is a no-op at inference, but a non-zero value in a config
        # being *trained* through would silently not be applied
        raise _not_yet(f"lora_dropout={config['lora_dropout']} (grilly applies no dropout)")
    if config.get("alpha_pattern") or config.get("rank_pattern"):
        raise _not_yet("per-module rank or alpha (alpha_pattern / rank_pattern)")


def _factors(path: Path) -> dict:
    """``{module path: {"lora_A": array, "lora_B": array}}`` from the file."""
    from safetensors.numpy import load_file

    out: dict = {}
    for name, array in load_file(str(path)).items():
        stripped = _PREFIX.sub("", name)
        match = _SUFFIX.search(stripped)
        if match is None:
            raise PeftConfigError(f"unexpected tensor in the adapter: {name!r}")
        out.setdefault(stripped[: match.start()], {})[match.group(1)] = array
    return out


def load_lora_adapter(model, directory, *, strict: bool = True) -> dict:
    """Apply the peft adapter in ``directory`` to ``model``, unmerged.

    Returns the wrappers by module path. ``strict`` refuses an adapter
    naming a module the model does not have, which is how a mismatched
    base is caught at load rather than as a wrong answer later.
    """
    directory = Path(directory)
    config = json.loads((directory / CONFIG).read_text(encoding="utf-8"))
    _check(config)
    rank, alpha = int(config["r"]), float(config["lora_alpha"])

    factors = _factors(directory / WEIGHTS)
    missing = [path for path, pair in factors.items() if set(pair) != {"lora_A", "lora_B"}]
    if missing:
        raise PeftConfigError(f"{len(missing)} modules have only one factor, e.g. {missing[0]!r}")

    known = {name for name, _ in model.named_modules()}
    absent = sorted(path for path in factors if path not in known)
    if absent and strict:
        raise PeftConfigError(
            f"the adapter names {len(absent)} modules this model does not have, "
            f"e.g. {absent[0]!r} — is it an adapter for a different base?"
        )
    targets = [path for path in factors if path in known]

    wrapped = apply_lora_(model, rank, alpha, targets)
    with no_grad():
        for path, adapter in wrapped.items():
            pair = factors[path]
            for which, module in (("lora_A", adapter.lora_A), ("lora_B", adapter.lora_B)):
                array = pair[which]
                if tuple(module.weight.shape) != tuple(array.shape):
                    raise PeftConfigError(
                        f"{path}.{which}: adapter has {tuple(array.shape)}, "
                        f"model wants {tuple(module.weight.shape)}"
                    )
                module.weight.copy_(from_numpy(array.astype("float32", copy=False)))
    return wrapped
