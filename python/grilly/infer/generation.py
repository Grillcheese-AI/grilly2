"""A decode loop with Hugging Face's ``generate()`` signature (CLAUDE.md
§11, step 4)::

    ids = model.generate(input_ids, max_new_tokens=50, do_sample=True,
                         temperature=0.7, top_k=20, top_p=0.8)

``GenerationMixin.generate`` takes ``inputs`` (or ``input_ids=``), a
``generation_config`` and HF's generation keyword arguments, with HF's
defaults. The supported ones are ``max_length``, ``max_new_tokens``,
``do_sample``, ``temperature``, ``top_k``, ``top_p``, ``repetition_penalty``,
``eos_token_id`` and ``pad_token_id``. It also takes ``attention_mask`` (all
ones: batches of equal length) and ``streamer``. Anything else raises,
naming it.

The prompt runs once, then one token per step against a GPU KV cache.
Sampling happens on the device (``grilly.infer.sampling``). Each step reads
back only the sampled token ids, for stopping at EOS and for the streamer.
Randomness comes from grilly's global generator, which ``manual_seed``
seeds, as torch's does for transformers.

It returns the sequences, prompt first, as int64 ``(batch, length)``.
Finished rows are padded with ``pad_token_id``, which defaults to the EOS
id, as in transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from .. import Tensor, _not_yet, _require_tensor, cat, int64, no_grad, tensor
from . import sampling

__all__ = ["CausalLMOutput", "GenerationConfig", "GenerationMixin"]


@dataclass
class CausalLMOutput:
    """What a causal language model's forward returns: ``logits`` and the
    cache it continued."""

    logits: Tensor
    past_key_values: object = None


@dataclass
class GenerationConfig:
    """The ``transformers.GenerationConfig`` fields grilly's loop uses, with
    their defaults."""

    max_length: int = 20
    max_new_tokens: int | None = None
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = 50
    top_p: float | None = 1.0
    repetition_penalty: float = 1.0
    eos_token_id: int | list | None = None
    pad_token_id: int | None = None
    num_beams: int = 1
    use_cache: bool = True


_FIELDS = {f.name for f in fields(GenerationConfig)}


class GenerationMixin:
    """``generate`` for a causal LM whose forward is
    ``forward(input_ids, past_key_values=None, logits_to_keep=0)`` and which
    has ``make_cache(batch_size, max_length)``."""

    generation_config: GenerationConfig | None = None

    def generate(self, inputs=None, generation_config: GenerationConfig | None = None, streamer=None, **kwargs) -> Tensor:
        if inputs is None:
            inputs = kwargs.pop("input_ids", None)
        if inputs is None:
            raise ValueError("generate() needs `inputs` (or `input_ids`): the prompt's token ids")
        ids = _require_tensor("inputs", inputs)
        if ids.dtype is not int64 or ids.dim() != 2:
            raise ValueError(f"`inputs` must be int64 (batch, length), got {ids.dtype} {tuple(ids.shape)}")
        mask = kwargs.pop("attention_mask", None)
        if mask is not None and not bool((_require_tensor("attention_mask", mask) == 1).all()):
            raise _not_yet("generate() with padding in attention_mask (batches of unequal length)")
        config = replace(generation_config or self.generation_config or GenerationConfig())
        unused = [k for k in kwargs if k not in _FIELDS]
        if unused:
            raise ValueError(f"The following `model_kwargs` are not used by the model: {unused}")
        for k, v in kwargs.items():
            setattr(config, k, v)
        if config.num_beams != 1:
            raise _not_yet("generate() with num_beams > 1 (beam search)")
        if not config.use_cache:
            raise _not_yet("generate(use_cache=False)")

        batch, prompt = ids.shape
        total = prompt + config.max_new_tokens if config.max_new_tokens is not None else config.max_length
        steps = total - prompt
        eos = config.eos_token_id
        eos = set() if eos is None else {eos} if isinstance(eos, int) else set(eos)
        pad = config.pad_token_id
        if pad is None and eos:
            pad = min(eos)
        rows = [list(r) for r in ids.tolist()]
        if streamer is not None:
            streamer.put(ids)
        if steps <= 0:
            if streamer is not None:
                streamer.end()
            return ids

        with no_grad():
            cache = self.make_cache(batch, total)
            history = ids if config.repetition_penalty not in (None, 1.0) else None
            unfinished = [True] * batch
            out = self(ids, past_key_values=cache, logits_to_keep=1)
            for step in range(steps):
                logits = out.logits.reshape(batch, -1)
                nxt = sampling.next_token(logits, history, do_sample=config.do_sample, temperature=config.temperature,
                                          top_k=config.top_k, top_p=config.top_p,
                                          repetition_penalty=config.repetition_penalty)
                toks = nxt.tolist()
                for r in range(batch):
                    if not unfinished[r]:
                        toks[r] = pad
                    elif toks[r] in eos:
                        unfinished[r] = False
                    rows[r].append(toks[r])
                if streamer is not None:
                    streamer.put(tensor(toks))
                if not any(unfinished) or step == steps - 1:
                    break
                step_ids = nxt.reshape(batch, 1) if all(unfinished) or pad is None else tensor([[t] for t in toks])
                if history is not None:
                    history = cat([history, step_ids], 1)
                out = self(step_ids, past_key_values=cache, logits_to_keep=1)
        if streamer is not None:
            streamer.end()
        return tensor(rows)
