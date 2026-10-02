"""A decode loop with Hugging Face's ``generate()`` signature (CLAUDE.md
§11, step 4)::

    ids = model.generate(input_ids, max_new_tokens=50, do_sample=True,
                         temperature=0.7, top_k=20, top_p=0.8)

``GenerationMixin.generate`` takes ``inputs`` (or ``input_ids=``), a
``generation_config`` and HF's generation keyword arguments, with HF's
defaults. The supported ones are ``max_length``, ``max_new_tokens``,
``do_sample``, ``temperature``, ``top_k``, ``top_p``, ``repetition_penalty``,
``eos_token_id`` and ``pad_token_id``. It also takes ``attention_mask`` and
``streamer``. Anything else raises, naming it.

**Batches of unequal length** go in left-padded, as transformers wants for a
decoder-only model: ``attention_mask`` 0 on each row's leading padding, 1 on
its tokens. Right padding is refused, since each row continues from its
last position. A padded batch runs on the model's
:class:`~grilly.infer.cache.PaddedKVCache` and **drops rows as they
finish**: once a quarter of the running rows have reached EOS, the cache,
the per-row state and any module holding per-row data (``select_rows``) are
cut down to the rows still going, so a batch stops paying for streams that
are done. The returned sequences are unchanged by it: every row is still
there, padded after its EOS. ``compact_finished=True`` (grilly's own field)
runs an equal-length batch the same way. Under sampling, dropping rows
changes which random numbers each row draws, so it samples differently
from the same seed without it; greedy decoding is unaffected.

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
    #: grilly's own: drop finished rows from an equal-length batch too (a
    #: left-padded batch always drops them). Not a transformers field.
    compact_finished: bool = False


_FIELDS = {f.name for f in fields(GenerationConfig)}


class GenerationMixin:
    """``generate`` for a causal LM whose forward is
    ``forward(input_ids, past_key_values=None, logits_to_keep=0)`` and which
    has ``make_cache(batch_size, max_length)``. A model with
    ``supports_padding = True`` also takes ``make_cache(..., keep=mask)``,
    returning a cache with ``select_rows``, and runs left-padded batches."""

    supports_padding = False

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
        keep = None if mask is None else _left_padding(_require_tensor("attention_mask", mask), tuple(ids.shape))
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
        padded = keep is not None or bool(config.compact_finished)
        if padded and not self.supports_padding:
            raise _not_yet(f"generate() with padding in attention_mask (batches of unequal length) "
                           f"or compact_finished for {type(self).__name__}")

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
            if padded:
                cache = self.make_cache(batch, total, keep=keep if keep is not None else [[1] * prompt] * batch)
            else:
                cache = self.make_cache(batch, total)
            history = ids if config.repetition_penalty not in (None, 1.0) else None
            unfinished = [True] * batch
            active = list(range(batch))  # the batch row each running row is
            undo: list = []
            try:
                out = self(ids, past_key_values=cache, logits_to_keep=1)
                for step in range(steps):
                    n = len(active)
                    logits = out.logits.reshape(n, -1)
                    nxt = sampling.next_token(logits, history, do_sample=config.do_sample,
                                              temperature=config.temperature, top_k=config.top_k,
                                              top_p=config.top_p, repetition_penalty=config.repetition_penalty)
                    drawn = nxt.tolist()
                    toks = [pad] * batch
                    for i, r in enumerate(active):
                        if unfinished[r]:
                            toks[r] = drawn[i]
                            if drawn[i] in eos:
                                unfinished[r] = False
                    for r in range(batch):
                        rows[r].append(toks[r])
                    if streamer is not None:
                        streamer.put(tensor(toks))
                    if not any(unfinished) or step == steps - 1:
                        break
                    going = [i for i, r in enumerate(active) if unfinished[r]]
                    if padded and 4 * len(going) <= 3 * n:  # a quarter or more of the rows are done
                        index = tensor(going)
                        self._select_rows(cache, index, undo)
                        if history is not None:
                            history = history.index_select(0, index)
                        active = [active[i] for i in going]
                        step_ids = tensor([[drawn[i]] for i in going])
                    elif len(going) == n or pad is None:
                        step_ids = nxt.reshape(n, 1)
                    else:
                        step_ids = tensor([[toks[r]] for r in active])
                    if history is not None:
                        history = cat([history, step_ids], 1)
                    out = self(step_ids, past_key_values=cache, logits_to_keep=1)
            finally:
                for restore in reversed(undo):  # modules get their whole batch back
                    restore()
        if streamer is not None:
            streamer.end()
        return tensor(rows)

    def _select_rows(self, cache, index, undo: list) -> None:
        """Keep only batch rows ``index`` (int64) of everything that holds a
        row per stream: the cache, and every submodule with
        ``select_rows`` (a LoRA population's per-row noise). A module's
        ``select_rows`` returns its undo, collected in ``undo`` so
        ``generate`` hands the modules back as it found them. A model with
        more per-row state extends this."""
        cache.select_rows(index)
        for module in self.modules():
            select = getattr(module, "select_rows", None) if module is not self else None
            if select is not None:
                undo.append(select(index))


def _left_padding(mask: Tensor, shape: tuple):
    """The rows of a 0/1 ``attention_mask``, or None when it is all ones.

    Padding must be on the left — zeros, then ones, and at least one one —
    because generation continues each row from its last position, and with
    right padding that position is a pad."""
    if tuple(mask.shape) != shape:
        raise ValueError(f"attention_mask is {tuple(mask.shape)} and the inputs are {shape}")
    rows = mask.tolist()
    if all(v == 1 for row in rows for v in row):
        return None
    for i, row in enumerate(rows):
        if any(v not in (0, 1) for v in row):
            raise ValueError("attention_mask must hold only 0s and 1s")
        first = row.index(1) if 1 in row else len(row)
        if first == len(row) or any(v != 1 for v in row[first:]):
            raise ValueError(
                f"attention_mask row {i} is not left-padded (zeros, then ones, at least one one): "
                "generate() continues each row from its last position, so padding goes on the left")
    return [[int(v) for v in row] for row in rows]
