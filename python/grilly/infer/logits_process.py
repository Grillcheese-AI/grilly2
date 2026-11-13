"""Logits processors for :meth:`GenerationMixin.generate`, with transformers'
names and call signature (``transformers.generation.logits_process``)::

    ids = model.generate(input_ids, max_new_tokens=20,
                         logits_processor=[SuppressTokensLogitsProcessor([13, 42])],
                         prefix_allowed_tokens_fn=lambda batch_id, sent: [5, 6, 7])

A processor is any callable ``(input_ids, scores) -> scores``: ``input_ids``
(rows, length) int64 are each running row's tokens so far, prompt first;
``scores`` (rows, vocab) float32 are that step's next-token logits. Both are
grilly tensors on the device, and a processor should keep them there:
reading the logits back to the host every step is what the hook exists to
avoid. ``generate`` runs the processors in order after the repetition penalty
and before temperature, top-k, top-p and the draw, as transformers does.

A left-padded batch drops rows as they finish, so a processor sees only the
rows still running; ``prefix_allowed_tokens_fn`` is still called with each
row's index in the batch as it was given.

A token is removed by setting its score to ``-inf``, as transformers does. A
row with every score at ``-inf`` has nothing to draw: as in transformers, that
is the caller's error.
"""

from __future__ import annotations

import math

from .. import Tensor, full, int64, tensor

__all__ = [
    "AllowedTokensLogitsProcessor",
    "LogitsProcessor",
    "LogitsProcessorList",
    "PrefixConstrainedLogitsProcessor",
    "SuppressTokensLogitsProcessor",
]


class LogitsProcessor:
    """The base transformers gives its processors: ``__call__(input_ids,
    scores) -> scores``. Subclassing it is optional; any callable works."""

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__} is a LogitsProcessor without __call__")


class LogitsProcessorList(list):
    """Processors applied in order, each to what the one before returned."""

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


class _TokenMask(LogitsProcessor):
    """Adds a fixed (vocab,) mask of 0 and -inf to every row. The mask is
    built on the device once per vocabulary size and reused at every step."""

    def __init__(self, token_ids, keep: bool):
        self.token_ids = sorted({int(t) for t in token_ids})
        if keep and not self.token_ids:
            raise ValueError("AllowedTokensLogitsProcessor needs at least one allowed token id")
        self._keep = keep
        self._masks: dict = {}

    def mask(self, vocab_size: int) -> Tensor:
        m = self._masks.get(vocab_size)
        if m is None:
            if self.token_ids and not 0 <= self.token_ids[0] <= self.token_ids[-1] < vocab_size:
                raise ValueError(f"token ids must lie in [0, {vocab_size}), got {self.token_ids[0]}..{self.token_ids[-1]}")
            fill, mark = (-math.inf, 0.0) if self._keep else (0.0, -math.inf)
            m = full((vocab_size,), fill)
            if self.token_ids:
                m.index_fill_(0, tensor(self.token_ids, dtype=int64), mark)
            self._masks[vocab_size] = m
        return m

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        return scores + self.mask(scores.shape[-1])


class AllowedTokensLogitsProcessor(_TokenMask):
    """Only ``allowed_ids`` can be drawn, at every step and in every row. (No
    transformers class does exactly this; it is ``prefix_allowed_tokens_fn``
    returning the same list each time, with the mask built once.)"""

    def __init__(self, allowed_ids):
        super().__init__(allowed_ids, keep=True)


class SuppressTokensLogitsProcessor(_TokenMask):
    """transformers' ``SuppressTokensLogitsProcessor``: ``suppress_tokens``
    can never be drawn."""

    def __init__(self, suppress_tokens):
        super().__init__(suppress_tokens, keep=False)


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    """transformers' processor for ``prefix_allowed_tokens_fn(batch_id,
    input_ids) -> list[int]``: each row may draw only the ids its call
    returns. ``input_ids`` is that row's tokens so far (1-D). The mask is
    built on the device each step from the rows' lists."""

    def __init__(self, prefix_allowed_tokens_fn, num_beams: int = 1):
        if num_beams != 1:
            raise ValueError("PrefixConstrainedLogitsProcessor: grilly's generate() has no beam search (num_beams=1)")
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        #: each running row's index in the batch as given; ``generate`` sets
        #: it when rows are dropped (None: row i is batch row i).
        self.batch_ids: list | None = None

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        rows, vocab = scores.shape
        batch_ids = self.batch_ids if self.batch_ids is not None else list(range(rows))
        allowed = []
        for i in range(rows):
            ids = [int(t) for t in self._prefix_allowed_tokens_fn(batch_ids[i], input_ids[i])]
            if not ids:
                raise ValueError(f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_ids[i]}. "
                                 "This means that the constraint is unsatisfiable. Please check your implementation "
                                 "of `prefix_allowed_tokens_fn` ")
            if min(ids) < 0 or max(ids) >= vocab:
                raise ValueError(f"`prefix_allowed_tokens_fn` returned ids outside [0, {vocab}) for batch ID {batch_ids[i]}")
            allowed.append(ids)
        width = max(len(a) for a in allowed)
        index = tensor([a + [a[0]] * (width - len(a)) for a in allowed], dtype=int64)  # a repeat writes the same 0
        mask = full((rows, vocab), -math.inf)
        mask.scatter_(1, index, value=0.0)
        return scores + mask


def as_list(logits_processor, prefix_allowed_tokens_fn) -> LogitsProcessorList:
    """``generate``'s two arguments as one list, in transformers' order: the
    prefix constraint, then the caller's processors. Empty when both are None."""
    out = LogitsProcessorList()
    if prefix_allowed_tokens_fn is not None:
        out.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn))
    if logits_processor is not None:
        out.extend(logits_processor if isinstance(logits_processor, (list, tuple)) else [logits_processor])
    for p in out:
        if not callable(p):
            raise TypeError(f"a logits processor must be callable (input_ids, scores) -> scores, got {type(p).__name__}")
    return out
