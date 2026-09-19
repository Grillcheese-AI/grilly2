"""Next-token sampling on the GPU, with Hugging Face's generation
parameters and processors (transformers 5.5, ``generate()``'s order):

- ``repetition_penalty`` over ``input_ids``: ``x < 0 ? x * r : x / r``;
- greedy (``do_sample=False``): the first index of the maximum;
- sampling: ``temperature``, then ``top_k``, then ``top_p``, then a draw
  from the softmax of what is left (as ``torch.multinomial``).

One kernel does all of it, on the device; the logits never come back to
the host. The draw uses grilly's generator, so a seed repeats its tokens
in grilly, not torch's numbers. Penalized and scaled scores equal HF's
bit for bit, so top-k keeps exactly HF's tokens. top-p keeps HF's tokens
except where a token's cumulative probability is within float32 rounding
of ``1 - top_p``.

The defaults are ``GenerationConfig``'s: greedy, ``temperature=1.0``,
``top_k=50``, ``top_p=1.0``, ``repetition_penalty=1.0``. ``None`` turns a
filter off, as in HF.
"""

from __future__ import annotations

from .. import Tensor, _C, _generator, _require_tensor

__all__ = ["next_token", "probabilities", "processed_logits"]


def _args(logits, input_ids, temperature, top_k, top_p, repetition_penalty):
    x = _require_tensor("logits", logits)
    ids = None if input_ids is None else _require_tensor("input_ids", input_ids)._t
    temperature = 1.0 if temperature is None else float(temperature)
    top_k = 0 if top_k is None else int(top_k)
    if top_k < 0:
        raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
    top_p = 1.0 if top_p is None else float(top_p)
    penalty = 1.0 if repetition_penalty is None else float(repetition_penalty)
    return x._t, ids, temperature, top_k, top_p, penalty


def next_token(logits, input_ids=None, *, do_sample=False, temperature=1.0, top_k=50, top_p=1.0,
               repetition_penalty=1.0, generator=None) -> Tensor:
    """The next token for each row of ``logits`` (batch, vocab), as int64
    (batch,). ``input_ids`` (batch, length) are the tokens so far, for the
    repetition penalty."""
    x, ids, t, k, p, r = _args(logits, input_ids, temperature, top_k, top_p, repetition_penalty)
    return Tensor._wrap(_C.next_tokens(x, ids, bool(do_sample), t, k, p, r, _generator(generator)._g))


def processed_logits(logits, input_ids=None, *, do_sample=False, temperature=1.0, repetition_penalty=1.0) -> Tensor:
    """The scores after the repetition penalty and, when sampling, the
    temperature: what top-k and top-p filter, or what greedy decoding takes
    the argmax of."""
    x, ids, t, _, _, r = _args(logits, input_ids, temperature, None, None, repetition_penalty)
    return Tensor._wrap(_C.processed_scores(x, ids, bool(do_sample), t, r))


def probabilities(logits, input_ids=None, *, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0) -> Tensor:
    """The distribution a sampled token is drawn from, (batch, vocab): the
    softmax after top-k and top-p, zero where a token was dropped."""
    x, ids, t, k, p, r = _args(logits, input_ids, temperature, top_k, top_p, repetition_penalty)
    return Tensor._wrap(_C.sample_probabilities(x, ids, t, k, p, r))
