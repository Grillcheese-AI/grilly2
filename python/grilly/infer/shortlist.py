"""A vocabulary shortlist head: greedy decoding's next token without scoring
every row of a large output head.

A model with a 128,000-token vocabulary scores all 128,000 rows of its head
for every token it writes. ``ShortlistHead`` scores only the union of two
short lists and picks the best of those exactly (fp32 dot products against
the trained rows):

- a **context list** — the head's rows grouped into clusters, each with a
  centroid and a token list; the clusters ranked by ``x . centroid`` (score
  descending, index ascending on a tie), their tokens in that order without
  repeats until ``context`` distinct ones, plus the previous token's
  successors when given;
- the ``near`` rows **nearest the state by Hamming distance** between sign
  bits (``x > 0`` against each row's), ties to the lower row.

The step runs on the GPU from the state to the token, in eight dispatches,
and reads back 16 bytes::

    head = ShortlistHead(model.lm_head.weight, centroids, cluster_ids,
                         cluster_offsets, successors, near=2048, context=8192)
    token = head.argmax(hidden[-1], prev=last_token)

Building the clusters is the caller's business; this is the machinery that
runs a route once it exists. ``ranks`` bounds how many clusters are scanned;
a state whose first ``ranks`` clusters hold fewer than ``context`` distinct
tokens takes all of them, and ``last.short`` says so.

The ranking uses the GPU's fp32 scores, so clusters whose scores are within
rounding of each other may rank in the other order than numpy's would; the
union is otherwise exact (distances are integers, the cut is a count).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as _np

import grilly as _g
from .. import Tensor, _C, _require_tensor

__all__ = ["Choice", "ShortlistHead"]


class Choice(NamedTuple):
    token: int
    rows: int          # how many rows were scored
    short: bool        # the scanned clusters held fewer than ``context`` tokens


def _int32(x, what: str) -> Tensor:
    if isinstance(x, Tensor):
        return x if x.dtype == _g.int32 else x.to(_g.int32)
    a = _np.asarray(x)
    if a.size and (a.min() < -(2 ** 31) or a.max() >= 2 ** 31):
        raise ValueError(f"{what}: values do not fit int32")
    return _g.from_numpy(_np.ascontiguousarray(a, dtype=_np.int32))


def _float32(x, what: str) -> Tensor:
    if isinstance(x, Tensor):
        return x if x.dtype == _g.float32 else x.to(_g.float32)
    return _g.from_numpy(_np.ascontiguousarray(x, dtype=_np.float32))


class ShortlistHead:
    """See the module docstring. ``weight`` (vocab, dim) is kept by reference;
    the other arrays may be numpy or grilly tensors."""

    def __init__(self, weight, centroids, cluster_ids, cluster_offsets, successors=None,
                 *, near: int = 2048, context: int = 8192, ranks: int = 256):
        weight = _require_tensor("weight", weight)
        self._head = _C.ShortlistHead(
            weight._t, _float32(centroids, "centroids")._t, _int32(cluster_ids, "cluster_ids")._t,
            _int32(cluster_offsets, "cluster_offsets")._t,
            None if successors is None else _int32(successors, "successors")._t,
            int(near), int(context), int(ranks),
        )
        self.last: Optional[Choice] = None

    @property
    def shape(self):
        return tuple(self._head.shape)

    def codes(self) -> Tensor:
        """The rows' packed sign bits, (vocab, dim / 32) int32."""
        return Tensor._wrap(self._head.codes())

    def argmax(self, x, prev: Optional[int] = None) -> int:
        """The next token for state ``x`` (dim elements) after ``prev``."""
        x = _require_tensor("x", x)
        self.last = Choice(*self._head.argmax(x._t, None if prev is None else int(prev)))
        return self.last.token

    def union(self, x, prev: Optional[int] = None):
        """The rows a step would score, ascending (int64 numpy), and whether
        the context list was short. For checking a route, not for decoding."""
        x = _require_tensor("x", x)
        rows, short = self._head.union(x._t, None if prev is None else int(prev))
        return _np.asarray(rows, dtype=_np.int64), bool(short)

    def profile(self, x, prev: Optional[int], stages: int) -> None:
        """Run the step's first ``stages`` dispatches (of 8) and wait."""
        self._head.profile(_require_tensor("x", x)._t, None if prev is None else int(prev), int(stages))
