"""Pattern separation: a sparse expansion layer.

    from grilly.snn import DentateGyrus

    dg = DentateGyrus(in_dim=32, out_dim=1024, sparsity=0.02)
    codes = dg(x)                       # (batch, 1024), 20 non-zeros a row

The dentate gyrus takes a few thousand entorhinal inputs onto a million
granule cells, of which very few fire at once. Expanding into a wider space
and then keeping only the strongest handful makes two similar inputs come
out with **non-overlapping** codes — which is what lets a downstream
associative memory store them as separate things rather than blend them.
That is pattern separation, and it is the one piece of v1's "capsule"
group that was not a linear layer wearing a different name (see
``docs/v1-migration.md``).

It measurably does that. Mean cosine over 40 pairs, 32 in, 1024 out, 2%
sparse, a pair made by adding noise to one vector:

| input cos | code cos |
|---:|---:|
| 0.9998 | 0.9823 |
| 0.9952 | 0.9413 |
| 0.9587 | 0.8434 |
| 0.8555 | 0.6539 |
| 0.6913 | 0.4435 |
| 0.4451 | 0.2191 |

The effect is an average, not a guarantee on any one pair: two inputs that
happen to pick the same winners come out as similar as they went in.

Three things happen, in order:

1. a **sparse random projection** — only ``sparsity`` of the weights are
   non-zero at all, so each output unit listens to a few inputs rather
   than all of them;
2. **k-winners-take-all** on the magnitude: the ``sparsity * out_dim``
   largest ``|activation|`` survive with their values, the rest go to
   zero. Note it ranks by magnitude and keeps the *signed* value, which is
   v1's rule;
3. **unit normalization** of what is left, so the code's length does not
   carry the input's.

Gradients flow to the surviving units only — which is what k-WTA means,
and is why this trains at all rather than being a fixed encoder.
"""

from __future__ import annotations

import builtins as _builtins

import grilly as _g
from .. import Tensor
from ..nn import Module, Parameter

__all__ = ["DentateGyrus"]


class DentateGyrus(Module):
    """Sparse expansion with k-winners-take-all.

    ``sparsity`` sets both how many weights are non-zero and how many
    outputs survive, as v1 had it: ``k = max(1, int(out_dim * sparsity))``.
    """

    def __init__(self, in_dim: _builtins.int = 32, out_dim: _builtins.int = 128,
                 sparsity: _builtins.float = 0.02, *, generator=None) -> None:
        super().__init__()
        if not 0.0 < sparsity <= 1.0:
            raise ValueError(f"DentateGyrus: sparsity must be in (0, 1], got {sparsity}")
        self.in_dim = _builtins.int(in_dim)
        self.out_dim = _builtins.int(out_dim)
        self.sparsity = _builtins.float(sparsity)
        self.k = _builtins.max(1, _builtins.int(self.out_dim * self.sparsity))
        self.weight = Parameter(self._sparse_weight(generator))

    def _sparse_weight(self, generator) -> Tensor:
        """``(out_dim, in_dim)`` with a ``sparsity`` fraction of entries
        drawn Xavier-uniform and the rest exactly zero."""
        total = self.out_dim * self.in_dim
        keep = _builtins.max(1, _builtins.int(total * self.sparsity))
        limit = (6.0 / (self.in_dim + self.out_dim)) ** 0.5
        values = _g.rand(total, generator=generator) * (2 * limit) - limit
        # the `keep` largest of a uniform draw is a uniform random subset
        chosen = _g.topk(_g.rand(total, generator=generator), keep).indices
        mask = _g.zeros(total).scatter(0, chosen, 1.0)
        return (values * mask).reshape(self.out_dim, self.in_dim)

    def forward(self, x) -> Tensor:
        """``(..., in_dim)`` to ``(..., out_dim)``, ``k`` non-zeros and unit length."""
        activations = _g.nn.functional.linear(x, self.weight)
        if activations.shape[-1] != self.out_dim:
            raise ValueError(f"DentateGyrus: expected (..., {self.in_dim}), got {tuple(x.shape)}")
        kept = _g.topk(activations.abs(), self.k, dim=-1).indices
        mask = _g.zeros(*activations.shape).scatter(-1, kept, 1.0)
        sparse = activations * mask
        norm = sparse.pow(2).sum(-1, keepdim=True).sqrt()
        # a row that came out all-zero has no direction to normalize to
        return sparse / _g.where(norm > 1e-8, norm, _g.full(tuple(norm.shape), 1.0))

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, sparsity={self.sparsity}, k={self.k}"
