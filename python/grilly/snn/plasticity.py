"""Local learning rules: Hebbian, Oja, and spike-timing-dependent.

Learning without a loss. Each of these changes a weight from what the two
neurons it connects are doing, and nothing else — no global error, no
backward pass. That is what makes them usable on a spiking network, where
the gradient has to be invented anyway (:mod:`grilly.snn.surrogate`), and
what makes them plausible as accounts of biological learning.

    from grilly.snn import plasticity

    learner = plasticity.STDPLearner(pre_dim=100, post_dim=10)
    for t in range(steps):
        pre, post = layer_input[t], layer_output[t]
        weight = weight + learner.step(pre, post)

Three rules, in order of how much they remember:

- :func:`hebbian_update` — "fire together, wire together". The weight
  follows the average product of the two activities. Unbounded on its own,
  so it takes a decay term.
- :func:`oja_update` — Hebb with the growth subtracted out. The weight
  vector converges to the first principal component of its input and stays
  unit length, without a decay to tune.
- :class:`STDPLearner` — the timing matters, not just the correlation.
  Pre firing *before* post strengthens; post before pre weakens. That
  asymmetry is what makes it different from Hebb, and it needs a trace of
  each side's recent spikes to see it.

No kernels: an outer product is a matmul and the traces are elementwise,
so these are ordinary tensor expressions. `grilly-snn`'s crate holds the
neuron step, which needed one.

**v1's `stdp_learning` was not STDP.** Its shader computed
``ltp += pre_t * post_t`` and ``ltd += post_t * pre_t`` — the same product
twice — so its update reduced to ``(lr_pot - lr_dep) * pre_t * post_t``,
which is Hebbian with a net rate and no timing asymmetry at all. The rule
here is the real one, and the old name raises rather than quietly changing
what a caller gets (``docs/v1-migration.md``).
"""

from __future__ import annotations

import builtins as _builtins

import grilly as _g
from .. import Tensor, _require_tensor

__all__ = ["STDPLearner", "hebbian_update", "oja_update", "stdp_update"]


def _flatten_activity(a: Tensor, what: str) -> Tensor:
    """``(B, T, d)`` or ``(B, d)`` or ``(d,)`` as ``(rows, d)``."""
    if a.dim() == 1:
        return a.reshape(1, a.shape[0])
    return a.reshape(a.numel() // a.shape[-1], a.shape[-1])


def hebbian_update(pre, post, weight, lr: _builtins.float = 0.01,
                   weight_decay: _builtins.float = 0.0) -> Tensor:
    """``W + lr * <post outer pre> - weight_decay * W``.

    ``pre`` and ``post`` are activities shaped ``(batch, time, dim)``, or
    anything whose last axis is the dimension; the correlation is averaged
    over everything before it. ``weight`` is ``(post_dim, pre_dim)``, the
    same way round as ``nn.Linear``'s.

    Without ``weight_decay`` this grows without bound — every correlated
    pair pushes the same way for ever. That is the rule's known failure and
    the reason :func:`oja_update` exists.
    """
    pre, post = _require_tensor("pre", pre), _require_tensor("post", post)
    weight = _require_tensor("weight", weight)
    p, q = _flatten_activity(pre, "pre"), _flatten_activity(post, "post")
    if p.shape[0] != q.shape[0]:
        raise ValueError(
            f"hebbian_update: pre has {p.shape[0]} samples and post has {q.shape[0]}"
        )
    if tuple(weight.shape) != (q.shape[1], p.shape[1]):
        raise ValueError(
            f"hebbian_update: weight must be ({q.shape[1]}, {p.shape[1]}), got {tuple(weight.shape)}"
        )
    correlation = (q.t() @ p) / p.shape[0]
    return weight + lr * correlation - weight_decay * weight


def oja_update(memory, x, eta: _builtins.float = 0.01) -> Tensor:
    """``m + eta * y * (x - y * m)`` with ``y = <m, x>``, per row.

    Oja's rule: Hebb, minus the term that would let the weight grow. What
    it subtracts is exactly the projection back along ``m``, so the row
    stays near unit length by itself and converges on the leading
    principal component of whatever ``x`` it is shown.

    ``memory`` is ``(n, dim)`` and ``x`` matches it, or is one ``(dim,)``
    vector shown to every row.
    """
    memory, x = _require_tensor("memory", memory), _require_tensor("x", x)
    if memory.dim() != 2:
        raise ValueError(f"oja_update: memory must be (n, dim), got {tuple(memory.shape)}")
    if x.dim() == 1:
        x = x.reshape(1, x.shape[0])
    if x.shape[-1] != memory.shape[1]:
        raise ValueError(
            f"oja_update: x must be (..., {memory.shape[1]}), got {tuple(x.shape)}"
        )
    y = (memory * x).sum(-1, keepdim=True)
    return memory + eta * y * (x - y * memory)


def stdp_update(pre_spike, post_spike, pre_trace, post_trace, *,
                trace_decay: _builtins.float = 0.95,
                lr_pot: _builtins.float = 0.01, lr_dep: _builtins.float = 0.01):
    """One step of trace-based STDP. Returns ``(delta_w, pre_trace, post_trace)``.

    The traces are updated first, then read::

        pre_trace  = decay * pre_trace  + pre_spike
        post_trace = decay * post_trace + post_spike
        delta_w    = lr_pot * (post_spike outer pre_trace)
                   - lr_dep * (post_trace outer pre_spike)

    The asymmetry is the whole rule. The first term fires when **post**
    spikes and asks how recently **pre** did — so pre-before-post
    potentiates. The second fires when **pre** spikes and asks how recently
    **post** did — so post-before-pre depresses. Swap the two and you get
    the same number both ways, which is what v1 did.

    Spikes are ``(batch, dim)`` or ``(dim,)``; the update is averaged over
    the batch.
    """
    pre_s = _flatten_activity(_require_tensor("pre_spike", pre_spike), "pre_spike")
    post_s = _flatten_activity(_require_tensor("post_spike", post_spike), "post_spike")
    pre_t = _flatten_activity(_require_tensor("pre_trace", pre_trace), "pre_trace")
    post_t = _flatten_activity(_require_tensor("post_trace", post_trace), "post_trace")
    if pre_s.shape[0] != post_s.shape[0]:
        raise ValueError(
            f"stdp_update: {pre_s.shape[0]} pre samples and {post_s.shape[0]} post samples"
        )

    pre_t = trace_decay * pre_t + pre_s
    post_t = trace_decay * post_t + post_s
    batch = pre_s.shape[0]
    potentiation = post_s.t() @ pre_t          # post fired: how recently did pre?
    depression = post_t.t() @ pre_s            # pre fired: how recently did post?
    delta = (lr_pot * potentiation - lr_dep * depression) / batch
    return delta, pre_t, post_t


class STDPLearner:
    """Trace-based STDP with the traces kept for you.

    ``step(pre_spike, post_spike)`` returns the weight update and carries
    the traces forward. :meth:`reset` clears them, and belongs wherever
    ``reset_net`` does — the traces are state in exactly the same way a
    membrane is.
    """

    def __init__(self, pre_dim: _builtins.int, post_dim: _builtins.int,
                 batch: _builtins.int = 1, trace_decay: _builtins.float = 0.95,
                 lr_pot: _builtins.float = 0.01, lr_dep: _builtins.float = 0.01) -> None:
        if not 0.0 <= trace_decay < 1.0:
            raise ValueError(f"trace_decay must be in [0, 1), got {trace_decay}")
        self.pre_dim = _builtins.int(pre_dim)
        self.post_dim = _builtins.int(post_dim)
        self.batch = _builtins.int(batch)
        self.trace_decay = _builtins.float(trace_decay)
        self.lr_pot = _builtins.float(lr_pot)
        self.lr_dep = _builtins.float(lr_dep)
        self.reset()

    def reset(self) -> None:
        """Forget both traces."""
        self.pre_trace = _g.zeros(self.batch, self.pre_dim)
        self.post_trace = _g.zeros(self.batch, self.post_dim)

    def step(self, pre_spike, post_spike) -> Tensor:
        """The weight update for this timestep, ``(post_dim, pre_dim)``."""
        delta, self.pre_trace, self.post_trace = stdp_update(
            pre_spike, post_spike, self.pre_trace, self.post_trace,
            trace_decay=self.trace_decay, lr_pot=self.lr_pot, lr_dep=self.lr_dep,
        )
        return delta

    def __repr__(self) -> str:
        return (f"STDPLearner(pre_dim={self.pre_dim}, post_dim={self.post_dim}, "
                f"batch={self.batch}, trace_decay={self.trace_decay}, "
                f"lr_pot={self.lr_pot}, lr_dep={self.lr_dep})")
