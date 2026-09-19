"""Non-reentrant gradient checkpointing and saved-tensor hooks.

Modelled on CubbyLLM's test_grad_checkpoint_with_memory_flows_grads_to_memory_params:
parameters used *inside* the checkpointed function but not passed to it must
get the same gradients as without checkpointing. Here that is exact (same
kernels, same order), stricter than CubbyLLM's atol=1e-4.
"""

import numpy as np
import pytest
import torch

import grilly
import grilly.nn.functional as GF
from grilly.autograd.graph import saved_tensors_hooks
from grilly.utils.checkpoint import CheckpointError, checkpoint


def rand(*shape, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, shape).astype(np.float32)


class Block:
    """Two linear layers; w2 is captured, never passed as an argument."""

    def __init__(self, seed):
        self.w1 = grilly.tensor(rand(16, 8, seed=seed), requires_grad=True)
        self.w2 = grilly.tensor(rand(8, 16, seed=seed + 1), requires_grad=True)
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return GF.linear(grilly.relu(GF.linear(x, self.w1)), self.w2)


def run(checkpointed):
    blocks = [Block(10), Block(20)]
    x = grilly.tensor(rand(4, 8, seed=1), requires_grad=True)
    h = x
    for b in blocks:
        h = checkpoint(b, h, use_reentrant=False) if checkpointed else b(h)
    (h + h).backward(grilly.tensor(rand(4, 8, seed=2)))
    grads = [x.grad.numpy()] + [p.grad.numpy() for b in blocks for p in (b.w1, b.w2)]
    return grads, [b.calls for b in blocks]


def test_checkpoint_gives_identical_grads_including_captured_params():
    plain, plain_calls = run(False)
    ckpt, ckpt_calls = run(True)
    assert plain_calls == [1, 1]
    assert ckpt_calls == [2, 2]                     # each block recomputed once
    for a, b in zip(plain, ckpt):
        assert np.abs(b).sum() > 0
        np.testing.assert_array_equal(a, b)


def test_checkpoint_keeps_only_indices():
    """Inside checkpoint, its hooks (the innermost) take every save and keep
    an index, not the tensor; hooks outside see none of those saves."""
    outer = []
    x = grilly.tensor(rand(4, 8, seed=3), requires_grad=True)
    with saved_tensors_hooks(lambda t: outer.append(t) or t, lambda t: t):
        Block(30)(x)
        plain = len(outer)                          # linear: x, w1; relu: out; linear: h, w2
        checkpoint(Block(30), x, use_reentrant=False)
    assert plain == 5 and len(outer) == plain

    from grilly.utils.checkpoint import _Frame
    frame = _Frame(None, (), {}, None)
    assert [frame.pack(grilly.ones(3)) for _ in range(3)] == [0, 1, 2]


def test_recompute_restores_rng_state():
    w = grilly.tensor(rand(8, 8, seed=5), requires_grad=True)
    draws = []

    def noisy(x):
        n = grilly.randn(4, 8)
        draws.append(n.numpy())
        return grilly.relu(GF.linear(x + n, w))

    x = grilly.tensor(rand(4, 8, seed=6), requires_grad=True)
    y = checkpoint(noisy, x, use_reentrant=False)
    grilly.randn(100)                               # move the global stream on
    y.backward(grilly.ones(4, 8))
    np.testing.assert_array_equal(draws[0], draws[1])


def test_changed_recomputation_raises_checkpoint_error():
    w = grilly.tensor(rand(8, 8, seed=7), requires_grad=True)
    calls = []

    def flaky(x):
        calls.append(1)
        h = GF.linear(x, w)
        return grilly.relu(h) if len(calls) == 1 else h  # saves differ on rerun

    y = checkpoint(flaky, grilly.tensor(rand(2, 8), requires_grad=True), use_reentrant=False)
    with pytest.raises(CheckpointError, match="different number of tensors"):
        y.backward(grilly.ones(2, 8))
    assert issubclass(CheckpointError, RuntimeError)
    assert issubclass(torch.utils.checkpoint.CheckpointError, RuntimeError)


def test_checkpoint_under_no_grad_just_runs():
    block = Block(40)
    with grilly.no_grad():
        y = checkpoint(block, grilly.ones(2, 8), use_reentrant=False)
    assert not y.requires_grad and block.calls == 1


@pytest.mark.parametrize("kwargs", [{}, {"use_reentrant": True}])
def test_reentrant_is_not_implemented(kwargs):
    with pytest.raises(NotImplementedError, match="use_reentrant=False"):
        checkpoint(Block(0), grilly.ones(2, 8), **kwargs)


def test_saved_tensors_hooks_see_saves_like_torch():
    """Both libraries pass every saved tensor through the hooks and give
    backward the unpacked values."""
    def run(lib, F):
        seen = []
        x = lib.ones(2, 3, requires_grad=True)
        w = lib.ones(4, 3, requires_grad=True)
        with lib.autograd.graph.saved_tensors_hooks(lambda t: seen.append(tuple(t.shape)) or t,
                                                    lambda t: t):
            y = lib.relu(F.linear(x, w))
        y.backward(lib.ones(2, 4))
        return seen, x.grad.detach().numpy()
    g_seen, g_grad = run(grilly, GF)
    t_seen, t_grad = run(torch, torch.nn.functional)
    np.testing.assert_array_equal(g_grad, t_grad)
    # torch's linear decomposes and saves differently; both must save x and w
    assert (2, 3) in g_seen and (4, 3) in g_seen and (2, 3) in t_seen
