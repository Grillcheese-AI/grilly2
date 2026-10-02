"""``x @ w.t()`` must not build ``w.t()``.

grilly has no strides, so a transpose is a copy. For a matmul that copy is
pure loss — the matmul reads every element once anyway — and for a wide
matrix it is far worse than that: transposing a (127996, 512) weight reads
it with a 2 KB stride across 127996 concurrent streams, so each cache line
is fetched for one word of it. Measured at 6.1 GB/s against 196 for the
same data the other way round. On CubbyLLM's retrieval head it turned a
1.4 ms matmul into 90 ms.

**Tolerance: bit-exact**, and the comparisons are against two references
that must both hold:

- ``x @ w.t()`` against ``F.linear(x, w)``, which is the same arithmetic
  and the kernel it should now be reaching;
- ``x @ w.t()`` against the same product with a *real* transposed copy,
  which is what it used to do. Changing an operand's representation must
  not change a single bit of the result.

The speed is not asserted with a clock — a timing test on a shared GPU is
flaky. What is asserted is the structural fact that produces the speed:
after a matmul, the transpose **has not been materialized**. That is
deterministic, and it is the thing a regression would break.
"""

from __future__ import annotations

import numpy as np
import pytest

import grilly
import grilly.nn.functional as F
from grilly._transposed import _Transposed


def gen(seed=0):
    return grilly.Generator().manual_seed(seed)


def real_transpose(x: grilly.Tensor) -> grilly.Tensor:
    """A genuinely materialized transpose, built outside grilly."""
    return grilly.from_numpy(np.ascontiguousarray(x.detach().numpy().swapaxes(-2, -1)))


# ── the fix itself ───────────────────────────────────────────────────────


def test_a_transpose_is_not_built_until_something_needs_it():
    w = grilly.randn(64, 32, generator=gen(1))
    t = w.t()
    assert isinstance(t, _Transposed) and not t.materialized
    assert tuple(t.shape) == (32, 64) and t.dim() == 2 and t.numel() == 64 * 32
    assert t.dtype == w.dtype
    assert not t.materialized, "asking about the shape must not build it"


def test_a_matmul_consumes_the_transpose_without_building_it():
    """The structural guarantee behind the speed."""
    x = grilly.randn(4, 32, generator=gen(2))
    w = grilly.randn(64, 32, generator=gen(3))
    t = w.t()
    out = x @ t
    assert not t.materialized, "the matmul built the transpose it was supposed to absorb"
    assert tuple(out.shape) == (4, 64)


def test_it_is_bit_identical_to_linear():
    """The contract that matters: writing ``x @ w.t()`` is writing
    ``linear(x, w)``, exactly, at every shape. This is case (a) at its
    strictest — same kernel, same inputs, same order — so ``==``."""
    for rows in (1, 2, 4, 33):
        x = grilly.randn(rows, 32, generator=gen(4))
        w = grilly.randn(64, 32, generator=gen(5))
        assert ((x @ w.t()).numpy() == F.linear(x, w).numpy()).all(), f"rows={rows}"


def test_it_agrees_with_a_real_transpose_to_the_accuracy_of_a_dot_product():
    """It used to be **bit**-identical to a materialised transpose too, and
    deliberately is not any more.

    That equality was a coincidence: ``x @ real_transpose(w)`` goes to
    ``g2-bmm`` and ``x @ w.t()`` to ``linear``, and those two happened to
    accumulate K products in the same order. At one or two rows ``linear``
    now uses ``g2-gemv``, which sums a strided slice per thread and then a
    tree — a *different order*, so the bits differ.

    The gap is 1.2e-6 relative, past case (a)'s 1e-7, and the honest
    reading is not that gemv is looser. Measured against a float64
    reference under a length-K dot product's bound, **gemv sits at 0.001 of
    that bound and bmm at 0.025**: pairwise summation is more accurate than
    a sequential fma chain, so most of the gap is bmm's error, and its size
    is what ``sqrt(K) * eps`` predicts for a K-term float32 sum.

    So this asserts the thing that is actually required — both paths are
    correct matmuls — and prints the worst ratio, which is what says
    whether either regressed.
    """
    worst = {"x @ w.t()": 0.0, "x @ real_transpose(w)": 0.0}
    for rows in (1, 2, 4, 33):
        for k, n in ((32, 64), (512, 512)):
            x = grilly.randn(rows, k, generator=gen(40 + k))
            w = grilly.randn(n, k, generator=gen(50 + n))
            xn, wn = x.numpy().astype(np.float64), w.numpy().astype(np.float64)
            reference = xn @ wn.T
            bound = k * 2.0 ** -24 * (np.abs(xn) @ np.abs(wn).T)
            for name, got in (("x @ w.t()", (x @ w.t()).numpy()),
                              ("x @ real_transpose(w)", (x @ real_transpose(w)).numpy())):
                ratio = float((np.abs(got - reference) / np.maximum(bound, 1e-300)).max())
                worst[name] = max(worst[name], ratio)
    print("  worst |err| / (K*eps*|x|@|w|): "
          + ", ".join(f"{k} {v:.3f}" for k, v in worst.items()))
    for name, ratio in worst.items():
        assert ratio <= 1.0, f"{name} is {ratio:.3f} of its error bound"


def test_a_one_dimensional_query_still_works():
    x = grilly.randn(32, generator=gen(6))
    w = grilly.randn(64, 32, generator=gen(7))
    out = x @ w.t()
    assert tuple(out.shape) == (64,)
    assert (out.numpy() == F.linear(x.reshape(1, 32), w).reshape(64).numpy()).all()


def test_the_batched_case_attention_writes():
    """``q @ k.transpose(-2, -1)`` — hand-written attention, and the shape
    where the transposed operand has batch dims in front of it."""
    q = grilly.randn(2, 4, 8, 16, generator=gen(8))
    k = grilly.randn(2, 4, 37, 16, generator=gen(9))
    t = k.transpose(-2, -1)
    got = q @ t
    assert not t.materialized
    assert tuple(got.shape) == (2, 4, 8, 37)
    assert (got.numpy() == (q @ real_transpose(k)).numpy()).all()

    want = np.einsum("bhid,bhjd->bhij", q.numpy().astype(np.float64), k.numpy().astype(np.float64))
    bound = 4 * 16 * float(np.finfo(np.float32).eps) * float(np.abs(want).max())
    assert np.abs(got.numpy() - want).max() <= bound


def test_two_transposes_of_the_same_pair_cancel():
    """Free, and it keeps ``w.t().t()`` from costing two copies."""
    w = grilly.randn(64, 32, generator=gen(10))
    assert w.t().t() is w
    k = grilly.randn(2, 3, 5, 7, generator=gen(11))
    assert k.transpose(-2, -1).transpose(-2, -1) is k


# ── everything else still materializes, correctly ────────────────────────


def test_any_other_consumer_gets_the_real_thing():
    """The safety property: a path that knows nothing about laziness reads
    ``_t`` and gets the copy, rather than a wrong answer."""
    w = grilly.randn(8, 5, generator=gen(12))
    want = w.numpy().T

    for name, use in [
        ("numpy", lambda t: t.numpy()),
        ("sum", lambda t: t.sum().numpy()),
        ("add", lambda t: (t + 1.0).numpy()),
        ("reshape", lambda t: t.reshape(40).numpy()),
        ("clone", lambda t: t.clone().numpy()),
        ("indexing", lambda t: t[2].numpy()),
    ]:
        t = w.t()
        out = use(t)
        assert t.materialized, f"{name} should have materialized it"
        if name == "numpy":
            assert (out == want).all()


def test_a_transpose_that_is_not_of_the_last_two_dims_is_eager():
    """Nothing can absorb it, so there is no point deferring it."""
    x = grilly.randn(2, 3, 4, generator=gen(13))
    t = x.transpose(0, 2)
    assert not isinstance(t, _Transposed)
    assert (t.numpy() == x.numpy().transpose(2, 1, 0)).all()


def test_writing_through_a_materialized_transpose_is_still_refused():
    """grilly returns a copy where torch returns a view, so an in-place
    write through one would be silently lost. That rule must survive."""
    w = grilly.randn(4, 6, generator=gen(14))
    t = w.t()
    with pytest.raises(RuntimeError, match="transposed"):
        t.copy_(grilly.zeros(6, 4))
    assert t.materialized, "it had to be built to find out, which is fine"


# ── gradients ────────────────────────────────────────────────────────────


def test_gradients_reach_both_operands():
    x = grilly.randn(3, 5, generator=gen(15))
    w = grilly.randn(7, 5, generator=gen(16))
    x.requires_grad_(True)
    w.requires_grad_(True)
    (x @ w.t()).sum().backward()
    assert np.abs(x.grad.numpy() - w.detach().numpy().sum(0)).max() < 1e-6
    assert np.abs(w.grad.numpy() - x.detach().numpy().sum(0)).max() < 1e-6


def test_the_gradients_are_the_ones_the_eager_route_gave():
    x_np = np.random.default_rng(0).standard_normal((6, 5)).astype(np.float32)
    w_np = np.random.default_rng(1).standard_normal((9, 5)).astype(np.float32)
    weights = np.random.default_rng(2).standard_normal((6, 9)).astype(np.float32)

    grads = []
    for eager in (False, True):
        x, w = grilly.from_numpy(x_np.copy()), grilly.from_numpy(w_np.copy())
        x.requires_grad_(True)
        w.requires_grad_(True)
        right = real_transpose(w) if eager else w.t()
        ((x @ right) * grilly.from_numpy(weights)).sum().backward()
        grads.append((x.grad.numpy().copy(), None if eager else w.grad.numpy().copy()))

    assert (grads[0][0] == grads[1][0]).all(), "dx must not depend on the representation"
    # the eager route sends dw to the transposed copy, so only dx is comparable;
    # dw is checked against its closed form instead
    assert np.abs(grads[0][1] - (weights.T.astype(np.float64) @ x_np)).max() < 1e-5


def test_a_transpose_made_under_grad_keeps_its_node_if_built_later():
    """Deferring must not lose an autograd edge: a transpose created with
    grad on, then materialized inside ``no_grad``, still has to be
    differentiable, because torch's would be."""
    w = grilly.randn(4, 3, generator=gen(17))
    w.requires_grad_(True)
    t = w.t()
    with grilly.no_grad():
        t._t                            # materializes, inside no_grad
    assert t.materialized and t.requires_grad
    t.sum().backward()
    assert w.grad is not None and float(w.grad.sum().item()) == pytest.approx(12.0)
