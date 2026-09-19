"""``grilly.compile`` (docs/fusion.md, "F2.2"): lazy capture of float32
elementwise chains inside a compiled callable, materialized into one fused
kernel (crates/grilly-core/src/fusion.rs) on first non-elementwise use.

Chains are checked bitwise against the same expression run eagerly on
grilly (grilly's fusion is bit-for-bit against its own unfused kernels,
crates/grilly-torch/tests/fusion.rs) and against torch within rtol 1e-6,
atol 1e-6, where grilly's own op is torch-accurate to that tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as TF

import grilly
import grilly.nn as gnn
import grilly.nn.functional as GF


def rand(*shape, seed=0, positive=False):
    a = np.random.default_rng(seed).uniform(-1, 1, shape).astype(np.float32)
    return np.abs(a) + 0.1 if positive else a


def both(a, requires_grad=False):
    return grilly.tensor(a, requires_grad=requires_grad), torch.tensor(a, requires_grad=requires_grad)


def close(g, t, rtol=1e-6, atol=1e-6):
    t = t.detach().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    g = g.detach().numpy() if isinstance(g, grilly.Tensor) else np.asarray(g)
    assert g.shape == t.shape, (g.shape, t.shape)
    np.testing.assert_allclose(g, t, rtol=rtol, atol=atol)


def bitwise(a, b):
    """grilly's fusion determinism guarantee: a fused graph returns what the
    same ops return one kernel at a time, bit for bit (NaN excepted)."""
    np.testing.assert_array_equal(a.detach().numpy(), b.detach().numpy())


@pytest.fixture(autouse=True)
def _reset_compiler_stats():
    grilly.compiler.reset()
    yield


# ── chains: compiled matches eager grilly bit for bit, and torch within tolerance ──

def _tanh_gelu(x):
    return 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x ** 3)).tanh())


def _silu(x):
    return x * x.sigmoid()


def _laplace_ish(x):
    return (-x ** 2).exp() / (1.0 + x.abs())


def _broadcast_chain(a, b, c):
    return a.tanh() * b.sigmoid() + c.exp()


CHAINS = [
    ("tanh_gelu_written_out", _tanh_gelu, lambda x: TF.gelu(x, approximate="tanh"), [(3, 5)]),
    ("F_gelu_tanh", lambda x: GF.gelu(x, approximate="tanh"), lambda x: TF.gelu(x, approximate="tanh"), [(3, 5)]),
    ("silu", _silu, TF.silu, [(4, 4)]),
    ("laplace_ish", _laplace_ish, lambda x: torch.exp(-x ** 2) / (1 + x.abs()), [(6,)]),
]


@pytest.mark.parametrize("name,fn,torch_fn,shapes", CHAINS, ids=[c[0] for c in CHAINS])
def test_chain_matches_eager_and_torch(name, fn, torch_fn, shapes):
    (shape,) = shapes
    a = rand(*shape, seed=1)
    gx, tx = both(a)

    compiled = grilly.compile(fn)
    out = compiled(gx)
    eager = fn(gx)
    bitwise(out, eager)
    close(out, torch_fn(tx))


def test_broadcast_chain_matches_eager_and_torch():
    a = rand(4, 1, 5, seed=1)
    b = rand(3, 1, seed=2)
    c = rand(5, seed=3)
    ga, gb, gc = grilly.tensor(a), grilly.tensor(b), grilly.tensor(c)
    ta, tb, tc = torch.tensor(a), torch.tensor(b), torch.tensor(c)

    compiled = grilly.compile(_broadcast_chain)
    out = compiled(ga, gb, gc)
    assert tuple(out.shape) == (4, 3, 5)
    bitwise(out, _broadcast_chain(ga, gb, gc))
    close(out, _broadcast_chain(ta, tb, tc))


# ── one materialization for a chain that is never forced early ─────────────

def _chain8(x):
    y = x + 1.0
    y = y * 2.0
    y = y.tanh()
    y = y.sigmoid()
    y = y.exp()
    y = y.abs()
    y = -y
    return y * 0.5


def test_eight_op_chain_materializes_once():
    x = grilly.tensor(rand(5, 5, seed=7))
    compiled = grilly.compile(_chain8)
    out = compiled(x)
    bitwise(out, _chain8(x))
    stats = grilly.compiler._stats()
    assert stats["materializations"] == 1
    assert len(stats["node_counts"]) == 1
    assert stats["node_counts"][0] > 8  # each op also carries its scalar params as nodes


# ── an intermediate forced early (matmul / sum) is not recomputed ──────────

def test_intermediate_forced_by_matmul_then_reused():
    x = grilly.tensor(rand(4, 4, seed=1))
    w = grilly.tensor(rand(4, 4, seed=2))

    def f(x, w):
        y = x.tanh()          # lazy
        z = grilly.matmul(y, w)  # forces y to materialize
        return y.sigmoid() + z.sum()  # reuses the materialized y

    compiled = grilly.compile(f)
    out = compiled(x, w)
    expected_y = x.tanh()
    expected = expected_y.sigmoid() + grilly.matmul(expected_y, w).sum()
    bitwise(out, expected)
    stats = grilly.compiler._stats()
    assert stats["materializations"] == 2  # y (forced by matmul), then the final sum


def test_intermediate_forced_by_sum_then_reused():
    x = grilly.tensor(rand(3, 4, seed=3))

    def f(x):
        y = x.exp()
        total = y.sum()  # forces y
        return y.tanh() + total

    compiled = grilly.compile(f)
    out = compiled(x)
    expected_y = x.exp()
    expected = expected_y.tanh() + expected_y.sum()
    bitwise(out, expected)


# ── mixed float/int code stays correct ──────────────────────────────────────

def test_mixed_float_and_int_ops():
    x = grilly.tensor(rand(5, seed=2))
    idx = grilly.tensor(np.array([0, 2, 4], dtype=np.int64))

    def f(x, idx):
        y = x.tanh()  # float, capturable
        picked = grilly.index_select(y, 0, idx)  # not elementwise: forces y, int index
        counts = (idx * 2 + 1)  # pure int64 arithmetic, unrelated to capture
        return picked + 1.0, counts

    compiled = grilly.compile(f)
    out, counts = compiled(x, idx)
    expected_y = x.tanh()
    bitwise(out, grilly.index_select(expected_y, 0, idx) + 1.0)
    np.testing.assert_array_equal(counts.numpy(), (idx * 2 + 1).numpy())
    assert counts.dtype is grilly.int64


# ── requires_grad falls back to eager; gradients still match ───────────────

def test_requires_grad_falls_back_to_eager_and_gradients_match():
    a = rand(4, seed=5)
    gx, tx = both(a, requires_grad=True)

    def f(x):
        return (x.tanh() * x.sigmoid() + 1.0).sum()

    compiled = grilly.compile(f)
    g_out = compiled(gx)
    t_out = f(tx)
    close(g_out, t_out)

    g_out.backward()
    t_out.backward()
    close(gx.grad, tx.grad)

    # nothing was captured: no LazyTensor was ever built
    assert grilly.compiler._stats()["materializations"] == 0


def test_no_grad_around_parameters_does_capture():
    """A Linear's output requires grad by default (its weight does), so
    torch.compile's fusion correctly declines under grad mode; under
    no_grad() (as an optimizer step runs), the same forward is fused."""
    lin = gnn.Linear(3, 3)
    x = grilly.tensor(rand(2, 3, seed=9))

    def f(x):
        return lin(x).tanh()

    compiled = grilly.compile(f)

    with grilly.no_grad():
        out = compiled(x)
    with grilly.no_grad():
        expected = f(x)
    bitwise(out, expected)
    assert grilly.compiler._stats()["materializations"] >= 1


# ── the API forms ───────────────────────────────────────────────────────────

def test_compile_as_plain_function():
    x = grilly.tensor(rand(4, seed=1))
    compiled = grilly.compile(lambda t: t.tanh() * 2.0)
    bitwise(compiled(x), x.tanh() * 2.0)


def test_compile_as_bare_decorator():
    @grilly.compile
    def f(x):
        return x.sigmoid() + 1.0

    x = grilly.tensor(rand(4, seed=2))
    bitwise(f(x), grilly.tensor(rand(4, seed=2)).sigmoid() + 1.0)


def test_compile_as_decorator_with_arguments():
    @grilly.compile(fullgraph=True, dynamic=False, mode="max-autotune", options={"foo": 1})
    def f(x):
        return x.tanh()

    x = grilly.tensor(rand(4, seed=3))
    bitwise(f(x), x.tanh())


def test_compile_on_module_returns_optimized_module():
    class M(gnn.Module):
        def __init__(self):
            super().__init__()
            self.lin = gnn.Linear(3, 3)

        def forward(self, x):
            return self.lin(x).tanh()

    m = M()
    om = grilly.compile(m)
    assert isinstance(om, grilly.OptimizedModule)

    # parameters shared, not copied
    m_params = list(m.parameters())
    om_params = list(om.parameters())
    assert len(m_params) == len(om_params) == 2
    assert all(a is b for a, b in zip(m_params, om_params))

    # attribute forwarding
    assert om.lin is m.lin
    m.extra_attr = 42
    assert om.extra_attr == 42

    # state_dict keys are the wrapped module's own (no "_orig_mod." prefix)
    assert list(om.state_dict().keys()) == list(m.state_dict().keys()) == ["lin.weight", "lin.bias"]

    x = grilly.tensor(rand(2, 3, seed=4))
    with grilly.no_grad():
        out = om(x)
        expected = m(x)
    bitwise(out, expected)


def test_module_compile_in_place():
    class M(gnn.Module):
        def __init__(self):
            super().__init__()
            self.lin = gnn.Linear(3, 3)

        def forward(self, x):
            return self.lin(x).tanh()

    m = M()
    m.compile()
    assert type(m) is M  # in place: no wrapper object, unlike grilly.compile(m)

    x = grilly.tensor(rand(2, 3, seed=5))
    with grilly.no_grad():
        out = m(x)
        expected_lin = m.lin(x)
        expected = expected_lin.tanh()
    bitwise(out, expected)


def test_disable_true_returns_input_unchanged():
    def f(x):
        return x + 1

    assert grilly.compile(f, disable=True) is f

    identity = grilly.compile(disable=True)
    assert identity(f) is f


def test_backend_eager_runs_without_fusion():
    x = grilly.tensor(rand(4, seed=6))

    @grilly.compile(backend="eager")
    def f(x):
        assert grilly.compiler.is_compiling()
        return x.tanh()

    before = grilly.compiler._stats()["materializations"]
    out = f(x)
    assert grilly.compiler._stats()["materializations"] == before  # never captured, nothing to materialize
    bitwise(out, x.tanh())


def test_backend_aot_eager_runs_without_fusion():
    x = grilly.tensor(rand(4, seed=6))
    f = grilly.compile(lambda t: t.tanh(), backend="aot_eager")
    before = grilly.compiler._stats()["materializations"]
    out = f(x)
    assert grilly.compiler._stats()["materializations"] == before
    bitwise(out, x.tanh())


def test_unknown_backend_raises():
    with pytest.raises(RuntimeError):
        grilly.compile(lambda x: x, backend="not-a-real-backend")


def test_is_compiling_true_only_inside():
    assert not grilly.compiler.is_compiling()

    seen = {}

    @grilly.compile
    def f(x):
        seen["inside"] = grilly.compiler.is_compiling()
        return x.tanh()

    f(grilly.tensor(rand(3, seed=1)))
    assert seen["inside"] is True
    assert not grilly.compiler.is_compiling()


def test_compiler_disable_suspends_capture():
    from grilly import _lazy

    calls = {}

    @grilly.compiler.disable
    def inner(x):
        calls["capturing"] = _lazy.is_capturing()
        return x.sigmoid()

    @grilly.compile
    def outer(x):
        return inner(x) + 1.0

    x = grilly.tensor(rand(3, seed=2))
    outer(x)
    assert calls["capturing"] is False


# ── scalar hyperparameters that change between calls must not recompile ────

def _adam_step(p, m, v, g, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    return p - lr * m / (v.sqrt() + eps), m, v


def test_adam_style_update_does_not_recompile_for_new_scalars():
    rng = np.random.default_rng(11)
    p = grilly.tensor(rng.uniform(-1, 1, 8).astype(np.float32))
    m0 = grilly.tensor(np.zeros(8, dtype=np.float32))
    v0 = grilly.tensor(np.zeros(8, dtype=np.float32))
    g = grilly.tensor(rng.uniform(-1, 1, 8).astype(np.float32))

    compiled = grilly.compile(_adam_step)
    k0 = grilly._C.generated_kernel_count()
    p1, m1, v1 = compiled(p, m0, v0, g, 0.1)
    k1 = grilly._C.generated_kernel_count()
    assert k1 > k0  # the first call compiles at least one kernel

    p2, m2, v2 = compiled(p, m0, v0, g, 0.01)  # only lr changed
    k2 = grilly._C.generated_kernel_count()
    assert k2 == k1  # same graph structure: no new kernel

    # and the values are still correct for the new lr
    ep2, em2, ev2 = _adam_step(p, m0, v0, g, 0.01)
    bitwise(p2, ep2)
    bitwise(m2, em2)
    bitwise(v2, ev2)


# ── returned values are always real tensors ─────────────────────────────────

def test_returned_values_are_real_tensors_in_tuples_and_dicts():
    from grilly._lazy import LazyTensor

    @grilly.compile
    def f(x):
        return {
            "a": x.tanh(),
            "b": (x.sigmoid(), [x.exp(), x.abs()]),
        }

    x = grilly.tensor(rand(4, seed=1))
    out = f(x)
    assert type(out["a"]) is grilly.Tensor
    assert not isinstance(out["a"], LazyTensor)
    b0, (b1, b2) = out["b"][0], out["b"][1]
    for t in (b0, b1, b2):
        assert type(t) is grilly.Tensor
        assert not isinstance(t, LazyTensor)
