"""``grilly.vsa`` — sparse block-code VSA (IBM NVSA), against v1 and numpy.

Block codes are ``(..., k, l)`` float32 with **one hot position per block**
(k = 80, l = 128 downstream). Feeding dense Gaussians to these operations
produces numbers that look like a semantics bug and are not (CLAUDE.md
§8.2), so the discrete tests build codes the way the algebra defines them
and the continuous tests say so in their names.

Tolerances, and what each one is for:

- **one-hot binding: bit-exact.** Each output sum has exactly one non-zero
  term, 1.0, so the result is an exact one-hot whatever order the zeros are
  added in. v1's shader got there by finding the hot positions and adding
  them, so the two agree bit for bit — ``==``, not ``allclose``.
- **against v1's ``BlockCodeOps``: one ulp of 1.0** (``float32`` eps,
  1.1920929e-07). v1's Python does the same convolution through an FFT
  pair, so where this port writes an exact 1.0 the FFT writes the nearest
  float to it. CLAUDE.md §7 records 1.19e-07 for v1's bridge against the
  same reference; that is this number to three figures, and the measured
  difference here is exactly it. Anything larger would be a real
  disagreement, so the bound is the ulp and not a round number near it.
- **continuous codes: 4 * l * eps.** l products summed in float32, against
  the same convolution in float64. The worst measured ratio is printed.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np
import pytest

import grilly
from grilly import vsa

K, L = 80, 128


def v1_block_ops():
    """v1's ``BlockCodeOps``, loaded from a v1 checkout if there is one.

    Pure numpy, so it needs no v1 build — but v1 is not a dependency, and a
    machine without it still runs every other test here.
    """
    env = os.environ.get("GRILLY_V1_DIR")
    root = pathlib.Path(env) if env else pathlib.Path(__file__).resolve().parents[2].parent / "grilly"
    path = root / "experimental" / "vsa" / "block_ops.py"
    if not path.is_file():
        pytest.skip(f"no v1 checkout with {path.name} (set GRILLY_V1_DIR)")
    import importlib.util

    spec = importlib.util.spec_from_file_location("_v1_block_ops", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_v1_block_ops"] = module
    spec.loader.exec_module(module)
    return module.BlockCodeOps


def one_hot(n, seed):
    """``(n, K, L)`` discrete codes, and the hot position of every block."""
    rng = np.random.default_rng(seed)
    hot = rng.integers(0, L, size=(n, K))
    a = np.zeros((n, K, L), dtype=np.float32)
    np.put_along_axis(a, hot[:, :, None], 1.0, axis=2)
    return a, hot


def reference(a, b, unbind=False):
    """The same convolution in float64, from the definition."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    out = np.zeros(a.shape)
    for i in range(a.shape[-1]):
        for j in range(a.shape[-1]):
            if unbind:
                out[..., i] += a[..., (i + j) % L] * b[..., j]
            else:
                out[..., i] += a[..., j] * b[..., (i + L - j) % L]
    return out


def test_binding_one_hot_codes_is_exact_and_lands_where_v1_put_it():
    a, ha = one_hot(8, 1)
    b, hb = one_hot(8, 2)
    got = vsa.bind(grilly.from_numpy(a), grilly.from_numpy(b)).numpy()

    want = np.zeros_like(a)                      # v1's rule: hot at (i + j) % l
    np.put_along_axis(want, ((ha + hb) % L)[:, :, None], 1.0, axis=2)
    assert (got == want).all(), "one-hot binding is not exact"


def test_unbinding_recovers_either_factor_exactly():
    x, _ = one_hot(16, 3)
    key, _ = one_hot(16, 4)
    tx, tk = grilly.from_numpy(x), grilly.from_numpy(key)
    composite = vsa.bind(tx, tk)
    assert (vsa.unbind(composite, tk).numpy() == x).all()
    assert (vsa.unbind(composite, tx).numpy() == key).all()


def test_binding_agrees_with_v1s_block_code_ops():
    """The agreement CLAUDE.md §7 records for v1's bridge, 1.19e-07: one
    ulp of the 1.0 this port writes exactly, which is what an FFT round
    trip costs. Not a semantic difference, and not a free tolerance —
    two ulp would mean the convolution disagreed somewhere."""
    ops = v1_block_ops()
    ulp = float(np.finfo(np.float32).eps)
    a, _ = one_hot(32, 5)
    b, _ = one_hot(32, 6)
    for name, ours, theirs in [
        ("bind", vsa.bind(grilly.from_numpy(a), grilly.from_numpy(b)).numpy(), ops.bind(a, b)),
        ("unbind", vsa.unbind(grilly.from_numpy(a), grilly.from_numpy(b)).numpy(), ops.unbind(a, b)),
    ]:
        worst = float(np.abs(ours - theirs).max())
        print(f"{name} vs v1 BlockCodeOps: max abs diff {worst:.7e} ({worst / ulp:.2f} ulp)")
        assert worst <= ulp, f"{name}: {worst:.7e} is {worst / ulp:.2f} ulp"


def test_similarity_and_bundling_agree_with_v1():
    ops = v1_block_ops()
    a, _ = one_hot(1, 7)
    b, _ = one_hot(1, 8)
    cb, _ = one_hot(64, 9)
    ta, tb, tcb = (grilly.from_numpy(x) for x in (a[0], b[0], cb))

    assert vsa.similarity(ta, ta) == pytest.approx(ops.similarity(a[0], a[0]), abs=1e-7)
    assert vsa.similarity(ta, tb) == pytest.approx(ops.similarity(a[0], b[0]), abs=1e-7)
    assert np.abs(vsa.similarity_batch(ta, tcb).numpy() - ops.similarity_batch(a[0], cb)).max() <= 1e-7
    assert np.abs(vsa.bundle([ta, tb]).numpy() - ops.bundle([a[0], b[0]], normalize=True)).max() <= 1e-7


def test_continuous_codes_match_a_float64_convolution():
    """Continuous codes are distributions per block, not one-hot: the
    general case the FFT path also covers."""
    rng = np.random.default_rng(11)
    a = rng.random((2, K, L)).astype(np.float32)
    b = rng.random((2, K, L)).astype(np.float32)
    a, b = a / a.sum(-1, keepdims=True), b / b.sum(-1, keepdims=True)
    got = vsa.bind(grilly.from_numpy(a), grilly.from_numpy(b)).numpy()
    bound = 4 * L * np.finfo(np.float32).eps * np.abs(a).max() * np.abs(b).max()
    worst = np.abs(got - reference(a, b)).max() / bound
    print(f"continuous bind: worst error / bound = {worst:.3f}")
    assert worst <= 1.0


def test_one_key_binds_against_a_whole_codebook():
    cb, hc = one_hot(32, 12)
    key, hk = one_hot(1, 13)
    got = vsa.bind(grilly.from_numpy(cb), grilly.from_numpy(key[0])).numpy()
    want = np.zeros_like(cb)
    np.put_along_axis(want, ((hc + hk) % L)[:, :, None], 1.0, axis=2)
    assert (got == want).all()


def test_a_bundle_of_codes_is_nearest_to_its_members():
    """What bundling is for: the superposition stays similar to everything
    in it and not to anything else."""
    cb = vsa.codebook(16, K, L, generator=grilly.Generator().manual_seed(0))
    bundled = vsa.bundle([cb[i] for i in range(4)])
    sims = vsa.similarity_batch(bundled, cb).numpy()
    assert sims[:4].min() > sims[4:].max(), f"members {sims[:4]} vs others {sims[4:].max()}"


def test_zero_is_the_identity_and_cyclic_shift_is_v1s():
    ops = v1_block_ops()
    a, _ = one_hot(1, 14)
    ta = grilly.from_numpy(a[0])
    assert (vsa.bind(ta, vsa.zero(K, L)).numpy() == a[0]).all()
    for n in (0, 1, 7, K, K + 3):
        assert (vsa.cyclic_shift(ta, n).numpy() == ops.cyclic_shift(a[0], n)).all(), f"shift {n}"


def test_discretize_and_flat_round_trip():
    ops = v1_block_ops()
    rng = np.random.default_rng(15)
    a = rng.random((3, K, L)).astype(np.float32)
    ta = grilly.from_numpy(a)
    assert (vsa.discretize(ta).numpy() == ops.discretize(a)).all()
    assert tuple(vsa.to_flat(ta).shape) == (3, K * L)
    assert (vsa.from_flat(vsa.to_flat(ta), K).numpy() == a).all()


@pytest.mark.parametrize("batch, required", [(64, 1.74), (512, 1.83)])
def test_batched_binding_beats_v1s_python_by_at_least_v1s_bridge_did(batch, required):
    """CLAUDE.md §7's measured speedups for v1's bridge over v1's Python, at
    the batch sizes they were quoted at. The GPU win is batch-only, so the
    batch size belongs beside every number here (§8.5)."""
    ops = v1_block_ops()
    a, _ = one_hot(batch, 16)
    b, _ = one_hot(batch, 17)
    ta, tb = grilly.from_numpy(a), grilly.from_numpy(b)

    def timed(fn, n):
        fn()
        start = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - start) / n

    theirs = timed(lambda: ops.bind(a, b), 5)
    ours = timed(lambda: vsa.bind(ta, tb).numpy(), 20)
    print(f"batch {batch}: v1 python {theirs * 1e3:.2f} ms, grilly2 {ours * 1e3:.2f} ms, "
          f"{theirs / ours:.2f}x")
    assert theirs / ours >= required, f"batch {batch}: {theirs / ours:.2f}x, needed {required}x"
