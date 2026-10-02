"""``g2-gemv`` — `linear` for one or two rows, which is decode's shape.

Reference: **float64**, under a bound derived from the op, with the worst
ratio printed (CLAUDE.md §9's preferred form). A length-K dot product in
float32 has error at most ``K * eps * (|x| @ |W|)``, so that is the bound;
a ratio to it is what says whether the kernel regressed, where a
hand-picked epsilon would not.

Why it exists: the tiled kernel does 16x16 tiles, so at one row fifteen of
every sixteen tile rows do nothing. Measured against the card's 432 GB/s at
M=1 before this kernel: (512,512) 62 GB/s, (127996,512) 193. CubbyLLM's
decode step spent 2.3 ms of its 5.5 ms token in `linear`.

The switch is by row count, and **two is measured, not assumed** — see
`GEMV_MAX_ROWS` in `crates/grilly-torch/src/kernels.rs` for the table. The
first guess of eight regressed M=4 at small `out_features`.
"""

from __future__ import annotations

import numpy as np
import pytest

import grilly
import grilly.nn.functional as F

# the boundary the kernel switches at; see kernels.rs::GEMV_MAX_ROWS
MAX_ROWS = 2


def bound_ratio(got, x, w, bias=None):
    """``|got - float64| / (K * eps * (|x| @ |W|))``, the worst over all
    outputs. At most 1.0 means the kernel is within a dot product's error."""
    xn, wn = np.asarray(x, np.float64), np.asarray(w, np.float64)
    reference = xn @ wn.T
    bound = xn.shape[-1] * 2.0 ** -24 * (np.abs(xn) @ np.abs(wn).T)
    if bias is not None:
        reference = reference + np.asarray(bias, np.float64)
        bound = bound + np.abs(np.asarray(bias, np.float64)) * 2.0 ** -24
    return float((np.abs(np.asarray(got, np.float64) - reference)
                  / np.maximum(bound, 1e-300)).max())


def rand(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("k,n", [(1, 1), (4, 3), (63, 65), (64, 64), (512, 512), (512, 4096)])
def test_it_is_within_a_dot_products_error(rows, k, n):
    """Including the sizes that exercise the strided loop's tail: K one
    below, at, and one above the 64-thread workgroup."""
    x, w = rand((rows, k), k), rand((n, k), n + 1)
    got = F.linear(grilly.from_numpy(x.copy()), grilly.from_numpy(w.copy())).numpy()
    assert bound_ratio(got, x, w) <= 1.0


@pytest.mark.parametrize("rows", [1, 2])
def test_the_bias_is_added(rows):
    x, w, b = rand((rows, 7), 1), rand((5, 7), 2), rand(5, 3)
    got = F.linear(grilly.from_numpy(x.copy()), grilly.from_numpy(w.copy()),
                   grilly.from_numpy(b.copy())).numpy()
    assert bound_ratio(got, x, w, b) <= 1.0


def test_a_vocabulary_sized_output_needs_two_workgroup_dimensions():
    """127996 workgroups is past the 65535 a single dimension allows, so
    the launch is 2-D and the shader recombines the id. A kernel that got
    that wrong would write only the first 65535 columns."""
    x, w = rand((1, 512), 9), rand((127996, 512), 10)
    got = F.linear(grilly.from_numpy(x.copy()), grilly.from_numpy(w.copy())).numpy()
    assert got.shape == (1, 127996)
    assert np.isfinite(got).all()
    # the columns past the one-dimensional limit are real, not zeros
    assert np.abs(got[0, 65535:]).mean() > 0.1 * np.abs(got[0, :65535]).mean()
    assert bound_ratio(got, x, w) <= 1.0


def test_the_switch_does_not_change_the_answer_beyond_the_bound():
    """Both sides of the row threshold have to be correct matmuls. They are
    *not* bit-identical to each other — different kernels sum in different
    orders — so what is asserted is that each is within the bound, and the
    ratios are printed so a regression on either side shows."""
    k, n = 512, 256
    w = rand((n, k), 11)
    ratios = {}
    for rows in (MAX_ROWS, MAX_ROWS + 1):          # gemv, then the tiled kernel
        x = rand((rows, k), 12)
        got = F.linear(grilly.from_numpy(x.copy()), grilly.from_numpy(w.copy())).numpy()
        ratios[f"rows={rows}"] = bound_ratio(got, x, w)
    print("  worst |err| / (K*eps*|x|@|W|): "
          + ", ".join(f"{k_}: {v:.3f}" for k_, v in ratios.items()))
    for name, ratio in ratios.items():
        assert ratio <= 1.0, f"{name} is {ratio:.3f} of its error bound"


def test_the_same_input_gives_the_same_bits_every_time():
    """Deterministic: each thread sums its strided slice in order and the
    workgroup combines in a fixed tree, so there is no run-to-run
    variation to hide a race. Case (a), bit-exact."""
    x = grilly.from_numpy(rand((1, 512), 13))
    w = grilly.from_numpy(rand((777, 512), 14))
    first = F.linear(x, w).numpy().copy()
    for _ in range(5):
        assert (F.linear(x, w).numpy() == first).all()


def test_it_matches_torch_within_the_same_bound():
    """The cross-implementation read (case (b)). torch sums differently
    again, so the comparison is through the shared float64 reference
    rather than against torch's bits."""
    torch = pytest.importorskip("torch")
    x, w = rand((1, 512), 15), rand((1024, 512), 16)
    mine = F.linear(grilly.from_numpy(x.copy()), grilly.from_numpy(w.copy())).numpy()
    theirs = torch.nn.functional.linear(torch.from_numpy(x.copy()),
                                        torch.from_numpy(w.copy())).numpy()
    assert bound_ratio(mine, x, w) <= 1.0
    assert bound_ratio(theirs, x, w) <= 1.0


def test_a_captured_step_can_use_it():
    """Decode is the reason it exists, and decode is recorded."""
    from grilly import _C

    x = grilly.from_numpy(rand((1, 64), 17))
    w = grilly.from_numpy(rand((128, 64), 18))
    with grilly.no_grad():
        want = F.linear(x, w).numpy().copy()
        _C.begin_capture()
        out = F.linear(x, w)
        graph = _C.end_capture()
        graph.replay()
    assert (out.numpy() == want).all()
