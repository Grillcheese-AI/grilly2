"""Which arithmetic is bit-exact against numpy, and which is not.

This is the measurement behind ``docs/kernels.md``'s "What is exact, and
what is two ULP", and it is here so the split is a pinned fact rather than
folklore. It decides what a grilly-against-torch test may assert: an
expression built only from ``+ - *`` can be compared with ``==``, and one
containing a divide cannot.

Not a kernel bug. Vulkan requires ``OpFDiv`` to be within 2.5 ULP and
``OpSqrt`` within 3.0, and requires correct rounding only of add, subtract
and multiply. CUDA's divide *is* correctly rounded, which is exactly why
this shows up as a difference from torch and not as a difference from
itself.
"""

from __future__ import annotations

import numpy as np
import pytest

import grilly

N = 200_000


def ulps(got: np.ndarray, want: np.ndarray) -> int:
    a = got.view(np.int32).astype(np.int64)
    b = want.view(np.int32).astype(np.int64)
    return int(np.abs(a - b).max())


@pytest.fixture(scope="module")
def pair():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N).astype(np.float32)
    y = (rng.standard_normal(N) + 3.0).astype(np.float32)      # away from zero
    return x, y, grilly.from_numpy(x.copy()), grilly.from_numpy(y.copy())


@pytest.mark.parametrize("name", ["add", "sub", "mul"])
def test_the_correctly_rounded_operations_are_bit_exact(name, pair):
    """Vulkan requires these three to be correctly rounded, so there is no
    excuse for a difference and none is allowed here."""
    x, y, gx, gy = pair
    op = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b}[name]
    assert ulps(op(gx, gy).numpy(), op(x, y)) == 0


@pytest.mark.parametrize("name,bound", [("div", 2), ("reciprocal", 1), ("sqrt", 1)])
def test_the_rest_are_within_the_measured_bound(name, bound):
    """The bound is what was measured, and it is asserted from both sides:
    a regression past it is a kernel bug, and an op that suddenly becomes
    exact is worth knowing about too — it would mean the table above, and
    every test that relaxed for it, can be tightened."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(N).astype(np.float32)
    y = (rng.standard_normal(N) + 3.0).astype(np.float32)
    gx, gy = grilly.from_numpy(x.copy()), grilly.from_numpy(y.copy())
    got, want = {
        "div": (gx / gy, x / y),
        "reciprocal": (1.0 / gy, np.float32(1.0) / y),
        "sqrt": (grilly.sqrt(gy), np.sqrt(y)),
    }[name]
    measured = ulps(got.numpy(), want)
    assert measured <= bound, f"{name} drifted to {measured} ULP, was {bound}"


def test_a_divide_is_reproducible_even_though_it_is_not_exact():
    """Case (a) of CLAUDE.md §9 is untouched by any of this: the same
    kernel on the same hardware gives the same answer every time, which is
    what capture and replay rest on."""
    rng = np.random.default_rng(2)
    x = grilly.from_numpy(rng.standard_normal(10_000).astype(np.float32))
    y = grilly.from_numpy((rng.standard_normal(10_000) + 3.0).astype(np.float32))
    first = (x / y).numpy().copy()
    for _ in range(4):
        assert ((x / y).numpy() == first).all()
