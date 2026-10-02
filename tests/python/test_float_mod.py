"""``%``, ``//`` and ``fmod`` on floats, and where the composition stops.

Reference: **torch** (case (b)). These are composed from divide-and-round
rather than given a kernel, so the comparison is against a different
implementation — but over integral float operands every step is exact, so
the bound asserted here is **0.0**, not an epsilon. That is the regime
that matters: a ring-buffer slot, ``pos % window``, kept in a float32
counter so a decode step can be captured.

The limit is real and is the composition's, not torch's: a true ``fmod``
is exact for every finite pair, this one forms ``floor(input/other) *
other`` and that product loses its low bits once it passes the mantissa.
The last two tests measure where that happens rather than assuming it,
because the first guess was wrong twice: the uncorrected formula already
disagreed at a dividend of 2**12, with a *negative* remainder, and the
boundary is set by the dividend and divisor together, not by the
quotient — a power-of-two divisor stays exact to 2**39.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import grilly

SIGNS = [(7.0, 4.0), (-7.0, 4.0), (7.0, -4.0), (-7.0, -4.0), (8.0, 4.0), (0.0, 4.0)]


@pytest.mark.parametrize("op", ["remainder", "fmod", "floor_divide"])
@pytest.mark.parametrize("a,b", SIGNS)
def test_the_signs_are_torchs(op, a, b):
    """The three differ only in how they round the quotient, and every
    difference between them shows up in the signs."""
    g = getattr(grilly, op)(grilly.tensor([a]), grilly.tensor([b]))
    t = getattr(torch, op)(torch.tensor([a]), torch.tensor([b]))
    assert g.numpy() == pytest.approx(t.numpy(), abs=0.0)


def test_the_operators_route_there_too():
    a, b = grilly.tensor([-7.0, 7.0]), grilly.tensor([4.0, -4.0])
    ta, tb = torch.tensor([-7.0, 7.0]), torch.tensor([4.0, -4.0])
    assert ((a % b).numpy() == (ta % tb).numpy()).all()
    assert ((a // b).numpy() == (ta // tb).numpy()).all()
    assert ((7.0 % b).numpy() == (7.0 % tb).numpy()).all()


def test_a_scalar_divisor_is_the_same_as_a_tensor_one():
    a = grilly.from_numpy(np.linspace(-20, 20, 41).astype(np.float32))
    assert ((a % 7.0).numpy() == (a % grilly.full((41,), 7.0)).numpy()).all()


def test_the_ring_slot_is_exact_over_the_whole_float32_integer_range():
    """What it is for. A position counter held in float32 is exact to
    2**24; the slot it picks has to be exact over all of that."""
    window = 512
    pos = np.array([0, 1, 511, 512, 513, 1 << 20, (1 << 24) - 1], dtype=np.float32)
    got = (grilly.from_numpy(pos) % float(window)).numpy()
    assert (got == pos % window).all()
    assert (got.astype(np.int64) == pos.astype(np.int64) % window).all()


def _sweep(b, exponent):
    rng = np.random.default_rng(0)
    a = ((rng.random(8192).astype(np.float32) * 2 - 1) * np.float32(2.0 ** exponent)).astype(np.float32)
    g = (grilly.from_numpy(a) % float(b)).numpy()
    t = (torch.from_numpy(a) % float(b)).numpy()
    return np.abs(g - t).max()


@pytest.mark.parametrize("b", [3.0, 512.0, -3.0, 7.0])
@pytest.mark.parametrize("exponent", [1, 7, 13, 19, 23])
def test_bit_exact_with_torch_while_the_dividend_fits_the_mantissa(b, exponent):
    """Bit-exact, not approximately: the repaired formula does the same
    arithmetic torch does whenever every step of it is representable."""
    assert _sweep(b, exponent) == 0.0


@pytest.mark.parametrize("exponent", [25, 31, 39])
def test_a_power_of_two_divisor_stays_exact_far_past_that(exponent):
    """Because then ``floor(q) * other`` is a shift, exact whatever the
    quotient. A ring window is usually a power of two, so this is the
    case the capture path actually runs."""
    assert _sweep(512.0, exponent) == 0.0


def test_and_where_it_stops(exponent=25):
    """The other side of the same fact, so nobody has to find it the hard
    way: past the mantissa the answer is wrong, not imprecise."""
    assert _sweep(3.0, exponent) == 2.0
    a = grilly.tensor([float(1 << 27)]) % 3.0
    assert float(a.numpy()[0]) != float(torch.tensor([float(1 << 27)]) % 3.0)


# ── the in-place arithmetic a device counter needs ───────────────────────
#
# Reference: torch (case (b)), but every form here is one kernel over the
# same inputs on both sides, so the bound asserted is 0.0.


@pytest.mark.parametrize("name", ["add_", "sub_", "mul_", "div_"])
@pytest.mark.parametrize("kind", ["scalar", "tensor", "broadcast"])
def test_the_in_place_forms_match_torch(name, kind):
    """``add_`` had a kernel for one shape of call and refused the rest —
    a scalar, a broadcast, an ``alpha``. A decode step advancing a
    position counter writes ``pos.add_(1.0)``, which was the refused one."""
    base = np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0
    g, t = grilly.from_numpy(base.copy()), torch.from_numpy(base.copy())
    arg = {"scalar": (2.5, 2.5),
           "tensor": (grilly.full((2, 3), 2.5), torch.full((2, 3), 2.5)),
           "broadcast": (grilly.full((3,), 2.5), torch.full((3,), 2.5))}[kind]
    getattr(g, name)(arg[0])
    getattr(t, name)(arg[1])
    assert (g.numpy() == t.numpy()).all()


def test_add_with_alpha_matches_torch():
    g, t = grilly.from_numpy(np.ones(4, dtype=np.float32)), torch.ones(4)
    g.add_(grilly.full((4,), 3.0), alpha=2.0)
    t.add_(torch.full((4,), 3.0), alpha=2.0)
    assert (g.numpy() == t.numpy()).all()


def test_an_in_place_write_of_the_wrong_shape_raises_like_torch():
    with pytest.raises(RuntimeError, match="doesn't match the broadcast shape"):
        grilly.ones((3,)).add_(grilly.ones((2, 3)))


def test_a_float32_counter_advances_exactly_over_a_long_decode():
    """2**24 tokens of ``pos.add_(1.0)``, spot-checked: the reason the
    counter is float32 and not int64 is that an int64 scalar has to be
    built on the host, which a capture refuses."""
    pos = grilly.zeros(())
    for _ in range(1000):
        pos.add_(1.0)
    assert float(pos.numpy()) == 1000.0
    far = grilly.full((), float((1 << 24) - 2))
    far.add_(1.0)
    assert float(far.numpy()) == float((1 << 24) - 1)
