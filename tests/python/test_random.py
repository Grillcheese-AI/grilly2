"""Seeded random numbers: deterministic within grilly, not torch's values.

Where torch's behaviour is checkable without its values (seed range,
generator isolation, return types, distributions), it is checked here.
"""

import numpy as np
import pytest
import torch

import grilly


def test_manual_seed_replays():
    grilly.manual_seed(0)
    a = grilly.randn(3, 4).numpy()
    b = grilly.rand(5).numpy()
    grilly.manual_seed(0)
    np.testing.assert_array_equal(grilly.randn(3, 4).numpy(), a)
    np.testing.assert_array_equal(grilly.rand(5).numpy(), b)


def test_stream_is_pinned():
    """The stream is part of the contract: changing it breaks users'
    reproducibility, so it must fail here first. Values are Philox4x32-10 +
    Box-Muller as specified in grilly-core's rng module."""
    z = grilly.randn(6, generator=grilly.Generator().manual_seed(0)).numpy()
    u = grilly.rand(4, generator=grilly.Generator().manual_seed(0)).numpy()
    # Block 0 of seed 0 is Random123's known answer 6627e8d5 e169c58d bc57ac4c 9b00dbd8.
    words = [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    np.testing.assert_array_equal(u, np.array([(w >> 8) * 2.0**-24 for w in words], np.float32))
    expect = []
    for a, b in [(words[0], words[1]), (words[2], words[3])]:
        r = np.sqrt(-2.0 * np.log(((a >> 8) + 1) * 2.0**-24))
        t = 2 * np.pi * ((b >> 8) * 2.0**-24)
        expect += [r * np.cos(t), r * np.sin(t)]
    np.testing.assert_array_equal(z[:4], np.array(expect, np.float32))


def test_generators_are_independent_of_the_global_stream():
    g = grilly.Generator().manual_seed(123)
    a = grilly.randn(8, generator=g).numpy()
    grilly.manual_seed(5)
    grilly.randn(100)  # consuming the default stream must not matter
    b = grilly.randn(8, generator=grilly.Generator().manual_seed(123)).numpy()
    np.testing.assert_array_equal(a, b)


def test_consecutive_draws_differ():
    grilly.manual_seed(0)
    assert not np.array_equal(grilly.randn(16).numpy(), grilly.randn(16).numpy())


def test_torch_shaped_api():
    g = grilly.Generator(device="cpu")
    assert g.manual_seed(0xC0DEB00C) is g            # returns self, like torch
    assert g.initial_seed() == 0xC0DEB00C
    assert isinstance(grilly.manual_seed(1), grilly.Generator)
    assert grilly.initial_seed() == 1
    for f in (grilly.randn, grilly.rand):
        assert f(2, 3).shape == f((2, 3)).shape == torch.randn(2, 3).shape
        assert f(()).shape == ()
    x = grilly.zeros(4, 5)
    assert grilly.randn_like(x).shape == grilly.rand_like(x).shape == (4, 5)


def test_default_initial_seed_matches_torch():
    assert grilly.Generator().initial_seed() == torch.Generator().initial_seed()


@pytest.mark.parametrize("seed", [0, 1, -1, -(2**63), 2**64 - 1])
def test_seed_range_matches_torch(seed):
    assert grilly.Generator().manual_seed(seed).initial_seed() == \
        torch.Generator().manual_seed(seed).initial_seed()


@pytest.mark.parametrize("seed", [2**64, -(2**63) - 1])
def test_seed_overflow_raises_like_torch(seed):
    with pytest.raises(Exception) as want:
        torch.Generator().manual_seed(seed)
    with pytest.raises(want.type, match=str(want.value)):
        grilly.Generator().manual_seed(seed)


def test_seed_is_nondeterministic_and_replayable():
    s = grilly.seed()
    assert grilly.initial_seed() == s
    a = grilly.randn(4).numpy()
    grilly.manual_seed(s)
    np.testing.assert_array_equal(grilly.randn(4).numpy(), a)


def test_distributions():
    grilly.manual_seed(0)
    z = grilly.randn(1 << 20).numpy().astype(np.float64)
    u = grilly.rand(1 << 20).numpy()
    assert abs(z.mean()) < 5e-3 and abs(z.var() - 1) < 7e-3
    assert u.min() >= 0 and u.max() < 1 and abs(u.mean() - 0.5) < 2e-3


@pytest.mark.parametrize("call", [
    lambda: grilly.randint(0, 2, (3,)),
    lambda: grilly.Generator(device="cuda"),
    lambda: grilly.Generator().get_state(),
    lambda: grilly.randn(3, out=grilly.zeros(3)),
])
def test_unimplemented_raises(call):
    with pytest.raises(NotImplementedError):
        call()
