"""``grilly.vsa.holographic`` and ``grilly.vsa.resonator``.

HRR is dense real vectors under circular convolution, and unlike the other
algebras here its **unbinding is approximate**: a recovered factor is the
original plus crosstalk from everything else the composite holds. So the
tests assert what HRR actually promises — that the right codebook entry
ranks first — rather than an exact recovery it was never going to give.

Tolerances:

- **against a float64 FFT: ``2 * d * eps * rms(a) * rms(b)``.** Each output
  is a float32 sum of d products, each of magnitude about
  ``rms(a) * rms(b)``, so that is where the rounding can reach. Derived, not
  picked: the measured error sits at 43% of it, and the factor of 2 is the
  only slack. The worst ratio is printed, which is what says an op
  regressed (CLAUDE.md §9(b)).
- **recovery: ranked first, by a margin.** Not a number on the similarity,
  which depends on d and on how much is bundled in.
"""

from __future__ import annotations

import numpy as np
import pytest

import grilly
from grilly.vsa import bipolar, holographic as hrr, resonator

D = 1024


def gen(seed=0):
    return grilly.Generator().manual_seed(seed)


def test_binding_matches_a_float64_fft():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, D)).astype(np.float32)
    b = rng.standard_normal((4, D)).astype(np.float32)
    ta, tb = grilly.from_numpy(a), grilly.from_numpy(b)
    fa, fb = np.fft.fft(a.astype(np.float64)), np.fft.fft(b.astype(np.float64))
    rms = lambda x: float(np.sqrt((x.astype(np.float64) ** 2).mean()))  # noqa: E731
    bound = 2 * D * float(np.finfo(np.float32).eps) * rms(a) * rms(b)

    for name, got, want in [
        ("bind", hrr.bind(ta, tb).numpy(), np.real(np.fft.ifft(fa * fb))),
        ("unbind", hrr.unbind(ta, tb).numpy(), np.real(np.fft.ifft(fa * np.conj(fb)))),
    ]:
        worst = float(np.abs(got - want).max()) / bound
        print(f"{name}: worst error / bound = {worst:.3f}")
        assert worst <= 1.0, f"{name}: {worst:.3f} of the bound"


def test_binding_is_commutative_and_random_vectors_are_unit_length():
    g = gen(2)
    a, b = hrr.random(D, generator=g), hrr.random(D, generator=g)
    assert abs(float(a.pow(2).sum().item()) - 1.0) < 1e-5
    assert np.abs(hrr.bind(a, b).numpy() - hrr.bind(b, a).numpy()).max() < 1e-6
    assert abs(hrr.similarity(a, b)) < 0.2, "two random HRR vectors are near 0"


def test_unbinding_recovers_the_right_entry_first():
    """HRR's promise: approximate recovery that a codebook can clean up."""
    g = gen(3)
    codebook = hrr.random(D, generator=g, n=32)
    key = hrr.random(D, generator=g)
    composite = hrr.bind(codebook[9], key)
    sims = hrr.similarity_batch(hrr.unbind(composite, key), codebook).numpy()
    assert sims.argmax() == 9
    assert sims[9] - np.partition(sims, -2)[-2] > 0.3, "recovery is not decisive"


def test_a_bundle_of_bindings_still_unbinds_to_the_right_pair():
    """Two role/filler pairs superposed, then one read back by its role —
    the thing HRR is used for."""
    g = gen(4)
    fillers = hrr.random(D, generator=g, n=16)
    role_a, role_b = hrr.random(D, generator=g), hrr.random(D, generator=g)
    composite = hrr.bundle([hrr.bind(role_a, fillers[2]), hrr.bind(role_b, fillers[11])])
    assert hrr.similarity_batch(hrr.unbind(composite, role_a), fillers).numpy().argmax() == 2
    assert hrr.similarity_batch(hrr.unbind(composite, role_b), fillers).numpy().argmax() == 11


def test_a_zero_vector_gives_zero_similarity_rather_than_a_nan():
    """v1's rule, and the right one: a vector with no direction is not
    similar to anything, and must not poison an argmax with NaN."""
    zero = grilly.zeros(D)
    assert hrr.similarity(zero, hrr.random(D, generator=gen(5))) == 0.0
    sims = hrr.similarity_batch(zero, hrr.random(D, generator=gen(6), n=4)).numpy()
    assert (sims == 0.0).all()


# ── resonator ────────────────────────────────────────────────────────────


def test_a_two_factor_binding_is_recovered():
    g = gen(7)
    codebooks = {"shape": bipolar.random(D, generator=g, n=8),
                 "colour": bipolar.random(D, generator=g, n=8)}
    net = resonator.ResonatorNetwork(codebooks, restarts=200, generator=g)
    composite = bipolar.bind(codebooks["shape"][3], codebooks["colour"][5])
    indices, estimates, iterations = net.factorize(composite)
    assert indices == {"shape": 3, "colour": 5}, f"got {indices} in {iterations} iterations"
    assert (estimates["shape"].numpy() == codebooks["shape"][3].numpy()).all()


def test_more_restarts_never_do_worse():
    """The restarts keep the best reconstruction, so raising the count can
    only find the same answer or a better one. This is the one guarantee
    the search makes, and the reason the parameter is safe to raise."""
    g = gen(8)
    codebooks = {f: bipolar.random(D, generator=g, n=16) for f in ("a", "b")}
    composite = bipolar.bind(codebooks["a"][12], codebooks["b"][4])
    truth = {"a": 12, "b": 4}
    few = sum(resonator.ResonatorNetwork(codebooks, restarts=1, generator=gen(i)).factorize(composite)[0] == truth
              for i in range(8))
    many = sum(resonator.ResonatorNetwork(codebooks, restarts=64, generator=gen(i)).factorize(composite)[0] == truth
               for i in range(8))
    print(f"restarts=1: {few}/8, restarts=64: {many}/8")
    assert many >= few


def test_the_remaining_factor_needs_no_search_when_the_others_are_known():
    g = gen(9)
    codebooks = {f: bipolar.random(D, generator=g, n=8) for f in ("a", "b", "c")}
    net = resonator.ResonatorNetwork(codebooks, generator=g)
    composite = bipolar.bind(bipolar.bind(codebooks["a"][1], codebooks["b"][2]), codebooks["c"][3])
    got = net.factorize_known(composite, {"a": codebooks["a"][1], "b": codebooks["b"][2]})
    assert (got.numpy() == codebooks["c"][3].numpy()).all(), "bipolar unbinding is exact"


def test_codebooks_of_different_widths_are_refused():
    g = gen(10)
    with pytest.raises(ValueError, match="same width"):
        resonator.ResonatorNetwork({"a": bipolar.random(64, generator=g, n=4),
                                    "b": bipolar.random(128, generator=g, n=4)})
    with pytest.raises(ValueError, match="at least one codebook"):
        resonator.ResonatorNetwork({})
