"""``grilly.snn.cells`` and ``grilly.snn.encoding``, and ``grilly.cvl``.

These are closed-form: a Gaussian tuning curve, a Bernoulli rate code, a
cosine feature map. So the references are the formulas themselves, in
float64 numpy, and the tolerances are what float32 costs:

- **tuning curves: 1e-6 relative.** An exponential of a sum of squares,
  which is a handful of float32 operations. Wider than that means the
  formula changed, not that the GPU rounded.
- **spike coding: statistical, and named as such.** A rate code is a
  Bernoulli sample, so the decoded value differs from the input by about
  ``sqrt(p(1-p)/t)``. The test asserts against *that* rather than a fixed
  epsilon, because an epsilon here would either be vacuous or flaky.
- **random Fourier features: the kernel they approximate**, to the Monte
  Carlo error of the feature count. The ``1/sqrt(d)`` scaling is the
  theory; the constant is measured, because the statistic is a worst-of-N
  and so lives in the tail. Across d from 1024 to 32768, the worst of 20
  pairs sits between 3.6 and 7.9 times ``1/sqrt(d)``, so the bound is 12
  and the measured ratio is printed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import grilly
from grilly import cvl
from grilly.snn import cells, encoding


def gen(seed=0):
    return grilly.Generator().manual_seed(seed)


def test_place_cell_is_a_gaussian_of_the_distance_to_each_field():
    rng = np.random.default_rng(1)
    pos = rng.standard_normal(3).astype(np.float32)
    centres = rng.standard_normal((32, 3)).astype(np.float32)
    width, top, floor = 0.7, 20.0, 0.1

    got = cells.place_cell(grilly.from_numpy(pos), grilly.from_numpy(centres), width, top, floor).numpy()
    d2 = ((pos.astype(np.float64) - centres.astype(np.float64)) ** 2).sum(-1)
    want = floor + (top - floor) * np.exp(-d2 / (2 * width * width + 1e-8))
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-6
    assert got.shape == (32,)


def test_a_place_cell_fires_hardest_at_its_own_field():
    """The property that makes it a population code: which unit is loudest
    says where the animal is."""
    g = gen(2)
    centres = grilly.rand(64, 2, generator=g) * 10
    for i in (0, 17, 63):
        rates = cells.place_cell(centres[i], centres, field_width=0.5)
        assert int(rates.argmax().item()) == i
        assert float(rates.max().item()) == pytest.approx(20.0, abs=1e-4), "at the centre, max_rate"


def test_place_cell_batches_and_a_single_position_agree():
    g = gen(3)
    centres = grilly.rand(16, 2, generator=g) * 4
    positions = grilly.rand(5, 2, generator=g) * 4
    batched = cells.place_cell(positions, centres).numpy()
    assert batched.shape == (5, 16)
    for i in range(5):
        assert (cells.place_cell(positions[i], centres).numpy() == batched[i]).all()


def test_time_cell_peaks_at_its_preferred_moment_and_leaks():
    preferred = grilly.arange(10, dtype=grilly.float32)
    rates, state = cells.time_cell(3.0, preferred, temporal_width=0.8)
    assert int(rates.argmax().item()) == 3
    assert (state.numpy() == rates.numpy()).all(), "the first state is the rates themselves"

    rates2, state2 = cells.time_cell(4.0, preferred, temporal_width=0.8, membrane_state=state)
    assert int(rates2.argmax().item()) == 4
    want = 0.9 * state.numpy() + 0.1 * rates2.numpy()
    assert np.abs(state2.numpy() - want).max() < 1e-6
    # the membrane lags the rates, which is the whole reason it exists
    assert int(state2.numpy().argmax()) == 3


def test_theta_gamma_encoding_is_the_outer_product_of_two_rhythms():
    got = cells.theta_gamma_encoding(0.05, n_theta=4, n_gamma=6, coupling_strength=0.5).numpy()
    theta = np.sin(2 * np.pi * 8.0 * 0.05 + np.linspace(0, 2 * np.pi, 4))
    gamma = np.sin(2 * np.pi * 40.0 * 0.05 + np.linspace(0, 2 * np.pi, 6))
    want = (0.5 * np.outer(theta, gamma)).ravel()
    assert got.shape == (24,)
    assert np.abs(got - want).max() < 1e-6


# ── spike coding ─────────────────────────────────────────────────────────


def test_a_rate_code_decodes_back_to_the_value_within_sampling_noise():
    """The bound is the standard error of a Bernoulli mean, not a chosen
    epsilon: with t steps the decoded value carries sqrt(p(1-p)/t) of
    noise, and four of those covers every element of a small batch."""
    g = gen(4)
    steps = 400
    x = grilly.rand(8, 64, generator=g)
    spikes = encoding.continuous_to_spikes(x, num_timesteps=steps, generator=g)
    back = encoding.spikes_to_continuous(spikes, time_window=steps).numpy()

    p = x.numpy()
    sigma = np.sqrt(np.maximum(p * (1 - p), 1e-12) / steps)
    worst = float((np.abs(back - p) / sigma).max())
    print(f"worst deviation = {worst:.2f} standard errors")
    assert worst < 5.0, f"{worst:.2f} standard errors is not sampling noise"


def test_spikes_are_zeros_and_ones_and_nothing_else():
    g = gen(5)
    spikes = encoding.continuous_to_spikes(grilly.rand(4, 16, generator=g), 20, generator=g)
    assert tuple(spikes.shape) == (4, 20, 16)
    assert set(np.unique(spikes.numpy()).tolist()) <= {0.0, 1.0}


def test_values_outside_zero_to_one_are_clamped_not_rescaled():
    """A rate cannot exceed one spike a step. Rescaling instead would make
    the code depend on the batch it arrived in."""
    g = gen(6)
    x = grilly.tensor([[-3.0, 0.0, 0.5, 1.0, 9.0]])
    spikes = encoding.continuous_to_spikes(x, num_timesteps=64, generator=g)
    rate = encoding.spikes_to_continuous(spikes, time_window=64).numpy()[0]
    assert rate[0] == 0.0 and rate[1] == 0.0
    assert rate[3] == 1.0 and rate[4] == 1.0
    assert 0.3 < rate[2] < 0.7


def test_the_decoder_reads_the_last_window_so_a_value_can_change():
    g = gen(7)
    early = encoding.continuous_to_spikes(grilly.full((1, 4), 0.0), 8, generator=g)
    late = encoding.continuous_to_spikes(grilly.full((1, 4), 1.0), 8, generator=g)
    train = grilly.cat([early, late], dim=1)
    assert (encoding.spikes_to_continuous(train, time_window=8).numpy() == 1.0).all()
    assert (encoding.spikes_to_continuous(train, time_window=16).numpy() == 0.5).all()


def test_a_projection_is_applied_before_coding_and_after_decoding():
    g = gen(8)
    w = grilly.randn(6, 4, generator=g)
    x = grilly.rand(2, 4, generator=g)
    spikes = encoding.continuous_to_spikes(x, 4, generator=g, projection_weights=w)
    assert tuple(spikes.shape) == (2, 4, 6), "the spike layer has the projection's width"
    back = encoding.spikes_to_continuous(spikes, 4, projection_weights=grilly.randn(3, 6, generator=g))
    assert tuple(back.shape) == (2, 3)


# ── random Fourier features, and the value readout on them ───────────────


def test_random_fourier_features_approximate_the_exponential_kernel():
    """``E[F(x).F(y)] = exp(x.y)`` for unit x, y — the identity the whole
    construction exists for.

    The error is Monte Carlo in the feature count, so it falls as
    ``1/sqrt(d)``; that part is derived. The constant is not, because a
    worst-of-20 is a tail statistic: measured at 3.6 to 7.9 over d from
    1024 to 32768, so the bound is 12 with room for the tail, and the ratio
    is printed so a real regression shows up as a number climbing rather
    than as a pass."""
    g = gen(9)
    d = 8192
    features = cvl.RandomFourierFeatures(in_dim=32, n_features=d, seed=0)

    def unit(seed):
        v = grilly.randn(32, generator=gen(seed))
        return v / v.pow(2).sum().sqrt()

    worst = 0.0
    for seed in range(6):
        x, y = unit(100 + seed), unit(200 + seed)
        got = float((features(x) * features(y)).sum().item())
        want = math.exp(float((x * y).sum().item()))
        worst = max(worst, abs(got - want) / want)
    bound = 12 / math.sqrt(d)
    print(f"worst relative kernel error = {worst:.4f}, {worst / bound:.2f} of the bound")
    assert worst < bound


def test_the_features_are_fixed_by_the_seed():
    a = cvl.RandomFourierFeatures(16, 256, seed=3)
    b = cvl.RandomFourierFeatures(16, 256, seed=3)
    c = cvl.RandomFourierFeatures(16, 256, seed=4)
    x = grilly.randn(16, generator=gen(10))
    assert (a(x).numpy() == b(x).numpy()).all(), "the same seed must give the same map"
    assert not (a(x).numpy() == c(x).numpy()).all()


def test_the_value_is_zero_until_something_is_observed():
    value = cvl.CVLValue()
    assert value.xi is None
    assert value.value(grilly.randn(64, generator=gen(11))) == 0.0


def test_a_rewarded_state_ranks_above_an_unrewarded_one():
    """What CVL claims: the readout is proportional to Q, so it orders
    actions correctly even though its scale means nothing."""
    features = cvl.RandomFourierFeatures(in_dim=16, n_features=4096, seed=1)
    good, bad = grilly.randn(16, generator=gen(12)), grilly.randn(16, generator=gen(13))
    good, bad = good / good.pow(2).sum().sqrt(), bad / bad.pow(2).sum().sqrt()

    value = cvl.CVLValue(gamma=0.9, beta=0.5)
    for _ in range(20):
        value.observe(features(good), reward=1.0)
        value.observe(features(bad), reward=-1.0)
    assert value.value(features(good)) > value.value(features(bad))


def test_gamma_outside_its_range_is_refused():
    """``1 / (1 - gamma)`` is the horizon; at gamma = 1 there is not one."""
    with pytest.raises(ValueError, match="gamma"):
        cvl.CVLValue(gamma=1.0)
    with pytest.raises(ValueError, match="gamma"):
        cvl.CVLValue(gamma=-0.1)


def test_features_of_the_wrong_width_are_refused():
    features = cvl.RandomFourierFeatures(in_dim=8, n_features=32, seed=0)
    with pytest.raises(ValueError, match=r"\(\.\.\., 8\)"):
        features(grilly.randn(9, generator=gen(14)))
