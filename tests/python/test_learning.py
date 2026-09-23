"""``grilly.learning`` — EWC, NLMS and whitening.

Closed-form arithmetic, so most of these check the formula against float64
numpy at **1e-6 relative** — a handful of float32 operations, and anything
wider means the formula changed rather than that the GPU rounded.

The two that are not formula checks are the ones that matter more, because
they test what the rules are *for*:

- NLMS is asked to recover a linear target from noise-free samples, which
  it must do exactly — an adaptive filter that does not converge on the
  easy case is broken whatever its arithmetic looks like;
- EWC is asked to make a parameter the old task depended on expensive to
  move, and one it did not, cheap. That ordering is the whole mechanism.
"""

from __future__ import annotations

import numpy as np
import pytest

import grilly
from grilly import learning


def gen(seed=0):
    return grilly.Generator().manual_seed(seed)


def arr(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


# ── the Fisher, and what EWC does with it ────────────────────────────────


def test_fisher_is_an_ema_of_the_squared_gradient():
    grads = grilly.from_numpy(arr(16, seed=1))
    first = learning.fisher_info(grads)
    want = 0.1 * grads.numpy().astype(np.float64) ** 2
    assert np.abs(first.numpy() - want).max() / np.abs(want).max() < 1e-6

    second = learning.fisher_info(grads, first, momentum=0.9)
    want2 = 0.9 * first.numpy().astype(np.float64) + want
    assert np.abs(second.numpy() - want2).max() / np.abs(want2).max() < 1e-6


def test_without_the_ema_the_fisher_accumulates():
    grads = grilly.from_numpy(arr(8, seed=2))
    total = None
    for _ in range(4):
        total = learning.fisher_info(grads, total, use_ema=False)
    assert np.abs(total.numpy() - 4 * grads.numpy() ** 2).max() < 1e-5


def test_reset_starts_over_and_none_means_the_same():
    grads = grilly.from_numpy(arr(8, seed=3))
    stale = learning.fisher_info(grilly.from_numpy(arr(8, seed=4) * 100))
    assert np.abs(learning.fisher_info(grads, stale, reset=True).numpy()
                  - learning.fisher_info(grads, None).numpy()).max() == 0.0


def test_fisher_normalize_sums_to_one_and_leaves_zero_alone():
    f = learning.fisher_info(grilly.from_numpy(arr(32, seed=5)))
    assert float(learning.fisher_normalize(f).sum().item()) == pytest.approx(1.0, abs=1e-5)
    zeros = grilly.zeros(8)
    assert (learning.fisher_normalize(zeros).numpy() == 0.0).all(), "no division by zero"


def test_ewc_penalises_the_parameters_the_old_task_depended_on():
    """The mechanism, not the formula: moving a high-Fisher parameter costs
    more than moving a low-Fisher one by the same amount."""
    old = grilly.zeros(2)
    fisher = grilly.tensor([10.0, 0.01])
    move_important = learning.ewc_penalty(grilly.tensor([1.0, 0.0]), old, fisher)
    move_free = learning.ewc_penalty(grilly.tensor([0.0, 1.0]), old, fisher)
    assert move_important > 100 * move_free
    assert learning.ewc_penalty(old, old, fisher) == 0.0, "staying put is free"


def test_ewc_penalty_is_the_weighted_square_it_claims_to_be():
    current, old, fisher = (grilly.from_numpy(arr(64, seed=s)) for s in (6, 7, 8))
    fisher = fisher.abs()
    got = learning.ewc_penalty(current, old, fisher, lambda_ewc=0.25)
    diff = current.numpy().astype(np.float64) - old.numpy()
    want = 0.5 * 0.25 * float((fisher.numpy().astype(np.float64) * diff * diff).sum())
    assert abs(got - want) / abs(want) < 1e-6


def test_the_natural_gradient_shrinks_the_directions_the_fisher_says_matter():
    grads = grilly.tensor([1.0, 1.0])
    scaled = learning.natural_gradient(grads, grilly.tensor([100.0, 0.01])).numpy()
    assert scaled[0] < scaled[1], "a high-Fisher direction takes a smaller step"
    exact = grads.numpy().astype(np.float64) / (np.array([100.0, 0.01]) + 1e-8)
    assert np.abs(scaled - exact).max() / np.abs(exact).max() < 1e-6


# ── NLMS ─────────────────────────────────────────────────────────────────


def test_nlms_recovers_a_linear_target_exactly():
    """Noise-free samples of a linear function: an adaptive filter has no
    excuse for not converging, so this asserts the weights, not a loss."""
    g = gen(10)
    target, true_bias = grilly.tensor([1.0, -2.0, 0.5, 3.0]), 0.25
    w, bias = grilly.zeros(4), 0.0
    for _ in range(300):
        x = grilly.randn(4, generator=g)
        y = float((x * target).sum().item()) + true_bias
        w, bias = learning.nlms_update(x, y, w, bias, mu=0.5)
    assert np.abs(w.numpy() - target.numpy()).max() < 1e-4
    assert abs(bias - true_bias) < 1e-4


def test_the_step_is_normalized_so_the_input_scale_does_not_change_it():
    """What the N in NLMS is for: the same correction whatever the features
    are scaled by, which is why one learning rate works across them."""
    small = grilly.tensor([1.0, 0.0, 0.0])
    large = small * 1000.0
    w_small, b_small = learning.nlms_update(small, 1.0, grilly.zeros(3), 0.0, mu=0.5, eps=0.0)
    w_large, b_large = learning.nlms_update(large, 1.0, grilly.zeros(3), 0.0, mu=0.5, eps=0.0)
    # the weight moves inversely with the input, so the PREDICTION moves the same
    assert float((w_small * small).sum().item()) == pytest.approx(
        float((w_large * large).sum().item()), rel=1e-4)
    assert b_small > 0 and b_large > 0


def test_nlms_predict_is_a_dot_product_and_ensemble_stacks_them():
    x, w = grilly.from_numpy(arr(16, seed=11)), grilly.from_numpy(arr(16, seed=12))
    want = float(x.numpy().astype(np.float64) @ w.numpy().astype(np.float64)) + 0.5
    assert learning.nlms_predict(x, w, 0.5) == pytest.approx(want, rel=1e-6)

    members = [grilly.from_numpy(arr(16, seed=20 + i)) for i in range(3)]
    biases = [0.0, 1.0, -1.0]
    got = learning.nlms_ensemble(x, members, biases).numpy()
    assert got.shape == (3,)
    assert got == pytest.approx([learning.nlms_predict(x, m, b) for m, b in zip(members, biases)],
                                rel=1e-6)


def test_an_ensemble_with_mismatched_lengths_is_refused():
    with pytest.raises(ValueError, match="weight vectors"):
        learning.nlms_ensemble(grilly.zeros(4), [grilly.zeros(4)], [0.0, 1.0])


def test_nlms_metrics_are_the_usual_two():
    errors = grilly.tensor([1.0, -2.0, 3.0, -4.0])
    got = learning.nlms_metrics(errors, 17)
    assert got["rmse"] == pytest.approx(float(np.sqrt(np.mean([1, 4, 9, 16]))), rel=1e-6)
    assert got["mae"] == pytest.approx(2.5, rel=1e-6)
    assert got["update_count"] == 17


# ── whitening ────────────────────────────────────────────────────────────


def test_whitening_leaves_zero_mean_and_unit_variance():
    data = grilly.from_numpy(arr(256, 8, seed=13) * 5.0 + 3.0)
    white, mean, std = learning.whitening_transform(data)
    assert np.abs(white.numpy().mean(0)).max() < 1e-5
    assert np.abs(white.numpy().std(0) - 1.0).max() < 1e-4
    assert np.abs(mean.numpy() - data.numpy().mean(0)).max() < 1e-5
    assert np.abs(std.numpy() - data.numpy().std(0)).max() < 1e-5


def test_given_statistics_are_used_rather_than_measured():
    """The reason to keep the pair: a second batch has to be whitened by
    the first one's statistics, or the two are not comparable."""
    train = grilly.from_numpy(arr(128, 4, seed=14) * 2.0 + 1.0)
    _, mean, std = learning.whitening_transform(train)
    test = grilly.from_numpy(arr(16, 4, seed=15) * 2.0 + 1.0)
    reused, mean2, std2 = learning.whitening_transform(test, mean, std)
    assert (mean2.numpy() == mean.numpy()).all() and (std2.numpy() == std.numpy()).all()
    assert (reused.numpy() == learning.whitening_apply(test, mean, std).numpy()).all()
    # measured on the small batch instead, the answer differs
    assert not np.allclose(reused.numpy(), learning.whitening_transform(test)[0].numpy())


def test_a_constant_feature_does_not_divide_by_zero():
    data = grilly.full((32, 3), 2.0)
    white, _, std = learning.whitening_transform(data)
    assert (std.numpy() == 0.0).all()
    assert np.isfinite(white.numpy()).all() and (white.numpy() == 0.0).all()
