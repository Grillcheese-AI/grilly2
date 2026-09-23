"""``grilly.snn.plasticity`` — Hebbian, Oja and STDP.

These are closed-form updates, so most checks are the formula against
float64 numpy at **1e-6 relative**. But a learning rule's formula is not
what it is *for*, so the tests that matter here are the three defining
properties, and each is asserted directly:

- **STDP is asymmetric in time.** Pre before post potentiates, post before
  pre depresses. Get that wrong and you have Hebbian learning with extra
  steps, which is exactly what v1 shipped (see the last test).
- **Oja normalizes itself.** The weight converges on the leading principal
  component of its input and stays unit length with no decay term to tune.
- **Hebbian grows without bound.** Which is the reason the other two
  exist, and is asserted rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

import grilly
from grilly.snn import plasticity


def gen(seed=0):
    return grilly.Generator().manual_seed(seed)


# ── the property that makes STDP STDP ────────────────────────────────────


def spike_train(order, steps=6):
    """One pre and one post spike, in the given order, two steps apart."""
    on, off = grilly.ones(1, 1), grilly.zeros(1, 1)
    pre_at, post_at = (1, 3) if order == "pre_first" else (3, 1)
    return [(on if t == pre_at else off, on if t == post_at else off) for t in range(steps)]


@pytest.mark.parametrize("order, sign", [("pre_first", 1.0), ("post_first", -1.0)])
def test_stdp_is_asymmetric_in_time(order, sign):
    """The whole rule. A symmetric one is Hebbian."""
    learner = plasticity.STDPLearner(pre_dim=1, post_dim=1, trace_decay=0.9,
                                     lr_pot=0.1, lr_dep=0.1)
    total = sum(float(learner.step(pre, post).item()) for pre, post in spike_train(order))
    print(f"{order}: total weight change {total:+.5f}")
    assert total * sign > 0.0, f"{order} should have moved the weight the other way"


def test_the_two_directions_are_mirror_images_at_equal_rates():
    """With ``lr_pot == lr_dep`` the same spike pair, reversed, gives the
    same magnitude the other way. It is what "asymmetric" means precisely:
    the sign follows the order and nothing else does."""
    def total(order):
        learner = plasticity.STDPLearner(1, 1, trace_decay=0.9, lr_pot=0.1, lr_dep=0.1)
        return sum(float(learner.step(p, q).item()) for p, q in spike_train(order))

    forward, backward = total("pre_first"), total("post_first")
    assert forward == pytest.approx(-backward, rel=1e-6)


def test_a_wider_gap_teaches_less():
    """The trace decays, so a pre spike long before a post one moves the
    weight less than one just before it. That is the rule's whole shape."""
    changes = []
    for gap in (1, 3, 8):
        learner = plasticity.STDPLearner(1, 1, trace_decay=0.8, lr_pot=0.1, lr_dep=0.0)
        on, off = grilly.ones(1, 1), grilly.zeros(1, 1)
        total = 0.0
        for t in range(gap + 2):
            pre = on if t == 0 else off
            post = on if t == gap else off
            total += float(learner.step(pre, post).item())
        changes.append(total)
    print(f"gap 1, 3, 8 -> {[round(c, 5) for c in changes]}")
    assert changes[0] > changes[1] > changes[2] > 0.0


def test_the_stdp_update_matches_a_float64_reference():
    rng = np.random.default_rng(1)
    pre = (rng.random((4, 12)) < 0.3).astype(np.float32)
    post = (rng.random((4, 5)) < 0.3).astype(np.float32)
    pre_t = rng.random((4, 12)).astype(np.float32)
    post_t = rng.random((4, 5)).astype(np.float32)
    decay, lr_pot, lr_dep = 0.9, 0.05, 0.03

    delta, new_pre, new_post = plasticity.stdp_update(
        *(grilly.from_numpy(a) for a in (pre, post, pre_t, post_t)),
        trace_decay=decay, lr_pot=lr_pot, lr_dep=lr_dep)

    want_pre = decay * pre_t.astype(np.float64) + pre
    want_post = decay * post_t.astype(np.float64) + post
    want = (lr_pot * (post.T.astype(np.float64) @ want_pre)
            - lr_dep * (want_post.T @ pre.astype(np.float64))) / 4
    assert np.abs(new_pre.numpy() - want_pre).max() < 1e-6
    assert np.abs(new_post.numpy() - want_post).max() < 1e-6
    assert np.abs(delta.numpy() - want).max() / np.abs(want).max() < 1e-6
    assert tuple(delta.shape) == (5, 12), "the update is shaped like the weight"


def test_the_traces_are_state_and_reset_clears_them():
    learner = plasticity.STDPLearner(3, 2, trace_decay=0.5)
    on = grilly.ones(1, 3)
    learner.step(on, grilly.zeros(1, 2))
    assert float(learner.pre_trace.sum().item()) > 0.0
    learner.reset()
    assert float(learner.pre_trace.sum().item()) == 0.0
    assert float(learner.post_trace.sum().item()) == 0.0


def test_a_decay_outside_its_range_is_refused():
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="trace_decay"):
            plasticity.STDPLearner(2, 2, trace_decay=bad)


# ── Oja ──────────────────────────────────────────────────────────────────


def test_oja_finds_the_leading_principal_component_and_stays_unit_length():
    """What the rule is for, and it needs no decay term to get there."""
    rng = np.random.default_rng(0)
    axis = np.array([0.6, 0.8], np.float32)
    memory = grilly.from_numpy((rng.standard_normal((1, 2)) * 0.1).astype(np.float32))
    for _ in range(4000):
        x = (rng.standard_normal() * axis).astype(np.float32).reshape(1, 2)
        memory = plasticity.oja_update(memory, grilly.from_numpy(x), eta=0.01)

    v = memory.numpy()[0]
    assert abs(float(np.linalg.norm(v)) - 1.0) < 0.02, "Oja normalizes itself"
    assert abs(float(v @ axis) / float(np.linalg.norm(v))) > 0.99, "and finds the axis"


def test_the_oja_update_matches_its_formula():
    rng = np.random.default_rng(2)
    m = rng.standard_normal((6, 8)).astype(np.float32)
    x = rng.standard_normal((6, 8)).astype(np.float32)
    got = plasticity.oja_update(grilly.from_numpy(m), grilly.from_numpy(x), eta=0.05).numpy()

    md, xd = m.astype(np.float64), x.astype(np.float64)
    y = (md * xd).sum(-1, keepdims=True)
    want = md + 0.05 * y * (xd - y * md)
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-6


def test_one_input_can_be_shown_to_every_row():
    m = grilly.randn(4, 3, generator=gen(3))
    one = grilly.randn(3, generator=gen(4))
    assert tuple(plasticity.oja_update(m, one).shape) == (4, 3)
    with pytest.raises(ValueError, match=r"\(\.\.\., 3\)"):
        plasticity.oja_update(m, grilly.randn(5, generator=gen(5)))


# ── Hebbian ──────────────────────────────────────────────────────────────


def test_hebbian_correlates_what_fires_together():
    pre = grilly.from_numpy(np.array([[[1.0, 0.0], [1.0, 0.0]]], np.float32))
    post = grilly.from_numpy(np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], np.float32))
    got = plasticity.hebbian_update(pre, post, grilly.zeros(3, 2), lr=0.5).numpy()
    want = np.zeros((3, 2), np.float32)
    want[0, 0] = 0.5
    assert (got == want).all(), "only the pair that fired together moved"


def test_hebbian_grows_without_bound_and_decay_is_what_stops_it():
    """The rule's known failure, asserted rather than assumed — it is the
    reason Oja exists."""
    pre, post = grilly.ones(1, 4), grilly.ones(1, 3)
    plain, decayed = grilly.zeros(3, 4), grilly.zeros(3, 4)
    for _ in range(500):
        plain = plasticity.hebbian_update(pre, post, plain, lr=0.05)
        decayed = plasticity.hebbian_update(pre, post, decayed, lr=0.05, weight_decay=0.05)
    assert float(plain.max().item()) > 20.0, "unbounded"
    assert float(decayed.max().item()) == pytest.approx(1.0, abs=0.05), "decay fixes it at lr/decay"


def test_a_mismatched_weight_is_refused():
    pre, post = grilly.ones(2, 4), grilly.ones(2, 3)
    with pytest.raises(ValueError, match=r"\(3, 4\)"):
        plasticity.hebbian_update(pre, post, grilly.zeros(4, 3))


# ── what v1 actually did ─────────────────────────────────────────────────


def test_v1s_stdp_rule_had_no_timing_asymmetry():
    """Not a test of grilly2 — a record of why the old name raises.

    v1's shader computed ``ltp += pre_t * post_t`` and
    ``ltd += post_t * pre_t``: the same product. Whatever the rates, the
    update is a multiple of one correlation, so reversing the spike order
    cannot change its sign. Reproduced here in numpy so the claim is
    checkable rather than a reading of the source.
    """
    def v1_delta(order, lr_pot=0.1, lr_dep=0.03, decay=0.9):
        pre_t = post_t = 0.0
        total = 0.0
        for pre, post in [(p.item(), q.item()) for p, q in spike_train(order)]:
            pre_t = decay * pre_t + (1 - decay) * pre
            post_t = decay * post_t + (1 - decay) * post
            ltp = pre_t * post_t
            ltd = post_t * pre_t          # v1's second line, verbatim
            total += lr_pot * ltp - lr_dep * ltd
        return total

    forward, backward = v1_delta("pre_first"), v1_delta("post_first")
    assert forward == backward, "v1's rule cannot tell the two orders apart"
    assert forward > 0, "and it potentiates either way, which is Hebbian"

    # grilly2's does tell them apart, which is the point
    def ours(order):
        learner = plasticity.STDPLearner(1, 1, trace_decay=0.9, lr_pot=0.1, lr_dep=0.03)
        return sum(float(learner.step(p, q).item()) for p, q in spike_train(order))

    assert ours("pre_first") > 0 > ours("post_first")
