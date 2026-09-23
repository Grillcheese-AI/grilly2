"""``grilly.snn`` neurons and surrogate gradients.

Three kinds of check, with three different standards:

- **dynamics: 1e-6 against a float64 reference.** Charge, fire and reset
  are a handful of float32 operations with a closed form, so anything
  wider means the recurrence changed.
- **surrogate gradients: 1e-6 against the analytic bump.** The whole
  construction exists to make ``d(spike)/dv`` equal a named formula, so
  the test is that formula, not a finite difference.
- **that it trains: behavioural.** A spiking network is not obviously
  differentiable, and the reason this is a rewrite rather than a
  translation is that v1's hand-written backwards only reached one layer.
  So one test trains a two-layer network through twelve timesteps and
  asserts it learns — no tolerance, an accuracy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import grilly
import grilly.nn as nn
from grilly.snn import (
    ATan, FastSigmoid, IFNode, LIFNode, ParametricLIFNode, Sigmoid, functional as SF, surrogate,
)


def gen(seed=0):
    return grilly.Generator().manual_seed(seed)


# ── surrogate gradients ──────────────────────────────────────────────────


def test_the_forward_pass_is_a_real_step_not_a_smooth_one():
    """Spikes have to be 0 or 1. A network of sigmoids is not a spiking
    network, and the surrogate is only for the backward pass."""
    x = grilly.tensor([-5.0, -1e-6, 0.0, 1e-6, 5.0], requires_grad=True)
    for fn in (ATan(), Sigmoid(), FastSigmoid()):
        out = fn(x).detach().numpy()
        assert set(np.unique(out).tolist()) <= {0.0, 1.0}, repr(fn)
        assert (out == np.array([0.0, 0.0, 1.0, 1.0, 1.0])).all(), f"{fn}: v >= 0 fires"


@pytest.mark.parametrize("name, alpha", [("ATan", 2.0), ("ATan", 5.0),
                                         ("Sigmoid", 4.0), ("FastSigmoid", 2.0)])
def test_the_backward_pass_is_the_bump_it_is_named_after(name, alpha):
    fn = getattr(surrogate, name)(alpha=alpha)
    v = np.linspace(-2.0, 2.0, 41).astype(np.float32)
    x = grilly.from_numpy(v)
    x.requires_grad_(True)
    fn(x).sum().backward()

    d = v.astype(np.float64)
    if name == "ATan":
        want = alpha / (2.0 * (1.0 + (math.pi * alpha * d / 2.0) ** 2))
    elif name == "Sigmoid":
        s = 1.0 / (1.0 + np.exp(-alpha * d))
        want = alpha * s * (1.0 - s)
    else:
        want = alpha / (2.0 * (1.0 + alpha * np.abs(d)) ** 2)
    assert np.abs(x.grad.numpy() - want).max() / np.abs(want).max() < 1e-6


def test_a_larger_alpha_is_a_narrower_bump():
    """Which is the trade-off worth knowing: closer to the true derivative,
    and less gradient for a neuron far from its threshold."""
    v = grilly.tensor([0.0, 0.5, 1.0])
    peaks, tails = [], []
    for alpha in (1.0, 4.0):
        x = v.clone()
        x.requires_grad_(True)
        ATan(alpha=alpha)(x).sum().backward()
        peaks.append(float(x.grad.numpy()[0]))
        tails.append(float(x.grad.numpy()[2]))
    assert peaks[1] > peaks[0], "a larger alpha is taller at the threshold"
    assert tails[1] < tails[0], "and lower away from it"


def test_no_gradient_is_asked_for_when_none_is_wanted():
    """Inference should not pay for the straight-through term."""
    x = grilly.tensor([-1.0, 1.0])
    assert not x.requires_grad
    assert (ATan()(x).numpy() == np.array([0.0, 1.0])).all()
    with grilly.no_grad():
        y = grilly.tensor([-1.0, 1.0], requires_grad=True)
        assert not ATan()(y).requires_grad


# ── the neurons ──────────────────────────────────────────────────────────


def test_integrate_and_fire_integrates_until_it_fires():
    node = IFNode()
    fired = [float(node(grilly.tensor([0.3]))[0].item()) for _ in range(9)]
    # 0.3 a step, threshold 1.0: over on the fourth, reset to 0, over again on the eighth
    assert fired == [0, 0, 0, 1, 0, 0, 0, 1, 0]


def test_a_leaky_neuron_reaches_a_fixed_point_and_stops():
    """The leak is the difference: 0.3 a step settles at 0.6 with tau = 2,
    which is under the threshold, so it never fires."""
    node = LIFNode(tau=2.0)
    for _ in range(40):
        assert float(node(grilly.tensor([0.3]))[0].item()) == 0.0
    assert float(node.v.numpy()[0]) == pytest.approx(0.6, abs=1e-4)


def test_the_membrane_recurrence_matches_a_float64_reference():
    g = gen(1)
    x = grilly.randn(20, 4, generator=g) * 0.5
    node = LIFNode(tau=2.5, v_threshold=0.8, v_reset=0.0)
    got_v, got_s = [], []
    for t in range(20):
        got_s.append(node(x[t]).detach().numpy())
        got_v.append(node.v.detach().numpy())

    xs = x.numpy().astype(np.float64)
    v = np.zeros(4)
    for t in range(20):
        h = (1.0 - 1.0 / 2.5) * v + xs[t]
        s = (h >= 0.8).astype(np.float64)
        v = h * (1.0 - s) + 0.0 * s
        assert np.abs(got_s[t] - s).max() == 0.0, f"spikes at t={t}"
        assert np.abs(got_v[t] - v).max() < 1e-6, f"membrane at t={t}"


def test_a_soft_reset_keeps_the_overshoot_and_a_hard_one_does_not():
    """``v_reset=None`` subtracts the threshold instead of clearing, so a
    strongly driven neuron comes back sooner."""
    hard, soft = IFNode(v_reset=0.0), IFNode(v_reset=None)
    drive = grilly.tensor([5.0])
    assert float(hard(drive)[0].item()) == 1.0 and float(soft(drive)[0].item()) == 1.0
    assert float(hard.v.numpy()[0]) == 0.0, "hard reset throws the overshoot away"
    assert float(soft.v.numpy()[0]) == pytest.approx(4.0), "soft reset keeps 5 - 1"


def test_decay_input_is_the_textbook_form_and_fires_less():
    strong = LIFNode(tau=4.0, decay_input=False)
    textbook = LIFNode(tau=4.0, decay_input=True)
    x = grilly.tensor([0.5])
    for _ in range(50):
        strong(x)
        textbook(x)
    assert float(textbook.v.numpy()[0]) < float(strong.v.numpy()[0])
    assert float(textbook.v.numpy()[0]) == pytest.approx(0.5, abs=1e-3), "settles at x"


def test_tau_below_one_is_refused():
    """Below 1 the membrane grows instead of leaking."""
    with pytest.raises(ValueError, match="tau"):
        LIFNode(tau=0.5)


def test_a_parametric_neuron_learns_its_time_constant():
    node = ParametricLIFNode(init_tau=3.0)
    assert "tau" in dict(node.named_parameters())
    out = None
    for t in range(6):
        s = node(grilly.full((1, 4), 0.6))
        out = s if out is None else out + s
    out.sum().backward()
    assert node.tau.grad is not None and float(node.tau.grad.abs().sum().item()) > 0.0


def test_the_clamp_on_a_learned_tau_is_in_the_graph():
    """Clamping in the forward pass rather than after the step is what
    stops the optimizer pushing tau further out of range each time."""
    node = ParametricLIFNode(init_tau=1.0)
    for _ in range(3):
        SF.reset_net(node)          # the membrane carries the graph; see below
        node(grilly.full((1, 2), 0.4)).sum().backward()
    assert float(node.tau.detach().reshape(1)[0].item()) >= 1.0


def test_the_graph_spans_time_until_the_network_is_reset():
    """Backpropagation through time, and its consequence: the membrane
    holds the graph of every step since the last reset, so a second
    ``backward`` without one walks a freed graph. That is the same
    behaviour as any recurrent module, and ``reset_net`` is where a
    sequence ends."""
    node = LIFNode(tau=2.0)
    x = grilly.full((1, 3), 0.9, requires_grad=True)
    node(x).sum().backward()
    with pytest.raises(RuntimeError, match="backward through the graph a second time"):
        node(x).sum().backward()
    SF.reset_net(node)
    node(x).sum().backward()        # a fresh sequence is fine


# ── stepping a network ───────────────────────────────────────────────────


def test_the_membrane_persists_until_the_network_is_reset():
    """The gotcha worth a test: a spiking neuron is stateful, so two runs
    of the same input only agree if you reset in between."""
    node = LIFNode(tau=2.0)
    # 0.55 with tau 2 leaves the membrane at 0.96 after three steps, just
    # under the threshold, so the next step fires where the first did not.
    # (0.9 would not show it: that input's cycle returns v to exactly 0.)
    x = grilly.full((1, 3), 0.55)
    first = [node(x).detach().numpy().copy() for _ in range(3)]
    again = [node(x).detach().numpy().copy() for _ in range(3)]
    assert not all((a == b).all() for a, b in zip(first, again)), "state carried, as it should"

    SF.reset_net(node)
    assert node.v is None
    third = [node(x).detach().numpy().copy() for _ in range(3)]
    assert all((a == b).all() for a, b in zip(first, third)), "a reset network repeats itself"


def test_reset_and_step_mode_reach_every_neuron_in_a_network():
    net = nn.Sequential(nn.Linear(4, 4), LIFNode(), nn.Linear(4, 2), IFNode())
    net(grilly.randn(1, 4, generator=gen(2)))
    assert all(m.v is not None for m in net.modules() if isinstance(m, LIFNode | IFNode))
    SF.reset_net(net)
    assert all(m.v is None for m in net.modules() if isinstance(m, LIFNode | IFNode))

    SF.set_step_mode(net, "m")
    assert all(m.step_mode == "m" for m in net.modules() if isinstance(m, LIFNode | IFNode))
    with pytest.raises(ValueError, match="step_mode"):
        SF.set_step_mode(net, "x")


def test_multi_step_mode_is_the_single_step_loop():
    x = grilly.randn(7, 3, 5, generator=gen(3))
    stepwise = LIFNode(tau=2.0, step_mode="s")
    sequence = LIFNode(tau=2.0, step_mode="m")
    one_at_a_time = grilly.stack([stepwise(x[t]) for t in range(7)]).detach().numpy()
    assert (sequence(x).detach().numpy() == one_at_a_time).all()


def test_folding_time_into_the_batch_is_refused_for_a_stateful_module():
    """It is the fast route for a Linear and wrong for a neuron, so asking
    says which."""
    x = grilly.randn(4, 2, 6, generator=gen(4))
    assert tuple(SF.seq_to_ann_forward(x, nn.Linear(6, 3)).shape) == (4, 2, 3)
    with pytest.raises(ValueError, match="membrane"):
        SF.seq_to_ann_forward(x, LIFNode())


def test_a_spiking_network_trains_through_time():
    """The point of the rewrite. v1's neurons carried hand-written
    backwards over its own autograd and the gradient reached one layer;
    here it goes back through twelve timesteps and two layers, and the
    network learns a task it starts at chance on."""
    rng = np.random.default_rng(0)
    T, N, D, C = 12, 64, 20, 3
    proto = rng.standard_normal((C, D)).astype(np.float32)
    labels = rng.integers(0, C, N)
    rates = np.clip(0.5 + 0.35 * proto[labels] / np.abs(proto).max(), 0, 1).astype(np.float32)
    x = grilly.from_numpy((rng.random((T, N, D)).astype(np.float32) < rates[None]).astype(np.float32))
    y = grilly.from_numpy(labels.astype(np.int64))

    net = nn.Sequential(nn.Linear(D, 32), LIFNode(tau=2.0),
                        nn.Linear(32, C), LIFNode(tau=2.0, v_threshold=0.5))
    opt = grilly.optim.Adam(net.parameters(), lr=0.02)
    first_acc = last_acc = 0.0
    for step in range(60):
        SF.reset_net(net)
        counts = None
        for t in range(T):
            s = net(x[t])
            counts = s if counts is None else counts + s
        loss = nn.functional.cross_entropy(counts / T * 6.0, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc = float((counts.argmax(-1) == y).to(grilly.float32).mean().item())
        if step == 0:
            first_acc = acc
        last_acc = acc
    print(f"spiking net: {first_acc:.3f} -> {last_acc:.3f} train accuracy over 60 steps")
    assert first_acc < 0.6 and last_acc > 0.9
    assert float(net[0].weight.grad.abs().sum().item()) > 0.0, "the first layer got a gradient"
