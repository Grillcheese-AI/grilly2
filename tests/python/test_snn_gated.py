"""``grilly.snn.GatedLIFCell``: minGRU's recurrence as a spiking neuron.

- **dynamics: against a float64 numpy reference** of the same equations,
  silent (minGRU) and firing (soft and hard reset);
- **the scan is the recurrence:** the parallel path and the stepped path
  agree, forward and backward, including from a membrane already charged;
- **the gradient goes back through time** to the gate's weights.
"""

from __future__ import annotations

import numpy as np
import pytest

import grilly
from grilly.snn import GatedLIFCell, functional as SF


def _cell(seed=0, **kw):
    grilly.manual_seed(seed)
    return GatedLIFCell(6, 5, **kw)


def _w(cell):
    f = lambda t: t.detach().numpy().astype(np.float64)
    return f(cell.gate.weight), f(cell.gate.bias), f(cell.current.weight), f(cell.current.bias)


def _reference(cell, xs, v_threshold, v_reset):
    """float64: v = (1-z) v + z I; fire at v >= th; soft (v_reset None) or hard reset."""
    wz, bz, wh, bh = _w(cell)
    v = np.full((xs.shape[1], wz.shape[0]), 0.0 if v_reset is None else v_reset)
    outs = []
    for x in xs.astype(np.float64):
        z = 1.0 / (1.0 + np.exp(-(x @ wz.T + bz)))
        h = (1.0 - z) * v + z * (x @ wh.T + bh)
        if v_threshold is None:
            v = h
            outs.append(h)
            continue
        s = (h >= v_threshold).astype(np.float64)
        v = h - v_threshold * s if v_reset is None else h * (1.0 - s) + v_reset * s
        outs.append(s)
    return np.stack(outs), v


def _xs(seed=1, t=12, n=3, scale=2.0):
    return (np.random.default_rng(seed).standard_normal((t, n, 6)) * scale).astype(np.float32)


def test_silent_it_is_mingru():
    cell = _cell(v_threshold=None)
    xs = _xs()
    want, _ = _reference(cell, xs, None, None)
    got = np.stack([cell(grilly.from_numpy(x)).detach().numpy() for x in xs])
    assert np.abs(got - want).max() < 1e-5


@pytest.mark.parametrize("v_reset", [None, 0.0, -0.2])
def test_firing_matches_the_reference(v_reset):
    cell = _cell(v_threshold=0.3, v_reset=v_reset)
    xs = _xs(scale=3.0)
    want, v_want = _reference(cell, xs, 0.3, v_reset)
    got = np.stack([cell(grilly.from_numpy(x)).detach().numpy() for x in xs])
    assert set(np.unique(got).tolist()) <= {0.0, 1.0}, "spikes are 0 or 1"
    assert 0.02 < got.mean() < 0.98, "the test should exercise both firing and silence"
    assert (got == want).all()
    assert np.abs(cell.v.detach().numpy() - v_want).max() < 1e-5


def test_the_scan_is_the_recurrence():
    xs = _xs(t=40)
    stepped = _cell(v_threshold=None)
    want = np.stack([stepped(grilly.from_numpy(x)).detach().numpy() for x in xs])
    scanned = _cell(v_threshold=None, step_mode="m")
    got = scanned(grilly.from_numpy(xs)).detach().numpy()
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-4
    assert np.abs(scanned.v.detach().numpy() - want[-1]).max() < 1e-4


def test_the_scan_starts_from_a_charged_membrane():
    xs = _xs(t=20)
    ref = _cell(v_threshold=None)
    want = np.stack([ref(grilly.from_numpy(x)).detach().numpy() for x in xs])
    cell = _cell(v_threshold=None)
    for x in xs[:7]:
        cell(grilly.from_numpy(x))                    # charge it stepwise...
    cell.step_mode = "m"
    got = cell(grilly.from_numpy(xs[7:])).detach().numpy()   # ...then scan the rest
    assert np.abs(got - want[7:]).max() / np.abs(want).max() < 1e-4


def test_the_scan_and_the_steps_have_the_same_gradient():
    xs = _xs(t=16)
    grads = []
    for mode in ("s", "m"):
        cell = _cell(v_threshold=None, step_mode=mode)
        if mode == "s":
            out = grilly.stack([cell(grilly.from_numpy(x)) for x in xs])
        else:
            out = cell(grilly.from_numpy(xs))
        (out * out).sum().backward()
        grads.append(cell.gate.weight.grad.numpy().copy())
    assert np.abs(grads[0] - grads[1]).max() / np.abs(grads[0]).max() < 1e-3


def test_the_gradient_reaches_the_gate_through_time():
    """Spikes at the last step only: the loss sees step 0's input through the
    membrane and the surrogate, or not at all."""
    cell = _cell(v_threshold=0.3)
    xs = _xs(t=10, scale=3.0)
    x0 = grilly.from_numpy(xs[0])
    x0.requires_grad_(True)
    cell(x0)
    for x in xs[1:-1]:
        cell(grilly.from_numpy(x))
    cell(grilly.from_numpy(xs[-1])).sum().backward()
    g = cell.gate.weight.grad.numpy()
    assert np.isfinite(g).all() and np.abs(g).max() > 0
    assert x0.grad is not None and np.abs(x0.grad.numpy()).max() > 0, "step 0 reaches the last step's spikes"


def test_reset_net_forgets_it():
    cell = _cell(v_threshold=None)
    xs = _xs(t=3)
    first = cell(grilly.from_numpy(xs[0])).detach().numpy()
    cell(grilly.from_numpy(xs[1]))
    SF.reset_net(cell)
    assert cell.v is None
    assert np.abs(cell(grilly.from_numpy(xs[0])).detach().numpy() - first).max() == 0
