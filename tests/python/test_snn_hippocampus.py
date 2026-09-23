"""``grilly.snn.DentateGyrus`` — sparse expansion with k-winners-take-all.

The mechanics are exact and tested as such: the weight has the sparsity it
was asked for, exactly ``k`` outputs survive, and each row comes out unit
length. Those use ``==`` or 1e-5, because they are counts and a norm.

The *point* of the layer is statistical, so it is tested statistically:
pattern separation means the codes of two similar inputs are **on average**
less similar than the inputs were. Any one pair can come out more similar —
two inputs that pick the same winners are identical afterwards — so
asserting on a single pair would be a flaky test of a true property. The
test averages, and says how many pairs it averaged over.
"""

from __future__ import annotations

import numpy as np
import pytest

import grilly
import grilly.nn.functional as F
from grilly.snn import DentateGyrus


def gen(seed=0):
    return grilly.Generator().manual_seed(seed)


def test_the_weight_has_the_sparsity_it_was_asked_for():
    dg = DentateGyrus(32, 1024, sparsity=0.02, generator=gen(1))
    w = dg.weight.detach().numpy()
    assert w.shape == (1024, 32)
    assert float((w != 0).mean()) == pytest.approx(0.02, abs=1e-3)


def test_exactly_k_outputs_survive_and_the_code_is_unit_length():
    dg = DentateGyrus(16, 512, sparsity=0.05, generator=gen(2))
    assert dg.k == 25
    out = dg(grilly.randn(8, 16, generator=gen(3))).detach().numpy()
    assert out.shape == (8, 512)
    assert set((out != 0).sum(1).tolist()) == {25}
    assert np.abs(np.linalg.norm(out, axis=1) - 1.0).max() < 1e-5


def test_the_survivors_are_the_largest_by_magnitude_and_keep_their_sign():
    """v1's rule, and the one worth stating: it ranks on ``|x|`` but keeps
    the signed value, so a strongly negative unit survives as negative."""
    dg = DentateGyrus(8, 64, sparsity=0.1, generator=gen(4))
    x = grilly.randn(1, 8, generator=gen(5))
    raw = F.linear(x, dg.weight).detach().numpy()[0]
    out = dg(x).detach().numpy()[0]

    survivors = np.flatnonzero(out)
    assert len(survivors) == dg.k
    cutoff = np.sort(np.abs(raw))[-dg.k]
    assert (np.abs(raw[survivors]) >= cutoff - 1e-6).all()
    # the kept values are the raw ones, rescaled by one positive number
    scale = out[survivors] / raw[survivors]
    assert np.abs(scale - scale[0]).max() < 1e-5 and scale[0] > 0
    assert (np.sign(out[survivors]) == np.sign(raw[survivors])).all()


def test_similar_inputs_come_out_less_similar_on_average():
    """Pattern separation, which is the reason the layer exists. Averaged
    over 40 pairs at each level, because it is a property of the ensemble
    and not of any one pair."""
    g = gen(6)
    dg = DentateGyrus(32, 1024, sparsity=0.02, generator=g)
    for noise in (0.1, 0.3, 0.6, 1.0):
        ins, outs = [], []
        for _ in range(40):
            a = grilly.randn(1, 32, generator=g)
            b = a + noise * grilly.randn(1, 32, generator=g)
            ins.append(float(F.cosine_similarity(a, b, dim=1).item()))
            outs.append(float(F.cosine_similarity(dg(a), dg(b), dim=1).item()))
        before, after = float(np.mean(ins)), float(np.mean(outs))
        print(f"noise {noise}: input cos {before:.4f} -> code cos {after:.4f}")
        assert after < before, f"no separation at noise {noise}"


def test_gradients_reach_the_surviving_units_and_no_others():
    """What makes it a layer rather than a fixed encoder — and the reason
    k-WTA is written as a mask rather than a gather."""
    dg = DentateGyrus(8, 32, sparsity=0.25, generator=gen(7))
    x = grilly.randn(1, 8, generator=gen(8))
    out = dg(x)
    out.sum().backward()
    grad = dg.weight.grad.detach().numpy()
    touched = np.flatnonzero((grad != 0).any(axis=1))
    survivors = np.flatnonzero(out.detach().numpy()[0])
    assert set(touched.tolist()) <= set(survivors.tolist()), "a dead unit got a gradient"
    assert len(touched) > 0


def test_a_sparsity_outside_its_range_is_refused():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="sparsity"):
            DentateGyrus(8, 32, sparsity=bad)


def test_it_is_an_ordinary_module():
    dg = DentateGyrus(4, 16, sparsity=0.5, generator=gen(9))
    assert "weight" in dict(dg.named_parameters())
    assert "sparsity=0.5" in repr(dg)
    state = dg.state_dict()
    twin = DentateGyrus(4, 16, sparsity=0.5, generator=gen(10))
    twin.load_state_dict(state)
    x = grilly.randn(2, 4, generator=gen(11))
    assert (twin(x).detach().numpy() == dg(x).detach().numpy()).all()
