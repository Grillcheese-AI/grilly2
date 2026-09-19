"""torch.einsum against a float64 reference.

Tolerance: |got - ref| <= 2 sqrt(T) eps * einsum(|operands|), with T an
upper bound on the products summed into each output (the product of the
operands' sizes over the output's size): einsum is sums and one bmm per
pair, each within sqrt(K) eps of its magnitudes. Gradients use the same
bound against torch float64.
"""

import math

import numpy as np
import pytest
import torch

import grilly

EPS = np.finfo(np.float32).eps
RNG = np.random.default_rng(12)

CASES = [
    # CubbyLLM (cubbyllm/core/generation.py, model/recall/read.py)
    ("bn,nrd->brd", [(3, 5), (5, 4, 6)]),
    ("bn,ndr->bdr", [(3, 5), (5, 6, 4)]),
    ("...d,rd->...r", [(2, 7, 6), (4, 6)]),
    ("...d,rd->...r", [(6,), (4, 6)]),
    ("bdr,brk->bdk", [(3, 6, 4), (3, 4, 6)]),
    ("bik,bjk->bij", [(2, 5, 8), (2, 7, 8)]),
    # its validation scripts
    ("bsd,btd->bst", [(2, 6, 8), (2, 6, 8)]),
    ("bst,btd->bsd", [(2, 6, 6), (2, 6, 8)]),
    ("boh,bh->bo", [(3, 4, 5), (3, 5)]),
    # the general machinery
    ("ij,jk", [(2, 3), (3, 4)]), ("ij,jk->", [(2, 3), (3, 4)]), ("i,i", [(3,), (3,)]),
    ("i,j->ij", [(3,), (4,)]), ("ij->ji", [(2, 3)]), ("ijk->j", [(2, 3, 4)]), ("ij", [(2, 3)]),
    ("ij,jk,kl->il", [(2, 3), (3, 4), (4, 5)]), ("bij,j->bi", [(2, 3, 4), (4,)]),
    ("...ij,...jk->...ik", [(2, 1, 3, 4), (5, 4, 2)]), ("ab,ab->a", [(4, 5), (1, 5)]),
    ("ba,ab->a", [(3, 2), (2, 3)]), ("i j , j k -> i k", [(2, 3), (3, 4)]), ("abc,cd->", [(2, 3, 4), (4, 5)]),
    (",->", [(), ()]), ("i,->i", [(3,), ()]),
]


@pytest.mark.parametrize("eq,shapes", CASES)
def test_einsum_matches_torch(eq, shapes):
    xs = [RNG.uniform(-1, 1, s).astype(np.float32) for s in shapes]
    g = [grilly.tensor(x, requires_grad=True) for x in xs]
    t64 = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in xs]
    got, ref = grilly.einsum(eq, *g), torch.einsum(eq, *t64)
    assert tuple(got.shape) == tuple(torch.einsum(eq, *[torch.tensor(x) for x in xs]).shape)
    mag_in = [torch.tensor(np.abs(x), dtype=torch.float64, requires_grad=True) for x in xs]
    mag = torch.einsum(eq, *mag_in)
    terms = max(1, math.prod(max(x.size, 1) for x in xs) // max(ref.numel(), 1))
    bound = 2 * math.sqrt(terms) * EPS
    assert np.all(np.abs(got.detach().numpy() - ref.detach().numpy()) <= bound * mag.detach().numpy() + 1e-30)
    w = RNG.uniform(-1, 1, tuple(ref.shape))
    (got * grilly.tensor(w.astype(np.float32))).sum().backward()
    (ref * torch.tensor(w)).sum().backward()
    (mag * torch.tensor(np.abs(w))).sum().backward()
    for a, b, m in zip(g, t64, mag_in):
        assert np.all(np.abs(a.grad.numpy() - b.grad.numpy()) <= bound * m.grad.numpy() + 1e-30)


def test_operand_list_form():
    a, b = RNG.standard_normal((2, 3)).astype(np.float32), RNG.standard_normal((3, 4)).astype(np.float32)
    np.testing.assert_allclose(grilly.einsum("ij,jk", [grilly.tensor(a), grilly.tensor(b)]).numpy(),
                               torch.einsum("ij,jk", [torch.tensor(a), torch.tensor(b)]).numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("eq,shapes,match", [
    ("ij,jk->ik", [(2, 3), (4, 4)], "subscript j has size 4 for operand 1 which does not broadcast"),
    ("ijk,jk->i", [(2, 3), (3, 4)], "does not match the number of dimensions"),
    ("ij->ik", [(2, 3)], "output subscript k does not appear"),
    ("ij,jk->ik", [(2, 3)], "fewer operands"),
    ("i,i->ii", [(2,), (2,)], "appears more than once in the output"),
])
def test_errors_match_torch(eq, shapes, match):
    for lib in (torch, grilly):
        with pytest.raises(RuntimeError, match=match):
            lib.einsum(eq, *[lib.ones(*s) for s in shapes])


def test_diagonal_is_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        grilly.einsum("ii->i", grilly.ones(3, 3))
