"""``grilly.vsa.packed`` — binary hypervectors, 32 bits to a word.

**Tolerance: none.** Every operation here is integer — XOR, a majority
vote, a popcount, a bit rotation — so every result is exact and the tests
use ``==``. A tolerance in this file would mean an op was wrong, not that
float32 was.

The reference is numpy doing the same thing on unpacked bits, which is also
what v1's ``_bridge.hdc_*_packed`` did: those never reached the GPU
("GPU path not yet wired into grilly_core"), so this port is the first time
the packed algebra runs on a card. The semantics are v1's, including
majority's tie rule.

This is **not** the block-code algebra (``test_vsa.py``). Different dtype,
different bind, different similarity baseline — 0.5 here means "unrelated",
where a block code's is 1/l (CLAUDE.md §8.3).
"""

from __future__ import annotations

import numpy as np
import pytest

import grilly
from grilly.vsa import packed

DIM = 2048
WORDS = DIM // 32


def bits(n, seed):
    """``(n, DIM)`` random bits, and the packed tensor holding them."""
    b = np.random.default_rng(seed).integers(0, 2, size=(n, DIM)).astype(np.uint8)
    return b, packed.from_bits(b)


def test_packing_round_trips_through_numpys_layout():
    b, x = bits(4, 1)
    assert tuple(x.shape) == (4, WORDS) and x.dtype == grilly.int32
    assert (packed.to_bits(x, DIM) == b).all()
    # the layout is numpy's own, which is what makes v1's files readable
    want = np.packbits(b, axis=-1, bitorder="little").view(np.int32)
    assert (x.numpy() == want).all()


def test_binding_is_xor_and_is_its_own_inverse():
    b, _ = bits(2, 2)
    x, y = packed.from_bits(b[0]), packed.from_bits(b[1])
    bound = packed.bind(x, y)
    assert (packed.to_bits(bound, DIM) == (b[0] ^ b[1])).all()
    assert (packed.unbind(bound, y).numpy() == x.numpy()).all()
    assert (packed.unbind(bound, x).numpy() == y.numpy()).all()


def test_similarity_is_hamming_and_unrelated_vectors_sit_at_a_half():
    b, x = bits(2, 3)
    assert packed.similarity(x[0], x[0]) == 1.0
    want = 1.0 - np.count_nonzero(b[0] ^ b[1]) / DIM
    assert packed.similarity(x[0], x[1]) == pytest.approx(want, abs=1e-12)
    assert abs(packed.similarity(x[0], x[1]) - 0.5) < 0.05, "random vectors are near 0.5, not 0"
    # the complement is 0.0, which is as far from 0.5 as identity is
    assert packed.similarity(x[0], packed.from_bits(1 - b[0])) == 0.0


def test_bundling_is_a_majority_vote_with_v1s_tie_rule():
    for n in (3, 4, 5, 8):                      # odd and even: ties only happen on even
        b, x = bits(n, 10 + n)
        got = packed.to_bits(packed.bundle(x), DIM)
        want = (b.sum(axis=0) * 2 > n).astype(np.uint8)
        assert (got == want).all(), f"n = {n}"


def test_a_bundle_is_nearest_to_its_members():
    """What bundling is for: the superposition stays similar to everything
    in it and unrelated to everything else."""
    _, cb = bits(64, 20)
    bundled = packed.bundle(cb[:5])
    sims = packed.similarity_batch(bundled, cb, DIM).numpy()
    assert sims[:5].min() > 0.6 and sims[5:].max() < 0.6, f"{sims[:5]} vs {sims[5:].max()}"


def test_permuting_is_numpys_roll_over_the_bits_and_reverses():
    b, x = bits(1, 30)
    for shift in (0, 1, 31, 32, 33, DIM - 1, DIM, DIM + 7, -5):
        got = packed.to_bits(packed.permute(x[0], shift, DIM), DIM)
        assert (got == np.roll(b[0], shift)).all(), f"shift {shift}"
    rotated = packed.permute(x[0], 7, DIM)
    assert (packed.permute(rotated, -7, DIM).numpy() == x[0].numpy()).all()
    assert packed.similarity(rotated, x[0]) < 0.55, "a permuted vector is unrelated to itself"


def test_top_k_returns_the_nearest_rows_nearest_first():
    b, cb = bits(200, 40)
    query = cb[7]
    indices, distances, sims = packed.topk(query, cb, k=5, dim=DIM)
    want = np.sort(np.count_nonzero(b ^ b[7], axis=1))[:5]
    assert indices.numpy()[0] == 7 and distances.numpy()[0] == 0
    assert (distances.numpy() == want).all()
    assert (sims.numpy() == (1.0 - want.astype(np.float32) / DIM)).all()
    assert list(distances.numpy()) == sorted(distances.numpy()), "nearest first"


def test_k_larger_than_the_codebook_is_the_whole_codebook():
    _, cb = bits(3, 50)
    indices, _, _ = packed.topk(cb[0], cb, k=99, dim=DIM)
    assert indices.numel() == 3


def test_a_dim_that_is_not_a_multiple_of_32_is_refused():
    """A partly used last word would make two vectors of the same dim
    compare differently depending on what the padding held."""
    with pytest.raises(ValueError, match="multiple of 32"):
        packed.random(100)
    with pytest.raises(ValueError, match="multiple of 32"):
        packed.from_bits(np.zeros((100,), dtype=np.uint8))


def test_the_two_algebras_do_not_accept_each_others_vectors():
    """v1 let a capability check for one gate the other, which is where the
    confusion started (CLAUDE.md §8.3). They must not silently interoperate."""
    from grilly import vsa

    block = vsa.random_discrete(8, 32)
    with pytest.raises(ValueError, match="int32 words"):
        packed.bind(block, block)
    with pytest.raises(Exception):
        vsa.bind(packed.random(256), packed.random(256))


# ── bipolar: the same vectors, dense ──────────────────────────────────────


def test_bipolar_bind_is_a_product_and_its_own_inverse():
    from grilly.vsa import bipolar

    g = grilly.Generator().manual_seed(0)
    a, b = bipolar.random(DIM, generator=g), bipolar.random(DIM, generator=g)
    assert (bipolar.bind(a, b).numpy() == a.numpy() * b.numpy()).all()
    assert (bipolar.unbind(bipolar.bind(a, b), b).numpy() == a.numpy()).all()
    assert bipolar.similarity(a, a) == 1.0
    assert abs(bipolar.similarity(a, b)) < 0.1, "random bipolar vectors sit near 0, not 0.5"


def test_bipolar_and_packed_rank_a_codebook_identically():
    """The relation that survives the representation change, and the reason
    a large store can hold bits and still be searched: similarity maps
    linearly, so the ORDER is the same."""
    from grilly.vsa import bipolar

    g = grilly.Generator().manual_seed(1)
    cb = bipolar.random(DIM, generator=g, n=32)
    query = cb[7]
    dense = bipolar.similarity_batch(query, cb).numpy()
    bits = packed.similarity_batch(bipolar.pack(query), bipolar.pack(cb), DIM).numpy()
    assert np.abs(dense - (2 * bits - 1)).max() < 1e-6
    assert (np.argsort(-dense) == np.argsort(-bits)).all()


def test_packing_round_trips_and_uses_v1s_convention():
    """``+1`` sets the bit, which is what v1's ``vsa_bitpack`` wrote, so a
    packed store written by v1 reads back as the same vectors."""
    from grilly.vsa import bipolar

    g = grilly.Generator().manual_seed(2)
    x = bipolar.random(DIM, generator=g)
    bits = packed.to_bits(bipolar.pack(x), DIM)
    assert (bits == (x.numpy() > 0).astype(np.uint8)).all()
    assert (bipolar.unpack(bipolar.pack(x), DIM).numpy() == x.numpy()).all()


def test_an_unnormalized_bundle_keeps_the_counts():
    """What the dense representation is for: the packed one cannot hold
    how many vectors voted each way."""
    from grilly.vsa import bipolar

    g = grilly.Generator().manual_seed(3)
    vectors = [bipolar.random(64, generator=g) for _ in range(5)]
    raw = bipolar.bundle(vectors, normalize=False).numpy()
    assert set(np.unique(raw)) <= {-5.0, -3.0, -1.0, 1.0, 3.0, 5.0}
    voted = bipolar.bundle(vectors).numpy()
    assert (voted == np.where(raw < 0, -1.0, 1.0)).all()


def test_from_key_is_deterministic_across_processes():
    from grilly.vsa import bipolar

    v = bipolar.from_key("cat", 256)
    assert (v.numpy() == bipolar.from_key("cat", 256).numpy()).all()
    assert set(np.unique(v.numpy())) <= {-1.0, 1.0}
    assert bipolar.similarity(v, bipolar.from_key("dog", 256)) < 0.3
