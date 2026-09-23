"""The v1 compatibility shims (``grilly._v1`` and the aliases under it).

grilly2 replaces grilly v1 in place — same distribution, same import name —
so v1's import paths have to keep working while they are moved. Two kinds
of name, and the difference between them is the point of this file:

- **ported**: the alias keeps v1's exact signature, numpy in and numpy out
  where v1 took numpy, warns once, and gives the same answer the new API
  does. Tested by ``==`` against ``grilly.vsa``: the alias is a reshape and
  a transfer over the same kernels, so there is nothing for a tolerance to
  absorb.
- **replaced**: v1's parallel framework. Raises ``NotImplementedError``,
  and — the part that matters — ``hasattr`` does **not** quietly answer
  False, because a capability check that falls through silently is how v1's
  own ``_detect_backend`` ended up reporting a tier it never used.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import warnings

import numpy as np
import pytest

import grilly
from grilly import vsa

K, L = 8, 32


def v1_module(relative: str, name: str):
    """A pure-Python module from a v1 checkout, or skip."""
    env = os.environ.get("GRILLY_V1_DIR")
    root = pathlib.Path(env) if env else pathlib.Path(__file__).resolve().parents[2].parent / "grilly"
    path = root.joinpath(*relative.split("/"))
    if not path.is_file():
        pytest.skip(f"no v1 checkout with {relative} (set GRILLY_V1_DIR)")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def one_hot_flat(n, seed):
    """v1's bridge layout: ``(n, K * L)`` float32, one hot per block."""
    rng = np.random.default_rng(seed)
    hot = rng.integers(0, L, size=(n, K))
    a = np.zeros((n, K, L), dtype=np.float32)
    np.put_along_axis(a, hot[:, :, None], 1.0, axis=2)
    return a.reshape(n, K * L), hot


def quiet():
    return warnings.catch_warnings()


# ── the tier check a downstream consumer actually makes ──────────────────


def test_a_v1_capability_check_finds_the_bridge():
    """CubbyLLM's ``cubbyllm/ops/vsa.py`` does exactly this, unedited, and
    must come out on the fast tier."""
    from grilly.backend import _bridge

    assert hasattr(_bridge, "blockcode_bind")
    assert _bridge.is_available()
    from grilly.experimental.vsa.block_ops import BlockCodeOps

    assert callable(BlockCodeOps.bind)


# ── ported names keep v1's signature ─────────────────────────────────────


def test_the_bridge_takes_flat_vectors_and_gives_the_same_answer():
    from grilly.backend import _bridge

    a, ha = one_hot_flat(4, 1)
    b, hb = one_hot_flat(4, 2)
    with quiet():
        warnings.simplefilter("ignore", DeprecationWarning)
        bound = _bridge.blockcode_bind(a, b, K, L)
        assert bound.shape == (4, K * L) and bound.dtype == np.float32
        recovered = _bridge.blockcode_unbind(bound, b, K, L)

    want = np.zeros((4, K, L), dtype=np.float32)
    np.put_along_axis(want, ((ha + hb) % L)[:, :, None], 1.0, axis=2)
    assert (bound == want.reshape(4, K * L)).all()
    assert (recovered == a).all()


def test_a_one_dimensional_vector_comes_back_one_dimensional():
    """v1 squeezed a flat input back to flat, and callers rely on it."""
    from grilly.backend import _bridge

    a, _ = one_hot_flat(1, 3)
    with quiet():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert _bridge.blockcode_bind(a[0], a[0], K, L).shape == (K * L,)
        assert _bridge.blockcode_bind(a, a, K, L).shape == (1, K * L)


def test_block_code_ops_agrees_with_the_new_api():
    from grilly.experimental.vsa.block_ops import BlockCodeOps

    a, _ = one_hot_flat(2, 4)
    b, _ = one_hot_flat(2, 5)
    a, b = a.reshape(2, K, L), b.reshape(2, K, L)
    with quiet():
        warnings.simplefilter("ignore", DeprecationWarning)
        got = BlockCodeOps.bind(a, b)
    want = vsa.bind(grilly.from_numpy(a), grilly.from_numpy(b)).numpy()
    assert (got == want).all(), "the alias is the same kernel; nothing may differ"


def test_the_packed_bridge_functions_keep_v1s_uint32_signature():
    from grilly.backend import _bridge

    rng = np.random.default_rng(6)
    words = 16
    q = rng.integers(0, 2 ** 32, words, dtype=np.uint64).astype(np.uint32)
    cb = rng.integers(0, 2 ** 32, (24, words), dtype=np.uint64).astype(np.uint32)
    with quiet():
        warnings.simplefilter("ignore", DeprecationWarning)
        bound = _bridge.hdc_bind_packed(q, cb[0])
        assert bound.dtype == np.uint32 and (bound == (q ^ cb[0])).all()
        sims = _bridge.hdc_similarity_packed(q, cb, words * 32)
        assert sims.dtype == np.float32 and sims.shape == (24,)
        top = _bridge.hamming_topk(q, cb, words * 32, k=4)
    assert sorted(top) == ["distances", "indices", "similarities"]
    assert top["indices"].dtype == np.int32 and top["distances"].dtype == np.int32
    assert list(top["distances"]) == sorted(top["distances"]), "nearest first"


def test_an_alias_warns_and_names_the_new_path():
    from grilly.backend import _bridge

    a, _ = one_hot_flat(1, 7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _bridge.blockcode_bind(a, a, K, L)
    assert len(caught) == 1 and issubclass(caught[0].category, DeprecationWarning)
    assert "grilly.vsa.bind" in str(caught[0].message)


# ── replaced names fail loudly ───────────────────────────────────────────


@pytest.mark.parametrize("name, points_at", [
    ("linear", "grilly.nn.Linear"),
    ("gelu", "grilly.nn.functional.gelu"),
    ("layernorm", "grilly.nn.LayerNorm"),
    ("softmax", "grilly.nn.functional.softmax"),
    ("attention_scores", "scaled_dot_product_attention"),
])
def test_a_replaced_bridge_op_raises_and_says_what_to_use(name, points_at):
    from grilly.backend import _bridge

    with pytest.raises(NotImplementedError, match=points_at.replace(".", r"\.")):
        getattr(_bridge, name)


def test_hasattr_on_a_replaced_name_raises_rather_than_answering_false():
    """The whole point. ``AttributeError`` is what ``hasattr`` swallows, so
    a replaced name raises something else: a capability check must not be
    able to mistake "we removed this" for "the GPU is missing"."""
    from grilly.backend import _bridge

    with pytest.raises(NotImplementedError):
        hasattr(_bridge, "linear")
    with pytest.raises(NotImplementedError):
        from grilly.backend._bridge import gelu  # noqa: F401


def test_a_replaced_backend_class_raises():
    import grilly.backend as backend

    with pytest.raises(NotImplementedError, match="grilly.nn.Conv2d"):
        backend.VulkanConv
    with pytest.raises(NotImplementedError, match="grilly.compile"):
        backend.jit


def test_a_name_v1_never_had_is_an_ordinary_attribute_error():
    """Only v1's own names are loud; a typo stays a typo, and the import
    machinery's dunder lookups keep working."""
    from grilly.backend import _bridge

    with pytest.raises(AttributeError, match="no attribute"):
        _bridge.not_a_v1_function
    assert not hasattr(_bridge, "__wrapped__")


# ── stable hashing ───────────────────────────────────────────────────────


def test_the_hash_digests_match_v1s():
    """These seed hypervectors, so a changed digest silently repoints every
    symbol in a store written by v1."""
    theirs = v1_module("utils/stable_hash.py", "_v1_stable_hash")
    from grilly.vsa import hashing

    assert hashing.using_blake3() == theirs.using_blake3()
    for parts in [("cat",), ("cat", 3), ("a", "b", b"c"), (1.5,)]:
        assert hashing.stable_u32(*parts) == theirs.stable_u32(*parts), parts
        assert hashing.stable_u64(*parts) == theirs.stable_u64(*parts), parts
        assert hashing.stable_bytes(*parts, out_len=48) == theirs.stable_bytes(*parts, out_len=48)
    assert hashing.stable_u32("cat") != hashing.stable_u32("cat", domain="other")
    assert (hashing.bipolar_from_key("cat", 256) == theirs.bipolar_from_key("cat", 256)).all()


def test_the_old_hash_path_still_works_and_warns():
    from grilly.utils import stable_hash
    from grilly.vsa import hashing

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = stable_hash.stable_u32("cat")
    assert got == hashing.stable_u32("cat")
    assert any("grilly.vsa.hashing" in str(c.message) for c in caught)
