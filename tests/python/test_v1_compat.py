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


# ── the bipolar bridge, against v1's compiled extension ──────────────────


def v1_extension():
    """v1's compiled module, if there is a build beside this checkout.

    These functions are host-side, so importing it costs nothing and opens
    no device. They are the reference because v1's *Python* docstrings are
    wrong about them — ``vsa_bind`` says "circular convolution" and is an
    element-wise product.
    """
    env = os.environ.get("GRILLY_V1_DIR")
    root = pathlib.Path(env) if env else pathlib.Path(__file__).resolve().parents[2].parent / "grilly"
    matches = sorted(root.glob("grilly_core*.pyd")) + sorted(root.glob("grilly_core*.so"))
    if not matches:
        pytest.skip("no built v1 extension beside this checkout (set GRILLY_V1_DIR)")
    # an extension has to be imported under the name its PyInit_ function
    # carries, which is the file stem up to the first dot
    name = matches[0].name.split(".")[0]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, matches[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as e:                      # a build for another Python
        del sys.modules[name]
        pytest.skip(f"v1's extension will not load here: {e}")
    return module


def bipolar_np(n, seed):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=n) * 2 - 1).astype(np.int8)


def test_the_bipolar_bridge_matches_v1s_compiled_semantics():
    core = v1_extension()
    from grilly.backend import _bridge

    a, b = bipolar_np(256, 1), bipolar_np(256, 2)
    with quiet():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert (_bridge.vsa_bind(a, b) == core.vsa_bind(a, b)).all()
        assert (_bridge.vsa_bundle(a, b) == core.vsa_bundle([a, b])).all()
        assert (_bridge.vsa_bitpack(a) == core.vsa_bitpack(a)).all()
        assert (_bridge.blake3_role("cat", 256) == core.blake3_role("cat", 256)).all()


def test_the_bipolar_bridge_returns_v1s_dtypes():
    """v1's C++ gave int8 back from bind and bundle, and uint32 from
    bitpack. A caller that indexes or packs the result depends on it."""
    from grilly.backend import _bridge

    a, b = bipolar_np(64, 3), bipolar_np(64, 4)
    with quiet():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert _bridge.vsa_bind(a, b).dtype == np.int8
        assert _bridge.vsa_unbind(_bridge.vsa_bind(a, b), b).dtype == np.int8
        assert (_bridge.vsa_unbind(_bridge.vsa_bind(a, b), b) == a).all()
        assert _bridge.vsa_bitpack(a).dtype == np.uint32
        assert _bridge.blake3_role("cat", 64).dtype == np.int8


def test_the_v1_bridge_wrappers_returned_none_for_these():
    """Not a grilly2 test — a record of why replacing them cannot break a
    caller. v1's wrappers passed a device handle the compiled functions do
    not take, and ``vsa_unbind`` was never compiled at all, so every call
    fell into the except and returned None."""
    core = v1_extension()
    a = bipolar_np(32, 5)
    assert not hasattr(core, "vsa_unbind")
    for call in (lambda: core.vsa_bind(None, a, a),
                 lambda: core.vsa_bundle(None, a, a),
                 lambda: core.vsa_bitpack(None, a)):
        with pytest.raises(TypeError):
            call()


def test_binary_and_holographic_ops_agree_with_the_new_api():
    from grilly.experimental.vsa import BinaryOps, HolographicOps
    from grilly.vsa import bipolar, holographic

    a, b = bipolar_np(256, 6).astype(np.float32), bipolar_np(256, 7).astype(np.float32)
    with quiet():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert (BinaryOps.bind(a, b) == bipolar.bind(grilly.from_numpy(a), grilly.from_numpy(b)).numpy()).all()
        assert BinaryOps.similarity(a, a) == 1.0
        h = HolographicOps.random_vector(256, seed=8)
        assert abs(np.linalg.norm(h) - 1.0) < 1e-5
        assert (HolographicOps.convolve(h, h)
                == holographic.bind(grilly.from_numpy(h), grilly.from_numpy(h)).numpy()).all()


def test_the_resonator_alias_keeps_v1s_return_order():
    """v1 returned (estimates, indices, iterations); the new API returns the
    indices first. The alias must not silently swap them."""
    from grilly.experimental.vsa import ResonatorNetwork

    rng = np.random.default_rng(9)
    cbs = {n: np.sign(rng.standard_normal((4, 256))).astype(np.float32) for n in ("a", "b")}
    composite = (cbs["a"][1] * cbs["b"][2]).astype(np.float32)
    with quiet():
        warnings.simplefilter("ignore", DeprecationWarning)
        estimates, indices, iterations = ResonatorNetwork(cbs).factorize(composite)
        recovered = ResonatorNetwork(cbs).factorize_partial(composite, {"a": cbs["a"][1]})
    assert isinstance(estimates, dict) and isinstance(indices, dict)
    assert set(estimates) == {"a", "b"} and all(isinstance(v, int) for v in indices.values())
    assert isinstance(iterations, int)
    assert (recovered == cbs["b"][2]).all(), "unbinding a known factor is exact"


# ── cells, the spike bridge, CVL ─────────────────────────────────────────


def test_the_cell_aliases_keep_v1s_numpy_signatures():
    from grilly.backend import _bridge
    from grilly.functional import cells as fcells
    from grilly.snn import cells as scells

    rng = np.random.default_rng(20)
    position = rng.standard_normal(2).astype(np.float32)
    centres = rng.standard_normal((12, 2)).astype(np.float32)
    with quiet():
        warnings.simplefilter("ignore", DeprecationWarning)
        rates = fcells.place_cell(position, centres)
        assert isinstance(rates, np.ndarray) and rates.shape == (12,)
        assert (rates == scells.place_cell(grilly.from_numpy(position),
                                           grilly.from_numpy(centres)).numpy()).all()
        # v1 returned a (rates, membrane) pair, and the bridge spelled the
        # width `time_constant` where the layer above said `temporal_width`
        pair = fcells.time_cell(1.0, np.arange(6, dtype=np.float32))
        assert isinstance(pair, tuple) and len(pair) == 2
        assert (_bridge.time_cell(1.0, np.arange(6, dtype=np.float32), time_constant=1.0)[0]
                == pair[0]).all()
        assert fcells.theta_gamma_encoding(0.1, n_theta=4, n_gamma=5).shape == (20,)


def test_the_spike_bridge_alias_round_trips():
    from grilly.functional import bridge

    x = np.random.default_rng(21).random((3, 8)).astype(np.float32)
    with quiet():
        warnings.simplefilter("ignore", DeprecationWarning)
        spikes = bridge.continuous_to_spikes(x, num_timesteps=64)
        assert spikes.shape == (3, 64, 8) and spikes.dtype == np.float32
        back = bridge.spikes_to_continuous(spikes, time_window=64)
    assert back.shape == x.shape
    assert np.abs(back - x).max() < 0.3, "a 64-step rate code is coarse but not wrong"


def test_an_encoding_v1_never_implemented_raises_instead_of_rate_coding():
    """v1 took encoding_type 1 and 2 and rate-coded anyway. Silently giving
    the wrong code is the failure mode this project exists to remove."""
    from grilly.functional import bridge

    x = np.zeros((1, 4), dtype=np.float32)
    with quiet():
        warnings.simplefilter("ignore", DeprecationWarning)
        for kind in (1, 2):
            with pytest.raises(NotImplementedError, match="never"):
                bridge.continuous_to_spikes(x, encoding_type=kind)
        with pytest.raises(NotImplementedError, match="never"):
            bridge.spikes_to_continuous(np.zeros((1, 2, 4), np.float32), encoding_type=1)
        with pytest.raises(NotImplementedError, match="Simplified"):
            bridge.bridge_temporal_weights(np.ones((2, 2), np.float32))


def test_the_cvl_alias_matches_v1s_maths():
    """v1's CVL was numpy end to end, so this is a like-for-like check of
    the formulas, not of a kernel."""
    from grilly.backend.cvl import CVLValue, RandomFourierFeatures

    with quiet():
        warnings.simplefilter("ignore", DeprecationWarning)
        features = RandomFourierFeatures(in_dim=8, n_features=256, seed=5)
        x = np.random.default_rng(22).standard_normal(8).astype(np.float32)
        got = features(x)
        want = np.sqrt(2 * np.e / 256) * np.cos(x @ features.W + features.b)
        assert got.shape == (256,)
        assert np.abs(got - want).max() < 1e-5

        value = CVLValue(gamma=0.9, beta=0.5)
        assert value.xi is None and value.value(got) == 0.0
        value.observe(got, 2.0)
        xi = 0.5 * np.zeros_like(got) + 0.5 * (got * 2.0)
        assert np.abs(value.xi - xi).max() < 1e-5
        assert value.value(got) == pytest.approx(float(got @ xi) / 0.1, rel=1e-4)


def test_the_v1_functional_namespace_is_not_torchs_F():
    """``grilly.functional`` is v1's; ``grilly.nn.functional`` is torch's.
    Asking v1's for a torch op says which one you want."""
    import grilly.functional as v1f

    with pytest.raises(NotImplementedError, match=r"grilly\.nn\.functional\.gelu"):
        v1f.gelu
    assert callable(v1f.place_cell)


def test_the_learning_aliases_keep_v1s_numpy_signatures():
    from grilly import learning as new
    from grilly.backend import _bridge
    from grilly.functional import learning as old

    rng = np.random.default_rng(30)
    grads = rng.standard_normal(12).astype(np.float32)
    fisher = np.abs(rng.standard_normal(12)).astype(np.float32)
    with quiet():
        warnings.simplefilter("ignore", DeprecationWarning)
        updated = old.fisher_info(grads, fisher)
        assert isinstance(updated, np.ndarray) and updated.dtype == np.float32
        assert (updated == new.fisher_info(grilly.from_numpy(grads),
                                           grilly.from_numpy(fisher)).numpy()).all()
        # v1 called it fisher_info_update at the bridge and fisher_info above it
        assert (_bridge.fisher_info_update(grads, fisher) == updated).all()

        assert isinstance(old.ewc_penalty(grads, grads * 0.5, fisher), float)
        assert isinstance(old.natural_gradient(grads, fisher), np.ndarray)

        # nlms_update returned a (weights, bias) pair of numpy and float
        weights, bias = old.nlms_update(grads, 1.0, np.zeros(12, np.float32), 0.0)
        assert isinstance(weights, np.ndarray) and isinstance(bias, float)

        white, mean, std = old.whitening_transform(rng.standard_normal((20, 4)).astype(np.float32))
        assert all(isinstance(a, np.ndarray) for a in (white, mean, std))
        assert white.shape == (20, 4) and mean.shape == (4,)
        assert (old.whitening_apply(white * 0 + 1.0, mean, std)
                == new.whitening_apply(grilly.full((20, 4), 1.0), grilly.from_numpy(mean),
                                       grilly.from_numpy(std)).numpy()).all()


def test_the_capsule_layers_point_at_linear_because_that_is_what_they_were():
    """Reading this group is what decided its migration: ``CapsuleProject``
    and ``SemanticEncoder`` have the same __init__ once their defaults are
    normalised and both return ``x @ weight.T + bias``. Porting them would
    have been adding a second name for nn.Linear."""
    import grilly.nn.capsule as capsule

    for name in ("CapsuleProject", "SemanticEncoder"):
        with pytest.raises(NotImplementedError, match=r"grilly\.nn\.Linear"):
            getattr(capsule, name)
    with pytest.raises(NotImplementedError, match=r"grilly\.snn\.DentateGyrus"):
        capsule.DentateGyrus


def test_a_capsule_checkpoint_loads_straight_into_a_linear():
    """The claim the error message makes, checked: same parameter names,
    same shapes, same arithmetic."""
    import grilly.nn as nn

    layer = nn.Linear(6, 3)
    v1_state = {"weight": grilly.randn(3, 6), "bias": grilly.randn(3)}
    layer.load_state_dict(v1_state)
    x = grilly.randn(2, 6)
    want = (x @ v1_state["weight"].t() + v1_state["bias"]).numpy()
    assert np.abs(layer(x).detach().numpy() - want).max() < 1e-6


def test_the_cosine_helpers_point_at_the_torch_name():
    import grilly.nn.functional as F
    from grilly.experimental.cognitive import capsule

    for name in ("cosine_similarity", "batch_cosine_similarity"):
        with pytest.raises(NotImplementedError, match="cosine_similarity"):
            getattr(capsule, name)

    # and the replacement the message gives actually produces v1's answer
    query = grilly.from_numpy(np.random.default_rng(40).standard_normal(8).astype(np.float32))
    matrix = grilly.from_numpy(np.random.default_rng(41).standard_normal((5, 8)).astype(np.float32))
    got = F.cosine_similarity(matrix, query.unsqueeze(0), dim=1).numpy()
    q, m = query.numpy().astype(np.float64), matrix.numpy().astype(np.float64)
    want = (m @ q) / (np.linalg.norm(m, axis=1) * np.linalg.norm(q))
    assert np.abs(got - want).max() < 1e-6


def test_the_moqe_names_point_at_int8_weight_only_with_the_reason():
    """Not ported, and the message carries the measurement: at decode the
    int8 linear is already at 87% of this card's memory bandwidth, so a
    W8A8 kernel reads the same weight bytes and saves nothing."""
    from grilly.backend import _bridge

    for name in ("moqe_dynamic_quantize", "moqe_fused_gemv", "moqe_route_and_gemv",
                 "ternary_gemm", "pack_ternary"):
        with pytest.raises(NotImplementedError, match="Int8WeightOnlyConfig"):
            getattr(_bridge, name)
    with pytest.raises(NotImplementedError, match="memory-bound"):
        _bridge.moqe_fused_gemv


def test_the_snn_aliases_keep_v1s_names_and_warn():
    from grilly.functional import snn as v1_snn
    from grilly.nn.snn_neurons import LIFNode as OldLIF
    from grilly.nn.snn_surrogate import ATan as OldATan
    from grilly.snn import LIFNode

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        node = OldLIF(tau=2.0, surrogate_function=OldATan(alpha=3.0))
    assert any("grilly.snn.LIFNode" in str(c.message) for c in caught)
    assert isinstance(node, LIFNode), "the alias is the ported class, not a copy"

    with quiet():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert float(node(grilly.tensor([1.5]))[0].item()) == 1.0
        v1_snn.reset_net(node)
    assert node.v is None

    # v1's element-wise neuron steps became modules that hold their own state
    for name in ("if_step", "lif_step"):
        with pytest.raises(NotImplementedError, match="membrane"):
            getattr(v1_snn, name)
