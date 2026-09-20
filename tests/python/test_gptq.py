"""GPTQ error compensation, against round-to-nearest.

What is asserted:

- with an **identity Hessian** GPTQ is round-to-nearest **exactly**, level
  for level. The inverse Cholesky is diagonal then, so no error is pushed
  anywhere, and any difference would mean the compensation is firing when
  it has nothing to compensate;
- with a **correlated** Hessian it is strictly better on the objective it
  optimizes, `||X (W - W')^T||`, which is the whole point: the weights move
  further from their originals in exchange for a closer output;
- the levels stay in range and the parameters keep the layout
  `Int4Tensor` reads, so a quantized weight is loadable.

Tolerance: the comparisons are ratios of measured errors, not thresholds
picked by hand. The one absolute number is that GPTQ must not be *worse*
than round-to-nearest on its own objective, which is a property of the
algorithm rather than of these weights.
"""

import numpy as np
import pytest

from grilly.infer.gptq import dequantize, gptq_quantize, rtn_quantize


def correlated_inputs(rng, n, k, strength=0.9):
    """Calibration rows whose columns are correlated, as real activations
    are — which is what gives GPTQ something to work with."""
    base = rng.standard_normal((n, k))
    mix = np.eye(k) + strength * np.triu(np.ones((k, k)), 1) * rng.standard_normal((k, k)) / np.sqrt(k)
    return (base @ mix).astype(np.float32)


def output_error(x, w, wq):
    return float(np.linalg.norm(x @ (w - wq).T) / np.linalg.norm(x @ w.T))


def test_an_identity_hessian_is_round_to_nearest_exactly():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 256)).astype(np.float32)
    for scale in (1.0, 7.5):  # any multiple of I: the damping is relative
        levels, scales, zeros = gptq_quantize(w, scale * np.eye(256), group_size=128)
        want_levels, want_scales, want_zeros = rtn_quantize(w, group_size=128)
        np.testing.assert_array_equal(levels, want_levels)
        np.testing.assert_allclose(scales, want_scales, rtol=0, atol=0)
        np.testing.assert_allclose(zeros, want_zeros, rtol=0, atol=0)


def test_it_beats_round_to_nearest_on_correlated_inputs():
    rng = np.random.default_rng(1)
    rows, k = 64, 512
    w = rng.standard_normal((rows, k)).astype(np.float32) * 0.1
    x = correlated_inputs(rng, 2048, k)
    hessian = 2.0 * (x.T @ x)

    plain = dequantize(*rtn_quantize(w, 128))
    compensated = dequantize(*gptq_quantize(w, hessian, group_size=128))
    plain_error = output_error(x, w, plain)
    gptq_error = output_error(x, w, compensated)
    print(f"output error: rtn {plain_error:.4e}, gptq {gptq_error:.4e}, {plain_error / gptq_error:.2f}x")
    assert gptq_error < plain_error

    # it trades weight error for output error, which is the mechanism
    weight_plain = float(np.linalg.norm(w - plain))
    weight_gptq = float(np.linalg.norm(w - compensated))
    print(f"weight error: rtn {weight_plain:.4e}, gptq {weight_gptq:.4e}")
    assert weight_gptq > weight_plain, "GPTQ should move the weights further, not nearer"


def test_it_is_never_worse_than_round_to_nearest_on_its_own_objective():
    rng = np.random.default_rng(2)
    for trial in range(4):
        rows, k = 16, 256
        w = rng.standard_normal((rows, k)).astype(np.float32)
        x = correlated_inputs(rng, 1024, k, strength=0.3 * trial)
        hessian = 2.0 * (x.T @ x)
        plain = output_error(x, w, dequantize(*rtn_quantize(w, 128)))
        gptq = output_error(x, w, dequantize(*gptq_quantize(w, hessian, group_size=128)))
        assert gptq <= plain * 1.02, f"trial {trial}: {gptq:.3e} against {plain:.3e}"


def test_levels_and_parameters_have_the_layout_int4tensor_reads():
    rng = np.random.default_rng(3)
    rows, k, group = 8, 384, 128
    w = rng.standard_normal((rows, k)).astype(np.float32)
    x = correlated_inputs(rng, 512, k)
    levels, scales, zeros = gptq_quantize(w, 2.0 * (x.T @ x), group_size=group)
    assert levels.shape == (rows, k) and levels.dtype == np.uint8
    assert scales.shape == zeros.shape == (rows, k // group)
    assert levels.min() >= 0 and levels.max() <= 15
    assert (scales > 0).all()
    # dequantize is q * scale + min, with min = zero - 8 * scale
    rebuilt = dequantize(levels, scales, zeros, group)
    assert rebuilt.shape == w.shape
    assert np.abs(rebuilt - w).max() < 8 * scales.max()


def test_refusals():
    w = np.zeros((4, 300), dtype=np.float32)
    with pytest.raises(ValueError, match="whole number of groups"):
        gptq_quantize(w, np.eye(300), group_size=128)
    with pytest.raises(ValueError, match="hessian must be"):
        gptq_quantize(np.zeros((4, 256), dtype=np.float32), np.eye(128), group_size=128)


def test_a_dead_input_feature_is_handled():
    """A column the calibration data never excited carries no information;
    GPTQ zeroes it rather than dividing by zero."""
    rng = np.random.default_rng(4)
    rows, k = 8, 256
    w = rng.standard_normal((rows, k)).astype(np.float32)
    x = correlated_inputs(rng, 512, k)
    x[:, 7] = 0.0
    levels, scales, zeros = gptq_quantize(w, 2.0 * (x.T @ x), group_size=128)
    rebuilt = dequantize(levels, scales, zeros)
    assert np.isfinite(rebuilt).all()
    assert abs(rebuilt[:, 7]).max() <= abs(rebuilt).max()


# ── the calibration pass over a real model ──────────────────────────────


def tiny_model():
    """A small Llama with in_features that are whole groups of 128."""
    import grilly
    from grilly.infer.models import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6,
                         qkv_bias=True, tie_word_embeddings=True)
    model = LlamaForCausalLM(config)
    rng = np.random.default_rng(0)
    state = {}
    for name, p in model.state_dict().items():
        values = rng.standard_normal(tuple(p.shape)).astype(np.float32)
        state[name] = grilly.from_numpy(values * 0.5 + 1.0 if "norm" in name else values * 0.1)
    model.load_state_dict(state, assign=True)
    model.tie_weights()
    return model


@pytest.mark.parametrize("act_order", [False, True])
def test_calibration_beats_round_to_nearest_on_a_model(act_order):
    """End to end: quantizing against each layer's real inputs must land
    closer to the float32 model than rounding each weight alone.

    Measured on the calibration data itself, which is what GPTQ optimizes —
    a property of the algorithm rather than of these random weights."""
    import grilly
    from grilly.infer.calibration import quantize_gptq_
    from grilly.infer.quantization import Int4Tensor, Int4WeightOnlyConfig, quantize_

    rng = np.random.default_rng(1)
    ids = grilly.tensor(rng.integers(0, 128, size=(4, 256)).tolist())
    with grilly.no_grad():
        want = tiny_model()(ids).logits.numpy().astype(np.float64)

    plain = tiny_model()
    quantize_(plain, Int4WeightOnlyConfig(group_size=128))
    compensated = tiny_model()
    quantize_gptq_(compensated, ids, group_size=128, act_order=act_order)
    assert isinstance(compensated.model.layers[0].mlp.down_proj.weight, Int4Tensor)
    assert not isinstance(compensated.lm_head.weight, Int4Tensor), "the head is left to the caller"

    with grilly.no_grad():
        rtn_error = float(np.linalg.norm(plain(ids).logits.numpy() - want) / np.linalg.norm(want))
        gptq_error = float(np.linalg.norm(compensated(ids).logits.numpy() - want) / np.linalg.norm(want))
    print(f"logit error (act_order={act_order}): rtn {rtn_error:.4e}, gptq {gptq_error:.4e}, "
          f"{rtn_error / gptq_error:.2f}x")
    assert gptq_error < rtn_error


def test_calibration_ids_windows_a_stream():
    import grilly
    from grilly.infer.calibration import calibration_ids

    class Tok:
        def encode(self, text):
            return [ord(c) % 97 for c in text]

    ids = calibration_ids(Tok(), ["hello world " * 200], samples=3, length=64)
    assert ids.shape == (3, 64) and ids.dtype is grilly.int64
    with pytest.raises(ValueError, match="shorter than one window"):
        calibration_ids(Tok(), ["short"], samples=1, length=64)
