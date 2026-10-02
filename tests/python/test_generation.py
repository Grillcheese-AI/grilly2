"""grilly.infer.models.LlamaForCausalLM and generate(), against transformers'
own Qwen2ForCausalLM in torch (float32, eager attention) with the same
weights.

Tolerances:
- logits: max |grilly - transformers| <= 1e-5 * max |logit|. That is case
  (b) in CLAUDE.md §9: two layers of different float32 implementations
  (matmul order, softmax, RMSNorm's mean). Measured ~2e-7.
- token sequences: equal. Greedy decoding picks the argmax, and the weights
  are drawn large enough that the models' top two logits are not within
  rounding of each other.
"""

import contextlib

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import grilly
from grilly.infer.models import LlamaConfig, LlamaForCausalLM
from grilly.infer.quantization import Int8WeightOnlyConfig, quantize_


def pair(tie=True, seed=0, init=0.2):
    """transformers' Qwen2 and grilly's model with the same random weights."""
    torch.manual_seed(seed)
    hc = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, max_position_embeddings=256, rope_theta=1e6, rms_norm_eps=1e-6,
                     tie_word_embeddings=tie, initializer_range=init, attn_implementation="eager")
    ref = Qwen2ForCausalLM(hc).eval()
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, rope_theta=1e6, rms_norm_eps=1e-6, qkv_bias=True, tie_word_embeddings=tie)
    model = LlamaForCausalLM(cfg)
    sd = {k: grilly.from_numpy(v.detach().numpy().copy()) for k, v in ref.state_dict().items()}
    result = model.load_state_dict(sd, assign=True)
    assert (result.missing_keys, result.unexpected_keys) == ([], [])
    model.tie_weights()
    return ref, model


PROMPTS = [[5, 17, 99, 3, 42, 7, 64], [11, 2, 88, 120, 9, 31, 4]]


@pytest.mark.parametrize("tie", [True, False], ids=["tied", "untied"])
def test_logits_match_transformers_full_and_cached(tie):
    ref, model = pair(tie)
    with torch.no_grad(), grilly.no_grad():
        want = ref(torch.tensor(PROMPTS)).logits.numpy()
        got = model(grilly.tensor(PROMPTS)).logits.numpy()
        scale = np.abs(want).max()
        assert np.abs(got - want).max() <= 1e-5 * scale
        cache = model.make_cache(2, 16)
        steps = [model(grilly.tensor([p[:3] for p in PROMPTS]), past_key_values=cache).logits.numpy()]
        for t in range(3, len(PROMPTS[0])):
            steps.append(model(grilly.tensor([[p[t]] for p in PROMPTS]), past_key_values=cache).logits.numpy())
        assert np.abs(np.concatenate(steps, 1) - want).max() <= 1e-5 * scale
        last = model(grilly.tensor(PROMPTS), logits_to_keep=1).logits
        assert tuple(last.shape) == (2, 1, 128)


@pytest.mark.parametrize("penalty", [1.0, 1.3])
def test_greedy_generate_matches_transformers(penalty):
    ref, model = pair(tie=False, seed=1)
    with torch.no_grad():
        want = ref.generate(torch.tensor(PROMPTS), max_new_tokens=16, do_sample=False, repetition_penalty=penalty).tolist()
    got = model.generate(grilly.tensor(PROMPTS), max_new_tokens=16, do_sample=False, repetition_penalty=penalty).tolist()
    assert got == want
    assert len(set(got[0][7:])) > 3  # the weights give varied tokens, so this compares something


def test_eos_stops_a_row_and_pads_as_transformers():
    ref, model = pair(seed=2)
    free = model.generate(grilly.tensor(PROMPTS), max_new_tokens=10).tolist()
    eos = free[0][7 + 3]  # row 0's fourth new token
    with torch.no_grad():
        want = ref.generate(torch.tensor(PROMPTS), max_new_tokens=10, do_sample=False, eos_token_id=eos, pad_token_id=0).tolist()
    got = model.generate(grilly.tensor(PROMPTS), max_new_tokens=10, eos_token_id=eos, pad_token_id=0).tolist()
    assert got == want
    one = model.generate(grilly.tensor(PROMPTS[:1]), max_new_tokens=10, eos_token_id=eos).tolist()
    assert one[0][-1] == eos and len(one[0]) == 7 + 4  # stops when every row has finished


def test_sampling_is_seeded_and_max_length_is_hfs_default():
    _, model = pair(seed=3)
    kw = dict(do_sample=True, temperature=0.8, top_k=20, top_p=0.9, max_new_tokens=12)
    grilly.manual_seed(7)
    a = model.generate(grilly.tensor(PROMPTS), **kw).tolist()
    grilly.manual_seed(7)
    b = model.generate(grilly.tensor(PROMPTS), **kw).tolist()
    assert a == b and all(0 <= t < 128 for row in a for t in row)
    assert tuple(model.generate(grilly.tensor(PROMPTS)).shape) == (2, 20)  # GenerationConfig().max_length


def test_streamer_gets_the_prompt_then_each_token():
    class Recorder:
        def __init__(self):
            self.calls = []

        def put(self, value):
            self.calls.append(value.tolist())

        def end(self):
            self.calls.append("end")

    _, model = pair(seed=4)
    rec = Recorder()
    out = model.generate(grilly.tensor(PROMPTS[:1]), max_new_tokens=5, streamer=rec).tolist()
    assert rec.calls[0] == PROMPTS[:1] and rec.calls[-1] == "end"
    assert [c[0] for c in rec.calls[1:-1]] == out[0][7:]


def test_a_quantized_model_generates():
    _, model = pair(tie=False, seed=5)
    with grilly.no_grad():
        before = model(grilly.tensor(PROMPTS)).logits.numpy()
        quantize_(model, Int8WeightOnlyConfig())
        after = model(grilly.tensor(PROMPTS)).logits.numpy()
    assert np.abs(after - before).max() <= 0.05 * np.abs(before).max()  # int8's error, not a bug's
    assert tuple(model.generate(grilly.tensor(PROMPTS), max_new_tokens=8).shape) == (2, 15)


def test_what_generate_refuses():
    _, model = pair(seed=6)
    ids = grilly.tensor(PROMPTS)
    with pytest.raises(NotImplementedError, match="beam"):
        model.generate(ids, num_beams=2)
    with pytest.raises(ValueError, match="not used by the model"):
        model.generate(ids, output_attentions=True)
    with pytest.raises(ValueError, match="int64"):
        model.generate(grilly.ones(2, 7))
    assert model.generate(ids, attention_mask=grilly.ones(2, 7, dtype=grilly.int64), max_new_tokens=2).shape[1] == 9


def qwen3_pair(tie=True, seed=0, init=0.2):
    """transformers' Qwen3 and grilly's model with the same random weights.
    Qwen3 adds an RMSNorm on each head of q and k before the rotary
    embedding, and carries no q/k/v bias."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    hc = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                     rope_theta=1e6, rms_norm_eps=1e-6, tie_word_embeddings=tie, initializer_range=init,
                     attn_implementation="eager")
    ref = Qwen3ForCausalLM(hc).eval()
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, rope_theta=1e6,
                      rms_norm_eps=1e-6, qkv_bias=False, qk_norm=True, tie_word_embeddings=tie)
    model = LlamaForCausalLM(cfg)
    state = {k: grilly.from_numpy(v.detach().numpy().copy()) for k, v in ref.state_dict().items()}
    result = model.load_state_dict(state, assign=True)
    assert (result.missing_keys, result.unexpected_keys) == ([], []), result
    model.tie_weights()
    return ref, model


def test_qwen3_logits_match_transformers():
    """The q/k head norms are the only difference from Qwen2, and they are
    easy to place wrong: after the rotation instead of before, or over the
    projection instead of over one head. Either mistake changes the logits,
    so this compares against transformers' own Qwen3 at the same tolerance
    as the Qwen2 case."""
    ref, model = qwen3_pair()
    ids = torch.tensor(PROMPTS)
    with torch.no_grad():
        want = ref(ids).logits.numpy()
    with grilly.no_grad():
        got = model(grilly.tensor(PROMPTS)).logits.numpy()
    worst = float(np.abs(got - want).max() / np.abs(want).max())
    print(f"qwen3 logits: worst {worst:.2e} of the 1e-5 bound")
    assert worst <= 1e-5


def test_qwen3_generates_what_transformers_generates():
    ref, model = qwen3_pair(tie=False, seed=3)
    ids = torch.tensor(PROMPTS)
    with torch.no_grad():
        want = ref.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=12,
                            do_sample=False, top_k=None, top_p=None, temperature=None).tolist()
    got = model.generate(grilly.tensor(PROMPTS), max_new_tokens=12, do_sample=False).tolist()
    assert got == want


def test_qwen3_fused_and_composed_agree():
    """The fused decode path normalizes each head with the rms_norm kernel
    on the projection; the composed path uses F.rms_norm over the last
    dimension after reshaping. They must agree."""
    _, model = qwen3_pair(seed=5)
    ids = grilly.tensor(PROMPTS)
    composed = model(ids).logits.detach().numpy()
    with grilly.no_grad():
        fused = model(ids).logits.numpy()
    worst = float(np.abs(fused - composed).max() / np.abs(composed).max())
    print(f"qwen3 fused vs composed: {worst:.2e}")
    assert worst <= 1e-5
    assert (fused.argmax(-1) == composed.argmax(-1)).all()


# ── left-padded batches ──────────────────────────────────────────────────
#
# Reference: transformers' own generate() on the same left-padded batch, and
# each prompt generated alone. A padded row holds its tokens p positions
# later than it would alone; rotary scores depend only on the distance
# between positions, so the shift moves nothing but rounding (cos and sin
# at other angles), and the logits bound is the one above.

UNEVEN = [[5, 17, 99, 3, 42, 7, 64], [11, 2, 88], [9, 31, 4, 120, 77]]


def left_padded(prompts, pad=0):
    width = max(len(p) for p in prompts)
    ids = [[pad] * (width - len(p)) + list(p) for p in prompts]
    mask = [[0] * (width - len(p)) + [1] * len(p) for p in prompts]
    return ids, mask


def alone(model, prompts, **kw):
    """Each prompt's new tokens, generated on its own and unpadded."""
    return [model.generate(grilly.tensor([p]), **kw).tolist()[0][len(p):] for p in prompts]


def new_tokens(rows, width):
    return [row[width:] for row in rows]


def spy_on_drops(monkeypatch, model):
    """Record every time generate() drops rows, so a test can check that
    what it compared really ran with rows dropped."""
    calls = []
    original = model._select_rows

    def spy(cache, index, undo):
        calls.append(index.tolist())
        original(cache, index, undo)

    monkeypatch.setattr(model, "_select_rows", spy)
    return calls


@pytest.mark.parametrize("grad", [False, True], ids=["fused", "composed"])
def test_a_left_padded_prompt_gives_each_row_its_own_logits(grad):
    """Every real position's logits equal that row's own, run unpadded. The
    padding's are not compared: they are attention over nothing."""
    _, model = pair(seed=7)
    ids, mask = left_padded(UNEVEN)
    width = len(ids[0])
    worst = 0.0
    with contextlib.nullcontext() if grad else grilly.no_grad():
        cache = model.make_cache(len(ids), width, keep=mask)
        got = model(grilly.tensor(ids), past_key_values=cache).logits.detach().numpy()
        assert np.isfinite(got).all(), "padding produced a NaN or inf"
        for r, p in enumerate(UNEVEN):
            want = model(grilly.tensor([p])).logits.detach().numpy()[0]
            worst = max(worst, float(np.abs(got[r, width - len(p):] - want).max() / np.abs(want).max()))
    print(f"padded vs alone ({'composed' if grad else 'fused'}): worst {worst:.2e} of max |logit|")
    assert worst <= 1e-5


@pytest.mark.parametrize("penalty", [1.0, 1.3])
def test_a_left_padded_batch_generates_what_transformers_generates(penalty):
    """Under a repetition penalty the pads are part of each row's history,
    in transformers as here."""
    ref, model = pair(tie=False, seed=1)
    ids, mask = left_padded(UNEVEN)
    with torch.no_grad():
        want = ref.generate(torch.tensor(ids), attention_mask=torch.tensor(mask), max_new_tokens=16,
                            do_sample=False, repetition_penalty=penalty, pad_token_id=0).tolist()
    got = model.generate(grilly.tensor(ids), attention_mask=grilly.tensor(mask), max_new_tokens=16,
                         do_sample=False, repetition_penalty=penalty).tolist()
    assert got == want
    assert all(len(set(row[7:])) > 3 for row in got)  # varied tokens: this compares something


def test_each_row_of_a_padded_batch_generates_what_it_generates_alone():
    _, model = pair(tie=False, seed=8)
    ids, mask = left_padded(UNEVEN)
    got = model.generate(grilly.tensor(ids), attention_mask=grilly.tensor(mask), max_new_tokens=16).tolist()
    assert new_tokens(got, 7) == alone(model, UNEVEN, max_new_tokens=16)
    assert [row[:7] for row in got] == ids  # the prompts come back as given, padding and all


def test_dropping_finished_rows_changes_no_sequence(monkeypatch):
    """Rows that reach EOS leave the batch; the rest must not move, and the
    finished ones come back padded after their EOS, as transformers pads."""
    ref, model = pair(tie=False, seed=2)
    ids, mask = left_padded(UNEVEN)
    free = model.generate(grilly.tensor(ids), attention_mask=grilly.tensor(mask), max_new_tokens=12).tolist()
    eos = free[1][7 + 2]  # row 1's third new token
    drops = spy_on_drops(monkeypatch, model)
    got = model.generate(grilly.tensor(ids), attention_mask=grilly.tensor(mask), max_new_tokens=12,
                         eos_token_id=eos, pad_token_id=0).tolist()
    with torch.no_grad():
        want = ref.generate(torch.tensor(ids), attention_mask=torch.tensor(mask), max_new_tokens=12,
                            do_sample=False, eos_token_id=eos, pad_token_id=0).tolist()
    assert drops, "no row finished early, so this compared nothing"
    assert got == want
    for row, solo in zip(new_tokens(got, 7), alone(model, UNEVEN, max_new_tokens=12, eos_token_id=eos)):
        assert row == solo + [0] * (len(row) - len(solo))


def test_compact_finished_drops_rows_of_an_equal_length_batch(monkeypatch):
    _, model = pair(seed=2)
    free = model.generate(grilly.tensor(PROMPTS), max_new_tokens=10).tolist()
    eos = free[0][7 + 3]
    plain = model.generate(grilly.tensor(PROMPTS), max_new_tokens=10, eos_token_id=eos, pad_token_id=0).tolist()
    drops = spy_on_drops(monkeypatch, model)
    compact = model.generate(grilly.tensor(PROMPTS), max_new_tokens=10, eos_token_id=eos, pad_token_id=0,
                             compact_finished=True).tolist()
    assert drops and compact == plain


def test_a_quantized_model_pads_through_its_fused_projection():
    """int8 weights fuse q, k and v into one projection, which the padded
    path reads by column like the unpadded one."""
    _, model = pair(tie=False, seed=5)
    quantize_(model, Int8WeightOnlyConfig())
    ids, mask = left_padded(UNEVEN)
    got = model.generate(grilly.tensor(ids), attention_mask=grilly.tensor(mask), max_new_tokens=10).tolist()
    assert new_tokens(got, 7) == alone(model, UNEVEN, max_new_tokens=10)


def test_qwen3_left_padded_generates_what_transformers_generates():
    ref, model = qwen3_pair(tie=False, seed=3)
    ids, mask = left_padded(UNEVEN)
    with torch.no_grad():
        want = ref.generate(torch.tensor(ids), attention_mask=torch.tensor(mask), max_new_tokens=12,
                            do_sample=False, top_k=None, top_p=None, temperature=None, pad_token_id=0).tolist()
    got = model.generate(grilly.tensor(ids), attention_mask=grilly.tensor(mask), max_new_tokens=12).tolist()
    assert got == want


def test_what_padding_generate_refuses():
    _, model = pair(seed=6)
    ids = grilly.tensor(PROMPTS)
    with pytest.raises(ValueError, match="left-padded"):  # right padding
        model.generate(ids, attention_mask=grilly.tensor([[1] * 6 + [0], [1] * 7]))
    with pytest.raises(ValueError, match="left-padded"):  # a row of nothing
        model.generate(ids, attention_mask=grilly.tensor([[0] * 7, [1] * 7]))
    with pytest.raises(ValueError, match="attention_mask is"):
        model.generate(ids, attention_mask=grilly.tensor([[1] * 6, [1] * 6]))

    class Unpadded(LlamaForCausalLM):
        supports_padding = False

    with pytest.raises(NotImplementedError, match="padding"):
        Unpadded(model.config).generate(ids, attention_mask=grilly.tensor([[0] + [1] * 6, [1] * 7]))


def test_a_padded_population_generates_what_each_member_generates_alone(monkeypatch):
    """ES's shape: members x prompts of different lengths in one left-padded
    batch, rows dropped as they finish. Each LoRA drops the same rows of its
    per-row noise, so every surviving row keeps its own member's
    perturbation — and the noise is whole again afterwards, for the next
    generation in the same population block."""
    from grilly.infer.lora import apply_lora_, lora_parameters, population
    from grilly.learning import EvolutionStrategy

    _, model = pair(tie=False, seed=9)
    targets = [f"model.layers.{i}.{m}" for i in range(2)
               for m in ("self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj")]
    apply_lora_(model, rank=2, alpha=4.0, targets=targets)
    rng = np.random.default_rng(3)
    with grilly.no_grad():  # a trained adapter, not peft's zero B
        for f in lora_parameters(model):
            f.copy_(grilly.from_numpy((0.1 * rng.standard_normal(tuple(f.shape))).astype(np.float32)))
    es = EvolutionStrategy(lora_parameters(model), sigma=0.05, lr=0.0, population=4, seed=11)
    prompts = UNEVEN[:2]
    ids, mask = left_padded(prompts)
    k = len(prompts)
    wide_ids = [ids[j] for _ in range(es.population) for j in range(k)]  # member m owns rows m*k .. m*k+k-1
    wide_mask = [mask[j] for _ in range(es.population) for j in range(k)]

    with population(model, es, repeat=k):
        free = model.generate(grilly.tensor(wide_ids), attention_mask=grilly.tensor(wide_mask),
                              max_new_tokens=10).tolist()
    eos = free[1][7 + 2]
    drops = spy_on_drops(monkeypatch, model)
    with population(model, es, repeat=k) as layers:
        got = model.generate(grilly.tensor(wide_ids), attention_mask=grilly.tensor(wide_mask),
                             max_new_tokens=10, eos_token_id=eos, pad_token_id=0).tolist()
        assert {layer._population["u_a"].shape[0] for layer in layers} == {es.population * k}
    assert drops, "no row finished early, so this compared nothing"
    for m in range(es.population):
        with es.member(m):
            solos = alone(model, prompts, max_new_tokens=10, eos_token_id=eos)
        for j, solo in enumerate(solos):
            row = got[m * k + j][7:]
            assert row == solo + [0] * (len(row) - len(solo)), f"member {m}, prompt {j}"
