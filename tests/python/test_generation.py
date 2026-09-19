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
    with pytest.raises(NotImplementedError, match="padding"):
        model.generate(ids, attention_mask=grilly.tensor([[0] + [1] * 6, [1] * 7]))
    with pytest.raises(ValueError, match="not used by the model"):
        model.generate(ids, output_attentions=True)
    with pytest.raises(ValueError, match="int64"):
        model.generate(grilly.ones(2, 7))
    assert model.generate(ids, attention_mask=grilly.ones(2, 7, dtype=grilly.int64), max_new_tokens=2).shape[1] == 9
