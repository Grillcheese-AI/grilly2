"""generate()'s ``logits_processor`` and ``prefix_allowed_tokens_fn``, and
grilly.infer.logits_process.

References:
- off: grilly against grilly. ``None`` runs the loop without them, and a
  processor that returns its input changes no token (the penalty, then the
  processors, then the sampling kernel, against the one fused kernel): token
  sequences equal, greedy, seeded sampling, padded batches with rows dropped.
- an allowed list: every new token is in it, at every step.
- transformers: its own ``generate`` on the same Qwen2 weights with the same
  processors (a bias added, tokens suppressed, a prefix function), greedy:
  token sequences equal. The weights are the ones test_generation.py already
  checks greedy decoding with, drawn so the top two logits are not within
  rounding of each other.
"""

import grilly
import pytest
import torch
from test_generation import PROMPTS, UNEVEN, left_padded, pair, spy_on_drops
from transformers.generation.logits_process import SuppressTokensLogitsProcessor as HFSuppress

from grilly.infer.logits_process import (AllowedTokensLogitsProcessor, LogitsProcessorList,
                                         SuppressTokensLogitsProcessor)


def identity(input_ids, scores):
    return scores


def test_none_is_the_loop_without_them():
    _, model = pair(seed=1)
    plain = model.generate(grilly.tensor(PROMPTS), max_new_tokens=12).tolist()
    assert model.generate(grilly.tensor(PROMPTS), max_new_tokens=12, logits_processor=None,
                          prefix_allowed_tokens_fn=None).tolist() == plain


@pytest.mark.parametrize("penalty", [1.0, 1.3])
def test_an_identity_processor_changes_no_token(penalty):
    _, model = pair(tie=False, seed=1)
    kw = dict(max_new_tokens=16, repetition_penalty=penalty)
    assert (model.generate(grilly.tensor(PROMPTS), logits_processor=[identity], **kw).tolist()
            == model.generate(grilly.tensor(PROMPTS), **kw).tolist())
    sample = dict(do_sample=True, temperature=0.8, top_k=20, top_p=0.9, **kw)
    grilly.manual_seed(7)
    a = model.generate(grilly.tensor(PROMPTS), logits_processor=identity, **sample).tolist()
    grilly.manual_seed(7)
    assert a == model.generate(grilly.tensor(PROMPTS), **sample).tolist()


def test_an_identity_processor_changes_no_token_when_rows_drop(monkeypatch):
    _, model = pair(seed=6)
    ids, mask = left_padded(UNEVEN)
    free = model.generate(grilly.tensor(ids), attention_mask=grilly.tensor(mask), max_new_tokens=12).tolist()
    eos = free[1][len(ids[0]) + 2]
    kw = dict(attention_mask=grilly.tensor(mask), max_new_tokens=12, eos_token_id=eos, pad_token_id=0)
    plain = model.generate(grilly.tensor(ids), **kw).tolist()
    drops = spy_on_drops(monkeypatch, model)
    assert model.generate(grilly.tensor(ids), logits_processor=[identity], **kw).tolist() == plain
    assert drops


@pytest.mark.parametrize("sample", [False, True], ids=["greedy", "sampled"])
@pytest.mark.parametrize("padded", [False, True], ids=["even", "padded"])
@pytest.mark.parametrize("how", ["processor", "prefix_fn"])
def test_an_allowed_list_holds_at_every_step(sample, padded, how):
    _, model = pair(seed=3)
    allowed = {3, 17, 40, 41, 99, 127}
    prompts, kw = (left_padded(UNEVEN)[0], {"attention_mask": grilly.tensor(left_padded(UNEVEN)[1])}) if padded \
        else (PROMPTS, {})
    if sample:
        grilly.manual_seed(11)
        kw.update(do_sample=True, temperature=1.5, top_k=50, top_p=0.95)
    if how == "processor":
        kw["logits_processor"] = [AllowedTokensLogitsProcessor(allowed)]
    else:
        kw["prefix_allowed_tokens_fn"] = lambda batch_id, sent: sorted(allowed)
    out = model.generate(grilly.tensor(prompts), max_new_tokens=24, **kw).tolist()
    new = [t for row in out for t in row[len(prompts[0]):]]
    assert set(new) <= allowed and len(set(new)) > 1


def test_suppressed_tokens_never_come():
    _, model = pair(seed=1)
    free = model.generate(grilly.tensor(PROMPTS), max_new_tokens=16).tolist()
    banned = sorted({t for row in free for t in row[7:]})[:4]
    out = model.generate(grilly.tensor(PROMPTS), max_new_tokens=16,
                         logits_processor=LogitsProcessorList([SuppressTokensLogitsProcessor(banned)])).tolist()
    assert not {t for row in out for t in row[7:]} & set(banned)


def test_each_row_is_steered_by_its_own_batch_index_after_rows_drop(monkeypatch):
    """Row 0 may only end; rows 1 and 2 have sets of their own. Once row 0
    ends, the batch is cut to two rows, and the function must still hear
    batch ids 1 and 2, with each row's own tokens."""
    _, model = pair(seed=2)
    ids, mask = left_padded(UNEVEN)
    eos, width = 7, len(ids[0])
    sets = {0: [eos], 1: list(range(20, 30)), 2: list(range(40, 50))}
    seen = []

    def fn(batch_id, sent):
        row = sent.tolist()
        assert row[:width] == ids[batch_id]
        seen.append((batch_id, len(row)))
        return sets[batch_id]

    drops = spy_on_drops(monkeypatch, model)
    out = model.generate(grilly.tensor(ids), attention_mask=grilly.tensor(mask), max_new_tokens=10, eos_token_id=eos,
                         pad_token_id=0, prefix_allowed_tokens_fn=fn).tolist()
    assert drops and out[0][width:] == [eos] + [0] * 9
    for b in (1, 2):
        assert set(out[b][width:]) <= set(sets[b])
    assert {b for b, n in seen if n > width + 1} == {1, 2}


def test_bad_arguments_are_named():
    _, model = pair(seed=1)
    with pytest.raises(TypeError, match="callable"):
        model.generate(grilly.tensor(PROMPTS), max_new_tokens=2, logits_processor=[3])
    with pytest.raises(ValueError, match="empty list"):
        model.generate(grilly.tensor(PROMPTS), max_new_tokens=2, prefix_allowed_tokens_fn=lambda b, s: [])
    with pytest.raises(ValueError, match="at least one"):
        AllowedTokensLogitsProcessor([])


BIAS = [((i * 37) % 11 - 5) * 0.4 for i in range(128)]
SUPPRESS = [1, 2, 64, 65, 66]


def window(batch_id, sent):
    """A different allowed window at each length and row (works on torch and grilly rows alike)."""
    start = (sent.shape[0] * 7 + batch_id * 13) % 80
    return list(range(start, start + 48))


@pytest.mark.parametrize("penalty", [1.0, 1.3])
@pytest.mark.parametrize("padded", [False, True], ids=["even", "padded"])
def test_greedy_with_processors_matches_transformers(penalty, padded):
    ref, model = pair(tie=False, seed=1)
    prompts, mask = left_padded(UNEVEN) if padded else (PROMPTS, None)
    hf_bias = torch.tensor(BIAS)
    g_bias = grilly.tensor(BIAS)
    kw = dict(max_new_tokens=16, do_sample=False, repetition_penalty=penalty, prefix_allowed_tokens_fn=window)
    with torch.no_grad():
        want = ref.generate(torch.tensor(prompts), attention_mask=None if mask is None else torch.tensor(mask),
                            logits_processor=[lambda i, s: s + hf_bias, HFSuppress(SUPPRESS)], **kw).tolist()
        bare = ref.generate(torch.tensor(prompts), attention_mask=None if mask is None else torch.tensor(mask),
                            max_new_tokens=16, do_sample=False, repetition_penalty=penalty).tolist()
    got = model.generate(grilly.tensor(prompts), attention_mask=None if mask is None else grilly.tensor(mask),
                         logits_processor=[lambda i, s: s + g_bias, SuppressTokensLogitsProcessor(SUPPRESS)], **kw).tolist()
    assert got == want
    assert want != bare and len({t for row in want for t in row[len(prompts[0]):]}) > 3  # the processors steered it
