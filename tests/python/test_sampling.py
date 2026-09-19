"""grilly.infer.sampling and torch.multinomial, against Hugging Face's own
logits processors (transformers, run in torch on the CPU).

Tolerances:
- processed scores (repetition penalty, temperature): bit-exact. Both are one
  multiplication or one correctly rounded division, as in torch.
- kept tokens after top-k: identical. top-k compares against a value both
  compute bit-exactly.
- kept tokens after top-p: identical, with two exceptions.
  - A token whose cumulative probability (HF's ascending cumsum) is within
    1e-6 of 1 - top_p may be decided differently. HF sums sorted float32
    probabilities in order, the kernel sums masses in a tree. The test
    counts these.
  - Among tokens tied at the cut, the kernel keeps as many as HF but maybe
    not the same ones: torch.sort orders ties unstably.
- probabilities: against a float64 softmax of HF's processed scores (which
  equal grilly's bit for bit), over the kept tokens. The bound derives from
  the kernel's arithmetic: (|x - max| + 8) * 2^-24 relative per token. That
  covers the rounding of x - max, exp's ulp, the compensated sum and the
  division. The worst ratio is printed. torch's own float32 softmax misses
  that bound at a 152K vocabulary: it is 1.6e-5 off float64 there, grilly
  1.1e-6. So torch is not the reference for this number.
- draws: a chi-square goodness-of-fit against the distribution, with a
  fixed seed. The statistic is deterministic, and its threshold is the
  p = 1e-4 quantile.
"""

import numpy as np
import pytest
import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

import grilly
from grilly.infer import sampling

RNG = np.random.default_rng(11)


def hf_scores(logits, ids, *, temperature=1.0, top_k=0, top_p=1.0, penalty=1.0, do_sample=True):
    """HF's processors, in generate()'s order and under its conditions."""
    procs = []
    if penalty != 1.0:
        procs.append(RepetitionPenaltyLogitsProcessor(penalty))
    if do_sample:
        if temperature != 1.0:
            procs.append(TemperatureLogitsWarper(temperature))
        if top_k:
            procs.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            procs.append(TopPLogitsWarper(top_p))
    return LogitsProcessorList(procs)(torch.from_numpy(ids), torch.from_numpy(logits.copy()))


def case(batch, vocab, scale=3.0, history=24):
    logits = (RNG.standard_normal((batch, vocab)) * scale).astype(np.float32)
    ids = RNG.integers(0, vocab, (batch, history))
    return logits, ids


@pytest.mark.parametrize("penalty", [1.0, 1.3, 0.8])
@pytest.mark.parametrize("temperature", [1.0, 0.7, 1.9])
def test_processed_scores_are_hfs_bit_for_bit(penalty, temperature):
    logits, ids = case(3, 997)
    for do_sample in (False, True):
        want = hf_scores(logits, ids, temperature=temperature, penalty=penalty, do_sample=do_sample).numpy()
        got = sampling.processed_logits(grilly.from_numpy(logits), grilly.from_numpy(ids), do_sample=do_sample,
                                        temperature=temperature, repetition_penalty=penalty).numpy()
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


WORST = [0.0]  # worst probability error / bound seen, printed at the end


def compare(logits, ids, **kw):
    """Assert grilly's filtered distribution is HF's; return how many top-p
    boundary tokens (within 1e-6 of 1 - top_p) were decided differently."""
    hf = hf_scores(logits, ids, **kw).numpy()
    hf_kept = np.isfinite(hf)
    got = sampling.probabilities(grilly.from_numpy(logits), grilly.from_numpy(ids), temperature=kw.get("temperature", 1.0),
                                 top_k=kw.get("top_k", 0), top_p=kw.get("top_p", 1.0),
                                 repetition_penalty=kw.get("penalty", 1.0)).numpy()
    kept = got > 0
    boundary = 0
    before_top_p = hf_scores(logits, ids, **{**kw, "top_p": 1.0}).numpy()
    for r in np.nonzero((hf_kept != kept).any(1))[0]:
        assert kw.get("top_p", 1.0) < 1.0, "top-k alone must keep exactly HF's tokens"
        differ = hf_kept[r] != kept[r]
        if len(set(before_top_p[r][differ])) == 1 and hf_kept[r].sum() == kept[r].sum():
            continue  # the same number of tokens tied at the cut; torch.sort picks which
        # HF's cumulative probability of each token, in its ascending order
        before = torch.from_numpy(before_top_p[r:r + 1])
        order = torch.sort(before, -1).indices
        cum = torch.zeros_like(before).scatter_(1, order, torch.softmax(before, -1).gather(1, order).cumsum(-1))[0]
        gap = np.abs(cum.numpy()[differ] - (1 - kw["top_p"]))
        assert (gap < 1e-6).all(), f"row {r}: tokens decided differently away from the top-p boundary: {gap}"
        boundary += int(differ.sum())
    # probabilities against float64, over grilly's kept tokens
    x = np.where(kept, hf_scores(logits, ids, **{**kw, "top_k": 0, "top_p": 1.0}).numpy(), -np.inf).astype(np.float64)
    m = x.max(1, keepdims=True)
    e = np.exp(x - m)
    ref = e / e.sum(1, keepdims=True)
    bound = (np.abs(np.where(kept, x - m, 0)) + 8) * 2.0**-24 * ref
    ratio = np.abs(got - ref)[kept] / bound[kept]
    WORST[0] = max(WORST[0], float(ratio.max()))
    assert (ratio <= 1).all(), f"probability error {ratio.max():.2f} x its bound"
    assert (got[~kept] == 0).all()
    return boundary


@pytest.mark.parametrize("vocab", [997, 151936], ids=["small", "qwen-vocab"])
@pytest.mark.parametrize("top_k,top_p", [(0, 1.0), (1, 1.0), (20, 1.0), (50, 0.8), (20, 0.8), (0, 0.9), (0, 0.5),
                                         (0, 0.0), (5000, 0.95)])
def test_filtered_distribution_is_hfs(vocab, top_k, top_p):
    boundary = 0
    for penalty, temperature in [(1.0, 1.0), (1.05, 0.7), (1.3, 1.5)]:
        logits, ids = case(3, vocab)
        boundary += compare(logits, ids, temperature=temperature, top_k=top_k, top_p=top_p, penalty=penalty)
    assert boundary <= 1, f"{boundary} top-p boundary tokens decided differently"


def test_ties_and_masked_logits_follow_hf():
    logits, ids = case(2, 300)
    logits[:, :40] = 5.0  # 40 tied maxima: top_k=20 keeps all of them, as torch.topk's threshold does
    logits[:, 100:200] = -np.inf  # masked, as a bad-words or suppression processor leaves them
    compare(logits, ids, top_k=20)
    compare(logits, ids, top_k=250, top_p=0.9)
    compare(logits, ids, top_p=0.3)


def test_greedy_is_hfs_argmax_including_ties():
    logits, ids = case(4, 500)
    logits[0, [7, 300]] = 50.0  # a tie: the first index wins, as torch.argmax
    for penalty in (1.0, 1.4):
        want = hf_scores(logits, ids, penalty=penalty, do_sample=False).argmax(-1).numpy()
        got = sampling.next_token(grilly.from_numpy(logits), grilly.from_numpy(ids), repetition_penalty=penalty).numpy()
        np.testing.assert_array_equal(got, want)
    assert sampling.next_token(grilly.from_numpy(logits)).dtype == grilly.int64


def test_zz_report_worst_probability_error():
    print(f"probabilities: worst error / bound = {WORST[0]:.3f}")


def chi2_threshold(dof, z=3.719):
    """The chi-square quantile at p = 1e-4 (z is the normal quantile),
    by the Wilson-Hilferty approximation."""
    return dof * (1 - 2 / (9 * dof) + z * (2 / (9 * dof)) ** 0.5) ** 3


def chi_square(counts, probs):
    n = counts.sum()
    live = probs > 0
    assert counts[~live].sum() == 0, "drew a token with zero probability"
    expected = n * probs[live]
    return float(((counts[live] - expected) ** 2 / expected).sum()), int(live.sum()) - 1


@pytest.mark.parametrize("top_k,top_p", [(0, 1.0), (10, 1.0), (0, 0.7)])
def test_draws_follow_the_distribution(top_k, top_p):
    vocab, rows = 20, 20000
    logits = np.tile(RNG.standard_normal(vocab).astype(np.float32), (rows, 1))
    g = grilly.Generator().manual_seed(1234)
    kw = dict(temperature=0.8, top_k=top_k, top_p=top_p)
    probs = sampling.probabilities(grilly.from_numpy(logits[:1]), **kw).numpy()[0]
    tokens = sampling.next_token(grilly.from_numpy(logits), do_sample=True, generator=g, **kw).numpy()
    stat, dof = chi_square(np.bincount(tokens, minlength=vocab), probs)
    assert stat < chi2_threshold(dof), f"chi-square {stat:.1f} with {dof} degrees of freedom"


def test_rows_beyond_one_dispatch_and_determinism():
    logits, ids = case(21, 400)  # 8 rows per dispatch: three dispatches
    kw = dict(do_sample=True, temperature=0.9, top_k=40, top_p=0.9, repetition_penalty=1.1)
    a = sampling.next_token(grilly.from_numpy(logits), grilly.from_numpy(ids), generator=grilly.Generator().manual_seed(5), **kw)
    b = sampling.next_token(grilly.from_numpy(logits), grilly.from_numpy(ids), generator=grilly.Generator().manual_seed(5), **kw)
    c = sampling.next_token(grilly.from_numpy(logits), grilly.from_numpy(ids), generator=grilly.Generator().manual_seed(6), **kw)
    np.testing.assert_array_equal(a.numpy(), b.numpy())
    assert not np.array_equal(a.numpy(), c.numpy())
    probs = sampling.probabilities(grilly.from_numpy(logits), grilly.from_numpy(ids), temperature=0.9, top_k=40, top_p=0.9,
                                   repetition_penalty=1.1).numpy()
    assert (probs[np.arange(21), a.numpy()] > 0).all()


def test_rows_with_nothing_to_draw_give_minus_one():
    # All -inf (or, for multinomial, all zero): HF's softmax would give NaN and
    # torch.multinomial raise. The kernel returns -1 instead. This case once ran
    # a 2^32-step loop on one GPU thread.
    dead = grilly.from_numpy(np.full((3, 151936), -np.inf, np.float32))
    for kw in (dict(), dict(top_k=0, top_p=0.9), dict(top_k=20, top_p=0.5, temperature=0.7)):
        assert sampling.next_token(dead, do_sample=True, **kw).tolist() == [-1, -1, -1]
    assert sampling.next_token(dead).tolist() == [0, 0, 0]  # greedy: torch.argmax picks the first


def test_validation_as_hf():
    x = grilly.zeros(1, 10)
    with pytest.raises(RuntimeError, match="temperature"):
        sampling.next_token(x, do_sample=True, temperature=0.0)
    with pytest.raises(RuntimeError, match="top_p"):
        sampling.next_token(x, do_sample=True, top_p=1.5)
    with pytest.raises(RuntimeError, match="penalty"):
        sampling.next_token(x, repetition_penalty=0.0)
    with pytest.raises(ValueError, match="top_k"):
        sampling.next_token(x, do_sample=True, top_k=-1)
    with pytest.raises(RuntimeError, match="batch, vocab"):
        sampling.next_token(grilly.zeros(10))
    # greedy ignores the sampling-only parameters, as HF does
    assert sampling.next_token(x, temperature=0.0).tolist() == [0]


def test_multinomial_as_torch():
    w = grilly.tensor([[0.0, 1.0, 0.0, 3.0], [2.0, 0.0, 0.0, 0.0]])
    out = grilly.multinomial(w, 1, generator=grilly.Generator().manual_seed(0))
    assert tuple(out.shape) == (2, 1) and out.dtype == grilly.int64
    assert out.numpy()[1, 0] == 0 and out.numpy()[0, 0] in (1, 3)
    assert tuple(grilly.multinomial(grilly.tensor([1.0, 2.0]), 1).shape) == (1,)
    many = grilly.multinomial(w, 6, replacement=True)
    assert tuple(many.shape) == (2, 6) and set(many.numpy()[0]) <= {1, 3} and set(many.numpy()[1]) == {0}
    assert tuple(w.multinomial(1).shape) == (2, 1)

    weights = RNG.uniform(0, 1, 10).astype(np.float32)
    weights[3] = 0.0
    draws = grilly.multinomial(grilly.from_numpy(np.tile(weights, (20000, 1))), 1,
                               generator=grilly.Generator().manual_seed(9)).numpy().ravel()
    stat, dof = chi_square(np.bincount(draws, minlength=10), weights / weights.sum())
    assert stat < chi2_threshold(dof)

    assert grilly.multinomial(grilly.zeros(1, 4), 1).tolist() == [[-1]]  # no positive weight
    with pytest.raises(RuntimeError, match="1 or 2 dim"):
        grilly.multinomial(grilly.ones(2, 2, 2), 1)
    with pytest.raises(RuntimeError, match="n_sample <= 0"):
        grilly.multinomial(w, 0)
    with pytest.raises(NotImplementedError, match="without replacement"):
        grilly.multinomial(w, 2)
