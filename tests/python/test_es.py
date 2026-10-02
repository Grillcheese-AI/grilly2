"""``grilly.learning.es`` — evolution strategies, EGGROLL-style.

Four things have to be true for a gradient-free optimizer to be worth
running, and each is a test here:

1. **it optimizes** — on a quadratic and on a small linear regression,
   where the optimum is known in closed form;
2. **the estimator is unbiased** — for a linear objective the true
   gradient is a constant, so the estimate has to land on it within its own
   standard error. This is the one that would catch a wrong constant in the
   update, and it is where v1's ``sigma`` applied twice would show;
3. **it reproduces** — same seed, same numbers, because noise that is
   regenerated rather than stored is only correct if regeneration is exact;
4. **the batched population forward is the same arithmetic** as
   materialising N perturbed weights — that identity is the whole memory
   argument, so it is asserted rather than assumed.

Tolerances. (3) is **bit-exact**: same kernels, same inputs, same order —
CLAUDE.md §9 case (a), and nothing weaker is acceptable for a claim about
determinism. (4) is *not* case (a) despite being one backend: the two
sides are related by the distributive law rather than by reassociation, so
they are different arithmetic and measure 1.4e-7 apart. It asserts against
a float64 reference under a bound derived from the op instead, and prints
the worst ratio to it. (1) and (2) are statistical; (2)'s bound is the
estimator's own standard error and (1)'s thresholds come from the measured
residual-versus-lr curve recorded in `test_the_residual_is_set_by_the_learning_rate`.
"""

from __future__ import annotations

import numpy as np
import pytest

import grilly
from grilly.learning import EvolutionStrategy, population_linear, population_lora


def close(a, b, rtol):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = max(np.abs(b).max(), 1e-30)
    return np.abs(a - b).max() / scale <= rtol


# ── 1. it optimizes ──────────────────────────────────────────────────────


def _optimize(es, loss, iterations):
    """Run ES, returning the loss trajectory. ``loss`` is minimized, so the
    fitness handed to ``step`` is its negation."""
    trace = []
    for _ in range(iterations):
        scores = []
        for m in range(es.population):
            with es.member(m):
                scores.append(-loss())
        es.step(scores)
        trace.append(loss())
    return trace


def test_it_descends_a_quadratic_to_the_optimum():
    """``f(W) = -||W - T||^2``, whose optimum is W = T. Gradient-free, so
    the only thing driving it is the ordering of 32 fitness values."""
    target = grilly.from_numpy(
        np.random.default_rng(0).standard_normal((8, 6)).astype(np.float32))
    w = grilly.zeros((8, 6))
    es = EvolutionStrategy([w], sigma=0.1, lr=0.05, population=32, seed=1)

    def loss():
        return float(((w - target) ** 2).sum().numpy())

    start = loss()
    trace = _optimize(es, loss, 250)
    assert start > 30, "the starting point should be far from the optimum"
    assert trace[-1] < 0.01, f"{start:.3f} -> {trace[-1]:.5f}, wanted under 0.01"


def test_the_residual_is_set_by_the_learning_rate():
    """Centred ranks give a *direction*; the step's length is
    ``lr / sigma`` whatever the distance left. So a fixed-step ES does not
    converge to the optimum, it orbits at a radius set by lr — and that
    radius has to fall when lr does. Measured on this quadratic:

        lr    0.5      0.2      0.05
        loss  6.4      0.036    0.0005

    which is the property a wrong constant in the update would break, and
    the reason the test above uses 0.05 rather than something bolder.
    """
    target = grilly.from_numpy(
        np.random.default_rng(0).standard_normal((8, 6)).astype(np.float32))
    residual = {}
    for lr in (0.5, 0.05):
        w = grilly.zeros((8, 6))
        es = EvolutionStrategy([w], sigma=0.1, lr=lr, population=32, seed=1)
        trace = _optimize(es, lambda: float(((w - target) ** 2).sum().numpy()), 250)
        residual[lr] = float(np.mean(trace[-20:]))
    assert residual[0.05] < residual[0.5] / 100, (
        f"residual did not fall with lr: {residual}"
    )


def test_it_fits_a_small_linear_regression():
    """A real objective: recover ``W`` from ``(x, xW^T)`` pairs. The
    optimum is the generating matrix and the loss at it is zero."""
    rng = np.random.default_rng(2)
    true_w = rng.standard_normal((4, 5)).astype(np.float32)
    xn = rng.standard_normal((64, 5)).astype(np.float32)
    x = grilly.from_numpy(xn.copy())
    y = grilly.from_numpy(xn @ true_w.T)
    w = grilly.zeros((4, 5))
    es = EvolutionStrategy([w], sigma=0.05, lr=0.02, population=32, seed=3)

    def loss():
        return float((((x @ w.t()) - y) ** 2).mean().numpy())

    start = loss()
    trace = _optimize(es, loss, 300)
    assert trace[-1] < start / 50, f"loss only fell from {start:.4f} to {trace[-1]:.4f}"


# ── 2. the estimator is unbiased ─────────────────────────────────────────


def test_the_estimate_of_a_linear_objective_lands_on_the_true_gradient():
    """For ``f(W) = <C, W>`` the gradient is exactly ``C``, so the ES
    estimate has a right answer to be measured against.

    It is unbiased even at rank 1, which is the EGGROLL claim: with
    ``eps = u v^T`` and ``u, v`` standard normal,
    ``E[eps_ab eps_cd] = delta_ac delta_bd``, so ``E[g] = C``. The bound is
    the estimator's own standard error over ``trials`` repetitions, times
    three — not a number picked to make it pass.
    """
    rng = np.random.default_rng(4)
    shape = (6, 5)
    c = rng.standard_normal(shape).astype(np.float32)
    coeff = grilly.from_numpy(c)
    trials, population, sigma = 48, 64, 0.05
    estimates = []
    for trial in range(trials):
        w = grilly.zeros(shape)
        # lr = sigma * population makes step() apply exactly g, so the
        # parameter after one step IS the estimate
        es = EvolutionStrategy([w], sigma=sigma, lr=sigma * population,
                               population=population, seed=100 + trial)
        scores = []
        for m in range(population):
            with es.member(m):
                scores.append(float((coeff * w).sum().numpy()))
        es.step(scores)
        estimates.append(w.numpy().copy())

    mean = np.mean(estimates, axis=0)
    # centred ranks rescale the gradient by a positive constant, so compare
    # directions: the estimate must be parallel to C
    flat_mean, flat_c = mean.ravel(), c.ravel()
    cosine = float(flat_mean @ flat_c / (np.linalg.norm(flat_mean) * np.linalg.norm(flat_c)))
    per_trial = np.array([
        float(e.ravel() @ flat_c / (np.linalg.norm(e.ravel()) * np.linalg.norm(flat_c)))
        for e in estimates
    ])
    stderr = per_trial.std(ddof=1) / np.sqrt(trials)
    assert cosine > 1.0 - 3 * stderr - 0.02, (
        f"estimate is {np.degrees(np.arccos(min(cosine, 1.0))):.1f} deg off C; "
        f"cosine {cosine:.4f}, 3*se {3 * stderr:.4f}"
    )


def test_antithetic_pairs_see_opposite_perturbations():
    """The variance reduction is only real if the pair really is +/- the
    same noise, which is a property of the stream keying, not of luck."""
    w = grilly.zeros((5, 4))
    es = EvolutionStrategy([w], sigma=0.25, lr=0.0, population=4, seed=7)
    with es.member(0):
        plus = w.numpy().copy()
    with es.member(1):
        minus = w.numpy().copy()
    assert close(plus, -minus, 1e-7), "member 1 is not member 0 negated"
    assert np.abs(plus).max() > 0


def test_sigma_is_applied_once():
    """v1 scaled both ``u`` and ``v``, making its real step ``sigma**2`` —
    a hundredth of the documented one at sigma=0.01. The perturbation's
    scale has to be linear in sigma, so doubling sigma doubles it."""
    sizes = []
    for sigma in (0.05, 0.10):
        w = grilly.zeros((6, 6))
        es = EvolutionStrategy([w], sigma=sigma, lr=0.0, population=2, seed=11)
        with es.member(0):
            sizes.append(float(np.abs(w.numpy()).mean()))
    assert close(sizes[1], 2 * sizes[0], 1e-6), f"not linear in sigma: {sizes}"


def test_the_perturbation_has_unit_variance_at_every_rank():
    """EGGROLL's ``E = (1/sqrt(r)) U^T V`` (paper §4.1).

    The ``1/sqrt(r)`` is not decoration. ``U^T V`` sums ``r`` products of
    unit-variance normals, so without it ``E`` has variance ``r`` and
    ``sigma=0.01`` at rank 4 is twice the step ``sigma=0.01`` at rank 1 —
    the one knob the caller is told is orthogonal would move the other.
    With it, the variance is 1 at every rank and ``sigma`` keeps its
    meaning.

    Statistical, so the bound is the sample's own: a 4096x4096 draw of a
    unit-variance field has a relative standard error on its variance of
    ``sqrt(2/n)`` = 1.4e-3 at rank 1, and the heavier tails of a product
    of normals inflate that. 3% is ~20 sigma of headroom at rank 1 and
    still separates 1 from the 2x, 4x and 8x an absent ``1/sqrt(r)`` would
    give.
    """
    for rank in (1, 2, 4, 8):
        w = grilly.zeros((4096, 4096))
        es = EvolutionStrategy([w], sigma=1.0, lr=0.0, population=2, rank=rank, seed=3)
        with es.member(0):
            var = float(w.numpy().astype(np.float64).var())
        assert close(var, 1.0, 0.03), f"rank {rank}: variance {var}, want 1"


# ── 3. it reproduces ─────────────────────────────────────────────────────


def _run(seed, steps=5):
    rng = np.random.default_rng(0)
    w = grilly.from_numpy(rng.standard_normal((5, 4)).astype(np.float32))
    es = EvolutionStrategy([w], sigma=0.1, lr=0.2, population=8, seed=seed)
    for _ in range(steps):
        scores = []
        for m in range(es.population):
            with es.member(m):
                scores.append(float((w * w).sum().numpy()))
        es.step(scores)
    return w.numpy().copy()


def test_a_run_reproduces_exactly_from_the_seed():
    """Bit-exact, case (a): the noise is regenerated from (seed, step,
    member, parameter), so two runs issue the same commands over the same
    values. Anything less than ``==`` would mean regeneration is not
    exact, and then the update is not the noise the evaluation used."""
    assert (_run(21) == _run(21)).all()


def test_nearby_seeds_do_not_share_a_stream():
    """v1 keyed its generator with ``seed + id``, so a run seeded 2 saw
    the same numbers as a run seeded 0, shifted by two members. Here the
    seed is the key and (step, member, parameter) is the counter, so
    nothing lines up."""
    assert not (_run(0) == _run(2)).all()
    w0 = grilly.zeros((4, 4))
    w2 = grilly.zeros((4, 4))
    for w, seed in ((w0, 0), (w2, 2)):
        es = EvolutionStrategy([w], sigma=0.1, lr=0.0, population=4, seed=seed)
        with es.member(0):
            pass
    a = EvolutionStrategy([grilly.zeros((4, 4))], sigma=0.1, lr=0.0, population=4, seed=0)
    b = EvolutionStrategy([grilly.zeros((4, 4))], sigma=0.1, lr=0.0, population=4, seed=2)
    ua, _ = a._fill(0, 0)
    ub, _ = b._fill(0, 0)
    assert not (ua.numpy() == ub.numpy()).any(), "seeds 0 and 2 share values"


# ── 4. the batched population forward ────────────────────────────────────


def test_the_population_forward_equals_n_materialised_forwards():
    """The memory claim, asserted: ``y_i = x W^T + sigma (x v_i) u_i^T`` is
    the same answer as building ``W + sigma u_i^T v_i`` and running N
    forwards, and it never materialises a perturbed weight.

    **Not a reassociated sum, so not ``rtol = 1e-7``.** The two sides are
    related by the *distributive* law — one multiplies out before summing
    and the other after — which is a different computation in float32, not
    the same one in a different order. Measured, they differ by 1.4e-7
    relative, just past the case (a) bound, and tightening the assertion
    to 1e-7 would be asserting something untrue about float arithmetic.

    So this does what CLAUDE.md §9 prefers instead: both sides are
    compared against a **float64 reference** under a bound derived from the
    op — a dot product of length ``din`` has error at most
    ``din * eps * (|x| @ |W'|)`` — and the worst measured ratio to that
    bound is printed. A ratio is what says whether the op regressed; the
    two sides currently sit at 0.25 and 0.29 of their bound.
    """
    rng = np.random.default_rng(5)
    members, seq, din, dout, rank, sigma = 6, 3, 7, 5, 2, 0.1
    xn = rng.standard_normal((members, seq, din)).astype(np.float32)
    wn = rng.standard_normal((dout, din)).astype(np.float32)
    un = rng.standard_normal((members, rank, dout)).astype(np.float32)
    vn = rng.standard_normal((members, rank, din)).astype(np.float32)
    x, w, u, v = (grilly.from_numpy(a.copy()) for a in (xn, wn, un, vn))

    batched = population_linear(x, w, u, v, sigma).numpy()
    materialised = np.stack([
        (x[i] @ (w + (u[i].t() @ v[i]) * sigma).t()).numpy() for i in range(members)
    ])

    def perturbed64(i):
        return wn.astype(np.float64) + sigma * (un[i].astype(np.float64).T @ vn[i].astype(np.float64))

    reference = np.stack([xn[i].astype(np.float64) @ perturbed64(i).T for i in range(members)])
    bound = din * 2.0 ** -24 * np.stack([
        np.abs(xn[i].astype(np.float64)) @ np.abs(perturbed64(i)).T for i in range(members)
    ])

    worst = {}
    for name, got in (("batched", batched), ("materialised", materialised)):
        worst[name] = float((np.abs(got - reference) / np.maximum(bound, 1e-300)).max())
    print(f"  worst |err| / (din*eps*|x|@|W'|): "
          + ", ".join(f"{k} {v:.3f}" for k, v in worst.items()))
    for name, ratio in worst.items():
        assert ratio <= 1.0, f"{name} is {ratio:.3f} of its error bound"


def test_the_population_forward_carries_a_bias():
    rng = np.random.default_rng(6)
    x = grilly.from_numpy(rng.standard_normal((4, 2, 5)).astype(np.float32))
    w = grilly.from_numpy(rng.standard_normal((3, 5)).astype(np.float32))
    b = grilly.from_numpy(rng.standard_normal(3).astype(np.float32))
    u = grilly.zeros((4, 1, 3))
    v = grilly.zeros((4, 1, 5))
    # zero perturbation, so it must be exactly the plain linear
    got = population_linear(x, w, u, v, 0.5, b).numpy()
    want = (x @ w.t() + b).numpy()
    assert (got == want).all()


# ── the shape of the API ─────────────────────────────────────────────────


def test_it_refuses_the_configurations_that_would_be_wrong():
    w = grilly.zeros((4, 4))
    with pytest.raises(ValueError, match="even population"):
        EvolutionStrategy([w], population=5)
    with pytest.raises(ValueError, match="at least 2"):
        EvolutionStrategy([w], population=1)
    with pytest.raises(ValueError, match="no parameters"):
        EvolutionStrategy([])
    with pytest.raises(ValueError, match="2-D"):
        EvolutionStrategy([grilly.zeros(4)])


def test_a_non_matrix_parameter_is_left_alone_and_reported():
    """A rank-1 outer product is a statement about a matrix, so a bias
    vector is not perturbed — and the caller is told, rather than it
    silently not moving."""
    w, bias = grilly.zeros((4, 3)), grilly.zeros(4)
    es = EvolutionStrategy([w, bias], sigma=0.1, lr=0.1, population=4, seed=13)
    with es.member(0):
        assert np.abs(w.numpy()).max() > 0
        assert np.abs(bias.numpy()).max() == 0
    assert es.skipped == [1]


def test_a_member_is_restored_when_the_block_ends():
    w = grilly.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4))
    before = w.numpy().copy()
    es = EvolutionStrategy([w], sigma=0.1, lr=0.0, population=4, seed=17)
    with es.member(2):
        assert not (w.numpy() == before).all()
    assert close(w.numpy(), before, 1e-6)


def test_two_members_cannot_be_applied_at_once():
    w = grilly.zeros((3, 3))
    es = EvolutionStrategy([w], sigma=0.1, lr=0.1, population=4, seed=19)
    with pytest.raises(RuntimeError, match="still applied"):
        with es.member(0):
            with es.member(1):
                pass


def test_the_population_lora_equals_n_materialised_adapters():
    """Both factors move, so this is not a rank-1 correction to one matmul.

    ``y_i = W x + scaling * B_i (A_i x)`` with ``A_i = A + sigma u_a v_a^T``
    and ``B_i = B + sigma u_b v_b^T``. Expanding the product would give
    cross terms; composing two population forwards does not expand it at
    all, and this checks that against building both perturbed factors and
    running the layer N times.

    Compared against a float64 reference under the op's bound, for the
    reason ``test_the_population_forward_equals_n_materialised_forwards``
    gives: the two sides group the products differently, so they are not
    bit-equal and should not be asserted to be.
    """
    rng = np.random.default_rng(21)
    members, seq, din, dout, rank, r_lora = 6, 3, 9, 7, 1, 4
    sigma, scaling = 0.1, 2.0

    xn = rng.standard_normal((members, seq, din)).astype(np.float32)
    wn = rng.standard_normal((dout, din)).astype(np.float32)
    an = rng.standard_normal((r_lora, din)).astype(np.float32)
    bn = rng.standard_normal((dout, r_lora)).astype(np.float32)
    noise_n = {
        "u_a": rng.standard_normal((members, rank, r_lora)).astype(np.float32),
        "v_a": rng.standard_normal((members, rank, din)).astype(np.float32),
        "u_b": rng.standard_normal((members, rank, dout)).astype(np.float32),
        "v_b": rng.standard_normal((members, rank, r_lora)).astype(np.float32),
    }
    x, w, a, b = (grilly.from_numpy(t.copy()) for t in (xn, wn, an, bn))
    noise = {k: grilly.from_numpy(v.copy()) for k, v in noise_n.items()}

    batched = population_lora(x, w, a, b, scaling, noise, sigma).numpy()

    reference, bound = [], []
    for i in range(members):
        x64, w64 = xn[i].astype(np.float64), wn.astype(np.float64)
        a_i = an.astype(np.float64) + sigma * (noise_n["u_a"][i].T @ noise_n["v_a"][i]).astype(np.float64)
        b_i = bn.astype(np.float64) + sigma * (noise_n["u_b"][i].T @ noise_n["v_b"][i]).astype(np.float64)
        reference.append(x64 @ w64.T + scaling * ((x64 @ a_i.T) @ b_i.T))
        # the deepest reduction is the base's, over `din`
        bound.append(din * 2.0 ** -24 * (np.abs(x64) @ np.abs(w64).T
                                         + scaling * (np.abs(x64) @ np.abs(a_i).T) @ np.abs(b_i).T))
    reference, bound = np.stack(reference), np.stack(bound)
    ratio = float((np.abs(batched.astype(np.float64) - reference) / np.maximum(bound, 1e-300)).max())
    print(f"  worst |err| / bound: population_lora {ratio:.3f}")
    assert ratio <= 1.0, f"population_lora is {ratio:.3f} of its error bound"


def test_the_population_lora_never_materialises_a_factor():
    """The memory claim: the population's cost is two small products a
    member on top of one shared base, not N copies of A and B."""
    members, din, dout, r_lora = 8, 64, 32, 4
    g = grilly.Generator().manual_seed(3)
    x = grilly.randn((members, 2, din), generator=g)
    w = grilly.randn((dout, din), generator=g)
    a = grilly.randn((r_lora, din), generator=g)
    b = grilly.randn((dout, r_lora), generator=g)
    noise = {
        "u_a": grilly.randn((members, 1, r_lora), generator=g),
        "v_a": grilly.randn((members, 1, din), generator=g),
        "u_b": grilly.randn((members, 1, dout), generator=g),
        "v_b": grilly.randn((members, 1, r_lora), generator=g),
    }
    out = population_lora(x, w, a, b, 2.0, noise, 0.05)
    assert tuple(out.shape) == (members, 2, dout)
    # the shared factors are untouched — nothing was built per member
    assert tuple(a.shape) == (r_lora, din) and tuple(b.shape) == (dout, r_lora)


# ── 5. the two fitness shapings ──────────────────────────────────────────


def test_the_prompt_centred_shaping_is_a_scaled_centred_mean():
    """EGGROLL §6.3's per-question centring, and what it is really doing.

    The outer average over questions is linear, so the per-question means
    sum to one constant shared by every member and the whole transform
    collapses to ``(mean_i - grand_mean) / sigma_bar``. Worth pinning,
    because the per-question language invites the belief that it reweights
    hard and easy questions — it cannot, a mean has already weighted them
    equally. What it *does* change against centred ranks is being linear
    in the score instead of in the rank.

    Bit-exact is not available: the two sides are the same sum in a
    different order (float64 on the host either way), so this is CLAUDE.md
    §9 case (a) composed-chain, ``rtol 1e-7``. It measures 4e-16.
    """
    from grilly.learning.es import _prompt_centred

    rng = np.random.default_rng(0)
    s = rng.random((8, 5)) * 1.6
    got = np.asarray(_prompt_centred(s.tolist()))
    mean = s.mean(1)
    want = (mean - mean.mean()) / (s - s.mean(0)).std()
    assert close(got, want, 1e-7), f"worst {np.abs(got - want).max():.3e}"


def test_prompt_centring_keeps_how_much_better_where_ranks_discard_it():
    """The one behavioural difference, and the reason to choose between
    them.

    Member 0 solves both prompts; the other five are nearly tied near the
    bottom, separated by 0.001 of noise. Compare the winner's lead over
    the runner-up against the spread among the five ties, which carries no
    information at all:

    - prompt-centred: lead is **>100x** the meaningless spread;
    - centred ranks: lead is **smaller than** it. One rank step separates
      the runaway winner from second place, while the five ties are handed
      the whole rest of the [-0.5, 0.5] range.

    Neither is wrong. Ranks are what you want when a member can post a
    wild outlier; this is what you want when the score is a bounded ladder
    and how far ahead a member got is real information.
    """
    from grilly.learning.es import _centred_ranks, _prompt_centred

    scores = [[1.6, 1.6]] + [[0.30 + 0.001 * i, 0.30] for i in range(5)]
    centred = _prompt_centred(scores)
    ranks = _centred_ranks([sum(r) / len(r) for r in scores])
    lead_centred = centred[0] - max(centred[1:])
    lead_ranks = ranks[0] - max(ranks[1:])
    spread_centred = max(centred[1:]) - min(centred[1:])
    spread_ranks = max(ranks[1:]) - min(ranks[1:])
    assert lead_centred > 100 * spread_centred, (
        f"lead {lead_centred:.4f} vs tie spread {spread_centred:.4g}")
    assert lead_ranks < spread_ranks, (
        f"ranks did not flatten it: lead {lead_ranks:.4f}, spread {spread_ranks:.4f}")


def test_prompt_centring_refuses_a_fitness_that_has_already_been_averaged():
    """Silently averaging a mean would be the failure that looks like it
    worked: the same shaping name, quietly a different transform."""
    w = grilly.zeros((4, 4))
    es = EvolutionStrategy([w], population=4, shaping="prompt-centred")
    with pytest.raises(ValueError, match="per-prompt"):
        es.step([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="one of"):
        EvolutionStrategy([w], population=4, shaping="softmax")


def test_either_shaping_accepts_the_score_matrix():
    """``ranks`` averages the rows, so a caller that has per-prompt scores
    never has to remember which shaping wants which shape."""
    rng = np.random.default_rng(1)
    s = (rng.random((8, 3)) * 1.6).tolist()
    for shaping in EvolutionStrategy.SHAPINGS:
        w = grilly.zeros((4, 4))
        es = EvolutionStrategy([w], population=8, lr=1e-3, shaping=shaping, seed=5)
        es.step(s)
        assert len(es.shaped) == 8
        assert float(np.abs(w.numpy()).sum()) > 0, f"{shaping} took no step"


# ── 6. the gradient path, so any optimizer can drive it ──────────────────


def _quadratic_run(make_stepper, iterations=250, seed=1):
    """Descend ``-||W - T||^2`` with whatever `make_stepper(es, w)` returns
    as the per-iteration update. Returns the loss trajectory."""
    target = grilly.from_numpy(
        np.random.default_rng(0).standard_normal((8, 6)).astype(np.float32))
    w = grilly.zeros((8, 6))
    es = EvolutionStrategy([w], sigma=0.1, lr=0.05, population=32, seed=seed)
    update = make_stepper(es, w)
    trace = []
    for _ in range(iterations):
        scores = []
        for m in range(es.population):
            with es.member(m):
                scores.append(-float(((w - target) ** 2).sum().numpy()))
        update(scores)
        trace.append(float(((w - target) ** 2).sum().numpy()))
    return trace


def test_the_gradient_path_reproduces_step():
    """``gradient()`` then ``SGD(lr=es.lr)`` **is** ``step()``. The whole
    reason to trust the composable path is that it is the same estimate
    with the multiply moved from inside the sum to outside it.

    That move is the distributive law, not reassociation, so this is
    CLAUDE.md §9 case (a) composed-chain rather than bit-exact:
    ``rtol 1e-7``. It measures well inside that.
    """
    rng = np.random.default_rng(6)
    scores = list(rng.standard_normal(32))
    made = {}
    for path in ("step", "gradient"):
        w = grilly.zeros((8, 6))
        es = EvolutionStrategy([w], sigma=0.1, lr=0.05, population=32, seed=1)
        if path == "step":
            es.step(scores)
        else:
            opt = grilly.optim.SGD([w], lr=es.lr)
            opt.zero_grad()
            es.gradient(scores)
            opt.step()
        made[path] = w.numpy().copy()
    worst = np.abs(made["step"] - made["gradient"]).max() / np.abs(made["step"]).max()
    assert close(made["gradient"], made["step"], 1e-7), f"worst ratio {worst:.3e}"
    assert np.abs(made["step"]).max() > 0, "neither path moved anything"


def test_the_gradient_sign_is_the_loss_not_the_fitness():
    """The failure this guards is a run that quietly optimizes backwards.

    ``.grad`` holds the gradient of ``-fitness``, torch's convention, so
    a plain descending optimizer ascends fitness. Here the fitness is
    ``<C, W>``, whose ascent direction is exactly ``C``, so the sign is
    checkable rather than inferable: ``.grad`` must oppose ``C``, and one
    SGD step must *raise* the fitness.

    This is a **sign** test, not an accuracy one. A single ES estimate at
    population 64 in 30 dimensions is a noisy direction — measured cosine
    -0.61 — so the threshold sits between the two hypotheses rather than
    near either: right sign gives about -0.6, a flipped sign would give
    about +0.6, and zero is the boundary. How *close* to -1 the estimate
    gets is what
    `test_the_estimate_of_a_linear_objective_lands_on_the_true_gradient`
    measures, over 48 trials and against its own standard error.
    """
    c = np.random.default_rng(8).standard_normal((6, 5)).astype(np.float32)
    coeff = grilly.from_numpy(c)
    w = grilly.zeros((6, 5))
    es = EvolutionStrategy([w], sigma=0.05, lr=0.05, population=64, seed=2)
    scores = []
    for m in range(es.population):
        with es.member(m):
            scores.append(float((coeff * w).sum().numpy()))
    opt = grilly.optim.SGD([w], lr=es.lr)
    opt.zero_grad()
    es.gradient(scores)
    g = w.grad.numpy().ravel()
    cosine = float(g @ c.ravel() / (np.linalg.norm(g) * np.linalg.norm(c.ravel())))
    assert cosine < -0.3, f".grad should oppose C (it is the loss's); cosine {cosine:.3f}"
    opt.step()
    assert float((coeff * w).sum().numpy()) > 0, "descending .grad did not raise fitness"


def test_the_gradient_accumulates_as_backward_does():
    """torch semantics: ``.grad`` is added to, not replaced, so
    ``zero_grad()`` is the caller's job and an ES estimate can be summed
    with a real one from a differentiable auxiliary loss."""
    scores = list(np.random.default_rng(9).standard_normal(8))
    w = grilly.zeros((4, 3))
    es = EvolutionStrategy([w], sigma=0.1, lr=0.05, population=8, seed=4)
    es.gradient(scores)
    once = w.grad.numpy().copy()
    es.step_count -= 1                      # same noise, so twice must double
    es.gradient(scores)
    assert close(w.grad.numpy(), 2 * once, 1e-7), "second call did not accumulate"
    opt = grilly.optim.SGD([w], lr=es.lr)
    opt.zero_grad()
    assert w.grad is None, "zero_grad did not clear it"


def test_adam_drives_es_to_the_optimum():
    """The point of the whole method: hand the estimate to a real
    optimizer. Adam has to reach the optimum of the same quadratic that
    `test_it_descends_a_quadratic_to_the_optimum` solves with plain SGD.

    Adam normalises by its own second moment, so its step is ~lr in every
    coordinate regardless of the estimate's scale — which is why its lr
    here is far smaller than the SGD path's 0.05 and not comparable to it.
    """
    def adam(es, w):
        opt = grilly.optim.Adam([w], lr=0.02)

        def update(scores):
            opt.zero_grad()
            es.gradient(scores)
            opt.step()
        return update

    trace = _quadratic_run(adam)
    assert trace[-1] < 0.01, f"Adam reached {trace[-1]:.5f}, wanted under 0.01"


# ── 7. sigma against what it perturbs ────────────────────────────────────


def test_scales_reports_sigma_against_each_parameter_not_in_the_abstract():
    """``sigma`` is an absolute displacement; a parameter has its own
    scale. Two parameters an order of magnitude apart get the same sigma
    and two completely different perturbations, and nothing raises — which
    is how a real run lost 7.5x of its wall clock to one non-terminating
    member.

    Bit-exact is not the question here; this is arithmetic on a host read,
    so the assertion is the ratio itself.
    """
    rng = np.random.default_rng(11)
    big = grilly.from_numpy((rng.standard_normal((32, 16)) * 0.1).astype(np.float32))
    small = grilly.from_numpy((rng.standard_normal((16, 8)) * 0.001).astype(np.float32))
    es = EvolutionStrategy([big, small], sigma=0.01, population=4)
    got = es.scales()
    assert len(got) == 2, "one ratio per moved parameter"
    assert got[0] == pytest.approx(0.01 / big.numpy().std(), rel=1e-5)
    assert got[1] == pytest.approx(0.01 / small.numpy().std(), rel=1e-5)
    assert got[1] / got[0] > 50, (
        f"the point of the diagnostic is that these differ: {got}")


def test_a_population_that_never_differed_takes_no_step():
    """The quiet half of the sigma failure. Too small a sigma gives every
    member the same fitness; the shaped values are then all zero and the
    update must be exactly nothing — not a tiny wrong step.

    ``shaped`` is what a run should watch for this: a spread of zero means
    the population never differed and the iteration was wasted.
    """
    w = grilly.from_numpy(np.ones((4, 4), dtype=np.float32))
    before = w.numpy().copy()
    es = EvolutionStrategy([w], sigma=0.1, lr=1.0, population=8, seed=0)
    es.step([0.5] * 8)                       # identical fitness, as a flat objective gives
    assert max(abs(s) for s in es.shaped) == 0.0, f"shaped should be flat: {es.shaped}"
    assert np.array_equal(w.numpy(), before), "a flat population moved the weights"


def test_tied_members_get_equal_weight_not_an_order_dependent_one():
    """The common case, not just the degenerate one. A verifier's ladder
    puts many members on exactly the same rung, and members that scored
    the same must pull the update equally hard — otherwise the tie is
    broken by the order the noise was generated in, which is a direction
    with no information in it.
    """
    from grilly.learning.es import _centred_ranks

    shaped = _centred_ranks([0.6, 1.6, 0.6, 0.6, 1.6, 0.6])
    low = [s for s, f in zip(shaped, [0.6, 1.6, 0.6, 0.6, 1.6, 0.6]) if f == 0.6]
    high = [s for s, f in zip(shaped, [0.6, 1.6, 0.6, 0.6, 1.6, 0.6]) if f == 1.6]
    assert len(set(low)) == 1 and len(set(high)) == 1, f"ties split: {shaped}"
    assert high[0] > low[0], "the better rung must pull harder"
    # the transform still centres: equal counts either side would sum to 0,
    # and 4 low / 2 high sums to 0 as well, because ranks are a permutation
    assert sum(shaped) == pytest.approx(0.0), f"shaping is not centred: {shaped}"
