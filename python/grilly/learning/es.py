"""Evolution strategies: optimize what you can only score, not differentiate.

The case this is for: the objective runs outside the model. A program the
model emits is compiled and executed by a VM, and what comes back is
"right" or "wrong" — there is no gradient through a verifier. ES asks only
for a number per candidate, so it fits, and it is forward-only, so a 12 GB
consumer card can run it.

The cost is the obvious one: a gradient estimate from N function values
instead of one backward pass. What makes it affordable at model scale is
the **low-rank perturbation** of EGGROLL (Sarkar et al., 2026,
arXiv:2511.16652): member i's perturbation is ``E_i = (1/sqrt(r)) U_i^T
V_i`` for ``U_i`` of shape ``(r, out)`` and ``V_i`` of ``(r, in)``, so a
population of N costs N*r*(out+in) numbers rather than N*out*in, and the
batched forward never materialises a perturbed weight at all —

    y_i = x_i W^T + sigma * (x_i V_i^T) U_i

is two small matrix products per member against the *shared* W. The
``1/sqrt(r)`` is what makes ``sigma`` mean the same thing at every rank:
with ``U`` and ``V`` standard normal it puts unit variance on every
element of ``E``, matching the full-rank Gaussian perturbation that
ordinary ES would draw (paper §4.1).

**Noise is never stored.** Element ``i`` of member ``m``'s perturbation is
a function of ``(seed, step, member, parameter, i)`` through a
counter-based Philox stream, so the update regenerates exactly the noise
the evaluation used. That is what makes the memory claim real: nothing is
kept between the forward and the update but N floats of fitness.

## What this does not do, and why

v1 shipped `eggroll-generate` / `eggroll-update`. The audit
(`docs/v1-shader-audit.md`) found four defects, and each one is a thing
this deliberately does differently:

- **sigma applied twice.** v1 scaled both ``u`` and ``v`` by sigma, so its
  actual step was ``sigma**2`` — at the documented sigma=0.01, a hundredth
  of the intended one. Here sigma multiplies the product, once.
- **strided seeds.** v1 keyed its stream with ``seed + id``, so a run
  seeded 2 saw the same numbers as a run seeded 0 shifted by two members.
  Here the key is the seed and the *stream* carries (step, member,
  parameter), so nothing collides by construction.
- **the merit rule.** v1 raised a per-weight "merit" multiplier whenever
  ``|avg_delta| > 1e-8``, which under dense Gaussian noise is true for
  99.96% of weights — a global step size compounding x1.02 a step, x2e4
  after a thousand. Dropped. If per-weight adaptivity is wanted, it comes
  with a test first.
- **top-k positive-only updates.** v1 kept the best few members and moved
  toward them. That throws away the half of the information that says
  where *not* to go, and it biases the estimate. Here the estimator is the
  standard one, ``g = 1/(N*sigma) * sum_i f_i eps_i``, over centred ranks
  and antithetic pairs — both of which are variance reduction, not
  selection.

## The step size is not in units of sigma, and that is a trap

The update is ``p += (lr/sigma) * (1/N) sum_i s_i E_i``, so **``lr/sigma``
is the number that sets the step**, and it is what the paper calls
``alpha``. At ``sigma=0.01, lr=0.02`` — a pair that looks unremarkable —
the measured per-element step is **7.4x sigma**: the update walks seven
times further than the radius at which the fitness was measured, so the
scores it is built from describe somewhere else entirely. The paper's
``alpha`` is 0.001 in both its LLM run and its frozen-base LoRA run, which
at ``sigma=0.01`` is ``lr=1e-5`` and a step of 0.0045 sigma. That is the
default here. Sweep it, but sweep it from there.

## Using it

    es = EvolutionStrategy(params, sigma=0.01, lr=1e-5, population=32)
    for _ in range(iterations):
        fitness = []
        for m in range(es.population):
            with es.member(m):
                fitness.append(evaluate(model))
        es.step(fitness)

``params`` are the tensors that move — for a LoRA adapter, its factors,
with the base frozen. Only 2-D parameters are perturbed; anything else is
left alone and reported by :attr:`skipped`, because a rank-1 outer product
is a statement about a matrix.

``step`` folds the estimate straight into the weights and stores no
gradient, which at r=64 over a 2.6B base saves 322 MB. To put the estimate
in ``.grad`` instead and let a real optimizer take the step — which is
what the paper does for everything but its Countdown run — use
``gradient``:

    opt = grilly.optim.AdamW(params, lr=1e-4)
    ...
    opt.zero_grad()
    es.gradient(fitness)
    opt.step()

One or the other per iteration, never both: each advances the noise
stream by one step.
"""

from __future__ import annotations

import builtins as _builtins
from contextlib import contextmanager

from .. import Tensor, _C, no_grad, zeros

__all__ = ["EvolutionStrategy", "population_linear", "population_lora"]


def _centred_ranks(fitness) -> list:
    """Fitness replaced by its rank, mapped to ``[-0.5, 0.5]``.

    The standard ES shaping, and it is what makes the method robust to the
    scale of the objective: a verifier that returns 0 or 1, and one that
    returns a log-likelihood, produce the same step. It also bounds the
    influence of a single outlier member, which a raw-fitness estimator
    does not.

    **Ties share the average of the ranks they span**, which is the
    ordinary rank transform and is load-bearing here rather than a nicety.
    Breaking ties by index would hand a population whose members all
    scored the same a full ``[-0.5, 0.5]`` spread — every pair weighted
    ``-1/(n-1)`` — and the optimizer would take a full-size step along a
    direction set by nothing but the order the noise happened to be
    generated in. A flat objective has to produce a flat shaping and no
    step; a verifier's reward *is* flat early on, and a binary one is flat
    across most of a population for a long time.
    """
    n = len(fitness)
    if n < 2:
        return [0.0] * n
    order = sorted(range(n), key=lambda i: fitness[i])
    ranks = [0.0] * n
    place = 0
    while place < n:
        end = place
        while end + 1 < n and fitness[order[end + 1]] == fitness[order[place]]:
            end += 1
        shared = (place + end) / 2 / (n - 1) - 0.5
        for k in range(place, end + 1):
            ranks[order[k]] = shared
        place = end + 1
    return ranks


def _prompt_centred(scores) -> list:
    """Per-prompt centring, divided by one global std (EGGROLL §6.3).

    The shaping the paper uses for its reasoning fine-tunes, where a
    member's fitness is a mean over questions:

        s_i = (1/m) sum_j (s_ij - mu_j) / sigma_bar

    ``mu_j`` is the *population's* mean on question j and ``sigma_bar`` is
    a single std over the whole centred matrix.

    **Read the per-question centring for less than it looks like.** The
    outer average is linear, so ``sum_j mu_j`` is the same constant for
    every member and the whole thing collapses to

        s_i = (mean_i - grand_mean) / sigma_bar

    — asserted in ``test_the_prompt_centred_shaping_is_a_scaled_centred_mean``
    to 4e-16. Subtracting a per-question mean cannot reweight questions,
    because a mean has already given them equal weight; a question nobody
    solves adds the same constant to every member either way. The paper's
    own remark about this is about the *denominator*: using one global
    ``sigma_bar`` instead of GRPO's per-question ``sigma_j`` is what keeps
    every question equally weighted, since per-question scaling would
    inflate a question where the population barely differs.

    So what this actually buys over :func:`_centred_ranks` is one thing:
    **it is linear in the score rather than in the rank.** A member that
    is much better gets a proportionally larger weight; centred ranks cap
    it at 0.5 no matter how much better it was. That is the right trade
    when the score is a bounded ladder with no heavy tail — which is the
    case the paper uses it for, and the case here — and the wrong one when
    a single member can post a wild outlier, which is why ranks stay the
    default.

    Returns one number per member. Its spread depends on how correlated
    members are across prompts: 1/sqrt(m) if prompt scores are
    independent, up to 1 if a member is uniformly better. At m=16 that is
    0.25 to 1.0 against centred ranks' fixed 0.289, so switching shaping
    can move the effective step by up to ~3.5x — :attr:`shaped` is kept so
    a run can log it rather than guess.
    """
    n = len(scores)
    m = len(scores[0])
    if any(len(row) != m for row in scores):
        raise ValueError("every member needs a score for every prompt")
    if n < 2 or m < 1:
        return [0.0] * n
    means = [sum(scores[i][j] for i in range(n)) / n for j in range(m)]
    dev = [[scores[i][j] - means[j] for j in range(m)] for i in range(n)]
    flat = [d for row in dev for d in row]
    sd = (sum(d * d for d in flat) / len(flat)) ** 0.5
    if sd <= 0.0:
        # every member scored the same on every prompt: no signal, no step
        return [0.0] * n
    return [sum(row) / (m * sd) for row in dev]


class EvolutionStrategy:
    """A forward-only optimizer, shaped like a torch one.

    ``params`` is an iterable of tensors, ``sigma`` the perturbation scale,
    ``lr`` the step size, ``population`` the number of members scored per
    step. ``antithetic`` pairs every member with its negation, which
    removes the estimator's first-order noise and requires an even
    population.

    ``rank`` is the perturbation's rank. It is **not** a free accuracy
    knob: a low-rank ``E`` has no score function in the ordinary sense, so
    EGGROLL substitutes the Gaussian one, ``S(E) = -E``, and the resulting
    estimator is an approximation whose error to the true ES gradient is
    ``O(1/r)`` (paper Theorem 4). Rank 1 is therefore *biased* — and is
    also what every LLM experiment in the paper uses, because the bias is
    small next to what a population of this size costs to reduce.

    ``lr`` is in units where ``sigma`` has already been divided out:
    ``p += (lr/sigma) * (1/N) sum_i s_i E_i``, so the paper's ``alpha`` is
    ``lr/sigma``, not ``lr``. Its LLM and HFT runs use ``alpha = 0.001``
    (Tables 3, 4).
    """

    #: ``lr/sigma`` — the paper's ``alpha``, which is the quantity that
    #: actually sets the step. Its LLM fine-tuning (Table 3) and its
    #: frozen-base LoRA run (Table 4) both use 0.001, across population
    #: sizes from 256 to 65536, so it is the one hyperparameter the paper
    #: does not vary. The defaults below are that number at sigma=0.01.
    PAPER_ALPHA = 0.001

    #: how a population's scores become the weights in the update
    SHAPINGS = ("ranks", "prompt-centred")

    def __init__(self, params, sigma: float = 0.01, lr: float = 1e-5,
                 population: int = 32, rank: int = 1, antithetic: bool = True,
                 seed: int = 0, shaping: str = "ranks") -> None:
        self.params = [p for p in params]
        if not self.params:
            raise ValueError("EvolutionStrategy got no parameters")
        if population < 2:
            raise ValueError(f"population must be at least 2, got {population}")
        if antithetic and population % 2:
            raise ValueError(
                f"antithetic pairs need an even population, got {population}"
            )
        if sigma <= 0 or rank < 1:
            raise ValueError("sigma must be positive and rank at least 1")
        if shaping not in self.SHAPINGS:
            raise ValueError(f"shaping must be one of {self.SHAPINGS}, got {shaping!r}")
        self.shaping = shaping
        #: the last step's shaped fitness, for logging its spread
        self.shaped = []
        self.sigma, self.lr = _builtins.float(sigma), _builtins.float(lr)
        self.population, self.rank = _builtins.int(population), _builtins.int(rank)
        self.antithetic, self.seed = _builtins.bool(antithetic), _builtins.int(seed)
        self.step_count = 0
        #: parameters left alone: a rank-1 outer product is about a matrix
        self.skipped = [i for i, p in enumerate(self.params) if p.dim() != 2]
        self._moved = [i for i, p in enumerate(self.params) if p.dim() == 2]
        if not self._moved:
            raise ValueError("EvolutionStrategy needs at least one 2-D parameter")
        self._applied = None
        #: ``E = (1/sqrt(r)) U^T V``, carried on ``U`` so every path that
        #: reads the noise gets it. Without it ``E``'s elements have
        #: variance ``r`` and ``sigma`` silently means something different
        #: at every rank; with it they have variance 1 at every rank, which
        #: is what makes a rank a choice about the estimator rather than
        #: about the step size. At rank 1 the factor is 1.0 and nothing
        #: changes.
        self._u_std = self.rank ** -0.5
        # regenerated, never stored: one (rank, out) and one (rank, in) per
        # moved parameter, reused as scratch across members and steps
        self._u = [zeros((self.rank, self.params[i].shape[0])) for i in self._moved]
        self._v = [zeros((self.rank, self.params[i].shape[1])) for i in self._moved]

    # ── the stream a member's noise comes from ───────────────────────────

    def _stream(self, pair: int, which: int, param: int) -> int:
        """A distinct counter-based stream per (step, pair, parameter, u/v).

        Distinct by construction rather than by luck: the parts are packed
        into disjoint bit ranges of the counter, so no two ever collide and
        two runs with nearby seeds share nothing (v1's `seed + id` keying
        made seeds 0 and 2 produce shifted copies of one stream)."""
        pairs = self.population // (2 if self.antithetic else 1)
        return ((self.step_count * pairs + pair) * len(self._moved) + param) * 2 + which

    def _fill(self, pair: int, param: int) -> tuple:
        """Member ``pair``'s ``u`` and ``v`` for moved parameter ``param``."""
        u, v = self._u[param], self._v[param]
        _C.philox_normal_(u._t, self.seed, self._stream(pair, 0, param), self._u_std)
        _C.philox_normal_(v._t, self.seed, self._stream(pair, 1, param), 1.0)
        return u, v

    def _pair_and_sign(self, member: int) -> tuple:
        if not self.antithetic:
            return member, 1.0
        return member // 2, 1.0 if member % 2 == 0 else -1.0

    def population_noise(self, param: int, repeat: int = 1) -> tuple:
        """Every member's ``u`` and ``v`` for moved parameter ``param``,
        stacked: ``(population * repeat, rank, out)`` and ``(..., in)``.

        This is what turns a population into a *batch*. Scoring members one
        at a time costs N forward passes; stacking their noise along the
        batch dimension costs one, because
        :func:`population_lora` applies each row's own perturbation against
        the shared weight. On CubbyLLM's emitter that is the difference
        between an ES iteration taking an hour and taking minutes.

        ``repeat`` tiles each member across that many consecutive rows, for
        a batch of (member, prompt) pairs: member 0 occupies rows
        ``0..repeat-1``, member 1 the next, and so on, which is the layout
        ``Tensor.repeat_interleave`` would give and what the caller's
        prompts must match.

        The antithetic sign is folded into ``u``, so member ``2j+1`` really
        is member ``2j`` negated — the same perturbation the one-at-a-time
        path applies, not an independent draw that merely looks like one.
        """
        from .. import cat

        us, vs = [], []
        for member in range(self.population):
            pair, sign = self._pair_and_sign(member)
            u, v = self._fill(pair, param)
            row_u = (u * sign if sign != 1.0 else u.clone()).reshape(1, self.rank, -1)
            row_v = v.clone().reshape(1, self.rank, -1)
            for _ in range(repeat):
                us.append(row_u)
                vs.append(row_v)
        return cat(tuple(us), 0), cat(tuple(vs), 0)

    @property
    def moved(self) -> list:
        """Indices into ``params`` of the tensors this optimizer moves."""
        return list(self._moved)

    def scales(self) -> list:
        """``sigma / std(p)`` for every moved parameter — **check this
        before a run**, because ``sigma`` alone says nothing.

        ``sigma`` is an absolute displacement and a parameter has its own
        scale, so the same ``sigma`` is a nudge for one tensor and a
        replacement for another. The ratio is the thing with meaning, and
        a set of parameters that were initialised differently will not
        share one: a LoRA adapter's ``A`` is init-scaled while its ``B``
        starts at zero and only grows, so after training they differ by
        most of an order of magnitude. Measured on a trained r=64 adapter
        over LFM2.5-2.6B, ``sigma=0.01`` is 0.75x the spread of every
        ``A`` and **5.8x** the spread of every ``B``.

        Both ends of the ratio fail, and neither raises:

        - **too large** — members stop being neighbours of the parameters
          and become unrelated models. For a language model that also
          costs wall clock, because a model emitting nonsense does not
          emit its stop token, and a batch is only as fast as its slowest
          row.
        - **too small** — the fitness is identical for every member, the
          shaped values are all zero, and the step is a no-op. Check
          :attr:`shaped` for this one; a spread of zero means the
          population never differed.

        A ratio somewhere around 0.1 to 2 is the usual working range, but
        it is a thing to measure against the objective rather than to take
        from here. This reads every parameter back to the host, so call it
        between iterations, never inside a capture.
        """
        from .. import sqrt

        out = []
        for i in self._moved:
            p = self.params[i]
            # torch would be p.std(); grilly has no std yet, and a
            # diagnostic is not the place to add one
            spread = float(sqrt(((p - p.mean()) ** 2).mean()).numpy())
            out.append(self.sigma / max(spread, 1e-30))
        return out

    # ── applying a member ────────────────────────────────────────────────

    def _add(self, member: int, scale: float) -> None:
        pair, sign = self._pair_and_sign(member)
        with no_grad():
            for slot, index in enumerate(self._moved):
                u, v = self._fill(pair, slot)
                # sigma once, on the product — not on u and on v
                self.params[index].add_(u.t() @ v, alpha=sign * scale)

    @contextmanager
    def member(self, index: int):
        """Apply member ``index``'s perturbation for the duration.

        Restored by subtracting the same regenerated noise rather than
        from a saved copy, so a population of any size costs no extra
        memory. The restore is exact when it is exact — float addition is
        not associative, so a parameter drifts by at most the rounding of
        one add-and-subtract per step, which is the price of not holding N
        copies of the model."""
        if not 0 <= index < self.population:
            raise IndexError(f"member {index} is outside a population of {self.population}")
        if self._applied is not None:
            raise RuntimeError(f"member {self._applied} is still applied")
        self._applied = index
        self._add(index, self.sigma)
        try:
            yield self
        finally:
            self._add(index, -self.sigma)
            self._applied = None

    # ── the update ───────────────────────────────────────────────────────

    def _shape(self, fitness) -> list:
        """The shaped fitness, validated. Also records :attr:`shaped`.

        ``fitness`` is one number per member, or — when the objective is a
        mean over prompts — one **row** per member, ``(population,
        prompts)``. A matrix is accepted under either shaping: ``"ranks"``
        averages the row first, ``"prompt-centred"`` needs it to measure
        ``sigma_bar`` across the matrix. Passing the matrix is therefore
        never wrong, and is what a caller that has one should do.
        """
        if self._applied is not None:
            raise RuntimeError(f"member {self._applied} is still applied")
        rows = list(fitness)
        if len(rows) != self.population:
            raise ValueError(
                f"expected {self.population} fitness values, got {len(rows)}"
            )
        matrix = bool(rows) and not isinstance(rows[0], (int, float))
        if matrix:
            rows = [[_builtins.float(x) for x in row] for row in rows]
        else:
            rows = [_builtins.float(f) for f in rows]
        if self.shaping == "prompt-centred":
            if not matrix:
                raise ValueError(
                    "shaping='prompt-centred' needs a (population, prompts) score "
                    "matrix; a single mean per member has already discarded what it "
                    "centres. Pass the per-prompt scores, or use shaping='ranks'."
                )
            self.shaped = _prompt_centred(rows)
        else:
            if matrix:
                rows = [sum(row) / max(len(row), 1) for row in rows]
            self.shaped = _centred_ranks(rows)
        return self.shaped

    def _weights(self, shaped):
        """``(pair, weight)`` for every pair that contributes anything.

        With antithetic pairs the sum over members collapses to one term
        per pair, ``(s_+ - s_-) eps`` — which is where the variance
        reduction shows up: a component of the noise that helps one member
        and hurts its twin cancels instead of being read as signal.
        """
        pairs = self.population // 2 if self.antithetic else self.population
        for pair in range(pairs):
            weight = (shaped[2 * pair] - shaped[2 * pair + 1] if self.antithetic
                      else shaped[pair])
            if weight != 0.0:
                yield pair, weight

    def step(self, fitness) -> None:
        """One ES step from the population's fitnesses (higher is better).

        ``p += lr/(N*sigma) * sum_i s_i eps_i`` with ``s`` the shaped
        fitness: the estimate and a plain SGD step in one pass, so no
        gradient is ever stored. At r=64 over a 2.6B base that is 80.5M
        values — 322 MB — that :meth:`gradient` would have to materialise
        and this does not, which is the reason both exist.

        ``fitness`` is one number per member, or a ``(population,
        prompts)`` matrix — see :meth:`_shape`. Use this **or**
        :meth:`gradient` per iteration, never both: each advances the
        noise stream by one step.
        """
        shaped = self._shape(fitness)
        scale = self.lr / (self.population * self.sigma)
        with no_grad():
            for slot, index in enumerate(self._moved):
                p = self.params[index]
                for pair, weight in self._weights(shaped):
                    u, v = self._fill(pair, slot)
                    p.add_(u.t() @ v, alpha=scale * weight)
        self.step_count += 1

    def gradient(self, fitness) -> None:
        """Put the ES estimate in every moved parameter's ``.grad``, so a
        ``grilly.optim`` optimizer can take the step instead.

        This is how EGGROLL is actually run: the paper uses AdamW for its
        RL sweeps (Tables 7-9) and Adam for the quantised distillation
        (§K.2), keeping plain gradient descent only for Countdown
        (Table 10). Momentum matters more here than it does for a real
        gradient, because an ES estimate from a population of tens is
        mostly noise and a moment estimate is what averages it across
        iterations.

            opt = grilly.optim.Adam(factors, lr=1e-5)
            ...
            opt.zero_grad()
            es.gradient(fitness)
            opt.step()

        **The sign is torch's, not ES's.** ``.grad`` gets the gradient of
        ``-fitness``, because that is the loss an optimizer descends —
        so the ordinary call above *ascends* fitness and
        ``maximize=True`` would send it backwards. ES maximises; torch
        optimizers minimise; the negation lives here so that nothing
        downstream has to know.

        ``lr`` is the optimizer's, not this object's: :attr:`lr` is not
        read at all on this path. ``optim.SGD(params, lr=es.lr)``
        reproduces :meth:`step` exactly, which is the invariant
        ``test_the_gradient_path_reproduces_step`` asserts.

        Accumulates into an existing ``.grad`` and creates one when there
        is none, as ``backward()`` does — so call ``opt.zero_grad()``
        first, and so an ES estimate can be added to a real gradient from
        a differentiable auxiliary loss.
        """
        shaped = self._shape(fitness)
        # negated: .grad is of the loss, and the loss is -fitness
        scale = -1.0 / (self.population * self.sigma)
        with no_grad():
            for slot, index in enumerate(self._moved):
                p = self.params[index]
                if p.grad is None:
                    p.grad = zeros(tuple(p.shape))
                g = p.grad
                for pair, weight in self._weights(shaped):
                    u, v = self._fill(pair, slot)
                    g.add_(u.t() @ v, alpha=scale * weight)
        self.step_count += 1

    def state_dict(self) -> dict:
        return {"sigma": self.sigma, "lr": self.lr, "population": self.population,
                "rank": self.rank, "antithetic": self.antithetic, "seed": self.seed,
                "shaping": self.shaping, "step_count": self.step_count}

    def load_state_dict(self, sd: dict) -> None:
        for key, value in sd.items():
            setattr(self, key, value)
        # `rank` is not just a number: it sets the noise scale and the
        # scratch shapes, so restoring it has to rebuild both or the noise
        # silently keeps the old rank's variance
        self._u_std = self.rank ** -0.5
        if self._u and self._u[0].shape[0] != self.rank:
            self._u = [zeros((self.rank, self.params[i].shape[0])) for i in self._moved]
            self._v = [zeros((self.rank, self.params[i].shape[1])) for i in self._moved]

    def __repr__(self) -> str:
        return (f"EvolutionStrategy(population={self.population}, sigma={self.sigma}, "
                f"lr={self.lr}, rank={self.rank}, antithetic={self.antithetic}, "
                f"step={self.step_count})")


def population_linear(x: Tensor, weight: Tensor, u: Tensor, v: Tensor,
                      sigma: float, bias: "Tensor | None" = None) -> Tensor:
    """A whole population's linear, against one shared weight.

    ``y_i = x_i W^T + sigma * (x_i v_i^T) u_i`` — member ``i``'s output
    under the perturbation ``sigma * u_i^T v_i``, computed **without
    materialising a perturbed weight**. That identity is the point of a
    low-rank perturbation: the population costs two small products per
    member on top of one shared matmul, instead of N copies of W.

    ``x`` is ``(N, S, in)``, ``weight`` ``(out, in)``, ``u`` ``(N, r, out)``
    and ``v`` ``(N, r, in)``; the result is ``(N, S, out)``.
    """
    from .. import bmm
    from ..nn.functional import linear

    if x.dim() != 3:
        raise ValueError(f"population_linear expects (N, S, in), got {list(x.shape)}")
    members = x.shape[0]
    if u.shape[0] != members or v.shape[0] != members:
        raise ValueError(
            f"u and v must have one entry per member: got {u.shape[0]}, {v.shape[0]} "
            f"for {members} members"
        )
    base = linear(x, weight, bias)                      # (N, S, out), shared W
    xv = bmm(x, v.transpose(1, 2))                      # (N, S, r)
    return base + bmm(xv, u) * sigma                    # (N, S, out)


def population_lora(x: Tensor, weight: Tensor, a: Tensor, b: Tensor, scaling: float,
                    noise: dict, sigma: float, bias: "Tensor | None" = None) -> Tensor:
    """A whole population's LoRA layer, against one shared frozen base.

    ``y_i = W x + scaling * B_i (A_i x)`` where both factors carry member
    ``i``'s perturbation. Expanding that product would give cross terms —
    it is not a rank-1 correction to a single matmul, because *two*
    matrices move — so it is not expanded: the perturbed ``A`` is applied
    by :func:`population_linear`, and the perturbed ``B`` is applied to its
    result by :func:`population_linear` again. Exact, and neither factor is
    ever materialised.

    ``noise`` carries ``u_a``/``v_a`` and ``u_b``/``v_b``, each with one
    entry per member. ``x`` is ``(N, S, in)``; the result is
    ``(N, S, out)``.

    The base is one matmul for the whole population, which is the point:
    at r=64 over a 2048-wide layer the two low-rank products are 6% of the
    base's work, so a population of 32 costs about twice one forward, not
    32 times it.
    """
    from ..nn.functional import linear

    hidden = population_linear(x, a, noise["u_a"], noise["v_a"], sigma)      # (N, S, r)
    delta = population_linear(hidden, b, noise["u_b"], noise["v_b"], sigma)  # (N, S, out)
    return linear(x, weight, bias) + delta * scaling
