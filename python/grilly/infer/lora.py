"""LoRA factors over a frozen base, kept separate rather than merged.

    y = W x  +  (alpha / r) * B (A x)

Merging — folding ``B A`` into ``W`` — is the right thing when an adapter
is finished and the base is float. It is the wrong thing here, twice over:

- **the base is int8.** Merging would mean dequantizing a 2.6B weight,
  adding, and requantizing, which costs the memory the quantization was
  for and loses the adapter in the rounding;
- **the factors are what moves.** Evolution strategies perturb ``A`` and
  ``B`` and score the result (``grilly.learning.es``); a merged adapter has
  nothing left to perturb, and re-merging every candidate would be a full
  weight rewrite per member.

So :class:`LoRALinear` keeps three tensors and spends two small matmuls a
call. At r=64 against a 2048-wide layer that is 6% of the base's work and
1.6% of its bytes.

**It adopts the base's parameters rather than owning a sub-module**, so a
wrapped layer still answers to ``...w1.weight`` and additionally to
``...w1.lora_A.weight``. Both the base checkpoint and a peft adapter
therefore load by their own names, with no remapping on either side —
peft's own layout renames the base to ``base_layer.weight``, which would
have meant remapping one of the two.

**It is not fusable.** The Llama-family attention concatenates q, k and v
into one int8 weight when it can; doing that to a wrapped module would
read the base and quietly drop the adapter. ``fusable = False`` is what
stops it, and `_fused_weight` honours it.
"""

from __future__ import annotations

from contextlib import contextmanager

from .. import Tensor, no_grad
from .. import nn
from ..nn import functional as F

__all__ = ["LoRALinear", "apply_lora_", "lora_parameters", "population"]


class LoRALinear(nn.Module):
    """A linear layer with a low-rank adapter beside its frozen weight."""

    #: the attention fusion concatenates base weights; that would lose this
    fusable = False

    def __init__(self, base: nn.Module, rank: int, alpha: float) -> None:
        super().__init__()
        out_features, in_features = base.weight.shape
        self.in_features, self.out_features, self.rank = in_features, out_features, int(rank)
        self.scaling = float(alpha) / float(rank)
        # adopt, do not nest: the base keeps its checkpoint name
        self.weight = base.weight
        if base.bias is not None:
            self.bias = base.bias
        else:
            self.bias = None
        self.lora_A = nn.Linear(in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, out_features, bias=False)
        with no_grad():                      # peft's init: B zero, so y is the base until trained
            self.lora_B.weight.zero_()
        #: set by `population`: every member's perturbation, batched
        self._population = None
        self._sigma = 0.0

    def forward(self, x: Tensor) -> Tensor:
        if self._population is not None:
            from ..learning.es import population_lora

            return population_lora(x, self.weight, self.lora_A.weight, self.lora_B.weight,
                                   self.scaling, self._population, self._sigma, self.bias)
        return F.linear(x, self.weight, self.bias) + self.lora_B(self.lora_A(x)) * self.scaling

    def factors(self) -> list:
        """``[A, B]`` — the tensors an optimizer moves."""
        return [self.lora_A.weight, self.lora_B.weight]

    def select_rows(self, index):
        """Keep the population noise of batch rows ``index`` (an int64
        tensor) only, for a batch that has dropped rows — ``generate`` drops
        streams as they finish. Returns the undo: the noise belongs to the
        ``population`` block, not to one generation, and must be whole again
        for the next. Outside a population there is nothing per row."""
        before = self._population
        if before is None:
            return lambda: None
        self._population = {name: noise.index_select(0, index) for name, noise in before.items()}

        def undo() -> None:
            self._population = before

        return undo

    def __repr__(self) -> str:
        return (f"LoRALinear(in_features={self.in_features}, out_features={self.out_features}, "
                f"r={self.rank}, scaling={self.scaling})")


def apply_lora_(model: nn.Module, rank: int, alpha: float, targets) -> dict:
    """Wrap every module named by ``targets`` in a :class:`LoRALinear`.

    ``targets`` is an iterable of dotted parameter-free module paths
    (``model.layers.0.feed_forward.w1``). Returns the wrappers by path, so
    a loader can put the factors straight into them.

    The base weights are untouched and stay frozen; nothing is merged.
    """
    wrapped = {}
    for path in targets:
        parent, _, leaf = path.rpartition(".")
        owner = model.get_submodule(parent) if parent else model
        base = getattr(owner, leaf, None)
        if base is None:
            raise KeyError(f"{path} is not a module of this model")
        if isinstance(base, LoRALinear):
            raise RuntimeError(f"{path} already carries a LoRA")
        if not hasattr(base, "weight"):
            raise TypeError(f"{path} has no weight to adapt ({type(base).__name__})")
        adapter = LoRALinear(base, rank, alpha)
        setattr(owner, leaf, adapter)
        wrapped[path] = adapter
    return wrapped


def lora_parameters(model: nn.Module) -> list:
    """Every LoRA factor in ``model``, in module order.

    What to hand an optimizer: ``EvolutionStrategy(lora_parameters(model),
    ...)`` tunes the adapter and nothing else, because these are the only
    tensors it is given.
    """
    factors = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            factors.extend(module.factors())
    return factors


@contextmanager
def population(model: nn.Module, es, repeat: int = 1):
    """Run a whole ES population as one batch, for the duration.

    Inside the block every :class:`LoRALinear` applies **each batch row's
    own** perturbation against the shared weight, so a forward over a batch
    of ``population * repeat`` rows scores the whole population at once.
    Outside it the model is its unperturbed self again.

    This is the difference between an ES iteration costing N forward passes
    and costing one. On CubbyLLM's emitter, scoring 32 members over 16
    prompts one at a time is 512 serial generations — about an hour; as a
    batch it is the same 512 sequences decoded together.

    ``repeat`` is how many consecutive rows each member owns, for a batch
    laid out as (member, prompt) pairs. The caller's inputs must use the
    same order: member 0's prompts first, then member 1's.

    **Prompts of different lengths go in left-padded.** ``generate``
    takes a left-padded ``attention_mask``, so ``repeat`` prompts per
    member need not share a length, and it drops rows as they finish —
    each :class:`LoRALinear` drops the same rows of its noise
    (``select_rows``) and gets them back when the generation ends. A
    population of one prompt, ``repeat=1``, needs neither.

    **What bounds ``population * repeat`` is memory, not the model.** The
    noise is materialized per row: one ``(rank, out)`` and one
    ``(rank, in)`` vector for every row of every adapted layer. On
    CubbyLLM's emitter that is about 1.3M values a row, so 32 members x 16
    prompts is ~2.6 GB of noise — on top of a 3.7 GB model and a KV cache
    for the same 512 streams, which does not fit 12 GB. 128 rows is ~650 MB
    and is also where that card's decode throughput peaks, so 32 members
    times 4 prompts, looped over the prompt set, is the shape to reach for.

    The factors themselves are untouched — the perturbation lives in the
    batched noise, never in the weights — so nothing has to be restored
    and an exception inside the block cannot leave the model perturbed.
    """
    layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    if not layers:
        raise RuntimeError("this model carries no LoRA to run a population over")
    index = {id(t): i for i, t in enumerate(es.params)}
    try:
        for layer in layers:
            a, b = layer.factors()
            noise = {}
            for name, tensor in (("a", a), ("b", b)):
                slot = es.moved.index(index[id(tensor)])
                noise[f"u_{name}"], noise[f"v_{name}"] = es.population_noise(slot, repeat)
            layer._population, layer._sigma = noise, es.sigma
        yield layers
    finally:
        for layer in layers:
            layer._population, layer._sigma = None, 0.0
