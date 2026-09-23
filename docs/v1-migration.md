# Migrating from grilly v1

grilly2 replaces grilly v1 **in place**: the same distribution name, the
same import name. Code written against v1 keeps importing `grilly` and
keeps working; what changes is what that name now resolves to.

Every v1 name falls into one of two groups, and the difference decides
everything about how it is treated:

| | what it is | what happens to it |
|---|---|---|
| **A — a domain library** | something the torch API has no name for: the VSA algebras, spiking neurons, plasticity rules, place and time cells, capsules, CVL | **ported.** Rewritten on grilly2's kernels, under its own name. The v1 path keeps working as a deprecated alias with v1's exact signature — numpy in and numpy out where v1 took numpy — and one `DeprecationWarning` naming the new path |
| **B — v1's parallel framework** | v1's own autograd, its own `nn` layers and optimizers, `VulkanTensor`, `torch_api`, and the raw bridge ops beneath them | **replaced, not reimplemented.** The torch API *is* the replacement; that is the entire point of the rewrite (CLAUDE.md §1). The v1 name raises `NotImplementedError` naming what to use instead |

A **B** name raises `NotImplementedError`, never `AttributeError`, and
never returns `None`. That is deliberate: `AttributeError` is what
`hasattr` swallows, and a capability check that quietly answers "no" is
exactly how v1's own `_detect_backend` ended up reporting a GPU tier it
never used (CLAUDE.md §8.3, §8.4). A migration that fails loudly is a
migration; one that falls through silently is a regression nobody notices.

## Who actually depends on v1

The order of work follows the consumers, not the size of the surface. From
a static scan of every checkout beside this one:

| repo | distinct v1 imports | files | what it needs |
|---|---:|---:|---|
| `grilly-experimental` | 423 | 212 | all of it — v1's own workbench |
| `cubemind` | 60 | 44 | `_bridge`, v1 `nn`/`optim`/autograd, `VulkanTensor`, `BlockCodeOps`, `hamming_topk`, place/time cells |
| `grillcheese` | 38 | 35 | `_bridge`, `BufferMixin` |
| `cubby-lm` (+ backup) | 13 | 33 | `_bridge` |
| `elephant` | 11 | 3 | `_bridge` |
| `grilly-next` | 9 | 3 | `_bridge` |
| `grilly-rust`, `grilly-cpp`, `optimum-grilly`, `neurips`, `neurips2026_submission`, `cubemind-v1-archive` | 1–6 | ≤8 | `VulkanCompute`, `_bridge` |
| `mowm` | 3 | 2 | `_bridge`, `CapsuleProject`, `VulkanSentenceTransformer` |
| **`CubbyLLM`** | **2** | **1** | `_bridge` (a tier check) and `BlockCodeOps` |
| `cubbyverse` | 2 | 1 | `CVLValue`, `RandomFourierFeatures` |

CubbyLLM is the smallest surface and the one that matters: its
`cubbyllm/ops/vsa.py` is the single seam onto the VSA algebra, and it is
the acceptance test for the port — **unedited**, it must detect the fast
tier and get the right numbers. It does.

## Status

### A — ported, or to port

| v1 name | grilly2 | status |
|---|---|---|
| `experimental.vsa.block_ops.BlockCodeOps` | `grilly.vsa` | **done.** One ulp of v1's numpy against the same one-hot inputs, and 9.7x/10.6x its throughput at batch 64/512 |
| `_bridge.blockcode_bind` / `_unbind` / `_similarity` | `grilly.vsa` | **done.** Bit-exact with v1's shader on one-hot codes, and unlike it, correct for continuous ones too |
| `_bridge.hdc_*_packed`, `_bridge.hamming_topk` | `grilly.vsa.packed` | **done.** v1's were numpy only ("GPU path not yet wired"); these run on the card |
| `utils.stable_hash` | `grilly.vsa.hashing` | **done.** Digests match v1's, which they must: they seed hypervectors |
| `experimental.vsa.ops.BinaryOps` | `grilly.vsa.bipolar` | **done.** No kernels: a product, a sum, a sign and a dot are ops the framework already had |
| `experimental.vsa.ops.HolographicOps` | `grilly.vsa.holographic` | **done.** No kernel either — HRR binding is the block-code kernel at `k = 1`. Direct, so O(d²): a clear win below 4096 and a wash at 10240, with the measured table in the module |
| `_bridge.vsa_bind` / `_unbind` / `_bundle` / `_bitpack`, `_bridge.blake3_role` | `grilly.vsa.bipolar` | **done**, and they work for the first time — v1's wrappers passed a device handle the compiled functions do not take, so every one returned `None` |
| `experimental.vsa.resonator.ResonatorNetwork` | `grilly.vsa.resonator` | **done**, faithfully, which means it inherits v1's weakness: 2/20 at 32 items a factor. The restart count v1 hard-coded is now a parameter (8/20 at 100 restarts). The published dynamics would fix it and are not implemented |
| `nn.snn_neurons` (`IFNode`, `LIFNode`, `ParametricLIFNode`), `nn.snn_surrogate` (`ATan`, `Sigmoid`, `FastSigmoid`), `functional.snn` (`reset_net`, `multi_step_forward`, `set_step_mode`) | `grilly.snn`, over the `grilly-snn` crate | **done.** A rewrite: v1's neurons carried hand-written `backward` methods over its own autograd, so a gradient reached one layer. These are `grilly.nn.Module`s and a test trains one through 12 timesteps, chance to 100%. The timestep is one fused kernel — 12 dispatches to 1, 12–16x — with the composed path kept for a learned `tau` and tested bit-exact against it |
| `nn.snn_synapses`, `nn.snn_ann2snn`, `nn.snn_attention`, `nn.snn_monitor`, `nn.snn_normalization`, `nn.snn_containers`, `_bridge.lif_step` / `gif_neuron_step` / `snn_node_*` | `grilly.snn` | to do — the neurons are there, the surrounding machinery is not |
| `_bridge.hebbian_learning`, `_bridge.oja_learning` | `grilly.snn.plasticity` | **done.** Hebb's arithmetic was right and is unchanged. Oja **never ran** in v1 — `oja_learning` is absent from its compiled extension, so the wrapper returned `None` every time |
| `_bridge.stdp_learning`, `nn.snn.HebbianLayer` / `STDPLayer` | `grilly.snn.plasticity.STDPLearner` | **B for the STDP one.** v1's rule was not STDP: its shader computed `ltp += pre_t * post_t` and `ltd += post_t * pre_t`, the same product, so reversing the spike order could not change the sign. It raises rather than reproducing the bug or quietly changing the numbers |
| `functional.cells` (`place_cell`, `time_cell`, `theta_gamma_encoding`) | `grilly.snn.cells` | **done.** No kernels: a squared distance, an exponential and an outer product |
| `functional.bridge` (`continuous_to_spikes`, `spikes_to_continuous`) | `grilly.snn.encoding` | **done.** v1 accepted an `encoding_type` of temporal or phase and rate-coded anyway; those now raise |
| `nn.cells` (`PlaceCell`, `TimeCell`, `ThetaGammaEncoder` as modules) | `grilly.snn.cells` | to do — the functions are there, the `nn.Module` wrappers are not, and nothing live imports them |
| `nn.capsule.DentateGyrus` | `grilly.snn.DentateGyrus` | **done.** The one layer in the capsule group that was not a Linear: a sparse projection then k-winners-take-all |
| `nn.capsule.CapsuleProject`, `.SemanticEncoder` | `grilly.nn.Linear` | **B, not A** — see below |
| `experimental.cognitive.capsule.cosine_similarity`, `.batch_cosine_similarity` | `grilly.nn.functional.cosine_similarity` | **B, not A** |
| `nn.capsule_embedding`, `backend.capsule_transformer`, the rest of `experimental.cognitive` | model code | not ported — build it in the repo that uses it |
| `backend.cvl` (`CVLValue`, `RandomFourierFeatures`) | `grilly.cvl` | **done.** The feature map is the only expensive part and it is now a matmul |
| `functional.learning` (EWC, Fisher, natural gradient, NLMS, whitening) | `grilly.learning` | **done.** Element-wise arithmetic and a dot product; no kernels, and v1's epsilons kept so a saved Fisher or NLMS weight behaves the same |
| `_bridge.moqe_*`, `_bridge.ternary_gemm`, `_bridge.pack_ternary` | `grilly.infer.quantization.Int8WeightOnlyConfig` | **B, not A** — measured: the decode matmul is already at 87% of this card's memory bandwidth, so quantizing activations too reads the same bytes and cannot pay (docs/inference.md) |
| `nn.vsa_lm`, `_bridge.vsa_lm_*` | `grilly.vsa` | to do |
| `backend.faiss`, `backend.hilbert`, `functional.faiss` | undecided | to do |

### B — replaced by the torch API

None of these are reimplemented. Each raises, naming the replacement.

| v1 | use instead |
|---|---|
| `nn.autograd.Variable`, `Function`, `GradFn`, and the ~90 functions beside them | `grilly.Tensor` with `requires_grad`, and `.backward()` |
| `nn.linear.Linear`, `nn.conv.Conv2d`, `nn.normalization*`, `nn.pooling`, `nn.dropout`, `nn.embedding`, `nn.activations`, `nn.loss`, `nn.containers`, `nn.rnn`, `nn.attention`, `nn.transformer`, `nn.init`, `nn.module`, `nn.parameter` | `grilly.nn` — the torch names, torch's argument order, torch's semantics |
| `optim.{Adam,AdamW,SGD,lr_scheduler,base.Optimizer}` | `grilly.optim` |
| `optim.{HypergradientAdamW,AutoHypergradientAdamW,NaturalGradient,NLMS,AffectAdam}` | no torch equivalent — **A, not B.** Listed above under "to do" |
| `utils.tensor_conversion.VulkanTensor` | `grilly.Tensor` |
| `torch_api` (all 8 modules) | grilly itself — the whole package is the torch API now |
| `backend.Vulkan*` (`VulkanCompute`, `VulkanCore`, `VulkanLearning`, `VulkanAttention`, `VulkanConv`, …) | grilly itself; there is no backend object to hold |
| `backend._bridge` ops that torch already names: `linear`, `gelu`, `relu`, `silu`, `tanh`, `softmax`, `layernorm`, `rmsnorm`, `conv2d`, `maxpool2d`, `attention_*`, `flash_attention2`, `rope`, `cross_entropy_*`, `adam_update`, `adamw_update`, `tensor_*`, `kv_cache_*` (~50 names) | `grilly.nn`, `grilly.nn.functional`, `grilly.optim`, `grilly.infer.cache` |
| `backend.jit` | `grilly.compile` |
| `backend.amp` | not available — grilly is float32 only so far, and will say so rather than pretend |
| `backend.buffer_pool`, `backend.shader_registry`, `backend.vma_wrapper` | grilly's own allocator and dispatch spine. No op names a `.spv` file (CLAUDE.md §5) |
| `nn.lora`, `nn.multimodal`, `nn.routing`, `nn.hippocampal`, `nn.selective_momentum`, `nn.projection_heads` | ordinary `grilly.nn` modules — these are model code, not framework, and belong in the repo that uses them |

### Not migrated at all

`benchmarks`, `examples`, `tutorials`, `scripts`, `tests`,
`experimental/resident_train`, `experimental/snn_event_driven`,
`grilly_datasets`, `experimental_datasets`, `tokenizer_impl` — v1's own
scaffolding. grilly2 depends on Hugging Face `tokenizers` rather than
carrying one (CLAUDE.md §11), and datasets are not a tensor framework's
job.

## The plan for the rest, and what needs deciding

VSA is done. The order for what is left follows the consumers above, but
three of the groups have a real choice in them, and those are for the owner:

`grilly.vsa` is complete: four algebras — block codes, bipolar,
packed and holographic — plus the resonator and the deterministic hashing
that seeds them. The bipolar question ("its own kernels, or a layer over
packed?") was settled by measurement, and the answer was neither: it needs
no kernels at all, because multiplication, a sum, a sign and a dot product
are ops the framework already has. `pack` and `unpack` bridge to the packed
representation exactly, and the two rank a codebook identically.

What is left:

1. **SNN, the rest of it.** The neurons and their surrogate gradients are
   done, and the rewrite turned out to be the easy part: the surrogate
   needs no custom autograd node, because
   `heaviside(x) + (g(x).detach() * x - (g(x).detach() * x).detach())` has
   the step as its value and `g` as its derivative, which composes with
   ordinary autograd. What is left is the surrounding machinery —
   synapses, the ANN-to-SNN converter, the spiking attention variants,
   monitors — and none of it has a live consumer, so it waits for one.

2. **Capsules** turned out to be the case the classification exists to
   catch, and the answer was not "port it". `CapsuleProject` and
   `SemanticEncoder` have identical `__init__` bodies once their default
   dimensions are normalised, and both return exactly
   `x @ self.weight.T + self.bias` — their own source says "Use linear
   backend for now (capsule-project is similar to linear)". They are
   `nn.Linear` under two names that are not about capsules.
   `cosine_similarity` and `batch_cosine_similarity` are
   `F.cosine_similarity`. Only `DentateGyrus` — a sparse projection then
   k-winners-take-all — was a layer torch has no name for, and that one is
   ported.

   Worth stating as a rule, because it is the trap in both directions: a
   domain *name* does not make a domain *library*. Porting those two layers
   would have put a second spelling of `nn.Linear` in the framework, which
   is v1's original mistake with the labels swapped.

The immediate next piece, if the answer is "keep going in consumer order",
is **cubemind**: 60 imports across 44 files, and the only live consumer
that needs v1's `nn`, `optim` and autograd. Those are all group B, so the
work there is not porting — it is migrating cubemind onto the torch API,
which is a change in *that* repo, not this one.
