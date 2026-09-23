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
| `nn.snn*` (10 modules: neurons, surrogates, synapses, containers, normalization, attention, ann2snn, monitor), `_bridge.lif_step` / `gif_neuron_step` / `snn_node_*` | `grilly.snn` | to do |
| `_bridge.stdp_learning` / `hebbian_learning` / `oja_learning`, `nn.snn.HebbianLayer` / `STDPLayer` | `grilly.snn` (plasticity) | to do |
| `functional.cells`, `nn.cells` (`PlaceCell`, `TimeCell`, `ThetaGammaEncoder`) | `grilly.snn` or its own module | to do |
| `nn.capsule`, `nn.capsule_embedding`, `backend.capsule_transformer` | undecided (see below) | to do |
| `backend.cvl` (`CVLValue`, `RandomFourierFeatures`) | undecided | to do |
| `functional.learning` (EWC, Fisher, natural gradient, NLMS, whitening) | undecided | to do |
| `_bridge.moqe_*`, `_bridge.ternary_gemm`, `_bridge.pack_ternary` | `grilly.infer` quantization | to do |
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

1. **SNN** — the largest group by far: 10 `nn.snn*` modules, ~60 classes,
   plus the plasticity rules and the bridge's neuron steps. It is a
   spiking-neural-network library that happens to sit on a tensor
   framework, and it is the one place where "port it" hides a design
   question: v1's SNN modules are built on v1's `Variable`, so porting them
   means rewriting them as `grilly.nn.Module`s with real autograd through
   surrogate gradients. That is a rewrite, not a translation, and it wants
   its own milestone. **Nobody in the live consumer list imports it** —
   only `grilly-experimental` and `cubemind` do — so it is not urgent, and
   the question is whether it ships at all or waits for a user.

2. **Capsules, CVL, and the learning rules** (EWC / Fisher / natural
   gradient / NLMS / whitening) — each has exactly one consumer, and two of
   those consumers are repos that are not being touched. **Decided: port
   them into grilly2**, as libraries with an API and tests, rather than
   pushing them into the one repo that happens to use each. The risk
   accepted is surface; the risk avoided is v1's other failure — a general
   capability reachable only through one consumer's code (CLAUDE.md §11).

The immediate next piece, if the answer is "keep going in consumer order",
is **cubemind**: 60 imports across 44 files, and the only live consumer
that needs v1's `nn`, `optim` and autograd. Those are all group B, so the
work there is not porting — it is migrating cubemind onto the torch API,
which is a change in *that* repo, not this one.
