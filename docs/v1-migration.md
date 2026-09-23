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
| `experimental.vsa.ops.BinaryOps` / `HolographicOps`, `_bridge.vsa_*` | `grilly.vsa` (a third algebra: bipolar int8) | to do |
| `experimental.vsa.resonator.ResonatorNetwork` | `grilly.vsa` | to do |
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

1. **The other two VSA algebras** — bipolar int8 (`vsa_bind`, `vsa_bundle`,
   `vsa_unbind`, `vsa_bitpack`) and the resonator network. Small, and they
   complete `grilly.vsa`. `blockcode_*` and `vsa_*` are **different
   algebras** and stay that way (CLAUDE.md §8.3); the only question is
   whether the bipolar one earns its own kernels or is a thin layer over
   `grilly.vsa.packed`, which it may well be — bipolar ±1 packed to bits is
   the same thing wearing a different dtype. That is a measurement, not a
   judgement call.

2. **SNN** — the largest group by far: 10 `nn.snn*` modules, ~60 classes,
   plus the plasticity rules and the bridge's neuron steps. It is a
   spiking-neural-network library that happens to sit on a tensor
   framework, and it is the one place where "port it" hides a design
   question: v1's SNN modules are built on v1's `Variable`, so porting them
   means rewriting them as `grilly.nn.Module`s with real autograd through
   surrogate gradients. That is a rewrite, not a translation, and it wants
   its own milestone. **Nobody in the live consumer list imports it** —
   only `grilly-experimental` and `cubemind` do — so it is not urgent, and
   the question is whether it ships at all or waits for a user.

3. **Capsules, CVL, and the learning rules** (EWC / Fisher / natural
   gradient / NLMS / whitening) — each has exactly one consumer, and two of
   those consumers are repos that are not being touched. The question is
   whether these are *libraries* (port them, with tests and an API) or
   *model code that lived in the wrong place* (move them into the repo that
   uses them, which is `cubbyverse` for CVL and `mowm` for capsules).
   Getting this wrong in the direction of "port everything" is how v1's
   surface reached 230 exported symbols.

The immediate next piece, if the answer is "keep going in consumer order",
is **cubemind**: 60 imports across 44 files, and the only live consumer
that needs v1's `nn`, `optim` and autograd. Those are all group B, so the
work there is not porting — it is migrating cubemind onto the torch API,
which is a change in *that* repo, not this one.
