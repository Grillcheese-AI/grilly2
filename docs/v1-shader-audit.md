# grilly v1 shader audit

2026-09-23. Scope: every shader source in grilly v1's `shaders/` — 257 top-level `.glsl`, 2 `.comp`, 9 in `experimental/`, 8 fragments in `snippets/` (276 files). Six reviewers each read one domain in full and checked every file for correctness, duplication and value. I re-checked every high-severity claim against the code before it went in here: two were downgraded, one was reframed, and one apparent contradiction with an earlier measurement was resolved. Usage comes from a name scan of grilly v1, grilly2, CubbyLLM, cubemind, mowm and cubbyverse. The per-shader table is the appendix, and the full findings (every medium and low bug with line numbers) are in `v1-shader-audit.json` beside this file.

**How to read "named in":** a shader counts as used when its name appears in code, comments included. That's an upper bound. `fft-bind`, for instance, is named only in a header comment (`cpp/include/grilly/cubemind/block_ops.h`), and no `.cpp` dispatches it. Where it matters below, I checked the dispatch itself.

## The short version

| | count |
|---|---:|
| working | 239 |
| broken (wrong results for normal inputs) | 25 |
| partial (a path unimplemented or a placeholder) | 12 |
| named by no code at all (excluding the 8 fragments) | 58 |
| missing SPIR-V | 19 (8 fragments by design; 2 `.comp`; all 9 in `experimental/`) |
| grilly2 verdict: superseded / port / drop | 162 / 29 / 86 |

- **CubbyLLM calls none of these directly.** Its only grilly seam is `cubbyllm/ops/vsa.py` → block codes, now `grilly.vsa` in grilly2. Nothing below changes a CubbyLLM result.
- **grilly v1's own users are exposed.** 26 broken or partial shaders are named in v1 production code. The ones I confirmed are actually dispatched: `flash-attention2` (v1's attention: `backend/attention.py`, `cpp/src/ops/attention.cpp`, the C++ autograd), `lora-forward/backward`, `loss-cross-entropy` (with label smoothing), `fused-mlp-gelu` / `fused-layernorm-linear`, and `moe-layer-grad-weight`.
- **grilly2 already has most of the worth-keeping domain logic** at the API level (`grilly.vsa`, `grilly.snn`, `grilly.learning`, `grilly.cvl`). Its agent had already caught v1's STDP bug on the way. What's genuinely missing is short (see *Worth porting*).
- **`experimental/` has no SPIR-V**, but `backend/experimental/vsa.py` dispatches `vsa-argmax-rows` from it, so that v1 path can't run as shipped.

## Verified high-severity bugs

Each one was re-checked against the source. "Called by" is the verified dispatch where I checked it, else the name scan.

| shader | line | what's wrong | called by |
|---|---|---|---|
| `flash-attention2` | 154, 167 | Every thread on a query row does an unsynchronised read-modify-write of `running_max` / `running_sum` / `output_accum`. Each weight is divided by the partial sum, then pass 2 divides by the final sum again. | v1 attention (Python + C++, dispatched) |
| `flash-attention2-rope` | 210, 220 | Same race and same double normalisation. Its RoPE is interleaved pairs, while `rope.glsl` and `g2-rope` use half-rotation. | v1 |
| `flash-attention3` | 103 | The Q·K dot is a `subgroupAdd` over a 64-thread workgroup. Correct only on 64-wide subgroups (AMD's default); partial scores on 32-wide (NVIDIA). | nothing |
| `chunked-sw-attention-backward` | 45 | Race on `sm_reduce[0]`: the next `dot_product()` writes it before every thread has read the last result. | v1 |
| `embedding-attention` | 145, 220 | The softmax pass treats a 16×16 workgroup as one row, so 15 of 16 rows get row 0's max. The output pass writes into the buffer other threads are still reading V from. | v1 (`functional/embedding.py`) |
| `embedding-pool-mask` | 49 | `mask_sums[b]` is written by one thread of one workgroup and read by other workgroups with no synchronisation. There's also an early return before `barrier()`. | v1 |
| `moe-layer-grad-weight` | 57 | `tileG` is loaded with the thread's own column, so it holds a diagonal; every dW is wrong. | v1 C++ MoE backward (dispatched) |
| `lora-forward` | 72, 112 | Early `return` before the tile loop's `barrier()`s. Tiles stay unfilled unless rank and batch are multiples of 16, so rank 4 or 8, or batch 1, gives garbage. | v1 LoRA (dispatched) |
| `lora-backward` | 98 | Same early-return pattern; `grad_A` is wrong. | v1 LoRA |
| `fnn-layernorm-backward` | 124 | The input gradient multiplies the mean terms by the current feature's gamma: `std_inv·g_i·(dy_i − (Σ + x̂_i·Σ')/N)`. Right only when gamma is constant. | v1 autograd |
| `fused-mlp-gelu`, `fused-layernorm-linear` | — | Reframed after checking `cpp/src/ops/fused.cpp`: the entry points take `dIn/dHidden/dOut` but never specialise the shader. It always computes the 768/3072/768 (SigLIP2) shape, so any other model's dimensions give wrong results. | v1 C++ + `_bridge.py` (dispatched) |
| `loss-cross-entropy` | 105 | Label smoothing adds `ε·log V` (a constant) instead of `ε·mean(−log p)`. For logits [0,0,10], target 2, ε=0.1 it returns 0.110; torch gives 0.667. Correct when smoothing is 0. | v1 (dispatched) |
| `contrastive-gradient` | 94 | The gradient is the dot-product triplet's, not the cosine triplet `contrastive-loss` computes. Wrong even for unit vectors. | v1 |
| `activation-softplus` | 35 | The cutoff tests `βx > 20/β`, so it's correct only at β=1. β=10, x=0.21 returns 0.21 (should be 0.2216); β=0.1, x=1000 returns inf. | v1 |
| `mf-softmax` | 61 | `relu(x − rowmax)` is always 0, so every row comes out uniform. | v1 |
| `stdp-learning` | 99 | The LTP and LTD terms are identical, so spike timing is ignored: it's order-blind Hebbian at rate `lr_p − lr_d`. | v1 (grilly2's `snn.plasticity` already replaced it) |
| `synapsis-stdp-update` | 52 | Potentiation only, from the product of batch means, so weights ratchet up to `weight_max`. | v1 |
| `eggroll-update` | 59 | Merit rises whenever `abs(avg_delta) > 1e-8`, true for 99.96% of weights at σ=0.01 (simulated). Merit compounds as a global step-size multiplier: ×2e4 after 1,000 steps at 1.01. | v1 C++ (dispatched) |
| `fft-bind` | 54, 115 | The shared array goes into `compute_fft` as an `inout` parameter, so each thread transforms a private copy and they race to write back; plus an unsynchronised bit-reversal write. Garbage for any input. | never dispatched (header comment only) |
| `vsa-argmax` | 29 | The block index is `gl_LocalInvocationID.z` with `local_size_z = 8`, so only blocks 0–7 are processed. At k=80, 72 blocks are never written. | nothing |
| `hdc-overlap-metrics` | 60 | Two 32-wide subgroups both write the workgroup's result. Correct only on 64-wide subgroups. | v1 |
| `hdc-bundle-subgroup` | 36 | Counts only the first `gl_SubgroupSize` input vectors; the rest are silently ignored. | nothing |
| `domain-classifier` | 72 | The pooled path uses only the first 256 feature dims. | v1 (CubbyLLM names it in a docstring only) |
| `batch-similarity` | 52 | Early return before `barrier()`: in-range outputs read unfilled tiles whenever a batch size isn't a multiple of 16. | nothing |
| `experimental/vsa-fft-convolve` | 70 | The "inverse FFT" stage is a placeholder that copies the real part. | v1 (and it has no SPIR-V) |
| `zero_collision_hash.comp` | 139, 121 | Doesn't compile (int64 buffer indices), and can spin forever when `METADATA_COUNT=1`. | nothing |

**Downgraded after checking:**

- `memory-read` has no branch for its documented softmax pass, but v1's caller (`backend/memory.py`) runs softmax separately, so that branch is dead, not wrong.
- The races in `nlms-update` (`mu[0]`), `sanger-gha` (`W`), `whitening-transform` (running mean/var) and `distillation-loss` (a shared slot) are real, but each shifts results by a second-order or `1 − momentum` sized amount. They're medium, not high.

## Conflicts between shaders

These matter when two shaders get swapped for each other, e.g. when porting.

- **SwiGLU convention.** `fused-swiglu` / `fused-geglu` / `fused-reglu` apply the activation to the *up* projection. `swiglu-fused`, `g2-swiglu` and LLaMA-style checkpoints apply it to the *gate*. Same weights, different answer.
- **RoPE layout.** `flash-attention2-rope` uses interleaved pairs; `rope` and `g2-rope` use half-rotation.
- **GELU form.** v1's `activation-gelu` and its backward use the tanh approximation; grilly2's `UN_GELU` uses exact erf. Never pair v1's backward with grilly2's forward.
- **FFT sign.** `fft-butterfly`'s forward transform is the complex conjugate of numpy's. Magnitudes and convolution round-trips are unaffected.
- **Bit order.** `circular-shift` assumes the opposite in-word bit order from the repo's packers and `g2-hdc-permute`. It's self-consistent, but it isn't a roll of the vector.
- **Byte-identical copies already in grilly2:** `g2-relu`, `g2-relu-backward`, `g2-add-inplace`, `g2-linear`, `g2-linear-backward`, `g2-residual` = `activation-relu`, `activation-relu-backward`, `elementwise-add`, `fnn-linear`, `fnn-linear-backward`, `fnn-residual`.
- **Subsumed:** `loss-sampled-bce-fused` is `loss-nce-fused` with the correction off. `convd_col2im_noatomic` is the deterministic version of `convd_col2im`.

## Worth porting to grilly2

The reviewers marked 29 as "port". Most turned out to be covered already by grilly2's Python domain modules, so here is what's actually missing:

| shader(s) | what grilly2 gains | fix needed first | CubbyLLM use |
|---|---|---|---|
| `mingru-forward`, `mingru-backward` | A native linear-recurrence scan. grilly2's `g2-scan` only does cumsum/logcumsumexp, so today MinGRU runs as the two-`logcumsumexp` Heinsen form. | Take an initial state in and return the final state out (fixed at zero now). The math matches CubbyLLM's recurrence exactly. | Prefill and training of every MinGRU layer |
| `prefix-scan-causal` (+ backward) | The generic scan behind the above | — | Same |
| `loss-ce-fused` | Cross-entropy loss and gradient in one dispatch with prompt masking; grilly2 has no CE kernel. | — | Training/SFT the 128k head on grilly2 |
| `loss-policy-grad-fused` | GRPO/REINFORCE head gradient | Could become a per-row weight inside `loss-ce-fused` | v9 RLVR on the program adapter |
| `loss-nce-fused` | Sampled softmax-free head | — | Cheaper 128k-head training |
| `adamw-update` | One-dispatch AdamW, verified correct (bias correction and decoupled decay); grilly2 has no optimizer kernel. | — | Any on-card fine-tuning |
| `eggroll-generate`, `eggroll-update` | Gradient-free (ES) low-rank tuning | Merit rule, σ² scaling, seed striding, and the update rule (see the eggroll prompt) | Cubby tuning its own adapters against the VM |
| `gemm_tiled` | Handles partial tiles correctly and does ~16× more work per thread than `g2-bmm`'s one-output kernel. A better base for matmul. | — | Prefill |
| `ternary-gemm` | A correct BitNet-style ternary GEMM | — | Future ternary trunk |
| `gemm-coopmat-shared` + `cast-f32-f16` | fp16 cooperative-matrix GEMM; the only one in the tree | `gemm_coopmat` silently needs N % 64 | Available on the RX 6750 XT: its driver (AMD 26.8.1, Vulkan 1.4.315) exposes `VK_KHR_cooperative_matrix` with `cooperativeMatrix = true`. RDNA2 has no WMMA instructions, so the driver has to map it onto packed-fp16 math: expect up to ~2× over fp32 (plus half the bytes), not the matrix-core jump of RDNA3 or NVIDIA. Measure against `gemm_tiled` before building on it. |
| `sanger-gha` | Streaming PCA (GHA) | The `W` race | Codebook sharpening before SimHash |
| `spike-propagate-batch` | Event-driven sparse SNN propagation | — | SNN cortex |

Already covered at the API level, so port the v1 kernel only if a measurement shows the grilly2 op is slow: `nlms-update`, `oja-learning`, `hebbian-learning`, `fisher-*`, `whitening-*` (`grilly.learning`, `grilly.snn.plasticity`); `vsa-binarize-pack`, `hamming-top1` (`grilly.vsa.bipolar.pack`, `grilly.vsa.packed.hamming/topk`); `vsa-argmax` (`grilly.vsa.discretize`); `fft-bitrev/butterfly` (holographic binding exists as `grilly.vsa.holographic`).

## What to delete

- **58 shaders are named by no code.** 35 are superseded by grilly2, 21 the reviewers marked drop, and 2 were marked port (`sanger-gha`, and `vsa-argmax`, which is covered as above). Those 56 are safe to delete from any copy that gets carried forward. The appendix filters by "named in: —".
- **The broken-and-dispatched shaders** can't be patched in place, because v1 stays untouched. The options are a KNOWN_ISSUES note in v1 pointing users to grilly2, or deleting them. Your call.
- **Don't carry forward:** `experimental/` (no SPIR-V, placeholders), `zero_collision_hash.comp` (doesn't compile), and the `fused-*glu` family (wrong convention, superseded by `g2-swiglu`).

## Appendix — every shader

| shader | status | named in | grilly2 | first finding |
|---|---|---|---|---|
| `activation-gcu.glsl` | working | v1 | drop |  |
| `activation-gcu-backward.glsl` | working | v1 | drop |  |
| `activation-gelu.glsl` | working | v1 | superseded |  |
| `activation-gelu-backward.glsl` | working | v1 | superseded |  |
| `activation-hardshrink.glsl` | working | — | superseded |  |
| `activation-hardsigmoid.glsl` | working | — | superseded |  |
| `activation-hardswish.glsl` | working | — | superseded |  |
| `activation-relu.glsl` | working | v1, v1 tests | superseded |  |
| `activation-relu-backward.glsl` | working | v1 misc, v1 | superseded |  |
| `activation-roswish.glsl` | working | v1 | drop |  |
| `activation-roswish-backward.glsl` | working | v1 | drop |  |
| `activation-silu.glsl` | working | v1 | superseded |  |
| `activation-silu-backward.glsl` | working | v1 | superseded |  |
| `activation-softmax.glsl` | working | v1 | superseded | medium: L45 The row max is initialised to -1e10 instead of -inf, so rows whose values are all below -1e10 produce NaN. |
| `activation-softmax-backward.glsl` | working | v1 | superseded | medium: L38 Performance cliff: one invocation serially walks each whole row twice, and neighbouring invocations read addre |
| `activation-softplus.glsl` | broken | v1 | superseded | high: L35 The cutoff compares beta*x against 20/beta (i.e. beta^2*x > 20) instead of beta*x > 20 as torch does. This giv |
| `activation-swiglu.glsl` | working | v1 | superseded |  |
| `activation-swiglu-backward.glsl` | working | v1 | superseded |  |
| `activation-tanh.glsl` | working | v1 | superseded |  |
| `activation-tanh-backward.glsl` | working | v1 | superseded |  |
| `adam-update.glsl` | working | v1 | drop |  |
| `adamw-update.glsl` | working | v1, v1 tests | port |  |
| `adaptive-avgpool-3x3.glsl` | working | v1 | superseded |  |
| `snippets/add.glsl` | working | fragment | superseded |  |
| `addition-linear.glsl` | working | v1 | drop |  |
| `snippets/addition_linear.glsl` | working | fragment | drop |  |
| `affect-adam.glsl` | working | v1 | superseded |  |
| `affect-mlp-backward.glsl` | partial | v1 | drop | medium: L216 Dropout used in affect-mlp-forward is ignored: dropped units still get a gradient and kept units miss the 1/(1 |
| `affect-mlp-forward.glsl` | working | v1 | drop | medium: L134 The layer-2 dropout seed (seed+1) makes layer-2 masks a fixed permutation of the layer-1 masks. |
| `attention-backward.glsl` | working | v1 | superseded | medium: L180 grad_K path recomputes the softmax row dot for every query inside the per-key loop, so each thread does O(S^2) |
| `attention-concat-heads.glsl` | working | v1 | drop |  |
| `attention-mask.glsl` | working | v1 | superseded |  |
| `attention-output.glsl` | working | v1 | superseded |  |
| `attention-output-gpt.glsl` | working | v1 | superseded |  |
| `attention-output-t5.glsl` | working | v1 | superseded |  |
| `attention-prosody-modulation.glsl` | working | v1 | drop |  |
| `attention-q-similarity.glsl` | working | — | drop | medium: L20 count = seq_len - 1 underflows when seq_len == 0. |
| `attention-qkv-merge.glsl` | working | v1 | superseded |  |
| `attention-qkv-split.glsl` | working | v1 | superseded |  |
| `attention-scores.glsl` | working | v1 | superseded |  |
| `attention-transpose-bhsd-bshd.glsl` | working | v1 | superseded |  |
| `avgpool2d-backward.glsl` | working | v1 | superseded | medium: L51 Each input-pixel thread scans the ENTIRE output plane (out_h*out_w) to find windows containing it, and recount |
| `avgpool2d-forward.glsl` | working | v1 | superseded |  |
| `bandit-solve.glsl` | working | v1 | drop | medium: L48 When n_arms > 128 the thread returns without writing its outputs. |
| `batch-similarity.glsl` | broken | — | superseded | high: L52 Early return before barrier(): out-of-range threads skip the tile loads and the barriers, so in-range outputs  |
| `batchnorm2d-backward.glsl` | working | v1 | superseded | medium: L100 One invocation per channel serially walks all N*H*W elements (up to three passes); with C<=256 the whole op ru |
| `batchnorm2d-forward.glsl` | working | v1 | superseded | medium: L95 Variance computed as E[x^2]-E[x]^2 in float32 (comment on line 76 claims Welford); catastrophic cancellation m |
| `blockcode-bind.glsl` | working | — | superseded | medium: L25 Assumes exactly one element > 0.5 per block, so it is not a general per-block circular convolution/correlation |
| `blockcode-similarity.glsl` | working | — | superseded |  |
| `blockcode-unbind.glsl` | working | — | superseded | medium: L24 Assumes exactly one element > 0.5 per block, so it is not a general per-block circular convolution/correlation |
| `bridge-continuous-to-spike.glsl` | working | v1 | superseded |  |
| `bridge-spike-to-continuous.glsl` | partial | v1 | superseded | medium: L70 Rate decoding divides by time_window, not by the number of steps actually summed. |
| `bridge-temporal-weights.glsl` | working | v1 | drop | medium: L25 exp overflows to +inf once t/time_window exceeds ~88.7. |
| `capsule-project.glsl` | working | consumer, grilly2, v1 | drop |  |
| `cast-f32-f16.glsl` | working | v1 | port |  |
| `chunked-sw-attention.glsl` | working | v1 | superseded | medium: L50 q_reg/o_reg are float[4], so head_dim > 256 indexes past them; head_dim is not checked. |
| `chunked-sw-attention-backward.glsl` | broken | v1 | superseded | high: L45 Write-after-read race on sm_reduce[0] between consecutive dot_product() calls: no barrier separates the read a |
| `circular-shift.glsl` | working | v1 | superseded | medium: L66 Rotation assumes MSB-first bit order within each word; the repo's packers (vsa-binarize-pack, hdc-permute-pack |
| `contrastive-gradient.glsl` | broken | v1 | drop | high: L94 Computes the gradient of the dot-product triplet (a.n - a.p), not of the cosine-distance triplet that contrast |
| `contrastive-loss.glsl` | working | v1 | drop |  |
| `conv1x1.glsl` | working | — | superseded |  |
| `conv1x1-backward-input.glsl` | working | — | superseded | medium: L41 Assumes the 32-invocation workgroup is one subgroup: subgroupAdd only reduces within a subgroup and only lane  |
| `conv1x1-backward-weight.glsl` | working | v1 | superseded | medium: L38 Every pixel thread atomically adds into all C_out*C_in weights: H*W-way float-atomic contention per address an |
| `conv1x1-subgroup.glsl` | working | — | superseded | medium: L57 Assumes subgroup size >= 32 (workgroup == one subgroup); subgroupAdd reduces only within a subgroup and only l |
| `conv2d-3x3-gelu.glsl` | working | v1 | drop |  |
| `conv2d-backward-input.glsl` | working | v1 | superseded |  |
| `conv2d-backward-weight.glsl` | working | v1 misc, v1 | superseded | medium: L75 barrier() is called inside a divergent branch: out-of-range invocations barrier and return, in-range ones neve |
| `conv2d-forward.glsl` | working | v1 | superseded |  |
| `convd_col2im.glsl` | working | v1 | drop |  |
| `convd_col2im_noatomic.glsl` | working | — | superseded |  |
| `convd_im2col.glsl` | working | v1 misc, v1, v1 tests | superseded |  |
| `cross-entropy-backward.glsl` | working | v1 misc, v1, v1 tests | drop |  |
| `decode-greedy.glsl` | working | v1 | superseded | medium: L43 One invocation scans the whole vocabulary serially, so single-token decode runs on one GPU thread. |
| `decode-sample.glsl` | working | v1 | superseded | medium: L41 When the uniform exceeds the float cumulative sum of the row, the loop exits without a hit and returns token 0 |
| `densenet-conv2d.glsl` | working | — | drop |  |
| `dequant-4bit.glsl` | working | — | superseded |  |
| `dg-sparse-expand.glsl` | working | v1 | drop | medium: L74 Top-k pass assumes output_dim <= 256 (workgroup size and shared array size). |
| `distillation-loss.glsl` | working | v1 | drop | medium: L162 Write-after-read race on shared memory: subgroup 0 overwrites shared_s_sum[0] with its KL partial with no barr |
| `domain-classifier.glsl` | partial | consumer, v1 | superseded | high: L72 Pooled path (pooling_type 0-2) uses only the first 256 feature dims (private pooled_feat[256]; loops capped at |
| `domain-combine-experts.glsl` | working | v1 | superseded |  |
| `domain-predict.glsl` | working | v1 | superseded |  |
| `domain-router.glsl` | working | v1 | superseded |  |
| `snippets/dropout.glsl` | working | fragment | drop | medium: L2 pcg_hash_drop omits PCG's LCG step (state*747796405u + 2891336453u), leaving only the output permutation. It h |
| `eggroll-generate.glsl` | working | v1 | port | medium: L69 The PRNG stream is keyed by seed+id, so calls with nearby seeds reuse the same samples. |
| `eggroll-update.glsl` | broken | v1 | port | high: L59 The merit criterion /avg_delta/>1e-8 is true for almost every element under dense Gaussian perturbations, so m |
| `elementwise-add.glsl` | working | v1 | superseded |  |
| `embedding-attention.glsl` | broken | v1 | drop | high: L145 Pass 2 softmax assumes one row per workgroup but the 16x16 workgroup holds 16 rows: loops stride by gl_WorkGro |
| `embedding-backward.glsl` | working | v1 | superseded | medium: L25 ids are not bounds-checked (no vocab_size push constant), while embedding-lookup zeroes out-of-range ids, so p |
| `embedding-ffn.glsl` | working | v1 | superseded |  |
| `embedding-lookup.glsl` | working | consumer, v1 | superseded |  |
| `embedding-lookup-tiled.glsl` | working | v1 | superseded |  |
| `embedding-normalize.glsl` | working | v1 | superseded |  |
| `embedding-pool.glsl` | working | v1 | superseded |  |
| `embedding-pool-mask.glsl` | broken | v1 | drop | high: L49 Mean path: mask_sums[b] is written only by the thread with dim_idx==0 AND lID==0, and read by other workgroups |
| `embedding-position.glsl` | working | v1 | drop |  |
| `faiss-distance.glsl` | working | v1 | superseded |  |
| `faiss-ivf-filter.glsl` | partial | v1 | drop | medium: L49 num_probe is ignored; only query_cluster_id[0] is probed. |
| `faiss-kmeans-update.glsl` | working | v1 | drop | medium: L48 Accumulates onto whatever centroids[] already holds (binding is named 'Current centroids'); there is no zeroin |
| `faiss-quantize.glsl` | working | v1 | superseded | medium: L35 Running minimum starts at 1e10 instead of +inf or the first distance. |
| `faiss-topk.glsl` | working | v1 | superseded | medium: L52 1e10 is both the scan's starting minimum and the only 'nothing found' signal; min_idx defaults to 0. |
| `fft-bind.glsl` | broken | v1 | superseded | high: L54 compute_fft takes the shared array as an 'inout' parameter; GLSL copies it in and out, so each invocation tran |
| `fft-bitrev.glsl` | working | v1 | port |  |
| `fft-butterfly.glsl` | working | v1 | port | medium: L50 Twiddle sign is reversed relative to the comment on line 49 and the standard DFT: 'forward' uses exp(+2*pi*i*p |
| `fft-magnitude.glsl` | working | v1 | superseded |  |
| `fft-normalize.glsl` | working | v1 | superseded |  |
| `fft-power-spectrum.glsl` | working | v1 | superseded |  |
| `fisher-ewc-penalty.glsl` | working | v1 | port |  |
| `fisher-info.glsl` | working | v1 | port |  |
| `fisher-natural-gradient.glsl` | working | v1 | superseded |  |
| `fisher-normalize.glsl` | working | v1 | superseded |  |
| `flash-attention2.glsl` | broken | v1 | drop | high: L154 Unsynchronized read-modify-write of running_max/running_sum/output_accum by all threads sharing a query row. |
| `flash-attention2-rope.glsl` | broken | v1 | drop | high: L210 Every thread of a tile does an unsynchronized read-modify-write of running_max/running_sum/output_accum for it |
| `flash-attention3.glsl` | broken | — | drop | high: L103 The Q.K dot product is reduced with subgroupAdd but spans all 64 invocations; with subgroups smaller than 64 e |
| `fnn-dropout.glsl` | working | v1 | superseded | medium: L44 Divides by keep_prob = 1 - dropout_prob with no guard; p = 1.0 is valid in torch (outputs zeros). |
| `fnn-layernorm.glsl` | working | v1 | superseded |  |
| `fnn-layernorm-backward.glsl` | broken | v1 | superseded | high: L124 grad_input formula multiplies the mean terms by the current feature's gamma: dx = r*g_i*(dy_i - (sum1 + xhat_i |
| `fnn-linear.glsl` | working | consumer, v1, v1 tests | superseded |  |
| `fnn-linear-backward.glsl` | working | v1 misc, v1 | superseded |  |
| `fnn-residual.glsl` | working | v1 | superseded |  |
| `fnn-xavier-init.glsl` | working | v1 | drop |  |
| `fused-geglu.glsl` | working | — | drop | medium: L81 The activation is applied to the 'up' projection, not the 'gate'. The standard GeGLU (and swiglu-fused.glsl's  |
| `fused-geglu-backward.glsl` | working | — | drop |  |
| `fused-layernorm-linear.glsl` | broken | v1 | superseded | high: L24 The C++ entry points (cpp/src/ops/fused.cpp) take dIn/dHidden/dOut but never specialize the shader: it always  |
| `fused-linear-gcu.glsl` | working | v1 | drop |  |
| `fused-linear-gelu.glsl` | working | v1 | superseded |  |
| `fused-linear-relu.glsl` | working | v1 | superseded |  |
| `fused-linear-roswish.glsl` | working | v1 | drop |  |
| `fused-linear-silu.glsl` | working | v1 | superseded |  |
| `fused-linear-tanh.glsl` | working | v1 | superseded |  |
| `fused-mlp-gelu.glsl` | broken | v1 | superseded | high: L33 The C++ entry points (cpp/src/ops/fused.cpp) take dIn/dHidden/dOut but never specialize the shader: it always  |
| `fused-qkv-projection.glsl` | working | — | superseded |  |
| `fused-reglu.glsl` | working | — | drop | medium: L73 The ReLU is applied to the 'up' projection. The standard ReGLU is ReLU(x W_gate) * (x W_up), so weights bound  |
| `fused-reglu-backward.glsl` | working | — | drop |  |
| `fused-swiglu.glsl` | working | — | superseded | medium: L78 SiLU is applied to the 'up' projection. The standard SwiGLU (LLaMA, and the sibling swiglu-fused.glsl and g2-s |
| `fused-swiglu-backward.glsl` | working | — | superseded |  |
| `gbf-fusion.glsl` | working | — | superseded |  |
| `snippets/gelu.glsl` | working | fragment | superseded |  |
| `gemm-bias-add.glsl` | working | v1 | superseded |  |
| `gemm-coopmat-shared.glsl` | working | v1 | port |  |
| `gemm-tiled.glsl` | working | v1 | superseded |  |
| `gemm_coopmat.glsl` | working | v1 | drop | medium: L37 Each workgroup computes 64 output columns (4 x 16), but the header and push-constant comments only require N p |
| `gemm_mnk.glsl` | working | consumer, v1, v1 tests | superseded |  |
| `gemm_tiled.glsl` | working | v1 | port |  |
| `gif-neuron.glsl` | working | v1 | drop |  |
| `gif-prosody.glsl` | working | — | drop |  |
| `gqa-attention.glsl` | working | v1 | superseded |  |
| `grid-cell.glsl` | working | consumer | drop |  |
| `snippets/grid_cell.glsl` | working | fragment | drop |  |
| `hamming-search.glsl` | working | v1 | superseded |  |
| `hamming-top1.glsl` | working | grilly2, v1 | port | medium: L50 vec4Count = words_per_vec / 4 truncates, and the per-entry stride (line 63) uses vec4Count. |
| `hamming-topk.glsl` | working | v1 | superseded | medium: L32 Query load covers at most 256 uvec4s (one per thread into shared_query[256]). |
| `hdc-bind-packed.glsl` | working | — | superseded |  |
| `hdc-bundle-packed.glsl` | working | — | superseded |  |
| `hdc-bundle-subgroup.glsl` | broken | — | superseded | high: L36 Each lane loads one input vector, so only the first gl_SubgroupSize vectors are counted; the rest are silently |
| `hdc-overlap-metrics.glsl` | broken | v1 | drop | high: L60 Reduction is subgroupAdd over a 64-thread workgroup; with a 32-wide subgroup there are two subgroups, and each |
| `hdc-permute-packed.glsl` | working | — | superseded | medium: L20 shift >= total_bits underflows the uint before the modulo. |
| `hdc-similarity-packed.glsl` | working | — | superseded |  |
| `hebbian-learning.glsl` | working | v1 | port |  |
| `hilbert_similarity.comp` | working | v1 | drop |  |
| `hmm-baum-welch.glsl` | partial | — | drop |  |
| `hmm-forward.glsl` | partial | — | drop |  |
| `hopfield-surprise.glsl` | partial | — | drop |  |
| `hybrid-blend.glsl` | working | — | superseded |  |
| `int8-gemm.glsl` | working | v1 misc, v1 | superseded |  |
| `snippets/layernorm.glsl` | working | fragment | superseded |  |
| `lif-neuron.glsl` | working | v1 | superseded |  |
| `lora-backward.glsl` | broken | v1 | superseded | high: L98 Phase 1 returns early for rank_idx >= rank or batch_idx >= batch_size before the tiled loop's barrier() calls  |
| `lora-forward.glsl` | broken | v1 | superseded | high: L72 Phase 0 returns early for col >= rank (or row >= batch_size) before the tile loop's barrier(); the exited invo |
| `loss-ce-fused.glsl` | working | v1, v1 tests | port | medium: L51 lds_max/lds_sum sized [16] assume subgroup size >= 16, but the workgroup is 256 wide. |
| `loss-cosine.glsl` | working | v1 | superseded |  |
| `loss-cross-entropy.glsl` | broken | v1, v1 tests | drop | high: L105 Label smoothing is wrong: it adds eps*log(V), a constant, instead of eps*mean_k(-log p_k), the cross-entropy a |
| `loss-fn-bce.glsl` | working | v1 | superseded |  |
| `loss-mse.glsl` | working | v1 | superseded |  |
| `loss-nce-fused.glsl` | working | v1 | port | medium: L91 lds_loss[16] assumes subgroup size >= 16 for a 256-wide workgroup. |
| `loss-policy-grad-fused.glsl` | working | v1 | port | medium: L29 lds_max/lds_sum sized [16] assume subgroup size >= 16 for a 256-wide workgroup. |
| `loss-sampled-bce-fused.glsl` | working | v1, v1 tests | drop | medium: L85 lds_loss[16] assumes subgroup size >= 16 for a 256-wide workgroup. |
| `lsq-stochastic-quant.glsl` | partial | — | drop |  |
| `lstm-cell-forward.glsl` | working | — | superseded |  |
| `maxpool-2x2.glsl` | working | v1 | drop | medium: L38 Input height is reconstructed as out_height*2 (only in_width is passed), so the channel stride is wrong for od |
| `maxpool2d-backward.glsl` | working | v1 | superseded | medium: L56 Each input-pixel thread scans every output index of its (n,c) plane: O(H_in*W_in*H_out*W_out) per plane. |
| `maxpool2d-forward.glsl` | working | v1 | superseded |  |
| `memory-context-aggregate.glsl` | working | v1 | superseded |  |
| `memory-inject-concat.glsl` | working | v1 | superseded |  |
| `memory-inject-gate.glsl` | working | v1 | superseded |  |
| `memory-inject-residual.glsl` | working | v1 | superseded |  |
| `memory-query-pooling.glsl` | working | v1 | superseded | medium: L58 The mean of every input dim is recomputed inside the projection loop by every output thread. |
| `memory-read.glsl` | working | v1 | superseded |  |
| `memory-write.glsl` | working | v1 | superseded |  |
| `mf-sigmoid.glsl` | working | v1 | superseded |  |
| `mf-softmax.glsl` | broken | v1 | drop | high: L61 relu(x - rowmax) is 0 for every element, because x<=max, so the sum is always 0 and pass 2 always takes the un |
| `mf-softplus.glsl` | working | v1 | drop |  |
| `mingru-backward.glsl` | working | v1 | port |  |
| `mingru-forward.glsl` | working | v1 | port |  |
| `moe-layer-backward.glsl` | working | v1 | drop | medium: L49 Early return before barrier() leaves tileG columns unloaded when d_model is not a multiple of 16. |
| `moe-layer-backward-vec4.glsl` | working | v1 | drop | medium: L32 Early return before the tile loop leaves shared tile columns unloaded (and barrier() in non-uniform flow) when |
| `moe-layer-fused.glsl` | working | v1 | drop | medium: L44 Early return before barrier() leaves tileX columns unloaded when d_model is not a multiple of 16. |
| `moe-layer-fused-vec4.glsl` | working | v1 | drop | medium: L34 Early return before barrier() leaves tileX columns unloaded when d_model/4 is not a multiple of 16. |
| `moe-layer-grad-weight.glsl` | broken | v1 | drop | high: L57 tileG is loaded with the thread's own output column (col = col_base+ty) instead of col_base+tx, so it holds th |
| `moe-router.glsl` | working | v1 | drop |  |
| `moqe-dynamic-quant.glsl` | working | v1 | drop | medium: L28 One block is assumed to be one subgroup; with subgroups smaller than 32 each subgroup computes its own absmax  |
| `moqe-fused-gemv.glsl` | working | v1 | superseded |  |
| `moqe-fused-gemv-dp4a.glsl` | working | consumer, grilly2, v1 | superseded |  |
| `moqe-gumbel-router.glsl` | working | — | drop | medium: L31 float(state)/2^32 rounds to exactly 1.0 for state >= 2^32-128, and -log(-log(1.0)) = +inf makes the weights Na |
| `moqe-hard-router.glsl` | working | — | drop | medium: L50 No bounds check on dim_idx or token_idx; dispatch X = embedding_dim with local_size 32 overruns when embedding |
| `nlms-ensemble.glsl` | working | v1 | drop | medium: L41 Local array gate_logits[64] silently drops experts beyond 64. |
| `nlms-metrics.glsl` | working | v1 | drop | medium: L43 last_error is written by every invocation without ordering, so it holds an arbitrary batch element's error, no |
| `nlms-predict.glsl` | working | v1 | superseded |  |
| `nlms-update.glsl` | working | v1 | port | medium: L63 Data race on mu[0]: thread 0 writes the decayed mu while other invocations read mu[0] at line 79 in the same d |
| `oja-learning.glsl` | working | consumer, v1 | port | medium: L59 Stable only when eta*//x//^2 < 1. With the header's own example sizes and unnormalised VSA inputs it diverges  |
| `perceiver-encode.glsl` | working | v1 | superseded | medium: L48 q_reg[64] and out_acc[64] are fixed-size; head_dim > 64 indexes past them. |
| `place-cell.glsl` | working | grilly2, v1 | drop | medium: L42 spatial_dims other than 2 or 3 silently gives distance 0, so every cell fires at max_rate. |
| `prefix-scan-causal.glsl` | working | consumer, v1 | port |  |
| `prefix-scan-causal-backward.glsl` | working | consumer, v1 | port |  |
| `reduce-sumsq.glsl` | working | v1 | superseded |  |
| `snippets/relu.glsl` | working | fragment | superseded |  |
| `residual-relu.glsl` | working | — | superseded |  |
| `resonator-bitpacked.glsl` | working | v1 | superseded |  |
| `rms-norm.glsl` | working | v1 | superseded |  |
| `rms-norm-backward.glsl` | working | v1 | superseded | medium: L57 Every element-thread recomputes its whole row's sum(x^2) and sum(g*w*x): O(n) work per element, O(n^2) per row |
| `rms-norm-linear-fused.glsl` | working | — | superseded |  |
| `rope.glsl` | working | grilly2, v1 | superseded |  |
| `sanger-gha.glsl` | working | — | port | medium: L65 Data race: invocation (j,d) reads W[k*dim+d] for k<j while invocation (k,d) writes that element (line 69) in t |
| `semantic-encoder.glsl` | working | v1 | superseded |  |
| `sign-activation.glsl` | working | v1 | port |  |
| `snippets/silu.glsl` | working | fragment | superseded |  |
| `snn-expand-time-dim.glsl` | working | — | drop |  |
| `snn-expert-readout.glsl` | working | — | superseded |  |
| `snn-matmul.glsl` | working | v1 | superseded |  |
| `snn-node-backward.glsl` | partial | v1 | superseded | medium: L56 Returns dL/dh, not dL/dx: the input scale 1/tau of the forward's decay_input=1 mode is missing, and there is n |
| `snn-node-forward.glsl` | working | v1 | superseded |  |
| `snn-readout.glsl` | working | v1 | superseded |  |
| `snn-rmsnorm.glsl` | working | v1 | superseded |  |
| `snn-softmax.glsl` | working | v1 | superseded |  |
| `snn-spiking-attention.glsl` | working | — | superseded |  |
| `snn-synapse-filter.glsl` | working | v1 | superseded |  |
| `snn-time-expand.glsl` | working | — | superseded |  |
| `softmax-fast.glsl` | partial | — | superseded | medium: L54 n_sgs = ceil(row_length/32) hardcodes a subgroup size of 32 instead of using gl_NumSubgroups. |
| `spike-propagate-batch.glsl` | working | v1 | port |  |
| `spike-scatter.glsl` | working | v1 | drop |  |
| `ssd-scan-chunks.glsl` | working | v1 | drop |  |
| `ssm-fused-math.glsl` | working | v1 | drop |  |
| `ssm-fused-uv.glsl` | working | v1 | drop |  |
| `stdp-learning.glsl` | broken | v1 | drop | high: L99 The LTP and LTD terms are identical (pre_t*post_t both), so the rule is order-blind Hebbian with rate lr_p - l |
| `surprise-momentum.glsl` | working | consumer, v1 | drop |  |
| `surprise-recall-blend.glsl` | working | — | drop |  |
| `swiglu-fused.glsl` | working | v1 | superseded |  |
| `synapse-dense.glsl` | working | v1 | drop |  |
| `synapsis-forward.glsl` | working | v1 | superseded |  |
| `synapsis-stdp-trace.glsl` | working | v1 | drop |  |
| `synapsis-stdp-update.glsl` | broken | v1 | drop | high: L52 No depression term and no timing, and it uses the product of batch means instead of the batch mean of products |
| `tensor-bmm.glsl` | working | v1 | superseded |  |
| `tensor-cat.glsl` | working | v1 | superseded |  |
| `tensor-index-select.glsl` | working | v1 | superseded |  |
| `tensor-transpose.glsl` | working | v1, v1 tests | superseded |  |
| `tensor-where.glsl` | working | v1 | superseded |  |
| `ternary-gemm.glsl` | working | v1, v1 tests | port |  |
| `theta-gamma-encoding.glsl` | working | v1 | drop | medium: L61 Position is normalised by (max_seq_len-1) to exactly one theta cycle, so the first and last positions get iden |
| `time-cell.glsl` | working | consumer, grilly2, v1 | drop |  |
| `transpose-f32-f16.glsl` | working | v1 | drop |  |
| `vsa-argmax.glsl` | broken | — | port | high: L29 The block index is gl_LocalInvocationID.z and local_size_z is fixed at 8 (line 11), so only blocks 0..7 are ev |
| `experimental/vsa-argmax-rows.glsl` | working | v1, v1 tests | superseded |  |
| `vsa-binarize-pack.glsl` | working | consumer | port |  |
| `experimental/vsa-bind.glsl` | working | v1 | superseded |  |
| `experimental/vsa-bind-batch.glsl` | working | v1 | superseded |  |
| `experimental/vsa-bundle.glsl` | working | v1 | superseded |  |
| `experimental/vsa-bundle-batch.glsl` | working | v1 | superseded |  |
| `vsa-explore.glsl` | partial | — | drop | medium: L94 t = tid % (seq_len - 1): seq_len=1 is a modulo by zero, and n_trials that is not a multiple of seq_len-1 over- |
| `experimental/vsa-fft-convolve.glsl` | partial | v1 | drop | high: L70 Stage 1 is not an inverse FFT: it copies the real part of the product spectrum divided by dim. |
| `vsa-logic-apply.glsl` | working | — | superseded |  |
| `experimental/vsa-mask-selected.glsl` | working | v1, v1 tests | superseded |  |
| `experimental/vsa-resonator-step.glsl` | working | v1, v1 tests | superseded |  |
| `experimental/vsa-similarity-batch.glsl` | working | v1 | superseded |  |
| `whitening-apply.glsl` | working | v1 | port |  |
| `whitening-batch-stats.glsl` | working | v1 | port |  |
| `whitening-transform.glsl` | working | v1 | drop | medium: L55 Data race: invocation (b=0,d) writes mu[d]/var[d] (lines 51,65) while invocations (b>0,d) read them (lines 55, |
| `zero_collision_hash.comp` | broken | — | drop | high: L139 Does not compile: buffers are indexed with int64_t, but GLSL requires a 32-bit scalar integer index. |
