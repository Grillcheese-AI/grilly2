# Inference (grilly-infer)

CLAUDE.md §11 sets the build order:
1. safetensors loading;
2. int8 (and int4) weight-only quantization;
3. GPU sampling;
4. a `generate()`-shaped decode loop;
5. wiring v1's KV-cache ops.

This file records each step as it lands.

The machinery is the Rust crate `crates/grilly-infer`, which depends on
`grilly-core` only, and the Python package `grilly.infer`. Neither knows
about Hugging Face. A model-zoo adapter can be layered over them and
deleted without breaking them.

## 1. Loading safetensors

```python
from grilly.infer.safetensors import load_file   # was: from safetensors.torch import load_file

model.load_state_dict(load_file("model.safetensors"))
```

`grilly.infer.safetensors` has the API of `safetensors.torch`:

| name | what it does |
|---|---|
| `load_file(filename, device="cpu")` | every tensor, in file order |
| `load(data: bytes)` | the same, from a file already in memory |
| `safe_open(filename, framework="pt", device="cpu")` | `keys()`, `offset_keys()`, `metadata()`, `get_tensor(name)`; a context manager |
| `SafetensorError` | a malformed file, a missing tensor, or use after close |

A missing file raises `FileNotFoundError`. `device` accepts what grilly's
factories accept. `"cpu"` is an alias for grilly's one device, so HF-style
`load_file(path, device="cpu")` calls work unchanged. `framework` must name
torch.

`safetensors.torch` itself cannot be used under `import grilly as torch`.
It refers to `torch.bfloat16`, `torch.uint8`, `torch.complex64` and other
dtypes at import, and grilly has none of them.

**How it loads.** The file is memory-mapped. Its header is parsed and
validated by Hugging Face's `safetensors` crate (offsets, sizes and
overlaps), not by a parser of grilly's own. Each tensor is uploaded when it
is read.

| file dtype | grilly dtype | conversion |
|---|---|---|
| F32 | float32 | none |
| BF16, F16 | float32 | widened, exact for every value |
| I64, I32 | int64, int32 | none |
| BOOL | bool | one byte to one u32 per element |
| anything else | — | `NotImplementedError` naming the tensor |

**bf16 and f16.** grilly has no half-precision dtypes yet (an open
decision). A bf16 or f16 checkpoint therefore loads with every value
unchanged, but its tensors report `float32` and use twice the memory. F64
is refused rather than rounded to float32.

**Not yet:**
- `get_slice` (it raises `NotImplementedError`);
- `save_file` / `save`;
- sharded checkpoints (`model.safetensors.index.json`).

The shard index is Hugging Face's convention, so reading it belongs in the
adapter.

**Tests.**
- `crates/grilly-infer/tests/safetensors.rs` uses Rust only.
- `tests/python/test_safetensors.py` checks, bit for bit against
  `safetensors.torch` and torch's `.float()`:
  - every one of the 65,536 bf16 and f16 bit patterns;
  - every supported dtype, and scalar and empty shapes;
  - key order, metadata and errors;
  - a torch model's checkpoint running in the same grilly model, in
    float32 and bf16;
  - all-MiniLM-L6-v2 from the local HF cache, when it is there.

### Measured: Qwen2.5-0.5B-Instruct

On the RX 6750 XT, with the file already in the OS cache: 290 tensors, all
BF16, 0.92 GiB on disk and 1.84 GiB as float32 (494M parameters).
`load_file` takes 1.5 s.

## Memory for weights (option A)

**The problem, measured on the first loader.** The weights occupied
3.00 GiB of VRAM for 1.84 GiB of data. v1's buffer pool rounds every
device allocation up to a power of two below 256 MB, and summing those
rounded sizes over Qwen's tensors predicts 2.97 GiB. A 4864×896 MLP weight
(17.4 MB) sat in a 32 MB bucket. Quantization would not have fixed this:
an int8 copy of that weight is 4.4 MB, which rounds to 8 MB.

**The decision (owner).** Weights get memory of exactly their size, never
pooled. Activations keep the pool's reuse.

**What changed.**
- `MemClass::Persistent` (`GC_MEM_DEVICE_PERSISTENT` in the shim)
  allocates exactly the requested size through v1's VMA allocator, not
  through its pool. It is destroyed when released. If a batch is recording
  at that moment, the release is deferred until the batch has run.
- Buffers of 1 MiB or more get a device-memory allocation of their own.
  Freeing one returns exactly that memory. A sub-allocation would share a
  VMA block with pooled buffers, which keep the block alive.
- Loaded tensors and the parameters of `nn.Linear`, `nn.LayerNorm` and
  `nn.Embedding` are persistent.
- `load_state_dict(assign=True)` is implemented as in torch: the loaded
  tensors become the parameters, with no copy, and each keeps the module's
  `requires_grad`.
- Transfers between the host and device memory are staged by the shim
  through two 64 MiB buffers, one chunk at a time, at any offset.
  - v1's `uploadStaged` and `downloadStaged` keep a per-thread staging
    buffer as large as the largest transfer ever made, and never free it.
    That was 0.51 GiB of system RAM after the embedding alone.
  - In-place RNG (`uniform_`, `normal_`, so every `nn.init`) now writes
    straight into the tensor. It used to upload into a pooled temporary
    and copy from it, and each parameter's temporary stayed in the pool.

**Measured**, on a grilly model with Qwen2.5-0.5B's parameter names and
shapes, built and then loaded from the checkpoint. The memory is the
process's own, from Windows' GPU counters.

| held after loading | first loader, `load_file` alone | model + copy, before the fix | model + `assign=True`, before the fix | model + copy or `assign=True`, now |
|---|---:|---:|---:|---:|
| dedicated VRAM | 3.00 GiB | 2.75 GiB | 3.51 GiB | **1.94 GiB** |
| shared (system RAM) | 0.54 GiB | 0.54 GiB | 0.54 GiB | **0.13 GiB** |

The weights are 1.84 GiB. The 0.13 GiB of system RAM is the two staging
buffers, and it no longer grows with the largest tensor.

The peak is 3.79 GiB, reached while the model's own parameters and the
loaded dict both exist. Streaming tensors from the file into the model
one at a time would remove it. That belongs with the decode-loop step.

Building the model takes about 3.7 s. I have not profiled it. The likely
cost is drawing 494M random initial values on the host, which a load then
overwrites. torch avoids this with meta-device initialisation, which
grilly does not have yet.

## 2. int8 weight-only quantization

```python
from grilly.infer.quantization import quantize_, Int8WeightOnlyConfig   # was: from torchao.quantization import ...

quantize_(model, Int8WeightOnlyConfig())
```

`grilly.infer.quantization` has torchao's API:

| name | what it does |
|---|---|
| `quantize_(model, config, filter_fn=..., device=None)` | quantizes, in place, the weight of every module `filter_fn(module, fqn)` accepts; by default every `nn.Linear` not quantized yet |
| `Int8WeightOnlyConfig(group_size=None, granularity=PerRow(), ...)` | version 2, per-row only |
| `PerRow` | the granularity: one scale per output feature |
| `Int8Tensor` | the quantized weight; `from_hp(w)`, `dequantize()`, `scale`, `nbytes` |

**Numerics: torchao's, exactly.**
- `scale = max|w| / 127.5` per output row, at least float32 eps;
- `q = clamp(round(w · (1/scale)), −128, 127)`, rounding half to even;
- `F.linear(x, q) = (x @ q.T) · scale + bias`, the scale applied to the
  finished dot product, as in torchao's weight-only path.

Both divisions have to be correctly rounded to reproduce torchao's values.
Vulkan allows `a / b` to be 2.5 ulp off, and on this card it is: the
native division missed IEEE's result on many of 2^24 random pairs.
`div_rn` in `g2-common.glsl` refines it with fused multiply-adds (a
Newton step on the reciprocal, then a residual correction). It matches
IEEE division on all 2^24 pairs, in `crates/grilly-core/tests/div_rn.rs`.

Checked against torchao 0.18's own code (its pure-Python quantization
path, run from the downloaded wheel): **0 of 10,019,412 int8 values and 0
scales differ**, over six weight shapes up to 4864×896. The linear outputs
agree to within 5e-7 of the output's magnitude.

**How it runs.**
- The weight is held as int8, four values to a 32-bit word, each row padded
  to 16 bytes, plus a float32 scale per row, all in persistent memory. The
  float32 weight is freed.
- `F.linear` goes to `g2-linear-q8`, which dequantizes inside the kernel.
  - 64 threads own one output feature and read its row 16 bytes per load.
    Up to 8 rows of `x` share each weight read.
  - `x` is read as `vec4` when its rows are 16-byte aligned.
  - Sums run in a fixed order, so results are deterministic.
- Any other use of the weight sees `dequantize()`, computed on demand.
- The weight is frozen. Gradients reach the layer's input and bias (the
  backward dequantizes for the duration of the step), so a quantized model
  runs with or without `no_grad`, as with torchao.

**Tests.**
- `crates/grilly-infer/tests/quant.rs` runs from Rust only:
  - bit-exact values, scales and dequantized weights, including zero rows
    and ties;
  - the linear against float64 under the error bound its summation order
    implies (worst measured ratio 0.043);
  - gradients.
- `tests/python/test_quantization.py` runs the same checks against
  torchao's formula in torch, plus:
  - a quantized model against torch running torchao's weight-only math
    (rtol 1e-5);
  - the API: filters, the configs torchao refuses, and memory.

### int4: half the bytes again, at a price

`Int4WeightOnlyConfig(group_size=128)` is torchao's group-wise asymmetric
scheme, and group-wise is what makes four bits usable at all — one scale
for a whole row would spend its sixteen levels on that row's largest value:

    scale = (max - min) / 15        over each group of `group_size`
    zero  = min + 8 * scale         (torchao's float zero-point domain)
    q     = clamp(round((w - min) / scale), 0, 15)

torchao is not installed here, so unlike the int8 transcription — which was
checked against torchao 0.18 itself — this one follows the published formula
only. What the tests assert is that grilly computes *that* formula exactly:
the levels, scales and zero points are bit-exact against it, because both
divisions are correctly rounded on the GPU (`div_rn`), and the linear is
measured against float64 under a bound derived from the kernel's summation.

Measured on Qwen2.5-0.5B-Instruct, end to end:

| | int8 | int4 (group 128) |
|---|---:|---:|
| weights + cache, persistent | 1.19 GiB | **0.89 GiB** |
| peak VRAM while generating | 1.24 GiB | **0.94 GiB** |
| decode | **157 tokens/s** | 104 tokens/s |
| relative error of a linear's output, real weights | **~1%** | ~11% |

So int4 is smaller, slower and much less accurate, and the text shows it —
coherent but visibly worse. Two things follow, both open:

- **the accuracy** is plain round-to-nearest, which is the weakest thing one
  can do at four bits. What to do about it is measured below;
- **the speed**: int4 reads half the bytes of int8 and is slower, so it is
  not memory-bound — it is the unpacking and the per-group scale. The
  kernel has already been through this once: holding a second pair of
  per-row accumulators spilled them to scratch and cost 40%, and reading
  the weight per row rather than per 16-byte vector cost another 60%.

The decode kernel structure mirrors `g2-linear-q8` deliberately (64 lanes
per output feature, `ROWS` rows per workgroup, a fixed tree at the end), and
a group is a multiple of 32 so that each 16-byte load of 32 levels lies in
one group and reads one scale.

#### A Hadamard rotation does not fix it here (measured)

Quantization is free to work in any orthogonal basis: for `y = W x`, insert
`Q Q^T = I` and get `y = (W Q)(Q^T x)`. A Hadamard `Q` spreads an outlier
across every channel, so the grid is no longer spent on it, and the
rotation folds into the weight when it is quantized — nothing at runtime.
This is what QuaRot, QuIP# and SpinQuant are built on, and it was the
obvious next step for the 11%.

`python tools/int4_rotation.py` measures it instead of assuming it, on the
real weights:

| weights | int4 | rotated | gain |
|---|---:|---:|---:|
| Qwen2.5-0.5B, mean over 19 weights | 1.07e-01 | 9.94e-02 | **1.07x** |
| synthetic gaussian | 9.66e-02 | 9.87e-02 | 0.98x |
| synthetic, 1% outliers x20 | 2.15e-01 | 9.19e-02 | **2.34x** |
| synthetic heavy-tailed (t, df 2) | 2.16e-01 | 9.73e-02 | **2.22x** |

The bottom rows are the control, and they say the method and the script
both work: where there are outliers, rotation removes them. **These weights
simply have few.** Within a group of 128 they are already close to
gaussian, so what four bits cost here is the sixteen levels themselves, not
a handful of large values.

Two things this does *not* say. Rotation is still the right tool for
quantizing **activations** (W8A8, W4A4) and the **KV cache**, where the
outliers are famously large — it just has nothing to do for these weights.
And the accuracy gap has other levers that this measurement points at
instead: a smaller group (32 instead of 128 measured 7.8% against 10.0%,
for 25% more parameter bytes), and error-compensated rounding (GPTQ-style),
which is the usual way weight-only int4 gets close to int8.

One practical note for anyone extending this: a Hadamard matrix of order
`n` needs `n` to be 1, 2 or a multiple of 4, and a *construction* to exist
for it. 896 = 4 x 224 works (Sylvester times Paley over GF(223)), but
Qwen's intermediate size 4864 = 2^8 x 19 has none of the usual ones, so a
full-width rotation there would have to be block-diagonal. The script
checks `H H^T = I` before using a matrix — the first version of this
measurement used a Hadamard that was not orthogonal (off by 9e-03) and
quietly reported the same conclusion for the wrong reason.

### Measured, RX 6750 XT

GPU time per call at Qwen2.5-0.5B's shapes (`tools/int8_linear.py`: 200
calls per batch, the shim's wait divided by 200). GB/s counts the weight
bytes read.

| layer (out × in) | rows | float32 µs | int8 µs | int8 GB/s | int8 vs float32 |
|---|---:|---:|---:|---:|---:|
| q/o_proj (896 × 896) | 1 | 24.8 | 5.8 | 139 | 4.3× faster |
| k/v_proj (128 × 896) | 1 | 22.9 | 3.4 | 33 | 6.7× faster |
| gate/up_proj (4864 × 896) | 1 | 76.0 | 21.1 | 208 | 3.6× faster |
| down_proj (896 × 4864) | 1 | 125.7 | 12.2 | 358 | 10.3× faster |
| lm_head (151936 × 896) | 1 | 2755 | 581 | 235 | 4.7× faster |
| gate/up_proj | 8 | 76.8 | 69.4 | 63 | 1.1× faster |
| down_proj | 8 | 128.2 | 50.3 | 87 | 2.6× faster |
| q/o_proj | 128 | 107.4 | 106.8 | 8 | 1.01× |
| k/v_proj | 128 | 28.5 | 31.1 | 4 | 0.92× |
| gate/up_proj | 128 | 566 | 246 | 18 | **2.30× faster** |
| down_proj | 128 | 607 | 622 | 7 | 0.98× |
| lm_head | 128 | 22464 | 16427 | 8 | 1.37× faster |

- **Decode (one row)** is where int8 pays most: 3.6–10.3× faster.
- **Prefill** used to be *slower* in int8 (0.62–0.83×) because the decode
  kernel re-reads the whole weight for every 8 rows — at 128 rows, sixteen
  times over, which is why it ran at about 1% of this card's bandwidth.
  `g2-linear-q8-tiled` fixes that (below); no layer is meaningfully slower
  than float32 now, and the MLP's larger half is 2.3× faster.

### Tiling, and why not dp4a

The obvious idea for a faster int8 matmul is the integer dot product
(`v_dot4_i32_i8` on RDNA 2, four int8 multiply-adds in one instruction;
v1's `moqe-fused-gemv-dp4a.glsl` is built on it). The measurements say it
is the wrong lever here, twice over:

- at **decode** the kernel is memory-bound and close to the roof —
  `down_proj` reads its weight at 358–380 GB/s on a card with ~432 GB/s of
  peak bandwidth. Integer dot does not make a weight byte smaller, so
  there is nothing for it to win. The narrow layers (`k_proj` at 33 GB/s)
  are launch-bound instead, which is why *fusing* q/k/v helped;
- at **prefill** the old kernel ran at 5–8 GB/s, so it was not short of
  arithmetic either — it was re-reading the weight.

And dp4a needs *both* operands in int8, while weight-only quantization
keeps activations in float32; it only becomes available with dynamic
activation quantization (torchao's `Int8DynamicActivationInt8WeightConfig`),
which changes the numerics. So the fix is data flow, not instructions.

`g2-linear-q8-tiled` gives each workgroup a 64×64 tile of the output and
walks k in steps of 32, staging x and the unpacked weight in shared memory;
each thread keeps a 4×4 micro-tile in registers. Every value loaded is then
used 64 times instead of once.

**It is chosen by shape, not by hope.** One tile is one workgroup, so a
narrow layer would leave the card idle: `k_proj` at 128 rows makes 4 tiles
for 40 compute units, and measured 115 µs tiled against 31 µs with the
decode kernel's finer grid. `linear` therefore tiles only above 16 rows
*and* 16 tiles (`tiled_for`), and the thresholds come from that table.

End to end on Qwen2.5-0.5B, prompt processing:

| prompt | before | after |
|---|---:|---:|
| 3,030 tokens | 2,180 ms | **1,060 ms** |
| 8,030 tokens | 8,206 ms | **5,231 ms** |

The 8K case gains less because attention is quadratic and by then it, not
the projections, is most of the time.

The two kernels sum in different orders — the decode one splits k over 64
lanes and adds them in a tree, the tiled one sums a row of k in one thread
— so `tests/quant.rs` derives its bound from whichever kernel a shape
uses, and both are measured against float64 (worst 0.043 of the bound).
- **Loading `x` as `vec4`** made decode 1.4–2.4× faster than scalar loads.
  The kernel had been limited by load instructions, not by bandwidth.

**Error** on Qwen2.5-0.5B's real weights, as the int8 output's relative
error against float32, on random normal inputs:
- mean 0.9–1.2% by layer type;
- worst 1.5% (down_proj and v_proj).

Real activations need the decode loop (step 4).

**Memory.** A grilly model with Qwen2.5-0.5B's shapes and weights holds
1.94 GiB of dedicated VRAM loaded, and **0.91 GiB** after `quantize_`:
- 0.33 GiB of int8 linear weights;
- the 0.51 GiB embedding, which stays float32 as with torchao's default
  filter.

`quantize_` takes 0.12 s for the whole model.

## 3. Sampling on the GPU

```python
from grilly.infer import sampling

token = sampling.next_token(logits, input_ids, do_sample=True, temperature=0.7,
                            top_k=20, top_p=0.8, repetition_penalty=1.05)
```

The parameters, defaults and processing order are Hugging Face's
`generate()` (transformers 5.5):
1. repetition penalty over `input_ids`: `x < 0 ? x·r : x/r`, applied in
   greedy decoding too;
2. greedy (`do_sample=False`): the first index of the maximum, as
   `torch.argmax`;
3. when sampling: temperature, then top-k, then top-p, then a draw from
   the softmax of what is left by the inverse CDF in token order, as
   `torch.multinomial`.

`None` turns a filter off. `processed_logits(...)` returns the scores after
the penalty and temperature. `probabilities(...)` returns the filtered
distribution. `torch.multinomial` (grilly's `multinomial` and
`Tensor.multinomial`) uses the same kernel.

**One kernel, no sort, nothing read back.** `g2-sample` runs one workgroup
per row. Only one uniform number per row travels, from grilly's generator
on the host into the push constants. The logits never leave the device.
- Top-k is a radix select of the k-th largest value: four passes of 8-bit
  digits over order-preserving integer keys.
- Top-p is a binary search over those keys for the smallest value whose
  cumulative mass passes (1 − p)·Z: at most 32 masked sums.
- HF instead sorts all 152K scores, twice per token when both filters are
  on.

**Numerics against HF's own processors**, run in torch
(`tests/python/test_sampling.py`):
- Penalized and scaled scores are HF's bit for bit, because both
  divisions use `div_rn`.
- Top-k keeps exactly HF's tokens, including every token tied with the
  k-th value.
- Top-p keeps HF's tokens, with two exceptions:
  - a token whose cumulative probability is within float32 rounding of
    1 − p can be decided differently;
  - among tokens tied at the cut, grilly keeps as many as HF, dropping the
    lowest indices first. HF's choice follows `torch.sort`, which is not
    stable on CPU, so only the count can match.
- Probabilities are within `(|x − max| + 8)·2⁻²⁴` of a float64 softmax
  (worst measured ratio 0.82). The kernel's sums are compensated.
- torch's own float32 softmax is 1.6e-5 off float64 at a 152K vocabulary;
  grilly's is 1.1e-6. So torch is not the reference for this number.
- Draws pass a chi-square test against the distribution, with a fixed
  seed. The same seed gives the same tokens in grilly, but not torch's
  numbers, per the RNG policy.

**Rows with nothing to draw.** A row that is all −inf, or all zero weights
for `multinomial`, returns −1. HF's softmax would give NaN there and
`torch.multinomial` would raise. torch checks that on the host, which would
cost a round trip here.

**A hang, found before this committed.** The first version of the draw
handled a row with no mass by falling into a loop that ran 2³² − 1
iterations on one GPU thread. The dev machine has the Windows GPU watchdog
off (`TdrLevel = 0`), so that loop froze the desktop for minutes and needed
a reset. The loop is now bounded, and a regression test covers the case.

**Speed.** Measured with launch blocking, one row, 151,936-token vocabulary:

| sampling setting | first version | tuned |
|---|---:|---:|
| greedy, with repetition penalty | 11.26 ms | **0.24 ms** |
| top_k 20 | 1.00 ms | **0.69 ms** |
| top_k 50, top_p 0.8, temperature 0.7, penalty 1.3 | 3.99 ms | **0.78 ms** |
| top_p 0.3, top_k off | 5.05 ms | **0.99 ms** |
| top_p 0.9, top_k off, flat distribution (6,200 kept) | 5.38 ms | **2.46 ms** |

Three changes, each measured on its own:
1. **Greedy.** grilly's generic argmax scanned each row on a single
   thread. Contiguous rows of 256 or more now take `g2-argreduce-rows`: a
   workgroup per row, then a tree that keeps torch's rule (NaN is the
   extreme, a tie goes to the lower index). That serves `torch.argmax` and
   `torch.argmin` too.
2. **Candidates in shared memory.**
   - After top-k, the survivors (up to 2048) are copied into shared memory
     in token order. Top-p, the tie rule, the draw and the distribution
     then run there, instead of 32 masked passes over the whole row.
   - With top-k off, the 2048 largest scores serve as a pool, but only
     when the mass below the pool is at most (1 − p)·Z, which proves HF
     drops everything below it. Otherwise the kernel takes the full-row
     path. The flat case above does, and the pool attempt costs it a
     little.
3. **1024 threads per workgroup** instead of 256. Each pass over a 152K row
   was bound by memory latency, not bandwidth.

A pass over the row now costs about 0.07 ms, which is the limit for one
workgroup. Going further would mean splitting a row across workgroups in
extra dispatches.

## 5. The KV cache

```python
from grilly.infer.cache import KVCache

cache = KVCache(num_layers, batch_size, num_kv_heads, head_dim, max_length)
cache.append(layer, keys, values)   # (batch, kv_heads, T, head_dim), written in place
out = cache.attend(layer, query)    # query (batch, heads, 1, head_dim)
```

**Why not v1's cache.** §11 said to wire v1's nine KV-cache ops, not
rebuild them. Reading `cpp/src/ops/kv_cache.cpp` showed three problems.
- **Host round trips.** `kvCacheAppend` takes host arrays: it copies them,
  uploads, runs a GEMM for its low-rank compression, downloads the result
  and copies it into a mapped buffer. `kvCacheDecode` uploads, runs a
  kernel, and downloads the rebuilt keys and values to host memory. In a
  decode loop, every layer's cache would cross PCIe twice per token.
- **Lossy storage.** Its storage is lossy by design: MLA projection, FP8
  keys, INT4/INT2 values.
- **A second path to the GPU.** Its ops drive v1's pool and command batch
  directly. Exposing them through the shim would add a second path to the
  GPU beside the dispatch spine, which §5 forbids.

On the owner's go-ahead (2026-09-19), the cache is built in grilly2 instead. v1 is
unchanged. v1's eviction policies (H2O, a learned eviction head) could be
ported onto this cache later, as options.

**What it is.**
- Keys and values are exact float32, preallocated per layer as
  `(batch, kv_heads, max_length, head_dim)` in persistent memory.
- `append` writes the new positions in place, with grilly's strided copy.
- `attend` runs `g2-attn-decode`: one query token against the filled
  positions, a workgroup per (batch, head). Grouped-query attention works
  by index, so the cache is never copied, sliced or read back.
- A prompt's own attention (prefill) stays ordinary causal
  `scaled_dot_product_attention` on its keys and values, which are then
  appended.
- `get_seq_length(layer)` has Hugging Face's name.
- Appending past `max_length` raises.

**Tests.**
- `crates/grilly-infer/tests/kv_cache.rs` runs from Rust only: prefill and
  decode steps against float64, grouped-query attention, head dims 1, 64,
  80 and 128 (worst error 0.39 of the bound).
- `tests/python/test_kv_cache.py` checks the same against float64 (worst
  0.09) and against torch's `scaled_dot_product_attention(enable_gqa=True)`
  (rtol 1e-5).
- One of its tests checks that a prompt's causal SDPA followed by decode
  steps from the cache equals causal attention over the whole sequence.

**Speed.** GPU time per call, 100 calls per batch, at Qwen2.5-0.5B's layout
(14 query heads over 2 KV heads, dim 64):

| filled positions | 128 | 1,024 | 4,096 | 16,384 |
|---|---:|---:|---:|---:|
| `attend`, one layer | 19 µs | 119 µs | 467 µs | 1,847 µs |

Appending one token's keys and values costs 6 µs.

The time grows linearly with length, and one workgroup per (batch, head)
is 14 of the card's 40 compute units at batch 1. The rest of the GPU is
idle no matter how long the cache is, which is what the next section fixes.

### Splitting the cache across workgroups ("flash decoding")

`g2-attn-decode-split` gives a workgroup a *slice* of the positions and
returns that slice's partial softmax unnormalized — its maximum `m`, its
`l = sum exp(s - m)` and its `o = sum exp(s - m) v` — and
`g2-attn-decode-merge` combines the slices per row with the usual online
softmax rescaling, `w_s = exp(m_s - max_s m_s)`. Inside a slice the
arithmetic is the single-workgroup kernel's, in the same order, so the
split changes only how the slices are combined.

`KVCache::attend` picks the number of slices itself: enough workgroups to
fill the card (256, measured), never fewer than 256 cache positions each,
at most 256 slices (the merge holds one weight per thread). One slice means
the original kernel and no extra dispatch, so short caches are untouched.

GPU time per `attend` call, same layout, by number of slices — each number
includes about 0.12 ms of submit and fence overhead, visible as the floor
at 256 positions:

| filled positions | 1 slice | 2 | 4 | 8 | 16 | 32 | chosen |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | **0.122** | 0.125 | 0.128 | 0.137 | 0.127 | 0.129 | 1 |
| 1,024 | 0.270 | 0.230 | **0.172** | 0.194 | 0.203 | 0.193 | 4 |
| 4,096 | 0.568 | 0.361 | 0.324 | 0.356 | **0.307** | 0.334 | 16 |
| 16,384 | 1.984 | 1.317 | 1.120 | 1.054 | **0.868** | 0.876 | 19 |

End to end on Qwen2.5-0.5B, int8, sampled, measured by forcing one slice
and rebuilding:

| prompt | one slice | split | |
|---|---:|---:|---|
| 40 tokens | 163.4 tokens/s | 163.4 | unchanged: one slice is chosen |
| 3,030 tokens | 69.0 | **103.9** | 1.51x (GPU wait 12.4 -> 6.6 ms/token) |
| 8,030 tokens | 33.8 | **68.0** | 2.01x (27.0 -> 11.8 ms/token) |

The split costs one extra dispatch per layer (273 -> 297 per token), which
is why it is not used when the cache is short.

**Tests.** `split_decode_attention_matches_float64_at_every_split` runs a
1,000-position cache at 1, 2, 3, 7, 16, 64 and 256 slices against the same
float64 reference and the same bound as the unsplit path (worst 0.043 of
the bound, against 0.393 unsplit — slicing shortens each sum). The
benchmark that produced the table above is
`cargo test --release -p grilly-infer --test kv_cache -- --ignored --nocapture split_bench`.

## 4. The decode loop, and the acceptance run

```python
from grilly.infer.models import LlamaConfig, LlamaForCausalLM
from grilly.infer.safetensors import load_file
from grilly.infer.quantization import quantize_, Int8WeightOnlyConfig

model = LlamaForCausalLM(config)                       # a LlamaConfig
model.load_state_dict(load_file("model.safetensors"), assign=True, strict=False)
model.tie_weights()
quantize_(model, Int8WeightOnlyConfig())
ids = model.generate(input_ids, max_new_tokens=128, do_sample=True,
                     temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.1)
```

**`generate()`** (`grilly.infer.generation.GenerationMixin`) has Hugging
Face's signature and defaults:
- **Inputs:** `inputs` or `input_ids=`, and a `generation_config` or its
  fields as keyword arguments.
- **Length:** `max_length` defaults to 20; `max_new_tokens` overrides it.
- **Sampling:** `do_sample`, `temperature`, `top_k`, `top_p`,
  `repetition_penalty`.
- **Stopping:** `eos_token_id` ends a row; `pad_token_id` defaults to the
  EOS id.
- **Streaming:** `streamer` gets HF's `put`/`end` calls, so
  `transformers.TextStreamer` works.
- **Randomness:** grilly's global generator, seeded by `manual_seed`.
- **Refused, by name:** beam search, padded batches, unknown keyword
  arguments.

How a step runs:
- The prompt runs once.
- Each step then runs one token against the KV cache.
- The token is sampled on the device.
- Only the sampled ids are read back, for EOS and the streamer.
- The next step's input stays on the GPU.

**The model** (`grilly.infer.models.LlamaForCausalLM`) is a Llama-family
decoder built from grilly's torch ops:
- RMSNorm; `nn.RMSNorm` and `F.rms_norm` are now in grilly with torch's
  names;
- rotary embeddings;
- grouped-query attention;
- a SwiGLU MLP;
- q/k/v bias for Qwen2.

Its parameter names are Hugging Face's, so checkpoints load with no
remapping. It is the machinery's own model: a safetensors file loads and
generates with no adapter installed. Reading `config.json` into a
`LlamaConfig` is left to the adapter.

**Tests against transformers' own Qwen2** (`tests/python/test_generation.py`,
the same random weights in both, float32):
- logits within 1e-5 of the logit scale (measured 2e-7), over a full
  prompt and token by token through the cache, tied and untied;
- greedy generation, with and without a repetition penalty, and EOS
  stopping with padding in a batch of two: identical token sequences;
- seeded sampling reproducible, the streamer protocol, a quantized model,
  and the refusals.

### The acceptance run (`tools/generate.py`), RX 6750 XT

Qwen2.5-0.5B-Instruct: loaded from its safetensors, quantized to int8, then
a chat prompt sampled with the model's own `generation_config.json`
(temperature 0.7, top_k 20, top_p 0.8, repetition penalty 1.1).

- **Correct:** greedy float32 generation is identical to transformers for
  32 new tokens, both without a repetition penalty and with the model's
  1.1. The comparison passes identical settings to both. transformers'
  `from_pretrained` also loads `generation_config.json`, so a bare
  `do_sample=False` there still applies the 1.1.
- **Coherent:** "The sky is blue because of the scattering of sunlight by
  tiny particles in the Earth's atmosphere, ..." The model's physics is its
  own.

| | |
|---|---:|
| decode (int8, sampled) | **168.4 tokens/s** (5.9 ms per token) |
| prompt of 40 tokens + first token | 74 ms |
| peak dedicated VRAM while generating | **1.24 GiB** |
| peak while loading and quantizing | 2.07 GiB |
| weights + KV cache in persistent memory | 1.19 GiB |

**Where a token's 5.9 ms goes:** 273 dispatches, 3.7 ms of GPU wait, and the
rest host-side recording. The loop is still host-bound, so the lever is
still dispatches per token.

## 6. The fused decode step

The first measurement of the loop said the same thing the dispatch-cost
measurement did (docs/submission.md): 1,145 dispatches for one token, 16.8
ms, 5.8 ms of it GPU. Per layer that was 47 dispatches — 16 for the rotary
embeddings, 12 for two RMSNorms, 7 for the transposes and slices around
attention, 6 for the MLP, 4 projections, 2 residual adds.

Four kernels and one weight layout take a layer to 11:

| | ops before | after |
|---|---:|---:|
| RMSNorm, and the residual add before it | 6 + 1, twice | **1**, twice (`g2-rms-norm` adds the residual and returns the sum) |
| rotary embedding of q | 7 | **1** (`g2-rope`, straight from the projection's columns) |
| rotary embedding of k, into the cache | 7 + 1 append | **2** (`KvCache::append_rotary`) |
| q, k, v projections | 3 | **1** (one int8 weight holding all three) |
| gate and up projections, silu, multiply | 4 | **2** (one weight, then `g2-swiglu`) |

None of it is new arithmetic. `g2-rope` and `g2-swiglu` call the same op
functions in the same order as the ops they replace and are bit-exact
against them (`tests/python/test_fused_ops.py`,
`crates/grilly-infer/tests/fused_ops.rs`). A fused int8 weight is exact
because the scales are per row: no row's dot product changes, and
`Int8Weight::cat` is asserted bit-for-bit against the separate weights.
`g2-rms-norm` is the one exception — its sum of squares is compensated
rather than reassociated — so it is measured against float64 instead:
within 2.44 ulp of a 4 ulp bound derived from the operations it performs.

Two things follow from that, and both are in the tests:

- **the difference at the logits is the network's, not the kernel's.** The
  norm differs from the composed ops by ~1 ulp; on a random 3-layer model
  that becomes 1e-4 at the logits. So the paths are not compared to each
  other under a fixed epsilon; both are compared against a float64
  reference of the same model, and the fused path must be no less accurate
  than the composed one (measured 7.3e-7 against 5.6e-7 — the same order).
- **the property a user sees is the tokens.** Greedy generation from the
  real Qwen2.5-0.5B is still identical to transformers for 32 tokens, with
  and without a repetition penalty.

The fused path runs only when no gradient is needed. With autograd on, the
same modules run as plain grilly ops, which carry gradients; the fused
kernels have no backward and do not need one.

**What it cost:** the fused q/k/v and gate/up weights are copies, so the
int8 weights are held twice for the projections that fuse — 1.04 GiB while
generating became 1.24. Quantizing straight from the checkpoint into the
fused layout would remove both that and the 2.98 GiB loading peak below;
that is the next step.

### 7. Quantizing while loading

`load_state_dict(load_file(path), assign=True)` followed by `quantize_`
holds three copies of the weights at once: the model's own (from its
constructor), the checkpoint's, and the int8 result.
`quantization.load_quantized_` holds one tensor at a time instead — it
reads the file in offset order, quantizes each weight that `filter_fn`
picks as it arrives, and lets the float32 tensor go before reading the
next. The int8 values are the same, bit for bit
(`tests/python/test_load_quantized.py`), because it is the same kernel on
the same values.

A weight the checkpoint does not carry is left alone, so a tied output
projection is restored and quantized afterwards:

    load_quantized_(model, path, Int8WeightOnlyConfig(), strict=False)
    model.tie_weights()
    quantize_(model, Int8WeightOnlyConfig())   # only what is still float32

That last line matters more than it looks: Qwen2.5-0.5B ties its 151,936-row
output projection to the embedding, and leaving it in float32 costs
**110.7 tokens/s against 168.4**.

**Measured** (Qwen2.5-0.5B, the shim's own high-water mark of persistent
bytes, `submit_stats()["persistent_peak"]`):

| | peak | alive after |
|---|---:|---:|
| load float32, then quantize | 2.98 GiB | 0.97 GiB |
| quantize while loading | **2.37 GiB** | 0.97 GiB |

and, inside the second number, where it now goes:

| after | alive | peak so far |
|---|---:|---:|
| the model's constructor | 1.84 GiB | 2.35 GiB |
| streaming the checkpoint | 1.35 GiB | 2.37 GiB |
| quantizing the tied head | 0.97 GiB | 2.37 GiB |

**The peak is now the model's own constructor, not the loading.** Building
`LlamaForCausalLM` allocates and initializes 1.84 GiB of float32
parameters that the checkpoint is about to overwrite, and the extra 0.5 GiB
on top is the untied `lm_head` weight, allocated and then freed when
`tie_weights` points it at the embedding. Streaming cannot help with
either: both happen before it runs.

What would is constructing the model without materializing weights, which
is what `from_pretrained` needs anyway (torch spells it
`with torch.device("meta")`, accelerate `init_empty_weights`). That is an
open decision for the Hugging Face adapter, not something to slip in
silently here.

The test asserts the peak, not just the values: it measures both paths'
high-water marks and requires the streaming one to stay within the
quantized weights plus a couple of float32 tensors, while the two-step
path exceeds it by a whole float32 copy. A memory claim that is only a
sentence in a document is not a claim.
