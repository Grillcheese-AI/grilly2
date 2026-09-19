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

### Measured, RX 6750 XT

GPU time per call at Qwen2.5-0.5B's shapes (`tools/int8_linear.py`: 200
calls per batch, the shim's wait divided by 200). GB/s counts the weight
bytes read.

| layer (out × in) | rows | float32 µs | int8 µs | int8 GB/s | int8 vs float32 |
|---|---:|---:|---:|---:|---:|
| q/o_proj (896 × 896) | 1 | 24.5 | 6.0 | 135 | 4.1× faster |
| k/v_proj (128 × 896) | 1 | 22.3 | 3.4 | 34 | 6.5× faster |
| gate/up_proj (4864 × 896) | 1 | 73.9 | 20.6 | 213 | 3.6× faster |
| down_proj (896 × 4864) | 1 | 122.4 | 11.7 | 372 | 10.4× faster |
| lm_head (151936 × 896) | 1 | 2745 | 570 | 240 | 4.8× faster |
| gate/up_proj | 8 | 74.8 | 67.2 | 65 | 1.1× faster |
| down_proj | 8 | 124.9 | 49.0 | 89 | 2.5× faster |
| gate/up_proj | 128 | 553 | 886 | 5 | **0.62×, slower** |
| down_proj | 128 | 596 | 714 | 6 | **0.83×, slower** |

- **Decode (one row)** is where int8 pays: 3.6–10.4× faster.
- **Prefill** of a 128-token prompt is slower in int8 (0.62–0.92×). The
  kernel re-reads the weight for each 8-row chunk. A tiled kernel that
  unpacks each int8 tile into shared memory once per 64 rows is the fix.
  It is not built yet.
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

**Speed, not tuned yet.** GPU time per call for one row, measured with
launch blocking:

| vocab | greedy | top_k 20 | top_k 50, top_p 0.8 | top_p 0.9 (top_k off) |
|---:|---:|---:|---:|---:|
| 10,000 | 0.8 ms (with penalty) | 0.14 ms | 0.38 ms | 0.45 ms |
| 151,936 | 11.3 ms (with penalty) | 1.0 ms | 4.0 ms | 5.4 ms |

That is too slow next to a decode step. One workgroup per row is four
waves on one CU of forty, so the passes are bound by memory latency, not
bandwidth. Greedy is slow because grilly's generic argmax scans a row on
one thread. Both are the next pass.
