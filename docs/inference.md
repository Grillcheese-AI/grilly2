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
