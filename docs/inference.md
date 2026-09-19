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

RX 6750 XT, the file already in the OS cache.

| | |
|---|---:|
| file | 290 tensors, all BF16, 0.92 GiB |
| weights as float32 | 494M parameters, 1.84 GiB |
| `load_file` | 1.7 s (1.1 GiB/s of float32 uploaded) |
| dedicated VRAM after the load | **3.00 GiB** |
| shared (system RAM) after the load | 0.51 GiB above the baseline |

The 1.16 GiB of VRAM above the weights' size is v1's buffer pool. It
rounds every device allocation up to a power of two below 256 MB. Summing
those rounded sizes over Qwen's tensors predicts 2.97 GiB, which matches
the measurement.

The largest waste is in the MLP weights:
- `gate_proj` and `up_proj`: 4864×896 float32 is 17.4 MB, held in a 32 MB
  bucket; 0.72 GiB across the model;
- `down_proj`: 0.36 GiB.

The 0.51 GiB of system RAM is the staging buffer for the 545 MB embedding.
It is larger than 256 MB, so the pool keeps it at its exact size after the
upload.

Quantization will not remove this. An int8 `gate_proj` is 4.4 MB and would
sit in an 8 MB bucket. Rounding suits short-lived activations, which are
reused. It is the wrong policy for weights, which are allocated once and
held for the whole run.
