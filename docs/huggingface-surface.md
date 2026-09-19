# The torch surface Hugging Face uses

**Status: measurement only.** It feeds the `grilly-huggingface` plan (CLAUDE.md
§4), which is proposed after the current task list, per the owner. ONNX is
out of grilly2's plan.

## How this was measured

`tools/torch_surface.py --grilly`, run from the repo's environment, where
grilly is importable. The tool skips paths containing `site-packages`, so
the packages were copied out first.

It counts module-level torch names only (`torch.cat`,
`torch.nn.Conv2d`), not tensor methods (`x.view`). Names used only in type
annotations count too; `torch.LongTensor` and `torch.FloatTensor` mostly
appear there, and still have to exist for modules to import.

Versions: transformers 5.5.1, safetensors 0.7.0, accelerate 1.15.0.

A first pass by a delegated agent reported far fewer resolving names (33
of 208 for the model set). That pass ran where grilly did not resolve:
CLAUDE.md trap 1. The numbers below are from a rerun and were
spot-checked. `torch.cat`, `torch.einsum`, `torch.no_grad` and
`torch.utils.checkpoint.checkpoint` all resolve.

## Results (2026-09-19)

| scanned | torch names | provided by grilly |
|---|---:|---:|
| model set: llama, mistral, gpt2, bert, t5, vit, siglip, whisper, qwen2, plus modeling_utils, activations, pytorch_utils, modeling_attn_mask_utils, cache_utils, generation/ | 208 | 100 |
| all of transformers | 614 | 182 |
| safetensors | 23 | 16 |
| accelerate | 362 | 59 |

### Missing in the model set, grouped (file counts in parentheses)

- **Legacy tensor-type names** (mostly annotations): `LongTensor` (17),
  `FloatTensor` (17), `BoolTensor` (3).
- **dtypes and type info:**
  - `finfo` (8);
  - `uint16`, `uint32`, `uint64`, `float8_e4m3fn`, `float8_e5m2`;
  - `set_default_dtype`, `get_default_dtype`, `get_default_device`,
    `contiguous_format`, `is_floating_point` (2).
- **Loading:** `torch.load`, `as_tensor`, `clone`. safetensors also
  needs `frombuffer`, `strided`, `complex64`, `uint16`/`32`/`64` and
  `__version__`.
- **Tensor math:**
  - `isin` (4), `tril` (3), `triu`, `unique` (2), `unique_consecutive`,
    `take_along_dim` (2), `std` (2), `repeat_interleave` (2),
    `nonzero` (2);
  - `flip`, `eye`, `erf`, `expm1`, `meshgrid`, `reshape`, `addmm`,
    `baddbmm`, `randperm`;
  - `multinomial` (2; generation's sampling).
- **Audio (whisper):** `stft`, `hann_window`.
- **nn.functional:** `pad` (2), `interpolate` (2), `softplus`, `mish`,
  `logsigmoid`, `l1_loss`.
- **nn modules:**
  - losses: `CrossEntropyLoss` (4), `MSELoss` (3),
    `BCEWithLogitsLoss` (3), `BCELoss`;
  - convolutions: `Conv1d` (2), `Conv2d` (3), `Conv3d`,
    `ConvTranspose1d`, `ConvTranspose2d`;
  - norms: `GroupNorm`, `BatchNorm1d`/`2d`/`3d`;
  - activations: `ReLU6`, `PReLU`, `LeakyReLU`, `Hardswish`;
  - other: `PixelShuffle`, `MultiheadAttention` (2).
- **Infrastructure:**
  - `inference_mode`, `jit.is_tracing` (2), `vmap`;
  - `compile` (2), `compiler.allow_in_graph` (2),
    `_dynamo.mark_static_address` (2);
  - `distributed.is_available` / `is_initialized` / `get_world_size`
    and `all_reduce`;
  - `cuda.*` (streams, graphs, memory stats), `xpu.*`, `mps.*`,
    `backends.*`, `Stream`, `distributions.*`, `version.hip`.

### Across all of transformers, the most-used missing names

`FloatTensor` (575 files), `LongTensor` (521), `nn.Conv2d` (201),
`F.pad` (191), `F.interpolate` (181), `nn.CrossEntropyLoss` (164),
`finfo` (121), `nn.Conv1d` (101), `nn.BCEWithLogitsLoss` (94),
`nn.MSELoss` (88), `jit.is_tracing` (79), `BoolTensor` (76),
`nn.BatchNorm2d` (63), `F.one_hot` (62), `linspace` (57),
`nn.GroupNorm` (51), `repeat_interleave` (45), `meshgrid` (44),
`as_tensor` (44), `F.softplus` (35), `IntTensor` (34),
`nn.ConvTranspose2d` (32), `get_default_dtype` (32), `tril` (26),
`nonzero` (25), `F.grid_sample` (24), `nn.AdaptiveAvgPool2d` (24),
`eye` (24).

### accelerate

Of the 303 names it uses that grilly lacks, almost all are
infrastructure:
- `distributed.*` (99: FSDP, distributed checkpointing, communication);
- other backends' device modules (`xpu`, `mlu`, `cuda`, `sdaa`, `musa`,
  `neuron`, `npu`, `hpu`: about 120 together);
- `profiler` (7) and `backends` (5).

## Observations for the plan (not decisions)

- **Transformers models are mostly within reach of core ops.** The biggest
  gaps are cheap names: the legacy tensor types, `finfo`, dtype defaults.
  After those come ordinary math grilly lacks (tril/triu, isin, pad,
  interpolate, one_hot, repeat_interleave), the loss modules, and the
  conv/norm family, which v1's bindings_conv and bindings_normalization
  cover.
- **Checkpoints need half precision.** HF weights are usually float16 or
  bfloat16. grilly computes in float32, so loading needs those dtypes at
  least as storage with conversion. Which half-precision support grilly2
  offers is a decision for the owner.
- **accelerate is mostly out of scope** for a single-GPU backend (CLAUDE.md
  §2: "Scope is not shortfall").
