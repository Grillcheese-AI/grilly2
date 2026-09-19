# Devices

**Decision (owner, 2026-09-19):** grilly runs on one Vulkan device and
reports it as `device(type='vulkan', index=0)`; `vulkan` is a device type torch
itself recognises. Wherever torch accepts a device, grilly also accepts
`"cpu"` (or `device("cpu")`) **as an alias for that Vulkan device**.

Why: idiomatic torch code chooses its device with
`"cuda" if torch.cuda.is_available() else "cpu"`, and then threads it through
`device=x.device` and `.to(device)`. With `torch.cuda.is_available()` returning
`False` and `"cpu"` accepted, such code runs unchanged on grilly's GPU. The
alternative (rejecting `"cpu"`) would force every consumer to edit its device
selection, which defeats the drop-in goal.

What this means precisely:

- `tensor.device` is always `device(type='vulkan', index=0)`, including for
  tensors created with `device="cpu"`. So `x.device == torch.device("cpu")`
  is `False`, as it would be for a CUDA tensor.
- `x.to("cpu")`, `x.cpu()`, `x.to(x.device)` and `module.to("cpu")` return
  the same object: there is nowhere else to move to.
- `torch.cuda.is_available()` is `False`, `device_count()` is 0, and
  `cuda.manual_seed` is ignored, as torch does without CUDA.
- `"cuda"`, `"mps"` and every other device type raise `NotImplementedError`.

Real CPU tensors (host memory, e.g. for data loading) would be a later
addition, only if a workload needs them.

## The meta device

`device("meta")` is torch's: a tensor with a shape and a dtype and no data.

```python
with grilly.device("meta"):
    model = LlamaForCausalLM(config)      # 494M parameters, 0 bytes allocated
model.load_state_dict(state, assign=True)  # real tensors take their place
model.to_empty()                           # or: uninitialized storage
```

It exists for one job: a model built the ordinary way allocates and
initializes every parameter that a checkpoint is about to overwrite. For
Qwen2.5-0.5B that was 1.84 GiB of float32 weights (plus 0.5 GiB for the
untied `lm_head`, freed a moment later by `tie_weights`), and it was the
peak of the whole load — larger than everything the streaming loader had
carefully avoided. It is also what capped the model size: a 7B fits in
12 GB once quantized to int8, but never gets there if 28 GB of float32
parameters have to exist first.

| Qwen2.5-0.5B, int8 | peak persistent | load time |
|---|---:|---:|
| built, then loaded and quantized | 2.37 GiB | 5.8 s |
| built on meta | **0.97 GiB** | **1.5 s** |

0.97 GiB is the weights themselves, so there is nothing left to remove.
The time is gone because nothing generates 494M random values that are
immediately thrown away.

What works on meta: `empty`, `zeros`, `ones`, `full`, `rand`, `randn` and
module parameters, whether the device comes from `with device("meta")` or
`nn.Linear(..., device="meta")`. Shapes, dtypes, `numel`, `requires_grad`,
`detach`, `state_dict` and `repr` behave; in-place initialization
(`uniform_`, `normal_`, `fill_`) is a no-op, so `reset_parameters` runs
unchanged. Anything needing data raises and names the way out.

The behaviour is torch's, checked against torch in
`tests/python/test_meta.py`: the same `repr`, the same
`NotImplementedError` from `Module.to()` (word for word), and the same
`UserWarning` — not an error — when a checkpoint is copied into a meta
parameter without `assign=True`.

A factory with no meta path **refuses** rather than quietly returning real
storage, because silently allocating under `with device("meta")` would
waste exactly the memory this is for.

One rule for anything that loads a meta-built model: **a parameter nobody
wrote is a tensor with no data.** `grilly.huggingface` checks for leftovers
and names them, so it fails at load time rather than in a kernel later.
