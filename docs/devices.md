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
