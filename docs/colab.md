# grilly2 on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Grillcheese-AI/grilly2/blob/main/notebooks/colab.ipynb)

`notebooks/colab.ipynb` runs grilly2 on a Colab NVIDIA GPU. It is the quickest
way to try grilly2 on real GPU hardware without installing anything, and the
one place its test suites run on an NVIDIA card.

grilly2 is built for the consumer AMD and Intel GPUs that torch will not
train on, and for Windows. On NVIDIA it uses the same Vulkan path as
everywhere else, not CUDA. So Colab shows whether the code is correct on a
third vendor's driver. It is not a benchmark of the hardware grilly2
targets.

## Running it

1. Open the notebook with the badge above.
2. **Runtime → Change runtime type → T4 GPU.** The notebook asks for a T4 by
   default. Other GPUs should work the same way, but only the T4 has been
   measured.
3. **Runtime → Run all.**

The steps, in order:

| step | what it does |
|---|---|
| 1. Vulkan | registers Colab's NVIDIA driver with the Vulkan loader ([below](#why-colab-needs-step-1)) |
| 2. SDK | downloads the LunarG Vulkan SDK (367 MB) for `vulkaninfo` and a source build |
| 3. install | `pip install grilly2[huggingface]`, or builds a branch from source |
| 4. check | runs `grilly-doctor`, then compares a small forward and backward pass with torch on CUDA |
| 5. generate | loads Qwen2.5-0.5B-Instruct from its safetensors checkpoint, quantizes it to int8, and samples up to 128 tokens |
| 6. tests | runs the Python and Rust suites on the GPU (optional) |

On a CPU-only runtime, step 1 installs Mesa's lavapipe instead and sets
`ALLOW_CPU_VULKAN=1`. Everything still runs, but slowly.

## Settings

The first code cell holds four settings:

| setting | default | meaning |
|---|---|---|
| `SDK_VERSION` | `"1.4.363.1"` | LunarG SDK release; the newest is listed at `https://vulkan.lunarg.com/sdk/latest/linux.txt` |
| `BUILD_FROM_SOURCE` | `False` | build grilly2 from `GRILLY_REF` instead of installing the PyPI wheel |
| `GRILLY_REF` | `"main"` | branch, tag or commit to build and to take the tests from |
| `RUN_TESTS` | `False` | run step 6 |

### Building a branch

To try work that has not been released, set `BUILD_FROM_SOURCE = True` and
`GRILLY_REF` to the branch name, then run all. The notebook installs Rust
and maturin, clones the repository at that ref, and builds a wheel against
the SDK from step 2.

Measured on a T4 runtime (2 vCPUs, 12 GB RAM), building the
`fix/flash-shared-48k` branch:

| | time |
|---|---:|
| `maturin build --release` | 1.2 min |
| the whole notebook, build included, tests skipped | 4.3 min |
| step 6: the Python suite, including pinning torch | 2.1 min |
| step 6: the Rust suite, compile and run | 6.5 min |

Any ref builds, including ones that predate SDK 1.4.363. That SDK moved the
Vulkan loader from `lib/` to `lib/VulkanLoader/lib/`, and step 2 adds that
directory to `LIBRARY_PATH` so older build scripts still link.

### Running the tests

`RUN_TESTS = True` runs both suites on the GPU with nothing skipped. CI skips
a few tests by name on lavapipe; here they all run.

The torch reference is pinned to the version CI uses, `torch==2.10.0` (CPU),
with `transformers==5.5.1`. Pinning replaces Colab's torch in that runtime.
It also uninstalls Colab's `torchvision` and `torchaudio`, which are built
for the newer torch and would break `transformers`' imports. Run step 6
last, or restart the runtime afterwards.

## NVIDIA status

**grilly2 0.1.0 cannot run attention on NVIDIA GPUs.** Its flash-attention
kernels declare 52–54 KiB of shared memory per workgroup. NVIDIA's Vulkan
driver allows 48 KiB (`maxComputeSharedMemorySize` = 49,152 bytes), so the
first prefill of any model loses the device (`VK_ERROR_DEVICE_LOST`). AMD and
lavapipe allow 64 KiB, which is why neither development nor CI caught it.

The source fixes it after 0.1.0. A test now holds every kernel to 48 KiB. Until
a release carries the fix, set `BUILD_FROM_SOURCE = True`. With the 0.1.0
wheel on an NVIDIA GPU, step 5 stops with that message rather than taking
the runtime down.

With the fix, on a T4 (driver 580.82.07, Ubuntu 24.04, Python 3.13):

| check | result |
|---|---|
| `grilly-doctor` | device `Tesla T4`, a kernel ran |
| forward and backward vs torch on CUDA | loss `0.1237742` on both; max \|dx\| difference `3.6e-11` |
| int8 Qwen2.5-0.5B, sampled | 59 tokens in 2.70 s (first call: includes the prompt and building every pipeline) |
| Python suite | 1,179 passed, 8 failed |
| Rust suite | 120 passed, 1 failed |

These failures happen only on NVIDIA and are still open:

- `test_logits_process.py::test_an_allowed_list_holds_at_every_step`, all
  four sampled variants: a sampled token falls outside the allowed list.
  Greedy decoding passes.
- `test_load_quantized.py`: `test_streaming_reports_missing_and_unexpected_keys`
  and `test_streaming_leaves_unquantized_weights_alone`. Both fail with an
  embedding index out of range.
- `test_es.py::test_the_gradient_path_reproduces_step`.
- `test_float_precision.py::test_the_rest_are_within_the_measured_bound[sqrt-1]`.
- Rust `transcendental_unary_ops_within_2_ulp`. This one is also skipped on
  lavapipe, because Vulkan bounds `log()` by absolute error rather than
  ULPs.

Until the sampling failures are fixed, treat sampled generation on NVIDIA
as untrusted. Greedy generation is fine.

## Why Colab needs step 1

Colab mounts the NVIDIA driver's user-space libraries in `/usr/lib64-nvidia`,
including its Vulkan driver, but registers it with nothing:

- `/etc/vulkan/icd.d/` is empty, so the Vulkan loader finds no driver and
  instance creation fails with `VK_ERROR_INCOMPATIBLE_DRIVER`.
- With a driver manifest alone, the loader logs `Could not get
  'vkCreateInstance' via 'vk_icdGetInstanceProcAddr'`. NVIDIA's Vulkan driver
  starts up through glvnd's EGL, and only Mesa's EGL vendor is registered.

Two JSON files fix both. The same fix works in any container that mounts
the NVIDIA driver without registering it:

```sh
# the driver manifest, for the Vulkan loader
mkdir -p /etc/vulkan/icd.d
cat > /etc/vulkan/icd.d/nvidia_icd.json <<'EOF'
{"file_format_version": "1.0.1",
 "ICD": {"library_path": "/usr/lib64-nvidia/libGLX_nvidia.so.0", "api_version": "1.3.0"}}
EOF
# the EGL vendor the Vulkan driver starts up through
mkdir -p /usr/share/glvnd/egl_vendor.d
cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{"file_format_version": "1.0.0",
 "ICD": {"library_path": "/usr/lib64-nvidia/libEGL_nvidia.so.0"}}
EOF
```

Point both at wherever the container mounts the driver. The notebook looks
the libraries up on `LD_LIBRARY_PATH` rather than assuming the path.

Do not `apt install libnvidia-gl-*` instead. It installs a second copy of the
driver's user-space libraries, and their version need not match the kernel
module that is actually running.

## When something fails

- **Run `grilly-doctor`.** It reports the interpreter, the install and the
  device, and runs a kernel.
- **Check what the loader sees.** `vulkaninfo --summary` (from the SDK, step
  2) lists the devices. `VK_LOADER_DEBUG=error,warn,driver` shows which
  driver manifests the loader found and why it rejected them.
- **Find the kernel behind `VK_ERROR_DEVICE_LOST`.** Set
  `GRILLY_LAUNCH_BLOCKING=1`. grilly then waits after every kernel, so the
  error surfaces at the Python call that launched the faulting kernel
  instead of at the next read.
- **Turn on the validation layer** that ships with the SDK. It names invalid
  usage that a driver may otherwise run and crash on. That is how the
  shared-memory limit above was found:

  ```sh
  SDK=/content/vulkan/1.4.363.1/x86_64
  VK_ADD_LAYER_PATH=$SDK/share/vulkan/explicit_layer.d \
  VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  LD_LIBRARY_PATH=/usr/lib64-nvidia:$SDK/lib \
  python your_script.py
  ```

- **Which Vulkan loader is in use.** The notebook's Python kernel keeps the
  loader it started with, Ubuntu's `libvulkan.so.1`. Programs started from
  the notebook use the SDK's. Step 4 prints which one the kernel has.

Report a failure at <https://github.com/Grillcheese-AI/grilly2/issues> with
the `grilly-doctor` output and the notebook cell that failed.
