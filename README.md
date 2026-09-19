# grilly2

The torch API on Vulkan, for the GPUs torch will not train on — consumer AMD and
Intel cards, and Windows. Early work; see `CLAUDE.md` for the plan and
`docs/boundary.md` for how the core talks to the GPU.

## Building

Requirements:

- Rust (stable, MSVC target on Windows);
- the LunarG Vulkan SDK, with `VULKAN_SDK` set;
- CMake and a C++17 compiler;
- a checkout of [grilly](https://github.com/Grillcheese-AI/grilly) (v1) beside
  this one, or `GRILLY_V1_DIR` pointing at one. grilly2 links v1's compiled
  Vulkan layer and uses its SPIR-V shaders; it does not modify v1.

```
cargo test
```

The first build compiles v1's `grilly_core_lib` with v1's own CMake project
(a couple of minutes). Useful variables:

| variable | effect |
|---|---|
| `GRILLY_V1_DIR` | v1 checkout to build against |
| `GRILLY_CORE_LIB_DIR` | directory with an already-built `grilly_core_lib`; skips the CMake build |
| `GRILLY_SHADER_DIR` | SPIR-V kernel directory for the default device; default is v1's `shaders/spv` |
| `GRILLY_PYTHON` | interpreter with torch and numpy, used by the op tests as the reference; default `python` |
| `GRILLY_VERBOSE` | set to 1 to print v1's device bring-up log |
| `GRILLY_LAUNCH_BLOCKING` | set to 1 to submit and wait after every kernel launch (as `CUDA_LAUNCH_BLOCKING`), for diagnosis |
| `CMAKE_GENERATOR` | CMake generator, e.g. `Visual Studio 17 2022` when a newer Visual Studio is installed without its C++ tools |

The GPU tests need a Vulkan 1.2 device. The op tests compare every op against
torch on CPU, so they also need torch installed for `GRILLY_PYTHON`.

## Python

```
python -m venv .venv
.venv/Scripts/pip install maturin numpy pytest torch   # torch: reference for tests only
.venv/Scripts/maturin develop --release
.venv/Scripts/python -m pytest
```

```python
import grilly as torch          # or: python -m pytest -p grilly.torch_alias
x = torch.tensor([[1.0, -2.0]])
torch.nn.functional.linear(x, torch.ones(3, 2))
```

To run a Hugging Face checkpoint (see `docs/huggingface.md`):

```python
from grilly.huggingface import AutoModelForCausalLM, AutoTokenizer   # transformers' API
from grilly.infer.quantization import Int8WeightOnlyConfig           # torchao's API

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                             quantization_config=Int8WeightOnlyConfig())
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ids = tokenizer.apply_chat_template([{"role": "user", "content": "Why is the sky blue?"}],
                                    add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(model.generate(ids, max_new_tokens=128)[0]))
```

`python tools/generate.py` is that run: Qwen2.5-0.5B-Instruct from the local
Hugging Face cache, int8, sampled, **166 tokens/s** on an RX 6750 XT, 1.24 GiB
of VRAM while generating, and greedy output identical to transformers.

The machinery underneath takes no Hugging Face files, and works with
`grilly.huggingface` uninstalled (see `docs/inference.md`):

```python
from grilly.infer.safetensors import load_file           # safetensors.torch's API
from grilly.infer.quantization import load_quantized_, Int8WeightOnlyConfig
load_quantized_(model, "model.safetensors", Int8WeightOnlyConfig())  # quantized as it loads
model.generate(input_ids, max_new_tokens=64, do_sample=True, top_p=0.8)
```

Only float32 tensors on the default GPU exist so far, plus torch's meta
device (`with grilly.device("meta")`) for building a model without
allocating the weights a checkpoint is about to replace. Anything torch supports
that grilly does not yet raises `NotImplementedError` or fails to resolve;
nothing silently behaves differently.

To see which torch symbols a codebase uses and how many grilly provides:

```
python tools/torch_surface.py path/to/project --grilly
```
