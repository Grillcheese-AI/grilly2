# grilly2

The torch API on Vulkan, for the GPUs torch will not train on — consumer AMD and
Intel cards, and Windows. Early work; see `CLAUDE.md` for the plan and
`docs/boundary.md` for how the core talks to the GPU.

## Building

Requirements:

- Rust (stable, MSVC target on Windows);
- the LunarG Vulkan SDK, with `VULKAN_SDK` set;
- a C++17 compiler.

```
git clone --recursive https://github.com/Grillcheese-AI/grilly2
cargo test
```

`--recursive` fetches VulkanMemoryAllocator, the one external dependency of
the Vulkan layer; `git submodule update --init --recursive` does it after
the fact. Nothing else is needed — no CMake, no second project, and **no
grilly v1 checkout**: the Vulkan device, buffer pool, pipeline cache and
command batch live in `crates/grilly-sys/vulkan/`, and every kernel is
compiled from `crates/grilly-core/shaders/` at build time.

Useful variables:

| variable | effect |
|---|---|
| `VMA_INCLUDE_DIR` | VulkanMemoryAllocator include directory; default is the submodule |
| `GRILLY_SHADER_DIR` | an extra SPIR-V directory, loaded after grilly2's own, so a kernel can be replaced without a rebuild |
| `GRILLY_PYTHON` | interpreter with torch and numpy, used by the op tests as the reference; default `python` |
| `GRILLY_VERBOSE` | set to 1 to print the device bring-up log |
| `GRILLY_LAUNCH_BLOCKING` | set to 1 to submit and wait after every kernel launch (as `CUDA_LAUNCH_BLOCKING`), for diagnosis |

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

`python tools/generate.py --max-new-tokens 64` is that run: Qwen2.5-0.5B-Instruct
from the local Hugging Face cache, int8, sampled, **142–168 tokens/s** on an
RX 6750 XT with the GPU otherwise idle, 1.24 GiB of VRAM while generating, and
greedy output identical to transformers.

The spread is run-to-run, not noise in the measurement: a 64-token decode is
about 0.4 s, short enough that the card's clock state decides the number. Quote
the range, not a best run.

A decode step spends much of its token in the host, deciding what to
launch. `grilly.graphed` records a step's GPU commands once and replays
them, which removes that cost without changing the commands or the
arithmetic — measured at 1.6-2.2x on a recurrent decode step, bit-exact
against running it eagerly (`docs/capture.md`):

```python
step = grilly.graphed(model.step)      # records on the second call with a shape
for t in range(tokens):
    y, state = step(x_t, state)        # every later one replays
```

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

## Beyond torch

The torch API is the core. Anything torch has no name for lives in its own
module, which is where domain vocabulary is allowed — `grilly.vsa` is the
first:

```python
from grilly import vsa

role, filler = vsa.random_discrete(80, 128), vsa.random_discrete(80, 128)
bound = vsa.bind(role, filler)                 # per-block circular convolution
assert vsa.similarity(vsa.unbind(bound, role), filler) == 1.0
```

Sparse block codes (IBM NVSA) and, in `grilly.vsa.packed`, binary
hypervectors 32 bits to a word. They are different algebras over different
dtypes and do not accept each other's vectors.

## Coming from grilly v1

grilly2 replaces v1 in place: same distribution, same import name. v1's
import paths still work, through aliases that keep v1's signatures and name
where each thing moved to. v1's *parallel framework* — its own autograd,
`nn` and optimizers — is not reimplemented, because the torch API is the
replacement; those names raise and say what to use. `docs/v1-migration.md`
is the table.

To see which torch symbols a codebase uses and how many grilly provides:

```
python tools/torch_surface.py path/to/project --grilly
```
