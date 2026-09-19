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
| `CMAKE_GENERATOR` | CMake generator, e.g. `Visual Studio 17 2022` when a newer Visual Studio is installed without its C++ tools |

The GPU tests need a Vulkan 1.2 device. The op tests compare every op against
torch on CPU, so they also need torch installed for `GRILLY_PYTHON`.
