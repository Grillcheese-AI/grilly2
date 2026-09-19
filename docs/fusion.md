# Kernel fusion

**Decision (owner, 2026-09-19): F2.** grilly2 keeps grilly v1's fusion design
and machinery and generates the kernels itself. The entry point is torch's
name for it, `torch.compile`. The options are in `docs/flex-attention.md`
("v1's attention kernels and shader fusion").

## What is kept from v1

v1's `ShaderFusionEngine` (`cpp/src/shader_fusion.cpp`) did four things,
and grilly2 keeps all four:
- generate one GLSL compute shader for a graph of elementwise ops, with
  intermediates in registers;
- compile it at runtime with shaderc;
- register it in `PipelineCache`;
- reuse it by a key.

The shaderc is the one v1 already links from the Vulkan SDK. The shim's
`gc_compile_shader` compiles with the build's glslc settings (`-O`,
Vulkan 1.2) and calls v1's `PipelineCache::loadSPIRV`. A generated kernel
then goes through the same dispatch spine as every other kernel:
registry, pipeline cache, bind, submit.

## What is different

v1's code generator knows five ops, and it silently skips any other op in
a graph. grilly2 generates calls to `g2_unary` and `g2_binary`, the
functions in `g2-common.glsl` that the unfused `g2-unary` and `g2-binary`
kernels also run. Every elementwise op therefore has one definition, and
the accuracy work on it applies to fused code too: Cephes exp and tanh,
sin and cos to 1 ULP, square-and-multiply pow, torch's NaN rules.

Every intermediate is `precise`, so the compiler cannot contract or
reassociate across ops. A fused graph therefore returns what the same
ops return one kernel at a time, bit for bit. The exception is a NaN's
sign and payload, which IEEE 754 leaves unspecified. Tested in
`crates/grilly-torch/tests/fusion.rs` for every op, with broadcasting and
several outputs.

Scalars such as a learning rate or a dropout scale are runtime
parameters, not literals. A graph compiles once, whatever values it
later runs with. Kernels are cached per device by their generated
source.

## Layers

1. `grilly_core::fusion::Graph`: inputs, parameters, unary and binary
   nodes, and one or more outputs. `run(inputs, params, shape)` launches
   one kernel.
2. `torch.compile` (next). Inside a compiled function, float elementwise
   ops build a lazy graph. The first use that needs data (a matmul, a
   reduction, `.item()`) compiles and runs the graph as one kernel.
3. Autograd (after that). The graph's derivatives are themselves an
   elementwise graph, so the backward is a fused kernel too, followed by
   sums over broadcast dims.

## Measured

On the RX 6750 XT, the tanh-approximation GELU over 16M elements takes
28.0 ms as 8 separate kernels and 3.1 ms as 1 fused kernel, with the same
result.
