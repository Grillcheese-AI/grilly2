"""Deprecated: grilly v1's ``backend`` package.

v1 reached the GPU through this: ``_bridge`` for the raw ops, and a
``Vulkan*`` class per domain above it. grilly2 has one dispatch path and
the torch API on top of it (CLAUDE.md §5), so the classes are gone and
``_bridge`` is kept only for the parts that were never torch ops — the VSA
algebras. See docs/v1-migration.md.
"""

from __future__ import annotations

from . import _bridge  # noqa: F401
from .._v1 import v1_module_getattr

_REPLACED = {
    "VulkanCompute": "grilly itself — tensors and ops, not a backend object",
    "VulkanCore": "grilly itself — the device is opened for you",
    "VulkanLearning": "grilly.optim and autograd",
    "VulkanAttention": "grilly.nn.functional.scaled_dot_product_attention",
    "VulkanConv": "grilly.nn.Conv2d",
    "VulkanNormalization": "grilly.nn.LayerNorm and grilly.nn.RMSNorm",
    "VulkanPooling": "grilly.nn.MaxPool2d and grilly.nn.AvgPool2d",
    "VulkanTensorOps": "grilly's tensor methods",
    "VulkanMemory": "grilly's allocator — there is nothing to drive by hand",
    "VulkanFNN": "grilly.nn.Linear and grilly.nn.Sequential",
    "VulkanSNN": "not ported yet (CLAUDE.md §4: grilly-snn)",
    "SNNCompute": "not ported yet (CLAUDE.md §4: grilly-snn)",
    "VulkanVSA": "grilly.vsa.packed — the bipolar/packed algebra",
    "CapsuleMemory": "not ported yet (CLAUDE.md §4)",
    "CapsuleTransformerConfig": "not ported yet (CLAUDE.md §4)",
    "amp": "grilly does not have autocast yet; float32 only",
    "jit": "grilly.compile",
    "buffer_pool": "grilly's allocator — there is nothing to drive by hand",
    "shader_registry": "grilly's dispatch spine — no op names a .spv file",
}

__getattr__ = v1_module_getattr(__name__, _REPLACED)
