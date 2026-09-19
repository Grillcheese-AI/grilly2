"""``torch.cuda``: grilly runs on Vulkan, so CUDA is never available. Code
written as ``"cuda" if torch.cuda.is_available() else "cpu"`` selects "cpu",
which grilly accepts as an alias for its Vulkan device."""

__all__ = ["device_count", "is_available", "is_initialized", "manual_seed", "manual_seed_all"]


def is_available() -> bool:
    return False


def device_count() -> int:
    return 0


def is_initialized() -> bool:
    return False


def manual_seed(seed) -> None:
    """Ignored without CUDA, as torch does."""


def manual_seed_all(seed) -> None:
    """Ignored without CUDA, as torch does."""
