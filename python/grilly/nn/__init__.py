"""grilly.nn, like torch.nn. Modules come with autograd; only
functional ops exist so far."""

from . import functional

__all__ = ["functional"]
