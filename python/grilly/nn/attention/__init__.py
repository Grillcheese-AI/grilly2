"""grilly.nn.attention, like torch.nn.attention (its ``flex_attention``
submodule; SDPA itself is ``grilly.nn.functional.scaled_dot_product_attention``)."""

from . import flex_attention

__all__ = ["flex_attention"]
