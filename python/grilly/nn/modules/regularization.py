"""``torch.nn.Dropout``."""

from __future__ import annotations

from .. import functional as F
from .module import Module


class Dropout(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout(input, self.p, self.training, self.inplace)

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"
