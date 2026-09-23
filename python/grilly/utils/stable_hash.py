"""Deprecated: v1's ``grilly.utils.stable_hash``. Use ``grilly.vsa.hashing``.

Same functions, same digests — the bytes have to match or hypervectors
written by v1 stop resolving. It moved because deterministic hashing is
there to seed hypervectors, which is VSA's business and not torch's.
"""

from __future__ import annotations

from .._v1 import moved
from ..vsa import hashing as _h

__all__ = [
    "bipolar_from_key", "digest", "stable_bytes", "stable_u32", "stable_u64", "using_blake3",
]


def using_blake3() -> bool:
    moved("grilly.utils.stable_hash.using_blake3", "grilly.vsa.hashing.using_blake3")
    return _h.using_blake3()


def digest(parts, *, domain: str = "grilly", out_len: int = 32) -> bytes:
    moved("grilly.utils.stable_hash.digest", "grilly.vsa.hashing.digest")
    return _h.digest(parts, domain=domain, out_len=out_len)


def stable_u32(*parts, domain: str = "grilly") -> int:
    moved("grilly.utils.stable_hash.stable_u32", "grilly.vsa.hashing.stable_u32")
    return _h.stable_u32(*parts, domain=domain)


def stable_u64(*parts, domain: str = "grilly") -> int:
    moved("grilly.utils.stable_hash.stable_u64", "grilly.vsa.hashing.stable_u64")
    return _h.stable_u64(*parts, domain=domain)


def stable_bytes(*parts, domain: str = "grilly", out_len: int = 32) -> bytes:
    moved("grilly.utils.stable_hash.stable_bytes", "grilly.vsa.hashing.stable_bytes")
    return _h.stable_bytes(*parts, domain=domain, out_len=out_len)


def bipolar_from_key(key: str, dim: int, *, domain: str = "grilly.bipolar"):
    moved("grilly.utils.stable_hash.bipolar_from_key", "grilly.vsa.hashing.bipolar_from_key")
    return _h.bipolar_from_key(key, dim, domain=domain)
