"""Deterministic hashing, for reproducible hypervector seeding.

A VSA system needs the same symbol to get the same hypervector in every
process, on every machine, for as long as the data lives. A seeded RNG does
not give you that across numpy versions; a hash stream does.

    from grilly.vsa import hashing

    seed = hashing.stable_u32("cat", domain="my-corpus")
    v = hashing.bipolar_from_key("cat", 10240)      # (10240,) of +1 / -1

v1's ``grilly.utils.stable_hash``, ported unchanged — the digests have to
match or vectors written by v1 stop resolving.

**BLAKE3 decides the bytes.** With the `blake3` package installed the
digests are canonical. Without it this falls back to `hashlib.blake2s` and
warns once, and those digests are *different*: a store written under one
and read under the other will not line up. Install `blake3` if the vectors
outlive the process.
"""

from __future__ import annotations

import hashlib
import warnings

__all__ = [
    "bipolar_from_key", "digest", "stable_bytes", "stable_u32", "stable_u64", "using_blake3",
]

try:
    from blake3 import blake3 as _blake3

    _USING_BLAKE3 = True
except Exception:  # pragma: no cover - depends on what is installed
    _blake3 = None
    _USING_BLAKE3 = False

_warned = False


def _warn_once() -> None:
    global _warned
    if not _warned:
        warnings.warn(
            "blake3 not installed; falling back to hashlib.blake2s for stable hashing. Its "
            "digests differ from blake3's, so vectors seeded under one will not match the "
            "other. Install blake3 for canonical outputs: `pip install blake3`.",
            RuntimeWarning,
            stacklevel=3,
        )
        _warned = True


def using_blake3() -> bool:
    """Whether the canonical hash is in use, rather than the fallback."""
    return _USING_BLAKE3


def _to_bytes(x) -> bytes:
    if isinstance(x, bytes):
        return x
    if isinstance(x, str):
        return x.encode("utf-8", errors="surrogatepass")
    return str(x).encode("utf-8", errors="surrogatepass")


def _join_parts(parts) -> bytes:
    # \x1f, the ASCII unit separator: rare enough in text that joining on it
    # cannot make two different part lists collide
    return b"\x1f".join(_to_bytes(p) for p in parts)


def digest(parts, *, domain: str = "grilly", out_len: int = 32) -> bytes:
    """`out_len` bytes from `parts`, namespaced by `domain`.

    The domain keeps two uses of the same key apart: the same word in two
    corpora gets two unrelated vectors.
    """
    msg = _join_parts([domain] + list(parts))
    if _USING_BLAKE3:
        return _blake3(msg).digest(out_len)
    _warn_once()
    # BLAKE2s tops out at 32 bytes, so a longer request chains counters
    if out_len <= 32:
        return hashlib.blake2s(msg, digest_size=out_len).digest()
    out = bytearray()
    counter = 0
    while len(out) < out_len:
        out.extend(hashlib.blake2s(msg + counter.to_bytes(4, "little")).digest())
        counter += 1
    return bytes(out[:out_len])


def stable_u32(*parts, domain: str = "grilly") -> int:
    """A 32-bit seed from `parts`."""
    return int.from_bytes(digest(parts, domain=domain, out_len=4), "little", signed=False)


def stable_u64(*parts, domain: str = "grilly") -> int:
    """A 64-bit seed from `parts`."""
    return int.from_bytes(digest(parts, domain=domain, out_len=8), "little", signed=False)


def stable_bytes(*parts, domain: str = "grilly", out_len: int = 32) -> bytes:
    """Raw digest bytes from `parts`."""
    return digest(parts, domain=domain, out_len=out_len)


def bipolar_from_key(key: str, dim: int, *, domain: str = "grilly.bipolar"):
    """A `dim`-wide +1/-1 vector for `key`, as a numpy float32 array.

    Straight from the hash stream rather than from a seeded RNG, so it does
    not move when numpy's generators do.
    """
    import numpy as np

    if dim <= 0:
        return np.zeros((0,), dtype=np.float32)
    nbytes = (dim + 7) // 8
    out = bytearray()
    counter = 0
    while len(out) < nbytes:
        out.extend(digest((key, counter), domain=domain, out_len=32))
        counter += 1
    bits = np.unpackbits(np.frombuffer(bytes(out[:nbytes]), dtype=np.uint8), bitorder="little")[:dim]
    return (bits.astype(np.int8) * 2 - 1).astype(np.float32)
