"""Does a Hadamard rotation help weight-only int4? (measurement, not a feature)

The idea (QuaRot, QuIP#, SpinQuant) is that quantization is free to work in
any orthogonal basis: for `y = W x`, insert `Q Q^T = I` to get
`y = (W Q) (Q^T x)`. A Hadamard `Q` spreads any outlier across all channels,
so the quantization grid stops being spent on one large value. Rotating the
weights costs nothing at inference, because `W Q` is folded in when the
weight is quantized.

This measures whether it is worth building for **weight-only** int4 on a
real model, by quantizing each weight with and without the rotation and
comparing the linear's output error.

    python tools/int4_rotation.py

The answer on Qwen2.5-0.5B is **no** — see docs/inference.md. The same
measurement shows 2.4x on deliberately outlier-heavy weights, so it is the
weights that are unrotatable, not the method or this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from generate import model_dir  # noqa: E402

GROUP = 128


def paley_hadamard(q: int) -> np.ndarray:
    """Hadamard of order q+1, Paley construction I (`H = I + S`, S skew), for
    a prime `q = 3 (mod 4)`."""
    if q % 4 != 3:
        raise ValueError(f"Paley I needs q = 3 (mod 4), got {q}")
    residues = {(i * i) % q for i in range(1, q)}
    chi = np.array([0 if a == 0 else (1 if a in residues else -1) for a in range(q)], dtype=np.float64)
    jacobsthal = np.array([[chi[(i - j) % q] for j in range(q)] for i in range(q)])
    s = np.zeros((q + 1, q + 1))
    s[0, 1:], s[1:, 0], s[1:, 1:] = 1.0, -1.0, jacobsthal
    return np.eye(q + 1) + s


def sylvester(n: int) -> np.ndarray:
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def hadamard(n: int) -> np.ndarray:
    """An orthonormal Hadamard of order `n`, where one is constructible here:
    a power of two (Sylvester), or 224 times one (Paley over GF(223)). The
    caller should check `H @ H.T == I`; the test in this file does."""
    if n & (n - 1) == 0:
        h = sylvester(n)
    elif n % 224 == 0 and (n // 224) & (n // 224 - 1) == 0:
        h = np.kron(sylvester(n // 224), paley_hadamard(223))
    else:
        raise ValueError(f"no Hadamard construction here for {n} "
                         "(a power of two, or 224 times one; 4864 = 2^8 * 19 is neither)")
    return h / np.sqrt(n)


def quantize_int4(w: np.ndarray, group: int = GROUP) -> np.ndarray:
    """torchao's group-wise asymmetric int4, dequantized again."""
    rows, k = w.shape
    groups = (k + group - 1) // group
    padded = np.pad(w, ((0, 0), (0, groups * group - k)), constant_values=np.nan)
    blocks = padded.reshape(rows, groups, group)
    lo, hi = np.nanmin(blocks, axis=2), np.nanmax(blocks, axis=2)
    scale = np.maximum((hi - lo) / 15.0, np.finfo(np.float32).eps)
    q = np.clip(np.round((blocks - lo[..., None]) / scale[..., None]), 0, 15)
    return (q * scale[..., None] + lo[..., None]).reshape(rows, -1)[:, :k].astype(np.float32)


def quantize_int8(w: np.ndarray) -> np.ndarray:
    """torchao's per-row symmetric int8, dequantized again."""
    amax = np.maximum(-np.minimum(w.min(1), 0), np.maximum(w.max(1), 0))
    scale = np.maximum(amax / 127.5, np.finfo(np.float32).eps)[:, None]
    return (np.clip(np.round(w / scale), -128, 127) * scale).astype(np.float32)


def error(x, w, wq, h=None) -> float:
    """Relative error of the quantized linear's output. With `h`, the weight
    was rotated, so the input is rotated too — the product is unchanged."""
    y = x @ w.T
    yq = (x @ h) @ wq.T if h is not None else x @ wq.T
    return float(np.linalg.norm(yq - y) / np.linalg.norm(y))


def check_hadamards() -> None:
    print("Hadamard matrices, orthogonality:")
    for n in (64, 128, 896, 4096):
        h = hadamard(n)
        print(f"  H_{n:<5} max|H H^T - I| = {np.abs(h @ h.T - np.eye(n)).max():.2e}")


def distributions(rng) -> None:
    """What rotation does to weights that do and do not have outliers: the
    control for the measurement below."""
    n, k = 256, 896
    h = hadamard(k).astype(np.float32)
    x = rng.standard_normal((8, k)).astype(np.float32)
    outliers = rng.standard_normal((n, k))
    outliers[rng.random((n, k)) < 0.01] *= 20
    print(f"\nsynthetic weights (int4, group {GROUP}):")
    for name, w in (("gaussian", rng.standard_normal((n, k))),
                    ("gaussian, 1% outliers x20", outliers),
                    ("heavy-tailed (student t, df 2)", rng.standard_t(2, (n, k)))):
        w = w.astype(np.float32)
        plain, rotated = error(x, w, quantize_int4(w)), error(x, w, quantize_int4(w @ h), h)
        print(f"  {name:32} int4 {plain:8.2e}  rotated {rotated:8.2e}  {plain / rotated:5.2f}x")


def real_weights(rng) -> None:
    from safetensors.torch import load_file
    state = load_file(str(model_dir(None) / "model.safetensors"))
    print("\nQwen2.5-0.5B-Instruct weights (int4, group 128), rotating what can be rotated:")
    print(f"  {'weight':38} {'int8':>9} {'int4':>9} {'rotated':>9} {'gain':>6}")
    gains = []
    for key, tensor in state.items():
        if tensor.dim() != 2 or key.endswith("norm.weight"):
            continue
        w = tensor.float().numpy()
        try:
            h = hadamard(w.shape[1]).astype(np.float32)
        except ValueError:
            continue
        if "layers." in key and not key.startswith(("model.layers.0.", "model.layers.12.", "model.layers.23.")):
            continue
        x = rng.standard_normal((8, w.shape[1])).astype(np.float32)
        eight, four = error(x, w, quantize_int8(w)), error(x, w, quantize_int4(w))
        rotated = error(x, w, quantize_int4(w @ h), h)
        gains.append(four / rotated)
        print(f"  {key:38} {eight:9.2e} {four:9.2e} {rotated:9.2e} {four / rotated:5.2f}x")
    if gains:
        print(f"  {'':38} {'':9} {'':9} {'mean gain':>9} {np.mean(gains):5.2f}x")


def main() -> None:
    rng = np.random.default_rng(0)
    check_hadamards()
    distributions(rng)
    real_weights(rng)
    print("\nRotation pays where there are outliers to spread. These weights have "
          "few, so it buys a few percent;\nwhat four bits cost here is the sixteen "
          "levels themselves (docs/inference.md).")


if __name__ == "__main__":
    main()
