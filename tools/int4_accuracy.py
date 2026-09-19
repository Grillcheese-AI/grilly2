"""What would make weight-only int4 accurate? (measurements, not features)

int4 costs ~11% relative error on a linear's output where int8 costs ~1%
(docs/inference.md). This measures the candidates on real weights before any
of them is built: a Hadamard rotation, a non-uniform codebook (NF4's idea,
and its MSE-optimal cousin Lloyd-Max), clipping, and the best any 16-level
scalar quantizer can do.

    python tools/int4_accuracy.py

## Part 1: the rotation

The idea (QuaRot, QuIP#, SpinQuant) is that quantization is free to work in
any orthogonal basis: for `y = W x`, insert `Q Q^T = I` to get
`y = (W Q) (Q^T x)`. A Hadamard `Q` spreads any outlier across all channels,
so the quantization grid stops being spent on one large value. Rotating the
weights costs nothing at inference, because `W Q` is folded in when the
weight is quantized.

This measures whether it is worth building for **weight-only** int4 on a
real model, by quantizing each weight with and without the rotation and
comparing the linear's output error.

The answer on Qwen2.5-0.5B is **no** — see docs/inference.md. The same
measurement shows 2.3x on deliberately outlier-heavy weights, so it is the
weights that are unrotatable, not the method or this script.

## Part 2: the codebook

A rotation makes a weight gaussian whether or not it started that way, and
a gaussian is exactly when a *non-uniform* codebook pays: uniform steps
spend the same resolution on the dense middle and the empty tails. QLoRA's
NF4 uses equal-probability quantiles; Lloyd-Max uses the levels that
minimize squared error, which is the same iteration as 1-D k-means.

The last row of that table is the one that decides the question: Lloyd-Max
fitted to each group separately is the best *any* 16-level scalar quantizer
with per-group parameters can do, whatever the codebook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from generate import model_dir  # noqa: E402

GROUP = 128
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_weights(name_or_path: str):
    """Every 2-D weight of a model already in the Hugging Face cache (or a
    directory), as float32 numpy, one shard at a time. Nothing is
    downloaded here."""
    import json

    from safetensors.torch import load_file

    path = Path(name_or_path)
    if path.is_dir():
        files = sorted(path.glob("model*.safetensors"))
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    else:
        from huggingface_hub import try_to_load_from_cache
        config_file = try_to_load_from_cache(name_or_path, "config.json")
        if not isinstance(config_file, str):
            sys.exit(f"{name_or_path} is not in the local Hugging Face cache and this tool "
                     "downloads nothing; pass a directory, or fetch it first")
        config = json.loads(Path(config_file).read_text(encoding="utf-8"))
        root = Path(config_file).parent
        files = sorted(root.glob("model*.safetensors"))
    if not files:
        sys.exit(f"no safetensors weights found for {name_or_path}")
    weights = {}
    for shard in files:
        for key, tensor in load_file(str(shard)).items():
            if tensor.dim() == 2 and not key.endswith("norm.weight"):
                weights[key] = tensor.float().numpy()
    return config, weights


def paley_hadamard(q: int) -> np.ndarray:
    """Hadamard of order q+1, Paley construction I (`H = I + S`, S skew), for
    a prime `q = 3 (mod 4)`."""
    if q % 4 != 3:
        raise ValueError(f"Paley I needs q = 3 (mod 4), got {q}")
    if q < 3 or any(q % d == 0 for d in range(2, int(q**0.5) + 1)):
        # a prime *power* needs that field's arithmetic, not integers mod q:
        # quadratic residues mod 27 do not give a Hadamard matrix, and the
        # result looks plausible until H H^T is checked
        raise ValueError(f"Paley I here needs a prime q, got {q}")
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
    a power of two (Sylvester), or a power of two times 12, 20 or 224
    (Paley I over GF(11), GF(19), GF(223)). The caller should check
    `H @ H.T == I`; `check_hadamards` in this file does - order 28 was in
    this list until that check rejected it, because 27 is a prime power and
    integers mod 27 are not its field.

    Which model this can rotate turns on that: Qwen3's 2048 and 6144 both
    have one (6144 = 12 x 512), Qwen2.5's 896 does (4 x 224) but its 4864 =
    2^8 * 19 does not.
    """
    if n & (n - 1) == 0:
        return sylvester(n) / np.sqrt(n)
    for base in (12, 20, 224):  # q = 11, 19, 223: prime and 3 (mod 4)
        rest = n // base
        if n % base == 0 and rest > 0 and rest & (rest - 1) == 0:
            return np.kron(sylvester(rest), paley_hadamard(base - 1)) / np.sqrt(n)
    raise ValueError(f"no Hadamard construction here for {n} (a power of two, or a power of "
                     "two times 12, 20 or 224)")


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


def _lloyd_from(samples: np.ndarray, book: np.ndarray, rounds: int) -> tuple[np.ndarray, float]:
    for _ in range(rounds):
        edges = (book[1:] + book[:-1]) / 2
        which = np.searchsorted(edges, samples)
        for i in range(len(book)):
            taken = samples[which == i]
            if taken.size:
                book[i] = taken.mean()
        book = np.sort(book)
    edges = (book[1:] + book[:-1]) / 2
    error = float(np.mean((book[np.searchsorted(edges, samples)] - samples) ** 2))
    return book, error


def lloyd_max(samples: np.ndarray, levels: int = 16, rounds: int = 60) -> np.ndarray:
    """The MSE-optimal `levels` reconstruction points for `samples`: 1-D
    k-means (Lloyd's algorithm).

    k-means in one dimension is not convex, and the starting point decides
    which optimum it finds: from the quantiles (equal counts) a group of 128
    normal samples converges to MSE 0.0184, from evenly spaced points to
    0.0093 — worse than plain uniform quantization and better than it,
    respectively. So several starts are tried and the best is kept.
    """
    lo, hi = samples.min(), samples.max()
    starts = [np.linspace(lo, hi, levels)]
    for span in (0.5, 0.7, 0.85):  # trimmed ranges, for heavy tails
        edge = span * max(abs(lo), abs(hi))
        starts.append(np.linspace(-edge, edge, levels))
    starts.append(np.quantile(samples, (np.arange(levels) + 0.5) / levels))
    best, best_error = None, np.inf
    for start in starts:
        book, error = _lloyd_from(samples, np.array(start, dtype=np.float64), rounds)
        if error < best_error:
            best, best_error = book, error
    return best


def normal_codebook(kind: str, levels: int = 16) -> np.ndarray:
    """A codebook for a unit normal: NF4's equal-probability quantiles, or
    Lloyd-Max's squared-error-optimal levels. NF4's is scaled so its
    outermost level is 1, because QLoRA normalizes each block by its absmax,
    not by its standard deviation."""
    rng = np.random.default_rng(11)
    samples = rng.standard_normal(2_000_000)
    if kind == "nf4":
        book = np.quantile(samples, (np.arange(levels) + 0.5) / levels)
        return book / np.abs(book).max()
    return lloyd_max(samples, levels)


def apply_codebook(w: np.ndarray, book: np.ndarray, group: int = GROUP, by: str = "std") -> np.ndarray:
    """Quantize each group by normalizing it and rounding to the nearest
    entry of `book` — two parameters per group, as the uniform scheme has,
    so the memory is identical. `by` is how a group is normalized: its
    standard deviation, or its absmax as QLoRA's NF4 does."""
    rows, k = w.shape
    groups = (k + group - 1) // group
    padded = np.pad(w, ((0, 0), (0, groups * group - k)), constant_values=np.nan)
    blocks = padded.reshape(rows, groups, group)
    mean = np.nanmean(blocks, axis=2, keepdims=True) if by == "std" else 0.0
    spread = np.nanstd(blocks, axis=2, keepdims=True) if by == "std"         else np.nanmax(np.abs(blocks), axis=2, keepdims=True)
    spread = np.maximum(spread, np.finfo(np.float32).eps)
    normalized = (blocks - mean) / spread
    edges = (book[1:] + book[:-1]) / 2
    picked = book[np.searchsorted(edges, np.nan_to_num(normalized))]
    return (picked * spread + mean).reshape(rows, -1)[:, :k].astype(np.float32)


def per_group_lloyd(w: np.ndarray, group: int = GROUP, rounds: int = 30) -> np.ndarray:
    """The best any 16-level scalar quantizer can do: Lloyd-Max fitted to
    each group on its own. Not implementable — it would need a codebook per
    group — but it bounds every codebook that is."""
    rows, k = w.shape
    groups = (k + group - 1) // group
    padded = np.pad(w, ((0, 0), (0, groups * group - k)), constant_values=np.nan)
    blocks = padded.reshape(-1, group)
    finite = np.nan_to_num(blocks, nan=np.nanmean(blocks))
    out = np.empty_like(finite)
    # in chunks: the assignment is (chunk, group, 16), and a whole lm_head
    # at once is 37 GiB
    chunk = max(1, 2 ** 22 // (group * 16))
    for start in range(0, finite.shape[0], chunk):
        part = finite[start:start + chunk]
        # evenly spaced starts, not quantiles: see lloyd_max
        lo, hi = part.min(axis=1, keepdims=True), part.max(axis=1, keepdims=True)
        book = lo + (hi - lo) * (np.arange(16) / 15.0)[None, :]  # (chunk, 16)
        for _ in range(rounds):
            which = np.abs(part[:, :, None] - book[:, None, :]).argmin(axis=2)
            for i in range(16):
                mask = which == i
                count = mask.sum(axis=1)
                total = np.where(mask, part, 0.0).sum(axis=1)
                book[:, i] = np.where(count > 0, total / np.maximum(count, 1), book[:, i])
        which = np.abs(part[:, :, None] - book[:, None, :]).argmin(axis=2)
        out[start:start + chunk] = np.take_along_axis(book, which, axis=1)
    return out.reshape(rows, -1)[:, :k].astype(np.float32)


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


def sample_keys(weights, per_kind: int = 3):
    """A few weights of each kind (q_proj, down_proj, ...), from layers
    spread through the model, plus whatever is outside the layers."""
    kinds = {}
    for key in weights:
        kind = key.split(".")[-2] if ".layers." in key else key
        kinds.setdefault(kind, []).append(key)
    chosen = []
    for kind, keys in kinds.items():
        step = max(1, len(keys) // per_kind)
        chosen.extend(keys[::step][:per_kind])
    return chosen


def real_weights(rng, model: str, weights) -> None:
    print(f"\n{model} weights (int4, group {GROUP}), rotating what can be rotated:")
    print(f"  {'weight':44} {'int8':>9} {'int4':>9} {'rotated':>9} {'gain':>6}")
    gains, skipped = [], set()
    for key in sample_keys(weights):
        w = weights[key]
        try:
            h = hadamard(w.shape[1]).astype(np.float32)
        except ValueError:
            skipped.add(w.shape[1])
            continue
        x = rng.standard_normal((8, w.shape[1])).astype(np.float32)
        eight, four = error(x, w, quantize_int8(w)), error(x, w, quantize_int4(w))
        rotated = error(x, w, quantize_int4(w @ h), h)
        gains.append(four / rotated)
        print(f"  {key:44} {eight:9.2e} {four:9.2e} {rotated:9.2e} {four / rotated:5.2f}x")
    if gains:
        print(f"  {'':44} {'':9} {'':9} {'mean gain':>9} {np.mean(gains):5.2f}x")
    for size in sorted(skipped):
        print(f"  (no Hadamard construction for {size}, so those weights are not in the table)")


def codebooks(rng, model: str, weights) -> None:
    """Uniform against non-uniform codebooks, with and without the rotation,
    and the bound on all of them."""
    # a few kinds, and not the 152K-row embedding: the bound below fits a
    # codebook to every group, which is minutes of numpy on a big weight
    keys = [k for k in sample_keys(weights, per_kind=1) if weights[k].shape[0] <= 8192][:6]
    keys = [k for k in keys if _rotatable(weights[k].shape[1])]
    nf4, lloyd = normal_codebook("nf4"), normal_codebook("lloyd")
    print(f"\n16-level codebooks (group {GROUP}), mean relative error over {len(keys)} weights of {model}:")
    print(f"  {'scheme':44} {'plain':>9} {'rotated':>9}")
    schemes = [
        ("uniform, min/max scale (what grilly does)", quantize_int4),
        ("NF4 codebook (quantiles, absmax per group)", lambda w: apply_codebook(w, nf4, by="absmax")),
        ("Lloyd-Max codebook for a normal", lambda w: apply_codebook(w, lloyd)),
        ("Lloyd-Max fitted per group (not implementable)", per_group_lloyd),
    ]
    rows = {}
    for key in keys:
        w = weights[key]
        h = hadamard(w.shape[1]).astype(np.float32)
        x = rng.standard_normal((8, w.shape[1])).astype(np.float32)
        for name, fn in schemes:
            rows.setdefault(name, ([], []))
            rows[name][0].append(error(x, w, fn(w)))
            rows[name][1].append(error(x, w, fn(w @ h), h))
    for name, _ in schemes:
        plain, rotated = rows[name]
        print(f"  {name:44} {np.mean(plain):9.2e} {np.mean(rotated):9.2e}")
    eights = [error(rng.standard_normal((8, weights[k].shape[1])).astype(np.float32), weights[k],
                    quantize_int8(weights[k])) for k in keys]
    print(f"  {'int8, for scale':44} {np.mean(eights):9.2e}")


def _rotatable(n: int) -> bool:
    try:
        hadamard(n)
        return True
    except ValueError:
        return False


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="a directory, or a repository id already in the Hugging Face cache")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    check_hadamards()
    distributions(rng)
    config, weights = load_weights(args.model)
    shapes = {k: config.get(k) for k in ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers")}
    print(f"\n{args.model}: {shapes}")
    real_weights(rng, args.model, weights)
    codebooks(rng, args.model, weights)


if __name__ == "__main__":
    main()
