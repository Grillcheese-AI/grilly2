"""int8 weight-only linear against float32 linear (CLAUDE.md §11, step 2).

Two measurements, each separate:

1. GPU time per call at Qwen2.5-0.5B's linear shapes, for decode (1 row)
   and prefill (8 and 128 rows). Each case records 200 identical calls in
   one batch, and the time is the shim's wait for that batch divided by
   200. The effective bandwidth counts the weight bytes read per call.
2. Output error on Qwen2.5-0.5B's real weights, when it is in the local
   Hugging Face cache: ||int8 linear - float32 linear|| / ||float32 linear||
   for every linear weight, on random normal inputs. Real activations need
   the decode loop (step 4).

    python tools/int8_linear.py
"""

from __future__ import annotations

import numpy as np

import grilly
import grilly.nn.functional as F
from grilly.infer.quantization import Int8Tensor

SHAPES = {  # name: (out_features, in_features)
    "q_proj / o_proj": (896, 896),
    "k_proj / v_proj": (128, 896),
    "gate_proj / up_proj": (4864, 896),
    "down_proj": (896, 4864),
    "lm_head (tied)": (151936, 896),
}
CALLS = 200


def gpu_us_per_call(fn) -> float:
    fn()
    grilly._C.synchronize()  # warm: pipelines, pools
    outs = [fn() for _ in range(CALLS)]
    grilly._C.synchronize()
    del outs
    return grilly._C.submit_stats()["wait_ns_last"] / 1e3 / CALLS


def timing() -> None:
    rng = np.random.default_rng(0)
    print(f"GPU us per call ({CALLS} calls per batch); GB/s = weight bytes read / time")
    print(f"{'layer':22} {'rows':>4} | {'f32 us':>9} {'GB/s':>6} | {'int8 us':>9} {'GB/s':>6} | speedup")
    with grilly.no_grad():
        for name, (n, k) in SHAPES.items():
            w = grilly.from_numpy((rng.standard_normal((n, k)) * 0.02).astype(np.float32))
            qw = Int8Tensor.from_hp(w)
            for m in (1, 8, 128):
                x = grilly.from_numpy(rng.standard_normal((m, k)).astype(np.float32))
                f32 = gpu_us_per_call(lambda: F.linear(x, w))
                i8 = gpu_us_per_call(lambda: F.linear(x, qw))
                print(f"{name:22} {m:>4} | {f32:>9.1f} {n * k * 4 / f32 / 1e3:>6.0f} | "
                      f"{i8:>9.1f} {qw.nbytes / i8 / 1e3:>6.0f} | {f32 / i8:>6.2f}x")


def real_weight_error() -> None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return
    path = try_to_load_from_cache("Qwen/Qwen2.5-0.5B-Instruct", "model.safetensors")
    if not isinstance(path, str):
        print("\nQwen2.5-0.5B-Instruct is not in the local Hugging Face cache; skipping the error measurement")
        return
    from grilly.infer.safetensors import safe_open

    rng = np.random.default_rng(1)
    errors = {}
    with safe_open(path, framework="pt") as f, grilly.no_grad():
        for key in f.keys():
            if not key.endswith(".weight") or key.endswith("norm.weight"):
                continue
            w = f.get_tensor(key)
            if w.dim() != 2:
                continue
            x = grilly.from_numpy(rng.standard_normal((4, w.shape[1])).astype(np.float32))
            y = F.linear(x, w).numpy()
            yq = F.linear(x, Int8Tensor.from_hp(w)).numpy()
            kind = key.split(".")[-2] if "layers" in key else key
            errors.setdefault(kind, []).append(float(np.linalg.norm(yq - y) / np.linalg.norm(y)))
    print("\nQwen2.5-0.5B-Instruct real weights: relative error of the int8 linear's output")
    print(f"{'weight':32} {'count':>5} {'mean':>9} {'max':>9}")
    for kind, e in errors.items():
        print(f"{kind:32} {len(e):>5} {np.mean(e):>9.2e} {np.max(e):>9.2e}")


if __name__ == "__main__":
    timing()
    real_weight_error()
