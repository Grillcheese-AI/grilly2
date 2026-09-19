"""grilly.infer.safetensors against the safetensors package (and torch).

Files are written by ``safetensors.torch.save_file`` and read back two ways:
by grilly, and by ``safetensors.torch.load_file`` as the reference.

Tolerance: loading is bit-exact. F32, integer and bool tensors are copied,
and BF16 and F16 widen to float32 exactly. So every value is compared bit for
bit with torch's ``.float()`` of the reference tensor; every one of the 65,536
bf16 and f16 bit patterns is checked. The one approximate comparison is a
model's forward pass on loaded weights against torch's: rtol 1e-5, atol 1e-6,
as for single forward passes in test_nn.py. That is case (b) in CLAUDE.md §9:
torch implements embedding, layer_norm, GELU and two matmuls differently.
"""

import os
import pathlib

import numpy as np
import pytest
import torch
from safetensors import safe_open as ref_open
from safetensors.torch import load_file as ref_load_file
from safetensors.torch import save, save_file

import grilly
import grilly.nn as gnn
from grilly.infer.safetensors import SafetensorError, load, load_file, safe_open


def bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a).view(np.uint32)


def same_values(g: grilly.Tensor, t: torch.Tensor) -> None:
    """g is t's value exactly: float32 for float t (widened), else t's dtype."""
    assert tuple(g.shape) == tuple(t.shape)
    if t.is_floating_point():
        assert g.dtype == grilly.float32
        want = t.float().numpy()
        got = g.numpy()
        nan = np.isnan(want)
        np.testing.assert_array_equal(np.isnan(got), nan)
        np.testing.assert_array_equal(bits(got)[~nan], bits(want)[~nan])
    else:
        assert g.dtype == {torch.int64: grilly.int64, torch.int32: grilly.int32, torch.bool: grilly.bool}[t.dtype]
        np.testing.assert_array_equal(g.numpy(), t.numpy())


def test_every_bf16_and_f16_bit_pattern_widens_as_torch_does(tmp_path):
    patterns = torch.from_numpy(np.arange(1 << 16, dtype=np.uint16).view(np.int16))
    tensors = {"bf16": patterns.view(torch.bfloat16).clone(), "f16": patterns.view(torch.float16).reshape(256, 256).clone()}
    path = tmp_path / "halves.safetensors"
    save_file(tensors, path)
    got, want = load_file(path), ref_load_file(path)
    for name in tensors:
        same_values(got[name], want[name])


def test_supported_dtypes_and_shapes(tmp_path):
    rng = np.random.default_rng(0)
    tensors = {
        "f32": torch.from_numpy(rng.standard_normal((3, 4, 5)).astype(np.float32)),
        "f32_specials": torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1e-45, 3.4e38]),
        "i64": torch.tensor([-(2**63), -1, 0, 2**63 - 1]),
        "i32": torch.tensor([-(2**31), 5, 2**31 - 1], dtype=torch.int32),
        "bool": torch.tensor([[True, False], [False, True]]),
        "scalar": torch.tensor(2.5),
        "empty": torch.zeros(0, 3),
        "bf16": torch.from_numpy(rng.standard_normal((7, 9)).astype(np.float32)).bfloat16(),
    }
    path = tmp_path / "mixed.safetensors"
    save_file(tensors, path)
    got, want = load_file(path), ref_load_file(path)
    assert list(got) == list(want)
    for name in tensors:
        same_values(got[name], want[name])


def test_unsupported_dtypes_raise_and_name_the_tensor(tmp_path):
    tensors = {
        "ok": torch.ones(2),
        "f64": torch.ones(2, dtype=torch.float64),
        "i8": torch.ones(2, dtype=torch.int8),
        "u8": torch.ones(2, dtype=torch.uint8),
        "i16": torch.ones(2, dtype=torch.int16),
    }
    path = tmp_path / "unsupported.safetensors"
    save_file(tensors, path)
    with safe_open(path, framework="pt") as f:
        assert f.keys() == sorted(tensors)  # the header lists every tensor
        np.testing.assert_array_equal(f.get_tensor("ok").numpy(), [1, 1])
        for name, dtype in [("f64", "F64"), ("i8", "I8"), ("u8", "U8"), ("i16", "I16")]:
            with pytest.raises(NotImplementedError, match=f'"{name}".*{dtype}'):
                f.get_tensor(name)
    with pytest.raises(NotImplementedError):
        load_file(path)


def test_keys_order_and_metadata_match_the_package(tmp_path):
    tensors = {"b": torch.zeros(3), "a": torch.ones(2), "c": torch.zeros(1, dtype=torch.int64)}
    path = tmp_path / "meta.safetensors"
    save_file(tensors, path, metadata={"format": "pt", "note": "x"})
    with safe_open(path, framework="pt") as g, ref_open(path, framework="pt") as t:
        assert g.keys() == t.keys()
        assert g.offset_keys() == t.offset_keys()
        assert g.metadata() == t.metadata() == {"format": "pt", "note": "x"}
    assert list(load_file(path)) == ref_open(path, framework="pt").offset_keys()

    save_file(tensors, tmp_path / "plain.safetensors")
    with safe_open(tmp_path / "plain.safetensors", framework="pt") as g:
        assert g.metadata() is None


def test_load_from_bytes_matches_load_file(tmp_path):
    tensors = {"w": torch.arange(6.0).reshape(2, 3), "i": torch.tensor([4, 5])}
    data = save(tensors)
    path = tmp_path / "b.safetensors"
    path.write_bytes(data)
    from_bytes, from_file = load(data), load_file(path)
    assert list(from_bytes) == list(from_file)
    for k in tensors:
        same_values(from_bytes[k], tensors[k])
        same_values(from_file[k], tensors[k])


def test_paths_devices_and_frameworks(tmp_path):
    path = tmp_path / "p.safetensors"
    save_file({"w": torch.ones(2)}, path)
    for p in (path, str(path), os.fspath(path)):
        assert list(load_file(p)) == ["w"]
    for d in ("cpu", "vulkan", "vulkan:0", grilly.device("vulkan", 0), None):
        assert list(load_file(path, device=d)) == ["w"]
    with pytest.raises(NotImplementedError, match="device 'cuda'"):
        load_file(path, device="cuda")
    for framework in ("pt", "torch", "pytorch"):
        with safe_open(path, framework=framework) as f:
            assert f.keys() == ["w"]
    with pytest.raises(NotImplementedError, match="framework='np'"):
        safe_open(path, framework="np")


def test_errors_match_the_package(tmp_path):
    path = tmp_path / "e.safetensors"
    save_file({"w": torch.ones(2)}, path)

    with safe_open(path, framework="pt") as f:
        with pytest.raises(SafetensorError, match="File does not contain tensor missing"):
            f.get_tensor("missing")
    for method in ("keys", "offset_keys", "metadata"):
        with pytest.raises(SafetensorError, match="File is closed"):
            getattr(f, method)()
    with pytest.raises(SafetensorError, match="File is closed"):
        f.get_tensor("w")

    with pytest.raises(FileNotFoundError, match="absent.safetensors"):
        load_file(tmp_path / "absent.safetensors")

    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"\x05\x00\x00\x00\x00\x00\x00\x00{bad}")
    with pytest.raises(SafetensorError, match="header"):
        load_file(bad)

    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(SafetensorError):
        load_file(truncated)

    empty = tmp_path / "empty.safetensors"
    empty.write_bytes(b"")
    with pytest.raises(SafetensorError):
        load_file(empty)

    with pytest.raises(SafetensorError):
        load(b"not a safetensors file")

    with safe_open(path, framework="pt") as f:
        with pytest.raises(NotImplementedError, match="get_slice"):
            f.get_slice("w")


def make(nn):
    return nn.Sequential(nn.Embedding(50, 16), nn.LayerNorm(16), nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 50))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_a_torch_checkpoint_runs_in_a_grilly_model(tmp_path, dtype):
    torch.manual_seed(0)
    reference = make(torch.nn).to(dtype)
    path = tmp_path / "model.safetensors"
    save_file(reference.state_dict(), path)

    model = make(gnn)
    result = model.load_state_dict(load_file(path))
    assert (result.missing_keys, result.unexpected_keys) == ([], [])

    # torch runs the checkpoint's weights widened to float32, as grilly holds them
    reference = reference.float()
    ids = [[1, 2, 3, 4], [5, 6, 7, 49]]
    with torch.no_grad(), grilly.no_grad():
        want = reference(torch.tensor(ids)).numpy()
        got = model(grilly.tensor(ids)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def cached(repo: str, filename: str):
    """A file from the local Hugging Face cache, or None. Nothing is downloaded."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    path = try_to_load_from_cache(repo, filename)
    return pathlib.Path(path) if isinstance(path, str) else None


def test_a_real_checkpoint_loads_exactly():
    path = cached("sentence-transformers/all-MiniLM-L6-v2", "model.safetensors")
    if path is None:
        pytest.skip("all-MiniLM-L6-v2 is not in the local Hugging Face cache")
    got, want = load_file(path), ref_load_file(path)
    assert list(got) == list(want)
    for name in want:
        same_values(got[name], want[name])
