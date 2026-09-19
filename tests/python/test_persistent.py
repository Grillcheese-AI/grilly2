"""Persistent memory for weights (docs/inference.md), and
``load_state_dict(assign=True)``.

Module parameters and tensors loaded from safetensors are held in device
memory of exactly their size, not in v1's pool, which rounds up to a power of
two and keeps released buffers. These tests read the device's
``persistent_bytes`` counter. The counts are exact integers, and every value
comparison is exact: loading and assigning copy or share bits and compute
nothing. The one forward pass compared with torch uses rtol 1e-5 / atol 1e-6,
as for single forward passes in test_nn.py.
"""

import gc

import numpy as np
import pytest
import torch
from safetensors.torch import load_file as torch_load_file
from safetensors.torch import save_file

import grilly
import grilly.nn as gnn
from grilly.infer.safetensors import load_file


def persistent_bytes() -> int:
    gc.collect()
    return grilly._C.submit_stats()["persistent_bytes"]


def test_module_parameters_are_exact_size_and_freed_with_the_module():
    before = persistent_bytes()
    lin, ln, emb = gnn.Linear(300, 70), gnn.LayerNorm(33), gnn.Embedding(50, 7)
    assert persistent_bytes() - before == 4 * (300 * 70 + 70 + 2 * 33 + 50 * 7)
    np.testing.assert_array_equal(ln.weight.detach().numpy(), np.ones(33, np.float32))
    np.testing.assert_array_equal(ln.bias.detach().numpy(), np.zeros(33, np.float32))
    del lin, ln, emb
    assert persistent_bytes() == before


def test_loaded_tensors_are_exact_size_and_freed_when_dropped(tmp_path):
    path = tmp_path / "w.safetensors"
    save_file({"w": torch.ones(4864, 896, dtype=torch.bfloat16), "i": torch.arange(5), "b": torch.ones(3) > 0}, path)
    before = persistent_bytes()
    sd = load_file(path)
    # bf16 widens to 4 bytes; int64 is 8; grilly stores bool in 4
    assert persistent_bytes() - before == 4864 * 896 * 4 + 5 * 8 + 3 * 4
    del sd
    assert persistent_bytes() == before


def make(nn):
    return nn.Sequential(nn.Embedding(50, 16), nn.LayerNorm(16), nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 50))


def param_bytes(model) -> int:
    return 4 * sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("assign", [False, True], ids=["copy", "assign"])
def test_loading_a_checkpoint_leaves_one_copy_of_the_weights(tmp_path, assign):
    torch.manual_seed(0)
    reference = make(torch.nn)
    path = tmp_path / "model.safetensors"
    save_file(reference.state_dict(), path)

    before = persistent_bytes()
    model = make(gnn)
    sd = load_file(path)
    assert persistent_bytes() - before == 2 * param_bytes(model)  # the model's own, and the loaded
    result = model.load_state_dict(sd, assign=assign)
    assert (result.missing_keys, result.unexpected_keys) == ([], [])
    del sd
    assert persistent_bytes() - before == param_bytes(model)

    ids = [[1, 2, 3, 4], [5, 6, 7, 49]]
    with torch.no_grad(), grilly.no_grad():
        want = reference(torch.tensor(ids)).numpy()
        got = model(grilly.tensor(ids)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_assign_shares_storage_and_keeps_requires_grad_as_torch_does(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "model.safetensors"
    save_file(make(torch.nn).state_dict(), path)

    tmodel = make(torch.nn)
    tmodel[0].weight.requires_grad_(False)
    tsd = torch_load_file(path)
    tmodel.load_state_dict(tsd, assign=True)

    model = make(gnn)
    model[0].weight.requires_grad_(False)
    sd = load_file(path)
    model.load_state_dict(sd, assign=True)

    for name, p in model.named_parameters():
        tp = dict(tmodel.named_parameters())[name]
        assert isinstance(p, gnn.Parameter) and isinstance(tp, torch.nn.Parameter)
        assert p.requires_grad == tp.requires_grad, name  # the module's flag, not the loaded tensor's
    assert not model[0].weight.requires_grad

    # the parameter is the loaded tensor, not a copy, as torch's shares its data_ptr
    assert tmodel[2].weight.data_ptr() == tsd["2.weight"].data_ptr()
    with grilly.no_grad():
        sd["2.weight"].fill_(3.0)
    np.testing.assert_array_equal(model[2].weight.detach().numpy(), np.full((32, 16), 3.0, np.float32))


def test_assign_replaces_buffers_and_still_checks_shapes():
    class WithBuffer(gnn.Module):
        def __init__(self, n):
            super().__init__()
            self.register_buffer("scale", grilly.ones(n))
            self.lin = gnn.Linear(n, n)

    m = WithBuffer(3)
    new_scale = grilly.tensor([2.0, 3.0, 4.0])
    sd = {"scale": new_scale, "lin.weight": grilly.zeros(3, 3), "lin.bias": grilly.zeros(3)}
    m.load_state_dict(sd, assign=True)
    assert m.scale is new_scale  # buffers are assigned as given, as torch does
    np.testing.assert_array_equal(m.lin.weight.detach().numpy(), np.zeros((3, 3), np.float32))

    before = m.lin.weight
    bad = dict(sd, **{"lin.weight": grilly.zeros(4, 3)})
    with pytest.raises(RuntimeError, match="size mismatch for lin.weight"):
        m.load_state_dict(bad, assign=True)
    assert m.lin.weight is before
