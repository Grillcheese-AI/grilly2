"""The meta device: tensors with a shape and a dtype and no data.

Tolerance: none of this is arithmetic. What is asserted is **behaviour
against torch**, which is installed here — the same repr, the same errors,
the same no-op warning when a checkpoint is copied into a meta parameter —
and **that nothing is allocated**, which is the point of the exercise and
is measured with the shim's persistent-memory counters rather than assumed.

Why it exists: a model built the ordinary way allocates every parameter it
is about to overwrite from a checkpoint. On a 12 GB card that is what caps
the model size, since a 7B in int8 fits but a 7B in float32 does not.
"""

import numpy as np
import pytest
import torch

import grilly
from grilly import device, nn
from grilly.nn import functional as F


def persistent():
    s = grilly._C.submit_stats()
    return s["persistent_bytes"], s["persistent_peak"]


@pytest.mark.parametrize("factory", ["empty", "zeros", "ones", "randn", "rand"])
def test_factories_build_meta_tensors_both_ways(factory):
    with device("meta"):
        inside = getattr(grilly, factory)(3, 4)
    explicit = getattr(grilly, factory)(3, 4, device="meta")
    want = getattr(torch, factory)(3, 4, device="meta")
    for got in (inside, explicit):
        assert got.is_meta is True
        assert tuple(got.shape) == tuple(want.shape)
        assert got.dtype is grilly.float32
        assert str(got.device) == str(want.device) == "meta"
        assert repr(got) == repr(want)
    assert grilly.empty(3, 4).is_meta is False, "the context is closed again"


def test_full_and_dtypes():
    with device("meta"):
        assert grilly.full((2, 3), 7).dtype is grilly.int64  # inferred from the value, as torch does
        assert grilly.full((2, 3), 0.5).dtype is grilly.float32
        assert grilly.zeros(2, dtype=grilly.int32).dtype is grilly.int32
        # torch's repr, with grilly's own dtype spelling
        assert repr(grilly.zeros(2, dtype=grilly.int32)) == repr(
            torch.zeros(2, dtype=torch.int32, device="meta")).replace("torch.int32", "grilly.int32")


def test_a_factory_without_a_meta_path_refuses_rather_than_allocating():
    """Silently returning real storage under `with device("meta")` is the
    failure that would waste the memory this is here to save."""
    with device("meta"):
        with pytest.raises(NotImplementedError, match="meta device"):
            grilly.arange(4)


def test_building_a_module_allocates_nothing():
    before, _ = persistent()
    with device("meta"):
        model = nn.Sequential(nn.Linear(4096, 4096), nn.LayerNorm(4096), nn.RMSNorm(4096),
                              nn.Embedding(1000, 4096), nn.Linear(4096, 8, bias=False))
    after, _ = persistent()
    assert after == before, f"{after - before} bytes allocated for a meta model"

    state = model.state_dict()
    assert [tuple(v.shape) for v in state.values()] == [
        (4096, 4096), (4096,), (4096,), (4096,), (4096,), (1000, 4096), (8, 4096)]
    assert all(v.is_meta for v in state.values())
    assert isinstance(model[0].weight, nn.Parameter), "a meta parameter is still a Parameter"
    assert model[0].weight.requires_grad is True
    assert model[4].weight.numel() == 8 * 4096


def test_an_operation_on_a_meta_tensor_says_what_to_do():
    with device("meta"):
        layer = nn.Linear(8, 8)
    with pytest.raises(NotImplementedError, match="no data"):
        layer(grilly.ones(1, 8))
    with pytest.raises(NotImplementedError, match="no data"):
        layer.weight.detach().numpy()
    with pytest.raises(NotImplementedError, match="no data"):
        F.relu(layer.weight)
    with pytest.raises(NotImplementedError, match="to_empty"):
        layer.weight.to("cpu")


def test_initialization_is_a_no_op_as_in_torch():
    """`reset_parameters` runs unchanged on meta, writing nothing."""
    with device("meta"):
        layer = nn.Linear(8, 8)
        layer.reset_parameters()
        nn.init.normal_(layer.weight)
        nn.init.zeros_(layer.bias)
        grilly.zeros(4).uniform_(0, 1)
    assert layer.weight.is_meta


def test_to_empty_gives_real_uninitialized_storage():
    with device("meta"):
        model = nn.Sequential(nn.Linear(16, 8), nn.LayerNorm(8))
        model[0].weight.requires_grad_(False)
    model.to_empty()
    assert not any(v.is_meta for v in model.state_dict().values())
    assert model(grilly.ones(2, 16)).shape == (2, 8)
    assert model[0].weight.requires_grad is False, "to_empty keeps requires_grad"
    assert model[0].bias.requires_grad is True
    with pytest.raises(NotImplementedError, match="one Vulkan device"):
        nn.Linear(4, 4).to_empty(device="cuda")


def test_load_state_dict_assign_replaces_meta_parameters():
    with device("meta"):
        model = nn.Linear(16, 4)
    values = {"weight": grilly.from_numpy(np.full((4, 16), 0.25, dtype=np.float32)),
              "bias": grilly.from_numpy(np.arange(4, dtype=np.float32))}
    model.load_state_dict(values, assign=True)
    assert not model.weight.is_meta
    got = model(grilly.ones(1, 16)).detach().numpy()
    np.testing.assert_allclose(got, np.array([[4.0, 5.0, 6.0, 7.0]]), rtol=1e-6)
    assert model.weight.requires_grad is True, "assign keeps the parameter's requires_grad"


def test_copying_into_a_meta_parameter_warns_and_does_nothing_as_torch_does():
    with device("meta"):
        model = nn.Linear(4, 4)
    with pytest.warns(UserWarning, match="no-op"):
        model.load_state_dict({"weight": grilly.zeros(4, 4), "bias": grilly.zeros(4)})
    assert model.weight.is_meta, "torch leaves the parameter on meta here too"

    with torch.device("meta"):  # the same call against torch
        ref = torch.nn.Linear(4, 4)
    with pytest.warns(UserWarning, match="no-op"):
        ref.load_state_dict({"weight": torch.zeros(4, 4), "bias": torch.zeros(4)})
    assert ref.weight.is_meta


def test_module_to_raises_exactly_as_torch_does():
    with device("meta"):
        model = nn.Linear(4, 4)
    with torch.device("meta"):
        ref = torch.nn.Linear(4, 4)
    with pytest.raises(NotImplementedError) as got:
        model.to("cpu")
    with pytest.raises(NotImplementedError) as want:
        ref.to("cpu")
    assert str(got.value) == str(want.value)
