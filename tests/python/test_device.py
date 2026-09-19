"""Devices: grilly's one Vulkan device, with "cpu" as an alias (docs/devices.md)."""

import pytest
import torch

import grilly
import grilly.nn as gnn


def test_device_objects_behave_like_torch():
    for spec in ["cpu", "vulkan", "vulkan:0", "cuda:1"]:
        g, t = grilly.device(spec), torch.device(spec)
        assert (repr(g), str(g), g.type, g.index) == (repr(t), str(t), t.type, t.index)
    assert grilly.device("vulkan", 0) == grilly.device("vulkan:0")
    assert len({grilly.device("cpu"), grilly.device("cpu")}) == 1
    for lib in (grilly, torch):
        with pytest.raises(RuntimeError, match="device type at start of device string"):
            lib.device("nonsense")


def test_tensors_live_on_the_vulkan_device():
    x = grilly.zeros(2, device="cpu")
    assert x.device == grilly.device("vulkan", 0) and x.device.type == "vulkan"
    assert not x.is_cuda
    y = grilly.ones(2, device=x.device)          # the torch idiom round-trips
    assert y.device == x.device


def test_moves_to_the_same_device_return_self_like_torch():
    x = grilly.ones(2)
    for target in ["cpu", grilly.device("cpu"), x.device, "vulkan"]:
        assert x.to(target) is x
    assert x.cpu() is x and x.to(grilly.float32) is x and x.to(x) is x
    c = x.to("cpu", copy=True)
    assert c is not x and c.tolist() == x.tolist()
    t = torch.ones(2)
    assert t.to("cpu") is t and t.cpu() is t


def test_the_cuda_idiom_selects_cpu_and_runs():
    dev = grilly.device("cuda" if grilly.cuda.is_available() else "cpu")
    assert dev.type == "cpu" and grilly.cuda.device_count() == 0
    m = gnn.Linear(3, 2).to(dev)
    out = m(grilly.randn(4, 3, device=dev).to(dev))
    assert out.device.type == "vulkan"
    grilly.cuda.manual_seed(0)                   # ignored without CUDA, as torch


@pytest.mark.parametrize("call", [
    lambda: grilly.zeros(2, device="cuda"),
    lambda: grilly.ones(2).to("cuda"),
    lambda: grilly.ones(2).cuda(),
    lambda: gnn.Linear(2, 2).to("mps"),
    lambda: grilly.zeros(2, device="vulkan:1"),
])
def test_other_devices_are_not_implemented(call):
    with pytest.raises(NotImplementedError, match="device|Vulkan"):
        call()


def test_torch_alias_resolves_cuda_module():
    import subprocess, sys
    code = "import grilly.torch_alias, torch; print(torch.cuda.is_available(), torch.device('cpu'))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "cpu"]
