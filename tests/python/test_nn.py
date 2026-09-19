"""grilly.nn, nn.functional, nn.init, nn.utils and optim against torch.

Models are built in both libraries with torch's weights loaded into grilly,
then run step by step: outputs, gradients and optimizer updates are compared.
Tolerance: rtol 1e-4, atol 1e-5 after several optimizer steps (errors from
the float32 ops compound through the updates); rtol 1e-5 / atol 1e-6 for
single forward/backward passes.
"""

import copy

import numpy as np
import pytest
import torch

import grilly
import grilly.nn as gnn
import grilly.nn.functional as GF


def rand(*shape, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, shape).astype(np.float32)


def g_(a):
    return grilly.tensor(np.asarray(a, np.float32))


def close(g, t, rtol=1e-5, atol=1e-6):
    t = t.detach().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    g = g.detach().numpy() if isinstance(g, grilly.Tensor) else np.asarray(g)
    assert g.shape == t.shape, (g.shape, t.shape)
    np.testing.assert_allclose(g, t, rtol=rtol, atol=atol)


def load_from_torch(gmodel, tmodel):
    gmodel.load_state_dict({k: g_(v.detach().numpy()) for k, v in tmodel.state_dict().items()})


def build(lib):
    nn = lib.nn
    return nn.Sequential(
        nn.Linear(6, 16), nn.GELU(), nn.LayerNorm(16),
        nn.Linear(16, 16), nn.Tanh(), nn.Identity(),
        nn.Linear(16, 3, bias=False),
    )


@pytest.mark.parametrize("opt_name,kwargs", [
    ("Adam", dict(lr=1e-2)),
    ("Adam", dict(lr=1e-2, weight_decay=0.1, amsgrad=True)),
    ("AdamW", dict(lr=1e-2, weight_decay=0.1)),
    ("SGD", dict(lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.01)),
])
def test_training_steps_match_torch(opt_name, kwargs):
    torch.manual_seed(0)
    tm = build(torch)
    gm = build(grilly)
    load_from_torch(gm, tm)
    topt = getattr(torch.optim, opt_name)(tm.parameters(), **kwargs)
    gopt = getattr(grilly.optim, opt_name)(gm.parameters(), **kwargs)
    for step in range(5):
        x, y = rand(8, 6, seed=10 + step), rand(8, 3, seed=20 + step)
        tloss = torch.nn.functional.mse_loss(tm(torch.tensor(x)), torch.tensor(y))
        gloss = GF.mse_loss(gm(g_(x)), g_(y))
        close(gloss, tloss, rtol=1e-4, atol=1e-5)
        topt.zero_grad()
        gopt.zero_grad()
        tloss.backward()
        gloss.backward()
        for (name, gp), tp in zip(gm.named_parameters(), tm.parameters()):
            close(gp.grad, tp.grad, rtol=1e-4, atol=1e-5)
        topt.step()
        gopt.step()
    for (name, gp), tp in zip(gm.named_parameters(), tm.parameters()):
        close(gp, tp, rtol=1e-4, atol=1e-5)


def test_parameter_registration_and_naming_match_torch():
    def make(lib):
        class Net(lib.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = lib.nn.Linear(2, 3)
                self.blocks = lib.nn.ModuleList([lib.nn.Linear(3, 3), lib.nn.Tanh()])
                self.scale = lib.nn.Parameter(lib.ones(3))
                self.register_buffer("running", lib.zeros(3))
                self.register_buffer("tmp", lib.zeros(1), persistent=False)
                self.shared = self.a                  # the same module twice
                self.plain = 5

            def forward(self, x):
                return self.blocks[1](self.blocks[0](self.a(x))) * self.scale
        return Net()

    g, t = make(grilly), make(torch)
    assert [n for n, _ in g.named_parameters()] == [n for n, _ in t.named_parameters()]
    assert [n for n, _ in g.named_buffers()] == [n for n, _ in t.named_buffers()]
    assert list(g.state_dict()) == list(t.state_dict())
    assert [n for n, _ in g.named_modules()] == [n for n, _ in t.named_modules()]
    assert [n for n, _ in g.named_children()] == [n for n, _ in t.named_children()]
    assert g.plain == 5 and isinstance(g.scale, gnn.Parameter) and g.scale.requires_grad
    assert g.train(False) is g and not g.blocks[0].training
    g.train()
    assert g.blocks[1].training
    g.requires_grad_(False)
    assert not any(p.requires_grad for p in g.parameters())
    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        g.missing
    with pytest.raises(TypeError):
        g.scale = grilly.ones(3)                     # a plain tensor onto a parameter slot


def test_load_state_dict_checks_like_torch():
    g = gnn.Linear(3, 2)
    good = {"weight": grilly.ones(2, 3), "bias": grilly.zeros(2)}
    assert g.load_state_dict(good).missing_keys == []
    close(g.weight, np.ones((2, 3)))
    with pytest.raises(RuntimeError, match="Missing key"):
        g.load_state_dict({"weight": grilly.ones(2, 3)})
    with pytest.raises(RuntimeError, match="Unexpected key"):
        g.load_state_dict({**good, "extra": grilly.ones(1)})
    with pytest.raises(RuntimeError, match="size mismatch"):
        g.load_state_dict({"weight": grilly.ones(3, 3), "bias": grilly.zeros(2)})
    result = g.load_state_dict({"weight": grilly.ones(2, 3)}, strict=False)
    assert result.missing_keys == ["bias"]


def test_containers_behave_like_torch():
    for lib in (grilly, torch):
        seq = lib.nn.Sequential(lib.nn.Tanh(), lib.nn.Identity(), lib.nn.Tanh())
        assert len(seq) == 3 and isinstance(seq[-1], lib.nn.Tanh)
        assert len(seq[1:]) == 2 and isinstance(seq[1:], lib.nn.Sequential)
        seq.append(lib.nn.Identity())
        assert len(seq) == 4
        ml = lib.nn.ModuleList([lib.nn.Tanh()])
        ml += [lib.nn.Identity()]
        ml.insert(0, lib.nn.Tanh())
        assert [type(m).__name__ for m in ml] == ["Tanh", "Tanh", "Identity"]
        md = lib.nn.ModuleDict({"a": lib.nn.Tanh()})
        md["b"] = lib.nn.Identity()
        assert list(md.keys()) == ["a", "b"] and "a" in md


def test_forward_missing_raises_like_torch():
    for lib in (grilly, torch):
        with pytest.raises(NotImplementedError, match="forward"):
            lib.nn.Module()(1)


def test_init_distributions_and_determinism():
    w = grilly.empty(64, 32)
    gnn.init.kaiming_uniform_(w, a=5 ** 0.5)
    bound = 1 / 32 ** 0.5
    v = w.numpy()
    assert v.min() >= -bound and v.max() <= bound and v.std() > bound / 3
    gnn.init.xavier_normal_(w)
    assert abs(w.numpy().std() - (2 / 96) ** 0.5) < 0.02
    gnn.init.constant_(w, 0.5)
    assert (w.numpy() == 0.5).all()
    a, b = gnn.Linear(5, 4), gnn.Linear(5, 4)
    grilly.manual_seed(3)
    a.reset_parameters()
    grilly.manual_seed(3)
    b.reset_parameters()
    close(a.weight, b.weight.detach().numpy(), rtol=0, atol=0)


def test_clip_grad_norm_matches_torch():
    grads = [rand(3, 4, seed=1) * 5, rand(7, seed=2) * 5]
    tp = [torch.zeros(*g.shape, requires_grad=True) for g in grads]
    gp = [grilly.zeros(*g.shape, requires_grad=True) for g in grads]
    for t, g, d in zip(tp, gp, grads):
        t.grad = torch.tensor(d)
        g.grad = g_(d)
    tn = torch.nn.utils.clip_grad_norm_(tp, 1.0)
    gn = gnn.utils.clip_grad_norm_(gp, 1.0)
    close(gn, tn)
    for t, g in zip(tp, gp):
        close(g.grad, t.grad)


@pytest.mark.parametrize("fn", [
    lambda F, a, b: F.cosine_similarity(a, b),
    lambda F, a, b: F.cosine_similarity(a, b * 1e-9, dim=1, eps=1e-8),
    lambda F, a, b: F.cross_entropy(a, F.softmax(b, 1)),
    lambda F, a, b: F.cross_entropy(a, F.softmax(b, 1), label_smoothing=0.1, reduction="sum"),
    lambda F, a, b: F.mse_loss(a, b, reduction="none"),
    lambda F, a, b: F.gelu(a, approximate="tanh") + F.silu(b) + F.sigmoid(a) * F.tanh(b),
    lambda F, a, b: F.layer_norm(a, (5,)) + F.normalize(b, dim=0),
])
def test_functional_matches_torch(fn):
    a, b = rand(4, 5, seed=5), rand(4, 5, seed=6)
    ga, ta = grilly.tensor(a, requires_grad=True), torch.tensor(a, requires_grad=True)
    gb, tb = grilly.tensor(b, requires_grad=True), torch.tensor(b, requires_grad=True)
    go, to = fn(GF, ga, gb), fn(torch.nn.functional, ta, tb)
    close(go, to)
    up = rand(*to.shape, seed=7)
    go.backward(g_(up))
    to.backward(torch.tensor(up))
    close(ga.grad, ta.grad, rtol=1e-4, atol=1e-5)
    close(gb.grad, tb.grad, rtol=1e-4, atol=1e-5)


def test_dropout_train_and_eval():
    d = gnn.Dropout(0.25)
    x = grilly.ones(200, 50)
    y = d(x).numpy()
    kept = y != 0
    assert abs(kept.mean() - 0.75) < 0.02
    np.testing.assert_allclose(y[kept], 1 / 0.75, rtol=1e-6)
    d.eval()
    assert d(x) is x


def test_deepcopy_gives_independent_parameters():
    m = gnn.Linear(3, 2)
    c = copy.deepcopy(m)
    close(c.weight, m.weight.detach().numpy(), rtol=0, atol=0)
    with grilly.no_grad():
        c.weight.fill_(0.0)
    assert m.weight.detach().numpy().any()
    assert isinstance(c.weight, gnn.Parameter) and c.weight.requires_grad


def test_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        gnn.Linear(2, 2).to("cuda")


@pytest.mark.parametrize("call", [
    lambda lib: lib.nn.Embedding(10, 4)(lib.zeros(3)),
    lambda lib: lib.nn.functional.cross_entropy(lib.zeros(4, 5), lib.zeros(4)),
])
def test_float_indices_raise_like_torch(call):
    with pytest.raises(RuntimeError):
        call(torch)
    with pytest.raises(RuntimeError):
        call(grilly)


def test_embedding_holds_a_weight_like_torch():
    for lib in (grilly, torch):
        e = lib.nn.Embedding(10, 4, padding_idx=0)
        assert tuple(e.weight.shape) == (10, 4) and e.weight.requires_grad
        assert not e.weight.detach().numpy()[0].any()
