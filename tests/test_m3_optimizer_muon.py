import torch

from onyx_model import M3Optimizer, zeropower_via_newtonschulz5


def test_newton_schulz_zeropower_returns_approximately_orthogonal_square():
    torch.manual_seed(0)
    G = torch.randn(8, 8, dtype=torch.float32)
    X = zeropower_via_newtonschulz5(G, steps=5)
    assert X.shape == G.shape
    assert torch.isfinite(X).all()

    I = torch.eye(G.size(0), dtype=torch.float32)
    XT_X = (X.float().T @ X.float())
    # Allow loose tolerance due to iterative approximation + dtype differences.
    assert torch.allclose(XT_X, I, atol=0.25, rtol=0.25)


def test_m3_optimizer_matrix_update_matches_newton_schulz_direction():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.float32))
    opt = M3Optimizer(
        [p],
        lr=0.1,
        momentum=0.0,
        beta_slow=0.9,
        slow_freq=1000,
        slow_weight=0.1,
        ns_steps=3,
        weight_decay=0.0,
        use_nesterov=True,
    )

    g = torch.randn_like(p)
    p.grad = g.clone()
    expected_update = zeropower_via_newtonschulz5(g, steps=3)

    opt.step()
    assert torch.allclose(p.data, -0.1 * expected_update, atol=1e-6, rtol=1e-6)


def test_m3_optimizer_vector_fallback_matches_sgd_momentum_nesterov():
    p = torch.nn.Parameter(torch.full((5,), 2.0, dtype=torch.float32))
    opt = M3Optimizer(
        [p],
        lr=0.1,
        momentum=0.9,
        beta_slow=0.99,
        slow_freq=50,
        slow_weight=0.1,
        ns_steps=5,
        weight_decay=0.1,
        use_nesterov=True,
    )

    p.grad = torch.ones_like(p)
    # Step 1:
    # decoupled wd: p *= (1 - lr*wd) => 2.0 * 0.99 = 1.98
    # momentum buf = g => 1
    # nesterov update = g + momentum*buf = 1 + 0.9*1 = 1.9
    # p -= lr*update => 1.98 - 0.19 = 1.79
    opt.step()
    assert torch.allclose(p.data, torch.full((5,), 1.79), atol=1e-6, rtol=0.0)


def test_m3_optimizer_slow_memory_updates_only_on_frequency():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float32))
    opt = M3Optimizer(
        [p],
        lr=0.01,
        momentum=0.0,
        beta_slow=0.5,
        slow_freq=2,
        slow_weight=0.1,
        ns_steps=1,
        weight_decay=0.0,
        use_nesterov=False,
    )
    g = torch.ones_like(p)

    p.grad = g
    opt.step()
    st = opt.state[p]
    assert st["step"] == 1
    assert torch.allclose(st["slow_memory"], torch.zeros_like(p))

    p.grad = g
    opt.step()
    st = opt.state[p]
    assert st["step"] == 2
    # slow_memory <- 0.5*0 + (1-0.5)*buf; buf == g when momentum=0
    assert torch.allclose(st["slow_memory"], 0.5 * g)

