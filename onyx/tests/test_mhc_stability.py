import torch
from onyx.experimental.mhc import sinkhorn_project, MHCMixer, MHCLiteMixer


def test_sinkhorn_stress_logits_finite():
    torch.manual_seed(0)
    scales = [1, 5, 10, 20, 50]
    for s in scales:
        mat = torch.randn(8, 8) * s
        mat = mat.clamp(min=-50, max=50)
        mat = mat.exp()
        proj = sinkhorn_project(mat, iters=50, eps=1e-8)
        assert torch.isfinite(proj).all()
        # rows/cols are approximately normalized
        row = proj.sum(dim=-1)
        col = proj.sum(dim=-2)
        assert torch.allclose(row, torch.ones_like(row), atol=5e-2)
        assert torch.allclose(col, torch.ones_like(col), atol=5e-2)


def test_sinkhorn_fp16_input_returns_fp32():
    mat = torch.rand(4, 4, dtype=torch.float16)
    proj = sinkhorn_project(mat, iters=5, eps=1e-8)
    assert proj.dtype == torch.float32


def test_mhc_mixer_no_nan_with_large_matrix():
    torch.manual_seed(0)
    mixer = MHCMixer(n_streams=2, mode="mhc", use_sinkhorn=True, sinkhorn_iters=5)
    with torch.no_grad():
        mixer.matrix.mul_(10.0)
    x = torch.randn(2, 4, 2, 8)
    out = mixer(x)
    assert torch.isfinite(out).all()


def test_mhc_mixer_train_steps_no_nan():
    torch.manual_seed(0)
    mixer = MHCMixer(n_streams=2, mode="mhc", use_sinkhorn=True, sinkhorn_iters=5)
    opt = torch.optim.SGD(mixer.parameters(), lr=1e-2)
    for _ in range(20):
        x = torch.randn(2, 4, 2, 8)
        out = mixer(x)
        loss = out.float().pow(2).mean()
        assert torch.isfinite(loss).all()
        opt.zero_grad()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in mixer.parameters())
        opt.step()


def test_mhc_lite_mixer_train_steps_no_nan():
    torch.manual_seed(0)
    mixer = MHCLiteMixer(n_streams=3)
    opt = torch.optim.SGD(mixer.parameters(), lr=1e-2)
    for _ in range(20):
        x = torch.randn(2, 4, 3, 8)
        out = mixer(x)
        loss = out.float().pow(2).mean()
        assert torch.isfinite(loss).all()
        opt.zero_grad()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in mixer.parameters())
        opt.step()
