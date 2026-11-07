"""gsplat render checks on synthetic Gaussians. Needs CUDA."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splat_io import Gaussians  # noqa: E402
from s5_offline_render import intrinsics, look_at_w2c, render  # noqa: E402

SH_C0 = 0.28209479177387814  # gsplat's DC basis constant: color = C0 * dc + 0.5

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def one_splat(position, rgb, opacity=0.999):
    dev = "cuda"
    colors = torch.zeros(1, 16, 3, device=dev)
    colors[0, 0] = (torch.tensor(rgb, device=dev) - 0.5) / SH_C0
    return Gaussians(
        means=torch.tensor([position], dtype=torch.float32, device=dev),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev),
        scales=torch.full((1, 3), 0.4, device=dev),
        opacities=torch.tensor([opacity], device=dev),
        colors=colors,
        sh_degree=3,
    )


def weighted_centroid(img):
    """Centroid (col, row) of the bright pixels, weighted by brightness."""
    w = img.sum(dim=2)
    total = w.sum()
    rows = torch.arange(img.shape[0], device=img.device, dtype=torch.float32)
    cols = torch.arange(img.shape[1], device=img.device, dtype=torch.float32)
    return float((w.sum(dim=0) * cols).sum() / total), float((w.sum(dim=1) * rows).sum() / total)


def camera_at_origin_looking_forward():
    eye = torch.tensor([0.0, 0.0, -3.0], device="cuda")
    target = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    return look_at_w2c(eye, target, device="cuda"), intrinsics(64, 64, 60.0, device="cuda")


def test_dc_color_round_trips_at_center():
    rgb = [1.0, 0.2, 0.0]
    g = one_splat([0.0, 0.0, 0.0], rgb)
    view, k = camera_at_origin_looking_forward()
    img = render(g, view, k, 64, 64)
    center = img[30:34, 30:34].mean(dim=(0, 1)).cpu()
    assert torch.allclose(center, torch.tensor(rgb) * 0.999, atol=0.03), center


def test_world_plus_x_is_image_right():
    view, k = camera_at_origin_looking_forward()
    img = render(one_splat([0.6, 0.0, 0.0], [1, 1, 1]), view, k, 64, 64)
    col, _ = weighted_centroid(img)
    assert col > 33.0, col


def test_world_plus_y_is_image_down_for_colmap_scenes():
    view, k = camera_at_origin_looking_forward()
    img = render(one_splat([0.0, 0.6, 0.0], [1, 1, 1]), view, k, 64, 64)
    _, row = weighted_centroid(img)
    assert row > 33.0, row


def test_xr_render_depth_sort_matches_gsplat():
    from xr_render import render_xr

    rng = torch.Generator(device="cuda").manual_seed(0)
    n = 300
    colors = torch.randn(n, 16, 3, device="cuda", generator=rng) * 0.3
    g = Gaussians(
        means=torch.randn(n, 3, device="cuda", generator=rng) * 0.5,
        quats=torch.nn.functional.normalize(torch.randn(n, 4, device="cuda", generator=rng), dim=1),
        scales=torch.rand(n, 3, device="cuda", generator=rng) * 0.1 + 0.01,
        opacities=torch.rand(n, device="cuda", generator=rng),
        colors=colors,
        sh_degree=3,
    )
    view, k = camera_at_origin_looking_forward()
    ref = render(g, view, k, 64, 64)
    ours = render_xr(g, view, k, 64, 64, sort="depth")
    assert torch.allclose(ref, ours, atol=1e-5)


def test_xr_render_near_distance_hides_close_splat():
    from xr_render import render_xr

    view, k = camera_at_origin_looking_forward()  # eye at z = -3
    g = one_splat([0.0, 0.0, -2.5], [1, 1, 1])     # 0.5 from the eye
    assert render_xr(g, view, k, 64, 64, near_distance=0.4).max() > 0.5
    assert render_xr(g, view, k, 64, 64, near_distance=0.6).max() == 0
