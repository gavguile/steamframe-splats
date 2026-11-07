"""OpenXR pose and FOV conversions, checked against known projections."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xr_camera import intrinsics_from_fov, pose_to_matrix, project, viewmat_from_pose  # noqa: E402

IDENTITY_Q = (0.0, 0.0, 0.0, 1.0)
W, H = 640, 480


def yaw_quat(degrees):
    """Quaternion (x, y, z, w) for a rotation about +Y. Negative is a right turn (forward -> +X)."""
    a = math.radians(degrees) / 2
    return (0.0, math.sin(a), 0.0, math.cos(a))


def test_point_ahead_projects_to_center_at_its_depth():
    view = viewmat_from_pose(pose_to_matrix(IDENTITY_Q, (0, 0, 0)))
    k = intrinsics_from_fov(-0.7, 0.7, 0.5, -0.5, W, H)
    u, v, z = project(view, k, np.array([[0.0, 0.0, -2.0]]))
    assert abs(u[0] - W / 2) < 1e-6 and abs(v[0] - H / 2) < 1e-6
    assert abs(z[0] - 2.0) < 1e-9


def test_symmetric_fov_principal_point_is_image_center():
    k = intrinsics_from_fov(-0.6, 0.6, 0.45, -0.45, W, H)
    assert abs(k[0, 2] - W / 2) < 1e-9 and abs(k[1, 2] - H / 2) < 1e-9
    assert abs(k[0, 0] - W / (2 * math.tan(0.6))) < 1e-9


def test_asymmetric_fov_edges_land_on_image_edges():
    # A typical headset: the view is wider to one side than the other.
    l, r, u, d = -0.9, 0.7, 0.8, -0.6
    k = intrinsics_from_fov(l, r, u, d, W, H)
    view = viewmat_from_pose(pose_to_matrix(IDENTITY_Q, (0, 0, 0)))
    # Points on the frustum edges at depth 1 (camera looks down -Z).
    pts = np.array([
        [math.tan(r), 0.0, -1.0],   # right edge
        [math.tan(l), 0.0, -1.0],   # left edge
        [0.0, math.tan(u), -1.0],   # top edge
        [0.0, math.tan(d), -1.0],   # bottom edge
    ])
    uu, vv, _ = project(view, k, pts)
    assert uu[0] == pytest.approx(W, abs=1e-6)
    assert uu[1] == pytest.approx(0.0, abs=1e-6)
    assert vv[2] == pytest.approx(0.0, abs=1e-6)  # top edge is row 0
    assert vv[3] == pytest.approx(H, abs=1e-6)    # bottom edge is the last row


def test_turning_right_puts_the_point_on_the_right_ahead_into_center():
    # Turn the head right by 90 degrees: forward becomes +X. A point at +X must be straight ahead.
    view = viewmat_from_pose(pose_to_matrix(yaw_quat(-90), (0, 0, 0)))
    k = intrinsics_from_fov(-0.7, 0.7, 0.5, -0.5, W, H)
    u, v, z = project(view, k, np.array([[1.0, 0.0, 0.0]]))
    assert abs(u[0] - W / 2) < 1e-6 and abs(v[0] - H / 2) < 1e-6
    assert abs(z[0] - 1.0) < 1e-9


def test_point_behind_after_a_right_turn_has_negative_depth():
    # After a 90-degree right turn, forward is +X, so -X is behind the head.
    view = viewmat_from_pose(pose_to_matrix(yaw_quat(-90), (0, 0, 0)))
    k = intrinsics_from_fov(-0.7, 0.7, 0.5, -0.5, W, H)
    _, _, z = project(view, k, np.array([[-1.0, 0.0, 0.0]]))
    assert z[0] == pytest.approx(-1.0, abs=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gsplat_honors_asymmetric_principal_point():
    from splat_io import Gaussians
    from s5_offline_render import render

    l, r, u, d = -0.9, 0.7, 0.8, -0.6
    k_np = intrinsics_from_fov(l, r, u, d, W, H)
    view_np = viewmat_from_pose(pose_to_matrix(IDENTITY_Q, (0, 0, 0)))
    target = np.array([[0.35, -0.2, -3.0]])
    u_pred, v_pred, _ = project(view_np, k_np, target)

    dev = "cuda"
    colors = torch.zeros(1, 16, 3, device=dev)
    colors[0, 0] = (torch.tensor([1.0, 1.0, 1.0], device=dev) - 0.5) / 0.28209479177387814
    g = Gaussians(
        means=torch.tensor(target, dtype=torch.float32, device=dev),
        quats=torch.tensor([[1.0, 0, 0, 0]], device=dev),
        scales=torch.full((1, 3), 0.05, device=dev),
        opacities=torch.tensor([0.999], device=dev),
        colors=colors,
        sh_degree=3,
    )
    view = torch.tensor(view_np, dtype=torch.float32, device=dev)
    k = torch.tensor(k_np, dtype=torch.float32, device=dev)
    img = render(g, view, k, W, H).cpu().numpy()
    lum = img.sum(axis=2)
    rows, cols = np.nonzero(lum > 0.05)
    assert len(cols) > 0
    cu = float((lum * np.arange(W)[None, :]).sum() / lum.sum())
    cv = float((lum * np.arange(H)[:, None]).sum() / lum.sum())
    assert abs(cu - u_pred[0]) < 1.0, (cu, u_pred[0])
    assert abs(cv - v_pred[0]) < 1.0, (cv, v_pred[0])
