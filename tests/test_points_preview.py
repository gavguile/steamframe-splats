"""Depth-buffered point drawing used by the S6 preview. Runs on CPU."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from s6_points_preview import draw_points  # noqa: E402
from xr_camera import intrinsics_from_fov, pose_to_matrix, viewmat_from_pose  # noqa: E402

W = H = 64


def identity_view():
    return torch.tensor(viewmat_from_pose(pose_to_matrix((0, 0, 0, 1), (0, 0, 0))), dtype=torch.float32)


def symmetric_k():
    return torch.tensor(intrinsics_from_fov(-0.5, 0.5, 0.5, -0.5, W, H), dtype=torch.float32)


def test_nearer_point_wins_on_a_shared_pixel():
    # Two points on the same ray: the far one is written first in the list, the near one second.
    pts = torch.tensor([[0.0, 0.0, -4.0], [0.0, 0.0, -2.0]])
    rgb = torch.tensor([[255, 0, 0], [0, 255, 0]], dtype=torch.uint8)
    buf = torch.zeros((H, W, 4), dtype=torch.uint8)
    draw_points(buf, pts, rgb, identity_view(), symmetric_k(), W, H)
    assert tuple(buf[H // 2, W // 2, :3].tolist()) == (0, 255, 0)


def test_footprint_fills_a_block_of_pixels():
    pts = torch.tensor([[0.0, 0.0, -2.0]])
    rgb = torch.tensor([[200, 100, 50]], dtype=torch.uint8)
    buf = torch.zeros((H, W, 4), dtype=torch.uint8)
    draw_points(buf, pts, rgb, identity_view(), symmetric_k(), W, H, radius=1)
    lit = buf[..., :3].sum(-1) > 0
    rows, cols = torch.nonzero(lit, as_tuple=True)
    assert int(lit.sum()) == 9
    assert int(rows.max() - rows.min()) == 2 and int(cols.max() - cols.min()) == 2


def test_drawing_is_deterministic_for_many_collisions():
    rng = np.random.default_rng(1)
    n = 20000
    pts = torch.tensor(rng.uniform(-0.3, 0.3, size=(n, 3)), dtype=torch.float32)
    pts[:, 2] = -torch.tensor(rng.uniform(1.0, 3.0, size=n), dtype=torch.float32)  # in front, many collide
    rgb = torch.tensor(rng.integers(0, 256, size=(n, 3)), dtype=torch.uint8)
    a = torch.zeros((H, W, 4), dtype=torch.uint8)
    b = torch.zeros((H, W, 4), dtype=torch.uint8)
    draw_points(a, pts, rgb, identity_view(), symmetric_k(), W, H)
    draw_points(b, pts, rgb, identity_view(), symmetric_k(), W, H)
    assert torch.equal(a, b)
