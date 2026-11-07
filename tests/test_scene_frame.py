"""Scene placement and GL row order."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scene_frame import estimate_frame  # noqa: E402
from xr_camera import intrinsics_from_fov, pose_to_matrix, project, viewmat_from_pose  # noqa: E402


def look(center, forward, down):
    """3x4 OpenCV world-to-camera for a camera at `center` looking along `forward` with image-down `down`."""
    z = forward / np.linalg.norm(forward)
    y = down - (down @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    r = np.stack([x, y, z])
    return np.c_[r, -r @ center]


def synthetic_scene():
    # World: +Y is down (COLMAP style), ground at y = +2, cameras at y = 0 (2 units above ground).
    down = np.array([0.0, 1.0, 0.0])
    cams = {}
    for i, ang in enumerate(np.linspace(0, 2 * np.pi, 12, endpoint=False)):
        c = np.array([5 * np.cos(ang), 0.0, 5 * np.sin(ang)])
        fwd = -c + np.array([0.0, 0.8, 0.0])  # aimed at the centre, pitched down a little
        cams[f"{i:05d}.jpg"] = look(c, fwd, down)
    rng = np.random.default_rng(0)
    ground = np.c_[rng.uniform(-8, 8, 4000), np.full(4000, 2.0), rng.uniform(-8, 8, 4000)]
    clutter = rng.uniform(-3, 3, size=(500, 3))
    return cams, np.concatenate([ground, clutter])


def test_up_and_scale_are_recovered():
    cams, pts = synthetic_scene()
    f = estimate_frame(cams, pts, "00000.jpg", camera_height_m=1.6)
    np.testing.assert_allclose(f.up_world, [0, -1, 0], atol=1e-6)
    assert abs(f.camera_height_units - 2.0) < 0.1
    assert abs(f.metres_per_unit - 0.8) < 0.05


def test_start_view_is_level_and_faces_the_photo_direction():
    cams, pts = synthetic_scene()
    f = estimate_frame(cams, pts, "00000.jpg")
    # A world point straight ahead of camera 0, at its height, must project to the image centre
    # when the head is at the local origin looking down -Z.
    c0 = -cams["00000.jpg"][:, :3].T @ cams["00000.jpg"][:, 3]
    ahead = c0 + np.array([-1.0, 0.0, 0.0]) * 3  # camera 0 sits at +X and faces the centre
    view_local = viewmat_from_pose(pose_to_matrix((0, 0, 0, 1), (0, 0, 0)))
    view_world = f.view_world(view_local)
    k = intrinsics_from_fov(-0.6, 0.6, 0.6, -0.6, 200, 200)
    u, v, z = project(view_world, k, ahead[None])
    assert abs(u[0] - 100) < 1e-6 and abs(v[0] - 100) < 1e-6 and z[0] > 0
    # Rotation part is orthonormal (rigid), so gsplat gets a valid view matrix.
    r = view_world[:3, :3]
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)


def test_head_motion_moves_in_metres():
    cams, pts = synthetic_scene()
    f = estimate_frame(cams, pts, "00000.jpg", camera_height_m=1.6)  # about 0.8 m per unit
    # Head moved one scene unit forward (local -Z): the world camera centre moves 1 unit forward.
    view_local = viewmat_from_pose(pose_to_matrix((0, 0, 0, 1), (0, 0, -f.metres_per_unit)))
    vw = f.view_world(view_local)
    centre = -vw[:3, :3].T @ vw[:3, 3]
    c0 = -cams["00000.jpg"][:, :3].T @ cams["00000.jpg"][:, 3]
    np.testing.assert_allclose(centre, c0 + np.array([-1.0, 0.0, 0.0]), atol=1e-9)


def test_swapchain_rows_are_flipped_for_gl():
    from viewer import to_swapchain_rgba

    img = torch.zeros(4, 3, 3)
    img[0] = 1.0  # top row white
    out = torch.zeros(4, 3, 4, dtype=torch.uint8)
    to_swapchain_rgba(img, out)
    assert out[3, :, :3].min() == 255  # GL row 3 is the top of the picture
    assert out[0, :, :3].max() == 0
