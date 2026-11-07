"""Place the COLMAP scene in the headset's local space.

The user starts where a training photo was taken, facing the same way, with the
horizon level. The scene is scaled to metres from an estimated camera height.

- Up: the average "up" of all training cameras. A hand-held camera is roughly
  upright, so the average is close to the true vertical. COLMAP's own axes are
  arbitrary.
- Start pose: the chosen photo's camera centre, with its forward direction
  flattened onto the horizontal plane. Head pitch and roll come from the headset.
- Scale: metres per scene unit = CAMERA_HEIGHT_M / (camera height above ground in
  scene units). The ground is the densest height band of the sparse points
  below the cameras.

Rendering does not move the Gaussians. Each eye's OpenXR view (local space,
metres) is composed with local_from_world to give a rigid world-to-camera matrix
in scene units. The camera position then stays in world units, so the SH view
direction stays correct with no SH rotation.
"""

from dataclasses import dataclass

import numpy as np

CAMERA_HEIGHT_M = 1.6  # assumed height of the hand-held camera above the ground


@dataclass
class SceneFrame:
    rotation: np.ndarray       # 3x3, local (OpenXR axes) from world directions
    origin_world: np.ndarray   # world point placed at the local origin (the start head position)
    metres_per_unit: float
    up_world: np.ndarray
    camera_height_units: float

    def local_from_world(self, p):
        p = np.atleast_2d(p)
        return self.metres_per_unit * (p - self.origin_world) @ self.rotation.T

    def view_world(self, view_local_cv):
        """Rigid world-to-camera (OpenCV, scene units) from a local-space viewmat (OpenCV, metres).

        view_local = [A_R | A_t]; local_from_world = s * R (p - C).
        Composite = [s A_R R | -s A_R R C + A_t], divided by s (projection is unchanged by a uniform scale).
        """
        a_r, a_t = view_local_cv[:3, :3], view_local_cv[:3, 3]
        s, r, c = self.metres_per_unit, self.rotation, self.origin_world
        out = np.eye(4)
        out[:3, :3] = a_r @ r
        out[:3, 3] = -(a_r @ r) @ c + a_t / s
        return out


def estimate_frame(cam_from_world_by_name, points_world, start_name, camera_height_m=CAMERA_HEIGHT_M):
    """cam_from_world_by_name: {name: 3x4 OpenCV world-to-camera}. points_world: (N, 3) sparse points."""
    rots = {n: m[:, :3] for n, m in cam_from_world_by_name.items()}
    centers = {n: -m[:, :3].T @ m[:, 3] for n, m in cam_from_world_by_name.items()}

    down = np.mean([r.T @ np.array([0.0, 1.0, 0.0]) for r in rots.values()], axis=0)
    up = -down / np.linalg.norm(down)

    r0 = rots[start_name]
    c0 = centers[start_name]
    fwd = r0.T @ np.array([0.0, 0.0, 1.0])
    fwd_h = fwd - (fwd @ up) * up
    fwd_h /= np.linalg.norm(fwd_h)
    right = np.cross(fwd_h, up)
    rotation = np.stack([right, up, -fwd_h])  # rows: local x, y, z in world coordinates

    # Ground: densest height band of points below the mean camera height.
    cam_heights = np.array([c @ up for c in centers.values()])
    mean_cam_h = cam_heights.mean()
    spread = np.ptp(cam_heights) + 1e-9
    heights = points_world @ up
    below = heights[(heights < mean_cam_h) & (heights > mean_cam_h - 20 * max(spread, 1.0))]
    hist, edges = np.histogram(below, bins=200)
    i = int(np.argmax(hist))
    ground_h = 0.5 * (edges[i] + edges[i + 1])
    cam_height_units = float(mean_cam_h - ground_h)
    s = camera_height_m / cam_height_units

    return SceneFrame(rotation=rotation, origin_world=c0, metres_per_unit=s,
                      up_world=up, camera_height_units=cam_height_units)


def load_colmap_frame(colmap_dir, start_name="00002.jpg"):
    import pycolmap
    rec = pycolmap.Reconstruction()
    rec.read_binary(str(colmap_dir))
    cams = {im.name: np.asarray(im.cam_from_world().matrix()) for im in rec.images.values()}
    pts = np.array([p.xyz for p in rec.points3D.values()])
    return estimate_frame(cams, pts, start_name)
