"""OpenXR view poses and field of view to OpenCV cameras (gsplat's convention).

OpenXR: right-handed, Y up, a view looks down -Z. Each view has a pose (camera
to local space) and an asymmetric field of view given as four angles.
OpenCV (gsplat): x right, y down, z forward.

Conversions:
  viewmat (world -> camera, OpenCV) = D @ inverse(pose_matrix), D = diag(1, -1, -1)
  K from the FOV angles, for a render of width W and height H:
    fx = W / (tan r - tan l),   cx = -W tan l / (tan r - tan l)
    fy = H / (tan u - tan d),   cy =  H tan u / (tan u - tan d)
  Here l, r, u, d are the left, right, up, down angles (left and down negative).
"""

import math

import numpy as np

D = np.diag([1.0, -1.0, -1.0, 1.0])


def pose_to_matrix(orientation, position):
    """4x4 camera-to-local matrix from an OpenXR quaternion (x, y, z, w) and position."""
    x, y, z, w = orientation
    m = np.eye(4)
    m[0, 0] = 1 - 2 * (y * y + z * z)
    m[0, 1] = 2 * (x * y - z * w)
    m[0, 2] = 2 * (x * z + y * w)
    m[1, 0] = 2 * (x * y + z * w)
    m[1, 1] = 1 - 2 * (x * x + z * z)
    m[1, 2] = 2 * (y * z - x * w)
    m[2, 0] = 2 * (x * z - y * w)
    m[2, 1] = 2 * (y * z + x * w)
    m[2, 2] = 1 - 2 * (x * x + y * y)
    m[:3, 3] = position
    return m


def viewmat_from_pose(camera_to_local):
    """World (local space) to OpenCV camera, from the OpenXR camera-to-local matrix."""
    return D @ np.linalg.inv(camera_to_local)


def intrinsics_from_fov(angle_left, angle_right, angle_up, angle_down, width, height):
    """3x3 pinhole K for an asymmetric OpenXR field of view at a render size of width x height."""
    tl, tr = math.tan(angle_left), math.tan(angle_right)
    tu, td = math.tan(angle_up), math.tan(angle_down)
    fx = width / (tr - tl)
    cx = -width * tl / (tr - tl)
    fy = height / (tu - td)
    cy = height * tu / (tu - td)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def project(viewmat, k, points_local):
    """Project local-space points to pixels. Returns (u, v, depth) for points in front of the camera."""
    p = np.c_[points_local, np.ones(len(points_local))]
    pc = (viewmat @ p.T).T[:, :3]
    z = pc[:, 2]
    u = k[0, 0] * pc[:, 0] / z + k[0, 2]
    v = k[1, 1] * pc[:, 1] / z + k[1, 2]
    return u, v, z
