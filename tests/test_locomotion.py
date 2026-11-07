"""Free-fly rig maths."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from locomotion import SPEED_M_S, Rig  # noqa: E402
from xr_camera import pose_to_matrix  # noqa: E402

HEAD = pose_to_matrix((0, 0, 0, 1), (0.0, 1.7, 0.0))  # standing, looking down -Z


def test_left_stick_forward_flies_where_the_head_looks():
    rig = Rig()
    rig.update(HEAD, (0, 1), (0, 0), boost=False, dt=1.0)
    np.testing.assert_allclose(rig.pos, [0, 0, -SPEED_M_S], atol=1e-9)


def test_looking_down_and_pushing_forward_descends():
    a = math.radians(-45) / 2
    head = pose_to_matrix((math.sin(a), 0, 0, math.cos(a)), (0, 1.7, 0))
    rig = Rig()
    rig.update(head, (0, 1), (0, 0), boost=False, dt=1.0)
    assert rig.pos[1] < -1.0 and rig.pos[2] < -1.0


def test_right_stick_up_rises_and_boost_triples_speed():
    rig = Rig()
    rig.update(HEAD, (0, 0), (0, 1), boost=True, dt=1.0)
    np.testing.assert_allclose(rig.pos, [0, 3 * SPEED_M_S, 0], atol=1e-9)


def test_small_stick_noise_is_ignored():
    rig = Rig()
    rig.update(HEAD, (0.1, -0.1), (0.05, 0.1), boost=False, dt=1.0)
    np.testing.assert_allclose(rig.pos, 0, atol=1e-12)


def test_snap_turn_right_once_per_flick_and_keeps_head_in_place():
    rig = Rig()
    for _ in range(5):  # stick held right: one turn only
        rig.update(HEAD, (0, 0), (1.0, 0), boost=False, dt=0.01)
    assert math.isclose(rig.yaw, math.radians(-30))
    head = rig.head_virtual(HEAD)
    np.testing.assert_allclose(head[:3, 3], [0, 1.7, 0], atol=1e-9)  # turned in place
    fwd = -head[:3, 2]
    assert fwd[0] > 0.4  # now facing right (+X)
    rig.update(HEAD, (0, 0), (0.0, 0), boost=False, dt=0.01)  # release
    rig.update(HEAD, (0, 0), (1.0, 0), boost=False, dt=0.01)  # flick again
    assert math.isclose(rig.yaw, math.radians(-60))
