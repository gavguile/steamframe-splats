"""Free-fly locomotion: a movable rig between the headset's local space and the scene.

virtual_from_local = translate(pos) @ rotate_y(yaw). The headset still tracks head
motion inside the rig; the controller moves and turns the rig.

Controls (applied by viewer.py):
  left stick    fly along where the head looks (forward/back) and sideways
  right stick   x: snap turn 30 degrees per flick; y: straight up/down
  grip (either) hold for 3x speed
  A (right)     back to the start pose
Snap turns rotate around the head, so the view turns in place.
"""

import math

import numpy as np

SPEED_M_S = 1.5
BOOST = 3.0
SNAP_DEG = 30.0
DEADZONE = 0.15
SNAP_ON, SNAP_OFF = 0.7, 0.3


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def deadzone(v):
    return 0.0 if abs(v) < DEADZONE else (v - math.copysign(DEADZONE, v)) / (1 - DEADZONE)


class Rig:
    def __init__(self):
        self.reset()

    def reset(self):
        self.yaw = 0.0
        self.pos = np.zeros(3)
        self._snap_armed = True

    def matrix(self):
        m = np.eye(4)
        m[:3, :3] = rot_y(self.yaw)
        m[:3, 3] = self.pos
        return m

    def head_virtual(self, head_local):
        return self.matrix() @ head_local

    def update(self, head_local, left_stick, right_stick, boost, dt):
        """head_local: 4x4 camera-to-local of the head (or an eye). Sticks: (x, y) in [-1, 1]."""
        head = self.head_virtual(head_local)
        fwd, right = -head[:3, 2], head[:3, 0]
        speed = SPEED_M_S * (BOOST if boost else 1.0)
        lx, ly = deadzone(left_stick[0]), deadzone(left_stick[1])
        ry = deadzone(right_stick[1])
        self.pos = self.pos + (fwd * ly + right * lx + np.array([0.0, 1.0, 0.0]) * ry) * speed * dt

        rx = right_stick[0]
        if self._snap_armed and abs(rx) > SNAP_ON:
            self.snap_turn(-math.copysign(math.radians(SNAP_DEG), rx), head[:3, 3])
            self._snap_armed = False
        elif abs(rx) < SNAP_OFF:
            self._snap_armed = True

    def snap_turn(self, angle, pivot):
        """Turn by `angle` (radians, positive = left) around a vertical axis through `pivot`."""
        r = rot_y(angle)
        self.yaw += angle
        self.pos = pivot + r @ (self.pos - pivot)
