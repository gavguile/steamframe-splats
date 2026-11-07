"""Load 3D Gaussian Splatting PLY files in the standard INRIA layout.

The layout is binary little-endian, one vertex per Gaussian, with fields:
  x y z, nx ny nz (unused), f_dc_0..2, f_rest_0..44, opacity,
  scale_0..2 (log scale), rot_0..3 (quaternion w, x, y, z; unnormalized).

The loader applies the activations the renderer expects:
  scales = exp(scale), opacities = sigmoid(opacity), quats normalized.
SH coefficients: f_rest is stored channel-major (all 15 R coefficients, then 15 G,
then 15 B). The result is (N, 16, 3): DC then 15 higher-order terms per channel.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SH_DEGREE = 3
SH_COEFFS = (SH_DEGREE + 1) ** 2  # 16


@dataclass
class Gaussians:
    means: torch.Tensor      # (N, 3)
    quats: torch.Tensor      # (N, 4) w, x, y, z; normalized
    scales: torch.Tensor     # (N, 3) linear scale
    opacities: torch.Tensor  # (N,)
    colors: torch.Tensor     # (N, 16, 3) SH coefficients
    sh_degree: int = SH_DEGREE

    def to(self, device):
        return Gaussians(self.means.to(device), self.quats.to(device), self.scales.to(device),
                         self.opacities.to(device), self.colors.to(device), self.sh_degree)

    def __len__(self):
        return self.means.shape[0]


def read_ply(path):
    """Return (property names, structured array of float32 vertices) from a binary PLY."""
    path = Path(path)
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            text = line.decode("ascii").strip()
            header.append(text)
            if text == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"{path}: only binary_little_endian PLY is supported")
        n_vertices = None
        names = []
        in_vertex = False
        for line in header:
            parts = line.split()
            if parts[:1] == ["element"]:
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    n_vertices = int(parts[2])
            elif parts[:1] == ["property"] and in_vertex:
                if parts[1] != "float":
                    raise ValueError(f"{path}: expected float properties, found {parts[1]}")
                names.append(parts[2])
        if n_vertices is None:
            raise ValueError(f"{path}: no vertex element")
        dtype = np.dtype([(name, "<f4") for name in names])
        data = np.fromfile(f, dtype=dtype, count=n_vertices)
    if data.shape[0] != n_vertices:
        raise ValueError(f"{path}: expected {n_vertices} vertices, read {data.shape[0]}")
    return names, data


def load_gaussians(path, device="cuda"):
    names, v = read_ply(path)
    needed = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"]
    missing = [n for n in needed if n not in names]
    if missing:
        raise ValueError(f"{path}: missing properties {missing}")
    n = v.shape[0]

    means = np.stack([v["x"], v["y"], v["z"]], axis=1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1))
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"]))
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)[:, None, :]

    rest_names = [f"f_rest_{i}" for i in range(3 * (SH_COEFFS - 1))]
    if all(name in names for name in rest_names):
        f_rest = np.stack([v[name] for name in rest_names], axis=1)  # (N, 45)
        f_rest = f_rest.reshape(n, 3, SH_COEFFS - 1).transpose(0, 2, 1)  # (N, 15, 3)
    else:
        f_rest = np.zeros((n, SH_COEFFS - 1, 3), dtype=np.float32)
    colors = np.concatenate([f_dc, f_rest], axis=1).astype(np.float32)  # (N, 16, 3)

    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    g = Gaussians(to_t(means), to_t(quats), to_t(scales), to_t(opacities), to_t(colors))
    return g.to(device)


def write_ply(path, means, quats, scales, opacities, colors):
    """Write a binary INRIA-layout PLY from already-activated values. Used for tests.

    colors: (N, 16, 3) SH coefficients. scales: linear scale (stored as log).
    """
    means = np.asarray(means, dtype=np.float32)
    n = means.shape[0]
    colors = np.asarray(colors, dtype=np.float32)
    f_dc = colors[:, 0, :]
    f_rest = colors[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)  # channel-major
    log_scales = np.log(np.asarray(scales, dtype=np.float32))
    logit_op = np.log(np.asarray(opacities, dtype=np.float32) / (1 - np.asarray(opacities, dtype=np.float32)))
    columns = {
        "x": means[:, 0], "y": means[:, 1], "z": means[:, 2],
        "nx": np.zeros(n, np.float32), "ny": np.zeros(n, np.float32), "nz": np.zeros(n, np.float32),
    }
    for i in range(3):
        columns[f"f_dc_{i}"] = f_dc[:, i]
    for i in range(f_rest.shape[1]):
        columns[f"f_rest_{i}"] = f_rest[:, i]
    columns["opacity"] = logit_op
    for i in range(3):
        columns[f"scale_{i}"] = log_scales[:, i]
    for i in range(4):
        columns[f"rot_{i}"] = np.asarray(quats, dtype=np.float32)[:, i]
    names = list(columns)
    data = np.zeros(n, dtype=np.dtype([(name, "<f4") for name in names]))
    for name in names:
        data[name] = columns[name]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {name}" for name in names]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())
