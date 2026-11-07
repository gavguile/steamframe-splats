"""Loader round trip on synthetic PLY files. No GPU needed."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splat_io import load_gaussians, write_ply  # noqa: E402


def test_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    n = 6
    means = rng.normal(size=(n, 3)).astype(np.float32)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    scales = rng.uniform(0.01, 0.5, size=(n, 3)).astype(np.float32)
    opacities = rng.uniform(0.05, 0.95, size=(n,)).astype(np.float32)
    colors = rng.normal(scale=0.3, size=(n, 16, 3)).astype(np.float32)
    path = tmp_path / "synthetic.ply"
    write_ply(path, means, quats, scales, opacities, colors)

    g = load_gaussians(path, device="cpu")

    assert len(g) == n
    np.testing.assert_allclose(g.means.numpy(), means, atol=1e-6)
    np.testing.assert_allclose(g.quats.numpy(), quats, atol=1e-6)
    np.testing.assert_allclose(g.scales.numpy(), scales, rtol=1e-5)
    np.testing.assert_allclose(g.opacities.numpy(), opacities, atol=1e-5)
    np.testing.assert_allclose(g.colors.numpy(), colors, atol=1e-6)
    assert g.sh_degree == 3


def test_channel_major_sh_layout(tmp_path):
    """f_rest is stored all-R, then all-G, then all-B. A single marked coefficient must land in the right slot."""
    colors = np.zeros((1, 16, 3), dtype=np.float32)
    colors[0, 5, 2] = 0.75  # SH index 5 (1-based 4 in f_rest), blue channel
    path = tmp_path / "sh.ply"
    write_ply(path, np.zeros((1, 3)), np.array([[1, 0, 0, 0]]), np.ones((1, 3)),
              np.array([0.5]), colors)
    g = load_gaussians(path, device="cpu")
    assert torch.isclose(g.colors[0, 5, 2], torch.tensor(0.75))
    assert torch.count_nonzero(g.colors) == 1
