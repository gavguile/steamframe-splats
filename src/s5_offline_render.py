"""S5: offline splat render with gsplat. No headset, no OpenXR.

Modes:
  --explore   four views around the scene as one contact-sheet PNG, to choose a camera
  --render    one fixed camera as a full-size PNG (2644 x 2644 by default)
  --timing    stereo frames (two eyes per frame) timed on the GPU, plus peak VRAM

Camera convention: OpenCV (x right, y down, z forward), the convention gsplat
uses. The scene is COLMAP-style, so world +y points down.
Scene scale is not metric. Eye separation is set as a fraction of the scene
radius, and the radius comes from the central 10-90% range of the Gaussian centers.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization
from PIL import Image

from splat_io import load_gaussians

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "s5"
SCENE = ROOT / "data" / "tnt-train" / "point_cloud.ply"
DOWN = (0.0, 1.0, 0.0)  # world "down" for COLMAP-style scenes


def look_at_w2c(eye, target, device="cuda", down=DOWN):
    """World-to-camera 4x4 for a camera at `eye` looking at `target` (OpenCV axes)."""
    eye = torch.as_tensor(eye, dtype=torch.float32, device=device)
    target = torch.as_tensor(target, dtype=torch.float32, device=device)
    z = target - eye
    z = z / z.norm()
    d = torch.as_tensor(down, dtype=torch.float32, device=device)
    y = d - (d @ z) * z
    y = y / y.norm()
    x = torch.cross(y, z, dim=0)  # OpenCV axes are right-handed: x = y cross z
    r_c2w = torch.stack([x, y, z], dim=1)
    r_w2c = r_c2w.T
    v = torch.eye(4, device=device)
    v[:3, :3] = r_w2c
    v[:3, 3] = -r_w2c @ eye
    return v


def intrinsics(width, height, fov_deg, device="cuda"):
    f = 0.5 * width / math.tan(math.radians(fov_deg) / 2)
    return torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]],
                        dtype=torch.float32, device=device)


@torch.no_grad()
def render(g, viewmat, k, width, height, near_plane=0.01):
    """Render one view. Returns (H, W, 3) float tensor on the GPU. near_plane is in scene units."""
    out, _, _ = rasterization(
        g.means, g.quats, g.scales, g.opacities, g.colors,
        viewmat[None], k[None], width, height, sh_degree=g.sh_degree, near_plane=near_plane)
    return out[0]


def to_uint8(img):
    return (img.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()


def scene_frame(g):
    """Center and radius from the central 10-90% range, which ignores outlier Gaussians."""
    center = torch.median(g.means, dim=0).values
    lo = torch.quantile(g.means, 0.10, dim=0)
    hi = torch.quantile(g.means, 0.90, dim=0)
    radius = float(0.5 * (hi - lo).max())
    return center, radius


def explore(g, out_dir, size=480, fov=60.0, dist_scale=1.6, name="explore.png"):
    center, radius = scene_frame(g)
    dist = dist_scale * radius
    k = intrinsics(size, size, fov)
    tiles = []
    for yaw in (0, 90, 180, 270):
        t = math.radians(yaw)
        eye = center + torch.tensor([math.sin(t), 0.0, -math.cos(t)], device=center.device) * dist
        eye = eye.clone()
        eye[1] = center[1]
        view = look_at_w2c(eye, center, device=center.device)
        tiles.append(to_uint8(render(g, view, k, size, size)))
    sheet = np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0)
    path = out_dir / name
    Image.fromarray(sheet).save(path)
    info = {"center": center.tolist(), "radius": radius, "dist": dist, "fov_deg": fov}
    return path, info


def fixed_view(g, yaw_deg, size, fov, dist_scale):
    center, radius = scene_frame(g)
    t = math.radians(yaw_deg)
    dist = dist_scale * radius
    eye = center + torch.tensor([math.sin(t), 0.0, -math.cos(t)], device=center.device) * dist
    eye = eye.clone()
    eye[1] = center[1]
    return look_at_w2c(eye, center, device=center.device), intrinsics(size, size, fov), radius


def colmap_square_view(name, size, colmap_dir=ROOT / "data" / "tnt-train" / "colmap"):
    """Pose of a training photo from COLMAP, rendered as a square with the same vertical FOV."""
    import pycolmap
    rec = pycolmap.Reconstruction()
    rec.read_binary(str(colmap_dir))
    cam = next(iter(rec.cameras.values()))
    _, fy, _, _ = (float(p) for p in cam.params)
    im = next(i for i in rec.images.values() if i.name == name)
    view = torch.eye(4, device="cuda")
    view[:3, :4] = torch.tensor(np.asarray(im.cam_from_world().matrix()), dtype=torch.float32, device="cuda")
    f = fy * size / cam.height
    k = torch.tensor([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]], dtype=torch.float32, device="cuda")
    return view, k


def stereo_timing(g, view, k, width, height, ipd, iters, warmup):
    left = view.clone()
    right = view.clone()
    left[0, 3] += ipd / 2.0   # camera-frame translation: left eye sits half a separation to the left
    right[0, 3] -= ipd / 2.0
    for _ in range(warmup):
        render(g, left, k, width, height)
        render(g, right, k, width, height)
    torch.cuda.synchronize()
    times = []
    nan_seen = False
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        a = render(g, left, k, width, height)
        b = render(g, right, k, width, height)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        nan_seen = nan_seen or bool(torch.isnan(a).any() or torch.isnan(b).any())
    t = np.asarray(times)
    return {
        "frames": iters,
        "width": width,
        "height": height,
        "ms_per_stereo_frame": {"mean": float(t.mean()), "p50": float(np.percentile(t, 50)),
                                "p95": float(np.percentile(t, 95)), "max": float(t.max())},
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "nan_seen": nan_seen,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="S5 offline gsplat render of the train scene.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--explore", action="store_true")
    mode.add_argument("--render", action="store_true")
    mode.add_argument("--timing", action="store_true")
    p.add_argument("--scene", default=str(SCENE))
    p.add_argument("--label", default="full", help="suffix for output file names, e.g. lod3")
    p.add_argument("--size", type=int, default=2644, help="square render size in pixels")
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--yaw", type=float, default=0.0, help="camera yaw around the scene, degrees")
    p.add_argument("--dist-scale", type=float, default=1.6, help="camera distance in scene radii")
    p.add_argument("--ipd-fraction", type=float, default=0.01, help="eye separation in scene radii")
    p.add_argument("--colmap-view", default=None,
                   help="training photo name, e.g. 00002.jpg; uses its COLMAP pose instead of the orbit camera")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args(argv)

    DATA.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    t0 = time.perf_counter()
    g = load_gaussians(args.scene, device=device)
    load_s = time.perf_counter() - t0
    print(f"loaded {len(g)} Gaussians in {load_s:.2f} s")

    if args.explore:
        path, info = explore(g, DATA, dist_scale=args.dist_scale,
                             name=f"explore_d{args.dist_scale:g}.png")
        print(f"wrote {path}")
        print(json.dumps(info, indent=2))
        return

    _, radius = scene_frame(g)
    if args.colmap_view:
        view, k = colmap_square_view(args.colmap_view, args.size)
        tag = f"colmap{Path(args.colmap_view).stem}"
    else:
        view, k, _ = fixed_view(g, args.yaw, args.size, args.fov, args.dist_scale)
        tag = f"yaw{int(args.yaw)}"
    ipd = args.ipd_fraction * radius
    if args.render:
        img = render(g, view, k, args.size, args.size)
        assert not torch.isnan(img).any(), "NaN in render output"
        path = DATA / f"render_{tag}_{args.size}_{args.label}.png"
        Image.fromarray(to_uint8(img)).save(path)
        print(f"wrote {path}")
        print(f"mean RGB {img.mean(dim=(0, 1)).tolist()}, NaN none")
        return

    torch.cuda.reset_peak_memory_stats()
    result = stereo_timing(g, view, k, args.size, args.size, ipd, args.iters, args.warmup)
    result.update({"scene": str(args.scene), "label": args.label, "gaussians": len(g),
                   "sh_degree": g.sh_degree, "radius": radius, "ipd": ipd, "load_s": load_s,
                   "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__})
    out = DATA / f"timing_{args.size}_{args.label}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
