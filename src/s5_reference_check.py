"""S5 reference check: render the train scene from COLMAP camera poses and compare with the photos.

Reference: the dataset's own photographs for these views (Tanks and Temples train,
nvs-bench/tanksandtemples, pinned revision). The COLMAP poses are in the same
world frame the scene was trained in, and they use the same world-to-camera
convention as our renderer.

Views 00002, 00004, 00006 are training views. 00009 is a held-out view in the
3DGS split (every 8th camera), so it measures generalization, not fit.

Control: the same render flipped left-right. A wrong camera convention would
show up as a score drop, so the control checks that the metric can tell.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pycolmap
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s5_offline_render import DATA, render  # noqa: E402
from splat_io import load_gaussians  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "data" / "tnt-train" / "point_cloud.ply"
COLMAP = ROOT / "data" / "tnt-train" / "colmap"
GT_DIR = ROOT / "data" / "tnt-train" / "gt"
VIEWS = {"00002.jpg": "train", "00004.jpg": "train", "00006.jpg": "train", "00009.jpg": "test"}
PSNR_THRESHOLD_DB = 20.0  # my chosen bar for "the render is right"; see brief


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="S5 reference check against training photos.")
    p.add_argument("--scene", default=str(SCENE))
    p.add_argument("--label", default="full", help="suffix for output file names, e.g. lod3")
    args = p.parse_args(argv)

    g = load_gaussians(args.scene, device="cuda")
    rec = pycolmap.Reconstruction()
    rec.read_binary(str(COLMAP))
    cam = next(iter(rec.cameras.values()))
    fx, fy, cx, cy = (float(p) for p in cam.params)
    w0, h0 = cam.width, cam.height

    results = []
    sheet_rows = []
    for name, split in VIEWS.items():
        im = next(i for i in rec.images.values() if i.name == name)
        t = torch.tensor(np.asarray(im.cam_from_world().matrix()), dtype=torch.float32, device="cuda")
        view = torch.eye(4, device="cuda")
        view[:3, :4] = t

        gt_img = Image.open(GT_DIR / name).convert("RGB")
        w, h = gt_img.size
        sx, sy = w / w0, h / h0
        k = torch.tensor([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]],
                         dtype=torch.float32, device="cuda")

        pred = render(g, view, k, w, h).clamp(0, 1).cpu().numpy()
        gt = np.asarray(gt_img, dtype=np.float32) / 255.0
        score = psnr(pred, gt)
        control = psnr(pred[:, ::-1], gt)  # horizontally mirrored render
        assert not np.isnan(pred).any()
        results.append({"view": name, "split": split, "psnr_db": score,
                        "mirror_control_db": control, "size": [w, h]})
        print(f"{name} ({split}): PSNR {score:.2f} dB | mirrored control {control:.2f} dB")

        row = np.concatenate([gt, pred], axis=1)
        sheet_rows.append((row * 255).astype(np.uint8))

    train = [r["psnr_db"] for r in results if r["split"] == "train"]
    test = [r["psnr_db"] for r in results if r["split"] == "test"]
    summary = {
        "scene": str(args.scene),
        "label": args.label,
        "gaussians": len(g),
        "threshold_db": PSNR_THRESHOLD_DB,
        "train_mean_db": float(np.mean(train)),
        "test_mean_db": float(np.mean(test)),
        "pass": bool(np.mean(train) >= PSNR_THRESHOLD_DB and np.mean(test) >= PSNR_THRESHOLD_DB),
        "views": results,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"reference_check_{args.label}.json").write_text(json.dumps(summary, indent=2))
    Image.fromarray(np.concatenate(sheet_rows[:2], axis=0)).save(DATA / f"reference_sheet_{args.label}.png")
    print(json.dumps({k: v for k, v in summary.items() if k != "views"}, indent=2))


if __name__ == "__main__":
    main()
