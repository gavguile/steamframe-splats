"""gsplat render for VR: splats are sorted by distance from the eye, not view depth.

gsplat's rasterization() sorts splats per tile by z, their depth along the view
direction. Turning the head changes z for everything off-centre, so overlapping
splats swap order and visibly pop. The straight-line distance from the eye does
not change under a pure head rotation, so sorting by it removes the popping that
rotation causes. Moving through the scene can still reorder splats, as in any 3DGS
renderer.

This is the same pipeline as gsplat.rasterization (classic mode, one camera,
unpacked), with the sort key swapped. Splats whose centre is closer than
near_distance to the eye are dropped before projection. The cut also uses
distance, so turning the head does not make splats appear and disappear at it.
"""

import math

import torch
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    spherical_harmonics,
)

TILE = 16


@torch.no_grad()
def render_xr(g, viewmat, k, width, height, near_distance=0.0, sort="distance"):
    """Render one view. Returns (H, W, 3) float on the GPU.

    near_distance: scene units; splats closer to the eye are not drawn.
    sort: 'distance' (from the eye) or 'depth' (gsplat's default, along the view).
    """
    campos = torch.linalg.inv(viewmat)[:3, 3]
    if near_distance > 0:
        keep = torch.linalg.norm(g.means - campos, dim=-1) >= near_distance
        g = type(g)(g.means[keep], g.quats[keep], g.scales[keep], g.opacities[keep], g.colors[keep], g.sh_degree)
    if g.means.shape[0] == 0:
        return torch.zeros((height, width, 3), device=viewmat.device)  # gsplat aborts on zero splats
    radii, means2d, depths, conics, _ = fully_fused_projection(
        g.means, None, g.quats, g.scales, viewmat[None], k[None], width, height,
        eps2d=0.3, packed=False, near_plane=0.01, far_plane=1e10,
        radius_clip=0.0, opacities=g.opacities)
    dirs = g.means - campos  # [N, 3]
    if sort == "distance":
        key = torch.linalg.norm(dirs, dim=-1)[None]  # [1, N]
    elif sort == "depth":
        key = depths
    else:
        raise ValueError(sort)
    masks = (radii > 0).all(dim=-1)  # [1, N]
    colors = spherical_harmonics(g.sh_degree, dirs[None], g.colors[None], masks=masks)
    colors = torch.clamp_min(colors + 0.5, 0.0)  # [1, N, 3]

    tw, th = math.ceil(width / TILE), math.ceil(height / TILE)
    _, isect_ids, flatten_ids = isect_tiles(means2d, radii, key, TILE, tw, th, packed=False, n_images=1)
    offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(1, th, tw)
    img, _ = rasterize_to_pixels(means2d, conics, colors, g.opacities[None], width, height, TILE,
                                 offsets, flatten_ids, packed=False)
    return img[0]
