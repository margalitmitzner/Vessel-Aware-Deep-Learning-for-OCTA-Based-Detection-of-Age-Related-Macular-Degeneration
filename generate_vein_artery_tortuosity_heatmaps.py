#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tortuosity hotspots + colored dot maps for artery/vein/large-vessel segmentations (IDs 10001–10300).

For each segmentation, this script computes:
  - Four Gaussian-blurred tortuosity hotspot heatmaps (grayscale PNG, exact HxW)
  - Four color hotspot maps (no colorbar, jet colormap, exact HxW)
  - One colored dot map overlay on the binary vessel mask (tortuosity-coded dots, jet colormap, no colorbar)

Inputs:
  Artery BMP : /home/mgmitzner/Tortuosity_training/GT_Artery/{ID}.bmp
  Vein   BMP : /home/mgmitzner/Tortuosity_training/GT_Vein/{ID}.bmp
  Large vsl  : /home/mgmitzner/Tortuosity_training/OCTA500_large_vessel_segmentations_6_mm/{ID}.png

Outputs:
  /home/mgmitzner/Tortuosity_training/11_05_heatmaps_for_DL/artery/hotspots/artery_{ID}_hotspot_sigma0.020_GRAY.png
  /home/mgmitzner/Tortuosity_training/11_05_heatmaps_for_DL/artery/hotspots/artery_{ID}_hotspot_sigma0.020_COLOR.png
  ...
  /home/mgmitzner/Tortuosity_training/11_05_heatmaps_for_DL/artery/dots/artery_{ID}_dots.png

  (Similarly for vein_{ID} in /vein/... and largevessel_{ID} in /large_vessel/...)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.ndimage import gaussian_filter
from skimage import io, morphology, measure
from skimage.morphology import remove_small_objects

# -------------------- I/O ROOTS --------------------
ART_ROOT = "/home/mgmitzner/Tortuosity_training/GT_Artery"
VEI_ROOT = "/home/mgmitzner/Tortuosity_training/GT_Vein"
LV_ROOT  = "/home/mgmitzner/Tortuosity_training/OCTA500_large_vessel_segmentations_6_mm"

BASE_OUT_ROOT = "/home/mgmitzner/Tortuosity_training/11_05_heatmaps_for_DL"

# -------------------- Params -----------------
# Four blur levels as a fraction of max(H,W).
SIGMA_FRACS = [0.020, 0.040, 0.060, 0.080]

# Length/tortuosity weighting for hotspot maps
LEN_POWER   = 1.0    # upweight longer edges linearly (set >1 for super-linear)
TAU_POWER   = 1.0    # upweight more tortuous edges (uses tau_excess ** TAU_POWER)
MIN_OBJ_SZ  = 20     # remove tiny specs before skeletonization
MIN_EDGE_PX = 8      # skip very short edge fragments
ROBUST_PCTL = 99.0   # robust normalization pctl for grayscale output

# How aggressively to focus on the most tortuous edges (shared for all vessel types)
TAU_EXCESS_PCTL = 85.0   # keep only edges above this percentile of tau_excess

# ----------------- Geometry helpers -----------------
def _neighbors8(y, x):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yield y + dy, x + dx

def _find_bifurcations(skel):
    H, W = skel.shape
    bif = []
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if skel[y, x]:
                deg = int(np.sum(skel[y - 1:y + 2, x - 1:x + 2]) - 1)
                if deg > 2:
                    bif.append((y, x))
    return bif

def _split_at_bifurcations(skel, bif):
    out = skel.copy()
    for y, x in bif:
        out[y, x] = 0
    return out

def _extract_edges(skel):
    """Return list of coordinate arrays for bifurcation-free edges."""
    bif = _find_bifurcations(skel)
    split = _split_at_bifurcations(skel, bif)
    labeled = measure.label(split, connectivity=2)
    return [prop.coords for prop in measure.regionprops(labeled)]

def _order_edge(coords):
    """Order edge pixels along a path using greedy neighbor following."""
    pts = [tuple(p) for p in np.asarray(coords, dtype=int)]
    S = set(pts)
    nbrs = {}
    for y, x in pts:
        nn = []
        for ny, nx in _neighbors8(y, x):
            if (ny, nx) in S:
                nn.append((ny, nx))
        nbrs[(y, x)] = nn

    endpoints = [p for p, nn in nbrs.items() if len(nn) == 1]
    start = endpoints[0] if endpoints else pts[0]

    ordered, visited = [], set()
    cur, prev = start, None
    while cur is not None:
        ordered.append(cur)
        visited.add(cur)
        nxt = None
        for q in nbrs[cur]:
            if q != prev and q not in visited:
                nxt = q
                break
        prev, cur = cur, nxt

    # catch loops / leftovers
    if len(ordered) < len(pts):
        for p in pts:
            if p not in visited:
                ordered.append(p)
    return np.array(ordered, dtype=float)

def _path_length(poly):
    if len(poly) < 2:
        return 0.0
    diffs = poly[1:] - poly[:-1]
    return float(np.linalg.norm(diffs, axis=1).sum())

def _tortuosity(poly):
    if len(poly) < 2:
        return np.nan
    L = _path_length(poly)
    chord = float(np.linalg.norm(poly[-1] - poly[0]))
    return L / chord if chord > 0 else np.nan

# ---------- Sampling helper for colored dots ----------
def sample_along_edge(poly, fractions):
    """
    Return (row, col) integer positions at given arc-length fractions along the polyline.
    poly: (N,2) array of [y,x]
    fractions: list of floats in [0,1]
    """
    if len(poly) < 2:
        return []
    diffs = poly[1:] - poly[:-1]
    seglen = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = cum[-1]
    if total <= 0:
        return []

    targets = [f * total for f in fractions]
    pts = []
    j = 1
    for t in targets:
        while j < len(cum) and cum[j] < t:
            j += 1
        j = min(j, len(cum) - 1)
        t0, t1 = cum[j - 1], cum[j]
        alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        p = poly[j - 1] * (1 - alpha) + poly[j] * alpha
        pts.append((int(round(p[0])), int(round(p[1]))))
    return pts

# --------------- Hotspot construction ----------------
def hotspot_density_from_seg(seg):
    """
    seg: binary array (H,W) with vessels == True

    Returns:
        impulses: (H,W) array with per-pixel weights aggregated across
                  ONLY the more tortuous edges (tau_excess >= TAU_EXCESS_PCTL).
    """
    seg = remove_small_objects(seg, MIN_OBJ_SZ)
    skel = morphology.skeletonize(seg).astype(np.uint8)
    H, W = skel.shape
    impulses = np.zeros((H, W), dtype=float)

    # -------- First pass: collect valid edges and their tau_excess --------
    edge_infos = []  # list of (poly, length_px, tau_excess)
    for edge in _extract_edges(skel):
        if len(edge) < MIN_EDGE_PX:
            continue

        poly = _order_edge(edge)
        tau = _tortuosity(poly)
        if not np.isfinite(tau) or tau <= 1.0:
            # straight or degenerate edge
            continue

        length_px  = max(len(poly), 1)
        tau_excess = tau - 1.0
        edge_infos.append((poly, length_px, tau_excess))

    if not edge_infos:
        # no tortuous edges found → return an all-zero map
        return impulses

    # -------- Compute tau_excess threshold (e.g., 85th percentile) --------
    tau_excess_vals = np.array([ei[2] for ei in edge_infos], dtype=float)
    tau_thresh = np.percentile(tau_excess_vals, TAU_EXCESS_PCTL)

    # -------- Second pass: accumulate only high-tortuosity edges --------
    for poly, length_px, tau_excess in edge_infos:
        if tau_excess < tau_thresh:
            # discard lower-tortuosity edges
            continue

        # edge weight (longer + more tortuous edges contribute more)
        edge_weight = (length_px ** LEN_POWER) * (tau_excess ** TAU_POWER)

        # distribute uniformly along the edge so total contribution = edge_weight
        per_pixel = edge_weight / length_px
        for (y, x) in poly.astype(int):
            if 0 <= y < H and 0 <= x < W:
                impulses[y, x] += per_pixel

    return impulses

def make_blurred_maps(impulses, sigma_fracs):
    H, W = impulses.shape
    M = max(H, W)
    maps = []
    for frac in sigma_fracs:
        sigma_px = max(1, int(round(frac * M)))
        d = gaussian_filter(impulses, sigma=sigma_px)
        maps.append(d)
    return maps

def robust_normalize(density, pctl=99.0):
    scale = np.percentile(density, pctl)
    if scale <= 1e-6:
        return np.zeros_like(density, dtype=np.float32)
    d = np.clip(density / scale, 0, 1).astype(np.float32)
    return d

def save_grayscale(d_norm, out_path):
    """Save exact HxW grayscale heatmap (uint8)."""
    u8 = (d_norm * 255.0 + 0.5).astype(np.uint8)
    io.imsave(out_path, u8)

def save_colormap_no_key(d_norm, out_path, cmap="jet"):
    """
    Save a color hotspot map with no colorbar/key (jet colormap).
    Output image has the same HxW as d_norm.
    """
    cmap_obj = cm.get_cmap(cmap)
    rgba = cmap_obj(d_norm)                 # (H,W,4), floats 0..1
    rgb  = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    io.imsave(out_path, rgb)

# --------------- Colored dot map construction ---------------
def colored_dot_map(seg_path: str, out_png: str,
                    min_obj_size: int = 20,
                    cmap_name: str = "jet"):
    """
    Build a colored dots map for one binary segmentation.
    - 1/2/3 dots per edge based on edge-length percentiles.
    - Dot color encodes edge tortuosity using a jet-like colormap.
    Output is RGB PNG with same HxW as the input segmentation, no colorbar.
    """
    img = io.imread(seg_path, as_gray=True)
    if img is None:
        raise FileNotFoundError(seg_path)

    # binarize and skeletonize (vessels are white, background black)
    if img.dtype != bool:
        binary = img > 0
    else:
        binary = img
    binary = remove_small_objects(binary, min_size=min_obj_size)
    skel = morphology.skeletonize(binary).astype(np.uint8)

    edges = _extract_edges(skel)

    ordered_list, lengths, torts = [], [], []
    for edge in edges:
        if len(edge) < MIN_EDGE_PX:
            continue
        poly = _order_edge(edge)
        L = _path_length(poly)
        T = _tortuosity(poly)
        # keep edges with finite tortuosity and positive length
        if np.isfinite(T) and L > 0:
            ordered_list.append(poly)
            lengths.append(L)
            torts.append(T)

    H, W = binary.shape

    # --- set up figure to be exactly HxW pixels ---
    dpi = 100.0
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])   # fill entire figure

    # Fade vessels while keeping background black
    faded = binary.astype(float) * 0.25   # 0 -> black, 1 -> 0.25 (dim gray)
    ax.imshow(faded, cmap="gray", vmin=0, vmax=1)
    ax.set_facecolor("black")
    ax.set_axis_off()

    if not ordered_list:
        print(f"No valid edges in {seg_path}. Writing background-only dot map.")
        fig.savefig(out_png, dpi=dpi, bbox_inches=None, pad_inches=0)
        plt.close(fig)
        return

    lengths = np.array(lengths, float)
    torts   = np.array(torts, float)

    # Percentile thresholds for edge-length bins
    p33 = np.percentile(lengths, 33.0)
    p66 = np.percentile(lengths, 66.0)

    # Normalize tortuosity for colormap (robust: 5–95th percentile clamps)
    lo, hi = np.percentile(torts, [5, 95])
    def norm_tort(t):
        return float(np.clip((t - lo) / (hi - lo + 1e-9), 0, 1))

    cmap = cm.get_cmap(cmap_name)

    # Plot dots along each edge
    for poly, L, T in zip(ordered_list, lengths, torts):
        c = cmap(norm_tort(T))  # RGBA color
        if L <= p33:
            fracs = [0.5]                       # 1 dot at center
        elif L <= p66:
            fracs = [1/3, 2/3]                  # 2 dots
        else:
            fracs = [0.25, 0.50, 0.75]          # 3 dots
        pts = sample_along_edge(poly, fracs)
        if not pts:
            continue
        ys = [p[0] for p in pts]
        xs = [p[1] for p in pts]
        ax.scatter(xs, ys, s=10, c=[c], marker='o', edgecolors='none')

    # No colorbar here
    fig.savefig(out_png, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"Wrote colored dot map: {out_png}")

# -------------------- Driver for one segmentation --------------------
def process_segmentation(seg_path: str, tag: str,
                         hotspot_dir: str,
                         dot_dir: str):
    """
    For a single segmentation:
      - builds impulses + 4 blurred hotspot maps (GRAY + COLOR, no keys)
      - builds one colored dot map (no key)
    """
    img = io.imread(seg_path, as_gray=True)
    if img is None:
        raise FileNotFoundError(seg_path)

    if img.dtype != bool:
        seg = img > 0    # vessels are white on black background
    else:
        seg = img

    impulses = hotspot_density_from_seg(seg)
    blurred = make_blurred_maps(impulses, SIGMA_FRACS)

    os.makedirs(hotspot_dir, exist_ok=True)
    os.makedirs(dot_dir, exist_ok=True)

    # Save grayscale + color hotspot maps for each sigma
    for frac, d in zip(SIGMA_FRACS, blurred):
        d_norm = robust_normalize(d, pctl=ROBUST_PCTL)

        g_path = os.path.join(hotspot_dir, f"{tag}_hotspot_sigma{frac:.3f}_GRAY.png")
        c_path = os.path.join(hotspot_dir, f"{tag}_hotspot_sigma{frac:.3f}_COLOR.png")

        save_grayscale(d_norm, g_path)
        save_colormap_no_key(d_norm, c_path, cmap="jet")

    # Colored dot map
    dot_path = os.path.join(dot_dir, f"{tag}_dots.png")
    colored_dot_map(seg_path, dot_path, min_obj_size=MIN_OBJ_SZ, cmap_name="jet")

    print(f"Done: {tag}")

# -------------------- Main --------------------
def main():
    # Process IDs 10001–10300 for each vessel type
    for ID in range(10001, 10301):
        print(f"\n=== Processing ID {ID} ===")

        # ----- Artery -----
        art_path = os.path.join(ART_ROOT, f"{ID}.bmp")
        if os.path.exists(art_path):
            art_out_root = os.path.join(BASE_OUT_ROOT, "artery")
            art_hotspot_dir = os.path.join(art_out_root, "hotspots")
            art_dot_dir     = os.path.join(art_out_root, "dots")

            tag = f"artery_{ID}"
            process_segmentation(
                art_path,
                tag=tag,
                hotspot_dir=art_hotspot_dir,
                dot_dir=art_dot_dir
            )
        else:
            print(f"Artery file not found: {art_path}")

        # ----- Vein -----
        vei_path = os.path.join(VEI_ROOT, f"{ID}.bmp")
        if os.path.exists(vei_path):
            vei_out_root = os.path.join(BASE_OUT_ROOT, "vein")
            vei_hotspot_dir = os.path.join(vei_out_root, "hotspots")
            vei_dot_dir     = os.path.join(vei_out_root, "dots")

            tag = f"vein_{ID}"
            process_segmentation(
                vei_path,
                tag=tag,
                hotspot_dir=vei_hotspot_dir,
                dot_dir=vei_dot_dir
            )
        else:
            print(f"Vein file not found: {vei_path}")

        # ----- Large vessels -----
        lv_path = os.path.join(LV_ROOT, f"{ID}.png")
        if os.path.exists(lv_path):
            lv_out_root = os.path.join(BASE_OUT_ROOT, "large_vessel")
            lv_hotspot_dir = os.path.join(lv_out_root, "hotspots")
            lv_dot_dir     = os.path.join(lv_out_root, "dots")

            tag = f"largevessel_{ID}"
            process_segmentation(
                lv_path,
                tag=tag,
                hotspot_dir=lv_hotspot_dir,
                dot_dir=lv_dot_dir
            )
        else:
            print(f"Large-vessel file not found: {lv_path}")

if __name__ == "__main__":
    # Generate hotspot heatmaps + colored dot maps for all IDs
    main()
