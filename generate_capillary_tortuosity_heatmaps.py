#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Capillary tortuosity hotspots (no dot maps)
USING GRAPH-BASED EDGE DIVISION (build_skeleton_graph).

For each capillary segmentation ID in [10001, 10300]:
  - Load capillary BMP from GT_Capillary
  - Thin to skeleton
  - Build skeleton graph edges
  - Compute tortuosity for each edge
  - Keep only high-tortuosity edges (top TAU_EXCESS_PCTL% by τ_excess)
  - Accumulate edge weights into an impulse map
  - Gaussian blur at several sigma fractions to get hotspot maps
  - Save grayscale + color hotspot PNGs

Inputs:


Outputs (examples):


import os
import numpy as np
import matplotlib.cm as cm

from skimage import io, morphology
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import gaussian_filter

# ----------------- paths / parameters -----------------

CAP_ROOT = 
EXT = ".bmp"

ID_MIN = 10001
ID_MAX = 10300

BASE_OUT_ROOT = 
HOTSPOT_DIR   = os.path.join(BASE_OUT_ROOT, "capillary", "hotspots")
os.makedirs(HOTSPOT_DIR, exist_ok=True)

USE_OTSU     = True
MIN_OBJ_SIZE = 5

# hotspot theory params (match artery/vein/large-vessel)
SIGMA_FRACS     = [0.020, 0.040, 0.060, 0.080]
LEN_POWER       = 1.0
TAU_POWER       = 1.0
TAU_EXCESS_PCTL = 85.0
ROBUST_PCTL     = 99.0

# ------------------------------------------------------
# helper: skeleton + preprocessing
# ------------------------------------------------------
def load_capillary_skeleton(path):
    """Load BMP, binarize, clean small specks, thin (capillary-style)."""
    img = io.imread(path, as_gray=True)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")

    if USE_OTSU:
        thr = threshold_otsu(img)
        binary = img > thr
    else:
        binary = img > 0.5

    binary = remove_small_objects(binary, min_size=MIN_OBJ_SIZE)
    skel = morphology.thin(binary).astype(np.uint8)
    return binary, skel

# ------------------------------------------------------
# helper: graph construction from skeleton pixels
# ------------------------------------------------------
def build_skeleton_graph(skel):
    """
    Build a graph of the skeleton.

    Returns:
        edges      : list of np.ndarray (Ni x 2) pixel coords for each edge
                     (ordered path between two nodes).
        node_to_edges, deg_map: returned for completeness (not used downstream).
    """
    H, W = skel.shape
    skeleton_pixels = list(zip(*np.nonzero(skel)))
    skeleton_pixels = [(int(r), int(c)) for r, c in skeleton_pixels]

    # 8-connected neighbors for every skeleton pixel
    nbrs = {}
    for r, c in skeleton_pixels:
        neigh = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and skel[rr, cc]:
                    neigh.append((rr, cc))
        nbrs[(r, c)] = neigh

    # degrees and nodes
    deg_map = {p: len(nn) for p, nn in nbrs.items()}
    nodes = [p for p, d in deg_map.items() if d != 2]  # endpoints + branchpoints

    edges = []
    node_to_edges = {}
    visited_pairs = set()  # undirected visited (min(p,q), max(p,q))

    def edge_key(p, q):
        return (min(p, q), max(p, q))

    for node in nodes:
        for q in nbrs[node]:
            k = edge_key(node, q)
            if k in visited_pairs:
                continue

            path = [node]
            prev = node
            cur = q

            while True:
                visited_pairs.add(edge_key(prev, cur))
                path.append(cur)

                # stop if we reached another node (and not the starting one again)
                if cur in nodes and cur != node:
                    break

                # otherwise walk forward
                neigh = [x for x in nbrs[cur] if x != prev]
                if not neigh:
                    # dead-end; cur itself is an endpoint node (degree 1)
                    break
                if len(neigh) > 1:
                    # we've hit a new branchpoint (should be in nodes)
                    break

                nxt = neigh[0]
                prev, cur = cur, nxt

            coords = np.array(path, dtype=float)
            if coords.shape[0] >= 1:   # keep every path with at least 1 pixel
                edge_idx = len(edges)
                edges.append(coords)
                start = tuple(path[0])
                end   = tuple(path[-1])
                node_to_edges.setdefault(start, []).append(edge_idx)
                node_to_edges.setdefault(end,   []).append(edge_idx)

    return edges, node_to_edges, deg_map

# ------------------------------------------------------
# tortuosity helpers
# ------------------------------------------------------
def path_length(coords):
    if coords.shape[0] < 2:
        return 0.0
    diffs = coords[1:] - coords[:-1]
    return float(np.linalg.norm(diffs, axis=1).sum())

def edge_tortuosity(coords):
    if coords.shape[0] < 2:
        return np.nan
    L = path_length(coords)
    chord = float(np.linalg.norm(coords[-1] - coords[0]))
    if chord <= 0:
        return np.nan
    return L / chord

# ------------------------------------------------------
# hotspot density from edges (same "theory" as large-vessel code)
# ------------------------------------------------------
def hotspot_density_from_edges(shape, edges):
    H, W = shape
    impulses = np.zeros((H, W), float)

    tau_excess_list = []
    edge_infos = []
    for coords in edges:
        tau = edge_tortuosity(coords)
        if not np.isfinite(tau) or tau <= 1.0:
            continue
        tau_ex = tau - 1.0
        L = max(coords.shape[0], 1)
        edge_infos.append((coords, L, tau_ex))
        tau_excess_list.append(tau_ex)

    if not edge_infos:
        return impulses

    tau_excess_vals = np.array(tau_excess_list, float)
    tau_thresh = np.percentile(tau_excess_vals, TAU_EXCESS_PCTL)

    for coords, L, tau_ex in edge_infos:
        if tau_ex < tau_thresh:
            continue
        edge_weight = (L ** LEN_POWER) * (tau_ex ** TAU_POWER)
        per_pixel   = edge_weight / L
        for (r, c) in coords.astype(int):
            if 0 <= r < H and 0 <= c < W:
                impulses[r, c] += per_pixel

    return impulses

def make_blurred_maps(impulses, sigma_fracs):
    H, W = impulses.shape
    M = max(H, W)
    maps = []
    for frac in sigma_fracs:
        sigma_px = max(1, int(round(frac * M)))
        maps.append(gaussian_filter(impulses, sigma=sigma_px))
    return maps

def robust_normalize(density, pctl=99.0):
    scale = np.percentile(density, pctl)
    if scale <= 1e-6:
        return np.zeros_like(density, dtype=np.float32)
    return np.clip(density / scale, 0, 1).astype(np.float32)

# ------------------------------------------------------
# save helpers
# ------------------------------------------------------
def save_grayscale(d_norm, out_path):
    u8 = (d_norm * 255.0 + 0.5).astype(np.uint8)
    io.imsave(out_path, u8)

def save_colormap_no_key(d_norm, out_path, cmap_name="jet"):
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(d_norm)
    rgb  = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    io.imsave(out_path, rgb)

# ------------------------------------------------------
# main: process all IDs
# ------------------------------------------------------
def main():
    print(f"Computing capillary hotspots for IDs {ID_MIN}–{ID_MAX}")
    for ID in range(ID_MIN, ID_MAX + 1):
        in_path = os.path.join(CAP_ROOT, f"{ID}{EXT}")
        if not os.path.exists(in_path):
            print(f"  [ID {ID}] file not found, skipping: {in_path}")
            continue

        print(f"\n=== Processing capillary ID {ID} ===")
        binary, skel = load_capillary_skeleton(in_path)
        H, W = skel.shape

        edges, node_to_edges, deg_map = build_skeleton_graph(skel)
        print(f"  Number of graph edges: {len(edges)}")

        impulses = hotspot_density_from_edges((H, W), edges)
        blurred  = make_blurred_maps(impulses, SIGMA_FRACS)

        tag = f"capillary_{ID}"
        for frac, d in zip(SIGMA_FRACS, blurred):
            d_norm = robust_normalize(d, pctl=ROBUST_PCTL)

            g_path = os.path.join(
                HOTSPOT_DIR, f"{tag}_hotspot_sigma{frac:.3f}_GRAY.png"
            )
            c_path = os.path.join(
                HOTSPOT_DIR, f"{tag}_hotspot_sigma{frac:.3f}_COLOR.png"
            )

            save_grayscale(d_norm, g_path)
            save_colormap_no_key(d_norm, c_path, cmap_name="jet")

            print("  Saved hotspot:", g_path)
            print("  Saved hotspot:", c_path)

    print("\nDone computing capillary hotspots for all IDs.\n")

if __name__ == "__main__":
    main()
