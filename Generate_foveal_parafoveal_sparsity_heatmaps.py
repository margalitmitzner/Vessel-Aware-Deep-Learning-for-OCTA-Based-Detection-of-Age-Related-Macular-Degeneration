#Generate heatmaps incorporating sparsity information for fovea and parafovea separately

import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# -------------------- PATHS --------------------
CAP_ROOT     = 
LV_ROOT      = 
FAZ_ROOT     = 

ARTERY_ROOT  = 
VEIN_ROOT    = 

OUT_ROOT = 
os.makedirs(OUT_ROOT, exist_ok=True)

# -------------------- PARAMETERS --------------------
SIGMA_FRACS = [0.02, 0.04, 0.06, 0.08]

R_CAP   = 10
R_LARGE = 20

FOV_MM = 6.0

FOVEAL_DIAM_MM = 1.0
PARA_DIAM_MM   = 3.0
FOVEAL_RADIUS_MM = FOVEAL_DIAM_MM / 2.0   # 0.5
PARA_RADIUS_MM   = PARA_DIAM_MM   / 2.0   # 1.5


# -------- Utility functions --------
def load_binary(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (mask > 0).astype(np.float32)


def disk_kernel(radius):
    r = int(radius)
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = (x * x + y * y) <= (r * r)
    return mask.astype(np.float32)


def local_density(mask, radius):
    kern = disk_kernel(radius)
    dens = cv2.filter2D(mask, -1, kern)
    if dens.max() > 0:
        dens = dens / dens.max()
    return dens.astype(np.float32)


def low_density_impulses(mask, radius, sparsity_pct=85.0, power=1.0):
    """
    EXACT SAME LOGIC as your original script:
      - density computed from full image
      - sparsity = 1 - density
      - percentile threshold computed across ALL vessel pixels (mask>0)
      - impulses placed only on sparsest vessel pixels globally
    """
    dens = local_density(mask, radius=radius)   # [0,1]
    sparsity = 1.0 - dens                       # [0,1]

    vessel_vals = sparsity[mask > 0]
    impulses = np.zeros_like(mask, dtype=np.float32)
    if vessel_vals.size == 0:
        return impulses

    thresh = np.percentile(vessel_vals, sparsity_pct)
    sel = (mask > 0) & (sparsity >= thresh)
    impulses[sel] = sparsity[sel] ** power
    return impulses


def robust_norm_in_roi(a, roi, p=99.0):
    """
    Robust normalization computed ONLY inside ROI.
    Prevents the 'all blue' issue when ROI is small and most pixels are zero.
    """
    vals = a[roi > 0]
    vals = vals[np.isfinite(vals)]
    # Optional: you can also drop exact zeros if you want:
    # vals = vals[vals > 0]

    if vals.size == 0:
        return np.zeros_like(a, dtype=np.float32)

    vmax = np.percentile(vals, p)
    if vmax < 1e-6:
        return np.zeros_like(a, dtype=np.float32)

    out = np.clip(a / vmax, 0, 1).astype(np.float32)
    return out


def save_gray(path, arr01):
    arr = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, arr)


def save_color(path, arr01):
    arr = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    cv2.imwrite(path, color)


# -------- FAZ-derived ROIs --------
def faz_center_from_mask(faz_mask):
    ys, xs = np.nonzero(faz_mask)
    if xs.size == 0:
        H, W = faz_mask.shape
        return float(H) / 2.0, float(W) / 2.0
    return ys.mean(), xs.mean()


def circle_roi(shape, cy, cx, radius_px):
    H, W = shape
    yy, xx = np.ogrid[:H, :W]
    roi = (yy - cy) ** 2 + (xx - cx) ** 2 <= float(radius_px) ** 2
    return roi.astype(np.float32)


def faz_foveal_and_parafoveal_ring_rois(faz_mask,
                                        fov_mm=6.0,
                                        foveal_radius_mm=0.5,
                                        para_radius_mm=1.5):
    H, W = faz_mask.shape
    cy, cx = faz_center_from_mask(faz_mask)

    px_per_mm = float(W) / float(fov_mm)  # assumes square scan
    r_fov_px  = int(round(foveal_radius_mm * px_per_mm))
    r_para_px = int(round(para_radius_mm   * px_per_mm))

    foveal_roi = circle_roi((H, W), cy, cx, r_fov_px)
    outer_roi  = circle_roi((H, W), cy, cx, r_para_px)
    ring_roi   = np.clip(outer_roi - foveal_roi, 0, 1)

    return foveal_roi.astype(np.float32), ring_roi.astype(np.float32)


# -------------- PROCESS ONE MASK --------------
def process_one_fovea_and_ring(vessel_mask, faz_mask, radius, prefix, ID):
    """
    Requirement:
      - sparsity is computed from the full image segmentations (global)
      - heatmaps are displayed separately for foveal 1mm zone and parafoveal 3mm ring

    Implementation:
      1) impulses_global from full mask (original logic)
      2) blur_global from full impulses (not ROI-masked)
      3) for each ROI: region_map = blur_global * ROI
      4) normalize using only ROI pixels (fixes all-blue outputs)
    """
    impulses_global = low_density_impulses(vessel_mask, radius)

    foveal_roi, ring_roi = faz_foveal_and_parafoveal_ring_rois(
        faz_mask,
        fov_mm=FOV_MM,
        foveal_radius_mm=FOVEAL_RADIUS_MM,
        para_radius_mm=PARA_RADIUS_MM
    )

    regions = [
        ("FOV1mm", foveal_roi),
        ("PARA3mmRING", ring_roi),
    ]

    H, W = impulses_global.shape
    M = max(H, W)

    # Blur on the WHOLE image, once per sigma
    for frac in SIGMA_FRACS:
        sigma = max(1, int(round(frac * M)))
        blur_global = gaussian_filter(impulses_global, sigma=sigma)

        for region_name, roi in regions:
            region_map = blur_global * roi

            # ROI-aware normalization (critical fix)
            region_norm = robust_norm_in_roi(region_map, roi, p=99.0)

            base = f"{prefix}_{ID}_{region_name}_sigma{frac:.3f}"
            gray_path  = os.path.join(OUT_ROOT, base + ".png")
            color_path = os.path.join(OUT_ROOT, base + "_COLOR.png")

            save_gray(gray_path, region_norm)
            save_color(color_path, region_norm)

            print("Saved:", gray_path)


# -------------- MAIN LOOP OVER IDS --------------
if __name__ == "__main__":
    for pid in range(10001, 10301):   # inclusive of 10300
        ID = str(pid)
        print(f"\n=== Processing ID {ID} (FAZ-centered 1mm fovea + 3mm parafoveal ring) ===")

        cap_path    = os.path.join(CAP_ROOT,    f"{ID}.bmp")
        lv_path     = os.path.join(LV_ROOT,     f"{ID}.png")
        faz_path    = os.path.join(FAZ_ROOT,    f"{ID}_FAZ.bmp")
        artery_path = os.path.join(ARTERY_ROOT, f"{ID}.bmp")
        vein_path   = os.path.join(VEIN_ROOT,   f"{ID}.bmp")

        if not os.path.exists(faz_path):
            print("Missing FAZ segmentation:", faz_path)
            continue

        try:
            faz_mask = load_binary(faz_path)
        except Exception as e:
            print(f"Error loading FAZ mask {ID}: {e}")
            continue

        # Large vessels
        if os.path.exists(lv_path):
            try:
                LV = load_binary(lv_path)
                process_one_fovea_and_ring(LV, faz_mask, radius=R_LARGE, prefix="largevessel", ID=ID)
            except Exception as e:
                print(f"Error processing large vessels {ID}: {e}")
        else:
            print("Missing large vessel file:", lv_path)

        # Capillaries
        if os.path.exists(cap_path):
            try:
                CAP = load_binary(cap_path)
                process_one_fovea_and_ring(CAP, faz_mask, radius=R_CAP, prefix="capillary", ID=ID)
            except Exception as e:
                print(f"Error processing capillaries {ID}: {e}")
        else:
            print("Missing capillary file:", cap_path)

        # Arteries
        if os.path.exists(artery_path):
            try:
                ARTERY = load_binary(artery_path)
                process_one_fovea_and_ring(ARTERY, faz_mask, radius=R_LARGE, prefix="artery", ID=ID)
            except Exception as e:
                print(f"Error processing arteries {ID}: {e}")
        else:
            print("Missing artery file:", artery_path)

        # Veins
        if os.path.exists(vein_path):
            try:
                VEIN = load_binary(vein_path)
                process_one_fovea_and_ring(VEIN, faz_mask, radius=R_LARGE, prefix="vein", ID=ID)
            except Exception as e:
                print(f"Error processing veins {ID}: {e}")
        else:
            print("Missing vein file:", vein_path)
