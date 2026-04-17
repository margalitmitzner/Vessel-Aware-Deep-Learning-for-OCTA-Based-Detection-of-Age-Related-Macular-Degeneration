#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# -------------------- PATHS --------------------
CAP_ROOT = "/home/mgmitzner/Tortuosity_training/GT_Capillary"
LV_ROOT  = "/home/mgmitzner/Tortuosity_training/OCTA500_large_vessel_segmentations_6_mm"

OUT_ROOT = "/home/mgmitzner/Tortuosity_training/11_12_25/low_vessel_density_maps/"
os.makedirs(OUT_ROOT, exist_ok=True)

# Gaussian sigmas (fractions of max image dimension)
SIGMA_FRACS = [0.02, 0.04, 0.06, 0.08]

# Neighborhood radii
R_CAP   = 10
R_LARGE = 20


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
    mask = x * x + y * y <= r * r
    return mask.astype(np.float32)


def local_density(mask, radius):
    kern = disk_kernel(radius)
    dens = cv2.filter2D(mask, -1, kern)
    if dens.max() > 0:
        dens = dens / dens.max()
    return dens.astype(np.float32)


def low_density_impulses(mask, radius, sparsity_pct=85.0, power=1.0):
    """
    Create an impulse map where *only the sparsest vessel pixels* get impulses.

    mask: (H,W) in {0,1}, ideally skeletonized
    radius: neighborhood radius for density estimation
    sparsity_pct: percentile of sparsity above which we keep impulses
    power: raise sparsity to this power to emphasize extremes
    """
    dens = local_density(mask, radius=radius)   # [0,1], high = many neighbors
    sparsity = 1.0 - dens                       # high = few neighbors

    vessel_vals = sparsity[mask > 0]
    impulses = np.zeros_like(mask, dtype=np.float32)

    if vessel_vals.size == 0:
        return impulses

    # Keep only top percentile of sparsest vessel pixels
    thresh = np.percentile(vessel_vals, sparsity_pct)

    sel = (mask > 0) & (sparsity >= thresh)
    impulses[sel] = sparsity[sel] ** power

    return impulses


def make_blurred_maps(impulses):
    H, W = impulses.shape
    M = max(H, W)
    out = []
    for frac in SIGMA_FRACS:
        sigma = max(1, int(round(frac * M)))
        blur = gaussian_filter(impulses, sigma=sigma)
        out.append(blur)
    return out


def robust_norm(a, p=99.0):
    vmax = np.percentile(a, p)
    if vmax < 1e-6:
        return np.zeros_like(a)
    return np.clip(a / vmax, 0, 1)


def save_gray(path, arr01):
    arr = (arr01 * 255).astype(np.uint8)
    cv2.imwrite(path, arr)


def save_color(path, arr01):
    """
    Save 0–1 float array as jet-colored PNG (uint8 BGR).
    """
    arr = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    cv2.imwrite(path, color)


# -------------- PROCESS ONE MASK --------------

def process_one(mask, radius, prefix, ID):
    impulses = low_density_impulses(mask, radius)
    blurred = make_blurred_maps(impulses)

    for frac, m in zip(SIGMA_FRACS, blurred):
        m_norm = robust_norm(m)

        gray_path = os.path.join(
            OUT_ROOT,
            f"{prefix}_{ID}_sigma{frac:.3f}.png"
        )
        color_path = os.path.join(
            OUT_ROOT,
            f"{prefix}_{ID}_sigma{frac:.3f}_COLOR.png"
        )

        save_gray(gray_path, m_norm)
        save_color(color_path, m_norm)

        print("Saved:", gray_path)
        print("Saved:", color_path)


# -------------- MAIN LOOP OVER IDS --------------

if __name__ == "__main__":
    for pid in range(10001, 10301):  # inclusive of 10300
        ID = str(pid)
        print(f"\n=== Processing ID {ID} ===\n")

        CAP_PATH = os.path.join(CAP_ROOT, f"{ID}.bmp")
        LV_PATH  = os.path.join(LV_ROOT,  f"{ID}.png")

        # Large vessel
        if os.path.exists(LV_PATH):
            try:
                LV = load_binary(LV_PATH)
                process_one(LV, radius=R_LARGE, prefix="largevessel", ID=ID)
            except Exception as e:
                print(f"Error processing large vessels {ID}: {e}")
        else:
            print("Missing large vessel file:", LV_PATH)

        # Capillaries
        if os.path.exists(CAP_PATH):
            try:
                CAP = load_binary(CAP_PATH)
                process_one(CAP, radius=R_CAP, prefix="capillary", ID=ID)
            except Exception as e:
                print(f"Error processing capillaries {ID}: {e}")
        else:
            print("Missing capillary file:", CAP_PATH)
