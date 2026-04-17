#!/usr/bin/env python3

# =========================================================================================
# PURPOSE:
# Train/evaluate a 1-channel ResNet-18 on RAW OCTA images with 5-fold CV, using external
# multiplicative attention from grayscale tortuosity hotspot heatmaps.
#
# This version runs separate experiment batches for:
#   • Artery hotspot heatmaps  (σ = 0.020, 0.040, 0.060, 0.080)
#   • Vein hotspot heatmaps    (σ = 0.020, 0.040, 0.060, 0.080)
#   • Large-vessel hotspots    (σ = 0.020, 0.040, 0.060, 0.080)
#   • Capillary hotspots       (σ = 0.020, 0.040, 0.060, 0.080)
#   • Inverted capillary hotspots (same capillary maps, but attention inverted so
#       darker / blacker regions receive higher attention)
#
# Each experiment saves:
#   • Text results:
#       {OUTPUT_ROOT}/ML_results_external_attention_{GROUP}_hotspot_sigma_PCT_<sigma>
#   • Best models:
#       {OUTPUT_ROOT}/models_external_attention_{GROUP}_sigma_<sigma>/best_resnet18_foldK.pth
#
# BINARY LABELING:
#   AMD    → 1 (positive)
#   NORMAL → 0 (negative)
#
# INPUTS:
#   Raw OCTA (png):
#       /home/mgmitzner/Tortuosity_training/OCTA500_images_6_mm/{ID}.png
#
#   Hotspot attention maps (grayscale "GRAY" PNGs):
#       Artery:
#         /home/mgmitzner/Tortuosity_training/11_12_25/11_05_heatmaps_for_DL/artery/hotspots/
#           artery_{ID}_hotspot_sigma0.020_GRAY.png
#
#       Vein:
#         /home/mgmitzner/Tortuosity_training/11_12_25/11_05_heatmaps_for_DL/vein/hotspots/
#           vein_{ID}_hotspot_sigma0.020_GRAY.png
#
#       Large vessels:
#         /home/mgmitzner/Tortuosity_training/11_12_25/11_05_heatmaps_for_DL/large_vessel/hotspots/
#           largevessel_{ID}_hotspot_sigma0.020_GRAY.png
#
#       Capillaries (and inverted capillaries, using the same files):
#         /home/mgmitzner/Tortuosity_training/11_12_25/11_05_heatmaps_for_DL/capillary/hotspots/
#           capillary_{ID}_hotspot_sigma0.020_GRAY.png
#
#       (σ values: 0.020, 0.040, 0.060, 0.080)
#
# PRETRAINED WEIGHTS:
#   Backbone: torchvision ResNet-18 pretrained on ImageNet-1K
#   Grayscale adaptation: 3-channel conv1 filters averaged (RGB mean) to init a 1-channel conv1.
#
# OPTIMIZATION:
#   - Warmup: train classification head only for WARMUP_EPOCHS (backbone frozen), LR=1e-3
#   - Fine-tune: unfreeze entire network, LR=1e-4
#   - Optimizer: Adam (weight decay 1e-4)
#   - Scheduler: ReduceLROnPlateau on validation Balanced Accuracy
#
# LOSS:
#   BCE Focal Loss with logits (single output neuron + sigmoid)
#   FL = - α_t * (1 - p_t)^γ * log(p_t), emphasizing AMD (positive class)
#   Settings: α=0.75 (for AMD), γ=2.0
#
# METRICS:
#   Accuracy, Balanced Accuracy, Specificity (class 0=NORMAL), Sensitivity (class 1=AMD), ROC AUC
# =========================================================================================

import os
import re
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix,
)

from tqdm import tqdm

# ------------------------ PATHS / CONSTANTS ------------------------
RAW_DIR = "/home/mgmitzner/Tortuosity_training/OCTA500_images_6_mm"  # raw OCTA PNGs
DEMOGRAPHICS_XLSX = "/home/mgmitzner/Tortuosity_training/Demographic_clinical_OCTA_500.xlsx"  # labels
OUTPUT_ROOT = "/home/mgmitzner/Tortuosity_training/11_12_25/output_a_v_lv_c_ic"  # where results/models are saved
os.makedirs(OUTPUT_ROOT, exist_ok=True)  # ensure output root exists

# ------------------------ ATTENTION SOURCES ------------------------
# Hotspot roots and filename prefixes per group
HOTSPOT_BASE = "/home/mgmitzner/Tortuosity_training/11_12_25/11_05_heatmaps_for_DL"

# (group_tag, filename_prefix, hotspot_dir, invert_heatmap_flag)
ATTN_GROUPS = [
    ("artery",        "artery",      os.path.join(HOTSPOT_BASE, "artery", "hotspots"),       False),
    ("vein",          "vein",        os.path.join(HOTSPOT_BASE, "vein", "hotspots"),         False),
    ("large_vessel",  "largevessel", os.path.join(HOTSPOT_BASE, "large_vessel", "hotspots"), False),
    ("capillary",     "capillary",   os.path.join(HOTSPOT_BASE, "capillary", "hotspots"),    False),
    # Inverse experiments: use same capillary GRAY maps, but invert attention
    ("capillary_inv", "capillary",   os.path.join(HOTSPOT_BASE, "capillary", "hotspots"),    True),
]

SIGMA_TAGS = ["0.020", "0.040", "0.060", "0.080"]  # corresponds to "sigma0.020" etc in filenames

ID_MIN, ID_MAX = 10001, 10300  # limit IDs to the OCTA500 range

N_FOLDS = 5
RANDOM_SEED = 42

BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_WORKERS = 2
IMG_SIZE = 224

# Pretraining + schedule
PRETRAINED = True                # ImageNet-1K weights
WARMUP_EPOCHS = 2                # head-only epochs
LR_WARMUP_HEAD = 1e-3
LR_FINETUNE_ALL = 1e-4
WEIGHT_DECAY = 1e-4

# Focal loss settings (AMD=1 is positive)
USE_BCE_FOCAL = True
FOCAL_ALPHA_POS = 0.75           # emphasize AMD
FOCAL_GAMMA = 2.0

SAVE_MODELS = True

# Attention fusion bounds used to convert normalized heatmap to per-pixel weights.
ATTEN_MIN = 0.5
ATTEN_MAX = 1.5

# ------------------------ REPRODUCIBILITY ------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------------ LABEL LOADING (AMD vs NORMAL only) ------------------------
def load_labels_frame(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)

    disease_col = None
    for c in ["Disease", "Disease_Name", "Diagnosis", "Label"]:
        if c in df.columns:
            disease_col = c
            break

    assert "ID" in df.columns, "Excel must contain an 'ID' column"
    assert disease_col is not None, "Excel must include a disease label column"

    df = df[(df["ID"] >= ID_MIN) & (df["ID"] <= ID_MAX)].copy()

    dz = df[disease_col].astype(str).str.strip().str.upper()
    mask = dz.isin(["AMD", "NORMAL"])
    df = df[mask].copy()
    dz = dz[mask]

    df["label"] = (dz == "AMD").astype(int)  # AMD=1, NORMAL=0
    return df[["ID", "label"]].copy()

# ------------------------ HEATMAP INDEXING ------------------------
def build_heatmap_index(hm_dir: str, sigma_tag: str, prefix: str) -> Dict[int, str]:
    """
    Return {ID: path} for files in hm_dir whose names match:
        {prefix}_{ID}_hotspot_sigma{sigma_tag}_GRAY.png

    Examples:
      artery_10001_hotspot_sigma0.020_GRAY.png
      vein_10001_hotspot_sigma0.040_GRAY.png
      largevessel_10001_hotspot_sigma0.060_GRAY.png
      capillary_10001_hotspot_sigma0.080_GRAY.png
    """
    pattern = rf"^{re.escape(prefix)}_(?P<id>\d+)_hotspot_sigma{sigma_tag}_GRAY\.png$"
    pat = re.compile(pattern, re.IGNORECASE)

    index: Dict[int, str] = {}
    if not os.path.isdir(hm_dir):
        return index

    for name in os.listdir(hm_dir):
        m = pat.match(name)
        if not m:
            continue
        pid = int(m.group("id"))
        index[pid] = os.path.join(hm_dir, name)

    return index

# ------------------------ DATASET WITH EXTERNAL ATTENTION ------------------------
class OCTAWithAttentionDataset(Dataset):
    """Fuse raw OCTA with hotspot grayscale map (multiplicative attention) → 1-channel image."""

    def __init__(self,
                 raw_paths: List[str],
                 hm_paths: List[str],
                 labels: List[int],
                 ids: List[int],
                 invert_heatmap: bool = False):
        self.raw_paths = raw_paths
        self.hm_paths = hm_paths
        self.labels = labels
        self.ids = ids
        self.invert_heatmap = invert_heatmap

        self.resize_to = (IMG_SIZE, IMG_SIZE)
        self.to_tensor = T.ToTensor()
        # ImageNet-like single-channel normalization
        self.normalize = T.Normalize(mean=[0.485], std=[0.229])

    def __len__(self) -> int:
        return len(self.raw_paths)

    def _load_gray_np(self, path: str) -> np.ndarray:
        return np.array(Image.open(path).convert("L"), dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw = self._load_gray_np(self.raw_paths[idx])  # 0..255
        hm  = self._load_gray_np(self.hm_paths[idx])   # 0..255

        # Normalize heatmap inside non-zero support, keep background at 0
        nz = hm > 0
        if nz.any():
            hm_nz = hm[nz]
            if hm_nz.max() > hm_nz.min():
                hm_norm = (hm - hm_nz.min()) / (hm_nz.max() - hm_nz.min())
                hm_norm[~nz] = 0.0
            else:
                hm_norm = np.zeros_like(hm, dtype=np.float32)
        else:
            hm_norm = np.zeros_like(hm, dtype=np.float32)

        # Optional inversion for "inverted capillaries" experiments:
        #   - original: bright = high tortuosity hotspot → high attention
        #   - inverted: dark = high attention (hm_norm := 1 - hm_norm)
        if self.invert_heatmap:
            hm_norm = 1.0 - hm_norm

        # Multiplicative attention (unchanged mechanism)
        W = ATTEN_MIN + hm_norm * (ATTEN_MAX - ATTEN_MIN)
        raw_fused = raw * W
        raw_fused_img = Image.fromarray(np.clip(raw_fused, 0, 255).astype(np.uint8)).resize(
            self.resize_to, resample=Image.BILINEAR
        )

        x = self.to_tensor(raw_fused_img)
        x = self.normalize(x)
        y = torch.tensor(self.labels[idx]).long()
        pid = self.ids[idx]
        return {"image": x, "label": y, "id": pid}

# ------------------------ MODEL: ResNet-18 (1 logit for BCE) ------------------------
def resnet18_gray_bce(pretrained: bool = PRETRAINED) -> nn.Module:
    """
    ResNet-18 with:
      - conv1 adapted to 1 input channel (initialized from RGB-mean if pretrained)
      - final FC → 1 output logit (for BCEWithLogits / Focal BCE)
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    w = model.conv1.weight.data.clone()  # (64,3,7,7)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        model.conv1.weight.data = w.mean(dim=1, keepdim=True)

    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, 1)  # single logit
    return model

# ------------------------ LOSS / METRICS ------------------------
class BCEFocalLoss(nn.Module):
    """Binary focal loss with logits."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)  # (N,)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (N,)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)  # (N,)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)  # (N,)
        focal = (1.0 - p_t).clamp(min=1e-6).pow(self.gamma)
        loss = alpha_t * focal * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # class 0 (NORMAL)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # class 1 (AMD)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "auc": auc,
    }

# ------------------------ TRAINING HELPERS ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # speed for fixed-size inputs

def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze all layers except the classification head (fc)."""
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = not (freeze)

def make_optimizer(model: nn.Module, lr: float) -> optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)

# ------------------------ ONE-FOLD TRAIN/EVAL ------------------------
def run_one_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    paths_raw: List[str],
    paths_hm: List[str],
    labels: List[int],
    ids: List[int],
    save_dir: str,
    invert_heatmap: bool,
) -> Dict[str, float]:

    train_ds = OCTAWithAttentionDataset(
        [paths_raw[i] for i in train_idx],
        [paths_hm[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [ids[i] for i in train_idx],
        invert_heatmap=invert_heatmap,
    )
    val_ds = OCTAWithAttentionDataset(
        [paths_raw[i] for i in val_idx],
        [paths_hm[i] for i in val_idx],
        [labels[i] for i in val_idx],
        [ids[i] for i in val_idx],
        invert_heatmap=invert_heatmap,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = resnet18_gray_bce(pretrained=PRETRAINED).to(device)

    # Loss: BCE Focal (emphasize AMD) or fallback BCE with logits
    if USE_BCE_FOCAL:
        criterion = BCEFocalLoss(alpha=FOCAL_ALPHA_POS, gamma=FOCAL_GAMMA, reduction="mean")
    else:
        criterion = nn.BCEWithLogitsLoss()

    # ---- Warmup (head only) ----
    freeze_backbone(model, freeze=True)
    optimizer = make_optimizer(model, lr=LR_WARMUP_HEAD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=0.5, patience=2)

    best_bal_acc = -1.0
    best_metrics = None

    for epoch in range(1, NUM_EPOCHS + 1):
        # Transition to fine-tune
        if epoch == WARMUP_EPOCHS + 1:
            freeze_backbone(model, freeze=False)
            optimizer = make_optimizer(model, lr=LR_FINETUNE_ALL)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                             factor=0.5, patience=2)
            print(f"[Fold {fold_idx}] Unfroze backbone → fine-tuning with LR={LR_FINETUNE_ALL:g}")

        # ----- Train -----
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True).float()  # BCE expects float targets

            optimizer.zero_grad()
            logits = model(x).squeeze(1)           # (N,)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ----- Validate -----
        model.eval()
        all_prob, all_pred, all_true = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True).float()
                logits = model(x).squeeze(1)       # (N,)
                probs = torch.sigmoid(logits)      # (N,)
                preds = (probs >= 0.5).long()

                all_prob.append(probs.cpu().numpy())
                all_pred.append(preds.cpu().numpy())
                all_true.append(y.cpu().numpy().astype(np.int64))

        all_prob = np.concatenate(all_prob) if all_prob else np.array([])
        all_pred = np.concatenate(all_pred) if all_pred else np.array([])
        all_true = np.concatenate(all_true) if all_true else np.array([])

        metrics = compute_metrics(all_true, all_prob, all_pred)
        scheduler.step(metrics["balanced_accuracy"])

        phase = "WARMUP" if epoch <= WARMUP_EPOCHS else "FINETUNE"
        print(f"[Fold {fold_idx}] ({phase}) Epoch {epoch:02d}/{NUM_EPOCHS} "
              f"train_loss={train_loss:.4f}  "
              f"acc={metrics['accuracy']:.4f}  "
              f"bal_acc={metrics['balanced_accuracy']:.4f}  "
              f"sens(AMD)={metrics['sensitivity']:.4f}  "
              f"spec(NORM)={metrics['specificity']:.4f}  "
              f"auc={metrics['auc']:.4f}")

        # Save best per fold
        if metrics["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = metrics["balanced_accuracy"]
            best_metrics = metrics
            if SAVE_MODELS:
                os.makedirs(save_dir, exist_ok=True)
                fold_ckpt = os.path.join(save_dir, f"best_resnet18_fold{fold_idx}.pth")
                torch.save(model.state_dict(), fold_ckpt)
                print(f"  ↳ [Fold {fold_idx}] Saved new best model to {fold_ckpt}")

    if best_metrics is None:
        best_metrics = {"accuracy": np.nan, "balanced_accuracy": np.nan,
                        "specificity": np.nan, "sensitivity": np.nan, "auc": np.nan}

    return best_metrics

# ------------------------ PAIRING RAW+HEATMAPS ------------------------
def collect_pairs(df_labels: pd.DataFrame,
                  hm_index: Dict[int, str]) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Intersect label IDs with available heatmaps and raw images.
    Keep only samples that have BOTH raw image and heatmap present.
    """
    paths_raw, paths_hm, labels, ids = [], [], [], []
    missing = 0
    for _, row in df_labels.iterrows():
        pid = int(row["ID"])
        if pid not in hm_index:
            missing += 1
            continue
        raw_path = os.path.join(RAW_DIR, f"{pid}.png")
        if not os.path.isfile(raw_path):
            missing += 1
            continue
        paths_raw.append(raw_path)
        paths_hm.append(hm_index[pid])
        labels.append(int(row["label"]))  # AMD=1, NORMAL=0
        ids.append(pid)
    print(f"Using {len(paths_raw)} matched pairs (skipped {missing} non-matches)")
    return paths_raw, paths_hm, labels, ids

# ------------------------ RUN ONE FULL EXPERIMENT ------------------------
def run_experiment(
    heatmap_dir: str,
    sigma_tag: str,
    group_tag: str,
    prefix: str,
    invert_heatmap: bool,
) -> None:
    """
    Run full 5-fold CV experiment for a given:
      - heatmap directory
      - sigma value (sigma_tag string like '0.020')
      - group_tag (e.g. 'artery', 'vein', 'large_vessel', 'capillary', 'capillary_inv')
      - filename prefix in hotspot PNGs (e.g. 'artery', 'vein', 'largevessel', 'capillary')
      - invert_heatmap: whether to use 1 - hm_norm before attention fusion
    """
    set_seed(RANDOM_SEED)
    df = load_labels_frame(DEMOGRAPHICS_XLSX)

    hm_index = build_heatmap_index(heatmap_dir, sigma_tag=sigma_tag, prefix=prefix)
    assert len(hm_index) > 0, (
        f"No hotspot GRAY maps found in {heatmap_dir} for prefix='{prefix}' "
        f"and sigma_tag='{sigma_tag}'"
    )

    paths_raw, paths_hm, labels, ids = collect_pairs(df, hm_index)
    assert len(paths_raw) > 0, f"No matched raw+heatmap pairs for {heatmap_dir}"

    labels_np = np.array(labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Model save dir is namespaced by group + sigma
    save_dir = os.path.join(OUTPUT_ROOT, f"models_external_attention_{group_tag}_sigma_{sigma_tag}")
    os.makedirs(save_dir, exist_ok=True)

    fold_results: List[Dict[str, float]] = []
    for k, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_np)), labels_np), start=1):
        print(f"\n========== Fold {k}/{N_FOLDS} (group={group_tag}, σ={sigma_tag}) ==========")
        print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)} | "
              f"Train AMD={int(labels_np[train_idx].sum())} | Val AMD={int(labels_np[val_idx].sum())}")
        set_seed(RANDOM_SEED + k)
        best_metrics = run_one_fold(
            k, train_idx, val_idx,
            paths_raw, paths_hm, labels, ids,
            save_dir=save_dir,
            invert_heatmap=invert_heatmap,
        )
        fold_results.append(best_metrics)

    def mean_std(vals: List[float]) -> Tuple[float, float]:
        arr = np.array(vals, dtype=float)
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    accs  = [m["accuracy"] for m in fold_results]
    bals  = [m["balanced_accuracy"] for m in fold_results]
    specs = [m["specificity"] for m in fold_results]
    senss = [m["sensitivity"] for m in fold_results]
    aucs  = [m["auc"] for m in fold_results]

    m_acc, s_acc   = mean_std(accs)
    m_bal, s_bal   = mean_std(bals)
    m_spec, s_spec = mean_std(specs)
    m_sens, s_sens = mean_std(senss)
    m_auc, s_auc   = mean_std(aucs)

    print("\n================ Overall 5-Fold Summary "
          f"(group={group_tag}, σ={sigma_tag}) ================")
    print(f"Accuracy           : {m_acc:.4f} ± {s_acc:.4f}")
    print(f"Balanced Accuracy  : {m_bal:.4f} ± {s_bal:.4f}")
    print(f"Specificity (0)    : {m_spec:.4f} ± {s_spec:.4f}")
    print(f"Sensitivity (1)    : {m_sens:.4f} ± {s_sens:.4f}")
    print(f"AUC                : {m_auc:.4f} ± {s_auc:.4f}")

    # Results file is namespaced by group + sigma
    results_filename = f"ML_results_external_attention_{group_tag}_hotspot_sigma_PCT_{sigma_tag}"
    results_path = os.path.join(OUTPUT_ROOT, results_filename)
    with open(results_path, "w") as f:
        f.write(f"Raw images dir   : {RAW_DIR}\n")
        f.write(f"Heatmaps dir     : {heatmap_dir}\n")
        f.write(f"Group            : {group_tag}\n")
        f.write(f"Prefix           : {prefix}\n")
        f.write(f"Invert heatmap   : {invert_heatmap}\n")
        f.write(f"Sigma tag        : {sigma_tag}\n")
        f.write(f"Demographics     : {DEMOGRAPHICS_XLSX}\n")
        f.write(f"ID range         : {ID_MIN}-{ID_MAX}\n")
        f.write(f"Pairs used       : {len(paths_raw)}\n")
        f.write(f"ATTEN_MIN..MAX   : {ATTEN_MIN}..{ATTEN_MAX}\n")
        f.write(f"Pretrained       : ResNet-18 ImageNet-1K; conv1 → grayscale via RGB-mean init\n")
        f.write(f"Loss             : BCE Focal (alpha={FOCAL_ALPHA_POS}, gamma={FOCAL_GAMMA})\n")
        f.write(f"Schedule         : Warmup {WARMUP_EPOCHS}e head@{LR_WARMUP_HEAD}, "
                f"then finetune all@{LR_FINETUNE_ALL}\n\n")
        for i, m in enumerate(fold_results, start=1):
            f.write(f"[Fold {i}] accuracy={m['accuracy']:.4f}, "
                    f"balanced_accuracy={m['balanced_accuracy']:.4f}, "
                    f"specificity(0)={m['specificity']:.4f}, "
                    f"sensitivity(1)={m['sensitivity']:.4f}, "
                    f"auc={m['auc']:.4f}\n")
        f.write("\n================ Overall 5-Fold Summary ================\n")
        f.write(f"Accuracy           : {m_acc:.4f} ± {s_acc:.4f}\n")
        f.write(f"Balanced Accuracy  : {m_bal:.4f} ± {s_bal:.4f}\n")
        f.write(f"Specificity (0)    : {m_spec:.4f} ± {s_spec:.4f}\n")
        f.write(f"Sensitivity (1)    : {m_sens:.4f} ± {s_sens:.4f}\n")
        f.write(f"AUC                : {m_auc:.4f} ± {s_auc:.4f}\n")
    print(f"\nSaved results to: {results_path}")

# ------------------------ MAIN: RUN ALL EXPERIMENTS ------------------------
if __name__ == "__main__":
    for group_tag, prefix, hm_dir, invert_heatmap in ATTN_GROUPS:
        for sigma_tag in SIGMA_TAGS:
            print("\n" + "=" * 78)
            print(f"Starting experiment for group={group_tag}, sigma_PCT={sigma_tag}")
            print(f"Using heatmaps at: {hm_dir}")
            print(f"Filename prefix: {prefix}_, invert_heatmap={invert_heatmap}")
            print("=" * 78)
            run_experiment(
                heatmap_dir=hm_dir,
                sigma_tag=sigma_tag,
                group_tag=group_tag,
                prefix=prefix,
                invert_heatmap=invert_heatmap,
            )
