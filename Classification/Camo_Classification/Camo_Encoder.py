#!/usr/bin/env python3
"""
Simple classifier comparison across three modalities: SPEC, POL, SPECPOL.
All use the same simple CNN architecture for fair comparison.
"""
from pathlib import Path
import os, csv, time, statistics as stats, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import BinaryAUROC
import torchvision.transforms.functional as TF
import tifffile as tiff
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics')

# ============================================================================
# CONFIGURATION
# ============================================================================
SEED = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MANIFEST = Path("/scratch/general/nfs1/u1528328/img_dir/Camoflage_Raw/camo_manifest_plants_or_camo_raw.csv")
IMG_SIZE = (64, 64)  # Smaller for speed
BANDS_INTENSITY = 106

# Training hyperparameters
EPOCHS = 10
LR = 3e-4
WD = 1e-4
BATCH_SIZE = 16

# Cross-validation
FOLD_COL = "cv_fold"
TEST_FOLDS = [0, 1, 2, 3, 4]
MODALITIES = ["SPEC", "POL", "SPECPOL"]

# Output
CKPT_DIR = Path("checkpoints_2D")
RESULTS_CSV = Path("results/summary_comparison.csv")

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# DETERMINISM
# ============================================================================
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================================
# DATA LOADING
# ============================================================================
def _ensure_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.shape[0] not in (106, 212) and arr.shape[-1] in (106, 212):
        arr = np.moveaxis(arr, -1, 0)
    return arr

def _clean_finite(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    bad = ~np.isfinite(arr)
    if bad.any():
        arr[bad] = 0.0
    return arr

def read_cube_106(path: str) -> np.ndarray:
    try:
        arr = tiff.imread(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read TIFF: {path}") from e
    arr = _ensure_chw(arr)
    arr = _clean_finite(arr)
    if arr.shape[0] < BANDS_INTENSITY:
        raise ValueError(f"{path}: expected >= {BANDS_INTENSITY} bands")
    return arr[:BANDS_INTENSITY, ...].astype(np.float32, copy=False)

# def make_spec(pol0, pol45, pol90, pol135) -> torch.Tensor:
#     """SPEC: Average angles, subsample 10 spectral bands -> (10,H,W)"""
#     cube = (pol0 + pol45 + pol90 + pol135) / 4.0
#     # Subsample every ~10th band for speed
#     cube = cube[::10, :, :][:10]  # Take 10 bands
#     t = torch.from_numpy(cube)
#     t = TF.resize(t, IMG_SIZE, antialias=True)
#     t = (t - t.mean()) / (t.std() + 1e-6)
#     return t.clamp_(-3.0, 3.0).contiguous()

def make_spec(pol0, pol45, pol90, pol135) -> torch.Tensor:
    """
    SPEC: Average angles, pick 10 bands:
      - 5 from NIR (~750–850 nm)
      - 5 spaced across shorter wavelengths (~450–750 nm)
    Output shape: (10, H, W)
    Assumes 106 bands linearly spaced from 450 to 850 nm.
    """
    cube = (pol0 + pol45 + pol90 + pol135) / 4.0  # (106, H, W)
    B = cube.shape[0]
    wl_min, wl_max = 450.0, 850.0
    wl_750 = 750.0

    # Map wavelength -> band index (0..B-1), assuming linear spacing
    def lam_to_idx(lam: float) -> int:
        return int(round((lam - wl_min) / (wl_max - wl_min) * (B - 1)))

    idx_750 = lam_to_idx(wl_750)
    idx_last = B - 1

    # 5 bands in NIR (750–850 nm), evenly spaced including ends
    nir_idx = np.linspace(idx_750, idx_last, 10).round().astype(int)

    # # 5 bands across shorter wavelengths (450–750 nm), evenly spaced
    # short_idx = np.linspace(0, max(idx_750 - 1, 0), 5).round().astype(int)

    # Order: NIR first (per request), then shorter wavelengths -> total 10
    pick = nir_idx

    # Safety: clamp and unique while preserving order (should already be unique)
    pick = np.clip(pick, 0, B - 1)
    # (Optional) ensure uniqueness without reordering much
    seen, ordered = set(), []
    for i in pick:
        if i not in seen:
            ordered.append(i); seen.add(i)
    pick = np.array(ordered, dtype=int)

    # Select and normalize
    cube = cube[pick, :, :]  # (10, H, W)
    t = torch.from_numpy(cube)
    t = TF.resize(t, IMG_SIZE, antialias=True)
    t = (t - t.mean()) / (t.std() + 1e-6)
    return t.clamp_(-3.0, 3.0).contiguous()


def make_pol(pol0, pol45, pol90, pol135) -> torch.Tensor:
    """POL: 4 polarization angles, spectral mean -> (4,H,W)"""
    # Average over spectral dimension
    p0 = pol0.mean(axis=0, keepdims=True)
    p45 = pol45.mean(axis=0, keepdims=True)
    p90 = pol90.mean(axis=0, keepdims=True)
    p135 = pol135.mean(axis=0, keepdims=True)
    
    x = np.concatenate([p0, p45, p90, p135], axis=0)  # (4,H,W)
    x = torch.from_numpy(x)
    x = TF.resize(x, IMG_SIZE, antialias=True)
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x.clamp_(-3.0, 3.0).contiguous()

def make_specpol(pol0, pol45, pol90, pol135) -> torch.Tensor:
    """SPECPOL: Fused representation -> (14,H,W)
    10 spectral bands + 4 polarization angles combined"""
    # Get 10 spectral bands (from averaged angles)
    spec = make_spec(pol0, pol45, pol90, pol135)  # (10,H,W)
    
    # Get 4 polarization angles (spectrally averaged)
    pol = make_pol(pol0, pol45, pol90, pol135)    # (4,H,W)
    
    # Concatenate: spectral + polarization info
    fused = torch.cat([spec, pol], dim=0)  # (14,H,W)
    return fused.contiguous()

class CamoDataset(Dataset):
    def __init__(self, manifest_csv: Path, split_folds, modality: str, fold_col: str, train: bool):
        self.df = pd.read_csv(manifest_csv)
        if split_folds is not None:
            self.df = self.df[self.df[fold_col].isin(split_folds)].reset_index(drop=True)
        self.modality = modality.upper()
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        pol0 = read_cube_106(r["pol0_path"])
        pol45 = read_cube_106(r["pol45_path"])
        pol90 = read_cube_106(r["pol90_path"])
        pol135 = read_cube_106(r["pol135_path"])

        if self.modality == "SPEC":
            x = make_spec(pol0, pol45, pol90, pol135)
        elif self.modality == "POL":
            x = make_pol(pol0, pol45, pol90, pol135)
        else:  # SPECPOL
            x = make_specpol(pol0, pol45, pol90, pol135)

        # Simple augmentation
        if self.train:
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=(-1,))
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=(-2,))

        y = torch.tensor(1 if str(r["label"]).lower() == "camo" else 0, dtype=torch.long)
        return x, y

def make_loaders(batch_size, modality, train_folds, test_folds, fold_col):
    ds_tr = CamoDataset(MANIFEST, train_folds, modality, fold_col, train=True)
    ds_te = CamoDataset(MANIFEST, test_folds, modality, fold_col, train=False)

    # Balance classes
    ys = [1 if str(ds_tr.df.iloc[i]["label"]).lower() == "camo" else 0 for i in range(len(ds_tr))]
    n0, n1 = ys.count(0), ys.count(1)
    w = [1.0 / max(n0, 1) if y == 0 else 1.0 / max(n1, 1) for y in ys]
    sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, 
                       num_workers=2, pin_memory=True, persistent_workers=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=True, persistent_workers=True)
    return dl_tr, dl_te

# ============================================================================
# SIMPLE CNN MODEL (SAME FOR ALL MODALITIES)
# ============================================================================
class SimpleClassifier(nn.Module):
    """Simple CNN - identical architecture for all modalities."""
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: in_channels -> 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64->32
            
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32->16
            
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16->8
            
            # Block 4: 128 -> 128
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def build_model(modality: str):
    """Build model with correct input channels for each modality."""
    if modality == "SPEC":
        return SimpleClassifier(in_channels=10)  # 10 spectral bands
    elif modality == "POL":
        return SimpleClassifier(in_channels=4)   # 4 polarization angles
    else:  # SPECPOL
        return SimpleClassifier(in_channels=14)  # 10 + 4 fused

# ============================================================================
# TRAINING
# ============================================================================
def train_one_epoch(model, dl, opt, loss_fn):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_sum += float(loss) * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return (loss_sum / total), (correct / total)

@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    total, correct = 0, 0
    auroc = BinaryAUROC().to(DEVICE)
    all_probs, all_y = [], []

    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]
        auroc.update(probs, y)
        all_probs.append(probs)
        all_y.append(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    all_probs = torch.cat(all_probs)
    all_y = torch.cat(all_y)

    # Find best F1 threshold
    thresholds = torch.linspace(0.1, 0.9, 50, device=DEVICE)
    best_f1, best_acc = 0.0, 0.0
    
    for t in thresholds:
        preds = (all_probs >= t).long()
        tp = ((preds == 1) & (all_y == 1)).sum().item()
        tn = ((preds == 0) & (all_y == 0)).sum().item()
        fp = ((preds == 1) & (all_y == 0)).sum().item()
        fn = ((preds == 0) & (all_y == 1)).sum().item()
        
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        
        if f1 > best_f1:
            best_f1 = f1
            best_acc = (tp + tn) / max(total, 1)

    auc = float(auroc.compute().cpu())
    return (correct / total), auc, best_f1, best_acc

def train_one_fold(modality, test_fold):
    train_folds = [k for k in TEST_FOLDS if k != test_fold]
    
    dl_tr, dl_te = make_loaders(BATCH_SIZE, modality, train_folds, [test_fold], FOLD_COL)
    print(f"[{modality} Fold {test_fold}] Train: {len(dl_tr.dataset)} | Test: {len(dl_te.dataset)}")

    model = build_model(modality).to(DEVICE)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(opt, T_max=EPOCHS)
    
    best_auc, best_state = -1.0, None

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, opt, loss_fn)
        te_acc, te_auc, te_f1, te_acc_f1 = evaluate(model, dl_te)
        scheduler.step()

        print(f"  Ep {ep:02d} | Train: {tr_loss:.3f}/{tr_acc:.3f} | "
              f"Val: Acc={te_acc:.3f} AUC={te_auc:.3f} F1={te_f1:.3f}")

        if te_auc > best_auc:
            best_auc = te_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Save best
    ckpt_path = CKPT_DIR / modality.lower() / f"{modality.lower()}_fold{test_fold}.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state:
        torch.save(best_state, ckpt_path)
        model.load_state_dict(best_state)

    # Final eval
    te_acc, te_auc, te_f1, te_acc_f1 = evaluate(model, dl_te)
    return dict(modality=modality, fold=test_fold, val_acc=te_acc, 
                val_auc=te_auc, val_f1=te_f1, ckpt=str(ckpt_path))

# ============================================================================
# MAIN
# ============================================================================
def main():
    seed_everything(SEED)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    t0 = time.time()
    
    for modality in MODALITIES:
        print(f"\n{'='*60}\n{modality} Modality\n{'='*60}")
        for k in TEST_FOLDS:
            row = train_one_fold(modality, k)
            all_rows.append(row)

    # Save results
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["modality", "fold", "val_acc", "val_auc", "val_f1", "ckpt"])
        w.writeheader()
        w.writerows(all_rows)

    # Summary by modality
    print(f"\n{'='*60}\nCOMPARISON SUMMARY\n{'='*60}")
    by_mod = {m: [r for r in all_rows if r["modality"] == m] for m in MODALITIES}
    
    for m, rows in by_mod.items():
        aucs = [r["val_auc"] for r in rows]
        accs = [r["val_acc"] for r in rows]
        f1s = [r["val_f1"] for r in rows]
        
        print(f"\n{m:8s}:")
        print(f"  AUC: {stats.mean(aucs):.3f} ± {stats.pstdev(aucs):.3f}")
        print(f"  Acc: {stats.mean(accs):.3f} ± {stats.pstdev(accs):.3f}")
        print(f"  F1:  {stats.mean(f1s):.3f} ± {stats.pstdev(f1s):.3f}")
    
    print(f"\n{'='*60}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")
    print(f"Results: {RESULTS_CSV}")

if __name__ == "__main__":
    main()