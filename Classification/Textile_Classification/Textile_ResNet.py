#!/usr/bin/env python3
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import update_bn
from torch.optim.swa_utils import AveragedModel

# ====== Config ======
FEATDIR  = Path("/uufs/chpc.utah.edu/common/home/u1528328/Classification/Textile_Classification/features_all")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SEED     = 1337
FOLDS    = 5

# Improved hyperparameters
EPOCHS   = 100
LR       = 5e-4  # Lower learning rate for stability
WD       = 5e-4  # Increased weight decay for regularization
BATCH    = 32    # Smaller batch size for better generalization
PATIENCE = 15

# SWA settings
SWA_EPOCHS = 8
SWA_LR     = 1e-4

# Augmentation (light)
MIXUP_ALPHA = 0.2
SPEC_NOISE_STD = 0.02  # Gaussian noise for SPEC/SPECPOL

# --- add with your imports ---
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# fixed class colors you asked for
COL = {"cotton": "#1f77b4", "felt": "#d62728", "nylon": "#2ca02c"}

def _canon_names(names):
    """
    Ensure label names are 'cotton', 'felt', 'nylon' in that order.
    If your run returns ['0','1','2'] (or ints), map them deterministically.
    Adjust the mapping here if your numeric labels map differently IRL.
    """
    # if already text labels, normalize case and return
    low = [str(n).lower() for n in names]
    if set(low) == {"cotton", "felt", "nylon"}:
        return ["cotton", "felt", "nylon"]

    # default numeric -> textile mapping (edit if needed)
    num_to_name = {0: "cotton", 1: "felt", 2: "nylon"}
    as_int = []
    for n in names:
        try:
            as_int.append(int(n))
        except Exception:
            # fallback: keep string, but you probably won't need this
            as_int.append(n)
    mapped = [num_to_name.get(k, str(k)) for k in as_int]
    return mapped

def _plot_three_confusions(results, save_path=None):
    """
    results: list of tuples -> [(title, cm, names), ...]
      - title: str (e.g., 'SPEC', 'POL', 'SPECPOL')
      - cm:    np.ndarray of shape (3,3)
      - names: iterable of class names; will be canonicalized to cotton/felt/nylon
    """
    title_map = {
        "SPEC":    "Spectral Classification",
        "POL":     "Polarimetric Classification",
        "SPECPOL": "Spectro-Polarimetric Classification"
    }

    data = []
    for (t, cm, names) in results:
        if cm is None:
            continue
        nm = _canon_names(list(names))
        pretty = title_map.get(t, t)  # fallback: keep t unchanged
        data.append((pretty, cm, nm))

    if not data:
        return

    n = len(data)
    vmax = max(int(cm.max()) for _, cm, _ in data) or 1
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.0), constrained_layout=True)
    if n == 1:
        axes = [axes]
    ims = []
    for ax, (title, cm, names) in zip(axes, data):
        im = ax.imshow(cm, vmin=0, vmax=vmax, aspect="equal", cmap="Greys"); ims.append(im)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("True", fontsize=12)
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=0, ha="center", fontsize=11)
        ax.set_yticklabels(names, fontsize=11)

        # color tick labels by class color
        for i, name in enumerate(names):
            color = COL.get(name.lower(), "#000000")
            ax.get_xticklabels()[i].set_color(color)
            ax.get_yticklabels()[i].set_color(color)

        # grid + spines
        ax.set_xticks(np.arange(-0.5, cm.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values(): spine.set_visible(False)

        # colored frames on diagonal
        for i, name in enumerate(names):
            color = COL.get(name.lower(), "#000000")
            ax.add_patch(Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                                   edgecolor=color, linewidth=2.2, alpha=0.95))
        # counts
        thresh = vmax * 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = int(cm[i, j])
                txt_color = "white" if cm[i, j] >= thresh else "black"
                ax.text(j, i, f"{val}", ha="center", va="center", fontsize=11, color=txt_color)

    cbar = fig.colorbar(ims[-1], ax=axes, shrink=0.90, pad=0.03)
    cbar.set_label("Count", fontsize=12)
    for t in cbar.ax.get_yticklabels(): t.set_fontsize(11)

    if save_path is not None:
        fig.savefig(save_path, dpi=450, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved confusion matrices to: {save_path}")
    else:
        plt.show()


# ====== Utils ======
def set_seed(seed=SEED):
    import random
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def try_load(name: str):
    x_path = FEATDIR / f"X_{name}.npy"
    y_path = FEATDIR / "y_all.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing {x_path} or {y_path}")
    X = np.load(x_path)
    y = np.load(y_path, allow_pickle=True)
    return X, y

def try_load_scatter(name: str):
    x_path = FEATDIR / f"X_{name}.npy"
    y_path = FEATDIR / "y_all_scatterograms.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing {x_path} or {y_path}")
    X = np.load(x_path)
    y = np.load(y_path, allow_pickle=True)
    return X, y

def infer_channels_and_bands(n_features: int) -> Tuple[int, int]:
    """
    Known layouts:
      - SPEC:     F=106        -> (C=1, B=106)
      - POL:      F=3 or 4     -> (C=3 or 4, B=1)
      - SPECPOL3: F=318=3*106  -> (C=3, B=106)
      - SPECPOL4: F=424=4*106  -> (C=4, B=106)
    """
    if n_features == 106:
        return 1, 106
    if n_features in (3, 4):
        return n_features, 1
    if n_features % 106 == 0:
        C = n_features // 106
        if C in (1, 3, 4, 6):
            return C, 106
    for C in (4, 3, 6, 1):
        if n_features % C == 0:
            return C, n_features // C
    return 1, n_features

def reshape_to_CB(x: np.ndarray) -> torch.Tensor:
    C, B = infer_channels_and_bands(x.shape[1])
    return torch.from_numpy(x).float().view(-1, C, B)

from sklearn.preprocessing import StandardScaler

def fold_standardize(X_np, tr_idx, va_idx):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_np[tr_idx])
    Xva = scaler.transform(X_np[va_idx])
    return Xtr, Xva

# ====== Dataset with augmentation ======
class TensorFeatDataset(Dataset):
    def __init__(self, X_cb: torch.Tensor, y_int: np.ndarray, augment=False, is_spec=False):
        self.X = X_cb
        self.y = torch.from_numpy(np.asarray(y_int, dtype=np.int64))
        self.augment = augment
        self.is_spec = is_spec  # Apply spectral noise only to SPEC/SPECPOL
        
    def __len__(self): return self.X.size(0)
    
    def __getitem__(self, i):
        x = self.X[i]
        if self.augment and SPEC_NOISE_STD > 0:
            # Add small Gaussian noise to spectral features
            x = x + torch.randn_like(x) * SPEC_NOISE_STD
        return x, self.y[i]

def to_int_labels(y_raw):
    arr = np.asarray(y_raw)
    if np.issubdtype(arr.dtype, np.number):
        y = arr.astype(int)
        names = [str(k) for k in sorted(np.unique(y))]
        return y, names
    classes = sorted(np.unique(arr).tolist())
    m = {c:i for i,c in enumerate(classes)}
    y = np.array([m[a] for a in arr], dtype=int)
    return y, classes

# ====== Mixup augmentation ======
@torch.no_grad()
def batch_mixup(x, y, alpha: float):
    if alpha <= 0: return x, y, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    xm  = lam * x + (1 - lam) * x[idx]
    return xm, (y, y[idx]), lam

def mixup_criterion(criterion, logits, y_tuple, lam):
    y_a, y_b = y_tuple
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

# ====== Attention Module ======
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):  # x: (N, C, B)
        # Global pooling
        y = x.mean(dim=2)  # (N, C)
        w = self.fc(y).unsqueeze(-1)  # (N, C, 1)
        return x * w

class DSConvBlock(nn.Module):
    """Depthwise-separable 1D conv block with residual, GN, and SiLU."""
    def __init__(self, c_in, c_out, k=5, p=2, stride=1, groups=None, drop=0.1):
        super().__init__()
        groups = groups or c_in  # depthwise
        self.dw = nn.Conv1d(c_in, c_in, kernel_size=k, padding=p, stride=stride,
                            groups=groups, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(8, c_in), num_channels=c_in)
        self.pw = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.dw(x)
        h = self.gn1(h); h = self.act(h)
        h = self.pw(h)
        h = self.gn2(h); h = self.drop(h); h = self.act(h)
        return h + self.proj(x)

class IdenticalEncoderNet(nn.Module):
    """
    Same encoder for SPEC, POL, SPECPOL:
    - Interpolate sequence length to L=128
    - Residual DS-Conv blocks x3
    - GAP+GMP -> FC
    """
    def __init__(self, in_ch: int, num_classes: int, B: int,
                 base: int = 64, L_fixed: int = 128, p_drop: float = 0.25):
        super().__init__()
        self.L_fixed = L_fixed

        # simple stem to lift channel dim a bit
        self.stem = nn.Conv1d(in_ch, base, kernel_size=1, bias=False)
        self.gn0  = nn.GroupNorm(num_groups=min(8, base), num_channels=base)
        self.act  = nn.SiLU(inplace=True)

        self.block1 = DSConvBlock(base,   base*2, k=7, p=3, drop=p_drop)
        self.block2 = DSConvBlock(base*2, base*3, k=5, p=2, drop=p_drop)
        self.block3 = DSConvBlock(base*3, base*3, k=3, p=1, drop=p_drop)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.out_dim = (base*3)*2
        self.fc  = nn.Linear(self.out_dim, num_classes)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, x):  # x: (N, C, B)
        # unify sequence length for ALL modalities
        x = F.interpolate(x, size=self.L_fixed, mode="linear", align_corners=False)

        x = self.act(self.gn0(self.stem(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x_avg = self.gap(x).squeeze(-1)
        x_max = self.gmp(x).squeeze(-1)
        emb   = torch.cat([x_avg, x_max], dim=1)
        return self.fc(emb)



# ====== Training function ======
def run_fold(X_np, y_np, spec_name="SPEC"):
    y_int, class_names = to_int_labels(y_np)
    C, B = infer_channels_and_bands(X_np.shape[1])
    print(f"[{spec_name}] Parsed features as C={C}, B={B} (F={X_np.shape[1]})")
    
    is_spec = (B > 1)  # True for SPEC/SPECPOL, False for POL

    X_all = reshape_to_CB(X_np)  # (N,C,B)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    accs, f1s = [], []
    oof = np.full_like(y_int, -1)
    cm_total = np.zeros((len(class_names), len(class_names)), dtype=int)

    for fold, (tr, va) in enumerate(skf.split(X_np, y_int), 1):
        # BEFORE: Xtr, Xva = X_all[tr].clone(), X_all[va].clone()
        Xtr_np, Xva_np = fold_standardize(X_np, tr, va)
        Xtr,   Xva     = reshape_to_CB(Xtr_np), reshape_to_CB(Xva_np)
        ytr, yva = y_int[tr], y_int[va]

        ds_tr = TensorFeatDataset(Xtr, ytr, augment=True, is_spec=is_spec)
        ds_va = TensorFeatDataset(Xva, yva, augment=False, is_spec=is_spec)

        # Balanced sampling
        counts  = np.bincount(ytr, minlength=len(class_names))
        weights = torch.DoubleTensor([1.0 / max(counts[c], 1) for c in ytr])
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        dl_tr = DataLoader(ds_tr, batch_size=BATCH, sampler=sampler, 
                          pin_memory=(DEVICE=="cuda"), num_workers=0)
        dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, 
                          pin_memory=(DEVICE=="cuda"), num_workers=0)

        # Create model
        model = IdenticalEncoderNet(C, len(class_names), B, base=48).to(DEVICE)
        
        # Loss with class weights
        inv = 1.0 / np.clip(counts, 1, None)
        alpha = torch.tensor(inv / inv.sum(), dtype=torch.float, device=DEVICE)
        criterion = nn.CrossEntropyLoss(weight=alpha, label_smoothing=0.1)

        opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
        # After creating opt
        warmup_epochs = 5
        total_epochs  = EPOCHS
        steps_per_epoch = max(1, len(dl_tr))
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps  = total_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)  # linear warmup
            # cosine decay after warmup
            progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        
        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

        # Mixup only for SPEC/SPECPOL
        use_mixup = (MIXUP_ALPHA > 0)   # instead of: is_spec and MIXUP_ALPHA > 0


        best_metric, best_state, bad = -1.0, None, 0

        # ===================== Training loop =====================
        for ep in range(1, EPOCHS + 1):
            model.train()
            for it, (xb, yb) in enumerate(dl_tr, 1):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                # Apply mixup for SPEC/SPECPOL
                if use_mixup and np.random.rand() < 0.5:
                    xb, y_tuple, lam = batch_mixup(xb, yb, MIXUP_ALPHA)
                else:
                    y_tuple, lam = None, None

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = model(xb)
                    if y_tuple is not None:
                        loss = mixup_criterion(criterion, logits, y_tuple, lam)
                    else:
                        loss = criterion(logits, yb)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step(ep - 1 + it / steps_per_epoch)

            # ---- Validation ----
            model.eval()
            preds, ys = [], []
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb = xb.to(DEVICE)
                    logits = model(xb)
                    preds.append(logits.argmax(dim=1).cpu().numpy())
                    ys.append(yb.numpy())
            ypred = np.concatenate(preds)
            ytrue = np.concatenate(ys)
            acc = accuracy_score(ytrue, ypred)
            f1m = f1_score(ytrue, ypred, average="macro")

            if f1m > best_metric + 1e-4:
                best_metric = f1m
                bad = 0
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break

        # ================= Load best model =================
        if best_state is not None:
            model.load_state_dict(best_state)

        # ===================== SWA (optional refinement) =====================
        swa_model = AveragedModel(model)
        swa_opt = AdamW(model.parameters(), lr=SWA_LR, weight_decay=WD)

        for _ in range(SWA_EPOCHS):
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                swa_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                swa_opt.step()
            swa_model.update_parameters(model)

            # build a *clean* dataloader (no noise/mixup) for BN
        ds_tr_clean = TensorFeatDataset(Xtr, ytr, augment=False, is_spec=is_spec)
        dl_tr_clean = DataLoader(ds_tr_clean, batch_size=BATCH, shuffle=False,
                                pin_memory=(DEVICE=="cuda"), num_workers=0)

        model.load_state_dict(swa_model.module.state_dict())
        update_bn(dl_tr_clean, model, device=DEVICE)   # <--- refresh BN running stats
        model.eval()
        swa_preds, swa_ys = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(DEVICE)
                logits = model(xb)
                swa_preds.append(logits.argmax(dim=1).cpu().numpy())
                swa_ys.append(yb.numpy())
        swa_pred = np.concatenate(swa_preds)
        ytrue = np.concatenate(swa_ys)
        swa_acc = accuracy_score(ytrue, swa_pred)
        swa_f1 = f1_score(ytrue, swa_pred, average="macro")

        # Use SWA if better, otherwise use best checkpoint
        if swa_f1 >= best_metric:
            final_acc, final_f1, final_pred = swa_acc, swa_f1, swa_pred
        else:
            model.load_state_dict(best_state)
            model.eval()
            preds = []
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb = xb.to(DEVICE)
                    logits = model(xb)
                    preds.append(logits.argmax(dim=1).cpu().numpy())
            final_pred = np.concatenate(preds)
            final_acc = accuracy_score(ytrue, final_pred)
            final_f1 = f1_score(ytrue, final_pred, average="macro")

        accs.append(final_acc)
        f1s.append(final_f1)
        oof[va] = final_pred
        print(f"[{spec_name}] Fold {fold}: Acc={final_acc:.3f} | Macro-F1={final_f1:.3f}")
        cm = confusion_matrix(ytrue, final_pred, labels=np.arange(len(class_names)))
        cm_total += cm

    print(f"\n[{spec_name}] OOF: Acc={np.mean(accs):.3f}±{np.std(accs, ddof=1):.3f} "
          f"| Macro-F1={np.mean(f1s):.3f}±{np.std(f1s, ddof=1):.3f}")
    print(f"Class names: {class_names}")
    print("Confusion matrix (OOF):\n", cm_total)
    return cm_total, class_names

# ====== Main ======
def main():
    set_seed()
    
    # Load data
    Xspec, y = try_load("spec_all")
    try:
        Xpol, _ = try_load("pol4_all")
    except FileNotFoundError:
        Xpol, _ = try_load("pol3_all")
    try:
        Xsp, _  = try_load("specpol4_all")
    except FileNotFoundError:
        Xsp, _  = try_load("specpol3_all")

    print("="*80)
    print("Starting Textile Classification with Improved Architecture")
    print("="*80)

    cm_spec, names_spec = run_fold(Xspec, y, spec_name="SPEC")
    print("\n" + "="*80 + "\n")
    cm_pol,  names_pol  = run_fold(Xpol,  y, spec_name="POL")
    print("\n" + "="*80 + "\n")
    cm_sp,   names_sp   = run_fold(Xsp,   y, spec_name="SPECPOL")


    # Build the results tuple and plot
    results = [
        ("SPEC",    cm_spec, names_spec),
        ("POL",     cm_pol,  names_pol),
        ("SPECPOL", cm_sp,   names_sp),
    ]
    _plot_three_confusions(results, save_path="Textile_Triple_CM.png")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()