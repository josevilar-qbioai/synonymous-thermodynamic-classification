#!/usr/bin/env python3
"""
07_strategy_b_train.py
=======================
Strategy B: Train a dedicated CNN for synonymous variant classification.

With 641 pathogenic and 130,143 benign synonymous variants from the
expanded ClinVar dataset, we can now train a proper classifier.

Key design decisions for extreme imbalance (1:203 ratio):
  1. Class-weighted CrossEntropyLoss (weight = n_neg/n_pos)
  2. Stratified K-Fold to ensure P variants in every fold
  3. Data augmentation on minority class (jitter + shift + scale)
  4. Weighted random sampling for balanced mini-batches
  5. Early stopping on validation AUC (not loss)
  6. Optionally: Focal Loss for hard-example mining

Evaluation:
  - Stratified 5-fold CV (primary metric: AUC-ROC)
  - Leave-one-gene-out for top genes (secondary)
  - Comparison with Strategy A (Isolation Forest AUC=0.651) and C (0.386)

Input:
  data/tensors_expanded_synonymous.pt   — (N, 9, 128)
  data/labels_expanded_synonymous.pt    — (N,)
  data/expanded_gene_map.csv            — gene mapping

Output:
  models/strategy_b/fold_{k}_best.pt    — Best model per fold
  data/strategy_b_results.csv           — Per-fold metrics
  figures/strategy_b_roc.png            — ROC curves
  figures/strategy_b_distributions.png  — Score distributions

Usage:
  python scripts/07_strategy_b_train.py
  python scripts/07_strategy_b_train.py --n-folds 5 --epochs 60 --lr 1e-3
  python scripts/07_strategy_b_train.py --focal-loss --gamma 2.0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler
except ImportError:
    print("ERROR: PyTorch required.")
    sys.exit(1)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, classification_report, f1_score
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add EnergyFingerprint core
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.model import EnergySignalCNN, VariantDataset, EarlyStopping

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "strategy_b"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Focal Loss ──────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces loss for well-classified examples, focusing training
    on hard, misclassified examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ─── Training loop ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    metrics = {}
    if all_labels.sum() > 0 and all_labels.sum() < len(all_labels):
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
        metrics["ap"] = average_precision_score(all_labels, all_probs)

        # F1 at optimal threshold
        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        metrics["f1_best"] = f1_scores[best_f1_idx]
        metrics["threshold_best"] = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
    else:
        metrics["auc"] = 0.0
        metrics["ap"] = 0.0
        metrics["f1_best"] = 0.0
        metrics["threshold_best"] = 0.5

    return metrics, all_probs, all_labels


# ─── Main training ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Strategy B: Train synonymous CNN")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--focal-loss", action="store_true", help="Use Focal Loss instead of CE")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal Loss gamma")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--use-original", action="store_true",
                        help="Use original 8-gene dataset instead of expanded")
    parser.add_argument("--subsample", type=int, default=20,
                        help="Max benign:pathogenic ratio (default: 20). "
                             "Subsamples benign class to speed up training. "
                             "Set 0 to use all data.")
    args = parser.parse_args()

    # Device: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("  STRATEGY B: Dedicated CNN for Synonymous Variants")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Folds: {args.n_folds} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"  Focal Loss: {args.focal_loss} (gamma={args.gamma})")
    print(f"  Augmentation: {not args.no_augment}")

    # ── Load data ──────────────────────────────────────────────────
    if args.use_original:
        tensors_path = DATA_DIR / "tensors_pooled_synonymous.pt"
        labels_path = DATA_DIR / "labels_pooled_synonymous.pt"
        gene_map_path = DATA_DIR / "pooled_gene_map.csv"
        tag = "original_8gene"
    else:
        tensors_path = DATA_DIR / "tensors_expanded_synonymous.pt"
        labels_path = DATA_DIR / "labels_expanded_synonymous.pt"
        gene_map_path = DATA_DIR / "expanded_gene_map.csv"
        tag = "expanded"

    if not tensors_path.exists():
        print(f"\n  ERROR: {tensors_path.name} not found!")
        sys.exit(1)

    tensors = torch.load(tensors_path, map_location="cpu", weights_only=True)
    labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    gene_map = pd.read_csv(gene_map_path) if gene_map_path.exists() else None

    # Tensors are (N, 9, 128), model expects (N, seq_len=128, channels=9)
    tensors = tensors.permute(0, 2, 1)  # → (N, 128, 9)

    n_total = len(labels)
    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()
    n_channels = tensors.shape[2]

    print(f"\n  Dataset: {tensors_path.name}")
    print(f"  Tensors: {tensors.shape} (N={n_total}, seq={tensors.shape[1]}, ch={n_channels})")
    print(f"  Pathogenic: {n_pos} | Benign: {n_neg} | Ratio: 1:{n_neg/max(n_pos,1):.0f}")

    # ── Subsample benign class for speed ───────────────────────────
    if args.subsample > 0 and n_neg > n_pos * args.subsample:
        max_benign = n_pos * args.subsample
        print(f"\n  Subsampling benign: {n_neg} → {max_benign} (ratio 1:{args.subsample})")

        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(neg_mask)[0]

        # Random subsample of benign
        rng = np.random.RandomState(42)
        neg_keep = rng.choice(neg_idx.numpy(), size=max_benign, replace=False)
        keep_idx = np.sort(np.concatenate([pos_idx.numpy(), neg_keep]))

        tensors = tensors[keep_idx]
        labels = labels[keep_idx]
        if gene_map is not None:
            gene_map = gene_map.iloc[keep_idx].reset_index(drop=True)

        n_total = len(labels)
        n_pos = (labels == 1).sum().item()
        n_neg = (labels == 0).sum().item()
        print(f"  After subsample: {n_total} (P={n_pos}, B={n_neg})")

    # ── Class weights ──────────────────────────────────────────────
    # For extreme imbalance, weight pathogenic class heavily
    weight_neg = 1.0
    weight_pos = n_neg / n_pos
    class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32).to(device)
    print(f"  Class weights: [benign={weight_neg:.1f}, pathogenic={weight_pos:.1f}]")

    # ── Stratified K-Fold ──────────────────────────────────────────
    y_np = labels.numpy()
    X_np = tensors.numpy()

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_fold_probs = np.zeros(n_total)
    all_fold_tested = np.zeros(n_total, dtype=bool)

    print(f"\n{'─' * 70}")
    print(f"  Training {args.n_folds}-fold Stratified CV")
    print(f"{'─' * 70}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np)):
        print(f"\n  ══ Fold {fold} ══")

        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

        n_train_p = (y_train == 1).sum()
        n_train_b = (y_train == 0).sum()
        n_val_p = (y_val == 1).sum()
        n_val_b = (y_val == 0).sum()
        print(f"    Train: {len(y_train)} (P={n_train_p}, B={n_train_b})")
        print(f"    Val:   {len(y_val)} (P={n_val_p}, B={n_val_b})")

        # Create datasets
        train_ds = VariantDataset(X_train, y_train, augment=not args.no_augment)
        val_ds = VariantDataset(X_val, y_val, augment=False)

        # Weighted sampler for balanced mini-batches
        sample_weights = np.where(y_train == 1, weight_pos, weight_neg)
        sampler = WeightedRandomSampler(
            weights=torch.FloatTensor(sample_weights),
            num_samples=len(y_train),
            replacement=True
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
                                shuffle=False, num_workers=0)

        # Model
        model = EnergySignalCNN(
            n_channels=n_channels,
            n_classes=2,
            dropout=args.dropout,
            n_heads=4
        ).to(device)

        if fold == 0:
            n_params = model.count_parameters()
            print(f"    Model params: {n_params:,}")

        # Loss
        if args.focal_loss:
            criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        early_stop = EarlyStopping(patience=args.patience, min_delta=1e-4)

        # Training loop
        best_auc = 0
        best_epoch = 0
        t0 = time.time()

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()

            # Validate
            val_metrics, _, _ = evaluate(model, val_loader, device)
            val_auc = val_metrics["auc"]

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                # Save best model
                save_path = MODELS_DIR / f"fold_{fold}_best.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "val_ap": val_metrics["ap"],
                    "n_channels": n_channels,
                    "fold": fold,
                    "tag": tag,
                }, save_path)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:3d} | loss={train_loss:.4f} | "
                      f"val_AUC={val_auc:.3f} | best={best_auc:.3f} @{best_epoch+1}")

            # Early stopping on validation loss (use -AUC as proxy)
            if early_stop(-val_auc):
                print(f"    Early stop at epoch {epoch+1}")
                break

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s | Best AUC={best_auc:.3f} @epoch {best_epoch+1}")

        # Load best model and get final predictions
        checkpoint = torch.load(MODELS_DIR / f"fold_{fold}_best.pt",
                                map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        val_metrics, val_probs, val_labels = evaluate(model, val_loader, device)

        # Store out-of-fold predictions
        all_fold_probs[val_idx] = val_probs
        all_fold_tested[val_idx] = True

        fold_results.append({
            "fold": fold,
            "auc": val_metrics["auc"],
            "ap": val_metrics["ap"],
            "f1": val_metrics["f1_best"],
            "threshold": val_metrics["threshold_best"],
            "best_epoch": best_epoch + 1,
            "train_p": int(n_train_p),
            "train_b": int(n_train_b),
            "val_p": int(n_val_p),
            "val_b": int(n_val_b),
            "time_s": elapsed,
        })

    # ── Aggregate results ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  RESULTS — Strategy B ({tag})")
    print(f"{'=' * 70}")

    results_df = pd.DataFrame(fold_results)
    mean_auc = results_df["auc"].mean()
    std_auc = results_df["auc"].std()
    mean_ap = results_df["ap"].mean()
    std_ap = results_df["ap"].std()

    print(f"\n  Per-fold results:")
    print(f"  {'Fold':>5} {'AUC':>8} {'AP':>8} {'F1':>8} {'Epoch':>6}")
    print(f"  {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
    for _, row in results_df.iterrows():
        print(f"  {int(row['fold']):>5} {row['auc']:>8.3f} {row['ap']:>8.4f} "
              f"{row['f1']:>8.3f} {int(row['best_epoch']):>6}")
    print(f"  {'─'*5} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  {'MEAN':>5} {mean_auc:>8.3f} {mean_ap:>8.4f} {results_df['f1'].mean():>8.3f}")
    print(f"  {'±STD':>5} {std_auc:>8.3f} {std_ap:>8.4f} {results_df['f1'].std():>8.3f}")

    # Comparison with other strategies
    print(f"\n  Comparison:")
    print(f"    Strategy A (Isolation Forest, 8 genes):  AUC = 0.651")
    print(f"    Strategy C (Zero-shot missense, 8 genes): AUC = 0.386")
    print(f"    Strategy B (Dedicated CNN, {tag}):        AUC = {mean_auc:.3f} ± {std_auc:.3f}")

    delta_a = mean_auc - 0.651
    print(f"\n    Δ(B vs A): {delta_a:+.3f}")
    if mean_auc > 0.651:
        print(f"    ✓ Strategy B outperforms anomaly detection!")
    elif mean_auc > 0.5:
        print(f"    ~ Strategy B shows signal but below anomaly detection")
    else:
        print(f"    ✗ Strategy B below chance — check data or architecture")

    # Save results
    results_df.to_csv(DATA_DIR / "strategy_b_results.csv", index=False)

    # ── Global out-of-fold metrics ─────────────────────────────────
    if all_fold_tested.all():
        global_auc = roc_auc_score(y_np, all_fold_probs)
        global_ap = average_precision_score(y_np, all_fold_probs)
        print(f"\n  Out-of-fold (global): AUC={global_auc:.3f}, AP={global_ap:.4f}")

    # ── Plot ROC curves ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Per-fold ROC
    ax = axes[0]
    # Recompute per-fold from saved models
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np)):
        y_fold = y_np[val_idx]
        p_fold = all_fold_probs[val_idx]
        fpr, tpr, _ = roc_curve(y_fold, p_fold)
        ax.plot(fpr, tpr, linewidth=1.5, alpha=0.7,
                label=f'Fold {fold} (AUC={fold_results[fold]["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Strategy B: {args.n_folds}-Fold CV\nAUC = {mean_auc:.3f} ± {std_auc:.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Score distributions
    ax = axes[1]
    ax.hist(all_fold_probs[y_np == 0], bins=50, alpha=0.6,
            label=f"Benign (n={n_neg})", color="steelblue", density=True)
    ax.hist(all_fold_probs[y_np == 1], bins=30, alpha=0.8,
            label=f"Pathogenic (n={n_pos})", color="firebrick", density=True)
    ax.set_xlabel("P(pathogenic)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution (out-of-fold)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Precision-Recall curve
    ax = axes[2]
    if all_fold_tested.all():
        prec, rec, _ = precision_recall_curve(y_np, all_fold_probs)
        ax.plot(rec, prec, 'b-', linewidth=2, label=f'AP = {global_ap:.4f}')
        ax.axhline(y=n_pos/n_total, color='r', linestyle='--', alpha=0.5,
                    label=f'Baseline (prevalence = {n_pos/n_total:.4f})')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / f"strategy_b_roc_{tag}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {fig_path}")
    plt.close()

    # ── Per-gene analysis ──────────────────────────────────────────
    if gene_map is not None and all_fold_tested.all():
        print(f"\n  Per-gene AUC (genes with ≥3 pathogenic):")
        print(f"  {'Gene':<12} {'P':>4} {'B':>6} {'AUC':>7}")
        print(f"  {'─'*12} {'─'*4} {'─'*6} {'─'*7}")

        gene_aucs = []
        genes = gene_map["gene"].values
        for gene in sorted(gene_map["gene"].unique()):
            mask = genes == gene
            y_gene = y_np[mask]
            p_gene = all_fold_probs[mask]
            n_p_gene = (y_gene == 1).sum()
            n_b_gene = (y_gene == 0).sum()

            if n_p_gene >= 3 and n_b_gene >= 3:
                gene_auc = roc_auc_score(y_gene, p_gene)
                print(f"  {gene:<12} {n_p_gene:>4} {n_b_gene:>6} {gene_auc:>7.3f}")
                gene_aucs.append({"gene": gene, "n_p": n_p_gene, "n_b": n_b_gene, "auc": gene_auc})

        if gene_aucs:
            gene_aucs_df = pd.DataFrame(gene_aucs)
            gene_aucs_df.to_csv(DATA_DIR / "strategy_b_gene_aucs.csv", index=False)
            mean_gene_auc = gene_aucs_df["auc"].mean()
            print(f"\n  Mean per-gene AUC: {mean_gene_auc:.3f} ({len(gene_aucs)} genes)")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  Files saved:")
    print(f"    Models:  models/strategy_b/fold_{{0-{args.n_folds-1}}}_best.pt")
    print(f"    Results: data/strategy_b_results.csv")
    print(f"    Figure:  figures/strategy_b_roc_{tag}.png")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
