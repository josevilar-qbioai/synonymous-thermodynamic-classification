#!/usr/bin/env python3
"""
09_leave_one_gene_out.py
=========================
Leave-One-Gene-Out (LOGO) validation for synonymous variant classification.

The most rigorous test of generalization: for each gene with ≥3 pathogenic
variants, train on ALL other genes and predict the held-out gene.

This answers: "Can the model classify synonymous variants in a gene it
has NEVER seen during training?" — the key publishable claim.

Design for speed on Mac M4:
  - Subsample benign in training set (default 1:20 ratio)
  - Only evaluate genes with ≥3 pathogenic (54 genes from Strategy B)
  - Train for fewer epochs (30) since we have many iterations
  - MPS (Metal) device support for Apple Silicon

Input:
  data/tensors_expanded_synonymous.pt
  data/labels_expanded_synonymous.pt
  data/expanded_gene_map.csv
  data/strategy_b_gene_aucs.csv (for comparison)

Output:
  data/logo_results.csv
  figures/logo_validation.png

Usage:
  python scripts/09_leave_one_gene_out.py
  python scripts/09_leave_one_gene_out.py --min-pathogenic 5
  python scripts/09_leave_one_gene_out.py --subsample 10 --epochs 40
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

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# EnergyFingerprint core
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from core.model import EnergySignalCNN, VariantDataset, EarlyStopping

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "logo"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Training helpers ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def predict(model, X_tensor, device, batch_size=512):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            logits = model(batch)
            p = F.softmax(logits, dim=1)[:, 1]
            probs.extend(p.cpu().numpy())
    return np.array(probs)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Leave-One-Gene-Out validation")
    parser.add_argument("--min-pathogenic", type=int, default=3,
                        help="Min pathogenic variants to test a gene (default: 3)")
    parser.add_argument("--subsample", type=int, default=20,
                        help="Max benign:pathogenic ratio in training (default: 20)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs per gene (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("  LEAVE-ONE-GENE-OUT (LOGO) VALIDATION")
    print("  Synonymous Variant Classification")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs} | LR: {args.lr} | Subsample: 1:{args.subsample}")

    # ── Load data ──────────────────────────────────────────────────
    tensors = torch.load(DATA_DIR / "tensors_expanded_synonymous.pt",
                         map_location="cpu", weights_only=True)
    labels = torch.load(DATA_DIR / "labels_expanded_synonymous.pt",
                        map_location="cpu", weights_only=True)
    gene_map = pd.read_csv(DATA_DIR / "expanded_gene_map.csv")

    # Transpose: (N, 9, 128) → (N, 128, 9) for model
    tensors = tensors.permute(0, 2, 1)
    n_channels = tensors.shape[2]

    genes = gene_map["gene"].values
    y = labels.numpy()

    print(f"\n  Total: {len(y)} variants (P={y.sum()}, B={(y==0).sum()})")
    print(f"  Genes: {len(np.unique(genes))}")

    # ── Identify testable genes ────────────────────────────────────
    gene_counts = pd.DataFrame({"gene": genes, "label": y})
    gene_p_counts = gene_counts[gene_counts["label"] == 1].groupby("gene").size()
    test_genes = gene_p_counts[gene_p_counts >= args.min_pathogenic].index.tolist()
    test_genes = sorted(test_genes)

    print(f"  Testable genes (≥{args.min_pathogenic} P): {len(test_genes)}")

    # Load Strategy B 5-fold results for comparison
    cv_aucs = {}
    cv_path = DATA_DIR / "strategy_b_gene_aucs.csv"
    if cv_path.exists():
        cv_df = pd.read_csv(cv_path)
        cv_aucs = dict(zip(cv_df["gene"], cv_df["auc"]))

    # ── LOGO loop ──────────────────────────────────────────────────
    results = []
    t_total = time.time()

    for gi, test_gene in enumerate(test_genes):
        t0 = time.time()

        # Split: test = this gene, train = all others
        test_mask = genes == test_gene
        train_mask = ~test_mask

        X_test = tensors[test_mask]
        y_test = y[test_mask]
        X_train_full = tensors[train_mask]
        y_train_full = y[train_mask]

        n_test_p = (y_test == 1).sum()
        n_test_b = (y_test == 0).sum()
        n_train_p = (y_train_full == 1).sum()
        n_train_b = (y_train_full == 0).sum()

        # Subsample benign in training
        if args.subsample > 0 and n_train_b > n_train_p * args.subsample:
            max_b = n_train_p * args.subsample
            pos_idx = np.where(y_train_full == 1)[0]
            neg_idx = np.where(y_train_full == 0)[0]
            rng = np.random.RandomState(42 + gi)
            neg_keep = rng.choice(neg_idx, size=max_b, replace=False)
            keep = np.sort(np.concatenate([pos_idx, neg_keep]))
            X_train = X_train_full[keep]
            y_train = y_train_full[keep]
        else:
            X_train = X_train_full
            y_train = y_train_full

        n_train_p_sub = (y_train == 1).sum()
        n_train_b_sub = (y_train == 0).sum()

        # Class weights
        w_pos = n_train_b_sub / max(n_train_p_sub, 1)
        class_weights = torch.tensor([1.0, w_pos], dtype=torch.float32).to(device)

        # Dataset + loader
        train_ds = VariantDataset(X_train.numpy(), y_train, augment=True)
        sample_w = np.where(y_train == 1, w_pos, 1.0)
        sampler = WeightedRandomSampler(torch.FloatTensor(sample_w),
                                         num_samples=len(y_train), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   sampler=sampler, num_workers=0)

        # Model
        model = EnergySignalCNN(n_channels=n_channels, n_classes=2,
                                 dropout=args.dropout, n_heads=4).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Train
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(args.epochs):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            if loss < best_loss - 1e-4:
                best_loss = loss
                patience_counter = 0
                # Save best
                torch.save(model.state_dict(), MODELS_DIR / f"logo_{test_gene}.pt")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break

        # Load best and predict
        model.load_state_dict(torch.load(MODELS_DIR / f"logo_{test_gene}.pt",
                                          map_location=device, weights_only=True))
        test_probs = predict(model, X_test, device)

        # Metrics
        if n_test_p > 0 and n_test_b > 0:
            auc = roc_auc_score(y_test, test_probs)
            ap = average_precision_score(y_test, test_probs)
        else:
            auc = ap = 0.0

        elapsed = time.time() - t0
        cv_auc = cv_aucs.get(test_gene, None)
        cv_str = f"{cv_auc:.3f}" if cv_auc else "  -  "
        delta = f"{auc - cv_auc:+.3f}" if cv_auc else "  -  "

        marker = ""
        if auc >= 0.8:
            marker = " ★★"
        elif auc >= 0.65:
            marker = " ★"
        elif auc < 0.5:
            marker = " ✗"

        print(f"  [{gi+1:2d}/{len(test_genes)}] {test_gene:<12} "
              f"LOGO={auc:.3f}  5CV={cv_str}  Δ={delta}  "
              f"(P={n_test_p}, B={n_test_b}) {elapsed:.0f}s{marker}")

        results.append({
            "gene": test_gene,
            "n_p": int(n_test_p),
            "n_b": int(n_test_b),
            "logo_auc": auc,
            "logo_ap": ap,
            "cv_auc": cv_auc,
            "delta_auc": auc - cv_auc if cv_auc else None,
            "train_p": int(n_train_p_sub),
            "train_b": int(n_train_b_sub),
            "time_s": elapsed,
        })

    # ── Results ────────────────────────────────────────────────────
    total_time = time.time() - t_total
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("logo_auc", ascending=False)

    mean_logo = results_df["logo_auc"].mean()
    std_logo = results_df["logo_auc"].std()
    mean_cv = results_df["cv_auc"].dropna().mean()
    median_logo = results_df["logo_auc"].median()
    n_above_05 = (results_df["logo_auc"] > 0.5).sum()
    n_above_065 = (results_df["logo_auc"] > 0.65).sum()
    n_above_08 = (results_df["logo_auc"] > 0.8).sum()

    print(f"\n{'=' * 70}")
    print(f"  LOGO RESULTS")
    print(f"{'=' * 70}")
    print(f"  Genes tested:     {len(results_df)}")
    print(f"  Total time:       {total_time/60:.1f} min")
    print(f"\n  LOGO AUC:         {mean_logo:.3f} ± {std_logo:.3f} (mean ± std)")
    print(f"  LOGO median:      {median_logo:.3f}")
    print(f"  5-fold CV AUC:    {mean_cv:.3f} (for comparison)")
    print(f"\n  Genes > 0.80:     {n_above_08} / {len(results_df)}")
    print(f"  Genes > 0.65:     {n_above_065} / {len(results_df)}")
    print(f"  Genes > 0.50:     {n_above_05} / {len(results_df)}")
    print(f"  Genes < 0.50:     {len(results_df) - n_above_05} / {len(results_df)}")

    # Comparison
    if "delta_auc" in results_df.columns:
        valid_delta = results_df["delta_auc"].dropna()
        if len(valid_delta) > 0:
            mean_delta = valid_delta.mean()
            print(f"\n  Mean Δ(LOGO - 5CV): {mean_delta:+.3f}")
            n_logo_better = (valid_delta > 0).sum()
            print(f"  LOGO > 5CV: {n_logo_better}/{len(valid_delta)} genes")

    # Save
    results_df.to_csv(DATA_DIR / "logo_results.csv", index=False)
    print(f"\n  Saved: data/logo_results.csv")

    # ── Figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: LOGO AUC bar chart
    ax = axes[0]
    df_sorted = results_df.sort_values("logo_auc", ascending=True)
    colors = ['firebrick' if a < 0.5 else 'orange' if a < 0.65
              else 'steelblue' if a < 0.8 else 'forestgreen'
              for a in df_sorted["logo_auc"]]
    bars = ax.barh(range(len(df_sorted)), df_sorted["logo_auc"], color=colors, height=0.7)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.axvline(x=mean_logo, color='blue', linestyle='-', alpha=0.5,
               label=f'Mean ({mean_logo:.3f})')

    # Label top and bottom genes
    gene_names = df_sorted["gene"].values
    aucs_sorted = df_sorted["logo_auc"].values
    for i in list(range(3)) + list(range(len(df_sorted)-5, len(df_sorted))):
        if 0 <= i < len(df_sorted):
            ax.text(aucs_sorted[i] + 0.01, i, gene_names[i], fontsize=6, va='center')

    ax.set_xlabel("AUC-ROC")
    ax.set_ylabel("Gene rank")
    ax.set_title(f"A. Leave-One-Gene-Out AUC\n(mean = {mean_logo:.3f} ± {std_logo:.3f})")
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0.1, 1.05)

    # Panel B: LOGO vs 5-fold CV scatter
    ax = axes[1]
    valid = results_df.dropna(subset=["cv_auc"])
    ax.scatter(valid["cv_auc"], valid["logo_auc"], s=valid["n_p"] * 12,
               alpha=0.7, c=valid["logo_auc"], cmap="RdYlGn",
               edgecolors='gray', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y = x')

    # Label outliers
    for _, row in valid.nlargest(3, "logo_auc").iterrows():
        ax.annotate(row["gene"], (row["cv_auc"], row["logo_auc"]),
                    fontsize=7, ha='left')
    for _, row in valid.nsmallest(3, "logo_auc").iterrows():
        ax.annotate(row["gene"], (row["cv_auc"], row["logo_auc"]),
                    fontsize=7, ha='left')

    from scipy import stats as sp_stats
    r, p = sp_stats.spearmanr(valid["cv_auc"], valid["logo_auc"])
    ax.text(0.05, 0.95, f"ρ = {r:.3f}\np = {p:.1e}",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel("5-Fold CV AUC")
    ax.set_ylabel("LOGO AUC")
    ax.set_title("B. Cross-validation vs Leave-One-Gene-Out")
    ax.legend(fontsize=8)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.1, 1.05)

    # Panel C: AUC distribution histogram
    ax = axes[2]
    ax.hist(results_df["logo_auc"], bins=15, color='steelblue',
            alpha=0.7, edgecolor='white', label='LOGO')
    if len(valid) > 0:
        ax.hist(valid["cv_auc"], bins=15, color='orange',
                alpha=0.5, edgecolor='white', label='5-fold CV')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=mean_logo, color='blue', linestyle='-', alpha=0.5)
    ax.set_xlabel("AUC-ROC")
    ax.set_ylabel("Number of genes")
    ax.set_title("C. AUC Distribution")
    ax.legend()

    plt.tight_layout()
    fig_path = FIGURES_DIR / "logo_validation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved: {fig_path}")
    plt.close()

    # ── Top / Bottom genes ─────────────────────────────────────────
    print(f"\n  Top 10 genes (LOGO):")
    for _, r in results_df.head(10).iterrows():
        print(f"    {r['gene']:<12} LOGO={r['logo_auc']:.3f}  (P={r['n_p']})")

    print(f"\n  Bottom 5 genes (LOGO):")
    for _, r in results_df.tail(5).iterrows():
        print(f"    {r['gene']:<12} LOGO={r['logo_auc']:.3f}  (P={r['n_p']})")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
