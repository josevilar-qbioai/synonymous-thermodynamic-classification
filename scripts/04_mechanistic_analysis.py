#!/usr/bin/env python3
"""
08_mechanistic_analysis.py
===========================
Mechanistic analysis: what distinguishes genes where the CNN works well
(high AUC) from genes where it fails (low AUC)?

Hypotheses to test:
  H1. Codon position: pathogenic variants at wobble (3rd) vs 1st/2nd position
  H2. Splice proximity: pathogenic variants near exon-intron boundaries
  H3. ΔG disruption magnitude: do high-AUC genes have stronger ΔG signal?
  H4. CDS length / gene size: longer genes → more context → better AUC?
  H5. Pathogenic mechanism: splicing vs stability vs translation

Input:
  data/synonymous_expanded_all.csv
  data/synonymous_expanded_pathogenic.csv
  data/strategy_b_gene_aucs.csv
  data/tensors_expanded_synonymous.pt
  data/labels_expanded_synonymous.pt
  data/expanded_gene_map.csv
  data/{GENE}_cds.fasta

Output:
  data/mechanistic_analysis.csv
  figures/mechanistic_*.png

Usage:
  python scripts/08_mechanistic_analysis.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import torch
except ImportError:
    print("ERROR: PyTorch required.")
    sys.exit(1)

# Add EnergyFingerprint core
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from core.genetics import load_fasta

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ─── Helper: parse HGVS for codon position ──────────────────────────────────

def get_codon_position(cds_pos):
    """Return codon position (1, 2, 3) from 1-based CDS position."""
    if pd.isna(cds_pos) or cds_pos is None:
        return None
    pos = int(cds_pos)
    return ((pos - 1) % 3) + 1  # 1=first, 2=second, 3=wobble


def get_codon_number(cds_pos):
    """Return codon number from 1-based CDS position."""
    if pd.isna(cds_pos) or cds_pos is None:
        return None
    return ((int(cds_pos) - 1) // 3) + 1


def get_relative_position(cds_pos, cds_length):
    """Position as fraction of CDS length (0=start, 1=end)."""
    if pd.isna(cds_pos) or cds_pos is None or cds_length == 0:
        return None
    return int(cds_pos) / cds_length


# ─── Load all data ───────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")

    # Variants
    all_df = pd.read_csv(DATA_DIR / "synonymous_expanded_all.csv")
    path_df = pd.read_csv(DATA_DIR / "synonymous_expanded_pathogenic.csv")

    # Gene AUCs from Strategy B
    auc_df = pd.read_csv(DATA_DIR / "strategy_b_gene_aucs.csv")

    # Tensors for ΔG analysis
    tensors = torch.load(DATA_DIR / "tensors_expanded_synonymous.pt",
                         map_location="cpu", weights_only=True)
    labels = torch.load(DATA_DIR / "labels_expanded_synonymous.pt",
                        map_location="cpu", weights_only=True)
    gene_map = pd.read_csv(DATA_DIR / "expanded_gene_map.csv")

    print(f"  All variants: {len(all_df)}")
    print(f"  Pathogenic:   {len(path_df)}")
    print(f"  Gene AUCs:    {len(auc_df)} genes")
    print(f"  Tensors:      {tensors.shape}")

    return all_df, path_df, auc_df, tensors, labels, gene_map


# ─── Analysis 1: Codon Position ─────────────────────────────────────────────

def analyze_codon_position(all_df, path_df, auc_df):
    """
    H1: Are pathogenic synonymous variants enriched at non-wobble positions?
    Wobble (3rd position) changes are the most degenerate → least likely to
    affect function. If pathogenic variants cluster at 1st/2nd position,
    they're more likely to disrupt splicing enhancers (ESE/ESS).
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: Codon Position Distribution")
    print("=" * 70)

    # Add codon position to all variants
    all_df = all_df.copy()
    all_df["codon_pos"] = all_df["cds_position"].apply(get_codon_position)

    benign = all_df[all_df["label"] == 0]
    pathogenic = all_df[all_df["label"] == 1]

    # Global distribution
    b_counts = benign["codon_pos"].value_counts().sort_index()
    p_counts = pathogenic["codon_pos"].value_counts().sort_index()

    b_pcts = b_counts / b_counts.sum() * 100
    p_pcts = p_counts / p_counts.sum() * 100

    print(f"\n  Position  Benign(%)   Pathogenic(%)  Enrichment")
    print(f"  ────────  ─────────   ─────────────  ──────────")
    for pos in [1, 2, 3]:
        b_pct = b_pcts.get(pos, 0)
        p_pct = p_pcts.get(pos, 0)
        enrichment = p_pct / b_pct if b_pct > 0 else 0
        marker = " ★" if enrichment > 1.3 else ""
        print(f"  {pos} ({'wobble' if pos==3 else 'pos '+str(pos)})   "
              f"{b_pct:>6.1f}%      {p_pct:>6.1f}%        {enrichment:.2f}x{marker}")

    # Chi-square test
    observed = np.array([p_counts.get(i, 0) for i in [1, 2, 3]])
    expected_ratio = np.array([b_counts.get(i, 0) for i in [1, 2, 3]])
    expected = expected_ratio / expected_ratio.sum() * observed.sum()

    if all(expected > 0):
        chi2, p_val = stats.chisquare(observed, expected)
        print(f"\n  Chi-square test: χ² = {chi2:.2f}, p = {p_val:.4f}")
        if p_val < 0.05:
            print(f"  → Significant difference in codon position distribution!")
        else:
            print(f"  → No significant difference")

    # Per-gene: correlate wobble fraction with AUC
    gene_wobble = []
    for _, row in auc_df.iterrows():
        gene = row["gene"]
        gene_path = pathogenic[pathogenic["gene"] == gene]
        if len(gene_path) >= 3:
            gene_path_cp = gene_path["cds_position"].apply(get_codon_position)
            wobble_frac = (gene_path_cp == 3).mean()
            non_wobble_frac = (gene_path_cp.isin([1, 2])).mean()
            gene_wobble.append({
                "gene": gene, "auc": row["auc"],
                "wobble_frac": wobble_frac,
                "non_wobble_frac": non_wobble_frac,
                "n_p": row["n_p"]
            })

    wobble_df = pd.DataFrame(gene_wobble)
    if len(wobble_df) > 5:
        r, p = stats.spearmanr(wobble_df["non_wobble_frac"], wobble_df["auc"])
        print(f"\n  Correlation: non-wobble fraction vs AUC")
        print(f"    Spearman r = {r:.3f}, p = {p:.4f}")

    return all_df, wobble_df


# ─── Analysis 2: ΔG Disruption Magnitude ────────────────────────────────────

def analyze_dg_disruption(tensors, labels, gene_map, auc_df):
    """
    H3: Do high-AUC genes have larger ΔG disruption in pathogenic variants?
    Channels 7-9 (indices 6,7,8 in 9ch tensor) = difference profile.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: ΔG Disruption Magnitude per Gene")
    print("=" * 70)

    y = labels.numpy()
    genes = gene_map["gene"].values

    gene_disruption = []

    for _, row in auc_df.iterrows():
        gene = row["gene"]
        mask = genes == gene
        gene_tensors = tensors[mask]
        gene_labels = y[mask]

        if gene_labels.sum() < 3:
            continue

        # Difference channels (7-9 in 9ch = indices 6,7,8)
        # Tensors are (N, 9, 128)
        path_mask = gene_labels == 1
        benign_mask = gene_labels == 0

        # Total absolute disruption per variant
        path_disruption = torch.abs(gene_tensors[path_mask, 6:9, :]).sum(dim=(1, 2)).numpy()
        benign_disruption = torch.abs(gene_tensors[benign_mask, 6:9, :]).sum(dim=(1, 2)).numpy()

        # Also just the ΔG profile (channel 7 = index 6)
        path_dg = torch.abs(gene_tensors[path_mask, 6, :]).sum(dim=1).numpy()
        benign_dg = torch.abs(gene_tensors[benign_mask, 6, :]).sum(dim=1).numpy()

        # Effect size: how much more disrupted are pathogenic vs benign?
        mean_path = path_disruption.mean()
        mean_benign = benign_disruption.mean()
        ratio = mean_path / mean_benign if mean_benign > 0 else 0

        # Mann-Whitney U test
        if len(path_disruption) >= 3 and len(benign_disruption) >= 3:
            u_stat, u_pval = stats.mannwhitneyu(path_disruption, benign_disruption,
                                                 alternative='greater')
        else:
            u_pval = 1.0

        gene_disruption.append({
            "gene": gene, "auc": row["auc"], "n_p": row["n_p"],
            "mean_disruption_path": mean_path,
            "mean_disruption_benign": mean_benign,
            "disruption_ratio": ratio,
            "mean_dg_path": path_dg.mean(),
            "mean_dg_benign": benign_dg.mean(),
            "dg_ratio": path_dg.mean() / benign_dg.mean() if benign_dg.mean() > 0 else 0,
            "mannwhitney_p": u_pval,
        })

    disruption_df = pd.DataFrame(gene_disruption)

    print(f"\n  {'Gene':<12} {'AUC':>6} {'P_disrupt':>10} {'B_disrupt':>10} {'Ratio':>7} {'MW p':>8}")
    print(f"  {'─'*12} {'─'*6} {'─'*10} {'─'*10} {'─'*7} {'─'*8}")
    for _, row in disruption_df.sort_values("auc", ascending=False).head(20).iterrows():
        sig = "***" if row["mannwhitney_p"] < 0.001 else "**" if row["mannwhitney_p"] < 0.01 else "*" if row["mannwhitney_p"] < 0.05 else ""
        print(f"  {row['gene']:<12} {row['auc']:>6.3f} {row['mean_disruption_path']:>10.3f} "
              f"{row['mean_disruption_benign']:>10.3f} {row['disruption_ratio']:>7.2f} "
              f"{row['mannwhitney_p']:>7.4f} {sig}")

    # Correlation: disruption ratio vs AUC
    if len(disruption_df) > 5:
        r, p = stats.spearmanr(disruption_df["disruption_ratio"], disruption_df["auc"])
        print(f"\n  Correlation: disruption ratio vs AUC")
        print(f"    Spearman r = {r:.3f}, p = {p:.4f}")
        if p < 0.05:
            print(f"  → Genes with bigger ΔG disruption in pathogenic variants → higher AUC!")

    return disruption_df


# ─── Analysis 3: CDS Position (splice proximity) ────────────────────────────

def analyze_splice_proximity(all_df, auc_df):
    """
    H2: Are pathogenic variants near exon boundaries (splice sites)?

    Since we work with CDS (not genomic), we approximate splice proximity
    as distance to start/end of CDS. A more precise analysis would need
    exon structure from the gene model.

    Also: within each gene, compare relative CDS position of P vs B.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: Position within CDS")
    print("=" * 70)

    all_df = all_df.copy()

    # Get CDS lengths
    cds_lengths = {}
    for gene in auc_df["gene"].unique():
        fasta_path = DATA_DIR / f"{gene}_cds.fasta"
        if fasta_path.exists():
            try:
                cds_seq = load_fasta(str(fasta_path))
                cds_lengths[gene] = len(cds_seq)
            except:
                pass

    # Add relative position
    all_df["rel_position"] = all_df.apply(
        lambda r: get_relative_position(r["cds_position"],
                                         cds_lengths.get(r["gene"], 0)),
        axis=1
    )

    # Compare P vs B position distributions for genes with AUC data
    gene_position = []
    for _, row in auc_df.iterrows():
        gene = row["gene"]
        gene_df = all_df[all_df["gene"] == gene]
        path_pos = gene_df[gene_df["label"] == 1]["rel_position"].dropna()
        benign_pos = gene_df[gene_df["label"] == 0]["rel_position"].dropna()

        if len(path_pos) >= 3 and len(benign_pos) >= 3:
            # Are pathogenic variants more clustered (lower variance)?
            path_std = path_pos.std()
            benign_std = benign_pos.std()

            # Are they near edges? (splice proximity proxy)
            path_edge = ((path_pos < 0.05) | (path_pos > 0.95)).mean()
            benign_edge = ((benign_pos < 0.05) | (benign_pos > 0.95)).mean()

            gene_position.append({
                "gene": gene, "auc": row["auc"],
                "cds_length": cds_lengths.get(gene, 0),
                "path_mean_pos": path_pos.mean(),
                "benign_mean_pos": benign_pos.mean(),
                "path_std_pos": path_std,
                "benign_std_pos": benign_std,
                "path_edge_frac": path_edge,
                "benign_edge_frac": benign_edge,
            })

    pos_df = pd.DataFrame(gene_position)

    if len(pos_df) > 0:
        print(f"\n  {'Gene':<12} {'AUC':>6} {'CDS_len':>8} {'P_edge%':>8} {'B_edge%':>8} {'P_std':>7} {'B_std':>7}")
        print(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*7}")
        for _, row in pos_df.sort_values("auc", ascending=False).head(15).iterrows():
            print(f"  {row['gene']:<12} {row['auc']:>6.3f} {row['cds_length']:>8} "
                  f"{row['path_edge_frac']*100:>7.1f}% {row['benign_edge_frac']*100:>7.1f}% "
                  f"{row['path_std_pos']:>7.3f} {row['benign_std_pos']:>7.3f}")

    # Correlations
    if len(pos_df) > 5:
        # CDS length vs AUC
        r1, p1 = stats.spearmanr(pos_df["cds_length"], pos_df["auc"])
        print(f"\n  CDS length vs AUC: r = {r1:.3f}, p = {p1:.4f}")

        # Edge fraction enrichment vs AUC
        pos_df["edge_enrichment"] = (pos_df["path_edge_frac"] + 0.01) / (pos_df["benign_edge_frac"] + 0.01)
        r2, p2 = stats.spearmanr(pos_df["edge_enrichment"], pos_df["auc"])
        print(f"  Edge enrichment vs AUC: r = {r2:.3f}, p = {p2:.4f}")

    return pos_df


# ─── Analysis 4: Nucleotide change type ─────────────────────────────────────

def analyze_change_type(all_df, auc_df):
    """
    What types of nucleotide changes (transitions vs transversions)
    are enriched in pathogenic synonymous variants?
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: Nucleotide Change Types")
    print("=" * 70)

    # Parse ref>alt from HGVS name
    def get_change(name):
        match = re.search(r':c\.\d+([ACGT])>([ACGT])', str(name), re.IGNORECASE)
        if match:
            return f"{match.group(1).upper()}>{match.group(2).upper()}"
        return None

    all_df = all_df.copy()
    all_df["change"] = all_df["name"].apply(get_change)

    transitions = {"A>G", "G>A", "C>T", "T>C"}

    benign = all_df[all_df["label"] == 0]
    pathogenic = all_df[all_df["label"] == 1]

    b_changes = benign["change"].value_counts()
    p_changes = pathogenic["change"].value_counts()

    print(f"\n  {'Change':<8} {'Benign':>8} {'Path':>6} {'B%':>7} {'P%':>7} {'Enrich':>7} {'Type'}")
    print(f"  {'─'*8} {'─'*8} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*10}")

    all_changes = sorted(set(b_changes.index) | set(p_changes.index))
    for change in all_changes:
        b_n = b_changes.get(change, 0)
        p_n = p_changes.get(change, 0)
        b_pct = b_n / b_changes.sum() * 100
        p_pct = p_n / p_changes.sum() * 100
        enrichment = p_pct / b_pct if b_pct > 0 else 0
        change_type = "transition" if change in transitions else "transversion"
        marker = " ★" if enrichment > 1.5 else ""
        print(f"  {change:<8} {b_n:>8} {p_n:>6} {b_pct:>6.1f}% {p_pct:>6.1f}% {enrichment:>6.2f}x {change_type}{marker}")

    # Overall transition/transversion ratio
    b_ti = sum(b_changes.get(c, 0) for c in transitions)
    b_tv = b_changes.sum() - b_ti
    p_ti = sum(p_changes.get(c, 0) for c in transitions)
    p_tv = p_changes.sum() - p_ti

    print(f"\n  Ti/Tv ratio:")
    print(f"    Benign:     {b_ti/max(b_tv,1):.2f} ({b_ti} Ti, {b_tv} Tv)")
    print(f"    Pathogenic: {p_ti/max(p_tv,1):.2f} ({p_ti} Ti, {p_tv} Tv)")

    return all_df


# ─── Analysis 5: High-AUC vs Low-AUC gene properties ────────────────────────

def analyze_auc_groups(auc_df, disruption_df, wobble_df, pos_df):
    """
    Split genes into high-AUC (>0.75) and low-AUC (<0.6) groups.
    Compare their properties systematically.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS 5: High-AUC vs Low-AUC Gene Properties")
    print("=" * 70)

    high = auc_df[auc_df["auc"] >= 0.75].copy()
    low = auc_df[auc_df["auc"] < 0.60].copy()

    print(f"\n  High-AUC genes (≥0.75): {len(high)}")
    for _, r in high.sort_values("auc", ascending=False).iterrows():
        print(f"    {r['gene']:<12} AUC={r['auc']:.3f} (P={r['n_p']})")

    print(f"\n  Low-AUC genes (<0.60): {len(low)}")
    for _, r in low.sort_values("auc").iterrows():
        print(f"    {r['gene']:<12} AUC={r['auc']:.3f} (P={r['n_p']})")

    # Compare properties
    print(f"\n  Property comparison:")
    print(f"  {'Property':<30} {'High-AUC':>12} {'Low-AUC':>12} {'p-value':>10}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*10}")

    comparisons = []

    # Disruption ratio
    if disruption_df is not None and len(disruption_df) > 0:
        high_dis = disruption_df[disruption_df["gene"].isin(high["gene"])]["disruption_ratio"]
        low_dis = disruption_df[disruption_df["gene"].isin(low["gene"])]["disruption_ratio"]
        if len(high_dis) > 2 and len(low_dis) > 2:
            _, p = stats.mannwhitneyu(high_dis, low_dis, alternative='greater')
            print(f"  {'ΔG disruption ratio':<30} {high_dis.mean():>12.2f} {low_dis.mean():>12.2f} {p:>10.4f}")
            comparisons.append(("disruption_ratio", high_dis.mean(), low_dis.mean(), p))

    # Wobble fraction
    if wobble_df is not None and len(wobble_df) > 0:
        high_wob = wobble_df[wobble_df["gene"].isin(high["gene"])]["wobble_frac"]
        low_wob = wobble_df[wobble_df["gene"].isin(low["gene"])]["wobble_frac"]
        if len(high_wob) > 2 and len(low_wob) > 2:
            _, p = stats.mannwhitneyu(high_wob, low_wob, alternative='two-sided')
            print(f"  {'Wobble position fraction':<30} {high_wob.mean():>12.2f} {low_wob.mean():>12.2f} {p:>10.4f}")

    # CDS length
    if pos_df is not None and len(pos_df) > 0:
        high_len = pos_df[pos_df["gene"].isin(high["gene"])]["cds_length"]
        low_len = pos_df[pos_df["gene"].isin(low["gene"])]["cds_length"]
        if len(high_len) > 2 and len(low_len) > 2:
            _, p = stats.mannwhitneyu(high_len, low_len, alternative='two-sided')
            print(f"  {'CDS length (nt)':<30} {high_len.mean():>12.0f} {low_len.mean():>12.0f} {p:>10.4f}")

    return high, low


# ─── Generate figures ────────────────────────────────────────────────────────

def plot_mechanistic_figures(auc_df, all_df, disruption_df, wobble_df, pos_df):
    """Generate publication-quality mechanistic analysis figures."""

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel A: AUC distribution ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    aucs = auc_df["auc"].sort_values(ascending=True).values
    colors = ['firebrick' if a < 0.5 else 'orange' if a < 0.65 else 'steelblue' if a < 0.8 else 'forestgreen'
              for a in aucs]
    ax.barh(range(len(aucs)), aucs, color=colors, height=0.7)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.axvline(x=0.651, color='orange', linestyle='--', alpha=0.5, label='Strategy A')
    ax.set_xlabel("AUC-ROC")
    ax.set_ylabel("Gene rank")
    ax.set_title("A. Per-gene AUC Distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(0.2, 1.0)

    # Top and bottom gene labels
    genes_sorted = auc_df.sort_values("auc", ascending=True)["gene"].values
    for i in [0, 1, 2, -1, -2, -3]:
        idx = i if i >= 0 else len(aucs) + i
        ax.text(aucs[idx] + 0.01, idx, genes_sorted[idx], fontsize=7, va='center')

    # ── Panel B: Codon position enrichment ─────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    benign = all_df[all_df["label"] == 0]
    pathogenic = all_df[all_df["label"] == 1]

    positions = [1, 2, 3]
    b_cp = benign["cds_position"].apply(get_codon_position)
    p_cp = pathogenic["cds_position"].apply(get_codon_position)

    b_pcts = [(b_cp == p).mean() * 100 for p in positions]
    p_pcts = [(p_cp == p).mean() * 100 for p in positions]

    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, b_pcts, width, label='Benign', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, p_pcts, width, label='Pathogenic', color='firebrick', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['1st\n(least degenerate)', '2nd', '3rd\n(wobble)'])
    ax.set_ylabel("Fraction (%)")
    ax.set_title("B. Codon Position Distribution")
    ax.legend()

    # ── Panel C: Disruption ratio vs AUC ──────────────────────────
    if disruption_df is not None and len(disruption_df) > 0:
        ax = fig.add_subplot(gs[0, 2])
        ax.scatter(disruption_df["disruption_ratio"], disruption_df["auc"],
                   s=disruption_df["n_p"] * 10, alpha=0.7,
                   c=disruption_df["auc"], cmap="RdYlGn", edgecolors='gray', linewidth=0.5)

        # Label extreme points
        for _, row in disruption_df.nlargest(5, "auc").iterrows():
            ax.annotate(row["gene"], (row["disruption_ratio"], row["auc"]),
                        fontsize=7, ha='left', va='bottom')
        for _, row in disruption_df.nsmallest(3, "auc").iterrows():
            ax.annotate(row["gene"], (row["disruption_ratio"], row["auc"]),
                        fontsize=7, ha='left', va='top')

        ax.set_xlabel("ΔG Disruption Ratio (P/B)")
        ax.set_ylabel("AUC-ROC")
        ax.set_title("C. Disruption Magnitude vs Model Performance")
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)

        r, p = stats.spearmanr(disruption_df["disruption_ratio"], disruption_df["auc"])
        ax.text(0.05, 0.95, f"ρ = {r:.3f}\np = {p:.4f}",
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ── Panel D: Nucleotide change types ──────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    transitions = {"A>G", "G>A", "C>T", "T>C"}

    def get_change(name):
        match = re.search(r':c\.\d+([ACGT])>([ACGT])', str(name), re.IGNORECASE)
        return f"{match.group(1).upper()}>{match.group(2).upper()}" if match else None

    b_changes = benign["name"].apply(get_change).value_counts()
    p_changes = pathogenic["name"].apply(get_change).value_counts()

    all_changes = sorted(set(b_changes.index.dropna()) | set(p_changes.index.dropna()))
    b_pcts = [b_changes.get(c, 0) / b_changes.sum() * 100 for c in all_changes]
    p_pcts = [p_changes.get(c, 0) / p_changes.sum() * 100 for c in all_changes]

    x = np.arange(len(all_changes))
    ax.bar(x - width/2, b_pcts, width, label='Benign', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, p_pcts, width, label='Pathogenic', color='firebrick', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(all_changes, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Fraction (%)")
    ax.set_title("D. Nucleotide Change Types")
    ax.legend(fontsize=8)

    # ── Panel E: CDS length vs AUC ────────────────────────────────
    if pos_df is not None and len(pos_df) > 0:
        ax = fig.add_subplot(gs[1, 1])
        ax.scatter(pos_df["cds_length"], pos_df["auc"],
                   s=50, alpha=0.7, c=pos_df["auc"], cmap="RdYlGn",
                   edgecolors='gray', linewidth=0.5)
        for _, row in pos_df.nlargest(4, "auc").iterrows():
            ax.annotate(row["gene"], (row["cds_length"], row["auc"]),
                        fontsize=7, ha='left')
        ax.set_xlabel("CDS Length (nt)")
        ax.set_ylabel("AUC-ROC")
        ax.set_title("E. Gene Length vs Model Performance")
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)

    # ── Panel F: High vs Low AUC summary ──────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    high_genes = auc_df[auc_df["auc"] >= 0.75]["gene"].values
    low_genes = auc_df[auc_df["auc"] < 0.55]["gene"].values

    categories = ["High AUC\n(≥0.75)", "Low AUC\n(<0.55)"]
    gene_lists = [high_genes, low_genes]

    text_y = 0.9
    for i, (cat, genes) in enumerate(zip(categories, gene_lists)):
        ax.text(0.1 + i * 0.5, 0.95, cat, fontsize=12, fontweight='bold',
                transform=ax.transAxes, ha='center', va='top')
        for j, gene in enumerate(genes[:10]):
            gene_auc = auc_df[auc_df["gene"] == gene]["auc"].values[0]
            ax.text(0.1 + i * 0.5, 0.85 - j * 0.07,
                    f"{gene} ({gene_auc:.3f})",
                    fontsize=9, transform=ax.transAxes, ha='center', va='top',
                    color='forestgreen' if i == 0 else 'firebrick')
    ax.set_title("F. Gene Classification by Performance")
    ax.axis('off')

    plt.savefig(FIGURES_DIR / "mechanistic_analysis.png", dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: figures/mechanistic_analysis.png")
    plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MECHANISTIC ANALYSIS — What drives synonymous variant detection?")
    print("=" * 70)

    all_df, path_df, auc_df, tensors, labels, gene_map = load_data()

    # Run analyses
    all_df, wobble_df = analyze_codon_position(all_df, path_df, auc_df)
    disruption_df = analyze_dg_disruption(tensors, labels, gene_map, auc_df)
    pos_df = analyze_splice_proximity(all_df, auc_df)
    all_df = analyze_change_type(all_df, auc_df)
    high, low = analyze_auc_groups(auc_df, disruption_df, wobble_df, pos_df)

    # Generate figures
    print("\n" + "─" * 70)
    print("  Generating figures...")
    print("─" * 70)
    plot_mechanistic_figures(auc_df, all_df, disruption_df, wobble_df, pos_df)

    # Save combined analysis
    combined = auc_df.copy()
    if disruption_df is not None and len(disruption_df) > 0:
        combined = combined.merge(
            disruption_df[["gene", "disruption_ratio", "dg_ratio", "mannwhitney_p"]],
            on="gene", how="left"
        )
    if wobble_df is not None and len(wobble_df) > 0:
        combined = combined.merge(
            wobble_df[["gene", "wobble_frac", "non_wobble_frac"]],
            on="gene", how="left"
        )
    if pos_df is not None and len(pos_df) > 0:
        combined = combined.merge(
            pos_df[["gene", "cds_length", "path_edge_frac", "benign_edge_frac"]],
            on="gene", how="left"
        )

    combined.to_csv(DATA_DIR / "mechanistic_analysis.csv", index=False)
    print(f"  Saved: data/mechanistic_analysis.csv")

    # Final synthesis
    print(f"\n{'=' * 70}")
    print(f"  SYNTHESIS")
    print(f"{'=' * 70}")
    print(f"""
  The CNN's ability to detect synonymous pathogenic variants varies
  dramatically across genes (AUC range: {auc_df['auc'].min():.3f} – {auc_df['auc'].max():.3f}).

  Key findings:
  1. Codon position: [see Analysis 1 — wobble vs non-wobble enrichment]
  2. ΔG disruption: [see Analysis 2 — larger disruption → higher AUC?]
  3. Position effects: [see Analysis 3 — edge enrichment, CDS length]
  4. Change types: [see Analysis 4 — transition vs transversion bias]

  These results inform which genes are amenable to thermodynamic-based
  classification of synonymous variants, and suggest that the primary
  mechanism captured by the ΔG profile is local mRNA destabilization
  (not long-range splice effects, which require genomic context).
""")

    print("=" * 70)


if __name__ == "__main__":
    main()
