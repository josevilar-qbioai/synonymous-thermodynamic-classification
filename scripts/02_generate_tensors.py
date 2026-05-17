#!/usr/bin/env python3
"""
02_generate_tensors.py
=======================
Generate 9-channel biophysical tensors for all synonymous variants.

For each gene in synonymous_expanded_all.csv:
  - Load CDS FASTA from data/{GENE}_cds.fasta
  - Parse HGVS to get CDS-relative position and alleles
  - Generate 9ch × 128 tensor
  - Pool all genes together

Input:
  data/synonymous_expanded_all.csv
  data/{GENE}_cds.fasta (for each gene)

Output:
  data/tensors_expanded_synonymous.pt  — PyTorch tensor (N × 9 × 128)
  data/labels_expanded_synonymous.pt   — Labels tensor (N,)
  data/expanded_gene_map.csv           — Gene + label mapping
  data/expanded_tensor_summary.csv     — Per-gene tensor counts

Usage:
  python scripts/02b_generate_tensors_expanded.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    print("ERROR: PyTorch required. Install: pip install torch")
    sys.exit(1)

# Add EnergyFingerprint core
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.energy import stacking_profile, STACKING_SANTALUCIA
from core.genetics import load_fasta

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WINDOW_SIZE = 128


# ─── Parse HGVS ─────────────────────────────────────────────────────────────

def parse_hgvs_coding(name: str) -> tuple:
    """Extract CDS position, ref, alt from HGVS c. notation."""
    match = re.search(r':c\.(\d+)([ACGT])>([ACGT])', name, re.IGNORECASE)
    if match:
        return int(match.group(1)), match.group(2).upper(), match.group(3).upper()
    return None, None, None


# ─── Tensor generation ──────────────────────────────────────────────────────

def generate_tensor(cds_seq: str, cds_position: int, ref: str, alt: str) -> np.ndarray | None:
    """Generate 9-channel × 128-position tensor for a variant."""
    pos_0 = cds_position - 1

    if pos_0 < 0 or pos_0 >= len(cds_seq):
        return None

    if cds_seq[pos_0].upper() != ref.upper():
        return None

    alt_seq = cds_seq[:pos_0] + alt + cds_seq[pos_0 + 1:]

    half_window = WINDOW_SIZE // 2
    start = max(0, pos_0 - half_window)
    end = min(len(cds_seq), pos_0 + half_window)

    ref_window = cds_seq[start:end]
    alt_window = alt_seq[start:end]

    ref_profile = stacking_profile(ref_window)
    alt_profile = stacking_profile(alt_window)

    def pad_or_truncate(profile, target_len=WINDOW_SIZE):
        if len(profile) >= target_len:
            return profile[:target_len]
        return np.concatenate([profile, np.zeros(target_len - len(profile))])

    ref_profile = pad_or_truncate(ref_profile)
    alt_profile = pad_or_truncate(alt_profile)
    diff_profile = alt_profile - ref_profile

    ref_grad = np.gradient(ref_profile)
    ref_curv = np.gradient(ref_grad)
    alt_grad = np.gradient(alt_profile)
    alt_curv = np.gradient(alt_grad)
    diff_grad = np.gradient(diff_profile)
    diff_curv = np.gradient(diff_grad)

    return np.stack([
        ref_profile, ref_grad, ref_curv,
        alt_profile, alt_grad, alt_curv,
        diff_profile, diff_grad, diff_curv
    ], axis=0)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  TENSOR GENERATION — Expanded Synonymous Dataset")
    print("=" * 70)

    csv_path = DATA_DIR / "synonymous_expanded_all.csv"
    if not csv_path.exists():
        print(f"\n  ERROR: {csv_path.name} not found!")
        print(f"  Run: python scripts/01b_download_all_synonymous_clinvar.py")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"\n  Loaded: {len(df)} variants across {df['gene'].nunique()} genes")
    print(f"  Pathogenic: {(df['label']==1).sum()} | Benign: {(df['label']==0).sum()}")

    # Process each gene
    all_tensors = []
    all_labels = []
    all_genes = []
    gene_stats = []

    for gene in sorted(df["gene"].unique()):
        gene_df = df[df["gene"] == gene]
        n_p = (gene_df["label"] == 1).sum()
        n_b = (gene_df["label"] == 0).sum()

        # Load CDS
        fasta_path = DATA_DIR / f"{gene}_cds.fasta"
        if not fasta_path.exists():
            print(f"  {gene:12s} | No CDS FASTA — skipping ({n_p}P, {n_b}B)")
            gene_stats.append({"gene": gene, "input_p": n_p, "input_b": n_b,
                              "tensors": 0, "tensor_p": 0, "tensor_b": 0,
                              "skipped": len(gene_df), "mismatches": 0, "status": "no_cds"})
            continue

        try:
            cds_seq = load_fasta(str(fasta_path))
        except Exception as e:
            print(f"  {gene:12s} | CDS load error: {e}")
            gene_stats.append({"gene": gene, "input_p": n_p, "input_b": n_b,
                              "tensors": 0, "tensor_p": 0, "tensor_b": 0,
                              "skipped": len(gene_df), "mismatches": 0, "status": "cds_error"})
            continue

        tensors = []
        labels = []
        skipped = 0
        mismatches = 0

        for _, row in gene_df.iterrows():
            name = str(row.get("name", ""))
            hgvs_pos, hgvs_ref, hgvs_alt = parse_hgvs_coding(name)

            if hgvs_pos is None:
                cds_pos = row.get("cds_position")
                if pd.notna(cds_pos):
                    hgvs_pos = int(cds_pos)
                    hgvs_ref = str(row.get("ref", "")).upper()
                    hgvs_alt = str(row.get("alt", "")).upper()
                else:
                    skipped += 1
                    continue

            tensor = generate_tensor(cds_seq, hgvs_pos, hgvs_ref, hgvs_alt)
            if tensor is not None:
                tensors.append(tensor)
                labels.append(int(row["label"]))
            else:
                mismatches += 1

        tp = sum(1 for l in labels if l == 1)
        tb = sum(1 for l in labels if l == 0)
        marker = " ★" if tp > 0 else ""
        print(f"  {gene:12s} | CDS {len(cds_seq):>5}nt | tensors: {len(tensors):>4} (P={tp}, B={tb}) "
              f"| skip={skipped} mismatch={mismatches}{marker}")

        if tensors:
            all_tensors.extend(tensors)
            all_labels.extend(labels)
            all_genes.extend([gene] * len(tensors))

        gene_stats.append({"gene": gene, "input_p": n_p, "input_b": n_b,
                          "tensors": len(tensors), "tensor_p": tp, "tensor_b": tb,
                          "skipped": skipped, "mismatches": mismatches, "status": "ok"})

    # Save pooled tensors
    if all_tensors:
        pooled_tensors = torch.tensor(np.array(all_tensors), dtype=torch.float32)
        pooled_labels = torch.tensor(all_labels, dtype=torch.long)

        torch.save(pooled_tensors, DATA_DIR / "tensors_expanded_synonymous.pt")
        torch.save(pooled_labels, DATA_DIR / "labels_expanded_synonymous.pt")

        gene_map = pd.DataFrame({"gene": all_genes, "label": all_labels})
        gene_map.to_csv(DATA_DIR / "expanded_gene_map.csv", index=False)

        total_p = sum(1 for l in all_labels if l == 1)
        total_b = sum(1 for l in all_labels if l == 0)

        print(f"\n{'=' * 70}")
        print(f"  POOLED EXPANDED: {len(all_tensors)} tensors (P={total_p}, B={total_b})")
        print(f"  Shape: {pooled_tensors.shape}")
        print(f"  Genes represented: {len(set(all_genes))}")
        print(f"\n  Saved:")
        print(f"    tensors_expanded_synonymous.pt")
        print(f"    labels_expanded_synonymous.pt")
        print(f"    expanded_gene_map.csv")

        # Comparison with original
        orig_tensors = DATA_DIR / "tensors_pooled_synonymous.pt"
        if orig_tensors.exists():
            orig = torch.load(orig_tensors, weights_only=True)
            print(f"\n  vs. original 8-gene dataset:")
            print(f"    Original: {orig.shape[0]} tensors")
            print(f"    Expanded: {pooled_tensors.shape[0]} tensors ({pooled_tensors.shape[0]/orig.shape[0]:.1f}x)")
            print(f"    New pathogenic: {total_p} (was ~26)")
    else:
        print("\n  No tensors generated!")

    # Save gene-level stats
    stats_df = pd.DataFrame(gene_stats)
    stats_df.to_csv(DATA_DIR / "expanded_tensor_summary.csv", index=False)
    print(f"\n  Gene stats: expanded_tensor_summary.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
