#!/usr/bin/env python3
"""
01_download_clinvar.py
=======================
Download ALL synonymous variants from ClinVar — no gene filter.

Scans the ENTIRE ClinVar database to maximize pathogenic synonymous variant
count for CNN training.

Strategy:
  1. Reuse cached variant_summary.txt.gz (or download if missing)
  2. Extract ALL synonymous SNVs with Pathogenic/Benign classification
  3. Auto-discover RefSeq transcript IDs via NCBI for new genes
  4. Download CDS FASTA for every gene with ≥1 pathogenic variant
  5. Save expanded dataset

IMPORTANT: Must be run LOCALLY (NCBI FTP + Entrez needed).

Output:
  data/synonymous_expanded_all.csv      — All synonymous P/B, all genes
  data/synonymous_expanded_pathogenic.csv — Pathogenic-only for inspection
  data/expanded_gene_summary.csv        — Per-gene P/B counts
  data/{GENE}_cds.fasta                 — CDS for each gene (auto-downloaded)

Usage:
  python scripts/01b_download_all_synonymous_clinvar.py
  python scripts/01b_download_all_synonymous_clinvar.py --min-benign 5
  python scripts/01b_download_all_synonymous_clinvar.py --pathogenic-only
"""

import os
import sys
import re
import gzip
import csv
import time
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from Bio import Entrez, SeqIO
except ImportError:
    print("ERROR: BioPython required. Install: pip install biopython")
    sys.exit(1)

# ─── Configuration ───────────────────────────────────────────────────────────

Entrez.email = "your_email@example.com"  # Replace with your NCBI-registered email
Entrez.api_key = os.environ.get("NCBI_API_KEY", None)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Optional: check for pre-downloaded CDS files
EF_STUDIES = None  # Set to path of existing CDS files if available

# Known RefSeq transcripts (avoids re-querying NCBI for common genes)
KNOWN_REFSEQ = {
    "BRCA1": "NM_007294.4",
    "TP53":  "NM_000546.6",
    "PTEN":  "NM_000314.8",
    "PALB2": "NM_024675.4",
    "CFTR":  "NM_000492.4",
    "HBB":   "NM_000518.5",
    "MECP2": "NM_004992.4",
    "SCN1A": "NM_001165963.4",
}

CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"

NCBI_DELAY = 0.35  # seconds between NCBI requests


# ─── Label encoding ─────────────────────────────────────────────────────────

def encode_significance(sig: str) -> int:
    s = sig.lower().strip()
    if "conflicting" in s: return -1
    if "uncertain" in s: return -1
    if "not provided" in s: return -1
    if "risk factor" in s: return -1
    if "drug response" in s: return -1
    if "pathogenic" in s: return 1
    if "benign" in s: return 0
    return -1


# ─── Step 1: Download (reuses cached file) ──────────────────────────────────

def download_variant_summary():
    output_path = DATA_DIR / "variant_summary.txt.gz"
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Using cached file: {output_path.name} ({size_mb:.1f} MB)")
        return output_path

    print(f"  Downloading from: {CLINVAR_URL}")
    print(f"  This may take 2-5 minutes...")
    import urllib.request
    urllib.request.urlretrieve(CLINVAR_URL, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Downloaded: {size_mb:.1f} MB")
    return output_path


# ─── Step 2: Extract ALL synonymous variants ────────────────────────────────

def filter_all_synonymous(gz_path: Path) -> list:
    """
    Parse variant_summary.txt.gz and extract ALL synonymous SNVs
    with clear P/B classification. NO gene filter.
    """
    print(f"\n  Parsing {gz_path.name} (full scan, no gene filter)...")

    variants = []
    total_lines = 0
    synonymous_found = 0

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().strip()
        headers = header_line.split("\t")
        col_map = {h.strip("#"): i for i, h in enumerate(headers)}

        # Column indices
        idx_type = col_map.get("Type")
        idx_name = col_map.get("Name")
        idx_gene = col_map.get("GeneSymbol")
        idx_sig = col_map.get("ClinicalSignificance", col_map.get("ClinSig"))
        idx_assembly = col_map.get("Assembly")
        idx_chr = col_map.get("Chromosome")
        idx_start = col_map.get("Start", col_map.get("PositionVCF"))
        idx_ref = col_map.get("ReferenceAlleleVCF", col_map.get("ReferenceAllele"))
        idx_alt = col_map.get("AlternateAlleleVCF", col_map.get("AlternateAllele"))
        idx_varid = col_map.get("VariationID", col_map.get("AlleleID"))
        idx_review = col_map.get("ReviewStatus")
        idx_origin = col_map.get("Origin")

        print(f"  Columns: {len(headers)}")

        for line in f:
            total_lines += 1
            if total_lines % 500000 == 0:
                print(f"    ...{total_lines:,} lines | {synonymous_found} synonymous P/B found")

            fields = line.strip().split("\t")

            # Must be SNV
            var_type = fields[idx_type] if idx_type is not None and len(fields) > idx_type else ""
            if "single nucleotide" not in var_type.lower():
                continue

            # Check synonymous via HGVS name
            name = fields[idx_name] if idx_name is not None and len(fields) > idx_name else ""
            is_synonymous = False
            if "(p.=" in name or "p.=" in name:
                is_synonymous = True
            elif "synonymous" in name.lower():
                is_synonymous = True
            elif "=" in name and "p." in name:
                if re.search(r'p\.[A-Z][a-z]{2}\d+=', name):
                    is_synonymous = True

            if not is_synonymous:
                continue

            # Assembly filter (GRCh38 only)
            assembly = fields[idx_assembly] if idx_assembly is not None and len(fields) > idx_assembly else ""
            if assembly and "GRCh38" not in assembly:
                continue

            # Significance filter
            sig = fields[idx_sig] if idx_sig is not None and len(fields) > idx_sig else ""
            label = encode_significance(sig)
            if label < 0:
                continue

            # Gene
            gene = fields[idx_gene] if idx_gene is not None and len(fields) > idx_gene else ""
            if not gene or gene == "-":
                continue

            # Multi-gene entries: take first gene
            if ";" in gene:
                gene = gene.split(";")[0].strip()

            synonymous_found += 1

            # Extract CDS position from HGVS
            cds_match = re.search(r':c\.(\d+)', name)
            cds_position = int(cds_match.group(1)) if cds_match else None

            variants.append({
                "variant_id": fields[idx_varid] if idx_varid is not None and len(fields) > idx_varid else str(synonymous_found),
                "gene": gene.upper(),
                "chromosome": fields[idx_chr] if idx_chr is not None and len(fields) > idx_chr else "",
                "position": fields[idx_start] if idx_start is not None and len(fields) > idx_start else "",
                "ref": fields[idx_ref] if idx_ref is not None and len(fields) > idx_ref else "",
                "alt": fields[idx_alt] if idx_alt is not None and len(fields) > idx_alt else "",
                "cds_position": cds_position,
                "name": name,
                "significance": sig,
                "review_status": fields[idx_review] if idx_review is not None and len(fields) > idx_review else "",
                "origin": fields[idx_origin] if idx_origin is not None and len(fields) > idx_origin else "",
                "label": label,
            })

    print(f"\n  Total lines: {total_lines:,}")
    print(f"  Synonymous P/B: {synonymous_found}")

    return variants


# ─── Step 3: Look up RefSeq transcript for new genes ────────────────────────

def lookup_refseq_transcript(gene: str) -> str | None:
    """
    Query NCBI Gene to find the canonical RefSeq NM_ transcript for a gene.
    Returns transcript accession like 'NM_001234.5' or None.
    """
    # Check known first
    if gene in KNOWN_REFSEQ:
        return KNOWN_REFSEQ[gene]

    try:
        # Search NCBI Gene
        handle = Entrez.esearch(
            db="gene",
            term=f'{gene}[Gene Name] AND "Homo sapiens"[Organism]',
            retmax=1
        )
        result = Entrez.read(handle)
        handle.close()
        time.sleep(NCBI_DELAY)

        if not result["IdList"]:
            return None

        gene_id = result["IdList"][0]

        # Fetch gene record to find RefSeq transcript
        handle = Entrez.efetch(db="gene", id=gene_id, retmode="xml")
        gene_data = Entrez.read(handle)
        handle.close()
        time.sleep(NCBI_DELAY)

        # Navigate XML to find NM_ accession
        # The structure varies, so we try multiple paths
        if gene_data:
            entry = gene_data[0] if isinstance(gene_data, list) else gene_data

            # Try to find in annotations / products
            try:
                locus = entry.get("Entrezgene_locus", [])
                for loc in locus:
                    products = loc.get("Gene-commentary_products", [])
                    for prod in products:
                        accession = prod.get("Gene-commentary_accession", "")
                        if accession.startswith("NM_"):
                            version = prod.get("Gene-commentary_version", "")
                            full_acc = f"{accession}.{version}" if version else accession
                            return full_acc
            except (KeyError, TypeError, AttributeError):
                pass

        # Fallback: search nucleotide database directly
        handle = Entrez.esearch(
            db="nucleotide",
            term=f'{gene}[Gene Name] AND "Homo sapiens"[Organism] AND RefSeq[Filter] AND NM_[Accession]',
            retmax=1,
            sort="relevance"
        )
        result = Entrez.read(handle)
        handle.close()
        time.sleep(NCBI_DELAY)

        if result["IdList"]:
            # Get the accession
            handle = Entrez.efetch(db="nucleotide", id=result["IdList"][0], rettype="acc", retmode="text")
            accession = handle.read().strip()
            handle.close()
            time.sleep(NCBI_DELAY)
            if accession.startswith("NM_"):
                return accession

        return None

    except Exception as e:
        print(f"      WARNING: NCBI lookup failed for {gene}: {e}")
        return None


# ─── Step 4: Download CDS FASTA ─────────────────────────────────────────────

def download_cds_fasta(gene: str, transcript: str) -> Path | None:
    """Download CDS FASTA from NCBI given a RefSeq transcript ID."""
    dest = DATA_DIR / f"{gene}_cds.fasta"

    if dest.exists():
        return dest

    # Check pre-existing CDS files (optional)
    if EF_STUDIES is not None:
        existing_path = EF_STUDIES / gene.lower() / "data" / f"{gene}_cds.fasta"
        if existing_path.exists():
            import shutil
            shutil.copy2(existing_path, dest)
            print(f"      Copied from cache: {gene}")
            return dest

    # Download
    try:
        handle = Entrez.efetch(
            db="nucleotide", id=transcript,
            rettype="fasta_cds_na", retmode="text"
        )
        content = handle.read()
        handle.close()
        time.sleep(NCBI_DELAY)

        if content.strip() and ">" in content:
            with open(dest, "w") as f:
                f.write(content)
            return dest
        else:
            print(f"      WARNING: Empty CDS for {gene} ({transcript})")
            return None

    except Exception as e:
        print(f"      ERROR downloading CDS {gene}: {e}")
        return None


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download ALL synonymous variants from ClinVar")
    parser.add_argument("--min-benign", type=int, default=0,
                        help="Minimum benign variants per gene to include (default: 0 = all)")
    parser.add_argument("--pathogenic-only", action="store_true",
                        help="Only download CDS for genes with pathogenic variants")
    parser.add_argument("--skip-cds-download", action="store_true",
                        help="Skip CDS FASTA downloads (just extract variants)")
    args = parser.parse_args()

    print("=" * 70)
    print("  EXPANDED SYNONYMOUS EXTRACTION — ALL ClinVar Genes")
    print("  (No gene filter — maximizing pathogenic count for Strategy B)")
    print("=" * 70)

    # Step 1: Download
    print("\n" + "─" * 70)
    print("STEP 1: Download / load ClinVar variant_summary")
    print("─" * 70)
    gz_path = download_variant_summary()

    # Step 2: Filter
    print("\n" + "─" * 70)
    print("STEP 2: Extract ALL synonymous SNVs (no gene filter)")
    print("─" * 70)
    variants = filter_all_synonymous(gz_path)

    if not variants:
        print("\n  No synonymous variants found!")
        sys.exit(1)

    # Organize by gene
    import pandas as pd
    df = pd.DataFrame(variants)

    total_p = (df["label"] == 1).sum()
    total_b = (df["label"] == 0).sum()
    n_genes = df["gene"].nunique()
    n_genes_p = df[df["label"] == 1]["gene"].nunique()

    print(f"\n  Total synonymous variants: {len(df)}")
    print(f"    Pathogenic: {total_p}")
    print(f"    Benign:     {total_b}")
    print(f"  Genes with data:  {n_genes}")
    print(f"  Genes with P:     {n_genes_p}")

    # Per-gene summary
    gene_summary = df.groupby("gene").agg(
        pathogenic=("label", lambda x: (x == 1).sum()),
        benign=("label", lambda x: (x == 0).sum()),
        total=("label", "count")
    ).reset_index()
    gene_summary = gene_summary.sort_values("pathogenic", ascending=False)

    print(f"\n  Top genes by pathogenic count:")
    print(f"  {'Gene':<12} {'P':>5} {'B':>7} {'Total':>7}")
    print(f"  {'─'*12} {'─'*5} {'─'*7} {'─'*7}")
    for _, row in gene_summary.head(30).iterrows():
        marker = " ★" if row["pathogenic"] > 0 else ""
        print(f"  {row['gene']:<12} {row['pathogenic']:>5} {row['benign']:>7} {row['total']:>7}{marker}")

    if n_genes > 30:
        remaining = n_genes - 30
        print(f"  ... and {remaining} more genes (benign only)")

    # Save all variants
    fieldnames = ["variant_id", "gene", "chromosome", "position", "ref", "alt",
                  "cds_position", "name", "significance", "review_status", "origin", "label"]

    all_path = DATA_DIR / "synonymous_expanded_all.csv"
    df.to_csv(all_path, index=False, columns=fieldnames)
    print(f"\n  Saved: {all_path.name} ({len(df)} rows)")

    # Save pathogenic-only for inspection
    path_df = df[df["label"] == 1]
    path_path = DATA_DIR / "synonymous_expanded_pathogenic.csv"
    path_df.to_csv(path_path, index=False, columns=fieldnames)
    print(f"  Saved: {path_path.name} ({len(path_df)} rows)")

    # Save gene summary
    gene_summary.to_csv(DATA_DIR / "expanded_gene_summary.csv", index=False)
    print(f"  Saved: expanded_gene_summary.csv")

    # Step 3: Download CDS FASTA for genes with pathogenic variants
    if not args.skip_cds_download:
        print("\n" + "─" * 70)
        print("STEP 3: Download CDS FASTA for new genes")
        print("─" * 70)

        if args.pathogenic_only:
            genes_to_download = path_df["gene"].unique().tolist()
            print(f"  Downloading CDS for {len(genes_to_download)} genes with pathogenic variants")
        else:
            # Download for genes with pathogenic variants + genes with ≥min_benign benign
            genes_with_p = set(path_df["gene"].unique())
            if args.min_benign > 0:
                genes_with_enough_b = set(
                    gene_summary[gene_summary["benign"] >= args.min_benign]["gene"]
                )
                genes_to_download = sorted(genes_with_p | genes_with_enough_b)
            else:
                genes_to_download = sorted(genes_with_p)
            print(f"  Downloading CDS for {len(genes_to_download)} genes")

        success = 0
        failed = []
        refseq_map = {}

        for i, gene in enumerate(genes_to_download):
            prefix = f"  [{i+1}/{len(genes_to_download)}] {gene}"

            # Check if already exists
            dest = DATA_DIR / f"{gene}_cds.fasta"
            if dest.exists():
                print(f"{prefix}: exists ✓")
                success += 1
                continue

            # Look up RefSeq transcript
            print(f"{prefix}: looking up RefSeq...", end=" ", flush=True)
            transcript = lookup_refseq_transcript(gene)

            if transcript:
                refseq_map[gene] = transcript
                print(f"{transcript} → ", end="", flush=True)
                result = download_cds_fasta(gene, transcript)
                if result:
                    print(f"✓")
                    success += 1
                else:
                    print(f"FAILED")
                    failed.append(gene)
            else:
                print(f"no RefSeq found ✗")
                failed.append(gene)

        print(f"\n  CDS download: {success} success, {len(failed)} failed")
        if failed:
            print(f"  Failed genes: {', '.join(failed)}")

        # Save RefSeq mapping
        if refseq_map:
            import json
            map_path = DATA_DIR / "expanded_refseq_map.json"
            # Merge with known
            full_map = {**KNOWN_REFSEQ, **refseq_map}
            with open(map_path, "w") as f:
                json.dump(full_map, fp=f, indent=2)
            print(f"  Saved: {map_path.name}")

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Expanded dataset:")
    print(f"    Total synonymous P/B: {len(df)}")
    print(f"    Pathogenic: {total_p} (across {n_genes_p} genes)")
    print(f"    Benign:     {total_b} (across {n_genes} genes)")
    print(f"\n  vs. original 8-gene dataset:")
    print(f"    Original pathogenic: ~29")
    print(f"    Expansion factor:    ~{total_p/29:.1f}x (if >29)")

    print(f"\n  NEXT STEPS:")
    print(f"    1. Inspect: data/synonymous_expanded_pathogenic.csv")
    print(f"    2. Generate tensors: python scripts/02b_generate_tensors_expanded.py")
    print(f"    3. Strategy B training with expanded dataset")
    print("=" * 70)


if __name__ == "__main__":
    main()
