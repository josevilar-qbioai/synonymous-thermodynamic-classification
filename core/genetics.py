"""
Utilidades genéticas compartidas: tabla de codones, traducción, parsing.
"""

import re

CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

AA3TO1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': '*',
}

AA1TO3 = {v: k for k, v in AA3TO1.items() if v != '*'}


def translate_cds(cds_seq):
    """Traduce secuencia CDS a proteína."""
    protein = []
    for i in range(0, len(cds_seq) - 2, 3):
        codon = cds_seq[i:i+3].upper()
        aa = CODON_TABLE.get(codon, 'X')
        if aa == '*':
            break
        protein.append(aa)
    return ''.join(protein)


def load_fasta(fasta_path):
    """Lee secuencia de un archivo FASTA."""
    with open(fasta_path) as f:
        lines = f.readlines()
    seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
    return seq.upper()


def find_missense_codon(wt_codon, target_aa):
    """Encuentra codón más cercano que codifica target_aa."""
    best_codon = None
    best_dist = 4
    for codon, aa in CODON_TABLE.items():
        if aa == target_aa:
            dist = sum(1 for a, b in zip(wt_codon, codon) if a != b)
            if dist < best_dist:
                best_dist = dist
                best_codon = codon
    return best_codon


def parse_protein_change(text):
    """
    Extrae (ref_aa, position, alt_aa) de múltiples formatos:
      - p.Cys61Gly / p.(Cys61Gly)
      - p.C61G (1-letter)
    """
    if not text:
        return None, None, None

    m = re.search(r'p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?', text)
    if m:
        ref = AA3TO1.get(m.group(1), '?')
        pos = int(m.group(2))
        alt = AA3TO1.get(m.group(3), '?')
        if ref != '?' and alt != '?' and alt != '*':
            return ref, pos, alt

    m = re.search(r'p\.([A-Z])(\d+)([A-Z])', text)
    if m:
        ref = m.group(1)
        pos = int(m.group(2))
        alt = m.group(3)
        if ref != alt and alt != '*':
            return ref, pos, alt

    return None, None, None
