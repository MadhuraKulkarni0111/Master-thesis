#!/usr/bin/env python3
"""
build_tai_weights.py

Compute species-specific tRNA Adaptation Index (tAI) weights
following dos Reis et al. (2004).

Input:
    GtRNAdb tRNA gene annotations.

Output:
    Species-specific tAI codon weights in CSV format.
"""

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from common_weight import CODON_TABLE, STOP_CODONS

# ------ Defining dorectoreis -------

DATA_DIR = Path(
    "/Users/madhurakulkarni/Desktop/master_thesis/feature_engineering/data"
)
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASES = "TCAG"
COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}

# dos Reis, Savva & Wernisch (2004) optimised s-values, ported from the
# reference `tAI` R package (get.ws function, github.com/mariodosreis/tai)
DEFAULT_S_VALUES = [0.0, 0.0, 0.0, 0.0, 0.41, 0.28, 0.9999, 0.68, 0.89]

GENE_NAME_ANTICODON_RE = re.compile(r"tRNA-[A-Za-z]{3}-([ACGTU]{3})", re.IGNORECASE)


# --------------------------------------------------------------------------
# Codon <-> anticodon bookkeeping
# --------------------------------------------------------------------------
def build_codon_list():
    """64 codons in the classic codon-table order: first base T,C,A,G outer,
    second base T,C,A,G, third base T,C,A,G inner (fastest-cycling)."""
    return [a + b + c for a in BASES for b in BASES for c in BASES]


def revcomp(seq: str) -> str:
    return "".join(COMPLEMENT[b] for b in reversed(seq.upper()))


# --------------------------------------------------------------------------
# Step A (optional helper): collapse a raw GtRNAdb per-gene list into
# gene-copy-number-per-anticodon counts
# --------------------------------------------------------------------------
def extract_anticodon_counts(path: str) -> dict:
    """
    Scans a text file (FASTA headers, or a plain gene-list file — anything
    containing GtRNAdb-style gene names) for tRNA-XXX-ANTICODON patterns
    and counts gene copies per anticodon.

    GtRNAdb names every tRNA gene like "tRNA-Phe-GAA-1-1" (isotype-anticodon-
    transcript#), so this simple regex works across their FASTA headers,
    gene lists, and GFF annotations alike.
    """
    counts = defaultdict(int)
    #unmatched_lines = 0
    with open(path) as fh:
        for line in fh:
            matches = GENE_NAME_ANTICODON_RE.findall(line)
            if matches:
                for anticodon in matches:
                    counts[anticodon.upper().replace("U", "T")] += 1

    if not counts:
        sys.exit(
            "ERROR: no 'tRNA-XXX-ANTICODON' style gene names found in the file.\n"
            "Make sure you downloaded the GtRNAdb gene list/FASTA for your genome "
            "(names look like 'tRNA-Phe-GAA-1-1'), or build the anticodon CSV "
            "yourself with columns: anticodon,gene_copy_number"
        )
    return dict(counts)


def write_anticodon_csv(counts: dict, outpath: str):
    with open(outpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["anticodon", "gene_copy_number"])
        for anticodon in sorted(counts):
            w.writerow([anticodon, counts[anticodon]])


def load_anticodon_csv(path: str) -> dict:
    counts = {}
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            anticodon = row["anticodon"].strip().upper().replace("U", "T")
            if len(anticodon) != 3 or any(b not in "ACGT" for b in anticodon):
                print(f"  WARNING: skipping malformed anticodon row: {row}")
                continue
            counts[anticodon] = counts.get(anticodon, 0) + int(row["gene_copy_number"])
    return counts


# --------------------------------------------------------------------------
# Step B: dos Reis et al. (2004) tAI weight calculation
# --------------------------------------------------------------------------
def get_ws(trna_gcn: dict, s=None, sking: int = 0) -> dict:
    """
    Compute relative adaptiveness (w) for every non-Met, non-stop codon.

    Parameters
    ----------
    trna_gcn : dict {anticodon: gene_copy_number}
    s        : list of 9 wobble penalties (default: dos Reis et al. 2004
               optimised values). s=0 means perfect Watson-Crick pairing;
               higher values mean weaker/less efficient wobble pairing.

    Returns
    -------
    dict {codon: relative_adaptiveness_weight}   (60 entries: 61 sense
    codons minus Met, which is excluded from tAI by convention)
    """
    s = list(s) if s is not None else DEFAULT_S_VALUES
    if len(s) != 9:
        raise ValueError("s must have exactly 9 values (dos Reis et al. 2004 convention)")
    p = [1 - x for x in s]

    codons = build_codon_list()
    trna_by_codon = {c: trna_gcn.get(revcomp(c), 0) for c in codons}

    W = {}
    for i in range(0, 64, 4):
        block = codons[i:i + 4]  # [NNT, NNC, NNA, NNG]
        t_T, t_C, t_A, t_G = (trna_by_codon[c] for c in block)
        
        W[block[0]] = p[0] * t_T + p[4] * t_C 
        W[block[1]] = p[1] * t_C + p[5] * t_T
        W[block[2]] = p[2] * t_A + p[6] * t_T
        W[block[3]] = p[3] * t_G + p[7] * t_A
    
    # Methionine: perfect-match contribution only, no wobble term
    W["ATG"] = p[3] * trna_by_codon["ATG"]

    for stop in STOP_CODONS:
        W.pop(stop, None)
    W.pop("ATG", None)  # tAI excludes Met by convention (dos Reis et al. 2004)

    max_w = max(W.values())
    if max_w == 0:
        sys.exit(
            "ERROR: every codon's absolute adaptiveness came out to zero. "
            "Check that your tRNA gene copy number CSV actually has non-zero "
            "counts and that anticodons are spelled correctly (DNA alphabet, "
            "e.g. GAA not GAA(gaa)/gaa-anticodon etc.)."
        )
    w = {c: v / max_w for c, v in W.items()}

    zero_codons = [c for c, v in w.items() if v == 0]
    if zero_codons:
        nonzero_vals = [v for v in w.values() if v > 0]
        gm = math.exp(sum(math.log(v) for v in nonzero_vals) / len(nonzero_vals))
        for c in zero_codons:
            w[c] = gm

    return w

# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------

def write_combined_weights(human_weights, mouse_weights, out_csv):
    codons = sorted(set(human_weights) | set(mouse_weights))

    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)

        writer.writerow(
            ["codon", "amino_acid", "tai_weight_human", "tai_weight_mouse"]
        )

        for codon in codons:
            writer.writerow([
                codon,
                CODON_TABLE.get(codon, ""),
                round(human_weights.get(codon, 0), 4),
                round(mouse_weights.get(codon, 0), 4),
            ])

def write_weights(weights: dict, csv_path: str):
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["codon", "tai_weight"])
        for codon in sorted(weights):
            w.writerow([codon, round(weights[codon], 4)])

def build_species(species):
    """
    Download the GtRNAdb FASTA, extract anticodon counts,
    compute tAI weights and save all outputs.
    Returns the weights dictionary.
    """

    print(f"\nProcessing {species}...")

    fasta_files = {
        "human": DATA_DIR / "hg38-tRNAs.fa",
        "mouse": DATA_DIR / "mm39-tRNAs.fa",
    }

    fasta = fasta_files[species]

    counts = extract_anticodon_counts(fasta)

    trna_csv = DATA_DIR / f"{species}_trna_gene_counts.csv"
    write_anticodon_csv(counts, trna_csv)

    trna_gcn = load_anticodon_csv(trna_csv)

    weights = get_ws(trna_gcn)

    csv_file = DATA_DIR / f"{species}_tai_weights.csv"

    write_weights(weights, csv_file)

    print(f"Saved:")
    print(f"   {trna_csv.name}")
    print(f"   {csv_file.name}")

    return weights

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():

    human_weights = build_species("human")

    mouse_weights = build_species("mouse")

    combined_csv = DATA_DIR / "combined_tai_weights.csv"

    write_combined_weights(
        human_weights,
        mouse_weights,
        combined_csv,
    )

    print("\nFinished.")
    print(f"Combined weights written to:\n{combined_csv}")

if __name__ == "__main__":
    main()
