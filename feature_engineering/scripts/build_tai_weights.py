#!/usr/bin/env python3
"""
build_tai_weights.py

Computes species-specific tRNA Adaptation Index (tAI) weights, following
dos Reis, Savva & Wernisch (2004), "Solving the riddle of codon usage
preferences: a test for translational selection", Nucleic Acids Res 32:5036-44.

This is the tRNA-based counterpart to build_cai_weights.py: instead of a
codon-usage reference gene set, tAI is built directly from tRNA GENE COPY
NUMBERS in the genome (a proxy for tRNA abundance), combined with wobble
base-pairing rules.

--------------------------------------------------------------------------
Where the tRNA gene copy numbers come from
--------------------------------------------------------------------------
Unlike CDS sequences (which come straight from NCBI), tRNA gene copy number
is NOT something you compute from protein-coding CDS. The standard,
purpose-built source is GtRNAdb (https://gtrnadb.ucsc.edu/), which hosts
tRNAscan-SE-predicted tRNA gene sets for essentially every sequenced genome,
including per-gene anticodon annotation.

Steps to get the input file this script needs:
  1. Go to https://gtrnadb.ucsc.edu/ and find your genome
     (e.g. Homo sapiens GRCh38, Mus musculus GRCm39).
  2. Download the genome's tRNA gene set (FASTA header or the tab-delimited
     gene list both work — every GtRNAdb tRNA gene is named like
     "tRNA-Phe-GAA-1-1", where GAA is the anticodon).
  3. Run this script's --extract-anticodons helper (below) to turn that
     raw file into the 2-column CSV this script expects, OR just build the
     CSV yourself with columns: anticodon,gene_copy_number

Expected input CSV (--trna-csv), one row per anticodon:
    anticodon,gene_copy_number
    GAA,10
    AGC,5
    ...
(29-31 rows typically, one per distinct anticodon found in the genome —
NOT one row per gene; if you have a per-gene list, use --extract-anticodons
first to collapse it into gene copy numbers per anticodon.)

--------------------------------------------------------------------------
Method (ported directly from the reference `tAI` R package by dos Reis,
https://github.com/mariodosreis/tai, function get.ws(), to guarantee the
wobble constants match the published method exactly)
--------------------------------------------------------------------------
  1. For each of the 64 codons, look up the gene copy number of the tRNA
     whose anticodon is the reverse complement of that codon (the
     "perfect Watson-Crick match" tRNA for that codon).
  2. Absolute adaptiveness W_i = sum over every tRNA that can decode codon i
     (perfect match + wobble-pairing tRNAs) of (1 - s_ij) * gene_copy_number,
     where s_ij is dos Reis et al.'s empirically-optimised wobble penalty:
         s = [0.0, 0.0, 0.0, 0.0, 0.41, 0.28, 0.9999, 0.68, 0.89]
     (0.0 = perfect Watson-Crick pairing, higher = weaker wobble pairing)
  3. Methionine (ATG) and stop codons are excluded, matching the original
     method ("STOP and methionine codons are ignored").
  4. Relative adaptiveness w_i = W_i / max(W) -- same normalisation style
     as CAI's RSCU/RSCU_max.
  5. Any codon with w_i = 0 (no decoding tRNA at all, perfect or wobble) is
     imputed with the geometric mean of the non-zero weights, so a single
     unused codon doesn't zero out an entire gene's tAI score.
  6. tAI of a gene = geometric mean of w_i across all its codons (excluding
     Met and stops) -- mechanically identical to how CAI is computed.

Dependencies: Python 3.8+, standard library only.

Usage:
    python build_tai_weights.py

Outputs:
    data/
      hg38-tRNAs.fa
      mm39-tRNAs.fa
      human_trna_gene_counts.csv
      mouse_trna_gene_counts.csv
      human_tai_weights.csv
      mouse_tai_weights.csv
      human_tai_weights.json
      mouse_tai_weights.json
      tai_weights.csv
"""

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
import urllib.request

# ------ Defining dorectoreis -------

DATA_DIR = Path(
    "/Users/madhurakulkarni/Desktop/master_thesis/feature_engineering/data"
)
DATA_DIR.mkdir(parents=True, exist_ok=True)

GTRNADB_URLS = {
    "human": "http://gtrnadb.ucsc.edu/genomes/eukaryota/Hsapi38/hg38-tRNAs.fa",
    "mouse": "http://gtrnadb.ucsc.edu/genomes/eukaryota/Mmusc39/mm39-tRNAs.fa",
}

# Helper function to dowlonad th files 

def download_trna_fasta(species):

    url = GTRNADB_URLS[species]
    outfile = DATA_DIR / Path(url).name

    if outfile.exists():
        print(f"[cached] {outfile}")
        return outfile

    raise FileNotFoundError(
        f"{outfile} not found.\n"
        "Please download it manually from\n"
        f"{url}"
    )

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

BASES = "TCAG"
COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
STOP_CODONS = {"TAA", "TAG", "TGA"}

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
    unmatched_lines = 0
    with open(path) as fh:
        for line in fh:
            matches = GENE_NAME_ANTICODON_RE.findall(line)
            if matches:
                for anticodon in matches:
                    counts[anticodon.upper().replace("U", "T")] += 1
            elif line.strip() and not line.startswith(("#", ">")) is False:
                pass  # header/comment lines are expected to not match; ignore

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
    sking    : 0 = Eukaryota (default), 1 = Prokaryota. Only affects the
               bacterial-specific Ile-AUA lysidine-modification special
               case (s[8]); irrelevant for eukaryotic genomes such as
               human/mouse.

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
        W[block[0]] = p[0] * t_T + p[4] * t_C   # NNT: perfect match + wobble from NNC-tRNA
        W[block[1]] = p[1] * t_C + p[5] * t_T   # NNC: perfect match + wobble from NNT-tRNA
        W[block[2]] = p[2] * t_A + p[6] * t_T   # NNA: perfect match + wobble from NNT-tRNA
        W[block[3]] = p[3] * t_G + p[7] * t_A   # NNG: perfect match + wobble from NNA-tRNA

    # Methionine: perfect-match contribution only, no wobble term
    W["ATG"] = p[3] * trna_by_codon["ATG"]

    # Bacteria-only special case: lysidine-modified anticodon reading Ile AUA
    # (irrelevant for human/mouse; included only for completeness/reuse)
    if sking == 1:
        W["ATA"] = p[8] * trna_by_codon["ATA"]

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
        print(f"  [tai] {len(zero_codons)} codon(s) had no decoding tRNA "
              f"(perfect or wobble) and were imputed with the geometric mean "
              f"({gm:.4f}): {', '.join(sorted(zero_codons))}")

    return w


def calculate_tai(cds_seq: str, weights: dict):
    """tAI of one CDS: geometric mean of codon weights, excluding Met and stops
    (matching dos Reis et al.'s convention)."""
    codons = [cds_seq[i:i + 3] for i in range(0, len(cds_seq) - 2, 3)]
    log_sum, n = 0.0, 0
    for c in codons:
        if c == "ATG" or c in STOP_CODONS:
            continue
        w = weights.get(c)
        if w and w > 0:
            log_sum += math.log(w)
            n += 1
    return math.exp(log_sum / n) if n else None


# --------------------------------------------------------------------------
# Minimal FASTA reader (matches build_cai_weights.py, no Biopython needed)
# --------------------------------------------------------------------------
def read_fasta(path):
    header, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


GENE_TAG_RE = re.compile(r"\[gene=([^\]]+)\]")


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

# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------
def write_weights(weights: dict, json_path: str, csv_path: str):
    with open(json_path, "w") as fh:
        json.dump(weights, fh, indent=2, sort_keys=True)
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

    fasta = download_trna_fasta(species)

    counts = extract_anticodon_counts(fasta)

    trna_csv = DATA_DIR / f"{species}_trna_gene_counts.csv"
    write_anticodon_csv(counts, trna_csv)

    trna_gcn = load_anticodon_csv(trna_csv)

    weights = get_ws(trna_gcn)

    json_file = DATA_DIR / f"{species}_tai_weights.json"
    csv_file = DATA_DIR / f"{species}_tai_weights.csv"

    write_weights(weights, json_file, csv_file)

    print(f"Saved:")
    print(f"   {trna_csv.name}")
    print(f"   {csv_file.name}")
    print(f"   {json_file.name}")

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
