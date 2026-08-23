#!/usr/bin/env python3
"""
build_cai_features.py

Generate species-specific Codon Adaptation Index (CAI) features
for human and mouse CDS.

CAI weights are calculated from ribosomal protein genes using
RSCU and the Sharp & Li (1987) method.

The script downloads CDS data from NCBI, calculates CAI weights,
scores valid CDS, and generates combined human/mouse tables.

"""

import csv
import glob
import math
import os
import re
import subprocess
import sys
import zipfile
from collections import defaultdict

from common_weight import CODON_TABLE, STOP_CODONS, read_fasta

# --------------------------------------------------------------------------
# Defaults matching your folder layout
# --------------------------------------------------------------------------
DEFAULT_OUTDIR = "/Users/madhurakulkarni/Desktop/master_thesis/feature_engineering/data"

SPECIES = {
    "human": "GCF_000001405.40",  # GRCh38.p14
    "mouse": "GCF_000001635.27",  # GRCm39
}

# --------------------------------------------------------------------------
# Standard genetic code (NCBI translation table 1)
# --------------------------------------------------------------------------

GENE_TAG_RE = re.compile(r"\[gene=([^\]]+)\]")
# Nuclear ribosomal protein genes used as the CAI reference set.
# Case-insensitive to cover human and mouse gene naming conventions.
DEFAULT_REF_GENE_RE = re.compile(r"^RP[LS]\d", re.IGNORECASE)


# --------------------------------------------------------------------------
# Step 1: download all CDS for a genome assembly
# --------------------------------------------------------------------------
def download_cds(accession: str, outdir: str) -> str:
    """Download CDS-from-genomic FASTA for an NCBI assembly accession
    using the `datasets` CLI. Returns path to the FASTA file. Cached:
    if the file already exists, skips re-downloading."""
    os.makedirs(outdir, exist_ok=True)
    cached = glob.glob(os.path.join(outdir, "ncbi_dataset", "data", "*", "cds_from_genomic.fna"))
    if cached:
        print(f"  [download] using cached file: {cached[0]}")
        return cached[0]

    zip_path = os.path.join(outdir, f"{accession}.zip")
    print(f"  [download] fetching CDS for {accession} ...")
    subprocess.run(
        ["datasets", "download", "genome", "accession", accession,
         "--include", "cds", "--filename", zip_path],
        check=True,
    )
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(outdir)

    found = glob.glob(os.path.join(outdir, "ncbi_dataset", "data", "*", "cds_from_genomic.fna"))
    if not found:
        sys.exit("ERROR: download succeeded but cds_from_genomic.fna was not found in the archive.")
    return found[0]

# Minimal FASTA reader imported from common_weights 

def extract_gene_name(header: str):
    m = GENE_TAG_RE.search(header)
    return m.group(1) if m else None

# --------------------------------------------------------------------------
# CDS quality control
# --------------------------------------------------------------------------
def is_valid_cds(seq: str) -> bool:
    seq = seq.upper()
    if len(seq) < 6 or len(seq) % 3 != 0:
        return False
    if seq[:3] != "ATG":
        return False
    codons = [seq[i:i + 3] for i in range(0, len(seq), 3)]
    if codons[-1] not in STOP_CODONS:
        return False
    if any(c in STOP_CODONS for c in codons[:-1]):
        return False
    if any(set(c) - set("ACGT") for c in codons):
        return False
    return True


# --------------------------------------------------------------------------
# Step 2: select ribosomal protein genes (RPL/RPS) as reference set
# --------------------------------------------------------------------------
def select_reference_set(fasta_path: str, gene_regex: re.Pattern):
    """Returns list of (gene, header, seq) passing QC and matching gene_regex,
    plus a list of ALL valid (gene, header, seq) CDS for genome-wide scoring."""
    ref_records = []
    all_valid = []
    n_total, n_valid, n_ref = 0, 0, 0

    for header, seq in read_fasta(fasta_path):
        n_total += 1
        seq = seq.upper()
        if not is_valid_cds(seq):
            continue
        n_valid += 1
        gene = extract_gene_name(header)
        all_valid.append((gene, header, seq))
        if gene and gene_regex.match(gene):
            ref_records.append((gene, header, seq))
            n_ref += 1

    print(f"  [select] {n_total} CDS records read, {n_valid} passed QC, "
          f"{n_ref} matched reference-gene pattern")
    return ref_records, all_valid


# --------------------------------------------------------------------------
# Step 3: RSCU
# --------------------------------------------------------------------------
def codon_family_map():
    fam = defaultdict(list)
    for codon, aa in CODON_TABLE.items():
        if aa != '*':
            fam[aa].append(codon)
    return fam


def compute_codon_counts(seqs):
    counts = defaultdict(int)
    for seq in seqs:
        codons = [seq[i:i + 3] for i in range(0, len(seq) - 3, 3)]  # drop final stop codon
        for c in codons:
            if c in CODON_TABLE and CODON_TABLE[c] != '*':
                counts[c] += 1
    return counts


def compute_rscu(counts, fam):
    rscu = {}
    for aa, codons in fam.items():
        total = sum(counts.get(c, 0) for c in codons)
        n = len(codons)
        if total == 0:
            for c in codons:
                rscu[c] = 0.0
            continue
        expected = total / n
        for c in codons:
            rscu[c] = counts.get(c, 0) / expected
    return rscu


# --------------------------------------------------------------------------
# Step 4: Sharp & Li (1987) CAI weights
# --------------------------------------------------------------------------
def compute_cai_weights(rscu, fam):
    weights = {}
    for aa, codons in fam.items():
        maxr = max(rscu[c] for c in codons)
        for c in codons:
            weights[c] = rscu[c] / maxr if maxr > 0 else 0.0
    return weights


def compute_cai(seq: str, weights: dict):
    codons = [seq[i:i + 3] for i in range(0, len(seq) - 3, 3)]
    log_sum, n = 0.0, 0
    for c in codons:
        aa = CODON_TABLE.get(c)
        if aa is None or aa == '*':
            continue
        w = weights.get(c)
        if w and w > 0:
            log_sum += math.log(w)
            n += 1
    return math.exp(log_sum / n) if n else None


# --------------------------------------------------------------------------
# Output writers (per-species)
# --------------------------------------------------------------------------
def write_reference_gene_table(ref_records, outpath):
    with open(outpath, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["gene", "length_nt", "header"])
        for gene, header, seq in ref_records:
            w.writerow([gene, len(seq), header])


def write_rscu_table(counts, rscu, fam, outpath):
    codon_to_aa = {c: aa for aa, codons in fam.items() for c in codons}
    with open(outpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["codon", "amino_acid", "count", "rscu"])
        for codon in sorted(codon_to_aa):
            w.writerow([codon, codon_to_aa[codon], counts.get(codon, 0), round(rscu[codon], 4)])


def write_weights(weights, fam, csv_path):
    codon_to_aa = {c: aa for aa, codons in fam.items() for c in codons}
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["codon", "amino_acid", "cai_weight"])
        for codon in sorted(codon_to_aa):
            w.writerow([codon, codon_to_aa[codon], round(weights[codon], 4)])


def write_genome_scores(all_valid, weights, outpath):
    with open(outpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gene", "header", "length_nt", "cai"])
        for gene, header, seq in all_valid:
            cai = compute_cai(seq, weights)
            w.writerow([gene or "", header, len(seq), "" if cai is None else round(cai, 4)])


# --------------------------------------------------------------------------
# Per-species runner
# --------------------------------------------------------------------------
def run_species(species: str, accession: str, base_outdir: str, gene_regex: re.Pattern):
    outdir = os.path.join(base_outdir, species)
    os.makedirs(outdir, exist_ok=True)
    print(f"\n===== {species} ({accession}) =====")

    fasta_path = download_cds(accession, outdir)
    ref_records, all_valid = select_reference_set(fasta_path, gene_regex)
    if len(ref_records) < 10:
        print(f"  WARNING: only {len(ref_records)} reference genes found for {species}. "
              "CAI weights may be unreliable. Check gene-symbol naming or relax --gene-regex.")

    fam = codon_family_map()
    ref_seqs = [seq for _, _, seq in ref_records]
    counts = compute_codon_counts(ref_seqs)
    rscu = compute_rscu(counts, fam)
    weights = compute_cai_weights(rscu, fam)

    write_reference_gene_table(ref_records, os.path.join(outdir, "reference_genes.tsv"))
    write_rscu_table(counts, rscu, fam, os.path.join(outdir, "rscu_table.csv"))
    write_weights(weights, fam,os.path.join(outdir, "cai_weights.csv"))
    write_genome_scores(all_valid, weights, os.path.join(outdir, "cai_scores.csv"))

    print(f"  [done] reference genes used: {len(ref_records)} | CDS scored: {len(all_valid)}")
    print(f"  [done] wrote reference_genes.tsv, rscu_table.csv, cai_weights.csv, "
          f"cai_scores.csv -> {outdir}/")
    return outdir, fam


# --------------------------------------------------------------------------
# Merge human + mouse into ML-ready combined tables
# --------------------------------------------------------------------------
def merge_weights(species_outdirs: dict, base_outdir: str):
    per_species = {}
    aa_lookup = {}
    for species, outdir in species_outdirs.items():
        with open(os.path.join(outdir, "cai_weights.csv")) as fh:
            reader = csv.DictReader(fh)
            d = {}
            for row in reader:
                d[row["codon"]] = float(row["cai_weight"])
                aa_lookup[row["codon"]] = row["amino_acid"]
            per_species[species] = d

    out_path = os.path.join(base_outdir, "combined_cai_weights.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["codon", "amino_acid"] + [f"cai_weight_{sp}" for sp in species_outdirs]
        w.writerow(header)
        for codon in sorted(aa_lookup):
            row = [codon, aa_lookup[codon]] + [round(per_species[sp][codon], 4) for sp in species_outdirs]
            w.writerow(row)
    print(f"\n[merge] wrote {out_path}")
    return out_path


def merge_scores(species_outdirs: dict, base_outdir: str):
    out_path = os.path.join(base_outdir, "combined_cai_scores.csv")
    with open(out_path, "w", newline="") as out_fh:
        w = csv.writer(out_fh)
        w.writerow(["species", "gene", "header", "length_nt", "cai"])
        for species, outdir in species_outdirs.items():
            with open(os.path.join(outdir, "cai_scores.csv")) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    w.writerow([species, row["gene"], row["header"], row["length_nt"], row["cai"]])
    print(f"[merge] wrote {out_path}")
    return out_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():

    gene_regex = DEFAULT_REF_GENE_RE
    outdir = DEFAULT_OUTDIR

    os.makedirs(outdir, exist_ok=True)

    species_outdirs = {}
    for species, accession in SPECIES.items():
        species_dir, _ = run_species(species, accession, outdir, gene_regex)
        species_outdirs[species] = species_dir

    merge_weights(species_outdirs, outdir)
    merge_scores(species_outdirs, outdir)

    print(f"\nDone. All outputs saved under: {outdir}/")
    print("  human/ , mouse/                 -> per-species intermediate files")
    print("  combined_cai_weights.csv         -> codon-level features")
    print("  combined_cai_scores.csv          -> per-gene CAI, long format")

if __name__ == "__main__":
    main()