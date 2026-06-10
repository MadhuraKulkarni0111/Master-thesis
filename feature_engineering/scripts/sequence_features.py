"""
sequence_features.py
====================
All sequence-level feature engineering functions.

Each function takes a raw DNA/RNA string (or a dataframe row and returns
either a scalar or a dictionary of {feature_name: value} ready to be
assembled into a feature matrix.

Functions
---------
extract_regions(row)     → (utr5, cds, utr3)  strings from a dataframe row
gc_content(seq)          → float
mono_freq(seq, label)    → dict  (4 features: label_A, label_T, label_G, label_C)
di_freq(seq, label)      → dict  (16 features: label_AA … label_TT)
codon_freq(cds_seq)      → dict  (61 features: codon_AAA … codon_TTT)
uaug_count(utr5_seq)     → int
build_features(df)       → pd.DataFrame  (133 features per gene)
"""

import numpy as np
import pandas as pd
from itertools import product

from config import STOP_CODONS

# Pre-compute all 61 sense codons once at import time
ALL_CODONS = [
    a + b + c
    for a, b, c in product("ACGT", repeat=3)
    if (a + b + c) not in STOP_CODONS
]


# ── Region extraction ─────────────────────────────────────────────────────────

def extract_regions(row):
    """
    Slice the full transcript sequence into its three biological regions.

    The Excel file stores the entire mRNA as a single string in tx_sequence,
    laid out as:   [5'UTR][CDS][3'UTR]
    The boundary positions are given by utr5_size and cds_size.

    Parameters
    ----------
    row : pd.Series
        A single dataframe row containing tx_sequence, utr5_size, cds_size.

    Returns
    -------
    utr5, cds, utr3 : tuple of str
        Three substrings in DNA alphabet (U replaced with T).
    """
    seq = str(row["tx_sequence"]).upper().replace("U", "T")
    u5  = int(row["utr5_size"]) if pd.notna(row["utr5_size"]) else 0
    cds = int(row["cds_size"])  if pd.notna(row["cds_size"])  else 0
    return seq[:u5], seq[u5 : u5 + cds], seq[u5 + cds:]


# ── Nucleotide composition ────────────────────────────────────────────────────

def gc_content(seq):
    """
    GC fraction of a sequence.

    GC base pairs are stronger than AT/AU, so GC-rich regions fold into
    more stable secondary structures, which can slow ribosome scanning
    (5'UTR) or elongation (CDS).

    Returns np.nan for empty sequences (e.g. genes with no annotated UTR).
    """
    if not seq:
        return np.nan
    return (seq.count("G") + seq.count("C")) / len(seq)


def mono_freq(seq, label):
    """
    Fraction of each individual nucleotide (A, T, G, C) in a region.

    Returns a dict with keys  label_A, label_T, label_G, label_C.
    Note: the four values sum to 1, so one is linearly redundant —
    models handle this via regularisation. (lasso and elastic net)

    Parameters
    ----------
    seq   : str   DNA sequence of the region
    label : str   prefix added to each key, e.g. "utr5", "cds", "utr3"
    """
    n = len(seq)
    if n == 0:
        return {f"{label}_{nt}": np.nan for nt in "ATGC"}
    return {f"{label}_{nt}": seq.count(nt) / n for nt in "ATGC"}


def di_freq(seq, label):
    """
    Fraction of each of the 16 possible dinucleotides in a region.

    Counts overlapping pairs: for a sequence of length N there are N-1
    dinucleotide positions. Dinucleotide context captures phenomena like
    CpG suppression and codon-boundary composition biases that single-
    nucleotide frequencies miss.

    Returns a dict with 16 keys:  label_AA, label_AT, … label_TT
    """
    dinucs = [a + b for a, b in product("ATGC", repeat=2)]
    n = len(seq) - 1
    if n <= 0:
        return {f"{label}_{d}": np.nan for d in dinucs} 
    return {f"{label}_{d}": seq.count(d) / n for d in dinucs} # --> normalisation


# ── Codon usage ───────────────────────────────────────────────────────────────

def codon_freq(cds_seq):
    """
    Relative frequency of each of the 61 sense codons within the CDS.

    Walks the CDS in non-overlapping triplets starting at position 0
    (reading frame 0). Stop codons and triplets containing N are skipped.

    Codon usage bias is the strongest known sequence-level predictor of
    translation efficiency: different synonymous codons are decoded at
    different speeds depending on the abundance of their cognate tRNAs,
    directly affecting ribosome density and protein output.

    Returns a dict with 61 keys: codon_AAA, codon_AAC, … codon_TTT
    Returns np.nan for all keys if the CDS is empty or contains no valid codons.
    """
    result = {f"codon_{c}": 0.0 for c in ALL_CODONS}
    codons_found = []

    for i in range(0, len(cds_seq) - 2, 3):
        codon = cds_seq[i : i + 3]
        if len(codon) == 3 and codon not in STOP_CODONS and "N" not in codon:
            codons_found.append(codon)

    total = len(codons_found)
    if total == 0:
        return {k: np.nan for k in result}

    for c in codons_found:
        result[f"codon_{c}"] += 1

    return {k: v / total for k, v in result.items()}


# ── Upstream AUG count ────────────────────────────────────────────────────────

def uaug_count(utr5_seq):
    """
    Count upstream AUG codons in the 5'UTR.

    Each ATG in the 5'UTR can act as an alternative start codon, recruiting
    ribosomes that then translate a short upstream ORF instead of scanning
    through to the main CDS start. More uAUGs generally correlates with
    lower translation efficiency of the main protein.
    """
    return utr5_seq.count("ATG")


# ── Master feature builder ────────────────────────────────────────────────────

def build_features(df):
    """
    Engineer all 133 features for every gene in the dataframe.

    For each row, calls extract_regions then all feature functions,
    merges their output dicts, and returns a DataFrame where:
      - rows   = genes (same index as input df)
      - columns = feature names

    Feature breakdown
    -----------------
    Region sizes (raw + log)  :  7   (utr5, cds, utr3, log versions + log_tx)
    GC content                :  4   (utr5, cds, utr3, full)
    uAUG count                :  1
    Mononucleotide frequencies: 12   (4 nt × 3 regions)
    Dinucleotide frequencies  : 48   (16 dinucs × 3 regions)
    Codon usage               : 61   (61 sense codons, CDS only)
    ─────────────────────────────────
    Total                     : 133

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from Excel. Must contain:
        tx_sequence, utr5_size, cds_size.

    Returns
    -------
    pd.DataFrame  shape (n_genes, 133)
    """
    records = []

    for _, row in df.iterrows():
        utr5, cds, utr3 = extract_regions(row)
        full = utr5 + cds + utr3
        feat = {}

        # region sizes — raw lengths and log1p-transformed versions
        # log1p compresses the wide range of lengths (30 nt to 20,000+ nt)
        # so long genes don't dominate linear model coefficients
        # feat["cds_size"]  = len(cds)
        # feat["utr3_size"] = len(utr3)
        # feat["log_utr5"]  = np.log1p(len(utr5))
        feat["log_cds"]   = np.log1p(len(cds))
        feat["log_utr3"]  = np.log1p(len(utr3))
        feat["log_tx"]    = np.log1p(len(full))

        # GC content per region
        feat["gc_utr5"] = gc_content(utr5)
        feat["gc_cds"]  = gc_content(cds)
        feat["gc_utr3"] = gc_content(utr3)
        feat["gc_full"] = gc_content(full)

        # upstream AUG count
        feat["uAUG_count"] = uaug_count(utr5)

        # mononucleotide frequencies — 4 nucleotides × 3 regions = 12
        feat.update(mono_freq(utr5, "utr5"))
        feat.update(mono_freq(cds,  "cds"))
        feat.update(mono_freq(utr3, "utr3"))

        # dinucleotide frequencies — 16 dinucs × 3 regions = 48
        feat.update(di_freq(utr5, "utr5"))
        feat.update(di_freq(cds,  "cds"))
        feat.update(di_freq(utr3, "utr3"))

        # codon usage — 61 sense codons from CDS only
        feat.update(codon_freq(cds))

        records.append(feat)

    return pd.DataFrame(records, index=df.index)
