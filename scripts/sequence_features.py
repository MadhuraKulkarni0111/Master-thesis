"""
sequence_features.py
====================
All sequence-level feature engineering functions.

Each function takes a raw DNA/RNA string (or a dataframe row) and returns
either a scalar or a dictionary of {feature_name: value} ready to be
assembled into a feature matrix.

This version organises features into named GROUPS (length, gc, mono, di,
kmer3, codon, uaug, kozak, cai, tai, mfe) so that ablation studies can
request a subset of groups instead of editing this file per run. See
ablations.py for how groups map onto ablation IDs A1-A13.

Functions
---------
extract_regions(row)              → (utr5, cds, utr3)  strings from a df row
build_features(df, species, ...)  → pd.DataFrame, optionally + column->group map
"""
print("extracting features")

import numpy as np
import pandas as pd
import math
import RNA
from itertools import product

from config import STOP_CODONS
from config import START_WINDOW_UPSTREAM
from config import START_WINDOW_DOWNSTREAM
from config import CAI_WEIGHTS_FILE
from config import TAI_WEIGHTS_FILE

_cai_df = pd.read_csv(CAI_WEIGHTS_FILE)
_tai_df = pd.read_csv(TAI_WEIGHTS_FILE)

# Pre-compute all 61 sense codons once at import time
ALL_CODONS = [
    a + b + c
    for a, b, c in product("ACGT", repeat=3)
    if (a + b + c) not in STOP_CODONS
]


# ── Region extraction ───────────────────────────────────────────────────────

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


def get_start_codon_window(utr5, cds, upstream=30, downstream=30):
    """
    Extract a window centered on the start codon: up to `upstream` nt from
    the end of the 5'UTR plus the first `downstream` nt of the CDS.
    """
    left = utr5[-upstream:] if len(utr5) >= upstream else utr5
    right = cds[:downstream]
    return left + right


# ── Nucleotide composition ──────────────────────────────────────────────────

def gc_content(seq):
    """GC fraction of a sequence. np.nan for empty sequences."""
    if not seq:
        return np.nan
    return (seq.count("G") + seq.count("C")) / len(seq)


def mono_freq(seq, label):
    """Fraction of each individual nucleotide (A, T, G, C) in a region."""
    n = len(seq)
    if n == 0:
        return {f"{label}_{nt}": np.nan for nt in "ATGC"}
    return {f"{label}_{nt}": seq.count(nt) / n for nt in "ATGC"}


def di_freq(seq, label):
    """Fraction of each of the 16 possible dinucleotides in a region."""
    dinucs = [a + b for a, b in product("ATGC", repeat=2)]
    n = len(seq) - 1
    if n <= 0:
        return {f"{label}_{d}": np.nan for d in dinucs}
    return {f"{label}_{d}": seq.count(d) / n for d in dinucs}


def kmer_freq(seq, label, k):
    """Fraction of each possible k-mer in a region (overlapping windows)."""
    kmers = ["".join(p) for p in product("ATGC", repeat=k)]
    n = len(seq) - k + 1
    if n <= 0:
        return {f"{label}_{kmer}": np.nan for kmer in kmers}
    return {f"{label}_{kmer}": seq.count(kmer) / n for kmer in kmers}


# ── Codon usage ──────────────────────────────────────────────────────────────

def codon_freq(cds_seq):
    """Relative frequency of each of the 61 sense codons within the CDS."""
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
    """Count upstream AUG codons in the 5'UTR."""
    return utr5_seq.count("ATG")


# ── Minimum folding energy ────────────────────────────────────────────────────

def mfe_fold(seq):
    """Compute minimum free energy (MFE) using ViennaRNA. np.nan if empty."""
    if not seq:
        return np.nan
    seq = seq.replace("T", "U")
    try:
        _, mfe = RNA.fold(seq)
        return mfe
    except Exception:
        return np.nan


# ── Codon Adaptation Index (CAI) ──────────────────────────────────────────────

def load_cai_weights(species: str):
    """Load species-specific CAI weights: dict {codon: weight}."""
    column = f"cai_weight_{species}"
    return dict(zip(_cai_df["codon"], _cai_df[column]))


def calculate_cai(cds_seq, weights):
    """Calculate Codon Adaptation Index."""
    codons = []
    for i in range(0, len(cds_seq) - 2, 3):
        codon = cds_seq[i:i + 3]
        if codon in weights:
            codons.append(weights[codon])
    if len(codons) == 0:
        return np.nan
    return math.exp(sum(math.log(w) for w in codons) / len(codons))


# ── tRNA Adaptation Index (TAI) ───────────────────────────────────────────────

def load_tai_weights(species: str):
    column = f"tai_weight_{species}"
    return dict(zip(_tai_df["codon"], _tai_df[column]))


def calculate_tai(cds_seq, weights):
    codons = [cds_seq[i:i + 3] for i in range(0, len(cds_seq) - 2, 3)]
    vals = []
    for c in codons:
        if c == "ATG" or c in STOP_CODONS:
            continue
        w = weights.get(c)
        if w and w > 0:
            vals.append(w)
    if not vals:
        return np.nan
    return math.exp(sum(math.log(w) for w in vals) / len(vals))


# ── Kozak Index Sequence Score (KISS) ─────────────────────────────────────────

def kiss_score(utr5, cds):
    """
    Kozak Similarity Score using positions -6 -5 -4 -3 -2 -1 AUG +4.
    Returns a score between 0 and 1.
    """
    if len(utr5) < 6 or len(cds) < 4:
        return np.nan

    context = utr5[-6:] + cds[:4]
    score = 0
    if context[0] == "G":
        score += 1
    if context[1] == "C":
        score += 1
    if context[2] == "C":
        score += 1
    if context[3] in ("A", "G"):
        score += 1
    if context[4] == "C":
        score += 1
    if context[5] == "C":
        score += 1
    if context[9] == "G":
        score += 1
    return score / 7


# ── Feature group functions ───────────────────────────────────────────────────
# Each takes (utr5, cds, utr3, full, weights=None, tai_weights=None) and
# returns a dict of {feature_name: value}. This is the single place that
# defines what "length", "gc", "mono", etc. mean — ablations.py just
# references these group names, it never touches biology.

def _feat_length(utr5, cds, utr3, full, **kw):
    return {
        "log_utr5": np.log1p(len(utr5)),
        "log_cds":  np.log1p(len(cds)),
        "log_utr3": np.log1p(len(utr3)),
        "log_tx":   np.log1p(len(full)),
    }


def _feat_gc(utr5, cds, utr3, full, **kw):
    return {
        "gc_utr5": gc_content(utr5),
        "gc_cds":  gc_content(cds),
        "gc_utr3": gc_content(utr3),
        "gc_full": gc_content(full),
    }


def _feat_mono(utr5, cds, utr3, full, **kw):
    feat = {}
    feat.update(mono_freq(utr5, "utr5"))
    feat.update(mono_freq(cds,  "cds"))
    feat.update(mono_freq(utr3, "utr3"))
    return feat


def _feat_di(utr5, cds, utr3, full, **kw):
    feat = {}
    feat.update(di_freq(utr5, "utr5"))
    feat.update(di_freq(cds,  "cds"))
    feat.update(di_freq(utr3, "utr3"))
    return feat


def _feat_kmer3(utr5, cds, utr3, full, **kw):
    feat = {}
    feat.update(kmer_freq(utr5, "utr5", 3))
    feat.update(kmer_freq(cds,  "cds",  3))
    feat.update(kmer_freq(utr3, "utr3", 3))
    return feat


def _feat_codon(utr5, cds, utr3, full, **kw):
    return codon_freq(cds)


def _feat_uaug(utr5, cds, utr3, full, **kw):
    return {"uAUG_count": uaug_count(utr5)}


def _feat_kozak(utr5, cds, utr3, full, **kw):
    return {"kozak_score": kiss_score(utr5, cds)}


def _feat_cai(utr5, cds, utr3, full, weights=None, **kw):
    return {"cai": calculate_cai(cds, weights)}


def _feat_tai(utr5, cds, utr3, full, tai_weights=None, **kw):
    return {"tai": calculate_tai(cds, tai_weights)}


def _feat_mfe(utr5, cds, utr3, full, **kw):
    window = get_start_codon_window(
        utr5, cds,
        upstream=START_WINDOW_UPSTREAM,
        downstream=START_WINDOW_DOWNSTREAM,
    )
    return {"mfe_start": mfe_fold(window)}


FEATURE_FUNCS = {
    "length": _feat_length,
    "gc":     _feat_gc,
    "mono":   _feat_mono,
    "di":     _feat_di,
    "kmer3":  _feat_kmer3,
    "codon":  _feat_codon,
    "uaug":   _feat_uaug,
    "kozak":  _feat_kozak,
    "cai":    _feat_cai,
    "tai":    _feat_tai,
    "mfe":    _feat_mfe,
}

# Regions a column can belong to. "window" = spans the UTR5/CDS boundary
# (kozak, mfe_start); "full" = whole-transcript aggregate (gc_full, log_tx).
VALID_REGIONS = {"utr5", "cds", "utr3", "full", "window"}

# Exact-name lookups for columns whose region/group can't be inferred from
# a simple prefix pattern (single global columns, not per-region triples).
_EXACT_TAGS = {
    "cai":          ("cai",    "cds"),
    "tai":          ("tai",    "cds"),
    "kozak_score":  ("kozak",  "window"),
    "mfe_start":    ("mfe",    "window"),
    "uAUG_count":   ("uaug",   "utr5"),
    "log_utr5":     ("length", "utr5"),
    "log_cds":      ("length", "cds"),
    "log_utr3":     ("length", "utr3"),
    "log_tx":       ("length", "full"),
    "gc_utr5":      ("gc",     "utr5"),
    "gc_cds":       ("gc",     "cds"),
    "gc_utr3":      ("gc",     "utr3"),
    "gc_full":      ("gc",     "full"),
}


def classify_column(colname):
    """
    Determine which feature GROUP and which REGION a column belongs to,
    purely from its name — no need to touch the data or cache anything.

    Used by data_loader to slice the cached full feature matrix down to
    whatever an ablation's groups/regions filter asks for (see
    ablations.py). Cheap enough (string parsing over ~300 column names)
    that it's recomputed on every load rather than cached to disk.

    Returns
    -------
    (group, region) : tuple of str
        group  in FEATURE_FUNCS.keys()
        region in VALID_REGIONS
        ("unknown", "unknown") if the column doesn't match any known
        pattern — should never happen for columns produced by
        build_features(); a mismatch here means a new feature was added
        to FEATURE_FUNCS without updating this classifier.
    """
    if colname in _EXACT_TAGS:
        return _EXACT_TAGS[colname]

    if colname.startswith("codon_"):
        return "codon", "cds"

    # mono/di/kmer3 columns are named "<region>_<nt-string>", e.g.
    # "utr5_A" (mono, 1 char), "cds_AT" (di, 2 chars), "utr3_AAA" (kmer3, 3 chars)
    if "_" in colname:
        prefix, suffix = colname.split("_", 1)
        if prefix in ("utr5", "cds", "utr3"):
            group_by_len = {1: "mono", 2: "di", 3: "kmer3"}
            group = group_by_len.get(len(suffix), "unknown")
            return group, prefix

    return "unknown", "unknown"


# ── Master feature builder ────────────────────────────────────────────────────

def build_features(df, species, groups=None):
    """
    Engineer sequence features for every gene in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from Excel. Must contain:
        tx_sequence, utr5_size, cds_size.
    species : str
        "human" or "mouse" — selects CAI/TAI weight columns. Only used
        if "cai" and/or "tai" are in `groups`.
    groups : list of str or None
        Which feature groups to compute (see FEATURE_FUNCS keys above).
        None (default) computes every group — this is what should be
        used to build the cached full feature matrix; ablations then
        slice columns out of that cache (via classify_column) rather
        than calling this again.

    Returns
    -------
    pd.DataFrame  (n_genes, n_features)
    """
    groups = list(FEATURE_FUNCS.keys()) if groups is None else groups
    unknown = set(groups) - set(FEATURE_FUNCS)
    if unknown:
        raise ValueError(f"Unknown feature group(s): {unknown}")

    needs_cai = "cai" in groups
    needs_tai = "tai" in groups
    weights     = load_cai_weights(species) if needs_cai else None
    tai_weights = load_tai_weights(species) if needs_tai else None

    records = []
    for _, row in df.iterrows():
        utr5, cds, utr3 = extract_regions(row)
        full = utr5 + cds + utr3
        feat = {}
        for g in groups:
            feat.update(FEATURE_FUNCS[g](
                utr5, cds, utr3, full,
                weights=weights, tai_weights=tai_weights,
            ))
        records.append(feat)

    print("Returning features")
    return pd.DataFrame(records, index=df.index)