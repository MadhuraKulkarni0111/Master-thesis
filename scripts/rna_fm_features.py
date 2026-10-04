"""
rna_fm_features.py
===================
Extract RNA-FM sequence embeddings as input features for the TE pipeline.

RNA-FM (Chen et al. 2022, https://github.com/ml4bio/RNA-FM) is a BERT-style
foundation model pretrained on ~23M non-coding RNA sequences. We use it here
as an additional feature source alongside the hand-crafted features in
sequence_features.py: instead of (only) engineered GC content, codon usage,
etc., RNA-FM gives a dense learned representation of the sequence that may
capture structural/functional signal the hand-crafted features miss.

Requires the `rna-fm` package (pip install rna-fm) and torch.

Each transcript is embedded once and cached to disk (RNAFM_CACHE_DIR, keyed
by a hash of the sequence itself, not the gene index), then re-used on
subsequent runs -- RNA-FM inference is by far the slowest step in this
pipeline, so caching matters a lot given how often the rest of the pipeline
gets re-run while tuning models.

Functions
---------
get_embedding(seq, pooling)
    -> np.ndarray, the (cached) embedding vector for one sequence.

build_rna_fm_features(df, seq_col="tx_sequence")
    -> pd.DataFrame, one row per gene, columns rnafm_0 ... rnafm_{d-1}.
"""

import hashlib
import os
import time

import numpy as np
import pandas as pd
import torch

from config import RNAFM_CACHE_DIR, RNAFM_MAX_LEN, RNAFM_POOLING

_model = None
_alphabet = None
_batch_converter = None
_device = None


# --------------------------------------------------------------------------
# Model loading (lazy singleton -- only loaded if/when actually needed)
# --------------------------------------------------------------------------
def _load_model():
    global _model, _alphabet, _batch_converter, _device
    if _model is not None:
        return _model, _alphabet, _batch_converter, _device

    import fm  # from the `rna-fm` package

    print("  [rna-fm] loading pretrained RNA-FM (rna_fm_t12) ...")
    _model, _alphabet = fm.pretrained.rna_fm_t12()
    _model.eval()
    _batch_converter = _alphabet.get_batch_converter()
    if torch.cuda.is_available():
        _device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _device = "mps"          # Apple Silicon laptops
    else:
        _device = "cpu"
    _model = _model.to(_device)
    print(f"  [rna-fm] model loaded on {_device}")
    return _model, _alphabet, _batch_converter, _device


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------
def _seq_hash(seq: str) -> str:
    """Content-addressed cache key. Two genes with an identical sequence
    (or the same gene re-run after a row-order change) hit the same cache
    entry, so the key is a hash of the sequence itself rather than the
    dataframe index. The pooling and max-length settings are part of the
    key, so changing either in config.py never silently reuses vectors
    computed under the old settings."""
    key = f"{RNAFM_POOLING}|{RNAFM_MAX_LEN}|{seq}"
    return hashlib.sha1(key.encode()).hexdigest()


def _cache_path(seq: str) -> str:
    return os.path.join(str(RNAFM_CACHE_DIR), f"{_seq_hash(seq)}.npy")


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------
def _prepare_sequence(seq: str, max_len: int = RNAFM_MAX_LEN) -> str:
    """RNA-FM's positional embeddings only go up to ~1024 tokens. Truncate
    from the 3' end so the 5'UTR + start-codon region -- usually the most
    information-dense part of the transcript for translation efficiency --
    is preserved; long 3'UTRs get cut off first. If your transcripts are
    CDS-dominant and the 3' signal matters more for your question, swap
    this for a start-codon-centred window instead."""
    seq = seq.upper().replace("T", "U")
    return seq[:max_len]


def _embed_one(seq: str, model, batch_converter, device, pooling: str) -> np.ndarray:
    data = [("seq", _prepare_sequence(seq))]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        out = model(tokens, repr_layers=[12])
    reps = out["representations"][12][0]  # (seq_len + 2, embed_dim) incl. BOS/EOS
    reps = reps[1:-1]                     # drop BOS/EOS tokens

    if pooling == "mean":
        vec = reps.mean(dim=0)
    elif pooling == "max":
        vec = reps.max(dim=0).values
    elif pooling == "cls":
        vec = out["representations"][12][0][0]
    else:
        raise ValueError(f"Unknown RNAFM_POOLING setting: {pooling!r}")

    return vec.cpu().numpy()


def get_embedding(seq: str, pooling: str = RNAFM_POOLING) -> np.ndarray:
    """Return the RNA-FM embedding vector for one sequence, using the
    on-disk cache if this exact sequence has been embedded before."""
    os.makedirs(str(RNAFM_CACHE_DIR), exist_ok=True)
    cpath = _cache_path(seq)
    if os.path.exists(cpath):
        return np.load(cpath)

    model, alphabet, batch_converter, device = _load_model()
    vec = _embed_one(seq, model, batch_converter, device, pooling)
    # write to a temp file then rename, so an interrupted run (Ctrl-C on a
    # laptop, a killed SLURM job) never leaves a half-written .npy behind
    tmp = cpath + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, vec)
    os.replace(tmp, cpath)
    return vec


# --------------------------------------------------------------------------
# Master builder -- mirrors sequence_features.build_features()
# --------------------------------------------------------------------------
def build_rna_fm_features(df: pd.DataFrame, seq_col: str = "tx_sequence") -> pd.DataFrame:
    """
    Embed every transcript in df with RNA-FM and return one row per gene.

    Parameters
    ----------
    df      : pd.DataFrame, must contain seq_col.
    seq_col : str, default "tx_sequence" -- the full 5'UTR+CDS+3'UTR string,
              same raw column sequence_features.py slices up for the
              hand-crafted features. Pass "cds" (if you add such a column)
              to embed CDS only instead.

    Returns
    -------
    pd.DataFrame, shape (n_genes, embedding_dim), index matches df,
    columns rnafm_0 ... rnafm_{d-1} (embedding_dim = 640 for rna_fm_t12).
    """
    print(f"  [rna-fm] embedding {len(df)} sequences from '{seq_col}' "
          f"(cache: {RNAFM_CACHE_DIR})")

    vectors = []
    n_cached = 0
    t0 = time.time()
    for i, seq in enumerate(df[seq_col].astype(str)):
        if os.path.exists(_cache_path(seq)):
            n_cached += 1
        vectors.append(get_embedding(seq))
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta_min = rate * (len(df) - i - 1) / 60
            print(f"    {i + 1}/{len(df)} done  "
                  f"({rate:.2f} s/seq, ~{eta_min:.0f} min left)", flush=True)

    print(f"  [rna-fm] {n_cached}/{len(df)} sequences served from cache")
    arr = np.vstack(vectors)
    cols = [f"rnafm_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=df.index, columns=cols)