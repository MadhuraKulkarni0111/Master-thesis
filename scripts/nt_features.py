"""
nt_features.py
==============
Extract Nucleotide Transformer v2 (NT-v2) sequence embeddings as input
features for the TE pipeline.

NT-v2 (Dalla-Torre et al., InstaDeep) is a DNA language model pretrained on
850 multi-species genomes. It tokenises DNA into 6-mers (single nucleotides
where a 6-mer is not possible) and is used here, like rna_fm_features.py, as a
FROZEN feature extractor: the pretrained weights are never changed; your
transcripts go through the network and the pooled last-layer vector becomes the
input to the downstream models.

Compared with RNA-FM: pretrained on genomic DNA (which includes coding
sequence), and 1000 tokens cover ~6000 nt instead of ~1000 nt, so far less of
each transcript is truncated.

Each transcript is embedded once and cached to disk (NT_CACHE_DIR), keyed by a
hash of (model name, pooling, max tokens, sequence).

Functions
---------
build_nt_features(df, seq_col="tx_sequence")
    -> pd.DataFrame, one row per gene, columns nt_0 ... nt_{d-1}.
"""

import hashlib
import os
import time

# jobs load the model from a local folder; never try the network
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch

from config import (NT_MODEL_NAME, NT_MODEL_DIR, NT_CACHE_DIR,
                    NT_MAX_TOKENS, NT_POOLING, NT_BATCH_SIZE)

_model = None
_tokenizer = None
_device = None


# --------------------------------------------------------------------------
# Model loading (lazy singleton)
# --------------------------------------------------------------------------
def _load_model():
    global _model, _tokenizer, _device
    if _model is not None:
        return _model, _tokenizer, _device

    model_dir = str(NT_MODEL_DIR)
    if not os.path.isdir(model_dir) or not os.listdir(model_dir):
        raise FileNotFoundError(
            f"NT-v2 model not found at:\n  {model_dir}\n"
            "Run `python download_nt_model.py` on the login node first "
            "(or change NT_MODEL_DIR in config.py)."
        )

    from transformers import AutoTokenizer, AutoModelForMaskedLM

    print(f"  [nt] loading {NT_MODEL_NAME} from {model_dir} ...")
    _tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True)
    _model = AutoModelForMaskedLM.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True)
    _model.eval()
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(_device)
    print(f"  [nt] model loaded on {_device}")
    return _model, _tokenizer, _device


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------
def _seq_hash(seq: str) -> str:
    key = f"{NT_MODEL_NAME}|{NT_POOLING}|{NT_MAX_TOKENS}|{seq}"
    return hashlib.sha1(key.encode()).hexdigest()


def _cache_path(seq: str) -> str:
    return os.path.join(str(NT_CACHE_DIR), f"{_seq_hash(seq)}.npy")


def _save_vec(path: str, vec: np.ndarray):
    # temp file + rename: an interrupted job never leaves a half-written .npy
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, vec)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------
def _prepare_sequence(seq: str) -> str:
    """NT-v2 expects DNA letters. Truncation to NT_MAX_TOKENS is done by the
    tokenizer, from the 3' end, so the 5'UTR + CDS start are always kept."""
    return seq.upper().replace("U", "T")


def _embed_batch(seqs):
    model, tokenizer, device = _load_model()
    enc = tokenizer.batch_encode_plus(
        [_prepare_sequence(s) for s in seqs],
        return_tensors="pt", padding=True, truncation=True,
        max_length=NT_MAX_TOKENS,
    )
    ids = enc["input_ids"].to(device)
    attn = ids != tokenizer.pad_token_id

    with torch.no_grad():
        out = model(ids, attention_mask=attn, encoder_attention_mask=attn,
                    output_hidden_states=True)
    hs = out["hidden_states"][-1]                      # (batch, tokens, dim)

    # exclude padding and the leading <cls> token from the pooled average
    pool_mask = attn.clone()
    pool_mask[:, 0] = False
    m = pool_mask.unsqueeze(-1).to(hs.dtype)

    if NT_POOLING == "mean":
        vec = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    elif NT_POOLING == "max":
        vec = hs.masked_fill(~pool_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
    elif NT_POOLING == "cls":
        vec = hs[:, 0]
    else:
        raise ValueError(f"Unknown NT_POOLING setting: {NT_POOLING!r}")

    return vec.float().cpu().numpy()


# --------------------------------------------------------------------------
# Master builder -- mirrors sequence_features.build_features()
# --------------------------------------------------------------------------
def build_nt_features(df: pd.DataFrame, seq_col: str = "tx_sequence") -> pd.DataFrame:
    """
    Embed every transcript in df with NT-v2 and return one row per gene.

    Returns
    -------
    pd.DataFrame, shape (n_genes, embedding_dim), index matches df,
    columns nt_0 ... nt_{d-1}.
    """
    os.makedirs(str(NT_CACHE_DIR), exist_ok=True)
    seqs = df[seq_col].astype(str).tolist()
    paths = [_cache_path(s) for s in seqs]

    todo = [i for i, p in enumerate(paths) if not os.path.exists(p)]
    print(f"  [nt] {len(seqs)} sequences, {len(seqs) - len(todo)} already cached, "
          f"{len(todo)} to embed (cache: {NT_CACHE_DIR})")

    # longest-first sorting keeps padding (and wasted compute) to a minimum
    todo.sort(key=lambda i: len(seqs[i]), reverse=True)
    t0 = time.time()
    done = 0
    for start in range(0, len(todo), NT_BATCH_SIZE):
        batch_idx = todo[start:start + NT_BATCH_SIZE]
        vecs = _embed_batch([seqs[i] for i in batch_idx])
        for i, v in zip(batch_idx, vecs):
            _save_vec(paths[i], v)
        done += len(batch_idx)
        if (start // NT_BATCH_SIZE) % 25 == 0 or done == len(todo):
            rate = (time.time() - t0) / done
            eta = rate * (len(todo) - done) / 60
            print(f"    {done}/{len(todo)} embedded  "
                  f"({rate:.2f} s/seq, ~{eta:.0f} min left)", flush=True)

    arr = np.vstack([np.load(p) for p in paths])
    cols = [f"nt_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=df.index, columns=cols)
