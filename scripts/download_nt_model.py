"""
download_nt_model.py
====================
Download the NT-v2 model (weights + tokenizer + custom modelling code) into
NT_MODEL_DIR (see config.py). Run ONCE on the login node, which has internet:

    python download_nt_model.py

Afterwards all jobs load the model from that local folder, offline.
Needs: pip install huggingface_hub
"""
import os

# this script is the one place that is allowed to use the network
os.environ["HF_HUB_OFFLINE"] = "0"

from huggingface_hub import snapshot_download
from config import NT_MODEL_NAME, NT_MODEL_DIR

print(f"Downloading {NT_MODEL_NAME}\n  -> {NT_MODEL_DIR}")
NT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=NT_MODEL_NAME, local_dir=str(NT_MODEL_DIR))

total = sum(f.stat().st_size for f in NT_MODEL_DIR.rglob("*") if f.is_file())
print(f"Done. {total / 1e9:.2f} GB in {NT_MODEL_DIR}")
print("Files:", sorted(p.name for p in NT_MODEL_DIR.iterdir() if p.is_file()))
