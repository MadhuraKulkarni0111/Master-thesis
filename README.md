# Translation Efficiency (TE) Feature Importance Pipeline

Predicts mRNA translation efficiency (TE) from sequence-derived features,
and measures which features matter, using a family of **ablation studies**.
Seven regression models are cross-validated on pre-defined folds for two
datasets (human HCT116, mouse 4T1). An optional extension adds RNA-FM
embeddings as features.

---

## 1. Repository layout

```
feature_engineering/
├── data/
│   ├── Human_data.xlsx              # input: human dataset
│   ├── Mouse_data.xlsx              # input: mouse dataset
│   ├── combined_cai_weights.csv     # codon weights for CAI
│   └── combined_tai_weights.csv     # codon weights for tAI
├── scripts/
│   ├── config.py                    # paths, hyperparameters, dataset list
│   ├── ablations.py                 # ablation definitions (A, B, C, D families)
│   ├── sequence_features.py         # feature engineering (11 feature groups)
│   ├── data_loader.py               # Excel loading, feature cache, column slicing
│   ├── models.py                    # 7 models + out-of-fold cross-validation
│   ├── model_results.py             # R², MAE, RMSE, Pearson, Spearman -> CSV
│   ├── importance.py                # feature importances -> CSV
│   ├── visualise.py                 # bar charts and heatmaps
│   ├── run_pipeline.py              # main entry point (ablations)
│   ├── rna_fm_features.py           # RNA-FM embedding extraction (optional)
│   ├── run_rnafm.py                 # entry point for the RNA-FM runs (optional)
│   ├── cache_features.sbatch        # SLURM: build the feature cache
│   ├── run_ablations.sbatch         # SLURM array: A1-A13
│   ├── run_ablations_caitai.sbatch  # SLURM array: A1-A6 + CAI/tAI
│   ├── run_ablations_bcd.sbatch     # SLURM array: B1-D4
│   ├── run_pipeline.slurm           # SLURM: single run
│   └── run_rnafm.sbatch             # SLURM array: RNA-FM runs (optional)
└── results/                         # created automatically
    ├── feature_cache/               # cached feature matrices (parquet)
    ├── rnafm_cache/                 # cached RNA-FM vectors (optional)
    └── <ablation_id>/               # one directory per ablation
```

All paths are derived from the location of `config.py`, so the project works
from any directory as long as the `data/` and `scripts/` layout is kept.

---

## 2. Input data

Each Excel file needs these columns:

| Column | Meaning |
|---|---|
| `tx_sequence` | Full transcript, laid out as 5'UTR + CDS + 3'UTR (T or U accepted) |
| `utr5_size` | Length of the 5'UTR in nucleotides |
| `cds_size` | Length of the CDS in nucleotides |
| `fold` | Integer fold assignment (e.g. 0-9) used for cross-validation |
| `TE_HCT116` / `TE_4T1` | Target TE value (human / mouse; set in `config.DATASETS`) |

The CAI and tAI CSVs need a `codon` column plus `cai_weight_human`,
`cai_weight_mouse` and `tai_weight_human`, `tai_weight_mouse`.

Rows with a missing target or sequence are dropped.

---

## 3. How the pipeline works

```
config.py + ablations.py
        |
data_loader.py     read Excel -> build/load cached full feature matrix
        |          -> select columns for the ablation -> NaN -> 0
models.py          out-of-fold CV on the dataset's folds, then refit on all data
        |
model_results.py   metrics per fold + pooled out-of-fold
importance.py      coefficients / MDI / gain / permutation importance
visualise.py       feature-importance bars, model-comparison heatmap
        |
results/<ablation_id>/
```

**Feature cache.** The full feature matrix (all 11 groups, including the slow
ViennaRNA MFE step) is computed **once per dataset** and saved as parquet.
Every ablation reads that cache and just selects columns, so ablations never
recompute features.

### Feature groups (326 columns total)

| Group | Columns | Description |
|---|---|---|
| `length` | 4 | log1p length of 5'UTR, CDS, 3'UTR, full transcript |
| `gc` | 4 | GC fraction per region and full transcript |
| `mono` | 12 | A/T/G/C frequency per region |
| `di` | 48 | 16 dinucleotide frequencies per region |
| `kmer3` | 192 | 64 trinucleotide frequencies per region (all frames) |
| `codon` | 61 | In-frame sense-codon usage in the CDS |
| `uaug` | 1 | Number of upstream AUGs in the 5'UTR |
| `kozak` | 1 | Kozak-context match score around the start codon |
| `cai` | 1 | Codon Adaptation Index |
| `tai` | 1 | tRNA Adaptation Index |
| `mfe` | 1 | Minimum free energy of a window around the start codon |

### Models

Lasso, ElasticNet, RandomForest, LightGBM, XGBoost, SVR (RBF), LinearSVM.
Hyperparameters live in `config.py`.

| Model | Importance reported |
|---|---|
| Lasso, ElasticNet, LinearSVM | absolute coefficient (signed coefficient also saved) |
| RandomForest | mean decrease in impurity |
| LightGBM, XGBoost | total gain |
| SVR | permutation importance (5 repeats, R² drop) |

### Ablation families (defined in `ablations.py`)

| Family | Question | IDs |
|---|---|---|
| A | Main progression: how much does each feature group add? | A1-A13 (A13 = full model) |
| A + CAI/tAI | Does codon optimality add signal before uAUG/Kozak? | `A{1..6}_cai`, `_tai`, `_cai_tai` |
| B | Location: how much does each region explain alone? | B1 (5'UTR), B2 (CDS), B3 (3'UTR), B4, B5 |
| C | Mechanism: initiation vs elongation | C1, C2, C3 |
| D | Representation: 3-mers vs codons vs start-centred | D1-D4 |

---

## 4. Setup

### 4.1 Environment

Use Python 3.11 or 3.12. Corrected `environment.yml`:

```yaml
name: master-thesis
channels:
  - conda-forge
  - bioconda
dependencies:
  - python=3.12
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - lightgbm
  - xgboost
  - openpyxl
  - pyarrow
  - viennarna
  - pip
  - pip:
      - --extra-index-url https://download.pytorch.org/whl/cu121
      - torch
      - rna-fm
```

```bash
conda env create -f environment.yml
conda activate master-thesis
```

Note: `torch` and `rna-fm` are only needed for the RNA-FM extension. 


### 4.4 Cluster-specific notes

- The SLURM scripts activate conda from a hard-coded path
  (`/home/mkulkar/software/miniforge3/...`). Edit it if your install differs.
- Create the log folder before submitting, because SLURM will not:
  `mkdir -p logs`.
  ```

---

## 5. Running the pipeline

All commands run from the `scripts/` directory.

### 5.1 Locally (small tests)

```bash
# Quick test: set MAX_SAMPLES = 500 in config.py first, then:
python run_pipeline.py --ablation A1 --dataset human

# One ablation, both datasets
python run_pipeline.py --ablation B1

# Everything 
python run_pipeline.py
```

`--dataset` accepts `human` or `mouse`. `--ablation` accepts any ID defined in
`ablations.py`. Set `MAX_SAMPLES = None` for the full dataset, can be changed if required.

### 5.2 On the cluster (recommended)

**Step 1: build the feature cache once.** 

```bash
mkdir -p logs
CACHE_JOB=$(sbatch --parsable cache_features.sbatch)
```

**Step 2: launch the ablation arrays after the cache finishes.**

```bash
sbatch --dependency=afterok:$CACHE_JOB run_ablations.sbatch          # A1-A13        (26 tasks)
sbatch --dependency=afterok:$CACHE_JOB run_ablations_caitai.sbatch   # A1-A6 CAI/tAI (36 tasks)
sbatch --dependency=afterok:$CACHE_JOB run_ablations_bcd.sbatch      # B1-D4         (24 tasks)
```

Each array task runs one (ablation, dataset) pair. Outputs go to separate
directories, so tasks can run in parallel safely.
---

## 6. Outputs

Each ablation gets its own directory, `results/<ablation_id>/`. For each
dataset (`<label>` = `Human_HCT116` or `Mouse_4T1`):

| File | Contents |
|---|---|
| `<label>_r2_results.csv` | R² per fold, mean, std, and pooled out-of-fold (`oof`) |
| `<label>_eval_results.csv` | R², MAE, RMSE, Pearson r, Spearman rho, same row layout |
| `<label>_top_features.csv` | top 50 features per model, with signed coefficient for linear models |
| `<label>_feature_importance.png` | per-model bar charts (importance and coefficients) |
| `<label>_model_comparison.png` | heatmap comparing top features across models |

**Which number to report:** the `oof` row is the pooled out-of-fold score, where
every gene is predicted exactly once by the fold that held it out. The
`mean`/`std` rows show stability across folds.

---



