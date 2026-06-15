# Sequence Analysis Pipeline

All files and folders currently being used can be found inside the folder labelled feature_engineering in branch separate_functions 

## Repository Structure

```text
.
├── requirements.txt
├── data/
│   ├── input/
├── scripts/
│   ├── config.py
│   ├── sequence_features.py
│   ├── data_loader.py
│   ├── models.py
│   ├── importance.py
│   ├── visualise.py
│   └── run_pipeline.py
└── results/
```
---

## Updating paths 

To run pipeline update the path to where your datasets are stored. This can be found in scripts/config. 

"/Users/madhurakulkarni/Desktop/master_thesis/feature_engineering/data/Human_data.xlsx"<- update this for input directory for datasets 

"TE_HCT116" <--- cell line under consideration

"Human_HCT116" 

"/Users/madhurakulkarni/Desktop/master_thesis/feature_engineering/results/" <- update this for output directory for results

---

## Pipeline Overview

The entire workflow follows a sequence:

Data → Feature Engineering → Model Training → Evaluation → Visualisation

1. Data Loading

   * Raw data is read from Excel files.
   * Basic preprocessing and validation are performed.

2. Feature Engineering

   * Sequence data is parsed and transformed into numerical features.
   * Missing values are handled and datasets are prepared for modeling.

3. Model Training

   * Machine learning models like Lasso, Elastic net, Random forest and LGBM are defined and trained.
   * Cross-validation is used for evaluation.

4. Feature Importance

   * Importance scores are extracted from trained models.
   * Results are saved for interpretation.

5. Visualisation

   * Plots such as bar charts and heatmaps are generated for analysis.

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Run the complete workflow with:

```bash
python scripts/run_pipeline.py
```
---

## Module Responsibilities

| Module                 | Responsibility                                                           |
| ---------------------- | ------------------------------------------------------------------------ |
| `config.py`            | Central configuration (paths, constants, hyperparameters, plot settings) |
| `sequence_features.py` | Sequence parsing and feature engineering                                 |
| `data_loader.py`       | Data loading, feature generation call, preprocessing, imputation         |
| `models.py`            | Model definitions, training, and cross-validation                        |
| `importance.py`        | Feature importance extraction and export                                 |
| `visualise.py`         | All plots and visual outputs                                             |

---

## Output

After running the pipeline, you will get:

* Trained model results (cross-validation performance)
* Feature importance CSV files
* Visualisations (plots and heatmaps)



