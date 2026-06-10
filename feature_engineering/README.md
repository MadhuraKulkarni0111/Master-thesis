# Sequence Analysis Pipeline

## Repository Structure

This project is organized into three branches:

* main – Stable production version
* code_updates – Ongoing development changes
* separate_functions – Modularized pipeline (this branch contains individual scripts)

The pipeline implementation in this branch is fully modular and follows a structured workflow from raw data to final outputs.

---

## Pipeline Overview

The entire workflow follows a sequence:

Data → Feature Engineering → Model Training → Evaluation → Visualisation**

1. Data Loading

   * Raw data is read from Excel files.
   * Basic preprocessing and validation are performed.

2. Feature Engineering

   * Sequence data is parsed and transformed into numerical features.
   * Missing values are handled and datasets are prepared for modeling.

3. Model Training

   * Machine learning models are defined and trained.
   * Cross-validation is used for evaluation.

4. Feature Importance

   * Importance scores are extracted from trained models.
   * Results are saved for interpretation.

5. Visualisation

   * Plots such as bar charts and heatmaps are generated for analysis.

---

## How to Run the Pipeline

```bash
python run_pipeline.py
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
