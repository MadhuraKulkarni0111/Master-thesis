# Sequence Analysis Pipeline

All files and folders currently being used can be found in the "main" branch. Another branch named separate_functions is available on which work pertaining embeddings will be followed before merging them into the main branch again. 

The master_thesis branch consists of some initial code writting and can be ignored for now, git will be cleaned and updated (branched will be renamed) over the week making it easier to navigate. 

## Repository Structure

```text
.
├── requirements.txt
├── data/
│   ├── human_translation_efficiency_data.xlsx
|   ├── mouse_translation_efficiency_data.xlsx
|   ├── combined_tai_weights.csv
|   └── combined_cai_weights.csv
├── scripts/
│   ├── config.py
│   ├── sequence_features.py
│   ├── data_loader.py
│   ├── models.py
│   ├── importance.py
│   ├── visualise.py
│   ├── model_results.py
│   ├── run_pipeline.py
|   |── build_cai_weights.py ----\
|   |── build_tai_weights.py ----- > scripts for genrating the weights for the calculation of tai and cai
|   └──common_weights.py -------/
├── requirements.yaml
└── results/
```
---

## Updating paths 

To run pipeline update the path to where your datasets are stored. This can be found in scripts/config.py. 

"/Users/madhurakulkarni/Desktop/master_thesis/feature_engineering/data/Human_data.xlsx"<- update this for input directory for datasets 

"/Users/madhurakulkarni/Desktop/master_thesis/feature_engineering/results/"<- update this for output directory for results

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

A .yaml file is avaialbel with all of the dependencies required to run the entire pipeline to create the environment 
use the below comman:

```bash
conda activate master-thesis
```

## Running the Pipeline

To run the pipline it is first required to run the scripts build_cai_weights.py and build_tai_weights.py
for the mouse and human data which serveds as an input for further feature importance calculation. 

Run the complete workflow with:

```bash
python scripts/build_cai_weights.py
python scripts/build_tai_weights.py
```
Running the script build_cai_weights.py and build_tai_weights.py will generate a csv file in the folder data which acts as an input for cai and tai calculation in the main pipeline (utilised in scripts/sequwnce_features).

Then run the main script for training the supervised models: 

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
| `model_results.py`     | Outputs coefficient and cv results into a csv file                       |
| `build_cai_weights.py` | Creats a csv with wweights for calcualtion of codon adaption index       |
| `build_tai_weights.py` | Creats a csv with wweights for calcualtion of tRNA adaption index       |

---

## Output

After running the build_cai_weights and build_tai_weights, you will get:

* csv files for weights of humna and mouse datset
* NCBI dodnloaded database for the assembled genome
* combined csv file with both information of human and mouse weights (used as in input for cai calculation)

After running the build_tai_weights, you will get:

* combined csv file with both information of human and mouse weights (used as in input for tai calculation)

After running the pipeline, you will get:

* Trained model results (cross-validation performance)
* Feature importance CSV files
* Visualisations (plots and heatmaps)




