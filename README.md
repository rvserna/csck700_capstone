# Benchmarking Explainability in Healthcare AI
### MSc Data Science & Artificial Intelligence - Capstone Project

## Overview
This project presents a reproducible benchmarking framework for Explainable AI (XAI) in healthcare. It evaluates explanation methods across the following dimensions:
1. Fidelity (alignment with model behaviour)
2. Robustness (stability under noisy and incomplete data)
3. Clinical alignment (agreement with clinically validated features)

The system is implemented as a Python-based pipeline and Streamlit dashboard, using the MIMIC-III clinical dataset.

## Key Features
1. End-to-end benchmarking pipeline
2. Comparison of Captum Integrated Gradients (baseline), SHAP, and LIME
3. Robustness testing via noise perturbation and feature masking
4. Clinical validation using a fixed set of clinically meaningful variables
5. Interactive Streamlit dashboard
6. Exportable outputs (CSV results, figures, PDF reports)

## Project Structure
```text
.
└── project-root/
    ├── README.md
    ├── requirements.txt
    ├── LICENSE.txt
    ├── data/
    │   ├── mimic-iii/
    │   │   ├── ADMISSIONS.csv
    │   │   ├── CHARTEVENTS.csv
    │   │   ├── DIAGNOSES_ICD.csv.gz
    │   │   ├── ICUSTAYS.csv
    │   │   ├── LABEVENTS.csv
    │   │   ├── PATIENTS.csv
    │   │   └── PROCEDURES_ICD.csv.gz
    │   └── joined_agg_dataset/
    │       └── model_features.parquet
    ├── preprocessing_validations/
    │   ├── inspect_mimic_table_columns.py
    │   ├── preprocessing_config.py
    │   ├── raw_dataset_summary.py
    │   ├── validate_admission_linkage.py
    │   ├── validate_itemid_mapping.py
    │   ├── validate_mimic_tables.py
    │   ├── validate_procedure_codes.py
    │   └── validate_value_ranges.py
    ├── xai_pipeline/
    │   ├── data_preparation.py
    │   ├── model_development.py
    │   ├── explanation_engine.py
    │   ├── benchmarking_engine.py
    │   ├── clinical_alignment.py
    │   └── reporting_interface.py
    └── outputs/
        ├── benchmark_run_config.json
        ├── comparative_top10_importance.png
        ├── interpretation_summary.txt
        ├── comparative_module_metrics_master.csv
        ├── explainer_agreement.csv
        ├── feature_importance.csv
        ├── method_summary.csv
        ├── model_comparison_summary.csv
        ├── pairwise_stats.csv
        └── preprocessing_checks/
            ├── itemid_mapping_check.csv
            ├── mimic_table_columns.csv
            ├── raw_dataset_summary_admission_type.txt
            ├── raw_dataset_summary_overall.csv
            ├── validate_admission_linkage.json
            ├── validate_mimic_tables.json
            ├── validate_procedure_codes.csv
            └── validate_value_ranges.csv
```

## Installation
### Required steps:
1. Clone the repository
2. Install dependencies:
   ```pip install -r requirements.txt```

### Core dependencies:
1. numpy
2. pandas
3. scikit-learn
4. torch
5. shap
6. lime
7. streamlit
8. duckdb
9. reportlab

## MIMIC-III Data Setup
Note: this project does not include access to MIMIC-III data.

### Access Requirements
To gain access to MIMIC-III data, complete the following:
1. Complete CITI training: https://physionet.org/content/mimiciii/view-required-training/1.4/#1
2. Request access via: https://physionet.org/content/mimiciii

### Required Files
Place the following files in the ```data/mimic-iii/``` folder.

1. ADMISSIONS.csv
2. CHARTEVENTS.csv
3. DIAGNOSES_ICD.csv.gz
4. ICUSTAYS.csv
5. LABEVENTS.csv
6. PATIENTS.csv
7. PROCEDURES_ICD.csv.gz

### Notes:
1. Data is de-identified but still controlled
2. Do not upload MIMIC data to GitHub
3. Ensure file paths match those in data_preparation.py

## Data Preparation
Run the following: 
```python xai_pipeline/data_preparation.py```

This will:
1. Load MIMIC tables via DuckDB
2. Aggregate to admission-level features
3. Output:
   ```data/joined_agg_dataset/model_features.parquet```

## Running the Benchmark
### Option 1: Direct script
Run the following: 
```python xai_pipeline/benchmarking_engine.py```

#### Outputs include:
1. Method comparison tables
2. Feature importance
3. Agreement metrics
4. Pairwise statistical tests

Outputs are saved in the 'outputs/' folder.

### Option 2: Streamlit Dashboard
Run the following: 
```streamlit run xai_pipeline/reporting_interface.py```

#### Features:
1. Configure experiment parameters
2. Run benchmark interactively
3. Visualise:
   - Stability
   - Fidelity
   - Clinical alignment
4. Export PDF and CSV

## Core Methodology

### Model
- Baseline: MLP neural network
- Comparators:
  - Logistic Regression
  - Random Forest

### Explanation Methods
- Integrated Gradients (Captum)
- SHAP (GradientExplainer)
- LIME (Tabular)

### Evaluation Dimensions
1. #### Fidelity
   - Spearman correlation with:
     - Permutation importance
     - Gradient-based importance
2. #### Robustness
   - Noise perturbation
   - Feature masking
   - Stability metrics
     - Rank correlation
     - Top-K overlap
3. #### Clinical Alignment
   - Compared against 14 clinically validated variables
   - Metrics:
     - Top-K overlap
     - Weighted overlap
     - Precision
     - Recall
    
## Outputs
### Files:
- ```method_summary.csv```
  - Main comparison table
- ```feature_importance.csv```
  - Feature-level importance
- ```explainer_agreement.csv```
  - Method agreement
- ```pairwise_stats.csv```
  - Statistical tests
- ```model_comparison_summary.csv```
  - Model benchmarking
- ```comparative_top_10_importance.png```
  - Visualisations
 
## Known Constraints
- MIMIC access required
- Runtime can be long
- LIME results may vary due to stochastic sampling
- Designed for single prediction task benchmarking

## Author
Robert Viens Serna

MSc Data Science & Artificial Intelligence
