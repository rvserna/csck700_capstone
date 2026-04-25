"""
Shared configuration used by preprocessing and validation support
scripts.

Purpose:
- Define the common project paths used by the raw data support
  utilities.
- Store the expected raw table names, feature lists, and valiadation
  mappings.
- Keep preprocessing checks aligned with the same schema used
  elsewhere.

Broader Context:
- data_preparation.pu build the model-ready feature table from these
  paths and mappings.
- The validation and inspection scripts reuse these constants to
  avoid local duplicates.
- clinical_alignment.py and benchmarking_engine.py depend indirectly
  on the same feature names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

# --------------------------------------------------------------------
# Project Paths
# --------------------------------------------------------------------
# Resolve shared project directories relative to this file so support
# scripts can be moved together without rewriting hard coded absolute
# paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MIMIC_DIR = DATA_DIR / "mimic-iii"
JOINED_DIR = DATA_DIR / "joined_agg_dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PREPROCESSING_OUTPUT_DIR = OUTPUT_DIR / "preprocessing_checks"

# Create key output folders up front so helper scripts can write their
# summaries without repeating same directory setup logic.
for _p in [JOINED_DIR, OUTPUT_DIR, PREPROCESSING_OUTPUT_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Raw Source Files and Shared Dataset Paths
# --------------------------------------------------------------------
# Keep all expected raw MIMIC file locations in one mapping so
# validation scripts and inspection helpers reference same source
# names.
MIMIC_FILES: Dict[str, Path] = {
    "ADMISSIONS": MIMIC_DIR / "ADMISSIONS.csv",
    "DIAGNOSES_ICD": MIMIC_DIR / "DIAGNOSES_ICD.csv.gz",
    "ICUSTAYS": MIMIC_DIR / "ICUSTAYS.csv",
    "LABEVENTS": MIMIC_DIR / "LABEVENTS.csv",
    "PATIENTS": MIMIC_DIR / "PATIENTS.csv",
    "PROCEDURES_ICD": MIMIC_DIR / "PROCEDURES_ICD.csv.gz",
    "CHARTEVENTS": MIMIC_DIR / "CHARTEVENTS.csv",
}

# Point to model-ready feature table used elsewhere in pipeline so
# support scripts can reference it consistently.
MODEL_FEATURES_PATH = JOINED_DIR / "model_features.parquet"

# --------------------------------------------------------------------
# Shared Modelling Schema
# --------------------------------------------------------------------
# Store canonical feature list so support scripts can validate or
# summarise same fields used by model pipeline.
MODEL_FEATURES: List[str] = [
    "age_at_admission",
    "gender",
    "icd_count",
    "procedure_icd_count",
    "procedure_row_count",
    "lab_count",
    "chart_count",
    "lab_avg",
    "lab_min",
    "lab_max",
    "chart_avg",
    "chart_min",
    "chart_max",
    "heart_rate_mean",
    "resp_rate_mean",
    "spo2_min",
    "mean_bp",
    "temp_max",
    "wbc_max",
    "creatinine_max",
    "bilirubin_max",
    "platelet_min",
    "glucose_max",
    "ventilator_event_count",
]

# Keep target label name centralised so support code does not depend
# on repeated string literal.
TARGET_COLUMN = "hospital_expire_flag"

#---------------------------------------------------------------------
# Clinical Reference Subset
# --------------------------------------------------------------------
# Store clinically reviewed subset separately because several checks
# focus on these variables rather than full feature list.
CLINICALLY_VALIDATED_FEATURES: Set[str] = {
    "age_at_admission",
    "icd_count",
    "procedure_icd_count",
    "heart_rate_mean",
    "resp_rate_mean",
    "spo2_min",
    "mean_bp",
    "temp_max",
    "wbc_max",
    "creatinine_max",
    "bilirubin_max",
    "platelet_min",
    "glucose_max",
    "ventilator_event_count",
}

# --------------------------------------------------------------------
# Required Raw Table columns
# --------------------------------------------------------------------
# Define minimum expected columns for each raw table so validation
# scripts can report missing structure in one consistent way.
REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "PATIENTS": ["SUBJECT_ID", "GENDER", "DOB"],
    "ADMISSIONS": [
        "SUBJECT_ID",
        "HADM_ID",
        "ADMITTIME",
        "DISCHTIME",
        "HOSPITAL_EXPIRE_FLAG",
    ],
    "ICUSTAYS": [
        "SUBJECT_ID",
        "HADM_ID",
        "ICUSTAY_ID",
        "INTIME",
        "OUTTIME",
    ],
    "DIAGNOSES_ICD": ["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
    "PROCEDURES_ICD": ["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
    "LABEVENTS": ["SUBJECT_ID", "HADM_ID", "ITEMID", "VALUENUM"],
    "CHARTEVENTS": ["SUBJECT_ID", "HADM_ID", "ITEMID", "VALUENUM"],
}

# --------------------------------------------------------------------
# ITEMID Mappings Used by Preprocessing Checks
# --------------------------------------------------------------------
# Map each engineered variable to ITEMIDs expected by current feature
# construction workflow so raw table checks can verify coverage.
EXPECTED_ITEMID_MAP = {
    "wbc_max": [51300],
    "creatinine_max": [50912],
    "bilirubin_max": [50885],
    "platelet_min": [51265],
    "glucose_max": [50931],
    "heart_rate_mean": [211],
    "resp_rate_mean": [618],
    "spo2_min": [646],
    "mean_bp": [456],
    "temp_max": [223761, 678],
    "ventilator_event_count": [720, 721],
}

# --------------------------------------------------------------------
# Plausible Value Ranges Used in Support Checks
# --------------------------------------------------------------------
# Store numeric plausibility bands in one place so range validation
# scripts can spply same thresholds consistently.
PLAUSIBLE_RANGES = {
    "age_at_admission": (15, 110),
    "icd_count": (0, 100),
    "procedure_icd_count": (0, 100),
    "procedure_row_count": (0, 150),
    "lab_count": (0, 20000),
    "chart_count": (0, 500000),
    "heart_rate_mean": (20, 220),
    "resp_rate_mean": (4, 80),
    "spo2_min": (30, 100),
    "mean_bp": (20, 200),
    "temp_max": (30, 43.5),
    "wbc_max": (0.1, 400),
    "creatinine_max": (0.1, 30),
    "bilirubin_max": (0.0, 60),
    "platelet_min": (1, 1500),
    "glucose_max": (20, 1500),
    "ventilator_event_count": (0, 20000),
}

# --------------------------------------------------------------------
# Compact Reporting Subset
# --------------------------------------------------------------------
# Keep smaller summary list for support outputs that focus on main
# clinically oriented variables rather than every engineered feature.
SUMMARY_FEATURES = [
    "age_at_admission",
    "icd_count",
    "procedure_icd_count",
    "heart_rate_mean",
    "resp_rate_mean",
    "spo2_min",
    "mean_bp",
    "temp_max",
    "wbc_max",
    "creatinine_max",
    "bilirubin_max",
    "platelet_min",
    "glucose_max",
    "ventilator_event_count",
]