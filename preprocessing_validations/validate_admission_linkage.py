"""
Validate linkeage consistency across core MIMIC source tables.

Purpose:
- Verify SUBJECT_ID and HADM_ID relationships are consistent before
  feature engineering.
- Identify missing or mismatched IDs that could break joins downsteam.
- Provide a simple summary of linkage coverage for quick review.

Broader Context:
- This script supports data_preparation.py by verifying that the
  underlying relational structure of the raw data is intact.
- All downstream aggregation assumes that admissions, ICU stays,
  diagnoses, and procedures correctly map back to valid patients
  and admissions.
"""

from __future__ import annotations

import json
import pandas as pd

from preprocessing_config import MIMIC_FILES, PREPROCESSING_OUTPUT_DIR

# --------------------------------------------------------------------
# Linkag Validation Helper
# --------------------------------------------------------------------
def _read_ids(path, cols):
    """
    Read only identifier columns required for linkage checks.

    Limiting read to small subset of columns keeps script fast and
    avoids loading unnecessary data from large source tables.
    """
    return pd.read_csv(
        path, 
        usecols=cols, 
        compression="infer", 
        low_memory=False,
    )

def validate_linkage() -> dict:
    """
    Build summary of identifier linkage across core tables.

    Returns dictionary with counts of valid mappings and
    any detected mismatches.
    """
    # Read only required identifier fields from each table.
    patients = _read_ids(
        MIMIC_FILES["PATIENTS"],
        ["SUBJECT_ID"],
    )
    admissions = _read_ids(
        MIMIC_FILES["ADMISSIONS"],
        ["SUBJECT_ID", "HADM_ID"],
    )
    icu = _read_ids(
        MIMIC_FILES["ICUSTAYS"],
        ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"],
    )
    diagnoses = _read_ids(
        MIMIC_FILES["DIAGNOSES_ICD"],
        ["SUBJECT_ID", "HADM_ID"],
    )
    procedures = _read_ids(
        MIMIC_FILES["PROCEDURES_ICD"],
        ["SUBJECT_ID", "HADM_ID"],
    )

    # Build reference ID sets used to check linkage consistency.
    patient_ids = set(patients["SUBJECT_ID"].dropna().unique())
    hadm_ids = set(admissions["HADM_ID"].dropna().unique())

    # Assemble compact summary of linkage coverage so results
    # can be quickly reviewed and saved as simple JSON report.
    report = {
        # Core dataset size indicators for quick sanity checking.
        "n_patients": int(len(patient_ids)),
        "n_admissions": int(len(hadm_ids)),
        "n_icu_stays": int(icu["ICUSTAY_ID"].nunique()),

        # Identify records that do not map correctly across tables.
        "admissions_missing_patient": int(
            (~admissions["SUBJECT_ID"].isin(patient_ids)).sum()
        ),
        "icu_missing_admission": int(
            (~icu["HADM_ID"].isin(hadm_ids)).sum()
        ),
        "diagnosis_missing_admission": int(
            (~diagnoses["HADM_ID"].isin(hadm_ids)).sum()
        ),
        "procedure_missing_admission": int(
            (~procedures["HADM_ID"].isin(hadm_ids)).sum()
        ),
        
        # Check for duplicate identifiers that may affect aggregation
        # logic.
        "duplicate_admissions": int(
            admissions.duplicated(subset=["HADM_ID"]).sum()
        ),
        "duplicate_icu_stays": int(
            icu.duplicated(subset=["ICUSTAY_ID"]).sum()
        ),
    }

    return report

# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------
def main() -> None:
    """
    Run linkage validation and save summary output.
    """
    # Generate linkage report once so same results are reused
    # for both file output and console display.
    report = validate_linkage()
    out_path = (
        PREPROCESSING_OUTPUT_DIR
        / "validate_admission_linkage.json"
    )
    
    # Save report as JSON file for later inspection.
    out_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    
    # Print results so direct run provides immediate feedback.
    print(json.dumps(report, indent=2))
    print(f"\nSaved admission linkage validation to: {out_path}")

if __name__ == "__main__":
    main()