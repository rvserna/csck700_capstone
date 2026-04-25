"""
Produce a concise summary of the raw MIMIC source tables before
feature construction.

Purpose:
- Report the scale of the main raw tables used downstream.
- Summarise the mortality label prevalence from the admissions
  table.
- Save a compact admission-type distribution for quick review.

Broader Context:
- preprocessing_config.py defines the raw file paths reused here.
- data_preperation.py performs the full feature engineering that
  follows this check.
- These outputs are intended as lightweight diagnostics rather than
  benchmark inputs.
"""

from __future__ import annotations

import pandas as pd

from preprocessing_config import MIMIC_FILES, PREPROCESSING_OUTPUT_DIR

# --------------------------------------------------------------------
# Raw Dataset Summary Helper
# --------------------------------------------------------------------
def raw_dataset_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build overall and admission-type summaries from raw source tables.

    First returned DataFrame contains one row dataset totals. Second
    returned DataFrame contains admission-type frequency table.
    """
    # Only read columns needed for this summary so script stays
    # focused on high level diagnostics rather than full table
    # processing.
    patients = pd.read_csv(
        MIMIC_FILES["PATIENTS"], 
        usecols=["SUBJECT_ID"], 
        compression="infer", 
        low_memory=False,
    )
    admissions = pd.read_csv(
        MIMIC_FILES["ADMISSIONS"],
        usecols=[
            "SUBJECT_ID",
            "HADM_ID",
            "HOSPITAL_EXPIRE_FLAG",
            "ADMISSION_TYPE",
        ],
        compression="infer",
        low_memory=False
    )
    icu = pd.read_csv(
        MIMIC_FILES["ICUSTAYS"], 
        usecols=["HADM_ID", "ICUSTAY_ID"], 
        compression="infer", 
        low_memory=False,
    )
    diagnoses = pd.read_csv(
        MIMIC_FILES["DIAGNOSES_ICD"], 
        usecols=["HADM_ID", "ICD9_CODE"], 
        compression="infer", 
        low_memory=False,
    )
    procedures = pd.read_csv(
        MIMIC_FILES["PROCEDURES_ICD"], 
        usecols=["HADM_ID", "ICD9_CODE"], 
        compression="infer", 
        dtype={"ICD9_CODE": str}, 
        low_memory=False,
    )

    # Assemble one row of dataset-wide counts so main summary can be
    # saved as simple CSV table and reviewed quickly.
    overall = pd.DataFrame(
        [
            {
                "n_patients": int(patients["SUBJECT_ID"].nunique()),
                "n_admissions": int(admissions["HADM_ID"].nunique()),
                "n_icu_stays": int(icu["ICUSTAY_ID"].nunique()),
                "mortality_rate_pct": round(
                    float(
                        pd.to_numeric(
                            admissions["HOSPITAL_EXPIRE_FLAG"],
                            errors="coerce",
                        ).mean()
                    )
                    * 100,
                    2,
                ),
                "mean_diagnoses_per_admission": round(
                    float(diagnoses.groupby("HADM_ID").size().mean()),
                    2,
                ),
                "mean_procedures_per_admission": round(
                    float(diagnoses.groupby("HADM_ID").size().mean()),
                    2,
                ),
            }
        ]
    )

    # Break out admission type separately because that distribution
    # is easier to review in its own frequency table.
    admission_type = (
        admissions["ADMISSION_TYPE"]
        .value_counts(dropna=False)
        .rename_axis("admission_type")
        .reset_index(name="count")
    )
    return overall, admission_type

# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------
def main() -> None:
    """
    Run the raw dataset summary script and save both summary outputs.
    """
    # Build both summary tables once so same results are reused for
    # export and for console preview.
    overall, admission_type = raw_dataset_summary()
    overall_path = (
        PREPROCESSING_OUTPUT_DIR
        / "raw_dataset_summary_overall.csv"
    )
    type_path = (
        PREPROCESSING_OUTPUT_DIR
        / "raw_dataset_summary_admission_type.csv"
    )
    overall.to_csv(overall_path, index=False)
    admission_type.to_csv(type_path, index=False)

    # Print saved summaries so direct run gives immediate confirmation
    # of what was written to disk.
    print(overall.to_string(index=False))
    print("\nAdmission type distribution:")
    print(admission_type.to_string(index=False))
    print(f"\nSaved overall raw summary to: {overall_path}")
    print(f"Saved admission-type summary to: {type_path}")

if __name__ == "__main__":
    main()