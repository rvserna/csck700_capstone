"""
Check whether selected ICD-9 Procedure codes are present in the raw
procedures table.

Purpose:
- Confirm that targeted clinical procedure codes appear in the
  PROCEDURES_ICD table.
- Count matching rows and admissions for each configured procedure
  group.
- Save one CSV output that can be reviewed before aggregation
  logic relies on it.

Broader Context:
- preprocessing_config.py provides the shared source table paths
  used here.
- data_preparation.py may depend on these procedure concepts through
  engineered features.
- This script checks code availability only and does not infer
  clinical meaning.
"""

from __future__ import annotations

import pandas as pd

from preprocessing_config import MIMIC_FILES, PREPROCESSING_OUTPUT_DIR

# --------------------------------------------------------------------
# Procedure Code Groups
# --------------------------------------------------------------------
# Store targeted procedure code groupings in one mapping so validation
# output shows which raw codes support each checked concept.
EXPECTED_PROCEDURE_CODES = {
    "mechanical_ventilation": ["9670", "9671", "9672"],
    "dialysis": ["3995", "5498"],
}

# --------------------------------------------------------------------
# Procedure Code Validation
# --------------------------------------------------------------------
def validate_procedure_codes() -> pd.DataFrame:
    """
    Validate whether configured procedure code groups appear in raw
    data.

    Returned DataFrame includes codes checked, matching row counts,
    unique admissions, and simple availability flag.
    """
    # Ready only fields needed for code presence check so script stays
    # lightwieght.
    df = pd.read_csv(
        MIMIC_FILES["PROCEDURES_ICD"],
        usecols=["HADM_ID", "ICD9_CODE"],
        compression="infer",
        dtype={"ICD9_CODE": str},
        low_memory=False,
    )

    rows = []

    # Evaluate each grouped concept separately so exported table is
    # easy to review and compare.
    for feature_name, codes in EXPECTED_PROCEDURE_CODES.items():
        subset = df[df["ICD9_CODE"].isin(codes)].copy()
        rows.append({
            "clinical_procedure": feature_name,
            "codes_checked": ",".join(codes),
            "matching_rows": int(len(subset)),
            "unique_admissions": int(subset["HADM_ID"].nunique()),
            "available": len(subset) > 0,
        })

    return pd.DataFrame(rows)

# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------
def main() -> None:
    """
    Run procedure code validation and save resulting CSV table.
    """
    # Reuse same output table for both console preview and saved CSV
    # export.
    out = validate_procedure_codes()
    out_path = (
        PREPROCESSING_OUTPUT_DIR
        / "validate_procedure_codes.csv"
    )
    out.to_csv(out_path, index=False)

    # Print validation table so diect runs provide immediate
    # visibility.
    print(out.to_string(index=False))
    print(f"\nSaved procedure code validation to: {out_path}")

if __name__ == "__main__":
    main()