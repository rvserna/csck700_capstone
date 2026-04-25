"""
Check observed raw VALUE ranges for selected ITEMIDs before feature
aggregation.

Purpose:
- Summarise the observed minimum and maximum values for selected
  ITEMIDs.
- Count values that fall below or above the configured plausibility
  bands.
- Save one CSV output that supports early raw data quality review.

Broader Context:
- preprocessing_config.py stores the ITEMID mapping and plausible
  ranges used here.
- data_preparation.py ater aggregates these raw values into
  admission-level features.
- This script focuses on raw value checks rather than linkage or
  schema validation.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import duckdb

from preprocessing_config import (
    EXPECTED_ITEMID_MAP,
    MIMIC_FILES,
    PLAUSIBLE_RANGES,
    PREPROCESSING_OUTPUT_DIR,
)

# --------------------------------------------------------------------
# Shared Range Mapping Constants
# --------------------------------------------------------------------
# Map engineered feature names to range keys used in shared
# plausibility configuration so validation loop can stay simple.
FEATURE_TO_PLAUSIBLE_RANGE = {
    "wbc_max": "wbc_max",
    "creatinine_max": "creatinine_max",
    "bilirubin_max": "bilirubin_max",
    "platelet_min": "platelet_min",
    "glucose_max": "glucose_max",
    "heart_rate_mean": "heart_rate_mean",
    "resp_rate_mean": "resp_rate_mean",
    "spo2_min": "spo2_min",
    "mean_bp": "mean_bp",
    "temp_max": "temp_max",
}

# Separate lab derived features from chart derived features so each
# check runs against correct raw source table.
LAB_FEATURES = {
    "wbc_max", 
    "creatinine_max", 
    "bilirubin_max", 
    "platelet_min", 
    "glucose_max"
}

# --------------------------------------------------------------------
# DuckDB Helpers
# --------------------------------------------------------------------
def _duckdb_path(path) -> str:
    """
    Convert filesystem path to DuckDB compatible string.
    """
    return Path(path).as_posix().replace("'", "''")

def _range_summary(path, itemids) -> dict:
    """
    Return compact observed value summary for one ITEMID group.

    Contains number of non-null values plus observed minimum and
    maximum values returned by DuckDB.
    """
    path = _duckdb_path(path)
    itemids_str = ", ".join(str(int(i)) for i in itemids)

    # Query CSV directly so summary can be generated without loading
    # full event table into pandas.
    query = f"""
    SELECT
        COUNT(VALUENUM) AS n_values,
        MIN(VALUENUM) AS min_observed,
        MAX(VALUENUM) AS max_observed,
    FROM read_csv_auto('{path}')
    WHERE ITEMID IN ({itemids_str})
        AND VALUENUM IS NOT NULL
    """

    row = duckdb.sql(query).df().iloc[0].to_dict()
    return row

def _out_of_range_counts(path, itemids, lo, hi) -> tuple[int, int]:
    """
    Count observed values below and above configured plausibility
    limits.
    """
    path = _duckdb_path(path)
    itemids_str = ", ".join(str(int(i)) for i in itemids)

    # Keep counting logic in SQL so raw event table does not need to
    # be materialised in memory for this support check.
    query = f"""
    SELECT
        SUM(CASE WHEN VALUENUM < {lo} THEN 1 ELSE 0 END) AS below_min,
        SUM(CASE WHEN VALUENUM > {hi} THEN 1 ELSE 0 END) AS above_max,
    FROM read_csv_auto('{path}')
    WHERE ITEMID IN ({itemids_str})
        AND VALUENUM IS NOT NULL
    """

    row = duckdb.sql(query).df().iloc[0]
    below_min = (
        int(row["below_min"])
        if pd.notna(row["below_min"])
        else 0
    )
    above_max = (
        int(row["above_max"])
        if pd.notna(row["above_max"])
        else 0
    )
    return below_min, above_max

# --------------------------------------------------------------------
# Value Range Validation
# --------------------------------------------------------------------
def validate_value_ranges() -> pd.DataFrame:
    """
    Validate observed raw ranges for configured ITEMID groups.
    
    Returned table records source table, ITEMIDs checked, observed
    value summary, and counts outside configured plausibility limits.
    """
    rows = []

    # Review each configured feature mapping so exported table refects
    # same ITEMID set expected by preprocessing workflow.
    for feature_name, itemids in EXPECTED_ITEMID_MAP.items():
        # Skip event count feature because its support check is based
        # on raw row presence rather than numeric VALUENUM
        # plausibility.
        if feature_name == "ventilator_event_count":
            continue

        source_table = (
            "LABEVENTS"
            if feature_name in LAB_FEATURES
            else "CHARTEVENTS"
        )
        source_path = MIMIC_FILES[source_table]

        # Pull configured plausibility band and observed raw summary
        # for current ITEMID group.
        lo, hi = PLAUSIBLE_RANGES[
            FEATURE_TO_PLAUSIBLE_RANGE[feature_name]
        ]
        summary = _range_summary(source_path, itemids)
        below_min, above_max = _out_of_range_counts(
            source_path,
            itemids,
            lo,
            hi,
        )

        rows.append(
            {
                "feature_name": feature_name,
                "source_table": source_table,
                "itemids": ",".join(str(i) for i in itemids),
                "n_values": (
                    int(summary["n_values"])
                    if pd.notna(summary["n_values"])
                    else 0
                ),
                "below_min": below_min,
                "above_max": above_max,
                "min_observed": (
                    None
                    if pd.isna(summary["min_observed"])
                    else float(summary["min_observed"])
                ),
                "max_observed": (
                    None
                    if pd.isna(summary["max_observed"])
                    else float(summary["max_observed"])
                ),
                "plausible_min": lo,
                "plausible_max": hi,
            }
        )

    return pd.DataFrame(rows)

# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------
def main() -> None:
    """
    Run value range validation and save results CSV table.
    """
    # Generate validation table once so saved file and printed
    # previous stay aligned.
    df = validate_value_ranges()
    out_path = PREPROCESSING_OUTPUT_DIR / "validate_value_ranges.csv"
    df.to_csv(out_path, index=False)

    # Print table so direct script runs provide immediate review view.
    print(df.to_string(index=False))
    print(f"\nSaved raw value range validation to: {out_path}")

if __name__ == "__main__":
    main()