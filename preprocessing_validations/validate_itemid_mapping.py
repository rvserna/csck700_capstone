"""
Validate whether the expected ITEMIDs are present in the raw event
tables.

Purpose:
- Check that each engineered feature still maps to observable raw
  ITEMIDs.
- Count matching rows for each expected ITEMID in CHARTEVENTS or
  LABEVENTS.
- Save one flat CSV table that can be reviewed before feature
  aggregation.

Broader Context:
- preprocessing_config.py stores the expected ITEMID mapping used
  here.
- data_preparation.pu relies on these raw ITEMID links when building
  features.
- This script checks raw presence only; it does not judge value
  plausibility.
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd

from preprocessing_config import (
    EXPECTED_ITEMID_MAP,
    MIMIC_FILES,
    PREPROCESSING_OUTPUT_DIR,
)

# --------------------------------------------------------------------
# Source Table Grouping
# --------------------------------------------------------------------
# Separate lab derived features from chart derived features so each
# ITEMID is checked against raw event table where it is expected to
# appear.
LAB_FEATURES = {
    "wbc_max",
    "creatinine_max",
    "bilirubin_max",
    "platelet_min",
    "glucose_max",
}

# --------------------------------------------------------------------
# DuckDB Helper
# --------------------------------------------------------------------
def _presence_counts(path: str, itemids: list[int]) -> dict[int, int]:
    """
    Count matching rows for each requested ITEMID in one raw event
    table.

    DuckDB is used here so check can query CSV directly without
    loading full source table into pandas first.
    """
    # Convert file path to DuckDB compatible string because these
    # support scripts may run on Windows-style project paths.
    path = Path(path).as_posix()
    path = path.replace("'", "''")
    itemids_str = ", ".join(str(int(i)) for i in itemids)

    # Group by ITMEID so output preserves one row count per expected
    # code.
    query = f"""
    SELECT
        ITEMID,
        COUNT(*) AS row_count
    FROM read_csv_auto('{path}')
    WHERE ITEMID IN ({itemids_str})
    GROUP BY ITEMID
    ORDER BY ITEMID
    """

    df = duckdb.sql(query).df()

    # Convert query output to dictionary first, then backfill zeros for
    # expected ITEMIDs that were not returned by query.
    found ={
        int(row["ITEMID"]): int(row["row_count"])
        for _, row in df.iterrows()
    }
    return {int(i): int(found.get(int(i), 0)) for i in itemids}

# --------------------------------------------------------------------
# ITEMID Mapping Validation
# --------------------------------------------------------------------
def map_expected_itemids() -> pd.DataFrame:
    """
    Validate configured ITEMID mapping against raw event tables.

    Returned DataFrame contains one row per feature-ITEMID combination
    and records source table, observed row count, and simple
    availability flag.
    """
    rows = []
    chart_path = MIMIC_FILES["CHARTEVENTS"]
    lab_path = MIMIC_FILES["LABEVENTS"]

    # Check every configured feature mapping so saved table reflects
    # full set of ITEMIDs expected by current preprocessing workflow.
    for feature_name, itemids in EXPECTED_ITEMID_MAP.items():
        source_table = (
            "LABEVENTS"
            if feature_name in LAB_FEATURES
            else "CHARTEVENTS"
        )
        path = lab_path if source_table == "LABEVENTS" else chart_path
        counts = _presence_counts(path, itemids)

        # Expand grouped counts back to one row per ITEMID because
        # that format is easier to inspect and export.
        for itemid, n in counts.items():
            rows.append({
                "feature_name": feature_name,
                "source_table": source_table,
                "itemid": itemid,
                "row_count": n,
                "available": n > 0,
            })
 
    return (
        pd.DataFrame(rows)
        .sort_values(["feature_name", "itemid"])
        .reset_index(drop=True)
    )

# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------
def main() -> None:
    """
    Run ITEMID mapping validation and save resulting CSV table.
    """
    # Build validation table once so console preview and saved output
    # stay in sync.
    df = map_expected_itemids()
    out_path = PREPROCESSING_OUTPUT_DIR / "itemid_mapping_check.csv"
    df.to_csv(out_path, index=False)

    # Print saved output for immediate review during direct script
    # runs.
    print(df.to_string(index=False))
    print(f"\nSaved ITEMID mapping check to: {out_path}")

if __name__ == "__main__":
    main()