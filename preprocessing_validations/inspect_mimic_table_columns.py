"""
Inspect raw MIMIC table schemas before downstream preprocessing
begins.

Purpose:
- Check whether each expected raw table is present.
- Record the visible column names for a lightweight schema review.
- Save one compact CSV summary that can be reviewed before feature
  building.

Broader Context:
- preprocessing_config.py defines the file paths used here.
- data_preparation.py relies on these source tables having the
  expected structure.
- The validation scripts in this folder use the same paths and
  raw-table assumptions.
"""
from __future__ import annotations

import argparse

import pandas as pd

from preprocessing_config import MIMIC_FILES, PREPROCESSING_OUTPUT_DIR

# --------------------------------------------------------------------
# Schema Inspection Helper
# --------------------------------------------------------------------
def inspect_tables(samples_rows: int = 5) -> pd.DataFrame:
    """
    Inspect each configured raw table and return compact schema
    summary.

    Output records whether file exists, where it is located, how many
    columns were observed, and how many preview rows were read
    successfully.
    """
    rows = []

    # Check each configured source table in one loop so summary stays
    # aligned with central preprocessing configuration.
    for table_name, path in MIMIC_FILES.items():
        # Record missing files explicitly so output still shows which
        # table needs attention instead of failing without context.
        if not path.exists():
            rows.append({
                "table": table_name,
                "exists": False,
                "path": str(path),
                "n_columns": None,
                "columns": None,
                "sample_rows_read": 0,
            })
            continue

        # Read only small preview because script is intended to
        # confirm structure, not to process full raw dataset.
        df = pd.read_csv(
            path, 
            nrows=samples_rows, 
            compression="infer", 
            low_memory=False,
        )

        # Store column names as one text field so saved CSV remains
        # easy to scan in spreadsheet view.
        rows.append({
            "table": table_name,
            "exists": True,
            "path": str(path),
            "n_columns": len(df.columns),
            "columns": ", ".join(df.columns.tolist()),
            "sample_rows_read": len(df),
        })

    return pd.DataFrame(rows)

# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------
def main() -> None:
    """
    Run schema inspection script and save resulting summary table.
    """
    # Keep command line interface minimal because script only needs
    # small preview row control for quick checks.
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rows", type=int, default=5)
    args = parser.parse_args()

    # Generate inspection table first, then reuse it for both console
    # output and saved CSV export.
    out_df = inspect_tables(samples_rows=args.sample_rows)
    out_path = PREPROCESSING_OUTPUT_DIR / "mimic_table_columns.csv"
    out_df.to_csv(out_path, index=False)

    # Print summary so direct script run gives immediate feedback.
    print(out_df.to_string(index=False))
    print(f"\nSaved table schema summary to: {out_path}")

if __name__ == "__main__":
    main()