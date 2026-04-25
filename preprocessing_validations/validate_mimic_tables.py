"""
Validate the basic structure and sampled content of the raw MIMCI
source tables.

Purpose:
- Check whether each configured raw table exists.
- Verify expected columns and sample a manageable subset for quick
  structual checks.
- Save one JSON summary covering missing columns, null counts,
  timestamps, and values.

Broader Context:
- preprocessing_config.py defines the expected raw files and
  required columns.
- data_preparation.py depends on these source tables being
  structurally usable.
- This script focuses on general table quality rather than
  feature-specific mappings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from preprocessing_config import (
    MIMIC_FILES,
    REQUIRED_COLUMNS,
    PREPROCESSING_OUTPUT_DIR,
)

# --------------------------------------------------------------------
# Shared Validation Constants
# --------------------------------------------------------------------
# Mark largest event tables separately so they can be sampled in
# chunks rather than being read in one call.
LARGE_TABLES = {"CHARTEVENTS", "LABEVENTS"}
KEY_COLUMNS = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]
TIME_COLUMNS = ["ADMITTIME", "DISCHTIME", "INTIME", "OUTTIME"]
VALUE_COLUMNS = ["VALUENUM"]

# --------------------------------------------------------------------
# Read Helpers
# --------------------------------------------------------------------
def _read_header(path: Path) -> list[str]:
    """
    Read only header row for one source table.

    This keeps required column check lightweight because it does not
    need to sample any data rows.
    """
    if path.suffix == ".gz":
        return pd.read_csv(
            path,
            compression="gzip",
            nrows=0,
        ).columns.tolist()

    return pd.read_csv(path, nrows=0).columns.tolist()

def _needed_columns(
    table_name: str,
    available_columns: list[str],
) -> list[str]:
    """
    Return subset of columns needed for sampled validation checks.

    Combines table-specific required fields with shared identifier,
    timestap, and numeric value columns when they are available.
    """
    wanted = set(REQUIRED_COLUMNS.get(table_name, []))
    wanted.update(KEY_COLUMNS)
    wanted.update(TIME_COLUMNS)
    wanted.update(VALUE_COLUMNS)
    return [c for c in available_columns if c in wanted]

def _read_sample(
    path: Path,
    usecols: list[str],
    nrows: int,
) -> pd.DataFrame:
    """
    Read sampled slice from standard size source table.
    """
    if path.suffix == ".gz":
        return pd.read_csv(
            path,
            compression="gzip",
            usecols=usecols,
            nrows=nrows,
            engine="python",
        )
    return pd.read_csv(
        path,
        usecols=usecols,
        nrows=nrows,
        engine="python",
    )

def _read_large_sample(
    path: Path,
    usecols: list[str],
    nrows: int,
) -> pd.DataFrame:
    """
    Read sampled slide from large source table using chunks.

    Chunked reading keeps validation script practical for largest
    event tables while still returning a normal pandas DataFrame
    for later checks.
    """
    chunks = []
    rows_read = 0

    # Limit chunk size so reader collects enough rows for sample
    # without taking on full table at once.
    reader_kwargs = {
        "usecols": usecols,
        "chunksize": min(nrows, 50000),
        "engine": "python",
    }
    if path.suffix == ".gz":
        reader_kwargs["compression"] = "gzip"

    # Stop reading once requested sample size has been reached.
    for chunk in pd.read_csv(path, **reader_kwargs):
        chunks.append(chunk)
        rows_read += len(chunk)
        if rows_read >= nrows:
            break

    # Return empty frame with expected columns when no chunks were
    # read so downstream summary code can still run cleanly.
    if not chunks:
        return pd.DataFrame(columns=usecols)
    
    return pd.concat(chunks, ignore_index=True).head(nrows)

def _timestamp_check(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
) -> Dict[str, int]:
    """
    Check paired timestamp fields for missing partners and invalid
    ordering.
    """
    out = {"invalid_order_rows": 0, "missing_pair_rows": 0}
    if start_col not in df.columns or end_col not in df.columns:
        return out
    
    # Coerce timestamps to datetime so invalid text values are
    # handled as missing during order checks.
    s = pd.to_datetime(df[start_col], errors="coerce")
    e = pd.to_datetime(df[end_col], errors="coerce")
    out["missing_pair_rows"] = int(((s.isna()) ^ (e.isna())).sum())
    out["invalid_order_rows"] = int(
        ((~s.isna()) & (~e.isna()) & (e < s)).sum()
    )
    return out

# --------------------------------------------------------------------
# Raw table validation
# --------------------------------------------------------------------
def validate_raw_tables(
    sample_rows: int | None = None,
) -> Dict[str, dict]:
    """
    Validate configured raw tables using sampled structural review.
    Each table summary includes file presence, column coverage,
    sampled null counts, and additional checks for keys, timestamps,
    and numeric values.
    """
    results: Dict[str, dict] = {}

    # Review each configured source table so saved JSON reflects same
    # central file mapping used elsewhere in project.
    for table_name, path in MIMIC_FILES.items():
        # Record missing files idrectly in output so validation
        # summary remains informative even when source table is
        # absent.
        if not path.exists():
            results[table_name] = {
                "exists": False,
                "path": str(path),
                "error": "file_not_found",
            }
            continue
        
        try:
            # Read header first so script can identify missing
            # required columns before pulling sample of data rows.
            available_columns = _read_header(path)
            required = REQUIRED_COLUMNS.get(table_name, [])
            missing = [c for c in required if c not in available_columns]
            usecols = _needed_columns(table_name, available_columns)

            # Use chunked sampling for largest tables so same
            # validation logic remains practical across all sources.
            if table_name in LARGE_TABLES:
                df = _read_large_sample(
                    path,
                    usecols=usecols,
                    nrows=sample_rows,
                )
            else:
                df = _read_sample(
                    path,
                    usecols=usecols,
                    nrows=sample_rows,
                )

            # Collect main structural checks in one dictionary so
            # result can be written directly to JSON.
            entry = {
                "exists": True,
                "path": str(path),
                "n_rows_checked": int(len(df)),
                "n_cols_total": int(len(available_columns)),
                "n_cols_checked": int(len(df.columns)),
                "missing_required_columns": missing,
                "null_counts_required": {
                    c: int(df[c].isna().sum())
                    for c in required
                    if c in df.columns
                },
            }

            # Check duplicate combinations only on key columns that
            # are actually present in sampled frame.
            key_cols = [c for c in KEY_COLUMNS if c in df.columns]
            if key_cols:
                entry["repeated_rows_on_available_keys"] = int(
                    df.duplicated(subset=key_cols).sum()
                )

            # Apply time-order checks only to tables that carry
            # relevant paired timestamp fields.
            if table_name == "ADMISSIONS":
                entry["time_checks"] = _timestamp_check(
                    df,
                    "ADMITTIME",
                    "DISCHTIME",
                )
            elif table_name == "ICUSTAYS":
                entry["time_checks"] = _timestamp_check(
                    df,
                    "INTIME",
                    "OUTTIME",
                )

            # Summarise VALUENUM only when that field is available
            # because not every raw source table contains event
            # values.
            if "VALUENUM" in df.columns:
                val = pd.to_numeric(df["VALUENUM"], errors="coerce")
                entry["valuenum_non_null"] = int(val.notna().sum())
                entry["valuenum_negative"] = int((val < 0).sum())
                entry["valuenum_zero"] = int((val == 0).sum())

            results[table_name] = entry

        except Exception as e:
            # Save table level error in summary so one failed read
            # does not stop wider validation report.
            results[table_name] = {
                "exists": True,
                "path": str(path),
                "error": str(e)
            }
    
    return results

# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------
def main() -> None:
    """
    Run the raw table validation and save the resulting JSON summary.
    """
    # Expose sample size as simple command line argument because
    # larger samples can be helpful when checking raw table quality.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-rows", 
        type=int, 
        default=5000, 
        help="Number of rows to sample from each table."
    )
    args = parser.parse_args()

    # Generate validation summary first, then reuse it for saving and
    # for console preview.
    results = validate_raw_tables(sample_rows=args.sample_rows)
    out_path = PREPROCESSING_OUTPUT_DIR / "validate_mimic_tables.json"
    out_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    # Print one grouped table summary at a time so console output
    # remains easy to scan during direct runs.
    for table_name, summary in results.items():
        print(f"\n[{table_name}]")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    print(f"\nSaved validation summary to: {out_path}")

if __name__ == "__main__":
    main()