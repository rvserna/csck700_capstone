"""
MODULE 1: Data Preparation

Purpose:
- Build an admission-level, model-ready tabular dataset from selected 
  MIMIC-III source tables.
- Define the fixed feature schema used across the full pipeline.
- Prepare clean numeric inputs for model training, explanation 
  generation, and benchmarking.

Broader Context:
- model_development.py trains mortality prediction models using the
  processed feature table created here.
- explanation_engine.py labels attribution outputs using the feature
  names defined in this module.
- benchmarking_engine.py uses the prepared arrays and feature schema
  when comparing fidelity, robustness, agreement, and clinical
  alignment.
- reporting_interface.py displays outputs derived from benchmark
  tables that depend on this prepared dataset.

Additional Notes:
- This module focuses on dataset construction and preprocessing rather
  than model training or explanation generation.
- Its main role is to ensure that every downstream component uses the
  same feature definitions, target field, and preprocessing workflow.
"""

import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import duckdb
except ImportError:
    duckdb = None

# --------------------------------------------------------------------
# Project Paths
# --------------------------------------------------------------------
# Resolve all paths relative to project root so file locations remain
# consistent across environments and easier to update centrally.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_DIR = DATA_DIR / "joined_agg_dataset"

# Ensure output directory exists before writing feature files so
# downstream export steps do not fail due to missing folders.
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Feature List Used Downstream
# --------------------------------------------------------------------
# Define canonical feature schema reused across modelling,
# explanations, benchmarking, and reporting to ensure consistency
# throughout pipeline. 
FEATURES: List[str] = [
    # Add demographic and admission structure features.
    "age_at_admission", 
    "gender", 
    "icd_count",
    "icu_stay_count",
    "procedure_icd_count",
    "procedure_row_count",
    "lab_count", 
    "chart_count",
    
    # Add aggregate statistics across all lab and chart events.
    "lab_avg", 
    "lab_min", 
    "lab_max",
    "chart_avg", 
    "chart_min", 
    "chart_max",

    # Add targeted vital sign features derived from chart events.
    "heart_rate_mean", 
    "resp_rate_mean", 
    "spo2_min", 
    "mean_bp", 
    "temp_max",

    # Add targeted laboratory features derived from lab events.
    "wbc_max", 
    "creatinine_max", 
    "bilirubin_max", 
    "platelet_min", 
    "glucose_max",
    
    # Add event-based proxy feature representing intervention
    # activity.
    "ventilator_event_count", 
]

# Define binary outcome column used across training and evaluation.
TARGET_COLUMN = "hospital_expire_flag"


# --------------------------------------------------------------------
# Default File Locations
# --------------------------------------------------------------------
# Define expected raw input files so dataset build can run with 
# standard configuration when no custom paths are provided.
DEFAULT_MIMIC_FILES: Dict[str, str] = {
    "ADMISSIONS": str(
        PROJECT_ROOT / "data/mimic-iii/ADMISSIONS.csv"
    ),
    "DIAGNOSES_ICD": str(
        PROJECT_ROOT / "data/mimic-iii/DIAGNOSES_ICD.csv.gz"
    ),
    "ICUSTAYS": str(
        PROJECT_ROOT / "data/mimic-iii/ICUSTAYS.csv"
    ),
    "LABEVENTS": str(
        PROJECT_ROOT / "data/mimic-iii/LABEVENTS.csv"
    ),
    "PATIENTS": str(
        PROJECT_ROOT / "data/mimic-iii/PATIENTS.csv"
    ),
    "PROCEDURES_ICD": str(
        PROJECT_ROOT / "data/mimic-iii/PROCEDURES_ICD.csv.gz"
    ),
    "CHARTEVENTS": str(
        PROJECT_ROOT / "data/mimic-iii/CHARTEVENTS.csv"
    ),
}

# --------------------------------------------------------------------
# ItemID Mapping Dictionaries
# --------------------------------------------------------------------
# Map each engineered feature to its corresponding MIMIC ITEMIDs
# so feature construction rules remain explicit and reproducible.
LAB_FEATURES: Dict[str, List[int]] = {
    # Use one lab ITEMID for each targeted lab feature.
    "wbc_max": [51300],
    "creatinine_max": [50912],
    "bilirubin_max": [50885],
    "platelet_min": [51265],
    "glucose_max": [50931],
}

VITAL_FEATURES: Dict[str, List[int]] = {
    # Use one chart event ITEMID for each targeted vital feature.
    "heart_rate_mean": [211],
    "resp_rate_mean": [618],
    "spo2_min": [646],
    "mean_bp": [456],
}

INTERVENTION_FEATURES: Dict[str, List[int]] = {
    # Use the two ITEMIDs that contribute to this event count feature.
    "ventilator_event_count": [720, 721],
}

# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------
def _sql_path(path_str: str) -> str:
    """
    Convert file system path into resolved POSIX style string.

    Ensures file paths are compatible with DuckDB SQL execution.
    """
    # Resolve path and convert to POSIX format so it can be safely
    # embedded inside SQL queries across operating systems.
    return Path(path_str).resolve().as_posix()


def _validate_input_files(files: Dict[str, str]) -> None:
    """
    Validate that all required MIMIC input files are present.

    Prevents partial dataset builds caused by missing inputs.
    """
    # Ensure all expeected dataset keys are provided before moving
    # forward.
    required = set(DEFAULT_MIMIC_FILES.keys())
    missing_keys = sorted(required - set(files.keys()))
    if missing_keys:
        raise ValueError(
            f"Missing required MIMIC file keys: {missing_keys}"
        )

    # Confirm each referenced file exists on disk before running
    # queries.
    for key in required:
        if not Path(files[key]).exists():
            raise FileNotFoundError(
                f"Missing input file for {key}: {files[key]}"
            )

# --------------------------------------------------------------------
# Main Dataset Construction
# --------------------------------------------------------------------
def build_model_features_parquet(
    files: Dict[str, str] | None = None,
    model_out_path: str = str(FEATURE_DIR / "model_features.parquet"),
    preview_rows: int = 10,
) -> str:
    """
    Build model-ready feature table from raw MIMIC-III files.

    Loads raw tables into DuckDB, creates admission level aggregates,
    engineers defined feature set, and saves result
    as Parquet file for downstream reuse.
    """
    # Ensure DuckDB is available since all transformations rely on
    # SQL execution.
    if duckdb is None:
        raise ImportError(
            "duckdb is required to run "
            "build_model_features_parquet()."
        )

    # Use default file paths when none are provided so function runs
    # with standard project structure.
    files = files or DEFAULT_MIMIC_FILES

    # Validate inputs before running build to avoid mid-pipeline
    # failures.
    _validate_input_files(files)
 
    # Resolve output path before execution so export location is
    # valid.
    model_out_path = str(Path(model_out_path).resolve())
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    out_path_sql = Path(model_out_path).resolve().as_posix()

    # Convert all input paths once so they can be reused in SQL
    # statements.
    admissions = _sql_path(files["ADMISSIONS"])
    diagnoses = _sql_path(files["DIAGNOSES_ICD"])
    icustays = _sql_path(files["ICUSTAYS"])
    labevents = _sql_path(files["LABEVENTS"])
    patients = _sql_path(files["PATIENTS"])
    procedures = _sql_path(files["PROCEDURES_ICD"])
    chartevents = _sql_path(files["CHARTEVENTS"])

    # Open DuckDB connection to execute all transformations in one
    # engine.
    con = duckdb.connect()

    # ----------------------------------------------------------------
    # Load Raw Source Tables
    # ----------------------------------------------------------------
    # Create temporary tables for admissions, patients, diagnoses,
    # procedures, ICU stays, lab events, and chart events so each
    # dataset can be referenced consistently across joins and
    # aggregation steps.
    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE admissions_raw AS
    SELECT 
        SUBJECT_ID, 
        HADM_ID, 
        ADMITTIME, 
        DISCHTIME, 
        ADMISSION_TYPE,
        HOSPITAL_EXPIRE_FLAG
    FROM read_csv_auto('{admissions}', types={{
        'SUBJECT_ID': 'BIGINT',
        'HADM_ID': 'BIGINT',
        'ADMITTIME': 'TIMESTAMP',
        'DISCHTIME': 'TIMESTAMP',
        'ADMISSION_TYPE': 'VARCHAR',
        'HOSPITAL_EXPIRE_FLAG': 'INTEGER'
    }})
    """)

    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE patients_raw AS
    SELECT SUBJECT_ID, GENDER, DOB, DOD
    FROM read_csv_auto('{patients}', types={{
        'SUBJECT_ID': 'BIGINT',
        'GENDER': 'VARCHAR',
        'DOB': 'TIMESTAMP',
        'DOD': 'TIMESTAMP'
    }})
    """)

    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE diagnoses_raw AS
    SELECT HADM_ID, ICD9_CODE
    FROM read_csv_auto('{diagnoses}', types={{
        'HADM_ID': 'BIGINT',
        'ICD9_CODE': 'VARCHAR'
    }})
    """)

    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE procedures_raw AS
    SELECT HADM_ID, ICD9_CODE
    FROM read_csv_auto('{procedures}', types={{
        'HADM_ID': 'BIGINT',
        'ICD9_CODE': 'VARCHAR'
    }})
    """)

    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE icustays_raw AS
    SELECT HADM_ID, ICUSTAY_ID, INTIME, OUTTIME
    FROM read_csv_auto('{icustays}', types={{
        'HADM_ID': 'BIGINT',
        'ICUSTAY_ID': 'BIGINT',
        'INTIME': 'TIMESTAMP',
        'OUTTIME': 'TIMESTAMP'
    }})
    """)

    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE labevents_raw AS
    SELECT HADM_ID, ITEMID, VALUENUM
    FROM read_csv_auto('{labevents}', types={{
        'HADM_ID': 'BIGINT',
        'ITEMID': 'INTEGER',
        'VALUENUM': 'DOUBLE'
    }})
    """)

    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE chartevents_raw AS
    SELECT HADM_ID, ITEMID, VALUENUM
    FROM read_csv_auto('{chartevents}', types={{
        'HADM_ID': 'BIGINT',
        'ITEMID': 'INTEGER',
        'VALUENUM': 'DOUBLE'
    }})
    """)

    # ----------------------------------------------------------------
    # Temperature Normalisation
    # ----------------------------------------------------------------
    # Convert temperature values into consistent Celsius scale so
    # aggregated temperature features are comparable across records.
    con.sql("""
    CREATE OR REPLACE VIEW temperature_normalized AS
    SELECT 
        HADM_ID,
        CASE
            -- Fahrenheit to Celsius
            WHEN ITEMID = 678 THEN (VALUENUM -32.0) * 5.0 / 9.0
            -- Already Celsius
            WHEN ITEMID = 223761 THEN VALUENUM
            ELSE NULL
        END AS temp_c
    FROM chartevents_raw
    WHERE HADM_ID IS NOT NULL
        AND ITEMID IN (678, 223761)
        AND VALUENUM IS NOT NULL
    """)

    # Aggregate maximum temperature per admission so feature captures
    # highest observed value during each hospital stay.
    con.sql("""
    CREATE OR REPLACE VIEW temp_max AS
    SELECT HADM_ID, MAX(temp_c) AS temp_max
    FROM temperature_normalized
    GROUP BY HADM_ID
    """)

    # ----------------------------------------------------------------
    # Admission Aggregates
    # ----------------------------------------------------------------
    # Collapse raw diagnosis rows to one record per admission so
    # feature table matches admission level unit used by predictive
    # model.
    con.sql("""
    CREATE OR REPLACE VIEW diagnoses_agg AS
    SELECT HADM_ID, COUNT(DISTINCT ICD9_CODE) AS icd_count
    FROM diagnoses_raw
    GROUP BY HADM_ID
    """)

    # Summarise procedure burden at admission level by keeping both
    # total number of procedure rows and number of distinct procedure
    # codes.
    con.sql("""
    CREATE OR REPLACE VIEW procedures_agg AS
    SELECT
        HADM_ID,
        COUNT(*) AS procedure_row_count,
        COUNT(DISTINCT ICD9_CODE) AS procedure_icd_count
    FROM procedures_raw
    GROUP BY HADM_ID
    """)

    # Build broad lab event summary features so each admission keeps
    # both event volume and overall numeric lab measurement ranges.
    con.sql("""
    CREATE OR REPLACE VIEW labevents_agg AS
    SELECT
        HADM_ID,
        COUNT(*) AS lab_count,
        AVG(VALUENUM) AS lab_avg,
        MIN(VALUENUM) AS lab_min,
        MAX(VALUENUM) AS lab_max
    FROM labevents_raw
    WHERE HADM_ID IS NOT NULL
        AND VALUENUM IS NOT NULL
    GROUP BY HADM_ID
    """)

    # ----------------------------------------------------------------
    # Specific Lab Features
    # ----------------------------------------------------------------
    # Build targeted lab features from named ITEMID mappings so
    # feature engineering rules remain explicit and easy to review.
    for name, itemids in LAB_FEATURES.items():
        # Choose aggregation rule from feature name so SQL view
        # matches intended feature definition.
        agg_func = "MAX" if "max" in name else "MIN"
        itemids_str = ", ".join(str(int(i)) for i in itemids)

        # Create one admission level view per targeted lab feature
        # so each variable can be joined later by its final feature
        # name.
        con.sql(f"""
        CREATE OR REPLACE VIEW {name} AS
        SELECT HADM_ID, {agg_func}(VALUENUM) AS {name}
        FROM labevents_raw
        WHERE HADM_ID IS NOT NULL
            AND ITEMID IN ({itemids_str}) 
            AND VALUENUM IS NOT NULL
        GROUP BY HADM_ID
        """)

    # ----------------------------------------------------------------
    # Generic Chart Event Aggregates
    # ----------------------------------------------------------------
    # Summarise all numeric chart event rows at admission level so
    # final table includes both overall charting volume and broad
    # value ranges.
    con.sql("""
    CREATE OR REPLACE VIEW chartevents_agg AS
    SELECT
        HADM_ID,
        COUNT(*) AS chart_count,
        AVG(VALUENUM) AS chart_avg,
        MIN(VALUENUM) AS chart_min,
        MAX(VALUENUM) AS chart_max
    FROM chartevents_raw
    WHERE HADM_ID IS NOT NULL AND VALUENUM IS NOT NULL
    GROUP BY HADM_ID
    """)

    # ----------------------------------------------------------------
    # Vital Sign Features 
    # ----------------------------------------------------------------
    # Build each targeted vital feature from its selected chart event
    # ITEMIDs. Aggregation rule follows meaning carried by feature
    # name.
    for name, itemids in VITAL_FEATURES.items():
        # Use AVG for mean style variables, MIN for min style
        # variables, and MAX for max style variables.
        agg_func = (
            "MAX" if "max" in name 
            else "MIN" if "min" in name 
            else "AVG"
        )
        itemids_str = ", ".join(str(int(i)) for i in itemids)

        # Create one admission level view per vital feature so later
        # join step can keep final feature labels clear and
        # consistent.
        con.sql(f"""
        CREATE OR REPLACE VIEW {name} AS
        SELECT HADM_ID, {agg_func}(VALUENUM) AS {name}
        FROM chartevents_raw
        WHERE HADM_ID IS NOT NULL
            AND ITEMID IN ({itemids_str}) 
            AND VALUENUM IS NOT NULL
        GROUP BY HADM_ID
        """)

    # ----------------------------------------------------------------
    # Intervention Proxies
    # ----------------------------------------------------------------
    # Count selected chart event rows for named intervention proxy so
    # final table retains one admission level intervention activity
    # feature.
    for name, itemids in INTERVENTION_FEATURES.items():
        itemids_str = ", ".join(str(int(v)) for v in itemids)

        # Build one view per intervention proxy so result can be
        # joined later using same feature name expected elsewhere in
        # pipeline.
        con.sql(f"""
        CREATE OR REPLACE VIEW {name} AS
        SELECT HADM_ID, COUNT(*) AS {name}
        FROM chartevents_raw
        WHERE HADM_ID IS NOT NULL
            AND ITEMID IN  ({itemids_str})
        GROUP BY HADM_ID
        """)

    # ----------------------------------------------------------------
    # ICU Stay Summary
    # ----------------------------------------------------------------
    # Collapse ICU stay rows to admission level so ICU activity can
    # be joined into final model-ready table without duplicating
    # admissions.
    con.sql("""
    CREATE OR REPLACE VIEW icustays_agg AS
    SELECT
        HADM_ID,
        COUNT(DISTINCT ICUSTAY_ID) AS icu_stay_count,
        MIN(INTIME) AS first_icu_intime,
        MAX(OUTTIME) AS last_icu_outtime
    FROM icustays_raw
    WHERE HADM_ID IS NOT NULL
    GROUP BY HADM_ID
    """)

    # ----------------------------------------------------------------
    # Final Model-Ready table
    # ----------------------------------------------------------------
    # Join engineered components into one row per admission so final
    # dataset matches structure expected by modelling pipeline.
    con.sql("""
    CREATE OR REPLACE VIEW model_features AS
    SELECT
        p.SUBJECT_ID,
        a.HADM_ID,
        i.icu_stay_count,
        
        -- Context fields retained before deriving modelling features
        LOWER(p.GENDER) AS gender_raw,
        p.DOB,
        p.DOD,
        a.ADMITTIME,
        a.DISCHTIME,
        a.ADMISSION_TYPE,
        i.first_icu_intime,
        i.last_icu_outtime,  

        -- Derive age at admission and cap extreme values to handle
        -- de-identified age representation in source data.      
        CASE
            WHEN p.DOB IS NOT NULL AND a.ADMITTIME IS NOT NULL THEN
                CASE
                    WHEN 
                        DATE_DIFF(
                            'day', 
                            CAST(p.DOB AS DATE),
                            CAST(a.ADMITTIME AS DATE)
                        ) / 365 > 120 
                    THEN 90
                    ELSE 
                        DATE_DIFF(
                            'day', 
                            CAST(p.DOB AS DATE), 
                            CAST(a.ADMITTIME AS DATE)
                        ) / 365
                END
            ELSE NULL
        END AS age_at_admission,
        
        -- Encode gender as numeric feature so it can be used directly
        -- in downstream modelling workflow.
        CASE
            WHEN UPPER(TRIM(p.GENDER)) = 'M' THEN 1
            WHEN UPPER(TRIM(p.GENDER)) = 'F' THEN 0
            ELSE NULL
        END AS gender,
        
        -- Diagnosis and procedure burden features
        d.icd_count,
        pr.procedure_icd_count,
        pr.procedure_row_count,
            
        -- Generic event aggregates
        l.lab_count,
        c.chart_count,
        l.lab_avg,
        l.lab_min,
        l.lab_max,
        c.chart_avg,
        c.chart_min,
        c.chart_max,
        
        -- Targeted vital features
        h.heart_rate_mean,
        r.resp_rate_mean,
        s.spo2_min,
        bp.mean_bp,
        t.temp_max,
            
        -- Targeted lab features
        w.wbc_max,
        cr.creatinine_max,
        b.bilirubin_max,
        pl.platelet_min,
        g.glucose_max,
            
        -- Intervention proxy
        ve.ventilator_event_count,
            
        -- Keep binary hospital mortality label for downstream 
        -- supervised learning
        a.HOSPITAL_EXPIRE_FLAG AS hospital_expire_flag
    
    FROM patients_raw AS p
    INNER JOIN admissions_raw AS a
        ON p.SUBJECT_ID = a.SUBJECT_ID
    INNER JOIN icustays_agg AS i
        ON a.HADM_ID = i.HADM_ID
    LEFT JOIN diagnoses_agg AS d
        ON a.HADM_ID = d.HADM_ID
    LEFT JOIN procedures_agg AS pr
        ON a.HADM_ID = pr.HADM_ID
    LEFT JOIN labevents_agg AS l
        ON a.HADM_ID = l.HADM_ID
    LEFT JOIN chartevents_agg AS c
        ON a.HADM_ID = c.HADM_ID
    LEFT JOIN wbc_max AS w
        ON a.HADM_ID = w.HADM_ID
    LEFT JOIN creatinine_max AS cr
        ON a.HADM_ID = cr.HADM_ID
    LEFT JOIN bilirubin_max AS b
        ON a.HADM_ID = b.HADM_ID
    LEFT JOIN platelet_min AS pl
        ON a.HADM_ID = pl.HADM_ID
    LEFT JOIN glucose_max AS g
        ON a.HADM_ID = g.HADM_ID
    LEFT JOIN heart_rate_mean AS h
        ON a.HADM_ID = h.HADM_ID
    LEFT JOIN resp_rate_mean AS r
        ON a.HADM_ID = r.HADM_ID
    LEFT JOIN spo2_min AS s
        ON a.HADM_ID = s.HADM_ID
    LEFT JOIN mean_bp AS bp
        ON a.HADM_ID = bp.HADM_ID
    LEFT JOIN temp_max AS t
        ON a.HADM_ID = t.HADM_ID
    LEFT JOIN ventilator_event_count AS ve
        ON a.HADM_ID = ve.HADM_ID
    WHERE a.HADM_ID IS NOT NULL
        AND (
            i.icu_stay_count IS NOT NULL
            OR h.heart_rate_mean IS NOT NULL
            OR r.resp_rate_mean IS NOT NULL
            OR s.spo2_min IS NOT NULL
            OR bp.mean_bp IS NOT NULL
            OR t.temp_max IS NOT NULL
            OR w.wbc_max IS NOT NULL
            OR cr.creatinine_max IS NOT NULL
            OR b.bilirubin_max IS NOT NULL
            OR pl.platelet_min IS NOT NULL
            OR g.glucose_max IS NOT NULL   
        )
    """)

    # Run optional preview query so build can be checked quickly
    # without opening saved Parquet file separately.
    if int(preview_rows) > 0:
        _ = con.sql(
            f"SELECT * FROM model_features LIMIT {int(preview_rows)}"
        ).df()

    # Save final admission level feature table as standard reusable
    # Parquet source for rest of benchmarking pipeline.
    con.sql(f"""
    COPY (SELECT * FROM model_features)
    TO '{out_path_sql}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # Close DuckDB connection after export is complete so build
    # finishes cleanly and releases SQL resources.
    con.close()
    return model_out_path

# --------------------------------------------------------------------
# Data Cleaning Helpers Used After Parquet File is Built
# --------------------------------------------------------------------
def _coerce_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise feature column names and coerce values to numeric
    form.

    Prepares saved feature tables for downstream modelling
    so later modules receive a consistent numeric feature frame.
    """
    # Copy input first so this cleaning step does not modify
    # original DataFrame.
    clean_df = df.copy()

    # Standardise column names to lowercase so later feature selection
    # does not depend on mixed casing in source table.
    clean_df.columns = [str(c).lower() for c in clean_df.columns]

    # Coerce each column to numeric form so modelling features do not
    # keep mixed text and numeric values.
    for col in clean_df.columns:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    return clean_df

# --------------------------------------------------------------------
# Load Feature Table for Modelling
# --------------------------------------------------------------------
def load_feature_table(
    data_path: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load saved Parquet feature table for modelling.

    Calidates target label, keeps canonical project
    feature set, drops unusable all-null columns, and returns
    retained feature order alongside target labels.
    """
    # Confirm saved feature table exists before attempting to read it.
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {data_path}"
        )
    
    # Load saved Parquet file and standardise column names so later
    # schema checks use one consistent naming format.
    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]

    # ----------------------------------------------------------------
    # Extract and Validate the Target Label
    # ----------------------------------------------------------------
    # Stop early if required binary outcome column is missing before
    # downstream supervised learning depends on that label.
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing required target column: {TARGET_COLUMN}"
        )
    
    # Convert label to numeric form and remove rows with missing
    # target values before modelling split is created.
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    keep = y.notna()
    df = df.loc[keep].reset_index(drop=True)
    y = y.loc[keep].astype(int)

    # Confirm that remaining labels are binary 0/1 values so target
    # matches expected classification task.
    unique_y = set(y.unique().tolist())
    if not unique_y.issubset({0, 1}):
        raise ValueError(
            f"{TARGET_COLUMN} must be binary with values "
            f"in {{0, 1}}, got {sorted(unique_y)}"
        )
    
    y = y.to_numpy()

    # ----------------------------------------------------------------
    # Select Modelling Features Expected by Rest of Pipeline
    # ----------------------------------------------------------------
    # Keep only canonical project features present in saved table so
    # downstream modules use one stable feature space and ordering.
    candidate_features = [
        f.lower() 
        for f in FEATURES 
        if f.lower() in df.columns
    ]
    if not candidate_features:
        raise ValueError(
            "None of the expected modelling features were "
            "found in the parquet table."
        )
    
    # Clean retained feature frame so later preprocessing steps
    # receive standardised numeric inputs.
    X_df = _coerce_feature_frame(df[candidate_features])

    # Drop feature columns that contain no usable values in saved
    # table because they cannot contribute to downstream modelling.
    all_null = [c for c in X_df.columns if X_df[c].isna().all()]
    if all_null:
        X_df = X_df.drop(columns=all_null)

    # Record retained feature order so explanations and benchmark
    # outputs can reuse same column order later.
    used_features = X_df.columns.tolist()
    if not used_features:
        raise ValueError(
            "No usable feature columns remain after dropping "
            "all-null columns."
        )

    return X_df, y, used_features

# --------------------------------------------------------------------
# Preprocessing for Model Input
# --------------------------------------------------------------------
def fit_preprocessor(X_train: pd.DataFrame):
    """
    Fit training data preprocessor used for model input.

    Applies median imputation followed by standard scaling
    and returns fitted preprocessing objects for reuse.
    """
    # Require pandas DataFrame so column based preprocessing stays
    # aligned with saved feature schema.
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    
    # Clean training frame before fitting preprocessing objects so
    # input columns are numeric and consistently named.
    X_train = _coerce_feature_frame(X_train)

    # Replace missing values with training set median so each feature
    # keeps a usable numeric value before scaling.
    imputer = SimpleImputer(strategy="median")

    # Standardise each feature using fitted training set mean and
    # standard deviation so model inputs share one numeric scale.
    scaler = StandardScaler()

    # Fit preprocesing objects on training data only, then return
    # transformed training array alongside fitted objects.
    X_imp = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imp)

    return np.asarray(X_scaled, dtype=np.float32), imputer, scaler

def transform_with_preprocessor(X: pd.DataFrame, imputer, scaler):
    """
    Apply fitted preprocessor to new feature data.

    Reuses training-fitted imputer and scaler so all
    later datasets are transformed with same preprocessing rules.
    """
    # Require pandas DataFrame so column handling stays consistent
    # with training preprocessing workflow.
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    
    # Clean input frame before transformation so fitted preprocessor
    # receives numeric columns in consistent format.
    X = _coerce_feature_frame(X)

    # Apply training-fitted iputer first, then scale imputed values
    # using same fitted standardisation parameters.
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)

    return np.asarray(X_scaled, dtype=np.float32)

# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Allow module to be run directly so model-ready parquet dataset
    # can be rebuilt without importing module elsewhere.
    try:
        output_path = build_model_features_parquet(preview_rows=0)
        print(f"Saved model-ready dataset to: {output_path}")
    except (ImportError, FileNotFoundError, ValueError, RuntimeError):
        # Print full traceback so direct script runs provide enough
        # detail to diagnose build failures.
        import traceback
        
        traceback.print_exc()

