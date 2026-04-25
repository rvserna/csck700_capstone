"""
MODULE 5: Clinical Alignment

Purpose:
- Store the clinically validated reference feature set used in the
  benchmarking workflow.
- Compare explainer outputs against clinically meaningful variables
  identified in the supporting literature.
- Provide overlap, weighted-overlap, precision, and recall utilities
  for clinical alignment evaluation.

Broader Context:
- benchmarking_engine.py calls this module to calculate clinical
  alignment metrics for Captum IG, SHAP, and LIME.
- reporting_interface.py displays the clinical alignment outputs in
  summary cards, tables, captions, and downloadable reports.
- Feature importance outputs are compared against the fixed
  reference set defined here.

Additional Notes:
- This module does not train models or generate explanations.
- Its main role is to keep the clinical reference set and alignment
  calculations consistent across the full benchmarking pipeline.
"""

from typing import List, Set, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# Clinical Reference Feature Set
# --------------------------------------------------------------------
# Store fixed clinical reference features in one shared set so all 
# alignment calculations use same comparsion baseline.
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

# Keep backward compatible alias so older code can still reference
# same clinical feature set without changing its import pattern.
clinically_validated: Set[str] = CLINICALLY_VALIDATED_FEATURES

# --------------------------------------------------------
# Accessor
# --------------------------------------------------------
def get_clinically_validated_features() -> Set[str]:
    """
    Return clinically validated reference feature set.

    A copy is returned so callers can work with clinical set without
    modifying shared module level reference object.
    """
    return set(CLINICALLY_VALIDATED_FEATURES)

# --------------------------------------------------------
# Internal helpers
# --------------------------------------------------------
def _validate_series(series: pd.Series) -> pd.Series:
    """
    Validate and standardise feature importance series.

    Converts values to numeric form, fills missing values, and 
    standardises index labels before comparison logic is applied.
    """
    # Require pandas Series so later ranking and set based operations
    # use one consistent input structure.
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")
    
    # Convert values to numeric form and replace missing values with
    # zero so overlap calculations can run without mixed types or
    # NaNs.
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)

    # Standardise feature labels as strings so they can be compared
    # reliability against clinical reference set.
    s.index = s.index.astype(str)
    return s

def _validate_k(k: int) -> int:
    """
    Validate requested Top-K value.

    Ensures ranking depth is positive integer before it is used in 
    overlap, precision, or recall calculations.
    """
    # Require positive value so Top-K calculations remain meaningful
    # and do not produce invalid slices.
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    return int(k)

def _top_k_series(series: pd.Series, k: int) -> pd.Series:
    """
    Return Top-K feature importance values after input validation.

    Keeps ranking step centralised so same validation and selection 
    logic is reused across module.
    """
    # Standardise input series first so ranking step works on one
    # clean numeric representation.
    s = _validate_series(series)
    k = _validate_k(k)

    # Keep k largest values so downstream functions all compare
    # against same Top-K ranking rule.
    return s.nlargest(k)

# --------------------------------------------------------------------
# Basic Overlap Utilities
# --------------------------------------------------------------------
def clinical_overlap_features(
    series: pd.Series,
    clinical_set: Set[str] | None = None,
    k: int = 10,
) -> List[str]:
    """
    Return clinically validated features present in Top-K ranking.

    Returned list preserves Top-K ranking order so downstream outputs 
    can display overlap features in their ranked sequence.
    """
    # Use shared clinical reference set when no custom comparison set
    # is supplied by caller.
    if clinical_set is None:
        clinical_set = CLINICALLY_VALIDATED_FEATURES

    # Rank importance values first, then keep only feature names that
    # also appear in clinical reference set.
    top = _top_k_series(series, k)
    return [
        f
        for f in top.index.astype(str).tolist()
        if f in clinical_set
    ]

def clinical_overlap_count(
    series: pd.Series,
    clinical_set: Set[str] | None = None,
    k: int = 10,
) -> int:
    """
    Return number of clinically validated features in Top-K ranking.

    Reduces overlap list to one count so later summaries can report
    size of clinical overlap directly.
    """
    # Reuse shared overlap feature helper so counting follows same
    # Top-K selection and clinical set matching logic.
    return len(
        clinical_overlap_features(
            series,
            clinical_set=clinical_set,
            k=k,
        )
    )

# --------------------------------------------------------------------
# Weighted Overlap
# --------------------------------------------------------------------
def weighted_clinical_overlap(
    series: pd.Series,
    clinical_set: Set[str] | None = None,
    k: int = 10,
) -> float:
    """
    Compute importance weighted clinical overlap within Top-K
    features.

    Returned value is proportion of Top-K importance mass assigned to
    clinically validated features and ranges from 0 to 1.
    """
    # Use shared clinical reference set unless caller provides
    # different comparison set.
    if clinical_set is None:
        clinical_set = CLINICALLY_VALIDATED_FEATURES

    # Restrict calculation to ranked Top-K values so weighted overlap
    # matches same feature subset used in other module metics.
    top = _top_k_series(series, k)
    total = float(top.sum())

    # Return zero when Top-K importance mass is zero so function
    # avoids division by zero and reports no weighted overlap.
    if total == 0.0:
        return 0.0
    
    # Sum importance assigned to clinical features only, then divide
    # by full Top-K importance mass.
    clinical_weight = float(
        top[top.index.astype(str).isin(clinical_set)].sum()
    )
    return clinical_weight / total

# --------------------------------------------------------------------
# Precision / Recall vs Clinical Set
# --------------------------------------------------------------------
def clinical_precision_recall(
    series: pd.Series,
    clinical_set: Set[str] | None = None,
    k: int = 10,
) -> Tuple[float, float]:
    """
    Compute Top-K precision and recall against clinical reference set.

    Precision is proportion of Top-K features that are clinically
    validated, and recall is proportion of clinical set captured 
    within that Top-K ranking.
    """
    # Use shared clinical reference set unless caller supplies
    # different set for comparison.
    if clinical_set is None:
        clinical_set = CLINICALLY_VALIDATED_FEATURES

    # Validate requested Top-K size, then reuse overlap count so both
    # metrics are based on same matched feature set.
    k = _validate_k(k)
    overlap = clinical_overlap_count(
        series, 
        clinical_set=clinical_set, 
        k=k,
    )

    # Precision measures how much of Top-K ranking is clinically
    # validated, while recall measures how much of clinical set is
    # captured.
    precision = overlap / k
    recall = (
        overlap / len(clinical_set) 
        if len(clinical_set) > 0 
        else np.nan
    )

    return float(precision), float(recall)

