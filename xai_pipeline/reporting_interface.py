"""
MODULE 6: Reporting Interface

Purpose:
- Provide the user-facing dashboard for configuring, running, and
  reviewing the benchmark.
- Read the structured outputs written by benchmarking_engine.py and
  convert them into tables, charts, captions, summary cards, and
  downloadable files.
- Present benchmark results in a form that is easier to interpret for
  non-technical audiences.

Broader Context:
- benchmarking_engine.py produces the result tables, figures, and
  configuration outputs displayed here.
- clinical_alignment.py provides the clinical reference set used by
  the overlap and alignment views.
- data_preparation.py, model_development.py, and explanation_engine.py
  feed this module indirectly through the benchmark outputs generated
  upstream.

Additional notes:
- This module is a presentation layer for the benchmarking artefact.
- It does not calculate the benchmark metrics itself. It reads,
  formats, compares, and explains outputs generated earlier in the
  pipeline.
"""

from dataclasses import asdict, fields, is_dataclass
from io import BytesIO
from pathlib import Path
import base64
import json
import zipfile
from typing import Dict, Any, Optional, List
import textwrap

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import benchmarking_engine as bench
import clinical_alignment as ca

# Configure page before other content is rendered so layout and
# sidebar state are established at start of session.
st.set_page_config(
    page_title="XAI Benchmarking Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Standardise alternate labels into one display name so same
# explainer is referenced consistently across cards, tables, plots,
# and captions.
METHOD_NAME_MAP ={
    "Captum_IntegratedGradients": "Captum IG",
    "Captum IG": "Captum IG",
    "Captum_IG": "Captum IG",
    "Integrated Gradients": "Captum IG",
    "SHAP": "SHAP",
    "LIME": "LIME",
}

# Map each display label back to exported result-table column
# so presentation logic can look up correct data field reliabily.
METHOD_COLUMN_MAP = {
    "Captum IG": "Captum_IG",
    "SHAP": "SHAP",
    "LIME": "LIME",    
}

# Keeps method colours consistent across cards, plots, and legends so
# each explainer remains visually recognisable throughout
# dashboard.
METHOD_COLOR_MAP = {
    "Captum IG": "#3A6EA5",
    "SHAP": "#E6862A",
    "LIME": "#4C9F70",
}

# --------------------------------------------------------------------
# General Helpers
# --------------------------------------------------------------------
def _safe_float(v, default=np.nan):
    """
    Convert value to float without allowing local parsing issue to
    break page.

    Reporting interface reads values from exported tables, widget
    inputs, and intermediate calculations. Returning a fallback keeps
    later scoring, sorting, and formatting steps stable when a value
    is missing or malformed.
    """
    # Attempt numeric conversation directly because many dashboard
    # metrics are stored as text after being read back from CSV
    # exports.
    try:
        return float(v)
    except (TypeError, ValueError):
        # Fall back to supplied default so one invalid value does not
        # stop wider interface from rendering.
        return default

def _safe_int(v, default=0):
    """
    Convert value to int for widget defaults, indexing, and other
    count-based logic.

    Used where interface needs dependable integer even if source value
    is missing, stored as text, or cannot be parsed cleanly.
    """
    # Convert cautiously because widget defaults, row counts, and
    # index-based selections all require valid integer values.
    try:
        return int(v)
    except (TypeError, ValueError):
        # Preserve usable fallback so single malformed value does not
        # break later interface logic.
        return default

def _first_existing(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Return first preferred column name that exists in supplied
    DataFrame.

    Reporting layer uses this helper to stay compatible with slightly
    different benchmark export schemas without duplicating same checks
    in each section.
    """
    # Check candidates in priority order so preferred field name is
    # chosen when more than one acceptable option is present.
    for c in candidates:
        if c in df.columns:
            return c
      
    # Return None when no acceptable column is present so calling
    # section can decide how to degrade gracefully.
    return None

def _clean_method_name(name: str) -> str:
    """
    Standardise raw explainer labels into one consistent dashboard
    display name.

    This avoids showing the same method under multiple labels across
    tables, captions, legends, and summary cards.
    """
    # Coerce non-string inputs to text first so later label-mapping
    # logic can still return a stable display value.
    if not isinstance(name, str):
        return str(name)
   
    # Apply central mapping so every downstream view uses same method
    # name.
    return METHOD_NAME_MAP.get(name, name)

def _normalize_01(series: pd.Series) -> pd.Series:
    """
    Convert values to numeric form and clip them to 0-1 range.

    Used for bounded summary metrics so later comparisons, averages,
    and display elements remain on a consistent scale.
    """
    # Convert first because exported metrics may be read back as
    # strings.
    s = pd.to_numeric(series, errors="coerce")

    # Restrict values to expected range before combining into summary
    # scores.
    s = s.clip(lower=0, upper=1)
    return s

def _normalize_method_columns(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """
    Normalise selceted importance columns so each column sums to one
    in absolute terms.

    Supports fairer visual comparison when different explainers
    produce importance values on different numeric scales.
    """
    # Work on a copy so this step does not alter original table
    # that may still be reused by another section.
    out = df.copy()

    # Apply same scaling logic to each requested importance column.
    for c in cols:
        # Convert to numeric absolute values because
        # feature-importance views in this module compare magnitude
        # rather than sign.
        s = pd.to_numeric(out[c], errors="coerce").abs().fillna(0.0)
        denom = float(s.sum())

        # Only divide when column contains usable signal.
        # Otherwise, leave all-zero series unchanged.
        out[c] = s / denom if denom > 0 else s
    return out

# --------------------------------------------------------------------
# Feature-Table Preparation Helpers
# These functions standardise feature-level exports used by Top-K,
# clinical-alignment, and overlap views so later sections read from
# one cleaned structure rather than repeating same preparation logic.
# --------------------------------------------------------------------
def _prepare_clinical_feature_df(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare feature-level table used by clinical-alignment views.

    Ensures that feature name field is text-based, clinical-reference 
    indicator is present, and resulting schema is stable before later
    sections rely on it.
    """
    # Copy input so clinical-alignment preparation does not mutate
    # original export that may still be used elsewhere.
    df = feature_df.copy()

    # If core feature-name column is unavailable, return input copy
    # as-is so calling section can decide how to handle incomplete
    # schema.
    if "Feature" not in df.columns:
        return df

    # Standardise feature labels as strings before membership checks
    # or merges.
    df["Feature"] = df["Feature"].astype(str)

    # Reuse an existing clinical flag when already present in export.
    if "Clinically_Validated" in df.columns:
        df["Clinically_Validated"] = pd.to_numeric(
            df["Clinically_Validated"], errors="coerce"
        ).fillna(0).astype(int)
        return df

    # Derive flag from fixed clinical reference set when export does
    # not already include that field.
    clinical_set = {
        str(f)
        for f in ca.get_clinically_validated_features()
    }
    df["Clinically_Validated"] = (
        df["Feature"]
        .isin(clinical_set)
        .astype(int)
    )
    return df

def _coerce_abs_numeric(series: pd.Series) -> pd.Series:
    """
    Convert series to numeric absolute values for ranking or plotting.

    Several reporting views compare attribution magnitude rather than
    direction, so this helper keeps that conversion consistent in one
    place.
    """
    # Use absolute magnitude because related dashboard sections
    # focus on how strongly a feature contributes, not whether
    # contribution is positive or negative.
    return pd.to_numeric(series, errors="coerce").abs()

def _get_method_column(
    method_name: str,
    df: Optional[pd.DataFrame] = None,
) -> str | None:
    """
    Return exported result-table column associated with a display
    method name.

    Keeping this lookup in one helper avoids repeating schema-mapping 
    logic across dashboard.
    """
    # Translate display label into column name used in saved outputs.
    method_col = METHOD_COLUMN_MAP.get(
        _clean_method_name(method_name)
    )
  
    # Stop early when display label does not map to a known result
    # column.
    if method_col is None:
        return None
  
    # When a DataFrame is supplied, confirm that resolved column is
    # actually present before caller tries to use it.
    if df is not None and method_col not in df.columns:
        return None
  
    return method_col

def _get_method_feature_frame(
    row: pd.Series, 
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    """
    Build cleaned feature-level table for one explanation method.

    Returned frame is reused by clinical-alignment, Top-K, overlap,
    and feature-importance views so those sections all rely on same
    preparation steps.
    """
    # Exit early when feature-level export is missing or empty so
    # interface can fail gracefully instead of raising downstream
    # errors.
    if feature_df is None or feature_df.empty:
        return pd.DataFrame(), None
  
    # Start from standardised clinical feature table so this helper
    # always works with one expected schema.
    df = _prepare_clinical_feature_df(feature_df)
    required = {"Feature", "Clinically_Validated"}

    # Confirm that minimum required columns are available before
    # continuing.
    if not required.issubset(df.columns):
        return pd.DataFrame(), None
  
    # Resolve explainer column from method name carried in the
    # summary row.
    method_col = _get_method_column(row.get("Method", ""), df)
    if method_col is None:
        return pd.DataFrame(), None
  
    # Keep only fields needed by later feature-level views.
    temp = df[
        ["Feature", "Clinically_Validated", method_col]
    ].copy()

    # Standardise shared columns before sorting, filtering, or
    # displaying them.
    temp["Feature"] = temp["Feature"].astype(str)
    temp["Clinically_Validated"] = pd.to_numeric(
        temp["Clinically_Validated"], errors="coerce"
    ).fillna(0).astype(int)

    # Convert method-specific values to numeric absolute magnitudes so
    # output can be ranked consistently.
    temp[method_col] = _coerce_abs_numeric(temp[method_col])

    # Drop rows that still lack usable importance values after
    # coercion.
    temp = temp.dropna(subset=[method_col])
    return temp, method_col

def _get_top_method_feature_frame(
    row: pd.Series, 
    feature_df: pd.DataFrame, 
    top_k: int,
) -> tuple[pd.DataFrame, str | None]:
    """
    Return Top-K feature rows for one method after cleaning and
    sorting.

    Ensures that all Top-K views are based on same cleaned
    feature table and same ranking logic.
    """
    # Start from method-specific feature frame so Top-K selection
    # is always based on shared prepration logic used elsewhere.
    temp, method_col = _get_method_feature_frame(row, feature_df)
    if temp.empty or method_col is None:
        return pd.DataFrame(), None
  
    # Sort by importance magnitude and keep only the requested
    # number of rows.
    return temp.sort_values(
        method_col,
        ascending=False,
    ).head(top_k), method_col

# --------------------------------------------------------------------
# Metric Interpretation Helpers
# These helpers translate numeric outputs into concise, defensible
# wording for cards and captions without overstating what benchmark
# shows.
# --------------------------------------------------------------------
def _auc_interpretation(auc: float) -> tuple[str, str]:
    """
    Translate ROC-AUC value into short explanatory dashboard text.

    Returned pair contains a concise headline and a supporting
    sentence that can be reused in summary cards or captions.
    """
    # Use ordered thresholds so interpretation remains consistent
    # wherever model performance is summarised in interface.
    if auc >= 0.90:
        return (
            "Excellent discriminative performance",
            "The predictive signal is very strong, so explanation "
            "comparisons can be interpreted with relatively greater "
            "confidence."
        )
    # Branch here so section can handle alternate data conditions
    # safely.
    if auc >= 0.80:
        return (
            "Strong discriminative performance",
            "The model separates classes well, providing a credible "
            "basis for downstream explanation benchmarking."
        )
    # Branch here so section can handle alternate data conditions
    # safely.
    if auc >= 0.70:
        return (
            "Acceptable discriminative performance",
            "The model has usable predictive signal, though "
            "explanation results should still be interpreted with "
            "some caution."
        )
    # Branch here so section can handle alternate data conditions
    # safely.
    if auc >= 0.60:
        return (
            "Weak discriminative performance",
            "The model captures some signal, but explanation " 
            "quality should be interpreted cautiously because the "
            "predictor is only modestly informative."
        )
    # Branch here so section can handle alternate data conditions
    # safely.
    if auc >= 0.50:
        return (
            "Very weak discriminative performance",
            "Performance is close to random, so explanation outputs "
            "may not reflect a reliably useful predictive signal."
        )
    return (
        "Performance worse than random",
        "This may indicate label inversion, instability, severe " 
        "data issues, or a weak underlying signal; explanation "
        "results should be treated very cautiously.Consider "
        "adjusting the experimental parameters to achieve greater "
        "predictive performance."
    )

def _agreement_band(v: float) -> str:
    """
    Convert numeric agreement value into a qualitative label.

    Used in cards and captions so agreement metrics are easier
    to interpret at a glance.
    """
    # Return explicit placeholder when value is missing so user
    # does not mistake an absent result for a low score.
    if pd.isna(v):
        return "Unavailable"
    if v >= 0.80:
        return "Very high agreement"
    if v >= 0.60:
        return "Strong agreement"
    if v >= 0.40:
        return "Moderate agreement"
    if v >= 0.20:
        return "Low agreement"
    if v >= 0:
        return "Very low agreement"
    return "Negative agreement"

def _format_metric(v: float, decimals: int = 3) -> str:
    """
    Format metric consistently for cards, tables, and captions.

    Centralising this step helps dashboard present numeric outputs
    in a uniform way across sections.
    """
    # Use single placeholder for missing values so interface stays
    # visually consistent.
    if pd.isna(v):
        return "N/A"
   
    # Apply requested precision once here rather than repeating
    # formatting code.
    return f"{float(v):.{decimals}f}"

def _top_k_default() -> int:
    """
    Return default Top-K value defined by benchmark configuration.

    Using this helper keeps Top-K defaults aligned with engine without
    hard-coding same value repeatedly in interface.
    """
    # Read setting from benchmark configuration so interface default
    # stays aligned with current engine defintion.
    return int(getattr(bench.RunConfig(), "top_k", 10))

# --------------------------------------------------------------------
# Output Parsing Helpers
# Benchmark can export multiple record types in one table, so these
# helpers isolate only rows needed for each reporting view.
# --------------------------------------------------------------------
def _get_method_summary_df(method_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract method-summary records used by main dashboard view.

    Benchmark exports can contain more than one record type, so this
    helper isolates rows intended for method-level summary displays.
    """
    # Work on a copy so later formatting does not mutate original
    # data loaded from disk.
    df = method_df.copy()

    # Prefer explicity method-summary rows when export includes
    # a RecordType field.
    if "RecordType" in df.columns:
        subset = df[
            df["RecordType"].fillna("")
            == "MethodSummary"
        ].copy()

        if not subset.empty:
            df = subset

    # Standardise method names so every later section uses same
    # labels.
    if "Method" in df.columns:
        df["Method"] = df["Method"].map(_clean_method_name)
    return df.reset_index(drop=True)

def _get_feature_importance_df(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract feature-importance records used throughout interface.

    Keeps later feature-level sections focused on relevant export rows
    without repeating same RecordType filter.
    """
    # Copy input so downstream preparation does not modify raw
    # export table.
    df = feature_df.copy()

    # Prefer explicit feature-importance rows when mixed export
    # includes a RecordType column.
    if "RecordType" in df.columns:
        subset = df[
            df["RecordType"].fillna("")
            == "FeatureImportance"
        ].copy()

        if not subset.empty:
            df = subset

    return df.reset_index(drop=True)

def _get_agreement_row(
    agreement_df: pd.DataFrame,
) -> pd.Series | None:
    """
    Return agreement record used by agreement-related views.

    Some exports may contain multiple record types, so this helper
    isolates explainer-agreement row when it is explicitly labelled.
    """
    # Exit early when no agreement data is available.
    if agreement_df is None or agreement_df.empty:
        return None
   
    # Prefer explicitly labelled agreement row when export provides one.
    if "RecordType" in agreement_df.columns:
        subset = agreement_df[
            agreement_df["RecordType"].fillna("")
            == "ExplainerAgreement"
        ].copy()
      
        if not subset.empty:
            return subset.iloc[0]
      
    # Fall back to first row when export already contains only
    # agreement data.
    return agreement_df.iloc[0]

def _get_overlap_feature_text(
    row: pd.Series, 
    feature_df: pd.DataFrame, 
    top_k: int,
) -> list[str]:
    """
    Return clinically validated features captured in one method's Top-K
    list.

    Output is used in captions and summary text where interface needs
    feature names themselves rather than just an overlap count.
    """
    # Build method-specific Top-K table first so overlap list reflects
    # same ordering shown elsewhere in dashboard.
    top_df, _ = _get_top_method_feature_frame(row, feature_df, top_k)
    if top_df.empty:
        return []

    # Keep only clinically validated features and return their names
    # in Top-K order. 
    return top_df.loc[
        top_df["Clinically_Validated"] == 1,
        "Feature",
    ].tolist()

def _get_missing_clinical_feature_list(
    row: pd.Series,
    feature_df: pd.DataFrame,
    top_k: int,
) -> list[str]:
    """
    Return clinically validated features that are absent from displayed
    Top-K set.

    Supports explanatory text that distinguishes between features
    captured by method and clinically relevant features that were
    not selected.
    """
    # Start from cleaned method-specific feature table so comparison
    # uses same preparation steps as visible Top-K views.
    temp, method_col = _get_method_feature_frame(row, feature_df)
    if temp.empty or method_col is None:
        return []
    
    # Identify which clinical features are present in displayed
    # Top-K set.
    top_df = temp.sort_values(
        method_col,
        ascending=False,
    ).head(top_k)

    captured_features = set(
        top_df.loc[
            top_df["Clinically_Validated"] == 1,
            "Feature",
        ].tolist()
    )

    # Build clinically validated reference list ordered by method's
    # own importance ranking so omissions can be reported in a
    # meaningful order.
    clinical_reference = temp.loc[
        temp["Clinically_Validated"] == 1
    ].sort_values(method_col, ascending=False)["Feature"].tolist()
  
    return [
        f
        for f in clinical_reference
        if f not in captured_features
    ]

def _build_reliability_table(
    method_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """
    Build method-level relilability summary table for reporting
    interface.

    Returned table combines selected robustness, fidelity, and
    clinical alignment metrics into a compact comparison view that
    can be reused by cards, rankings, or tables.
    """
    # Start from a copy so score construction does not modify original
    # method-summary table loaded from disk.
    df = method_df.copy()

    # Resolve columns needed for each score component while remaining
    # tolerant of small schema differences across runs.
    method_col = _first_existing(df, ["Method"])
    noise_col = _first_existing(df, ["Stability_Spearman_Noise_Mean"])
    mask_col = _first_existing(
        df,
        ["Stability_Spearman_Masking_Mean"],
    )
    fid_perm_col = _first_existing(
        df,
        ["Fidelity_vs_Permutation_Spearman_Mean"],
    )
    fid_grad_col = _first_existing(
        df,
        ["Fidelity_vs_GradInput_Spearman_Mean"],
    )
    clin_overlap_col = _first_existing(
            df,
            [
                f"Top{top_k}_Clinical_Overlap",
                "Top10_Clinical_Overlap",
            ],
        )
    weighted_clin_col = _first_existing(
        df,
        ["Weighted_Clinical_Overlap"],
    )
    clin_recall_col = _first_existing(df, ["Clinical_Recall"])

    # Return empty, correctly shaped frame when minimum required
    # method field is absent.
    if method_col is None:
        return pd.DataFrame(columns=["Method", "Reliability_Score"])

    out = pd.DataFrame()
    out["Method"] = df[method_col].map(_clean_method_name)

    # Normalise main component metrics so they can be combined on a
    # common 0-1 scale.
    out["Noise_Stability"] = (
        _normalize_01(df[noise_col])
        if noise_col
        else np.nan
    )

    out["Masking_Stability"] = (
        _normalize_01(df[mask_col])
        if mask_col
        else np.nan
    )

    out["Permutation_Fidelity"] = (
        _normalize_01(df[fid_perm_col])
        if fid_perm_col
        else np.nan
    )

    out["GradInput_Fidelity"] = (
        _normalize_01(df[fid_grad_col])
        if fid_grad_col
        else np.nan
    )

    # Convert Top-K clinical overlap counts to a comparable 0-1 range.
    if clin_overlap_col:
        out["Clinical_Overlap_Norm"] = (
            pd.to_numeric(
                df[clin_overlap_col],
                errors="coerce",
            )
            / float(top_k)
        )
    else:
        out["Clinical_Overlap_Norm"] = np.nan

    # Keep other clinical metrics on same bounded scale used
    # elsewhere.
    out["Weighted_Clinical_Overlap"] = (
        _normalize_01(df[weighted_clin_col])
        if weighted_clin_col
        else np.nan
    )

    out["Clinical_Recall"] = (
        _normalize_01(df[clin_recall_col])
        if clin_recall_col
        else np.nan
    )

    # Build three component scores first so final composite remains
    # easy to inspect.
    out["Robustness_Score"] = out[
        [
            "Noise_Stability",
            "Masking_Stability",
        ]
    ].mean(axis=1)

    out["Fidelity_Score"] = out[
        [
            "Permutation_Fidelity",
            "GradInput_Fidelity",
        ]
    ].mean(axis=1)

    out["Clinical_Score"] = out[
        [
            "Clinical_Overlap_Norm",
            "Weighted_Clinical_Overlap",
        ]
    ].mean(axis=1)
   

    # Combine components using reporting weights defined for this
    # summary view.
    out["Reliability_Score"] = (
        0.40 * out["Robustness_Score"] +
        0.35 * out["Fidelity_Score"] +
        0.25 * out["Clinical_Score"]
    )

    # Present highest-scoring methods first for straightforward
    # comparison.
    out = (
        out.sort_values(
            "Reliability_Score",
            ascending=False,
        )
        .reset_index(drop=True)
    )
    return out

def _build_risk_flags(
    model_auc: float, 
    method_df: pd.DataFrame, 
    agreement_df: pd.DataFrame
) -> list[tuple[str, str]]:
    """
    Build concise list of warning flags for current benchmark results.

    Returned list highlights model-performance concerns, robustness
    issues, and negative explainer agreement so dashboard can surface
    notable risks without requiring user to inspect each table
    manually.
    """
    # Collect each triggered issue as a title-and-description pair so
    # list can be rendered directly in summary sections.
    flags: list[tuple[str, str]] = []

    # Flag weak model discrimintation first because later explanation
    # results are harder to interpret when predictive signal itself is
    # poor.
    if model_auc < 0.50:
        flags.append(
            (
                "Model Performance Risk",
                (
                    f"ROC-AUC = {model_auc:.3f}, which is "
                    "worse than random."
                ),
            )
        )
    elif model_auc < 0.60:
        flags.append(
            (
                "Weak Model Signal",
                (
                    f"ROC-AUC = {model_auc:.3f}, indicating "
                    "weak discrimination."
                ),
            )
        )

    # Identify explanation methods with weak robustness under noise
    # perturbation.
    if "Stability_Spearman_Noise_Mean" in method_df.columns:
        poor_noise = (
            method_df.loc[
                pd.to_numeric(
                    method_df["Stability_Spearman_Noise_Mean"],
                    errors="coerce",
                ) < 0.40,
                "Method",
            ]
            .dropna()
            .map(_clean_method_name)
            .tolist()
        )

        if poor_noise:
            flags.append(
                (
                    "Noise Sensitivity",
                    (
                        "Low noise robustness detected for "
                        f"{', '.join(poor_noise)}."
                    ),
                )
            )

    # Identify explanation methods that drop below masking robustness
    # threshold.
    if "Stability_Spearman_Masking_Mean" in method_df.columns:
        poor_mask = (
            method_df.loc[
                pd.to_numeric(
                    method_df["Stability_Spearman_Masking_Mean"],
                    errors="coerce",
                ) < 0.40,
                "Method",
            ]
            .dropna()
            .map(_clean_method_name)
            .tolist()
        )

        if poor_mask:
            flags.append(
                (
                    "Masking Sensitivity",
                    (
                        f"Low masking robustness detected for "
                        f"{', '.join(poor_mask)}."
                    ),
                )
            )

    # Inpsect agreement export for negative global rank agreement
    # between explainers.
    agr = _get_agreement_row(agreement_df)
    if agr is not None:
        for label, col in [
            ("Captum IG vs SHAP", "SHAP_vs_Captum_Spearman"),
            ("Captum IG vs LIME", "LIME_vs_Captum_Spearman"),
            ("SHAP vs LIME", "SHAP_vs_LIME_Spearman"),
        ]:
            if col in agr.index:
                v = pd.to_numeric(agr[col], errors="coerce")
                if pd.notna(v) and v < 0:
                    flags.append(
                        (
                            "Conflicting Rankings",
                            (
                                f"{label} shows negative global "
                                f"agreement ({v:.3f})."
                            ),
                        )
                    )

    # Provide explicity all-clear message when no thresholds were
    # triggered.
    if not flags:
        flags.append(
            (
                "No Major Warning Flags",
                (
                    "No major reliability risks were triggered "
                    "by the current thresholds."
                ),
            )
        )

    return flags

# --------------------------------------------------------------------
# PDF / Downloads
# These helpers convert current benchmark outputs into downloadable
# artefacts so results can be reviewed outside live Streamlit session.
# --------------------------------------------------------------------
def make_pdf_report(
    *,
    title: str, 
    cfg: Any, 
    model_auc: float,
    method_df: pd.DataFrame, 
    agreement_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    fig_path: str | None,
    model_comparison_df: Optional[pd.DataFrame],
) -> bytes:
    """
    Build compact PDF summary of current benchmark run.

    Combines most reusable outputs from interface into one
    downloadable document. Includes run title, model ROC-AUC,
    configuration values, summary tables, and saved Top-K
    feature-importance figure when that image is available on disk.
    """
    # Create in-memory buffer first so PDF can be returned directly to
    # Streamlit without writing a temporary file to disk.
    buf = BytesIO()

    # Initialise reportlab canvas that will receive all text, tables,
    # and optional figure content for export.
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Define main page boundaries once so every drawing helper use
    # same printable area.
    left = 45
    right = width - 45
    usable_width = right - left
    top = height - 50
    bottom = 55

    # Centralise spacing values so PDF layout stays visually
    # consistent across headings, wrapped text, and table cells.
    line_gap = 12
    section_gap = 18
    small_gap = 6
    header_gap = 14
    cell_pad_x = 3
    cell_pad_y = 4

    def new_page(
        font_name: str = "Helvetica-Bold",
        font_size: int = 9,
    ) -> int:
        """
        Start new PDF page and restore the working font.

        Returning reset y-position allows calling code to continue
        drawing content without repeating page-setup logic.
        """
        # Advance to next page before any new content is drawn.
        c.showPage()

        # Reapply active font because reportlab resets page state when
        # new page begins.
        c.setFont(font_name, font_size)
        return top
   
    def wrap_text(
        text: str,
        max_width: int,
        font_name: str,
        font_size: int,
    ) -> List[str]:
        """
        Split text into multiple lines that fit within a target width.

        Used throughout PDF writer so long labels, values, and table
        headings remain readable instead of running beyond page or
        cell boundary.
        """
        # Normalise input first so later width checks always operate
        # on a string value.
        text = "" if text is None else str(text)
        words = text.split()

        # Return one empty line for blank inputs so calling code does
        # not need separate empty-text branch.
        if not words:
            return [""]
        lines = []
        line = words[0]

        # Build each line word by word and move to new line only when
        # current candidate would exceed the allowed width.
        for word in words[1:]:
            trial = f"{line} {word}"
            if (
                c.stringWidth(trial, font_name, font_size)
                <= max_width
            ):
                line = trial
            else:
                lines.append(line)
                line = word

        # Append final line after all words have been processed.
        lines.append(line)
        return lines

    def draw_wrapped_text(
        text: str,
        x: int,
        y: int,
        max_width: int,
        font_name: str = "Helvetica-Bold",
        font_size: int = 9,
        leading: int = 12,
    ) -> int:
        """
        Draw wrapped text and return next available vertical position.

        Keeps multi-line drawing behaviour consistent across
        configuration section and any other free-text content added
        later.
        """
        # Apply text style before drawing wrapped lines.
        c.setFont(font_name, font_size)
        lines = wrap_text(text, max_width, font_name, font_size)

        # Render each line in sequence and move cursor downward using
        # supplied line spacing.
        for line in lines:
            c.drawString(x, y, line)
            y -= leading
        return y

    def draw_section_title(text: str, y: int) -> int:
        """
        Draw one section heading and return next y-position.

        Keeping section-title rendering in one helper makes exported
        PDF easier to maintain if heading size or spacing is adjusted
        later.
        """
        # Use larger bold font so section titles are clearly separated
        # from table content and configuration values.
        c.setFont("Helvetica-Bold", 13)
        c.drawString(left, y, text)
        return y - header_gap
   
    def format_df_for_pdf(
        df: pd.DataFrame,
        max_rows: int = 8,
    ) -> pd.DataFrame:
        """
        Prepare DataFrame for compact PDF display.

        Export only needs a concise snapshot of each table, so this
        helper limites number of rows and converts values to short
        display strings.
        """
        # Work on a trimmed copy so PDF-specific formatting does not
        # modify original DataFrame used elsewhere in interface.
        out = df.copy().head(max_rows)

        # Format each column for display so numeric values are short
        # and text values remain printable inside PDF table cells.
        for col in out.columns:
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].map(
                    lambda v: (
                        ""
                        if pd.isna(v)
                        else f"{float(v):.3f}"
                    )
                )
            else:
                out[col] = out[col].astype(str)
        return out
  
    def draw_simple_table(
        df: pd.DataFrame, 
        y: int, 
        col_widths: list[int] | None = None, 
        font_size: int = 8,
        header_font_size: int = 8,
        leading: int = 10,
    ) -> int:
        """
        Draw compact grid-style table and return next y-position.

        Table renderer is intentionally simple because export only
        needs readable summary tables rather than fully styled
        spreadsheet layout.
        """
        # Show short placeholder when a section has no rows so PDF
        # still explains why table content is absent.
        if df.empty:
            c.setFont("Helvetica", 9)
            c.drawString(left, y, "No data available.")
            return y - line_gap
      
        # Convert incoming table to PDF-friendly display version
        # before widths, heights, and wrapped text are calculated.
        df = format_df_for_pdf(df)
        ncols = len(df.columns)

        # Use equal column widths when no custom layout has been
        # supplied for current table.
        if col_widths is None:
            base_w = usable_width / max(ncols, 1)
            col_widths = [base_w] * ncols

        header_lines = []

        # Wrap each header label to width of its column so long names
        # fit inside table boundary.
        for col, w in zip(df.columns, col_widths):
            header_lines.append(
                wrap_text(
                    str(col),
                    int(w - 2 * cell_pad_x),
                    "Helvetica-Bold",
                    header_font_size,
                )
            )

        # Size header row based on tallest wrapped heading.
        header_line_count = max(len(lines) for lines in header_lines)
        header_height = header_line_count * leading + 2 * cell_pad_y

        # Move to new page before drawing table header when there is
        # no longer enough vertical space on current page.
        if y - header_height < bottom:
            y = new_page("Helvetica", 9)
  
        x = left
        c.setFont("Helvetica-Bold", header_font_size)

        # Draw each header cell and then place its wrapped label
        # inside bordered rectangle.
        for col, w, lines in zip(
            df.columns,
            col_widths,
            header_lines,
        ):
            c.rect(
                x,
                y - header_height,
                w,
                header_height,
                stroke=1,
                fill=0,
            )
            text_y = y - cell_pad_y - header_font_size
            for line in lines:
                c.drawString(x + cell_pad_x, text_y, line)
                text_y -= leading
            x += w

        # Move below completed header row before drawing data rows.
        y -= header_height

        c.setFont("Helvetica", font_size)

        # Draw one bordered row at a time so row height can adjust
        # to wrapped cell content.
        for _, row in df.iterrows():
            wrapped_cells = []

            # Wrap every cell value using width of its destination
            # column.
            for val, w in zip(row.tolist(), col_widths):
                wrapped_cells.append(
                    wrap_text(
                        str(val),
                        int(w - 2 * cell_pad_x),
                        "Helvetica",
                        font_size,
                    )
                )

            # Use tallest cell to determine height of full row.
            row_line_count = max(
                len(lines)
                for lines in wrapped_cells
            )
            row_height = row_line_count * leading + 2 * cell_pad_y

            # Continue table on new page when next row would extend
            # beyond lower page margin.
            if y - row_height < bottom:
                y = new_page("Helvetica", 9)

            x = left

            # Draw each cell boundary and then place wrapped text
            # inside it.
            for lines, w in zip(wrapped_cells, col_widths):
                c.rect(
                    x,
                    y - row_height,
                    w,
                    row_height,
                    stroke=1,
                    fill=0,
                )
                text_y = y - cell_pad_y - font_size
                for line in lines:
                    c.drawString(x + cell_pad_x, text_y, line)
                    text_y -= leading
                x += w

            # Step downward to next row position.
            y -= row_height

        return y - small_gap
  
    # Start writing at top content boundary of first page.
    y = top

    # Draw report title first so document purpose is immediately
    # clear.
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, y, title)
    y -= section_gap

    # Display model ROC-AUC near top because it provides essential
    # context for explanation results that follow.
    c.setFont("Helvetica-Bold", 11)
    c.drawString(
        left,
        y,
        f"Model ROC-AUC (Test Subset): {model_auc:.3f}",
    )
    y -= header_gap

    # Introduce run-configuration section before listing individual
    # configuration fields.
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y, "Run Configuration")
    y -= header_gap

    # Convert configuration object to a dictionary when possible so
    # key-value pairs can be written line by line.
    try:
        cfg_dict = asdict(cfg)
    except (TypeError, ValueError):
        cfg_dict = {}

    # Write each configuration entry using wrapped text so longer
    # values remain inside printable page area.
    for k, v in cfg_dict.items():
        y = draw_wrapped_text(
            f"{k}: {v}", 
            left + 10, 
            y, 
            usable_width - 10, 
            font_name="Helvetica",
            font_size=9, 
            leading=11
        )

        # Start new page when remaining space becomes too small for
        # next configuration item.
        if y < bottom + 12:
            y = new_page("Helvetica", 9)

    y -= small_gap

    # Add model comparison section before explanation method summaries
    # so PDF captures broader predictive context for selected MLP.
    if (
        model_comparison_df is not None
        and not model_comparison_df.empty
    ):
        # Start new page when there is not enough vertical space to
        # render model comparison table cleanly.
        if y < 170:
            y = new_page()

        # Add section heading so comparison table is clearly
        # separated from explanation-specific outputs that follow.
        y = draw_section_title("Model Comparison", y)

        # Create a working copy of model comparison data so
        # filtering and transformations do not affect original
        # benchmarking ouputs.
        model_pdf_df = model_comparison_df.copy()

        # Convert key metric columns to numeric types to ensure
        # correct sorting and prevent formatting issues during
        # PDF rendering.
        for col in [
            "Used_for_Explanations",
            "Validation_AUC",
            "Test_AUC",
            "Test_AP",
            "Test_Brier",
            "Test_ECE",
        ]:
            if col in model_pdf_df.columns:
                model_pdf_df[col] = pd.to_numeric(
                    model_pdf_df[col],
                    errors="coerce",
                )
           
        # Build a reduced summary table with one representative row
        # per model. This avoids exporting multiple tuning rows
        # (e.g., repeated MLP entries) and ensures PDF presents a
        # clear model-level comparison.
        pdf_rows = []

        # Select MLP row used for explanations to reflect fixed model
        # used throughout benchmarking pipeline.
        if {
            "Model",
            "Used_for_Explanations",
        }.issubset(model_pdf_df.columns):
            mlp_used = model_pdf_df[
                (model_pdf_df["Model"] == "MLP") &
                (model_pdf_df["Used_for_Explanations"] == 1)
            ]
            if not mlp_used.empty:
                pdf_rows.append(mlp_used.iloc[[0]])

        # For comparator models, select best-performing
        # configuration based on held-out Test AUC where available,
        # or Validation AUC as a fallback.
        for model_name in ["Logistic Regression", "Random Forest"]:
            subset = model_pdf_df[
                model_pdf_df["Model"] == model_name
            ].copy()
            if subset.empty:
                continue

            # Sort models by Test AUC to identify strongest
            # configuration.
            if "Test_AUC" in subset.columns:
                subset = subset.sort_values(
                    "Test_AUC",
                    ascending=False,
                    na_position="last",
                )
            elif "Validation_AUC" in subset.columns:
                subset = subset.sort_values(
                    "Validation_AUC",
                    ascending=False,
                    na_position="last",
                )

            pdf_rows.append(subset.iloc[[0]])

        # If representative rows were successfully identified,
        # combine them into a compact table for display in PDF.
        if pdf_rows:
            model_pdf_df = pd.concat(pdf_rows, ignore_index=True)

        # Keep only compact PDF columns so table fits on page.
        if "Used_for_Explanations" in model_pdf_df.columns:
            model_pdf_df["Used_for_Explanations"] = (
                pd.to_numeric(
                    model_pdf_df["Used_for_Explanations"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                .map({1: "Yes", 0: "No"})
            )

        model_pdf_df = model_pdf_df.rename(
            columns={
                "Used_for_Explanations": "Explainer",
                "Validation_AUC": "Val AUC",
                "Test_AUC": "Test AUC",
                "Test_AP": "Test AP",
                "Test_Brier": "Brier",
                "Test_ECE": "ECE",
            }
        )

        model_cols = [
            c
            for c in [
                "Model",
                "Explainer",
                "Val AUC",
                "Test AUC",
                "Test AP",
                "Brier",
                "ECE",
            ]
            if c in model_pdf_df.columns
        ]

        y = draw_simple_table(
            (
                model_pdf_df[model_cols]
                if model_cols
                else pd.DataFrame()
            ),
            y,
            col_widths=(
                [110, 65, 65, 65, 65, 65, 65]
                if len(model_cols) == 7
                else None
            ),
            font_size=8,
            header_font_size=8,
            leading=10,
        )

        # Add spacing after table so next section does not begin
        # too close to model comparison output.
        y -= section_gap

    # Add method summary before more specialised agreement and
    # pairwise sections so PDF follows same broad-to-detailed
    # reading flow used in dashboard.
    if y < 180:
        y = new_page()
    y = draw_section_title("Method Summary", y)

    # Rename selected columns to shorter labels so they fit more
    # cleanly inside export table.
    method_df_display = method_df.rename(columns={
        "Stability_Spearman_Noise_Mean": "Noise Stability",
        "Stability_Spearman_Masking_Mean": "Masking Stability",
        "Fidelity_vs_Permutation_Spearman_Mean": "Fidelity (Perm)",
        "Fidelity_vs_GradInput_Spearman_Mean": "Fidelity (GradInput)",
        "Weighted_Clinical_Overlap": "Clinical Overlap",
        "Clinical_Recall": "Clinical Recall",
    })

    # Keep only summary columns that are actually present in current
    # run output.
    method_cols = [c for c in [
        "Method",
        "Noise Stability",
        "Masking Stability",
        "Fidelity (Perm)",
        "Fidelity (GradInput)",
        "Clinical Overlap",
        "Clinical Recall",
    ] if c in method_df_display.columns]

    # Draw method summary table using shortened labels prepared above.
    y = draw_simple_table(
        (
            method_df_display[method_cols]
            if method_cols
            else pd.DataFrame()
        ),
        y,
        col_widths=(
            [100, 105, 105, 125, 105, 95]
            if len(method_cols) == 6
            else None
        ),
    )
    y -= section_gap

    # Add agreement section next so cross-method consistency appears
    # before formal pairwise statistics.
    if y < 140:
        y = new_page()
    y = draw_section_title("Agreement Summary", y)

    # Select only agreement columns that exist in exported table.
    agreement_cols = [c for c in [
        "SHAP_vs_Captum_Spearman",
        "LIME_vs_Captum_Spearman",
        "SHAP_vs_LIME_Spearman",
    ] if c in agreement_df.columns]

    # Draw agreement summary using equal-width columns when all
    # three values are available.
    y = draw_simple_table(
        (
            agreement_df[agreement_cols]
            if agreement_cols
            else pd.DataFrame()
        ), 
        y,
        col_widths=(
            [usable_width / 3] * 3
            if len(agreement_cols) == 3
            else None
        ),
    )
    y -= section_gap

    # Add pairwise statistics after agreement table because it
    # provides more detailed comparison layer. Branch here so 
    # section can handle alternate data conditions safely.
    if y < 170:
        y = new_page()
    y = draw_section_title("Pairwise Statistical Comparison", y)

    # Keep pairwise PDF table compact and move long text below it.
    pairwise_cols = [
        c
        for c in [
            "Comparison",
            "Wilcoxon_stat",
            "p_value",
            "p_adj_bonferroni",
            "Rank_Biserial",
            "Effect_size_label",
        ]
        if c in pairwise_df.columns
    ]

    pairwise_pdf_df = pairwise_df[pairwise_cols].rename(
        columns={
            "Wilcoxon_stat": "Wilcoxon",
            "p_value": "p",
            "p_adj_bonferroni": "p adj.",
            "Rank_Biserial": "Rank bis.",
            "Effect_size_label": "Effect",
        }
    )

    y = draw_simple_table(
        (
            pairwise_pdf_df
            if pairwise_cols
            else pd.DataFrame()
        ),
        y,
        col_widths=(
            [120, 70, 55, 60, 70, 70]
            if len(pairwise_pdf_df.columns) == 6
            else None
        ),
        font_size=8,
        header_font_size=8,
        leading=10,
    )
    y -= section_gap

    if "Implication" in pairwise_df.columns and not pairwise_df.empty:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(left, y, "Pairwise Interpretation")
        y -= header_gap

        for _, row in pairwise_df.iterrows():
            implication = str(row.get("Implication", "")).strip()

            if not implication:
                continue

            if y < 90:
                y = new_page()

            y = draw_wrapped_text(
                implication,
                left,
                y,
                usable_width,
                font_name="Helvetica",
                font_size=9,
                leading=11,
            )
            y -= small_gap

    # Add saved figure on a separate page when a figure path has been
    # provided and image file still exists.
    if fig_path:
        fig_file = Path(fig_path)
        if fig_file.exists():
            y = new_page("Helvetica-Bold", 14)
            c.drawString(50, y, "Top-K Feature Importance Plot")
            y -= header_gap

            # Try to embed saved figure directly in PDF. If that
            # fails, replace image with a short explanatory note
            # instead of ending full export with an error.
            try:
                c.drawImage(
                    str(fig_file),
                    left,
                    120,
                    width=usable_width,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            except (TypeError, ValueError):
                c.setFont("Helvetica-Bold", 10)
                c.drawString(
                    left,
                    y - 20,
                    (
                        "(Could not embed figure image, but it "
                        "was generated on disk.)"
                    ),
                )

    # Finalise canvas so all content is written into in-memory
    # buffer.
    c.save()
    buf.seek(0)
    return buf.getvalue()

def _make_zip_bundle(files: Dict[str, bytes]) -> bytes:
    """
    Package named byte streams into one in-memory ZIP archive.

    Used by download controls so multiple exported files can
    be delivered through one button without creating a temporary
    archive on disk.
    """
    # Create archive in memory because interface only needs final
    # byte content for download.
    zip_buf = BytesIO()

    # Write each provided file entry into ZIP bundle under its
    # target name.
    with zipfile.ZipFile(
        zip_buf,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        # Iterate through available items so each option is handled
        # with same display logic.
        for name, content in files.items():
            z.writestr(name, content)

    # Rewind buffer so caller receives full archive from the start.
    zip_buf.seek(0)
    return zip_buf.getvalue()

# --------------------------------------------------------------------
# UI Rendering Helpers
# These helpers centralise repeated HTML fragments so headings, cards,
# captions, and dividers keep same structure throughout the interface.
# --------------------------------------------------------------------
def _section_header(title: str) -> None:
    """
    Render standard header block used for major dashboard sections.

    Using one helper keeps larger section labels visually consistent
    across full reporting page.
    """
    # Send one short HTML block to Streamlit so each major section
    # uses same wrapper element and CSS class.
    st.markdown(
        f"""
        <div class="section-header-box">{title}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def _subsection_header(title: str) -> None:
    """
    Render the shared subsection title style used within larger
    sections.

    This keeps secondary headings consistent without repeating the
    raw HTML in every analytical block.
    """
    # Use subsection CSS class so text inherits established visual
    # styling defined elsewhere in interface.
    st.markdown(
        f"<h3 class='sub-section'>{title}</h3>",
        unsafe_allow_html=True,
    )

def _info_card(
    title: str,
    value: str,
    subtitle: str = "",
    kind: str = "normal",
) -> None:
    """
    Render metric card for summary values shown across dashboard.

    Combines a label, a main value, and a short supporting line, while
    kind parameter selects appropriate visual state.
    """
    # Map requested card state to its CSS class so warning, success,
    # and default cards all share one construction path.
    klass = {
        "normal": "metric-card",
        "warning": "metric-card metric-warning",
        "danger": "metric-card metric-danger",
        "success": "metric-card metric-success",
    }.get(kind, "metric-card")

    # Assemble mtric card as one hTML fragment so Streamlit renders
    # title, value, and subtitle together.
    html = (
        f'<div class="{klass}">'
        f'<div class="metric-title">{title}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-subtitle">{subtitle}</div>'
        f'</div>'
    )    
    st.markdown(html, unsafe_allow_html=True)

def _insight_card(
    title: str,
    body: str,
    tone: str = "neutral",
    icon: str = "",
) -> None:
    """
    Render interpretation card for explanatory notes and summary
    messages.

    Used when interface needs a short narrative block rather
    than a single headline metric.
    """
    # Translate requested tone into matching CSS class so different
    # message types can share same underlying structure.
    css = {
        "neutral": "insight-card",
        "warning": "insight-card insight-warning",
        "danger": "insight-card insight-danger",
        "success": "insight-card insight-success",
        "primary": "insight-card insight-primary",
    }.get(tone, "insight-card")

    # Icon wrapper requires that an icon string has been passed
    # into helper.
    icon_html = (
        f"<div class='insight-icon'>{icon}</div>"
        if icon
        else ""
    )
  
    # Assemble title, optional icon, and body text into one styled
    # HTML block before sending it to Streamlit.
    html = (
        f'<div class="{css}">'
        f'{icon_html}'
        f'<div class="insight-title">{title}</div>'
        f'<div class="insight-body">{body}</div>'
        f'</div>'
    )
       
    st.markdown(html, unsafe_allow_html=True)

def _caption(text: str) -> None:
    """
    Render standard caption block displayed beneath visuals.

    Centralising caption rendering keeps explnatory text consitent
    across charts, tables, and summary components.
    """
    # Wrap caption in shared CSS class so all explantory notes use
    # same spacing and typography.
    st.markdown(
        f"<div class='caption-container'>{text}</div>",
        unsafe_allow_html=True,
    )

def _render_column_divider() -> None:
    """
    Render styled vertical divider used between dashboard columns.

    Divider is a presentation helper only. It provides visual
    separation between side-by-side panels without introducing
    additional content.
    """
    # Insert one styled HTML div that uses a vertical gradient
    # so divider remains visible without overpowering neighbouring
    # content.
    st.markdown(
        """
        <div style="
            height: 700px;
            width: 2px;
            margin: 0 auto;
            background: linear-gradient(
                180deg,
                rgba(79,129,189,0.0) 0%,
                #4F81BD 20%,
                #6FA8DC 50%,
                #4F81BD 80%,
                rgba(79,129,189,0.0) 100%
            );
            border-radius: 2px;
            opacity: 0.85;
        "></div>
        """,
        unsafe_allow_html=True,
    )

def _render_horizontal_divider() -> None:
    """
    Render styled horizontal divider used between major content
    blocks.

    Keeps section-break styling consistent wherever a full-width
    divider is needed.
    """
    # Insert one full-width gradient divider so adjacent dashboard
    # blocks are seperated without repeating raw HTML in each section.
    st.markdown(
        """
        <div style="
            height: 3px;
            width: 100%;
            margin: 1.6rem 0 1.8rem 0;
            background: linear-gradient(
                90deg,
                rgba(29,129,189,0.0) 0%,
                #4F81BD 20%,
                #6FA8DC 50%,
                #4F81BD 80%,
            rgba(79,129,189,0.0) 100%
            );
            border-radius: 2px;
            opacity: 0.85;
        "></div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------
# Dynamic captions for charts
# --------------------------------------------------------------------
def _render_model_context_caption(model_auc: float) -> None:
    """
    Render caption used in model-context section.

    Gives reader context for rest of dashboard by connected current
    ROC-AUC value to level of confidence that can be placed in later
    explanation comparisons.

    Frames later explanation results in context of current
    predictive performance level. 
    """
    try:
        # Stop early when model score is missing so interface does
        # not display partially formed caption.
        if pd.isna(model_auc):
            return

        # Select interpretation sentence that matches current model
        # performance band used in this run.  
        if model_auc >= 0.80:
            insight = (
                "Strong model performance; explanation differences "
                "likely reflect model behaviour."
            )
        elif model_auc >= 0.70:
            insight = (
                "Adequate performance; interpret explanation "
                "differences with supporting metrics."
            )
        elif model_auc >= 0.60:
            insight = (
                "Moderate performance; differences may reflect "
                "model limitations."
            )
        else:
            insight = (
                "Low performance; explanation outputs may "
                "be unreliable."
            )
      
        # Render caption as a syled HTML bloc so it matches visual
        # presentation used throughout later reporting sections.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 0.6rem;
                max-width: 75%;
                margin-right: auto;
                margin-left: auto;
                padding: 0 12px;
                line-height: 1.55
            ">
                Model performance provides context for interpreting
                all results. {insight} Subsequent charts reflect
                explanation behaviour under this model quality.
            </div>
            """,
            unsafe_allow_html=True,
        )
    except (KeyError, IndexError, TypeError, ValueError):
        # Keep caption rendering non-blocking so a formatting issue
        # here does not interrupt rest of dashboard.
        pass

def _render_best_methods_caption() -> None:
    """
    Render caption used in best-methods summary section.

    Fixed explanatory caption because surrounding cards already
    contain run-specific winners and do not require additional data
    lookup.
    """
    # Render consistent explanatory note beneath best-method summary
    # so reader understands how to interpret dominance across
    # categories.
    st.markdown(
        """
        <div style="
            text-align: center;
            font-size: 0.95rem;
            color: #2F3E4E;
            margin-top: 2.0rem;
            max-width: 620px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.5;
        ">
            This section shows the top method across key evaluation
            dimensions. Consistent perfromance indicates a more
            reliable approach, while variation reflects trade-offs.
        </div>
        """,
        unsafe_allow_html=True,
    )

def _render_tradeoff_caption(rel_df: pd.DataFrame) -> None:
    """
    Render caption for robustness-fidelity-clinical trade-off visual.

    Identifies strongest and weakest overall profiles using
    reliability score already computed upstream.
    """
    try:
        # Work on a copy so caption helper does not alter table reused
        # by nearby charts or summary cards.
        df = rel_df.copy()

        # Standardise method label type before selecting rows that
        # will be referenced in caption text.
        df["Method"] = df["Method"].astype(str)

        # Only coninue when summary score needed for ranking is
        # available.
        score_col = (
            "Reliability_Score"
            if "Reliability_Score" in df.columns
            else None
        )

        if score_col is None:
            return

        # Rank methods so caption can describe highest- and lowest-
        # scoring overall profiles in the visual.
        best_row = df.sort_values(score_col, ascending=False).iloc[0]
        worst_row = df.sort_values(score_col, ascending=True).iloc[0]
      
        best_method = best_row["Method"]
        worst_method = worst_row["Method"]

        # Render completed interpretation below chart.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 0.6rem;
                max-width: 620px;
                margin-left: auto;
                margin-right: auto;
            ">
                This chart compares method performance across
                key evaluation dimensions. {best_method} shows
                the strongest overall profile, while {worst_method} 
                is less consistent. More balanced profiles indicate
                more reliable performance across metrics.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except (KeyError, IndexError, TypeError, ValueError):
        # Keep rest of page usable even if caption cannot be built.
        pass

def _render_reliability_caption(rel_df: pd.DataFrame) -> None:
    """
    Render caption for composite reliability comparison chart.

    Reports which method has highest reliability score and
    places that result in context of explanation consistency.
    """
    try:
        # Use a defensive copy so later interface sections still
        # receive original reliability table unchanged.
        df = rel_df.copy()

        # Order rows by precomputed reliability score to identify
        # strongest and weakest methods in chart.
        best_row = df.sort_values(
            "Reliability_Score",
            ascending=False,
        ).iloc[0]
        worst_row = df.sort_values(
            "Reliability_Score",
            ascending=True,
        ).iloc[0]
       
        best_method = best_row["Method"]
        best_score = best_row["Reliability_Score"]
        worst_method = worst_row["Method"]

        # Render summary text below figure using same centred
        # caption style applied in later reporting helpers.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 4.5rem;
                max-width: 620px;
                margin-left: auto;
                margin-right: auto;
            ">
                This chart compares composite reliability scores
                across methods. {best_method} has the highest score
                ({best_score}), while {worst_method} is substantially
                lower. Higher scores indicate more consistent and
                stable explanation behaviour.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except (KeyError, IndexError, TypeError, ValueError):
        # Fail quietly so one caption issue does not stop broader
        # interface.
        pass

def _render_reliability_decomposition_caption(
    rel_df: pd.DataFrame,
) -> None:
    """
    Render caption for reliability-decomposition chart.

    Explains how each method's overall reliability score is
    distributed across component dimensions shown in stacked chart.
    """
    try:
        # Copy input table so caption logic remains isolated from
        # data structure used elsewhere in page.
        df = rel_df.copy()

        # Reuse overall reliability score to identify strongest and
        # weakest methods referenced in decomposition explanation.
        best_row = df.sort_values(
            "Reliability_Score",
            ascending=False,
        ).iloc[0]
        worst_row = df.sort_values(
            "Reliability_Score",
            ascending=True,
        ).iloc[0]
      
        best_method = best_row["Method"]
        worst_method = worst_row["Method"]

        # Render explanatory note that clarifies how this
        # decomposition chart differs from nearby overall-score chart.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 0.6rem;
                max-width: 620px;
                margin-left: auto;
                margin-right: auto;
            ">
                This chart shows how each method's normalised
                reliability score is distributed across robustness,
                fidelity, and clinical alignment. {best_method} is
                more balanced across components, while {worst_method}
                is driven by fewer dimensions. More even distributions
                indicate more consistent performance across metrics.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except (KeyError, IndexError, TypeError, ValueError):
        # Keep dashboard resilient if caption cannot be computed.
        pass

def _render_stability_caption(method_df: pd.DataFrame) -> None:
    """
    Render caption for stability chart.

    Compares average performance of each method across noise
    and masking stability metrics displayed in chart.
    """
    try:
        # Work on a local copy so any temporary columns created
        # here are not carried into later sections.
        df = method_df.copy()

        # Standardise method names so caption matches labels shown in
        # chart legend and summary cards.
        df["Method"] = df["Method"].map(_clean_method_name)

        # Locate stability columns using shared schema helper so
        # caption remains compatible with minor export-name
        # variations.
        noise_col = _first_existing(
            df,
            ["Stability_Spearman_Noise_Mean"],
        )
        mask_col = _first_existing(
            df,
            ["Stability_Spearman_Masking_Mean"],
        )

        if not noise_col or not mask_col:
            return
      
        # Average the two perturbation results so caption can 
        # summarise overall stability pattern in one comparison.
        df["Avg_Stability"] = (
            pd.to_numeric(df[noise_col], errors="coerce") +
            pd.to_numeric(df[mask_col], errors="coerce")
        ) / 2

        best_method = df.sort_values(
            "Avg_Stability",
            ascending=False,
        ).iloc[0]["Method"]
        worst_method = df.sort_values(
            "Avg_Stability",
            ascending=True,
        ).iloc[0]["Method"]

        # Render interpretation text below chart.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 0.6rem;
                max-width: 620px;
                margin-left: auto;
                margin-right: auto;
            ">
                Stability is measured using Spearman rank correlation
                of feature importance. {best_method} remains more
                stable across conditions, while {worst_method} shows 
                greater decline. Higher values indicate more consistent
                explanations under perturbations.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except (KeyError, IndexError, TypeError, ValueError):
        # Preserve page rendering even if this caption cannot be
        # produced.
        pass

def _render_fidelity_caption(method_df: pd.DataFrame) -> None:
    """
    Render caption for fidelity chart.

    Summarises agreement with two fidelity reference baselines
    used in method-level comparison view.
    """
    try:
        # Copy table before preparing temporary summary columns.
        df = method_df.copy()
        df["Method"] = df["Method"].map(_clean_method_name)

        # Identify available fidelity columns using shared helper.
        perm_col = _first_existing(
            df,
            ["Fidelity_vs_Permutation_Spearman_Mean"],
        )
        grad_col = _first_existing(
            df,
            ["Fidelity_vs_GradInput_Spearman_Mean"],
        )

        if not perm_col or not grad_col:
            return
      
        # Combine two fidelity references into one average score so
        # caption can describe overall alignment in a compact way.
        df["Avg_Fidelity"] = (
            pd.to_numeric(df[perm_col], errors="coerce") +
            pd.to_numeric(df[grad_col], errors="coerce")
        ) / 2

        best_method = df.sort_values(
            "Avg_Fidelity",
            ascending=False,
        ).iloc[0]["Method"]
        worst_method = df.sort_values(
            "Avg_Fidelity",
            ascending=True,
        ).iloc[0]["Method"]

        # Render final caption below chart.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 0.6rem;
                max-width: 620px;
                margin-left: auto;
                margin-right: auto;
            ">
                This chart evaluates how closely each method aligns
                with model behaviour using permutation and
                gradient-based baselines. {best_method} shows the
                strongest agreement, while {worst_method} shows
                weaker alignment. Higher fidelity indicates closer
                correspondence with model behaviour.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except (KeyError, IndexError, TypeError, ValueError):
        # Avoid blocking interface if caption formatting fails.
        pass

def _render_topk_stability_overlap_caption(
    method_df: pd.DataFrame,
    top_k: int,
) -> None:
    """
    Render caption for Top-K stability-overlap chart.

    Compares how well each method preserves its highest-ranked
    features across noise and masking perturbation conditions.
    """
    try:
        # Define candidate column names so caption still works when
        # exported schema uses one of several Top-K naming patterns.
        noise_candidates = [
            f"Stability_Top{top_k}_Noise_Overlap_Mean",
            f"Stability_Top{top_k}_Noise_Overlap",
            f"Top{top_k}_Stability_Noise_Overlap",
            "Top10_Stability_Noise_Overlap",
        ]
        mask_candidates = [
            f"Stability_Top{top_k}_Masking_Overlap_Mean",
            f"Stability_Top{top_k}_Masking_Overlap",
            f"Top{top_k}_Stability_Masking_Overlap",
            "Top10_Stability_Masking_Overlap",
        ]
      
        # Locate first available pair of overlap columns before
        # continuing.
        noise_col = _first_existing(method_df, noise_candidates)
        mask_col = _first_existing(method_df, mask_candidates)

        if (
            not noise_col
            or not mask_col
            or "Method" not in method_df.columns
        ):
            return
  
        # Prepare local working table for caption-specific
        # calculations.
        df = method_df.copy()
        df["Method"] = df["Method"].map(_clean_method_name)
        df["Noise"] = pd.to_numeric(df[noise_col], errors="coerce")
        df["Masking"] = pd.to_numeric(df[mask_col], errors="coerce")

        # Average overlap summarises overall Top-K preservation,
        # while gap shows how similarly method behaves across the
        # two perturbations.
        df["Avg_Overlap"] = df[["Noise", "Masking"]].mean(axis=1)
        df["Gap"] = (df["Noise"] - df["Masking"]).abs()

        best_row = df.sort_values(
            "Avg_Overlap",
            ascending=False,
        ).iloc[0]
        most_consistent_row = df.sort_values(
            "Gap",
            ascending=True,
        ).iloc[0]
      
        best_method = best_row["Method"]
        best_score = best_row["Avg_Overlap"]
        consistent_method = most_consistent_row["Method"]

        # Render interpretation below chart using same caption
        # pattern as the later reporting helpers.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 0.6rem;
                max-width: 620px;
                margin-left: auto;
                margin-right: auto;
            ">
                This chart compares how consistently each method
                preserves Top-{top_k} feature rankings under noise
                and masking. {best_method} shows the highest overlap
                ({best_score:.2f}), while {consistent_method} behaves
                most similarly across both conditions. Higher overlap
                indicates more stable feature rankings under
                perturbations.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except (KeyError, IndexError, TypeError, ValueError):
        # Do not let a caption error interrupt rest of results page.
        pass

def _render_fidelity_robustness_tradeoff_caption(
    rel_df: pd.DataFrame,
) -> None:
    """
    Render caption for fidelity-versus-robustness trade-off chart.

    Summarises which method sits closest to the desirable upper-right
    region of comparison plot, where both robustness and fidelity are
    higher.
    """
    try:
        # Require the fields used to position methods on trade-off
        # chart.
        needed = {"Method", "Robustness_Score", "Fidelity_Score"}
        if rel_df.empty or not needed.issubset(rel_df.columns):
            return

        # Work on a copy so later sections can still use original
        # table unchanged.
        df = rel_df.copy()

        # Standardise method labels before they are inserted into
        # caption text.
        df["Method"] = df["Method"].map(_clean_method_name)

        # Coerce plotting metrics to numeric form so ranking logic
        # remains consistent.
        df["Robustness_Score"] = pd.to_numeric(
            df["Robustness_Score"],
            errors="coerce",
        )
        df["Fidelity_Score"] = pd.to_numeric(
            df["Fidelity_Score"],
            errors="coerce",
        )

        # Measure how far each method sits from ideal top-right
        # position. Smaller values indicate a better balance
        # between the two dimensions.
        df["Distance_to_TopRight"] = (
            (1 - df["Robustness_Score"]) ** 2 +
            (1 - df["Fidelity_Score"]) ** 2.
        )

        # Identify method with strongest combined position on chart.
        best_balanced_row= df.sort_values(
            "Distance_to_TopRight",
            ascending=True,
        ).iloc[0]
       
        # Identify weakest overall position using both plotted
        # dimensions.
        weakest_row = df.sort_values(
            ["Robustness_Score", "Fidelity_Score"],
            ascending=[True, True]
        ).iloc[0]
       
        best_method = best_balanced_row["Method"]
        weak_method = weakest_row["Method"]

        # Render one interpretation block below chart.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 0.6rem;
                max-width: 620px;
                margin-left: auto;
                margin-right: auto;
            ">
                This chart positions methods by robustness and
                fidelity, showing the trade-off between stability
                and alignment with model behaviour. {best_method}
                is closest to the top-right, indicating the strongest
                combined performance, while {weak_method} scores
                lower on both dimensions. Methods nearer the top-right
                show more stable and better-aligned explanations.
            </div>
            """,
            unsafe_allow_html=True
        )

    except (TypeError, ValueError):
        # Suppress local caption failures so wider dashboard
        # still renders.
        pass

def _render_clinical_alignment_summary_caption(
    method_df: pd.DataFrame,
    top_k: int,
) -> None:
    """
    Render caption for clinical-alignment summary cards.

    Explains how Top-K comparison and broader clinical
    overlap metrics should be interpreted together.
    """
    try:
        # Method summary table supplies the method-level metrics
        # used here.
        if "Method" not in method_df.columns:
            return

        # Work on a copy so downstream sections can still reuse
        # original export.
        d = method_df.copy()

        # Standardise method names before they are inserted into
        # explanatory text.
        d["Method"] = d["Method"].map(_clean_method_name)

        # Retrive available clinical-alignment columns for selected
        # Top-K.
        top_overlap_col = _first_existing(
            d,
            [
                f"Top{top_k}_Clinical_Overlap",
                "Top10_Clinical_Overlap",
            ],
        )
        weighted_col = _first_existing(
            d,
            ["Weighted_Clinical_Overlap"],
        )

        # Identify which methods lead main overlap views used
        # in summary.                 
        best_overlap = (
            d.sort_values(
                top_overlap_col,
                ascending=False,
            ).iloc[0]["Method"]
            if top_overlap_col
            else ""
        )
        best_weighted = (
            d.sort_values(
                weighted_col,
                ascending=False,
            ).iloc[0]["Method"]
            if weighted_col
            else ""
        )

        # Tailor middle sentence so caption reflects whether one
        # method leads both views.
        if best_overlap == best_weighted:
            middle_sentence = (
                f"{best_overlap} achieves the strongest performance "
                "across both Top-K overlap and weighted overlap, "
                "indicating consistently strong clinical alignment."
            )
        else:
            middle_sentence = (
                f"{best_overlap} achieves the strongest Top-K "
                f"overlap, while {best_weighted} shows the highest "
                "weighted overlap, indicating broader feature "
                "relevance."
            )

        # Render longer explanatory block used below the clinical
        # summary cards.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 1.5rem;
                max-width: 75%;
                margin-left: auto;
                margin-right: auto;
                padding: 0 12px;
            ">
            This summary evaluates how well each method prioritises
            clinically relevant features using Top-{top_k} overlap,
            weighted overlap, precision, and recall against a fixed
            clinical reference set. {middle_sentence} Higher overlap
            and precision indicate stronger alignment within
            top-ranked features, while recall reflects coverage of
            the full clinical set.
            </div>
            """,
            unsafe_allow_html=True
        )

    except (KeyError, IndexError, TypeError, ValueError):
        # Keep caption issues isolated so main dashboard content still
        # loads.
        pass

def _render_clinical_coverage_analysis_caption(
    method_df: pd.DataFrame,
    top_k: int,
) -> None:
    """
    Render caption for clinical coverage analysis chart.

    Explains how many clinically validated variables each method
    captures within its current Top-K feature set.
    """
    try:
        # Locate Top-K overlap column that matches current run schema.
        overlap_col = _first_existing(
            method_df,
            [
                f"Top{top_k}_Clinical_Overlap",
                "Top10_Clinical_Overlap",
            ],
        )
        if overlap_col is None or "Method" not in method_df.columns:
            return

        # Work on a copy so original method table remains available
        # elsewhere.
        df = method_df.copy()

        # Standardise method names before they are displayed in
        # caption.
        df["Method"] = df["Method"].map(_clean_method_name)

        # Convert overlap values to numeric counts for comparison
        # text.
        df["Captured"] = pd.to_numeric(
            df[overlap_col],
            errors="coerce",
        )

        # Drop rows that do not have usable coverage values before
        # selecting strongest and weakest methods.
        df = df.dropna(subset=["Captured"])
        if df.empty:
            return

        # Derive complement so chart can be described as captured
        # versus missed.
        df["Missed"] = float(top_k) - df["Captured"]
        df["Missed"] = df["Missed"].clip(lower=0)

        # Find strongest and weakest coverage results for selected
        # Top-K.
        best_row = df.sort_values("Captured", ascending=False).iloc[0]
        worst_row = df.sort_values("Captured", ascending=True).iloc[0]

        best_method = best_row["Method"]
        best_captured = int(best_row["Captured"])
        worst_method = worst_row["Method"]
        worst_captured = int(worst_row["Captured"])

        # Adjust wording when methods capture same number of clinical
        # features.
        if best_captured == worst_captured:
            middle_sentence = (
                f"All methods capture a similar number of clinically "
                f"validated features within the Top={top_k}, "
                "indicating limited separation by this metric."
            )
        else:
            middle_sentence = (
                f"{best_method} captures the most clinically "
                f"validated features ({best_captured}/{top_k}), "
                f"while {worst_method} captures fewer "
                f"({worst_captured}/{top_k})."
            )

        # Render caption directly below chart.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 0.6rem;
                max-width: 620px;
                margin-left: auto;
                margin-right: auto;
            ">
                This chart shows how many clinically validated
                features are captured within each method's Top-{top_k}
                ranked variables. {middle_sentence} Higher captured
                counts indicate stronger coverage of clinically
                relevant features.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except (KeyError, IndexError, TypeError, ValueError) as e:
        # Surface caption problem locally because this section
        # already reports caption errors.
        st.warning(f"Clinical coverage analysis caption error: {e}")

def _render_clinical_feature_consistency_caption(
    feature_df: pd.DataFrame,
) -> None:
    """
    Render caption for clinical feature consistency heatmap.

    Compares relative emphasis each explanation methods places on
    subset of features marked as clinically validated.
    """
    try:
        # Heatmap requires features names, three method columns,
        # and clinical flag.
        df = feature_df.copy()
        needed = {
            "Feature",
            "Captum_IG",
            "SHAP",
            "LIME",
            "Clinically_Validated",
        }
        if not needed.issubset(df.columns):
            return

        # Standardise clinical indicator before filtering feature set.
        df["Clinically_Validated"] = pd.to_numeric(
            df["Clinically_Validated"], errors="coerce"
        ).fillna(0).astype(int)

        # Keep only clinically validated rows used by heatmap.
        df = df[df["Clinically_Validated"] == 1].copy()

        if df.empty:
            return
        
        # Convert each method column to absolute numeric importance
        # values.
        for col in ["Captum_IG", "SHAP", "LIME"]:
            df[col] = (
                pd.to_numeric(
                    df[col],
                    errors="coerce",
                )
                .abs()
                .fillna(0)
            )
        
        # Summarise average emphasis each method places on validated
        # feature set.
        method_means = {
            "Captum IG": df["Captum_IG"].mean(),
            "SHAP": df["SHAP"].mean(),
            "LIME": df["LIME"].mean(),
        }

        strongest_method = max(method_means, key=method_means.get)
        weakest_method = min(method_means, key=method_means.get)

        # Render interpretation block below heatmap.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 0.6rem;
                max-width: 620px;
                margin-left: auto;
                margin-right: auto;
            ">
                This heatmap shows the average importance each method
                assigns to clinically validated features.
                {strongest_method} places the greatest emphasis, while
                {weakest_method} assigns lower importance. Stronger
                and more consistent attribution across these features
                indicates closer alignment with clinical expectations.
            </div>
            """,
            unsafe_allow_html=True
        )

    except (KeyError, IndexError, TypeError, ValueError) as e:
        # Keep rest of section available even if caption calculation
        # fails.
        st.warning(f"Clinical feature consistency caption error: {e}")

def _render_clinical_recall_curve_caption(
    feature_df: pd.DataFrame,
    max_k: int,
) -> None:
    """
    Render caption for clinical coverage depth plot.

    Explains which method accumulates clinically validated features
    more quickly as Top-K threshold expands.
    """
    try:
        # Recall curve relies on feature names, method scores,
        # and clinical flag.
        df = feature_df.copy()
        needed = {
            "Feature",
            "Captum_IG",
            "SHAP",
            "LIME",
            "Clinically_Validated"
        }
        if not needed.issubset(df.columns):
            return
        
        # Standardise clinical indicator before computing recall
        # values.
        df["Clinically_Validated"] = pd.to_numeric(
            df["Clinically_Validated"], errors="coerce"
        ).fillna(0).astype(int)
        
        total_clinical = int(df["Clinically_Validated"].sum())
        if total_clinical == 0:
            return
        
        # Store recall-at-max-k value for each explanation method.
        method_scores = {}

        # Evaluate each method against same clinically validated
        # reference set.
        for col, name in zip(
            ["Captum_IG", "SHAP", "LIME"],
            ["Captum IG", "SHAP", "LIME"],
        ):
            temp = df[["Feature", "Clinically_Validated", col]].copy()
            
            # Convert one method column to absolute numeric magnitude
            # before ranking.
            temp[col] = pd.to_numeric(
                temp[col],
                errors="coerce",
            ).abs()

            # Rank features from highest to lowest importance for
            # recall calculation.
            temp = (
                temp.dropna(
                    subset=[col],
                )
                .sort_values(
                    col,
                    ascending=False,
                )
            )

            # Measure how much of validated feature set appears within
            # current depth.
            topk = temp.head(max_k)
            recall = int(
                topk["Clinically_Validated"].sum()
            ) / float(total_clinical)
            method_scores[name] = recall

        best_method = max(method_scores, key=method_scores.get)
        worst_method = min(method_scores, key=method_scores.get)

        # Render explanatory caption below chart.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 0.6rem;
                max-width: 620px;
                margin-left: auto;
                margin-right: auto;
            ">
                This curve shows how quickly each method captures
                clinically validated features as more top-ranked
                variables are included. {best_method} reaches higher
                recall earlier, while {worst_method} increases more
                gradually. Steeper curves indicate that clinically
                relevant features appear earlier in the ranking.
            </div>
            """,
            unsafe_allow_html=True
        )

    except (KeyError, IndexError, TypeError, ValueError) as e:
        # Report local caption issues without interrupting wider page.
        st.warning(f"Clinical coverage analysis caption error: {e}")

def _render_global_agreement_caption(
    agreement_df: pd.DataFrame,
) -> None:
    """
    Render caption for global agreement heatmap.

    Summarises which pair of explanation methods is most aligned and
    which pair diverges most in global feature ranking.
    """
    try:
        # Retrieve one agreement row used throughout agreement
        # section.
        agr = _get_agreement_row(agreement_df)
        if agr is None:
            return
        
        # Extract pairwise global agreement scores used by heatmap.
        shap_captum = agr.get("SHAP_vs_Captum_Spearman")
        lime_captum = agr.get("LIME_vs_Captum_Spearman")
        shap_lime = agr.get("SHAP_vs_LIME_Spearman")

        # Store available pair labels and values for comparison text.
        pairs = {
            "Captum IG and SHAP": shap_captum,
            "Captum IG and LIME": lime_captum,
            "SHAP AND LIME": shap_lime,
        }

        # Drop missing values so only valid method pairs are compared.
        pairs = {k: v for k, v in pairs.items() if pd.notna(v)}
        if not pairs:
            return
        
        strongest_pair = max(pairs, key=lambda k: pairs[k])
        weakest_pair = min(pairs, key=lambda k: pairs[k])

        strongest_val = pairs[strongest_pair]
        weakest_val = pairs[weakest_pair]

        # Translate strongest score into concise descriptive wording.
        if strongest_val > 0.7:
            agreement_desc = "strong agreement"
        elif strongest_val > 0.4:
            agreement_desc = "moderate agreement"
        else:
            agreement_desc = "limited agreement"

        # Render summary interpretation below heatmap.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 0.6rem;
                max-width: 620px;
                margin-left: auto;
                margin-right: auto;
            ">
                This heatmap shows rank correlation between methods
                based on global feature importance. {strongest_pair}
                show the highest alignment ({strongest_val:.2f})
                {agreement_desc}, while {weakest_pair} show the lowest
                agreement ({weakest_val:.2f}). Higher correlation 
                indicates similar feature rankings, while low or 
                negative values suggest differing behaviour.
            </div>
            """,
            unsafe_allow_html=True
        )

    except (KeyError, IndexError, TypeError, ValueError) as e:
        # Keep agreement section visible even if caption generation
        # fails.
        st.warning(f"Clinical feature consistency caption error: {e}")

def _render_agreement_key_insights_caption(
    agreement_df: pd.DataFrame,
) -> None:
    """
    Render caption for key agreement insights cards.

    Translates pairwise agreement values into one short summary
    sentence for agreement highlight cards.
    """
    try:
        # Retrieve agreement values already used elsewhere in
        # this section.
        agr = _get_agreement_row(agreement_df)
        if agr is None:
            return
        
        # Build labelled method-pair list used for
        # strongest-versus-weakest text.
        pairs = [
            ("Captum IG vs SHAP", _safe_float(
                agr.get("SHAP_vs_Captum_Spearman"))
            ),
            ("Captum IG vs LIME", _safe_float(
                agr.get("LIME_vs_Captum_Spearman"))
            ),
            ("SHAP vs LIME", _safe_float(
                agr.get("SHAP_vs_LIME_Spearman"))
            ),
        ]

        # Drop unavailable values before comparing pairwise agreement.
        pairs = [(name, val) for name, val in pairs if pd.notna(val)]
        if not pairs:
            return
        
        strongest_label, strongest_value = max(
            pairs,
            key=lambda x: x[1],
        )
        weakest_label, weakest_value = min(pairs, key=lambda x: x[1])

        # Adjust sentence structure based on whether agreement is 
        # similar or negative.
        if strongest_label == weakest_label:
            middle_sentence = (
                "All method pairs show a similar level of agreement, "
                "indicating limited seperation in global "
                "feature-ranking consistency."
            )
        elif weakest_value < 0:
            middle_sentence = (
                f"{strongest_label} shows the strongest agreement "
                f"({strongest_value:.3f}), while {weakest_label} "
                f"shows negative agreement ({weakest_value:.3f})."
            )
        else:
            middle_sentence = (
                f"{strongest_label} shows the strongest agreement "
                f"({strongest_value:.3f}), while {weakest_label} "
                f"shows weaker agreement ({weakest_value:.3f})."
            )
        
        # Render explanatory block below agreement insight cards.
        st.markdown(
        f"""
        <div style="
            text-align: center;
            font-size: 0.95rem;
            color: #2F3E4E;
            margin-top: 0.6rem;
            max-width: 620px;
            margin-left: auto;
            margin-right: auto;
        ">
            This section summarises agreement between methods 
            in global feature importance rankings. {middle_sentence} 
            Higher agreement indicates more consistent behaviour,
            while lower or negative values suggest divergence across
            methods. 
        </div>
        """,
        unsafe_allow_html=True
    )
        
    except (KeyError, IndexError, TypeError, ValueError) as e:
        # Limit caption failures to this local block.
        st.warning(f"Key agreement insights caption error: {e}")

def _render_topk_importance_caption(
    feature_df: pd.DataFrame, 
    top_k: int = 10, 
    baseline_method: str = "Captum_IG"
) -> None:
    """
    Render caption for Top-K feature importance chart.

    Describes how explanation methods compare over baseline
    Top-K feature set used in feature-importance visual.
    """
    try:
        # Require feature name column and one score column for each
        # method.
        df = feature_df.copy()
        needed = {"Feature", baseline_method, "SHAP", "LIME"}
        if not needed.issubset(df.columns):
            return

        # Convert each method column to absolute numeric values
        # before comparison.
        for c in [baseline_method, "SHAP", "LIME"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").abs()
    
        # Keep only Top-K rows defined by baseline method ranking.
        df = df.sort_values(
            baseline_method,
            ascending=False,
        ).head(top_k)

        # Summarise average magnitude assigned by each explanation
        # method.
        method_means = {
            "Captum IG": df[baseline_method].mean(),
            "SHAP": df["SHAP"].mean(),
            "LIME": df["LIME"].mean(),
        }

        strongest_method = max(method_means, key=method_means.get)
        weakest_method = min(method_means, key=method_means.get)

        # Measure spread across methods to find most disputed feature.
        df["Spread"] = (
            df[[baseline_method, "SHAP", "LIME"]].max(axis=1)
            - df[[baseline_method, "SHAP", "LIME"]].min(axis=1)
        )

        high_variance_feature = (
            df.sort_values(
                "Spread",
                ascending=False,
            )
            .iloc[0]["Feature"]
        )

        # Render interpretation below chart.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 0.6rem;
                max-width: 620px;
                margin-left: auto;
                margin-right: auto;
            ">
                This chart shows the Top-{top_k} features identified 
                by each method based on absolute attribution.
                {strongest_method} assigns higher overall importance,
                while {weakest_method} assigns lower values. The
                largest difference occurs for '{high_variance_feature}',
                where methods diverge most in ranking.
            </div>
            """,
            unsafe_allow_html=True
        )

    except (KeyError, IndexError, TypeError, ValueError) as e:
        # Keep figure rendering available eve if caption cannot be
        # created.
        st.warning(f"Clinical coverage analysis caption error: {e}")

def _render_feature_variability_caption(
    feature_df: pd.DataFrame,
    top_n: int = 10,
) -> None:
    """
    Render caption for feature attribution variability chart.

    Highlights which featuers show most disagreement across
    explanation methods within plotted subset.
    """
    try:
        # Determine which explainer columns are available in current
        # feature table.
        cols = [
            c
            for c in ["Captum_IG", "SHAP", "LIME"]
            if c in feature_df.columns
        ]
        if "Feature" not in feature_df.columns or len(cols) < 2:
            return
        
        # Work on a copy so later feature views still receive original
        # table.
        df = feature_df.copy()

        # Convert all available explainer columns to absolute numeric
        # values.
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").abs()
    
        # Compute row-level spread statistics used for variability
        # summary.
        df["Attribution_SD"] = df[cols].std(axis=1)
        df["Attribution_Mean"] = df[cols].mean(axis=1)

        # Focus caption on most variable rows shown in chart.
        df = (
            df.sort_values(
                "Attribution_SD",
                ascending=False,
            )
            .head(top_n)
        )

        most_variable = df.iloc[0]["Feature"]
        least_variable = (
            df.sort_values(
                "Attribution_SD",
                ascending=True,
            )
            .iloc[0]["Feature"]
        )

        avg_sd = df["Attribution_SD"].mean()

        # Translate average spread into concise interpretation text.
        if avg_sd > 0.08:
            variability_desc = (
                "substantial disagreement between method"
            )
        elif avg_sd > 0.05:
            variability_desc = (
                "moderate variation across methods"
            )
        else:
            variability_desc = (
                "relatively consistent attribution patterns"
            )

        # Render explanatory caption below chart.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 0.6rem;
                max-width: 620px;
                margin-left: auto;
                margin-right: auto;
            ">
                This chart shows variability in feature importance
                across methods, with error bars indicating spread.
                '{most_variable}' varies most, while '{least_variable}'
                is more consistent, indicating {variability_desc}.
                Greater variability reflects differences in how
                features are ranked across methods.
            </div>
            """,
            unsafe_allow_html=True
        )

    except (KeyError, IndexError, TypeError, ValueError) as e:
        # Keep caption failures local to this feaure-variability
        # block.
        st.warning(f"Clinical coverage analysis caption error: {e}")

def _render_instance_stability_caption(
        used_features: list[str],
        original_vals: np.ndarray,
        noise_vals: np.ndarray,
        mask_vals: np.ndarray,
        method_name: str,
        instance_id: int,
) -> None:
    """
    Render caption for patient-level explanation stability view.

    Compares how one instance's displayed feature attributions shift
    under two perturbation conditions shown in chart.
    """
    try:
        # Exit early when no feature labels are available for instance
        # view.
        if not used_features:
            return

        # Assemble instance-level attribution table used for local
        # comparison.
        df = pd.DataFrame({
            "Feature": [str(f) for f in used_features],
            "Original": np.asarray(original_vals, dtype=float),
            "Noise": np.asarray(noise_vals, dtype=float),
            "Masking": np.asarray(mask_vals, dtype=float),
        })

        if df.empty:
            return
        
        # Convert all displaed attribution series to absolute numeric
        # values.
        for col in ["Original", "Noise", "Masking"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").abs()

        # Remove invalid values before computing rank movement
        # summaries.
        df = df.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna(
            subset=["Original", "Noise", "Masking"]
        )
        if df.empty:
            return

        # Rank each condition so movement can be described in terms of
        # position shifts.
        for col in ["Original", "Noise", "Masking"]:
            df[f"{col}_Rank"] = df[col].rank(
                ascending=False,
                method="min",
            )

        # Measure how far each feature moves away from original
        # ranking.
        df["Noise_Shift"] = (
            df["Noise_Rank"] - df["Original_Rank"]
        ).abs()

        df["Mask_Shift"] = (
            df["Masking_Rank"] - df["Original_Rank"]
        ).abs()

        df["Max_Shift"] = df[
            [
                "Noise_Shift",
                "Mask_Shift",
            ]
        ].max(axis=1)

        most_shifted = str(
            df.sort_values(
                "Max_Shift",
                ascending=False,
            ).iloc[0]["Feature"]
        )
        most_stable = str(
            df.sort_values(
                "Max_Shift",
                ascending=True,
            ).iloc[0]["Feature"]
        )

        # Summarise whether noise or masking produces larger average
        # movement.
        avg_noise_shift = float(df["Noise_Shift"].mean())
        avg_mask_shift = float(df["Mask_Shift"].mean())

        if avg_noise_shift > avg_mask_shift:
            perturb_desc = (
                "Noise causes larger attribution changes than "
                "masking for this instance."
            )
        elif avg_mask_shift > avg_noise_shift:
            perturb_desc = (
                "Masking causes larger attribution changes "
                "than noise for this instance."
            )
        else:
            perturb_desc = (
                "Noise and masking cause similar attribution "
                "changes for this instance."
            )

        # Render one instance-specific explanation block below chart.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 0.6rem;
                max-width: 620px;
                margin-left: auto;
                margin-right: auto;
            ">
                This chart shows how feature attributions for 
                Instance {instance_id} change under noise and masking 
                using {_clean_method_name(method_name)}. {most_shifted}
                changes most, while {most_stable} remains more stable; 
                {perturb_desc} Smaller changes indicate more stable 
                local explanations under perturbation.
            </div>
            """,
            unsafe_allow_html=True
        )

    except (KeyError, IndexError, TypeError, ValueError):
        # Do not let caption issue block instance-level chart.
        pass

def _render_rank_shift_caption(
        used_features: list[str],
        original_vals: np.ndarray,
        noise_vals: np.ndarray,
        mask_vals: np.ndarray,
        method_name: str,
        instance_id: int,
        top_k: int = 10,
) -> None:
    """
    Render caption for feature rank-shift plot.

    Focuses on how original Top-K features reorder under
    two perturbation conditions for selected instance.
    """
    try:
        # Build instance table used to compare original, noise, and
        # masking ranks.
        df = pd.DataFrame({
            "Feature": used_features,
            "Original": np.abs(original_vals),
            "Noise": np.abs(noise_vals),
            "Masking": np.abs(mask_vals),
        })

        # Rank each condition so feature reordering can be measured
        # directly.
        for col in ["Original", "Noise", "Masking"]:
            df[f"{col}_Rank"] = df[col].rank(
                ascending=False,
                method="min"
            )

        # Limit summary to features that were originally in Top-K.
        top_orig = df.sort_values("Original_Rank").head(top_k).copy()

        # Measure how far each original Top-K feature moves under each
        # perturbation.
        top_orig["Noise_Shift"] = (
            top_orig["Noise_Rank"] - top_orig["Original_Rank"]
        ).abs()

        top_orig["Mask_Shift"] = (
            top_orig["Masking_Rank"] - top_orig["Original_Rank"]
        ).abs()

        top_orig["Max_Shift"] = top_orig[
            [
                "Noise_Shift",
                "Mask_Shift",
            ]
        ].max(axis=1)

        most_stable = (
            top_orig.sort_values(
                "Max_Shift",
                ascending=True,
            )
            .iloc[0]["Feature"]
        )

        most_shifted = (
            top_orig.sort_values(
                "Max_Shift",
                ascending=False,
            )
            .iloc[0]["Feature"]
        )

        # Compare average movement caused by each perturbation type.
        avg_noise_shift = top_orig["Noise_Shift"].mean()
        avg_mask_shift = top_orig["Mask_Shift"].mean()

        if avg_mask_shift > avg_noise_shift:
            perturb_desc = (
                "Masking causes larger rank changes than noise "
                "for this instance."
            )
        elif avg_noise_shift > avg_mask_shift:
            perturb_desc = (
                "Noise causes larger rank changes than masking "
                "for this instance."
            )
        else:
            perturb_desc = (
                "Noise and masking produce similar rank changes "
                "for this instance."
            )

        # Render rank-shift interpretation below chart.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 0.95rem;
                color: #2F3E4E;
                margin-top: 0.6rem;
                max-width: 75%;
                margin-left: auto;
                margin-right: auto;
            ">
                This chart tracks how Top-{top_k} features for 
                Instance {instance_id} change in rank under noise 
                and masking using {method_name}. '{most_shifted}' 
                changes most, while '{most_stable}' remains more 
                stable; {perturb_desc} Smaller rank shifts indicate 
                more stable local explanations under perturbation.
            </div>
            """,
            unsafe_allow_html=True
        )

    except (KeyError, IndexError, TypeError, ValueError) as e:
        # Surface local issue because caption already reports
        # detailed errors.
        st.warning(f"Instance stability caption error: {e}")


# --------------------------------------------------------------------
# Charts
# Build main comparison visuals used across dashboard.
# Each function focuses on one chart or chart group, then calls
# matching caption helper so visual and its interpretation remain
# paired in interface.
# --------------------------------------------------------------------
def _render_model_context(model_auc: float) -> None:
    """
    Render model context summary section.

    Appears early in dashboard because benchmark results should be 
    interpreted in context of underlying predictive performance.
    Includes concise metric card and short interpretation card.
    """
    # Translate numeric ROC-AUC value into short label and explanation
    # so user can quickly understand current model context.
    label, explanation = _auc_interpretation(model_auc)
    # Use card tone to signal whether current model performance is
    # more reassuring, moderate, or concerning for later explanation
    # analysis.
    kind = (
        "success"
        if model_auc >= 0.70
        else (
            "warning"
            if model_auc >= 0.50
            else "danger"
        )
    )

    # Use one column for headline metrics and one for short
    # interpretation.
    c1, c2 = st.columns([1, 2])

    with c1:
        # Show headline predictive-performance value.
        _info_card(
            "Model ROC-AUC",
            f"{model_auc:.3f}",
            "Test Subset",
            kind=kind,
        )
    with c2:
        # Use stronger warning tone when model performance is weak.
        tone = (
            "danger"
            if model_auc < 0.50
            else (
                "warning"
                if model_auc < 0.60
                else "neutral"
            )
        )
        
        # Present qualitative interpretation beside metric.
        _insight_card(
            "Model Signal Warning",
            f"{label}. {explanation}",
            tone=tone,
        )

def _render_best_method_cards(rel_df: pd.DataFrame) -> None:
    """
    Render best-method summary cards.

    Highlights top-performing method for main composite
    evaluation dimensions to provide users with concise comparison
    before they review detailed charts.
    """
    def _best(col: str) -> str:
        """
        Return best-performing method summary for one metric column.

        Returned string keeps display logic compact by combining
        winning method name and its score in one value.
        """
        # Stop early when requested metric is unavailable so card can
        # still render without breaking wider layout.
        if col not in rel_df.columns or rel_df[col].dropna().empty:
            return "N/A"
        
        # Sort descending because higher composite scores indicate
        # stronger performance.
        row = rel_df.sort_values(col, ascending=False).iloc[0]
        return f"{row['Method']} ({row[col]:.3f})"
    
    # Build four high-level summary cards shown in this section.
    cards = [
        ("Best Reliability", _best("Reliability_Score")),
        ("Best Robustness", _best("Robustness_Score")),
        ("Best Fidelity", _best("Fidelity_Score")),
        ("Best Clinical Alignment", _best("Clinical_Score")),
    ]

    # Use two-by-two card layout so summary remains compact.
    row1 = st.columns(2, gap="medium")
    for col, (title, value) in zip(row1, cards[:2]):
        with col:
            # Split combined "Method (Score)" string so card can show
            # method as main value and score as supporting text.
            _info_card(
                title, 
                value.split(" (")[0],
                (
                    value[value.find("("):]
                    .replace("(", "")
                    .replace(")", "")
                    if "(" in value
                    else ""
                ),
            )

    # Add small spacer between first and second card so section
    # does not feel compressed.
    st.markdown(
        "<div style='height: 4.5rem;'></div>",
        unsafe_allow_html=True,
    )

    row2 = st.columns(2, gap="medium")
    for col, (title, value) in zip(row2, cards[2:]):
        with col:
            # Apply same split logic to second row for consistent
            # card styling.
            _info_card(
                title, 
                value.split(" (")[0], 
                (
                    value[value.find("("):]
                    .replace("(", "")
                    .replace(")", "")
                    if "(" in value
                    else ""
                ),
            )

    # Add corresponding caption below cards.
    _render_best_methods_caption()

def _render_radar_chart(rel_df: pd.DataFrame) -> None:
    """
    Render radar chart that compares method performance across main
    benchmark dimensions.

    Combines robustness, fidelity, clinical alignment, and overall
    reliability into one visual so users can compare general shape
    of each method's performance profile.
    """
    # Define four dimensions plotted on radar chart and shorter
    # display labels shown around chart.
    metrics = [
        "Robustness_Score",
        "Fidelity_Score",
        "Clinical_Score",
        "Reliability_Score",
    ]
    labels = ["Robustness", "Fidelity", "Clinical", "Overall"]

    # Fix axis positions so category order remains stable across runs.
    theta_base = np.array([np.pi / 2, np.pi, 3 * np.pi / 2, 0.0])

    # Build polar figure used for radar-style comparison.
    fig = plt.figure(figsize=(4.2, 4.4))
    ax = fig.add_subplot(111, polar=True)

    # Offset each method slightly so markers and lines remain easier
    # to separate when values are similar.
    offset_map = {
        "Captum IG": -0.055,
        "SHAP": 0.000,
        "LIME": 0.055,
    }

    # Keep method styling consistent with rest of interface.
    style_map = {
        "Captum IG": {
            "color": "#3A6EA5", 
            "marker": "o", 
            "linestyle": "-", 
            "linewidth": 3.0, 
            "markersize": 7, 
            "zorder": 4, 
            "alpha": 0.98,
        },
        "SHAP": {
            "color": "#E6862A",
            "marker": "s", 
            "linestyle": (0, (1, 1)), 
            "linewidth": 3.0, 
            "markersize": 7, 
            "zorder": 5, 
            "alpha": 0.98,
        },
        "LIME": {
            "color": "#4C9F70",
            "marker": "^", 
            "linestyle": (0, (6, 2)), 
            "linewidth": 3.2, 
            "markersize": 8, 
            "zorder": 6, 
            "alpha": 0.98,
        },
    }

    # Plot one closed shape per method so user can compare performance
    # profiles across all four dimensions at once.
    for _, row in rel_df.iterrows():
        method = str(row["Method"]) 

        # Convert selected metrics into a numeric sequence for
        # plotting.
        vals = [
            float(row[m]) if pd.notna(row[m]) else 0.0 
            for m in metrics
        ]
        vals_closed = vals + [vals[0]]

        # Fall back to a default style if an unexpected method label
        # appears.
        style = style_map.get(
            method,
            {
                "color": "#4C78A8",
                "marker": "o", 
                "linestyle": "-", 
                "linewidth": 2.8, 
                "markersize": 7, 
                "zorder": 4, 
                "alpha": 0.98,
            },
        )

        # Apply method-specific angular offset so overlapping lines
        # stay easier to read.
        theta_plot = theta_base + offset_map.get(method, 0.0)
        theta_plot_closed = np.concatenate(
            [
                theta_plot,
                [theta_plot[0]],
            ]
        )

        # Draw method profile and add a light white stroke below line
        # to imporve seperation where traces are close together.
        ax.plot(
            theta_plot_closed, 
            vals_closed,
            label=method,
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=style["markersize"],
            alpha=style["alpha"],
            zorder=style["zorder"],
            solid_capstyle="round",
            solid_joinstyle="round",
            path_effects=[
                pe.Stroke(
                    linewidth=style["linewidth"] + 1.5,
                    foreground="white",
                ),
                pe.Normal(),
            ],
        )

    # Show four benchmark dimensions at fixed axis locations.
    ax.set_xticks(theta_base)
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="x", pad=6)

    # Adjust label alignment by position for optimal text spacing
    # around chart.
    for label, angle in zip(ax.get_xticklabels(), theta_base):
        angle_deg = np.degrees(angle)


        if np.isclose(angle_deg, 90): 
            label.set_verticalalignment("bottom")
        elif np.isclose(angle_deg, 270):
            label.set_verticalalignment("top")
        elif np.isclose(angle_deg, 0):
            label.set_horizontalalignment("left")
        elif np.isclose(angle_deg, 180):
            label.set_horizontalalignment("right")

    # Keep radial scale bounded between 0 and 1 because these are
    # normalised composite-style scores.
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["0.2", "0.4", "0.6", "0.8", "1.0"], 
        fontsize=10,
        fontweight="bold",
        color="#333333",
    )
    ax.set_rlabel_position(20)

    # Use light grid so chart remains readable without distracting
    # from plotted method profiles.
    ax.grid(alpha=0.22, linewidth=0.9)
    ax.spines["polar"].set_linewidth(1.0)

    # Place legend beneath figure so it does not crowd plotting area.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False, 
        fontsize=9,
        handletextpad=0.6,
        columnspacing=1.6,
    )

    # Tighten figure margins before rendering it in Streamlit.
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.10, right=0.90)
    
    # Display completed Matplotlib figure in dashboard.
    st.pyplot(fig, use_container_width=False)

    # Add corresponding explantory caption below chart.
    _render_tradeoff_caption(rel_df)

def _render_reliability_bars(rel_df: pd.DataFrame) -> None:
    """
    Render bar chart that compares overall reliability across methods.

    Isolates final composite reliability score so user can 
    compare overall ranking without extra detail shown in radar chart.
    """
    # Build bar-chart figure used for single-score comparison.
    fig, ax = plt.subplots(figsize=(5.6, 4.8))

    # Use DatFrame row order as plotting order for this view.
    x = np.arange(len(rel_df))
    vals = rel_df["Reliability_Score"].astype(float).values

    # Keep method colours aligned with wider dashboard palette.
    colors = ["#3A6EA5", "#E6862A", "#4C9F70"]

    # Daw one bar per method.
    ax.bar(x, vals, color=colors)

    # Configure axis labels and bounds for a 0-1 composite score
    # display.
    ax.set_xticks(x)
    ax.set_xticklabels(rel_df["Method"], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=10, fontweight="bold")

    # Keep chart styling simple and consistent with rest of interface.
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Print each score above its bar so exact values remain visible
    # without hovering.
    for i, v in enumerate(vals):
        ax.text(
            i,
            v + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    
    # Final layout adjustments before showing chart.
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.18, left=0.12, right=0.96)
    
    # Render chart in dashboard.
    st.pyplot(fig, use_container_width=True)

    # Render corresponding caption below chart.
    _render_reliability_caption(rel_df)

def _render_reliability_decomposition(rel_df: pd.DataFrame) -> None:
    """
    Render stacked bar chart that shows how reliability-related
    dimensions contribute to total non-normalised score mix.

    Complements composite reliability score by showing balance
    between robustnes, fidelity, and clinical alignment for each
    method.
    """
    # Stop early when there is no data to decompose for this section.
    if rel_df.empty:
        st.info("Reliability decomposition not available.")

    # Build figure used for stacked comparison.
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    x = np.arange(len(rel_df))

    # Extract three component scores that are combined in this view.
    robustness_raw = rel_df["Robustness_Score"].astype(float).values
    fidelity_raw = rel_df["Fidelity_Score"].astype(float).values
    clinical_raw = rel_df["Clinical_Score"].astype(float).values

    # Convert raw component values into proportional contributions
    # so each bar sums to 100 percent.
    totals = robustness_raw + fidelity_raw + clinical_raw
    robustness = robustness_raw / totals
    fidelity = fidelity_raw / totals
    clinical = clinical_raw / totals

    # Use distinct colours for each component so stack composition
    # remains easy to read.
    robustness_color = "#7A8FA6"
    fidelity_color = "#C05A8D"
    clinical_color = "#6B5FB5"

    # Draw stacked bars from bottom to top in same component order
    # used in legend.
    ax.bar(x, robustness, label="Robustness", color=robustness_color)
    ax.bar(
        x,
        fidelity,
        bottom=robustness,
        label="Fidelity",
        color=fidelity_color,
    )
    ax.bar(
        x,
        clinical,
        bottom=robustness + fidelity,
        label="Clinical",
        color=clinical_color,
    )

    # Label methods on x-axis and show y-axis as a proportional
    # contribution scale.
    ax.set_xticks(x)
    ax.set_xticklabels(rel_df["Method"], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(
        "Contribution to Score (%)",
        fontsize=10,
        fontweight="bold",
    )

    # Use same light grid and clean frame style as other bar charts.
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add-in percentage labels when a segment is large enough to keep
    # text readable.
    for i in range(len(rel_df)):
        if robustness[i] > 0.05:
            ax.text(
                i,
                robustness[i] / 2,
                f"{robustness[i]*100:.0f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )

        if fidelity[i] > 0.05:
            ax.text(
                i,
                robustness[i] + fidelity[i] / 2,
                f"{fidelity[i]*100:.0f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )

        if clinical[i] > 0.05:
            ax.text(
                i,
                robustness[i] + fidelity[i] + clinical[i] / 2,
                f"{clinical[i]*100:.0f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )

    # Format y-axis as percentages to match normalised stacked layout.
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Keep legend below chart so bars retain more vertical space.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    # Final spacing adjustments before rendering.
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.18, left=0.12, right=0.96)
    
    # Render chart.
    st.pyplot(fig, use_container_width=True)

    # Render corresponding caption.
    _render_reliability_decomposition_caption(rel_df)

def _render_method_performance_summary(
    method_df: pd.DataFrame,
) -> None:
    """
    Render side-by-side robustness and fidelity comparison charts.

    Gives user direct visual comparison of main method-level
    performance outputs by placing stability and fidelity next to
    each other in one shared view.
    """
    # Define expected columns for two method-level comparisons shown
    # here.
    stability_cols = [
        "Stability_Spearman_Noise_Mean",
        "Stability_Spearman_Masking_Mean",
    ]
    fidelity_cols = [
        "Fidelity_vs_Permutation_Spearman_Mean",
        "Fidelity_vs_GradInput_Spearman_Mean",
    ]
    
    # Retain only columns that are actually present so section can
    # validate availability before plotting.
    stability_cols = [
        c
        for c in stability_cols
        if c in method_df.columns
    ]
    fidelity_cols = [
        c
        for c in fidelity_cols
        if c in method_df.columns
    ]
    
    # Stop early when required schema is incomplete.
    if (
        len(stability_cols) != 2
        or len(fidelity_cols) != 2
        or "Method" not in method_df.columns
    ):
        st.info("Method performance summary data is not available.")
        return

    # Work on a copy so display formatting does not modify original
    # results table.
    plot_df = method_df.copy()

    # Standardise method labels so they match rest of dashboard.
    plot_df["Method"] = plot_df["Method"].map(_clean_method_name)

    # Shared x positions and bar width used by both charts.
    methods = plot_df["Method"].values
    x = np.arange(len(methods))
    width = 0.35

    # Split section into two visual panels with a divider in middle.
    left_col, divider_col, right_col = st.columns([1, 0.02, 1])

    with divider_col:
        # Render divider once between paired charts.
        _render_column_divider()

    with left_col:
        # Introduce left-side chart as robustness/stability view.
        _subsection_header("Explanation Stability")

        # Build grouped bar chart comparing noise and masking
        # stability.
        fig1, ax1 = plt.subplots(figsize=(5.4, 4.6))
        ax1.bar(
            x - width /2,
            pd.to_numeric(
                plot_df[stability_cols[0]],
                errors="coerce",
            ),
            width,
            label="Noise",
            color="#5F748C",
        )
        ax1.bar(
            x + width / 2,
            pd.to_numeric(
                plot_df[stability_cols[1]],
                errors="coerce",
            ),
            width,
            label="Masking",
            color="#A6B6C7",
        )

        # Configure axis labels and limits for Spearman-based
        # robustness scores.
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, fontsize=10)
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel(
            "Spearman Correlation (Original vs Perturbed)",
            fontsize=10,
            fontweight="bold",
        )
        
        # Apply shared dashboard bar chart styling.
        ax1.grid(axis="y", alpha=0.25)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False, 
            fontsize=9,
        )

        # Tighten spacing before rendering.
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.subplots_adjust(
            top=0.88,
            bottom=0.18,
            left=0.12,
            right=0.96,
        )
        
        # Render chart.
        st.pyplot(fig1)

        # Render corresponding caption.
        _render_stability_caption(method_df)

    with right_col:
        # Introduce right-side chart as fidelity comparison view.
        _subsection_header("Explanation Fidelity")

        # Build grouped bar chart comparing two fidelity reference
        # baselines.
        fig2, ax2 = plt.subplots(figsize=(5.4, 4.6))
        ax2.bar(
            x - width /2,
            pd.to_numeric(plot_df[fidelity_cols[0]], errors="coerce"),
            width,
            label="Permutation",
            color="#9E476F",
        )
        ax2.bar(
            x + width / 2,
            pd.to_numeric(plot_df[fidelity_cols[1]], errors="coerce"),
            width,
            label="Gradient x Input",
            color="#D98AAE",
        )

        # Configure axis lables and bounds for fidelity comparison.
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, fontsize=10)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Score", fontsize=10, fontweight="bold")
        
        # Reuse same styling as left-side comparison chart.
        ax2.grid(axis="y", alpha=0.25)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False, 
            fontsize=9,
        )

        # Final layout adjustments before rendering. 
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.subplots_adjust(
            top=0.88,
            bottom=0.18,
            left=0.12,
            right=0.96,
        )
        
        # Render chart.
        st.pyplot(fig2)

        # Render corresponding caption.
        _render_fidelity_caption(method_df)

def _render_topk_stability_overlap(
    method_df: pd.DataFrame,
    top_k: int,
) -> None:
    """
    Render bar chart that shows Top-K stability overlap across
    methods.

    Compares how consistently each method preserves its top-ranked
    features under noise and masking perturbations.
    """
    # Define possible column names to handle variations in engine
    # output formats.
    noise_candidates = [
        f"Stability_Top{top_k}_Noise_Overlap_Mean",
        f"Stability_Top{top_k}_Noise_Overlap",
        f"Top{top_k}_Stability_Noise_Overlap",
        "Top10_Stability_Noise_Overlap",
    ]
    mask_candidates = [
        f"Stability_Top{top_k}_Masking_Overlap_Mean",
        f"Stability_Top{top_k}_Masking_Overlap",
        f"Top{top_k}_Stability_Masking_Overlap",
        "Top10_Stability_Masking_Overlap",
    ]

    # Resolve first available column for each perturbation type.
    noise_col = _first_existing(method_df, noise_candidates)
    mask_col = _first_existing(method_df, mask_candidates)

    # Exit early if required metrics are missing.
    if not noise_col or not mask_col:
        st.info(
            "Top-K stability overlap data not available in "
            "the current engine output."
        )
        return

    # Prepare data used for plotting.
    df = method_df[["Method", noise_col, mask_col]].copy()
    df["Noise"] = pd.to_numeric(df[noise_col], errors="coerce")
    df["Masking"] = pd.to_numeric(df[mask_col], errors="coerce")

    # Create grouped bar chart for noise versus masking overlap.
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    x = np.arange(len(df))
    width = 0.35

    ax.bar(
        x - width / 2,
        df["Noise"],
        width=width,
        label="Noise",
        color="#5F748C",
    )
    ax.bar(
        x + width / 2,
        df["Masking"],
        width=width,
        label="Masking",
        color="#A6B6C7",
    )

    # Configure axes and labels.
    ax.set_xticks(x)
    ax.set_xticklabels(df["Method"].map(_clean_method_name))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"Top-{top_k} Overlap", fontweight="bold")

    # Apply consistent styling across dashboard charts.
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Position legend below chart to maximise vertical space.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    # Adjust layout before rendering.
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.subplots_adjust(top=0.88, bottom=0.18, left=0.12, right=0.96)
    
    # Render chart.
    st.pyplot(fig, use_container_width=False)

    # Render corresponding caption.
    _render_topk_stability_overlap_caption(method_df, top_k)

def _render_fidelity_vs_robustness_tradeoff(
    rel_df: pd.DataFrame,
) -> None:
    """
    Render plot showing trade-off between robustness and fidelity.

    Each method in plot is positioned based on its stability under
    perturbation and its alignment with model behaviour.
    """
    # Ensure required columns are available before plotting.
    needed = {"Method", "Robustness_Score", "Fidelity_Score"}
    if rel_df.empty or not needed.issubset(rel_df.columns):
        st.info(
            "Fidelity vs robustness trade-off data not available."
        )
        return

    # Create plotting figure.
    fig, ax = plt.subplots(figsize=(6.2, 6.8))

    # Work on a copy to avoid modifying original dataset.
    plot_df = rel_df.copy()
    plot_df["Robustness_Score"] = pd.to_numeric(
        plot_df["Robustness_Score"],
        errors="coerce",
    )
    plot_df["Fidelity_Score"] = pd.to_numeric(
        plot_df["Fidelity_Score"],
        errors="coerce",
    )

    color_map = METHOD_COLOR_MAP

    # Define marker styles to visually distinguish methods.
    marker_map = {
        "Captum IG": "o",
        "SHAP": "s",
        "LIME": "^",
    }

    # Offset labels slightly so they do not overlap with markers.
    label_offsets = {
        "Captum IG": (0.018, 0.020),
        "SHAP": (0.018, -0.025),
        "LIME": (0.018, 0.018),
    }

    # Plot each method as a point in robustness-fidelity space.
    for _, row in plot_df.iterrows():
        method = _clean_method_name(row["Method"])
        x = row["Robustness_Score"]
        y = row["Fidelity_Score"]

        # Skip rows with missing values.
        if pd.isna(x) or pd.isna(y):
            continue

        ax.scatter(
            x,
            y,
            s=220,
            color=color_map.get(method, "#4C78A8"),
            marker=marker_map.get(method, "o"),
            edgecolor="white",
            linewidths=1.8,
            zorder=3,
        )

        # Position each label relative to each marker.
        dx, dy = label_offsets.get(method, (0.012, 0.012))

        ax.text(
            x + dx,
            y + dy,
            method,
            fontsize=11,
            fontweight="bold",
            color="#1F2A44",
            ha="left",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="none",
                alpha=0.88
            ),
            path_effects=[
                pe.withStroke(
                    linewidth=2.5,
                    foreground="white",
                )
            ],
            zorder=4
        )

    # Configure axes using normalised score ranges.
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(
        "Robustness",
        fontsize=12,
        fontweight="bold",
        labelpad=12,
    )
    ax.set_ylabel(
        "Fidelity",
        fontsize=12,
        fontweight="bold",
        labelpad=12,
    )
    ax.tick_params(axis="both", labelsize=10)

    # Apply conistent visual styling.
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add quadrant guide lines.
    ax.axvline(
        0.50,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )
    ax.axhline(
        0.50,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )

    # Add light shading to improve visual grouping of performance
    # regions.
    ax.axvspan(0.5, 1, ymin=0.5, ymax=1, alpha=0.08)
    ax.axvspan(0, 0.5, ymin=0.5, ymax=1, alpha=0.04)
    ax.axvspan(0.5, 1, ymin=0, ymax=0.5, alpha=0.04)
    ax.axvspan(0, 0.5, ymin=0, ymax=0.5, alpha=0.02)

    # Add quadrant annotations for interpretation.
    ax.text(
        0.75,
        0.92,
        "Best Performance\n(High Fidelity & Robustness)",
        ha="center",
        fontsize=9,
        alpha=0.7,
    )
    ax.text(
        0.25,
        0.92,
        "High Fidelity\nLow Robustness",
        ha="center",
        fontsize=8,
        alpha=0.6,
    )
    ax.text(
        0.75,
        0.08,
        "High Robustness\nLow Fidelity",
        ha="center",
        fontsize=8,
        alpha=0.6,
    )
    ax.text(
        0.25,
        0.08,
        "Weakest Performance\n(Low Fidelity & Robustness)",
        ha="center",
        fontsize=8,
        alpha=0.5,
    )

    # Add final layout adjustments.
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.18, left=0.12, right=0.96)
    
    # Render chart.
    st.pyplot(fig, use_container_width=True)

    # Render corresponding caption.
    _render_fidelity_robustness_tradeoff_caption(rel_df)

def _render_clinical_summary_cards(
    method_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    top_k: int,
) -> None:
    """
    Render cards that show clinical alignment results for each method.

    Summarises overlap, weighted overlap, precision, and recall,
    and also lists which clincal features were captured or missed.
    """
    # Resolve metric columns needed for summary cards.
    overlap_col = _first_existing(
        method_df, 
        [f"Top{top_k}_Clinical_Overlap", "Top10_Clinical_Overlap"]
    )
    weighted_col = _first_existing(
        method_df,
        ["Weighted_Clinical_Overlap"],
    )
    recall_col = _first_existing(method_df, ["Clinical_Recall"])
    precision_col = _first_existing(method_df, ["Clinical_Precision"])

    # Stop early if main overlap metric is unavailable.
    if overlap_col is None:
        st.info("Clinical summary metrics not available.")
        return
    
    color_map = METHOD_COLOR_MAP

    def _adjust_color(hex_color: str, factor: float = 0.75) -> str:
        """
        Lighten base colour for card backgrounds and borders.
        """
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return f"#{r:02X}{g:02X}{b:02X}"
    
    # Create one card column per explanation method.
    cols = st.columns(len(method_df))
    for ui_col, (_, row) in zip(cols, method_df.iterrows()):
        with ui_col:
            # Build card colour scheme from method's display colour.
            method = _clean_method_name(row["Method"])
            base_color = color_map.get(method, "#4F81BD")
            card_bg = _adjust_color(base_color, 0.75)
            border_color = _adjust_color(base_color, 0.55)

            # Read main clinical-alignment metrics for this method.
            overlap = pd.to_numeric(row.get(overlap_col), errors="coerce")
            weighted = pd.to_numeric(
                row.get(weighted_col), errors="coerce"
            ) if weighted_col else np.nan
            recall = (
                pd.to_numeric(row.get(recall_col), errors="coerce")
                if recall_col
                else np.nan
            )
            precision = pd.to_numeric(
                row.get(precision_col), errors="coerce"
            ) if precision_col else np.nan

            # Retrieve captured and missing clinical feature lists.
            overlap_features = _get_overlap_feature_text(
                row,
                feature_df,
                top_k,
            )
            missing_features = _get_missing_clinical_feature_list(
                row,
                feature_df,
                top_k,
            )

            # Convert captured features into styled chips.
            feature_boxes_html = (
                "".join(
                    f'<div class="feature-chip">'
                    f"{feat}</div>"
                    for feat in overlap_features
                )
                if overlap_features
                else (
                    '<div class="feature-chip feature-chip-empty">'
                    "None</div>"
                )
            )

            # Convert missing features into styled chips.
            missing_boxes_html = (
                "".join(
                    f'<div class="feature-chip feature-chip-missing">'
                    f"{feat}</div>"
                    for feat in missing_features
                )
                if missing_features
                else (
                    '<div class="feature-chip feature-chip-empty">'
                    "None</div>"
                )
            )

            # Assemble full HTML card for this method.
            card_html = (
                f'<div class="panel-card" '
                f'style="background: {card_bg}; '
                f'border: 1.5px solid {border_color};">'
                f'<div class="panel-title" '
                f'style="color: {base_color}; font-weight: 700;">'
                f'{method}</div>'
                f'<div class="mini-grid">'
                f'<div class="mini-box">'
                f'<div class="mini-label">Top-{top_k} Overlap</div>'
                f'<div class="mini-value" style="color: {base_color};">'
                f'{_format_metric(overlap, 0)}</div>'
                f'</div>'
                f'<div class="mini-box">'
                f'<div class="mini-label">Weighted Overlap</div>'
                f'<div class="mini-value" style="color: {base_color};">'
                f'{_format_metric(weighted)}</div>'
                f'</div>'
                f'<div class="mini-box">'
                f'<div class="mini-label">Precision</div>'
                f'<div class="mini-value" style="color: {base_color};">'
                f'{_format_metric(precision)}</div>'
                f'</div>'
                f'<div class="mini-box">'
                f'<div class="mini-label">Recall</div>'
                f'<div class="mini-value" style="color: {base_color};">'
                f'{_format_metric(recall)}</div>'
                f'</div>'
                f'</div>'
                f'<div class="feature-list-card">'
                f'<div class="feature-list-title">'
                f'Captured Clinical Features in Top-{top_k}'
                f'</div>'
                f'<div class="feature-chip-grid">'
                f'{feature_boxes_html}'
                f'</div>'
                f'</div>'
                f'<div class="feature-list-card feature-list-card-missing">'
                f'<div class="feature-list-title">'
                f'Missing Clinical Features (Not in Top-{top_k})'
                f'</div>'
                f'<div class="feature-chip-grid">'
                f'{missing_boxes_html}'
                f'</div>'
                f'</div>'
                f'</div>'
            )
            
            # Render completed card.
            st.markdown(card_html, unsafe_allow_html=True)

    # Render corresponding caption below card group.
    _render_clinical_alignment_summary_caption(
        method_df,
        top_k,
    )

def _render_clinical_coverage_analysis(
    method_df: pd.DataFrame,
    top_k: int,
) -> None:
    """
    Render stacked bar chart showing captured versus missed
    Top-K features.

    Compares how many clinically validated features each method
    keeps within its Top-K ranked variables.
    """
    # Find overlap column used to build captured versus missed counts.
    overlap_col = _first_existing(
        method_df, 
        [f"Top{top_k}_Clinical_Overlap", "Top10_Clinical_Overlap"]
    )
    if not overlap_col:
        st.info("Clinical coverage analysis not available.")
        return

    # Prepare captured and missed counts for each method.
    df = method_df[["Method", overlap_col]].copy()
    df["Captured"] = pd.to_numeric(df[overlap_col], errors="coerce")
    df["Missed"] = float(top_k) - df["Captured"]
    df["Missed"] = df["Missed"].clip(lower=0)

    # Build stacked bar chart.
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    x = np.arange(len(df))
    captured_color = "#6C63AC"
    missed_color = "#DCD6F4"

    # Plot captured counts first, then missed counts above them.
    ax.bar(
        x,
        df["Captured"],
        label="Captured",
        color=captured_color,
        width=0.62,
    )
    ax.bar(
        x,
        df["Missed"],
        bottom=df["Captured"],
        label="Missed",
        color=missed_color,
        width=0.62,
    )

    # Configure axis labels and Top-K count range.
    ax.set_xticks(x)
    ax.set_xticklabels(
        df["Method"].map(_clean_method_name),
        fontsize=9,
    )
    ax.set_ylim(0, top_k)
    ax.set_ylabel(
        f"Count within Top-{top_k}",
        fontsize=10,
        fontweight="bold",
    )

    # Apply standard dashboard bar-chart styling.
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Place legend below plot.
    ax.legend(
        frameon=False, 
        fontsize=9, 
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        borderaxespad=0.0,
        columnspacing=1.8,
        handletextpad=0.7,
    )

    # Add numeric labels inside each stacked bar segment.
    for i, row in df.iterrows():
        captured = row["Captured"]
        missed = row["Missed"]

        if pd.notna(captured) and captured > 0:
            ax.text(
                i, 
                captured / 2, 
                f"{int(captured)}", 
                ha="center", 
                va="center", 
                fontsize=9,
                fontweight="bold",
                color="white"
            )

        if pd.notna(missed) and missed > 0:    
            ax.text(
                i, 
                row["Captured"] + row["Missed"] / 2, 
                f"{int(row['Missed'])}", 
                ha="center", 
                va="center", 
                fontsize=9,
                fontweight="bold",
                color="#1F2A33"
            )

    # Apply final layout adjustments before rendering.
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.90)
    
    # Render chart.
    st.pyplot(fig, use_container_width=False)

    # Render corresponding caption.
    _render_clinical_coverage_analysis_caption(method_df, top_k)

def _render_clinical_feature_consistency_map(
    feature_df: pd.DataFrame,
    top_n: int = 12,
) -> None:
    """
    Render heatmap of clinically validated feature consistency
    across methods.

    Compares how strongly each method weights clinically
    validated featured retained for display.
    """
    df = feature_df.copy()
    needed = {"Feature", "Captum_IG", "SHAP", "LIME"}

    # Stop early if required method columns are missing.
    if not needed.issubset(df.columns):
        st.info("Clinical feature consistency map not available.")
        return

    # Restrict heatmap to clinically validated features when
    # available.
    if "Clinically_Validated" in df.columns:
        df = df[
            df["Clinically_Validated"]
            .fillna(0)
            .astype(int)
            == 1
        ].copy()

    # Stop early if no validated features remain after filtering.
    if df.empty:
        st.info(
            "No clinically validated features were flagged in "
            "the current feature output."
        )
        return

    method_cols = ["Captum_IG", "SHAP", "LIME"]

    # Convert method attributions to numeric absolute values.
    for c in method_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").abs()

    # Normalise each method column before comparing them side by side.
    df = _normalize_method_columns(df, method_cols)
    df["MeanImportance"] = df[method_cols].mean(axis=1)
    df = df.sort_values("MeanImportance", ascending=False).head(top_n)

    # Build heatmap matrix from selected features.
    mat = df[method_cols].values
    fig, ax = plt.subplots(figsize=(5.4, 6.0))

    # Use one purple gradient to keep view visually distinct.
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "clinical_purple_scale",
        ["#F7F5FF", "#B1A7E0", "#5A4FB2"]
    )

    im = ax.imshow(
        mat, 
        aspect="auto", 
        cmap=cmap,
        vmin=0,
        vmax=np.nanmax(mat),
    )

    # Label methods across columns and features down the rows.
    ax.set_xticks(np.arange(len(method_cols)))
    ax.set_xticklabels(["Captum IG", "SHAP", "LIME"], fontsize=10)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["Feature"].tolist(), fontsize=10)

    # Use median visible value to choose readable text colour.
    valid_vals = mat[np.isfinite(mat)]
    threshold = np.nanmedian(valid_vals) if valid_vals.size else 0.0
    
    # Write normalised value inside each heatmap cell.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if pd.notna(val):
                txt_color = "white" if val >= threshold else "#111111"
                ax.text(
                    j, 
                    i, 
                    f"{val:.2f}", 
                    ha="center", 
                    va="center", 
                    fontsize=8, 
                    color=txt_color, 
                    fontweight="bold"
                )
    
    # Add horiztonal colour bar below heatmap.
    cbar = plt.colorbar(
        im, 
        ax=ax,
        orientation="horizontal",
        fraction=0.035, 
        pad=0.10,
        aspect=40,
    )
    cbar.set_label(
        "Normalised Mean Absolute Attribution",
        fontsize=10,
        labelpad=4,
    )
    cbar.ax.tick_params(labelsize=9, pad=2)

    # Apply final layout adjustments before rendering.
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.24, top=0.92)
    
    # Render heatmap.
    st.pyplot(fig, use_container_width=False)

    # Render corresponding caption.
    _render_clinical_feature_consistency_caption(feature_df)

def _render_clinical_recall_curve(
    feature_df: pd.DataFrame,
    max_k: int = 15,
) -> None:
    """
    Render clinical recall curve across multiple Top-K values.

    Shows how quickly each method accumulates clinically validated
    features as more top-ranked variables are included.
    """
    df = feature_df.copy()
    needed = {
        "Feature",
        "Captum_IG",
        "SHAP",
        "LIME",
        "Clinically_Validated",
    }
    
    # Stop early if required feature and validation columns are
    # missing.
    if not needed.issubset(df.columns):
        st.info(
            "Clinical recall curve requires feature importances "
            "and a Clinically_Validated flag."
        )
        return
   
    # Convert clinical validation flag into clean numeric indicator.
    df["Clinically_Validated"] = pd.to_numeric(
        df["Clinically_Validated"], errors="coerce"
    ).fillna(0).astype(int)

    # Stop early if no validated features are present in current
    # output.
    total_clinical = int(df["Clinically_Validated"].sum())
    if total_clinical == 0:
        st.info(
            "No clinically validated features are flagged in "
            "the current output."
        )
        return
    
    # Map raw output column names to display names used in interface.
    method_map = {
        "Captum_IG": "Captum IG",
        "SHAP": "SHAP",
        "LIME": "LIME",
    }

    color_map = METHOD_COLOR_MAP

    # Keep marker shapes consistent with wider dashboard.
    marker_map = {
        "Captum IG": "o",
        "SHAP": "s",
        "LIME": "^",
    }

    # Build chart used for recall-depth comparison.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    # Limit Top-K to available number of rows.
    max_k = min(max_k, len(df))
    k_values = list(range(1, max_k + 1))

    # Plot one recall curve per explanation method.
    for raw_method, display_method in method_map.items():
        method_df = df[
            [
                "Feature",
                "Clinically_Validated",
                raw_method,
            ]
        ].copy()
        method_df[raw_method] = pd.to_numeric(
            method_df[raw_method],
            errors="coerce",
        ).abs()
        method_df = method_df.dropna(subset=[raw_method])

        # Skip methods with no usable values in current output.
        if method_df.empty:
            continue

        # Rank features from most important to least important for
        # this method.
        method_df = method_df.sort_values(
            raw_method,
            ascending=False,
        ).reset_index(drop=True)

        recalls = []
        for k in k_values:
            # Measure recall after including current Top-K feature
            # set.
            topk = method_df.head(k)
            hits = int(topk["Clinically_Validated"].sum())
            recall_at_k = hits / float(total_clinical)
            recalls.append(recall_at_k)

        # Use a distinct line style per method for readability.
        linestyle_map = {
            "Captum IG": "-",
            "SHAP": "--",
            "LIME": "-.",
        }

        ax.plot(
            k_values,
            recalls,
            label=display_method,
            color=color_map.get(display_method, "#4C78A8"),
            marker=marker_map.get(display_method, "o"),
            linestyle=linestyle_map.get(display_method, "-"),
            linewidth=2.6,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )

    # Configure axes for Top-K recall range.
    ax.set_xlim(1, max_k)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(list(range(2, max_k + 1, 2)))
    ax.set_xlabel(
        "Top-K Features",
        fontsize=13,
        fontweight="bold",
        labelpad=6,
    )
    ax.set_ylabel(
        "Clinical Recall",
        fontsize=13,
        fontweight="bold",
        labelpad=6,
    )
    ax.tick_params(axis="both", labelsize=11)
    
    # Apply line-chart styling.
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=11,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    # Apply final spacing adjustments before rendering.
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.88,
        bottom=0.18,
        left=0.10,
        right=0.99,
    )
    
    # Center chart within page layout.
    _, col2, _ = st.columns([1,2.4,1])
    with col2:
        # Render chart.
        st.pyplot(fig, use_container_width=True)
        # Render corresponding caption.
        _render_clinical_recall_curve_caption(feature_df, max_k)

def _render_agreement_heatmap(agreement_df: pd.DataFrame) -> None:
    """
    Render heatmap showing global agreement between explanation
    methods.

    Compares rank corelation of global feature importance
    patterns across the three explanation methods.
    """
    # Retreive single agreement row used by this section.
    agr = _get_agreement_row(agreement_df)
    if agr is None:
        st.info("Agreement data not available.")
        return

    # Build symmetric agremeent matrix for heatmap.
    mat = pd.DataFrame(
        [
            [
                1.0,
                agr.get("SHAP_vs_Captum_Spearman", np.nan),
                agr.get("LIME_vs_Captum_Spearman", np.nan),
            ],
            [
                agr.get("SHAP_vs_Captum_Spearman", np.nan),
                1.0,
                agr.get("SHAP_vs_LIME_Spearman", np.nan),
            ],
            [
                agr.get("LIME_vs_Captum_Spearman", np.nan),
                agr.get("SHAP_vs_LIME_Spearman", np.nan),
                1.0,
            ],
        ],
        index=["Captum IG", "SHAP", "LIME"],
        columns=["Captum IG", "SHAP", "LIME"],
    ).astype(float)

    # Create heatmap.
    fig, ax = plt.subplots(figsize=(5.4, 5.7))

    # Use diverging scale so negative and positive agreement are
    # distinct.
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "agreement_diverging",
        ["#C65A5A", "#F4F4F4", "#3F7F5F"]
    )
    im = ax.imshow(
        mat.values,
        aspect="equal",
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    
    # Label both axes with method names.
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, fontsize=10)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=10)
    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="y", pad=2)

    # Write correlation value inside each heatmap cell.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = float(mat.iloc[i, j])
            text_color = "white" if abs(val) > 0.45 else "#111111"
            ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color=text_color,
            )

    # Add horizontal colour bar below matrix.
    cbar = plt.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        fraction=0.035,
        pad=0.10,
        aspect=40,
    )
    cbar.set_label("Spearman Correlation", fontsize=10, labelpad=4)
    cbar.ax.tick_params(labelsize=9, pad=1)

    # Apply final layout adjustments before rendering.
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.24, top=0.92)
  
    # Render heatmap.
    st.pyplot(fig, use_container_width=False)

    # Render corresponding caption.
    _render_global_agreement_caption(agreement_df)

def _render_agreement_key_insights(
    agreement_df: pd.DataFrame,
) -> None:
    """
    Render key takeaway cards for agreement results.

    Highlights strongest and weakest agreement pairings across
    explanation methods.
    """
    # Retrieve agreement row used to build summary cards.
    agr = _get_agreement_row(agreement_df)
    if agr is None:
        st.info("Agreement insights not available.")
        return

    # Collect available pairwise agreement values.
    pairs = [
        (
            "Captum IG vs SHAP",
            _safe_float(agr.get("SHAP_vs_Captum_Spearman"))
        ),
        (
            "Captum IG vs LIME",
            _safe_float(agr.get("LIME_vs_Captum_Spearman"))
        ),
        (
            "SHAP vs LIME",
            _safe_float(agr.get("SHAP_vs_LIME_Spearman"))
        ),
    ]
    pairs = [(name, val) for name, val in pairs if pd.notna(val)]

    # Stop early if no pairwise values are available.
    if not pairs:
        st.info("Agreement insights not available.")
        return

    # Identify strongest and weakest agreement pairings.
    best_pair = max(pairs, key=lambda x: x[1])
    worse_pair = min(pairs, key=lambda x: x[1])

    # Show strongest agreement card first.
    _insight_card(
        "Strongest Agreement",
        (
            f"{best_pair[0]} shows "
            f"{_agreement_band(best_pair[1]).lower()} "
            f"({best_pair[1]:.3f})."
        ),
        tone="success" if best_pair[1] >= 0.6 else "neutral"
    )

    # Show weakest agreement card below it.
    _insight_card(
        "Weakest Agreement",
        (
            f"{worse_pair[0]} shows "
            f"{_agreement_band(worse_pair[1]).lower()} "
            f"({worse_pair[1]:.3f})."
        ),
        tone="danger" if worse_pair[1] < 0 else "warning"
    )

    # Render corresponding caption below takeaway cards.
    _render_agreement_key_insights_caption(agreement_df)

def _render_risk_failure_modes(
    model_auc: 
    float, 
    method_df: 
    pd.DataFrame, 
    agreement_df: pd.DataFrame,
) -> None:
    """
    Render warning-style insight cards for benchmark outcomes that
    may need caution.

    Summarise potential reliability issues derived from model
    performance, method-level results, and cross-method agreement
    so user can quickly see where interpretation should be more
    careful.
    """
    # Build warning messages first so layout logic can focus only on
    # how resulting cards should be arranged on page.
    flags = _build_risk_flags(model_auc, method_df, agreement_df)

    def _centered_columns(n: int):
        """
        Return column layout that keeps a small number of cards
        visually centred.

        A one- or two-card row is padded with spacer columns so
        warnings do not appear left-aligned when there are only 
        a few items to show.
        """
        # Centre a single card by surrounding it with equal spacer
        # columns.
        if n == 1:
            cols = st.columns([1,2, 1], gap="small")
            return [cols[1]]
        
        # Centre two cards by placing them between narrower spacer
        # columns.
        elif n == 2:
            cols = st.columns([0.5, 1, 1, 0.5], gap="small")
            return [cols[1], cols[2]]
        
        # For three or more cards, use a standard evenly spaced row.
        else:
            return st.columns(n, gap="small")

    # Keep first row to maximum of three cards so section stays
    # readable and consistent with surrounding dashboard layout.
    first_row = flags[:3]
    row_cols = _centered_columns(len(first_row))

    # Render each warning card and assign a stronger tone to items
    # that describe risk or disagreement.
    for col, (title, body) in zip(row_cols, first_row):
        with col:
            tone = (
                "danger"
                if (
                    "risk" in title.lower()
                    or "conflicting" in title.lower()
                )
                else "warning"    
            )
            _insight_card(title, body, tone=tone)

    # Render any remaining warning cards on a second row separated
    # by a small gap so the two rows do not visually run together.
    remaining = flags[3:]
    if remaining:
        st.markdown(
            "<div style='height: 6px;'></div>",
            unsafe_allow_html=True,
        )
        extra_cols = _centered_columns(min(3, len(remaining)))

        # Reuse same tone-selection logic so all warning cards follow
        # same visual rules regardless of which row they appear in.
        for col, (title, body) in zip(extra_cols, flags[3:]):
            with col:
                tone = (
                    "danger"
                    if (
                        "risk" in title.lower()
                        or "conflicting" in title.lower()
                    )
                    else "warning"
                )
                _insight_card(title, body, tone=tone)

def _render_topk_importance_chart(
    feature_df: pd.DataFrame,
    top_k: int = 10,
    baseline_method: str = "Captum_IG",
) -> None:
    """
    Render grouper bar chart used to compare Top-K feature importance
    by method.

    Ranks features by chosen baseline method, then shows same features
    side by side for Captum IG, SHAP, and LIME after normalising each
    method column to comparable scale.
    """
    # Work on a copy so any display-specific coercion or sorting does
    # not alter original feature table reused by other sections.
    df = feature_df.copy()
    needed = {"Feature", baseline_method, "SHAP", "LIME"}

    # Stop early when required feature labels or explainer columns
    # are missing.
    if not needed.issubset(df.columns):
        st.info("Top-K importance chart not available.")
        return

    # Convert all plotted method columns to absolute numeric values
    # because this comparison focuses on attribution magnitude rather
    # than direction.
    for c in [baseline_method, "SHAP", "LIME"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").abs()

    # Normalise each method column before plotting so differences in
    # raw explainer scale do not dominate the visual comparison.
    df = _normalize_method_columns(
        df,
        [baseline_method, "SHAP", "LIME"],
    )
    
    # Select Top-K features according to chosen baseline method.
    df = df.sort_values(baseline_method, ascending=False).head(top_k)

    # Reverse plotting order so highest-ranked feature appears at top
    # of chart once x-axis labels are read from left to right.
    plot_df = df.iloc[::-1]

    # Build grouped bar chart with one bar position per feature
    # and one offset per evaluation method.
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    x = np.arange(len(plot_df))
    width = 0.25

    # Plot baseline method first so feature ranking and bar order
    # match main reference used in this section.
    ax.bar(
        x - width, 
        plot_df[baseline_method], 
        width=width, 
        label="Captum IG", 
        color="#3A6EA5", 
        edgecolor="white", 
        linewidth=0.6,
    )

    # Plot SHAP values for same feature rows to support direct
    # comparison.
    ax.bar(
        x, plot_df["SHAP"], 
        width=width, 
        label="SHAP", 
        color="#E6862A", 
        edgecolor="white", 
        linewidth=0.6,
    )

    # Plot LIME values using third bar position in each feature group.
    ax.bar(
        x + width, 
        plot_df["LIME"], 
        width=width, 
        label="LIME", 
        color="#4C9F70", 
        edgecolor="white", 
        linewidth=0.6,
    )

    # Apply feature labels after ranking so x-axis matches plotted
    # Top-K set.
    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_df["Feature"],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    
    # Label axes clearly because values have already been normalised
    # and should not be interpreted as raw attribution units.
    ax.set_ylabel(
        "Normalised Absolute Attribution",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel(
        "Features",
        fontsize=10,
        fontweight="bold",
        labelpad=10,
    )
    
    # Set y-axis limit from largest plotted bar so full range remains
    # visible while still leaving some space above tallest group.
    ymax = max(
        plot_df[baseline_method].max(),
        plot_df["SHAP"].max(),
        plot_df["LIME"].max(),
    )
    ax.set_ylim(0, ymax * 1.15)

    # Use light gridlines and simplified spines so grouped bars
    # remain main visual focus.
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Place legend below plot.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.55),
        frameon=False,
        ncol=3,
        fontsize=9,
        handlelength=1.5,
        columnspacing=1.5,
    )

    # Apply final layout adjustments before rending.
    fig.subplots_adjust(bottom=0.33, left=0.12, right=0.96, top=0.88)
    
    # Render plot.
    st.pyplot(fig, use_container_width=True)

    # Render corresponding caption.
    _render_topk_importance_caption(
        feature_df,
        top_k,
        baseline_method,
    )

def _render_feature_variability(
    feature_df: pd.DataFrame,
    top_n: int = 10,
) -> None:
    """
    Render feature-variability chart based on mean attribution and
    spread.

    Highlights features whose importance differs more across
    explanation methods by plotting mean absolute attribution
    together with standard deviation across available method columns.
    """
    # Keep only explainer columns that are actually present in
    # exported feature table so section stays compatible with runs
    # where some methods were disabled.
    cols = [
        c
        for c in ["Captum_IG", "SHAP", "LIME"]
        if c in feature_df.columns
    ]
    
    # Exit early when feature labels or enough explainer columns
    # are missing.
    if "Feature" not in feature_df.columns or len(cols) < 2:
        st.info("Feature variability data not available.")
        return

    # Copy source table before applying chart-specific numeric
    # conversions.
    df = feature_df.copy()

    # Covert each explainer column to absolute numeric values so
    # variability metric reflects magnitude differences rather than
    # attribution sign.
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").abs()

    # Normalise explainer columns before computing variability so
    # scale differences between methods do not inflate spread measure.
    df = _normalize_method_columns(df, cols)
    
    # Compute average contribution and between-method spread for
    # each feature.
    df["Attribution_SD"] = df[cols].std(axis=1)
    df["Attribution_Mean"] = df[cols].mean(axis=1)

    # Keep features with largest variation across methods.
    df = df.sort_values("Attribution_SD", ascending=False).head(top_n)

    # Build single bar chart with error bars so both average
    # importance and amount of cross-method variation are visible
    # in one view.
    fig, ax = plt.subplots(figsize=(5.4, 4.6))

    x = np.arange(len(df))

    ax.bar(
        x, 
        df["Attribution_Mean"], 
        yerr=df["Attribution_SD"],
        color="#7A8FA6",
        edgecolor="white",
        linewidth=0.8    
    )
    
    # Label x-axis with feature names after sorting so order matches
    # highest-variability rows selected above.
    ax.set_xticks(x)
    ax.set_xticklabels(df["Feature"], rotation=45, ha="right")

    # Set y-limit using highest mean-plus-spread value so all bars
    # and error bars remain visible.
    ax.set_ylim(
        0,
        (
            df["Attribution_Mean"] + df["Attribution_SD"]
        ).max() * 1.15,
    )
    ax.set_ylabel(
        "Mean Attribution with SD",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel(
        "Features",
        fontsize=10,
        fontweight="bold",
        labelpad=18,
    )
    
    # Keep visual style consistent with other dashboard charts.
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Tighten layout while leaving space for rotated feature labels.
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.subplots_adjust(bottom=0.18, left=0.12, right=0.96, top=0.88)

    # Render chart.
    st.pyplot(fig)

    # Render corresponding caption.
    _render_feature_variability_caption(feature_df)

def _render_instance_level_section(out: dict) -> None:
    """
    Render instance-level comparison across original, noise, and
    masking views.

    Let user choose one patient or instance and one explanation
    method, then compares strongest features across available
    scenarios. It also prepares aligned scenario tables for
    accompanying caption helper.
    """
    # Read instance-level attribution table produced by benchmark
    # engine.
    instance_attr_df = out.get("instance_attr_df")

    # Stop early when required per-instance export is missing or
    # empty.
    if instance_attr_df is None or len(instance_attr_df) == 0:
        st.info(
            "Instance-level explanation stability requires "
            "per-instance attribution outputs. "
        )
        return

    # Confirm that key identifying fields needed for filtering
    # and plotting are available before section tries to build
    # controls or charts.
    required = {"InstanceID", "Scenario", "Feature"}
    if not required.issubset(instance_attr_df.columns):
        st.info(
            "Instance-level attribution data is missing "
            "required columns."
        )
        return

    # Limit explainer choices to method columns present in this run.
    method_options = [
        c
        for c in ["Captum_IG", "SHAP", "LIME"]
        if c in instance_attr_df.columns
    ]
    if not method_options:
        st.info("No per-instance attribution columns were found.")
        return

    # Provide one selector for instance and one for explanation method
    # so chart can update interactively without changing wider page
    # flow.
    c1, c2 = st.columns(2)
    with c1:
        instance_ids = sorted(
            instance_attr_df["InstanceID"]
            .dropna()
            .unique()
            .tolist()
        )
        selected_id = st.selectbox(
            "Select Patient / Instance",
            instance_ids,
            key="instance_select",
        )
    with c2:
        selected_method = st.selectbox(
            "Select Explanation Method", 
            method_options, 
            key="instance_method",
        )

    # Filter full table down to selected instance before deriving
    # scenarios or plotting feature-level results.
    plot_df = instance_attr_df.loc[
        instance_attr_df["InstanceID"] == selected_id
    ].copy()
    scenarios = [
        s
        for s in ["Original", "Noise", "Masking"]
        if s
        in (
            plot_df["Scenario"]
            .dropna()
            .unique()
            .tolist()
        )
    ]
    
    # Stop when selected instance does not contain any of expected
    # scenario labels used by this section.
    if not scenarios:
        st.info("No valid scenarios found for the selected instance.")
        return

    # Create one horizontal bar chart per scenario and share y-axis so
    # feature ordering can be compared more easily across panels.
    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(5.2 * len(scenarios), 4.8),
        sharey=True,
    )
    if len(scenarios) == 1:
        axes = [axes]

    # Plot Top-10 absolute attributions for each available scenario.
    for ax, scenario in zip(axes, scenarios):
        sub = plot_df.loc[plot_df["Scenario"] == scenario].copy()
        sub[selected_method] = pd.to_numeric(
            sub[selected_method],
            errors="coerce",
        ).abs()
        sub = sub.sort_values(selected_method, ascending=False).head(10)

        # Keep scenario colours consistent so same perturbation type
        # is easy to recognise across runs.
        color_map = {
            "Original": "#7A8FA6",
            "Noise": "#9AA9BB",
            "Masking": "#C6D1DD",
        }
        
        ax.barh(
            sub["Feature"],
            sub[selected_method],
            color=color_map.get(scenario, "#FDBE85"),
            edgecolor="white",
            linewidth=0.8,
        )
        ax.invert_yaxis()
        ax.set_title(scenario, fontsize=14, fontweight="bold")
        ax.set_xlabel(
            "Absolute Attribution",
            fontsize=12,
            fontweight="bold",
            labelpad=12,
        )
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="x", alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Add one overall title so multi-panel chart is clearly tied to
    # selected instance rather than each scenario separately.
    fig.suptitle(
        f"Stability for Patient / Instance: {selected_id}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Build one cleaned feature table per scenario so caption helper
    # can compare aligned values across Original, Noise, and Masking.
    scenario_tables = {}

    for scenario in ["Original", "Noise", "Masking"]:
        sub = plot_df.loc[
            plot_df["Scenario"] == scenario,
            [
                "Feature",
                selected_method,
            ],
        ].copy()

        if sub.empty:
            continue

        # Standardise feature labels and attribution values before
        # aggregation.
        sub["Feature"] = sub["Feature"].astype(str)
        sub[selected_method] = pd.to_numeric(
            sub[selected_method],
            errors="coerce",
        ).abs()
        sub = sub.dropna(subset=[selected_method])
        if sub.empty:
            continue

        # Average repeated rows, if present, so each scenario
        # contributes one value per feature to aligned comparison
        # table.
        sub = sub.groupby(
            "Feature",
            as_index=False,
        )[selected_method].mean()

        scenario_tables[scenario] = sub.rename(
            columns={
                selected_method: scenario,
            }
        )
 
    # Only render caption when all three scenario tables are available
    # and can be aligned on shared feature column.
    if all(
        k in scenario_tables
        for k in ["Original", "Noise", "Masking"]
    ):
        aligned = (
            scenario_tables["Original"]
            .merge(scenario_tables["Noise"], on="Feature", how="inner")
            .merge(scenario_tables["Masking"], on="Feature", how="inner")
            .sort_values("Original", ascending=False)
            .reset_index(drop=True)
        )

        # Pass aligned arrays into caption helper so it can describe
        # how same features behave across three scenario views.
        if not aligned.empty:
            _render_instance_stability_caption(
                used_features=aligned["Feature"].tolist(),
                original_vals=aligned["Original"].to_numpy(),
                noise_vals=aligned["Noise"].to_numpy(),
                mask_vals=aligned["Masking"].to_numpy(),
                method_name=selected_method,
                instance_id=selected_id,
            )

def _render_rank_change_plot(out: dict, top_k: int = 10) -> None:
    """
    Render instance-level rank-change chart across available
    scenarios.

    Starts from selected instance and method, converts attribution
    magnitudes into within-scenario feature ranks, and then shows
    how those ranks shift from original view to peturbed views.
    """
    # Read per-instance attribution export needed for rank-based
    # comparison.
    instance_attr_df = out.get("instance_attr_df")

    # Exit early when engine did not return instance-level table.
    if instance_attr_df is None or len(instance_attr_df) == 0:
        st.info(
            "Rank change analysis requires per-instance attirubtion "
            "outputs from the engine. "
        )
        return

    # Check for identifying columns required to filter selected
    # instance and scenario rows before any rank calculations are
    # attempted.
    required = {"InstanceID", "Scenario", "Feature"}
    if not required.issubset(instance_attr_df.columns):
        st.info("Rank change analysis is missing required columns.")
        return 
   
    # Limit method choices to attribution columns actually
    # available in table.
    method_options = [
        c
        for c in ["Captum_IG", "SHAP", "LIME"]
        if c in instance_attr_df.columns
    ]
    if not method_options:
        st.info(
            "No attribution columns were found for rank "
            "change analysis."
        )
        return
   
    # Provide one selector for instance and one for explanation
    # method used to derive rank trajectories.
    c1, c2 = st.columns(2)
    with c1:
        instance_ids = sorted(
            instance_attr_df["InstanceID"]
            .dropna()
            .unique()
            .tolist()
        )
        selected_id = st.selectbox(
            "Select Patient / Instance for Rank Change",
            instance_ids,
            key="rank_change_instance_select",
        )
    with c2:
        selected_method = st.selectbox(
            "Select Explanation Method for Rank Change",
            method_options,
            key="rank_change_method_select",
        )
    
    # Restrict working table to selected instance before computing
    # ranks.
    plot_df = instance_attr_df.loc[
        instance_attr_df["InstanceID"] == selected_id
    ].copy()
    scenarios = [
        s 
        for s in ["Original", "Noise", "Masking"] 
        if s in plot_df["Scenario"].dropna().unique().tolist()
    ]

    # Rank-change view needs at least two scenarios to show any
    # movement.
    if len(scenarios) < 2:
        st.info(
            "At least two scenarios are required to display "
            "rank changes."
        )
        return

    # Convert selected method column to absolute numeric values
    # before ranking.
    plot_df[selected_method] = pd.to_numeric(
        plot_df[selected_method],
        errors="coerce",
    ).abs()
    plot_df = plot_df.dropna(subset=[selected_method])

    # Stop when no usable values remain after coercion.
    if plot_df.empty:
        st.info(
            "No valid attribution values were found for "
            "the selected method."
        )
        return
    
    # Build one feature-rank table per scenario so results can be
    # merged into a single cross-scenario comparison frame.
    rank_tables = []
    for scenario in scenarios:
        sub = plot_df.loc[
            plot_df["Scenario"] == scenario,
            [
                "Feature",
                selected_method,
            ],
        ].copy()

        if sub.empty:
            continue

        # Rank features within scenario from highest absolute
        # attribution to lowest.
        sub = (
            sub.sort_values(
                selected_method,
                ascending=False,
            )
            .reset_index(drop=True)
        )
        sub["Rank"] = np.arange(1, len(sub) + 1)
        sub = sub[["Feature", "Rank"]].rename(
            columns={"Rank": scenario},
        )

        rank_tables.append(sub)

    # Stop if none of scenario-specific rank tables could be created.
    if not rank_tables:
        st.info(
            "Rank tables could not be created for the selected instance."
        )
        return
    
    # Merge scenario rank tables on feature name so each row tracks
    # one feature across available scenarios.
    rank_df = rank_tables[0]
    for tbl in rank_tables[1:]:
        rank_df = rank_df.merge(tbl, on="Feature", how="outer")

    # Chart is anchored on original scenario, so stop if that rank
    # column is absent.
    if "Original" not in rank_df.columns:
        st.info(
            "Original scenario is required for rank change analysis."
        )
        return
    
    # Keep only Top-K features from original scenario so chart remains
    # focused on most important starting features.
    rank_df = (
        rank_df.sort_values(
            "Original",
            ascending=True,
        )
        .head(top_k)
        .copy()
    )

    # Create rank-shift figure with one x-position per scenario.
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    x_positions = np.arange(len(scenarios))

    # Use consistent vertical guide colours so three secnario columns
    # remain easy to distinguish when reading lines and labels.
    color_map = {
        "Original": "#D94801",
        "Noise": "#FD8D3C",
        "Masking": "#FDBE85",
    }

    # Draw one line per feature to show how its rank moves between
    # scenarios.
    for _, row in rank_df.iterrows():
        y_vals = []
        x_vals = []

        # Collect only scenario positions where feature has a
        # defined rank.
        for idx, scenario in enumerate(scenarios):
            val = row.get(scenario, np.nan)
            if pd.notna(val):
                x_vals.append(idx)
                y_vals.append(float(val))

        # Connect available ranks when feature appears in at least
        # two scenarios.
        if len(x_vals) >= 2:
            ax.plot(
                x_vals,
                y_vals,
                linewidth=1.8,
                alpha=0.75,
            )

        # Mark available scenario points even when full line is
        # not present.
        if len(x_vals) >= 1:
            ax.scatter(
                x_vals,
                y_vals,
                s=45,
                zorder=3,
            )

    # Prepare labels so plotted feature trajectories can be identified
    # without separate legend.
    label_rows = []
    for _, row in rank_df.iterrows():
        label_y = row.get("Masking", np.nan)

        # Prefer last scenario position for label, but fall back to
        # earlier scenarios when later ones are missing.
        if pd.isna(label_y):
            label_y = row.get("Noise", np.nan)   

        if pd.isna(label_y):
            label_y = row.get("Original", np.nan)

        if pd.notna(label_y):
            feature_label = str(row["Feature"])

            # Highlight original top-ranked feature directly in
            # label text.
            if (
                pd.notna(row.get("Original", np.nan))
                and int(row["Original"]) == 1
            ):
                feature_label = f"#1 {feature_label}"

            label_rows.append({
                "y": float(label_y),
                "label": feature_label,
            })

    # Order labels from highest to lowest placement before spacing
    # them out.
    label_rows = sorted(label_rows, key=lambda d: d["y"])

    # Apply minimum vertical separation so nearby text labels do not
    # overlap.
    min_gap = 1.1
    adjusted_y = []

    for item in label_rows:
        y = item["y"]
        if adjusted_y and abs(y - adjusted_y[-1]) < min_gap:
            y = adjusted_y[-1] + min_gap
        adjusted_y.append(y)
        item["y_adj"] = y

    # Draw feature labels just to right of final scenario column.
    for item in label_rows:
        ax.text(
            len(scenarios) - 1 + 0.08,
            item["y_adj"],
            item["label"],
            fontsize=9,
            va="center",
            ha="left",
            clip_on=False,
            fontweight=(
                "bold" 
                if item["label"].startswith("#1")
                else None
            ),
        )

    # Label axes and title after rank trajectories are drawn.
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_xlim(-0.1, len(scenarios) - 1 + 0.35)
    ax.set_ylabel("Feature Rank", fontsize=10, fontweight="bold")
    ax.set_title(
        f"Rank Change (Top-{top_k} Original Features)",
        fontsize=9,
        fontweight="bold",
        pad=12,
    )

    # Set y-axis from observed maximum rank and invert it so rank 1
    # appears at top of chart.
    scenario_cols = [s for s in scenarios if s in rank_df.columns]
    numeric_vals = rank_df[scenario_cols].to_numpy(dtype=float)
    finite_vals = numeric_vals[np.isfinite(numeric_vals)]
    max_rank = int(finite_vals.max()) if finite_vals.size > 0 else top_k

    ax.set_ylim(max_rank + 0.5, 0.5)
    ax.grid(axis="y", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add light vertical guides for each scenario column for improved
    # readability.
    for idx, scenario in enumerate(scenarios):
        ax.axvline(
            idx,
            color=color_map.get(scenario, "#CCCCCC"),
            alpha=0.10, linewidth=8,
        )

    # Reserve space on right for direct feature labels.
    plt.subplots_adjust(left=0.10, right=0.86, top=0.86, bottom=0.16)
    st.pyplot(fig, use_container_width=True)

    # Convert merged rank table into aligned arrays for paired
    # caption helper.
    used_features = rank_df["Feature"].astype(str).tolist()
    original_vals = rank_df["Original"].to_numpy(dtype=float)
    noise_vals = (
        rank_df["Noise"].to_numpy(dtype=float) 
        if "Noise" in rank_df.columns 
        else original_vals.copy()
    )
    mask_vals = (
        rank_df["Masking"].to_numpy(dtype=float) 
        if "Masking" in rank_df.columns 
        else original_vals.copy()
    )
    method_name = _clean_method_name(selected_method)
    instance_id = int(selected_id)

    # Render caption after chart.
    _render_rank_shift_caption(
        used_features,
        original_vals,
        noise_vals,
        mask_vals,
        method_name,
        instance_id,
        top_k=top_k,
    )

def _render_advanced_analysis(
        method_df: pd.DataFrame, 
        agreement_df: pd.DataFrame, 
        pairwise_df: pd.DataFrame, 
        feature_df: pd.DataFrame,
        out: dict,
        top_k: int,
) -> None:
    """
    Render expandable section that groups dashboard's supporting
    tables.

    Gives user access to underlying benchmark tables used
    throughout page along with a few derived views that help explain
    how methods differ on stability, clinical coverage, and
    instance-level rank shift.
    """
    # Keep detailed tables inside a collapsed panel so main narrative
    # remains readable while still preserving access to full outputs.
    with st.expander(
        "Show advanced statistical analysis and full tables",
        expanded=False,
    ):
        _subsection_header(
            "MLP vs Logistic Regression vs Random Forest"
        )
        model_comparison_df = out.get("model_comparison_df")

        if model_comparison_df is None or len(model_comparison_df) == 0:
            st.info("Model comparison table not available.")
        else:
            model_comparison_df = model_comparison_df.copy()

            # Round main numeric fields for cleaner display without
            # changing stored benchmark outputs.
            for col in [
                "Validation_AUC", 
                "Test_AUC", 
                "Test_AP",
                "Test_Brier",
                "Test_ECE",
                "hidden_dim", 
                "lr", 
                "weight_decay", 
                "batch_size", 
                "C",
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
            ]:
                if col in model_comparison_df.columns:
                    model_comparison_df[col] = pd.to_numeric(
                        model_comparison_df[col], errors="coerce"
                    )

            preferred_cols = [
                "Model",
                "Used_for_Explanations",
                "Explanation_Eligible",
                "Model_Role",
                "Validation_AUC",
                "Test_AUC",
                "Test_AP",
                "Test_Brier",
                "Test_ECE",
                "hidden_dim",
                "lr",
                "weight_decay",
                "batch_size",
                "C",
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
            ]

            display_cols = [
                c
                for c in preferred_cols
                if c in model_comparison_df.columns
            ]
            remaining_cols = [
                c
                for c in model_comparison_df.columns
                if c not in display_cols
            ]
            model_comparison_df = model_comparison_df[
                display_cols + remaining_cols
            ]

            st.dataframe(
                model_comparison_df,
                use_container_width=True,
                hide_index=True,
            )

        # Surface LIME repeatability diagnostic.
        _subsection_header("LIME Repeatability")
        lime_repeatability = out.get("lime_repeatability", {}) or {}

        if not lime_repeatability:
            st.info("LIME repeatability results not available.")
        else:
            spearman_key = "mean_instance_spearman"
            topk_key = f"mean_instance_top{int(top_k)}_overlap"

            repeatability_df = pd.DataFrame(
                [
                    {
                        "Metric": "LIME Repeatability",
                        "Mean_Instance_Spearman": (
                            lime_repeatability.get(
                                spearman_key,
                                np.nan,
                            )
                        ),
                        "Mean_Signed_Instance_Spearman": (
                            lime_repeatability.get(
                                "mean_signed_instance_spearman",
                                np.nan,
                            )
                        ),
                        f"Mean_Instance_Top{int(top_k)}_Overlap": (
                            lime_repeatability.get(
                                topk_key,
                                np.nan,
                            )
                        ),
                        "LIME_Repeatability_Repeats": getattr(
                            out.get("config"),
                            "lime_repeatability_repeats",
                            np.nan,
                        ),
                        "Sample_Around_Instance": getattr(
                            out.get("config"),
                            "lime_sample_around_instance",
                            np.nan,
                        ),
                    }
                ]
            )

            st.dataframe(
                repeatability_df,
                use_container_width=True,
                hide_index=True,
            )

        # Show paiwise statisical comparison table when it is
        # available.
        _subsection_header("Statistical Tests")
        if pairwise_df is None or pairwise_df.empty:
            st.info("Pairwise statistical results not available.")
        else:
            preferred_cols = [
                "Comparison",
                "Wilcoxon_stat",
                "p_value",
                "p_adj_bonferroni",
                "Rank_Biserial",
                "Effect_size",
                "Significance_Label",
                "Implication",
            ]
            display_cols = [
                c
                for c in preferred_cols
                if c in pairwise_df.columns
            ]
            st.dataframe(
                (
                    pairwise_df[display_cols]
                    if display_cols
                    else pairwise_df
                ),
                use_container_width=True,
                hide_index=True,
            )

        # Display method-level sumamry table used across several
        # later view.
        _subsection_header("Method Summary")
        if method_df is None or method_df.empty:
            st.info("Method summary table not available.")
        else:
            st.dataframe(
                method_df,
                use_container_width=True,
                hide_index=True,
            )

        # Surface cross-method agreement export without altering
        # its schema.
        _subsection_header("Agreement")
        if agreement_df is None or agreement_df.empty:
            st.info("Agreement table not available.")
        else:
            st.dataframe(
                agreement_df,
                use_container_width=True,
                hide_index=True,
            )

        # Show feature-level importance export that underpins Top-K
        # views.
        _subsection_header("Feature Importance Table")
        if feature_df is None or feature_df.empty:
            st.info("Feature importane table not available.")
        else:
            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True,
            )

        # Resolve Top-K overlap columns defensively so view remains
        # compatible with alternate export labels.
        _subsection_header("Top-K Stability Overlap")
        topk_noise_col = _first_existing(
            method_df, 
            [
                f"Stability_Top{top_k}_Noise_Overlap_Mean",
                f"Top{top_k}_Noise_Stability",
            ],
        )
        topk_mask_col = _first_existing(
            method_df, 
            [
                f"Stability_Top{top_k}_Masking_Overlap_Mean",
                f"Top{top_k}_Masking_Stability",
            ],
        )

        # Render reduced table that focuses only on method label and
        # two Top-K stability fields most relevant to this subsection.
        if (
            topk_noise_col
            and topk_mask_col
            and "Method" in method_df.columns
        ):
            overlap_df = method_df[
                ["Method", topk_noise_col, topk_mask_col]
            ].copy()
            overlap_df["Method"] = overlap_df["Method"].map(
                _clean_method_name
            )
            st.dataframe(
                overlap_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Top-K stability overlap table not available.")

        # Collect clinical-alignment fields present in current
        # summary table instead of assuming every export uses same
        # schema.
        _subsection_header("Clinical Coverage")
        clinical_cols = [
            c for c in [
                "Method",
                f"Top{top_k}_Clinical_Overlap",
                "Weighted_Clinical_Overlap",
                "Clinical_Precision",
                "Clinical_Recall",
                f"Clinical_Top{top_k}_Overlap_Features",
            ]
            if method_df is not None and c in method_df.columns
        ]

        # Show clinical summary in one compact table when at least one
        # of the expected clinical columns is available.
        if clinical_cols:
            clinical_df = method_df[clinical_cols].copy()
            if "Method" in clinical_df.columns:
                clinical_df["Method"] = clinical_df["Method"].map(
                    _clean_method_name
                )
            st.dataframe(
                clinical_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Clinical coverage table not available.")

        # Build method-by-method list of clincally validated features
        # that do not appear in each method's Top-K set.
        _subsection_header("Missing Clinical Features by Method")
        feature_df_prepped = _prepare_clinical_feature_df(feature_df)
        if (
            method_df is None
            or method_df.empty
            or feature_df_prepped is None
            or feature_df_prepped.empty
        ):
            st.info("Missing clinical feature table not available.")
        else:
            missing_rows = []

            # Evaluate each method row separately so output makes it
            # clear which clinically validated featured were not
            # captured.
            for _, row in method_df.iterrows():
                method = _clean_method_name(row.get("Method", ""))
                missing = _get_missing_clinical_feature_list(
                    row,
                    feature_df_prepped,
                    top_k,
                )
                missing_rows.append(
                    {
                        "Method": method,
                        "Missing_Clinical_Features": (
                            ", ".join(missing) 
                            if missing 
                            else "None"
                        ),
                        "MissingCount": len(missing),
                    }
                )
           
            missing_df = pd.DataFrame(missing_rows)
            st.dataframe(
                missing_df,
                use_container_width=True,
                hide_index=True,
            )

        # Add instance-level diagnostic table showing how feature
        # ranks shift across original and perturbed scenarios.
        _subsection_header("Instance-Level Rank Shift")
        instance_attr_df = out.get("instance_attr_df")
        if instance_attr_df is None or len(instance_attr_df) == 0:
            st.info("Instance-level attribution data not available.")
        else:
            required = {"InstanceID", "Scenario", "Feature"}

            # Confirm that minimum identifying fields exist before
            # trying to build rank-shift details.
            if not required.issubset(instance_attr_df.columns):
                st.info(
                    "Rank shift detail table is missing "
                    "required columns."
                )
            else:
                method_options = [
                    c
                    for c in ["Captum_IG", "SHAP", "LIME"]
                    if c in instance_attr_df.columns
                ]
                
                # Stop early when no explainer columns are present
                # in export.
                if not method_options:
                    st.info(
                        "No attribution columns were found for "
                        "rank shift details."
                    )
                else:
                    c1, c2 = st.columns(2)

                    # Let user choose instance and explanation method
                    # used for detailed rank-shift table.
                    with c1:
                        instance_ids = sorted(
                            instance_attr_df["InstanceID"]
                            .dropna()
                            .unique()
                            .tolist()
                        )
                        selected_id = st.selectbox(
                            "Select Patient / Instance for Rank "
                            "Shift Table",
                            instance_ids,
                            key="advanced_rank_change_instance_select",
                        )

                    with c2:
                        selected_method = st.selectbox(
                            "Select Explanation Method for Rank "
                            "Shift Table ",
                            method_options,
                            key="advanced_rank_change_method_select",
                        )

                    # Restrict analysis to selected instance so later
                    # rank calculations compare scenarios for same
                    # case only.
                    plot_df = instance_attr_df.loc[
                        instance_attr_df["InstanceID"] == selected_id
                    ].copy()
                  
                    scenarios = [
                        s
                        for s in ["Original", "Noise", "Masking"]
                        if s in plot_df["Scenario"].dropna().unique().tolist()
                    ]
                  
                    # Rank shift view requires at least two scenarios
                    # to show any change across conditions.
                    if len(scenarios) < 2:
                        st.info(
                            "At least two scenarios are required to "
                            "display rank shift details."
                        )
                    else:
                        # Convert selected attribution column to
                        # numeric magnitude and pivot table into
                        # one row per feature.
                        pivot_df = (
                            plot_df[["Feature", "Scenario", selected_method]]
                            .copy()
                            .assign(
                                **{
                                    selected_method: lambda d: pd.to_numeric(
                                        d[selected_method],
                                        errors="coerce",
                                    ).abs()
                                }
                            )
                            .pivot_table(
                                index="Feature",
                                columns="Scenario",
                                values=selected_method,
                                aggfunc="mean",
                            )
                        )

                        # Convert attribution magnitudes into
                        # descending ranks so table focuses on
                        # relative feature ordering.
                        rank_df = pivot_df.rank(
                            axis=0,
                            ascending=False,
                            method="min",
                        )
                        rank_df = rank_df.rename(
                            columns={
                                "Original": "Original",
                                "Noise": "Noise",
                                "Masking": "Masking",
                            }
                        ).reset_index()

                        # Keep original Top-K rows first so detail
                        # table aligns with dashboard's Top-K framing
                        # elsewhere.
                        if "Original" in rank_df.columns:
                            rank_df = rank_df.sort_values(
                                "Original",
                                ascending=True,
                            ).head(top_k)
                        
                        # Add signed rank difference columns so upward
                        # or downward movement can be read directly
                        # from table.
                        if (
                            "Noise" in rank_df.columns
                            and "Original" in rank_df.columns
                        ):
                            rank_df["Noise_Shift"] = (
                                rank_df["Noise"] - rank_df["Original"]
                            )

                        if (
                            "Masking" in rank_df.columns
                            and "Original" in rank_df.columns
                        ):
                            rank_df["Masking_Shift"] = (
                                rank_df["Masking"] - rank_df["Original"]
                            )

                        st.dataframe(
                            rank_df,
                            use_container_width=True,
                            hide_index=True,
                        )

def _render_key_takeaways(
        rel_df: pd.DataFrame,
        model_auc: float,
        agreement_df: pd.DataFrame, 
        method_df: pd.DataFrame, 
        top_k: int
) -> None:
    """
    Render high-level takeaway cards that summarise main benchmark
    results.

    Text in these cards is derived from reliability table, agreement
    table, model ROC-AUC, and clinical-overlap fields so summary
    remains tied to exported benchmark outputs.
    """
    # Read leading method from reliability ranking so section can
    # state which explainer performs best overall.
    best_method = (
        rel_df.iloc[0]["Method"]
        if not rel_df.empty
        else "N/A"
    )
    best_rel = (
        rel_df.iloc[0]["Reliability_Score"]
        if not rel_df.empty
        else np.nan
    )

    # Start with a neutral fallback message in case agreement table
    # is unavailable or does not contain usable values.
    agr = _get_agreement_row(agreement_df)
    best_agreement_text = "Agreement data not available."
    if agr is not None:
        pair_map = {
            "Captum IG vs SHAP": _safe_float(
                agr.get("SHAP_vs_Captum_Spearman")
            ),
            "Captum IG vs LIME": _safe_float(
                agr.get("LIME_vs_Captum_Spearman")
            ),
            "SHAP vs LIME": _safe_float(
                agr.get("SHAP_vs_LIME_Spearman")
            ),
        }

        # Remove missing values before selecting strongest agreement
        # pair.
        pair_map = {k: v for k, v in pair_map.items() if pd.notna(v)}
        if pair_map:
            best_pair = max(pair_map.items(), key=lambda x: x[1])
            best_agreement_text = (
                f"{best_pair[0]} shows the strongest global "
                f"agreement ({best_pair[1]:.3f})."
            )

    # Resolve preferred clinical overlap field so section works with
    # both dyanmic and fallback Top-K column names.
    overlap_col = _first_existing(
        method_df, 
        [f"Top{top_k}_Clinical_Overlap", "Top10_Clinical_Overlap"],
    )
    clinical_text = "Clinical overlap data not available."
    if overlap_col:
        temp = method_df[["Method", overlap_col]].copy()
        temp[overlap_col] = pd.to_numeric(
            temp[overlap_col],
            errors="coerce",
        )
        temp = temp.sort_values(overlap_col, ascending=False)

        # Describe leading method only when at least one valid row
        # remains.
        if not temp.empty:
            top_method = _clean_method_name(temp.iloc[0]["Method"])
            top_count = int(temp.iloc[0][overlap_col])
            clinical_text = (
                f"{top_method} captures the most clinically "
                f"validated features in the top-{top_k} set "
                f"({top_count})."
            )

    # Reuse model-performance interpretation helper so wording stays
    # consistent with rest of dashboard.
    risk_text = _auc_interpretation(model_auc)[1]

    # Centre overall recommendation card so it remais visually
    # distinct from three supporting takeaway cards below.
    _, top_center, _ = st.columns(
        [1, 2, 1],
        gap="small",
    )

    with top_center:
        _insight_card(
            "Overall Recommendation",
            (
                f"{best_method} achieves the highest composite "
                f"reliability score ({_format_metric(best_rel)})."
            ), 
            tone="primary",
            icon="\u2B50",
        )

    # Add small spacer before three summary cards.
    st.markdown(
        "<div style='height: 6px;'></div>",
        unsafe_allow_html=True,
    )
    
    c1, c2, c3 = st.columns(3, gap="small")
    
    # Summarise model discrimintation in same card style used
    # elsewhere.
    with c1:
        _insight_card(
            "Baseline Model Takeaway",
            (
                "Baseline Model ROC-AUC is "
                f"{model_auc:.3f}. {risk_text}"
            ),
            tone="neutral",
            icon="\U0001F4CA",
        )

    # Report strongest explainer to explainer agreement relationship.
    with c2:
        _insight_card(
            "Agreement Takeaway", 
            best_agreement_text,
            tone="neutral",
            icon="\U0001F517",
        )
    
    # Highlight which method captures most clinically validated
    # features.
    with c3:
        _insight_card(
            "Clinical Takeaway", 
            clinical_text, 
            tone="neutral",
            icon="\u2695"
        )  

def _render_evaluation_against_objectives(
        rel_df: pd.DataFrame,
        method_df: pd.DataFrame,
        top_k: int,
) -> None:
    """
    Render dashboard section that evaluates benchmark results against
    project's three core objectives of fidelity, robustness, and
    clinical alignment.

    Identifies strongest method for each objective using summary
    tables already prepared elsewhere in interface. It then presents
    those results as three aligned insight cards followed by short
    conclusion card that links findings back to research question.
    """
    # Stop early when reliability summary is missing so dashboard can
    # continue rendering without raising errors in this section.
    if rel_df is None or rel_df.empty:
        st.info(
            "Evaluation against research objectives is not available."
        )

    # Find preferred clinical overlap column for current Top-K
    # setting. Fallback keeps this section compatible with exports
    # that still use a fixed Top-10 naming pattern.
    overlap_col = _first_existing(
        method_df,
        [f"Top{top_k}_Clinical_Overlap", "Top10_Clinical_Overlap"],
    )

    # Initialise per-objective winners so text below can fall
    # back to neutral message when metric is unavailable.
    best_fidelity = None
    best_robustness = None
    best_clinical = None

    # Identify method with strongest fidelity score when that metric
    # is available in reliability summary table.
    if (
        "Fidelity_Score" in rel_df.columns
        and rel_df["Fidelity_Score"].dropna().any()
    ):
        row = rel_df.sort_values(
            "Fidelity_Score", 
            ascending=False,
        ).iloc[0]
        best_fidelity = (
            _clean_method_name(row["Method"]), 
            float(row["Fidelity_Score"]),
        )
    
    # Identify method with strongest robustness score so dashboard can
    # summarise which explainer remains most stable under
    # perturbation.
    if (
        "Robustness_Score" in rel_df.columns 
        and rel_df["Robustness_Score"].dropna().any()
    ):
        row = rel_df.sort_values(
            "Robustness_Score", 
            ascending=False,
        ).iloc[0]
        best_robustness = (
            _clean_method_name(row["Method"]),
            float(row["Robustness_Score"]),
        )

    # Use method summary table to determine which explainer captures
    # most clinically validated features within selected Top-K set.
    if overlap_col and "Method" in method_df.columns:
        tmp = method_df[["Method", overlap_col]].copy()
        tmp[overlap_col] = pd.to_numeric(
            tmp[overlap_col],
            errors="coerce",
        )
        tmp = tmp.dropna(subset=[overlap_col])

        # Keep only rows with usable overlap values before
        # ranking methods.
        if not tmp.empty:
            row = tmp.sort_values(
                overlap_col,
                ascending=False,
            ).iloc[0]
            best_clinical = (
                _clean_method_name(row["Method"]),
                float(row[overlap_col]),
            )

    # Build short explantory text for fidelity objective card.
    # Fallback wording avoids overstating results when metric
    # is unavailable.
    fidelity_text = (
        f"{best_fidelity[0]} best reflects model behaviour "
        "based on the highed fidelity score "
        f"({_format_metric(best_fidelity[1])})."
        if best_fidelity 
        else "Fidelity results are not available."
    )

    # Summarise which method appears most robust under noise or
    # incomplete input.
    robustness_text = (
        f"{best_robustness[0]} handles noisy or incomplete "
        "EHR input most reliably based on the highest "
        f"robustness score ({_format_metric(best_robustness[1])})."
        if best_robustness
        else "Robustness results are not available."
    )

    # Summarise which method shows strongest clinical alignment
    # according to selected Top-K overlap metric.
    clinical_text = (
        f"{best_clinical[0]} shows the strongest alignment "
        "with clinically meaningful variables, capturing "
        f" {int(best_clinical[1])} of the Top-{top_k} clinically "
        "validated features."
        if best_clinical
        else "Clinical alignment results are not available."
    )

    # Create three column layout so each project objective is
    # presented with same visual weight and structure.
    c1, c2, c3 = st.columns(3, gap="small")

    # Objective 1 focuses on how closely each explanation method
    # reflects model behaviour.
    with c1:
        _insight_card(
            "Objective 1: Fidelity",
            (
                "Does it reflect model behaviour?<br><br>" 
                + fidelity_text
            ),
            tone="primary",
            icon="\U0001F4CA",
        )

    # Objective 2 focuses on whether explanations remain dependable
    # when inputs are perturbed to represent noisier or less complete
    # EHR data.
    with c2:
        _insight_card(
            "Objective 2: Robustness",
            (
                "Does it handle noisy EHR data?<br><br>" 
                + robustness_text
            ),
            tone="primary",
            icon="\U0001F4CA",
        )

    # Objective 3 focuses on whether highly ranked features align with
    # project's clinically validated reference set.
    with c3:
        _insight_card(
            "Objective 3: Clinical Alignment",
            (
                "Does it match clinical reasoning?<br><br>" 
                + clinical_text
            ),
            tone="primary",
            icon="\U0001F4CA",
        )
    
    # Add small spacer before conclusion card so final takeaway reads
    # as a separate summary rather than part of three card grid.
    st.markdown(
        "<div style='height: 8px;'></div>", 
        unsafe_allow_html=True,
    )

    # Link objective level findings back to wider dissertation aim
    # using one concise statement that connects benchmark outputs to
    # research question.
    overall_text = (
        "These results address the research question by showing "
        "which method performs best and how they behave under "
        "realistic clinical conditions across fidelity, robustness, "
        "and alignment with clinically validated predictors."
    )

    # Present final summary as standaone card to close section.
    _insight_card(
        "Research Question Link",
        overall_text,
        tone="success",
        icon="\U0001F4CA"
    )

# --------------------------------------------------------------------
# Presentation theme and layout styling
# Custom CSS below defines visual theme used through dashboard.
# Centralises layout, colour, spacing, and component styling so
# interface appears consistent across header, sidebar, cards, tables,
# controls, and supporting text.
# --------------------------------------------------------------------
def _inject_css() -> None:
    """
    Inject full custom CSS theme used across dashboard.

    Keeping dashboad styling in one function makes presentation layer
    easier to review, maintain, and update. Rules below customise
    default Streamlit elements so interface is readable and consistent
    throughout.
    """
    # Inject custom theme before rest of interface is rendered so
    # later components inherit intended colours, spacing, and layout
    # rules.
    st.markdown(
        """
        <style>
        /* Keep native Streamlit header present without forcing */
        /* its internal controls. */
        [data-testid="stHeader"] { 
            background: transparent !important; 
        }

        /* Hide top-right Streamlit toolbar that contains */
        /* Deploy and menu. */
        [data-testid="stToolbar"] {
            display: none !important;
            visibility: hidden !important;
        }

        /* Apply outer dashboard background gradient and page */
        /* padding around main content region. */ 
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(
                135deg,
                #0B1F33 0%,
                #FFFFFF 40%,
                #FFFFFF 60%,
                #0B1F33 100%
            );
            padding: 12px 24px 20px 24px !important;
        }

        /* Style main content container so dashboard sits inside */
        /* centred white panel. */
        [data-testid="stMainBlockContainer"] {
            background: #FFFFFF;
            border-radius: 18px;
            padding: 1.2rem 2rem 2rem 2rem;
            max-width: 1320px !important;
            width: calc(100% - 32px) !important;
            margin: 16px auto !important;
            box-shadow:
                0 -12px 30px rgba(0,0,0,0.12),
                0 20px 40px rgba(0,0,0,0.12),
                0 6px 12px rgba(0,0,0,0.08);
    
            border: 3px solid #8AB0D6 !important;
            overflow: visible !important;
        }

        /* Keep first content block as-is, do not pull into */
        /* header area. */
        .block-container > div:first-child {
            margin-top: 0 !important;
        }

        /* Give sidebar its own panel styling so controls are */
        /* visually separate from results. */
        [data-testid="stSidebar"] {
            background-color: #D7E4F1 !important;
            border: 3px solid #8AB0D6 !important;
            box-shadow: 6px 0 16px rgba(0, 0, 0, 0.08) !important;
            border-radius: 18px !important;
        }

        /* Centre main sidebar headings and apply dashboard */
        /* accent colour. */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            text-align: center !important;
            color: #234E70 !important;
            font-weight: 750 !important;
        }

        /* Enlarge main sidebar heading so control area has */
        /* clear title. */
        [data-testid="stSidebar"] h1 {
            font-size: 2.1rem !important;
            margin-top: 0rem !important;
            margin-bottom: 0.2rem !important;
            padding-top: 0 !important;
        }

        /* Style sidebar subsection headings so parameter groups */
        /* remain easy to scan. */
        [data-testid="stSidebar"] h3 {
            font-size: 1.55rem !important;
            margin-top: 0.4rem !important;
            margin-bottom: 0.4rem !important;
            color: #111111 !important;
        }

        /* Centre sidebar labels and helper text to match */
        /* surrounding control layout. */
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown p {
            text-align: center !important;
        }

        /* Give BaseWeb input containers a white field background */
        /* with rounded corners. */
        [data-testid="stSidebar"] [data-baseweb="input"] {
            background-color: #FFFFFF !important;
            border-radius: 10px !important;
            border: 1px solid #B7C9D6 !important;
        }

        /* Keep sidebar input values dark so entered text remains */
        /* readable against white field. */
        [data-testid="stSidebar"] [data-baseweb="input"] input {
            color: #111111 !important;
            -webkit-text-fill-color: #111111 !important;
            background-color: #FFFFFF !important;
        }

        /* Restyle trailing control region on sidebar inputs so */
        /* it matches darker action colour. */
        [data-testid="stSidebar"] [data-baseweb="input"] > div:last-child {
            background-color: #111827 !important;
            border-left: 1px solid #111827 !important;
            border-radius: 0 10px 10px 0 !important;
        }

        /* Keep icons inside trailing sidebar input control */
        /* white for contrast. */
        [data-testid="stSidebar"]
        [data-baseweb="input"]
        > div:last-child button,
        [data-testid="stSidebar"]
        [data-baseweb="input"]
        > div:last-child button * {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
        }

        /* Make slider labels span available width and stay */
        /* centred above each slider. */
        [data-testid="stSidebar"] .stSlider label {
            width: 100% !important;
            text-align: center !important;
        }

        /* Add vertical spacing below each slider so control */
        /* groups do not seem crowded. */
        [data-testid="stSidebar"] .stSlider {
            margin-bottom: 0.8rem !important;
        }

        /* Expand each checkbox row across sidebar width so labels */
        /* align consistently. */
        [data-testid="stSidebar"] .stCheckbox {
            display: flex !important;
            justify-content: flex-start !important;
            width: 100% !important;
        }

        /* Align checkbox control and label text on one line for */
        /* cleaner settings rows. */
        [data-testid="stSidebar"] .stCheckbox > label {
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
            justify-content: flex-start !important;
            width: 100% !important;
            text-align: left !important;
        }

        /* Set checkbox accent color to dashboard's highlight tone. */
        [data-testid="stSidebar"] input[type="checkbox"] {
            accent-color: #EF5A47;
        }

        /* Replace standard sidebar divider with wider gradient */
        /* rule. */
        [data-testid="stSidebar"] hr{
            border: none !important;
            height: 3px !important;
            background: linear-gradient(
                to right, 
                transparent, 
                #234E70, 
                #8AB0D6, 
                #234E70, 
                transparent
            ) !important;
            margin: 12px auto !important;
            width: 95% !important;
            border-radius: 4px !important;
        }

        /* Remove extra gap above first sidebar text input so */
        /* control stack starts higher. */
        [data-testid="stSidebar"] .stTextInput:first-of-type {
            margin-top: 0 !important;
        }

        /* Centre sidebar button container so run control sits */
        /* prominently in panel. */
        [data-testid="stSidebar"] .stButton {
            display: flex !important;
            justify-content: center !important;
            width: 100% !important;
            margin-top: 0.6rem !important;
        }

        /* Style sidebar run botton so it stands out as main */
        /* action control. */
        [data-testid="stSidebar"] .stButton > button {
            width: 78% !important;
            min-height: 54px !important;
            border-radius: 14px !important;
            background-color: #18314F !important;
            color: #FFFFFF !important;
            border: none !important;
            font-size: 1.6rem !important;
            font-weight: 800 !important;
            line-height: 1.1 !important;
            box-shadow: none !important;
        }

        /* Darken sidebar run button slightly on hover to provide */
        /* feedback without changing layout. */
        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #12263D !important;
            color: #FFFFFF !important;
        }

        /* Keep all nested text and icon elements inside sidebar */
        /* run button white. */
        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] .stButton > button * ,
        [data-testid="stSidebar"] .stButton > button span,
        [data-testid="stSidebar"] .stButton > button div,
        [data-testid="stSidebar"] .stButton > button button p {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }

        /* Preserve same white foreground when sidebar run button */
        /* is hovered. */
        [data-testid="stSidebar"] .stButton > button:hover,
        [data-testid="stSidebar"] .stButton > button:hover *,
        [data-testid="stSidebar"] .stButton > button:hover span,
        [data-testid="stSidebar"] .stButton > button:hover div,
        [data-testid="stSidebar"] .stButton > button:hover p {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }

        /* Keep sidebar text-input wrappers white so they remain */
        /* visually distinct. */
        [data-testid="stSidebar"] .stTextInput [data-baseweb="input"] {
            background-color: #FFFFFF !important;
        }

        /* Ensure sidebar text input values remain dark and */
        /* easy to read. */
        [data-testid="stSidebar"]
        .stTextInput
        [data-baseweb="input"] input {
            color: #111111 !important;
            -webkit-text-fill-color: #111111 !important;
            background-color: #FFFFFF !important;
        }

        /* Force standard button text across interface to */
        /* stay white for contrast. */
        .stButton button,
        .stButton button *,
        .stButton button span,
        .stButton button div {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }

        /* Slightly increase weight and size of standard app */
        /* buttons outside sidebar. */
        .stButton > button {
            font-weight: 700 !important;
            font-size: 1.05rem;
        }

        /* Increase emphasis on standard buttons when hovered. */
        .stButton > button:hover {
            font-weight: 800;
        }

        /* Remove default number input chrome so custom field */
        /* styling controls appearance. */
        [data-testid="stNumberInput"] {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
        }

        /* Style number-input fields with centred dark text */
        /* on white background. */
        [data-testid="stNumberInput"] input {
            background-color: #FFFFFF !important;
            color: #0F1F33 !important;
            -webkit-text-fill-color: #0F1F33 !important;
            border-radius: 9px !important;
            font-weight: 700 !important;
            text-align: center !important;
            font-size: 1rem !important;
            border: none !important;
            height: 40px !important;
            box-shadow: none !important;
        }

        /* Remove heavier broswer focus styling so field matches */
        /* rest of custom theme. */
        [data-testid="stNumberInput"] input:focus {
            outline: none !important;
            box-shadow: none !important;
        }

        /* Style number input stepper region with dark background */ 
        /* behind increment controls. */
        [data-testid="stNumberInput"] div:has(button) {
            background: #0F1F33;
            border-radius: 8px;
        }

        /* Format number input step buttons so they remain readable */
        /* and visually balanced. */
        [data-testid="stNumberInput"] button {
            background-color: transparent !important;
            color: #FFFFFF !important;
            border: none !important;
            font-size: 1.05rem;
            font-weight: 800;
            width: 34px;
            height: 40px;
            min-height: 40px !important;
            box-shadow: none !important;
        }

        /* Add hover state to number input step buttons for */
        /* interactive feedback. */
        [data-testid="stNumberInput"] button:hover {
            background-color: #1B314D !important;
            color: #FFFFFF !important;
            border-radius: 6px !important;
        }

        /* Add small gap between number field and nearby controls. */
        [data-testid="stNumberInput"] > div {
            gap: 4px !important;
        }

        /* Tighten spacing below number input labels and */
        /* add slightly stronger emphasis. */
        [data-testid="stNumberInput"] label {
            margin-bottom: 0.35rem !important;
            font-weight: 500 !important;
        }

        /* Remove internal gaps so number input field and */
        /* stepper appear joined. */
        [data-testid="stNumberInput"] > div[data-baseweb="input"] {
            gap: 0 !important;
        }

        /* Style trailing number input segment so it matches */
        /* darker stepper design. */
        [data-testid="stNumberInput"]
        > div[data-baseweb="input"]
        > div:last-child {
            background: #0F1F33 !important;
            border-radius: 0 9px 9px 0 !important;
            border-left: 1px solid rgba(255,255,255,0.08) !important;
        }

        /* Keep normal top spacing for page body. */
        .block-container {
            padding-top: 1.25rem !important;
        }

        /* Create branded page header panel that contains logo */ 
        /* title, and introductory text. */
        .header-container {
            background: linear-gradient(
                180deg, 
                #D4E1EF 0%, 
                #EEF4FA 100%
            );
            padding: 26px 34px 24px 34px;
            border: 1px solid #C7D2DA;
            border-radius: 16px;
            box-sizing: border-box;
            margin: 0 auto 24px auto;
            text-align: center;
            width: 100%;
            max-width: 1000px;
            min-height: 80px
        }

        /* Centre logo within custom header panel. */
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 2px 0 12px 0;
        }

        /* Contol logo image size so it scales cleanly within header. */
        .logo-container img {
            width: 330px;
            max-width: 100%;
            height: auto;
        }

        /* Style main dashboard title with strong emphasis */
        /* and tight spacing. */
        .main-title {
            text-align: center;
            margin: 6px 0 8px 0;
            color: #111111 !important;
            font-weight: 800 !important;
            letter-spacing: 0.3px;
            font-size: 2.7rem;
            line-height: 1.15;
        }

        /* Add short accent divider below main title. */
        .header-divider {
            width: 82px;
            height: 4px;
            background: #355070;
            margin: 12px auto 18px auto;
            border-radius: 2px;
        }

        /* Format introductory paragraph below page title. */
        .intro-text {
            max-width: 900px;
            margin: 0 auto;
            text-align: center;
            font-size: 1.05rem;
            line-height: 1.7;
            color: #111111 !important;
        }

        /* Ensure custom section heading colours are inherited */
        /* consistently by nested elements. */
        h2.main-section,
        h2.main-section *,
        h3.sub-section,
        h3.sub-section * {
            color: inherit !important;
        }

        /* Style major section headers as wide dark bars to create */
        /* stronger visual structure. */
        .section-header-box {
            max-width: 1300px;
            margin: 24px auto 18px auto;
            padding: 5px 12px;
            line-height: 1.2;
            text-align: center;
            font-size: 2.1rem;
            font-weight: 900 !important;
            letter-spacing: 0.3px;
            color: #FFFFFF !important;
            background: #18314F;
            border: 1px solid #18314F;
            border-radius: 10px;
            box-shadow: none;
        }

        /* Style subsection headings with accent color and */
        /* centred alignment. */
        h3.sub-section {
            color: #3F8FD2 !important;
            font-weight: 700;
            font-size: 1.8rem;
            text-align: center;
            margin-bottom: 0.8rem;
            margin-top: 0.4rem;
        }

        /* Define base appearance for metric summary cards. */
        .metric-card {
            background: #F7FAFC;
            border: 3px solid #D9E2EC;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            border-radius: 14px;
            padding: 18px 16px;
            text-align: center;
            height: 100%;
            min-height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            box-sizing: border-box;
        }

        /* Apply warning state colours to metric cards. */
        .metric-warning {
            background: #FFF7E8;
            border-color: #E0B95F;
            color: #0B1F33 !important;
        }

        /* Apply danger state colours to metric cards. */
        .metric-danger {
            background: #FCECEC;
            border-color: #E37474;
            color: #0B1F33 !important;
        }

        /* Apply success state colours to metric cards. */
        .metric-success {
            border-color: #8FC79D;
            color: #0B1F33 !important;
        }

        /* Style label shown at top of each metric card. */
        .metric-title {
            font-size: 0.98rem;
            font-weight: 700;
            margin-bottom: 8px;
            color: #0B1F33 !important;
        }

        /* Style main numeric value displayed inside each */
        /* metric card. */
        .metric-value {
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 6px;
            color: #0B1F33 !important;
        }

        /* Format supporting explanatory text shown below */
        /* metric values. */
        .metric-subtitle {
            font-size: 0.95rem;
            line-height: 1.5;
            color: #0B1F33 !important;
        }

        /* Define base appearance for interpretation and */
        /* takeaway cards. */
        .insight-card {
            background: #F7FAFC;
            border: 3px solid #D9E2EC;
            border-radius: 16px;
            padding: 16px 16px;
            text-align: center;
            min-height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            height: 100%;
            width: 100%;
            margin-bottom: 14px;
            box-sizing: border-box;
            align-items: center;
            gap: 6px;
            color: #0B1F33 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }

        /* Apply highlighted primary style used for */
        /* more prominent insight cards. */
        .insight-primary {
            background: linear-gradient(
                180deg,
                #F3F8F4 0%,
                #FFFFFF 100%
            );
            border: 2px solid #8FC79D;
            box-shadow: 0 6px 16px rgba(80, 140, 95, 0.14);
            min-height: 160px;
        }

        /* Apply warning colours to insight cards. */
        .insight-warning {
            background: #FFF7E8;
            border-color: #E0B95F;
            color: #0B1F33 !important;
        }

        /* Apply danger colours to insight cards. */
        .insight-danger {
            background: #FCECEC;
            border-color: #E37474;
            color: #0B1F33 !important;
        }

        /* Apply success colours to insight cards. */
        .insight-success {
            background: #EAF7EC;
            border-color: #8FC79D;
            color: #0B1F33 !important;
        }

        /* Style heading inside each insight card. */
        .insight-title {
            font-weight: 800;
            font-size: 1.05rem;
            line-height: 1.35;
            margin-bottom: 4px;
            text-align: center;
            color: #0B1F33 !important;
            max-width: 95%;
        }

        /* Format narrative text displayed inside each insight card. */
        .insight-body {
            line-height: 1.65;
            text-align: center;
            color: #0B1F33 !important;
            font-size: 0.98rem;
            max-width: 95%;
            margin: auto;
        }

        /* Style icons displayed at top of each insight card. */
        .insight-icon {
            font-size: 1.55rem;
            line-height: 1;
            margin-bottom: 4px;
        }

        /* Keep all text inside insight cards on dashboard's */
        /* dark foreground colour. */
        .insight-card,
        .insight-card * {
            color: #0B1F33 !important;
        }

        /* Preserve same foreground colour for warning insight cards */
        /* and nested content. */
        .insight-warning,
        .insight-warning * {
            color: #0B1F33 !important;
        }

        /* Preserve same foreground colour for danger insight cards */
        /* and nested content. */
        .insight-danger,
        .insight-danger * {
            color: #0B1F33 !important;
        }

        /* Preserve same foreground colour for success insight cards */
        /* and nested content. */
        .insight-success,
        .insight-success * {
            color: #0B1F33 !important;
        }

        /* Preserve same foreground colour for primary insight cards */
        /* and nested content. */
        .insight-primary,
        .insight-primary * {
            color: #0B1F33 !important;
        }

        /* Format background, border, and padding for reusable */
        /* information panels. */
        .panel-card {
            background: #EAF3FB;
            border: 1px solid #C7D2DA;
            border-radius: 12px;
            padding: 14px 16px;
            height: 100%;
            color: #0B1F33 !important;
        }

        /* Style title displayed at top of each panel card. */
        .panel-title {
            font-size: 1.05rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 12px;
            color: #0B1F33 !important;
        }

        /* Arrange smaller summary boxes in two column grid. */
        .mini-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        /* Create compact statistics boxes that sit inside */
        /* panel cards. */
        .mini-box {
            background: #FFFFFF;
            border: 1px solid #D9E2EC;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            color: #0B1F33 !important;
        }

        /* Format descriptive label inside each compact */
        /* statistics box. */
        .mini-label {
            font-size: 0.88rem;
            margin-bottom: 6px;
            color: #0B1F33 !important;
        }

        /* Emphasise main value displayed inside each compact */
        /* statistics box. */
        .mini-value {
            font-size: 1.4rem;
            font-weight: 800;
            color: #0B1F33 !important;
        }

        /* Format reusable chart captions so interpretation text */
        /* appears centred below figures. */
        .caption-container {
            max-width: 860px;
            margin: 0.8rem auto 1.6rem auto;
            text-align: center;
            line-height: 1.7;
            font-size: 1.0rem;
            color: #111111 !important;
        }

        /* Centre each download button container below its */
        /* related content. */
        .stDownloadButton {
            display: flex !important;
            justify-content: center !important;
            margin-bottom: 12px !important
        }

        /* Style download button to match lighter dashboard */
        /* accent palette. */
        div[data-testid="stDownloadButton"] > button,
        .stDownloadButton button {
            background-color: #D7E4F1 !important;
            color: #0B1F33 !important;
            -webkit-text-fill-color: #0B1F33 !important;
            border: none !important;
            border-radius: 14px !important;
            font-weight: 800 !important;
            font-size: 1.35rem !important;
            min-height: 60px !important;
            padding: 16px 24px !important;
            width: 100% !important;
            letter-spacing: 0.3px;
        }

        /* Keep all nested download button text and wrappers on */
        /* same dark foreground colour. */
        div[data-testid="stDownloadButton"] > button *,
        div[data-testid="stDownloadButton"] > button span,
        div[data-testid="stDownloadButton"] > button div,
        .stDownloadButton button *,
        .stDownloadButton button span,
        .stDownloadButton button div {
            color: #0B1F33 !important;
            -webkit-text-fill-color: #0B1F33 !important;
            font-weight: 800 !important;
        }

        /* Preserve same appearance when download button is */
        /* hovered. */
        div[data-testid="stDownloadButton"] > button:hover,
        .stDownloadButton button:hover {
            background-color: #D7E4F1 !important;
            color: #0B1F33 !important;
            -webkit-text-fill-color: #0B1F33 !important;
        }

        /* Force standard metric cards and their nested text to use */
        /* dashboard's dark foreground. */
        .metric-card,
        .metric-card * {
            color: #0B1F33 !important;
        }

        /* Force warning metric cards and nested text to use same */
        /* foreground colour. */
        .metric-warning,
        .metric-warning * {
            color: #0B1F33 !important;
        }

        /* Force danger metric cards and nested text to use same */
        /* foreground colour. */
        .metric-danger,
        .metric-danger * {
            color: #0B1F33 !important;
        }

        /* Force success metric cards and nested text to use same */
        /* foreground colour. */
        .metric-success,
        .metric-success * {
            color: #0B1F33 !important;
        }

        /* Reconfirm title colour used at top of each metric card. */
        .metric-title {
            color: #0B1F33 !important;
        }
        
        /* Reconfirm main numeric value colour used inside */
        /* each metric card. */
        .metric-value {
            color: #0B1F33 !important;
        }

        /* Reconfirm supporting text colour displayed below */
        /* metric values. */
        .metric-subtitle {
            color: #0B1F33 !important;
        }

        /* Style collapsed expander header so it reads as */
        /* clickable summary control. */
        [data-testid="stExpander"] summary {
            background-color: #EAF3FB !important;
            color: #0B1F33 !important;
            border: 1px solid #C7D2DA !important;
            border-radius: 10px !important;
            padding: 10px 14px !important;
            font-weight: 600;
        }

        /* Darken expander header slightly when section is open. */
        [data-testid="stExpander"][aria-expanded="true"] summary {
            background-color: #D8E6F2 !important;
            color: #0B1F33 !important;
        }

        /* Add hover feedback to expander headers without changing */
        /* overall theme. */
        [data-testid="stExpander"] summary:hover {
            background-color: #D8E6F2 !important;
            color: #0B1F33 !important;
        }

        /* Keep expander headers visually consistent when focused */
        /* or clicked. */
        [data-testid="stExpander"] summary:focus,
        [data-testid="stExpander"] summary:active {
            background-color: #D8E6F2 !important;
            color: #0B1F33 !important;
            outline: none !important;
        }

        /* Style expander body so opened content appears attached */
        /* to summary header. */
        [data-testid="stExpander"] > div{
            background-color: #FFFFFF !important;
            border: 1px solid #D9E2EC !important;
            border-top: none !important;
            border-radius: 0 0 10px 10px !important;
            padding: 12px !important;
        }

        /* Give dropdown option lists white background */
        /* for readability. */
        div[role="listbox"] {
            background-color: #FFFFFF !important;
            color: #0F1F33 !important;
        }

        /* Set default text colour for dropdown options. */
        div[role="option"] {
            color: #0F1F33 !important;
        }

        /* Highlight hovered dropdown options with */
        /* light accent fill. */
        div[role="option"]:hover {
            background-color: #EAF3FB !important;
        } 

        /* Add border around dropdown option lists so popover */
        /* seems more defined. */
        div[role="listbox"] {
            background-color: #FFFFFF !important;
            border: 1px solid #C7D2DA !important;
        }

        /* Keep dropdown option rows on white background with */
        /* dark text. */
        div[role="option"] {
            background-color: #FFFFFF !important;
            color: #0B1F33 !important;
        }
        
        /* Hightlight hovered dropdown options with lighter */
        /* accent tone. */
        div[role="option"]:hover {
            background-color: #EAF3FB !important;
            color: #0B1F33 !important;
        }

        /* Add vertical spacing around alert messages so notices */
        /* do not crowd nearby sections. */
        [data-testid="stAlert"] {
            margin-top: 10px !important;
            margin-bottom: 18px !important;
        }

        /* Reduce internal alert padding slightly to keep */
        /* notices compact. */
        .stAlert {
            padding: 8px 12px !important;
        }

        /* Make Streamlit columns stretch their child containers */ 
        /* so adjacent cards can sharesame height. */
        [data-testid="column"] > div {
            display: flex;
            height: 100%;
        }

        /* Let insight cards expand to fill available column */
        /* height. */
        [data-testid="column"] .insight-card {
            flex: 1;
            height: 100%;
        }

        /* Force common text elements to use dark foreground */
        /* for readability. */
        p, label, span {
            color: #111111;
        }

        /* Tighten paragraph spacing so explantory text blocks */
        /* appear compact. */
        p {
            margin-bottom: 0.4rem !important;
        }

        /* Apply dashboard styling to BaseWeb select controls */
        /* used by Streamlit dropdowns. */
        [data-baseweb="select"] > div {
            background-color: #D7E4F1 !important;
            border-radius: 10px !important;
            border: 1px solid #8AB0D6 !important;
        }

        /* Keep selected dropdown text dark and readable. */
        [data-baseweb="select"] span {
            color: #0B1F33 !important;
            font-weight: 600 !important;
        }

        /* Style visible selectbox field to match rest of */
        /* control palette. */
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
             background-color: #D8E6F2 !important;
             border: 1px solid #8AB0D6 !important;
             border-radius: 10px !important;
             min-height: 44px !important;
             box-shadow: none !important;
        }

        /* Keep selectbox text, placeholders, and wrappers dark */
        /* throughout. */
        [data-testid="stSelectbox"] [data-baseweb="select"] div,
        [data-testid="stSelectbox"] [data-baseweb="select"] span,
        [data-testid="stSelectbox"] [data-baseweb="select"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] input::placeholder {
            color: #111111 !important;
            -webkit-text-fill-color: #111111 !important;
            opacity: 1 !important;
            font-weight: 500 !important;
        }

        /* Match inner selectbox wrapper to outer field styling. */
        [data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
            background-color: #D8E6F2 !important;
            color: #111111 !important;
        }

        /* Darken selectbox arrow icon so it remains visible */
        /* against pale background. */
        [data-testid="stSelectbox"] [data-baseweb="select"] svg,
        [data-testid="stSelectbox"] [data-baseweb="select"] svg * {
            color: #111111 !important;
            fill: #111111 !important;
            stroke: #111111 !important;
        }

        /* Style dropdown popover panel so option list matches */
        /* visible control. */
        div[data-baseweb="popover"] > div {
            background-color: #D8E6F2 !important;
            border: 1px solid #8AB0D6 !important;
            border-radius: 10px !important;
            box-shadow: 0 6px 16px rgba(0,0,0,0.08) !important;
            overflow: hidden !important;
        }

        /* Remove extra padding from dropdown menu list for */
        /* tighter layout. */
        ul[data-baseweb="menu"] {
            background-color: #D8E6F2 !important;
            padding: 4px 0 !important;
            margin: 0 !important;
        }

        /* Give each dropdown option row readable dark colour */
        /* and consistent background. */
        ul[data-baseweb="menu"] li {
            background-color: #D8E6F2 !important;
            color: #0B1F33 !important;
        }

        /* Force nested dropdown-option text to inherit same */
        /* dark colour as main row. */
        ul[data-baseweb="menu"] li *,
        ul[data-baseweb="menu"] li span,
        ul[data-baseweb="menu"] li div {
            color: #111111 !important;
            -webkit-text-fill-color: #111111 !important;
        }

        /* Highlight active or hovered dropdown option with */
        /* darker accent fill. */
        ul[data-baseweb="menu"] li[aria-selected="true"],
        ul[data-baseweb="menu"] li:hover {
            background-color: #BBD1EA !important;
            color: #0B1F33 !important;
        }

        /* Keep text inside hovered or selected dropdown options */
        /* dark and readable. */
        ul[data-baseweb="menu"] li[aria-selected="true"] *,
        ul[data-baseweb="menu"] li:hover * {
            color: #0B1F33 !important;
            -webkit-text-fill-color: #0B1F33 !important;
        }

        /* Create card used to display clinically captured feature */
        /* lists in distinct panel style. */
        .feature-list-card {
            margin-top: 12px;
            padding: 10px 12px 12px 12px;
            background: #F7FAFD;
            border: 1px solid #C9D6E2;
            border-radius: 10px;
        }

        /* Create card used to display clinically missing feature */
        /* lists. */
        .feature-list-card-missing {
            margin-top: 10px;
            background: #F7FAFD;
            border: 1px solid #D8C6C6;
        }

        /* Format title at top of each feature list card. */
        .feature-list-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: #1F2D3D;
            margin-bottom: 8px;
            text-align: left;
        }

        /* Arrange feature chips in flexible wrapping layout. */
        .feature-chip-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: flex-start;
        }

        /* Style individual feature chips for readability and */
        /* consistency. */
        .feature-chip {
            display: inline-block;
            padding: 7px 10px;
            background: #EAF3FB;
            border: 1px solid #C7D2DA;
            border-radius: 8px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #0B1F33;
            line-height: 1.2;
            white-space: nowrap;
        }

        /* Style placeholder chip shown when no features are */
        /* available. */
        .feature-chip-empty {
            background: #F3F5F7;
            color: #5C6B77;
            font-weight: 500;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------------------------
# Page shell: header, sidebar, and application entry point
# Groups dashboard header, sidebar settings, and main page flow.
# Keeping these pieces together makes it easier to follow how
# interface is assembled from page load through result rendering.
# --------------------------------------------------------------------
def _render_header() -> None:
    """
    Render branded header shown at top of reporting dashboard.

    Responsible for presentation only. It prepares logo,
    title, divider, and introductory project text, then sends
    completed HTML block to Streamlit for display.
    """
    # Reference expected logo file once so existence check and file
    # read both use same path.
    logo_path = Path("logo/University-of-Liverpool.png")

    # Start with empty logo block so header can still render cleanly
    # when image file is unavailable.
    logo_html = ""

    # Only attempt to read and embed image when logo file is present
    # This avoids file-loading error during page setup.
    if logo_path.exists():
        # Read image in binary mode because base64 conversion
        # expects raw bytes.
        with open(logo_path, "rb") as f:
            # Convert image to base64 so it can be embedded directly
            # into HTML without relyng on seperate static file route.
            logo_base64 = base64.b64encode(f.read()).decode()

        # Build logo container seperately so it can be inserted into
        # main header block only when logo is available.
        logo_html = f"""
<div class="logo-container">
<img src="data:image/png;base64,{logo_base64}">
</div>
"""

    # Assemble complete header markup in one string so title, divider,
    # and introductory text are kept together as one visual block.
    header_html = textwrap.dedent(f"""
<div class="header-container">
{logo_html}
<div class="main-title">Healthcare XAI Benchmarking Dashboard</div>
<div class="header-divider"></div>
<div class="intro-text">
This dashboard presents an interactive comparison of two explainable
AI (XAI) methods (SHAP and LIME) on MIMIC-III clinical data,
evaluating stability, fidelity, and clinical alignment. Captum 
(Integrated Gradients) is included as a baseline for reference.
<br><br>
Robert Viens Serna, MSc in Data Science and Artificial 
Intelligence<br>
University of Liverpool, CSCK700 Computer Science Capstone Project
August 2025 C
</div>
</div>
""").strip()
    
    # Render prepared HTML exactly once so header appears as a single,
    # consistently styled panel at top of page.
    st.markdown(header_html, unsafe_allow_html=True)

# --------------------------------------------------------------------
# Sidebar Configuration
# Converts benchmark configuration into editable sidebar controls
# and then rebuilds configuration object from selected values.
# --------------------------------------------------------------------
def _build_sidebar_config() -> tuple[Any, bool]:
    """
    Build sidebar controls used to configure a benchmark run.

    Starts from benchmark configuration dataclass, exposes
    selected settings through Streamlit widgets, and returns rebuilt
    configuration object and state of Run Benchmark button.
    """
    # Add clear title so purpose of sidebar is obvious before
    # individual controls are shown.
    st.sidebar.title("Settings")

    # Insert divider below heading to seperate title from first
    # group of controls.
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    # Create default run configuration first so sidebar widgets
    # reflect same baseline settings used by benchmark engine.
    try:
        cfg = bench.RunConfig()
    except Exception as e:
        # Surface configuration errors in sidebar because that is
        # where user is interacting with run settings.
        st.sidebar.error(f"Could not instantiate RunConfig: {e}")
        raise
    
    # Some code paths may not return dataclass-like object. In that
    # case, provide only run button and return early rather than
    # attempting to render field-based widgets.
    if not is_dataclass(cfg):
        st.sidebar.markdown("<hr>", unsafe_allow_html=True)
        run_button = st.sidebar.button(
            "Run Benchmark", 
            use_container_width=True
        )
        return cfg, run_button

    # Convert dataclass to dictionary so widget values can be updated
    # incrementally as user changes settings.
    cfg_values = asdict(cfg)

    # Centralise sidebar labels so displayed wording stays consistent
    # even if internal confiugration field names vary.
    label_map = {
        "data_path": "Model-Ready Data Path (Parquet)",
        "parquet_path": "Model-Ready Data Path (Parquet)",
        "test_size": "Test Size",
        "test_subset_rows": "Test Subset Rows",
        "training_epochs": "Training Epochs",
        "epochs": "Training Epochs",
        "run_shap": "Run SHAP",
        "run_lime": "Run LIME",
        "noise_std": "Noise Standard Deviation",
        "mask_frac": "Mask Fraction",
        "mask_runs": "Mask Runs",
        "perm_repeats": "Permutation Repeats",
        "background_n": "Number of Background Samples (SHAP)",
        "lime_sample_size": "LIME Instances to Explain",
        "sample_size": "Sample Size",
        "lime_num_samples": "Number of Samples",
        "lime_repeats": "Repeated LIME Runs",
        "lime_sample_around_instance": "Sample Around Instance",
        "discretize_continuous": "Discretize Continuous Features",
        "top_k": "Top Features to Compare (K)",
    }

    # Show dataset path field using whichever path attribute is
    # available in current configuration object.
    if hasattr(cfg, "data_path"):
        cfg_values["data_path"] = st.sidebar.text_input(
            "Model-Ready Dataset Path (Parquet)",
            value=str(cfg_values["data_path"])
        )

    elif hasattr(cfg, "parquet_path"):
        cfg_values["parquet_path"] = st.sidebar.text_input(
            "Model-Ready Dataset Path (Parquet)",
            value=str(cfg_values["parquet_path"])
        )
    
    # Seperate file-path field from next group of run controls.
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    # Expose test split as slider when configuration includes that
    # setting.
    if hasattr(cfg, "test_size"):
        cfg_values["test_size"] = st.sidebar.slider(
            "Test Size",
            min_value=0.05,
            max_value=0.50,
            value=float(cfg_values["test_size"]),
            step=0.01,
        )

    # Bind sidebar subset size control to RunConfig field used by
    # benchmarking engine.
    cfg_values["n_debug"] = st.sidebar.number_input(
        "Test Subset Rows",
        value=int(
            cfg_values.get("n_debug", getattr(bench, "N_DEBUG", 40))
        ),
        step=50,
        min_value=10,
        max_value=5000,
    )

    # Bind epoch control to RunConfig field used by
    # benchmarking engine during model training.
    cfg_values["n_train_epochs"] = st.sidebar.number_input(
        "Training Epochs",
        value=int(
            cfg_values.get(
                "n_train_epochs", 
                getattr(bench, "N_TRAIN_EPOCHS", 15)
            )
        ),
        step=50,
        min_value=5,
        max_value=200,
    )

    # Support alternative training-epoch fields names so sidebar
    # remains compatible with more than one configuration schema.
    if hasattr(cfg, "training_epochs"):
        cfg_values["training_epochs"] = st.sidebar.number_input(
            "Training Epochs",
            value=int(cfg_values["training_epochs"]),
            step=5,
            min_value=5,
            max_value=200,
        )

    elif hasattr(cfg, "epochs"):
        cfg_values["epochs"] = st.sidebar.number_input(
            "Training Epochs",
            value=int(cfg_values["epochs"]),
            step=5,
            min_value=5,
            max_value=200,
        )
    else: 
        cfg_values["training_epochs"] = st.sidebar.number_input(
            "Training Epochs",
            value=int(getattr(bench, "N_TRAIN_EPOCHS", 50)),
            step=5,
            min_value=5,
            max_value=200,
        )

    # ----------------------------------------------------------------
    # Method toggles
    # ----------------------------------------------------------------
    # Group explainer checkboxes under dedicated heading so user can
    # quickly see which methods will be included in run.
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("### Methods")

    # Iterate over dataclass fields so method-related controls are
    # drawn from configuration itself rather than being hard-coded
    # multiple times.
    for field in fields(cfg):
        name = field.name
        value = cfg_values[name]
        low_name = name.lower()

        # Only render checkboxes for explainer toggle fields handled
        # in this section.
        if low_name in {"run_shap", "run_lime"}:
            cfg_values[name] = st.sidebar.checkbox(
                label_map.get(low_name, name), 
                value=bool(value),
            )

    # ----------------------------------------------------------------
    # Robustness Controls
    # ----------------------------------------------------------------
    # Place perturbation settings together because they all affect
    # robustness analysis performed during benchmarking.
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("### Robustness")

    for field in fields(cfg):
        name = field.name
        value = cfg_values[name]
        low_name = name.lower()

        # Render noise level as numeric input because small decimal
        # changes may matter to benchmark configuration.
        if low_name in {"noise_std"}:
            cfg_values[name] = st.sidebar.number_input(
                label_map.get(low_name, name), 
                value=float(value), 
                format="%.4f",
            )

        # Rendering masking fraction as slider because setting is
        # naturally bounded between 0 and 1.
        elif low_name in {"mask_frac"}:   
            cfg_values[name] = st.sidebar.slider(
                label_map.get(low_name, name), 
                min_value=0.0, 
                max_value=1.0, 
                value=float(value), 
                step=0.01,
            )

        # Render number of masking repetitions as integer input.
        elif low_name in {"mask_runs"}:   
            cfg_values[name] = st.sidebar.number_input(
                label_map.get(low_name, name), 
                value=int(value), 
                step=1,
            )  

    # ----------------------------------------------------------------
    # Baselines and Repeat Counts
    # ----------------------------------------------------------------
    # Keep repeat-based controls together because they shape size of
    # several downstream benchmark procedures.
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("### Baselines / Repeats")

    for field in fields(cfg):
        name = field.name
        value = cfg_values[name]
        low_name = name.lower()
            
        # Settings are integer counts, so number inputs are
        # appropriate.
        if low_name in {"perm_repeats", "background_n"}:
            cfg_values[name] = st.sidebar.number_input(
                label_map.get(low_name, name), 
                value=int(value), 
                step=10,
            )

    # ----------------------------------------------------------------
    # LIME settings
    # ----------------------------------------------------------------
    # Keep LIME-specific controls together so they are easy to find
    # without scanning entire sidebar.
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("### LIME")

    # Track whether discretisation control has already been drawn
    # under existing field name.
    discretize_rendered = False
    
    for field in fields(cfg):
        name = field.name
        value = cfg_values[name]
        low_name = name.lower()

        # Render LIME sample count as integer input so user can change
        # explanation samplig budget directly.
        if low_name in {"lime_num_samples", "num_samples"}:
            cfg_values[name] = st.sidebar.number_input(
                label_map.get(low_name, name), 
                value=int(value), 
                step=100,
            )

        # Accept several possible field names for discretisation
        # toggle so interface stays compatible with alternate
        # configuration schemas.
        elif low_name in {
            "lime_discretize",
            "discretize_continuous",
            "discretize",
        }:
            cfg_values[name] = st.sidebar.checkbox(
                "Discretize Continuous Features",
                value=bool(value),
                key="discretize_continuous_checkbox"
            )
            discretize_rendered = True

    # Add fallback discretisation checkbox when no compatible field
    # was found in dataclass loop above.
    if not discretize_rendered:
        cfg_values["lime_discretize"] = st.sidebar.checkbox(
            "Discretize Continuous Features",
            value=bool(getattr(bench, "LIME_DISCRETIZE", False)),
            key="discretize_continuous_checkbox",
        )
    
    # ----------------------------------------------------------------
    # Top-K feature setting
    # ----------------------------------------------------------------
    # Keep Top-K control separate because it affects several
    # comparison and visualisation sections later in dashboard.
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("### Top Features")
    for field in fields(cfg):
        name = field.name
        value = cfg_values[name]
        low_name = name.lower()

        # Only render control for Top-K setting in this section.
        if low_name == "top_k":
            cfg_values[name] = st.sidebar.number_input(
                label_map.get(low_name, "Top-K Features"),
                value=int(value) if value is not None else 10,
                step=1,
                min_value=1,
                max_value=50,
            )
    
    # Place run button after all configuration inputs so user reviews
    # full setup before starting benchmark.
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    run_btn = st.sidebar.button(
        "Run Benchmark", 
        use_container_width=True,
    )

    # Rebuild config using only fields that exist in RunConfig.
    valid_fields = {f.name for f in fields(cfg)}
    clean_cfg_values = {
        k: v
        for k, v in cfg_values.items()
        if k in valid_fields
    }
    
    try:
        rebuilt_cfg = bench.RunConfig(**clean_cfg_values)
    except TypeError:
        rebuilt_cfg = cfg

    # Some configurations may use differently named field for LIME
    # discretisation option. Attach value manually when needed.
    if "discretize_continuous" in cfg_values and not hasattr(
        rebuilt_cfg, 
        "discretize_continuous",
    ):
        setattr(
            rebuilt_cfg,
            "discretize_continuous", 
            cfg_values["discretize_continuous"],
        )

    return rebuilt_cfg, run_btn

# --------------------------------------------------------------------
# Main Application Flow
# Applies styling, gathers user esttings, optionally runs
# benchmark, and renders result sections in intended order.
# --------------------------------------------------------------------
def main() -> None:
    """
    Run reporting interface from page setup through output rendering.

    Establishes page layout, injects custom styling, reads sidebar
    configuration, executes benchmark when requested, and then 
    displays results sections and download options.
    """

    # Apply custom CSS before rendering main content so later elements
    # inherit intended styling.
    _inject_css()

    # Render branded title block and introductory context at top of
    # page.
    _render_header()

    # Read sidebar inputs and capture updated configuration and
    # current state of run button.
    cfg, run_btn = _build_sidebar_config()

    # Reserve placeholder near top of page for status updates.
    status_box = st.empty()

    # Keep status helper nested because it is only used inside main
    # page flow.
    def render_status(message: str, kind: str = "info"):
        """
        Render status banner displayed below main header.

        Provides consistent place for guidance, progress messages,
        and completion feedback during interaction with dashboard.
        """
        # Choose background and text colours from status type so
        # banner reflects whether message is informational or confirms
        # success.
        bg = "#D8E6F2" if kind == "info" else "#EAF7EC"
        text = "#1F3A5F" if kind == "info" else "#1E7E34"

        # Use reference message with similar length so banner keeps
        # stable height when visible message changes.
        reference_message = (
            "Configure experimental parameters within the Settings "
            "sidebar and run the benchmark to evaluate the results."
            "Please note that benchmark may take a few minutes to "
            "complete."
        )

        # Build banner as one HTML block so layout, colours, and
        # spacing are applied consistently each time message is
        # updated.
        html = f"""
        <div 
            style="
                position: relative;
                background-color: {bg};
                color: {text};
                padding: 14px 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 0.95rem;
                margin-top: 10px;
                font-weight: 500;
                box-sizing: border-box;
            "
        >
            <div style="visibility: hidden; line-height: 1.4;">
                {reference_message}
            </div>
            <div 
                style="
                    position: absolute;
                    top: 14px;
                    left: 20px;
                    right: 30px;
                    bottom: 14px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                    line-height: 1.4;
                "
            >
                {message}
            </div>
        </div>
        """
        
        # Replace contents of reserved placehoder with latest status
        # message.
        status_box.markdown(html, unsafe_allow_html=True)

    # Show instructional message and stop early when user has not
    # started benchmark run yet.
    if not run_btn:
        render_status(
            "Configure experimental parameters within the Settings "
            "sidebar and run the benchmark to evaluate the results. "
            "Please note that the benchmark may take a few minutes "
            "to complete."
        )
        return
    
    # Update page to show that benchmark has started before engine
    # is called.
    render_status("Running benchmark...")

    # Execute benchmark using current sidebar configuration.
    out = bench.run_benchmark(cfg)

    # Replace progress message with completion message once run
    # finishes.
    render_status("Benchmark complete.", kind="success")      

    # Extract main outputs used repeatedly across remaining dashboard
    # sections.
    model_auc = float(out["model_auc"])

    # Work on copies of returned tables so later display-specific
    # edits do not modify original objects stored in output
    # dictionary.
    method_df = _get_method_summary_df(out["method_df"].copy())
    agreement_df = out["agreement_df"].copy()
    feature_df = _get_feature_importance_df(out["feature_df"].copy())
    pairwise_df = out["pairwise_df"].copy()
    model_comparison_df = out.get(
        "model_comparison_df",
        pd.DataFrame(),
    ).copy()

    # Read active Top-K setting once so every section uses same value.
    top_k = _safe_int(
        getattr(cfg, "top_k", _top_k_default()), _top_k_default()
    )

    # Build reliability summary table that supports several of
    # high-level comparison views below.
    rel_df = _build_reliability_table(method_df, top_k) 

    # ----------------------------------------------------------------
    # 1. Model Context
    # ----------------------------------------------------------------
    # Start with model-level context because later explanation reults
    # are easier to read once baseline predictive performance is
    # visible.
    _section_header("1. Model Context")
    st.markdown(
        '<div class="section-divider"></div>', 
        unsafe_allow_html=True,
    )   
    _subsection_header("ROC-AUC and Model Warning")
    _render_model_context(model_auc)
    _render_model_context_caption(model_auc)

    # ----------------------------------------------------------------
    # 2. Overall Method Comparison
    # ----------------------------------------------------------------
    # Present broad method comparison before moving into more
    # specialised sections such as robustness or clinical alignment.
    _section_header("2. Overall Method Comparison")
    st.markdown(
        '<div class="section-divider"></div>', 
        unsafe_allow_html=True,
    )

    # Create two-column layout with narrow divider column between
    # panels.
    c1, c_div, c2 = st.columns([1, 0.02, 1])

    with c1:
        _subsection_header("Best-Performing Methods")
        # Add vertical spacing so left and right panels remain
        # visually balanced.
        st.markdown(
            "<div style='padding-bottom: 5.5rem;'>", 
            unsafe_allow_html=True,
        )
        _render_best_method_cards(rel_df)

    with c_div:
        _render_column_divider()

    with c2:
        _subsection_header("Method Trade-off Profile")
        _render_radar_chart(rel_df)

    _render_horizontal_divider()
    st.markdown(
        "<div style='margin-top: 1.8rem;'></div>", 
        unsafe_allow_html=True,
    )

    c1, c_div, c2 = st.columns([1, 0.02, 1])

    with c1:
        _subsection_header("Composite Reliability Score")
        _render_reliability_bars(rel_df)

    with c_div:
        _render_column_divider()

    with c2:
        _subsection_header(
            "Normalised Reliability Score Decomposition"
        )
        _render_reliability_decomposition(rel_df)

    # ----------------------------------------------------------------
    # 3. Robustness & Stability
    # ----------------------------------------------------------------
    # Keep robustness views together because they all describe how
    # explanation behaviour changes under perturbation.
    _section_header("3. Robustness & Stability")
    _render_method_performance_summary(method_df)
    _render_horizontal_divider()

    c1, c_div, c2 = st.columns([1, 0.02, 1])

    with c1:
        _subsection_header(f"Top-{top_k} Stability Overlap")
        _render_topk_stability_overlap(method_df, top_k)

    with c_div:
        _render_column_divider()

    with c2:
        _subsection_header("Fidelity vs Robustness Trade-off")
        _render_fidelity_vs_robustness_tradeoff(rel_df)

    # ----------------------------------------------------------------
    # 4. Clinical Alignment
    # ----------------------------------------------------------------
    # Group clinical-alignment views so user can review all
    # clinically focused outputs in one place.
    _section_header("4. Clinical Alignment")
    _subsection_header("Clinical Alignment Summary")
    _render_clinical_summary_cards(method_df, feature_df, top_k)
    _render_horizontal_divider()
    st.markdown(
        "<div style='height: 20px;'></div>", 
        unsafe_allow_html=True,
    )

    c1, c_div, c2 = st.columns([1, 0.02, 1])

    with c1:
        _subsection_header("Clinical Coverage Analysis")
        _render_clinical_coverage_analysis(method_df, top_k)

    with c_div:
        _render_column_divider()

    with c2:
        _subsection_header("Clinical Feature Consistency")
        _render_clinical_feature_consistency_map(feature_df, top_n=12)

    _render_horizontal_divider()
    _subsection_header("Clinical Coverage Depth")
    _render_clinical_recall_curve(feature_df, max_k=max(15, top_k))

    # ----------------------------------------------------------------
    # 5. Agreement & Consistency
    # ----------------------------------------------------------------
    # Keep agreement section separate from robustness so cross-method
    # comparison is easy to locate.
    _section_header("5. Agreement & Consistency")
    
    c1, c_div, c2 = st.columns([1, 0.02, 1])

    with c1:
        _subsection_header("Global Agreement")
        _render_agreement_heatmap(agreement_df)

    with c_div:
        _render_column_divider()

    with c2:
        _subsection_header("Key Agreement Insights")
        _render_agreement_key_insights(agreement_df)

    # ----------------------------------------------------------------
    # 6. Feature Insights
    # ----------------------------------------------------------------
    # Focuses on which features appear most important and
    # how their attribution values vary across methods.
    _section_header("6. Feature Insights")
    
    c1, c_div, c2 = st.columns([1, 0.02, 1])

    with c1:
        _subsection_header(f"Top-{top_k} Feature Importance")
        _render_topk_importance_chart(
            feature_df,
            top_k=top_k,
            baseline_method="Captum_IG",
        )

    with c_div:
        _render_column_divider()

    with c2:
        _subsection_header("Feature Attribution Variability")
        _render_feature_variability(feature_df, top_n=top_k)

    # ----------------------------------------------------------------
    # 7. Instance-Level Analysis
    # ----------------------------------------------------------------
    # Present patient-level views after global summaries so reader
    # moves from higher-level comparisons to more detailed examples.
    _section_header("7. Instance-Level Analysis")
    _subsection_header("Patient-Level Explanation Stability")
    _render_instance_level_section(out)
    _render_horizontal_divider()
    _subsection_header("Feature Rank Shifts Across Perturbations")
    _render_rank_change_plot(out, top_k=top_k)

    # ----------------------------------------------------------------
    # 8. Risk & Failure Modes
    # ----------------------------------------------------------------
    # Keep cautionary views together so main limitations of
    # current run are visible in one place.
    _section_header("8. Risk & Failure Modes")
    _subsection_header(
        "Model Risk, Method Instability, and Disagreement"
    )
    _render_risk_failure_modes(model_auc, method_df, agreement_df)
    _caption(
        "These indicators highlight key limitations that may reduce "
        "confidence in the benchmark, including weak model signal, " 
        "unstable explanations, and conflicting feature rankings."
    )

    # ----------------------------------------------------------------
    # 9. Advanced Analysis
    # ----------------------------------------------------------------
    # Place detailed optional views after main findings so primary
    # narrative remains easy to follow.
    _section_header("9. Advanced Analysis")
    _subsection_header("Optional Expandable Details")
    _render_advanced_analysis(
        method_df,
        agreement_df,
        pairwise_df,
        feature_df,
        out,
        top_k,
    )

    # ----------------------------------------------------------------
    # 10. Evaluation Against Research Objectives
    # ----------------------------------------------------------------
    # Link dashboard outputs back to stated evaluation goals of
    # project.
    _section_header("10. Evaluation Against Research Objectives")
    _render_evaluation_against_objectives(rel_df, method_df, top_k)
    _caption(
        "This section links the benchmark outputs to the research "
        "question by showing how each method performs across "
        "fidelity, robustness, and clinical alignment."
    )

    # ----------------------------------------------------------------
    # 11. Key Takeaways
    # ----------------------------------------------------------------
    # End analytical sections with concise summary to make main
    # results easier to review.
    _section_header("11. Key Insights and Takeaways")
    _subsection_header("Summary")
    _render_key_takeaways(
        rel_df,
        model_auc,
        agreement_df,
        method_df,
        top_k,
    )
    _caption(
        "These cards summarise key findings from the current "
        "benchmark run for quick interpretation."
    )

    # ----------------------------------------------------------------
    # Downloads
    # ----------------------------------------------------------------
    # Keep export controls at end so users can review dashboard first
    # and then download generated outputs.
    _section_header("Download Outputs")

    # Build short text summary for export using key values already
    # shown in dashboard.
    summary_text = []
    summary_text.append(f"Model ROC-AUC: {model_auc:.3f}")

    # Only add recommended method line when reliability table contains
    # at least one row.
    if not rel_df.empty:
        best = rel_df.iloc[0]
        summary_text.append(
            f"Recommended Method: {best['Method']} "
            f"({best['Reliability_Score']:.3f})"
        )
    
    # Convert text and configuration objects to bytes because download
    # helpers expect byte-like content.
    summary_bytes = "\n".join(summary_text).encode("utf-8")
    config_bytes = (
        json.dumps(asdict(cfg), indent=2).encode("utf-8")
        if is_dataclass(cfg)
        else b"{}"
    )

    # Package main tabular and text outputs into ZIP bundle for
    # download.
    zip_bytes = _make_zip_bundle(
        {
            "method_summary.csv": method_df.to_csv(
                index=False
            ).encode("utf-8"),
            "agreement.csv": agreement_df.to_csv(
                index=False
            ).encode("utf-8"),
            "feature_importance.csv": feature_df.to_csv(
                index=False
            ).encode("utf-8"),
            "pairwise_stats.csv": pairwise_df.to_csv(
                index=False
            ).encode("utf-8"),
            "run_config.json": config_bytes,
            "interpretation_summary.txt": summary_bytes,
            "model_comparison_summary.csv": (
                model_comparison_df.to_csv(index=False).encode(
                    "utf-8"
                )
            ),
        }
    )

    # Build PDF report using current benchmark outputs so formatted
    # summary can also be downloaded directly from interface.
    pdf_bytes = make_pdf_report(
        title="Healthcare XAI Benchmark Report",
        cfg=cfg,
        model_auc=model_auc,
        method_df=method_df,
        agreement_df=agreement_df,
        pairwise_df=pairwise_df,
        fig_path=out.get("fig_path"),
        model_comparison_df=out.get("model_comparison_df"),
    )

    # Place two download buttons side by side to keep export controls
    # compact.
    c1, c2 = st.columns(2)

    with c1:
        # Provide one-click access to ZIP bundle containing core text
        # and table outputs.
        st.download_button(
            "Download CSV/TXT bundle (ZIP)",
            data=zip_bytes,
            file_name="xai_benchmark_outputs.zip",
            mime="application/zip",
            use_container_width=True,
        )

    with c2:
        # Provide one-click acesss to PDF summary report generated
        # from current run.
        st.download_button(
            "Download PDF Report",
            data=pdf_bytes,
            file_name="xai_benchmark_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# Preserve standard Python entry point so interface can still be
# launched directly when script is executed as main module.
if __name__ == "__main__":
    main()