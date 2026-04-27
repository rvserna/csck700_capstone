"""
MODULE 4: Benchmarking Engine

Purpose:
- Run the end to end benchmarking workflow for the project.
- Train the baseline model and generate explanation outputs on a
  shared cohort.
- Compare explanation methods using fidelity, robustness, agreement,
  and clinical-alignment metrics.
- Save structured result tables and support files reused by the
  Streamlit reporting interface.

Broader Context:
- data_preparation.py loads the model-ready feature table and applies
  the fitted preprocessing workflow.
- model_development.py defines the PyTorch mortality model and related
  prediction utilities.
- explanation_engine.py produces the global and instance-level
  attributions compared here.
- clinical_alignment.py provides the fixed reference feature set used
  in the clinical-alignment metrics.
- reporting_interface.py reads the tables and files written here for
  interactive display and download.

Additional Notes:
- This module coordinates the benchmark rather than introducing new
  explanation methods. 
- Its main role is to keep the workflow consistent so that each method
  is compared on the same prepared data, trained model, evaluation
  cohort, and output structure.
"""

import json
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

import data_preparation
import model_development
import explanation_engine
import clinical_alignment

# Suppress non-critical warnings so benchmark runs remain clean and
# readable.
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------
# Paths / Configuration
#---------------------------------------------------------------------
# Keep main benchmark input and output locations together so updates
# can be made in one place without searching through module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "joined_agg_dataset"
    / "model_features.parquet"
)

# Create shared output folder before any benchmark artefacts are
# written. This avoids write failures caused by missing
# destination folder.
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Define standard output file names reused by reporting interface.
# Keeping these names together makes saved output structure clearer.
OUT_MASTER_CSV = OUT_DIR / "comparative_module_metrics_master.csv"
OUT_METHOD_CSV = OUT_DIR / "method_summary.csv"
OUT_AGREEMENT_CSV = OUT_DIR / "explainer_agreement.csv"
OUT_FEATURE_CSV = OUT_DIR / "feature_importance.csv"
OUT_PAIRWISE_CSV = OUT_DIR / "pairwise_stats.csv"
OUT_MODEL_COMPARISON_CSV = OUT_DIR / "model_comparison_summary.csv"
OUT_SUMMARY_TXT = OUT_DIR / "interpretation_summary.txt"
OUT_RUN_CONFIG = OUT_DIR / "benchmark_run_config.json"

# Store shared benchmark defaults used across module.
# Values are reused by helper functions and run configuration
# settings.
RANDOM_STATE = 42
TOP_K = 10

# Store default LIME settings used when local surrogate explanations
# are created.
LIME_NUM_SAMPLES = 300
LIME_DISCRETIZE = True
LIME_REPEATS = 1
LIME_REPEATABILITY_REPEATS = 5
LIME_SAMPLE_AROUND_INSTANCE = False

# Store perturbation settings used in robustness checks.
# Keeping them fixed here makes noise and masking workflow consistent.
NOISE_STD = 0.20
N_NOISE_RUNS = 5

# Store masking settings used for feature removal style robustness
# tests. Values control how much information is hidden in each
# masking run.
MASK_FRAC = 0.40
MASK_RUNS = 3
MASK_VALUE = 0.0

# Store runtime controls that balance execution time and result
# coverage. Defaults shape size and duration of benchmark run.
N_DEBUG = 20
N_TRAIN_EPOCHS = 15
BACKGROUND_N = 20      # SHAP background sample size
PERM_REPEATS = 5       # permutation importance repetitions
BOOTSTRAP_N = 250      # bootstrap iterations for confidence intervals

# Apply default seed settings when module loads, which supports
# repeatable runs when same configuration is used again.
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# Reduce backend variation where possible during repeated runs.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------
# Run configuration
# --------------------------------------------------------------------
@dataclass
class RunConfig:
    """
    Store configurable values that define one benchmark run.

    Keeps main run settings together so engine,
    saved configuration export, and reporting interface use same
    values.
    """
    # Store feature table path used as main benchmark input.
    data_path: str = str(DATA_PATH)

    # Store train-test split used to create held-out evaluation set.
    test_size: float = 0.2

    # Store random state used for repeatable splits and repeated
    # sampling.
    random_state: int = RANDOM_STATE

    # Store runtime controls exposed in interface, which affect
    # sample size, training time, and run duration.
    n_debug: int = N_DEBUG
    n_train_epochs: int = N_TRAIN_EPOCHS

    # Store method toggles used by benchmark run. Captum IG is always
    # run here, while SHAP and LIME can be toggled on or off.
    run_shap: bool = True
    run_lime: bool = True

    # Store perturbation settings used during robustness testing.
    noise_std: float = NOISE_STD
    n_noise_runs: int = N_NOISE_RUNS
    mask_frac: float = MASK_FRAC
    mask_runs: int = MASK_RUNS
    mask_value: float = MASK_VALUE

    # Store baseline and repeat settings used in later fidelity
    # and robustness steps.
    background_n: int = BACKGROUND_N
    perm_repeats: int = PERM_REPEATS

    # Store main LIME settings used during benchmark run. A single 
    # repeat is used so robustness is evaluated externally rather
    # than being smoothed by averaging multiple local surrogate
    # samples inside one call.
    lime_num_samples: int = LIME_NUM_SAMPLES
    lime_discretize: bool = LIME_DISCRETIZE
    lime_repeats: int = LIME_REPEATS

    # Store separate repeat count used only for LIME repeatability
    # diagnostic. Keeps main benchmark lightweight while still
    # allowing a focused check of LIME's internal stochastic
    # variation.
    lime_repeatability_repeats: int = LIME_REPEATABILITY_REPEATS

    # Store whether LIME should sample around each explained instance.
    # Keeping this explicit makes it easier to track how local
    # sampling strategy may affect repeated explanation stability.
    lime_sample_around_instance: bool = LIME_SAMPLE_AROUND_INSTANCE

    # Store number of cross-validation folds used during comparative
    # tuning so parameter evaluation is based on more than one split.
    cv_folds: int = 2

    # Control whether weighted sampling is used during MLP training
    # so class imbalance can be handled more directly during minibatch
    # contruction.
    use_weighted_sampler: bool = True

    # Store shared Top-K value used by overlap metrics.
    top_k: int = TOP_K

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------
# Convert run configuration to readable JSON before saving it.
# Fallback keeps Path-like and datetime values serialisable if
# needed.
def json_dumps_safe(obj: dict) -> str:
    """
    Convert configuration dictionary to readable JSON text.

    Used when benchmark settings are exported so
    saved run configuration is easy to inspect outside script.
    """
    return json.dumps(obj, indent=2, default=str)

def paired_rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute paired rank biserial effect size for matched vectors.

    Summarises direction and relative strength of
    difference between paired values after zero differences are
    removed, complementing Wilcoxon signed-rank test by adding
    interpretable effect size estimate rather than relying on
    statistical significance alone.
    """
    # Convert both inputs to numeric arrays so matched comparison is
    # applied to one consistent representation.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Compute paired difference because effect size is based on how
    # first vector compares with second at each matched position.
    diff = x - y

    # Remove missing and zero differences so statistic is based
    # only on informative paired comparisons that contribute
    # directional evidence.
    diff = diff[~np.isnan(diff)]
    diff = diff[diff != 0]

    # Return NaN when no usable paired differences remain after
    # cleaning.
    if diff.size == 0:
        return np.nan

    # Rank absolute paired differences so larger paired departures
    # receive greater influence regardless of direction.
    abs_diff = np.abs(diff)
    order = abs_diff.argsort()
    ranks = np.empty_like(abs_diff, dtype=float)
    ranks[order] = np.arange(1, len(abs_diff) + 1, dtype=float)

    # Separate positive and negative rank mass separately so final
    # score reflects which side dominate across matched pairs.
    pos = float(ranks[diff > 0].sum())
    neg = float(ranks[diff < 0].sum())
    denom = pos + neg

    # Stop early if denominator is unusable after ranking.
    if denom == 0:
        return np.nan

    # Scaled signed rank imbalance to -1 to 1 range so later
    # interpretation stays consistent across comparisons.
    return float((pos - neg) / denom)

def interpret_significance(p_adj: float) -> str:
    """
    Map adjusted p-value to concise qualitative significance label.

    Converts numeric significance results into short, readable
    descriptions so pairwise comparison outputs can be interpreted
    without inspecting raw p-values directly.
    """
    # Return placeholder when adjusted p-value is unavailable so
    # missing results are not misinterpreted as non-significant
    # findings.
    if pd.isna(p_adj):
        return "not tested"

    # Apply ordered thresholds so interpretaiton remains consistent
    # across all pairwise comparisons in benchmark outputs.
    if p_adj < 0.001:
        return "very strong evidence of a difference"
    if p_adj < 0.01:
        return "strong evidence of a difference"
    if p_adj < 0.05:
        return "evidence of a difference"

    # Use neutral description when result does not meet
    # significance threshold after adjustmennt.
    return "no statistically reliable difference detected"

def interpret_rank_biserial(effect: float) -> str:
    """
    Convert rank biserial effect size into short qualitative label.

    Returned label is used in exported summaries and dashboard text so
    size of paired difference is easier to read than inspecting
    raw statistic alone.
    """
    # Return explicit placeholder when effect size is unavailable so
    # missing results are not mistaken for negligible ones.
    if pd.isna(effect):
        return "not available"

    # Use absolute magnitude here because direction is described
    # separately in pairwise implication text.
    a = abs(float(effect))

    # Apply ordered thresholds so effect size wording remains
    # consistent across benchmark outputs.
    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "small"
    if a < 0.50:
        return "moderate"
    return "large"

def pairwise_implication_text(
    comparison: str,
    p_adj: float,
    effect: float,
) -> str:
    """
    Return short plain language summary for one pairwise comparison
    result.

    Combines adjusted significance and paired effect size
    interpretation so eported pairwise table is easier to read without
    requiring user to interpret each numeric field separately.
    """
    # Return short fallback sentence when either adjusted p-value or
    # effect size is unavailable for comparison.
    if pd.isna(p_adj) or pd.isna(effect):
        return (
            "Pairwise interpretation unavailable for this comparison."
        )

    # Convert numeric results to concise labels before building
    # final sentence.
    significance = interpret_significance(p_adj)
    effect_label = interpret_rank_biserial(effect)

    # Describe which method tends to assign larger normalised
    # importance values.
    direction = (
        "the first named method tends to assign larger normalised "
        "importance values"
        if effect > 0
        else (
            "the second named method tends to assign larger "
            "normalised importance values"
        )
    )

    # Keep significance results and effect size in same sentence so
    # difference is not described only in binary terms.
    if p_adj < 0.05:
        return (
            f"{comparison}: {significance}; {direction} "
            f"with a {effect_label} paired effect."
        )

    # Highlight larger effect size even when adjusted p-value is not
    # statistically significant so magnitude is still visible to reader.
    if effect_label in {"moderate", "large"}:
        return (
            f"{comparison}: {significance}; the observed directional "
            f"difference was {effect_label} in magnitude, but it was "
            "not statistically reliable in this run."
        )

    # Provide cautious interpretation when both statistical evidence and
    # effect size are limited.
    return (
        f"{comparison}: {significance}; any observed difference was "
        f"{effect_label} in magnitude and is best treated as "
        "inconclusive in this run."
    )

# Convert feature importance series to an L1-normalised form.
# This puts compared vectors on same overall scale.
def l1_normalize(s: pd.Series) -> pd.Series:
    """
    Scale feature importance series so its absolute values sum to
    one.

    Used before pairwise comparisons so methods with different
    raw magnitudes can still be compared on common scale.
    """
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)

    # Use sum of absolute values so positive and negative entries
    # do not cancel each other out during scaling.
    denom = float(np.abs(s).sum())
    return s if denom == 0 else s / denom

# Compute Spearman correlation for two arrays.
# Checks below prevent invalid comparisons from raising misleading
# results.
def spearman_safe(a, b) -> float:
    """
    Compute Spearman correlation with basic validity checks.

    Missing, mismatched, or constant inputs return NaN so later
    summary steps can handle them explicitly.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Return NaN when vectors cannot be compared meaningfully.
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return np.nan
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
 
    # Extract correlation value returned by scipy.
    stat = spearmanr(a, b).correlation
    return float(stat) if stat is not None else np.nan

def bootstrap_ci(
    values, 
    n_boot: int = BOOTSTRAP_N, 
    ci: float = 0.95, 
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Estimate mean and confidence interval from repeated bootstrap
    samples.

    Resamples observed values with replacement to approximate
    sampling distribution of mean, returning a point estimate and
    corresponding confidence interval bounds.
    """
    # Convert inputs to numeric array and removes missing values so
    # bootstrap operates on a clean set of observations.
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]

    # Return placeholders when no usable values remain after cleaning
    # so downstream reporting can handle missing results explicitly.
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
 
    # Create reproducible random generator so repeated runs yield
    # consistent bootstram estimates.
    rng = np.random.default_rng(seed)
    boots = np.empty(int(n_boot), dtype=float)

    # Resample with replacement so each bootstrap draw matches size
    # of original sample and captures sampling variability.
    for i in range (int(n_boot)):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boots[i] = float(np.nanmean(sample))

    # Compute confidence interval bounds from bootstrap distribution.
    alpha = 1.0 - float(ci)
    return (
        float(np.nanmean(vals)),
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
    )

def _global_from_instance_abs(instance_abs: np.ndarray) -> np.ndarray:
    """
    Convert instance-level attribution values into one global feature
    vector.

    Mean is taken across instances so one value remains per feature.
    """
    return np.nanmean(np.asarray(instance_abs, dtype=float), axis=0)

def _make_ig_baseline(
    X_train: np.ndarray,
    mode: str = "median",
) -> np.ndarray:
    """
    Create reference vector used by Integrated Gradients.

    Same baseline can then be reused across repeated IG calls.

    Supported modes:
    - 'zero': one zero value per feature
    - 'mean': feature-wise mean of training matrix
    - 'median': feature-wise median of training matrix
    """
    X_train = np.asarray(X_train, dtype=float)

    # Build one baseline value per feature using requested summary
    # rule.
    if mode == "zero":
        return np.zeros(X_train.shape[1], dtype=np.float32)
    if mode == "mean":
        return np.nanmean(X_train, axis=0).astype(np.float32)
    if mode == "median":
        return np.nanmedian(X_train, axis=0).astype(np.float32)
    raise ValueError(f"Unknown IG baseline mode: {mode}")

def topk_overlap(a: pd.Series, b: pd.Series, k: int) -> float:
    """
    Compute proportion of shared feature names in two Top-K lists.

    Global Top-K overlap score is proportion of shared features inside
    selected Top-K sets.
    """
    a_top = set(
        pd.to_numeric(
            a,
            errors="coerce",
        ).fillna(0.0).nlargest(k).index
    )
    b_top = set(
        pd.to_numeric(
            b,
            errors="coerce",
        ).fillna(0.0).nlargest(k).index
    )
    return float(len(a_top & b_top) / float(k))

def _instance_topk_overlap(
    base_row: np.ndarray,
    pert_row: np.ndarray,
    k: int,
) -> float:
    """
    Compute instance-level Top-K overlap for one original and
    perturbed row.

    Creates stricter instance level stability score than global
    summary alone.
    """
    # Rank features by absolute attribution so comparison focuses on
    # strength rather than sign of each attribution.
    base_idx = set(np.argsort(-np.abs(base_row))[:k])
    pert_idx = set(np.argsort(-np.abs(pert_row))[:k])

    return float(len(base_idx & pert_idx) / float(k))

def _mean_instance_topk_overlap(
    base_instance_abs: np.ndarray | None,
    pert_instance_abs: np.ndarray | None,
    k: int,  
) -> float:
    """
    Compute mean instance-level Top-K overlap across all compared
    rows.

    Summarises how stable top-ranked features remain after
    perturbation.
    """
    # Return NaN when either attribution table is unavailable.
    if base_instance_abs is None or pert_instance_abs is None:
        return np.nan

    base = np.asarray(base_instance_abs, dtype=float)
    pert = np.asarray(pert_instance_abs, dtype=float)

    # Stop when two arrays do not share same 2D structure.
    if base.ndim != 2 or pert.ndim != 2 or base.shape != pert.shape:
        return np.nan

    # Score each row pair separately, then average resulting
    # overlaps.
    vals = [
        _instance_topk_overlap(
            base[i],
            pert[i],
            k,
        )
        for i in range(base.shape[0])
    ]
    return float(np.nanmean(vals)) if len(vals) else np.nan

def _mean_instance_spearman(
    base_instance_abs: np.ndarray | None,
    pert_instance_abs: np.ndarray | None,
) -> float:
    """
    Compute mean instance-level Spearman correlation across all
    compared rows.

    Provides rank-based stability summary at row level.
    """
    # Return NaN when either attribution table is missing.
    if base_instance_abs is None or pert_instance_abs is None:
        return np.nan
  
    base = np.asarray(base_instance_abs, dtype=float)
    pert = np.asarray(pert_instance_abs, dtype=float)

    # Stop when two arrays are not aligned by row and column.
    if base.ndim != 2 or pert.ndim != 2 or base.shape != pert.shape:
        return np.nan
  
    # Compute one correlation per row pair, then remove invalid
    # results before taking overall mean.
    vals = [
        spearman_safe(base[i], pert[i])
        for i in range(base.shape[0])
    ]
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    return (
        float(np.nanmean(vals))
        if len(vals)
        else np.nan
    )

def apply_random_feature_mask(
    X: np.ndarray, 
    mask_frac: float,
    mask_value: float = 0.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Randomly replace fraction of features in each row with one
    mask value.

    Used for scaled data, where values near 0 represent a 
    feature value close to training mean after preprocessing.
    """
    # Convert input to numeric NumPy array so masking is applied
    # consistently even if input arrives in another form.
    X = np.asarray(X, dtype=float)

    # Confirm input has expected row by column structure before
    # selecting features to mask.
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")
 
    # Restrict masking fraction to a valid proportion so number
    # of masked columns can be calculated safely.
    if not 0.0 <= float(mask_frac) <= 1.0:
        raise ValueError("mask_frac must be between 0 and 1.")

    # Work on a copy so original array remains available for any later
    # comparison or reuse in benchmark.
    X_masked = X.copy()
    n_rows, n_features = X_masked.shape

    # Convert requested fraction into column count and ensure that at
    # least one feature is masked in each row.
    n_to_mask = max(1, int(round(float(mask_frac) * n_features)))
    rng = np.random.default_rng(int(random_state))

    # Mask different random subset of features for each row so test
    # reflects row level variation rather than one shared mask
    # pattern.
    for i in range(n_rows):
        cols = rng.choice(n_features, size=n_to_mask, replace=False)
        X_masked[i, cols] = float(mask_value)
    return X_masked

def apply_featurewise_noise_raw(
    X_df: pd.DataFrame,
    train_df: pd.DataFrame,
    noise_scale: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Add feature-wise Gaussian noise to raw input data.

    Noise is scaled by standard deviation set by training for each
    feature so perturbation size stays proportional to feature's
    typical range.
    """
    # Copy and coerce evaluation data to numeric form so added noise
    # is applied to clean numeric matrix.
    X = X_df.copy().apply(pd.to_numeric, errors="coerce")

    # Align training reference columns to evaluation frame so
    # feature wise standard deviations match same column order.
    train_ref = (
        train_df.reindex(columns=X.columns)
        .apply(pd.to_numeric, errors="coerce")
    )

    # Use dedicated generator so repeated runs stay reproducible when
    # same random state is supplied.
    rng = np.random.default_rng(int(random_state))

    # Estimate one standard deviation per feature and replace unusable
    # values so noise can still be generated safely.
    std = (
        train_ref.std(axis=0, skipna=True)
        .replace(0, np.nan)
        .fillna(1.0)
    )

    # Generate zero centred Gaussian noise and scale it feature by
    # feature so higher variance columns receive proportionally
    # larger perturbations.
    noise = rng.normal(
        loc=0.0, 
        scale=float(noise_scale), 
        size=X.shape,
    ) * std.to_numpy(dtype=float).reshape(1, -1)

    # Add generated noise to raw values and return result in
    # original DataFrame layout.
    arr = X.to_numpy(dtype=float) + noise
    return pd.DataFrame(arr, columns=X.columns, index=X.index)

def apply_random_feature_mask_raw(
    X_df: pd.DataFrame,
    mask_frac = float,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Randomly mask fraction of raw features in each row with missing
    values.

    Inserts NaNs rather than fixed numeric replacements so
    exisitng preprocessing pipeline can handle masked values
    consistently. Fitted imputer and scaler later process these
    missing entries.
    """
    # Check requested masking fraction first so invalid inputs are
    # caught before any work is done.
    if not 0.0 <= float(mask_frac) <= 1.0:
        raise ValueError("mask_frac must be between 0 and 1.")
    
    # Copy and coerce raw inputs to numeric form so missing values can
    # be inserted into consistent array representation.
    X = X_df.copy().apply(pd.to_numeric, errors="coerce")
    n_rows, n_features = X.shape

    # Convert masking fraction into feature count and ensure that each
    # row has at least one masked value.
    n_to_mask = max(1, int(round(float(mask_frac) * n_features)))

    # Create random generator once so same seed produces same mask
    # pattern across repeated runs.
    rng = np.random.default_rng(int(random_state))
    arr = X.to_numpy(dtype=float)

    # Apply row-specific random mask so missing value test reflects
    # different feature omissions across cohort.
    for i in range(n_rows):
        cols = rng.choice(n_features, size=n_to_mask, replace=False)
        arr[i, cols] = np.nan

    return pd.DataFrame(arr, columns=X.columns, index=X.index)

def get_clinically_validated_set() -> set[str]:
    """
    Return clinical reference features as set of strings.

    Converting values to strings keeps later membership checks
    consistent with feature names used throughout exported tables.
    """
    return {
        str(v)
        for v in clinical_alignment.get_clinically_validated_features()
    }

def _rank_desc(s: pd.Series) -> pd.Series:
    """
    Rank series from highest to lowest value to create descending
    ranks for exported importance column.

    Higher importance values receive smaller rank numbers. Minimum 
    rank method is used so tied values receive same highest applicable
    rank rather than being split across multiple positions.
    """
    return s.rank(method="min", ascending=False)


# --------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------
def compute_permutation_importance(
    model, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    feature_names: List[str],
    perm_repeats: int = PERM_REPEATS,
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """
    Compute permutation importance on held-out data.

    Returned values are converted to absolute magnitudes so later
    fidelity comparison focuses on relative feature importance size.
    """
    # Wrap PyTorch model in project helper so scikit-learn's
    # permutation importance function can call it in standard way.
    wrapped = model_development.TorchWrapper(model)

    # Recalculate model performance after repeated feature shuffling
    # so average performance drop can be used as inportance signal.
    perm = permutation_importance(
        wrapped,
        X_test,
        y_test,
        n_repeats=int(perm_repeats),
        random_state=int(random_state),
        scoring="roc_auc",
    )

    # Return mean absolute importance per feature using supplied
    # feature names as index.
    return pd.Series(
        np.abs(perm.importances_mean),
        index=feature_names,
    )

def compute_grad_input_baseline(
    model, 
    X_ref: np.ndarray, 
    feature_names: List[str]
) -> pd.Series:
    """
    Compute gradient-based reference importance vector.

    Averages absolute gradients across reference cohort and
    scales them by observed feature spread in that cohort.
    """
    # Switch model to evaluation mode so baseline is generated from
    # stable forward pass configuration.
    model.eval()

    # Build tensor that tracks gradients because baseline depends on
    # how model output changes with respect to each input feature.
    X_t = torch.tensor(X_ref, dtype=torch.float32, requires_grad=True)
    logits = model(X_t)

    # Backpropagate from mean logit so one gradient value is produced
    # for every feature in every reference row.
    logits.mean().backward()
    grads = X_t.grad.detach().cpu().numpy()

    # Average absolute gradients across rows so output becomes one
    # feature level importance vector.
    grad_mag = np.nanmean(np.abs(grads), axis=0)
    x_std = np.nanstd(X_ref, axis=0)

    # Scale gradient magnitudes by feature variability so baseline
    # reflect both sensitivity and observed spread.
    scores = grad_mag * x_std
    return pd.Series(np.abs(scores), index=feature_names)

# -----------------------------------------------
# Explainer Dispatch / Robustness
# -----------------------------------------------
# Route method name to matching explanation function.
# Keeps later benchmark code method agnostic.
def _compute_explainer(
    explainer_name: str,
    model,
    X_train: np.ndarray,
    X_eval: np.ndarray,
    feature_names: List[str],
    *,
    random_state: int,
    background_n: int,
    lime_num_samples: int,
    lime_discretize: bool,
    lime_num_features: Optional[int],
    lime_repeats: int,
    lime_sample_around_instance: bool,
    ig_baseline_vec: np.ndarray,
) -> Tuple[pd.Series, np.ndarray]:
    """
    Run requested explainer and return global and instance-level
    outputs.

    Returned structure is kept consistent across methods so later
    benchmarking steps can compare methods without separate code
    paths.
    """
    # Route Integrated Gradients requests to project explanation
    # helper and pass shared baseline vector used for benchmark run.
    if explainer_name == "ig":
        return explanation_engine.compute_integrated_gradients(
            model,
            X_eval,
            feature_names,
            baseline=ig_baseline_vec,
        )
  
    # Route SHAP requests through shared explanation interface so
    # training background and evaluation cohort stay aligned.
    if explainer_name == "shap":
        return explanation_engine.compute_shap_gradients(
            model,
            X_train,
            X_eval,
            feature_names,
            random_state=random_state,
            background_n=int(background_n),
        )

    # Route LIME requests through same dispatcher so its outputs
    # same contract as other explainers.  
    if explainer_name == "lime":
        return explanation_engine.compute_lime(
            model,
            X_train,
            X_eval,
            feature_names,
            random_state=random_state,
            num_samples=int(lime_num_samples),
            discretize_continuous=bool(lime_discretize),
            num_features_to_explain=lime_num_features,
            n_repeats=int(lime_repeats),
            sample_around_instance=bool(lime_sample_around_instance),
        )
  
    # Raise clear error when unknown label is supplied so invalid
    # method names are caught immediately.
    raise ValueError(f"Unknown explainer_name: {explainer_name}")

def _evaluate_robustness(
    explainer_name: str,
    model,
    X_train_scaled: np.ndarray,
    X_base_raw: pd.DataFrame,
    feature_names: List[str],
    base_instance_abs: np.ndarray | None,
    *,
    perturbation_fn,
    runs: int,
    random_state: int,
    top_k: int,
    background_n: int,
    lime_num_samples: int,
    lime_discretize: bool,
    lime_num_features: int,
    lime_repeats: int,
    lime_sample_around_instance: bool,
    ig_baseline_vec: np.ndarray,
    imputer,
    scaler,
    use_scaled_masking: bool = False,
    mask_frac: float = MASK_FRAC,
    mask_value: float = MASK_VALUE,
) -> Dict[str, Any]:
    """
    Evaluate one explainer under repeated perturbations.

    Returned dictionary includes bootstrap confidence intervals for
    instance level stability and one representative perturbed results
    array.
    """
    # Stop early when no base attributions are available because
    # stability comparison depends on valid original reference array.
    if base_instance_abs is None:
        return {
            "spearman_ci": (np.nan, np.nan, np.nan),
            "topk_ci": (np.nan, np.nan, np.nan),
            "representative_instance_abs": None,
        }

    # Convert base attributions to numeric array so repeated
    # similarity calculations use one consistent format.
    base = np.asarray(base_instance_abs, dtype=float)
    if base.ndim != 2:
        return {
            "spearman_ci": (np.nan, np.nan, np.nan),
            "topk_ci": (np.nan, np.nan, np.nan),
            "representative_instance_abs": None,            
        }

    # Collect one score per perturbation run so bootstrap intervals
    # can be estimated after loop finishes.
    spearman_scores: List[float] = []
    topk_scores: List[float] = []
    run_arrays: List[np.ndarray] = []

    # Transform base raw cohort with fitted preprocessing objects so
    # scaled reference version is available when needed.
    X_base_scaled = data_preparation.transform_with_preprocessor(
        X_base_raw.reindex(columns=feature_names),
        imputer,
        scaler,
    )

    # Repeat perturbation and explanation process several times so
    # stability estimate reflects more than one random draw.
    for run_idx in range(int(runs)):
        seed = int(random_state) + run_idx

        # Apply masking directly in scaled space when that mode is
        # requested so perturbation stays in already transformed
        # feature space.
        if use_scaled_masking:
            X_pert = apply_random_feature_mask(
                X_base_scaled,
                mask_frac=float(mask_frac),
                mask_value=float(mask_value),
                random_state=seed,
            )
            X_train_ref = X_train_scaled
        else:
            # Apply selected raw space perturbation to copy of base
            # cohort so each run starts from same original data.
            X_pert_raw = perturbation_fn(X_base_raw.copy(), seed)
            X_pert_raw = X_pert_raw.reindex(columns=feature_names)

            # Reuse fitted preprocessing workflow so perturbed raw
            # inputs are transformed in same way as original data.
            X_pert_scaled = (
                data_preparation.transform_with_preprocessor(
                    X_pert_raw,
                    imputer,
                    scaler,
                )
            )

            # Apply scaled masking after raw space missing value
            # insertion when perturbation function specifically
            # uses raw masking.
            if (
                perturbation_fn.__name__
                == "apply_random_feature_mask_raw"
            ):
                X_pert = apply_random_feature_mask(
                    X_pert_scaled,
                    mask_frac=float(mask_frac),
                    mask_value=0.0,
                    random_state=seed,
                )
            else:
                X_pert = X_pert_scaled

            # Keep explainer training reference fixed so each run
            # is compared against same training background.
            X_train_ref = X_train_scaled

        # Convert LIME sample count once per run so explainer call
        # reads same integer value regardless of input type.
        lime_samples = int(lime_num_samples)

        # Regenerate explanations on pertubed cohort using shared
        # dispatcher so all methods follow same calling pattern.
        _, pert_inst = _compute_explainer(
            explainer_name,
            model,
            X_train_ref,
            X_pert,
            feature_names,
            random_state=seed,
            background_n=background_n,
            lime_num_samples=lime_samples,
            lime_discretize=lime_discretize,
            ig_baseline_vec=ig_baseline_vec,
            lime_num_features=lime_num_features,
            lime_repeats=lime_repeats,
            lime_sample_around_instance=lime_sample_around_instance,
        )

        # Record one instance level similarity score per run for both
        # stability measures used later in summary outputs.
        spearman_scores.append(
            _mean_instance_spearman(base, pert_inst)
        )
        topk_scores.append(
            _mean_instance_topk_overlap(
                base,
                pert_inst,
                top_k,
            )
        )
        run_arrays.append(pert_inst)

    # Choose one representative pertubed result for dashboard display
    # by selecting run whose Top-K score is closest to median.
    rep_idx = 0
    topk_clean = np.asarray(topk_scores, dtype=float)
    valid_idx = np.where(~np.isnan(topk_clean))[0]
    if len(valid_idx):
        valid_scores = topk_clean[valid_idx]
        median_score = np.nanmedian(valid_scores)
        rep_idx = int(
            valid_idx[np.argmin(np.abs(valid_scores - median_score))]
        )

    # Return bootstrap intervals and representative perturbed array so
    # both summary metrics and dashboard visuals can reuse same
    # results.
    return {
        "spearman_ci": bootstrap_ci(
            spearman_scores,
            seed=int(random_state) + 2000,
        ),
        "topk_ci": bootstrap_ci(
            topk_scores,
            seed=int(random_state) + 3000,
        ),
        "representative_instance_abs": (
            run_arrays[rep_idx] if run_arrays else None
        ),
    }

def bootstrap_fidelity_ci(
    instance_abs: Optional[np.ndarray],
    baseline_series: pd.Series,
    feature_names: List[str],
    n_boot: int = BOOTSTRAP_N,
    ci: float = 0.95,
    seed: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    """
    Estimate fidelity mean and confidence interval by bootstrap
    resampling.

    Returned tuple contains mean score, lower confidence bound, and 
    upper confidence bound in that order.
    """
    # Stop early when no instance level attributions are available
    # because bootstrap procedure depends on valid 2D attribution
    # array.
    if instance_abs is None:
        return (np.nan, np.nan, np.nan)

    # Convert input to numeric array and confirm it contains rows
    # before attempting any resampling.
    arr = np.asarray(instance_abs, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return (np.nan, np.nan, np.nan)

    n = arr.shape[0]

    # Create random generator once so bootstrap samples are repeatable
    # when same seed is used again.
    rng = np.random.default_rng(seed)

    # Align baseline series to benchmark feature order before
    # comparing it with resampled global importance vectors.
    base = baseline_series.reindex(feature_names).fillna(0.0).values

    scores = []
    for _ in range(int(n_boot)):
        # Resample explanation rows with replacement so each bootstrap
        # draw reflects another plausible sample from explanation
        # cohort.
        idx = rng.integers(0, n, size=n)
        g = _global_from_instance_abs(arr[idx, :]).reshape(-1)

        # Convert both vectors to absolute magnitudes because fidelity
        # view compares feature importance size rather than direction.
        g = np.abs(g)
        base_norm = np.abs(base)

        # Normalise both vectors when possible so rank comparison is
        # not driven by overall scale differences.
        g_sum = np.sum(g)
        b_sum = np.sum(base_norm)
        if g_sum > 0:
            g = g / g_sum
        if b_sum > 0:
            base_norm = base_norm / b_sum
  
        # Store one Spearman score per resample so final interval can
        # be estimated from full bootstrap distribution.
        scores.append(spearman_safe(g, base_norm))

    return bootstrap_ci(scores, n_boot=n_boot, ci=ci, seed=seed + 1)


# --------------------------------------------------------------------
# Output Building
# --------------------------------------------------------------------
def _method_rows(
    *,
    top_k: int,
    ig_series: pd.Series,
    shap_series: pd.Series | None,
    lime_series: pd.Series | None,
    ig_noise: Dict[str, Any],
    shap_noise: Dict[str, Any],
    lime_noise: Dict[str, Any],
    ig_mask: Dict[str, Any],
    shap_mask: Dict[str, Any],
    lime_mask: Dict[str, Any],
    ig_fid_perm_ci,
    shap_fid_perm_ci,
    lime_fid_perm_ci,
    ig_fid_grad_ci,
    shap_fid_grad_ci,
    lime_fid_grad_ci,
) -> pd.DataFrame:
    """
    Build one summary row per enabled explainer.

    Combines main robustness, fidelity, and clinical alignment
    outputs into method level table shown in dashboard.
    """
    # Load shared clinical reference set once so it can be reused
    # for each method.
    clin_set = get_clinically_validated_set()

    # Derive clinical alignment values from one explainer importance
    # series. Missing methods return empty or unavailable values so
    # table still builds cleanly.
    def clinical_parts(series: pd.Series | None):
        if series is None:
            return [], np.nan, np.nan, np.nan
  
        # Use absolute importance because clinical comparison
        # focuses on strength, rather than whether attribution
        # direction is postive or negative.
        series_norm = series.abs()

        # Collect overlapping features and summary scores used
        # later in the row.
        feats = clinical_alignment.clinical_overlap_features(
            series_norm,
            clin_set,
            top_k,
        )
        weighted = clinical_alignment.weighted_clinical_overlap(
            series_norm,
            clin_set,
            top_k,
        )
        precision, recall = (
            clinical_alignment.clinical_precision_recall(
                series_norm,
                clin_set,
                top_k,
            )
        )
        return feats, weighted, precision, recall

    # Start with Captum IG because it is always expected in benchmark.
    # Optional methods are appended only when their outputs are
    # available.
    method_specs = [
        (
            "Captum IG", 
            ig_series, 
            ig_noise,
            ig_mask,
            ig_fid_perm_ci, 
            ig_fid_grad_ci,
        ),
    ]

    # Add SHAP only when that explainer was enabled and produced
    # results.
    if shap_series is not None:
        method_specs.append(
            (
                "SHAP", 
                shap_series,
                shap_noise,
                shap_mask,
                shap_fid_perm_ci, 
                shap_fid_grad_ci,
            )
        )

    # Add LIME only when that explainer was enabled and produced
    # results.
    if lime_series is not None:
        method_specs.append(
            (
                "LIME", 
                lime_series,
                lime_noise,
                lime_mask,
                lime_fid_perm_ci, 
                lime_fid_grad_ci,
            )
        )

    # Build one exported summary row for each enabled method.
    rows = []
    for (
        method,
        series,
        noise_res,
        mask_res,
        fid_perm_ci,
        fid_grad_ci,
    ) in method_specs:
        # Reuse helper so all methods follow same clinical alignment
        # logic.
        clin_feats, weighted, precision, recall = clinical_parts(
            series
        )
        rows.append(
            {
                "Method": method,

                # Store noise and masking stability values with their
                # confidence bounds so reporting layer can show both
                # central value and uncertainty.
                "Stability_Spearman_Noise_Mean": (
                    noise_res["spearman_ci"][0]
                ),
                "Stability_Spearman_Noise_CI_Low": (
                    noise_res["spearman_ci"][1]
                ),
                "Stability_Spearman_Noise_CI_High": (
                    noise_res["spearman_ci"][2]
                ),
                "Stability_Spearman_Masking_Mean": (
                    mask_res["spearman_ci"][0]
                ),
                "Stability_Spearman_Masking_CI_Low": (
                    mask_res["spearman_ci"][1]
                ),
                "Stability_Spearman_Masking_CI_High": (
                    mask_res["spearman_ci"][2]
                ),
                f"Stability_Top{top_k}_Noise_Overlap_Mean": (
                    noise_res["topk_ci"][0]
                ),
                f"Stability_Top{top_k}_Noise_Overlap_CI_Low": (
                    noise_res["topk_ci"][1]
                ),
                f"Stability_Top{top_k}_Noise_Overlap_CI_High": (
                    noise_res["topk_ci"][2]
                ),
                f"Stability_Top{top_k}_Masking_Overlap_Mean": (
                    mask_res["topk_ci"][0]
                ),
                f"Stability_Top{top_k}_Masking_Overlap_CI_Low": (
                    mask_res["topk_ci"][1]
                ),
                f"Stability_Top{top_k}_Masking_Overlap_CI_High": (
                    mask_res["topk_ci"][2]
                ),

                # Save fidelity scores against both reference baselines so
                # dashboard can compare each explainer to same benchmark
                # points.
                "Fidelity_vs_Permutation_Spearman_Mean": (
                    fid_perm_ci[0]
                ),
                "Fidelity_vs_Permutation_Spearman_CI_Low": (
                    fid_perm_ci[1]
                ),
                "Fidelity_vs_Permutation_Spearman_CI_High": (
                    fid_perm_ci[2]
                ),
                "Fidelity_vs_GradInput_Spearman_Mean": (
                    fid_grad_ci[0]
                ),
                "Fidelity_vs_GradInput_Spearman_CI_Low": (
                    fid_grad_ci[1]
                ),
                "Fidelity_vs_GradInput_Spearman_CI_High": (
                    fid_grad_ci[2]
                ), 

                # Store both numeric clinical metrics and overlapping
                # feature names so summary table can support quick review
                # and more detailed interpretation.
                f"Top{top_k}_Clinical_Overlap": len(clin_feats),
                "Weighted_Clinical_Overlap": weighted,
                "Clinical_Precision": precision,
                "Clinical_Recall": recall,
                f"Clinical_Top{top_k}_Overlap_Features": (
                    ", ".join(clin_feats) if clin_feats else ""
                ),
            }
        )

    # Convert collected rows into method level export expected
    # downstream.
    return pd.DataFrame(rows)

def build_outputs(
    *,
    model_importance: pd.Series,
    grad_input_baseline: pd.Series,
    ig_series: pd.Series,
    shap_series: pd.Series | None,
    lime_series: pd.Series | None,
    ig_noise: Dict[str, Any],
    shap_noise: Dict[str, Any],
    lime_noise: Dict[str, Any],
    ig_mask: Dict[str, Any],
    shap_mask: Dict[str, Any],
    lime_mask: Dict[str, Any],
    ig_fid_perm_ci,
    shap_fid_perm_ci,
    lime_fid_perm_ci,
    ig_fid_grad_ci,
    shap_fid_grad_ci,
    lime_fid_grad_ci,
    top_k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Assemble main benchmark tables returned by engine.

    Returns method summary, explainer agreement, feature importance,
    and pairwise statistical comparison tables in one consistent
    format.
    """
    # Build method level summary first because it gathers main results
    # that are reused throughout reporting interface.
    method_df = _method_rows(
        top_k=top_k,
        ig_series=ig_series,
        shap_series=shap_series,
        lime_series=lime_series,
        ig_noise=ig_noise,
        shap_noise=shap_noise,
        lime_noise=lime_noise,
        ig_mask=ig_mask,
        shap_mask=shap_mask,
        lime_mask=lime_mask,
        ig_fid_perm_ci=ig_fid_perm_ci,
        shap_fid_perm_ci=shap_fid_perm_ci,
        lime_fid_perm_ci=lime_fid_perm_ci,
        ig_fid_grad_ci=ig_fid_grad_ci,
        shap_fid_grad_ci=shap_fid_grad_ci,
        lime_fid_grad_ci=lime_fid_grad_ci,
    )

    # Compute method to method rank correlation scores only when
    # necessary series are present, which keeps optional explainers
    # from breaking export.
    shap_vs_captum_rho = (
        spearman_safe(shap_series.values, ig_series.values)
        if shap_series is not None
        else np.nan
    )
    lime_vs_captum_rho = (
        spearman_safe(lime_series.values, ig_series.values)
        if lime_series is not None
        else np.nan
    )
    shap_vs_lime_rho = (
        spearman_safe(
            shap_series.values,
            lime_series.values,
        )
        if shap_series is not None and lime_series is not None
        else np.nan
    )

    # Compute Top-K overlap values alongside Spearman agreement
    # scores. Gives dashboard both rank based and feature set
    # comparison.
    shap_vs_captum_topk = (
        topk_overlap(
            shap_series,
            ig_series,
            top_k,
        )
        if shap_series is not None
        else np.nan
    )
    lime_vs_captum_topk = (
        topk_overlap(
            lime_series,
            ig_series,
            top_k,
        )
        if lime_series is not None
        else np.nan
    )
    shap_vs_lime_topk = (
        topk_overlap(
            shap_series,
            lime_series,
            top_k,
        )
        if shap_series is not None and lime_series is not None
        else np.nan
    )

    # Store method agreement values in one row table becaues these
    # metrics describe run as a whole rather than one feature at a
    # time.
    agreement_df = pd.DataFrame([{
        "SHAP_vs_Captum_Spearman": shap_vs_captum_rho,
        "LIME_vs_Captum_Spearman": lime_vs_captum_rho,
        "SHAP_vs_LIME_Spearman": shap_vs_lime_rho,
        f"SHAP_vs_Captum_Top{top_k}_Overlap": shap_vs_captum_topk,
        f"LIME_vs_Captum_Top{top_k}_Overlap": lime_vs_captum_topk,
        f"SHAP_vs_LIME_Top{top_k}_Overlap": shap_vs_lime_topk,
    }])

    # Use Captum IG feature order as shared feature index so all
    # exported importance columns line up on same feature list.
    feature_index = ig_series.index
    feature_df = pd.DataFrame({
        "Feature": feature_index.astype(str),
        "Permutation": model_importance.reindex(feature_index).values,
        "GradInput": grad_input_baseline.reindex(
            feature_index
        ).values,
        "Captum_IG": ig_series.reindex(feature_index).values,
        "SHAP": (
            shap_series.reindex(feature_index).values 
            if shap_series is not None and len(shap_series) > 0
            else np.full(len(feature_index), np.nan)
        ),
        "LIME": (
            lime_series.reindex(feature_index).values 
            if lime_series is not None and len(lime_series) > 0
            else np.full(len(feature_index), np.nan)
        ),
    })

    # Add clinical refernce flag when it is not already present so
    # later clinical alignment views can rely on one stable field
    # name.
    if "Clinically_Validated" not in feature_df.columns:
        clin_set = get_clinically_validated_set()
        feature_df["Clinically_Validated"] = (
            feature_df["Feature"]
            .astype(str)
            .isin(clin_set)
            .astype(int)
        )
    else:
        # Coerce any existing flag to clean numeric format for
        # consistent export.
        feature_df["Clinically_Validated"] = pd.to_numeric(
            feature_df["Clinically_Validated"], errors="coerce"
        ).fillna(0).astype(int)

    # Keep only methods that actually exist in run so pairwise
    # comparison logic stays aligned with enabled explainers.
    methods = {"Captum_IG": ig_series}
    if shap_series is not None:
        methods["SHAP"] = shap_series
    if lime_series is not None:
        methods["LIME"] = lime_series

    # Build comparison list dynamically so unavailable methods are
    # skipped instead of creating empty or misleading pariwise
    # outputs.
    comps = []
    if "SHAP" in methods:
        comps.append(("SHAP", "Captum_IG"))
    if "LIME" in methods:
        comps.append(("LIME", "Captum_IG"))
    if "SHAP" in methods and "LIME" in methods:
        comps.append(("SHAP", "LIME"))

    # Run pairwise comparisons across each enabled method pairing
    # and store statistical outputs used later in dashboard and
    # exported summaries.
    pairwise_rows = []
    pvals = []
    for a, b in comps:
        # Normalise both importance vectors before testing so
        # comparison is based on relative importance patterns
        # rather than raw scale differences.
        sa = l1_normalize(methods[a].abs())
        sb = l1_normalize(methods[b].abs())

        # Apply paired Wilcoxon test to aligned feature importance
        # vectors. Fall back to missing values when test cannot be
        # computed safely.
        try:
            stat, p = wilcoxon(sa.values, sb.values)
        except (TypeError, ValueError):
            stat, p = np.nan, np.nan

        # Copmute paired effect size separately so result captures
        # both statistic significance and direction and magnitude
        # of difference.
        rank_biserial = paired_rank_biserial(sa.values, sb.values)
  
        # Collect raw p-values for later multiple testing adjustment.
        pvals.append(p)

        # Store one summary row per comparison so pairwise table can
        # be reused across dashboard, PDF output, and interpretation
        # summary.
        pairwise_rows.append({
            "Comparison": f"{a} vs {b}",
            "Wilcoxon_stat": stat,
            "p_value": p,
            "Rank_Biserial": rank_biserial,
            "Effect_size_label": (
                interpret_rank_biserial(rank_biserial) 
                if pd.notna(rank_biserial) 
                else "n/a"
            )
        })

    # Apply Bonferroni adjustment only when pairwise tests were
    # generated. Keeps stored p-values aligned with number of
    # comparison made.
    if pairwise_rows:
        pvals_arr = np.array(
            [p if pd.notna(p) else 1.0 for p in pvals],
            dtype=float,
        )
        _, p_adj, _, _ = multipletests(pvals_arr, method="bonferroni")
        for row, p_b in zip(pairwise_rows, p_adj):
            row["p_adj_bonferroni"] = float(p_b)
            row["Significance_Label"] = interpret_significance(
                float(p_b)
            )
            row["Implication"] = (
                pairwise_implication_text(
                    row["Comparison"],
                    float(p_b),
                    row.get("Rank_Biserial", np.nan),
                )
            )
   
    return (
        method_df,
        agreement_df,
        feature_df,
        pd.DataFrame(pairwise_rows),
    )

def plot_topk(
    ig_series: pd.Series,
    shap_series: pd.Series | None,
    lime_series: pd.Series | None,
    k: int = TOP_K,
    figure_path: Path | None = None,
) -> str:
    """
    Save comparative Top-K feature importance figure and return its
    path.

    Uses top Captum IG features as shared display set so
    enabled methods can be compared against same feature subset.
    """
    # Use standard output location when no custom path is supplied.
    if figure_path is None:
        figure_path = (
            OUT_DIR / f"comparative_top{int(k)}_importance.png"
        )

    # Build one aligned DataFrame so all plotted importance values use
    # same feature order.
    feature_index = ig_series.index
    df = pd.DataFrame({"Captum_IG": ig_series.reindex(feature_index)})
    if shap_series is not None:
        df["SHAP"] = shap_series.reindex(feature_index)
    if lime_series is not None:
        df["LIME"] = lime_series.reindex(feature_index)

    # Select Top-K features based on Captum IG ranking used as
    # baseline view.
    top = df.nlargest(int(k), "Captum_IG")

    # Use horizontal bar chart so longer feature names remain easier
    # to read.
    ax = top.plot(kind="barh", figsize=(10, 6))
    ax.invert_yaxis()
    ax.set_xlabel("Global Importance (Absolute Attribution)")
    ax.set_ylabel("Feature")
    ax.set_title(
        f"Top-{int(k)} Features by Captum IG Baseline Ranking"
    )

    # Tighten layout before saving so labels and titles fit more
    # cleanly.
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()
    return str(figure_path)

# ---------------------------------
#  Master CSV / Save / Summary
# ---------------------------------
def get_master_columns(top_k: int) -> List[str]:
    """
    Return preferred column order for long-form master CSV export.

    Keeps exported schema stable across runs so downstream reading,
    review, and dashboard loading remain consistent.
    """
    return [
        "Method",
        "Stability_Spearman_Noise_Mean",
        "Stability_Spearman_Noise_CI_Low",
        "Stability_Spearman_Noise_CI_High",
        "Stability_Spearman_Masking_Mean",
        "Stability_Spearman_Masking_CI_Low",
        "Stability_Spearman_Masking_CI_High",
        f"Stability_Top{top_k}_Noise_Overlap_Mean",
        f"Stability_Top{top_k}_Masking_Overlap_Mean",
        "Fidelity_vs_Permutation_Spearman_Mean",
        "Fidelity_vs_Permutation_Spearman_CI_Low",
        "Fidelity_vs_Permutation_Spearman_CI_High",
        "Fidelity_vs_GradInput_Spearman_Mean",
        "Fidelity_vs_GradInput_Spearman_CI_Low",
        "Fidelity_vs_GradInput_Spearman_CI_High",
        f"Top{top_k}_Clinical_Overlap",
        f"Clinical_Top{top_k}_Overlap_Features",
        "Weighted_Clinical_Overlap",
        "Clinical_Precision",
        "Clinical_Recall",
        "RecordType",
        "SHAP_vs_Captum_Spearman",
        "LIME_vs_Captum_Spearman",
        "SHAP_vs_LIME_Spearman",
        f"SHAP_vs_Captum_Top{top_k}_Overlap",
        f"LIME_vs_Captum_Top{top_k}_Overlap",
        f"SHAP_vs_LIME_Top{top_k}_Overlap",
        "Feature",
        "Clinically_Validated",
        "Model_Permutation",
        "GradInput_Baseline",
        "Captum_IG",
        "SHAP",
        "LIME",
        "Model_Permutation_Rank",
        "GradInput_Baseline_Rank",
        "Captum_IG_Rank",
        "SHAP_Rank",
        "LIME_Rank",
        "Comparison",
        "Wilcoxon_stat",
        "p_value",
        "p_adj_bonferroni",
        "Rank_Biserial",
        "Effect_size_label",
        "Significance_Label",
        "Implication",
]

def assemble_master_metrics(
    *,
    method_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Combine benchmark result tables into one long-form export.

    Combined file keeps method level, agreement level, feature level,
    and pairwise statistics records together in one saved structure.
    """
    # Collect each block as separate DataFrame first, then concatenate
    # them at end.
    rows: List[pd.DataFrame] = []
    master_columns = get_master_columns(top_k)

    # Standardise Captum method label before export so master file
    # uses one consistent name across method level and feature level
    # records.
    md = method_df.copy()
    md["Method"] = md["Method"].replace({
        "Captum IG": "Captum_IntegratedGradients",
        "Captum_IntegratedGradients": "Captum_IntegratedGradients",
    })

    # Mark these rows as method summary records so downstream readers
    # can separate them easily.
    md["RecordType"] = "MethodSummary"
    if f"Clinical_Top{top_k}_Overlap_Features" not in md.columns:
        md[f"Clinical_Top{top_k}_Overlap_Features"] = ""
    rows.append(md)

    # Keep only the first agreement row because table is expected
    # to contain one run level summary rather than multiple repeated
    # rows.
    if len(agreement_df) > 0:
        ag = agreement_df.iloc[[0]].copy()
        ag["RecordType"] = "ExplainerAgreement"
        rows.append(ag)

    # Rename baseline columns to their master export names and
    # add any missing importance fields so final schema stays
    # consistent.
    fd = feature_df.copy().rename(
        columns={
            "Permutation": "Model_Permutation",
            "GradInput": "GradInput_Baseline",
        }
    )
    for c in [
        "Model_Permutation",
        "GradInput_Baseline",
        "Captum_IG",
        "SHAP",
        "LIME",
    ]:
        if c not in fd.columns:
            fd[c] = np.nan

    # Rebuild clinical flag from shared reference set so feature rows
    # always include clear clinical alignment indicator.
    clin_set = get_clinically_validated_set()
    fd["Clinically_Validated"] = fd["Feature"].astype(
        str
    ).isin(clin_set)

    # Add descending ranks for each importance column so master
    # export can support ranking based views without recomputing
    # them later.
    fd["Model_Permutation_Rank"] = _rank_desc(
        fd["Model_Permutation"]
    )
    fd["GradInput_Baseline_Rank"] = _rank_desc(
        fd["GradInput_Baseline"]
    )
    fd["Captum_IG_Rank"] = _rank_desc(fd["Captum_IG"])
    fd["SHAP_Rank"] = _rank_desc(fd["SHAP"])
    fd["LIME_Rank"] = _rank_desc(fd["LIME"])

    # Tag these rows so they can be separated from other record types
    # after export.
    fd["RecordType"] = "FeatureImportance"
    rows.append(fd)

    # Add pairwise statistics block as final record type in master
    # file.
    pw = pairwise_df.copy()
    pw["RecordType"] = "PairwiseStats"
    rows.append(pw)

    # Merge all record blocks into one long form export table.
    master = pd.concat(rows, ignore_index=True, sort=False)

    # Add any missing columns from preferred schema so column layout
    # remains stable.
    for c in master_columns:
        if c not in master.columns:
            master[c] = np.nan

    # Return master file with preferred columns first and any extras
    # placed after them.
    return master[
        master_columns
        + [c for c in master.columns if c not in master_columns]
    ]

def save_outputs(
        method_df: pd.DataFrame,
        agreement_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        pairwise_df: pd.DataFrame,
        model_comparison_df: pd.DataFrame | None = None,
        out_dir: Path = OUT_DIR,
        top_k: int = TOP_K,
) -> Dict[str, str]:
    """
    Save main CSV outputs for current run and return their paths.

    Writes core benchmark tables and also rebuilds combined master
    export so all save files stay aligned for same run.
    """
    # Create output folder before writing any files so later save
    # steps do not fail because destination directory is missing.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write each benchmark tables to its standard CSV path so
    # reporting interface can load expected files without extra
    # path logic.
    method_df.to_csv(OUT_METHOD_CSV, index=False)
    agreement_df.to_csv(OUT_AGREEMENT_CSV, index=False)
    feature_df.to_csv(OUT_FEATURE_CSV, index=False)
    pairwise_df.to_csv(OUT_PAIRWISE_CSV, index=False)

    # Save model comparison table when it is available.
    if (
        model_comparison_df is not None
        and not model_comparison_df.empty
    ):
        model_comparison_df.to_csv(
            OUT_MODEL_COMPARISON_CSV,
            index=False,
        )


    # Rebuild master export from latest in-memory tables so
    # combined file matches same method, agreement, feature, and
    # pairwise results.
    master_df = assemble_master_metrics(
        method_df=method_df,
        agreement_df=agreement_df,
        feature_df=feature_df,
        pairwise_df=pairwise_df,
        top_k=top_k,
    )
    master_df.to_csv(OUT_MASTER_CSV, index=False)

    # Return saved file paths in one dictionary so caller can pass
    # them directly into later interface or download logic.
    return {
        "master_csv": str(OUT_MASTER_CSV),
        "method_csv": str(OUT_METHOD_CSV),
        "agreement_csv": str(OUT_AGREEMENT_CSV),
        "feature_csv": str(OUT_FEATURE_CSV),
        "pairwise_csv": str(OUT_PAIRWISE_CSV),
    }

def write_interpretation_summary(
        method_df: pd.DataFrame,
        agreement_df: pd.DataFrame,
        pairwise_df: pd.DataFrame,
        out_path: Path,
        *,
        model_auc: float,
        top_k: int,
        model_comparison_df: pd.DataFrame | None = None,
):
    """
    Write short summary file for current benchmark run.

    Highlights strongest method level results first, then
    adds agreement and pairwise comparison values in simple text
    format.
    """
    def best_method(
        col: str,
        *,
        ascending: bool = False,
        absolute: bool = False,
    ):
        """
        Return method name with best value for given metric.

        """
        # Stop early if expected metric or method label is missing so
        # summary can degrade cleanly when column is unavailable.
        if (
            col not in method_df.columns
            or "Method" not in method_df.columns
        ):
            return None
   
        # Convert chosen metric to numeric form and remove missing
        # rows before selecting highest scoring method.
        tmp = method_df[["Method", col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna(subset=[col])

        # Return early when no valid values remain so summary lines
        # can handle missing metrics without raising errors.
        if tmp.empty:
            return None
   
        # Apply absolute comparison when requested so effect size
        # metrics are ranked by magnitude rather than direction.
        if absolute:
            tmp[col] = tmp[col].abs()

        # Sort values according to metric direction so best-performing
        # method appears at top of table.
        tmp = tmp.sort_values(col, ascending=ascending)
   
        # Return method corresponding to best observed value for this
        # metric so summary highlights strongest result.
        return tmp.iloc[0]["Method"]
    
    def best_model(
        col: str,
        *,
        ascending: bool = False,
    ):
        """
        Return model name with best value for given model metric.
        """
        # Stop early if model comparison output is unavailable so
        # summary can degrade cleanly when model metrics are absent.
        if model_comparison_df is None or model_comparison_df.empty:
            return None
        
        # Stop early if expected metric or model label is missing so
        # summary does not fail when export schema changes.
        if (
            col not in model_comparison_df.columns
            or "Model" not in model_comparison_df.columns
        ):
            return None
        
        # Convert chosen metric to numeric form and remove missing
        # rows before selecting best-performing model.
        tmp = model_comparison_df[["Model", col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna(subset=[col])

        # Return early when no valid values remain so summary lines
        # can handle missing model metrics without raising errors.
        if tmp.empty:
            return None
        
        # Sort values according to metric direction. For AUC, higher
        # is better; for Brier score and ECE, lower is better.
        tmp = tmp.sort_values(col, ascending=ascending)

        # Return model and metric value so summary gives useful
        # comparative context rather than only naming the model.
        row = tmp.iloc[0]
        return f"{row['Model']} ({row[col]:.3f})"
    
    def strongest_pairwise_effect():
        """
        Return pairwise comparison with strongest rank-biserial effect.
        """
        # Stop early if pairwise output is unavailable so summary can
        # degrade cleanly when statistical comparisons are absent.
        if pairwise_df is None or pairwise_df.empty:
            return None
        
        # Stop early if expected effect size or comparison label is
        # missing so summary does not fail when export schema changes.
        if (
            "Rank_Biserial" not in pairwise_df.columns
            or "Comparison" not in pairwise_df.columns
        ):
            return None
        
        # Convert rank biserial values to numeric form and remove
        # missing rows before selecting strongest pairwise effect.
        tmp = pairwise_df[["Comparison", "Rank_Biserial"]].copy()
        tmp["Rank_Biserial"] = pd.to_numeric(
            tmp["Rank_Biserial"],
            errors="coerce",
        )
        tmp = tmp.dropna(subset=["Rank_Biserial"])

        # Return early when no valid effects remain so summary lines
        # can handle missing statistical outputs without raising
        # errors.
        if tmp.empty:
            return None
        
        # Rank effects by absolute magnitude because strongest effect
        # depends on size of difference, not direction of difference.
        tmp["Abs_Rank_Biserial"] = tmp["Rank_Biserial"].abs()

        # Select comparison with largest absolute rank-biserial value.
        row = tmp.sort_values(
            "Abs_Rank_Biserial",
            ascending=False,
        ).iloc[0]

        # Return comparison and signed effect size so direction is
        # still visible after ranking by absolute magnitude.
        return f"{row['Comparison']} ({row['Rank_Biserial']:.3f})"

    # Start with run heading, timestamp, and model context so saved
    # text file can still be understood when viewed outside dashboard.
    lines = [
        "Comparative Explainability Benchmark - Summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Model & Task:",
        (
            "- Binary classification (PyTorch MLP); permutation "
            "baseline uses ROC-AUC."
        ),
        f"- Observed ROC-AUC on held-out test subset: {model_auc:.3f}",
        "- Model comparison context:",
        (
            "- Highest test AUC (comparative context): "
            f"{best_model('Test_AUC')}"
        ),
        "",
        "Key Quantitative Takeaways:",
        f"- Best noise robustness (Spearman): "
        f"{best_method('Stability_Spearman_Noise_Mean')}",
        f"- Best masking robustness (Spearman): "
        f"{best_method('Stability_Spearman_Masking_Mean')}",
        f"- Highest fidelity vs permutation: "
        f"{best_method('Fidelity_vs_Permutation_Spearman_Mean')}",
        f"- Highest fidelity vs gradient input: "
        f"{best_method('Fidelity_vs_GradInput_Spearman_Mean')}",
        f"- Highest clinical alignment (Top-{top_k} overlap): "
        f"{best_method(f'Top{top_k}_Clinical_Overlap')}",
        f"- Most accurate probabilities (Brier score): "
        f"{best_model('Test_Brier', ascending=True)}",
        f"- Best probability calibration (ECE): "
        f"{best_model('Test_ECE', ascending=True)}",
        f"- Strongest effect size (rank-biserial): "
        f"{strongest_pairwise_effect()}",
        "",
        "Explainer Agreement (Global):",
    ]

    # Add global agreement values when that run-level table is
    # available so summary includes cross-method similarity
    # information.
    if agreement_df is not None and not agreement_df.empty:
        agr = agreement_df.iloc[0].to_dict()
        for k in[
            "SHAP_vs_Captum_Spearman",
            "LIME_vs_Captum_Spearman",
            "SHAP_vs_LIME_Spearman",
            f"SHAP_vs_Captum_Top{top_k}_Overlap",
            f"LIME_vs_Captum_Top{top_k}_Overlap",
            f"SHAP_vs_LIME_Top{top_k}_Overlap",
        ]:
            if k in agr and pd.notna(agr[k]):
                lines.append(f"- {k}: {agr[k]:.3f}")

    # Add pairwise statistics section after agreement block so
    # saved text follows same summary flow as dashboard.
    lines.extend([
        "",
        "Pairwise Statistical Comparisons "
        "(Wilcoxon on normalized importances; rank-biserial "
        "effect size):"
    ])

    # Format each pairwise comparison into one readable line so text
    # file can be skimmed quickly without opening full tables.
    for _, r in pairwise_df.iterrows():
        p_txt = (
            "n/a"
            if pd.isna(r.get("p_adj_bonferroni"))
            else f"{r['p_adj_bonferroni']:.4g}"
        )
        d_txt = (
            "n/a"
            if pd.isna(r.get("Rank_Biserial"))
            else f"{r['Rank_Biserial']:.3f}"
        )
        sig_txt = r.get("Significance_Label", "n/a")
        implication = r.get("Implication", "")

        lines.append(
            f"- {r['Comparison']}: p_adj={p_txt}, "
            f"rank_biserial={d_txt}, {sig_txt}."
        )

        if implication:
            lines.append(f" Implication: {implication}")

    # Close with short note that explains role of text file relative
    # to more detailed interactive dashboard views.
    lines.extend([
        "",
        "Note:",
        "This file provides a concise summary of benchmark "
        "outputs. Detailed interpretation is presented in the "
        "interactive dashboard."
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------
# Main Benchmark Orchestration
# --------------------------------------------------------------------
def _to_long_instance_df(
    feature_names: list[str],
    ig_arr: np.ndarray | None,
    shap_arr: np.ndarray | None,
    lime_arr: np.ndarray | None,
    scenario: str,
    expected_n: int,
) -> pd.DataFrame:
    """
    Convert wide instance level attribution arrays into long
    DataFrame.

    Long format makes it easier for interface to filter by scenario,
    instance, feature, and explanation method in one table.
    """
    # Return empty table with expected schema when no attribution
    # arrays are available so downstream display code can still
    # run safely.
    if ig_arr is None and shap_arr is None and lime_arr is None:
        return pd.DataFrame(
            columns=[
                "InstanceID",
                "Scenario",
                "Feature",
                "Captum_IG",
                "SHAP",
                "LIME",
            ]
        )

    # Check each provided array before reshaping so stored instance
    # level results all use one consistent row count and two
    # dimensional structure.
    for name, arr in {
        "Captum_IG": ig_arr,
        "SHAP": shap_arr,
        "LIME": lime_arr,
    }.items():
        if arr is not None:
            arr = np.asarray(arr)

            if arr.ndim != 2:
                raise ValueError(
                    f"{name} array must be 2D, got shape {arr.shape}"
                )
            if arr.shape[0] != expected_n:
                raise ValueError(
                    f"{name} has {arr.shape[0]} rows, "
                    f"expected {expected_n}. "
                    f"This means the evaluation cohort is inconsistent."
                )

    # Build one row per intance feature combination so each scenario
    # can be explored at feature level detail inside reporting
    # interface.
    rows = []
    for i in range(expected_n):
        for j, feat in enumerate(feature_names):
            rows.append(
                {
                    "InstanceID": i,
                    "Scenario": scenario,
                    "Feature": feat,
                    "Captum_IG": (
                        float(ig_arr[i, j])
                        if ig_arr is not None
                        else np.nan
                    ),
                    "SHAP": (
                        float(shap_arr[i, j])
                        if shap_arr is not None
                        else np.nan
                    ),
                    "LIME": (
                        float(lime_arr[i, j])
                        if lime_arr is not None
                        else np.nan
                    ),
                }
            )

    # Convert collected rows into one DataFrame so later sections
    # can filter and plot instance level values without additional
    # reshaping.
    return pd.DataFrame(rows)  

def _evaluate_lime_repeatability(
    model,
    X_train: np.ndarray,
    X_eval: np.ndarray,
    feature_names: List[str],
    *,
    random_state: int,
    num_samples: int,
    discretize_continuous: bool,
    num_features_to_explain: int,
    repeats: int,
    top_k: int,
    sample_around_instance: bool,
) -> Dict[str, float]:
    """
    Estimate how repeatable LIME is across repeated local sampling
    runs.

    Isolates LIME's internal stochastic variation by rerunning
    explainer on same evaluation cohort with different random seeds.
    Reported separately from external robustness tests so internal
    sampling variation is not confused with noise or masking
    sensitivity.
    """
    # A single repeat cannot produce a repeatability estimate, so
    # return explicit missing values when fewer than two runs are
    # requested.
    if int(repeats) < 2:
        return {
            "mean_instance_spearman": np.nan,
            f"mean_instance_top{int(top_k)}_overlap": np.nan,
            "mean_signed_instance_spearman": np.nan,
        }
   
    # Store absolute and signed attribution matrices separately so
    # diagnostic can report both magnitude stability and directional
    # stability.
    abs_run_arrays: List[np.ndarray] = []
    signed_run_arrays: List[np.ndarray] = []

    # Run LIME explainer multiple times with different random seeds so
    # repeatability reflects variation in local sampling process.
    for rep_idx in range(int(repeats)):
        _, inst_signed = explanation_engine.compute_lime(
            model,
            X_train,
            X_eval,
            feature_names,
            random_state=int(random_state) + rep_idx,
            num_samples=int(num_samples),
            discretize_continuous=bool(discretize_continuous),
            return_signed_instance=True,
            num_features_to_explain=int(num_features_to_explain),
            n_repeats=1,
            sample_around_instance=bool(sample_around_instance),
        )
   
        # Convert returned explanations to numeric array so
        # comparisons across runs can be computed consistently.
        inst_signed = np.asarray(inst_signed, dtype=float)

        # Store signed attributions so directional agreement between
        # runs can be evaluated directly.
        signed_run_arrays.append(inst_signed)

        # Store absolute attributions separately so stability can also
        # be assessed based on importance magnitude rather than
        # direction.
        abs_run_arrays.append(np.abs(inst_signed))

    # Compare each pair of repeated runs so repeatability reflects
    # agreement across independently generated LIME explanations on
    # same cohort.
    spearman_vals: List[float] = []
    topk_vals: List[float] = []
    signed_spearman_vals: List[float] = []

    for i, abs_i in enumerate(abs_run_arrays):
        for j, abs_j in enumerate(abs_run_arrays[i + 1:], start=i + 1):

            # Measure agreement in absolute importance rankings so
            # stability reflects consistency in feature importance
            # magnitude.
            spearman_vals.append(
                _mean_instance_spearman(
                    abs_i,
                    abs_j,
                )
            )

            # Measure agreement in top-K feature sets so stability
            # reflects consistency in most influential features
            # identified per instance.
            topk_vals.append(
                _mean_instance_topk_overlap(
                    abs_i,
                    abs_j,
                    int(top_k),
                )
            )

            # Measure agreement in signed attributions so directional
            # consistency is also captured alongside magnitude-based
            # stability.
            signed_spearman_vals.append(
                _mean_instance_spearman(
                    signed_run_arrays[i],
                    signed_run_arrays[j],
                )
            )

    # Return mean agreement across all repeated run pairs so result
    # summarises LIME's internal repeatability in one compact
    # diagnostic.
    return {
        "mean_instance_spearman": (
            float(np.nanmean(spearman_vals))
            if spearman_vals
            else np.nan
        ),
        f"mean_instance_top{int(top_k)}_overlap": (
            float(np.nanmean(topk_vals))
            if topk_vals
            else np.nan
        ),
        "mean_signed_instance_spearman": (
            float(np.nanmean(signed_spearman_vals))
            if signed_spearman_vals
            else np.nan
        ),
    }

# Run full benchmark from data loading through export generation.
def run_benchmark(cfg: RunConfig) -> Dict[str, Any]:
    """
    Run full benchmark using supplied configuration.

    Loads data, trains model, computes explanation outputs,
    evaluates benchmark metrics, writes main files, and returns 
    in-memory objects needed by Streamlit interface.
    """
    # Reset NumPy and PyTorch random seeds at start of run so
    # repeated executions with same configuration remain more
    # consistent.
    rs = int(cfg.random_state)
    np.random.seed(rs)
    torch.manual_seed(rs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rs)

    # Load saved feature table and keep returned feature order
    # intact. Same order is reused later by explainers, baselines,
    # and output tables.
    X_df, y, used_features = (
        data_preparation.load_feature_table(cfg.data_path)
    )

    # Ask LIME to return weights across full feature set used
    # in benchmark so its outputs line up with same features
    # used elsewhere.
    lime_num_features = len(used_features)

    # Split dataset into training and test partitions.
    # Held-out test data is reserved for model evaluation and
    # permutation importance.
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=float(cfg.test_size),
        stratify=y,
        random_state=rs,
    )
    # Fit preprocessing on training data only, then reuse same fitted
    # objects for held-out test set and later perturbed inputs.
    X_train, imputer, scaler = (
        data_preparation.fit_preprocessor(X_train_df)
    )
    X_test = (
        data_preparation.transform_with_preprocessor(
            X_test_df,
            imputer,
            scaler,
        )
    )

    # Build single reference baseline reused across Integrated
    # Gradients calls so all IG explanations use same starting
    # reference point.
    ig_baseline_vec = _make_ig_baseline(X_train, mode="median")

    # Keep raw-data version of training frame for perturbation scaling
    # in noise analysis where raw feature variation is needed.
    X_train_raw_df = X_train_df[used_features].copy()

    # Select shared explanation cohort used by all explanation
    # methods. Keeping one common subset makes later cross-method
    # comparisons more direct.
    n_explain = min(int(cfg.n_debug), len(X_test))
    rng = np.random.default_rng(rs)
    explain_idx = np.sort(
        rng.choice(len(X_test), size=n_explain, replace=False)
    )

    X_explain = X_test[explain_idx]
    X_explain_df = X_test_df.iloc[explain_idx].copy()

    # Tune PyTorch MLP first because it remains the model used for 
    # explanation benchmark itself.
    model, best_mlp_params, mlp_search_results = (
        model_development.tune_mlp(
            X_train=X_train,
            y_train=np.asarray(y_train, dtype=np.float32),
            random_state=rs,
            epochs=int(cfg.n_train_epochs),
            val_frac=0.15,
            patience=10,
            min_delta=1e-4,
            use_class_weighting=True,
            use_weighted_sampler=bool(cfg.use_weighted_sampler),
            cv_folds=int(cfg.cv_folds),
        )
    )

    # Tune comparator models on same processed inputs so benchmark
    # can place fixed MLP explanation model in comparative context.
    logistic_model, logistic_params, logistic_search_results = (
        model_development.tune_logistic_regression(
            X_train=X_train,
            y_train=np.asarray(y_train, dtype=int),
            random_state=rs,
            cv_folds=int(cfg.cv_folds),
        )
    )
    rf_model, rf_params, rf_search_results = (
        model_development.tune_random_forest(
            X_train=X_train,
            y_train=np.asarray(y_train, dtype=int),
            random_state=rs,
            cv_folds=int(cfg.cv_folds),
        )
    )

    # Summarise all model families in one table so run records MLP
    # configuration used for explanations and its comparative
    # performance context.
    model_comparison_df = pd.DataFrame(
        mlp_search_results
        + logistic_search_results
        + rf_search_results
    )

    # Track which row corresponds to fixed explanation model and which
    # model families are eligible for downstream explanation pipeline.
    model_comparison_df["Used_for_Explanations"] = False
    model_comparison_df["Explanation_Eligible"] = model_comparison_df[
        "Model"
    ].eq("MLP")

    if best_mlp_params:
        # Identify MLP row used for explanations by matching fitted
        # parameter values so comparison table highlights that
        # configuration.
        explanation_model_mask = (
            (model_comparison_df["Model"] == "MLP")
            & (
                model_comparison_df["hidden_dim"]
                == int(best_mlp_params["hidden_dim"])
            )
            & (
                model_comparison_df["lr"]
                == float(best_mlp_params["lr"])
            )
            & (
                model_comparison_df["weight_decay"]
                == float(best_mlp_params["weight_decay"])
            )
            & (
                model_comparison_df["batch_size"]
                == int(best_mlp_params["batch_size"])
            )
        )
        model_comparison_df.loc[
            explanation_model_mask,
            "Used_for_Explanations",
        ] = True

        # Label role of each model family so table distinguishes
        # explanation model from predictor-only comparators.
        model_comparison_df["Model_Role"] = np.where(
            model_comparison_df["Model"].eq("MLP"),
            "Fixed explanation model",
            "Predictive comparator only",
        )

    # Evaluate each fitted model on same held-out test split so
    # comparison table includes one shared set of final performance
    # metrics.
    metric_by_model = {
        "MLP": model_development.evaluate_model_metrics(
            model,
            X_test,
            y_test,
        ),
        "Logistic Regression": model_development.evaluate_model_metrics(
            logistic_model,
            X_test,
            y_test,
        ),
        "Random Forest": model_development.evaluate_model_metrics(
            rf_model,
            X_test,
            y_test,
        ),
    }

    # Write each held-out test metric back to rows for corresponding
    # model family so tuning summaries and final results appear
    # together.
    for metric_col, metric_key in [
        ("Test_AUC", "roc_auc"), 
        ("Test_AP", "average_precision"), 
        ("Test_Brier", "brier_score"), 
        ("Test_ECE", "ece"),
    ]:
        model_comparison_df[metric_col] = np.nan

        for model_name, metrics in metric_by_model.items():
            model_comparison_df.loc[
                model_comparison_df["Model"] == model_name,
                metric_col,
            ] = float(metrics[metric_key])

    # Compute first fidelity baseline using permutation importance on
    # held-out test data.
    model_importance = compute_permutation_importance(
        model, 
        X_test, 
        y_test,
        used_features,
        perm_repeats=int(cfg.perm_repeats),
        random_state=rs,
    )
    # Compute second fidelity baseline from model gradients on
    # explanation cohort so fidelity can be compared against
    # two reference views.
    grad_input_baseline = compute_grad_input_baseline(
        model,
        X_explain,
        used_features,
    )

    # Generate original explanation outputs on shared explanation
    # cohort. Unperturbed explanations act as baseline for later
    # robustness checks.
    ig_series, ig_inst_abs = _compute_explainer(
        "ig",
        model,
        X_train,
        X_explain,
        used_features,
        random_state=rs,
        background_n=int(cfg.background_n),
        lime_num_samples=int(cfg.lime_num_samples),
        lime_discretize=bool(cfg.lime_discretize),
        ig_baseline_vec=ig_baseline_vec,
        lime_num_features=lime_num_features, 
        lime_repeats=int(cfg.lime_repeats),
        lime_sample_around_instance=bool(
            cfg.lime_sample_around_instance
        ),
    )

    shap_series, shap_inst_abs = (None, None)
    if cfg.run_shap:
        # Only compute SHAP outputs when enabled so run can skip
        # additional work when that explainer is not requested.
        shap_series, shap_inst_abs = _compute_explainer(
            "shap", 
            model,
            X_train,
            X_explain,
            used_features,
            random_state=rs,
            background_n=int(cfg.background_n),
            lime_num_samples=int(cfg.lime_num_samples),
            lime_discretize=bool(cfg.lime_discretize),
            ig_baseline_vec=ig_baseline_vec,
            lime_num_features=lime_num_features, 
            lime_repeats=int(cfg.lime_repeats),
            lime_sample_around_instance=bool(
                cfg.lime_sample_around_instance
            ),
        )

    lime_series, lime_inst_abs = (None, None)
    lime_repeatability = {
        "mean_instance_spearman": np.nan,
        f"mean_instance_top{int(cfg.top_k)}_overlap": np.nan,
        "mean_signed_instance_spearman": np.nan,
    }

    if cfg.run_lime:
        # Only compute LIME outputs when enabled so this method
        # remains an optional part of same benchmarking workflow.
        lime_series, lime_inst_abs = _compute_explainer(
            "lime",
            model,
            X_train,
            X_explain,
            used_features,
            random_state=rs,
            background_n=int(cfg.background_n),
            lime_num_samples=int(cfg.lime_num_samples),
            lime_discretize=bool(cfg.lime_discretize),
            ig_baseline_vec=ig_baseline_vec,
            lime_num_features=lime_num_features, 
            lime_repeats=int(cfg.lime_repeats), 
            lime_sample_around_instance=bool(
                cfg.lime_sample_around_instance
            ),
        )

        # Estimate repeatability separately by rerunning LIME with
        # different seeds so internal sampling variation is not
        # conflated with external robustness tests.
        lime_repeatability = _evaluate_lime_repeatability(
            model,
            X_train,
            X_explain,
            used_features,
            random_state=rs,
            num_samples=int(cfg.lime_num_samples),
            discretize_continuous=bool(cfg.lime_discretize),
            num_features_to_explain=lime_num_features,
            repeats=int(cfg.lime_repeatability_repeats),
            sample_around_instance=bool(
                cfg.lime_sample_around_instance
            ),
            top_k=int(cfg.top_k),
        )

    # Define two perturbation generators used in robustness analysis.
    # Noise changes observed values and masking removes part of input
    # information.
    def noise_fn(x_raw: pd.DataFrame, seed: int) -> pd.DataFrame:
        """
        Apply feature-wise noise using training set variation.
        """
        return apply_featurewise_noise_raw(
            x_raw,
            X_train_raw_df,
            noise_scale=float(cfg.noise_std),
            random_state=seed,
        )

    def mask_fn(x_raw: pd.DataFrame, seed: int) -> pd.DataFrame:
        """
        Apply random raw space feature masking.
        """
        return apply_random_feature_mask_raw(
            x_raw,
            mask_frac=float(cfg.mask_frac),
            random_state=seed,
        )

    # Run noise robustness test for Captum IG using shared explanation
    # cohort so resulting stability values remain comparable across
    # methods.
    ig_noise = _evaluate_robustness(
        "ig",
        model,
        X_train,
        X_explain_df[used_features],
        used_features,
        ig_inst_abs,
        perturbation_fn=noise_fn,
        runs=int(cfg.n_noise_runs),
        random_state=rs,
        top_k=int(cfg.top_k),
        background_n=int(cfg.background_n),
        lime_num_samples=int(cfg.lime_num_samples),
        lime_discretize=bool(cfg.lime_discretize),
        ig_baseline_vec=ig_baseline_vec,
        lime_num_features=lime_num_features,
        imputer=imputer,
        scaler=scaler,
        lime_repeats=int(cfg.lime_repeats),
        lime_sample_around_instance=bool(
            cfg.lime_sample_around_instance
        ),
    )

    # Run masking robustness test for Captum IG using same base cohort
    # so masking stability is assessed on same explanation rows.
    ig_mask = _evaluate_robustness(
        "ig",
        model,
        X_train,
        X_explain_df[used_features],
        used_features,
        ig_inst_abs,
        perturbation_fn=mask_fn,
        runs=int(cfg.mask_runs),
        random_state=rs,
        top_k=int(cfg.top_k),
        background_n=int(cfg.background_n),
        lime_num_samples=int(cfg.lime_num_samples),
        lime_discretize=bool(cfg.lime_discretize),
        ig_baseline_vec=ig_baseline_vec,
        lime_num_features=lime_num_features,
        imputer=imputer,
        scaler=scaler,
        use_scaled_masking=True,
        mask_frac=cfg.mask_frac,
        mask_value=cfg.mask_value,
        lime_repeats=int(cfg.lime_repeats),
        lime_sample_around_instance=bool(
            cfg.lime_sample_around_instance
        ),
    )

    # Start with empty placeholder results so later output building
    # code can stil rely on one consistent dictionary structure.
    shap_noise = {
        "spearman_ci": (np.nan, np.nan, np.nan), 
        "topk_ci": (np.nan, np.nan, np.nan), 
        "representative_instance_abs": None,
    }
    shap_mask = {
        "spearman_ci": (np.nan, np.nan, np.nan), 
        "topk_ci": (np.nan, np.nan, np.nan), 
        "representative_instance_abs": None,
    }
    if cfg.run_shap:
        # Evaluate SHAP under repeated noise perturbations using same
        # run settings used for other explainers.
        shap_noise = _evaluate_robustness(
            "shap",
            model,
            X_train,
            X_explain_df[used_features],
            used_features, 
            shap_inst_abs,
            perturbation_fn=noise_fn,
            runs=int(cfg.n_noise_runs),
            random_state=rs,
            top_k=int(cfg.top_k),
            background_n=int(cfg.background_n),
            lime_num_samples=int(cfg.lime_num_samples),
            lime_discretize=bool(cfg.lime_discretize),
            ig_baseline_vec=ig_baseline_vec,
            lime_num_features=lime_num_features,
            imputer=imputer, 
            scaler=scaler,
            lime_repeats=int(cfg.lime_repeats),
            lime_sample_around_instance=bool(
                cfg.lime_sample_around_instance
            ),
        )

        # Evaluate SHAP under masking perturbations so exported method
        # summary includes both robustness conditions.
        shap_mask = _evaluate_robustness(
            "shap",
            model,
            X_train,
            X_explain_df[used_features],
            used_features,
            shap_inst_abs,
            perturbation_fn=mask_fn,
            runs=int(cfg.mask_runs),
            random_state=rs,
            top_k=int(cfg.top_k),
            background_n=int(cfg.background_n),
            lime_num_samples=int(cfg.lime_num_samples),
            lime_discretize=bool(cfg.lime_discretize),
            ig_baseline_vec=ig_baseline_vec,
            lime_num_features=lime_num_features,
            imputer=imputer, 
            scaler=scaler,
            use_scaled_masking=True,
            mask_frac=cfg.mask_frac,
            mask_value=cfg.mask_value,
            lime_repeats=int(cfg.lime_repeats),
            lime_sample_around_instance=bool(
                cfg.lime_sample_around_instance
            ),
        )

    # Start with empty placeholder results for LIME as well so later
    # output structure stays stable whether or not LIME is enabled.
    lime_noise = {
        "spearman_ci": (np.nan, np.nan, np.nan), 
        "topk_ci": (np.nan, np.nan, np.nan), 
        "representative_instance_abs": None,
    }
    lime_mask = {
        "spearman_ci": (np.nan, np.nan, np.nan), 
        "topk_ci": (np.nan, np.nan, np.nan), 
        "representative_instance_abs": None,
    }
    if cfg.run_lime:
        # Evaluate LIME under repeated noise perturbations using same
        # benchmark configuration used for other methods.
        lime_noise = _evaluate_robustness(
            "lime",
            model,
            X_train,
            X_explain_df[used_features],
            used_features,
            lime_inst_abs,
            perturbation_fn=noise_fn,
            runs=int(cfg.n_noise_runs),
            random_state=rs,
            top_k=int(cfg.top_k),
            background_n=int(cfg.background_n),
            lime_num_samples=int(cfg.lime_num_samples),
            lime_discretize=bool(cfg.lime_discretize),
            ig_baseline_vec=ig_baseline_vec,
            lime_num_features=lime_num_features,
            imputer=imputer,
            scaler=scaler,
            lime_repeats=int(cfg.lime_repeats),
            lime_sample_around_instance=bool(
                cfg.lime_sample_around_instance
            ),
        )

        # Evaluate LIME under masking perturbations so final saved outputs
        # include same two robustness views as other methods.
        lime_mask = _evaluate_robustness(
            "lime",
            model,
            X_train,
            X_explain_df[used_features],
            used_features, 
            lime_inst_abs,
            perturbation_fn=mask_fn,
            runs=int(cfg.mask_runs),
            random_state=rs,
            top_k=int(cfg.top_k),
            background_n=int(cfg.background_n),
            lime_num_samples=int(cfg.lime_num_samples),
            lime_discretize=bool(cfg.lime_discretize),
            ig_baseline_vec=ig_baseline_vec,
            lime_num_features=lime_num_features,
            imputer=imputer,
            scaler=scaler,
            use_scaled_masking=True,
            mask_frac=cfg.mask_frac,
            mask_value=cfg.mask_value,
            lime_repeats=int(cfg.lime_repeats),
            lime_sample_around_instance=bool(
                cfg.lime_sample_around_instance
            ),
        ) 

    # Create DataFrame that stores original and representative
    # perturbed attribution arrays so reporting interface can show
    # instance level comparisons without recomputing benchmark.
    instance_attr_df = pd.concat(
        [
            _to_long_instance_df(
                used_features,
                ig_inst_abs if ig_inst_abs is not None else None,
                shap_inst_abs if shap_inst_abs is not None else None,
                lime_inst_abs if lime_inst_abs is not None else None,
                "Original",
                len(X_explain),
            ),
            _to_long_instance_df(
                used_features,
                (
                    ig_noise["representative_instance_abs"]
                    if ig_noise["representative_instance_abs"] is not None
                    else None
                ),
                (
                    shap_noise["representative_instance_abs"]
                    if shap_noise["representative_instance_abs"] is not None
                    else None
                ),
                (
                    lime_noise["representative_instance_abs"]
                    if lime_noise["representative_instance_abs"] is not None
                    else None
                ),
                "Noise",
                len(X_explain),
            ),
            _to_long_instance_df(
                used_features,
                (
                    ig_mask["representative_instance_abs"]
                    if ig_mask["representative_instance_abs"] is not None
                    else None
                ),
                (
                    shap_mask["representative_instance_abs"]
                    if shap_mask["representative_instance_abs"] is not None
                    else None
                ),
                (
                    lime_mask["representative_instance_abs"]
                    if lime_mask["representative_instance_abs"] is not None
                    else None
                ),
                "Masking",
                len(X_explain),
            ),
        ],
        ignore_index=True,
    )

    # Estimate fidelity uncertainty by resampling explanation cohort
    # so saved fidelity results include interval information rather
    # than one point estimate.
    ig_fid_perm_ci = bootstrap_fidelity_ci(
        ig_inst_abs,
        model_importance,
        used_features,
        seed=rs,
    )
    shap_fid_perm_ci = bootstrap_fidelity_ci(
        shap_inst_abs,
        model_importance,
        used_features,
        seed=rs,
    )
    lime_fid_perm_ci = bootstrap_fidelity_ci(
        lime_inst_abs,
        model_importance,
        used_features,
        seed=rs,
    )

    # Repeat same interval estimation against gradient baseline so
    # both fidelity reference views are treated same way.
    ig_fid_grad_ci = bootstrap_fidelity_ci(
        ig_inst_abs,
        grad_input_baseline,
        used_features,
        seed=rs,
    )
    shap_fid_grad_ci = bootstrap_fidelity_ci(
        shap_inst_abs,
        grad_input_baseline,
        used_features,
        seed=rs,
    )
    lime_fid_grad_ci = bootstrap_fidelity_ci(
        lime_inst_abs,
        grad_input_baseline,
        used_features,
        seed=rs,
    )

    # Build final structured tables consumed by dashboard and saved
    # outputs so all later exports read from one shared result set.
    method_df, agreement_df, feature_df, pairwise_df = build_outputs(
        model_importance=model_importance,
        grad_input_baseline=grad_input_baseline,
        ig_series=ig_series,
        shap_series=shap_series,
        lime_series=lime_series,
        ig_noise=ig_noise,
        shap_noise=shap_noise,
        lime_noise=lime_noise,
        ig_mask=ig_mask,
        shap_mask=shap_mask,
        lime_mask=lime_mask,
        ig_fid_perm_ci=ig_fid_perm_ci,
        shap_fid_perm_ci=shap_fid_perm_ci,
        lime_fid_perm_ci=lime_fid_perm_ci,
        ig_fid_grad_ci=ig_fid_grad_ci,
        shap_fid_grad_ci=shap_fid_grad_ci,
        lime_fid_grad_ci=lime_fid_grad_ci,
        top_k=int(cfg.top_k),
    )

    # Save long-form master CSV for current run so combined export
    # stays available alongside main summary tables.
    master_df = assemble_master_metrics(
        method_df=method_df,
        agreement_df=agreement_df,
        feature_df=feature_df,
        pairwise_df=pairwise_df,
        top_k=int(cfg.top_k),
    )
    master_df.to_csv(OUT_MASTER_CSV, index=False)

    # Save Top-K comparison plot after feature level outputs are ready
    # so interface can display latest run figure.
    fig_path = plot_topk(
        ig_series,
        shap_series,
        lime_series,
        k=int(cfg.top_k),
    )

    # Measure predictive quality of trained model on held-out test set
    # before writing interpretation summary.
    model_auc = float(
        model_development.evaluate_auc(model, X_test, y_test)
    )
    if model_auc < 0.60:
        warnings.warn(
            f"Model ROC-AUC is low ({model_auc:.3f})."
            "Explanation fidelity and robustness results should "
            "be interpreted cautiously."
        )

    # Write plain-text summary file for quick review outside
    # dashboard.
    write_interpretation_summary(
        method_df, 
        agreement_df, 
        pairwise_df, 
        OUT_SUMMARY_TXT, 
        model_auc=model_auc,
        top_k=int(cfg.top_k),
        model_comparison_df=model_comparison_df,
    )

    # Save run configuration so exported results can be tied back
    # to parameter settings used for this benchmark execution.
    OUT_RUN_CONFIG.write_text(
        json_dumps_safe(asdict(cfg)),
        encoding="utf-8",
    )

    # Return in-memory objects needed by reporting interface so page
    # can render latest run without reloading saved files first.
    return {
        "config": cfg,
        "model_auc": model_auc,
        "explanation_model_name": "Tuned MLP",
        "explanation_model_params": best_mlp_params,
        "logistic_model_params": logistic_params,
        "random_forest_params": rf_params,
        "model_comparison_df": model_comparison_df,
        "lime_repeatability": lime_repeatability,
        "method_df": method_df,
        "agreement_df": agreement_df,
        "feature_df": feature_df,
        "pairwise_df": pairwise_df,
        "figure_path": str(fig_path),
        "summary_txt_path": str(OUT_SUMMARY_TXT),
        "master_df": master_df,
        "instance_attr_df": instance_attr_df,
    }

# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------
def main():
    """
    Run benchmark using default configuration when script is executed
    directly.

    Keeps default run simple by executing full pipeline,
    saving outputs, and printing short summary to console.
    """
    # Create default configuration so benchmark runs with
    # standard parameters defined in RunConfig dataclass.
    cfg = RunConfig()

    # Execute full benchmarking workflow and collect all returned
    # outputs so they can be reused for saving and quick inspection.
    out = run_benchmark(cfg)

    # Save main CSV exports from completed run so results are
    # available to reporting interface and external review.
    save_outputs(
        out["method_df"], 
        out["agreement_df"], 
        out["feature_df"], 
        out["pairwise_df"],
        model_comparison_df=out.get("model_comparison_df"),
        top_k=out["config"].top_k,
    )

    # Print short console summary so user can confirm run completed
    # and view key result tables without opening dashboard.
    print(
        "\n=== Captum (IG Baseline) vs SHAP vs LIME "
        "comparison complete ==="
    )
    print(out["method_df"])
    print("\nAgreement (vs Captum baseline):")
    print(out["agreement_df"])

# Only run default benchmark when this file is executed directly.
# Prevents benchmark from running automatically when imported as
# module.
if __name__ == "__main__":
    main()
