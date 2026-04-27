"""
MODULE 3: Explanation Engine

Purpose:
- Generate feature attributions for the trained mortality model.
- Provide one shared interface for Captum Integrated Gradients, SHAP,
  and LIME.
- Return both global and instance-level attribution outputs for
  benchmarking and reporting.

Broader Context:
- data_preparation.py provides the processed arrays and feature names
  used when explanations are generated.
- model_development.py provides the trained model being explained.
- benchmarking_engine.py repeatedly calls this module to compare
  explanation methods under shared evaluation conditions.
- reporting_interface.py visualises the attribution outputs produced
  here.

Additional Notes:
- This module generates explanations but does not calculate the full
  benchmark comparison metrics.
- Its main role is to standardise attribution outputs so Captum IG,
  SHAP, and LIME can be compared consistently downstream.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

# --------------------------------------------------------------------
# SHAP Model Wrapper
# --------------------------------------------------------------------
class ShapModelWrapper(torch.nn.Module):
    """
    Wrap trained model so SHAP receives a consistent output shape.

    SHAP expects an explicity second output dimension, so this wrapper
    standardises model output to shape (n, 1) when needed.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Run wrapped model and return output shape compatible with
        SHAP.
        """
        # Run original model first so wrapper preserves same
        # prediction behaviour used elsewhere in pipeline.
        out = self.model(x)

        # Expand 1D outputs to (n, 1) so SHAP receives one consistent
        # shape regardless of how underlying model returns logits.
        if out.dim() == 1:
            out = out.unsqueeze(1)

        return out

# --------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------
def _validate_inputs(
    X: np.ndarray, 
    feature_names: List[str], 
    *, 
    X_name: str,
) -> None:
    """
    Validate explainer inputs before attribution is computed.

    Confirms that input matrix is two dimension, not empty, and 
    aligned with expected feature list.
    """
    # Require NumPy array so explainer functions all receive one
    # consistent input type.
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{X_name} must be a NumPy array.")
    
    # Confirm explainer input is 2D feature matrix rather than
    # flattened vector or higher dimensional structure.
    if X.ndim != 2:
        raise ValueError(f"{X_name} must be 2D, got shape {X.shape}.")
    
    # Require at least one row and one feature column so attribution 
    # can be computed on usable matrix.
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(
            f"{X_name} must have non-zero rows and "
            f"columns, got shape {X.shape}."
        )
    
    # Check that feature names match input column count so later
    # attribution outputs remain correctly labelled.
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"len(feature_names)={len(feature_names)} does not match "
            f"{X_name}.shape[1]={X.shape[1]}."
        )

def _to_global_series(
    instance_abs: np.ndarray,
    feature_names: List[str],
    name: str,
) -> pd.Series:
    """
    Convert instance level attributions into one global importance 
    series.

    Global importance for each feature is mean attribution magnitude
    across all explained instance.
    """
    # Convert to numeric array first so aggregation is applied to one
    # consistent representation.
    arr = np.asarray(instance_abs, dtype=float)

    # Validate expected 2D shape before reducing across instances
    # so feature alignment is preserved.
    if arr.ndim != 2 or arr.shape[1] != len(feature_names):
        raise ValueError(
            f"{name} attributions must be 2D "
            f"with {len(feature_names)} columns."
        )
    
    # Average across rows so returned series contains one global 
    # importance value per feature.
    vals = arr.mean(axis=0)

    return pd.Series(vals, index=feature_names, name=name)

def _prepare_baseline(
    X: np.ndarray,
    baseline: np.ndarray | None,
) -> np.ndarray:
    """
    Standardise baseline handline for Integrated Gradients.

    Supports no baseline, one shared baseline vector, or one
    row aligned baseline matrix that already matches input shape.
    """
    # Use all zero baseline when none is supplied so method can still
    # run with simple default reference point.
    if baseline is None:
        return np.zeros_like(X, dtype=np.float32)
    
    base = np.asarray(baseline, dtype=np.float32)

    # Repeat single baseline vector across all rows so each instance 
    # is compared against same reference vector.
    if base.ndim == 1:
        if base.shape[0] != X.shape[1]:
            raise ValueError(
                f"Baseline length {base.shape[0]} does "
                f"not match feature count {X.shape[1]}."
            )
        return np.repeat(base.reshape(1, -1), X.shape[0], axis=0)
    
    # Keep 2D baseline as-is when it already matches input matrix
    # so each row can use its aligned baseline directly.
    if base.ndim == 2:
        if base.shape != X.shape:
            raise ValueError(
                f"Baseline shape {base.shape} does not match "
                f"X shape {X.shape}."
            )
        return base
    
    raise ValueError("Baseline must be None, 1D, or 2D.")

def _normalize_instance_abs(arr: np.ndarray) -> np.ndarray:
    """
    Convert attributions to absolute, row normalised values.

    Places methods on more comparable scale for later overlap,
    fidelity, and robustness comparison.
    """
    # Convert to numeric array and take absolute values so benchmark 
    # focuses on attribution magnitude instead of direction.
    arr = np.asarray(arr, dtype=float)
    arr = np.abs(arr)

    # Normalise each row to sum to one so instance level attribution 
    # vectors can be compared on common scale.
    denom = arr.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return arr / denom

@torch.no_grad()
def _predict_two_class_proba(model, x: np.ndarray) -> np.ndarray:
    """
    Return probabilities in two column form of 
    [P(class 0), P(class 1)].

    Gives LIME prediction function in format expected for 
    binary classification.
    """
    # Convert supplied data to float32 tensor so it matches
    # trained model's expected numeric input type.
    x_t = torch.tensor(
        np.asarray(x, dtype=np.float32), 
        dtype=torch.float32,
    )
    logits = model(x_t)

    # Convert logits to probabilities for postive class, then clip 
    # slightly away from exact boundaries.
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)

    # Return probabilities in two column format so downstream
    # consumers can read both class probabilities directly.
    return np.column_stack([1.0 - probs, probs])

#---------------------------------------------------------------------
# Integrated Gradients
# --------------------------------------------------------------------
def compute_integrated_gradients(
    model,
    X_test_small: np.ndarray,
    feature_names: List[str],
    *,
    baseline: np.ndarray | None = None,
) -> Tuple[pd.Series, np.ndarray]:
    """
    Compute Captum Integrated Gradients attributions.

    Returns both global feature importance series and instance level
    absolute attributions used later in benchmarking and reporting.
    """
    # Import Captum locally so module can still load when only
    # SHAP or LIME explanations are being used.
    from captum.attr import IntegratedGradients

    # Validate evaluation matrix before computing attributions so
    # feature labels and matrix shape stay aligned.
    _validate_inputs(
        X_test_small, 
        feature_names, 
        X_name="X_test_small",
    )
    model.eval()

    # Convert evaluation data and prepare baseline in matching shape
    # before passing both inputs to Captum.
    X = np.asarray(X_test_small, dtype=np.float32)
    baseline_np = _prepare_baseline(X, baseline)

    # Compare each evaluation example against chosen baseline so
    # Captum can compute Integrated Gradients attributions.
    ig = IntegratedGradients(model)
    attributions = ig.attribute(
        torch.tensor(X, dtype=torch.float32), 
        baselines=torch.tensor(baseline_np, dtype=torch.float32),
    )

    # Convert Captum output to standard NumPy array so same 
    # downstream logic can be reused across methods.
    arr = np.asarray(attributions.detach().cpu().numpy(), dtype=float)

    # Restore expected 2D shape if unexpected output format appears
    # so later steps still receive one row per instance and one column
    # per feature.
    if arr.ndim != 2:
        arr = arr.reshape(X.shape[0], X.shape[1])
  
    # Convert raw attributions into absolute, row normalised form
    # so method is on same comparison scale used elsewhere in
    # benchmark.
    instance_abs = _normalize_instance_abs(arr)    

    # Collapse instance level values into one global feature
    # importance summary for later tables and plots.
    series = _to_global_series(
        instance_abs, 
        feature_names, 
        "Captum_IG",
    )
    return series, instance_abs

# --------------------------------------------------------------------
# SHAP GradientExplainer
# --------------------------------------------------------------------
def compute_shap_gradients(
        model,
        X_train: np.ndarray,
        X_test_small: np.ndarray,
        feature_names: List[str],
        *,
        background_n: int = 128,
        random_state: int = 42,
) -> Tuple[pd.Series, np.ndarray]:
    """
    Compute SHAP GradientExplainer attributions.

    Returns both global importance series and instance level absolute
    attributions used later in fidelity and robustness evaluation.
    """
    # Import SHAP locally because it is only needed when SHAP
    # explanations are requested.
    import shap

    # Validate training and evaluation arrays first so SHAP explainer
    # receives matrices that match expected feature schema.
    _validate_inputs(X_train, feature_names, X_name="X_train")
    _validate_inputs(
        X_test_small, 
        feature_names, 
        X_name="X_test_small",
    )
    if background_n <= 0:
        raise ValueError("background_n must be positive.")
    
    model.eval()

    # Select reproducible background sample from training set so SHAP
    # uses stable reference distribution across repeated runs.
    rng = np.random.default_rng(random_state)
    bg_n = min(int(background_n), len(X_train))
    bg_idx = rng.choice(len(X_train), size=bg_n, replace=False)

    background = torch.tensor(
        X_train[bg_idx], 
        dtype=torch.float32,
    )
    X_eval_t = torch.tensor(
        X_test_small, 
        dtype=torch.float32,
    )
    # Wrap model so SHAP receives one consistent output shape before
    # GradientExplainer is initialised.
    wrapped_model = ShapModelWrapper(model)
    explainer = shap.GradientExplainer(wrapped_model, background)

    # Compute SHAP values for evaluation cohor that will be
    # benchmarked against other explanation methods.
    shap_vals = explainer.shap_values(X_eval_t)

    # Handle different output formats SHAP can return so final array
    # is standardised before later processing.
    if isinstance(shap_vals, (list, tuple)):
        shap_arr = np.asarray(shap_vals[0], dtype=float)
    else:
        shap_arr = np.asarray(shap_vals, dtype=float)

    # Remove singleton output dimension when present so result returns
    # to expected instance by feature layout.
    if shap_arr.ndim == 3 and shap_arr.shape[-1] == 1:
        shap_arr = shap_arr[..., 0]

    # Restore single explained row to 2D form so downstream logic
    # still receives matrix rather than flattened vector.
    if shap_arr.ndim == 1:
        shap_arr = shap_arr.reshape(1, -1)

    # Apply one final defensive reshape so array follows project's
    # standard explanation shape.
    if shap_arr.ndim != 2:
        shap_arr = shap_arr.reshape(
            X_test_small.shape[0], 
            len(feature_names),
        )

    # Convert SHAP values to shared absolute, row normalised form
    # used in later benchmarking comparisons.
    instance_abs = _normalize_instance_abs(shap_arr)
    series = _to_global_series(instance_abs, feature_names, "SHAP")

    return series, instance_abs

# ------------------------------------------------------------
# LIME
# ------------------------------------------------------------
def compute_lime(
        model,
        X_train: np.ndarray,
        X_test_small: np.ndarray,
        feature_names: List[str],
        *,
        random_state: int = 42,
        num_samples: int = 1000,
        discretize_continuous: bool = False,
        return_signed_instance: bool = False,
        num_features_to_explain: int | None = None,
        n_repeats: int = 1,
        sample_around_instance: bool = False,
) -> Tuple[pd.Series, np.ndarray]:
    """
    Compute LIME tabular explanations.

    Returns global importance series and instance level attributions,
    with option to return signed per-instance weights instead of 
    absolute normalised version. Repeated runs are supported so caller
    can either average multiple local surrogate fits or isolate LIME's
    internal stochastic variation by running one repeat at a time with
    different seeds when requested.
    """
    # Import LIME locally because it is only needed when LIME
    # explanations are requested.
    from lime.lime_tabular import LimeTabularExplainer

    # Validate training and evaluation matrices before LIME is
    # initialised so feature schema stays consistent.
    _validate_inputs(X_train, feature_names, X_name="X_train")
    _validate_inputs(
        X_test_small, 
        feature_names, 
        X_name="X_test_small",
    )
    
    # Require positive sample count so function does not silently 
    # proceed with invalid sample configuration.
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be positive.")
    
    # Require positive repeat count so function does not silently
    # proceed with invalid repeat configuration.
    if int(n_repeats) <= 0:
        raise ValueError("n_repeats must be positive.")
    
    model.eval()

    # Convert both arrays to float32 so explainer and prediction 
    # function work from one consistent numeric representation.
    X_train = np.asarray(X_train, dtype=np.float32)
    X_eval = np.asarray(X_test_small, dtype=np.float32)

    n_rows, n_features = X_eval.shape

    # Control how many features LIME is allowed to return for each
    # instance so output stays bounded by supplied setting and 
    # feature count.
    lime_num_features = int(num_features_to_explain or n_features)
    lime_num_features = max(1, min(lime_num_features, n_features))

    # Store signed weights first so function can optionally return 
    # signed explanation matrix before absolute normalisation.
    signed = np.zeros((n_rows, n_features), dtype=float)

    # Supply LIME with prediction function returning two-class
    # probabilities in format expected by explainer.
    predict_fn = lambda z: _predict_two_class_proba(model, z)

    # Explain each row one at a time because LIME generates local 
    # explanations independently for each instance.
    for i, row in enumerate(X_eval):
        # Collect one signed explanation vector per repeat so
        # repeated local samples can be averaged for current row.
        rep_signed = np.zeros(
            (int(n_repeats), n_features), 
            dtype=float,
        )
        
        # Rebuild explainer inside each repeat so repeated runs use
        # different local sampling seeds rather than one reused state.
        for rep_idx in range(int(n_repeats)):
            # Recreate explainer with new seed on each repeat
            # so repeated LIME runs reflect sampling variation more
            # honestly.
            explainer = LimeTabularExplainer(
                training_data=X_train,
                feature_names=feature_names,
                class_names=["survived", "expired"],
                mode="classification",
                discretize_continuous=bool(discretize_continuous),
                sample_around_instance=bool(sample_around_instance),
                random_state=int(random_state) + rep_idx,
            )
            exp = explainer.explain_instance(
                data_row=row,
                predict_fn=predict_fn,
                num_features=lime_num_features,
                num_samples=int(num_samples),
            )

            # Read weights from local_exp when available because
            # that is the preferred explanation structure exposed
            # by LIME.
            if hasattr(exp, "local_exp") and 1 in exp.local_exp:
                for feat_idx, weight in exp.local_exp[1]:
                    rep_signed[rep_idx, int(feat_idx)] = float(weight)
            else:
                # Fall back to as_map() when that is accessible
                # structure so method still returns a usable
                # explanation matrix.
                for feat_idx, weight in exp.as_map().get(1, []):
                    rep_signed[rep_idx, int(feat_idx)] = float(weight)
        
        # Average repeated signed local explanations for this row
        # so final output is less dependent on one stochastic 
        # neighbourhood sample.
        signed[i] = rep_signed.mean(axis=0)

    # Convert signed weights into shared absolute, row normalised
    # representation used for cross-method comparisons.
    instance_abs = _normalize_instance_abs(signed)
    series = _to_global_series(instance_abs, feature_names, "LIME")

    # Return signed matrix when explicitly requested so caller can
    # inspect direction as well as magnitude.
    if return_signed_instance:
        return series, signed

    return series, instance_abs