"""
MODULE 2: Model Development

Purpose:
- Define and train the baseline neural network used in the
  benchmarking workflow.
- Train comparator models for broader predictive performance context.
- Produce probability estimates for the binary mortality prediction
  task.
- Provide model-side utilities reused by the benchmarking pipeline.

Broader Context:
- data_preparation.py builds and preprocesses the tabular inputs used
  for model fitting.
- explanation_engine.py relies on the trained neural network when
  generating gradient-based and local explanations.
- benchmarking_engine.py trains, tunes, evaluates, and compares the
  models defined here.
- reporting_interface.py displays benchmark outputs that depend on the
  trained model and comparator results.

Additional Notes:
- This module focuses on model training, tuning, prediction, and
  evaluation utilities.
- Its main role is to provide a consistent prediction model and
  comparator context for the explainability benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Iterable, Any
from itertools import product

import numpy as np
import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split, StratifiedKFold

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
@dataclass
class ModelConfig:
    """
    Store main configuration values used to train baseline model.

    Grouping these settings in one dataclass keeps training defaults
    easier to review, reuse, and pass between modules.
    """
    hidden_dim: int = 64
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 64
    val_frac: float = 0.15
    patience: int = 10
    min_delta: float = 1e-4
    weight_decay: float = 1e-4
    use_class_weighting: bool = True
    use_weighted_sampler: bool = False
    random_state: Optional[int] = 42

# Store compact default search space so benchmark can compare a few
# reasonable MLP settings.
DEFAULT_MLP_SEARCH_SPACE: Dict[str, Iterable[Any]] = {
    "hidden_dim": (32, 64, 128),
    "lr": (1e-3, 5e-4),
    "weight_decay": (0.0, 1e-4),
    "batch_size": (32, 64),
}

# Store small set of regularisation strengths for logistic regression
# comparator so baseline comparison remains lightweight and
# repeatable.
DEFAULT_LOGISTIC_C_VALUES: tuple[float, ...] = (0.1, 1.0, 10.0)

DEFAULT_RF_SEARCH_SPACE: Dict[str, Iterable[Any]] = {
    "n_estimators": (200, 400),
    "max_depth": (None, 8),
    "min_samples_leaf": (1, 5),
}

# --------------------------------------------------------------------
# Model Definition
# --------------------------------------------------------------------
class MortalityMLP(nn.Module):
    """
    Baseline multilayer perceptron (MLP) for binary mortality
    prediction.

    Model uses one hidden layer and returns one logit per row, which 
    is later converted to probability when needed.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        """
        Build compact feed-forward network so project has one 
        consistent baseline model for training and explanation.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return one raw model score for each input row.
        """
        # Squeeze final dimension so output shape matches one
        # dimensional label tensor used by loss function.
        return self.net(x).squeeze(1)
    
# --------------------------------------------------------------------
# Internal Utilities
# --------------------------------------------------------------------
def _set_seed(random_state: Optional[int]) -> None:
    """
    Set random seeds for NumPy and PyTorch.

    Helps repeated runs stay more consistent when same configuration
    is used again.
    """
    # Stop early when no random state is supplied so caller can
    # intentionally allow non-fixed randomness.
    if random_state is None:
        return
    
    # Apply same seed to NumPy and PyTorch so main training steps
    # share one reproducible random state.
    np.random.seed(int(random_state))
    torch.manual_seed(int(random_state))

    # Extend same seed setting to CUDA when GPU is available so
    # device-specific randomness is reduced as well.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_state))

    # Reduce nondeterministic backend behaviour where possible so
    # later training runs remain easier to compare.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def _validate_training_inputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> None:
    """
    Validate training arrays before model fitting begins.

    Catches shape and type problems early so training does not fail
    later inside optimisation loop.
    """
    # Require NumPy arrays so rest of training logic can rely on
    # one expected input type.
    if not isinstance(X_train, np.ndarray):
        raise TypeError("X_train must be a NumPy array.")
    if not isinstance(y_train, np.ndarray):
        raise TypeError("y_train must be a NumPy array.")
    
    # Confirm feature matrix and label array use expected number of
    # dimensions for tabular binary classification.
    if X_train.ndim != 2:
        raise ValueError(
            f"X_train must be 2D, got shape {X_train.shape}."
        )
    if y_train.ndim !=1:
        raise ValueError(
            f"y_train must be 1D, got shape {y_train.shape}."
        )
 
    # Check that row counts match before model, optimiser, and loss
    # function are initialised.
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train row mismatch: "
            f"{len(X_train)} vs {len(y_train)}."
        )
 
    # Require at least one row and one feature column so model
    # fitting has usable training data.
    if len(X_train) == 0 or X_train.shape[1] == 0:
        raise ValueError(
            "X_train must contain at least one row and one "
            "feature column."
        )
 

def _can_stratify(y: np.ndarray) -> bool:
    """
    Check whether stratified train validation split is feasible.

    Stratified splitting requires at least two classes and enough
    observations in each class so both training and validation
    partitions contain representative samples.
    """
    # Convert labels to integers to ensure consistent counts across
    # inputs.
    values, counts = np.unique(y.astype(int), return_counts=True)

    # Require at least two classes and at least two observations in
    # smallest class so both partitions can include that class.
    return len(values) >= 2 and counts.min() >= 2

def _positive_class_weight(y: np.ndarray) -> Optional[float]:
    """
    Compute simple positive class weight for binary classification.

    Weight is used to count class imbalance by scaling loss
    contribution of positive examples relative to negative examples.
    """
    # Flatten label array so counting logic is applied to 1D vector.
    y = np.asarray(y).reshape(-1)

    # Count number of positive and negative labels.
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))

    # Return None when one class is missing, since weighting cannot be
    # computed meaningfully in that case.
    if pos <= 0 or neg <= 0:
        return None
    
    # Scale positive class inversely to its frequency so minority
    # class receives greater emphasis during training.
    return neg / pos

def _make_epoch_indices(
    y: np.ndarray, 
    use_weighted_sampler: bool, 
    random_generator: np.random.Generator,
) -> np.ndarray:
    """
    Generate index ordering for one training epoch.

    When weighted sampling is enabled, indices are sampled with
    replacement using inverse class frequency so minority classes
    appear more often. Otherwise, standard random permutation is
    returned.
    """
    # Determine number of training rows available.
    n = len(y)

    # Use simple shuffled ordering when weighted sampling is 
    # disabled or when no rows are available.
    if not use_weighted_sampler or n == 0:
        return random_generator.permutation(n)
    
    # Convert labels to integers so class counts can be computed
    # reliably.
    y_int = np.asarray(y, dtype=int)

    # Count number of examples in each class.
    counts = np.bincount(y_int, minlength=2).astype(float)

    # Fall back to standard shuffling when any class is missing, since
    # weighted sampling cannot be constructed safely in that case.
    if np.any(counts == 0):
        return random_generator.permutation(n)
    
    # Compute inverse frequency class weights so rarer classes are
    # sampled more often.
    class_weights = 1.0 / counts

    # Assign each sample a weight based on its class membership.
    sample_weights = class_weights[y_int]

    # Normalise weights so they form valid probability distribution.
    sample_weights = sample_weights / sample_weights.sum()

    # Sample indices with replacement according to computed weights so
    # each epoch reflects a balanced class representation.
    return random_generator.choice(
        n,
        size=n,
        replace=True,
        p=sample_weights,
    )
    
# --------------------------------------------------------------------
# Training
# --------------------------------------------------------------------
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    batch_size: int = 64,
    val_frac: float = 0.15,
    patience: int = 10,
    min_delta: float = 1e-4,
    weight_decay: float = 1e-4,
    use_class_weighting: bool = True,
    use_weighted_sampler: bool = False,
    random_state: Optional[int] = 42,
) -> nn.Module:
    """
    Train baseline MLP on preprocessed NumPy arrays.

    Training workflow supports mini-batch optimisation, internal
    validation split when feasible, early stopping, and optional class
    imbalance handling through positive class weighting and weighted
    sampling across training epochs.
    """
    # Validate raw inputs before any tensors or training objects
    # are created so errors are raised in one clear place.
    _validate_training_inputs(X_train, y_train)
    _set_seed(random_state)

    rng = np.random.default_rng(random_state)

    # Convert arrays to float32 so they match tensor dtype used
    # throughout training and prediction.
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)

    # ----------------------------------------------------------------
    # Validation Split
    # ----------------------------------------------------------------
    # Use validation split when requested fraction is sensible and
    # label distribution supports stratification.
    use_validation = (
        0.0 < float(val_frac) < 0.5
        and _can_stratify(y_train)
    )
    if use_validation:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=float(val_frac),
            stratify=y_train.astype(int),
            random_state=random_state,
        )
    else:
        # Fall back to training-only monitoring when proper validation
        # split cannot be created from available data.
        X_tr, y_tr = X_train, y_train
        X_val, y_val = None, None

    # Covert NumPy arrays to tensors once so main training loop can
    # focus on optimisation rather than repeated type conversion.
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

    X_val_t = (
        torch.tensor(X_val, dtype=torch.float32)
        if X_val is not None
        else None
    )
    y_val_t = (
        torch.tensor(y_val, dtype=torch.float32)
        if y_val is not None
        else None
    )

    # Build baseline model using current number of input features so
    # network matches supplied training matrix.
    mlp_model = MortalityMLP(
        input_dim=X_train.shape[1], 
        hidden_dim=hidden_dim,
    )

    # Compute positive class loss weight when requested so minority
    # positive cases contribute more strongly during optimisation.
    pos_weight_value = (
        _positive_class_weight(y_tr)
        if use_class_weighting
        else None
    )

    if pos_weight_value is not None:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [pos_weight_value],
                dtype=torch.float32,
            )
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Use Adam for compact, stable optimisation on tabular baseline
    # model.
    optimizer = torch.optim.Adam(
        mlp_model.parameters(), 
        lr=float(lr), 
        weight_decay=float(weight_decay),
    )

    # Track best observed model state so final returned model
    # reflects lowest monitored loss rather than last completed
    # epoch.
    best_state = {
        k: v.detach().cpu().clone()
        for k, v in mlp_model.state_dict().items()
    }
    best_metric = np.inf
    bad_epochs = 0

    # Run main training loop for requested number of epochs unless
    # early stopping is triggered first.
    for _ in range(int(epochs)):
        mlp_model.train()

        # Generate row ordering for this epoch. When weighted
        # sampling is enabled, minority class rows can appear more
        # often in minibatches.
        epoch_idx = _make_epoch_indices(
            y_tr, 
            bool(use_weighted_sampler), 
            rng,
        )

        # Process one minibatch at a time so model is trained with
        # stochastic gradient updates rather than full-batch
        # optimisation.
        for start in range(0, len(epoch_idx), int(batch_size)):
            idx = epoch_idx[start:start + int(batch_size)]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]

            optimizer.zero_grad()
            logits = mlp_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Evaluate monitored loss after each epoch. Prefer validation
        # loss when validation split is available, otherwise fall back
        # to training loss.
        mlp_model.eval()
        with torch.no_grad():
            if X_val_t is not None and y_val_t is not None:
                metric = float(
                    criterion(mlp_model(X_val_t), y_val_t).item()
                )
            else:
                metric = float(
                    criterion(mlp_model(X_tr_t), y_tr_t).item()
                )
        
        # Save model state whenever monitored loss improves by at 
        # least minimum required margin.
        if metric < (best_metric - float(min_delta)):
            best_metric = metric
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in mlp_model.state_dict().items()
            }
            bad_epochs = 0
        else:
            # Count consecutive non-improving epochs so training can
            # stop early once progress has stalled.
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    # Restore best performing state before returning so later 
    # evalaution and explanation steps use strongest observed model.
    mlp_model.load_state_dict(best_state)
    mlp_model.eval()
    return mlp_model

# --------------------------------------------------------------------
# Model Comparison Helpers
# --------------------------------------------------------------------
def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: str | None = "balanced",
    random_state: Optional[int] = 42,
) -> LogisticRegression:
    """
    Train logistic regression comparator on same processed inputs.

    Provides simple linear reference model so benchmark can compare 
    tuned MLP against lower-complexity baseline trained on same
    feature set.
    """
    # Validate shared training inputs first so comparator receives
    # same checked arrays expected by neural network training path.
    _validate_training_inputs(X_train, y_train)

    # Fit balanced logistic regression model so comparator remains
    # simple while still accounting for class imbalance in training
    # labels.
    logistic_model = LogisticRegression(
        C=float(C),
        max_iter=int(max_iter),
        class_weight=class_weight,
        random_state=random_state,
        solver="liblinear",
    )
    logistic_model.fit(
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train, dtype=int).reshape(-1),
    )
    return logistic_model

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    class_weight: str | None = "balanced_subsample",
    random_state: Optional[int] = 42,
) -> RandomForestClassifier:
    """
    Train random forest comparator on same processed inputs.

    Adds non-linear tree-based reference point so benchmark can place 
    tuned MLP in broader predictive context than linear comparator
    alone.
    """
    # Validate shared training inputs first so tree-based comparator
    # uses same checked arrays as other model families.
    _validate_training_inputs(X_train, y_train)

    # Fit balanced random forest so comparator can model non-linear
    # structure while still accounting for class imbalance in labels.
    rf_model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=(
            None
            if max_depth in (None, "None")
            else int(max_depth)
        ),
        min_samples_leaf=int(min_samples_leaf),
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_model.fit(
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train, dtype=int).reshape(-1),
    )
    return rf_model

def _iter_cv_splits(
    y: np.ndarray,
    cv_folds: int,
    random_state: Optional[int],
):
    """
    Return stratified cross validation splits for supplied label
    vector.

    Keeps fold generation in one place so MLP, logistic regression,
    and random forest tuning paths all use same validation structure.
    """
    # Standardise labels as one dimensional array before checking
    # whether stratified splitting is feasible.
    y = np.asarray(y).reshape(-1)

    # Stop early when label distribution cannot support a stratified
    # split across requested folds.
    if not _can_stratify(y):
        raise ValueError(
            "Cross validation requires at least two "
            "classes with at least tro rows each."
        )
    
    # Build stratified splitter so class balance is preserved more
    # consistently across validation folds.
    splitter = StratifiedKFold(
        n_splits=int(cv_folds), 
        shuffle=True, 
        random_state=random_state,
    )
    return splitter.split(np.zeros(len(y)), y.astype(int))

# --------------------------------------------------------------------
# Prediction Utilities
# --------------------------------------------------------------------

def _predict_positive_proba_any(
    estimator: Any,
    X: np.ndarray,
) -> np.ndarray:
    """
    Return positive class probabilities from either PyTorch or sklearn
    model.

    Keeps downstream evaluation code model-agnostic so calibration 
    and discrimination metrics can be calculated consistently across
    MLP, logistic regression, and random forest comparators.
    """
    # Convert inputs once to numeric array so both PyTorch and
    # sklearn branches receive same basic data representation.
    X = np.asarray(X, dtype=np.float32)

    # Use predict_proba when it is available on non-PyTorch estimator
    # because comparator models natively expose class probabilities
    # that way.
    if (
        hasattr(estimator, "predict_proba")
        and not isinstance(estimator, nn.Module)
    ):
        probs = np.asarray(estimator.predict_proba(X), dtype=float)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        return probs.reshape(-1)
    
    # Otherwise treat model as PyTorch network and convert logits to
    # probabilities with logistic sigmoid.
    if isinstance(estimator, nn.Module):
        estimator.eval()
        with torch.no_grad():
            logits = estimator(torch.tensor(X, dtype=torch.float32))
        return (
            torch.sigmoid(logits)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    
    # Raise clear error when supplied estimator does not match either
    # supported prediction path.
    raise TypeError(
        "Unsupported model type for probability prediction."
    )

@torch.no_grad()
def predict_proba(estimator: nn.Module, X: np.ndarray) -> np.ndarray:
    """
    Predict positive-class probabilities for PyTorch model.

    Provides consistent interface for obtaining probability outputs
    from trained MLP. Delegates actual computation to shared
    probability utility so same logic can be reused across model
    evaluation and comparison steps.
    """
    # Delegate probability computation to shared helper so same logic
    # is used for both PyTorch and comparator models.
    return _predict_positive_proba_any(estimator, X)

def predict_labels(
    estimator: nn.Module, 
    X: np.ndarray, 
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Predict binary class labels from probabilities.

    Decision threshold remains configurable so same helper can
    support different classification cutoffs when needed.
    """
    # Apply chosen decision threshold to positive class probabilities
    # returned by predict_proba.
    return (predict_proba(estimator, X) >= float(threshold)).astype(int)

def expected_calibration_error(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    n_bins: int = 10,
) -> float:
    """
    Estimate simple expected calibration error from predicted
    probabilities.

    Summarises how closely predicted probabilities align with
    observed outcome frequencies across set of probability bins.
    Included so model comparison is not based on discrimination alone.
    """
    # Convert both inputs to clean one dimensional arrays so binning
    # logic works on stable numeric representation.
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.clip(
        np.asarray(y_prob, dtype=float).reshape(-1),
        1e-7,
        1 - 1e-7,
    )
    
    # Split probability range into fixed bins so calibration can
    # be compared across low and high confidence predictions.
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0

    # Accumulate weighted calibration gap across non-empty bins
    # so final score reflects both error size and bin coverage.
    for i in range(int(n_bins)):
        left = bins[i]
        right = bins[i + 1]
        if i == int(n_bins) - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return float(ece)

def evaluate_model_metrics(
    estimator: Any, 
    X: np.ndarray, 
    y_true: np.ndarray,
) -> Dict[str, float]:
    """
    Compute main held-out evaluation metrics for one candidate model.

    Returned metrics include discrimination and calibration summaries
    so model comparison does not rely on ROC-AUC alone when selecting
    or contextualising explanation model.
    """
    # Standardise labels as one dimensional array so metric functions 
    # operate on consistent input format.
    y_true = np.asarray(y_true).reshape(-1)

    # Ensure both classes are present so discrimination metrics
    # such as ROC-AUC can be computed without failure.
    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "Evaluation requires at least two classes in y_true."
        )
    
    # Generate predicted probabilities using shared helper so
    # evaluation logic remains consistent across PyTorch and
    # sklearn based models.
    probs = np.clip(
        _predict_positive_proba_any(estimator, X), 
        1e-7, 
        1 - 1e-7,
    )

    # Compute discrimination and calibration metrics together so
    # each model can be assessed on ranking quality and probability
    # reliability.
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "average_precision": float(
            average_precision_score(y_true, probs)
        ),
        "brier_score": float(brier_score_loss(y_true, probs)),
        "ece": float(
            expected_calibration_error(y_true, probs, n_bins=10)
        ),
    }

def evaluate_auc(
    estimator: Any,
    X: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    Evaluate ROC-AUC on dataset to measure ranking quality across
    all thresholds.

    Provides lightweight interface when only discrimination
    performance is required, while reusing shared evaluation
    logic to ensure consistency with full metric set.
    """
    # Delegate to main evaluation helper so ROC-AUC is computed
    # using same probability predictions and preprocessing steps.
    return float(evaluate_model_metrics(estimator, X, y_true)["roc_auc"])

def tune_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    search_space: Dict[str, Iterable[Any]] | None = None,
    random_state: Optional[int] = 42,
    epochs: int = 50,
    val_frac: float = 0.15,
    patience: int = 10,
    min_delta: float = 1e-4,
    use_class_weighting: bool = True,
    use_weighted_sampler: bool = True,
    cv_folds: int = 3,
) -> tuple[nn.Module, dict, list[dict]]:
    """
    Tune baseline MLP over compact, predefined search space.

    Returns refit benchmark model, final parameter set, and full list
    of candidate validation results for later reporting.
    """
    # Validate once at start so each candidate loop can focus on
    # fitting and scoring rather than repeating same input checks.
    _validate_training_inputs(X_train, y_train)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    space = search_space or DEFAULT_MLP_SEARCH_SPACE

    # Track mean cross validation score for each candidate setting so
    # final parameter comparison is based on repeated validation folds
    # rather than one split.
    results: list[dict] = []
    best_auc = -np.inf
    best_params: dict[str, Any] | None = None

    # Evaluate each candidate configuration across stratified folds
    # so class balance is preserved more consistently during
    # comparative tuning.
    for hidden_dim, lr, weight_decay, batch_size in product(
        tuple(space.get("hidden_dim", (64,))),
        tuple(space.get("lr", (1e-3,))),
        tuple(space.get("weight_decay", (1e-4,))),
        tuple(space.get("batch_size", (64,))),
    ):
        # Collect one mtric dictionary per fold so candidate can later
        # be summarised by mean cross validation performance.
        fold_metrics: list[dict] = []

        # Refit model on each fold's training partition and score it
        # on corresponding validation partition.
        for fold_idx, (fit_idx, val_idx) in enumerate(
            _iter_cv_splits(y_train, cv_folds, random_state)
        ):
            # Train one fold specific MLP using current candidate
            # settings so its validation performance can contribute to
            # cross-validation summary.
            fold_model = train_model(
                X_train=X_train[fit_idx],
                y_train=y_train[fit_idx],
                epochs=int(epochs),
                lr=float(lr),
                hidden_dim=int(hidden_dim),
                batch_size=int(batch_size),
                val_frac=float(val_frac),
                patience=int(patience),
                min_delta=float(min_delta),
                weight_decay=float(weight_decay),
                use_class_weighting=bool(use_class_weighting),
                use_weighted_sampler=bool(use_weighted_sampler),
                random_state=(
                    None 
                    if random_state is None 
                    else int(random_state) + fold_idx
                ),
            )
            fold_metrics.append(
                evaluate_model_metrics(
                    fold_model, 
                    X_train[val_idx],
                    y_train[val_idx],
                )
            )

        # Store one summary row for candidate so full tuning history
        # can be exported and reviewed later.
        row = {
            "Model": "MLP",
            "Candidate": (
                f"MLP(h={int(hidden_dim)}, "
                f"lr={float(lr):.0e}, "
                f"wd={float(weight_decay):.0e}, "
                f"bs={int(batch_size)})"
            ),
            "CV_AUC_Mean": float(
                np.mean([m["roc_auc"] for m in fold_metrics])
            ),
            "CV_AUC_STD": float(
                np.std(
                    [m["roc_auc"] for m in fold_metrics],
                    ddof=0,
                )
            ),
            "CV_AP_Mean": float(
                np.mean(
                    [m["average_precision"] for m in fold_metrics]
                )
            ),
            "CV_Brier_Mean": float(
                np.mean([m["brier_score"] for m in fold_metrics])
            ),
            "CV_ECE_Mean": float(
                np.mean([m["ece"] for m in fold_metrics])
            ),
            "hidden_dim": int(hidden_dim),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "CV_Folds": int(cv_folds),
        }
        results.append(row)
        if row["CV_AUC_Mean"] > best_auc:
            best_auc = row["CV_AUC_Mean"]
            best_params = {
                "hidden_dim": int(hidden_dim),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "batch_size": int(batch_size),
                "cv_auc_mean": float(row["CV_AUC_Mean"]),
                "cv_auc_std": float(row["CV_AUC_STD"]),
                "cv_ece_mean": float(row["CV_ECE_Mean"]),
                "cv_brier_mean": float(row["CV_Brier_Mean"]),
                "cv_ap_mean": float(row["CV_AP_Mean"]),
            }

    if best_params is None:
        raise RuntimeError(
            "MLP tuning did not evalaute any candidate settings."
        )
    
    # Refit selected configuration on full training set so returned
    # model uses all available training rows before main benchmark
    # begins.
    best_model = train_model(
        X_train=X_train,
        y_train=y_train,
        epochs=int(epochs),
        lr=float(best_params["lr"]),
        hidden_dim=int(best_params["hidden_dim"]),
        batch_size=int(best_params["batch_size"]),
        val_frac=float(val_frac),
        patience=int(patience),
        min_delta=float(min_delta),
        weight_decay=float(best_params["weight_decay"]),
        use_class_weighting=bool(use_class_weighting),
        use_weighted_sampler=bool(use_weighted_sampler),
        random_state=random_state,
    )
    return best_model, best_params, results

def tune_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    c_values: Iterable[float] = DEFAULT_LOGISTIC_C_VALUES,
    random_state: Optional[int] = 42,
    max_iter: int = 1000,
    cv_folds: int = 5,
) -> tuple[LogisticRegression, dict, list[dict]]:
    """
    Tune logistic regression across small regularisation grid using
    stratified cross validation.

    Provides simple linear baseline so fixed neural explanation model 
    (MLP) can be interpreted relative to lower-complexity alternative.
    The search is compact to limit runtime while still avoiding 
    reliance on a single untested regularisation setting.
    """
    # Validate inputs first so all candidate models operate on same
    # checked training arrays.
    _validate_training_inputs(X_train, y_train)

    # Convert inputs once to stable numeric formats used by sklearn
    # esimators.
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int).reshape(-1)

    # Track all candidate results and best observed AUC score.
    results: list[dict] = []
    best_auc = -np.inf
    best_params: dict[str, Any] | None = None

    # Evaluate each regularisation setting across stratified folds so
    # performance reflects repeated validation rather than a single
    # split.
    for c_val in tuple(c_values):
        fold_metrics: list[dict] = []

        # Train and evaluate one model per fold using shared CV
        # splitter.
        for fold_idx, (fit_idx, val_idx) in enumerate(
            _iter_cv_splits(y_train, cv_folds, random_state)
        ):
            candidate = train_logistic_regression(
                X_train[fit_idx],
                y_train[fit_idx],
                C=float(c_val),
                max_iter=int(max_iter),
                random_state=(
                    None 
                    if random_state is None 
                    else int(random_state) + fold_idx
                ),
            )

            # Compute full evaluation metrics so each candidate
            # can be compared on discimination and calibration
            # behaviour.
            fold_metrics.append(
                evaluate_model_metrics(
                    candidate,
                    X_train[val_idx],
                    y_train[val_idx],
                )
            )

        # Aggregate fold metrics into single summary row for
        # candidate.
        row = {
            "Model": "Logistic Regression",
            "Candidate": f"LogisticRegression(C={float(c_val):.3g})",
            "CV_AUC_Mean": float(
                np.mean([m["roc_auc"] for m in fold_metrics])
            ),
            "CV_AUC_STD": float(
                np.std([m["roc_auc"] for m in fold_metrics], ddof=0)
            ),
            "CV_AP_Mean": float(
                np.mean(
                    [m["average_precision"] for m in fold_metrics]
                )
            ),
            "CV_Brier_Mean": float(
                np.mean([m["brier_score"] for m in fold_metrics])
            ),
            "CV_ECE_Mean": float(
                np.mean([m["ece"] for m in fold_metrics])
            ),
            "C": float(c_val),
            "CV_Folds": int(cv_folds),
        }
        results.append(row)

        # Update best configuration when mean CV AUC improves.
        if row["CV_AUC_Mean"] > best_auc:
            best_auc = row["CV_AUC_Mean"]
            best_params = {
                "C": float(c_val),
                "cv_auc_mean": float(row["CV_AUC_Mean"]),
                "cv_auc_std": float(row["CV_AUC_STD"]),
                "cv_ece_mean": float(row["CV_ECE_Mean"]),
                "cv_brier_mean": float(row["CV_Brier_Mean"]),
                "cv_ap_mean": float(row["CV_AP_Mean"]),
            }
    
    # Ensure at least one candidate was evaluated successfully.
    if best_params is None:
        raise RuntimeError(
            "Logistic regression tuning did not evaluate "
            "any candidate settings."
        )
    
    # Refit best configuration on full training set so returned
    # model uses all available data before downstream benchmarking.
    best_model = train_logistic_regression(
        X_train,
        y_train,
        C=float(best_params["C"]),
        max_iter=int(max_iter),
        random_state=random_state,
    )

    return best_model, best_params, results

def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    search_space: Dict[str, Iterable[Any]] | None = None,
    random_state: Optional[int] = 42,
    cv_folds: int = 5,
) -> tuple[RandomForestClassifier, dict, list[dict]]:
    """
    Tune random forest comparator across compact hyperparameter grid.

    Adds non-linear tree based comparator so benchmark can place 
    fixed MLP explanation model in broader predictive context without
    greatly increasing runtime. Search space is intentionally limited
    to keep tuning efficient while still exploring key structural
    parameters.
    """
    # Validate inputs so all candidate models use consistent
    # training data.
    _validate_training_inputs(X_train, y_train)

    # Convert arrays to stable numeric formats for sklearn training.
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int).reshape(-1)

    # Use default search space when no custom configuration is
    # supplied.
    space = search_space or DEFAULT_RF_SEARCH_SPACE

    # Track candidate results and best observed AUC score.
    results: list[dict] = []
    best_auc = -np.inf
    best_params: dict[str, Any] | None = None

    # Evaluate each comination of tree-based hyperparameters
    # across CV folds.
    for n_estimators, max_depth, min_samples_leaf in product(
        tuple(space.get("n_estimators", (300,))),
        tuple(space.get("max_depth", (None,))),
        tuple(space.get("min_samples_leaf", (1,))),
    ):
        fold_metrics: list[dict] = []

        # Train and evaluate one forest per fold so performance
        # reflects repeated validation rather than single split.
        for fold_idx, (fit_idx, val_idx) in enumerate(
            _iter_cv_splits(y_train, cv_folds, random_state)
        ):
            candidate = train_random_forest(
                X_train[fit_idx],
                y_train[fit_idx],
                n_estimators=int(n_estimators),
                max_depth=max_depth,
                min_samples_leaf=int(min_samples_leaf),
                random_state=(
                    None 
                    if random_state is None
                    else int(random_state) + fold_idx
                ),
            )

            # Evaluate candiate using shared metric set.
            fold_metrics.append(
                evaluate_model_metrics(
                    candidate,
                    X_train[val_idx],
                    y_train[val_idx],
                )
            )

        # Aggregate fold metrics into summary row for this configuration.
        row = {
            "Model": "Random Forest",
            "Candidate": (
                f"RandomForest(n={int(n_estimators)}, "
                f"depth={max_depth}, "
                f"leaf={int(min_samples_leaf)})"
            ),
            "CV_AUC_Mean": float(
                np.mean([m["roc_auc"] for m in fold_metrics])
            ),
            "CV_AUC_STD": float(
                np.std(
                    [m["roc_auc"] for m in fold_metrics],
                    ddof=0,
                )
            ),
            "CV_AP_Mean": float(
                np.mean([m["average_precision"] for m in fold_metrics])
            ),
            "CV_Brier_Mean": float(
                np.mean([m["brier_score"] for m in fold_metrics])
            ),
            "CV_ECE_Mean": float(
                np.mean([m["ece"] for m in fold_metrics])
            ),
            "n_estimators": int(n_estimators),
            "max_depth": max_depth,
            "min_samples_leaf": int(min_samples_leaf),
            "CV_Folds": int(cv_folds),
        }
        results.append(row)

        # Update best configuration when mean CV AUC improves.
        if row["CV_AUC_Mean"] > best_auc:
            best_auc = row["CV_AUC_Mean"]
            best_params = {
                "n_estimators": int(n_estimators),
                "max_depth": max_depth,
                "min_samples_leaf": int(min_samples_leaf),
                "cv_auc_mean": float(row["CV_AUC_Mean"]),
                "cv_auc_std": float(row["CV_AUC_STD"]),
                "cv_ece_mean": float(row["CV_ECE_Mean"]),
                "cv_brier_mean": float(row["CV_Brier_Mean"]),
                "cv_ap_mean": float(row["CV_AP_Mean"]),
            }

    # Ensure at least one candidate configuration was evaluated.
    if best_params is None:
        raise RuntimeError(
            "Random forest tuning did not evaluate any candidate settings."
        )
    
    # Refit best configuration on full training data so returned
    # model is ready for downstream benchmarking and comparison.
    best_model = train_random_forest(
        X_train,
        y_train,
        n_estimators=int(best_params["n_estimators"]),
        max_depth=best_params["max_depth"],
        min_samples_leaf=int(best_params["min_samples_leaf"]),
        random_state=random_state,
    )

    return best_model, best_params, results

# --------------------------------------------------------------------
# Sklearn-Style Wrapper
# --------------------------------------------------------------------
class TorchWrapper:
    """
    Minimal sklearn-compatible wrapper around trained PyTorch model.

    benchmarking_engine.py uses this wrapper when computing
    permutation importance through sklearn utilities.
    """
    _estimator_type = "classifier"
    
    def __init__(self, estimator: nn.Module) -> None:
        """
        Store the already trained PyTorch model so sklearn-style 
        helpers can call prediction methods through this wrapper.
        """
        self.model = estimator
        self.classes_ = np.array([0, 1])
    
    def fit(
        self,
        _X: np.ndarray,
        _y: np.ndarray | None = None,
    ) -> "TorchWrapper":
        """
        Return the wrapper unchanged for sklearn API compatibility.
        """
        # Wrapped model is already trained before wrapper is created,
        # so this method only preserves expected sklearn interface.
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities in sklearn's two column format.
        """
        # Clip probabilities slightly so downstream utilities avoid
        # exact boundary values at 0 or 1.
        probs = np.clip(
            _predict_positive_proba_any(self.model, X),
            1e-7, 
            1 - 1e-7,
        )

        # Return probabilities as [P(class 0), P(class 1)] so output
        # matches sklearn conventions.
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return binary class predictions through shared helper.
        """
        # Reuse shared prediction helper so label generation stays
        # consistent with rest of module.
        return predict_labels(self.model, X)
    
    def __sklearn_is_fitted__(self) -> bool:
        """
        Indicate that wrapped estimator is ready for sklearn
        utilities.
        """
        return True
    
    def get_params(self, _deep: bool = True) -> dict:
        """
        Return minimal parameter dictionary for sklearn compatibility.
        """
        return {"model": self.model}
    
    def set_params(self, **params) -> "TorchWrapper":
        """
        Set wrapper attributes following sklearn's parameter pattern.
        """
        # Mirror sklearn's parameter setting behaviour so utilities
        # that expect this pattern can still interact with wrapper.
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
# --------------------------------------------------------------------
# Smoke Test
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Run small smoke test when file is executed directly so end to
    # end training path can be checked quickly.
    demo_rng = np.random.default_rng(42)
    X_demo = demo_rng.normal(size=(200, 10)).astype(np.float32)
    y_demo = demo_rng.integers(0, 2, size=200)

    cfg = ModelConfig()
    model = train_model(
        X_demo,
        y_demo,
        epochs=cfg.epochs,
        lr=cfg.lr,
        hidden_dim=cfg.hidden_dim,
        use_weighted_sampler=cfg.use_weighted_sampler,
        random_state=cfg.random_state,
    )
    print(f"Demo ROC-AUC: {evaluate_auc(model, X_demo, y_demo):.3f}")
