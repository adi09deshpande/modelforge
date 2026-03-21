import io
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    StratifiedKFold,
    KFold,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# =====================================================
# MODEL REGISTRY
# =====================================================
MODEL_REGISTRY = {
    "Classification": {
        "Logistic Regression":         LogisticRegression,
        "Random Forest Classifier":    RandomForestClassifier,
        "Decision Tree Classifier":    DecisionTreeClassifier,
    },
    "Regression": {
        "Linear Regression":           LinearRegression,
        "Random Forest Regressor":     RandomForestRegressor,
        "Decision Tree Regressor":     DecisionTreeRegressor,
    },
}

# =====================================================
# HYPERPARAMETER SEARCH SPACES
# =====================================================
PARAM_GRID = {
    # ── Classification ─────────────────────────────
    "Logistic Regression": {
        "grid": {
            "C":        [0.01, 0.1, 1.0, 10.0],
            "max_iter": [200, 500, 1000],
            "solver":   ["lbfgs", "liblinear"],
        },
        "random": {
            "C":        [0.001, 0.01, 0.1, 1, 5, 10, 50, 100],
            "max_iter": [100, 200, 500, 1000, 2000],
            "solver":   ["lbfgs", "liblinear", "saga"],
            "penalty":  ["l2"],
        },
    },
    "Random Forest Classifier": {
        "grid": {
            "n_estimators": [50, 100, 200],
            "max_depth":    [None, 5, 10, 20],
            "min_samples_split": [2, 5],
        },
        "random": {
            "n_estimators":      [50, 100, 200, 300, 500],
            "max_depth":         [None, 3, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
            "max_features":      ["sqrt", "log2", None],
        },
    },
    "Decision Tree Classifier": {
        "grid": {
            "max_depth":         [None, 3, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "criterion":         ["gini", "entropy"],
        },
        "random": {
            "max_depth":         [None, 2, 3, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf":  [1, 2, 4, 8],
            "criterion":         ["gini", "entropy"],
            "max_features":      ["sqrt", "log2", None],
        },
    },
    # ── Regression ─────────────────────────────────
    "Linear Regression": {
        "grid":   {},   # No hyperparameters to tune
        "random": {},
    },
    "Random Forest Regressor": {
        "grid": {
            "n_estimators": [50, 100, 200],
            "max_depth":    [None, 5, 10, 20],
            "min_samples_split": [2, 5],
        },
        "random": {
            "n_estimators":      [50, 100, 200, 300, 500],
            "max_depth":         [None, 3, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
            "max_features":      ["sqrt", "log2", None],
        },
    },
    "Decision Tree Regressor": {
        "grid": {
            "max_depth":         [None, 3, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "criterion":         ["squared_error", "absolute_error"],
        },
        "random": {
            "max_depth":         [None, 2, 3, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf":  [1, 2, 4, 8],
            "criterion":         ["squared_error", "absolute_error"],
            "max_features":      ["sqrt", "log2", None],
        },
    },
}

# =====================================================
# SCORING MAP
# =====================================================
SCORING_MAP = {
    "Classification": "f1_weighted",
    "Regression":     "r2",
}


# =====================================================
# SAFE PARAM MERGE
# =====================================================
def build_model(
    problem_type: str,
    model_name: str,
    user_params: Optional[Dict] = None,
):
    if problem_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported problem type: {problem_type}")
    if model_name not in MODEL_REGISTRY[problem_type]:
        raise ValueError(f"Unsupported model: {model_name}")

    model_class = MODEL_REGISTRY[problem_type][model_name]
    default_model = model_class()
    params = default_model.get_params()

    if user_params:
        for k, v in user_params.items():
            if k in params and v is not None:
                params[k] = v

    return model_class(**params)


# =====================================================
# SAFE PARAM EXTRACTION
# =====================================================
def extract_safe_params(model) -> Dict:
    safe = {}
    for k, v in model.get_params().items():
        if isinstance(v, (int, float, str, bool, type(None))):
            safe[k] = v
        else:
            safe[k] = str(v)
    return safe


# =====================================================
# PREPROCESSOR
# =====================================================
def build_preprocessor(
    X: pd.DataFrame,
    encoding: Optional[str],
    scaling: Optional[str],
) -> ColumnTransformer:

    num_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []

    if num_cols:
        if scaling == "standard":
            num_pipeline = Pipeline([("scaler", StandardScaler())])
        elif scaling == "minmax":
            num_pipeline = Pipeline([("scaler", MinMaxScaler())])
        else:
            num_pipeline = "passthrough"
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        if encoding == "onehot":
            cat_pipeline = Pipeline([
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
        elif encoding == "label":
            cat_pipeline = Pipeline([
                ("encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ])
        else:
            cat_pipeline = "passthrough"
        transformers.append(("cat", cat_pipeline, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# =====================================================
# AUTO-TUNING
# =====================================================
def auto_tune(
    model_class,
    param_space: Dict,
    X_train_transformed: np.ndarray,
    y_train,
    problem_type: str,
    method: str,          # "grid" | "random" | "bayesian"
    cv: int,
    random_state: int,
) -> Tuple[object, Dict]:
    """
    Returns (best_estimator, best_params).
    Falls back to default if param_space is empty (e.g. Linear Regression).
    """
    scoring = SCORING_MAP[problem_type]

    if not param_space:
        # Nothing to tune (Linear Regression)
        model = model_class(random_state=random_state) if _has_random_state(model_class) else model_class()
        model.fit(X_train_transformed, y_train)
        return model, {}

    base_model = model_class(random_state=random_state) if _has_random_state(model_class) else model_class()

    if method == "grid":
        searcher = GridSearchCV(
            estimator=base_model,
            param_grid=param_space,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
    elif method == "bayesian":
        # Use RandomizedSearchCV with more iterations as Bayesian proxy
        # (scikit-optimize not always available; this is safe fallback)
        try:
            from skopt import BayesSearchCV
            from skopt.space import Real, Integer, Categorical

            bayes_space = _convert_to_skopt_space(param_space)
            searcher = BayesSearchCV(
                estimator=base_model,
                search_spaces=bayes_space,
                n_iter=30,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                refit=True,
                random_state=random_state,
                verbose=0,
            )
        except ImportError:
            # scikit-optimize not installed → fall back to RandomizedSearchCV
            searcher = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_space,
                n_iter=30,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                refit=True,
                random_state=random_state,
                verbose=0,
            )
    else:
        # Default: random search
        searcher = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_space,
            n_iter=20,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
            random_state=random_state,
            verbose=0,
        )

    searcher.fit(X_train_transformed, y_train)
    return searcher.best_estimator_, searcher.best_params_


def _has_random_state(model_class) -> bool:
    import inspect
    sig = inspect.signature(model_class.__init__)
    return "random_state" in sig.parameters


def _convert_to_skopt_space(param_grid: Dict):
    """Convert plain lists to skopt search spaces."""
    from skopt.space import Categorical
    return {k: Categorical(v) for k, v in param_grid.items()}


# =====================================================
# CROSS-VALIDATION SCORES
# =====================================================
def run_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y,
    problem_type: str,
    cv: int,
    random_state: int,
) -> Dict:
    """
    Run k-fold CV on the FULL dataset and return per-fold + mean/std scores.
    """
    if problem_type == "Classification":
        y = y.astype(str)
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scorers = {
            "accuracy":  "accuracy",
            "f1":        "f1_weighted",
            "precision": "precision_weighted",
            "recall":    "recall_weighted",
        }
    else:
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        scorers = {
            "r2":   "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae":  "neg_mean_absolute_error",
        }

    cv_results = {}
    for metric_name, scorer in scorers.items():
        scores = cross_val_score(pipeline, X, y, cv=kf, scoring=scorer, n_jobs=-1)

        # Flip negative scorers
        if scorer.startswith("neg_"):
            scores = -scores

        cv_results[metric_name] = {
            "scores": [round(float(s), 4) for s in scores],
            "mean":   round(float(scores.mean()), 4),
            "std":    round(float(scores.std()), 4),
        }

    return cv_results


# =====================================================
# MAIN TRAIN + EVALUATE
# =====================================================
def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    model_name: str,
    test_size: float,
    random_state: int,
    encoding: Optional[str],
    scaling: Optional[str],
    stratify: bool,
    user_params: Optional[Dict] = None,

    # ── NEW parameters ──────────────────────────────
    tuning_method: str = "manual",   # manual | grid | random | bayesian
    cv_folds: int = 0,               # 0 = disabled, 3/5/10 = enabled
) -> Tuple[bytes, Dict, Dict]:

    # ── Target safety ─────────────────────────────────────────────────────
    if problem_type == "Classification":
        y = y.astype(str)
        value_counts = y.value_counts()
        if stratify and (value_counts < 2).any():
            stratify = False

    strat = y if stratify and problem_type == "Classification" else None

    # ── Split ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    # ── Preprocessor ──────────────────────────────────────────────────────
    preprocessor = build_preprocessor(X_train, encoding, scaling)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed  = preprocessor.transform(X_test)

    # ── Model selection & tuning ───────────────────────────────────────────
    model_class = MODEL_REGISTRY[problem_type][model_name]
    best_params: Dict = {}

    if tuning_method == "manual":
        model = build_model(problem_type, model_name, user_params)
        model.fit(X_train_transformed, y_train)

    else:
        # Auto-tuning: grid / random / bayesian
        space_key = "grid" if tuning_method == "grid" else "random"
        param_space = PARAM_GRID.get(model_name, {}).get(space_key, {})

        cv_for_tuning = max(cv_folds, 3)   # at least 3-fold for tuning

        model, best_params = auto_tune(
            model_class=model_class,
            param_space=param_space,
            X_train_transformed=X_train_transformed,
            y_train=y_train,
            problem_type=problem_type,
            method=tuning_method,
            cv=cv_for_tuning,
            random_state=random_state,
        )

    # ── Full pipeline (needed for CV + serialisation) ──────────────────────
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model",      model),
    ])
    # Preprocessor already fitted; refit=False not available in Pipeline
    # So we rebuild with the TUNED model and fit on train set only:
    final_pipeline = Pipeline(steps=[
        ("preprocess", build_preprocessor(X_train, encoding, scaling)),
        ("model",      model),
    ])
    final_pipeline.fit(X_train, y_train)

    y_pred = final_pipeline.predict(X_test)

    # ── Metrics ────────────────────────────────────────────────────────────
    if problem_type == "Classification":
        metrics = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        unique_classes = np.unique(y_test)
        if (
            len(unique_classes) == 2
            and hasattr(final_pipeline.named_steps["model"], "predict_proba")
        ):
            y_test_bin = (y_test == unique_classes[1]).astype(int)
            y_prob = final_pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
            metrics["roc_auc"]   = float(roc_auc_score(y_test_bin, y_prob))
            metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    else:
        metrics = {
            "r2":   float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae":  float(mean_absolute_error(y_test, y_pred)),
        }

    # ── Best params from auto-tuning ───────────────────────────────────────
    if best_params:
        metrics["best_params"] = {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in best_params.items()
        }

    # ── Cross-validation ───────────────────────────────────────────────────
    if cv_folds >= 3:
        cv_results = run_cross_validation(
            pipeline=final_pipeline,
            X=X,
            y=y,
            problem_type=problem_type,
            cv=cv_folds,
            random_state=random_state,
        )
        metrics["cross_validation"] = cv_results

    # ── Serialize ──────────────────────────────────────────────────────────
    buffer = io.BytesIO()
    joblib.dump(final_pipeline, buffer)

    params = extract_safe_params(model)

    return buffer.getvalue(), params, metrics
