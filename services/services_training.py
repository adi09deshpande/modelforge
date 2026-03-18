import io
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from sklearn.model_selection import train_test_split
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
        "Logistic Regression": LogisticRegression,
        "Random Forest Classifier": RandomForestClassifier,
        "Decision Tree Classifier": DecisionTreeClassifier,
    },
    "Regression": {
        "Linear Regression": LinearRegression,
        "Random Forest Regressor": RandomForestRegressor,
        "Decision Tree Regressor": DecisionTreeRegressor,
    },
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
            cat_pipeline = Pipeline(
                [
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse=False,
                        ),
                    )
                ]
            )
        elif encoding == "label":
            cat_pipeline = Pipeline(
                [
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    )
                ]
            )
        else:
            cat_pipeline = "passthrough"

        transformers.append(("cat", cat_pipeline, cat_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# =====================================================
# TRAIN + EVALUATE (BACKGROUND SAFE)
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
) -> Tuple[bytes, Dict, Dict]:

    # -------------------------------------------------
    # TARGET SAFETY (CRITICAL FIX)
    # -------------------------------------------------
    if problem_type == "Classification":
        y = y.astype(str)

        # Disable stratify if class counts < 2
        value_counts = y.value_counts()
        if stratify and (value_counts < 2).any():
            stratify = False

    strat = y if stratify and problem_type == "Classification" else None

    # -------------------------------------------------
    # SPLIT
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    # -------------------------------------------------
    # PIPELINE
    # -------------------------------------------------
    preprocessor = build_preprocessor(X_train, encoding, scaling)
    model = build_model(problem_type, model_name, user_params)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # -------------------------------------------------
    # TRAIN
    # -------------------------------------------------
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------
    if problem_type == "Classification":
        metrics = classification_report(
            y_test,
            y_pred,
            output_dict=True,
        )

        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # ROC + AUC (binary only)
        unique_classes = np.unique(y_test)

        if (
            len(unique_classes) == 2
            and hasattr(pipeline.named_steps["model"], "predict_proba")
        ):
            # Convert labels to 0/1 for ROC safety
            y_test_bin = (y_test == unique_classes[1]).astype(int)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
            auc_score = roc_auc_score(y_test_bin, y_prob)

            metrics["roc_auc"] = float(auc_score)
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            }

    else:
        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        }

    # -------------------------------------------------
    # SERIALIZE PIPELINE
    # -------------------------------------------------
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)

    params = extract_safe_params(model)

    return buffer.getvalue(), params, metrics
