# services/services_params.py
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# =====================================================
# DEFAULT PARAMETER MAP
# =====================================================
MODEL_PARAM_MAP = {
    "Logistic Regression": LogisticRegression,
    "Linear Regression": LinearRegression,
    "Random Forest Classifier": RandomForestClassifier,
    "Random Forest Regressor": RandomForestRegressor,
    "Decision Tree Classifier": DecisionTreeClassifier,
    "Decision Tree Regressor": DecisionTreeRegressor,
}


# =====================================================
# GET DEFAULT PARAMS FROM SKLEARN
# =====================================================
def get_default_params(model_name: str) -> Dict:
    if model_name not in MODEL_PARAM_MAP:
        raise ValueError(f"Unknown model: {model_name}")

    model_cls = MODEL_PARAM_MAP[model_name]
    return model_cls().get_params()


# =====================================================
# MERGE USER PARAMS SAFELY
# =====================================================
def resolve_model_params(model_name: str, user_params: Optional[Dict]) -> Dict:
    defaults = get_default_params(model_name)

    if not user_params:
        return defaults

    resolved = defaults.copy()
    for k, v in user_params.items():
        if k in resolved:
            try:
                resolved[k] = type(resolved[k])(v)
            except Exception:
                pass  # fallback to default

    return resolved

