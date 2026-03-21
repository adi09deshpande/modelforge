"""
Feature Selection service for ModelForge.
Supports: Feature Importance, Correlation Threshold, RFE.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder


# =====================================================
# HELPERS
# =====================================================

def _encode_target(y: pd.Series) -> pd.Series:
    """Encode string targets for tree models."""
    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        return pd.Series(le.fit_transform(y.astype(str)), index=y.index)
    return y


def _safe_X(X: pd.DataFrame) -> pd.DataFrame:
    """Drop non-numeric columns and fill NaN for model fitting."""
    X_num = X.select_dtypes(include=["int", "float"]).copy()
    X_num = X_num.fillna(X_num.median())
    return X_num


# =====================================================
# METHOD 1: FEATURE IMPORTANCE (TREE-BASED)
# =====================================================
def feature_importance_selection(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    top_n: int = 10,
    threshold: float = 0.01,
) -> Dict:
    """
    Use Random Forest feature importances to rank and select features.
    Returns ranked features with importance scores.
    """
    X_num = _safe_X(X)
    y_enc = _encode_target(y)

    if problem_type == "Classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_num, y_enc)

    importance_scores = model.feature_importances_
    feature_names = X_num.columns.tolist()

    ranked = sorted(
        zip(feature_names, importance_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # Apply threshold + top_n
    selected = [
        {"feature": f, "importance": round(float(imp), 6)}
        for f, imp in ranked
        if imp >= threshold
    ][:top_n]

    removed = [
        {"feature": f, "importance": round(float(imp), 6)}
        for f, imp in ranked
        if imp < threshold or ranked.index((f, imp)) >= top_n
    ]

    return {
        "method": "feature_importance",
        "selected": selected,
        "removed": removed,
        "selected_features": [s["feature"] for s in selected],
        "threshold_used": threshold,
        "top_n": top_n,
    }


# =====================================================
# METHOD 2: CORRELATION THRESHOLD
# =====================================================
def correlation_selection(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    correlation_threshold: float = 0.9,
    target_correlation_min: float = 0.0,
) -> Dict:
    """
    Remove features that are highly correlated with each other.
    Optionally keep only features that correlate with the target.
    """
    X_num = _safe_X(X)

    # Correlation matrix
    corr_matrix = X_num.corr().abs()

    # Upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Features to drop (correlated with another feature above threshold)
    to_drop = [
        col for col in upper.columns
        if any(upper[col] > correlation_threshold)
    ]

    selected_features = [f for f in X_num.columns if f not in to_drop]

    # Correlation with target (for numeric targets)
    target_corr = {}
    try:
        y_num = pd.to_numeric(y, errors="coerce")
        if y_num.notna().sum() > 0:
            for col in X_num.columns:
                corr_val = X_num[col].corr(y_num)
                if not np.isnan(corr_val):
                    target_corr[col] = round(float(abs(corr_val)), 4)
    except Exception:
        pass

    # Build detailed results
    selected = []
    removed = []

    for f in X_num.columns:
        info = {
            "feature": f,
            "target_correlation": target_corr.get(f, None),
        }
        if f in to_drop:
            removed.append(info)
        else:
            selected.append(info)

    return {
        "method": "correlation",
        "selected": selected,
        "removed": removed,
        "selected_features": selected_features,
        "correlation_threshold": correlation_threshold,
        "target_correlations": target_corr,
    }


# =====================================================
# METHOD 3: RECURSIVE FEATURE ELIMINATION (RFE)
# =====================================================
def rfe_selection(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    n_features: int = 5,
) -> Dict:
    """
    Use RFE with Logistic/Linear Regression as estimator.
    Returns ranked features with support and ranking.
    """
    X_num = _safe_X(X)
    y_enc = _encode_target(y)

    n_features = min(n_features, X_num.shape[1])

    if problem_type == "Classification":
        estimator = LogisticRegression(max_iter=1000, random_state=42)
    else:
        estimator = LinearRegression()

    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X_num, y_enc)

    feature_names = X_num.columns.tolist()
    support = rfe.support_
    ranking = rfe.ranking_

    selected = [
        {"feature": f, "rfe_rank": int(r)}
        for f, s, r in zip(feature_names, support, ranking)
        if s
    ]
    removed = [
        {"feature": f, "rfe_rank": int(r)}
        for f, s, r in zip(feature_names, support, ranking)
        if not s
    ]

    # Sort by rank
    selected.sort(key=lambda x: x["rfe_rank"])
    removed.sort(key=lambda x: x["rfe_rank"])

    return {
        "method": "rfe",
        "selected": selected,
        "removed": removed,
        "selected_features": [s["feature"] for s in selected],
        "n_features_selected": n_features,
    }


# =====================================================
# COMBINED RUNNER
# =====================================================
def run_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    method: str,
    top_n: int = 10,
    importance_threshold: float = 0.01,
    correlation_threshold: float = 0.9,
    n_rfe_features: int = 5,
) -> Dict:
    """
    Single entry point for all feature selection methods.
    method: "importance" | "correlation" | "rfe"
    """
    if method == "importance":
        return feature_importance_selection(
            X, y, problem_type,
            top_n=top_n,
            threshold=importance_threshold,
        )
    elif method == "correlation":
        return correlation_selection(
            X, y, problem_type,
            correlation_threshold=correlation_threshold,
        )
    elif method == "rfe":
        return rfe_selection(
            X, y, problem_type,
            n_features=n_rfe_features,
        )
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
