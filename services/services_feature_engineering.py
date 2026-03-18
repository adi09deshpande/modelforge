import pandas as pd
import numpy as np

# =====================================================
# HELPERS
# =====================================================

def _to_numeric(series: pd.Series) -> pd.Series:
    """Safely coerce to numeric"""
    return pd.to_numeric(series, errors="coerce")


# =====================================================
# NUMERIC FEATURE
# =====================================================
def create_numeric_feature(df, col1, col2, operation, new_name):
    df = df.copy()

    a = _to_numeric(df[col1])
    b = _to_numeric(df[col2]) if col2 else None

    if operation == "sum":
        df[new_name] = a + (0 if b is None else b)

    elif operation == "diff":
        df[new_name] = a - (0 if b is None else b)

    elif operation == "product":
        df[new_name] = a * (1 if b is None else b)

    elif operation == "ratio":
        denom = (1 if b is None else b).replace(0, np.nan)
        df[new_name] = a / denom

    else:
        raise ValueError("Invalid numeric operation")

    return df


# =====================================================
# NUMERIC TRANSFORMS
# =====================================================
def transform_numeric(df, column, transform, new_name, power=None, bins=None):
    df = df.copy()
    s = _to_numeric(df[column])

    if transform == "log":
        df[new_name] = np.log1p(s.clip(lower=0))

    elif transform == "square":
        df[new_name] = s ** 2

    elif transform == "sqrt":
        df[new_name] = np.sqrt(s.clip(lower=0))

    elif transform == "power":
        df[new_name] = s ** (power or 2)

    elif transform == "bin":
        df[new_name] = pd.cut(s, bins=bins or 5, labels=False)

    else:
        raise ValueError("Invalid numeric transform")

    return df


# =====================================================
# DATE FEATURES
# =====================================================
def extract_date_features(df, column, features, prefix, keep_original):
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce")
    dt = df[column].dt

    for f in features:
        name = f"{prefix}_{f.lower()}"

        if f == "year":
            df[name] = dt.year
        elif f == "month":
            df[name] = dt.month
        elif f == "day":
            df[name] = dt.day
        elif f == "weekday":
            df[name] = dt.weekday
        elif f == "hour":
            df[name] = dt.hour
        elif f == "minute":
            df[name] = dt.minute
        elif f == "second":
            df[name] = dt.second
        elif f == "quarter":
            df[name] = dt.quarter

    if not keep_original:
        df.drop(columns=[column], inplace=True)

    return df


# =====================================================
# AGE FEATURE
# =====================================================
def create_age_feature(df, dob_column, new_name, keep_dob):
    df = df.copy()

    dob = pd.to_datetime(df[dob_column], errors="coerce")
    today = pd.Timestamp.today()

    df[new_name] = ((today - dob).dt.days / 365).astype("Int64")

    if not keep_dob:
        df.drop(columns=[dob_column], inplace=True)

    return df
