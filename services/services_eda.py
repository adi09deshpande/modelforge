import pandas as pd
from pandas.api.types import is_numeric_dtype


# -----------------------------
# TYPE CONVERSION (ML-SAFE)
# -----------------------------
def convert_dtype(df: pd.DataFrame, col: str, dtype: str) -> pd.DataFrame:
    df = df.copy()

    if col not in df.columns:
        raise ValueError(f"Column not found: {col}")

    if dtype == "int":
        # Step 1: force numeric
        numeric = pd.to_numeric(df[col], errors="coerce")

        # Step 2: decide policy (FLOOR)
        numeric = numeric.fillna(pd.NA).apply(
            lambda x: int(x) if pd.notna(x) else pd.NA
        )

        # Step 3: assign as nullable int
        df[col] = numeric.astype("Int64", errors="ignore")

    elif dtype == "float":
        df[col] = pd.to_numeric(df[col], errors="coerce")

    elif dtype == "str":
        df[col] = df[col].astype(str)

    elif dtype == "category":
        df[col] = df[col].astype("category")

    else:
        raise ValueError("Invalid dtype")

    return df


# -----------------------------
# MISSING VALUES
# -----------------------------
def handle_missing(df: pd.DataFrame, strategy: str, custom=None) -> pd.DataFrame:
    df = df.copy()

    if strategy == "drop":
        return df.dropna()

    if strategy in {"mean", "median"}:
        for col in df.columns:
            if is_numeric_dtype(df[col]):
                if strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].median())
        return df

    if strategy == "mode":
        for col in df.columns:
            if df[col].isnull().any():
                modes = df[col].mode(dropna=True)
                if not modes.empty:
                    df[col] = df[col].fillna(modes.iloc[0])
        return df

    if strategy == "custom":
        if custom is None:
            raise ValueError("Custom value must be provided")
        return df.fillna(custom)

    raise ValueError("Invalid missing value strategy")


# -----------------------------
# DUPLICATES
# -----------------------------
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(ignore_index=True).copy()


# -----------------------------
# DROP COLUMNS
# -----------------------------
def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    return df.drop(columns=cols).copy()
