import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


# =====================================================
# TRAIN–TEST SPLIT
# =====================================================
def split_dataset(
    df: pd.DataFrame,
    X_cols: List[str],
    y_col: str,
    test_size: float = 0.2,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """ 
    Split dataset into train and test sets.
    """

    X = df[X_cols].copy()
    y = df[y_col].copy()

    strat = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=strat,
    )

    return X_train, X_test, y_train, y_test


# =====================================================
# ENCODING
# =====================================================
def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str,
) -> pd.DataFrame:
    """
    Encode categorical columns.

    method:
        - "label"
        - "onehot"
    """

    df = df.copy()

    if method == "label":
        le = LabelEncoder()
        for c in columns:
            df[c] = le.fit_transform(df[c].astype(str))

    elif method == "onehot":
        df = pd.get_dummies(df, columns=columns, drop_first=True)

    else:
        raise ValueError("Invalid encoding method")

    return df


# =====================================================
# SCALING
# =====================================================
def scale_numeric(
    df: pd.DataFrame,
    columns: List[str],
    method: str,
) -> pd.DataFrame:
    """
    Scale numeric columns.

    method:
        - "standard"
        - "minmax"
    """

    df = df.copy()

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling method")

    df[columns] = scaler.fit_transform(df[columns])
    return df
