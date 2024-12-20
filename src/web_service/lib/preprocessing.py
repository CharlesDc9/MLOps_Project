from typing import Optional, Tuple

import numpy as np
import pandas as pd


def encode_categorical_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns as dummy variables."""
    return pd.get_dummies(df)


def extract_x_y(
    df: pd.DataFrame, training: bool = False
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Split dataframe between x and y."""
    if training:
        x = df.drop(columns="Rings")
        y = df["Rings"].values
        return x, y
    return df, None
