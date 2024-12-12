from typing import Tuple

import numpy as np
import pandas as pd


def encode_categorical_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns as dummy variables."""
    return pd.get_dummies(df)


def extract_x_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Split dataframe between x and y."""
    x = df.drop(columns="Rings")
    y = df["Rings"].values
    return x, y
