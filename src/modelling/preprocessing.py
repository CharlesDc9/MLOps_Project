from typing import Tuple

import numpy as np
import pandas as pd
from prefect import flow, task


@task(
    name="Encode categorical columns",
    tags=["Encode categorical columns"],
    retries=2,
    retry_delay_seconds=60,
)
def encode_categorical_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns as dummy variables."""
    return pd.get_dummies(df)


@task(
    name="Extract X and y",
    tags=["Extracting X and y"],
    retries=2,
    retry_delay_seconds=60,
)
def extract_x_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Split dataframe between x and y."""
    x = df.drop(columns="Rings")
    y = df["Rings"].values
    return x, y


@flow(name="Preprocess data", log_prints=True)
def process_data_flow(filepath: str) -> pd.DataFrame:
    """Use a flow to preprocess data."""
    df = pd.read_csv(filepath)
    df = encode_categorical_cols(df)
    return extract_x_y(df)
