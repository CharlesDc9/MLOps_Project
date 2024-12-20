from typing import List

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator

from .models import AgeInput
from .preprocessing import encode_categorical_cols, extract_x_y


def run_inference(model: BaseEstimator, inputs: List[AgeInput]) -> int:
    """Run inference on a list of AgeInput objects."""
    logger.info("Running inference")
    df = pd.DataFrame([input.__dict__ for input in inputs])
    # Define column mapping to match training data
    column_mapping = {
        "Whole_weight": "Whole weight",
        "Shucked_weight": "Shucked weight",
        "Viscera_weight": "Viscera weight",
        "Shell_weight": "Shell weight",
    }

    # Rename columns
    df = df.rename(columns=column_mapping)

    df = encode_categorical_cols(df)
    x, _ = extract_x_y(df)
    y = model.predict(x)
    logger.info(f"Predictions: {y}")
    return round(y[0])
