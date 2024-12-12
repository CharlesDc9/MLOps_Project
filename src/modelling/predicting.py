from pathlib import Path

import numpy as np
import pandas as pd
from prefect import task
from sklearn.ensemble import RandomForestRegressor
from utils import load_pickle


@task(name="Predict rings", tags=["prediction"])
def predict(
    x: pd.DataFrame, model: RandomForestRegressor = None, artifacts_filepath: str = None
) -> np.ndarray:
    """Make predictions with a pre-trained RandomForestRegressor."""
    if model is None:
        model = load_pickle(Path(artifacts_filepath) / "model.pkl")

    return model.predict(x)
