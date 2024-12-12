import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from utils import load_pickle


def predict(
    x: pd.DataFrame, model: RandomForestRegressor = None, artifacts_filepath: str = None
) -> np.ndarray:
    """Make predictions with a pre-trained RandomForestRegressor."""
    if model is None:
        model = load_pickle(os.path.join(artifacts_filepath, "model.pkl"))

    return model.predict(x)
