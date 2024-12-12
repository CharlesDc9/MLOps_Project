import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def create_and_train_model(
    x: pd.DataFrame, y: np.ndarray, **params
) -> RandomForestRegressor:
    """Train and return a RandomForestRegressor model with parameters **params."""
    if params is None:
        params = {}
    return RandomForestRegressor(**params).fit(x, y)
