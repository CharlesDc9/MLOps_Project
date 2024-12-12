import pickle
from pathlib import Path
from typing import Any


def load_pickle(path: str):
    """Load pickle object."""
    with open(path, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


def save_pickle(path: str, obj: Any):
    """Save pickle object."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def get_model_path():
    base_dir = Path(
        "/Users/charlesdecian/Documents/projet_ml/ML-Ops-Project-/src/web_service/local_objects"
    )
    return base_dir / "model.pkl"
