import argparse
from pathlib import Path

import pandas as pd
from preprocessing import encode_categorical_cols, extract_x_y
from training import create_and_train_model
from utils import get_model_path, save_pickle


def main(trainset_path: Path, n_trees: int) -> None:
    """Train a model using the data at the given path and save the model (pickle)."""
    # Read data
    df = pd.read_csv(trainset_path)
    # Preprocess data
    df = encode_categorical_cols(df)
    x, y = extract_x_y(df)
    # Train model
    model = create_and_train_model(x, y, n_estimators=n_trees)

    # Pickle model
    file_path = get_model_path()
    save_pickle(file_path, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    parser.add_argument(
        "--n_trees", type=int, default=100, help="Number of trees for RF"
    )
    args = parser.parse_args()
    main(args.trainset_path, args.n_trees)
