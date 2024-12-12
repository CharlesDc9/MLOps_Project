from pathlib import Path

from app_config import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
)
from fastapi import FastAPI
from lib.inference import run_inference
from lib.models import AgeInput, AgeOutput
from lib.utils import load_model

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)

# Define paths as constants
BASE_PATH = Path("/Users/charlesdecian/Documents/projet_ml/ML-Ops-Project-")
MODEL_PATH = BASE_PATH / "src" / "web_service" / "local_objects" / "model.pkl"


@app.get("/")
def home() -> dict:
    """Home route."""
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=AgeOutput, status_code=201)
def predict(payload: AgeInput) -> dict:
    """Predict the age of an abalone shellfish."""
    print("Current working directory:", Path.cwd())

    # Check if model file exists
    if MODEL_PATH.exists():
        print(f"File '{MODEL_PATH}' exists.")
    else:
        print(f"File '{MODEL_PATH}' does not exist.")

    model = load_model(MODEL_PATH)
    y = run_inference(model, [payload])
    return {"age": y}
