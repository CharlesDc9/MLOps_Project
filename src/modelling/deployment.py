### This is a purely theoretical file to test the deployment of the model
### Just running the training.py is enough here

from pathlib import Path

from prefect import serve
from prefect.server.schemas.schedules import CronSchedule
from training import train_model_workflow

# Constants for paths
BASE_PATH = Path("/Users/charlesdecian/Documents/projet_ml/ML-Ops-Project-")
DATA_PATH = BASE_PATH / "abalone.csv"
ARTIFACTS_PATH = BASE_PATH / "src" / "web_service" / "local_objects"

if __name__ == "__main__":
    data_path = DATA_PATH
    artifacts_path = ARTIFACTS_PATH

    train_model_deployment = train_model_workflow.to_deployment(
        name="Model training Deployment Example",
        version="0.1.0",
        tags=["training", "model"],
        schedule=CronSchedule(cron="0 0 * * 0"),  # Weekly
        parameters={
            "filepath": str(data_path),
            "artifacts_filepath": str(artifacts_path),
        },
    )

    serve(train_model_deployment)
