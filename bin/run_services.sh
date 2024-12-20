#!/bin/bash

# Set Prefect API URL
prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api

# Start Prefect server in the background
prefect server start --host 0.0.0.0 --port 4200 &

# Wait for Prefect server to be ready
sleep 10

# Start FastAPI with Uvicorn
cd /src/web_service
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Run deployment.py to set up Prefect deployment
cd /src/modelling
python deployment.py

# Keep container running
tail -f /dev/null
