#!/bin/bash

echo "Running GCP Pose Estimation Inference..."

python -m src.inference \
    --config configs/config.yaml

echo "Inference complete. Output saved to predictions.json"