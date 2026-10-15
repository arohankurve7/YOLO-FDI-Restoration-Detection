# YOLO-FDI-Restoration-Detection

AI-assisted panoramic dental X-ray analysis using YOLO instance segmentation, FDI tooth numbering, and restoration detection.

## Overview

This research prototype analyzes panoramic dental radiographs (OPGs) to identify individual tooth instances, assign FDI tooth numbers, estimate tooth presence or absence, and visualize findings through an interactive Flask dashboard.

A separate YOLO-based restoration detection module supports identification of dental restorations.

## Features

- YOLO-based tooth instance segmentation
- Anatomical arch separation and physical tooth-instance cleanup
- Missing-aware FDI tooth numbering (ISO 3950)
- Structured 32-position permanent dentition registry
- Tooth presence and missing-position analysis
- Interactive anatomical tooth chart
- Tooth-wise findings and zoomed visualization
- Standalone restoration detection for implants, bridges, crowns, and dental fillings
- Evaluation scripts for tooth and restoration predictions

## Processing Architecture

Panoramic OPG
    |
    v
YOLO Tooth Instance Segmentation
    |
    v
Instance Cleanup and Anatomical Resolution
    |
    v
Missing-Aware FDI Assignment
    |
    v
32-Position Tooth Registry
    |
    v
Interactive Flask Dashboard

Restoration Detection:
Panoramic OPG -> YOLO Restoration Segmentation -> Restoration Predictions and Overlays

## Requirements

- Python 3.10
- Ultralytics YOLO
- PyTorch
- OpenCV
- NumPy
- Flask

Install dependencies:

    pip install -r requirements.txt

## Model Weights

Trained model weights are not included in this repository.

Expected local paths:

Tooth segmentation:
    models/fdi_seg_v1/best.pt

Restoration segmentation:
    runs/segment/runs/restorations_seg_v1/weights/best.pt

These files must be supplied locally before running the corresponding inference modules.

## Run the FDI Dashboard

From the project root:

    python -m src.inference.fdi_dashboard_app

Open:

    http://127.0.0.1:5000

Upload a panoramic OPG image to visualize tooth segmentation, FDI numbering, tooth presence, and tooth-wise findings.

## Run Restoration Detection

From the project root:

    python -m src.inference.restoration_inference --image path/to/opg.png

Optional arguments:

    --model PATH
    --conf FLOAT

Restoration predictions and overlays are generated in the local results directory.

## Repository Structure

    configs/                 Model configuration
    scripts/                 Utility scripts
    src/assets/              Tooth-chart SVG assets
    src/inference/           FDI and restoration inference
    src/evaluation/          Evaluation utilities
    src/segmentation_demo/   Segmentation experiments and evaluation

## Current Limitations

- Restoration inference is currently separate from the Flask FDI dashboard.
- Model performance depends on radiograph quality and training-data coverage.
- Predicted missing tooth positions require clinical verification.
- Trained model weights and medical-image datasets are excluded from GitHub.

## Research Disclaimer

This software is an experimental research prototype for AI-assisted dental image analysis. It is not a certified medical device and must not be used as a substitute for professional dental diagnosis.
