# AI Image Recognition for Quality Control

A beginner-friendly Python project for exploring automated visual quality inspection in manufacturing. The Streamlit dashboard accepts a product image, checks basic image-capture indicators, and creates a downloadable CSV report. An optional transfer-learning script is included for training a Good-vs-Defective image classifier with labelled data.

## Features

- Upload product images (PNG, JPG, JPEG, WEBP, BMP)
- View image dimensions and simple brightness, sharpness, and edge-density indicators
- Adjust screening thresholds
- Download an inspection report as CSV
- Optional MobileNetV2 transfer-learning training workflow
- Streamlit dashboard suitable for deployment on Streamlit Community Cloud

## Important limitation

The default dashboard checks image-capture quality; it does **not** reliably detect manufacturing defects such as scratches or cracks. The optional classifier requires your own correctly labelled images and must be evaluated on unseen examples before any real-world use.

## Run locally on macOS

1. Install Python 3.10 or newer.
2. Open this folder in VS Code and open Terminal.
3. Run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app.py
```

Streamlit will show a local URL, usually `http://localhost:8501`.

## Train the optional deep-learning model

Put images into the following folders (the folder names are the labels):

```text
data/
  train/
    Good/
    Defective/
  validation/
    Good/
    Defective/
```

Use a representative set of images in all four folders. Validation images should be separate from training images.

Then run:

```bash
pip install -r requirements-training.txt
python train_model.py
```

This creates `model.keras`. The current dashboard remains a visual-inspection prototype; integrating predictions into the dashboard and validating them for your product line is a further development step.

## Deploy a live dashboard

1. Push this repository to GitHub.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**.
4. Choose this repository, branch `main`, and main file path `app.py` (if these files are at the repository root).
5. Click **Deploy** and wait for the app URL.

## Project structure

```text
Image_Recognition_Quality_Control/
├── app.py
├── train_model.py
├── requirements.txt
├── requirements-training.txt
├── README.md
├── PROJECT_REPORT.md
├── .gitignore
└── data/
    ├── train/
    │   ├── Good/
    │   └── Defective/
    └── validation/
        ├── Good/
        └── Defective/
```
