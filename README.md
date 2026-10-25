# Image Recognition for Quality Control 🔍

## About the Project

In manufacturing, checking product quality is important because even small defects can affect the final product. Manual inspection can take time, so I created this Python project to explore how image processing can help with quality control.

This project uses image analysis to check basic quality indicators such as brightness, sharpness, and possible blurriness. It provides a simple Streamlit dashboard where users can upload a product image and review the inspection results.

I built this project as part of my learning journey in Python, Artificial Intelligence, and Machine Learning.

🚀 Live Dashboard

Try the application here:
image-recognition-quality-control-zscgg4n7gbxk64ifvwhtmr

https://image-recognition-quality-control-zscgg4n7gbxk64ifvwhthmr.streamlit.app/

## What Can You Do With It?

- Upload a product image for inspection.
- Check whether an image may be blurry.
- Identify images that may be too dark or overexposed.
- View image-quality measurements and inspection results.
- Get a simple status indicating whether the image needs further review.

## Technologies Used

- **Python** — main programming language
- **Streamlit** — interactive web dashboard
- **Pillow (PIL)** — image loading and processing
- **NumPy** — numerical calculations
- **scikit-image** — image filtering and edge analysis

## Project Structure

```text
Image_Recognition_Quality_Control/
├── app.py
├── train_model.py
├── requirements.txt
├── requirements-training.txt
├── PROJECT_REPORT.md
├── README.md
└── data/
    ├── train/
    │   ├── Good/
    │   └── Defective/
    └── validation/
        ├── Good/
        └── Defective/
```

The `data` folders are intended for product images. Add suitable images to these folders when preparing a dataset for model training.

## Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/nirajyadav-cse/Image-Recognition-Quality-Controls.git
cd Image-Recognition-Quality-Controls
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Start the dashboard

```bash
python3 -m streamlit run app.py
```

Streamlit will display a local address, usually `http://localhost:8501`. Open that address in your browser.

## Important Note

The current image inspection features use basic image-processing measurements. Brightness and sharpness checks can help flag images for review, but they do not prove that a physical product is defective. Reliable defect classification requires a suitable labelled dataset and a trained and tested machine-learning model.

## Future Improvements

- Train a model using real images of good and defective products.
- Improve defect detection using deep learning.
- Add model evaluation metrics and test results.
- Support batch image inspection.
- Deploy the dashboard online so it can be accessed through a public link.

## Author

**Niraj Yadav**

B.Tech Computer Science and Engineering — AI & ML

GitHub: [nirajyadav-cse](https://github.com/nirajyadav-cse)

---

*This project is part of my hands-on learning in Python, image processing, and AI-based quality control.*
