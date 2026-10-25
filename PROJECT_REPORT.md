# Project Report: Image Recognition for Quality Control

## 1. Introduction
Manufacturing companies need consistent product inspection. Manual inspection can be time-consuming and may vary between inspectors. Computer vision can help screen images and support quality-control workflows.

## 2. Problem statement
The goal is to build a prototype that accepts product images, provides basic visual-capture checks, and demonstrates how a labelled image classifier could be trained to distinguish good products from defective products.

## 3. Objectives
1. Build a Python dashboard for image upload and inspection.
2. Calculate image indicators such as brightness, sharpness, and edge density.
3. Produce a downloadable inspection report.
4. Provide an optional deep-learning training pipeline using MobileNetV2 transfer learning.

## 4. Technologies
- Python
- Streamlit
- Pillow and NumPy for image handling
- Pandas for reports
- TensorFlow and MobileNetV2 for optional model training

## 5. Methodology
1. Upload an image and correct its EXIF orientation.
2. Convert the image to RGB and grayscale representations.
3. Calculate mean brightness, an edge-based sharpness indicator, and edge density.
4. Compare brightness and sharpness against user-adjustable thresholds.
5. Display the results and allow CSV export.
6. For a dataset-based experiment, place labelled images in Good and Defective folders and train the optional transfer-learning model.

## 6. Expected outcome
The dashboard demonstrates a simple inspection workflow and creates a repeatable CSV report. With a sufficient, representative labelled dataset, the optional training script can be used to begin an image-classification experiment.

## 7. Evaluation plan
For a trained classifier, reserve separate validation and test images. Report accuracy, precision, recall, F1-score, and a confusion matrix. In quality control, false negatives (defective items predicted as good) are especially important. Evaluate across lighting conditions, product variants, and camera setups.

## 8. Limitations
The included default dashboard evaluates image-capture indicators, not actual product defects. A deep-learning model cannot be meaningfully trained without labelled examples. The training script is optional and must be tested on unseen images. This prototype is for learning and demonstration, not safety-critical production use.

## 9. Future improvements
- Collect a larger, balanced dataset of good and defective products.
- Add model predictions to the dashboard after training and validating the classifier.
- Include explainability visualizations and confidence scores.
- Connect results to a database or manufacturing execution system.
- Add automated alerts and a review queue for uncertain predictions.

## 10. Conclusion
This project provides a practical starting point for learning image-based quality-control workflows with Python. It combines an interactive dashboard, downloadable reports, and an optional transfer-learning path while clearly identifying the need for real labelled data and careful validation.
