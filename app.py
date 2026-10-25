from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageStat, ImageFilter

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.keras"

st.set_page_config(page_title="AI Quality Control", page_icon="🏭", layout="wide")
st.title("🏭 AI Image Recognition for Quality Control")
st.write("Upload a product image to inspect its visual quality. This prototype uses image-quality indicators when no trained model is available.")
st.info("Demo note: brightness, blur, and edge density are screening indicators—not proof of a manufacturing defect. Train and validate a model on your own labelled product images before using it for real quality decisions.")

with st.sidebar:
    st.header("Inspection settings")
    blur_threshold = st.slider("Blur threshold (lower = sharper)", 0.0, 500.0, 80.0, 5.0)
    min_brightness = st.slider("Minimum brightness", 0, 255, 35)
    max_brightness = st.slider("Maximum brightness", 0, 255, 235)
    st.caption("Tune these values for your lighting and camera setup.")

uploaded = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg", "webp", "bmp"])

def inspect_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    gray = image.convert("L")
    gray_arr = np.asarray(gray, dtype=np.float32)
    brightness = float(gray_arr.mean())
    # Variance of a Laplacian-like filter response is a simple blur indicator.
    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    sharpness = float(edges.var())
    edge_density = float((edges > 35).mean() * 100)
    issues = []
    if sharpness < blur_threshold:
        issues.append("Image may be blurry")
    if brightness < min_brightness:
        issues.append("Image may be too dark")
    elif brightness > max_brightness:
        issues.append("Image may be overexposed")
    status = "Review required" if issues else "No obvious image-quality issue detected"
    return image, brightness, sharpness, edge_density, issues, status

if uploaded:
    try:
        image = Image.open(uploaded)
        processed, brightness, sharpness, edge_density, issues, status = inspect_image(image)
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Uploaded product")
            st.image(processed, use_container_width=True)
            st.caption(f"Image size: {processed.width} × {processed.height} pixels")
        with right:
            st.subheader("Inspection results")
            c1, c2 = st.columns(2)
            c1.metric("Mean brightness", f"{brightness:.1f}/255")
            c2.metric("Sharpness indicator", f"{sharpness:.1f}")
            c3, c4 = st.columns(2)
            c3.metric("Edge density", f"{edge_density:.1f}%")
            c4.metric("File format", image.format or "Unknown")
            if issues:
                st.warning(status)
                for issue in issues:
                    st.write(f"- {issue}")
            else:
                st.success(status)
            st.write("**Interpretation:** these measurements check the image capture quality. They do not reliably identify scratches, cracks, missing parts, or other product defects without a labelled dataset and a trained model.")

        report = pd.DataFrame([{
            "filename": uploaded.name,
            "width_px": processed.width,
            "height_px": processed.height,
            "mean_brightness": round(brightness, 2),
            "sharpness_indicator": round(sharpness, 2),
            "edge_density_percent": round(edge_density, 2),
            "screening_result": status,
            "notes": "; ".join(issues) if issues else "No image capture warning"
        }])
        st.subheader("Inspection report")
        st.dataframe(report, use_container_width=True, hide_index=True)
        st.download_button("Download report as CSV", report.to_csv(index=False).encode("utf-8"),
                           file_name="quality_inspection_report.csv", mime="text/csv")
    except Exception as exc:
        st.error(f"Could not read this image: {exc}")
else:
    st.markdown("### How to use")
    st.markdown("1. Upload a clear photo of a product.\\n2. Review the image-quality indicators.\\n3. Download the inspection report as CSV.\\n4. For real defect recognition, train the optional model with labelled Good and Defective images.")
    st.subheader("Project workflow")
    st.write("Image upload → preprocessing → image-quality checks → inspection summary → downloadable report")

with st.expander("Optional deep-learning model training"):
    st.write("The repository includes `train_model.py` for transfer learning with MobileNetV2. Add labelled images to `data/train/Good`, `data/train/Defective`, and matching validation folders, install `requirements-training.txt`, then run `python train_model.py`. The trained model is a starting point and must be validated for your actual products.")
