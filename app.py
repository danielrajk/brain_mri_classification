import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
from fpdf import FPDF

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="NeuroScan AI – Brain MRI XAI",
    page_icon="🧠",
    layout="wide"
)

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 224
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CONF_THRESHOLD = 60

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_mri_v2.keras")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_trained_model():
    model = load_model(MODEL_PATH, compile=False)
    dummy = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
    model(dummy)
    return model

model = load_trained_model()
base_model = model.get_layer("efficientnetb0")

# ===============================
# PREPROCESS
# ===============================
def preprocess(image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

# ===============================
# GRAD-CAM
# ===============================
def gradcam(img_array):
    last_conv = base_model.get_layer("top_conv")
    conv_model = tf.keras.Model(base_model.input, last_conv.output)

    with tf.GradientTape() as tape:
        conv_out = conv_model(img_array)
        tape.watch(conv_out)

        x = conv_out
        for layer in model.layers[1:]:
            x = layer(x)

        preds = x
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)[0]
    conv_out = conv_out[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_out.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_out[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-10)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original = np.uint8(img_array[0] * 255)

    return cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

# ===============================
# SALIENCY MAP
# ===============================
def saliency_map(img_array):
    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        loss = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(loss, img_tensor)[0]
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)

    saliency = saliency.numpy()
    saliency = saliency / (saliency.max() + 1e-10)
    saliency = cv2.resize(saliency, (IMG_SIZE, IMG_SIZE))

    return np.uint8(255 * saliency)

# ===============================
# PDF REPORT
# ===============================
def generate_pdf(pred_class, confidence, probs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "NeuroScan AI - Brain MRI Report", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, f"Predicted Class: {pred_class}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.2f} %", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, "Class Probabilities:", ln=True)
    for cls, p in zip(CLASS_NAMES, probs):
        pdf.cell(0, 8, f"{cls}: {p:.2f} %", ln=True)

    return pdf.output(dest="S").encode("latin-1")

# ===============================
# HEADER
# ===============================
st.markdown("""
<h1 style="text-align:center;">🧠 NeuroScan AI</h1>
<h4 style="text-align:center; color:gray;">
Brain MRI Classification with Explainable AI
</h4>
<hr>
""", unsafe_allow_html=True)

# ===============================
# FILE UPLOAD
# ===============================
uploaded = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ===============================
# MAIN PIPELINE
# ===============================
if uploaded:
    image = Image.open(uploaded)
    input_tensor = preprocess(image)

    preds = model.predict(input_tensor)
    probs = preds[0] * 100
    idx = int(np.argmax(probs))
    confidence = probs[idx]

    tab1, tab2, tab3, tab4 = st.tabs([
        "Diagnosis",
        "Explainability",
        "Report",
        "Comparison"
    ])

    # ---------- TAB 1: DIAGNOSIS ----------
    with tab1:
        c1, c2 = st.columns(2)

        c1.subheader("Input MRI")
        c1.image(image, use_container_width=True)

        c2.subheader("Model Result")
        c2.metric("Predicted Class", CLASS_NAMES[idx])
        c2.metric("Confidence", f"{confidence:.2f}%")

        if confidence < CONF_THRESHOLD:
            c2.warning("Low confidence prediction.")

        if CLASS_NAMES[idx] == "No Tumor":
            c2.success("No tumor detected.")
        else:
            c2.error(f"Tumor detected: {CLASS_NAMES[idx]}")

        st.markdown("### Class Probabilities")
        for i, cls in enumerate(CLASS_NAMES):
            st.progress(int(probs[i]), text=f"{cls}: {probs[i]:.2f}%")

    # ---------- TAB 2: EXPLAINABILITY ----------
    with tab2:
        x1, x2 = st.columns(2)

        x1.subheader("Grad-CAM")
        x1.image(gradcam(input_tensor), use_container_width=True)
        x1.caption("Region-based explanation highlighting influential areas.")

        x2.subheader("Saliency Map")
        x2.image(saliency_map(input_tensor), use_container_width=True)
        x2.caption("Pixel-level sensitivity visualization.")

    # ---------- TAB 3: REPORT ----------
    with tab3:
        st.subheader("Download Diagnostic Report")
        pdf = generate_pdf(CLASS_NAMES[idx], confidence, probs)

        st.download_button(
            "Download PDF Report",
            pdf,
            file_name="NeuroScan_Report.pdf",
            mime="application/pdf"
        )

    # ---------- TAB 4: COMPARISON ----------
    with tab4:
        second = st.file_uploader(
            "Upload Second MRI",
            type=["jpg", "jpeg", "png"],
            key="second"
        )

        if second:
            img2 = Image.open(second)
            tensor2 = preprocess(img2)
            p2 = model.predict(tensor2)[0] * 100
            idx2 = int(np.argmax(p2))

            c1, c2 = st.columns(2)

            c1.image(image, caption="MRI 1", use_container_width=True)
            c1.metric("Prediction", CLASS_NAMES[idx])
            c1.metric("Confidence", f"{confidence:.2f}%")
            for i, cls in enumerate(CLASS_NAMES):
                c1.progress(int(probs[i]), text=f"{cls}: {probs[i]:.2f}%")

            c2.image(img2, caption="MRI 2", use_container_width=True)
            c2.metric("Prediction", CLASS_NAMES[idx2])
            c2.metric("Confidence", f"{p2[idx2]:.2f}%")
            for i, cls in enumerate(CLASS_NAMES):
                c2.progress(int(p2[i]), text=f"{cls}: {p2[i]:.2f}%")

# ===============================
# FOOTER
# ===============================
st.markdown("""
<hr>
<p style="text-align:center; color:gray;">
For research and educational use only.
</p>
""", unsafe_allow_html=True)
