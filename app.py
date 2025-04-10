import streamlit as st
from PIL import Image
import cv2
import os
import numpy as np
from ultralytics import YOLO
import tempfile
from datetime import datetime

# Tumor info articles (detailed descriptions)
TUMOR_INFO = {
    "glioma": "**Glioma** is a common type of tumor that originates from glial cells in the brain or spine. Gliomas can be low-grade (slow-growing) or high-grade (aggressive). Symptoms depend on the tumor's location and may include headaches, seizures, or changes in personality. Treatment often involves surgery, radiation therapy, and chemotherapy.",
    "meningioma": "**Meningioma** is a tumor that arises from the meninges, the membranes that cover the brain and spinal cord. Most meningiomas are benign, but their location in the brain can still cause serious health problems. Common symptoms include headaches, vision problems, or seizures. Treatment may involve surgical removal or observation in non-symptomatic cases.",
    "pituitary": "**Pituitary tumors** form in the pituitary gland, which controls several hormone-producing glands in the body. These tumors may cause hormonal imbalances affecting growth, metabolism, and reproductive functions. Treatment may include medication, hormone therapy, or surgery.",
    "tumor": "**Brain Tumor** is a general term for abnormal growths of cells in the brain. Tumors can be benign (non-cancerous) or malignant (cancerous). They can affect brain function depending on their size and location. Common treatments include surgery, radiation, and chemotherapy."
}

st.set_page_config(page_title="Brain Tumor Detection | MRI Analyzer", layout="wide", page_icon="🧠")

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model_path = "best.pt"
model = load_model(model_path)

# ---------------------------- CSS Styling ----------------------------
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .title {
            font-size: 3rem;
            font-weight: bold;
            color: #00adb5;
        }
        .subtext {
            color: #aaa;
        }
        .footer {
            text-align: center;
            color: #888;
            margin-top: 3rem;
            font-size: 0.9rem;
        }
        .stButton>button {
            background-color: #00adb5;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------- App Title ----------------------------

st.markdown("<h1 class='title'>🧠 Brain Tumor Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Automated MRI analysis using YOLO | A real-world AI Health Project</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.header("🛠 Settings")
    threshold = st.slider("🎯 Detection Confidence", 0.1, 1.0, 0.25, 0.01)
    st.markdown("Upload high-resolution **MRI images** to detect brain tumors with confidence level filtering.")
    st.markdown("⚠️ Only `.jpg`, `.jpeg`, `.png` formats are supported.")
    st.markdown("---")
    st.caption("Built by Himanta| Email: z.a.n.himanta@gmail.com")

# ---------------------------- Functions ----------------------------
def detect_tumor(image, model, threshold):
    image_np = np.array(image.convert("RGB"))
    results = model(image_np, conf=threshold)
    annotated = image_np.copy()
    labels = []

    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            class_id = int(cls)
            class_name = model.names[class_id]
            confidence = round(float(conf) * 100, 2)
            label = f"{class_name} ({confidence}%)"
            labels.append(label)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return annotated, labels

def save_image(image_array):
    _, output = tempfile.mkstemp(suffix=".jpg")
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output, image_bgr)
    return output


# Helper to annotate image and extract prediction data
def annotate_and_extract(image_np, results):
    annotated_img = image_np.copy()
    found_classes = set()

    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            class_id = int(cls)
            class_name = model.names[class_id].lower()
            confidence = round(float(conf) * 100, 2)

            label = f"{class_name} ({confidence}%)"
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            found_classes.add(class_name)

    return annotated_img, list(found_classes)

# ---------------------------- Upload Section ----------------------------

uploaded_files = st.file_uploader("Upload MRI Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("Results")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)

        results = model(img_np, conf=threshold)
        annotated_img, found_classes = annotate_and_extract(img_np, results)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Original MRI", use_container_width=True)
        with col2:
            st.image(annotated_img, caption="Detected Tumor(s)", use_container_width=True)

        if found_classes:
            st.markdown("### Tumor Information")
            for tumor_type in found_classes:
                if tumor_type in TUMOR_INFO:
                    st.markdown(TUMOR_INFO[tumor_type])
                else:
                    st.markdown(f"**{tumor_type.title()}**: Additional information will be added soon.")

        # Download option
        annotated_pil = Image.fromarray(annotated_img)
        download_btn = st.download_button(
            label="📥 Download Annotated Image",
            data=cv2.imencode(".jpg", annotated_img)[1].tobytes(),
            file_name=f"tumor_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            mime="image/jpeg"
        )
else:
    st.info("Upload one or more MRI images to begin detection.")


# ---------------------------- About Section ----------------------------
with st.expander("ℹ️ About this Project", expanded=False):
    st.markdown("""
    **Brain Tumor Detection** using deep learning and computer vision is a crucial step towards AI-driven diagnostics.
    
    - **Model**: YOLO, trained on a labeled dataset of brain MRI scans.
    - **Input**: MRI images (axial slices).
    - **Output**: Detected tumor regions with bounding boxes and confidence.
    - **Use Case**: Hospitals, radiology labs, clinical research, and automated diagnostics.
    """)

    
