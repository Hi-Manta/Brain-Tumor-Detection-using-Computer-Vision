# 🧠 Brain Tumor Detection using Computer Vision

This project leverages **YOLOv8** and **Streamlit** to detect brain tumors in MRI images using deep learning and computer vision. It’s designed to help visualize tumor locations and provide informative feedback on the type of tumor found.

---

## 🚀 Live Demo

Try the app live: [Streamlit Cloud Deployment](https://YOUR_STREAMLIT_APP_LINK_HERE)

---

## 📌 Features

- 🔍 Detects tumors from MRI images with bounding boxes
- 🧠 Classifies types: `Glioma`, `Meningioma`, `Pituitary`, and more
- 📸 Annotated image preview with download option
- 💡 Displays medical info about detected tumor types
- ⚙️ Adjustable detection confidence threshold
- 🌐 Deployed using Streamlit Cloud

---

## 📷 Sample Usage

<p align="center">
  <img src="screenshots/example_detection.png" width="700"/>
</p>

---

## 🧠 Tumor Types

| Tumor Type   | Description |
|--------------|-------------|
| **Glioma** | Tumor from glial cells, can be low or high grade |
| **Meningioma** | Arises from the meninges (usually benign) |
| **Pituitary** | Forms in the pituitary gland, affects hormone balance |
| **General Tumor** | Any non-specified tumor in the brain area |

---

## 🛠 Tech Stack

- **Model:** YOLOv8 (Ultralytics)
- **App:** Streamlit
- **Image Handling:** OpenCV, Pillow
- **Language:** Python 3.9+

---

## ✅ Requirements

Install the required libraries using:

```bash
pip install -r requirements.txt
```
---

🧪 Run Locally
```bash
streamlit run app.py
```

## 🧰 Deployment Tips for Streamlit Cloud
If deploying on Streamlit Cloud, make sure your requirements.txt includes:
```bash
streamlit
ultralytics
opencv-python-headless
Pillow
numpy
```
⚠️ Avoid using opencv-python on Streamlit Cloud, as it may cause libGL.so.1 errors.

---

## 👨‍💻 Author
Zahid Al Noor Himanta
📧 Email: z.a.n.himanta@gmail.com
🌐 LinkedIn: https://www.linkedin.com/in/zahid-al-noor-himanta/

