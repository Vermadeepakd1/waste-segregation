import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import time
import plotly.express as px
from pathlib import Path

# Page config
st.set_page_config(page_title="♻ Waste Segregation AI", layout="wide", initial_sidebar_state="expanded")

# Load model once, cache for performance
@st.cache_resource
def load_model():
    model_path = Path("best.pt")
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model at {model_path}: {e}")
        return None

model = load_model()
if not model:
    st.stop()

# Header
st.markdown('<h1 style="text-align:center; color:#2E7D32;">♻ AI-Driven Waste Segregation System</h1>', unsafe_allow_html=True)
st.markdown('<h4 style="text-align:center; color:#555;">Real-time Waste Classification using YOLOv8</h4>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar controls
st.sidebar.title("Control Panel")
mode = st.sidebar.radio("Select Mode:", ["Image Detection", "Video Detection", "Live Webcam", "Dashboard", "About"])
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
max_frames = st.sidebar.slider("Max Video Frames", 1, 500, 100)
skip_frames = st.sidebar.slider("Frame Skip Interval", 1, 5, 1)

# Image Detection
if mode == "Image Detection":
    st.header("📷 Image Detection")
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], key="img_upload")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')  # Alpha channel fix
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        with st.spinner("Detecting..."):
            results = model(img_bgr, conf=confidence, verbose=False)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original")
        with col2:
            annotated_img = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Detected")

        detections = []
        if results[0].boxes:
            for idx, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                cls_name = model.names[cls_id]
                detections.append({'ID': idx+1, 'Class': cls_name, 'Confidence': f"{conf_score:.1%}"})

            st.subheader("Detections")
            st.dataframe(pd.DataFrame(detections))
        else:
            st.info("No waste detected in this image.")

# Video Detection
elif mode == "Video Detection":
    st.header("🎬 Video Detection")
    uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov', 'mkv'], key="vid_upload")
    
    if uploaded_video is not None:
        temp_path = f"temp_video_{int(time.time())}.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        frame_idx = 0
        detection_count = 0
        class_counts = {}
        sample_frames = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= max_frames * skip_frames:
                break
            
            frame_idx += 1
            if frame_idx % skip_frames != 0:
                continue

            results = model(frame, conf=confidence, verbose=False)
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                detection_count += 1
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            if len(sample_frames) < 3:
                sample_frames.append(results[0].plot())
            
            progress_bar.progress(min(frame_idx/(max_frames*skip_frames), 1.0))
            status_text.text(f"Processing frame {frame_idx} of {total_frames}")
        
        cap.release()
        status_text.text(f"Processing complete! {frame_idx} frames processed")
        progress_bar.empty()
        
        st.success(f"Total detections: {detection_count}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Detections", detection_count)
        col2.metric("Frames Processed", frame_idx)
        col3.metric("Duration (s)", f"{total_frames/fps:.2f}")
        col4.metric("Classes Found", len(class_counts))
        
        if class_counts:
            st.subheader("Detections per Class")
            df_classes = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"]).sort_values(by="Count", ascending=False)
            fig = px.bar(df_classes, x="Class", y="Count", color="Class", title="Class Distribution")
            st.plotly_chart(fig)

        if sample_frames:
            st.subheader("Sample Frames")
            cols = st.columns(min(3, len(sample_frames)))
            for i, img in enumerate(sample_frames):
                cols[i].image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        Path(temp_path).unlink()

# Live Webcam Detection
elif mode == "Live Webcam":
    st.header("📹 Live Webcam Detection")
    
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False

    if not st.session_state.webcam_running:
        if st.button("Start Webcam Detection", key="start_webcam"):
            st.session_state.webcam_running = True
    else:
        if st.button("Stop Webcam Detection", key="stop_webcam"):
            st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam is not accessible.")
            st.session_state.webcam_running = False
        else:
            frame_placeholder = st.empty()
            fps_placeholder = st.empty()
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to capture frame.")
                    break
                
                results = model(frame, conf=confidence, verbose=False)
                annotated = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                frame_placeholder.image(annotated_rgb)
                fps_placeholder.text(f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}")
                time.sleep(0.01)

            cap.release()
            cv2.destroyAllWindows()
    else:
        st.info("Webcam detection stopped. Click start to begin.")

# Dashboard
elif mode == "Dashboard":
    st.header("📊 Dashboard")
    
    metrics = {
        "Metric": ["mAP@0.5", "Precision", "Recall", "mAP@0.5:0.95"],
        "Score": [0.546, 0.594, 0.494, 0.38]
    }
    df_metrics = pd.DataFrame(metrics)
    
    fig1 = px.bar(df_metrics, x="Metric", y="Score", color="Score", color_continuous_scale="Viridis",
                  title="Overall Model Metrics")
    st.plotly_chart(fig1)
    
    st.markdown("---")
    
    st.markdown("### Class-wise performance")
    class_perf = {
        "Class": ["GLASS", "METAL", "BIODEGRADABLE", "CARDBOARD", "PLASTIC", "PAPER"],
        "mAP50": [0.804, 0.703, 0.632, 0.61, 0.454, 0.076]
    }
    df_class_perf = pd.DataFrame(class_perf)

    fig2 = px.bar(df_class_perf, x="Class", y="mAP50", color="mAP50", color_continuous_scale="RdYlGn",
                  title="Per Class mAP@0.5")
    st.plotly_chart(fig2)

# About
elif mode == "About":
    st.header("ℹ About This Project")
    st.write("""
    This project detects and classifies waste in real-time using a custom-trained YOLOv8 model.

    Trained on Roboflow Garbage Classification dataset.

    Developed by IIITDM Kurnool team for Inter IIIT Hackathon 2025.

    Classes detected:
    - GLASS, METAL, BIODEGRADABLE, CARDBOARD, PLASTIC, PAPER

    Features:
    - Image, video, and live webcam detection
    - Confidence threshold tuning
    - Real-time analytics and dashboard
    """)

st.markdown("---")
st.markdown("<center>© 2025 AI Waste Segregation System | IIITDM Kurnool</center>", unsafe_allow_html=True)
