"""
Space Station Safety Detection - Streamlit Web App
Real-time detection with webcam, image, and video upload support
Deployable on Streamlit Cloud (free tier)
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
from ultralytics import YOLO
import torch
import time

# Page config
st.set_page_config(
    page_title="Space Station Safety Detector",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .detection-box {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Classes and colors
CLASSES = ['OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm', 
           'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher']

CLASS_COLORS = {
    0: (255, 0, 0),      # OxygenTank - Red
    1: (0, 255, 0),      # NitrogenTank - Green
    2: (255, 255, 0),    # FirstAidBox - Yellow
    3: (255, 0, 255),    # FireAlarm - Magenta
    4: (0, 255, 255),    # SafetySwitchPanel - Cyan
    5: (255, 128, 0),    # EmergencyPhone - Orange
    6: (128, 0, 255)     # FireExtinguisher - Purple
}

@st.cache_resource
def load_model(model_path='runs/winning/yolov8x_champion/weights/best.pt'):
    """Load and cache the YOLO model"""
    try:
        model = YOLO(model_path)
        return model
    except:
        # Fallback to pretrained model
        st.warning("Custom model not found. Using pretrained YOLOv8n.")
        return YOLO('yolov8n.pt')

def draw_detections(image, results, conf_threshold):
    """Draw bounding boxes on image"""
    annotated = image.copy()
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        
        for box in boxes:
            # Extract box info
            conf = float(box.conf[0])
            
            if conf < conf_threshold:
                continue
            
            cls = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Get class info
            class_name = CLASSES[cls] if cls < len(CLASSES) else f"Class_{cls}"
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Draw label background
            label = f"{class_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated

def get_detection_stats(results, conf_threshold):
    """Get detection statistics"""
    stats = {cls: 0 for cls in CLASSES}
    total = 0
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        
        for box in boxes:
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                cls = int(box.cls[0])
                if cls < len(CLASSES):
                    stats[CLASSES[cls]] += 1
                    total += 1
    
    return stats, total

def main():
    # Header
    st.markdown('<p class="main-header">🛸 Space Station Safety Object Detector</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        <b>Duality AI Challenge #2 - Championship Solution</b><br>
        Real-time detection of safety equipment using YOLOv8x
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path",
        value="runs/winning/yolov8x_champion/weights/best.pt",
        help="Path to trained model weights"
    )
    
    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    # Image size
    img_size = st.sidebar.selectbox(
        "Image Size",
        options=[640, 1280, 1920],
        index=1,
        help="Higher = more accurate but slower"
    )
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_path)
    
    # Check CUDA
    device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.success(f"✅ Model loaded on {device_info}")
    
    # Mode selection
    st.sidebar.markdown("---")
    st.sidebar.title("📷 Input Mode")
    
    mode = st.sidebar.radio(
        "Select input source:",
        options=["Upload Image", "Upload Video", "Webcam (Live)", "Sample Images"],
        index=0
    )
    
    # Class legend
    st.sidebar.markdown("---")
    st.sidebar.title("🎨 Class Legend")
    
    for idx, class_name in enumerate(CLASSES):
        color = CLASS_COLORS[idx]
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        st.sidebar.markdown(
            f"<div style='background-color:{color_hex}; padding:5px; margin:2px; "
            f"border-radius:3px; color:white;'>{class_name}</div>",
            unsafe_allow_html=True
        )
    
    # Main content
    if mode == "Upload Image":
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image for detection"
        )
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert RGB to BGR
            if len(image_np.shape) == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_np
            
            # Display original
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Image")
                st.image(image, use_container_width=True)
            
            # Run detection
            with st.spinner("Running detection..."):
                start_time = time.time()
                results = model.predict(
                    image_bgr,
                    imgsz=img_size,
                    conf=conf_threshold,
                    verbose=False
                )
                inference_time = time.time() - start_time
            
            # Draw results
            annotated = draw_detections(image_bgr, results, conf_threshold)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.markdown("### Detection Result")
                st.image(annotated_rgb, use_container_width=True)
            
            # Statistics
            stats, total = get_detection_stats(results, conf_threshold)
            
            st.markdown("---")
            st.markdown("### 📊 Detection Statistics")
            
            metric_cols = st.columns(4)
            metric_cols[0].metric("Total Detections", total)
            metric_cols[1].metric("Inference Time", f"{inference_time:.3f}s")
            metric_cols[2].metric("FPS", f"{1/inference_time:.1f}")
            metric_cols[3].metric("Image Size", f"{img_size}px")
            
            if total > 0:
                st.markdown("### Detected Objects")
                chart_data = {k: v for k, v in stats.items() if v > 0}
                st.bar_chart(chart_data)
    
    elif mode == "Upload Video":
        st.subheader("🎥 Upload Video")
        
        uploaded_file = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video for detection"
        )
        
        if uploaded_file is not None:
            # Save to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            
            # Open video
            cap = cv2.VideoCapture(tfile.name)
            
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            
            fps_list = []
            
            st.info("Processing video... (Press 'Stop' in browser to interrupt)")
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Run detection
                start_time = time.time()
                results = model.predict(
                    frame,
                    imgsz=img_size,
                    conf=conf_threshold,
                    verbose=False
                )
                inference_time = time.time() - start_time
                fps = 1 / inference_time if inference_time > 0 else 0
                fps_list.append(fps)
                
                # Draw results
                annotated = draw_detections(frame, results, conf_threshold)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                # Display
                frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
                
                # Stats
                stats, total = get_detection_stats(results, conf_threshold)
                stats_placeholder.markdown(
                    f"**Detections:** {total} | **FPS:** {fps:.1f} | "
                    f"**Avg FPS:** {np.mean(fps_list):.1f}"
                )
            
            cap.release()
            st.success(f"✅ Video processing complete! Avg FPS: {np.mean(fps_list):.1f}")
    
    elif mode == "Webcam (Live)":
        st.subheader("📹 Live Webcam Detection")
        
        st.warning("⚠️ Webcam support requires local deployment. For cloud deployment, use image/video upload.")
        
        st.markdown("""
        ### Instructions for Local Deployment:
        
        1. Install streamlit-webrtc:
           ```bash
           pip install streamlit-webrtc
           ```
        
        2. Run locally:
           ```bash
           streamlit run streamlit_app.py
           ```
        
        3. Allow camera access in browser
        """)
    
    else:  # Sample Images
        st.subheader("🖼️ Sample Images")
        
        st.info("Load sample images from dataset for quick testing")
        
        # This would load from your test set
        st.markdown("*Feature coming soon - connect to your test dataset folder*")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <b>Space Station Safety Object Detection</b><br>
        Championship Solution | Duality AI Challenge #2 | 2025<br>
        <br>
        <i>Powered by YOLOv8x + Falcon Synthetic Data</i>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
