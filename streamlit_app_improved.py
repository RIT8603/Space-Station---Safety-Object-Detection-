"""
Space Station Safety Detection - IMPROVED VERSION
- Better detection for multiple objects
- Handles partial/occluded objects
- Working webcam support
- Adjustable detection parameters
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
    page_title="Space Station Safety Detector (Improved)",
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
        st.warning("Custom model not found. Using pretrained YOLOv8n.")
        return YOLO('yolov8n.pt')

def draw_detections(image, results, conf_threshold):
    """Draw bounding boxes on image with improved visibility"""
    annotated = image.copy()
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        
        for box in boxes:
            conf = float(box.conf[0])
            
            if conf < conf_threshold:
                continue
            
            cls = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Get class info
            class_name = CLASSES[cls] if cls < len(CLASSES) else f"Class_{cls}"
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            
            # Draw box with thickness based on confidence
            thickness = max(2, int(conf * 5))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label background
            label = f"{class_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
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
    st.markdown('<p class="main-header">🛸 Space Station Safety Detector (IMPROVED)</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        <b>Enhanced Detection:</b> Multiple objects • Partial visibility • Better accuracy<br>
        <b>New:</b> Working webcam support!
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Detection Settings")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path",
        value="runs/winning/yolov8x_champion/weights/best.pt",
        help="Path to trained model weights"
    )
    
    # IMPROVED: Lower confidence threshold for detecting more objects
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
        help="Lower = detect more objects (including partial)"
    )
    
    # NEW: IoU threshold control
    iou_threshold = st.sidebar.slider(
        "IoU (Overlap) Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.05,
        help="Lower = detect more overlapping objects"
    )
    
    # NEW: Max detections control
    max_det = st.sidebar.number_input(
        "Max Detections",
        min_value=10,
        max_value=1000,
        value=300,
        step=10,
        help="Maximum objects to detect per image"
    )
    
    # Image size
    img_size = st.sidebar.selectbox(
        "Image Size",
        options=[640, 1280, 1920],
        index=1,
        help="Higher = more accurate but slower"
    )
    
    # NEW: Augmentation toggle
    use_augment = st.sidebar.checkbox(
        "Test-Time Augmentation",
        value=False,
        help="Detects more objects but 4x slower"
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
        options=["Upload Image", "Upload Video", "Webcam (Live)"],
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
            
            # Run detection with IMPROVED parameters
            with st.spinner("Running enhanced detection..."):
                start_time = time.time()
                results = model.predict(
                    image_bgr,
                    imgsz=img_size,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=max_det,
                    agnostic_nms=False,
                    augment=use_augment,
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
            
            metric_cols = st.columns(5)
            metric_cols[0].metric("Total Detections", total)
            metric_cols[1].metric("Inference Time", f"{inference_time:.3f}s")
            metric_cols[2].metric("FPS", f"{1/inference_time:.1f}")
            metric_cols[3].metric("Image Size", f"{img_size}px")
            metric_cols[4].metric("Confidence", f"{conf_threshold:.2f}")
            
            if total > 0:
                st.markdown("### Detected Objects by Class")
                chart_data = {k: v for k, v in stats.items() if v > 0}
                st.bar_chart(chart_data)
                
                # Detailed breakdown
                with st.expander("📋 Detailed Detection List"):
                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        detection_data = []
                        for i, box in enumerate(boxes):
                            conf = float(box.conf[0])
                            if conf >= conf_threshold:
                                cls = int(box.cls[0])
                                class_name = CLASSES[cls] if cls < len(CLASSES) else f"Class_{cls}"
                                xyxy = box.xyxy[0].cpu().numpy()
                                detection_data.append({
                                    "#": i+1,
                                    "Class": class_name,
                                    "Confidence": f"{conf:.2%}",
                                    "Box": f"({int(xyxy[0])}, {int(xyxy[1])}) → ({int(xyxy[2])}, {int(xyxy[3])})"
                                })
                        
                        if detection_data:
                            st.table(detection_data)
    
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
            progress_bar = st.progress(0)
            
            fps_list = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            st.info(f"Processing {total_frames} frames...")
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Run detection with improved parameters
                start_time = time.time()
                results = model.predict(
                    frame,
                    imgsz=img_size,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=max_det,
                    agnostic_nms=False,
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
                    f"**Frame:** {frame_count}/{total_frames} | "
                    f"**Detections:** {total} | **FPS:** {fps:.1f} | "
                    f"**Avg FPS:** {np.mean(fps_list):.1f}"
                )
                
                # Progress
                progress_bar.progress(min(frame_count / total_frames, 1.0))
                
                frame_count += 1
            
            cap.release()
            progress_bar.progress(1.0)
            st.success(f"✅ Video processing complete! Avg FPS: {np.mean(fps_list):.1f}")
    
    elif mode == "Webcam (Live)":
        st.subheader("📹 Live Webcam Detection")
        
        st.info("🎥 Click 'Start Webcam' to begin real-time detection. The webcam will detect all objects including partial/overlapping ones.")
        
        # Webcam controls
        col1, col2, col3 = st.columns(3)
        with col1:
            start_webcam = st.button("🎥 Start Webcam", key="start_btn", type="primary")
        with col2:
            stop_webcam = st.button("🛑 Stop Webcam", key="stop_btn")
        with col3:
            camera_index = st.number_input("Camera Index", 0, 10, 0, help="Usually 0 for built-in, 1 for external")
        
        # Placeholders
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Initialize session state
        if 'webcam_running' not in st.session_state:
            st.session_state.webcam_running = False
        
        if stop_webcam:
            st.session_state.webcam_running = False
            status_placeholder.warning("🛑 Webcam stopped by user")
        
        if start_webcam:
            st.session_state.webcam_running = True
            
            try:
                cap = cv2.VideoCapture(camera_index)
                
                # Set webcam properties for better quality
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                if not cap.isOpened():
                    status_placeholder.error(f"""
                    ❌ Cannot access webcam at index {camera_index}
                    
                    **Troubleshooting:**
                    - Try different camera index (0, 1, 2...)
                    - Close other apps using webcam (Zoom, Teams, Skype)
                    - Check Windows Camera privacy settings
                    - Restart your browser
                    """)
                    st.session_state.webcam_running = False
                else:
                    status_placeholder.success("✅ Webcam connected! Detecting objects...")
                    
                    fps_list = []
                    frame_count = 0
                    max_frames = 10000  # Safety limit
                    
                    while st.session_state.webcam_running and frame_count < max_frames:
                        ret, frame = cap.read()
                        
                        if not ret:
                            status_placeholder.error("❌ Failed to read frame from webcam")
                            break
                        
                        # Run detection with improved parameters
                        start_time = time.time()
                        results = model.predict(
                            frame,
                            imgsz=img_size,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            max_det=max_det,
                            agnostic_nms=False,
                            verbose=False,
                            stream=True  # Stream mode for better performance
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
                            f"**🎯 Detections:** {total} | "
                            f"**⚡ FPS:** {fps:.1f} | "
                            f"**📊 Avg FPS:** {np.mean(fps_list[-30:]):.1f} | "
                            f"**🖼️ Frames:** {frame_count}"
                        )
                        
                        frame_count += 1
                        
                        # Small delay to prevent overwhelming
                        time.sleep(0.01)
                    
                    cap.release()
                    st.session_state.webcam_running = False
                    status_placeholder.success(f"✅ Webcam session ended. Processed {frame_count} frames.")
                    
            except Exception as e:
                status_placeholder.error(f"""
                ❌ Webcam error: {str(e)}
                
                **Common Solutions:**
                - Install: `pip install opencv-python`
                - Restart Streamlit: Press Ctrl+C and run again
                - Check antivirus/firewall blocking camera access
                - Update webcam drivers
                """)
                st.session_state.webcam_running = False

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <b>🚀 Space Station Safety Object Detection (Enhanced)</b><br>
        Improved for: Multiple objects • Partial visibility • Overlapping detection<br>
        <br>
        <i>Powered by YOLOv8x + Optimized Detection Parameters</i>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
