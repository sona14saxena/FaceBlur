import streamlit as st
import cv2
import os
import tempfile
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

st.set_page_config(page_title="FaceBlur | Privacy-by-Design", page_icon="🛡️", layout="centered")

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f0f1a; }
    h1 { color: #a78bfa !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 8px 20px;
        color: #cdd6f4;
    }
    .stTabs [aria-selected="true"] {
        background-color: #7c3aed !important;
        color: white !important;
    }
    .stDownloadButton > button {
        background-color: #7c3aed;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stDownloadButton > button:hover { background-color: #6d28d9; }
    .info-box {
        background: #1e1e2e;
        border-left: 4px solid #7c3aed;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 16px;
        color: #cdd6f4;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ FaceBlur")
st.caption("Real-Time Face Anonymization · Privacy-by-Design")

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_detector():
    model_path = 'face.tflite'
    if not os.path.exists(model_path):
        with st.spinner("Downloading AI model for the first time..."):
            urllib.request.urlretrieve(
                'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
                model_path
            )
    opts = vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=model_path))
    return vision.FaceDetector.create_from_options(opts)

detector = load_detector()

# ─── Core Blur Function ────────────────────────────────────────────────────────
def blur_faces(img_bgr):
    """Takes a BGR image (numpy array), returns it with all detected faces blurred."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    h, w = img_bgr.shape[:2]
    faces_found = len(result.detections)
    for det in result.detections:
        b = det.bounding_box
        x1 = max(0, b.origin_x)
        y1 = max(0, b.origin_y)
        x2 = min(w, b.origin_x + b.width)
        y2 = min(h, b.origin_y + b.height)
        if x2 > x1 and y2 > y1:
            img_bgr[y1:y2, x1:x2] = cv2.GaussianBlur(img_bgr[y1:y2, x1:x2], (99, 99), 30)
    return img_bgr, faces_found

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📷 Webcam Capture", "📁 Upload File"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: WEBCAM
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="info-box">📌 Your webcam will open directly in the browser. Take a snapshot — FaceBlur will automatically detect and blur all faces. Then download your privacy-protected image.</div>', unsafe_allow_html=True)

    img_file = st.camera_input("Click the camera button below to take a snapshot")

    if img_file is not None:
        # Decode the captured image
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Detecting and blurring faces..."):
            blurred, count = blur_faces(img_bgr.copy())

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
        with col2:
            st.image(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB), caption=f"Blurred ({count} face{'s' if count != 1 else ''} found)", use_column_width=True)

        _, buffer = cv2.imencode(".jpg", blurred)
        st.download_button(
            label="⬇️ Download Privacy-Protected Image",
            data=buffer.tobytes(),
            file_name="faceblur_output.jpg",
            mime="image/jpeg"
        )

# ══════════════════════════════════════════════════════════════════════
# TAB 2: UPLOAD
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="info-box">📌 Upload a photo or video from your device. FaceBlur will process every frame and give you a blurred file ready to download and share safely.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Select an Image or Video", 
        type=['jpg', 'jpeg', 'png', 'mp4', 'mov'],
        label_visibility="collapsed"
    )

    if uploaded is not None:
        ext = uploaded.name.lower().split(".")[-1]

        # ── Image Processing ──────────────────────────────────────────
        if ext in ('jpg', 'jpeg', 'png'):
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Detecting and blurring faces..."):
                blurred, count = blur_faces(img_bgr.copy())

            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
            with col2:
                st.image(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB), caption=f"Blurred ({count} face{'s' if count != 1 else ''} found)", use_column_width=True)

            _, buffer = cv2.imencode(".jpg", blurred)
            st.download_button(
                label="⬇️ Download Blurred Image",
                data=buffer.tobytes(),
                file_name="faceblur_output.jpg",
                mime="image/jpeg"
            )

        # ── Video Processing ──────────────────────────────────────────
        elif ext in ('mp4', 'mov'):
            st.info("Processing your video. This may take a few minutes depending on its length...")

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
            tfile.write(uploaded.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, int(fps), (width, height))

            progress = st.progress(0, text="Processing frames...")
            frame_n = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                blurred_frame, _ = blur_faces(frame)
                writer.write(blurred_frame)
                frame_n += 1
                if total > 0:
                    pct = min(frame_n / total, 1.0)
                    progress.progress(pct, text=f"Processing frame {frame_n} of {total}...")

            cap.release()
            writer.release()
            progress.progress(1.0, text="Done!")
            st.success(f"✅ Done! Processed {frame_n} frames.")

            with open(out_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Blurred Video",
                    data=f,
                    file_name="faceblur_output.mp4",
                    mime="video/mp4"
                )
