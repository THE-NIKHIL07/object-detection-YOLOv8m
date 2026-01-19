import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import json
import tempfile
import os
import time

st.set_page_config(page_title="Object Detection and Tracking", layout="wide")

st.markdown("""
<style>
body,.stApp,.main .block-container{background:white;color:black}
.sidebar .sidebar-content{background:black;color:white}
.stButton>button,.stDownloadButton>button{
    background:#333;color:white;border:1px solid white;width:100%
}
h1{color:black;}
.footer{
    position:fixed;bottom:0;width:100%;
    background:white;color:black;
    text-align:center;font-weight:bold;
    padding:12px;border-top:2px solid black
}
.main .block-container{padding-bottom:120px}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Object Detection and Tracking</h1>", unsafe_allow_html=True)

source = st.sidebar.selectbox("Input Source", ("Photo", "Video", "Webcam"))

confidence = st.sidebar.select_slider(
    "Confidence Threshold",
    options=[round(i * 0.05, 2) for i in range(1, 21)],
    value=0.5
)

TRACK_JSON = "track_records.json"

if "track_data" not in st.session_state:
    st.session_state.track_data = []

if "stop" not in st.session_state:
    st.session_state.stop = True

@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

colors = {}
def get_color(cls_name):
    if cls_name not in colors:
        colors[cls_name] = tuple(np.random.randint(0,255,3).tolist())
    return colors[cls_name]

def process_frame(frame, frame_num):
    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.track(
        rgb,
        persist=True,
        conf=confidence,
        tracker="botsort.yaml",
        verbose=False
    )

    objects = []

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, obj_id, cls in zip(boxes, ids, clss):
            label = model.names[cls]
            objects.append({"id": int(obj_id), "class": label})
            color = get_color(label)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Big class name
            cv2.putText(
                frame,
                f"{label}",
                (box[0], box[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

            # Smaller ID below class
            cv2.putText(
                frame,
                f"ID:{obj_id}",
                (box[0], box[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,0,0),
                1
            )

    if objects:
        st.session_state.track_data.append({"frame": frame_num, "objects": objects})
    return frame

stframe = st.empty()

if source == "Photo":
    file = st.file_uploader("Upload Image", ["jpg", "jpeg", "png"])
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        st.session_state.track_data = []
        out = process_frame(img, 1)
        stframe.image(out, channels="BGR", use_column_width=True)
        with open(TRACK_JSON, "w") as f:
            json.dump(st.session_state.track_data, f, indent=4)

elif source == "Video":
    file = st.file_uploader("Upload Video", ["mp4", "avi", "mov"])
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)
        st.session_state.track_data = []
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            out = process_frame(frame, frame_num)
            stframe.image(out, channels="BGR", use_column_width=True)
            time.sleep(0.005)
        cap.release()
        with open(TRACK_JSON, "w") as f:
            json.dump(st.session_state.track_data, f, indent=4)
        st.success("Video finished")

elif source == "Webcam":
    st.sidebar.markdown("### Webcam Controls")
    cam_type = st.sidebar.radio("Camera Type", ("Laptop Webcam", "Android Phone (IP Camera)"))
    ip_url = None
    if cam_type == "Android Phone (IP Camera)":
        ip_url = st.sidebar.text_input("IP Camera URL", placeholder="http://192.168.1.5:8080/video")
    start = st.sidebar.button("▶ Start Webcam")
    stop = st.sidebar.button("⏹ Stop Webcam")
    if start:
        st.session_state.stop = False
        st.session_state.track_data = []
    if stop:
        st.session_state.stop = True
        with open(TRACK_JSON, "w") as f:
            json.dump(st.session_state.track_data, f, indent=4)
    if not st.session_state.stop:
        if cam_type == "Laptop Webcam":
            cap = cv2.VideoCapture(0)
        else:
            if not ip_url:
                st.warning("⚠ Please enter Android IP camera URL")
                st.session_state.stop = True
                cap = None
            else:
                cap = cv2.VideoCapture(ip_url)
        if cap is None or not cap.isOpened():
            st.error("❌ Unable to access camera. Check webcam or IP camera URL.")
            st.session_state.stop = True
        else:
            frame_num = 0
            while cap.isOpened() and not st.session_state.stop:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠ Camera stream interrupted")
                    break
                frame_num += 1
                out = process_frame(frame, frame_num)
                stframe.image(out, channels="BGR", use_column_width=True)
                time.sleep(0.005)
            cap.release()

if os.path.exists(TRACK_JSON) and len(st.session_state.track_data) > 0:
    with open(TRACK_JSON, "rb") as f:
        st.sidebar.download_button("Download Track Records", f, "track_records.json", "application/json")

st.markdown("<div class='footer'>MADE BY THE-NIKHIL07 © 2026</div>", unsafe_allow_html=True)
