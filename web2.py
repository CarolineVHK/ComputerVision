import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av  # PyAV: Pythonic binding to FFmpeg
import cv2  # OpenCV
import cvlib as cv  # Object detection
from cvlib.object_detection import draw_bbox  # Drawing bounding boxes


# Streamlit app
th1 = st.slider("Threshold1", 0, 1000, 100)
th2 = st.slider("Threshold2", 0, 1000, 200)


class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.th1 = th1
        self.th2 = th2

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the frame to a numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Object detection
        bbox, label, conf = cv.detect_common_objects(img, confidence=0.2, model='yolov3', enable_gpu=False)
        img = draw_bbox(img, bbox, label, conf)

        # Canny edge detection
        edges = cv2.Canny(img, self.th1, self.th2)

        # Convert grayscale edges to BGR for display
        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Initialize the Streamlit WebRTC component
webrtc_streamer(
    key="sample",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor
)