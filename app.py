import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av #is a package from PyAV is pythonic binding FFmpeg = video/audio
import cv2

th1 = st.slider("Threshold1", 0, 1000, 100)
th2 = st.slider("Threshold2", 0, 1000, 200)

def callback(frame:av.VideoFrame) -> av.VideoFrame:
    #convert into a numpy array
    img = frame.to_ndarray(format="bgr24") # 24 bits channel colors
    #img is now an np arry and can be implemented as 
    img = cv2.Canny(img, th1, th2) # 2 threshold parameters
    #convertColor
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)



    return av.VideoFrame.from_ndarray(frame.from_ndarray(img, format="bgr24"))

webrtc_streamer(key="sample",
                video_frame_callback=callback)
