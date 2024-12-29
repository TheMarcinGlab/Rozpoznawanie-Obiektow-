import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import streamlit as st

# Funkcja do przetwarzania klatek za pomocą YOLO (lub dowolnego algorytmu)
def process_frame_with_yolo(frame):
    # Przykład przetwarzania (konwersja na odcienie szarości)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    return processed_frame

# Klasa przetwarzania wideo
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")  # Konwersja klatki do formatu OpenCV
            processed_img = process_frame_with_yolo(img)  # Przetwarzanie YOLO
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        except Exception as e:
            st.error(f"Error in video processing: {e}")
            return frame  # Zwróć oryginalną klatkę w razie błędu

# Interfejs Streamlit
st.title("Streamlit WebRTC - YOLO Integration")
st.sidebar.title("Options")
page = st.sidebar.radio("Choose a page:", ["Home", "YOLO WebRTC", "About"])

if page == "Home":
    st.write("Welcome to the YOLO WebRTC example!")

elif page == "YOLO WebRTC":
    st.write("Real-time object detection using WebRTC and YOLO.")
    
    webrtc_streamer(
        key="yolo-webrtc",
        video_processor_factory=VideoProcessor,
        rtc_configuration={
            "iceServers": [
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": "stun:global.stun.twilio.com:3478?transport=udp"}
            ]
        },
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        mode=WebRtcMode.SENDRECV,
    )

elif page == "About":
    st.write("This is a demo application for integrating YOLO with Streamlit WebRTC.")
