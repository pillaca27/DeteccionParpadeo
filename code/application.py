from flask import Flask, render_template
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from DeteccionParpadeo.drowsinessDetection import VideoProcessor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    class MyVideoProcessor(VideoProcessor):
        def __init__(self):
            super().__init__()
            self.option = st.selectbox(
                "Choose the model to use",
                ("cnnCat2.h5", "cnnCat3.h5", "cnnCat8.h5"),
            )
    
    return webrtc_streamer(
        key="video-feed",
        video_processor_factory=MyVideoProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
    )

if __name__ == '__main__':
    app.run()
