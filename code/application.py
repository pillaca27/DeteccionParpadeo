from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import streamlit as st
import cv2
import os
import av
import io

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras.models import load_model
import numpy as np
from pygame import mixer
import time


st.title("SafeTrip")

st.text(
    "This proyect contain some models with different accuracy from which\nyou can choose to Detect Drowsiness with the camera conected."
)

# mixer.init()
# sound = mixer.Sound("alarm.wav")
# st.image(image_data)


face = cv2.CascadeClassifier("E:/DeteccionParpadeo/files/haarcascade_frontalface_alt.xml")
leye = cv2.CascadeClassifier("E:/DeteccionParpadeo/files/haarcascade_lefteye_2splits.xml")
reye = cv2.CascadeClassifier("E:/DeteccionParpadeo/files/haarcascade_righteye_2splits.xml")

lbl = ["Close", "Open"]
path = os.getcwd()


class VideoProcessor:
    score = 0
    option = st.selectbox(
        "Choose the model to use",
        ("cnnCat2.h5", "cnnCat3.h5", "cnnCat8.h5"),
    )

    def recv(self, frame):
        model = load_model("./models/" + self.option)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        count = 0

        thicc = 2
        rpred = [99]
        lpred = [99]
        frm = frame.to_ndarray(format="bgr24")
        height, width = frm.shape[:2]
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(
            gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25)
        )
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)
        cv2.rectangle(
            frm, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (100, 100, 100), 1)
        for (x, y, w, h) in right_eye:
            r_eye = frm[y : y + h, x : x + w]
            count = count + 1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = model.predict(r_eye)[0]
            # print(round(rpred[0]))

            if round(rpred[0]) == 1:
                lbl = "Open"
            if round(rpred[0]) == 0:
                lbl = "Closed"
            break
        for (x, y, w, h) in left_eye:
            l_eye = frm[y : y + h, x : x + w]
            count = count + 1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            lpred = model.predict(l_eye)[0]
            # print(round(lpred[0]))

            if round(lpred[0]) == 1:
                lbl = "Open"
            if round(lpred[0]) == 0:
                lbl = "Closed"
            break
        if round(rpred[0]) == 1 and round(lpred[0]) == 1:
            self.score = self.score + 1
            cv2.putText(
                frm,
                "Closed",
                (10, height - 20),
                font,
                1,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        # if(rpred[0]==1 or lpred[0]==1):
        else:
            self.score = self.score - 1
            cv2.putText(
                frm,
                "Open",
                (10, height - 20),
                font,
                1,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if self.score < 0:
            self.score = 0
        cv2.putText(
            frm,
            "Score:" + str(self.score),
            (100, height - 20),
            font,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if self.score > 55:
            # person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path, "image.jpg"), frm)
            #  try:
            # sound.play()

            # except:  # isplaying = False
            #    pass

            if thicc < 3:
                thicc = thicc + 2
            else:
                thicc = thicc - 2
                if thicc < 2:
                    thicc = 2
            cv2.rectangle(frm, (0, 0), (width, height), (0, 0, 255), thicc)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


webrtc_streamer(
    key="key",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
)
st.subheader("Created by Carlos Garcia Lezcano")
